"""End-to-end deterministic + single-LLM research pipeline (Phase 0 & Phase 1).

Orchestrates Layer 1 (Adapters), Layer 2 (Quant/Detectors/Structures/Part A),
Layer 3 (Single-LLM Synthesis & Deterministic Post-LLM Verifier), and
Layer 4 (Alert-Only Output & Immutable Audit Log).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Sequence

from nifty_analyst.adapters.calendar_and_contracts import (
    ContractMasterError,
    load_nse_holiday_calendar,
    parse_contract_master,
)
from nifty_analyst.adapters.eod_vix_events import (
    load_eod_volatility_history,
    load_events_calendar,
)
from nifty_analyst.adapters.kotak_neo import (
    FixtureKotakNeoAdapter,
    KotakAdapterError,
    KotakNeoReadOnlyPort,
    ProvisionalSerialRateLimiter,
    UnverifiedLiveKotakNeoAdapter,
    parse_and_validate_option_chain,
    slice_completed_candles,
)
from nifty_analyst.config import AnalystConfig
from nifty_analyst.engine.detectors import (
    analyze_price_structure,
    run_two_track_detectors,
)
from nifty_analyst.engine.pricing import (
    compute_time_to_expiry_years,
    run_chain_qa_suite,
)
from nifty_analyst.engine.structures import (
    compute_positioning_evidence,
    compute_skew_and_curvature,
    generate_candidate_structures,
)
from nifty_analyst.output.audit_log import AuditDeliveryResult, deliver_run_output
from nifty_analyst.scrubber import ensure_ist, is_approved_schedule_slot
from nifty_analyst.state import (
    CandidateStructure,
    ContractContext,
    EventCalendarEvidence,
    FinalOutput,
    MarketEvidence,
    NiftyOptionsState,
    OpenPosition,
    PortfolioExposureState,
    PositioningEvidence,
    PriceStructureEvidence,
    RunMetadata,
    TimestampedDatum,
    VerifierReport,
    VolatilityEvidence,
)
from nifty_analyst.synthesis.llm_node import LLMProvider, build_llm_provider
from nifty_analyst.synthesis.verifier import verify_llm_synthesis


def build_portfolio_exposure_state(
    open_positions: Sequence[OpenPosition],
    portfolio_cap_rs: float = 10000.0,
) -> PortfolioExposureState:
    """Build PortfolioExposureState counting EVERY position's planned stop in full (zero netting)."""
    positions_list = list(open_positions)
    summed_stops = round(sum(p.planned_stop_rs for p in positions_list), 2)
    remaining = round(max(0.0, portfolio_cap_rs - summed_stops), 2)

    groups: dict[str, list[str]] = {"BULLISH": [], "BEARISH": [], "NEUTRAL": []}
    for p in positions_list:
        if p.directional_sign > 0:
            groups["BULLISH"].append(p.position_id)
        elif p.directional_sign < 0:
            groups["BEARISH"].append(p.position_id)
        else:
            groups["NEUTRAL"].append(p.position_id)

    return PortfolioExposureState(
        open_positions=positions_list,
        summed_planned_stops_rs=summed_stops,
        remaining_stop_budget_rs=remaining,
        directional_exposure_groups=groups,
    )


def _rank_candidate(
    cand: CandidateStructure,
    ivp_90d: float | str,
    primary_direction: str | None,
) -> float:
    """Multi-factor candidate score (Section 6.2(3)): IV-regime fit, risk-reward, PoP, liquidity.

    Never penalizes a structure for departing from Part B habitual style.
    """
    if cand.part_a_overall_status == "FAIL":
        return -1e9

    score = 0.0
    if primary_direction and cand.direction == primary_direction:
        score += 40.0

    # IV-regime fit (vega sign vs IVP):
    # Low IVP (<45) favours positive/neutral vega (debit spreads);
    # High IVP (>=50) favours negative vega (credit spreads / condors).
    if isinstance(ivp_90d, (int, float)):
        if ivp_90d < 45.0 and not cand.is_net_credit:
            score += 20.0
        elif ivp_90d >= 50.0 and cand.is_net_credit:
            score += 20.0

    rr = cand.max_profit_rs / max(1.0, cand.contractual_max_loss_rs)
    score += min(25.0, rr * 15.0)
    score += 0.3 * cand.pop_lognormal_expiry
    score += 0.15 * cand.structure_liquidity_score
    return round(score, 4)


def _create_insufficient_data_state(
    run_id: str,
    dec_ist: dt.datetime,
    config: AnalystConfig,
    portfolio_state: PortfolioExposureState,
    reason: str,
    contract_context: ContractContext | None = None,
) -> NiftyOptionsState:
    fallback_spot = TimestampedDatum.create(
        value=0.0,
        as_of_dt=dec_ist,
        fetched_at_dt=dec_ist,
        source_id="halt_fallback",
        max_staleness_seconds=config.max_quote_staleness_seconds,
        verification_tag="HALTED",
    )
    ctx = contract_context or ContractContext(
        symbol=config.symbol,
        spot=fallback_spot,
        active_expiries=[],
    )
    empty_ps = PriceStructureEvidence(
        trend_1d_state="RANGE",
        higher_high_higher_low_repaired=False,
        first_pullback_in_downtrend_warning=False,
        ma20_1h=0.0,
        ema50_1h=0.0,
        last_completed_1d_bar=None,
        last_completed_1h_bar=None,
        final_15m_bar_completed=False,
        stripped_forming_bars_count=0,
        verified_level_touches=[],
    )
    empty_vol = VolatilityEvidence(
        india_vix=TimestampedDatum.create(
            None, dec_ist, dec_ist, "unverified", 180, "UNVERIFIED"
        ),
        atm_iv_by_expiry={},
        constant_maturity_30d_iv=0.0,
        eod_cm30_latest="UNKNOWN",
        iv_source_label="LIVE_MID_INFERRED",
        ivp_30d="UNKNOWN",
        ivp_90d="UNKNOWN",
        ivp_252d="UNKNOWN",
        iv_history_sessions_count=0,
        iv_history_gaps_flagged=[reason],
        realized_vol_20d=0.0,
        iv_rv_spread=0.0,
        skew_25d_put_minus_call=0.0,
        wing_curvature=0.0,
        chain_qa_passed=False,
        chain_qa_notes=[reason],
    )
    empty_pos = PositioningEvidence(
        target_expiry="N/A",
        pcr_oi=0.0,
        pcr_volume=0.0,
        call_oi_wall_strike=0.0,
        call_oi_wall_contracts=0,
        put_oi_wall_strike=0.0,
        put_oi_wall_contracts=0,
        max_pain_strike=0.0,
    )
    empty_ev_cal = EventCalendarEvidence(
        events_within_dte=[],
        calendar_file_age_days=0.0,
        is_calendar_stale_warning=False,
        source_verification_tag="UNVERIFIED",
    )
    meta = RunMetadata(
        run_id=run_id,
        decision_timestamp_ist=dec_ist.isoformat(),
        mode=config.mode,
        data_version_hash="halt_no_data",
        model_ids=[config.llm_model_id],
        config_hash=config.config_hash(),
        scheduled_slot_approved=is_approved_schedule_slot(dec_ist),
    )
    fo = FinalOutput(
        verdict="INSUFFICIENT DATA",
        is_live_ready=False,
        research_only_reason=reason,
        primary_proposal=None,
        alternative_candidates=[],
        habitual_style_comparison_note="",
        reasoning_summary=f"Run halted with INSUFFICIENT DATA: {reason}",
        part_a_summary_status="FAIL",
        research_rule_challenge_note="",
        confidence_note=f"HALTED: {reason}",
        staleness_report=[reason],
        llm_citations=[],
        verifier_report=VerifierReport(
            total_claims_checked=0,
            unverified_claims_count=0,
            unsupported_claim_rate=0.0,
            part_a_reverified=False,
            passed_verification=False,
            failure_reasons=[reason],
        ),
    )
    return NiftyOptionsState(
        run_metadata=meta,
        contract_context=ctx,
        portfolio_exposure_state=portfolio_state,
        market_evidence=MarketEvidence(
            price_structure=empty_ps,
            volatility_state=empty_vol,
            derivatives_positioning=empty_pos,
            event_calendar=empty_ev_cal,
            triggered_detectors=[],
        ),
        candidate_structures=[],
        debate_state=None,
        final_output=fo,
    )


def run_analyst_pipeline(
    decision_dt_ist: dt.datetime,
    config: AnalystConfig,
    broker_adapter: KotakNeoReadOnlyPort | None = None,
    llm_provider: LLMProvider | None = None,
    open_positions: Sequence[OpenPosition] | None = None,
    target_expiry_override: str | None = None,
    force_render_for_inspection: bool = False,
) -> tuple[NiftyOptionsState, AuditDeliveryResult]:
    """Execute the full Phase 0 + Phase 1 NIFTY 50 Options Research Analyst pipeline."""
    dec_ist = ensure_ist(decision_dt_ist)
    scrubber = config.build_scrubber()
    run_id = f"run_{dec_ist.strftime('%Y%m%d_%H%M%S')}_{config.config_hash()[:6]}"

    portfolio_state = build_portfolio_exposure_state(
        open_positions or [],
        portfolio_cap_rs=config.portfolio_summed_stop_cap_rs,
    )

    fixtures_path = Path(config.fixtures_dir)
    if broker_adapter is None:
        limiter = ProvisionalSerialRateLimiter(
            min_interval_seconds=config.kotak_min_request_interval_seconds
        )
        if config.mode == "OFFLINE":
            broker_adapter = FixtureKotakNeoAdapter(
                fixtures_dir=fixtures_path,
                rate_limiter=limiter,
                enforce_schedule_in_live_mode=False,
            )
        else:
            broker_adapter = UnverifiedLiveKotakNeoAdapter(rate_limiter=limiter)

    try:
        holidays, holiday_descs = load_nse_holiday_calendar(
            fixtures_path / "nse_holidays_2026.json"
        )
        raw_master = broker_adapter.fetch_contract_master_raw(dec_ist)
        expected_cal_expiries = raw_master.get("expected_calendar_expiries")
        contract_ctx = parse_contract_master(
            raw_master=raw_master,
            decision_dt_ist=dec_ist,
            holidays=holidays,
            holiday_descriptions=holiday_descs,
            expected_calendar_expiries=expected_cal_expiries,
            max_staleness_seconds=config.max_quote_staleness_seconds,
        )
    except (ContractMasterError, KotakAdapterError, FileNotFoundError, KeyError, ValueError) as exc:
        state = _create_insufficient_data_state(
            run_id=run_id,
            dec_ist=dec_ist,
            config=config,
            portfolio_state=portfolio_state,
            reason=f"Contract master / adapter failure: {exc}",
        )
        delivery = deliver_run_output(
            state,
            config.audit_log_path,
            scrubber=scrubber,
            force_render_for_inspection=force_render_for_inspection,
        )
        return state, delivery

    if contract_ctx.spot.is_stale:
        state = _create_insufficient_data_state(
            run_id=run_id,
            dec_ist=dec_ist,
            config=config,
            portfolio_state=portfolio_state,
            reason=(
                f"Spot quote is STALE (as_of={contract_ctx.spot.as_of_ist}, "
                f"max_staleness={config.max_quote_staleness_seconds}s)."
            ),
            contract_context=contract_ctx,
        )
        delivery = deliver_run_output(
            state,
            config.audit_log_path,
            scrubber=scrubber,
            force_render_for_inspection=force_render_for_inspection,
        )
        return state, delivery

    # Select target expiry (prefer nearest expiry with DTE >= 2 sessions unless overridden)
    target_expiry_meta = contract_ctx.active_expiries[0]
    if target_expiry_override:
        for exp in contract_ctx.active_expiries:
            if exp.expiry_date == target_expiry_override:
                target_expiry_meta = exp
                break
    else:
        for exp in contract_ctx.active_expiries:
            if exp.dte_trading_sessions >= 2:
                target_expiry_meta = exp
                break

    try:
        raw_chain = broker_adapter.fetch_option_chain_raw(
            target_expiry_meta.expiry_date, dec_ist
        )
        chain_env = parse_and_validate_option_chain(
            raw_chain=raw_chain,
            expiry_date=target_expiry_meta.expiry_date,
            decision_dt_ist=dec_ist,
            max_staleness_seconds=config.max_quote_staleness_seconds,
        )
        raw_candles = broker_adapter.fetch_index_candles_raw(dec_ist)
        completed_1d, completed_1h, completed_15m, stripped_bars = slice_completed_candles(
            raw_candles, dec_ist
        )
        vix_datum = broker_adapter.fetch_india_vix_raw(dec_ist)
    except KotakAdapterError as exc:
        state = _create_insufficient_data_state(
            run_id=run_id,
            dec_ist=dec_ist,
            config=config,
            portfolio_state=portfolio_state,
            reason=f"Broker quote / candle fetch failure: {exc}",
            contract_context=contract_ctx,
        )
        delivery = deliver_run_output(
            state,
            config.audit_log_path,
            scrubber=scrubber,
            force_render_for_inspection=force_render_for_inspection,
        )
        return state, delivery

    if chain_env.any_stale:
        state = _create_insufficient_data_state(
            run_id=run_id,
            dec_ist=dec_ist,
            config=config,
            portfolio_state=portfolio_state,
            reason=(
                f"Active option quotes for expiry {target_expiry_meta.expiry_date} exceed "
                f"{config.max_quote_staleness_seconds}s staleness bound at strikes: {chain_env.stale_strikes}."
            ),
            contract_context=contract_ctx,
        )
        delivery = deliver_run_output(
            state,
            config.audit_log_path,
            scrubber=scrubber,
            force_render_for_inspection=force_render_for_inspection,
        )
        return state, delivery

    spot_val = contract_ctx.spot.value
    t_years = compute_time_to_expiry_years(dec_ist, target_expiry_meta.expiry_date)
    qa_result = run_chain_qa_suite(
        spot=spot_val,
        quotes=chain_env.quotes,
        time_to_expiry_years=t_years,
        risk_free_rate=config.risk_free_rate_annual,
    )

    if not qa_result.passed:
        state = _create_insufficient_data_state(
            run_id=run_id,
            dec_ist=dec_ist,
            config=config,
            portfolio_state=portfolio_state,
            reason=f"Chain QA aborted ({qa_result.failure_code}): {'; '.join(qa_result.notes)}",
            contract_context=contract_ctx,
        )
        delivery = deliver_run_output(
            state,
            config.audit_log_path,
            scrubber=scrubber,
            force_render_for_inspection=force_render_for_inspection,
        )
        return state, delivery

    atm_iv = round(0.5 * (qa_result.atm_call_iv + qa_result.atm_put_iv), 6)
    # Basis policy: the percentiled quantity is the LATEST COMPLETED EOD cm30 at-forward
    # IV from the validated official-settlement series (current_cm30_iv=None -> adapter
    # uses its own latest clamped record). The live intraday spot-ATM IV below is a
    # different basis (intraday mid, nearest listed expiry) and is NEVER percentiled.
    eod_vol_summary = load_eod_volatility_history(
        eod_archive_path=fixtures_path / "nse_eod_volatility_history.json",
        decision_dt_ist=dec_ist,
        current_cm30_iv=None,
        holidays=holidays,
        risk_free_rate=config.risk_free_rate_annual,
        special_sessions=getattr(config, "special_sessions", None),
    )
    eod_cm30_latest = (
        round(eod_vol_summary.records[-1].constant_maturity_30d_iv, 6)
        if eod_vol_summary.records else "UNKNOWN"
    )
    skew_25d, wing_curv = compute_skew_and_curvature(
        quotes=chain_env.quotes,
        implied_forward=qa_result.implied_forward,
        atm_iv=atm_iv,
        time_to_expiry_years=t_years,
        risk_free_rate=config.risk_free_rate_annual,
    )

    vol_evidence = VolatilityEvidence(
        india_vix=vix_datum,
        atm_iv_by_expiry={target_expiry_meta.expiry_date: atm_iv},
        constant_maturity_30d_iv=atm_iv,
        eod_cm30_latest=eod_cm30_latest,
        iv_source_label="LIVE_INTRADAY_SPOT_ATM_NEAREST_EXPIRY",
        ivp_30d=eod_vol_summary.ivp_30d,
        ivp_90d=eod_vol_summary.ivp_90d,
        ivp_252d=eod_vol_summary.ivp_252d,
        iv_history_sessions_count=len(eod_vol_summary.records),
        iv_history_gaps_flagged=eod_vol_summary.continuity_gaps_flagged,
        realized_vol_20d=eod_vol_summary.realized_vol_20d,
        iv_rv_spread=round(atm_iv - eod_vol_summary.realized_vol_20d, 4),
        skew_25d_put_minus_call=skew_25d,
        wing_curvature=wing_curv,
        chain_qa_passed=qa_result.passed,
        chain_qa_notes=qa_result.notes,
    )

    pos_evidence = compute_positioning_evidence(
        expiry_date=target_expiry_meta.expiry_date,
        quotes=chain_env.quotes,
    )
    price_structure = analyze_price_structure(
        completed_1d=completed_1d,
        completed_1h=completed_1h,
        completed_final_15m=completed_15m,
        stripped_forming_bars_count=stripped_bars,
        extra_reference_levels={
            "PUT_OI_WALL": pos_evidence.put_oi_wall_strike,
            "CALL_OI_WALL": pos_evidence.call_oi_wall_strike,
        },
    )
    event_cal = load_events_calendar(
        events_file_path=fixtures_path / "vix_and_events.json",
        decision_dt_ist=dec_ist,
        max_expiry_date_str=contract_ctx.active_expiries[-1].expiry_date,
        max_calendar_age_days=config.max_event_calendar_age_days,
    )
    detectors = run_two_track_detectors(
        spot=spot_val,
        price_structure=price_structure,
        volatility=vol_evidence,
        positioning=pos_evidence,
    )

    candidates = generate_candidate_structures(
        expiry=target_expiry_meta,
        quotes=chain_env.quotes,
        stale_strikes=chain_env.stale_strikes,
        spot=spot_val,
        implied_forward=qa_result.implied_forward,
        atm_iv=atm_iv,
        time_to_expiry_years=t_years,
        empirical_daily_log_returns=eod_vol_summary.empirical_daily_log_returns,
        triggered_detectors=detectors,
        portfolio_state=portfolio_state,
        price_structure=price_structure,
        volatility=vol_evidence,
        config=config,
    )

    # Determine primary direction from confirmed quant regime detectors
    confirmed_quant_dirs = [
        d.direction for d in detectors if d.confirmation_status == "CONFIRMED_QUANT_REGIME"
    ]
    primary_dir = confirmed_quant_dirs[0] if confirmed_quant_dirs else None

    # Primary must be certifiable: Part A overall PASS only. UNKNOWN-bound candidates
    # (e.g. expiry-hold exercise-STT tail) stay research-only in the Section 6 comparison
    # and can never become the primary proposal or trigger a SUGGESTION.
    eligible = [
        c for c in candidates if c.part_a_overall_status == "PASS"
    ]
    eligible.sort(
        key=lambda c: _rank_candidate(c, vol_evidence.ivp_90d, primary_dir),
        reverse=True,
    )

    selected_primary: CandidateStructure | None = None
    if eligible and detectors:
        selected_primary = eligible[0]

    alternatives = [
        c for c in candidates if selected_primary is None or c.structure_id != selected_primary.structure_id
    ]

    data_hash_payload = json.dumps(
        {
            "spot": contract_ctx.spot.value,
            "spot_as_of": contract_ctx.spot.as_of_ist,
            "chain_earliest": chain_env.earliest_quote_ist,
            "chain_latest": chain_env.latest_quote_ist,
            "eod_sessions": len(eod_vol_summary.records),
        },
        sort_keys=True,
    )
    data_version_hash = hashlib.sha256(data_hash_payload.encode("utf-8")).hexdigest()[:16]

    provider = llm_provider or build_llm_provider(config)
    meta = RunMetadata(
        run_id=run_id,
        decision_timestamp_ist=dec_ist.isoformat(),
        mode=config.mode,
        data_version_hash=data_version_hash,
        model_ids=[config.llm_model_id],
        config_hash=config.config_hash(),
        scheduled_slot_approved=is_approved_schedule_slot(dec_ist),
    )

    pre_llm_state = NiftyOptionsState(
        run_metadata=meta,
        contract_context=contract_ctx,
        portfolio_exposure_state=portfolio_state,
        market_evidence=MarketEvidence(
            price_structure=price_structure,
            volatility_state=vol_evidence,
            derivatives_positioning=pos_evidence,
            event_calendar=event_cal,
            triggered_detectors=detectors,
        ),
        candidate_structures=candidates,
        debate_state=None,
        final_output=None,
    )

    sanitised_json = json.dumps(scrubber.scrub_obj(pre_llm_state.to_dict()), indent=2)
    llm_resp = provider.synthesize(
        sanitised_state_json=sanitised_json,
        state=pre_llm_state,
        primary=selected_primary,
        alternatives=alternatives,
    )

    verified_citations, verifier_report = verify_llm_synthesis(
        state=pre_llm_state,
        raw_citations=llm_resp.citations,
        selected_primary=selected_primary,
    )

    # Chosen verified IVP lookbacks: 30 and 90 sessions (explicit). ivp_252d is
    # intentionally NOT in this gate: a 252-session IVP is unavailable while history
    # holds fewer than 252 verified sessions, and unavailability by history length
    # must not block candidates when the configured 30/90 lookbacks are verified.
    ivp_unknown = (
        vol_evidence.ivp_30d == "UNKNOWN" or vol_evidence.ivp_90d == "UNKNOWN"
    )
    # A SUGGESTION requires a certified primary: Part A overall PASS. Rule 4 gates the
    # KNOWN gross contractual payoff bound against the Rs 10,000 ceiling (7 Nov 2026
    # decision); an unbounded all-in expiry-hold path is disclosed via the all-in cost
    # label, never gated UNKNOWN and never a strategy ban. A primary that FAILs any
    # Part A rule cannot qualify; the run stays WAIT with candidates visible
    # research-only in Section 6.
    if (
        selected_primary is not None
        and verifier_report.passed_verification
        and selected_primary.part_a_overall_status == "PASS"
    ):
        verdict = "SUGGESTION"
    else:
        verdict = "WAIT"
        selected_primary = None

    is_live_ready = (
        verdict == "SUGGESTION"
        and not ivp_unknown
        and selected_primary is not None
        and selected_primary.part_a_overall_status == "PASS"
    )
    research_only_reason = ""
    if not is_live_ready:
        if ivp_unknown:
            research_only_reason = "IVP is UNKNOWN; output is labelled research-only per Section 4.2 & 5A.8."
        elif verdict != "SUGGESTION":
            research_only_reason = f"Verdict is {verdict}."
        elif selected_primary is not None:
            research_only_reason = (
                f"Primary candidate Part A status is {selected_primary.part_a_overall_status}."
            )

    staleness_notes = [
        f"Spot as-of {contract_ctx.spot.as_of_ist} (stale={contract_ctx.spot.is_stale})",
        f"Chain quotes [{chain_env.earliest_quote_ist} .. {chain_env.latest_quote_ist}] (any_stale={chain_env.any_stale})",
    ]
    if event_cal.is_calendar_stale_warning:
        staleness_notes.append(
            f"WARNING: Events calendar file is {event_cal.calendar_file_age_days:.1f} days old (> 14 days)."
        )

    final_output = FinalOutput(
        verdict=verdict,
        is_live_ready=is_live_ready,
        research_only_reason=research_only_reason,
        primary_proposal=selected_primary,
        alternative_candidates=alternatives,
        habitual_style_comparison_note=llm_resp.habitual_comparison_note,
        reasoning_summary=llm_resp.reasoning_summary,
        part_a_summary_status=(
            selected_primary.part_a_overall_status if selected_primary else "FAIL"
        ),
        research_rule_challenge_note=llm_resp.research_rule_challenge_note,
        confidence_note=llm_resp.confidence_note,
        staleness_report=staleness_notes,
        llm_citations=verified_citations,
        verifier_report=verifier_report,
    )

    completed_state = NiftyOptionsState(
        run_metadata=meta,
        contract_context=contract_ctx,
        portfolio_exposure_state=portfolio_state,
        market_evidence=pre_llm_state.market_evidence,
        candidate_structures=candidates,
        debate_state=None,
        final_output=final_output,
    )

    delivery = deliver_run_output(
        state=completed_state,
        audit_log_path=config.audit_log_path,
        scrubber=scrubber,
        force_render_for_inspection=force_render_for_inspection,
    )
    return completed_state, delivery
