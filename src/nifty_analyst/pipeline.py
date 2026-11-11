"""End-to-end deterministic + single-pass persona research pipeline (v0.4.0).

Orchestrates:
- Layer 1: Read-Only Adapters (Contract Master, Multi-Expiry Chains, Completed Candles, EOD IVP, Events)
- Layer 2: Deterministic Quant Engine (Black-76, Chain QA, Variance-Interpolated CM30 IV,
  RSI/ATR/Stretch Technicals, Symmetric Two-Track Detectors, Multi-Expiry Candidate Structures,
  and STT-Free Part A Gatekeeper)
- Layer 3: Single-Pass TradingAgents Persona Synthesis (Bull/Bear/Vol Debate, Risk Triad,
  Consolidator) & Deterministic Two-Way Citation + Prose Verifier
- Layer 4: Intraday Deduplication, Payoff Diagram, Alert-Only Output & Immutable Audit Log
"""

from __future__ import annotations

import dataclasses
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
    ChainSnapshotEnvelope,
    FixtureKotakNeoAdapter,
    KotakAdapterError,
    KotakNeoReadOnlyPort,
    LiveKotakNeoAdapter,
    ProvisionalSerialRateLimiter,
    parse_and_validate_option_chain,
    slice_completed_candles,
)
from nifty_analyst.config import AnalystConfig
from nifty_analyst.engine.detectors import (
    analyze_price_structure,
    run_two_track_detectors,
)
from nifty_analyst.engine.pricing import (
    ChainQAResult,
    LegPayoffSpec,
    compute_time_to_expiry_years,
    compute_variance_interpolated_cm30_iv,
    run_chain_qa_suite,
)
from nifty_analyst.engine.structures import (
    compute_positioning_evidence,
    compute_skew_and_curvature,
    generate_candidate_structures,
)
from nifty_analyst.output.audit_log import AuditDeliveryResult, deliver_run_output
from nifty_analyst.output.intraday_memory import load_same_day_intraday_context
from nifty_analyst.scrubber import ensure_ist, is_approved_schedule_slot
from nifty_analyst.state import (
    CandidateStructure,
    ConsolidatorState,
    ContractContext,
    EventCalendarEvidence,
    ExpiryMetadata,
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
from nifty_analyst.synthesis.personas import format_whatsapp_digest
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
    """Rank candidates strictly by Part A certification and Technical + Options Surface score.

    Never penalizes a structure for departing from Part B habitual style and includes zero STT bias.
    """
    if cand.part_a_overall_status != "PASS":
        return -1e9

    score = cand.technical_options_score
    if primary_direction and cand.direction == primary_direction:
        score += 10.0

    if isinstance(ivp_90d, (int, float)):
        if ivp_90d < 45.0 and not cand.is_net_credit:
            score += 4.0
        elif ivp_90d >= 50.0 and cand.is_net_credit:
            score += 4.0

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
        whatsapp_digest=f"*NIFTY Options Desk — INSUFFICIENT DATA*\n• Halt Reason: {reason}",
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


def _maybe_render_payoff_chart(
    cand: CandidateStructure | None,
    run_id: str,
    config: AnalystConfig,
) -> str:
    if cand is None or not config.payoff_chart_dir:
        return ""
    try:
        from nifty_analyst.output.payoff_diagram import render_payoff_diagram

        out_dir = Path(config.payoff_chart_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{run_id}_{cand.structure_id}.png"
        slip = config.slippage_buffer_points_per_leg
        specs = [
            LegPayoffSpec(
                strike=l.strike,
                option_type=l.option_type,
                side=l.side,
                ratio=l.ratio,
                executable_entry_price=(
                    min(l.bid, max(0.05, l.bid - slip)) if l.side == "SHORT" else (l.ask + slip)
                ),
            )
            for l in cand.legs
        ]
        render_payoff_diagram(
            legs=specs,
            lot_size=cand.lot_size,
            lots=cand.lots,
            statutory_charges_rs=cand.statutory_charges.total_charges_rs,
            title=f"{cand.structure_id} ({cand.structure_type})",
            out_path=str(out_file),
        )
        return str(out_file)
    except Exception:
        return ""


def run_analyst_pipeline(
    decision_dt_ist: dt.datetime,
    config: AnalystConfig,
    broker_adapter: KotakNeoReadOnlyPort | None = None,
    llm_provider: LLMProvider | None = None,
    open_positions: Sequence[OpenPosition] | None = None,
    target_expiry_override: str | None = None,
    force_render_for_inspection: bool = False,
) -> tuple[NiftyOptionsState, AuditDeliveryResult]:
    """Execute the full NIFTY 50 Options Research Analyst pipeline (single-pass, zero multicall)."""
    dec_ist = ensure_ist(decision_dt_ist)
    scrubber = config.build_scrubber()
    run_id = f"run_{dec_ist.strftime('%Y%m%d_%H%M%S')}_{config.config_hash()[:6]}"

    portfolio_state = build_portfolio_exposure_state(
        open_positions or [],
        portfolio_cap_rs=config.portfolio_summed_stop_cap_rs,
    )

    try:
        holidays, holiday_descs = load_nse_holiday_calendar(config.resolved_holidays_path())
        if broker_adapter is None:
            if config.mode == "OFFLINE":
                broker_adapter = FixtureKotakNeoAdapter(
                    fixtures_dir=Path(config.fixtures_dir),
                    rate_limiter=ProvisionalSerialRateLimiter(min_interval_seconds=0.0),
                    enforce_schedule_in_live_mode=False,
                )
            else:
                live_limiter = ProvisionalSerialRateLimiter(
                    min_interval_seconds=config.kotak_min_request_interval_seconds
                )
                trading_cal = {
                    "holiday_dates": frozenset(d.isoformat() for d in holidays),
                    "special_sessions": config.special_sessions,
                }
                broker_adapter = LiveKotakNeoAdapter(
                    kotak_api_root=config.kotak_api_root or None,
                    rate_limiter=live_limiter,
                    enforce_schedule_in_live_mode=True,
                    trading_calendar=trading_cal,  # type: ignore[arg-type]
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

    # Select anchor expiry (prefer nearest expiry with DTE >= 2 sessions unless overridden)
    anchor_expiry_meta = contract_ctx.active_expiries[0]
    if target_expiry_override:
        for exp in contract_ctx.active_expiries:
            if exp.expiry_date == target_expiry_override:
                anchor_expiry_meta = exp
                break
        expiries_to_evaluate = [anchor_expiry_meta]
    else:
        for exp in contract_ctx.active_expiries:
            if exp.dte_trading_sessions >= 2:
                anchor_expiry_meta = exp
                break
        if config.evaluate_all_active_expiries:
            expiries_to_evaluate = [
                exp for exp in contract_ctx.active_expiries if exp.dte_trading_sessions >= 1
            ] or [anchor_expiry_meta]
        else:
            expiries_to_evaluate = [anchor_expiry_meta]

    # Ensure anchor expiry is fetched & validated first
    expiries_ordered = [anchor_expiry_meta] + [
        e for e in expiries_to_evaluate if e.expiry_date != anchor_expiry_meta.expiry_date
    ]

    evaluated_chains: list[tuple[ExpiryMetadata, ChainSnapshotEnvelope, ChainQAResult, float, float]] = []
    spot_val = contract_ctx.spot.value

    try:
        raw_candles = broker_adapter.fetch_index_candles_raw(dec_ist)
        completed_1d, completed_1h, completed_15m, stripped_bars = slice_completed_candles(
            raw_candles, dec_ist
        )
        vix_datum = broker_adapter.fetch_india_vix_raw(dec_ist)

        for idx_exp, exp_meta in enumerate(expiries_ordered):
            is_anchor = idx_exp == 0
            try:
                raw_chain = broker_adapter.fetch_option_chain_raw(exp_meta.expiry_date, dec_ist)
                chain_env = parse_and_validate_option_chain(
                    raw_chain=raw_chain,
                    expiry_date=exp_meta.expiry_date,
                    decision_dt_ist=dec_ist,
                    max_staleness_seconds=config.max_quote_staleness_seconds,
                )
            except KotakAdapterError:
                if is_anchor:
                    raise
                continue

            if chain_env.any_stale:
                if is_anchor:
                    state = _create_insufficient_data_state(
                        run_id=run_id,
                        dec_ist=dec_ist,
                        config=config,
                        portfolio_state=portfolio_state,
                        reason=(
                            f"Active option quotes for expiry {exp_meta.expiry_date} exceed "
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
                continue

            t_yrs = compute_time_to_expiry_years(dec_ist, exp_meta.expiry_date)
            qa_res = run_chain_qa_suite(
                spot=spot_val,
                quotes=chain_env.quotes,
                time_to_expiry_years=t_yrs,
                risk_free_rate=config.risk_free_rate_annual,
            )
            if not qa_res.passed:
                if is_anchor:
                    state = _create_insufficient_data_state(
                        run_id=run_id,
                        dec_ist=dec_ist,
                        config=config,
                        portfolio_state=portfolio_state,
                        reason=f"Chain QA aborted ({qa_res.failure_code}): {'; '.join(qa_res.notes)}",
                        contract_context=contract_ctx,
                    )
                    delivery = deliver_run_output(
                        state,
                        config.audit_log_path,
                        scrubber=scrubber,
                        force_render_for_inspection=force_render_for_inspection,
                    )
                    return state, delivery
                continue

            exp_atm_iv = round(0.5 * (qa_res.atm_call_iv + qa_res.atm_put_iv), 6)
            evaluated_chains.append((exp_meta, chain_env, qa_res, t_yrs, exp_atm_iv))

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

    anchor_meta, anchor_chain_env, anchor_qa, anchor_t_yrs, anchor_atm_iv = evaluated_chains[0]
    atm_iv_by_expiry = {m.expiry_date: iv for (m, _, _, _, iv) in evaluated_chains}
    t_and_ivs = [(t_y, iv) for (_, _, _, t_y, iv) in evaluated_chains]
    cm30_iv, ts_slope, ts_regime = compute_variance_interpolated_cm30_iv(t_and_ivs, target_days=30.0)
    iv_source_lbl = (
        "LIVE_VARIANCE_INTERPOLATED_CM30"
        if len(evaluated_chains) >= 2
        else "LIVE_INTRADAY_SPOT_ATM_NEAREST_EXPIRY"
    )

    eod_vol_summary = load_eod_volatility_history(
        eod_archive_path=config.resolved_eod_archive_path(),
        decision_dt_ist=dec_ist,
        current_cm30_iv=None,
        holidays=holidays,
        risk_free_rate=config.risk_free_rate_annual,
        special_sessions=config.special_sessions,
    )
    eod_cm30_latest = (
        round(eod_vol_summary.records[-1].constant_maturity_30d_iv, 6)
        if eod_vol_summary.records
        else "UNKNOWN"
    )
    skew_25d, wing_curv = compute_skew_and_curvature(
        quotes=anchor_chain_env.quotes,
        implied_forward=anchor_qa.implied_forward,
        atm_iv=anchor_atm_iv,
        time_to_expiry_years=anchor_t_yrs,
        risk_free_rate=config.risk_free_rate_annual,
    )

    vol_evidence = VolatilityEvidence(
        india_vix=vix_datum,
        atm_iv_by_expiry=atm_iv_by_expiry,
        constant_maturity_30d_iv=cm30_iv,
        eod_cm30_latest=eod_cm30_latest,
        iv_source_label=iv_source_lbl,
        ivp_30d=eod_vol_summary.ivp_30d,
        ivp_90d=eod_vol_summary.ivp_90d,
        ivp_252d=eod_vol_summary.ivp_252d,
        iv_history_sessions_count=len(eod_vol_summary.records),
        iv_history_gaps_flagged=eod_vol_summary.continuity_gaps_flagged,
        realized_vol_20d=eod_vol_summary.realized_vol_20d,
        iv_rv_spread=round(cm30_iv - eod_vol_summary.realized_vol_20d, 4),
        skew_25d_put_minus_call=skew_25d,
        wing_curvature=wing_curv,
        chain_qa_passed=anchor_qa.passed,
        chain_qa_notes=anchor_qa.notes,
        term_structure_slope_near_minus_far=ts_slope,
        term_structure_regime=ts_regime,
    )

    pos_raw = compute_positioning_evidence(
        expiry_date=anchor_meta.expiry_date,
        quotes=anchor_chain_env.quotes,
    )
    intraday_ctx = load_same_day_intraday_context(
        audit_log_path=config.audit_log_path,
        decision_dt_ist=dec_ist,
        current_spot=spot_val,
        current_cm30_iv=cm30_iv,
        current_pcr_oi=pos_raw.pcr_oi,
    )
    pos_evidence = dataclasses.replace(
        pos_raw,
        intraday_spot_change_pts=intraday_ctx.intraday_spot_change_pts,
        intraday_atm_iv_change_pts=intraday_ctx.intraday_atm_iv_change_pts,
        intraday_pcr_oi_change=intraday_ctx.intraday_pcr_oi_change,
        prior_intraday_runs_count=intraday_ctx.prior_intraday_runs_count,
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
        events_file_path=config.resolved_events_path(),
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

    candidates: list[CandidateStructure] = []
    for exp_meta, chain_env, qa_res, t_yrs, exp_atm_iv in evaluated_chains:
        exp_pos = (
            pos_evidence
            if exp_meta.expiry_date == anchor_meta.expiry_date
            else compute_positioning_evidence(exp_meta.expiry_date, chain_env.quotes)
        )
        candidates.extend(
            generate_candidate_structures(
                expiry=exp_meta,
                quotes=chain_env.quotes,
                stale_strikes=chain_env.stale_strikes,
                spot=spot_val,
                implied_forward=qa_res.implied_forward,
                atm_iv=exp_atm_iv,
                time_to_expiry_years=t_yrs,
                empirical_daily_log_returns=eod_vol_summary.empirical_daily_log_returns,
                triggered_detectors=detectors,
                portfolio_state=portfolio_state,
                price_structure=price_structure,
                volatility=vol_evidence,
                config=config,
                positioning=exp_pos,
            )
        )

    confirmed_quant_dirs = [
        d.direction for d in detectors if d.confirmation_status == "CONFIRMED_QUANT_REGIME"
    ]
    primary_dir = confirmed_quant_dirs[0] if confirmed_quant_dirs else None

    eligible = [c for c in candidates if c.part_a_overall_status == "PASS"]
    eligible.sort(
        key=lambda c: _rank_candidate(c, vol_evidence.ivp_90d, primary_dir),
        reverse=True,
    )

    selected_primary: CandidateStructure | None = eligible[0] if (eligible and detectors) else None
    alternatives = [
        c
        for c in candidates
        if selected_primary is None or c.structure_id != selected_primary.structure_id
    ]

    earliest_q = min(env.earliest_quote_ist for (_, env, _, _, _) in evaluated_chains)
    latest_q = max(env.latest_quote_ist for (_, env, _, _, _) in evaluated_chains)
    data_hash_payload = json.dumps(
        {
            "spot": contract_ctx.spot.value,
            "spot_as_of": contract_ctx.spot.as_of_ist,
            "chain_earliest": earliest_q,
            "chain_latest": latest_q,
            "expiries_evaluated": [m.expiry_date for (m, _, _, _, _) in evaluated_chains],
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
        scheduled_slot_approved=is_approved_schedule_slot(
            dec_ist,
            holiday_dates=frozenset(d.isoformat() for d in holidays),
            special_sessions=config.special_sessions,
        ),
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

    # Honour Consolidator's selection (must be 'WAIT' or a certified Part-A-PASS candidate)
    consolidator_rec_id: str | None = None
    if llm_resp.consolidator_state is not None:
        consolidator_rec_id = llm_resp.consolidator_state.recommended_structure_id
        if consolidator_rec_id == "WAIT":
            selected_primary = None
        else:
            cand_by_id = {c.structure_id: c for c in candidates}
            if consolidator_rec_id in cand_by_id:
                selected_primary = cand_by_id[consolidator_rec_id]

    prose_blocks: list[str] = [llm_resp.reasoning_summary]
    if llm_resp.debate_state is not None:
        prose_blocks.extend(
            [
                llm_resp.debate_state.bull_thesis,
                llm_resp.debate_state.bear_thesis,
                llm_resp.debate_state.vol_regime_thesis,
            ]
        )
    if llm_resp.consolidator_state is not None:
        prose_blocks.append(llm_resp.consolidator_state.final_recommendation_rationale)

    verified_citations, verifier_report = verify_llm_synthesis(
        state=pre_llm_state,
        raw_citations=llm_resp.citations,
        selected_primary=selected_primary,
        prose_texts=prose_blocks,
        consolidator_recommended_id=consolidator_rec_id,
    )

    ivp_unknown = (
        vol_evidence.ivp_30d == "UNKNOWN" or vol_evidence.ivp_90d == "UNKNOWN"
    )
    any_fetch_time_unverified = any(
        "FETCH_TIME_NOT_EXCHANGE" in env.source_verification_tag
        for (_, env, _, _, _) in evaluated_chains
    )

    if (
        selected_primary is not None
        and verifier_report.passed_verification
        and selected_primary.part_a_overall_status == "PASS"
    ):
        verdict = "SUGGESTION"
    else:
        verdict = "WAIT"
        selected_primary = None

    alternatives = [
        c
        for c in candidates
        if selected_primary is None or c.structure_id != selected_primary.structure_id
    ]

    is_live_ready = (
        verdict == "SUGGESTION"
        and not ivp_unknown
        and not any_fetch_time_unverified
        and selected_primary is not None
        and selected_primary.part_a_overall_status == "PASS"
    )
    research_only_reason = ""
    if not is_live_ready:
        if ivp_unknown:
            research_only_reason = (
                "IVP is UNKNOWN; output is labelled research-only per Section 4.2 & 5A.8."
            )
        elif any_fetch_time_unverified:
            research_only_reason = (
                "Option chain quote timestamps use FETCH_TIME_NOT_EXCHANGE (unverified exchange quote as-of)."
            )
        elif verdict != "SUGGESTION":
            research_only_reason = f"Verdict is {verdict}."
        elif selected_primary is not None:
            research_only_reason = (
                f"Primary candidate Part A status is {selected_primary.part_a_overall_status}."
            )

    duplicate_suppressed = bool(
        config.deduplicate_intraday_alerts
        and verdict == "SUGGESTION"
        and is_live_ready
        and selected_primary is not None
        and selected_primary.structure_id in intraday_ctx.alerted_structure_ids_today
    )

    # Ensure Consolidator state & WhatsApp digest reflect final post-verifier verdict
    final_consolidator = llm_resp.consolidator_state
    if llm_resp.debate_state is not None:
        deterministic_wa = format_whatsapp_digest(
            state=pre_llm_state,
            selected=selected_primary,
            debate=llm_resp.debate_state,
            consolidator_conviction=(
                final_consolidator.conviction
                if (final_consolidator and verdict == "SUGGESTION")
                else "SIT_OUT"
            ),
            is_live_ready=is_live_ready,
        )
        if final_consolidator is not None:
            if verdict != "SUGGESTION":
                final_consolidator = ConsolidatorState(
                    recommended_structure_id="WAIT",
                    conviction="SIT_OUT",
                    final_recommendation_rationale=final_consolidator.final_recommendation_rationale,
                    whatsapp_digest=deterministic_wa,
                )
            elif not final_consolidator.whatsapp_digest:
                final_consolidator = dataclasses.replace(
                    final_consolidator, whatsapp_digest=deterministic_wa
                )
        whatsapp_digest = (
            final_consolidator.whatsapp_digest if final_consolidator else deterministic_wa
        )
    else:
        whatsapp_digest = ""

    chart_path = _maybe_render_payoff_chart(selected_primary, run_id, config)

    staleness_notes = [
        f"Spot as-of {contract_ctx.spot.as_of_ist} (stale={contract_ctx.spot.is_stale})",
        f"Chain quotes across {len(evaluated_chains)} expiry(s) [{earliest_q} .. {latest_q}] (any_stale=False)",
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
        whatsapp_digest=whatsapp_digest,
        duplicate_alert_suppressed=duplicate_suppressed,
        payoff_diagram_path=chart_path,
    )

    completed_state = NiftyOptionsState(
        run_metadata=meta,
        contract_context=contract_ctx,
        portfolio_exposure_state=portfolio_state,
        market_evidence=pre_llm_state.market_evidence,
        candidate_structures=candidates,
        debate_state=llm_resp.debate_state,
        final_output=final_output,
        risk_triad_state=llm_resp.risk_triad_state,
        consolidator_state=final_consolidator,
    )

    delivery = deliver_run_output(
        state=completed_state,
        audit_log_path=config.audit_log_path,
        scrubber=scrubber,
        force_render_for_inspection=force_render_for_inspection,
    )
    return completed_state, delivery
