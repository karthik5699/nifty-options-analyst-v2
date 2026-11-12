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
from dataclasses import dataclass
import datetime as dt
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

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
    KotakRateLimit429Error,
    KotakSharedStateError,
    LiveKotakNeoAdapter,
    ProvisionalSerialRateLimiter,
    parse_and_validate_option_chain,
    slice_completed_candles,
)
from nifty_analyst.adapters.open_positions import load_open_positions
from nifty_analyst.adapters.request_safety import PacedReadTransport
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
from nifty_analyst.scrubber import IST, ensure_ist, is_approved_schedule_slot
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
from nifty_analyst.synthesis.evidence_snapshot import (
    EvidenceSnapshotClockError,
    StaleEvidenceSnapshotError,
    compute_state_evidence_sha256,
    has_existing_snapshot_for_artifact,
    load_and_verify_snapshot_for_resume,
    mint_unique_capture_identities,
    parse_strict_ist_timestamp,
    persist_immutable_snapshot_bundle,
    validate_capture_and_resume_clocks,
)
from nifty_analyst.synthesis.llm_node import (
    LLMProvider,
    _resolve_snapshot_registry_root,
    build_llm_provider,
)
from nifty_analyst.synthesis.personas import (
    PERSONA_DEFINITIONS,
    build_compact_desk_brief,
    build_single_pass_persona_prompt,
    build_verified_report_narrative,
)
from nifty_analyst.synthesis.verifier import (
    assess_market_regime,
    is_candidate_compatible_with_regime,
    verify_llm_synthesis,
)


def build_portfolio_exposure_state(
    open_positions: Sequence[OpenPosition],
    portfolio_cap_rs: float = 10000.0,
    portfolio_state_status: Literal["VERIFIED", "PORTFOLIO_STATE_UNKNOWN"] = "VERIFIED",
    portfolio_state_error: str = "",
) -> PortfolioExposureState:
    """Build PortfolioExposureState counting EVERY position's planned stop in full (zero netting)."""
    positions_list = list(open_positions)
    summed_stops = float(round(sum((p.planned_stop_rs for p in positions_list), 0.0), 2))
    remaining = float(round(max(0.0, portfolio_cap_rs - summed_stops), 2))

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
        portfolio_state_status=portfolio_state_status,
        portfolio_state_error=portfolio_state_error,
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


_UNVERIFIED_QUOTE_TS_TAGS: tuple[str, ...] = (
    "FETCH_TIME_NOT_EXCHANGE",
    "UNVERIFIED_KOTAK_REST_LSTUP_TIME",
    "UNVERIFIED_CANDIDATE_FIELD",
)

_COVERAGE_BUCKET_KEYS: tuple[str, ...] = (
    "listed",
    "selected",
    "requested",
    "returned",
    "missing",
    "rejected",
    "priced",
)


@dataclass(frozen=True)
class BoundedExpiryPlan:
    """Bounded automatic expiry & per-expiry depth-token budget selection."""

    anchor_expiry: ExpiryMetadata
    required_expiries: tuple[ExpiryMetadata, ...]
    selected_expiries: tuple[ExpiryMetadata, ...]
    unlisted_required_expiries: tuple[str, ...]
    token_budget_by_expiry: dict[str, int]
    remaining_request_budget: int
    single_expiry_budget_limited: bool


def select_bounded_expiries_and_budgets(
    active_expiries: Sequence[ExpiryMetadata],
    open_positions: Sequence[OpenPosition],
    config: AnalystConfig,
    broker_adapter: Any,
    target_expiry_override: str | None = None,
) -> BoundedExpiryPlan:
    """Deterministically select required expiries and depth-token budgets within the pacing cap.

    Never evaluates all listed expiries blindly in `LIVE_MORNING_SNAPSHOT` mode:
    1. Anchor expiry (`dte_trading_sessions >= 2` preferred, or `target_expiry_override`).
    2. Any active expiry carrying monitored `open_positions` (`pos.expiry_date`, or
       `pos.dte_trading_sessions` fallback when `pos.neo_symbols` is non-empty).
    3. Next active comparison expiry (`dte_trading_sessions >= 1`, `!= anchor`) when
       `evaluate_all_active_expiries` is enabled so multi-expiry surface / CM30 can be evaluated.
    """
    active_list = list(active_expiries)
    active_by_date = {e.expiry_date: e for e in active_list}
    anchor_expiry = active_list[0]

    if target_expiry_override:
        if target_expiry_override in active_by_date:
            anchor_expiry = active_by_date[target_expiry_override]
            unlisted_override: tuple[str, ...] = ()
        else:
            unlisted_override = (target_expiry_override,)
        return BoundedExpiryPlan(
            anchor_expiry=anchor_expiry,
            required_expiries=(anchor_expiry,),
            selected_expiries=(anchor_expiry,),
            unlisted_required_expiries=unlisted_override,
            token_budget_by_expiry={},
            remaining_request_budget=999,
            single_expiry_budget_limited=False,
        )

    for exp in active_list:
        if exp.dte_trading_sessions >= 2:
            anchor_expiry = exp
            break

    unlisted_req: list[str] = []
    monitored_exps: list[ExpiryMetadata] = []
    monitored_seen: set[str] = set()
    held_count_by_exp: dict[str, int] = {}

    for pos in open_positions:
        exp_str = (pos.expiry_date or "").strip()
        n_syms = len([ns for ns in pos.neo_symbols if ns])
        if exp_str:
            if exp_str in active_by_date:
                exp_m = active_by_date[exp_str]
                held_count_by_exp[exp_str] = held_count_by_exp.get(exp_str, 0) + n_syms
                if exp_str not in monitored_seen:
                    monitored_seen.add(exp_str)
                    monitored_exps.append(exp_m)
            elif exp_str not in unlisted_req:
                unlisted_req.append(exp_str)
        elif n_syms > 0 and pos.dte_trading_sessions > 0:
            matched_dte = next(
                (e for e in active_list if e.dte_trading_sessions == pos.dte_trading_sessions),
                None,
            )
            if matched_dte is not None:
                held_count_by_exp[matched_dte.expiry_date] = (
                    held_count_by_exp.get(matched_dte.expiry_date, 0) + n_syms
                )
                if matched_dte.expiry_date not in monitored_seen:
                    monitored_seen.add(matched_dte.expiry_date)
                    monitored_exps.append(matched_dte)

    if config.mode != "LIVE_MORNING_SNAPSHOT":
        if config.evaluate_all_active_expiries:
            base_exps = [e for e in active_list if e.dte_trading_sessions >= 1] or [anchor_expiry]
        else:
            base_exps = [anchor_expiry]
        ordered_off = [anchor_expiry] + [
            e for e in base_exps if e.expiry_date != anchor_expiry.expiry_date
        ]
        seen_off = {e.expiry_date for e in ordered_off}
        for m_exp in monitored_exps:
            if m_exp.expiry_date not in seen_off:
                ordered_off.append(m_exp)
                seen_off.add(m_exp.expiry_date)
        return BoundedExpiryPlan(
            anchor_expiry=anchor_expiry,
            required_expiries=tuple(ordered_off),
            selected_expiries=tuple(ordered_off),
            unlisted_required_expiries=tuple(unlisted_req),
            token_budget_by_expiry={},
            remaining_request_budget=999,
            single_expiry_budget_limited=False,
        )

    # LIVE_MORNING_SNAPSHOT bounded expiry selection
    req_list: list[ExpiryMetadata] = [anchor_expiry]
    req_dates: set[str] = {anchor_expiry.expiry_date}
    for m_exp in monitored_exps:
        if m_exp.expiry_date not in req_dates:
            req_list.append(m_exp)
            req_dates.add(m_exp.expiry_date)

    optional_list: list[ExpiryMetadata] = []
    if config.evaluate_all_active_expiries:
        next_comp = next(
            (
                e
                for e in active_list
                if e.dte_trading_sessions >= 1 and e.expiry_date != anchor_expiry.expiry_date
            ),
            None,
        )
        if next_comp is not None and next_comp.expiry_date not in req_dates:
            if len(req_list) < 2:
                req_list.append(next_comp)
                req_dates.add(next_comp.expiry_date)
            else:
                optional_list.append(next_comp)

    limiter = getattr(broker_adapter, "rate_limiter", None)
    has_live_paced_cap = (
        isinstance(limiter, PacedReadTransport)
        and limiter.min_interval_seconds > 0.0
        and limiter.max_queue_wait_seconds > 0.0
    )
    if has_live_paced_cap:
        eff_interval = limiter.min_interval_seconds + getattr(limiter, "strict_guard_seconds", 0.0)
        if limiter.request_count == 0 and getattr(limiter, "_last_request_ts", None) is None:
            rem_requests = 1 + int(
                math.floor((limiter.max_queue_wait_seconds + 1e-9) / eff_interval)
            )
        else:
            rem_wait = max(0.0, limiter.max_queue_wait_seconds - limiter.cumulative_wait_seconds)
            rem_requests = int(math.floor((rem_wait + 1e-9) / eff_interval))
    elif (
        config.kotak_min_request_interval_seconds > 0.0
        and config.kotak_max_queue_wait_seconds > 0.0
    ):
        total_cold_slots = 1 + int(
            math.floor(
                (config.kotak_max_queue_wait_seconds + 1e-9)
                / config.kotak_min_request_interval_seconds
            )
        )
        rem_requests = max(1, total_cold_slots - 7)
    else:
        rem_requests = 999

    batch_size = max(
        1,
        min(
            50,
            int(getattr(broker_adapter, "quotes_batch_size", 0) or config.kotak_quotes_batch_size),
        ),
    )
    default_max_tokens = max(
        6,
        int(
            getattr(broker_adapter, "max_depth_tokens_per_expiry", 0)
            or config.kotak_max_depth_tokens_per_expiry
        ),
    )

    def _min_tok_for(exp_m: ExpiryMetadata, is_anchor_exp: bool) -> int:
        held_n = held_count_by_exp.get(exp_m.expiry_date, 0)
        parity_plus_held = max(6, held_n + 6)
        if is_anchor_exp:
            # Keep directional wings on the primary anchor chain (>= 12 or default_max_tokens)
            return max(parity_plus_held, min(default_max_tokens, 12))
        return parity_plus_held

    def _pref_tok_for(exp_m: ExpiryMetadata, is_anchor_exp: bool) -> int:
        if is_anchor_exp:
            return max(default_max_tokens, _min_tok_for(exp_m, True))
        return max(_min_tok_for(exp_m, False), min(default_max_tokens, 8))

    def _calls_for(num_tokens: int) -> int:
        return 1 + int(math.ceil(num_tokens / float(batch_size)))

    token_budgets: dict[str, int] = {}
    selected: list[ExpiryMetadata] = []
    single_budget_limited = False

    pref_costs = [_calls_for(_pref_tok_for(e, i == 0)) for i, e in enumerate(req_list)]
    min_costs = [_calls_for(_min_tok_for(e, i == 0)) for i, e in enumerate(req_list)]

    if sum(pref_costs) <= rem_requests:
        selected = list(req_list)
        for i, e in enumerate(req_list):
            token_budgets[e.expiry_date] = _pref_tok_for(e, i == 0)
        used_calls = sum(pref_costs)
    elif sum(min_costs) <= rem_requests:
        selected = list(req_list)
        for i, e in enumerate(req_list):
            token_budgets[e.expiry_date] = _min_tok_for(e, i == 0)
        used_calls = sum(min_costs)
    else:
        # Cannot fit all required expiries simultaneously; fit greedily in priority order
        if len(req_list) >= 2 and (min_costs[0] + min_costs[1]) > rem_requests:
            # Even 2 expiries cannot fit -> single-expiry budget degradation (e.g. batch_size <= 2)
            selected = [anchor_expiry]
            single_budget_limited = True
            if has_live_paced_cap:
                max_fit_tok = max(6, (max(2, rem_requests) - 1) * batch_size)
                token_budgets[anchor_expiry.expiry_date] = min(default_max_tokens, max_fit_tok)
            else:
                token_budgets[anchor_expiry.expiry_date] = default_max_tokens
            used_calls = _calls_for(token_budgets[anchor_expiry.expiry_date])
        else:
            used_calls = 0
            for i, e in enumerate(req_list):
                tok_i = _min_tok_for(e, i == 0)
                c_i = _calls_for(tok_i)
                if i == 0 or (used_calls + c_i) <= rem_requests:
                    selected.append(e)
                    token_budgets[e.expiry_date] = tok_i
                    used_calls += c_i

    # Include optional comparison expiry when all required expiries fitted with spare budget
    if len(selected) == len(req_list) and optional_list:
        for e_opt in optional_list:
            tok_opt = _pref_tok_for(e_opt, False)
            c_opt = _calls_for(tok_opt)
            if (used_calls + c_opt) <= rem_requests:
                selected.append(e_opt)
                token_budgets[e_opt.expiry_date] = tok_opt
                used_calls += c_opt

    return BoundedExpiryPlan(
        anchor_expiry=anchor_expiry,
        required_expiries=tuple(req_list),
        selected_expiries=tuple(selected),
        unlisted_required_expiries=tuple(unlisted_req),
        token_budget_by_expiry=token_budgets,
        remaining_request_budget=rem_requests,
        single_expiry_budget_limited=single_budget_limited,
    )


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
    if cand is None or not cand.full_size_fill_supported or not config.payoff_chart_dir:
        return ""
    try:
        from nifty_analyst.output.payoff_diagram import render_candidate_payoff_diagram
    except ImportError:
        return ""

    out_dir = Path(config.payoff_chart_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{run_id}_{cand.structure_id}.png"
    try:
        render_candidate_payoff_diagram(
            cand=cand,
            out_path=str(out_file),
            slippage_buffer_points_per_leg=config.slippage_buffer_points_per_leg,
        )
        return str(out_file)
    except ModuleNotFoundError:
        return ""


def run_analyst_pipeline(
    decision_dt_ist: dt.datetime,
    config: AnalystConfig,
    broker_adapter: KotakNeoReadOnlyPort | None = None,
    llm_provider: LLMProvider | None = None,
    open_positions: Sequence[OpenPosition] | None = None,
    target_expiry_override: str | None = None,
    force_render_for_inspection: bool = False,
    suppress_trader_alerts: bool = False,
    wall_clock_fn: Callable[[], dt.datetime] | None = None,
) -> tuple[NiftyOptionsState, AuditDeliveryResult]:
    """Execute the full NIFTY 50 Options Research Analyst pipeline (single-pass, zero multicall)."""
    dec_ist = ensure_ist(decision_dt_ist)
    clock_fn = wall_clock_fn or (lambda: dt.datetime.now(tz=IST))
    scrubber = config.build_scrubber()
    registry_root = _resolve_snapshot_registry_root(config)

    # -------------------------------------------------------------------------
    # ITEM 2: RESUME MODE (zero broker reads, immutable snapshot & 180s check)
    # -------------------------------------------------------------------------
    resuming_from_snapshot = (
        llm_provider is None
        and config.llm_provider == "agent_artifact"
        and bool(config.agent_synthesis_artifact_path)
        and has_existing_snapshot_for_artifact(config.agent_synthesis_artifact_path, registry_root)
    )
    if resuming_from_snapshot:
        resume_now_dt = clock_fn()
        loaded = load_and_verify_snapshot_for_resume(
            artifact_path=config.agent_synthesis_artifact_path,
            resume_now_dt=resume_now_dt,
            expected_decision_timestamp_ist=dec_ist.isoformat(),
            expected_config_hash=config.config_hash(),
            max_age_seconds=float(config.max_quote_staleness_seconds),
            registry_root=registry_root,
        )
        pre_llm_state = loaded.pre_llm_state
        meta = pre_llm_state.run_metadata
        run_id = meta.run_id
        contract_ctx = pre_llm_state.contract_context
        portfolio_state = pre_llm_state.portfolio_exposure_state
        vol_evidence = pre_llm_state.market_evidence.volatility_state
        pos_evidence = pre_llm_state.market_evidence.derivatives_positioning
        event_cal = pre_llm_state.market_evidence.event_calendar
        detectors = pre_llm_state.market_evidence.triggered_detectors
        candidates = pre_llm_state.candidate_structures
        chain_provenance = loaded.chain_provenance
        all_rejected_books: list[dict[str, Any]] = []
        earliest_q = (
            min(str(c["earliest_quote_ist"]) for c in chain_provenance)
            if chain_provenance
            else contract_ctx.spot.as_of_ist
        )
        latest_q = (
            max(str(c["latest_quote_ist"]) for c in chain_provenance)
            if chain_provenance
            else contract_ctx.spot.as_of_ist
        )
        evaluated_chains_count = max(1, len(chain_provenance))
        any_fetch_time_unverified = any(
            any(tok in str(c.get("source_verification_tag", "")) for tok in _UNVERIFIED_QUOTE_TS_TAGS)
            for c in chain_provenance
        )
        intraday_ctx = load_same_day_intraday_context(
            audit_log_path=config.audit_log_path,
            decision_dt_ist=dec_ist,
            current_spot=contract_ctx.spot.value,
            current_cm30_iv=vol_evidence.constant_maturity_30d_iv,
            current_pcr_oi=pos_evidence.pcr_oi,
        )
    else:
        # ---------------------------------------------------------------------
        # INITIAL CAPTURE MODE: stamp actual capture_started_at_ist FIRST
        # ---------------------------------------------------------------------
        capture_started_raw = clock_fn()
        if capture_started_raw.tzinfo is None or capture_started_raw.utcoffset() is None:
            parse_strict_ist_timestamp("", "capture_started_at_ist")
        capture_started_dt = capture_started_raw.astimezone(IST)
        run_id = f"run_{dec_ist.strftime('%Y%m%d_%H%M%S')}_{config.config_hash()[:6]}"

        if open_positions is not None:
            portfolio_state = build_portfolio_exposure_state(
                open_positions,
                portfolio_cap_rs=config.portfolio_summed_stop_cap_rs,
            )
        else:
            pos_load = load_open_positions(
                config.resolved_open_positions_path(),
                fail_closed_if_unconfigured=(config.mode == "LIVE_MORNING_SNAPSHOT"),
            )
            portfolio_state = build_portfolio_exposure_state(
                pos_load.positions,
                portfolio_cap_rs=config.portfolio_summed_stop_cap_rs,
                portfolio_state_status=pos_load.status,
                portfolio_state_error=pos_load.error,
            )

        held_neo_symbols = [
            ns for pos in portfolio_state.open_positions for ns in pos.neo_symbols
        ]

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
                    shared_state_dir = (
                        Path(config.cache_dir)
                        if config.cache_dir
                        else (Path.home() / "nifty-analyst")
                    )
                    cooldown_path = shared_state_dir / "kotak_429_cooldown.json"
                    pacing_state_path = shared_state_dir / "kotak_pacing_state.json"
                    live_limiter = PacedReadTransport(
                        min_interval_seconds=config.kotak_min_request_interval_seconds,
                        max_queue_wait_seconds=config.kotak_max_queue_wait_seconds,
                        cooldown_file_path=cooldown_path,
                        pacing_state_file_path=pacing_state_path,
                    )
                    trading_cal = {
                        "holiday_dates": frozenset(d.isoformat() for d in holidays),
                        "special_sessions": config.special_sessions,
                    }
                    broker_adapter = LiveKotakNeoAdapter(
                        kotak_api_root=config.kotak_api_root or None,
                        rate_limiter=live_limiter,
                        enforce_schedule_in_live_mode=True,
                        depth_strikes_each_side=config.kotak_chain_strike_count,
                        quotes_batch_size=config.kotak_quotes_batch_size,
                        max_depth_tokens_per_expiry=config.kotak_max_depth_tokens_per_expiry,
                        extra_neo_symbols=held_neo_symbols,
                        cache_dir=config.cache_dir or None,
                        trading_calendar=trading_cal,  # type: ignore[arg-type]
                    )
            elif held_neo_symbols and hasattr(broker_adapter, "extra_neo_symbols"):
                existing_extra = list(getattr(broker_adapter, "extra_neo_symbols") or ())
                broker_adapter.extra_neo_symbols = tuple(  # type: ignore[attr-defined]
                    dict.fromkeys([*existing_extra, *held_neo_symbols])
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
                suppress_trader_alerts=suppress_trader_alerts,
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
                suppress_trader_alerts=suppress_trader_alerts,
            )
            return state, delivery

        evaluated_chains: list[
            tuple[ExpiryMetadata, ChainSnapshotEnvelope, ChainQAResult, float, float]
        ] = []
        all_rejected_books = []
        per_expiry_coverage: dict[str, dict[str, Any]] = {}
        live_expiry_lots: dict[str, int] = {}
        spot_val = contract_ctx.spot.value

        try:
            raw_candles = broker_adapter.fetch_index_candles_raw(dec_ist)
            completed_1d, completed_1h, completed_15m, stripped_bars = slice_completed_candles(
                raw_candles, dec_ist
            )
            vix_datum = broker_adapter.fetch_india_vix_raw(dec_ist)

            expiry_plan = select_bounded_expiries_and_budgets(
                active_expiries=contract_ctx.active_expiries,
                open_positions=portfolio_state.open_positions,
                config=config,
                broker_adapter=broker_adapter,
                target_expiry_override=target_expiry_override,
            )
            expiries_ordered = list(expiry_plan.selected_expiries)
            fetch_chain_params = inspect.signature(
                broker_adapter.fetch_option_chain_raw
            ).parameters
            supports_token_override = "max_depth_tokens_override" in fetch_chain_params

            for idx_exp, exp_meta in enumerate(expiries_ordered):
                is_anchor = idx_exp == 0
                tok_override = expiry_plan.token_budget_by_expiry.get(exp_meta.expiry_date)
                try:
                    if supports_token_override and tok_override is not None:
                        raw_chain = broker_adapter.fetch_option_chain_raw(  # type: ignore[call-arg]
                            exp_meta.expiry_date,
                            dec_ist,
                            max_depth_tokens_override=tok_override,
                        )
                    else:
                        raw_chain = broker_adapter.fetch_option_chain_raw(
                            exp_meta.expiry_date, dec_ist
                        )
                    chain_env = parse_and_validate_option_chain(
                        raw_chain=raw_chain,
                        expiry_date=exp_meta.expiry_date,
                        decision_dt_ist=dec_ist,
                        max_staleness_seconds=config.max_quote_staleness_seconds,
                    )
                except (KotakRateLimit429Error, KotakSharedStateError):
                    # HTTP 429 or shared-safety-state failure MUST hard-stop the entire run immediately
                    raise
                except KotakAdapterError as exc:
                    per_expiry_coverage[exp_meta.expiry_date] = {
                        "expiry_date": exp_meta.expiry_date,
                        "status": f"FETCH_FAILED:{exc}",
                        **{k: [] for k in _COVERAGE_BUCKET_KEYS},
                        "counts": {k: 0 for k in _COVERAGE_BUCKET_KEYS},
                    }
                    if is_anchor:
                        raise
                    continue

                if chain_env.coverage_ledger:
                    per_expiry_coverage[exp_meta.expiry_date] = dict(chain_env.coverage_ledger)

                if chain_env.rejected_depth_books:
                    all_rejected_books.extend(chain_env.rejected_depth_books)
                if chain_env.lot_size is not None and chain_env.lot_size > 0:
                    live_expiry_lots[exp_meta.expiry_date] = chain_env.lot_size
                    if chain_env.lot_size != exp_meta.lot_size:
                        exp_meta = dataclasses.replace(exp_meta, lot_size=chain_env.lot_size)

                if chain_env.any_stale:
                    if exp_meta.expiry_date in per_expiry_coverage:
                        per_expiry_coverage[exp_meta.expiry_date]["status"] = "STALE_QUOTES_HALT"
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
                            suppress_trader_alerts=suppress_trader_alerts,
                            extra_audit_metadata=(
                                {"rejected_depth_books": all_rejected_books}
                                if all_rejected_books
                                else None
                            ),
                        )
                        return state, delivery
                    continue

                t_yrs = compute_time_to_expiry_years(dec_ist, exp_meta.expiry_date)
                qa_res = run_chain_qa_suite(
                    spot=spot_val,
                    quotes=chain_env.quotes,
                    time_to_expiry_years=t_yrs,
                    risk_free_rate=config.risk_free_rate_annual,
                    max_leg_timestamp_skew_seconds=config.max_leg_timestamp_skew_seconds,
                )
                if not qa_res.passed:
                    if exp_meta.expiry_date in per_expiry_coverage:
                        per_expiry_coverage[exp_meta.expiry_date]["status"] = (
                            f"CHAIN_QA_FAILED:{qa_res.failure_code}"
                        )
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
                            suppress_trader_alerts=suppress_trader_alerts,
                            extra_audit_metadata=(
                                {"rejected_depth_books": all_rejected_books}
                                if all_rejected_books
                                else None
                            ),
                        )
                        return state, delivery
                    continue

                if exp_meta.expiry_date in per_expiry_coverage:
                    per_expiry_coverage[exp_meta.expiry_date]["status"] = "PRICED"
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
                suppress_trader_alerts=suppress_trader_alerts,
                extra_audit_metadata=(
                    {"rejected_depth_books": all_rejected_books} if all_rejected_books else None
                ),
            )
            return state, delivery

        if live_expiry_lots:
            updated_active = [
                dataclasses.replace(e, lot_size=live_expiry_lots[e.expiry_date])
                if e.expiry_date in live_expiry_lots and live_expiry_lots[e.expiry_date] != e.lot_size
                else e
                for e in contract_ctx.active_expiries
            ]
            contract_ctx = dataclasses.replace(
                contract_ctx,
                active_expiries=updated_active,
                lot_size_change_detected=len({e.lot_size for e in updated_active}) > 1,
            )

        # Item 3 & 5: Explicit 7-bucket coverage ledger & truthful surface-comparison certification
        priced_expiry_dates = [m.expiry_date for (m, _, _, _, _) in evaluated_chains]
        priced_expiry_set = set(priced_expiry_dates)
        required_expiry_dates = list(
            dict.fromkeys(
                [e.expiry_date for e in expiry_plan.required_expiries]
                + list(expiry_plan.unlisted_required_expiries)
            )
        )
        missing_required_expiries = [
            d for d in required_expiry_dates if d not in priced_expiry_set
        ]
        covered_priced_neos = {
            ns for (_, env, _, _, _) in evaluated_chains for ns in env.priced_neo_symbols
        }
        uncovered_held_neos = sorted(
            {ns for ns in held_neo_symbols if ns and ns not in covered_priced_neos}
        )
        any_missing_requested_quotes = any(
            bool(env.coverage_ledger.get("missing")) for (_, env, _, _, _) in evaluated_chains
        )

        totals_counts: dict[str, int] = {k: 0 for k in _COVERAGE_BUCKET_KEYS}
        for cov_entry in per_expiry_coverage.values():
            c_map = cov_entry.get("counts") or {}
            for k in _COVERAGE_BUCKET_KEYS:
                totals_counts[k] += int(c_map.get(k, 0) or 0)

        run_coverage_ledger: dict[str, Any] = {
            "listed_expiries": [e.expiry_date for e in contract_ctx.active_expiries],
            "required_expiries": required_expiry_dates,
            "selected_expiries": [e.expiry_date for e in expiry_plan.selected_expiries],
            "priced_expiries": priced_expiry_dates,
            "unlisted_required_expiries": list(expiry_plan.unlisted_required_expiries),
            "missing_required_expiries": missing_required_expiries,
            "required_monitored_contracts": sorted({ns for ns in held_neo_symbols if ns}),
            "priced_monitored_contracts": sorted(
                {ns for ns in held_neo_symbols if ns and ns in covered_priced_neos}
            ),
            "uncovered_monitored_contracts": uncovered_held_neos,
            "by_expiry": per_expiry_coverage,
            "totals": totals_counts,
            "transport_counts": {
                "adapter_call_count": int(getattr(broker_adapter, "adapter_call_count", 0) or 0),
                "cache_hit_count": int(getattr(broker_adapter, "cache_hit_count", 0) or 0),
                "http_request_count": int(getattr(broker_adapter, "http_request_count", 0) or 0),
            },
        }

        surface_coverage_status: Literal[
            "MULTI_EXPIRY_FULL", "SINGLE_EXPIRY_DEGRADED", "INCOMPLETE_SURFACE_COVERAGE"
        ]
        if (
            expiry_plan.single_expiry_budget_limited
            and len(contract_ctx.active_expiries) > 1
            and len(evaluated_chains) == 1
        ):
            surface_coverage_status = "SINGLE_EXPIRY_DEGRADED"
            surface_coverage_detail = (
                f"Evaluated anchor expiry {evaluated_chains[0][0].expiry_date} only out of "
                f"{len(contract_ctx.active_expiries)} listed expiries under small-batch request budget; "
                "multi-expiry surface comparison is NOT certified."
            )
        elif (
            missing_required_expiries
            or uncovered_held_neos
            or any_missing_requested_quotes
            or (
                config.evaluate_all_active_expiries
                and not target_expiry_override
                and len(contract_ctx.active_expiries) > 1
                and len(evaluated_chains) < 2
            )
        ):
            surface_coverage_status = "INCOMPLETE_SURFACE_COVERAGE"
            inc_parts: list[str] = []
            if missing_required_expiries:
                inc_parts.append(f"unpriced_required_expiries={missing_required_expiries}")
            if uncovered_held_neos:
                inc_parts.append(f"unpriced_required_contracts={uncovered_held_neos}")
            if any_missing_requested_quotes:
                missing_syms = sorted(
                    {
                        ns
                        for (_, env, _, _, _) in evaluated_chains
                        for ns in (env.coverage_ledger.get("missing") or [])
                    }
                )
                inc_parts.append(f"missing_requested_quotes={missing_syms}")
            if not inc_parts:
                inc_parts.append(
                    f"priced_expiries={len(evaluated_chains)}/{len(contract_ctx.active_expiries)}"
                )
            surface_coverage_detail = (
                "INCOMPLETE_SURFACE_COVERAGE (" + "; ".join(inc_parts) + "); "
                "multi-expiry surface comparison is NOT certified."
            )
        else:
            surface_coverage_status = "MULTI_EXPIRY_FULL"
            surface_coverage_detail = (
                f"Priced {len(evaluated_chains)} expiry(s) {priced_expiry_dates} "
                f"(listed={len(contract_ctx.active_expiries)})."
            )

        anchor_meta, anchor_chain_env, anchor_qa, anchor_t_yrs, anchor_atm_iv = evaluated_chains[0]
        atm_iv_by_expiry = {m.expiry_date: iv for (m, _, _, _, iv) in evaluated_chains}
        if surface_coverage_status == "MULTI_EXPIRY_FULL" and len(evaluated_chains) >= 2:
            t_and_ivs = [(t_y, iv) for (_, _, _, t_y, iv) in evaluated_chains]
            cm30_iv, ts_slope, ts_regime = compute_variance_interpolated_cm30_iv(
                t_and_ivs, target_days=30.0
            )
            iv_source_lbl = "LIVE_VARIANCE_INTERPOLATED_CM30"
        else:
            # Never certify multi-expiry CM30 / term-structure regime when required expiries or legs are unpriced
            cm30_iv, ts_slope, ts_regime = compute_variance_interpolated_cm30_iv(
                [(anchor_t_yrs, anchor_atm_iv)], target_days=30.0
            )
            iv_source_lbl = (
                "UNCERTIFIED_INCOMPLETE_SURFACE"
                if surface_coverage_status == "INCOMPLETE_SURFACE_COVERAGE"
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

        qa_notes_combined = list(anchor_qa.notes)
        if all_rejected_books:
            qa_notes_combined.append(
                f"Rejected {len(all_rejected_books)} non-executable depth book(s) to None/UNKNOWN; "
                "raw payloads logged in audit record."
            )
        if surface_coverage_status == "SINGLE_EXPIRY_DEGRADED":
            qa_notes_combined.append(
                f"SINGLE_EXPIRY_DEGRADED: evaluated anchor expiry {anchor_meta.expiry_date} only "
                f"out of {len(contract_ctx.active_expiries)} active expiries (multi-expiry surface "
                f"interpolation inactive under small-batch pacing cap)."
            )
        elif surface_coverage_status == "INCOMPLETE_SURFACE_COVERAGE":
            qa_notes_combined.append(surface_coverage_detail)
        if uncovered_held_neos:
            qa_notes_combined.append(
                f"UNCOVERED_HELD_CONTRACTS: open-position symbol(s) {uncovered_held_neos} "
                "belong to non-evaluated expiries or lacked depth quotes on this run."
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
            chain_qa_notes=qa_notes_combined,
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

        candidates = []
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

        earliest_q = min(env.earliest_quote_ist for (_, env, _, _, _) in evaluated_chains)
        latest_q = max(env.latest_quote_ist for (_, env, _, _, _) in evaluated_chains)
        evaluated_chains_count = len(evaluated_chains)
        any_fetch_time_unverified = any(
            any(tok in env.source_verification_tag for tok in _UNVERIFIED_QUOTE_TS_TAGS)
            for (_, env, _, _, _) in evaluated_chains
        )
        chain_provenance = [
            {
                "expiry_date": m.expiry_date,
                "earliest_quote_ist": env.earliest_quote_ist,
                "latest_quote_ist": env.latest_quote_ist,
                "fetch_started_at_ist": env.fetch_started_at_ist,
                "fetch_completed_at_ist": env.fetch_completed_at_ist,
                "source_verification_tag": env.source_verification_tag,
                "lot_size": m.lot_size,
            }
            for (m, env, _, _, _) in evaluated_chains
        ]

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

        capture_completed_raw = clock_fn()
        if capture_completed_raw.tzinfo is None or capture_completed_raw.utcoffset() is None:
            parse_strict_ist_timestamp("", "capture_completed_at_ist")
        capture_completed_dt = capture_completed_raw.astimezone(IST)
        validate_capture_and_resume_clocks(
            capture_started_at_ist=capture_started_dt.isoformat(),
            capture_completed_at_ist=capture_completed_dt.isoformat(),
            resume_now_dt=capture_completed_dt,
            max_age_seconds=float(config.max_quote_staleness_seconds),
        )

        slot_approved = is_approved_schedule_slot(
            dec_ist,
            holiday_dates=frozenset(d.isoformat() for d in holidays),
            special_sessions=config.special_sessions,
        )
        provisional_meta = RunMetadata(
            run_id=run_id,
            decision_timestamp_ist=dec_ist.isoformat(),
            mode=config.mode,
            data_version_hash=data_version_hash,
            model_ids=[config.llm_model_id],
            config_hash=config.config_hash(),
            scheduled_slot_approved=slot_approved,
            capture_started_at_ist=capture_started_dt.isoformat(),
            capture_completed_at_ist=capture_completed_dt.isoformat(),
            surface_coverage_status=surface_coverage_status,
            surface_coverage_detail=surface_coverage_detail,
            uncovered_held_neo_symbols=uncovered_held_neos,
            coverage_ledger=run_coverage_ledger,
        )
        provisional_state = NiftyOptionsState(
            run_metadata=provisional_meta,
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
        _, ev_sha256 = compute_state_evidence_sha256(
            provisional_state, scrubber=scrubber, chain_provenance=chain_provenance
        )
        run_id, snapshot_id = mint_unique_capture_identities(
            decision_dt_ist=dec_ist,
            capture_started_dt_ist=capture_started_dt,
            config_hash=config.config_hash(),
            evidence_sha256=ev_sha256,
        )
        meta = dataclasses.replace(
            provisional_meta,
            run_id=run_id,
            snapshot_id=snapshot_id,
            evidence_sha256=ev_sha256,
        )
        pre_llm_state = dataclasses.replace(provisional_state, run_metadata=meta)

        if (
            llm_provider is None
            and config.llm_provider == "agent_artifact"
            and config.agent_synthesis_artifact_path
        ):
            sanitised_state_obj = scrubber.scrub_obj(pre_llm_state.to_dict())
            desk_brief = build_compact_desk_brief(pre_llm_state)
            req_payload = {
                "run_id": run_id,
                "snapshot_id": snapshot_id,
                "evidence_sha256": ev_sha256,
                "decision_timestamp_ist": meta.decision_timestamp_ist,
                "capture_started_at_ist": meta.capture_started_at_ist,
                "capture_completed_at_ist": meta.capture_completed_at_ist,
                "max_resume_age_seconds": float(config.max_quote_staleness_seconds),
                "compact_desk_brief": desk_brief,
                "persona_instructions": PERSONA_DEFINITIONS,
                "single_pass_prompt": build_single_pass_persona_prompt(desk_brief),
                "sanitised_state": sanitised_state_obj,
            }
            persist_immutable_snapshot_bundle(
                artifact_path=config.agent_synthesis_artifact_path,
                state=pre_llm_state,
                request_payload=req_payload,
                scrubber=scrubber,
                chain_provenance=chain_provenance,
                registry_root=registry_root,
            )

    # -------------------------------------------------------------------------
    # LAYER 3 & 4: SINGLE-PASS SYNTHESIS, STRICT VERIFICATION & BOUND DELIVERY
    # -------------------------------------------------------------------------
    independent_regime = assess_market_regime(pre_llm_state)
    primary_dir = (
        independent_regime
        if independent_regime in ("BULLISH", "BEARISH", "NEUTRAL")
        else None
    )

    eligible = [
        c
        for c in candidates
        if is_candidate_compatible_with_regime(c, independent_regime, pre_llm_state)
    ]
    eligible.sort(
        key=lambda c: _rank_candidate(c, vol_evidence.ivp_90d, primary_dir),
        reverse=True,
    )
    eligible_ids = {c.structure_id for c in eligible}
    cand_by_id = {c.structure_id: c for c in candidates}

    default_primary: CandidateStructure | None = (
        eligible[0] if (eligible and independent_regime != "CONTESTED_WAIT") else None
    )
    default_alternatives = [
        c
        for c in candidates
        if default_primary is None or c.structure_id != default_primary.structure_id
    ]

    provider = llm_provider or build_llm_provider(
        config,
        wall_clock_fn=clock_fn,
        preloaded_bundle=(loaded if resuming_from_snapshot else None),
    )
    sanitised_json = json.dumps(scrubber.scrub_obj(pre_llm_state.to_dict()), indent=2)
    llm_resp = provider.synthesize(
        sanitised_state_json=sanitised_json,
        state=pre_llm_state,
        primary=default_primary,
        alternatives=default_alternatives,
    )

    # ITEM 1: Determine final candidate or WAIT strictly FIRST (never fallback to eligible[0] on bad ID)
    consolidator_rec_id: str | None = None
    selected_primary: CandidateStructure | None = default_primary
    if llm_resp.consolidator_state is not None:
        consolidator_rec_id = llm_resp.consolidator_state.recommended_structure_id
        if consolidator_rec_id == "WAIT":
            selected_primary = None
        else:
            rec_cand = cand_by_id.get(consolidator_rec_id)
            if rec_cand is not None and rec_cand.structure_id in eligible_ids:
                selected_primary = rec_cand
            else:
                selected_primary = None

    portfolio_unknown = portfolio_state.portfolio_state_status != "VERIFIED"
    ivp_unknown = vol_evidence.ivp_30d == "UNKNOWN" or vol_evidence.ivp_90d == "UNKNOWN"
    live_gates_unverified = config.mode == "LIVE_MORNING_SNAPSHOT"
    uncovered_held = bool(meta.uncovered_held_neo_symbols)
    surface_incomplete = meta.surface_coverage_status != "MULTI_EXPIRY_FULL"

    prelim_is_live_ready = (
        selected_primary is not None
        and selected_primary.structure_id in eligible_ids
        and not portfolio_unknown
        and not ivp_unknown
        and not any_fetch_time_unverified
        and not uncovered_held
        and not surface_incomplete
        and not live_gates_unverified
    )

    # Collect ALL delivered prose fields + supplied whatsapp_digest for candidate-scoped verification
    prose_fields_map: dict[str, str] = {
        "reasoning_summary": llm_resp.reasoning_summary,
        "habitual_comparison_note": llm_resp.habitual_comparison_note,
        "research_rule_challenge_note": llm_resp.research_rule_challenge_note,
        "confidence_note": llm_resp.confidence_note,
    }
    if llm_resp.debate_state is not None:
        prose_fields_map["bull_thesis"] = llm_resp.debate_state.bull_thesis
        prose_fields_map["bear_thesis"] = llm_resp.debate_state.bear_thesis
        prose_fields_map["vol_regime_thesis"] = llm_resp.debate_state.vol_regime_thesis
        prose_fields_map["rebuttal_round"] = llm_resp.debate_state.rebuttal_round
    if llm_resp.risk_triad_state is not None:
        prose_fields_map["convexity_advocate_note"] = (
            llm_resp.risk_triad_state.convexity_advocate_note
        )
        prose_fields_map["tail_guardian_note"] = llm_resp.risk_triad_state.tail_guardian_note
        prose_fields_map["neutral_arbiter_note"] = llm_resp.risk_triad_state.neutral_arbiter_note
    supplied_digest: str | None = None
    if llm_resp.consolidator_state is not None:
        prose_fields_map["final_recommendation_rationale"] = (
            llm_resp.consolidator_state.final_recommendation_rationale
        )
        supplied_digest = llm_resp.consolidator_state.whatsapp_digest

    verified_citations, verifier_report = verify_llm_synthesis(
        state=pre_llm_state,
        raw_citations=llm_resp.citations,
        selected_primary=selected_primary,
        consolidator_recommended_id=consolidator_rec_id,
        prose_fields=prose_fields_map,
        supplied_whatsapp_digest=supplied_digest,
        expected_is_live_ready=prelim_is_live_ready,
        debate_dominant_regime=(
            llm_resp.debate_state.dominant_regime if llm_resp.debate_state is not None else None
        ),
    )

    if (
        selected_primary is not None
        and verifier_report.passed_verification
        and selected_primary.structure_id in eligible_ids
    ):
        verdict: Literal["SUGGESTION", "WAIT"] = "SUGGESTION"
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
        and prelim_is_live_ready
        and selected_primary is not None
        and selected_primary.part_a_overall_status == "PASS"
        and selected_primary.full_size_fill_supported
    )

    research_only_reason = ""
    if not is_live_ready:
        reasons: list[str] = []
        if portfolio_unknown:
            reasons.append(f"PORTFOLIO_STATE_UNKNOWN: {portfolio_state.portfolio_state_error}")
        if uncovered_held:
            reasons.append(
                f"UNCOVERED_HELD_CONTRACTS: {meta.uncovered_held_neo_symbols} not depth-covered on evaluated expiry(s)."
            )
        if surface_incomplete:
            reasons.append(
                f"{meta.surface_coverage_status}: {meta.surface_coverage_detail or 'multi-expiry surface incomplete'}."
            )
        if ivp_unknown:
            reasons.append(
                "IVP is UNKNOWN; output is labelled research-only per Section 4.2 & 5A.8."
            )
        if any_fetch_time_unverified:
            reasons.append(
                "Option chain quote timestamps use unverified semantics (`FETCH_TIME_NOT_EXCHANGE` / `UNVERIFIED_KOTAK_REST_LSTUP_TIME`)."
            )
        if live_gates_unverified:
            reasons.append(
                "LIVE_GATES_UNVERIFIED: exchange quote timestamps, multi-expiry/held-contract coverage, "
                "and HTTP-boundary 2.5s pacing inside Kotak_API wrappers remain unverified."
            )
        if not verifier_report.passed_verification:
            reasons.append(
                "Post-synthesis deterministic verifier rejected prose/digest/citation claims (`WAIT`)."
            )
        elif verdict != "SUGGESTION":
            reasons.append(f"Verdict is {verdict}.")
        elif selected_primary is not None and selected_primary.part_a_overall_status != "PASS":
            reasons.append(
                f"Primary candidate Part A status is {selected_primary.part_a_overall_status}."
            )
        research_only_reason = " | ".join(reasons)

    duplicate_suppressed = bool(
        config.deduplicate_intraday_alerts
        and verdict == "SUGGESTION"
        and is_live_ready
        and selected_primary is not None
        and selected_primary.structure_id in intraday_ctx.alerted_structure_ids_today
    )

    # ITEM 1: Generate ALL report & persona text deterministically from the final verified state
    # across every report, state field, digest, and audit log (never copy free-form LLM prose).
    narrative = build_verified_report_narrative(
        state=pre_llm_state,
        selected_primary=selected_primary,
        eligible_candidates=eligible,
        verdict=verdict,
        is_live_ready=is_live_ready,
        verifier_passed=verifier_report.passed_verification,
        research_only_reason=research_only_reason,
    )
    final_debate = narrative.debate_state
    final_triad = narrative.risk_triad_state
    final_consolidator: ConsolidatorState | None = narrative.consolidator_state
    final_reasoning = narrative.reasoning_summary
    final_habitual = narrative.habitual_comparison_note
    final_challenge = narrative.research_rule_challenge_note
    final_confidence = narrative.confidence_note
    whatsapp_digest = narrative.whatsapp_digest

    chart_path = _maybe_render_payoff_chart(selected_primary, run_id, config)

    # Re-verify the 180-second capture-start deadline immediately before emitting final output
    # so a run started at e.g. 179s that finishes at 190s fails closed before delivery.
    try:
        final_now_dt = clock_fn()
        validate_capture_and_resume_clocks(
            capture_started_at_ist=meta.capture_started_at_ist,
            capture_completed_at_ist=meta.capture_completed_at_ist,
            resume_now_dt=final_now_dt,
            max_age_seconds=float(config.max_quote_staleness_seconds),
        )
    except (StaleEvidenceSnapshotError, EvidenceSnapshotClockError):
        if chart_path:
            Path(chart_path).unlink(missing_ok=True)
        raise

    staleness_notes = [
        f"Spot as-of {contract_ctx.spot.as_of_ist} (stale={contract_ctx.spot.is_stale})",
        f"Chain quotes across {evaluated_chains_count} expiry(s) [{earliest_q} .. {latest_q}] (any_stale=False)",
    ]
    if meta.surface_coverage_status == "SINGLE_EXPIRY_DEGRADED":
        staleness_notes.append(
            "Surface coverage: SINGLE_EXPIRY_DEGRADED (anchor expiry only; multi-expiry surface not evaluated)."
        )
    elif meta.surface_coverage_status == "INCOMPLETE_SURFACE_COVERAGE":
        staleness_notes.append(
            f"Surface coverage: {meta.surface_coverage_detail or 'INCOMPLETE_SURFACE_COVERAGE (multi-expiry surface NOT certified).'}"
        )
    if meta.uncovered_held_neo_symbols:
        staleness_notes.append(
            f"WARNING: Held open-position contract(s) {meta.uncovered_held_neo_symbols} not depth-covered."
        )
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
        habitual_style_comparison_note=final_habitual,
        reasoning_summary=final_reasoning,
        part_a_summary_status=(
            selected_primary.part_a_overall_status if selected_primary else "FAIL"
        ),
        research_rule_challenge_note=final_challenge,
        confidence_note=final_confidence,
        staleness_report=staleness_notes,
        llm_citations=(
            verified_citations
            if (verifier_report.passed_verification and verdict == "SUGGESTION")
            else []
        ),
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
        debate_state=final_debate,
        final_output=final_output,
        risk_triad_state=final_triad,
        consolidator_state=final_consolidator,
    )

    extra_diag: dict[str, Any] = {}
    if all_rejected_books:
        extra_diag["rejected_depth_books"] = all_rejected_books
    if meta.coverage_ledger:
        extra_diag["coverage_ledger"] = meta.coverage_ledger

    delivery = deliver_run_output(
        state=completed_state,
        audit_log_path=config.audit_log_path,
        scrubber=scrubber,
        force_render_for_inspection=force_render_for_inspection,
        suppress_trader_alerts=suppress_trader_alerts,
        extra_audit_metadata=(extra_diag if extra_diag else None),
    )
    return completed_state, delivery
