"""TradingAgents-inspired (arXiv:2412.20138) Single-Pass Multi-Persona Framework.

Designed for zero-multicall execution (compatible with a single LLM agent inside WhatsApp,
local Ollama/vLLM, or offline deterministic mode):
1. Compact Role-Scoped Desk Brief (<1,200 tokens, scrubbed, zero raw-chain bloat).
2. Adversarial Bull / Bear / Volatility-Regime Debate (`DebateState`).
3. Three-Perspective Options Risk Triad (`RiskTriadState`: Convexity Advocate, Tail Guardian,
   Neutral Desk Arbiter).
4. Chief Options Strategist Consolidator (`ConsolidatorState`) producing the final structure
   selection (`recommended_structure_id` or `"WAIT"`) and a mobile-ready `whatsapp_digest`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import json
from typing import Any, Literal, Sequence

from nifty_analyst.scrubber import SecretScrubber
from nifty_analyst.state import (
    CandidateStructure,
    ConsolidatorState,
    DebateState,
    LLMClaimCitation,
    NiftyOptionsState,
    OptionLeg,
    RiskTriadState,
)
from nifty_analyst.synthesis.verifier import (
    assess_market_regime,
    is_candidate_compatible_with_regime,
    resolve_state_path,
    values_match,
)

PERSONA_DEFINITIONS: dict[str, str] = {
    "bull_researcher": (
        "Bull Researcher: Construct the strongest upside / support-defense case strictly from "
        "completed 1D/1H trend structure, geometric MA20/EMA50 touches, 1H RSI, and Put OI Wall support."
    ),
    "bear_researcher": (
        "Bear & Red-Team Invalidation Researcher: Challenge upside complacency using Call OI Wall "
        "overhead supply, 1H ATR noise relative to invalidation distance, downtrend/range risks, "
        "and macro/expiry catalysts within DTE."
    ),
    "vol_regime_specialist": (
        "Volatility & Skew Regime Specialist: Diagnose the options surface from 30d/90d IVP, "
        "IV-RV spread, multi-expiry term structure (CONTANGO vs BACKWARDATION), and 25-delta "
        "Put-Call skew to dictate short-vega credit vs long-vega debit vs butterfly selection."
    ),
    "risk_triad": (
        "Risk Management Triad: (a) Convexity Advocate argues for the highest technical+options "
        "edge candidate; (b) Conservative Tail Guardian scrutinizes 3% overnight gap + vol-jump "
        "stress, short-leg delta, and portfolio summed stops; (c) Neutral Desk Arbiter reconciles both."
    ),
    "consolidator": (
        "Chief Options Strategist (Consolidator): Weigh the Bull/Bear/Vol debate and Risk Triad "
        "to select ONE certified Part-A-PASS `recommended_structure_id` (or `'WAIT'` if edge is "
        "contested/insufficient), assign conviction (`HIGH`, `MODERATE`, `SIT_OUT`), write the "
        "consolidated rationale, and format a concise WhatsApp digest."
    ),
}


def _leg_effective_entry_price(leg: OptionLeg) -> float:
    if leg.executable_entry_price is not None:
        return leg.executable_entry_price
    if leg.vwap_fill_price is not None:
        return leg.vwap_fill_price
    return leg.mid


def _legs_compact_str(cand: CandidateStructure) -> str:
    parts = [
        f"{l.side} {l.ratio}x {int(l.strike)}{l.option_type}@{_leg_effective_entry_price(l):.2f}(d={l.delta:+.2f})"
        for l in cand.legs
    ]
    return " / ".join(parts)


def _legs_exec_summary(cand: CandidateStructure) -> str:
    return " | ".join(
        f"{'S' if l.side == 'SHORT' else 'B'} {l.ratio}x {int(l.strike)}{l.option_type}@{_leg_effective_entry_price(l):.2f}"
        for l in cand.legs
    )


def build_compact_desk_brief(
    state: NiftyOptionsState,
    scrubber: SecretScrubber | None = None,
    max_pass_candidates: int = 8,
) -> dict[str, Any]:
    """Build a compact (<1,200 token) role-scoped Desk Brief keyed to `NiftyOptionsState`."""
    scrubber = scrubber or SecretScrubber()
    ps = state.market_evidence.price_structure
    vol = state.market_evidence.volatility_state
    pos = state.market_evidence.derivatives_positioning
    ev_cal = state.market_evidence.event_calendar

    indexed_pass: list[tuple[int, CandidateStructure]] = [
        (idx, c)
        for idx, c in enumerate(state.candidate_structures)
        if c.part_a_overall_status == "PASS"
    ]
    indexed_pass.sort(key=lambda pair: pair[1].technical_options_score, reverse=True)

    pass_cards: list[dict[str, Any]] = []
    for state_idx, c in indexed_pass[:max_pass_candidates]:
        pass_cards.append(
            {
                "state_index": state_idx,
                "structure_id": c.structure_id,
                "structure_type": c.structure_type,
                "direction": c.direction,
                "expiry_date": c.expiry_date,
                "dte_sessions": c.dte_trading_sessions,
                "lot_size": c.lot_size,
                "lots": c.lots,
                "legs": _legs_compact_str(c),
                "entry_type": "CREDIT" if c.is_net_credit else "DEBIT",
                "net_pts_per_unit": c.executable_net_points_per_unit,
                "contractual_max_loss_rs": c.contractual_max_loss_rs,
                "planned_stop_rs": c.planned_stop_rs,
                "portfolio_summed_stop_after_trade_rs": c.portfolio_summed_stop_after_trade_rs,
                "max_profit_rs": c.max_profit_rs,
                "breakevens_net": c.breakevens_net,
                "pop_lognormal_pct": c.pop_lognormal_expiry,
                "net_delta": c.net_delta,
                "net_theta_daily_rs": c.net_theta_daily_rs,
                "net_vega_rs": c.net_vega_rs,
                "overnight_gap_stress_3pct_loss_rs": c.overnight_gap_stress_3pct_loss_rs,
                "invalidation_level": c.invalidation_level,
                "technical_options_score": c.technical_options_score,
                "execution_fill_basis": c.execution_fill_basis,
                "full_size_fill_supported": c.full_size_fill_supported,
                "price_assumption_label": c.price_assumption_label,
            }
        )

    rejected_summary: list[dict[str, Any]] = []
    for idx, c in enumerate(state.candidate_structures):
        if c.part_a_overall_status != "PASS":
            failed_rules = [
                r_id for r_id, chk in c.part_a_checks.items() if chk.status in ("FAIL", "UNKNOWN")
            ]
            rejected_summary.append(
                {
                    "state_index": idx,
                    "structure_id": c.structure_id,
                    "status": c.part_a_overall_status,
                    "non_pass_rules": failed_rules,
                    "contractual_max_loss_rs": c.contractual_max_loss_rs,
                    "execution_fill_basis": c.execution_fill_basis,
                    "full_size_fill_supported": c.full_size_fill_supported,
                }
            )

    brief = {
        "run_context": {
            "run_id": state.run_metadata.run_id,
            "snapshot_id": state.run_metadata.snapshot_id,
            "evidence_sha256": state.run_metadata.evidence_sha256,
            "decision_timestamp_ist": state.run_metadata.decision_timestamp_ist,
            "mode": state.run_metadata.mode,
            "surface_coverage_status": state.run_metadata.surface_coverage_status,
            "uncovered_held_neo_symbols": state.run_metadata.uncovered_held_neo_symbols,
            "spot": state.contract_context.spot.value,
            "active_expiries": [
                {
                    "expiry_date": e.expiry_date,
                    "dte_sessions": e.dte_trading_sessions,
                    "lot_size": e.lot_size,
                    "is_monthly": e.is_monthly,
                }
                for e in state.contract_context.active_expiries
            ],
        },
        "technical_structure": {
            "trend_1d_state": ps.trend_1d_state,
            "hh_hl_repaired": ps.higher_high_higher_low_repaired,
            "first_pullback_warning": ps.first_pullback_in_downtrend_warning,
            "ma20_1h": ps.ma20_1h,
            "ema50_1h": ps.ema50_1h,
            "rsi_14_1h": ps.rsi_14_1h,
            "atr_14_1h": ps.atr_14_1h,
            "atr_14_1d": ps.atr_14_1d,
            "stretch_zscore_1h": ps.stretch_zscore_1h,
            "last_1h_close": ps.last_completed_1h_bar.close if ps.last_completed_1h_bar else None,
            "verified_touches": [
                {
                    "level": t.level_name,
                    "value": t.level_value,
                    "touched": t.touched_geometrically,
                    "rejection": t.wick_rejection_direction,
                }
                for t in ps.verified_level_touches
                if t.touched_geometrically
            ],
        },
        "volatility_and_skew": {
            "constant_maturity_30d_iv": vol.constant_maturity_30d_iv,
            "eod_cm30_latest": vol.eod_cm30_latest,
            "atm_iv_by_expiry": vol.atm_iv_by_expiry,
            "term_structure_regime": vol.term_structure_regime,
            "term_structure_slope_near_minus_far": vol.term_structure_slope_near_minus_far,
            "ivp_30d": vol.ivp_30d,
            "ivp_90d": vol.ivp_90d,
            "ivp_252d": vol.ivp_252d,
            "realized_vol_20d": vol.realized_vol_20d,
            "iv_rv_spread": vol.iv_rv_spread,
            "skew_25d_put_minus_call": vol.skew_25d_put_minus_call,
            "india_vix": vol.india_vix.value,
        },
        "positioning_and_events": {
            "target_expiry": pos.target_expiry,
            "pcr_oi": pos.pcr_oi,
            "pcr_volume": pos.pcr_volume,
            "put_oi_wall_strike": pos.put_oi_wall_strike,
            "put_oi_wall_contracts": pos.put_oi_wall_contracts,
            "call_oi_wall_strike": pos.call_oi_wall_strike,
            "call_oi_wall_contracts": pos.call_oi_wall_contracts,
            "max_pain_strike": pos.max_pain_strike,
            "intraday_spot_change_pts": pos.intraday_spot_change_pts,
            "intraday_atm_iv_change_pts": pos.intraday_atm_iv_change_pts,
            "intraday_pcr_oi_change": pos.intraday_pcr_oi_change,
            "events_within_dte": [
                {"date": e.event_date, "name": e.event_name, "impact": e.impact}
                for e in ev_cal.events_within_dte
            ],
        },
        "triggered_detectors": [
            {
                "detector_id": d.detector_id,
                "status": d.confirmation_status,
                "direction": d.direction,
                "invalidation_level": d.invalidation_level,
                "summary": d.summary,
            }
            for d in state.market_evidence.triggered_detectors
        ],
        "portfolio_exposure": {
            "open_positions_count": len(state.portfolio_exposure_state.open_positions),
            "summed_planned_stops_rs": state.portfolio_exposure_state.summed_planned_stops_rs,
            "remaining_stop_budget_rs": state.portfolio_exposure_state.remaining_stop_budget_rs,
        },
        "certified_pass_candidates": pass_cards,
        "non_pass_candidates_summary": rejected_summary[:6],
    }
    return scrubber.scrub_obj(brief)


def build_single_pass_persona_prompt(desk_brief: dict[str, Any]) -> str:
    """Build a single-call prompt instructing an LLM to execute all 5 TradingAgents personas."""
    return (
        "You are a NIFTY 50 Index Options Research Desk operating in STRICT SINGLE-PASS MODE "
        "(inspired by TradingAgents, arXiv:2412.20138). Do NOT invent any prices, Greeks, IVs, "
        "or Rupee amounts; cite ONLY numbers present in DESK_BRIEF.\n\n"
        "Execute these 5 personas sequentially inside one JSON response:\n"
        f"1. {PERSONA_DEFINITIONS['bull_researcher']}\n"
        f"2. {PERSONA_DEFINITIONS['bear_researcher']}\n"
        f"3. {PERSONA_DEFINITIONS['vol_regime_specialist']}\n"
        f"4. {PERSONA_DEFINITIONS['risk_triad']}\n"
        f"5. {PERSONA_DEFINITIONS['consolidator']}\n\n"
        "Return a single valid JSON object with keys:\n"
        "- debate_state: {bull_thesis, bear_thesis, vol_regime_thesis, rebuttal_round, "
        "dominant_regime ('BULLISH'|'BEARISH'|'NEUTRAL'|'CONTESTED_WAIT')}\n"
        "- risk_triad: {convexity_advocate_note, tail_guardian_note, neutral_arbiter_note}\n"
        "- consolidator: {recommended_structure_id (must be a structure_id from "
        "certified_pass_candidates or 'WAIT'), conviction ('HIGH'|'MODERATE'|'SIT_OUT'), "
        "final_recommendation_rationale, whatsapp_digest}\n"
        "- reasoning_summary, habitual_comparison_note, research_rule_challenge_note, "
        "confidence_note\n"
        "- citations: array of {claim_text, state_field_path, cited_value} referencing exact "
        "NiftyOptionsState paths (use state_index from certified_pass_candidates for "
        "candidate_structures[state_index].*).\n\n"
        f"DESK_BRIEF:\n{json.dumps(desk_brief, indent=2)}"
    )


def is_pre_llm_live_ready_eligible(
    state: NiftyOptionsState,
    chosen: CandidateStructure | None,
) -> bool:
    """Return True only when all pre-LLM deterministic & live-readiness gates are satisfied.

    In `LIVE_MORNING_SNAPSHOT` mode this always returns False until exchange quote timestamps,
    multi-expiry / held-contract surface coverage, and HTTP-boundary 2.5s pacing inside
    `~/Kotak_API` wrappers are independently verified (Item 5).
    """
    vol = state.market_evidence.volatility_state
    return (
        chosen is not None
        and chosen.part_a_overall_status == "PASS"
        and chosen.full_size_fill_supported
        and state.run_metadata.mode != "LIVE_MORNING_SNAPSHOT"
        and state.contract_context.spot.verification_tag != "FETCH_TIME_NOT_EXCHANGE"
        and state.portfolio_exposure_state.portfolio_state_status == "VERIFIED"
        and vol.ivp_30d != "UNKNOWN"
        and vol.ivp_90d != "UNKNOWN"
        and not state.run_metadata.uncovered_held_neo_symbols
    )


def format_whatsapp_digest(
    state: NiftyOptionsState,
    selected: CandidateStructure | None,
    debate: DebateState,
    consolidator_conviction: Literal["HIGH", "MODERATE", "SIT_OUT"],
    is_live_ready: bool,
) -> str:
    """Render a crisp, mobile-first WhatsApp digest grounded 100% in deterministic state."""
    spot = state.contract_context.spot.value
    ps = state.market_evidence.price_structure
    vol = state.market_evidence.volatility_state
    pos = state.market_evidence.derivatives_positioning

    iv_pct = vol.constant_maturity_30d_iv * 100.0
    skew_pts = vol.skew_25d_put_minus_call * 100.0
    verified_regime = assess_market_regime(state)

    if selected is None or not is_candidate_compatible_with_regime(selected, verified_regime, state):
        return (
            f"*NIFTY Options Desk — WAIT (SIT_OUT)*\n"
            f"• Spot: {spot:.1f} | 1D: {ps.trend_1d_state} | 1H RSI: {ps.rsi_14_1h:.1f} | Debate: {verified_regime}\n"
            f"• Surface: CM30 IV {iv_pct:.2f}% | 90d IVP {vol.ivp_90d} | Skew {skew_pts:+.2f} pts ({vol.term_structure_regime})\n"
            f"• OI Bracket: Put Wall {pos.put_oi_wall_strike:.0f} / Call Wall {pos.call_oi_wall_strike:.0f} (PCR {pos.pcr_oi:.2f})\n"
            f"• Decision: Sit out (`WAIT`) — no structure cleared deterministic Part A, full-size depth, & Consolidator gates."
        )

    readiness_tag = "LIVE-READY" if is_live_ready else "RESEARCH-ONLY"
    entry_kind = "Credit" if selected.is_net_credit else "Debit"
    be_txt = ", ".join(f"{b:.0f}" for b in selected.breakevens_net) or "N/A"
    legs_txt = _legs_exec_summary(selected)
    return (
        f"*NIFTY Options Desk — SUGGESTION ({consolidator_conviction} | {readiness_tag})*\n"
        f"• *Pick*: `{selected.structure_id}` ({selected.structure_type}, {selected.direction}, {selected.expiry_date}, {selected.dte_trading_sessions} DTE)\n"
        f"• *Legs* ({selected.lots} lot = {selected.lot_size * selected.lots} qty, {selected.execution_fill_basis}): {legs_txt}\n"
        f"• *Entry*: Net {entry_kind} {selected.executable_net_points_per_unit:.2f} pts | Tech+Opt Score: {selected.technical_options_score:.1f}/100\n"
        f"• *Market*: Spot {spot:.1f} | 1D {ps.trend_1d_state} (RSI {ps.rsi_14_1h:.1f}) | 90d IVP {vol.ivp_90d} | Skew {skew_pts:+.2f}v | Debate: {verified_regime}\n"
        f"• *Risk (4 Distinct Caps)*: Max Loss Rs {selected.contractual_max_loss_rs:,.0f} | Stop Rs {selected.planned_stop_rs:,.0f} (~{selected.indicative_stop_index_level:.0f}) | Summed Stop Rs {selected.portfolio_summed_stop_after_trade_rs:,.0f}/10k | Margin ~Rs {selected.margin_estimate_rs:,.0f}\n"
        f"• *Edge & Exit*: Net BE {be_txt} | PoP {selected.pop_lognormal_expiry:.1f}% | Invalidation {selected.invalidation_level:.0f} ({selected.invalidation_candle_basis})"
    )


def _is_strict_eligible(cand: CandidateStructure) -> bool:
    return (
        cand.part_a_overall_status == "PASS"
        and cand.full_size_fill_supported
        and all(chk.status not in ("FAIL", "UNKNOWN") for chk in cand.part_a_checks.values())
    )


def build_deterministic_personas(
    state: NiftyOptionsState,
    default_primary: CandidateStructure | None,
    eligible_candidates: Sequence[CandidateStructure],
) -> tuple[DebateState, RiskTriadState, ConsolidatorState, CandidateStructure | None]:
    """Deterministically execute the Bull/Bear/Vol Debate, Risk Triad, and Consolidator."""
    spot = state.contract_context.spot.value
    ps = state.market_evidence.price_structure
    vol = state.market_evidence.volatility_state
    pos = state.market_evidence.derivatives_positioning
    evs = state.market_evidence.event_calendar.events_within_dte

    ivp_90 = vol.ivp_90d
    iv_rv_pts = vol.iv_rv_spread * 100.0
    skew_pts = vol.skew_25d_put_minus_call * 100.0

    # Assess market regime FIRST and 100% independently of any supplied/eligible candidate.
    dom_regime: Literal["BULLISH", "BEARISH", "NEUTRAL", "CONTESTED_WAIT"] = assess_market_regime(
        state
    )

    strict_eligible = [c for c in eligible_candidates if _is_strict_eligible(c)]
    compatible_eligible = [
        c for c in strict_eligible if is_candidate_compatible_with_regime(c, dom_regime, state)
    ]

    supplied_conflicted = False
    chosen: CandidateStructure | None = None
    if default_primary is not None:
        if is_candidate_compatible_with_regime(default_primary, dom_regime, state):
            chosen = default_primary
        else:
            supplied_conflicted = _is_strict_eligible(default_primary) and dom_regime != "CONTESTED_WAIT"
            chosen = None
    elif compatible_eligible:
        chosen = max(
            compatible_eligible,
            key=lambda c: (1 if c.direction == dom_regime else 0, c.technical_options_score),
        )

    # 1. Bull Researcher Thesis
    touches_bull = [
        t.level_name
        for t in ps.verified_level_touches
        if t.touched_geometrically and t.wick_rejection_direction == "BULLISH"
    ]
    touch_clause = (
        f"reinforced by verified bullish rejection at {', '.join(touches_bull)}"
        if touches_bull
        else f"holding relative to 1H MA20 ({ps.ma20_1h:.2f}) and 1H EMA50 ({ps.ema50_1h:.2f})"
    )
    bull_thesis = (
        f"Spot at {spot:.2f} sits above Put OI wall support at {pos.put_oi_wall_strike:.0f} "
        f"({pos.put_oi_wall_contracts:,} contracts, PCR OI {pos.pcr_oi:.2f}) with 1D structure "
        f"`{ps.trend_1d_state}` (HH+HL repaired={ps.higher_high_higher_low_repaired}), 1H RSI={ps.rsi_14_1h:.1f}, "
        f"{touch_clause}."
    )

    # 2. Bear / Red-Team Invalidation Thesis
    ev_clause = (
        f"Upcoming catalyst(s) within DTE ({', '.join(f'{e.event_name} on {e.event_date}' for e in evs)}) add jump risk."
        if evs
        else "No scheduled high-impact macro event inside active DTE, but intraday whipsaw risk remains."
    )
    bear_thesis = (
        f"Overhead Call OI wall at {pos.call_oi_wall_strike:.0f} ({pos.call_oi_wall_contracts:,} contracts) "
        f"caps upside expansion; 1H ATR is {ps.atr_14_1h:.1f} pts (1D ATR {ps.atr_14_1d:.1f} pts, "
        f"stretch z-score {ps.stretch_zscore_1h:+.2f} SD"
        + (
            ", first-pullback-in-downtrend warning ACTIVE"
            if ps.first_pullback_in_downtrend_warning
            else ""
        )
        + f"). {ev_clause}"
    )

    # 3. Volatility & Skew Specialist Thesis
    if isinstance(ivp_90, (int, float)) and ivp_90 >= 50.0:
        vol_pref = (
            "Elevated 90d IVP and positive IV-RV spread favour net-credit / short-vega structures "
            "(Bull Put Spread, Bear Call Spread, Iron Condor) over paying rich debit."
        )
    elif isinstance(ivp_90, (int, float)) and ivp_90 < 45.0:
        vol_pref = (
            "Compressed 90d IVP favours defined-risk debit verticals or low-cost butterflies "
            "to avoid short-vega expansion risk."
        )
    else:
        vol_pref = (
            "Neutral/unverified IVP regime demands balanced theta-to-gap-risk selection across tenors."
        )
    vol_thesis = (
        f"CM30 IV={vol.constant_maturity_30d_iv * 100:.2f}% ({vol.term_structure_regime}, "
        f"near-minus-far slope {vol.term_structure_slope_near_minus_far * 100:+.2f} vol pts), "
        f"90d IVP={ivp_90}, 20d RV={vol.realized_vol_20d * 100:.2f}% (IV-RV={iv_rv_pts:+.2f} vol pts), "
        f"25d Put-Call skew={skew_pts:+.2f} vol pts. {vol_pref}"
    )

    rebuttal = (
        f"Synthesis rebuttal: Dominant regime resolved as `{dom_regime}`. "
        f"Technical structure (`{ps.trend_1d_state}`, RSI {ps.rsi_14_1h:.1f}) and options surface "
        f"(90d IVP={ivp_90}, skew={skew_pts:+.2f} vol pts) align only when invalidation is anchored "
        f"beyond 1H ATR ({ps.atr_14_1h:.1f} pts) noise and capped by the Rs 2,000 managed stop."
    )

    debate = DebateState(
        bull_thesis=bull_thesis,
        bear_thesis=bear_thesis,
        vol_regime_thesis=vol_thesis,
        rebuttal_round=rebuttal,
        dominant_regime=dom_regime,
    )

    # 4. Risk Management Triad
    if chosen is not None:
        pool = compatible_eligible if compatible_eligible else [chosen]
        top_edge = max(pool, key=lambda c: c.technical_options_score)
        lowest_gap = min(
            pool,
            key=lambda c: (c.overnight_gap_stress_3pct_loss_rs, c.contractual_max_loss_rs),
        )
        entry_word = "Credit" if chosen.is_net_credit else "Debit"
        if top_edge.structure_id == chosen.structure_id:
            convexity_note = (
                f"Champions `{chosen.structure_id}` ({chosen.structure_type}, {chosen.direction}, "
                f"{chosen.dte_trading_sessions} DTE): highest Tech+Options Score "
                f"{chosen.technical_options_score:.1f}/100, max profit Rs {chosen.max_profit_rs:,.2f} "
                f"vs contractual bound Rs {chosen.contractual_max_loss_rs:,.2f} "
                f"(PoP {chosen.pop_lognormal_expiry:.1f}%)."
            )
        else:
            convexity_note = (
                f"Notes highest raw Tech+Options Score on `{top_edge.structure_id}` "
                f"({top_edge.structure_type}, {top_edge.dte_trading_sessions} DTE, "
                f"Tech+Options Score {top_edge.technical_options_score:.1f}/100, "
                f"max profit Rs {top_edge.max_profit_rs:,.2f} vs contractual bound Rs {top_edge.contractual_max_loss_rs:,.2f}). "
                f"Selected primary `{chosen.structure_id}` ({chosen.structure_type}, {chosen.direction}, "
                f"{chosen.dte_trading_sessions} DTE) carries Tech+Options Score "
                f"{chosen.technical_options_score:.1f}/100 and contractual bound "
                f"Rs {chosen.contractual_max_loss_rs:,.2f} (PoP {chosen.pop_lognormal_expiry:.1f}%)."
            )
        tail_note = (
            f"Flags 3% overnight gap + vol-jump MTM stress on `{chosen.structure_id}` at "
            f"Rs {chosen.overnight_gap_stress_3pct_loss_rs:,.2f} (contractual cap Rs {chosen.contractual_max_loss_rs:,.2f}, "
            f"max short |delta|={chosen.short_leg_max_abs_delta:.3f}); lowest gap-stress alternative is "
            f"`{lowest_gap.structure_id}` at Rs {lowest_gap.overnight_gap_stress_3pct_loss_rs:,.2f}. "
            f"Portfolio summed stop post-trade = Rs {chosen.portfolio_summed_stop_after_trade_rs:,.2f}/10,000."
        )
        arbiter_note = (
            f"Endorses `{chosen.structure_id}` ({chosen.direction}, `{chosen.expiry_date}`, "
            f"{chosen.lots} lot = {chosen.lot_size * chosen.lots} units, lot_size={chosen.lot_size}, "
            f"legs: {_legs_exec_summary(chosen)}, Net {entry_word} {chosen.executable_net_points_per_unit:.2f} pts) "
            f"combining Part A PASS compliance, Tech+Options Score {chosen.technical_options_score:.1f}/100, "
            f"and clear completed-candle invalidation at {chosen.invalidation_level:.2f}."
        )
    elif supplied_conflicted:
        convexity_note = (
            f"Supplied candidate direction conflicts with confirmed `{dom_regime}` market regime; "
            "no directional mismatch structure advocated."
        )
        tail_note = "Capital preservation takes precedence on regime-candidate conflict: zero new risk recommended."
        arbiter_note = "Unanimous Risk Triad decision to sit out (`WAIT`) due to regime conflict."
    elif dom_regime == "CONTESTED_WAIT":
        convexity_note = (
            "Market regime is `CONTESTED_WAIT` (missing confirmed regime signal or conflicting signals); "
            "no structure advocated."
        )
        tail_note = "Capital preservation takes precedence: zero new risk recommended."
        arbiter_note = "Unanimous Risk Triad decision to sit out (`WAIT`)."
    else:
        convexity_note = "No Part-A-PASS structure compatible with confirmed regime available to advocate."
        tail_note = "Capital preservation takes precedence: zero new risk recommended."
        arbiter_note = "Unanimous Risk Triad decision to sit out (`WAIT`)."

    risk_triad = RiskTriadState(
        convexity_advocate_note=convexity_note,
        tail_guardian_note=tail_note,
        neutral_arbiter_note=arbiter_note,
    )

    # 5. Consolidator Persona
    if chosen is not None and dom_regime != "CONTESTED_WAIT":
        conviction: Literal["HIGH", "MODERATE", "SIT_OUT"] = (
            "HIGH"
            if (
                chosen.technical_options_score >= 68.0
                and vol.ivp_90d != "UNKNOWN"
                and chosen.direction == dom_regime
            )
            else "MODERATE"
        )
        rec_id = chosen.structure_id
        rationale = (
            f"Market regime assessed as `{dom_regime}` from `{ps.trend_1d_state}` structure "
            f"(1H RSI {ps.rsi_14_1h:.1f}) and options surface (90d IVP {ivp_90}, 25d skew "
            f"{skew_pts:+.2f} vol pts, {vol.term_structure_regime}, OI walls {pos.put_oi_wall_strike:.0f} PE / "
            f"{pos.call_oi_wall_strike:.0f} CE). Consolidator selects `{chosen.structure_id}` "
            f"({chosen.structure_type}, {chosen.direction}, on `{chosen.expiry_date}`, "
            f"{chosen.dte_trading_sessions} DTE, Tech+Opt Score {chosen.technical_options_score:.1f}/100) "
            f"with {conviction} conviction, satisfying all binding Part A limits "
            f"(contractual max loss Rs {chosen.contractual_max_loss_rs:,.2f}, "
            f"margin estimate Rs {chosen.margin_estimate_rs:,.2f}, "
            f"managed stop Rs {chosen.planned_stop_rs:,.2f}, "
            f"summed portfolio stop Rs {chosen.portfolio_summed_stop_after_trade_rs:,.2f})."
        )
    else:
        chosen = None
        conviction = "SIT_OUT"
        rec_id = "WAIT"
        if supplied_conflicted:
            rationale = (
                f"Consolidator recommends `WAIT` (`SIT_OUT`): market regime assessed as `{dom_regime}`, "
                "and supplied candidate conflicts with confirmed market regime."
            )
        elif dom_regime == "CONTESTED_WAIT":
            rationale = (
                "Consolidator recommends `WAIT` (`SIT_OUT`): market regime is `CONTESTED_WAIT` "
                "(missing confirmed regime signal or conflicting technical/volatility signals)."
            )
        else:
            rationale = (
                f"Consolidator recommends `WAIT` (`SIT_OUT`): market regime assessed as `{dom_regime}`, "
                "and no compatible structure passed all binding Part A and full-size depth gates."
            )

    is_live_ready_preview = is_pre_llm_live_ready_eligible(state, chosen)
    wa_digest = format_whatsapp_digest(
        state=state,
        selected=chosen,
        debate=debate,
        consolidator_conviction=conviction,
        is_live_ready=is_live_ready_preview,
    )

    consolidator = ConsolidatorState(
        recommended_structure_id=rec_id,
        conviction=conviction,
        final_recommendation_rationale=rationale,
        whatsapp_digest=wa_digest,
    )
    return debate, risk_triad, consolidator, chosen


@dataclass(frozen=True)
class VerifiedReportNarrative:
    """Complete post-verification narrative generated 100% from deterministic verified state."""

    debate_state: DebateState
    risk_triad_state: RiskTriadState
    consolidator_state: ConsolidatorState
    reasoning_summary: str
    habitual_comparison_note: str
    research_rule_challenge_note: str
    confidence_note: str
    whatsapp_digest: str


def build_verified_report_narrative(
    state: NiftyOptionsState,
    selected_primary: CandidateStructure | None,
    eligible_candidates: Sequence[CandidateStructure],
    verdict: Literal["SUGGESTION", "WAIT"],
    is_live_ready: bool,
    verifier_passed: bool = True,
    research_only_reason: str = "",
) -> VerifiedReportNarrative:
    """Generate EVERY report narrative field strictly from the final verified state (zero LLM prose)."""
    spot = state.contract_context.spot.value
    ps = state.market_evidence.price_structure
    vol = state.market_evidence.volatility_state
    pos = state.market_evidence.derivatives_positioning
    dom_regime = assess_market_regime(state)

    supplied_for_personas: CandidateStructure | None = None
    active_eligible: list[CandidateStructure] = []
    if verdict == "SUGGESTION" and verifier_passed and selected_primary is not None:
        supplied_for_personas = selected_primary
        if is_candidate_compatible_with_regime(selected_primary, dom_regime, state):
            active_eligible = list(eligible_candidates)

    debate, triad, cons_base, chosen = build_deterministic_personas(
        state=state,
        default_primary=supplied_for_personas,
        eligible_candidates=active_eligible,
    )

    effective_live_ready = bool(is_live_ready and chosen is not None and verifier_passed)
    conviction: Literal["HIGH", "MODERATE", "SIT_OUT"] = (
        cons_base.conviction if chosen is not None else "SIT_OUT"
    )
    wa_digest = format_whatsapp_digest(
        state=state,
        selected=chosen,
        debate=debate,
        consolidator_conviction=conviction,
        is_live_ready=effective_live_ready,
    )

    if not verifier_passed:
        consolidator = ConsolidatorState(
            recommended_structure_id="WAIT",
            conviction="SIT_OUT",
            final_recommendation_rationale=(
                "Deterministic verifier rejected unverified or contradictory synthesis claims; "
                "forced fail-closed to `WAIT` (`SIT_OUT`)."
            ),
            whatsapp_digest=wa_digest,
        )
        reasoning = (
            f"Spot is at {spot} (`contract_context.spot.value`) with market regime assessed as "
            f"`{debate.dominant_regime}`. Post-synthesis deterministic verification failed; "
            "actionable prose excluded and verdict forced to `WAIT`."
        )
        habitual_note = "Excluded synthesis comparison prose due to deterministic verifier rejection."
        challenge_note = "None (verifier fail-closed `WAIT`)."
        conf_note = "RESEARCH-ONLY: Post-synthesis verifier failed (`WAIT`; no trade proposed)."
        return VerifiedReportNarrative(
            debate_state=debate,
            risk_triad_state=triad,
            consolidator_state=consolidator,
            reasoning_summary=reasoning,
            habitual_comparison_note=habitual_note,
            research_rule_challenge_note=challenge_note,
            confidence_note=conf_note,
            whatsapp_digest=wa_digest,
        )

    if chosen is not None:
        consolidator = dataclasses.replace(
            cons_base,
            recommended_structure_id=chosen.structure_id,
            conviction=conviction,
            whatsapp_digest=wa_digest,
        )
        entry_word = "Credit" if chosen.is_net_credit else "Debit"
        reasoning = (
            f"Spot is at {spot} (`contract_context.spot.value`) with 1D structure in "
            f"`{ps.trend_1d_state}` (`market_evidence.price_structure.trend_1d_state`), "
            f"dominant market regime `{debate.dominant_regime}`, "
            f"completed 1H MA20 at {ps.ma20_1h} (`market_evidence.price_structure.ma20_1h`), "
            f"and 1H RSI(14) at {ps.rsi_14_1h} (`market_evidence.price_structure.rsi_14_1h`). "
            f"90-session IVP stands at {vol.ivp_90d} (`market_evidence.volatility_state.ivp_90d`) "
            f"in `{vol.term_structure_regime}` term structure with Put OI support at "
            f"{pos.put_oi_wall_strike} and Call OI resistance at {pos.call_oi_wall_strike}. "
            f"Following Bull/Bear/Vol debate and Risk Triad review, Consolidator selects "
            f"`{chosen.structure_id}` ({chosen.structure_type}, {chosen.direction}, `{chosen.expiry_date}`, "
            f"{chosen.lots} lot = {chosen.lot_size * chosen.lots} units, legs: {_legs_exec_summary(chosen)}, "
            f"Net {entry_word} {chosen.executable_net_points_per_unit:.2f} pts, "
            f"Tech+Options Score {chosen.technical_options_score:.1f}/100) with contractual max loss "
            f"Rs {chosen.contractual_max_loss_rs:.2f} (<= Rs 10,000 ceiling), margin estimate "
            f"Rs {chosen.margin_estimate_rs:.2f}, managed stop "
            f"Rs {chosen.planned_stop_rs:.2f}, and portfolio summed stop "
            f"Rs {chosen.portfolio_summed_stop_after_trade_rs:.2f}. Thesis invalidates on a "
            f"{chosen.invalidation_candle_basis} beyond {chosen.invalidation_level}."
        )
    else:
        consolidator = ConsolidatorState(
            recommended_structure_id="WAIT",
            conviction="SIT_OUT",
            final_recommendation_rationale=cons_base.final_recommendation_rationale,
            whatsapp_digest=wa_digest,
        )
        reasoning = (
            f"Spot is at {spot} (`contract_context.spot.value`) with 1D trend `{ps.trend_1d_state}`, "
            f"90-session IVP at {vol.ivp_90d}, and market regime assessed as `{debate.dominant_regime}`. "
            "Consolidator recommends sitting out (`WAIT`): no candidate satisfied deterministic Part A "
            "rules, full-size depth, and confirmed regime compatibility."
        )

    habitual_note = (
        "All defined-risk families (Bull/Bear Credit & Debit Verticals, Iron Condors, Iron Butterflies, "
        "Directional Butterflies, and Legged Condor Components) were priced on executable depth and scored "
        "solely on technical alignment and options surface edge (STT excluded per trader directive)."
    )

    wide_failed = [
        c
        for c in state.candidate_structures
        if c.width_points >= 200.0 and c.contractual_max_loss_rs > 10000.0
    ]
    if wide_failed:
        w = wide_failed[0]
        challenge_note = (
            f"RESEARCH NOTE (Between-Trade Context): Habitual 200-pt width structure `{w.structure_id}` "
            f"(lot_size={w.lot_size}) has a contractual max loss of Rs {w.contractual_max_loss_rs:.2f}, "
            f"exceeding the Rs 10,000 per-trade ceiling unless net credit exceeds "
            f"{(w.width_points - 10000.0 / w.lot_size):.1f} pts or wing width is 150 pts / smaller lot expiry."
        )
    else:
        challenge_note = (
            "None. All Part A rules and Part B style heuristics are consistent on this snapshot."
        )

    if effective_live_ready:
        conf_note = (
            f"LIVE-READY ({conviction} conviction; IVP verified and all Part A & full-size depth checks PASS)."
        )
    elif state.run_metadata.mode == "LIVE_MORNING_SNAPSHOT":
        conf_note = (
            "RESEARCH-ONLY: Live readiness gates remain OFF (unverified exchange quote timestamps, "
            "single-expiry / held-contract surface coverage, and HTTP-boundary 2.5s pacing)."
        )
    elif state.portfolio_exposure_state.portfolio_state_status != "VERIFIED":
        conf_note = "RESEARCH-ONLY: Portfolio open-position state is UNKNOWN (fail-closed)."
    elif vol.ivp_30d == "UNKNOWN" or vol.ivp_90d == "UNKNOWN":
        conf_note = (
            "RESEARCH-ONLY: Verified historical IVP is UNKNOWN; output cannot be "
            "used for live execution until IVP history is populated."
        )
    elif chosen is None:
        conf_note = "RESEARCH-ONLY: Consolidator / Part A verdict is WAIT (no live trade proposed)."
    elif research_only_reason:
        safe_reason = research_only_reason.replace("LIVE-READY", "LIVE_READY")
        conf_note = f"RESEARCH-ONLY: {safe_reason}"
    else:
        conf_note = f"RESEARCH-ONLY: primary candidate Part A status is {chosen.part_a_overall_status}."

    return VerifiedReportNarrative(
        debate_state=debate,
        risk_triad_state=triad,
        consolidator_state=consolidator,
        reasoning_summary=reasoning,
        habitual_comparison_note=habitual_note,
        research_rule_challenge_note=challenge_note,
        confidence_note=conf_note,
        whatsapp_digest=wa_digest,
    )


def normalize_state_narrative_from_verified_facts(state: NiftyOptionsState) -> NiftyOptionsState:
    """Rebuild ALL report & persona text in `state` strictly from verified deterministic fields.

    Guarantees that even if a caller mutates `debate_state`, `risk_triad_state`, `consolidator_state`,
    `final_output.*`, or `llm_citations[*].claim_text` directly before rendering or auditing, zero
    unverified free-form text can ever appear in any report, digest, or audit record.
    """
    fo = state.final_output
    if fo is None:
        strict_el = [c for c in state.candidate_structures if _is_strict_eligible(c)]
        deb, triad, cons, _ = build_deterministic_personas(state, None, strict_el)
        return dataclasses.replace(
            state,
            debate_state=deb if state.debate_state is not None else None,
            risk_triad_state=triad if state.risk_triad_state is not None else None,
            consolidator_state=cons if state.consolidator_state is not None else None,
        )

    if fo.verdict == "INSUFFICIENT DATA":
        reason = fo.research_only_reason or (
            fo.staleness_report[0] if fo.staleness_report else "Data halt"
        )
        clean_fo = dataclasses.replace(
            fo,
            is_live_ready=False,
            primary_proposal=None,
            alternative_candidates=[],
            habitual_style_comparison_note="",
            reasoning_summary=f"Run halted with INSUFFICIENT DATA: {reason}",
            part_a_summary_status="FAIL",
            research_rule_challenge_note="",
            confidence_note=f"HALTED: {reason}",
            llm_citations=[],
            whatsapp_digest=f"*NIFTY Options Desk — INSUFFICIENT DATA*\n• Halt Reason: {reason}",
        )
        return dataclasses.replace(
            state,
            debate_state=None,
            risk_triad_state=None,
            consolidator_state=None,
            final_output=clean_fo,
        )

    cand_by_id = {c.structure_id: c for c in state.candidate_structures}
    strict_eligible = [c for c in state.candidate_structures if _is_strict_eligible(c)]
    dom_regime = assess_market_regime(state)

    supplied_cand: CandidateStructure | None = None
    verified_primary: CandidateStructure | None = None
    if (
        fo.verdict == "SUGGESTION"
        and fo.verifier_report.passed_verification
        and fo.primary_proposal is not None
    ):
        canonical_cand = cand_by_id.get(fo.primary_proposal.structure_id, fo.primary_proposal)
        supplied_cand = canonical_cand
        if is_candidate_compatible_with_regime(canonical_cand, dom_regime, state):
            verified_primary = canonical_cand

    effective_verdict: Literal["SUGGESTION", "WAIT"] = (
        "SUGGESTION" if verified_primary is not None else "WAIT"
    )
    effective_live_ready = bool(
        effective_verdict == "SUGGESTION"
        and fo.is_live_ready
        and is_pre_llm_live_ready_eligible(state, verified_primary)
    )

    narrative = build_verified_report_narrative(
        state=state,
        selected_primary=(verified_primary if verified_primary is not None else supplied_cand),
        eligible_candidates=(strict_eligible if verified_primary is not None else []),
        verdict=("SUGGESTION" if supplied_cand is not None else effective_verdict),
        is_live_ready=effective_live_ready,
        verifier_passed=fo.verifier_report.passed_verification,
        research_only_reason=fo.research_only_reason,
    )

    state_dict = state.to_dict()
    clean_citations: list[LLMClaimCitation] = []
    if effective_verdict == "SUGGESTION" and fo.verifier_report.passed_verification:
        for cit in fo.llm_citations:
            found, actual_val = resolve_state_path(state_dict, cit.state_field_path)
            if found and values_match(actual_val, cit.cited_value):
                clean_citations.append(
                    LLMClaimCitation(
                        claim_text=f"Verified `{cit.state_field_path}` = {actual_val!r}",
                        state_field_path=cit.state_field_path,
                        cited_value=cit.cited_value,
                        verified_in_state=True,
                        verification_note=f"Verified against {cit.state_field_path!r} == {actual_val!r}.",
                    )
                )

    alternatives = [
        c
        for c in state.candidate_structures
        if verified_primary is None or c.structure_id != verified_primary.structure_id
    ]
    clean_fo = dataclasses.replace(
        fo,
        verdict=effective_verdict,
        is_live_ready=effective_live_ready,
        primary_proposal=verified_primary,
        alternative_candidates=alternatives,
        habitual_style_comparison_note=narrative.habitual_comparison_note,
        reasoning_summary=narrative.reasoning_summary,
        part_a_summary_status=(
            verified_primary.part_a_overall_status if verified_primary is not None else "FAIL"
        ),
        research_rule_challenge_note=narrative.research_rule_challenge_note,
        confidence_note=narrative.confidence_note,
        llm_citations=clean_citations,
        whatsapp_digest=narrative.whatsapp_digest,
        payoff_diagram_path=(fo.payoff_diagram_path if verified_primary is not None else ""),
    )
    return dataclasses.replace(
        state,
        debate_state=narrative.debate_state,
        risk_triad_state=narrative.risk_triad_state,
        consolidator_state=narrative.consolidator_state,
        final_output=clean_fo,
    )


def parse_optional_persona_states(
    payload: dict[str, Any],
    fallback_debate: DebateState,
    fallback_triad: RiskTriadState,
    fallback_consolidator: ConsolidatorState,
) -> tuple[DebateState, RiskTriadState, ConsolidatorState]:
    """Parse optional `debate_state`, `risk_triad`, and `consolidator` dicts from an LLM/WhatsApp artifact."""
    raw_deb = payload.get("debate_state")
    if isinstance(raw_deb, dict) and raw_deb.get("bull_thesis") and raw_deb.get("bear_thesis"):
        dom = str(raw_deb.get("dominant_regime", fallback_debate.dominant_regime)).upper()
        if dom not in ("BULLISH", "BEARISH", "NEUTRAL", "CONTESTED_WAIT"):
            dom = fallback_debate.dominant_regime
        debate = DebateState(
            bull_thesis=str(raw_deb["bull_thesis"]),
            bear_thesis=str(raw_deb["bear_thesis"]),
            vol_regime_thesis=str(raw_deb.get("vol_regime_thesis", fallback_debate.vol_regime_thesis)),
            rebuttal_round=str(raw_deb.get("rebuttal_round", fallback_debate.rebuttal_round)),
            dominant_regime=dom,  # type: ignore[arg-type]
        )
    else:
        debate = fallback_debate

    raw_triad = payload.get("risk_triad") or payload.get("risk_triad_state")
    if isinstance(raw_triad, dict) and raw_triad.get("neutral_arbiter_note"):
        triad = RiskTriadState(
            convexity_advocate_note=str(
                raw_triad.get("convexity_advocate_note", fallback_triad.convexity_advocate_note)
            ),
            tail_guardian_note=str(
                raw_triad.get("tail_guardian_note", fallback_triad.tail_guardian_note)
            ),
            neutral_arbiter_note=str(raw_triad["neutral_arbiter_note"]),
        )
    else:
        triad = fallback_triad

    raw_cons = payload.get("consolidator") or payload.get("consolidator_state")
    if isinstance(raw_cons, dict) and raw_cons.get("recommended_structure_id"):
        conv = str(raw_cons.get("conviction", fallback_consolidator.conviction)).upper()
        if conv not in ("HIGH", "MODERATE", "SIT_OUT"):
            conv = fallback_consolidator.conviction
        consolidator = ConsolidatorState(
            recommended_structure_id=str(raw_cons["recommended_structure_id"]).strip(),
            conviction=conv,  # type: ignore[arg-type]
            final_recommendation_rationale=str(
                raw_cons.get(
                    "final_recommendation_rationale",
                    fallback_consolidator.final_recommendation_rationale,
                )
            ),
            whatsapp_digest=str(
                raw_cons.get("whatsapp_digest", fallback_consolidator.whatsapp_digest)
            ),
        )
    else:
        consolidator = fallback_consolidator

    return debate, triad, consolidator
