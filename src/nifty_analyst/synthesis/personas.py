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

import json
from typing import Any, Literal, Sequence

from nifty_analyst.scrubber import SecretScrubber
from nifty_analyst.state import (
    CandidateStructure,
    ConsolidatorState,
    DebateState,
    NiftyOptionsState,
    RiskTriadState,
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


def _legs_compact_str(cand: CandidateStructure) -> str:
    parts = [
        f"{l.side} {l.ratio}x {int(l.strike)}{l.option_type}@{l.mid:.1f}(d={l.delta:+.2f})"
        for l in cand.legs
    ]
    return " / ".join(parts)


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
                }
            )

    brief = {
        "run_context": {
            "run_id": state.run_metadata.run_id,
            "decision_timestamp_ist": state.run_metadata.decision_timestamp_ist,
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

    if selected is None:
        return (
            f"*NIFTY Options Desk — WAIT ({consolidator_conviction})*\n"
            f"• Spot: {spot:.1f} | 1D: {ps.trend_1d_state} | 1H RSI: {ps.rsi_14_1h:.1f}\n"
            f"• Surface: CM30 IV {iv_pct:.2f}% | 90d IVP {vol.ivp_90d} | Skew {skew_pts:+.2f} pts ({vol.term_structure_regime})\n"
            f"• OI Bracket: Put Wall {pos.put_oi_wall_strike:.0f} / Call Wall {pos.call_oi_wall_strike:.0f} (PCR {pos.pcr_oi:.2f})\n"
            f"• Decision: Sit out (`WAIT`) — no structure cleared deterministic Part A & Consolidator conviction gates."
        )

    readiness_tag = "LIVE-READY" if is_live_ready else "RESEARCH-ONLY"
    entry_kind = "Credit" if selected.is_net_credit else "Debit"
    be_txt = ", ".join(f"{b:.0f}" for b in selected.breakevens_net) or "N/A"
    legs_txt = " | ".join(
        f"{'S' if l.side == 'SHORT' else 'B'} {l.ratio}x {int(l.strike)}{l.option_type}@{l.mid:.1f}"
        for l in selected.legs
    )
    return (
        f"*NIFTY Options Desk — SUGGESTION ({consolidator_conviction} | {readiness_tag})*\n"
        f"• *Pick*: `{selected.structure_id}` ({selected.structure_type}, {selected.direction}, {selected.dte_trading_sessions} DTE)\n"
        f"• *Legs* ({selected.lots} lot = {selected.lot_size * selected.lots} qty): {legs_txt}\n"
        f"• *Entry*: Net {entry_kind} {selected.executable_net_points_per_unit:.2f} pts | Tech+Opt Score: {selected.technical_options_score:.1f}/100\n"
        f"• *Market*: Spot {spot:.1f} | 1D {ps.trend_1d_state} (RSI {ps.rsi_14_1h:.1f}) | 90d IVP {vol.ivp_90d} | Skew {skew_pts:+.2f}v | Debate: {debate.dominant_regime}\n"
        f"• *Risk (4 Distinct Caps)*: Max Loss Rs {selected.contractual_max_loss_rs:,.0f} | Stop Rs {selected.planned_stop_rs:,.0f} (~{selected.indicative_stop_index_level:.0f}) | Summed Stop Rs {selected.portfolio_summed_stop_after_trade_rs:,.0f}/10k | Margin ~Rs {selected.margin_estimate_rs:,.0f}\n"
        f"• *Edge & Exit*: Net BE {be_txt} | PoP {selected.pop_lognormal_expiry:.1f}% | Invalidation {selected.invalidation_level:.0f} ({selected.invalidation_candle_basis})"
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
    detectors = state.market_evidence.triggered_detectors

    ivp_90 = vol.ivp_90d
    iv_rv_pts = vol.iv_rv_spread * 100.0
    skew_pts = vol.skew_25d_put_minus_call * 100.0

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

    confirmed_dirs = [
        d.direction for d in detectors if d.confirmation_status == "CONFIRMED_QUANT_REGIME"
    ]
    if not detectors or not eligible_candidates:
        dom_regime: Literal["BULLISH", "BEARISH", "NEUTRAL", "CONTESTED_WAIT"] = "CONTESTED_WAIT"
    elif "BULLISH" in confirmed_dirs and ps.trend_1d_state in ("UPTREND", "DOWNTREND_REPAIRED_HH_HL"):
        dom_regime = "BULLISH"
    elif "BEARISH" in confirmed_dirs and ps.trend_1d_state == "DOWNTREND":
        dom_regime = "BEARISH"
    elif "NEUTRAL" in confirmed_dirs:
        dom_regime = "NEUTRAL"
    else:
        dom_regime = confirmed_dirs[0] if confirmed_dirs else "NEUTRAL"

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
    if eligible_candidates:
        top_edge = max(eligible_candidates, key=lambda c: c.technical_options_score)
        lowest_gap = min(
            eligible_candidates,
            key=lambda c: (c.overnight_gap_stress_3pct_loss_rs, c.contractual_max_loss_rs),
        )
        chosen = default_primary or top_edge
        convexity_note = (
            f"Champions `{top_edge.structure_id}` ({top_edge.structure_type}, {top_edge.dte_trading_sessions} DTE): "
            f"highest Tech+Options Score {top_edge.technical_options_score:.1f}/100, "
            f"max profit Rs {top_edge.max_profit_rs:,.2f} vs contractual bound Rs {top_edge.contractual_max_loss_rs:,.2f} "
            f"(PoP {top_edge.pop_lognormal_expiry:.1f}%)."
        )
        tail_note = (
            f"Flags 3% overnight gap + vol-jump MTM stress on `{chosen.structure_id}` at "
            f"Rs {chosen.overnight_gap_stress_3pct_loss_rs:,.2f} (contractual cap Rs {chosen.contractual_max_loss_rs:,.2f}, "
            f"max short |delta|={chosen.short_leg_max_abs_delta:.3f}); lowest gap-stress alternative is "
            f"`{lowest_gap.structure_id}` at Rs {lowest_gap.overnight_gap_stress_3pct_loss_rs:,.2f}. "
            f"Portfolio summed stop post-trade = Rs {chosen.portfolio_summed_stop_after_trade_rs:,.2f}/10,000."
        )
        arbiter_note = (
            f"Endorses `{chosen.structure_id}` ({chosen.expiry_date}, lot_size={chosen.lot_size}) "
            f"combining Part A PASS compliance, Tech+Options score {chosen.technical_options_score:.1f}/100, "
            f"and clear completed-candle invalidation at {chosen.invalidation_level:.2f}."
        )
    else:
        chosen = None
        convexity_note = "No Part-A-PASS structure available to advocate."
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
            if chosen.technical_options_score >= 68.0 and vol.ivp_90d != "UNKNOWN"
            else "MODERATE"
        )
        rec_id = chosen.structure_id
        rationale = (
            f"Consolidator selects `{chosen.structure_id}` ({chosen.structure_type} on `{chosen.expiry_date}`, "
            f"{chosen.dte_trading_sessions} DTE, Tech+Opt Score {chosen.technical_options_score:.1f}/100) with "
            f"{conviction} conviction. Technical `{ps.trend_1d_state}` structure (1H RSI {ps.rsi_14_1h:.1f}) and "
            f"options surface (90d IVP {ivp_90}, 25d skew {skew_pts:+.2f} vol pts, {vol.term_structure_regime}) "
            f"align with OI wall protection ({pos.put_oi_wall_strike:.0f} PE / {pos.call_oi_wall_strike:.0f} CE), "
            f"while satisfying all binding Part A limits (contractual bound Rs {chosen.contractual_max_loss_rs:,.2f}, "
            f"managed stop Rs {chosen.planned_stop_rs:,.2f}, summed portfolio stop Rs {chosen.portfolio_summed_stop_after_trade_rs:,.2f})."
        )
    else:
        chosen = None
        conviction = "SIT_OUT"
        rec_id = "WAIT"
        rationale = (
            "Consolidator recommends `WAIT` (`SIT_OUT`): either no structure passed all binding Part A "
            "gates or technical/volatility signals lack confirmed regime alignment."
        )

    is_live_ready_preview = (
        chosen is not None
        and vol.ivp_30d != "UNKNOWN"
        and vol.ivp_90d != "UNKNOWN"
        and chosen.part_a_overall_status == "PASS"
    )
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
