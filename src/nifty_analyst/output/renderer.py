"""Deterministic Markdown Report Renderer implementing all 7 sections of Plan v3 Section 7.

Every rupee, Greek, IV, breakeven, Technical+Options score, and Part A check status is
rendered directly from the deterministic NiftyOptionsState fields, never from free-form LLM text.
"""

from __future__ import annotations

from nifty_analyst.scrubber import SecretScrubber
from nifty_analyst.state import CandidateStructure, NiftyOptionsState


def _format_candidate_row(c: CandidateStructure) -> str:
    kind_dir = f"{c.structure_type} ({c.direction}, {c.expiry_date}, {c.dte_trading_sessions} DTE, lot={c.lot_size})"
    style_str = "Habitual" if c.matches_part_b_style else "Open Compare"
    be_str = ", ".join(f"{b:.1f}" for b in c.breakevens_net) or "N/A"
    if c.full_size_fill_supported:
        net_str = f"{'Credit' if c.is_net_credit else 'Debit'} {c.executable_net_points_per_unit:.2f} pts ({c.execution_fill_basis})"
        loss_str = f"contractual max loss Rs {c.contractual_max_loss_rs:,.2f}"
    else:
        net_str = (
            f"INDICATIVE Touch {'Credit' if c.is_net_credit else 'Debit'} "
            f"{c.indicative_touch_net_points_per_unit:.2f} pts (full-size entry=UNKNOWN; {c.execution_fill_basis})"
        )
        loss_str = (
            f"INDICATIVE touch bound Rs {c.indicative_touch_max_loss_rs:,.2f} "
            f"(full-size max loss=UNKNOWN; Assumption: {c.price_assumption_label})"
        )
    return (
        f"- `{c.structure_id}` - {kind_dir}: Tech+Opt Score **{c.technical_options_score:.1f}/100**; "
        f"net {net_str}; {loss_str}; "
        f"planned stop Rs {c.planned_stop_rs:,.2f}; net BE {be_str}; "
        f"PoP (lognormal / empirical) {c.pop_lognormal_expiry:.1f}% / {c.pop_empirical_expiry:.1f}%; "
        f"net vega Rs {c.net_vega_rs:+.2f}; Part A **{c.part_a_overall_status}** ({style_str})."
    )


def render_analyst_report(
    state: NiftyOptionsState,
    scrubber: SecretScrubber | None = None,
) -> str:
    """Render the 7-section research report from NiftyOptionsState."""
    scrubber = scrubber or SecretScrubber()
    meta = state.run_metadata
    ctx = state.contract_context
    ev = state.market_evidence
    ps = ev.price_structure
    vol = ev.volatility_state
    pos = ev.derivatives_positioning
    fo = state.final_output

    verdict = fo.verdict if fo else "INSUFFICIENT DATA"
    live_badge = (
        "LIVE-READY (All Part A & IVP Gates Verified)"
        if (fo and fo.is_live_ready)
        else f"RESEARCH-ONLY ({fo.research_only_reason if fo else 'Unverified state'})"
    )

    lines: list[str] = []
    lines.append("# NIFTY 50 Options Research & Strategy Analyst Report")
    lines.append("")
    lines.append("## 1. Header & Data Freshness Banner")
    lines.append(f"- **Verdict**: **{verdict}**")
    lines.append(f"- **Readiness Status**: **{live_badge}**")
    if fo and fo.duplicate_alert_suppressed:
        lines.append(
            "- **Intraday Alert Deduplication**: `SUPPRESSED` (identical structure already alerted earlier today)"
        )
    lines.append(f"- **Run ID**: `{meta.run_id}`")
    lines.append(f"- **Decision Timestamp (IST)**: `{meta.decision_timestamp_ist}`")
    lines.append(
        f"- **Data Mode**: `{meta.mode}` (Scheduled Slot Approved: `{meta.scheduled_slot_approved}`)"
    )
    lines.append(
        f"- **NIFTY Spot**: `{ctx.spot.value:.2f}` (as-of `{ctx.spot.as_of_ist}`, stale=`{ctx.spot.is_stale}`, source=`{ctx.spot.source_id}`)"
    )
    lines.append(
        f"- **Data Version Hash**: `{meta.data_version_hash}` | **Config Hash**: `{meta.config_hash}`"
    )
    if meta.snapshot_id:
        lines.append(
            f"- **Evidence Snapshot**: `{meta.snapshot_id}` (SHA256: `{meta.evidence_sha256}`, "
            f"Captured: `{meta.capture_started_at_ist}` .. `{meta.capture_completed_at_ist}`)"
        )
    if meta.surface_coverage_status == "SINGLE_EXPIRY_DEGRADED":
        lines.append(
            "- **Surface Coverage**: `SINGLE_EXPIRY_DEGRADED` (anchor expiry only; multi-expiry surface interpolation not active on this path)"
        )
    if meta.uncovered_held_neo_symbols:
        lines.append(
            f"- **Uncovered Held Contracts Warning**: `{meta.uncovered_held_neo_symbols}` not depth-covered on evaluated expiry(s)"
        )

    lines.append("- **Active Contract-Master Expiries (Zero Hardcoded Weekdays/Lot Sizes)**:")
    for exp in ctx.active_expiries:
        hol_str = f" — *{exp.holiday_note}*" if exp.is_holiday_adjusted else ""
        lines.append(
            f"  - Expiry `{exp.expiry_date}`: **{exp.dte_trading_sessions} trading session(s) DTE**, "
            f"**Lot Size = {exp.lot_size}**, Holiday-Adjusted = `{exp.is_holiday_adjusted}`{hol_str}"
        )
    if ctx.lot_size_change_detected:
        lines.append(
            "  - **Notice**: Lot-size change detected across active expiries; each structure uses its exact contract-master lot size."
        )
    for warn in ctx.calendar_discrepancy_warnings:
        lines.append(f"  - **Calendar Cross-Check Warning**: {warn}")
    if fo and fo.staleness_report:
        lines.append(f"- **Staleness Report**: {'; '.join(fo.staleness_report)}")
    lines.append("")

    # Section 2: Evidence Summary
    lines.append("## 2. Evidence Summary (Keyed to `NiftyOptionsState`)")
    lines.append(
        f"- **Completed-Candle Structure & Technicals (`market_evidence.price_structure`)**: "
        f"1D Trend = `{ps.trend_1d_state}` (HH+HL Repaired = `{ps.higher_high_higher_low_repaired}`, "
        f"First-Pullback Warning = `{ps.first_pullback_in_downtrend_warning}`); "
        f"1H MA20 = `{ps.ma20_1h:.2f}`, 1H EMA50 = `{ps.ema50_1h:.2f}`, "
        f"1H RSI(14) = `{ps.rsi_14_1h:.2f}`, 1H ATR(14) = `{ps.atr_14_1h:.2f} pts`, "
        f"1D ATR(14) = `{ps.atr_14_1d:.2f} pts`, 1H Stretch Z-Score = `{ps.stretch_zscore_1h:+.2f} SD`; "
        f"Stripped Forming Bars = `{ps.stripped_forming_bars_count}`; "
        f"Final 15m Bar (15:15-15:30 IST) Completed = `{ps.final_15m_bar_completed}`."
    )
    if ps.last_completed_1h_bar:
        b1h = ps.last_completed_1h_bar
        lines.append(
            f"  - Last Completed 1H Bar: `{b1h.open_time_ist}` to `{b1h.close_time_ist}` "
            f"(O={b1h.open:.2f}, H={b1h.high:.2f}, L={b1h.low:.2f}, C={b1h.close:.2f})."
        )
    for vt in ps.verified_level_touches:
        lines.append(
            f"  - Geometric Touch Check (`{vt.level_name}` @ `{vt.level_value:.2f}`): "
            f"Touched = `{vt.touched_geometrically}` on bar [{vt.bar_low:.2f}, {vt.bar_high:.2f}], "
            f"Wick Rejection = `{vt.wick_rejection_direction}`."
        )

    atm_by_exp_txt = ", ".join(
        f"{k}: {v * 100:.2f}%" for k, v in sorted(vol.atm_iv_by_expiry.items())
    )
    ivp_252_note = (
        f"`{vol.ivp_252d}` (verified {vol.iv_history_sessions_count}-session rank)"
        if isinstance(vol.ivp_252d, (int, float))
        else f"`{vol.ivp_252d}` (UNAVAILABLE: {vol.iv_history_sessions_count} verified EOD sessions < 252, no substitute)"
    )
    lines.append(
        f"- **Volatility, Term Structure & Skew (`market_evidence.volatility_state`)**: "
        f"Live Constant-Maturity 30d IV = `{vol.constant_maturity_30d_iv * 100:.2f}%` (`{vol.iv_source_label}`), "
        f"ATM IV by Expiry = `{atm_by_exp_txt}`, "
        f"Term Structure = `{vol.term_structure_regime}` (Near-minus-Far Slope = `{vol.term_structure_slope_near_minus_far * 100:+.2f} vol pts`), "
        f"Latest Completed EOD cm30 IV (percentiled series) = `{vol.eod_cm30_latest}`, "
        f"IVP 30d / 90d = `{vol.ivp_30d}` / `{vol.ivp_90d}`; 252d IVP = {ivp_252_note}, "
        f"20d RV = `{vol.realized_vol_20d * 100:.2f}%`, IV-RV Spread = `{vol.iv_rv_spread * 100:+.2f}%`, "
        f"25d Put-Call Skew = `{vol.skew_25d_put_minus_call * 100:+.2f} vol pts`, "
        f"India VIX = `{vol.india_vix.value}` (Tag: `{vol.india_vix.verification_tag}`)."
    )
    lines.append(
        f"- **Derivatives Positioning (`market_evidence.derivatives_positioning`, Expiry `{pos.target_expiry}`)**: "
        f"PCR (OI) = `{pos.pcr_oi:.2f}`, PCR (Vol) = `{pos.pcr_volume:.2f}`, "
        f"Put OI Wall = `{pos.put_oi_wall_strike:.0f}` ({pos.put_oi_wall_contracts:,} OI), "
        f"Call OI Wall = `{pos.call_oi_wall_strike:.0f}` ({pos.call_oi_wall_contracts:,} OI), "
        f"Max Pain = `{pos.max_pain_strike:.0f}`."
    )
    if pos.prior_intraday_runs_count > 0:
        lines.append(
            f"  - Same-Day Intraday Shift (across `{pos.prior_intraday_runs_count}` earlier slot(s)): "
            f"Spot `{pos.intraday_spot_change_pts:+.2f} pts`, "
            f"CM30 IV `{pos.intraday_atm_iv_change_pts:+.2f} vol pts`, "
            f"PCR (OI) `{pos.intraday_pcr_oi_change:+.4f}`."
        )
    ev_list = (
        ", ".join(
            f"{e.event_date}: {e.event_name} ({e.impact})"
            for e in ev.event_calendar.events_within_dte
        )
        or "None within active DTE"
    )
    lines.append(
        f"- **Events Within DTE (`market_evidence.event_calendar`, Tag: `{ev.event_calendar.source_verification_tag}`)**: "
        f"{ev_list} (Calendar File Age = `{ev.event_calendar.calendar_file_age_days:.1f} days`, "
        f"Stale Warning = `{ev.event_calendar.is_calendar_stale_warning}`)."
    )
    lines.append("- **Triggered Setup Detectors (`market_evidence.triggered_detectors`)**:")
    for det in ev.triggered_detectors:
        lines.append(
            f"  - `{det.detector_id}` [{det.track} | `{det.confirmation_status}` | {det.direction}]: "
            f"{det.summary} (Invalidation: `{det.invalidation_level:.2f}` on `{det.invalidation_candle_basis}`)."
        )
    lines.append("")

    # Section 3: Multi-Persona Debate, Risk Triad, Consolidator & Citation Verification
    lines.append(
        "## 3. Multi-Persona Debate, Risk Triad & Consolidator Recommendation (`TradingAgents` Single-Pass)"
    )
    if state.debate_state is not None:
        deb = state.debate_state
        lines.append("### Adversarial Bull / Bear / Volatility-Regime Debate")
        lines.append(f"- **Bull Researcher**: {deb.bull_thesis}")
        lines.append(f"- **Bear / Red-Team Researcher**: {deb.bear_thesis}")
        if deb.vol_regime_thesis:
            lines.append(f"- **Volatility & Skew Specialist**: {deb.vol_regime_thesis}")
        if deb.rebuttal_round:
            lines.append(f"- **Debate Rebuttal & Dominant Regime (`{deb.dominant_regime}`)**: {deb.rebuttal_round}")
        lines.append("")

    if state.risk_triad_state is not None:
        rt = state.risk_triad_state
        lines.append("### Options Risk Management Triad")
        lines.append(f"- **Convexity & Edge Advocate**: {rt.convexity_advocate_note}")
        lines.append(f"- **Conservative Tail Guardian**: {rt.tail_guardian_note}")
        lines.append(f"- **Neutral Desk Arbiter**: {rt.neutral_arbiter_note}")
        lines.append("")

    if state.consolidator_state is not None:
        cs = state.consolidator_state
        lines.append("### Chief Options Strategist (Consolidator Verdict)")
        lines.append(
            f"- **Consolidator Pick**: `{cs.recommended_structure_id}` (Conviction: **{cs.conviction}**)"
        )
        lines.append(f"- **Consolidated Rationale**: {cs.final_recommendation_rationale}")
        lines.append("")

    if fo:
        lines.append("### Grounded Reasoning & Deterministic Verifier Audit")
        lines.append(fo.reasoning_summary)
        lines.append("")
        vr = fo.verifier_report
        lines.append(
            f"- **Deterministic Two-Way Verifier**: Checked `{vr.total_claims_checked}` state citations "
            f"(Unverified = `{vr.unverified_claims_count}`, unsupported rate `{vr.unsupported_claim_rate * 100:.2f}%`, threshold `<= 2.00%`); "
            f"Prose Numeric/ID Claims Checked = `{vr.prose_numbers_checked}` (Uncited/Confabulated = `{len(vr.uncited_prose_numbers)}`); "
            f"Part A & Consolidator Re-verified = `{vr.part_a_reverified}`; Verification Passed = `{vr.passed_verification}`."
        )
    lines.append("")

    # Section 4: Primary Structure Specification
    lines.append("## 4. Primary Structure Specification")
    primary = fo.primary_proposal if fo else None
    if primary is None:
        lines.append(
            f"No live structure proposed (`Verdict = {verdict}`). Sitting out is a first-class decision."
        )
    else:
        lines.append(
            f"- **Structure ID**: `{primary.structure_id}` (`{primary.structure_type}`, Direction: `{primary.direction}`, "
            f"Tech+Options Score: **{primary.technical_options_score:.1f}/100**)"
        )
        lines.append(
            f"- **Expiry & Sizing**: Expiry `{primary.expiry_date}` ({primary.dte_trading_sessions} trading sessions DTE), "
            f"Lot Size = `{primary.lot_size}`, Lots = `{primary.lots}` (Total Quantity = `{primary.lot_size * primary.lots}`), "
            f"Wing Width = `{primary.width_points:.0f} pts`."
        )
        lines.append(
            f"- **Executable Entry (`{primary.execution_fill_basis}`)**: Net `{'Credit' if primary.is_net_credit else 'Debit'}` = "
            f"`{primary.executable_net_points_per_unit:.2f} pts/unit` (after `{primary.slippage_buffer_points_per_unit:.2f} pts` "
            f"slippage buffer; Non-STT Execution Fees = `Rs {primary.statutory_charges.total_charges_rs:.2f}`). "
            f"Assumption: `{primary.price_assumption_label}`"
        )
        if fo and fo.payoff_diagram_path:
            lines.append(f"- **Payoff Diagram PNG**: `{fo.payoff_diagram_path}`")
        lines.append("")
        lines.append("### Legs Specification")
        for leg in primary.legs:
            touch_str = (
                f"{leg.touch_available_qty:,} units"
                if leg.touch_available_qty is not None
                else "UNKNOWN"
            )
            vis_str = (
                f"{leg.visible_depth_qty:,} units ({leg.top_book_depth_lots} lots)"
                if leg.visible_depth_qty is not None
                else f"{leg.top_book_depth_lots} lots"
            )
            walk_str = (
                f"; walked VWAP {leg.vwap_fill_price:.4f} ({leg.fill_basis})"
                if leg.vwap_fill_price is not None and leg.fill_basis == "FULL_SIZE_BOOK_WALK"
                else f" ({leg.fill_basis})"
            )
            lines.append(
                f"- **{leg.side} {leg.ratio}x {leg.strike:.0f} {leg.option_type}**: "
                f"bid {leg.bid:.2f} / ask {leg.ask:.2f} / mid {leg.mid:.2f}{walk_str}; IV (mid) {leg.iv * 100:.2f}%; "
                f"delta {leg.delta:+.3f}, gamma {leg.gamma:.5f}, daily theta {leg.theta_daily:+.2f}, "
                f"vega (1% vol) {leg.vega_per_vol_pt:+.2f}; OI {leg.oi:,}; "
                f"touch qty {touch_str} | visible depth {vis_str}; "
                f"liquidity score {leg.liquidity_score:.1f}."
            )
        lines.append("")
        lines.append("### The Four Distinct Rupee Quantities (Never Merged)")
        lines.append(
            f"- **1. Contractual Max Loss (Piecewise Expiry Payoff Bound): Rs {primary.contractual_max_loss_rs:,.2f}** "
            f"against the Rs 10,000.00 ceiling. Method: `{primary.contractual_max_loss_method}`. "
            f"**Gap Warning**: pre-expiry 3% overnight gap + vol-jump MTM stress = "
            f"`Rs {primary.overnight_gap_stress_3pct_loss_rs:,.2f}`."
        )
        lines.append(
            f"- **2. Margin / Capital Estimate: Rs {primary.margin_estimate_rs:,.2f}** "
            f"(source: `{primary.margin_source}`; never conflated with contractual max loss or stop)."
        )
        lines.append(
            f"- **3. Planned Per-Position Managed Stop: Rs {primary.planned_stop_rs:,.2f}** "
            f"against the Rs 2,000.00 hard rule. Indicative spread mark ~`{primary.indicative_stop_spread_value_per_unit:.2f} pts` "
            f"(Black-76 spot trigger ~`{primary.indicative_stop_index_level:.1f}`). {primary.indicative_stop_note}"
        )
        lines.append(
            f"- **4. Portfolio Summed-Stop Impact: Rs {primary.portfolio_summed_stop_after_trade_rs:,.2f}** "
            f"against the Rs 10,000.00 portfolio cap. Existing open stops (`Rs {state.portfolio_exposure_state.summed_planned_stops_rs:,.2f}`) + "
            f"this trade (`Rs {primary.planned_stop_rs:,.2f}`). **Every position counted in full; zero correlation netting.**"
        )
        lines.append("")
        lines.append("### Breakevens, Modelled PoP, Written Invalidation & Management Plan")
        lines.append(
            f"- **Breakevens (Gross)**: `{primary.breakevens_gross}` | **Breakevens (Net of Execution Fees)**: `{primary.breakevens_net}`"
        )
        lines.append(
            f"- **Modelled Terminal PoP**: Lognormal = `{primary.pop_lognormal_expiry:.2f}%`, "
            f"Empirical = `{primary.pop_empirical_expiry:.2f}%` — *{primary.pop_assumptions_label}*"
        )
        lines.append(
            f"- **Written Invalidation (22 Sep 2026 Rule)**: Index Level `{primary.invalidation_level:.2f}` "
            f"on `{primary.invalidation_candle_basis}` basis — {primary.invalidation_justification}"
        )
        lines.append(f"- **Management Plan**: {primary.management_plan}")
    lines.append("")

    # Section 5: Part A Check Matrix
    lines.append("## 5. Part A Binding Rule Check Matrix")
    target_for_matrix = primary or (
        state.candidate_structures[0] if state.candidate_structures else None
    )
    if target_for_matrix is None:
        lines.append("No candidate structures evaluated due to data halt (`INSUFFICIENT DATA`).")
    else:
        lines.append(
            f"Showing Part A matrix for `{target_for_matrix.structure_id}` (Overall Status: **{target_for_matrix.part_a_overall_status}**):"
        )
        lines.append("")
        for r_id, chk in target_for_matrix.part_a_checks.items():
            lines.append(
                f"- `{r_id}` - {chk.rule_name} ({chk.source_date}): **{chk.status}**. Evidence: {chk.evidence}"
            )
    lines.append("")

    # Section 6: Trade-Off Comparison Across Expiries & Structure Families
    lines.append(
        "## 6. Open Defined-Risk Trade-Off Comparison (Multi-Expiry Credit vs Debit vs Butterfly vs Condor)"
    )
    if fo and fo.habitual_style_comparison_note:
        lines.append(fo.habitual_style_comparison_note)
        lines.append("")
    if state.candidate_structures:
        ordered_cands = sorted(
            state.candidate_structures,
            key=lambda c: (c.part_a_overall_status == "PASS", c.technical_options_score),
            reverse=True,
        )
        for cand in ordered_cands:
            lines.append(_format_candidate_row(cand))
    lines.append("")

    # Section 7: Between-Trade Research Notes
    lines.append("## 7. Between-Trade Research Notes & Confidence")
    if fo:
        lines.append(f"- **Research / Heuristic Observation**: {fo.research_rule_challenge_note}")
        lines.append(f"- **Confidence & Readiness Note**: {fo.confidence_note}")
    lines.append("")
    lines.append("---")
    lines.append(
        "*Advisory Notice: This system is strictly read-only and has zero order execution capability. "
        "All trade decisions and execution rests solely with the human trader.*"
    )

    return scrubber.scrub_text("\n".join(lines))
