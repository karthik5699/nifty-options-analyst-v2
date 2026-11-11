"""Part A Deterministic Gatekeeper (Sections 5A, 6.2, 6.3, 8.3).

Evaluates all 10 binding trading rules returning PASS / FAIL / NOT_APPLICABLE / UNKNOWN
with source dates and exact evidence cited.
Enforces:
- Leg-by-leg applicability of the 1 DTE |delta| < 0.25 rule on short legs (applies to a debit
  spread's short leg too; net debit/credit is never the deciding factor).
- Three separate money policies, never merged:
  1. Rs 2,000 per-position managed stop
  2. Rs 10,000 portfolio summed-stop cap (every position's planned stop counted in full, zero correlation netting)
  3. Rs 10,000 per-trade contractual payoff-bound ceiling (gross KNOWN bound; all-in costs disclosed separately)
"""

from __future__ import annotations

import math
from typing import Sequence

from nifty_analyst.config import AnalystConfig
from nifty_analyst.state import (
    CandleBasis,
    OptionLeg,
    PartARuleCheck,
    PortfolioExposureState,
    PriceStructureEvidence,
    RuleStatus,
    VolatilityEvidence,
)


def compute_portfolio_summed_stop_rs(
    open_positions_stops_rs: Sequence[float],
    proposed_trade_stop_rs: float,
) -> float:
    """Sum the full planned stop of EVERY open position plus the proposed trade (Section 4.3, 5A.3).

    Correlated same-direction positions are NEVER netted, discounted, or merged.
    """
    return round(
        sum(float(s) for s in open_positions_stops_rs) + float(proposed_trade_stop_rs),
        2,
    )


def evaluate_part_a_rules(
    *,
    is_defined_risk: bool,
    legs: Sequence[OptionLeg],
    dte_trading_sessions: int,
    planned_stop_rs: float,
    portfolio_state: PortfolioExposureState,
    contractual_max_loss_rs: float,
    all_in_worst_case_rs: float | None = None,
    all_in_managed_exit_worst_rs: float | None = None,
    expiry_hold_bounded: bool = True,
    invalidation_level: float | None,
    invalidation_candle_basis: CandleBasis | None,
    invalidation_justification: str,
    price_structure: PriceStructureEvidence,
    volatility: VolatilityEvidence,
    config: AnalystConfig,
    proposes_automatic_rule_change: bool = False,
) -> tuple[dict[str, PartARuleCheck], RuleStatus]:
    """Evaluate all 10 binding Part A rules deterministically."""
    checks: dict[str, PartARuleCheck] = {}

    # Rule 1: Defined risk only
    short_legs = [leg for leg in legs if leg.side == "SHORT"]
    long_legs = [leg for leg in legs if leg.side == "LONG"]
    # Defined risk requires a bounded payoff AND a hedged structure: >=2 legs, >=1 long AND >=1 short.
    # Genuine all-long multi-leg exposure is naked buying (policy departure) and must FAIL,
    # not slip through via ">=1 long, >=2 legs".
    r1_pass = bool(
        is_defined_risk
        and math.isfinite(contractual_max_loss_rs)
        and len(legs) >= 2
        and len(long_legs) >= 1
        and len(short_legs) >= 1
    )
    checks["RULE_1_DEFINED_RISK_ONLY"] = PartARuleCheck(
        rule_id="RULE_1_DEFINED_RISK_ONLY",
        rule_name="Defined risk only (bounded contractual max loss by construction)",
        source_date="Standing Rule",
        status="PASS" if r1_pass else "FAIL",
        evidence=(
            f"is_defined_risk={is_defined_risk}, legs={len(legs)} "
            f"(long={len(long_legs)}, short={len(short_legs)}; both >= 1 required), "
            f"contractual payoff bound (gross, KNOWN) = Rs {contractual_max_loss_rs:.2f}."
        ),
    )

    # Rule 2: Rs 2,000 per-position managed stop
    r2_pass = (
        0.0 < planned_stop_rs <= config.per_position_managed_stop_rs + 1e-6
    )
    checks["RULE_2_PER_POSITION_STOP_2000_RS"] = PartARuleCheck(
        rule_id="RULE_2_PER_POSITION_STOP_2000_RS",
        rule_name="Rs 2,000 per-position managed stop (indicative at suggestion; operative from fill)",
        source_date="Standing Rule",
        status="PASS" if r2_pass else "FAIL",
        evidence=(
            f"planned_stop_rs=Rs {planned_stop_rs:.2f} <= limit Rs {config.per_position_managed_stop_rs:.2f}."
        ),
    )

    # Rule 3: Rs 10,000 portfolio summed-stop cap (zero correlation netting)
    open_stops = [pos.planned_stop_rs for pos in portfolio_state.open_positions]
    total_summed_stop = compute_portfolio_summed_stop_rs(open_stops, planned_stop_rs)
    r3_pass = total_summed_stop <= config.portfolio_summed_stop_cap_rs + 1e-6
    checks["RULE_3_PORTFOLIO_SUMMED_STOP_10000_RS"] = PartARuleCheck(
        rule_id="RULE_3_PORTFOLIO_SUMMED_STOP_10000_RS",
        rule_name="Rs 10,000 portfolio summed-stop cap (every position counted in full, no netting)",
        source_date="Standing Rule",
        status="PASS" if r3_pass else "FAIL",
        evidence=(
            f"open_positions_count={len(open_stops)}, existing_summed_stops=Rs {sum(open_stops):.2f}, "
            f"proposed_stop=Rs {planned_stop_rs:.2f}, total_summed_stop=Rs {total_summed_stop:.2f} "
            f"(cap Rs {config.portfolio_summed_stop_cap_rs:.2f})."
        ),
    )

    # Rule 4: Rs 10,000 per-trade maximum loss ceiling - BOTH measurements retained.
    # 7 Nov 2026 corrections: the KNOWN gross contractual payoff bound and the all-in
    # cost quantities are computed and labeled separately. User cap-policy authority is
    # UNRESOLVED on which quantity the cap means, so qualification HOLDS wherever the
    # unresolved interpretation materially changes the outcome:
    #   - gross bound > ceiling: FAIL (breaches every reading of the cap);
    #   - gross <= ceiling AND all-in worst case finite AND <= ceiling: PASS;
    #   - gross <= ceiling but (all-in worst case > ceiling) OR (expiry-hold path
    #     unbounded, so no true all-in bound exists): UNKNOWN (research-only HOLD,
    #     never an alert, until the user resolves the cap's meaning). The
    #     charges-inclusive managed-exit estimate under the ceiling does NOT settle
    #     authority for the unbounded-tail case - that path is not excluded by any
    #     user approval on record.
    # The hold is candidate-specific authority uncertainty, not a strategy-family ban.
    # The all-in path is never certified via managed exit.
    r4_status: RuleStatus
    gross_over = (
        not math.isfinite(contractual_max_loss_rs)
        or contractual_max_loss_rs > config.per_trade_contractual_max_loss_ceiling_rs + 1e-6
    )
    all_in = all_in_worst_case_rs
    all_in_unbounded = all_in is not None and not math.isfinite(all_in)
    all_in_over = (
        all_in is not None
        and math.isfinite(all_in)
        and all_in > config.per_trade_contractual_max_loss_ceiling_rs + 1e-6
    )
    if gross_over:
        r4_status = "FAIL"
    elif all_in_unbounded or all_in_over:
        r4_status = "UNKNOWN"  # research-only hold pending user cap-policy authority
    else:
        r4_status = "PASS"
    all_in_note = ""
    if all_in_worst_case_rs is not None:
        if math.isfinite(all_in_worst_case_rs):
            all_in_note = (
                f" All-in cost-inclusive worst case (separate label): Rs {all_in_worst_case_rs:.2f}"
                + (
                    f" (managed-exit path Rs {all_in_managed_exit_worst_rs:.2f} incl. charges; "
                    f"path-dependent exercise STT) - cost-model estimate, not contractual."
                    if all_in_managed_exit_worst_rs is not None else " - cost-model estimate, not contractual."
                )
            )
        else:
            all_in_note = (
                " All-in cost UNBOUNDED if held to expiry (exercise-STT tail on ITM long "
                "call(s)); path-dependent cost, not contractual - disclosed, not gated; "
                "managed-exit plan does not certify a bound."
            )
    if r4_status == "UNKNOWN":
        if all_in_unbounded:
            r4_ev = (
                f"HOLD: gross contractual payoff bound (KNOWN) = Rs {contractual_max_loss_rs:.2f} within "
                f"ceiling Rs {config.per_trade_contractual_max_loss_ceiling_rs:.2f}, and the "
                f"charges-inclusive managed-exit estimate is "
                + (
                    f"Rs {all_in_managed_exit_worst_rs:.2f}, but the expiry-hold path is UNBOUNDED "
                    f"(exercise-STT tail) - no true all-in bound exists and the managed-exit estimate "
                    f"does not settle the cap's meaning. "
                    if all_in_managed_exit_worst_rs is not None else
                    "unbounded on the expiry-hold path - no true all-in bound exists. "
                )
                + f"Which quantity the Rs 10,000 cap means is pending user authority - candidate "
                f"is research-only, no alert.{all_in_note}"
            )
        else:
            r4_ev = (
                f"HOLD: gross contractual payoff bound (KNOWN) = Rs {contractual_max_loss_rs:.2f} within "
                f"ceiling Rs {config.per_trade_contractual_max_loss_ceiling_rs:.2f}, but the all-in "
                f"cost-inclusive worst case = Rs {all_in:.2f} exceeds it. Which quantity the Rs 10,000 "
                f"cap means is pending user authority - candidate is research-only, no alert.{all_in_note}"
            )
    else:
        r4_ev = (
            f"contractual payoff bound (gross, KNOWN) = Rs {contractual_max_loss_rs:.2f} vs ceiling "
            f"Rs {config.per_trade_contractual_max_loss_ceiling_rs:.2f}.{all_in_note}"
        )
    checks["RULE_4_CONTRACTUAL_MAX_LOSS_10000_RS"] = PartARuleCheck(
        rule_id="RULE_4_CONTRACTUAL_MAX_LOSS_10000_RS",
        rule_name="Rs 10,000 per-trade maximum loss ceiling (gross KNOWN bound + all-in costs both measured; disagreement or unbounded expiry-hold tail holds research-only pending user authority)",
        source_date="6 Nov 2026, labels split 7 Nov 2026",
        status=r4_status,
        evidence=r4_ev,
    )

    # Rule 5: Expiry risk (26 Oct 2026) - leg-by-leg applicability on short legs at <= 1 DTE
    if not short_legs:
        r5_status: RuleStatus = "NOT_APPLICABLE"
        r5_ev = "Structure has zero short option legs; 1 DTE short-delta rule is NOT_APPLICABLE."
    elif dte_trading_sessions > 1:
        r5_status = "NOT_APPLICABLE"
        r5_ev = (
            f"dte_trading_sessions={dte_trading_sessions} (> 1 DTE session); "
            f"short-leg max |delta|={max(abs(l.delta) for l in short_legs):.3f} monitored for 1 DTE gate."
        )
    else:
        violating_shorts = [
            leg for leg in short_legs if abs(leg.delta) >= config.max_short_delta_at_1dte
        ]
        if violating_shorts:
            r5_status = "FAIL"
            viol_desc = ", ".join(
                f"{l.strike}{l.option_type} (|delta|={abs(l.delta):.3f})" for l in violating_shorts
            )
            r5_ev = (
                f"At {dte_trading_sessions} DTE session(s), short leg(s) [{viol_desc}] have "
                f"|delta| >= {config.max_short_delta_at_1dte:.2f} (evaluated leg-by-leg regardless of debit/credit)."
            )
        else:
            r5_status = "PASS"
            max_sd = max(abs(l.delta) for l in short_legs)
            r5_ev = (
                f"At {dte_trading_sessions} DTE session(s), all {len(short_legs)} short leg(s) "
                f"have |delta| < {config.max_short_delta_at_1dte:.2f} (max |delta|={max_sd:.3f})."
            )
    checks["RULE_5_EXPIRY_1DTE_SHORT_DELTA"] = PartARuleCheck(
        rule_id="RULE_5_EXPIRY_1DTE_SHORT_DELTA",
        rule_name="1 DTE short-leg |delta| < 0.25 evaluated leg-by-leg",
        source_date="26 Oct 2026",
        status=r5_status,
        evidence=r5_ev,
    )

    # Rule 6: Written invalidation before entry (22 Sep 2026)
    has_valid_invalidation = (
        invalidation_level is not None
        and invalidation_level > 0.0
        and invalidation_candle_basis in ("COMPLETED_DAILY_CLOSE", "COMPLETED_1H_CLOSE")
        and bool(invalidation_justification.strip())
    )
    checks["RULE_6_WRITTEN_INVALIDATION_BEFORE_ENTRY"] = PartARuleCheck(
        rule_id="RULE_6_WRITTEN_INVALIDATION_BEFORE_ENTRY",
        rule_name="Written index invalidation level on completed-close basis before entry",
        source_date="22 Sep 2026",
        status="PASS" if has_valid_invalidation else "FAIL",
        evidence=(
            f"invalidation_level={invalidation_level}, basis={invalidation_candle_basis}, "
            f"justification={invalidation_justification!r}."
        ),
    )

    # Rule 7: Signal hygiene (completed candles only + geometric touch verification)
    has_completed_bars = (
        price_structure.last_completed_1h_bar is not None
        and price_structure.last_completed_1d_bar is not None
    )
    checks["RULE_7_COMPLETED_CANDLE_AND_TOUCH_HYGIENE"] = PartARuleCheck(
        rule_id="RULE_7_COMPLETED_CANDLE_AND_TOUCH_HYGIENE",
        rule_name="Signal hygiene: completed candles only & geometrically verified touches",
        source_date="Standing Rule",
        status="PASS" if has_completed_bars else "FAIL",
        evidence=(
            f"last_completed_1h_close={price_structure.last_completed_1h_bar.close_time_ist if price_structure.last_completed_1h_bar else None}, "
            f"stripped_forming_bars={price_structure.stripped_forming_bars_count}, "
            f"verified_touches_checked={len(price_structure.verified_level_touches)}."
        ),
    )

    # Rule 8: IVP and liquidity gates (mandatory for live readiness)
    failed_liq_legs = [
        leg
        for leg in legs
        if (
            leg.is_stale
            or leg.liquidity_score < config.min_leg_liquidity_score
            or leg.oi < config.min_strike_oi
            or leg.top_book_depth_lots < config.min_top_book_depth_lots
        )
    ]
    ivp_unknown = (
        volatility.ivp_30d == "UNKNOWN"
        or volatility.ivp_90d == "UNKNOWN"
    )
    if failed_liq_legs or not volatility.chain_qa_passed:
        r8_status: RuleStatus = "FAIL"
        r8_ev = (
            f"chain_qa_passed={volatility.chain_qa_passed}, "
            f"failed_liquidity_legs={[f'{l.strike}{l.option_type}' for l in failed_liq_legs]}."
        )
    elif ivp_unknown:
        r8_status = "UNKNOWN"
        r8_ev = (
            f"All {len(legs)} legs pass liquidity (min score={min(l.liquidity_score for l in legs):.1f}), "
            f"but IVP is UNKNOWN (ivp_30d={volatility.ivp_30d}, ivp_90d={volatility.ivp_90d}); "
            f"output must be labelled research-only."
        )
    else:
        r8_status = "PASS"
        r8_ev = (
            f"All {len(legs)} legs pass liquidity (min score={min(l.liquidity_score for l in legs):.1f}), "
            f"chain_qa_passed=True, ivp_30d={volatility.ivp_30d}, ivp_90d={volatility.ivp_90d}."
        )
    checks["RULE_8_IVP_AND_LIQUIDITY_GATES"] = PartARuleCheck(
        rule_id="RULE_8_IVP_AND_LIQUIDITY_GATES",
        rule_name="Mandatory IVP and per-leg liquidity gates",
        source_date="Standing Rule",
        status=r8_status,
        evidence=r8_ev,
    )

    # Rule 9: Rules change only between trades, only by the trader
    checks["RULE_9_NO_AUTOMATIC_RULE_CHANGES"] = PartARuleCheck(
        rule_id="RULE_9_NO_AUTOMATIC_RULE_CHANGES",
        rule_name="Rules change only between trades, only by trader",
        source_date="Standing Rule",
        status="FAIL" if proposes_automatic_rule_change else "PASS",
        evidence=(
            "No automatic rule mutation proposed."
            if not proposes_automatic_rule_change
            else "Candidate attempts to override a binding Part A rule."
        ),
    )

    # Rule 10: Human final decision (advisory only)
    checks["RULE_10_HUMAN_FINAL_DECISION"] = PartARuleCheck(
        rule_id="RULE_10_HUMAN_FINAL_DECISION",
        rule_name="Advisory output only; trader approves or rejects; zero execution capability",
        source_date="Standing Rule",
        status="PASS",
        evidence="System is strictly read-only with zero broker write or order-placement methods.",
    )

    statuses = [c.status for c in checks.values()]
    if "FAIL" in statuses:
        overall: RuleStatus = "FAIL"
    elif "UNKNOWN" in statuses:
        overall = "UNKNOWN"
    else:
        overall = "PASS"

    return checks, overall
