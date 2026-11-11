"""Option chain analytics, statutory charges, terminal PoP, and defined-risk candidate structure generator.

Implements Sections 4.2, 4.3, 6.1, 6.2, and 6.3 of Implementation Plan v3.
"""

from __future__ import annotations

import math
from typing import Literal, Sequence

from nifty_analyst.config import AnalystConfig
from nifty_analyst.engine.part_a_gatekeeper import (
    compute_portfolio_summed_stop_rs,
    evaluate_part_a_rules,
)
from nifty_analyst.engine.pricing import (
    LegPayoffSpec,
    RawStrikeQuote,
    black76_greeks,
    compute_piecewise_expiry_payoff,
    evaluate_leg_expiry_pnl_per_unit,
    norm_cdf,
    solve_iv_black76,
)
from nifty_analyst.state import (
    CandleBasis,
    CandidateStructure,
    DirectionType,
    ExpiryMetadata,
    OptionLeg,
    PortfolioExposureState,
    PositioningEvidence,
    PriceStructureEvidence,
    StatutoryChargesBreakdown,
    StructureKind,
    TriggeredDetector,
    VolatilityEvidence,
)


def compute_leg_liquidity_score(
    bid: float,
    ask: float,
    oi: int,
    volume: int,
    depth_lots: int,
    max_rel_spread_pct: float = 5.0,
    min_oi: int = 10000,
    min_depth_lots: int = 5,
) -> float:
    """Compute 0-100 liquidity score from relative bid-ask spread, book depth, volume, and OI."""
    mid = 0.5 * (bid + ask)
    if mid <= 0.0 or ask < bid:
        return 0.0
    rel_spread_pct = 100.0 * (ask - bid) / mid
    if rel_spread_pct > max_rel_spread_pct or oi < min_oi or depth_lots < min_depth_lots:
        # Penalize heavily so any leg failing spread/depth/OI thresholds fails the liquidity gate (<50)
        spread_score = max(0.0, 25.0 * (1.0 - rel_spread_pct / (max_rel_spread_pct * 2.0)))
        return round(min(40.0, spread_score), 2)

    spread_component = 40.0 * max(0.0, 1.0 - (rel_spread_pct / max_rel_spread_pct))
    oi_component = 25.0 * min(1.0, math.log10(max(1, oi)) / 6.0)
    vol_component = 20.0 * min(1.0, math.log10(max(1, volume + 1)) / 5.0)
    depth_component = 15.0 * min(1.0, float(depth_lots) / 20.0)
    return round(min(100.0, spread_component + oi_component + vol_component + depth_component), 2)


def build_option_leg(
    quote: RawStrikeQuote,
    option_type: Literal["CE", "PE"],
    side: Literal["LONG", "SHORT"],
    ratio: int,
    implied_forward: float,
    time_to_expiry_years: float,
    risk_free_rate: float,
    is_stale: bool,
    config: AnalystConfig,
) -> OptionLeg:
    """Build a priced OptionLeg with IV at bid/mid/ask, analytical Greeks, and liquidity score."""
    if option_type == "CE":
        bid, ask = quote.call_bid, quote.call_ask
        oi, vol, depth = quote.call_oi, quote.call_volume, quote.call_depth_lots
    else:
        bid, ask = quote.put_bid, quote.put_ask
        oi, vol, depth = quote.put_oi, quote.put_volume, quote.put_depth_lots

    mid = 0.5 * (bid + ask)
    iv_mid = solve_iv_black76(
        mid, implied_forward, quote.strike, time_to_expiry_years, risk_free_rate, option_type
    )
    iv_bid = solve_iv_black76(
        bid, implied_forward, quote.strike, time_to_expiry_years, risk_free_rate, option_type
    )
    iv_ask = solve_iv_black76(
        ask, implied_forward, quote.strike, time_to_expiry_years, risk_free_rate, option_type
    )
    greeks = black76_greeks(
        implied_forward, quote.strike, time_to_expiry_years, risk_free_rate, iv_mid, option_type
    )
    liq_score = compute_leg_liquidity_score(
        bid=bid,
        ask=ask,
        oi=oi,
        volume=vol,
        depth_lots=depth,
        max_rel_spread_pct=config.max_relative_bid_ask_spread_pct,
        min_oi=config.min_strike_oi,
        min_depth_lots=config.min_top_book_depth_lots,
    )

    return OptionLeg(
        strike=quote.strike,
        option_type=option_type,
        side=side,
        ratio=ratio,
        bid=round(bid, 2),
        ask=round(ask, 2),
        mid=round(mid, 2),
        iv=round(iv_mid, 6),
        iv_bid=round(iv_bid, 6),
        iv_ask=round(iv_ask, 6),
        delta=round(greeks.delta, 4),
        gamma=round(greeks.gamma, 6),
        theta_daily=round(greeks.theta_daily, 4),
        vega_per_vol_pt=round(greeks.vega_per_vol_pt, 4),
        oi=oi,
        volume=vol,
        top_book_depth_lots=depth,
        quote_timestamp_ist=quote.quote_timestamp_ist,
        is_stale=is_stale,
        liquidity_score=liq_score,
    )


def compute_skew_and_curvature(
    quotes: Sequence[RawStrikeQuote],
    implied_forward: float,
    atm_iv: float,
    time_to_expiry_years: float,
    risk_free_rate: float,
) -> tuple[float, float]:
    """Compute 25-delta put IV minus 25-delta call IV skew and wing curvature."""
    best_put_25d_iv = atm_iv
    best_put_diff = 999.0
    best_call_25d_iv = atm_iv
    best_call_diff = 999.0

    for q in quotes:
        c_iv = solve_iv_black76(
            q.call_mid, implied_forward, q.strike, time_to_expiry_years, risk_free_rate, "CE"
        )
        c_g = black76_greeks(
            implied_forward, q.strike, time_to_expiry_years, risk_free_rate, c_iv, "CE"
        )
        if abs(c_g.delta - 0.25) < best_call_diff:
            best_call_diff = abs(c_g.delta - 0.25)
            best_call_25d_iv = c_iv

        p_iv = solve_iv_black76(
            q.put_mid, implied_forward, q.strike, time_to_expiry_years, risk_free_rate, "PE"
        )
        p_g = black76_greeks(
            implied_forward, q.strike, time_to_expiry_years, risk_free_rate, p_iv, "PE"
        )
        if abs(abs(p_g.delta) - 0.25) < best_put_diff:
            best_put_diff = abs(abs(p_g.delta) - 0.25)
            best_put_25d_iv = p_iv

    skew_25d = round(best_put_25d_iv - best_call_25d_iv, 6)
    wing_curvature = round(0.5 * (best_put_25d_iv + best_call_25d_iv) - atm_iv, 6)
    return skew_25d, wing_curvature


def compute_positioning_evidence(
    expiry_date: str,
    quotes: Sequence[RawStrikeQuote],
) -> PositioningEvidence:
    """Compute PCR (OI and volume), Call/Put OI walls, and Max Pain strike."""
    total_call_oi = sum(max(0, q.call_oi) for q in quotes)
    total_put_oi = sum(max(0, q.put_oi) for q in quotes)
    total_call_vol = sum(max(0, q.call_volume) for q in quotes)
    total_put_vol = sum(max(0, q.put_volume) for q in quotes)

    pcr_oi = round(total_put_oi / float(max(1, total_call_oi)), 4)
    pcr_vol = round(total_put_vol / float(max(1, total_call_vol)), 4)

    call_wall_q = max(quotes, key=lambda q: q.call_oi)
    put_wall_q = max(quotes, key=lambda q: q.put_oi)

    # Max pain strike: strike minimizing total intrinsic payout to option holders
    best_pain_strike = quotes[0].strike
    min_pain_value = math.inf
    for candidate in quotes:
        s_exp = candidate.strike
        total_pain = 0.0
        for q in quotes:
            call_intrinsic = max(0.0, s_exp - q.strike) * q.call_oi
            put_intrinsic = max(0.0, q.strike - s_exp) * q.put_oi
            total_pain += call_intrinsic + put_intrinsic
        if total_pain < min_pain_value:
            min_pain_value = total_pain
            best_pain_strike = s_exp

    return PositioningEvidence(
        target_expiry=expiry_date,
        pcr_oi=pcr_oi,
        pcr_volume=pcr_vol,
        call_oi_wall_strike=call_wall_q.strike,
        call_oi_wall_contracts=call_wall_q.call_oi,
        put_oi_wall_strike=put_wall_q.strike,
        put_oi_wall_contracts=put_wall_q.put_oi,
        max_pain_strike=best_pain_strike,
    )


# Finance Act 2026, effective 1 Apr 2026: exercised/expired-ITM options attract STT at
# 0.15% of intrinsic value, payable by the PURCHASER (long holder); short legs pay no
# settlement STT (charged at entry sale). Re-verify before go-live.
EXERCISE_STT_RATE = 0.0015


def compute_indian_statutory_charges_rs(
    legs: Sequence[OptionLeg],
    lot_size: int,
    lots: int,
    brokerage_per_order_rs: float = 20.0,
) -> StatutoryChargesBreakdown:
    """Compute Indian NSE options statutory costs (round-trip entry + exit estimate).

    Rates verified against primary sources with effective dates (re-verify before go-live):
    - STT on sell-side option premium: 0.15% (Finance Act 2026, effective 1 Apr 2026; was 0.10%).
      Exercised/expired-ITM options: 0.15% of intrinsic payable by the PURCHASER (long) only;
      short legs pay no additional STT at settlement (charged at entry sale). Handled separately
      in build_candidate_structure via the worst-case-spot exercise allowance.
    - NSE transaction charges on options premium: 0.03553% total (NSE/FA/73061, 27 Feb 2026:
      Rs 3,552.99/crore + Rs 0.01/crore IPFT; prior FA/64323 Sep 2024 Rs 3,503 + Rs 50 IPFT).
    - SEBI turnover fee: 0.0001% (Rs 10/crore). Stamp duty: 0.003% buy-side. GST: 18% on
      brokerage + transaction + SEBI charges.
    """
    qty = float(lot_size * lots)
    sell_premium_turnover = 0.0
    buy_premium_turnover = 0.0
    num_orders = 0

    for leg in legs:
        leg_qty = qty * float(leg.ratio)
        # Round-trip (entry + exit prior to expiry involves 1 buy and 1 sell per leg)
        entry_val = (leg.bid if leg.side == "SHORT" else leg.ask) * leg_qty
        exit_val = leg.mid * leg_qty
        if leg.side == "SHORT":
            sell_premium_turnover += entry_val
            buy_premium_turnover += exit_val
        else:
            buy_premium_turnover += entry_val
            sell_premium_turnover += exit_val
        num_orders += 2  # entry order + exit order

    total_premium_turnover = sell_premium_turnover + buy_premium_turnover
    stt_rs = 0.0015 * sell_premium_turnover  # Finance Act 2026, w.e.f. 1 Apr 2026
    exchange_txn_rs = 0.0003553 * total_premium_turnover  # NSE/FA/73061 total incl. IPFT
    sebi_rs = 0.000001 * total_premium_turnover
    stamp_duty_rs = 0.00003 * buy_premium_turnover
    brokerage_rs = float(num_orders) * brokerage_per_order_rs
    gst_rs = 0.18 * (brokerage_rs + exchange_txn_rs + sebi_rs)
    total_rs = stt_rs + exchange_txn_rs + sebi_rs + stamp_duty_rs + brokerage_rs + gst_rs

    return StatutoryChargesBreakdown(
        stt_rs=round(stt_rs, 2),
        exchange_txn_charges_rs=round(exchange_txn_rs, 2),
        sebi_Charges_rs=round(sebi_rs, 2),
        stamp_duty_rs=round(stamp_duty_rs, 2),
        gst_rs=round(gst_rs, 2),
        brokerage_rs=round(brokerage_rs, 2),
        total_charges_rs=round(total_rs, 2),
    )


def compute_terminal_pop(
    legs: Sequence[LegPayoffSpec],
    implied_forward: float,
    atm_iv: float,
    time_to_expiry_years: float,
    dte_trading_sessions: int,
    empirical_daily_log_returns: Sequence[float],
    charges_per_unit_points: float,
) -> tuple[float, float]:
    """Compute terminal-distribution PoP under (1) IV-implied lognormal and (2) empirical returns.

    Per Section 4.3(6):
    'terminal-distribution probability under two stated distributions (IV-implied lognormal;
    empirical return distribution). Always labelled. It describes the expiry payoff only and
    does not reflect stop-managed or invalidation-managed early exits... Never inferred from 1 minus |delta|.'
    """
    t = max(time_to_expiry_years, 1e-6)
    vol = max(atm_iv, 0.05)
    sqrt_t = math.sqrt(t)

    # 1. Lognormal integration via 201-point Gauss/quantile grid over Z in [-4.0, +4.0]
    n_steps = 200
    z_min, z_max = -4.0, 4.0
    dz = (z_max - z_min) / float(n_steps)
    lognormal_prob = 0.0
    for i in range(n_steps):
        z_mid = z_min + (i + 0.5) * dz
        s_t = implied_forward * math.exp(-0.5 * vol * vol * t + vol * sqrt_t * z_mid)
        pnl_net = sum(evaluate_leg_expiry_pnl_per_unit(leg, s_t) for leg in legs) - charges_per_unit_points
        if pnl_net > 0.0:
            p_slice = norm_cdf(z_min + (i + 1) * dz) - norm_cdf(z_min + i * dz)
            lognormal_prob += p_slice

    # 2. Empirical historical return distribution over horizon = max(1, dte_trading_sessions)
    horizon = max(1, dte_trading_sessions)
    if len(empirical_daily_log_returns) >= horizon + 5:
        rolling_multi_day_returns: list[float] = []
        for idx in range(len(empirical_daily_log_returns) - horizon + 1):
            cum_log_ret = sum(empirical_daily_log_returns[idx : idx + horizon])
            rolling_multi_day_returns.append(cum_log_ret)
        wins = 0
        for r_log in rolling_multi_day_returns:
            s_emp = implied_forward * math.exp(r_log)
            pnl_net = sum(evaluate_leg_expiry_pnl_per_unit(leg, s_emp) for leg in legs) - charges_per_unit_points
            if pnl_net > 0.0:
                wins += 1
        empirical_prob = wins / float(len(rolling_multi_day_returns))
    else:
        empirical_prob = lognormal_prob

    return round(100.0 * lognormal_prob, 2), round(100.0 * empirical_prob, 2)


def estimate_exchange_hedged_margin_rs(
    spot: float,
    legs: Sequence[OptionLeg],
    contractual_max_loss_rs: float,
    lot_size: int,
    lots: int,
) -> float:
    """Estimate NSE SPAN + Exposure hedged margin (`EXCHANGE_FORMULA_ESTIMATE`)."""
    short_legs = [l for l in legs if l.side == "SHORT"]
    qty = float(lot_size * lots)
    if not short_legs:
        # Pure debit structure with no short legs or fully covered debit vertical
        net_debit_pts = sum(
            (l.ask if l.side == "LONG" else -l.bid) * float(l.ratio) for l in legs
        )
        return round(max(contractual_max_loss_rs, net_debit_pts * qty), 2)

    # Standard NSE hedged spread/condor margin approximation: exposure margin (~2% of notional per short wing)
    # plus contractual max loss buffer
    exposure_per_lot = 0.02 * spot * float(lot_size)
    short_wings = max(1, len({l.option_type for l in short_legs}))
    estimated = contractual_max_loss_rs + float(lots) * exposure_per_lot * (0.65 if short_wings == 2 else 0.55)
    return round(estimated, 2)


def build_candidate_structure(
    *,
    structure_id: str,
    structure_type: StructureKind,
    direction: DirectionType,
    expiry: ExpiryMetadata,
    legs: list[OptionLeg],
    spot: float,
    implied_forward: float,
    atm_iv: float,
    time_to_expiry_years: float,
    empirical_daily_log_returns: Sequence[float],
    invalidation_level: float,
    invalidation_candle_basis: CandleBasis,
    invalidation_justification: str,
    matches_part_b_style: bool,
    portfolio_state: PortfolioExposureState,
    price_structure: PriceStructureEvidence,
    volatility: VolatilityEvidence,
    config: AnalystConfig,
    forced_lots: int | None = None,
) -> CandidateStructure:
    """Price and evaluate a defined-risk candidate structure, including leg-wise piecewise max loss and Part A checks."""
    lot_size = expiry.lot_size

    # Boundary rejection: ratio spreads (unbalanced long/short totals per option type) are
    # UNSUPPORTED - the boundedness engine conservatively treats unbalanced shapes as
    # undefined-risk, so reject them explicitly here instead of relying on missing fields.
    for opt_type in ("CE", "PE"):
        long_r = sum(l.ratio for l in legs if l.option_type == opt_type and l.side == "LONG")
        short_r = sum(l.ratio for l in legs if l.option_type == opt_type and l.side == "SHORT")
        if long_r != short_r:
            raise ValueError(
                f"UNSUPPORTED_RATIO_STRUCTURE: {opt_type} long ratio {long_r} != short ratio {short_r}; "
                "ratio / unbalanced structures are not supported by the pricing engine."
            )

    # Executable entry per unit with slippage buffer per leg (short legs at bid - slip, long legs at ask + slip)
    slip_per_leg = config.slippage_buffer_points_per_leg
    payoff_specs_1lot: list[LegPayoffSpec] = []
    signed_cashflow_pts = 0.0  # positive = net credit received, negative = net debit paid
    total_slip_pts = 0.0

    for leg in legs:
        if leg.side == "SHORT":
            exec_px = min(leg.bid, max(0.05, leg.bid - slip_per_leg))  # never assume fill above best bid
            signed_cashflow_pts += float(leg.ratio) * exec_px
        else:
            exec_px = leg.ask + slip_per_leg
            signed_cashflow_pts -= float(leg.ratio) * exec_px
        total_slip_pts += float(leg.ratio) * slip_per_leg
        payoff_specs_1lot.append(
            LegPayoffSpec(
                strike=leg.strike,
                option_type=leg.option_type,
                side=leg.side,
                ratio=leg.ratio,
                executable_entry_price=exec_px,
            )
        )

    is_net_credit = signed_cashflow_pts >= 0.0
    exec_net_pts = round(abs(signed_cashflow_pts), 2)

    # First evaluate 1 lot to determine lot count sizing per Section 6.3
    charges_1lot = compute_indian_statutory_charges_rs(
        legs, lot_size, 1, config.brokerage_per_order_rs
    )
    pw_1lot = compute_piecewise_expiry_payoff(
        payoff_specs_1lot, lot_size, 1, charges_1lot.total_charges_rs,
        exercise_stt_rate=EXERCISE_STT_RATE,
    )

    if forced_lots is not None:
        lots = forced_lots
    else:
        # Section 6.3: start at 1 lot; only add lots if contractual max loss stays <= Rs 10,000
        # AND added planned stop keeps portfolio summed stops <= Rs 10,000.
        lots = 1
        if pw_1lot.is_defined_risk and 0.0 < pw_1lot.max_loss_rs <= config.per_trade_contractual_max_loss_ceiling_rs:
            max_lots_by_contractual = int(
                config.per_trade_contractual_max_loss_ceiling_rs // pw_1lot.max_loss_rs
            )
            open_stops = sum(p.planned_stop_rs for p in portfolio_state.open_positions)
            rem_stop_budget = max(0.0, config.portfolio_summed_stop_cap_rs - open_stops)
            max_lots_by_stop = int(rem_stop_budget // config.per_position_managed_stop_rs)
            lots = max(1, min(max_lots_by_contractual, max_lots_by_stop, 1))

    charges = compute_indian_statutory_charges_rs(
        legs, lot_size, lots, config.brokerage_per_order_rs
    )
    pw = compute_piecewise_expiry_payoff(
        payoff_specs_1lot, lot_size, lots, charges.total_charges_rs,
        exercise_stt_rate=EXERCISE_STT_RATE,
    )

    total_qty = float(lot_size * lots)
    strikes = sorted({l.strike for l in legs})
    width_points = (max(strikes) - min(strikes)) if len(strikes) >= 2 else 0.0

    # Net Greeks
    net_delta = sum(
        (l.delta if l.side == "LONG" else -l.delta) * float(l.ratio) for l in legs
    )
    net_gamma = sum(
        (l.gamma if l.side == "LONG" else -l.gamma) * float(l.ratio) for l in legs
    )
    net_theta_daily_rs = (
        sum((l.theta_daily if l.side == "LONG" else -l.theta_daily) * float(l.ratio) for l in legs)
        * total_qty
    )
    net_vega_rs = (
        sum(
            (l.vega_per_vol_pt if l.side == "LONG" else -l.vega_per_vol_pt) * float(l.ratio)
            for l in legs
        )
        * total_qty
    )

    short_legs = [l for l in legs if l.side == "SHORT"]
    short_leg_max_abs_delta = max((abs(l.delta) for l in short_legs), default=0.0)
    structure_liq_score = min((l.liquidity_score for l in legs), default=0.0)

    # Planned stop (Rs 2,000 rule) & indicative spread mark / index level (Section 4.3(3))
    planned_stop_rs = min(config.per_position_managed_stop_rs, pw.max_loss_rs) if pw.max_loss_rs > 0 else config.per_position_managed_stop_rs
    # Always enforce the Rs 2,000 managed stop rule cap
    planned_stop_rs = round(min(config.per_position_managed_stop_rs, planned_stop_rs), 2)
    stop_loss_pts_per_unit = planned_stop_rs / total_qty
    if is_net_credit:
        indicative_stop_spread_val = round(exec_net_pts + stop_loss_pts_per_unit, 2)
    else:
        indicative_stop_spread_val = round(max(0.0, exec_net_pts - stop_loss_pts_per_unit), 2)

    if abs(net_delta) > 0.02:
        delta_move_pts = -stop_loss_pts_per_unit / net_delta
        indicative_stop_index_lvl = round(spot + delta_move_pts, 2)
    else:
        indicative_stop_index_lvl = round(invalidation_level, 2)

    indicative_stop_note = (
        "INDICATIVE SUGGESTION-TIME MARK ONLY (model/IV/time-dependent). "
        "The operative Rs 2,000 stop is always computed from the actual executed fill "
        "(real fill prices, quantity, and charges) at execution time. Gaps and fast tapes "
        "can fill worse than the stop; the stop is not a guaranteed loss limit."
    )

    open_stops = [p.planned_stop_rs for p in portfolio_state.open_positions]
    summed_stop_after_trade = compute_portfolio_summed_stop_rs(open_stops, planned_stop_rs)

    margin_est_rs = estimate_exchange_hedged_margin_rs(
        spot, legs, pw.max_loss_rs, lot_size, lots
    )

    charges_pts_per_unit = charges.total_charges_rs / total_qty
    pop_lognormal, pop_empirical = compute_terminal_pop(
        legs=payoff_specs_1lot,
        implied_forward=implied_forward,
        atm_iv=atm_iv,
        time_to_expiry_years=time_to_expiry_years,
        dte_trading_sessions=expiry.dte_trading_sessions,
        empirical_daily_log_returns=empirical_daily_log_returns,
        charges_per_unit_points=charges_pts_per_unit,
    )

    # Overnight 3% adverse gap stress loss (bounded by contractual max loss)
    gap_spot_down = spot * 0.97
    gap_spot_up = spot * 1.03
    pnl_down_rs = sum(evaluate_leg_expiry_pnl_per_unit(sp, gap_spot_down) for sp in payoff_specs_1lot) * total_qty
    pnl_up_rs = sum(evaluate_leg_expiry_pnl_per_unit(sp, gap_spot_up) for sp in payoff_specs_1lot) * total_qty
    worst_gap_loss_rs = round(max(0.0, -min(pnl_down_rs, pnl_up_rs)), 2)

    mgmt_plan = (
        f"1. Exit immediately if position mark-to-market loss reaches Rs {planned_stop_rs:.0f} "
        f"(operative stop computed from actual fill; indicative spread value ~{indicative_stop_spread_val:.1f} pts). "
        f"2. Exit on completed-candle close beyond invalidation level {invalidation_level:.1f} ({invalidation_candle_basis}). "
        f"3. Review at 50-60% of max profit (Rs {0.55 * pw.max_profit_rs:.0f}). "
        f"4. Do not carry into 1 DTE session if any short leg has |delta| >= 0.25 (26 Oct 2026 rule)."
    )

    # 7 Nov 2026 decision (supersedes the 6 Nov NET-all-in-as-contractual semantics):
    # the CONTRACTUAL payoff bound is the gross expiry-payoff worst case - KNOWN exactly
    # for defined-risk structures. All-in costs are a SEPARATE labeled quantity: fixed
    # statutory charges are computable, but exercise STT is path-dependent (which legs
    # finish ITM), so the all-in worst case is a cost-model worst case, never contractual,
    # and never certified by the managed-exit plan (stops/1 DTE do not guarantee exit).
    # When the expiry-hold exercise-STT tail is unbounded (any net-covered long call),
    # the all-in quantity is reported as unbounded-if-held-to-expiry - disclosed, not
    # gated UNKNOWN and never a strategy ban.
    contractual_payoff_bound_rs = round(pw.gross_max_loss_per_unit_points * total_qty, 2)
    all_in_managed_exit_worst_rs = round(pw.max_loss_rs, 2)
    if pw.expiry_hold_bounded:
        all_in_worst_case_rs = round(max(pw.max_loss_rs, pw.expiry_hold_max_loss_rs), 2)
        all_in_cost_label = (
            "COST_MODEL_WORST_CASE: fixed statutory charges plus path-dependent exercise "
            "STT (Finance Act 2026, 0.15% intrinsic on ITM long legs); not contractual"
        )
        expiry_hold_unbounded = False
    else:
        all_in_worst_case_rs = math.inf
        all_in_cost_label = (
            "UNBOUNDED_IF_HELD_TO_EXPIRY: exercise-STT tail on ITM long call(s) grows "
            "without limit; managed-exit/1 DTE plan applies but does NOT certify a bound; "
            "path-dependent cost, not contractual"
        )
        expiry_hold_unbounded = True
    exercise_stt_rs = (
        round(pw.expiry_hold_max_loss_rs - pw.max_loss_rs, 2) if pw.expiry_hold_bounded else math.inf
    )

    part_a_checks, part_a_overall = evaluate_part_a_rules(
        is_defined_risk=pw.is_defined_risk,
        legs=legs,
        dte_trading_sessions=expiry.dte_trading_sessions,
        planned_stop_rs=planned_stop_rs,
        portfolio_state=portfolio_state,
        contractual_max_loss_rs=contractual_payoff_bound_rs,
        all_in_worst_case_rs=all_in_worst_case_rs,
        all_in_managed_exit_worst_rs=all_in_managed_exit_worst_rs,
        expiry_hold_bounded=not expiry_hold_unbounded,
        invalidation_level=invalidation_level,
        invalidation_candle_basis=invalidation_candle_basis,
        invalidation_justification=invalidation_justification,
        price_structure=price_structure,
        volatility=volatility,
        config=config,
    )

    return CandidateStructure(
        structure_id=structure_id,
        structure_type=structure_type,
        direction=direction,
        expiry_date=expiry.expiry_date,
        dte_trading_sessions=expiry.dte_trading_sessions,
        lot_size=lot_size,
        lots=lots,
        width_points=width_points,
        legs=legs,
        is_net_credit=is_net_credit,
        executable_net_points_per_unit=exec_net_pts,
        slippage_buffer_points_per_unit=round(total_slip_pts, 2),
        statutory_charges=charges,
        contractual_max_loss_rs=contractual_payoff_bound_rs,
        contractual_max_loss_method=(
            f"GROSS_CONTRACTUAL_PAYOFF_BOUND (KNOWN): exact piecewise expiry-payoff worst case "
            f"Rs {contractual_payoff_bound_rs:,.2f} at worst spot {pw.worst_case_spot:,.0f}; "
            f"all-in costs labeled separately (managed-exit path worst "
            f"Rs {all_in_managed_exit_worst_rs:,.2f} incl. charges allowance "
            f"Rs {charges.total_charges_rs:,.2f}; "
            + (
                f"expiry-hold path incl. exercise STT on ITM longs is UNBOUNDED (long-call tail) - "
                f"disclosed, not gated; managed exit before expiry is mandatory under the plan but "
                f"does not certify a bound)"
                if not pw.expiry_hold_bounded else
                f"expiry-hold path incl. exercise STT worst Rs {pw.expiry_hold_max_loss_rs:,.2f} "
                f"at spot {pw.expiry_hold_worst_spot:,.0f})"
            )
        ),
        all_in_worst_case_rs=all_in_worst_case_rs,
        all_in_managed_exit_worst_rs=all_in_managed_exit_worst_rs,
        all_in_cost_label=all_in_cost_label,
        margin_estimate_rs=margin_est_rs,
        margin_source="EXCHANGE_FORMULA_ESTIMATE",
        planned_stop_rs=planned_stop_rs,
        indicative_stop_spread_value_per_unit=indicative_stop_spread_val,
        indicative_stop_index_level=indicative_stop_index_lvl,
        indicative_stop_note=indicative_stop_note,
        portfolio_summed_stop_after_trade_rs=summed_stop_after_trade,
        max_profit_rs=pw.max_profit_rs,
        breakevens_gross=pw.breakevens_gross,
        breakevens_net=pw.breakevens_net,
        pop_lognormal_expiry=pop_lognormal,
        pop_empirical_expiry=pop_empirical,
        pop_assumptions_label=(
            "MODEL_DEPENDENT_EXPIRY_HORIZON_ONLY: Terminal payoff probability under "
            "(1) IV-implied lognormal and (2) empirical EOD return distribution. "
            "Does not reflect stop-managed or invalidation-managed early exits."
        ),
        net_delta=round(net_delta, 4),
        net_gamma=round(net_gamma, 6),
        net_theta_daily_rs=round(net_theta_daily_rs, 2),
        net_vega_rs=round(net_vega_rs, 2),
        short_leg_max_abs_delta=round(short_leg_max_abs_delta, 4),
        structure_liquidity_score=round(structure_liq_score, 2),
        overnight_gap_stress_3pct_loss_rs=worst_gap_loss_rs,
        invalidation_level=round(invalidation_level, 2),
        invalidation_candle_basis=invalidation_candle_basis,
        invalidation_justification=invalidation_justification,
        management_plan=mgmt_plan,
        part_a_checks=part_a_checks,
        part_a_overall_status=part_a_overall,
        matches_part_b_style=matches_part_b_style,
    )


def _find_quote_by_strike(
    quotes: Sequence[RawStrikeQuote], target_strike: float
) -> RawStrikeQuote:
    return min(quotes, key=lambda q: abs(q.strike - target_strike))


def _find_quote_by_target_delta(
    quotes: Sequence[RawStrikeQuote],
    option_type: Literal["CE", "PE"],
    target_abs_delta: float,
    implied_forward: float,
    time_to_expiry_years: float,
    risk_free_rate: float,
) -> RawStrikeQuote:
    best_q = quotes[0]
    best_diff = 999.0
    for q in quotes:
        mid = q.call_mid if option_type == "CE" else q.put_mid
        iv = solve_iv_black76(
            mid, implied_forward, q.strike, time_to_expiry_years, risk_free_rate, option_type
        )
        g = black76_greeks(
            implied_forward, q.strike, time_to_expiry_years, risk_free_rate, iv, option_type
        )
        diff = abs(abs(g.delta) - target_abs_delta)
        if diff < best_diff:
            best_diff = diff
            best_q = q
    return best_q


def generate_candidate_structures(
    *,
    expiry: ExpiryMetadata,
    quotes: Sequence[RawStrikeQuote],
    stale_strikes: Sequence[float],
    spot: float,
    implied_forward: float,
    atm_iv: float,
    time_to_expiry_years: float,
    empirical_daily_log_returns: Sequence[float],
    triggered_detectors: Sequence[TriggeredDetector],
    portfolio_state: PortfolioExposureState,
    price_structure: PriceStructureEvidence,
    volatility: VolatilityEvidence,
    config: AnalystConfig,
) -> list[CandidateStructure]:
    """Generate side-by-side defined-risk structures across Bullish, Bearish, and Neutral setups (Section 6.1).

    Generates credit spreads, debit spreads, iron condors, and butterflies on equal footing.
    """
    if len(quotes) < 5:
        return []

    # Boundary rejection: a candidate batch must be a single expiry. Mixed-expiry (calendar)
    # structures are UNSUPPORTED and rejected explicitly rather than silently mispriced.
    expiries_present = {q.expiry_date for q in quotes if q.expiry_date}
    if len(expiries_present) > 1:
        raise ValueError(
            f"UNSUPPORTED_MIXED_EXPIRY_CHAIN: quotes span expiries {sorted(expiries_present)}; "
            "calendar / mixed-expiry structures are not supported by the pricing engine."
        )

    stale_set = set(stale_strikes)

    # Determine invalidation anchor per direction from triggered detectors or price structure
    bull_inv = price_structure.ema50_1h if price_structure.ema50_1h > 0 else (spot - 200.0)
    bull_just = "Completed daily close below 1H EMA50 / Put OI support invalidates bullish structure."
    bear_inv = (
        price_structure.last_completed_1d_bar.high
        if price_structure.last_completed_1d_bar
        else (spot + 200.0)
    )
    bear_just = "Completed daily close above prior 1D high invalidates bearish structure."
    neut_inv = spot + 300.0
    neut_just = "Completed daily close outside OI wall bracket invalidates neutral structure."

    for det in triggered_detectors:
        if det.direction == "BULLISH":
            bull_inv = det.invalidation_level
            bull_just = det.invalidation_justification
        elif det.direction == "BEARISH":
            bear_inv = det.invalidation_level
            bear_just = det.invalidation_justification
        elif det.direction == "NEUTRAL":
            neut_inv = det.invalidation_level
            neut_just = det.invalidation_justification

    def make_leg(
        q: RawStrikeQuote, opt_type: Literal["CE", "PE"], side: Literal["LONG", "SHORT"], ratio: int = 1
    ) -> OptionLeg:
        # Per-leg ratio > 1 is legitimate for balanced bodies (e.g. butterfly shorts).
        # Ratio SPREADS (unbalanced long vs short totals per type) are rejected at
        # structure level below, not here.
        return build_option_leg(
            quote=q,
            option_type=opt_type,
            side=side,
            ratio=ratio,
            implied_forward=implied_forward,
            time_to_expiry_years=time_to_expiry_years,
            risk_free_rate=config.risk_free_rate_annual,
            is_stale=(q.strike in stale_set),
            config=config,
        )

    candidates: list[CandidateStructure] = []

    # 1. BULL PUT SPREAD (Credit, usual style: ~0.30 delta short put, 200-pt wing)
    q_short_put_30 = _find_quote_by_target_delta(
        quotes, "PE", 0.30, implied_forward, time_to_expiry_years, config.risk_free_rate_annual
    )
    q_long_put_200 = _find_quote_by_strike(quotes, q_short_put_30.strike - 200.0)
    if q_long_put_200.strike < q_short_put_30.strike:
        candidates.append(
            build_candidate_structure(
                structure_id=f"BPS_{expiry.expiry_date}_{int(q_short_put_30.strike)}_{int(q_long_put_200.strike)}",
                structure_type="BULL_PUT_SPREAD",
                direction="BULLISH",
                expiry=expiry,
                legs=[
                    make_leg(q_short_put_30, "PE", "SHORT", 1),
                    make_leg(q_long_put_200, "PE", "LONG", 1),
                ],
                spot=spot,
                implied_forward=implied_forward,
                atm_iv=atm_iv,
                time_to_expiry_years=time_to_expiry_years,
                empirical_daily_log_returns=empirical_daily_log_returns,
                invalidation_level=bull_inv,
                invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
                invalidation_justification=bull_just,
                matches_part_b_style=True,
                portfolio_state=portfolio_state,
                price_structure=price_structure,
                volatility=volatility,
                config=config,
            )
        )

    # 1B. BULL PUT SPREAD (Tighter 150-pt wing to ensure contractual max loss <= Rs 10,000 when 200-pt exceeds cap)
    q_long_put_150 = _find_quote_by_strike(quotes, q_short_put_30.strike - 150.0)
    if (
        q_long_put_150.strike < q_short_put_30.strike
        and q_long_put_150.strike != q_long_put_200.strike
    ):
        candidates.append(
            build_candidate_structure(
                structure_id=f"BPS_{expiry.expiry_date}_{int(q_short_put_30.strike)}_{int(q_long_put_150.strike)}",
                structure_type="BULL_PUT_SPREAD",
                direction="BULLISH",
                expiry=expiry,
                legs=[
                    make_leg(q_short_put_30, "PE", "SHORT", 1),
                    make_leg(q_long_put_150, "PE", "LONG", 1),
                ],
                spot=spot,
                implied_forward=implied_forward,
                atm_iv=atm_iv,
                time_to_expiry_years=time_to_expiry_years,
                empirical_daily_log_returns=empirical_daily_log_returns,
                invalidation_level=bull_inv,
                invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
                invalidation_justification=bull_just,
                matches_part_b_style=True,
                portfolio_state=portfolio_state,
                price_structure=price_structure,
                volatility=volatility,
                config=config,
            )
        )

    # 2. BULL CALL SPREAD (Debit, evaluated on equal footing per Section 6.1 & 6.2)
    q_long_call_45 = _find_quote_by_target_delta(
        quotes, "CE", 0.45, implied_forward, time_to_expiry_years, config.risk_free_rate_annual
    )
    q_short_call_up = _find_quote_by_strike(quotes, q_long_call_45.strike + 200.0)
    if q_short_call_up.strike > q_long_call_45.strike:
        candidates.append(
            build_candidate_structure(
                structure_id=f"BCS_{expiry.expiry_date}_{int(q_long_call_45.strike)}_{int(q_short_call_up.strike)}",
                structure_type="BULL_CALL_SPREAD",
                direction="BULLISH",
                expiry=expiry,
                legs=[
                    make_leg(q_long_call_45, "CE", "LONG", 1),
                    make_leg(q_short_call_up, "CE", "SHORT", 1),
                ],
                spot=spot,
                implied_forward=implied_forward,
                atm_iv=atm_iv,
                time_to_expiry_years=time_to_expiry_years,
                empirical_daily_log_returns=empirical_daily_log_returns,
                invalidation_level=bull_inv,
                invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
                invalidation_justification=bull_just,
                matches_part_b_style=False,
                portfolio_state=portfolio_state,
                price_structure=price_structure,
                volatility=volatility,
                config=config,
            )
        )

    # 3. DEFINED-RISK BULLISH CALL BUTTERFLY (1-2-1 Call Butterfly above spot)
    q_bfly_mid = _find_quote_by_strike(quotes, spot + 150.0)
    q_bfly_low = _find_quote_by_strike(quotes, q_bfly_mid.strike - 150.0)
    q_bfly_high = _find_quote_by_strike(quotes, q_bfly_mid.strike + 150.0)
    if q_bfly_low.strike < q_bfly_mid.strike < q_bfly_high.strike:
        candidates.append(
            build_candidate_structure(
                structure_id=f"BULL_BFLY_{expiry.expiry_date}_{int(q_bfly_low.strike)}_{int(q_bfly_mid.strike)}_{int(q_bfly_high.strike)}",
                structure_type="BULLISH_BUTTERFLY",
                direction="BULLISH",
                expiry=expiry,
                legs=[
                    make_leg(q_bfly_low, "CE", "LONG", 1),
                    make_leg(q_bfly_mid, "CE", "SHORT", 2),
                    make_leg(q_bfly_high, "CE", "LONG", 1),
                ],
                spot=spot,
                implied_forward=implied_forward,
                atm_iv=atm_iv,
                time_to_expiry_years=time_to_expiry_years,
                empirical_daily_log_returns=empirical_daily_log_returns,
                invalidation_level=bull_inv,
                invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
                invalidation_justification=bull_just,
                matches_part_b_style=False,
                portfolio_state=portfolio_state,
                price_structure=price_structure,
                volatility=volatility,
                config=config,
            )
        )

    # 4. BEAR CALL SPREAD (Credit, ~0.30 delta short call, 150-pt wing)
    q_short_call_30 = _find_quote_by_target_delta(
        quotes, "CE", 0.30, implied_forward, time_to_expiry_years, config.risk_free_rate_annual
    )
    q_long_call_hedge = _find_quote_by_strike(quotes, q_short_call_30.strike + 150.0)
    if q_long_call_hedge.strike > q_short_call_30.strike:
        candidates.append(
            build_candidate_structure(
                structure_id=f"BEAR_CS_{expiry.expiry_date}_{int(q_short_call_30.strike)}_{int(q_long_call_hedge.strike)}",
                structure_type="BEAR_CALL_SPREAD",
                direction="BEARISH",
                expiry=expiry,
                legs=[
                    make_leg(q_short_call_30, "CE", "SHORT", 1),
                    make_leg(q_long_call_hedge, "CE", "LONG", 1),
                ],
                spot=spot,
                implied_forward=implied_forward,
                atm_iv=atm_iv,
                time_to_expiry_years=time_to_expiry_years,
                empirical_daily_log_returns=empirical_daily_log_returns,
                invalidation_level=bear_inv,
                invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
                invalidation_justification=bear_just,
                matches_part_b_style=True,
                portfolio_state=portfolio_state,
                price_structure=price_structure,
                volatility=volatility,
                config=config,
            )
        )

    # 5. BEAR PUT SPREAD (Debit, evaluated on equal footing)
    q_long_put_45 = _find_quote_by_target_delta(
        quotes, "PE", 0.45, implied_forward, time_to_expiry_years, config.risk_free_rate_annual
    )
    q_short_put_dn = _find_quote_by_strike(quotes, q_long_put_45.strike - 150.0)
    if q_short_put_dn.strike < q_long_put_45.strike:
        candidates.append(
            build_candidate_structure(
                structure_id=f"BEAR_PS_{expiry.expiry_date}_{int(q_long_put_45.strike)}_{int(q_short_put_dn.strike)}",
                structure_type="BEAR_PUT_SPREAD",
                direction="BEARISH",
                expiry=expiry,
                legs=[
                    make_leg(q_long_put_45, "PE", "LONG", 1),
                    make_leg(q_short_put_dn, "PE", "SHORT", 1),
                ],
                spot=spot,
                implied_forward=implied_forward,
                atm_iv=atm_iv,
                time_to_expiry_years=time_to_expiry_years,
                empirical_daily_log_returns=empirical_daily_log_returns,
                invalidation_level=bear_inv,
                invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
                invalidation_justification=bear_just,
                matches_part_b_style=False,
                portfolio_state=portfolio_state,
                price_structure=price_structure,
                volatility=volatility,
                config=config,
            )
        )

    # 6. IRON CONDOR (Neutral: ~0.15 delta shorts per side, 150-pt wings)
    q_ic_short_put = _find_quote_by_target_delta(
        quotes, "PE", 0.16, implied_forward, time_to_expiry_years, config.risk_free_rate_annual
    )
    q_ic_long_put = _find_quote_by_strike(quotes, q_ic_short_put.strike - 150.0)
    q_ic_short_call = _find_quote_by_target_delta(
        quotes, "CE", 0.16, implied_forward, time_to_expiry_years, config.risk_free_rate_annual
    )
    q_ic_long_call = _find_quote_by_strike(quotes, q_ic_short_call.strike + 150.0)
    if (
        q_ic_long_put.strike < q_ic_short_put.strike < q_ic_short_call.strike < q_ic_long_call.strike
    ):
        candidates.append(
            build_candidate_structure(
                structure_id=(
                    f"IC_{expiry.expiry_date}_{int(q_ic_long_put.strike)}_"
                    f"{int(q_ic_short_put.strike)}_{int(q_ic_short_call.strike)}_{int(q_ic_long_call.strike)}"
                ),
                structure_type="IRON_CONDOR",
                direction="NEUTRAL",
                expiry=expiry,
                legs=[
                    make_leg(q_ic_long_put, "PE", "LONG", 1),
                    make_leg(q_ic_short_put, "PE", "SHORT", 1),
                    make_leg(q_ic_short_call, "CE", "SHORT", 1),
                    make_leg(q_ic_long_call, "CE", "LONG", 1),
                ],
                spot=spot,
                implied_forward=implied_forward,
                atm_iv=atm_iv,
                time_to_expiry_years=time_to_expiry_years,
                empirical_daily_log_returns=empirical_daily_log_returns,
                invalidation_level=neut_inv,
                invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
                invalidation_justification=neut_just,
                matches_part_b_style=True,
                portfolio_state=portfolio_state,
                price_structure=price_structure,
                volatility=volatility,
                config=config,
            )
        )

    return candidates
