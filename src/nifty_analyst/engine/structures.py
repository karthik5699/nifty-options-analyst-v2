"""Option chain analytics, non-STT execution charges, Black-76 MTM risk analytics,
technical+options composite scoring, and defined-risk candidate structure generator.

Implements Sections 4.2, 4.3, 6.1, 6.2, and 6.3 of Implementation Plan v3.
STT is excluded per trader directive so technical analysis and options surface analytics
drive structure selection across verticals, condors, and butterflies on equal footing.
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
    black76_price,
    brentq_pure,
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
    """Build a priced OptionLeg with IV at bid/mid/ask, analytical Greeks, and side-aware liquidity score."""
    depth = quote.depth_lots_for(option_type, side)
    if option_type == "CE":
        if not quote.has_executable_call or quote.call_bid is None or quote.call_ask is None:
            raise ValueError(
                f"Cannot build OptionLeg from unpriced/non-executable CE quote at strike {quote.strike} "
                f"(bid={quote.call_bid}, ask={quote.call_ask})."
            )
        bid, ask = quote.call_bid, quote.call_ask
        oi, vol = quote.call_oi, quote.call_volume
    else:
        if not quote.has_executable_put or quote.put_bid is None or quote.put_ask is None:
            raise ValueError(
                f"Cannot build OptionLeg from unpriced/non-executable PE quote at strike {quote.strike} "
                f"(bid={quote.put_bid}, ask={quote.put_ask})."
            )
        bid, ask = quote.put_bid, quote.put_ask
        oi, vol = quote.put_oi, quote.put_volume

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
        if q.has_executable_call and q.call_mid is not None:
            c_iv = solve_iv_black76(
                q.call_mid, implied_forward, q.strike, time_to_expiry_years, risk_free_rate, "CE"
            )
            c_g = black76_greeks(
                implied_forward, q.strike, time_to_expiry_years, risk_free_rate, c_iv, "CE"
            )
            if abs(c_g.delta - 0.25) < best_call_diff:
                best_call_diff = abs(c_g.delta - 0.25)
                best_call_25d_iv = c_iv

        if q.has_executable_put and q.put_mid is not None:
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


def compute_indian_statutory_charges_rs(
    legs: Sequence[OptionLeg],
    lot_size: int,
    lots: int,
    brokerage_per_order_rs: float = 20.0,
) -> StatutoryChargesBreakdown:
    """Compute round-trip non-STT execution costs (exchange txn, SEBI, stamp duty, brokerage, GST).

    STT is intentionally excluded per trader directive so technical analysis and options
    surface analytics drive strategy selection rather than tax artifacts.
    """
    qty = float(lot_size * lots)
    sell_premium_turnover = 0.0
    buy_premium_turnover = 0.0
    num_orders = 0

    for leg in legs:
        leg_qty = qty * float(leg.ratio)
        entry_val = (leg.bid if leg.side == "SHORT" else leg.ask) * leg_qty
        exit_val = leg.mid * leg_qty
        if leg.side == "SHORT":
            sell_premium_turnover += entry_val
            buy_premium_turnover += exit_val
        else:
            buy_premium_turnover += entry_val
            sell_premium_turnover += exit_val
        num_orders += 2

    total_premium_turnover = sell_premium_turnover + buy_premium_turnover
    exchange_txn_rs = 0.0003553 * total_premium_turnover  # NSE/FA/73061 total incl. IPFT
    sebi_rs = 0.000001 * total_premium_turnover
    stamp_duty_rs = 0.00003 * buy_premium_turnover
    brokerage_rs = float(num_orders) * brokerage_per_order_rs
    gst_rs = 0.18 * (brokerage_rs + exchange_txn_rs + sebi_rs)
    total_rs = exchange_txn_rs + sebi_rs + stamp_duty_rs + brokerage_rs + gst_rs

    return StatutoryChargesBreakdown(
        exchange_txn_charges_rs=round(exchange_txn_rs, 2),
        sebi_charges_rs=round(sebi_rs, 2),
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
    """Compute terminal-distribution PoP under (1) IV-implied lognormal and (2) empirical returns."""
    t = max(time_to_expiry_years, 1e-6)
    vol = max(atm_iv, 0.05)
    sqrt_t = math.sqrt(t)

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
        net_debit_pts = sum(
            (l.ask if l.side == "LONG" else -l.bid) * float(l.ratio) for l in legs
        )
        return round(max(contractual_max_loss_rs, net_debit_pts * qty), 2)

    exposure_per_lot = 0.02 * spot * float(lot_size)
    short_wings = max(1, len({l.option_type for l in short_legs}))
    estimated = contractual_max_loss_rs + float(lots) * exposure_per_lot * (
        0.65 if short_wings == 2 else 0.55
    )
    return round(estimated, 2)


def _mtm_pnl_points_at_spot(
    legs: Sequence[OptionLeg],
    payoff_specs: Sequence[LegPayoffSpec],
    spot_candidate: float,
    spot_ref: float,
    implied_forward_ref: float,
    time_to_expiry_years: float,
    risk_free_rate: float,
    iv_shift: float = 0.0,
) -> float:
    """Compute Black-76 mark-to-market P&L in points per unit at a shocked spot and IV shift."""
    fwd_candidate = implied_forward_ref * (spot_candidate / max(1.0, spot_ref))
    pnl_pts = 0.0
    for leg, spec in zip(legs, payoff_specs):
        shocked_iv = max(0.01, leg.iv + iv_shift)
        theo_px = black76_price(
            fwd_candidate,
            leg.strike,
            max(1e-6, time_to_expiry_years),
            risk_free_rate,
            shocked_iv,
            leg.option_type,
        )
        if leg.side == "LONG":
            pnl_pts += float(leg.ratio) * (theo_px - spec.executable_entry_price)
        else:
            pnl_pts += float(leg.ratio) * (spec.executable_entry_price - theo_px)
    return pnl_pts


def _solve_indicative_stop_index_level(
    legs: Sequence[OptionLeg],
    payoff_specs: Sequence[LegPayoffSpec],
    spot: float,
    implied_forward: float,
    time_to_expiry_years: float,
    risk_free_rate: float,
    stop_loss_pts_per_unit: float,
    net_delta: float,
    invalidation_level: float,
) -> float:
    """Solve Black-76 MTM spot root where position MTM loss reaches `stop_loss_pts_per_unit`."""
    target_pnl = -abs(stop_loss_pts_per_unit)

    def obj(s_val: float) -> float:
        return (
            _mtm_pnl_points_at_spot(
                legs,
                payoff_specs,
                s_val,
                spot,
                implied_forward,
                time_to_expiry_years,
                risk_free_rate,
                iv_shift=0.0,
            )
            - target_pnl
        )

    if abs(net_delta) > 0.02:
        low_b, high_b = (spot * 0.85, spot) if net_delta > 0 else (spot, spot * 1.15)
        try:
            if obj(low_b) * obj(high_b) <= 0.0:
                return round(brentq_pure(obj, low_b, high_b, xtol=1e-3), 2)
        except ValueError:
            pass
        delta_move_pts = -stop_loss_pts_per_unit / net_delta
        return round(spot + delta_move_pts, 2)

    # Neutral structure (Iron Condor / Butterfly): check both wings and report closer stop trigger
    candidates_found: list[float] = []
    for low_b, high_b in ((spot * 0.90, spot), (spot, spot * 1.10)):
        try:
            if obj(low_b) * obj(high_b) <= 0.0:
                candidates_found.append(brentq_pure(obj, low_b, high_b, xtol=1e-3))
        except ValueError:
            continue
    if candidates_found:
        closest = min(candidates_found, key=lambda lvl: abs(lvl - spot))
        return round(closest, 2)
    return round(invalidation_level, 2)


def _compute_overnight_gap_stress_rs(
    legs: Sequence[OptionLeg],
    payoff_specs: Sequence[LegPayoffSpec],
    spot: float,
    implied_forward: float,
    time_to_expiry_years: float,
    risk_free_rate: float,
    total_qty: float,
    contractual_max_loss_rs: float,
) -> float:
    """Compute pre-expiry overnight +/-3% spot gap stress with asymmetric VIX jump (+4 / +2 vol pts)."""
    t_next_open = max(1e-6, time_to_expiry_years - (1.0 / 365.0))
    pnl_down_pts = _mtm_pnl_points_at_spot(
        legs,
        payoff_specs,
        spot * 0.97,
        spot,
        implied_forward,
        t_next_open,
        risk_free_rate,
        iv_shift=0.04,
    )
    pnl_up_pts = _mtm_pnl_points_at_spot(
        legs,
        payoff_specs,
        spot * 1.03,
        spot,
        implied_forward,
        t_next_open,
        risk_free_rate,
        iv_shift=0.02,
    )
    worst_mtm_loss_rs = max(0.0, -min(pnl_down_pts, pnl_up_pts) * total_qty)
    if math.isfinite(contractual_max_loss_rs):
        return round(min(contractual_max_loss_rs, worst_mtm_loss_rs), 2)
    return round(worst_mtm_loss_rs, 2)


def compute_technical_options_score(
    *,
    structure_type: StructureKind,
    direction: DirectionType,
    is_net_credit: bool,
    dte_trading_sessions: int,
    legs: Sequence[OptionLeg],
    contractual_max_loss_rs: float,
    max_profit_rs: float,
    pop_lognormal_expiry: float,
    net_theta_daily_rs: float,
    net_vega_rs: float,
    structure_liquidity_score: float,
    spot: float,
    price_structure: PriceStructureEvidence,
    volatility: VolatilityEvidence,
    positioning: PositioningEvidence | None = None,
    triggered_detectors: Sequence[TriggeredDetector] = (),
) -> float:
    """Compute deterministic 0-100 Technical + Options Surface quality score.

    Driven strictly by completed-candle technical structure, RSI/ATR regime, IVP & IV-RV
    spread, 25-delta skew, OI wall protection, PoP, risk-reward, and book liquidity.
    Contains zero STT bias and zero Part B habitual-style bias.
    """
    score = 0.0

    # 1. Technical Trend, Detector & Momentum Alignment (0..35 pts)
    confirmed_dirs = {
        d.direction for d in triggered_detectors if d.confirmation_status == "CONFIRMED_QUANT_REGIME"
    }
    if direction in confirmed_dirs:
        score += 22.0
    elif direction == "BULLISH" and price_structure.trend_1d_state in ("UPTREND", "DOWNTREND_REPAIRED_HH_HL"):
        score += 18.0
    elif direction == "BEARISH" and price_structure.trend_1d_state == "DOWNTREND":
        score += 18.0
    elif direction == "NEUTRAL" and price_structure.trend_1d_state == "RANGE":
        score += 18.0

    last_1h = price_structure.last_completed_1h_bar
    rsi = price_structure.rsi_14_1h
    if direction == "BULLISH":
        if last_1h and last_1h.close >= price_structure.ma20_1h > 0:
            score += 5.0
        if 45.0 <= rsi <= 72.0:
            score += 4.0
        elif rsi > 78.0 and not is_net_credit:
            score -= 4.0  # Chasing overbought debit calls penalized
    elif direction == "BEARISH":
        if last_1h and price_structure.ma20_1h > 0 and last_1h.close <= price_structure.ma20_1h:
            score += 5.0
        if 28.0 <= rsi <= 55.0:
            score += 4.0
        elif rsi < 22.0 and not is_net_credit:
            score -= 4.0
    else:
        if 38.0 <= rsi <= 62.0 and abs(price_structure.stretch_zscore_1h) <= 1.5:
            score += 9.0

    for vt in price_structure.verified_level_touches:
        if vt.touched_geometrically and vt.wick_rejection_direction == direction:
            score += 4.0
            break

    # 2. Volatility Surface, IV-RV Spread, Skew & Term Structure Fit (0..30 pts)
    ivp_90 = volatility.ivp_90d
    if isinstance(ivp_90, (int, float)):
        if ivp_90 >= 50.0:
            if is_net_credit or net_vega_rs <= 0.0:
                score += 16.0
            elif structure_type in ("BULLISH_BUTTERFLY", "BEARISH_BUTTERFLY"):
                score += 12.0
        elif ivp_90 < 45.0:
            if not is_net_credit and net_vega_rs >= 0.0:
                score += 16.0
    if volatility.iv_rv_spread >= 0.015 and net_theta_daily_rs > 0.0:
        score += 4.0
    elif volatility.iv_rv_spread <= -0.005 and net_vega_rs > 0.0:
        score += 4.0

    skew = volatility.skew_25d_put_minus_call
    if skew >= 0.01:
        if structure_type in ("BULL_PUT_SPREAD", "IRON_CONDOR", "LEGGED_IRON_CONDOR_COMPONENT"):
            score += 5.0  # Harvesting rich downside put skew
        elif structure_type == "BULL_CALL_SPREAD":
            score += 4.0  # Financing upside call spread at relatively cheaper call IV
    elif skew <= -0.005 and structure_type in ("BEAR_CALL_SPREAD", "IRON_CONDOR"):
        score += 5.0

    if 3 <= dte_trading_sessions <= 12:
        score += 5.0
    elif dte_trading_sessions >= 2:
        score += 2.5

    # 3. Options Payoff Economics, OI Wall Protection & Liquidity (0..35 pts)
    rr = max_profit_rs / max(1.0, contractual_max_loss_rs)
    score += min(12.0, rr * 12.0)
    score += min(14.0, 0.16 * pop_lognormal_expiry)

    if positioning is not None:
        short_puts = [l.strike for l in legs if l.side == "SHORT" and l.option_type == "PE"]
        short_calls = [l.strike for l in legs if l.side == "SHORT" and l.option_type == "CE"]
        put_protected = (not short_puts) or max(short_puts) <= positioning.put_oi_wall_strike + 50.0
        call_protected = (not short_calls) or min(short_calls) >= positioning.call_oi_wall_strike - 50.0
        if (short_puts or short_calls) and put_protected and call_protected:
            score += 4.0

    score += min(5.0, 0.05 * structure_liquidity_score)
    return round(max(0.0, min(100.0, score)), 2)


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
    positioning: PositioningEvidence | None = None,
    triggered_detectors: Sequence[TriggeredDetector] = (),
    forced_lots: int | None = None,
) -> CandidateStructure:
    """Price and evaluate a defined-risk candidate structure with leg-wise piecewise max loss and Part A checks."""
    lot_size = expiry.lot_size

    # Boundary rejection: ratio spreads (unbalanced long/short totals per option type) are UNSUPPORTED.
    for opt_type in ("CE", "PE"):
        long_r = sum(l.ratio for l in legs if l.option_type == opt_type and l.side == "LONG")
        short_r = sum(l.ratio for l in legs if l.option_type == opt_type and l.side == "SHORT")
        if long_r != short_r:
            raise ValueError(
                f"UNSUPPORTED_RATIO_STRUCTURE: {opt_type} long ratio {long_r} != short ratio {short_r}; "
                "ratio / unbalanced structures are not supported by the pricing engine."
            )

    slip_per_leg = config.slippage_buffer_points_per_leg
    payoff_specs_1lot: list[LegPayoffSpec] = []
    signed_cashflow_pts = 0.0
    total_slip_pts = 0.0

    for leg in legs:
        if leg.side == "SHORT":
            exec_px = min(leg.bid, max(0.05, leg.bid - slip_per_leg))
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

    charges_1lot = compute_indian_statutory_charges_rs(
        legs, lot_size, 1, config.brokerage_per_order_rs
    )
    pw_1lot = compute_piecewise_expiry_payoff(
        payoff_specs_1lot, lot_size, 1, charges_1lot.total_charges_rs
    )

    if forced_lots is not None:
        lots = forced_lots
    else:
        lots = 1
        if (
            pw_1lot.is_defined_risk
            and 0.0 < pw_1lot.gross_max_loss_rs <= config.per_trade_contractual_max_loss_ceiling_rs
        ):
            max_lots_by_contractual = int(
                config.per_trade_contractual_max_loss_ceiling_rs // pw_1lot.gross_max_loss_rs
            )
            open_stops = sum(p.planned_stop_rs for p in portfolio_state.open_positions)
            rem_stop_budget = max(0.0, config.portfolio_summed_stop_cap_rs - open_stops)
            max_lots_by_stop = int(rem_stop_budget // config.per_position_managed_stop_rs)
            lots = max(1, min(max_lots_by_contractual, max_lots_by_stop, config.max_lots_per_trade))

    charges = compute_indian_statutory_charges_rs(
        legs, lot_size, lots, config.brokerage_per_order_rs
    )
    pw = compute_piecewise_expiry_payoff(
        payoff_specs_1lot, lot_size, lots, charges.total_charges_rs
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

    contractual_payoff_bound_rs = pw.gross_max_loss_rs

    planned_stop_rs = (
        min(config.per_position_managed_stop_rs, contractual_payoff_bound_rs)
        if contractual_payoff_bound_rs > 0
        else config.per_position_managed_stop_rs
    )
    planned_stop_rs = round(min(config.per_position_managed_stop_rs, planned_stop_rs), 2)
    stop_loss_pts_per_unit = planned_stop_rs / total_qty
    if is_net_credit:
        indicative_stop_spread_val = round(exec_net_pts + stop_loss_pts_per_unit, 2)
    else:
        indicative_stop_spread_val = round(max(0.0, exec_net_pts - stop_loss_pts_per_unit), 2)

    indicative_stop_index_lvl = _solve_indicative_stop_index_level(
        legs=legs,
        payoff_specs=payoff_specs_1lot,
        spot=spot,
        implied_forward=implied_forward,
        time_to_expiry_years=time_to_expiry_years,
        risk_free_rate=config.risk_free_rate_annual,
        stop_loss_pts_per_unit=stop_loss_pts_per_unit,
        net_delta=net_delta,
        invalidation_level=invalidation_level,
    )

    indicative_stop_note = (
        "INDICATIVE SUGGESTION-TIME BLACK-76 MTM MARK ONLY (model/IV/time-dependent). "
        "The operative Rs 2,000 stop is always computed from the actual executed fill "
        "(real fill prices and quantity) at execution time. Gaps and fast tapes "
        "can fill worse than the stop; the stop is not a guaranteed loss limit."
    )

    open_stops = [p.planned_stop_rs for p in portfolio_state.open_positions]
    summed_stop_after_trade = compute_portfolio_summed_stop_rs(open_stops, planned_stop_rs)

    margin_est_rs = estimate_exchange_hedged_margin_rs(
        spot, legs, contractual_payoff_bound_rs, lot_size, lots
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

    worst_gap_loss_rs = _compute_overnight_gap_stress_rs(
        legs=legs,
        payoff_specs=payoff_specs_1lot,
        spot=spot,
        implied_forward=implied_forward,
        time_to_expiry_years=time_to_expiry_years,
        risk_free_rate=config.risk_free_rate_annual,
        total_qty=total_qty,
        contractual_max_loss_rs=contractual_payoff_bound_rs,
    )

    mgmt_plan = (
        f"1. Exit immediately if position mark-to-market loss reaches Rs {planned_stop_rs:.0f} "
        f"(operative stop computed from actual fill; indicative spread value ~{indicative_stop_spread_val:.1f} pts, "
        f"Black-76 spot trigger ~{indicative_stop_index_lvl:.1f}). "
        f"2. Exit on completed-candle close beyond invalidation level {invalidation_level:.1f} ({invalidation_candle_basis}). "
        f"3. Review at 50-60% of max profit (Rs {0.55 * pw.max_profit_rs:.0f}). "
        f"4. Do not carry into 1 DTE session if any short leg has |delta| >= 0.25 (26 Oct 2026 rule)."
    )

    part_a_checks, part_a_overall = evaluate_part_a_rules(
        is_defined_risk=pw.is_defined_risk,
        legs=legs,
        dte_trading_sessions=expiry.dte_trading_sessions,
        planned_stop_rs=planned_stop_rs,
        portfolio_state=portfolio_state,
        contractual_max_loss_rs=contractual_payoff_bound_rs,
        invalidation_level=invalidation_level,
        invalidation_candle_basis=invalidation_candle_basis,
        invalidation_justification=invalidation_justification,
        price_structure=price_structure,
        volatility=volatility,
        config=config,
    )

    tech_opt_score = compute_technical_options_score(
        structure_type=structure_type,
        direction=direction,
        is_net_credit=is_net_credit,
        dte_trading_sessions=expiry.dte_trading_sessions,
        legs=legs,
        contractual_max_loss_rs=contractual_payoff_bound_rs,
        max_profit_rs=pw.max_profit_rs,
        pop_lognormal_expiry=pop_lognormal,
        net_theta_daily_rs=net_theta_daily_rs,
        net_vega_rs=net_vega_rs,
        structure_liquidity_score=structure_liq_score,
        spot=spot,
        price_structure=price_structure,
        volatility=volatility,
        positioning=positioning,
        triggered_detectors=triggered_detectors,
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
            f"LEG_WISE_PIECEWISE_EXPIRY_PAYOFF: exact contractual worst case "
            f"Rs {contractual_payoff_bound_rs:,.2f} at spot {pw.worst_case_spot:,.0f} "
            f"(non-STT execution fees Rs {charges.total_charges_rs:,.2f})"
        ),
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
        technical_options_score=tech_opt_score,
    )


def _find_priced_quote_by_strike(
    quotes: Sequence[RawStrikeQuote],
    target_strike: float,
    option_type: Literal["CE", "PE"] | Literal["BOTH"],
) -> RawStrikeQuote | None:
    """Locate the chain strike nearest `target_strike` and return it ONLY if its book is executable."""
    if not quotes:
        return None
    cand = min(quotes, key=lambda q: abs(q.strike - target_strike))
    if option_type == "CE":
        return cand if cand.has_executable_call else None
    if option_type == "PE":
        return cand if cand.has_executable_put else None
    return cand if cand.has_both_executable else None


def _find_quote_by_target_delta(
    quotes: Sequence[RawStrikeQuote],
    option_type: Literal["CE", "PE"],
    target_abs_delta: float,
    implied_forward: float,
    time_to_expiry_years: float,
    risk_free_rate: float,
) -> RawStrikeQuote | None:
    priced = [
        q
        for q in quotes
        if (q.has_executable_call if option_type == "CE" else q.has_executable_put)
    ]
    if not priced:
        return None
    best_q = priced[0]
    best_diff = 999.0
    for q in priced:
        mid = q.call_mid if option_type == "CE" else q.put_mid
        if mid is None:
            continue
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
    positioning: PositioningEvidence | None = None,
) -> list[CandidateStructure]:
    """Generate side-by-side defined-risk structures across Bullish, Bearish, and Neutral setups (Section 6.1).

    Generates credit spreads, debit spreads, iron condors, legged condor components,
    iron butterflies, and directional butterflies on equal footing.
    Strikes with `None` (unpriced/rejected) books are strictly skipped and never zero-filled.
    """
    if len(quotes) < 5:
        return []

    expiries_present = {q.expiry_date for q in quotes if q.expiry_date}
    if len(expiries_present) > 1:
        raise ValueError(
            f"UNSUPPORTED_MIXED_EXPIRY_CHAIN: quotes span expiries {sorted(expiries_present)}; "
            "calendar / mixed-expiry structures are not supported by the pricing engine."
        )

    stale_set = set(stale_strikes)
    pos = positioning or compute_positioning_evidence(expiry.expiry_date, quotes)

    bull_inv = price_structure.ema50_1h if price_structure.ema50_1h > 0 else (spot - 200.0)
    bull_just = "Completed daily close below 1H EMA50 / Put OI support invalidates bullish structure."
    bear_inv = (
        price_structure.last_completed_1d_bar.high
        if price_structure.last_completed_1d_bar
        else (spot + 200.0)
    )
    bear_just = "Completed daily close above prior 1D high invalidates bearish structure."
    neut_inv = pos.call_oi_wall_strike if pos.call_oi_wall_strike > spot else (spot + 300.0)
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

    common_kwargs = dict(
        expiry=expiry,
        spot=spot,
        implied_forward=implied_forward,
        atm_iv=atm_iv,
        time_to_expiry_years=time_to_expiry_years,
        empirical_daily_log_returns=empirical_daily_log_returns,
        portfolio_state=portfolio_state,
        price_structure=price_structure,
        volatility=volatility,
        config=config,
        positioning=pos,
        triggered_detectors=triggered_detectors,
    )

    candidates: list[CandidateStructure] = []

    # 1A & 1B. BULL PUT SPREAD (Credit, 200-pt & 150-pt wings)
    q_short_put_30 = _find_quote_by_target_delta(
        quotes, "PE", 0.30, implied_forward, time_to_expiry_years, config.risk_free_rate_annual
    )
    if q_short_put_30 is not None:
        q_long_put_200 = _find_priced_quote_by_strike(quotes, q_short_put_30.strike - 200.0, "PE")
        if q_long_put_200 is not None and q_long_put_200.strike < q_short_put_30.strike:
            candidates.append(
                build_candidate_structure(
                    structure_id=f"BPS_{expiry.expiry_date}_{int(q_short_put_30.strike)}_{int(q_long_put_200.strike)}",
                    structure_type="BULL_PUT_SPREAD",
                    direction="BULLISH",
                    legs=[
                        make_leg(q_short_put_30, "PE", "SHORT", 1),
                        make_leg(q_long_put_200, "PE", "LONG", 1),
                    ],
                    invalidation_level=bull_inv,
                    invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
                    invalidation_justification=bull_just,
                    matches_part_b_style=True,
                    **common_kwargs,
                )
            )

        q_long_put_150 = _find_priced_quote_by_strike(quotes, q_short_put_30.strike - 150.0, "PE")
        if (
            q_long_put_150 is not None
            and q_long_put_150.strike < q_short_put_30.strike
            and (q_long_put_200 is None or q_long_put_150.strike != q_long_put_200.strike)
        ):
            candidates.append(
                build_candidate_structure(
                    structure_id=f"BPS_{expiry.expiry_date}_{int(q_short_put_30.strike)}_{int(q_long_put_150.strike)}",
                    structure_type="BULL_PUT_SPREAD",
                    direction="BULLISH",
                    legs=[
                        make_leg(q_short_put_30, "PE", "SHORT", 1),
                        make_leg(q_long_put_150, "PE", "LONG", 1),
                    ],
                    invalidation_level=bull_inv,
                    invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
                    invalidation_justification=bull_just,
                    matches_part_b_style=True,
                    **common_kwargs,
                )
            )

    # 2. BULL CALL SPREAD (Debit, evaluated on equal footing)
    q_long_call_45 = _find_quote_by_target_delta(
        quotes, "CE", 0.45, implied_forward, time_to_expiry_years, config.risk_free_rate_annual
    )
    if q_long_call_45 is not None:
        q_short_call_up = _find_priced_quote_by_strike(
            quotes, q_long_call_45.strike + 200.0, "CE"
        ) or _find_priced_quote_by_strike(quotes, q_long_call_45.strike + 150.0, "CE")
        if q_short_call_up is not None and q_short_call_up.strike > q_long_call_45.strike:
            candidates.append(
                build_candidate_structure(
                    structure_id=f"BCS_{expiry.expiry_date}_{int(q_long_call_45.strike)}_{int(q_short_call_up.strike)}",
                    structure_type="BULL_CALL_SPREAD",
                    direction="BULLISH",
                    legs=[
                        make_leg(q_long_call_45, "CE", "LONG", 1),
                        make_leg(q_short_call_up, "CE", "SHORT", 1),
                    ],
                    invalidation_level=bull_inv,
                    invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
                    invalidation_justification=bull_just,
                    matches_part_b_style=False,
                    **common_kwargs,
                )
            )

    # 3. BULLISH CALL BUTTERFLY (1-2-1 Call Butterfly above spot)
    q_bfly_mid = _find_priced_quote_by_strike(quotes, spot + 150.0, "CE")
    if q_bfly_mid is not None:
        q_bfly_low = _find_priced_quote_by_strike(quotes, q_bfly_mid.strike - 150.0, "CE")
        q_bfly_high = _find_priced_quote_by_strike(quotes, q_bfly_mid.strike + 150.0, "CE")
        if (
            q_bfly_low is not None
            and q_bfly_high is not None
            and q_bfly_low.strike < q_bfly_mid.strike < q_bfly_high.strike
        ):
            candidates.append(
                build_candidate_structure(
                    structure_id=f"BULL_BFLY_{expiry.expiry_date}_{int(q_bfly_low.strike)}_{int(q_bfly_mid.strike)}_{int(q_bfly_high.strike)}",
                    structure_type="BULLISH_BUTTERFLY",
                    direction="BULLISH",
                    legs=[
                        make_leg(q_bfly_low, "CE", "LONG", 1),
                        make_leg(q_bfly_mid, "CE", "SHORT", 2),
                        make_leg(q_bfly_high, "CE", "LONG", 1),
                    ],
                    invalidation_level=bull_inv,
                    invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
                    invalidation_justification=bull_just,
                    matches_part_b_style=False,
                    **common_kwargs,
                )
            )

    # 4A & 4B. BEAR CALL SPREAD (Credit, 150-pt & 200-pt wings)
    q_short_call_30 = _find_quote_by_target_delta(
        quotes, "CE", 0.30, implied_forward, time_to_expiry_years, config.risk_free_rate_annual
    )
    if q_short_call_30 is not None:
        q_long_call_150 = _find_priced_quote_by_strike(quotes, q_short_call_30.strike + 150.0, "CE")
        if q_long_call_150 is not None and q_long_call_150.strike > q_short_call_30.strike:
            candidates.append(
                build_candidate_structure(
                    structure_id=f"BEAR_CS_{expiry.expiry_date}_{int(q_short_call_30.strike)}_{int(q_long_call_150.strike)}",
                    structure_type="BEAR_CALL_SPREAD",
                    direction="BEARISH",
                    legs=[
                        make_leg(q_short_call_30, "CE", "SHORT", 1),
                        make_leg(q_long_call_150, "CE", "LONG", 1),
                    ],
                    invalidation_level=bear_inv,
                    invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
                    invalidation_justification=bear_just,
                    matches_part_b_style=True,
                    **common_kwargs,
                )
            )

        q_long_call_200 = _find_priced_quote_by_strike(quotes, q_short_call_30.strike + 200.0, "CE")
        if (
            q_long_call_200 is not None
            and q_long_call_200.strike > q_short_call_30.strike
            and (q_long_call_150 is None or q_long_call_200.strike != q_long_call_150.strike)
        ):
            candidates.append(
                build_candidate_structure(
                    structure_id=f"BEAR_CS_{expiry.expiry_date}_{int(q_short_call_30.strike)}_{int(q_long_call_200.strike)}",
                    structure_type="BEAR_CALL_SPREAD",
                    direction="BEARISH",
                    legs=[
                        make_leg(q_short_call_30, "CE", "SHORT", 1),
                        make_leg(q_long_call_200, "CE", "LONG", 1),
                    ],
                    invalidation_level=bear_inv,
                    invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
                    invalidation_justification=bear_just,
                    matches_part_b_style=True,
                    **common_kwargs,
                )
            )

    # 5. BEAR PUT SPREAD (Debit, evaluated on equal footing)
    q_long_put_45 = _find_quote_by_target_delta(
        quotes, "PE", 0.45, implied_forward, time_to_expiry_years, config.risk_free_rate_annual
    )
    if q_long_put_45 is not None:
        q_short_put_dn = _find_priced_quote_by_strike(quotes, q_long_put_45.strike - 150.0, "PE")
        if q_short_put_dn is not None and q_short_put_dn.strike < q_long_put_45.strike:
            candidates.append(
                build_candidate_structure(
                    structure_id=f"BEAR_PS_{expiry.expiry_date}_{int(q_long_put_45.strike)}_{int(q_short_put_dn.strike)}",
                    structure_type="BEAR_PUT_SPREAD",
                    direction="BEARISH",
                    legs=[
                        make_leg(q_long_put_45, "PE", "LONG", 1),
                        make_leg(q_short_put_dn, "PE", "SHORT", 1),
                    ],
                    invalidation_level=bear_inv,
                    invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
                    invalidation_justification=bear_just,
                    matches_part_b_style=False,
                    **common_kwargs,
                )
            )

    # 6. BEARISH PUT BUTTERFLY (1-2-1 Put Butterfly below spot)
    q_pbfly_mid = _find_priced_quote_by_strike(quotes, spot - 150.0, "PE")
    if q_pbfly_mid is not None:
        q_pbfly_low = _find_priced_quote_by_strike(quotes, q_pbfly_mid.strike - 150.0, "PE")
        q_pbfly_high = _find_priced_quote_by_strike(quotes, q_pbfly_mid.strike + 150.0, "PE")
        if (
            q_pbfly_low is not None
            and q_pbfly_high is not None
            and q_pbfly_low.strike < q_pbfly_mid.strike < q_pbfly_high.strike
        ):
            candidates.append(
                build_candidate_structure(
                    structure_id=f"BEAR_BFLY_{expiry.expiry_date}_{int(q_pbfly_low.strike)}_{int(q_pbfly_mid.strike)}_{int(q_pbfly_high.strike)}",
                    structure_type="BEARISH_BUTTERFLY",
                    direction="BEARISH",
                    legs=[
                        make_leg(q_pbfly_high, "PE", "LONG", 1),
                        make_leg(q_pbfly_mid, "PE", "SHORT", 2),
                        make_leg(q_pbfly_low, "PE", "LONG", 1),
                    ],
                    invalidation_level=bear_inv,
                    invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
                    invalidation_justification=bear_just,
                    matches_part_b_style=False,
                    **common_kwargs,
                )
            )

    # 7. IRON CONDOR (Neutral: ~0.16 delta shorts per side, 150-pt wings)
    q_ic_short_put = _find_quote_by_target_delta(
        quotes, "PE", 0.16, implied_forward, time_to_expiry_years, config.risk_free_rate_annual
    )
    q_ic_short_call = _find_quote_by_target_delta(
        quotes, "CE", 0.16, implied_forward, time_to_expiry_years, config.risk_free_rate_annual
    )
    if q_ic_short_put is not None and q_ic_short_call is not None:
        q_ic_long_put = _find_priced_quote_by_strike(quotes, q_ic_short_put.strike - 150.0, "PE")
        q_ic_long_call = _find_priced_quote_by_strike(quotes, q_ic_short_call.strike + 150.0, "CE")
        if (
            q_ic_long_put is not None
            and q_ic_long_call is not None
            and q_ic_long_put.strike < q_ic_short_put.strike < q_ic_short_call.strike < q_ic_long_call.strike
        ):
            candidates.append(
                build_candidate_structure(
                    structure_id=(
                        f"IC_{expiry.expiry_date}_{int(q_ic_long_put.strike)}_"
                        f"{int(q_ic_short_put.strike)}_{int(q_ic_short_call.strike)}_{int(q_ic_long_call.strike)}"
                    ),
                    structure_type="IRON_CONDOR",
                    direction="NEUTRAL",
                    legs=[
                        make_leg(q_ic_long_put, "PE", "LONG", 1),
                        make_leg(q_ic_short_put, "PE", "SHORT", 1),
                        make_leg(q_ic_short_call, "CE", "SHORT", 1),
                        make_leg(q_ic_long_call, "CE", "LONG", 1),
                    ],
                    invalidation_level=neut_inv,
                    invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
                    invalidation_justification=neut_just,
                    matches_part_b_style=True,
                    **common_kwargs,
                )
            )

    # 8. IRON BUTTERFLY (Neutral high-theta pin around synthetic forward, 150-pt wings)
    q_ib_atm = _find_priced_quote_by_strike(quotes, implied_forward, "BOTH")
    if q_ib_atm is not None:
        q_ib_low_put = _find_priced_quote_by_strike(quotes, q_ib_atm.strike - 150.0, "PE")
        q_ib_high_call = _find_priced_quote_by_strike(quotes, q_ib_atm.strike + 150.0, "CE")
        if (
            q_ib_low_put is not None
            and q_ib_high_call is not None
            and q_ib_low_put.strike < q_ib_atm.strike < q_ib_high_call.strike
        ):
            candidates.append(
                build_candidate_structure(
                    structure_id=(
                        f"IBFLY_{expiry.expiry_date}_{int(q_ib_low_put.strike)}_"
                        f"{int(q_ib_atm.strike)}_{int(q_ib_high_call.strike)}"
                    ),
                    structure_type="IRON_BUTTERFLY",
                    direction="NEUTRAL",
                    legs=[
                        make_leg(q_ib_low_put, "PE", "LONG", 1),
                        make_leg(q_ib_atm, "PE", "SHORT", 1),
                        make_leg(q_ib_atm, "CE", "SHORT", 1),
                        make_leg(q_ib_high_call, "CE", "LONG", 1),
                    ],
                    invalidation_level=neut_inv,
                    invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
                    invalidation_justification=neut_just,
                    matches_part_b_style=False,
                    **common_kwargs,
                )
            )

    # 9. LEGGED IRON CONDOR COMPONENT (OI-wall-anchored ~0.20 delta wing vertical)
    legging_bull = price_structure.trend_1d_state != "DOWNTREND"
    if legging_bull:
        q_leg_sp = _find_quote_by_target_delta(
            quotes, "PE", 0.20, implied_forward, time_to_expiry_years, config.risk_free_rate_annual
        )
        if q_leg_sp is not None:
            q_leg_lp = _find_priced_quote_by_strike(quotes, q_leg_sp.strike - 150.0, "PE")
            if q_leg_lp is not None and q_leg_lp.strike < q_leg_sp.strike:
                candidates.append(
                    build_candidate_structure(
                        structure_id=f"LEG_IC_PE_{expiry.expiry_date}_{int(q_leg_sp.strike)}_{int(q_leg_lp.strike)}",
                        structure_type="LEGGED_IRON_CONDOR_COMPONENT",
                        direction="BULLISH",
                        legs=[
                            make_leg(q_leg_sp, "PE", "SHORT", 1),
                            make_leg(q_leg_lp, "PE", "LONG", 1),
                        ],
                        invalidation_level=bull_inv,
                        invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
                        invalidation_justification=bull_just,
                        matches_part_b_style=True,
                        **common_kwargs,
                    )
                )
    else:
        q_leg_sc = _find_quote_by_target_delta(
            quotes, "CE", 0.20, implied_forward, time_to_expiry_years, config.risk_free_rate_annual
        )
        if q_leg_sc is not None:
            q_leg_lc = _find_priced_quote_by_strike(quotes, q_leg_sc.strike + 150.0, "CE")
            if q_leg_lc is not None and q_leg_lc.strike > q_leg_sc.strike:
                candidates.append(
                    build_candidate_structure(
                        structure_id=f"LEG_IC_CE_{expiry.expiry_date}_{int(q_leg_sc.strike)}_{int(q_leg_lc.strike)}",
                        structure_type="LEGGED_IRON_CONDOR_COMPONENT",
                        direction="BEARISH",
                        legs=[
                            make_leg(q_leg_sc, "CE", "SHORT", 1),
                            make_leg(q_leg_lc, "CE", "LONG", 1),
                        ],
                        invalidation_level=bear_inv,
                        invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
                        invalidation_justification=bear_just,
                        matches_part_b_style=True,
                        **common_kwargs,
                    )
                )

    return candidates
