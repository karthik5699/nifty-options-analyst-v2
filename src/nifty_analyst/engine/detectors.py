"""Completed-candle signal hygiene, technical indicators, and Two-Track Setup Detector Registry (Section 5).

- Track 1 (PART_C_HISTORICAL): tagged HISTORICAL_REFERENCE_UNCONFIRMED; context only, never gates.
- Track 2 (NOVEL_QUANT): symmetric Bullish, Bearish, Neutral, and Skew regime detectors
  (CONFIRMED_QUANT_REGIME) plus candidate stretch metrics tagged DEFINITION_OPEN.
"""

from __future__ import annotations

import math
from typing import Literal, Sequence

from nifty_analyst.state import (
    CandleBar,
    PositioningEvidence,
    PriceStructureEvidence,
    TriggeredDetector,
    VerifiedTouch,
    VolatilityEvidence,
)


def compute_sma(values: Sequence[float], period: int) -> float:
    if not values:
        return 0.0
    window = values[-period:] if len(values) >= period else values
    return sum(window) / float(len(window))


def compute_ema(values: Sequence[float], period: int) -> float:
    if not values:
        return 0.0
    k = 2.0 / (float(period) + 1.0)
    ema = values[0]
    for v in values[1:]:
        ema = v * k + ema * (1.0 - k)
    return ema


def compute_rsi_14(closes: Sequence[float], period: int = 14) -> float:
    """Compute Wilder's 14-period RSI on completed candle closes."""
    if len(closes) < 2:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    seed_len = min(period, len(deltas))
    gains = [max(0.0, d) for d in deltas[:seed_len]]
    losses = [max(0.0, -d) for d in deltas[:seed_len]]
    avg_gain = sum(gains) / float(seed_len)
    avg_loss = sum(losses) / float(seed_len)

    for d in deltas[seed_len:]:
        gain = max(0.0, d)
        loss = max(0.0, -d)
        avg_gain = (avg_gain * (period - 1) + gain) / float(period)
        avg_loss = (avg_loss * (period - 1) + loss) / float(period)

    if avg_loss <= 1e-9:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 2)


def compute_atr_14(bars: Sequence[CandleBar], period: int = 14) -> float:
    """Compute Wilder's 14-period Average True Range (ATR) in index points on completed bars."""
    if not bars:
        return 0.0
    true_ranges: list[float] = [bars[0].high - bars[0].low]
    for i in range(1, len(bars)):
        prev_close = bars[i - 1].close
        b = bars[i]
        tr = max(b.high - b.low, abs(b.high - prev_close), abs(b.low - prev_close))
        true_ranges.append(tr)

    seed_len = min(period, len(true_ranges))
    atr = sum(true_ranges[:seed_len]) / float(seed_len)
    for tr in true_ranges[seed_len:]:
        atr = (atr * (period - 1) + tr) / float(period)
    return round(atr, 2)


def compute_stretch_zscore(
    reference_price: float,
    closes: Sequence[float],
    period: int = 20,
) -> float:
    """Compute standardized distance `(price - SMA_20) / stdev_20` on completed bars."""
    if len(closes) < 3:
        return 0.0
    window = closes[-period:] if len(closes) >= period else closes
    mean_v = sum(window) / float(len(window))
    var_v = sum((x - mean_v) ** 2 for x in window) / float(max(1, len(window) - 1))
    sd = math.sqrt(var_v)
    if sd <= 1e-6:
        return 0.0
    return round((reference_price - mean_v) / sd, 2)


def verify_geometric_touch(
    bar: CandleBar,
    level_name: str,
    level_value: float,
    tolerance_points: float = 0.0,
) -> VerifiedTouch:
    """Geometrically verify whether a completed bar's [low, high] interval intersected level_value.

    Per Section 5.1: 'a geometric check that the completed bar's [low, high] interval
    intersected the reference line, with zero tolerance unless the trader settles one.'
    """
    touched = (bar.low - tolerance_points) <= level_value <= (bar.high + tolerance_points)
    body_low = min(bar.open, bar.close)
    body_high = max(bar.open, bar.close)

    wick_dir: Literal["BULLISH", "BEARISH", "NONE"] = "NONE"
    if touched:
        if bar.close > level_value and bar.low <= level_value <= body_low:
            wick_dir = "BULLISH"
        elif bar.close < level_value and body_high <= level_value <= bar.high:
            wick_dir = "BEARISH"
        elif bar.close >= level_value:
            wick_dir = "BULLISH"
        else:
            wick_dir = "BEARISH"

    return VerifiedTouch(
        level_name=level_name,
        level_value=round(level_value, 2),
        bar_timeframe=bar.timeframe,
        bar_close_time_ist=bar.close_time_ist,
        bar_low=bar.low,
        bar_high=bar.high,
        bar_close=bar.close,
        touched_geometrically=touched,
        wick_rejection_direction=wick_dir,
    )


def analyze_price_structure(
    completed_1d: Sequence[CandleBar],
    completed_1h: Sequence[CandleBar],
    completed_final_15m: CandleBar | None,
    stripped_forming_bars_count: int,
    extra_reference_levels: dict[str, float] | None = None,
) -> PriceStructureEvidence:
    """Analyze completed 1D and 1H bars for trend state, HH+HL repair, MAs, RSI, ATR, and touches."""
    closes_1h = [b.close for b in completed_1h]
    ma20_1h = round(compute_sma(closes_1h, 20), 2) if closes_1h else 0.0
    ema50_1h = round(compute_ema(closes_1h, 50), 2) if closes_1h else 0.0
    rsi_14_1h = compute_rsi_14(closes_1h, 14)
    atr_14_1h = compute_atr_14(completed_1h, 14)
    atr_14_1d = compute_atr_14(completed_1d, 14)
    stretch_z_1h = (
        compute_stretch_zscore(closes_1h[-1], closes_1h, 20) if closes_1h else 0.0
    )

    # Evaluate 1D trend state and 8 Oct 2026 downtrend repair rule (daily higher high + higher low)
    trend_1d: Literal["UPTREND", "DOWNTREND", "RANGE", "DOWNTREND_REPAIRED_HH_HL"] = "RANGE"
    hh_hl_repaired = False
    first_pullback_warning = False

    if len(completed_1d) >= 4:
        b_minus3, b_minus2, b_minus1, b_last = completed_1d[-4:]
        recent_hh = b_last.high > b_minus1.high
        recent_hl = b_last.low > b_minus1.low
        prior_lower_highs = b_minus1.high < b_minus2.high and b_minus2.high <= b_minus3.high

        if recent_hh and recent_hl and b_last.close > b_minus2.high:
            if prior_lower_highs:
                trend_1d = "DOWNTREND_REPAIRED_HH_HL"
            else:
                trend_1d = "UPTREND"
            hh_hl_repaired = True
        elif b_last.high < b_minus1.high and b_last.low < b_minus1.low:
            trend_1d = "DOWNTREND"
            hh_hl_repaired = False
        elif prior_lower_highs and not (recent_hh and recent_hl):
            trend_1d = "DOWNTREND"
            hh_hl_repaired = False
            first_pullback_warning = True
        else:
            trend_1d = "RANGE"
            hh_hl_repaired = recent_hh and recent_hl

    last_1d = completed_1d[-1] if completed_1d else None
    last_1h = completed_1h[-1] if completed_1h else None

    verified_touches: list[VerifiedTouch] = []
    if last_1h is not None:
        ref_levels: dict[str, float] = {
            "1H_MA20": ma20_1h,
            "1H_EMA50": ema50_1h,
        }
        if last_1d is not None:
            ref_levels["PREV_1D_HIGH"] = last_1d.high
            ref_levels["PREV_1D_LOW"] = last_1d.low
        if extra_reference_levels:
            ref_levels.update(extra_reference_levels)

        for lvl_name, lvl_val in ref_levels.items():
            if lvl_val > 0:
                vt = verify_geometric_touch(last_1h, lvl_name, lvl_val, tolerance_points=0.0)
                verified_touches.append(vt)

    return PriceStructureEvidence(
        trend_1d_state=trend_1d,
        higher_high_higher_low_repaired=hh_hl_repaired,
        first_pullback_in_downtrend_warning=first_pullback_warning,
        ma20_1h=ma20_1h,
        ema50_1h=ema50_1h,
        last_completed_1d_bar=last_1d,
        last_completed_1h_bar=last_1h,
        final_15m_bar_completed=completed_final_15m is not None,
        stripped_forming_bars_count=stripped_forming_bars_count,
        verified_level_touches=verified_touches,
        rsi_14_1h=rsi_14_1h,
        atr_14_1h=atr_14_1h,
        atr_14_1d=atr_14_1d,
        stretch_zscore_1h=stretch_z_1h,
    )


def run_two_track_detectors(
    spot: float,
    price_structure: PriceStructureEvidence,
    volatility: VolatilityEvidence,
    positioning: PositioningEvidence,
) -> list[TriggeredDetector]:
    """Run both Track 1 (Historical Reference) and Track 2 (Novel Quant Regime) setup detectors."""
    detectors: list[TriggeredDetector] = []
    last_1h = price_structure.last_completed_1h_bar
    last_1d = price_structure.last_completed_1d_bar
    if last_1h is None:
        return detectors

    touch_by_name = {t.level_name: t for t in price_structure.verified_level_touches}
    t_ma20 = touch_by_name.get("1H_MA20")
    t_ema50 = touch_by_name.get("1H_EMA50")

    # -------------------------------------------------------------------------
    # TRACK 1: Historical Reference Detectors (Context Only, Never Gates)
    # -------------------------------------------------------------------------
    if t_ma20 and t_ma20.touched_geometrically:
        direction = "BULLISH" if t_ma20.wick_rejection_direction == "BULLISH" else "BEARISH"
        inv_level = (
            (last_1d.low if last_1d else last_1h.low - 50.0)
            if direction == "BULLISH"
            else (last_1d.high if last_1d else last_1h.high + 50.0)
        )
        detectors.append(
            TriggeredDetector(
                detector_id="HIST_1H_MA20_TOUCH",
                detector_name="1H MA20 Completed-Bar Wick Touch (Historical Variant)",
                track="PART_C_HISTORICAL",
                confirmation_status="HISTORICAL_REFERENCE_UNCONFIRMED",
                direction=direction,
                invalidation_level=round(inv_level, 2),
                invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
                invalidation_justification=(
                    f"Completed daily close beyond {round(inv_level, 2)} invalidates the 1H MA20 "
                    f"({price_structure.ma20_1h}) rejection structure."
                ),
                summary=(
                    f"Completed 1H bar ({last_1h.close_time_ist}) geometrically touched 1H MA20 "
                    f"({price_structure.ma20_1h}) with {t_ma20.wick_rejection_direction} close ({last_1h.close}). "
                    f"Tagged HISTORICAL_REFERENCE_UNCONFIRMED (context only, never blocks or gates)."
                ),
                evidence_fields_cited=[
                    "market_evidence.price_structure.ma20_1h",
                    "market_evidence.price_structure.last_completed_1h_bar.low",
                    "market_evidence.price_structure.last_completed_1h_bar.high",
                    "market_evidence.price_structure.last_completed_1h_bar.close",
                ],
            )
        )

    if (
        t_ma20
        and t_ema50
        and t_ma20.touched_geometrically
        and t_ema50.touched_geometrically
    ):
        direction = "BULLISH" if last_1h.close >= price_structure.ema50_1h else "BEARISH"
        inv_level = (
            min(last_1h.low, price_structure.ema50_1h - 40.0)
            if direction == "BULLISH"
            else max(last_1h.high, price_structure.ema50_1h + 40.0)
        )
        detectors.append(
            TriggeredDetector(
                detector_id="HIST_1H_MA20_EMA50_CONFLUENCE",
                detector_name="1H MA20 + EMA50 Dual Touch Confluence (Historical Variant)",
                track="PART_C_HISTORICAL",
                confirmation_status="HISTORICAL_REFERENCE_UNCONFIRMED",
                direction=direction,
                invalidation_level=round(inv_level, 2),
                invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
                invalidation_justification=(
                    f"Completed daily close beyond {round(inv_level, 2)} breaks the dual MA20/EMA50 confluence."
                ),
                summary=(
                    f"Completed 1H bar touched both 1H MA20 ({price_structure.ma20_1h}) and "
                    f"1H EMA50 ({price_structure.ema50_1h}). Tagged HISTORICAL_REFERENCE_UNCONFIRMED."
                ),
                evidence_fields_cited=[
                    "market_evidence.price_structure.ma20_1h",
                    "market_evidence.price_structure.ema50_1h",
                ],
            )
        )

    if price_structure.first_pullback_in_downtrend_warning:
        inv_level = last_1d.high if last_1d else spot + 150.0
        detectors.append(
            TriggeredDetector(
                detector_id="HIST_DOWNTREND_FIRST_PULLBACK_NOTE",
                detector_name="Downtrend Unrepaired Pullback Context (8 Oct 2026 Note)",
                track="PART_C_HISTORICAL",
                confirmation_status="HISTORICAL_REFERENCE_UNCONFIRMED",
                direction="BEARISH",
                invalidation_level=round(inv_level, 2),
                invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
                invalidation_justification=(
                    f"Completed daily close above prior swing high {round(inv_level, 2)} repairs downtrend."
                ),
                summary=(
                    "1D structure is in downtrend without daily higher-high + higher-low repair "
                    "(per 8 Oct 2026 note: 'never buy on the first pullback; rather, sell')."
                ),
                evidence_fields_cited=[
                    "market_evidence.price_structure.trend_1d_state",
                    "market_evidence.price_structure.higher_high_higher_low_repaired",
                ],
            )
        )

    # -------------------------------------------------------------------------
    # TRACK 2: Symmetric Novel & Regime-Driven Detectors (First-Class Citizens)
    # -------------------------------------------------------------------------
    ivp_90 = volatility.ivp_90d
    numeric_ivp = float(ivp_90) if isinstance(ivp_90, (int, float)) else 50.0

    # 2A: High-IV Bullish Regime + Put OI Wall Support
    if numeric_ivp >= 50.0 and price_structure.trend_1d_state in ("UPTREND", "DOWNTREND_REPAIRED_HH_HL"):
        inv_level = (
            float(positioning.put_oi_wall_strike)
            if positioning.put_oi_wall_strike < spot
            else (spot - 200.0)
        )
        detectors.append(
            TriggeredDetector(
                detector_id="NOVEL_HIGH_IV_BULLISH_PUT_WALL",
                detector_name="Elevated IVP + Repaired Uptrend Above Put OI Wall",
                track="NOVEL_QUANT",
                confirmation_status="CONFIRMED_QUANT_REGIME",
                direction="BULLISH",
                invalidation_level=round(inv_level, 2),
                invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
                invalidation_justification=(
                    f"Completed daily close below Put OI Wall ({round(inv_level, 2)}) invalidates bullish support."
                ),
                summary=(
                    f"90d IVP is {ivp_90} with 1D trend {price_structure.trend_1d_state}, "
                    f"1H RSI={price_structure.rsi_14_1h:.1f}, and Put OI wall at {positioning.put_oi_wall_strike} "
                    f"({positioning.put_oi_wall_contracts} contracts). Favours short-vega bullish credit or butterfly structures."
                ),
                evidence_fields_cited=[
                    "market_evidence.volatility_state.ivp_90d",
                    "market_evidence.price_structure.trend_1d_state",
                    "market_evidence.derivatives_positioning.put_oi_wall_strike",
                ],
            )
        )

    # 2B: Low-IV Bullish Compression Regime
    if numeric_ivp < 50.0 and price_structure.trend_1d_state in ("UPTREND", "DOWNTREND_REPAIRED_HH_HL"):
        inv_level = price_structure.ema50_1h if price_structure.ema50_1h < spot else (spot - 150.0)
        detectors.append(
            TriggeredDetector(
                detector_id="NOVEL_LOW_IV_COMPRESSION_BULLISH",
                detector_name="Low-IVP Volatility Compression Bullish Setup (Debit-Favourable)",
                track="NOVEL_QUANT",
                confirmation_status="CONFIRMED_QUANT_REGIME",
                direction="BULLISH",
                invalidation_level=round(inv_level, 2),
                invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
                invalidation_justification=(
                    f"Completed daily close below {round(inv_level, 2)} negates upside expansion thesis."
                ),
                summary=(
                    f"90d IVP is compressed ({ivp_90}) with positive trend structure (1H RSI={price_structure.rsi_14_1h:.1f}); "
                    f"long-vega debit structures (Bull Call Spread) avoid short-vega expansion risk."
                ),
                evidence_fields_cited=[
                    "market_evidence.volatility_state.ivp_90d",
                    "market_evidence.volatility_state.iv_rv_spread",
                ],
            )
        )

    # 2C: High-IV Bearish Regime + Call OI Wall Resistance (Symmetric Bearish Detector)
    if numeric_ivp >= 50.0 and price_structure.trend_1d_state == "DOWNTREND":
        inv_level = (
            float(positioning.call_oi_wall_strike)
            if positioning.call_oi_wall_strike > spot
            else (last_1d.high if last_1d and last_1d.high > spot else spot + 200.0)
        )
        detectors.append(
            TriggeredDetector(
                detector_id="NOVEL_HIGH_IV_BEARISH_CALL_WALL",
                detector_name="Elevated IVP + Downtrend Below Call OI Wall",
                track="NOVEL_QUANT",
                confirmation_status="CONFIRMED_QUANT_REGIME",
                direction="BEARISH",
                invalidation_level=round(inv_level, 2),
                invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
                invalidation_justification=(
                    f"Completed daily close above Call OI Wall / swing resistance ({round(inv_level, 2)}) invalidates bearish thesis."
                ),
                summary=(
                    f"90d IVP is {ivp_90} with 1D trend DOWNTREND (1H RSI={price_structure.rsi_14_1h:.1f}) "
                    f"capped below Call OI wall at {positioning.call_oi_wall_strike} "
                    f"({positioning.call_oi_wall_contracts} contracts). Favours Bear Call Spread or Bearish Butterfly."
                ),
                evidence_fields_cited=[
                    "market_evidence.volatility_state.ivp_90d",
                    "market_evidence.price_structure.trend_1d_state",
                    "market_evidence.derivatives_positioning.call_oi_wall_strike",
                ],
            )
        )

    # 2D: Low-IV Bearish Compression Regime (Symmetric Bearish Debit Detector)
    if numeric_ivp < 50.0 and price_structure.trend_1d_state == "DOWNTREND":
        inv_level = (
            price_structure.ema50_1h
            if price_structure.ema50_1h > spot
            else (last_1d.high if last_1d and last_1d.high > spot else spot + 150.0)
        )
        detectors.append(
            TriggeredDetector(
                detector_id="NOVEL_LOW_IV_COMPRESSION_BEARISH",
                detector_name="Low-IVP Volatility Compression Bearish Setup (Debit-Favourable)",
                track="NOVEL_QUANT",
                confirmation_status="CONFIRMED_QUANT_REGIME",
                direction="BEARISH",
                invalidation_level=round(inv_level, 2),
                invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
                invalidation_justification=(
                    f"Completed daily close above {round(inv_level, 2)} negates downside breakdown thesis."
                ),
                summary=(
                    f"90d IVP is compressed ({ivp_90}) in an active 1D DOWNTREND; "
                    f"long-vega Bear Put Spread captures downside convex expansion without short-vol exposure."
                ),
                evidence_fields_cited=[
                    "market_evidence.volatility_state.ivp_90d",
                    "market_evidence.price_structure.trend_1d_state",
                ],
            )
        )

    # 2E: High-IV Range-Bound OI-Bracketed Regime (Favours Iron Condor / Iron Butterfly)
    if (
        numeric_ivp >= 55.0
        and positioning.put_oi_wall_strike < spot < positioning.call_oi_wall_strike
    ):
        detectors.append(
            TriggeredDetector(
                detector_id="NOVEL_OI_BRACKETED_NEUTRAL_RANGE",
                detector_name="High-IV OI-Wall Bracketed Neutral Regime",
                track="NOVEL_QUANT",
                confirmation_status="CONFIRMED_QUANT_REGIME",
                direction="NEUTRAL",
                invalidation_level=round(positioning.call_oi_wall_strike, 2),
                invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
                invalidation_justification=(
                    f"Completed daily close outside [{positioning.put_oi_wall_strike}, "
                    f"{positioning.call_oi_wall_strike}] OI walls invalidates neutral range."
                ),
                summary=(
                    f"Spot ({spot}) is bracketed between Put OI wall ({positioning.put_oi_wall_strike}) "
                    f"and Call OI wall ({positioning.call_oi_wall_strike}) with 90d IVP={ivp_90} and "
                    f"1H RSI={price_structure.rsi_14_1h:.1f}."
                ),
                evidence_fields_cited=[
                    "market_evidence.derivatives_positioning.put_oi_wall_strike",
                    "market_evidence.derivatives_positioning.call_oi_wall_strike",
                    "market_evidence.volatility_state.ivp_90d",
                ],
            )
        )

    # 2F: 25-Delta Skew Dislocation Detector
    skew_pts = volatility.skew_25d_put_minus_call * 100.0
    if abs(skew_pts) >= 2.5:
        skew_dir: Literal["BULLISH", "BEARISH"] = "BULLISH" if skew_pts > 0 else "BEARISH"
        inv_lvl = (
            float(positioning.put_oi_wall_strike)
            if skew_dir == "BULLISH" and positioning.put_oi_wall_strike < spot
            else (
                float(positioning.call_oi_wall_strike)
                if skew_dir == "BEARISH" and positioning.call_oi_wall_strike > spot
                else (spot - 200.0 if skew_dir == "BULLISH" else spot + 200.0)
            )
        )
        detectors.append(
            TriggeredDetector(
                detector_id="NOVEL_SKEW_DISLOCATION",
                detector_name="25-Delta Put-Call Skew Richness Dislocation",
                track="NOVEL_QUANT",
                confirmation_status="CONFIRMED_QUANT_REGIME",
                direction=skew_dir,
                invalidation_level=round(inv_lvl, 2),
                invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
                invalidation_justification=(
                    f"Completed daily close beyond {round(inv_lvl, 2)} invalidates skew-harvest thesis."
                ),
                summary=(
                    f"25-delta Put-Call skew is {skew_pts:+.2f} vol pts, indicating elevated "
                    f"{'downside put' if skew_pts > 0 else 'upside call'} wing richness relative to the opposite wing."
                ),
                evidence_fields_cited=[
                    "market_evidence.volatility_state.skew_25d_put_minus_call",
                ],
            )
        )

    # 2G: Candidate Stretch Metric (Tagged DEFINITION_OPEN per Section 5.2)
    if price_structure.ma20_1h > 0:
        dist_pts = abs(spot - price_structure.ma20_1h)
        if dist_pts >= 250.0 or abs(price_structure.stretch_zscore_1h) >= 2.2:
            direction = "BEARISH" if spot > price_structure.ma20_1h else "BULLISH"
            inv_level = (spot + 150.0) if direction == "BEARISH" else (spot - 150.0)
            detectors.append(
                TriggeredDetector(
                    detector_id="NOVEL_STRETCH_METRIC_CANDIDATE",
                    detector_name="Spot vs 1H MA20 Stretch Candidate (Definition Open)",
                    track="NOVEL_QUANT",
                    confirmation_status="DEFINITION_OPEN",
                    direction=direction,
                    invalidation_level=round(inv_level, 2),
                    invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
                    invalidation_justification=(
                        f"Completed daily close beyond {round(inv_level, 2)} invalidates mean-reversion."
                    ),
                    summary=(
                        f"Spot ({spot}) is {dist_pts:.1f} pts ({price_structure.stretch_zscore_1h:+.2f} SD) "
                        f"from 1H MA20 ({price_structure.ma20_1h}). "
                        f"Reported as DEFINITION_OPEN until trader settles 'stretched' definition."
                    ),
                    evidence_fields_cited=[
                        "contract_context.spot.value",
                        "market_evidence.price_structure.ma20_1h",
                        "market_evidence.price_structure.stretch_zscore_1h",
                    ],
                )
            )

    return detectors
