"""Completed-candle signal hygiene and Two-Track Setup Detector Registry (Section 5).

- Track 1 (PART_C_HISTORICAL): tagged HISTORICAL_REFERENCE_UNCONFIRMED; context only, never gates.
- Track 2 (NOVEL_QUANT): regime-driven detectors (volatility compression/expansion, high-IV range,
  skew dislocation, and candidate stretch metrics tagged DEFINITION_OPEN).
"""

from __future__ import annotations

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
    """Analyze completed 1D and 1H bars for trend state, HH+HL repair, MAs, and verified touches."""
    closes_1h = [b.close for b in completed_1h]
    ma20_1h = round(compute_sma(closes_1h, 20), 2) if closes_1h else 0.0
    ema50_1h = round(compute_ema(closes_1h, 50), 2) if closes_1h else 0.0

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
    # TRACK 2: Novel & Regime-Driven Detectors (First-Class Citizens)
    # -------------------------------------------------------------------------
    ivp_90 = volatility.ivp_90d
    numeric_ivp = float(ivp_90) if isinstance(ivp_90, (int, float)) else 50.0

    # 2A: High-IV Regime + Verified Support/OI Wall Defense (Favours Credit or Neutral Structures)
    if numeric_ivp >= 50.0 and price_structure.trend_1d_state in ("UPTREND", "DOWNTREND_REPAIRED_HH_HL"):
        inv_level = float(positioning.put_oi_wall_strike) if positioning.put_oi_wall_strike < spot else (spot - 200.0)
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
                    f"90d IVP is {ivp_90} with 1D trend {price_structure.trend_1d_state} and "
                    f"Put OI wall at {positioning.put_oi_wall_strike} ({positioning.put_oi_wall_contracts} contracts). "
                    f"Compares bullish credit spreads, debit spreads, and butterflies side by side."
                ),
                evidence_fields_cited=[
                    "market_evidence.volatility_state.ivp_90d",
                    "market_evidence.price_structure.trend_1d_state",
                    "market_evidence.derivatives_positioning.put_oi_wall_strike",
                ],
            )
        )

    # 2B: Volatility Compression Setup (Low IVP -> Favours Debit Structures)
    if numeric_ivp < 50.0 and price_structure.trend_1d_state in ("UPTREND", "DOWNTREND_REPAIRED_HH_HL"):
        inv_level = price_structure.ema50_1h if price_structure.ema50_1h < spot else (spot - 150.0)
        detectors.append(
            TriggeredDetector(
                detector_id="NOVEL_LOW_IV_COMPRESSION_BULLISH",
                detector_name="Low-IVP Volatility Compression Directional Setup (Debit-Favourable)",
                track="NOVEL_QUANT",
                confirmation_status="CONFIRMED_QUANT_REGIME",
                direction="BULLISH",
                invalidation_level=round(inv_level, 2),
                invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
                invalidation_justification=(
                    f"Completed daily close below {round(inv_level, 2)} negates upside expansion thesis."
                ),
                summary=(
                    f"90d IVP is compressed ({ivp_90}) with positive trend structure; long-vega debit structures "
                    f"(Bull Call Spread) avoid short-vega expansion risk."
                ),
                evidence_fields_cited=[
                    "market_evidence.volatility_state.ivp_90d",
                    "market_evidence.volatility_state.iv_rv_spread",
                ],
            )
        )

    # 2C: High-IV Range-Bound OI-Bracketed Regime (Favours Iron Condor / Butterfly)
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
                    f"and Call OI wall ({positioning.call_oi_wall_strike}) with 90d IVP={ivp_90}."
                ),
                evidence_fields_cited=[
                    "market_evidence.derivatives_positioning.put_oi_wall_strike",
                    "market_evidence.derivatives_positioning.call_oi_wall_strike",
                    "market_evidence.volatility_state.ivp_90d",
                ],
            )
        )

    # 2D: Candidate Stretch Metric (Tagged DEFINITION_OPEN per Section 5.2)
    if price_structure.ma20_1h > 0:
        dist_pts = abs(spot - price_structure.ma20_1h)
        if dist_pts >= 250.0:
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
                        f"Spot ({spot}) is {dist_pts:.1f} pts from 1H MA20 ({price_structure.ma20_1h}). "
                        f"Reported as DEFINITION_OPEN until trader settles 'stretched' definition."
                    ),
                    evidence_fields_cited=[
                        "contract_context.spot.value",
                        "market_evidence.price_structure.ma20_1h",
                    ],
                )
            )

    return detectors
