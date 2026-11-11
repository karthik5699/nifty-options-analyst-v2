"""Unit tests for deterministic pricing, Black-76/BSM Greeks, Brent IV solver, Chain QA suite, and piecewise payoff math (Section 8.1)."""

from __future__ import annotations

import datetime as dt
import math

import pytest

from nifty_analyst.engine.pricing import (
    LegPayoffSpec,
    RawStrikeQuote,
    black76_greeks,
    black76_price,
    bsm_price,
    compute_piecewise_expiry_payoff,
    compute_time_to_expiry_years,
    run_chain_qa_suite,
    solve_iv_black76,
)
from nifty_analyst.scrubber import IST


def test_black76_and_bsm_reference_values() -> None:
    """Verify Black-76 prices, put-call parity identity, and Greeks against exact analytical values."""
    forward = 24200.0
    strike = 24200.0
    t = 10.0 / 365.0
    r = 0.0675
    sigma = 0.15

    call_px = black76_price(forward, strike, t, r, sigma, "CE")
    put_px = black76_price(forward, strike, t, r, sigma, "PE")

    # At F == K, Black-76 Call and Put prices must be strictly equal
    assert math.isclose(call_px, put_px, rel_tol=1e-9, abs_tol=1e-9)
    assert 235.0 < call_px < 245.0

    # Non-ATM put-call parity: C - P == exp(-rT) * (F - K)
    k_otm = 24400.0
    c_otm = black76_price(forward, k_otm, t, r, sigma, "CE")
    p_itm = black76_price(forward, k_otm, t, r, sigma, "PE")
    expected_diff = math.exp(-r * t) * (forward - k_otm)
    assert math.isclose(c_otm - p_itm, expected_diff, abs_tol=1e-8)

    # BSM equivalence with continuous dividend yield q
    q = 0.012
    spot = forward * math.exp(-(r - q) * t)
    bsm_c = bsm_price(spot, k_otm, t, r, q, sigma, "CE")
    assert math.isclose(bsm_c, c_otm, abs_tol=1e-8)

    # Analytical Greeks checks
    g_call = black76_greeks(forward, strike, t, r, sigma, "CE")
    g_put = black76_greeks(forward, strike, t, r, sigma, "PE")
    df = math.exp(-r * t)
    # Call delta - Put delta == exp(-rT)
    assert math.isclose(g_call.delta - g_put.delta, df, abs_tol=1e-7)
    # Gamma and Vega identical for Call and Put at same strike
    assert math.isclose(g_call.gamma, g_put.gamma, abs_tol=1e-9)
    assert math.isclose(g_call.vega_per_vol_pt, g_put.vega_per_vol_pt, abs_tol=1e-9)
    assert g_call.gamma > 0.0
    assert g_call.vega_per_vol_pt > 0.0
    assert g_call.theta_daily < 0.0


def test_brent_iv_solver_roundtrip() -> None:
    """Verify Brent's method IV solver recovers true volatility across strikes and vol levels."""
    forward = 24250.0
    t = 7.0 / 365.0
    r = 0.0675
    for true_sigma in (0.08, 0.145, 0.28, 0.65):
        for strike in (23800.0, 24250.0, 24600.0):
            for opt_type in ("CE", "PE"):
                px = black76_price(forward, strike, t, r, true_sigma, opt_type)  # type: ignore[arg-type]
                solved = solve_iv_black76(px, forward, strike, t, r, opt_type)  # type: ignore[arg-type]
                assert math.isclose(solved, true_sigma, abs_tol=1e-5)


def _build_synthetic_chain(
    forward: float = 24220.0,
    t: float = 6.0 / 365.0,
    r: float = 0.0675,
    sigma: float = 0.14,
) -> list[RawStrikeQuote]:
    quotes: list[RawStrikeQuote] = []
    for k in range(23900, 24550, 50):
        strike = float(k)
        c = black76_price(forward, strike, t, r, sigma, "CE")
        p = black76_price(forward, strike, t, r, sigma, "PE")
        quotes.append(
            RawStrikeQuote(
                strike=strike,
                call_bid=round(c - 0.5, 2),
                call_ask=round(c + 0.5, 2),
                put_bid=round(p - 0.5, 2),
                put_ask=round(p + 0.5, 2),
                call_oi=100000,
                put_oi=100000,
                call_volume=25000,
                put_volume=25000,
                quote_timestamp_ist="2026-11-06T10:14:45+05:30",
            )
        )
    return quotes


def test_chain_qa_suite_passes_valid_and_catches_violations() -> None:
    """Verify Put-Call Parity, ATM Anchor, Monotonicity, and Butterfly Arbitrage QA checks."""
    spot = 24200.0
    t = 6.0 / 365.0
    r = 0.0675
    valid_chain = _build_synthetic_chain(forward=24220.0, t=t, r=r, sigma=0.14)

    qa_ok = run_chain_qa_suite(spot, valid_chain, t, r)
    assert qa_ok.passed is True
    assert qa_ok.failure_code is None

    # 1. Corrupt put-call parity on one near-ATM strike
    bad_parity = list(valid_chain)
    target = bad_parity[6]  # 24200 strike
    bad_parity[6] = RawStrikeQuote(
        strike=target.strike,
        call_bid=target.call_bid + 65.0,
        call_ask=target.call_ask + 65.0,
        put_bid=target.put_bid,
        put_ask=target.put_ask,
    )
    qa_parity = run_chain_qa_suite(spot, bad_parity, t, r)
    assert qa_parity.passed is False
    assert qa_parity.failure_code == "CHAIN_PARITY_FAILURE"

    # 2. Corrupt monotonicity on deep strike without breaking near-ATM parity
    bad_mono = list(valid_chain)
    first_q = bad_mono[0]
    second_q = bad_mono[1]
    bad_mono[1] = RawStrikeQuote(
        strike=second_q.strike,
        call_bid=first_q.call_bid + 15.0,
        call_ask=first_q.call_ask + 15.0,
        put_bid=second_q.put_bid,
        put_ask=second_q.put_ask,
    )
    qa_mono = run_chain_qa_suite(spot, bad_mono, t, r)
    assert qa_mono.passed is False
    assert qa_mono.failure_code == "MONOTONICITY_FAILURE"


def test_piecewise_expiry_payoff_across_verticals_condors_and_butterflies() -> None:
    """Verify contractual max loss is computed leg-wise from piecewise expiry payoff (Section 4.3)."""
    lot_size = 75

    # 1. Bull Put Credit Spread: Short 24100 PE @ 85, Long 23950 PE @ 35 (width 150, net credit 50)
    # Exact max loss = (150 - 50) * 75 * 1 = Rs 7,500
    bps = compute_piecewise_expiry_payoff(
        legs=[
            LegPayoffSpec(24100.0, "PE", "SHORT", 1, 85.0),
            LegPayoffSpec(23950.0, "PE", "LONG", 1, 35.0),
        ],
        lot_size=lot_size,
        lots=1,
    )
    assert bps.is_defined_risk is True
    assert bps.max_loss_rs == 7500.0
    assert bps.max_profit_rs == 3750.0
    assert bps.breakevens_gross == [24050.0]

    # 2. Bull Call Debit Spread: Long 24200 CE @ 140, Short 24400 CE @ 60 (net debit 80)
    # Exact max loss = 80 * 75 * 1 = Rs 6,000
    bcs = compute_piecewise_expiry_payoff(
        legs=[
            LegPayoffSpec(24200.0, "CE", "LONG", 1, 140.0),
            LegPayoffSpec(24400.0, "CE", "SHORT", 1, 60.0),
        ],
        lot_size=lot_size,
        lots=1,
    )
    assert bcs.is_defined_risk is True
    assert bcs.max_loss_rs == 6000.0
    assert bcs.max_profit_rs == (200.0 - 80.0) * 75.0

    # 3. Asymmetric / Unequal-Wing Iron Condor (150-pt put wing vs 250-pt call wing)
    # Net credit = (40 - 15) + (45 - 10) = 60 pts.
    # Worst wing is 250-pt call wing -> max loss per unit = 250 - 60 = 190 pts -> Rs 14,250
    asym_ic = compute_piecewise_expiry_payoff(
        legs=[
            LegPayoffSpec(23850.0, "PE", "LONG", 1, 15.0),
            LegPayoffSpec(24000.0, "PE", "SHORT", 1, 40.0),
            LegPayoffSpec(24400.0, "CE", "SHORT", 1, 45.0),
            LegPayoffSpec(24650.0, "CE", "LONG", 1, 10.0),
        ],
        lot_size=lot_size,
        lots=1,
    )
    assert asym_ic.is_defined_risk is True
    assert asym_ic.max_loss_per_unit_points == 190.0
    assert asym_ic.max_loss_rs == 14250.0

    # 4. 1-2-1 Call Butterfly (Long 1x 24100 CE @ 160, Short 2x 24250 CE @ 85, Long 1x 24400 CE @ 35)
    # Net debit = 160 - 170 + 35 = 25 pts -> Max loss = 25 * 75 = Rs 1,875; Max profit at 24250 = (150 - 25)*75 = Rs 9,375
    bfly = compute_piecewise_expiry_payoff(
        legs=[
            LegPayoffSpec(24100.0, "CE", "LONG", 1, 160.0),
            LegPayoffSpec(24250.0, "CE", "SHORT", 2, 85.0),
            LegPayoffSpec(24400.0, "CE", "LONG", 1, 35.0),
        ],
        lot_size=lot_size,
        lots=1,
    )
    assert bfly.is_defined_risk is True
    assert bfly.max_loss_rs == 1875.0
    assert bfly.max_profit_rs == 9375.0

    # 5. Naked Short Call (must be flagged as undefined risk!)
    naked_call = compute_piecewise_expiry_payoff(
        legs=[LegPayoffSpec(24500.0, "CE", "SHORT", 1, 50.0)],
        lot_size=lot_size,
        lots=1,
    )
    assert naked_call.is_defined_risk is False
    assert math.isinf(naked_call.max_loss_rs)


def test_time_to_expiry_ist_enforcement() -> None:
    """Naive datetimes must be rejected; IST datetimes compute exact time to 15:30 IST."""
    naive_dt = dt.datetime(2026, 11, 6, 10, 15, 0)
    with pytest.raises(ValueError, match="Naive datetime is prohibited"):
        compute_time_to_expiry_years(naive_dt, "2026-11-12")

    ist_dt = dt.datetime(2026, 11, 6, 15, 30, 0, tzinfo=IST)
    t_years = compute_time_to_expiry_years(ist_dt, "2026-11-13")
    assert math.isclose(t_years, 7.0 / 365.0, rel_tol=1e-6)
