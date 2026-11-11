"""Deterministic Black-76 / BSM pricing, Brent's IV solver, Greeks, Chain QA suite,
constant-maturity variance interpolation, and piecewise expiry payoff engine.

Implements Section 4 of Implementation Plan v3 with zero LLM involvement.
STT is intentionally excluded per trader directive so technical analysis and options
surface analytics drive structure selection.
"""

from __future__ import annotations

import datetime as dt
import math
import statistics
from dataclasses import dataclass
from typing import Literal, Sequence

from nifty_analyst.scrubber import IST, ensure_ist

OptionType = Literal["CE", "PE"]
TermStructureRegime = Literal["CONTANGO", "BACKWARDATION", "FLAT", "SINGLE_EXPIRY"]

_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / _SQRT_2))


def norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def compute_time_to_expiry_years(
    decision_dt_ist: dt.datetime,
    expiry_date_str: str,
) -> float:
    """Compute time to expiry in years measured strictly to 15:30:00 IST on expiry_date."""
    dec_ist = ensure_ist(decision_dt_ist)
    exp_date = dt.date.fromisoformat(expiry_date_str)
    exp_dt_ist = dt.datetime.combine(exp_date, dt.time(15, 30, 0), tzinfo=IST)
    seconds_remaining = (exp_dt_ist - dec_ist).total_seconds()
    if seconds_remaining <= 0:
        return 1e-6  # Expired or at exact expiry boundary
    return seconds_remaining / (365.0 * 86400.0)


def black76_price(
    forward: float,
    strike: float,
    time_to_expiry_years: float,
    risk_free_rate: float,
    sigma: float,
    option_type: OptionType,
) -> float:
    """Analytical Black-76 European option price on forward F."""
    if forward <= 0.0 or strike <= 0.0:
        raise ValueError("Forward and strike must be strictly positive.")
    t = max(time_to_expiry_years, 1e-8)
    vol = max(sigma, 1e-8)
    df = math.exp(-risk_free_rate * t)
    sqrt_t = math.sqrt(t)
    d1 = (math.log(forward / strike) + 0.5 * vol * vol * t) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    if option_type == "CE":
        return df * (forward * norm_cdf(d1) - strike * norm_cdf(d2))
    if option_type == "PE":
        return df * (strike * norm_cdf(-d2) - forward * norm_cdf(-d1))
    raise ValueError(f"Unsupported option_type: {option_type}")


def bsm_price(
    spot: float,
    strike: float,
    time_to_expiry_years: float,
    risk_free_rate: float,
    dividend_yield: float,
    sigma: float,
    option_type: OptionType,
) -> float:
    """Black-Scholes-Merton European option price with continuous dividend yield q."""
    t = max(time_to_expiry_years, 1e-8)
    forward = spot * math.exp((risk_free_rate - dividend_yield) * t)
    return black76_price(forward, strike, t, risk_free_rate, sigma, option_type)


@dataclass(frozen=True)
class GreeksResult:
    delta: float
    gamma: float
    theta_daily: float
    vega_per_vol_pt: float


def black76_greeks(
    forward: float,
    strike: float,
    time_to_expiry_years: float,
    risk_free_rate: float,
    sigma: float,
    option_type: OptionType,
) -> GreeksResult:
    """Analytical Black-76 Greeks (delta, gamma, daily theta, vega per 1% vol point)."""
    t = max(time_to_expiry_years, 1e-8)
    vol = max(sigma, 1e-8)
    df = math.exp(-risk_free_rate * t)
    sqrt_t = math.sqrt(t)
    d1 = (math.log(forward / strike) + 0.5 * vol * vol * t) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    pdf_d1 = norm_pdf(d1)

    if option_type == "CE":
        delta = df * norm_cdf(d1)
        theta_annual = (
            -(forward * df * pdf_d1 * vol) / (2.0 * sqrt_t)
            + risk_free_rate * forward * df * norm_cdf(d1)
            - risk_free_rate * strike * df * norm_cdf(d2)
        )
    elif option_type == "PE":
        delta = -df * norm_cdf(-d1)
        theta_annual = (
            -(forward * df * pdf_d1 * vol) / (2.0 * sqrt_t)
            - risk_free_rate * forward * df * norm_cdf(-d1)
            + risk_free_rate * strike * df * norm_cdf(-d2)
        )
    else:
        raise ValueError(f"Unsupported option_type: {option_type}")

    gamma = (df * pdf_d1) / (forward * vol * sqrt_t)
    vega_per_vol_pt = (forward * df * pdf_d1 * sqrt_t) / 100.0
    theta_daily = theta_annual / 365.0

    return GreeksResult(
        delta=delta,
        gamma=gamma,
        theta_daily=theta_daily,
        vega_per_vol_pt=vega_per_vol_pt,
    )


def brentq_pure(
    func,
    a: float,
    b: float,
    xtol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """Pure-Python Brent's method root-finder on [a, b]."""
    fa = func(a)
    fb = func(b)
    if abs(fa) < xtol:
        return a
    if abs(fb) < xtol:
        return b
    if fa * fb > 0:
        raise ValueError(f"Root not bracketed on [{a}, {b}]: f(a)={fa}, f(b)={fb}")

    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    mflag = True
    d = 0.0

    for _ in range(max_iter):
        if abs(b - a) < xtol or abs(fb) < xtol:
            return b
        if fa != fc and fb != fc:
            # Inverse quadratic interpolation
            s = (
                a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
            )
        else:
            # Secant method
            s = b - fb * (b - a) / (fb - fa)

        cond1 = not ((3.0 * a + b) / 4.0 < s < b or b < s < (3.0 * a + b) / 4.0)
        cond2 = mflag and abs(s - b) >= abs(b - c) / 2.0
        cond3 = (not mflag) and abs(s - b) >= abs(c - d) / 2.0
        cond4 = mflag and abs(b - c) < xtol
        cond5 = (not mflag) and abs(c - d) < xtol

        if cond1 or cond2 or cond3 or cond4 or cond5:
            s = 0.5 * (a + b)
            mflag = True
        else:
            mflag = False

        fs = func(s)
        d = c
        c = b
        fc = fb

        if fa * fs < 0:
            b = s
            fb = fs
        else:
            a = s
            fa = fs

        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

    return b


def solve_iv_black76(
    market_price: float,
    forward: float,
    strike: float,
    time_to_expiry_years: float,
    risk_free_rate: float,
    option_type: OptionType,
    sigma_low: float = 0.01,
    sigma_high: float = 3.00,
) -> float:
    """Solve Black-76 implied volatility using Brent's method over [0.01, 3.00] (Section 4.1)."""
    t = max(time_to_expiry_years, 1e-8)
    df = math.exp(-risk_free_rate * t)
    intrinsic = df * max(0.0, (forward - strike) if option_type == "CE" else (strike - forward))
    if market_price <= intrinsic + 1e-6:
        return sigma_low

    def obj(sigma: float) -> float:
        return black76_price(forward, strike, t, risk_free_rate, sigma, option_type) - market_price

    low_val = obj(sigma_low)
    high_val = obj(sigma_high)
    if low_val >= 0.0:
        return sigma_low
    if high_val <= 0.0:
        return sigma_high
    return brentq_pure(obj, sigma_low, sigma_high)


def compute_variance_interpolated_cm30_iv(
    expiry_t_and_ivs: Sequence[tuple[float, float]],
    target_days: float = 30.0,
) -> tuple[float, float, TermStructureRegime]:
    """Compute constant-maturity 30-day ATM IV via total-variance interpolation across tenors.

    Returns (cm30_iv, near_minus_far_vol_slope, term_structure_regime).
    """
    valid = sorted(
        [(float(t), float(iv)) for t, iv in expiry_t_and_ivs if t > 0.0 and iv > 0.0],
        key=lambda pair: pair[0],
    )
    if not valid:
        return 0.0, 0.0, "SINGLE_EXPIRY"
    if len(valid) == 1:
        return round(valid[0][1], 6), 0.0, "SINGLE_EXPIRY"

    near_t, near_iv = valid[0]
    far_t, far_iv = valid[-1]
    slope = round(near_iv - far_iv, 6)
    if slope > 0.005:
        regime: TermStructureRegime = "BACKWARDATION"
    elif slope < -0.005:
        regime = "CONTANGO"
    else:
        regime = "FLAT"

    target_t = target_days / 365.0
    if target_t <= valid[0][0]:
        return round(valid[0][1], 6), slope, regime

    # Bracketed total-variance linear interpolation V(T) = sigma(T)^2 * T
    for i in range(len(valid) - 1):
        t1, iv1 = valid[i]
        t2, iv2 = valid[i + 1]
        if t1 <= target_t <= t2:
            if abs(t2 - t1) < 1e-9:
                return round(0.5 * (iv1 + iv2), 6), slope, regime
            v1 = iv1 * iv1 * t1
            v2 = iv2 * iv2 * t2
            w = (target_t - t1) / (t2 - t1)
            v_target = v1 + w * (v2 - v1)
            cm_iv = math.sqrt(max(1e-8, v_target / target_t))
            return round(cm_iv, 6), slope, regime

    # Target tenor lies beyond furthest active expiry: extrapolate only when forward variance > 0,
    # clamped safely around observed tenors to prevent short-tenor noise blow-up.
    t1, iv1 = valid[-2]
    t2, iv2 = valid[-1]
    v1 = iv1 * iv1 * t1
    v2 = iv2 * iv2 * t2
    if t2 - t1 > 1e-6 and v2 > v1:
        fwd_var = (v2 - v1) / (t2 - t1)
        v_target = v2 + fwd_var * (target_t - t2)
        raw_cm = math.sqrt(max(1e-8, v_target / target_t))
        min_obs = min(iv for _, iv in valid)
        max_obs = max(iv for _, iv in valid)
        clamped = min(max(raw_cm, 0.85 * min_obs), 1.15 * max_obs)
        return round(clamped, 6), slope, regime

    return round(far_iv, 6), slope, regime


@dataclass(frozen=True)
class RawStrikeQuote:
    """Strike-level call/put quote pair for Chain QA and pricing."""

    strike: float
    call_bid: float
    call_ask: float
    put_bid: float
    put_ask: float
    call_oi: int = 0
    put_oi: int = 0
    call_volume: int = 0
    put_volume: int = 0
    call_depth_lots: int = 10
    put_depth_lots: int = 10
    quote_timestamp_ist: str = ""
    expiry_date: str = ""  # populated by adapter; used to reject mixed-expiry chains
    call_bid_depth_lots: int | None = None
    call_ask_depth_lots: int | None = None
    put_bid_depth_lots: int | None = None
    put_ask_depth_lots: int | None = None

    @property
    def call_mid(self) -> float:
        return 0.5 * (self.call_bid + self.call_ask)

    @property
    def put_mid(self) -> float:
        return 0.5 * (self.put_bid + self.put_ask)

    def depth_lots_for(self, option_type: OptionType, side: Literal["LONG", "SHORT"]) -> int:
        """Return executable top-of-book depth for the active order side (LONG lifts ask; SHORT hits bid)."""
        if option_type == "CE":
            if side == "LONG" and self.call_ask_depth_lots is not None:
                return self.call_ask_depth_lots
            if side == "SHORT" and self.call_bid_depth_lots is not None:
                return self.call_bid_depth_lots
            return self.call_depth_lots
        if side == "LONG" and self.put_ask_depth_lots is not None:
            return self.put_ask_depth_lots
        if side == "SHORT" and self.put_bid_depth_lots is not None:
            return self.put_bid_depth_lots
        return self.put_depth_lots


@dataclass(frozen=True)
class ChainQAResult:
    passed: bool
    failure_code: str | None
    implied_forward: float
    atm_strike: float
    atm_call_iv: float
    atm_put_iv: float
    parity_max_deviation_points: float
    notes: list[str]


def infer_forward_from_near_atm_parity(
    spot: float,
    quotes: Sequence[RawStrikeQuote],
    time_to_expiry_years: float,
    risk_free_rate: float,
    num_near_atm: int = 5,
) -> tuple[float, list[float]]:
    """Compute synthetic forward F = K + exp(rT)*(C_mid - P_mid) across 5 strikes nearest ATM."""
    if not quotes:
        raise ValueError("Empty option chain quotes.")
    sorted_by_dist = sorted(quotes, key=lambda q: abs(q.strike - spot))
    near_atm = sorted_by_dist[: min(num_near_atm, len(sorted_by_dist))]
    inv_df = math.exp(risk_free_rate * max(time_to_expiry_years, 1e-8))
    implied_forwards = [
        q.strike + inv_df * (q.call_mid - q.put_mid)
        for q in near_atm
    ]
    median_f = float(statistics.median(implied_forwards))
    return median_f, implied_forwards


def run_chain_qa_suite(
    spot: float,
    quotes: Sequence[RawStrikeQuote],
    time_to_expiry_years: float,
    risk_free_rate: float,
    parity_tolerance_points: float = 25.0,
    atm_iv_diff_limit: float = 0.005,  # 0.5 vol points (0.5% = 0.005)
    arbitrage_epsilon: float = 0.50,
) -> ChainQAResult:
    """Execute mandatory Section 4.1 QA suite before any option chain is used.

    Checks:
    1. Put-call parity consistency across the 5 strikes nearest ATM (aborts with CHAIN_PARITY_FAILURE).
    2. ATM-vs-spot anchor check (|delta| ~ 0.50 strike within 1 strike interval of synthetic forward;
       ATM call IV and put IV agree within 0.5 vol points).
    3. Monotonicity and vertical-spread / butterfly no-arbitrage checks.
    """
    notes: list[str] = []
    if len(quotes) < 3:
        return ChainQAResult(
            passed=False,
            failure_code="CHAIN_INSUFFICIENT_STRIKES",
            implied_forward=spot,
            atm_strike=spot,
            atm_call_iv=0.0,
            atm_put_iv=0.0,
            parity_max_deviation_points=0.0,
            notes=["Fewer than 3 strikes in option chain."],
        )

    ordered = sorted(quotes, key=lambda q: q.strike)
    strike_intervals = [
        ordered[i + 1].strike - ordered[i].strike
        for i in range(len(ordered) - 1)
    ]
    strike_step = min(strike_intervals) if strike_intervals else 50.0

    # 1. Put-call parity check across 5 strikes nearest ATM
    implied_f, near_forwards = infer_forward_from_near_atm_parity(
        spot, ordered, time_to_expiry_years, risk_free_rate, num_near_atm=5
    )
    max_dev = max(abs(f - implied_f) for f in near_forwards)
    if max_dev > parity_tolerance_points:
        notes.append(
            f"CHAIN_PARITY_FAILURE: max implied-forward deviation across 5 near-ATM strikes "
            f"is {max_dev:.2f} pts > {parity_tolerance_points:.2f} pts."
        )
        return ChainQAResult(
            passed=False,
            failure_code="CHAIN_PARITY_FAILURE",
            implied_forward=implied_f,
            atm_strike=ordered[0].strike,
            atm_call_iv=0.0,
            atm_put_iv=0.0,
            parity_max_deviation_points=max_dev,
            notes=notes,
        )
    notes.append(f"Put-call parity QA passed (implied F={implied_f:.2f}, max dev={max_dev:.2f} pts).")

    # 2. ATM-vs-spot anchor check
    atm_quote = min(ordered, key=lambda q: abs(q.strike - implied_f))
    atm_call_iv = solve_iv_black76(
        atm_quote.call_mid, implied_f, atm_quote.strike, time_to_expiry_years, risk_free_rate, "CE"
    )
    atm_put_iv = solve_iv_black76(
        atm_quote.put_mid, implied_f, atm_quote.strike, time_to_expiry_years, risk_free_rate, "PE"
    )

    # Find strike whose call |delta| is closest to 0.50
    best_delta_50_strike = atm_quote.strike
    best_delta_diff = 999.0
    for q in ordered:
        c_iv = solve_iv_black76(
            q.call_mid, implied_f, q.strike, time_to_expiry_years, risk_free_rate, "CE"
        )
        g = black76_greeks(implied_f, q.strike, time_to_expiry_years, risk_free_rate, c_iv, "CE")
        diff = abs(abs(g.delta) - 0.50)
        if diff < best_delta_diff:
            best_delta_diff = diff
            best_delta_50_strike = q.strike

    if abs(best_delta_50_strike - implied_f) > strike_step * 1.05:
        notes.append(
            f"ATM_ANCHOR_FAILURE: |delta|~0.50 strike ({best_delta_50_strike}) is more than "
            f"one strike interval ({strike_step}) away from synthetic forward ({implied_f:.2f})."
        )
        return ChainQAResult(
            passed=False,
            failure_code="ATM_ANCHOR_FAILURE",
            implied_forward=implied_f,
            atm_strike=atm_quote.strike,
            atm_call_iv=atm_call_iv,
            atm_put_iv=atm_put_iv,
            parity_max_deviation_points=max_dev,
            notes=notes,
        )

    iv_diff = abs(atm_call_iv - atm_put_iv)
    if iv_diff > atm_iv_diff_limit:
        notes.append(
            f"ATM_ANCHOR_FAILURE: ATM call IV ({atm_call_iv * 100:.2f}%) and put IV "
            f"({atm_put_iv * 100:.2f}%) differ by {iv_diff * 100:.2f} vol pts (> {atm_iv_diff_limit * 100:.2f})."
        )
        return ChainQAResult(
            passed=False,
            failure_code="ATM_ANCHOR_FAILURE",
            implied_forward=implied_f,
            atm_strike=atm_quote.strike,
            atm_call_iv=atm_call_iv,
            atm_put_iv=atm_put_iv,
            parity_max_deviation_points=max_dev,
            notes=notes,
        )
    notes.append(
        f"ATM anchor QA passed (ATM={atm_quote.strike}, Call IV={atm_call_iv * 100:.2f}%, Put IV={atm_put_iv * 100:.2f}%)."
    )

    # 3. Monotonicity and vertical-spread / butterfly no-arbitrage checks
    df = math.exp(-risk_free_rate * max(time_to_expiry_years, 1e-8))
    for i in range(len(ordered) - 1):
        q1, q2 = ordered[i], ordered[i + 1]
        dk = q2.strike - q1.strike
        max_vertical = df * dk

        # Call monotonicity: C(K1) >= C(K2)
        if q1.call_mid + arbitrage_epsilon < q2.call_mid:
            notes.append(
                f"MONOTONICITY_FAILURE: Call mid increases from {q1.strike} ({q1.call_mid}) to {q2.strike} ({q2.call_mid})."
            )
            return ChainQAResult(
                passed=False,
                failure_code="MONOTONICITY_FAILURE",
                implied_forward=implied_f,
                atm_strike=atm_quote.strike,
                atm_call_iv=atm_call_iv,
                atm_put_iv=atm_put_iv,
                parity_max_deviation_points=max_dev,
                notes=notes,
            )

        # Put monotonicity: P(K1) <= P(K2)
        if q1.put_mid > q2.put_mid + arbitrage_epsilon:
            notes.append(
                f"MONOTONICITY_FAILURE: Put mid decreases from {q1.strike} ({q1.put_mid}) to {q2.strike} ({q2.put_mid})."
            )
            return ChainQAResult(
                passed=False,
                failure_code="MONOTONICITY_FAILURE",
                implied_forward=implied_f,
                atm_strike=atm_quote.strike,
                atm_call_iv=atm_call_iv,
                atm_put_iv=atm_put_iv,
                parity_max_deviation_points=max_dev,
                notes=notes,
            )

        # Vertical spread upper bound
        if (q1.call_mid - q2.call_mid) > max_vertical + arbitrage_epsilon:
            notes.append(
                f"VERTICAL_ARBITRAGE_FAILURE: Call vertical {q1.strike}/{q2.strike} exceeds discounted width."
            )
            return ChainQAResult(
                passed=False,
                failure_code="VERTICAL_ARBITRAGE_FAILURE",
                implied_forward=implied_f,
                atm_strike=atm_quote.strike,
                atm_call_iv=atm_call_iv,
                atm_put_iv=atm_put_iv,
                parity_max_deviation_points=max_dev,
                notes=notes,
            )
        if (q2.put_mid - q1.put_mid) > max_vertical + arbitrage_epsilon:
            notes.append(
                f"VERTICAL_ARBITRAGE_FAILURE: Put vertical {q1.strike}/{q2.strike} exceeds discounted width."
            )
            return ChainQAResult(
                passed=False,
                failure_code="VERTICAL_ARBITRAGE_FAILURE",
                implied_forward=implied_f,
                atm_strike=atm_quote.strike,
                atm_call_iv=atm_call_iv,
                atm_put_iv=atm_put_iv,
                parity_max_deviation_points=max_dev,
                notes=notes,
            )

    # Butterfly convexity check on equidistant consecutive triples
    for i in range(len(ordered) - 2):
        q1, q2, q3 = ordered[i], ordered[i + 1], ordered[i + 2]
        if abs((q2.strike - q1.strike) - (q3.strike - q2.strike)) < 1e-6:
            call_bfly = q1.call_mid - 2.0 * q2.call_mid + q3.call_mid
            put_bfly = q1.put_mid - 2.0 * q2.put_mid + q3.put_mid
            if call_bfly < -arbitrage_epsilon or put_bfly < -arbitrage_epsilon:
                notes.append(
                    f"BUTTERFLY_ARBITRAGE_FAILURE: Non-convex butterfly at {q1.strike}/{q2.strike}/{q3.strike}."
                )
                return ChainQAResult(
                    passed=False,
                    failure_code="BUTTERFLY_ARBITRAGE_FAILURE",
                    implied_forward=implied_f,
                    atm_strike=atm_quote.strike,
                    atm_call_iv=atm_call_iv,
                    atm_put_iv=atm_put_iv,
                    parity_max_deviation_points=max_dev,
                    notes=notes,
                )

    notes.append("Monotonicity, vertical-spread, and butterfly no-arbitrage checks passed.")
    return ChainQAResult(
        passed=True,
        failure_code=None,
        implied_forward=implied_f,
        atm_strike=atm_quote.strike,
        atm_call_iv=atm_call_iv,
        atm_put_iv=atm_put_iv,
        parity_max_deviation_points=max_dev,
        notes=notes,
    )


@dataclass(frozen=True)
class LegPayoffSpec:
    """Specification of a single option leg for leg-wise piecewise expiry payoff evaluation."""

    strike: float
    option_type: OptionType
    side: Literal["LONG", "SHORT"]
    ratio: int
    executable_entry_price: float  # after slippage buffer


@dataclass(frozen=True)
class PiecewisePayoffAnalysis:
    is_defined_risk: bool
    max_loss_per_unit_points: float
    max_loss_rs: float
    max_profit_per_unit_points: float
    max_profit_rs: float
    breakevens_gross: list[float]
    breakevens_net: list[float]
    method_description: str
    worst_case_spot: float = 0.0
    gross_max_loss_per_unit_points: float = 0.0
    gross_max_loss_rs: float = 0.0


def evaluate_leg_expiry_pnl_per_unit(leg: LegPayoffSpec, spot_at_expiry: float) -> float:
    """Compute net expiry P&L in index points for one leg at terminal spot S_T."""
    if leg.option_type == "CE":
        intrinsic = max(0.0, spot_at_expiry - leg.strike)
    else:
        intrinsic = max(0.0, leg.strike - spot_at_expiry)

    if leg.side == "LONG":
        return float(leg.ratio) * (intrinsic - leg.executable_entry_price)
    return float(leg.ratio) * (leg.executable_entry_price - intrinsic)


def compute_piecewise_expiry_payoff(
    legs: Sequence[LegPayoffSpec],
    lot_size: int,
    lots: int,
    statutory_charges_rs: float = 0.0,
) -> PiecewisePayoffAnalysis:
    """Compute contractual max loss leg-wise from the piecewise-linear expiry payoff (Section 4.3).

    Because European option expiry payoffs are piecewise linear with kinks solely at the leg
    strikes, the minimum and maximum over all S_T >= 0 occur either at S_T = 0, at one of the
    leg strikes K_i, or asymptotically as S_T -> +infinity.
    STT is excluded per trader directive; only non-STT execution fees shift net breakevens/P&L.
    """
    if not legs:
        raise ValueError("Cannot evaluate payoff of empty leg list.")
    if lot_size <= 0 or lots <= 0:
        raise ValueError("lot_size and lots must be positive integers.")

    total_quantity = lot_size * lots
    strikes = sorted({leg.strike for leg in legs})
    k_min = min(strikes)
    k_max = max(strikes)

    # Check asymptotic slope for S_T > k_max (only Call legs contribute slope as S_T -> +inf)
    high_spot_slope = 0.0
    for leg in legs:
        if leg.option_type == "CE":
            high_spot_slope += float(leg.ratio) if leg.side == "LONG" else -float(leg.ratio)

    if high_spot_slope < -1e-9:
        return PiecewisePayoffAnalysis(
            is_defined_risk=False,
            max_loss_per_unit_points=math.inf,
            max_loss_rs=math.inf,
            max_profit_per_unit_points=0.0,
            max_profit_rs=0.0,
            breakevens_gross=[],
            breakevens_net=[],
            method_description="UNDEFINED_RISK_UNBOUNDED_UPSIDE_SLOPE",
            gross_max_loss_per_unit_points=math.inf,
            gross_max_loss_rs=math.inf,
        )

    grid = sorted(
        {0.0, max(0.0, k_min - 500.0), max(0.0, k_min - 50.0)}
        | set(strikes)
        | {k_max + 50.0, k_max + 500.0, k_max * 2.0}
    )

    def gross_pnl_pts(s_t: float) -> float:
        return sum(evaluate_leg_expiry_pnl_per_unit(leg, s_t) for leg in legs)

    charges_per_unit_pts = statutory_charges_rs / float(total_quantity)

    def net_pnl_pts(s_t: float) -> float:
        return gross_pnl_pts(s_t) - charges_per_unit_pts

    pnl_values = [net_pnl_pts(s) for s in grid]
    min_pnl_pts = min(pnl_values)
    max_pnl_pts = max(pnl_values)

    # Check if unhedged short puts exist below k_min
    low_spot_put_net_short = 0.0
    for leg in legs:
        if leg.option_type == "PE":
            low_spot_put_net_short += float(leg.ratio) if leg.side == "LONG" else -float(leg.ratio)

    if low_spot_put_net_short < -1e-9:
        return PiecewisePayoffAnalysis(
            is_defined_risk=False,
            max_loss_per_unit_points=math.inf,
            max_loss_rs=math.inf,
            max_profit_per_unit_points=max(0.0, max_pnl_pts),
            max_profit_rs=max(0.0, max_pnl_pts * total_quantity),
            breakevens_gross=[],
            breakevens_net=[],
            method_description="UNDEFINED_RISK_UNHEDGED_SHORT_PUT",
            gross_max_loss_per_unit_points=math.inf,
            gross_max_loss_rs=math.inf,
        )

    max_loss_per_unit_pts = max(0.0, -min_pnl_pts)
    max_profit_per_unit_pts = max(0.0, max_pnl_pts)
    worst_case_spot = grid[pnl_values.index(min_pnl_pts)]
    gross_max_loss_per_unit_pts = max(0.0, -min(gross_pnl_pts(s) for s in grid))
    gross_max_loss_rs = round(gross_max_loss_per_unit_pts * total_quantity, 2)
    max_loss_rs = round(max_loss_per_unit_pts * total_quantity, 2)
    max_profit_rs = round(max_profit_per_unit_pts * total_quantity, 2)

    breakevens_gross: list[float] = []
    breakevens_net: list[float] = []

    for i in range(len(grid) - 1):
        s1, s2 = grid[i], grid[i + 1]
        if s2 - s1 < 1e-9:
            continue
        y1_gross, y2_gross = gross_pnl_pts(s1), gross_pnl_pts(s2)
        if (y1_gross <= 0.0 < y2_gross) or (y2_gross <= 0.0 < y1_gross):
            root_g = s1 - y1_gross * (s2 - s1) / (y2_gross - y1_gross)
            if k_min - 1000.0 <= root_g <= k_max + 1000.0:
                breakevens_gross.append(round(root_g, 2))

        y1_net = net_pnl_pts(s1)
        y2_net = net_pnl_pts(s2)
        if (y1_net <= 0.0 < y2_net) or (y2_net <= 0.0 < y1_net):
            root_n = s1 - y1_net * (s2 - s1) / (y2_net - y1_net)
            if k_min - 1000.0 <= root_n <= k_max + 1000.0:
                breakevens_net.append(round(root_n, 2))

    return PiecewisePayoffAnalysis(
        is_defined_risk=True,
        max_loss_per_unit_points=round(max_loss_per_unit_pts, 4),
        max_loss_rs=max_loss_rs,
        max_profit_per_unit_points=round(max_profit_per_unit_pts, 4),
        max_profit_rs=max_profit_rs,
        breakevens_gross=sorted(set(breakevens_gross)),
        breakevens_net=sorted(set(breakevens_net)),
        method_description="LEG_WISE_PIECEWISE_EXPIRY_PAYOFF",
        worst_case_spot=worst_case_spot,
        gross_max_loss_per_unit_points=round(gross_max_loss_per_unit_pts, 4),
        gross_max_loss_rs=gross_max_loss_rs,
    )
