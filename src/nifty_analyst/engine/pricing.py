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
    """Strike-level call/put quote pair for Chain QA and pricing.

    Missing, unfetched, LTP-only, empty, locked, or crossed books MUST carry `None`
    for bid/ask (never `0.0` and never `ltp`) so unpriced strikes cannot leak into
    marks, breakevens, stops, or Chain QA.
    """

    strike: float
    call_bid: float | None
    call_ask: float | None
    put_bid: float | None
    put_ask: float | None
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
    call_touch_bid_qty: int | None = None
    call_touch_ask_qty: int | None = None
    put_touch_bid_qty: int | None = None
    put_touch_ask_qty: int | None = None
    call_visible_buy_qty: int | None = None
    call_visible_sell_qty: int | None = None
    put_visible_buy_qty: int | None = None
    put_visible_sell_qty: int | None = None
    call_buy_levels: tuple[tuple[float, int], ...] = ()
    call_sell_levels: tuple[tuple[float, int], ...] = ()
    put_buy_levels: tuple[tuple[float, int], ...] = ()
    put_sell_levels: tuple[tuple[float, int], ...] = ()
    call_quote_timestamp_ist: str = ""
    put_quote_timestamp_ist: str = ""
    fetch_started_at_ist: str = ""
    fetch_completed_at_ist: str = ""
    call_ts_semantics: str = ""
    put_ts_semantics: str = ""

    def quote_timestamp_for(self, option_type: OptionType) -> str:
        """Return the per-contract quote timestamp for CE or PE, falling back to strike timestamp."""
        if option_type == "CE":
            return self.call_quote_timestamp_ist or self.quote_timestamp_ist
        return self.put_quote_timestamp_ist or self.quote_timestamp_ist

    @property
    def has_executable_call(self) -> bool:
        """Structural touch usability (`0 < call_bid < call_ask`). Does not imply full-size fill."""
        return (
            self.call_bid is not None
            and self.call_ask is not None
            and 0.0 < self.call_bid < self.call_ask
        )

    @property
    def has_usable_call_touch(self) -> bool:
        return self.has_executable_call

    @property
    def has_executable_put(self) -> bool:
        """Structural touch usability (`0 < put_bid < put_ask`). Does not imply full-size fill."""
        return (
            self.put_bid is not None
            and self.put_ask is not None
            and 0.0 < self.put_bid < self.put_ask
        )

    @property
    def has_usable_put_touch(self) -> bool:
        return self.has_executable_put

    @property
    def has_both_executable(self) -> bool:
        return self.has_executable_call and self.has_executable_put

    @property
    def call_mid(self) -> float | None:
        if not self.has_executable_call or self.call_bid is None or self.call_ask is None:
            return None
        return 0.5 * (self.call_bid + self.call_ask)

    @property
    def put_mid(self) -> float | None:
        if not self.has_executable_put or self.put_bid is None or self.put_ask is None:
            return None
        return 0.5 * (self.put_bid + self.put_ask)

    def depth_lots_for(self, option_type: OptionType, side: Literal["LONG", "SHORT"]) -> int:
        """Return visible book depth in lots for the active order side (LONG lifts ask; SHORT hits bid)."""
        if option_type == "CE":
            if not self.has_executable_call:
                return 0
            if side == "LONG" and self.call_ask_depth_lots is not None:
                return self.call_ask_depth_lots
            if side == "SHORT" and self.call_bid_depth_lots is not None:
                return self.call_bid_depth_lots
            return self.call_depth_lots
        if not self.has_executable_put:
            return 0
        if side == "LONG" and self.put_ask_depth_lots is not None:
            return self.put_ask_depth_lots
        if side == "SHORT" and self.put_bid_depth_lots is not None:
            return self.put_bid_depth_lots
        return self.put_depth_lots

    def touch_qty_for(self, option_type: OptionType, side: Literal["LONG", "SHORT"]) -> int | None:
        """Return actual units available at the displayed touch (`best_ask` for LONG, `best_bid` for SHORT)."""
        if option_type == "CE":
            if not self.has_executable_call:
                return None
            return self.call_touch_ask_qty if side == "LONG" else self.call_touch_bid_qty
        if not self.has_executable_put:
            return None
        return self.put_touch_ask_qty if side == "LONG" else self.put_touch_bid_qty

    def visible_qty_for(self, option_type: OptionType, side: Literal["LONG", "SHORT"]) -> int | None:
        """Return total visible book units summed across quoted levels on the active side."""
        if option_type == "CE":
            if not self.has_executable_call:
                return None
            return self.call_visible_sell_qty if side == "LONG" else self.call_visible_buy_qty
        if not self.has_executable_put:
            return None
        return self.put_visible_sell_qty if side == "LONG" else self.put_visible_buy_qty

    def levels_for(
        self, option_type: OptionType, side: Literal["LONG", "SHORT"]
    ) -> tuple[tuple[float, int], ...]:
        """Return ordered `(price, quantity)` depth levels on the active side (`sell` for LONG, `buy` for SHORT)."""
        if option_type == "CE":
            if not self.has_executable_call:
                return ()
            return self.call_sell_levels if side == "LONG" else self.call_buy_levels
        if not self.has_executable_put:
            return ()
        return self.put_sell_levels if side == "LONG" else self.put_buy_levels


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
    """Compute synthetic forward F = K + exp(rT)*(C_mid - P_mid) across near-ATM strikes with executable books."""
    both_priced = [q for q in quotes if q.has_both_executable]
    if not both_priced:
        raise ValueError("Empty or unpriced option chain quotes (no strike has executable CE & PE books).")
    sorted_by_dist = sorted(both_priced, key=lambda q: abs(q.strike - spot))
    near_atm = sorted_by_dist[: min(num_near_atm, len(sorted_by_dist))]
    inv_df = math.exp(risk_free_rate * max(time_to_expiry_years, 1e-8))
    implied_forwards = [
        q.strike + inv_df * (float(q.call_mid) - float(q.put_mid))  # type: ignore[arg-type]
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
    max_leg_timestamp_skew_seconds: float = 30.0,
) -> ChainQAResult:
    """Execute mandatory Section 4.1 QA suite before any option chain is used.

    Supports both full-chain fixtures and targeted live depth funnels (where non-shortlisted
    far strikes carry `None` bid/ask):
    1. Verifies the 3 strikes nearest `spot` have executable CE & PE books (`CHAIN_ATM_DEPTH_MISSING`).
    2. Put-call parity consistency across up to 5 near-ATM priced strikes (`CHAIN_PARITY_FAILURE`).
    3. ATM-vs-spot anchor check (|delta| ~ 0.50 strike within 1 strike interval of synthetic forward;
       ATM call IV and put IV agree within 0.5 vol points).
    4. Monotonicity and vertical-spread / butterfly no-arbitrage checks across all priced strikes.
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
        if ordered[i + 1].strike > ordered[i].strike
    ]
    strike_step = min(strike_intervals) if strike_intervals else 50.0

    theo_fwd = spot * math.exp(risk_free_rate * max(time_to_expiry_years, 0.0))
    nearest_three = sorted(ordered, key=lambda q: abs(q.strike - spot))[:3]
    nearest_fwd_three = sorted(ordered, key=lambda q: abs(q.strike - theo_fwd))[:3]
    missing_spot_atm = [q.strike for q in nearest_three if not q.has_both_executable]
    missing_fwd_atm = [q.strike for q in nearest_fwd_three if not q.has_both_executable]
    both_priced = [q for q in ordered if q.has_both_executable]
    if (missing_spot_atm and missing_fwd_atm) or len(both_priced) < 3:
        missing_atm = missing_spot_atm
        notes.append(
            f"CHAIN_ATM_DEPTH_MISSING: near-ATM strikes {missing_atm} lack executable two-sided "
            f"depth books (priced_pairs={len(both_priced)})."
        )
        return ChainQAResult(
            passed=False,
            failure_code="CHAIN_ATM_DEPTH_MISSING",
            implied_forward=spot,
            atm_strike=nearest_three[0].strike if nearest_three else spot,
            atm_call_iv=0.0,
            atm_put_iv=0.0,
            parity_max_deviation_points=0.0,
            notes=notes,
        )

    # 0b. Per-leg timestamp synchronization across near-ATM CE/PE parity pairs
    parity_candidates = sorted(both_priced, key=lambda q: abs(q.strike - spot))[:5]
    for q_pair in parity_candidates:
        c_ts = q_pair.quote_timestamp_for("CE")
        p_ts = q_pair.quote_timestamp_for("PE")
        if c_ts and p_ts:
            try:
                c_dt = ensure_ist(dt.datetime.fromisoformat(c_ts))
                p_dt = ensure_ist(dt.datetime.fromisoformat(p_ts))
                skew_s = abs((c_dt - p_dt).total_seconds())
            except ValueError:
                notes.append(
                    f"CHAIN_PARITY_LEG_TIMESTAMP_DISAGREEMENT: malformed CE/PE timestamp at strike {q_pair.strike}."
                )
                return ChainQAResult(
                    passed=False,
                    failure_code="CHAIN_PARITY_LEG_TIMESTAMP_DISAGREEMENT",
                    implied_forward=spot,
                    atm_strike=q_pair.strike,
                    atm_call_iv=0.0,
                    atm_put_iv=0.0,
                    parity_max_deviation_points=0.0,
                    notes=notes,
                )
            if skew_s > max_leg_timestamp_skew_seconds:
                notes.append(
                    f"CHAIN_PARITY_LEG_TIMESTAMP_DISAGREEMENT: strike {q_pair.strike} CE ({c_ts}) vs "
                    f"PE ({p_ts}) diverge by {skew_s:.2f}s > {max_leg_timestamp_skew_seconds:.2f}s."
                )
                return ChainQAResult(
                    passed=False,
                    failure_code="CHAIN_PARITY_LEG_TIMESTAMP_DISAGREEMENT",
                    implied_forward=spot,
                    atm_strike=q_pair.strike,
                    atm_call_iv=0.0,
                    atm_put_iv=0.0,
                    parity_max_deviation_points=0.0,
                    notes=notes,
                )

    # 1. Put-call parity check across up to 5 near-ATM strikes with executable books
    implied_f, near_forwards = infer_forward_from_near_atm_parity(
        spot, both_priced, time_to_expiry_years, risk_free_rate, num_near_atm=5
    )
    max_dev = max(abs(f - implied_f) for f in near_forwards)
    if max_dev > parity_tolerance_points:
        notes.append(
            f"CHAIN_PARITY_FAILURE: max implied-forward deviation across {len(near_forwards)} near-ATM strikes "
            f"is {max_dev:.2f} pts > {parity_tolerance_points:.2f} pts."
        )
        return ChainQAResult(
            passed=False,
            failure_code="CHAIN_PARITY_FAILURE",
            implied_forward=implied_f,
            atm_strike=both_priced[0].strike,
            atm_call_iv=0.0,
            atm_put_iv=0.0,
            parity_max_deviation_points=max_dev,
            notes=notes,
        )
    notes.append(f"Put-call parity QA passed (implied F={implied_f:.2f}, max dev={max_dev:.2f} pts).")

    # 2. ATM-vs-spot anchor check
    atm_quote = min(both_priced, key=lambda q: abs(q.strike - implied_f))
    assert atm_quote.call_mid is not None and atm_quote.put_mid is not None
    atm_call_iv = solve_iv_black76(
        atm_quote.call_mid, implied_f, atm_quote.strike, time_to_expiry_years, risk_free_rate, "CE"
    )
    atm_put_iv = solve_iv_black76(
        atm_quote.put_mid, implied_f, atm_quote.strike, time_to_expiry_years, risk_free_rate, "PE"
    )

    call_priced = [q for q in ordered if q.has_executable_call]
    put_priced = [q for q in ordered if q.has_executable_put]

    # Find priced strike whose call |delta| is closest to 0.50
    best_delta_50_strike = atm_quote.strike
    best_delta_diff = 999.0
    for q in call_priced:
        assert q.call_mid is not None
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

    # 3. Monotonicity and vertical-spread / butterfly no-arbitrage checks across priced strikes
    df = math.exp(-risk_free_rate * max(time_to_expiry_years, 1e-8))
    for i in range(len(call_priced) - 1):
        q1, q2 = call_priced[i], call_priced[i + 1]
        assert q1.call_mid is not None and q2.call_mid is not None
        dk = q2.strike - q1.strike
        max_vertical = df * dk
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

    for i in range(len(put_priced) - 1):
        q1, q2 = put_priced[i], put_priced[i + 1]
        assert q1.put_mid is not None and q2.put_mid is not None
        dk = q2.strike - q1.strike
        max_vertical = df * dk
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

    # Butterfly convexity check on equidistant consecutive priced triples
    for i in range(len(call_priced) - 2):
        q1, q2, q3 = call_priced[i], call_priced[i + 1], call_priced[i + 2]
        assert q1.call_mid is not None and q2.call_mid is not None and q3.call_mid is not None
        if abs((q2.strike - q1.strike) - (q3.strike - q2.strike)) < 1e-6:
            call_bfly = q1.call_mid - 2.0 * q2.call_mid + q3.call_mid
            if call_bfly < -arbitrage_epsilon:
                notes.append(
                    f"BUTTERFLY_ARBITRAGE_FAILURE: Non-convex call butterfly at {q1.strike}/{q2.strike}/{q3.strike}."
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

    for i in range(len(put_priced) - 2):
        q1, q2, q3 = put_priced[i], put_priced[i + 1], put_priced[i + 2]
        assert q1.put_mid is not None and q2.put_mid is not None and q3.put_mid is not None
        if abs((q2.strike - q1.strike) - (q3.strike - q2.strike)) < 1e-6:
            put_bfly = q1.put_mid - 2.0 * q2.put_mid + q3.put_mid
            if put_bfly < -arbitrage_epsilon:
                notes.append(
                    f"BUTTERFLY_ARBITRAGE_FAILURE: Non-convex put butterfly at {q1.strike}/{q2.strike}/{q3.strike}."
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
