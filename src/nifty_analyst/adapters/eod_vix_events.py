"""Adapter 4 (NSE EOD Volatility History Only) and Adapter 7 (Events Calendar).

STRICT CONSTRAINTS (Implementation Plan v3, Sections 3, 4.2, 8.5):
- Official NSE EOD archives are used SOLELY to build the constant-maturity 30-day ATM IV
  history required for IVP (30/90/252 sessions) and historical OI context.
- NEVER used for strategy backtesting, replay, or P&L simulation.
- Every EOD-inferred IV datum is labelled 'EOD_CLOSE_INFERRED'.
- Point-in-time clamped to trade_date <= decision_date.
- Validates continuity and sanity of the ATM IV series (flags session gaps).
"""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from nifty_analyst.adapters.calendar_and_contracts import count_trading_sessions
from nifty_analyst.engine.pricing import compute_time_to_expiry_years, solve_iv_black76
from nifty_analyst.scrubber import IST, ensure_ist
from nifty_analyst.state import CalendarEvent, EventCalendarEvidence


@dataclass(frozen=True)
class EODAtmIvRecord:
    trade_date: str
    spot_close: float
    constant_maturity_30d_iv: float
    realized_return_log: float
    source_label: Literal["EOD_CLOSE_INFERRED"] = "EOD_CLOSE_INFERRED"


@dataclass(frozen=True)
class EODVolatilityHistorySummary:
    records: list[EODAtmIvRecord]
    ivp_30d: float | Literal["UNKNOWN"]
    ivp_90d: float | Literal["UNKNOWN"]
    ivp_252d: float | Literal["UNKNOWN"]
    realized_vol_20d: float
    empirical_daily_log_returns: list[float]
    continuity_gaps_flagged: list[str]
    sanity_passed: bool


def infer_eod_atm_iv_from_archive_row(
    trade_date_str: str,
    spot_close: float,
    near_expiry_str: str,
    atm_strike: float,
    atm_call_close: float,
    atm_put_close: float,
    risk_free_rate: float = 0.0675,
) -> float:
    """Infer ATM IV from official NSE EOD closing prices using Black-76 put-call parity forward."""
    trade_dt = dt.datetime.combine(
        dt.date.fromisoformat(trade_date_str), dt.time(15, 30, 0), tzinfo=IST
    )
    t_years = compute_time_to_expiry_years(trade_dt, near_expiry_str)
    inv_df = math.exp(risk_free_rate * t_years)
    implied_f = atm_strike + inv_df * (atm_call_close - atm_put_close)
    if implied_f <= 0:
        implied_f = spot_close
    iv_c = solve_iv_black76(atm_call_close, implied_f, atm_strike, t_years, risk_free_rate, "CE")
    iv_p = solve_iv_black76(atm_put_close, implied_f, atm_strike, t_years, risk_free_rate, "PE")
    return 0.5 * (iv_c + iv_p)


def compute_iv_percentile(
    current_iv: float,
    historical_ivs: list[float],
    window: int,
) -> float | Literal["UNKNOWN"]:
    """Compute percentage of trailing `window` sessions with ATM IV <= current_iv.

    Returns 'UNKNOWN' if fewer than `window` verified sessions are available.
    """
    if len(historical_ivs) < window or window <= 0:
        return "UNKNOWN"
    subset = historical_ivs[-window:]
    count_le = sum(1 for val in subset if val <= current_iv)
    return round(100.0 * count_le / float(len(subset)), 2)


EOD_PUBLICATION_CUTOFF = dt.time(19, 0)  # NSE UDiFF bhavcopy is published by ~19:00 IST


def _last_trading_day_on_or_before(
    d: dt.date, holidays: set[dt.date], special_sessions: dict | None = None
) -> dt.date:
    """Trading day = weekday non-holiday, OR a declared special session (e.g. Muhurat
    Sunday 2026-11-08 per NSE FAOP76940). No weekday-only hardcode."""
    special = special_sessions or {}
    cursor = d
    while True:
        iso = cursor.isoformat()
        if iso in special or (cursor.weekday() < 5 and cursor not in holidays):
            return cursor
        cursor -= dt.timedelta(days=1)


def load_eod_volatility_history(
    eod_archive_path: str | Path,
    decision_dt_ist: dt.datetime,
    current_cm30_iv: float | None = None,
    holidays: set[dt.date] | None = None,
    risk_free_rate: float = 0.0675,
    special_sessions: dict | None = None,
) -> EODVolatilityHistorySummary:
    """Load point-in-time clamped NSE EOD archive solely for IVP and RV context (Section 3, 4.2, 8.5).

    Basis policy (7 Nov 2026): the percentiled quantity is the LATEST COMPLETED EOD cm30
    at-forward IV from this same series (same construction, same settlement-surface basis).
    A live intraday IV is NEVER percentiled against this series. Callers pass
    current_cm30_iv=None in production; an explicit value is accepted only for tests.

    Clamp + staleness: sessions with trade_date on the decision date count only once the
    bhavcopy should be published (decision time >= 19:00 IST). The archive's latest
    session must equal the last trading day on or before that cutoff; otherwise the
    archive is stale (missed refresh / missing publication) and every IVP is UNKNOWN.
    """
    holidays = holidays or set()
    dec_ist = ensure_ist(decision_dt_ist)
    cutoff_date = dec_ist.date() if dec_ist.time() >= EOD_PUBLICATION_CUTOFF else dec_ist.date() - dt.timedelta(days=1)
    dec_date_str = cutoff_date.isoformat()
    path = Path(eod_archive_path)
    if not path.exists():
        return EODVolatilityHistorySummary(
            records=[],
            ivp_30d="UNKNOWN",
            ivp_90d="UNKNOWN",
            ivp_252d="UNKNOWN",
            realized_vol_20d=0.12,
            empirical_daily_log_returns=[],
            continuity_gaps_flagged=["EOD archive file missing; IVP reported as UNKNOWN."],
            sanity_passed=False,
        )

    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("eod_sessions", [])
    # Point-in-time clamp: trade_date <= publication-aware cutoff date
    clamped_rows = [r for r in rows if str(r["trade_date"]) <= dec_date_str]
    clamped_rows.sort(key=lambda r: str(r["trade_date"]))

    # Staleness guard: the latest archived session must be the last trading day on or
    # before the cutoff. A previous session's archive is NOT indefinitely valid.
    expected_last = _last_trading_day_on_or_before(cutoff_date, holidays, special_sessions)
    stale = not clamped_rows or str(clamped_rows[-1]["trade_date"]) < expected_last.isoformat()
    if stale:
        latest = str(clamped_rows[-1]["trade_date"]) if clamped_rows else "none"
        return EODVolatilityHistorySummary(
            records=[],
            ivp_30d="UNKNOWN",
            ivp_90d="UNKNOWN",
            ivp_252d="UNKNOWN",
            realized_vol_20d=0.12,
            empirical_daily_log_returns=[],
            continuity_gaps_flagged=[
                f"EOD archive STALE: latest session {latest}, expected >= "
                f"{expected_last.isoformat()} (missed refresh or missing publication); "
                f"IVP reported as UNKNOWN."
            ],
            sanity_passed=False,
        )
    if current_cm30_iv is None:
        current_cm30_iv = float(clamped_rows[-1]["cm30_atm_iv"])

    records: list[EODAtmIvRecord] = []
    gaps_flagged: list[str] = []
    sanity_passed = True

    prev_spot: float | None = None
    prev_date: dt.date | None = None

    for row in clamped_rows:
        t_date_str = str(row["trade_date"])
        t_date = dt.date.fromisoformat(t_date_str)
        spot_close = float(row["spot_close"])

        if "cm30_atm_iv" in row and row["cm30_atm_iv"] is not None:
            cm_iv = float(row["cm30_atm_iv"])
        else:
            cm_iv = infer_eod_atm_iv_from_archive_row(
                trade_date_str=t_date_str,
                spot_close=spot_close,
                near_expiry_str=str(row["near_expiry"]),
                atm_strike=float(row["atm_strike"]),
                atm_call_close=float(row["atm_call_close"]),
                atm_put_close=float(row["atm_put_close"]),
                risk_free_rate=risk_free_rate,
            )

        if not (0.04 <= cm_iv <= 1.50):
            sanity_passed = False
            gaps_flagged.append(
                f"Sanity warning on {t_date_str}: inferred EOD ATM IV {cm_iv:.4f} outside [0.04, 1.50]."
            )

        if prev_date is not None:
            missed_sessions = count_trading_sessions(prev_date, t_date, holidays) - 1
            if missed_sessions > 0:
                gaps_flagged.append(
                    f"Continuity gap between {prev_date.isoformat()} and {t_date_str}: "
                    f"{missed_sessions} missing trading session(s)."
                )

        log_ret = math.log(spot_close / prev_spot) if (prev_spot and prev_spot > 0) else 0.0
        prev_spot = spot_close
        prev_date = t_date

        records.append(
            EODAtmIvRecord(
                trade_date=t_date_str,
                spot_close=spot_close,
                constant_maturity_30d_iv=round(cm_iv, 6),
                realized_return_log=round(log_ret, 6),
                source_label="EOD_CLOSE_INFERRED",
            )
        )

    iv_series = [r.constant_maturity_30d_iv for r in records]
    log_returns = [r.realized_return_log for r in records[1:]]

    if len(log_returns) >= 20:
        recent_20 = log_returns[-20:]
        mean_r = sum(recent_20) / 20.0
        var_r = sum((x - mean_r) ** 2 for x in recent_20) / 19.0
        rv_20d = math.sqrt(var_r * 252.0)
    else:
        rv_20d = 0.12

    ivp_30 = compute_iv_percentile(current_cm30_iv, iv_series, 30)
    ivp_90 = compute_iv_percentile(current_cm30_iv, iv_series, 90)
    ivp_252 = compute_iv_percentile(current_cm30_iv, iv_series, 252)

    return EODVolatilityHistorySummary(
        records=records,
        ivp_30d=ivp_30,
        ivp_90d=ivp_90,
        ivp_252d=ivp_252,
        realized_vol_20d=round(rv_20d, 4),
        empirical_daily_log_returns=log_returns,
        continuity_gaps_flagged=gaps_flagged,
        sanity_passed=sanity_passed,
    )


def load_events_calendar(
    events_file_path: str | Path,
    decision_dt_ist: dt.datetime,
    max_expiry_date_str: str,
    max_calendar_age_days: int = 14,
) -> EventCalendarEvidence:
    """Load human-maintained events calendar [UNVERIFIED SOURCE], flagging if >14 days old."""
    dec_ist = ensure_ist(decision_dt_ist)
    path = Path(events_file_path)
    if not path.exists():
        return EventCalendarEvidence(
            events_within_dte=[],
            calendar_file_age_days=999.0,
            is_calendar_stale_warning=True,
            source_verification_tag="UNVERIFIED",
        )

    raw = json.loads(path.read_text(encoding="utf-8"))
    ev_block = raw.get("events_calendar", raw)
    last_updated_str = str(ev_block.get("last_updated_ist", dec_ist.isoformat()))
    last_updated_dt = ensure_ist(dt.datetime.fromisoformat(last_updated_str))
    age_days = max(0.0, (dec_ist - last_updated_dt).total_seconds() / 86400.0)
    is_stale = age_days > float(max_calendar_age_days)

    dec_date_str = dec_ist.date().isoformat()
    upcoming: list[CalendarEvent] = []
    for item in ev_block.get("events", []):
        edate = str(item["event_date"])
        if dec_date_str <= edate <= max_expiry_date_str:
            upcoming.append(
                CalendarEvent(
                    event_date=edate,
                    event_name=str(item["event_name"]),
                    category=str(item.get("category", "MACRO")),
                    impact=item.get("impact", "HIGH"),
                )
            )

    return EventCalendarEvidence(
        events_within_dte=upcoming,
        calendar_file_age_days=round(age_days, 2),
        is_calendar_stale_warning=is_stale,
        source_verification_tag="UNVERIFIED",
    )
