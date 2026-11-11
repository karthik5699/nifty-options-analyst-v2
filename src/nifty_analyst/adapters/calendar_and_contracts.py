"""Adapter 1 (Contract Master & Expiry) and Adapter 5 (Trading Calendar).

Zero hardcoded weekdays or lot sizes. All expiries and lot sizes come strictly
from the parsed contract master, cross-checked against the NSE holiday calendar.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from nifty_analyst.scrubber import IST, ensure_ist
from nifty_analyst.state import ContractContext, ExpiryMetadata, TimestampedDatum


class ContractMasterError(RuntimeError):
    """Raised when the contract master is missing, empty, or corrupt (triggers INSUFFICIENT DATA)."""


def count_trading_sessions(
    start_date: dt.date,
    end_date: dt.date,
    holidays: set[dt.date],
) -> int:
    """Count trading sessions strictly after start_date up to and including end_date.

    - Same day (expiry today before 15:30 IST): 0 DTE sessions remaining after today.
    - Next trading session: 1 DTE session.
    """
    if end_date <= start_date:
        return 0
    sessions = 0
    cursor = start_date + dt.timedelta(days=1)
    while cursor <= end_date:
        if cursor.weekday() < 5 and cursor not in holidays:
            sessions += 1
        cursor += dt.timedelta(days=1)
    return sessions


def load_nse_holiday_calendar(calendar_path: str | Path) -> tuple[set[dt.date], dict[str, str]]:
    """Load official NSE holiday dates and descriptions from JSON fixture/config."""
    path = Path(calendar_path)
    if not path.exists():
        raise ContractMasterError(f"NSE trading holiday calendar not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    holidays: set[dt.date] = set()
    descriptions: dict[str, str] = {}
    for entry in raw.get("holidays", []):
        d_str = entry["date"]
        holidays.add(dt.date.fromisoformat(d_str))
        descriptions[d_str] = entry.get("description", "NSE Holiday")
    return holidays, descriptions


def parse_contract_master(
    raw_master: dict[str, Any],
    decision_dt_ist: dt.datetime,
    holidays: set[dt.date],
    holiday_descriptions: dict[str, str] | None = None,
    expected_calendar_expiries: list[str] | None = None,
    max_staleness_seconds: int = 180,
) -> ContractContext:
    """Parse contract master payload into ContractContext with zero hardcoded weekdays or lot sizes."""
    dec_ist = ensure_ist(decision_dt_ist)
    holiday_descriptions = holiday_descriptions or {}

    symbol = raw_master.get("symbol", "").strip()
    if symbol != "NIFTY":
        raise ContractMasterError(f"Invalid or missing symbol in contract master: {symbol!r}")

    spot_info = raw_master.get("spot")
    if not isinstance(spot_info, dict) or "value" not in spot_info or "as_of_ist" not in spot_info:
        raise ContractMasterError("Contract master missing spot value or as_of_ist timestamp.")

    spot_as_of = ensure_ist(dt.datetime.fromisoformat(spot_info["as_of_ist"]))
    spot_datum = TimestampedDatum.create(
        value=float(spot_info["value"]),
        as_of_dt=spot_as_of,
        fetched_at_dt=dec_ist,
        source_id=str(spot_info.get("source_id", "kotak_neo_scrip_master")),
        max_staleness_seconds=max_staleness_seconds,
        verification_tag=str(spot_info.get("verification_tag", "UNVERIFIED_SDK_SCHEMA")),
    )

    raw_expiries = raw_master.get("expiries", [])
    if not raw_expiries:
        raise ContractMasterError("Contract master contains zero expiries.")

    active_expiries: list[ExpiryMetadata] = []
    lot_sizes_seen: set[int] = set()

    for item in raw_expiries:
        if "expiry_date" not in item or "lot_size" not in item:
            raise ContractMasterError(
                "Expiry record in contract master missing expiry_date or lot_size."
            )
        exp_date_str = str(item["expiry_date"])
        exp_date = dt.date.fromisoformat(exp_date_str)
        if exp_date < dec_ist.date():
            continue

        lot_size = int(item["lot_size"])
        if lot_size <= 0:
            raise ContractMasterError(f"Invalid lot_size {lot_size} for expiry {exp_date_str}")
        lot_sizes_seen.add(lot_size)

        dte_sessions = count_trading_sessions(dec_ist.date(), exp_date, holidays)
        exp_ts_ist = dt.datetime.combine(exp_date, dt.time(15, 30, 0), tzinfo=IST).isoformat()

        # Check whether the day immediately following expiry_date is an NSE holiday
        next_day = exp_date + dt.timedelta(days=1)
        next_day_str = next_day.isoformat()
        inferred_holiday_shift = next_day in holidays
        is_holiday_adjusted = bool(item.get("is_holiday_adjusted", inferred_holiday_shift))

        holiday_note = str(item.get("holiday_note", "")).strip()
        if not holiday_note and inferred_holiday_shift:
            h_desc = holiday_descriptions.get(next_day_str, "NSE Holiday")
            holiday_note = (
                f"Holiday-shifted expiry on {exp_date.strftime('%A %d %b %Y')} "
                f"ahead of {next_day_str} ({h_desc})."
            )

        active_expiries.append(
            ExpiryMetadata(
                expiry_date=exp_date_str,
                expiry_timestamp_ist=exp_ts_ist,
                dte_trading_sessions=dte_sessions,
                lot_size=lot_size,
                is_holiday_adjusted=is_holiday_adjusted,
                holiday_note=holiday_note,
                is_monthly=bool(item.get("is_monthly", False)),
            )
        )

    if not active_expiries:
        raise ContractMasterError("No active future or current-day expiries in contract master.")

    active_expiries.sort(key=lambda e: e.expiry_date)
    lot_size_change_detected = len(lot_sizes_seen) > 1

    discrepancy_warnings: list[str] = []
    if expected_calendar_expiries is not None:
        live_dates = [e.expiry_date for e in active_expiries]
        for cal_exp in expected_calendar_expiries:
            if cal_exp >= dec_ist.date().isoformat() and cal_exp not in live_dates:
                discrepancy_warnings.append(
                    f"Calendar expiry {cal_exp} absent from live contract master; "
                    f"deferring strictly to live contract master ({live_dates})."
                )

    return ContractContext(
        symbol=symbol,
        spot=spot_datum,
        active_expiries=active_expiries,
        lot_size_change_detected=lot_size_change_detected,
        calendar_discrepancy_warnings=discrepancy_warnings,
    )
