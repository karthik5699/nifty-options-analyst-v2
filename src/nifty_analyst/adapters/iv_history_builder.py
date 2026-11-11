"""NSE UDiFF EOD Volatility Series Builder & Exact Symbol Filter (Adapter 4 Helper).

STRICT CONSTRAINTS (Implementation Plan v3, Sections 3, 4.2, 8.5):
- Used SOLELY to build/refresh the constant-maturity 30-day ATM IV (`cm30_atm_iv`) EOD archive
  required for IVP (30 / 90 / 252 sessions) and 20-day realized volatility.
- NEVER used for strategy backtesting, replay, or P&L simulation.
- Enforces exact ticker match `row.get('TckrSymb') == 'NIFTY'` so `BANKNIFTY`, `FINNIFTY`,
  and `MIDCPNIFTY` rows never leak into NIFTY 50 futures/options series.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from nifty_analyst.adapters.eod_vix_events import infer_eod_atm_iv_from_archive_row
from nifty_analyst.engine.pricing import (
    compute_time_to_expiry_years,
    compute_variance_interpolated_cm30_iv,
)
from nifty_analyst.scrubber import IST

NIFTY_MIN_VALID_PRICE = 15000.0
NIFTY_MAX_VALID_PRICE = 35000.0
NIFTY_MAX_FUTURES_BASIS_PCT = 0.06


def filter_exact_nifty_udiff_rows(
    rows: Iterable[dict[str, Any]],
    instrument_code: str | None = None,
) -> list[dict[str, Any]]:
    """Filter NSE UDiFF bhavcopy rows using exact `TckrSymb == 'NIFTY'` equality.

    Substring checks are strictly forbidden because `BANKNIFTY`, `FINNIFTY`, and
    `MIDCPNIFTY` contain `'NIFTY'`.
    """
    matched: list[dict[str, Any]] = []
    for row in rows:
        if row.get("TckrSymb") == "NIFTY":
            if instrument_code is not None and row.get("FinInstrmTp") != instrument_code:
                continue
            matched.append(row)
    return matched


def validate_nifty_futures_row(row: dict[str, Any]) -> tuple[float, float]:
    """Validate a NIFTY index futures (`IDF`) UDiFF row for symbol, price range, and basis."""
    if row.get("TckrSymb") != "NIFTY":
        raise ValueError(f"Non-NIFTY symbol rejected: {row.get('TckrSymb')!r}")
    px_raw = row.get("SttlmPric") or row.get("ClsPric")
    und_raw = row.get("UndrlygPric")
    if px_raw in (None, "") or und_raw in (None, ""):
        raise ValueError(f"Missing futures settlement/underlying price in row: {row}")
    px = float(px_raw)
    und = float(und_raw)
    if not (NIFTY_MIN_VALID_PRICE < px < NIFTY_MAX_VALID_PRICE):
        raise ValueError(f"Futures price {px} outside valid NIFTY range [{NIFTY_MIN_VALID_PRICE}, {NIFTY_MAX_VALID_PRICE}]")
    if not (NIFTY_MIN_VALID_PRICE < und < NIFTY_MAX_VALID_PRICE):
        raise ValueError(f"Underlying price {und} outside valid NIFTY range [{NIFTY_MIN_VALID_PRICE}, {NIFTY_MAX_VALID_PRICE}]")
    if abs(px - und) / und >= NIFTY_MAX_FUTURES_BASIS_PCT:
        raise ValueError(f"Implausible futures basis >= 6%: fut={px}, und={und}")
    return px, und


def validate_nifty_options_row(row: dict[str, Any]) -> float:
    """Validate a NIFTY index options (`IDO`) UDiFF row for exact symbol and underlying range."""
    if row.get("TckrSymb") != "NIFTY":
        raise ValueError(f"Non-NIFTY symbol rejected: {row.get('TckrSymb')!r}")
    und_raw = row.get("UndrlygPric")
    if und_raw in (None, ""):
        raise ValueError(f"Missing UndrlygPric in option row: {row}")
    und = float(und_raw)
    if not (NIFTY_MIN_VALID_PRICE < und < NIFTY_MAX_VALID_PRICE):
        raise ValueError(f"Options underlying {und} outside valid NIFTY range [{NIFTY_MIN_VALID_PRICE}, {NIFTY_MAX_VALID_PRICE}]")
    return und


def build_eod_cm30_session_from_udiff(
    trade_date_str: str,
    udiff_rows: Sequence[dict[str, Any]],
    risk_free_rate: float = 0.0675,
) -> dict[str, Any]:
    """Compute a single EOD `cm30_atm_iv` archive record from official NSE UDiFF option rows."""
    opt_rows = filter_exact_nifty_udiff_rows(udiff_rows, instrument_code="IDO")
    if not opt_rows:
        raise ValueError(f"No NIFTY IDO rows found for {trade_date_str}")

    spot_close = validate_nifty_options_row(opt_rows[0])
    by_expiry_strike: dict[str, dict[float, dict[str, float]]] = {}
    for r in opt_rows:
        validate_nifty_options_row(r)
        exp = str(r["XpryDt"]).strip()
        if exp <= trade_date_str:
            continue
        strike = float(r["StrkPric"])
        opt_tp = str(r["OptnTp"]).strip().upper()
        cls_px = float(r.get("SttlmPric") or r.get("ClsPric") or 0.0)
        if cls_px <= 0.0 or opt_tp not in ("CE", "PE"):
            continue
        by_expiry_strike.setdefault(exp, {}).setdefault(strike, {})[opt_tp] = cls_px

    if not by_expiry_strike:
        raise ValueError(f"No post-trade-date active NIFTY expiries found on {trade_date_str}")

    trade_dt = dt.datetime.combine(
        dt.date.fromisoformat(trade_date_str), dt.time(15, 30, 0), tzinfo=IST
    )
    expiry_t_and_ivs: list[tuple[float, float]] = []
    sorted_expiries = sorted(by_expiry_strike.keys())
    for exp in sorted_expiries:
        strikes_map = by_expiry_strike[exp]
        paired_strikes = [k for k, v in strikes_map.items() if "CE" in v and "PE" in v]
        if not paired_strikes:
            continue
        atm_k = min(paired_strikes, key=lambda k: abs(k - spot_close))
        c_px = strikes_map[atm_k]["CE"]
        p_px = strikes_map[atm_k]["PE"]
        atm_iv = infer_eod_atm_iv_from_archive_row(
            trade_date_str=trade_date_str,
            spot_close=spot_close,
            near_expiry_str=exp,
            atm_strike=atm_k,
            atm_call_close=c_px,
            atm_put_close=p_px,
            risk_free_rate=risk_free_rate,
        )
        t_yrs = compute_time_to_expiry_years(trade_dt, exp)
        expiry_t_and_ivs.append((t_yrs, atm_iv))

    if not expiry_t_and_ivs:
        raise ValueError(f"Could not solve ATM IV for any NIFTY expiry on {trade_date_str}")

    cm30_iv, _, _ = compute_variance_interpolated_cm30_iv(expiry_t_and_ivs, target_days=30.0)
    return {
        "trade_date": trade_date_str,
        "spot_close": round(spot_close, 2),
        "cm30_atm_iv": round(cm30_iv, 6),
        "near_expiry": sorted_expiries[0],
        "source_label": "EOD_CLOSE_INFERRED",
    }


def refresh_eod_volatility_archive(
    archive_path: str | Path,
    out_path: str | Path | None = None,
    udiff_csv_paths: Sequence[str | Path] | None = None,
    max_trade_date: str | None = None,
    risk_free_rate: float = 0.0675,
) -> Path:
    """Refresh or export a point-in-time EOD volatility archive JSON from local UDiFF CSVs."""
    src_path = Path(archive_path)
    target_path = Path(out_path) if out_path else src_path

    sessions_by_date: dict[str, dict[str, Any]] = {}
    if src_path.exists():
        existing = json.loads(src_path.read_text(encoding="utf-8"))
        for row in existing.get("eod_sessions", []):
            sessions_by_date[str(row["trade_date"])] = dict(row)

    for csv_p in udiff_csv_paths or []:
        with Path(csv_p).open("r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        by_date: dict[str, list[dict[str, Any]]] = {}
        for r in filter_exact_nifty_udiff_rows(rows, instrument_code="IDO"):
            t_dt = str(r.get("TradDt", "")).strip()
            if t_dt:
                by_date.setdefault(t_dt, []).append(r)
        for t_dt, day_rows in sorted(by_date.items()):
            rec = build_eod_cm30_session_from_udiff(t_dt, day_rows, risk_free_rate=risk_free_rate)
            sessions_by_date[t_dt] = rec

    ordered = [
        sessions_by_date[d]
        for d in sorted(sessions_by_date.keys())
        if max_trade_date is None or d <= max_trade_date
    ]
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps({"eod_sessions": ordered}, indent=2), encoding="utf-8")
    return target_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Refresh NSE EOD CM30 IV archive from local UDiFF bhavcopy CSVs (IVP only)."
    )
    parser.add_argument("--archive", required=True, help="Existing EOD volatility archive JSON")
    parser.add_argument("--out", default=None, help="Output JSON path (defaults to --archive)")
    parser.add_argument("--date", default=None, help="Optional max trade_date clamp YYYY-MM-DD")
    parser.add_argument(
        "--udiff-csv",
        action="append",
        default=[],
        help="Local NSE UDiFF bhavcopy CSV file(s) to ingest",
    )
    args = parser.parse_args(argv)
    refresh_eod_volatility_archive(
        archive_path=args.archive,
        out_path=args.out,
        udiff_csv_paths=args.udiff_csv,
        max_trade_date=args.date,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
