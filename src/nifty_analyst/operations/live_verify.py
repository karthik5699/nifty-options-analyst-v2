"""Supervised read-only live verification checklist runner (Section 7 / `P2-2`).

Executes each supervised read-only verification gate under a single-stream file lock
with `PacedReadTransport` (`2.5s` spacing, `35s` queue cap, zero `429` retry) and
trader alerts strictly suppressed (`suppress_trader_alerts=True`).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nifty_analyst.adapters.calendar_and_contracts import load_nse_holiday_calendar
from nifty_analyst.adapters.kotak_neo import (
    KotakNeoReadOnlyPort,
    LiveKotakNeoAdapter,
    parse_and_validate_option_chain,
    slice_completed_candles,
)
from nifty_analyst.adapters.request_safety import PacedReadTransport
from nifty_analyst.config import AnalystConfig
from nifty_analyst.operations.decision_driver import SingleInstanceFileLock
from nifty_analyst.pipeline import run_analyst_pipeline
from nifty_analyst.scrubber import IST, ensure_ist


@dataclass(frozen=True)
class LiveVerifyCheckResult:
    step: str
    passed: bool
    summary: str
    details: dict[str, Any]


def verify_live_chain_step(
    adapter: LiveKotakNeoAdapter,
    expiry_date: str,
    decision_dt_ist: dt.datetime,
    expected_strikes: list[float] | None = None,
    min_strike_count: int = 75,
) -> LiveVerifyCheckResult:
    """Section 7.1: Chain at count=40 (~81 strikes) without depth, asserting far strikes present."""
    raw = adapter.fetch_option_chain_raw(expiry_date, decision_dt_ist, fetch_depth=False)
    strikes = [float(s["strike"]) for s in raw.get("strikes", [])]
    expected = expected_strikes or []
    missing = [st for st in expected if st not in set(strikes)]
    passed = len(strikes) >= min_strike_count and not missing
    return LiveVerifyCheckResult(
        step="chain",
        passed=passed,
        summary=(
            f"Fetched {len(strikes)} strikes for {expiry_date} (lot_size={raw.get('lot_size')}); "
            f"missing_expected={missing}"
        ),
        details={
            "expiry_date": expiry_date,
            "strike_count": len(strikes),
            "min_strike": min(strikes) if strikes else None,
            "max_strike": max(strikes) if strikes else None,
            "lot_size": raw.get("lot_size"),
            "missing_expected_strikes": missing,
        },
    )


def verify_live_depth_step(
    adapter: LiveKotakNeoAdapter,
    expiry_date: str,
    decision_dt_ist: dt.datetime,
) -> LiveVerifyCheckResult:
    """Section 7.2: Targeted small-batch depth pull, asserting nonzero bid<ask and live mktLot."""
    raw = adapter.fetch_option_chain_raw(expiry_date, decision_dt_ist, fetch_depth=True)
    env = parse_and_validate_option_chain(raw, expiry_date, decision_dt_ist)
    priced_calls = [q for q in env.quotes if q.has_executable_call]
    priced_puts = [q for q in env.quotes if q.has_executable_put]
    any_zero_leak = any(
        (q.call_bid == 0.0 or q.call_ask == 0.0 or q.put_bid == 0.0 or q.put_ask == 0.0)
        for q in env.quotes
    )
    passed = (
        bool(priced_calls)
        and bool(priced_puts)
        and (env.lot_size is not None and env.lot_size > 0)
        and not any_zero_leak
    )
    return LiveVerifyCheckResult(
        step="depth",
        passed=passed,
        summary=(
            f"Priced {len(priced_calls)} CE and {len(priced_puts)} PE books "
            f"(mktLot={env.lot_size}, rejected={len(env.rejected_depth_books)}, zero_leak={any_zero_leak})"
        ),
        details={
            "expiry_date": expiry_date,
            "lot_size": env.lot_size,
            "shortlisted_count": len(raw.get("shortlisted_neo_symbols", [])),
            "executable_ce_count": len(priced_calls),
            "executable_pe_count": len(priced_puts),
            "rejected_books_count": len(env.rejected_depth_books),
            "any_zero_price_leak": any_zero_leak,
        },
    )


def verify_live_candles_step(
    adapter: KotakNeoReadOnlyPort,
    decision_dt_ist: dt.datetime,
) -> LiveVerifyCheckResult:
    """Section 7.3: Candle pulls across D / 60min / 15min with completed-bar slicing."""
    raw = adapter.fetch_index_candles_raw(decision_dt_ist)
    c1d, c1h, c15m, stripped = slice_completed_candles(raw, decision_dt_ist)
    passed = len(c1d) >= 50 and len(c1h) >= 5
    return LiveVerifyCheckResult(
        step="candles",
        passed=passed,
        summary=(
            f"Completed bars: 1D={len(c1d)}, 1H={len(c1h)}, "
            f"final_15m={'PRESENT' if c15m else 'NONE'}, stripped_forming={stripped}"
        ),
        details={
            "completed_1d_count": len(c1d),
            "completed_1h_count": len(c1h),
            "has_completed_final_15m": c15m is not None,
            "stripped_forming_bars_count": stripped,
        },
    )


def verify_live_dry_run_step(
    decision_dt_ist: dt.datetime,
    config: AnalystConfig,
    broker_adapter: KotakNeoReadOnlyPort | None = None,
) -> LiveVerifyCheckResult:
    """Section 7.4: End-to-end slot dry-run with recommendations OFF (`suppress_trader_alerts=True`)."""
    state, delivery = run_analyst_pipeline(
        decision_dt_ist=decision_dt_ist,
        config=config,
        broker_adapter=broker_adapter,
        suppress_trader_alerts=True,
    )
    fo = state.final_output
    assert fo is not None
    # Must suppress trader-facing alert and remain not-live-ready while FETCH_TIME_NOT_EXCHANGE holds
    passed = (
        delivery.alerted_trader is False
        and fo.verdict in ("SUGGESTION", "WAIT", "INSUFFICIENT DATA")
        and fo.is_live_ready is False
        and (fo.verdict == "INSUFFICIENT DATA" or fo.verifier_report.passed_verification)
    )
    return LiveVerifyCheckResult(
        step="dry-run",
        passed=passed,
        summary=(
            f"verdict={fo.verdict}, is_live_ready={fo.is_live_ready}, "
            f"alerted_trader={delivery.alerted_trader}, "
            f"reason={fo.research_only_reason!r}"
        ),
        details={
            "verdict": fo.verdict,
            "is_live_ready": fo.is_live_ready,
            "alerted_trader": delivery.alerted_trader,
            "research_only_reason": fo.research_only_reason,
            "verifier_passed": fo.verifier_report.passed_verification,
            "candidates_evaluated": len(state.candidate_structures),
        },
    )


def _build_supervised_live_adapter(config: AnalystConfig, batch_size: int) -> LiveKotakNeoAdapter:
    holidays, _ = load_nse_holiday_calendar(config.resolved_holidays_path())
    shared_state_dir = (
        Path(config.cache_dir)
        if config.cache_dir
        else Path.home() / "nifty-analyst"
    )
    cooldown_path = shared_state_dir / "kotak_429_cooldown.json"
    pacing_state_path = shared_state_dir / "kotak_pacing_state.json"
    limiter = PacedReadTransport(
        min_interval_seconds=config.kotak_min_request_interval_seconds,
        max_queue_wait_seconds=config.kotak_max_queue_wait_seconds,
        cooldown_file_path=cooldown_path,
        pacing_state_file_path=pacing_state_path,
    )
    return LiveKotakNeoAdapter(
        kotak_api_root=config.kotak_api_root or None,
        rate_limiter=limiter,
        enforce_schedule_in_live_mode=True,
        depth_strikes_each_side=config.kotak_chain_strike_count,
        quotes_batch_size=batch_size,
        max_depth_tokens_per_expiry=config.kotak_max_depth_tokens_per_expiry,
        cache_dir=config.cache_dir or None,
        trading_calendar={
            "holiday_dates": frozenset(d.isoformat() for d in holidays),
            "special_sessions": config.special_sessions,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nifty-live-verify",
        description="Supervised read-only live verification runner (Section 7)",
    )
    parser.add_argument(
        "--step",
        choices=("chain", "depth", "candles", "dry-run"),
        required=True,
        help="Supervised verification step to execute",
    )
    parser.add_argument(
        "--decision-time-ist",
        default=None,
        help="Approved IST slot ISO-8601 timestamp (defaults to current Asia/Kolkata wall clock)",
    )
    parser.add_argument(
        "--expiry",
        default=None,
        help="Target expiry YYYY-MM-DD (required for chain and depth steps)",
    )
    parser.add_argument(
        "--expect-strikes",
        default="",
        help="Comma-separated strikes expected in count=40 chain (e.g. 24700,24900)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Quotes token batch size (start at verified 2 before bounded probe)",
    )
    parser.add_argument(
        "--lock-file",
        default=None,
        help="Optional path to single-stream lock file",
    )
    args = parser.parse_args(argv)

    config = AnalystConfig.from_env()
    config = dataclasses.replace(config, kotak_quotes_batch_size=args.batch_size)
    dec_dt = (
        ensure_ist(dt.datetime.fromisoformat(args.decision_time_ist))
        if args.decision_time_ist
        else dt.datetime.now(tz=IST)
    )
    lock_path = (
        Path(args.lock_file)
        if args.lock_file
        else (
            Path(config.cache_dir) / "kotak_single_stream.lock"
            if config.cache_dir
            else Path.home() / "nifty-analyst" / "kotak_single_stream.lock"
        )
    )

    with SingleInstanceFileLock(lock_path):
        if args.step == "dry-run":
            res = verify_live_dry_run_step(dec_dt, config)
        else:
            adapter = _build_supervised_live_adapter(config, batch_size=args.batch_size)
            if args.step == "candles":
                res = verify_live_candles_step(adapter, dec_dt)
            else:
                if not args.expiry:
                    master = adapter.fetch_contract_master_raw(dec_dt)
                    args.expiry = master["expiries"][0]["expiry_date"]
                if args.step == "chain":
                    exp_st = [
                        float(x.strip()) for x in args.expect_strikes.split(",") if x.strip()
                    ]
                    res = verify_live_chain_step(adapter, args.expiry, dec_dt, exp_st)
                else:
                    res = verify_live_depth_step(adapter, args.expiry, dec_dt)

    print(json.dumps(dataclasses.asdict(res), indent=2))
    return 0 if res.passed else 1


if __name__ == "__main__":
    sys.exit(main())
