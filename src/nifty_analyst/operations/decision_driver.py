"""Calendar-aware IST schedule driver & single-stream broker lock (`P2-2`).

Drives the 5 approved market-day IST decision slots (`10:15, 11:15, 12:15, 13:15, 14:15`),
respecting NSE F&O holidays, weekends, verified special circular windows (e.g. Muhurat),
and single-instance file locking so two processes never hit Kotak concurrently.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import fcntl
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from nifty_analyst.adapters.calendar_and_contracts import load_nse_holiday_calendar
from nifty_analyst.adapters.kotak_neo import KotakNeoReadOnlyPort
from nifty_analyst.config import AnalystConfig
from nifty_analyst.output.audit_log import AuditDeliveryResult
from nifty_analyst.pipeline import run_analyst_pipeline
from nifty_analyst.scrubber import APPROVED_RUN_TIMES_IST, IST, ensure_ist, is_approved_schedule_slot
from nifty_analyst.state import NiftyOptionsState, OpenPosition
from nifty_analyst.synthesis.llm_node import LLMProvider


class ConcurrentBrokerStreamError(RuntimeError):
    """Raised when another analyst or verification process holds the single-stream broker lock."""


class SingleInstanceFileLock:
    """Non-blocking exclusive file lock guaranteeing a single coordinated Kotak read stream."""

    def __init__(self, lock_path: str | Path) -> None:
        self.lock_path = Path(lock_path).expanduser()
        self._fd: int | None = None

    def acquire(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise ConcurrentBrokerStreamError(
                f"Another Kotak read stream is already active (lock held at {self.lock_path}). "
                "Aborting to preserve single-stream 2.5s serial pacing."
            ) from exc
        os.ftruncate(fd, 0)
        os.write(fd, f"pid={os.getpid()}\n".encode("utf-8"))
        self._fd = fd

    def release(self) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None

    def __enter__(self) -> "SingleInstanceFileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.release()


def get_approved_slots_for_date(
    target_date: dt.date,
    holidays: set[dt.date] | frozenset[dt.date],
    special_sessions: dict[str, tuple[str, str]] | None = None,
) -> list[dt.datetime]:
    """Return the approved IST run timestamps for `target_date` (empty on holidays/weekends)."""
    iso_d = target_date.isoformat()
    if special_sessions and iso_d in special_sessions:
        open_s, close_s = special_sessions[iso_d]
        oh, om = (int(x) for x in open_s.split(":"))
        ch, cm = (int(x) for x in close_s.split(":"))
        open_dt = dt.datetime.combine(target_date, dt.time(oh, om), tzinfo=IST)
        close_dt = dt.datetime.combine(target_date, dt.time(ch, cm), tzinfo=IST)
        if close_dt < open_dt:
            return []
        # Pick any standard hourly slot inside [open_dt, close_dt], or open_dt + 5 min for short windows
        in_window = [
            dt.datetime.combine(target_date, s.replace(tzinfo=None), tzinfo=IST)
            for s in APPROVED_RUN_TIMES_IST
        ]
        valid_standard = [s_dt for s_dt in in_window if open_dt <= s_dt <= close_dt]
        if valid_standard:
            return valid_standard
        special_slot = min(open_dt + dt.timedelta(minutes=5), close_dt)
        return [special_slot]

    if target_date.weekday() >= 5 or target_date in holidays:
        return []

    return [
        dt.datetime.combine(target_date, s.replace(tzinfo=None), tzinfo=IST)
        for s in APPROVED_RUN_TIMES_IST
    ]


def next_approved_slot_ist(
    now_ist: dt.datetime,
    holidays: set[dt.date] | frozenset[dt.date],
    special_sessions: dict[str, tuple[str, str]] | None = None,
    max_lookahead_days: int = 14,
) -> dt.datetime | None:
    """Return the earliest approved IST run slot at or after `now_ist` within lookahead."""
    cur_ist = ensure_ist(now_ist)
    for offset in range(max(1, max_lookahead_days + 1)):
        cand_date = cur_ist.date() + dt.timedelta(days=offset)
        for slot_dt in get_approved_slots_for_date(cand_date, holidays, special_sessions):
            if slot_dt >= cur_ist:
                return slot_dt
    return None


@dataclass(frozen=True)
class ScheduledSlotResult:
    executed: bool
    slot_approved: bool
    decision_timestamp_ist: str
    skip_reason: str
    state: NiftyOptionsState | None = None
    delivery: AuditDeliveryResult | None = None


def run_scheduled_slot(
    now_ist: dt.datetime,
    config: AnalystConfig,
    lock_path: str | Path | None = None,
    tolerance_seconds: int = 300,
    broker_adapter: KotakNeoReadOnlyPort | None = None,
    llm_provider: LLMProvider | None = None,
    open_positions: Sequence[OpenPosition] | None = None,
    suppress_trader_alerts: bool = True,
) -> ScheduledSlotResult:
    """Execute one scheduled slot if `now_ist` is calendar-approved and no concurrent stream holds the lock."""
    dec_ist = ensure_ist(now_ist)
    holidays, holiday_descs = load_nse_holiday_calendar(config.resolved_holidays_path())
    holiday_iso = frozenset(d.isoformat() for d in holidays)

    approved = is_approved_schedule_slot(
        dec_ist,
        tolerance_seconds=tolerance_seconds,
        holiday_dates=holiday_iso,
        special_sessions=config.special_sessions,
    )
    if not approved:
        d_iso = dec_ist.date().isoformat()
        if dec_ist.date() in holidays:
            reason = f"SKIPPED_NSE_HOLIDAY ({d_iso}: {holiday_descs.get(d_iso, 'NSE Holiday')})"
        elif dec_ist.weekday() >= 5:
            reason = f"SKIPPED_WEEKEND ({d_iso})"
        else:
            reason = f"SKIPPED_OUTSIDE_APPROVED_SLOT ({dec_ist.isoformat()})"
        return ScheduledSlotResult(
            executed=False,
            slot_approved=False,
            decision_timestamp_ist=dec_ist.isoformat(),
            skip_reason=reason,
        )

    resolved_lock = (
        Path(lock_path)
        if lock_path is not None
        else (
            Path(config.cache_dir) / "kotak_single_stream.lock"
            if config.cache_dir
            else Path.home() / "nifty-analyst" / "kotak_single_stream.lock"
        )
    )
    with SingleInstanceFileLock(resolved_lock):
        state, delivery = run_analyst_pipeline(
            decision_dt_ist=dec_ist,
            config=config,
            broker_adapter=broker_adapter,
            llm_provider=llm_provider,
            open_positions=open_positions,
            suppress_trader_alerts=suppress_trader_alerts,
        )
    return ScheduledSlotResult(
        executed=True,
        slot_approved=True,
        decision_timestamp_ist=dec_ist.isoformat(),
        skip_reason="",
        state=state,
        delivery=delivery,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nifty-decision-driver",
        description="Calendar-aware IST decision-slot driver with single-stream lock (`P2-2`)",
    )
    parser.add_argument(
        "--now-ist",
        default=None,
        help="Optional IST ISO-8601 timestamp (defaults to current Asia/Kolkata wall clock)",
    )
    parser.add_argument(
        "--show-next",
        action="store_true",
        help="Print the next calendar-approved IST decision slot and exit",
    )
    parser.add_argument(
        "--open-positions",
        default=None,
        help="Optional path to open_positions.json overriding NIFTY_OPEN_POSITIONS_FILE",
    )
    parser.add_argument(
        "--lock-file",
        default=None,
        help="Optional path to single-stream lock file",
    )
    parser.add_argument(
        "--whatsapp",
        action="store_true",
        help="Emit concise WhatsApp digest instead of full report when actionable",
    )
    parser.add_argument(
        "--suppress-alerts",
        action="store_true",
        default=True,
        help="Keep trader recommendations/alerts suppressed (default: True while live gates are OFF)",
    )
    parser.add_argument(
        "--enable-alerts",
        action="store_false",
        dest="suppress_alerts",
        help="Allow trader alerts only when explicitly authorized and all live-readiness gates pass",
    )
    args = parser.parse_args(argv)

    config = AnalystConfig.from_env()
    if args.open_positions is not None:
        config = dataclasses.replace(config, open_positions_file=args.open_positions)

    now_dt = (
        ensure_ist(dt.datetime.fromisoformat(args.now_ist))
        if args.now_ist
        else dt.datetime.now(tz=IST)
    )
    holidays, _ = load_nse_holiday_calendar(config.resolved_holidays_path())

    if args.show_next:
        nxt = next_approved_slot_ist(now_dt, holidays, config.special_sessions)
        print(nxt.isoformat() if nxt else "NONE_WITHIN_LOOKAHEAD")
        return 0

    res = run_scheduled_slot(
        now_ist=now_dt,
        config=config,
        lock_path=args.lock_file,
        suppress_trader_alerts=args.suppress_alerts,
    )
    if not res.executed:
        return 0

    if res.delivery is not None:
        if args.whatsapp and res.delivery.whatsapp_digest:
            print(res.delivery.whatsapp_digest)
        elif res.delivery.user_facing_message:
            print(res.delivery.user_facing_message)
    return 0


if __name__ == "__main__":
    sys.exit(main())
