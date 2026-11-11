"""Command-Line Interface for the NIFTY 50 Options Research & Strategy Analyst."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from nifty_analyst.config import AnalystConfig
from nifty_analyst.pipeline import run_analyst_pipeline
from nifty_analyst.scrubber import APPROVED_RUN_TIMES_IST, IST, ensure_ist


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nifty-analyst",
        description="Read-only NIFTY 50 Index Options Research & Strategy Analyst (Implementation Plan v3)",
    )
    parser.add_argument(
        "--decision-time-ist",
        default="2026-11-06T10:15:00+05:30",
        help="IST ISO-8601 decision timestamp (default: 2026-11-06T10:15:00+05:30)",
    )
    parser.add_argument(
        "--expiry",
        default=None,
        help="Optional target expiry date YYYY-MM-DD from the contract master",
    )
    parser.add_argument(
        "--inspect-offline",
        action="store_true",
        help="Print the full report even if verdict is WAIT or INSUFFICIENT DATA (for Phase 0 gate inspection)",
    )
    parser.add_argument(
        "--write-report",
        default=None,
        help="Optional file path to write the rendered Markdown report",
    )
    parser.add_argument(
        "--write-state-json",
        default=None,
        help="Optional file path to write the frozen NiftyOptionsState JSON",
    )
    parser.add_argument(
        "--show-schedule",
        action="store_true",
        help="Display the 5 approved IST market-day run times and exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.show_schedule:
        slots = ", ".join(t.strftime("%H:%M IST") for t in APPROVED_RUN_TIMES_IST)
        print(f"Approved Market-Day Run Schedule (Settled 6 Nov 2026): {slots}")
        return 0

    dec_dt = ensure_ist(dt.datetime.fromisoformat(args.decision_time_ist)).astimezone(IST)
    config = AnalystConfig.from_env()
    state, delivery = run_analyst_pipeline(
        decision_dt_ist=dec_dt,
        config=config,
        target_expiry_override=args.expiry,
        force_render_for_inspection=args.inspect_offline,
    )

    if args.write_report:
        out_path = Path(args.write_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(delivery.full_report_markdown, encoding="utf-8")

    if args.write_state_json:
        json_path = Path(args.write_state_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(state.to_json(indent=2), encoding="utf-8")

    # Alert-only output policy: print report only when verdict == SUGGESTION or --inspect-offline
    if delivery.user_facing_message:
        print(delivery.user_facing_message)
    else:
        # Silent audit log write on WAIT / INSUFFICIENT DATA per Section 10
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
