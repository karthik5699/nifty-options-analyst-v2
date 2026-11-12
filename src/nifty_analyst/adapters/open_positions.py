"""Fail-closed open positions loader for Part A Rule 3 (Rs 10,000 portfolio summed-stop cap).

Per the 9 Nov 2026 specification and Instinct's P1-3 review:
- Reads `NIFTY_OPEN_POSITIONS_FILE` (defaulting to `~/nifty-analyst/open_positions.json` in live mode).
- Never silently assumes zero open positions when a required positions file is missing, unreadable,
  or malformed: instead returns `PORTFOLIO_STATE_UNKNOWN`, forcing Rule 3 to `UNKNOWN` and blocking
  `LIVE-READY` output.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from nifty_analyst.state import OpenPosition


@dataclass(frozen=True)
class OpenPositionsLoadResult:
    status: Literal["VERIFIED", "PORTFOLIO_STATE_UNKNOWN"]
    positions: list[OpenPosition]
    error: str = ""
    source_path: str = ""


def _parse_directional_sign(item: dict[str, Any]) -> Literal[1, 0, -1]:
    if "directional_sign" in item and item["directional_sign"] is not None:
        raw_sign = int(item["directional_sign"])
        if raw_sign not in (1, 0, -1):
            raise ValueError(f"directional_sign must be 1, 0, or -1; got {raw_sign}")
        return raw_sign  # type: ignore[return-value]

    direction = str(item.get("direction", "NEUTRAL")).strip().upper()
    if direction in ("BULLISH", "LONG", "+1", "1"):
        return 1
    if direction in ("BEARISH", "SHORT", "-1"):
        return -1
    if direction in ("NEUTRAL", "0"):
        return 0
    raise ValueError(f"Unrecognized direction {direction!r} in open position entry.")


def load_open_positions(
    positions_file: str | Path | None,
    *,
    fail_closed_if_unconfigured: bool = False,
) -> OpenPositionsLoadResult:
    """Load open positions from JSON file, failing closed on missing/unreadable/malformed input.

    Accepts either:
    - A top-level JSON array `[ {...}, ... ]` (empty `[]` means verified zero open positions), OR
    - A top-level JSON object `{"open_positions": [...]}` / `{"positions": [...]}`.
    """
    if positions_file is None or str(positions_file).strip() == "":
        if fail_closed_if_unconfigured:
            return OpenPositionsLoadResult(
                status="PORTFOLIO_STATE_UNKNOWN",
                positions=[],
                error="NIFTY_OPEN_POSITIONS_FILE is not configured; refusing to assume zero positions.",
                source_path="",
            )
        return OpenPositionsLoadResult(
            status="VERIFIED",
            positions=[],
            error="",
            source_path="",
        )

    path = Path(positions_file).expanduser()
    if not path.exists() or not path.is_file():
        return OpenPositionsLoadResult(
            status="PORTFOLIO_STATE_UNKNOWN",
            positions=[],
            error=f"Open positions file missing or not a regular file: {path}",
            source_path=str(path),
        )

    try:
        raw_text = path.read_text(encoding="utf-8")
        payload = json.loads(raw_text)
    except Exception as exc:
        return OpenPositionsLoadResult(
            status="PORTFOLIO_STATE_UNKNOWN",
            positions=[],
            error=f"Failed to read/parse open positions JSON at {path}: {exc}",
            source_path=str(path),
        )

    if isinstance(payload, list):
        raw_items = payload
    elif isinstance(payload, dict):
        if "open_positions" in payload and isinstance(payload["open_positions"], list):
            raw_items = payload["open_positions"]
        elif "positions" in payload and isinstance(payload["positions"], list):
            raw_items = payload["positions"]
        else:
            return OpenPositionsLoadResult(
                status="PORTFOLIO_STATE_UNKNOWN",
                positions=[],
                error=(
                    f"Open positions JSON object at {path} must contain an 'open_positions' "
                    "or 'positions' list."
                ),
                source_path=str(path),
            )
    else:
        return OpenPositionsLoadResult(
            status="PORTFOLIO_STATE_UNKNOWN",
            positions=[],
            error=f"Open positions JSON at {path} must be a list or object, got {type(payload).__name__}.",
            source_path=str(path),
        )

    parsed: list[OpenPosition] = []
    try:
        for idx, item in enumerate(raw_items):
            if not isinstance(item, dict):
                raise ValueError(f"Entry #{idx} is not a JSON object.")
            pos_id = str(
                item.get("position_id")
                or item.get("id")
                or item.get("structure_id")
                or ""
            ).strip()
            if not pos_id:
                raise ValueError(f"Entry #{idx} missing non-empty position_id.")

            if "planned_stop_rs" not in item or item["planned_stop_rs"] is None:
                raise ValueError(f"Position {pos_id!r} missing mandatory planned_stop_rs.")
            stop_rs = float(item["planned_stop_rs"])
            if not math.isfinite(stop_rs) or stop_rs <= 0.0:
                raise ValueError(
                    f"Position {pos_id!r} has non-positive or non-finite planned_stop_rs={stop_rs!r}."
                )

            sign = _parse_directional_sign(item)
            struct_type = str(item.get("structure_type") or "OPEN_POSITION").strip()
            cur_val = float(item.get("current_executable_spread_value_rs") or 0.0)
            raw_deltas = item.get("short_leg_deltas") or []
            if not isinstance(raw_deltas, list):
                raise ValueError(f"Position {pos_id!r} short_leg_deltas must be a list.")
            deltas = [float(d) for d in raw_deltas]
            dte = int(item.get("dte_trading_sessions", 1))

            raw_neos = item.get("neo_symbols") or item.get("instrument_tokens") or []
            neo_syms: list[str] = []
            if isinstance(raw_neos, list):
                for ns in raw_neos:
                    if isinstance(ns, str) and "|" in ns:
                        neo_syms.append(ns.strip())
                    elif isinstance(ns, dict):
                        seg = str(ns.get("exchange_segment") or "").strip()
                        tok = str(ns.get("instrument_token") or "").strip()
                        if seg and tok:
                            neo_syms.append(f"{seg}|{tok}")

            exp_date = str(item.get("expiry_date") or "").strip()
            parsed.append(
                OpenPosition(
                    position_id=pos_id,
                    structure_type=struct_type,
                    directional_sign=sign,
                    planned_stop_rs=round(stop_rs, 2),
                    current_executable_spread_value_rs=round(cur_val, 2),
                    short_leg_deltas=deltas,
                    dte_trading_sessions=dte,
                    neo_symbols=neo_syms,
                    expiry_date=exp_date,
                )
            )
    except Exception as exc:
        return OpenPositionsLoadResult(
            status="PORTFOLIO_STATE_UNKNOWN",
            positions=[],
            error=f"Malformed position record in {path}: {exc}",
            source_path=str(path),
        )

    return OpenPositionsLoadResult(
        status="VERIFIED",
        positions=parsed,
        error="",
        source_path=str(path),
    )
