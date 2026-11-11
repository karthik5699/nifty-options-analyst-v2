"""Read-only Kotak API request contract validators, depth-book normalizer, and paced transport.

Enforces the live-verified safety invariants:
1. `validate_history`: requires `exchange|instrument` neosymbol (e.g. `nse_cm|Nifty 50`),
   approved interval (`1min/3min/5min/10min/15min/30min/60min/D/W`), and <= 180-day span on D/W.
2. `validate_quotes_batch`: requires 1..50 `{exchange_segment, instrument_token}` dicts.
3. `interpret_depth_book`: sorts buy levels high-to-low and sell levels low-to-high; fails closed
   to explicit `None` (`UNKNOWN`/unpriced, never `0.0` or `ltp`) on empty, locked, or crossed books,
   preserving the raw payload for audit logging.
4. `PacedReadTransport`: enforces 2.5s serial spacing, a 35s per-run queue wait cap, and a persisted
   HTTP 429 cooldown hard-stop with zero auto-retries.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

ALLOWED_HISTORY_INTERVALS: frozenset[str] = frozenset(
    {"1min", "3min", "5min", "10min", "15min", "30min", "60min", "D", "W"}
)

MAX_HISTORY_SPAN_DAYS: dict[str, int] = {
    "1min": 30,
    "3min": 30,
    "5min": 30,
    "10min": 90,
    "15min": 90,
    "30min": 90,
    "60min": 90,
    "D": 180,
    "W": 180,
}


def _to_pos_float(val: Any) -> float | None:
    if val is None or isinstance(val, bool):
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if not (f > 0.0) or f != f:  # NaN / <= 0 check
        return None
    return f


def _to_pos_int(val: Any) -> int | None:
    if val is None or isinstance(val, bool):
        return None
    try:
        i = int(float(val))
    except (TypeError, ValueError):
        return None
    return i if i > 0 else None


def validate_history(
    neosymbol: str,
    interval: str,
    from_date: str,
    to_date: str,
) -> tuple[dt.date, dt.date]:
    """Validate historical_data arguments before any broker call is attempted."""
    if not isinstance(neosymbol, str) or "|" not in neosymbol:
        raise ValueError(
            f"Invalid neosymbol {neosymbol!r}: expected 'exchange|instrument' (e.g. 'nse_cm|Nifty 50')."
        )
    exch, inst = neosymbol.split("|", 1)
    if not exch.strip() or not inst.strip():
        raise ValueError(
            f"Invalid neosymbol {neosymbol!r}: both exchange and instrument must be non-empty."
        )

    if interval not in ALLOWED_HISTORY_INTERVALS:
        raise ValueError(
            f"Invalid historical_data interval {interval!r}; allowed={sorted(ALLOWED_HISTORY_INTERVALS)}."
        )

    try:
        d_from = dt.date.fromisoformat(str(from_date)[:10])
        d_to = dt.date.fromisoformat(str(to_date)[:10])
    except ValueError as exc:
        raise ValueError(
            f"Invalid historical_data date format ({from_date!r} .. {to_date!r}); expected YYYY-MM-DD."
        ) from exc

    if d_from > d_to:
        raise ValueError(f"historical_data from_date ({d_from}) is after to_date ({d_to}).")

    span_days = (d_to - d_from).days
    max_span = MAX_HISTORY_SPAN_DAYS[interval]
    if span_days > max_span:
        raise ValueError(
            f"historical_data span {span_days} days for interval {interval!r} exceeds "
            f"maximum allowed {max_span} days ({d_from} .. {d_to})."
        )
    return d_from, d_to


def chunk_date_range(
    from_date: dt.date,
    to_date: dt.date,
    max_chunk_days: int = 179,
) -> list[tuple[str, str]]:
    """Split [from_date, to_date] into contiguous inclusive YYYY-MM-DD windows of <= max_chunk_days."""
    if max_chunk_days < 1:
        raise ValueError("max_chunk_days must be >= 1")
    if from_date > to_date:
        raise ValueError(f"from_date ({from_date}) > to_date ({to_date})")

    chunks: list[tuple[str, str]] = []
    cur = from_date
    step = dt.timedelta(days=max_chunk_days)
    one_day = dt.timedelta(days=1)
    while cur <= to_date:
        end = min(to_date, cur + step)
        chunks.append((cur.isoformat(), end.isoformat()))
        cur = end + one_day
    return chunks


def validate_quotes_batch(
    instrument_tokens: Sequence[dict[str, Any]],
    max_tokens: int = 50,
) -> list[dict[str, str]]:
    """Validate client.quotes instrument_tokens batch (1..50 tokens, non-empty segment & token)."""
    if not isinstance(instrument_tokens, Sequence) or isinstance(instrument_tokens, (str, bytes)):
        raise ValueError("instrument_tokens must be a non-empty sequence of token dicts.")
    n = len(instrument_tokens)
    if n < 1 or n > max_tokens:
        raise ValueError(
            f"quotes instrument_tokens count {n} outside allowed [1, {max_tokens}] guard."
        )
    normalized: list[dict[str, str]] = []
    for idx, item in enumerate(instrument_tokens):
        if not isinstance(item, dict):
            raise ValueError(f"instrument_tokens[{idx}] must be a dict, got {type(item).__name__}.")
        seg = str(item.get("exchange_segment") or "").strip()
        tok = str(item.get("instrument_token") or "").strip()
        if not seg or not tok:
            raise ValueError(
                f"instrument_tokens[{idx}] missing exchange_segment or instrument_token: {item!r}"
            )
        normalized.append({"exchange_segment": seg, "instrument_token": tok})
    return normalized


@dataclass(frozen=True)
class DepthBookInterpretation:
    """Normalized order-book top/visible depth result.

    When `is_executable` is False, `best_bid` and `best_ask` are strictly `None`
    (never `0.0` and never `ltp`), and `raw_payload` is retained for audit diagnostics.
    """

    is_executable: bool
    best_bid: float | None
    best_ask: float | None
    bid_depth_lots: int | None
    ask_depth_lots: int | None
    resolved_lot_size: int
    rejection_reason: str = ""
    raw_payload: dict[str, Any] | None = None


def interpret_depth_book(
    quote_payload: dict[str, Any] | None,
    fallback_lot_size: int,
) -> DepthBookInterpretation:
    """Normalize a Kotak `quotes` depth book best-first and fail closed on invalid/crossed books."""
    if not isinstance(quote_payload, dict) or not quote_payload:
        return DepthBookInterpretation(
            is_executable=False,
            best_bid=None,
            best_ask=None,
            bid_depth_lots=None,
            ask_depth_lots=None,
            resolved_lot_size=max(0, int(fallback_lot_size)),
            rejection_reason="MISSING_QUOTE_PAYLOAD",
            raw_payload=quote_payload if isinstance(quote_payload, dict) else None,
        )

    contract_lot = (
        _to_pos_int(quote_payload.get("mktLot"))
        or _to_pos_int(quote_payload.get("lot_size"))
        or _to_pos_int(quote_payload.get("lotSize"))
        or _to_pos_int(fallback_lot_size)
    )
    if contract_lot is None or contract_lot <= 0:
        return DepthBookInterpretation(
            is_executable=False,
            best_bid=None,
            best_ask=None,
            bid_depth_lots=None,
            ask_depth_lots=None,
            resolved_lot_size=0,
            rejection_reason="INVALID_CONTRACT_LOT_SIZE",
            raw_payload=dict(quote_payload),
        )

    depth = quote_payload.get("depth")
    if not isinstance(depth, dict):
        return DepthBookInterpretation(
            is_executable=False,
            best_bid=None,
            best_ask=None,
            bid_depth_lots=None,
            ask_depth_lots=None,
            resolved_lot_size=contract_lot,
            rejection_reason="MISSING_DEPTH_MAP",
            raw_payload=dict(quote_payload),
        )

    raw_buys = depth.get("buy") if isinstance(depth.get("buy"), list) else []
    raw_sells = depth.get("sell") if isinstance(depth.get("sell"), list) else []

    valid_buys: list[tuple[float, int]] = []
    for lvl in raw_buys:
        if not isinstance(lvl, dict):
            continue
        p = _to_pos_float(lvl.get("price") if "price" in lvl else lvl.get("prc"))
        q = _to_pos_int(lvl.get("quantity") if "quantity" in lvl else lvl.get("qty"))
        if p is not None and q is not None:
            valid_buys.append((p, q))

    valid_sells: list[tuple[float, int]] = []
    for lvl in raw_sells:
        if not isinstance(lvl, dict):
            continue
        p = _to_pos_float(lvl.get("price") if "price" in lvl else lvl.get("prc"))
        q = _to_pos_int(lvl.get("quantity") if "quantity" in lvl else lvl.get("qty"))
        if p is not None and q is not None:
            valid_sells.append((p, q))

    if not valid_buys or not valid_sells:
        reason = (
            "EMPTY_BUY_AND_SELL_BOOK"
            if (not valid_buys and not valid_sells)
            else ("EMPTY_BUY_BOOK" if not valid_buys else "EMPTY_SELL_BOOK")
        )
        return DepthBookInterpretation(
            is_executable=False,
            best_bid=None,
            best_ask=None,
            bid_depth_lots=None,
            ask_depth_lots=None,
            resolved_lot_size=contract_lot,
            rejection_reason=reason,
            raw_payload=dict(quote_payload),
        )

    # Sort best-first: highest bid first, lowest ask first
    valid_buys.sort(key=lambda pair: pair[0], reverse=True)
    valid_sells.sort(key=lambda pair: pair[0])

    best_bid = valid_buys[0][0]
    best_ask = valid_sells[0][0]
    if best_bid >= best_ask:
        lock_or_cross = "LOCKED_BOOK" if best_bid == best_ask else "CROSSED_BOOK"
        return DepthBookInterpretation(
            is_executable=False,
            best_bid=None,
            best_ask=None,
            bid_depth_lots=None,
            ask_depth_lots=None,
            resolved_lot_size=contract_lot,
            rejection_reason=f"{lock_or_cross}:bid={best_bid}>=ask={best_ask}",
            raw_payload=dict(quote_payload),
        )

    vis_buy_qty = sum(q for _, q in valid_buys)
    vis_sell_qty = sum(q for _, q in valid_sells)
    bid_lots = vis_buy_qty // contract_lot
    ask_lots = vis_sell_qty // contract_lot

    return DepthBookInterpretation(
        is_executable=True,
        best_bid=round(best_bid, 4),
        best_ask=round(best_ask, 4),
        bid_depth_lots=bid_lots,
        ask_depth_lots=ask_lots,
        resolved_lot_size=contract_lot,
        rejection_reason="",
        raw_payload=None,
    )


class PacedQueueCapExceededError(RuntimeError):
    """Raised when cumulative serial pacing wait would breach the 35s per-run queue cap."""


class PacedCooldownActiveError(RuntimeError):
    """Raised when a persisted HTTP 429 cooldown is still active."""


class PacedReadTransport:
    """Serial read-only request pacer enforcing 2.5s spacing, 35s queue cap, and 429 cooldown."""

    def __init__(
        self,
        min_interval_seconds: float = 2.5,
        max_queue_wait_seconds: float = 35.0,
        cooldown_seconds_on_429: float = 900.0,
        cooldown_file_path: str | Path | None = None,
        clock_fn: Callable[[], float] = time.monotonic,
        wall_time_fn: Callable[[], float] = time.time,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval_seconds = float(min_interval_seconds)
        self.max_queue_wait_seconds = float(max_queue_wait_seconds)
        self.cooldown_seconds_on_429 = float(cooldown_seconds_on_429)
        self.cooldown_file_path = Path(cooldown_file_path) if cooldown_file_path else None
        self._clock_fn = clock_fn
        self._wall_time_fn = wall_time_fn
        self._sleep_fn = sleep_fn
        self._last_request_ts: float | None = None
        self.request_count: int = 0
        self.cumulative_wait_seconds: float = 0.0

    def check_persisted_429_cooldown(self) -> None:
        """Raise PacedCooldownActiveError if a persisted 429 cooldown has not yet expired."""
        if self.cooldown_file_path is None or not self.cooldown_file_path.exists():
            return
        try:
            payload = json.loads(self.cooldown_file_path.read_text(encoding="utf-8"))
            until_epoch = float(payload.get("cooldown_until_epoch", 0.0))
        except Exception:
            return
        now_epoch = self._wall_time_fn()
        if now_epoch < until_epoch:
            rem = round(until_epoch - now_epoch, 1)
            raise PacedCooldownActiveError(
                f"Persisted HTTP 429 cooldown active for another {rem}s "
                f"(reason={payload.get('reason', 'HTTP_429')}); refusing broker call."
            )

    def record_429_cooldown(self, reason: str) -> None:
        """Persist HTTP 429 cooldown timestamp to disk so subsequent runs do not probe the API."""
        if self.cooldown_file_path is None:
            return
        try:
            self.cooldown_file_path.parent.mkdir(parents=True, exist_ok=True)
            now_epoch = self._wall_time_fn()
            payload = {
                "recorded_at_epoch": now_epoch,
                "cooldown_until_epoch": now_epoch + self.cooldown_seconds_on_429,
                "reason": str(reason)[:240],
            }
            tmp = self.cooldown_file_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self.cooldown_file_path)
        except Exception:
            pass

    def wait_for_slot(self) -> None:
        """Enforce serial interval and 35s cumulative queue cap before issuing a read call."""
        self.check_persisted_429_cooldown()
        now = self._clock_fn()
        wait_needed = 0.0
        if self._last_request_ts is not None and self.min_interval_seconds > 0.0:
            elapsed = now - self._last_request_ts
            if elapsed < self.min_interval_seconds:
                wait_needed = self.min_interval_seconds - elapsed

        projected_wait = self.cumulative_wait_seconds + wait_needed
        if self.max_queue_wait_seconds > 0.0 and projected_wait > self.max_queue_wait_seconds + 1e-9:
            raise PacedQueueCapExceededError(
                f"Serial pacing queue cap exceeded: projected wait {projected_wait:.2f}s > "
                f"max_queue_wait_seconds {self.max_queue_wait_seconds:.2f}s after {self.request_count} calls."
            )

        if wait_needed > 0.0:
            self._sleep_fn(wait_needed)
            self.cumulative_wait_seconds += wait_needed

        self._last_request_ts = self._clock_fn()
        self.request_count += 1
