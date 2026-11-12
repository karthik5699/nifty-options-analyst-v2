"""Read-only Kotak API request contract validators, depth-book normalizer, and paced transport.

Enforces the live-verified safety invariants:
1. `validate_history`: requires `exchange|instrument` neosymbol (e.g. `nse_cm|Nifty 50`),
   approved interval (`1min/3min/5min/10min/15min/30min/60min/D/W`), and <= 180-day span on D/W.
2. `validate_quotes_batch`: requires 1..50 `{exchange_segment, instrument_token}` dicts.
3. `interpret_depth_book`: validates original source order before pricing; fails closed
   to explicit `None` (`UNKNOWN`/unpriced, never `0.0` or `ltp`) on empty, locked, or crossed books,
   preserving the raw payload for audit logging.
4. `PacedReadTransport`: enforces 2.5s serial spacing, a 35s per-run queue wait cap, and a persisted
   HTTP 429 cooldown hard-stop with zero auto-retries.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from nifty_analyst.scrubber import IST, ensure_ist

try:
    import httpx as _httpx

    _HttpBaseTransport: type[Any] = _httpx.BaseTransport
except ImportError:  # pragma: no cover - offline environments without kotakneoapi/httpx
    _httpx = None  # type: ignore[assignment]
    _HttpBaseTransport = object

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
    if not math.isfinite(f) or not (f > 0.0):  # NaN / <= 0 check
        return None
    return f


def _to_pos_int(val: Any) -> int | None:
    if val is None or isinstance(val, bool):
        return None
    try:
        i = int(float(val))
    except (TypeError, ValueError, OverflowError):
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

    NOTE ON `is_executable` / `is_structurally_valid`:
    `is_executable` indicates **structural book acceptance only** (monotonic source order,
    positive integer quantities, `0 < best_bid < best_ask`). It NEVER implies full-size
    execution certainty at the displayed touch. Use `touch_bid_qty` / `touch_ask_qty` and
    `walk_depth_levels(buy_levels|sell_levels, required_units)` to evaluate full-size fills.

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
    touch_bid_qty: int | None = None
    touch_ask_qty: int | None = None
    visible_buy_qty: int | None = None
    visible_sell_qty: int | None = None
    buy_levels: tuple[tuple[float, int], ...] = ()
    sell_levels: tuple[tuple[float, int], ...] = ()
    # Diagnostic-only sorted snapshots when source order is inconsistent.
    # NEVER used for best_bid, best_ask, walk_depth_levels, pricing, or risk!
    diagnostic_sorted_buy_levels: tuple[tuple[float, int], ...] = ()
    diagnostic_sorted_sell_levels: tuple[tuple[float, int], ...] = ()

    @property
    def is_structurally_valid(self) -> bool:
        """Qualified alias clarifying `is_executable` is structural touch validity only."""
        return self.is_executable


def walk_depth_levels(
    levels: Sequence[tuple[float, int] | Sequence[float | int]],
    required_units: int,
) -> tuple[float | None, int]:
    """Compute exact quantity-weighted fill price across ordered depth levels for `required_units`.

    Returns `(vwap_price, total_available_units)` when `total_available_units >= required_units`;
    otherwise returns `(None, total_available_units)` without extrapolating unquoted liquidity.
    """
    if required_units <= 0:
        raise ValueError(f"required_units must be > 0, got {required_units}")
    total_avail = 0
    remaining = required_units
    weighted_notional = 0.0
    for row in levels:
        if len(row) < 2:
            continue
        px = float(row[0])
        qty = int(row[1])
        if px <= 0.0 or qty <= 0:
            continue
        total_avail += qty
        if remaining > 0:
            take = min(remaining, qty)
            weighted_notional += px * float(take)
            remaining -= take
    if remaining > 0:
        return None, total_avail
    return weighted_notional / float(required_units), total_avail


def interpret_depth_book(
    quote_payload: dict[str, Any] | None,
    fallback_lot_size: int,
) -> DepthBookInterpretation:
    """Reject inconsistent source order before pricing; preserve rejected raw data.

    Structurally usable touch prices are not a full-size execution guarantee.
    """
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

    # Validate original order before sorting. Sorting mixed updates cannot
    # certify which levels are current, and row-one fallback is not safe.
    def reject(reason: str) -> DepthBookInterpretation:
        return DepthBookInterpretation(
            is_executable=False, best_bid=None, best_ask=None,
            bid_depth_lots=None, ask_depth_lots=None,
            resolved_lot_size=contract_lot, rejection_reason=reason,
            raw_payload=dict(quote_payload),
        )

    valid_sides: dict[str, list[tuple[float, int]]] = {}
    for side in ("buy", "sell"):
        rows = depth.get(side)
        if rows is None:
            rows = []
        if not isinstance(rows, list):
            return reject("DEPTH_SCHEMA_INVALID")
        parsed: list[tuple[float, int]] = []
        for row in rows:
            if not isinstance(row, dict):
                return reject("DEPTH_ROW_INVALID")
            pr = row.get("price") if "price" in row else row.get("prc")
            qr = row.get("quantity") if "quantity" in row else row.get("qty")
            if isinstance(pr, bool) or isinstance(qr, bool):
                return reject("DEPTH_ROW_INVALID")
            try:
                price, quantity = float(pr), float(qr)
            except (TypeError, ValueError, OverflowError):
                return reject("DEPTH_ROW_INVALID")
            if not math.isfinite(price) or not math.isfinite(quantity) or not quantity.is_integer():
                return reject("DEPTH_ROW_INVALID")
            if price == 0 and quantity == 0:
                continue
            if price <= 0 or quantity <= 0:
                return reject("DEPTH_ROW_INVALID")
            parsed.append((price, int(quantity)))
        valid_sides[side] = parsed
    valid_buys, valid_sells = valid_sides["buy"], valid_sides["sell"]

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

    buy_prices = [p for p, _ in valid_buys]
    sell_prices = [p for p, _ in valid_sells]
    if buy_prices != sorted(buy_prices, reverse=True) or sell_prices != sorted(sell_prices):
        return DepthBookInterpretation(
            is_executable=False,
            best_bid=None,
            best_ask=None,
            bid_depth_lots=None,
            ask_depth_lots=None,
            resolved_lot_size=contract_lot,
            rejection_reason="DEPTH_LEVEL_ORDER_INCONSISTENT",
            raw_payload=dict(quote_payload),
            diagnostic_sorted_buy_levels=tuple(
                (round(p, 4), int(q))
                for p, q in sorted(valid_buys, key=lambda r: r[0], reverse=True)
            ),
            diagnostic_sorted_sell_levels=tuple(
                (round(p, 4), int(q))
                for p, q in sorted(valid_sells, key=lambda r: r[0])
            ),
        )
    # Ordered duplicate prices are valid. No source reordering or row-one fallback.

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

    touch_bid_qty = sum(q for p, q in valid_buys if p == best_bid)
    touch_ask_qty = sum(q for p, q in valid_sells if p == best_ask)
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
        touch_bid_qty=touch_bid_qty,
        touch_ask_qty=touch_ask_qty,
        visible_buy_qty=vis_buy_qty,
        visible_sell_qty=vis_sell_qty,
        buy_levels=tuple((round(p, 4), int(q)) for p, q in valid_buys),
        sell_levels=tuple((round(p, 4), int(q)) for p, q in valid_sells),
    )


@dataclass(frozen=True)
class ContractTimestampEvaluation:
    """Per-contract quote timestamp normalization & provenance record.

    Separates `fetch_started_at_ist` / `fetch_completed_at_ist` from per-contract
    `normalized_quote_as_of_ist`. Never substitutes fetch time for missing, malformed,
    or wrong-unit contract timestamps, and keeps unknown REST field semantics (`lstup_time`)
    explicitly tagged `UNVERIFIED_KOTAK_REST_LSTUP_TIME` until upstream REST semantics
    are verified.
    """

    is_valid: bool
    is_certified: bool
    normalized_quote_as_of_ist: str | None
    raw_value: Any
    source_field: str
    detected_unit: str
    semantics_status: str
    age_seconds: float | None = None
    rejection_reason: str = ""


_EPOCH_SECONDS_MIN_2000 = 946_684_800.0
_EPOCH_SECONDS_MAX_2100 = 4_102_444_800.0
_EPOCH_MILLISECONDS_FLOOR = 100_000_000_000.0

_VERIFIED_CONTRACT_TS_FIELDS: tuple[str, ...] = (
    "quote_as_of_ist",
    "book_update_time_ist",
)
_CANDIDATE_CONTRACT_TS_FIELDS: tuple[str, ...] = (
    "lstup_time",
    "exchange_timestamp_ist",
    "quote_updated_at",
)


def normalize_contract_quote_timestamp(
    quote_payload: dict[str, Any] | None,
    reference_dt_ist: dt.datetime,
    *,
    max_staleness_seconds: float = 180.0,
    max_future_skew_seconds: float = 5.0,
    verified_semantics: bool = False,
    require_present: bool = True,
) -> ContractTimestampEvaluation:
    """Normalize and validate a per-contract quote timestamp without fetch-time substitution.

    - Preserves `raw_value`, `source_field`, `detected_unit`, and `semantics_status`.
    - Kotak REST `lstup_time` appears in `docs/functions/market_data/quotes.md` sample JSON
      as a 10-digit Unix-second string (e.g. `"1794455573"`), but is omitted from the REST
      Response Fields table; therefore `semantics_status` defaults to
      `"UNVERIFIED_KOTAK_REST_LSTUP_TIME"` (`is_certified=False`) unless `verified_semantics=True`
      or an explicitly verified field (`quote_as_of_ist` / `book_update_time_ist`) is supplied.
    - Rejects missing (when `require_present=True`), malformed, boolean, NaN/Inf, sentinel (`<= 0`,
      e.g. `0` or `-2209008600`), millisecond/microsecond (`>= 1e11`), out-of-range (`< 2000` or
      `> 2100`), naive-timezone ISO, future (`age < -max_future_skew_seconds`), and stale
      (`age > max_staleness_seconds`) values — NEVER replacing them with fetch time.
    """
    ref_ist = ensure_ist(reference_dt_ist)
    if not isinstance(quote_payload, dict):
        return ContractTimestampEvaluation(
            is_valid=not require_present,
            is_certified=False,
            normalized_quote_as_of_ist=None,
            raw_value=None,
            source_field="",
            detected_unit="MISSING",
            semantics_status="MISSING_CONTRACT_TIMESTAMP",
            rejection_reason="QUOTE_TIMESTAMP_MISSING" if require_present else "",
        )

    source_field = ""
    raw_value: Any = None
    field_inherently_verified = False
    for k in _VERIFIED_CONTRACT_TS_FIELDS:
        if k in quote_payload and quote_payload[k] is not None:
            source_field = k
            raw_value = quote_payload[k]
            field_inherently_verified = True
            break
    if not source_field:
        for k in _CANDIDATE_CONTRACT_TS_FIELDS:
            if k in quote_payload and quote_payload[k] is not None:
                source_field = k
                raw_value = quote_payload[k]
                break

    if not source_field or raw_value is None:
        return ContractTimestampEvaluation(
            is_valid=not require_present,
            is_certified=False,
            normalized_quote_as_of_ist=None,
            raw_value=None,
            source_field="",
            detected_unit="MISSING",
            semantics_status="MISSING_CONTRACT_TIMESTAMP",
            rejection_reason="QUOTE_TIMESTAMP_MISSING" if require_present else "",
        )

    if field_inherently_verified or verified_semantics:
        sem_status = "VERIFIED_CONTRACT_QUOTE_AS_OF"
    elif source_field == "lstup_time":
        sem_status = "UNVERIFIED_KOTAK_REST_LSTUP_TIME"
    else:
        sem_status = "UNVERIFIED_CANDIDATE_FIELD"

    if isinstance(raw_value, bool):
        return ContractTimestampEvaluation(
            is_valid=False,
            is_certified=False,
            normalized_quote_as_of_ist=None,
            raw_value=raw_value,
            source_field=source_field,
            detected_unit="MALFORMED",
            semantics_status="INVALID_CONTRACT_TIMESTAMP",
            rejection_reason=f"QUOTE_TIMESTAMP_MALFORMED:{source_field}=bool",
        )

    parsed_dt: dt.datetime | None = None
    detected_unit = ""

    if isinstance(raw_value, str):
        s = raw_value.strip()
        if not s:
            return ContractTimestampEvaluation(
                is_valid=False,
                is_certified=False,
                normalized_quote_as_of_ist=None,
                raw_value=raw_value,
                source_field=source_field,
                detected_unit="MALFORMED",
                semantics_status="INVALID_CONTRACT_TIMESTAMP",
                rejection_reason=f"QUOTE_TIMESTAMP_MALFORMED:{source_field}=empty",
            )
        # Check ISO-8601 datetime string before numeric epoch parsing
        if "T" in s or ("-" in s[1:] and ":" in s):
            iso_s = s[:-1] + "+00:00" if s.endswith("Z") else s
            try:
                cand_dt = dt.datetime.fromisoformat(iso_s)
            except ValueError:
                return ContractTimestampEvaluation(
                    is_valid=False,
                    is_certified=False,
                    normalized_quote_as_of_ist=None,
                    raw_value=raw_value,
                    source_field=source_field,
                    detected_unit="MALFORMED",
                    semantics_status="INVALID_CONTRACT_TIMESTAMP",
                    rejection_reason=f"QUOTE_TIMESTAMP_MALFORMED:{source_field}={s!r}",
                )
            if cand_dt.tzinfo is None or cand_dt.utcoffset() is None:
                return ContractTimestampEvaluation(
                    is_valid=False,
                    is_certified=False,
                    normalized_quote_as_of_ist=None,
                    raw_value=raw_value,
                    source_field=source_field,
                    detected_unit="NAIVE_ISO8601",
                    semantics_status="INVALID_CONTRACT_TIMESTAMP",
                    rejection_reason=f"QUOTE_TIMESTAMP_NAIVE_TIMEZONE:{source_field}={s!r}",
                )
            parsed_dt = cand_dt.astimezone(IST)
            detected_unit = "ISO8601_IST"

    if parsed_dt is None:
        try:
            num_val = float(raw_value)
        except (TypeError, ValueError, OverflowError):
            return ContractTimestampEvaluation(
                is_valid=False,
                is_certified=False,
                normalized_quote_as_of_ist=None,
                raw_value=raw_value,
                source_field=source_field,
                detected_unit="MALFORMED",
                semantics_status="INVALID_CONTRACT_TIMESTAMP",
                rejection_reason=f"QUOTE_TIMESTAMP_MALFORMED:{source_field}={raw_value!r}",
            )
        if not math.isfinite(num_val):
            return ContractTimestampEvaluation(
                is_valid=False,
                is_certified=False,
                normalized_quote_as_of_ist=None,
                raw_value=raw_value,
                source_field=source_field,
                detected_unit="MALFORMED",
                semantics_status="INVALID_CONTRACT_TIMESTAMP",
                rejection_reason=f"QUOTE_TIMESTAMP_MALFORMED:{source_field}=non_finite",
            )
        if num_val <= 0.0:
            return ContractTimestampEvaluation(
                is_valid=False,
                is_certified=False,
                normalized_quote_as_of_ist=None,
                raw_value=raw_value,
                source_field=source_field,
                detected_unit="WRONG_UNIT_SENTINEL",
                semantics_status="INVALID_CONTRACT_TIMESTAMP",
                rejection_reason=f"QUOTE_TIMESTAMP_WRONG_UNIT_SENTINEL:{source_field}={raw_value!r}",
            )
        if num_val >= _EPOCH_MILLISECONDS_FLOOR:
            return ContractTimestampEvaluation(
                is_valid=False,
                is_certified=False,
                normalized_quote_as_of_ist=None,
                raw_value=raw_value,
                source_field=source_field,
                detected_unit="WRONG_UNIT_EPOCH_MS",
                semantics_status="INVALID_CONTRACT_TIMESTAMP",
                rejection_reason=f"QUOTE_TIMESTAMP_WRONG_UNIT_MS_OR_NS:{source_field}={raw_value!r}",
            )
        if num_val < _EPOCH_SECONDS_MIN_2000 or num_val > _EPOCH_SECONDS_MAX_2100:
            return ContractTimestampEvaluation(
                is_valid=False,
                is_certified=False,
                normalized_quote_as_of_ist=None,
                raw_value=raw_value,
                source_field=source_field,
                detected_unit="WRONG_UNIT_OUT_OF_RANGE",
                semantics_status="INVALID_CONTRACT_TIMESTAMP",
                rejection_reason=f"QUOTE_TIMESTAMP_WRONG_UNIT_RANGE:{source_field}={raw_value!r}",
            )
        parsed_dt = dt.datetime.fromtimestamp(num_val, tz=dt.timezone.utc).astimezone(IST)
        detected_unit = "UNIX_EPOCH_SECONDS"

    norm_iso = parsed_dt.isoformat()
    age_sec = round((ref_ist - parsed_dt).total_seconds(), 6)
    if age_sec < -float(max_future_skew_seconds):
        return ContractTimestampEvaluation(
            is_valid=False,
            is_certified=False,
            normalized_quote_as_of_ist=norm_iso,
            raw_value=raw_value,
            source_field=source_field,
            detected_unit=detected_unit,
            semantics_status=sem_status,
            age_seconds=age_sec,
            rejection_reason=(
                f"QUOTE_TIMESTAMP_FUTURE_SKEW:{source_field}={norm_iso}"
                f"(age={age_sec:.3f}s<-{max_future_skew_seconds}s)"
            ),
        )
    if age_sec > float(max_staleness_seconds):
        return ContractTimestampEvaluation(
            is_valid=False,
            is_certified=False,
            normalized_quote_as_of_ist=norm_iso,
            raw_value=raw_value,
            source_field=source_field,
            detected_unit=detected_unit,
            semantics_status=sem_status,
            age_seconds=age_sec,
            rejection_reason=(
                f"QUOTE_TIMESTAMP_STALE:{source_field}={norm_iso}"
                f"(age={age_sec:.3f}s>{max_staleness_seconds}s)"
            ),
        )

    return ContractTimestampEvaluation(
        is_valid=True,
        is_certified=(sem_status == "VERIFIED_CONTRACT_QUOTE_AS_OF"),
        normalized_quote_as_of_ist=norm_iso,
        raw_value=raw_value,
        source_field=source_field,
        detected_unit=detected_unit,
        semantics_status=sem_status,
        age_seconds=age_sec,
        rejection_reason="",
    )


class PacedQueueCapExceededError(RuntimeError):
    """Raised when cumulative serial pacing wait would breach the 35s per-run queue cap."""


class PacedCooldownActiveError(RuntimeError):
    """Raised when a persisted HTTP 429 cooldown is still active."""


class HttpRateLimit429Error(RuntimeError):
    """Raised immediately at the HTTP transport boundary upon HTTP 429 with zero retry."""


class PacedReadTransport:
    """Serial read-only request pacer enforcing 2.5s spacing, 35s queue cap, and 429 cooldown.

    LIMITATION (`HTTP_BOUNDARY_PACING_UNVERIFIED`):
    Standalone wrapper-level `PacedReadTransport` spaces top-level adapter calls unless wired
    at the underlying `NeoAPI` HTTP client boundary via `SharedPacedHttpTransport` (which
    counts and paces every actual HTTP `GET` request across compound `Kotak_API` wrapper calls
    and coordinates cross-process spacing via `pacing_state_file_path`).
    """

    HTTP_BOUNDARY_PACING_UNVERIFIED: bool = True

    def __init__(
        self,
        min_interval_seconds: float = 2.5,
        max_queue_wait_seconds: float = 35.0,
        cooldown_seconds_on_429: float = 900.0,
        cooldown_file_path: str | Path | None = None,
        pacing_state_file_path: str | Path | None = None,
        strict_guard_seconds: float = 0.0,
        early_wake_guard_seconds: float = 0.001,
        clock_fn: Callable[[], float] = time.monotonic,
        wall_time_fn: Callable[[], float] = time.time,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval_seconds = float(min_interval_seconds)
        self.max_queue_wait_seconds = float(max_queue_wait_seconds)
        self.cooldown_seconds_on_429 = float(cooldown_seconds_on_429)
        self.cooldown_file_path = Path(cooldown_file_path) if cooldown_file_path else None
        self.pacing_state_file_path = (
            Path(pacing_state_file_path) if pacing_state_file_path else None
        )
        self.strict_guard_seconds = max(0.0, float(strict_guard_seconds))
        self.early_wake_guard_seconds = max(0.0, float(early_wake_guard_seconds))
        self._clock_fn = clock_fn
        self._wall_time_fn = wall_time_fn
        self._sleep_fn = sleep_fn
        self._last_request_ts: float | None = None
        self.request_count: int = 0
        self.http_request_count: int = 0
        self.cumulative_wait_seconds: float = 0.0
        self.actual_slept_seconds: float = 0.0
        self.request_timestamps: list[float] = []
        self.http_boundary_attached: bool = False

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

    def _wait_in_process(self) -> float:
        now = self._clock_fn()
        effective_interval = (
            self.min_interval_seconds + self.strict_guard_seconds
            if self.min_interval_seconds > 0.0
            else 0.0
        )
        target_ts: float | None = None
        wait_needed = 0.0
        if self._last_request_ts is not None and effective_interval > 0.0:
            target_ts = self._last_request_ts + effective_interval
            if now < target_ts:
                wait_needed = target_ts - now

        # Also incorporate cross-process wall-clock spacing when pacing_state_file_path is active
        if self.pacing_state_file_path is not None and effective_interval > 0.0:
            try:
                if self.pacing_state_file_path.exists():
                    st = json.loads(self.pacing_state_file_path.read_text(encoding="utf-8"))
                    last_wall = float(st.get("last_request_wall_epoch", 0.0))
                    wall_elapsed = self._wall_time_fn() - last_wall
                    if 0.0 <= wall_elapsed < effective_interval:
                        cross_wait = effective_interval - wall_elapsed
                        if cross_wait > wait_needed:
                            wait_needed = cross_wait
                            target_ts = now + wait_needed
            except Exception:
                pass

        projected_wait = self.cumulative_wait_seconds + wait_needed
        if self.max_queue_wait_seconds > 0.0 and projected_wait > self.max_queue_wait_seconds + 1e-9:
            raise PacedQueueCapExceededError(
                f"Serial pacing queue cap exceeded: projected wait {projected_wait:.2f}s > "
                f"max_queue_wait_seconds {self.max_queue_wait_seconds:.2f}s after {self.request_count} calls."
            )

        actual_slept = 0.0
        if wait_needed > 0.0 and target_ts is not None:
            first_pass = True
            while True:
                cur = self._clock_fn()
                rem = target_ts - cur
                if rem <= 0.0:
                    break
                sleep_dur = rem if first_pass else max(rem, self.early_wake_guard_seconds)
                first_pass = False
                self._sleep_fn(sleep_dur)
                after = self._clock_fn()
                if after <= cur:
                    # Non-advancing synthetic clock in unit test (`sleep_fn=lambda s: None`)
                    actual_slept += wait_needed
                    break
                actual_slept += after - cur
            self.cumulative_wait_seconds += wait_needed
            self.actual_slept_seconds += actual_slept

        stamp = self._clock_fn()
        self._last_request_ts = stamp
        self.request_timestamps.append(stamp)
        self.request_count += 1
        if self.pacing_state_file_path is not None:
            try:
                self.pacing_state_file_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_s = self.pacing_state_file_path.with_suffix(".tmp")
                tmp_s.write_text(
                    json.dumps({"last_request_wall_epoch": self._wall_time_fn()}),
                    encoding="utf-8",
                )
                tmp_s.replace(self.pacing_state_file_path)
            except Exception:
                pass
        return stamp

    def wait_for_slot(self) -> None:
        """Enforce monotonic `>= min_interval_seconds` spacing, 35s queue cap, and cross-process lock."""
        self.check_persisted_429_cooldown()
        if self.pacing_state_file_path is None:
            self._wait_in_process()
            return
        self.pacing_state_file_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.pacing_state_file_path.with_suffix(".lock")
        with lock_path.open("a+", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                self.check_persisted_429_cooldown()
                self._wait_in_process()
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


class SharedPacedHttpTransport(_HttpBaseTransport):
    """HTTP-boundary read-only transport (`httpx.BaseTransport`) shared across compound calls.

    - Wraps an `httpx.BaseTransport` (e.g. `NeoAPI(..., transport=...)` or `rest_client._client._transport`).
    - Rejects any non-`GET` HTTP method before touching the wire.
    - Enforces shared monotonic `>= 2.5s` spacing (with `strict_guard_seconds` + early-wakeup loop)
      on EVERY actual HTTP request, even when a single `Kotak_API` wrapper method issues multiple
      internal HTTP calls (`option_chain`, `spot`, `futures`).
    - Records `http_request_count` and `http_request_timestamps` separately from adapter-level calls.
    - Immediately records persisted 429 cooldown and raises `HttpRateLimit429Error` with ZERO retry
      whenever an HTTP 429 status or exception occurs.
    """

    def __init__(
        self,
        inner_transport: Any,
        pacer: PacedReadTransport,
    ) -> None:
        self._inner = inner_transport
        self.pacer = pacer
        self.pacer.http_boundary_attached = True
        self.http_request_count: int = 0
        self.http_request_timestamps: list[float] = []

    def handle_request(self, request: Any) -> Any:
        method = str(getattr(request, "method", "GET") or "GET").upper()
        if method != "GET":
            raise PermissionError(
                f"SharedPacedHttpTransport blocked non-GET HTTP method {method!r}; "
                "analyst market-data transport is strictly read-only."
            )
        self.pacer.wait_for_slot()
        self.http_request_count += 1
        self.pacer.http_request_count += 1
        if self.pacer.request_timestamps:
            self.http_request_timestamps.append(self.pacer.request_timestamps[-1])

        try:
            response = self._inner.handle_request(request)
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "too many" in msg.lower():
                self.pacer.record_429_cooldown(msg)
                raise HttpRateLimit429Error(
                    f"HTTP 429 at Kotak HTTP transport boundary ({msg[:200]}); hard stop, zero retries."
                ) from exc
            raise

        status_code = getattr(response, "status_code", None)
        if status_code == 429 or str(status_code) == "429":
            self.pacer.record_429_cooldown("HTTP_429_STATUS")
            raise HttpRateLimit429Error(
                "HTTP 429 response at Kotak HTTP transport boundary; hard stop, zero retries."
            )
        return response

    def close(self) -> None:
        close_fn = getattr(self._inner, "close", None)
        if callable(close_fn):
            close_fn()
