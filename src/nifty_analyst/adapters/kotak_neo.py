"""Adapter 2 (Option Chain & Quotes) and Adapter 3 (Historical Index OHLC) for Kotak Neo.

STRICT CONSTRAINTS (Implementation Plan v3, Sections 3 & 10 + 11 Nov Live Review Fixes):
- Read-only interface: ZERO order placement, modification, or broker write calls exist anywhere.
- Targeted two-stage live depth funnel (`P1-1`):
  1. Fetches option chain structure (`count=40` default, ~81 strikes) WITHOUT `include_depth=True`.
  2. Shortlists near-ATM + candidate wing + held-position `neo_symbol`s and fetches 5-level depth
     in verified small batches (`quotes_batch_size=2` default) via `client.quotes(..., quote_type="all")`.
  3. Normalizes books best-first via `interpret_depth_book`; empty, locked, crossed, unfetched, or
     LTP-only books are represented as explicit `None` (`UNKNOWN`/unpriced), NEVER `0.0` and NEVER `ltp`.
  4. Rejected raw depth payloads are preserved for audit logging, and live per-contract `mktLot`
     drives lot conversion and expiry rupee math.
- Historical candle safety (`P1-2`):
  - Enforces `validate_history` (`nse_cm|Nifty 50`, intervals `D` / `60min` / `15min`, `<= 179d` chunks
    for `D`), handles both bare-list (intraday) and dict-wrapped (daily) payloads, and caches completed
    prior-day `1D` candles per IST trading date to stay comfortably inside the 35s pacing queue cap.
"""

from __future__ import annotations

import abc
import contextvars
import datetime as dt
import inspect
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, NoReturn, Sequence

from nifty_analyst.adapters.request_safety import (
    CooldownPersistence429Error,
    HttpRateLimit429Error,
    PacedCooldownActiveError,
    PacedQueueCapExceededError,
    PacedReadTransport,
    SharedPacedHttpTransport,
    SharedSafetyStateError,
    attach_paced_transport_to_client,
    build_paced_client_transport,
    chunk_date_range,
    interpret_depth_book,
    normalize_contract_quote_timestamp,
    validate_history,
    validate_quotes_batch,
)
from nifty_analyst.engine.pricing import RawStrikeQuote
from nifty_analyst.scrubber import IST, ensure_ist, is_approved_schedule_slot
from nifty_analyst.state import CandleBar, TimestampedDatum


class KotakAdapterError(RuntimeError):
    """Base error for read-only Kotak adapter failures."""


class KotakRateLimit429Error(KotakAdapterError):
    """Raised immediately on HTTP 429; halts the run with zero retry and zero probing."""


class KotakSharedStateError(KotakAdapterError, SharedSafetyStateError):
    """Raised (fail-closed) when shared pacing or 429 cooldown state is corrupt or unwritable."""


class KotakCooldownPersistence429Error(KotakRateLimit429Error, KotakSharedStateError):
    """Raised when HTTP 429 occurs AND persisting the shared cooldown state file fails."""


_IN_AUTH_LOGIN_FLOW: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_IN_AUTH_LOGIN_FLOW", default=False
)


class MissingQuoteTimestampError(KotakAdapterError):
    """Raised when an option quote is missing a quote-level update_time timestamp."""


class StaleOptionQuoteError(KotakAdapterError):
    """Raised when one or more active option strike quotes exceed the 180s staleness bound."""


class UnapprovedScheduleSlotError(KotakAdapterError):
    """Raised when a live broker read is attempted outside the 5 approved IST run times."""


@dataclass(frozen=True)
class ChainSnapshotEnvelope:
    expiry_date: str
    quotes: list[RawStrikeQuote]
    earliest_quote_ist: str
    latest_quote_ist: str
    any_stale: bool
    stale_strikes: list[float]
    source_verification_tag: str
    lot_size: int | None = None
    rejected_depth_books: list[dict[str, Any]] = field(default_factory=list)
    priced_neo_symbols: list[str] = field(default_factory=list)
    chain_neo_symbols: list[str] = field(default_factory=list)
    fetch_started_at_ist: str = ""
    fetch_completed_at_ist: str = ""
    coverage_ledger: dict[str, Any] = field(default_factory=dict)


class ProvisionalSerialRateLimiter:
    """Single-threaded serial rate limiter (default 1 request per 2.5s, provisional/unverified).

    Never retries on HTTP 429 and never probes rate limits.
    """

    def __init__(
        self,
        min_interval_seconds: float = 2.5,
        clock_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))
        self._clock_fn = clock_fn
        self._sleep_fn = sleep_fn
        self._last_request_ts: float | None = None
        self.request_count: int = 0

    def wait_for_slot(self) -> None:
        now = self._clock_fn()
        if self._last_request_ts is not None:
            elapsed = now - self._last_request_ts
            if elapsed < self.min_interval_seconds:
                self._sleep_fn(self.min_interval_seconds - elapsed)
        self._last_request_ts = self._clock_fn()
        self.request_count += 1


class KotakNeoReadOnlyPort(abc.ABC):
    """Abstract read-only port for Kotak Neo market data.

    Contains strictly read-only methods:
    - fetch_contract_master_raw
    - fetch_option_chain_raw
    - fetch_index_candles_raw
    - fetch_india_vix_raw
    NO order placement, modification, cancellation, or write methods exist.
    """

    @abc.abstractmethod
    def fetch_contract_master_raw(self, decision_dt_ist: dt.datetime) -> dict[str, Any]:
        """Read-only fetch of NIFTY contract master and active expiries."""

    @abc.abstractmethod
    def fetch_option_chain_raw(
        self, expiry_date: str, decision_dt_ist: dt.datetime
    ) -> dict[str, Any]:
        """Read-only fetch of NIFTY option chain quotes with quote-level timestamps."""

    @abc.abstractmethod
    def fetch_index_candles_raw(self, decision_dt_ist: dt.datetime) -> dict[str, Any]:
        """Read-only fetch of NIFTY 1D, 1H, and 15M index OHLC candles."""

    @abc.abstractmethod
    def fetch_india_vix_raw(
        self, decision_dt_ist: dt.datetime
    ) -> TimestampedDatum[float | None]:
        """Read-only fetch of India VIX [UNVERIFIED SOURCE]."""


class FixtureKotakNeoAdapter(KotakNeoReadOnlyPort):
    """Fixture-backed read-only adapter for 100% offline execution and contract testing."""

    def __init__(
        self,
        fixtures_dir: str | Path,
        rate_limiter: ProvisionalSerialRateLimiter | PacedReadTransport | None = None,
        simulate_http_429: bool = False,
        enforce_schedule_in_live_mode: bool = False,
    ) -> None:
        self.fixtures_dir = Path(fixtures_dir)
        self.rate_limiter = rate_limiter or ProvisionalSerialRateLimiter(min_interval_seconds=0.0)
        self.simulate_http_429 = simulate_http_429
        self.enforce_schedule_in_live_mode = enforce_schedule_in_live_mode
        self._cache: dict[str, Any] = {}

    def _guard_request(self, cache_key: str, decision_dt_ist: dt.datetime) -> None:
        dec_ist = ensure_ist(decision_dt_ist)
        if self.enforce_schedule_in_live_mode and not is_approved_schedule_slot(dec_ist):
            raise UnapprovedScheduleSlotError(
                f"Live Kotak read rejected at {dec_ist.isoformat()}: outside the 5 approved "
                f"IST run slots (10:15, 11:15, 12:15, 13:15, 14:15 IST)."
            )
        if cache_key in self._cache:
            return
        self.rate_limiter.wait_for_slot()
        if self.simulate_http_429:
            raise KotakRateLimit429Error(
                "HTTP 429 Too Many Requests received from Kotak Neo API. "
                "Stopping run immediately with zero retries and zero limit probing."
            )

    def fetch_contract_master_raw(self, decision_dt_ist: dt.datetime) -> dict[str, Any]:
        cache_key = f"contract_master:{ensure_ist(decision_dt_ist).date().isoformat()}"
        self._guard_request(cache_key, decision_dt_ist)
        if cache_key not in self._cache:
            path = self.fixtures_dir / "contract_master.json"
            self._cache[cache_key] = json.loads(path.read_text(encoding="utf-8"))
        return self._cache[cache_key]

    def fetch_option_chain_raw(
        self, expiry_date: str, decision_dt_ist: dt.datetime
    ) -> dict[str, Any]:
        cache_key = f"chain:{expiry_date}:{ensure_ist(decision_dt_ist).isoformat()}"
        self._guard_request(cache_key, decision_dt_ist)
        if cache_key not in self._cache:
            path = self.fixtures_dir / "option_chains.json"
            all_chains = json.loads(path.read_text(encoding="utf-8"))
            if expiry_date not in all_chains.get("chains_by_expiry", {}):
                raise KotakAdapterError(f"Expiry {expiry_date} not found in option_chains.json")
            chain_copy = json.loads(json.dumps(all_chains["chains_by_expiry"][expiry_date]))
            lot_sz = int(chain_copy.get("lot_size") or 0)
            if lot_sz <= 0:
                master_path = self.fixtures_dir / "contract_master.json"
                if master_path.exists():
                    master_obj = json.loads(master_path.read_text(encoding="utf-8"))
                    for exp_row in master_obj.get("expiries", []):
                        if str(exp_row.get("expiry_date")) == expiry_date:
                            lot_sz = int(exp_row.get("lot_size") or 0)
                            break
            if lot_sz <= 0:
                lot_sz = 75
            chain_copy.setdefault("lot_size", lot_sz)
            for st_row in chain_copy.get("strikes", []):
                c_lots = int(st_row.get("call_depth_lots", 10))
                p_lots = int(st_row.get("put_depth_lots", 10))
                c_qty = max(0, c_lots * lot_sz)
                p_qty = max(0, p_lots * lot_sz)
                if st_row.get("call_bid") and st_row.get("call_ask") and c_qty > 0:
                    st_row.setdefault("call_touch_bid_qty", c_qty)
                    st_row.setdefault("call_touch_ask_qty", c_qty)
                    st_row.setdefault("call_visible_buy_qty", c_qty)
                    st_row.setdefault("call_visible_sell_qty", c_qty)
                    st_row.setdefault("call_buy_levels", [[float(st_row["call_bid"]), c_qty]])
                    st_row.setdefault("call_sell_levels", [[float(st_row["call_ask"]), c_qty]])
                if st_row.get("put_bid") and st_row.get("put_ask") and p_qty > 0:
                    st_row.setdefault("put_touch_bid_qty", p_qty)
                    st_row.setdefault("put_touch_ask_qty", p_qty)
                    st_row.setdefault("put_visible_buy_qty", p_qty)
                    st_row.setdefault("put_visible_sell_qty", p_qty)
                    st_row.setdefault("put_buy_levels", [[float(st_row["put_bid"]), p_qty]])
                    st_row.setdefault("put_sell_levels", [[float(st_row["put_ask"]), p_qty]])
            self._cache[cache_key] = chain_copy
        return self._cache[cache_key]

    def fetch_index_candles_raw(self, decision_dt_ist: dt.datetime) -> dict[str, Any]:
        cache_key = f"candles:{ensure_ist(decision_dt_ist).isoformat()}"
        self._guard_request(cache_key, decision_dt_ist)
        if cache_key not in self._cache:
            path = self.fixtures_dir / "index_candles.json"
            self._cache[cache_key] = json.loads(path.read_text(encoding="utf-8"))
        return self._cache[cache_key]

    def fetch_india_vix_raw(
        self, decision_dt_ist: dt.datetime
    ) -> TimestampedDatum[float | None]:
        dec_ist = ensure_ist(decision_dt_ist)
        cache_key = f"vix:{dec_ist.isoformat()}"
        self._guard_request(cache_key, dec_ist)
        path = self.fixtures_dir / "vix_and_events.json"
        if not path.exists():
            return TimestampedDatum.create(
                value=None,
                as_of_dt=dec_ist,
                fetched_at_dt=dec_ist,
                source_id="kotak_vix_unverified",
                max_staleness_seconds=180,
                verification_tag="UNVERIFIED",
            )
        raw = json.loads(path.read_text(encoding="utf-8")).get("india_vix", {})
        as_of = ensure_ist(dt.datetime.fromisoformat(raw.get("as_of_ist", dec_ist.isoformat())))
        return TimestampedDatum.create(
            value=float(raw["value"]) if raw.get("value") is not None else None,
            as_of_dt=as_of,
            fetched_at_dt=dec_ist,
            source_id=str(raw.get("source_id", "india_vix_feed_unverified")),
            max_staleness_seconds=180,
            verification_tag="UNVERIFIED",
        )


class UnverifiedLiveKotakNeoAdapter(KotakNeoReadOnlyPort):
    """Isolated stub for live Kotak Neo SDK integration [UNVERIFIED SDK INTERFACE]."""

    ENDPOINT_VERIFICATION_STATUS = "UNVERIFIED"

    def __init__(self, rate_limiter: ProvisionalSerialRateLimiter | None = None) -> None:
        self.rate_limiter = rate_limiter or ProvisionalSerialRateLimiter(min_interval_seconds=2.5)

    def _raise_unverified(self, endpoint_tag: str) -> Any:
        raise KotakAdapterError(
            f"[UNVERIFIED SDK INTERFACE: {endpoint_tag}] Real Kotak Neo SDK method names and "
            f"response schemas are not yet verified against live broker documentation. "
            f"Use FixtureKotakNeoAdapter in OFFLINE mode until SDK mappings are verified."
        )

    def fetch_contract_master_raw(self, decision_dt_ist: dt.datetime) -> dict[str, Any]:
        return self._raise_unverified("KOTAK_SCRIP_MASTER_ENDPOINT_UNVERIFIED")

    def fetch_option_chain_raw(
        self, expiry_date: str, decision_dt_ist: dt.datetime
    ) -> dict[str, Any]:
        return self._raise_unverified("KOTAK_QUOTES_CHAIN_ENDPOINT_UNVERIFIED")

    def fetch_index_candles_raw(self, decision_dt_ist: dt.datetime) -> dict[str, Any]:
        return self._raise_unverified("KOTAK_HISTORICAL_CANDLES_ENDPOINT_UNVERIFIED")

    def fetch_india_vix_raw(
        self, decision_dt_ist: dt.datetime
    ) -> TimestampedDatum[float | None]:
        return self._raise_unverified("KOTAK_INDIA_VIX_ENDPOINT_UNVERIFIED")


def _coerce_executable_pair(
    bid_raw: Any, ask_raw: Any, ltp_only: bool
) -> tuple[float | None, float | None]:
    """Return `(bid, ask)` ONLY when both are finite positive numbers with `bid < ask`.

    Otherwise returns `(None, None)` — NEVER `0.0` and NEVER `ltp`.
    """
    if ltp_only or bid_raw is None or ask_raw is None:
        return None, None
    try:
        b = float(bid_raw)
        a = float(ask_raw)
    except (TypeError, ValueError):
        return None, None
    if not (b > 0.0 and a > 0.0 and b < a):
        return None, None
    return b, a


def parse_and_validate_option_chain(
    raw_chain: dict[str, Any],
    expiry_date: str,
    decision_dt_ist: dt.datetime,
    max_staleness_seconds: int = 180,
) -> ChainSnapshotEnvelope:
    """Parse raw option chain quotes, enforcing quote-level update_time and 180s staleness bound.

    Missing, LTP-only, empty, locked, or crossed books remain `None` (`UNKNOWN`/unpriced),
    never `0.0`, preventing zero-price contamination in marks, stops, or Chain QA.
    """
    dec_ist = ensure_ist(decision_dt_ist)
    strikes_raw = raw_chain.get("strikes", [])
    if not strikes_raw:
        raise KotakAdapterError(f"Empty option chain strikes for expiry {expiry_date}.")

    parsed_quotes: list[RawStrikeQuote] = []
    quote_times: list[dt.datetime] = []
    stale_strikes: list[float] = []
    ts_sources = {str(it.get("quote_ts_source", "EXCHANGE")) for it in strikes_raw}

    for item in strikes_raw:
        strike_val = float(item["strike"])
        c_asof = str(item.get("call_quote_as_of_ist") or "").strip()
        p_asof = str(item.get("put_quote_as_of_ist") or "").strip()
        ts_str = str(item.get("update_time_ist", "")).strip()
        if not ts_str and (c_asof or p_asof):
            ts_str = c_asof or p_asof
        if not ts_str:
            raise MissingQuoteTimestampError(
                f"Option quote at strike {strike_val} (expiry {expiry_date}) is missing "
                f"mandatory quote-level update_time_ist timestamp."
            )
        quote_dt = ensure_ist(dt.datetime.fromisoformat(ts_str))
        quote_times.append(quote_dt)

        strike_is_stale = False
        age_sec = (dec_ist - quote_dt).total_seconds()
        if age_sec > max_staleness_seconds or age_sec < -5.0:
            strike_is_stale = True

        c_quote_iso = ""
        p_quote_iso = ""
        for side_iso, target_side in ((c_asof, "CE"), (p_asof, "PE")):
            if side_iso:
                side_dt = ensure_ist(dt.datetime.fromisoformat(side_iso))
                quote_times.append(side_dt)
                side_age = (dec_ist - side_dt).total_seconds()
                if side_age > max_staleness_seconds or side_age < -5.0:
                    strike_is_stale = True
                if target_side == "CE":
                    c_quote_iso = side_dt.isoformat()
                else:
                    p_quote_iso = side_dt.isoformat()

        if strike_is_stale:
            stale_strikes.append(strike_val)

        c_bid, c_ask = _coerce_executable_pair(
            item.get("call_bid"), item.get("call_ask"), bool(item.get("call_ltp_only"))
        )
        p_bid, p_ask = _coerce_executable_pair(
            item.get("put_bid"), item.get("put_ask"), bool(item.get("put_ltp_only"))
        )

        c_exec = c_bid is not None and c_ask is not None
        p_exec = p_bid is not None and p_ask is not None

        def _opt_int(key: str, active: bool) -> int | None:
            v = item.get(key)
            return int(v) if (active and v is not None) else None

        def _opt_levels(key: str, active: bool) -> tuple[tuple[float, int], ...]:
            if not active:
                return ()
            raw_lvls = item.get(key)
            if not isinstance(raw_lvls, (list, tuple)):
                return ()
            out_lvls: list[tuple[float, int]] = []
            for r in raw_lvls:
                if isinstance(r, (list, tuple)) and len(r) >= 2:
                    out_lvls.append((float(r[0]), int(r[1])))
            return tuple(out_lvls)

        c_prov = item.get("call_ts_provenance") if isinstance(item.get("call_ts_provenance"), dict) else {}
        p_prov = item.get("put_ts_provenance") if isinstance(item.get("put_ts_provenance"), dict) else {}

        parsed_quotes.append(
            RawStrikeQuote(
                strike=strike_val,
                call_bid=c_bid,
                call_ask=c_ask,
                put_bid=p_bid,
                put_ask=p_ask,
                call_oi=int(item.get("call_oi", 0)),
                put_oi=int(item.get("put_oi", 0)),
                call_volume=int(item.get("call_volume", 0)),
                put_volume=int(item.get("put_volume", 0)),
                call_depth_lots=int(item.get("call_depth_lots", 10)) if c_exec else 0,
                put_depth_lots=int(item.get("put_depth_lots", 10)) if p_exec else 0,
                quote_timestamp_ist=quote_dt.isoformat(),
                expiry_date=expiry_date,
                call_bid_depth_lots=_opt_int("call_bid_depth_lots", c_exec),
                call_ask_depth_lots=_opt_int("call_ask_depth_lots", c_exec),
                put_bid_depth_lots=_opt_int("put_bid_depth_lots", p_exec),
                put_ask_depth_lots=_opt_int("put_ask_depth_lots", p_exec),
                call_touch_bid_qty=_opt_int("call_touch_bid_qty", c_exec),
                call_touch_ask_qty=_opt_int("call_touch_ask_qty", c_exec),
                put_touch_bid_qty=_opt_int("put_touch_bid_qty", p_exec),
                put_touch_ask_qty=_opt_int("put_touch_ask_qty", p_exec),
                call_visible_buy_qty=_opt_int("call_visible_buy_qty", c_exec),
                call_visible_sell_qty=_opt_int("call_visible_sell_qty", c_exec),
                put_visible_buy_qty=_opt_int("put_visible_buy_qty", p_exec),
                put_visible_sell_qty=_opt_int("put_visible_sell_qty", p_exec),
                call_buy_levels=_opt_levels("call_buy_levels", c_exec),
                call_sell_levels=_opt_levels("call_sell_levels", c_exec),
                put_buy_levels=_opt_levels("put_buy_levels", p_exec),
                put_sell_levels=_opt_levels("put_sell_levels", p_exec),
                call_quote_timestamp_ist=c_quote_iso,
                put_quote_timestamp_ist=p_quote_iso,
                fetch_started_at_ist=str(item.get("fetch_started_at_ist") or raw_chain.get("fetch_started_at_ist") or ""),
                fetch_completed_at_ist=str(item.get("fetch_completed_at_ist") or raw_chain.get("fetch_completed_at_ist") or ""),
                call_ts_semantics=str(c_prov.get("semantics_status") or ""),
                put_ts_semantics=str(p_prov.get("semantics_status") or ""),
            )
        )

    earliest = min(quote_times).isoformat()
    latest = max(quote_times).isoformat()
    tag = str(raw_chain.get("verification_tag", "UNVERIFIED_KOTAK_SCHEMA"))
    verified_only_sources = {"EXCHANGE", "VERIFIED_CONTRACT_QUOTE_AS_OF"}
    if not ts_sources.issubset(verified_only_sources):
        tag = tag + "|TS_" + "+".join(sorted(ts_sources))

    raw_lot = raw_chain.get("lot_size")
    chain_lot = int(raw_lot) if (raw_lot is not None and int(raw_lot) > 0) else None
    rejected_books = list(raw_chain.get("rejected_depth_books") or [])
    priced_neos = [str(s) for s in (raw_chain.get("priced_neo_symbols") or [])]
    chain_neos = [str(s) for s in (raw_chain.get("chain_neo_symbols") or [])]
    cov_ledger = dict(raw_chain.get("coverage_ledger") or {})
    if not cov_ledger:
        priced_fallback = list(priced_neos)
        listed_fallback = list(chain_neos)
        cov_ledger = {
            "expiry_date": expiry_date,
            "listed": listed_fallback,
            "selected": list(raw_chain.get("shortlisted_neo_symbols") or priced_fallback),
            "requested": list(raw_chain.get("requested_neo_symbols") or priced_fallback),
            "returned": list(raw_chain.get("returned_neo_symbols") or priced_fallback),
            "missing": list(raw_chain.get("missing_neo_symbols") or []),
            "rejected": [str(r.get("neo_symbol") or "") for r in rejected_books if r.get("neo_symbol")],
            "priced": priced_fallback,
            "counts": {
                "listed": len(listed_fallback),
                "selected": len(raw_chain.get("shortlisted_neo_symbols") or priced_fallback),
                "requested": len(raw_chain.get("requested_neo_symbols") or priced_fallback),
                "returned": len(raw_chain.get("returned_neo_symbols") or priced_fallback),
                "missing": len(raw_chain.get("missing_neo_symbols") or []),
                "rejected": len(rejected_books),
                "priced": len(priced_fallback),
            },
        }

    return ChainSnapshotEnvelope(
        expiry_date=expiry_date,
        quotes=sorted(parsed_quotes, key=lambda q: q.strike),
        earliest_quote_ist=earliest,
        latest_quote_ist=latest,
        any_stale=len(stale_strikes) > 0,
        stale_strikes=stale_strikes,
        source_verification_tag=tag,
        lot_size=chain_lot,
        rejected_depth_books=rejected_books,
        priced_neo_symbols=priced_neos,
        chain_neo_symbols=chain_neos,
        fetch_started_at_ist=str(raw_chain.get("fetch_started_at_ist") or ""),
        fetch_completed_at_ist=str(raw_chain.get("fetch_completed_at_ist") or ""),
        coverage_ledger=cov_ledger,
    )


# Standard NSE intraday candle slots: six 1H bars (09:15-15:15) + final 15m bar (15:15-15:30 IST)
VALID_1H_OPEN_CLOSE_PAIRS: set[tuple[dt.time, dt.time]] = {
    (dt.time(9, 15), dt.time(10, 15)),
    (dt.time(10, 15), dt.time(11, 15)),
    (dt.time(11, 15), dt.time(12, 15)),
    (dt.time(12, 15), dt.time(13, 15)),
    (dt.time(13, 15), dt.time(14, 15)),
    (dt.time(14, 15), dt.time(15, 15)),
}
FINAL_15M_OPEN_CLOSE_PAIR: tuple[dt.time, dt.time] = (dt.time(15, 15), dt.time(15, 30))


def _parse_kotak_candle_timestamp(raw_ts: Any, is_daily: bool = False) -> dt.datetime | None:
    """Parse Kotak historical_data candle timestamp into IST datetime."""
    if raw_ts is None:
        return None
    if isinstance(raw_ts, (int, float)):
        # Epoch seconds or milliseconds
        sec = float(raw_ts) / 1000.0 if float(raw_ts) > 1e11 else float(raw_ts)
        return dt.datetime.fromtimestamp(sec, tz=IST)
    s = str(raw_ts).strip()
    if not s:
        return None
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        d = dt.date.fromisoformat(s)
        t = dt.time(9, 15) if is_daily else dt.time(0, 0)
        return dt.datetime.combine(d, t, tzinfo=IST)
    s_norm = s.replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(s_norm)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d-%m-%Y %H:%M:%S"):
            try:
                parsed = dt.datetime.strptime(s, fmt)
                break
            except ValueError:
                continue
        else:
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=IST)
    return parsed.astimezone(IST)


def normalize_kotak_history_rows(resp: Any, interval: str) -> list[dict[str, Any]]:
    """Normalize both live-verified Kotak historical_data shapes (bare list and dict wrapper)."""
    if isinstance(resp, list):
        raw_rows = resp
    elif isinstance(resp, dict):
        data = resp.get("data")
        if isinstance(data, list):
            raw_rows = data
        elif isinstance(data, dict):
            raw_rows = data.get("candles") or data.get("candle") or []
        else:
            raw_rows = resp.get("candles") or resp.get("candle") or []
    else:
        raw_rows = []

    is_daily = interval in ("D", "W")
    out: list[dict[str, Any]] = []
    for c in raw_rows:
        if isinstance(c, (list, tuple)) and len(c) >= 5:
            t_raw, o_raw, h_raw, l_raw, cl_raw = c[0], c[1], c[2], c[3], c[4]
            v_raw = c[5] if len(c) >= 6 else 0.0
            close_raw = None
        elif isinstance(c, dict):
            t_raw = (
                c.get("open_time_ist")
                or c.get("open_time")
                or c.get("datetime")
                or c.get("time")
                or c.get("timestamp")
                or c.get("date")
                or c.get("dt")
            )
            close_raw = c.get("close_time_ist") or c.get("close_time")
            o_raw = c.get("open") if "open" in c else c.get("o")
            h_raw = c.get("high") if "high" in c else c.get("h")
            l_raw = c.get("low") if "low" in c else c.get("l")
            cl_raw = c.get("close") if "close" in c else c.get("c")
            v_raw = c.get("volume") if "volume" in c else c.get("vol", 0.0)
        else:
            continue

        open_dt = _parse_kotak_candle_timestamp(t_raw, is_daily=is_daily)
        if open_dt is None:
            continue
        if is_daily and open_dt.timetz().replace(tzinfo=None) == dt.time(0, 0):
            open_dt = open_dt.replace(hour=9, minute=15, second=0, microsecond=0)

        close_dt = _parse_kotak_candle_timestamp(close_raw, is_daily=False) if close_raw else None
        if close_dt is None:
            if interval == "D":
                close_dt = open_dt.replace(hour=15, minute=30, second=0, microsecond=0)
            elif interval == "W":
                close_dt = (open_dt + dt.timedelta(days=4)).replace(
                    hour=15, minute=30, second=0, microsecond=0
                )
            elif interval == "60min":
                session_end = open_dt.replace(hour=15, minute=30, second=0, microsecond=0)
                close_dt = min(open_dt + dt.timedelta(minutes=60), session_end)
            elif interval == "15min":
                close_dt = open_dt + dt.timedelta(minutes=15)
            else:
                mins = int(interval.replace("min", "")) if interval.endswith("min") else 60
                close_dt = open_dt + dt.timedelta(minutes=mins)

        pair = (open_dt.timetz().replace(tzinfo=None), close_dt.timetz().replace(tzinfo=None))
        if interval == "60min" and pair not in VALID_1H_OPEN_CLOSE_PAIRS:
            # Skip trailing 15:15-15:30 stub bar emitted inside some 60min feeds; handled by 15min.
            continue
        if interval == "15min" and pair != FINAL_15M_OPEN_CLOSE_PAIR:
            continue

        out.append(
            {
                "open_time_ist": open_dt.isoformat(),
                "close_time_ist": close_dt.isoformat(),
                "open": float(o_raw or 0.0),
                "high": float(h_raw or 0.0),
                "low": float(l_raw or 0.0),
                "close": float(cl_raw or 0.0),
                "volume": float(v_raw or 0.0),
            }
        )
    return out


def slice_completed_candles(
    raw_candles: dict[str, Any],
    decision_dt_ist: dt.datetime,
) -> tuple[list[CandleBar], list[CandleBar], CandleBar | None, int]:
    """Filter 1D, 1H, and final 15M candles to strictly completed bars (close_time_ist <= decision_dt_ist)."""
    dec_ist = ensure_ist(decision_dt_ist)
    stripped_count = 0

    completed_1d: list[CandleBar] = []
    for item in raw_candles.get("candles_1d", []):
        open_dt = ensure_ist(dt.datetime.fromisoformat(item["open_time_ist"]))
        close_dt = ensure_ist(dt.datetime.fromisoformat(item["close_time_ist"]))
        if close_dt <= dec_ist:
            completed_1d.append(
                CandleBar(
                    timeframe="1D",
                    open_time_ist=open_dt.isoformat(),
                    close_time_ist=close_dt.isoformat(),
                    open=float(item["open"]),
                    high=float(item["high"]),
                    low=float(item["low"]),
                    close=float(item["close"]),
                    volume=float(item.get("volume", 0.0)),
                    is_final_15m_bar=False,
                )
            )
        else:
            stripped_count += 1

    completed_1h: list[CandleBar] = []
    for item in raw_candles.get("candles_1h", []):
        open_dt = ensure_ist(dt.datetime.fromisoformat(item["open_time_ist"]))
        close_dt = ensure_ist(dt.datetime.fromisoformat(item["close_time_ist"]))
        pair = (open_dt.timetz().replace(tzinfo=None), close_dt.timetz().replace(tzinfo=None))
        if pair not in VALID_1H_OPEN_CLOSE_PAIRS:
            raise KotakAdapterError(
                f"Invalid 1H candle boundary {pair}; expected NSE 09:15-15:15 hourly slots."
            )
        if close_dt <= dec_ist:
            completed_1h.append(
                CandleBar(
                    timeframe="1H",
                    open_time_ist=open_dt.isoformat(),
                    close_time_ist=close_dt.isoformat(),
                    open=float(item["open"]),
                    high=float(item["high"]),
                    low=float(item["low"]),
                    close=float(item["close"]),
                    volume=float(item.get("volume", 0.0)),
                    is_final_15m_bar=False,
                )
            )
        else:
            stripped_count += 1

    completed_final_15m: CandleBar | None = None
    for item in raw_candles.get("candles_15m_closing", []):
        open_dt = ensure_ist(dt.datetime.fromisoformat(item["open_time_ist"]))
        close_dt = ensure_ist(dt.datetime.fromisoformat(item["close_time_ist"]))
        pair = (open_dt.timetz().replace(tzinfo=None), close_dt.timetz().replace(tzinfo=None))
        if pair != FINAL_15M_OPEN_CLOSE_PAIR:
            raise KotakAdapterError(
                f"Invalid final 15m candle boundary {pair}; expected (15:15, 15:30) IST."
            )
        if close_dt <= dec_ist:
            completed_final_15m = CandleBar(
                timeframe="15M",
                open_time_ist=open_dt.isoformat(),
                close_time_ist=close_dt.isoformat(),
                open=float(item["open"]),
                high=float(item["high"]),
                low=float(item["low"]),
                close=float(item["close"]),
                volume=float(item.get("volume", 0.0)),
                is_final_15m_bar=True,
            )
        else:
            stripped_count += 1

    return completed_1d, completed_1h, completed_final_15m, stripped_count


def _select_target_depth_neosymbols(
    strikes_raw: Sequence[dict[str, Any]],
    reference_price: float,
    max_tokens: int = 12,
    extra_neo_symbols: Sequence[str] = (),
) -> list[str]:
    """Select targeted `neo_symbol`s for live `client.quotes` depth pulls (Stage A funnel).

    Priority order (deduplicated, capped at `max_tokens` so 2-token batches stay <= 35s cap):
    1. Held-position contracts (`extra_neo_symbols`) present on this chain.
    2. 3 strikes nearest `reference_price` (`CE` + `PE`, 6 tokens) for put-call parity,
       synthetic forward `F`, ATM IV anchor, debit long legs, and Iron Butterfly center.
    3. Primary directional & wing strikes (`~0.30d PE` & `-150`, `~0.30d CE` & `+150`,
       `+200 CE` debit wing, `ATM-150 PE`, `ATM+150 CE`, plus `4th/5th` near-ATM pairs and
       `200`-pt / `0.16d` wings when `max_tokens` allows).
    """
    if not strikes_raw:
        return []

    selected: list[str] = []
    seen: set[str] = set()

    chain_neos: set[str] = set()
    by_strike: dict[float, dict[str, Any]] = {}
    for row in strikes_raw:
        k = float(row.get("strike_price") or 0.0)
        if k <= 0.0:
            continue
        by_strike[k] = row
        for side in ("CE", "PE"):
            ns = (row.get(side) or {}).get("neo_symbol")
            if ns and "|" in str(ns):
                chain_neos.add(str(ns).strip())

    # Required monitored open-position contracts belonging to this chain are ALWAYS included
    # even when away from ATM, without starving the 3 near-ATM parity pairs (6 tokens).
    required_on_chain: list[str] = []
    req_seen: set[str] = set()
    for ns in extra_neo_symbols:
        s_req = str(ns or "").strip()
        if s_req in chain_neos and s_req not in req_seen:
            req_seen.add(s_req)
            required_on_chain.append(s_req)

    effective_cap = (
        max(int(max_tokens), len(required_on_chain) + 6)
        if required_on_chain
        else int(max_tokens)
    )

    def _add(ns: Any) -> None:
        s = str(ns or "").strip()
        if s and "|" in s and s not in seen and len(selected) < effective_cap:
            seen.add(s)
            selected.append(s)

    # 1. Held open-position tokens belonging to this chain
    for ns in required_on_chain:
        _add(ns)

    if not by_strike:
        return selected

    strikes_sorted = sorted(by_strike.keys())
    ref = reference_price if reference_price > 0.0 else strikes_sorted[len(strikes_sorted) // 2]
    nearest_atm = sorted(strikes_sorted, key=lambda k: abs(k - ref))
    atm_k = nearest_atm[0]

    # 2. Core 3 near-ATM strikes (CE + PE = 6 tokens)
    for k in nearest_atm[:3]:
        row = by_strike[k]
        _add((row.get("CE") or {}).get("neo_symbol"))
        _add((row.get("PE") or {}).get("neo_symbol"))

    def _nearest_strike(target_k: float) -> float:
        return min(strikes_sorted, key=lambda k: abs(k - target_k))

    def _strike_for_indicative_delta(side: str, target_abs_d: float, fallback_offset: float) -> float:
        best_k: float | None = None
        best_diff = 999.0
        for k in strikes_sorted:
            leg = by_strike[k].get(side) or {}
            g = leg.get("greeks") if isinstance(leg.get("greeks"), dict) else {}
            d_val = g.get("delta")
            if d_val is not None:
                try:
                    diff = abs(abs(float(d_val)) - target_abs_d)
                    if diff < best_diff:
                        best_diff = diff
                        best_k = k
                except (TypeError, ValueError):
                    pass
        return best_k if best_k is not None else _nearest_strike(ref + fallback_offset)

    # 3. Directional & neutral wing pairs (side-specific so we never waste tokens on deep ITM opposite legs)
    k_sp30 = _strike_for_indicative_delta("PE", 0.30, -200.0)
    k_sc30 = _strike_for_indicative_delta("CE", 0.30, +200.0)
    k_lc45 = _strike_for_indicative_delta("CE", 0.45, +50.0)
    k_lp45 = _strike_for_indicative_delta("PE", 0.45, -50.0)

    priority_side_strikes: list[tuple[str, float]] = [
        ("PE", k_sp30),
        ("PE", _nearest_strike(k_sp30 - 150.0)),
        ("CE", k_sc30),
        ("CE", _nearest_strike(k_sc30 + 150.0)),
        ("CE", _nearest_strike(k_lc45 + 200.0)),
        ("PE", _nearest_strike(k_lp45 - 150.0)),
        ("PE", _nearest_strike(atm_k - 150.0)),
        ("CE", _nearest_strike(atm_k + 150.0)),
        ("PE", _nearest_strike(k_sp30 - 200.0)),
        ("CE", _nearest_strike(k_sc30 + 200.0)),
    ]
    for side, k in priority_side_strikes:
        leg = by_strike.get(k, {}).get(side) or {}
        _add(leg.get("neo_symbol"))

    # 4. Additional near-ATM pairs & IC ~0.16d wings if budget remains
    for k in nearest_atm[3:5]:
        row = by_strike[k]
        _add((row.get("CE") or {}).get("neo_symbol"))
        _add((row.get("PE") or {}).get("neo_symbol"))

    k_sp16 = _strike_for_indicative_delta("PE", 0.16, -350.0)
    k_sc16 = _strike_for_indicative_delta("CE", 0.16, +350.0)
    for side, k in [
        ("PE", k_sp16),
        ("PE", _nearest_strike(k_sp16 - 150.0)),
        ("CE", k_sc16),
        ("CE", _nearest_strike(k_sc16 + 150.0)),
    ]:
        leg = by_strike.get(k, {}).get(side) or {}
        _add(leg.get("neo_symbol"))

    return selected


class LiveKotakNeoAdapter(KotakNeoReadOnlyPort):
    """Live Kotak Neo SDK adapter with targeted `client.quotes` depth and chunked `validate_history`.

    - Authentication: reuses ~/Kotak_API kotak_option_chain.auth.AuthManager.
    - Rate limiting: `PacedReadTransport` (2.5s serial spacing, 35s queue cap, persisted 429 cooldown)
      or `ProvisionalSerialRateLimiter`. Any HTTP 429 stops the run immediately with zero retries.
    - Option chain & depth (`P1-1`, `P3.1`):
      Calls `get_full_option_chain(..., count=40)` without `include_depth=True`, shortlists target
      `neo_symbol`s, fetches `client.quotes(..., quote_type="all")` in verified small batches
      (default 2 tokens/call), normalizes via `interpret_depth_book`, converts raw quantities via
      live per-contract `mktLot`, and leaves missing/rejected/unfetched books as `None`.
    - Historical candles (`P1-2`):
      Calls `client.historical_data(neosymbol="nse_cm|Nifty 50", interval=...)` validated via
      `validate_history`, chunks `"D"` into `<= 179`-day windows, caches completed prior-day `"D"`
      bars per IST trading date, and parses both bare-list and dict-wrapper responses.
    """

    ENDPOINT_VERIFICATION_STATUS = "UNVERIFIED"

    def __init__(
        self,
        kotak_api_root: str | Path | None = None,
        rate_limiter: ProvisionalSerialRateLimiter | PacedReadTransport | None = None,
        enforce_schedule_in_live_mode: bool = True,
        depth_strikes_each_side: int = 40,
        quotes_batch_size: int = 2,
        max_depth_tokens_per_expiry: int = 12,
        extra_neo_symbols: Sequence[str] = (),
        cache_dir: str | Path | None = None,
        trading_calendar: dict[str, Any] | None = None,
    ) -> None:
        import sys as _sys

        self.kotak_api_root = Path(kotak_api_root or (Path.home() / "Kotak_API"))
        if str(self.kotak_api_root) not in _sys.path:
            _sys.path.insert(0, str(self.kotak_api_root))
        try:
            from kotak_option_chain.auth import AuthManager, ConfigStore
            from kotak_option_chain.option_chain import OptionChainService
        except ImportError as exc:
            raise KotakAdapterError(
                f"Kotak repo import failed at {self.kotak_api_root}: {exc}. "
                "The pinned ~/Kotak_API checkout is required for the live adapter."
            )
        self.cache_dir = Path(cache_dir) if cache_dir else None
        shared_state_dir = (
            self.cache_dir if self.cache_dir is not None else (Path.home() / "nifty-analyst")
        )
        cooldown_file = shared_state_dir / "kotak_429_cooldown.json"
        pacing_state_file = shared_state_dir / "kotak_pacing_state.json"
        self.rate_limiter = rate_limiter or PacedReadTransport(
            min_interval_seconds=2.5,
            max_queue_wait_seconds=35.0,
            cooldown_file_path=cooldown_file,
            pacing_state_file_path=pacing_state_file,
            strict_guard_seconds=0.0,
            early_wake_guard_seconds=0.001,
        )
        self._wired_auth_pairs: set[tuple[int, int]] = set()
        auth_kwargs: dict[str, Any] = {"store": ConfigStore()}
        try:
            auth_sig = inspect.signature(AuthManager.__init__)
            cf_param = auth_sig.parameters.get("client_factory")
            if cf_param is not None and callable(cf_param.default):
                auth_kwargs["client_factory"] = self._wrap_client_factory(cf_param.default)
        except (TypeError, ValueError):
            pass
        self._auth = AuthManager(**auth_kwargs)
        self._service = OptionChainService(auth_manager=self._auth)
        self.enforce_schedule = enforce_schedule_in_live_mode
        self.depth_strikes_each_side = depth_strikes_each_side
        self.quotes_batch_size = max(1, min(50, int(quotes_batch_size)))
        self.max_depth_tokens_per_expiry = max(4, int(max_depth_tokens_per_expiry))
        self.extra_neo_symbols = tuple(extra_neo_symbols)
        self.trading_calendar = trading_calendar or {}
        self.quote_ts_semantics_verified: bool = False
        self.require_contract_quote_timestamp: bool = False
        self.max_quote_staleness_seconds: float = 180.0
        self.max_future_skew_seconds: float = 5.0
        self.adapter_call_count: int = 0
        self.cache_hit_count: int = 0
        self._cache: dict[str, Any] = {}
        self._ensure_transport_wired()

    # ----- plumbing -----

    def _now_ist(self) -> dt.datetime:
        fn = getattr(self, "_wall_clock_ist_fn", None)
        if callable(fn):
            return ensure_ist(fn())
        return dt.datetime.now(tz=IST)

    def _wrap_client_factory(self, orig_factory: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap `AuthManager.client_factory` (`NeoAPI`) so EVERY fresh read-only client is constructed
        with `SharedPacedHttpTransport` bound to `self.rate_limiter` (leaving TOTP login untouched).
        """
        pair_tag = (id(self), id(getattr(self, "rate_limiter", None)))
        if getattr(orig_factory, "_nifty_analyst_factory_pair", None) == pair_tag:
            return orig_factory

        accepts_transport_kw = False
        try:
            for pname, pinfo in inspect.signature(orig_factory).parameters.items():
                if pname == "transport" and pinfo.kind in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                ):
                    accepts_transport_kw = True
                    break
        except (TypeError, ValueError):
            accepts_transport_kw = False

        def _paced_client_factory(*args: Any, **kwargs: Any) -> Any:
            lim = getattr(self, "rate_limiter", None)
            if _IN_AUTH_LOGIN_FLOW.get() or not isinstance(lim, PacedReadTransport):
                return orig_factory(*args, **kwargs)
            call_kwargs = dict(kwargs)
            if accepts_transport_kw and call_kwargs.get("transport") is not None:
                paced_tr = build_paced_client_transport(
                    lim,
                    call_kwargs.get("transport"),
                    http2=bool(call_kwargs.get("http2", True)),
                    limits=call_kwargs.get("limits"),
                    create_default_if_none=False,
                )
                if paced_tr is not None:
                    call_kwargs["transport"] = paced_tr
            client = orig_factory(*args, **call_kwargs)
            attach_paced_transport_to_client(client, lim)
            return client

        setattr(_paced_client_factory, "_nifty_analyst_factory_pair", pair_tag)
        return _paced_client_factory

    def _wire_auth_object(self, auth_obj: Any) -> None:
        """Idempotently wire `client_factory`, `require_authenticated_client`, and `login_with_daily_totp`."""
        lim = getattr(self, "rate_limiter", None)
        if auth_obj is None or not isinstance(lim, PacedReadTransport):
            return
        wired_pairs = getattr(self, "_wired_auth_pairs", None)
        if wired_pairs is None:
            wired_pairs = set()
            self._wired_auth_pairs = wired_pairs
        pair_key = (id(auth_obj), id(lim))
        if pair_key in wired_pairs:
            return
        wired_pairs.add(pair_key)

        # Preserve daily TOTP login (`POST` totp_login / totp_validate) completely untouched
        login_fn = getattr(auth_obj, "login_with_daily_totp", None)
        if callable(login_fn) and not getattr(login_fn, "_nifty_analyst_login_wrapped", False):

            def _login_guard(*args: Any, _orig_login: Callable[..., Any] = login_fn, **kwargs: Any) -> Any:
                tok = _IN_AUTH_LOGIN_FLOW.set(True)
                try:
                    return _orig_login(*args, **kwargs)
                finally:
                    _IN_AUTH_LOGIN_FLOW.reset(tok)

            setattr(_login_guard, "_nifty_analyst_login_wrapped", True)
            try:
                setattr(auth_obj, "login_with_daily_totp", _login_guard)
            except Exception:
                pass

        # 1. Wire `client_factory` at client construction boundary
        cur_factory = getattr(auth_obj, "client_factory", None)
        if callable(cur_factory):
            try:
                setattr(auth_obj, "client_factory", self._wrap_client_factory(cur_factory))
                lim.http_boundary_attached = True
            except Exception:
                pass

        # 2. Wire client-returning helper methods so even custom fresh-client helpers without
        #    `client_factory` (or helpers that wrap `session._transport` after `client_factory`)
        #    have `SharedPacedHttpTransport` attached on EVERY returned client.
        for method_name in (
            "require_authenticated_client",
            "require_read_only_client",
            "create_read_only_client",
            "get_authenticated_client",
            "create_authenticated_client",
            "build_authenticated_client",
            "get_client",
            "create_client",
            "build_client",
            "_create_client",
        ):
            orig_m = getattr(auth_obj, method_name, None)
            if not callable(orig_m):
                continue
            if getattr(orig_m, "_nifty_analyst_method_pair", None) == pair_key:
                continue

            def _client_helper_wrapper(
                *args: Any,
                _orig: Callable[..., Any] = orig_m,
                **kwargs: Any,
            ) -> Any:
                res = _orig(*args, **kwargs)
                cur_lim = getattr(self, "rate_limiter", None)
                if not _IN_AUTH_LOGIN_FLOW.get() and isinstance(cur_lim, PacedReadTransport):
                    cli = res[0] if isinstance(res, tuple) and res else res
                    attach_paced_transport_to_client(cli, cur_lim)
                return res

            setattr(_client_helper_wrapper, "_nifty_analyst_method_pair", pair_key)
            try:
                setattr(auth_obj, method_name, _client_helper_wrapper)
            except Exception:
                pass

    def _ensure_transport_wired(self) -> None:
        self._wire_auth_object(getattr(self, "_auth", None))
        svc = getattr(self, "_service", None)
        if svc is not None:
            for svc_auth_attr in ("auth", "_auth", "auth_manager", "_auth_manager"):
                self._wire_auth_object(getattr(svc, svc_auth_attr, None))

    def _acquire_authenticated_client(self) -> Any:
        self._ensure_transport_wired()
        res = self._auth.require_authenticated_client()
        cli = res[0] if isinstance(res, tuple) and res else res
        lim = getattr(self, "rate_limiter", None)
        if isinstance(lim, PacedReadTransport):
            attach_paced_transport_to_client(cli, lim)
        return cli

    @property
    def http_request_count(self) -> int:
        lim = getattr(self, "rate_limiter", None)
        return int(getattr(lim, "http_request_count", 0) or 0)

    def _assert_schedule_approved(self, decision_dt_ist: dt.datetime) -> dt.datetime:
        dec_ist = ensure_ist(decision_dt_ist)
        if self.enforce_schedule and not is_approved_schedule_slot(
            dec_ist,
            holiday_dates=self.trading_calendar.get("holiday_dates"),
            special_sessions=self.trading_calendar.get("special_sessions"),
        ):
            raise UnapprovedScheduleSlotError(
                f"Live Kotak read rejected at {dec_ist.isoformat()}: outside the 5 approved "
                f"IST run slots (10:15, 11:15, 12:15, 13:15, 14:15 IST) or a verified "
                f"holiday / non-session date."
            )
        return dec_ist

    def _raise_mapped_transport_error(self, exc: BaseException) -> NoReturn:
        if isinstance(exc, KotakAdapterError):
            raise exc
        if isinstance(exc, CooldownPersistence429Error):
            raise KotakCooldownPersistence429Error(str(exc)) from exc
        if isinstance(exc, SharedSafetyStateError):
            raise KotakSharedStateError(str(exc)) from exc
        if isinstance(exc, (HttpRateLimit429Error, PacedCooldownActiveError)):
            raise KotakRateLimit429Error(str(exc)) from exc
        if isinstance(exc, PacedQueueCapExceededError):
            raise KotakAdapterError(str(exc)) from exc
        raise KotakAdapterError(str(exc)) from exc

    def _guard_request(self, cache_key: str, decision_dt_ist: dt.datetime) -> bool:
        self._assert_schedule_approved(decision_dt_ist)
        if cache_key in self._cache:
            return False
        try:
            if getattr(self.rate_limiter, "http_boundary_attached", False):
                chk = getattr(self.rate_limiter, "check_persisted_429_cooldown", None)
                if callable(chk):
                    chk()
                return False
            self.rate_limiter.wait_for_slot()
            return True
        except (
            CooldownPersistence429Error,
            SharedSafetyStateError,
            HttpRateLimit429Error,
            PacedCooldownActiveError,
            PacedQueueCapExceededError,
        ) as exc:
            self._raise_mapped_transport_error(exc)

    def _record_429_if_supported(self, reason: str) -> None:
        rec_fn = getattr(self.rate_limiter, "record_429_cooldown", None)
        if callable(rec_fn):
            try:
                rec_fn(reason)
            except SharedSafetyStateError as state_exc:
                raise KotakCooldownPersistence429Error(
                    f"HTTP 429 from Kotak Neo API ({reason[:160]}) AND failed to persist "
                    f"shared cooldown state: {state_exc}"
                ) from state_exc

    def _call(self, cache_key: str, decision_dt_ist: dt.datetime, fn: Callable[[], Any]) -> Any:
        self.adapter_call_count = int(getattr(self, "adapter_call_count", 0)) + 1
        self._ensure_transport_wired()

        acquired_pre_slot = self._guard_request(cache_key, decision_dt_ist)
        if cache_key in self._cache:
            self.cache_hit_count = int(getattr(self, "cache_hit_count", 0)) + 1
            return self._cache[cache_key]

        lim = getattr(self, "rate_limiter", None)
        if hasattr(lim, "_call_aborted_error"):
            lim._call_aborted_error = None
        if acquired_pre_slot and hasattr(lim, "_pending_pre_guard_slot"):
            lim._pending_pre_guard_slot = True

        try:
            resp = fn()
            swallowed_abort = getattr(lim, "_call_aborted_error", None)
            if swallowed_abort is not None:
                lim._call_aborted_error = None
                self._raise_mapped_transport_error(swallowed_abort)
        except KotakAdapterError:
            swallowed_abort = getattr(lim, "_call_aborted_error", None)
            if swallowed_abort is not None:
                lim._call_aborted_error = None
                if isinstance(
                    swallowed_abort,
                    (CooldownPersistence429Error, HttpRateLimit429Error, SharedSafetyStateError),
                ):
                    self._raise_mapped_transport_error(swallowed_abort)
            raise
        except (
            CooldownPersistence429Error,
            SharedSafetyStateError,
            HttpRateLimit429Error,
            PacedCooldownActiveError,
            PacedQueueCapExceededError,
        ) as exc:
            if hasattr(lim, "_call_aborted_error"):
                lim._call_aborted_error = None
            self._raise_mapped_transport_error(exc)
        except Exception as exc:
            swallowed_abort = getattr(lim, "_call_aborted_error", None)
            if hasattr(lim, "_call_aborted_error"):
                lim._call_aborted_error = None
            if isinstance(
                swallowed_abort,
                (CooldownPersistence429Error, HttpRateLimit429Error, SharedSafetyStateError),
            ):
                self._raise_mapped_transport_error(swallowed_abort)
            msg = str(exc)
            if "429" in msg or "too many" in msg.lower():
                self._record_429_if_supported(msg)
                raise KotakRateLimit429Error(
                    f"HTTP 429 from Kotak Neo API ({msg[:200]}). Hard stop; zero retries."
                ) from exc
            raise KotakAdapterError(f"Kotak SDK call failed ({cache_key}): {msg}") from exc
        finally:
            if hasattr(lim, "_pending_pre_guard_slot"):
                lim._pending_pre_guard_slot = False
            if hasattr(lim, "_call_aborted_error"):
                lim._call_aborted_error = None

        if isinstance(resp, dict):
            status = str(resp.get("status") or resp.get("stCode") or "")
            if "429" in status:
                self._record_429_if_supported(status)
                raise KotakRateLimit429Error("HTTP 429 from Kotak Neo API. Hard stop; zero retries.")
            if str(resp.get("status")) == "UNAUTHORIZED" or "session" in str(resp.get("error", "")).lower():
                raise KotakAdapterError(
                    f"Kotak session rejected by server ({status or resp.get('error')}); "
                    "re-authentication via the daily TOTP login flow is required."
                )
        self._cache[cache_key] = resp
        return resp

    # ----- port methods -----

    def fetch_contract_master_raw(self, decision_dt_ist: dt.datetime) -> dict[str, Any]:
        dec_ist = ensure_ist(decision_dt_ist)
        exp_resp = self._call(
            f"expiries:{dec_ist.date().isoformat()}",
            dec_ist,
            lambda: self._service.get_expiries("NIFTY", exchange="nse_fo", instrument_type="option"),
        )
        if exp_resp.get("status") != "OK":
            raise KotakAdapterError(f"Kotak expiries fetch failed: {exp_resp}")
        expiries = list(exp_resp.get("expiries") or [])

        fut_resp = self._call(
            f"futures_chain:{dec_ist.isoformat()}",
            dec_ist,
            lambda: self._service.get_futures_chain("NIFTY", exchange="nse_fo"),
        )
        if fut_resp.get("status") != "OK":
            raise KotakAdapterError(f"Kotak futures metadata fetch failed: {fut_resp}")
        lot_size = int(fut_resp.get("lot_size") or 0)
        if lot_size <= 0:
            raise KotakAdapterError("Kotak futures metadata returned no usable lot size.")

        raw_fut_ltp = fut_resp.get("nearest_futures_ltp")
        spot_val: float | None = None
        spot_source = "kotak_neo_nearest_futures_ltp"
        if raw_fut_ltp is not None:
            try:
                cand_fut = float(raw_fut_ltp)
                if cand_fut > 0.0:
                    spot_val = cand_fut
            except (TypeError, ValueError):
                spot_val = None

        if spot_val is None:
            fetch_spot_fn = getattr(self._service, "_fetch_spot_quote", None)
            if callable(fetch_spot_fn):
                spot_payload = self._call(
                    f"spot_index:{dec_ist.isoformat()}",
                    dec_ist,
                    lambda: fetch_spot_fn(self._acquire_authenticated_client(), "NIFTY"),
                )
            else:
                spot_client = self._acquire_authenticated_client()
                if hasattr(spot_client, "quotes"):
                    spot_batch = validate_quotes_batch(
                        [{"exchange_segment": "nse_cm", "instrument_token": "Nifty 50"}]
                    )
                    spot_payload = self._call(
                        f"spot_index:{dec_ist.isoformat()}",
                        dec_ist,
                        lambda: spot_client.quotes(instrument_tokens=spot_batch, quote_type="all"),
                    )
                else:
                    spot_payload = None

            cand_spot: float | None = None
            if isinstance(spot_payload, dict) and spot_payload.get("ltp") is not None:
                try:
                    cand_spot = float(spot_payload["ltp"])
                except (TypeError, ValueError):
                    cand_spot = None
            elif isinstance(spot_payload, list) and spot_payload and isinstance(spot_payload[0], dict):
                try:
                    cand_spot = float(spot_payload[0].get("ltp") or 0.0)
                except (TypeError, ValueError):
                    cand_spot = None

            if cand_spot is not None and cand_spot > 0.0:
                spot_val = cand_spot
                spot_source = "kotak_neo_index_spot_nse_cm"
            else:
                raise KotakAdapterError(
                    "No usable underlying reference from Kotak: both nearest futures LTP and "
                    "nse_cm|Nifty 50 index spot LTP are unavailable."
                )

        return {
            "symbol": "NIFTY",
            "spot": {
                "value": float(spot_val),
                "as_of_ist": dec_ist.isoformat(),
                "source_id": spot_source,
                "verification_tag": "UNVERIFIED_KOTAK_SCHEMA",
            },
            "expiries": [
                {
                    "expiry_date": str(e),
                    # Initial lot_size from futures common_data; overridden per expiry by live
                    # option chain `mktLot` in `fetch_option_chain_raw` -> `ChainSnapshotEnvelope.lot_size`.
                    "lot_size": lot_size,
                    "is_holiday_adjusted": False,
                    "holiday_note": "",
                    "is_monthly": False,
                }
                for e in expiries
            ],
        }

    def _fetch_targeted_quotes_by_neosymbol(
        self,
        neo_symbols: Sequence[str],
        expiry_date: str,
        dec_ist: dt.datetime,
        *,
        initial_client: Any = None,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
        """Fetch `client.quotes(..., quote_type='all')` in verified small batches (default 2 tokens).

        Standing API rule: HTTP 429, shared-state errors, or session errors hard-stop immediately
        with zero retries; a transient non-429 quote failure gets at most ONE retry after
        `quote_retry_delay_seconds` (default 30.0s) and is then flagged in `failed_errors`.
        """
        if not neo_symbols:
            return {}, {}
        first_cli = initial_client if initial_client is not None else self._acquire_authenticated_client()
        if not hasattr(first_cli, "quotes"):
            return {}, {}
        seeded_clients: list[Any] = [first_cli]

        def _next_quotes_client() -> Any:
            if seeded_clients:
                return seeded_clients.pop(0)
            return self._acquire_authenticated_client()

        tokens: list[tuple[str, dict[str, str]]] = []
        for ns in neo_symbols:
            if not ns or "|" not in ns:
                continue
            seg, tok = ns.split("|", 1)
            seg_s, tok_s = seg.strip(), tok.strip()
            if seg_s and tok_s:
                tokens.append((f"{seg_s}|{tok_s}", {"exchange_segment": seg_s, "instrument_token": tok_s}))

        batch_sz = max(1, min(50, int(getattr(self, "quotes_batch_size", 2))))
        retry_delay = float(getattr(self, "quote_retry_delay_seconds", 30.0))
        sleep_fn = getattr(self, "_quote_retry_sleep_fn", time.sleep)
        quotes_by_key: dict[str, dict[str, Any]] = {}
        failed_errors: dict[str, str] = {}

        for i in range(0, len(tokens), batch_sz):
            chunk = tokens[i : i + batch_sz]
            validated_batch = validate_quotes_batch([t[1] for t in chunk])
            batch_id = ",".join(t[0] for t in chunk)
            try:
                resp = self._call(
                    f"quotes:{expiry_date}:{dec_ist.isoformat()}:{batch_id}",
                    dec_ist,
                    lambda b=validated_batch: _next_quotes_client().quotes(
                        instrument_tokens=b, quote_type="all"
                    ),
                )
            except (KotakRateLimit429Error, KotakSharedStateError, UnapprovedScheduleSlotError):
                raise
            except KotakAdapterError as first_exc:
                if "re-authentication" in str(first_exc) or "queue cap" in str(first_exc).lower():
                    raise
                if retry_delay > 0.0:
                    sleep_fn(retry_delay)
                try:
                    resp = self._call(
                        f"quotes:{expiry_date}:{dec_ist.isoformat()}:{batch_id}:retry1",
                        dec_ist,
                        lambda b=validated_batch: _next_quotes_client().quotes(
                            instrument_tokens=b, quote_type="all"
                        ),
                    )
                except (KotakRateLimit429Error, KotakSharedStateError, UnapprovedScheduleSlotError):
                    raise
                except KotakAdapterError as retry_exc:
                    if "re-authentication" in str(retry_exc) or "queue cap" in str(retry_exc).lower():
                        raise
                    for ns_key, _ in chunk:
                        failed_errors[ns_key] = f"QUOTE_FETCH_FAILED_AFTER_RETRY:{retry_exc}"
                    continue

            items: list[Any]
            if isinstance(resp, list):
                items = resp
            elif isinstance(resp, dict) and isinstance(resp.get("data"), list):
                items = resp["data"]
            else:
                items = []

            for q in items:
                if not isinstance(q, dict):
                    continue
                exch = str(q.get("exchange") or q.get("exchange_segment") or "").strip()
                etok = str(
                    q.get("exchange_token") or q.get("instrument_token") or q.get("tk") or ""
                ).strip()
                if exch and etok:
                    quotes_by_key[f"{exch}|{etok}"] = q
                if etok:
                    quotes_by_key.setdefault(etok, q)

        return quotes_by_key, failed_errors

    def fetch_option_chain_raw(
        self,
        expiry_date: str,
        decision_dt_ist: dt.datetime,
        fetch_depth: bool = True,
        max_depth_tokens_override: int | None = None,
    ) -> dict[str, Any]:
        dec_ist = self._assert_schedule_approved(decision_dt_ist)
        fetch_started_dt = self._now_ist()
        strike_count = int(getattr(self, "depth_strikes_each_side", 40))
        # Never pass include_depth=True: OptionChainService._fetch_depth_by_neosymbol uses
        # 25-token batches with 0.04s sleep and swallows 400/429 exceptions.
        resp = self._call(
            f"chain:{expiry_date}:{dec_ist.isoformat()}",
            dec_ist,
            lambda: self._service.get_full_option_chain(
                "NIFTY", exchange="nse_fo", expiry=expiry_date, count=strike_count
            ),
        )
        if resp.get("status") != "OK":
            raise KotakAdapterError(f"Kotak option chain fetch failed for {expiry_date}: {resp}")
        strikes_raw = resp.get("strikes") or resp.get("chain") or []

        raw_lot = resp.get("lot_size")
        expiry_lot_size = int(raw_lot) if raw_lot is not None else 0
        if expiry_lot_size <= 0:
            # Check if futures metadata already cached a lot size in this run
            fut_cached = self._cache.get(f"futures_chain:{dec_ist.isoformat()}") or {}
            expiry_lot_size = int(fut_cached.get("lot_size") or 0)

        underlying_info = resp.get("underlying_prices") if isinstance(resp.get("underlying_prices"), dict) else {}
        base_ref = float(
            underlying_info.get("reference_price")
            or underlying_info.get("spot_price")
            or underlying_info.get("futures_price")
            or 0.0
        )
        fut_ref = float(underlying_info.get("futures_price") or 0.0)
        if fut_ref <= 0.0 and base_ref > 0.0:
            try:
                dte_d = max(0, (dt.date.fromisoformat(expiry_date) - dec_ist.date()).days)
                fut_ref = base_ref * math.exp(0.0675 * (dte_d / 365.0))
            except ValueError:
                fut_ref = base_ref
        # Shift shortlist center to forward when forward carry exceeds 1 strike interval (> 50 pts)
        ref_price = fut_ref if (fut_ref > 0.0 and abs(fut_ref - base_ref) > 50.0) else base_ref

        max_tokens = (
            int(max_depth_tokens_override)
            if max_depth_tokens_override is not None
            else int(getattr(self, "max_depth_tokens_per_expiry", 12))
        )
        extra_neos = tuple(getattr(self, "extra_neo_symbols", ()))
        shortlisted_neos = (
            _select_target_depth_neosymbols(
                strikes_raw=strikes_raw,
                reference_price=ref_price,
                max_tokens=max_tokens,
                extra_neo_symbols=extra_neos,
            )
            if fetch_depth
            else []
        )

        initial_quotes_cli = self._acquire_authenticated_client() if shortlisted_neos else None
        client_supports_quotes = bool(
            shortlisted_neos and initial_quotes_cli is not None and hasattr(initial_quotes_cli, "quotes")
        )
        requested_neos: list[str] = list(shortlisted_neos) if client_supports_quotes else []
        requested_set = set(requested_neos)

        quotes_map, failed_quote_errors = (
            self._fetch_targeted_quotes_by_neosymbol(
                requested_neos, expiry_date, dec_ist, initial_client=initial_quotes_cli
            )
            if requested_neos
            else ({}, {})
        )
        fetch_completed_dt = self._now_ist()
        if fetch_completed_dt < fetch_started_dt:
            fetch_completed_dt = fetch_started_dt

        if quotes_map and expiry_lot_size <= 0:
            # Check if any quote payload itself carries mktLot
            for q_item in quotes_map.values():
                cand_lot = q_item.get("mktLot") or q_item.get("lot_size") or q_item.get("lotSize")
                if cand_lot is not None and int(cand_lot) > 0:
                    expiry_lot_size = int(cand_lot)
                    break
            if expiry_lot_size <= 0:
                raise KotakAdapterError(
                    f"Option chain for {expiry_date} is missing live mktLot / lot_size; "
                    "cannot convert raw depth quantities to lots."
                )

        ts_semantics_verified = bool(getattr(self, "quote_ts_semantics_verified", False))
        require_contract_ts = (
            bool(getattr(self, "require_contract_quote_timestamp", False)) or ts_semantics_verified
        )
        max_stale_sec = float(getattr(self, "max_quote_staleness_seconds", 180.0))
        max_future_sec = float(getattr(self, "max_future_skew_seconds", 5.0))

        resolved_contract_lots: set[int] = set()
        rejected_depth_books: list[dict[str, Any]] = []
        priced_neo_symbols: list[str] = []
        returned_neo_symbols: list[str] = []
        missing_neo_symbols: list[str] = []
        rejected_neo_symbols: list[str] = []
        all_chain_neo_symbols: list[str] = []
        strikes: list[dict[str, Any]] = []

        _EMPTY_EXT: dict[str, Any] = {
            "touch_bid_qty": None,
            "touch_ask_qty": None,
            "visible_buy_qty": None,
            "visible_sell_qty": None,
            "buy_levels": [],
            "sell_levels": [],
            "quote_response_ts_candidates": {},
            "quote_as_of_ist": None,
            "ts_semantics": "",
            "ts_provenance": {},
        }

        def _extract_quote_ts_candidates(q_obj: dict[str, Any] | None) -> dict[str, str]:
            if not isinstance(q_obj, dict):
                return {}
            found_ts: dict[str, str] = {}
            for k_ts in (
                "quote_as_of_ist",
                "book_update_time_ist",
                "exchange_timestamp_ist",
                "exchange_timestamp",
                "last_traded_time",
                "lstup_time",
                "lut",
                "fdtm",
                "quote_updated_at",
                "updated_at",
                "timestamp",
            ):
                val_ts = q_obj.get(k_ts)
                if val_ts is not None and str(val_ts).strip():
                    found_ts[k_ts] = str(val_ts).strip()
            return found_ts

        def _resolve_leg_book(
            leg: dict[str, Any], strike_val: float, opt_type: str
        ) -> tuple[float | None, float | None, int, int | None, int | None, bool, dict[str, Any]]:
            ns = str(leg.get("neo_symbol") or "").strip()
            if ns and "|" in ns:
                all_chain_neo_symbols.append(ns)
            tok_only = ns.split("|", 1)[1].strip() if "|" in ns else ""

            if ns and ns in requested_set:
                if ns in failed_quote_errors:
                    missing_neo_symbols.append(ns)
                    rejected_depth_books.append(
                        {
                            "expiry_date": expiry_date,
                            "strike": strike_val,
                            "option_type": opt_type,
                            "neo_symbol": ns,
                            "rejection_reason": failed_quote_errors[ns],
                            "raw_payload": None,
                        }
                    )
                    return None, None, 0, None, None, bool(leg.get("ltp") is not None), dict(_EMPTY_EXT)

                q_payload = quotes_map.get(ns)
                if q_payload is None and tok_only:
                    q_payload = quotes_map.get(tok_only)

                if q_payload is None:
                    missing_neo_symbols.append(ns)
                    rejected_depth_books.append(
                        {
                            "expiry_date": expiry_date,
                            "strike": strike_val,
                            "option_type": opt_type,
                            "neo_symbol": ns,
                            "rejection_reason": "MISSING_FROM_QUOTES_RESPONSE",
                            "raw_payload": None,
                        }
                    )
                    return None, None, 0, None, None, bool(leg.get("ltp") is not None), dict(_EMPTY_EXT)

                returned_neo_symbols.append(ns)
                q_ts_cands = _extract_quote_ts_candidates(q_payload)
                ts_eval = normalize_contract_quote_timestamp(
                    q_payload,
                    dec_ist,
                    max_staleness_seconds=max_stale_sec,
                    max_future_skew_seconds=max_future_sec,
                    verified_semantics=ts_semantics_verified,
                    require_present=require_contract_ts,
                )
                ts_prov = {
                    "raw_value": ts_eval.raw_value,
                    "source_field": ts_eval.source_field,
                    "detected_unit": ts_eval.detected_unit,
                    "semantics_status": ts_eval.semantics_status,
                    "normalized_quote_as_of_ist": ts_eval.normalized_quote_as_of_ist,
                    "rejection_reason": ts_eval.rejection_reason,
                }
                parsed_book = interpret_depth_book(q_payload, fallback_lot_size=expiry_lot_size)
                if not parsed_book.is_structurally_valid:
                    rejected_neo_symbols.append(ns)
                    rejected_depth_books.append(
                        {
                            "expiry_date": expiry_date,
                            "strike": strike_val,
                            "option_type": opt_type,
                            "neo_symbol": ns,
                            "rejection_reason": parsed_book.rejection_reason,
                            "raw_payload": parsed_book.raw_payload,
                            "ts_provenance": ts_prov,
                        }
                    )
                    ext_rej = dict(_EMPTY_EXT)
                    ext_rej["quote_response_ts_candidates"] = q_ts_cands
                    ext_rej["ts_provenance"] = ts_prov
                    return None, None, 0, None, None, bool(leg.get("ltp") is not None), ext_rej

                if not ts_eval.is_valid:
                    rejected_neo_symbols.append(ns)
                    rejected_depth_books.append(
                        {
                            "expiry_date": expiry_date,
                            "strike": strike_val,
                            "option_type": opt_type,
                            "neo_symbol": ns,
                            "rejection_reason": ts_eval.rejection_reason,
                            "raw_payload": dict(q_payload),
                            "ts_provenance": ts_prov,
                        }
                    )
                    ext_ts_rej = dict(_EMPTY_EXT)
                    ext_ts_rej["quote_response_ts_candidates"] = q_ts_cands
                    ext_ts_rej["ts_provenance"] = ts_prov
                    return None, None, 0, None, None, bool(leg.get("ltp") is not None), ext_ts_rej

                resolved_contract_lots.add(parsed_book.resolved_lot_size)
                priced_neo_symbols.append(ns)
                b_lots = parsed_book.bid_depth_lots or 0
                a_lots = parsed_book.ask_depth_lots or 0
                conservative_lots = min(b_lots, a_lots) if (b_lots and a_lots) else (b_lots or a_lots)
                return (
                    parsed_book.best_bid,
                    parsed_book.best_ask,
                    conservative_lots,
                    parsed_book.bid_depth_lots,
                    parsed_book.ask_depth_lots,
                    False,
                    {
                        "touch_bid_qty": parsed_book.touch_bid_qty,
                        "touch_ask_qty": parsed_book.touch_ask_qty,
                        "visible_buy_qty": parsed_book.visible_buy_qty,
                        "visible_sell_qty": parsed_book.visible_sell_qty,
                        "buy_levels": [[p, q] for p, q in parsed_book.buy_levels],
                        "sell_levels": [[p, q] for p, q in parsed_book.sell_levels],
                        "quote_response_ts_candidates": q_ts_cands,
                        "quote_as_of_ist": ts_eval.normalized_quote_as_of_ist,
                        "ts_semantics": ts_eval.semantics_status,
                        "ts_provenance": ts_prov,
                    },
                )

            # Support synthetic unit-test chain payloads that embed pre-normalized `market_depth` / `depth`
            inline_md = leg.get("market_depth") or leg.get("depth")
            if isinstance(inline_md, dict) and inline_md:
                if isinstance(inline_md.get("buy"), list) or isinstance(inline_md.get("sell"), list):
                    parsed_inline = interpret_depth_book(
                        {"depth": inline_md, "mktLot": expiry_lot_size},
                        fallback_lot_size=expiry_lot_size,
                    )
                    if parsed_inline.is_structurally_valid:
                        resolved_contract_lots.add(parsed_inline.resolved_lot_size)
                        if ns:
                            priced_neo_symbols.append(ns)
                        b_l = parsed_inline.bid_depth_lots or 0
                        a_l = parsed_inline.ask_depth_lots or 0
                        return (
                            parsed_inline.best_bid,
                            parsed_inline.best_ask,
                            min(b_l, a_l) if (b_l and a_l) else (b_l or a_l),
                            parsed_inline.bid_depth_lots,
                            parsed_inline.ask_depth_lots,
                            False,
                            {
                                "touch_bid_qty": parsed_inline.touch_bid_qty,
                                "touch_ask_qty": parsed_inline.touch_ask_qty,
                                "visible_buy_qty": parsed_inline.visible_buy_qty,
                                "visible_sell_qty": parsed_inline.visible_sell_qty,
                                "buy_levels": [[p, q] for p, q in parsed_inline.buy_levels],
                                "sell_levels": [[p, q] for p, q in parsed_inline.sell_levels],
                                "quote_response_ts_candidates": {},
                                "quote_as_of_ist": None,
                                "ts_semantics": "",
                                "ts_provenance": {},
                            },
                        )
                    if ns:
                        rejected_neo_symbols.append(ns)
                    rejected_depth_books.append(
                        {
                            "expiry_date": expiry_date,
                            "strike": strike_val,
                            "option_type": opt_type,
                            "neo_symbol": ns,
                            "rejection_reason": parsed_inline.rejection_reason,
                            "raw_payload": parsed_inline.raw_payload,
                        }
                    )
                    return None, None, 0, None, None, bool(leg.get("ltp") is not None), dict(_EMPTY_EXT)

                b_val, a_val = _coerce_executable_pair(
                    inline_md.get("best_bid"), inline_md.get("best_ask"), False
                )
                if b_val is not None and a_val is not None:
                    if ns:
                        priced_neo_symbols.append(ns)
                    lot_div = expiry_lot_size if expiry_lot_size > 0 else 1
                    vis_b = int(inline_md["total_buy_qty"]) if inline_md.get("total_buy_qty") is not None else None
                    vis_s = int(inline_md["total_sell_qty"]) if inline_md.get("total_sell_qty") is not None else None
                    buy_lots = (
                        int(inline_md["total_buy_qty_lots"])
                        if inline_md.get("total_buy_qty_lots") is not None
                        else (vis_b // lot_div if vis_b is not None and expiry_lot_size > 0 else None)
                    )
                    sell_lots = (
                        int(inline_md["total_sell_qty_lots"])
                        if inline_md.get("total_sell_qty_lots") is not None
                        else (vis_s // lot_div if vis_s is not None and expiry_lot_size > 0 else None)
                    )
                    d_lots = (
                        min(buy_lots, sell_lots)
                        if (buy_lots is not None and sell_lots is not None)
                        else int(buy_lots or sell_lots or 0)
                    )
                    # Summary-only inline quote has no level array: touch_bid_qty/touch_ask_qty stay None
                    # so full-size touch fill is never assumed without level-by-level depth evidence.
                    return (
                        b_val,
                        a_val,
                        d_lots,
                        buy_lots,
                        sell_lots,
                        False,
                        {
                            "touch_bid_qty": None,
                            "touch_ask_qty": None,
                            "visible_buy_qty": vis_b,
                            "visible_sell_qty": vis_s,
                            "buy_levels": [],
                            "sell_levels": [],
                            "quote_response_ts_candidates": {},
                            "quote_as_of_ist": None,
                            "ts_semantics": "",
                            "ts_provenance": {},
                        },
                    )

                if ns:
                    rejected_neo_symbols.append(ns)
                rejected_depth_books.append(
                    {
                        "expiry_date": expiry_date,
                        "strike": strike_val,
                        "option_type": opt_type,
                        "neo_symbol": ns,
                        "rejection_reason": "INLINE_BOOK_NON_EXECUTABLE",
                        "raw_payload": dict(inline_md),
                    }
                )

            # Unfetched or LTP-only strike: NEVER substitute LTP or 0.0 as bid/ask.
            return None, None, 0, None, None, bool(leg.get("ltp") is not None), dict(_EMPTY_EXT)

        for row in strikes_raw:
            strike_val = float(row.get("strike_price") or 0.0)
            ce = row.get("CE") or {}
            pe = row.get("PE") or {}

            ts_candidates: dict[str, str] = {}
            for cand in (
                "exchange_timestamp_ist",
                "quote_updated_at",
                "updated_at",
                "last_update",
                "lstup_time",
                "timestamp",
                "ltp_timestamp",
                "last_trade_time",
            ):
                v = row.get(cand) or ce.get(cand) or pe.get(cand)
                if v:
                    ts_candidates[cand] = str(v)

            c_bid, c_ask, c_lots, c_b_lots, c_a_lots, ce_ltp_only, c_ext = _resolve_leg_book(
                ce, strike_val, "CE"
            )
            p_bid, p_ask, p_lots, p_b_lots, p_a_lots, pe_ltp_only, p_ext = _resolve_leg_book(
                pe, strike_val, "PE"
            )
            for side_lbl, ext_ts in (("CE_quote", c_ext["quote_response_ts_candidates"]), ("PE_quote", p_ext["quote_response_ts_candidates"])):
                for k_ts, v_ts in ext_ts.items():
                    ts_candidates[f"{side_lbl}.{k_ts}"] = v_ts

            c_as_of = c_ext["quote_as_of_ist"]
            p_as_of = p_ext["quote_as_of_ist"]
            present_as_ofs = [x for x in (c_as_of, p_as_of) if x]
            present_sems = {
                s for s in (c_ext["ts_semantics"], p_ext["ts_semantics"]) if s
            }
            if present_as_ofs:
                strike_update_iso = min(present_as_ofs)
                if present_sems == {"VERIFIED_CONTRACT_QUOTE_AS_OF"} and (
                    (c_bid is None or c_as_of is not None)
                    and (p_bid is None or p_as_of is not None)
                ):
                    strike_ts_source = "VERIFIED_CONTRACT_QUOTE_AS_OF"
                elif "UNVERIFIED_KOTAK_REST_LSTUP_TIME" in present_sems:
                    strike_ts_source = "UNVERIFIED_KOTAK_REST_LSTUP_TIME"
                else:
                    strike_ts_source = "UNVERIFIED_CANDIDATE_FIELD"
            else:
                strike_update_iso = dec_ist.isoformat()
                strike_ts_source = "FETCH_TIME_NOT_EXCHANGE"

            strikes.append(
                {
                    "strike": strike_val,
                    "call_bid": c_bid,
                    "call_ask": c_ask,
                    "put_bid": p_bid,
                    "put_ask": p_ask,
                    "call_oi": int(ce.get("oi_current") or 0),
                    "put_oi": int(pe.get("oi_current") or 0),
                    "call_volume": int(ce.get("volume") or 0),
                    "put_volume": int(pe.get("volume") or 0),
                    "call_depth_lots": c_lots,
                    "put_depth_lots": p_lots,
                    "call_bid_depth_lots": c_b_lots,
                    "call_ask_depth_lots": c_a_lots,
                    "put_bid_depth_lots": p_b_lots,
                    "put_ask_depth_lots": p_a_lots,
                    "call_touch_bid_qty": c_ext["touch_bid_qty"],
                    "call_touch_ask_qty": c_ext["touch_ask_qty"],
                    "put_touch_bid_qty": p_ext["touch_bid_qty"],
                    "put_touch_ask_qty": p_ext["touch_ask_qty"],
                    "call_visible_buy_qty": c_ext["visible_buy_qty"],
                    "call_visible_sell_qty": c_ext["visible_sell_qty"],
                    "put_visible_buy_qty": p_ext["visible_buy_qty"],
                    "put_visible_sell_qty": p_ext["visible_sell_qty"],
                    "call_buy_levels": c_ext["buy_levels"],
                    "call_sell_levels": c_ext["sell_levels"],
                    "put_buy_levels": p_ext["buy_levels"],
                    "put_sell_levels": p_ext["sell_levels"],
                    "call_ltp_only": ce_ltp_only,
                    "put_ltp_only": pe_ltp_only,
                    "fetch_started_at_ist": fetch_started_dt.isoformat(),
                    "fetch_completed_at_ist": fetch_completed_dt.isoformat(),
                    "call_quote_as_of_ist": c_as_of,
                    "put_quote_as_of_ist": p_as_of,
                    "call_ts_provenance": c_ext["ts_provenance"],
                    "put_ts_provenance": p_ext["ts_provenance"],
                    "update_time_ist": strike_update_iso,
                    "quote_ts_source": strike_ts_source,
                    "exchange_ts_candidates": ts_candidates,
                }
            )

        if len(resolved_contract_lots) == 1:
            final_lot_size = next(iter(resolved_contract_lots))
        elif len(resolved_contract_lots) > 1:
            raise KotakAdapterError(
                f"Inconsistent live mktLot values within single expiry {expiry_date}: {sorted(resolved_contract_lots)}"
            )
        else:
            final_lot_size = expiry_lot_size

        def _dedup_seq(seq: Sequence[str]) -> list[str]:
            seen_local: set[str] = set()
            out_local: list[str] = []
            for item_s in seq:
                if item_s and item_s not in seen_local:
                    seen_local.add(item_s)
                    out_local.append(item_s)
            return out_local

        listed_u = _dedup_seq(all_chain_neo_symbols)
        selected_u = _dedup_seq(shortlisted_neos)
        requested_u = _dedup_seq(requested_neos)
        returned_u = _dedup_seq(returned_neo_symbols)
        # Account for any requested symbol not even present in strikes_raw (if ever passed)
        accounted_req = set(returned_u) | set(missing_neo_symbols)
        for req_ns in requested_u:
            if req_ns not in accounted_req:
                if req_ns in quotes_map or req_ns.split("|", 1)[1] in quotes_map:
                    returned_u.append(req_ns)
                else:
                    missing_neo_symbols.append(req_ns)
        missing_u = _dedup_seq(missing_neo_symbols)
        rejected_u = _dedup_seq(rejected_neo_symbols)
        priced_u = _dedup_seq(priced_neo_symbols)

        coverage_ledger = {
            "expiry_date": expiry_date,
            "listed": listed_u,
            "selected": selected_u,
            "requested": requested_u,
            "returned": returned_u,
            "missing": missing_u,
            "rejected": rejected_u,
            "priced": priced_u,
            "counts": {
                "listed": len(listed_u),
                "selected": len(selected_u),
                "requested": len(requested_u),
                "returned": len(returned_u),
                "missing": len(missing_u),
                "rejected": len(rejected_u),
                "priced": len(priced_u),
            },
        }

        return {
            "verification_tag": "UNVERIFIED_KOTAK_SCHEMA",
            "lot_size": final_lot_size if final_lot_size > 0 else None,
            "fetch_started_at_ist": fetch_started_dt.isoformat(),
            "fetch_completed_at_ist": fetch_completed_dt.isoformat(),
            "shortlisted_neo_symbols": selected_u,
            "requested_neo_symbols": requested_u,
            "returned_neo_symbols": returned_u,
            "missing_neo_symbols": missing_u,
            "rejected_neo_symbols": rejected_u,
            "priced_neo_symbols": priced_u,
            "chain_neo_symbols": listed_u,
            "coverage_ledger": coverage_ledger,
            "rejected_depth_books": rejected_depth_books,
            "strikes": strikes,
        }

    def fetch_index_candles_raw(self, decision_dt_ist: dt.datetime) -> dict[str, Any]:
        dec_ist = self._assert_schedule_approved(decision_dt_ist)
        neosymbol = "nse_cm|Nifty 50"

        upstream_validate: Callable[..., Any] | None = None
        try:
            from kotak_option_chain.request_safety import validate_history as _up_vh  # type: ignore[import-not-found]

            upstream_validate = _up_vh
        except Exception:
            upstream_validate = None

        def _validated_hist_call(interval: str, from_str: str, to_str: str) -> Any:
            validate_history(neosymbol, interval, from_str, to_str)
            if callable(upstream_validate):
                upstream_validate(
                    neosymbol=neosymbol,
                    interval=interval,
                    from_date=from_str,
                    to_date=to_str,
                )
            return self._call(
                f"hist:{neosymbol}:{interval}:{from_str}:{to_str}",
                dec_ist,
                lambda: self._acquire_authenticated_client().historical_data(
                    neosymbol=neosymbol,
                    interval=interval,
                    from_date=from_str,
                    to_date=to_str,
                ),
            )

        def _fetch_chunked_rows(
            interval: str, total_days: int, max_chunk_days: int
        ) -> list[dict[str, Any]]:
            end_d = dec_ist.date()
            start_d = end_d - dt.timedelta(days=total_days)
            merged_by_open: dict[str, dict[str, Any]] = {}
            for c_from, c_to in chunk_date_range(start_d, end_d, max_chunk_days=max_chunk_days):
                resp = _validated_hist_call(interval, c_from, c_to)
                for row in normalize_kotak_history_rows(resp, interval):
                    merged_by_open[row["open_time_ist"]] = row
            return [merged_by_open[k] for k in sorted(merged_by_open.keys())]

        # Completed 1D candles never mutate across the 10:15-14:15 IST intraday slots on the same date.
        cache_dir = getattr(self, "cache_dir", None)
        d1_cache_file = (
            Path(cache_dir) / f"nifty_1d_completed_{dec_ist.date().isoformat()}.json"
            if cache_dir
            else None
        )
        candles_1d: list[dict[str, Any]] | None = None
        if d1_cache_file is not None and d1_cache_file.exists():
            # Enforce schedule guard without consuming a network pacing slot for local disk cache
            self._assert_schedule_approved(dec_ist)
            try:
                cached_rows = json.loads(d1_cache_file.read_text(encoding="utf-8"))
                if isinstance(cached_rows, list) and cached_rows:
                    candles_1d = cached_rows
            except Exception:
                candles_1d = None

        if candles_1d is None:
            candles_1d = _fetch_chunked_rows("D", total_days=400, max_chunk_days=179)
            if d1_cache_file is not None and candles_1d:
                try:
                    d1_cache_file.parent.mkdir(parents=True, exist_ok=True)
                    completed_prior = [
                        r
                        for r in candles_1d
                        if ensure_ist(dt.datetime.fromisoformat(r["close_time_ist"])) <= dec_ist
                    ]
                    if completed_prior:
                        tmp_f = d1_cache_file.with_suffix(".tmp")
                        tmp_f.write_text(json.dumps(completed_prior), encoding="utf-8")
                        tmp_f.replace(d1_cache_file)
                except Exception:
                    pass

        candles_1h = _fetch_chunked_rows("60min", total_days=30, max_chunk_days=89)
        candles_15m = _fetch_chunked_rows("15min", total_days=10, max_chunk_days=89)

        return {
            "candles_1d": candles_1d,
            "candles_1h": candles_1h,
            "candles_15m_closing": candles_15m,
        }

    def fetch_india_vix_raw(
        self, decision_dt_ist: dt.datetime
    ) -> TimestampedDatum[float | None]:
        dec_ist = ensure_ist(decision_dt_ist)
        return TimestampedDatum.create(
            value=None,
            as_of_dt=dec_ist,
            fetched_at_dt=dec_ist,
            source_id="kotak_vix_endpoint_unverified",
            max_staleness_seconds=180,
            verification_tag="UNVERIFIED",
        )
