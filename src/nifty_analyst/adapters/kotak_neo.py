"""Adapter 2 (Option Chain & Quotes) and Adapter 3 (Historical Index OHLC) for Kotak Neo.

STRICT CONSTRAINTS (Implementation Plan v3, Sections 3 & 10):
- Read-only interface: ZERO order placement, modification, or broker write calls exist anywhere.
- UNVERIFIED INTERFACES: Exact Kotak Neo SDK method names, response schemas, and rate limits
  are tagged UNVERIFIED and isolated behind KotakNeoReadOnlyPort with a fixture-backed mock
  for offline execution.
- Quote-level update_time (IST) is mandatory; quotes missing timestamps are rejected.
- Max staleness: 180 seconds for active strikes during market hours.
- Provisional serial rate limiter: default 1 request per 2.5 seconds; any HTTP 429 stops
  the run immediately and reports with zero automatic retry and zero limit probing.
- Completed-candle lock: only bars whose close_time_ist <= decision_timestamp_ist are returned,
  verifying the 6 hourly bars (09:15-15:15) and the final 15-minute bar (15:15-15:30 IST).
"""

from __future__ import annotations

import abc
import datetime as dt
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from nifty_analyst.engine.pricing import RawStrikeQuote
from nifty_analyst.scrubber import IST, ensure_ist, is_approved_schedule_slot
from nifty_analyst.state import CandleBar, TimestampedDatum


class KotakAdapterError(RuntimeError):
    """Base error for read-only Kotak adapter failures."""


class KotakRateLimit429Error(KotakAdapterError):
    """Raised immediately on HTTP 429; halts the run with zero retry and zero probing."""


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
        rate_limiter: ProvisionalSerialRateLimiter | None = None,
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
        cache_key = f"contract_master:{ ensure_ist(decision_dt_ist).date().isoformat() }"
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
            self._cache[cache_key] = all_chains["chains_by_expiry"][expiry_date]
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
    """Isolated stub for live Kotak Neo SDK integration [UNVERIFIED SDK INTERFACE].

    Per Hard Constraint 4:
    'the exact Kotak Neo SDK method names, response schemas, and rate limits are not verified.
    Do not invent API calls. Isolate every SDK interaction behind a small adapter interface
    with a clearly marked fixture-backed mock for offline development, and tag each
    real-endpoint mapping as UNVERIFIED until confirmed against the actual SDK documentation.'
    """

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


def parse_and_validate_option_chain(
    raw_chain: dict[str, Any],
    expiry_date: str,
    decision_dt_ist: dt.datetime,
    max_staleness_seconds: int = 180,
) -> ChainSnapshotEnvelope:
    """Parse raw option chain quotes, enforcing quote-level update_time and 180s staleness bound."""
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
        ts_str = str(item.get("update_time_ist", "")).strip()
        if not ts_str:
            raise MissingQuoteTimestampError(
                f"Option quote at strike {strike_val} (expiry {expiry_date}) is missing "
                f"mandatory quote-level update_time_ist timestamp."
            )
        if item.get("call_ltp_only") or item.get("put_ltp_only"):
            # LTP-only side is not an executable quote; zero its depth fields so downstream
            # liquidity/entry rules can never price a leg from a bare last-trade print.
            item = dict(item)
            if item.get("call_ltp_only"):
                item["call_bid"] = item["call_ask"] = 0.0
                item["call_depth_lots"] = 0
                item["call_bid_depth_lots"] = 0
                item["call_ask_depth_lots"] = 0
            if item.get("put_ltp_only"):
                item["put_bid"] = item["put_ask"] = 0.0
                item["put_depth_lots"] = 0
                item["put_bid_depth_lots"] = 0
                item["put_ask_depth_lots"] = 0
        quote_dt = ensure_ist(dt.datetime.fromisoformat(ts_str))
        quote_times.append(quote_dt)

        age_sec = (dec_ist - quote_dt).total_seconds()
        if age_sec > max_staleness_seconds or age_sec < -5.0:
            stale_strikes.append(strike_val)

        parsed_quotes.append(
            RawStrikeQuote(
                strike=strike_val,
                call_bid=float(item["call_bid"]),
                call_ask=float(item["call_ask"]),
                put_bid=float(item["put_bid"]),
                put_ask=float(item["put_ask"]),
                call_oi=int(item.get("call_oi", 0)),
                put_oi=int(item.get("put_oi", 0)),
                call_volume=int(item.get("call_volume", 0)),
                put_volume=int(item.get("put_volume", 0)),
                call_depth_lots=int(item.get("call_depth_lots", 10)),
                put_depth_lots=int(item.get("put_depth_lots", 10)),
                quote_timestamp_ist=quote_dt.isoformat(),
                expiry_date=expiry_date,
                call_bid_depth_lots=(
                    int(item["call_bid_depth_lots"])
                    if item.get("call_bid_depth_lots") is not None
                    else None
                ),
                call_ask_depth_lots=(
                    int(item["call_ask_depth_lots"])
                    if item.get("call_ask_depth_lots") is not None
                    else None
                ),
                put_bid_depth_lots=(
                    int(item["put_bid_depth_lots"])
                    if item.get("put_bid_depth_lots") is not None
                    else None
                ),
                put_ask_depth_lots=(
                    int(item["put_ask_depth_lots"])
                    if item.get("put_ask_depth_lots") is not None
                    else None
                ),
            )
        )

    earliest = min(quote_times).isoformat()
    latest = max(quote_times).isoformat()
    tag = str(raw_chain.get("verification_tag", "UNVERIFIED_KOTAK_SCHEMA"))
    if ts_sources != {"EXCHANGE"}:
        tag = tag + "|TS_" + "+".join(sorted(ts_sources))
    return ChainSnapshotEnvelope(
        expiry_date=expiry_date,
        quotes=sorted(parsed_quotes, key=lambda q: q.strike),
        earliest_quote_ist=earliest,
        latest_quote_ist=latest,
        any_stale=len(stale_strikes) > 0,
        stale_strikes=stale_strikes,
        source_verification_tag=tag,
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


def slice_completed_candles(
    raw_candles: dict[str, Any],
    decision_dt_ist: dt.datetime,
) -> tuple[list[CandleBar], list[CandleBar], CandleBar | None, int]:
    """Filter 1D, 1H, and final 15M candles to strictly completed bars (close_time_ist <= decision_dt_ist).

    Returns (completed_1d_bars, completed_1h_bars, completed_final_15m_bar, stripped_forming_count).
    """
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


class LiveKotakNeoAdapter(KotakNeoReadOnlyPort):
    """Live Kotak Neo SDK adapter [SDK RESPONSE-SCHEMA MAPPINGS UNVERIFIED UNTIL LIVE-TESTED].

    - Authentication: reuses ~/Kotak_API kotak_option_chain.auth.AuthManager. The daily TOTP
      login is performed at the dedicated login wake; require_authenticated_client() raises
      AuthStateError (SETUP_REQUIRED / TOTP_REQUIRED) otherwise - surfaced as KotakAdapterError,
      never silently bypassed.
    - Rate limiting: ProvisionalSerialRateLimiter (default 2.5s serial). Any HTTP 429 stops the
      run immediately with zero retries and zero probing (Hard Constraint).
    - Schedule guard: optional enforce_schedule_in_live_mode restricts reads to the 5 approved
      IST run slots.
    - Caching: one fetch per (endpoint, expiry, decision timestamp) per adapter instance; the
      pipeline calls each endpoint once per run.
    - Response-schema mappings reuse ~/Kotak_API kotak_option_chain.option_chain parsing helpers
      (the reviewed repo code); they remain tagged UNVERIFIED until confirmed against live
      responses in the first supervised run.
    """

    ENDPOINT_VERIFICATION_STATUS = "UNVERIFIED"

    def __init__(
        self,
        kotak_api_root: str | Path | None = None,
        rate_limiter: ProvisionalSerialRateLimiter | None = None,
        enforce_schedule_in_live_mode: bool = True,
        depth_strikes_each_side: int = 10,
        trading_calendar: dict[str, frozenset] | None = None,
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
        self._auth = AuthManager(store=ConfigStore())
        self._service = OptionChainService(auth_manager=self._auth)
        self.rate_limiter = rate_limiter or ProvisionalSerialRateLimiter(min_interval_seconds=2.5)
        self.enforce_schedule = enforce_schedule_in_live_mode
        self.depth_strikes_each_side = depth_strikes_each_side
        # Verified NSE FO calendar: {"holiday_dates": frozenset, "special_session_dates": frozenset}.
        # Absent calendar = conservative default (weekends rejected, weekdays allowed).
        self.trading_calendar = trading_calendar or {}
        self._cache: dict[str, Any] = {}

    # ----- plumbing -----

    def _guard_request(self, cache_key: str, decision_dt_ist: dt.datetime) -> None:
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
        if cache_key not in self._cache:
            self.rate_limiter.wait_for_slot()

    def _call(self, cache_key: str, decision_dt_ist: dt.datetime, fn) -> Any:
        self._guard_request(cache_key, decision_dt_ist)
        if cache_key in self._cache:
            return self._cache[cache_key]
        try:
            resp = fn()
        except Exception as exc:  # SDK exceptions - check for rate limit, otherwise surface
            msg = str(exc)
            if "429" in msg or "too many" in msg.lower():
                raise KotakRateLimit429Error(
                    f"HTTP 429 from Kotak Neo API ({msg[:200]}). Hard stop; zero retries."
                )
            raise
        if isinstance(resp, dict):
            status = str(resp.get("status") or resp.get("stCode") or "")
            if "429" in status:
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
            f"expiries:{dec_ist.date().isoformat()}", dec_ist,
            lambda: self._service.get_expiries("NIFTY", exchange="nse_fo", instrument_type="option"),
        )
        if exp_resp.get("status") != "OK":
            raise KotakAdapterError(f"Kotak expiries fetch failed: {exp_resp}")
        expiries = list(exp_resp.get("expiries") or [])

        fut_resp = self._call(
            f"futures_chain:{dec_ist.isoformat()}", dec_ist,
            lambda: self._service.get_futures_chain("NIFTY", exchange="nse_fo"),
        )
        if fut_resp.get("status") != "OK":
            raise KotakAdapterError(f"Kotak futures metadata fetch failed: {fut_resp}")
        lot_size = int(fut_resp.get("lot_size") or 0)
        if lot_size <= 0:
            raise KotakAdapterError("Kotak futures metadata returned no usable lot size.")

        spot_val = fut_resp.get("nearest_futures_ltp")
        spot_source = "kotak_neo_nearest_futures_ltp"
        if not spot_val:
            spot_payload = self._call(
                f"spot_index:{dec_ist.isoformat()}", dec_ist,
                lambda: self._service._fetch_spot_quote(self._auth.require_authenticated_client()[0], "NIFTY"),
            )
            if spot_payload and spot_payload.get("display_symbol"):
                raise KotakAdapterError("Index spot quote requires verified ltp mapping; got display only.")
            raise KotakAdapterError("No usable spot reference from Kotak (futures LTP missing).")

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
                    "lot_size": lot_size,
                    "is_holiday_adjusted": False,
                    "holiday_note": "",
                    "is_monthly": False,
                }
                for e in expiries
            ],
        }

    def fetch_option_chain_raw(
        self, expiry_date: str, decision_dt_ist: dt.datetime
    ) -> dict[str, Any]:
        dec_ist = ensure_ist(decision_dt_ist)
        resp = self._call(
            f"chain:{expiry_date}:{dec_ist.isoformat()}", dec_ist,
            lambda: self._service.get_full_option_chain(
                "NIFTY", exchange="nse_fo", expiry=expiry_date, count=self.depth_strikes_each_side,
            ),
        )
        if resp.get("status") != "OK":
            raise KotakAdapterError(f"Kotak option chain fetch failed for {expiry_date}: {resp}")
        strikes_raw = resp.get("strikes") or resp.get("chain") or []
        strikes: list[dict[str, Any]] = []
        for row in strikes_raw:
            ce = row.get("CE") or {}
            pe = row.get("PE") or {}
            # Timestamp semantics (7 Nov 2026 decision): a last-trade time (ltp_timestamp,
            # last_trade_time) is NOT a bid/ask quote as-of, and no field's semantics are
            # verified until the first live session. Candidate fields are preserved verbatim
            # for inspection but NEVER drive the staleness bound until semantics are verified.
            exch_ts = None
            ts_candidates = {}
            for cand in ("exchange_timestamp_ist", "quote_updated_at", "updated_at",
                         "last_update", "timestamp", "ltp_timestamp", "last_trade_time"):
                v = row.get(cand) or ce.get(cand) or pe.get(cand)
                if v:
                    ts_candidates[cand] = str(v)
            ce_depth = ce.get("depth") or {}
            pe_depth = pe.get("depth") or {}
            ce_ltp_only = not ce_depth and ce.get("ltp") is not None
            pe_ltp_only = not pe_depth and pe.get("ltp") is not None
            ce_buy_lots = (
                int(ce_depth["total_buy_qty_lots"])
                if ce_depth.get("total_buy_qty_lots") is not None
                else None
            )
            ce_sell_lots = (
                int(ce_depth["total_sell_qty_lots"])
                if ce_depth.get("total_sell_qty_lots") is not None
                else None
            )
            pe_buy_lots = (
                int(pe_depth["total_buy_qty_lots"])
                if pe_depth.get("total_buy_qty_lots") is not None
                else None
            )
            pe_sell_lots = (
                int(pe_depth["total_sell_qty_lots"])
                if pe_depth.get("total_sell_qty_lots") is not None
                else None
            )
            strikes.append(
                {
                    "strike": float(row.get("strike_price") or 0.0),
                    "call_bid": float(ce_depth.get("best_bid") or ce.get("ltp") or 0.0),
                    "call_ask": float(ce_depth.get("best_ask") or ce.get("ltp") or 0.0),
                    "put_bid": float(pe_depth.get("best_bid") or pe.get("ltp") or 0.0),
                    "put_ask": float(pe_depth.get("best_ask") or pe.get("ltp") or 0.0),
                    "call_oi": int(ce.get("oi_current") or 0),
                    "put_oi": int(pe.get("oi_current") or 0),
                    "call_volume": int(ce.get("volume") or 0),
                    "put_volume": int(pe.get("volume") or 0),
                    "call_depth_lots": int(
                        ce_depth.get("total_buy_qty_lots")
                        or ce_depth.get("total_sell_qty_lots")
                        or 0
                    ),
                    "put_depth_lots": int(
                        pe_depth.get("total_sell_qty_lots")
                        or pe_depth.get("total_buy_qty_lots")
                        or 0
                    ),
                    "call_bid_depth_lots": ce_buy_lots,
                    "call_ask_depth_lots": ce_sell_lots,
                    "put_bid_depth_lots": pe_buy_lots,
                    "put_ask_depth_lots": pe_sell_lots,
                    # LTP is a last-trade print, not an executable quote: flag it so no leg is
                    # ever priced as executable from LTP alone (depth lots stay 0).
                    "call_ltp_only": ce_ltp_only,
                    "put_ltp_only": pe_ltp_only,
                    # Staleness always measures retrieval freshness until a verified quote-
                    # update timestamp exists; live readiness stays BLOCKED meanwhile.
                    "update_time_ist": dec_ist.isoformat(),
                    "quote_ts_source": "FETCH_TIME_NOT_EXCHANGE",
                    "exchange_ts_candidates": ts_candidates,
                }
            )
        return {
            "verification_tag": "UNVERIFIED_KOTAK_SCHEMA",
            "strikes": strikes,
        }

    def fetch_index_candles_raw(self, decision_dt_ist: dt.datetime) -> dict[str, Any]:
        dec_ist = ensure_ist(decision_dt_ist)
        client, _, _ = self._auth.require_authenticated_client()

        def _hist(interval: str, days: int) -> Any:
            from_dt = (dec_ist - dt.timedelta(days=days)).strftime("%Y-%m-%d")
            to_dt = dec_ist.strftime("%Y-%m-%d")
            return self._call(
                f"hist:{interval}:{from_dt}:{to_dt}", dec_ist,
                lambda: client.historical_data(
                    neosymbol="Nifty 50", interval=interval, from_date=from_dt, to_date=to_dt
                ),
            )

        def _rows(resp: Any) -> list[dict[str, Any]]:
            data = (resp.get("data") if isinstance(resp, dict) else resp) or []
            if isinstance(data, dict):
                data = data.get("candles") or data.get("candle") or []
            out = []
            for c in data:
                t_open = c.get("time") or c.get("open_time") or c.get("timestamp") or ""
                out.append(
                    {
                        "open_time_ist": str(t_open),
                        "close_time_ist": str(c.get("close_time") or ""),
                        "open": float(c.get("open") or c.get("o") or 0.0),
                        "high": float(c.get("high") or c.get("h") or 0.0),
                        "low": float(c.get("low") or c.get("l") or 0.0),
                        "close": float(c.get("close") or c.get("c") or 0.0),
                        "volume": float(c.get("volume") or c.get("vol") or 0.0),
                    }
                )
            return out

        return {
            "candles_1d": _rows(_hist("1day", 400)),
            "candles_1h": _rows(_hist("60minute", 30)),
            "candles_15m_closing": _rows(_hist("15minute", 10)),
        }

    def fetch_india_vix_raw(
        self, decision_dt_ist: dt.datetime
    ) -> TimestampedDatum[float | None]:
        dec_ist = ensure_ist(decision_dt_ist)
        # No verified Kotak VIX endpoint [UNVERIFIED]; VIX stays context-only per plan, so a
        # missing value must not block the run. Report None with an UNVERIFIED tag.
        return TimestampedDatum.create(
            value=None,
            as_of_dt=dec_ist,
            fetched_at_dt=dec_ist,
            source_id="kotak_vix_endpoint_unverified",
            max_staleness_seconds=180,
            verification_tag="UNVERIFIED",
        )
