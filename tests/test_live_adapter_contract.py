"""LiveKotakNeoAdapter fixture contract tests (mocked service layer, zero network).

Covers: normalization, candle boundaries, quote as-of/staleness, chain sync, expiry
handling, 429 hard stop, schedule guard (no network outside approved IST slots).
The adapter is constructed via __new__ so the pinned ~/Kotak_API repo is never imported.
"""
import datetime as dt
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nifty_analyst.adapters.kotak_neo import (
    ChainSnapshotEnvelope,
    KotakAdapterError,
    KotakRateLimit429Error,
    LiveKotakNeoAdapter,
    MissingQuoteTimestampError,
    ProvisionalSerialRateLimiter,
    UnapprovedScheduleSlotError,
    parse_and_validate_option_chain,
    slice_completed_candles,
)
from nifty_analyst.scrubber import IST

DEC = dt.datetime(2026, 11, 9, 10, 15, tzinfo=IST)   # Monday 10:15 IST approved slot
OFFSLOT = dt.datetime(2026, 11, 9, 9, 0, tzinfo=IST)  # outside all slots


class FakeService:
    """Records calls; scripted responses per method name."""
    def __init__(self, responses=None, exc=None):
        self.responses = responses or {}
        self.exc = exc or {}
        self.calls = []

    def _do(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))
        if name in self.exc:
            raise self.exc[name]
        return self.responses.get(name, {"status": "OK"})

    def get_expiries(self, *a, **k): return self._do("get_expiries", *a, **k)
    def get_futures_chain(self, *a, **k): return self._do("get_futures_chain", *a, **k)
    def get_full_option_chain(self, *a, **k): return self._do("get_full_option_chain", *a, **k)


class _AuthStub:
    def require_authenticated_client(self):
        return (object(), None, None)


def make_adapter(service, enforce=True, min_interval=0.0, calendar=None):
    a = LiveKotakNeoAdapter.__new__(LiveKotakNeoAdapter)
    a._service = service
    a._auth = _AuthStub()
    a.rate_limiter = ProvisionalSerialRateLimiter(min_interval_seconds=min_interval)
    a.enforce_schedule = enforce
    a.depth_strikes_each_side = 10
    a.trading_calendar = calendar or {}
    a._cache = {}
    return a


OK_SERVICE = {
    "get_expiries": {"status": "OK", "expiries": ["2026-11-17", "2026-11-24"]},
    "get_futures_chain": {"status": "OK", "lot_size": 65, "nearest_futures_ltp": 25050.5},
}


class TestScheduleGuard(unittest.TestCase):
    def test_offslot_request_never_touches_service(self):
        svc = FakeService(OK_SERVICE)
        a = make_adapter(svc, enforce=True)
        with self.assertRaises(UnapprovedScheduleSlotError):
            a.fetch_contract_master_raw(OFFSLOT)
        self.assertEqual(svc.calls, [], "no network call may happen outside approved slots")

    def test_weekend_slot_rejected(self):
        svc = FakeService(OK_SERVICE)
        a = make_adapter(svc, enforce=True)
        sat = dt.datetime(2026, 11, 7, 10, 15, tzinfo=IST)
        with self.assertRaises(UnapprovedScheduleSlotError):
            a.fetch_contract_master_raw(sat)
        self.assertEqual(svc.calls, [])

    def test_approved_slot_calls_service_once_per_cache_key(self):
        svc = FakeService(OK_SERVICE)
        a = make_adapter(svc, enforce=True)
        out = a.fetch_contract_master_raw(DEC)
        self.assertEqual(out["symbol"], "NIFTY")
        self.assertEqual(out["spot"]["value"], 25050.5)
        self.assertEqual([e["expiry_date"] for e in out["expiries"]], ["2026-11-17", "2026-11-24"])
        self.assertEqual(out["expiries"][0]["lot_size"], 65)
        n = len(svc.calls)
        a.fetch_contract_master_raw(DEC)  # cached: no new service calls
        self.assertEqual(len(svc.calls), n)


class TestRateLimit429(unittest.TestCase):
    def test_exception_429_hard_stop_zero_retries(self):
        svc = FakeService(OK_SERVICE, exc={"get_expiries": Exception("HTTP 429 too many requests")})
        a = make_adapter(svc)
        with self.assertRaises(KotakRateLimit429Error):
            a.fetch_contract_master_raw(DEC)
        self.assertEqual(len(svc.calls), 1, "zero retries on 429")

    def test_dict_status_429_hard_stop(self):
        svc = FakeService({"get_expiries": {"status": "429", "stCode": "429"}})
        a = make_adapter(svc)
        with self.assertRaises(KotakRateLimit429Error):
            a.fetch_contract_master_raw(DEC)
        self.assertEqual(len(svc.calls), 1)

    def test_serial_limiter_enforces_interval(self):
        sleeps = []
        clock = [1000.0]
        lim = ProvisionalSerialRateLimiter(
            min_interval_seconds=2.5,
            clock_fn=lambda: clock[0],
            sleep_fn=lambda s: (sleeps.append(s), clock.__setitem__(0, clock[0] + s)),
        )
        lim.wait_for_slot()
        lim.wait_for_slot()
        self.assertEqual(sleeps, [2.5])
        self.assertEqual(lim.request_count, 2)


class TestCalendarAwareGuard(unittest.TestCase):
    def test_holiday_weekday_rejected_when_calendar_supplied(self):
        svc = FakeService(OK_SERVICE)
        cal = {"holiday_dates": frozenset({"2026-11-24"})}  # Tuesday, FO holiday
        a = make_adapter(svc, calendar=cal)
        hol = dt.datetime(2026, 11, 24, 10, 15, tzinfo=IST)
        with self.assertRaises(UnapprovedScheduleSlotError):
            a.fetch_contract_master_raw(hol)
        self.assertEqual(svc.calls, [])

    def test_special_sunday_session_interval_enforced(self):
        svc = FakeService(OK_SERVICE)
        cal = {"special_sessions": {"2026-11-08": ("14:20", "15:40")}}  # Muhurat F&O window
        sun_before = dt.datetime(2026, 11, 8, 14, 15, tzinfo=IST)   # before session open
        sun_in = dt.datetime(2026, 11, 8, 14, 25, tzinfo=IST)       # inside session
        a = make_adapter(svc)  # no calendar: weekend rejected (8 Nov is login-only)
        with self.assertRaises(UnapprovedScheduleSlotError):
            a.fetch_contract_master_raw(sun_in)
        self.assertEqual(svc.calls, [])
        a2 = make_adapter(svc, calendar=cal)
        with self.assertRaises(UnapprovedScheduleSlotError):
            a2.fetch_contract_master_raw(sun_before)  # before open: still rejected
        self.assertEqual(svc.calls, [])
        out = a2.fetch_contract_master_raw(sun_in)
        self.assertEqual(out["symbol"], "NIFTY")


class TestContractMaster(unittest.TestCase):
    def test_non_ok_expiries_raises(self):
        svc = FakeService({"get_expiries": {"status": "FAILED", "error": "bad"}})
        a = make_adapter(svc)
        with self.assertRaises(KotakAdapterError):
            a.fetch_contract_master_raw(DEC)

    def test_zero_lot_size_raises(self):
        svc = FakeService({**OK_SERVICE, "get_futures_chain": {"status": "OK", "lot_size": 0}})
        a = make_adapter(svc)
        with self.assertRaises(KotakAdapterError):
            a.fetch_contract_master_raw(DEC)

    def test_missing_spot_raises_no_silent_fallback(self):
        svc = FakeService({**OK_SERVICE, "get_futures_chain": {"status": "OK", "lot_size": 65}})
        svc._fetch_spot_quote = lambda client, sym: {"display_symbol": "NIFTY"}
        a = make_adapter(svc)
        with self.assertRaises(KotakAdapterError):
            a.fetch_contract_master_raw(DEC)


class TestOptionChainNormalization(unittest.TestCase):
    CHAIN_RESP = {"status": "OK", "strikes": [
        {"strike_price": 25000, "CE": {"depth": {"best_bid": "101.5", "best_ask": "103.2",
          "total_buy_qty_lots": "120"}, "oi_current": "5500", "volume": "900"},
         "PE": {"depth": {"best_bid": "88.0", "best_ask": "89.5",
          "total_sell_qty_lots": "140"}, "oi_current": "6100", "volume": "1100"}},
        {"strike_price": 25050, "CE": {"ltp": 77.25}, "PE": {"ltp": 110.1}},
    ]}

    def test_normalization_and_expiry_routing(self):
        svc = FakeService({"get_full_option_chain": self.CHAIN_RESP})
        a = make_adapter(svc)
        raw = a.fetch_option_chain_raw("2026-11-17", DEC)
        # service received the exact requested expiry
        name, args, kwargs = svc.calls[0]
        self.assertEqual(kwargs.get("expiry") or (args[2] if len(args) > 2 else None), "2026-11-17")
        self.assertEqual(raw["verification_tag"], "UNVERIFIED_KOTAK_SCHEMA")
        s0, s1 = raw["strikes"]
        self.assertEqual((s0["strike"], s0["call_bid"], s0["put_ask"]), (25000.0, 101.5, 89.5))
        self.assertEqual((s0["call_oi"], s0["put_volume"], s0["call_depth_lots"]), (5500, 1100, 120))
        # ltp fallback when depth missing
        self.assertEqual((s1["call_bid"], s1["call_ask"], s1["put_bid"]), (77.25, 77.25, 110.1))
        # fetch-time stamped for staleness bound
        self.assertEqual(s0["update_time_ist"], DEC.isoformat())
        # different expiry = new service call (no cross-expiry cache bleed)
        a.fetch_option_chain_raw("2026-11-24", DEC)
        self.assertEqual(len(svc.calls), 2)
        a.fetch_option_chain_raw("2026-11-17", DEC)  # same key cached
        self.assertEqual(len(svc.calls), 2)

    def test_empty_chain_raises(self):
        svc = FakeService({"get_full_option_chain": {"status": "OK", "strikes": []}})
        a = make_adapter(svc)
        raw = a.fetch_option_chain_raw("2026-11-17", DEC)
        with self.assertRaises(KotakAdapterError):
            parse_and_validate_option_chain(raw, "2026-11-17", DEC)

    def test_non_ok_chain_raises(self):
        svc = FakeService({"get_full_option_chain": {"status": "FAILED"}})
        a = make_adapter(svc)
        with self.assertRaises(KotakAdapterError):
            a.fetch_option_chain_raw("2026-11-17", DEC)


class TestLtpOnlyNotExecutable(unittest.TestCase):
    def test_ltp_only_side_zeroed_and_ts_source_tagged(self):
        resp = {"status": "OK", "strikes": [
            {"strike_price": 25000, "CE": {"ltp": 77.25},
             "PE": {"depth": {"best_bid": "88.0", "best_ask": "89.5",
                              "total_sell_qty_lots": "140"}, "oi_current": "6100", "volume": "1100"}},
        ]}
        svc = FakeService({"get_full_option_chain": resp})
        a = make_adapter(svc)
        raw = a.fetch_option_chain_raw("2026-11-17", DEC)
        self.assertTrue(raw["strikes"][0]["call_ltp_only"])
        self.assertEqual(raw["strikes"][0]["quote_ts_source"], "FETCH_TIME_NOT_EXCHANGE")
        env = parse_and_validate_option_chain(raw, "2026-11-17", DEC)
        q = env.quotes[0]
        self.assertEqual((q.call_bid, q.call_ask, q.call_depth_lots), (0.0, 0.0, 0),
                         "LTP-only side must never price an executable leg")
        self.assertEqual((q.put_bid, q.put_ask, q.put_depth_lots), (88.0, 89.5, 140))
        self.assertIn("TS_FETCH_TIME_NOT_EXCHANGE", env.source_verification_tag)


class TestExchangeAsOf(unittest.TestCase):
    def test_candidate_fields_preserved_but_never_drive_staleness(self):
        # Until field semantics are verified live, ltp/last-trade/update fields are NOT a
        # quote as-of: staleness always uses fetch time and the tag says so.
        resp = {"status": "OK", "strikes": [
            {"strike_price": 25000, "exchange_timestamp_ist": "2026-11-09T10:14:30+05:30",
             "last_trade_time": "10:14:31",
             "CE": {"depth": {"best_bid": 101.5, "best_ask": 103.2, "total_buy_qty_lots": 120},
                    "ltp_timestamp": "x"},
             "PE": {"depth": {"best_bid": 88.0, "best_ask": 89.5, "total_sell_qty_lots": 140}}},
        ]}
        svc = FakeService({"get_full_option_chain": resp})
        a = make_adapter(svc)
        raw = a.fetch_option_chain_raw("2026-11-17", DEC)
        s0 = raw["strikes"][0]
        self.assertEqual(s0["quote_ts_source"], "FETCH_TIME_NOT_EXCHANGE")
        self.assertEqual(s0["update_time_ist"], DEC.isoformat())
        self.assertEqual(s0["exchange_ts_candidates"].get("exchange_timestamp_ist"),
                         "2026-11-09T10:14:30+05:30")
        self.assertEqual(s0["exchange_ts_candidates"].get("last_trade_time"), "10:14:31")
        env = parse_and_validate_option_chain(raw, "2026-11-17", DEC)
        self.assertIn("FETCH_TIME", env.source_verification_tag)

    def test_missing_exchange_ts_marks_fetch_time(self):
        svc = FakeService({"get_full_option_chain": {"status": "OK", "strikes": [
            {"strike_price": 25000,
             "CE": {"depth": {"best_bid": 1, "best_ask": 2, "total_buy_qty_lots": 1}},
             "PE": {"depth": {"best_bid": 1, "best_ask": 2, "total_sell_qty_lots": 1}}}]}})
        a = make_adapter(svc)
        raw = a.fetch_option_chain_raw("2026-11-17", DEC)
        self.assertEqual(raw["strikes"][0]["quote_ts_source"], "FETCH_TIME_NOT_EXCHANGE")


class TestQuoteAsOfStaleness(unittest.TestCase):
    def _raw(self, ts, strike="25000"):
        return {"strikes": [{"strike": strike, "call_bid": "1", "call_ask": "2",
                             "put_bid": "3", "put_ask": "4", "update_time_ist": ts}]}

    def test_missing_timestamp_rejected(self):
        raw = {"strikes": [{"strike": 25000, "call_bid": 1, "call_ask": 2,
                            "put_bid": 3, "put_ask": 4}]}
        with self.assertRaises(MissingQuoteTimestampError):
            parse_and_validate_option_chain(raw, "2026-11-17", DEC)

    def test_fresh_quote_not_stale_and_sorted(self):
        ts = (DEC - dt.timedelta(seconds=179)).isoformat()
        raw = {"strikes": [self._raw(ts, "25050")["strikes"][0], self._raw(ts)["strikes"][0]]}
        env = parse_and_validate_option_chain(raw, "2026-11-17", DEC)
        self.assertFalse(env.any_stale)
        self.assertEqual([q.strike for q in env.quotes], [25000.0, 25050.0])
        self.assertEqual(env.source_verification_tag, "UNVERIFIED_KOTAK_SCHEMA")

    def test_old_quote_flagged_stale(self):
        ts = (DEC - dt.timedelta(seconds=181)).isoformat()
        env = parse_and_validate_option_chain(self._raw(ts), "2026-11-17", DEC)
        self.assertTrue(env.any_stale)
        self.assertEqual(env.stale_strikes, [25000.0])

    def test_future_quote_beyond_skew_flagged(self):
        ts = (DEC + dt.timedelta(seconds=30)).isoformat()
        env = parse_and_validate_option_chain(self._raw(ts), "2026-11-17", DEC)
        self.assertTrue(env.any_stale)


class TestCandleBoundaries(unittest.TestCase):
    def _bar(self, o, c, day="2026-11-09"):
        return {"open_time_ist": f"{day}T{o}:00+05:30", "close_time_ist": f"{day}T{c}:00+05:30",
                "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 100}

    def test_valid_boundaries_and_forming_strip(self):
        dec = dt.datetime(2026, 11, 9, 12, 15, tzinfo=IST)
        raw = {
            "candles_1d": [self._bar("09:15", "15:30", day="2026-11-06"),
                           self._bar("09:15", "15:30")],  # today's 1D forming
            "candles_1h": [self._bar("09:15", "10:15"), self._bar("10:15", "11:15"),
                           self._bar("11:15", "12:15"), self._bar("12:15", "13:15")],
            "candles_15m_closing": [self._bar("15:15", "15:30")],
        }
        d1, h1, m15, stripped = slice_completed_candles(raw, dec)
        self.assertEqual(len(d1), 1)          # prior day only
        self.assertEqual(len(h1), 3)          # bar closing exactly at decision is completed
        self.assertIsNone(m15)                # closing bar not yet formed
        self.assertEqual(stripped, 3)         # forming 1D + forming 1H + 15m
        self.assertTrue(all(b.timeframe == "1H" for b in h1))

    def test_invalid_1h_boundary_rejected(self):
        raw = {"candles_1h": [self._bar("10:00", "11:00")]}
        with self.assertRaises(KotakAdapterError):
            slice_completed_candles(raw, DEC)

    def test_invalid_final_15m_boundary_rejected(self):
        raw = {"candles_15m_closing": [self._bar("15:15", "15:31")]}
        with self.assertRaises(KotakAdapterError):
            slice_completed_candles(raw, DEC)


def test_live_adapter_contract_suite():
    """Module-level entry so run_tests.py (function discovery) executes the TestCase suite."""
    suite = unittest.TestSuite()
    for cls in (TestScheduleGuard, TestCalendarAwareGuard, TestRateLimit429, TestContractMaster,
                TestOptionChainNormalization, TestLtpOnlyNotExecutable, TestExchangeAsOf,
                TestQuoteAsOfStaleness, TestCandleBoundaries):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
    result = unittest.TextTestRunner(verbosity=0).run(suite)
    assert result.wasSuccessful(), "live adapter contract suite failed"


if __name__ == "__main__":
    unittest.main()
