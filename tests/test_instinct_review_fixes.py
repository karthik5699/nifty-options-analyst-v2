"""Contract and regression tests for all 11 Nov 2026 live-readiness review fixes (`P1-1`..`P3`).

Covers:
1. `P1-1` & `P3.1` & `P3.3`: Two-stage targeted live depth funnel (`count=40` without `include_depth`,
   2-token `client.quotes(..., quote_type="all")` batches, nonzero `bid < ask`, live `mktLot=65`
   propagation overriding contract-master `75` in `CandidateStructure.lot_size`, and `None` on unfetched strikes).
2. `P1-1.4` & `P3.4`: Recorded depth book sorting & rejection (`interpret_depth_book` on unsorted,
   empty, locked, and crossed books -> `None`, never `0.0` or `ltp`, with raw payload logged to audit JSONL).
3. Standing API quote retry rule: 1 retry after 30s on transient non-429 quote batch failure then flagged;
   zero retry and immediate hard stop on HTTP `429`.
4. `P1-2`: `validate_history` enforcement (`nse_cm|Nifty 50`, `"D"`/`"60min"`/`"15min"`, `<= 179d` chunks),
   both dict-wrapped and bare-list history shapes, and same-day `1D` disk cache within the `35s` queue cap.
5. `PacedReadTransport`: `35s` queue cap (`<= 14` serial calls) and persisted HTTP `429` cooldown file.
6. `P1-3`: CLI `--open-positions` & `NIFTY_OPEN_POSITIONS_FILE` driving Rule 3 `FAIL` and failing closed
   to `PORTFOLIO_STATE_UNKNOWN` (`RULE_3 == "UNKNOWN"`, `is_live_ready == False`) on missing/corrupt files.
7. `P2-1`: Cash spot (`nse_cm|Nifty 50`) fallback when `nearest_futures_ltp` is missing.
8. `P2-2`: Calendar-aware `decision_driver`, `SingleInstanceFileLock`, and `live_verify` steps.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from pathlib import Path
from typing import Any

import pytest

from nifty_analyst.adapters.calendar_and_contracts import load_nse_holiday_calendar
from nifty_analyst.adapters.kotak_neo import (
    FixtureKotakNeoAdapter,
    KotakAdapterError,
    KotakRateLimit429Error,
    LiveKotakNeoAdapter,
    ProvisionalSerialRateLimiter,
    parse_and_validate_option_chain,
    slice_completed_candles,
)
from nifty_analyst.adapters.request_safety import (
    PacedCooldownActiveError,
    PacedQueueCapExceededError,
    PacedReadTransport,
    interpret_depth_book,
    validate_history,
    validate_quotes_batch,
)
from nifty_analyst.cli import main as cli_main
from nifty_analyst.config import AnalystConfig
from nifty_analyst.engine.pricing import RawStrikeQuote, black76_price
from nifty_analyst.engine.structures import build_option_leg
from nifty_analyst.operations.decision_driver import (
    ConcurrentBrokerStreamError,
    SingleInstanceFileLock,
    get_approved_slots_for_date,
    next_approved_slot_ist,
    run_scheduled_slot,
)
from nifty_analyst.operations.live_verify import (
    verify_live_candles_step,
    verify_live_chain_step,
    verify_live_depth_step,
    verify_live_dry_run_step,
)
from nifty_analyst.pipeline import run_analyst_pipeline
from nifty_analyst.scrubber import IST

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = REPO_ROOT / "fixtures"
DEC_MON = dt.datetime(2026, 11, 9, 10, 15, 0, tzinfo=IST)


def _build_realistic_81_strike_chain_and_quotes(
    spot: float = 25000.0,
    mkt_lot: int = 65,
    t_years: float = 8.0 / 365.25,
    sigma: float = 0.145,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Generate an 81-strike (`count=40`) `OptionChainService` payload (no inline depth) + `quotes` map."""
    strikes_raw: list[dict[str, Any]] = []
    quotes_by_tok: dict[str, dict[str, Any]] = {}
    center = int(round(spot / 50.0) * 50)

    for idx, k in enumerate(range(center - 40 * 50, center + 40 * 50 + 50, 50)):
        ce_tok = f"{50000 + idx * 2}"
        pe_tok = f"{50001 + idx * 2}"
        c_theo = max(2.5, black76_price(spot, float(k), t_years, 0.065, sigma, "CE"))
        p_theo = max(2.5, black76_price(spot, float(k), t_years, 0.065, sigma, "PE"))

        strikes_raw.append(
            {
                "strike_price": float(k),
                "CE": {
                    "neo_symbol": f"nse_fo|{ce_tok}",
                    "ltp": round(c_theo, 2),
                    "oi_current": 120000,
                    "volume": 35000,
                },
                "PE": {
                    "neo_symbol": f"nse_fo|{pe_tok}",
                    "ltp": round(p_theo, 2),
                    "oi_current": 135000,
                    "volume": 40000,
                },
            }
        )

        for tok, mid in ((ce_tok, c_theo), (pe_tok, p_theo)):
            half_sp = max(0.25, round(mid * 0.006, 2))
            b1 = round(max(0.10, mid - half_sp), 2)
            a1 = round(mid + half_sp, 2)
            quotes_by_tok[tok] = {
                "exchange": "nse_fo",
                "exchange_token": tok,
                "mktLot": str(mkt_lot),
                "ltp": str(round(mid, 2)),
                "depth": {
                    "buy": [
                        {"price": str(b1), "quantity": str(mkt_lot * 8), "orders": "4"},
                        {"price": str(round(max(0.05, b1 - 0.15), 2)), "quantity": str(mkt_lot * 12), "orders": "6"},
                    ],
                    "sell": [
                        {"price": str(a1), "quantity": str(mkt_lot * 10), "orders": "5"},
                        {"price": str(round(a1 + 0.15, 2)), "quantity": str(mkt_lot * 15), "orders": "7"},
                    ],
                },
            }

    chain_resp = {
        "status": "OK",
        "lot_size": mkt_lot,
        "underlying_prices": {"reference_price": spot, "futures_price": spot + 25.0, "spot_price": spot},
        "strikes": strikes_raw,
    }
    return chain_resp, quotes_by_tok


class _RealisticClientStub:
    def __init__(self, quotes_by_tok: dict[str, dict[str, Any]]) -> None:
        self.quotes_by_tok = quotes_by_tok
        self.quotes_batches: list[list[dict[str, str]]] = []
        self.hist_calls: list[dict[str, str]] = []

    def quotes(self, instrument_tokens: list[dict[str, str]], quote_type: str = "all") -> list[dict[str, Any]]:
        assert quote_type == "all"
        validate_quotes_batch(instrument_tokens)
        self.quotes_batches.append(list(instrument_tokens))
        out: list[dict[str, Any]] = []
        for item in instrument_tokens:
            tok = item["instrument_token"]
            if tok == "Nifty 50" and item["exchange_segment"] == "nse_cm":
                out.append({"exchange": "nse_cm", "exchange_token": "Nifty 50", "ltp": "25012.35"})
            elif tok in self.quotes_by_tok:
                out.append(self.quotes_by_tok[tok])
        return out

    def historical_data(
        self, neosymbol: str, interval: str, from_date: str, to_date: str
    ) -> Any:
        # Enforce real request_safety validation on every historical_data call!
        validate_history(neosymbol, interval, from_date, to_date)
        self.hist_calls.append(
            {"neosymbol": neosymbol, "interval": interval, "from_date": from_date, "to_date": to_date}
        )
        start_d = dt.date.fromisoformat(from_date)
        end_d = dt.date.fromisoformat(to_date)

        if interval == "D":
            # Dict-wrapped daily shape (omitting close_time)
            rows = []
            cur = start_d
            px = 24600.0
            while cur <= end_d:
                if cur.weekday() < 5:
                    rows.append(
                        {
                            "time": f"{cur.isoformat()}T09:15:00+05:30",
                            "open": px,
                            "high": px + 95.0,
                            "low": px - 70.0,
                            "close": px + 25.0,
                            "volume": 250000,
                        }
                    )
                    px += 6.0
                cur += dt.timedelta(days=1)
            return {"status": "success", "data": rows}

        if interval == "60min":
            # Bare-list intraday shape including a trailing 15:15-15:30 stub that must be ignored
            bars: list[dict[str, Any]] = []
            cur = start_d
            while cur <= end_d:
                if cur.weekday() < 5:
                    for h in (9, 10, 11, 12, 13, 14):
                        bars.append(
                            {
                                "datetime": f"{cur.isoformat()}T{h:02d}:15:00+05:30",
                                "open": 24980.0,
                                "high": 25040.0,
                                "low": 24950.0,
                                "close": 25010.0,
                                "volume": 40000,
                            }
                        )
                    # Trailing 15:15 stub emitted by Kotak 60min feed
                    bars.append(
                        {
                            "datetime": f"{cur.isoformat()}T15:15:00+05:30",
                            "open": 25010.0,
                            "high": 25025.0,
                            "low": 25000.0,
                            "close": 25015.0,
                            "volume": 12000,
                        }
                    )
                cur += dt.timedelta(days=1)
            return bars

        # 15min bare-list shape with full-session 15m bars (adapter must filter to 15:15-15:30 closing bar)
        bars_15m: list[dict[str, Any]] = []
        cur = start_d
        while cur <= end_d:
            if cur.weekday() < 5 and cur < end_d:
                bars_15m.append(
                    {
                        "datetime": f"{cur.isoformat()}T09:15:00+05:30",
                        "open": 24950.0,
                        "high": 24990.0,
                        "low": 24940.0,
                        "close": 24975.0,
                        "volume": 15000,
                    }
                )
                bars_15m.append(
                    {
                        "datetime": f"{cur.isoformat()}T15:15:00+05:30",
                        "open": 25005.0,
                        "high": 25030.0,
                        "low": 24995.0,
                        "close": 25020.0,
                        "volume": 28000,
                    }
                )
            cur += dt.timedelta(days=1)
        return bars_15m


class _RealisticServiceStub:
    def __init__(self, chain_resp: dict[str, Any], fut_lot_size: int = 75, fut_ltp: float | None = 25020.0) -> None:
        self.chain_resp = chain_resp
        self.fut_lot_size = fut_lot_size
        self.fut_ltp = fut_ltp
        self.chain_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def get_expiries(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "OK", "expiries": ["2026-11-12", "2026-11-19"]}

    def get_futures_chain(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "status": "OK",
            "lot_size": self.fut_lot_size,
            "nearest_futures_ltp": self.fut_ltp,
        }

    def get_full_option_chain(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.chain_calls.append((args, kwargs))
        return self.chain_resp


def _make_live_adapter_with_stubs(
    service: Any,
    client: Any,
    limiter: Any | None = None,
    cache_dir: Path | None = None,
    quotes_batch_size: int = 2,
    max_depth_tokens: int = 12,
) -> LiveKotakNeoAdapter:
    adapter = LiveKotakNeoAdapter.__new__(LiveKotakNeoAdapter)
    adapter._service = service

    class _Auth:
        def require_authenticated_client(self_inner) -> tuple[Any, None, None]:
            return (client, None, None)

    adapter._auth = _Auth()
    adapter.rate_limiter = limiter or ProvisionalSerialRateLimiter(min_interval_seconds=0.0)
    adapter.enforce_schedule = True
    adapter.depth_strikes_each_side = 40
    adapter.quotes_batch_size = quotes_batch_size
    adapter.max_depth_tokens_per_expiry = max_depth_tokens
    adapter.extra_neo_symbols = ()
    adapter.cache_dir = cache_dir
    adapter.trading_calendar = {}
    adapter._cache = {}
    return adapter


def test_p1_1_targeted_two_stage_depth_funnel_and_mktlot_propagation(tmp_path: Path) -> None:
    """Verify count=40 without include_depth, 2-token quotes batches, nonzero bid<ask, None on unfetched,
    and per-contract live mktLot=65 overriding futures lot_size=75 across CandidateStructure rupee math.
    """
    chain_resp, quotes_by_tok = _build_realistic_81_strike_chain_and_quotes(spot=25000.0, mkt_lot=65)
    client = _RealisticClientStub(quotes_by_tok)
    # Intentionally set futures lot_size=75 so we prove live option-chain mktLot=65 overrides it!
    svc = _RealisticServiceStub(chain_resp, fut_lot_size=75, fut_ltp=25000.0)
    adapter = _make_live_adapter_with_stubs(svc, client, cache_dir=tmp_path)

    raw_chain = adapter.fetch_option_chain_raw("2026-11-12", DEC_MON)
    assert len(svc.chain_calls) == 1
    _, chain_kwargs = svc.chain_calls[0]
    assert "include_depth" not in chain_kwargs
    assert chain_kwargs.get("count") == 40
    assert len(raw_chain["strikes"]) == 81

    # Verified 2-token batching
    assert len(client.quotes_batches) == 6
    assert all(1 <= len(b) <= 2 for b in client.quotes_batches)

    env = parse_and_validate_option_chain(raw_chain, "2026-11-12", DEC_MON)
    assert env.lot_size == 65
    by_strike = {q.strike: q for q in env.quotes}

    # ATM 25000 has executable depth and exact qty // 65 lot conversion (20 buy lots, 25 sell lots -> conservative 20)
    atm = by_strike[25000.0]
    assert atm.has_both_executable is True
    assert atm.call_bid is not None and atm.call_ask is not None and 0 < atm.call_bid < atm.call_ask
    assert atm.call_bid_depth_lots == 20
    assert atm.call_ask_depth_lots == 25
    assert atm.call_depth_lots == 20

    # Far OTM strike (23000) was not in the 12-token shortlist -> MUST be None (never 0.0, never LTP)
    far = by_strike[23000.0]
    assert (far.call_bid, far.call_ask, far.put_bid, far.put_ask) == (None, None, None, None)
    assert far.call_depth_lots == 0

    # Run full pipeline with this adapter and verify CandidateStructure.lot_size == 65 (not 75!)
    pos_file = tmp_path / "open_positions.json"
    pos_file.write_text("[]", encoding="utf-8")
    cfg = AnalystConfig(
        mode="LIVE_MORNING_SNAPSHOT",
        fixtures_dir=str(FIXTURES_DIR),
        open_positions_file=str(pos_file),
        audit_log_path=str(tmp_path / "audit_p1_1.jsonl"),
        cache_dir=str(tmp_path / "cache"),
    )
    state, delivery = run_analyst_pipeline(DEC_MON, cfg, broker_adapter=adapter)
    assert state.contract_context.active_expiries[0].lot_size == 65
    assert len(state.candidate_structures) > 0
    assert all(c.lot_size == 65 for c in state.candidate_structures)
    # Still NOT live-ready because quote timestamps use FETCH_TIME_NOT_EXCHANGE
    assert state.final_output is not None
    assert state.final_output.is_live_ready is False
    assert delivery.alerted_trader is False


def test_p1_1_4_and_p3_4_depth_book_rejection_sorting_and_audit_logging(tmp_path: Path) -> None:
    """Verify unsorted books sort best-first; empty/locked/crossed books return None (never 0.0)
    and preserve raw payloads into audit_log.jsonl.
    """
    # 1. Unsorted 5-level book -> sorts highest bid (102.5) and lowest ask (103.0)
    unsorted_payload = {
        "exchange": "nse_fo",
        "exchange_token": "60001",
        "mktLot": "65",
        "depth": {
            "buy": [
                {"price": "101.0", "quantity": "130"},
                {"price": "102.5", "quantity": "195"},
            ],
            "sell": [
                {"price": "104.5", "quantity": "260"},
                {"price": "103.0", "quantity": "130"},
            ],
        },
    }
    ok_book = interpret_depth_book(unsorted_payload, fallback_lot_size=65)
    assert ok_book.is_executable is True
    assert (ok_book.best_bid, ok_book.best_ask) == (102.5, 103.0)
    assert (ok_book.bid_depth_lots, ok_book.ask_depth_lots) == (5, 6)

    # 2. Empty, locked, and crossed books -> fail closed to None with raw_payload preserved
    for bad_payload, expected_code in (
        ({"mktLot": "65", "depth": {"buy": [], "sell": [{"price": "10", "quantity": "65"}]}}, "EMPTY_BUY_BOOK"),
        (
            {
                "mktLot": "65",
                "depth": {
                    "buy": [{"price": "50.0", "quantity": "65"}],
                    "sell": [{"price": "50.0", "quantity": "65"}],
                },
            },
            "LOCKED_BOOK",
        ),
        (
            {
                "mktLot": "65",
                "depth": {
                    "buy": [{"price": "48.0", "quantity": "65"}, {"price": "52.0", "quantity": "65"}],
                    "sell": [{"price": "53.0", "quantity": "65"}, {"price": "51.0", "quantity": "65"}],
                },
            },
            "CROSSED_BOOK",
        ),
    ):
        rej = interpret_depth_book(bad_payload, fallback_lot_size=65)
        assert rej.is_executable is False
        assert (rej.best_bid, rej.best_ask) == (None, None)
        assert expected_code in rej.rejection_reason
        assert rej.raw_payload == bad_payload

    # 3. build_option_leg strictly rejects None or 0.0 bid/ask
    cfg_dummy = AnalystConfig()
    unpriced_q = RawStrikeQuote(
        strike=25000.0,
        call_bid=None,
        call_ask=105.0,
        put_bid=95.0,
        put_ask=96.0,
        call_oi=50000,
        put_oi=50000,
        call_volume=10000,
        put_volume=10000,
        call_depth_lots=10,
        put_depth_lots=10,
        quote_timestamp_ist=DEC_MON.isoformat(),
    )
    zero_q = dataclasses.replace(unpriced_q, call_bid=0.0)
    with pytest.raises(ValueError, match="unpriced/non-executable"):
        build_option_leg(unpriced_q, "CE", "LONG", 1, 25000.0, 0.02, 0.065, False, cfg_dummy)
    with pytest.raises(ValueError, match="unpriced/non-executable"):
        build_option_leg(zero_q, "CE", "LONG", 1, 25000.0, 0.02, 0.065, False, cfg_dummy)

    # 4. Inject a crossed wing book in a full pipeline run -> logged in audit_log.jsonl, zero 0.0 leaks
    chain_resp, quotes_by_tok = _build_realistic_81_strike_chain_and_quotes(spot=25000.0, mkt_lot=65)
    # Corrupt shortlisted wing strike 25200 CE in quotes_by_tok into a crossed book
    wing_ce_ns = next(
        s["CE"]["neo_symbol"] for s in chain_resp["strikes"] if s["strike_price"] == 25200.0
    )
    wing_tok = wing_ce_ns.split("|")[1]
    quotes_by_tok[wing_tok]["depth"] = {
        "buy": [{"price": "45.0", "quantity": "130"}],
        "sell": [{"price": "44.0", "quantity": "130"}],
    }

    client = _RealisticClientStub(quotes_by_tok)
    svc = _RealisticServiceStub(chain_resp, fut_lot_size=65, fut_ltp=25000.0)
    adapter = _make_live_adapter_with_stubs(svc, client, cache_dir=tmp_path, max_depth_tokens=20)
    pos_file = tmp_path / "open_positions.json"
    pos_file.write_text("[]", encoding="utf-8")
    audit_path = tmp_path / "audit_rejected.jsonl"
    cfg = AnalystConfig(
        mode="LIVE_MORNING_SNAPSHOT",
        fixtures_dir=str(FIXTURES_DIR),
        open_positions_file=str(pos_file),
        audit_log_path=str(audit_path),
        cache_dir=str(tmp_path / "cache"),
    )
    state, _ = run_analyst_pipeline(DEC_MON, cfg, broker_adapter=adapter)
    assert any("Rejected" in n for n in state.market_evidence.volatility_state.chain_qa_notes)
    for cand in state.candidate_structures:
        for leg in cand.legs:
            assert leg.bid > 0.0 and leg.ask > leg.bid
            assert not (leg.strike == 25200.0 and leg.option_type == "CE")

    audit_rec = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[-1])
    logged_rej = audit_rec["adapter_diagnostics"]["rejected_depth_books"]
    assert any(r["neo_symbol"] == wing_ce_ns and "CROSSED_BOOK" in r["rejection_reason"] for r in logged_rej)


def test_standing_api_rule_single_30s_quote_retry_then_flagged_and_zero_429_retry() -> None:
    """Verify transient non-429 quote failure gets 1 retry after 30s then is flagged as None,
    whereas HTTP 429 stops immediately with zero retries and zero 30s sleep.
    """
    chain_resp, quotes_by_tok = _build_realistic_81_strike_chain_and_quotes(spot=25000.0, mkt_lot=65)
    slept: list[float] = []

    class _FlakyQuotesClient(_RealisticClientStub):
        def __init__(self, qmap: dict[str, dict[str, Any]], mode: str) -> None:
            super().__init__(qmap)
            self.mode = mode
            self.attempts = 0

        def quotes(self, instrument_tokens: list[dict[str, str]], quote_type: str = "all") -> list[dict[str, Any]]:
            self.attempts += 1
            if self.mode == "transient_fail_always" and self.attempts <= 2:
                raise RuntimeError("Gateway read timeout on quote batch")
            if self.mode == "429":
                raise RuntimeError("HTTP 429 Too Many Requests")
            return super().quotes(instrument_tokens, quote_type=quote_type)

    # Case A: Non-429 quote batch fails initial + 1 retry after 30s -> flagged in rejected_depth_books, bid/ask=None
    flaky_client = _FlakyQuotesClient(quotes_by_tok, mode="transient_fail_always")
    svc = _RealisticServiceStub(chain_resp, fut_lot_size=65, fut_ltp=25000.0)
    adapter = _make_live_adapter_with_stubs(svc, flaky_client, max_depth_tokens=4)
    adapter.quote_retry_delay_seconds = 30.0
    adapter._quote_retry_sleep_fn = lambda s: slept.append(s)

    raw = adapter.fetch_option_chain_raw("2026-11-12", DEC_MON)
    assert slept == [30.0]
    assert any("QUOTE_FETCH_FAILED_AFTER_RETRY" in r["rejection_reason"] for r in raw["rejected_depth_books"])

    # Case B: HTTP 429 -> immediate hard stop, zero retries, zero 30s sleep
    slept_429: list[float] = []
    client_429 = _FlakyQuotesClient(quotes_by_tok, mode="429")
    adapter_429 = _make_live_adapter_with_stubs(svc, client_429, max_depth_tokens=4)
    adapter_429.quote_retry_delay_seconds = 30.0
    adapter_429._quote_retry_sleep_fn = lambda s: slept_429.append(s)

    with pytest.raises(KotakRateLimit429Error):
        adapter_429.fetch_option_chain_raw("2026-11-12", DEC_MON)
    assert client_429.attempts == 1
    assert slept_429 == []


def test_p1_2_candle_history_validation_chunking_both_shapes_and_same_day_1d_cache(tmp_path: Path) -> None:
    """Verify validate_history rejects broken legacy params, chunks D <= 179d, parses both shapes,
    and reuses same-day 1D cache on subsequent intraday slots within the 35s queue cap.
    """
    # Legacy broken calls raise ValueError immediately before any network call
    with pytest.raises(ValueError, match="exchange\\|instrument"):
        validate_history("Nifty 50", "D", "2026-10-01", "2026-11-09")
    with pytest.raises(ValueError, match="Invalid historical_data interval"):
        validate_history("nse_cm|Nifty 50", "1day", "2026-10-01", "2026-11-09")
    with pytest.raises(ValueError, match="exceeds maximum allowed"):
        validate_history("nse_cm|Nifty 50", "D", "2025-10-01", "2026-11-09")

    chain_resp, quotes_by_tok = _build_realistic_81_strike_chain_and_quotes()
    client = _RealisticClientStub(quotes_by_tok)
    svc = _RealisticServiceStub(chain_resp)

    fake_clock = [1000.0]
    transport = PacedReadTransport(
        min_interval_seconds=2.5,
        max_queue_wait_seconds=35.0,
        cooldown_file_path=tmp_path / "cooldown.json",
        clock_fn=lambda: fake_clock[0],
        sleep_fn=lambda s: fake_clock.__setitem__(0, fake_clock[0] + s),
    )
    adapter = _make_live_adapter_with_stubs(svc, client, limiter=transport, cache_dir=tmp_path)

    # Cold 10:15 IST run: 3 chunks of "D" + 1 "60min" + 1 "15min" = 5 history calls
    raw_1015 = adapter.fetch_index_candles_raw(DEC_MON)
    assert [c["interval"] for c in client.hist_calls] == ["D", "D", "D", "60min", "15min"]
    assert all(c["neosymbol"] == "nse_cm|Nifty 50" for c in client.hist_calls)
    c1d, c1h, c15m, stripped = slice_completed_candles(raw_1015, DEC_MON)
    assert len(c1d) >= 250
    assert len(c1h) >= 15
    assert c15m is not None and c15m.is_final_15m_bar is True
    assert stripped >= 1

    # Warm 11:15 IST slot on same date: reuses 1D disk cache -> 0 new "D" calls!
    dec_1115 = dt.datetime(2026, 11, 9, 11, 15, 0, tzinfo=IST)
    transport_warm = PacedReadTransport(
        min_interval_seconds=2.5,
        max_queue_wait_seconds=35.0,
        clock_fn=lambda: fake_clock[0],
        sleep_fn=lambda s: fake_clock.__setitem__(0, fake_clock[0] + s),
    )
    client.hist_calls.clear()
    adapter_warm = _make_live_adapter_with_stubs(svc, client, limiter=transport_warm, cache_dir=tmp_path)
    adapter_warm.fetch_contract_master_raw(dec_1115)
    adapter_warm.fetch_index_candles_raw(dec_1115)
    adapter_warm.fetch_option_chain_raw("2026-11-12", dec_1115)

    assert [c["interval"] for c in client.hist_calls] == ["60min", "15min"]
    # Warm full slot budget: 2 (master) + 2 (intraday candles) + 1 (chain) + 6 (2-token quotes) = 11 calls (25.0s <= 35.0s!)
    assert transport_warm.request_count == 11
    assert transport_warm.cumulative_wait_seconds == 25.0


def test_paced_read_transport_35s_queue_cap_and_persisted_429_cooldown(tmp_path: Path) -> None:
    """Verify 14 serial calls (32.5s wait) pass, 15th call breaches 35s cap, and 429 cooldown persists."""
    clock = [0.0]
    wall = [1_700_000_000.0]
    cooldown_file = tmp_path / "kotak_429_cooldown.json"

    pacer = PacedReadTransport(
        min_interval_seconds=2.5,
        max_queue_wait_seconds=35.0,
        cooldown_seconds_on_429=900.0,
        cooldown_file_path=cooldown_file,
        clock_fn=lambda: clock[0],
        wall_time_fn=lambda: wall[0],
        sleep_fn=lambda s: clock.__setitem__(0, clock[0] + s),
    )
    # 14 calls = 13 intervals * 2.5s = 32.5s <= 35.0s
    for _ in range(14):
        pacer.wait_for_slot()
    assert pacer.request_count == 14
    assert pacer.cumulative_wait_seconds == 32.5

    # 16th call would push wait to 37.5s > 35.0s (15th is exact 35.0s boundary)
    pacer.wait_for_slot()
    assert pacer.cumulative_wait_seconds == 35.0
    with pytest.raises(PacedQueueCapExceededError):
        pacer.wait_for_slot()

    # Persist 429 cooldown -> blocks a brand-new transport instance until cooldown expires
    pacer.record_429_cooldown("HTTP 429 test")
    pacer_next = PacedReadTransport(
        cooldown_file_path=cooldown_file,
        clock_fn=lambda: clock[0],
        wall_time_fn=lambda: wall[0] + 120.0,  # Only 2 min elapsed of 15 min cooldown
        sleep_fn=lambda s: None,
    )
    with pytest.raises(PacedCooldownActiveError):
        pacer_next.wait_for_slot()


def test_p1_3_cli_open_positions_rule3_cap_and_fail_closed_unknown(tmp_path: Path) -> None:
    """Verify --open-positions enforces Rule 3 Rs 10,000 cap and fails closed on missing/corrupt files."""
    import os

    old_audit = os.environ.get("AUDIT_LOG_PATH")
    os.environ["AUDIT_LOG_PATH"] = str(tmp_path / "cli_audit.jsonl")
    try:
        # 1. Existing open positions summing to Rs 9,500 -> adding Rs 2,000 = Rs 11,500 > Rs 10,000 -> Rule 3 FAIL
        pos_cap = tmp_path / "positions_9500.json"
        pos_cap.write_text(
            json.dumps(
                [
                    {"position_id": "P1", "structure_type": "BULL_PUT_SPREAD", "expiry_date": "2026-11-12", "planned_stop_rs": 5000.0},
                    {"position_id": "P2", "structure_type": "BEAR_CALL_SPREAD", "expiry_date": "2026-11-12", "planned_stop_rs": 4500.0},
                ]
            ),
            encoding="utf-8",
        )
        out_cap = tmp_path / "state_cap.json"
        rc = cli_main(
            [
                "--decision-time-ist",
                "2026-11-06T10:15:00+05:30",
                "--open-positions",
                str(pos_cap),
                "--write-state-json",
                str(out_cap),
            ]
        )
        assert rc == 0
        st_cap = json.loads(out_cap.read_text(encoding="utf-8"))
        assert st_cap["portfolio_exposure_state"]["portfolio_state_status"] == "VERIFIED"
        assert st_cap["portfolio_exposure_state"]["summed_planned_stops_rs"] == 9500.0
        assert st_cap["final_output"]["verdict"] == "WAIT"
        assert st_cap["final_output"]["is_live_ready"] is False
        assert all(
            c["part_a_checks"]["RULE_3_PORTFOLIO_SUMMED_STOP_10000_RS"]["status"] == "FAIL"
            for c in st_cap["candidate_structures"]
        )

        # 2. Missing open_positions.json -> PORTFOLIO_STATE_UNKNOWN, Rule 3 == UNKNOWN, WAIT, not live-ready
        missing_pos = tmp_path / "does_not_exist.json"
        out_missing = tmp_path / "state_missing.json"
        cli_main(
            [
                "--decision-time-ist",
                "2026-11-06T10:15:00+05:30",
                "--open-positions",
                str(missing_pos),
                "--write-state-json",
                str(out_missing),
            ]
        )
        st_miss = json.loads(out_missing.read_text(encoding="utf-8"))
        assert st_miss["portfolio_exposure_state"]["portfolio_state_status"] == "PORTFOLIO_STATE_UNKNOWN"
        assert st_miss["final_output"]["verdict"] == "WAIT"
        assert st_miss["final_output"]["is_live_ready"] is False
        assert "PORTFOLIO_STATE_UNKNOWN" in st_miss["final_output"]["research_only_reason"]
        assert all(
            c["part_a_checks"]["RULE_3_PORTFOLIO_SUMMED_STOP_10000_RS"]["status"] == "UNKNOWN"
            for c in st_miss["candidate_structures"]
        )

        # 3. Malformed JSON open_positions.json -> also fails closed to PORTFOLIO_STATE_UNKNOWN
        bad_pos = tmp_path / "corrupt.json"
        bad_pos.write_text("{not valid json", encoding="utf-8")
        out_bad = tmp_path / "state_bad.json"
        cli_main(
            [
                "--decision-time-ist",
                "2026-11-06T10:15:00+05:30",
                "--open-positions",
                str(bad_pos),
                "--write-state-json",
                str(out_bad),
            ]
        )
        st_bad = json.loads(out_bad.read_text(encoding="utf-8"))
        assert st_bad["portfolio_exposure_state"]["portfolio_state_status"] == "PORTFOLIO_STATE_UNKNOWN"
        assert st_bad["final_output"]["verdict"] == "WAIT"
    finally:
        if old_audit is None:
            os.environ.pop("AUDIT_LOG_PATH", None)
        else:
            os.environ["AUDIT_LOG_PATH"] = old_audit


def test_p2_1_cash_spot_fallback_when_nearest_futures_ltp_missing() -> None:
    """Verify fetch_contract_master_raw falls back cleanly to nse_cm|Nifty 50 spot when futures LTP is None."""
    chain_resp, quotes_by_tok = _build_realistic_81_strike_chain_and_quotes()
    client = _RealisticClientStub(quotes_by_tok)
    svc = _RealisticServiceStub(chain_resp, fut_lot_size=65, fut_ltp=None)
    adapter = _make_live_adapter_with_stubs(svc, client)

    master = adapter.fetch_contract_master_raw(DEC_MON)
    assert master["spot"]["value"] == 25012.35
    assert master["spot"]["source_id"] == "kotak_neo_index_spot_nse_cm"


def test_p2_2_decision_driver_calendar_slots_single_stream_lock_and_live_verify(tmp_path: Path) -> None:
    """Verify calendar-aware IST decision driver, single-stream file lock, and supervised live_verify steps."""
    holidays, _ = load_nse_holiday_calendar(FIXTURES_DIR / "nse_holidays_2026.json")
    special = {"2026-11-08": ("14:20", "15:40")}

    # Regular Friday -> 5 approved slots
    fri_slots = get_approved_slots_for_date(dt.date(2026, 11, 6), holidays, special)
    assert [s.strftime("%H:%M") for s in fri_slots] == ["10:15", "11:15", "12:15", "13:15", "14:15"]

    # Tuesday NSE holiday (2026-11-24 Guru Nanak Jayanti) -> 0 slots
    assert get_approved_slots_for_date(dt.date(2026, 11, 24), holidays, special) == []
    # Saturday -> 0 slots
    assert get_approved_slots_for_date(dt.date(2026, 11, 7), holidays, special) == []
    # Muhurat Sunday -> 1 approved slot inside 14:20..15:40
    sun_slots = get_approved_slots_for_date(dt.date(2026, 11, 8), holidays, special)
    assert len(sun_slots) == 1 and sun_slots[0].strftime("%H:%M") == "14:25"
    assert next_approved_slot_ist(
        dt.datetime(2026, 11, 7, 18, 0, tzinfo=IST), holidays, special
    ) == sun_slots[0]

    # SingleInstanceFileLock blocks concurrent stream
    lock_file = tmp_path / "kotak_stream.lock"
    with SingleInstanceFileLock(lock_file):
        with pytest.raises(ConcurrentBrokerStreamError):
            with SingleInstanceFileLock(lock_file):
                pass

    # Scheduled slot driver skips holidays without running pipeline, executes approved Friday slot
    cfg = AnalystConfig(
        mode="OFFLINE",
        fixtures_dir=str(FIXTURES_DIR),
        audit_log_path=str(tmp_path / "driver_audit.jsonl"),
        special_sessions=special,
    )
    skip_res = run_scheduled_slot(dt.datetime(2026, 11, 24, 10, 15, tzinfo=IST), cfg, lock_path=lock_file)
    assert skip_res.executed is False
    assert "SKIPPED_NSE_HOLIDAY" in skip_res.skip_reason

    run_res = run_scheduled_slot(
        dt.datetime(2026, 11, 6, 10, 15, tzinfo=IST),
        cfg,
        lock_path=lock_file,
        suppress_trader_alerts=True,
    )
    assert run_res.executed is True
    assert run_res.delivery is not None and run_res.delivery.alerted_trader is False

    # Supervised live_verify steps (Section 7.1 - 7.4) against realistic stub adapter
    chain_resp, quotes_by_tok = _build_realistic_81_strike_chain_and_quotes(spot=25000.0, mkt_lot=65)
    client = _RealisticClientStub(quotes_by_tok)
    svc = _RealisticServiceStub(chain_resp, fut_lot_size=65, fut_ltp=25000.0)
    live_stub = _make_live_adapter_with_stubs(svc, client, cache_dir=tmp_path)

    r_chain = verify_live_chain_step(live_stub, "2026-11-12", DEC_MON, expected_strikes=[24700.0, 24900.0])
    assert r_chain.passed is True and r_chain.details["strike_count"] == 81

    # Fresh adapter cache so depth step fetches with fetch_depth=True
    live_stub_depth = _make_live_adapter_with_stubs(svc, client, cache_dir=tmp_path)
    r_depth = verify_live_depth_step(live_stub_depth, "2026-11-12", DEC_MON)
    assert r_depth.passed is True and r_depth.details["lot_size"] == 65

    r_candles = verify_live_candles_step(live_stub_depth, DEC_MON)
    assert r_candles.passed is True

    pos_ok = tmp_path / "pos_ok.json"
    pos_ok.write_text("[]", encoding="utf-8")
    cfg_live = dataclasses.replace(cfg, mode="LIVE_MORNING_SNAPSHOT", open_positions_file=str(pos_ok))
    r_dry = verify_live_dry_run_step(DEC_MON, cfg_live, broker_adapter=live_stub_depth)
    assert r_dry.passed is True
    assert r_dry.details["alerted_trader"] is False
    assert r_dry.details["is_live_ready"] is False
