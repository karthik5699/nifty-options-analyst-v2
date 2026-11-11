"""Adapter contract tests (Section 8.2 & 8.5):
- Quote timestamp extraction & rejection of missing timestamps
- 180-second staleness detection
- Holiday-shifted expiry (Monday 23 Nov 2026 ahead of Tuesday 24 Nov 2026 holiday)
- Lot-size change handling (75 -> 50)
- Completed-candle slicing including the final 15-minute bar (15:15-15:30 IST)
- Provisional rate limiter & immediate HTTP 429 stop with zero retries
- Five approved IST run times
- EOD volatility history continuity and sanity check
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from nifty_analyst.adapters.calendar_and_contracts import (
    load_nse_holiday_calendar,
    parse_contract_master,
)
from nifty_analyst.adapters.eod_vix_events import load_eod_volatility_history
from nifty_analyst.adapters.kotak_neo import (
    FixtureKotakNeoAdapter,
    KotakRateLimit429Error,
    MissingQuoteTimestampError,
    ProvisionalSerialRateLimiter,
    UnapprovedScheduleSlotError,
    parse_and_validate_option_chain,
    slice_completed_candles,
)
from nifty_analyst.scrubber import IST, is_approved_schedule_slot

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def test_contract_master_holiday_shifted_expiry_and_lot_size_change() -> None:
    """Verify Monday 23 Nov 2026 holiday-shifted expiry and lot-size change (75 -> 50) parsing."""
    holidays, descs = load_nse_holiday_calendar(FIXTURES_DIR / "nse_holidays_2026.json")
    adapter = FixtureKotakNeoAdapter(FIXTURES_DIR)
    dec_dt = dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST)

    raw_master = adapter.fetch_contract_master_raw(dec_dt)
    ctx = parse_contract_master(
        raw_master=raw_master,
        decision_dt_ist=dec_dt,
        holidays=holidays,
        holiday_descriptions=descs,
        expected_calendar_expiries=["2026-11-12", "2026-11-19", "2026-11-26"],
    )

    exp_by_date = {e.expiry_date: e for e in ctx.active_expiries}
    assert "2026-11-23" in exp_by_date
    nov_monthly = exp_by_date["2026-11-23"]
    # Monday 23 Nov 2026 is weekday 0 (Monday), holiday-shifted ahead of 24 Nov 2026
    assert dt.date.fromisoformat(nov_monthly.expiry_date).weekday() == 0
    assert nov_monthly.is_holiday_adjusted is True
    assert "23 Nov 2026" in nov_monthly.holiday_note
    assert nov_monthly.lot_size == 50
    assert exp_by_date["2026-11-12"].lot_size == 75
    assert ctx.lot_size_change_detected is True
    # Calendar discrepancy warning because 2026-11-26 was shifted to 2026-11-23 in live master
    assert len(ctx.calendar_discrepancy_warnings) == 1
    assert "2026-11-26" in ctx.calendar_discrepancy_warnings[0]


def test_quote_timestamp_extraction_and_180s_staleness_trip() -> None:
    """Verify missing quote timestamps raise MissingQuoteTimestampError and >180s trips staleness."""
    adapter = FixtureKotakNeoAdapter(FIXTURES_DIR)
    dec_dt = dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST)
    raw_chain = adapter.fetch_option_chain_raw("2026-11-12", dec_dt)

    # Fresh at 10:15:00 when quotes are at 10:14:35 (25s age <= 180s)
    env_ok = parse_and_validate_option_chain(raw_chain, "2026-11-12", dec_dt, max_staleness_seconds=180)
    assert env_ok.any_stale is False
    assert len(env_ok.stale_strikes) == 0

    # Stale at 10:18:30 (235s age > 180s)
    late_dt = dt.datetime(2026, 11, 6, 10, 18, 30, tzinfo=IST)
    env_stale = parse_and_validate_option_chain(raw_chain, "2026-11-12", late_dt, max_staleness_seconds=180)
    assert env_stale.any_stale is True
    assert len(env_stale.stale_strikes) == len(raw_chain["strikes"])

    # Missing quote timestamp must be rejected immediately
    corrupt_chain = {
        "strikes": [
            {
                "strike": 24200.0,
                "call_bid": 150.0,
                "call_ask": 152.0,
                "put_bid": 140.0,
                "put_ask": 142.0,
                "update_time_ist": "",
            }
        ]
    }
    with pytest.raises(MissingQuoteTimestampError):
        parse_and_validate_option_chain(corrupt_chain, "2026-11-12", dec_dt)


def test_completed_candle_slicing_and_final_15m_bar() -> None:
    """Verify forming bars are stripped and the final 15-minute bar (15:15-15:30 IST) is recognised."""
    adapter = FixtureKotakNeoAdapter(FIXTURES_DIR)
    dec_dt = dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST)
    raw_candles = adapter.fetch_index_candles_raw(dec_dt)

    c1d, c1h, c15m, stripped = slice_completed_candles(raw_candles, dec_dt)
    # The forming 10:15-11:15 bar on 2026-11-06 must be stripped at 10:15:00 IST
    assert stripped == 1
    assert c1h[-1].open_time_ist == "2026-11-06T09:15:00+05:30"
    assert c1h[-1].close_time_ist == "2026-11-06T10:15:00+05:30"

    # The final 15-minute closing bar (15:15-15:30 IST on 2026-11-05) is recognized and marked completed
    assert c15m is not None
    assert c15m.is_final_15m_bar is True
    assert c15m.open_time_ist == "2026-11-05T15:15:00+05:30"
    assert c15m.close_time_ist == "2026-11-05T15:30:00+05:30"


def test_provisional_rate_limiter_and_immediate_429_stop_with_zero_retry() -> None:
    """Verify provisional 2.5s rate limiter and immediate stop on HTTP 429 without retry."""
    fake_clock = [100.0]
    slept: list[float] = []

    def clock() -> float:
        return fake_clock[0]

    def sleep(sec: float) -> None:
        slept.append(sec)
        fake_clock[0] += sec

    limiter = ProvisionalSerialRateLimiter(min_interval_seconds=2.5, clock_fn=clock, sleep_fn=sleep)
    limiter.wait_for_slot()
    fake_clock[0] += 1.0  # Only 1.0s elapsed
    limiter.wait_for_slot()
    assert len(slept) == 1
    assert abs(slept[0] - 1.5) < 1e-6

    # HTTP 429 stops immediately on first call with zero retries
    adapter_429 = FixtureKotakNeoAdapter(
        FIXTURES_DIR, rate_limiter=limiter, simulate_http_429=True
    )
    calls_before = limiter.request_count
    with pytest.raises(KotakRateLimit429Error, match="HTTP 429"):
        adapter_429.fetch_contract_master_raw(dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST))
    # Exactly 1 request attempt was made (zero automatic retries!)
    assert limiter.request_count == calls_before + 1


def test_five_approved_schedule_slots() -> None:
    """Verify the five approved IST run times (10:15, 11:15, 12:15, 13:15, 14:15 IST)."""
    for hour in (10, 11, 12, 13, 14):
        ts = dt.datetime(2026, 11, 6, hour, 15, 0, tzinfo=IST)
        assert is_approved_schedule_slot(ts) is True

    # 15:15 IST is outside the approved schedule
    unapproved_ts = dt.datetime(2026, 11, 6, 15, 15, 0, tzinfo=IST)
    assert is_approved_schedule_slot(unapproved_ts) is False

    adapter_live_sched = FixtureKotakNeoAdapter(
        FIXTURES_DIR, enforce_schedule_in_live_mode=True
    )
    with pytest.raises(UnapprovedScheduleSlotError):
        adapter_live_sched.fetch_contract_master_raw(unapproved_ts)


def test_eod_volatility_history_continuity_and_inferred_label(tmp_path: Path) -> None:
    """Verify EOD volatility loader labels records EOD_CLOSE_INFERRED and flags session gaps."""
    holidays, _ = load_nse_holiday_calendar(FIXTURES_DIR / "nse_holidays_2026.json")
    dec_dt = dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST)

    summary = load_eod_volatility_history(
        FIXTURES_DIR / "nse_eod_volatility_history.json",
        decision_dt_ist=dec_dt,
        current_cm30_iv=0.145,
        holidays=holidays,
    )
    assert len(summary.records) == 95
    assert all(r.source_label == "EOD_CLOSE_INFERRED" for r in summary.records)
    assert summary.sanity_passed is True
    assert len(summary.continuity_gaps_flagged) == 0
    assert isinstance(summary.ivp_30d, float)
    assert isinstance(summary.ivp_90d, float)
    # 95 sessions < 252, so 252-day IVP must be reported as UNKNOWN (never faked)
    assert summary.ivp_252d == "UNKNOWN"
