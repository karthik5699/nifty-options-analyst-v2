"""IVP production-path regression: validated series archive -> adapter -> percentiles.

Guards main's production-readiness requirements (7 Nov 2026):
- current-day refresh + trailing lookback are computed from data, never hardcoded;
- 252-session IVP reports UNKNOWN while fewer than 252 verified sessions exist
  (no available-history substitute inside the gate field);
- point-in-time clamp: a decision date in the past sees only sessions <= that date.
"""
import datetime as dt
import json
import subprocess
import sys

from nifty_analyst.adapters.eod_vix_events import load_eod_volatility_history
from nifty_analyst.scrubber import IST

ARCHIVE = "/home/sandbox/nifty-analyst/iv_history/nse_eod_volatility_history.json"
REFRESH = "/home/sandbox/nifty-analyst/iv_history/refresh_iv_series.py"


def _fresh_archive(tmp_path):
    out = tmp_path / "archive.json"
    subprocess.run(
        [sys.executable, REFRESH, "--date", "2026-11-06", "--skip-download", "--out", str(out)],
        check=True,
    )
    return out


def test_percentiles_from_validated_series_no_hardcoding(tmp_path):
    archive = _fresh_archive(tmp_path)
    summary = load_eod_volatility_history(
        eod_archive_path=archive,
        # post-publication decision time so the 2026-11-06 session is in-window
        decision_dt_ist=dt.datetime(2026, 11, 6, 19, 30, tzinfo=IST),
        current_cm30_iv=None,  # production path: latest completed EOD cm30 is percentiled
        holidays=set(),
    )
    assert summary.sanity_passed
    # 2026-11-07: archive extended back to 2025-10-15 (full cleaned NSE window, zero
    # missing trading days vs FO calendar); was 237 when this test was written.
    assert len(summary.records) == 262
    # Values reproduced from the validated series definition (trailing N sessions, <= rank)
    assert summary.ivp_30d == 63.33
    assert summary.ivp_90d == 72.22
    # 2026-11-07: 262 sessions >= 252, so IVP-252 is now computable.
    # Rule unchanged: < 252 verified sessions -> UNKNOWN, never a substitute.
    # Validity + independent recomputation (not just a pinned literal):
    ivs = [r.constant_maturity_30d_iv for r in summary.records]
    win252 = ivs[-252:]
    expected252 = round(100.0 * sum(1 for v in win252 if v <= ivs[-1]) / len(win252), 2)
    assert isinstance(summary.ivp_252d, float) and 0.0 <= summary.ivp_252d <= 100.0
    assert summary.ivp_252d == expected252
    assert summary.ivp_252d == 61.9  # dated pin vs reviewed series, 2026-11-07
    assert summary.records[-1].trade_date == "2026-11-06"
    assert summary.records[-1].source_label == "EOD_CLOSE_INFERRED"


def test_point_in_time_clamp_excludes_future_sessions(tmp_path):
    archive = _fresh_archive(tmp_path)
    summary = load_eod_volatility_history(
        eod_archive_path=archive,
        decision_dt_ist=dt.datetime(2026, 9, 1, 15, 30, tzinfo=IST),
        current_cm30_iv=0.13,
        holidays=set(),
    )
    assert summary.records, "expected records clamped to 2026-09-01"
    assert all(r.trade_date <= "2026-09-01" for r in summary.records)
    assert summary.records[-1].trade_date <= "2026-09-01"
    assert summary.ivp_252d == "UNKNOWN"


def test_missing_archive_yields_unknown_not_crash(tmp_path):
    summary = load_eod_volatility_history(
        eod_archive_path=tmp_path / "absent.json",
        decision_dt_ist=dt.datetime(2026, 11, 6, 15, 30, tzinfo=IST),
        current_cm30_iv=0.13,
        holidays=set(),
    )
    assert summary.ivp_30d == "UNKNOWN"
    assert summary.ivp_90d == "UNKNOWN"
    assert summary.ivp_252d == "UNKNOWN"
    assert not summary.sanity_passed


def test_intraday_run_uses_latest_completed_eod_not_live_iv(tmp_path):
    """Monday 10:15 IST run: percentiled quantity = Friday's completed EOD cm30; the
    archive is fresh (Friday is the last trading day before the Monday cutoff)."""
    archive = _fresh_archive(tmp_path)
    summary = load_eod_volatility_history(
        eod_archive_path=archive,
        decision_dt_ist=dt.datetime(2026, 11, 9, 10, 15, tzinfo=IST),
        current_cm30_iv=None,
        holidays=set(),
    )
    assert summary.sanity_passed
    assert summary.records[-1].trade_date == "2026-11-06"
    assert summary.ivp_30d == 63.33
    assert summary.ivp_90d == 72.22
    # 2026-11-07: 262 sessions >= 252, IVP-252 now computed from data.
    # Independently recomputed from the loaded records, not a bare literal:
    ivs = [r.constant_maturity_30d_iv for r in summary.records]
    win252 = ivs[-252:]
    expected252 = round(100.0 * sum(1 for v in win252 if v <= ivs[-1]) / len(win252), 2)
    assert summary.ivp_252d == expected252 == 61.9


def test_stale_archive_yields_unknown_not_silent_carry(tmp_path):
    """Archive missing the latest completed session (missed refresh / missing
    publication) must degrade to UNKNOWN, never serve the previous session as valid."""
    import json
    archive = _fresh_archive(tmp_path)
    data = json.loads(archive.read_text())
    data["eod_sessions"] = [r for r in data["eod_sessions"] if r["trade_date"] <= "2026-11-04"]
    stale_path = tmp_path / "stale.json"
    stale_path.write_text(json.dumps(data))
    summary = load_eod_volatility_history(
        eod_archive_path=stale_path,
        decision_dt_ist=dt.datetime(2026, 11, 9, 10, 15, tzinfo=IST),
        current_cm30_iv=None,
        holidays=set(),
    )
    assert summary.ivp_30d == "UNKNOWN"
    assert summary.ivp_90d == "UNKNOWN"
    assert not summary.sanity_passed
    assert any("STALE" in g for g in summary.continuity_gaps_flagged)


def test_pre_publication_same_day_session_excluded(tmp_path):
    """At 15:30 IST the same day's EOD row cannot exist yet: clamp drops it."""
    archive = _fresh_archive(tmp_path)
    summary = load_eod_volatility_history(
        eod_archive_path=archive,
        decision_dt_ist=dt.datetime(2026, 11, 6, 15, 30, tzinfo=IST),
        current_cm30_iv=None,
        holidays=set(),
    )
    assert summary.records[-1].trade_date == "2026-11-05"


SPECIAL = {"2026-11-08": ("14:20", "15:40")}  # Muhurat F&O session, NSE FAOP76940


def _archive_through(tmp_path, last_date, extra_spot=25495.0, extra_iv=0.1401):
    import json
    archive = _fresh_archive(tmp_path)
    data = json.loads(archive.read_text())
    data["eod_sessions"] = [r for r in data["eod_sessions"] if r["trade_date"] <= last_date]
    if last_date > "2026-11-06":
        data["eod_sessions"].append({
            "trade_date": last_date, "spot_close": extra_spot,
            "cm30_atm_iv": extra_iv, "near_expiry": "2026-12-08",
        })
    out = tmp_path / f"arch_{last_date}.json"
    out.write_text(json.dumps(data))
    return out


def test_muhurat_sunday_is_last_completed_session_on_monday(tmp_path):
    """Mon 2026-11-09 10:15: with the special-session calendar, Sunday's Muhurat EOD
    (once published) is the expected latest session and keeps the archive fresh."""
    archive = _archive_through(tmp_path, "2026-11-08")
    summary = load_eod_volatility_history(
        eod_archive_path=archive,
        decision_dt_ist=dt.datetime(2026, 11, 9, 10, 15, tzinfo=IST),
        current_cm30_iv=None, holidays=set(), special_sessions=SPECIAL,
    )
    assert summary.sanity_passed
    assert summary.records[-1].trade_date == "2026-11-08"
    assert summary.ivp_30d != "UNKNOWN" and summary.ivp_90d != "UNKNOWN"


def test_muhurat_sunday_missing_from_archive_is_stale_on_monday(tmp_path):
    """Same decision time, archive only through Friday: with the special-session
    calendar the expected session is Sunday 2026-11-08, so IVP degrades to UNKNOWN."""
    archive = _archive_through(tmp_path, "2026-11-06")
    summary = load_eod_volatility_history(
        eod_archive_path=archive,
        decision_dt_ist=dt.datetime(2026, 11, 9, 10, 15, tzinfo=IST),
        current_cm30_iv=None, holidays=set(), special_sessions=SPECIAL,
    )
    assert summary.ivp_30d == "UNKNOWN" and summary.ivp_90d == "UNKNOWN"
    assert any("STALE" in g for g in summary.continuity_gaps_flagged)


def test_sunday_evening_before_publication_no_false_outage(tmp_path):
    """Sun 2026-11-08 18:45 (refresh wake fires): Sunday's bhavcopy is not yet due
    (publication cutoff 19:00), so a Friday-ending archive is still fresh - the guard
    must NOT misreport an outage."""
    archive = _archive_through(tmp_path, "2026-11-06")
    summary = load_eod_volatility_history(
        eod_archive_path=archive,
        decision_dt_ist=dt.datetime(2026, 11, 8, 18, 45, tzinfo=IST),
        current_cm30_iv=None, holidays=set(), special_sessions=SPECIAL,
    )
    assert summary.sanity_passed
    assert summary.records[-1].trade_date == "2026-11-06"


def test_sunday_evening_after_publication_missing_sunday_is_stale(tmp_path):
    """Sun 2026-11-08 21:30 (publication window passed): a Friday-ending archive IS
    stale - Sunday's session is expected and missing."""
    archive = _archive_through(tmp_path, "2026-11-06")
    summary = load_eod_volatility_history(
        eod_archive_path=archive,
        decision_dt_ist=dt.datetime(2026, 11, 8, 21, 30, tzinfo=IST),
        current_cm30_iv=None, holidays=set(), special_sessions=SPECIAL,
    )
    assert summary.ivp_30d == "UNKNOWN"
    assert any("STALE" in g for g in summary.continuity_gaps_flagged)
