"""IVP production-path regression: validated series archive -> adapter -> percentiles.

Guards production-readiness requirements (zero external /home/sandbox paths):
- current-day UDiFF refresh + trailing lookback are computed from data, never hardcoded;
- >= 252 verified sessions computes exact 252d IVP; < 252 reports UNKNOWN (no substitute);
- point-in-time clamp: a decision date in the past sees only sessions <= that date,
  respecting the 19:00 IST NSE bhavcopy publication cutoff and declared special sessions.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
from pathlib import Path

from nifty_analyst.adapters.eod_vix_events import load_eod_volatility_history
from nifty_analyst.adapters.iv_history_builder import refresh_eod_volatility_archive
from nifty_analyst.engine.pricing import black76_price
from nifty_analyst.scrubber import IST

SPECIAL = {"2026-11-08": ("14:20", "15:40")}  # Muhurat F&O session, NSE FAOP76940


def _build_262_session_archive(tmp_path: Path) -> Path:
    """Build a deterministic 262-weekday EOD archive ending 2026-11-06 plus one UDiFF refresh day."""
    end_date = dt.date(2026, 11, 5)
    weekdays: list[dt.date] = []
    cur = end_date
    while len(weekdays) < 261:
        if cur.weekday() < 5:
            weekdays.append(cur)
        cur -= dt.timedelta(days=1)
    weekdays.reverse()

    seed_sessions = []
    for idx, d in enumerate(weekdays):
        spot = round(23500.0 + 700.0 * math.sin(idx / 19.0) + idx * 2.5, 2)
        cm30 = round(0.115 + 0.035 * (0.5 + 0.5 * math.sin(idx / 11.0)), 6)
        seed_sessions.append(
            {
                "trade_date": d.isoformat(),
                "spot_close": spot,
                "cm30_atm_iv": cm30,
                "source_label": "EOD_CLOSE_INFERRED",
            }
        )

    seed_path = tmp_path / "seed_261.json"
    seed_path.write_text(json.dumps({"eod_sessions": seed_sessions}), encoding="utf-8")

    # Append 2026-11-06 via actual UDiFF CSV ingestion (with BANKNIFTY decoy row to verify filter)
    udiff_csv = tmp_path / "udiff_20261106.csv"
    spot_06 = 24180.5
    strike_atm = 24200.0
    r = 0.0675
    exp_near = "2026-11-23"
    exp_far = "2026-12-24"
    t_near = (
        dt.datetime(2026, 11, 23, 15, 30, tzinfo=IST)
        - dt.datetime(2026, 11, 6, 15, 30, tzinfo=IST)
    ).total_seconds() / (365.0 * 86400.0)
    t_far = (
        dt.datetime(2026, 12, 24, 15, 30, tzinfo=IST)
        - dt.datetime(2026, 11, 6, 15, 30, tzinfo=IST)
    ).total_seconds() / (365.0 * 86400.0)

    c_near = black76_price(spot_06, strike_atm, t_near, r, 0.141, "CE")
    p_near = black76_price(spot_06, strike_atm, t_near, r, 0.141, "PE")
    c_far = black76_price(spot_06, strike_atm, t_far, r, 0.144, "CE")
    p_far = black76_price(spot_06, strike_atm, t_far, r, 0.144, "PE")

    with udiff_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "TradDt",
                "TckrSymb",
                "FinInstrmTp",
                "XpryDt",
                "StrkPric",
                "OptnTp",
                "SttlmPric",
                "UndrlygPric",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "TradDt": "2026-11-06",
                "TckrSymb": "BANKNIFTY",
                "FinInstrmTp": "IDO",
                "XpryDt": exp_near,
                "StrkPric": "51800",
                "OptnTp": "CE",
                "SttlmPric": "420.0",
                "UndrlygPric": "51790.0",
            }
        )
        for exp, c_px, p_px in ((exp_near, c_near, p_near), (exp_far, c_far, p_far)):
            writer.writerow(
                {
                    "TradDt": "2026-11-06",
                    "TckrSymb": "NIFTY",
                    "FinInstrmTp": "IDO",
                    "XpryDt": exp,
                    "StrkPric": f"{strike_atm:.1f}",
                    "OptnTp": "CE",
                    "SttlmPric": f"{c_px:.4f}",
                    "UndrlygPric": f"{spot_06:.2f}",
                }
            )
            writer.writerow(
                {
                    "TradDt": "2026-11-06",
                    "TckrSymb": "NIFTY",
                    "FinInstrmTp": "IDO",
                    "XpryDt": exp,
                    "StrkPric": f"{strike_atm:.1f}",
                    "OptnTp": "PE",
                    "SttlmPric": f"{p_px:.4f}",
                    "UndrlygPric": f"{spot_06:.2f}",
                }
            )

    out = tmp_path / "archive.json"
    return refresh_eod_volatility_archive(
        archive_path=seed_path,
        out_path=out,
        udiff_csv_paths=[udiff_csv],
        max_trade_date="2026-11-06",
    )


def test_percentiles_from_validated_series_no_hardcoding(tmp_path: Path) -> None:
    archive = _build_262_session_archive(tmp_path)
    summary = load_eod_volatility_history(
        eod_archive_path=archive,
        decision_dt_ist=dt.datetime(2026, 11, 6, 19, 30, tzinfo=IST),
        current_cm30_iv=None,
        holidays=set(),
    )
    assert summary.sanity_passed
    assert len(summary.records) == 262
    ivs = [r.constant_maturity_30d_iv for r in summary.records]
    cur = ivs[-1]
    expected_30 = round(100.0 * sum(1 for v in ivs[-30:] if v <= cur) / 30.0, 2)
    expected_90 = round(100.0 * sum(1 for v in ivs[-90:] if v <= cur) / 90.0, 2)
    expected_252 = round(100.0 * sum(1 for v in ivs[-252:] if v <= cur) / 252.0, 2)
    assert summary.ivp_30d == expected_30
    assert summary.ivp_90d == expected_90
    assert isinstance(summary.ivp_252d, float) and 0.0 <= summary.ivp_252d <= 100.0
    assert summary.ivp_252d == expected_252
    assert summary.records[-1].trade_date == "2026-11-06"
    assert summary.records[-1].source_label == "EOD_CLOSE_INFERRED"


def test_point_in_time_clamp_excludes_future_sessions(tmp_path: Path) -> None:
    archive = _build_262_session_archive(tmp_path)
    summary = load_eod_volatility_history(
        eod_archive_path=archive,
        decision_dt_ist=dt.datetime(2026, 9, 1, 15, 30, tzinfo=IST),
        current_cm30_iv=0.13,
        holidays=set(),
    )
    assert summary.records, "expected records clamped to <= 2026-09-01"
    assert all(r.trade_date <= "2026-09-01" for r in summary.records)
    assert len(summary.records) < 252
    assert summary.ivp_252d == "UNKNOWN"


def test_missing_archive_yields_unknown_not_crash(tmp_path: Path) -> None:
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


def test_intraday_run_uses_latest_completed_eod_not_live_iv(tmp_path: Path) -> None:
    archive = _build_262_session_archive(tmp_path)
    summary = load_eod_volatility_history(
        eod_archive_path=archive,
        decision_dt_ist=dt.datetime(2026, 11, 9, 10, 15, tzinfo=IST),
        current_cm30_iv=None,
        holidays=set(),
    )
    assert summary.sanity_passed
    assert summary.records[-1].trade_date == "2026-11-06"
    ivs = [r.constant_maturity_30d_iv for r in summary.records]
    expected252 = round(100.0 * sum(1 for v in ivs[-252:] if v <= ivs[-1]) / 252.0, 2)
    assert summary.ivp_252d == expected252


def test_stale_archive_yields_unknown_not_silent_carry(tmp_path: Path) -> None:
    archive = _build_262_session_archive(tmp_path)
    data = json.loads(archive.read_text(encoding="utf-8"))
    data["eod_sessions"] = [r for r in data["eod_sessions"] if r["trade_date"] <= "2026-11-04"]
    stale_path = tmp_path / "stale.json"
    stale_path.write_text(json.dumps(data), encoding="utf-8")
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


def test_pre_publication_same_day_session_excluded(tmp_path: Path) -> None:
    archive = _build_262_session_archive(tmp_path)
    summary = load_eod_volatility_history(
        eod_archive_path=archive,
        decision_dt_ist=dt.datetime(2026, 11, 6, 15, 30, tzinfo=IST),
        current_cm30_iv=None,
        holidays=set(),
    )
    assert summary.records[-1].trade_date == "2026-11-05"


def _archive_through(
    tmp_path: Path, last_date: str, extra_spot: float = 24250.0, extra_iv: float = 0.1401
) -> Path:
    archive = _build_262_session_archive(tmp_path)
    data = json.loads(archive.read_text(encoding="utf-8"))
    data["eod_sessions"] = [r for r in data["eod_sessions"] if r["trade_date"] <= last_date]
    if last_date > "2026-11-06":
        data["eod_sessions"].append(
            {
                "trade_date": last_date,
                "spot_close": extra_spot,
                "cm30_atm_iv": extra_iv,
                "near_expiry": "2026-12-08",
            }
        )
    out = tmp_path / f"arch_{last_date}.json"
    out.write_text(json.dumps(data), encoding="utf-8")
    return out


def test_muhurat_sunday_is_last_completed_session_on_monday(tmp_path: Path) -> None:
    archive = _archive_through(tmp_path, "2026-11-08")
    summary = load_eod_volatility_history(
        eod_archive_path=archive,
        decision_dt_ist=dt.datetime(2026, 11, 9, 10, 15, tzinfo=IST),
        current_cm30_iv=None,
        holidays=set(),
        special_sessions=SPECIAL,
    )
    assert summary.sanity_passed
    assert summary.records[-1].trade_date == "2026-11-08"
    assert summary.ivp_30d != "UNKNOWN" and summary.ivp_90d != "UNKNOWN"


def test_muhurat_sunday_missing_from_archive_is_stale_on_monday(tmp_path: Path) -> None:
    archive = _archive_through(tmp_path, "2026-11-06")
    summary = load_eod_volatility_history(
        eod_archive_path=archive,
        decision_dt_ist=dt.datetime(2026, 11, 9, 10, 15, tzinfo=IST),
        current_cm30_iv=None,
        holidays=set(),
        special_sessions=SPECIAL,
    )
    assert summary.ivp_30d == "UNKNOWN" and summary.ivp_90d == "UNKNOWN"
    assert any("STALE" in g for g in summary.continuity_gaps_flagged)


def test_sunday_evening_before_publication_no_false_outage(tmp_path: Path) -> None:
    archive = _archive_through(tmp_path, "2026-11-06")
    summary = load_eod_volatility_history(
        eod_archive_path=archive,
        decision_dt_ist=dt.datetime(2026, 11, 8, 18, 45, tzinfo=IST),
        current_cm30_iv=None,
        holidays=set(),
        special_sessions=SPECIAL,
    )
    assert summary.sanity_passed
    assert summary.records[-1].trade_date == "2026-11-06"


def test_sunday_evening_after_publication_missing_sunday_is_stale(tmp_path: Path) -> None:
    archive = _archive_through(tmp_path, "2026-11-06")
    summary = load_eod_volatility_history(
        eod_archive_path=archive,
        decision_dt_ist=dt.datetime(2026, 11, 8, 21, 30, tzinfo=IST),
        current_cm30_iv=None,
        holidays=set(),
        special_sessions=SPECIAL,
    )
    assert summary.ivp_30d == "UNKNOWN"
    assert any("STALE" in g for g in summary.continuity_gaps_flagged)
