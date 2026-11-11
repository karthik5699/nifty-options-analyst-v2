"""Same-Day Intraday Context & Duplicate-Alert Guard (Zero Outcomes / Zero P&L).

STRICT CONSTRAINTS (Sections 9 & 10):
- Reads ONLY prior run inputs and alerted structure IDs on the same IST trading date
  (`prior_dt < decision_dt_ist`) from `audit_log.jsonl`.
- Computes ZERO outcomes, ZERO P&L, ZERO settlement values, and feeds NO performance loop.
- Point-in-time safe: strictly filters `prior_dt < decision_dt_ist` so re-running the exact
  same timestamp (e.g., 5-trial determinism checks) sees identical prior intraday state.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

from nifty_analyst.scrubber import ensure_ist


@dataclass(frozen=True)
class IntradayContextSummary:
    prior_intraday_runs_count: int
    intraday_spot_change_pts: float | None
    intraday_atm_iv_change_pts: float | None
    intraday_pcr_oi_change: float | None
    alerted_structure_ids_today: frozenset[str]


def load_same_day_intraday_context(
    audit_log_path: str | Path,
    decision_dt_ist: dt.datetime,
    current_spot: float,
    current_cm30_iv: float,
    current_pcr_oi: float,
) -> IntradayContextSummary:
    """Compute intraday market deltas vs earliest same-day run and collect alerted structure IDs."""
    dec_ist = ensure_ist(decision_dt_ist)
    target_date = dec_ist.date()
    path = Path(audit_log_path)
    if not path.is_file():
        return IntradayContextSummary(
            prior_intraday_runs_count=0,
            intraday_spot_change_pts=None,
            intraday_atm_iv_change_pts=None,
            intraday_pcr_oi_change=None,
            alerted_structure_ids_today=frozenset(),
        )

    prior_records: list[tuple[dt.datetime, dict]] = []
    alerted_ids: set[str] = set()

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            ts_raw = rec.get("decision_timestamp_ist", "")
            if not ts_raw:
                continue
            rec_dt = ensure_ist(dt.datetime.fromisoformat(ts_raw))
        except Exception:
            continue

        if rec_dt.date() != target_date or rec_dt >= dec_ist:
            continue

        prior_records.append((rec_dt, rec))
        fo = rec.get("final_output") or {}
        if (
            rec.get("verdict") == "SUGGESTION"
            and rec.get("is_live_ready") is True
            and not fo.get("duplicate_alert_suppressed", False)
        ):
            primary = fo.get("primary_proposal") or {}
            sid = primary.get("structure_id")
            if sid:
                alerted_ids.add(str(sid))

    if not prior_records:
        return IntradayContextSummary(
            prior_intraday_runs_count=0,
            intraday_spot_change_pts=None,
            intraday_atm_iv_change_pts=None,
            intraday_pcr_oi_change=None,
            alerted_structure_ids_today=frozenset(),
        )

    prior_records.sort(key=lambda pair: pair[0])
    earliest = prior_records[0][1]
    ev = earliest.get("frozen_evidence_state") or {}
    spot_0 = ((ev.get("contract_context") or {}).get("spot") or {}).get("value")
    vol_0 = ((ev.get("market_evidence") or {}).get("volatility_state") or {}).get(
        "constant_maturity_30d_iv"
    )
    pcr_0 = ((ev.get("market_evidence") or {}).get("derivatives_positioning") or {}).get("pcr_oi")

    spot_chg = round(current_spot - float(spot_0), 2) if isinstance(spot_0, (int, float)) else None
    iv_chg = (
        round((current_cm30_iv - float(vol_0)) * 100.0, 2)
        if isinstance(vol_0, (int, float))
        else None
    )
    pcr_chg = round(current_pcr_oi - float(pcr_0), 4) if isinstance(pcr_0, (int, float)) else None

    return IntradayContextSummary(
        prior_intraday_runs_count=len(prior_records),
        intraday_spot_change_pts=spot_chg,
        intraday_atm_iv_change_pts=iv_chg,
        intraday_pcr_oi_change=pcr_chg,
        alerted_structure_ids_today=frozenset(alerted_ids),
    )
