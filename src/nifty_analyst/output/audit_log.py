"""Immutable Audit Log and Alert-Only Output Gate (Sections 9 & 10).

STRICT CONSTRAINTS:
- Appends an immutable record of run inputs and outputs only: run_id, timestamp_ist,
  data_version_hash, model_ids, frozen_evidence_state, candidates_considered, verdict,
  and full report.
- Records NO outcomes, NO P&L, NO settlement, and feeds NO performance loop.
- Alert-only output: produces user-facing report output ONLY when verdict == 'SUGGESTION'
  (unless force_render_for_inspection=True in Phase 0 offline gate review).
  'WAIT' and 'INSUFFICIENT DATA' are written quietly to the audit log with no routine message.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nifty_analyst.output.renderer import render_analyst_report
from nifty_analyst.scrubber import SecretScrubber
from nifty_analyst.state import NiftyOptionsState


@dataclass(frozen=True)
class AuditDeliveryResult:
    run_id: str
    verdict: str
    alerted_trader: bool
    user_facing_message: str
    audit_log_path: str
    record_sha256: str
    full_report_markdown: str


def append_immutable_audit_record(
    state: NiftyOptionsState,
    report_markdown: str,
    audit_log_path: str | Path,
    scrubber: SecretScrubber | None = None,
) -> tuple[dict[str, Any], str]:
    """Append an immutable JSONL record of inputs and outputs (zero outcomes/P&L)."""
    scrubber = scrubber or SecretScrubber()
    state_dict = scrubber.scrub_obj(state.to_dict())
    clean_report = scrubber.scrub_text(report_markdown)

    verdict = state.final_output.verdict if state.final_output else "INSUFFICIENT DATA"
    record_body: dict[str, Any] = {
        "run_id": state.run_metadata.run_id,
        "decision_timestamp_ist": state.run_metadata.decision_timestamp_ist,
        "mode": state.run_metadata.mode,
        "data_version_hash": state.run_metadata.data_version_hash,
        "config_hash": state.run_metadata.config_hash,
        "model_ids": state.run_metadata.model_ids,
        "verdict": verdict,
        "is_live_ready": state.final_output.is_live_ready if state.final_output else False,
        "frozen_evidence_state": {
            "contract_context": state_dict["contract_context"],
            "portfolio_exposure_state": state_dict["portfolio_exposure_state"],
            "market_evidence": state_dict["market_evidence"],
        },
        "candidates_considered": state_dict["candidate_structures"],
        "final_output": state_dict["final_output"],
        "report_markdown": clean_report,
    }

    canonical_json = json.dumps(record_body, sort_keys=True)
    record_sha256 = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    record_body["record_sha256"] = record_sha256

    path = Path(audit_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record_body, sort_keys=False) + "\n")

    return record_body, record_sha256


def deliver_run_output(
    state: NiftyOptionsState,
    audit_log_path: str | Path,
    scrubber: SecretScrubber | None = None,
    force_render_for_inspection: bool = False,
) -> AuditDeliveryResult:
    """Enforce alert-only delivery policy and write immutable audit record.

    - If verdict == 'SUGGESTION' (or force_render_for_inspection=True), returns the full report.
    - If verdict in ('WAIT', 'INSUFFICIENT DATA') and not force_render_for_inspection,
      writes to the audit log and returns an empty user_facing_message (no routine ping).
    """
    scrubber = scrubber or SecretScrubber()
    full_report = render_analyst_report(state, scrubber=scrubber)
    _, sha256_hex = append_immutable_audit_record(
        state=state,
        report_markdown=full_report,
        audit_log_path=audit_log_path,
        scrubber=scrubber,
    )

    verdict = state.final_output.verdict if state.final_output else "INSUFFICIENT DATA"
    # Alert only a live-ready SUGGESTION (IVP verified + Part A PASS); never an
    # uncertified or research-only verdict, even when it renders for inspection.
    alerted = verdict == "SUGGESTION" and bool(
        state.final_output and state.final_output.is_live_ready
    )
    if alerted or force_render_for_inspection:
        user_msg = full_report
    else:
        user_msg = ""

    return AuditDeliveryResult(
        run_id=state.run_metadata.run_id,
        verdict=verdict,
        alerted_trader=alerted,
        user_facing_message=user_msg,
        audit_log_path=str(audit_log_path),
        record_sha256=sha256_hex,
        full_report_markdown=full_report,
    )
