"""Immutable Evidence Snapshot & 180-Second Capture-to-Resume Gate (Item 2).

Guarantees:
1. Captures once and persists the exact sanitized pre-LLM evidence (`contract_context`,
   `portfolio_exposure_state`, `market_evidence`, `candidate_structures`), capture-start /
   capture-completed IST clocks, source provenance, contract metadata, candidate identities,
   and `config_hash` / `code_version`.
2. Canonicalizes deterministic JSON (`sort_keys=True, separators=(',', ':'), allow_nan=False`)
   and computes `evidence_sha256` after scrubbing all runtime secrets/credentials.
3. Enforces write-once immutability via `os.O_CREAT | os.O_EXCL` (`0o444`) on `<artifact>.snapshot.json`,
   `<artifact>.request.json`, and an independent target pin record so a caller cannot silently
   overwrite a snapshot or replace both `.snapshot.json` and the expected hash.
4. Enforces the 180-second limit strictly from **actual `capture_started_at_ist`** through resume
   validation (never from `decision_timestamp_ist` or `capture_completed_at_ist`), rejecting
   missing, naive, inconsistent, future, or expired clocks without ever resetting the clock.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nifty_analyst.scrubber import IST, SecretScrubber
from nifty_analyst.state import (
    CalendarEvent,
    CandleBar,
    CandidateStructure,
    ContractContext,
    EventCalendarEvidence,
    ExpiryMetadata,
    MarketEvidence,
    NiftyOptionsState,
    OpenPosition,
    OptionLeg,
    PartARuleCheck,
    PortfolioExposureState,
    PositioningEvidence,
    PriceStructureEvidence,
    RunMetadata,
    StatutoryChargesBreakdown,
    TimestampedDatum,
    TriggeredDetector,
    VerifiedTouch,
    VolatilityEvidence,
)

EVIDENCE_SNAPSHOT_SCHEMA_VERSION = "nifty_v2_evidence_snapshot_v1"
EVIDENCE_CODE_VERSION = "nifty-options-analyst-v2/post-ce3e9a5"


class EvidenceSnapshotError(ValueError):
    """Base fail-closed error for missing, tampered, overwritten, or expired evidence snapshots."""


class SnapshotOverwriteBlockedError(EvidenceSnapshotError):
    """Raised when any caller attempts to overwrite an existing write-once snapshot/request/pin."""


class SnapshotTamperOrMismatchError(EvidenceSnapshotError):
    """Raised when snapshot bytes, candidate identities, pin, or SHA256 fail verification."""


class EvidenceSnapshotClockError(EvidenceSnapshotError):
    """Raised when capture-start / capture-end / resume clocks are missing, naive, inconsistent, or in the future."""


class StaleEvidenceSnapshotError(EvidenceSnapshotError):
    """Raised when elapsed wall time from actual `capture_started_at_ist` exceeds 180 seconds."""


def canonical_json_bytes(obj: Any) -> bytes:
    """Deterministic canonical JSON serialization (no NaN/Infinity, sorted keys, compact separators)."""
    rendered = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return rendered.encode("utf-8")


def sha256_hex_of_obj(obj: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


def parse_strict_ist_timestamp(raw: Any, field_name: str) -> dt.datetime:
    """Parse an ISO-8601 timestamp requiring an explicit timezone offset (never assuming local/naive)."""
    if not isinstance(raw, str) or not raw.strip():
        raise EvidenceSnapshotClockError(
            f"MISSING_CAPTURE_CLOCK: required timestamp {field_name!r} is missing or empty."
        )
    try:
        parsed = dt.datetime.fromisoformat(raw.strip())
    except ValueError as exc:
        raise EvidenceSnapshotClockError(
            f"INVALID_CAPTURE_CLOCK: {field_name!r} ({raw!r}) is not valid ISO-8601."
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise EvidenceSnapshotClockError(
            f"NAIVE_CAPTURE_CLOCK: {field_name!r} ({raw!r}) lacks explicit timezone offset."
        )
    return parsed.astimezone(IST)


def validate_capture_and_resume_clocks(
    capture_started_at_ist: str,
    capture_completed_at_ist: str,
    resume_now_dt: dt.datetime,
    max_age_seconds: float = 180.0,
) -> float:
    """Validate capture-start, capture-completed, and resume clocks; return elapsed seconds from capture-start.

    Boundary policy: `0.0 <= (resume_now - capture_started_at).total_seconds() <= max_age_seconds`
    is allowed; `> max_age_seconds` or `< 0.0` (future capture clock) fails closed.
    """
    if resume_now_dt.tzinfo is None or resume_now_dt.utcoffset() is None:
        raise EvidenceSnapshotClockError(
            "NAIVE_RESUME_CLOCK: resume_now_dt must be timezone-aware."
        )
    now_ist = resume_now_dt.astimezone(IST)
    start_ist = parse_strict_ist_timestamp(capture_started_at_ist, "capture_started_at_ist")
    end_ist = parse_strict_ist_timestamp(capture_completed_at_ist, "capture_completed_at_ist")

    if end_ist < start_ist:
        raise EvidenceSnapshotClockError(
            f"INCONSISTENT_CAPTURE_CLOCK_ORDER: capture_completed_at_ist ({end_ist.isoformat()}) "
            f"< capture_started_at_ist ({start_ist.isoformat()})."
        )
    if start_ist > now_ist:
        raise EvidenceSnapshotClockError(
            f"FUTURE_CAPTURE_CLOCK: capture_started_at_ist ({start_ist.isoformat()}) is in the "
            f"future relative to resume clock ({now_ist.isoformat()})."
        )
    if end_ist > now_ist:
        raise EvidenceSnapshotClockError(
            f"FUTURE_CAPTURE_CLOCK: capture_completed_at_ist ({end_ist.isoformat()}) is in the "
            f"future relative to resume clock ({now_ist.isoformat()})."
        )

    elapsed_sec = (now_ist - start_ist).total_seconds()
    if elapsed_sec > float(max_age_seconds):
        raise StaleEvidenceSnapshotError(
            f"SNAPSHOT_RESUME_EXPIRED_180S: elapsed {elapsed_sec:.3f}s from actual "
            f"capture_started_at_ist ({start_ist.isoformat()}) exceeds max_age_seconds={float(max_age_seconds):.1f}s."
        )
    return elapsed_sec


def _candidate_identities_from_dicts(cand_dicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "structure_id": str(c["structure_id"]),
            "structure_type": str(c["structure_type"]),
            "expiry_date": str(c["expiry_date"]),
            "lot_size": int(c["lot_size"]),
            "lots": int(c["lots"]),
            "part_a_overall_status": str(c["part_a_overall_status"]),
            "execution_fill_basis": str(c.get("execution_fill_basis", "FULL_SIZE_TOUCH")),
            "full_size_fill_supported": bool(c.get("full_size_fill_supported", True)),
            "contractual_max_loss_rs": float(c["contractual_max_loss_rs"]),
        }
        for c in cand_dicts
    ]


def build_canonical_evidence_body(
    state: NiftyOptionsState,
    scrubber: SecretScrubber | None = None,
    chain_provenance: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the scrubbed deterministic pre-LLM evidence dictionary whose SHA256 seals the run.

    Post-LLM fields (`debate_state`, `risk_triad_state`, `consolidator_state`, `final_output`)
    and derived identity fields (`run_id`, `snapshot_id`, `evidence_sha256`) are excluded so
    recomputing `evidence_sha256` before and after synthesis produces the exact same digest.
    """
    scrubber = scrubber or SecretScrubber()
    raw = scrubber.scrub_obj(state.to_dict())
    meta = raw["run_metadata"]
    ctx = raw["contract_context"]
    port = raw["portfolio_exposure_state"]
    mev = raw["market_evidence"]
    cands = raw["candidate_structures"]

    provenance = {
        "spot_source_id": ctx["spot"]["source_id"],
        "spot_as_of_ist": ctx["spot"]["as_of_ist"],
        "spot_fetched_at_ist": ctx["spot"]["fetched_at_ist"],
        "spot_verification_tag": ctx["spot"]["verification_tag"],
        "india_vix_source_id": mev["volatility_state"]["india_vix"]["source_id"],
        "india_vix_verification_tag": mev["volatility_state"]["india_vix"]["verification_tag"],
        "iv_source_label": mev["volatility_state"]["iv_source_label"],
        "event_calendar_tag": mev["event_calendar"]["source_verification_tag"],
        "portfolio_state_status": port["portfolio_state_status"],
        "chains": scrubber.scrub_obj(chain_provenance) if chain_provenance is not None else [],
    }

    return {
        "schema_version": EVIDENCE_SNAPSHOT_SCHEMA_VERSION,
        "code_version": EVIDENCE_CODE_VERSION,
        "config_hash": str(meta["config_hash"]),
        "mode": str(meta["mode"]),
        "decision_timestamp_ist": str(meta["decision_timestamp_ist"]),
        "data_version_hash": str(meta["data_version_hash"]),
        "scheduled_slot_approved": bool(meta["scheduled_slot_approved"]),
        "model_ids": list(meta["model_ids"]),
        "capture_started_at_ist": str(meta.get("capture_started_at_ist") or ""),
        "capture_completed_at_ist": str(meta.get("capture_completed_at_ist") or ""),
        "surface_coverage_status": str(meta.get("surface_coverage_status") or "MULTI_EXPIRY_FULL"),
        "uncovered_held_neo_symbols": list(meta.get("uncovered_held_neo_symbols") or []),
        "source_provenance": provenance,
        "candidate_identities": _candidate_identities_from_dicts(cands),
        "pre_llm_state": {
            "contract_context": ctx,
            "portfolio_exposure_state": port,
            "market_evidence": mev,
            "candidate_structures": cands,
        },
    }


def compute_state_evidence_sha256(
    state: NiftyOptionsState,
    scrubber: SecretScrubber | None = None,
    chain_provenance: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], str]:
    body = build_canonical_evidence_body(
        state=state,
        scrubber=scrubber,
        chain_provenance=chain_provenance,
    )
    return body, sha256_hex_of_obj(body)


def mint_unique_capture_identities(
    decision_dt_ist: dt.datetime,
    capture_started_dt_ist: dt.datetime,
    config_hash: str,
    evidence_sha256: str,
) -> tuple[str, str]:
    """Mint globally unique `(run_id, snapshot_id)` bound to `evidence_sha256` and a fresh nonce."""
    nonce = secrets.token_hex(4)
    dec_compact = decision_dt_ist.astimezone(IST).strftime("%Y%m%d_%H%M%S")
    cap_compact = capture_started_dt_ist.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"run_{dec_compact}_{config_hash[:6]}_{evidence_sha256[:8]}_{nonce[:4]}"
    snapshot_id = f"snap_{cap_compact}_{nonce}_{evidence_sha256[:16]}"
    return run_id, snapshot_id


def snapshot_paths_for_artifact(
    artifact_path: str | Path,
    registry_root: str | Path | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Return `(snapshot_path, request_path, target_pin_path, id_seal_dir)` for `artifact_path`."""
    art = Path(artifact_path).expanduser().resolve()
    snap_path = art.with_name(art.name + ".snapshot.json")
    req_path = art.with_name(art.name + ".request.json")
    reg_dir = (
        Path(registry_root).expanduser().resolve()
        if registry_root
        else (art.parent / ".snapshot_registry")
    )
    target_key = hashlib.sha256(str(art).encode("utf-8")).hexdigest()[:24]
    pin_path = reg_dir / "by_target" / f"{target_key}.pin.json"
    id_seal_dir = reg_dir / "by_id"
    return snap_path, req_path, pin_path, id_seal_dir


def has_existing_snapshot_for_artifact(
    artifact_path: str | Path,
    registry_root: str | Path | None = None,
) -> bool:
    snap_path, req_path, pin_path, _ = snapshot_paths_for_artifact(artifact_path, registry_root)
    return snap_path.exists() or req_path.exists() or pin_path.exists()


def write_once_json(path: Path, payload: dict[str, Any]) -> None:
    """Create `path` exclusively (`O_CREAT | O_EXCL`, mode `0o444`). Fails closed if `path` exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_bytes = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    except FileExistsError as exc:
        raise SnapshotOverwriteBlockedError(
            f"SNAPSHOT_OVERWRITE_BLOCKED: immutable file already exists at {path}; "
            "refusing to overwrite original snapshot/request/pin."
        ) from exc
    try:
        os.write(fd, raw_bytes)
        os.fsync(fd)
    finally:
        os.close(fd)


@dataclass(frozen=True)
class LoadedSnapshotBundle:
    snapshot_id: str
    run_id: str
    evidence_sha256: str
    capture_started_at_ist: str
    capture_completed_at_ist: str
    elapsed_seconds_from_capture_start: float
    pre_llm_state: NiftyOptionsState
    request_payload: dict[str, Any]
    chain_provenance: list[dict[str, Any]]


def persist_immutable_snapshot_bundle(
    *,
    artifact_path: str | Path,
    state: NiftyOptionsState,
    request_payload: dict[str, Any],
    scrubber: SecretScrubber | None = None,
    chain_provenance: list[dict[str, Any]] | None = None,
    registry_root: str | Path | None = None,
) -> tuple[Path, Path]:
    """Persist write-once `.snapshot.json`, `.request.json`, target `.pin.json`, and ID `.seal.json`."""
    scrubber = scrubber or SecretScrubber()
    meta = state.run_metadata
    if not meta.snapshot_id or not meta.evidence_sha256:
        raise EvidenceSnapshotError("Cannot persist snapshot bundle without snapshot_id and evidence_sha256.")

    # Validate capture clocks upfront so malformed/missing clocks can never be persisted
    parse_strict_ist_timestamp(meta.capture_started_at_ist, "capture_started_at_ist")
    parse_strict_ist_timestamp(meta.capture_completed_at_ist, "capture_completed_at_ist")

    canonical_body, computed_sha = compute_state_evidence_sha256(
        state=state,
        scrubber=scrubber,
        chain_provenance=chain_provenance,
    )
    if computed_sha != meta.evidence_sha256:
        raise SnapshotTamperOrMismatchError(
            f"Internal snapshot hash mismatch on persist: expected {meta.evidence_sha256}, computed {computed_sha}."
        )

    snap_path, req_path, pin_path, id_seal_dir = snapshot_paths_for_artifact(
        artifact_path, registry_root
    )
    seal_path = id_seal_dir / f"{meta.snapshot_id}.seal.json"

    snapshot_doc = {
        "snapshot_id": meta.snapshot_id,
        "run_id": meta.run_id,
        "evidence_sha256": computed_sha,
        "decision_timestamp_ist": meta.decision_timestamp_ist,
        "capture_started_at_ist": meta.capture_started_at_ist,
        "capture_completed_at_ist": meta.capture_completed_at_ist,
        "canonical_body": canonical_body,
    }
    pin_doc = {
        "snapshot_id": meta.snapshot_id,
        "run_id": meta.run_id,
        "evidence_sha256": computed_sha,
        "decision_timestamp_ist": meta.decision_timestamp_ist,
        "config_hash": meta.config_hash,
        "capture_started_at_ist": meta.capture_started_at_ist,
        "capture_completed_at_ist": meta.capture_completed_at_ist,
    }

    clean_req = scrubber.scrub_obj(request_payload)
    write_once_json(pin_path, pin_doc)
    write_once_json(seal_path, pin_doc)
    write_once_json(snap_path, snapshot_doc)
    write_once_json(req_path, clean_req)
    return snap_path, req_path


def _reconstruct_datum(d: dict[str, Any]) -> TimestampedDatum[Any]:
    return TimestampedDatum(
        value=d["value"],
        as_of_ist=str(d["as_of_ist"]),
        fetched_at_ist=str(d["fetched_at_ist"]),
        source_id=str(d["source_id"]),
        max_staleness_seconds=int(d["max_staleness_seconds"]),
        is_stale=bool(d["is_stale"]),
        verification_tag=str(d.get("verification_tag", "VERIFIED")),
    )


def _reconstruct_candle(d: dict[str, Any] | None) -> CandleBar | None:
    if d is None:
        return None
    return CandleBar(
        timeframe=d["timeframe"],
        open_time_ist=str(d["open_time_ist"]),
        close_time_ist=str(d["close_time_ist"]),
        open=float(d["open"]),
        high=float(d["high"]),
        low=float(d["low"]),
        close=float(d["close"]),
        volume=float(d.get("volume", 0.0)),
        is_final_15m_bar=bool(d.get("is_final_15m_bar", False)),
    )


def reconstruct_state_from_snapshot_doc(snapshot_doc: dict[str, Any]) -> NiftyOptionsState:
    """Reconstruct typed `NiftyOptionsState` directly from verified `snapshot_doc` bytes."""
    body = snapshot_doc["canonical_body"]
    pre = body["pre_llm_state"]
    ctx_d = pre["contract_context"]
    port_d = pre["portfolio_exposure_state"]
    mev_d = pre["market_evidence"]
    ps_d = mev_d["price_structure"]
    vol_d = mev_d["volatility_state"]
    pos_d = mev_d["derivatives_positioning"]
    cal_d = mev_d["event_calendar"]

    meta = RunMetadata(
        run_id=str(snapshot_doc["run_id"]),
        decision_timestamp_ist=str(body["decision_timestamp_ist"]),
        mode=body["mode"],
        data_version_hash=str(body["data_version_hash"]),
        model_ids=[str(m) for m in body["model_ids"]],
        config_hash=str(body["config_hash"]),
        scheduled_slot_approved=bool(body["scheduled_slot_approved"]),
        snapshot_id=str(snapshot_doc["snapshot_id"]),
        evidence_sha256=str(snapshot_doc["evidence_sha256"]),
        capture_started_at_ist=str(body["capture_started_at_ist"]),
        capture_completed_at_ist=str(body["capture_completed_at_ist"]),
        surface_coverage_status=body.get("surface_coverage_status", "MULTI_EXPIRY_FULL"),
        uncovered_held_neo_symbols=[str(x) for x in body.get("uncovered_held_neo_symbols", [])],
    )

    contract_ctx = ContractContext(
        symbol=str(ctx_d["symbol"]),
        spot=_reconstruct_datum(ctx_d["spot"]),
        active_expiries=[
            ExpiryMetadata(
                expiry_date=str(e["expiry_date"]),
                expiry_timestamp_ist=str(e["expiry_timestamp_ist"]),
                dte_trading_sessions=int(e["dte_trading_sessions"]),
                lot_size=int(e["lot_size"]),
                is_holiday_adjusted=bool(e["is_holiday_adjusted"]),
                holiday_note=str(e.get("holiday_note", "")),
                is_monthly=bool(e.get("is_monthly", False)),
            )
            for e in ctx_d["active_expiries"]
        ],
        lot_size_change_detected=bool(ctx_d.get("lot_size_change_detected", False)),
        calendar_discrepancy_warnings=[str(w) for w in ctx_d.get("calendar_discrepancy_warnings", [])],
    )

    portfolio = PortfolioExposureState(
        open_positions=[
            OpenPosition(
                position_id=str(p["position_id"]),
                structure_type=str(p["structure_type"]),
                directional_sign=p["directional_sign"],
                planned_stop_rs=float(p["planned_stop_rs"]),
                current_executable_spread_value_rs=float(p["current_executable_spread_value_rs"]),
                short_leg_deltas=[float(x) for x in p["short_leg_deltas"]],
                dte_trading_sessions=int(p["dte_trading_sessions"]),
                neo_symbols=[str(ns) for ns in p.get("neo_symbols", [])],
            )
            for p in port_d["open_positions"]
        ],
        summed_planned_stops_rs=float(port_d["summed_planned_stops_rs"]),
        remaining_stop_budget_rs=float(port_d["remaining_stop_budget_rs"]),
        directional_exposure_groups={
            str(k): [str(v) for v in vals]
            for k, vals in port_d["directional_exposure_groups"].items()
        },
        portfolio_state_status=port_d.get("portfolio_state_status", "VERIFIED"),
        portfolio_state_error=str(port_d.get("portfolio_state_error", "")),
    )

    price_structure = PriceStructureEvidence(
        trend_1d_state=ps_d["trend_1d_state"],
        higher_high_higher_low_repaired=bool(ps_d["higher_high_higher_low_repaired"]),
        first_pullback_in_downtrend_warning=bool(ps_d["first_pullback_in_downtrend_warning"]),
        ma20_1h=float(ps_d["ma20_1h"]),
        ema50_1h=float(ps_d["ema50_1h"]),
        last_completed_1d_bar=_reconstruct_candle(ps_d.get("last_completed_1d_bar")),
        last_completed_1h_bar=_reconstruct_candle(ps_d.get("last_completed_1h_bar")),
        final_15m_bar_completed=bool(ps_d["final_15m_bar_completed"]),
        stripped_forming_bars_count=int(ps_d["stripped_forming_bars_count"]),
        verified_level_touches=[
            VerifiedTouch(
                level_name=str(vt["level_name"]),
                level_value=float(vt["level_value"]),
                bar_timeframe=str(vt["bar_timeframe"]),
                bar_close_time_ist=str(vt["bar_close_time_ist"]),
                bar_low=float(vt["bar_low"]),
                bar_high=float(vt["bar_high"]),
                bar_close=float(vt["bar_close"]),
                touched_geometrically=bool(vt["touched_geometrically"]),
                wick_rejection_direction=vt["wick_rejection_direction"],
            )
            for vt in ps_d["verified_level_touches"]
        ],
        rsi_14_1h=float(ps_d.get("rsi_14_1h", 50.0)),
        atr_14_1h=float(ps_d.get("atr_14_1h", 0.0)),
        atr_14_1d=float(ps_d.get("atr_14_1d", 0.0)),
        stretch_zscore_1h=float(ps_d.get("stretch_zscore_1h", 0.0)),
    )

    vol = VolatilityEvidence(
        india_vix=_reconstruct_datum(vol_d["india_vix"]),
        atm_iv_by_expiry={str(k): float(v) for k, v in vol_d["atm_iv_by_expiry"].items()},
        constant_maturity_30d_iv=float(vol_d["constant_maturity_30d_iv"]),
        eod_cm30_latest=(
            float(vol_d["eod_cm30_latest"])
            if isinstance(vol_d["eod_cm30_latest"], (int, float))
            else str(vol_d["eod_cm30_latest"])
        ),
        iv_source_label=vol_d["iv_source_label"],
        ivp_30d=float(vol_d["ivp_30d"]) if isinstance(vol_d["ivp_30d"], (int, float)) else "UNKNOWN",
        ivp_90d=float(vol_d["ivp_90d"]) if isinstance(vol_d["ivp_90d"], (int, float)) else "UNKNOWN",
        ivp_252d=float(vol_d["ivp_252d"]) if isinstance(vol_d["ivp_252d"], (int, float)) else "UNKNOWN",
        iv_history_sessions_count=int(vol_d["iv_history_sessions_count"]),
        iv_history_gaps_flagged=[str(g) for g in vol_d["iv_history_gaps_flagged"]],
        realized_vol_20d=float(vol_d["realized_vol_20d"]),
        iv_rv_spread=float(vol_d["iv_rv_spread"]),
        skew_25d_put_minus_call=float(vol_d["skew_25d_put_minus_call"]),
        wing_curvature=float(vol_d["wing_curvature"]),
        chain_qa_passed=bool(vol_d["chain_qa_passed"]),
        chain_qa_notes=[str(n) for n in vol_d["chain_qa_notes"]],
        term_structure_slope_near_minus_far=float(vol_d.get("term_structure_slope_near_minus_far", 0.0)),
        term_structure_regime=vol_d.get("term_structure_regime", "SINGLE_EXPIRY"),
    )

    pos = PositioningEvidence(
        target_expiry=str(pos_d["target_expiry"]),
        pcr_oi=float(pos_d["pcr_oi"]),
        pcr_volume=float(pos_d["pcr_volume"]),
        call_oi_wall_strike=float(pos_d["call_oi_wall_strike"]),
        call_oi_wall_contracts=int(pos_d["call_oi_wall_contracts"]),
        put_oi_wall_strike=float(pos_d["put_oi_wall_strike"]),
        put_oi_wall_contracts=int(pos_d["put_oi_wall_contracts"]),
        max_pain_strike=float(pos_d["max_pain_strike"]),
        intraday_spot_change_pts=(
            float(pos_d["intraday_spot_change_pts"])
            if pos_d.get("intraday_spot_change_pts") is not None
            else None
        ),
        intraday_atm_iv_change_pts=(
            float(pos_d["intraday_atm_iv_change_pts"])
            if pos_d.get("intraday_atm_iv_change_pts") is not None
            else None
        ),
        intraday_pcr_oi_change=(
            float(pos_d["intraday_pcr_oi_change"])
            if pos_d.get("intraday_pcr_oi_change") is not None
            else None
        ),
        prior_intraday_runs_count=int(pos_d.get("prior_intraday_runs_count", 0)),
    )

    ev_cal = EventCalendarEvidence(
        events_within_dte=[
            CalendarEvent(
                event_date=str(ev["event_date"]),
                event_name=str(ev["event_name"]),
                category=str(ev["category"]),
                impact=ev["impact"],
            )
            for ev in cal_d["events_within_dte"]
        ],
        calendar_file_age_days=float(cal_d["calendar_file_age_days"]),
        is_calendar_stale_warning=bool(cal_d["is_calendar_stale_warning"]),
        source_verification_tag=str(cal_d.get("source_verification_tag", "UNVERIFIED")),
    )

    detectors = [
        TriggeredDetector(
            detector_id=str(d["detector_id"]),
            detector_name=str(d["detector_name"]),
            track=d["track"],
            confirmation_status=d["confirmation_status"],
            direction=d["direction"],
            invalidation_level=float(d["invalidation_level"]),
            invalidation_candle_basis=d["invalidation_candle_basis"],
            invalidation_justification=str(d["invalidation_justification"]),
            summary=str(d["summary"]),
            evidence_fields_cited=[str(f) for f in d["evidence_fields_cited"]],
        )
        for d in mev_d["triggered_detectors"]
    ]

    candidates: list[CandidateStructure] = []
    for cd in pre["candidate_structures"]:
        sc_d = cd["statutory_charges"]
        legs = [
            OptionLeg(
                strike=float(ld["strike"]),
                option_type=ld["option_type"],
                side=ld["side"],
                ratio=int(ld["ratio"]),
                bid=float(ld["bid"]),
                ask=float(ld["ask"]),
                mid=float(ld["mid"]),
                iv=float(ld["iv"]),
                iv_bid=float(ld["iv_bid"]),
                iv_ask=float(ld["iv_ask"]),
                delta=float(ld["delta"]),
                gamma=float(ld["gamma"]),
                theta_daily=float(ld["theta_daily"]),
                vega_per_vol_pt=float(ld["vega_per_vol_pt"]),
                oi=int(ld["oi"]),
                volume=int(ld["volume"]),
                top_book_depth_lots=int(ld["top_book_depth_lots"]),
                quote_timestamp_ist=str(ld["quote_timestamp_ist"]),
                is_stale=bool(ld["is_stale"]),
                liquidity_score=float(ld["liquidity_score"]),
                touch_available_qty=(
                    int(ld["touch_available_qty"])
                    if ld.get("touch_available_qty") is not None
                    else None
                ),
                visible_depth_qty=(
                    int(ld["visible_depth_qty"])
                    if ld.get("visible_depth_qty") is not None
                    else None
                ),
                book_levels=[[float(r[0]), int(r[1])] for r in ld.get("book_levels", [])],
                vwap_fill_price=(
                    float(ld["vwap_fill_price"])
                    if ld.get("vwap_fill_price") is not None
                    else None
                ),
                fill_basis=ld.get("fill_basis", "FULL_SIZE_TOUCH"),
            )
            for ld in cd["legs"]
        ]
        checks = {
            str(rk): PartARuleCheck(
                rule_id=str(rv["rule_id"]),
                rule_name=str(rv["rule_name"]),
                source_date=str(rv["source_date"]),
                status=rv["status"],
                evidence=str(rv["evidence"]),
            )
            for rk, rv in cd["part_a_checks"].items()
        }
        candidates.append(
            CandidateStructure(
                structure_id=str(cd["structure_id"]),
                structure_type=cd["structure_type"],
                direction=cd["direction"],
                expiry_date=str(cd["expiry_date"]),
                dte_trading_sessions=int(cd["dte_trading_sessions"]),
                lot_size=int(cd["lot_size"]),
                lots=int(cd["lots"]),
                width_points=float(cd["width_points"]),
                legs=legs,
                is_net_credit=bool(cd["is_net_credit"]),
                executable_net_points_per_unit=float(cd["executable_net_points_per_unit"]),
                slippage_buffer_points_per_unit=float(cd["slippage_buffer_points_per_unit"]),
                statutory_charges=StatutoryChargesBreakdown(
                    exchange_txn_charges_rs=float(sc_d["exchange_txn_charges_rs"]),
                    sebi_charges_rs=float(sc_d["sebi_charges_rs"]),
                    stamp_duty_rs=float(sc_d["stamp_duty_rs"]),
                    gst_rs=float(sc_d["gst_rs"]),
                    brokerage_rs=float(sc_d["brokerage_rs"]),
                    total_charges_rs=float(sc_d["total_charges_rs"]),
                ),
                contractual_max_loss_rs=float(cd["contractual_max_loss_rs"]),
                contractual_max_loss_method=str(cd["contractual_max_loss_method"]),
                margin_estimate_rs=float(cd["margin_estimate_rs"]),
                margin_source=cd["margin_source"],
                planned_stop_rs=float(cd["planned_stop_rs"]),
                indicative_stop_spread_value_per_unit=float(
                    cd["indicative_stop_spread_value_per_unit"]
                ),
                indicative_stop_index_level=float(cd["indicative_stop_index_level"]),
                indicative_stop_note=str(cd["indicative_stop_note"]),
                portfolio_summed_stop_after_trade_rs=float(
                    cd["portfolio_summed_stop_after_trade_rs"]
                ),
                max_profit_rs=float(cd["max_profit_rs"]),
                breakevens_gross=[float(x) for x in cd["breakevens_gross"]],
                breakevens_net=[float(x) for x in cd["breakevens_net"]],
                pop_lognormal_expiry=float(cd["pop_lognormal_expiry"]),
                pop_empirical_expiry=float(cd["pop_empirical_expiry"]),
                pop_assumptions_label=str(cd["pop_assumptions_label"]),
                net_delta=float(cd["net_delta"]),
                net_gamma=float(cd["net_gamma"]),
                net_theta_daily_rs=float(cd["net_theta_daily_rs"]),
                net_vega_rs=float(cd["net_vega_rs"]),
                short_leg_max_abs_delta=float(cd["short_leg_max_abs_delta"]),
                structure_liquidity_score=float(cd["structure_liquidity_score"]),
                overnight_gap_stress_3pct_loss_rs=float(cd["overnight_gap_stress_3pct_loss_rs"]),
                invalidation_level=float(cd["invalidation_level"]),
                invalidation_candle_basis=cd["invalidation_candle_basis"],
                invalidation_justification=str(cd["invalidation_justification"]),
                management_plan=str(cd["management_plan"]),
                part_a_checks=checks,
                part_a_overall_status=cd["part_a_overall_status"],
                matches_part_b_style=bool(cd["matches_part_b_style"]),
                technical_options_score=float(cd.get("technical_options_score", 0.0)),
                execution_fill_basis=cd.get("execution_fill_basis", "FULL_SIZE_TOUCH"),
                full_size_fill_supported=bool(cd.get("full_size_fill_supported", True)),
                full_size_executable_net_points_per_unit=(
                    float(cd["full_size_executable_net_points_per_unit"])
                    if cd.get("full_size_executable_net_points_per_unit") is not None
                    else None
                ),
                full_size_contractual_max_loss_rs=(
                    float(cd["full_size_contractual_max_loss_rs"])
                    if cd.get("full_size_contractual_max_loss_rs") is not None
                    else None
                ),
                indicative_touch_net_points_per_unit=float(
                    cd.get("indicative_touch_net_points_per_unit", 0.0)
                ),
                indicative_touch_max_loss_rs=float(cd.get("indicative_touch_max_loss_rs", 0.0)),
                price_assumption_label=str(cd.get("price_assumption_label", "")),
            )
        )

    return NiftyOptionsState(
        run_metadata=meta,
        contract_context=contract_ctx,
        portfolio_exposure_state=portfolio,
        market_evidence=MarketEvidence(
            price_structure=price_structure,
            volatility_state=vol,
            derivatives_positioning=pos,
            event_calendar=ev_cal,
            triggered_detectors=detectors,
        ),
        candidate_structures=candidates,
        debate_state=None,
        final_output=None,
    )


def load_and_verify_snapshot_for_resume(
    *,
    artifact_path: str | Path,
    resume_now_dt: dt.datetime,
    expected_decision_timestamp_ist: str | None = None,
    expected_config_hash: str | None = None,
    max_age_seconds: float = 180.0,
    registry_root: str | Path | None = None,
) -> LoadedSnapshotBundle:
    """Load, cross-verify against target pin & ID seal, recompute SHA256, and enforce 180s capture-start age."""
    snap_path, req_path, pin_path, id_seal_dir = snapshot_paths_for_artifact(
        artifact_path, registry_root
    )
    if not snap_path.is_file() or not req_path.is_file() or not pin_path.is_file():
        raise SnapshotTamperOrMismatchError(
            f"MISSING_EVIDENCE_SNAPSHOT_FILES: incomplete snapshot bundle for {artifact_path} "
            f"(snapshot={snap_path.is_file()}, request={req_path.is_file()}, pin={pin_path.is_file()})."
        )

    try:
        snap_doc = json.loads(snap_path.read_text(encoding="utf-8"))
        req_doc = json.loads(req_path.read_text(encoding="utf-8"))
        pin_doc = json.loads(pin_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SnapshotTamperOrMismatchError(
            f"CORRUPT_EVIDENCE_SNAPSHOT_JSON: failed to parse snapshot/request/pin JSON ({exc})."
        ) from exc

    if not isinstance(snap_doc, dict) or not isinstance(snap_doc.get("canonical_body"), dict):
        raise SnapshotTamperOrMismatchError("INVALID_EVIDENCE_SNAPSHOT_SCHEMA: missing canonical_body.")
    if not isinstance(req_doc, dict) or not isinstance(pin_doc, dict):
        raise SnapshotTamperOrMismatchError("INVALID_EVIDENCE_SNAPSHOT_SCHEMA: invalid request or pin object.")

    snapshot_id = str(pin_doc.get("snapshot_id") or "")
    seal_path = id_seal_dir / f"{snapshot_id}.seal.json"
    if not snapshot_id or not seal_path.is_file():
        raise SnapshotTamperOrMismatchError(
            f"MISSING_SNAPSHOT_ID_SEAL: seal record missing for snapshot_id={snapshot_id!r}."
        )
    try:
        seal_doc = json.loads(seal_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SnapshotTamperOrMismatchError(f"CORRUPT_SNAPSHOT_ID_SEAL: {exc}") from exc

    # 1. Verify pin == seal == snapshot header == request header (defeats replacing both snapshot + hash!)
    for key in (
        "snapshot_id",
        "run_id",
        "evidence_sha256",
        "decision_timestamp_ist",
        "capture_started_at_ist",
        "capture_completed_at_ist",
    ):
        pin_v = pin_doc.get(key)
        seal_v = seal_doc.get(key)
        snap_v = snap_doc.get(key)
        req_v = req_doc.get(key)
        if not pin_v or len({pin_v, seal_v, snap_v, req_v}) != 1:
            raise SnapshotTamperOrMismatchError(
                f"SNAPSHOT_PIN_MISMATCH on {key!r}: pin={pin_v!r}, seal={seal_v!r}, "
                f"snapshot={snap_v!r}, request={req_v!r}."
            )

    body = snap_doc["canonical_body"]
    if expected_decision_timestamp_ist is not None and pin_doc["decision_timestamp_ist"] != expected_decision_timestamp_ist:
        raise SnapshotTamperOrMismatchError(
            f"FOREIGN_DECISION_TIMESTAMP: snapshot bound to {pin_doc['decision_timestamp_ist']!r}, "
            f"caller passed {expected_decision_timestamp_ist!r}."
        )
    if expected_config_hash is not None and pin_doc.get("config_hash") != expected_config_hash:
        raise SnapshotTamperOrMismatchError(
            f"FOREIGN_CONFIG_HASH: snapshot bound to config_hash={pin_doc.get('config_hash')!r}, "
            f"caller config_hash={expected_config_hash!r}."
        )

    # 2. Recompute canonical SHA256 over canonical_body and verify candidate_identities match pre_llm_state
    recomputed_sha = sha256_hex_of_obj(body)
    if recomputed_sha != pin_doc["evidence_sha256"]:
        raise SnapshotTamperOrMismatchError(
            f"EVIDENCE_SHA256_MISMATCH: stored canonical_body hashed to {recomputed_sha}, "
            f"expected pinned {pin_doc['evidence_sha256']}."
        )

    pre = body.get("pre_llm_state")
    if not isinstance(pre, dict) or not isinstance(pre.get("candidate_structures"), list):
        raise SnapshotTamperOrMismatchError("INVALID_PRE_LLM_STATE_IN_SNAPSHOT.")
    derived_identities = _candidate_identities_from_dicts(pre["candidate_structures"])
    if body.get("candidate_identities") != derived_identities:
        raise SnapshotTamperOrMismatchError(
            "CANDIDATE_IDENTITY_MISMATCH: candidate_identities do not match pre_llm_state.candidate_structures."
        )

    # 3. Validate original capture clocks & enforce 180-second limit from ACTUAL capture_started_at_ist
    if (
        body.get("capture_started_at_ist") != pin_doc["capture_started_at_ist"]
        or body.get("capture_completed_at_ist") != pin_doc["capture_completed_at_ist"]
    ):
        raise SnapshotTamperOrMismatchError("CAPTURE_CLOCK_BODY_MISMATCH.")

    elapsed_sec = validate_capture_and_resume_clocks(
        capture_started_at_ist=str(pin_doc["capture_started_at_ist"]),
        capture_completed_at_ist=str(pin_doc["capture_completed_at_ist"]),
        resume_now_dt=resume_now_dt,
        max_age_seconds=max_age_seconds,
    )

    reconstructed = reconstruct_state_from_snapshot_doc(snap_doc)
    # Final round-trip canonical hash check on reconstructed typed state
    prov_chains = list((body.get("source_provenance") or {}).get("chains") or [])
    _, roundtrip_sha = compute_state_evidence_sha256(
        reconstructed,
        chain_provenance=prov_chains,
    )
    if roundtrip_sha != pin_doc["evidence_sha256"]:
        raise SnapshotTamperOrMismatchError(
            f"RECONSTRUCTED_STATE_HASH_MISMATCH: {roundtrip_sha} != {pin_doc['evidence_sha256']}."
        )

    return LoadedSnapshotBundle(
        snapshot_id=str(pin_doc["snapshot_id"]),
        run_id=str(pin_doc["run_id"]),
        evidence_sha256=str(pin_doc["evidence_sha256"]),
        capture_started_at_ist=str(pin_doc["capture_started_at_ist"]),
        capture_completed_at_ist=str(pin_doc["capture_completed_at_ist"]),
        elapsed_seconds_from_capture_start=elapsed_sec,
        pre_llm_state=reconstructed,
        request_payload=req_doc,
        chain_provenance=prov_chains,
    )
