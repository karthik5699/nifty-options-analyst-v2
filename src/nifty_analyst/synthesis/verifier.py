"""Deterministic Two-Way Post-LLM Citation, Prose Numeric & Part A Verifier (Sections 2.1, 7, 8.4, 11).

1. Forward check: Resolves every `state_field_path` cited by the LLM against `NiftyOptionsState`.
2. Reverse prose check: Scans LLM prose (reasoning, Bull/Bear/Vol debate, Risk Triad, Consolidator)
   for invented structure IDs, ungrounded `Rs <amount>` figures, and confabulated NIFTY index levels.
3. Consolidator & Part A re-verification: Confirms any recommended structure exists in
   `state.candidate_structures` and holds `part_a_overall_status == 'PASS'`.
"""

from __future__ import annotations

import math
import re
from typing import Any, Sequence

from nifty_analyst.state import (
    CandidateStructure,
    LLMClaimCitation,
    NiftyOptionsState,
    VerifierReport,
)

_INDEX_STEP_RE = re.compile(r"^([a-zA-Z0-9_]+)(?:\[(\d+)\])?$")
_STRUCT_ID_RE = re.compile(
    r"\b(?:BPS|BCS|BULL_BFLY|BEAR_CS|BEAR_PS|BEAR_BFLY|IC|IBFLY|LEG_IC_(?:PE|CE))_\d{4}-\d{2}-\d{2}_[0-9_]+\b"
)
_RUPEE_RE = re.compile(r"\bRs\.?\s*([0-9][0-9,]*(?:\.[0-9]+)?)")
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_INDEX_LEVEL_RE = re.compile(r"(?<![0-9-])(1[5-9]\d{3}|2\d{4}|3[0-4]\d{3})(?:\.(\d+))?(?![0-9-])")

_POLICY_CONSTANTS: tuple[float, ...] = (2000.0, 10000.0, 0.0, 100.0, 150.0, 200.0)


def resolve_state_path(state_dict: dict[str, Any], path: str) -> tuple[bool, Any]:
    """Resolve a dotted/indexed path (e.g. 'candidate_structures[0].contractual_max_loss_rs') in state_dict."""
    if not path or not isinstance(state_dict, dict):
        return False, None
    current: Any = state_dict
    for part in path.strip().split("."):
        m = _INDEX_STEP_RE.match(part)
        if not m:
            return False, None
        key = m.group(1)
        idx_str = m.group(2)
        if not isinstance(current, dict) or key not in current:
            return False, None
        current = current[key]
        if idx_str is not None:
            idx = int(idx_str)
            if not isinstance(current, (list, tuple)) or idx < 0 or idx >= len(current):
                return False, None
            current = current[idx]
    return True, current


def values_match(actual: Any, cited: Any, float_tol: float = 0.05) -> bool:
    """Compare actual state value against LLM-cited value with strict numeric tolerance."""
    if isinstance(actual, bool) or isinstance(cited, bool):
        return bool(actual) is bool(cited)
    if isinstance(actual, (int, float)) and isinstance(cited, (int, float)):
        if math.isnan(actual) or math.isnan(cited):
            return False
        return abs(float(actual) - float(cited)) <= float_tol
    return str(actual).strip() == str(cited).strip()


def _collect_state_numbers(obj: Any, out: list[float]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        fval = float(obj)
        if math.isfinite(fval):
            out.append(fval)
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_state_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _collect_state_numbers(item, out)


def _scan_prose_for_hallucinations(
    state: NiftyOptionsState,
    state_dict: dict[str, Any],
    prose_texts: Sequence[str],
) -> tuple[int, list[str]]:
    """Verify structure IDs, `Rs <amount>` claims, and NIFTY index levels in LLM prose."""
    if not prose_texts:
        return 0, []

    known_numbers: list[float] = list(_POLICY_CONSTANTS)
    _collect_state_numbers(state_dict, known_numbers)
    for c in state.candidate_structures:
        known_numbers.append(0.55 * c.max_profit_rs)
        known_numbers.append(c.width_points - (10000.0 / max(1, c.lot_size)))

    valid_struct_ids = {c.structure_id for c in state.candidate_structures}
    checked_count = 0
    uncited: list[str] = []

    def _matches_known(val: float, tol: float = 1.05) -> bool:
        return any(abs(val - k) <= tol for k in known_numbers)

    for text in prose_texts:
        if not text:
            continue
        # Remove ISO dates and structure IDs before scanning standalone numbers
        for sid in _STRUCT_ID_RE.findall(text):
            checked_count += 1
            if sid not in valid_struct_ids:
                uncited.append(f"unknown_structure_id:{sid}")

        stripped = _STRUCT_ID_RE.sub(" ", text)
        stripped = _DATE_RE.sub(" ", stripped)

        for m_rs in _RUPEE_RE.finditer(stripped):
            raw_num = m_rs.group(1).replace(",", "")
            try:
                val = float(raw_num)
            except ValueError:
                continue
            checked_count += 1
            if not _matches_known(val, tol=1.05):
                uncited.append(f"Rs {raw_num}")

        # Strip matched Rs amounts so we don't double-check `Rs 20,000` as an index level
        stripped_no_rs = _RUPEE_RE.sub(" ", stripped)
        for m_lvl in _INDEX_LEVEL_RE.finditer(stripped_no_rs):
            whole, frac = m_lvl.group(1), m_lvl.group(2)
            raw_lvl = f"{whole}.{frac}" if frac else whole
            val = float(raw_lvl)
            checked_count += 1
            if not _matches_known(val, tol=1.05):
                uncited.append(f"index_level:{raw_lvl}")

    return checked_count, uncited


def verify_llm_synthesis(
    state: NiftyOptionsState,
    raw_citations: Sequence[dict[str, Any]],
    selected_primary: CandidateStructure | None,
    max_unsupported_claim_rate: float = 0.02,
    prose_texts: Sequence[str] = (),
    consolidator_recommended_id: str | None = None,
) -> tuple[list[LLMClaimCitation], VerifierReport]:
    """Deterministically verify every LLM-cited figure, prose claims, and Part A compliance."""
    state_dict = state.to_dict()
    verified_citations: list[LLMClaimCitation] = []
    unverified_count = 0
    failure_reasons: list[str] = []

    for item in raw_citations:
        claim_text = str(item.get("claim_text", "")).strip()
        field_path = str(item.get("state_field_path", "")).strip()
        cited_val = item.get("cited_value")

        found, actual_val = resolve_state_path(state_dict, field_path)
        if not found:
            unverified_count += 1
            note = f"Path {field_path!r} does not exist in NiftyOptionsState."
            failure_reasons.append(note)
            verified_citations.append(
                LLMClaimCitation(
                    claim_text=claim_text,
                    state_field_path=field_path,
                    cited_value=str(cited_val),
                    verified_in_state=False,
                    verification_note=note,
                )
            )
            continue

        matched = values_match(actual_val, cited_val)
        if not matched:
            unverified_count += 1
            note = (
                f"Value mismatch at {field_path!r}: LLM cited {cited_val!r}, "
                f"actual state has {actual_val!r}."
            )
            failure_reasons.append(note)
        else:
            note = f"Verified against {field_path!r} == {actual_val!r}."

        verified_citations.append(
            LLMClaimCitation(
                claim_text=claim_text,
                state_field_path=field_path,
                cited_value=(
                    cited_val if isinstance(cited_val, (int, float, str, bool)) else str(cited_val)
                ),
                verified_in_state=matched,
                verification_note=note,
            )
        )

    total = len(verified_citations)
    unsupported_rate = (unverified_count / float(total)) if total > 0 else 0.0

    # Reverse prose hallucination scan
    prose_checked, uncited_prose = _scan_prose_for_hallucinations(state, state_dict, prose_texts)
    if uncited_prose:
        failure_reasons.append(
            f"PROSE_NUMERIC_HALLUCINATION: ungrounded prose claims detected {uncited_prose}."
        )

    # Re-verify Consolidator recommendation & Part A compliance of primary proposal
    part_a_reverified = True
    if consolidator_recommended_id and consolidator_recommended_id != "WAIT":
        by_id = {c.structure_id: c for c in state.candidate_structures}
        rec_cand = by_id.get(consolidator_recommended_id)
        if rec_cand is None:
            part_a_reverified = False
            failure_reasons.append(
                f"Consolidator recommended non-existent structure {consolidator_recommended_id!r}."
            )
        elif rec_cand.part_a_overall_status != "PASS":
            part_a_reverified = False
            failure_reasons.append(
                f"Consolidator recommended {consolidator_recommended_id!r} whose Part A status is "
                f"{rec_cand.part_a_overall_status!r} (must be 'PASS')."
            )

    if selected_primary is not None:
        any_fail = any(
            chk.status == "FAIL" for chk in selected_primary.part_a_checks.values()
        )
        if any_fail or selected_primary.part_a_overall_status != "PASS":
            part_a_reverified = False
            failure_reasons.append(
                f"Primary structure {selected_primary.structure_id} failed Part A re-verification."
            )
        if selected_primary.contractual_max_loss_rs > 10000.0 + 1e-6:
            part_a_reverified = False
            failure_reasons.append(
                f"Primary structure contractual_max_loss_rs (Rs {selected_primary.contractual_max_loss_rs:.2f}) "
                f"exceeds Rs 10,000 ceiling."
            )
        if selected_primary.portfolio_summed_stop_after_trade_rs > 10000.0 + 1e-6:
            part_a_reverified = False
            failure_reasons.append(
                f"Portfolio summed stop after trade (Rs {selected_primary.portfolio_summed_stop_after_trade_rs:.2f}) "
                f"exceeds Rs 10,000 cap."
            )

    if unsupported_rate > 0.05:
        failure_reasons.append(
            f"LLM_CONFABULATION_HARD_STOP: unsupported claim rate {unsupported_rate * 100:.1f}% > 5.0%."
        )

    passed = (
        total > 0
        and unsupported_rate <= max_unsupported_claim_rate
        and part_a_reverified
        and len(uncited_prose) == 0
    )

    report = VerifierReport(
        total_claims_checked=total,
        unverified_claims_count=unverified_count,
        unsupported_claim_rate=round(unsupported_rate, 4),
        part_a_reverified=part_a_reverified,
        passed_verification=passed,
        failure_reasons=failure_reasons,
        prose_numbers_checked=prose_checked,
        uncited_prose_numbers=uncited_prose,
    )
    return verified_citations, report
