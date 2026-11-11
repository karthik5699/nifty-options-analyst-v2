"""Deterministic Post-LLM Citation & Part A Verifier (Sections 2.1, 7, 8.4, 11).

Resolves every state_field_path cited by the LLM against the actual NiftyOptionsState
dictionary, checks numeric/string equality, and re-verifies Part A statuses before
any output can reach the trader.
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


def verify_llm_synthesis(
    state: NiftyOptionsState,
    raw_citations: Sequence[dict[str, Any]],
    selected_primary: CandidateStructure | None,
    max_unsupported_claim_rate: float = 0.02,
) -> tuple[list[LLMClaimCitation], VerifierReport]:
    """Deterministically verify every LLM-cited figure and re-verify Part A compliance."""
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
                cited_value=cited_val if isinstance(cited_val, (int, float, str, bool)) else str(cited_val),
                verified_in_state=matched,
                verification_note=note,
            )
        )

    total = len(verified_citations)
    unsupported_rate = (unverified_count / float(total)) if total > 0 else 0.0

    # Re-verify Part A compliance of primary proposal
    part_a_reverified = True
    if selected_primary is not None:
        any_fail = any(
            chk.status == "FAIL" for chk in selected_primary.part_a_checks.values()
        )
        if any_fail or selected_primary.part_a_overall_status == "FAIL":
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
    )

    report = VerifierReport(
        total_claims_checked=total,
        unverified_claims_count=unverified_count,
        unsupported_claim_rate=round(unsupported_rate, 4),
        part_a_reverified=part_a_reverified,
        passed_verification=passed,
        failure_reasons=failure_reasons,
    )
    return verified_citations, report
