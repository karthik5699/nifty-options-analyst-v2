"""Deterministic Two-Way Post-LLM Citation, Semantic Prose, Digest & Part A Verifier.

Enforces:
1. Forward citation check: Resolves every `state_field_path` cited by the LLM against `NiftyOptionsState`.
2. Candidate-scoped & metric-meaning semantic prose check across ALL delivered prose fields
   (`reasoning_summary`, `bull_thesis`, `bear_thesis`, `vol_regime_thesis`, `rebuttal_round`,
   `convexity_advocate_note`, `tail_guardian_note`, `neutral_arbiter_note`,
   `final_recommendation_rationale`, `habitual_comparison_note`, `research_rule_challenge_note`,
   and `confidence_note`). A number borrowed from another valid candidate or swapped across
   metric meanings (e.g. margin claimed as contractual max loss, strike claimed as spot) fails!
3. Strict supplied `whatsapp_digest` validation against the exact post-selection primary candidate
   (contract/structure ID, expiry, leg strike/type/side/ratio, lot/unit quantity, candidate-scoped
   risk figures, and readiness label).
4. Consolidator, full-size depth support, & Part A re-verification.
"""

from __future__ import annotations

import dataclasses
import math
import re
from typing import Any, Mapping, Sequence

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
_FOREIGN_UNDERSCORE_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])[A-Z][A-Z0-9]*_[A-Z0-9_-]+(?![A-Za-z0-9_-])"
)
_RUPEE_RE = re.compile(r"\bRs\.?\s*([0-9][0-9,]*(?:\.[0-9]+)?)")
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_INDEX_LEVEL_RE = re.compile(r"(?<![0-9-])(1[5-9]\d{3}|2\d{4}|3[0-4]\d{3})(?:\.(\d+))?(?![0-9-])")
_COMPACT_LEG_RE = re.compile(
    r"(?:(?P<side>S|B|SHORT|LONG)\s+)?(?:(?P<ratio>\d+)x\s*)?(?P<strike>1[5-9]\d{3}|2\d{4}|3[0-4]\d{3})\s*(?P<opt>CE|PE)\b"
)
_VERB_LEG_RE = re.compile(
    r"\b(?P<verb>Buy|Sell|Long|Short)\s+(?:(?P<ratio>\d+)x\s*)?(?P<strike>1[5-9]\d{3}|2\d{4}|3[0-4]\d{3})\s*(?P<opt>CE|PE)\b",
    re.IGNORECASE,
)

_POLICY_CONSTANTS: tuple[float, ...] = (2000.0, 10000.0, 0.0, 100.0, 150.0, 200.0)

_KNOWN_ENUM_TOKENS: frozenset[str] = frozenset(
    {
        "BULL_PUT_SPREAD",
        "BULL_CALL_SPREAD",
        "BEAR_CALL_SPREAD",
        "BEAR_PUT_SPREAD",
        "IRON_CONDOR",
        "LEGGED_IRON_CONDOR_COMPONENT",
        "IRON_BUTTERFLY",
        "BULLISH_BUTTERFLY",
        "BEARISH_BUTTERFLY",
        "CUSTOM_DEFINED_RISK",
        "COMPLETED_DAILY_CLOSE",
        "COMPLETED_1H_CLOSE",
        "DOWNTREND_REPAIRED_HH_HL",
        "SINGLE_EXPIRY",
        "SINGLE_EXPIRY_DEGRADED",
        "MULTI_EXPIRY_FULL",
        "CONTESTED_WAIT",
        "SIT_OUT",
        "NOT_APPLICABLE",
        "PART_C_HISTORICAL",
        "NOVEL_QUANT",
        "HISTORICAL_REFERENCE_UNCONFIRMED",
        "CONFIRMED_QUANT_REGIME",
        "DEFINITION_OPEN",
        "LIVE_MID_INFERRED",
        "EOD_CLOSE_INFERRED",
        "LIVE_INTRADAY_SPOT_ATM_NEAREST_EXPIRY",
        "LIVE_VARIANCE_INTERPOLATED_CM30",
        "EXCHANGE_FORMULA_ESTIMATE",
        "BROKER_CALCULATOR_UNVERIFIED",
        "PORTFOLIO_STATE_UNKNOWN",
        "LIVE_MORNING_SNAPSHOT",
        "FULL_SIZE_TOUCH",
        "FULL_SIZE_BOOK_WALK",
        "INDICATIVE_TOUCH_ONLY",
        "PUT_OI_WALL",
        "CALL_OI_WALL",
        "FETCH_TIME_NOT_EXCHANGE",
        "UNVERIFIED_KOTAK_SCHEMA",
        "UNVERIFIED_SDK_SCHEMA",
    }
)


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


def _collect_numbers(obj: Any, out: list[float]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        fval = float(obj)
        if math.isfinite(fval):
            out.append(fval)
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _collect_numbers(item, out)


def _candidate_numbers(cand: CandidateStructure) -> list[float]:
    nums: list[float] = []
    _collect_numbers(dataclasses.asdict(cand), nums)
    nums.append(0.55 * cand.max_profit_rs)
    nums.append(cand.width_points - (10000.0 / max(1, cand.lot_size)))
    nums.append(float(cand.lot_size * cand.lots))
    return nums


def _market_non_candidate_numbers(state_dict: dict[str, Any]) -> list[float]:
    nums: list[float] = list(_POLICY_CONSTANTS)
    for key in ("contract_context", "portfolio_exposure_state", "market_evidence"):
        _collect_numbers(state_dict.get(key), nums)
    return nums


def _has_affirmative_live_ready(text: str) -> bool:
    for m in re.finditer(r"LIVE-READY", text, flags=re.IGNORECASE):
        prefix = text[max(0, m.start() - 16) : m.start()].upper()
        if "NOT " in prefix or "NEVER " in prefix or "NON-" in prefix:
            continue
        return True
    return False


def _allowed_tokens_for_state(state: NiftyOptionsState) -> set[str]:
    allowed = set(_KNOWN_ENUM_TOKENS)
    for c in state.candidate_structures:
        allowed.add(c.structure_id)
        allowed.update(c.part_a_checks.keys())
    for d in state.market_evidence.triggered_detectors:
        allowed.add(d.detector_id)
    for vt in state.market_evidence.price_structure.verified_level_touches:
        allowed.add(vt.level_name)
    return allowed


def validate_supplied_digest(
    digest: str,
    state: NiftyOptionsState,
    selected_primary: CandidateStructure | None,
    expected_is_live_ready: bool | None = None,
) -> tuple[int, list[str]]:
    """Strictly validate a supplied `whatsapp_digest` against the exact final `selected_primary`."""
    if not digest or not digest.strip():
        return 0, []

    checked = 0
    errors: list[str] = []
    text = digest.strip()

    if expected_is_live_ready is False and _has_affirmative_live_ready(text):
        checked += 1
        errors.append("digest:FALSE_LIVE_READY_CLAIM")

    if selected_primary is None:
        for sid in _STRUCT_ID_RE.findall(text):
            checked += 1
            errors.append(f"digest:ACTIONABLE_STRUCTURE_ON_WAIT:{sid}")
        if re.search(r"\bSUGGESTION\b", text):
            checked += 1
            errors.append("digest:SUGGESTION_BANNER_ON_WAIT")
        if re.search(r"\b(?:Buy|Sell|Short|Long)\s+[A-Za-z0-9_]{3,}", text, re.IGNORECASE):
            checked += 1
            errors.append("digest:TRADE_CALL_ON_WAIT")
        for m_leg in _COMPACT_LEG_RE.finditer(text):
            if m_leg.group("side"):
                checked += 1
                errors.append(f"digest:LEG_ORDER_ON_WAIT:{m_leg.group(0)}")
        return checked, errors

    # `selected_primary` is non-None: EVERY trade field in digest must match `selected_primary` ONLY
    allowed_enum_tokens = (_KNOWN_ENUM_TOKENS | {selected_primary.structure_id}) - {
        c.structure_id
        for c in state.candidate_structures
        if c.structure_id != selected_primary.structure_id
    }

    for sid in _STRUCT_ID_RE.findall(text):
        checked += 1
        if sid != selected_primary.structure_id:
            errors.append(
                f"digest:WRONG_STRUCTURE_ID:{sid}!=expected:{selected_primary.structure_id}"
            )

    text_without_struct_ids = _STRUCT_ID_RE.sub(" ", text)
    for tok in _FOREIGN_UNDERSCORE_TOKEN_RE.findall(text_without_struct_ids):
        checked += 1
        if tok not in allowed_enum_tokens:
            errors.append(f"digest:WRONG_CONTRACT_TOKEN:{tok}")

    for dt_str in _DATE_RE.findall(text):
        checked += 1
        if dt_str != selected_primary.expiry_date:
            errors.append(
                f"digest:WRONG_EXPIRY:{dt_str}!=expected:{selected_primary.expiry_date}"
            )

    # Validate any DTE claim
    for m_dte in re.finditer(r"\b(\d+)\s*DTE\b", text, re.IGNORECASE):
        checked += 1
        if int(m_dte.group(1)) != selected_primary.dte_trading_sessions:
            errors.append(
                f"digest:WRONG_DTE:{m_dte.group(1)}!=expected:{selected_primary.dte_trading_sessions}"
            )

    # Validate lot & unit quantity claims
    expected_qty = selected_primary.lot_size * selected_primary.lots
    for m_lot in re.finditer(r"\b(\d+)\s*lots?\b", text, re.IGNORECASE):
        checked += 1
        if int(m_lot.group(1)) != selected_primary.lots:
            errors.append(f"digest:WRONG_LOTS:{m_lot.group(1)}!=expected:{selected_primary.lots}")
    for m_qty in re.finditer(r"\b(\d+)\s*(?:qty|units)\b", text, re.IGNORECASE):
        checked += 1
        if int(m_qty.group(1)) != expected_qty:
            errors.append(f"digest:WRONG_QUANTITY:{m_qty.group(1)}!=expected:{expected_qty}")

    # Validate leg strike / option_type / side / ratio against selected_primary.legs
    legs_by_key = {(int(round(l.strike)), l.option_type): l for l in selected_primary.legs}
    for m_leg in _COMPACT_LEG_RE.finditer(text):
        k_int = int(m_leg.group("strike"))
        opt = m_leg.group("opt")
        side_tok = m_leg.group("side")
        ratio_tok = m_leg.group("ratio")
        checked += 1
        actual_leg = legs_by_key.get((k_int, opt))
        if actual_leg is None:
            errors.append(f"digest:WRONG_LEG_STRIKE_OR_TYPE:{k_int}{opt}")
            continue
        if side_tok:
            exp_short = actual_leg.side == "SHORT"
            got_short = side_tok in ("S", "SHORT")
            if exp_short != got_short:
                errors.append(
                    f"digest:WRONG_LEG_SIDE:{side_tok}_{k_int}{opt}!=expected:{actual_leg.side}"
                )
        if ratio_tok and int(ratio_tok) != actual_leg.ratio:
            errors.append(
                f"digest:WRONG_LEG_RATIO:{ratio_tok}x_{k_int}{opt}!=expected:{actual_leg.ratio}x"
            )

    for m_vleg in _VERB_LEG_RE.finditer(text):
        verb = m_vleg.group("verb").upper()
        k_int = int(m_vleg.group("strike"))
        opt = m_vleg.group("opt").upper()
        checked += 1
        actual_leg = legs_by_key.get((k_int, opt))
        if actual_leg is None:
            errors.append(f"digest:WRONG_VERB_LEG:{verb}_{k_int}{opt}")
            continue
        got_side = "LONG" if verb in ("BUY", "LONG") else "SHORT"
        if got_side != actual_leg.side:
            errors.append(
                f"digest:WRONG_VERB_SIDE:{verb}_{k_int}{opt}!=expected:{actual_leg.side}"
            )

    # Validate Net Credit / Net Debit direction if stated
    for m_entry in re.finditer(
        r"\bNet\s+(Credit|Debit)\s+([0-9]+(?:\.[0-9]+)?)\s*pts", text, re.IGNORECASE
    ):
        checked += 1
        got_credit = m_entry.group(1).lower() == "credit"
        if got_credit != selected_primary.is_net_credit:
            errors.append(f"digest:WRONG_CREDIT_DEBIT_SIDE:{m_entry.group(1)}")
        pts_v = float(m_entry.group(2))
        if abs(pts_v - selected_primary.executable_net_points_per_unit) > 0.25:
            errors.append(
                f"digest:WRONG_ENTRY_POINTS:{pts_v}!=expected:{selected_primary.executable_net_points_per_unit}"
            )

    # Candidate-scoped Rupee check (ONLY selected_primary's rupee figures + policy caps allowed!)
    allowed_selected_rs = [
        selected_primary.contractual_max_loss_rs,
        selected_primary.planned_stop_rs,
        selected_primary.portfolio_summed_stop_after_trade_rs,
        selected_primary.margin_estimate_rs,
        selected_primary.max_profit_rs,
        selected_primary.overnight_gap_stress_3pct_loss_rs,
        selected_primary.statutory_charges.total_charges_rs,
        0.55 * selected_primary.max_profit_rs,
        state.portfolio_exposure_state.summed_planned_stops_rs,
        2000.0,
        10000.0,
    ]
    stripped = _STRUCT_ID_RE.sub(" ", text)
    stripped = _DATE_RE.sub(" ", stripped)

    for m_rs in _RUPEE_RE.finditer(stripped):
        raw_num = m_rs.group(1).replace(",", "")
        val = float(raw_num)
        checked += 1
        if not any(abs(val - a) <= 1.05 for a in allowed_selected_rs):
            errors.append(f"digest:WRONG_RS_FOR_SELECTED:Rs {raw_num}")

    # Label-bound Rupee checks inside digest (catches swapping margin vs max loss on same candidate!)
    for pat, expected_val, tag in (
        (
            r"(?:max(?:imum)?\s+loss|contractual(?:\s+max(?:imum)?)?\s+(?:loss|bound|cap))\s*(?:of|is|=|:)?\s*Rs\.?\s*([0-9,]+(?:\.[0-9]+)?)",
            selected_primary.contractual_max_loss_rs,
            "MAX_LOSS",
        ),
        (
            r"(?<!summed\s)(?<!portfolio\s)(?:managed\s+stop|planned\s+stop|stop)\s*(?:of|is|=|:)?\s*Rs\.?\s*([0-9,]+(?:\.[0-9]+)?)",
            selected_primary.planned_stop_rs,
            "PLANNED_STOP",
        ),
        (
            r"(?:summed\s+(?:portfolio\s+)?stop|portfolio\s+summed\s+stop)\s*(?:of|is|=|:)?\s*Rs\.?\s*([0-9,]+(?:\.[0-9]+)?)",
            selected_primary.portfolio_summed_stop_after_trade_rs,
            "SUMMED_STOP",
        ),
        (
            r"margin(?:\s+estimate)?\s*(?:of|is|=|:|~)?\s*~?\s*Rs\.?\s*([0-9,]+(?:\.[0-9]+)?)",
            selected_primary.margin_estimate_rs,
            "MARGIN",
        ),
    ):
        for m_lbl in re.finditer(pat, stripped, flags=re.IGNORECASE):
            val = float(m_lbl.group(1).replace(",", ""))
            checked += 1
            if abs(val - expected_val) > 1.05:
                errors.append(
                    f"digest:SEMANTIC_{tag}_MISMATCH:Rs {val:.2f}!=expected:Rs {expected_val:.2f}"
                )

    return checked, errors


def _check_sentence_semantic_labels(
    field_name: str,
    sentence: str,
    state: NiftyOptionsState,
    scoped_candidates: Sequence[CandidateStructure],
    errors: list[str],
) -> int:
    """Bind labelled numeric claims in a prose sentence to their exact state/candidate meaning."""
    checked = 0
    spot = state.contract_context.spot.value
    ps = state.market_evidence.price_structure
    vol = state.market_evidence.volatility_state
    pos = state.market_evidence.derivatives_positioning

    market_bindings: list[tuple[str, float | str, float, str]] = [
        (r"\bspot(?:\s+is)?(?:\s+at)?\s*[=:]?\s*`?([0-9,]+(?:\.[0-9]+)?)`?", spot, 1.05, "spot"),
        (r"\b1H\s+MA20(?:\s+at)?\s*[=(:]?\s*`?([0-9,]+(?:\.[0-9]+)?)`?", ps.ma20_1h, 0.55, "ma20_1h"),
        (r"\b1H\s+EMA50(?:\s+at)?\s*[=(:]?\s*`?([0-9,]+(?:\.[0-9]+)?)`?", ps.ema50_1h, 0.55, "ema50_1h"),
        (r"\b1H\s+RSI(?:\(14\))?(?:\s+at)?\s*[=:]?\s*`?([0-9]+(?:\.[0-9]+)?)`?", ps.rsi_14_1h, 0.35, "rsi_14_1h"),
        (
            r"\bPut\s+OI\s+(?:wall(?:\s+support)?|support)\s+(?:at|=)\s*`?([0-9,]+(?:\.[0-9]+)?)`?",
            pos.put_oi_wall_strike,
            1.05,
            "put_oi_wall_strike",
        ),
        (
            r"\bCall\s+OI\s+(?:wall|resistance)\s+(?:at|=)\s*`?([0-9,]+(?:\.[0-9]+)?)`?",
            pos.call_oi_wall_strike,
            1.05,
            "call_oi_wall_strike",
        ),
    ]
    if isinstance(vol.ivp_90d, (int, float)):
        market_bindings.append(
            (
                r"\b90(?:-session|d)\s+IVP(?:\s+stands\s+at|\s+at)?\s*[=:]?\s*`?([0-9]+(?:\.[0-9]+)?)`?",
                float(vol.ivp_90d),
                0.35,
                "ivp_90d",
            )
        )

    for pat, actual_num, tol, label in market_bindings:
        for m in re.finditer(pat, sentence, flags=re.IGNORECASE):
            val = float(m.group(1).replace(",", ""))
            checked += 1
            if abs(val - float(actual_num)) > tol:
                errors.append(
                    f"{field_name}:SEMANTIC_{label.upper()}_MISMATCH:{val}!=expected:{actual_num}"
                )

    # Candidate-meaning bindings (checked against `scoped_candidates` for this sentence!)
    cand_bindings: list[tuple[str, Any, float, str]] = [
        (
            r"(?:contractual\s+(?:max(?:imum)?\s+loss|payoff\s+bound|bound|cap)|max(?:imum)?\s+loss)\s*(?:of|is|=|:)?\s*Rs\.?\s*([0-9,]+(?:\.[0-9]+)?)",
            lambda c: [c.contractual_max_loss_rs],
            1.05,
            "CONTRACTUAL_MAX_LOSS",
        ),
        (
            r"(?:managed|planned)\s+stop\s*(?:of|is|=|:)?\s*Rs\.?\s*([0-9,]+(?:\.[0-9]+)?)",
            lambda c: [c.planned_stop_rs, 2000.0],
            1.05,
            "PLANNED_STOP",
        ),
        (
            r"(?:portfolio\s+summed\s+stop|summed\s+portfolio\s+stop)(?:\s+after\s+trade|\s+post-trade)?\s*(?:of|is|=|:)?\s*Rs\.?\s*([0-9,]+(?:\.[0-9]+)?)",
            lambda c: [c.portfolio_summed_stop_after_trade_rs],
            1.05,
            "PORTFOLIO_SUMMED_STOP",
        ),
        (
            r"\bmargin(?:\s+estimate)?\s*(?:of|is|=|:|~)?\s*~?\s*Rs\.?\s*([0-9,]+(?:\.[0-9]+)?)",
            lambda c: [c.margin_estimate_rs],
            1.05,
            "MARGIN_ESTIMATE",
        ),
        (
            r"\bmax\s+profit\s*(?:of|is|=|:)?\s*Rs\.?\s*([0-9,]+(?:\.[0-9]+)?)",
            lambda c: [c.max_profit_rs],
            1.05,
            "MAX_PROFIT",
        ),
        (
            r"Tech\+(?:Options|Opt)\s+Score\s*([0-9]+(?:\.[0-9]+)?)",
            lambda c: [c.technical_options_score],
            0.25,
            "TECH_OPT_SCORE",
        ),
        (
            r"\blot_size\s*=\s*(\d+)\b",
            lambda c: [float(c.lot_size)],
            0.01,
            "LOT_SIZE",
        ),
        (
            r"\binvalidat(?:es|ion)(?:(?!1H|1D|15M)[^0-9.\n]){0,45}?\b(?:at|beyond)\s*`?([0-9,]+(?:\.[0-9]+)?)`?(?![A-Za-z])",
            lambda c: [c.invalidation_level],
            1.05,
            "INVALIDATION_LEVEL",
        ),
        (
            r"(?:overnight\s+gap[^R\n]{0,45}?stress[^R\n]{0,45}?|gap-stress\s+alternative[^R\n]{0,45}?)\bRs\.?\s*([0-9,]+(?:\.[0-9]+)?)",
            lambda c: [c.overnight_gap_stress_3pct_loss_rs],
            1.05,
            "OVERNIGHT_GAP_STRESS",
        ),
    ]

    for pat, extractor, tol, tag in cand_bindings:
        for m in re.finditer(pat, sentence, flags=re.IGNORECASE):
            val = float(m.group(1).replace(",", ""))
            checked += 1
            if not scoped_candidates:
                # Policy constant mention like "capped by the Rs 2,000 managed stop" when no candidate is scoped
                if tag == "PLANNED_STOP" and abs(val - 2000.0) <= 1.05:
                    continue
                errors.append(f"{field_name}:UNSCOPED_{tag}:{val}")
                continue
            allowed_vals = [v for c in scoped_candidates for v in extractor(c)]
            if not any(abs(val - av) <= tol for av in allowed_vals):
                errors.append(f"{field_name}:SEMANTIC_{tag}_MISMATCH:{val}")

    return checked


def _scan_prose_fields_semantically(
    state: NiftyOptionsState,
    state_dict: dict[str, Any],
    prose_fields: Mapping[str, str],
    selected_primary: CandidateStructure | None,
    expected_is_live_ready: bool | None = None,
) -> tuple[int, list[str]]:
    """Verify every delivered prose field with candidate-scoped & metric-meaning binding."""
    if not prose_fields:
        return 0, []

    market_nums = _market_non_candidate_numbers(state_dict)
    cand_by_id = {c.structure_id: c for c in state.candidate_structures}
    allowed_tokens = _allowed_tokens_for_state(state)

    checked_count = 0
    uncited: list[str] = []

    selection_bound_fields = {
        "final_recommendation_rationale",
        "reasoning_summary",
        "neutral_arbiter_note",
    }

    for field_name, text in prose_fields.items():
        if not text:
            continue

        if expected_is_live_ready is False and _has_affirmative_live_ready(text):
            checked_count += 1
            uncited.append(f"{field_name}:FALSE_LIVE_READY_CLAIM")

        # 1. Check structure IDs and foreign underscore tokens across the whole field
        for sid in _STRUCT_ID_RE.findall(text):
            checked_count += 1
            if sid not in cand_by_id:
                uncited.append(f"{field_name}:unknown_structure_id:{sid}")

        # Any selection-verb claim ("selects X", "recommends X", "endorses X") in ANY prose field
        # must match `selected_primary`, and `final_recommendation_rationale` / `neutral_arbiter_note`
        # may only cite `selected_primary`.
        for m_sel in re.finditer(
            r"\b(?:selects|recommends|endorses|proposes)\s+`?(" + _STRUCT_ID_RE.pattern + r")`?",
            text,
            flags=re.IGNORECASE,
        ):
            claimed_sid = m_sel.group(1)
            if selected_primary is None or claimed_sid != selected_primary.structure_id:
                checked_count += 1
                uncited.append(
                    f"{field_name}:contradictory_selection_claim:{claimed_sid}!=selected:{selected_primary.structure_id if selected_primary else 'WAIT'}"
                )

        if field_name in ("final_recommendation_rationale", "neutral_arbiter_note"):
            for sid in _STRUCT_ID_RE.findall(text):
                if selected_primary is None or sid != selected_primary.structure_id:
                    checked_count += 1
                    uncited.append(
                        f"{field_name}:contradictory_candidate_id:{sid}!=selected:{selected_primary.structure_id if selected_primary else 'WAIT'}"
                    )

        # Remove backticked dotted state paths and already-validated structure IDs before scanning tokens
        text_no_paths = re.sub(r"`[a-z0-9_.\[\]]+`", " ", text)
        text_no_paths = _STRUCT_ID_RE.sub(" ", text_no_paths)
        for tok in _FOREIGN_UNDERSCORE_TOKEN_RE.findall(text_no_paths):
            checked_count += 1
            if tok not in allowed_tokens:
                uncited.append(f"{field_name}:unknown_contract_or_token:{tok}")

        # 2. Split into sentences and enforce candidate-scoped + semantic-label number binding.
        #    When a sentence explicitly names candidate ID(s), numbers in that sentence are bound to
        #    those exact named candidate(s); otherwise candidate numbers are bound strictly to
        #    `selected_primary` (never to arbitrary other candidates in state!).
        sentences = [s.strip() for s in re.split(r"(?<=[.!?;])\s+|\n+", text) if s.strip()]
        for sent in sentences:
            mentioned_ids = [sid for sid in _STRUCT_ID_RE.findall(sent) if sid in cand_by_id]
            if mentioned_ids:
                scoped_cands: list[CandidateStructure] = [cand_by_id[sid] for sid in mentioned_ids]
            else:
                scoped_cands = [selected_primary] if selected_primary is not None else []

            checked_count += _check_sentence_semantic_labels(
                field_name=field_name,
                sentence=sent,
                state=state,
                scoped_candidates=scoped_cands,
                errors=uncited,
            )

            allowed_nums = list(market_nums)
            for c in scoped_cands:
                allowed_nums.extend(_candidate_numbers(c))

            stripped = _STRUCT_ID_RE.sub(" ", sent)
            stripped = _DATE_RE.sub(" ", stripped)

            for m_rs in _RUPEE_RE.finditer(stripped):
                raw_num = m_rs.group(1).replace(",", "")
                try:
                    val = float(raw_num)
                except ValueError:
                    continue
                checked_count += 1
                if not any(abs(val - k) <= 1.05 for k in allowed_nums):
                    uncited.append(f"{field_name}:Rs {raw_num}")

            stripped_no_rs = _RUPEE_RE.sub(" ", stripped)
            for m_lvl in _INDEX_LEVEL_RE.finditer(stripped_no_rs):
                whole, frac = m_lvl.group(1), m_lvl.group(2)
                raw_lvl = f"{whole}.{frac}" if frac else whole
                val = float(raw_lvl)
                checked_count += 1
                # ExcludeOI contract counts written like "105,000 contracts" (comma-formatted above; bare 5-digit checked here)
                if not any(abs(val - k) <= 1.05 for k in allowed_nums):
                    uncited.append(f"{field_name}:index_level:{raw_lvl}")

    return checked_count, uncited


def verify_llm_synthesis(
    state: NiftyOptionsState,
    raw_citations: Sequence[dict[str, Any]],
    selected_primary: CandidateStructure | None,
    max_unsupported_claim_rate: float = 0.02,
    prose_texts: Sequence[str] = (),
    consolidator_recommended_id: str | None = None,
    prose_fields: Mapping[str, str] | None = None,
    supplied_whatsapp_digest: str | None = None,
    expected_is_live_ready: bool | None = None,
) -> tuple[list[LLMClaimCitation], VerifierReport]:
    """Deterministically verify citations, all delivered prose fields, supplied digest, and Part A."""
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

    merged_prose_fields: dict[str, str] = {}
    if prose_fields:
        merged_prose_fields.update(prose_fields)
    for idx_p, txt in enumerate(prose_texts):
        key = f"prose_text_{idx_p}"
        if txt and txt not in merged_prose_fields.values():
            merged_prose_fields[key] = txt

    prose_checked, uncited_prose = _scan_prose_fields_semantically(
        state=state,
        state_dict=state_dict,
        prose_fields=merged_prose_fields,
        selected_primary=selected_primary,
        expected_is_live_ready=expected_is_live_ready,
    )

    if supplied_whatsapp_digest is not None:
        dig_checked, dig_errors = validate_supplied_digest(
            digest=supplied_whatsapp_digest,
            state=state,
            selected_primary=selected_primary,
            expected_is_live_ready=expected_is_live_ready,
        )
        prose_checked += dig_checked
        uncited_prose.extend(dig_errors)

    if uncited_prose:
        failure_reasons.append(
            f"PROSE_OR_DIGEST_SEMANTIC_MISMATCH: ungrounded/contradictory claims detected {uncited_prose}."
        )

    # Re-verify Consolidator recommendation & Part A + full-size depth compliance of primary proposal
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
        elif not rec_cand.full_size_fill_supported:
            part_a_reverified = False
            failure_reasons.append(
                f"Consolidator recommended {consolidator_recommended_id!r} without verified "
                f"full-size depth ({rec_cand.execution_fill_basis})."
            )

    if selected_primary is not None:
        any_non_pass = any(
            chk.status in ("FAIL", "UNKNOWN") for chk in selected_primary.part_a_checks.values()
        )
        if (
            any_non_pass
            or selected_primary.part_a_overall_status != "PASS"
            or not selected_primary.full_size_fill_supported
        ):
            part_a_reverified = False
            failure_reasons.append(
                f"Primary structure {selected_primary.structure_id} failed Part A or full-size depth re-verification."
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
