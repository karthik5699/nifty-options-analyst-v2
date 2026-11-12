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
_RUPEE_PREFIX_PAT = r"(?:₹|\b(?:Rs\.?|INR))\s*"
_RUPEE_RE = re.compile(_RUPEE_PREFIX_PAT + r"([0-9][0-9,]*(?:\.[0-9]+)?)")
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_INDEX_LEVEL_RE = re.compile(r"(?<![0-9-])(1[5-9]\d{3}|2\d{4}|3[0-4]\d{3})(?:\.(\d+))?(?![0-9-])")
_COMPACT_LEG_RE = re.compile(
    r"(?:(?P<side>S|B|SHORT|LONG)\s+)?(?:(?P<ratio>\d+)x\s*)?(?P<strike>1[5-9]\d{3}|2\d{4}|3[0-4]\d{3})\s*(?P<opt>CE|PE)\b"
)
_VERB_LEG_RE = re.compile(
    r"\b(?P<verb>Buy|Sell|Long|Short)\s+(?:(?P<ratio>\d+)x\s*)?(?P<strike>1[5-9]\d{3}|2\d{4}|3[0-4]\d{3})\s*(?P<opt>CE|PE)\b",
    re.IGNORECASE,
)
_LEG_TRAILING_PRICE_RE = re.compile(
    r"^\s*(?:@|at|for|=|around|near)\s*~?\s*(?:₹|\b(?:Rs\.?|INR)\s*)?([0-9][0-9,]*(?:\.[0-9]+)?)(?:\s*(?:pts?|points?))?",
    re.IGNORECASE,
)
_POINTS_VALUE_RE = re.compile(
    r"\b([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:pts?|points?)\b",
    re.IGNORECASE,
)
_SELECTED_TRADE_SUBJECT_PAT = (
    r"(?:"
    r"(?:selected|chosen|recommended|primary|proposed)\s+"
    r"(?:trade|selection|pick|proposal|recommendation|position|spread|strategy|condor|butterfly|vertical|structure)"
    r"|(?:the|this|our)\s+(?:trade|selection|pick|proposal|recommendation|position)"
    r")"
)
_DIR_WORD_PAT = r"(?P<dir>bullish|bearish|(?:delta[-\s]+|market[-\s]+)?neutral)"
_TRADE_DIR_BEFORE_RE = re.compile(
    r"\b(?:selected|chosen|recommended|primary|proposed)\s+(?:a\s+|an\s+)?"
    + _DIR_WORD_PAT
    + r"(?:\s+(?:defined-risk|options?|credit|debit|vertical))?\s+"
    r"(?:trade|selection|pick|proposal|recommendation|position|spread|strategy|condor|butterfly|vertical|structure)\b",
    re.IGNORECASE,
)
_TRADE_DIR_AFTER_RE = re.compile(
    r"\b"
    + _SELECTED_TRADE_SUBJECT_PAT
    + r"\b[^.!?;,\n]{0,35}?\b(?:is|are|remains|stays|was|as|with\s+a|has\s+a|carries\s+a)\s+"
    r"(?:a\s+|an\s+|strictly\s+|purely\s+|decidedly\s+|net\s+)?"
    + _DIR_WORD_PAT
    + r"\b",
    re.IGNORECASE,
)
_STRUCT_ID_DIR_RE = re.compile(
    r"`?(?P<sid>"
    + _STRUCT_ID_RE.pattern
    + r")`?[^.!?;,\n]{0,25}?\b(?:is|as|remains|carries\s+a)\s+(?:a\s+|an\s+)?"
    + _DIR_WORD_PAT
    + r"\b",
    re.IGNORECASE,
)
_PROFIT_GUARANTEE_RE = re.compile(
    r"\b(?:"
    r"guaranteed\s+(?:to\s+)?(?:be\s+)?(?:profit\w*|gain\w*|win\w*|return\w*|make\s+money|succeed\w*)"
    r"|(?:profit\w*|gain\w*|win\w*|return\w*)\s+(?:is|are)\s+guaranteed"
    r"|(?:cannot|can't|can\s+not|never|impossible\s+to)\s+lose"
    r"|risk[-\s]+free"
    r"|no\s+(?:risk|chance|possibility|probability)\s+of\s+(?:any\s+)?loss"
    r"|zero\s+(?:max(?:imum)?\s+)?(?:loss|downside)"
    r"|100(?:\.0+)?%\s*(?:pop|win\w*|probabilit\w*|certain\w*|safe\w*|profit\w*)"
    r"|sure[-\s]*shot"
    r"|free\s+money"
    r"|certain\s+profit\w*"
    r")\b",
    re.IGNORECASE,
)
_FOREIGN_INSTRUMENT_RE = re.compile(
    r"\b(?:BANKNIFTY|BANK\s+NIFTY|FINNIFTY|FIN\s+NIFTY|MIDCPNIFTY|MIDCAP\s+NIFTY|"
    r"NIFTYNXT50|NIFTY\s+NEXT\s*50|NIFTYIT|NIFTY\s+IT|SENSEX|BANKEX)\b",
    re.IGNORECASE,
)
_ACTIONABLE_VERB_RE = re.compile(
    r"(?<![A-Za-z-])\b(?P<verb>Buy|Sell|Enter|Execute)\b(?![A-Za-z-])",
    re.IGNORECASE,
)
_STRICT_BUY_SELL_LEG_RE = re.compile(
    r"^(?P<verb>Buy|Sell)\s+"
    r"(?:(?P<qty>\d[\d,]*)\s*(?P<qkind>lots?|units?|qty)\s+(?:of\s+)?)?"
    r"(?:(?P<ratio>\d+)x\s*)?"
    r"(?:NIFTY\s+)?"
    r"(?P<strike>1[5-9]\d{3}|2\d{4}|3[0-4]\d{3})\s*(?P<opt>CE|PE)\b",
    re.IGNORECASE,
)
_STRICT_ENTER_EXEC_LEG_RE = re.compile(
    r"^(?P<verb>Enter|Execute)\s+"
    r"(?:(?P<qty>\d[\d,]*)\s*(?P<qkind>lots?|units?|qty)\s+(?:of\s+)?)?"
    r"(?:(?P<ratio>\d+)x\s*)?"
    r"(?:NIFTY\s+)?"
    r"(?P<strike>1[5-9]\d{3}|2\d{4}|3[0-4]\d{3})\s*(?P<opt>CE|PE)\b",
    re.IGNORECASE,
)
_STRICT_ENTER_EXEC_STRUCT_RE = re.compile(
    r"^(?:Enter|Execute)\s+`?(?P<sid>" + _STRUCT_ID_RE.pattern + r")`?",
    re.IGNORECASE,
)
_LONG_SHORT_INSTRUMENT_RE = re.compile(
    r"(?<![A-Za-z-])\b(?P<verb>Long|Short)(?![A-Za-z-])\s+(?:"
    r"(?:\d[\d,]*\s*(?:lots?|units?|qty|contracts?)\s*(?:of\s+)?)?(?:(?:NIFTY\s+)?futures?\b|$|[.,;])"
    r"|(?!(?:CE|PE)\b)(?P<sym>[A-Z][A-Z0-9]{1,15})\b"
    r")"
)
_UPPER_WORD_RE = re.compile(r"\b[A-Z]{4,}[A-Z0-9]*\b")

_ALLOWED_STATIC_UPPER_WORDS: frozenset[str] = frozenset(
    {
        "NIFTY",
        "WAIT",
        "PASS",
        "FAIL",
        "FAILED",
        "UNKNOWN",
        "UNVERIFIED",
        "VERIFIED",
        "UNAVAILABLE",
        "BULLISH",
        "BEARISH",
        "NEUTRAL",
        "UPTREND",
        "DOWNTREND",
        "RANGE",
        "CONTANGO",
        "BACKWARDATION",
        "FLAT",
        "HIGH",
        "MODERATE",
        "MEDIUM",
        "SHORT",
        "LONG",
        "RESEARCH",
        "NOTE",
        "ONLY",
        "LIVE",
        "READY",
        "ACTIVE",
        "HALTED",
        "SUPPRESSED",
        "SUGGESTION",
        "INDICATIVE",
        "INSUFFICIENT",
        "DATA",
        "MA20",
        "EMA50",
        "CM30",
        "SEBI",
        "VWAP",
        "BLACK",
        "NONE",
        "TRUE",
        "FALSE",
        "JSON",
        "JSONL",
        "HTTP",
        "POST",
        "MODE",
        "HARD",
        "STRICT",
        "SINGLE",
        "CONSTRAINTS",
        "CREDIT",
        "DEBIT",
        "OPTIONS",
        "ANALYST",
        "OFFLINE",
        "STALE",
        "ZERO",
        "EXPIRY",
        "MACRO",
        "INR",
    }
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


def _allowed_upper_words_for_state(state: NiftyOptionsState) -> set[str]:
    allowed = set(_ALLOWED_STATIC_UPPER_WORDS)
    allowed.update(_UPPER_WORD_RE.findall(state.contract_context.symbol.upper()))
    for ev in state.market_evidence.event_calendar.events_within_dte:
        allowed.update(_UPPER_WORD_RE.findall(ev.event_name.upper()))
        allowed.update(_UPPER_WORD_RE.findall(ev.category.upper()))
    for det in state.market_evidence.triggered_detectors:
        allowed.update(_UPPER_WORD_RE.findall(det.detector_name.upper()))
    return allowed


def _normalize_dir_word(raw_dir: str) -> str:
    low = raw_dir.lower()
    if "neutral" in low:
        return "NEUTRAL"
    if "bull" in low:
        return "BULLISH"
    return "BEARISH"


def _allowed_rupee_numbers(
    state: NiftyOptionsState, scoped_cands: Sequence[CandidateStructure]
) -> list[float]:
    nums: list[float] = [
        2000.0,
        10000.0,
        state.portfolio_exposure_state.summed_planned_stops_rs,
        state.portfolio_exposure_state.remaining_stop_budget_rs,
    ]
    for op in state.portfolio_exposure_state.open_positions:
        nums.append(op.planned_stop_rs)
        nums.append(abs(op.current_executable_spread_value_rs))
    for c in scoped_cands:
        nums.extend(
            [
                c.contractual_max_loss_rs,
                c.planned_stop_rs,
                c.portfolio_summed_stop_after_trade_rs,
                c.margin_estimate_rs,
                c.max_profit_rs,
                c.overnight_gap_stress_3pct_loss_rs,
                c.statutory_charges.total_charges_rs,
                0.55 * c.max_profit_rs,
                abs(c.net_theta_daily_rs),
                abs(c.net_vega_rs),
            ]
        )
        if not c.full_size_fill_supported:
            nums.append(c.indicative_touch_max_loss_rs)
    return nums


def _rupee_matches_allowed(val: float, allowed_rs: Sequence[float]) -> bool:
    return any(abs(val - k) <= (0.01 if abs(k) < 5.0 else 1.05) for k in allowed_rs)


def _allowed_points_values(
    state: NiftyOptionsState, scoped_cands: Sequence[CandidateStructure]
) -> list[float]:
    ps = state.market_evidence.price_structure
    vol = state.market_evidence.volatility_state
    pos = state.market_evidence.derivatives_positioning
    pts: list[float] = [
        ps.atr_14_1h,
        ps.atr_14_1d,
        abs(vol.skew_25d_put_minus_call * 100.0),
        abs(vol.iv_rv_spread * 100.0),
        abs(vol.term_structure_slope_near_minus_far * 100.0),
        100.0,
        150.0,
        200.0,
    ]
    if pos.intraday_spot_change_pts is not None:
        pts.append(abs(pos.intraday_spot_change_pts))
    for c in scoped_cands:
        pts.extend(
            [
                c.executable_net_points_per_unit,
                c.slippage_buffer_points_per_unit,
                c.width_points,
                abs(c.indicative_stop_spread_value_per_unit),
                abs(c.width_points - 10000.0 / max(1, c.lot_size)),
            ]
        )
        if not c.full_size_fill_supported:
            pts.append(c.indicative_touch_net_points_per_unit)
        for leg in c.legs:
            if leg.executable_entry_price is not None:
                pts.append(leg.executable_entry_price)
            if leg.vwap_fill_price is not None:
                pts.append(leg.vwap_fill_price)
            if leg.fill_basis != "FULL_SIZE_BOOK_WALK" or (
                leg.executable_entry_price is None and leg.vwap_fill_price is None
            ):
                pts.extend([leg.mid, leg.bid, leg.ask])
    return pts


def _leg_price_matches_candidate(
    cand: CandidateStructure,
    strike_int: int,
    opt_type: str,
    price_val: float,
    expected_side: str | None = None,
    expected_ratio: int | None = None,
) -> bool:
    for leg in cand.legs:
        if int(round(leg.strike)) == strike_int and leg.option_type == opt_type:
            if expected_side is not None and leg.side != expected_side:
                continue
            if expected_ratio is not None and leg.ratio != expected_ratio:
                continue
            allowed_p: list[float] = []
            if leg.executable_entry_price is not None:
                allowed_p.append(leg.executable_entry_price)
            if leg.vwap_fill_price is not None:
                allowed_p.append(leg.vwap_fill_price)
            if leg.fill_basis != "FULL_SIZE_BOOK_WALK" or not allowed_p:
                allowed_p.extend([leg.mid, leg.bid, leg.ask])
            if any(abs(price_val - ap) <= 0.55 for ap in allowed_p):
                return True
    return False


def _leg_matches_candidate(
    cand: CandidateStructure,
    strike_int: int,
    opt_type: str,
    expected_side: str | None = None,
    expected_ratio: int | None = None,
    expected_qty: int | None = None,
    qty_kind: str | None = None,
) -> bool:
    if expected_qty is not None and qty_kind is not None:
        qk = qty_kind.lower()
        if qk.startswith("lot"):
            if expected_qty != cand.lots:
                return False
        else:
            if expected_qty != cand.lot_size * cand.lots:
                return False
    for leg in cand.legs:
        if int(round(leg.strike)) == strike_int and leg.option_type == opt_type:
            if expected_side is not None and leg.side != expected_side:
                continue
            if expected_ratio is not None and leg.ratio != expected_ratio:
                continue
            return True
    return False


def _check_trade_directives_quantities_and_symbols(
    field_name: str,
    text: str,
    state: NiftyOptionsState,
    scoped_candidates: Sequence[CandidateStructure],
    selected_primary: CandidateStructure | None,
    allowed_upper_words: set[str],
    errors: list[str],
    is_digest: bool = False,
) -> int:
    """Reject foreign symbols, wrong unit/lot quantities, and unbound trade directives in ANY text."""
    checked = 0
    pos = state.market_evidence.derivatives_positioning

    # 1. Explicit foreign index / instrument blocklist (case-insensitive)
    for m_fi in _FOREIGN_INSTRUMENT_RE.finditer(text):
        checked += 1
        errors.append(f"{field_name}:FOREIGN_INSTRUMENT:{m_fi.group(0)}")

    # 2. Unlisted uppercase ticker / contract words (>= 4 chars) & qualitative profit-guarantee blocklist
    clean_upper_scan = re.sub(r"`[a-z0-9_.\[\]]+`", " ", text)
    for _ in _PROFIT_GUARANTEE_RE.finditer(clean_upper_scan):
        checked += 1
        errors.append(f"{field_name}:FALSE_PROFIT_GUARANTEE_CLAIM")

    clean_upper_scan = _STRUCT_ID_RE.sub(" ", clean_upper_scan)
    clean_upper_scan = _FOREIGN_UNDERSCORE_TOKEN_RE.sub(" ", clean_upper_scan)
    for uw in _UPPER_WORD_RE.findall(clean_upper_scan):
        checked += 1
        if uw not in allowed_upper_words:
            errors.append(f"{field_name}:FOREIGN_UPPER_SYMBOL:{uw}")

    # 2b. Trade / selection direction binding (`bullish` / `bearish` / `neutral` must match candidate direction)
    for dir_re in (_TRADE_DIR_BEFORE_RE, _TRADE_DIR_AFTER_RE, _STRUCT_ID_DIR_RE):
        for m_dir in dir_re.finditer(text):
            claimed_dir = _normalize_dir_word(m_dir.group("dir"))
            checked += 1
            if not scoped_candidates:
                errors.append(f"{field_name}:TRADE_DIRECTION_ON_WAIT:{claimed_dir}")
            elif not any(c.direction == claimed_dir for c in scoped_candidates):
                errors.append(
                    f"{field_name}:WRONG_TRADE_DIRECTION:{claimed_dir}!=expected:{sorted({c.direction for c in scoped_candidates})}"
                )

    # 3. Unit, lot, and contract count binding
    for m_qty in re.finditer(r"\b(\d[\d,]*)\s*(?:units?|qty|quantities)\b", text, re.IGNORECASE):
        q_val = int(m_qty.group(1).replace(",", ""))
        checked += 1
        allowed_units = {c.lot_size * c.lots for c in scoped_candidates}
        if not allowed_units or q_val not in allowed_units:
            errors.append(f"{field_name}:WRONG_QUANTITY:{q_val}!=expected:{sorted(allowed_units)}")

    for m_lot in re.finditer(r"\b(\d[\d,]*)\s*lots?\b", text, re.IGNORECASE):
        l_val = int(m_lot.group(1).replace(",", ""))
        checked += 1
        allowed_lots = {c.lots for c in scoped_candidates}
        if not allowed_lots or l_val not in allowed_lots:
            errors.append(f"{field_name}:WRONG_LOTS:{l_val}!=expected:{sorted(allowed_lots)}")

    for m_con in re.finditer(r"\b(\d[\d,]*)\s*contracts?\b", text, re.IGNORECASE):
        c_val = int(m_con.group(1).replace(",", ""))
        checked += 1
        prefix = text[max(0, m_con.start() - 20) : m_con.start()]
        if re.search(r"\b(?:buy|sell|long|short|enter|execute|trade)\s*$", prefix, re.IGNORECASE):
            allowed_contracts = {c.lots for c in scoped_candidates}
        else:
            allowed_contracts = {
                pos.put_oi_wall_contracts,
                pos.call_oi_wall_contracts,
            } | {c.lots for c in scoped_candidates}
        if c_val not in allowed_contracts:
            errors.append(f"{field_name}:WRONG_CONTRACTS:{c_val}")

    # 4. Actionable trade-verb directives (`Buy`, `Sell`, `Enter`, `Execute`) must match
    #    `selected_primary` ONLY (never a non-selected candidate or unbound symbol/quantity!)
    for m_act in _ACTIONABLE_VERB_RE.finditer(text):
        verb = m_act.group("verb").upper()
        tail = text[m_act.start() :]
        checked += 1
        if selected_primary is None:
            errors.append(f"{field_name}:TRADE_DIRECTIVE_ON_WAIT:{verb}")
            continue
        if verb in ("BUY", "SELL"):
            m_bs = _STRICT_BUY_SELL_LEG_RE.match(tail)
            if m_bs is None:
                errors.append(f"{field_name}:UNBOUND_TRADE_DIRECTIVE:{verb}")
                continue
            exp_side = "LONG" if verb == "BUY" else "SHORT"
            k_int = int(m_bs.group("strike"))
            opt_t = m_bs.group("opt").upper()
            r_int = int(m_bs.group("ratio")) if m_bs.group("ratio") else None
            q_int = int(m_bs.group("qty").replace(",", "")) if m_bs.group("qty") else None
            q_kind = m_bs.group("qkind")
            if not _leg_matches_candidate(
                selected_primary, k_int, opt_t, exp_side, r_int, q_int, q_kind
            ):
                errors.append(f"{field_name}:WRONG_DIRECTIVE_LEG:{verb}_{k_int}{opt_t}")
                continue
            m_tp = _LEG_TRAILING_PRICE_RE.match(tail[m_bs.end() :])
            if m_tp is not None:
                p_val = float(m_tp.group(1).replace(",", ""))
                checked += 1
                if not _leg_price_matches_candidate(
                    selected_primary, k_int, opt_t, p_val, exp_side, r_int
                ):
                    errors.append(f"{field_name}:WRONG_LEG_ENTRY_PRICE:{k_int}{opt_t}@{p_val}")
        else:
            m_es = _STRICT_ENTER_EXEC_STRUCT_RE.match(tail)
            if m_es is not None:
                if m_es.group("sid") != selected_primary.structure_id:
                    errors.append(
                        f"{field_name}:WRONG_DIRECTIVE_STRUCTURE:{m_es.group('sid')}!=expected:{selected_primary.structure_id}"
                    )
                continue
            m_el = _STRICT_ENTER_EXEC_LEG_RE.match(tail)
            if m_el is None:
                errors.append(f"{field_name}:UNBOUND_ENTER_EXECUTE:{verb}")
                continue
            k_int = int(m_el.group("strike"))
            opt_t = m_el.group("opt").upper()
            r_int = int(m_el.group("ratio")) if m_el.group("ratio") else None
            q_int = int(m_el.group("qty").replace(",", "")) if m_el.group("qty") else None
            q_kind = m_el.group("qkind")
            if not _leg_matches_candidate(
                selected_primary, k_int, opt_t, None, r_int, q_int, q_kind
            ):
                errors.append(f"{field_name}:WRONG_ENTER_EXECUTE_LEG:{k_int}{opt_t}")
                continue
            m_tp = _LEG_TRAILING_PRICE_RE.match(tail[m_el.end() :])
            if m_tp is not None:
                p_val = float(m_tp.group(1).replace(",", ""))
                checked += 1
                if not _leg_price_matches_candidate(
                    selected_primary, k_int, opt_t, p_val, None, r_int
                ):
                    errors.append(f"{field_name}:WRONG_LEG_ENTRY_PRICE:{k_int}{opt_t}@{p_val}")

    # 5. `Long` / `Short` futures or ticker directives, plus `_VERB_LEG_RE` & `_COMPACT_LEG_RE`
    for m_ls in _LONG_SHORT_INSTRUMENT_RE.finditer(text):
        checked += 1
        errors.append(f"{field_name}:UNBOUND_LONG_SHORT_TARGET:{m_ls.group('verb').upper()}")

    for m_vleg in _VERB_LEG_RE.finditer(text):
        verb = m_vleg.group("verb").upper()
        k_int = int(m_vleg.group("strike"))
        opt = m_vleg.group("opt").upper()
        r_int = int(m_vleg.group("ratio")) if m_vleg.group("ratio") else None
        exp_side = "LONG" if verb in ("BUY", "LONG") else "SHORT"
        checked += 1
        if not scoped_candidates or not any(
            _leg_matches_candidate(c, k_int, opt, exp_side, r_int) for c in scoped_candidates
        ):
            errors.append(f"{field_name}:WRONG_VERB_LEG:{verb}_{k_int}{opt}")
            continue
        m_tp = _LEG_TRAILING_PRICE_RE.match(text[m_vleg.end() :])
        if m_tp is not None:
            p_val = float(m_tp.group(1).replace(",", ""))
            checked += 1
            if not any(
                _leg_price_matches_candidate(c, k_int, opt, p_val, exp_side, r_int)
                for c in scoped_candidates
            ):
                errors.append(f"{field_name}:WRONG_LEG_ENTRY_PRICE:{k_int}{opt}@{p_val}")

    oi_ref_strikes = {
        (int(round(pos.put_oi_wall_strike)), "PE"),
        (int(round(pos.call_oi_wall_strike)), "CE"),
        (int(round(pos.max_pain_strike)), "PE"),
        (int(round(pos.max_pain_strike)), "CE"),
    }
    for m_leg in _COMPACT_LEG_RE.finditer(text):
        k_int = int(m_leg.group("strike"))
        opt = m_leg.group("opt")
        side_tok = m_leg.group("side")
        ratio_tok = m_leg.group("ratio")
        r_int = int(ratio_tok) if ratio_tok else None
        exp_side: str | None = None
        if side_tok:
            exp_side = "SHORT" if side_tok in ("S", "SHORT") else "LONG"
        checked += 1
        m_tp = _LEG_TRAILING_PRICE_RE.match(text[m_leg.end() :])
        if (
            not is_digest
            and side_tok is None
            and ratio_tok is None
            and m_tp is None
            and (k_int, opt) in oi_ref_strikes
        ):
            continue
        if not scoped_candidates or not any(
            _leg_matches_candidate(c, k_int, opt, exp_side, r_int) for c in scoped_candidates
        ):
            errors.append(f"{field_name}:WRONG_LEG_SPEC:{k_int}{opt}")
            continue
        if m_tp is not None:
            p_val = float(m_tp.group(1).replace(",", ""))
            checked += 1
            if not any(
                _leg_price_matches_candidate(c, k_int, opt, p_val, exp_side, r_int)
                for c in scoped_candidates
            ):
                errors.append(f"{field_name}:WRONG_LEG_ENTRY_PRICE:{k_int}{opt}@{p_val}")

    # 6. Expiry date, DTE, and Net Credit / Debit claims
    text_no_sids = _STRUCT_ID_RE.sub(" ", text)
    if is_digest:
        allowed_dates = {selected_primary.expiry_date} if selected_primary is not None else set()
    else:
        allowed_dates = (
            {c.expiry_date for c in scoped_candidates}
            | {e.event_date for e in state.market_evidence.event_calendar.events_within_dte}
            | {state.run_metadata.decision_timestamp_ist[:10]}
        )
    for dt_str in _DATE_RE.findall(text_no_sids):
        checked += 1
        if dt_str not in allowed_dates:
            errors.append(f"{field_name}:WRONG_DATE_OR_EXPIRY:{dt_str}")

    for m_dte in re.finditer(r"\b(\d+)\s*DTE\b", text, re.IGNORECASE):
        dte_val = int(m_dte.group(1))
        checked += 1
        allowed_dtes = {c.dte_trading_sessions for c in scoped_candidates}
        if not allowed_dtes or dte_val not in allowed_dtes:
            errors.append(f"{field_name}:WRONG_DTE:{dte_val}!=expected:{sorted(allowed_dtes)}")

    for m_entry in re.finditer(
        r"\bNet\s+(Credit|Debit)\s+([0-9]+(?:\.[0-9]+)?)\s*pts", text, re.IGNORECASE
    ):
        checked += 1
        got_credit = m_entry.group(1).lower() == "credit"
        pts_v = float(m_entry.group(2))
        if not scoped_candidates or not any(
            c.is_net_credit == got_credit
            and abs(pts_v - c.executable_net_points_per_unit) <= 0.25
            for c in scoped_candidates
        ):
            errors.append(f"{field_name}:WRONG_NET_ENTRY:{pts_v}")

    allowed_pts = _allowed_points_values(state, scoped_candidates)
    for m_pts in _POINTS_VALUE_RE.finditer(text):
        prefix_vol = text[max(0, m_pts.start() - 6) : m_pts.start()].lower()
        if prefix_vol.endswith("vol "):
            continue
        p_val = float(m_pts.group(1).replace(",", ""))
        checked += 1
        if not any(abs(p_val - ap) <= 0.55 for ap in allowed_pts):
            errors.append(f"{field_name}:WRONG_POINTS_VALUE:{p_val}")

    return checked


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

    allowed_upper_words = _allowed_upper_words_for_state(state)

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
        checked += _check_trade_directives_quantities_and_symbols(
            field_name="digest",
            text=text,
            state=state,
            scoped_candidates=[],
            selected_primary=None,
            allowed_upper_words=allowed_upper_words,
            errors=errors,
            is_digest=True,
        )
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

    checked += _check_trade_directives_quantities_and_symbols(
        field_name="digest",
        text=text,
        state=state,
        scoped_candidates=[selected_primary],
        selected_primary=selected_primary,
        allowed_upper_words=allowed_upper_words,
        errors=errors,
        is_digest=True,
    )

    # Candidate-scoped Rupee check (ONLY selected_primary's rupee figures + policy caps allowed!)
    allowed_selected_rs = _allowed_rupee_numbers(state, [selected_primary])
    stripped = _STRUCT_ID_RE.sub(" ", text)
    stripped = _DATE_RE.sub(" ", stripped)

    for m_rs in _RUPEE_RE.finditer(stripped):
        raw_num = m_rs.group(1).replace(",", "")
        val = float(raw_num)
        checked += 1
        if not _rupee_matches_allowed(val, allowed_selected_rs):
            errors.append(f"digest:WRONG_RS_FOR_SELECTED:Rs {raw_num}")

    # Label-bound Rupee checks inside digest (catches swapping margin vs max loss on same candidate!)
    for pat, expected_val, tag in (
        (
            r"(?:max(?:imum)?\s+loss|contractual(?:\s+max(?:imum)?)?\s+(?:loss|bound|cap))\s*(?:of|is|=|:)?\s*"
            + _RUPEE_PREFIX_PAT
            + r"([0-9,]+(?:\.[0-9]+)?)",
            selected_primary.contractual_max_loss_rs,
            "MAX_LOSS",
        ),
        (
            r"(?<!summed\s)(?<!portfolio\s)(?:managed\s+stop|planned\s+stop|stop)\s*(?:of|is|=|:)?\s*"
            + _RUPEE_PREFIX_PAT
            + r"([0-9,]+(?:\.[0-9]+)?)",
            selected_primary.planned_stop_rs,
            "PLANNED_STOP",
        ),
        (
            r"(?:summed\s+(?:portfolio\s+)?stop|portfolio\s+summed\s+stop)\s*(?:of|is|=|:)?\s*"
            + _RUPEE_PREFIX_PAT
            + r"([0-9,]+(?:\.[0-9]+)?)",
            selected_primary.portfolio_summed_stop_after_trade_rs,
            "SUMMED_STOP",
        ),
        (
            r"margin(?:\s+estimate)?\s*(?:of|is|=|:|~)?\s*~?\s*"
            + _RUPEE_PREFIX_PAT
            + r"([0-9,]+(?:\.[0-9]+)?)",
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
            r"(?:contractual\s+(?:max(?:imum)?\s+loss|payoff\s+bound|bound|cap)|max(?:imum)?\s+loss)\s*(?:of|is|=|:)?\s*"
            + _RUPEE_PREFIX_PAT
            + r"([0-9,]+(?:\.[0-9]+)?)",
            lambda c: [c.contractual_max_loss_rs],
            1.05,
            "CONTRACTUAL_MAX_LOSS",
        ),
        (
            r"(?:managed|planned)\s+stop\s*(?:of|is|=|:)?\s*"
            + _RUPEE_PREFIX_PAT
            + r"([0-9,]+(?:\.[0-9]+)?)",
            lambda c: [c.planned_stop_rs, 2000.0],
            1.05,
            "PLANNED_STOP",
        ),
        (
            r"(?:portfolio\s+summed\s+stop|summed\s+portfolio\s+stop)(?:\s+after\s+trade|\s+post-trade)?\s*(?:of|is|=|:)?\s*"
            + _RUPEE_PREFIX_PAT
            + r"([0-9,]+(?:\.[0-9]+)?)",
            lambda c: [c.portfolio_summed_stop_after_trade_rs],
            1.05,
            "PORTFOLIO_SUMMED_STOP",
        ),
        (
            r"\bmargin(?:\s+estimate)?\s*(?:of|is|=|:|~)?\s*~?\s*"
            + _RUPEE_PREFIX_PAT
            + r"([0-9,]+(?:\.[0-9]+)?)",
            lambda c: [c.margin_estimate_rs],
            1.05,
            "MARGIN_ESTIMATE",
        ),
        (
            r"\bmax\s+profit\s*(?:of|is|=|:)?\s*"
            + _RUPEE_PREFIX_PAT
            + r"([0-9,]+(?:\.[0-9]+)?)",
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
            r"(?:overnight\s+gap[^\n]{0,45}?stress[^\n]{0,45}?|gap-stress\s+alternative[^\n]{0,45}?)"
            + _RUPEE_PREFIX_PAT
            + r"([0-9,]+(?:\.[0-9]+)?)",
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
    allowed_upper_words = _allowed_upper_words_for_state(state)

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

            checked_count += _check_trade_directives_quantities_and_symbols(
                field_name=field_name,
                text=sent,
                state=state,
                scoped_candidates=scoped_cands,
                selected_primary=selected_primary,
                allowed_upper_words=allowed_upper_words,
                errors=uncited,
                is_digest=False,
            )

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
            allowed_rs = _allowed_rupee_numbers(state, scoped_cands)

            stripped = _STRUCT_ID_RE.sub(" ", sent)
            stripped = _DATE_RE.sub(" ", stripped)

            for m_rs in _RUPEE_RE.finditer(stripped):
                raw_num = m_rs.group(1).replace(",", "")
                try:
                    val = float(raw_num)
                except ValueError:
                    continue
                checked_count += 1
                if not _rupee_matches_allowed(val, allowed_rs):
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
    debate_dominant_regime: str | None = None,
) -> tuple[list[LLMClaimCitation], VerifierReport]:
    """Deterministically verify citations, all delivered prose fields, supplied digest, and Part A."""
    state_dict = state.to_dict()
    verified_citations: list[LLMClaimCitation] = []
    unverified_count = 0
    failure_reasons: list[str] = []

    for item in raw_citations:
        field_path = str(item.get("state_field_path", "")).strip()
        cited_val = item.get("cited_value")

        found, actual_val = resolve_state_path(state_dict, field_path)
        if not found:
            unverified_count += 1
            note = f"Path {field_path!r} does not exist in NiftyOptionsState."
            failure_reasons.append(note)
            verified_citations.append(
                LLMClaimCitation(
                    claim_text=f"Unverified `{field_path}`",
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
                claim_text=(
                    f"Verified `{field_path}` = {actual_val!r}"
                    if matched
                    else f"Mismatched `{field_path}`"
                ),
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

    for idx_c, item in enumerate(raw_citations):
        ctext = str(item.get("claim_text", "")).strip()
        if ctext:
            merged_prose_fields[f"citation_claim_{idx_c}"] = ctext

    prose_checked, uncited_prose = _scan_prose_fields_semantically(
        state=state,
        state_dict=state_dict,
        prose_fields=merged_prose_fields,
        selected_primary=selected_primary,
        expected_is_live_ready=expected_is_live_ready,
    )

    if selected_primary is not None and debate_dominant_regime is not None:
        dom_up = debate_dominant_regime.strip().upper()
        prose_checked += 1
        if dom_up == "CONTESTED_WAIT":
            uncited_prose.append("debate_state:DOMINANT_REGIME_CONTESTED_WAIT_ON_SUGGESTION")
        elif (
            (selected_primary.direction == "BULLISH" and dom_up == "BEARISH")
            or (selected_primary.direction == "BEARISH" and dom_up == "BULLISH")
        ):
            uncited_prose.append(
                f"debate_state:DOMINANT_REGIME_CONTRADICTS_SELECTED:{dom_up}!=selected:{selected_primary.direction}"
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
