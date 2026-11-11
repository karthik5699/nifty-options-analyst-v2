"""Typed, JSON-serialisable shared state object (NiftyOptionsState) per Section 2.2."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

from nifty_analyst.scrubber import ensure_ist

T = TypeVar("T")

RuleStatus = Literal["PASS", "FAIL", "NOT_APPLICABLE", "UNKNOWN"]
VerdictType = Literal["SUGGESTION", "WAIT", "INSUFFICIENT DATA"]
DetectorTrack = Literal["PART_C_HISTORICAL", "NOVEL_QUANT"]
CandleBasis = Literal["COMPLETED_DAILY_CLOSE", "COMPLETED_1H_CLOSE"]
DirectionType = Literal["BULLISH", "BEARISH", "NEUTRAL"]
StructureKind = Literal[
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
]


@dataclass(frozen=True)
class TimestampedDatum(Generic[T]):
    """Envelope required by Section 3 for every ingested market datum."""

    value: T
    as_of_ist: str
    fetched_at_ist: str
    source_id: str
    max_staleness_seconds: int
    is_stale: bool
    verification_tag: str = "VERIFIED"

    @classmethod
    def create(
        cls,
        value: T,
        as_of_dt: dt.datetime,
        fetched_at_dt: dt.datetime,
        source_id: str,
        max_staleness_seconds: int,
        verification_tag: str = "VERIFIED",
    ) -> TimestampedDatum[T]:
        as_of_ist = ensure_ist(as_of_dt)
        fetched_ist = ensure_ist(fetched_at_dt)
        age_seconds = (fetched_ist - as_of_ist).total_seconds()
        is_stale = age_seconds > max_staleness_seconds or age_seconds < -5.0
        return cls(
            value=value,
            as_of_ist=as_of_ist.isoformat(),
            fetched_at_ist=fetched_ist.isoformat(),
            source_id=source_id,
            max_staleness_seconds=max_staleness_seconds,
            is_stale=is_stale,
            verification_tag=verification_tag,
        )


@dataclass(frozen=True)
class RunMetadata:
    run_id: str
    decision_timestamp_ist: str
    mode: Literal["OFFLINE", "LIVE_MORNING_SNAPSHOT"]
    data_version_hash: str
    model_ids: list[str]
    config_hash: str
    scheduled_slot_approved: bool
    snapshot_id: str = ""
    evidence_sha256: str = ""
    capture_started_at_ist: str = ""
    capture_completed_at_ist: str = ""
    surface_coverage_status: Literal["MULTI_EXPIRY_FULL", "SINGLE_EXPIRY_DEGRADED"] = (
        "MULTI_EXPIRY_FULL"
    )
    uncovered_held_neo_symbols: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExpiryMetadata:
    expiry_date: str
    expiry_timestamp_ist: str
    dte_trading_sessions: int
    lot_size: int
    is_holiday_adjusted: bool
    holiday_note: str = ""
    is_monthly: bool = False


@dataclass(frozen=True)
class ContractContext:
    symbol: str
    spot: TimestampedDatum[float]
    active_expiries: list[ExpiryMetadata]
    lot_size_change_detected: bool = False
    calendar_discrepancy_warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class OpenPosition:
    position_id: str
    structure_type: str
    directional_sign: Literal[1, 0, -1]  # +1 bullish, 0 neutral, -1 bearish
    planned_stop_rs: float
    current_executable_spread_value_rs: float
    short_leg_deltas: list[float]
    dte_trading_sessions: int
    neo_symbols: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PortfolioExposureState:
    """Tracks open positions and the Rs 10,000 portfolio summed-stop cap (Section 4.3, 5A.3)."""

    open_positions: list[OpenPosition]
    summed_planned_stops_rs: float
    remaining_stop_budget_rs: float
    directional_exposure_groups: dict[str, list[str]]  # Grouping for exposure report only; never netted
    portfolio_state_status: Literal["VERIFIED", "PORTFOLIO_STATE_UNKNOWN"] = "VERIFIED"
    portfolio_state_error: str = ""


@dataclass(frozen=True)
class CandleBar:
    timeframe: Literal["1D", "1H", "15M"]
    open_time_ist: str
    close_time_ist: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    is_final_15m_bar: bool = False


@dataclass(frozen=True)
class VerifiedTouch:
    level_name: str
    level_value: float
    bar_timeframe: str
    bar_close_time_ist: str
    bar_low: float
    bar_high: float
    bar_close: float
    touched_geometrically: bool
    wick_rejection_direction: Literal["BULLISH", "BEARISH", "NONE"]


@dataclass(frozen=True)
class PriceStructureEvidence:
    trend_1d_state: Literal["UPTREND", "DOWNTREND", "RANGE", "DOWNTREND_REPAIRED_HH_HL"]
    higher_high_higher_low_repaired: bool
    first_pullback_in_downtrend_warning: bool
    ma20_1h: float
    ema50_1h: float
    last_completed_1d_bar: CandleBar | None
    last_completed_1h_bar: CandleBar | None
    final_15m_bar_completed: bool
    stripped_forming_bars_count: int
    verified_level_touches: list[VerifiedTouch]
    rsi_14_1h: float = 50.0
    atr_14_1h: float = 0.0
    atr_14_1d: float = 0.0
    stretch_zscore_1h: float = 0.0


@dataclass(frozen=True)
class VolatilityEvidence:
    india_vix: TimestampedDatum[float | None]
    atm_iv_by_expiry: dict[str, float]
    constant_maturity_30d_iv: float
    eod_cm30_latest: float | str  # latest completed EOD cm30 (the percentiled quantity); "UNKNOWN" if stale/missing
    iv_source_label: Literal[
        "LIVE_MID_INFERRED",
        "EOD_CLOSE_INFERRED",
        "LIVE_INTRADAY_SPOT_ATM_NEAREST_EXPIRY",
        "LIVE_VARIANCE_INTERPOLATED_CM30",
    ]
    ivp_30d: float | Literal["UNKNOWN"]
    ivp_90d: float | Literal["UNKNOWN"]
    ivp_252d: float | Literal["UNKNOWN"]
    iv_history_sessions_count: int
    iv_history_gaps_flagged: list[str]
    realized_vol_20d: float
    iv_rv_spread: float
    skew_25d_put_minus_call: float
    wing_curvature: float
    chain_qa_passed: bool
    chain_qa_notes: list[str]
    term_structure_slope_near_minus_far: float = 0.0
    term_structure_regime: Literal["CONTANGO", "BACKWARDATION", "FLAT", "SINGLE_EXPIRY"] = (
        "SINGLE_EXPIRY"
    )


@dataclass(frozen=True)
class PositioningEvidence:
    target_expiry: str
    pcr_oi: float
    pcr_volume: float
    call_oi_wall_strike: float
    call_oi_wall_contracts: int
    put_oi_wall_strike: float
    put_oi_wall_contracts: int
    max_pain_strike: float
    intraday_spot_change_pts: float | None = None
    intraday_atm_iv_change_pts: float | None = None
    intraday_pcr_oi_change: float | None = None
    prior_intraday_runs_count: int = 0


@dataclass(frozen=True)
class CalendarEvent:
    event_date: str
    event_name: str
    category: str
    impact: Literal["HIGH", "MEDIUM", "LOW"]


@dataclass(frozen=True)
class EventCalendarEvidence:
    events_within_dte: list[CalendarEvent]
    calendar_file_age_days: float
    is_calendar_stale_warning: bool
    source_verification_tag: str = "UNVERIFIED"


@dataclass(frozen=True)
class TriggeredDetector:
    detector_id: str
    detector_name: str
    track: DetectorTrack
    confirmation_status: Literal[
        "HISTORICAL_REFERENCE_UNCONFIRMED",
        "CONFIRMED_QUANT_REGIME",
        "DEFINITION_OPEN",
    ]
    direction: DirectionType
    invalidation_level: float
    invalidation_candle_basis: CandleBasis
    invalidation_justification: str
    summary: str
    evidence_fields_cited: list[str]


@dataclass(frozen=True)
class MarketEvidence:
    price_structure: PriceStructureEvidence
    volatility_state: VolatilityEvidence
    derivatives_positioning: PositioningEvidence
    event_calendar: EventCalendarEvidence
    triggered_detectors: list[TriggeredDetector]


@dataclass(frozen=True)
class OptionLeg:
    strike: float
    option_type: Literal["CE", "PE"]
    side: Literal["LONG", "SHORT"]
    ratio: int
    bid: float
    ask: float
    mid: float
    iv: float
    iv_bid: float
    iv_ask: float
    delta: float
    gamma: float
    theta_daily: float
    vega_per_vol_pt: float
    oi: int
    volume: int
    top_book_depth_lots: int
    quote_timestamp_ist: str
    is_stale: bool
    liquidity_score: float
    touch_available_qty: int | None = None
    visible_depth_qty: int | None = None
    book_levels: list[list[float]] = field(default_factory=list)
    vwap_fill_price: float | None = None
    executable_entry_price: float | None = None
    fill_basis: Literal["FULL_SIZE_TOUCH", "FULL_SIZE_BOOK_WALK", "INDICATIVE_TOUCH_ONLY"] = (
        "FULL_SIZE_TOUCH"
    )


@dataclass(frozen=True)
class PartARuleCheck:
    rule_id: str
    rule_name: str
    source_date: str
    status: RuleStatus
    evidence: str


@dataclass(frozen=True)
class StatutoryChargesBreakdown:
    """Round-trip non-STT execution charges (exchange, SEBI, stamp duty, brokerage, GST).

    STT is intentionally excluded per trader directive so technical analysis and options
    surface analytics drive strategy selection rather than tax artifacts.
    """

    exchange_txn_charges_rs: float
    sebi_charges_rs: float
    stamp_duty_rs: float
    gst_rs: float
    brokerage_rs: float
    total_charges_rs: float


@dataclass(frozen=True)
class CandidateStructure:
    structure_id: str
    structure_type: StructureKind
    direction: DirectionType
    expiry_date: str
    dte_trading_sessions: int
    lot_size: int
    lots: int
    width_points: float
    legs: list[OptionLeg]

    # Executable entry & payoff metrics (Section 4.3)
    is_net_credit: bool
    executable_net_points_per_unit: float  # positive = credit received or debit paid per unit
    slippage_buffer_points_per_unit: float
    statutory_charges: StatutoryChargesBreakdown

    # The Four Distinct Rupee Quantities (Section 4.3, 6.3, 7.4) - NEVER MERGED
    contractual_max_loss_rs: float  # exact leg-wise piecewise expiry payoff bound
    contractual_max_loss_method: str
    margin_estimate_rs: float
    margin_source: Literal["EXCHANGE_FORMULA_ESTIMATE", "BROKER_CALCULATOR_UNVERIFIED"]
    planned_stop_rs: float
    indicative_stop_spread_value_per_unit: float
    indicative_stop_index_level: float
    indicative_stop_note: str
    portfolio_summed_stop_after_trade_rs: float

    # Breakevens, PoP, Greeks, Liquidity
    max_profit_rs: float
    breakevens_gross: list[float]
    breakevens_net: list[float]
    pop_lognormal_expiry: float
    pop_empirical_expiry: float
    pop_assumptions_label: str
    net_delta: float
    net_gamma: float
    net_theta_daily_rs: float
    net_vega_rs: float
    short_leg_max_abs_delta: float
    structure_liquidity_score: float
    overnight_gap_stress_3pct_loss_rs: float

    # Written invalidation & management plan
    invalidation_level: float
    invalidation_candle_basis: CandleBasis
    invalidation_justification: str
    management_plan: str

    # Part A, Part B & technical/options ranking score
    part_a_checks: dict[str, PartARuleCheck]
    part_a_overall_status: RuleStatus
    matches_part_b_style: bool
    technical_options_score: float = 0.0
    execution_fill_basis: Literal[
        "FULL_SIZE_TOUCH", "FULL_SIZE_BOOK_WALK", "INDICATIVE_TOUCH_ONLY"
    ] = "FULL_SIZE_TOUCH"
    full_size_fill_supported: bool = True
    full_size_executable_net_points_per_unit: float | None = None
    full_size_contractual_max_loss_rs: float | None = None
    indicative_touch_net_points_per_unit: float = 0.0
    indicative_touch_max_loss_rs: float = 0.0
    price_assumption_label: str = ""


@dataclass(frozen=True)
class DebateState:
    """TradingAgents-inspired structured adversarial Bull / Bear / Volatility-Regime debate."""

    bull_thesis: str
    bear_thesis: str
    vol_regime_thesis: str = ""
    rebuttal_round: str = ""
    dominant_regime: Literal["BULLISH", "BEARISH", "NEUTRAL", "CONTESTED_WAIT"] = "NEUTRAL"


@dataclass(frozen=True)
class RiskTriadState:
    """TradingAgents-inspired 3-perspective options risk evaluation on Part-A-PASS candidates."""

    convexity_advocate_note: str
    tail_guardian_note: str
    neutral_arbiter_note: str


@dataclass(frozen=True)
class ConsolidatorState:
    """Chief Options Strategist persona synthesizing the debate + risk triad into a final verdict."""

    recommended_structure_id: str  # Certified Part-A-PASS structure_id or "WAIT"
    conviction: Literal["HIGH", "MODERATE", "SIT_OUT"]
    final_recommendation_rationale: str
    whatsapp_digest: str


@dataclass(frozen=True)
class LLMClaimCitation:
    claim_text: str
    state_field_path: str
    cited_value: float | str | bool
    verified_in_state: bool
    verification_note: str = ""


@dataclass(frozen=True)
class VerifierReport:
    total_claims_checked: int
    unverified_claims_count: int
    unsupported_claim_rate: float
    part_a_reverified: bool
    passed_verification: bool
    failure_reasons: list[str]
    prose_numbers_checked: int = 0
    uncited_prose_numbers: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FinalOutput:
    verdict: VerdictType
    is_live_ready: bool
    research_only_reason: str
    primary_proposal: CandidateStructure | None
    alternative_candidates: list[CandidateStructure]
    habitual_style_comparison_note: str
    reasoning_summary: str
    part_a_summary_status: RuleStatus
    research_rule_challenge_note: str
    confidence_note: str
    staleness_report: list[str]
    llm_citations: list[LLMClaimCitation]
    verifier_report: VerifierReport
    whatsapp_digest: str = ""
    duplicate_alert_suppressed: bool = False
    payoff_diagram_path: str = ""


@dataclass(frozen=True)
class NiftyOptionsState:
    """Top-level shared state object passed across all layers."""

    run_metadata: RunMetadata
    contract_context: ContractContext
    portfolio_exposure_state: PortfolioExposureState
    market_evidence: MarketEvidence
    candidate_structures: list[CandidateStructure]
    debate_state: DebateState | None
    final_output: FinalOutput | None = None
    risk_triad_state: RiskTriadState | None = None
    consolidator_state: ConsolidatorState | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)
