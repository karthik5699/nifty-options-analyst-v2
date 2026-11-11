"""Part A encoding tests, TradingAgents single-pass persona synthesis & Consolidator tests,
two-way prose hallucination verifier tests, secret scrubber tests, intraday alert deduplication,
and end-to-end offline pipeline tests (Sections 8.3, 8.4, 9, 10).
"""

from __future__ import annotations

import ast
import dataclasses
import datetime as dt
import json
import os
import shutil
from pathlib import Path

import pytest

from nifty_analyst.config import AnalystConfig
from nifty_analyst.engine.part_a_gatekeeper import evaluate_part_a_rules
from nifty_analyst.engine.pricing import LegPayoffSpec, compute_piecewise_expiry_payoff
from nifty_analyst.pipeline import (
    build_portfolio_exposure_state,
    run_analyst_pipeline,
)
from nifty_analyst.scrubber import IST, SecretScrubber, load_chmod600_env_file
from nifty_analyst.state import (
    CandleBar,
    OpenPosition,
    OptionLeg,
    PriceStructureEvidence,
    TimestampedDatum,
    VolatilityEvidence,
)
from nifty_analyst.synthesis.llm_node import (
    AgentArtifactSynthesisProvider,
    HostedBudgetGuardedLLMProvider,
    PaidLLMSpendNotAuthorizedError,
)
from nifty_analyst.synthesis.verifier import verify_llm_synthesis

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = REPO_ROOT / "fixtures"


def _sample_leg(
    strike: float,
    option_type: str,
    side: str,
    delta: float,
    liquidity_score: float = 85.0,
) -> OptionLeg:
    return OptionLeg(
        strike=strike,
        option_type=option_type,  # type: ignore[arg-type]
        side=side,  # type: ignore[arg-type]
        ratio=1,
        bid=80.0,
        ask=81.0,
        mid=80.5,
        iv=0.14,
        iv_bid=0.139,
        iv_ask=0.141,
        delta=delta,
        gamma=0.001,
        theta_daily=-6.5,
        vega_per_vol_pt=12.0,
        oi=120000,
        volume=35000,
        top_book_depth_lots=20,
        quote_timestamp_ist="2026-11-06T10:14:35+05:30",
        is_stale=False,
        liquidity_score=liquidity_score,
    )


def _sample_price_and_vol(
    ivp_90d: float | str = 72.0,
) -> tuple[PriceStructureEvidence, VolatilityEvidence]:
    now = dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST)
    bar_1d = CandleBar(
        "1D", "2026-11-05T09:15:00+05:30", "2026-11-05T15:30:00+05:30", 24100, 24240, 24020, 24190
    )
    bar_1h = CandleBar(
        "1H", "2026-11-06T09:15:00+05:30", "2026-11-06T10:15:00+05:30", 24160, 24215, 24095, 24200
    )
    ps = PriceStructureEvidence(
        trend_1d_state="DOWNTREND_REPAIRED_HH_HL",
        higher_high_higher_low_repaired=True,
        first_pullback_in_downtrend_warning=False,
        ma20_1h=24115.0,
        ema50_1h=24060.0,
        last_completed_1d_bar=bar_1d,
        last_completed_1h_bar=bar_1h,
        final_15m_bar_completed=True,
        stripped_forming_bars_count=1,
        verified_level_touches=[],
    )
    vol = VolatilityEvidence(
        india_vix=TimestampedDatum.create(14.25, now, now, "vix", 180, "UNVERIFIED"),
        atm_iv_by_expiry={"2026-11-12": 0.145},
        constant_maturity_30d_iv=0.145,
        eod_cm30_latest=0.1401,
        iv_source_label="LIVE_INTRADAY_SPOT_ATM_NEAREST_EXPIRY",
        ivp_30d=70.0 if ivp_90d != "UNKNOWN" else "UNKNOWN",
        ivp_90d=ivp_90d,  # type: ignore[arg-type]
        ivp_252d="UNKNOWN",
        iv_history_sessions_count=95,
        iv_history_gaps_flagged=[],
        realized_vol_20d=0.118,
        iv_rv_spread=0.027,
        skew_25d_put_minus_call=0.015,
        wing_curvature=0.004,
        chain_qa_passed=True,
        chain_qa_notes=["QA passed"],
    )
    return ps, vol


def test_part_a_1dte_short_delta_rule_on_debit_spread_and_full_stop_summing() -> None:
    """Verify Section 8.3:
    1. 1 DTE |delta| < 0.25 rule applied leg-by-leg to a Bull Call DEBIT spread's short call leg.
    2. Full summing of correlated same-direction positions' stops without netting.
    3. Three separate money policies (Rs 2,000 stop, Rs 10,000 summed stop, Rs 10,000 contractual max loss).
    4. PASS / FAIL / NOT_APPLICABLE / UNKNOWN states.
    """
    cfg = AnalystConfig(fixtures_dir=str(FIXTURES_DIR))
    ps, vol = _sample_price_and_vol(ivp_90d=72.0)

    debit_legs_high_short_delta = [
        _sample_leg(24200.0, "CE", "LONG", delta=0.52),
        _sample_leg(24400.0, "CE", "SHORT", delta=0.29),
    ]

    # Case A: At 4 DTE sessions, Rule 5 is NOT_APPLICABLE
    checks_4dte, overall_4dte = evaluate_part_a_rules(
        is_defined_risk=True,
        legs=debit_legs_high_short_delta,
        dte_trading_sessions=4,
        planned_stop_rs=2000.0,
        portfolio_state=build_portfolio_exposure_state([]),
        contractual_max_loss_rs=6000.0,
        invalidation_level=24050.0,
        invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
        invalidation_justification="Daily close below 24050",
        price_structure=ps,
        volatility=vol,
        config=cfg,
    )
    assert checks_4dte["RULE_5_EXPIRY_1DTE_SHORT_DELTA"].status == "NOT_APPLICABLE"
    assert overall_4dte == "PASS"

    # Case B: At 1 DTE session, short call |delta|=0.29 >= 0.25 -> FAIL Rule 5
    checks_1dte_fail, overall_1dte_fail = evaluate_part_a_rules(
        is_defined_risk=True,
        legs=debit_legs_high_short_delta,
        dte_trading_sessions=1,
        planned_stop_rs=2000.0,
        portfolio_state=build_portfolio_exposure_state([]),
        contractual_max_loss_rs=6000.0,
        invalidation_level=24050.0,
        invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
        invalidation_justification="Daily close below 24050",
        price_structure=ps,
        volatility=vol,
        config=cfg,
    )
    assert checks_1dte_fail["RULE_5_EXPIRY_1DTE_SHORT_DELTA"].status == "FAIL"
    assert overall_1dte_fail == "FAIL"

    # Case C: At 1 DTE session, short call |delta|=0.19 < 0.25 -> PASS Rule 5
    debit_legs_low_short_delta = [
        _sample_leg(24200.0, "CE", "LONG", delta=0.52),
        _sample_leg(24500.0, "CE", "SHORT", delta=0.19),
    ]
    checks_1dte_pass, overall_1dte_pass = evaluate_part_a_rules(
        is_defined_risk=True,
        legs=debit_legs_low_short_delta,
        dte_trading_sessions=1,
        planned_stop_rs=2000.0,
        portfolio_state=build_portfolio_exposure_state([]),
        contractual_max_loss_rs=6000.0,
        invalidation_level=24050.0,
        invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
        invalidation_justification="Daily close below 24050",
        price_structure=ps,
        volatility=vol,
        config=cfg,
    )
    assert checks_1dte_pass["RULE_5_EXPIRY_1DTE_SHORT_DELTA"].status == "PASS"
    assert overall_1dte_pass == "PASS"

    # Case D: Correlated positions full stop summing without netting
    correlated_positions = [
        OpenPosition(f"pos_{i}", "BULL_PUT_SPREAD", 1, 2000.0, 45.0, [-0.25], 4)
        for i in range(5)
    ]
    port_full = build_portfolio_exposure_state(correlated_positions, portfolio_cap_rs=10000.0)
    assert port_full.summed_planned_stops_rs == 10000.0
    checks_port_fail, overall_port_fail = evaluate_part_a_rules(
        is_defined_risk=True,
        legs=debit_legs_low_short_delta,
        dte_trading_sessions=4,
        planned_stop_rs=2000.0,
        portfolio_state=port_full,
        contractual_max_loss_rs=6000.0,
        invalidation_level=24050.0,
        invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
        invalidation_justification="Daily close below 24050",
        price_structure=ps,
        volatility=vol,
        config=cfg,
    )
    assert checks_port_fail["RULE_3_PORTFOLIO_SUMMED_STOP_10000_RS"].status == "FAIL"
    assert overall_port_fail == "FAIL"

    # Case E: Contractual max loss > Rs 10,000 ceiling fails Rule 4
    checks_cmax_fail, _ = evaluate_part_a_rules(
        is_defined_risk=True,
        legs=debit_legs_low_short_delta,
        dte_trading_sessions=4,
        planned_stop_rs=2000.0,
        portfolio_state=build_portfolio_exposure_state([]),
        contractual_max_loss_rs=10500.0,
        invalidation_level=24050.0,
        invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
        invalidation_justification="Daily close below 24050",
        price_structure=ps,
        volatility=vol,
        config=cfg,
    )
    assert checks_cmax_fail["RULE_2_PER_POSITION_STOP_2000_RS"].status == "PASS"
    assert checks_cmax_fail["RULE_4_CONTRACTUAL_MAX_LOSS_10000_RS"].status == "FAIL"

    # Case F: IVP UNKNOWN returns UNKNOWN on Rule 8
    _, vol_unknown = _sample_price_and_vol(ivp_90d="UNKNOWN")
    checks_unk, overall_unk = evaluate_part_a_rules(
        is_defined_risk=True,
        legs=debit_legs_low_short_delta,
        dte_trading_sessions=4,
        planned_stop_rs=2000.0,
        portfolio_state=build_portfolio_exposure_state([]),
        contractual_max_loss_rs=6000.0,
        invalidation_level=24050.0,
        invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
        invalidation_justification="Daily close below 24050",
        price_structure=ps,
        volatility=vol_unknown,
        config=cfg,
    )
    assert checks_unk["RULE_8_IVP_AND_LIQUIDITY_GATES"].status == "UNKNOWN"
    assert overall_unk == "UNKNOWN"


def test_end_to_end_offline_pipeline_and_repeat_stability(tmp_path: Path) -> None:
    """Verify end-to-end offline execution, alert-only delivery, audit log, and 5-trial stability."""
    audit_file = tmp_path / "audit_log.jsonl"
    cfg = AnalystConfig(
        mode="OFFLINE",
        fixtures_dir=str(FIXTURES_DIR),
        audit_log_path=str(audit_file),
    )
    dec_dt = dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST)

    verdicts: list[str] = []
    primary_ids: list[str] = []
    for _ in range(5):
        state, delivery = run_analyst_pipeline(dec_dt, cfg)
        assert state.final_output is not None
        verdicts.append(state.final_output.verdict)
        assert state.final_output.primary_proposal is not None
        primary_ids.append(state.final_output.primary_proposal.structure_id)
        assert delivery.alerted_trader is True
        assert "The Four Distinct Rupee Quantities" in delivery.user_facing_message
        assert delivery.whatsapp_digest.startswith("*NIFTY Options Desk — SUGGESTION")
        assert state.final_output.verifier_report.passed_verification is True
        assert state.final_output.verifier_report.unsupported_claim_rate == 0.0

    assert len(set(verdicts)) == 1 and verdicts[0] == "SUGGESTION"
    assert len(set(primary_ids)) == 1
    assert len(audit_file.read_text(encoding="utf-8").strip().splitlines()) == 5

    bad_citations = [
        {
            "claim_text": "Confabulated spot price",
            "state_field_path": "contract_context.spot.value",
            "cited_value": 99999.0,
        }
    ]
    _, bad_rep = verify_llm_synthesis(state, bad_citations, state.final_output.primary_proposal)
    assert bad_rep.passed_verification is False
    assert bad_rep.unverified_claims_count == 1


def test_alert_only_silence_on_wait_and_insufficient_data(tmp_path: Path) -> None:
    """Verify WAIT and INSUFFICIENT DATA write to audit log with empty user_facing_message."""
    audit_file = tmp_path / "audit_wait.jsonl"
    cfg = AnalystConfig(
        mode="OFFLINE",
        fixtures_dir=str(FIXTURES_DIR),
        audit_log_path=str(audit_file),
    )
    stale_dt = dt.datetime(2026, 11, 6, 10, 25, 0, tzinfo=IST)
    state, delivery = run_analyst_pipeline(stale_dt, cfg, force_render_for_inspection=False)
    assert state.final_output is not None
    assert state.final_output.verdict == "INSUFFICIENT DATA"
    assert delivery.alerted_trader is False
    assert delivery.user_facing_message == ""
    assert delivery.whatsapp_digest == ""
    assert audit_file.exists()


def test_secret_scrubber_chmod600_and_paid_llm_guard(tmp_path: Path) -> None:
    """Verify secret scrubbing, OptionLeg.side preservation, chmod-600 check, and paid LLM guard."""
    scrubber = SecretScrubber(explicit_secrets=["SUPER_SECRET_KOTAK_TOKEN_12345"])
    dirty = {
        "kotak_session_token": "SUPER_SECRET_KOTAK_TOKEN_12345",
        "kotak_sid": "my_secret_sid",
        "pan": "ABCDE1234F",
        "side": "SHORT",
        "calendar_discrepancy_warnings": [" shifted expiry "],
        "note": "Contact +91 9876543210 or PAN ABCDE1234F with token SUPER_SECRET_KOTAK_TOKEN_12345",
    }
    cleaned = scrubber.scrub_obj(dirty)
    assert cleaned["kotak_session_token"] == "[REDACTED_KEY_VALUE]"
    assert cleaned["kotak_sid"] == "[REDACTED_KEY_VALUE]"
    assert cleaned["pan"] == "[REDACTED_KEY_VALUE]"
    # Critical regression fix: non-secret keys containing 'sid' or 'pan' substrings MUST NOT be redacted!
    assert cleaned["side"] == "SHORT"
    assert cleaned["calendar_discrepancy_warnings"] == [" shifted expiry "]
    assert "SUPER_SECRET_KOTAK_TOKEN_12345" not in cleaned["note"]
    assert "9876543210" not in cleaned["note"]
    assert "ABCDE1234F" not in cleaned["note"]

    ext_file = tmp_path / "external_secrets.env"
    ext_file.write_text("KOTAK_SID=test_sid_val\n", encoding="utf-8")
    os.chmod(ext_file, 0o644)
    with pytest.raises(PermissionError, match="insecure permissions"):
        load_chmod600_env_file(ext_file, repo_root=REPO_ROOT)

    os.chmod(ext_file, 0o600)
    loaded = load_chmod600_env_file(ext_file, repo_root=REPO_ROOT)
    assert loaded["KOTAK_SID"] == "test_sid_val"

    cfg_blocked = AnalystConfig(llm_provider="hosted_api", allow_paid_llm_spend=False)
    provider = HostedBudgetGuardedLLMProvider(cfg_blocked)
    with pytest.raises(PaidLLMSpendNotAuthorizedError):
        provider.synthesize("{}", None, None, [])  # type: ignore[arg-type]


def test_no_broker_write_or_backtest_code_in_source_tree() -> None:
    """Verify no order placement/modification or backtesting functions exist anywhere in src/."""
    src_dir = REPO_ROOT / "src" / "nifty_analyst"
    prohibited_func_prefixes = (
        "place_order",
        "modify_order",
        "cancel_order",
        "execute_order",
        "run_backtest",
        "simulate_pnl",
        "compute_sharpe",
    )
    for py_file in src_dir.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fname = node.name.lower()
                assert not any(
                    fname.startswith(pref) for pref in prohibited_func_prefixes
                ), f"Prohibited function {node.name} found in {py_file}"


def test_rule4_stt_free_symmetric_contractual_bound_on_calls_and_puts(tmp_path: Path) -> None:
    """Verify STT is completely removed and call-containing structures (Bull Call, Bear Call,
    Iron Condor, Butterflies) achieve Part A PASS on equal footing with Put structures.
    """
    cfg = AnalystConfig(
        mode="OFFLINE",
        fixtures_dir=str(FIXTURES_DIR),
        audit_log_path=str(tmp_path / "audit_stt_free.jsonl"),
    )
    ps, vol = _sample_price_and_vol(ivp_90d=72.0)

    call_specs = [
        LegPayoffSpec(strike=24200.0, option_type="CE", side="LONG", ratio=1, executable_entry_price=140.0),
        LegPayoffSpec(strike=24400.0, option_type="CE", side="SHORT", ratio=1, executable_entry_price=60.0),
    ]
    pw_call = compute_piecewise_expiry_payoff(call_specs, lot_size=75, lots=1, statutory_charges_rs=100.0)
    assert pw_call.is_defined_risk is True
    assert pw_call.gross_max_loss_rs == 6000.0
    assert "STT" not in pw_call.method_description

    call_legs = [
        _sample_leg(24200.0, "CE", "LONG", delta=0.52),
        _sample_leg(24400.0, "CE", "SHORT", delta=0.19),
    ]
    checks_pass, overall_pass = evaluate_part_a_rules(
        is_defined_risk=True,
        legs=call_legs,
        dte_trading_sessions=4,
        planned_stop_rs=2000.0,
        portfolio_state=build_portfolio_exposure_state([]),
        contractual_max_loss_rs=pw_call.gross_max_loss_rs,
        invalidation_level=24050.0,
        invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
        invalidation_justification="Daily close below 24050",
        price_structure=ps,
        volatility=vol,
        config=cfg,
    )
    assert checks_pass["RULE_4_CONTRACTUAL_MAX_LOSS_10000_RS"].status == "PASS"
    assert overall_pass == "PASS"

    # End-to-end check: Iron Condors, Bear Call Spreads, Bull Call Spreads, and Butterflies
    # are certified PASS across active expiries when within Rs 10,000 contractual ceiling!
    state, _ = run_analyst_pipeline(dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST), cfg)
    pass_types = {
        c.structure_type for c in state.candidate_structures if c.part_a_overall_status == "PASS"
    }
    assert "BULL_PUT_SPREAD" in pass_types
    assert "BULL_CALL_SPREAD" in pass_types
    assert "BEAR_CALL_SPREAD" in pass_types
    assert "IRON_CONDOR" in pass_types
    assert "BULLISH_BUTTERFLY" in pass_types
    for c in state.candidate_structures:
        assert not hasattr(c.statutory_charges, "stt_rs")


def test_multi_expiry_personas_consolidator_and_prose_verifier(tmp_path: Path) -> None:
    """Verify multi-expiry evaluation, single-pass TradingAgents personas, Consolidator override,
    and two-way prose numeric hallucination guard.
    """
    cfg = AnalystConfig(
        mode="OFFLINE",
        fixtures_dir=str(FIXTURES_DIR),
        audit_log_path=str(tmp_path / "audit_personas.jsonl"),
    )
    dec_dt = dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST)
    state, _ = run_analyst_pipeline(dec_dt, cfg)

    # All 3 active expiries evaluated & CM30 variance-interpolated
    expiries_in_cands = {c.expiry_date for c in state.candidate_structures}
    assert expiries_in_cands == {"2026-11-12", "2026-11-19", "2026-11-23"}
    assert state.market_evidence.volatility_state.iv_source_label == "LIVE_VARIANCE_INTERPOLATED_CM30"
    assert state.debate_state is not None and state.debate_state.vol_regime_thesis != ""
    assert state.risk_triad_state is not None
    assert state.consolidator_state is not None
    assert state.consolidator_state.recommended_structure_id != "WAIT"

    # Two-way prose verifier MUST catch an invented rupee amount or index level in prose
    # even when the `citations` array itself only cites valid state fields!
    valid_citations = [
        {
            "claim_text": "Spot check",
            "state_field_path": "contract_context.spot.value",
            "cited_value": state.contract_context.spot.value,
        }
    ]
    _, prose_fail_rep = verify_llm_synthesis(
        state=state,
        raw_citations=valid_citations,
        selected_primary=state.final_output.primary_proposal if state.final_output else None,
        prose_texts=["Spot surged to 29876.5 with max loss Rs 3,219.00 on BPS_2026-11-12_99900_99700."],
    )
    assert prose_fail_rep.passed_verification is False
    assert len(prose_fail_rep.uncited_prose_numbers) >= 2


def test_same_day_intraday_deltas_and_duplicate_alert_suppression(tmp_path: Path) -> None:
    """Verify 10:15 IST alerts once and 11:15 IST computes intraday deltas + suppresses duplicate structure alert."""
    fx = tmp_path / "fixtures"
    shutil.copytree(FIXTURES_DIR, fx)
    audit_path = tmp_path / "audit_intraday.jsonl"
    cfg = AnalystConfig(
        mode="OFFLINE",
        fixtures_dir=str(fx),
        audit_log_path=str(audit_path),
        deduplicate_intraday_alerts=True,
    )

    t_1015 = dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST)
    state_1, deliv_1 = run_analyst_pipeline(t_1015, cfg)
    assert deliv_1.alerted_trader is True
    assert state_1.final_output is not None and state_1.final_output.primary_proposal is not None

    # Advance fixture timestamps by +1h to 11:14:35 IST (with identical prices so same structure wins)
    cm = json.loads((fx / "contract_master.json").read_text(encoding="utf-8"))
    cm["spot"]["as_of_ist"] = "2026-11-06T11:14:40+05:30"
    (fx / "contract_master.json").write_text(json.dumps(cm), encoding="utf-8")

    chains = json.loads((fx / "option_chains.json").read_text(encoding="utf-8"))
    for exp_data in chains["chains_by_expiry"].values():
        for row in exp_data["strikes"]:
            row["update_time_ist"] = "2026-11-06T11:14:35+05:30"
    (fx / "option_chains.json").write_text(json.dumps(chains), encoding="utf-8")

    t_1115 = dt.datetime(2026, 11, 6, 11, 15, 0, tzinfo=IST)
    state_2, deliv_2 = run_analyst_pipeline(t_1115, cfg)
    assert state_2.market_evidence.derivatives_positioning.prior_intraday_runs_count == 1
    assert state_2.market_evidence.derivatives_positioning.intraday_spot_change_pts == 0.0
    assert state_2.final_output is not None
    assert state_2.final_output.verdict == "SUGGESTION"
    assert state_2.final_output.duplicate_alert_suppressed is True
    assert deliv_2.alerted_trader is False
    assert deliv_2.user_facing_message == ""


def test_agent_artifact_provider_interface_round_trip(tmp_path: Path) -> None:
    """External WhatsApp/operator single-pass synthesis artifact seam verification."""
    dec_dt = dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST)
    artifact = tmp_path / "agent_synthesis.json"
    request = tmp_path / "agent_synthesis.json.request.json"

    cfg = AnalystConfig(
        mode="OFFLINE",
        fixtures_dir=str(FIXTURES_DIR),
        audit_log_path=str(tmp_path / "audit.jsonl"),
        llm_provider="agent_artifact",
        agent_synthesis_artifact_path=str(artifact),
    )

    # Phase A: no artifact yet -> run raises and exports compact_desk_brief + persona_instructions + sanitised_state
    with pytest.raises(FileNotFoundError):
        run_analyst_pipeline(dec_dt, cfg)
    assert request.is_file()
    req = json.loads(request.read_text(encoding="utf-8"))
    assert req["run_id"].startswith("run_20261106_101500")
    assert "compact_desk_brief" in req and "persona_instructions" in req
    evidence = req["sanitised_state"]
    assert "market_evidence" in evidence and "candidate_structures" in evidence

    # WhatsApp LLM agent selects a certified PASS Iron Condor via Consolidator persona
    pass_cards = req["compact_desk_brief"]["certified_pass_candidates"]
    ic_card = next((c for c in pass_cards if c["structure_type"] == "IRON_CONDOR"), pass_cards[0])
    idx = ic_card["state_index"]
    bound_val = ic_card["contractual_max_loss_rs"]
    ivp_val = evidence["market_evidence"]["volatility_state"]["ivp_90d"]

    artifact.write_text(
        json.dumps(
            {
                "model_id": "whatsapp-persona-agent-v1",
                "run_id": req["run_id"],
                "snapshot_id": req["snapshot_id"],
                "evidence_sha256": req["evidence_sha256"],
                "decision_timestamp_ist": req["decision_timestamp_ist"],
                "reasoning_summary": "Single-pass Bull/Bear/Vol debate and Consolidator synthesis.",
                "habitual_comparison_note": "Evaluated verticals, butterflies, and condors without STT.",
                "research_rule_challenge_note": "No research-rule challenge.",
                "confidence_note": "Confidence per verified citations only.",
                "debate_state": {
                    "bull_thesis": "Support holds above Put OI wall.",
                    "bear_thesis": "Call OI wall caps upside breakout.",
                    "vol_regime_thesis": "Elevated IVP favours short-vega wings.",
                    "rebuttal_round": "Bracketed range favours neutral condor.",
                    "dominant_regime": "NEUTRAL",
                },
                "risk_triad": {
                    "convexity_advocate_note": "Clean credit harvest on both wings.",
                    "tail_guardian_note": "Defined 150-pt wings cap tail risk.",
                    "neutral_arbiter_note": "Endorses certified Iron Condor.",
                },
                "consolidator": {
                    "recommended_structure_id": ic_card["structure_id"],
                    "conviction": "HIGH",
                    "final_recommendation_rationale": "Consolidator selects certified PASS structure.",
                    "whatsapp_digest": f"*WhatsApp Pick*: `{ic_card['structure_id']}`",
                },
                "citations": [
                    {
                        "claim_text": f"Candidate contractual payoff bound is Rs {bound_val:.2f}.",
                        "state_field_path": f"candidate_structures[{idx}].contractual_max_loss_rs",
                        "cited_value": bound_val,
                    },
                    {
                        "claim_text": f"90-session IVP stands at {ivp_val}.",
                        "state_field_path": "market_evidence.volatility_state.ivp_90d",
                        "cited_value": ivp_val,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    # Phase B: resumed run consumes the bound artifact and honours Consolidator's PASS pick!
    state2, delivery2 = run_analyst_pipeline(dec_dt, cfg)
    assert state2.final_output is not None
    assert state2.final_output.verifier_report.passed_verification is True
    assert state2.final_output.primary_proposal is not None
    assert state2.final_output.primary_proposal.structure_id == ic_card["structure_id"]
    assert delivery2.alerted_trader is True
    assert ic_card["structure_id"] in delivery2.whatsapp_digest

    resp = AgentArtifactSynthesisProvider(str(artifact)).synthesize(
        json.dumps(evidence), state2, None, []
    )
    assert resp.estimated_cost_inr == 0.0 and resp.estimated_tokens_used == 0
    assert resp.model_id == "whatsapp-persona-agent-v1"

    other_meta = dataclasses.replace(
        state2.run_metadata,
        run_id="run_20261106_111500_abcdef",
        decision_timestamp_ist="2026-11-06T11:15:00+05:30",
    )
    other_state = dataclasses.replace(state2, run_metadata=other_meta)
    with pytest.raises(ValueError, match="stale or foreign artifact"):
        AgentArtifactSynthesisProvider(str(artifact)).synthesize(
            json.dumps(evidence), other_state, None, []
        )

    tampered = json.loads(artifact.read_text(encoding="utf-8"))
    tampered["run_id"] = "run_19990101_000000_deadbeef"
    artifact.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="stale or foreign artifact"):
        run_analyst_pipeline(dec_dt, cfg)

    state_req = json.loads(request.read_text(encoding="utf-8"))
    bad_payload = {
        "model_id": "whatsapp-persona-agent-v1",
        "run_id": state_req["run_id"],
        "snapshot_id": state_req["snapshot_id"],
        "evidence_sha256": state_req["evidence_sha256"],
        "decision_timestamp_ist": state_req["decision_timestamp_ist"],
        "reasoning_summary": "x",
        "habitual_comparison_note": "x",
        "research_rule_challenge_note": "x",
        "confidence_note": "x",
        "citations": [
            {
                "claim_text": "fabricated",
                "state_field_path": "market_evidence.nope.field",
                "cited_value": 0,
            },
        ]
        * 10,
    }
    artifact.write_text(json.dumps(bad_payload), encoding="utf-8")
    state3, delivery3 = run_analyst_pipeline(dec_dt, cfg)
    assert state3.final_output is not None
    assert state3.final_output.verifier_report.passed_verification is False
    assert state3.final_output.verdict != "SUGGESTION"
    assert delivery3.alerted_trader is False

    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps({"model_id": "x"}), encoding="utf-8")
    with pytest.raises(ValueError):
        AgentArtifactSynthesisProvider(str(incomplete)).synthesize(
            json.dumps(evidence), state2, None, []
        )
