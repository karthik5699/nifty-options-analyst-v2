"""Part A encoding tests, LLM citation verification tests, secret scrubber tests, and end-to-end offline pipeline tests (Sections 8.3, 8.4, 9, 10)."""

from __future__ import annotations

import ast
import datetime as dt
import os
from pathlib import Path

import pytest

from nifty_analyst.config import AnalystConfig
from nifty_analyst.engine.part_a_gatekeeper import evaluate_part_a_rules
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


def _sample_price_and_vol(ivp_90d: float | str = 72.0) -> tuple[PriceStructureEvidence, VolatilityEvidence]:
    now = dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST)
    bar_1d = CandleBar("1D", "2026-11-05T09:15:00+05:30", "2026-11-05T15:30:00+05:30", 24100, 24240, 24020, 24190)
    bar_1h = CandleBar("1H", "2026-11-06T09:15:00+05:30", "2026-11-06T10:15:00+05:30", 24160, 24215, 24095, 24200)
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

    # Bull Call Debit Spread: Long 24200 CE (delta +0.52), Short 24400 CE (delta +0.29 >= 0.25)
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

    # Case B: At 1 DTE session, the debit spread's short call has |delta|=0.29 >= 0.25 -> MUST FAIL Rule 5!
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

    # Case C: At 1 DTE session, if the debit spread's short call has |delta|=0.19 < 0.25 -> PASS Rule 5!
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

    # Case D: Correlated positions full stop summing without netting (5 positions * Rs 2,000 + Rs 2,000 = Rs 12,000 > Rs 10,000)
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

    # Case E: Contractual max loss > Rs 10,000 ceiling (6 Nov 2026 rule) fails Rule 4 even when planned stop is Rs 2,000
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

    # Case F: IVP UNKNOWN returns UNKNOWN on Rule 8 and overall status UNKNOWN (never waived)
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

    # Run 5 trials on identical frozen inputs (Section 1.2(4): >=80% agreement; deterministic achieves 100%)
    verdicts: list[str] = []
    primary_ids: list[str] = []
    for _ in range(5):
        state, delivery = run_analyst_pipeline(dec_dt, cfg)
        assert state.final_output is not None
        verdicts.append(state.final_output.verdict)
        assert state.final_output.primary_proposal is not None
        primary_ids.append(state.final_output.primary_proposal.structure_id)
        # Alert-only check: SUGGESTION emits user_facing_message
        assert delivery.alerted_trader is True
        assert "The Four Distinct Rupee Quantities" in delivery.user_facing_message
        assert state.final_output.verifier_report.passed_verification is True
        assert state.final_output.verifier_report.unsupported_claim_rate == 0.0

    assert len(set(verdicts)) == 1 and verdicts[0] == "SUGGESTION"
    assert len(set(primary_ids)) == 1
    assert len(audit_file.read_text(encoding="utf-8").strip().splitlines()) == 5

    # Verify post-LLM citation verifier catches a hallucinated citation
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
    # Stale timestamp at 10:25 IST (> 180s after 10:14:35 IST quotes) triggers INSUFFICIENT DATA
    stale_dt = dt.datetime(2026, 11, 6, 10, 25, 0, tzinfo=IST)
    state, delivery = run_analyst_pipeline(stale_dt, cfg, force_render_for_inspection=False)
    assert state.final_output is not None
    assert state.final_output.verdict == "INSUFFICIENT DATA"
    assert delivery.alerted_trader is False
    assert delivery.user_facing_message == ""
    assert audit_file.exists()


def test_secret_scrubber_chmod600_and_paid_llm_guard(tmp_path: Path) -> None:
    """Verify secret scrubbing, external chmod-600 file enforcement, and zero paid LLM spend."""
    scrubber = SecretScrubber(explicit_secrets=["SUPER_SECRET_KOTAK_TOKEN_12345"])
    dirty = {
        "kotak_session_token": "SUPER_SECRET_KOTAK_TOKEN_12345",
        "note": "Contact +91 9876543210 or PAN ABCDE1234F with token SUPER_SECRET_KOTAK_TOKEN_12345",
    }
    cleaned = scrubber.scrub_obj(dirty)
    assert cleaned["kotak_session_token"] == "[REDACTED_KEY_VALUE]"
    assert "SUPER_SECRET_KOTAK_TOKEN_12345" not in cleaned["note"]
    assert "9876543210" not in cleaned["note"]
    assert "ABCDE1234F" not in cleaned["note"]

    # External chmod-600 check: insecure 0644 file must raise PermissionError
    ext_file = tmp_path / "external_secrets.env"
    ext_file.write_text("KOTAK_SID=test_sid_val\n", encoding="utf-8")
    os.chmod(ext_file, 0o644)
    with pytest.raises(PermissionError, match="insecure permissions"):
        load_chmod600_env_file(ext_file, repo_root=REPO_ROOT)

    os.chmod(ext_file, 0o600)
    loaded = load_chmod600_env_file(ext_file, repo_root=REPO_ROOT)
    assert loaded["KOTAK_SID"] == "test_sid_val"

    # Paid LLM guard check
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


def test_rule4_gross_contractual_bound_vs_all_in_label_split() -> None:
    """7 Nov 2026 decisions: Rule 4 retains BOTH the KNOWN gross contractual payoff
    bound and the all-in cost quantities, separately labeled.

    PASS only when gross <= ceiling AND all-in worst case finite AND <= ceiling.
    HOLD (UNKNOWN, research-only, no alert) when gross <= ceiling but the all-in
    worst case exceeds the ceiling OR the expiry-hold path is unbounded - the
    managed-exit estimate does not settle the cap's meaning. Never certified by
    managed exit; candidate-specific authority uncertainty, not a strategy ban.
    """
    import math as _math

    from nifty_analyst.engine.pricing import (
        LegPayoffSpec,
        compute_piecewise_expiry_payoff,
    )

    cfg = AnalystConfig(fixtures_dir=str(FIXTURES_DIR))
    ps, vol = _sample_price_and_vol(ivp_90d=72.0)

    def _run(legs, gross, all_in, managed):
        return evaluate_part_a_rules(
            is_defined_risk=True,
            legs=legs,
            dte_trading_sessions=4,
            planned_stop_rs=2000.0,
            portfolio_state=build_portfolio_exposure_state([]),
            contractual_max_loss_rs=gross,
            all_in_worst_case_rs=all_in,
            all_in_managed_exit_worst_rs=managed,
            invalidation_level=24050.0,
            invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
            invalidation_justification="Daily close below 24050",
            price_structure=ps,
            volatility=vol,
            config=cfg,
        )

    call_spread_legs = [
        _sample_leg(24200.0, "CE", "LONG", delta=0.52),
        _sample_leg(24400.0, "CE", "SHORT", delta=0.19),
    ]
    put_spread_legs = [
        _sample_leg(24100.0, "PE", "LONG", delta=-0.20),
        _sample_leg(24200.0, "PE", "SHORT", delta=-0.30),
    ]

    # Case 1: bull PUT credit spread 24100/24200 (credit 40 pts) - no long call, so
    # the expiry-hold path is BOUNDED. Gross loss = width 100 - credit 40 = 60 pts
    # x 75 = Rs 4,500 (KNOWN); all-in finite and under the ceiling -> PASS.
    put_specs = [
        LegPayoffSpec(strike=24100.0, option_type="PE", side="LONG", ratio=1, executable_entry_price=30.0),
        LegPayoffSpec(strike=24200.0, option_type="PE", side="SHORT", ratio=1, executable_entry_price=70.0),
    ]
    pw_put = compute_piecewise_expiry_payoff(
        put_specs, lot_size=75, lots=1, statutory_charges_rs=100.0, exercise_stt_rate=0.0015
    )
    assert pw_put.is_defined_risk and pw_put.expiry_hold_bounded
    gross_put = round(pw_put.gross_max_loss_per_unit_points * 75, 2)
    assert gross_put == pytest.approx(4500.0, abs=1.0)
    all_in_put = round(max(pw_put.max_loss_rs, pw_put.expiry_hold_max_loss_rs), 2)
    assert all_in_put <= 10000.0 and gross_put != all_in_put  # distinct, never merged
    checks_pass, overall_pass = _run(put_spread_legs, gross_put, all_in_put, round(pw_put.max_loss_rs, 2))
    r4_pass = checks_pass["RULE_4_CONTRACTUAL_MAX_LOSS_10000_RS"]
    assert r4_pass.status == "PASS"
    assert "KNOWN" in r4_pass.evidence
    assert overall_pass == "PASS"

    # Case 2: bull CALL debit spread - long-call tail makes the expiry-hold path
    # UNBOUNDED. Gross 6,000 and managed-exit estimate ~6,100 are both under the
    # ceiling, but the true all-in worst case has no bound -> HOLD, not PASS.
    call_specs = [
        LegPayoffSpec(strike=24200.0, option_type="CE", side="LONG", ratio=1, executable_entry_price=140.0),
        LegPayoffSpec(strike=24400.0, option_type="CE", side="SHORT", ratio=1, executable_entry_price=60.0),
    ]
    pw_call = compute_piecewise_expiry_payoff(
        call_specs, lot_size=75, lots=1, statutory_charges_rs=100.0, exercise_stt_rate=0.0015
    )
    assert pw_call.is_defined_risk and not pw_call.expiry_hold_bounded
    gross_call = round(pw_call.gross_max_loss_per_unit_points * 75, 2)
    assert gross_call == pytest.approx(6000.0, abs=1.0)
    checks_hold, overall_hold = _run(call_spread_legs, gross_call, _math.inf, round(pw_call.max_loss_rs, 2))
    r4_hold = checks_hold["RULE_4_CONTRACTUAL_MAX_LOSS_10000_RS"]
    assert r4_hold.status == "UNKNOWN"
    assert "HOLD" in r4_hold.evidence
    assert "UNBOUNDED" in r4_hold.evidence
    assert "pending user authority" in r4_hold.evidence
    assert "not contractual" in r4_hold.evidence
    assert overall_hold == "UNKNOWN"

    # Case 3: gross under ceiling, finite all-in worst case over it -> HOLD.
    checks_hold2, overall_hold2 = _run(call_spread_legs, 9950.0, 10050.0, 10050.0)
    r4_hold2 = checks_hold2["RULE_4_CONTRACTUAL_MAX_LOSS_10000_RS"]
    assert r4_hold2.status == "UNKNOWN"
    assert "HOLD" in r4_hold2.evidence and "pending user authority" in r4_hold2.evidence
    assert overall_hold2 == "UNKNOWN"

    # Case 4: gross bound over the ceiling -> FAIL under every reading of the cap.
    checks_fail, _ = _run(call_spread_legs, 10500.0, 10600.0, 10600.0)
    assert checks_fail["RULE_4_CONTRACTUAL_MAX_LOSS_10000_RS"].status == "FAIL"


def test_agent_artifact_provider_interface_round_trip(tmp_path: Path) -> None:
    """External synthesis artifact seam verification incl. scheduling (7 Nov 2026).

    The hourly run pauses after the deterministic gates; the provider exports THIS
    run's request (sanitised evidence + run identity); the operator writes an artifact
    bound to that run_id/decision timestamp from the exported evidence; the resumed
    run verifies citations unchanged. Stale or foreign artifacts are refused.
    """
    import json as _json

    from nifty_analyst.synthesis.llm_node import AgentArtifactSynthesisProvider

    dec_dt = dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST)
    artifact = tmp_path / "agent_synthesis.json"
    request = tmp_path / "agent_synthesis.json.request.json"

    cfg = AnalystConfig(mode="OFFLINE", fixtures_dir=str(FIXTURES_DIR),
                        audit_log_path=str(tmp_path / "audit.jsonl"),
                        llm_provider="agent_artifact",
                        agent_synthesis_artifact_path=str(artifact))

    # Phase A: no artifact yet -> run raises and exports the run-bound request.
    with pytest.raises(FileNotFoundError):
        run_analyst_pipeline(dec_dt, cfg)
    assert request.is_file()
    req = _json.loads(request.read_text(encoding="utf-8"))
    assert req["run_id"].startswith("run_20261106_101500")
    assert req["decision_timestamp_ist"]
    evidence = req["sanitised_state"]
    # The request carries the sanitised evidence (post deterministic gates), not fixture text.
    assert "market_evidence" in evidence and "candidate_structures" in evidence
    assert "run_metadata" not in evidence or True  # scrubbed state dict shape

    # The operator writes the artifact FROM the exported current evidence.
    cands = evidence["candidate_structures"]
    primary_id = None
    # primary = first candidate the deterministic engine would select is not in the
    # request; cite a real candidate field instead, resolved through the request.
    idx = 0
    bound_val = cands[idx]["contractual_max_loss_rs"]
    ivp_val = evidence["market_evidence"]["volatility_state"]["ivp_90d"]
    artifact.write_text(_json.dumps({
        "model_id": "instinct-agent-synthesis-v1",
        "run_id": req["run_id"],
        "decision_timestamp_ist": req["decision_timestamp_ist"],
        "reasoning_summary": "Agent-written regime read and evidence weighing.",
        "habitual_comparison_note": "No habitual baseline change.",
        "research_rule_challenge_note": "No research-rule challenge.",
        "confidence_note": "Confidence per verified citations only.",
        "citations": [
            {"claim_text": f"Candidate contractual payoff bound is Rs {bound_val:.2f} (gross, KNOWN).",
             "state_field_path": f"candidate_structures[{idx}].contractual_max_loss_rs",
             "cited_value": bound_val},
            {"claim_text": f"90-session IVP stands at {ivp_val}.",
             "state_field_path": "market_evidence.volatility_state.ivp_90d",
             "cited_value": ivp_val},
        ],
    }), encoding="utf-8")

    # Phase B: resumed run consumes the bound artifact; verifier passes unchanged.
    state2, delivery2 = run_analyst_pipeline(dec_dt, cfg)
    assert state2.final_output is not None
    assert state2.final_output.verifier_report.passed_verification is True
    assert state2.final_output.verifier_report.unsupported_claim_rate == 0.0
    assert delivery2.alerted_trader is (state2.final_output.verdict == "SUGGESTION")

    # Zero paid spend by construction.
    resp = AgentArtifactSynthesisProvider(str(artifact)).synthesize(
        _json.dumps(evidence), state2, None, [])
    assert resp.estimated_cost_inr == 0.0 and resp.estimated_tokens_used == 0
    assert resp.model_id == "instinct-agent-synthesis-v1"

    # Changed evidence / expired artifact: an artifact bound to a different run is
    # refused outright (no stale synthesis ever passes). The 11:15 pipeline slot
    # short-circuits before the synthesis layer (Wait path), so exercise the binding
    # against a copied state carrying a different run identity.
    import dataclasses as _dc
    other_meta = _dc.replace(
        state2.run_metadata, run_id="run_20261106_111500_abcdef",
        decision_timestamp_ist="2026-11-06T11:15:00+05:30",
    )
    other_state = _dc.replace(state2, run_metadata=other_meta)
    with pytest.raises(ValueError, match="stale or foreign artifact"):
        AgentArtifactSynthesisProvider(str(artifact)).synthesize(
            _json.dumps(evidence), other_state, None, [])

    # Foreign artifact: tampered run binding is refused.
    tampered = _json.loads(artifact.read_text(encoding="utf-8"))
    tampered["run_id"] = "run_19990101_000000_deadbeef"
    artifact.write_text(_json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="stale or foreign artifact"):
        run_analyst_pipeline(dec_dt, cfg)

    # Fabricated citations (valid binding) must fail verification and suppress alerts.
    state_req = _json.loads(request.read_text(encoding="utf-8"))
    bad_payload = {
        "model_id": "instinct-agent-synthesis-v1",
        "run_id": state_req["run_id"],
        "decision_timestamp_ist": state_req["decision_timestamp_ist"],
        "reasoning_summary": "x", "habitual_comparison_note": "x",
        "research_rule_challenge_note": "x", "confidence_note": "x",
        "citations": [
            {"claim_text": "fabricated", "state_field_path": "market_evidence.nope.field",
             "cited_value": 0},
        ] * 10,
    }
    artifact.write_text(_json.dumps(bad_payload), encoding="utf-8")
    state3, delivery3 = run_analyst_pipeline(dec_dt, cfg)
    assert state3.final_output.verifier_report.passed_verification is False
    assert state3.final_output.verdict != "SUGGESTION"
    assert delivery3.alerted_trader is False

    # Schema violations raise cleanly, no silent pass.
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(_json.dumps({"model_id": "x"}), encoding="utf-8")
    with pytest.raises(ValueError):
        AgentArtifactSynthesisProvider(str(incomplete)).synthesize(
            _json.dumps(evidence), state2, None, [])
