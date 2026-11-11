"""Adversarial & negative regression suite for post-ce3e9a5 hardening (Items 1-5).

Maps directly to every acceptance requirement in `NIFTY OPTIONS ANALYST V2: NEXT FIXES`:
- Item 1: Bind every delivered message, prose field, chart, report, and audit record to the
  final verified selection (reject/replace contradictory digests, cross-candidate borrowed
  numbers, metric-meaning swaps, and prose tampering across all 12 delivered fields).
- Item 2: Freeze canonical evidence snapshot with write-once immutability, SHA256 + target/ID
  seals, zero broker reads on resume, and strict 180-second limit from actual capture-start
  using a deterministic fake clock (no real sleeps).
- Item 3: Preserve monotonic source-order depth guard (`DEPTH_LEVEL_ORDER_INCONSISTENT`) and
  qualify `DepthBookInterpretation.is_structurally_valid`.
- Item 4: Retain small-touch observations as explicitly `INDICATIVE_TOUCH_ONLY` without blanket
  halts while walking full book depth (`walk_depth_levels`) before certifying Rule 4 (Rs 10,000).
- Item 5: Keep live readiness OFF (`is_live_ready=False`, `suppress_trader_alerts=True` default)
  and truthfully report `SINGLE_EXPIRY_DEGRADED`, `uncovered_held_neo_symbols`,
  `FETCH_TIME_NOT_EXCHANGE`, and `PacedReadTransport.HTTP_BOUNDARY_PACING_UNVERIFIED`.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import inspect
import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any

import pytest

from nifty_analyst.adapters.kotak_neo import (
    FixtureKotakNeoAdapter,
    KotakNeoReadOnlyPort,
)
from nifty_analyst.adapters.request_safety import (
    PacedReadTransport,
    interpret_depth_book,
    walk_depth_levels,
)
from nifty_analyst.config import AnalystConfig
from nifty_analyst.engine.pricing import RawStrikeQuote
from nifty_analyst.engine.structures import (
    build_candidate_structure,
    build_option_leg,
)
from nifty_analyst.operations.decision_driver import run_scheduled_slot
from nifty_analyst.output.renderer import render_analyst_report
from nifty_analyst.pipeline import (
    build_portfolio_exposure_state,
    run_analyst_pipeline,
)
from nifty_analyst.scrubber import IST
from nifty_analyst.state import (
    ConsolidatorState,
    ExpiryMetadata,
    NiftyOptionsState,
    OpenPosition,
    PriceStructureEvidence,
    TimestampedDatum,
    VolatilityEvidence,
)
from nifty_analyst.synthesis.evidence_snapshot import (
    EvidenceSnapshotClockError,
    SnapshotOverwriteBlockedError,
    SnapshotTamperOrMismatchError,
    StaleEvidenceSnapshotError,
    persist_immutable_snapshot_bundle,
    sha256_hex_of_obj,
    snapshot_paths_for_artifact,
    write_once_json,
)
from nifty_analyst.synthesis.llm_node import (
    AgentArtifactSynthesisProvider,
    LLMProvider,
    LLMSynthesisResponse,
    OfflineDeterministicSynthesisProvider,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "fixtures"


class _StaticOverrideProvider(LLMProvider):
    """Test helper wrapping `OfflineDeterministicSynthesisProvider` with targeted mutations."""

    def __init__(self, mutator) -> None:
        self._base = OfflineDeterministicSynthesisProvider()
        self._mutator = mutator

    def synthesize(
        self,
        sanitised_state_json: str,
        state: NiftyOptionsState,
        primary: Any,
        alternatives: list[Any],
    ) -> LLMSynthesisResponse:
        base_resp = self._base.synthesize(sanitised_state_json, state, primary, alternatives)
        return self._mutator(base_resp, state, primary, alternatives)


class _ForbiddenBrokerAdapter(KotakNeoReadOnlyPort):
    """Spy adapter that fails the test immediately if ANY broker read is attempted."""

    def fetch_contract_master_raw(self, decision_dt_ist: dt.datetime) -> dict[str, Any]:
        raise AssertionError("BROKER_READ_FORBIDDEN_ON_RESUME: fetch_contract_master_raw")

    def fetch_option_chain_raw(
        self, expiry_date: str, decision_dt_ist: dt.datetime
    ) -> dict[str, Any]:
        raise AssertionError(f"BROKER_READ_FORBIDDEN_ON_RESUME: fetch_option_chain_raw({expiry_date})")

    def fetch_index_candles_raw(self, decision_dt_ist: dt.datetime) -> dict[str, Any]:
        raise AssertionError("BROKER_READ_FORBIDDEN_ON_RESUME: fetch_index_candles_raw")

    def fetch_india_vix_raw(
        self, decision_dt_ist: dt.datetime
    ) -> TimestampedDatum[float | None]:
        raise AssertionError("BROKER_READ_FORBIDDEN_ON_RESUME: fetch_india_vix_raw")


def _write_valid_agent_artifact(
    artifact_path: Path,
    req_payload: dict[str, Any],
    chosen_card: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pass_cards = req_payload["compact_desk_brief"]["certified_pass_candidates"]
    card = chosen_card or pass_cards[0]
    idx = card["state_index"]
    bound_val = card["contractual_max_loss_rs"]
    ivp_val = req_payload["sanitised_state"]["market_evidence"]["volatility_state"]["ivp_90d"]
    doc = {
        "model_id": "whatsapp-persona-agent-v1",
        "run_id": req_payload["run_id"],
        "snapshot_id": req_payload["snapshot_id"],
        "evidence_sha256": req_payload["evidence_sha256"],
        "decision_timestamp_ist": req_payload["decision_timestamp_ist"],
        "reasoning_summary": f"Consolidator selects `{card['structure_id']}` on verified rules.",
        "habitual_comparison_note": "Evaluated verticals, condors, and butterflies without STT.",
        "research_rule_challenge_note": "No research-rule challenge.",
        "confidence_note": "Confidence bounded by verified citations.",
        "debate_state": {
            "bull_thesis": "Support holds above Put OI wall.",
            "bear_thesis": "Overhead Call OI wall caps upside.",
            "vol_regime_thesis": "Surface regime aligns with defined-risk spread.",
            "rebuttal_round": "Completed-candle invalidation governs thesis.",
            "dominant_regime": "BULLISH",
        },
        "risk_triad": {
            "convexity_advocate_note": f"Champions `{card['structure_id']}` risk-reward.",
            "tail_guardian_note": "Defined wings cap overnight gap loss.",
            "neutral_arbiter_note": f"Endorses `{card['structure_id']}`.",
        },
        "consolidator": {
            "recommended_structure_id": card["structure_id"],
            "conviction": "HIGH",
            "final_recommendation_rationale": f"Consolidator selects `{card['structure_id']}`.",
            "whatsapp_digest": f"*WhatsApp Pick*: `{card['structure_id']}`",
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
    artifact_path.write_text(json.dumps(doc), encoding="utf-8")
    return doc


# ============================================================================
# ITEM 1: BIND EVERY DELIVERED MESSAGE TO THE FINAL VERIFIED SELECTION
# ============================================================================


def test_item1_reproduced_wrong_contract_digest_rejected_and_scrubbed(tmp_path: Path) -> None:
    """Exact reproduced failure: `Buy WRONG_CONTRACT now. Maximum loss Rs 999999.00.` is rejected."""
    audit_path = tmp_path / "audit_repro.jsonl"
    cfg = AnalystConfig(
        mode="OFFLINE",
        fixtures_dir=str(FIXTURES_DIR),
        audit_log_path=str(audit_path),
    )
    dec_dt = dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST)

    poison_digest = "Buy WRONG_CONTRACT now. Maximum loss Rs 999999.00."

    def _mutate(resp: LLMSynthesisResponse, _s: NiftyOptionsState, _p: Any, _a: list[Any]):
        assert resp.consolidator_state is not None
        return dataclasses.replace(
            resp,
            consolidator_state=dataclasses.replace(
                resp.consolidator_state,
                whatsapp_digest=poison_digest,
            ),
        )

    state, delivery = run_analyst_pipeline(
        dec_dt,
        cfg,
        llm_provider=_StaticOverrideProvider(_mutate),
        force_render_for_inspection=False,
    )
    assert state.final_output is not None
    assert state.final_output.verifier_report.passed_verification is False
    assert state.final_output.verdict == "WAIT"
    assert state.final_output.primary_proposal is None
    assert delivery.alerted_trader is False
    assert delivery.user_facing_message == ""
    assert delivery.whatsapp_digest == ""

    # Contradictory digest never survives in final_output, consolidator_state, report, or audit log
    for forbidden in ("WRONG_CONTRACT", "999999"):
        assert forbidden not in state.final_output.whatsapp_digest
        assert state.consolidator_state is not None
        assert forbidden not in state.consolidator_state.whatsapp_digest
        assert forbidden not in delivery.full_report_markdown

    audit_row = json.loads(audit_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert audit_row["final_output"]["verdict"] == "WAIT"
    assert audit_row["final_output"]["primary_proposal"] is None
    assert "WRONG_CONTRACT" not in audit_row["final_output"]["whatsapp_digest"]
    assert "WRONG_CONTRACT" not in audit_row["consolidator_state"]["whatsapp_digest"]


def test_item1_digest_field_defects_contract_expiry_side_risk_quantity_rejected(
    tmp_path: Path,
) -> None:
    """Valid selection + wrong contract, expiry, leg side, risk amount, or quantity in digest -> rejected."""
    dec_dt = dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST)
    base_cfg = AnalystConfig(
        mode="OFFLINE",
        fixtures_dir=str(FIXTURES_DIR),
        audit_log_path=str(tmp_path / "base.jsonl"),
    )
    base_state, _ = run_analyst_pipeline(dec_dt, base_cfg)
    assert base_state.final_output is not None and base_state.final_output.primary_proposal is not None
    cand_a = base_state.final_output.primary_proposal
    other_cands = [
        c
        for c in base_state.candidate_structures
        if c.structure_id != cand_a.structure_id and c.part_a_overall_status == "PASS"
    ]
    assert other_cands
    cand_b = other_cands[0]

    leg0 = cand_a.legs[0]
    wrong_side_code = "B" if leg0.side == "SHORT" else "S"
    wrong_verb = "Buy" if leg0.side == "SHORT" else "Sell"
    wrong_qty = cand_a.lot_size * (cand_a.lots + 1)

    bad_digests = {
        "wrong_structure_id": f"*Pick*: `{cand_b.structure_id}`",
        "wrong_expiry": f"*Pick*: `{cand_a.structure_id}` (`2026-12-31`, {cand_a.dte_trading_sessions} DTE)",
        "wrong_compact_leg_side": (
            f"*Pick*: `{cand_a.structure_id}` | Legs: {wrong_side_code} {int(leg0.strike)}{leg0.option_type}"
        ),
        "wrong_verb_leg_side": (
            f"*Pick*: `{cand_a.structure_id}` | {wrong_verb} {int(leg0.strike)} {leg0.option_type}"
        ),
        "wrong_risk_amount": (
            f"*Pick*: `{cand_a.structure_id}` | Maximum loss Rs {cand_a.margin_estimate_rs:.2f}"
        ),
        "wrong_lots": f"*Pick*: `{cand_a.structure_id}` | Size: 3 lots",
        "wrong_unit_quantity": f"*Pick*: `{cand_a.structure_id}` | Size: {wrong_qty} qty",
    }

    for label, bad_digest in bad_digests.items():
        cfg_i = AnalystConfig(
            mode="OFFLINE",
            fixtures_dir=str(FIXTURES_DIR),
            audit_log_path=str(tmp_path / f"audit_{label}.jsonl"),
        )

        def _mut(resp: LLMSynthesisResponse, _s, _p, _a, d=bad_digest):
            assert resp.consolidator_state is not None
            return dataclasses.replace(
                resp,
                consolidator_state=dataclasses.replace(resp.consolidator_state, whatsapp_digest=d),
            )

        st_i, deliv_i = run_analyst_pipeline(dec_dt, cfg_i, llm_provider=_StaticOverrideProvider(_mut))
        assert st_i.final_output is not None, label
        assert st_i.final_output.verifier_report.passed_verification is False, label
        assert st_i.final_output.verdict == "WAIT", label
        assert st_i.final_output.primary_proposal is None, label
        assert deliv_i.alerted_trader is False, label
        assert bad_digest not in st_i.final_output.whatsapp_digest, label


def test_item1_cross_candidate_valid_number_and_metric_swap_fail_semantic_binding(
    tmp_path: Path,
) -> None:
    """A valid number borrowed from another candidate or swapped across metrics fails semantic binding."""
    dec_dt = dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST)
    base_cfg = AnalystConfig(
        mode="OFFLINE",
        fixtures_dir=str(FIXTURES_DIR),
        audit_log_path=str(tmp_path / "base_sem.jsonl"),
    )
    base_state, _ = run_analyst_pipeline(dec_dt, base_cfg)
    assert base_state.final_output is not None and base_state.final_output.primary_proposal is not None
    cand_a = base_state.final_output.primary_proposal
    cand_b = next(
        c
        for c in base_state.candidate_structures
        if c.structure_id != cand_a.structure_id
        and abs(c.contractual_max_loss_rs - cand_a.contractual_max_loss_rs) > 25.0
    )

    # 1. Borrow `cand_b.contractual_max_loss_rs` (valid in state!) inside rationale for `cand_a`
    def _borrow_in_rationale(resp: LLMSynthesisResponse, _s, _p, _a):
        assert resp.consolidator_state is not None
        return dataclasses.replace(
            resp,
            consolidator_state=dataclasses.replace(
                resp.consolidator_state,
                final_recommendation_rationale=(
                    f"Consolidator selects `{cand_a.structure_id}` with contractual max loss "
                    f"Rs {cand_b.contractual_max_loss_rs:.2f}."
                ),
            ),
        )

    # 2. Borrow `cand_b.contractual_max_loss_rs` in an unscoped sentence inside `tail_guardian_note`
    def _borrow_unscoped_in_triad(resp: LLMSynthesisResponse, _s, _p, _a):
        assert resp.risk_triad_state is not None
        return dataclasses.replace(
            resp,
            risk_triad_state=dataclasses.replace(
                resp.risk_triad_state,
                tail_guardian_note=f"Contractual max loss is Rs {cand_b.contractual_max_loss_rs:.2f}.",
            ),
        )

    # 3. Swap `cand_a.margin_estimate_rs` into `contractual max loss` on `cand_a` itself!
    def _swap_margin_for_max_loss(resp: LLMSynthesisResponse, _s, _p, _a):
        return dataclasses.replace(
            resp,
            reasoning_summary=(
                f"Consolidator selects `{cand_a.structure_id}` with contractual max loss "
                f"Rs {cand_a.margin_estimate_rs:.2f}."
            ),
        )

    # 4. Swap a valid strike (`cand_a.legs[0].strike`) into `Spot is at ...`!
    wrong_spot = cand_a.legs[0].strike
    if abs(wrong_spot - base_state.contract_context.spot.value) <= 2.0:
        wrong_spot = cand_a.legs[-1].strike

    def _swap_strike_for_spot(resp: LLMSynthesisResponse, _s, _p, _a):
        assert resp.debate_state is not None
        return dataclasses.replace(
            resp,
            debate_state=dataclasses.replace(
                resp.debate_state,
                bull_thesis=f"Spot is at {wrong_spot:.2f} with bullish structure.",
            ),
        )

    for idx, mut in enumerate(
        (
            _borrow_in_rationale,
            _borrow_unscoped_in_triad,
            _swap_margin_for_max_loss,
            _swap_strike_for_spot,
        )
    ):
        cfg_i = AnalystConfig(
            mode="OFFLINE",
            fixtures_dir=str(FIXTURES_DIR),
            audit_log_path=str(tmp_path / f"sem_{idx}.jsonl"),
        )
        st_i, deliv_i = run_analyst_pipeline(dec_dt, cfg_i, llm_provider=_StaticOverrideProvider(mut))
        assert st_i.final_output is not None
        assert st_i.final_output.verifier_report.passed_verification is False, f"case {idx}"
        assert st_i.final_output.verdict == "WAIT", f"case {idx}"
        assert deliv_i.alerted_trader is False, f"case {idx}"


def test_item1_tampering_across_all_twelve_delivered_prose_fields_caught(tmp_path: Path) -> None:
    """Tampering in EACH of the 12 delivered prose fields is caught and excluded from delivery."""
    dec_dt = dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST)
    marker = "TAMPER_MARKER_RS_88888"
    bad_sentence = f" [{marker}] Contractual max loss is Rs 88888.00."

    top_level_fields = (
        "reasoning_summary",
        "habitual_comparison_note",
        "research_rule_challenge_note",
        "confidence_note",
    )
    debate_fields = (
        "bull_thesis",
        "bear_thesis",
        "vol_regime_thesis",
        "rebuttal_round",
    )
    triad_fields = (
        "convexity_advocate_note",
        "tail_guardian_note",
        "neutral_arbiter_note",
    )

    all_mutations: list[tuple[str, Any]] = []
    for fname in top_level_fields:
        all_mutations.append(
            (
                fname,
                lambda resp, _s, _p, _a, f=fname: dataclasses.replace(
                    resp, **{f: getattr(resp, f) + bad_sentence}
                ),
            )
        )
    for fname in debate_fields:
        all_mutations.append(
            (
                fname,
                lambda resp, _s, _p, _a, f=fname: dataclasses.replace(
                    resp,
                    debate_state=dataclasses.replace(
                        resp.debate_state,
                        **{f: getattr(resp.debate_state, f) + bad_sentence},
                    ),
                ),
            )
        )
    for fname in triad_fields:
        all_mutations.append(
            (
                fname,
                lambda resp, _s, _p, _a, f=fname: dataclasses.replace(
                    resp,
                    risk_triad_state=dataclasses.replace(
                        resp.risk_triad_state,
                        **{f: getattr(resp.risk_triad_state, f) + bad_sentence},
                    ),
                ),
            )
        )
    all_mutations.append(
        (
            "final_recommendation_rationale",
            lambda resp, _s, _p, _a: dataclasses.replace(
                resp,
                consolidator_state=dataclasses.replace(
                    resp.consolidator_state,
                    final_recommendation_rationale=(
                        resp.consolidator_state.final_recommendation_rationale + bad_sentence
                    ),
                ),
            ),
        )
    )
    assert len(all_mutations) == 12

    for fname, mut in all_mutations:
        cfg_f = AnalystConfig(
            mode="OFFLINE",
            fixtures_dir=str(FIXTURES_DIR),
            audit_log_path=str(tmp_path / f"prose_{fname}.jsonl"),
        )
        st_f, deliv_f = run_analyst_pipeline(
            dec_dt,
            cfg_f,
            llm_provider=_StaticOverrideProvider(mut),
            force_render_for_inspection=True,
        )
        assert st_f.final_output is not None, fname
        assert st_f.final_output.verifier_report.passed_verification is False, fname
        assert st_f.final_output.verdict == "WAIT", fname
        assert st_f.final_output.primary_proposal is None, fname
        assert deliv_f.alerted_trader is False, fname
        assert marker not in deliv_f.full_report_markdown, fname


def test_item1_non_first_candidate_selection_end_to_end_agreement(tmp_path: Path) -> None:
    """Consolidator selects a non-first PASS candidate: proposal, digest, report, chart, and audit agree."""
    dec_dt = dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST)
    charts_dir = tmp_path / "charts"
    audit_path = tmp_path / "audit_nonfirst.jsonl"
    artifact_path = tmp_path / "nonfirst_synth.json"
    req_path = tmp_path / "nonfirst_synth.json.request.json"

    cfg = AnalystConfig(
        mode="OFFLINE",
        fixtures_dir=str(FIXTURES_DIR),
        audit_log_path=str(audit_path),
        payoff_chart_dir=str(charts_dir),
        llm_provider="agent_artifact",
        agent_synthesis_artifact_path=str(artifact_path),
    )

    with pytest.raises(FileNotFoundError):
        run_analyst_pipeline(dec_dt, cfg)

    req = json.loads(req_path.read_text(encoding="utf-8"))
    pass_cards = req["compact_desk_brief"]["certified_pass_candidates"]
    assert len(pass_cards) >= 3
    first_id = pass_cards[0]["structure_id"]
    chosen_nonfirst = pass_cards[2]
    target_id = chosen_nonfirst["structure_id"]
    assert target_id != first_id

    _write_valid_agent_artifact(artifact_path, req, chosen_card=chosen_nonfirst)
    state, delivery = run_analyst_pipeline(dec_dt, cfg, force_render_for_inspection=True)

    assert state.final_output is not None
    assert state.final_output.verdict == "SUGGESTION"
    assert state.final_output.verifier_report.passed_verification is True
    assert state.final_output.primary_proposal is not None
    assert state.final_output.primary_proposal.structure_id == target_id
    assert target_id in state.final_output.whatsapp_digest
    assert first_id not in state.final_output.whatsapp_digest
    assert f"**Structure ID**: `{target_id}`" in delivery.full_report_markdown

    if state.final_output.payoff_diagram_path:
        assert target_id in state.final_output.payoff_diagram_path
        assert first_id not in state.final_output.payoff_diagram_path

    audit_row = json.loads(audit_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert audit_row["consolidator_state"]["recommended_structure_id"] == target_id
    assert audit_row["final_output"]["primary_proposal"]["structure_id"] == target_id
    assert target_id in audit_row["final_output"]["whatsapp_digest"]


def test_item1_unknown_id_failed_part_a_wait_and_verifier_failure_never_alert(
    tmp_path: Path,
) -> None:
    """Unknown ID, failed Part A candidate, WAIT, or verifier failure never produce a trade alert."""
    dec_dt = dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST)
    probe_cfg = AnalystConfig(
        mode="OFFLINE",
        fixtures_dir=str(FIXTURES_DIR),
        audit_log_path=str(tmp_path / "probe.jsonl"),
    )
    probe_state, _ = run_analyst_pipeline(dec_dt, probe_cfg)
    failed_cand = next(
        c for c in probe_state.candidate_structures if c.part_a_overall_status == "FAIL"
    )

    cases = {
        "unknown_id": lambda resp, _s, _p, _a: dataclasses.replace(
            resp,
            consolidator_state=ConsolidatorState(
                recommended_structure_id="BCS_2026-11-12_99900_99950",
                conviction="HIGH",
                final_recommendation_rationale="Unknown contract.",
                whatsapp_digest="*Pick*: `BCS_2026-11-12_99900_99950`",
            ),
        ),
        "failed_part_a_id": lambda resp, _s, _p, _a: dataclasses.replace(
            resp,
            consolidator_state=ConsolidatorState(
                recommended_structure_id=failed_cand.structure_id,
                conviction="HIGH",
                final_recommendation_rationale=f"Selects `{failed_cand.structure_id}`.",
                whatsapp_digest=f"*Pick*: `{failed_cand.structure_id}`",
            ),
        ),
        "explicit_wait": lambda resp, _s, _p, _a: dataclasses.replace(
            resp,
            consolidator_state=ConsolidatorState(
                recommended_structure_id="WAIT",
                conviction="SIT_OUT",
                final_recommendation_rationale="Standing aside (`WAIT`).",
                whatsapp_digest="*NIFTY Options Desk — WAIT*",
            ),
        ),
        "citation_failure": lambda resp, _s, _p, _a: dataclasses.replace(
            resp,
            citations=[
                {
                    "claim_text": "Fabricated spot",
                    "state_field_path": "contract_context.spot.value",
                    "cited_value": 99999.0,
                }
            ],
        ),
    }

    for name, mut in cases.items():
        cfg_c = AnalystConfig(
            mode="OFFLINE",
            fixtures_dir=str(FIXTURES_DIR),
            audit_log_path=str(tmp_path / f"gate_{name}.jsonl"),
            payoff_chart_dir=str(tmp_path / f"charts_{name}"),
        )
        st_c, deliv_c = run_analyst_pipeline(dec_dt, cfg_c, llm_provider=_StaticOverrideProvider(mut))
        assert st_c.final_output is not None, name
        assert st_c.final_output.verdict == "WAIT", name
        assert st_c.final_output.primary_proposal is None, name
        assert st_c.final_output.payoff_diagram_path == "", name
        assert deliv_c.alerted_trader is False, name
        assert deliv_c.user_facing_message == "", name
        assert deliv_c.whatsapp_digest == "", name


# ============================================================================
# ITEM 2: FREEZE EVIDENCE AND ENFORCE THE 180-SECOND CAPTURE/RESUME LIMIT
# ============================================================================


def test_item2_changed_market_data_same_timestamp_and_config_cannot_reuse_snapshot(
    tmp_path: Path,
) -> None:
    """Same decision_timestamp_ist & config with changed market data produces new identity and rejects old artifact."""
    fx_dir = tmp_path / "fx"
    shutil.copytree(FIXTURES_DIR, fx_dir)
    dec_dt = dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST)
    t0 = dt.datetime(2026, 11, 6, 10, 15, 5, tzinfo=IST)

    art1 = tmp_path / "slot_run1.json"
    cfg1 = AnalystConfig(
        mode="OFFLINE",
        fixtures_dir=str(fx_dir),
        audit_log_path=str(tmp_path / "a1.jsonl"),
        llm_provider="agent_artifact",
        agent_synthesis_artifact_path=str(art1),
    )
    with pytest.raises(FileNotFoundError):
        run_analyst_pipeline(dec_dt, cfg1, wall_clock_fn=lambda: t0)

    req1 = json.loads((tmp_path / "slot_run1.json.request.json").read_text(encoding="utf-8"))
    art1_doc = _write_valid_agent_artifact(art1, req1)

    # Mutate underlying market data in-place inside `fx_dir` while keeping `dec_dt` intact!
    master_file = fx_dir / "contract_master.json"
    master_obj = json.loads(master_file.read_text(encoding="utf-8"))
    master_obj["spot"]["value"] = float(master_obj["spot"]["value"]) + 55.0
    master_file.write_text(json.dumps(master_obj), encoding="utf-8")

    # Capture 2 at the EXACT same `dec_dt` and `t0`
    art2 = tmp_path / "slot_run2.json"
    cfg2 = AnalystConfig(
        mode="OFFLINE",
        fixtures_dir=str(fx_dir),
        audit_log_path=str(tmp_path / "a1.jsonl"),
        llm_provider="agent_artifact",
        agent_synthesis_artifact_path=str(art2),
    )
    with pytest.raises(FileNotFoundError):
        run_analyst_pipeline(dec_dt, cfg2, wall_clock_fn=lambda: t0)

    req2 = json.loads((tmp_path / "slot_run2.json.request.json").read_text(encoding="utf-8"))
    assert req2["evidence_sha256"] != req1["evidence_sha256"]
    assert req2["snapshot_id"] != req1["snapshot_id"]
    assert req2["run_id"] != req1["run_id"]

    # Attempting to feed Run 1's artifact into Run 2 fails closed!
    art2.write_text(json.dumps(art1_doc), encoding="utf-8")
    with pytest.raises(SnapshotTamperOrMismatchError):
        run_analyst_pipeline(dec_dt, cfg2, wall_clock_fn=lambda: t0 + dt.timedelta(seconds=10))


def test_item2_tamper_matrix_and_dual_snapshot_hash_replacement_rejected(tmp_path: Path) -> None:
    """1-byte edit, mutated candidate, wrong hash/ID, AND simultaneous snapshot+hash replacement fail closed."""
    dec_dt = dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST)
    t0 = dt.datetime(2026, 11, 6, 10, 15, 1, tzinfo=IST)

    def _fresh_bundle(tag: str) -> tuple[AnalystConfig, Path, Path, Path]:
        sub = tmp_path / tag
        sub.mkdir(parents=True, exist_ok=True)
        art = sub / "synth.json"
        cfg = AnalystConfig(
            mode="OFFLINE",
            fixtures_dir=str(FIXTURES_DIR),
            audit_log_path=str(sub / "audit.jsonl"),
            llm_provider="agent_artifact",
            agent_synthesis_artifact_path=str(art),
        )
        with pytest.raises(FileNotFoundError):
            run_analyst_pipeline(dec_dt, cfg, wall_clock_fn=lambda: t0)
        snap_p, req_p, _, _ = snapshot_paths_for_artifact(art)
        req_doc = json.loads(req_p.read_text(encoding="utf-8"))
        _write_valid_agent_artifact(art, req_doc)
        return cfg, art, snap_p, req_p

    # Case A: 1-byte tamper in `.snapshot.json`
    cfg_a, _, snap_a, _ = _fresh_bundle("one_byte")
    os.chmod(snap_a, 0o644)
    raw_a = snap_a.read_text(encoding="utf-8")
    snap_a.write_text(raw_a.replace("UPTREND", "RANGE", 1), encoding="utf-8")
    with pytest.raises(SnapshotTamperOrMismatchError, match="EVIDENCE_SHA256_MISMATCH"):
        run_analyst_pipeline(dec_dt, cfg_a, wall_clock_fn=lambda: t0 + dt.timedelta(seconds=5))

    # Case B: Mutated candidate in `.snapshot.json`
    cfg_b, _, snap_b, _ = _fresh_bundle("cand_mut")
    os.chmod(snap_b, 0o644)
    doc_b = json.loads(snap_b.read_text(encoding="utf-8"))
    doc_b["canonical_body"]["pre_llm_state"]["candidate_structures"][0]["contractual_max_loss_rs"] = 1.0
    snap_b.write_text(json.dumps(doc_b), encoding="utf-8")
    with pytest.raises(SnapshotTamperOrMismatchError, match="EVIDENCE_SHA256_MISMATCH"):
        run_analyst_pipeline(dec_dt, cfg_b, wall_clock_fn=lambda: t0 + dt.timedelta(seconds=5))

    # Case C: Wrong `evidence_sha256` in `artifact.json`
    cfg_c, art_c, _, _ = _fresh_bundle("bad_sha")
    art_doc_c = json.loads(art_c.read_text(encoding="utf-8"))
    art_doc_c["evidence_sha256"] = "0" * 64
    art_c.write_text(json.dumps(art_doc_c), encoding="utf-8")
    with pytest.raises(SnapshotTamperOrMismatchError, match="stale or foreign artifact"):
        run_analyst_pipeline(dec_dt, cfg_c, wall_clock_fn=lambda: t0 + dt.timedelta(seconds=5))

    # Case D: Wrong `snapshot_id` in `artifact.json`
    cfg_d, art_d, _, _ = _fresh_bundle("bad_snap_id")
    art_doc_d = json.loads(art_d.read_text(encoding="utf-8"))
    art_doc_d["snapshot_id"] = "snap_forged_000000"
    art_d.write_text(json.dumps(art_doc_d), encoding="utf-8")
    with pytest.raises(SnapshotTamperOrMismatchError, match="stale or foreign artifact"):
        run_analyst_pipeline(dec_dt, cfg_d, wall_clock_fn=lambda: t0 + dt.timedelta(seconds=5))

    # Case E: Attacker replaces BOTH `.snapshot.json` AND expected hash in `.request.json` + `artifact.json`!
    cfg_e, art_e, snap_e, req_e = _fresh_bundle("dual_replace")
    os.chmod(snap_e, 0o644)
    os.chmod(req_e, 0o644)
    doc_e = json.loads(snap_e.read_text(encoding="utf-8"))
    doc_e["canonical_body"]["pre_llm_state"]["contract_context"]["spot"]["value"] = 25999.0
    forged_sha = sha256_hex_of_obj(doc_e["canonical_body"])
    doc_e["evidence_sha256"] = forged_sha
    snap_e.write_text(json.dumps(doc_e), encoding="utf-8")

    req_doc_e = json.loads(req_e.read_text(encoding="utf-8"))
    req_doc_e["evidence_sha256"] = forged_sha
    req_e.write_text(json.dumps(req_doc_e), encoding="utf-8")

    art_doc_e = json.loads(art_e.read_text(encoding="utf-8"))
    art_doc_e["evidence_sha256"] = forged_sha
    art_e.write_text(json.dumps(art_doc_e), encoding="utf-8")

    with pytest.raises(SnapshotTamperOrMismatchError, match="SNAPSHOT_PIN_MISMATCH"):
        run_analyst_pipeline(dec_dt, cfg_e, wall_clock_fn=lambda: t0 + dt.timedelta(seconds=5))


def test_item2_write_once_immutability_and_secret_scrubbing(tmp_path: Path) -> None:
    """Attempted overwrite raises SnapshotOverwriteBlockedError leaving original bytes unchanged; secrets scrubbed."""
    dec_dt = dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST)
    t0 = dt.datetime(2026, 11, 6, 10, 15, 2, tzinfo=IST)
    art = tmp_path / "imm.json"

    secret_token = "SECRET_KOTAK_SESSION_TOKEN_987654"
    cfg = AnalystConfig(
        mode="OFFLINE",
        fixtures_dir=str(FIXTURES_DIR),
        audit_log_path=str(tmp_path / "imm_audit.jsonl"),
        llm_provider="agent_artifact",
        agent_synthesis_artifact_path=str(art),
        _runtime_secrets={"KOTAK_ACCESS_TOKEN": secret_token},
    )
    with pytest.raises(FileNotFoundError):
        run_analyst_pipeline(dec_dt, cfg, wall_clock_fn=lambda: t0)

    snap_p, req_p, pin_p, _ = snapshot_paths_for_artifact(art)
    orig_snap_bytes = snap_p.read_bytes()
    orig_req_bytes = req_p.read_bytes()
    assert secret_token.encode("utf-8") not in orig_snap_bytes
    assert secret_token.encode("utf-8") not in orig_req_bytes
    assert (stat.S_IMODE(snap_p.stat().st_mode) & stat.S_IWUSR) == 0

    with pytest.raises(SnapshotOverwriteBlockedError):
        write_once_json(snap_p, {"tampered": True})
    with pytest.raises(SnapshotOverwriteBlockedError):
        write_once_json(req_p, {"tampered": True})
    with pytest.raises(SnapshotOverwriteBlockedError):
        write_once_json(pin_p, {"tampered": True})

    # Re-polling while artifact is still absent does NOT overwrite snapshot or request bytes!
    with pytest.raises(FileNotFoundError):
        run_analyst_pipeline(
            dec_dt,
            cfg,
            broker_adapter=_ForbiddenBrokerAdapter(),
            wall_clock_fn=lambda: t0 + dt.timedelta(seconds=15),
        )

    assert snap_p.read_bytes() == orig_snap_bytes
    assert req_p.read_bytes() == orig_req_bytes

    # Regression guard: 64-char hex SHA256 containing a 10-digit run (e.g. `b6903574982f`)
    # must NEVER be corrupted by Indian mobile-number scrubbing, while real phone numbers ARE redacted!
    scrubber = cfg.build_scrubber()
    tricky_sha = "00f76818d0706e86acf87f61b0a4faea0ca6afdd2f2fcef66217b6903574982f"
    assert scrubber.scrub_text(tricky_sha) == tricky_sha
    assert "[REDACTED_PHONE]" in scrubber.scrub_text("Call +91-9876543210 or 8765432109")


def test_item2_crash_restart_resume_zero_broker_reads_and_identity_propagation(
    tmp_path: Path,
) -> None:
    """Crash/restart resumes exact stored bytes & original capture-start clock with ZERO broker reads."""
    fx_copy = tmp_path / "ephemeral_fx"
    shutil.copytree(FIXTURES_DIR, fx_copy)

    dec_dt = dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST)
    t_start = dt.datetime(2026, 11, 6, 10, 15, 3, tzinfo=IST)
    t_end = dt.datetime(2026, 11, 6, 10, 15, 17, tzinfo=IST)
    capture_ticks = iter([t_start, t_end])

    art = tmp_path / "resume_art.json"
    audit_path = tmp_path / "resume_audit.jsonl"
    cfg = AnalystConfig(
        mode="OFFLINE",
        fixtures_dir=str(fx_copy),
        audit_log_path=str(audit_path),
        llm_provider="agent_artifact",
        agent_synthesis_artifact_path=str(art),
    )

    # Phase A: capture once -> persists immutable snapshot & exits waiting for external artifact
    with pytest.raises(FileNotFoundError):
        run_analyst_pipeline(dec_dt, cfg, wall_clock_fn=lambda: next(capture_ticks, t_end))

    snap_p, req_p, _, _ = snapshot_paths_for_artifact(art)
    orig_snap_bytes = snap_p.read_bytes()
    orig_req_bytes = req_p.read_bytes()
    req_doc = json.loads(orig_req_bytes.decode("utf-8"))
    assert req_doc["capture_started_at_ist"] == t_start.isoformat()
    assert req_doc["capture_completed_at_ist"] == t_end.isoformat()

    # Delete market fixture directory completely to prove resume reads ONLY stored snapshot!
    shutil.rmtree(fx_copy)
    _write_valid_agent_artifact(art, req_doc)

    t_resume = t_start + dt.timedelta(seconds=95)
    state, delivery = run_analyst_pipeline(
        dec_dt,
        cfg,
        broker_adapter=_ForbiddenBrokerAdapter(),
        wall_clock_fn=lambda: t_resume,
        force_render_for_inspection=True,
    )

    assert snap_p.read_bytes() == orig_snap_bytes
    assert req_p.read_bytes() == orig_req_bytes
    assert state.run_metadata.capture_started_at_ist == t_start.isoformat()
    assert state.run_metadata.capture_completed_at_ist == t_end.isoformat()
    assert state.run_metadata.snapshot_id == req_doc["snapshot_id"]
    assert state.run_metadata.evidence_sha256 == req_doc["evidence_sha256"]
    assert state.run_metadata.run_id == req_doc["run_id"]
    assert req_doc["snapshot_id"] in delivery.full_report_markdown
    assert req_doc["evidence_sha256"] in delivery.full_report_markdown

    audit_row = json.loads(audit_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert audit_row["snapshot_id"] == req_doc["snapshot_id"]
    assert audit_row["evidence_sha256"] == req_doc["evidence_sha256"]


def test_item2_fake_clock_180s_from_actual_capture_start_and_bad_clocks(tmp_path: Path) -> None:
    """Enforces 180.0s strictly from actual capture_started_at_ist (not capture_end or decision_dt) via fake clock."""
    # Ensure ZERO real sleep calls appear in this test
    assert ("time." + "sleep") not in inspect.getsource(
        test_item2_fake_clock_180s_from_actual_capture_start_and_bad_clocks
    )

    dec_dt = dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST)
    # Actual capture starts 30s BEFORE decision_dt and finishes 25s later -> distinguishes all 3 clocks!
    t_capture_start = dt.datetime(2026, 11, 6, 10, 14, 30, tzinfo=IST)
    t_capture_end = dt.datetime(2026, 11, 6, 10, 14, 55, tzinfo=IST)
    ticks = iter([t_capture_start, t_capture_end])

    art = tmp_path / "bound180.json"
    cfg = AnalystConfig(
        mode="OFFLINE",
        fixtures_dir=str(FIXTURES_DIR),
        audit_log_path=str(tmp_path / "b180.jsonl"),
        llm_provider="agent_artifact",
        agent_synthesis_artifact_path=str(art),
    )

    with pytest.raises(FileNotFoundError):
        run_analyst_pipeline(dec_dt, cfg, wall_clock_fn=lambda: next(ticks, t_capture_end))

    _, req_p, _, _ = snapshot_paths_for_artifact(art)
    req_doc = json.loads(req_p.read_text(encoding="utf-8"))
    _write_valid_agent_artifact(art, req_doc)

    # 1. Exact allowed boundary: `t_capture_start + 180.0s` -> SUCCEEDS
    state_ok, _ = run_analyst_pipeline(
        dec_dt,
        cfg,
        broker_adapter=_ForbiddenBrokerAdapter(),
        wall_clock_fn=lambda: t_capture_start + dt.timedelta(seconds=180.0),
    )
    assert state_ok.final_output is not None
    assert state_ok.final_output.verdict == "SUGGESTION"

    # 2. `t_capture_start + 180.001s` (only 155.001s after capture_end & 150.001s after dec_dt!) -> REJECTED!
    with pytest.raises(StaleEvidenceSnapshotError, match="SNAPSHOT_RESUME_EXPIRED_180S"):
        run_analyst_pipeline(
            dec_dt,
            cfg,
            broker_adapter=_ForbiddenBrokerAdapter(),
            wall_clock_fn=lambda: t_capture_start + dt.timedelta(seconds=180.001),
        )

    # 3. Retrying or rewriting artifact does NOT reset the capture-start clock!
    _write_valid_agent_artifact(art, req_doc)
    with pytest.raises(StaleEvidenceSnapshotError, match="SNAPSHOT_RESUME_EXPIRED_180S"):
        run_analyst_pipeline(
            dec_dt,
            cfg,
            broker_adapter=_ForbiddenBrokerAdapter(),
            wall_clock_fn=lambda: t_capture_start + dt.timedelta(seconds=185.0),
        )

    # 4. Future capture clock (`resume_now < capture_started_at_ist`) -> REJECTED!
    with pytest.raises(EvidenceSnapshotClockError, match="FUTURE_CAPTURE_CLOCK"):
        run_analyst_pipeline(
            dec_dt,
            cfg,
            broker_adapter=_ForbiddenBrokerAdapter(),
            wall_clock_fn=lambda: t_capture_start - dt.timedelta(seconds=1.0),
        )

    # 5. Naive resume clock (`tzinfo=None`) -> REJECTED!
    with pytest.raises(EvidenceSnapshotClockError, match="NAIVE_RESUME_CLOCK"):
        run_analyst_pipeline(
            dec_dt,
            cfg,
            broker_adapter=_ForbiddenBrokerAdapter(),
            wall_clock_fn=lambda: dt.datetime(2026, 11, 6, 10, 15, 10),
        )

    # 6. Naive initial capture clock & inverted capture clock (`completed < started`) -> REJECTED!
    cfg_naive = AnalystConfig(
        mode="OFFLINE",
        fixtures_dir=str(FIXTURES_DIR),
        audit_log_path=str(tmp_path / "naive.jsonl"),
        llm_provider="agent_artifact",
        agent_synthesis_artifact_path=str(tmp_path / "naive_art.json"),
    )
    with pytest.raises(EvidenceSnapshotClockError):
        run_analyst_pipeline(dec_dt, cfg_naive, wall_clock_fn=lambda: dt.datetime(2026, 11, 6, 10, 15, 0))

    inv_ticks = iter([t_capture_end, t_capture_start])
    cfg_inv = AnalystConfig(
        mode="OFFLINE",
        fixtures_dir=str(FIXTURES_DIR),
        audit_log_path=str(tmp_path / "inv.jsonl"),
        llm_provider="agent_artifact",
        agent_synthesis_artifact_path=str(tmp_path / "inv_art.json"),
    )
    with pytest.raises(EvidenceSnapshotClockError, match="INCONSISTENT_CAPTURE_CLOCK_ORDER"):
        run_analyst_pipeline(dec_dt, cfg_inv, wall_clock_fn=lambda: next(inv_ticks, t_capture_start))


# ============================================================================
# ITEM 3 & ITEM 4: SOURCE-ORDER GUARD + SMALL TOUCH INDICATIVE VS BOOK WALK
# ============================================================================


def test_item3_source_order_guard_preserved_and_structurally_valid_qualified() -> None:
    """Monotonic source-order guard rejects unsorted levels without sorting; qualifies structural validity."""
    bad_buy = {
        "mktLot": 65,
        "depth": {
            "buy": [{"price": "95.0", "quantity": "65"}, {"price": "100.0", "quantity": "65"}],
            "sell": [{"price": "101.0", "quantity": "65"}, {"price": "102.0", "quantity": "65"}],
        },
    }
    res_bad = interpret_depth_book(bad_buy, fallback_lot_size=65)
    assert res_bad.is_structurally_valid is False
    assert res_bad.is_executable is False
    assert res_bad.best_bid is None and res_bad.best_ask is None
    assert res_bad.rejection_reason == "DEPTH_LEVEL_ORDER_INCONSISTENT"
    assert res_bad.raw_payload is not None

    dup_ordered = {
        "mktLot": 65,
        "depth": {
            "buy": [
                {"price": "100.0", "quantity": "25"},
                {"price": "100.0", "quantity": "40"},
                {"price": "0", "quantity": "0"},
            ],
            "sell": [
                {"price": "101.0", "quantity": "65"},
                {"price": "0", "quantity": "0"},
            ],
        },
    }
    res_dup = interpret_depth_book(dup_ordered, fallback_lot_size=65)
    assert res_dup.is_structurally_valid is True
    assert res_dup.touch_bid_qty == 65
    assert res_dup.touch_ask_qty == 65
    assert res_dup.visible_buy_qty == 65


def test_item4_book_walk_reproduction_and_rule4_rejection_on_deep_slippage() -> None:
    """Reproduces 1 unit at 100/101 + 649 units at 90/110 (lot_size=65): walks to ~90.154 / ~109.862 & fails Rule 4."""
    book = {
        "mktLot": 65,
        "depth": {
            "buy": [{"price": "100", "quantity": "1"}, {"price": "90", "quantity": "649"}],
            "sell": [{"price": "101", "quantity": "1"}, {"price": "110", "quantity": "649"}],
        },
    }
    parsed = interpret_depth_book(book, fallback_lot_size=65)
    assert parsed.is_structurally_valid is True
    assert parsed.best_bid == 100.0 and parsed.best_ask == 101.0
    assert parsed.touch_bid_qty == 1 and parsed.touch_ask_qty == 1
    assert parsed.visible_buy_qty == 650 and parsed.visible_sell_qty == 650
    assert parsed.bid_depth_lots == 10 and parsed.ask_depth_lots == 10

    bid_vwap, avail_b = walk_depth_levels(parsed.buy_levels, 65)
    ask_vwap, avail_s = walk_depth_levels(parsed.sell_levels, 65)
    assert avail_b == 650 and avail_s == 650
    assert bid_vwap is not None and abs(bid_vwap - ((100.0 * 1 + 90.0 * 64) / 65.0)) < 1e-6
    assert ask_vwap is not None and abs(ask_vwap - ((101.0 * 1 + 110.0 * 64) / 65.0)) < 1e-6
    assert abs(bid_vwap - 90.153846) < 1e-4
    assert abs(ask_vwap - 109.861538) < 1e-4

    # Construct 200-pt Bull Put Spread (lot_size=65) whose touch quotes look <= Rs 10,000 max loss
    # (touch net credit ~64.5 pts -> touch max loss ~Rs 8,975), while 65-unit book walk yields
    # net credit ~34.5 pts -> walked contractual max loss ~Rs 10,920 > Rs 10,000 -> Rule 4 FAILS!
    q_ts = "2026-11-06T10:15:00+05:30"
    short_raw = RawStrikeQuote(
        strike=24200.0,
        call_bid=250.0,
        call_ask=252.0,
        put_bid=100.0,
        put_ask=101.0,
        call_oi=50000,
        put_oi=90000,
        call_volume=10000,
        put_volume=25000,
        call_depth_lots=10,
        put_depth_lots=10,
        quote_timestamp_ist=q_ts,
        put_touch_bid_qty=1,
        put_touch_ask_qty=1,
        put_visible_buy_qty=650,
        put_visible_sell_qty=650,
        put_buy_levels=[[100.0, 1], [84.0, 649]],
        put_sell_levels=[[101.0, 1], [110.0, 649]],
    )
    long_raw = RawStrikeQuote(
        strike=24000.0,
        call_bid=420.0,
        call_ask=422.0,
        put_bid=34.0,
        put_ask=35.0,
        call_oi=30000,
        put_oi=80000,
        call_volume=8000,
        put_volume=20000,
        call_depth_lots=10,
        put_depth_lots=10,
        quote_timestamp_ist=q_ts,
        put_touch_bid_qty=1,
        put_touch_ask_qty=1,
        put_visible_buy_qty=650,
        put_visible_sell_qty=650,
        put_buy_levels=[[34.0, 1], [25.0, 649]],
        put_sell_levels=[[35.0, 1], [50.0, 649]],
    )

    cfg_u = AnalystConfig()
    short_leg = build_option_leg(
        short_raw, "PE", "SHORT", 1, 24250.0, 10.0 / 365.0, 0.067, False, cfg_u
    )
    long_leg = build_option_leg(
        long_raw, "PE", "LONG", 1, 24250.0, 10.0 / 365.0, 0.067, False, cfg_u
    )

    ps = PriceStructureEvidence(
        trend_1d_state="UPTREND",
        higher_high_higher_low_repaired=True,
        first_pullback_in_downtrend_warning=False,
        ma20_1h=24180.0,
        ema50_1h=24120.0,
        last_completed_1d_bar=None,
        last_completed_1h_bar=None,
        final_15m_bar_completed=True,
        stripped_forming_bars_count=0,
        verified_level_touches=[],
    )
    vol = VolatilityEvidence(
        india_vix=TimestampedDatum.create(
            14.2,
            dt.datetime(2026, 11, 6, 10, 15, tzinfo=IST),
            dt.datetime(2026, 11, 6, 10, 15, tzinfo=IST),
            "vix",
            180,
            "VERIFIED",
        ),
        atm_iv_by_expiry={"2026-11-26": 0.14},
        constant_maturity_30d_iv=0.14,
        eod_cm30_latest=0.14,
        iv_source_label="LIVE_MID_INFERRED",
        ivp_30d=60.0,
        ivp_90d=62.0,
        ivp_252d=58.0,
        iv_history_sessions_count=252,
        iv_history_gaps_flagged=[],
        realized_vol_20d=0.12,
        iv_rv_spread=0.02,
        skew_25d_put_minus_call=0.01,
        wing_curvature=0.005,
        chain_qa_passed=True,
        chain_qa_notes=[],
    )
    exp_meta = ExpiryMetadata(
        expiry_date="2026-11-26",
        expiry_timestamp_ist="2026-11-26T15:30:00+05:30",
        dte_trading_sessions=10,
        lot_size=65,
        is_holiday_adjusted=False,
        is_monthly=True,
    )
    port_state = build_portfolio_exposure_state([], portfolio_cap_rs=10000.0)

    cand = build_candidate_structure(
        structure_id="BPS_2026-11-26_24200_24000",
        structure_type="BULL_PUT_SPREAD",
        direction="BULLISH",
        expiry=exp_meta,
        legs=[short_leg, long_leg],
        spot=24240.0,
        implied_forward=24250.0,
        atm_iv=0.14,
        time_to_expiry_years=10.0 / 365.0,
        empirical_daily_log_returns=[0.004, -0.003] * 25,
        invalidation_level=24150.0,
        invalidation_candle_basis="COMPLETED_1H_CLOSE",
        invalidation_justification="Below 1H support.",
        matches_part_b_style=True,
        portfolio_state=port_state,
        price_structure=ps,
        volatility=vol,
        config=cfg_u,
    )

    # Touch scenario alone was <= Rs 10,000, but 65-unit book walk exceeds Rs 10,000 -> Rule 4 FAILS!
    assert cand.indicative_touch_max_loss_rs < 10000.0
    assert cand.execution_fill_basis == "FULL_SIZE_BOOK_WALK"
    assert cand.full_size_fill_supported is True
    assert cand.full_size_contractual_max_loss_rs is not None
    assert cand.full_size_contractual_max_loss_rs > 10000.0
    assert cand.part_a_checks["RULE_4_CONTRACTUAL_MAX_LOSS_10000_RS"].status == "FAIL"
    assert cand.part_a_overall_status == "FAIL"


def test_item4_small_touch_indicative_kept_not_blanket_halted_and_never_certifies_rule4(
    tmp_path: Path,
) -> None:
    """Small available touch quantity (< 1 lot) stays available as INDICATIVE, never halts, never PASSes Rule 4."""
    assert AnalystConfig().min_top_book_depth_lots == 0

    fx_thin = tmp_path / "fx_thin"
    shutil.copytree(FIXTURES_DIR, fx_thin)
    chains_file = fx_thin / "option_chains.json"
    chains_obj = json.loads(chains_file.read_text(encoding="utf-8"))
    for exp_chain in chains_obj["chains_by_expiry"].values():
        for row in exp_chain["strikes"]:
            cb, ca = float(row["call_bid"]), float(row["call_ask"])
            pb, pa = float(row["put_bid"]), float(row["put_ask"])
            row["call_depth_lots"] = 0
            row["put_depth_lots"] = 0
            row["call_touch_bid_qty"] = 2
            row["call_touch_ask_qty"] = 2
            row["call_visible_buy_qty"] = 2
            row["call_visible_sell_qty"] = 2
            row["call_buy_levels"] = [[cb, 2]]
            row["call_sell_levels"] = [[ca, 2]]
            row["put_touch_bid_qty"] = 2
            row["put_touch_ask_qty"] = 2
            row["put_visible_buy_qty"] = 2
            row["put_visible_sell_qty"] = 2
            row["put_buy_levels"] = [[pb, 2]]
            row["put_sell_levels"] = [[pa, 2]]
    chains_file.write_text(json.dumps(chains_obj), encoding="utf-8")

    cfg = AnalystConfig(
        mode="OFFLINE",
        fixtures_dir=str(fx_thin),
        audit_log_path=str(tmp_path / "thin_audit.jsonl"),
    )
    dec_dt = dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST)
    state, delivery = run_analyst_pipeline(dec_dt, cfg, force_render_for_inspection=True)

    # NOT blanket-halted: candidates are retained for research with explicit INDICATIVE scenario labels!
    assert state.final_output is not None
    assert state.final_output.verdict == "WAIT"
    assert len(state.candidate_structures) > 0
    for c in state.candidate_structures:
        assert c.execution_fill_basis == "INDICATIVE_TOUCH_ONLY"
        assert c.full_size_fill_supported is False
        assert c.full_size_executable_net_points_per_unit is None
        assert c.full_size_contractual_max_loss_rs is None
        assert "INDICATIVE_TOUCH_SCENARIO_ONLY" in c.price_assumption_label
        assert c.part_a_checks["RULE_4_CONTRACTUAL_MAX_LOSS_10000_RS"].status in ("UNKNOWN", "FAIL")
        assert c.part_a_checks["RULE_4_CONTRACTUAL_MAX_LOSS_10000_RS"].status != "PASS"
        assert c.part_a_overall_status != "PASS"
        for leg in c.legs:
            assert leg.touch_available_qty == 2
            assert leg.visible_depth_qty == 2

    assert state.final_output.is_live_ready is False
    assert delivery.alerted_trader is False
    rep = render_analyst_report(state)
    assert "INDICATIVE_TOUCH_SCENARIO_ONLY" in rep


# ============================================================================
# ITEM 5: KEEP LIVE READINESS OFF UNTIL THE OTHER LIVE GATES PASS
# ============================================================================


class _CleanTimestampLiveMockAdapter(FixtureKotakNeoAdapter):
    """Simulates a live adapter even if quote timestamps look exchange-verified, to prove other gates hold."""

    def fetch_option_chain_raw(
        self, expiry_date: str, decision_dt_ist: dt.datetime
    ) -> dict[str, Any]:
        raw = json.loads(json.dumps(super().fetch_option_chain_raw(expiry_date, decision_dt_ist)))
        raw["source_verification_tag"] = "EXCHANGE_FEED_TIMESTAMP"
        raw["priced_neo_symbols"] = [f"nse_fo|{expiry_date}_ATM"]
        raw["chain_neo_symbols"] = [f"nse_fo|{expiry_date}_ATM"]
        return raw


def test_item5_live_readiness_gates_off_degraded_surface_uncovered_held_and_alerts_off(
    tmp_path: Path,
) -> None:
    """Live mode stays RESEARCH-ONLY, documents SINGLE_EXPIRY_DEGRADED + uncovered held contracts, alerts OFF."""
    assert PacedReadTransport.HTTP_BOUNDARY_PACING_UNVERIFIED is True

    dec_dt = dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST)
    pos_file = tmp_path / "open_pos.json"
    pos_file.write_text("[]", encoding="utf-8")

    # 1. Held contract on non-anchor expiry (`nse_fo|99999`) under default small-batch live mode (`batch_size=1`)
    held_far = OpenPosition(
        position_id="HELD_FAR_EXPIRY_IC",
        structure_type="IRON_CONDOR",
        directional_sign=0,
        planned_stop_rs=1500.0,
        current_executable_spread_value_rs=1200.0,
        short_leg_deltas=[-0.15, 0.14],
        dte_trading_sessions=8,
        neo_symbols=["nse_fo|99999"],
    )
    cfg_live = AnalystConfig(
        mode="LIVE_MORNING_SNAPSHOT",
        fixtures_dir=str(FIXTURES_DIR),
        open_positions_file=str(pos_file),
        audit_log_path=str(tmp_path / "live_audit.jsonl"),
        kotak_quotes_batch_size=1,
    )
    adapter = _CleanTimestampLiveMockAdapter(FIXTURES_DIR)
    state_live, deliv_live = run_analyst_pipeline(
        dec_dt,
        cfg_live,
        broker_adapter=adapter,
        open_positions=[held_far],
        force_render_for_inspection=True,
    )

    assert state_live.run_metadata.surface_coverage_status == "SINGLE_EXPIRY_DEGRADED"
    assert state_live.run_metadata.uncovered_held_neo_symbols == ["nse_fo|99999"]
    assert state_live.final_output is not None
    assert state_live.final_output.is_live_ready is False
    assert "LIVE_GATES_UNVERIFIED" in state_live.final_output.research_only_reason
    assert "UNCOVERED_HELD_CONTRACTS" in state_live.final_output.research_only_reason
    assert deliv_live.alerted_trader is False
    assert "SINGLE_EXPIRY_DEGRADED" in deliv_live.full_report_markdown
    assert "nse_fo|99999" in deliv_live.full_report_markdown

    # 2. Even with ZERO held contracts and clean quote tags, LIVE_MORNING_SNAPSHOT keeps `is_live_ready=False`!
    state_live2, deliv_live2 = run_analyst_pipeline(
        dec_dt,
        cfg_live,
        broker_adapter=_CleanTimestampLiveMockAdapter(FIXTURES_DIR),
        open_positions=[],
        force_render_for_inspection=True,
    )
    assert state_live2.final_output is not None
    assert state_live2.final_output.is_live_ready is False
    assert "LIVE_GATES_UNVERIFIED" in state_live2.final_output.research_only_reason
    assert deliv_live2.alerted_trader is False
    assert "LIVE-READY" not in state_live2.final_output.confidence_note
    assert "LIVE-READY" not in state_live2.final_output.whatsapp_digest

    # 3. `decision_driver.run_scheduled_slot` defaults to `suppress_trader_alerts=True` (recommendations OFF)
    sig = inspect.signature(run_scheduled_slot)
    assert sig.parameters["suppress_trader_alerts"].default is True

    cfg_off = AnalystConfig(
        mode="OFFLINE",
        fixtures_dir=str(FIXTURES_DIR),
        audit_log_path=str(tmp_path / "slot_default_off.jsonl"),
        cache_dir=str(tmp_path / "slot_cache"),
    )
    slot_res = run_scheduled_slot(dec_dt, cfg_off)
    assert slot_res.executed is True
    assert slot_res.delivery is not None
    assert slot_res.delivery.alerted_trader is False
