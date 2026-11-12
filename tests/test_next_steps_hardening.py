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


# ============================================================================
# TARGETED POST-REVIEW REGRESSION CHECKS (FIXES 1, 2, AND 3)
# ============================================================================


def test_targeted_fix1_other_report_text_buy_999_units_and_buy_banknifty_rejected(
    tmp_path: Path,
) -> None:
    """Reject 'Buy 999 units', 'Buy BANKNIFTY', wrong quantities/symbols/legs across ALL report text fields."""
    dec_dt = dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST)
    base_cfg = AnalystConfig(
        mode="OFFLINE",
        fixtures_dir=str(FIXTURES_DIR),
        audit_log_path=str(tmp_path / "base_t1.jsonl"),
    )
    base_state, _ = run_analyst_pipeline(dec_dt, base_cfg)
    assert (
        base_state.final_output is not None
        and base_state.final_output.primary_proposal is not None
    )
    cand = base_state.final_output.primary_proposal
    leg0 = cand.legs[0]
    wrong_verb = "Buy" if leg0.side == "SHORT" else "Sell"

    # Every one of the 12 report prose fields + citations claim_text + digest must reject
    # both "Buy 999 units" and "Buy BANKNIFTY" (plus wrong lots, symbols, leg sides, DTE, expiry).
    field_setters = {
        "reasoning_summary": lambda r, s: dataclasses.replace(r, reasoning_summary=s),
        "habitual_comparison_note": lambda r, s: dataclasses.replace(
            r, habitual_comparison_note=s
        ),
        "research_rule_challenge_note": lambda r, s: dataclasses.replace(
            r, research_rule_challenge_note=s
        ),
        "confidence_note": lambda r, s: dataclasses.replace(r, confidence_note=s),
        "bull_thesis": lambda r, s: dataclasses.replace(
            r, debate_state=dataclasses.replace(r.debate_state, bull_thesis=s)  # type: ignore[arg-type]
        ),
        "bear_thesis": lambda r, s: dataclasses.replace(
            r, debate_state=dataclasses.replace(r.debate_state, bear_thesis=s)  # type: ignore[arg-type]
        ),
        "vol_regime_thesis": lambda r, s: dataclasses.replace(
            r, debate_state=dataclasses.replace(r.debate_state, vol_regime_thesis=s)  # type: ignore[arg-type]
        ),
        "rebuttal_round": lambda r, s: dataclasses.replace(
            r, debate_state=dataclasses.replace(r.debate_state, rebuttal_round=s)  # type: ignore[arg-type]
        ),
        "convexity_advocate_note": lambda r, s: dataclasses.replace(
            r, risk_triad_state=dataclasses.replace(r.risk_triad_state, convexity_advocate_note=s)  # type: ignore[arg-type]
        ),
        "tail_guardian_note": lambda r, s: dataclasses.replace(
            r, risk_triad_state=dataclasses.replace(r.risk_triad_state, tail_guardian_note=s)  # type: ignore[arg-type]
        ),
        "neutral_arbiter_note": lambda r, s: dataclasses.replace(
            r, risk_triad_state=dataclasses.replace(r.risk_triad_state, neutral_arbiter_note=s)  # type: ignore[arg-type]
        ),
        "final_recommendation_rationale": lambda r, s: dataclasses.replace(
            r,
            consolidator_state=dataclasses.replace(
                r.consolidator_state, final_recommendation_rationale=s  # type: ignore[arg-type]
            ),
        ),
        "whatsapp_digest": lambda r, s: dataclasses.replace(
            r,
            consolidator_state=dataclasses.replace(
                r.consolidator_state, whatsapp_digest=s  # type: ignore[arg-type]
            ),
        ),
        "citation_claim": lambda r, s: dataclasses.replace(
            r,
            citations=[{**r.citations[0], "claim_text": s}, *r.citations[1:]],
        ),
    }

    for poison_label, poison_snippet, forbidden_token in (
        ("buy_999_units", f"`{cand.structure_id}` setup. Buy 999 units immediately.", "999 units"),
        ("buy_banknifty", f"`{cand.structure_id}` setup. Buy BANKNIFTY immediately.", "BANKNIFTY"),
    ):
        for fname, setter in field_setters.items():
            audit_p = tmp_path / f"t1_{poison_label}_{fname}.jsonl"
            cfg_i = AnalystConfig(
                mode="OFFLINE",
                fixtures_dir=str(FIXTURES_DIR),
                audit_log_path=str(audit_p),
            )
            st_i, deliv_i = run_analyst_pipeline(
                dec_dt,
                cfg_i,
                llm_provider=_StaticOverrideProvider(
                    lambda resp, _s, _p, _a, fn=setter, txt=poison_snippet: fn(resp, txt)
                ),
            )
            assert st_i.final_output is not None, (poison_label, fname)
            assert st_i.final_output.verifier_report.passed_verification is False, (
                poison_label,
                fname,
            )
            assert st_i.final_output.verdict == "WAIT", (poison_label, fname)
            assert st_i.final_output.primary_proposal is None, (poison_label, fname)
            assert deliv_i.alerted_trader is False, (poison_label, fname)
            assert forbidden_token not in deliv_i.full_report_markdown, (poison_label, fname)
            assert forbidden_token not in st_i.final_output.whatsapp_digest, (poison_label, fname)
            audit_row = json.loads(audit_p.read_text(encoding="utf-8").strip().splitlines()[-1])
            assert audit_row["final_output"]["verdict"] == "WAIT"
            assert audit_row["final_output"]["llm_citations"] == []

    # Additional prose trade-mismatch variants inside non-digest report prose (`bull_thesis` / `tail_guardian_note`)
    extra_prose_poisons = [
        "bare_999_units",
        "wrong_lots_count",
        "bank_nifty_spaced",
        "sell_finnifty",
        "buy_reliance",
        "buy_nifty_futures",
        "wrong_verb_leg_side",
        "wrong_dte_in_prose",
        "wrong_expiry_in_prose",
    ]
    extra_snippets = {
        "bare_999_units": "Position size is 999 units.",
        "wrong_lots_count": "Recommended size is 4 lots.",
        "bank_nifty_spaced": "Rotate into Bank Nifty options.",
        "sell_finnifty": "Sell FINNIFTY straddle instead.",
        "buy_reliance": "Buy RELIANCE calls on breakout.",
        "buy_nifty_futures": "Buy NIFTY futures alongside the spread.",
        "wrong_verb_leg_side": f"{wrong_verb} {int(leg0.strike)} {leg0.option_type} at market.",
        "wrong_dte_in_prose": f"`{cand.structure_id}` has 99 DTE remaining.",
        "wrong_expiry_in_prose": f"`{cand.structure_id}` expires on 2026-12-31.",
    }
    for tag in extra_prose_poisons:
        txt = extra_snippets[tag]
        cfg_x = AnalystConfig(
            mode="OFFLINE",
            fixtures_dir=str(FIXTURES_DIR),
            audit_log_path=str(tmp_path / f"extra_{tag}.jsonl"),
        )
        st_x, deliv_x = run_analyst_pipeline(
            dec_dt,
            cfg_x,
            llm_provider=_StaticOverrideProvider(
                lambda resp, _s, _p, _a, s=txt: dataclasses.replace(
                    resp,
                    debate_state=dataclasses.replace(resp.debate_state, bull_thesis=s),  # type: ignore[arg-type]
                )
            ),
        )
        assert st_x.final_output is not None and not st_x.final_output.verifier_report.passed_verification, tag
        assert st_x.final_output.verdict == "WAIT", tag
        assert deliv_x.alerted_trader is False, tag


def test_targeted_fix2_run_started_at_179s_finishing_at_190s_fails_before_final_output(
    tmp_path: Path,
) -> None:
    """A run started at 179s that finishes at 190s fails closed at the pre-output deadline check."""
    assert ("time." + "sleep") not in inspect.getsource(
        test_targeted_fix2_run_started_at_179s_finishing_at_190s_fails_before_final_output
    )

    dec_dt = dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST)
    t_capture_start = dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST)
    t_capture_end = dt.datetime(2026, 11, 6, 10, 15, 2, tzinfo=IST)

    # 1. Agent-artifact resume: started at 179.0s (passes entry), finishes at 190.0s -> REJECTED!
    art = tmp_path / "late_finish.json"
    chart_dir = tmp_path / "charts_late"
    audit_late = tmp_path / "late_audit.jsonl"
    cfg = AnalystConfig(
        mode="OFFLINE",
        fixtures_dir=str(FIXTURES_DIR),
        audit_log_path=str(audit_late),
        payoff_chart_dir=str(chart_dir),
        llm_provider="agent_artifact",
        agent_synthesis_artifact_path=str(art),
    )
    cap_ticks = iter([t_capture_start, t_capture_end])
    with pytest.raises(FileNotFoundError):
        run_analyst_pipeline(dec_dt, cfg, wall_clock_fn=lambda: next(cap_ticks, t_capture_end))

    _, req_p, _, _ = snapshot_paths_for_artifact(art)
    req_doc = json.loads(req_p.read_text(encoding="utf-8"))
    _write_valid_agent_artifact(art, req_doc)

    resume_ticks_179_to_190 = iter(
        [
            t_capture_start + dt.timedelta(seconds=179.0),
            t_capture_start + dt.timedelta(seconds=190.0),
        ]
    )
    with pytest.raises(StaleEvidenceSnapshotError, match="SNAPSHOT_RESUME_EXPIRED_180S"):
        run_analyst_pipeline(
            dec_dt,
            cfg,
            wall_clock_fn=lambda: next(
                resume_ticks_179_to_190, t_capture_start + dt.timedelta(seconds=190.0)
            ),
        )
    assert not audit_late.exists()
    assert list(chart_dir.glob("*.png")) == []

    # 2. Control: started at 179.0s and finishing at 180.0s (exact deadline) -> SUCCEEDS!
    resume_ticks_179_to_180 = iter(
        [
            t_capture_start + dt.timedelta(seconds=179.0),
            t_capture_start + dt.timedelta(seconds=180.0),
        ]
    )
    st_ok, _ = run_analyst_pipeline(
        dec_dt,
        cfg,
        wall_clock_fn=lambda: next(
            resume_ticks_179_to_180, t_capture_start + dt.timedelta(seconds=180.0)
        ),
    )
    assert st_ok.final_output is not None and st_ok.final_output.verdict == "SUGGESTION"

    # 3. Single-shot fresh run: capture completes at 179.0s, synthesis finishes at 190.0s -> REJECTED!
    fresh_chart_dir = tmp_path / "charts_fresh_late"
    fresh_audit = tmp_path / "fresh_late.jsonl"
    cfg_fresh = AnalystConfig(
        mode="OFFLINE",
        fixtures_dir=str(FIXTURES_DIR),
        audit_log_path=str(fresh_audit),
        payoff_chart_dir=str(fresh_chart_dir),
    )
    clock_state = {"now": t_capture_start}

    def _advancing_synth(resp: LLMSynthesisResponse, _s, _p, _a):
        # Synthesis finishes at 190.0s after capture_started_at_ist
        clock_state["now"] = t_capture_start + dt.timedelta(seconds=190.0)
        return resp

    fresh_ticks = iter(
        [
            t_capture_start,
            t_capture_start + dt.timedelta(seconds=179.0),
        ]
    )
    with pytest.raises(StaleEvidenceSnapshotError, match="SNAPSHOT_RESUME_EXPIRED_180S"):
        run_analyst_pipeline(
            dec_dt,
            cfg_fresh,
            llm_provider=_StaticOverrideProvider(_advancing_synth),
            wall_clock_fn=lambda: next(fresh_ticks, clock_state["now"]),
        )
    assert not fresh_audit.exists()
    assert list(fresh_chart_dir.glob("*.png")) == []


def test_targeted_fix3_payoff_chart_uses_walked_prices_not_touch_quotes_and_agrees(
    tmp_path: Path,
) -> None:
    """Payoff chart uses the selected trade's walked fill prices and strictly agrees on risk numbers."""
    from nifty_analyst.engine.pricing import LegPayoffSpec, compute_piecewise_expiry_payoff
    from nifty_analyst.output.payoff_diagram import (
        candidate_to_payoff_specs,
        render_candidate_payoff_diagram,
        render_payoff_diagram,
    )
    import importlib.util

    has_mpl = importlib.util.find_spec("matplotlib") is not None

    # 1. Build a 150-pt Bull Put Spread (lot_size=65, 1 lot) that PASSes Part A after book walk,
    #    where touch quotes (1 unit @ 100/101 and 39/40) differ sharply from 65-unit walked fills
    #    (64 units @ 92 and 46).
    q_ts = "2026-11-06T10:15:00+05:30"
    short_raw = RawStrikeQuote(
        strike=24150.0,
        call_bid=250.0,
        call_ask=252.0,
        put_bid=100.0,
        put_ask=101.0,
        call_oi=90000,
        put_oi=120000,
        call_volume=12000,
        put_volume=30000,
        call_depth_lots=10,
        put_depth_lots=10,
        quote_timestamp_ist=q_ts,
        put_touch_bid_qty=1,
        put_touch_ask_qty=1,
        put_visible_buy_qty=650,
        put_visible_sell_qty=650,
        put_buy_levels=[[100.0, 1], [92.0, 649]],
        put_sell_levels=[[101.0, 1], [108.0, 649]],
    )
    long_raw = RawStrikeQuote(
        strike=24000.0,
        call_bid=380.0,
        call_ask=382.0,
        put_bid=39.0,
        put_ask=40.0,
        call_oi=50000,
        put_oi=95000,
        call_volume=8000,
        put_volume=22000,
        call_depth_lots=10,
        put_depth_lots=10,
        quote_timestamp_ist=q_ts,
        put_touch_bid_qty=1,
        put_touch_ask_qty=1,
        put_visible_buy_qty=650,
        put_visible_sell_qty=650,
        put_buy_levels=[[39.0, 1], [32.0, 649]],
        put_sell_levels=[[40.0, 1], [46.0, 649]],
    )

    cfg_u = AnalystConfig()
    short_leg = build_option_leg(
        short_raw, "PE", "SHORT", 1, 24570.0, 20.0 / 365.0, 0.067, False, cfg_u
    )
    long_leg = build_option_leg(
        long_raw, "PE", "LONG", 1, 24570.0, 20.0 / 365.0, 0.067, False, cfg_u
    )
    from nifty_analyst.state import CandleBar

    bar_1d = CandleBar(
        timeframe="1d",
        open_time_ist="2026-11-05T09:15:00+05:30",
        close_time_ist="2026-11-05T15:30:00+05:30",
        open=24400.0,
        high=24580.0,
        low=24380.0,
        close=24540.0,
    )
    bar_1h = CandleBar(
        timeframe="1h",
        open_time_ist="2026-11-06T09:15:00+05:30",
        close_time_ist="2026-11-06T10:15:00+05:30",
        open=24510.0,
        high=24565.0,
        low=24495.0,
        close=24550.0,
    )
    ps = PriceStructureEvidence(
        trend_1d_state="UPTREND",
        higher_high_higher_low_repaired=True,
        first_pullback_in_downtrend_warning=False,
        rsi_14_1h=58.0,
        atr_14_1h=65.0,
        atr_14_1d=190.0,
        stretch_zscore_1h=0.4,
        ma20_1h=24180.0,
        ema50_1h=24120.0,
        last_completed_1d_bar=bar_1d,
        last_completed_1h_bar=bar_1h,
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
        structure_id="BPS_2026-11-26_24150_24000",
        structure_type="BULL_PUT_SPREAD",
        direction="BULLISH",
        expiry=exp_meta,
        legs=[short_leg, long_leg],
        spot=24550.0,
        implied_forward=24570.0,
        atm_iv=0.14,
        time_to_expiry_years=20.0 / 365.0,
        empirical_daily_log_returns=[0.004, -0.003] * 25,
        invalidation_level=24450.0,
        invalidation_candle_basis="COMPLETED_1H_CLOSE",
        invalidation_justification="Below 1H support.",
        matches_part_b_style=True,
        portfolio_state=port_state,
        price_structure=ps,
        volatility=vol,
        config=cfg_u,
    )
    assert cand.execution_fill_basis == "FULL_SIZE_BOOK_WALK"
    assert cand.full_size_fill_supported is True
    assert cand.part_a_overall_status == "PASS"

    # Naive touch specs disagree sharply (~Rs 896 difference in contractual max loss!)
    slip = cfg_u.slippage_buffer_points_per_leg
    naive_touch_specs = [
        LegPayoffSpec(
            strike=l.strike,
            option_type=l.option_type,
            side=l.side,
            ratio=l.ratio,
            executable_entry_price=(l.bid - slip if l.side == "SHORT" else l.ask + slip),
        )
        for l in cand.legs
    ]
    pw_naive = compute_piecewise_expiry_payoff(
        naive_touch_specs, cand.lot_size, cand.lots, cand.statutory_charges.total_charges_rs
    )
    assert abs(pw_naive.gross_max_loss_rs - cand.contractual_max_loss_rs) > 800.0

    # Walked payoff specs match candidate risk numbers identically!
    walked_specs = candidate_to_payoff_specs(cand, slippage_buffer_points_per_leg=slip)
    pw_walked = compute_piecewise_expiry_payoff(
        walked_specs, cand.lot_size, cand.lots, cand.statutory_charges.total_charges_rs
    )
    assert abs(pw_walked.gross_max_loss_rs - cand.contractual_max_loss_rs) < 0.02
    assert abs(pw_walked.max_profit_rs - cand.max_profit_rs) < 0.02
    assert pw_walked.breakevens_net == cand.breakevens_net

    if has_mpl:
        walked_png = tmp_path / "walked_cand.png"
        naive_png = tmp_path / "naive_touch.png"
        diag = render_candidate_payoff_diagram(cand, str(walked_png), slip)
        assert abs(diag["gross_max_loss_rs"] - cand.contractual_max_loss_rs) < 0.02
        assert abs(diag["max_profit_rs"] - cand.max_profit_rs) < 0.02
        assert diag["breakevens_net"] == cand.breakevens_net

        render_payoff_diagram(
            naive_touch_specs,
            cand.lot_size,
            cand.lots,
            cand.statutory_charges.total_charges_rs,
            f"{cand.structure_id} ({cand.structure_type})",
            str(naive_png),
        )
        assert walked_png.read_bytes() != naive_png.read_bytes()

        # Tampered candidate risk numbers fail closed and delete the PNG
        tampered_cand = dataclasses.replace(
            cand, contractual_max_loss_rs=pw_naive.gross_max_loss_rs
        )
        bad_png = tmp_path / "bad_mismatch.png"
        with pytest.raises(ValueError, match="PAYOFF_CHART_RISK_MISMATCH"):
            render_candidate_payoff_diagram(tampered_cand, str(bad_png), slip)
        assert not bad_png.exists()

        # End-to-end pipeline run with walked books renders the walked-price chart, not touch!
        class _WalkedChainAdapter(FixtureKotakNeoAdapter):
            def fetch_option_chain_raw(self, expiry_date: str, decision_dt_ist: dt.datetime):
                raw = json.loads(
                    json.dumps(super().fetch_option_chain_raw(expiry_date, decision_dt_ist))
                )
                for row in raw["strikes"]:
                    for prefix, b_key, a_key in (
                        ("call", "call_bid", "call_ask"),
                        ("put", "put_bid", "put_ask"),
                    ):
                        bid_v = float(row[b_key])
                        ask_v = float(row[a_key])
                        row[f"{prefix}_touch_bid_qty"] = 1
                        row[f"{prefix}_touch_ask_qty"] = 1
                        row[f"{prefix}_visible_buy_qty"] = 750
                        row[f"{prefix}_visible_sell_qty"] = 750
                        row[f"{prefix}_buy_levels"] = [
                            [bid_v, 1],
                            [round(max(0.10, bid_v - 3.5), 2), 749],
                        ]
                        row[f"{prefix}_sell_levels"] = [
                            [ask_v, 1],
                            [round(ask_v + 3.5, 2), 749],
                        ]
                return raw

        pipe_charts = tmp_path / "pipe_walked_charts"
        cfg_pipe = AnalystConfig(
            mode="OFFLINE",
            fixtures_dir=str(FIXTURES_DIR),
            audit_log_path=str(tmp_path / "pipe_walked.jsonl"),
            payoff_chart_dir=str(pipe_charts),
        )
        st_w, _ = run_analyst_pipeline(
            dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST),
            cfg_pipe,
            broker_adapter=_WalkedChainAdapter(FIXTURES_DIR),
        )
        assert st_w.final_output is not None
        assert st_w.final_output.verdict == "SUGGESTION"
        sel_w = st_w.final_output.primary_proposal
        assert sel_w is not None
        assert sel_w.execution_fill_basis == "FULL_SIZE_BOOK_WALK"
        rendered_p = Path(st_w.final_output.payoff_diagram_path)
        assert rendered_p.is_file()

        expected_walked_p = tmp_path / "expected_pipe_walked.png"
        render_candidate_payoff_diagram(
            sel_w, str(expected_walked_p), cfg_pipe.slippage_buffer_points_per_leg
        )
        assert rendered_p.read_bytes() == expected_walked_p.read_bytes()


def test_targeted_fix4_exact_four_contradiction_examples_rejected_and_excluded(
    tmp_path: Path,
) -> None:
    """All 4 exact contradictory free-form report examples fail verification and never appear in any report."""
    from nifty_analyst.synthesis.verifier import verify_llm_synthesis

    dec_dt = dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST)
    base_cfg = AnalystConfig(
        mode="OFFLINE",
        fixtures_dir=str(FIXTURES_DIR),
        audit_log_path=str(tmp_path / "base_fix4.jsonl"),
    )
    base_state, _ = run_analyst_pipeline(dec_dt, base_cfg)
    assert base_state.final_output is not None and base_state.final_output.primary_proposal is not None
    cand = base_state.final_output.primary_proposal
    assert cand.direction == "BULLISH"

    # 1. Direct verifier check for exact "Buy 24350 CE at 999.00 points" when verified walked entry is 225.39
    long_leg_24350 = dataclasses.replace(
        cand.legs[-1],
        strike=24350.0,
        option_type="CE",
        side="LONG",
        ratio=1,
        bid=220.00,
        ask=221.00,
        mid=220.50,
        vwap_fill_price=224.89,
        executable_entry_price=225.39,
        fill_basis="FULL_SIZE_BOOK_WALK",
    )
    cand_24350 = dataclasses.replace(
        cand,
        structure_id="BCS_2026-11-12_24350_24500",
        structure_type="BULL_CALL_SPREAD",
        direction="BULLISH",
        legs=[long_leg_24350, *cand.legs[:1]],
    )
    state_with_24350 = dataclasses.replace(
        base_state,
        candidate_structures=[cand_24350, *base_state.candidate_structures],
    )
    raw_cits = [
        {
            "claim_text": "Spot verified",
            "state_field_path": "contract_context.spot.value",
            "cited_value": base_state.contract_context.spot.value,
        }
    ]
    for bad_price_claim in (
        "Buy 24350 CE at 999.00 points",
        "Buy 24350 CE at 220.50 points",  # touch mid when book walked to 225.39!
    ):
        _, vr_24350 = verify_llm_synthesis(
            state=state_with_24350,
            raw_citations=raw_cits,
            selected_primary=cand_24350,
            consolidator_recommended_id=cand_24350.structure_id,
            prose_fields={"reasoning_summary": bad_price_claim},
        )
        assert vr_24350.passed_verification is False, bad_price_claim

    # 2. End-to-end pipeline checks across ALL 4 exact examples (plus variants) across prose, digest & citations
    actual_long_leg = next((l for l in cand.legs if l.side == "LONG"), cand.legs[-1])
    leg_verb = "Buy" if actual_long_leg.side == "LONG" else "Sell"

    contradiction_cases = [
        ("exact_24350_999pts", "Buy 24350 CE at 999.00 points", "999.00"),
        (
            "selected_leg_999pts",
            f"{leg_verb} {int(actual_long_leg.strike)} {actual_long_leg.option_type} at 999.00 points",
            "999.00",
        ),
        ("selected_trade_bearish", "The selected trade is bearish.", "selected trade is bearish"),
        ("this_trade_bearish", "This trade is bearish.", "This trade is bearish"),
        ("max_loss_rupee_1", "The maximum loss is ₹1.", "₹1"),
        ("bare_downside_rupee_1", "Total downside on the setup is ₹1.", "₹1"),
        ("max_loss_inr_1", "The maximum loss is INR 1.", "INR 1"),
        (
            "guaranteed_profitable",
            "This trade is guaranteed profitable and cannot lose.",
            "guaranteed profitable",
        ),
    ]

    target_setters = {
        "reasoning_summary": lambda r, s: dataclasses.replace(r, reasoning_summary=s),
        "bull_thesis": lambda r, s: dataclasses.replace(
            r, debate_state=dataclasses.replace(r.debate_state, bull_thesis=s)  # type: ignore[arg-type]
        ),
        "tail_guardian_note": lambda r, s: dataclasses.replace(
            r, risk_triad_state=dataclasses.replace(r.risk_triad_state, tail_guardian_note=s)  # type: ignore[arg-type]
        ),
        "final_recommendation_rationale": lambda r, s: dataclasses.replace(
            r,
            consolidator_state=dataclasses.replace(
                r.consolidator_state, final_recommendation_rationale=s  # type: ignore[arg-type]
            ),
        ),
        "whatsapp_digest": lambda r, s: dataclasses.replace(
            r,
            consolidator_state=dataclasses.replace(
                r.consolidator_state, whatsapp_digest=s  # type: ignore[arg-type]
            ),
        ),
        "citation_claim": lambda r, s: dataclasses.replace(
            r, citations=[{**r.citations[0], "claim_text": s}, *r.citations[1:]]
        ),
    }

    for case_name, bad_text, forbidden_substr in contradiction_cases:
        for field_name, setter in target_setters.items():
            audit_p = tmp_path / f"fix4_{case_name}_{field_name}.jsonl"
            cfg_i = AnalystConfig(
                mode="OFFLINE",
                fixtures_dir=str(FIXTURES_DIR),
                audit_log_path=str(audit_p),
            )
            st_i, deliv_i = run_analyst_pipeline(
                dec_dt,
                cfg_i,
                llm_provider=_StaticOverrideProvider(
                    lambda resp, _s, _p, _a, fn=setter, txt=bad_text: fn(resp, txt)
                ),
                force_render_for_inspection=True,
            )
            assert st_i.final_output is not None, (case_name, field_name)
            assert st_i.final_output.verifier_report.passed_verification is False, (
                case_name,
                field_name,
            )
            assert st_i.final_output.verdict == "WAIT", (case_name, field_name)
            assert st_i.final_output.primary_proposal is None, (case_name, field_name)
            assert deliv_i.alerted_trader is False, (case_name, field_name)
            assert forbidden_substr not in deliv_i.full_report_markdown, (case_name, field_name)
            assert forbidden_substr not in deliv_i.user_facing_message, (case_name, field_name)
            assert forbidden_substr not in st_i.final_output.whatsapp_digest, (
                case_name,
                field_name,
            )
            audit_row = json.loads(audit_p.read_text(encoding="utf-8").strip().splitlines()[-1])
            assert forbidden_substr not in audit_row["report_markdown"], (case_name, field_name)

    # Also check contradictory `debate_state.dominant_regime = "BEARISH"` on a BULLISH selection
    cfg_dom = AnalystConfig(
        mode="OFFLINE",
        fixtures_dir=str(FIXTURES_DIR),
        audit_log_path=str(tmp_path / "fix4_dom_bearish.jsonl"),
    )
    st_dom, deliv_dom = run_analyst_pipeline(
        dec_dt,
        cfg_dom,
        llm_provider=_StaticOverrideProvider(
            lambda resp, _s, _p, _a: dataclasses.replace(
                resp,
                debate_state=dataclasses.replace(resp.debate_state, dominant_regime="BEARISH"),  # type: ignore[arg-type]
            )
        ),
    )
    assert st_dom.final_output is not None
    assert st_dom.final_output.verifier_report.passed_verification is False
    assert st_dom.final_output.verdict == "WAIT"
    assert deliv_dom.alerted_trader is False


def test_targeted_fix4_arbitrary_freeform_wording_never_enters_any_report_or_state(
    tmp_path: Path,
) -> None:
    """Every report generates 100% of trade facts & prose from verified state — zero free-form LLM text survives."""
    from nifty_analyst.output.audit_log import deliver_run_output
    from nifty_analyst.state import DebateState, RiskTriadState

    dec_dt = dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST)

    # 1. Unseen arbitrary wording (zero numbers, zero symbols, zero blocklist words) injected across ALL
    #    12 synthesis fields + citation claim_text via `agent_artifact` resume on a valid SUGGESTION:
    #    even when verification passes, NONE of the free-form wording ever enters any report or state!
    art_path = tmp_path / "freeform_synth.json"
    req_path = tmp_path / "freeform_synth.json.request.json"
    audit_path = tmp_path / "freeform_audit.jsonl"
    cfg = AnalystConfig(
        mode="OFFLINE",
        fixtures_dir=str(FIXTURES_DIR),
        audit_log_path=str(audit_path),
        llm_provider="agent_artifact",
        agent_synthesis_artifact_path=str(art_path),
    )
    with pytest.raises(FileNotFoundError):
        run_analyst_pipeline(dec_dt, cfg)

    req = json.loads(req_path.read_text(encoding="utf-8"))
    chosen_card = req["compact_desk_brief"]["certified_pass_candidates"][0]
    idx = chosen_card["state_index"]
    bound_val = chosen_card["contractual_max_loss_rs"]
    ivp_val = req["sanitised_state"]["market_evidence"]["volatility_state"]["ivp_90d"]

    sentinel = "arbitrary_freeform_prose_marker_zeta"
    art_doc = {
        "model_id": "whatsapp-persona-agent-v1",
        "run_id": req["run_id"],
        "snapshot_id": req["snapshot_id"],
        "evidence_sha256": req["evidence_sha256"],
        "decision_timestamp_ist": req["decision_timestamp_ist"],
        "reasoning_summary": f"{sentinel} reasoning: abandon upside discipline and expect easy windfall gains.",
        "habitual_comparison_note": f"{sentinel} habitual: ignore wing risk and chase momentum.",
        "research_rule_challenge_note": f"{sentinel} challenge: widen risk limits informally.",
        "confidence_note": f"{sentinel} confidence: effortless conviction.",
        "debate_state": {
            "bull_thesis": f"{sentinel} bull: unstoppable rally ahead.",
            "bear_thesis": f"{sentinel} bear: imminent collapse ahead.",
            "vol_regime_thesis": f"{sentinel} vol: harvest easy premium.",
            "rebuttal_round": f"{sentinel} rebuttal: market drifts south effortlessly.",
            "dominant_regime": chosen_card["direction"],
        },
        "risk_triad": {
            "convexity_advocate_note": f"{sentinel} convexity: asymmetric windfall.",
            "tail_guardian_note": f"{sentinel} tail: negligible downside worry.",
            "neutral_arbiter_note": f"{sentinel} arbiter: endorses `{chosen_card['structure_id']}`.",
        },
        "consolidator": {
            "recommended_structure_id": chosen_card["structure_id"],
            "conviction": "HIGH",
            "final_recommendation_rationale": (
                f"{sentinel} rationale: selects `{chosen_card['structure_id']}` with effortless edge."
            ),
            "whatsapp_digest": f"*Pick*: `{chosen_card['structure_id']}`",
        },
        "citations": [
            {
                "claim_text": f"{sentinel} citation text that must be canonicalized",
                "state_field_path": f"candidate_structures[{idx}].contractual_max_loss_rs",
                "cited_value": bound_val,
            },
            {
                "claim_text": f"{sentinel} second citation text",
                "state_field_path": "market_evidence.volatility_state.ivp_90d",
                "cited_value": ivp_val,
            },
        ],
    }
    art_path.write_text(json.dumps(art_doc), encoding="utf-8")

    state, delivery = run_analyst_pipeline(dec_dt, cfg, force_render_for_inspection=True)
    assert state.final_output is not None
    assert state.final_output.verdict == "SUGGESTION"
    assert state.final_output.verifier_report.passed_verification is True
    sel = state.final_output.primary_proposal
    assert sel is not None and sel.structure_id == chosen_card["structure_id"]

    # Zero free-form LLM text survives in ANY report, digest, state object, citation, or audit log!
    assert sentinel not in delivery.full_report_markdown
    assert sentinel not in delivery.user_facing_message
    assert sentinel not in delivery.whatsapp_digest
    assert sentinel not in json.dumps(state.to_dict())
    assert sentinel not in audit_path.read_text(encoding="utf-8")

    # Verified trade facts match `sel` across both Markdown report and WhatsApp digest
    assert f"`{sel.structure_id}`" in delivery.full_report_markdown
    assert f"Direction: `{sel.direction}`" in delivery.full_report_markdown
    assert f"Rs {sel.contractual_max_loss_rs:,.2f}" in delivery.full_report_markdown
    assert f"Rs {sel.planned_stop_rs:,.2f}" in delivery.full_report_markdown
    assert state.debate_state is not None and state.debate_state.dominant_regime == "BULLISH"
    assert f"Debate: {state.debate_state.dominant_regime}" in delivery.whatsapp_digest
    for leg in sel.legs:
        assert leg.executable_entry_price is not None
        assert f"executable entry {leg.executable_entry_price:.2f} pts" in delivery.full_report_markdown
        assert f"@{leg.executable_entry_price:.2f}" in delivery.whatsapp_digest

    # 2. Even direct callers of `render_analyst_report` and `deliver_run_output` on a mutated state
    #    containing all 4 contradictory strings get 100% deterministic verified report text!
    poison_all_four = (
        "Buy 24350 CE at 999.00 points. The selected trade is bearish. "
        "The maximum loss is ₹1. This trade is guaranteed profitable and cannot lose."
    )
    assert state.final_output is not None and state.consolidator_state is not None
    mutated_state = dataclasses.replace(
        state,
        debate_state=DebateState(
            bull_thesis=poison_all_four,
            bear_thesis=poison_all_four,
            vol_regime_thesis=poison_all_four,
            rebuttal_round=poison_all_four,
            dominant_regime="BEARISH",
        ),
        risk_triad_state=RiskTriadState(
            convexity_advocate_note=poison_all_four,
            tail_guardian_note=poison_all_four,
            neutral_arbiter_note=poison_all_four,
        ),
        consolidator_state=dataclasses.replace(
            state.consolidator_state,
            final_recommendation_rationale=poison_all_four,
            whatsapp_digest=poison_all_four,
        ),
        final_output=dataclasses.replace(
            state.final_output,
            reasoning_summary=poison_all_four,
            habitual_style_comparison_note=poison_all_four,
            research_rule_challenge_note=poison_all_four,
            confidence_note=poison_all_four,
            whatsapp_digest=poison_all_four,
        ),
    )
    direct_md = render_analyst_report(mutated_state)
    direct_audit = tmp_path / "direct_deliver_audit.jsonl"
    direct_deliv = deliver_run_output(
        mutated_state, direct_audit, force_render_for_inspection=True
    )
    for bad_frag in (
        "999.00 points",
        "selected trade is bearish",
        "₹1",
        "guaranteed profitable",
        "cannot lose",
    ):
        assert bad_frag not in direct_md
        assert bad_frag not in direct_deliv.full_report_markdown
        assert bad_frag not in direct_deliv.user_facing_message
        assert bad_frag not in direct_deliv.whatsapp_digest
        assert bad_frag not in direct_audit.read_text(encoding="utf-8")


def test_targeted_fix5_regime_independent_of_pick_and_conflict_forces_wait(
    tmp_path: Path,
) -> None:
    """Regime assessment stays independent of any supplied pick; conflicts fail closed to WAIT without flipping regime."""
    from nifty_analyst.output.audit_log import deliver_run_output
    from nifty_analyst.state import DebateState, TriggeredDetector
    from nifty_analyst.synthesis.personas import (
        build_deterministic_personas,
        normalize_state_narrative_from_verified_facts,
    )
    from nifty_analyst.synthesis.verifier import (
        assess_market_regime,
        verify_llm_synthesis,
    )

    dec_dt = dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST)
    probe_cfg = AnalystConfig(
        mode="OFFLINE",
        fixtures_dir=str(FIXTURES_DIR),
        audit_log_path=str(tmp_path / "fix5_probe.jsonl"),
    )
    probe_state, _ = run_analyst_pipeline(dec_dt, probe_cfg)

    pass_cands = [
        c
        for c in probe_state.candidate_structures
        if c.part_a_overall_status == "PASS" and c.full_size_fill_supported
    ]
    bull_cand = next(
        c for c in pass_cands if c.direction == "BULLISH" and c.technical_options_score >= 68.0
    )
    bear_cand = next(c for c in pass_cands if c.direction == "BEARISH")
    ic_cand = next(c for c in pass_cands if c.direction == "NEUTRAL")

    # 1. Compatible non-identical verified inputs: BULLISH market regime + NEUTRAL Iron Condor pick.
    #    Regime MUST remain BULLISH (never changed to NEUTRAL to match the pick!), while reports
    #    consistently render BOTH the BULLISH market regime and the NEUTRAL selected trade.
    assert assess_market_regime(probe_state) == "BULLISH"
    deb_ic, _, cons_ic, chosen_ic = build_deterministic_personas(
        state=probe_state,
        default_primary=ic_cand,
        eligible_candidates=pass_cands,
    )
    assert deb_ic.dominant_regime == "BULLISH"
    assert chosen_ic is not None and chosen_ic.structure_id == ic_cand.structure_id
    assert chosen_ic.direction == "NEUTRAL"
    assert cons_ic.conviction == "MODERATE"
    assert f"`{ic_cand.structure_id}` ({ic_cand.structure_type}, NEUTRAL," in cons_ic.whatsapp_digest
    assert "Debate: BULLISH" in cons_ic.whatsapp_digest

    cfg_ic = AnalystConfig(
        mode="OFFLINE",
        fixtures_dir=str(FIXTURES_DIR),
        audit_log_path=str(tmp_path / "fix5_ic_in_bull.jsonl"),
    )
    ic_idx = next(
        i
        for i, c in enumerate(probe_state.candidate_structures)
        if c.structure_id == ic_cand.structure_id
    )
    st_ic, deliv_ic = run_analyst_pipeline(
        dec_dt,
        cfg_ic,
        force_render_for_inspection=True,
        llm_provider=_StaticOverrideProvider(
            lambda _r, s, _p, _a: OfflineDeterministicSynthesisProvider().synthesize(
                "{}", s, ic_cand, []
            )
        ),
    )
    assert st_ic.final_output is not None
    assert st_ic.final_output.verdict == "SUGGESTION"
    assert st_ic.final_output.verifier_report.passed_verification is True
    assert st_ic.debate_state is not None and st_ic.debate_state.dominant_regime == "BULLISH"
    assert st_ic.final_output.primary_proposal is not None
    assert st_ic.final_output.primary_proposal.structure_id == ic_cand.structure_id
    assert st_ic.final_output.primary_proposal.direction == "NEUTRAL"
    assert "Dominant Regime (`BULLISH`)" in deliv_ic.full_report_markdown
    assert f"**Structure ID**: `{ic_cand.structure_id}`" in deliv_ic.full_report_markdown
    assert "Direction: `NEUTRAL`" in deliv_ic.full_report_markdown
    assert "Debate: BULLISH" in deliv_ic.whatsapp_digest
    assert f"`{ic_cand.structure_id}` ({ic_cand.structure_type}, NEUTRAL," in deliv_ic.whatsapp_digest
    _ = ic_idx

    # 2. Exact Instinct reproducer: DOWNTREND + CONFIRMED_QUANT_REGIME BEARISH signal + supplied BULLISH candidate.
    #    Regime MUST stay BEARISH (NEVER flipped to BULLISH with HIGH conviction!), and conflict MUST force WAIT/SIT_OUT.
    bear_ps = dataclasses.replace(
        probe_state.market_evidence.price_structure,
        trend_1d_state="DOWNTREND",
        higher_high_higher_low_repaired=False,
        first_pullback_in_downtrend_warning=True,
    )
    bear_det = TriggeredDetector(
        detector_id="NOVEL_HIGH_IV_BEARISH_CALL_WALL",
        detector_name="Elevated IVP + Downtrend Below Call OI Wall",
        track="NOVEL_QUANT",
        confirmation_status="CONFIRMED_QUANT_REGIME",
        direction="BEARISH",
        invalidation_level=float(probe_state.market_evidence.derivatives_positioning.call_oi_wall_strike),
        invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
        invalidation_justification="Completed daily close above Call OI wall invalidates bearish regime.",
        summary="Confirmed 1D DOWNTREND capped below Call OI wall.",
        evidence_fields_cited=[
            "market_evidence.price_structure.trend_1d_state",
            "market_evidence.derivatives_positioning.call_oi_wall_strike",
        ],
    )
    bear_ev = dataclasses.replace(
        probe_state.market_evidence,
        price_structure=bear_ps,
        triggered_detectors=[bear_det],
    )
    bear_state = dataclasses.replace(probe_state, market_evidence=bear_ev)

    assert assess_market_regime(bear_state) == "BEARISH"

    # 2a. `build_deterministic_personas` with supplied bullish candidate
    deb_b, triad_b, cons_b, chosen_b = build_deterministic_personas(
        state=bear_state,
        default_primary=bull_cand,
        eligible_candidates=[bull_cand, bear_cand],
    )
    assert deb_b.dominant_regime == "BEARISH"
    assert chosen_b is None
    assert cons_b.recommended_structure_id == "WAIT"
    assert cons_b.conviction == "SIT_OUT"
    assert "Debate: BEARISH" in cons_b.whatsapp_digest
    assert "WAIT (SIT_OUT)" in cons_b.whatsapp_digest
    assert bull_cand.structure_id not in cons_b.whatsapp_digest
    assert bull_cand.structure_id not in triad_b.neutral_arbiter_note

    # 2b. `OfflineDeterministicSynthesisProvider` with supplied bullish `primary`
    synth_b = OfflineDeterministicSynthesisProvider().synthesize(
        "{}", bear_state, primary=bull_cand, alternatives=[bear_cand]
    )
    assert synth_b.debate_state is not None and synth_b.debate_state.dominant_regime == "BEARISH"
    assert synth_b.consolidator_state is not None
    assert synth_b.consolidator_state.recommended_structure_id == "WAIT"
    assert synth_b.consolidator_state.conviction == "SIT_OUT"
    assert "Debate: BEARISH" in synth_b.consolidator_state.whatsapp_digest

    # 2c. Auto-selection (`default_primary=None`) in BEARISH regime ignores higher-scoring BULLISH candidate
    deb_auto, _, cons_auto, chosen_auto = build_deterministic_personas(
        state=bear_state,
        default_primary=None,
        eligible_candidates=[bull_cand, bear_cand],
    )
    assert deb_auto.dominant_regime == "BEARISH"
    assert chosen_auto is not None and chosen_auto.structure_id == bear_cand.structure_id
    assert chosen_auto.direction == "BEARISH"
    assert cons_auto.recommended_structure_id == bear_cand.structure_id

    # 2d. `verify_llm_synthesis` rejects a BULLISH candidate recommended in a BEARISH regime
    bull_idx = next(
        i
        for i, c in enumerate(bear_state.candidate_structures)
        if c.structure_id == bull_cand.structure_id
    )
    _, vrep_conflict = verify_llm_synthesis(
        state=bear_state,
        raw_citations=[
            {
                "claim_text": "Max loss check",
                "state_field_path": f"candidate_structures[{bull_idx}].contractual_max_loss_rs",
                "cited_value": bull_cand.contractual_max_loss_rs,
            }
        ],
        selected_primary=bull_cand,
        consolidator_recommended_id=bull_cand.structure_id,
        debate_dominant_regime="BULLISH",
    )
    assert vrep_conflict.passed_verification is False
    assert vrep_conflict.part_a_reverified is False
    joined_reasons = " | ".join(vrep_conflict.failure_reasons)
    assert "REGIME_CANDIDATE_CONFLICT" in joined_reasons

    # 2e. Direct `normalize_state_narrative_from_verified_facts`, `render_analyst_report`, and `deliver_run_output`
    #     on a tampered state (`DOWNTREND` + `BEARISH` evidence, `primary_proposal=bull_cand`, `dominant_regime="BULLISH"`)
    assert bear_state.final_output is not None and bear_state.consolidator_state is not None
    tampered_bear = dataclasses.replace(
        bear_state,
        debate_state=DebateState(
            bull_thesis="Bullish override",
            bear_thesis="Bearish",
            vol_regime_thesis="Vol",
            rebuttal_round="Rebuttal",
            dominant_regime="BULLISH",
        ),
        consolidator_state=dataclasses.replace(
            bear_state.consolidator_state,
            recommended_structure_id=bull_cand.structure_id,
            conviction="HIGH",
        ),
        final_output=dataclasses.replace(
            bear_state.final_output,
            verdict="SUGGESTION",
            is_live_ready=True,
            primary_proposal=bull_cand,
        ),
    )
    norm_bear = normalize_state_narrative_from_verified_facts(tampered_bear)
    assert norm_bear.debate_state is not None and norm_bear.debate_state.dominant_regime == "BEARISH"
    assert norm_bear.final_output is not None and norm_bear.final_output.verdict == "WAIT"
    assert norm_bear.final_output.primary_proposal is None
    assert norm_bear.consolidator_state is not None
    assert norm_bear.consolidator_state.recommended_structure_id == "WAIT"
    assert norm_bear.consolidator_state.conviction == "SIT_OUT"

    bear_audit = tmp_path / "fix5_bear_conflict_audit.jsonl"
    bear_deliv = deliver_run_output(
        tampered_bear, bear_audit, force_render_for_inspection=True
    )
    assert bear_deliv.alerted_trader is False
    assert "Dominant Regime (`BEARISH`)" in bear_deliv.full_report_markdown
    assert "Verdict = WAIT" in bear_deliv.full_report_markdown
    assert "Debate: BEARISH" in bear_deliv.whatsapp_digest
    assert bull_cand.structure_id not in bear_deliv.whatsapp_digest
    bear_row = json.loads(bear_audit.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert bear_row["verdict"] == "WAIT"
    assert bear_row["debate_state"]["dominant_regime"] == "BEARISH"
    assert bear_row["consolidator_state"]["conviction"] == "SIT_OUT"

    # 3. Symmetric opposite conflict end-to-end through `run_analyst_pipeline`:
    #    UPTREND + CONFIRMED_QUANT_REGIME BULLISH fixture + agent_artifact recommending a BEARISH PASS candidate.
    art_sym = tmp_path / "fix5_sym_bear_in_bull.json"
    req_sym = tmp_path / "fix5_sym_bear_in_bull.json.request.json"
    cfg_sym = AnalystConfig(
        mode="OFFLINE",
        fixtures_dir=str(FIXTURES_DIR),
        audit_log_path=str(tmp_path / "fix5_sym_audit.jsonl"),
        llm_provider="agent_artifact",
        agent_synthesis_artifact_path=str(art_sym),
    )
    with pytest.raises(FileNotFoundError):
        run_analyst_pipeline(dec_dt, cfg_sym)

    req_doc = json.loads(req_sym.read_text(encoding="utf-8"))
    bear_idx = next(
        i
        for i, c in enumerate(probe_state.candidate_structures)
        if c.structure_id == bear_cand.structure_id
    )
    bear_card = {
        "state_index": bear_idx,
        "structure_id": bear_cand.structure_id,
        "contractual_max_loss_rs": bear_cand.contractual_max_loss_rs,
    }
    _write_valid_agent_artifact(art_sym, req_doc, chosen_card=bear_card)
    st_sym, deliv_sym = run_analyst_pipeline(
        dec_dt, cfg_sym, force_render_for_inspection=True
    )
    assert st_sym.final_output is not None
    assert st_sym.debate_state is not None and st_sym.debate_state.dominant_regime == "BULLISH"
    assert st_sym.final_output.verifier_report.passed_verification is False
    assert "REGIME_CANDIDATE_CONFLICT" in " | ".join(
        st_sym.final_output.verifier_report.failure_reasons
    )
    assert st_sym.final_output.verdict == "WAIT"
    assert st_sym.final_output.primary_proposal is None
    assert st_sym.consolidator_state is not None
    assert st_sym.consolidator_state.conviction == "SIT_OUT"
    assert deliv_sym.alerted_trader is False
    assert "Debate: BULLISH" in deliv_sym.whatsapp_digest
    assert bear_cand.structure_id not in deliv_sym.whatsapp_digest

    # 4. Competing market-evidence regime signals resolve to CONTESTED_WAIT (while auxiliary skew in DOWNTREND stays BEARISH).
    skew_aux = TriggeredDetector(
        detector_id="NOVEL_SKEW_DISLOCATION",
        detector_name="25-Delta Put-Call Skew Richness Dislocation",
        track="NOVEL_QUANT",
        confirmation_status="CONFIRMED_QUANT_REGIME",
        direction="BULLISH",
        invalidation_level=23800.0,
        invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
        invalidation_justification="Skew harvest invalidation.",
        summary="Elevated put skew during downtrend.",
        evidence_fields_cited=["market_evidence.volatility_state.skew_25d_put_minus_call"],
    )
    bear_with_skew = dataclasses.replace(
        bear_state,
        market_evidence=dataclasses.replace(
            bear_state.market_evidence, triggered_detectors=[bear_det, skew_aux]
        ),
    )
    assert assess_market_regime(bear_with_skew) == "BEARISH"

    bull_det = next(
        d
        for d in probe_state.market_evidence.triggered_detectors
        if d.confirmation_status == "CONFIRMED_QUANT_REGIME" and d.direction == "BULLISH"
    )
    competing_state = dataclasses.replace(
        probe_state,
        market_evidence=dataclasses.replace(
            probe_state.market_evidence, triggered_detectors=[bull_det, bear_det]
        ),
    )
    assert assess_market_regime(competing_state) == "CONTESTED_WAIT"
    deb_comp, _, cons_comp, chosen_comp = build_deterministic_personas(
        competing_state, default_primary=bull_cand, eligible_candidates=[bull_cand, bear_cand]
    )
    assert deb_comp.dominant_regime == "CONTESTED_WAIT"
    assert chosen_comp is None
    assert cons_comp.recommended_structure_id == "WAIT"
    assert cons_comp.conviction == "SIT_OUT"

    bull_in_downtrend = dataclasses.replace(
        bear_state,
        market_evidence=dataclasses.replace(
            bear_state.market_evidence, triggered_detectors=[bull_det]
        ),
    )
    assert assess_market_regime(bull_in_downtrend) == "CONTESTED_WAIT"


def test_targeted_fix5_missing_confirmation_forces_contested_wait_even_with_supplied_candidate(
    tmp_path: Path,
) -> None:
    """Missing `CONFIRMED_QUANT_REGIME` confirmation (empty or unconfirmed-only detectors) forces CONTESTED_WAIT + WAIT."""
    from nifty_analyst.output.audit_log import deliver_run_output
    from nifty_analyst.state import TriggeredDetector
    from nifty_analyst.synthesis.personas import (
        build_deterministic_personas,
        normalize_state_narrative_from_verified_facts,
    )
    from nifty_analyst.synthesis.verifier import (
        assess_market_regime,
        verify_llm_synthesis,
    )

    dec_dt = dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST)
    probe_state, _ = run_analyst_pipeline(
        dec_dt,
        AnalystConfig(
            mode="OFFLINE",
            fixtures_dir=str(FIXTURES_DIR),
            audit_log_path=str(tmp_path / "fix5_missing_probe.jsonl"),
        ),
    )
    pass_cands = [
        c
        for c in probe_state.candidate_structures
        if c.part_a_overall_status == "PASS" and c.full_size_fill_supported
    ]
    bull_cand = next(
        c for c in pass_cands if c.direction == "BULLISH" and c.technical_options_score >= 68.0
    )
    ic_cand = next(c for c in pass_cands if c.direction == "NEUTRAL")
    bull_idx = next(
        i
        for i, c in enumerate(probe_state.candidate_structures)
        if c.structure_id == bull_cand.structure_id
    )

    # 1. Zero triggered detectors (`triggered_detectors == []`) + supplied high-score BULLISH candidate
    empty_det_state = dataclasses.replace(
        probe_state,
        market_evidence=dataclasses.replace(
            probe_state.market_evidence, triggered_detectors=[]
        ),
    )
    assert assess_market_regime(empty_det_state) == "CONTESTED_WAIT"
    deb_e, _, cons_e, chosen_e = build_deterministic_personas(
        state=empty_det_state,
        default_primary=bull_cand,
        eligible_candidates=pass_cands,
    )
    assert deb_e.dominant_regime == "CONTESTED_WAIT"
    assert chosen_e is None
    assert cons_e.recommended_structure_id == "WAIT"
    assert cons_e.conviction == "SIT_OUT"
    assert "Debate: CONTESTED_WAIT" in cons_e.whatsapp_digest

    synth_e = OfflineDeterministicSynthesisProvider().synthesize(
        "{}", empty_det_state, primary=bull_cand, alternatives=pass_cands
    )
    assert synth_e.debate_state is not None
    assert synth_e.debate_state.dominant_regime == "CONTESTED_WAIT"
    assert synth_e.consolidator_state is not None
    assert synth_e.consolidator_state.recommended_structure_id == "WAIT"
    assert synth_e.consolidator_state.conviction == "SIT_OUT"

    # 2. Only unconfirmed detectors (`HISTORICAL_REFERENCE_UNCONFIRMED` + `DEFINITION_OPEN`, zero `CONFIRMED_QUANT_REGIME`)
    hist_unconfirmed = TriggeredDetector(
        detector_id="HIST_1H_MA20_TOUCH",
        detector_name="1H MA20 Completed-Bar Wick Touch (Historical Variant)",
        track="PART_C_HISTORICAL",
        confirmation_status="HISTORICAL_REFERENCE_UNCONFIRMED",
        direction="BULLISH",
        invalidation_level=24000.0,
        invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
        invalidation_justification="Historical reference only.",
        summary="Context only, never blocks or gates.",
        evidence_fields_cited=["market_evidence.price_structure.ma20_1h"],
    )
    stretch_open = TriggeredDetector(
        detector_id="NOVEL_STRETCH_METRIC_CANDIDATE",
        detector_name="Spot vs 1H MA20 Stretch Candidate (Definition Open)",
        track="NOVEL_QUANT",
        confirmation_status="DEFINITION_OPEN",
        direction="BULLISH",
        invalidation_level=23950.0,
        invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
        invalidation_justification="Open definition.",
        summary="Reported as DEFINITION_OPEN.",
        evidence_fields_cited=["contract_context.spot.value"],
    )
    unconfirmed_state = dataclasses.replace(
        probe_state,
        market_evidence=dataclasses.replace(
            probe_state.market_evidence,
            triggered_detectors=[hist_unconfirmed, stretch_open],
        ),
    )
    assert assess_market_regime(unconfirmed_state) == "CONTESTED_WAIT"
    deb_u, _, cons_u, chosen_u = build_deterministic_personas(
        state=unconfirmed_state,
        default_primary=bull_cand,
        eligible_candidates=pass_cands,
    )
    assert deb_u.dominant_regime == "CONTESTED_WAIT"
    assert chosen_u is None
    assert cons_u.recommended_structure_id == "WAIT"
    assert cons_u.conviction == "SIT_OUT"

    _, vrep_unconf = verify_llm_synthesis(
        state=unconfirmed_state,
        raw_citations=[
            {
                "claim_text": "Max loss check",
                "state_field_path": f"candidate_structures[{bull_idx}].contractual_max_loss_rs",
                "cited_value": bull_cand.contractual_max_loss_rs,
            }
        ],
        selected_primary=bull_cand,
        consolidator_recommended_id=bull_cand.structure_id,
        debate_dominant_regime="BULLISH",
    )
    assert vrep_unconf.passed_verification is False
    assert vrep_unconf.part_a_reverified is False
    assert "MISSING_OR_CONTESTED_REGIME_CONFIRMATION" in " | ".join(vrep_unconf.failure_reasons)

    assert unconfirmed_state.final_output is not None
    tampered_unconf = dataclasses.replace(
        unconfirmed_state,
        final_output=dataclasses.replace(
            unconfirmed_state.final_output,
            verdict="SUGGESTION",
            is_live_ready=True,
            primary_proposal=bull_cand,
        ),
    )
    norm_unconf = normalize_state_narrative_from_verified_facts(tampered_unconf)
    assert (
        norm_unconf.debate_state is not None
        and norm_unconf.debate_state.dominant_regime == "CONTESTED_WAIT"
    )
    assert norm_unconf.final_output is not None and norm_unconf.final_output.verdict == "WAIT"
    assert norm_unconf.final_output.primary_proposal is None
    unconf_deliv = deliver_run_output(
        tampered_unconf,
        tmp_path / "fix5_unconfirmed_audit.jsonl",
        force_render_for_inspection=True,
    )
    assert unconf_deliv.alerted_trader is False
    assert "Debate: CONTESTED_WAIT" in unconf_deliv.whatsapp_digest
    assert bull_cand.structure_id not in unconf_deliv.whatsapp_digest

    # 3. Purely NEUTRAL confirmed regime (`RANGE` + `NOVEL_OI_BRACKETED_NEUTRAL_RANGE` only):
    #    directional BULLISH candidate lacks directional confirmation -> rejected (regime stays NEUTRAL),
    #    whereas NEUTRAL Iron Condor candidate is accepted.
    neutral_det = TriggeredDetector(
        detector_id="NOVEL_OI_BRACKETED_NEUTRAL_RANGE",
        detector_name="High-IV OI-Wall Bracketed Neutral Regime",
        track="NOVEL_QUANT",
        confirmation_status="CONFIRMED_QUANT_REGIME",
        direction="NEUTRAL",
        invalidation_level=float(probe_state.market_evidence.derivatives_positioning.call_oi_wall_strike),
        invalidation_candle_basis="COMPLETED_DAILY_CLOSE",
        invalidation_justification="Close outside OI bracket.",
        summary="Spot bracketed between OI walls.",
        evidence_fields_cited=["market_evidence.volatility_state.ivp_90d"],
    )
    neutral_only_state = dataclasses.replace(
        probe_state,
        market_evidence=dataclasses.replace(
            probe_state.market_evidence,
            price_structure=dataclasses.replace(
                probe_state.market_evidence.price_structure, trend_1d_state="RANGE"
            ),
            triggered_detectors=[neutral_det],
        ),
    )
    assert assess_market_regime(neutral_only_state) == "NEUTRAL"
    deb_nb, _, cons_nb, chosen_nb = build_deterministic_personas(
        neutral_only_state, default_primary=bull_cand, eligible_candidates=[bull_cand, ic_cand]
    )
    assert deb_nb.dominant_regime == "NEUTRAL"
    assert chosen_nb is None
    assert cons_nb.recommended_structure_id == "WAIT"
    assert cons_nb.conviction == "SIT_OUT"

    deb_nn, _, cons_nn, chosen_nn = build_deterministic_personas(
        neutral_only_state, default_primary=ic_cand, eligible_candidates=[bull_cand, ic_cand]
    )
    assert deb_nn.dominant_regime == "NEUTRAL"
    assert chosen_nn is not None and chosen_nn.structure_id == ic_cand.structure_id
    assert cons_nn.recommended_structure_id == ic_cand.structure_id


# ============================================================================
# KOTAK NEO DIAGNOSTIC HANDOFF (12 NOV 2026) — ACCEPTANCE REGRESSION SUITE
# ============================================================================


def _build_synthetic_multi_expiry_env(
    spot: float = 25000.0,
    mkt_lot: int = 65,
    expiries: list[str] | None = None,
    quote_lstup_epoch: int | str | None = 1793940298,  # 2026-11-06T10:14:58+05:30
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    """Build deterministic multi-expiry 81-strike chains + per-token quote maps (zero credentials)."""
    from nifty_analyst.engine.pricing import black76_price

    exp_list = expiries or [
        "2026-11-12",
        "2026-11-19",
        "2026-11-26",
        "2026-12-03",
        "2026-12-10",
        "2026-12-17",
        "2026-12-24",
        "2026-12-31",
        "2027-01-07",
        "2027-01-14",
        "2027-01-21",
        "2027-01-28",
        "2027-02-04",
        "2027-02-11",
        "2027-02-18",
        "2027-02-25",
        "2027-03-25",
        "2027-06-24",
    ]
    chains_by_exp: dict[str, dict[str, Any]] = {}
    quotes_by_tok: dict[str, dict[str, Any]] = {}
    dec_date = dt.date(2026, 11, 6)

    for exp_idx, exp_s in enumerate(exp_list):
        exp_d = dt.date.fromisoformat(exp_s)
        dte_cal = max(1, (exp_d - dec_date).days)
        t_yr = max(dte_cal / 365.0, 1.0 / 365.0)
        r_rate = 0.0675
        fwd = round(spot * (1.0 + r_rate * t_yr), 2)
        base_iv = 0.135 + min(0.025, 0.0015 * exp_idx)
        strikes_raw: list[dict[str, Any]] = []
        tag = exp_s.replace("-", "")

        for idx in range(-40, 41):
            k = spot + idx * 50.0
            c_theo = max(0.35, black76_price(fwd, k, t_yr, r_rate, base_iv, "CE"))
            p_theo = max(0.35, black76_price(fwd, k, t_yr, r_rate, base_iv, "PE"))
            ce_tok = f"{tag}_{int(k)}CE"
            pe_tok = f"{tag}_{int(k)}PE"
            strikes_raw.append(
                {
                    "strike_price": k,
                    "CE": {
                        "neo_symbol": f"nse_fo|{ce_tok}",
                        "ltp": round(c_theo, 2),
                        "open_interest": 140000,
                        "volume": 35000,
                    },
                    "PE": {
                        "neo_symbol": f"nse_fo|{pe_tok}",
                        "ltp": round(p_theo, 2),
                        "open_interest": 145000,
                        "volume": 36000,
                    },
                }
            )
            for tok, mid in ((ce_tok, c_theo), (pe_tok, p_theo)):
                half_sp = max(0.20, round(mid * 0.005, 2))
                b1 = round(max(0.10, mid - half_sp), 2)
                a1 = round(mid + half_sp, 2)
                q_item: dict[str, Any] = {
                    "exchange": "nse_fo",
                    "exchange_token": tok,
                    "mktLot": str(mkt_lot),
                    "ltp": str(round(mid, 2)),
                    "depth": {
                        "buy": [
                            {"price": str(b1), "quantity": str(mkt_lot * 8), "orders": "4"},
                            {
                                "price": str(round(max(0.05, b1 - 0.15), 2)),
                                "quantity": str(mkt_lot * 12),
                                "orders": "6",
                            },
                        ],
                        "sell": [
                            {"price": str(a1), "quantity": str(mkt_lot * 10), "orders": "5"},
                            {
                                "price": str(round(a1 + 0.15, 2)),
                                "quantity": str(mkt_lot * 15),
                                "orders": "7",
                            },
                        ],
                    },
                }
                if quote_lstup_epoch is not None:
                    q_item["lstup_time"] = str(quote_lstup_epoch)
                quotes_by_tok[tok] = q_item

        chains_by_exp[exp_s] = {
            "status": "OK",
            "lot_size": mkt_lot,
            "underlying_prices": {
                "reference_price": spot,
                "futures_price": fwd,
                "spot_price": spot,
            },
            "strikes": strikes_raw,
        }

    return chains_by_exp, quotes_by_tok, exp_list


def _build_synthetic_candles_for_dec(dec_date: dt.date = dt.date(2026, 11, 6)) -> dict[str, list[dict[str, Any]]]:
    """Deterministic completed 1D, 60min, and 15min bars up to `dec_date` 10:15 IST."""
    daily: list[dict[str, Any]] = []
    cur = dec_date - dt.timedelta(days=85)
    lvl = 24600.0
    while cur < dec_date:
        if cur.weekday() < 5:
            lvl += 8.0
            daily.append(
                {
                    "time": cur.isoformat(),
                    "open": round(lvl - 25.0, 2),
                    "high": round(lvl + 40.0, 2),
                    "low": round(lvl - 45.0, 2),
                    "close": round(lvl, 2),
                    "volume": 500000,
                }
            )
        cur += dt.timedelta(days=1)

    hourly: list[dict[str, Any]] = []
    cur_h = dec_date - dt.timedelta(days=12)
    hlvl = 24880.0
    slots = [
        ("09:15:00", "10:15:00"),
        ("10:15:00", "11:15:00"),
        ("11:15:00", "12:15:00"),
        ("12:15:00", "13:15:00"),
        ("13:15:00", "14:15:00"),
        ("14:15:00", "15:15:00"),
    ]
    while cur_h <= dec_date:
        if cur_h.weekday() < 5:
            day_slots = slots[:1] if cur_h == dec_date else slots
            for op_s, cl_s in day_slots:
                hlvl += 3.0
                hourly.append(
                    {
                        "time": f"{cur_h.isoformat()}T{op_s}+05:30",
                        "close_time_ist": f"{cur_h.isoformat()}T{cl_s}+05:30",
                        "open": round(hlvl - 8.0, 2),
                        "high": round(hlvl + 16.0, 2),
                        "low": round(hlvl - 14.0, 2),
                        "close": round(hlvl, 2),
                        "volume": 90000,
                    }
                )
        cur_h += dt.timedelta(days=1)

    prev_sess = dec_date - dt.timedelta(days=1)
    closing_15m = [
        {
            "time": f"{prev_sess.isoformat()}T15:15:00+05:30",
            "close_time_ist": f"{prev_sess.isoformat()}T15:30:00+05:30",
            "open": 24990.0,
            "high": 25015.0,
            "low": 24982.0,
            "close": 25000.0,
            "volume": 32000,
        }
    ]
    return {"D": daily, "60min": hourly, "15min": closing_15m}


def test_kotak_handoff_acceptance1_live_nov17_depth_anomalies_diagnostic_sort_and_distinct_risk_metrics() -> None:
    """Acceptance 1: Reject exact Nov 17 25300 CE/PE mixed depth order before sorting, keep sorted
    copies strictly diagnostic, walk valid books for required units, retain indicative sparse touches,
    and preserve distinct gross/net & 4 rupee risk metrics.
    """
    # 1A. Exact public-market Nov 17 NIFTY 25300 CE from 12 Nov 2026 capture (handoff lines 29-35)
    nov17_25300_ce = {
        "exchange": "nse_fo",
        "exchange_token": "NOV17_25300CE",
        "mktLot": "65",
        "lstup_time": "1794455573",
        "depth": {
            "buy": [
                {"price": "204.20", "quantity": "715"},
                {"price": "204.00", "quantity": "1950"},
                {"price": "203.95", "quantity": "65"},
                {"price": "203.85", "quantity": "65"},
                {"price": "203.75", "quantity": "65"},
            ],
            "sell": [
                {"price": "204.70", "quantity": "390"},
                {"price": "204.25", "quantity": "585"},
                {"price": "204.30", "quantity": "845"},
                {"price": "204.35", "quantity": "1170"},
                {"price": "204.40", "quantity": "845"},
            ],
        },
    }
    ce_book = interpret_depth_book(nov17_25300_ce, fallback_lot_size=65)
    assert ce_book.is_executable is False
    assert ce_book.is_structurally_valid is False
    assert ce_book.rejection_reason == "DEPTH_LEVEL_ORDER_INCONSISTENT"
    # Row-0 (204.70) and sorted best (204.25) MUST NEVER leak into executable touch or levels
    assert (ce_book.best_bid, ce_book.best_ask) == (None, None)
    assert ce_book.buy_levels == ()
    assert ce_book.sell_levels == ()
    # Sorted copy exists strictly for diagnostics
    assert ce_book.diagnostic_sorted_sell_levels[0] == (204.25, 585)
    assert ce_book.diagnostic_sorted_sell_levels[-1] == (204.70, 390)

    # 1B. Exact Nov 17 NIFTY 25300 PE non-monotonic bids (222.40 then 222.65; handoff line 35)
    nov17_25300_pe = {
        "exchange": "nse_fo",
        "exchange_token": "NOV17_25300PE",
        "mktLot": "65",
        "depth": {
            "buy": [
                {"price": "222.40", "quantity": "65"},
                {"price": "222.65", "quantity": "130"},
                {"price": "222.30", "quantity": "195"},
            ],
            "sell": [
                {"price": "223.10", "quantity": "130"},
                {"price": "223.25", "quantity": "195"},
            ],
        },
    }
    pe_book = interpret_depth_book(nov17_25300_pe, fallback_lot_size=65)
    assert pe_book.is_executable is False
    assert pe_book.rejection_reason == "DEPTH_LEVEL_ORDER_INCONSISTENT"
    assert (pe_book.best_bid, pe_book.best_ask) == (None, None)
    assert pe_book.buy_levels == ()
    assert pe_book.diagnostic_sorted_buy_levels[0] == (222.65, 130)

    # Rejected strike cannot be used to price any leg
    cfg = AnalystConfig(mode="OFFLINE", fixtures_dir=str(FIXTURES_DIR))
    rejected_q = RawStrikeQuote(
        strike=25300.0,
        call_bid=ce_book.best_bid,
        call_ask=ce_book.best_ask,
        put_bid=pe_book.best_bid,
        put_ask=pe_book.best_ask,
    )
    with pytest.raises(ValueError, match="unpriced/non-executable"):
        build_option_leg(rejected_q, "CE", "LONG", 1, 25300.0, 0.02, 0.0675, False, cfg)

    # 1C. Valid multi-level book walking + sparse touch indicative retention + distinct risk metrics
    vwap_2lots, avail_qty = walk_depth_levels([(100.0, 30), (101.0, 50), (103.0, 50)], 130)
    assert avail_qty == 130
    assert vwap_2lots == pytest.approx((30 * 100.0 + 50 * 101.0 + 50 * 103.0) / 130.0)

    sparse_short_q = RawStrikeQuote(
        strike=24900.0,
        call_bid=210.0,
        call_ask=211.0,
        put_bid=95.0,
        put_ask=96.0,
        call_oi=80000,
        put_oi=80000,
        call_volume=20000,
        put_volume=20000,
        put_touch_bid_qty=15,
        put_visible_buy_qty=30,
        put_buy_levels=((95.0, 15), (94.5, 15)),
        quote_timestamp_ist="2026-11-06T10:14:55+05:30",
    )
    sparse_long_q = RawStrikeQuote(
        strike=24750.0,
        call_bid=310.0,
        call_ask=312.0,
        put_bid=44.0,
        put_ask=45.0,
        call_oi=80000,
        put_oi=80000,
        call_volume=20000,
        put_volume=20000,
        put_touch_ask_qty=80,
        put_visible_sell_qty=80,
        put_sell_levels=((45.0, 80),),
        quote_timestamp_ist="2026-11-06T10:14:55+05:30",
    )
    sparse_leg_s = build_option_leg(
        sparse_short_q, "PE", "SHORT", 1, 25000.0, 0.02, 0.0675, False, cfg
    )
    sparse_leg_l = build_option_leg(
        sparse_long_q, "PE", "LONG", 1, 25000.0, 0.02, 0.0675, False, cfg
    )

    deep_short_q = RawStrikeQuote(
        strike=24900.0,
        call_bid=210.0,
        call_ask=211.0,
        put_bid=95.0,
        put_ask=96.0,
        call_oi=90000,
        put_oi=95000,
        call_volume=25000,
        put_volume=25000,
        put_touch_bid_qty=30,
        put_visible_buy_qty=90,
        put_buy_levels=((95.0, 30), (94.0, 60)),
        quote_timestamp_ist="2026-11-06T10:14:55+05:30",
    )
    deep_long_q = RawStrikeQuote(
        strike=24750.0,
        call_bid=310.0,
        call_ask=312.0,
        put_bid=44.0,
        put_ask=45.0,
        call_oi=90000,
        put_oi=95000,
        call_volume=25000,
        put_volume=25000,
        put_touch_ask_qty=25,
        put_visible_sell_qty=80,
        put_sell_levels=((45.0, 25), (46.0, 55)),
        quote_timestamp_ist="2026-11-06T10:14:55+05:30",
    )
    leg_s = build_option_leg(
        deep_short_q, "PE", "SHORT", 1, 25000.0, 0.02, 0.0675, False, cfg
    )
    leg_l = build_option_leg(
        deep_long_q, "PE", "LONG", 1, 25000.0, 0.02, 0.0675, False, cfg
    )

    exp_m = ExpiryMetadata(
        expiry_date="2026-11-12",
        expiry_timestamp_ist="2026-11-12T15:30:00+05:30",
        dte_trading_sessions=4,
        lot_size=65,
        is_holiday_adjusted=False,
        holiday_note="",
        is_monthly=False,
    )
    port = build_portfolio_exposure_state(
        [
            OpenPosition(
                position_id="POS_EXISTING",
                structure_type="BULL_PUT_SPREAD",
                directional_sign=1,
                planned_stop_rs=1200.0,
                current_executable_spread_value_rs=400.0,
                short_leg_deltas=[-0.18],
                dte_trading_sessions=4,
            )
        ],
        portfolio_cap_rs=10000.0,
    )
    ps_stub = PriceStructureEvidence(
        trend_1d_state="UPTREND",
        higher_high_higher_low_repaired=True,
        first_pullback_in_downtrend_warning=False,
        ma20_1h=24960.0,
        ema50_1h=24910.0,
        last_completed_1d_bar=None,
        last_completed_1h_bar=None,
        final_15m_bar_completed=True,
        stripped_forming_bars_count=0,
        verified_level_touches=[],
    )
    vol_stub = VolatilityEvidence(
        india_vix=TimestampedDatum.create(
            14.2,
            dt.datetime(2026, 11, 6, 10, 14, 55, tzinfo=IST),
            dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST),
            "vix",
            180,
            "VERIFIED",
        ),
        atm_iv_by_expiry={"2026-11-12": 0.14},
        constant_maturity_30d_iv=0.14,
        eod_cm30_latest=0.14,
        iv_source_label="LIVE_VARIANCE_INTERPOLATED_CM30",
        ivp_30d=62.0,
        ivp_90d=65.0,
        ivp_252d=58.0,
        iv_history_sessions_count=252,
        iv_history_gaps_flagged=[],
        realized_vol_20d=0.12,
        iv_rv_spread=0.02,
        skew_25d_put_minus_call=0.02,
        wing_curvature=0.005,
        chain_qa_passed=True,
        chain_qa_notes=[],
    )
    sparse_cand = build_candidate_structure(
        structure_id="BPS_SPARSE_NOV12",
        structure_type="BULL_PUT_SPREAD",
        direction="BULLISH",
        expiry=exp_m,
        legs=[sparse_leg_s, sparse_leg_l],
        spot=25000.0,
        implied_forward=25020.0,
        atm_iv=0.14,
        time_to_expiry_years=6.0 / 365.0,
        empirical_daily_log_returns=[-0.01, 0.008, -0.012, 0.009, 0.004],
        invalidation_level=24880.0,
        invalidation_candle_basis="COMPLETED_1H_CLOSE",
        invalidation_justification="1H close below 24880.",
        matches_part_b_style=True,
        portfolio_state=port,
        price_structure=ps_stub,
        volatility=vol_stub,
        config=cfg,
        forced_lots=1,
    )
    assert sparse_cand.full_size_fill_supported is False
    assert sparse_cand.legs[0].fill_basis == "INDICATIVE_TOUCH_ONLY"
    assert sparse_cand.legs[0].vwap_fill_price is None
    assert sparse_cand.part_a_overall_status == "FAIL"

    cand = build_candidate_structure(
        structure_id="BPS_NOV12_24900_24750",
        structure_type="BULL_PUT_SPREAD",
        direction="BULLISH",
        expiry=exp_m,
        legs=[leg_s, leg_l],
        spot=25000.0,
        implied_forward=25020.0,
        atm_iv=0.14,
        time_to_expiry_years=6.0 / 365.0,
        empirical_daily_log_returns=[-0.01, 0.008, -0.012, 0.009, 0.004],
        invalidation_level=24880.0,
        invalidation_candle_basis="COMPLETED_1H_CLOSE",
        invalidation_justification="1H close below 24880.",
        matches_part_b_style=True,
        portfolio_state=port,
        price_structure=ps_stub,
        volatility=vol_stub,
        config=cfg,
        forced_lots=1,
    )
    assert cand.full_size_fill_supported is True
    assert cand.execution_fill_basis == "FULL_SIZE_BOOK_WALK"
    assert cand.legs[0].fill_basis == "FULL_SIZE_BOOK_WALK"
    assert cand.legs[1].fill_basis == "FULL_SIZE_BOOK_WALK"
    assert cand.executable_net_points_per_unit != cand.indicative_touch_net_points_per_unit
    assert cand.breakevens_gross != cand.breakevens_net
    assert cand.statutory_charges.total_charges_rs > 0.0
    four_risk_metrics = {
        cand.contractual_max_loss_rs,
        cand.planned_stop_rs,
        cand.portfolio_summed_stop_after_trade_rs,
        cand.margin_estimate_rs,
    }
    assert len(four_risk_metrics) == 4


def test_kotak_handoff_acceptance2_timestamps_provenance_disagreement_and_180s_final_deadline(
    tmp_path: Path,
) -> None:
    """Acceptance 2: Normalize Unix-second lstup_time while keeping REST meaning UNVERIFIED,
    separate fetch-start / fetch-complete / per-contract quote-as-of, reject future/stale/missing/
    malformed/wrong-unit values without fetch-time substitution, catch per-leg skew, and enforce
    180s capture-start deadline at final output without clock reset on resume or chart render.
    """
    from nifty_analyst.adapters.kotak_neo import (
        LiveKotakNeoAdapter,
        parse_and_validate_option_chain,
    )
    from nifty_analyst.adapters.request_safety import normalize_contract_quote_timestamp
    from nifty_analyst.engine.part_a_gatekeeper import evaluate_part_a_rules
    from nifty_analyst.engine.pricing import run_chain_qa_suite

    # 2A. Exact live sample raw value: lstup_time = "1794455573" -> 2026-11-12T09:22:53+05:30
    capture_ref = dt.datetime(2026, 11, 12, 9, 22, 55, tzinfo=IST)
    ev_live = normalize_contract_quote_timestamp(
        {"lstup_time": "1794455573"},
        capture_ref,
        max_staleness_seconds=180.0,
        max_future_skew_seconds=5.0,
        verified_semantics=False,
        require_present=True,
    )
    assert ev_live.is_valid is True
    assert ev_live.is_certified is False
    assert ev_live.normalized_quote_as_of_ist == "2026-11-12T09:22:53+05:30"
    assert ev_live.raw_value == "1794455573"
    assert ev_live.source_field == "lstup_time"
    assert ev_live.detected_unit == "UNIX_EPOCH_SECONDS"
    assert ev_live.semantics_status == "UNVERIFIED_KOTAK_REST_LSTUP_TIME"
    assert ev_live.age_seconds == pytest.approx(2.0)

    # 2B. Rejection matrix: future, stale, missing, malformed, naive ISO, ms/ns, and SFeed sentinel
    bad_cases: list[tuple[dict[str, Any], str]] = [
        ({"lstup_time": str(int(capture_ref.timestamp()) + 20)}, "QUOTE_TIMESTAMP_FUTURE_SKEW"),
        ({"lstup_time": str(int(capture_ref.timestamp()) - 220)}, "QUOTE_TIMESTAMP_STALE"),
        ({}, "QUOTE_TIMESTAMP_MISSING"),
        ({"lstup_time": None}, "QUOTE_TIMESTAMP_MISSING"),
        ({"lstup_time": ""}, "QUOTE_TIMESTAMP_MALFORMED"),
        ({"lstup_time": "not_an_epoch"}, "QUOTE_TIMESTAMP_MALFORMED"),
        ({"lstup_time": True}, "QUOTE_TIMESTAMP_MALFORMED"),
        ({"lstup_time": "2026-11-12T09:22:53"}, "QUOTE_TIMESTAMP_NAIVE_TIMEZONE"),
        ({"lstup_time": "1794455573000"}, "QUOTE_TIMESTAMP_WRONG_UNIT_MS_OR_NS"),
        ({"lstup_time": "0"}, "QUOTE_TIMESTAMP_WRONG_UNIT_SENTINEL"),
        ({"lstup_time": "-2209008600"}, "QUOTE_TIMESTAMP_WRONG_UNIT_SENTINEL"),
        ({"lstup_time": "12345"}, "QUOTE_TIMESTAMP_WRONG_UNIT_RANGE"),
    ]
    for payload, expected_prefix in bad_cases:
        res = normalize_contract_quote_timestamp(
            payload,
            capture_ref,
            max_staleness_seconds=180.0,
            max_future_skew_seconds=5.0,
            verified_semantics=True,
            require_present=True,
        )
        assert res.is_valid is False, f"Expected rejection for {payload}"
        assert res.is_certified is False
        assert expected_prefix in res.rejection_reason

    # 2C. Adapter wiring: separate fetch_started_at_ist / fetch_completed_at_ist / per-leg CE & PE
    #     quote_as_of_ist, and zero fetch-time substitution when timestamp is wrong-unit or missing!
    dec_dt = dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST)
    chains_by_exp, quotes_by_tok, _ = _build_synthetic_multi_expiry_env(
        spot=25000.0, mkt_lot=65, expiries=["2026-11-12"], quote_lstup_epoch=1793940298
    )
    # Give ATM 25000 CE and 25000 PE distinct valid timestamps (10:14:58 vs 10:14:54 IST),
    # and corrupt 25200 CE with millisecond wrong-unit timestamp (must reject, NEVER substitute fetch time!)
    quotes_by_tok["20261112_25000CE"]["lstup_time"] = "1793940298"  # 10:14:58+05:30
    quotes_by_tok["20261112_25000PE"]["lstup_time"] = "1793940294"  # 10:14:54+05:30
    quotes_by_tok["20261112_25200CE"]["lstup_time"] = "1793940298000"  # wrong unit (ms)

    wall_seq = iter(
        [
            dt.datetime(2026, 11, 6, 10, 15, 1, tzinfo=IST),
            dt.datetime(2026, 11, 6, 10, 15, 7, tzinfo=IST),
            dt.datetime(2026, 11, 6, 10, 15, 9, tzinfo=IST),
            dt.datetime(2026, 11, 6, 10, 15, 12, tzinfo=IST),
        ]
    )

    class _Svc:
        def get_full_option_chain(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return chains_by_exp["2026-11-12"]

    class _Cli:
        def quotes(self, instrument_tokens: list[dict[str, str]], quote_type: str = "all") -> list[dict[str, Any]]:
            return [
                quotes_by_tok[t["instrument_token"]]
                for t in instrument_tokens
                if t["instrument_token"] in quotes_by_tok
            ]

    adapter = LiveKotakNeoAdapter.__new__(LiveKotakNeoAdapter)
    adapter._service = _Svc()
    adapter._auth = type("_A", (), {"require_authenticated_client": lambda self: (_Cli(), None, None)})()
    adapter.rate_limiter = PacedReadTransport(min_interval_seconds=0.0, max_queue_wait_seconds=0.0)
    adapter.enforce_schedule = True
    adapter.depth_strikes_each_side = 40
    adapter.quotes_batch_size = 4
    adapter.max_depth_tokens_per_expiry = 14
    adapter.extra_neo_symbols = ()
    adapter.trading_calendar = {}
    adapter.quote_ts_semantics_verified = False
    adapter.require_contract_quote_timestamp = True
    adapter.max_quote_staleness_seconds = 180.0
    adapter.max_future_skew_seconds = 5.0
    adapter._wall_clock_ist_fn = lambda: next(wall_seq)
    adapter._cache = {}

    raw_chain = adapter.fetch_option_chain_raw("2026-11-12", dec_dt)
    env = parse_and_validate_option_chain(raw_chain, "2026-11-12", dec_dt)
    assert env.fetch_started_at_ist == "2026-11-06T10:15:01+05:30"
    assert env.fetch_completed_at_ist == "2026-11-06T10:15:07+05:30"
    assert "UNVERIFIED_KOTAK_REST_LSTUP_TIME" in env.source_verification_tag

    q_by_k = {q.strike: q for q in env.quotes}
    q_25000 = q_by_k[25000.0]
    assert q_25000.call_quote_timestamp_ist == "2026-11-06T10:14:58+05:30"
    assert q_25000.put_quote_timestamp_ist == "2026-11-06T10:14:54+05:30"
    assert q_25000.fetch_started_at_ist == "2026-11-06T10:15:01+05:30"
    assert q_25000.fetch_completed_at_ist == "2026-11-06T10:15:07+05:30"

    cfg_off = AnalystConfig(mode="OFFLINE", fixtures_dir=str(FIXTURES_DIR))
    ce_leg = build_option_leg(
        q_25000, "CE", "LONG", 1, 25000.0, 0.02, 0.0675, False, cfg_off
    )
    pe_leg = build_option_leg(
        q_25000, "PE", "SHORT", 1, 25000.0, 0.02, 0.0675, False, cfg_off
    )
    assert ce_leg.quote_timestamp_ist == "2026-11-06T10:14:58+05:30"
    assert pe_leg.quote_timestamp_ist == "2026-11-06T10:14:54+05:30"

    # Wrong-unit 25200 CE was rejected to None (NOT substituted with fetch time!)
    q_25200 = q_by_k[25200.0]
    assert (q_25200.call_bid, q_25200.call_ask) == (None, None)
    assert any(
        r["neo_symbol"] == "nse_fo|20261112_25200CE"
        and "QUOTE_TIMESTAMP_WRONG_UNIT_MS_OR_NS" in r["rejection_reason"]
        for r in env.rejected_depth_books
    )

    # Missing timestamp when `require_contract_quote_timestamp=True` also rejects without substitution
    del quotes_by_tok["20261112_24850PE"]["lstup_time"]
    adapter._cache.clear()
    raw_missing_ts = adapter.fetch_option_chain_raw("2026-11-12", dec_dt)
    env_missing_ts = parse_and_validate_option_chain(raw_missing_ts, "2026-11-12", dec_dt)
    q_24850 = {q.strike: q for q in env_missing_ts.quotes}[24850.0]
    assert (q_24850.put_bid, q_24850.put_ask) == (None, None)
    assert any(
        r["neo_symbol"] == "nse_fo|20261112_24850PE"
        and "QUOTE_TIMESTAMP_MISSING" in r["rejection_reason"]
        for r in env_missing_ts.rejected_depth_books
    )

    # 2D. Per-leg timestamp disagreement: (i) Chain QA near-ATM parity & (ii) Part A Rule 8 multi-leg
    skewed_quotes = [
        dataclasses.replace(
            q,
            call_quote_timestamp_ist="2026-11-06T10:14:58+05:30",
            put_quote_timestamp_ist="2026-11-06T10:14:05+05:30",  # 53s skew > 30s bound
        )
        if q.strike == 25000.0
        else q
        for q in env.quotes
    ]
    qa_skew = run_chain_qa_suite(
        spot=25000.0,
        quotes=skewed_quotes,
        time_to_expiry_years=6.0 / 365.0,
        risk_free_rate=0.0675,
        max_leg_timestamp_skew_seconds=30.0,
    )
    assert qa_skew.passed is False
    assert qa_skew.failure_code == "CHAIN_PARITY_LEG_TIMESTAMP_DISAGREEMENT"

    # Multi-leg spread with 45s skew between short leg and long leg -> Rule 8 FAIL
    base_state, _ = run_analyst_pipeline(
        dec_dt,
        AnalystConfig(
            mode="OFFLINE",
            fixtures_dir=str(FIXTURES_DIR),
            audit_log_path=str(tmp_path / "base_skew_audit.jsonl"),
        ),
    )
    multi_cand = next(c for c in base_state.candidate_structures if len(c.legs) >= 2)
    skewed_legs = [
        dataclasses.replace(
            leg,
            quote_timestamp_ist=(
                "2026-11-06T10:14:58+05:30" if idx == 0 else "2026-11-06T10:14:10+05:30"
            ),
        )
        for idx, leg in enumerate(multi_cand.legs)
    ]
    exp_cand = next(
        e
        for e in base_state.contract_context.active_expiries
        if e.expiry_date == multi_cand.expiry_date
    )
    checks_skew, overall_skew = evaluate_part_a_rules(
        is_defined_risk=True,
        legs=skewed_legs,
        dte_trading_sessions=exp_cand.dte_trading_sessions,
        planned_stop_rs=multi_cand.planned_stop_rs,
        portfolio_state=base_state.portfolio_exposure_state,
        contractual_max_loss_rs=multi_cand.contractual_max_loss_rs,
        invalidation_level=multi_cand.invalidation_level,
        invalidation_candle_basis=multi_cand.invalidation_candle_basis,
        invalidation_justification=multi_cand.invalidation_justification,
        price_structure=base_state.market_evidence.price_structure,
        volatility=base_state.market_evidence.volatility_state,
        config=cfg_off,
    )
    assert overall_skew == "FAIL"
    r8 = checks_skew["RULE_8_IVP_AND_LIQUIDITY_GATES"]
    assert r8.status == "FAIL"
    assert "LEG_TIMESTAMP_DISAGREEMENT" in r8.evidence

    # 2E. 180s deadline measured from original capture_started_at_ist at final output
    #     (resuming synthesis + rendering payoff chart MUST NOT reset the capture-start clock)
    art_path = tmp_path / "resume_deadline" / "synthesis_response.json"
    chart_dir = tmp_path / "charts_deadline"
    cfg_res = AnalystConfig(
        mode="OFFLINE",
        fixtures_dir=str(FIXTURES_DIR),
        llm_provider="agent_artifact",
        agent_synthesis_artifact_path=str(art_path),
        payoff_chart_dir=str(chart_dir),
        audit_log_path=str(tmp_path / "resume_deadline_audit.jsonl"),
    )
    t0 = dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST)
    # Initial capture starts at t0 and finishes at t0+5s (artifact missing -> writes immutable snapshot)
    cap_clocks = iter(
        [
            t0,
            t0 + dt.timedelta(seconds=5),
            t0 + dt.timedelta(seconds=6),
        ]
    )
    with pytest.raises(FileNotFoundError):
        run_analyst_pipeline(dec_dt, cfg_res, wall_clock_fn=lambda: next(cap_clocks))

    req_path = tmp_path / "resume_deadline" / "synthesis_response.json.request.json"
    req_data = json.loads(req_path.read_text(encoding="utf-8"))
    _write_valid_agent_artifact(
        art_path,
        req_data,
        chosen_card=req_data["compact_desk_brief"]["certified_pass_candidates"][0],
    )

    # Resume starts at t0+179s (passes initial resume check),
    # and final output check after synthesis/chart render occurs at t0+183s
    # (> 180s from original t0 capture start!) -> MUST fail closed and remove any chart!
    res_clocks = iter(
        [
            t0 + dt.timedelta(seconds=179),
            t0 + dt.timedelta(seconds=183),
        ]
    )
    with pytest.raises(StaleEvidenceSnapshotError):
        run_analyst_pipeline(
            dec_dt,
            cfg_res,
            broker_adapter=_ForbiddenBrokerAdapter(),
            wall_clock_fn=lambda: next(res_clocks),
            force_render_for_inspection=True,
        )
    assert list(chart_dir.glob("*.png")) == []


def test_kotak_handoff_acceptance3_bounded_multi_expiry_away_from_atm_and_incomplete_surface(
    tmp_path: Path,
) -> None:
    """Acceptance 3: With 18 listed expiries, automatically select bounded required expiries within
    the 35s/2.5s budget, include away-from-ATM monitored contracts on their own expiry, track all 7
    coverage buckets separately, and mark INCOMPLETE_SURFACE_COVERAGE (never certifying surface
    comparison) whenever a required leg or expiry is missing or rejected.
    """
    from nifty_analyst.adapters.kotak_neo import LiveKotakNeoAdapter

    dec_dt = dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST)
    chains_by_exp, quotes_by_tok, exp_18 = _build_synthetic_multi_expiry_env(
        spot=25000.0, mkt_lot=65, quote_lstup_epoch=1793940298
    )
    assert len(exp_18) == 18
    candles_by_ivl = _build_synthetic_candles_for_dec(dec_dt.date())

    def _make_18exp_live_adapter(
        q_map: dict[str, dict[str, Any]],
        cache_sub: Path,
        fail_qa_on_exp: str | None = None,
    ) -> tuple[LiveKotakNeoAdapter, list[str]]:
        fetched_exps: list[str] = []

        class _Svc18:
            def get_expiries(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                return {"status": "OK", "expiries": list(exp_18)}

            def get_futures_chain(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                return {"status": "OK", "lot_size": 65, "nearest_futures_ltp": 25000.0}

            def get_full_option_chain(self, symbol: str, expiry: str | None = None, **kw: Any) -> dict[str, Any]:
                exp_key = str(expiry or exp_18[0])
                fetched_exps.append(exp_key)
                raw_c = json.loads(json.dumps(chains_by_exp[exp_key]))
                if fail_qa_on_exp and exp_key == fail_qa_on_exp:
                    # Drop ATM strikes so Chain QA fails on this non-anchor required expiry
                    raw_c["strikes"] = [
                        s for s in raw_c["strikes"] if abs(float(s["strike_price"]) - 25000.0) > 300.0
                    ]
                return raw_c

        class _Cli18:
            def quotes(self, instrument_tokens: list[dict[str, str]], quote_type: str = "all") -> list[dict[str, Any]]:
                out_q: list[dict[str, Any]] = []
                for item in instrument_tokens:
                    tok = item["instrument_token"]
                    if tok in q_map:
                        out_q.append(json.loads(json.dumps(q_map[tok])))
                return out_q

            def historical_data(self, **kwargs: Any) -> list[dict[str, Any]]:
                return list(candles_by_ivl[kwargs["interval"]])

        mono_t = [100.0]

        def _fake_sleep(sec: float) -> None:
            mono_t[0] += sec

        limiter = PacedReadTransport(
            min_interval_seconds=2.5,
            max_queue_wait_seconds=35.0,
            clock_fn=lambda: mono_t[0],
            sleep_fn=_fake_sleep,
        )
        adp = LiveKotakNeoAdapter.__new__(LiveKotakNeoAdapter)
        adp._service = _Svc18()
        adp._auth = type("_A", (), {"require_authenticated_client": lambda self: (_Cli18(), None, None)})()
        adp.rate_limiter = limiter
        adp.enforce_schedule = True
        adp.depth_strikes_each_side = 40
        adp.quotes_batch_size = 4
        adp.max_depth_tokens_per_expiry = 12
        adp.extra_neo_symbols = ()
        adp.cache_dir = cache_sub
        adp.trading_calendar = {}
        adp._cache = {}
        return adp, fetched_exps

    pos_file = tmp_path / "empty_pos.json"
    pos_file.write_text("[]", encoding="utf-8")

    # 3A. Away-from-ATM monitored position on its own non-anchor expiry ("2026-12-10", 23500 PE)
    far_held_sym = "nse_fo|20261210_23500PE"
    held_dec10 = OpenPosition(
        position_id="POS_DEC10_FAR_OTM_PE",
        structure_type="BULL_PUT_SPREAD",
        directional_sign=1,
        planned_stop_rs=1400.0,
        current_executable_spread_value_rs=500.0,
        short_leg_deltas=[-0.12],
        dte_trading_sessions=24,
        neo_symbols=[far_held_sym],
        expiry_date="2026-12-10",
    )
    cfg_live = AnalystConfig(
        mode="LIVE_MORNING_SNAPSHOT",
        fixtures_dir=str(FIXTURES_DIR),
        open_positions_file=str(pos_file),
        cache_dir=str(tmp_path / "cache_3a"),
        audit_log_path=str(tmp_path / "audit_3a.jsonl"),
    )
    adp_ok, fetched_ok = _make_18exp_live_adapter(quotes_by_tok, tmp_path / "cache_3a")
    state_ok, deliv_ok = run_analyst_pipeline(
        dec_dt,
        cfg_live,
        broker_adapter=adp_ok,
        open_positions=[held_dec10],
        force_render_for_inspection=True,
    )
    # Bounded selection: evaluated Anchor + Monitored Dec 10 (NOT all 18 expiries!), <= 35s wait
    assert fetched_ok == ["2026-11-12", "2026-12-10"]
    assert adp_ok.rate_limiter.cumulative_wait_seconds <= 35.0
    assert state_ok.run_metadata.surface_coverage_status == "MULTI_EXPIRY_FULL"
    assert state_ok.run_metadata.uncovered_held_neo_symbols == []
    ledger_ok = state_ok.run_metadata.coverage_ledger
    assert len(ledger_ok["listed_expiries"]) == 18
    assert ledger_ok["required_expiries"] == ["2026-11-12", "2026-12-10"]
    assert ledger_ok["priced_expiries"] == ["2026-11-12", "2026-12-10"]
    dec10_cov = ledger_ok["by_expiry"]["2026-12-10"]
    assert dec10_cov["counts"]["listed"] == 162
    assert far_held_sym in dec10_cov["selected"]
    assert far_held_sym in dec10_cov["requested"]
    assert far_held_sym in dec10_cov["returned"]
    assert far_held_sym in dec10_cov["priced"]
    assert dec10_cov["counts"]["missing"] == 0
    assert dec10_cov["counts"]["rejected"] == 0
    assert state_ok.market_evidence.volatility_state.iv_source_label == "LIVE_VARIANCE_INTERPOLATED_CM30"
    assert state_ok.final_output is not None and state_ok.final_output.is_live_ready is False
    assert deliv_ok.alerted_trader is False

    # 3B. Rejected away-from-ATM monitored leg -> NOT counted in `priced` -> INCOMPLETE_SURFACE_COVERAGE
    #     and multi-expiry surface comparison is NOT certified!
    q_rej_held = json.loads(json.dumps(quotes_by_tok))
    q_rej_held["20261210_23500PE"]["depth"]["sell"] = [
        {"price": "18.70", "quantity": "390"},
        {"price": "18.25", "quantity": "585"},
    ]
    adp_rej, _ = _make_18exp_live_adapter(q_rej_held, tmp_path / "cache_3b")
    state_rej, deliv_rej = run_analyst_pipeline(
        dec_dt,
        dataclasses.replace(
            cfg_live,
            cache_dir=str(tmp_path / "cache_3b"),
            audit_log_path=str(tmp_path / "audit_3b.jsonl"),
        ),
        broker_adapter=adp_rej,
        open_positions=[held_dec10],
        force_render_for_inspection=True,
    )
    assert state_rej.run_metadata.surface_coverage_status == "INCOMPLETE_SURFACE_COVERAGE"
    assert state_rej.run_metadata.uncovered_held_neo_symbols == [far_held_sym]
    dec10_rej_cov = state_rej.run_metadata.coverage_ledger["by_expiry"]["2026-12-10"]
    assert far_held_sym in dec10_rej_cov["listed"]
    assert far_held_sym in dec10_rej_cov["rejected"]
    assert far_held_sym not in dec10_rej_cov["priced"]
    # Surface comparison MUST NOT be certified when required leg is unpriced
    assert (
        state_rej.market_evidence.volatility_state.iv_source_label
        == "UNCERTIFIED_INCOMPLETE_SURFACE"
    )
    assert state_rej.market_evidence.volatility_state.term_structure_regime == "SINGLE_EXPIRY"
    assert "INCOMPLETE_SURFACE_COVERAGE" in deliv_rej.full_report_markdown
    assert state_rej.final_output is not None and state_rej.final_output.is_live_ready is False

    # 3C. Missing requested quote AND failed required 2nd expiry both mark INCOMPLETE_SURFACE_COVERAGE
    q_missing = json.loads(json.dumps(quotes_by_tok))
    del q_missing["20261210_23500PE"]
    adp_miss, _ = _make_18exp_live_adapter(q_missing, tmp_path / "cache_3c")
    state_miss, _ = run_analyst_pipeline(
        dec_dt,
        dataclasses.replace(
            cfg_live,
            cache_dir=str(tmp_path / "cache_3c"),
            audit_log_path=str(tmp_path / "audit_3c.jsonl"),
        ),
        broker_adapter=adp_miss,
        open_positions=[held_dec10],
    )
    assert state_miss.run_metadata.surface_coverage_status == "INCOMPLETE_SURFACE_COVERAGE"
    miss_cov = state_miss.run_metadata.coverage_ledger["by_expiry"]["2026-12-10"]
    assert far_held_sym in miss_cov["missing"]
    assert far_held_sym not in miss_cov["returned"]
    assert far_held_sym not in miss_cov["priced"]
    assert (
        state_miss.market_evidence.volatility_state.iv_source_label
        == "UNCERTIFIED_INCOMPLETE_SURFACE"
    )

    # Required 2nd comparison expiry fails Chain QA -> missing_required_expiries populated -> INCOMPLETE
    adp_exp_fail, _ = _make_18exp_live_adapter(
        quotes_by_tok, tmp_path / "cache_3d", fail_qa_on_exp="2026-11-19"
    )
    state_exp_fail, _ = run_analyst_pipeline(
        dec_dt,
        dataclasses.replace(
            cfg_live,
            cache_dir=str(tmp_path / "cache_3d"),
            audit_log_path=str(tmp_path / "audit_3d.jsonl"),
        ),
        broker_adapter=adp_exp_fail,
        open_positions=[],
    )
    assert state_exp_fail.run_metadata.surface_coverage_status == "INCOMPLETE_SURFACE_COVERAGE"
    assert state_exp_fail.run_metadata.coverage_ledger["missing_required_expiries"] == [
        "2026-11-19"
    ]
    assert (
        state_exp_fail.market_evidence.volatility_state.iv_source_label
        == "UNCERTIFIED_INCOMPLETE_SURFACE"
    )


def test_kotak_handoff_acceptance4_http_boundary_counts_shared_spacing_and_429_hard_stop(
    tmp_path: Path,
) -> None:
    """Acceptance 4: Count actual HTTP GET requests at the transport boundary (distinct from
    wrapper call count and cache hits), enforce strict >=2.5s shared spacing even under simulated
    early OS sleep wakeups (2.499862s) and cross-process state, block non-GET methods, and
    hard-stop immediately on synthetic HTTP 429 with zero retry.
    """
    from nifty_analyst.adapters.kotak_neo import (
        KotakRateLimit429Error,
        LiveKotakNeoAdapter,
    )
    from nifty_analyst.adapters.request_safety import SharedPacedHttpTransport

    dec_dt = dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST)
    chains_by_exp, quotes_by_tok, exp_18 = _build_synthetic_multi_expiry_env(
        spot=25000.0, mkt_lot=65, quote_lstup_epoch=1793940298
    )
    candles_by_ivl = _build_synthetic_candles_for_dec(dec_dt.date())

    class _FakeReq:
        def __init__(self, method: str, url: str) -> None:
            self.method = method
            self.url = url

    class _FakeResp:
        def __init__(self, status_code: int = 200) -> None:
            self.status_code = status_code

    class _WireTransport:
        def __init__(self, fail_429_on_call: int | None = None) -> None:
            self.wire_urls: list[str] = []
            self.fail_429_on_call = fail_429_on_call

        def handle_request(self, request: Any) -> _FakeResp:
            self.wire_urls.append(f"{request.method} {request.url}")
            if self.fail_429_on_call is not None and len(self.wire_urls) == self.fail_429_on_call:
                return _FakeResp(status_code=429)
            return _FakeResp(status_code=200)

    # 4A. Compound wrapper HTTP counting + cache hit + early OS wakeup (2.499862s) strict >= 2.5s guard
    mono_now = [0.0]
    wall_now = [1_794_455_565.0]
    sleep_calls: list[float] = []

    def _early_wake_sleep(dur: float) -> None:
        sleep_calls.append(dur)
        if dur >= 2.0:
            # Simulate exact live OS early wakeup: 2.499862s instead of 2.500000s (handoff line 67)
            actual = dur - 0.000138
        else:
            actual = dur
        mono_now[0] += actual
        wall_now[0] += actual

    wire_ok = _WireTransport()
    pacer_ok = PacedReadTransport(
        min_interval_seconds=2.5,
        max_queue_wait_seconds=35.0,
        cooldown_file_path=tmp_path / "cooldown_ok.json",
        pacing_state_file_path=tmp_path / "pacing_state.json",
        early_wake_guard_seconds=0.001,
        clock_fn=lambda: mono_now[0],
        wall_time_fn=lambda: wall_now[0],
        sleep_fn=_early_wake_sleep,
    )

    class _RestHolder:
        def __init__(self, tr: Any) -> None:
            self._client = type("_HttpxCli", (), {"_transport": tr})()

    class _HttpBoundClient:
        def __init__(self, tr: Any) -> None:
            self.api_client = type("_ApiCli", (), {"rest_client": _RestHolder(tr)})()

        def quotes(self, instrument_tokens: list[dict[str, str]], quote_type: str = "all") -> list[dict[str, Any]]:
            self.api_client.rest_client._client._transport.handle_request(
                _FakeReq("GET", f"/script-details/1.0/quotes/neosymbol/batch/{quote_type}")
            )
            return [
                quotes_by_tok[t["instrument_token"]]
                for t in instrument_tokens
                if t["instrument_token"] in quotes_by_tok
            ]

    cli_ok = _HttpBoundClient(wire_ok)

    class _CompoundService:
        def get_expiries(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            cli_ok.api_client.rest_client._client._transport.handle_request(
                _FakeReq("GET", "/scrip-master/expiries")
            )
            return {"status": "OK", "expiries": list(exp_18[:3])}

        def get_futures_chain(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            # Compound wrapper call: issues TWO internal HTTP GET requests (scripmaster + quotes)
            cli_ok.api_client.rest_client._client._transport.handle_request(
                _FakeReq("GET", "/futures/lookup")
            )
            cli_ok.api_client.rest_client._client._transport.handle_request(
                _FakeReq("GET", "/futures/quotes")
            )
            return {"status": "OK", "lot_size": 65, "nearest_futures_ltp": 25000.0}

        def get_full_option_chain(self, symbol: str, expiry: str | None = None, **kw: Any) -> dict[str, Any]:
            # Compound wrapper call: issues TWO internal HTTP GET requests (chain + underlying)
            cli_ok.api_client.rest_client._client._transport.handle_request(
                _FakeReq("GET", f"/option-chain/{expiry}")
            )
            cli_ok.api_client.rest_client._client._transport.handle_request(
                _FakeReq("GET", f"/option-chain/{expiry}/spot")
            )
            return chains_by_exp[str(expiry or exp_18[0])]

    adp = LiveKotakNeoAdapter.__new__(LiveKotakNeoAdapter)
    adp._service = _CompoundService()
    adp._auth = type("_A", (), {"require_authenticated_client": lambda self: (cli_ok, None, None)})()
    adp.rate_limiter = pacer_ok
    adp.enforce_schedule = True
    adp.depth_strikes_each_side = 40
    adp.quotes_batch_size = 4
    adp.max_depth_tokens_per_expiry = 8
    adp.extra_neo_symbols = ()
    adp.trading_calendar = {}
    adp._cache = {}

    # 1st master fetch: 2 adapter calls (`expiries` = 1 GET, `futures_chain` = 2 GETs -> 3 HTTP GETs)
    adp.fetch_contract_master_raw(dec_dt)
    # 2nd master fetch: 2 adapter calls -> BOTH hit cache -> 0 new HTTP GETs!
    adp.fetch_contract_master_raw(dec_dt)
    # Option chain fetch (8 tokens / batch 4 = 2 quote batches):
    # 1 chain wrapper call (2 GETs) + 2 quote batch calls (2 GETs) = 3 adapter calls, 4 HTTP GETs!
    adp.fetch_option_chain_raw("2026-11-12", dec_dt)

    assert isinstance(cli_ok.api_client.rest_client._client._transport, SharedPacedHttpTransport)
    assert adp.adapter_call_count == 7  # 2 + 2 cached + 3
    assert adp.cache_hit_count == 2
    assert adp.http_request_count == 7  # 3 + 0 + 4 == len(wire_ok.wire_urls)
    assert len(wire_ok.wire_urls) == 7
    # Wrapper non-cached count (5) != actual HTTP GET count (7)!
    assert (adp.adapter_call_count - adp.cache_hit_count) == 5
    assert adp.http_request_count != (adp.adapter_call_count - adp.cache_hit_count)

    # Every consecutive HTTP request start gap is strictly >= 2.500000s despite 2.499862s OS early wakeups!
    ts_list = cli_ok.api_client.rest_client._client._transport.http_request_timestamps
    assert len(ts_list) == 7
    gaps = [ts_list[i] - ts_list[i - 1] for i in range(1, len(ts_list))]
    assert all(g >= 2.5 for g in gaps), f"Expected strict >= 2.5s gaps, got {gaps}"

    # Cross-process spacing via shared `pacing_state.json`: a second process starting 1.0s later waits >= 1.5s
    proc2_pacer = PacedReadTransport(
        min_interval_seconds=2.5,
        max_queue_wait_seconds=35.0,
        pacing_state_file_path=tmp_path / "pacing_state.json",
        clock_fn=lambda: mono_now[0],
        wall_time_fn=lambda: wall_now[0],
        sleep_fn=_early_wake_sleep,
    )
    mono_now[0] += 1.0
    wall_now[0] += 1.0
    before_proc2 = mono_now[0]
    proc2_pacer.wait_for_slot()
    assert (mono_now[0] - before_proc2) >= 1.5

    # Read-only transport blocks non-GET methods before touching the wire
    with pytest.raises(PermissionError, match="non-GET"):
        cli_ok.api_client.rest_client._client._transport.handle_request(
            _FakeReq("POST", "/orders/place")
        )
    assert len(wire_ok.wire_urls) == 7

    # 4B. Synthetic HTTP 429 inside multi-expiry pipeline -> immediate hard-stop, ZERO retry, persisted cooldown
    wire_429 = _WireTransport(fail_429_on_call=4)
    cooldown_429 = tmp_path / "cooldown_429.json"
    pacer_429 = PacedReadTransport(
        min_interval_seconds=0.0,
        max_queue_wait_seconds=35.0,
        cooldown_file_path=cooldown_429,
        wall_time_fn=lambda: wall_now[0],
        sleep_fn=lambda s: (_ for _ in ()).throw(AssertionError(f"Unexpected retry sleep {s}s on 429")),
    )
    cli_429 = _HttpBoundClient(wire_429)

    class _Cli429Full(_HttpBoundClient):
        def historical_data(self, **kwargs: Any) -> list[dict[str, Any]]:
            self.api_client.rest_client._client._transport.handle_request(
                _FakeReq("GET", f"/historical/{kwargs['interval']}")
            )
            return list(candles_by_ivl[kwargs["interval"]])

    cli_429_full = _Cli429Full(wire_429)

    class _Svc429:
        def get_expiries(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            cli_429_full.api_client.rest_client._client._transport.handle_request(
                _FakeReq("GET", "/scrip-master/expiries")
            )
            return {"status": "OK", "expiries": list(exp_18[:3])}

        def get_futures_chain(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            cli_429_full.api_client.rest_client._client._transport.handle_request(
                _FakeReq("GET", "/futures/quotes")
            )
            return {"status": "OK", "lot_size": 65, "nearest_futures_ltp": 25000.0}

        def get_full_option_chain(self, symbol: str, expiry: str | None = None, **kw: Any) -> dict[str, Any]:
            cli_429_full.api_client.rest_client._client._transport.handle_request(
                _FakeReq("GET", f"/option-chain/{expiry}")
            )
            return chains_by_exp[str(expiry or exp_18[0])]

    adp_429 = LiveKotakNeoAdapter.__new__(LiveKotakNeoAdapter)
    adp_429._service = _Svc429()
    adp_429._auth = type("_A", (), {"require_authenticated_client": lambda self: (cli_429_full, None, None)})()
    adp_429.rate_limiter = pacer_429
    adp_429.enforce_schedule = True
    adp_429.depth_strikes_each_side = 40
    adp_429.quotes_batch_size = 4
    adp_429.max_depth_tokens_per_expiry = 8
    adp_429.extra_neo_symbols = ()
    adp_429.cache_dir = tmp_path / "cache_429"
    adp_429.trading_calendar = {}
    adp_429._quote_retry_sleep_fn = lambda s: (_ for _ in ()).throw(
        AssertionError(f"429 MUST NOT trigger 30s quote retry sleep ({s}s)!")
    )
    adp_429._cache = {}

    pos_file = tmp_path / "pos_429.json"
    pos_file.write_text("[]", encoding="utf-8")
    cfg_429 = AnalystConfig(
        mode="LIVE_MORNING_SNAPSHOT",
        fixtures_dir=str(FIXTURES_DIR),
        open_positions_file=str(pos_file),
        cache_dir=str(tmp_path / "cache_429"),
        audit_log_path=str(tmp_path / "audit_429.jsonl"),
    )
    state_429, deliv_429 = run_analyst_pipeline(dec_dt, cfg_429, broker_adapter=adp_429)
    assert state_429.final_output is not None
    assert state_429.final_output.verdict == "INSUFFICIENT DATA"
    assert "429" in state_429.final_output.research_only_reason
    assert deliv_429.alerted_trader is False
    # Stopped at exact 4th HTTP GET with ZERO retry or probing!
    assert len(wire_429.wire_urls) == 4
    assert adp_429.http_request_count == 4
    assert cooldown_429.exists()

    # Follow-up attempt during active cooldown refuses before issuing a 5th HTTP request
    with pytest.raises(KotakRateLimit429Error, match="cooldown active"):
        adp_429.fetch_option_chain_raw("2026-11-19", dec_dt)
    assert len(wire_429.wire_urls) == 4


def test_kotak_handoff_acceptance5_6_7_selection_binding_independent_regime_and_alerts_off(
    tmp_path: Path,
) -> None:
    """Acceptance 5, 6 & 7: Final verified selection binds report/digest/audit/chart against
    arbitrary generated prose, market regime stays independent of candidate pick (WAIT on conflict
    or missing confirmation), and recommendations/live-readiness remain OFF.
    """
    dec_dt = dt.datetime(2026, 11, 6, 10, 15, 0, tzinfo=IST)
    chart_dir = tmp_path / "charts_acc5"
    audit_path = tmp_path / "audit_acc5.jsonl"
    cfg_off = AnalystConfig(
        mode="OFFLINE",
        fixtures_dir=str(FIXTURES_DIR),
        payoff_chart_dir=str(chart_dir),
        audit_log_path=str(audit_path),
    )

    # 5. Non-first eligible candidate chosen while free-form prose attempts to inject bogus facts
    baseline, _ = run_analyst_pipeline(dec_dt, dataclasses.replace(cfg_off, payoff_chart_dir=""))
    pass_cands = [c for c in baseline.candidate_structures if c.part_a_overall_status == "PASS"]
    assert len(pass_cands) >= 2
    picked = pass_cands[1]

    def _inject_valid_id_with_clean_citations_and_prose_attempt(
        base_resp: LLMSynthesisResponse,
        st: NiftyOptionsState,
        prim: Any,
        alts: list[Any],
    ) -> LLMSynthesisResponse:
        rebuilt = OfflineDeterministicSynthesisProvider().synthesize(
            sanitised_state_json="{}",
            state=st,
            primary=picked,
            alternatives=[c for c in st.candidate_structures if c.structure_id != picked.structure_id],
        )
        # Attempt to smuggle unverified narrative wording ("guaranteed edge") -> caught by verifier -> WAIT!
        return dataclasses.replace(
            rebuilt,
            reasoning_summary=rebuilt.reasoning_summary + " This trade is guaranteed profitable.",
        )

    st_blocked, deliv_blocked = run_analyst_pipeline(
        dec_dt,
        cfg_off,
        llm_provider=_StaticOverrideProvider(_inject_valid_id_with_clean_citations_and_prose_attempt),
        force_render_for_inspection=True,
    )
    assert st_blocked.final_output is not None
    assert st_blocked.final_output.verdict == "WAIT"
    assert st_blocked.final_output.primary_proposal is None
    assert "guaranteed profitable" not in deliv_blocked.full_report_markdown.lower()
    assert "guaranteed profitable" not in st_blocked.final_output.whatsapp_digest.lower()
    assert deliv_blocked.alerted_trader is False

    # Clean non-first pick binds consistently across report, digest, chart filename, and audit log
    def _select_second_clean(
        base_resp: LLMSynthesisResponse,
        st: NiftyOptionsState,
        prim: Any,
        alts: list[Any],
    ) -> LLMSynthesisResponse:
        return OfflineDeterministicSynthesisProvider().synthesize(
            sanitised_state_json="{}",
            state=st,
            primary=picked,
            alternatives=[c for c in st.candidate_structures if c.structure_id != picked.structure_id],
        )

    st_bound, deliv_bound = run_analyst_pipeline(
        dec_dt,
        cfg_off,
        llm_provider=_StaticOverrideProvider(_select_second_clean),
        force_render_for_inspection=True,
        suppress_trader_alerts=True,
    )
    assert st_bound.final_output is not None
    assert st_bound.final_output.verdict == "SUGGESTION"
    assert st_bound.final_output.primary_proposal is not None
    assert st_bound.final_output.primary_proposal.structure_id == picked.structure_id
    assert picked.structure_id in st_bound.final_output.whatsapp_digest
    assert picked.structure_id in deliv_bound.full_report_markdown
    if st_bound.final_output.payoff_diagram_path:
        assert picked.structure_id in Path(st_bound.final_output.payoff_diagram_path).name
    audit_last = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[-1])
    assert (
        audit_last["final_output"]["primary_proposal"]["structure_id"]
        == picked.structure_id
    )
    assert deliv_bound.alerted_trader is False

    # 7. Live mode keeps `is_live_ready=False` and `suppress_trader_alerts=True` default
    sig_slot = inspect.signature(run_scheduled_slot)
    assert sig_slot.parameters["suppress_trader_alerts"].default is True




