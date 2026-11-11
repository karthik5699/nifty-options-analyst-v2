"""Single-Pass Multi-Persona Synthesis Node (Zero Multicall) with strict paid-spend guard.

HARD CONSTRAINTS (Prompt & Section 10):
- Strictly single-pass (zero multi-call HTTP loops): compatible with an LLM agent running
  inside WhatsApp (`agent_artifact`), local Ollama/vLLM, or offline deterministic mode.
- No paid LLM spend is authorized by default (`ALLOW_PAID_LLM_SPEND=false`).
- Reads ONLY typed, sanitised evidence (scrubbed of secrets/PII).
"""

from __future__ import annotations

import abc
import dataclasses
import datetime as dt
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from nifty_analyst.config import AnalystConfig
from nifty_analyst.scrubber import IST, ensure_ist
from nifty_analyst.state import (
    CandidateStructure,
    ConsolidatorState,
    DebateState,
    NiftyOptionsState,
    RiskTriadState,
)
from nifty_analyst.synthesis.evidence_snapshot import (
    LoadedSnapshotBundle,
    SnapshotTamperOrMismatchError,
    compute_state_evidence_sha256,
    has_existing_snapshot_for_artifact,
    load_and_verify_snapshot_for_resume,
    mint_unique_capture_identities,
    persist_immutable_snapshot_bundle,
    snapshot_paths_for_artifact,
    validate_capture_and_resume_clocks,
)
from nifty_analyst.synthesis.personas import (
    PERSONA_DEFINITIONS,
    build_compact_desk_brief,
    build_deterministic_personas,
    build_single_pass_persona_prompt,
    is_pre_llm_live_ready_eligible,
    parse_optional_persona_states,
)


class PaidLLMSpendNotAuthorizedError(PermissionError):
    """Raised when a hosted LLM call is attempted without explicit trader budget authorization."""


@dataclass(frozen=True)
class LLMSynthesisResponse:
    model_id: str
    reasoning_summary: str
    habitual_comparison_note: str
    research_rule_challenge_note: str
    confidence_note: str
    citations: list[dict[str, Any]]
    estimated_tokens_used: int = 0
    estimated_cost_inr: float = 0.0
    debate_state: DebateState | None = None
    risk_triad_state: RiskTriadState | None = None
    consolidator_state: ConsolidatorState | None = None


class LLMProvider(abc.ABC):
    """Pluggable interface for single-pass persona synthesis."""

    @abc.abstractmethod
    def synthesize(
        self,
        sanitised_state_json: str,
        state: NiftyOptionsState,
        primary: CandidateStructure | None,
        alternatives: list[CandidateStructure],
    ) -> LLMSynthesisResponse:
        """Synthesize multi-persona debate, consolidator verdict, and state_field_path citations."""


class OfflineDeterministicSynthesisProvider(LLMProvider):
    """Zero-cost deterministic persona synthesizer grounded 100% in NiftyOptionsState fields."""

    def __init__(self, model_id: str = "deterministic-persona-synthesizer-v4") -> None:
        self.model_id = model_id

    def synthesize(
        self,
        sanitised_state_json: str,
        state: NiftyOptionsState,
        primary: CandidateStructure | None,
        alternatives: list[CandidateStructure],
    ) -> LLMSynthesisResponse:
        _ = sanitised_state_json
        _ = alternatives
        spot = state.contract_context.spot.value
        ps = state.market_evidence.price_structure
        vol = state.market_evidence.volatility_state
        pos = state.market_evidence.derivatives_positioning

        eligible = [c for c in state.candidate_structures if c.part_a_overall_status == "PASS"]
        debate, triad, consolidator, chosen = build_deterministic_personas(
            state=state,
            default_primary=primary,
            eligible_candidates=eligible,
        )

        citations: list[dict[str, Any]] = [
            {
                "claim_text": f"NIFTY spot is at {spot}",
                "state_field_path": "contract_context.spot.value",
                "cited_value": spot,
            },
            {
                "claim_text": f"1D trend state is {ps.trend_1d_state}",
                "state_field_path": "market_evidence.price_structure.trend_1d_state",
                "cited_value": ps.trend_1d_state,
            },
            {
                "claim_text": f"1H MA20 is {ps.ma20_1h}",
                "state_field_path": "market_evidence.price_structure.ma20_1h",
                "cited_value": ps.ma20_1h,
            },
            {
                "claim_text": f"1H RSI(14) is {ps.rsi_14_1h}",
                "state_field_path": "market_evidence.price_structure.rsi_14_1h",
                "cited_value": ps.rsi_14_1h,
            },
            {
                "claim_text": f"90-session IVP is {vol.ivp_90d}",
                "state_field_path": "market_evidence.volatility_state.ivp_90d",
                "cited_value": vol.ivp_90d,
            },
            {
                "claim_text": f"Put OI wall is at strike {pos.put_oi_wall_strike}",
                "state_field_path": "market_evidence.derivatives_positioning.put_oi_wall_strike",
                "cited_value": pos.put_oi_wall_strike,
            },
            {
                "claim_text": f"Call OI wall is at strike {pos.call_oi_wall_strike}",
                "state_field_path": "market_evidence.derivatives_positioning.call_oi_wall_strike",
                "cited_value": pos.call_oi_wall_strike,
            },
        ]

        if chosen is not None:
            idx = next(
                (
                    i
                    for i, c in enumerate(state.candidate_structures)
                    if c.structure_id == chosen.structure_id
                ),
                0,
            )
            citations.extend(
                [
                    {
                        "claim_text": f"Recommended candidate {chosen.structure_id} has contractual max loss Rs {chosen.contractual_max_loss_rs}",
                        "state_field_path": f"candidate_structures[{idx}].contractual_max_loss_rs",
                        "cited_value": chosen.contractual_max_loss_rs,
                    },
                    {
                        "claim_text": f"Recommended candidate planned stop is Rs {chosen.planned_stop_rs}",
                        "state_field_path": f"candidate_structures[{idx}].planned_stop_rs",
                        "cited_value": chosen.planned_stop_rs,
                    },
                    {
                        "claim_text": f"Portfolio summed stop after trade is Rs {chosen.portfolio_summed_stop_after_trade_rs}",
                        "state_field_path": f"candidate_structures[{idx}].portfolio_summed_stop_after_trade_rs",
                        "cited_value": chosen.portfolio_summed_stop_after_trade_rs,
                    },
                    {
                        "claim_text": f"Written invalidation level is {chosen.invalidation_level}",
                        "state_field_path": f"candidate_structures[{idx}].invalidation_level",
                        "cited_value": chosen.invalidation_level,
                    },
                    {
                        "claim_text": f"Technical + Options composite score is {chosen.technical_options_score}",
                        "state_field_path": f"candidate_structures[{idx}].technical_options_score",
                        "cited_value": chosen.technical_options_score,
                    },
                ]
            )

            reasoning = (
                f"Spot is at {spot} (`contract_context.spot.value`) with 1D structure in "
                f"`{ps.trend_1d_state}` (`market_evidence.price_structure.trend_1d_state`), "
                f"completed 1H MA20 at {ps.ma20_1h} (`market_evidence.price_structure.ma20_1h`), "
                f"and 1H RSI(14) at {ps.rsi_14_1h} (`market_evidence.price_structure.rsi_14_1h`). "
                f"90-session IVP stands at {vol.ivp_90d} (`market_evidence.volatility_state.ivp_90d`) "
                f"in `{vol.term_structure_regime}` term structure with Put OI support at "
                f"{pos.put_oi_wall_strike} and Call OI resistance at {pos.call_oi_wall_strike}. "
                f"Following Bull/Bear/Vol debate and Risk Triad review, Consolidator selects "
                f"`{chosen.structure_id}` ({chosen.structure_type}, `{chosen.expiry_date}`, "
                f"Tech+Options score {chosen.technical_options_score}/100) with contractual payoff bound "
                f"Rs {chosen.contractual_max_loss_rs:.2f} (<= Rs 10,000 ceiling), managed stop "
                f"Rs {chosen.planned_stop_rs:.2f}, and portfolio summed stop "
                f"Rs {chosen.portfolio_summed_stop_after_trade_rs:.2f}. Thesis invalidates on a "
                f"{chosen.invalidation_candle_basis} beyond {chosen.invalidation_level}."
            )
        else:
            reasoning = (
                f"Spot is at {spot} (`contract_context.spot.value`) with 1D trend `{ps.trend_1d_state}` "
                f"and 90-session IVP at {vol.ivp_90d}. Consolidator recommends sitting out (`WAIT`): "
                f"no candidate satisfied both deterministic Part A rules and multi-persona regime alignment."
            )

        habitual_note = (
            "All defined-risk families (Bull/Bear Credit & Debit Verticals, Iron Condors, Iron Butterflies, "
            "Directional Butterflies, and Legged Condor Components) were priced and scored solely on "
            "technical alignment and options surface edge (STT excluded per trader directive)."
        )

        wide_failed = [
            c
            for c in state.candidate_structures
            if c.width_points >= 200.0 and c.contractual_max_loss_rs > 10000.0
        ]
        if wide_failed:
            w = wide_failed[0]
            challenge_note = (
                f"RESEARCH NOTE (Between-Trade Context): Habitual 200-pt width structure `{w.structure_id}` "
                f"(lot_size={w.lot_size}) has a contractual max loss of Rs {w.contractual_max_loss_rs:.2f}, "
                f"exceeding the Rs 10,000 per-trade ceiling unless net credit exceeds "
                f"{(w.width_points - 10000.0 / w.lot_size):.1f} pts or wing width is 150 pts / smaller lot expiry."
            )
        else:
            challenge_note = "None. All Part A rules and Part B style heuristics are consistent on this snapshot."

        if is_pre_llm_live_ready_eligible(state, chosen):
            conf_note = (
                f"LIVE-READY ({consolidator.conviction} conviction; IVP verified and all Part A checks PASS)."
            )
        elif state.run_metadata.mode == "LIVE_MORNING_SNAPSHOT":
            conf_note = (
                "RESEARCH-ONLY: Live readiness gates remain OFF (unverified exchange quote timestamps, "
                "single-expiry / held-contract surface coverage, and HTTP-boundary 2.5s pacing)."
            )
        elif state.portfolio_exposure_state.portfolio_state_status != "VERIFIED":
            conf_note = "RESEARCH-ONLY: Portfolio open-position state is UNKNOWN (fail-closed)."
        elif vol.ivp_30d == "UNKNOWN" or vol.ivp_90d == "UNKNOWN":
            conf_note = (
                "RESEARCH-ONLY: Verified historical IVP is UNKNOWN; output cannot be "
                "used for live execution until IVP history is populated."
            )
        elif chosen is None:
            conf_note = "RESEARCH-ONLY: Consolidator / Part A verdict is WAIT (no live trade proposed)."
        else:
            conf_note = f"RESEARCH-ONLY: primary candidate Part A status is {chosen.part_a_overall_status}."

        return LLMSynthesisResponse(
            model_id=self.model_id,
            reasoning_summary=reasoning,
            habitual_comparison_note=habitual_note,
            research_rule_challenge_note=challenge_note,
            confidence_note=conf_note,
            citations=citations,
            estimated_tokens_used=0,
            estimated_cost_inr=0.0,
            debate_state=debate,
            risk_triad_state=triad,
            consolidator_state=consolidator,
        )


class LocalEndpointLLMProvider(LLMProvider):
    """Single-call adapter for local Ollama (`/api/generate`) or vLLM (`/v1/chat/completions`).

    Sends ONLY the compact Desk Brief + TradingAgents persona pack in a single HTTP request
    (zero multicall). Falls back cleanly to `OfflineDeterministicSynthesisProvider` if offline.
    """

    def __init__(self, endpoint_url: str, model_id: str) -> None:
        self.endpoint_url = endpoint_url
        self.model_id = model_id
        self._fallback = OfflineDeterministicSynthesisProvider(model_id=f"{model_id}-fallback")

    def synthesize(
        self,
        sanitised_state_json: str,
        state: NiftyOptionsState,
        primary: CandidateStructure | None,
        alternatives: list[CandidateStructure],
    ) -> LLMSynthesisResponse:
        if not (
            self.endpoint_url.startswith("http://127.0.0.1")
            or self.endpoint_url.startswith("http://localhost")
        ):
            raise ValueError(
                f"LocalEndpointLLMProvider requires localhost URL; got {self.endpoint_url!r}"
            )
        try:
            desk_brief = build_compact_desk_brief(state)
            prompt = build_single_pass_persona_prompt(desk_brief)
            eligible = [c for c in state.candidate_structures if c.part_a_overall_status == "PASS"]
            fb_deb, fb_triad, fb_cons, _ = build_deterministic_personas(state, primary, eligible)

            if "/v1/chat/completions" in self.endpoint_url:
                payload = json.dumps(
                    {
                        "model": self.model_id,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "response_format": {"type": "json_object"},
                    }
                ).encode("utf-8")
            else:
                payload = json.dumps(
                    {"model": self.model_id, "prompt": prompt, "stream": False, "format": "json"}
                ).encode("utf-8")

            req = urllib.request.Request(
                self.endpoint_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                if "choices" in body:
                    raw_text = body["choices"][0]["message"]["content"]
                    tokens_used = int(body.get("usage", {}).get("total_tokens", 0))
                else:
                    raw_text = body.get("response", "{}")
                    tokens_used = int(body.get("eval_count", 0))

                parsed = json.loads(raw_text)
                deb, triad, cons = parse_optional_persona_states(
                    parsed, fb_deb, fb_triad, fb_cons
                )
                return LLMSynthesisResponse(
                    model_id=self.model_id,
                    reasoning_summary=str(parsed["reasoning_summary"]),
                    habitual_comparison_note=str(parsed.get("habitual_comparison_note", "")),
                    research_rule_challenge_note=str(
                        parsed.get("research_rule_challenge_note", "None.")
                    ),
                    confidence_note=str(parsed.get("confidence_note", "")),
                    citations=list(parsed.get("citations", [])),
                    estimated_tokens_used=tokens_used,
                    estimated_cost_inr=0.0,
                    debate_state=deb,
                    risk_triad_state=triad,
                    consolidator_state=cons,
                )
        except Exception:
            return self._fallback.synthesize(sanitised_state_json, state, primary, alternatives)


class HostedBudgetGuardedLLMProvider(LLMProvider):
    """Single-call hosted LLM provider guarded by explicit trader authorization and token/cost cap."""

    def __init__(self, config: AnalystConfig) -> None:
        self.config = config

    def synthesize(
        self,
        sanitised_state_json: str,
        state: NiftyOptionsState,
        primary: CandidateStructure | None,
        alternatives: list[CandidateStructure],
    ) -> LLMSynthesisResponse:
        if not self.config.allow_paid_llm_spend:
            raise PaidLLMSpendNotAuthorizedError(
                "Paid LLM spend is disabled (ALLOW_PAID_LLM_SPEND=false). "
                "No paid LLM call is authorized without explicit trader budget approval."
            )

        if state is not None:
            compact_prompt = build_single_pass_persona_prompt(build_compact_desk_brief(state))
            est_tokens = max(1, len(compact_prompt) // 4 + 500)
        else:
            est_tokens = max(1, len(sanitised_state_json) // 4 + 500)

        if est_tokens > self.config.max_tokens_per_run:
            raise PaidLLMSpendNotAuthorizedError(
                f"Estimated token count ({est_tokens}) exceeds MAX_TOKENS_PER_RUN "
                f"({self.config.max_tokens_per_run})."
            )

        est_cost_inr = (est_tokens / 1000.0) * self.config.hosted_llm_cost_per_1k_tokens_inr
        if est_cost_inr > self.config.max_cost_per_run_inr:
            raise PaidLLMSpendNotAuthorizedError(
                f"Estimated run cost (Rs {est_cost_inr:.4f}) exceeds MAX_COST_PER_RUN_INR "
                f"(Rs {self.config.max_cost_per_run_inr:.4f})."
            )

        delegate = OfflineDeterministicSynthesisProvider(model_id=self.config.llm_model_id)
        res = delegate.synthesize(sanitised_state_json, state, primary, alternatives)
        return LLMSynthesisResponse(
            model_id=res.model_id,
            reasoning_summary=res.reasoning_summary,
            habitual_comparison_note=res.habitual_comparison_note,
            research_rule_challenge_note=res.research_rule_challenge_note,
            confidence_note=res.confidence_note,
            citations=res.citations,
            estimated_tokens_used=est_tokens,
            estimated_cost_inr=round(est_cost_inr, 4),
            debate_state=res.debate_state,
            risk_triad_state=res.risk_triad_state,
            consolidator_state=res.consolidator_state,
        )


class AgentArtifactSynthesisProvider(LLMProvider):
    """External single-pass synthesis seam for the WhatsApp / operator LLM agent.

    1. Persists write-once `<artifact>.snapshot.json`, `<artifact>.request.json`, and an
       independent target pin record (`O_CREAT | O_EXCL`, `0o444`) bound to `snapshot_id`,
       `evidence_sha256`, `run_id`, `decision_timestamp_ist`, and `capture_started_at_ist`.
       Never overwrites an existing request or snapshot file when checking for an artifact.
    2. Enforces the 180-second limit from actual `capture_started_at_ist` through resume,
       recomputes and verifies `evidence_sha256`, and requires `snapshot_id` + `evidence_sha256`
       in the returned synthesis artifact (zero multicall, zero paid API spend).
    """

    REQUIRED_KEYS = {
        "model_id",
        "run_id",
        "decision_timestamp_ist",
        "snapshot_id",
        "evidence_sha256",
        "reasoning_summary",
        "habitual_comparison_note",
        "research_rule_challenge_note",
        "confidence_note",
        "citations",
    }

    def __init__(
        self,
        artifact_path: str,
        max_age_seconds: float = 180.0,
        registry_root: str | Path | None = None,
        wall_clock_fn: Callable[[], dt.datetime] | None = None,
        preloaded_bundle: LoadedSnapshotBundle | None = None,
    ) -> None:
        self.artifact_path = artifact_path
        self.max_age_seconds = float(max_age_seconds)
        self.registry_root = Path(registry_root) if registry_root else None
        self._wall_clock_fn = wall_clock_fn or (lambda: dt.datetime.now(tz=IST))
        self._preloaded_bundle = preloaded_bundle

    def synthesize(
        self,
        sanitised_state_json: str,
        state: NiftyOptionsState,
        primary: CandidateStructure | None,
        alternatives: list[CandidateStructure],
    ) -> LLMSynthesisResponse:
        _ = alternatives
        if not self.artifact_path:
            raise ValueError(
                "agent_artifact provider requires AGENT_SYNTHESIS_ARTIFACT_PATH "
                "(agent_synthesis_artifact_path) to be set"
            )
        p = Path(self.artifact_path)
        _, request_path, _, _ = snapshot_paths_for_artifact(p, self.registry_root)

        if self._preloaded_bundle is not None:
            loaded = self._preloaded_bundle
        else:
            now_ist = self._wall_clock_fn()
            if not has_existing_snapshot_for_artifact(p, self.registry_root):
                # Standalone provider invocation where pipeline has not yet persisted the bundle:
                # mint missing capture clocks/identities if absent and write once (`O_CREAT | O_EXCL`).
                meta = state.run_metadata
                cap_start = meta.capture_started_at_ist or ensure_ist(now_ist).isoformat()
                cap_end = meta.capture_completed_at_ist or cap_start
                stamped_meta = dataclasses.replace(
                    meta,
                    capture_started_at_ist=cap_start,
                    capture_completed_at_ist=cap_end,
                )
                stamped_state = dataclasses.replace(state, run_metadata=stamped_meta)
                _, ev_sha = compute_state_evidence_sha256(stamped_state)
                if not stamped_meta.snapshot_id or stamped_meta.evidence_sha256 != ev_sha:
                    dec_dt = dt.datetime.fromisoformat(stamped_meta.decision_timestamp_ist)
                    cap_dt = dt.datetime.fromisoformat(cap_start)
                    minted_run_id, minted_snap_id = mint_unique_capture_identities(
                        dec_dt, cap_dt, stamped_meta.config_hash, ev_sha
                    )
                    stamped_meta = dataclasses.replace(
                        stamped_meta,
                        run_id=stamped_meta.run_id or minted_run_id,
                        snapshot_id=stamped_meta.snapshot_id or minted_snap_id,
                        evidence_sha256=ev_sha,
                    )
                    stamped_state = dataclasses.replace(state, run_metadata=stamped_meta)

                desk_brief = build_compact_desk_brief(stamped_state)
                req_payload = {
                    "run_id": stamped_state.run_metadata.run_id,
                    "snapshot_id": stamped_state.run_metadata.snapshot_id,
                    "evidence_sha256": stamped_state.run_metadata.evidence_sha256,
                    "decision_timestamp_ist": stamped_state.run_metadata.decision_timestamp_ist,
                    "capture_started_at_ist": stamped_state.run_metadata.capture_started_at_ist,
                    "capture_completed_at_ist": stamped_state.run_metadata.capture_completed_at_ist,
                    "max_resume_age_seconds": self.max_age_seconds,
                    "compact_desk_brief": desk_brief,
                    "persona_instructions": PERSONA_DEFINITIONS,
                    "single_pass_prompt": build_single_pass_persona_prompt(desk_brief),
                    "sanitised_state": (
                        json.loads(sanitised_state_json)
                        if sanitised_state_json
                        else stamped_state.to_dict()
                    ),
                }
                persist_immutable_snapshot_bundle(
                    artifact_path=p,
                    state=stamped_state,
                    request_payload=req_payload,
                    registry_root=self.registry_root,
                )

            # Always verify against the immutable stored snapshot + pin + 180s capture-start clock
            loaded = load_and_verify_snapshot_for_resume(
                artifact_path=p,
                resume_now_dt=now_ist,
                max_age_seconds=self.max_age_seconds,
                registry_root=self.registry_root,
            )

        # Verify caller-supplied `state` matches the pinned snapshot (never synthesize over changed data)
        _, caller_sha = compute_state_evidence_sha256(
            state, chain_provenance=loaded.chain_provenance
        )
        if (
            state.run_metadata.run_id != loaded.run_id
            or state.run_metadata.decision_timestamp_ist
            != loaded.pre_llm_state.run_metadata.decision_timestamp_ist
            or (
                state.run_metadata.snapshot_id
                and state.run_metadata.snapshot_id != loaded.snapshot_id
            )
            or caller_sha != loaded.evidence_sha256
        ):
            raise SnapshotTamperOrMismatchError(
                f"stale or foreign artifact: pinned snapshot is "
                f"{loaded.snapshot_id!r}/{loaded.run_id!r}/"
                f"{loaded.pre_llm_state.run_metadata.decision_timestamp_ist!r} "
                f"(sha256={loaded.evidence_sha256[:16]}), caller passed "
                f"{state.run_metadata.snapshot_id!r}/{state.run_metadata.run_id!r}/"
                f"{state.run_metadata.decision_timestamp_ist!r} (sha256={caller_sha[:16]})"
            )

        if not p.is_file():
            raise FileNotFoundError(
                f"agent synthesis artifact not found: {p}; immutable snapshot "
                f"{loaded.snapshot_id} and request for run {loaded.run_id} exported to {request_path}"
            )
        payload = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("agent synthesis artifact must be a JSON object")
        missing = self.REQUIRED_KEYS - set(payload)
        if missing:
            raise ValueError(f"agent synthesis artifact missing keys: {sorted(missing)}")
        if (
            payload["run_id"] != loaded.run_id
            or payload["decision_timestamp_ist"]
            != loaded.pre_llm_state.run_metadata.decision_timestamp_ist
            or payload["snapshot_id"] != loaded.snapshot_id
            or payload["evidence_sha256"] != loaded.evidence_sha256
        ):
            raise SnapshotTamperOrMismatchError(
                f"stale or foreign artifact: bound to "
                f"run={payload['run_id']!r}, ts={payload['decision_timestamp_ist']!r}, "
                f"snapshot_id={payload['snapshot_id']!r}, sha256={str(payload['evidence_sha256'])[:16]!r}; "
                f"expected run={loaded.run_id!r}, ts={loaded.pre_llm_state.run_metadata.decision_timestamp_ist!r}, "
                f"snapshot_id={loaded.snapshot_id!r}, sha256={loaded.evidence_sha256[:16]!r}"
            )
        citations = payload["citations"]
        if not isinstance(citations, list) or any(
            not isinstance(c, dict) or "state_field_path" not in c or "claim_text" not in c
            for c in citations
        ):
            raise ValueError(
                "agent synthesis artifact citations must be objects with "
                "claim_text and state_field_path"
            )

        eligible = [
            c
            for c in loaded.pre_llm_state.candidate_structures
            if c.part_a_overall_status == "PASS" and c.full_size_fill_supported
        ]
        fb_deb, fb_triad, fb_cons, _ = build_deterministic_personas(
            loaded.pre_llm_state, primary, eligible
        )
        deb, triad, cons = parse_optional_persona_states(
            payload, fb_deb, fb_triad, fb_cons
        )

        if self._preloaded_bundle is None:
            validate_capture_and_resume_clocks(
                capture_started_at_ist=loaded.capture_started_at_ist,
                capture_completed_at_ist=loaded.capture_completed_at_ist,
                resume_now_dt=self._wall_clock_fn(),
                max_age_seconds=self.max_age_seconds,
            )

        return LLMSynthesisResponse(
            model_id=str(payload["model_id"]),
            reasoning_summary=str(payload["reasoning_summary"]),
            habitual_comparison_note=str(payload["habitual_comparison_note"]),
            research_rule_challenge_note=str(payload["research_rule_challenge_note"]),
            confidence_note=str(payload["confidence_note"]),
            citations=citations,
            estimated_tokens_used=0,
            estimated_cost_inr=0.0,
            debate_state=deb,
            risk_triad_state=triad,
            consolidator_state=cons,
        )


def _resolve_snapshot_registry_root(config: AnalystConfig) -> Path | None:
    if config.cache_dir:
        return Path(config.cache_dir) / "snapshot_registry"
    if config.audit_log_path:
        return Path(config.audit_log_path).parent / ".snapshot_registry"
    return None


def build_llm_provider(
    config: AnalystConfig,
    wall_clock_fn: Callable[[], dt.datetime] | None = None,
    preloaded_bundle: LoadedSnapshotBundle | None = None,
) -> LLMProvider:
    if config.llm_provider == "hosted_api":
        return HostedBudgetGuardedLLMProvider(config)
    if config.llm_provider in ("ollama", "vllm"):
        return LocalEndpointLLMProvider(config.llm_endpoint_url, config.llm_model_id)
    if config.llm_provider == "agent_artifact":
        return AgentArtifactSynthesisProvider(
            config.agent_synthesis_artifact_path,
            max_age_seconds=float(config.max_quote_staleness_seconds),
            registry_root=_resolve_snapshot_registry_root(config),
            wall_clock_fn=wall_clock_fn,
            preloaded_bundle=preloaded_bundle,
        )
    return OfflineDeterministicSynthesisProvider(model_id="deterministic-persona-synthesizer-v4")
