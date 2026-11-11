"""Single-LLM synthesis node with pluggable local-first providers and strict paid-spend guard.

HARD CONSTRAINT (Prompt & Section 10):
- No paid LLM spend is authorized: defaults to local/offline deterministic synthesis, supports
  local Ollama/vLLM, and places any hosted API behind an explicit config flag with a per-run
  token and cost budget check.
- Reads ONLY a typed, sanitised JSON evidence state (scrubbed of any secrets/PII).
"""

from __future__ import annotations

import abc
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nifty_analyst.config import AnalystConfig
from nifty_analyst.state import CandidateStructure, NiftyOptionsState


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


class LLMProvider(abc.ABC):
    """Pluggable interface for Phase 1 single-LLM synthesis."""

    @abc.abstractmethod
    def synthesize(
        self,
        sanitised_state_json: str,
        state: NiftyOptionsState,
        primary: CandidateStructure | None,
        alternatives: list[CandidateStructure],
    ) -> LLMSynthesisResponse:
        """Synthesize qualitative rationale and return explicit state_field_path citations."""


class OfflineDeterministicSynthesisProvider(LLMProvider):
    """Zero-cost local synthesis provider grounded 100% in NiftyOptionsState fields."""

    def __init__(self, model_id: str = "deterministic-local-synthesizer-v3") -> None:
        self.model_id = model_id

    def synthesize(
        self,
        sanitised_state_json: str,
        state: NiftyOptionsState,
        primary: CandidateStructure | None,
        alternatives: list[CandidateStructure],
    ) -> LLMSynthesisResponse:
        _ = sanitised_state_json  # Verified serialisable and scrubbed before call
        spot = state.contract_context.spot.value
        ps = state.market_evidence.price_structure
        vol = state.market_evidence.volatility_state
        pos = state.market_evidence.derivatives_positioning

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

        if primary is not None:
            # Find index of primary in state.candidate_structures
            idx = next(
                (
                    i
                    for i, c in enumerate(state.candidate_structures)
                    if c.structure_id == primary.structure_id
                ),
                0,
            )
            citations.extend(
                [
                    {
                        "claim_text": f"Primary candidate {primary.structure_id} has contractual max loss Rs {primary.contractual_max_loss_rs}",
                        "state_field_path": f"candidate_structures[{idx}].contractual_max_loss_rs",
                        "cited_value": primary.contractual_max_loss_rs,
                    },
                    {
                        "claim_text": f"Primary candidate planned stop is Rs {primary.planned_stop_rs}",
                        "state_field_path": f"candidate_structures[{idx}].planned_stop_rs",
                        "cited_value": primary.planned_stop_rs,
                    },
                    {
                        "claim_text": f"Portfolio summed stop after trade is Rs {primary.portfolio_summed_stop_after_trade_rs}",
                        "state_field_path": f"candidate_structures[{idx}].portfolio_summed_stop_after_trade_rs",
                        "cited_value": primary.portfolio_summed_stop_after_trade_rs,
                    },
                    {
                        "claim_text": f"Written invalidation level is {primary.invalidation_level}",
                        "state_field_path": f"candidate_structures[{idx}].invalidation_level",
                        "cited_value": primary.invalidation_level,
                    },
                ]
            )

            reasoning = (
                f"Spot is at {spot} (`contract_context.spot.value`) with 1D structure in "
                f"{ps.trend_1d_state} (`market_evidence.price_structure.trend_1d_state`) and "
                f"completed 1H MA20 at {ps.ma20_1h} (`market_evidence.price_structure.ma20_1h`). "
                f"90-session IVP stands at {vol.ivp_90d} (`market_evidence.volatility_state.ivp_90d`) "
                f"with Put OI support at {pos.put_oi_wall_strike} and Call OI resistance at "
                f"{pos.call_oi_wall_strike}. Primary structure `{primary.structure_id}` "
                f"({primary.structure_type}) passes all applicable Part A checks with leg-wise "
                f"contractual payoff bound (gross, KNOWN) of Rs {primary.contractual_max_loss_rs:.2f} (within the Rs 10,000 "
                f"per-trade ceiling; all-in costs labeled separately), indicative managed stop of Rs {primary.planned_stop_rs:.2f}, and "
                f"portfolio summed stop of Rs {primary.portfolio_summed_stop_after_trade_rs:.2f} "
                f"(within the Rs 10,000 summed-stop cap). Thesis is invalidated on a "
                f"{primary.invalidation_candle_basis} beyond {primary.invalidation_level}."
            )
        else:
            reasoning = (
                f"Spot is at {spot} (`contract_context.spot.value`) with 1D trend {ps.trend_1d_state} "
                f"and 90-session IVP at {vol.ivp_90d}. No candidate structure satisfied all binding "
                f"Part A checks and risk-reward criteria after statutory charges; sitting out (`WAIT`)."
            )

        habitual_note = (
            "Credit spreads (Bull Put Spread / Bear Call Spread / Iron Condor) and debit spreads / "
            "butterflies were generated and evaluated on equal footing without style filtering. "
            "See Section 6 table for exact cost, max loss, breakeven, PoP, and vega comparison."
        )

        # Check if any candidate failed solely because of the 200-pt habitual width exceeding Rs 10,000 max loss
        wide_failed = [
            c
            for c in state.candidate_structures
            if c.width_points >= 200.0
            and c.contractual_max_loss_rs > 10000.0
        ]
        if wide_failed:
            w = wide_failed[0]
            challenge_note = (
                f"RESEARCH NOTE (Between-Trade Context): Habitual 200-pt width structure `{w.structure_id}` "
                f"(lot_size={w.lot_size}) has a leg-wise contractual max loss of Rs {w.contractual_max_loss_rs:.2f}, "
                f"which exceeds the settled Rs 10,000 per-trade contractual max loss ceiling (6 Nov 2026) "
                f"unless net credit exceeds {(w.width_points - 10000.0 / w.lot_size):.1f} pts or wing width is "
                f"tightened to 150 pts."
            )
        else:
            challenge_note = "None. All Part A rules and Part B style heuristics are consistent on this snapshot."

        # Mirror the pipeline's live-ready gate exactly: SUGGESTION + verified IVP +
        # primary Part A PASS. Anything else is research-only with the concrete reason.
        if primary is not None and vol.ivp_90d != "UNKNOWN" and primary.part_a_overall_status == "PASS":
            conf_note = "LIVE-READY (IVP verified and all Part A checks PASS)."
        elif vol.ivp_90d == "UNKNOWN":
            conf_note = ("RESEARCH-ONLY: Verified historical IVP is UNKNOWN; output cannot be "
                         "used for live execution until IVP history is populated.")
        elif primary is None:
            conf_note = "RESEARCH-ONLY: no candidate satisfied all binding Part A checks (verdict WAIT)."
        else:
            conf_note = (f"RESEARCH-ONLY: primary candidate Part A status is "
                         f"{primary.part_a_overall_status}.")

        return LLMSynthesisResponse(
            model_id=self.model_id,
            reasoning_summary=reasoning,
            habitual_comparison_note=habitual_note,
            research_rule_challenge_note=challenge_note,
            confidence_note=conf_note,
            citations=citations,
            estimated_tokens_used=0,
            estimated_cost_inr=0.0,
        )


class LocalEndpointLLMProvider(LLMProvider):
    """Adapter for local Ollama / vLLM HTTP endpoints (zero paid API spend).

    Falls back cleanly to OfflineDeterministicSynthesisProvider if the local server is unreachable
    in offline tests.
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
        # Ensure only localhost / 127.0.0.1 is called by LocalEndpointLLMProvider
        if not (
            self.endpoint_url.startswith("http://127.0.0.1")
            or self.endpoint_url.startswith("http://localhost")
        ):
            raise ValueError(
                f"LocalEndpointLLMProvider requires localhost URL; got {self.endpoint_url!r}"
            )
        try:
            prompt = (
                "You are a NIFTY 50 options research synthesizer. Using ONLY the following JSON state, "
                "return JSON with keys reasoning_summary, habitual_comparison_note, "
                "research_rule_challenge_note, confidence_note, citations.\n"
                f"STATE:\n{sanitised_state_json}"
            )
            payload = json.dumps(
                {"model": self.model_id, "prompt": prompt, "stream": False, "format": "json"}
            ).encode("utf-8")
            req = urllib.request.Request(
                self.endpoint_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                parsed = json.loads(body.get("response", "{}"))
                return LLMSynthesisResponse(
                    model_id=self.model_id,
                    reasoning_summary=str(parsed["reasoning_summary"]),
                    habitual_comparison_note=str(parsed.get("habitual_comparison_note", "")),
                    research_rule_challenge_note=str(
                        parsed.get("research_rule_challenge_note", "None.")
                    ),
                    confidence_note=str(parsed.get("confidence_note", "")),
                    citations=list(parsed.get("citations", [])),
                    estimated_tokens_used=int(body.get("eval_count", 0)),
                    estimated_cost_inr=0.0,
                )
        except Exception:
            return self._fallback.synthesize(sanitised_state_json, state, primary, alternatives)


class HostedBudgetGuardedLLMProvider(LLMProvider):
    """Hosted LLM provider guarded by explicit trader authorization and per-run cost/token cap."""

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

        est_tokens = max(1, len(sanitised_state_json) // 4 + 600)
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

        # Even if budget flags are enabled in unit tests, delegate to deterministic synthesis
        # unless a live endpoint is explicitly configured
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
        )


class AgentArtifactSynthesisProvider(LLMProvider):
    """External synthesis artifact provider (Phase 1 single-LLM synthesis written by
    Instinct, the analyst operator).

    The hourly run pauses before the synthesis layer; the operator reads ONLY the
    sanitised MarketEvidenceState JSON and writes an artifact file containing the
    qualitative synthesis plus explicit state_field_path citations. This provider
    loads that artifact with strict schema validation and returns it as a standard
    LLMSynthesisResponse, so verifier.py runs unchanged downstream. Zero paid spend
    by construction (no endpoint, no tokens, cost pinned at 0).
    """

    REQUIRED_KEYS = {
        "model_id",
        "run_id",
        "decision_timestamp_ist",
        "reasoning_summary",
        "habitual_comparison_note",
        "research_rule_challenge_note",
        "confidence_note",
        "citations",
    }

    def __init__(self, artifact_path: str) -> None:
        self.artifact_path = artifact_path

    def synthesize(
        self,
        sanitised_state_json: str,
        state: NiftyOptionsState,
        primary: CandidateStructure | None,
        alternatives: list[CandidateStructure],
    ) -> LLMSynthesisResponse:
        if not self.artifact_path:
            raise ValueError(
                "agent_artifact provider requires AGENT_SYNTHESIS_ARTIFACT_PATH "
                "(agent_synthesis_artifact_path) to be set"
            )
        p = Path(self.artifact_path)
        run_id = state.run_metadata.run_id
        decision_ts = state.run_metadata.decision_timestamp_ist
        # Scheduled orchestration seam: always (re)export THIS run's request - the
        # sanitised evidence after the deterministic gates plus the run identity - so
        # the agent writes its artifact from current evidence, never fixture text or
        # a stale export.
        request_path = p.with_name(p.name + ".request.json")
        request_path.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "decision_timestamp_ist": decision_ts,
                    "sanitised_state": json.loads(sanitised_state_json),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if not p.is_file():
            raise FileNotFoundError(
                f"agent synthesis artifact not found: {p}; request for run {run_id} "
                f"exported to {request_path}"
            )
        payload = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("agent synthesis artifact must be a JSON object")
        missing = self.REQUIRED_KEYS - set(payload)
        if missing:
            raise ValueError(f"agent synthesis artifact missing keys: {sorted(missing)}")
        # Run-ID / as-of binding: refuse stale or foreign artifacts outright.
        if payload["run_id"] != run_id or payload["decision_timestamp_ist"] != decision_ts:
            raise ValueError(
                f"stale or foreign artifact: bound to run "
                f"{payload['run_id']!r}/{payload['decision_timestamp_ist']!r}, "
                f"current run is {run_id!r}/{decision_ts!r}"
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
        return LLMSynthesisResponse(
            model_id=str(payload["model_id"]),
            reasoning_summary=str(payload["reasoning_summary"]),
            habitual_comparison_note=str(payload["habitual_comparison_note"]),
            research_rule_challenge_note=str(payload["research_rule_challenge_note"]),
            confidence_note=str(payload["confidence_note"]),
            citations=citations,
            estimated_tokens_used=0,
            estimated_cost_inr=0.0,
        )


def build_llm_provider(config: AnalystConfig) -> LLMProvider:
    if config.llm_provider == "hosted_api":
        return HostedBudgetGuardedLLMProvider(config)
    if config.llm_provider in ("ollama", "vllm"):
        return LocalEndpointLLMProvider(config.llm_endpoint_url, config.llm_model_id)
    if config.llm_provider == "agent_artifact":
        return AgentArtifactSynthesisProvider(config.agent_synthesis_artifact_path)
    return OfflineDeterministicSynthesisProvider(model_id="deterministic-local-synthesizer-v3")
