"""Configuration loader with strict read-only and zero-paid-LLM-spend defaults."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from nifty_analyst.scrubber import SecretScrubber, load_chmod600_env_file

RunMode = Literal["OFFLINE", "LIVE_MORNING_SNAPSHOT"]
LLMProviderName = Literal[
    "offline_deterministic", "ollama", "vllm", "hosted_api", "agent_artifact"
]


@dataclass(frozen=True)
class AnalystConfig:
    """Immutable runtime configuration for the NIFTY 50 Options Research Analyst."""

    mode: RunMode = "OFFLINE"
    symbol: str = "NIFTY"

    # Provisional unverified serial request-rate limit (Section 3, Adapter 2)
    kotak_min_request_interval_seconds: float = 2.5
    kotak_max_queue_wait_seconds: float = 35.0
    kotak_chain_strike_count: int = 40
    kotak_quotes_batch_size: int = 2
    kotak_max_depth_tokens_per_expiry: int = 12
    max_quote_staleness_seconds: int = 180
    kotak_api_root: str = ""

    # Pricing & Execution Cost parameters (STT intentionally excluded per trader directive)
    risk_free_rate_annual: float = 0.0675
    slippage_buffer_points_per_leg: float = 0.50
    brokerage_per_order_rs: float = 20.0

    # Three Separate Money Policies (Section 4.3, 5A, 6.3) - NEVER MERGED
    per_position_managed_stop_rs: float = 2000.0
    portfolio_summed_stop_cap_rs: float = 10000.0
    per_trade_contractual_max_loss_ceiling_rs: float = 10000.0
    max_lots_per_trade: int = 1

    # Expiry 1 DTE short-leg delta limit (26 Oct 2026 rule)
    max_short_delta_at_1dte: float = 0.25

    # Multi-expiry & intraday deduplication controls
    evaluate_all_active_expiries: bool = True
    deduplicate_intraday_alerts: bool = True

    # Liquidity & IVP gate thresholds
    # Note (Item 4): min_top_book_depth_lots defaults to 0 so small touch quantities are kept
    # available as explicitly INDICATIVE rather than blanket-halted; full-size execution vs
    # indicative-touch scenario is enforced via touch_available_qty & walk_depth_levels.
    min_leg_liquidity_score: float = 50.0
    max_relative_bid_ask_spread_pct: float = 5.0
    min_strike_oi: int = 10000
    min_top_book_depth_lots: int = 0
    max_event_calendar_age_days: int = 14

    # Single-Pass Persona Synthesis & Spend Guard (Section 10 - Zero Multicall)
    llm_provider: LLMProviderName = "offline_deterministic"
    llm_model_id: str = "qwen2.5:14b-instruct"
    llm_endpoint_url: str = "http://127.0.0.1:11434/api/generate"
    # External synthesis seam: path to the artifact written by the WhatsApp/operator LLM agent
    # from the compact sanitised Desk Brief + TradingAgents persona pack (provider "agent_artifact").
    agent_synthesis_artifact_path: str = ""
    allow_paid_llm_spend: bool = False
    max_tokens_per_run: int = 4000
    max_cost_per_run_inr: float = 0.0
    hosted_llm_cost_per_1k_tokens_inr: float = 0.0

    # Paths (empty optional overrides fall back to fixtures_dir defaults)
    fixtures_dir: str = ""
    holidays_file_path: str = ""
    eod_archive_path: str = ""
    events_file_path: str = ""
    open_positions_file: str = ""
    payoff_chart_dir: str = ""
    cache_dir: str = ""
    audit_log_path: str = ""

    # Optional verified NSE special sessions {"YYYY-MM-DD": ("HH:MM", "HH:MM")}
    special_sessions: dict[str, tuple[str, str]] = field(default_factory=dict)

    # Runtime credentials (never serialized into config_hash or state)
    _runtime_secrets: dict[str, str] = field(default_factory=dict, repr=False, compare=False)

    def config_hash(self) -> str:
        """Deterministic SHA-256 hash of non-secret configuration fields."""
        public_dict = {
            k: v for k, v in asdict(self).items() if not k.startswith("_")
        }
        payload = json.dumps(public_dict, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def build_scrubber(self) -> SecretScrubber:
        return SecretScrubber(list(self._runtime_secrets.values()))

    def resolved_holidays_path(self) -> Path:
        if self.holidays_file_path:
            return Path(self.holidays_file_path)
        return Path(self.fixtures_dir) / "nse_holidays_2026.json"

    def resolved_eod_archive_path(self) -> Path:
        if self.eod_archive_path:
            return Path(self.eod_archive_path)
        return Path(self.fixtures_dir) / "nse_eod_volatility_history.json"

    def resolved_events_path(self) -> Path:
        if self.events_file_path:
            return Path(self.events_file_path)
        return Path(self.fixtures_dir) / "vix_and_events.json"

    def resolved_open_positions_path(self) -> Path | None:
        if self.open_positions_file:
            return Path(self.open_positions_file).expanduser()
        if self.mode == "LIVE_MORNING_SNAPSHOT":
            return Path.home() / "nifty-analyst" / "open_positions.json"
        return None

    @classmethod
    def from_env(cls, repo_root: Path | None = None) -> AnalystConfig:
        env = dict(os.environ)
        secret_file = env.get("NIFTY_ANALYST_SECRET_FILE", "").strip()
        if secret_file:
            file_vars = load_chmod600_env_file(secret_file, repo_root=repo_root)
            env.update(file_vars)

        base_dir = repo_root or Path(__file__).resolve().parents[2]
        default_fixtures = str(base_dir / "fixtures")
        default_cache = str(base_dir / ".cache")
        default_audit = str(base_dir / "logs" / "audit_log.jsonl")

        secrets: dict[str, str] = {}
        for k in ("KOTAK_CONSUMER_KEY", "KOTAK_SESSION_TOKEN", "KOTAK_SID", "HOSTED_LLM_API_KEY"):
            val = env.get(k, "").strip()
            if val and val != "PLACEHOLDER_DO_NOT_COMMIT":
                secrets[k] = val

        mode_val = env.get("NIFTY_ANALYST_MODE", "OFFLINE").strip().upper()
        mode: RunMode = (
            "LIVE_MORNING_SNAPSHOT" if mode_val == "LIVE_MORNING_SNAPSHOT" else "OFFLINE"
        )

        provider_raw = env.get("LLM_PROVIDER", "offline_deterministic").strip().lower()
        provider: LLMProviderName = "offline_deterministic"
        if provider_raw in (
            "offline_deterministic",
            "ollama",
            "vllm",
            "hosted_api",
            "agent_artifact",
        ):
            provider = provider_raw  # type: ignore[assignment]

        return cls(
            mode=mode,
            kotak_min_request_interval_seconds=float(
                env.get("KOTAK_MIN_REQUEST_INTERVAL_SECONDS", "2.5")
            ),
            kotak_max_queue_wait_seconds=float(
                env.get("KOTAK_MAX_QUEUE_WAIT_SECONDS", "35.0")
            ),
            kotak_chain_strike_count=max(1, int(env.get("KOTAK_CHAIN_STRIKE_COUNT", "40"))),
            kotak_quotes_batch_size=max(1, min(50, int(env.get("KOTAK_QUOTES_BATCH_SIZE", "2")))),
            kotak_max_depth_tokens_per_expiry=max(
                4, int(env.get("KOTAK_MAX_DEPTH_TOKENS_PER_EXPIRY", "12"))
            ),
            max_quote_staleness_seconds=int(env.get("MAX_QUOTE_STALENESS_SECONDS", "180")),
            kotak_api_root=env.get("KOTAK_API_ROOT", "").strip(),
            risk_free_rate_annual=float(env.get("RISK_FREE_RATE_ANNUAL", "0.0675")),
            slippage_buffer_points_per_leg=float(
                env.get("SLIPPAGE_BUFFER_POINTS_PER_LEG", "0.50")
            ),
            brokerage_per_order_rs=float(env.get("BROKERAGE_PER_ORDER_RS", "20.0")),
            max_lots_per_trade=max(1, int(env.get("MAX_LOTS_PER_TRADE", "1"))),
            evaluate_all_active_expiries=env.get("EVALUATE_ALL_ACTIVE_EXPIRIES", "true")
            .strip()
            .lower()
            == "true",
            deduplicate_intraday_alerts=env.get("DEDUPLICATE_INTRADAY_ALERTS", "true")
            .strip()
            .lower()
            == "true",
            llm_provider=provider,
            llm_model_id=env.get("LLM_MODEL_ID", "qwen2.5:14b-instruct").strip(),
            llm_endpoint_url=env.get(
                "LLM_ENDPOINT_URL", "http://127.0.0.1:11434/api/generate"
            ).strip(),
            agent_synthesis_artifact_path=env.get("AGENT_SYNTHESIS_ARTIFACT_PATH", "").strip(),
            allow_paid_llm_spend=env.get("ALLOW_PAID_LLM_SPEND", "false").strip().lower()
            == "true",
            max_tokens_per_run=int(env.get("MAX_TOKENS_PER_RUN", "4000")),
            max_cost_per_run_inr=float(env.get("MAX_COST_PER_RUN_INR", "0.0")),
            hosted_llm_cost_per_1k_tokens_inr=float(
                env.get("HOSTED_LLM_COST_PER_1K_TOKENS_INR", "0.0")
            ),
            fixtures_dir=env.get("NIFTY_ANALYST_FIXTURES_DIR", default_fixtures),
            holidays_file_path=env.get("NIFTY_ANALYST_HOLIDAYS_PATH", "").strip(),
            eod_archive_path=env.get("NIFTY_ANALYST_EOD_ARCHIVE_PATH", "").strip(),
            events_file_path=env.get("NIFTY_ANALYST_EVENTS_PATH", "").strip(),
            open_positions_file=env.get("NIFTY_OPEN_POSITIONS_FILE", "").strip(),
            payoff_chart_dir=env.get("NIFTY_ANALYST_PAYOFF_CHART_DIR", "").strip(),
            cache_dir=env.get("NIFTY_ANALYST_CACHE_DIR", default_cache),
            audit_log_path=env.get("NIFTY_ANALYST_AUDIT_LOG_PATH", default_audit),
            _runtime_secrets=secrets,
        )
