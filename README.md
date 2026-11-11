# NIFTY 50 Options Research & Strategy Analyst (`v0.4.0`)

Quantitative read-only research and defined-risk strategy desk for **NIFTY 50 Index Options**, combining a **100% deterministic Black-76 & surface-analytics engine** with **single-pass multi-persona synthesis** inspired by [*TradingAgents: Multi-Agents LLM Financial Trading Framework*](https://arxiv.org/abs/2412.20138) (Xiao et al., 2024).

> **Scope & Hard Compliance Boundaries**
> - **Strictly Read-Only**: Zero order placement, modification, cancellation, or broker write calls exist anywhere in the codebase (enforced via AST regression tests).
> - **100% Deterministic Quantitative Math**: Black-76 pricing, Brent implied-volatility solving, variance-interpolated constant-maturity 30-day IV (`CM30`), analytical Greeks, Wilder `RSI(14)` / `ATR(14)`, piecewise expiry payoff bounds, Black-76 stop triggers, and all 10 Part A gatekeeper rules are computed deterministically in pure Python—never by an LLM.
> - **Technical & Options Surface Driven (STT Excluded)**: Candidate ranking (`0–100` `technical_options_score`) and Part A Rule 4 (`<= Rs 10,000` contractual payoff bound) evaluate pure option-chain geometry, IV percentile, term structure, skew, and technical alignment across credit spreads, debit spreads, butterflies, and iron condors without exercise-STT distortion.
> - **Single-Pass Multi-Persona Synthesis (WhatsApp Agent Ready)**: Zero multi-call HTTP loops. A single structured pass runs the **Bull Researcher**, **Bear / Red-Team Researcher**, **Volatility & Skew Specialist**, **Options Risk Management Triad** (*Convexity Advocate*, *Tail Guardian*, *Neutral Arbiter*), and **Chief Options Strategist (Consolidator)**, audited by a deterministic two-way citation and prose hallucination verifier.
> - **Zero Backtesting / Zero Performance Tracking**: No historical strategy backtesting, simulated P&L, Sharpe/win-rate tracking, or outcome logging exists in the codebase. Official NSE UDiFF EOD archives are used **solely** to construct the point-in-time `CM30` ATM IV series required for IVP.
> - **Zero Paid LLM Spend by Default**: Defaults to `offline_deterministic`, local `local_endpoint` (`ollama` / `vllm`), or zero-API-cost `agent_artifact` handoff for external/WhatsApp LLM agents (`ALLOW_PAID_LLM_SPEND=false`).

---

## Architecture Overview

1. **Layer 1 — Read-Only Adapters & Data Hygiene (`src/nifty_analyst/adapters/`)**
   - Dynamic contract master & holiday-aware NSE trading session calendar (`Asia/Kolkata` IST enforced; zero hardcoded expiry weekdays or lot sizes).
   - Quote-level `180s` staleness gating (`FETCH_TIME_NOT_EXCHANGE` timestamps blocked from `LIVE-READY`), side-specific bid/ask depth, `2.5s` serial rate limiting in live mode, and immediate halt on HTTP `429`.
   - Pure-Python NSE UDiFF EOD volatility archive builder (`iv_history_builder.py`) enforcing exact `TckrSymb == "NIFTY"` filtering (`BANKNIFTY` / `FINNIFTY` / `MIDCPNIFTY` excluded).
2. **Layer 2 — Deterministic Quant, Technical & Structure Engine (`src/nifty_analyst/engine/`)**
   - Completed-candle slicing (forming intraday bars stripped; `15:15–15:30 IST` bar checked separately), `1D` HH/HL trend state, `1H` `MA20` / `EMA50`, Wilder `RSI(14)`, `ATR(14)`, stretch z-score, and geometric wick-rejection touch checks.
   - Multi-expiry option surface analytics: variance-interpolated `CM30` IV, near-vs-far term structure (`BACKWARDATION` / `CONTANGO` / `FLAT`), 25-delta put-call skew, OI walls, PCR, and Max Pain.
   - Multi-expiry defined-risk generator across **10 structure families** (`BULL_PUT_SPREAD`, `BULL_CALL_SPREAD`, `BULLISH_BUTTERFLY`, `BEAR_CALL_SPREAD`, `BEAR_PUT_SPREAD`, `BEARISH_BUTTERFLY`, `IRON_CONDOR`, `IRON_BUTTERFLY`, `LEGGED_IRON_CONDOR_COMPONENT`), Black-76 MTM stop root-finding, and pre-expiry `3%` overnight gap + asymmetric vol-jump stress testing.
   - All **10 Part A binding rules** keeping the **four rupee risk figures strictly separate**:
     1. Contractual Max Loss (`<= Rs 10,000` piecewise expiry payoff ceiling)
     2. Exchange Hedged Margin Estimate (`EXCHANGE_FORMULA_ESTIMATE`)
     3. Planned Per-Position Managed Stop (`<= Rs 2,000`)
     4. Portfolio Summed-Stop Impact (`<= Rs 10,000`, every open position counted in full with zero correlation netting)
3. **Layer 3 — Single-Pass `TradingAgents` Persona Synthesis & Two-Way Verifier (`src/nifty_analyst/synthesis/`)**
   - Produces adversarial **Bull / Bear / Volatility** arguments, **Risk Management Triad** deliberation, and the **Consolidator** verdict in one pass.
   - **Deterministic Two-Way Verifier** checks every `state_field_path` citation (`<= 2%` unsupported rate), verifies the Consolidator pick is `Part A == PASS`, and reverse-scans prose for confabulated structure IDs, ungrounded `Rs` amounts, or invented NIFTY index levels.
4. **Layer 4 — Reports, WhatsApp Digest, Intraday Delta Memory & Audit Log (`src/nifty_analyst/output/`)**
   - 7-section Markdown desk report (`renderer.py`), concise mobile-formatted WhatsApp digest (`personas.py`), optional payoff PNG chart (`payoff_diagram.py`), point-in-time same-day delta & duplicate-alert suppression (`intraday_memory.py`), and append-only scrubbed JSONL audit log (`audit_log.py`).

---

## Repository Layout

```text
nifty_options_analyst/
├── pyproject.toml
├── requirements.txt
├── .env.example                         # Config template (store real secrets outside repo with chmod 600)
├── run_tests.py                         # Zero-dependency test runner (40 suites / checks)
├── docs/
│   ├── implementation_plan_v3.md        # Core compliance & domain specification
│   ├── build_prompt.txt                 # Original build specification
│   ├── sample_offline_report.md         # Full 7-section sample report from bundled fixtures
│   └── sample_offline_state.json        # Frozen sample NiftyOptionsState JSON
├── fixtures/
│   ├── nse_holidays_2026.json           # NSE holiday calendar (including Tue 24 Nov 2026 holiday)
│   ├── contract_master.json             # Active expiries (including Mon 23 Nov holiday-shifted lot=50 expiry)
│   ├── option_chains.json               # Arbitrage-free Black-76 chains with quote-level IST timestamps
│   ├── index_candles.json               # 1D, 1H, and 15:15-15:30 IST candles + forming bar
│   ├── nse_eod_volatility_history.json  # Verified EOD CM30 IV history fixture
│   └── vix_and_events.json              # India VIX & macro/expiry events calendar
├── src/nifty_analyst/
│   ├── config.py                        # Environment & chmod-600 external secret loader
│   ├── scrubber.py                      # IST slot validator & tokenized PII/credential scrubber
│   ├── state.py                         # Typed JSON-serialisable NiftyOptionsState
│   ├── pipeline.py                      # End-to-end 4-layer pipeline orchestrator
│   ├── cli.py                           # CLI entry point (`nifty-analyst`)
│   ├── adapters/
│   │   ├── calendar_and_contracts.py    # Contract master & holiday/special-session NSE calendar
│   │   ├── kotak_neo.py                 # Read-only Kotak Neo adapter, 180s staleness & 429 halt
│   │   ├── eod_vix_events.py            # Point-in-time EOD IVP loader, VIX & event calendar adapters
│   │   └── iv_history_builder.py        # Pure-Python NSE UDiFF EOD CM30 IV archive builder CLI
│   ├── engine/
│   │   ├── pricing.py                   # Black-76, BSM, Brent IV, CM30 interpolation, Greeks & Chain QA
│   │   ├── detectors.py                 # Completed-candle technicals (RSI, ATR, Z-score) & regime detectors
│   │   ├── structures.py                # Multi-expiry defined-risk generator, Black-76 stops & Tech+Opt score
│   │   └── part_a_gatekeeper.py         # All 10 Part A binding compliance rules
│   ├── synthesis/
│   │   ├── personas.py                  # Compact Desk Brief, TradingAgents personas & WhatsApp formatter
│   │   ├── llm_node.py                  # Single-pass providers (offline, local, agent_artifact, budget-hosted)
│   │   └── verifier.py                  # Two-way deterministic state citation & prose verifier
│   └── output/
│       ├── renderer.py                  # 7-section Markdown report renderer
│       ├── payoff_diagram.py            # Expiry payoff PNG chart generator (optional matplotlib)
│       ├── intraday_memory.py           # Same-day market deltas & duplicate structure suppression
│       └── audit_log.py                 # Immutable JSONL audit log & alert-only delivery gate
└── tests/                               # Comprehensive offline verification suite
```

---

## Quick Start

### 1. Run the Verification Test Suite
Runs 100% offline using only the Python 3.11+ standard library:
```bash
python3 run_tests.py
```

### 2. Generate a Full 7-Section Desk Report (Offline Fixture Mode)
```bash
PYTHONPATH=src python3 -m nifty_analyst.cli \
  --decision-time-ist "2026-11-06T10:15:00+05:30" \
  --inspect-offline \
  --write-report docs/sample_offline_report.md \
  --write-state-json docs/sample_offline_state.json
```

### 3. WhatsApp Agent Execution Workflow (Single-Pass, Zero Multicall)

#### Direct Concise WhatsApp Digest (`--whatsapp`)
Prints the mobile-formatted digest (verdict, exact legs, Tech+Options score, 4 separate risk caps, net breakeven, and invalidation level) directly to `stdout`:
```bash
PYTHONPATH=src python3 -m nifty_analyst.cli \
  --decision-time-ist "2026-11-06T10:15:00+05:30" \
  --whatsapp \
  --inspect-offline
```
*(In scheduled alert-only mode without `--inspect-offline`, `stdout` stays completely silent whenever the verdict is `WAIT`, `NO_TRADE`, `INSUFFICIENT_DATA`, not `LIVE-READY`, or a duplicate of an already-alerted same-day structure.)*

#### Single-Pass External / WhatsApp LLM Agent Handoff (`LLM_PROVIDER=agent_artifact`)
When an external LLM agent inside WhatsApp wants to author the multi-persona debate & Consolidator rationale itself in a single turn:
1. Run with `LLM_PROVIDER=agent_artifact` and `LLM_AGENT_ARTIFACT_DIR=/path/to/handoff`.
2. If `<run_id>.response.json` is not yet present, the CLI writes `<run_id>.request.json` containing:
   - `compact_desk_brief` (`<1,200` tokens of pre-computed surface, technicals, and Part A candidate table)
   - `persona_instructions` & `single_pass_prompt` (Bull, Bear, Vol, Risk Triad, and Consolidator JSON schema)
3. Once the WhatsApp LLM agent writes `<run_id>.response.json`, re-running validates every citation and prose number through `verify_synthesis_against_state` before emitting the alert.

### 4. Refresh NSE UDiFF EOD Volatility History (For `252d` IVP)
```bash
PYTHONPATH=src python3 -m nifty_analyst.adapters.iv_history_builder \
  --fo-dir /path/to/udiff_fo_bhavcopies \
  --op-dir /path/to/udiff_op_bhavcopies \
  --output /path/to/nse_eod_volatility_history.json
```

### 5. Configure External Secrets for Live Read-Only Kotak Neo Mode
Never store credentials inside the repository. Keep secrets outside the repo with `0600` permissions:
```bash
mkdir -p ~/.config/nifty_analyst
cp .env.example ~/.config/nifty_analyst/config.env
chmod 600 ~/.config/nifty_analyst/config.env
export NIFTY_ANALYST_SECRET_FILE=~/.config/nifty_analyst/config.env
```
`load_chmod600_env_file` raises a `PermissionError` if the secret file resides inside the repository or has group/world permissions (`mode & 0o077 != 0`).
