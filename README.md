# NIFTY 50 Options Research & Strategy Analyst (`v0.3.0` — Implementation Plan v3)

> **Scope & Compliance Notice**: This repository implements **Phase 0** (Offline Deterministic Foundation) and **Phase 1** (Live Read-Only Snapshot Adapter + Single-LLM Evidence Synthesis & Deterministic Post-LLM Citation Verifier) of **Implementation Plan v3** for the NIFTY 50 Index Options Research & Strategy Analyst.
> - **Strictly Read-Only**: Zero order placement, order modification, or broker write calls exist anywhere in the codebase (not even stubbed).
> - **Zero Backtesting / Zero Performance Tracking**: Per the trader's 6 Nov 2026 instruction, there is no historical strategy backtesting, no simulated P&L, no win-rate or Sharpe calculation, and no outcome tracking in the audit log. Official NSE EOD archives are used **solely** to build the point-in-time constant-maturity 30-day ATM IV series required for IVP.
> - **Zero Paid LLM Spend Authorized by Default**: Pluggable LLM synthesis defaults to a zero-cost deterministic local synthesizer (`offline_deterministic`) or local `ollama`/`vllm`, with any hosted API locked behind `ALLOW_PAID_LLM_SPEND=false` and a per-run token and INR cost guard.

---

## Repository Layout

```text
nifty_options_analyst/
├── pyproject.toml
├── requirements.txt
├── .env.example                       # Placeholder config template (keep real secrets outside repo with chmod 600)
├── run_tests.py                       # Zero-dependency standard-library test runner (also compatible with pytest)
├── docs/
│   ├── implementation_plan_v3.md      # Attached Implementation Plan v3 specification
│   ├── build_prompt.txt               # Attached Build Prompt specification
│   ├── sample_offline_report.md       # Sample Phase 0/1 7-section report generated from bundled fixtures
│   └── sample_offline_state.json      # Sample frozen NiftyOptionsState JSON
├── fixtures/
│   ├── nse_holidays_2026.json         # NSE holiday calendar (including Tue 24 Nov 2026 holiday)
│   ├── contract_master.json           # Live contract master fixture (includes Mon 23 Nov 2026 holiday-shifted expiry & lot-size change)
│   ├── option_chains.json             # Arbitrage-free Black-76 option chains with quote-level IST timestamps
│   ├── index_candles.json             # 1D, 1H (09:15-15:15), and final 15m (15:15-15:30 IST) candles + forming bar
│   ├── nse_eod_volatility_history.json# 95 trading sessions of EOD ATM IV history (labelled EOD_CLOSE_INFERRED)
│   └── vix_and_events.json            # India VIX & human-maintained events calendar (tagged UNVERIFIED)
├── src/nifty_analyst/
│   ├── config.py                      # Environment & external chmod-600 secret file loader
│   ├── scrubber.py                    # IST timezone enforcement, 5 approved run slots, deterministic PII/secret scrubber
│   ├── state.py                       # Typed, JSON-serialisable NiftyOptionsState & TimestampedDatum
│   ├── pipeline.py                    # End-to-end orchestrator across Layers 1-4
│   ├── cli.py                         # CLI entry point (`nifty-analyst`)
│   ├── adapters/
│   │   ├── calendar_and_contracts.py  # Adapter 1 (Contract Master) & Adapter 5 (Trading Calendar)
│   │   ├── kotak_neo.py               # Adapter 2 (Quotes) & Adapter 3 (Candles), 180s staleness, 2.5s rate limit, 429 halt
│   │   └── eod_vix_events.py          # Adapter 4 (NSE EOD IVP series only), Adapter 6 (VIX), Adapter 7 (Events)
│   ├── engine/
│   │   ├── pricing.py                 # Black-76, BSM, Brent IV solver, analytical Greeks, Chain QA suite, piecewise expiry payoff
│   │   ├── detectors.py               # Completed-candle lock, geometric touch check, Two-Track Setup Detector Registry
│   │   ├── structures.py              # Credit/debit spreads, condors, butterflies, statutory charges, terminal PoP
│   │   └── part_a_gatekeeper.py       # All 10 Part A binding rules (PASS / FAIL / NOT_APPLICABLE / UNKNOWN)
│   ├── synthesis/
│   │   ├── llm_node.py                # Single-LLM synthesis node (sanitised JSON input only) + paid-spend guard
│   │   └── verifier.py                # Deterministic post-LLM citation & Part A verifier
│   └── output/
│       ├── renderer.py                # 7-section Markdown report renderer (numbers strictly from state)
│       └── audit_log.py               # Immutable JSONL audit log + alert-only delivery gate
└── tests/
    ├── test_pricing_and_qa.py         # Pricing math, IV solver, Chain QA suite, piecewise expiry payoff tests
    ├── test_adapters_and_hygiene.py   # Quote timestamps, 180s staleness, Mon 23 Nov 2026 expiry, lot size, 15m bar, 429 stop
    └── test_part_a_and_pipeline.py    # Part A encoding (1 DTE debit spread short leg, correlated stop summing), verifier & e2e tests
```

---

## Quick Start (100% Offline, Zero Network Access Required)

### 1. Run the Essential Correctness Test Suite
```bash
python3 run_tests.py
```

### 2. Run the Analyst Pipeline Offline Against Bundled Fixtures
```bash
PYTHONPATH=src python3 -m nifty_analyst.cli \
  --decision-time-ist "2026-11-06T10:15:00+05:30" \
  --inspect-offline \
  --write-report docs/sample_offline_report.md \
  --write-state-json docs/sample_offline_state.json
```

### 3. Configure External Secrets (For Phase 1 Live Read-Only Mode)
Never place credentials inside the repository. Copy `.env.example` outside the repository and restrict permissions to `0600`:
```bash
mkdir -p ~/.config/nifty_analyst
cp .env.example ~/.config/nifty_analyst/config.env
chmod 600 ~/.config/nifty_analyst/config.env
export NIFTY_ANALYST_SECRET_FILE=~/.config/nifty_analyst/config.env
```
If the file is inside the repository or has group/other permissions (`mode & 0o077 != 0`), `load_chmod600_env_file` raises a `PermissionError` and refuses to load it.

---

## Explicit Open / Unverified Items (Reported Faithfully Per Build Prompt)

1. **Kotak Neo SDK Method Names, Schemas & Rate Limits (`[UNVERIFIED]`)**:
   - Isolated behind `KotakNeoReadOnlyPort` in `src/nifty_analyst/adapters/kotak_neo.py`.
   - `UnverifiedLiveKotakNeoAdapter` tags each live endpoint (`KOTAK_SCRIP_MASTER_ENDPOINT_UNVERIFIED`, `KOTAK_QUOTES_CHAIN_ENDPOINT_UNVERIFIED`, `KOTAK_HISTORICAL_CANDLES_ENDPOINT_UNVERIFIED`, `KOTAK_INDIA_VIX_ENDPOINT_UNVERIFIED`) as `UNVERIFIED` until confirmed against the trader's installed Kotak Neo SDK version.
   - Rate limiting uses the provisional serial interval of `1 request per 2.5 seconds` (`KOTAK_MIN_REQUEST_INTERVAL_SECONDS=2.5`) and halts immediately on any HTTP 429 (`KotakRateLimit429Error`) with zero retries.
2. **India VIX Live & Historical Feed (`[UNVERIFIED]`)**:
   - Tagged `UNVERIFIED` in `TimestampedDatum`; IVP is computed from the NIFTY constant-maturity 30-day ATM IV series as primary.
3. **Broker Margin Calculator Endpoint (`[UNVERIFIED]`)**:
   - Margin estimates use the exchange hedged-margin formula and are explicitly labelled `EXCHANGE_FORMULA_ESTIMATE`.
4. **Trader Decisions Pending Before Live Phase 1 (`[PENDING TRADER]`)**:
   - Authorised LLM budget per run/month (`ALLOW_PAID_LLM_SPEND` defaults to `false`).
   - Capital/margin ceiling to report against.
   - Section 5C historical rule reconciliation (tagged `HISTORICAL_REFERENCE_UNCONFIRMED`; "stretched" metric tagged `DEFINITION_OPEN`; scalp exception kept strictly out of scope).
