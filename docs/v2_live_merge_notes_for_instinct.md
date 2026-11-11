# NIFTY Options Analyst v2 — Verification of Instinct’s Review & Merge Architecture Notes

**Date**: 11 Nov 2026  
**Target Audience**: Instinct (running baseline maintainer) & Karthik  
**Scope**: Technical cross-check of Instinct’s 11 Nov 2026 review against `karthik5699/nifty-options-analyst-v2` (`4bd1a49`) and `karthik5699/Kotak_API` (`option_chain.py`), plus three architectural merge guards required whether executing **Path B** (porting `v2` modules into the running baseline) or **Path A / Path C** (bringing baseline live adapters + operational runners into `v2`).

> **Status Notice**: `v2` remains an offline-verified research build (`40/40` test suites passing). Do **not** switch live broker runs directly onto unmodified `v2` until the live adapter & open-positions merges below are completed and verified under supervised read-only checks.

---

## 1. Provenance: Why `v2` Diverged on the Live Layer

`nifty-options-analyst-v2` was branched from an offline `v0.3.0` snapshot that predated the 8–11 Nov 2026 supervised broker patches in the running baseline (`request_safety.py`, `interpret_depth_book`, chunked `historical_data`, fail-closed `open_positions.json` CLI loading, and `hourly_run.py` / `decision_driver.py` / `live_verify.py` / `smoke_open.py`).

All findings in Instinct’s review were cross-checked directly against `src/nifty_analyst/adapters/kotak_neo.py` and `karthik5699/Kotak_API/kotak_option_chain/option_chain.py` and are **100% confirmed**.

---

## 2. Item-by-Item Confirmation of Instinct’s Findings

### P1-1. Live Option Chain & Depth Wiring — Confirmed
- **Confirmed Defect**: `LiveKotakNeoAdapter.fetch_option_chain_raw` (`src/nifty_analyst/adapters/kotak_neo.py`) calls `get_full_option_chain(..., count=self.depth_strikes_each_side)` without `include_depth=True` (`option_chain.py:415` defaults `include_depth=False`), reads `ce.get("depth")` instead of `leg["market_depth"]` (`option_chain.py:631`), and looks for `total_buy_qty_lots` instead of raw `total_buy_qty` / `total_sell_qty` (`option_chain.py:300-301`).
- **Why `include_depth=True` Must NOT Be Used**: In `Kotak_API`, `_fetch_depth_by_neosymbol` (`option_chain.py:258-310`) chunks 25 tokens at a time with `time.sleep(0.04)` and `except Exception: pass`—which would silently swallow HTTP `400`/`429` errors and violate `PacedReadTransport` (`2.5s` spacing, zero auto-retry on `429`).
- **Agreed Fix**:
  1. Pull chain structure with `count=40` (~81 strikes) in one call per evaluated expiry without `include_depth=True`.
  2. Fetch executable books via `client.quotes(instrument_tokens=[...], quote_type="all")` in verified small batches only for shortlisted tokens + held-position contracts.
  3. Sort/normalize books best-first; reject empty, locked, or crossed (`bid >= ask`) books as explicit **`None` (`UNKNOWN` / unpriced)**—never `0.0` and never `ltp`.
  4. Convert raw quantities to lots using live per-contract `mktLot` read on the same run (never hardcoded).
  5. Preserve raw payloads in the audit log whenever a book is rejected so feed empties stay distinguishable from parser rejections.

### P1-2. Historical Candle Request Contract — Confirmed
- **Confirmed Defect**: `LiveKotakNeoAdapter.fetch_index_candles_raw` passes `neosymbol="Nifty 50"`, `interval="1day" | "60minute" | "15minute"`, and an unchunked `400`-day span—all rejected by `validate_history`.
- **Agreed Fix**: Use `"nse_cm|Nifty 50"`, intervals `"D"`, `"60min"`, `"15min"` (plus `"W"` if retained from baseline), chunk daily/weekly spans to `<= 179` days, merge/dedupe by candle timestamp, and parse both response shapes (bare `list` on intraday, `dict` wrapper on daily).

### P1-3. Rule 3 (`Rs 10,000` Portfolio Summed-Stop Cap) CLI Input — Confirmed
- **Confirmed Defect**: `evaluate_part_a_rules` (`src/nifty_analyst/engine/part_a_gatekeeper.py`) and `run_analyst_pipeline` (`src/nifty_analyst/pipeline.py`) enforce Rule 3 with zero correlation netting, but `src/nifty_analyst/cli.py` has no `--open-positions` / `NIFTY_OPEN_POSITIONS_FILE` loader and defaults to `[]`.
- **Agreed Fix**: Port the baseline’s fail-closed `load_open_positions` (`NIFTY_OPEN_POSITIONS_FILE`, defaulting to `~/nifty-analyst/open_positions.json`). In live mode, a missing or unreadable positions file must fail closed (`PORTFOLIO_STATE_UNKNOWN`) and block `LIVE-READY`.

### P2-1. Dead Spot Fallback & Per-Expiry `mktLot` — Confirmed
- **Confirmed Defect 1**: Lines `569–571` of `src/nifty_analyst/adapters/kotak_neo.py` raise `KotakAdapterError` in both branches even though `Kotak_API._fetch_spot_quote` (`option_chain.py:244-253`) returns `"ltp"` alongside `"display_symbol"`. Keep `source_id` explicit (`kotak_neo_nearest_futures_ltp` vs `kotak_neo_index_spot_nse_cm`) so futures basis vs cash index provenance is unambiguous.
- **Confirmed Defect 2 (Multi-Expiry Lot Size)**: Lines `558–590` of `src/nifty_analyst/adapters/kotak_neo.py` stamp a single `fut_resp["lot_size"]` onto every option expiry in `expiries`. Because `v2` evaluates multiple active expiries (`evaluate_all_active_expiries=True`) and lot sizes can change across NSE expiry cycles, each expiry/contract must carry its own live `mktLot`.

### P2-2 & P3. Operational Layer & Minor Items — Confirmed
- Agree on keeping/porting the baseline’s operational layer (`hourly_run.py`, `decision_driver.py`, `live_verify.py`, `smoke_open.py`) and its calendar/driver tests, raising default chain `count` from `10` to `40`, relying on transport-level `PacedReadTransport` for `429` hard-stop cooldowns (with adapter check as defense-in-depth), and aligning Python version metadata (`>=3.10`).

---

## 3. Three Critical Merge Guards (Interactions Between Baseline Live Rules & `v2` Engine)

Whether porting `v2` modules into the baseline (**Path B**) or fixing `v2` in place (**Path A**), these three interactions must be handled explicitly so the two designs do not collide:

### Guard 1 — Two-Stage Targeted Depth Funnel vs. `RawStrikeQuote` & `run_chain_qa_suite`
- **The Collision**:
  - With `count=40` (81 strikes = 162 contracts per expiry) and targeted `client.quotes` depth pulls (`P1-1`), ~65–70 of the 81 strikes will intentionally have **no depth fetched** (`bid=None, ask=None`).
  - Currently in `v2`, `RawStrikeQuote` (`src/nifty_analyst/engine/pricing.py:307`) types `call_bid` / `call_ask` / `put_bid` / `put_ask` as non-null `float`, and `run_chain_qa_suite` (`pricing.py:384-525`) runs put-call parity (5 near-ATM strikes) and call/put monotonicity across the entire `quotes` sequence.
- **Required Design**:
  1. **Stage A (Full 81-Strike Chain Pass)**: Compute `Put OI Wall`, `Call OI Wall`, `PCR (OI & Volume)`, and `Max Pain` across all 81 strikes from the chain response. Use strike distance / indicative Black-76 deltas from chain data solely to shortlist which strikes need live depth:
     - the **5 strikes nearest ATM** (`CE` + `PE`, 10 tokens — required for parity, synthetic forward `F`, and ATM IV anchor),
     - the **shortlisted candidate wing strikes** (`~0.45–0.55d`, `~0.25–0.30d`, `~0.15–0.22d`, plus `150/200`-pt wing partners), and
     - any **open-position contracts** from `open_positions.json`.
  2. **Stage B (Targeted Depth Enrichment & Priced-Only QA)**:
     - Change `RawStrikeQuote` bid/ask fields to `float | None` (never `0.0` for missing/crossed/unfetched books).
     - Run `run_chain_qa_suite` parity check on the 5 depth-fetched near-ATM strikes (failing closed with `CHAIN_ATM_DEPTH_MISSING` if any near-ATM book is `None`/crossed), and run monotonicity / vertical no-arbitrage checks strictly across strikes with non-null executable `bid` and `ask`.
     - In `enrich_chain_greeks` and `generate_candidate_structures` (`src/nifty_analyst/engine/structures.py`), skip any strike leg whose executable `bid` or `ask` is `None` so an unpriced or LTP-only strike can never enter mark, breakeven, stop, or PoP math.

### Guard 2 — Multi-Expiry Request Budget vs. `request_safety.py`’s `35s` Queue Cap
- **The Collision**:
  - At `2.5s` minimum spacing, a `35s` queue cap permits at most **`~14` HTTP requests per run**.
  - On a cold run, uncached candles (`3x D` chunks + `1x 60min` + `1x 15min` = 5 calls, or 6 with `W`) + contract/expiry metadata (`2` calls) already consume **7–8 calls (`17.5s–20.0s`)**, leaving only **6–7 calls** for chains and depth batches.
  - If `evaluate_all_active_expiries=True` pulls 3 option chains + multiple small depth batches across all 3 expiries simultaneously, it will breach the `35s` queue cap.
- **Required Design**:
  1. **Same-Day Disk Cache for Completed `1D` / `W` Candles**: Completed daily and weekly bars are immutable across the five intraday slots (`10:15`, `11:15`, `12:15`, `13:15`, `14:15` IST). Caching validated `1D`/`W` history keyed by `(trading_date_ist, symbol)` saves `3–4` HTTP calls (`7.5s–10.0s`) on intraday slots after morning warmup.
  2. **Chain-First Expiry Filtering Before Depth Calls**: Pull chain summaries (`1` call per active expiry, up to `2–3` expiries) to compute `compute_variance_interpolated_cm30_iv` (`pricing.py:243`) and term-structure slope (`BACKWARDATION` / `CONTANGO` / `FLAT`), then pull targeted `client.quotes` depth batches **only** for the primary target-DTE expiry’s shortlisted strikes + held positions (staying within budget until a bounded live test determines the true safe `quotes` batch limit between `2` and `50` tokens).

### Guard 3 — Preserving Explicit User Directives & Hard Invariants Across Path B
If porting `v2` modules into the running baseline (**Path B**), ensure these explicit user decisions in `v2` are preserved rather than overwritten by older baseline logic:
1. **Zero STT in Candidate Ranking, Payoff Diagrams & Rule 4**:
   - Karthik explicitly directed removing STT calculations (`EXERCISE_STT_RATE`, `stt_rs`, and the `all_in_worst_case_rs = inf` / `HOLD` gate on ITM calls) so technical structure and options surface analytics govern suggestions rather than exercise-tax artifacts.
   - Rule 4 (`RULE_4_CONTRACTUAL_MAX_LOSS_10000_RS`) is a symmetric `PASS`/`FAIL` check on `contractual_max_loss_rs <= 10,000` (including non-STT execution fees: brokerage, exchange txn, SEBI, stamp, GST).
2. **Single-Pass Multi-Persona Execution (Zero Multicall HTTP Loops)**:
   - Execution is invoked inside Karthik’s **WhatsApp LLM agent** workflow and must remain strictly **single-pass** (zero multi-round LLM API calls).
   - Keep `v2`’s single-pass persona pack (`src/nifty_analyst/synthesis/personas.py`):
     - **Adversarial Debate**: *Bull Researcher*, *Bear / Red-Team Researcher*, *Volatility & Skew Specialist*
     - **Options Risk Management Triad**: *Convexity & Edge Advocate*, *Conservative Tail Guardian*, *Neutral Desk Arbiter*
     - **Final Decision Persona**: *Chief Options Strategist (Consolidator)*
   - Preserve both `--whatsapp` (`format_whatsapp_digest`) and `LLM_PROVIDER=agent_artifact` (`build_compact_desk_brief` `<1,200` tokens + `persona_instructions` + `single_pass_prompt`), gated by the two-way deterministic verifier (`src/nifty_analyst/synthesis/verifier.py`).
3. **All Binding Risk & Safety Rules Intact**:
   - Strictly read-only (zero order placement/modification/cancellation).
   - Four separate, unmerged rupee figures: `Rs 10,000` contractual max-loss ceiling, exchange margin estimate, `Rs 2,000` per-position managed stop, and `Rs 10,000` portfolio summed-stop cap (zero correlation netting).
   - 1 DTE short-leg `|delta| < 0.25` checked leg-by-leg across both credit and debit structures.
   - Written completed-close index invalidation level before entry.
   - `FETCH_TIME_NOT_EXCHANGE` blocks `LIVE-READY` until exchange quote-timestamp semantics are verified live.
   - `ALLOW_PAID_LLM_SPEND=false` by default.

---

## 4. Module Map of `v2` Components Ready for Porting / Merge

| Capability | `v2` File(s) | Notes for Merge |
| :--- | :--- | :--- |
| **Two-Way Citation & Prose Verifier** | `src/nifty_analyst/synthesis/verifier.py` | Forward `state_field_path` check + Consolidator `Part A == PASS` check + reverse prose scan for invented structure IDs, `Rs` figures, and NIFTY index levels. |
| **Single-Pass Persona Pack & WhatsApp Digest** | `src/nifty_analyst/synthesis/personas.py`, `src/nifty_analyst/synthesis/llm_node.py` | Bull/Bear/Vol debate, Risk Triad, Consolidator, `<1,200`-token `compact_desk_brief` for `agent_artifact`, and `format_whatsapp_digest`. |
| **STT-Free 10-Family Structure Engine & Tech+Opt Score** | `src/nifty_analyst/engine/structures.py`, `src/nifty_analyst/engine/part_a_gatekeeper.py` | Adds `BULLISH_BUTTERFLY`, `BEARISH_BUTTERFLY`, `IRON_BUTTERFLY`, `LEGGED_IRON_CONDOR_COMPONENT`, Black-76 MTM stop root-finding, `3%` gap + vol-jump stress, and `0–100` `compute_technical_options_score`. Apply **Guard 1** (`float | None` bid/ask skip) when merging with targeted live depth. |
| **Surface Variance CM30 IV & Wilder Technicals** | `src/nifty_analyst/engine/pricing.py`, `src/nifty_analyst/engine/detectors.py` | `compute_variance_interpolated_cm30_iv`, term-structure slope/regime, Wilder `RSI(14)` / `ATR(14)`, stretch z-score, and symmetric regime detectors. |
| **Same-Day Intraday Delta & Alert Dedup Memory** | `src/nifty_analyst/output/intraday_memory.py`, `src/nifty_analyst/output/audit_log.py` | Point-in-time (`prior_dt < dec_ist`) spot/IV/PCR deltas and duplicate same-day `structure_id` alert suppression (zero P&L/outcome tracking). |
| **Payoff Diagram & 7-Section Desk Renderer** | `src/nifty_analyst/output/payoff_diagram.py`, `src/nifty_analyst/output/renderer.py` | STT-free expiry payoff PNG chart (lazy `matplotlib` import) and updated Markdown desk report. |
| **Tokenized Secret Scrubber** | `src/nifty_analyst/scrubber.py` | `is_sensitive_key_name` scrubs `kotak_sid`, `pan`, `token`, etc. without false-positive scrubbing `OptionLeg.side` or `calendar_discrepancy_warnings`. |
