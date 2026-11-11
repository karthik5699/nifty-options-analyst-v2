# Implementation Plan v3: NIFTY 50 Options Research & Strategy Analyst


> **Scope & Compliance Notice**: This document is an implementation plan for a research-only decision-support system. It contains no source code, no trade advice, and no broker write or order-execution capability in any phase. Version 3 incorporates the trader's steering of 6 Nov 2026: **no backtesting and no historical strategy-performance validation of any kind** (only essential software, pricing, data and risk-calculation correctness checks remain), a **Rs 10,000 maximum contractual loss per trade** (ceiling, not target), and **approval to run five hourly checks per market day (10:15, 11:15, 12:15, 13:15, 14:15 IST) fetching the Kotak API data each run needs**, alerting only on qualifying setups. Items marked **[UNVERIFIED]** or **[PENDING TRADER]** are not assumed until settled by verification or the trader's answer.


---


## Phased Delivery Roadmap (Overview)


Each phase delivers a standalone, usable research capability and ends with a **Trader Stop-or-Go Gate** based on correctness and usability, never on trading performance. The trader's two existing user-facing briefs (technical EOD brief and NSE EOD derivatives brief) remain completely untouched throughout.


**Phase 0: Deterministic Foundation**
- Deliverable: contract-master parser, deterministic IV/Greeks/pricing engine, Part A rule checker, candidate structure generator, offline report renderer.
- Data: offline. Official NSE EOD archives are used only to build volatility history for IVP context, never to backtest strategies.
- LLM usage: zero calls.
- Trader gate: he inspects the correctness-check results (Section 8) and one sample offline report before any API spend or live broker quote.


**Phase 1: Live Snapshot + Single-LLM Synthesis**
- Deliverable: read-only Kotak Neo adapter (throttled, cached), live evidence state, single-LLM synthesis with deterministic citation verification, audit log.
- Data: read-only live quotes plus EOD volatility history. Live Kotak reads are approved (6 Nov 2026): five scheduled runs per market day at 10:15, 11:15, 12:15, 13:15 and 14:15 IST, each fetching the data it needs.
- LLM usage: 1 call per run.
- Trader gate: he reviews live-read reliability (staleness, rate limits) and output quality over several sessions.


**Phase 2 (Optional): Bull/Bear Debate Layer**
- Deliverable: LangGraph state machine adding structured Bull/Bear invalidation debate before synthesis.
- Data: same as Phase 1.
- LLM usage: 3-4 calls per run.
- Trader gate: kept only if his side-by-side review finds the debate improves reasoning quality (sharper invalidations, fewer unsupported claims). Not a performance judgement.


---


## 1. Scope and Success Criteria


### 1.1 Operational Scope & Hard Boundaries
- **Instrument**: NIFTY 50 index options (NSE). The trader's equity momentum book is out of scope.
- **Read-Only Advisory Boundary**: The system reads market and option-chain data, detects setups, compares defined-risk structures, runs deterministic risk checks, and outputs a structured research proposal (`SUGGESTION`, `WAIT`, or `INSUFFICIENT DATA`). It has **zero** order-placement, order-modification, or broker write permissions in any phase.
- **Defined-Risk Mandate**: Every proposed structure must have bounded contractual maximum loss by construction (vertical credit or debit spreads, iron condors, iron butterflies). Naked long or short options are prohibited by construction. (**[PENDING TRADER]**: whether rare scalps are any exception is unresolved; naked exposure stays strictly out of scope unless he explicitly settles it.)
- **Existing Tools Preserved**: No import, modification, rewiring, or runtime dependency on the trader's two existing briefs.


### 1.2 Success Criteria (Correctness & Process Only)
1. **Numerical & Rule Integrity**: 100% of Greeks, IVs, IVP, breakevens, contractual max losses, probabilities of profit, and Part A checks are computed by unit-tested deterministic functions, never estimated by an LLM.
2. **Zero Part A Breaches**: 0% of outputs labelled compliant `SUGGESTION` violate any applicable binding rule in Section 5A of the briefing rules.
3. **Evidence Grounding**: under 2% of claims in LLM-generated rationales fail automated citation verification against the typed fields of the current run's evidence state.
4. **Stability**: on identical frozen input snapshots, repeated runs (5 trials) agree on the primary regime at least 80% of the time.
5. **Valid Sit-Out Behaviour**: the system outputs `WAIT` or `INSUFFICIENT DATA` whenever data is stale or incomplete, liquidity or IVP gates fail, Part A budgets are exhausted, or no setup has clear invalidation.
6. **Explicitly not success criteria**: historical P&L, win rate, Sharpe ratio, backtest outperformance, or any historical strategy-performance measure. Per the trader's 6 Nov 2026 instruction, none of these are measured in any phase.


### 1.3 Hard Stop Conditions
- **Stop the LLM layer**: LLM repeatedly confabulates prices, candle completions, or chain numbers (over 5% ungrounded-claim rate after prompt and schema tightening).
- **Stop live broker reads**: rate-limit errors (HTTP 429) or timestamp gaps cannot be eliminated via caching and throttling after live reads are approved and begin.
- **Stop the project**: per-decision API cost exceeds the trader's budget without commensurate analytical value over his existing two briefs.


---


## 2. Architecture: Components, Data Flow, State Schema


### 2.1 Design Principle
LLMs are confined to qualitative synthesis, adversarial invalidation testing, and trade-off articulation. Every number comes from deterministic, unit-tested code. No LLM output is shown to the trader until a deterministic verifier re-checks every cited figure and every Part A result.


**Layer 1 - Deterministic Ingestion (no LLM)**: NSE EOD archive adapter (volatility history only), Kotak Neo read-only adapter (contract master, expiries, quotes, index candles), VIX and events adapters, all feeding a snapshot validator (timestamps, staleness, lot size and expiry checks).


**Layer 2 - Deterministic Quant & Setup Engine (no LLM)**: pricing/IV/Greeks engine with QA suite, completed-candle and price-structure detectors, chain and volatility analytics (IVP, skew, OI walls, liquidity), candidate structure generator and pricer, and the Part A deterministic gatekeeper (`PASS` / `FAIL` / `NOT_APPLICABLE` / `UNKNOWN`).


**Layer 3 - Reasoning (LLM, Phases 1-2)**: single-LLM synthesis (Phase 1) or Bull/Bear debate plus synthesizer (Phase 2), citing evidence-state fields only, followed by deterministic output and citation verification.


**Layer 4 - Human Gate & Audit Log**: structured report to the trader; immutable audit record of inputs and outputs. The trader decides everything. The system never acts.


### 2.2 Shared State Schema (`NiftyOptionsState`)
One typed, JSON-serialisable object per run (no unstructured chat history):
- `run_metadata`: run id, decision timestamp (IST), mode (`OFFLINE` or `LIVE_MORNING_SNAPSHOT`), data version hash, model ids, config hash.
- `contract_context`: symbol, spot price and as-of timestamp, active expiries (each with expiry date, DTE in trading sessions, lot size, holiday-adjusted flag) read live from the contract master.
- `portfolio_exposure_state`: open positions with directional sign, planned stop in Rs, current executable spread value, short-leg deltas; `summed_planned_stops_rs` and `remaining_stop_budget_rs`.
- `market_evidence`: completed-candle price structure (1D trend state, last completed 1H bar with exact timestamps, verified level touches), volatility state (India VIX, ATM IV by expiry, IVP, IV-RV spread, 25-delta skew), derivatives positioning (PCR, OI walls, max pain), event calendar, and triggered setup detectors (tagged `PART_C_HISTORICAL` or `NOVEL_QUANT`, each with an invalidation level and candle basis).
- `candidate_structures`: deterministically generated and priced defined-risk structures with legs, expiry, DTE, lot size, lots, width, executable net credit/debit, contractual max loss, margin estimate with source, planned stop, executable stop spread value, breakevens, modelled PoP with assumptions, short-leg max |delta|, liquidity score, Part A check matrix, and a `matches_part_b_style` flag.
- `debate_state` (Phase 2 only): bull case, bear case, one rebuttal round.
- `final_output`: verdict, primary proposal, alternative comparison, Part A summary, research rule-challenge note (if any), confidence note, staleness report.


---


## 3. Data Adapters, Throttling, and Capability Gaps


Every datum is wrapped in a `TimestampedDatum` envelope: value, as-of timestamp (IST), fetched-at timestamp, source id, max staleness seconds, is-stale flag.


**Adapter 1 - Contract Master & Expiry** (Kotak Neo scrip master / expiry list)
- Fetched once per trading morning, cached per day, never hardcoded. Expiry dates, lot sizes and holiday adjustments come only from here (live example: the November 2026 monthly expiry is Monday 23 Nov ahead of the 24 Nov holiday; a weekday rule would miss this). On failure: halt with `INSUFFICIENT DATA`.


**Adapter 2 - Option Chain & Quotes** (Kotak Neo chain, quotes, market depth)
- Quote-level `update_time` must be extracted (the tested parser dropped timestamps). Max staleness 180 seconds for active strikes during market hours.
- **Live reads are approved for the five scheduled runs** (Section 3.1, item 5). **Request rate is provisional and unverified**: default config value 1 request per 2.5 seconds (configurable), single-threaded, with local caching; replace it with the broker's documented rate limit once that is verified. The only verified fact today is HTTP 429 after two rapid requests in testing, so on any 429 the run stops immediately and reports; no automatic retry, no backoff storm, no probing for the account's limits. On stale quote: mark affected strikes `STALE` and abort pricing.


**Adapter 3 - Historical Index OHLC** (Kotak Neo 1H and 1D index candles)
- A bar counts as completed only when the decision timestamp is past its close; forming bars are stripped. **Candle boundaries must be verified against the actual broker feed**, including the final 15-minute bar (15:15-15:30 IST) after the six hourly bars.


**Adapter 4 - NSE EOD Files, volatility history only** (official NSE EOD derivatives and index archives)
- Used solely to build the historical ATM IV series that IVP requires, and historical strike-level OI context. **Not used for strategy backtesting of any kind.** Point-in-time clamped to trade date. Immutable local store.


**Adapter 5 - Trading Calendar** (live contract master expiries + official NSE holiday calendar)
- Refreshed monthly, cross-checked against live expiries every run; discrepancies defer to the live contract master and raise a warning.


**Adapter 6 - India VIX [UNVERIFIED SOURCE]** (candidate: Kotak index quote or NSE VIX archive)
- Must be tested in Phase 1 whether the broker serves live and historical India VIX. Until settled, IVP is computed from the NIFTY ATM IV series as primary, with VIX marked `UNVERIFIED`.


**Adapter 7 - Events Calendar [UNVERIFIED SOURCE]** (human-maintained version-controlled calendar file: RBI, budget, FOMC/CPI, heavyweight earnings, expiry shifts)
- Starts as a manual file updated monthly; automated feeds tested later. Live mode warns if the file is over 14 days old.


### 3.1 Explicit Data Capability Gap List
1. **No broker-provided IV, Greeks, or IVP (verified gap)**: computed deterministically in Layer 2.
2. **No historical synchronised bid/ask surfaces (verified gap)**: IV is always model-inferred, current or historical; history provides closing prices only, so historical IV is inferred from EOD closes and labelled `EOD_CLOSE_INFERRED`. (This supports the IVP series only; it is not used to test strategies.)
3. **Broker chain parser timestamp loss (verified gap)**: parser rewritten in Phase 1; quotes without timestamps rejected.
4. **Aggressive broker rate limiting (verified gap)**: see Adapter 2; the safe request rate is unknown and must be discovered cautiously.
5. **Live-read approval (settled 6 Nov 2026)**: the trader has approved the analyst to run hourly at 10:15, 11:15, 12:15, 13:15 and 14:15 IST on market days, fetching the Kotak API data each run needs. Still not approved: any order capability (permanently out of scope for this system), LLM API spend beyond a budget he sets, and any widening of the schedule, instruments, or data capture beyond what the runs need. Each of those requires his explicit fresh approval and is not granted by this document.
6. **No reliance on upstream TradingAgents dataflows**: purpose-built NIFTY adapters only.


---


## 4. Derived Analytics & Deterministic Pricing Engine


### 4.1 IV & Greeks Engine (with mandatory QA)
- **Model**: Black-76 with implied forward from near-ATM put-call parity, or BSM with continuous dividend yield and a stated risk-free rate convention. Time to expiry measured to 15:30 IST on the contract master's expiry date.
- **IV Solver**: Brent's method over sigma in [0.01, 3.00], on mid price for live quotes or settlement/close for EOD history (labelled `EOD_CLOSE_INFERRED`). Wide markets: IV computed separately at bid, mid, ask.
- **QA suite (runs before any chain is used)**: put-call parity consistency across the 5 strikes nearest ATM (abort with `CHAIN_PARITY_FAILURE` on violation); ATM-vs-spot anchor check (|delta| approx 0.50 strike within one strike interval of the synthetic forward; ATM call and put IVs agree within 0.5 vol points); monotonicity and vertical-spread no-arbitrage checks.


### 4.2 Volatility, Positioning, and Liquidity
- **IVP**: trailing 30/90/252-session percentile of the constant-maturity 30-day ATM IV series (point-in-time clamped). **IVP is mandatory for live readiness**: if no verified IV history exists yet, IVP is reported as `UNKNOWN` and every affected output is labelled research-only; the requirement is not removed, relaxed, or replaced by a proxy gate.
- **Skew**: 25-delta put IV minus 25-delta call IV, plus wing curvature.
- **OI walls and PCR** for the target and nearby expiries.
- **Liquidity score (0-100)** per strike and structure from relative spread, top-of-book depth, volume, and OI. Any leg failing the spread or depth threshold fails the Part A liquidity gate.


### 4.3 Payoff, Four Rupee Quantities, Breakeven, PoP, Margin
For every candidate (lot size L from the live contract master, lot count per Section 6.3):
1. **Executable entry**: worst-case executable prices (short legs at bid, long legs at ask) plus configurable slippage buffer.
2. **Contractual max loss (Rs)**: computed leg-wise from the piecewise expiry payoff of the actual structure (evaluating net payoff across all strikes and wings, taking the minimum), not from a generic formula. The closed-form shortcuts apply only where they are exact: (width minus net credit) x L x lots for a single vertical credit spread; net debit x L x lots for a single vertical debit spread; worst wing for a balanced condor. Butterflies, unequal-wing condors, and any asymmetric structure require the leg-wise piecewise evaluation. This is the worst case if the market gaps through any stop.
3. **Planned stop (Rs 2,000 rule)**: at suggestion time the engine shows the indicative spread value at which the position loses Rs 2,000, plus a scenario-based indicative index level for it (clearly marked as dependent on IV, time and model assumptions, not an exact correspondence). **The operative stop is always computed from the actual executed entry (real fill prices, quantity, and charges) at execution time; suggestion-time marks are indicative only.** Gaps and fast tapes can fill worse than the stop; the stop is not a guaranteed loss limit.
4. **Portfolio summed-stop check (Rs 10,000 cap)**: sums the full planned stop of **every** open position plus the proposed trade. Correlated same-direction positions are grouped for directional-exposure reporting only; **their stops are never netted, discounted, or merged for the cap - every position's planned stop counts in full.** The total must remain under Rs 10,000.
5. **Breakevens**: gross and net of statutory costs (STT, exchange charges, SEBI fees, stamp duty, GST, brokerage).
6. **Modelled PoP (labelled model-dependent and expiry-horizon)**: terminal-distribution probability under two stated distributions (IV-implied lognormal; empirical return distribution). Always labelled. **It describes the expiry payoff only and does not reflect stop-managed or invalidation-managed early exits**; it is one input, not a prediction of the managed trade's outcome. Never inferred from 1 minus |delta|.
7. **Margin estimate**: from the broker's margin calculator if verified available in Phase 1, else from the exchange hedged-margin formula labelled `EXCHANGE_FORMULA_ESTIMATE`. Never assumed.


---


## 5. Market State and Setup Detection


### 5.1 Binding Signal Hygiene
1. **Completed-candle lock**: only candles whose close timestamp is before the decision time enter indicators and triggers. Candle boundaries are verified against the broker feed, including the final 15-minute bar (15:15-15:30).
2. **Actual touch verification**: a geometric check that the completed bar's [low, high] interval intersected the reference line, with zero tolerance unless the trader settles one.


### 5.2 Two-Track Detector Registry
**Track 1 - Historical reference detectors (context, not gates)**: his documented 1H MA20 / MA20+EMA50 / EMA50 wick-rejection variants, downtrend structure state (daily higher high + higher low repair per his 8 Oct 2026 note "never buy on the first pullback; rather, sell"), and retest status. Until he reconciles Section 5C of his rules, all Track 1 detectors are tagged `HISTORICAL_REFERENCE_UNCONFIRMED` and none of them filters or blocks a suggestion.


**Track 2 - Novel and regime-driven detectors (open discovery)**: volatility compression/expansion setups favouring debit structures, high-IV range setups favouring credit or neutral structures, skew dislocation setups, and candidate stretch metrics (reported as `DEFINITION_OPEN` until he defines "stretched"). Novel setups outside his historical catalog are first-class citizens.


---


## 5A. Binding Trading Rules (Part A) - Complete and Self-Contained


This section reproduces the trader's full current binding policy so this document stands alone. Nothing here references external documents; there are no unstated rules. Each rule carries its source date where recorded.


1. **Defined risk only**: every suggested structure has bounded contractual maximum loss by construction (vertical spreads, condors, butterflies). Naked long or short options are prohibited. A possible scalp exception is unresolved and out of scope unless he explicitly settles it.
2. **Rs 2,000 per-position managed stop**: exit immediately when the position loses Rs 2,000, with no negotiation and no waiting for candle confirmation. The operative stop is computed from the actual executed fill (prices, quantity, charges). Gaps and fast tapes can fill worse; the stop is not a guaranteed loss limit.
3. **Rs 10,000 portfolio summed-stop cap**: the sum of every open position's full planned stop plus the proposed trade's stop must stay under Rs 10,000. Correlated positions are never netted or discounted; correlation is exposure grouping only.
4. **Rs 10,000 per-trade contractual maximum loss ceiling (6 Nov 2026)**: hard ceiling, not a target, computed leg-wise from the piecewise expiry payoff (Section 4.3). Not a capital or margin figure.
5. **Expiry risk (26 Oct 2026)**: for any structure with short option legs, do not carry the trade into 1 DTE unless the short strike's absolute delta is below 0.25, evaluated leg by leg; days counted in trading sessions. Applicability is decided per leg, never by the structure's net debit or credit.
6. **Written invalidation before entry (22 Sep 2026)**: every suggestion carries an exact index invalidation level on a completed-daily-close basis; a completed-1H-close basis may be chosen per trade with stated justification.
7. **Signal hygiene**: completed candles only, with geometrically verified actual line touches (Section 5.1).
8. **IVP and liquidity gates**: mandatory for live readiness. If no verified IV history exists yet, IVP is reported UNKNOWN and outputs are labelled research-only; the requirement is never waived. Any leg failing the liquidity gate fails the structure.
9. **Rules change only between trades, only by him**: any evidence-backed challenge to these rules is presented as a labelled research proposal, never as a live-ready suggestion, and never applied automatically.
10. **Human final decision**: every output is advisory. He approves or rejects. The system never acts.


Style context (Part B, context only, never gates): he usually sells credit spreads with the short strike near 0.30 delta (for condors, shorts near 0.15 delta per side), hedge width usually about 200 points, rolls only the untested side at 0.40 delta on a completed 1H candle, reviews at 50-60% of maximum profit, and checks level, then delta, then IVP, then liquidity. His 8 Oct 2026 downtrend note: never buy the first pullback; bullish premium selling in a downtrend waits for a daily higher high and higher low. Historical variants (older trigger forms, 20-40 DTE window, one cycle per expiry, condor delta ideas, "stretched") are unconfirmed reference only; detectors for them stay tagged HISTORICAL_REFERENCE_UNCONFIRMED until he reconciles them.


---


## 6. Strategy Selection Policy (Open Defined-Risk Comparison)


### 6.1 Candidate Generation
For each detected setup, generate and price side by side, across active expiries:
- **Bullish**: bull put spread (credit, his usual style), bull call spread (debit, on equal footing), defined-risk bullish butterfly.
- **Bearish**: bear call spread (credit), bear put spread (debit), defined-risk bearish butterfly.
- **Neutral/range**: legged iron condor components, simultaneous iron condor, iron butterfly.


### 6.2 Selection Rules
1. **Hard pre-filter (Part A only)**: a candidate failing any applicable binding rule is `PART_A_FAIL` and cannot be a compliant suggestion. Each Part A check applies only where meaningful for the structure, leg by leg (the 1 DTE |delta| < 0.25 rule applies to short legs, so it binds a debit spread's short leg too; net debit or credit is never the deciding factor). A check that does not apply is `NOT_APPLICABLE`, never a failure.
2. **Zero filtering by style or history**: a bull call spread (or any defined-risk structure) passing Part A is never filtered out or labelled non-compliant because he usually sells credit spreads.
3. **Multi-factor comparison**: IV-regime fit (vega sign vs IVP), skew and strike geometry, breakeven distance and required move, contractual max loss vs the Rs 2,000 stop budget, capital and margin efficiency, liquidity and execution cost.
4. **Style departure vs rule departure**: a Part-A-compliant suggestion outside his usual style is a normal `SUGGESTION` with a side-by-side table against his habitual structure. A proposal to depart from an actual Part A rule is isolated as `RESEARCH PROPOSAL - NOT CURRENTLY COMPLIANT`, never presented as live-ready, and no rule ever changes automatically.
5. **First-class sit-out**: if nothing offers positive risk-reward after costs, the verdict is `WAIT`.


### 6.3 Position Sizing
- **Maximum contractual loss per trade: Rs 10,000 (settled 6 Nov 2026).** This is a hard ceiling, not a target: every proposed structure must have contractual max loss at or under Rs 10,000, computed leg-wise from the piecewise expiry payoff as Section 4.3 requires (formula shortcuts only where exact, single verticals), and the gatekeeper encodes it as a binding Part A check with that source date. It is a contractual-loss ceiling, not a capital or margin figure.
- **Three separate policies coexist and are never merged**: the Rs 2,000 per-position managed stop, the Rs 10,000 portfolio summed-stop cap (every position's planned stop counted in full), and the Rs 10,000 per-trade contractual-loss ceiling. None replaces or relaxes another.
- **Lot count derivation**: candidates are computed starting at one lot as a computational starting point (not a stated preference of his), and additional lots only when the resulting contractual max loss stays under the per-trade ceiling and the added planned stop keeps the portfolio summed stops under Rs 10,000. The system has no authority to size his actual trade; it reports each candidate's lot count, resulting max loss and margin (from broker or exchange sources only, Section 4.3), and he decides.


---


## 7. Decision and Output Format


Every run produces one report with verdict `SUGGESTION`, `WAIT`, or `INSUFFICIENT DATA`, containing:
1. **Header & data freshness banner**: verdict, run timestamp, data mode, spot and chain as-of timestamps, staleness status, contract-master facts (expiry date, DTE in sessions, lot size, holiday note).
2. **Evidence summary** keyed to state fields: completed-candle structure, volatility and skew, OI walls and liquidity, events within DTE.
3. **Reasoning summary** (Phase 2: adversarial bull vs bear with cited evidence fields and explicit invalidations; why the verdict survived or why `WAIT` won).
4. **Primary structure specification** (if `SUGGESTION`): legs with strikes, bid/ask, delta, IV; **the four distinct rupee quantities side by side** (contractual max loss with the gap warning; margin with its source; the Rs 2,000 planned stop as an indicative suggestion-time mark, noting the operative stop comes from the actual fill; portfolio summed-stop impact with every position counted in full); breakevens gross and net; modelled PoP with its model-dependent, expiry-horizon label; written invalidation level and candle basis; management plan (review threshold, 1 DTE rule check).
5. **Part A check matrix**: every binding rule with source date, status (`PASS` / `FAIL` / `NOT_APPLICABLE` / `UNKNOWN`), and the exact evidence used.
6. **Trade-off comparison vs usual style**: side-by-side table against his habitual structure on cost, max loss, breakeven, PoP, and vega/skew exposure.
7. **Between-trade research notes** (if any): evidence-backed challenges to heuristics or rules, presented per his presentation preference, never as live-ready suggestions.


---


## 8. Correctness Checks (Not Backtesting)


The trader has excluded backtesting and historical strategy-performance validation from this project. What remains is ordinary software correctness verification, which is required before any output can be trusted:


1. **Pricing engine unit tests**: IV solver and Greeks against known reference values; put-call parity, monotonicity, and no-arbitrage checks on hand-constructed chains; ATM anchor check on known cases.
2. **Adapter contract tests**: every quote carries a timestamp; staleness flags trip correctly; contract-master parsing handles a holiday-shifted expiry (the Monday 23 Nov 2026 case) and a lot-size change; candle slicing recognises completed bars including the final 15-minute bar (15:15-15:30).
3. **Part A encoding tests**: each binding rule exercised on hand-worked pass, fail, not-applicable, and unknown examples, including leg-by-leg applicability (the 1 DTE rule on a debit spread's short leg) and the full summing of correlated positions' stops.
4. **LLM output quality checks**: citation verification against the evidence state, and repeat-run stability on frozen inputs.
5. **Volatility history build check**: the EOD-derived ATM IV series used for IVP is validated for continuity and sanity (gaps flagged), as a data-correctness matter only.


**What is explicitly absent from the current scope, per his 6 Nov 2026 instruction**: no historical replay of strategies, no simulated P&L, no out-of-sample testing, no win-rate or Sharpe measurement, no forward paper-performance tracking. This is his current scoping decision, not a standing prohibition: outcome logging or forward review can be added later only if he asks for it, and nothing in any phase reintroduces it on its own.


---


## 9. Audit Log (No Performance Tracking)


Each run appends an immutable record: run id, timestamp, data version hash, model ids, frozen evidence state, candidates considered, verdict, and full report. Purpose: debugging, auditability, and letting the trader review what the system saw and said. **The log records no outcomes, no P&L, and no settlement, and feeds no performance loop.**


---


## 10. Cost, Operations, Schedule, and Secrets


- **LLM budget**: Phase 0: zero calls. Phase 1: 1 call per run. Phase 2: 3-4 calls per run. Actual cost per decision must be derived at build time from the chosen model's current pricing and the measured token budget per run, and checked against the budget he sets; no cost figure is asserted here. A local model has no API bill, but it is not zero-cost (hardware, power and maintenance).
- **Schedule (settled 6 Nov 2026)**: five runs per market day at 10:15, 11:15, 12:15, 13:15 and 14:15 IST, timed just after each completed 1H candle, fetching the Kotak data each run needs. The 15:15 candle and the 15:30 close are outside his requested window. Any change to this schedule requires his fresh approval.
- **Delivery**: the trader hears from the system only when a run finds a suitable qualifying setup (verdict `SUGGESTION`), with the recommended options strategy and its contractual max loss under the Rs 10,000 ceiling. `WAIT` and `INSUFFICIENT DATA` outcomes go to the audit log with no routine message. No hourly pings.
- **Secrets**: broker credentials and API keys live only in the OS keychain or a chmod-600 file outside the repository. A deterministic scrubber strips tokens and account identifiers before anything is logged, printed, or sent to an LLM. The LLM receives only sanitised numeric state.


---


## 11. Risks and Mitigations


- **Stale or desynchronised option quotes**: quote-level timestamps, 180-second staleness bound, put-call parity abort, `INSUFFICIENT DATA` on staleness.
- **Confabulated LLM evidence**: mandatory evidence-field citations; deterministic post-LLM verifier re-checks every figure and Part A result; rupee and Greek figures rendered from state, not prose.
- **One model arguing with itself (Phase 2)**: structurally opposed prompts, optional heterogeneous providers, his side-by-side quality review; drop the layer if it adds nothing.
- **Broker rate limits (429)**: provisional unverified request-rate config (default 1 per 2.5 seconds until the documented limit is verified), single-threaded limiter, disk caching, narrow strike window; any 429 stops the run immediately and reports, with no retry and no limit probing.
- **Gap risk beyond the Rs 2,000 stop**: contractual max loss always displayed next to the stop, overnight-gap stress in the risk table, and the settled Rs 10,000 per-trade contractual-loss ceiling enforced on every candidate.
- **Holiday / expiry / lot-size drift**: zero hardcoded weekdays or lot sizes; live contract master every run.
- **IVP unavailable early on**: requirement stays mandatory for live readiness; outputs are labelled research-only with IVP `UNKNOWN` until a verified history exists.


---


## 12. Ordered Build List


1. **Phase 0a**: project skeleton, secret-scrubbing config, contract-master/expiry/calendar parsers, NSE EOD volatility-history loader.
2. **Phase 0b**: pricing, IV, Greeks engine with the QA suite (Section 8.1).
3. **Phase 0c**: candle hygiene, two-track detectors, candidate generator, Part A gatekeeper (Section 8.3 tests), offline report renderer. **Trader Stop-or-Go.**
4. **Phase 1a**: read-only Kotak Neo adapter with timestamp extraction and cache, wired to the five approved run times at the provisional request-rate config; any 429 stops the run.
5. **Phase 1b**: single-LLM synthesis node, citation verifier, audit log. **Trader Stop-or-Go.**
6. **Phase 2 (optional)**: Bull/Bear debate layer, trader quality review. **Trader Stop-or-Go.**


---


## Decisions for the Trader


**Settled 6 Nov 2026**: maximum contractual loss per trade is Rs 10,000, a ceiling (Section 6.3). Run schedule: five checks per market day at 10:15, 11:15, 12:15, 13:15 and 14:15 IST, fetching the Kotak data each run needs, alerting only on a qualifying setup (Section 10).


**Answer before Phase 1 goes live**:
1. LLM budget per decision/month and authorised models (a local model avoids API bills but is not zero-cost).
2. Capital/margin ceilings the system should assume and report against.


**Can wait** (answer any time before they matter):
3. Which Section 5C history items are current preferences (trigger variants, 20-40 DTE window, one cycle per expiry, defining "stretched", the scalp exception).
4. How heuristic challenges should be presented (inline vs separate between-trade note).
5. Daily chain snapshot archive beyond what the runs themselves need: approve or decline (declined by default).


**Authority gaps this plan does not fill**: any order capability, LLM API spend beyond a budget he sets, any schedule or instrument widening beyond the five approved daily NIFTY runs, and any data capture beyond what those runs need. Each requires the trader's explicit separate approval. This document grants none.