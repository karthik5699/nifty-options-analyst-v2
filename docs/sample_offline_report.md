# NIFTY 50 Options Research & Strategy Analyst Report

## 1. Header & Data Freshness Banner
- **Verdict**: **SUGGESTION**
- **Readiness Status**: **LIVE-READY (All Part A & IVP Gates Verified)**
- **Run ID**: `run_20261106_101500_4eacc9`
- **Decision Timestamp (IST)**: `2026-11-06T10:15:00+05:30`
- **Data Mode**: `OFFLINE` (Scheduled Slot Approved: `True`)
- **NIFTY Spot**: `24200.00` (as-of `2026-11-06T10:14:40+05:30`, stale=`False`, source=`kotak_neo_scrip_master_fixture`)
- **Data Version Hash**: `d165528354990f5e` | **Config Hash**: `4eacc966c50b61f9`
- **Active Contract-Master Expiries (Zero Hardcoded Weekdays/Lot Sizes)**:
  - Expiry `2026-11-12`: **3 trading session(s) DTE**, **Lot Size = 75**, Holiday-Adjusted = `False`
  - Expiry `2026-11-19`: **8 trading session(s) DTE**, **Lot Size = 75**, Holiday-Adjusted = `False`
  - Expiry `2026-11-23`: **10 trading session(s) DTE**, **Lot Size = 50**, Holiday-Adjusted = `True` — *Monthly expiry shifted to Monday 23 Nov 2026 ahead of 24 Nov 2026 NSE holiday; lot size revised to 50.*
  - **Notice**: Lot-size change detected across active expiries; each structure uses its exact contract-master lot size.
- **Staleness Report**: Spot as-of 2026-11-06T10:14:40+05:30 (stale=False); Chain quotes across 3 expiry(s) [2026-11-06T10:14:35+05:30 .. 2026-11-06T10:14:35+05:30] (any_stale=False)

## 2. Evidence Summary (Keyed to `NiftyOptionsState`)
- **Completed-Candle Structure & Technicals (`market_evidence.price_structure`)**: 1D Trend = `UPTREND` (HH+HL Repaired = `True`, First-Pullback Warning = `False`); 1H MA20 = `24076.50`, 1H EMA50 = `24031.00`, 1H RSI(14) = `100.00`, 1H ATR(14) = `50.36 pts`, 1D ATR(14) = `220.00 pts`, 1H Stretch Z-Score = `+2.35 SD`; Stripped Forming Bars = `1`; Final 15m Bar (15:15-15:30 IST) Completed = `True`.
  - Last Completed 1H Bar: `2026-11-06T09:15:00+05:30` to `2026-11-06T10:15:00+05:30` (O=24160.00, H=24215.00, L=24095.00, C=24200.00).
  - Geometric Touch Check (`1H_MA20` @ `24076.50`): Touched = `False` on bar [24095.00, 24215.00], Wick Rejection = `NONE`.
  - Geometric Touch Check (`1H_EMA50` @ `24031.00`): Touched = `False` on bar [24095.00, 24215.00], Wick Rejection = `NONE`.
  - Geometric Touch Check (`PREV_1D_HIGH` @ `24240.00`): Touched = `False` on bar [24095.00, 24215.00], Wick Rejection = `NONE`.
  - Geometric Touch Check (`PREV_1D_LOW` @ `24020.00`): Touched = `False` on bar [24095.00, 24215.00], Wick Rejection = `NONE`.
  - Geometric Touch Check (`PUT_OI_WALL` @ `24000.00`): Touched = `False` on bar [24095.00, 24215.00], Wick Rejection = `NONE`.
  - Geometric Touch Check (`CALL_OI_WALL` @ `24500.00`): Touched = `False` on bar [24095.00, 24215.00], Wick Rejection = `NONE`.
- **Volatility, Term Structure & Skew (`market_evidence.volatility_state`)**: Live Constant-Maturity 30d IV = `14.47%` (`LIVE_VARIANCE_INTERPOLATED_CM30`), ATM IV by Expiry = `2026-11-12: 14.49%, 2026-11-19: 14.50%, 2026-11-23: 14.49%`, Term Structure = `FLAT` (Near-minus-Far Slope = `+0.00 vol pts`), Latest Completed EOD cm30 IV (percentiled series) = `0.103`, IVP 30d / 90d = `6.67` / `4.44`; 252d IVP = `UNKNOWN` (UNAVAILABLE: 95 verified EOD sessions < 252, no substitute), 20d RV = `5.07%`, IV-RV Spread = `+9.40%`, 25d Put-Call Skew = `+0.27 vol pts`, India VIX = `14.25` (Tag: `UNVERIFIED`).
- **Derivatives Positioning (`market_evidence.derivatives_positioning`, Expiry `2026-11-12`)**: PCR (OI) = `1.05`, PCR (Vol) = `1.05`, Put OI Wall = `24000` (210,000 OI), Call OI Wall = `24500` (185,000 OI), Max Pain = `24200`.
- **Events Within DTE (`market_evidence.event_calendar`, Tag: `UNVERIFIED`)**: 2026-11-12: India CPI Inflation Release (MEDIUM), 2026-11-23: Holiday-Shifted Monthly NIFTY Expiry (Ahead of 24 Nov Guru Nanak Jayanti) (HIGH) (Calendar File Age = `5.0 days`, Stale Warning = `False`).
- **Triggered Setup Detectors (`market_evidence.triggered_detectors`)**:
  - `NOVEL_LOW_IV_COMPRESSION_BULLISH` [NOVEL_QUANT | `CONFIRMED_QUANT_REGIME` | BULLISH]: 90d IVP is compressed (4.44) with positive trend structure (1H RSI=100.0); long-vega debit structures (Bull Call Spread) avoid short-vega expansion risk. (Invalidation: `24031.00` on `COMPLETED_DAILY_CLOSE`).
  - `NOVEL_STRETCH_METRIC_CANDIDATE` [NOVEL_QUANT | `DEFINITION_OPEN` | BEARISH]: Spot (24200.0) is 123.5 pts (+2.35 SD) from 1H MA20 (24076.5). Reported as DEFINITION_OPEN until trader settles 'stretched' definition. (Invalidation: `24350.00` on `COMPLETED_DAILY_CLOSE`).

## 3. Multi-Persona Debate, Risk Triad & Consolidator Recommendation (`TradingAgents` Single-Pass)
### Adversarial Bull / Bear / Volatility-Regime Debate
- **Bull Researcher**: Spot at 24200.00 sits above Put OI wall support at 24000 (210,000 contracts, PCR OI 1.05) with 1D structure `UPTREND` (HH+HL repaired=True), 1H RSI=100.0, holding relative to 1H MA20 (24076.50) and 1H EMA50 (24031.00).
- **Bear / Red-Team Researcher**: Overhead Call OI wall at 24500 (185,000 contracts) caps upside expansion; 1H ATR is 50.4 pts (1D ATR 220.0 pts, stretch z-score +2.35 SD). Upcoming catalyst(s) within DTE (India CPI Inflation Release on 2026-11-12, Holiday-Shifted Monthly NIFTY Expiry (Ahead of 24 Nov Guru Nanak Jayanti) on 2026-11-23) add jump risk.
- **Volatility & Skew Specialist**: CM30 IV=14.47% (FLAT, near-minus-far slope +0.00 vol pts), 90d IVP=4.44, 20d RV=5.07% (IV-RV=+9.40 vol pts), 25d Put-Call skew=+0.27 vol pts. Compressed 90d IVP favours defined-risk debit verticals or low-cost butterflies to avoid short-vega expansion risk.
- **Debate Rebuttal & Dominant Regime (`BULLISH`)**: Synthesis rebuttal: Dominant regime resolved as `BULLISH`. Technical structure (`UPTREND`, RSI 100.0) and options surface (90d IVP=4.44, skew=+0.27 vol pts) align only when invalidation is anchored beyond 1H ATR (50.4 pts) noise and capped by the Rs 2,000 managed stop.

### Options Risk Management Triad
- **Convexity & Edge Advocate**: Champions `BCS_2026-11-19_24350_24550` (BULL_CALL_SPREAD, 8 DTE): highest Tech+Options Score 70.3/100, max profit Rs 8,774.23 vs contractual bound Rs 6,107.25 (PoP 39.0%).
- **Conservative Tail Guardian**: Flags 3% overnight gap + vol-jump MTM stress on `BCS_2026-11-19_24350_24550` at Rs 4,181.37 (contractual cap Rs 6,107.25, max short |delta|=0.336); lowest gap-stress alternative is `IBFLY_2026-11-23_24150_24300_24450` at Rs 638.05. Portfolio summed stop post-trade = Rs 2,000.00/10,000.
- **Neutral Desk Arbiter**: Endorses `BCS_2026-11-19_24350_24550` (2026-11-19, lot_size=75) combining Part A PASS compliance, Tech+Options score 70.3/100, and clear completed-candle invalidation at 24031.00.

### Chief Options Strategist (Consolidator Verdict)
- **Consolidator Pick**: `BCS_2026-11-19_24350_24550` (Conviction: **HIGH**)
- **Consolidated Rationale**: Consolidator selects `BCS_2026-11-19_24350_24550` (BULL_CALL_SPREAD on `2026-11-19`, 8 DTE, Tech+Opt Score 70.3/100) with HIGH conviction. Technical `UPTREND` structure (1H RSI 100.0) and options surface (90d IVP 4.44, 25d skew +0.27 vol pts, FLAT) align with OI wall protection (24000 PE / 24500 CE), while satisfying all binding Part A limits (contractual bound Rs 6,107.25, managed stop Rs 2,000.00, summed portfolio stop Rs 2,000.00).

### Grounded Reasoning & Deterministic Verifier Audit
Spot is at 24200.0 (`contract_context.spot.value`) with 1D structure in `UPTREND` (`market_evidence.price_structure.trend_1d_state`), completed 1H MA20 at 24076.5 (`market_evidence.price_structure.ma20_1h`), and 1H RSI(14) at 100.0 (`market_evidence.price_structure.rsi_14_1h`). 90-session IVP stands at 4.44 (`market_evidence.volatility_state.ivp_90d`) in `FLAT` term structure with Put OI support at 24000.0 and Call OI resistance at 24500.0. Following Bull/Bear/Vol debate and Risk Triad review, Consolidator selects `BCS_2026-11-19_24350_24550` (BULL_CALL_SPREAD, `2026-11-19`, Tech+Options score 70.28/100) with contractual payoff bound Rs 6107.25 (<= Rs 10,000 ceiling), managed stop Rs 2000.00, and portfolio summed stop Rs 2000.00. Thesis invalidates on a COMPLETED_DAILY_CLOSE beyond 24031.0.

- **Deterministic Two-Way Verifier**: Checked `12` state citations (Unverified = `0`, unsupported rate `0.00%`, threshold `<= 2.00%`); Prose Numeric/ID Claims Checked = `21` (Uncited/Confabulated = `0`); Part A & Consolidator Re-verified = `True`; Verification Passed = `True`.

## 4. Primary Structure Specification
- **Structure ID**: `BCS_2026-11-19_24350_24550` (`BULL_CALL_SPREAD`, Direction: `BULLISH`, Tech+Options Score: **70.3/100**)
- **Expiry & Sizing**: Expiry `2026-11-19` (8 trading sessions DTE), Lot Size = `75`, Lots = `1` (Total Quantity = `75`), Wing Width = `200 pts`.
- **Executable Entry**: Net `Debit` = `81.43 pts/unit` (after `1.00 pts` slippage buffer; Non-STT Execution Fees = `Rs 118.52`).

### Legs Specification
- **LONG 1x 24350 CE**: bid 221.89 / ask 224.89 / mid 223.39; IV (mid) 14.46%; delta +0.450, gamma 0.00059, daily theta -9.94, vega (1% vol) +18.24; OI 95,000; book depth 25 lots; liquidity score 83.0.
- **SHORT 1x 24550 CE**: bid 144.46 / ask 146.80 / mid 145.63; IV (mid) 14.39%; delta +0.336, gamma 0.00055, daily theta -9.12, vega (1% vol) +16.81; OI 95,000; book depth 25 lots; liquidity score 80.9.

### The Four Distinct Rupee Quantities (Never Merged)
- **1. Contractual Max Loss (Piecewise Expiry Payoff Bound): Rs 6,107.25** against the Rs 10,000.00 ceiling. Method: `LEG_WISE_PIECEWISE_EXPIRY_PAYOFF: exact contractual worst case Rs 6,107.25 at spot 0 (non-STT execution fees Rs 118.52)`. **Gap Warning**: pre-expiry 3% overnight gap + vol-jump MTM stress = `Rs 4,181.37`.
- **2. Margin / Capital Estimate: Rs 26,072.25** (source: `EXCHANGE_FORMULA_ESTIMATE`; never conflated with contractual max loss or stop).
- **3. Planned Per-Position Managed Stop: Rs 2,000.00** against the Rs 2,000.00 hard rule. Indicative spread mark ~`54.76 pts` (Black-76 spot trigger ~`23988.0`). INDICATIVE SUGGESTION-TIME BLACK-76 MTM MARK ONLY (model/IV/time-dependent). The operative Rs 2,000 stop is always computed from the actual executed fill (real fill prices and quantity) at execution time. Gaps and fast tapes can fill worse than the stop; the stop is not a guaranteed loss limit.
- **4. Portfolio Summed-Stop Impact: Rs 2,000.00** against the Rs 10,000.00 portfolio cap. Existing open stops (`Rs 0.00`) + this trade (`Rs 2,000.00`). **Every position counted in full; zero correlation netting.**

### Breakevens, Modelled PoP, Written Invalidation & Management Plan
- **Breakevens (Gross)**: `[24431.43]` | **Breakevens (Net of Execution Fees)**: `[24433.01]`
- **Modelled Terminal PoP**: Lognormal = `38.97%`, Empirical = `43.68%` — *MODEL_DEPENDENT_EXPIRY_HORIZON_ONLY: Terminal payoff probability under (1) IV-implied lognormal and (2) empirical EOD return distribution. Does not reflect stop-managed or invalidation-managed early exits.*
- **Written Invalidation (22 Sep 2026 Rule)**: Index Level `24031.00` on `COMPLETED_DAILY_CLOSE` basis — Completed daily close below 24031.0 negates upside expansion thesis.
- **Management Plan**: 1. Exit immediately if position mark-to-market loss reaches Rs 2000 (operative stop computed from actual fill; indicative spread value ~54.8 pts, Black-76 spot trigger ~23988.0). 2. Exit on completed-candle close beyond invalidation level 24031.0 (COMPLETED_DAILY_CLOSE). 3. Review at 50-60% of max profit (Rs 4826). 4. Do not carry into 1 DTE session if any short leg has |delta| >= 0.25 (26 Oct 2026 rule).

## 5. Part A Binding Rule Check Matrix
Showing Part A matrix for `BCS_2026-11-19_24350_24550` (Overall Status: **PASS**):

- `RULE_1_DEFINED_RISK_ONLY` - Defined risk only (bounded contractual max loss by construction) (Standing Rule): **PASS**. Evidence: is_defined_risk=True, legs=2 (long=1, short=1; both >= 1 required), contractual payoff bound = Rs 6107.25.
- `RULE_2_PER_POSITION_STOP_2000_RS` - Rs 2,000 per-position managed stop (indicative at suggestion; operative from fill) (Standing Rule): **PASS**. Evidence: planned_stop_rs=Rs 2000.00 <= limit Rs 2000.00.
- `RULE_3_PORTFOLIO_SUMMED_STOP_10000_RS` - Rs 10,000 portfolio summed-stop cap (every position counted in full, no netting) (Standing Rule): **PASS**. Evidence: open_positions_count=0, existing_summed_stops=Rs 0.00, proposed_stop=Rs 2000.00, total_summed_stop=Rs 2000.00 (cap Rs 10000.00).
- `RULE_4_CONTRACTUAL_MAX_LOSS_10000_RS` - Rs 10,000 per-trade contractual maximum loss ceiling (piecewise expiry payoff bound) (6 Nov 2026): **PASS**. Evidence: contractual_max_loss_rs = Rs 6107.25 vs ceiling Rs 10000.00.
- `RULE_5_EXPIRY_1DTE_SHORT_DELTA` - 1 DTE short-leg |delta| < 0.25 evaluated leg-by-leg (26 Oct 2026): **NOT_APPLICABLE**. Evidence: dte_trading_sessions=8 (> 1 DTE session); short-leg max |delta|=0.336 monitored for 1 DTE gate.
- `RULE_6_WRITTEN_INVALIDATION_BEFORE_ENTRY` - Written index invalidation level on completed-close basis before entry (22 Sep 2026): **PASS**. Evidence: invalidation_level=24031.0, basis=COMPLETED_DAILY_CLOSE, justification='Completed daily close below 24031.0 negates upside expansion thesis.'.
- `RULE_7_COMPLETED_CANDLE_AND_TOUCH_HYGIENE` - Signal hygiene: completed candles only & geometrically verified touches (Standing Rule): **PASS**. Evidence: last_completed_1h_close=2026-11-06T10:15:00+05:30, stripped_forming_bars=1, verified_touches_checked=6.
- `RULE_8_IVP_AND_LIQUIDITY_GATES` - Mandatory IVP and per-leg liquidity gates (Standing Rule): **PASS**. Evidence: All 2 legs pass liquidity (min score=80.9), chain_qa_passed=True, ivp_30d=6.67, ivp_90d=4.44.
- `RULE_9_NO_AUTOMATIC_RULE_CHANGES` - Rules change only between trades, only by trader (Standing Rule): **PASS**. Evidence: No automatic rule mutation proposed.
- `RULE_10_HUMAN_FINAL_DECISION` - Advisory output only; trader approves or rejects; zero execution capability (Standing Rule): **PASS**. Evidence: System is strictly read-only with zero broker write or order-placement methods.

## 6. Open Defined-Risk Trade-Off Comparison (Multi-Expiry Credit vs Debit vs Butterfly vs Condor)
All defined-risk families (Bull/Bear Credit & Debit Verticals, Iron Condors, Iron Butterflies, Directional Butterflies, and Legged Condor Components) were priced and scored solely on technical alignment and options surface edge (STT excluded per trader directive).

- `BCS_2026-11-19_24350_24550` - BULL_CALL_SPREAD (BULLISH, 2026-11-19, 8 DTE, lot=75): Tech+Opt Score **70.3/100**; net Debit 81.43 pts; contractual max loss Rs 6,107.25; planned stop Rs 2,000.00; net BE 24433.0; PoP (lognormal / empirical) 39.0% / 43.7%; net vega Rs +106.94; Part A **PASS** (Open Compare).
- `BCS_2026-11-23_24400_24600` - BULL_CALL_SPREAD (BULLISH, 2026-11-23, 10 DTE, lot=50): Tech+Opt Score **70.3/100**; net Debit 81.07 pts; contractual max loss Rs 4,053.50; planned stop Rs 2,000.00; net BE 24483.3; PoP (lognormal / empirical) 39.0% / 42.4%; net vega Rs +72.57; Part A **PASS** (Open Compare).
- `BCS_2026-11-12_24300_24500` - BULL_CALL_SPREAD (BULLISH, 2026-11-12, 3 DTE, lot=75): Tech+Opt Score **70.0/100**; net Debit 73.87 pts; contractual max loss Rs 5,540.25; planned stop Rs 2,000.00; net BE 24375.3; PoP (lognormal / empirical) 37.5% / 38.0%; net vega Rs +138.45; Part A **PASS** (Open Compare).
- `BPS_2026-11-23_23900_23700` - BULL_PUT_SPREAD (BULLISH, 2026-11-23, 10 DTE, lot=50): Tech+Opt Score **59.3/100**; net Credit 49.81 pts; contractual max loss Rs 7,509.50; planned stop Rs 2,000.00; net BE 23852.3; PoP (lognormal / empirical) 71.2% / 100.0%; net vega Rs -140.29; Part A **PASS** (Habitual).
- `BPS_2026-11-12_24000_23850` - BULL_PUT_SPREAD (BULLISH, 2026-11-12, 3 DTE, lot=75): Tech+Opt Score **59.1/100**; net Credit 35.98 pts; contractual max loss Rs 8,551.50; planned stop Rs 2,000.00; net BE 23965.4; PoP (lognormal / empirical) 71.2% / 88.0%; net vega Rs -162.50; Part A **PASS** (Habitual).
- `BPS_2026-11-23_23900_23750` - BULL_PUT_SPREAD (BULLISH, 2026-11-23, 10 DTE, lot=50): Tech+Opt Score **59.1/100**; net Credit 38.07 pts; contractual max loss Rs 5,596.50; planned stop Rs 2,000.00; net BE 23864.0; PoP (lognormal / empirical) 69.8% / 100.0%; net vega Rs -102.29; Part A **PASS** (Habitual).
- `BPS_2026-11-19_23900_23750` - BULL_PUT_SPREAD (BULLISH, 2026-11-19, 8 DTE, lot=75): Tech+Opt Score **59.1/100**; net Credit 35.79 pts; contractual max loss Rs 8,565.75; planned stop Rs 2,000.00; net BE 23865.7; PoP (lognormal / empirical) 71.2% / 81.6%; net vega Rs -161.68; Part A **PASS** (Habitual).
- `LEG_IC_PE_2026-11-12_23850_23700` - LEGGED_IRON_CONDOR_COMPONENT (BULLISH, 2026-11-12, 3 DTE, lot=75): Tech+Opt Score **59.0/100**; net Credit 22.67 pts; contractual max loss Rs 9,549.75; planned stop Rs 2,000.00; net BE 23828.7; PoP (lognormal / empirical) 81.0% / 100.0%; net vega Rs -184.18; Part A **PASS** (Habitual).
- `LEG_IC_PE_2026-11-19_23700_23600` - LEGGED_IRON_CONDOR_COMPONENT (BULLISH, 2026-11-19, 8 DTE, lot=75): Tech+Opt Score **58.9/100**; net Credit 15.97 pts; contractual max loss Rs 6,302.25; planned stop Rs 2,000.00; net BE 23685.4; PoP (lognormal / empirical) 80.0% / 100.0%; net vega Rs -122.17; Part A **PASS** (Habitual).
- `LEG_IC_PE_2026-11-23_23650_23600` - LEGGED_IRON_CONDOR_COMPONENT (BULLISH, 2026-11-23, 10 DTE, lot=50): Tech+Opt Score **58.4/100**; net Credit 7.40 pts; contractual max loss Rs 2,130.00; planned stop Rs 2,000.00; net BE 23644.6; PoP (lognormal / empirical) 80.0% / 100.0%; net vega Rs -40.20; Part A **PASS** (Habitual).
- `BULL_BFLY_2026-11-12_24200_24350_24500` - BULLISH_BUTTERFLY (BULLISH, 2026-11-12, 3 DTE, lot=75): Tech+Opt Score **51.5/100**; net Debit 24.96 pts; contractual max loss Rs 1,872.00; planned stop Rs 1,872.00; net BE 24227.3, 24472.7; PoP (lognormal / empirical) 21.2% / 26.1%; net vega Rs -89.58; Part A **PASS** (Open Compare).
- `BULL_BFLY_2026-11-19_24200_24350_24500` - BULLISH_BUTTERFLY (BULLISH, 2026-11-19, 8 DTE, lot=75): Tech+Opt Score **50.4/100**; net Debit 21.06 pts; contractual max loss Rs 1,579.50; planned stop Rs 1,579.50; net BE 24223.7, 24476.3; PoP (lognormal / empirical) 14.2% / 12.6%; net vega Rs -67.10; Part A **PASS** (Open Compare).
- `BULL_BFLY_2026-11-23_24200_24350_24500` - BULLISH_BUTTERFLY (BULLISH, 2026-11-23, 10 DTE, lot=50): Tech+Opt Score **50.2/100**; net Debit 19.67 pts; contractual max loss Rs 983.50; planned stop Rs 983.50; net BE 24223.5, 24476.5; PoP (lognormal / empirical) 12.6% / 16.5%; net vega Rs -39.72; Part A **PASS** (Open Compare).
- `BEAR_PS_2026-11-23_24200_24050` - BEAR_PUT_SPREAD (BEARISH, 2026-11-23, 10 DTE, lot=50): Tech+Opt Score **48.1/100**; net Debit 66.61 pts; contractual max loss Rs 3,330.50; planned stop Rs 2,000.00; net BE 24131.1; PoP (lognormal / empirical) 43.6% / 32.9%; net vega Rs +42.82; Part A **PASS** (Open Compare).
- `BEAR_PS_2026-11-19_24200_24050` - BEAR_PUT_SPREAD (BEARISH, 2026-11-19, 8 DTE, lot=75): Tech+Opt Score **48.0/100**; net Debit 66.52 pts; contractual max loss Rs 4,989.00; planned stop Rs 2,000.00; net BE 24131.9; PoP (lognormal / empirical) 43.6% / 37.9%; net vega Rs +64.07; Part A **PASS** (Open Compare).
- `BEAR_PS_2026-11-12_24200_24050` - BEAR_PUT_SPREAD (BEARISH, 2026-11-12, 3 DTE, lot=75): Tech+Opt Score **47.8/100**; net Debit 64.67 pts; contractual max loss Rs 4,850.25; planned stop Rs 2,000.00; net BE 24133.8; PoP (lognormal / empirical) 42.1% / 38.0%; net vega Rs +69.22; Part A **PASS** (Open Compare).
- `BEAR_CS_2026-11-19_24600_24750` - BEAR_CALL_SPREAD (BEARISH, 2026-11-19, 8 DTE, lot=75): Tech+Opt Score **32.3/100**; net Credit 37.38 pts; contractual max loss Rs 8,446.50; planned stop Rs 2,000.00; net BE 24635.9; PoP (lognormal / empirical) 71.2% / 69.0%; net vega Rs -156.14; Part A **PASS** (Habitual).
- `BEAR_CS_2026-11-12_24450_24600` - BEAR_CALL_SPREAD (BEARISH, 2026-11-12, 3 DTE, lot=75): Tech+Opt Score **32.2/100**; net Credit 36.76 pts; contractual max loss Rs 8,493.00; planned stop Rs 2,000.00; net BE 24485.4; PoP (lognormal / empirical) 71.2% / 76.1%; net vega Rs -159.97; Part A **PASS** (Habitual).
- `BEAR_BFLY_2026-11-12_23900_24050_24200` - BEARISH_BUTTERFLY (BEARISH, 2026-11-12, 3 DTE, lot=75): Tech+Opt Score **32.1/100**; net Debit 23.52 pts; contractual max loss Rs 1,764.00; planned stop Rs 1,764.00; net BE 23925.8, 24174.2; PoP (lognormal / empirical) 19.1% / 42.4%; net vega Rs -76.33; Part A **PASS** (Open Compare).
- `BEAR_CS_2026-11-23_24700_24800` - BEAR_CALL_SPREAD (BEARISH, 2026-11-23, 10 DTE, lot=50): Tech+Opt Score **32.1/100**; net Credit 23.80 pts; contractual max loss Rs 3,810.00; planned stop Rs 2,000.00; net BE 24721.7; PoP (lognormal / empirical) 72.6% / 74.1%; net vega Rs -69.29; Part A **PASS** (Habitual).
- `BEAR_BFLY_2026-11-19_23900_24050_24200` - BEARISH_BUTTERFLY (BEARISH, 2026-11-19, 8 DTE, lot=75): Tech+Opt Score **31.2/100**; net Debit 19.95 pts; contractual max loss Rs 1,496.25; planned stop Rs 1,496.25; net BE 23922.5, 24177.5; PoP (lognormal / empirical) 13.7% / 18.4%; net vega Rs -57.34; Part A **PASS** (Open Compare).
- `BEAR_BFLY_2026-11-23_23900_24050_24200` - BEARISH_BUTTERFLY (BEARISH, 2026-11-23, 10 DTE, lot=50): Tech+Opt Score **31.0/100**; net Debit 18.88 pts; contractual max loss Rs 944.00; planned stop Rs 944.00; net BE 23922.4, 24177.6; PoP (lognormal / empirical) 12.2% / 23.5%; net vega Rs -33.96; Part A **PASS** (Open Compare).
- `IC_2026-11-12_23600_23750_24700_24800` - IRON_CONDOR (NEUTRAL, 2026-11-12, 3 DTE, lot=75): Tech+Opt Score **30.5/100**; net Credit 26.93 pts; contractual max loss Rs 9,230.25; planned stop Rs 2,000.00; net BE 23725.7, 24724.3; PoP (lognormal / empirical) 72.0% / 100.0%; net vega Rs -299.96; Part A **PASS** (Habitual).
- `IBFLY_2026-11-12_24100_24250_24400` - IRON_BUTTERFLY (NEUTRAL, 2026-11-12, 3 DTE, lot=75): Tech+Opt Score **28.2/100**; net Credit 123.69 pts; contractual max loss Rs 1,973.25; planned stop Rs 1,973.25; net BE 24129.3, 24370.7; PoP (lognormal / empirical) 20.5% / 23.9%; net vega Rs -98.35; Part A **PASS** (Open Compare).
- `IBFLY_2026-11-19_24100_24250_24400` - IRON_BUTTERFLY (NEUTRAL, 2026-11-19, 8 DTE, lot=75): Tech+Opt Score **27.6/100**; net Credit 128.28 pts; contractual max loss Rs 1,629.00; planned stop Rs 1,629.00; net BE 24125.0, 24375.0; PoP (lognormal / empirical) 15.8% / 14.9%; net vega Rs -68.24; Part A **PASS** (Open Compare).
- `IBFLY_2026-11-23_24150_24300_24450` - IRON_BUTTERFLY (NEUTRAL, 2026-11-23, 10 DTE, lot=50): Tech+Opt Score **27.2/100**; net Credit 129.81 pts; contractual max loss Rs 1,009.50; planned stop Rs 1,009.50; net BE 24174.9, 24425.1; PoP (lognormal / empirical) 12.7% / 16.5%; net vega Rs -40.01; Part A **PASS** (Open Compare).
- `BPS_2026-11-19_23900_23700` - BULL_PUT_SPREAD (BULLISH, 2026-11-19, 8 DTE, lot=75): Tech+Opt Score **59.2/100**; net Credit 46.42 pts; contractual max loss Rs 11,518.50; planned stop Rs 2,000.00; net BE 23855.0; PoP (lognormal / empirical) 72.6% / 83.9%; net vega Rs -220.98; Part A **FAIL** (Habitual).
- `BPS_2026-11-12_24000_23800` - BULL_PUT_SPREAD (BULLISH, 2026-11-12, 3 DTE, lot=75): Tech+Opt Score **59.1/100**; net Credit 45.52 pts; contractual max loss Rs 11,586.00; planned stop Rs 2,000.00; net BE 23955.9; PoP (lognormal / empirical) 72.6% / 92.4%; net vega Rs -223.58; Part A **FAIL** (Habitual).
- `BEAR_CS_2026-11-19_24600_24800` - BEAR_CALL_SPREAD (BEARISH, 2026-11-19, 8 DTE, lot=75): Tech+Opt Score **32.4/100**; net Credit 48.48 pts; contractual max loss Rs 11,364.00; planned stop Rs 2,000.00; net BE 24647.0; PoP (lognormal / empirical) 72.6% / 70.1%; net vega Rs -215.03; Part A **FAIL** (Habitual).
- `BEAR_CS_2026-11-12_24450_24650` - BEAR_CALL_SPREAD (BEARISH, 2026-11-12, 3 DTE, lot=75): Tech+Opt Score **32.2/100**; net Credit 46.50 pts; contractual max loss Rs 11,512.50; planned stop Rs 2,000.00; net BE 24495.1; PoP (lognormal / empirical) 72.6% / 77.2%; net vega Rs -221.41; Part A **FAIL** (Habitual).

## 7. Between-Trade Research Notes & Confidence
- **Research / Heuristic Observation**: RESEARCH NOTE (Between-Trade Context): Habitual 200-pt width structure `BPS_2026-11-12_24000_23800` (lot_size=75) has a contractual max loss of Rs 11586.00, exceeding the Rs 10,000 per-trade ceiling unless net credit exceeds 66.7 pts or wing width is 150 pts / smaller lot expiry.
- **Confidence & Readiness Note**: LIVE-READY (HIGH conviction; IVP verified and all Part A checks PASS).

---
*Advisory Notice: This system is strictly read-only and has zero order execution capability. All trade decisions and execution rests solely with the human trader.*