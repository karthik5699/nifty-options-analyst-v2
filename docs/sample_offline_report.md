# NIFTY 50 Options Research & Strategy Analyst Report

## 1. Header & Data Freshness Banner
- **Verdict**: **SUGGESTION**
- **Readiness Status**: **LIVE-READY (All Part A & IVP Gates Verified)**
- **Run ID**: `run_20261106_101500_5f7fb9`
- **Decision Timestamp (IST)**: `2026-11-06T10:15:00+05:30`
- **Data Mode**: `OFFLINE` (Scheduled Slot Approved: `True`)
- **NIFTY Spot**: `24200.00` (as-of `2026-11-06T10:14:40+05:30`, stale=`False`, source=`kotak_neo_scrip_master_fixture`)
- **Data Version Hash**: `410cb203281165c4` | **Config Hash**: `5f7fb995dc54670f`
- **Active Contract-Master Expiries (Zero Hardcoded Weekdays/Lot Sizes)**:
  - Expiry `2026-11-12`: **3 trading session(s) DTE**, **Lot Size = 75**, Holiday-Adjusted = `False`
  - Expiry `2026-11-19`: **8 trading session(s) DTE**, **Lot Size = 75**, Holiday-Adjusted = `False`
  - Expiry `2026-11-23`: **10 trading session(s) DTE**, **Lot Size = 50**, Holiday-Adjusted = `True` — *Monthly expiry shifted to Monday 23 Nov 2026 ahead of 24 Nov 2026 NSE holiday; lot size revised to 50.*
  - **Notice**: Lot-size change detected across active expiries; each structure uses its exact contract-master lot size.
- **Staleness Report**: Spot as-of 2026-11-06T10:14:40+05:30 (stale=False); Chain quotes [2026-11-06T10:14:35+05:30 .. 2026-11-06T10:14:35+05:30] (any_stale=False)

## 2. Evidence Summary (Keyed to `NiftyOptionsState`)
- **Completed-Candle Structure (`market_evidence.price_structure`)**: 1D Trend = `UPTREND` (HH+HL Repaired = `True`, First-Pullback Warning = `False`); 1H MA20 = `24076.50`, 1H EMA50 = `24031.00`; Stripped Forming Bars = `1`; Final 15m Bar (15:15-15:30 IST) Completed = `True`.
  - Last Completed 1H Bar: `2026-11-06T09:15:00+05:30` to `2026-11-06T10:15:00+05:30` (O=24160.00, H=24215.00, L=24095.00, C=24200.00).
  - Geometric Touch Check (`1H_MA20` @ `24076.50`): Touched = `False` on bar [24095.00, 24215.00], Wick Rejection = `NONE`.
  - Geometric Touch Check (`1H_EMA50` @ `24031.00`): Touched = `False` on bar [24095.00, 24215.00], Wick Rejection = `NONE`.
  - Geometric Touch Check (`PREV_1D_HIGH` @ `24240.00`): Touched = `False` on bar [24095.00, 24215.00], Wick Rejection = `NONE`.
  - Geometric Touch Check (`PREV_1D_LOW` @ `24020.00`): Touched = `False` on bar [24095.00, 24215.00], Wick Rejection = `NONE`.
  - Geometric Touch Check (`PUT_OI_WALL` @ `24000.00`): Touched = `False` on bar [24095.00, 24215.00], Wick Rejection = `NONE`.
  - Geometric Touch Check (`CALL_OI_WALL` @ `24500.00`): Touched = `False` on bar [24095.00, 24215.00], Wick Rejection = `NONE`.
- **Volatility & Skew (`market_evidence.volatility_state`)**: Constant-Maturity 30d ATM IV = `14.49%` (`LIVE_MID_INFERRED`), IVP 30d / 90d / 252d = `86.67` / `82.22` / `UNKNOWN` (95 EOD sessions), 20d RV = `5.07%`, IV-RV Spread = `+9.42%`, 25d Put-Call Skew = `+0.27 vol pts`, India VIX = `14.25` (Tag: `UNVERIFIED`).
- **Derivatives Positioning (`market_evidence.derivatives_positioning`, Expiry `2026-11-12`)**: PCR (OI) = `1.05`, PCR (Vol) = `1.05`, Put OI Wall = `24000` (210,000 OI), Call OI Wall = `24500` (185,000 OI), Max Pain = `24200`.
- **Events Within DTE (`market_evidence.event_calendar`, Tag: `UNVERIFIED`)**: 2026-11-12: India CPI Inflation Release (MEDIUM), 2026-11-23: Holiday-Shifted Monthly NIFTY Expiry (Ahead of 24 Nov Guru Nanak Jayanti) (HIGH) (Calendar File Age = `5.0 days`, Stale Warning = `False`).
- **Triggered Setup Detectors (`market_evidence.triggered_detectors`)**:
  - `NOVEL_HIGH_IV_BULLISH_PUT_WALL` [NOVEL_QUANT | `CONFIRMED_QUANT_REGIME` | BULLISH]: 90d IVP is 82.22 with 1D trend UPTREND and Put OI wall at 24000.0 (210000 contracts). Compares bullish credit spreads, debit spreads, and butterflies side by side. (Invalidation: `24000.00` on `COMPLETED_DAILY_CLOSE`).
  - `NOVEL_OI_BRACKETED_NEUTRAL_RANGE` [NOVEL_QUANT | `CONFIRMED_QUANT_REGIME` | NEUTRAL]: Spot (24200.0) is bracketed between Put OI wall (24000.0) and Call OI wall (24500.0) with 90d IVP=82.22. (Invalidation: `24500.00` on `COMPLETED_DAILY_CLOSE`).

## 3. Reasoning Summary & Deterministic Citation Verification
Spot is at 24200.0 (`contract_context.spot.value`) with 1D structure in UPTREND (`market_evidence.price_structure.trend_1d_state`) and completed 1H MA20 at 24076.5 (`market_evidence.price_structure.ma20_1h`). 90-session IVP stands at 82.22 (`market_evidence.volatility_state.ivp_90d`) with Put OI support at 24000.0 and Call OI resistance at 24500.0. Primary structure `BPS_2026-11-12_24000_23850` (BULL_PUT_SPREAD) passes all applicable Part A checks with leg-wise contractual max loss of Rs 8551.50 (within the Rs 10,000 per-trade ceiling), indicative managed stop of Rs 2000.00, and portfolio summed stop of Rs 2000.00 (within the Rs 10,000 summed-stop cap). Thesis is invalidated on a COMPLETED_DAILY_CLOSE beyond 24000.0.

- **Deterministic Post-LLM Verifier**: Checked `10` state citations; Unverified = `0` (`0.00%` unsupported rate, threshold `<= 2.00%`); Part A Re-verified = `True`; Verification Passed = `True`.

## 4. Primary Structure Specification
- **Structure ID**: `BPS_2026-11-12_24000_23850` (`BULL_PUT_SPREAD`, Direction: `BULLISH`)
- **Expiry & Sizing**: Expiry `2026-11-12` (3 trading sessions DTE), Lot Size = `75`, Lots = `1` (Total Quantity = `75`), Wing Width = `150 pts`.
- **Executable Entry**: Net `Credit` = `35.98 pts/unit` (after `1.00 pts` slippage buffer; Round-trip Statutory Charges = `Rs 114.49`).

### Legs Specification
| Side | Ratio | Strike | Type | Bid | Ask | Mid | IV (Mid) | Delta | Gamma | Daily Theta | Vega (1% Vol) | OI | Liq Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SHORT | 1x | 24000 | PE | 90.62 | 92.08 | 91.35 | 14.60% | -0.306 | 0.00076 | -13.00 | +11.09 | 210,000 | 83.8 |
| LONG | 1x | 23850 | PE | 52.78 | 53.64 | 53.21 | 14.66% | -0.203 | 0.00061 | -10.51 | +8.92 | 105,000 | 81.2 |

### The Four Distinct Rupee Quantities (Side by Side — Never Merged)
| Rupee Quantity | Value (Rs) | Binding Ceiling / Cap | Source & Operational Warning |
| :--- | :--- | :--- | :--- |
| **1. Contractual Maximum Loss** | **Rs 8,551.50** | <= Rs 10,000.00 ceiling (6 Nov 2026) | Method: `LEG_WISE_PIECEWISE_EXPIRY_PAYOFF`. **Gap Warning**: Worst-case expiry loss if market gaps through the Rs 2,000 stop (3% overnight gap stress = Rs 8,551.50). |
| **2. Margin / Capital Estimate** | **Rs 28,516.50** | Reported from source | Source: `EXCHANGE_FORMULA_ESTIMATE` (never conflated with contractual max loss or stop). |
| **3. Planned Per-Position Managed Stop** | **Rs 2,000.00** | <= Rs 2,000.00 hard rule | Indicative spread mark ~`62.65 pts` (indicative index ~`23942.3`). INDICATIVE SUGGESTION-TIME MARK ONLY (model/IV/time-dependent). The operative Rs 2,000 stop is always computed from the actual executed fill (real fill prices, quantity, and charges) at execution time. Gaps and fast tapes can fill worse than the stop; the stop is not a guaranteed loss limit. |
| **4. Portfolio Summed-Stop Impact** | **Rs 2,000.00** | <= Rs 10,000.00 portfolio cap | Existing open stops (`Rs 0.00`) + this trade (`Rs 2,000.00`). **Every position counted in full; zero correlation netting.** |

### Breakevens, Modelled PoP, Written Invalidation & Management Plan
- **Breakevens (Gross)**: `[23964.02]` | **Breakevens (Net of Statutory Charges)**: `[23965.55]`
- **Modelled Terminal PoP**: Lognormal = `71.22%`, Empirical = `88.04%` — *MODEL_DEPENDENT_EXPIRY_HORIZON_ONLY: Terminal payoff probability under (1) IV-implied lognormal and (2) empirical EOD return distribution. Does not reflect stop-managed or invalidation-managed early exits.*
- **Written Invalidation (22 Sep 2026 Rule)**: Index Level `24000.00` on `COMPLETED_DAILY_CLOSE` basis — Completed daily close below Put OI Wall (24000.0) invalidates bullish support.
- **Management Plan**: 1. Exit immediately if position mark-to-market loss reaches Rs 2000 (operative stop computed from actual fill; indicative spread value ~62.6 pts). 2. Exit on completed-candle close beyond invalidation level 24000.0 (COMPLETED_DAILY_CLOSE). 3. Review at 50-60% of max profit (Rs 1484). 4. Do not carry into 1 DTE session if any short leg has |delta| >= 0.25 (26 Oct 2026 rule).

## 5. Part A Binding Rule Check Matrix
Showing Part A matrix for `BPS_2026-11-12_24000_23850` (Overall Status: **PASS**):

| Rule ID | Binding Rule Description | Source Date | Status | Exact Evidence Used |
| :--- | :--- | :---: | :---: | :--- |
| `RULE_1_DEFINED_RISK_ONLY` | Defined risk only (bounded contractual max loss by construction) | Standing Rule | **PASS** | is_defined_risk=True, legs=2 (long=1, short=1), contractual_max_loss_rs=Rs 8551.50. |
| `RULE_2_PER_POSITION_STOP_2000_RS` | Rs 2,000 per-position managed stop (indicative at suggestion; operative from fill) | Standing Rule | **PASS** | planned_stop_rs=Rs 2000.00 <= limit Rs 2000.00. |
| `RULE_3_PORTFOLIO_SUMMED_STOP_10000_RS` | Rs 10,000 portfolio summed-stop cap (every position counted in full, no netting) | Standing Rule | **PASS** | open_positions_count=0, existing_summed_stops=Rs 0.00, proposed_stop=Rs 2000.00, total_summed_stop=Rs 2000.00 (cap Rs 10000.00). |
| `RULE_4_CONTRACTUAL_MAX_LOSS_10000_RS` | Rs 10,000 per-trade contractual maximum loss ceiling (piecewise expiry payoff) | 6 Nov 2026 | **PASS** | contractual_max_loss_rs=Rs 8551.50 vs ceiling Rs 10000.00 (computed leg-wise). |
| `RULE_5_EXPIRY_1DTE_SHORT_DELTA` | 1 DTE short-leg |delta| < 0.25 evaluated leg-by-leg | 26 Oct 2026 | **NOT_APPLICABLE** | dte_trading_sessions=3 (> 1 DTE session); short-leg max |delta|=0.306 monitored for 1 DTE gate. |
| `RULE_6_WRITTEN_INVALIDATION_BEFORE_ENTRY` | Written index invalidation level on completed-close basis before entry | 22 Sep 2026 | **PASS** | invalidation_level=24000.0, basis=COMPLETED_DAILY_CLOSE, justification='Completed daily close below Put OI Wall (24000.0) invalidates bullish support.'. |
| `RULE_7_COMPLETED_CANDLE_AND_TOUCH_HYGIENE` | Signal hygiene: completed candles only & geometrically verified touches | Standing Rule | **PASS** | last_completed_1h_close=2026-11-06T10:15:00+05:30, stripped_forming_bars=1, verified_touches_checked=6. |
| `RULE_8_IVP_AND_LIQUIDITY_GATES` | Mandatory IVP and per-leg liquidity gates | Standing Rule | **PASS** | All 2 legs pass liquidity (min score=81.2), chain_qa_passed=True, ivp_30d=86.67, ivp_90d=82.22. |
| `RULE_9_NO_AUTOMATIC_RULE_CHANGES` | Rules change only between trades, only by trader | Standing Rule | **PASS** | No automatic rule mutation proposed. |
| `RULE_10_HUMAN_FINAL_DECISION` | Advisory output only; trader approves or rejects; zero execution capability | Standing Rule | **PASS** | System is strictly read-only with zero broker write or order-placement methods. |

## 6. Open Defined-Risk Trade-Off Comparison (Credit vs Debit vs Butterfly vs Condor)
Credit spreads (Bull Put Spread / Bear Call Spread / Iron Condor) and debit spreads / butterflies were generated and evaluated on equal footing without style filtering. See Section 6 table for exact cost, max loss, breakeven, PoP, and vega comparison.

| Structure ID | Type & Direction | Net Credit/Debit | Contractual Max Loss | Planned Stop | Net Breakeven(s) | PoP (Lognorm / Emp) | Net Vega (Rs) | Part A Status | Matches Part B Style |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `BPS_2026-11-12_24000_23800` | BULL_PUT_SPREAD (BULLISH) | Cr 45.52 pts | Rs 11,586.00 | Rs 2,000.00 | 23956.0 | 72.6% / 92.4% | Rs -223.58 | **FAIL** | Yes (Habitual) |
| `BPS_2026-11-12_24000_23850` | BULL_PUT_SPREAD (BULLISH) | Cr 35.98 pts | Rs 8,551.50 | Rs 2,000.00 | 23965.5 | 71.2% / 88.0% | Rs -162.50 | **PASS** | Yes (Habitual) |
| `BCS_2026-11-12_24300_24500` | BULL_CALL_SPREAD (BULLISH) | Dr 73.87 pts | Rs 5,540.25 | Rs 2,000.00 | 24375.5 | 37.5% / 37.0% | Rs +138.45 | **PASS** | No (Open Compare) |
| `BULL_BFLY_2026-11-12_24200_24350_24500` | BULLISH_BUTTERFLY (BULLISH) | Dr 24.96 pts | Rs 1,872.00 | Rs 1,872.00 | 24227.8, 24472.2 | 19.9% / 26.1% | Rs -89.58 | **PASS** | No (Open Compare) |
| `BEAR_CS_2026-11-12_24450_24600` | BEAR_CALL_SPREAD (BEARISH) | Cr 36.76 pts | Rs 8,493.00 | Rs 2,000.00 | 24485.2 | 71.2% / 76.1% | Rs -159.97 | **PASS** | Yes (Habitual) |
| `BEAR_PS_2026-11-12_24200_24050` | BEAR_PUT_SPREAD (BEARISH) | Dr 64.67 pts | Rs 4,850.25 | Rs 2,000.00 | 24133.6 | 42.1% / 38.0% | Rs +69.22 | **PASS** | No (Open Compare) |
| `IC_2026-11-12_23600_23750_24700_24800` | IRON_CONDOR (NEUTRAL) | Cr 26.93 pts | Rs 9,230.25 | Rs 2,000.00 | 23725.8, 24724.2 | 72.0% / 100.0% | Rs -299.96 | **PASS** | Yes (Habitual) |

## 7. Between-Trade Research Notes & Confidence
- **Research / Heuristic Observation**: RESEARCH NOTE (Between-Trade Context): Habitual 200-pt width structure `BPS_2026-11-12_24000_23800` (lot_size=75) has a leg-wise contractual max loss of Rs 11586.00, which exceeds the settled Rs 10,000 per-trade contractual max loss ceiling (6 Nov 2026) unless net credit exceeds 66.7 pts or wing width is tightened to 150 pts.
- **Confidence & Readiness Note**: LIVE-READY (IVP verified and all Part A checks PASS).

---
*Advisory Notice: This system is strictly read-only and has zero order execution capability. All trade decisions and execution rests solely with the human trader.*