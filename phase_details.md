# Phase Details (3-7) — Archived from MEMORY.md

> These phases are COMPLETE. Kept for reference only.

## Phase 3 Sweep (2026-03-18) — script: `ai_context/phase3_backtest_sweep.py`
- **R012 (EURUSD macd_lc)**: `macd_lc only` wins. n=3,cd=20 OOS PF 20.48. Avg 16.11 (15/20 beat benchmark 12.79).
- **GJ010 (GBPJPY)**: `fsb_strong` — 20/20 beat benchmark. `macd_lc` top OOS PF 47.60 (low trade count). TV validate next.
- **EA012 (Napoleon)**: Score=4 always on Forex Renko. **STOP.**
- **EA013 (ESCGO)**: Best OOS PF 10.67 on 17 OOS trades. **INCONCLUSIVE.**
- **EA014 (Alpha Sniper)**: imb_threshold dead on Renko. Best OOS PF 7.33. **STOP.**
- Outputs: `r012_phase3_results.json`, `gj010_phase3_results.json`, `ea_phase3_creative_results.json`

## Phase 4 Sweep (2026-03-18) — scripts: `renko/bc_master_sweep_v2.py`, `ai_context/phase4_backtest_sweep.py`

**bc_master_sweep_v2 — 6 new BC L1 oscillator gates × 3 instruments × 12 combos = 216 runs:**
| Gate | GBPJPY avg | EURUSD avg | EURAUD avg |
|------|-----------|-----------|-----------|
| sto_tso | **32.82** | 9.60 | 7.10 |
| tso_pink | 30.16 | 10.57 | 6.24 |
| sto_reg | 28.52 | **11.57** | 7.32 |
| mcp_reg | 22.71 | 9.42 | 6.33 |
| ddl_pos | 19.84 | 10.04 | 6.89 |
| ddl_mcp | 18.93 | 8.70 | 5.99 |
*V1 benchmarks: GBPJPY macd_lc=28.76/fsb_strong=30.08 | EURUSD macd_lc=15.77 | EURAUD best=9.11*
- **GBPJPY**: `sto_tso` and `tso_pink` beat ALL v1 gates. Best single: sto_reg n=5,cd=30 OOS PF 48.96.
- **EURUSD**: `sto_reg` marginal improvement only (11.57 vs baseline 11.44). All below macd_lc.
- **EURAUD**: All gates below benchmark 10.62. BC/FS/oscillator gate family exhausted.
- Output: `ai_context/bc_sweep_v2_results.json`

**EA015 (STO Swing Reversal, EURAUD)**: 36 combos. Best OOS PF 6.83. 0/36 beat 10.62. **STOP.**
**EA016 (MCP+DDL Dual Momentum, EURAUD)**: 12 combos. Best OOS PF 5.46. 0/12 beat 10.62. **STOP.**
- Output: `ai_context/ea_phase4_creative_results.json`

## Phase 5 Sweep (2026-03-18) — script: `ai_context/phase5_backtest_sweep.py`

**EA017 (EURAUD baseline on 0.0007)**: OOS PF 3.93–4.65. Larger brick did NOT improve base edge.
**EA018 (VP+div+session on 0.0007)**: Best OOS PF 9.24. 0/96 beat benchmark 10.62.
**GJ011 (GJ008 + sto_tso + macd_lc)**: **27/48 beat benchmark 21.33.** Best OOS PF 48.75.
- Output: `ai_context/ea_phase5_results.json`

## Phase 6 Sweep (2026-03-18) — script: `renko/phase6_sweep.py`
Pure Renko, no candle data. 20 untapped indicator gates × 12 params × 3 instruments = 732 runs.
Enrichment: `renko/phase6_enrichment.py` (CCI, Ichimoku, Williams %R, Donchian, ESCGO, DDL, MOTN, MK).

**10 gates beat baseline on ALL 3 instruments.** Top per-instrument gates:
| Gate | EURUSD | GBPJPY | EURAUD |
|------|--------|--------|--------|
| ema_cross | **6.45** | 10.45 | 4.92 |
| escgo_cross | 5.59 | **12.22** | 5.51 |
| mk_regime | N/A | **12.53** | N/A |
| ichi_cloud | **6.38** | 9.20 | **5.55** |
- Output: `ai_context/phase6_results.json`

## Phase 6 Combo Sweep — script: `renko/phase6_combo_sweep.py`
**Key insight:** Low overlap = better combos. 24% overlap (mk+escgo) → +3.82; 82% overlap (mk+psar) → +0.83.
- Output: `ai_context/phase6_combo_results.json`

## Phase 7 Stacking Sweep — script: `renko/phase7_stacking_sweep.py`
**Key insight: Renko ADX is a viable replacement for candle ADX.** Stacking 3+ layers achieves competitive results.
- GBPJPY: svr_p6_sto avg **26.35** (beats GJ008 21.33)
- EURUSD: top 16.22 (beats R008 12.79)
- Output: `ai_context/phase7_results.json`
