# Strategy Details Archive
Detailed findings moved from MEMORY.md to keep that file under 200 lines.

## EURUSD Renko Research (IS: 2023-01-23→2025-09-30 | OOS: 2025-10-01→2026-03-05)

**R001 — N Consecutive Bricks (TV validated ✓)**
- All 16 combos PF > 12 IS, PF 4.7–7.3 OOS. MaxDD < 0.5%.
- Best IS: n=2, cd=30 → PF 15.70. Best OOS: n=3, cd=30 → PF 7.33

**R002 — Brick Count Reversal (TV validated ✓)**
- IS: All 16 combos PF 11–14, WR 60–64%. NOT counter-trend — run-initiation detection.
- OOS: PF 4.28–7.12. Pine key: loop `for j = 1 to n_bricks` (vs R001's `0 to n_bricks-1`)
- **Parameter flip**: R001 OOS best at n=3, R002 OOS best at n=5 — complementary timing

**R005 — Hybrid Master (NEGATIVE)**
- Reversal mode dead: all 36 reversal combos PF < 1.
- trail_bricks=2 uniformly worse; first-opposing-brick exit is optimal.

**R006 — Brick Alternation Filter (NEGATIVE)**
- 54 combos IS sweep. Best PF 15.63 vs R001 15.70 — essentially unchanged.

**R007 — R001+R002 Combined System (OOS CONFIRMED ✓)**
- IS: PF 11.96–13.00 | OOS: PF 5.32–6.12, Net $253–$402
- TV validated: n=3, cd=10 → TV 2324t/$3492.93 vs Python 2314t/$3483.98 ✓
- Design: R002 priority (no cooldown); R001 fallback (flat gate + cooldown)

**R008 — R007 + Candle ADX(25) + Vol(1.5) + Session(13) (TV VERIFIED ✓)**
- IS: 2024-01-01→2025-09-30 | OOS: 2025-10-01→2026-02-28 (constrained by HISTDATA)
- Gate stacking (n=5, cd=30): Baseline PF 6.37 → +ADX 8.31 → +Vol 9.66 → +Sess 12.79 OOS
- Tag analysis: Vol<0.5 PF 25.4, Vol>2.5 PF 6.8 → vol_max=1.5 confirmed
- Session: Asian 10.3 | London 14.8 | Lon+NY 18.7
- TV validation: IS PF 21.87/371t | OOS PF 8.75/56t (TV IS→OOS decay 60% vs Python 15%)

**R010 — PSAR Opposing Gate (NEGATIVE)**
- Tag analysis showed opposing-PSAR PF 21.51 vs aligned 15.99 — but IS correlation doesn't hold OOS
- OOS top 5 entirely R008 baseline (psar_gate=False). REJECTED.

**Exit modification research — first-brick exit is optimal:**
- exit_vol_min [0.3-0.7]: uniformly negative
- exit_confirm_bricks [1-3]: catastrophically negative (2→4.64, 3→3.92 vs baseline 12.79)
- Winners hold 13.8h vs losers 6.5h — edge lives in duration asymmetry

**Gate details (Pine):**
- ADX: `adx_ok = adx_threshold<=0 or (not na(adx_5m) and adx_5m>=adx_threshold)`
- Vol: `vol_ok = vol_max<=0 or vol_ratio<=vol_max` (vol_ratio = volume/EMA20vol)
- Session: `bar_hour = hour(time, "UTC")` / `session_ok = session_start<=0 or bar_hour>=session_start`

## EURAUD Renko Research (IS: 2023-07-20→2025-09-30 | OOS: 2025-10-01→2026-03-17)

**EA001 — R007 baseline**: All 12 combos PF 8.28–9.21 IS (~50% OOS decay).

**EA002 — dgtrd Gate Sweep (576 combos):**
- div_gate ✓ | vp_gate ✓ | session_start=13 ✓ | raff_gate ✗ | do_gate ✗
- Critical: div-alone inflates IS but collapses OOS. Adding vp_gate LOWERS IS but RAISES OOS.

**EA008 — R007 + VP POC + MACD Div + Session=13 (TV VERIFIED ✓)**
- IS PF 10.20/270t → TV 10.27/276t ✓ | OOS PF 10.62/72t → TV 8.68/73t (AUD conversion delta)
- Gate stacking (n=5,cd=30): Baseline 4.67 → +div+ss13: 7.27 → +vp+div+ss13: **10.62**
- VP implementation: `ta.pivothigh/low(high/low, 20, 20)` triggers recompute; NaN-pass

## GBPJPY Renko Research (IS: 2024-11-21→2025-09-30 | OOS: 2025-10-01→2026-02-28)

**GJ007 — R001+R002 Combined baseline:**
- IS: PF 17.02–17.84, WR 67–69% | OOS: PF 8.69–10.84

**GJ008 — GJ007 + ADX(25) + Vol(1.5) + Session(13) (TV VERIFIED ✓)**
- n=5,cd=20 reference: sess=0 OOS 13.34 → sess=7 OOS 13.26 → sess=13 OOS **21.33** (-19% decay)
- TV validation: IS PF 25.16/359t | OOS PF 16.91/152t | TV IS→OOS decay 33%
- Trade count ~1.65x Python — consistent OANDA vs HISTDATA ADX source divergence

## BC/FS Gate Sweep v1 (bc_master_sweep.py)
Results in `ai_context/bc_sweep_results.json` and `ai_context/bc_sweep_findings.md`.
- **Best 2-instrument gates**: macd_lc (EURUSD 15.77, GBPJPY 28.76), fsb_strong (EURUSD 14.12, GBPJPY 30.08)
- EURUSD best: macd_lc n=3,cd=20 OOS PF 20.48 | GBPJPY best: st n=5,cd=30 OOS PF 48.75
- EURAUD: best gate 9.11 (st_fsb_macd) — no gate beat EA008 benchmark 10.62
