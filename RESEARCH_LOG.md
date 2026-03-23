# Research Log

> **Renko sweep v3** — 2026-03-22. New entry strategies (R022-R024) + exit optimization (R025).
> Commission: OANDA spread-equivalent. Fixed qty: 1000 units (FX). $1,000 initial capital.

## v3 — New Entries + Exit Optimization

### Multi-Instrument Global Top 10 (R022-R024, IS: 2023-01-23 to 2025-09-30)

| Rank | Strategy | Inst | Brick | PF | Net | Trades | WR% | Max DD% | Params |
|------|----------|------|-------|----|-----|--------|-----|---------|--------|
| 1 | R024 Keltner | USDJPY | 0.05 | **30.33** | $153,030 | 452 | **73.9%** | -3.69% | mult=1.5, atr=14, cd=30, ADX>20 |
| 2 | R024 Keltner | USDJPY | 0.05 | 28.91 | $194,996 | 601 | 74.5% | -8.62% | mult=2.5, atr=14, cd=20 |
| 3 | R022 Ichimoku | GBPJPY | 0.05 | **27.44** | $112,355 | 334 | **74.9%** | -4.27% | tenkan=14, cd=30, TK=yes |
| 4 | R023 Williams | GBPJPY | 0.05 | **25.97** | $74,500 | 250 | **72.4%** | -1.97% | wpr=21, cd=30, ADX>20 |
| 5 | R024 Keltner | USDJPY | 0.05 | 24.67 | $141,184 | 437 | 68.9% | -3.98% | mult=2.5, atr=14, cd=30, ADX>20 |
| 6 | R023 Williams | USDJPY | 0.05 | **23.88** | $151,504 | 476 | **68.9%** | -9.42% | wpr=21, cd=10, ADX>20 |
| 7 | R022 Ichimoku | USDJPY | 0.05 | 23.71 | $125,601 | 399 | 72.2% | -12.53% | tenkan=14, cd=30, TK=yes |
| 8 | R024 Keltner | GBPJPY | 0.05 | **23.14** | $158,469 | 527 | **72.3%** | -1.12% | mult=1.5, atr=14, cd=20 |
| 9 | R024 Keltner | GBPJPY | 0.1 | 17.66 | $262,436 | 602 | 65.8% | -10.12% | mult=1.5, atr=10, cd=20, ADX>20 |
| 10 | R022 Ichimoku | GBPJPY | 0.1 | 15.96 | $225,619 | 567 | 64.4% | -2.65% | tenkan=9, cd=20, TK=yes |

### Best Per Instrument

| Inst | Best Strategy | Brick | PF | WR% | Max DD% |
|------|--------------|-------|----|-----|---------|
| **USDJPY** | R024 Keltner | 0.05 | **30.33** | 73.9% | -3.69% |
| **GBPJPY** | R022 Ichimoku | 0.05 | **27.44** | 74.9% | -4.27% |
| **EURAUD** | R022 Ichimoku | 0.0006 | **15.99** | 64.2% | -0.27% |
| **GBPUSD** | R024 Keltner | 0.0004 | **15.29** | 66.8% | -0.15% |
| **EURUSD** | R024 Keltner | 0.0004 | **14.72** | 66.9% | -0.15% |

### Key Findings (v3)

**Multi-instrument sweep (3 strategies × 5 instruments × 18 FX files = ~3,240 backtests):**
- **R024 Keltner on USDJPY 0.05 is the new project champion: PF=30.3, 73.9% WR.** This is 3.7× better than the best v2 result (R012 on USDJPY, PF=8.2).
- **JPY pairs dominate** — USDJPY and GBPJPY take all global top 8 slots. Larger pip movements suit Renko brick filtering.
- **All 3 strategies generalize across instruments** — every strategy × instrument combo is profitable on IS.
- **Small bricks (0.05 JPY, 0.0004 USD pairs) produce higher PF** — more trades, more signal.
- **R024 Keltner is the overall winner** — best on 3/5 instruments. ATR-envelope breakout is the ideal Renko entry.
- **R022 Ichimoku excels on GBPJPY** — 74.9% win rate, the highest in the project.
- **R023 Williams works best on JPY pairs** — PF 23-26 with ADX gate.

**Exit optimization (Option D, R025):**
- **First-opposing-brick exit is already optimal on Renko.** Confirmed across 144 combos.
- All alternatives (trailing, min-hold, Supertrend, KAMA) reduce PF from ~15 to 3-8.
- **Why:** Renko bricks already filter noise — the first opposing brick IS the signal.

### OOS Validation (2025-10-01 to 2026-03-05)

| Strategy | Inst | Brick | IS PF | OOS PF | OOS WR% | OOS DD% | OOS Trades | Verdict |
|----------|------|-------|-------|--------|---------|---------|------------|---------|
| R024 Keltner | USDJPY | 0.05 | 30.33 | **26.86** | 72.4% | -0.77% | 105 | **PASS** |
| R022 Ichimoku | USDJPY | 0.05 | 23.71 | **21.92** | 71.2% | -1.77% | 104 | **PASS** |
| R024 Keltner | GBPJPY | 0.05 | 23.14 | **15.48** | 64.6% | -4.79% | 223 | **PASS** |
| R022 Ichimoku | GBPJPY | 0.05 | 27.44 | **15.68** | 62.2% | -7.73% | 156 | **PASS** |
| R023 Williams | GBPJPY | 0.05 | 25.97 | **11.38** | 59.6% | -5.91% | 178 | **PASS** |
| R023 Williams | USDJPY | 0.05 | 23.88 | **10.74** | 58.7% | -5.21% | 104 | **PASS** |
| R024 Keltner | EURUSD | 0.0004 | 14.72 | **9.36** | 58.3% | -0.11% | 60 | **PASS** |
| R022 Ichimoku | EURUSD | 0.0004 | 14.37 | **9.26** | 58.2% | -0.15% | 67 | **PASS** |

**All 8 OOS tests pass.** Key observations:
- **OOS PF remains extremely high** — range 9.3–26.9, well above the PF>2 threshold for live trading.
- **Expected PF decay of 30-60%** from IS to OOS — normal and healthy (no overfitting).
- **JPY pairs hold up best** — R024 Keltner on USDJPY barely decays (30.3→26.9), suggesting genuine edge.
- **Win rates remain >58%** across all combos in OOS — consistent with IS.
- **R024 Keltner is the strongest OOS performer** — top OOS PF on both USDJPY and GBPJPY.

---

> **Renko sweep v2** — 2026-03-22. 12 strategies swept across 6 instruments × 20 Renko files (~13,340 total runs).
> Commission: OANDA spread-equivalent. Fixed qty: 1000 units (FX) / 0.01 BTC. $1,000 initial capital.

## Global Top 10 (All Strategies × All Instruments)

| Rank | Strategy | Inst | Brick | PF | Net | Trades | Win% | Max DD% | Params |
|------|----------|------|-------|----|-----|--------|------|---------|--------|
| 1 | R012 Regime Donchian | USDJPY | 0.05 | 8.212 | $182,913 | 618 | 60.5% | -5.29% | n=40, exit=5, chop<61.8, cd=12 |
| 2 | R001 Donchian | USDJPY | 0.05 | 8.145 | $182,750 | 619 | 60.4% | -5.17% | n_entry=40, n_exit=5, cd=12 |
| 3 | R011 Vol Donchian | USDJPY | 0.05 | 7.862 | $177,590 | 612 | 60.0% | -6.59% | n=40, exit=5, obv(20), cd=12 |
| 4 | R001 Donchian | GBPJPY | 0.05 | 7.364 | $219,472 | 828 | 60.5% | -7.82% | n_entry=20, n_exit=5, cd=12 |
| 5 | R012 Regime Donchian | GBPJPY | 0.05 | 7.357 | $219,239 | 827 | 60.5% | -8.12% | n=20, exit=5, squeeze, cd=12 |
| 6 | R011 Vol Donchian | GBPJPY | 0.05 | 7.254 | $156,743 | 575 | 60.7% | -10.43% | n=40, exit=5, mfi(20), cd=12 |
| 7 | R004 BB Squeeze | USDJPY | 0.05 | 6.983 | $295,567 | 1057 | 57.4% | -18.52% | bb=14, std=1.5, sq=0.003, cd=3 |
| 8 | R004 BB Squeeze | GBPJPY | 0.05 | 6.014 | $245,371 | 944 | 58.3% | -14.46% | bb=14, std=2.0, sq=0.003, cd=3 |
| 9 | R005 MACD | USDJPY | 0.05 | 5.974 | $89,922 | 387 | 53.7% | -9.65% | fast=8, slow=21, sig=7, cd=12 |
| 10 | R009 Fisher+AO | USDJPY | 0.05 | 5.570 | $135,022 | 843 | 55.4% | -18.39% | fisher=14, ao=8/21, cd=6 |

## Leaderboard by Instrument (Best Result Per Strategy, Sorted by PF)

### USDJPY

| Rank | Strategy | Brick | PF | Net | Trades | Win% | Max DD% | Params |
|------|----------|-------|----|-----|--------|------|---------|--------|
| 1 | R012 Regime Donchian | 0.05 | 8.212 | $182,913 | 618 | 60.5% | -5.29% | n=40, exit=5, chop<61.8, cd=12 |
| 2 | R001 Donchian | 0.05 | 8.145 | $182,750 | 619 | 60.4% | -5.17% | n_entry=40, n_exit=5, cd=12 |
| 3 | R011 Vol Donchian | 0.05 | 7.862 | $177,590 | 612 | 60.0% | -6.59% | n=40, exit=5, obv(20), cd=12 |
| 4 | R004 BB Squeeze | 0.05 | 6.983 | $295,567 | 1057 | 57.4% | -18.52% | bb=14, std=1.5, sq=0.003, cd=3 |
| 5 | R005 MACD | 0.05 | 5.974 | $89,922 | 387 | 53.7% | -9.65% | fast=8, slow=21, sig=7, cd=12 |
| 6 | R009 Fisher+AO | 0.05 | 5.570 | $135,022 | 843 | 55.4% | -18.39% | fisher=14, ao=8/21, cd=6 |
| 7 | R006 ST+ADX | 0.05 | 5.108 | $271,095 | 1280 | 53.2% | -10.69% | atr=7, mult=2.0, adx=25, cd=3 |
| 8 | R003 Supertrend | 0.05 | 4.809 | $188,553 | 939 | 52.4% | -26.49% | atr=7, mult=2.0, cd=12 |
| 9 | R002 EMA+ADX | 0.15 | 4.493 | $28,164 | 39 | 46.2% | -49.22% | fast=5, slow=34, adx=30, cd=6 |
| 10 | R007 RSI Reversion | 0.1 | 4.376 | $10,468 | 43 | 81.4% | -60.21% | rsi=14, os=25, ob=75, adx<35 |
| 11 | R010 PSAR+KAMA | 0.05 | 3.661 | $268,374 | 1254 | 51.4% | -19.37% | sar=0.02/0.2, kama=10/2/60 |
| 12 | R008 Stoch+CCI | — | — | — | — | — | — | Too few trades across all bricks |

### GBPJPY

| Rank | Strategy | Brick | PF | Net | Trades | Win% | Max DD% | Params |
|------|----------|-------|----|-----|--------|------|---------|--------|
| 1 | R001 Donchian | 0.05 | 7.364 | $219,472 | 828 | 60.5% | -7.82% | n_entry=20, n_exit=5, cd=12 |
| 2 | R012 Regime Donchian | 0.05 | 7.357 | $219,239 | 827 | 60.5% | -8.12% | n=20, exit=5, squeeze, cd=12 |
| 3 | R011 Vol Donchian | 0.05 | 7.254 | $156,743 | 575 | 60.7% | -10.43% | n=40, exit=5, mfi(20), cd=12 |
| 4 | R004 BB Squeeze | 0.05 | 6.014 | $245,371 | 944 | 58.3% | -14.46% | bb=14, std=2.0, sq=0.003, cd=3 |
| 5 | R006 ST+ADX | 0.05 | 4.621 | $180,257 | 968 | 53.5% | -39.19% | atr=14, mult=2.0, adx=25, cd=6 |
| 6 | R003 Supertrend | 0.05 | 4.513 | $175,360 | 943 | 53.7% | -39.19% | atr=14, mult=2.0, cd=12 |
| 7 | R009 Fisher+AO | 0.05 | 4.494 | $87,622 | 674 | 53.6% | -12.51% | fisher=14, ao=5/34, cd=6 |
| 8 | R005 MACD | 0.05 | 4.478 | $67,706 | 398 | 52.5% | -26.26% | fast=8, slow=21, sig=7, cd=12 |
| 9 | R002 EMA+ADX | 0.05 | 3.627 | $45,845 | 163 | 45.4% | -92.46% | fast=5, slow=34, adx=30, cd=6 |
| 10 | R010 PSAR+KAMA | 0.05 | 3.488 | $221,170 | 1135 | 52.5% | -27.52% | sar=0.02/0.3, kama=10/2/60 |
| 11 | R007 RSI Reversion | 0.1 | 1.823 | $7,540 | 53 | 81.1% | -186.14% | rsi=14, os=25, ob=75, adx<35 |

### EURUSD

| Rank | Strategy | Brick | PF | Net | Trades | Win% | Max DD% | Params |
|------|----------|-------|----|-----|--------|------|---------|--------|
| 1 | R001 Donchian | 0.0004 | 5.749 | $877 | 517 | 58.0% | -0.52% | n_entry=60, n_exit=5, cd=12 |
| 2 | R012 Regime Donchian | 0.0005 | 5.184 | $1,211 | 652 | 56.4% | -0.63% | n=40, exit=5, chop<55, cd=6 |
| 3 | R011 Vol Donchian | 0.0005 | 5.040 | $1,180 | 646 | 55.9% | -0.64% | n=40, exit=5, obv(14), cd=6 |
| 4 | R004 BB Squeeze | 0.0004 | 4.452 | $1,698 | 1169 | 52.2% | -0.70% | bb=14, std=1.5, sq=0.003, cd=3 |
| 5 | R009 Fisher+AO | 0.0004 | 3.543 | $663 | 904 | 50.9% | -0.82% | fisher=14, ao=5/34, cd=3 |
| 6 | R005 MACD | 0.0004 | 3.278 | $513 | 514 | 46.5% | -0.79% | fast=8, slow=26, sig=7, cd=3 |
| 7 | R003 Supertrend | 0.0004 | 3.186 | $1,022 | 1012 | 48.4% | -0.93% | atr=14, mult=2.0, cd=12 |
| 8 | R006 ST+ADX | 0.0004 | 3.161 | $1,787 | 1790 | 48.2% | -0.83% | atr=14, mult=2.0, adx=15, cd=3 |
| 9 | R002 EMA+ADX | 0.0008 | 2.623 | $232 | 100 | 37.0% | -2.43% | fast=5, slow=34, adx=25, cd=3 |
| 10 | R010 PSAR+KAMA | 0.0005 | 2.447 | $1,014 | 833 | 48.4% | -1.36% | sar=0.02/0.2, kama=20/2/60 |
| 11 | R007 RSI Reversion | 0.0004 | 2.023 | $39 | 61 | 77.0% | -1.38% | rsi=14, os=25, ob=75, adx<35 |

### EURAUD

| Rank | Strategy | Brick | PF | Net | Trades | Win% | Max DD% | Params |
|------|----------|-------|----|-----|--------|------|---------|--------|
| 1 | R001 Donchian | 0.0006 | 5.236 | $2,028 | 855 | 54.4% | -1.15% | n_entry=20, n_exit=5, cd=12 |
| 2 | R012 Regime Donchian | 0.0006 | 5.233 | $2,027 | 854 | 54.3% | -1.15% | n=20, exit=5, squeeze, cd=12 |
| 3 | R011 Vol Donchian | 0.0006 | 5.029 | $1,886 | 812 | 53.6% | -1.07% | n=20, exit=5, mfi(14), cd=12 |
| 4 | R004 BB Squeeze | 0.0006 | 4.334 | $2,505 | 1159 | 51.0% | -0.93% | bb=14, std=1.5, sq=0.005, cd=6 |
| 5 | R009 Fisher+AO | 0.0006 | 3.831 | $962 | 824 | 54.7% | -0.81% | fisher=14, ao=5/34, cd=6 |
| 6 | R002 EMA+ADX | 0.0006 | 3.579 | $310 | 90 | 44.4% | -1.49% | fast=5, slow=55, adx=30, cd=6 |
| 7 | R006 ST+ADX | 0.0006 | 3.391 | $1,673 | 1003 | 48.4% | -1.16% | atr=7, mult=2.0, adx=25, cd=6 |
| 8 | R005 MACD | 0.0006 | 3.299 | $665 | 442 | 45.2% | -1.13% | fast=8, slow=26, sig=7, cd=12 |
| 9 | R003 Supertrend | 0.0006 | 3.205 | $1,550 | 1014 | 48.0% | -1.66% | atr=7, mult=2.0, cd=12 |
| 10 | R007 RSI Reversion | 0.0012 | 2.788 | $80 | 35 | 77.1% | -3.02% | rsi=14, os=25, ob=75, adx<35 |
| 11 | R010 PSAR+KAMA | 0.0006 | 2.588 | $1,348 | 866 | 47.7% | -1.96% | sar=0.02/0.2, kama=20/2/30 |

### GBPUSD

| Rank | Strategy | Brick | PF | Net | Trades | Win% | Max DD% | Params |
|------|----------|-------|----|-----|--------|------|---------|--------|
| 1 | R004 BB Squeeze | 0.0004 | 5.134 | $64 | 33 | 48.5% | -0.30% | bb=14, std=2.5, sq=0.002, cd=3 |
| 2 | R011 Vol Donchian | 0.0004 | 4.880 | $904 | 600 | 56.3% | -0.64% | n=40, exit=5, cmf(20), cd=12 |
| 3 | R001 Donchian | 0.0004 | 4.788 | $990 | 670 | 54.2% | -0.73% | n_entry=40, n_exit=5, cd=3 |
| 4 | R012 Regime Donchian | 0.0004 | 4.786 | $989 | 670 | 54.2% | -0.73% | n=40, exit=5, chop<61.8, cd=6 |
| 5 | R006 ST+ADX | 0.0004 | 3.518 | $1,369 | 1245 | 52.2% | -0.64% | atr=7, mult=2.0, adx=20, cd=6 |
| 6 | R009 Fisher+AO | 0.0004 | 3.463 | $782 | 1053 | 51.1% | -0.92% | fisher=14, ao=8/34, cd=3 |
| 7 | R005 MACD | 0.0004 | 3.414 | $422 | 406 | 44.8% | -0.84% | fast=8, slow=26, sig=7, cd=12 |
| 8 | R003 Supertrend | 0.0004 | 3.397 | $1,075 | 1011 | 51.2% | -0.72% | atr=7, mult=2.0, cd=12 |
| 9 | R002 EMA+ADX | 0.0008 | 2.992 | $205 | 41 | 36.6% | -2.12% | fast=9, slow=55, adx=30, cd=6 |
| 10 | R010 PSAR+KAMA | 0.0004 | 2.652 | $1,341 | 1257 | 48.8% | -1.14% | sar=0.02/0.3, kama=10/2/30 |
| 11 | R007 RSI Reversion | 0.0008 | 2.110 | $57 | 46 | 71.7% | -3.46% | rsi=14, os=25, ob=75, adx<35 |

### BTCUSD

| Rank | Strategy | Brick | PF | Net | Trades | Win% | Max DD% | Params |
|------|----------|-------|----|-----|--------|------|---------|--------|
| 1 | R012 Regime Donchian | 150 | 3.845 | $3,469 | 648 | 54.5% | -1.85% | n=40, exit=5, chop<61.8, cd=12 |
| 2 | R001 Donchian | 150 | 3.823 | $3,463 | 649 | 54.4% | -1.86% | n_entry=40, n_exit=5, cd=12 |
| 3 | R011 Vol Donchian | 150 | 3.823 | $3,463 | 649 | 54.4% | -1.86% | n=40, exit=5, cmf(14), cd=12 |
| 4 | R005 MACD | 150 | 2.848 | $1,642 | 388 | 45.6% | -3.39% | fast=8, slow=21, sig=9, cd=12 |
| 5 | R002 EMA+ADX | 300 | 2.371 | $606 | 78 | 42.3% | -5.75% | fast=5, slow=21, adx=30, cd=6 |
| 6 | R006 ST+ADX | 150 | 2.242 | $3,169 | 1051 | 44.9% | -2.87% | atr=10, mult=2.0, adx=25, cd=6 |
| 7 | R003 Supertrend | 150 | 2.167 | $4,161 | 1412 | 44.1% | -3.62% | atr=7, mult=2.0, cd=6 |
| 8 | R009 Fisher+AO | 150 | 2.100 | $1,802 | 895 | 46.4% | -3.56% | fisher=10, ao=5/21, cd=6 |
| 9 | R010 PSAR+KAMA | 150 | 1.991 | $3,731 | 1208 | 45.6% | -3.70% | sar=0.02/0.3, kama=10/2/60 |

---

## Key Findings (v2 — 12 strategies)

### Strategy Tier List

**Tier 1 — PF 7+ on best instrument (Donchian family):**
1. **R012 Regime-Gated Donchian** — New #1. PF 8.212 (USDJPY). Chop Index < 61.8 gate barely filters but slightly improves PF over raw R001.
2. **R001 Raw Donchian** — PF 8.145 (USDJPY). The baseline is already near-optimal.
3. **R011 Volume-Confirmed Donchian** — PF 7.862 (USDJPY). OBV filter reduces trades slightly, marginally less PF but better DD.

**Tier 2 — PF 5-7 (strong signals):**
4. **R004 BB Squeeze** — PF 6.983. Highest absolute net ($295K on USDJPY). Best for volume + profit.
5. **R005 MACD** — PF 5.974. Fewer trades, high quality. Same-side-of-zero gate is key.
6. **R009 Fisher+AO** — PF 5.570. NEW entry type. Fisher Transform is excellent on Renko. AO confirmation helps.

**Tier 3 — PF 3-5 (viable):**
7. **R006 Supertrend+ADX** — PF 5.108. ADX gate works on every instrument.
8. **R003 Supertrend** — PF 4.809. Solid raw performance.
9. **R010 PSAR+KAMA** — PF 3.661. Good trade volume, lower PF. KAMA confirmation always helps.
10. **R002 EMA+ADX** — PF 4.493 but only 39 trades. Statistically suspect.

**Tier 4 — Mean-reversion (poor fit for Renko):**
11. **R007 RSI Reversion** — PF 4.376 on USDJPY but only 43 trades, 60% DD. High win rate (81%) but terrible risk profile.
12. **R008 Stoch+CCI** — Effectively dead. Too few qualifying trades. Dual oscillator gate is too restrictive on Renko.

### New Insights from v2

- **Mean-reversion doesn't work on Renko** — R007/R008 produce few trades with poor risk profiles. Renko bars inherently filter noise, leaving mainly trend — which mean-reversion fights against.
- **Fisher Transform is the best new entry** — R009 is the only new strategy to crack Tier 2. Period=14 universally best.
- **Regime gates barely help Donchian** — R012 (chop gate) beats R001 by only 0.8% PF. The Donchian signal is already so good on Renko that gating adds minimal value.
- **Volume filters don't consistently improve Donchian** — R011 matches or slightly trails R001. Volume data on Renko may not be meaningful (bars form on price, not time).
- **PSAR+KAMA is the best "new" trend signal** — PF 3.66, 1254 trades. Different signal logic to Donchian, could ensemble well.

### Instrument Rankings (unchanged)

1. **USDJPY 0.05** — Dominant. 11/12 strategies profitable here.
2. **GBPJPY 0.05** — Strong #2. JPY crosses have the strongest Renko trends.
3. **EURUSD 0.0004** — Steady. Ultra-low DD (<1%).
4. **EURAUD 0.0006** — Consistent middle performer.
5. **GBPUSD 0.0004** — Similar to EURUSD.
6. **BTCUSD 150** — Weakest FX-like. Higher DD.

---

## R001 — Donchian Trend (Renko)

**Status:** Complete — 720 runs

**Winner:** USDJPY brick=0.05, PF 8.145, 619 trades, 60.4% WR, -5.17% DD
- Params: n_entry=40, n_exit=5, cooldown=12
- Profitable on ALL 6 instruments (PF 3.8–8.1)

---

## R002 — EMA + ADX (Renko)

**Status:** Complete — 1,440 runs

**Winner:** USDJPY brick=0.15, PF 4.493, 39 trades, 46.2% WR
- Low trade counts — statistically suspect on Renko

---

## R003 — Supertrend (Renko)

**Status:** Complete — 540 runs

**Winner:** USDJPY brick=0.05, PF 4.809, 939 trades, 52.4% WR
- Params: atr=7, mult=2.0, cooldown=12

---

## R004 — BB Squeeze (Renko)

**Status:** Complete — 720 runs

**Winner:** USDJPY brick=0.05, PF 6.983, 1057 trades, 57.4% WR
- Params: bb=14, std=1.5, squeeze_pct=0.003, cooldown=3
- Highest absolute net profit of any strategy

---

## R005 — MACD (Renko)

**Status:** Complete — 480 runs

**Winner:** USDJPY brick=0.05, PF 5.974, 387 trades, 53.7% WR
- Params: fast=8, slow=21, signal=7, cooldown=12

---

## R006 — Supertrend + ADX (Renko)

**Status:** Complete — 4,320 runs

**Winner:** USDJPY brick=0.05, PF 5.108, 1280 trades, 53.2% WR
- ADX gate confirms as universal quality improver

---

## R007 — RSI Mean-Reversion (Renko)

**Status:** Complete — 960 runs

**Winner:** USDJPY brick=0.1, PF 4.376, 43 trades, 81.4% WR, -60.21% DD
- High win rate but catastrophic drawdown. Mean-reversion is a poor fit for Renko.
- ADX < 35 (ranging gate) is essential — without it, results are worse

---

## R008 — Stochastic + CCI (Renko)

**Status:** Complete — 640 runs

**Verdict:** DEAD. Only 31 qualifying trades across the best combo. Dual oscillator
requirement is too restrictive on Renko bars.

---

## R009 — Fisher Transform + AO (Renko)

**Status:** Complete — 960 runs

**Winner:** USDJPY brick=0.05, PF 5.570, 843 trades, 55.4% WR, -18.39% DD
- Params: fisher_period=14, ao_fast=8, ao_slow=21, require_ao=True, cooldown=6
- Best new (non-Donchian) entry signal discovered. Fisher Transform is excellent on Renko.
- AO confirmation consistently helps (require_ao=True always beats False)

---

## R010 — PSAR + KAMA (Renko)

**Status:** Complete — 640 runs

**Winner:** USDJPY brick=0.05, PF 3.661, 1254 trades, 51.4% WR, -19.37% DD
- Params: sar=0.02/0.02/0.2, kama=10/2/60, require_kama=True
- Different signal logic to Donchian — potential ensemble candidate
- KAMA confirmation always improves results (require_kama=True always wins)

---

## R011 — Volume-Confirmed Donchian (Renko)

**Status:** Complete — 960 runs

**Winner:** USDJPY brick=0.05, PF 7.862, 612 trades, 60.0% WR, -6.59% DD
- Params: n_entry=40, n_exit=5, vol_indicator=obv, vol_period=20, cooldown=12
- Volume filter slightly reduces trades but doesn't meaningfully improve PF
- On BTCUSD, volume filter has zero effect (same PF as R001)

---

## R012 — Regime-Gated Donchian (Renko)

**Status:** Complete — 960 runs

**Winner:** USDJPY brick=0.05, PF 8.212, 618 trades, 60.5% WR, -5.29% DD
- Params: n_entry=40, n_exit=5, regime=chop, chop_threshold=61.8, cooldown=12
- Marginally beats raw R001 (+0.8% PF). Chop gate filters only 1 trade.
- Squeeze regime mode performs identically to raw Donchian (no effective filtering)

---

## R016 — DI Crossover (Renko)

**Status:** Complete — 32 runs (EURUSD 0.0004)

**Winner:** PF 13.994, 846 trades, 63.5% WR, -0.15% DD
- Params: adx_threshold=0, req_brick=True, cooldown=3
- Uses +DI/-DI crossover as the **primary entry signal** (not just an ADX level gate)
- Brick confirmation (req_brick=True) is essential — without it, PF drops to 9.3
- No ADX gate needed (adx=0 beats adx=15/20/25) — DI cross is already directional
- High trade count (846) with 63.5% WR — statistically robust
- **New finding:** DI crossover is a Tier 1 entry signal on EURUSD Renko, competitive with Donchian family

---

## R017 — EMA Stack Momentum (Renko)

**Status:** Complete — 24 runs (EURUSD 0.0004)

**Winner:** PF 16.773, 348 trades, 67.0% WR, -0.14% DD
- Params: require_ema200=True, adx_threshold=20, cooldown=30
- Entry: EMA9 > EMA21 > EMA50 > EMA200 (longs) with brick confirmation
- **Highest PF of any new strategy** — EMA200 requirement consistently improves PF (+2-3 PF over no-EMA200)
- ADX gate helps: PF 16.8 (adx=20) vs PF 15.7 (adx=0 with ema200)
- Fewer trades (348) but highest win rate (67%) and lowest DD (-0.14%)
- Without EMA200: PF 15.3, 465 trades, 67.3% WR — still excellent

---

## R018 — Stochastic Trend Entry (Renko)

**Status:** Complete — 72 runs (EURUSD 0.0004)

**Winner:** PF 9.015, 107 trades, 52.3% WR, -0.31% DD
- Params: os_level=30, ob_level=80, trend_filter=ema, adx_threshold=20, cooldown=20
- Entry: Stoch %K crosses %D from oversold/overbought in trend direction
- **EMA trend filter is essential** — Supertrend filter produces too few trades (<30)
- "Both" filter (ST+EMA) restricts too heavily
- Trade count is marginal (107) — on the edge of statistical significance
- Wider oversold zone (30) works better than narrow (20) — more entry opportunities
- **Verdict:** Viable but marginal. Much better than R007 (RSI mean-reversion, failed) because it trades WITH the trend, but trade count limits confidence.

---

## R019 — KAMA Slope + BB %B Confluence (Renko)

**Status:** Complete — 54 runs (EURUSD 0.0004)

**Winner:** PF 12.794, 414 trades, 60.9% WR, -0.16% DD
- Params: bb_pctb_level=0.5, req_brick=True, adx_threshold=0, cooldown=20
- Entry: KAMA slope turns positive + BB %B > 0.5 (longs), reverse for shorts
- BB %B level 0.5 is optimal (0.4 and 0.6 are slightly worse)
- Brick confirmation essential (req_brick=True always wins)
- ADX gate doesn't help — KAMA slope + BB %B is already a strong dual filter
- Good trade count (414) with solid WR (60.9%)

---

## R020 — Multi-Signal Voting Confluence (Renko)

**Status:** Complete — 27 runs (EURUSD 0.0004)

**Winner:** PF 13.289, 719 trades, 61.1% WR, -0.20% DD
- Params: min_votes=5, adx_threshold=0, cooldown=20
- Votes: brick direction + MACD hist + RSI>50 + Supertrend + KAMA slope
- **5-of-5 unanimous agreement is ALWAYS best** — every weaker threshold (3, 4) is worse
- ADX gate is unnecessary when all 5 signals agree
- High trade count even at unanimous (719-1262 depending on cooldown)
- **Key finding:** unanimous agreement across independent signals produces the cleanest entries. This is an ensemble approach that naturally filters noise.
- Lower cooldown (cd=10) gives more trades (1099) with only slight PF reduction (13.25)

---

## R021 — Squeeze + Brick Count Hybrid (Renko)

**Status:** DEAD — 108 runs, zero qualifying trades

**Verdict:** The dual requirement (squeeze release + N consecutive bricks) is too restrictive on EURUSD 0.0004. Squeeze releases are rare events on Renko and requiring N directional bricks simultaneously produces zero trades across all 108 parameter combinations. Not viable.

---

## v3 Research Summary (R016-R021)

### New Strategy Tier List (EURUSD 0.0004)

**Tier 1 — PF 13+ (strong):**
1. **R017 EMA Stack** — PF 16.77, 348 trades, 67.0% WR. New #1 on EURUSD. EMA alignment with EMA200 gate.
2. **R016 DI Crossover** — PF 13.99, 846 trades, 63.5% WR. Highest trade count among Tier 1. DI cross entry.
3. **R020 Vote Confluence** — PF 13.29, 719 trades, 61.1% WR. 5-of-5 unanimous voting. Robust ensemble.

**Tier 2 — PF 9-13 (solid):**
4. **R019 KAMA+BB%B** — PF 12.79, 414 trades, 60.9% WR. KAMA slope change + band position.
5. **R018 Stoch Trend** — PF 9.02, 107 trades, 52.3% WR. Low trade count limits confidence.

**Dead:**
6. **R021 Squeeze+Bricks** — Zero trades. Dual requirement too restrictive.

### Key Insights from v3

- **EMA alignment is the best new entry signal** — R017 beats all previous EURUSD-specific strategies (R001-R015)
- **DI crossover works as a primary signal** — not just a level gate. req_brick=True is essential.
- **Unanimous voting (5/5) always beats partial** — confirms that signal quality > quantity
- **Dual rare-event strategies don't work on Renko** — R021 (squeeze + bricks) joins R008 (stoch+CCI) as dead
- **ADX gate is unnecessary** when entry signal is already directionally filtered (R016, R019, R020)
- **Brick confirmation is universally helpful** across all new strategies
