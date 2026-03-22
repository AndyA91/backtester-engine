# Research Log

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
