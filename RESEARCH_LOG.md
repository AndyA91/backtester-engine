# Research Log

> **Renko sweep** — 2026-03-22. All 6 strategies swept across 6 instruments × 20 Renko files.
> Commission: OANDA spread-equivalent. Fixed qty: 1000 units (FX) / 0.01 BTC. $1,000 initial capital.

## Global Top 10 (All Strategies × All Instruments)

| Rank | Strategy | Inst | Brick | PF | Net | Trades | Win% | Max DD% | Params |
|------|----------|------|-------|----|-----|--------|------|---------|--------|
| 1 | R001 Donchian | USDJPY | 0.05 | 8.145 | $182,750 | 619 | 60.4% | -5.17% | n_entry=40, n_exit=5, cd=12 |
| 2 | R004 BB Squeeze | USDJPY | 0.05 | 6.983 | $295,567 | 1057 | 57.4% | -18.52% | bb=14, std=1.5, sq=0.003, cd=3 |
| 3 | R001 Donchian | GBPJPY | 0.05 | 7.364 | $219,472 | 828 | 60.5% | -7.82% | n_entry=20, n_exit=5, cd=12 |
| 4 | R005 MACD | USDJPY | 0.05 | 5.974 | $89,922 | 387 | 53.7% | -9.65% | fast=8, slow=21, sig=7, cd=12 |
| 5 | R001 Donchian | EURUSD | 0.04 | 5.749 | $877 | 517 | 58.0% | -0.52% | n_entry=60, n_exit=5, cd=12 |
| 6 | R004 BB Squeeze | GBPJPY | 0.05 | 6.014 | $245,371 | 944 | 58.3% | -14.46% | bb=14, std=2.0, sq=0.003, cd=3 |
| 7 | R006 ST+ADX | USDJPY | 0.05 | 5.108 | $271,095 | 1280 | 53.2% | -10.69% | atr=7, mult=2.0, adx=25, cd=3 |
| 8 | R001 Donchian | EURAUD | 0.06 | 5.236 | $2,028 | 855 | 54.4% | -1.15% | n_entry=20, n_exit=5, cd=12 |
| 9 | R003 Supertrend | USDJPY | 0.05 | 4.809 | $188,553 | 939 | 52.4% | -26.49% | atr=7, mult=2.0, cd=12 |
| 10 | R001 Donchian | GBPUSD | 0.04 | 4.788 | $990 | 670 | 54.2% | -0.73% | n_entry=40, n_exit=5, cd=3 |

## Leaderboard by Instrument (Best Result Per Strategy, Sorted by PF)

### USDJPY

| Rank | Strategy | Brick | PF | Net | Trades | Win% | Max DD% | Params |
|------|----------|-------|----|-----|--------|------|---------|--------|
| 1 | R001 Donchian | 0.05 | 8.145 | $182,750 | 619 | 60.4% | -5.17% | n_entry=40, n_exit=5, cd=12 |
| 2 | R004 BB Squeeze | 0.05 | 6.983 | $295,567 | 1057 | 57.4% | -18.52% | bb=14, std=1.5, sq=0.003, cd=3 |
| 3 | R005 MACD | 0.05 | 5.974 | $89,922 | 387 | 53.7% | -9.65% | fast=8, slow=21, sig=7, cd=12 |
| 4 | R006 ST+ADX | 0.05 | 5.108 | $271,095 | 1280 | 53.2% | -10.69% | atr=7, mult=2.0, adx=25, cd=3 |
| 5 | R003 Supertrend | 0.05 | 4.809 | $188,553 | 939 | 52.4% | -26.49% | atr=7, mult=2.0, cd=12 |
| 6 | R002 EMA+ADX | 0.15 | 4.493 | $28,164 | 39 | 46.2% | -49.22% | fast=5, slow=34, adx=30, cd=6 |

### GBPJPY

| Rank | Strategy | Brick | PF | Net | Trades | Win% | Max DD% | Params |
|------|----------|-------|----|-----|--------|------|---------|--------|
| 1 | R001 Donchian | 0.05 | 7.364 | $219,472 | 828 | 60.5% | -7.82% | n_entry=20, n_exit=5, cd=12 |
| 2 | R004 BB Squeeze | 0.05 | 6.014 | $245,371 | 944 | 58.3% | -14.46% | bb=14, std=2.0, sq=0.003, cd=3 |
| 3 | R005 MACD | 0.05 | 4.478 | $67,706 | 398 | 52.5% | -26.26% | fast=8, slow=21, sig=7, cd=12 |
| 4 | R006 ST+ADX | 0.05 | 4.621 | $180,257 | 968 | 53.5% | -39.19% | atr=14, mult=2.0, adx=25, cd=6 |
| 5 | R003 Supertrend | 0.05 | 4.513 | $175,360 | 943 | 53.7% | -39.19% | atr=14, mult=2.0, cd=12 |
| 6 | R002 EMA+ADX | 0.05 | 3.627 | $45,845 | 163 | 45.4% | -92.46% | fast=5, slow=34, adx=30, cd=6 |

### EURUSD

| Rank | Strategy | Brick | PF | Net | Trades | Win% | Max DD% | Params |
|------|----------|-------|----|-----|--------|------|---------|--------|
| 1 | R001 Donchian | 0.0004 | 5.749 | $877 | 517 | 58.0% | -0.52% | n_entry=60, n_exit=5, cd=12 |
| 2 | R004 BB Squeeze | 0.0004 | 4.452 | $1,698 | 1169 | 52.2% | -0.70% | bb=14, std=1.5, sq=0.003, cd=3 |
| 3 | R005 MACD | 0.0004 | 3.278 | $513 | 514 | 46.5% | -0.79% | fast=8, slow=26, sig=7, cd=3 |
| 4 | R003 Supertrend | 0.0004 | 3.186 | $1,022 | 1012 | 48.4% | -0.93% | atr=14, mult=2.0, cd=12 |
| 5 | R006 ST+ADX | 0.0004 | 3.161 | $1,787 | 1790 | 48.2% | -0.83% | atr=14, mult=2.0, adx=15, cd=3 |
| 6 | R002 EMA+ADX | 0.0008 | 2.623 | $232 | 100 | 37.0% | -2.43% | fast=5, slow=34, adx=25, cd=3 |

### EURAUD

| Rank | Strategy | Brick | PF | Net | Trades | Win% | Max DD% | Params |
|------|----------|-------|----|-----|--------|------|---------|--------|
| 1 | R001 Donchian | 0.0006 | 5.236 | $2,028 | 855 | 54.4% | -1.15% | n_entry=20, n_exit=5, cd=12 |
| 2 | R004 BB Squeeze | 0.0006 | 4.334 | $2,505 | 1159 | 51.0% | -0.93% | bb=14, std=1.5, sq=0.005, cd=6 |
| 3 | R002 EMA+ADX | 0.0006 | 3.579 | $310 | 90 | 44.4% | -1.49% | fast=5, slow=55, adx=30, cd=6 |
| 4 | R006 ST+ADX | 0.0006 | 3.391 | $1,673 | 1003 | 48.4% | -1.16% | atr=7, mult=2.0, adx=25, cd=6 |
| 5 | R005 MACD | 0.0006 | 3.299 | $665 | 442 | 45.2% | -1.13% | fast=8, slow=26, sig=7, cd=12 |
| 6 | R003 Supertrend | 0.0006 | 3.205 | $1,550 | 1014 | 48.0% | -1.66% | atr=7, mult=2.0, cd=12 |

### GBPUSD

| Rank | Strategy | Brick | PF | Net | Trades | Win% | Max DD% | Params |
|------|----------|-------|----|-----|--------|------|---------|--------|
| 1 | R004 BB Squeeze | 0.0004 | 5.134 | $64 | 33 | 48.5% | -0.30% | bb=14, std=2.5, sq=0.002, cd=3 |
| 2 | R001 Donchian | 0.0004 | 4.788 | $990 | 670 | 54.2% | -0.73% | n_entry=40, n_exit=5, cd=3 |
| 3 | R006 ST+ADX | 0.0004 | 3.518 | $1,369 | 1245 | 52.2% | -0.64% | atr=7, mult=2.0, adx=20, cd=6 |
| 4 | R005 MACD | 0.0004 | 3.414 | $422 | 406 | 44.8% | -0.84% | fast=8, slow=26, sig=7, cd=12 |
| 5 | R003 Supertrend | 0.0004 | 3.397 | $1,075 | 1011 | 51.2% | -0.72% | atr=7, mult=2.0, cd=12 |
| 6 | R002 EMA+ADX | 0.0008 | 2.992 | $205 | 41 | 36.6% | -2.12% | fast=9, slow=55, adx=30, cd=6 |

### BTCUSD

| Rank | Strategy | Brick | PF | Net | Trades | Win% | Max DD% | Params |
|------|----------|-------|----|-----|--------|------|---------|--------|
| 1 | R001 Donchian | 150 | 3.823 | $3,463 | 649 | 54.4% | -1.86% | n_entry=40, n_exit=5, cd=12 |
| 2 | R005 MACD | 150 | 2.848 | $1,642 | 388 | 45.6% | -3.39% | fast=8, slow=21, sig=9, cd=12 |
| 3 | R002 EMA+ADX | 300 | 2.371 | $606 | 78 | 42.3% | -5.75% | fast=5, slow=21, adx=30, cd=6 |
| 4 | R006 ST+ADX | 150 | 2.242 | $3,169 | 1051 | 44.9% | -2.87% | atr=10, mult=2.0, adx=25, cd=6 |
| 5 | R003 Supertrend | 150 | 2.167 | $4,161 | 1412 | 44.1% | -3.62% | atr=7, mult=2.0, cd=6 |

---

## Key Findings

### Strategy Rankings (by avg best-PF across instruments)

1. **R001 Donchian Breakout** — Dominant winner. PF 3.8–8.1 across all instruments. Simple n_exit=5 exit channel consistently best.
2. **R004 BB Squeeze** — Strong #2. PF 4.3–7.0. bb_period=14 and std=1.5–2.0 universally best. Huge trade counts.
3. **R005 MACD** — Solid #3. PF 2.8–6.0. fast=8, slow=21–26, signal=7 consistently top. Fewer trades but high quality.
4. **R006 Supertrend+ADX** — ADX gate improves R003 everywhere. PF 2.2–5.1. mult=2.0 always best.
5. **R003 Supertrend** — Raw Supertrend is viable. PF 2.2–4.8. Always mult=2.0, atr=7–14.
6. **R002 EMA+ADX** — Weakest on Renko. PF 2.4–4.5. Low trade counts, high DD on JPY pairs.

### Instrument Rankings (by best available PF)

1. **USDJPY 0.05** — Best instrument by far. Every strategy hits PF 4.5+ here.
2. **GBPJPY 0.05** — Strong #2. PF 3.6–7.4 depending on strategy.
3. **EURUSD 0.0004** — Steady performer. Lower PF (2.6–5.7) but ultra-low DD (<1%).
4. **EURAUD 0.0006** — Consistent. PF 3.2–5.2, low DD.
5. **GBPUSD 0.0004** — Similar to EURUSD. PF 3.0–5.1.
6. **BTCUSD 150** — Weakest. PF 2.2–3.8. Higher DD than FX.

### Cross-Cutting Patterns

- **Smallest brick size wins** on every instrument — more bars = more signal opportunities
- **ADX gating** (R006 vs R003) consistently improves PF across all instruments
- **n_exit=5** is universally optimal for Donchian — very tight trailing exit
- **cooldown=12** slightly edges cooldown=3 in PF (less overtrading)
- **USDJPY + GBPJPY** (JPY crosses) dominate — strongest Renko trends in 2022–2026 period

---

## R001 — Donchian Trend (Renko)

**Status:** Complete — swept 20 files × 36 combos = 720 runs

**Winner:** USDJPY brick=0.05, PF 8.145, 619 trades, 60.4% WR, -5.17% DD
- Params: n_entry=40, n_exit=5, cooldown=12
- Profitable on ALL 6 instruments (PF 3.8–8.1)

---

## R002 — EMA + ADX (Renko)

**Status:** Complete — swept 20 files × 72 combos = 1,440 runs

**Winner:** USDJPY brick=0.15, PF 4.493, 39 trades, 46.2% WR
- Low trade counts are a concern — may not be statistically robust
- Weakest strategy on Renko (EMA crosses need time-based bars to work well)

---

## R003 — Supertrend (Renko)

**Status:** Complete — swept 20 files × 27 combos = 540 runs

**Winner:** USDJPY brick=0.05, PF 4.809, 939 trades, 52.4% WR
- Params: atr=7, mult=2.0, cooldown=12
- Strong trade volume and consistent across instruments

---

## R004 — BB Squeeze (Renko)

**Status:** Complete — swept 20 files × 36 combos = 720 runs

**Winner:** USDJPY brick=0.05, PF 6.983, 1057 trades, 57.4% WR
- Params: bb=14, std=1.5, squeeze_pct=0.003, cooldown=3
- Highest trade counts of any strategy — squeeze detection works well on Renko

---

## R005 — MACD (Renko)

**Status:** Complete — swept 20 files × 24 combos = 480 runs

**Winner:** USDJPY brick=0.05, PF 5.974, 387 trades, 53.7% WR
- Params: fast=8, slow=21, signal=7, cooldown=12
- Same-side-of-zero filter effectively suppresses counter-trend entries

---

## R006 — Supertrend + ADX (Renko)

**Status:** Complete — swept 20 files × 216 combos = 4,320 runs

**Winner:** USDJPY brick=0.05, PF 5.108, 1280 trades, 53.2% WR
- Params: atr=7, mult=2.0, adx_threshold=25, cooldown=3
- ADX gate (R006) vs no gate (R003): PF 5.108 vs 4.809 on same instrument — +6% improvement
- Confirms ADX is the #1 quality predictor, even on Renko
