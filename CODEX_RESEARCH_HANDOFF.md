# Strategy Research Handoff - Autonomous Iteration Loop

## Your Role
You are a quantitative strategy researcher. Your job is to continuously design, test, and iterate
trading strategies on EURUSD 5m data, logging every result. The human will check in periodically
to review progress.

**IS = 2024-01-01 to 2025-09-30** (21 months — covers USD-strength 2024 and EUR-rally 2025,
two distinct regimes).
**OOS = 2025-10-01 to 2026-02-28** — sealed until a strong winner emerges. Do not touch.

---

## Immediate Task (pick up here)

R001–R005 are complete. R006 is written and ready to run:

```bash
cd strategies/research
python runner.py r006_supertrend_adx
```

After the run: update `RESEARCH_LOG.md` (leaderboard + R006 results block + analysis).
Then continue to R007 per the plan below.

---

## Results Summary (R001–R005)

| ID | Strategy | Best PF | Trades | Net $ | Key finding |
|----|----------|---------|--------|-------|-------------|
| R001 | Donchian + 1H KAMA slope | 0.8719 | 1136 | -$113 | No edge. All 36 combos PF < 1. |
| R002 | EMA cross + ADX + 1H slope | 1.0096 | 84 | +$0.60 | Only strategy above PF 1. Barely. |
| R003 | Supertrend flip + 1H agree | 0.9171 | 614 | -$41 | Best trade count among losers. |
| R004 | BB squeeze breakout | 0.7780 | 1216 | -$133 | Worst performer. Squeeze fires too often. |
| R005 | MACD cross + 1H KAMA | 0.7370 | 1071 | -$99 | Both-sides-of-zero filter doesn't help. |

**The single consistent pattern across all 5 strategies**: every strategy that lacked ADX
produced 600–2000+ trades at PF < 1. R002, the only strategy with ADX, was the only one
above PF 1. ADX(14) is definitively the edge-gating mechanism on this dataset.

---

## R006 — Supertrend + ADX (file written, run this now)

**File**: `r006_supertrend_adx.py`
**Logic**: R003's best signal (Supertrend flip, 1H agree=True) + ADX(14) regime gate.
`adx_threshold=0` reproduces the unfiltered R003 baseline for direct comparison.

**Param grid** (48 combos):
- `atr_period`: [7, 10, 14]
- `multiplier`: [3.0, 4.0]
- `adx_threshold`: [0, 20, 25, 30]  ← 0 = baseline
- `cooldown`: [12, 24]

**What to look for in results:**
- Compare `adx=0` rows (≈ R003) against `adx=20/25/30` rows for the same atr/mult combo.
  This isolates ADX's exact contribution.
- If ADX lifts PF above 1.5 with ≥ 60 trades → strong candidate for R007 refinement.
- If best PF is still < 1.2 across all ADX levels → ADX doesn't help Supertrend;
  move to R007 with a fundamentally different signal type (see below).

---

## R007 Plan — Branch on R006 Outcome

### If R006 finds PF ≥ 1.5 with ≥ 60 trades:
Build R007 as a **targeted refinement** of the best R006 combo:
- Widen the ADX sweep around the winning threshold (e.g. if adx=25 wins, test [22,25,28,32])
- Test `di_spread` filter: require `plus_di - minus_di > threshold` (adds directionality on top of ADX strength)
- Test tighter multiplier steps around the winning value
- Goal: push PF above 2.0 with ≥ 80 trades

### If R006 best PF < 1.2 (ADX does not help Supertrend):
ADX only helped EMA-cross in R002 (barely). Trend-following as a class may lack edge on
this dataset. Pivot to **mean reversion** for R007:

**R007 idea — RSI Fade + BB Reversion**:
- Entry: RSI(14) < 30 → long; RSI(14) > 70 → short (oversold/overbought fade)
- Filter: price outside Bollinger Band (2σ) confirms the extreme
- Exit: RSI crosses back through 50, or price returns to BB mid
- Session: 07:00–22:00 UTC
- Sweep: rsi_period [10,14], rsi_ob [65,70], rsi_os [30,35], bb_period [14,20], cooldown [6,12,24]
- Hypothesis: EURUSD 5m is mean-reverting in ranging conditions, which dominates the IS period

**R007 idea — London Open Breakout**:
- Entry: first 5m bar that breaks the previous session's high/low (07:00–08:00 UTC range)
- Filter: breakout direction agrees with 4H trend (EMA slope or Supertrend on 4H)
- Exit: fixed ATR-based target (1.5× ATR) or end-of-session (17:00 UTC)
- Hypothesis: London open creates a directional push that fades or extends cleanly

---

## R008+ — If Still No Edge After R007

At this point, step back and analyze:
1. **Long vs short split**: run `--long-only` and `--short-only` variants of the best strategy.
   If one direction dominates, drop the other.
2. **Sub-period check**: split IS into 2024 (USD-strength) vs Jan–Sep 2025 (EUR-rally).
   A strategy that works in both sub-periods is more robust than one that only works in one.
3. **Volatility regime**: compute rolling 20-bar ATR percentile at entry. If wins cluster in
   high-ATR environments, add a volatility filter (enter only when ATR > Nth percentile).

---

## Critical Coding Rules

### Lookahead prevention — the most common source of bugs

**Multi-input indicators** (ATR, ADX, Supertrend, Stochastic): compute on raw OHLC,
then `.shift(1)` the **output** series. NEVER shift the input DataFrame.

```python
# CORRECT — Pitfall #7 pattern
adx_result = calc_adx(df_5m, di_period=14, adx_period=14)
adx = pd.Series(adx_result["adx"], index=df_5m.index).shift(1).values

st_result = calc_supertrend(df_5m, period=atr_period, multiplier=multiplier)
direction_5m = pd.Series(st_result["direction"]).shift(1).values

# WRONG — shifts H/L/C simultaneously, collapses RMA → NaN cascade → zero trades
st_result = calc_supertrend(df_5m.shift(1), ...)
```

**Single-series indicators** (EMA, SMA, BB, MACD, KAMA): shifting the input Close is safe.

```python
# CORRECT
ema_fast = calc_ema(df_5m["Close"].shift(1), length=9).values
```

### HTF alignment — always shift before merge_asof

```python
htf = pd.DataFrame({
    "Date":  htf_series.index,
    "value": htf_series.diff().shift(1).values,  # shift(1) BEFORE merge
})
merged = pd.merge_asof(ltf.sort_values("Date"), htf.sort_values("Date"),
                       on="Date", direction="backward")
```

### Using r006_supertrend_adx.py as template for new strategies
It contains all three correct patterns: Supertrend output shift, ADX output shift,
1H HTF direction shift before merge_asof. Use it as the reference implementation.

---

## Quality Bar

| Metric | Minimum to log | Good | Strong candidate |
|--------|---------------|------|-----------------|
| PF | any | > 1.5 | > 2.5 |
| Trades (IS) | any | ≥ 60 | ≥ 100 |
| Max DD% | — | < 15% | < 8% |
| Expectancy | — | > $0.03 | > $0.10 |

**OOS unlock threshold**: PF ≥ 2.0 AND trades ≥ 60 AND expectancy ≥ $0.05 on IS period.
Do not open OOS until at least one strategy clears all three bars.

---

## How to Add a New Strategy

1. Create `strategies/research/rNNN_strategy_name.py`
2. Export `DESCRIPTION`, `HYPOTHESIS`, `PARAM_GRID`, and
   `generate_signals(df_5m, df_1h, df_1d, **params) -> df_5m`
3. Use `r006_supertrend_adx.py` as template (has all three correct shift patterns)
4. Run: `python runner.py rNNN_strategy_name`
5. Log results in `RESEARCH_LOG.md`

---

## What to Log After Each Run (RESEARCH_LOG.md)

1. Update the **Leaderboard** table at the top (best result per strategy, sorted by PF)
2. Fill in the strategy's iteration block: hypothesis, param sweep, top 5 results table, analysis
3. State explicitly what the next strategy will be and why, based on the results

---

## Data Files

- `data/HISTDATA_EURUSD_5m.csv` — 161k bars, Jan 2024–Feb 2026
- `data/HISTDATA_EURUSD_1h.csv` — 1H bars, same range
- `data/HISTDATA_EURUSD_1d.csv` — Daily bars (22:00 UTC boundary), same range

Scaled-epoch timestamps (e.g. `1704146`) are auto-corrected in `engine/data.py`.

---

## Indicator Library (all in `indicators/` — do not re-implement)

ATR, ADX, KAMA, RSI, Supertrend, Bollinger Bands, Stochastic, Donchian, VWAP,
EMA/SMA/WMA/HMA/DEMA/TEMA, MACD, Ichimoku, Parabolic SAR, CCI, Williams %R,
OBV, MFI, CMF, Squeeze (TTM), Choppiness Index.

---

## Runner Config (current)

```python
BacktestConfig(
    initial_capital  = 1000.0,
    commission_pct   = 0.0046,   # $0.05/side per 1k units at ~1.10 ≈ $0.10 round-trip (OANDA)
    slippage_ticks   = 0,
    qty_type         = "fixed",
    qty_value        = 1000.0,   # 1,000 units per trade
    pyramiding       = 1,
    take_profit_pct  = 0.0,
    stop_loss_pct    = 0.0,
)
MIN_TRADES_FOR_RANK = 60
```
