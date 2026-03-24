# Test Coverage Analysis

## Current State

The project has **zero automated tests** (no pytest, unittest, or any test framework). Testing is done via manual comparison scripts (`dc_v2_test.py`, `dc_v3_test.py`, `dc_v2_validate.py`, `dc_v3_validate.py`) that compare engine output against TradingView CSV exports. There is no CI/CD pipeline.

This means any regression in the core engine — indicators, fill logic, KPI math — would go unnoticed until a strategy produces visibly wrong results.

---

## Priority 1: Core Engine Unit Tests (Critical)

### 1a. `_check_tpsl_fill()` — TP/SL fill logic (`engine.py:436`)

This function has **6 code paths** (gap-through, intrabar TP-only, SL-only, both-hit heuristic, offset vs pct vs absolute price) and is the most bug-prone part of the engine. It is purely functional (no side effects), making it trivially testable.

**Tests needed:**
- Long TP hit at exact level
- Long SL hit at exact level
- Short TP hit (bar low <= level) and short SL hit (bar high >= level)
- Gap-through: bar opens past TP or SL → fill at Open, not level
- Both TP and SL hit intrabar → TV heuristic (Open closer to favorable extreme wins)
- Priority: `tp_price` > `tp_offset` > `tp_pct` (same for SL)
- Edge: all disabled (tp_pct=0, sl_pct=0, no columns) → returns `(None, "")`

### 1b. Indicator functions — TV-matching correctness

Each indicator claims to match TradingView's `ta.*` exactly. A single-value regression breaks every strategy using it. Test against known TradingView output.

**Tests needed (highest-value indicators first):**
- `calc_ema()` — seed from SMA of first N valid values, NaN-leading input handling, NaN carry-forward
- `calc_smma()` / `calc_rsi()` — Wilder's smoothing, not simple rolling mean
- `calc_atr()` — true range formula (3-way max), then SMMA
- `calc_macd()` — tuple return (line, signal, histogram)
- `calc_donchian()` — highest/lowest over period
- `detect_crossover()` / `detect_crossunder()` — edge: equal values on prior bar, NaN handling
- `calc_wma()`, `calc_hma()`, `calc_ehma()`, `calc_thma()` — weight correctness
- `calc_gaussian()` — cascaded EMA with pole count clamping (1-4)
- `calc_ichimoku()` — multi-component output

**Approach:** Create a small (30-50 bar) synthetic OHLCV DataFrame. Compute each indicator. Assert values at specific indices against hand-calculated or TV-exported reference values.

### 1c. `compute_kpis()` — KPI math (`engine.py:1554`)

Pure function, easy to test with a crafted list of `Trade` objects.

**Tests needed:**
- Win rate, profit factor, avg trade with known trades
- Edge: no trades → returns `{"error": "No trades executed"}`
- Edge: all winners (gross_loss=0) → profit_factor = inf
- Edge: open trades (no exit_date) counted correctly
- Consecutive wins/losses counting
- Commission totals

---

## Priority 2: Backtest Engine Integration Tests (High)

### 2a. `run_backtest()` — long-only engine (`engine.py:549`)

**Tests needed:**
- Simple 1-trade scenario: entry signal on bar N, fills at bar N+1 Open, exit signal fills at bar M+1 Open
- PnL = qty * (exit - entry) - commissions
- `process_orders_on_close=True` → fill at same bar's Close, not next Open
- Pyramiding: multiple entries, `long_exit` closes all
- Position sizing: `percent_of_equity`, `cash`, `fixed`
- Date filtering: trades only within `start_date`/`end_date`
- TP/SL interaction with signal exits
- Edge: entry and exit signal on same bar
- Edge: entry signal with insufficient equity (after losses)

### 2b. `run_backtest_long_short()` — long+short engine (`engine.py:904`)

**Tests needed:**
- Long entry → long exit → short entry → short exit sequence
- Reversal: long position open + short_entry → closes long, opens short
- Short PnL: qty * (entry - exit) - commissions
- Pyramiding in both directions
- `process_orders_on_close` conflict detection (warning)
- Intrabar drawdown: uses bar Low for longs, bar High for shorts

---

## Priority 3: Data Loading Tests (Medium)

### 3a. `load_tv_export()` — CSV parsing (`data.py`)

**Tests needed:**
- Standard TV CSV format parsed correctly (OHLCV columns, DatetimeIndex)
- Last bar dropped (unfinished candle rule)
- Date range filtering works
- Warm-up bars preserved before start_date
- Edge: CSV with missing columns → clear error message

### 3b. `load_renko_export()` — Renko CSV parsing (`renko/data.py`)

**Tests needed:**
- Fractional timestamps preserved (no duplicate indices)
- `brick_up` column computed correctly (Close > Open)
- Last brick dropped
- Edge: < 2 bricks → ValueError
- Edge: missing file → FileNotFoundError with helpful message

### 3c. `_parse_date_range()` — date string parsing (`data.py:25`)

**Tests needed:**
- Em-dash separator: `"Jan 02, 2018 — Feb 17, 2026"`
- Hyphen fallback
- Intraday timestamps get `%H:%M` format
- Daily timestamps get date-only format

---

## Priority 4: Standalone Indicator Modules (Medium)

The `indicators/` directory has 15+ standalone modules (adx, bbands, supertrend, parabolic_sar, stochastic, squeeze, etc.) used by the Renko indicator enrichment pipeline. None have tests.

**Highest-value tests:**
- `indicators/supertrend.py` — direction flips, ATR-based bands
- `indicators/parabolic_sar.py` — state machine (acceleration factor logic)
- `indicators/adx.py` — DI+/DI- and ADX smoothing
- `indicators/stochastic.py` — %K smoothing, %D signal line
- `indicators/squeeze.py` — BB-inside-Keltner detection + momentum

**Approach:** Same synthetic DataFrame strategy. Validate output shape, NaN warm-up period length, and spot-check key values.

---

## Priority 5: Renko Indicator Enrichment (Lower)

### `add_renko_indicators()` (`renko/indicators.py:78`)

**Tests needed:**
- All 25+ columns are added to the DataFrame
- All columns are shifted by 1 bar (pre-shifted convention)
- No NaN in the body of the DataFrame (after warm-up)
- Column names match documentation

---

## Recommended Test Infrastructure

```
tests/
├── conftest.py              # Shared fixtures: synthetic OHLCV DataFrames, sample Trade lists
├── test_indicators.py       # Priority 1b: calc_ema, calc_rsi, calc_atr, etc.
├── test_tpsl.py             # Priority 1a: _check_tpsl_fill edge cases
├── test_kpis.py             # Priority 1c: compute_kpis math
├── test_backtest.py         # Priority 2a: run_backtest long-only
├── test_backtest_ls.py      # Priority 2b: run_backtest_long_short
├── test_data.py             # Priority 3: load_tv_export, parse helpers
├── test_renko_data.py       # Priority 3b: load_renko_export
├── test_indicator_modules.py # Priority 4: standalone indicator modules
└── test_renko_indicators.py # Priority 5: add_renko_indicators pipeline
```

**Framework:** pytest (standard, no dependencies beyond what's already used).

**Fixtures (`conftest.py`):**
- `tiny_ohlcv()` — 50-bar synthetic DataFrame with predictable Open/High/Low/Close
- `sample_trades()` — list of Trade objects with known PnL for KPI tests
- `ohlcv_with_signals()` — DataFrame with long_entry/long_exit columns pre-set

---

## Impact Summary

| Priority | Area | Risk if Untested | Effort |
|----------|------|-------------------|--------|
| 1a | TP/SL fill logic | Wrong fill prices → invalid strategy results | Low (pure function) |
| 1b | Indicator math | Silent regressions break all strategies | Medium |
| 1c | KPI computation | Misleading performance metrics | Low (pure function) |
| 2 | Backtest engines | Wrong trade execution → false strategy validation | Medium-High |
| 3 | Data loading | Bad data silently corrupts backtests | Low-Medium |
| 4 | Indicator modules | Renko pipeline produces wrong signals | Medium |
| 5 | Renko enrichment | Pre-shift convention violated → lookahead bias | Low |

**Recommended starting point:** Priorities 1a + 1c (pure functions, highest ROI, can be written in <1 hour) followed by 1b (indicators).
