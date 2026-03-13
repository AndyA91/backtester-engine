# Trading Strategies Project Memory

> **This file** = WHAT to do: rules, API surface, available indicators, pitfalls, coding standards.
> **BACKTESTING.md** = HOW things work under the hood: formulas, fill mechanics, edge cases, TV quirks.

## Key Rules
- **Always read MEMORY.md first** before starting work on any strategy.
- **Never suggest strategy code without backtest KPIs** — always run the backtest first. If given PineScript code, convert it to Python and run it through the Backtest Engine to get KPIs before presenting the strategy.
- **Warm up moving averages**: Fetch chart data going at least 2x as far back as the longest MA period before the strategy start date.
- **PineScript version**: Always use V6 or later when analysing or writing TradingView Pine Script code.
- **File naming**: When outputting a strategy filename, increment the number by one (or add it if missing). This number must also be reflected in the strategy title.
- **PineScript for best version**: When testing multiple strategy variations, automatically create the PineScript V6 file for the winning version (best backtest results). No need for the user to ask.
- **Data source**: Use TV CSV exports from `data/` when available (`load_tv_export()`). If the required data file is missing, **do NOT silently fetch it** — tell the user the file is missing and ask permission to fetch from an exchange using `fetch_crypto()`. Explain which exchange it will come from (e.g. "I'll fetch DOGE/USDT daily from Binance") so the user can open the same chart on that exchange in TradingView to compare numbers. If the user says numbers don't match, suggest exporting chart data directly from TradingView for an exact match. Explain how: on the TV chart, click the **Export chart data…** button (small download icon in the bottom-right of the chart pane), save the CSV, and place it in the `data/` directory.
- **Slippage: NOT SIMULATED** — we cannot simulate slippage because it requires super-granular tick/order-book data which is expensive to obtain and analyse. Always set slippage to 0 in the engine and mention this in every backtest result.
- **Backtest results must always show**: (1) which chart data was used and where it came from, (2) all strategy settings so the user can verify they match TV, (3) the slippage note (set to 0, not simulated), (4) First Order and Last Order dates (for verifying the trading range matches TV).
- **KPI display rule — always show both profit lines and use percentages**:
  1. **Total P&L (incl. open)** — first line, matches TV's Overview "Total P&L" = net_profit + open_profit. This is what the user sees first in TV.
  2. **Net Profit (closed)** — second line, matches TV's Excel "Net Profit" = sum of closed trade PnLs only.
  3. **Always show % alongside $** for Net Profit and Max Drawdown — users compare across different account sizes, so absolute $ alone is not useful.
  4. Show Max Drawdown as both $ and %.
- **Start-date safeguard**: The user's TV CSV export may not go back to the beginning of the chart. The engine auto-adjusts `start_date` to the first available bar and prints a warning. To match KPIs, the PineScript strategy **and** the Python backtest must use the same start date. Always code PineScript strategies with a configurable start date input. When the engine adjusts the date, tell the user to set the same date in TV's strategy properties (Date Range → Start Date).
- **Integration tests before shipping**: Run ALL test strategies in `dev/strategies/` and compare results with their `.xlsx` (TV data). Each test strategy is a triplet: `.py` (backtest), `.pine` (PineScript), `.xlsx` (TV export). Check: trade count, net profit, PF, win rate, max DD $ and %. Only ship if all tests pass. Current tests: example_ema_cross, test_gaussian_channel_ls, test_ichimoku_1, test_gaussian_channel_ls_flip, test_ema_cross_tpsl, test_ema_cross_tpsl_abs, test_easein_allout_1_0_atx, test_ema_cross_pct_equity, test_ema_cross_cash, test_ichimoku_5_0_trailing_sl_4h, test_buy_and_hold_1_0, test_simple_trend_regime_filter_1_0, test_native_trail_1_0, test_skyllet_trend_1_4_4h, test_fixed_qty, test_closedtrades, test_pyramiding.

## Pine Script Sanitization (applied to EVERY .pine before backtesting)

**MANDATORY**: Before converting any PineScript strategy to Python, check and fix ALL of these in `strategy()`. WARN the user about every change. If the Pine code violates any of them, **stop and fix it first**:

1. The Pine code must be modified to comply before conversion can proceed.
2. The user must apply the same changes in TradingView and re-export XLSX — otherwise TV numbers will differ.
3. Return the ENTIRE sanitized .pine to the user (not just changed lines).
4. Save the sanitized `.pine` in `strategies/`. The `.pine` and `.py` MUST share the same filename.

| # | Setting | Required Value | Why |
|---|---|---|---|
| 1 | **Date range** | Add `start_date="2018-01-01"` + `timeCondition` gate if missing | Exchange data starts ~2017; need 1yr warmup for indicators. **This is the #1 most common mistake.** |
| 2 | **Commission** | `commission_value=0.1, commission_type=strategy.commission.percent` if `0` or missing | Zero commission is unrealistic and gives misleading results |
| 3 | **Slippage** | `slippage = 0` | Engine cannot simulate slippage — requires tick/order-book data we don't have |
| 4 | **Margin Long / Margin Short** | `margin_long = 0, margin_short = 0` | TV's default 100% margin creates spurious margin-call mini-trades. TV has known bugs with non-zero margin. |
| 5 | **Bar Magnifier** | `use_bar_magnifier = false` | Engine uses TV's heuristic for intrabar TP/SL fill order. Bar Magnifier uses lower-TF tick data we don't have. |
| 6 | **Recalculate after order is filled** | `calc_on_order_fills = false` | **Forward-looking bias**: TV's own docs warn this causes forward-looking bias — must NEVER be used. |
| 7 | **On every tick** | `calc_on_every_tick = false` | Engine computes signals on bar close only. |
| 8 | **Initial capital** | Add `initial_capital=1000` if missing | Pine defaults to $1M. Doesn't affect percentages but makes dollar amounts mismatch if not set. Always set explicitly. |
| 9 | **XLSX data coverage** | XLSX trade range ⊆ CSV date range | If the XLSX shows trades outside CSV range, numbers will differ. Ask user to provide matching data or narrow the date range. |

**Quick self-check before proceeding:**
> Does the Pine have a date range? Is commission set to 0.1%? If NO to either, STOP and fix it first.

**How to communicate this to the user:**

> "Before I convert this strategy, I need to sanitize the Pine script. I found these issues: [list violations]. I'll fix the Pine code and give you the full sanitized version — you'll need to paste it into TradingView and re-export the XLSX so the numbers match. Should I proceed?"

## PineScript Coding Standards
- **`active` parameter**: Always use Pine Script V6's `active` parameter on checkboxes and pulldowns with a "No" option, so all related controls disable in tandem with the main settings input control.
- **Tooltips**: Always add comprehensive tooltips to ALL commands and fields in the Settings Inputs tab. Every `input.*()` call must have a `tooltip=` argument explaining what the setting does.

## Backtesting Engine
- Location: `backtest_engine/dev/` (development) and `backtest_engine/ship/` (sellable)
- Structure:
  - `engine/` — business logic: `engine.py` (core + indicators), `data.py` (data loaders), `__init__.py` (re-exports)
  - `data/` — chart data: CSV files + `cache/` for Bitstamp API
  - `strategies/` — strategy scripts: `example_ema_cross.py` (reference), proprietary tests
- Data: TV-exported CSV (`data/INDEX_BTCUSD, 1D.csv`) — exact same OHLC as TradingView
- Also supports Bitstamp API fetch (fallback)
- See [BACKTESTING.md](BACKTESTING.md) for TV matching internals (formulas, fill mechanics, edge cases)
- **Last bar is always dropped** (unfinished candle) in both data loaders
- All imports come from one place: `from engine import load_tv_export, BacktestConfig, ...`

## Strategy Template
Full copy-paste template for a new strategy:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import (
    load_tv_export,
    BacktestConfig, run_backtest, print_kpis,
    calc_ema, detect_crossover, detect_crossunder,
)

def my_strategy_signals(df, fast=9, slow=21):
    df = df.copy()
    df["fast_ema"] = calc_ema(df["Close"], fast)
    df["slow_ema"] = calc_ema(df["Close"], slow)
    df["long_entry"] = detect_crossover(df["fast_ema"], df["slow_ema"])
    df["long_exit"] = detect_crossunder(df["fast_ema"], df["slow_ema"])
    return df

def main():
    df = load_tv_export("INDEX_BTCUSD, 1D.csv")
    df = my_strategy_signals(df)

    config = BacktestConfig(
        initial_capital=1000.0,
        commission_pct=0.1,
        slippage_ticks=0,
        qty_type="percent_of_equity",
        qty_value=100.0,
        start_date="2018-01-01",
        end_date="2069-12-31",
    )

    kpis = run_backtest(df, config)
    print_kpis(kpis)

if __name__ == "__main__":
    main()
```

## Signal Generator Pitfall — timeCondition
When a strategy's signal generator tracks internal state (position, cooldown timer, highest-since-entry, etc.), it **must** include a time-range check matching Pine's `timeCondition`. Without it, if entry conditions happen to be True on bars **before** `start_date`, the generator enters a phantom position that the engine never fills (engine only acts within the trading range). This desyncs the generator's internal state from the engine and causes missing/wrong trades.

**Rule**: Any signal generator that uses internal position tracking must accept `start_date`/`end_date` and gate **both entry AND exit conditions** with `bar_in_range`:
```python
def generate_signals(df, start_date="2018-01-01", end_date="2069-12-31"):
    ts_start = pd.Timestamp(start_date)
    ts_end = pd.Timestamp(end_date)
    ...
    for i in range(n):
        bar_in_range = ts_start <= dates[i] <= ts_end
        close_long = (<exit conditions>) and bar_in_range  # matches Pine: closeLongCondition ... and timeCondition
        long_cond = (<entry conditions>) and bar_in_range  # matches Pine: longCondition ... and timeCondition
```

Indicators and non-trading state (chop detection thresholds, etc.) compute on ALL bars — only trading actions are gated. This matches PineScript where indicators always compute but `timeCondition` gates strategy.entry()/close_all().

**Why exits need gating too**: Exit conditions gate VWAP resets (VWAP anchors to last close signal) and affect state like `in_early_buy_signal` which modifies future close signals. Without exit gating, VWAP accumulates differently than PineScript.

**Why entries need gating**: Without it, phantom trades on early bars set `had_trades = True` (Pine's `strategy.max_contracts_held_all > 0`), enabling buyback/re-entry signals that should not be available yet.

Signal generators that don't track state (e.g. pure crossover signals without cooldown/trailing stop) are unaffected — the engine handles position tracking.

**`prevWasExitL` re-entry suppression**: When Pine code uses `prevWasExitL = (strategy.position_size[1] > 0) and (strategy.position_size == 0)` to suppress immediate re-entry after exits, the stateful signal generator must update this flag after BOTH signal exits (step 1: fills at Open) AND stop-loss exits (step 2: intrabar fills). A common bug is only checking after signal fills, missing the SL case — this allows re-entry on the bar immediately after a stop-loss exit when it should be suppressed.

**Bug discovered on:** test_ichimoku_5_0_trailing_sl_4h with ETH data. Entry conditions were True on 2017-12-31 (before start_date=2018-01-01). Generator entered phantom position, engine ignored it, all subsequent trades shifted by one.

## Engine Architecture
- `engine/engine.py` is **strategy-agnostic** — accepts any DataFrame with signal columns
- **Long-only (`run_backtest`):** requires `Open`, `High`, `Low`, `Close`, `long_entry`, `long_exit`
- **Long+Short (`run_backtest_long_short`):** also requires `short_entry`, `short_exit`
- **`Trade` dataclass** includes `direction` field (`"long"` or `"short"`) — set automatically by the engine
- **`print_trades()`** auto-shows a `Dir` column (LONG/SHORT) when short trades are present; long-only output stays clean
- Strategy signals are generated by separate functions in `strategies/` (e.g. `ema_cross_signals()`)
- To add a new strategy: create a file in `strategies/`, write a `*_signals(df)` function, then pass to `run_backtest(df, config)` or `run_backtest_long_short(df, config)`

## Required DataFrame Columns

**`run_backtest()` (long-only):**

| Column | Type | Required | Description |
|---|---|---|---|
| `Open` | float | Yes | Bar open price |
| `High` | float | Yes | Bar high price (used for intrabar drawdown + TP/SL) |
| `Low` | float | Yes | Bar low price (used for intrabar drawdown + TP/SL) |
| `Close` | float | Yes | Bar close price |
| `long_entry` | bool | Yes | True on bars where a long entry signal fires |
| `long_exit` | bool | Yes | True on bars where a long exit signal fires |
| `entry_qty` | float | No | Per-bar entry size in asset units (for pyramiding) |
| `tp_price` | float | No | Absolute take-profit price level |
| `sl_price` | float | No | Absolute stop-loss price level |
| `tp_offset` | float | No | TP distance from entry price (engine computes level) |
| `sl_offset` | float | No | SL distance from entry price (engine computes level) |

**`run_backtest_long_short()` — adds:**

| Column | Type | Required | Description |
|---|---|---|---|
| `short_entry` | bool | Yes | True on bars where a short entry signal fires |
| `short_exit` | bool | Yes | True on bars where a short exit signal fires |

## Available Indicators (in engine/engine.py)
- `calc_ema(series, length)` — EMA matching `ta.ema()`
- `calc_smma(series, length)` — Smoothed MA / RMA matching `ta.rma()`
- `calc_sma(series, length)` — Simple Moving Average matching `ta.sma()`
- `calc_rsi(series, length)` — RSI matching `ta.rsi()` — uses `ta.rma()` (SMMA/Wilder's) for gain/loss averaging
- `calc_atr(df, length)` — Average True Range matching `ta.atr()` = `ta.rma(ta.tr(), length)`
- `calc_macd(series, fast, slow, signal)` — MACD matching `ta.macd()`, returns `(macd_line, signal_line, histogram)`
- `calc_wma(series, length)` — Weighted MA matching `ta.wma()`
- `calc_hma(series, length)` — Hull Moving Average
- `calc_ehma(series, length)` — Exponential Hull MA
- `calc_thma(series, length)` — Triple Hull MA
- `calc_gaussian(series, length, poles)` — Gaussian filter (cascaded EMAs, 1–4 poles)
- `calc_highest(series, length)` — Highest value over N bars matching `ta.highest()`
- `calc_lowest(series, length)` — Lowest value over N bars matching `ta.lowest()`
- `calc_donchian(high, low, length)` — Donchian channel, returns `(upper, lower, mid)`
- `calc_obv(close, volume)` — On-Balance Volume matching `ta.obv`
- `calc_ichimoku(high, low, conv, base, span_b, disp)` — Ichimoku Cloud, returns dict with `conversion`, `base`, `lead_a`, `lead_b`, `displaced_lead_a`, `displaced_lead_b`
- `detect_crossover(fast, slow)` / `detect_crossunder(fast, slow)` — signal detection
- `get_source(df, source)` — price source selector (close/open/high/low/hl2/hlc3/ohlc4)
- All indicators handle NaN-leading input (safe to chain/cascade)

## Missing Indicators (manual implementation needed)
These PineScript built-ins are NOT in the engine indicator library. When converting strategies that use them, implement manually:

| Pine function | Python implementation |
|---|---|
| `ta.vwma(src, len)` | `(src * volume).rolling(len).sum() / volume.rolling(len).sum()` |
| `ta.barssince(cond)` | Iterative counter; returns `na` (use large sentinel like 99999) when condition was never true. `na >= 0` is `false` in Pine. |
| Supertrend | Stateful: adaptive upper/lower bands with ratcheting (`fuL`/`flL`) + direction tracking. Must be iterative. |
| Zero-Lag EMA | `calc_ema(close + (close - close[lag]), length)` where `lag = floor((length-1)/2)` |

## TradingView Settings for Matching
To get identical results between this engine and TradingView, set these in your TV strategy properties:

| Setting | Value |
|---|---|
| Margin Long | **0%** |
| Margin Short | **0%** |
| Slippage | **0** |
| Commission | Match your `commission_pct` (e.g. 0.1%) |
| `calc_on_every_tick` | `false` |
| `calc_on_order_fills` | `false` |
| `fill_orders_on_standard_ohlc` | `true` |
| `use_bar_magnifier` | `false` |
| `process_orders_on_close` | Match your `BacktestConfig.process_orders_on_close` (default: `false`) |

Setting margin to 100% (TV default) causes spurious margin-call mini-trades that inflate trade count.

## TradingView Matching Behavior
- EMA: standard formula, multiplier=2/(len+1), seed with SMA of first `len` bars
- Signals: crossover/crossunder detected on bar close
- Order fill: **next bar open** (calc_on_every_tick=false, fill_orders_on_standard_ohlc=true). With `process_orders_on_close=True`: fill at **same bar Close** instead.
- **Sizing** — controlled by `BacktestConfig.qty_type` and `qty_value`:
  - `"percent_of_equity"` (default, qty_value=100.0): invest N% of equity. At 100%, commission-adjusted at fill time: `trade_value = equity / (1 + rate)`, `qty = trade_value / fill_price`. At <100%, qty computed at signal time using bar Close.
  - `"cash"` (e.g. qty_value=500): invest a fixed dollar amount. `qty = cash_value / close` computed at signal time, fills at next bar's Open. Commission is charged on top (not deducted from the cash amount).
  - `"fixed"` (e.g. qty_value=0.1): invest a fixed number of asset units per entry. Matches PineScript's `strategy.fixed` / `default_qty_type=strategy.fixed`. `qty = qty_value` regardless of equity or price. Commission is charged on top.
- Commission: pct of trade value, applied on BOTH entry and exit
- **Margin: set to 0%** for long and short in TV to match engine (100% margin causes spurious margin-call mini-trades)
- **Slippage: set to 0** — slippage simulation is not possible without expensive tick-level data; always set to 0 in both the engine and TV for matching
- **Max Drawdown: intrabar methodology** matching TV's "Max equity drawdown (intrabar)":
  - Uses bar `Low` price for worst-case equity during open long positions (not just Close)
  - Uses bar `High` price for worst-case equity during open short positions
  - Peak equity only updates when **flat** (no open position) — unrealised mark-to-market highs do NOT update the peak
  - **Peak equity on reversal fills**: In flip/always-in-market strategies, `position_qty` is never 0 at end-of-bar because exit+entry fill on the same bar. Peak equity MUST update at the instant between exit fill and entry fill (v4.0 fix).
  - Max DD ($) and max DD (%) are tracked as **independent maximums** — they may occur at different trades
  - Ref: https://www.tradingview.com/support/solutions/43000681690
  - Required columns: `High` and `Low` (in addition to `Open`, `Close`)

## Short Selling Cash Model
- `run_backtest_long_short()` supports long+short positions (never simultaneously)
- Short PnL: `qty * (entry_price - exit_price) - entry_commission - exit_commission`
- Intrabar DD for shorts uses bar `High` (worst case for short = price spike up)
- **Critical:** At short exit, settle from `cash` (not mark-to-market `equity`). Using `equity` causes drift because mark-to-market is computed at a different price (prev bar close) than the fill price (current bar open). The correct formula: `cash = cash + gross_pnl - exit_commission`
- **Bug history:** Originally used `cash = equity + gross_pnl - exit_commission` which caused a $30K cumulative error over 63 trades on a $10K account due to compounding mark-to-market drift through each short position

## Reversal Logic (Long ↔ Short)
- TV's `strategy.entry("Short")` reverses: closes any open long AND opens a short on the same bar
- Engine supports this via reversal detection in `run_backtest_long_short()`:
  - If `pending_long_exit` AND `short_entry`, both queue → exit fills first, then entry fills at same bar's Open
  - If `pending_short_exit` AND `long_entry`, same treatment
- Strategy must set cross-signals: `short_entry` should also trigger `long_exit`, and `long_entry` should also trigger `short_exit`
- **Important**: When running both long-only and L+S backtests, use **separate DataFrame copies** — reversal signal modifications (e.g. `long_exit |= short_cross_under`) corrupt the long-only signals

## Net Profit & Open P&L
- `net_profit` = sum of **closed** trade PnLs (matches TV's "Net Profit")
- `open_profit` = unrealised P&L from any position still open at data end
- `final_equity` = `initial_capital + net_profit + open_profit`
- Open trades (exit_date=None) are included in the trades list but excluded from KPI stats

## Take Profit & Stop Loss
The engine supports TP/SL exits matching PineScript's `strategy.exit()` with `limit=` and `stop=`. Three ways to set levels (priority order — first match wins):

1. **Absolute price columns** (`tp_price` / `sl_price` in DataFrame) — per-bar absolute prices. Use when TP/SL levels are known ahead of time (e.g. fixed support/resistance).
2. **Offset columns** (`tp_offset` / `sl_offset` in DataFrame) — per-bar distance from entry price. The engine computes the absolute level at fill time: long TP = entry + offset, long SL = entry - offset (reversed for shorts). Use for ATR-based or any distance-based TP/SL.
3. **Config percentages** (`take_profit_pct` / `stop_loss_pct` in BacktestConfig) — fixed percentage from entry price. Simplest approach. Set to 0.0 to disable (default).

**TP/SL fill behavior** (matches TV without Bar Magnifier):
- Fills at **exact TP/SL price** on the bar where High/Low reaches it — not at next bar Open
- **Gap-through**: if bar Open already past TP/SL level → fills at Open
- **Both TP and SL hit on same bar**: TV heuristic — Open closer to favourable extreme → that side hit first
- **Entry bar skipped**: TP/SL is not checked on the entry bar itself (matches TV's order placement timing)
- **Signal vs TP/SL on same bar**: signal exits (at Open) take priority over TP/SL (intrabar)

**Timing for offset/price columns**: In PineScript, `strategy.exit()` runs at bar close and sets levels for the next bar. So offset columns should use **shifted** values: `df["tp_offset"] = (atr * mult).shift(1)`.

**Backward compatible**: When all TP/SL mechanisms are disabled (defaults), the engine behaves identically to signal-only mode.

### Trailing Stop via strategy.exit()

PineScript's `strategy.exit()` trailing stop has three parameters that must be used together:

| Parameter | Role | Required? |
|---|---|---|
| `trail_price` | Activation level (absolute price) | One of these two |
| `trail_points` | Activation level (profit in ticks from entry) | is required |
| `trail_offset` | Trailing distance in ticks from highest/lowest price | Always required |

**Critical**: `trail_offset` alone does NOT activate a trailing stop — it is silently ignored.
Use `trail_points=0` for immediate activation.

**Tick conversion**: trail distance in price = `trail_offset × syminfo.mintick`.
- INDEX:BTCUSD → mintick = 1.0
- Exchange pairs (BINANCE:BTCUSDT) → mintick = 0.01

**When converting PineScript to Python**:
- If Pine has `trail_offset` without `trail_points`/`trail_price` → no trailing stop is active, skip it
- If Pine has both → implement via stateful signal generator with `sl_price` column:
  ```python
  # Set for next bar (matches strategy.exit timing)
  sl_price_arr[i + 1] = highest_since_entry - trail_offset_price
  ```
  where `trail_offset_price = trail_offset_ticks × mintick`

**Native trailing stop implementation rules** (learned from test_native_trail_1_0):
1. **Exclude entry bar from tracking**: TV's trail order is placed at bar close and starts tracking from the NEXT bar. Do NOT include the entry bar's High/Low in `highest_since`/`lowest_since` — only update from `i > entry_bar_idx`:
   ```python
   if position == 1 and i > entry_bar_idx:
       highest_since = max(highest_since, highs[i])
   ```
2. **Monotonic ratchet**: The trail stop only moves in the favorable direction (up for longs, down for shorts). Even when ATR changes cause the raw trail level to move unfavorably, the trail level must not retreat:
   ```python
   raw_trail = highest_since - trail_off
   trail_stop = max(prev_trail, raw_trail)  # monotonic for longs
   prev_trail = trail_stop
   ```
3. **Combined stop + trail**: When Pine uses both `stop=` and `trail_points/trail_offset`, the effective SL = `max(static_stop, trail_stop)` for longs, `min(static_stop, trail_stop)` for shorts (whichever is tighter)
4. **Pine's `math.round()` vs Python's `round()`**: Pine uses standard rounding (half rounds up), Python uses banker's rounding (half rounds to even). Use `int(x + 0.5)` for positive values
5. **Stateful generator required**: Must simulate TP/SL exits internally to keep position state in sync with the engine, including entry_price, highest_since, and entry_bar_idx tracking

## Gaussian Channel (IIR Filter) — NOT `calc_gaussian`
Many TradingView strategies use the "Gaussian Channel" indicator (by DonovanWall and others). This uses a recursive N-pole IIR filter with binomial coefficients — it is **NOT** the same as `calc_gaussian()` in the engine (which is cascaded EMAs).

The IIR filter formula (`f_filt9x` in Pine):
```
f[i] = α^N × src[i] + Σ(k=1..N) (-1)^(k+1) × C(N,k) × (1-α)^k × f[i-k]
```
where C(N,k) are binomial coefficients.

**Critical pitfall**: The alpha/beta computation uses `1.414` (truncated √2), NOT `2`:
```python
# CORRECT — matches Pine's math.pow(1.414, 2/NS)
beta = (1 - cos(2*pi / period)) / (1.414 ** (2.0 / poles) - 1)
# WRONG — using 2 instead of 1.414 gives completely different channel
beta = (1 - cos(2*pi / period)) / (2 ** (2.0 / poles) - 1)
```
This single constant difference produces a completely different alpha, different filter output, and wrong trade signals.

Python implementation:
```python
from math import comb, cos, asin, sqrt
def gaussian_npole_iir(alpha, src, n_poles):
    """N-pole Gaussian IIR filter matching Pine's f_filt9x."""
    x = 1.0 - alpha
    n = len(src)
    f = np.zeros(n)
    for i in range(n):
        s = src[i] if not np.isnan(src[i]) else 0.0
        val = alpha ** n_poles * s
        for k in range(1, n_poles + 1):
            prev = f[i - k] if i >= k else 0.0
            val += (-1) ** (k + 1) * comb(n_poles, k) * x ** k * prev
        f[i] = val
    return f
```

## Process Orders on Close
When PineScript sets `process_orders_on_close = true`, orders fill at the **same bar's Close** instead of waiting for the next bar's Open. Set `BacktestConfig(process_orders_on_close=True)` to match.

**How it works:**
- Signal detected at bar Close → fills immediately at Close price (no pending queue)
- Exits fill before entries (same ordering as default mode)
- TP/SL from `strategy.exit(stop=)` still fills intrabar on subsequent bars at exact price
- `strategy.exit(stop=X)` set at bar Close: if Close breaches X → treated as signal exit at Close. Otherwise → pending for next bar's intrabar check.
- Entry bar = signal bar; TP/SL starts checking from next bar (`i > entry_bar_idx`)

**Signal generator considerations:**
- Entry price = `closes[i]` (not `opens[i+1]` as in default mode)
- ATR stop: check if Close already breaches stop → mark as signal exit. Otherwise set `sl_price_arr[i+1]` for intrabar check.
- Stateful generator still required for position tracking

**Backward compatible:** Default is `False` — existing strategies unchanged.

### Signal Generator Pre-fill Pitfall
With `process_orders_on_close`, Pine's `strategy.position_size` is **pre-fill** during script calc. Signal generators MUST use a snapshot of position state at bar start for ALL condition checks:
- `pos = position` at start of signal block (after TP/SL simulation)
- Use `pos` (not the live `position`) for entry/management condition checks
- Update `position` only at the END of the bar (deferred update)
- Add **reversal guards**: when a reversal entry fires, suppress the old position's management block (matches Pine v1.4 pattern)
- The engine automatically warns if it detects reversal + same-bar exit conflicts

Without this, management blocks fire on reversal bars (because position was updated immediately), creating spurious exit signals that desync the generator from the engine.

## Pyramiding & Variable Position Sizing
The engine supports pyramiding (multiple simultaneous entries) and per-bar entry quantities, matching PineScript's `pyramiding=N` and `strategy.entry(qty=...)`. Both `run_backtest()` (long-only) and `run_backtest_long_short()` support pyramiding > 1.

**BacktestConfig setting:**
- `pyramiding: int = 1` — maximum number of simultaneous open sub-positions. Set to 1 (default) for single-position strategies. Set higher for pyramiding strategies (e.g. `pyramiding=100` for ease-in patterns, `pyramiding=2` for L+S pyramid adds).

**Optional DataFrame column:**
- `entry_qty` (float) — per-bar entry size in asset units. When present, the engine uses this qty instead of computing from equity. Matches PineScript's `strategy.entry(qty=...)`. When absent, the engine sizes from equity as before.

**Behavior:**
- Each sub-position is tracked as a separate `Trade` with its own entry date, price, qty, and P&L
- `long_exit=True` triggers `strategy.close_all()` — closes every open sub-position at the next bar's Open
- Each closed sub-position becomes a separate trade in the trade list
- Signal detection uses `len(open_positions) < config.pyramiding` instead of `position_qty == 0`
- TP/SL uses the **first** sub-position's entry price for level calculation and closes all positions when triggered

**Backward compatible**: When `pyramiding=1` (default) and no `entry_qty` column, behavior is identical to single-position mode.

**Ease-in pattern example** (PineScript `strategy.fixed` with `pyramiding=100`):
- Strategy signal sets `long_entry=True` and `entry_qty = equity_to_invest / close` on each bar it wants to add
- Engine fills each entry at next bar's Open with the specified qty
- On close signal, all sub-positions are closed at once

## Data Loading
- `load_tv_export()` preserves `OnBalanceVolume` column if present in the TV CSV export — useful for strategies that use `ta.obv`
- `load_tv_export()` only drops rows where OHLC data is NaN — auxiliary columns like OBV with NaN on the first bar do NOT cause bar removal
- `fetch_btc_daily()` fetches from Bitstamp API as fallback (BTC/USD daily only)
- `fetch_crypto(symbol, timeframe, start, end)` — multi-exchange crypto fetcher via `ccxt`:
  - **Symbol formats**: `"BTC"`, `"BTCUSDT"`, `"BTC/USDT"`, `"BINANCE:SOLUSDT"` — auto-resolves quote currency (USDT→USD→BUSD→USDC)
  - **Timeframes**: TV notation (`"240"`, `"1D"`, `"W"`) or ccxt notation (`"4h"`, `"1d"`, `"1w"`)
  - **Exchange fallback chain**: binance → coinbase → kraken → bybit → kucoin → okx → gate → bitget → mexc. Uses first exchange returning ≥50 candles.
  - **Ticker discovery**: CoinGecko `/api/v3/coins/list` for error messages when symbol not found
  - **Caching**: Saves to `data/cache/` as CSV; subsequent calls load from cache. Pass `use_cache=False` to force refetch.
  - **Returns**: Same DataFrame format as other loaders (Open, High, Low, Close, Volume; DatetimeIndex)
  - **Last bar dropped** (unfinished candle), same as other loaders
  - **Note**: Exchange prices may differ slightly from TradingView INDEX prices (e.g. INDEX:BTCUSD). For exact TV matching, use `load_tv_export()` with a TV CSV export.
  - **Requires**: `pip install ccxt` (not in base requirements.txt yet — dev only)
- `read_tv_xlsx_dates(xlsx_path)` — reads dates from a TV XLSX export's Properties sheet. Returns dict:
  - `pine_start`, `pine_end`: Pine's Start Date / End Date input parameters (timeCondition dates)
  - `range_start`, `range_end`: Observed Trading Range (actual trade dates)
  - Handles all TV property name variants (`"Start Date"`, `"Date Start"`, `"Backtest Start Date"`)
  - Use `pine_start`/`pine_end` for signal generator dates, `range_end` for BacktestConfig when exit depends on `last_bar_index`
  - **Only for XLSX validation** — without XLSX, use dates from Pine script params and available chart data

## Multi-Timeframe (MTF) Rules
- **Fetch each timeframe separately** from Bitstamp (or TV export per TF)
- **NEVER allow look-ahead bias** when using a higher timeframe (HTF) as a filter/signal
- A higher TF bar (e.g. 1W) is **not closed yet** while trading on a lower TF (e.g. 1D)
  → The HTF value must only update once the HTF bar actually closes
  → On a daily chart using weekly data: the weekly value only changes on the weekly close bar (e.g. Sunday/Monday), NOT mid-week
- Implementation approach: forward-fill (`ffill`) the HTF indicator onto the LTF index, shifted by one HTF bar, so each LTF bar only sees the **last completed** HTF value
- Always verify: no LTF bar should reference an HTF bar whose close date is in the future relative to that LTF bar

## Reference Backtest: EMA Cross on INDEX:BTCUSD 1D
- Data: TradingView export `INDEX_BTCUSD, 1D.csv`
- **Long-only (EMA 9/21):** Net Profit $9,180 (917.95%), 60 trades, 33.33% win rate, PF 1.813, Max DD -$3,420 (-61.01%)
- **Long+Short (EMA 9/21 long + EMA 5/13 short, with reversals):** Net Profit $1,501 (150.11%), 155 trades, PF 1.167, Open P&L ~$513
- First trade: Entry 2018-01-06 @ $16,955.45
- **TV with 100% margin**: 86 trades (59 real + 27 margin calls), Net Profit $9,736 (973.59%)
- **TV with 0% margin**: matches Python's 60 trades

## Known Precision Limitations

### Pyramiding L+S with percent_of_equity sizing — near-match only
- **Result**: 188 engine trades vs 190 TV trades (~4.8% net profit diff)
- **Root cause**: Sub-cent OHLC precision differences (e.g., engine `13340.705000` vs TV's internal float) compound through RSI's exponential smoothing (SMMA/RMA) over ~960 bars. This shifts RSI crossunders near the 40/60 thresholds by 1 bar, which cascades through percent-of-equity sizing into all subsequent trades.
- **Not a logic bug**: The engine's pyramiding logic, RSI calculation, and SMMA algorithm are all verified correct. The gap comes from floating-point accumulation in indicator calculations — unfixable without TV's exact internal float values.
- **3 missing trades pattern**: All 3 TV extras follow the same pattern — reversal from 2 fully-pyramided longs, then a SHORT pyramid add the next day. The deferred pyramid gate (`n_positions < pyramiding`) blocks these because `n_positions` still reflects the old direction's count at check time.
- **Test tolerance**: ±3 trades, ±6% net profit, ±10% commission (test_pyramiding.py)

## Requirements
- Python 3.10+
- pandas, numpy, requests, ccxt, openpyxl (see `requirements.txt`)
