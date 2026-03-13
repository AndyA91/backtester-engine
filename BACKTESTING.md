# Backtesting Engine — TradingView Matching Internals

> **This file** = HOW things work under the hood: formulas, fill mechanics, edge cases, TV quirks.
> **MEMORY.md** = WHAT to do: rules, API surface, available indicators, pitfalls, coding standards.

## Order of Operations Per Bar (matching TV)
1. Fill any pending orders at this bar's **Open** price
2. Update equity mark-to-market at bar's **Close**
3. Detect signals (crossover/crossunder) at bar's **Close**
4. Queue pending orders for next bar

## EMA Calculation
- Multiplier: `2 / (length + 1)`
- Seed value: SMA of the first `length` bars
- Applied from index `length-1` onward
- Seed location doesn't matter after ~100 bars (converges exponentially)
- **NaN handling**: If input is NaN after the seed, the EMA carries forward the last valid value (matches Pine behavior). SMMA/RMA uses the same carry-forward logic.

## Position Sizing
Controlled by `BacktestConfig.qty_type` and `BacktestConfig.qty_value`.

**percent_of_equity** (default, qty_value=100.0):
- At 100%: commission-adjusted at fill time: `trade_value = equity / (1 + commission_rate)`, `qty = trade_value / fill_price`. Ensures total outlay never exceeds equity.
- At <100% (e.g. 50%): qty computed at signal time using bar Close: `target = equity * pct / 100`, `qty = target / ((1 + rate) * close)`. Fills at next bar's Open.

**cash** (e.g. qty_value=500):
- Fixed dollar amount per trade. `qty = cash_value / close` computed at signal time.
- Fills at next bar's Open. Trade value = qty * open (may differ slightly from cash_value).
- Commission is charged on top (not deducted from the cash amount).

TV computes qty at signal time (bar close) for both cash and partial percent_of_equity. The engine matches this behavior.

## Pyramiding & Variable Entry Quantities
- `BacktestConfig(pyramiding=N)` allows up to N simultaneous sub-positions (default: 1)
- Optional `entry_qty` column in DataFrame: per-bar entry size in asset units, matching `strategy.entry(qty=...)`
- When `entry_qty` is present, the engine uses the specified qty; when absent, sizes from equity
- `long_exit=True` closes ALL open sub-positions at once (`strategy.close_all()`)
- Each sub-position tracked as a separate Trade with individual P&L
- Signal detection: entry allowed when `len(open_positions) < config.pyramiding`
- TP/SL with pyramiding: uses first sub-position's entry price, closes all on trigger

## Commission
- Percent-based: `commission_value / 100` (e.g., 0.1% → 0.001)
- Applied on **both** entry and exit
- Entry commission = `position_value * rate`
- Exit commission = `exit_value * rate`
- Net PnL = gross PnL - entry_commission - exit_commission

## Crossover / Crossunder Detection
- Crossover: `prev_fast <= prev_slow AND curr_fast > curr_slow`
- Crossunder: `prev_fast >= prev_slow AND curr_fast < curr_slow`
- Uses shift(1) for previous bar values
- **NaN handling**: If any input is NaN, the comparison returns False (not NaN). This prevents signal loss after NaN gaps in data.

## TradingView PineScript Settings (must match)
- `calc_on_every_tick=false` → signals on bar close, fill on next bar open
- `fill_orders_on_standard_ohlc=true` → fill at open price
- **`margin_long=0, margin_short=0`** → CRITICAL: set to 0% to avoid margin call mini-trades
- With 100% margin, TV generates ~27 spurious margin-call trades (tiny qty, -0.20% PnL each)
- These margin calls happen when entry signal fires and TV does a margin check
- They alter equity slightly, causing trade count and PnL to diverge from our engine

## Data Source
- Use TradingView CSV export for exact OHLC match (e.g., `INDEX_BTCUSD, 1D.csv`)
- TV export format: `time` (unix ts), `open`, `high`, `low`, `close` — no volume
- TV exports may include auxiliary columns like `OnBalanceVolume` — the loader preserves these
- `load_tv_export()` only drops rows where OHLC data is NaN — auxiliary columns with NaN on the first bar (e.g. OBV) do not cause bar removal
- Bitstamp API available as fallback but prices differ slightly from INDEX:BTCUSD
- Yahoo Finance BTC-USD also differs — not recommended

## Reversal Behavior (Long ↔ Short)
- TV's `strategy.entry("Short")` implicitly reverses: closes any open long AND opens short on the same bar
- The engine supports this via reversal detection at step 3:
  - If both `long_exit` and `short_entry` fire on the same bar, both are queued
  - On the next bar's Open: exit fills first (position → flat), then entry fills immediately after
- Strategy signals must include cross-triggers: `short_entry` should also mark `long_exit`, and vice versa
- Without reversal signals, the engine treats long and short as independent (positions never overlap)

## Peak Equity Update on Reversal (v4.0 fix)
- Peak equity normally updates only when flat (`position_qty == 0`) at end of bar
- In flip/always-in-market strategies, exit+entry fill on the same bar — `position_qty` is never 0 at end-of-bar
- **Fix:** Peak equity updates at the instant between exit fill and entry fill, while the position is momentarily flat
- Without this fix, peak equity stays at `initial_capital` forever, producing wildly incorrect Max Drawdown values

## Net Profit vs Open P&L
- `net_profit` = sum of **closed** trade PnLs only (matches TV's "Net Profit")
- `open_profit` = unrealised P&L of any position still open when data ends
- `final_equity` = `initial_capital + net_profit + open_profit`
- TV reports these separately: "Net Profit" (closed) and "Open P&L" (unrealised)

## Process Orders on Close (`process_orders_on_close = true`)
When enabled, orders fill at the **same bar's Close** instead of the next bar's Open.

**Order of operations per bar (replaces the default flow):**
1. Check TP/SL intrabar (from previous bar's `strategy.exit()`) — fills at exact stop/limit price
2. Intrabar drawdown check (bar's Low for longs, High for shorts)
3. Mark-to-market equity at Close
4. Detect signals at Close
5. **Fill at Close**: exits first, then entries (same bar — no pending queue)

**Key differences from default mode:**
- Fill price = bar's Close (not next bar's Open)
- Entry bar = signal bar (TP/SL skip check `i > entry_bar_idx` starts from next bar)
- No intrabar drawdown on entry bar (position opened at Close, not during bar's range)
- `strategy.exit(stop=X)` placed at Close: if Close breaches X → treated as signal exit at Close. Otherwise → checked intrabar on subsequent bars.

**Position sizing:**
- 100% equity: `trade_value = equity / (1 + rate)`, `qty = trade_value / Close`. Equity is mark-to-market at Close before the fill.
- Cash / partial equity: qty computed at signal time using Close (same price as fill → no difference).

**Backward compatible:** Default is `False` — existing strategies are unaffected.

### Pitfall: Pre-fill Position State in Signal Generators

**Problem**: With `process_orders_on_close`, Pine's `strategy.position_size` during script calc reflects the position **BEFORE** any fills on the current bar. If your signal generator updates position state immediately after setting entry signals, management blocks fire incorrectly:

```python
# ❌ BUG: position updates immediately, management fires on reversal bar
if le and position <= 0:       # SHORT → LONG reversal
    short_exit[i] = True
    position = 1               # updated immediately!
    long_entry[i] = True
if position == 1:              # fires because we just set position=1
    if lx:
        long_exit[i] = True    # spurious exit on entry bar!
```

**Fix**: Snapshot position at bar start, use it for ALL condition checks:

```python
pos = position                 # pre-fill snapshot
long_reversal = le and pos <= 0
if long_reversal:
    short_exit[i] = True
    long_entry[i] = True
    next_pos = 1               # deferred update
elif pos == 1:                 # uses pre-fill — skips on reversal bar
    if lx:
        long_exit[i] = True
        next_pos = 0
if next_pos is not None:
    position = next_pos        # apply at end of bar
```

**Why it matters**: The spurious exit-on-entry-bar causes a signal desync between the generator and the engine. The generator thinks the position exited, but the engine's step 3.5 processes exits before entries and misses the long exit (position was still short when checked). This cascades through all subsequent trades.

**Pine reversal guards** (v1.4 pattern): When a reversal entry fires, the old position's management block should be suppressed:
```pine
longReversalFiring  = longEntry and strategy.position_size <= 0
shortReversalFiring = shortEntry and strategy.position_size >= 0
if strategy.position_size > 0 and not shortReversalFiring
    // long management
if strategy.position_size < 0 and not longReversalFiring
    // short management
```

**Engine detection**: The engine automatically warns when it detects reversal + same-bar exit conflicts in the signal DataFrame (entry + exit + counter-exit on the same bar).

## Validation Approach
- Compare trade count (must match exactly with 0% margin)
- Compare first 5 trades (entry date, price, qty)
- Compare net profit, PF, max DD, win rate
- Small qty differences OK (commission drag accumulates slightly differently)
