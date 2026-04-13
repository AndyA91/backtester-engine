# CLAUDE.md — Backtester Engine

## What This Project Is

A Python backtesting engine that matches TradingView's strategy execution exactly. Used to develop, sweep, and validate Renko trading strategies before deploying Pine Script to TradingView for live trading on OANDA.

## Project Structure

```
engine/          Core backtest engine (engine.py, data.py). All imports: `from engine import ...`
indicators/      Python indicator modules (.py) + Pine reference implementations (.pine)
  LuxAlgo/       6 ported LuxAlgo indicators
renko/           Renko-specific research framework
  strategies/    Strategy modules (.py) + Pine scripts (.pine)
  runner.py      Sweep runner (parallel, IS/OOS split)
  data.py        Renko CSV loader (fractional timestamps)
  indicators.py  Pre-shifted indicator enrichment (44 columns)
  config.py      MAX_WORKERS = cpu_count() - 4
  *.py           Sweep scripts (btc_hf_sweep.py, btc_mk_sweep.py, etc.)
data/            Market data CSVs organized by instrument subdirectory
  BTCUSD/        BTC Renko 150/300 + daily/weekly
  EURUSD/        EURUSD Renko 0.0004-0.0012 + daily
  GBPJPY/        GBPJPY Renko 0.05-0.15
  EURAUD/        EURAUD Renko 0.0006-0.0018
  ...            GBPUSD, USDJPY, USDCHF, MYM, USO, crypto
ai_context/      Sweep result JSONs (for cross-session analysis)
tvresults/       TradingView strategy export CSVs (ground truth)
liveresults/     OANDA live trade logs + TV comparison CSVs
```

## How to Run Things

### Single strategy backtest
```bash
cd renko && python runner.py r001_brick_count
python runner.py r001_brick_count --renko "OANDA_BTCUSD.SPOT.US, 1S renko 150.csv"
```

### Sweep scripts (parallel)
```bash
python renko/btc_mk_sweep.py              # BTC Momentum King sweep
python renko/btc_hf_sweep.py              # BTC high-frequency sweep
python renko/forex_luxalgo_sweep.py       # Forex LuxAlgo sweep
```

All sweeps use `ProcessPoolExecutor(max_workers=MAX_WORKERS)` from `renko/config.py`.

## Sweep Workflow Protocol — MANDATORY

Every sweep follows this exact protocol. **Skipping a step is a process violation.** This protocol exists because sweeps deposit lessons into memory only if pre/post rituals run consistently — otherwise each sweep is a one-shot result that gets forgotten.

### Before launching ANY sweep
1. **Run `/pre-sweep-check <script>` first.** This walks through hypothesis, baseline, falsification, dead-end check, look-ahead audit, and compute budget. Output is a `GO` / `GO WITH FIXES` / `NO-GO` verdict.
2. **Wait for user GO** before launching the sweep. Even if pre-check returns `GO`, the user has the final approval.
3. **A PreToolUse hook will warn** if you attempt to launch a sweep script via Bash without a recent pre-check marker. The warning is informational — don't override it without a stated reason (e.g. "re-running an already-pre-checked sweep with no changes" or "smoke test").

### After a sweep completes
1. **Run `/sweep-postmortem <script-or-result-json>` immediately.** Don't move on to other tasks first — surprises decay fast.
2. **A PostToolUse hook will inject a MANDATORY reminder** when it detects a sweep script just finished. Treat this as a hard rule, not a suggestion.
3. The postmortem is the *only* mechanism that updates `meta_rules.md`, `cross_instrument_carryover.md`, `phase_status.md`, `backtest_counter.md`, and topic files. Skipping it means the sweep produced a JSON nobody will ever read again.

### When the user requests a sweep
The default behavior is:
1. Locate or create the sweep script
2. Run `/pre-sweep-check` against it
3. Surface the verdict, await user GO
4. Launch the sweep via Bash
5. When the PostToolUse hook fires, immediately run `/sweep-postmortem`
6. Apply the postmortem's proposed memory updates after user confirmation

The user should NOT have to type the slash commands. They run on autopilot as part of the sweep flow. Slash commands remain available for explicit ad-hoc use (e.g. postmortem on an old result).

### When NOT to run the protocol
- Smoke tests (e.g. `python renko/foo_sweep.py --dryrun` if such a flag exists)
- Re-launching an identical sweep within the same session for debugging (pre-check already ran)
- The user explicitly says "skip the protocol" or "just run it"

In all cases, state out loud what you're doing and why ("skipping pre-check because this is a re-run of the sweep we just pre-checked at HH:MM").

## Renko Strategy Conventions

### Strategy module structure
Every strategy in `renko/strategies/` must export:
- `DESCRIPTION` — one-line summary
- `HYPOTHESIS` — what edge we expect and why
- `PARAM_GRID` — dict of param lists for sweep
- `generate_signals(df, **params)` — returns df with `long_entry`, `long_exit`, `short_entry`, `short_exit` bool columns

Copy `renko/strategies/_template.py` for new strategies.

### Pre-shifted indicators
`renko/indicators.py` adds 44 columns to the DataFrame, ALL pre-shifted by 1 bar. This means at row `i`, the indicator value was computed through bar `i-1`. **Use values at `[i]` directly in signal loops — no additional shifting needed.**

Available columns: `adx`, `plus_di`, `minus_di`, `rsi`, `macd`, `macd_sig`, `macd_hist`, `ema9/21/50/200`, `atr`, `vol_ema`, `vol_ratio`, `chop`, `st_dir`, `bb_upper/lower/mid/bw/pct_b`, `kama`, `kama_slope`, `cmf`, `mfi`, `obv`, `obv_ema`, `psar_dir`, `stoch_k/d`, `sq_momentum`, `sq_on`, `mk_momentum`, `mk_signal`, `mk_strength`, `mk_cross_up`, `mk_cross_dn`, `mk_nz`

### Exit convention
First brick in the opposing direction = exit. This is the standard for all Renko strategies.

### IS/OOS split
- **IS**: Start auto-detected per instrument → 2025-09-30
- **OOS**: 2025-10-01 → 2026-03-19 (sealed)

### BTC-specific conventions
- **Long only** — no shorting on OANDA BTC
- **Cash mode** — `qty_type="cash"`, `qty_value=20` ($20 notional per trade)
- **Commission**: 0.0046%
- **Exit**: First down brick (optimal — all alternatives tested and degraded PF)
- **No volume data**: BTC Renko exports have Volume=0 for IS period

### Forex conventions
- **Long + Short** — `run_backtest_long_short`
- **Fixed qty** — `qty_type="fixed"`, `qty_value=1000`
- **Commission**: 0.0046%

## Pine Script Sanitization Checklist

**MANDATORY before writing ANY Python strategy code or converting Pine to Python.**

| # | Setting | Required Value | Why |
|---|---------|---------------|-----|
| 1 | `margin_long` / `margin_short` | `0` | 100% margin creates spurious margin-call mini-trades |
| 2 | `start_date` | Add as `input.time` if missing | Must match Python IS/OOS dates |
| 3 | `calc_on_order_fills` | `false` | Forward-looking bias (TV docs warn about this) |
| 4 | `commission_value` | `0.0046` (percent) | Match OANDA costs |
| 5 | `calc_on_every_tick` | `false` | Engine computes on bar close only |
| 6 | `use_bar_magnifier` | `false` | Engine doesn't have sub-bar tick data |
| 7 | `initial_capital` | Set explicitly (e.g. `1000`) | Pine defaults to $1M |

## Critical Pitfalls

### 1. Pre-shifted indicators — shift OUTPUT, not input
Multi-input indicators (ADX, ATR, Stochastic, PSAR): compute on raw OHLC, then `.shift(1)` the output series. Never shift input columns.

### 2. Stateful signal generators — gate on `bar_in_range`
Any generator with cooldown/position state MUST gate ALL state mutations on `bar_in_range`. Without this, phantom trades before `start_date` desync generator state from the engine.

### 3. Pine HTF slope — never diff on LTF series
`slope = kama_htf - kama_htf[1]` on a lower timeframe gives 0 intraday. Use two `request.security()` calls with offset.

### 4. Renko data loader — fractional timestamps
Use `pd.to_datetime(df["time"], unit="s")` NOT `.astype("int64")`. Fractional timestamps distinguish multiple bricks formed within the same second.

### 5. merge_asof HTF alignment
Must `.shift(1)` the HTF data BEFORE merging to avoid look-ahead bias.

### 6. TV CSV export time = bar CLOSE time
Pine's `time` = bar OPEN time. Off by one bar if not accounted for.

### 7. BOM-safe CSV parsing
TradingView CSV exports use BOM encoding. Always use `encoding='utf-8-sig'`.

### 8. OANDA daily bars
Open at **22:00 UTC**, not midnight. Use native `OANDA_EURUSD, 1D.csv`.

## TV Validation Workflow

1. Sanitize Pine (checklist above)
2. Load on TV Renko chart with matching brick size
3. Compare: expect <=5% trade count delta and <=10% net profit delta
4. **Non-USD instruments**: PF delta can exceed +/-10% (AUD/JPY conversion). Use trade count + native-currency net profit instead.

## Key Documentation

- `BACKTESTING.md` — engine internals: fill mechanics, order of operations, TV quirks, position sizing, TP/SL, pyramiding
- `MEMORY.md` (root) — legacy engine docs, strategy template, indicator API, reversal logic
- Agent memory at `~/.claude/projects/.../memory/MEMORY.md` — live portfolio, research phases, sweep results

## Current State

See agent memory (`MEMORY.md` in the memory directory) for:
- Live portfolio (6 strategies, 6 instruments)
- Research phase status and results
- BTC strategy findings
- Data availability
