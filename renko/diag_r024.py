"""
Diagnostic: compare "fresh" vs "resume" entry quality in r024_kama_ribbon_pullback.

Fresh  = first brick of new alignment (bull_align[i] True, bull_align[i-1] False)
Resume = re-entry within existing alignment (bull_align[i-1] already True)
"""

import contextlib
import io
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent / "strategies"))

from engine import BacktestConfig, run_backtest_long_short
from renko.data import load_renko_export
from renko.indicators import add_renko_indicators
from renko.strategies.r024_kama_ribbon_pullback import generate_signals, RIBBONS, _KAMA_CACHE

import numpy as np
import pandas as pd
from indicators.kama import calc_kama

# ── Config ──────────────────────────────────────────────────────────────────
import os
RENKO_FILE = os.environ.get("RENKO_FILE", "OANDA_EURUSD, 1S renko 0.0004.csv")
IS_START = "2023-01-23"
IS_END   = "2025-09-30"

PARAMS = dict(
    entry_mode="resume",
    ribbon="3L_5_13_34",
    cooldown=3,
    adx_gate=0,
    exit_on_brick=True,
)

# ── Load data & generate signals ────────────────────────────────────────────
print("Loading renko data...")
df = load_renko_export(RENKO_FILE)
add_renko_indicators(df)

# Clear KAMA cache (shared mutable in strategy module)
_KAMA_CACHE.clear()

print("Generating signals...")
df_sig = generate_signals(df.copy(), **PARAMS)

# ── Recompute bull_align / bear_align for tagging ───────────────────────────
lengths = RIBBONS[PARAMS["ribbon"]]
close = df_sig["Close"]

kama_fast = calc_kama(close, length=lengths[0]).shift(1).values
kama_mid  = calc_kama(close, length=lengths[1]).shift(1).values
kama_slow = calc_kama(close, length=lengths[2]).shift(1).values

any_nan = np.isnan(kama_fast) | np.isnan(kama_mid) | np.isnan(kama_slow)
bull_align = (kama_fast > kama_mid) & (kama_mid > kama_slow) & ~any_nan
bear_align = (kama_fast < kama_mid) & (kama_mid < kama_slow) & ~any_nan

df_sig["bull_align"] = bull_align
df_sig["bear_align"] = bear_align

# ── Run backtest to get individual trades ───────────────────────────────────
print("Running backtest...")
cfg = BacktestConfig(
    initial_capital=1000.0,
    commission_pct=0.0046,
    slippage_ticks=0,
    qty_type="fixed",
    qty_value=1000.0,
    pyramiding=1,
    start_date=IS_START,
    end_date=IS_END,
    take_profit_pct=0.0,
    stop_loss_pct=0.0,
)

with contextlib.redirect_stdout(io.StringIO()):
    kpis = run_backtest_long_short(df_sig, cfg)

trades = kpis.get("trades", [])
print(f"Total trades from engine: {len(trades)}")

# ── Tag each trade as "fresh" or "resume" ───────────────────────────────────
# Build a date-to-positional-index lookup using the DatetimeIndex
date_to_idx = {d: i for i, d in enumerate(df_sig.index)}

fresh_trades = []
resume_trades = []

for t in trades:
    if t.exit_price is None:
        continue  # skip open trades

    entry_dt = t.entry_date
    # Find the bar index matching entry_date
    idx = date_to_idx.get(entry_dt)
    if idx is None:
        # Try matching via pandas Timestamp normalization
        entry_ts = pd.Timestamp(entry_dt)
        for col_dt, col_idx in date_to_idx.items():
            if pd.Timestamp(col_dt) == entry_ts:
                idx = col_idx
                break
    if idx is None or idx < 2:
        resume_trades.append(t)
        continue

    # Trade fills at bar idx (next bar open after signal).
    # Signal bar = idx - 1.  For "fresh", alignment at signal_bar - 1 was False.
    signal_bar = idx - 1
    if t.direction == "long":
        was_aligned_before = bool(bull_align[signal_bar - 1])
    else:
        was_aligned_before = bool(bear_align[signal_bar - 1])

    if was_aligned_before:
        resume_trades.append(t)  # alignment already existed → re-entry
    else:
        fresh_trades.append(t)   # alignment just formed → fresh

# ── Stats helper ────────────────────────────────────────────────────────────
def calc_stats(label, trade_list):
    if not trade_list:
        print(f"\n{'='*50}")
        print(f"  {label}: 0 trades")
        print(f"{'='*50}")
        return

    pnls = [t.pnl for t in trade_list]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    count = len(pnls)
    win_rate = len(wins) / count * 100 if count else 0
    avg_pnl = sum(pnls) / count if count else 0
    total_win = sum(wins) if wins else 0
    total_loss = abs(sum(losses)) if losses else 0
    pf = total_win / total_loss if total_loss > 0 else float("inf")
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    net = sum(pnls)

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Count     : {count}")
    print(f"  Win rate  : {win_rate:.1f}%")
    print(f"  Net P&L   : {net:.2f}")
    print(f"  Avg P&L   : {avg_pnl:.4f}")
    print(f"  Avg Win   : {avg_win:.4f}")
    print(f"  Avg Loss  : {avg_loss:.4f}")
    pf_str = f"{pf:.4f}" if not math.isinf(pf) else "INF"
    print(f"  PF        : {pf_str}")
    print(f"  Total Win : {total_win:.2f}")
    print(f"  Total Loss: {total_loss:.2f}")

# ── Print results ───────────────────────────────────────────────────────────
print(f"\nParams: {PARAMS}")
print(f"Period: {IS_START} to {IS_END}")

calc_stats("FRESH entries (new alignment)", fresh_trades)
calc_stats("RESUME entries (re-entry within alignment)", resume_trades)
calc_stats("ALL trades", fresh_trades + resume_trades)
