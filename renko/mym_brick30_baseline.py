"""
MYM brick 30 baseline check — single config (streak=12, cd=10).

Tests the existing live config on the new brick 30 ETH data in two windows:
  1. Validation window (2025-01-06 -> 2026-03-19) for apples-to-apples vs the
     existing brick 15 baseline (PF=44.81 in memory)
  2. Full period (2022-10-16 -> 2026-04-07) which includes today's incident

NOT a sweep. One config, two windows. Sub-minute runtime.

Slices the DataFrame BEFORE generating signals so the stateful `pos` variable
in the signal generator does not get contaminated by phantom pre-window trades
(see look_ahead_redflags.md L2).
"""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import BacktestConfig, run_backtest_long_short
from renko.data import load_renko_export

RENKO_FILE = "CBOT_MINI_MYM1!, 1S ETH renko 30.csv"
STREAK = 12
COOLDOWN = 10

WINDOWS = [
    ("Validation window — apples-to-apples vs brick 15 baseline",
     "2025-01-06", "2026-03-20"),
    ("Full period — INCLUDES April 7 incident",
     None, None),
]


def generate_signals(df, min_brick_streak, cooldown):
    """Same logic as mym_renko_streak_sweep.py — pure brick streak entry,
    brick flip exit, cooldown bars between trades."""
    brick_up = df["brick_up"].values
    n = len(df)
    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    warmup = min_brick_streak + 2
    last_exit_bar = -999_999
    pos = 0

    for i in range(warmup, n):
        if pos == 1 and not brick_up[i]:
            long_exit[i] = True
            pos = 0
            last_exit_bar = i
        elif pos == -1 and brick_up[i]:
            short_exit[i] = True
            pos = 0
            last_exit_bar = i

        if pos == 0 and (i - last_exit_bar) >= cooldown:
            last_n = brick_up[i - min_brick_streak:i]
            prev_n = brick_up[i - min_brick_streak - 1:i - 1]
            brk_long_ok  = bool(np.all(last_n)) and not bool(np.all(prev_n))
            brk_short_ok = bool(not np.any(last_n)) and not bool(not np.any(prev_n))
            if brk_long_ok:
                long_entry[i] = True
                pos = 1
            elif brk_short_ok:
                short_entry[i] = True
                pos = -1

    df = df.copy()
    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df


def fmt_price(p):
    return f"{p:.0f}" if p is not None else "open"


def fmt_pnl(p):
    return f"${p:>7.2f}" if p is not None else "  open  "


def run_one(df_full, label, start, end):
    print(f"\n{'=' * 78}")
    print(f"  {label}")
    print(f"{'=' * 78}")
    if start is not None:
        df = df_full.loc[start:end].copy()
    else:
        df = df_full.copy()
    print(f"  Bricks in window: {len(df):,}")
    print(f"  Date range:       {df.index[0]} -> {df.index[-1]}")

    df = generate_signals(df, STREAK, COOLDOWN)
    cfg = BacktestConfig(
        initial_capital=10000.0,
        commission_pct=0.002,
        qty_type="fixed",
        qty_value=1,
        start_date="2000-01-01",
        end_date="2099-12-31",
    )
    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest_long_short(df, cfg)

    print()
    print(f"  Total trades:   {kpis.get('total_trades', 0)}")
    print(f"  Win rate:       {kpis.get('win_rate', 0):.1f}%")
    print(f"  Profit factor:  {kpis.get('profit_factor', 0):.2f}")
    print(f"  Net profit:     ${kpis.get('net_profit', 0):,.2f}")
    print(f"  Max drawdown:   ${kpis.get('max_drawdown', 0):,.2f}")
    print(f"  Avg trade:      ${kpis.get('avg_trade', 0):,.2f}")
    print(f"  Avg W/L ratio:  {kpis.get('avg_win_loss_ratio', 0):.2f}")

    trades = kpis.get("trades", [])
    n_long  = sum(1 for t in trades if t.direction == "long"  and t.exit_date is not None)
    n_short = sum(1 for t in trades if t.direction == "short" and t.exit_date is not None)
    print(f"  Long / Short:   {n_long} / {n_short}")

    # Largest losses (top 5) — diagnose worst-case behavior
    closed = [t for t in trades if t.exit_date is not None and t.pnl is not None]
    losers = sorted(closed, key=lambda t: t.pnl)[:5]
    if losers:
        print(f"\n  Top 5 worst losses in this window:")
        print(f"    {'entry_date':<20} {'side':<6} {'entry':>7} {'exit':>7} {'pnl':>10}")
        for t in losers:
            print(f"    {str(t.entry_date):<20} {t.direction:<6} "
                  f"{t.entry_price:>7.0f} {fmt_price(t.exit_price):>7} "
                  f"{fmt_pnl(t.pnl):>10}")

    # April 7 specifically — the incident day
    apr7_start = pd.Timestamp("2026-04-07 00:00:00")
    apr7_end   = pd.Timestamp("2026-04-08 00:00:00")
    apr7 = [t for t in trades
            if t.entry_date is not None
            and apr7_start <= pd.Timestamp(t.entry_date) < apr7_end]
    if apr7:
        apr7_pnl = sum(t.pnl for t in apr7 if t.pnl is not None)
        print(f"\n  April 7 2026 trades (the incident day): {len(apr7)}")
        print(f"  April 7 net P&L: ${apr7_pnl:,.2f}")
        print(f"    {'entry_time':<20} {'side':<6} {'entry':>7} {'exit':>7} {'pnl':>10}")
        for t in apr7:
            print(f"    {str(t.entry_date):<20} {t.direction:<6} "
                  f"{t.entry_price:>7.0f} {fmt_price(t.exit_price):>7} "
                  f"{fmt_pnl(t.pnl):>10}")


def main():
    print(f"Loading {RENKO_FILE}...")
    df = load_renko_export(RENKO_FILE)
    print(f"  {len(df):,} bricks loaded, {df.index[0]} -> {df.index[-1]}")

    for label, start, end in WINDOWS:
        run_one(df, label, start, end)


if __name__ == "__main__":
    main()
