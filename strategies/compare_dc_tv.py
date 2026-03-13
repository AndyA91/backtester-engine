"""Compare TV Donchian Mean Reversion results vs Python engine."""
import sys, io, contextlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from engine import load_tv_export, BacktestConfig, run_backtest_long_short, print_kpis, calc_donchian

# --- Parse TV CSV ---
tv = pd.read_csv("tvresults/Donchian_Mean_Reversion_v1_OANDA_EURUSD_2026-03-03.csv")
entries = tv[tv["Type"].str.startswith("Entry")]
exits = tv[tv["Type"].str.startswith("Exit")]
n_trades_tv = len(entries)

wins_tv = exits[exits["Net P&L USD"] > 0]
losses_tv = exits[exits["Net P&L USD"] <= 0]
gross_profit_tv = wins_tv["Net P&L USD"].sum()
gross_loss_tv = losses_tv["Net P&L USD"].sum()
pf_tv = gross_profit_tv / abs(gross_loss_tv) if gross_loss_tv != 0 else float("inf")
cum_pnl_tv = exits.iloc[-1]["Cumulative P&L USD"]

print("=" * 60)
print("  TRADINGVIEW RESULTS")
print("=" * 60)
print(f"  Total Trades:     {n_trades_tv}")
print(f"  Winners:          {len(wins_tv)} ({len(wins_tv)/n_trades_tv*100:.1f}%)")
print(f"  Losers:           {len(losses_tv)} ({len(losses_tv)/n_trades_tv*100:.1f}%)")
print(f"  Profit Factor:    {pf_tv:.3f}")
print(f"  Cumulative PnL:   ${cum_pnl_tv}")
print(f"  Gross Profit:     ${gross_profit_tv:.2f}")
print(f"  Gross Loss:       ${gross_loss_tv:.2f}")

# --- Run engine with Donchian Mean Reversion signals ---
import numpy as np

df = load_tv_export("OANDA_EURUSD, 5.csv")
sd = str(df.index[0].date())
ed = "2069-12-31"

# Compute Donchian Channel
upper, lower, mid = calc_donchian(df["High"], df["Low"], 50)
upper = upper.values; lower = lower.values; mid = mid.values
c = df["Close"].values; n = len(c)

# Generate signals (same logic as Pine)
le = np.zeros(n, bool); lx = np.zeros(n, bool)
se = np.zeros(n, bool); sx = np.zeros(n, bool)
pos = 0; bst = 99999; cd = 60

for i in range(2, n):
    bst += 1
    if np.isnan(upper[i]) or np.isnan(upper[i-1]): continue
    ls = c[i-1] <= lower[i-1] and c[i] > lower[i]
    ss = c[i-1] >= upper[i-1] and c[i] < upper[i]
    cuv = c[i-1] <= mid[i-1] and c[i] > mid[i]
    cdv = c[i-1] >= mid[i-1] and c[i] < mid[i]
    if pos == 1 and (cuv or ss):   lx[i] = True; pos = 0; bst = 0
    elif pos == -1 and (cdv or ls): sx[i] = True; pos = 0; bst = 0
    if pos == 0 and bst >= cd:
        if ls:   le[i] = True; pos = 1; bst = 0
        elif ss: se[i] = True; pos = -1; bst = 0

df2 = df.copy()
df2["long_entry"] = le; df2["long_exit"] = lx | se
df2["short_entry"] = se; df2["short_exit"] = sx | le

config = BacktestConfig(
    initial_capital=1000.0, commission_pct=0.0043, slippage_ticks=0,
    qty_type="fixed", qty_value=1000.0, pyramiding=1,
    start_date=sd, end_date=ed,
)
kpis = run_backtest_long_short(df2, config)

print("\n" + "=" * 60)
print("  ENGINE RESULTS")
print("=" * 60)
print_kpis(kpis)

# --- Trade-by-trade comparison ---
print("\n" + "=" * 60)
print("  TRADE-BY-TRADE COMPARISON (first 15)")
print("=" * 60)

engine_trades = [t for t in kpis["trades"] if t.exit_date is not None]

header = f"{'#':>3} {'Dir':>6}  {'TV Entry$':>10} {'Eng Entry$':>10} {'TV PnL$':>8} {'Eng PnL$':>8} {'Match':>5}"
print(header)
print("-" * len(header))

match_count = 0
for i in range(min(15, n_trades_tv, len(engine_trades))):
    tv_entry = entries.iloc[i]
    tv_exit = exits.iloc[i]
    tv_dir = tv_entry["Signal"]
    tv_entry_price = tv_entry["Price USD"]
    tv_pnl = tv_exit["Net P&L USD"]

    et = engine_trades[i]
    eng_dir = et.direction.upper()
    eng_entry_price = et.entry_price
    eng_pnl = et.pnl

    price_match = abs(tv_entry_price - eng_entry_price) < 0.001
    pnl_match = abs(tv_pnl - eng_pnl) < 0.5
    match = "OK" if price_match and pnl_match else "DIFF"
    if match == "OK":
        match_count += 1

    print(f"{i+1:>3} {tv_dir:>6}  {tv_entry_price:>10.5f} {eng_entry_price:>10.5f} {tv_pnl:>8.2f} {eng_pnl:>8.2f} {match:>5}")

total_compared = min(15, n_trades_tv, len(engine_trades))
print(f"\nMatched: {match_count}/{total_compared} trades")

# --- Summary comparison ---
print("\n" + "=" * 60)
print("  SUMMARY COMPARISON")
print("=" * 60)
print(f"{'Metric':<25} {'TradingView':>12} {'Engine':>12} {'Diff':>8}")
print("-" * 60)
print(f"{'Total Trades':<25} {n_trades_tv:>12} {len(engine_trades):>12} {len(engine_trades)-n_trades_tv:>8}")
print(f"{'Win Rate %':<25} {len(wins_tv)/n_trades_tv*100:>11.1f}% {kpis['win_rate']:>11.1f}%")
print(f"{'Profit Factor':<25} {pf_tv:>12.3f} {kpis['profit_factor']:>12.3f} {kpis['profit_factor']-pf_tv:>8.3f}")
print(f"{'Net PnL $':<25} {cum_pnl_tv:>12} {kpis['net_profit']:>12.2f}")
print(f"{'Gross Profit $':<25} {gross_profit_tv:>12.2f} {kpis['gross_profit']:>12.2f}")
print(f"{'Gross Loss $':<25} {gross_loss_tv:>12.2f} {kpis['gross_loss']:>12.2f}")
