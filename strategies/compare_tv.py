"""Compare TV exported results to engine backtest results."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

# --- Parse TV CSV ---
tv = pd.read_csv("tvresults/Gaussian_Channel_Reversal_OANDA_EURUSD_2026-03-03.csv")
entries = tv[tv["Type"].str.startswith("Entry")]
exits = tv[tv["Type"].str.startswith("Exit")]
n_trades_tv = len(entries)

# TV summary
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

# --- Run engine ---
from engine import load_tv_export, BacktestConfig, run_backtest_long_short, print_kpis
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gaussian_channel_eurusd_1 import gaussian_channel_signals

df = load_tv_export("OANDA_EURUSD, 1.csv")
start_date = str(df.index[0].date())
end_date = "2069-12-31"
config = BacktestConfig(
    initial_capital=1000.0, commission_pct=0.0085, slippage_ticks=0,
    qty_type="fixed", qty_value=1000.0, pyramiding=1,
    start_date=start_date, end_date=end_date,
    take_profit_pct=0.20, stop_loss_pct=0.25
)
df_sig = gaussian_channel_signals(df, period=500, poles=4, mult=5.0,
    cooldown_bars=90, start_date=start_date, end_date=end_date)
df_sig["long_exit"] = df_sig["long_exit"] | df_sig["short_entry"]
df_sig["short_exit"] = df_sig["short_exit"] | df_sig["long_entry"]
kpis = run_backtest_long_short(df_sig, config)

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
    # TV trade
    trade_num = i + 1
    tv_entry = entries.iloc[i]
    tv_exit = exits.iloc[i]
    tv_dir = tv_entry["Signal"]
    tv_entry_price = tv_entry["Price USD"]
    tv_pnl = tv_exit["Net P&L USD"]

    # Engine trade
    et = engine_trades[i]
    eng_dir = et.direction.upper()
    eng_entry_price = et.entry_price
    eng_pnl = et.pnl

    # Check match
    price_match = abs(tv_entry_price - eng_entry_price) < 0.001
    pnl_match = abs(tv_pnl - eng_pnl) < 0.5
    match = "OK" if price_match and pnl_match else "DIFF"
    if match == "OK":
        match_count += 1

    print(f"{trade_num:>3} {tv_dir:>6}  {tv_entry_price:>10.5f} {eng_entry_price:>10.5f} {tv_pnl:>8.2f} {eng_pnl:>8.2f} {match:>5}")

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
