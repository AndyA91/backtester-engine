"""
Example: EMA Crossover strategy backtest (Long-only + Long & Short).

Demonstrates how to use the backtesting engine with an EMA 9/21 crossover
strategy on INDEX:BTCUSD 1D data. Runs both run_backtest (long-only) and
run_backtest_long_short (long + short) to verify both engine paths.

Chart data: INDEX:BTCUSD 1D (TradingView export)
Slippage: NOT simulated (set to 0) — requires tick-level data which is
          expensive to obtain. Set slippage to 0 in both engine and TV.

Settings (match these in TradingView for comparison):
- 2018-01-01 to 2069-12-31
- Initial capital: $1,000
- 100% of equity per trade
- 0.1% commission
- 0 slippage
- Fast EMA: 9, Slow EMA: 21
- Margin long/short: 0%

Expected results (long-only):
- Net Profit: $9,180 (917.95%)
- Total Trades: 60
- Win Rate: 33.33%
- Profit Factor: 1.813
- Max Drawdown (intrabar): -$3,420 (-61.01%)

Expected results (long + short, EMA 9/21 long + EMA 5/13 short, with reversals):
- Net Profit: $1,501 (150.11%)
- Total Trades: 155
- Profit Factor: 1.167
- Open P&L: ~$513 (unrealised, not included in Net Profit)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import load_tv_export
from engine import (
    BacktestConfig, calc_ema, detect_crossover, detect_crossunder,
    ema_cross_signals, run_backtest, run_backtest_long_short,
    print_kpis, print_trades,
)


def main():
    # Load TradingView-exported INDEX:BTCUSD 1D data
    # The file goes back to 2014, giving plenty of warmup before 2018-01-01.
    # The last bar is automatically dropped (unfinished candle).
    df = load_tv_export("INDEX_BTCUSD, 1D.csv")

    print(f"\nData range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Total bars: {len(df)}")

    warmup_bars = len(df[df.index < "2018-01-01"])
    print(f"Warmup bars (before 2018-01-01): {warmup_bars}")

    # Configure backtest to match TradingView settings
    config = BacktestConfig(
        initial_capital=1000.0,
        commission_pct=0.1,
        slippage_ticks=0,
        qty_type="percent_of_equity",
        qty_value=100.0,
        pyramiding=1,
        start_date="2018-01-01",
        end_date="2069-12-31",
    )

    # ---- 1. Long-only backtest ------------------------------------------------
    # Use original EMA 9/21 signals (no reversal modifications)
    df_long = df.copy()
    df_long = ema_cross_signals(df_long, fast_len=9, slow_len=21)
    kpis = run_backtest(df_long, config)

    print("\n" + "=" * 60)
    print("  BACKTEST CONFIGURATION")
    print("=" * 60)
    print(f"  Chart Data:       INDEX:BTCUSD 1D (TradingView export)")
    print(f"  Date Range:       {kpis['actual_start_date']} to {kpis['actual_end_date']}")
    print(f"  Initial Capital:  ${config.initial_capital:,.0f}")
    print(f"  Order Size:       {config.qty_value:.0f}% of equity")
    print(f"  Commission:       {config.commission_pct}%")
    print(f"  Slippage:         {config.slippage_ticks} (NOT simulated — requires tick-level data)")
    print(f"  Margin Long/Short: 0%")
    print(f"  Strategy:         EMA Crossover (Fast: 9, Slow: 21)")
    print("=" * 60)

    print("\n--- LONG-ONLY (run_backtest) ---")
    print_kpis(kpis)
    print_trades(kpis["trades"], max_trades=10)

    # Print first few trades in detail for verification
    print("\n\nDETAILED FIRST 5 TRADES (for TradingView comparison):")
    print("=" * 80)
    for i, t in enumerate(kpis["trades"][:5], 1):
        print(f"\nTrade #{i}:")
        print(f"  Entry: {t.entry_date.date()} @ ${t.entry_price:,.2f}")
        print(f"  Qty:   {t.entry_qty:.8f} BTC")
        print(f"  Entry Commission: ${t.entry_commission:.4f}")
        print(f"  Exit:  {t.exit_date.date() if t.exit_date else 'OPEN'} @ ${t.exit_price:,.2f}" if t.exit_price else "  Exit: OPEN")
        print(f"  Exit Commission:  ${t.exit_commission:.4f}")
        print(f"  PnL:   ${t.pnl:,.2f} ({t.pnl_pct:.2f}%)" if t.pnl else "  PnL: N/A")

    # ---- 2. Long + Short backtest ---------------------------------------------
    # Build signals with reversal logic (separate copy to avoid affecting long-only)
    df_ls = df.copy()
    df_ls = ema_cross_signals(df_ls, fast_len=9, slow_len=21)

    # Add short signals using a separate EMA pair (5/13)
    df_ls["short_fast"] = calc_ema(df_ls["Close"], 5)
    df_ls["short_slow"] = calc_ema(df_ls["Close"], 13)
    short_cross_under = detect_crossunder(df_ls["short_fast"], df_ls["short_slow"])
    short_cross_over = detect_crossover(df_ls["short_fast"], df_ls["short_slow"])

    # In TradingView, strategy.entry("Short") reverses the position:
    # it closes the long AND opens a short on the same bar. To match,
    # a short_entry must also act as a long_exit, and vice versa.
    df_ls["short_entry"] = short_cross_under
    df_ls["short_exit"] = short_cross_over | df_ls["long_entry"]  # long entry also closes short
    df_ls["long_exit"] = df_ls["long_exit"] | short_cross_under   # short entry also closes long

    kpis_ls = run_backtest_long_short(df_ls, config)

    print("\n\n--- LONG + SHORT (run_backtest_long_short) ---")
    print_kpis(kpis_ls)
    print_trades(kpis_ls["trades"], max_trades=10)

    # TV comparison from XLSX (L+S run)
    import pandas as pd
    xlsx_path = str(Path(__file__).resolve().parent / "example_ema_cross.xlsx")
    compare_to_tv(kpis_ls, xlsx_path)


def compare_to_tv(kpis, xlsx_path):
    import pandas as pd
    tv_overview = pd.read_excel(xlsx_path)
    def tv_val(name):
        row = tv_overview[tv_overview.iloc[:, 0] == name]
        return row.iloc[0, 1] if not row.empty else None

    tv_net_pct = tv_overview[tv_overview.iloc[:, 0] == "Net profit"].iloc[0, 2]
    tv_comm = tv_val("Commission paid")
    tv_max_dd_pct = tv_overview[tv_overview.iloc[:, 0] == "Max equity drawdown (intrabar)"].iloc[0, 2]
    tv_trades_analysis = pd.read_excel(xlsx_path, sheet_name="Trades analysis")
    tv_risk_adj = pd.read_excel(xlsx_path, sheet_name="Risk-adjusted performance")
    _ta_row = tv_trades_analysis[tv_trades_analysis.iloc[:, 0] == "Percent profitable"]
    tv_wr = _ta_row.iloc[0, 2] if not _ta_row.empty else 0.0
    _pf_row = tv_risk_adj[tv_risk_adj.iloc[:, 0] == "Profit factor"]
    tv_pf = _pf_row.iloc[0, 1] if not _pf_row.empty else 0.0
    tv_gp = tv_val("Gross profit")
    tv_gl = tv_val("Gross loss")

    tv_trades_df = pd.read_excel(xlsx_path, sheet_name="List of trades")
    # Filter out open trades (Signal="Open" on exit rows)
    exit_rows = tv_trades_df[tv_trades_df["Type"].str.contains("Exit")]
    open_trades = exit_rows[exit_rows["Signal"] == "Open"]
    open_trade_nums = set(open_trades["Trade #"].values)
    tv_entry_long = tv_trades_df[(tv_trades_df["Type"] == "Entry long") & ~tv_trades_df["Trade #"].isin(open_trade_nums)]
    tv_entry_short = tv_trades_df[(tv_trades_df["Type"] == "Entry short") & ~tv_trades_df["Trade #"].isin(open_trade_nums)]
    n_tv_trades = len(tv_entry_long) + len(tv_entry_short)

    closed = [t for t in kpis["trades"] if t.exit_date is not None]

    net_pct_diff = abs(kpis["net_profit_pct"] - tv_net_pct) / abs(tv_net_pct) * 100 if tv_net_pct != 0 else 0
    dd_pct_diff = abs(abs(kpis["max_drawdown_pct"]) - tv_max_dd_pct) / tv_max_dd_pct * 100 if tv_max_dd_pct != 0 else 0
    comm_diff = abs(kpis["total_commission"] - tv_comm)

    print(f"\n  --- TV Comparison (from XLSX, L+S) ---")
    print(f"  {'Metric':<25s} {'Engine':>14s} {'TV':>14s}")
    print(f"  {'-'*55}")
    print(f"  {'Trades':<25s} {len(closed):>14d} {n_tv_trades:>14d}")
    print(f"  {'Net Profit %':<25s} {kpis['net_profit_pct']:>13.2f}% {tv_net_pct:>13.2f}%")
    print(f"  {'Max Drawdown %':<25s} {abs(kpis['max_drawdown_pct']):>13.2f}% {tv_max_dd_pct:>13.2f}%")
    print(f"  {'Commission $':<25s} ${kpis['total_commission']:>13,.2f} ${tv_comm:>13,.2f}")
    print(f"  {'Win Rate %':<25s} {kpis['win_rate']:>13.2f}% {tv_wr:>13.2f}%")
    print(f"  {'Profit Factor':<25s} {kpis['profit_factor']:>14.3f} {tv_pf:>14.3f}")
    print(f"  {'Gross Profit $':<25s} ${kpis['gross_profit']:>13,.2f} ${tv_gp:>13,.2f}")
    print(f"  {'Gross Loss $':<25s} ${abs(kpis['gross_loss']):>13,.2f} ${tv_gl:>13,.2f}")
    print(f"  Net diff: {net_pct_diff:.3f}%  DD diff: {dd_pct_diff:.3f}%  Comm diff: ${comm_diff:.2f}")

    wr_ok = abs(kpis["win_rate"] - tv_wr) < 0.01  # exact match expected
    pf_diff = abs(kpis["profit_factor"] - tv_pf) / tv_pf * 100 if tv_pf != 0 else 0
    gp_diff = abs(kpis["gross_profit"] - tv_gp) / abs(tv_gp) * 100 if tv_gp != 0 else 0
    gl_diff = abs(abs(kpis["gross_loss"]) - tv_gl) / tv_gl * 100 if tv_gl != 0 else 0
    print(f"  WR ok: {wr_ok}  PF diff: {pf_diff:.3f}%  GP diff: {gp_diff:.3f}%  GL diff: {gl_diff:.3f}%")

    # Trade-by-trade comparison
    tv_all_entries = tv_trades_df[tv_trades_df["Type"].str.startswith("Entry")].sort_values("Trade #").reset_index(drop=True)
    tv_all_exits = tv_trades_df[tv_trades_df["Type"].str.startswith("Exit")].sort_values("Trade #").reset_index(drop=True)
    open_nums = set(tv_all_exits[tv_all_exits["Signal"] == "Open"]["Trade #"].values)
    tv_all_entries = tv_all_entries[~tv_all_entries["Trade #"].isin(open_nums)].reset_index(drop=True)
    tv_closed_exits = tv_all_exits[~tv_all_exits["Trade #"].isin(open_nums)].reset_index(drop=True)

    mismatches = 0
    n = min(len(closed), len(tv_all_entries))
    for i in range(n):
        et = closed[i]
        tv_e = tv_all_entries.iloc[i]
        tv_date = pd.Timestamp(tv_e["Date and time"])
        tv_price = tv_e["Price USD"]
        tv_pnl = tv_e["Net P&L USD"]

        date_ok = et.entry_date == tv_date
        price_ok = abs(et.entry_price - tv_price) < 0.50
        pnl_ok = abs(et.pnl - tv_pnl) < max(1.0, abs(tv_pnl) * 0.005)

        if i < len(tv_closed_exits):
            tv_x = tv_closed_exits.iloc[i]
            tv_exit_date = pd.Timestamp(tv_x["Date and time"])
            tv_exit_price = tv_x["Price USD"]
            exit_date_ok = et.exit_date == tv_exit_date
            exit_price_ok = abs(et.exit_price - tv_exit_price) < 0.50
        else:
            exit_date_ok = True
            exit_price_ok = True

        if not (date_ok and price_ok and exit_date_ok and exit_price_ok and pnl_ok):
            mismatches += 1
            if mismatches <= 5:
                print(f"    MISMATCH trade {i+1}:")
                if not date_ok: print(f"      Date:  engine={et.entry_date}  TV={tv_date}")
                if not price_ok: print(f"      Price: engine={et.entry_price:.2f}  TV={tv_price:.2f}")
                if not exit_date_ok: print(f"      ExitDate: engine={et.exit_date}  TV={tv_exit_date}")
                if not exit_price_ok: print(f"      ExitPrice: engine={et.exit_price:.2f}  TV={tv_exit_price:.2f}")
                if not pnl_ok: print(f"      PnL:   engine={et.pnl:.2f}  TV={tv_pnl:.2f}")

    if len(closed) != len(tv_all_entries):
        mismatches += abs(len(closed) - len(tv_all_entries))
    print(f"  Trade-by-trade: {n - min(mismatches, n)}/{n} match ({mismatches} mismatches)")

    ok = (len(closed) == n_tv_trades and net_pct_diff < 0.2 and dd_pct_diff < 0.2
          and comm_diff < 5.0
          and pf_diff < 0.5 and wr_ok and gp_diff < 0.5 and gl_diff < 0.5
          and mismatches == 0)
    print(f"  {'✅' if ok else '❌'} {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    main()
