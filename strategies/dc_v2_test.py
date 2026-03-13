"""
Donchian Mean Reversion v2 — Test improvements:
  1. 48-bar max hold time (exit if trade hasn't hit midline in 4 hours)
  2. Friday close (flatten before weekend)

Compare v1 (original) vs v2 (improved) side by side.
"""
import sys, io, contextlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from engine import load_tv_export, BacktestConfig, run_backtest_long_short, print_kpis, calc_donchian


def run_bt(df, sd, ed, tp=0.0, sl=0.0, qty=1000.0):
    cfg = BacktestConfig(
        initial_capital=1000.0, commission_pct=0.0043, slippage_ticks=0,
        qty_type="fixed", qty_value=qty, pyramiding=1,
        start_date=sd, end_date=ed,
        take_profit_pct=tp, stop_loss_pct=sl,
    )
    return run_backtest_long_short(df, cfg)


def dc_signals_v1(df, period=50, cd=60):
    """Original — midline exit only."""
    upper, lower, mid = calc_donchian(df["High"], df["Low"], period)
    upper = upper.values; lower = lower.values; mid = mid.values
    c = df["Close"].values; n = len(c)
    le = np.zeros(n, bool); lx = np.zeros(n, bool)
    se = np.zeros(n, bool); sx = np.zeros(n, bool)
    pos = 0; bst = 99999

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
    return df2


def dc_signals_v2(df, period=50, cd=60, max_hold=48):
    """Improved — midline exit + 48-bar max hold + Friday close."""
    upper, lower, mid = calc_donchian(df["High"], df["Low"], period)
    upper = upper.values; lower = lower.values; mid = mid.values
    c = df["Close"].values; n = len(c)
    hours = df.index.hour
    days = df.index.dayofweek  # 0=Mon ... 4=Fri

    le = np.zeros(n, bool); lx = np.zeros(n, bool)
    se = np.zeros(n, bool); sx = np.zeros(n, bool)
    pos = 0; bst = 99999; hold = 0

    for i in range(2, n):
        bst += 1
        if pos != 0: hold += 1
        if np.isnan(upper[i]) or np.isnan(upper[i-1]): continue

        ls = c[i-1] <= lower[i-1] and c[i] > lower[i]
        ss = c[i-1] >= upper[i-1] and c[i] < upper[i]
        cuv = c[i-1] <= mid[i-1] and c[i] > mid[i]
        cdv = c[i-1] >= mid[i-1] and c[i] < mid[i]

        # Friday close: exit all positions after 15:00 UTC on Friday
        friday_close = days[i] == 4 and hours[i] >= 15

        # Max hold time
        timed_out = hold >= max_hold

        # Exit conditions
        if pos == 1 and (cuv or ss or timed_out or friday_close):
            lx[i] = True; pos = 0; bst = 0; hold = 0
        elif pos == -1 and (cdv or ls or timed_out or friday_close):
            sx[i] = True; pos = 0; bst = 0; hold = 0

        # Entry conditions (no new entries after Friday 15:00)
        if pos == 0 and bst >= cd and not friday_close:
            if ls:   le[i] = True; pos = 1; bst = 0; hold = 0
            elif ss: se[i] = True; pos = -1; bst = 0; hold = 0

    df2 = df.copy()
    df2["long_entry"] = le; df2["long_exit"] = lx | se
    df2["short_entry"] = se; df2["short_exit"] = sx | le
    return df2


def main():
    df = load_tv_export("OANDA_EURUSD, 5.csv")
    sd = str(df.index[0].date())
    ed = "2069-12-31"
    oos = "2026-02-17"
    n_days = np.busday_count(df.index[0].date(), df.index[-1].date())

    print("=" * 70)
    print("  DONCHIAN MEAN REVERSION — v1 vs v2 COMPARISON")
    print("  v1: Original (midline exit only)")
    print("  v2: +48-bar max hold + Friday close")
    print("=" * 70)

    # --- v1 ---
    print("\n--- v1 (ORIGINAL) ---")
    df_v1 = dc_signals_v1(df)
    kpis_v1 = run_bt(df_v1, sd, ed)
    print_kpis(kpis_v1)

    # v1 OOS
    print("\n--- v1 OOS (from 2026-02-17) ---")
    kpis_v1_oos = run_bt(df_v1, oos, ed)
    print_kpis(kpis_v1_oos)

    # --- Test different max hold values ---
    print("\n" + "=" * 70)
    print("  MAX HOLD SWEEP (testing different values)")
    print("=" * 70)
    print(f"  {'MaxHold':>8} {'Trades':>6} {'WR%':>6} {'PF':>6} {'Net$':>8} {'DD%':>7} | {'OOS-Tr':>6} {'OOS-PF':>7} {'OOS-Net':>8} {'OOS-DD':>7}")
    print(f"  {'-'*85}")

    for mh in [24, 36, 48, 60, 72, 96, 120, 999]:
        df_test = dc_signals_v2(df, max_hold=mh)
        k = run_bt(df_test, sd, ed)
        ko = run_bt(df_test, oos, ed)
        trades = [t for t in k["trades"] if t.exit_date is not None]
        oos_trades = [t for t in ko["trades"] if t.exit_date is not None]
        pf = k.get("profit_factor", 0) or 0
        opf = ko.get("profit_factor", 0) or 0
        fpf = f"{pf:.2f}" if pf < 100 else "INF"
        fopf = f"{opf:.2f}" if opf < 100 else "INF"
        label = f"{mh}" if mh < 999 else "none"
        print(f"  {label:>8} {len(trades):>6} {k.get('win_rate',0):>5.1f}% {fpf:>6} {k.get('net_profit',0):>+8.2f} {k.get('max_drawdown_pct',0):>6.2f}% | {len(oos_trades):>6} {fopf:>7} {ko.get('net_profit',0):>+8.2f} {ko.get('max_drawdown_pct',0):>6.2f}%")

    # --- v2 (with best max_hold) ---
    print("\n\n--- v2 FULL (48-bar max hold + Friday close) ---")
    df_v2 = dc_signals_v2(df, max_hold=48)
    kpis_v2 = run_bt(df_v2, sd, ed)
    print_kpis(kpis_v2)

    print("\n--- v2 OOS (from 2026-02-17) ---")
    kpis_v2_oos = run_bt(df_v2, oos, ed)
    print_kpis(kpis_v2_oos)

    # --- Side by side ---
    trades_v1 = [t for t in kpis_v1["trades"] if t.exit_date is not None]
    trades_v2 = [t for t in kpis_v2["trades"] if t.exit_date is not None]
    oos_v1 = [t for t in kpis_v1_oos["trades"] if t.exit_date is not None]
    oos_v2 = [t for t in kpis_v2_oos["trades"] if t.exit_date is not None]

    print("\n" + "=" * 70)
    print("  SIDE-BY-SIDE SUMMARY")
    print("=" * 70)
    v1_net = f"${kpis_v1.get('net_profit',0):.2f}"
    v2_net = f"${kpis_v2.get('net_profit',0):.2f}"
    v1_avgw = f"${kpis_v1.get('avg_winning_trade',0):.2f}"
    v2_avgw = f"${kpis_v2.get('avg_winning_trade',0):.2f}"
    v1_avgl = f"${kpis_v1.get('avg_losing_trade',0):.2f}"
    v2_avgl = f"${kpis_v2.get('avg_losing_trade',0):.2f}"
    v1_lloss = f"${kpis_v1.get('largest_loss',0):.2f}"
    v2_lloss = f"${kpis_v2.get('largest_loss',0):.2f}"
    v1o_net = f"${kpis_v1_oos.get('net_profit',0):.2f}"
    v2o_net = f"${kpis_v2_oos.get('net_profit',0):.2f}"
    pf1 = kpis_v1.get("profit_factor", 0) or 0
    pf2 = kpis_v2.get("profit_factor", 0) or 0
    opf1 = kpis_v1_oos.get("profit_factor", 0) or 0
    opf2 = kpis_v2_oos.get("profit_factor", 0) or 0

    print(f"  {'Metric':<25} {'v1 (original)':>15} {'v2 (improved)':>15}")
    print(f"  {'-'*55}")
    print(f"  {'IS Trades':<25} {len(trades_v1):>15} {len(trades_v2):>15}")
    print(f"  {'IS Win Rate':<25} {kpis_v1.get('win_rate',0):>14.1f}% {kpis_v2.get('win_rate',0):>14.1f}%")
    print(f"  {'IS Profit Factor':<25} {pf1:>15.3f} {pf2:>15.3f}")
    print(f"  {'IS Net PnL':<25} {v1_net:>15} {v2_net:>15}")
    print(f"  {'IS Max DD':<25} {kpis_v1.get('max_drawdown_pct',0):>14.2f}% {kpis_v2.get('max_drawdown_pct',0):>14.2f}%")
    print(f"  {'IS Avg Win':<25} {v1_avgw:>15} {v2_avgw:>15}")
    print(f"  {'IS Avg Loss':<25} {v1_avgl:>15} {v2_avgl:>15}")
    print(f"  {'IS Largest Loss':<25} {v1_lloss:>15} {v2_lloss:>15}")
    print()
    print(f"  {'OOS Trades':<25} {len(oos_v1):>15} {len(oos_v2):>15}")
    print(f"  {'OOS Win Rate':<25} {kpis_v1_oos.get('win_rate',0):>14.1f}% {kpis_v2_oos.get('win_rate',0):>14.1f}%")
    print(f"  {'OOS Profit Factor':<25} {opf1:>15.3f} {opf2:>15.3f}")
    print(f"  {'OOS Net PnL':<25} {v1o_net:>15} {v2o_net:>15}")
    print(f"  {'OOS Max DD':<25} {kpis_v1_oos.get('max_drawdown_pct',0):>14.2f}% {kpis_v2_oos.get('max_drawdown_pct',0):>14.2f}%")


if __name__ == "__main__":
    main()
