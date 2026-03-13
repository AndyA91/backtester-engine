"""
Validate Donchian Mean Reversion v2 — Python engine vs TradingView CSV.
Parses the TV trade-list CSV and compares entry-by-entry against the engine.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from engine import load_tv_export, BacktestConfig, run_backtest_long_short, calc_donchian


# ── Signal generator (same as dc_v2_test.py) ─────────────────────────────────
def dc_signals_v2(df, period=50, cd=60, max_hold=48):
    upper, lower, mid = calc_donchian(df["High"], df["Low"], period)
    upper = upper.values; lower = lower.values; mid = mid.values
    c = df["Close"].values; n = len(c)
    hours = df.index.hour
    days  = df.index.dayofweek  # 0=Mon … 4=Fri

    le = np.zeros(n, bool); lx = np.zeros(n, bool)
    se = np.zeros(n, bool); sx = np.zeros(n, bool)
    pos = 0; bst = 99999; hold = 0

    for i in range(2, n):
        bst += 1
        if pos != 0: hold += 1
        if np.isnan(upper[i]) or np.isnan(upper[i-1]): continue

        ls  = c[i-1] <= lower[i-1] and c[i] > lower[i]
        ss  = c[i-1] >= upper[i-1] and c[i] < upper[i]
        cuv = c[i-1] <= mid[i-1]   and c[i] > mid[i]
        cdv = c[i-1] >= mid[i-1]   and c[i] < mid[i]

        friday_close = days[i] == 4 and hours[i] >= 15
        timed_out    = hold >= max_hold

        if pos == 1  and (cuv or ss or timed_out or friday_close):
            lx[i] = True; pos = 0; bst = 0; hold = 0
        elif pos == -1 and (cdv or ls or timed_out or friday_close):
            sx[i] = True; pos = 0; bst = 0; hold = 0

        if pos == 0 and bst >= cd and not friday_close:
            if ls:    le[i] = True; pos = 1;  bst = 0; hold = 0
            elif ss:  se[i] = True; pos = -1; bst = 0; hold = 0

    df2 = df.copy()
    df2["long_entry"]  = le;       df2["long_exit"]  = lx | se
    df2["short_entry"] = se;       df2["short_exit"] = sx | le
    return df2


# ── Parse TV CSV ──────────────────────────────────────────────────────────────
def load_tv_trades(csv_path):
    raw = pd.read_csv(csv_path)
    raw.columns = [c.strip() for c in raw.columns]
    raw["Date and time"] = pd.to_datetime(raw["Date and time"])

    entries = raw[raw["Type"].str.contains("Entry")].copy()
    exits   = raw[raw["Type"].str.contains("Exit")].copy()

    trades = []
    for _, erow in entries.iterrows():
        num  = erow["Trade #"]
        xrow = exits[exits["Trade #"] == num]
        if xrow.empty:
            continue
        xrow = xrow.iloc[0]
        direction = "long" if "long" in erow["Type"].lower() else "short"
        trades.append({
            "num":       num,
            "dir":       direction,
            "entry_dt":  erow["Date and time"],
            "entry_px":  erow["Price USD"],
            "exit_dt":   xrow["Date and time"],
            "exit_px":   xrow["Price USD"],
            "pnl":       xrow["Net P&L USD"],
        })
    return pd.DataFrame(trades).sort_values("entry_dt").reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    tv_csv  = Path(__file__).parent.parent / "tvresults" / "Donchian_Mean_Reversion_v2_OANDA_EURUSD_2026-03-03 (1).csv"
    data_csv = "OANDA_EURUSD, 5.csv"

    df = load_tv_export(data_csv)
    sd = str(df.index[0].date())
    ed = "2069-12-31"

    df2 = dc_signals_v2(df)
    cfg = BacktestConfig(
        initial_capital=1000.0, commission_pct=0.0043, slippage_ticks=0,
        qty_type="fixed", qty_value=1000.0, pyramiding=1,
        start_date=sd, end_date=ed,
    )
    kpis = run_backtest_long_short(df2, cfg)
    eng_trades = [t for t in kpis["trades"] if t.exit_date is not None]

    tv_trades = load_tv_trades(tv_csv)

    print("=" * 72)
    print("  DONCHIAN MEAN REVERSION v2 — ENGINE vs TradingView VALIDATION")
    print("=" * 72)
    print(f"\n  TV  trades : {len(tv_trades)}")
    print(f"  Eng trades : {len(eng_trades)}")
    print(f"  TV  Net PnL: ${tv_trades['pnl'].sum():.2f}")
    print(f"  Eng Net PnL: ${kpis['net_profit']:.2f}")
    print(f"  TV  Max DD : {kpis.get('max_drawdown_pct',0):.2f}%  (engine)")

    n = max(len(tv_trades), len(eng_trades))
    cols = f"  {'#':>3}  {'Dir':<5}  {'TV Entry$':>10}  {'Eng Entry$':>10}  {'TV PnL$':>8}  {'Eng PnL$':>8}  {'Status'}"
    print(f"\n{cols}")
    print("  " + "-" * 70)

    mismatches = 0
    first_mismatch = None
    match_count = 0

    for i in range(n):
        if i >= len(tv_trades):
            t = eng_trades[i]
            d = "LONG" if t.direction == "long" else "SHRT"
            print(f"  {i+1:>3}  {d:<5}  {'---':>10}  {t.entry_price:>10.5f}  {'---':>8}  {t.pnl:>8.2f}  ENGINE ONLY")
            mismatches += 1
            continue
        if i >= len(eng_trades):
            tv = tv_trades.iloc[i]
            d = "LONG" if tv.dir == "long" else "SHRT"
            print(f"  {i+1:>3}  {d:<5}  {tv.entry_px:>10.5f}  {'---':>10}  {tv.pnl:>8.2f}  {'---':>8}  TV ONLY")
            mismatches += 1
            continue

        tv  = tv_trades.iloc[i]
        eng = eng_trades[i]
        d_tv  = "LONG" if tv.dir == "long"           else "SHRT"
        d_eng = "LONG" if eng.direction == "long"     else "SHRT"

        # Match criteria: direction + entry price within 0.00002
        dir_ok  = tv.dir == eng.direction
        px_ok   = abs(tv.entry_px - eng.entry_price) < 0.00002
        pnl_ok  = abs(tv.pnl - eng.pnl) < 0.015   # ~1 pip tolerance

        if dir_ok and px_ok and pnl_ok:
            status = "OK"
            match_count += 1
        elif dir_ok and px_ok:
            status = f"PnL diff ({tv.pnl:.2f} vs {eng.pnl:.2f})"
            mismatches += 1
            if first_mismatch is None: first_mismatch = i + 1
        else:
            status = f"MISMATCH dir={d_tv}vs{d_eng} px={tv.entry_px:.5f}vs{eng.entry_price:.5f}"
            mismatches += 1
            if first_mismatch is None: first_mismatch = i + 1

        if i < 20 or status != "OK":
            print(f"  {i+1:>3}  {d_tv:<5}  {tv.entry_px:>10.5f}  {eng.entry_price:>10.5f}  {tv.pnl:>8.2f}  {eng.pnl:>8.2f}  {status}")

    print(f"\n  Matched : {match_count}/{min(len(tv_trades), len(eng_trades))}")
    if first_mismatch:
        print(f"  First mismatch at trade #{first_mismatch}")
    else:
        print("  All compared trades match!")

    # Show TV-only or Engine-only trades at the divergence point if counts differ
    if len(tv_trades) != len(eng_trades):
        print(f"\n  Trade count differs: TV={len(tv_trades)}, Engine={len(eng_trades)}")
        # Find first divergent trade
        for i in range(min(len(tv_trades), len(eng_trades))):
            tv  = tv_trades.iloc[i]
            eng = eng_trades[i]
            if abs(tv.entry_px - eng.entry_price) >= 0.00002 or tv.dir != eng.direction:
                print(f"\n  Divergence at index {i} (trade #{i+1}):")
                print(f"    TV : {tv.dir:<5} entry={tv.entry_dt}  px={tv.entry_px:.5f}  pnl={tv.pnl:.2f}")
                print(f"    Eng: {eng.direction:<5} entry={eng.entry_date}  px={eng.entry_price:.5f}  pnl={eng.pnl:.2f}")
                # Show next 5 from each side
                print(f"\n  Next 5 TV trades from #{i+1}:")
                for j in range(i, min(i+6, len(tv_trades))):
                    r = tv_trades.iloc[j]
                    print(f"    #{j+1:>3} {r.dir:<5} {str(r.entry_dt):>20} @ {r.entry_px:.5f}  pnl={r.pnl:>7.2f}")
                print(f"\n  Next 5 Engine trades from #{i+1}:")
                for j in range(i, min(i+6, len(eng_trades))):
                    t = eng_trades[j]
                    print(f"    #{j+1:>3} {t.direction:<5} {str(t.entry_date):>20} @ {t.entry_price:.5f}  pnl={t.pnl:>7.2f}")
                break

    # Summary stats comparison
    tv_wins  = (tv_trades["pnl"] > 0).sum()
    tv_losses= (tv_trades["pnl"] < 0).sum()
    tv_wr    = tv_wins / len(tv_trades) * 100
    tv_gp    = tv_trades[tv_trades["pnl"] > 0]["pnl"].sum()
    tv_gl    = tv_trades[tv_trades["pnl"] < 0]["pnl"].sum()
    tv_pf    = tv_gp / abs(tv_gl) if tv_gl != 0 else float("inf")

    print(f"\n{'Metric':<25} {'TradingView':>14} {'Engine':>14} {'Diff':>10}")
    print("-" * 65)
    print(f"{'Total Trades':<25} {len(tv_trades):>14} {len(eng_trades):>14} {len(tv_trades)-len(eng_trades):>+10}")
    print(f"{'Win Rate %':<25} {tv_wr:>13.1f}% {kpis.get('win_rate',0):>13.1f}%")
    print(f"{'Profit Factor':<25} {tv_pf:>14.3f} {kpis.get('profit_factor',0):>14.3f} {tv_pf-kpis.get('profit_factor',0):>+10.3f}")
    print(f"{'Net PnL $':<25} {tv_trades['pnl'].sum():>14.2f} {kpis.get('net_profit',0):>14.2f} {tv_trades['pnl'].sum()-kpis.get('net_profit',0):>+10.2f}")
    print(f"{'Gross Profit $':<25} {tv_gp:>14.2f} {kpis.get('gross_profit',0):>14.2f}")
    print(f"{'Gross Loss $':<25} {tv_gl:>14.2f} {kpis.get('gross_loss',0):>14.2f}")


if __name__ == "__main__":
    main()
