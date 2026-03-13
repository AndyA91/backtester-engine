"""Compare engine vs TV CSV for gaussian_channel_reversal_1 (P=150, m=8.0, cd=135)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from math import comb, cos, pi
from engine import load_tv_export, BacktestConfig, run_backtest_long_short, print_kpis

# ── IIR filter ────────────────────────────────────────────────────────────────
def _iir_alpha(period, poles):
    beta = (1 - cos(2*pi / period)) / (1.414 ** (2.0 / poles) - 1)
    return -beta + (beta**2 + 2*beta)**0.5

def _iir_filter(alpha, src, n_poles):
    x = 1.0 - alpha; n = len(src); f = np.zeros(n)
    for i in range(n):
        s = src[i] if not np.isnan(src[i]) else 0.0
        val = alpha**n_poles * s
        for k in range(1, n_poles+1):
            prev = f[i-k] if i >= k else 0.0
            val += (-1)**(k+1) * comb(n_poles, k) * x**k * prev
        f[i] = val
    return f

# ── Signal generation ─────────────────────────────────────────────────────────
df = load_tv_export("OANDA_EURUSD, 2 (1).csv")
close = df["Close"].values; highs = df["High"].values; lows = df["Low"].values
n = len(close); dates = df.index

per, pol, mult, cd = 150, 4, 8.0, 135
alpha = _iir_alpha(per, pol)
prev_c = np.roll(close, 1); prev_c[0] = close[0]
tr  = np.maximum(highs-lows, np.maximum(np.abs(highs-prev_c), np.abs(lows-prev_c)))
mid = _iir_filter(alpha, close, pol)
ftr = _iir_filter(alpha, tr,    pol)
gc_up = mid + ftr * mult
gc_lo = mid - ftr * mult

le, lx, se, sx = (np.zeros(n, bool) for _ in range(4))
pos = 0; bst = cd

for i in range(1, n):
    bst += 1
    pc = close[i-1]; cc = close[i]
    cam = pc <= mid[i-1] and cc >  mid[i]
    cbm = pc >= mid[i-1] and cc <  mid[i]
    cbu = pc >= gc_up[i-1] and cc < gc_up[i]
    cal = pc <= gc_lo[i-1] and cc > gc_lo[i]
    if   pos ==  1 and cbm: lx[i] = True; pos = 0; bst = 0
    elif pos == -1 and cam: sx[i] = True; pos = 0; bst = 0
    if pos == 0 and bst >= cd:
        if   cal: le[i] = True; pos =  1; bst = 0
        elif cbu: se[i] = True; pos = -1; bst = 0

df2 = df.copy()
df2["long_entry"]  = le;    df2["long_exit"]  = lx | se
df2["short_entry"] = se;    df2["short_exit"] = sx | le

config = BacktestConfig(
    initial_capital=1000.0, commission_pct=0.0085, slippage_ticks=0,
    qty_type="fixed", qty_value=1000.0, pyramiding=1,
    start_date="2000-01-01", end_date="2069-12-31",
)
kpis = run_backtest_long_short(df2, config)

print_kpis(kpis)

closed = [t for t in kpis["trades"] if t.exit_date is not None]
open_t = [t for t in kpis["trades"] if t.exit_date is None]
print(f"\nClosed: {len(closed)}  Open: {len(open_t)}")

# ── Load TV CSV ───────────────────────────────────────────────────────────────
tv_path = Path(__file__).resolve().parent.parent / "tvresults" / \
          "Gaussian_Channel_Reversal_1_OANDA_EURUSD_2026-03-03.csv"
tv = pd.read_csv(tv_path)
tv_entries = tv[tv["Type"].str.startswith("Entry")].reset_index(drop=True)
tv_exits   = tv[tv["Type"].str.startswith("Exit")].reset_index(drop=True)
tv_closed  = tv_exits[tv_exits["Signal"] != "Open"].reset_index(drop=True)
tv_open    = tv_exits[tv_exits["Signal"] == "Open"]

tv_net_closed = tv_closed["Net P&L USD"].sum()
tv_open_pnl   = tv_open["Net P&L USD"].sum() if not tv_open.empty else 0.0

print(f"\n{'='*60}")
print(f"  TV vs Engine comparison")
print(f"{'='*60}")
print(f"  {'Metric':<28} {'Engine':>12} {'TV':>12}")
print(f"  {'-'*54}")
print(f"  {'Closed trades':<28} {len(closed):>12} {len(tv_closed):>12}")
print(f"  {'Net Profit (closed) $':<28} {kpis['net_profit']:>12.2f} {tv_net_closed:>12.2f}")
print(f"  {'Open P&L $':<28} {kpis['open_profit']:>12.2f} {tv_open_pnl:>12.2f}")
print(f"  {'Total P&L $':<28} {kpis['net_profit']+kpis['open_profit']:>12.2f} {tv_net_closed+tv_open_pnl:>12.2f}")
print(f"  {'Win Rate %':<28} {kpis['win_rate']:>12.1f} {(tv_closed['Net P&L USD']>0).mean()*100:>12.1f}")
print(f"  {'Profit Factor':<28} {kpis['profit_factor']:>12.3f}")

# ── First TV entry (to check data coverage) ────────────────────────────────────
tv_first_entry = pd.Timestamp(tv_entries.iloc[0]["Date and time"])
engine_first   = closed[0].entry_date if closed else None
print(f"\n  TV first entry:     {tv_first_entry}")
print(f"  Engine first entry: {engine_first}")
print(f"  Engine data start:  {df.index[0]}")

# ── Trade-by-trade comparison (align by date) ─────────────────────────────────
print(f"\n{'='*60}")
print(f"  First 10 closed trades — side-by-side")
print(f"{'='*60}")
print(f"  {'#':>3}  {'Dir':<5}  {'Engine Entry':>20}  {'TV Entry':>20}  {'EngPnL':>8}  {'TVPnL':>8}  Match")
print(f"  {'-'*80}")

# Build TV trade list aligned to closed
tv_entry_list = tv_entries[~tv_entries["Trade #"].isin(tv_open["Trade #"].values)].reset_index(drop=True)

n_compare = min(len(closed), len(tv_entry_list), 10)
mismatches = 0
for i in range(n_compare):
    et = closed[i]
    tv_e = tv_entry_list.iloc[i]
    tv_date = pd.Timestamp(tv_e["Date and time"])
    tv_pnl  = tv_closed.iloc[i]["Net P&L USD"] if i < len(tv_closed) else float("nan")

    date_ok = abs((et.entry_date - tv_date).total_seconds()) < 300  # within 5 min
    pnl_ok  = abs(et.pnl - tv_pnl) < max(0.20, abs(tv_pnl) * 0.05)
    ok = date_ok and pnl_ok
    if not ok:
        mismatches += 1
    marker = "OK" if ok else "!!"
    print(f"  {i+1:>3}  {et.direction.upper():<5}  {str(et.entry_date):>20}  "
          f"{str(tv_date):>20}  {et.pnl:>8.2f}  {tv_pnl:>8.2f}  {marker}")

print(f"\n  Matched {n_compare - mismatches}/{n_compare} of first {n_compare} trades")
