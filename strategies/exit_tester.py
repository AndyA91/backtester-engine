"""
Exit improvement tester v3 — uses engine's built-in TP/SL.
Also tests max hold at signal level (the only thing signal-level exit can do reliably).
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from math import comb, cos, pi

from engine import load_tv_export, BacktestConfig, run_backtest_long_short

RESULTS_FILE = Path(__file__).resolve().parent / "exit_results.txt"

def _iir_alpha(period, poles):
    beta = (1 - cos(2*pi / period)) / (1.414 ** (2.0 / poles) - 1)
    return -beta + (beta**2 + 2*beta)**0.5

def _iir_filter(alpha, src, n_poles):
    x = 1.0 - alpha; n = len(src); f = np.zeros(n)
    for i in range(n):
        s = src[i] if not np.isnan(src[i]) else 0.0
        val = alpha**n_poles * s
        for k in range(1, n_poles+1):
            prev = f[i-k] if i>=k else 0.0
            val += (-1)**(k+1) * comb(n_poles, k) * x**k * prev
        f[i] = val
    return f


def gen_signals(close, gc_mid, gc_up, gc_lo, n, cooldown, max_hold=0):
    """Base reversal signals + optional max hold exit."""
    le, lx, se, sx = (np.zeros(n, bool) for _ in range(4))
    pos = 0; bst = cooldown; bars_in = 0

    for i in range(1, n):
        bst += 1
        if pos != 0:
            bars_in += 1
        pc = close[i-1]; cc = close[i]

        cal = pc <= gc_lo[i-1] and cc > gc_lo[i]
        cbu = pc >= gc_up[i-1] and cc < gc_up[i]
        cam = pc <= gc_mid[i-1] and cc > gc_mid[i]
        cbm = pc >= gc_mid[i-1] and cc < gc_mid[i]

        # Exits
        should_exit = max_hold > 0 and bars_in >= max_hold
        if pos == 1 and (cbm or should_exit):
            lx[i] = True; pos = 0; bst = 0; bars_in = 0
        elif pos == -1 and (cam or should_exit):
            sx[i] = True; pos = 0; bst = 0; bars_in = 0

        # Entries
        if pos == 0 and bst >= cooldown:
            if cal:
                le[i] = True; pos = 1; bst = 0; bars_in = 0
            elif cbu:
                se[i] = True; pos = -1; bst = 0; bars_in = 0

    return le, lx, se, sx


def run_test(df_raw, close, gc_mid, gc_up, gc_lo, n, name,
             tp_pct=0.0, sl_pct=0.0, max_hold=0):
    le, lx, se, sx = gen_signals(close, gc_mid, gc_up, gc_lo, n,
                                  cooldown=90, max_hold=max_hold)
    if not le.any() and not se.any():
        return None

    df = df_raw.copy()
    df["long_entry"] = le; df["long_exit"] = lx | se
    df["short_entry"] = se; df["short_exit"] = sx | le

    start = str(df_raw.index[0].date()); end = "2069-12-31"
    config = BacktestConfig(
        initial_capital=1000.0, commission_pct=0.0085, slippage_ticks=0,
        qty_type="fixed", qty_value=1000.0, pyramiding=1,
        start_date=start, end_date=end,
        take_profit_pct=tp_pct,
        stop_loss_pct=sl_pct,
    )

    try:
        kpis = run_backtest_long_short(df, config)
        closed = [t for t in kpis["trades"] if t.exit_date is not None]
        if len(closed) < 3:
            return None
        return {
            "name": name, "tr": len(closed), "wr": kpis["win_rate"],
            "pf": kpis["profit_factor"], "net": kpis["net_profit"],
            "dd_pct": kpis["max_drawdown_pct"], "avg": kpis["avg_trade"],
            "lg_win": kpis["largest_winning"], "lg_loss": kpis["largest_losing"],
            "comm": kpis["total_commission"],
        }
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def write_results(results):
    ranked = sorted(results, key=lambda r: (-r["pf"], r["dd_pct"]))
    with open(RESULTS_FILE, "w") as f:
        f.write(f"EXIT IMPROVEMENT RESULTS — {len(ranked)} variants tested\n")
        f.write(f"Base: GC Reversal P=500 m=5.0 cd=90 | OANDA EURUSD 1-min\n")
        f.write(f"TP/SL are percentage-based from entry price\n")
        f.write(f"Ranked by Profit Factor\n\n")
        hdr = f"{'#':>3}  {'Exit Type':<35} {'Tr':>4} {'WR%':>6} {'PF':>7} {'Net$':>8} {'DD%':>6} {'Avg$':>6} {'LgW':>5} {'LgL':>6}"
        f.write(hdr + "\n" + "-"*len(hdr) + "\n")
        for i, r in enumerate(ranked, 1):
            f.write(f"{i:>3}  {r['name']:<35} {r['tr']:>4} {r['wr']:>5.1f}% {r['pf']:>7.3f} "
                    f"{r['net']:>7.2f} {r['dd_pct']:>5.2f}% {r['avg']:>5.2f} {r['lg_win']:>5.2f} {r['lg_loss']:>5.2f}\n")


def main():
    t0 = time.time()
    df_raw = load_tv_export("OANDA_EURUSD, 1.csv")

    close = df_raw["Close"].values
    highs = df_raw["High"].values
    lows = df_raw["Low"].values
    n = len(close)

    # Precompute channel
    alpha = _iir_alpha(500, 4)
    prev_c = np.roll(close, 1); prev_c[0] = close[0]
    tr = np.maximum(highs-lows, np.maximum(np.abs(highs-prev_c), np.abs(lows-prev_c)))
    gc_mid = _iir_filter(alpha, close, 4)
    ftr = _iir_filter(alpha, tr, 4)
    gc_up = gc_mid + ftr * 5.0
    gc_lo = gc_mid - ftr * 5.0

    # For EURUSD at ~1.18:
    #   10 pips = 0.0010 = 0.085% of price
    #   20 pips = 0.0020 = 0.170%
    #   30 pips = 0.0030 = 0.254%
    #   50 pips = 0.0050 = 0.424%
    # So sl_pct/tp_pct values roughly correspond to:
    #   0.05% ≈ 6 pips
    #   0.10% ≈ 12 pips
    #   0.15% ≈ 18 pips
    #   0.20% ≈ 24 pips
    #   0.25% ≈ 30 pips
    #   0.50% ≈ 59 pips

    # (name, tp_pct, sl_pct, max_hold)
    tests = [
        # Baseline
        ("Baseline (midline only)",            0,    0,    0),

        # SL only
        ("SL 0.05% (~6 pips)",                 0,    0.05, 0),
        ("SL 0.10% (~12 pips)",                0,    0.10, 0),
        ("SL 0.15% (~18 pips)",                0,    0.15, 0),
        ("SL 0.20% (~24 pips)",                0,    0.20, 0),
        ("SL 0.25% (~30 pips)",                0,    0.25, 0),
        ("SL 0.30% (~35 pips)",                0,    0.30, 0),
        ("SL 0.50% (~59 pips)",                0,    0.50, 0),

        # TP only
        ("TP 0.05% (~6 pips)",                 0.05, 0,    0),
        ("TP 0.10% (~12 pips)",                0.10, 0,    0),
        ("TP 0.15% (~18 pips)",                0.15, 0,    0),
        ("TP 0.20% (~24 pips)",                0.20, 0,    0),
        ("TP 0.25% (~30 pips)",                0.25, 0,    0),

        # SL + TP combos
        ("SL 0.10 + TP 0.10",                  0.10, 0.10, 0),
        ("SL 0.10 + TP 0.15",                  0.15, 0.10, 0),
        ("SL 0.10 + TP 0.20",                  0.20, 0.10, 0),
        ("SL 0.15 + TP 0.10",                  0.10, 0.15, 0),
        ("SL 0.15 + TP 0.15",                  0.15, 0.15, 0),
        ("SL 0.15 + TP 0.20",                  0.20, 0.15, 0),
        ("SL 0.20 + TP 0.10",                  0.10, 0.20, 0),
        ("SL 0.20 + TP 0.15",                  0.15, 0.20, 0),
        ("SL 0.20 + TP 0.20",                  0.20, 0.20, 0),
        ("SL 0.25 + TP 0.15",                  0.15, 0.25, 0),
        ("SL 0.25 + TP 0.20",                  0.20, 0.25, 0),
        ("SL 0.25 + TP 0.25",                  0.25, 0.25, 0),
        ("SL 0.30 + TP 0.20",                  0.20, 0.30, 0),

        # Max hold only
        ("MaxHold 120 (2hr)",                   0,    0,    120),
        ("MaxHold 180 (3hr)",                   0,    0,    180),
        ("MaxHold 240 (4hr)",                   0,    0,    240),
        ("MaxHold 360 (6hr)",                   0,    0,    360),

        # SL + max hold
        ("SL 0.20 + MH 240",                   0,    0.20, 240),
        ("SL 0.25 + MH 240",                   0,    0.25, 240),
        ("SL 0.25 + MH 360",                   0,    0.25, 360),

        # SL + TP + max hold (triple)
        ("SL 0.15 + TP 0.15 + MH 240",         0.15, 0.15, 240),
        ("SL 0.20 + TP 0.15 + MH 240",         0.15, 0.20, 240),
        ("SL 0.25 + TP 0.15 + MH 240",         0.15, 0.25, 240),
        ("SL 0.25 + TP 0.20 + MH 240",         0.20, 0.25, 240),
    ]

    results = []
    for idx, (name, tp, sl, mh) in enumerate(tests, 1):
        print(f"[{idx}/{len(tests)}] {name} ...", end=" ", flush=True)
        r = run_test(df_raw, close, gc_mid, gc_up, gc_lo, n, name,
                     tp_pct=tp, sl_pct=sl, max_hold=mh)
        if r:
            results.append(r)
            print(f"PF={r['pf']:.3f} Net=${r['net']:.2f} DD={r['dd_pct']:.2f}% WR={r['wr']:.1f}% Tr={r['tr']}")
        else:
            print("skipped (<3 trades)")
        write_results(results)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s — {len(results)} valid / {len(tests)} tested")
    print(f"Results: {RESULTS_FILE}")

    if results:
        ranked = sorted(results, key=lambda r: (-r["pf"], r["dd_pct"]))
        print(f"\nTOP 5:")
        for i, r in enumerate(ranked[:5], 1):
            print(f"  {i}. {r['name']}  PF={r['pf']:.3f} Net=${r['net']:.2f} DD={r['dd_pct']:.2f}% WR={r['wr']:.1f}% Tr={r['tr']}")

if __name__ == "__main__":
    main()
