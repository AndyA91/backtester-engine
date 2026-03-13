"""
Gaussian Channel — sequential test of handpicked combos.
Writes results to gc_results.txt after each test.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from math import comb, cos, pi

from engine import load_tv_export, BacktestConfig, run_backtest_long_short


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

def run_combo(df_raw, close, highs, lows, n, config, combo, iir_cache, ts_start, ts_end):
    per, pol, mult, cd, sig = combo
    key = (per, pol)
    if key not in iir_cache:
        alpha = _iir_alpha(per, pol)
        prev_c = np.roll(close, 1); prev_c[0] = close[0]
        tr = np.maximum(highs-lows, np.maximum(np.abs(highs-prev_c), np.abs(lows-prev_c)))
        iir_cache[key] = (_iir_filter(alpha, close, pol), _iir_filter(alpha, tr, pol))

    mid, ftr = iir_cache[key]
    gc_up = mid + ftr * mult
    gc_lo = mid - ftr * mult
    dates = df_raw.index

    le,lx,se,sx = (np.zeros(n,bool) for _ in range(4))
    pos=0; bst=cd; slope_len = max(10, per//10)

    for i in range(1,n):
        bst+=1; pc=close[i-1]; cc=close[i]
        # Common crossovers
        cau = pc<=gc_up[i-1] and cc>gc_up[i]
        cbl = pc>=gc_lo[i-1] and cc<gc_lo[i]
        cam = pc<=mid[i-1] and cc>mid[i]
        cbm = pc>=mid[i-1] and cc<mid[i]

        if not (ts_start <= dates[i] <= ts_end): continue

        if sig == "basic":
            if pos==1 and cbm: lx[i]=True; pos=0; bst=0
            elif pos==-1 and cam: sx[i]=True; pos=0; bst=0
            if pos==0 and bst>=cd:
                if cau: le[i]=True; pos=1; bst=0
                elif cbl: se[i]=True; pos=-1; bst=0

        elif sig == "trend":
            mr = i>=slope_len and mid[i]>mid[i-slope_len]
            mf = i>=slope_len and mid[i]<mid[i-slope_len]
            if pos==1 and cbm: lx[i]=True; pos=0; bst=0
            elif pos==-1 and cam: sx[i]=True; pos=0; bst=0
            if pos==0 and bst>=cd:
                if cau and mr: le[i]=True; pos=1; bst=0
                elif cbl and mf: se[i]=True; pos=-1; bst=0

        elif sig == "reversal":
            cbu = pc>=gc_up[i-1] and cc<gc_up[i]
            cal = pc<=gc_lo[i-1] and cc>gc_lo[i]
            if pos==1 and cbm: lx[i]=True; pos=0; bst=0
            elif pos==-1 and cam: sx[i]=True; pos=0; bst=0
            if pos==0 and bst>=cd:
                if cal: le[i]=True; pos=1; bst=0
                elif cbu: se[i]=True; pos=-1; bst=0

    if not le.any() and not se.any():
        return None

    df = df_raw.copy()
    df["long_entry"]=le; df["long_exit"]=lx|se
    df["short_entry"]=se; df["short_exit"]=sx|le

    kpis = run_backtest_long_short(df, config)
    closed = [t for t in kpis["trades"] if t.exit_date is not None]
    if len(closed) < 3: return None

    return {
        "sig":sig, "per":per, "pol":pol, "mult":mult, "cd":cd,
        "tr":len(closed), "wr":kpis["win_rate"], "pf":kpis["profit_factor"],
        "net":kpis["net_profit"], "net_pct":kpis["net_profit_pct"],
        "dd":kpis["max_drawdown"], "dd_pct":kpis["max_drawdown_pct"],
        "avg":kpis["avg_trade"],
    }

def write_results(results, results_file):
    ranked = sorted(results, key=lambda r: (-r["pf"], r["dd_pct"]))
    with open(results_file, "w") as f:
        f.write(f"GAUSSIAN CHANNEL RESULTS — {len(ranked)} combos tested\n")
        f.write(f"Fixed 1000 units, ~$0.10/side | Ranked by Profit Factor\n\n")
        hdr = f"{'#':>3}  {'Type':<10} {'Per':>4} {'Mult':>4} {'CD':>4}  {'Tr':>4} {'WR%':>6} {'PF':>7} {'Net$':>8} {'DD%':>6} {'Avg$':>6}"
        f.write(hdr + "\n" + "-"*len(hdr) + "\n")
        for i, r in enumerate(ranked, 1):
            f.write(f"{i:>3}  {r['sig']:<10} {r['per']:>4} {r['mult']:>4.1f} {r['cd']:>4}  "
                    f"{r['tr']:>4} {r['wr']:>5.1f}% {r['pf']:>7.3f} {r['net']:>7.2f} {r['dd_pct']:>5.2f}% {r['avg']:>5.2f}\n")

def main():
    t0 = time.time()
    filename = sys.argv[1] if len(sys.argv) > 1 else "OANDA_EURUSD, 1.csv"

    # Results file named after the input data so runs don't overwrite each other
    stem = Path(filename).stem.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
    results_file = Path(__file__).resolve().parent / f"gc_results_{stem}.txt"

    df_raw = load_tv_export(filename)
    start = "2000-01-01"; end = "2069-12-31"
    ts_start = df_raw.index[0]; ts_end = pd.Timestamp(end)
    config = BacktestConfig(
        initial_capital=1000.0, commission_pct=0.0085, slippage_ticks=0,
        qty_type="fixed", qty_value=1000.0, pyramiding=1,
        start_date=start, end_date=end,
    )
    close = df_raw["Close"].values; highs = df_raw["High"].values; lows = df_raw["Low"].values
    n = len(close)

    # Fine-tune around broad-sweep winner: P=150, mult=8.0, cd=120
    combos = [
        (per, 4, mult, cd, "reversal")
        for per  in [120, 130, 140, 150, 160, 175]
        for mult in [7.0, 7.5, 8.0, 8.5, 9.0]
        for cd   in [90, 105, 120, 135, 150]
    ]

    iir_cache = {}
    results = []

    for idx, combo in enumerate(combos, 1):
        per, pol, mult, cd, sig = combo
        print(f"[{idx}/{len(combos)}] {sig} P={per} m={mult} cd={cd} ...", end=" ", flush=True)
        r = run_combo(df_raw, close, highs, lows, n, config, combo, iir_cache, ts_start, ts_end)
        if r:
            results.append(r)
            print(f"PF={r['pf']:.3f} Net=${r['net']:.2f} DD={r['dd_pct']:.2f}% WR={r['wr']:.1f}% Trades={r['tr']}")
        else:
            print("skipped (no/few trades)")
        write_results(results, results_file)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s — {len(results)} valid combos")
    print(f"Results: {results_file}")

    ranked = sorted(results, key=lambda r: (-r["pf"], r["dd_pct"]))
    print(f"\nBEST BY PROFIT FACTOR:")
    for i, r in enumerate(ranked[:5],1):
        print(f"  {i}. {r['sig']} P={r['per']} m={r['mult']} cd={r['cd']}  "
              f"PF={r['pf']:.3f} Net=${r['net']:.2f} DD={r['dd_pct']:.2f}% Trades={r['tr']}")

if __name__ == "__main__":
    main()
