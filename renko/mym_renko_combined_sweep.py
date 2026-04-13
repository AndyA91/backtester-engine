"""
MYM Renko SHA — Combined Sweep (3 questions in one)

Tests:
1. Is SHA necessary? (use_sha=False = brick_streak alone)
2. Re-sweep SHA params with brick_streak locked (find true optimum)
3. Sweep brick_streak values 5-10

Base: cd=20, brick_flip exit, exit_tolerance=1

Usage:
    python renko/mym_renko_combined_sweep.py
"""

import contextlib
import io
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import BacktestConfig, run_backtest_long_short
from renko.data import load_renko_export
from renko.config import MAX_WORKERS

RENKO_FILE = "CBOT_MINI_MYM1!, 1S renko 15.csv"

BACKTEST_CONFIG = BacktestConfig(
    initial_capital=10000.0,
    commission_pct=0.002,
    qty_type="fixed",
    qty_value=1,
    start_date="2000-01-01",
    end_date="2099-12-31",
)

OUTPUT_FILE = ROOT / "ai_context" / "mym_renko_combined_results.json"

# Test SHA on/off + extended SHA grid + extended brick_streak
PARAM_GRID = {
    "use_sha":          [False, True],
    "fast_len":         [3, 5, 8, 10, 12, 15, 20],
    "slow_len":         [10, 14, 20, 25, 30, 40],
    "min_brick_streak": [3, 4, 5, 6, 7, 8, 10],
    "cooldown":         [10, 20, 30],
}


def _ema(src, length):
    out = np.empty_like(src, dtype=float)
    out[0] = src[0]
    k = 2.0 / (length + 1)
    for i in range(1, len(src)):
        out[i] = src[i] * k + out[i - 1] * (1 - k)
    return out


def _smoothed_ha(o, h, l, c, length):
    s_o = _ema(o, length)
    s_h = _ema(h, length)
    s_l = _ema(l, length)
    s_c = _ema(c, length)
    n = len(o)
    ha_open = np.empty(n, dtype=float)
    ha_close = (s_o + s_h + s_l + s_c) / 4.0
    ha_open[0] = (s_o[0] + s_c[0]) / 2.0
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
    return ha_close >= ha_open


def generate_signals(df, use_sha, fast_len, slow_len, min_brick_streak, cooldown):
    o = df["Open"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    c = df["Close"].values.astype(float)
    brick_up = df["brick_up"].values
    n = len(df)

    if use_sha:
        f_bull = _smoothed_ha(o, h, l, c, fast_len)
        s_bull = _smoothed_ha(o, h, l, c, slow_len)
    else:
        f_bull = np.zeros(n, dtype=bool)
        s_bull = np.zeros(n, dtype=bool)

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    warmup = max(fast_len, slow_len, min_brick_streak) + 2
    last_exit_bar = -999_999
    pos = 0

    for i in range(warmup, n):
        # Brick exit (tolerance=1)
        if pos == 1 and not brick_up[i]:
            long_exit[i] = True
            pos = 0
            last_exit_bar = i
        elif pos == -1 and brick_up[i]:
            short_exit[i] = True
            pos = 0
            last_exit_bar = i

        # Entry
        if pos == 0 and (i - last_exit_bar) >= cooldown:
            # Brick streak check
            last_n = brick_up[i - min_brick_streak:i]
            brk_long_ok  = bool(np.all(last_n))
            brk_short_ok = bool(not np.any(last_n))

            if use_sha:
                fast_bull_flip = f_bull[i] and not f_bull[i - 1]
                fast_bear_flip = not f_bull[i] and f_bull[i - 1]

                if fast_bull_flip and s_bull[i] and brk_long_ok:
                    long_entry[i] = True
                    pos = 1
                elif fast_bear_flip and not s_bull[i] and brk_short_ok:
                    short_entry[i] = True
                    pos = -1
            else:
                # Pure brick streak entry: enter on the first brick that completes the streak
                # AND is the first brick after a non-streak (avoid re-entering on the same streak)
                if brk_long_ok and not bool(np.all(brick_up[i - min_brick_streak - 1:i - 1])):
                    long_entry[i] = True
                    pos = 1
                elif brk_short_ok and not bool(not np.any(brick_up[i - min_brick_streak - 1:i - 1])):
                    short_entry[i] = True
                    pos = -1

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df


def _run_single(params, df_pickle):
    import pickle
    df = pickle.loads(df_pickle)

    # Skip SHA params if SHA disabled (deduplicates results)
    if not params["use_sha"] and (params["fast_len"] != 3 or params["slow_len"] != 10):
        return None

    df = generate_signals(df, **params)

    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest_long_short(df, BACKTEST_CONFIG)

    total_trades = kpis.get("total_trades", 0)
    if total_trades < 10:
        return None

    return {
        "params": params,
        "total_trades": total_trades,
        "win_rate": round(kpis.get("win_rate", 0), 1),
        "profit_factor": round(kpis.get("profit_factor", 0), 2),
        "net_profit": round(kpis.get("net_profit", 0), 2),
        "max_drawdown": round(kpis.get("max_drawdown", 0), 2),
        "avg_trade": round(kpis.get("avg_trade", 0), 2),
        "avg_win_loss_ratio": round(kpis.get("avg_win_loss_ratio", 0), 2),
    }


def main():
    import pickle

    print(f"Loading {RENKO_FILE}...")
    df = load_renko_export(RENKO_FILE)
    df_pickle = pickle.dumps(df)

    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    # Filter fast < slow when SHA enabled
    combos = [c for c in combos if (not c["use_sha"]) or c["fast_len"] < c["slow_len"]]
    print(f"  {len(combos)} param combos (after filters)")
    print(f"  Workers: {MAX_WORKERS}")

    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run_single, p, df_pickle): p for p in combos}

        done = 0
        for future in as_completed(futures):
            done += 1
            if done % 200 == 0 or done == len(combos):
                elapsed = time.time() - t0
                print(f"  [{done}/{len(combos)}] {elapsed:.1f}s")
            r = future.result()
            if r is not None:
                results.append(r)

    elapsed = time.time() - t0
    print(f"\n  {len(results)} viable configs in {elapsed:.1f}s\n")

    results.sort(key=lambda r: r["profit_factor"], reverse=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump({"results": results}, f, indent=2, default=str)

    # === Q1: Is SHA necessary? ===
    print(f"{'='*110}")
    print(f"  Q1: IS SHA NECESSARY?")
    print(f"{'='*110}")
    no_sha = [r for r in results if not r["params"]["use_sha"]]
    no_sha.sort(key=lambda r: r["profit_factor"], reverse=True)
    print("  Best configs WITHOUT SHA:")
    print(f"  {'BrkStrk':>8} {'CD':>3} | {'PF':>7} {'T':>5} {'WR%':>5} {'Net$':>10} {'AvgT':>8} {'DD':>9}")
    for r in no_sha[:10]:
        p = r["params"]
        print(f"  {p['min_brick_streak']:>8} {p['cooldown']:>3} | "
              f"{r['profit_factor']:>7.2f} {r['total_trades']:>5} {r['win_rate']:>5.1f} "
              f"{r['net_profit']:>10.2f} {r['avg_trade']:>8.2f} {r['max_drawdown']:>9.2f}")

    # === Q2/Q3: Top SHA configs ===
    print(f"\n{'='*110}")
    print(f"  Q2/Q3: TOP SHA CONFIGS (extended brick_streak + SHA params)")
    print(f"{'='*110}")
    with_sha = [r for r in results if r["params"]["use_sha"]]
    with_sha.sort(key=lambda r: r["profit_factor"], reverse=True)
    print(f"  {'F':>2} {'S':>2} {'BrkStrk':>8} {'CD':>3} | {'PF':>7} {'T':>5} {'WR%':>5} {'Net$':>10} {'AvgT':>8} {'DD':>9}")
    for r in with_sha[:25]:
        p = r["params"]
        print(f"  {p['fast_len']:>2} {p['slow_len']:>2} {p['min_brick_streak']:>8} {p['cooldown']:>3} | "
              f"{r['profit_factor']:>7.2f} {r['total_trades']:>5} {r['win_rate']:>5.1f} "
              f"{r['net_profit']:>10.2f} {r['avg_trade']:>8.2f} {r['max_drawdown']:>9.2f}")

    # === Comparison ===
    print(f"\n{'='*110}")
    print(f"  HEAD-TO-HEAD: Best SHA vs Best Non-SHA")
    print(f"{'='*110}")
    if no_sha and with_sha:
        best_no = no_sha[0]
        best_yes = with_sha[0]
        print(f"  Best without SHA: PF={best_no['profit_factor']:.2f}, {best_no['total_trades']}T, "
              f"WR={best_no['win_rate']}%, Net=${best_no['net_profit']:.0f}")
        print(f"  Best with SHA:    PF={best_yes['profit_factor']:.2f}, {best_yes['total_trades']}T, "
              f"WR={best_yes['win_rate']}%, Net=${best_yes['net_profit']:.0f}")
        print(f"  SHA contributes: PF +{best_yes['profit_factor']-best_no['profit_factor']:.2f}, "
              f"Net +${best_yes['net_profit']-best_no['net_profit']:.0f}")


if __name__ == "__main__":
    main()
