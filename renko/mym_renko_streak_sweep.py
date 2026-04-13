"""
MYM Renko Streak Sweep — Pure brick streak strategy (no SHA)

Fine-grained sweep around the winning non-SHA config to find optimal
brick_streak / cooldown combination.

Base: brick_flip exit, exit_tolerance=1

Usage:
    python renko/mym_renko_streak_sweep.py
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

OUTPUT_FILE = ROOT / "ai_context" / "mym_renko_streak_results.json"

PARAM_GRID = {
    "min_brick_streak": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    "cooldown":         [0, 3, 5, 7, 10, 12, 15, 20, 25, 30],
}


def generate_signals(df, min_brick_streak, cooldown):
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
        # Brick exit
        if pos == 1 and not brick_up[i]:
            long_exit[i] = True
            pos = 0
            last_exit_bar = i
        elif pos == -1 and brick_up[i]:
            short_exit[i] = True
            pos = 0
            last_exit_bar = i

        # Entry: pure brick streak — last N bricks all match direction
        if pos == 0 and (i - last_exit_bar) >= cooldown:
            last_n = brick_up[i - min_brick_streak:i]
            # Avoid re-entering on same streak by requiring a non-streak bar before
            prev_n = brick_up[i - min_brick_streak - 1:i - 1]

            brk_long_ok  = bool(np.all(last_n)) and not bool(np.all(prev_n))
            brk_short_ok = bool(not np.any(last_n)) and not bool(not np.any(prev_n))

            if brk_long_ok:
                long_entry[i] = True
                pos = 1
            elif brk_short_ok:
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
    df = generate_signals(df, **params)

    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest_long_short(df, BACKTEST_CONFIG)

    total_trades = kpis.get("total_trades", 0)
    if total_trades < 10:
        return None

    trades_list = kpis.get("trades", [])
    n_long = sum(1 for t in trades_list if t.direction == "long" and t.exit_date is not None)
    n_short = sum(1 for t in trades_list if t.direction == "short" and t.exit_date is not None)

    return {
        "params": params,
        "total_trades": total_trades,
        "win_rate": round(kpis.get("win_rate", 0), 1),
        "profit_factor": round(kpis.get("profit_factor", 0), 2),
        "net_profit": round(kpis.get("net_profit", 0), 2),
        "max_drawdown": round(kpis.get("max_drawdown", 0), 2),
        "avg_trade": round(kpis.get("avg_trade", 0), 2),
        "avg_win_loss_ratio": round(kpis.get("avg_win_loss_ratio", 0), 2),
        "long_trades": n_long,
        "short_trades": n_short,
    }


def main():
    import pickle

    print(f"Loading {RENKO_FILE}...")
    df = load_renko_export(RENKO_FILE)
    df_pickle = pickle.dumps(df)

    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"  {len(combos)} combos")

    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run_single, p, df_pickle): p for p in combos}
        for future in as_completed(futures):
            r = future.result()
            if r is not None:
                results.append(r)

    elapsed = time.time() - t0
    print(f"  {len(results)} viable in {elapsed:.1f}s\n")

    results.sort(key=lambda r: r["profit_factor"], reverse=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump({"results": results}, f, indent=2, default=str)

    # === TOP 30 by PF ===
    print(f"{'='*110}")
    print(f"  TOP 30 BY PROFIT FACTOR")
    print(f"{'='*110}")
    print(f"  {'BrkStrk':>8} {'CD':>3} | {'PF':>7} {'T':>5} {'WR%':>5} {'Net$':>10} {'AvgT':>8} {'DD':>9} {'L/S':>9}")
    print(f"  {'-'*15} | {'-'*55}")
    for r in results[:30]:
        p = r["params"]
        ls = f"{r['long_trades']}/{r['short_trades']}"
        print(f"  {p['min_brick_streak']:>8} {p['cooldown']:>3} | "
              f"{r['profit_factor']:>7.2f} {r['total_trades']:>5} {r['win_rate']:>5.1f} "
              f"{r['net_profit']:>10.2f} {r['avg_trade']:>8.2f} {r['max_drawdown']:>9.2f} "
              f"{ls:>9}")

    # === TOP 10 by Net Profit ===
    print(f"\n{'='*110}")
    print(f"  TOP 10 BY NET PROFIT")
    print(f"{'='*110}")
    by_net = sorted(results, key=lambda r: r["net_profit"], reverse=True)
    for r in by_net[:10]:
        p = r["params"]
        ls = f"{r['long_trades']}/{r['short_trades']}"
        print(f"  {p['min_brick_streak']:>8} {p['cooldown']:>3} | "
              f"{r['profit_factor']:>7.2f} {r['total_trades']:>5} {r['win_rate']:>5.1f} "
              f"{r['net_profit']:>10.2f} {r['avg_trade']:>8.2f} {r['max_drawdown']:>9.2f} "
              f"{ls:>9}")

    # === Heatmap ===
    print(f"\n{'='*110}")
    print(f"  PROFIT FACTOR HEATMAP (rows=brick_streak, cols=cooldown)")
    print(f"{'='*110}")
    streaks  = sorted(set(r["params"]["min_brick_streak"] for r in results))
    cooldowns = sorted(set(r["params"]["cooldown"] for r in results))
    print(f"  {'Streak\\CD':>10} | " + " ".join(f"{cd:>6}" for cd in cooldowns))
    print(f"  {'-'*10} | " + "-" * (7 * len(cooldowns)))
    for s in streaks:
        row = [f"{s:>10}", "|"]
        for cd in cooldowns:
            match = next((r for r in results if r["params"]["min_brick_streak"] == s and r["params"]["cooldown"] == cd), None)
            row.append(f"{match['profit_factor']:>6.1f}" if match else f"{'-':>6}")
        print("  " + " ".join(row))


if __name__ == "__main__":
    main()
