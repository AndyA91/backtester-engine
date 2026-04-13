"""
MYM Renko SHA — Loser Filter Sweep

Takes the winning base config (f=12, s=14, cd=20, brick_flip) and tests filters
to eliminate the immediate-reversal losers.

Filters tested:
- min_slow_streak: slow SHA must hold color N bricks before entry
- min_brick_streak: previous N bricks must agree with entry direction

Usage:
    python renko/mym_renko_sha_filter_sweep.py
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

OUTPUT_FILE = ROOT / "ai_context" / "mym_renko_sha_filter_results.json"

# Locked base config (winner from previous sweep)
BASE = {"fast_len": 12, "slow_len": 14, "cooldown": 20}

PARAM_GRID = {
    "min_slow_streak":  [0, 2, 3, 5, 8, 12, 15, 20],
    "min_brick_streak": [0, 1, 2, 3, 4, 5],  # N of last N bricks agree
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


def generate_signals(df, fast_len, slow_len, cooldown, min_slow_streak, min_brick_streak):
    o = df["Open"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    c = df["Close"].values.astype(float)
    brick_up = df["brick_up"].values
    n = len(df)

    f_bull = _smoothed_ha(o, h, l, c, fast_len)
    s_bull = _smoothed_ha(o, h, l, c, slow_len)

    # Slow streak counts
    slow_streaks = np.zeros(n, dtype=int)
    count = 1
    for i in range(1, n):
        count = count + 1 if s_bull[i] == s_bull[i - 1] else 1
        slow_streaks[i] = count

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    warmup = max(fast_len, slow_len, min_brick_streak) + 1
    last_exit_bar = -999_999
    pos = 0

    for i in range(warmup, n):
        fast_bull_flip = f_bull[i] and not f_bull[i - 1]
        fast_bear_flip = not f_bull[i] and f_bull[i - 1]

        # Brick exit
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
            streak_ok = (min_slow_streak == 0) or (slow_streaks[i] >= min_slow_streak)
            if not streak_ok:
                continue

            # Brick streak filter: last N bricks must all match direction
            brick_ok_long  = True
            brick_ok_short = True
            if min_brick_streak > 0:
                last_bricks = brick_up[i - min_brick_streak:i]
                brick_ok_long  = bool(np.all(last_bricks))
                brick_ok_short = bool(not np.any(last_bricks))

            if fast_bull_flip and s_bull[i] and brick_ok_long:
                long_entry[i] = True
                pos = 1
            elif fast_bear_flip and not s_bull[i] and brick_ok_short:
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
    full_params = {**BASE, **params}
    df = generate_signals(df, **full_params)

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
    print(f"  {len(combos)} filter combos")
    print(f"  Base config: {BASE}")

    results = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run_single, p, df_pickle): p for p in combos}
        for future in as_completed(futures):
            r = future.result()
            if r is not None:
                results.append(r)

    elapsed = time.time() - t0
    print(f"  {len(results)} viable configs in {elapsed:.1f}s\n")

    results.sort(key=lambda r: r["profit_factor"], reverse=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump({"base": BASE, "results": results}, f, indent=2, default=str)

    print(f"{'='*100}")
    print(f"  TOP 30 — Filter Combos (base: f=12, s=14, cd=20, brick_flip)")
    print(f"{'='*100}")
    print(f"  {'SlowStrk':>9} {'BrkStrk':>8} | {'PF':>7} {'T':>5} {'WR%':>5} {'Net$':>10} {'AvgT':>8} {'DD':>9} {'W/L':>5}")
    print(f"  {'-'*20} | {'-'*55}")

    for r in results[:30]:
        p = r["params"]
        print(f"  {p['min_slow_streak']:>9} {p['min_brick_streak']:>8} | "
              f"{r['profit_factor']:>7.2f} {r['total_trades']:>5} {r['win_rate']:>5.1f} "
              f"{r['net_profit']:>10.2f} {r['avg_trade']:>8.2f} {r['max_drawdown']:>9.2f} "
              f"{r['avg_win_loss_ratio']:>5.2f}")


if __name__ == "__main__":
    main()
