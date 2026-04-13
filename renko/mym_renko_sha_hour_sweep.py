"""
MYM Renko SHA — Hour Filter Sweep

Tests blocking various combinations of weak hours.
Base config: f=12, s=14, cd=20, brick_streak=5, exit_tolerance=1

From TV trade analysis:
  Hour 05: 11T, 45.5% WR, $174  (weak — filter candidate)
  Hour 13: 2T, 50% WR, -$3.80   (weak — filter candidate)
  Hour 16: 2T, 50% WR, $3.70    (weak — filter candidate)
  Hour 22: 4T, 50% WR, $187     (questionable)

Strong hours to keep:
  Hour 07: 37T, 78.4% WR, $2,262
  Hour 08: 19T, 94.7% WR, $1,981
  Hour 10: 15T, 86.7% WR, $872

Usage:
    python renko/mym_renko_sha_hour_sweep.py
"""

import contextlib
import io
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from itertools import combinations

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

OUTPUT_FILE = ROOT / "ai_context" / "mym_renko_sha_hour_results.json"

BASE = {
    "fast_len": 12,
    "slow_len": 14,
    "cooldown": 20,
    "min_brick_streak": 5,
}

# Weak hours from TV analysis (UTC, since data index is naive UTC)
# Note: TV ET hours = data UTC hours - 4 (EDT) or -5 (EST). Data is in UTC.
# Hour 05 ET = 09 UTC, Hour 13 ET = 17 UTC, Hour 16 ET = 20 UTC, Hour 22 ET = 02 UTC
# Easier to compute ET from index inside the loop.

# Block sets to test (hours in ET)
BLOCK_SETS = [
    [],                    # baseline
    [5],                   # block weakest single
    [13],
    [16],
    [22],
    [5, 13],
    [5, 13, 16],
    [5, 13, 16, 22],
    [13, 16],
    [13, 16, 22],
    # Time windows
    [0, 1, 2, 3, 4, 5],    # block all overnight (00-05 ET)
    [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],  # block all afternoon/evening
]


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


def generate_signals(df, fast_len, slow_len, cooldown, min_brick_streak, blocked_et_hours):
    o = df["Open"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    c = df["Close"].values.astype(float)
    brick_up = df["brick_up"].values
    n = len(df)

    # Pre-compute ET hour for each bar (UTC index, EDT = UTC-4)
    # For simplicity assume EDT year-round (close enough for sweep purposes).
    et_hours = np.array([(t.hour - 4) % 24 for t in df.index])
    blocked = np.array([h in blocked_et_hours for h in et_hours])

    f_bull = _smoothed_ha(o, h, l, c, fast_len)
    s_bull = _smoothed_ha(o, h, l, c, slow_len)

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

        # Brick exit (tolerance=1)
        if pos == 1 and not brick_up[i]:
            long_exit[i] = True
            pos = 0
            last_exit_bar = i
        elif pos == -1 and brick_up[i]:
            short_exit[i] = True
            pos = 0
            last_exit_bar = i

        # Entry (block weak hours)
        if pos == 0 and (i - last_exit_bar) >= cooldown and not blocked[i]:
            if min_brick_streak > 0:
                last_n = brick_up[i - min_brick_streak:i]
                brk_long_ok  = bool(np.all(last_n))
                brk_short_ok = bool(not np.any(last_n))
            else:
                brk_long_ok = brk_short_ok = True

            if fast_bull_flip and s_bull[i] and brk_long_ok:
                long_entry[i] = True
                pos = 1
            elif fast_bear_flip and not s_bull[i] and brk_short_ok:
                short_entry[i] = True
                pos = -1

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df


def _run_single(blocked_et_hours, df_pickle):
    import pickle
    df = pickle.loads(df_pickle)
    df = generate_signals(df, **BASE, blocked_et_hours=blocked_et_hours)

    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest_long_short(df, BACKTEST_CONFIG)

    total_trades = kpis.get("total_trades", 0)
    if total_trades < 5:
        return None

    return {
        "blocked_hours": list(blocked_et_hours),
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

    print(f"  {len(BLOCK_SETS)} block sets to test")

    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run_single, bs, df_pickle): bs for bs in BLOCK_SETS}
        for future in as_completed(futures):
            r = future.result()
            if r is not None:
                results.append(r)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s\n")

    results.sort(key=lambda r: r["profit_factor"], reverse=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump({"base": BASE, "results": results}, f, indent=2, default=str)

    print(f"{'='*110}")
    print(f"  HOUR FILTER SWEEP — sorted by PF")
    print(f"{'='*110}")
    print(f"  {'Blocked ET Hours':>40} | {'PF':>7} {'T':>5} {'WR%':>5} {'Net$':>10} {'AvgT':>8} {'DD':>9} {'W/L':>5}")
    print(f"  {'-'*40} | {'-'*55}")

    for r in results:
        bh = ",".join(str(h) for h in r["blocked_hours"]) or "(none)"
        if len(bh) > 40:
            bh = bh[:37] + "..."
        print(f"  {bh:>40} | "
              f"{r['profit_factor']:>7.2f} {r['total_trades']:>5} {r['win_rate']:>5.1f} "
              f"{r['net_profit']:>10.2f} {r['avg_trade']:>8.2f} {r['max_drawdown']:>9.2f} "
              f"{r['avg_win_loss_ratio']:>5.2f}")


if __name__ == "__main__":
    main()
