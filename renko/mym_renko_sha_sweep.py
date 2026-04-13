"""
MYM Renko Dual SHA Sweep — Brick size 15 only

Usage:
    python renko/mym_renko_sha_sweep.py
"""

import contextlib
import io
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import BacktestConfig, run_backtest_long_short
from renko.data import load_renko_export
from renko.strategies.mym_renko_sha import generate_signals
from renko.config import MAX_WORKERS

RENKO_FILE = "CBOT_MINI_MYM1!, 1S renko 15.csv"

BACKTEST_CONFIG = BacktestConfig(
    initial_capital=10000.0,
    commission_pct=0.002,
    qty_type="fixed",
    qty_value=1,
    start_date="2000-01-01",
    end_date="2099-12-31",
    process_orders_on_close=False,
)

MIN_TRADES = 10
OUTPUT_FILE = ROOT / "ai_context" / "mym_renko_sha_sweep_results.json"

PARAM_GRID = {
    "fast_len":        [3, 5, 8, 10, 12, 15],
    "slow_len":        [14, 20, 25, 30, 40, 50],
    "cooldown":        [0, 3, 5, 10, 15, 20],
    "min_slow_streak": [0, 3, 5, 8, 12],
    "exit_mode":       ["brick_flip", "sha_flip", "both"],
}


def _run_single(params: dict, df_pickle: bytes) -> dict | None:
    import pickle
    df = pickle.loads(df_pickle)
    df = generate_signals(df, **params)

    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest_long_short(df, BACKTEST_CONFIG)

    total_trades = kpis.get("total_trades", 0)
    if total_trades < MIN_TRADES:
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
    combos = [c for c in combos if c["fast_len"] < c["slow_len"]]
    print(f"  {len(combos)} parameter combos")
    print(f"  Workers: {MAX_WORKERS}")

    results = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_run_single, params, df_pickle): params
            for params in combos
        }

        done = 0
        for future in as_completed(futures):
            done += 1
            if done % 200 == 0 or done == len(combos):
                elapsed = time.time() - t0
                print(f"  [{done}/{len(combos)}] {elapsed:.1f}s")

            result = future.result()
            if result is not None:
                results.append(result)

    elapsed = time.time() - t0
    print(f"\nCompleted {len(combos)} runs in {elapsed:.1f}s")
    print(f"  {len(results)} configs with >= {MIN_TRADES} trades")

    results.sort(key=lambda r: r["profit_factor"], reverse=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump({
            "sweep": "mym_renko_sha_b15",
            "data": RENKO_FILE,
            "bars": len(df),
            "total_combos": len(combos),
            "configs_with_trades": len(results),
            "min_trades": MIN_TRADES,
            "results": results,
        }, f, indent=2, default=str)
    print(f"  Saved to {OUTPUT_FILE}")

    print(f"\n{'='*110}")
    print(f"  TOP 30 by Profit Factor (min {MIN_TRADES} trades)")
    print(f"{'='*110}")
    print(f"  {'F':>2} {'S':>2} {'CD':>2} {'Strk':>4} {'Exit':>10} | "
          f"{'PF':>7} {'T':>4} {'WR%':>5} {'Net$':>8} {'AvgT':>7} {'DD':>8} {'W/L':>5} {'L/S':>6}")
    print(f"  {'-'*30} | {'-'*65}")

    for r in results[:30]:
        p = r["params"]
        ls = f"{r['long_trades']}/{r['short_trades']}"
        print(f"  {p['fast_len']:>2} {p['slow_len']:>2} {p['cooldown']:>2} "
              f"{p['min_slow_streak']:>4} {p['exit_mode']:>10} | "
              f"{r['profit_factor']:>7.2f} {r['total_trades']:>4} {r['win_rate']:>5.1f} "
              f"{r['net_profit']:>8.2f} {r['avg_trade']:>7.2f} {r['max_drawdown']:>8.2f} "
              f"{r['avg_win_loss_ratio']:>5.2f} {ls:>6}")

    # Summary by exit mode
    print(f"\n  BEST PER EXIT MODE:")
    for em in ["brick_flip", "sha_flip", "both"]:
        em_results = [r for r in results if r["params"]["exit_mode"] == em]
        if em_results:
            best = em_results[0]
            p = best["params"]
            print(f"    {em:>10}: PF={best['profit_factor']:.2f}, {best['total_trades']}T, "
                  f"WR={best['win_rate']}%, Net=${best['net_profit']:.0f}, "
                  f"f={p['fast_len']} s={p['slow_len']} cd={p['cooldown']} strk={p['min_slow_streak']}")


if __name__ == "__main__":
    main()
