"""
MYM Dual Smoothed HA -- Stage 1 Sweep

Sweeps fast_len, slow_len, cooldown on MYM 1-min candle data.
Uses the same engine (run_backtest_long_short) as Renko strategies.

Commission: ~0.0013% per side approximates flat $0.62/side on $46K notional.
Engine PnL is in "points" (multiply by $0.50 for real MYM dollars).

Usage:
    python candle/mym_sha_sweep.py
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
from candle.data import load_candle_csv
from candle.strategies.mym_dual_sha import PARAM_GRID, generate_signals
from renko.config import MAX_WORKERS

# -- Config --------------------------------------------------------------------
CANDLE_FILE = "CBOT_MINI_MYM1!, 1.csv"
INSTRUMENT_DIR = "MYM"

# MYM futures: flat commission ~$0.90/side. On ~$46K notional = ~0.002%/side.
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
OUTPUT_FILE = ROOT / "ai_context" / "mym_sha_sweep_results.json"


def _load_data():
    return load_candle_csv(CANDLE_FILE, INSTRUMENT_DIR)


def _run_single(params: dict, df_pickle: bytes) -> dict | None:
    """Run a single backtest config. Called in worker process."""
    import pickle
    df = pickle.loads(df_pickle)

    df = generate_signals(df, **params)

    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest_long_short(df, BACKTEST_CONFIG)

    total_trades = kpis.get("total_trades", 0)
    if total_trades < MIN_TRADES:
        return None

    # Count long/short from trades list
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

    print(f"Loading {CANDLE_FILE}...")
    df = _load_data()
    print(f"  {len(df)} bars, {df.index[0]} to {df.index[-1]}")

    df_pickle = pickle.dumps(df)

    # Build param combos
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Filter: fast_len must be < slow_len
    combos = [c for c in combos if c["fast_len"] < c["slow_len"]]
    print(f"  {len(combos)} parameter combos (fast < slow filter applied)")
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
            if done % 50 == 0 or done == len(combos):
                elapsed = time.time() - t0
                print(f"  [{done}/{len(combos)}] {elapsed:.1f}s")

            result = future.result()
            if result is not None:
                results.append(result)

    elapsed = time.time() - t0
    print(f"\nCompleted {len(combos)} runs in {elapsed:.1f}s")
    print(f"  {len(results)} configs with >= {MIN_TRADES} trades")

    # Sort by profit factor
    results.sort(key=lambda r: r["profit_factor"], reverse=True)

    # Save results
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump({
            "sweep": "mym_dual_sha_stage1",
            "data": CANDLE_FILE,
            "bars": len(df),
            "date_range": f"{df.index[0]} to {df.index[-1]}",
            "total_combos": len(combos),
            "configs_with_trades": len(results),
            "min_trades": MIN_TRADES,
            "results": results,
        }, f, indent=2, default=str)
    print(f"  Saved to {OUTPUT_FILE}")

    # Print top 20
    print(f"\n{'='*95}")
    print(f"  TOP 20 by Profit Factor (min {MIN_TRADES} trades)")
    print(f"{'='*95}")
    print(f"  {'Fast':>4} {'Slow':>4} {'CD':>3} | {'PF':>7} {'Trades':>6} {'WR%':>5} {'Net$':>8} {'AvgT':>8} {'MaxDD':>8} {'W/L':>5} {'L/S':>7}")
    print(f"  {'-'*4} {'-'*4} {'-'*3} | {'-'*7} {'-'*6} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*5} {'-'*7}")

    for r in results[:20]:
        p = r["params"]
        ls = f"{r['long_trades']}/{r['short_trades']}"
        print(f"  {p['fast_len']:>4} {p['slow_len']:>4} {p['cooldown']:>3} | "
              f"{r['profit_factor']:>7.2f} {r['total_trades']:>6} {r['win_rate']:>5.1f} "
              f"{r['net_profit']:>8.2f} {r['avg_trade']:>8.2f} {r['max_drawdown']:>8.2f} "
              f"{r['avg_win_loss_ratio']:>5.2f} {ls:>7}")


if __name__ == "__main__":
    main()
