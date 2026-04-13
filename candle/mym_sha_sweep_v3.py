"""
MYM Dual SHA -- Stage 3: Multi-Timeframe Sweep

Sweeps 1m, 3m, 5m, 10m, 15m, 30m timeframes with RMA ATR.
Uses v3 signal generator (fixes Pine ATR mismatch).

Usage:
    python candle/mym_sha_sweep_v3.py
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
from candle.data import load_candle_csv, resample_ohlc
from candle.strategies.mym_dual_sha_v3 import generate_signals
from renko.config import MAX_WORKERS

# -- Config --------------------------------------------------------------------
CANDLE_FILE = "CBOT_MINI_MYM1!, 1.csv"
INSTRUMENT_DIR = "MYM"

BACKTEST_CONFIG = BacktestConfig(
    initial_capital=10000.0,
    commission_pct=0.002,
    qty_type="fixed",
    qty_value=1,
    start_date="2000-01-01",
    end_date="2099-12-31",
    process_orders_on_close=False,
)

MIN_TRADES = 5
OUTPUT_FILE = ROOT / "ai_context" / "mym_sha_sweep_v3_results.json"

# Timeframes to test
TIMEFRAMES = [1, 3, 5, 10, 15, 30]

# Parameter grid per timeframe
PARAM_GRID = {
    "fast_len":         [3, 5, 8, 12],
    "slow_len":         [14, 22, 30, 50],
    "cooldown":         [5, 15, 30],
    "min_slow_streak":  [0, 5, 12],
    "session_mode":     ["rth"],
    "exit_mode":        ["atr_only"],
    "atr_sl_mult":      [1.0, 1.5, 2.0],
    "atr_tp_mult":      [2.0, 3.0, 4.0],
}


def _run_single(params: dict, df_pickle: bytes, tf: int) -> dict | None:
    """Run a single backtest config."""
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
        "tf": tf,
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


def _build_combos():
    """Build all parameter combos, filtering fast < slow."""
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    combos = [c for c in combos if c["fast_len"] < c["slow_len"]]
    return combos


def main():
    import pickle

    print(f"Loading {CANDLE_FILE}...")
    df_1m = load_candle_csv(CANDLE_FILE, INSTRUMENT_DIR)
    print(f"  {len(df_1m)} bars (1m), {df_1m.index[0]} to {df_1m.index[-1]}")

    base_combos = _build_combos()
    print(f"  {len(base_combos)} param combos per timeframe")
    print(f"  {len(TIMEFRAMES)} timeframes: {TIMEFRAMES}")
    print(f"  Total runs: {len(base_combos) * len(TIMEFRAMES)}")
    print(f"  Workers: {MAX_WORKERS}")

    all_results = []
    t0 = time.time()

    for tf in TIMEFRAMES:
        df_tf = resample_ohlc(df_1m, tf) if tf > 1 else df_1m.copy()
        df_pickle = pickle.dumps(df_tf)
        print(f"\n  TF={tf}m: {len(df_tf)} bars")

        results_tf = []
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(_run_single, params, df_pickle, tf): params
                for params in base_combos
            }

            done = 0
            for future in as_completed(futures):
                done += 1
                if done % 100 == 0 or done == len(base_combos):
                    elapsed = time.time() - t0
                    print(f"    [{done}/{len(base_combos)}] {elapsed:.1f}s")

                result = future.result()
                if result is not None:
                    results_tf.append(result)

        all_results.extend(results_tf)
        # Quick summary per TF
        if results_tf:
            best = max(results_tf, key=lambda r: r["profit_factor"])
            print(f"    {len(results_tf)} viable configs. Best PF={best['profit_factor']:.2f} "
                  f"({best['total_trades']}T, {best['win_rate']}% WR)")
        else:
            print(f"    No configs with >= {MIN_TRADES} trades")

    elapsed = time.time() - t0
    print(f"\nCompleted {len(base_combos) * len(TIMEFRAMES)} runs in {elapsed:.1f}s")
    print(f"  {len(all_results)} total configs with >= {MIN_TRADES} trades")

    all_results.sort(key=lambda r: r["profit_factor"], reverse=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump({
            "sweep": "mym_dual_sha_stage3_multi_tf",
            "data": CANDLE_FILE,
            "timeframes": TIMEFRAMES,
            "combos_per_tf": len(base_combos),
            "total_viable": len(all_results),
            "min_trades": MIN_TRADES,
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"  Saved to {OUTPUT_FILE}")

    # Print top 30
    print(f"\n{'='*125}")
    print(f"  TOP 30 by Profit Factor (min {MIN_TRADES} trades)")
    print(f"{'='*125}")
    print(f"  {'TF':>3} {'F':>2} {'S':>2} {'CD':>2} {'Strk':>4} {'SL':>4} {'TP':>4} | "
          f"{'PF':>7} {'T':>4} {'WR%':>5} {'Net$':>8} {'AvgT':>7} {'DD':>8} {'W/L':>5} {'L/S':>6}")
    print(f"  {'-'*30} | {'-'*65}")

    for r in all_results[:30]:
        p = r["params"]
        ls = f"{r['long_trades']}/{r['short_trades']}"
        print(f"  {r['tf']:>2}m {p['fast_len']:>2} {p['slow_len']:>2} {p['cooldown']:>2} "
              f"{p['min_slow_streak']:>4} {p['atr_sl_mult']:>4.1f} {p['atr_tp_mult']:>4.1f} | "
              f"{r['profit_factor']:>7.2f} {r['total_trades']:>4} {r['win_rate']:>5.1f} "
              f"{r['net_profit']:>8.2f} {r['avg_trade']:>7.2f} {r['max_drawdown']:>8.2f} "
              f"{r['avg_win_loss_ratio']:>5.2f} {ls:>6}")

    # Summary per TF
    print(f"\n  BEST PER TIMEFRAME:")
    for tf in TIMEFRAMES:
        tf_results = [r for r in all_results if r["tf"] == tf]
        if tf_results:
            best = tf_results[0]  # already sorted by PF
            p = best["params"]
            print(f"    {tf:>2}m: PF={best['profit_factor']:.2f}, {best['total_trades']}T, "
                  f"WR={best['win_rate']}%, Net=${best['net_profit']:.0f}, "
                  f"f={p['fast_len']} s={p['slow_len']} cd={p['cooldown']} "
                  f"strk={p['min_slow_streak']} sl={p['atr_sl_mult']} tp={p['atr_tp_mult']}")
        else:
            print(f"    {tf:>2}m: No viable configs")


if __name__ == "__main__":
    main()
