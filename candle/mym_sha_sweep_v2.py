"""
MYM Dual SHA -- Stage 2 Sweep

Takes top base configs from Stage 1 and layers:
- Slow streak filter: [0, 3, 5, 8, 12]
- Session filter: [none, rth, rth_skip15]
- Exit mode: [sha_flip, atr_only, sha_or_atr]
- ATR stop/target multipliers

Usage:
    python candle/mym_sha_sweep_v2.py
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
from candle.strategies.mym_dual_sha_v2 import generate_signals
from renko.config import MAX_WORKERS

# -- Config --------------------------------------------------------------------
CANDLE_FILE = "CBOT_MINI_MYM1!, 1.csv"
INSTRUMENT_DIR = "MYM"

# MYM futures: $0.90/side commission
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
OUTPUT_FILE = ROOT / "ai_context" / "mym_sha_sweep_v2_results.json"

# Top base configs from Stage 1
BASE_CONFIGS = [
    {"fast_len": 3,  "slow_len": 14, "cooldown": 30},
    {"fast_len": 8,  "slow_len": 50, "cooldown": 30},
    {"fast_len": 3,  "slow_len": 25, "cooldown": 30},
    {"fast_len": 8,  "slow_len": 40, "cooldown": 30},
    {"fast_len": 3,  "slow_len": 18, "cooldown": 30},
    {"fast_len": 3,  "slow_len": 22, "cooldown": 30},
]

# Stage 2 parameters to layer
STAGE2_GRID = {
    "min_slow_streak": [0, 3, 5, 8, 12],
    "session_mode": ["none", "rth", "rth_skip15"],
    "exit_mode": ["sha_flip", "atr_only", "sha_or_atr"],
}

# ATR configs per exit mode
ATR_CONFIGS = {
    "sha_flip":   [{"atr_sl_mult": 0, "atr_tp_mult": 0}],  # no ATR exits
    "atr_only":   [
        {"atr_sl_mult": 1.0, "atr_tp_mult": 2.0},
        {"atr_sl_mult": 1.5, "atr_tp_mult": 3.0},
        {"atr_sl_mult": 2.0, "atr_tp_mult": 4.0},
        {"atr_sl_mult": 1.0, "atr_tp_mult": 3.0},
        {"atr_sl_mult": 1.5, "atr_tp_mult": 4.0},
    ],
    "sha_or_atr": [
        {"atr_sl_mult": 1.5, "atr_tp_mult": 3.0},
        {"atr_sl_mult": 2.0, "atr_tp_mult": 4.0},
        {"atr_sl_mult": 2.5, "atr_tp_mult": 5.0},
    ],
}


def _run_single(params: dict, df_pickle: bytes) -> dict | None:
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
    """Build all parameter combos: base x stage2 x atr configs."""
    combos = []
    for base in BASE_CONFIGS:
        for streak in STAGE2_GRID["min_slow_streak"]:
            for session in STAGE2_GRID["session_mode"]:
                for exit_mode in STAGE2_GRID["exit_mode"]:
                    for atr_cfg in ATR_CONFIGS[exit_mode]:
                        combo = {
                            **base,
                            "min_slow_streak": streak,
                            "session_mode": session,
                            "exit_mode": exit_mode,
                            **atr_cfg,
                        }
                        combos.append(combo)
    return combos


def main():
    import pickle

    print(f"Loading {CANDLE_FILE}...")
    df = load_candle_csv(CANDLE_FILE, INSTRUMENT_DIR)
    print(f"  {len(df)} bars, {df.index[0]} to {df.index[-1]}")

    df_pickle = pickle.dumps(df)

    combos = _build_combos()
    print(f"  {len(combos)} parameter combos")
    print(f"  {len(BASE_CONFIGS)} base configs x streak/session/exit layers")
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
            if done % 100 == 0 or done == len(combos):
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
            "sweep": "mym_dual_sha_stage2",
            "data": CANDLE_FILE,
            "bars": len(df),
            "date_range": f"{df.index[0]} to {df.index[-1]}",
            "total_combos": len(combos),
            "configs_with_trades": len(results),
            "min_trades": MIN_TRADES,
            "results": results,
        }, f, indent=2, default=str)
    print(f"  Saved to {OUTPUT_FILE}")

    # Print top 25
    print(f"\n{'='*120}")
    print(f"  TOP 25 by Profit Factor (min {MIN_TRADES} trades)")
    print(f"{'='*120}")
    print(f"  {'F':>2} {'S':>2} {'CD':>2} {'Strk':>4} {'Session':>10} {'Exit':>10} {'SL':>4} {'TP':>4} | "
          f"{'PF':>6} {'T':>4} {'WR%':>5} {'Net$':>8} {'AvgT':>7} {'DD':>8} {'W/L':>5} {'L/S':>6}")
    print(f"  {'-'*75} | {'-'*60}")

    for r in results[:25]:
        p = r["params"]
        sl = f"{p.get('atr_sl_mult', 0):.1f}" if p.get('atr_sl_mult', 0) > 0 else "  -"
        tp = f"{p.get('atr_tp_mult', 0):.1f}" if p.get('atr_tp_mult', 0) > 0 else "  -"
        ls = f"{r['long_trades']}/{r['short_trades']}"
        print(f"  {p['fast_len']:>2} {p['slow_len']:>2} {p['cooldown']:>2} {p['min_slow_streak']:>4} "
              f"{p['session_mode']:>10} {p['exit_mode']:>10} {sl:>4} {tp:>4} | "
              f"{r['profit_factor']:>6.2f} {r['total_trades']:>4} {r['win_rate']:>5.1f} "
              f"{r['net_profit']:>8.2f} {r['avg_trade']:>7.2f} {r['max_drawdown']:>8.2f} "
              f"{r['avg_win_loss_ratio']:>5.2f} {ls:>6}")


if __name__ == "__main__":
    main()
