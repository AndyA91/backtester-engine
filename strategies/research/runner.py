"""
Research strategy runner.

Each strategy module must export:
  DESCRIPTION  str   — one-line summary
  HYPOTHESIS   str   — what edge we expect and why
  PARAM_GRID   dict  — {param_name: [values, ...]} for the sweep
  generate_signals(df_5m, df_1h, df_1d, **params) -> pd.DataFrame
    Returns df_5m with added columns: long_entry, long_exit, short_entry, short_exit (bool).

Usage:
  python runner.py r001_donchian_trend
  python runner.py r001_donchian_trend --start 2024-01-01 --end 2024-12-31
"""

import argparse
import contextlib
import importlib
import io
import itertools
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from engine import BacktestConfig, load_tv_export, run_backtest_long_short

IS_START = "2024-01-01"
IS_END   = "2025-09-30"
MIN_TRADES_FOR_RANK = 60  # <30 trades is statistical noise; 60 gives tighter PF confidence


def load_data():
    df_5m = load_tv_export("HISTDATA_EURUSD_5m.csv")
    df_1h = load_tv_export("HISTDATA_EURUSD_1h.csv")
    df_1d = load_tv_export("HISTDATA_EURUSD_1d.csv")
    return df_5m, df_1h, df_1d


def run_single(df_5m, df_1h, df_1d, generate_signals, params, start, end):
    df_sig = generate_signals(df_5m.copy(), df_1h, df_1d, **params)
    cfg = BacktestConfig(
        initial_capital=1000.0,
        commission_pct=0.0046,  # $0.05/side per 1k units at ~1.10 ≈ $0.10 round-trip (OANDA)
        slippage_ticks=0,
        qty_type="fixed",
        qty_value=1000.0,
        pyramiding=1,
        start_date=start,
        end_date=end,
        take_profit_pct=0.0,
        stop_loss_pct=0.0,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest_long_short(df_sig, cfg)
    pf = kpis.get("profit_factor", 0.0) or 0.0
    return {
        "pf":          float("inf") if math.isinf(pf) else float(pf),
        "net":         float(kpis.get("net_profit", 0.0) or 0.0),
        "trades":      int(kpis.get("total_trades", 0) or 0),
        "win_rate":    float(kpis.get("win_rate", 0.0) or 0.0),
        "max_dd_pct":  float(kpis.get("max_drawdown_pct", 0.0) or 0.0),
        "expectancy":  float(kpis.get("avg_trade", 0.0) or 0.0),
        "avg_wl":      float(kpis.get("avg_win_loss_ratio", 0.0) or 0.0),
        "params":      params,
    }


def fmt_pf(pf):
    return "INF" if math.isinf(pf) else f"{pf:.4f}"


def rank_key(result):
    # Prefer statistically usable samples before raw PF/net ordering.
    qualifies = result["trades"] >= MIN_TRADES_FOR_RANK
    pf_score = result["pf"] if not math.isinf(result["pf"]) else 1e12
    return (qualifies, pf_score, result["net"])


def sweep(strategy_name, start=IS_START, end=IS_END, verbose=True):
    mod = importlib.import_module(strategy_name)
    grid = mod.PARAM_GRID
    keys = list(grid.keys())
    combos = [dict(zip(keys, v)) for v in itertools.product(*grid.values())]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Strategy : {strategy_name}")
        print(f"Desc     : {mod.DESCRIPTION}")
        print(f"Period   : {start} to {end}")
        print(f"Combos   : {len(combos)}")
        print(f"{'='*60}")

    print("Loading data...")
    df_5m, df_1h, df_1d = load_data()

    results = []
    for i, params in enumerate(combos, 1):
        r = run_single(df_5m, df_1h, df_1d, mod.generate_signals, params, start, end)
        results.append(r)
        if verbose:
            print(f"  [{i:>3}/{len(combos)}] PF={fmt_pf(r['pf'])} Net={r['net']:>7.2f} "
                  f"T={r['trades']:>4} WR={r['win_rate']:>5.1f}% "
                  f"DD={r['max_dd_pct']:>5.2f}% Exp={r['expectancy']:>6.3f} | {params}")

    results.sort(key=rank_key, reverse=True)

    print(f"\n--- Top 5 ---")
    for r in results[:5]:
        rank_note = "OK" if r["trades"] >= MIN_TRADES_FOR_RANK else "LOW_T"
        print(f"  [{rank_note}] PF={fmt_pf(r['pf'])} Net={r['net']:>7.2f} T={r['trades']:>4} "
              f"WR={r['win_rate']:>5.1f}% DD={r['max_dd_pct']:>5.2f}% "
              f"Exp={r['expectancy']:>6.3f} | {r['params']}")

    return results, mod


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("strategy", help="Strategy module name (e.g. r001_donchian_trend)")
    parser.add_argument("--start", default=IS_START)
    parser.add_argument("--end", default=IS_END)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sweep(args.strategy, args.start, args.end)
