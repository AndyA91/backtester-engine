"""
Renko research strategy runner.

Each strategy module must export:
  DESCRIPTION  str   — one-line summary
  HYPOTHESIS   str   — what edge we expect and why
  PARAM_GRID   dict  — {param_name: [values, ...]} for the sweep
  generate_signals(df, **params) -> pd.DataFrame
    Returns df with added columns: long_entry, long_exit, short_entry, short_exit (bool).
    NOTE: single df only — no HTF frames (Renko has no natural HTF).

IS/OOS split:
  IS:  2023-01-23 → 2025-09-30  (~2y 9m)
  OOS: 2025-10-01 → 2026-03-05  (sealed until a winner is confirmed)

Usage:
  cd renko
  python runner.py r001_brick_count
  python runner.py r001_brick_count --start 2023-01-23 --end 2025-09-30
"""

import argparse
import contextlib
import importlib
import io
import itertools
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import BacktestConfig, run_backtest_long_short
from renko.data import load_renko_export
from renko.indicators import add_renko_indicators

RENKO_FILE = "OANDA_EURUSD, 1S renko 0.0004.csv"

IS_START = "2023-01-23"
IS_END   = "2025-09-30"
MIN_TRADES_FOR_RANK = 60


def load_data(renko_file=None):
    df = load_renko_export(renko_file or RENKO_FILE)
    add_renko_indicators(df)
    return df


def run_single(df, generate_signals, params, start, end,
               commission_pct=0.0046, initial_capital=1000.0):
    df_sig = generate_signals(df.copy(), **params)
    cfg = BacktestConfig(
        initial_capital=initial_capital,
        commission_pct=commission_pct,
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
        "pf":         float("inf") if math.isinf(pf) else float(pf),
        "net":        float(kpis.get("net_profit", 0.0) or 0.0),
        "trades":     int(kpis.get("total_trades", 0) or 0),
        "win_rate":   float(kpis.get("win_rate", 0.0) or 0.0),
        "max_dd_pct": float(kpis.get("max_drawdown_pct", 0.0) or 0.0),
        "expectancy": float(kpis.get("avg_trade", 0.0) or 0.0),
        "avg_wl":     float(kpis.get("avg_win_loss_ratio", 0.0) or 0.0),
        "params":     params,
    }


def fmt_pf(pf):
    return "INF" if math.isinf(pf) else f"{pf:.4f}"


def rank_key(result):
    qualifies = result["trades"] >= MIN_TRADES_FOR_RANK
    pf_score = result["pf"] if not math.isinf(result["pf"]) else 1e12
    return (qualifies, pf_score, result["net"])


def sweep(strategy_name, start=IS_START, end=IS_END, verbose=True, renko_file=None):
    global _renko_file
    _renko_file = renko_file
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

    commission_pct   = getattr(mod, "COMMISSION_PCT",   0.0046)
    initial_capital  = getattr(mod, "INITIAL_CAPITAL",  1000.0)

    if verbose and (commission_pct != 0.0046 or initial_capital != 1000.0):
        print(f"Commission: {commission_pct:.4f}%  |  Initial capital: {initial_capital:,.0f}")

    print("Loading data...")
    df = load_data(getattr(mod, "RENKO_FILE", None) or _renko_file)

    results = []
    for i, params in enumerate(combos, 1):
        r = run_single(df, mod.generate_signals, params, start, end,
                       commission_pct=commission_pct, initial_capital=initial_capital)
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
    parser.add_argument("strategy", help="Strategy module name (e.g. r001_brick_count)")
    parser.add_argument("--start",  default=IS_START)
    parser.add_argument("--end",    default=IS_END)
    parser.add_argument("--renko",  default=None,
                        help="Renko CSV filename in data/ (default: 0.0004)")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent / "strategies"))
    sweep(args.strategy, args.start, args.end, renko_file=args.renko)
