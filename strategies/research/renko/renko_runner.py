"""
Renko research runner — sweeps all instruments × brick sizes × param combos.

Each strategy module must export:
  DESCRIPTION  str
  HYPOTHESIS   str
  PARAM_GRID   dict  — {param_name: [values, ...]}
  generate_signals(df, **params) -> pd.DataFrame  (single-frame, no MTF)

Usage:
  python renko_runner.py r001_donchian_trend              # all instruments
  python renko_runner.py r002_ema_adx --instrument EURUSD # one instrument
  python renko_runner.py r006_supertrend_adx --top 10     # show top 10
"""

import argparse
import contextlib
import importlib
import io
import itertools
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from engine import BacktestConfig, load_tv_export, run_backtest_long_short

MIN_TRADES = 30  # Renko has fewer bars than 5m data; lower threshold

# Instrument configs: qty sizing & commission matching OANDA micro lots
# commission_pct = spread cost as % of price (approx)
INSTRUMENTS = {
    "EURUSD": {"commission_pct": 0.0043, "qty": 1000.0},
    "GBPUSD": {"commission_pct": 0.0043, "qty": 1000.0},
    "GBPJPY": {"commission_pct": 0.0043, "qty": 1000.0},
    "USDJPY": {"commission_pct": 0.0043, "qty": 1000.0},
    "EURAUD": {"commission_pct": 0.0050, "qty": 1000.0},
    "BTCUSD": {"commission_pct": 0.10,   "qty": 0.01},
}


def discover_renko_files():
    """Find all Renko CSV files grouped by instrument."""
    data_dir = ROOT / "data"
    files = {}
    for f in sorted(data_dir.glob("OANDA_*renko*.csv")):
        name = f.name
        # Extract instrument: OANDA_EURUSD, 1S renko 0.0005.csv -> EURUSD
        parts = name.split(",")
        if len(parts) < 2:
            continue
        inst = parts[0].replace("OANDA_", "")
        # Handle BTCUSD.SPOT.US -> BTCUSD
        if "BTCUSD" in inst:
            inst = "BTCUSD"
        # Extract brick size from filename
        brick = name.split("renko ")[-1].replace(".csv", "")
        if inst not in files:
            files[inst] = []
        files[inst].append({"path": f.name, "brick": brick, "bars": None})
    return files


def run_single(df, params, generate_signals, instrument):
    cfg_info = INSTRUMENTS.get(instrument, INSTRUMENTS["EURUSD"])
    df_sig = generate_signals(df.copy(), **params)
    cfg = BacktestConfig(
        initial_capital=1000.0,
        commission_pct=cfg_info["commission_pct"],
        slippage_ticks=0,
        qty_type="fixed",
        qty_value=cfg_info["qty"],
        pyramiding=1,
        start_date="2018-01-01",
        end_date="2069-12-31",
        take_profit_pct=0.0,
        stop_loss_pct=0.0,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest_long_short(df_sig, cfg)
    pf = kpis.get("profit_factor", 0.0) or 0.0
    return {
        "pf":     float("inf") if math.isinf(pf) else float(pf),
        "net":    float(kpis.get("net_profit", 0.0) or 0.0),
        "trades": int(kpis.get("total_trades", 0) or 0),
        "wr":     float(kpis.get("win_rate", 0.0) or 0.0),
        "dd":     float(kpis.get("max_drawdown_pct", 0.0) or 0.0),
        "exp":    float(kpis.get("avg_trade", 0.0) or 0.0),
        "params": params,
    }


def fmt_pf(pf):
    return "INF" if math.isinf(pf) else f"{pf:.3f}"


def rank_key(r):
    qualifies = r["trades"] >= MIN_TRADES
    pf_score = r["pf"] if not math.isinf(r["pf"]) else 1e12
    return (qualifies, pf_score, r["net"])


def sweep_instrument(mod, df, instrument, brick, verbose=True):
    """Run param sweep on a single instrument/brick combo. Returns list of results."""
    grid = mod.PARAM_GRID
    keys = list(grid.keys())
    combos = [dict(zip(keys, v)) for v in itertools.product(*grid.values())]

    results = []
    for i, params in enumerate(combos, 1):
        r = run_single(df, params, mod.generate_signals, instrument)
        r["instrument"] = instrument
        r["brick"] = brick
        results.append(r)
        if verbose:
            print(f"    [{i:>3}/{len(combos)}] PF={fmt_pf(r['pf']):>6} "
                  f"Net={r['net']:>8.2f} T={r['trades']:>4} "
                  f"WR={r['wr']:>5.1f}% DD={r['dd']:>5.2f}% | {params}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("strategy", help="e.g. r001_donchian_trend")
    parser.add_argument("--instrument", "-i", default=None, help="Filter to one instrument")
    parser.add_argument("--top", type=int, default=10, help="Show top N results")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    mod = importlib.import_module(args.strategy)

    grid = mod.PARAM_GRID
    keys = list(grid.keys())
    n_combos = 1
    for v in grid.values():
        n_combos *= len(v)

    renko_files = discover_renko_files()
    if args.instrument:
        key = args.instrument.upper()
        renko_files = {k: v for k, v in renko_files.items() if k == key}

    total_files = sum(len(v) for v in renko_files.values())

    print(f"\n{'='*70}")
    print(f"  RENKO RESEARCH SWEEP: {args.strategy}")
    print(f"  {mod.DESCRIPTION}")
    print(f"  Instruments: {len(renko_files)} | Files: {total_files} | "
          f"Combos/file: {n_combos} | Total runs: {total_files * n_combos}")
    print(f"{'='*70}")

    all_results = []

    for inst, file_list in sorted(renko_files.items()):
        for finfo in file_list:
            fname = finfo["path"]
            brick = finfo["brick"]
            print(f"\n  --- {inst} brick={brick} ({fname}) ---")
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    df = load_tv_export(fname)
                bars = len(df)
            except Exception as e:
                print(f"    SKIP: {e}")
                continue
            print(f"    Bars: {bars}")
            results = sweep_instrument(mod, df, inst, brick, verbose=not args.quiet)
            all_results.extend(results)

    # Global ranking
    all_results.sort(key=rank_key, reverse=True)

    print(f"\n{'='*70}")
    print(f"  TOP {args.top} ACROSS ALL INSTRUMENTS")
    print(f"{'='*70}")
    print(f"  {'Rank':>4}  {'Inst':<8} {'Brick':<8} {'PF':>7} {'Net':>9} "
          f"{'Trades':>6} {'WR%':>6} {'DD%':>6}  Params")
    print(f"  {'-'*90}")

    for rank, r in enumerate(all_results[:args.top], 1):
        tag = "OK" if r["trades"] >= MIN_TRADES else "LT"
        print(f"  {rank:>3}. [{tag}] {r['instrument']:<8} {r['brick']:<8} "
              f"PF={fmt_pf(r['pf']):>6} Net={r['net']:>8.2f} "
              f"T={r['trades']:>4} WR={r['wr']:>5.1f}% DD={r['dd']:>5.2f}%  "
              f"{r['params']}")

    # Per-instrument best
    print(f"\n  BEST PER INSTRUMENT:")
    print(f"  {'-'*90}")
    seen = set()
    for r in all_results:
        key = r["instrument"]
        if key in seen:
            continue
        if r["trades"] < MIN_TRADES:
            continue
        seen.add(key)
        print(f"  {key:<8} brick={r['brick']:<8} PF={fmt_pf(r['pf']):>6} "
              f"Net={r['net']:>8.2f} T={r['trades']:>4} WR={r['wr']:>5.1f}% "
              f"DD={r['dd']:>5.2f}%  {r['params']}")

    return all_results


if __name__ == "__main__":
    main()
