#!/usr/bin/env python3
"""
phase1_uso_sweep.py -- USO Gate Discovery (Long Only)

Phase 1 sweep for USO stock Renko. Discovers which indicator gates
improve the base N-brick momentum entry signal.

Base signal: N consecutive up bricks -> long entry, first down brick -> exit.
All gates use only the standard 32 pre-shifted indicators (no external exports).

Sweep dimensions:
  1. Brick count (n): [2, 3, 4, 5, 6]                             = 5
  2. Cooldown (cd): [3, 5, 10, 15, 20, 30]                        = 6
  3. ADX threshold: [0, 20, 25, 30, 35, 40]                       = 6
  4. Trend gate: [none, ema_cross, stoch_cross, psar_dir,
                  st_dir, kama_slope_pos, macd_cross]              = 7
  5. Momentum gate: [none, rsi_above_50, bb_pct_b, cmf_positive]  = 4

Total: 5 x 6 x 6 x 7 x 4 = 5,040 combos (10,080 IS+OOS backtests)

Usage:
  python stocks/phase1_uso_sweep.py
"""

import contextlib
import io
import itertools
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stocks.config import MAX_WORKERS

# -- Config -------------------------------------------------------------------

RENKO_FILE = "BATS_USO, 1S renko 0.25.csv"
IS_START   = "2015-07-10"
IS_END     = "2025-09-30"
OOS_START  = "2025-10-01"
OOS_END    = "2026-03-25"
COMMISSION = 0.0
CAPITAL    = 10000.0

# -- Sweep dimensions ---------------------------------------------------------

PARAM_GRID = {
    "n_bricks": [2, 3, 4, 5, 6],
    "cooldown": [3, 5, 10, 15, 20, 30],
}

ADX_THRESHOLDS = [0, 20, 25, 30, 35, 40]   # 0 = no ADX filter

TREND_GATES = [
    "none", "ema_cross", "stoch_cross", "psar_dir",
    "st_dir", "kama_slope_pos", "macd_cross",
]

MOMENTUM_GATES = [
    "none", "rsi_above_50", "bb_pct_b", "cmf_positive",
]


# -- Gate computation (long-only, returns single bool array) -------------------

def _compute_trend_gate(df, name):
    n = len(df)
    if name == "none":
        return np.ones(n, dtype=bool)
    if name == "ema_cross":
        e9, e21 = df["ema9"].values, df["ema21"].values
        m = np.isnan(e9) | np.isnan(e21)
        return m | (e9 > e21)
    if name == "stoch_cross":
        k, d = df["stoch_k"].values, df["stoch_d"].values
        m = np.isnan(k) | np.isnan(d)
        return m | (k > d)
    if name == "psar_dir":
        v = df["psar_dir"].values
        return np.isnan(v) | (v > 0)
    if name == "st_dir":
        v = df["st_dir"].values
        return np.isnan(v) | (v > 0)
    if name == "kama_slope_pos":
        v = df["kama_slope"].values
        return np.isnan(v) | (v > 0)
    if name == "macd_cross":
        m, s = df["macd"].values, df["macd_sig"].values
        nan = np.isnan(m) | np.isnan(s)
        return nan | (m > s)
    raise ValueError(f"Unknown trend gate: {name}")


def _compute_momentum_gate(df, name):
    n = len(df)
    if name == "none":
        return np.ones(n, dtype=bool)
    if name == "rsi_above_50":
        v = df["rsi"].values
        return np.isnan(v) | (v > 50)
    if name == "bb_pct_b":
        v = df["bb_pct_b"].values
        return np.isnan(v) | (v > 0.5)
    if name == "cmf_positive":
        v = df["cmf"].values
        return np.isnan(v) | (v > 0)
    raise ValueError(f"Unknown momentum gate: {name}")


def _compute_all_gates(df):
    gates = {}
    adx = df["adx"].values
    adx_nan = np.isnan(adx)
    for at in ADX_THRESHOLDS:
        if at == 0:
            gates["adx_0"] = np.ones(len(df), dtype=bool)
        else:
            gates[f"adx_{at}"] = adx_nan | (adx >= at)
    for tg in TREND_GATES:
        gates[f"trend:{tg}"] = _compute_trend_gate(df, tg)
    for mg in MOMENTUM_GATES:
        gates[f"mom:{mg}"] = _compute_momentum_gate(df, mg)
    return gates


# -- Signal generator (long only) ---------------------------------------------

def _generate_signals(brick_up, n_bricks, cooldown, gate_long_ok):
    """N-brick momentum entry, first down brick exit. Returns (long_entry, long_exit)."""
    n = len(brick_up)
    long_entry = np.zeros(n, dtype=bool)
    long_exit  = np.zeros(n, dtype=bool)

    last_trade_bar = -999_999
    warmup = max(n_bricks + 1, 200)

    for i in range(warmup, n):
        long_exit[i] = not brick_up[i]

        if (i - last_trade_bar) < cooldown:
            continue
        if not gate_long_ok[i]:
            continue

        window = brick_up[i - n_bricks + 1 : i + 1]
        if bool(np.all(window)):
            long_entry[i] = True
            last_trade_bar = i

    return long_entry, long_exit


# -- Backtest runner -----------------------------------------------------------

def _run_backtest_kpis(df_sig, start, end):
    from engine import BacktestConfig, run_backtest

    cfg = BacktestConfig(
        initial_capital=CAPITAL,
        commission_pct=COMMISSION,
        slippage_ticks=0,
        qty_type="fixed",
        qty_value=1,
        pyramiding=1,
        start_date=start,
        end_date=end,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest(df_sig, cfg)

    pf = kpis.get("profit_factor", 0.0) or 0.0
    return {
        "pf":     float("inf") if math.isinf(pf) else float(pf),
        "net":    float(kpis.get("net_profit", 0.0) or 0.0),
        "trades": int(kpis.get("total_trades", 0) or 0),
        "wr":     float(kpis.get("win_rate", 0.0) or 0.0),
        "dd":     float(kpis.get("max_drawdown_pct", 0.0) or 0.0),
    }


# -- Parallel worker -----------------------------------------------------------

_worker_cache = {}


def _worker_init():
    """Lazy-load data + gates once per worker process."""
    if "df" in _worker_cache:
        return
    sys.path.insert(0, str(ROOT))
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators

    with contextlib.redirect_stdout(io.StringIO()):
        df = load_renko_export(RENKO_FILE)
        add_renko_indicators(df)

    _worker_cache["df"] = df
    _worker_cache["brick_up"] = df["brick_up"].values
    _worker_cache["gates"] = _compute_all_gates(df)


def _run_one(task):
    """Process a single combo: generate signals + IS/OOS backtests."""
    adx_t, trend, mom, n_bricks, cooldown = task
    _worker_init()

    df    = _worker_cache["df"]
    brick_up = _worker_cache["brick_up"]
    gates = _worker_cache["gates"]

    # Combine gates
    gate_long = gates[f"adx_{adx_t}"].copy()
    gate_long &= gates[f"trend:{trend}"]
    gate_long &= gates[f"mom:{mom}"]

    # Generate signals
    long_entry, long_exit = _generate_signals(brick_up, n_bricks, cooldown, gate_long)

    df_sig = df.copy()
    df_sig["long_entry"] = long_entry
    df_sig["long_exit"]  = long_exit

    # Run IS + OOS
    is_r  = _run_backtest_kpis(df_sig, IS_START, IS_END)
    oos_r = _run_backtest_kpis(df_sig, OOS_START, OOS_END)

    is_pf  = is_r["pf"]
    oos_pf = oos_r["pf"]
    decay = ((oos_pf - is_pf) / is_pf * 100) \
            if is_pf > 0 and not math.isinf(is_pf) else float("nan")

    adx_label = f"a{adx_t}" if adx_t > 0 else "a0"
    stack_label = f"{adx_label}_{trend}_{mom}"

    return {
        "stack":       stack_label,
        "trend_gate":  trend,
        "mom_gate":    mom,
        "adx_thresh":  adx_t,
        "n_bricks":    n_bricks,
        "cooldown":    cooldown,
        "is_pf":       is_pf,
        "is_trades":   is_r["trades"],
        "is_net":      is_r["net"],
        "is_wr":       is_r["wr"],
        "is_dd":       is_r["dd"],
        "oos_pf":      oos_pf,
        "oos_trades":  oos_r["trades"],
        "oos_net":     oos_r["net"],
        "oos_wr":      oos_r["wr"],
        "oos_dd":      oos_r["dd"],
        "decay_pct":   decay,
    }


# -- Sweep (parallel) ---------------------------------------------------------

def run_sweep():
    # Build all tasks
    param_combos = list(itertools.product(*PARAM_GRID.values()))
    gate_combos = list(itertools.product(ADX_THRESHOLDS, TREND_GATES, MOMENTUM_GATES))

    tasks = []
    for adx_t, trend, mom in gate_combos:
        for n_bricks, cooldown in param_combos:
            tasks.append((adx_t, trend, mom, n_bricks, cooldown))

    total = len(tasks)
    n_workers = min(total, MAX_WORKERS)
    print(f"Sweeping {total} combos across {n_workers} workers...", flush=True)

    results = []
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_one, t): t for t in tasks}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            done += 1
            if done % 500 == 0 or done == total:
                print(
                    f"  {done:>5}/{total} | {r['stack']:<40} "
                    f"n={r['n_bricks']} cd={r['cooldown']:>2} | "
                    f"IS PF={r['is_pf']:>7.2f} T={r['is_trades']:>4} | "
                    f"OOS PF={r['oos_pf']:>7.2f} T={r['oos_trades']:>4}",
                    flush=True,
                )

    print(f"Complete -- {len(results)} results", flush=True)
    return results


# -- Summary -------------------------------------------------------------------

def _summarize(results):
    viable = [r for r in results if r["oos_trades"] >= 10]
    viable.sort(
        key=lambda r: (r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6),
        reverse=True,
    )

    print(f"\n{'='*100}")
    print(f"  USO $0.25 Renko -- Phase 1 Gate Discovery (Long Only)")
    print(f"  Total combos: {len(results)} | Viable (OOS trades >= 10): {len(viable)}")
    print(f"{'='*100}")

    # Top 25
    print(f"\n  Top 25 (OOS trades >= 10):")
    print(f"  {'Stack':<40} {'n':>2} {'cd':>3} | {'IS PF':>7} {'T':>5} {'WR%':>6} | "
          f"{'OOS PF':>7} {'T':>5} {'WR%':>6} {'$Net':>8} {'Decay':>7}")
    print(f"  {'-'*105}")
    for r in viable[:25]:
        dec_s = f"{r['decay_pct']:>+6.1f}%" if not math.isnan(r["decay_pct"]) else "   NaN"
        print(f"  {r['stack']:<40} {r['n_bricks']:>2} {r['cooldown']:>3} | "
              f"{r['is_pf']:>7.2f} {r['is_trades']:>5} {r['is_wr']:>6.1f}% | "
              f"{r['oos_pf']:>7.2f} {r['oos_trades']:>5} {r['oos_wr']:>6.1f}% "
              f"{r['oos_net']:>8.2f} {dec_s}")

    # By trend gate
    print(f"\n  By trend gate (avg OOS PF, viable):")
    for tg in TREND_GATES:
        tv = [r for r in viable if r["trend_gate"] == tg]
        if tv:
            finite = [r["oos_pf"] for r in tv if not math.isinf(r["oos_pf"])]
            avg = sum(finite) / max(1, len(finite))
            avg_t = sum(r["oos_trades"] for r in tv) / len(tv)
            print(f"    {tg:<20} avg PF={avg:>7.2f}  avg T={avg_t:>6.1f}  N={len(tv):>4}")

    # By momentum gate
    print(f"\n  By momentum gate (avg OOS PF, viable):")
    for mg in MOMENTUM_GATES:
        mv = [r for r in viable if r["mom_gate"] == mg]
        if mv:
            finite = [r["oos_pf"] for r in mv if not math.isinf(r["oos_pf"])]
            avg = sum(finite) / max(1, len(finite))
            avg_t = sum(r["oos_trades"] for r in mv) / len(mv)
            print(f"    {mg:<20} avg PF={avg:>7.2f}  avg T={avg_t:>6.1f}  N={len(mv):>4}")

    # By ADX threshold
    print(f"\n  By ADX threshold (avg OOS PF, viable):")
    for at in ADX_THRESHOLDS:
        av = [r for r in viable if r["adx_thresh"] == at]
        if av:
            finite = [r["oos_pf"] for r in av if not math.isinf(r["oos_pf"])]
            avg = sum(finite) / max(1, len(finite))
            avg_t = sum(r["oos_trades"] for r in av) / len(av)
            label = f"ADX>={at}" if at > 0 else "no ADX"
            print(f"    {label:<20} avg PF={avg:>7.2f}  avg T={avg_t:>6.1f}  N={len(av):>4}")

    # By n_bricks
    print(f"\n  By brick count (avg OOS PF, viable):")
    for nb in PARAM_GRID["n_bricks"]:
        nv = [r for r in viable if r["n_bricks"] == nb]
        if nv:
            finite = [r["oos_pf"] for r in nv if not math.isinf(r["oos_pf"])]
            avg = sum(finite) / max(1, len(finite))
            avg_t = sum(r["oos_trades"] for r in nv) / len(nv)
            print(f"    n={nb:<3}              avg PF={avg:>7.2f}  avg T={avg_t:>6.1f}  N={len(nv):>4}")

    # By cooldown
    print(f"\n  By cooldown (avg OOS PF, viable):")
    for cd in PARAM_GRID["cooldown"]:
        cv = [r for r in viable if r["cooldown"] == cd]
        if cv:
            finite = [r["oos_pf"] for r in cv if not math.isinf(r["oos_pf"])]
            avg = sum(finite) / max(1, len(finite))
            avg_t = sum(r["oos_trades"] for r in cv) / len(cv)
            print(f"    cd={cd:<3}             avg PF={avg:>7.2f}  avg T={avg_t:>6.1f}  N={len(cv):>4}")

    # Overall best
    if viable:
        best = viable[0]
        print(f"\n  BEST: {best['stack']} n={best['n_bricks']} cd={best['cooldown']}")
        print(f"        IS  PF={best['is_pf']:.2f} T={best['is_trades']} WR={best['is_wr']:.1f}%")
        print(f"        OOS PF={best['oos_pf']:.2f} T={best['oos_trades']} WR={best['oos_wr']:.1f}% "
              f"Net=${best['oos_net']:.2f}")


# -- Main ---------------------------------------------------------------------

def main():
    out_path = ROOT / "ai_context" / "stock_phase1_results.json"
    out_path.parent.mkdir(exist_ok=True)

    n_params = len(list(itertools.product(*PARAM_GRID.values())))
    n_sweep = len(ADX_THRESHOLDS) * len(TREND_GATES) * len(MOMENTUM_GATES)
    total = n_sweep * n_params

    print("Stock Phase 1: USO Gate Discovery (Long Only)")
    print(f"  Instrument     : USO (BATS_USO, 1S renko 0.25)")
    print(f"  IS period      : {IS_START} to {IS_END}")
    print(f"  OOS period     : {OOS_START} to {OOS_END}")
    print(f"  Commission     : {COMMISSION}%")
    print(f"  Capital        : ${CAPITAL:,.0f}")
    print(f"  Sizing         : 1 share per trade")
    print(f"  Param combos   : {n_params}")
    print(f"  Gate combos    : {n_sweep}")
    print(f"  Total runs     : {total} ({total * 2} IS+OOS backtests)")
    print(f"  Workers        : {MAX_WORKERS}")
    print(f"  Output         : {out_path}")
    print()

    results = run_sweep()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {len(results)} results -> {out_path}")

    _summarize(results)


if __name__ == "__main__":
    main()
