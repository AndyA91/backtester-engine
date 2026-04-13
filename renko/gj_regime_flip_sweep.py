"""
GBPJPY 0.09 — GJ001 Regime Flip Sweep
======================================

First exploratory sweep of the ER regime-flip strategy on GBPJPY Renko 0.09.

Entry: ER regime flips RANGE → TREND + brick direction at flip bar.
Exit:  First opposing brick (R1 locked at sl_bricks=1).
TP:    Optional bracket in bricks (0 = disabled).

Splits (3-way per R22)
----------------------
  TRAIN     2023-03-09 -> 2025-09-30
  VALIDATE  2025-10-01 -> 2025-12-31
  HOLDOUT   2026-01-01 -> data end (2026-04-10, sealed)

Sweep grid (~2,000 combos × 3 windows = ~6,000 backtests)
---------------------------------------------------------
  er_lookback:   [6, 8, 10, 14, 20]       = 5
  er_smooth:     [1, 3, 5]                 = 3
  trend_thresh:  [0.35, 0.45, 0.55]        = 3
  range_thresh:  [0.15, 0.25]              = 2
  persistence:   [0, 2]                    = 2
  tp_bricks:     [0, 3, 5, 8]              = 4
  trail_bricks:  [1, 2, 3]                 = 3

trail_bricks=1 is equivalent to first-opposing-brick exit (R1 baseline).
KAMA/MK not used — pure ER regime.

Decision rules (LOCKED before viewing HOLDOUT)
----------------------------------------------
Finalist filter (TRAIN → VALIDATE):
  - VALIDATE trades >= 10
  - TRAIN trades >= 30
  - VALIDATE net > 0
  - TRAIN PF > 1.0
  - WR delta TRAIN → VALIDATE in [-10, +15]pp

Ranking: min(TRAIN PF, VAL PF) descending (R26 stability).
HOLDOUT deploy gate: PF >= 1.5, T >= 8, net > 0, collapse < 30%.

Compute budget: ~2,160 backtests, ~3-5 minutes.

Usage
-----
  python renko/gj_regime_flip_sweep.py
"""

import contextlib
import io
import itertools
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "renko" / "strategies"))

from engine import BacktestConfig, run_backtest_long_short
from renko.data import load_renko_export
from renko.indicators import add_renko_indicators
from renko.config import MAX_WORKERS as _RAW_WORKERS

MAX_WORKERS = min(_RAW_WORKERS, 6)  # Cap to avoid OOM with 20K-bar dataset

import gj001_regime_flip as strat

# ── Data ─────────────────────────────────────────────────────────────────────
RENKO_FILE = "OANDA_GBPJPY, 1S renko 0.09.csv"

# ── 3-way splits ─────────────────────────────────────────────────────────────
TRAIN_START    = "2023-03-09"
TRAIN_END      = "2025-09-30"
VALIDATE_START = "2025-10-01"
VALIDATE_END   = "2025-12-31"
HOLDOUT_START  = "2026-01-01"
HOLDOUT_END    = "2099-12-31"

WINDOWS = [
    ("TRAIN",    TRAIN_START,    TRAIN_END),
    ("VALIDATE", VALIDATE_START, VALIDATE_END),
    ("HOLDOUT",  HOLDOUT_START,  HOLDOUT_END),
]

# ── Sweep grid ───────────────────────────────────────────────────────────────
PARAM_GRID = {
    "er_lookback":   [6, 8, 10, 14, 20],
    "er_smooth":     [1, 3, 5],
    "trend_thresh":  [0.35, 0.45, 0.55],
    "range_thresh":  [0.15, 0.25],
    "persistence":   [0, 2],
    "tp_bricks":     [0, 3, 5, 8],
    "trail_bricks":  [1, 2, 3],
}

# ── Filters / thresholds ────────────────────────────────────────────────────
MIN_TRAIN_TRADES    = 30
MIN_VALIDATE_TRADES = 10
WR_DELTA_MIN        = -10.0
WR_DELTA_MAX        = 15.0
TOP_N_FINALISTS     = 10

HOLDOUT_RULES = {
    "min_pf":           1.5,
    "min_trades":       8,
    "min_net_profit":   0.0,
    "max_collapse_pct": 0.30,
}

OUTPUT_FILE = ROOT / "ai_context" / "gj_regime_flip_results.json"


def make_cfg(start, end):
    return BacktestConfig(
        initial_capital=1000.0,
        commission_pct=0.0046,
        slippage_ticks=0,
        qty_type="fixed",
        qty_value=1000.0,
        pyramiding=1,
        start_date=start,
        end_date=end,
        take_profit_pct=0.0,
        stop_loss_pct=0.0,
    )


# ── Worker ───────────────────────────────────────────────────────────────────
_worker_cache = {}


def _run_one_window(args):
    params, window_name, start, end = args

    if "df" not in _worker_cache:
        df = load_renko_export(RENKO_FILE)
        add_renko_indicators(df)
        _worker_cache["df"] = df

    df = _worker_cache["df"]

    # Clear ER cache when params change lookback/smooth (keyed internally)
    # No need — cache is keyed by (lookback, smooth), safe across combos

    strat._ER_CACHE.clear()
    df_sig = strat.generate_signals(df.copy(), **params)
    cfg = make_cfg(start, end)
    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest_long_short(df_sig, cfg)
    pf = kpis.get("profit_factor", 0.0) or 0.0
    return {
        "window":     window_name,
        "params":     params,
        "pf":         float("inf") if math.isinf(pf) else float(pf),
        "net":        float(kpis.get("net_profit", 0.0) or 0.0),
        "trades":     int(kpis.get("total_trades", 0) or 0),
        "win_rate":   float(kpis.get("win_rate", 0.0) or 0.0),
        "max_dd_pct": float(kpis.get("max_drawdown_pct", 0.0) or 0.0),
        "expectancy": float(kpis.get("avg_trade", 0.0) or 0.0),
    }


def fmt_pf(pf):
    return "INF" if math.isinf(pf) else f"{pf:.3f}"


def fmt_row(r):
    return (f"PF={fmt_pf(r['pf']):>6} Net={r['net']:>8.2f} T={r['trades']:>4} "
            f"WR={r['win_rate']:>5.1f}% DD={r['max_dd_pct']:>5.2f}% "
            f"Exp={r['expectancy']:>7.4f}")


def label(p):
    return (f"er{p['er_lookback']}s{p['er_smooth']} "
            f"t{p['trend_thresh']}r{p['range_thresh']} "
            f"p{p['persistence']} tp{p['tp_bricks']}b tr{p['trail_bricks']}b")


def evaluate(results):
    by_combo = {}
    for r in results:
        key = tuple(sorted(r["params"].items()))
        by_combo.setdefault(key, {})[r["window"]] = r

    finalists = []
    for key, wmap in by_combo.items():
        train = wmap.get("TRAIN")
        val   = wmap.get("VALIDATE")
        if not train or not val:
            continue
        if train["trades"] < MIN_TRAIN_TRADES:
            continue
        if val["trades"] < MIN_VALIDATE_TRADES:
            continue
        if train["pf"] <= 1.0:
            continue
        wr_delta = val["win_rate"] - train["win_rate"]
        if wr_delta < WR_DELTA_MIN or wr_delta > WR_DELTA_MAX:
            continue
        if val["net"] <= 0:
            continue

        train_pf = 1e12 if math.isinf(train["pf"]) else train["pf"]
        val_pf   = 1e12 if math.isinf(val["pf"])   else val["pf"]
        stability_pf = min(train_pf, val_pf)

        finalists.append({
            "params":       dict(key),
            "train":        train,
            "validate":     val,
            "wr_delta":     wr_delta,
            "stability_pf": stability_pf,
        })

    finalists.sort(key=lambda f: (f["stability_pf"], f["validate"]["net"]), reverse=True)
    top = finalists[:TOP_N_FINALISTS]

    print()
    print("=" * 78)
    print(f"GBPJPY 0.09  —  {len(finalists)} finalists passed TRAIN→VAL filters")
    print("=" * 78)

    if not top:
        print("NO FINALISTS — all combos rejected by TRAIN/VAL filters.")
        return top

    print(f"\n--- Top {len(top)} finalists (by R26 stability = min(TRAIN PF, VAL PF)) ---")
    for i, f in enumerate(top, 1):
        stab = "INF" if f["stability_pf"] >= 1e11 else f"{f['stability_pf']:.3f}"
        print(f"  [{i}] {label(f['params'])}  stability={stab}")
        print(f"      TRAIN    {fmt_row(f['train'])}")
        print(f"      VALIDATE {fmt_row(f['validate'])}  dwr={f['wr_delta']:+.1f}pp")

    # HOLDOUT deploy gate
    print()
    print("-" * 78)
    print(f"HOLDOUT DEPLOY GATE  "
          f"(rules: PF>={HOLDOUT_RULES['min_pf']}, "
          f"T>={HOLDOUT_RULES['min_trades']}, net>0, collapse<30%)")
    print("-" * 78)
    for i, f in enumerate(top, 1):
        key = tuple(sorted(f["params"].items()))
        hold = by_combo[key].get("HOLDOUT")
        f["holdout"] = hold
        if not hold:
            f["passes"] = False
            print(f"  [{i}] NO HOLDOUT DATA  {label(f['params'])}")
            continue

        hold_pf = 1e12 if math.isinf(hold["pf"]) else hold["pf"]
        collapse_floor = f["stability_pf"] * (1 - HOLDOUT_RULES["max_collapse_pct"])
        no_collapse = hold_pf >= collapse_floor

        passes = (
            hold["pf"]     >= HOLDOUT_RULES["min_pf"]       and
            hold["trades"] >= HOLDOUT_RULES["min_trades"]   and
            hold["net"]    >  HOLDOUT_RULES["min_net_profit"] and
            no_collapse
        )
        f["passes"] = passes
        f["holdout_collapse_pct"] = (
            None if f["stability_pf"] >= 1e11
            else (f["stability_pf"] - hold_pf) / f["stability_pf"]
        )
        tag = "PASS" if passes else "FAIL"
        collapse_tag = ""
        if f["holdout_collapse_pct"] is not None and not no_collapse:
            collapse_tag = f"  [collapse {f['holdout_collapse_pct']*100:.0f}% > 30%]"
        print(f"  [{i}] {tag}  {label(f['params'])}{collapse_tag}")
        print(f"      HOLDOUT  {fmt_row(hold)}")

    n_pass = sum(1 for f in top if f.get("passes"))
    print()
    print(f"VERDICT: {n_pass}/{len(top)} finalists passed deploy gate.")
    return top


def main():
    t0 = time.time()

    keys = list(PARAM_GRID.keys())
    combos = [dict(zip(keys, v)) for v in itertools.product(*PARAM_GRID.values())]

    # Filter invalid combos: trend_thresh must be > range_thresh
    combos = [c for c in combos if c["trend_thresh"] > c["range_thresh"]]

    n_tasks = len(combos) * len(WINDOWS)

    print("=" * 78)
    print("GBPJPY 0.09 — GJ001 Regime Flip Sweep")
    print("=" * 78)
    print(f"Strategy   : gj001_regime_flip (ER regime flip + brick direction)")
    print(f"Data       : {RENKO_FILE}")
    print(f"TRAIN      : {TRAIN_START} -> {TRAIN_END}")
    print(f"VALIDATE   : {VALIDATE_START} -> {VALIDATE_END}")
    print(f"HOLDOUT    : {HOLDOUT_START} -> data end (DEPLOY GATE)")
    print(f"Combos     : {len(combos)} (after trend>range filter)")
    print(f"Backtests  : {n_tasks}")
    print(f"Workers    : {min(n_tasks, MAX_WORKERS)}")
    print("=" * 78)

    tasks = []
    for p in combos:
        for window_name, start, end in WINDOWS:
            tasks.append((p, window_name, start, end))

    results = []
    done = 0
    with ProcessPoolExecutor(max_workers=min(len(tasks), MAX_WORKERS)) as pool:
        futures = {pool.submit(_run_one_window, t): t for t in tasks}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            done += 1
            if done % 100 == 0 or done == len(tasks):
                print(f"  [{done:>4}/{len(tasks)}] {r['window']:<8} "
                      f"{fmt_row(r)} | {label(r['params'])}")

    top = evaluate(results)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump({
            "strategy":   "gj001_regime_flip",
            "instrument": "GBPJPY",
            "brick_size": 0.09,
            "renko_file": RENKO_FILE,
            "splits": {
                "TRAIN":    [TRAIN_START,    TRAIN_END],
                "VALIDATE": [VALIDATE_START, VALIDATE_END],
                "HOLDOUT":  [HOLDOUT_START,  HOLDOUT_END],
            },
            "param_grid":      PARAM_GRID,
            "total_combos":    len(combos),
            "total_backtests": n_tasks,
            "filters": {
                "min_train_trades":    MIN_TRAIN_TRADES,
                "min_validate_trades": MIN_VALIDATE_TRADES,
                "wr_delta_min":        WR_DELTA_MIN,
                "wr_delta_max":        WR_DELTA_MAX,
            },
            "holdout_rules":   HOLDOUT_RULES,
            "all_results":     results,
            "finalists":       top,
        }, f, indent=2, default=str)

    print(f"\nResults saved -> {OUTPUT_FILE}")
    print(f"Wallclock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
