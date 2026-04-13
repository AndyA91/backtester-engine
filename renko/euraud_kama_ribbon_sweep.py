"""
EURAUD 0.0006 — R030 3-Line KAMA Ribbon Sweep (TRAIN/VALIDATE/HOLDOUT)

Purpose
-------
First test of the 3-line KAMA ribbon (R030, stripped: flip-entry + gray-exit only)
on EURAUD 0.0006. Carryover C-new: EURAUD has NOT been tested with any KAMA
ribbon structure before. Phase 19 killed KAMA on EURUSD 0.0006 specifically, but
per meta-rule R20 (cross-instrument carryover is not free), each instrument must
be independently validated.

Discipline (per R22)
--------------------
User explicitly asked for 3-way split. Although EURAUD isn't a futures instrument,
3-way is the right call for a first-pass sweep of a new strategy/instrument combo:
it forces a locked decision rule and avoids OOS regime luck.

  TRAIN     2023-07-20 -> 2025-09-30   (~2.2 years, sweep runs here)
  VALIDATE  2025-10-01 -> 2025-12-31   (3 months, finalist selection)
  HOLDOUT   2026-01-01 -> 2026-03-14   (~2.5 months, sealed — tested once)

Decision rules (LOCKED before viewing HOLDOUT)
---------------------------------------------
Finalist filter (TRAIN -> VALIDATE):
  - VALIDATE trades >= 10  (EURAUD 0.0006 is slow; strategy will be selective)
  - TRAIN trades >= 40
  - VALIDATE WR delta vs TRAIN in [-5, +15]pp  (avoid curve fit and regime luck)
  - VALIDATE net profit > 0
  - Rank finalists by VALIDATE PF, break ties by VALIDATE net

Holdout acceptance (for a config to be called "passing"):
  - HOLDOUT PF >= 2.0
  - HOLDOUT WR >= 50%
  - HOLDOUT trades >= 8
  - HOLDOUT net profit > 0

Compute budget
--------------
  4 fast × 4 mid × 4 slow × 1 fast_sc × 1 slow_sc = 64 combos
  Of these, ~56 are valid (fast < mid < slow); invalid short-circuit instantly.
  64 combos × 3 windows = 192 backtests + 0 baselines.
  Wallclock estimate: <30 seconds on 20 workers.

Usage
-----
  python renko/euraud_kama_ribbon_sweep.py
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

# Force UTF-8 stdout on Windows
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
from renko.config import MAX_WORKERS

import r030_kama_ribbon_3line as strat

RENKO_FILE = "OANDA_EURAUD, 1S renko 0.0006.csv"

# ── Splits ───────────────────────────────────────────────────────────────────
TRAIN_START    = "2023-07-20"
TRAIN_END      = "2025-09-30"
VALIDATE_START = "2025-10-01"
VALIDATE_END   = "2025-12-31"
HOLDOUT_START  = "2026-01-01"
HOLDOUT_END    = "2026-03-14"

WINDOWS = [
    ("TRAIN",    TRAIN_START,    TRAIN_END),
    ("VALIDATE", VALIDATE_START, VALIDATE_END),
    ("HOLDOUT",  HOLDOUT_START,  HOLDOUT_END),
]

# ── Backtest config (standard forex) ─────────────────────────────────────────
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


# ── Filters / thresholds (LOCKED before viewing HOLDOUT) ────────────────────
MIN_TRAIN_TRADES        = 40
MIN_VALIDATE_TRADES     = 10
WR_DELTA_MIN            = -5.0
WR_DELTA_MAX            = 15.0
TOP_N_FINALISTS         = 5

HOLDOUT_RULES = {
    "min_pf":         2.0,
    "min_wr":         50.0,
    "min_trades":     8,
    "min_net_profit": 0.0,
}

OUTPUT_FILE = ROOT / "ai_context" / "euraud_kama_ribbon_results.json"


# ─── Worker ───────────────────────────────────────────────────────────────────
_worker_cache = {}


def _run_one_window(args):
    """Run ONE (combo, window) backtest. Worker caches data + indicators."""
    params, window_name, start, end = args

    if "df" not in _worker_cache:
        df = load_renko_export(RENKO_FILE)
        add_renko_indicators(df)
        _worker_cache["df"] = df
        # Clear KAMA cache at worker boot
        strat._KAMA_CACHE.clear()

    df = _worker_cache["df"]
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
    return (f"PF={fmt_pf(r['pf']):>6} Net={r['net']:>7.2f} T={r['trades']:>4} "
            f"WR={r['win_rate']:>5.1f}% DD={r['max_dd_pct']:>5.2f}% "
            f"Exp={r['expectancy']:>6.3f}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    # Build combo list (only valid fast<mid<slow combos kept — invalid ones
    # would just short-circuit but we skip them up front to save schedule overhead)
    grid = strat.PARAM_GRID
    keys = list(grid.keys())
    combos = []
    for vals in itertools.product(*grid.values()):
        p = dict(zip(keys, vals))
        if p["fast_len"] < p["mid_len"] < p["slow_len"]:
            combos.append(p)

    print("=" * 70)
    print("EURAUD 0.0006 — R030 KAMA Ribbon 3-Line Sweep")
    print("=" * 70)
    print(f"Strategy   : r030_kama_ribbon_3line")
    print(f"Data       : {RENKO_FILE}")
    print(f"TRAIN      : {TRAIN_START} -> {TRAIN_END}")
    print(f"VALIDATE   : {VALIDATE_START} -> {VALIDATE_END}")
    print(f"HOLDOUT    : {HOLDOUT_START} -> {HOLDOUT_END}  (sealed)")
    print(f"Valid      : {len(combos)} combos × 3 windows = {len(combos)*3} backtests")
    print(f"Workers    : {min(len(combos)*3, MAX_WORKERS)}")
    print("=" * 70)

    # Build task list: (params, window_name, start, end)
    tasks = []
    for p in combos:
        for window_name, start, end in WINDOWS:
            tasks.append((p, window_name, start, end))

    results = []  # list of dicts
    done = 0
    with ProcessPoolExecutor(max_workers=min(len(tasks), MAX_WORKERS)) as pool:
        futures = {pool.submit(_run_one_window, t): t for t in tasks}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            done += 1
            if done % 25 == 0 or done == len(tasks):
                print(f"  [{done:>4}/{len(tasks)}] {r['window']:<8} {fmt_row(r)} | {r['params']}")

    # Pivot results: {param_key_str: {window: result_dict}}
    by_combo = {}
    for r in results:
        key = tuple(sorted(r["params"].items()))
        by_combo.setdefault(key, {})[r["window"]] = r

    # ── Finalist selection (TRAIN + VALIDATE only — HOLDOUT sealed) ─────────
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
        wr_delta = val["win_rate"] - train["win_rate"]
        if wr_delta < WR_DELTA_MIN or wr_delta > WR_DELTA_MAX:
            continue
        if val["net"] <= 0:
            continue
        finalists.append({
            "params":   dict(key),
            "train":    train,
            "validate": val,
            "wr_delta": wr_delta,
        })

    # Rank by VALIDATE PF, tiebreak by VALIDATE net
    finalists.sort(key=lambda f: (
        f["validate"]["pf"] if not math.isinf(f["validate"]["pf"]) else 1e12,
        f["validate"]["net"],
    ), reverse=True)

    top_finalists = finalists[:TOP_N_FINALISTS]

    print()
    print("=" * 70)
    print(f"TRAIN → VALIDATE: {len(finalists)} finalists passed filters "
          f"(min_train={MIN_TRAIN_TRADES}, min_val={MIN_VALIDATE_TRADES}, "
          f"wr_delta∈[{WR_DELTA_MIN},{WR_DELTA_MAX}])")
    print("=" * 70)

    if not top_finalists:
        print("NO FINALISTS — strategy does not produce a valid TRAIN→VALIDATE config.")
        print("HOLDOUT will NOT be evaluated.")
    else:
        print(f"\n--- Top {len(top_finalists)} finalists (by VALIDATE PF) ---")
        for i, f in enumerate(top_finalists, 1):
            print(f"  [{i}] {f['params']}")
            print(f"      TRAIN    {fmt_row(f['train'])}")
            print(f"      VALIDATE {fmt_row(f['validate'])}  Δwr={f['wr_delta']:+.1f}pp")

        # ── HOLDOUT evaluation (sealed until now) ───────────────────────────
        print()
        print("=" * 70)
        print(f"HOLDOUT EVALUATION (rules locked: "
              f"PF≥{HOLDOUT_RULES['min_pf']}, WR≥{HOLDOUT_RULES['min_wr']}%, "
              f"T≥{HOLDOUT_RULES['min_trades']}, net>0)")
        print("=" * 70)

        for i, f in enumerate(top_finalists, 1):
            key = tuple(sorted(f["params"].items()))
            hold = by_combo[key].get("HOLDOUT")
            f["holdout"] = hold
            if not hold:
                f["passes"] = False
                print(f"  [{i}] NO HOLDOUT DATA  {f['params']}")
                continue
            passes = (
                hold["pf"]       >= HOLDOUT_RULES["min_pf"]       and
                hold["win_rate"] >= HOLDOUT_RULES["min_wr"]       and
                hold["trades"]   >= HOLDOUT_RULES["min_trades"]   and
                hold["net"]      >  HOLDOUT_RULES["min_net_profit"]
            )
            f["passes"] = passes
            tag = "PASS" if passes else "FAIL"
            print(f"  [{i}] {tag}  {f['params']}")
            print(f"      HOLDOUT  {fmt_row(hold)}")

        n_pass = sum(1 for f in top_finalists if f.get("passes"))
        print()
        print(f"HOLDOUT VERDICT: {n_pass}/{len(top_finalists)} finalists passed.")

    # ── Persist ──────────────────────────────────────────────────────────────
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump({
            "strategy":   "r030_kama_ribbon_3line",
            "instrument": "EURAUD 0.0006",
            "data_file":  RENKO_FILE,
            "splits": {
                "TRAIN":    [TRAIN_START,    TRAIN_END],
                "VALIDATE": [VALIDATE_START, VALIDATE_END],
                "HOLDOUT":  [HOLDOUT_START,  HOLDOUT_END],
            },
            "filters": {
                "min_train_trades":    MIN_TRAIN_TRADES,
                "min_validate_trades": MIN_VALIDATE_TRADES,
                "wr_delta_min":        WR_DELTA_MIN,
                "wr_delta_max":        WR_DELTA_MAX,
            },
            "holdout_rules":  HOLDOUT_RULES,
            "total_combos":   len(combos),
            "total_backtests": len(tasks),
            "all_results":    results,
            "finalists":      top_finalists,
        }, f, indent=2, default=str)

    print(f"\nResults saved → {OUTPUT_FILE}")
    print(f"Wallclock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
