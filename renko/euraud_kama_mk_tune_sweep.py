"""
EURAUD 0.0006 — R034 MK v4 Tuning Sweep (TRAIN+VAL-only selection)

Follow-up to R033. Takes the R033 winner (Config #2) and sweeps ONLY the
MK v4 internal parameters to see if the defaults can be improved.

**CRITICAL METHODOLOGY**: This sweep follows the R26 candidate rule — finalists
are ranked by MIN(TRAIN PF, VAL PF), NOT by single-window PF. HOLDOUT and
FRESH are reported POST-HOC as informational only and are NOT used in any
selection logic. This preserves FRESH as a sealed window for future use.

Ranking metric: min(TRAIN PF, VAL PF) — stability, not peak
Finalist filter: TRAIN trades >= 50, VAL trades >= 10, both PFs > 1.0

Config #2 baseline (default MK, from R033 TV validation):
  TRAIN    PF=2.30  WR=65.6%  T=482
  VAL      PF=2.09  WR=63.6%  T= 55
  HOLDOUT  PF=2.29  WR=65.4%  T= 52
  FRESH    PF=2.53  WR=68.0%  T= 25

To "win" this sweep, a tuned config must:
  1. Pass finalist filters (TRAIN T>=50, VAL T>=10, TRAIN+VAL PF > 1.0)
  2. Improve on Config #2's min(TRAIN PF, VAL PF) = 2.09
  3. Report HOLDOUT + FRESH post-hoc for sanity comparison

The informational HOLDOUT/FRESH comparison is INTERPRETED, not RANKED.
If a tuned config beats defaults on TRAIN+VAL AND ALSO on HOLDOUT+FRESH,
that's a weak positive signal worth fresh-data revalidation.
If a tuned config beats defaults on TRAIN+VAL but loses on HOLDOUT+FRESH,
that's overfitting — MK defaults are better.

Splits
------
  TRAIN     2023-07-20 -> 2025-09-30
  VALIDATE  2025-10-01 -> 2025-12-31
  HOLDOUT   2026-01-01 -> 2026-03-14   (sealed — post-hoc only)
  FRESH     2026-03-15 -> data end     (sealed — post-hoc only)

Compute budget
--------------
  2 mk_mode × 3 ema × 3 max_smooth × 3 neutral × 3 strength = 162 combos
  162 × 4 windows = 648 backtests
  MK v4 computation is the slow part; per-worker cache helps.
  Wallclock estimate: ~3-5 min.

Usage
-----
  python renko/euraud_kama_mk_tune_sweep.py
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
from renko.config import MAX_WORKERS

import r034_kama_mk_tuned as strat

RENKO_FILE = "OANDA_EURAUD, 1S renko 0.0006.csv"

TRAIN_START    = "2023-07-20"
TRAIN_END      = "2025-09-30"
VALIDATE_START = "2025-10-01"
VALIDATE_END   = "2025-12-31"
HOLDOUT_START  = "2026-01-01"
HOLDOUT_END    = "2026-03-14"
FRESH_START    = "2026-03-15"
FRESH_END      = "2099-12-31"

WINDOWS = [
    ("TRAIN",    TRAIN_START,    TRAIN_END),
    ("VALIDATE", VALIDATE_START, VALIDATE_END),
    ("HOLDOUT",  HOLDOUT_START,  HOLDOUT_END),
    ("FRESH",    FRESH_START,    FRESH_END),
]

# Config #2 baseline (default MK) — for comparison
BASELINE_NAME = "defaults (mk=block_flat ema=14 ms=10 nz=0.3 st=0.6)"
BASELINE_MIN_PF = 2.09  # min(TRAIN 2.30, VAL 2.09) — from TV validation


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


MIN_TRAIN_TRADES    = 50
MIN_VALIDATE_TRADES = 10
TOP_N_FINALISTS     = 10

OUTPUT_FILE = ROOT / "ai_context" / "euraud_kama_mk_tune_results.json"


_worker_cache = {}


def _run_one_window(args):
    params, window_name, start, end = args
    if "df" not in _worker_cache:
        df = load_renko_export(RENKO_FILE)
        add_renko_indicators(df)
        _worker_cache["df"] = df
        strat._KAMA_CACHE.clear()
        strat._MK_CACHE.clear()

    df = _worker_cache["df"]
    df_sig = strat.generate_signals(df, **params)
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


def label(p):
    return (f"mk:{p['mk_mode']:<10} ema{p['ema_length']:>2} "
            f"ms{p['max_smooth']:>2} nz{p['neutral_atr_pct']} "
            f"st{p['strength_threshold']}")


def main():
    t0 = time.time()

    grid = strat.PARAM_GRID
    keys = list(grid.keys())
    combos = [dict(zip(keys, v)) for v in itertools.product(*grid.values())]

    print("=" * 78)
    print("EURAUD 0.0006 — R034 MK v4 Tuning Sweep (TRAIN+VAL-only selection)")
    print("=" * 78)
    print(f"Strategy : r034_kama_mk_tuned (Config #2 LOCKED)")
    print(f"Baseline : {BASELINE_NAME}")
    print(f"           min(TRAIN PF, VAL PF) = {BASELINE_MIN_PF}")
    print(f"Combos   : {len(combos)} × 4 windows = {len(combos)*4} backtests")
    print(f"Workers  : {min(len(combos)*4, MAX_WORKERS)}")
    print(f"RANKING  : min(TRAIN PF, VAL PF)  -- R26 candidate rule")
    print(f"HOLDOUT  : POST-HOC informational (NOT used in selection)")
    print(f"FRESH    : POST-HOC informational (NOT used in selection)")
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
            if done % 50 == 0 or done == len(tasks):
                print(f"  [{done:>4}/{len(tasks)}] {r['window']:<8} "
                      f"{fmt_row(r)} | {label(r['params'])}")

    # Pivot by combo
    by_combo = {}
    for r in results:
        p = r["params"]
        key = (p["mk_mode"], p["ema_length"], p["max_smooth"],
               p["neutral_atr_pct"], p["strength_threshold"])
        by_combo.setdefault(key, {})[r["window"]] = r

    # Finalist selection: TRAIN+VAL only
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
        if train["pf"] <= 1.0 or val["pf"] <= 1.0:
            continue
        min_pf = min(train["pf"], val["pf"])
        finalists.append({
            "params":   {"mk_mode": key[0], "ema_length": key[1],
                         "max_smooth": key[2], "neutral_atr_pct": key[3],
                         "strength_threshold": key[4]},
            "train":    train,
            "validate": val,
            "min_pf":   min_pf,
        })

    # Sort by min_pf descending
    finalists.sort(key=lambda f: f["min_pf"], reverse=True)
    top = finalists[:TOP_N_FINALISTS]

    print()
    print("=" * 78)
    print(f"TRAIN+VAL finalists: {len(finalists)} passed filters  "
          f"(baseline min_pf = {BASELINE_MIN_PF})")
    print("=" * 78)
    if not top:
        print("NO FINALISTS.")
    else:
        print(f"\n--- Top {len(top)} finalists (by min(TRAIN PF, VAL PF)) ---")
        for i, f in enumerate(top, 1):
            beat = "BEAT" if f["min_pf"] > BASELINE_MIN_PF else "    "
            print(f"  [{i}] {beat}  min_PF={f['min_pf']:.3f}  "
                  f"{label(f['params'])}")
            print(f"      TRAIN    {fmt_row(f['train'])}")
            print(f"      VALIDATE {fmt_row(f['validate'])}")

        # POST-HOC: HOLDOUT + FRESH for the top finalists ──────────────────
        print()
        print("=" * 78)
        print("POST-HOC HOLDOUT + FRESH  (informational only, NOT used in selection)")
        print("=" * 78)
        print(f"Baseline R033 Config #2 TV results (for comparison):")
        print(f"  HOLDOUT  PF=2.29  WR=65.4%  T=52")
        print(f"  FRESH    PF=2.53  WR=68.0%  T=25")
        print()
        for i, f in enumerate(top, 1):
            key = (f["params"]["mk_mode"], f["params"]["ema_length"],
                   f["params"]["max_smooth"], f["params"]["neutral_atr_pct"],
                   f["params"]["strength_threshold"])
            hold = by_combo[key].get("HOLDOUT")
            fresh = by_combo[key].get("FRESH")
            f["holdout"] = hold
            f["fresh"]   = fresh
            print(f"  [{i}] {label(f['params'])}")
            if hold:
                print(f"      HOLDOUT  {fmt_row(hold)}")
            if fresh:
                print(f"      FRESH    {fmt_row(fresh)}")

        # Count how many finalists ALSO improved on HOLDOUT+FRESH vs baseline
        improved_all = 0
        for f in top:
            hold = f.get("holdout")
            fresh = f.get("fresh")
            if hold and fresh and hold["pf"] > 2.29 and fresh["pf"] > 2.53:
                improved_all += 1
        print()
        print(f"Top {len(top)} finalists that ALSO beat Config #2 on "
              f"HOLDOUT AND FRESH: {improved_all}")
        if improved_all > 0:
            print("  ^ Weak positive signal — candidates for fresh-data revalidation.")
        else:
            print("  ^ No top finalist beats Config #2 across all 4 windows.")
            print("    Interpretation: MK defaults are near-optimal for this setup,")
            print("    OR TRAIN+VAL tuning is overfitting even with stability ranking.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump({
            "strategy":   "r034_kama_mk_tuned",
            "instrument": "EURAUD 0.0006",
            "data_file":  RENKO_FILE,
            "baseline": {
                "name": BASELINE_NAME,
                "min_pf": BASELINE_MIN_PF,
                "holdout_pf": 2.29, "fresh_pf": 2.53,
            },
            "splits": {
                "TRAIN":    [TRAIN_START,    TRAIN_END],
                "VALIDATE": [VALIDATE_START, VALIDATE_END],
                "HOLDOUT":  [HOLDOUT_START,  HOLDOUT_END],
                "FRESH":    [FRESH_START,    FRESH_END],
            },
            "ranking_metric": "min(TRAIN_PF, VALIDATE_PF)  -- R26 candidate rule",
            "filters": {
                "min_train_trades":    MIN_TRAIN_TRADES,
                "min_validate_trades": MIN_VALIDATE_TRADES,
                "min_pf_per_window":   1.0,
            },
            "total_combos":    len(combos),
            "total_backtests": len(tasks),
            "all_results":     results,
            "finalists":       top,
        }, f, indent=2, default=str)

    print(f"\nResults saved -> {OUTPUT_FILE}")
    print(f"Wallclock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
