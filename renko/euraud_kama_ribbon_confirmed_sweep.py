"""
EURAUD 0.0006 — R032 KAMA Ribbon 5/13/30 + Confirmation Delay Sweep

Third sweep in the EURAUD KAMA ribbon series. Tests whether delaying entry
by N bricks of continued alignment filters chop-wobble losers (median ~2-3
bar lifespan) while preserving real-trend winners (median ~17 bar lifespan).

Ribbon is LOCKED to the R030 best: 5/13/30. Tests:
  - confirm_bars ∈ [1, 2, 3, 5, 8]
  - use_adx_gate ∈ [False, True]   (tests whether delay alone is enough)

Sanity checks built in:
  - confirm_bars=1, use_adx_gate=False  → reproduces R030 no-gate
  - confirm_bars=1, use_adx_gate=True   → reproduces R031 adx>=20 winner

HOLDOUT WARNING: this is the 3rd sweep on the same EURAUD 0.0006 HOLDOUT
window. Per R22 extension, family-wise error rate is now non-trivial. If
anything passes here, treat it as a fresh-data-candidate only — do NOT
deploy without refreshing data through April 2026+ and re-running the
winning config ONCE with decision rules locked beforehand.

Splits
------
  TRAIN     2023-07-20 -> 2025-09-30
  VALIDATE  2025-10-01 -> 2025-12-31
  HOLDOUT   2026-01-01 -> 2026-03-14

Decision rules (LOCKED before viewing HOLDOUT)
----------------------------------------------
Finalist filter (TRAIN -> VALIDATE):
  - VALIDATE trades >= 10
  - TRAIN trades >= 30
  - WR delta in [-5, +15]pp
  - VALIDATE net > 0

Holdout pass (must clear ALL):
  - HOLDOUT PF >= 2.5                 (bumped from 2.0 — R031's 2.12 was weak)
  - HOLDOUT WR >= 52%                 (bumped from 50% — tighten bar)
  - HOLDOUT trades >= 8
  - HOLDOUT net > R031 winner net ($75.17)   (NEW: must beat prior winner)

The bumped bar reflects that we've already produced a "weak pass" (R031)
that is still 5x below live EA022. A third-sweep winner must materially
improve on R031 to be worth fresh-data revalidation.

Compute budget
--------------
  5 × 2 = 10 combos × 3 windows = 30 backtests
  Wallclock: <30 seconds on 20 workers.

Usage
-----
  python renko/euraud_kama_ribbon_confirmed_sweep.py
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

import r032_kama_ribbon_confirmed as strat

RENKO_FILE = "OANDA_EURAUD, 1S renko 0.0006.csv"

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

MIN_TRAIN_TRADES        = 30
MIN_VALIDATE_TRADES     = 10
WR_DELTA_MIN            = -5.0
WR_DELTA_MAX            = 15.0
TOP_N_FINALISTS         = 10

# Bumped bar — must materially beat R031 adx>=20 (PF=2.12, net=$75.17)
HOLDOUT_RULES = {
    "min_pf":         2.5,
    "min_wr":         52.0,
    "min_trades":     8,
    "min_net_profit": 75.17,
}

OUTPUT_FILE = ROOT / "ai_context" / "euraud_kama_ribbon_confirmed_results.json"


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


_worker_cache = {}


def _run_one_window(args):
    params, window_name, start, end = args
    if "df" not in _worker_cache:
        df = load_renko_export(RENKO_FILE)
        add_renko_indicators(df)
        _worker_cache["df"] = df
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


def label(p):
    gate = "adx20" if p["use_adx_gate"] else "nogate"
    return f"N={p['confirm_bars']:<2} {gate}"


def main():
    t0 = time.time()

    grid = strat.PARAM_GRID
    keys = list(grid.keys())
    combos = [dict(zip(keys, v)) for v in itertools.product(*grid.values())]

    print("=" * 70)
    print("EURAUD 0.0006 — R032 KAMA Ribbon Confirmation-Delay Sweep")
    print("=" * 70)
    print(f"Strategy   : r032_kama_ribbon_confirmed (ribbon LOCKED 5/13/30)")
    print(f"Data       : {RENKO_FILE}")
    print(f"TRAIN      : {TRAIN_START} -> {TRAIN_END}")
    print(f"VALIDATE   : {VALIDATE_START} -> {VALIDATE_END}")
    print(f"HOLDOUT    : {HOLDOUT_START} -> {HOLDOUT_END}  (3rd use — flag)")
    print(f"Combos     : {len(combos)} × 3 windows = {len(combos)*3} backtests")
    print(f"Workers    : {min(len(combos)*3, MAX_WORKERS)}")
    print(f"Bar bumped : PF>=2.5, WR>=52%, net > $75.17 (R031 winner)")
    print("=" * 70)

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
            if done % 10 == 0 or done == len(tasks):
                print(f"  [{done:>3}/{len(tasks)}] {r['window']:<8} "
                      f"{fmt_row(r)} | {label(r['params'])}")

    by_combo = {}
    for r in results:
        key = (r["params"]["confirm_bars"], r["params"]["use_adx_gate"])
        by_combo.setdefault(key, {})[r["window"]] = r

    print()
    print("=" * 70)
    print("PER-COMBO SUMMARY (all windows)")
    print("=" * 70)
    order = sorted(by_combo.keys())
    for key in order:
        wmap = by_combo[key]
        p = {"confirm_bars": key[0], "use_adx_gate": key[1]}
        print(f"\n  {label(p)}")
        for wname, _, _ in WINDOWS:
            r = wmap.get(wname)
            if r:
                print(f"    {wname:<8} {fmt_row(r)}")

    # Finalist selection
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
            "params":   {"confirm_bars": key[0], "use_adx_gate": key[1]},
            "train":    train,
            "validate": val,
            "wr_delta": wr_delta,
        })

    finalists.sort(key=lambda f: (
        f["validate"]["pf"] if not math.isinf(f["validate"]["pf"]) else 1e12,
        f["validate"]["net"],
    ), reverse=True)

    top_finalists = finalists[:TOP_N_FINALISTS]

    print()
    print("=" * 70)
    print(f"TRAIN → VALIDATE: {len(finalists)} finalists passed filters")
    print("=" * 70)

    if not top_finalists:
        print("NO FINALISTS.")
    else:
        print(f"\n--- Top {len(top_finalists)} finalists (by VALIDATE PF) ---")
        for i, f in enumerate(top_finalists, 1):
            print(f"  [{i}] {label(f['params'])}")
            print(f"      TRAIN    {fmt_row(f['train'])}")
            print(f"      VALIDATE {fmt_row(f['validate'])}  Δwr={f['wr_delta']:+.1f}pp")

        print()
        print("=" * 70)
        print(f"HOLDOUT EVALUATION (BUMPED rules: PF≥{HOLDOUT_RULES['min_pf']}, "
              f"WR≥{HOLDOUT_RULES['min_wr']}%, T≥{HOLDOUT_RULES['min_trades']}, "
              f"net>${HOLDOUT_RULES['min_net_profit']})")
        print("=" * 70)

        for i, f in enumerate(top_finalists, 1):
            key = (f["params"]["confirm_bars"], f["params"]["use_adx_gate"])
            hold = by_combo[key].get("HOLDOUT")
            f["holdout"] = hold
            if not hold:
                f["passes"] = False
                print(f"  [{i}] NO HOLDOUT DATA  {label(f['params'])}")
                continue
            passes = (
                hold["pf"]       >= HOLDOUT_RULES["min_pf"]       and
                hold["win_rate"] >= HOLDOUT_RULES["min_wr"]       and
                hold["trades"]   >= HOLDOUT_RULES["min_trades"]   and
                hold["net"]      >  HOLDOUT_RULES["min_net_profit"]
            )
            f["passes"] = passes
            tag = "PASS" if passes else "FAIL"
            print(f"  [{i}] {tag}  {label(f['params'])}")
            print(f"      HOLDOUT  {fmt_row(hold)}")

        n_pass = sum(1 for f in top_finalists if f.get("passes"))
        print()
        print(f"HOLDOUT VERDICT: {n_pass}/{len(top_finalists)} finalists passed BUMPED bar.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump({
            "strategy":   "r032_kama_ribbon_confirmed",
            "instrument": "EURAUD 0.0006",
            "data_file":  RENKO_FILE,
            "locked_ribbon": {"fast": 5, "mid": 13, "slow": 30, "fast_sc": 2, "slow_sc": 30},
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
            "holdout_use_count": 3,
        }, f, indent=2, default=str)

    print(f"\nResults saved → {OUTPUT_FILE}")
    print(f"Wallclock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
