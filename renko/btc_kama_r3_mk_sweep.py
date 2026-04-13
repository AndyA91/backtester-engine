"""
BTC 150 — R035 KAMA Ribbon 3L + MK v4 Bracket Sweep (long-only, 4-way split)

BTC port of the EURAUD R033 sweep. Same mechanism, BTC conventions:
  - LONG ONLY (run_backtest, not run_backtest_long_short)
  - Cash mode position sizing (qty_value=20)
  - Brick size 150 (BTC Renko 150)
  - No volume data (BTC Renko exports have Volume=0 for IS period)

First TV test (kama_ribbon_btc.pine) showed:
  Full PF=3.11  WR=56.6%  T=412  W/L=2.39  MaxDD=$1.86

Best KAMA-family result on any instrument so far. But only 7 FRESH trades —
sample too small to claim deploy. This sweep tests whether the parameter
surface supports better candidates and confirms MK strict filtering generalizes.

Splits (4-way per R22 + R22 extension)
--------------------------------------
  TRAIN     2024-06-04 -> 2025-09-30  (~16 months)
  VALIDATE  2025-10-01 -> 2025-12-31  (3 months)
  HOLDOUT   2026-01-01 -> 2026-03-14
  FRESH     2026-03-15 -> data end    (NEW sealed window — DEPLOY GATE)

Decision rules (LOCKED before viewing FRESH)
--------------------------------------------
Per R26 candidate rule: rank by min(TRAIN, VALIDATE) PF — stability, not peak.

Finalist filter:
  - TRAIN trades >= 50
  - VALIDATE trades >= 10
  - VAL net > 0
  - TRAIN PF > 1.5 (stricter than EURAUD because BTC has higher PF baselines)
  - VAL PF > 1.5

FRESH deploy gate (per R26 extension — REJECTION GATE not informational):
  - FRESH PF >= 1.5
  - FRESH trades >= 5  (BTC FRESH window is small — relax from 8)
  - FRESH net > 0

Compute budget
--------------
  5 TP × 4 SL × 4 dist × 4 MK = 320 combos × 4 windows = 1,280 backtests
  Wallclock estimate: ~2-4 min on 20 workers.

Usage
-----
  python renko/btc_kama_r3_mk_sweep.py
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

from engine import BacktestConfig, run_backtest
from renko.data import load_renko_export
from renko.indicators import add_renko_indicators
from renko.config import MAX_WORKERS

import r035_kama_btc_long as strat

RENKO_FILE = "OANDA_BTCUSD.SPOT.US, 1S renko 150.csv"

# ── 4-way splits (BTC data starts 2024-06-04) ────────────────────────────────
TRAIN_START    = "2024-06-04"
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


def make_cfg(start, end):
    return BacktestConfig(
        initial_capital=1000.0,
        commission_pct=0.0046,
        slippage_ticks=0,
        qty_type="cash",
        qty_value=20.0,         # $20 notional per trade (BTC convention)
        pyramiding=1,
        start_date=start,
        end_date=end,
        take_profit_pct=0.0,
        stop_loss_pct=0.0,
    )


# ── Filters / thresholds ────────────────────────────────────────────────────
MIN_TRAIN_TRADES    = 50
MIN_VALIDATE_TRADES = 10
MIN_TRAIN_PF        = 1.5
MIN_VAL_PF          = 1.5
TOP_N_FINALISTS     = 10

# FRESH deploy gate (R26 extension — rejection gate, not informational)
FRESH_RULES = {
    "min_pf":         1.5,
    "min_trades":     5,    # BTC FRESH is sparse; relax from 8
    "min_net_profit": 0.0,
}

OUTPUT_FILE = ROOT / "ai_context" / "btc_kama_r3_mk_results.json"


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
        kpis = run_backtest(df_sig, cfg)
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
    dist = "off" if p['max_dist_bricks'] >= 1000 else f"d{p['max_dist_bricks']}"
    return f"tp{int(p['tp_dist'])} sl{p['sl_bricks']}b {dist} mk:{p['mk_mode']}"


def main():
    t0 = time.time()

    grid = strat.PARAM_GRID
    keys = list(grid.keys())
    combos = [dict(zip(keys, v)) for v in itertools.product(*grid.values())]

    print("=" * 78)
    print("BTC 150 — R035 KAMA Ribbon 3L + MK v4 Bracket Sweep (LONG ONLY)")
    print("=" * 78)
    print(f"Strategy : r035_kama_btc_long (KAMA LOCKED 5/13/60)")
    print(f"Data     : {RENKO_FILE}")
    print(f"TRAIN    : {TRAIN_START} -> {TRAIN_END}")
    print(f"VALIDATE : {VALIDATE_START} -> {VALIDATE_END}")
    print(f"HOLDOUT  : {HOLDOUT_START} -> {HOLDOUT_END}")
    print(f"FRESH    : {FRESH_START} -> data end  (DEPLOY GATE)")
    print(f"Combos   : {len(combos)} × 4 windows = {len(combos)*4} backtests")
    print(f"Workers  : {min(len(combos)*4, MAX_WORKERS)}")
    print(f"RANKING  : min(TRAIN, VAL) PF — R26 stability rule")
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

    by_combo = {}
    for r in results:
        p = r["params"]
        key = (p["tp_dist"], p["sl_bricks"], p["max_dist_bricks"], p["mk_mode"])
        by_combo.setdefault(key, {})[r["window"]] = r

    # ── Finalist selection: TRAIN+VAL only, ranked by min(TRAIN, VAL) PF ────
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
        if val["net"] <= 0:
            continue
        if train["pf"] < MIN_TRAIN_PF:
            continue
        if val["pf"] < MIN_VAL_PF:
            continue
        min_pf = min(train["pf"], val["pf"])
        finalists.append({
            "params":   {"tp_dist": key[0], "sl_bricks": key[1],
                         "max_dist_bricks": key[2], "mk_mode": key[3]},
            "train":    train,
            "validate": val,
            "min_pf":   min_pf,
        })

    finalists.sort(key=lambda f: f["min_pf"], reverse=True)
    top = finalists[:TOP_N_FINALISTS]

    print()
    print("=" * 78)
    print(f"TRAIN+VAL finalists: {len(finalists)} passed filters")
    print("=" * 78)

    if not top:
        print("NO FINALISTS — no combos passed TRAIN+VAL filters.")
    else:
        print(f"\n--- Top {len(top)} finalists (by min(TRAIN, VAL) PF) ---")
        for i, f in enumerate(top, 1):
            print(f"  [{i}] min_PF={f['min_pf']:.3f}  {label(f['params'])}")
            print(f"      TRAIN    {fmt_row(f['train'])}")
            print(f"      VALIDATE {fmt_row(f['validate'])}")

        # HOLDOUT context
        print()
        print("=" * 78)
        print("HOLDOUT (informational)")
        print("=" * 78)
        for i, f in enumerate(top, 1):
            key = (f["params"]["tp_dist"], f["params"]["sl_bricks"],
                   f["params"]["max_dist_bricks"], f["params"]["mk_mode"])
            hold = by_combo[key].get("HOLDOUT")
            f["holdout"] = hold
            if hold:
                print(f"  [{i}] {label(f['params'])}")
                print(f"      HOLDOUT  {fmt_row(hold)}")

        # FRESH deploy gate
        print()
        print("=" * 78)
        print(f"FRESH WINDOW — DEPLOY GATE  "
              f"(rules: PF>={FRESH_RULES['min_pf']}, "
              f"T>={FRESH_RULES['min_trades']}, net>0)")
        print("=" * 78)
        for i, f in enumerate(top, 1):
            key = (f["params"]["tp_dist"], f["params"]["sl_bricks"],
                   f["params"]["max_dist_bricks"], f["params"]["mk_mode"])
            fresh = by_combo[key].get("FRESH")
            f["fresh"] = fresh
            if not fresh:
                f["passes"] = False
                print(f"  [{i}] NO FRESH DATA  {label(f['params'])}")
                continue
            passes = (
                fresh["pf"]     >= FRESH_RULES["min_pf"]       and
                fresh["trades"] >= FRESH_RULES["min_trades"]   and
                fresh["net"]    >  FRESH_RULES["min_net_profit"]
            )
            f["passes"] = passes
            tag = "PASS" if passes else "FAIL"
            print(f"  [{i}] {tag}  {label(f['params'])}")
            print(f"      FRESH    {fmt_row(fresh)}")

        n_pass = sum(1 for f in top if f.get("passes"))
        print()
        print(f"FRESH VERDICT: {n_pass}/{len(top)} top finalists passed deploy gate.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump({
            "strategy":   "r035_kama_btc_long",
            "instrument": "BTC 150",
            "data_file":  RENKO_FILE,
            "locked_kama": {"k1": 5, "k2": 13, "k3": 60, "fast_sc": 2, "slow_sc": 30},
            "splits": {
                "TRAIN":    [TRAIN_START,    TRAIN_END],
                "VALIDATE": [VALIDATE_START, VALIDATE_END],
                "HOLDOUT":  [HOLDOUT_START,  HOLDOUT_END],
                "FRESH":    [FRESH_START,    FRESH_END],
            },
            "filters": {
                "min_train_trades":    MIN_TRAIN_TRADES,
                "min_validate_trades": MIN_VALIDATE_TRADES,
                "min_train_pf":        MIN_TRAIN_PF,
                "min_val_pf":          MIN_VAL_PF,
            },
            "ranking_metric": "min(TRAIN_PF, VAL_PF)",
            "fresh_rules":   FRESH_RULES,
            "total_combos":  len(combos),
            "total_backtests": len(tasks),
            "all_results":   results,
            "finalists":     top,
        }, f, indent=2, default=str)

    print(f"\nResults saved -> {OUTPUT_FILE}")
    print(f"Wallclock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
