"""
EURAUD 0.0006 — R033 KAMA Ribbon 3L + MK v4 Bracket Sweep (4-way split)

The interactive Pine session produced two near-deployable configurations:
  - KAMA R5 (tight SL=1):  Full PF=2.67, FRESH PF=0.59  (FAILED FRESH)
  - KAMA R3 (wider SL):    Full PF=1.58, FRESH PF=1.33  (SURVIVED FRESH)

Both used the same 5-line / 3-line KAMA structure with TP/SL bracket. The
trade-off is clear: tight SL = higher PF but blows up in bad regimes; wider
SL = lower PF but robust. This sweep tests whether a SL between 1 and 4
bricks finds a sweet spot, and whether MK v4 + distance filters preserve
the asymmetric W/L of tight SL while filtering out the chop entries.

Data has been REFRESHED (TV re-export 2026-04-08, includes data through
April 8). Per L16, brick boundaries have shifted vs the prior R030/R031/R032
files — historical numbers won't match exactly, but the new file is the
canonical source going forward.

Splits (4-way per R22 + R22 extension)
--------------------------------------
  TRAIN     2023-07-20 -> 2025-09-30
  VALIDATE  2025-10-01 -> 2025-12-31
  HOLDOUT   2026-01-01 -> 2026-03-14   (used in 3 prior sweeps)
  FRESH     2026-03-15 -> data end     (NEW sealed window — DEPLOY GATE)

The deploy decision is on FRESH. HOLDOUT is reported for context comparison
with the prior 3 sweeps, but contamination history makes it untrustworthy
as a sole gate.

Sweep grid (240 combos × 4 windows = 960 backtests)
---------------------------------------------------
  TP distance (price units): [0.0018, 0.0024, 0.0030, 0.0036, 0.0048]   = 5
  SL distance (bricks):      [1, 2, 3, 4]                                = 4
  Distance filter (bricks):  [4, 8, 12, 1000(off)]                       = 4
  MK v4 mode:                ["off", "direction", "block_flat", "strong"] = 4

KAMA periods LOCKED at 5/13/60 (visual choice).
MK v4 internal params LOCKED at indicator defaults.
SL=1 / no gray exit (gray was 0 fires in TV with bracket dominance).

Decision rules (LOCKED before viewing FRESH)
--------------------------------------------
Finalist filter (TRAIN -> VALIDATE):
  - VALIDATE trades >= 10
  - TRAIN trades >= 50
  - VAL net > 0
  - WR delta TRAIN -> VALIDATE in [-8, +12]pp

Holdout sanity (informational only — not a deploy gate):
  - HOLDOUT trades >= 8
  - HOLDOUT net > 0

FRESH deploy gate (this is the only window that decides deploy):
  - FRESH trades >= 8
  - FRESH net > 0
  - FRESH PF >= 1.5

Compute budget: ~960 backtests, <3 minutes on 20 workers.

Usage
-----
  python renko/euraud_kama_r3_mk_sweep.py
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

import r033_kama_ribbon_3l_mk as strat

RENKO_FILE = "OANDA_EURAUD, 1S renko 0.0006.csv"

# ── 4-way splits ──────────────────────────────────────────────────────────────
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


# ── Filters / thresholds ────────────────────────────────────────────────────
MIN_TRAIN_TRADES        = 50
MIN_VALIDATE_TRADES     = 10
WR_DELTA_MIN            = -8.0
WR_DELTA_MAX            = 12.0
TOP_N_FINALISTS         = 10

# FRESH deploy gate (the only window that decides deploy claim)
FRESH_RULES = {
    "min_pf":         1.5,
    "min_trades":     8,
    "min_net_profit": 0.0,
}

OUTPUT_FILE = ROOT / "ai_context" / "euraud_kama_r3_mk_results.json"


# ─── Worker ───────────────────────────────────────────────────────────────────
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
    dist = "off" if p['max_dist_bricks'] >= 1000 else f"d{p['max_dist_bricks']}"
    return f"tp{p['tp_dist']:.4f} sl{p['sl_bricks']}b {dist} mk:{p['mk_mode']}"


def main():
    t0 = time.time()

    grid = strat.PARAM_GRID
    keys = list(grid.keys())
    combos = [dict(zip(keys, v)) for v in itertools.product(*grid.values())]

    print("=" * 78)
    print("EURAUD 0.0006 — R033 KAMA Ribbon 3L + MK v4 Bracket Sweep")
    print("=" * 78)
    print(f"Strategy   : r033_kama_ribbon_3l_mk (KAMA LOCKED 5/13/60)")
    print(f"Data       : {RENKO_FILE}")
    print(f"TRAIN      : {TRAIN_START} -> {TRAIN_END}")
    print(f"VALIDATE   : {VALIDATE_START} -> {VALIDATE_END}")
    print(f"HOLDOUT    : {HOLDOUT_START} -> {HOLDOUT_END}  (4th use)")
    print(f"FRESH      : {FRESH_START} -> data end  (DEPLOY GATE)")
    print(f"Combos     : {len(combos)} × 4 windows = {len(combos)*4} backtests")
    print(f"Workers    : {min(len(combos)*4, MAX_WORKERS)}")
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

    # Pivot
    by_combo = {}
    for r in results:
        key = (r["params"]["tp_dist"], r["params"]["sl_bricks"],
               r["params"]["max_dist_bricks"], r["params"]["mk_mode"])
        by_combo.setdefault(key, {})[r["window"]] = r

    # ── Finalist selection (TRAIN/VAL only — HOLDOUT/FRESH sealed) ──────────
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
            "params": {"tp_dist": key[0], "sl_bricks": key[1],
                       "max_dist_bricks": key[2], "mk_mode": key[3]},
            "train":    train,
            "validate": val,
            "wr_delta": wr_delta,
        })

    # Rank by VAL PF, tiebreak by VAL net
    finalists.sort(key=lambda f: (
        f["validate"]["pf"] if not math.isinf(f["validate"]["pf"]) else 1e12,
        f["validate"]["net"],
    ), reverse=True)

    top = finalists[:TOP_N_FINALISTS]

    print()
    print("=" * 78)
    print(f"TRAIN -> VALIDATE: {len(finalists)} finalists passed filters")
    print("=" * 78)

    if not top:
        print("NO FINALISTS — TRAIN/VAL filters rejected all combos.")
    else:
        print(f"\n--- Top {len(top)} finalists (by VAL PF) ---")
        for i, f in enumerate(top, 1):
            print(f"  [{i}] {label(f['params'])}")
            print(f"      TRAIN    {fmt_row(f['train'])}")
            print(f"      VALIDATE {fmt_row(f['validate'])}  Δwr={f['wr_delta']:+.1f}pp")

        # ── HOLDOUT context (informational only) ────────────────────────────
        print()
        print("=" * 78)
        print("HOLDOUT CONTEXT (4th use — informational only, NOT deploy gate)")
        print("=" * 78)
        for i, f in enumerate(top, 1):
            key = (f["params"]["tp_dist"], f["params"]["sl_bricks"],
                   f["params"]["max_dist_bricks"], f["params"]["mk_mode"])
            hold = by_combo[key].get("HOLDOUT")
            f["holdout"] = hold
            if hold:
                print(f"  [{i}] {label(f['params'])}")
                print(f"      HOLDOUT  {fmt_row(hold)}")

        # ── FRESH evaluation (the deploy gate) ──────────────────────────────
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
        print(f"FRESH VERDICT: {n_pass}/{len(top)} finalists passed deploy gate.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump({
            "strategy":   "r033_kama_ribbon_3l_mk",
            "instrument": "EURAUD 0.0006",
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
                "wr_delta_min":        WR_DELTA_MIN,
                "wr_delta_max":        WR_DELTA_MAX,
            },
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
