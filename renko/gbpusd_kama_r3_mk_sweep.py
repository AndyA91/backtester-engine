"""
GBPUSD — R033 KAMA Ribbon 3L + MK v4 Bracket Sweep across 3 brick sizes
========================================================================

Cross-instrument carryover from EURAUD R033 (config #2 TV-validated, see
`euraud_kama_r3_mk_results.md`). Per R20 (cross-instrument carryover is NOT
free): re-optimize on GBPUSD, do NOT import EURAUD's winning params.

Tests R033 on three GBPUSD brick sizes:
  - 0.0004  (4 pips)  — current GU001 live brick, fastest
  - 0.0005  (5 pips)  — middle, freshest data (through 2026-04-08)
  - 0.0008  (8 pips)  — slower, partially aligned with C5 (bigger-brick study)

Strategy hardcodes BRICK_SIZE = 0.0006 — we monkey-patch `strat.BRICK_SIZE`
inside each worker before calling generate_signals(). TP is parameterized in
BRICKS (not absolute price) so the geometry scales naturally per chart.

Splits (3-way per R22 — futures-discipline-grade)
-------------------------------------------------
  TRAIN     data start -> 2025-09-30
  VALIDATE  2025-10-01 -> 2025-12-31
  HOLDOUT   2026-01-01 -> data end  (sealed; opened ONCE for finalists)

GBPUSD data ends 2026-03-19 for 0.0004 and 0.0008 files (≈11 weeks of holdout)
and 2026-04-08 for 0.0005. No 4-way FRESH split — not enough room.

Sweep grid (320 combos × 3 windows × 3 bricks = 2,880 backtests)
----------------------------------------------------------------
  TP distance (BRICKS):      [3, 4, 5, 6, 8]                            = 5
  SL distance (bricks):      [1, 2, 3, 4]                                = 4
  Distance filter (bricks):  [4, 8, 12, 1000(off)]                       = 4
  MK v4 mode:                ["off", "direction", "block_flat", "strong"] = 4

KAMA periods LOCKED at 5/13/60 (same as EURAUD sweep — visual choice).
MK v4 internals locked at indicator defaults. Gray exit OFF (bracket only).

Decision rules (LOCKED before viewing HOLDOUT)
----------------------------------------------
Finalist filter (TRAIN -> VALIDATE):
  - VALIDATE trades >= 10
  - TRAIN trades >= 50
  - VALIDATE net > 0
  - TRAIN PF > 1.0
  - WR delta TRAIN -> VALIDATE in [-8, +12]pp

Ranking (per R26 stability rule — burned us on R031 + R033):
  - Sort by min(TRAIN PF, VALIDATE PF) descending  (stability, NOT single-window peak)
  - Tiebreak by VALIDATE net profit
  - HOLDOUT is NEVER used in selection — sealed deploy gate only

HOLDOUT deploy gate (the only window that decides deploy):
  - HOLDOUT trades >= 8
  - HOLDOUT net > 0
  - HOLDOUT PF >= 1.5
  - Per R26/R34 extension: also rejected if HOLDOUT PF collapses
    >30% below min(TRAIN PF, VAL PF) — that's the zero-sum-shuffle pattern

Compute budget: ~2,880 backtests, ~5–8 minutes on 20 workers.

Usage
-----
  python renko/gbpusd_kama_r3_mk_sweep.py
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

# ── Brick configs ─────────────────────────────────────────────────────────────
BRICK_FILES = {
    0.0004: "OANDA_GBPUSD, 1S renko 0.0004.csv",
    0.0005: "OANDA_GBPUSD, 1S renko 0.0005.csv",
    0.0008: "OANDA_GBPUSD, 1S renko 0.0008.csv",
}

# ── 3-way splits ──────────────────────────────────────────────────────────────
TRAIN_START    = "2022-01-01"  # earlier than all GBPUSD files; engine starts at first bar
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

# ── Sweep grid (TP in BRICKS for cross-brick scaling) ─────────────────────────
PARAM_GRID = {
    "tp_bricks":       [3, 4, 5, 6, 8],
    "sl_bricks":       [1, 2, 3, 4],
    "max_dist_bricks": [4, 8, 12, 1000],
    "mk_mode":         ["off", "direction", "block_flat", "strong"],
}


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


# ── Filters / thresholds ──────────────────────────────────────────────────────
MIN_TRAIN_TRADES        = 50
MIN_VALIDATE_TRADES     = 10
WR_DELTA_MIN            = -8.0
WR_DELTA_MAX            = 12.0
TOP_N_FINALISTS         = 10

HOLDOUT_RULES = {
    "min_pf":           1.5,
    "min_trades":       8,
    "min_net_profit":   0.0,
    "max_collapse_pct": 0.30,  # R34: reject if HOLDOUT PF collapses >30% from stability
}

OUTPUT_FILE = ROOT / "ai_context" / "gbpusd_kama_r3_mk_results.json"


# ─── Worker ───────────────────────────────────────────────────────────────────
_worker_cache = {}  # keyed by brick_size -> df


def _load_for_brick(brick_size):
    if brick_size not in _worker_cache:
        df = load_renko_export(BRICK_FILES[brick_size])
        add_renko_indicators(df)
        _worker_cache[brick_size] = df
        # Caches inside the strategy are KAMA-on-close + MK-on-OHLC; both keyed
        # only by KAMA length / MK params, NOT by brick. They MUST be cleared
        # whenever we switch brick (different df = different close series).
        strat._KAMA_CACHE.clear()
        strat._MK_CACHE.clear()
        _worker_cache["_active_brick"] = brick_size
    elif _worker_cache.get("_active_brick") != brick_size:
        strat._KAMA_CACHE.clear()
        strat._MK_CACHE.clear()
        _worker_cache["_active_brick"] = brick_size
    return _worker_cache[brick_size]


def _run_one_window(args):
    brick_size, params, window_name, start, end = args
    df = _load_for_brick(brick_size)

    # Monkey-patch strategy's BRICK_SIZE constant for this run.
    # Used inside generate_signals() for max_dist_bricks AND sl_offset.
    strat.BRICK_SIZE = brick_size

    # Convert tp_bricks -> tp_dist (price units), the strategy's native param
    strat_params = {
        "tp_dist":         params["tp_bricks"] * brick_size,
        "sl_bricks":       params["sl_bricks"],
        "max_dist_bricks": params["max_dist_bricks"],
        "mk_mode":         params["mk_mode"],
    }

    df_sig = strat.generate_signals(df, **strat_params)
    cfg = make_cfg(start, end)
    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest_long_short(df_sig, cfg)
    pf = kpis.get("profit_factor", 0.0) or 0.0
    return {
        "brick_size": brick_size,
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
            f"Exp={r['expectancy']:>6.4f}")


def label(p):
    dist = "off" if p['max_dist_bricks'] >= 1000 else f"d{p['max_dist_bricks']}"
    return (f"tp{p['tp_bricks']}b sl{p['sl_bricks']}b {dist} mk:{p['mk_mode']}")


def evaluate_brick(brick_size, results):
    by_combo = {}
    for r in results:
        if r["brick_size"] != brick_size:
            continue
        key = (r["params"]["tp_bricks"], r["params"]["sl_bricks"],
               r["params"]["max_dist_bricks"], r["params"]["mk_mode"])
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

        # R26 stability metric: min(TRAIN PF, VAL PF). Inf treated as 1e12.
        train_pf = 1e12 if math.isinf(train["pf"]) else train["pf"]
        val_pf   = 1e12 if math.isinf(val["pf"])   else val["pf"]
        stability_pf = min(train_pf, val_pf)

        finalists.append({
            "brick_size": brick_size,
            "params": {"tp_bricks": key[0], "sl_bricks": key[1],
                       "max_dist_bricks": key[2], "mk_mode": key[3]},
            "train":        train,
            "validate":     val,
            "wr_delta":     wr_delta,
            "stability_pf": stability_pf,
        })

    # R26: rank by stability (min TRAIN/VAL PF), tiebreak by VAL net
    finalists.sort(key=lambda f: (
        f["stability_pf"],
        f["validate"]["net"],
    ), reverse=True)

    top = finalists[:TOP_N_FINALISTS]

    print()
    print("=" * 78)
    print(f"BRICK {brick_size}  —  {len(finalists)} finalists passed TRAIN→VAL filters")
    print("=" * 78)

    if not top:
        print("NO FINALISTS — TRAIN/VAL filters rejected all combos for this brick.")
        return top

    print(f"\n--- Top {len(top)} finalists (by R26 stability = min(TRAIN PF, VAL PF)) ---")
    for i, f in enumerate(top, 1):
        stab = "INF" if f["stability_pf"] >= 1e11 else f"{f['stability_pf']:.3f}"
        print(f"  [{i}] {label(f['params'])}  stability={stab}")
        print(f"      TRAIN    {fmt_row(f['train'])}")
        print(f"      VALIDATE {fmt_row(f['validate'])}  Δwr={f['wr_delta']:+.1f}pp")

    # HOLDOUT deploy gate
    print()
    print("-" * 78)
    print(f"HOLDOUT DEPLOY GATE  "
          f"(rules: PF>={HOLDOUT_RULES['min_pf']}, "
          f"T>={HOLDOUT_RULES['min_trades']}, net>0)")
    print("-" * 78)
    for i, f in enumerate(top, 1):
        key = (f["params"]["tp_bricks"], f["params"]["sl_bricks"],
               f["params"]["max_dist_bricks"], f["params"]["mk_mode"])
        hold = by_combo[key].get("HOLDOUT")
        f["holdout"] = hold
        if not hold:
            f["passes"] = False
            print(f"  [{i}] NO HOLDOUT DATA  {label(f['params'])}")
            continue
        # R34 extension: reject if HOLDOUT PF collapses >30% below stability PF
        # (catches zero-sum-shuffle "winners" that look good on TRAIN/VAL)
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
    print(f"BRICK {brick_size} VERDICT: {n_pass}/{len(top)} finalists passed deploy gate.")
    return top


def main():
    t0 = time.time()

    keys = list(PARAM_GRID.keys())
    combos = [dict(zip(keys, v)) for v in itertools.product(*PARAM_GRID.values())]

    print("=" * 78)
    print("GBPUSD — R033 KAMA Ribbon 3L + MK v4 Bracket Sweep (3 bricks)")
    print("=" * 78)
    print(f"Strategy   : r033_kama_ribbon_3l_mk (KAMA LOCKED 5/13/60)")
    print(f"Bricks     : {list(BRICK_FILES.keys())}")
    print(f"TRAIN      : {TRAIN_START} -> {TRAIN_END}  (data start auto-detected per file)")
    print(f"VALIDATE   : {VALIDATE_START} -> {VALIDATE_END}")
    print(f"HOLDOUT    : {HOLDOUT_START} -> data end  (DEPLOY GATE)")
    n_tasks = len(combos) * len(WINDOWS) * len(BRICK_FILES)
    print(f"Combos     : {len(combos)} × {len(WINDOWS)} windows × {len(BRICK_FILES)} bricks = {n_tasks} backtests")
    print(f"Workers    : {min(n_tasks, MAX_WORKERS)}")
    print("=" * 78)

    tasks = []
    for brick_size in BRICK_FILES:
        for p in combos:
            for window_name, start, end in WINDOWS:
                tasks.append((brick_size, p, window_name, start, end))

    results = []
    done = 0
    with ProcessPoolExecutor(max_workers=min(len(tasks), MAX_WORKERS)) as pool:
        futures = {pool.submit(_run_one_window, t): t for t in tasks}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            done += 1
            if done % 200 == 0 or done == len(tasks):
                print(f"  [{done:>4}/{len(tasks)}] b={r['brick_size']} "
                      f"{r['window']:<8} {fmt_row(r)} | {label(r['params'])}")

    # Per-brick evaluation
    all_top = {}
    for brick_size in BRICK_FILES:
        all_top[brick_size] = evaluate_brick(brick_size, results)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump({
            "strategy":    "r033_kama_ribbon_3l_mk",
            "instrument":  "GBPUSD",
            "brick_files": BRICK_FILES,
            "locked_kama": {"k1": 5, "k2": 13, "k3": 60, "fast_sc": 2, "slow_sc": 30},
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
            "holdout_rules":   HOLDOUT_RULES,
            "total_combos":    len(combos),
            "total_backtests": len(tasks),
            "all_results":     results,
            "finalists_by_brick": {str(k): v for k, v in all_top.items()},
        }, f, indent=2, default=str)

    print(f"\nResults saved -> {OUTPUT_FILE}")
    print(f"Wallclock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
