"""
MYM Brick 30 — MYM003 Regime Flip + Re-Entry Sweep
====================================================

Follow-up to baseline sweep. The regime-flip strategy only enters once per
RANGE→TREND flip — it often sits flat for the rest of a long trending regime
after being stopped out. This sweep tests re-entry within the same regime.

New logic:
  1. Initial entry on RANGE→TREND flip (same as baseline)
  2. Bias direction LOCKED at flip bar (up-brick = long bias for the regime)
  3. After stop-out within same TREND regime, re-enter in bias direction after
     `reentry_streak` consecutive bricks + `reentry_cooldown` bars
  4. Bias resets when regime leaves TREND

Grid: 3 winning base configs × reentry params
  Base configs (locked from baseline sweep top 3):
    #1: er14 s1 t0.55 r0.15 p2
    #2: er20 s1 t0.45 r0.25 p2
    #3: er20 s1 t0.45 r0.15 p2
  reentry_streak:   [0, 1, 2, 3, 4]     (0 = baseline, no re-entry)
  reentry_cooldown: [0, 3, 5, 10]

  3 × 5 × 4 = 60 combos × 3 windows = 180 backtests.

Decision rules: same as baseline sweep.
Ranking: min(TRAIN PF, VAL PF) stability (R26).
HOLDOUT gate: PF >= 1.5, T >= 8, net > 0, collapse < 30%.

Usage
-----
  python renko/mym_regime_reentry_sweep.py
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

MAX_WORKERS = min(_RAW_WORKERS, 6)

import mym003_regime_flip as strat

# ── Data ─────────────────────────────────────────────────────────────────────
RENKO_FILE = "CBOT_MINI_MYM1!, 1S ETH renko 30.csv"

# ── 3-way splits ─────────────────────────────────────────────────────────────
TRAIN_START    = "2022-10-16"
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

# ── Locked base configs (3 top finalists from baseline sweep) ────────────────
BASE_CONFIGS = [
    {
        "name":         "C1_er14s1_t055r015",
        "er_lookback":  14, "er_smooth": 1,
        "trend_thresh": 0.55, "range_thresh": 0.15,
        "persistence":  2, "tp_bricks": 0, "min_streak": 1,
    },
    {
        "name":         "C2_er20s1_t045r025",
        "er_lookback":  20, "er_smooth": 1,
        "trend_thresh": 0.45, "range_thresh": 0.25,
        "persistence":  2, "tp_bricks": 0, "min_streak": 1,
    },
    {
        "name":         "C3_er20s1_t045r015",
        "er_lookback":  20, "er_smooth": 1,
        "trend_thresh": 0.45, "range_thresh": 0.15,
        "persistence":  2, "tp_bricks": 0, "min_streak": 1,
    },
]

# ── Re-entry dims ────────────────────────────────────────────────────────────
REENTRY_STREAK   = [0, 1, 2, 3, 4]
REENTRY_COOLDOWN = [0, 3, 5, 10]

# ── Filters / thresholds ────────────────────────────────────────────────────
MIN_TRAIN_TRADES    = 30
MIN_VALIDATE_TRADES = 10
WR_DELTA_MIN        = -10.0
WR_DELTA_MAX        = 15.0
TOP_N_FINALISTS     = 15

HOLDOUT_RULES = {
    "min_pf":           1.5,
    "min_trades":       8,
    "min_net_profit":   0.0,
    "max_collapse_pct": 0.30,
}

OUTPUT_FILE = ROOT / "ai_context" / "mym_regime_reentry_results.json"


def make_cfg(start, end):
    return BacktestConfig(
        initial_capital=strat.INITIAL_CAPITAL,
        commission_pct=strat.COMMISSION_PCT,
        slippage_ticks=0,
        qty_type="fixed",
        qty_value=0.5,
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
    strat._ER_CACHE.clear()

    # Remove 'name' from params when passing to strategy
    gen_params = {k: v for k, v in params.items() if k != "name"}
    df_sig = strat.generate_signals(df.copy(), **gen_params)

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
    return (f"{p['name']} rs{p['reentry_streak']} rcd{p['reentry_cooldown']}")


def evaluate(results):
    by_combo = {}
    for r in results:
        key = tuple(sorted((k, v) for k, v in r["params"].items()))
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
    print(f"MYM Brick 30 (Re-Entry)  —  {len(finalists)} finalists passed TRAIN→VAL")
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
    print(f"HOLDOUT DEPLOY GATE")
    print("-" * 78)
    for i, f in enumerate(top, 1):
        key = tuple(sorted((k, v) for k, v in f["params"].items()))
        hold = by_combo[key].get("HOLDOUT")
        f["holdout"] = hold
        if not hold:
            f["passes"] = False
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

    # Re-entry analysis
    print()
    print("-" * 78)
    print("RE-ENTRY IMPACT (HOLDOUT only, by base config + reentry_streak)")
    print("-" * 78)
    holdout_by_combo = {}
    for key, wmap in by_combo.items():
        h = wmap.get("HOLDOUT")
        if not h:
            continue
        p = dict(key)
        holdout_by_combo[(p["name"], p["reentry_streak"], p["reentry_cooldown"])] = h

    for base in BASE_CONFIGS:
        name = base["name"]
        print(f"\n  {name}:")
        for rs in REENTRY_STREAK:
            best_net = -999999
            best_rcd = None
            best_row = None
            for rcd in REENTRY_COOLDOWN:
                h = holdout_by_combo.get((name, rs, rcd))
                if h and h["net"] > best_net:
                    best_net = h["net"]
                    best_rcd = rcd
                    best_row = h
            if best_row:
                print(f"    rs{rs}: best rcd={best_rcd}  PF={best_row['pf']:.2f} T={best_row['trades']:3d} "
                      f"WR={best_row['win_rate']:.1f}% Net=${best_row['net']:.0f}")

    return top


def main():
    t0 = time.time()

    # Build combos: cross base configs with reentry dims
    combos = []
    for base in BASE_CONFIGS:
        for rs, rcd in itertools.product(REENTRY_STREAK, REENTRY_COOLDOWN):
            combo = dict(base)
            combo["reentry_streak"]   = rs
            combo["reentry_cooldown"] = rcd
            combos.append(combo)

    n_tasks = len(combos) * len(WINDOWS)

    print("=" * 78)
    print("MYM Brick 30 — MYM003 Regime Flip + Re-Entry Sweep")
    print("=" * 78)
    print(f"Strategy   : mym003_regime_flip (regime flip + re-entry in locked bias)")
    print(f"Data       : {RENKO_FILE}")
    print(f"TRAIN      : {TRAIN_START} -> {TRAIN_END}")
    print(f"VALIDATE   : {VALIDATE_START} -> {VALIDATE_END}")
    print(f"HOLDOUT    : {HOLDOUT_START} -> data end")
    print(f"Base configs: {len(BASE_CONFIGS)}")
    print(f"Re-entry grid: streak={REENTRY_STREAK} × cooldown={REENTRY_COOLDOWN}")
    print(f"Combos     : {len(combos)}")
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
            if done % 20 == 0 or done == len(tasks):
                print(f"  [{done:>3}/{len(tasks)}] {r['window']:<8} "
                      f"{fmt_row(r)} | {label(r['params'])}")

    top = evaluate(results)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump({
            "strategy":    "mym003_regime_flip_reentry",
            "instrument":  "MYM",
            "brick_size":  30,
            "renko_file":  RENKO_FILE,
            "base_configs": BASE_CONFIGS,
            "reentry_streak":   REENTRY_STREAK,
            "reentry_cooldown": REENTRY_COOLDOWN,
            "qty_value":   0.5,
            "commission_pct": strat.COMMISSION_PCT,
            "splits": {
                "TRAIN":    [TRAIN_START,    TRAIN_END],
                "VALIDATE": [VALIDATE_START, VALIDATE_END],
                "HOLDOUT":  [HOLDOUT_START,  HOLDOUT_END],
            },
            "total_combos":    len(combos),
            "total_backtests": n_tasks,
            "holdout_rules":   HOLDOUT_RULES,
            "all_results":     results,
            "finalists":       top,
        }, f, indent=2, default=str)

    print(f"\nResults saved -> {OUTPUT_FILE}")
    print(f"Wallclock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
