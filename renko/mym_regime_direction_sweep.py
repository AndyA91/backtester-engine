"""
MYM Brick 30 — MYM003 Direction Confirmation Sweep
====================================================

Hypothesis: Adding an independent trend-direction indicator as a gate on the
initial regime-flip entry should filter false flip-bar reads where the brick
happens to be counter-trend. The regime is currently direction-agnostic
(TREND vs RANGE only); direction comes solely from the flip bar's brick.

Grid:
  3 base configs (C1/C2/C3 from ER-gated sweep, locked at their best erGate)
  × 12 direction filter variants:
    - none (baseline)
    - kama (kama_slope sign)
    - ema9, ema21, ema50 (close vs EMA)
    - di (plus_di vs minus_di)
    - macd (macd_hist sign)
    - st (supertrend direction)
    - net_bricks with lookback 5, 10, 14, 20

  3 × 12 = 36 combos × 3 windows = 108 backtests

Usage
-----
  python renko/mym_regime_direction_sweep.py
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

RENKO_FILE = "CBOT_MINI_MYM1!, 1S ETH renko 30.csv"

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

# Locked base configs from ER-gated sweep (Sweep 5 best settings)
BASE_CONFIGS = [
    {"name": "C1_er14s1_t055r015",
     "er_lookback": 14, "er_smooth": 1,
     "trend_thresh": 0.55, "range_thresh": 0.15,
     "persistence": 2, "tp_bricks": 0, "min_streak": 1,
     "reentry_streak": 1, "reentry_cooldown": 0,
     "max_reentries_per_regime": 999, "min_reentry_er": 0.65},
    {"name": "C2_er20s1_t045r025",
     "er_lookback": 20, "er_smooth": 1,
     "trend_thresh": 0.45, "range_thresh": 0.25,
     "persistence": 2, "tp_bricks": 0, "min_streak": 1,
     "reentry_streak": 1, "reentry_cooldown": 0,
     "max_reentries_per_regime": 999, "min_reentry_er": 0.65},
    {"name": "C3_er20s1_t045r015",
     "er_lookback": 20, "er_smooth": 1,
     "trend_thresh": 0.45, "range_thresh": 0.15,
     "persistence": 2, "tp_bricks": 0, "min_streak": 1,
     "reentry_streak": 1, "reentry_cooldown": 0,
     "max_reentries_per_regime": 999, "min_reentry_er": 0.65},
]

# Direction filter variants: (dir_filter, dir_ema_len, net_brick_lookback)
DIR_VARIANTS = [
    ("none",       50,  10),   # baseline — no direction filter
    ("kama",       50,  10),   # kama_slope sign
    ("ema",         9,  10),   # close > ema9
    ("ema",        21,  10),   # close > ema21
    ("ema",        50,  10),   # close > ema50
    ("di",         50,  10),   # plus_di > minus_di
    ("macd",       50,  10),   # macd_hist sign
    ("st",         50,  10),   # supertrend direction
    ("net_bricks", 50,   5),   # net bricks lookback 5
    ("net_bricks", 50,  10),   # net bricks lookback 10
    ("net_bricks", 50,  14),   # net bricks lookback 14
    ("net_bricks", 50,  20),   # net bricks lookback 20
]

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

OUTPUT_FILE = ROOT / "ai_context" / "mym_regime_direction_results.json"


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


_worker_cache = {}


def _run_one_window(args):
    params, window_name, start, end = args

    if "df" not in _worker_cache:
        df = load_renko_export(RENKO_FILE)
        add_renko_indicators(df)
        _worker_cache["df"] = df

    df = _worker_cache["df"]
    strat._ER_CACHE.clear()

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


def dir_label(p):
    df = p.get("dir_filter", "none")
    if df == "none":
        return "no_filter"
    if df == "ema":
        return f"ema{p.get('dir_ema_len', 50)}"
    if df == "net_bricks":
        return f"netB{p.get('net_brick_lookback', 10)}"
    return df


def label(p):
    return f"{p['name']} {dir_label(p)}"


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
    print(f"MYM Brick 30 (Direction Confirmation)  —  {len(finalists)} finalists passed TRAIN->VAL")
    print("=" * 78)

    if not top:
        print("NO FINALISTS")
        return top

    print(f"\n--- Top {len(top)} finalists (by R26 stability) ---")
    for i, f in enumerate(top, 1):
        stab = "INF" if f["stability_pf"] >= 1e11 else f"{f['stability_pf']:.3f}"
        print(f"  [{i}] {label(f['params'])}  stability={stab}")
        print(f"      TRAIN    {fmt_row(f['train'])}")
        print(f"      VALIDATE {fmt_row(f['validate'])}  dwr={f['wr_delta']:+.1f}pp")

    print()
    print("-" * 78)
    print("HOLDOUT DEPLOY GATE")
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

    # Direction filter comparison table
    print()
    print("-" * 78)
    print("DIRECTION FILTER COMPARISON (HOLDOUT, per base config)")
    print("-" * 78)
    holdout_by_key = {}
    for key, wmap in by_combo.items():
        h = wmap.get("HOLDOUT")
        if not h:
            continue
        p = dict(key)
        holdout_by_key[(p["name"], p.get("dir_filter", "none"),
                        p.get("dir_ema_len", 50), p.get("net_brick_lookback", 10))] = h

    for base in BASE_CONFIGS:
        name = base["name"]
        print(f"\n  {name}:")
        print(f"    {'filter':>12}  {'PF':>7}  {'T':>4}  {'WR':>5}  {'Net':>9}  {'DD':>7}")
        for df_name, ema_len, nb_lb in DIR_VARIANTS:
            h = holdout_by_key.get((name, df_name, ema_len, nb_lb))
            if h:
                filt_label = dir_label({"dir_filter": df_name, "dir_ema_len": ema_len,
                                        "net_brick_lookback": nb_lb})
                print(f"    {filt_label:>12}  {fmt_pf(h['pf']):>7}  {h['trades']:>4d}  "
                      f"{h['win_rate']:>5.1f}%  ${h['net']:>8.0f}  {h['max_dd_pct']:>6.2f}%")

    return top


def main():
    t0 = time.time()

    combos = []
    for base in BASE_CONFIGS:
        for df_name, ema_len, nb_lb in DIR_VARIANTS:
            combo = dict(base)
            combo["dir_filter"]         = df_name
            combo["dir_ema_len"]        = ema_len
            combo["net_brick_lookback"] = nb_lb
            combos.append(combo)

    n_tasks = len(combos) * len(WINDOWS)

    print("=" * 78)
    print("MYM Brick 30 — MYM003 Direction Confirmation Sweep")
    print("=" * 78)
    print(f"Strategy   : mym003_regime_flip (regime flip + direction filter)")
    print(f"Data       : {RENKO_FILE}")
    print(f"Base configs: {len(BASE_CONFIGS)} (locked from ER-gated sweep)")
    print(f"Dir variants: {len(DIR_VARIANTS)}")
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
            "strategy":    "mym003_regime_flip_direction",
            "instrument":  "MYM",
            "brick_size":  30,
            "renko_file":  RENKO_FILE,
            "base_configs": BASE_CONFIGS,
            "dir_variants": [{"dir_filter": d, "dir_ema_len": e, "net_brick_lookback": n}
                             for d, e, n in DIR_VARIANTS],
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
