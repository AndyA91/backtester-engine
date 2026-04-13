"""
GBPUSD 0.0005 — R033 KAMA Ribbon Length Sweep (parallel on 2 brackets)
======================================================================

Follow-up to gbpusd_kama_r3_mk_sweep (2026-04-08), which found two structurally
different finalists on GBPUSD 0.0005:

  Bracket A (Config #5, R/R-grinder):
    tp_bricks=6, sl_bricks=1, max_dist_bricks=12, mk_mode=direction
    HOLDOUT PF=3.28, 89T, 41.6% WR

  Bracket B (Config #9, EURAUD-style high-WR):
    tp_bricks=3, sl_bricks=1, max_dist_bricks=1000(off), mk_mode=strong
    HOLDOUT PF=3.50, 31T, 61.3% WR

Both used the visually-chosen KAMA ribbon (K1=5, K2=13, K3=60). KAMA lengths
were never swept — they were locked from a Pine session and carried forward.
This sweep tunes the KAMA dimension on BOTH brackets in parallel to:

  1. Test whether 5/13/60 is genuinely near-optimal or just a starting point
  2. See if both brackets respond similarly to KAMA changes (= bracket geometry
     drives the edge) or if a specific (KAMA, bracket) pairing dominates
  3. Validate that Config #9's high-WR profile is robust to KAMA tuning, not
     a single-point fluke

⚠️ R034 trap awareness
----------------------
This is structurally similar to R034 EURAUD MK-tune (lock the winner, sweep
indicator internals → zero-sum shuffle, no real edge). The differences:
  - KAMA lengths change WHAT the signal sees (different time horizons), not
    how it smooths. R034 only varied MK downstream smoothing knobs.
  - K=5/13/60 was a visual choice, not a swept optimum. Genuine room may exist.
  - Even a flat result is informative — confirms bracket drives the edge.

Sweep grid
----------
KAMA lengths (with K1 < K2 < K3 constraint):
  K1 (fast)  : [3, 5, 8]                  = 3
  K2 (mid)   : [10, 13, 18, 25]           = 4
  K3 (slow)  : [45, 55, 60, 75, 90]       = 5

After K1<K2<K3 filter: 60 valid (length triple) combos.

Locked: KAMA fast SC=2, slow SC=30. MK v4 internals at indicator defaults.
Gray exit OFF (bracket-only). Brick=0.0005. Bracket params per the two
finalists above.

Total: 60 KAMA × 2 brackets × 3 windows = 360 backtests.

Splits (3-way per R22)
----------------------
  TRAIN     data start -> 2025-09-30
  VALIDATE  2025-10-01 -> 2025-12-31
  HOLDOUT   2026-01-01 -> 2026-04-08  (sealed deploy gate)

Methodology (R26 + R34, locked before viewing HOLDOUT)
------------------------------------------------------
- Rank by min(TRAIN PF, VAL PF) — stability, not single-window peak
- Reject if HOLDOUT PF collapses >30% below stability
- Absolute HOLDOUT gate: PF≥1.5, T≥8, net>0
- Per R20: even if a "better" config emerges, MUST TV-validate before claiming improvement

Compute budget: 360 backtests, ~3-5 minutes on 20 workers.

Usage
-----
  python renko/gbpusd_kama_lengths_sweep.py
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

# ── Locked context ────────────────────────────────────────────────────────────
RENKO_FILE = "OANDA_GBPUSD, 1S renko 0.0005.csv"
BRICK_SIZE = 0.0005

# Two bracket finalists from gbpusd_kama_r3_mk_sweep (locked here, KAMA varies)
BRACKETS = {
    "A_grinder": {  # Config #5: R/R-grinder
        "tp_bricks":       6,
        "sl_bricks":       1,
        "max_dist_bricks": 12,
        "mk_mode":         "direction",
    },
    "B_highwr": {  # Config #9: EURAUD-style high-WR
        "tp_bricks":       3,
        "sl_bricks":       1,
        "max_dist_bricks": 1000,
        "mk_mode":         "strong",
    },
}

# ── 3-way splits ──────────────────────────────────────────────────────────────
TRAIN_START    = "2022-01-01"  # earlier than file start (2023-02-21); engine auto-clips
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

# ── KAMA grid ─────────────────────────────────────────────────────────────────
K1_GRID = [3, 5, 8]
K2_GRID = [10, 13, 18, 25]
K3_GRID = [45, 55, 60, 75, 90]


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

OUTPUT_FILE = ROOT / "ai_context" / "gbpusd_kama_lengths_results.json"


# ─── Worker ───────────────────────────────────────────────────────────────────
_worker_cache = {}


def _load_df():
    if "df" not in _worker_cache:
        df = load_renko_export(RENKO_FILE)
        add_renko_indicators(df)
        _worker_cache["df"] = df
        # KAMA cache keyed by (length, fast_sc, slow_sc) — accumulates across
        # combos, no clearing needed since lengths don't collide.
        strat._KAMA_CACHE.clear()
        strat._MK_CACHE.clear()
        strat.BRICK_SIZE = BRICK_SIZE
    return _worker_cache["df"]


def _run_one(args):
    bracket_name, bracket, kama, window_name, start, end = args
    df = _load_df()

    # Monkey-patch KAMA lengths for this combo. The KAMA cache is keyed by
    # length so different combos coexist in cache without interference.
    strat.K1_LEN = kama["k1"]
    strat.K2_LEN = kama["k2"]
    strat.K3_LEN = kama["k3"]

    strat_params = {
        "tp_dist":         bracket["tp_bricks"] * BRICK_SIZE,
        "sl_bricks":       bracket["sl_bricks"],
        "max_dist_bricks": bracket["max_dist_bricks"],
        "mk_mode":         bracket["mk_mode"],
    }

    df_sig = strat.generate_signals(df, **strat_params)
    cfg = make_cfg(start, end)
    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest_long_short(df_sig, cfg)
    pf = kpis.get("profit_factor", 0.0) or 0.0
    return {
        "bracket":    bracket_name,
        "kama":       kama,
        "window":     window_name,
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


def label(kama):
    return f"k1={kama['k1']:>2} k2={kama['k2']:>2} k3={kama['k3']:>2}"


def evaluate_bracket(bracket_name, bracket, results):
    by_combo = {}
    for r in results:
        if r["bracket"] != bracket_name:
            continue
        key = (r["kama"]["k1"], r["kama"]["k2"], r["kama"]["k3"])
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
            "bracket": bracket_name,
            "kama":    {"k1": key[0], "k2": key[1], "k3": key[2]},
            "train":        train,
            "validate":     val,
            "wr_delta":     wr_delta,
            "stability_pf": stability_pf,
        })

    finalists.sort(key=lambda f: (f["stability_pf"], f["validate"]["net"]),
                   reverse=True)
    top = finalists[:TOP_N_FINALISTS]

    print()
    print("=" * 78)
    print(f"BRACKET {bracket_name}  —  {len(finalists)} finalists passed TRAIN→VAL filters")
    print(f"  locked: tp={bracket['tp_bricks']}b sl={bracket['sl_bricks']}b "
          f"dist={'off' if bracket['max_dist_bricks']>=1000 else bracket['max_dist_bricks']} "
          f"mk={bracket['mk_mode']}")
    print("=" * 78)

    if not top:
        print("NO FINALISTS — TRAIN/VAL filters rejected all combos.")
        return top

    print(f"\n--- Top {len(top)} finalists (by R26 stability = min(TRAIN PF, VAL PF)) ---")
    for i, f in enumerate(top, 1):
        stab = "INF" if f["stability_pf"] >= 1e11 else f"{f['stability_pf']:.3f}"
        print(f"  [{i}] {label(f['kama'])}  stability={stab}")
        print(f"      TRAIN    {fmt_row(f['train'])}")
        print(f"      VALIDATE {fmt_row(f['validate'])}  Δwr={f['wr_delta']:+.1f}pp")

    print()
    print("-" * 78)
    print(f"HOLDOUT DEPLOY GATE  "
          f"(rules: PF>={HOLDOUT_RULES['min_pf']}, "
          f"T>={HOLDOUT_RULES['min_trades']}, net>0, no >30% collapse)")
    print("-" * 78)
    for i, f in enumerate(top, 1):
        key = (f["kama"]["k1"], f["kama"]["k2"], f["kama"]["k3"])
        hold = by_combo[key].get("HOLDOUT")
        f["holdout"] = hold
        if not hold:
            f["passes"] = False
            print(f"  [{i}] NO HOLDOUT DATA  {label(f['kama'])}")
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
        print(f"  [{i}] {tag}  {label(f['kama'])}{collapse_tag}")
        print(f"      HOLDOUT  {fmt_row(hold)}")

    n_pass = sum(1 for f in top if f.get("passes"))
    print()
    print(f"BRACKET {bracket_name} VERDICT: {n_pass}/{len(top)} finalists passed deploy gate.")
    return top


def main():
    t0 = time.time()

    # Build valid (k1, k2, k3) triples with K1 < K2 < K3
    kama_combos = []
    for k1 in K1_GRID:
        for k2 in K2_GRID:
            if k2 <= k1:
                continue
            for k3 in K3_GRID:
                if k3 <= k2:
                    continue
                kama_combos.append({"k1": k1, "k2": k2, "k3": k3})

    print("=" * 78)
    print("GBPUSD 0.0005 — R033 KAMA Length Sweep (2 brackets in parallel)")
    print("=" * 78)
    print(f"Strategy   : r033_kama_ribbon_3l_mk (KAMA SC LOCKED 2/30, MK locked at defaults)")
    print(f"Data       : {RENKO_FILE}")
    print(f"TRAIN      : {TRAIN_START} -> {TRAIN_END}  (auto-clipped to file start)")
    print(f"VALIDATE   : {VALIDATE_START} -> {VALIDATE_END}")
    print(f"HOLDOUT    : {HOLDOUT_START} -> data end  (DEPLOY GATE)")
    print(f"KAMA combos: {len(kama_combos)} valid triples (K1<K2<K3)")
    print(f"Brackets   : {list(BRACKETS.keys())}")
    n_tasks = len(kama_combos) * len(BRACKETS) * len(WINDOWS)
    print(f"Total      : {len(kama_combos)} KAMA × {len(BRACKETS)} brackets × {len(WINDOWS)} windows = {n_tasks} backtests")
    print(f"Workers    : {min(n_tasks, MAX_WORKERS)}")
    print("=" * 78)

    tasks = []
    for bracket_name, bracket in BRACKETS.items():
        for kama in kama_combos:
            for window_name, start, end in WINDOWS:
                tasks.append((bracket_name, bracket, kama, window_name, start, end))

    results = []
    done = 0
    with ProcessPoolExecutor(max_workers=min(len(tasks), MAX_WORKERS)) as pool:
        futures = {pool.submit(_run_one, t): t for t in tasks}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            done += 1
            if done % 60 == 0 or done == len(tasks):
                print(f"  [{done:>4}/{len(tasks)}] {r['bracket']:<10} "
                      f"{r['window']:<8} {fmt_row(r)} | {label(r['kama'])}",
                      flush=True)

    # Per-bracket evaluation
    all_top = {}
    for bracket_name, bracket in BRACKETS.items():
        all_top[bracket_name] = evaluate_bracket(bracket_name, bracket, results)

    # ── Cross-bracket comparison ────────────────────────────────────────────
    print()
    print("=" * 78)
    print("CROSS-BRACKET COMPARISON — does 5/13/60 hold up vs the swept space?")
    print("=" * 78)
    for bracket_name in BRACKETS:
        baseline_key = (5, 13, 60)
        baseline_results = [r for r in results
                            if r["bracket"] == bracket_name and
                            (r["kama"]["k1"], r["kama"]["k2"], r["kama"]["k3"]) == baseline_key]
        if not baseline_results:
            continue
        bw = {r["window"]: r for r in baseline_results}
        print(f"\n  {bracket_name} — baseline 5/13/60:")
        for w in ("TRAIN", "VALIDATE", "HOLDOUT"):
            if w in bw:
                print(f"    {w:<10} {fmt_row(bw[w])}")
        # Where does 5/13/60 rank in the stability finalist list?
        top = all_top[bracket_name]
        rank = next((i + 1 for i, f in enumerate(top)
                     if (f["kama"]["k1"], f["kama"]["k2"], f["kama"]["k3"]) == baseline_key),
                    None)
        if rank is None:
            print(f"    Baseline 5/13/60 NOT in top {TOP_N_FINALISTS} stability finalists.")
        else:
            print(f"    Baseline 5/13/60 ranks #{rank} in stability finalists.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump({
            "strategy":    "r033_kama_ribbon_3l_mk",
            "instrument":  "GBPUSD 0.0005",
            "data_file":   RENKO_FILE,
            "brackets":    BRACKETS,
            "kama_grid":   {"k1": K1_GRID, "k2": K2_GRID, "k3": K3_GRID,
                            "constraint": "K1<K2<K3"},
            "locked_kama_sc": {"fast": 2, "slow": 30},
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
            "total_kama_combos": len(kama_combos),
            "total_backtests":   len(tasks),
            "all_results":     results,
            "finalists_by_bracket": all_top,
        }, f, indent=2, default=str)

    print(f"\nResults saved -> {OUTPUT_FILE}")
    print(f"Wallclock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
