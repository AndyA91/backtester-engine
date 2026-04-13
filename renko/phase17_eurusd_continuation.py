"""
Phase 17 — EURUSD 0.0006 Trend Continuation Overlay (Stage 1)

Purpose
-------
R029 Band Bounce (LIVE 2026-04-06) is a pure mean-reversion strategy with an
ADX < 25 ceiling. By design it sits idle (or fires the wrong side at small
size) during sustained trends. This sweep tests whether a brick-streak-based
trend continuation entry has standalone edge on EURUSD 0.0006 — the same
brick that R029 trades — so we can later combine it with R029 as a
complementary overlay (Stage 2).

Per meta_rule R3 (memory): test pure brick streak as the baseline FIRST before
stacking indicators. The optional ADX floor here is the natural complement
to R029's ADX ceiling — together the two strategies would partition the
regime space (R029 in chop, continuation in trend).

Per meta_rule R20: cross-instrument carryover from MYM_STREAK is NOT free.
The MYM brick-streak edge has to earn its keep on EURUSD 0.0006 specifically.

Per meta_rule R22: R029 is live, so 3-way TRAIN/VALIDATE/HOLDOUT — never
2-way IS/OOS — when touching live strategy.

Three-way data split
--------------------
  TRAIN     2023-01-02 -> 2025-09-30   (matches R029's IS boundary)
  VALIDATE  2025-10-01 -> 2025-12-31   (~3 months)
  HOLDOUT   2026-01-01 -> 2026-03-25   (~12 weeks, sealed during sweep)

NOTE: data ends 2026-03-25. The April 7 trend in the user's screenshot is
NOT in the holdout. If Stage 1 finds a winner we should refresh data before
finalising deploy.

Workflow
--------
1. For each (streak, cooldown, adx_floor) combo, run on TRAIN + VALIDATE
2. Pick top 5 finalists by VALIDATE PF subject to filters:
   - >= 30 VALIDATE trades
   - TRAIN -> VALIDATE WR delta >= -5pp (no major regime decay)
   - TRAIN -> VALIDATE WR delta <= +15pp (R23 — large jumps = regime artifact)
3. Run finalists on HOLDOUT exactly ONCE
4. Decision rules (locked BEFORE looking at HOLDOUT, per R22):
   - HOLDOUT PF >= 4
   - HOLDOUT WR >= 50%
   - HOLDOUT trades >= 30
   - HOLDOUT net profit > 0

Slices the DataFrame BEFORE generating signals so the stateful `pos` variable
in the signal generator can't be contaminated by phantom out-of-window trades
(handles look_ahead_redflags.md L2 / meta_rule R18).

Usage
-----
    python renko/phase17_eurusd_continuation.py
"""

import contextlib
import io
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import BacktestConfig, run_backtest_long_short
from renko.data import load_renko_export
from renko.indicators import add_renko_indicators
from renko.config import MAX_WORKERS

RENKO_FILE = "OANDA_EURUSD, 1S renko 0.0006.csv"

# ── Data splits (3-way per R22) ──────────────────────────────────────────────
TRAIN_START    = "2023-01-02"
TRAIN_END      = "2025-09-30"
VALIDATE_START = "2025-10-01"
VALIDATE_END   = "2025-12-31"
HOLDOUT_START  = "2026-01-01"
HOLDOUT_END    = "2026-03-26"  # exclusive — captures all of March 25

# ── Backtest config — matches R029 Pine settings exactly ────────────────────
BACKTEST_CONFIG = BacktestConfig(
    initial_capital=1000.0,
    commission_pct=0.0046,
    slippage_ticks=0,
    qty_type="fixed",
    qty_value=1000.0,
    pyramiding=1,
    start_date="2000-01-01",
    end_date="2099-12-31",
    take_profit_pct=0.0,
    stop_loss_pct=0.0,
)

OUTPUT_FILE = ROOT / "ai_context" / "phase17_eurusd_continuation_results.json"

# ── Param grid ───────────────────────────────────────────────────────────────
# Per R3: test pure streak baseline (adx_floor=0) AND with-trend-filter
# variants. Streak values bracket the MYM brick-streak sweet spot (12) plus
# wider edges. ADX floor 25 is the natural complement to R029's ADX < 25.
PARAM_GRID = {
    "min_brick_streak": [6, 8, 10, 12, 15],
    "cooldown":         [5, 10, 20],
    "adx_floor":        [0, 20, 25, 30],   # 0 = pure streak (R3 baseline)
}

# ── Filters / thresholds ─────────────────────────────────────────────────────
MIN_TRADES_PER_WINDOW       = 10    # workers reject configs with too few trades
MIN_VALIDATE_TRADES_FINAL   = 30    # finalist eligibility
MAX_TRAIN_VAL_WR_DROP       = 5.0   # WR may drop at most 5pp TRAIN -> VALIDATE
MAX_TRAIN_VAL_WR_JUMP       = 15.0  # WR may rise at most 15pp (R23)
TOP_N_FINALISTS             = 5

# ── Holdout deployment decision rules (locked BEFORE running holdout) ───────
RULES = {
    "min_pf":          4.0,
    "min_wr":          50.0,
    "min_trades":      30,
    "min_net_profit":  0.0,
}


def generate_signals(df, min_brick_streak, cooldown, adx_floor):
    """Pure brick-streak entry with optional ADX floor.

    LONG  : last `min_brick_streak` bricks all UP, prior brick was DOWN
            (streak just completed), AND adx[i] >= adx_floor (if floor > 0)
    SHORT : symmetric on down bricks
    EXIT  : first opposing brick (R1)

    Caller must pass an already-sliced DataFrame so the `pos` state stays clean
    (R18 — stateful generators must gate ALL state mutations on bar_in_range,
    achieved here by slicing upstream).
    """
    brick_up = df["brick_up"].values
    adx = df["adx"].values  # pre-shifted by add_renko_indicators
    n = len(df)

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    warmup = max(min_brick_streak + 2, 30)  # 30 covers ADX(14) seeding
    last_exit_bar = -999_999
    pos = 0

    for i in range(warmup, n):
        # Exit on first opposing brick
        if pos == 1 and not brick_up[i]:
            long_exit[i] = True
            pos = 0
            last_exit_bar = i
        elif pos == -1 and brick_up[i]:
            short_exit[i] = True
            pos = 0
            last_exit_bar = i

        if pos != 0:
            continue
        if (i - last_exit_bar) < cooldown:
            continue

        # ADX floor gate (0 = off; only fire continuation in trending regime)
        if adx_floor > 0:
            a = adx[i]
            if np.isnan(a) or a < adx_floor:
                continue

        # Streak-just-completed check: last N all same dir, prior N not
        last_n = brick_up[i - min_brick_streak:i]
        prev_n = brick_up[i - min_brick_streak - 1:i - 1]
        brk_long_ok  = bool(np.all(last_n)) and not bool(np.all(prev_n))
        brk_short_ok = bool(not np.any(last_n)) and not bool(not np.any(prev_n))

        if brk_long_ok:
            long_entry[i] = True
            pos = 1
        elif brk_short_ok:
            short_entry[i] = True
            pos = -1

    df = df.copy()
    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df


def run_window(df_slice, params):
    """Run a single backtest on a sliced DataFrame. Return KPI dict or None
    if too few trades."""
    df = generate_signals(df_slice, **params)
    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest_long_short(df, BACKTEST_CONFIG)

    n_trades = kpis.get("total_trades", 0)
    if n_trades < MIN_TRADES_PER_WINDOW:
        return None

    trades = kpis.get("trades", [])
    closed = [t for t in trades if t.exit_date is not None and t.pnl is not None]
    worst_loss = min((t.pnl for t in closed), default=0.0)

    n_long  = sum(1 for t in trades if t.direction == "long"  and t.exit_date is not None)
    n_short = sum(1 for t in trades if t.direction == "short" and t.exit_date is not None)

    return {
        "trades":     n_trades,
        "wr":         round(kpis.get("win_rate", 0), 2),
        "pf":         round(kpis.get("profit_factor", 0), 2),
        "net_profit": round(kpis.get("net_profit", 0), 2),
        "max_dd":     round(kpis.get("max_drawdown", 0), 2),
        "avg_trade":  round(kpis.get("avg_trade", 0), 2),
        "avg_wl":     round(kpis.get("avg_win_loss_ratio", 0), 2),
        "worst_loss": round(worst_loss, 2),
        "n_long":     n_long,
        "n_short":    n_short,
    }


def _run_combo_worker(params, df_train_pickle, df_val_pickle):
    import pickle
    df_train = pickle.loads(df_train_pickle)
    df_val   = pickle.loads(df_val_pickle)

    train_kpis = run_window(df_train, params)
    val_kpis   = run_window(df_val,   params)

    if train_kpis is None or val_kpis is None:
        return None

    return {"params": params, "train": train_kpis, "validate": val_kpis}


def pick_finalists(results, top_n=TOP_N_FINALISTS):
    """Apply finalist filters and rank by VALIDATE PF descending."""
    eligible = []
    for r in results:
        train = r["train"]
        val   = r["validate"]
        if val["trades"] < MIN_VALIDATE_TRADES_FINAL:
            continue
        wr_delta = val["wr"] - train["wr"]
        if wr_delta < -MAX_TRAIN_VAL_WR_DROP:
            continue
        if wr_delta > MAX_TRAIN_VAL_WR_JUMP:
            # Likely regime artifact (R23) — flag and exclude
            continue
        eligible.append(r)
    eligible.sort(key=lambda r: r["validate"]["pf"], reverse=True)
    return eligible[:top_n]


def run_holdout(df_holdout, finalists):
    out = []
    for f in finalists:
        params = f["params"]
        h_kpis = run_window(df_holdout, params)
        if h_kpis is None:
            h_kpis = {
                "trades": 0, "wr": 0, "pf": 0, "net_profit": 0,
                "max_dd": 0, "avg_trade": 0, "avg_wl": 0, "worst_loss": 0,
                "n_long": 0, "n_short": 0,
            }
        out.append({
            "params":   params,
            "train":    f["train"],
            "validate": f["validate"],
            "holdout":  h_kpis,
        })
    return out


def evaluate_against_rules(holdout):
    failures = []
    if holdout["trades"] < RULES["min_trades"]:
        failures.append(f"Trades {holdout['trades']} < {RULES['min_trades']}")
    if holdout["pf"] < RULES["min_pf"]:
        failures.append(f"PF {holdout['pf']} < {RULES['min_pf']}")
    if holdout["wr"] < RULES["min_wr"]:
        failures.append(f"WR {holdout['wr']}% < {RULES['min_wr']}%")
    if holdout["net_profit"] < RULES["min_net_profit"]:
        failures.append(f"Net {holdout['net_profit']} < {RULES['min_net_profit']}")
    return (len(failures) == 0, failures)


def print_top_table(label, items, sort_key, top_n=30):
    print(f"\n{'=' * 120}")
    print(f"  {label}")
    print(f"{'=' * 120}")
    items_sorted = sorted(items, key=sort_key, reverse=True)[:top_n]
    print(f"  {'Strk':>4} {'CD':>4} {'AdxF':>5} | "
          f"{'TR PF':>7} {'TR T':>5} {'TR WR':>6} {'TR Net':>9} | "
          f"{'VL PF':>7} {'VL T':>5} {'VL WR':>6} {'VL Net':>9} {'L/S':>7}")
    print(f"  {'-'*15} | {'-'*36} | {'-'*44}")
    for r in items_sorted:
        p = r["params"]; tr = r["train"]; vl = r["validate"]
        print(f"  {p['min_brick_streak']:>4} {p['cooldown']:>4} {p['adx_floor']:>5} | "
              f"{tr['pf']:>7.2f} {tr['trades']:>5} {tr['wr']:>6.1f} {tr['net_profit']:>9.2f} | "
              f"{vl['pf']:>7.2f} {vl['trades']:>5} {vl['wr']:>6.1f} {vl['net_profit']:>9.2f} "
              f"{vl['n_long']}/{vl['n_short']:>3}")


def print_heatmap(label, results, value_key, fmt=".1f"):
    """Heatmap aggregated over adx_floor (rows=streak, cols=cooldown).
    Picks the BEST adx_floor per (streak, cooldown) cell."""
    print(f"\n{'=' * 120}")
    print(f"  {label}")
    print(f"{'=' * 120}")
    streaks   = sorted(set(r["params"]["min_brick_streak"] for r in results))
    cooldowns = sorted(set(r["params"]["cooldown"] for r in results))
    header = "  " + f"{'S\\CD':>5} | " + " ".join(f"{cd:>8}" for cd in cooldowns)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for s in streaks:
        row = [f"{s:>5}", "|"]
        for cd in cooldowns:
            cell = [r for r in results
                    if r["params"]["min_brick_streak"] == s
                    and r["params"]["cooldown"] == cd]
            if cell:
                best = max(cell, key=lambda r: r["validate"][value_key])
                v = best["validate"][value_key]
                af = best["params"]["adx_floor"]
                row.append(f"{v:>6{fmt}}@{af}")
            else:
                row.append(f"{'-':>8}")
        print("  " + " ".join(row))


def main():
    import pickle

    print(f"Loading {RENKO_FILE}...")
    df = load_renko_export(RENKO_FILE)
    print(f"  {len(df):,} bricks loaded, {df.index[0]} -> {df.index[-1]}")

    print("Computing indicators (ADX, etc.)...")
    add_renko_indicators(df)

    df_train    = df.loc[TRAIN_START:TRAIN_END].copy()
    df_validate = df.loc[VALIDATE_START:VALIDATE_END].copy()
    df_holdout  = df.loc[HOLDOUT_START:HOLDOUT_END].copy()
    print(f"  TRAIN:    {len(df_train):>6,} bricks  ({df_train.index[0]} -> {df_train.index[-1]})")
    print(f"  VALIDATE: {len(df_validate):>6,} bricks  ({df_validate.index[0]} -> {df_validate.index[-1]})")
    print(f"  HOLDOUT:  {len(df_holdout):>6,} bricks  ({df_holdout.index[0]} -> {df_holdout.index[-1]})  [SEALED until finalists picked]")

    df_train_pickle = pickle.dumps(df_train)
    df_val_pickle   = pickle.dumps(df_validate)

    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"\n  {len(combos)} param combos x 2 windows = {len(combos)*2} backtests (HOLDOUT excluded)")

    print(f"\nRunning sweep on TRAIN+VALIDATE using {MAX_WORKERS} workers...")
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run_combo_worker, p, df_train_pickle, df_val_pickle): p for p in combos}
        for future in as_completed(futures):
            r = future.result()
            if r is not None:
                results.append(r)
    elapsed_sweep = time.time() - t0
    print(f"  {len(results)} viable in {elapsed_sweep:.1f}s\n")

    if not results:
        print("NO viable configs (all rejected by min trade count). Exiting.")
        return

    # ── Reports on TRAIN+VALIDATE results ────────────────────────────────────
    print_top_table("TOP 30 BY VALIDATE PROFIT FACTOR",
                    results, lambda r: r["validate"]["pf"], top_n=30)
    print_top_table("TOP 10 BY VALIDATE NET PROFIT",
                    results, lambda r: r["validate"]["net_profit"], top_n=10)
    print_top_table("TOP 10 BY TRAIN->VALIDATE WR DELTA (R15: positive small = signal; R23: huge = artifact)",
                    results, lambda r: r["validate"]["wr"] - r["train"]["wr"], top_n=10)
    print_heatmap("VALIDATE PF HEATMAP — best adx_floor per cell (rows=streak, cols=cooldown)",
                  results, "pf", fmt=".1f")
    print_heatmap("VALIDATE TRADE COUNT HEATMAP — best adx_floor per cell",
                  results, "trades", fmt=".0f")

    # Pure-streak baseline rows for R3 comparison
    pure = [r for r in results if r["params"]["adx_floor"] == 0]
    if pure:
        print_top_table("R3 BASELINE — pure brick streak only (adx_floor=0)",
                        pure, lambda r: r["validate"]["pf"], top_n=15)

    # ── Pick finalists and run holdout ──────────────────────────────────────
    print(f"\n{'=' * 120}")
    print(f"  FINALIST SELECTION")
    print(f"{'=' * 120}")
    print(f"  Filters: VALIDATE trades >= {MIN_VALIDATE_TRADES_FINAL}, "
          f"TRAIN->VALIDATE WR delta in [-{MAX_TRAIN_VAL_WR_DROP}, +{MAX_TRAIN_VAL_WR_JUMP}]pp")
    finalists = pick_finalists(results, top_n=TOP_N_FINALISTS)
    print(f"  {len(finalists)} finalists picked from {len(results)} viable configs")

    if not finalists:
        print(f"\n  NO FINALISTS PASSED FILTERS — holdout step skipped.")
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w") as f:
            json.dump({"results": results, "finalists": [], "holdout": []},
                      f, indent=2, default=str)
        return

    print(f"\n  Top {len(finalists)} finalists by VALIDATE PF:")
    for i, fz in enumerate(finalists, 1):
        p = fz["params"]; vl = fz["validate"]; tr = fz["train"]
        print(f"    {i}. streak={p['min_brick_streak']:>2} cd={p['cooldown']:>3} adxF={p['adx_floor']:>3}  "
              f"VAL PF={vl['pf']:>6.2f}  T={vl['trades']:>4}  "
              f"WR={vl['wr']:>5.1f}  Net=${vl['net_profit']:>9.2f}  "
              f"(TRAIN PF={tr['pf']:>5.2f}, WR delta={vl['wr']-tr['wr']:+.1f}pp)")

    # ── HOLDOUT: run finalists ONCE ─────────────────────────────────────────
    print(f"\n{'=' * 120}")
    print(f"  HOLDOUT RESULTS — finalists run on SEALED holdout window for the first time")
    print(f"  Window: {HOLDOUT_START} -> {HOLDOUT_END} ({len(df_holdout):,} bricks)")
    print(f"{'=' * 120}")
    print(f"  Decision rules: PF>={RULES['min_pf']}, WR>={RULES['min_wr']}%, "
          f"Trades>={RULES['min_trades']}, Net>{RULES['min_net_profit']}")

    holdout_results = run_holdout(df_holdout, finalists)

    print(f"\n  {'#':>2}  {'Strk':>4} {'CD':>4} {'AdxF':>5} | "
          f"{'PF':>6} {'T':>4} {'WR':>5} {'Net$':>10} {'DD':>9} {'WLoss':>8} {'L/S':>6} | DECISION")
    print(f"  {'-'*2}  {'-'*15} | {'-'*54} {'-'*6} | -------")
    for i, r in enumerate(holdout_results, 1):
        p = r["params"]; h = r["holdout"]
        passes, failures = evaluate_against_rules(h)
        decision = "DEPLOY" if passes else "REJECT"
        print(f"  {i:>2}. {p['min_brick_streak']:>4} {p['cooldown']:>4} {p['adx_floor']:>5} | "
              f"{h['pf']:>6.2f} {h['trades']:>4} {h['wr']:>5.1f} "
              f"{h['net_profit']:>10.2f} {h['max_dd']:>9.2f} {h['worst_loss']:>8.2f} "
              f"{h['n_long']}/{h['n_short']:>3} | "
              f"{decision}")
        if failures:
            for fmsg in failures:
                print(f"      x {fmsg}")

    # ── Final deploy recommendation ─────────────────────────────────────────
    deployable = [r for r in holdout_results if evaluate_against_rules(r["holdout"])[0]]
    print(f"\n{'=' * 120}")
    if deployable:
        winner = deployable[0]
        p = winner["params"]
        print(f"  STAGE 1 PASS: streak={p['min_brick_streak']}, cooldown={p['cooldown']}, adx_floor={p['adx_floor']}")
        print(f"    Holdout PF={winner['holdout']['pf']:.2f}  "
              f"T={winner['holdout']['trades']}  "
              f"WR={winner['holdout']['wr']:.1f}%  "
              f"Net=${winner['holdout']['net_profit']:.2f}")
        print(f"  -> Proceed to Stage 2: combine with R029 MR overlay")
    else:
        print(f"  STAGE 1 FAIL: no finalist passed holdout decision rules.")
        print(f"  Trend continuation overlay does NOT have standalone edge on EURUSD 0.0006.")
        print(f"  Do NOT proceed to Stage 2. Reassess (try HTF EMA filter, different brick, etc.)")
    print(f"{'=' * 120}")
    print(f"\nNOTE: data ends 2026-03-25, so the April 7 trend in the screenshot is")
    print(f"      NOT in this holdout. If a winner is found, refresh data before")
    print(f"      finalising and re-run holdout.")

    # ── Save full results ───────────────────────────────────────────────────
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump({
            "config": {
                "renko_file": RENKO_FILE,
                "train_window":    [TRAIN_START, TRAIN_END],
                "validate_window": [VALIDATE_START, VALIDATE_END],
                "holdout_window":  [HOLDOUT_START, HOLDOUT_END],
                "param_grid": PARAM_GRID,
                "finalist_filters": {
                    "min_validate_trades": MIN_VALIDATE_TRADES_FINAL,
                    "max_train_val_wr_drop": MAX_TRAIN_VAL_WR_DROP,
                    "max_train_val_wr_jump": MAX_TRAIN_VAL_WR_JUMP,
                },
                "deployment_rules": RULES,
            },
            "results": results,
            "finalists": [{"params": fz["params"], "train": fz["train"], "validate": fz["validate"]}
                          for fz in finalists],
            "holdout": holdout_results,
        }, f, indent=2, default=str)

    elapsed_total = time.time() - t0
    print(f"\nTotal elapsed: {elapsed_total:.1f}s")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
