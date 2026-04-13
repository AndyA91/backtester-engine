"""
MYM Brick 30 Streak Sweep — comprehensive search with TRAIN/VALIDATE/HOLDOUT.

Three-way data split:
  IS      = 2022-10-16 -> 2024-12-31  (sweep here, find candidate configs)
  OOS     = 2025-01-01 -> 2025-09-30  (validate top candidates from IS)
  HOLDOUT = 2025-10-01 -> 2026-04-07  (sealed during sweep, run finalists ONCE)

Workflow:
  1. For each (streak, cooldown) combo, run on IS and OOS slices
  2. Pick top 5 finalists by OOS profit factor subject to filters:
     - >=50 OOS trades (R14 sample size floor)
     - Worst single-trade loss >= -$50 in BOTH IS and OOS (latency tolerance)
     - IS->OOS WR delta >= -5pp (no major regime decay)
  3. Run those 5 finalists on the HOLDOUT slice
  4. Evaluate each finalist against the deployment decision rules:
     - HOLDOUT PF >= 10
     - HOLDOUT WR >= 55%
     - HOLDOUT max DD >= -$300
     - HOLDOUT worst single-trade loss >= -$50
     - HOLDOUT net profit > 0
     - April 7 net P&L >= -$50

Slices the DataFrame BEFORE generating signals so the stateful `pos` variable
in the signal generator can't be contaminated by phantom out-of-window trades
(handles look_ahead_redflags.md L2).

Usage:
    python renko/mym_brick30_streak_sweep.py
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
from renko.config import MAX_WORKERS

RENKO_FILE = "CBOT_MINI_MYM1!, 1S ETH renko 30.csv"

# ── Data splits ──────────────────────────────────────────────────────────────
IS_START      = "2022-10-16"
IS_END        = "2024-12-31"
OOS_START     = "2025-01-01"
OOS_END       = "2025-09-30"
HOLDOUT_START = "2025-10-01"
HOLDOUT_END   = "2026-04-08"  # exclusive — captures all of April 7

# ── Backtest config ──────────────────────────────────────────────────────────
BACKTEST_CONFIG = BacktestConfig(
    initial_capital=10000.0,
    commission_pct=0.002,
    qty_type="fixed",
    qty_value=1,
    start_date="2000-01-01",
    end_date="2099-12-31",
)

OUTPUT_FILE = ROOT / "ai_context" / "mym_brick30_streak_sweep_results.json"

# ── Param grid ───────────────────────────────────────────────────────────────
PARAM_GRID = {
    "min_brick_streak": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 25],
    "cooldown":         [0, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 75, 100],
}

# ── Filters / thresholds ─────────────────────────────────────────────────────
MIN_TRADES_PER_WINDOW       = 20    # workers reject configs with too few trades
MIN_OOS_TRADES_FINALIST     = 50    # finalist eligibility (R14 floor adapted)
MAX_WORST_LOSS_FINALIST     = -50   # finalist eligibility — worst trade not worse than -$50
MAX_IS_OOS_WR_DROP          = 5.0   # finalist eligibility — IS->OOS WR delta not worse than -5pp
TOP_N_FINALISTS             = 5

# ── Holdout deployment decision rules ────────────────────────────────────────
RULES = {
    "min_pf":          10.0,
    "min_wr":          55.0,
    "max_dd_floor":    -300.0,   # max DD must be >= this (less bad than)
    "min_worst_loss":  -50.0,    # worst single trade must be >= this
    "min_net_profit":  0.0,
    "min_apr7_pnl":    -50.0,    # April 7 net P&L must be >= this
}


def generate_signals(df, min_brick_streak, cooldown):
    """Pure brick streak entry, brick flip exit, cooldown bars between trades.
    Same logic as renko/mym_renko_streak_sweep.py.
    Caller must pass an already-sliced DataFrame so the `pos` state stays clean."""
    brick_up = df["brick_up"].values
    n = len(df)
    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    warmup = min_brick_streak + 2
    last_exit_bar = -999_999
    pos = 0

    for i in range(warmup, n):
        if pos == 1 and not brick_up[i]:
            long_exit[i] = True
            pos = 0
            last_exit_bar = i
        elif pos == -1 and brick_up[i]:
            short_exit[i] = True
            pos = 0
            last_exit_bar = i

        if pos == 0 and (i - last_exit_bar) >= cooldown:
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


def _run_combo_worker(params, df_is_pickle, df_oos_pickle):
    import pickle
    df_is  = pickle.loads(df_is_pickle)
    df_oos = pickle.loads(df_oos_pickle)

    is_kpis  = run_window(df_is,  params)
    oos_kpis = run_window(df_oos, params)

    if is_kpis is None or oos_kpis is None:
        return None

    return {"params": params, "is": is_kpis, "oos": oos_kpis}


def pick_finalists(results, top_n=TOP_N_FINALISTS):
    """Apply finalist filters and rank by OOS PF descending."""
    eligible = []
    for r in results:
        oos = r["oos"]
        is_  = r["is"]
        if oos["trades"] < MIN_OOS_TRADES_FINALIST:
            continue
        if oos["worst_loss"] < MAX_WORST_LOSS_FINALIST:
            continue
        if is_["worst_loss"] < MAX_WORST_LOSS_FINALIST:
            continue
        wr_delta = oos["wr"] - is_["wr"]
        if wr_delta < -MAX_IS_OOS_WR_DROP:
            continue
        eligible.append(r)
    eligible.sort(key=lambda r: r["oos"]["pf"], reverse=True)
    return eligible[:top_n]


def run_holdout(df_holdout, finalists):
    """Run finalists on the HOLDOUT slice. Also extract April 7 trades for
    each so we can compare to the live incident."""
    apr7_start = pd.Timestamp("2026-04-07 00:00:00")
    apr7_end   = pd.Timestamp("2026-04-08 00:00:00")

    out = []
    for f in finalists:
        params = f["params"]
        df = generate_signals(df_holdout, **params)
        with contextlib.redirect_stdout(io.StringIO()):
            kpis = run_backtest_long_short(df, BACKTEST_CONFIG)

        trades = kpis.get("trades", [])
        closed = [t for t in trades if t.exit_date is not None and t.pnl is not None]
        worst_loss = min((t.pnl for t in closed), default=0.0)

        apr7_trades_raw = [
            t for t in trades
            if t.entry_date is not None
            and apr7_start <= pd.Timestamp(t.entry_date) < apr7_end
        ]
        apr7_trades = [
            {
                "entry": str(t.entry_date),
                "side":  t.direction,
                "entry_price": float(t.entry_price),
                "exit_price":  float(t.exit_price) if t.exit_price is not None else None,
                "pnl":   float(t.pnl) if t.pnl is not None else None,
            }
            for t in apr7_trades_raw
        ]
        apr7_pnl = sum(t["pnl"] for t in apr7_trades if t["pnl"] is not None)

        holdout_kpis = {
            "trades":     kpis.get("total_trades", 0),
            "wr":         round(kpis.get("win_rate", 0), 2),
            "pf":         round(kpis.get("profit_factor", 0), 2),
            "net_profit": round(kpis.get("net_profit", 0), 2),
            "max_dd":     round(kpis.get("max_drawdown", 0), 2),
            "avg_trade":  round(kpis.get("avg_trade", 0), 2),
            "avg_wl":     round(kpis.get("avg_win_loss_ratio", 0), 2),
            "worst_loss": round(worst_loss, 2),
            "apr7_pnl":   round(apr7_pnl, 2),
            "apr7_trades": apr7_trades,
            "n_long":     sum(1 for t in trades if t.direction == "long"  and t.exit_date is not None),
            "n_short":    sum(1 for t in trades if t.direction == "short" and t.exit_date is not None),
        }
        out.append({"params": params, "is": f["is"], "oos": f["oos"], "holdout": holdout_kpis})
    return out


def evaluate_against_rules(holdout):
    """Check holdout result against deployment decision rules.
    Returns (passes, list_of_failures)."""
    failures = []
    if holdout["pf"] < RULES["min_pf"]:
        failures.append(f"PF {holdout['pf']} < {RULES['min_pf']}")
    if holdout["wr"] < RULES["min_wr"]:
        failures.append(f"WR {holdout['wr']}% < {RULES['min_wr']}%")
    if holdout["max_dd"] < RULES["max_dd_floor"]:
        failures.append(f"MaxDD {holdout['max_dd']} < {RULES['max_dd_floor']}")
    if holdout["worst_loss"] < RULES["min_worst_loss"]:
        failures.append(f"WorstLoss {holdout['worst_loss']} < {RULES['min_worst_loss']}")
    if holdout["net_profit"] < RULES["min_net_profit"]:
        failures.append(f"NetProfit {holdout['net_profit']} < {RULES['min_net_profit']}")
    if holdout["apr7_pnl"] < RULES["min_apr7_pnl"]:
        failures.append(f"Apr7PnL {holdout['apr7_pnl']} < {RULES['min_apr7_pnl']}")
    return (len(failures) == 0, failures)


def print_top_table(label, items, sort_key, top_n=30):
    print(f"\n{'=' * 110}")
    print(f"  {label}")
    print(f"{'=' * 110}")
    items_sorted = sorted(items, key=sort_key, reverse=True)[:top_n]
    print(f"  {'Strk':>4} {'CD':>4} | {'IS PF':>7} {'IS T':>5} {'IS WR':>6} | "
          f"{'OOS PF':>7} {'OOS T':>5} {'OOS WR':>6} {'OOS Net$':>10} {'OOS DD':>9} {'WLoss':>8}")
    print(f"  {'-'*9} | {'-'*22} | {'-'*55}")
    for r in items_sorted:
        p = r["params"]; is_ = r["is"]; oos = r["oos"]
        print(f"  {p['min_brick_streak']:>4} {p['cooldown']:>4} | "
              f"{is_['pf']:>7.2f} {is_['trades']:>5} {is_['wr']:>6.1f} | "
              f"{oos['pf']:>7.2f} {oos['trades']:>5} {oos['wr']:>6.1f} "
              f"{oos['net_profit']:>10.2f} {oos['max_dd']:>9.2f} {oos['worst_loss']:>8.2f}")


def print_heatmap(label, results, value_key, fmt=".1f"):
    print(f"\n{'=' * 110}")
    print(f"  {label}")
    print(f"{'=' * 110}")
    streaks   = sorted(set(r["params"]["min_brick_streak"] for r in results))
    cooldowns = sorted(set(r["params"]["cooldown"] for r in results))
    header = "  " + f"{'S\\CD':>5} | " + " ".join(f"{cd:>6}" for cd in cooldowns)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for s in streaks:
        row = [f"{s:>5}", "|"]
        for cd in cooldowns:
            match = next((r for r in results
                          if r["params"]["min_brick_streak"] == s
                          and r["params"]["cooldown"] == cd), None)
            if match:
                v = match["oos"][value_key]
                row.append(f"{v:>6{fmt}}")
            else:
                row.append(f"{'-':>6}")
        print("  " + " ".join(row))


def main():
    import pickle

    print(f"Loading {RENKO_FILE}...")
    df = load_renko_export(RENKO_FILE)
    print(f"  {len(df):,} bricks loaded, {df.index[0]} -> {df.index[-1]}")

    df_is      = df.loc[IS_START:IS_END].copy()
    df_oos     = df.loc[OOS_START:OOS_END].copy()
    df_holdout = df.loc[HOLDOUT_START:HOLDOUT_END].copy()
    print(f"  IS:      {len(df_is):>6,} bricks  ({df_is.index[0]} -> {df_is.index[-1]})")
    print(f"  OOS:     {len(df_oos):>6,} bricks  ({df_oos.index[0]} -> {df_oos.index[-1]})")
    print(f"  HOLDOUT: {len(df_holdout):>6,} bricks  ({df_holdout.index[0]} -> {df_holdout.index[-1]})  [SEALED until finalists picked]")

    df_is_pickle  = pickle.dumps(df_is)
    df_oos_pickle = pickle.dumps(df_oos)

    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"\n  {len(combos)} param combos x 2 windows = {len(combos)*2} backtests (HOLDOUT excluded)")

    print(f"\nRunning sweep on IS+OOS using {MAX_WORKERS} workers...")
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run_combo_worker, p, df_is_pickle, df_oos_pickle): p for p in combos}
        for future in as_completed(futures):
            r = future.result()
            if r is not None:
                results.append(r)
    elapsed_sweep = time.time() - t0
    print(f"  {len(results)} viable in {elapsed_sweep:.1f}s\n")

    # ── Reports on IS+OOS results ────────────────────────────────────────────
    print_top_table("TOP 30 BY OOS PROFIT FACTOR",
                    results, lambda r: r["oos"]["pf"], top_n=30)
    print_top_table("TOP 10 BY OOS NET PROFIT",
                    results, lambda r: r["oos"]["net_profit"], top_n=10)
    print_top_table("TOP 10 BY IS->OOS WR DELTA (R15: positive = genuine signal)",
                    results, lambda r: r["oos"]["wr"] - r["is"]["wr"], top_n=10)
    print_heatmap("OOS PROFIT FACTOR HEATMAP (rows=streak, cols=cooldown)",
                  results, "pf", fmt=".1f")
    print_heatmap("OOS TRADE COUNT HEATMAP (rows=streak, cols=cooldown)",
                  results, "trades", fmt=".0f")

    # ── Pick finalists and run holdout ──────────────────────────────────────
    print(f"\n{'=' * 110}")
    print(f"  FINALIST SELECTION")
    print(f"{'=' * 110}")
    print(f"  Filters: OOS trades >= {MIN_OOS_TRADES_FINALIST}, "
          f"worst trade in IS+OOS >= {MAX_WORST_LOSS_FINALIST}, "
          f"IS->OOS WR delta >= -{MAX_IS_OOS_WR_DROP}pp")
    finalists = pick_finalists(results, top_n=TOP_N_FINALISTS)
    print(f"  {len(finalists)} finalists picked from {len(results)} viable configs")

    if not finalists:
        print(f"\n  NO FINALISTS PASSED FILTERS — holdout step skipped.")
        print(f"  Saving sweep results without holdout...")
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w") as f:
            json.dump({"results": results, "finalists": [], "holdout": []}, f, indent=2, default=str)
        return

    print(f"\n  Top {len(finalists)} finalists by OOS PF:")
    for i, f in enumerate(finalists, 1):
        p = f["params"]; oos = f["oos"]; is_ = f["is"]
        print(f"    {i}. streak={p['min_brick_streak']:>2} cd={p['cooldown']:>3}  "
              f"OOS PF={oos['pf']:>6.2f}  T={oos['trades']:>4}  "
              f"WR={oos['wr']:>5.1f}  Net=${oos['net_profit']:>9.2f}  "
              f"WLoss=${oos['worst_loss']:>6.2f}  "
              f"(IS PF={is_['pf']:>5.2f}, WR delta={oos['wr']-is_['wr']:+.1f}pp)")

    # ── HOLDOUT: run finalists ONCE ─────────────────────────────────────────
    print(f"\n{'=' * 110}")
    print(f"  HOLDOUT RESULTS — finalists run on SEALED holdout window for the first time")
    print(f"  Window: {HOLDOUT_START} -> {HOLDOUT_END} ({len(df_holdout):,} bricks)")
    print(f"{'=' * 110}")
    print(f"  Decision rules: PF>={RULES['min_pf']}, WR>={RULES['min_wr']}%, "
          f"MaxDD>={RULES['max_dd_floor']}, WorstLoss>={RULES['min_worst_loss']}, "
          f"Net>{RULES['min_net_profit']}, Apr7PnL>={RULES['min_apr7_pnl']}")

    holdout_results = run_holdout(df_holdout, finalists)

    print(f"\n  {'#':>2}  {'Strk':>4} {'CD':>4} | "
          f"{'PF':>6} {'T':>4} {'WR':>5} {'Net$':>10} {'DD':>9} {'WLoss':>8} {'Apr7':>9} | DECISION")
    print(f"  {'-'*2}  {'-'*9} | {'-'*55} | -------")
    for i, r in enumerate(holdout_results, 1):
        p = r["params"]; h = r["holdout"]
        passes, failures = evaluate_against_rules(h)
        decision = "DEPLOY" if passes else "REJECT"
        print(f"  {i:>2}. {p['min_brick_streak']:>4} {p['cooldown']:>4} | "
              f"{h['pf']:>6.2f} {h['trades']:>4} {h['wr']:>5.1f} "
              f"{h['net_profit']:>10.2f} {h['max_dd']:>9.2f} {h['worst_loss']:>8.2f} {h['apr7_pnl']:>9.2f} | "
              f"{decision}")
        if failures:
            for fmsg in failures:
                print(f"      x {fmsg}")
        if h["apr7_trades"]:
            print(f"      April 7 trades on this finalist:")
            for t in h["apr7_trades"]:
                exit_str = f"{t['exit_price']:.0f}" if t["exit_price"] is not None else "open"
                pnl_str  = f"${t['pnl']:>7.2f}" if t["pnl"] is not None else "  open"
                print(f"        {t['entry']}  {t['side']:<5}  entry={t['entry_price']:.0f}  exit={exit_str}  pnl={pnl_str}")

    # ── Final deploy recommendation ─────────────────────────────────────────
    deployable = [r for r in holdout_results if evaluate_against_rules(r["holdout"])[0]]
    print(f"\n{'=' * 110}")
    if deployable:
        winner = deployable[0]  # already sorted by OOS PF descending from finalist selection
        p = winner["params"]
        print(f"  DEPLOY RECOMMENDATION: streak={p['min_brick_streak']}, cooldown={p['cooldown']}")
        print(f"    Holdout PF={winner['holdout']['pf']:.2f}  "
              f"Trades={winner['holdout']['trades']}  "
              f"WR={winner['holdout']['wr']:.1f}%  "
              f"Net=${winner['holdout']['net_profit']:.2f}  "
              f"Apr7=${winner['holdout']['apr7_pnl']:.2f}")
    else:
        print(f"  NO FINALIST PASSED HOLDOUT — do NOT redeploy. Reassess strategy.")
    print(f"{'=' * 110}")

    # ── Save full results ───────────────────────────────────────────────────
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump({
            "config": {
                "renko_file": RENKO_FILE,
                "is_window":      [IS_START, IS_END],
                "oos_window":     [OOS_START, OOS_END],
                "holdout_window": [HOLDOUT_START, HOLDOUT_END],
                "param_grid": PARAM_GRID,
                "finalist_filters": {
                    "min_oos_trades": MIN_OOS_TRADES_FINALIST,
                    "max_worst_loss": MAX_WORST_LOSS_FINALIST,
                    "max_is_oos_wr_drop": MAX_IS_OOS_WR_DROP,
                },
                "deployment_rules": RULES,
            },
            "results": results,
            "finalists": [{"params": f["params"], "is": f["is"], "oos": f["oos"]} for f in finalists],
            "holdout": holdout_results,
        }, f, indent=2, default=str)

    elapsed_total = time.time() - t0
    print(f"\nTotal elapsed: {elapsed_total:.1f}s")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
