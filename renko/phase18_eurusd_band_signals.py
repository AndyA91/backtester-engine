"""
Phase 18 — EURUSD 0.0006 Bollinger Band Signal Reuse

Purpose
-------
Phase 17 found brick streak (the MYM_STREAK signal) does NOT transfer to
EURUSD 0.0006 — streaks signal exhaustion, not continuation. The user
correctly pointed out we already have Bollinger Band data on R029's chart
and asked: can we use what we already have?

Two angles, both reuse R029's existing BB indicator stack (bb_pct_b, bb_mid):

ANGLE A — Band Walk Continuation (additive overlay)
    LONG  : bb_pct_b >= walk_threshold for `walk_streak` consecutive bricks,
            ALL bricks up
    SHORT : bb_pct_b <= (1 - walk_threshold) for `walk_streak` bricks,
            ALL bricks down
    EXIT  : first opposing brick (R1)
    Hypothesis: persistent band walk + brick agreement = trending regime,
    structurally different from naked brick streak (Phase 17 dead end).

ANGLE B — bb_mid Slope Filter on R029 (subtractive R029 improvement)
    Block R029 long  entries when bb_mid is FALLING fast (long fights downtrend)
    Block R029 short entries when bb_mid is RISING  fast (short fights uptrend)
    Hypothesis: R029's worst trades are MR entries that fight the moving mean.
    Filtering them out should improve net profit without changing R029's
    fundamental design.

Both angles are Pine-portable in ~10 lines of additional code, reuse the
exact same brick (0.0006) and exit (first opposing brick), and decouple
from the failed brick-streak hypothesis.

Three-way data split (per R22)
------------------------------
  TRAIN     2023-01-02 -> 2025-09-30   (matches R029's IS boundary)
  VALIDATE  2025-10-01 -> 2025-12-31   (~3 months)
  HOLDOUT   2026-01-01 -> 2026-03-25   (~12 weeks, sealed)

Decision rules
--------------
ANGLE A (standalone overlay):
    - HOLDOUT PF >= 4
    - HOLDOUT WR >= 50%
    - HOLDOUT trades >= 20
    - HOLDOUT net profit > 0

ANGLE B (R029 improvement, comparison to baseline):
    - HOLDOUT net profit >= baseline_net * 1.05  (at least +5%)
    - HOLDOUT WR >= baseline_wr - 2.0            (max 2pp degradation)
    - HOLDOUT trades >= baseline_trades * 0.70   (max 30% trade reduction)

Baseline = R029 with its locked Pine params (bt=0.20, cd=2, rsi=45, adx=25,
vol=1.5) run on the same Python engine and same data windows. This is what
we compare Angle B finalists against — not the published TV numbers, which
may have small Pine vs Python translation differences.

Usage
-----
    python renko/phase18_eurusd_band_signals.py
"""

import contextlib
import io
import itertools
import json
import sys
import time

# Force UTF-8 stdout on Windows so box-drawing chars don't crash the print
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import BacktestConfig, run_backtest_long_short
from renko.data import load_renko_export
from renko.indicators import add_renko_indicators
from renko.config import MAX_WORKERS

RENKO_FILE = "OANDA_EURUSD, 1S renko 0.0006.csv"

# ── Data splits ──────────────────────────────────────────────────────────────
TRAIN_START    = "2023-01-02"
TRAIN_END      = "2025-09-30"
VALIDATE_START = "2025-10-01"
VALIDATE_END   = "2025-12-31"
HOLDOUT_START  = "2026-01-01"
HOLDOUT_END    = "2026-03-26"

# ── Backtest config (matches R029 Pine) ─────────────────────────────────────
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

OUTPUT_FILE = ROOT / "ai_context" / "phase18_eurusd_band_signals_results.json"

# ── R029 locked Pine params (the live config) ───────────────────────────────
R029_PARAMS = {
    "band_thresh": 0.20,
    "cooldown":    2,
    "rsi_thresh":  45,
    "adx_max":     25,
    "vol_max":     1.5,
}

# ── Param grids ──────────────────────────────────────────────────────────────
ANGLE_A_GRID = {
    "walk_threshold": [0.70, 0.75, 0.80, 0.85],
    "walk_streak":    [3, 5, 7, 10],
    "cooldown":       [5, 10, 20],
}

ANGLE_B_GRID = {
    "slope_lookback":  [10, 20, 40],
    "slope_threshold": [0.0005, 0.001, 0.002, 0.003],
}

# ── Filters / thresholds ─────────────────────────────────────────────────────
MIN_TRADES_PER_WINDOW       = 5     # workers reject configs below this
MIN_VALIDATE_TRADES_FINAL_A = 15    # Angle A finalist eligibility (selective signal)
MAX_TRAIN_VAL_WR_DROP       = 5.0
MAX_TRAIN_VAL_WR_JUMP       = 15.0
TOP_N_FINALISTS_A           = 5
TOP_N_FINALISTS_B           = 5

# ── Holdout decision rules (locked BEFORE viewing holdout) ──────────────────
RULES_A = {
    "min_pf":         4.0,
    "min_wr":         50.0,
    "min_trades":     20,
    "min_net_profit": 0.0,
}

RULES_B = {
    "min_net_multiplier":   1.05,   # >= baseline_net * 1.05
    "max_wr_degradation":   2.0,    # WR can drop at most 2pp from baseline
    "min_trade_fraction":   0.70,   # trade count >= 70% of baseline
}


# ── Signal generators ───────────────────────────────────────────────────────


def gen_band_walk(df, walk_threshold, walk_streak, cooldown):
    """ANGLE A: brick streak + at-the-band gate.

    LONG  : last `walk_streak` bricks all UP AND last `walk_streak` values of
            bb_pct_b all >= walk_threshold (price walking the upper band)
    SHORT : symmetric on lower band
    EXIT  : first opposing brick

    Caller must pass an already-sliced DataFrame so the `pos` state stays
    clean (R18).
    """
    brick_up = df["brick_up"].values
    pct_b    = df["bb_pct_b"].values  # pre-shifted
    n = len(df)

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    warmup = max(walk_streak + 2, 30)
    last_exit_bar = -999_999
    pos = 0
    upper = walk_threshold
    lower = 1.0 - walk_threshold

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

        # Last walk_streak bricks all same direction AND all at the band
        last_bricks = brick_up[i - walk_streak:i]
        last_pct    = pct_b[i - walk_streak:i]

        if np.any(np.isnan(last_pct)):
            continue

        all_up      = bool(np.all(last_bricks))
        all_dn      = bool(not np.any(last_bricks))
        walk_up_ok  = bool(np.all(last_pct >= upper))
        walk_dn_ok  = bool(np.all(last_pct <= lower))

        if all_up and walk_up_ok:
            long_entry[i] = True
            pos = 1
        elif all_dn and walk_dn_ok:
            short_entry[i] = True
            pos = -1

    df = df.copy()
    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df


def gen_r029_with_slope_filter(df, band_thresh, cooldown, rsi_thresh, adx_max,
                               vol_max, slope_lookback, slope_threshold):
    """R029 Band Bounce MR with optional bb_mid slope filter (ANGLE B).

    Reproduces R029's locked Pine logic exactly, then optionally blocks:
        - LONG  entries when bb_mid is FALLING (mean trending down)
        - SHORT entries when bb_mid is RISING  (mean trending up)

    slope_threshold = 0 disables the filter (used for the baseline run).
    """
    brick_up   = df["brick_up"].values
    pct_b      = df["bb_pct_b"].values
    rsi        = df["rsi"].values
    adx        = df["adx"].values
    vol_ratio  = df["vol_ratio"].values
    bb_mid     = df["bb_mid"].values
    n = len(df)

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    warmup = max(slope_lookback + 2, 30)
    last_trade_bar = -999_999
    pos = 0

    upper_band = 1.0 - band_thresh
    lower_band = band_thresh
    rsi_short_thresh = 100.0 - rsi_thresh

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # Exit: first opposing brick
        if pos == 1 and not up:
            long_exit[i] = True
            pos = 0
        elif pos == -1 and up:
            short_exit[i] = True
            pos = 0

        if pos != 0:
            continue
        if (i - last_trade_bar) < cooldown:
            continue

        # Volume gate
        vr = vol_ratio[i]
        if vol_max > 0 and not np.isnan(vr) and vr > vol_max:
            continue

        # ADX ceiling (R029 ranging-only)
        a = adx[i]
        if adx_max > 0 and not np.isnan(a) and a > adx_max:
            continue

        pct = pct_b[i]
        if np.isnan(pct):
            continue

        # Identify candidate side (R029 entry conditions)
        cand = 0
        if pct <= lower_band and up:
            # LONG: at lower band + reversal up brick
            r = rsi[i]
            if rsi_thresh > 0 and not np.isnan(r) and r >= rsi_thresh:
                continue
            cand = 1
        elif pct >= upper_band and not up:
            # SHORT: at upper band + reversal down brick
            r = rsi[i]
            if rsi_thresh > 0 and not np.isnan(r) and r <= rsi_short_thresh:
                continue
            cand = -1
        else:
            continue

        # bb_mid slope filter (ANGLE B specific)
        if slope_threshold > 0 and slope_lookback > 0:
            mid_now  = bb_mid[i]
            mid_prev = bb_mid[i - slope_lookback]
            if not np.isnan(mid_now) and not np.isnan(mid_prev):
                slope = mid_now - mid_prev
                if cand == 1 and slope < -slope_threshold:
                    continue  # mean falling, long fights downtrend
                if cand == -1 and slope > slope_threshold:
                    continue  # mean rising, short fights uptrend

        if cand == 1:
            long_entry[i] = True
            pos = 1
            last_trade_bar = i
        else:
            short_entry[i] = True
            pos = -1
            last_trade_bar = i

    df = df.copy()
    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df


# ── Backtest harness ────────────────────────────────────────────────────────


def run_window(df_signals):
    """Run a single backtest on a DataFrame that already has signal columns."""
    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest_long_short(df_signals, BACKTEST_CONFIG)

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


def _run_angle_a_worker(params, df_train_pickle, df_val_pickle):
    import pickle
    df_train = pickle.loads(df_train_pickle)
    df_val   = pickle.loads(df_val_pickle)

    train_kpis = run_window(gen_band_walk(df_train, **params))
    val_kpis   = run_window(gen_band_walk(df_val,   **params))

    if train_kpis is None or val_kpis is None:
        return None
    return {"params": params, "train": train_kpis, "validate": val_kpis}


def _run_angle_b_worker(params, df_train_pickle, df_val_pickle):
    import pickle
    df_train = pickle.loads(df_train_pickle)
    df_val   = pickle.loads(df_val_pickle)

    full_params = dict(R029_PARAMS, **params)
    train_kpis = run_window(gen_r029_with_slope_filter(df_train, **full_params))
    val_kpis   = run_window(gen_r029_with_slope_filter(df_val,   **full_params))

    if train_kpis is None or val_kpis is None:
        return None
    return {"params": params, "train": train_kpis, "validate": val_kpis}


# ── Finalist selection / holdout ────────────────────────────────────────────


def pick_finalists_a(results, top_n=TOP_N_FINALISTS_A):
    eligible = []
    for r in results:
        train = r["train"]
        val   = r["validate"]
        if val["trades"] < MIN_VALIDATE_TRADES_FINAL_A:
            continue
        wr_delta = val["wr"] - train["wr"]
        if wr_delta < -MAX_TRAIN_VAL_WR_DROP:
            continue
        if wr_delta > MAX_TRAIN_VAL_WR_JUMP:
            continue
        eligible.append(r)
    eligible.sort(key=lambda r: r["validate"]["pf"], reverse=True)
    return eligible[:top_n]


def pick_finalists_b(results, baseline_validate, top_n=TOP_N_FINALISTS_B):
    """Angle B finalists: pick configs whose VALIDATE-window net profit beats
    baseline VALIDATE net profit AND don't crash trade count."""
    base_net    = baseline_validate["net_profit"]
    base_trades = baseline_validate["trades"]
    eligible = []
    for r in results:
        v = r["validate"]
        if v["trades"] < base_trades * 0.70:
            continue
        if v["net_profit"] <= base_net:
            continue
        eligible.append(r)
    eligible.sort(key=lambda r: r["validate"]["net_profit"], reverse=True)
    return eligible[:top_n]


def evaluate_a(holdout):
    failures = []
    if holdout["trades"] < RULES_A["min_trades"]:
        failures.append(f"T {holdout['trades']} < {RULES_A['min_trades']}")
    if holdout["pf"] < RULES_A["min_pf"]:
        failures.append(f"PF {holdout['pf']} < {RULES_A['min_pf']}")
    if holdout["wr"] < RULES_A["min_wr"]:
        failures.append(f"WR {holdout['wr']}% < {RULES_A['min_wr']}%")
    if holdout["net_profit"] < RULES_A["min_net_profit"]:
        failures.append(f"Net {holdout['net_profit']} < {RULES_A['min_net_profit']}")
    return (len(failures) == 0, failures)


def evaluate_b(holdout, baseline_holdout):
    base_net    = baseline_holdout["net_profit"]
    base_wr     = baseline_holdout["wr"]
    base_trades = baseline_holdout["trades"]
    failures = []

    min_net = base_net * RULES_B["min_net_multiplier"] if base_net > 0 else base_net + 5.0
    if holdout["net_profit"] < min_net:
        failures.append(f"Net {holdout['net_profit']:.2f} < {min_net:.2f} (baseline*1.05)")
    if holdout["wr"] < base_wr - RULES_B["max_wr_degradation"]:
        failures.append(f"WR {holdout['wr']}% < {base_wr - RULES_B['max_wr_degradation']:.1f}% (baseline-2pp)")
    if holdout["trades"] < base_trades * RULES_B["min_trade_fraction"]:
        failures.append(f"T {holdout['trades']} < {base_trades * RULES_B['min_trade_fraction']:.0f} (70% baseline)")
    return (len(failures) == 0, failures)


def print_top_table_a(label, items, sort_key, top_n=20):
    print(f"\n{'=' * 120}")
    print(f"  {label}")
    print(f"{'=' * 120}")
    items_sorted = sorted(items, key=sort_key, reverse=True)[:top_n]
    print(f"  {'Thr':>5} {'Strk':>5} {'CD':>4} | "
          f"{'TR PF':>7} {'TR T':>5} {'TR WR':>6} {'TR Net':>9} | "
          f"{'VL PF':>7} {'VL T':>5} {'VL WR':>6} {'VL Net':>9} {'L/S':>7}")
    print(f"  {'-'*16} | {'-'*36} | {'-'*44}")
    for r in items_sorted:
        p = r["params"]; tr = r["train"]; vl = r["validate"]
        print(f"  {p['walk_threshold']:>5.2f} {p['walk_streak']:>5} {p['cooldown']:>4} | "
              f"{tr['pf']:>7.2f} {tr['trades']:>5} {tr['wr']:>6.1f} {tr['net_profit']:>9.2f} | "
              f"{vl['pf']:>7.2f} {vl['trades']:>5} {vl['wr']:>6.1f} {vl['net_profit']:>9.2f} "
              f"{vl['n_long']}/{vl['n_short']:>3}")


def print_top_table_b(label, items, sort_key, baseline_validate, top_n=20):
    print(f"\n{'=' * 120}")
    print(f"  {label}")
    print(f"{'=' * 120}")
    base = baseline_validate
    print(f"  BASELINE: T={base['trades']} WR={base['wr']:.1f} PF={base['pf']:.2f} Net=${base['net_profit']:.2f}")
    items_sorted = sorted(items, key=sort_key, reverse=True)[:top_n]
    print(f"  {'LkBk':>5} {'SlopeT':>8} | "
          f"{'TR PF':>7} {'TR T':>5} {'TR WR':>6} {'TR Net':>9} | "
          f"{'VL PF':>7} {'VL T':>5} {'VL WR':>6} {'VL Net':>9} {'dNet':>8} {'dWR':>6} {'dT':>5}")
    print(f"  {'-'*15} | {'-'*36} | {'-'*60}")
    for r in items_sorted:
        p = r["params"]; tr = r["train"]; vl = r["validate"]
        d_net = vl["net_profit"] - base["net_profit"]
        d_wr  = vl["wr"]         - base["wr"]
        d_t   = vl["trades"]     - base["trades"]
        print(f"  {p['slope_lookback']:>5} {p['slope_threshold']:>8.4f} | "
              f"{tr['pf']:>7.2f} {tr['trades']:>5} {tr['wr']:>6.1f} {tr['net_profit']:>9.2f} | "
              f"{vl['pf']:>7.2f} {vl['trades']:>5} {vl['wr']:>6.1f} {vl['net_profit']:>9.2f} "
              f"{d_net:>+8.2f} {d_wr:>+6.1f} {d_t:>+5}")


def main():
    import pickle

    print(f"Loading {RENKO_FILE}...")
    df = load_renko_export(RENKO_FILE)
    print(f"  {len(df):,} bricks loaded, {df.index[0]} -> {df.index[-1]}")

    print("Computing indicators...")
    add_renko_indicators(df)

    df_train    = df.loc[TRAIN_START:TRAIN_END].copy()
    df_validate = df.loc[VALIDATE_START:VALIDATE_END].copy()
    df_holdout  = df.loc[HOLDOUT_START:HOLDOUT_END].copy()
    print(f"  TRAIN:    {len(df_train):>6,} bricks  ({df_train.index[0]} -> {df_train.index[-1]})")
    print(f"  VALIDATE: {len(df_validate):>6,} bricks  ({df_validate.index[0]} -> {df_validate.index[-1]})")
    print(f"  HOLDOUT:  {len(df_holdout):>6,} bricks  ({df_holdout.index[0]} -> {df_holdout.index[-1]})  [SEALED]")

    df_train_pickle = pickle.dumps(df_train)
    df_val_pickle   = pickle.dumps(df_validate)

    # ───────────────────────────────────────────────────────────────────────
    # R029 BASELINE — locked Pine params, no slope filter, all 3 windows
    # ───────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 120}")
    print(f"  R029 BASELINE (locked Pine params, no slope filter, Python engine)")
    print(f"{'=' * 120}")
    baseline_params = dict(R029_PARAMS, slope_lookback=20, slope_threshold=0.0)
    base_train    = run_window(gen_r029_with_slope_filter(df_train,    **baseline_params))
    base_validate = run_window(gen_r029_with_slope_filter(df_validate, **baseline_params))
    base_holdout  = run_window(gen_r029_with_slope_filter(df_holdout,  **baseline_params))
    for label, k in [("TRAIN", base_train), ("VALIDATE", base_validate), ("HOLDOUT", base_holdout)]:
        if k is None:
            print(f"  {label:<10} (too few trades)")
        else:
            print(f"  {label:<10} T={k['trades']:>4}  WR={k['wr']:>5.1f}%  PF={k['pf']:>6.2f}  "
                  f"Net=${k['net_profit']:>9.2f}  L/S={k['n_long']}/{k['n_short']}  "
                  f"WLoss=${k['worst_loss']:>6.2f}")
    if base_validate is None or base_holdout is None:
        print("\n  ERROR: baseline produced too few trades — cannot run Angle B comparison")
        return

    # ───────────────────────────────────────────────────────────────────────
    # ANGLE A — Band Walk Continuation sweep
    # ───────────────────────────────────────────────────────────────────────
    keys_a = list(ANGLE_A_GRID.keys())
    vals_a = list(ANGLE_A_GRID.values())
    combos_a = [dict(zip(keys_a, v)) for v in itertools.product(*vals_a)]
    print(f"\n{'=' * 120}")
    print(f"  ANGLE A — BAND WALK CONTINUATION SWEEP")
    print(f"{'=' * 120}")
    print(f"  {len(combos_a)} combos x 2 windows = {len(combos_a)*2} backtests")

    t0 = time.time()
    results_a = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run_angle_a_worker, p, df_train_pickle, df_val_pickle): p
                   for p in combos_a}
        for future in as_completed(futures):
            r = future.result()
            if r is not None:
                results_a.append(r)
    print(f"  {len(results_a)} viable in {time.time()-t0:.1f}s")

    if results_a:
        print_top_table_a("ANGLE A — TOP 20 BY VALIDATE PF", results_a,
                          lambda r: r["validate"]["pf"])
        print_top_table_a("ANGLE A — TOP 10 BY VALIDATE NET PROFIT", results_a,
                          lambda r: r["validate"]["net_profit"], top_n=10)

    # ───────────────────────────────────────────────────────────────────────
    # ANGLE B — bb_mid Slope Filter on R029 sweep
    # ───────────────────────────────────────────────────────────────────────
    keys_b = list(ANGLE_B_GRID.keys())
    vals_b = list(ANGLE_B_GRID.values())
    combos_b = [dict(zip(keys_b, v)) for v in itertools.product(*vals_b)]
    print(f"\n{'=' * 120}")
    print(f"  ANGLE B — BB_MID SLOPE FILTER ON R029 SWEEP")
    print(f"{'=' * 120}")
    print(f"  {len(combos_b)} combos x 2 windows = {len(combos_b)*2} backtests")

    t0 = time.time()
    results_b = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run_angle_b_worker, p, df_train_pickle, df_val_pickle): p
                   for p in combos_b}
        for future in as_completed(futures):
            r = future.result()
            if r is not None:
                results_b.append(r)
    print(f"  {len(results_b)} viable in {time.time()-t0:.1f}s")

    if results_b:
        print_top_table_b("ANGLE B — TOP 20 BY VALIDATE NET PROFIT", results_b,
                          lambda r: r["validate"]["net_profit"], base_validate)

    # ───────────────────────────────────────────────────────────────────────
    # FINALISTS + HOLDOUT
    # ───────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 120}")
    print(f"  FINALIST SELECTION + HOLDOUT")
    print(f"{'=' * 120}")

    finalists_a = pick_finalists_a(results_a) if results_a else []
    finalists_b = pick_finalists_b(results_b, base_validate) if results_b else []
    print(f"  Angle A finalists: {len(finalists_a)}  (filter: VAL T>={MIN_VALIDATE_TRADES_FINAL_A}, "
          f"WR delta in [-{MAX_TRAIN_VAL_WR_DROP},+{MAX_TRAIN_VAL_WR_JUMP}]pp)")
    print(f"  Angle B finalists: {len(finalists_b)}  (filter: VAL trades>=70% baseline AND VAL net>baseline)")

    # Run holdout for Angle A finalists
    holdout_a = []
    if finalists_a:
        print(f"\n  ── ANGLE A HOLDOUT ──")
        print(f"  Decision rules: PF>={RULES_A['min_pf']}, WR>={RULES_A['min_wr']}%, "
              f"T>={RULES_A['min_trades']}, Net>{RULES_A['min_net_profit']}")
        for f in finalists_a:
            h = run_window(gen_band_walk(df_holdout, **f["params"]))
            if h is None:
                h = {"trades": 0, "wr": 0, "pf": 0, "net_profit": 0,
                     "max_dd": 0, "avg_trade": 0, "avg_wl": 0, "worst_loss": 0,
                     "n_long": 0, "n_short": 0}
            holdout_a.append({"params": f["params"], "train": f["train"],
                              "validate": f["validate"], "holdout": h})

        for i, r in enumerate(holdout_a, 1):
            p = r["params"]; h = r["holdout"]
            passes, failures = evaluate_a(h)
            decision = "PASS" if passes else "REJECT"
            print(f"  {i}. thr={p['walk_threshold']:.2f} strk={p['walk_streak']} cd={p['cooldown']:>2} | "
                  f"PF={h['pf']:>6.2f} T={h['trades']:>3} WR={h['wr']:>5.1f}% "
                  f"Net=${h['net_profit']:>8.2f} L/S={h['n_long']}/{h['n_short']} | {decision}")
            for fmsg in failures:
                print(f"      x {fmsg}")

    # Run holdout for Angle B finalists
    holdout_b = []
    if finalists_b:
        print(f"\n  ── ANGLE B HOLDOUT ──")
        bn = base_holdout["net_profit"]
        print(f"  Baseline HOLDOUT: T={base_holdout['trades']} WR={base_holdout['wr']:.1f}% "
              f"PF={base_holdout['pf']:.2f} Net=${bn:.2f}")
        min_net = bn * RULES_B["min_net_multiplier"] if bn > 0 else bn + 5.0
        print(f"  Decision rules: Net>=${min_net:.2f}, "
              f"WR>={base_holdout['wr']-RULES_B['max_wr_degradation']:.1f}%, "
              f"T>={base_holdout['trades']*RULES_B['min_trade_fraction']:.0f}")

        for f in finalists_b:
            full_params = dict(R029_PARAMS, **f["params"])
            h = run_window(gen_r029_with_slope_filter(df_holdout, **full_params))
            if h is None:
                h = {"trades": 0, "wr": 0, "pf": 0, "net_profit": 0,
                     "max_dd": 0, "avg_trade": 0, "avg_wl": 0, "worst_loss": 0,
                     "n_long": 0, "n_short": 0}
            holdout_b.append({"params": f["params"], "train": f["train"],
                              "validate": f["validate"], "holdout": h})

        for i, r in enumerate(holdout_b, 1):
            p = r["params"]; h = r["holdout"]
            passes, failures = evaluate_b(h, base_holdout)
            decision = "PASS" if passes else "REJECT"
            d_net = h["net_profit"] - base_holdout["net_profit"]
            d_wr  = h["wr"]         - base_holdout["wr"]
            d_t   = h["trades"]     - base_holdout["trades"]
            print(f"  {i}. lkbk={p['slope_lookback']:>2} thr={p['slope_threshold']:.4f} | "
                  f"PF={h['pf']:>6.2f} T={h['trades']:>3} WR={h['wr']:>5.1f}% "
                  f"Net=${h['net_profit']:>8.2f}  "
                  f"dNet={d_net:+7.2f} dWR={d_wr:+5.1f} dT={d_t:+3} | {decision}")
            for fmsg in failures:
                print(f"      x {fmsg}")

    # ───────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ───────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 120}")
    print(f"  PHASE 18 SUMMARY")
    print(f"{'=' * 120}")
    a_pass = [r for r in holdout_a if evaluate_a(r["holdout"])[0]]
    b_pass = [r for r in holdout_b if evaluate_b(r["holdout"], base_holdout)[0]]
    print(f"  Angle A (band walk continuation): {len(a_pass)} configs PASS holdout")
    print(f"  Angle B (bb_mid slope filter):    {len(b_pass)} configs PASS holdout")
    if a_pass:
        w = a_pass[0]; p = w["params"]
        print(f"\n  ANGLE A WINNER: thr={p['walk_threshold']:.2f} strk={p['walk_streak']} cd={p['cooldown']}")
        print(f"    HOLDOUT: PF={w['holdout']['pf']:.2f} T={w['holdout']['trades']} "
              f"WR={w['holdout']['wr']:.1f}% Net=${w['holdout']['net_profit']:.2f}")
        print(f"    -> Proceed to Stage 2: combine with R029 MR")
    if b_pass:
        w = b_pass[0]; p = w["params"]
        print(f"\n  ANGLE B WINNER: lookback={p['slope_lookback']} threshold={p['slope_threshold']}")
        print(f"    HOLDOUT: PF={w['holdout']['pf']:.2f} T={w['holdout']['trades']} "
              f"WR={w['holdout']['wr']:.1f}% Net=${w['holdout']['net_profit']:.2f}")
        d_net = w['holdout']['net_profit'] - base_holdout['net_profit']
        print(f"    Improvement: dNet=${d_net:+.2f} ({d_net/base_holdout['net_profit']*100:+.1f}%)")
        print(f"    -> Update R029 in place with slope filter")
    if not a_pass and not b_pass:
        print(f"\n  BOTH ANGLES FAIL holdout. R029 MR-only design is the right shape for")
        print(f"  EURUSD 0.0006. Recommend pivoting to: ESCGO gate (C2), HTF threshold (C3),")
        print(f"  or testing brick streak on a non-EURUSD instrument (GBPJPY 0.05).")
    print(f"{'=' * 120}")
    print(f"\nNOTE: data ends 2026-03-25. April 7 trends in screenshots are NOT in this")
    print(f"      holdout. If a winner is found, refresh data before deploy.")

    # ── Save ────────────────────────────────────────────────────────────────
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump({
            "config": {
                "renko_file": RENKO_FILE,
                "train_window":    [TRAIN_START, TRAIN_END],
                "validate_window": [VALIDATE_START, VALIDATE_END],
                "holdout_window":  [HOLDOUT_START, HOLDOUT_END],
                "r029_params": R029_PARAMS,
                "angle_a_grid": ANGLE_A_GRID,
                "angle_b_grid": ANGLE_B_GRID,
                "rules_a": RULES_A,
                "rules_b": RULES_B,
            },
            "baseline": {
                "train":    base_train,
                "validate": base_validate,
                "holdout":  base_holdout,
            },
            "angle_a": {
                "results":   results_a,
                "finalists": [{"params": f["params"], "train": f["train"], "validate": f["validate"]}
                              for f in finalists_a],
                "holdout":   holdout_a,
            },
            "angle_b": {
                "results":   results_b,
                "finalists": [{"params": f["params"], "train": f["train"], "validate": f["validate"]}
                              for f in finalists_b],
                "holdout":   holdout_b,
            },
        }, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
