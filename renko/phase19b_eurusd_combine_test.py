"""
Phase 19b — R029 + Donchian Breakout Overlay Combine Test

Purpose
-------
Phase 19 produced two borderline "passes" — Angle A pullback (statistically
indistinguishable from coin flip) and Angle B Donchian breakout (51.1% WR
with 47 trades, plausibly real signal). The fundamental question that
Phase 17/18/19 NEVER asked is:

    Does R029 + breakout (combined) produce more net profit than R029 alone?

All prior sweeps tested STANDALONE strategies and compared their absolute
metrics to a fixed bar. None tested the actual deployment configuration
(R029 with a trend-continuation overlay layered on top). This sweep fixes
that gap.

Methodology
-----------
For each (breakout_lookback, inter_cooldown) combo:
  1. Run R029 BASELINE alone on TRAIN/VALIDATE/HOLDOUT
  2. Run R029+breakout COMBINED on TRAIN/VALIDATE/HOLDOUT
  3. Compute the delta: combined.net - baseline.net per window
  4. Track entry-type breakdown: how many trades came from R029 vs breakout
  5. Compute the standalone signal overlap (set intersection of R029 entry
     bars and breakout entry bars when run independently)

The combined generator has a state-aware loop:
  - Single position at a time (no pyramiding)
  - At each bar: check exit first
  - When flat and out of inter-strategy cooldown:
      - R029 has PRIORITY (its conditions checked first)
      - If R029 doesn't fire, check breakout conditions
      - Whichever fires gets entered, with entry_type tagged
  - Same first-opposing-brick exit (R1) for both entry types

Decision rules (locked BEFORE viewing HOLDOUT)
----------------------------------------------
Combined deployment requires:
  1. HOLDOUT combined net >= R029 baseline net * 1.10 (>= +10%)
  2. HOLDOUT combined WR >= 55%
  3. Breakout entry fraction >= 30% of total entries
     (otherwise the overlay is barely contributing — it's just R029)
  4. HOLDOUT combined trades >= R029 baseline trades (no trade loss)

Failing any of these = REJECT.

Three-way splits (per R22)
--------------------------
  TRAIN     2023-01-02 -> 2025-09-30
  VALIDATE  2025-10-01 -> 2025-12-31
  HOLDOUT   2026-01-01 -> 2026-03-25  (sealed during sweep)

NOTE: This is the FOURTH sweep using the same HOLDOUT window (after Phase
17, 18, 19). Multiple-comparison risk is now substantial. Even if a config
passes here, the deploy decision REQUIRES data refresh through April 7+
and one more re-run on the fresh window before any TV validation.

Param grid
----------
  breakout_lookback: [20, 30, 40, 60]   (4 — Angle B's better-performing values)
  inter_cooldown:    [0, 5, 10]          (3 — bricks between R029 exit and breakout entry)
  Total combos: 12
  Plus 1 R029 baseline run = 13 unique configs
  Backtests: 13 × 3 windows = 39

Wallclock estimate: ~10 seconds.

Usage
-----
    python renko/phase19b_eurusd_combine_test.py
"""

import contextlib
import io
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

# Force UTF-8 stdout on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import BacktestConfig, run_backtest_long_short
from renko.data import load_renko_export
from renko.indicators import add_renko_indicators

RENKO_FILE = "OANDA_EURUSD, 1S renko 0.0006.csv"

# ── Data splits (3-way per R22) ──────────────────────────────────────────────
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

OUTPUT_FILE = ROOT / "ai_context" / "phase19b_combine_test_results.json"

# ── R029 locked Pine params ─────────────────────────────────────────────────
R029_PARAMS = {
    "band_thresh": 0.20,
    "cooldown":    2,
    "rsi_thresh":  45,
    "adx_max":     25,
    "vol_max":     1.5,
}

# ── Param grid for the combine test ─────────────────────────────────────────
BREAKOUT_LOOKBACKS = [20, 30, 40, 60]
INTER_COOLDOWNS    = [0, 5, 10]
# Locked at Angle B winner's other params:
BREAKOUT_USE_HIGH_LOW    = False
BREAKOUT_COOLDOWN        = 5
BREAKOUT_REQUIRE_BRICKDIR = False

# ── Decision rules (locked BEFORE viewing holdout) ──────────────────────────
RULES = {
    "min_net_multiplier":      1.10,  # combined net >= baseline * 1.10
    "min_wr":                  55.0,  # combined WR >= 55%
    "min_breakout_fraction":   0.30,  # breakout entries / total >= 30%
    "min_total_trade_ratio":   1.00,  # combined trades >= baseline trades
}


# ─── R029 standalone generator (same as Phase 19) ──────────────────────────


def gen_r029_baseline(df):
    brick_up   = df["brick_up"].values
    pct_b      = df["bb_pct_b"].values
    rsi        = df["rsi"].values
    adx        = df["adx"].values
    vol_ratio  = df["vol_ratio"].values
    n = len(df)

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    warmup = 30
    last_trade_bar = -999_999
    pos = 0

    upper_band = 1.0 - R029_PARAMS["band_thresh"]
    lower_band = R029_PARAMS["band_thresh"]
    rsi_long_t  = R029_PARAMS["rsi_thresh"]
    rsi_short_t = 100.0 - R029_PARAMS["rsi_thresh"]
    cd          = R029_PARAMS["cooldown"]
    adx_max     = R029_PARAMS["adx_max"]
    vol_max     = R029_PARAMS["vol_max"]

    for i in range(warmup, n):
        up = bool(brick_up[i])
        if pos == 1 and not up:
            long_exit[i] = True; pos = 0
        elif pos == -1 and up:
            short_exit[i] = True; pos = 0
        if pos != 0:
            continue
        if (i - last_trade_bar) < cd:
            continue
        if not np.isnan(vol_ratio[i]) and vol_ratio[i] > vol_max:
            continue
        if not np.isnan(adx[i]) and adx[i] > adx_max:
            continue
        if np.isnan(pct_b[i]):
            continue
        if pct_b[i] <= lower_band and up:
            if not np.isnan(rsi[i]) and rsi[i] >= rsi_long_t:
                continue
            long_entry[i] = True; pos = 1; last_trade_bar = i
        elif pct_b[i] >= upper_band and not up:
            if not np.isnan(rsi[i]) and rsi[i] <= rsi_short_t:
                continue
            short_entry[i] = True; pos = -1; last_trade_bar = i

    df = df.copy()
    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df


# ─── Standalone breakout generator (same as Phase 19 Angle B) ──────────────


def gen_breakout_standalone(df, breakout_lookback):
    brick_up = df["brick_up"].values
    close    = df["Close"].values
    n = len(df)

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    warmup = max(breakout_lookback + 2, 30)
    last_exit_bar = -999_999
    pos = 0

    for i in range(warmup, n):
        up = bool(brick_up[i])
        if pos == 1 and not up:
            long_exit[i] = True; pos = 0; last_exit_bar = i
        elif pos == -1 and up:
            short_exit[i] = True; pos = 0; last_exit_bar = i
        if pos != 0:
            continue
        if (i - last_exit_bar) < BREAKOUT_COOLDOWN:
            continue

        upper_extreme = float(np.max(close[i - breakout_lookback:i]))
        lower_extreme = float(np.min(close[i - breakout_lookback:i]))

        if close[i] > upper_extreme:
            long_entry[i] = True; pos = 1; last_exit_bar = i
        elif close[i] < lower_extreme:
            short_entry[i] = True; pos = -1; last_exit_bar = i

    df = df.copy()
    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df


# ─── COMBINED generator: R029 priority + Donchian breakout overlay ─────────


def gen_combined(df, breakout_lookback, inter_cooldown):
    """Combined R029 + breakout. R029 has priority. Single position at a time.

    Tracks entry_type for each trade so we can count R029 vs breakout entries
    in the post-processing. Same first-opposing-brick exit for both.
    """
    brick_up   = df["brick_up"].values
    pct_b      = df["bb_pct_b"].values
    rsi        = df["rsi"].values
    adx        = df["adx"].values
    vol_ratio  = df["vol_ratio"].values
    close      = df["Close"].values
    n = len(df)

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)
    entry_type  = np.zeros(n, dtype=np.int8)  # 0=none, 1=r029, 2=breakout

    warmup = max(breakout_lookback + 2, 30)
    last_trade_bar = -999_999
    last_r029_bar  = -999_999
    last_bo_bar    = -999_999
    pos = 0

    upper_band = 1.0 - R029_PARAMS["band_thresh"]
    lower_band = R029_PARAMS["band_thresh"]
    rsi_long_t  = R029_PARAMS["rsi_thresh"]
    rsi_short_t = 100.0 - R029_PARAMS["rsi_thresh"]
    r029_cd     = R029_PARAMS["cooldown"]
    adx_max     = R029_PARAMS["adx_max"]
    vol_max     = R029_PARAMS["vol_max"]

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # Exit (same for both entry types)
        if pos == 1 and not up:
            long_exit[i] = True; pos = 0; last_trade_bar = i
        elif pos == -1 and up:
            short_exit[i] = True; pos = 0; last_trade_bar = i

        if pos != 0:
            continue
        if (i - last_trade_bar) < inter_cooldown:
            continue

        # === R029 entry conditions (PRIORITY) ===
        r029_long  = False
        r029_short = False
        if (i - last_r029_bar) >= r029_cd:
            vol_ok = np.isnan(vol_ratio[i]) or vol_ratio[i] <= vol_max
            adx_ok = np.isnan(adx[i]) or adx[i] <= adx_max
            if vol_ok and adx_ok and not np.isnan(pct_b[i]):
                if pct_b[i] <= lower_band and up:
                    if np.isnan(rsi[i]) or rsi[i] < rsi_long_t:
                        r029_long = True
                elif pct_b[i] >= upper_band and not up:
                    if np.isnan(rsi[i]) or rsi[i] > rsi_short_t:
                        r029_short = True

        if r029_long:
            long_entry[i] = True; pos = 1
            entry_type[i] = 1
            last_trade_bar = i; last_r029_bar = i
            continue
        if r029_short:
            short_entry[i] = True; pos = -1
            entry_type[i] = 1
            last_trade_bar = i; last_r029_bar = i
            continue

        # === Breakout entry conditions (R029 didn't fire) ===
        if (i - last_bo_bar) < BREAKOUT_COOLDOWN:
            continue

        upper_extreme = float(np.max(close[i - breakout_lookback:i]))
        lower_extreme = float(np.min(close[i - breakout_lookback:i]))

        if close[i] > upper_extreme:
            long_entry[i] = True; pos = 1
            entry_type[i] = 2
            last_trade_bar = i; last_bo_bar = i
        elif close[i] < lower_extreme:
            short_entry[i] = True; pos = -1
            entry_type[i] = 2
            last_trade_bar = i; last_bo_bar = i

    df = df.copy()
    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    df["_entry_type"] = entry_type
    return df


# ── Backtest harness ────────────────────────────────────────────────────────


def run_window(df_signals):
    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest_long_short(df_signals, BACKTEST_CONFIG)
    n_trades = kpis.get("total_trades", 0)
    if n_trades < 1:
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


def count_entry_types(df_signals):
    """Count how many of the entries were R029 (type=1) vs breakout (type=2)."""
    et = df_signals["_entry_type"].values
    n_r029 = int(np.sum(et == 1))
    n_breakout = int(np.sum(et == 2))
    return n_r029, n_breakout


def standalone_overlap(df, breakout_lookback):
    """Compute the standalone signal overlap: of bars where breakout fires
    when run alone, how many also have R029 firing when run alone."""
    df_r029 = gen_r029_baseline(df)
    df_bo   = gen_breakout_standalone(df, breakout_lookback)

    r029_bars = set(np.where(df_r029["long_entry"].values | df_r029["short_entry"].values)[0])
    bo_bars   = set(np.where(df_bo["long_entry"].values   | df_bo["short_entry"].values)[0])

    if not bo_bars:
        return 0.0, 0, 0
    overlap = len(r029_bars & bo_bars)
    return overlap / len(bo_bars), len(r029_bars), len(bo_bars)


def evaluate(combined_holdout, baseline_holdout, breakout_fraction):
    failures = []
    base_net = baseline_holdout["net_profit"]
    if base_net <= 0:
        threshold_net = base_net + 5.0
    else:
        threshold_net = base_net * RULES["min_net_multiplier"]
    if combined_holdout["net_profit"] < threshold_net:
        failures.append(f"Net {combined_holdout['net_profit']:.2f} < {threshold_net:.2f} (baseline*1.10)")
    if combined_holdout["wr"] < RULES["min_wr"]:
        failures.append(f"WR {combined_holdout['wr']:.1f} < {RULES['min_wr']}")
    if breakout_fraction < RULES["min_breakout_fraction"]:
        failures.append(f"Breakout fraction {breakout_fraction*100:.1f}% < {RULES['min_breakout_fraction']*100:.0f}%")
    if combined_holdout["trades"] < baseline_holdout["trades"] * RULES["min_total_trade_ratio"]:
        failures.append(f"Trades {combined_holdout['trades']} < {baseline_holdout['trades']} (baseline)")
    return (len(failures) == 0, failures)


def main():
    print(f"Loading {RENKO_FILE}...")
    df = load_renko_export(RENKO_FILE)
    print(f"  {len(df):,} bricks loaded")

    print("Computing indicators...")
    add_renko_indicators(df)

    df_train    = df.loc[TRAIN_START:TRAIN_END].copy()
    df_validate = df.loc[VALIDATE_START:VALIDATE_END].copy()
    df_holdout  = df.loc[HOLDOUT_START:HOLDOUT_END].copy()
    print(f"  TRAIN:    {len(df_train):>6,} bricks")
    print(f"  VALIDATE: {len(df_validate):>6,} bricks")
    print(f"  HOLDOUT:  {len(df_holdout):>6,} bricks  [SEALED]")

    # ── R029 BASELINE on all 3 windows ───────────────────────────────────────
    print(f"\n{'=' * 120}")
    print(f"  R029 BASELINE (Python-native, locked Pine params)")
    print(f"{'=' * 120}")
    base_train    = run_window(gen_r029_baseline(df_train))
    base_validate = run_window(gen_r029_baseline(df_validate))
    base_holdout  = run_window(gen_r029_baseline(df_holdout))
    for label, k in [("TRAIN", base_train), ("VALIDATE", base_validate), ("HOLDOUT", base_holdout)]:
        if k is None:
            print(f"  {label:<10} (too few trades)")
        else:
            print(f"  {label:<10} T={k['trades']:>4}  WR={k['wr']:>5.1f}%  PF={k['pf']:>6.2f}  "
                  f"Net=${k['net_profit']:>9.2f}  L/S={k['n_long']}/{k['n_short']}")

    # ── Standalone overlap diagnostic ────────────────────────────────────────
    print(f"\n{'=' * 120}")
    print(f"  STANDALONE SIGNAL OVERLAP (HOLDOUT)")
    print(f"  Of the bars where breakout fires alone, how many also have R029 firing alone?")
    print(f"{'=' * 120}")
    print(f"  {'Lookback':>10} | {'R029 bars':>10} | {'BO bars':>10} | {'Overlap %':>12}")
    overlap_data = {}
    for lookback in BREAKOUT_LOOKBACKS:
        ov_pct, n_r029, n_bo = standalone_overlap(df_holdout, lookback)
        overlap_data[lookback] = (ov_pct, n_r029, n_bo)
        print(f"  {lookback:>10} | {n_r029:>10} | {n_bo:>10} | {ov_pct*100:>11.1f}%")

    # ── Combined sweep ───────────────────────────────────────────────────────
    print(f"\n{'=' * 120}")
    print(f"  COMBINED SWEEP — R029 + Donchian breakout overlay")
    print(f"{'=' * 120}")
    combos = list(itertools.product(BREAKOUT_LOOKBACKS, INTER_COOLDOWNS))
    print(f"  {len(combos)} combos × 3 windows = {len(combos)*3} backtests")

    t0 = time.time()
    results = []
    for lookback, inter_cd in combos:
        train_df_signals = gen_combined(df_train, lookback, inter_cd)
        val_df_signals   = gen_combined(df_validate, lookback, inter_cd)
        hld_df_signals   = gen_combined(df_holdout, lookback, inter_cd)

        train_kpis = run_window(train_df_signals)
        val_kpis   = run_window(val_df_signals)
        hld_kpis   = run_window(hld_df_signals)

        train_r029, train_bo = count_entry_types(train_df_signals)
        val_r029,   val_bo   = count_entry_types(val_df_signals)
        hld_r029,   hld_bo   = count_entry_types(hld_df_signals)

        results.append({
            "lookback": lookback,
            "inter_cooldown": inter_cd,
            "train":    train_kpis,
            "validate": val_kpis,
            "holdout":  hld_kpis,
            "train_r029": train_r029, "train_bo": train_bo,
            "val_r029": val_r029, "val_bo": val_bo,
            "hld_r029": hld_r029, "hld_bo": hld_bo,
        })

    print(f"  Completed in {time.time()-t0:.1f}s")

    # ── Display results ──────────────────────────────────────────────────────
    print(f"\n{'=' * 120}")
    print(f"  COMBINED RESULTS — HOLDOUT (the deploy-decision window)")
    print(f"  Baseline: T={base_holdout['trades']} WR={base_holdout['wr']:.1f}% "
          f"PF={base_holdout['pf']:.2f} Net=${base_holdout['net_profit']:.2f}")
    print(f"{'=' * 120}")
    print(f"  {'LkBk':>5} {'IntCd':>5} | "
          f"{'PF':>6} {'T':>4} {'WR':>6} {'Net':>9} | "
          f"{'R029':>5} {'BO':>4} {'BO%':>5} | "
          f"{'dNet$':>9} {'dNet%':>7} {'dWR':>6} {'dT':>5} | DECISION")
    print(f"  {'-'*12} | {'-'*30} | {'-'*15} | {'-'*32} | --------")
    for r in results:
        h = r["holdout"]
        n_r029 = r["hld_r029"]
        n_bo   = r["hld_bo"]
        total  = n_r029 + n_bo
        bo_frac = n_bo / total if total > 0 else 0.0
        d_net = h["net_profit"] - base_holdout["net_profit"]
        d_net_pct = d_net / base_holdout["net_profit"] * 100 if base_holdout["net_profit"] != 0 else 0
        d_wr  = h["wr"] - base_holdout["wr"]
        d_t   = h["trades"] - base_holdout["trades"]

        ok, failures = evaluate(h, base_holdout, bo_frac)
        decision = "PASS" if ok else "REJECT"

        print(f"  {r['lookback']:>5} {r['inter_cooldown']:>5} | "
              f"{h['pf']:>6.2f} {h['trades']:>4} {h['wr']:>5.1f}% {h['net_profit']:>9.2f} | "
              f"{n_r029:>5} {n_bo:>4} {bo_frac*100:>4.0f}% | "
              f"{d_net:>+9.2f} {d_net_pct:>+6.1f}% {d_wr:>+5.1f} {d_t:>+5} | "
              f"{decision}")
        if failures:
            for f in failures:
                print(f"      x {f}")

    # ── VALIDATE window for context ──────────────────────────────────────────
    print(f"\n{'=' * 120}")
    print(f"  COMBINED RESULTS — VALIDATE (for sanity checking)")
    print(f"  Baseline: T={base_validate['trades']} WR={base_validate['wr']:.1f}% "
          f"PF={base_validate['pf']:.2f} Net=${base_validate['net_profit']:.2f}")
    print(f"{'=' * 120}")
    print(f"  {'LkBk':>5} {'IntCd':>5} | "
          f"{'PF':>6} {'T':>4} {'WR':>6} {'Net':>9} | "
          f"{'R029':>5} {'BO':>4} {'BO%':>5} | {'dNet$':>9} {'dWR':>6}")
    print(f"  {'-'*12} | {'-'*30} | {'-'*15} | {'-'*16}")
    for r in results:
        v = r["validate"]
        n_r029 = r["val_r029"]
        n_bo   = r["val_bo"]
        total  = n_r029 + n_bo
        bo_frac = n_bo / total if total > 0 else 0.0
        d_net = v["net_profit"] - base_validate["net_profit"]
        d_wr  = v["wr"] - base_validate["wr"]
        print(f"  {r['lookback']:>5} {r['inter_cooldown']:>5} | "
              f"{v['pf']:>6.2f} {v['trades']:>4} {v['wr']:>5.1f}% {v['net_profit']:>9.2f} | "
              f"{n_r029:>5} {n_bo:>4} {bo_frac*100:>4.0f}% | "
              f"{d_net:>+9.2f} {d_wr:>+5.1f}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 120}")
    print(f"  PHASE 19b SUMMARY")
    print(f"{'=' * 120}")
    passing = []
    for r in results:
        h = r["holdout"]
        bo_frac = r["hld_bo"] / max(r["hld_r029"] + r["hld_bo"], 1)
        ok, _ = evaluate(h, base_holdout, bo_frac)
        if ok:
            passing.append(r)

    print(f"  Configs that PASS HOLDOUT decision rules: {len(passing)} / {len(results)}")
    print(f"  Decision rules: combined_net >= baseline_net * 1.10 ($)")
    print(f"                  combined_WR >= 55%")
    print(f"                  breakout_fraction >= 30%")
    print(f"                  combined_trades >= baseline_trades")

    if passing:
        print(f"\n  WINNERS:")
        for w in passing:
            h = w["holdout"]
            bo_frac = w["hld_bo"] / max(w["hld_r029"] + w["hld_bo"], 1)
            d_net = h["net_profit"] - base_holdout["net_profit"]
            print(f"    lookback={w['lookback']} inter_cd={w['inter_cooldown']}: "
                  f"PF={h['pf']:.2f} T={h['trades']} WR={h['wr']:.1f}% "
                  f"Net=${h['net_profit']:.2f} (dNet=${d_net:+.2f}) "
                  f"BO%={bo_frac*100:.0f}%")
        print(f"\n  -> CANDIDATE for fresh-data validation")
        print(f"     IMPORTANT: this is the 4th sweep on the same HOLDOUT.")
        print(f"     Refresh data through April 7+ and re-run before any deploy decision.")
    else:
        print(f"\n  NO COMBINED CONFIG PASSES HOLDOUT decision rules.")
        print(f"  R029 baseline alone is the best strategy for this brick.")
        print(f"  R029 is design-locked. Pivot recommendations:")
        print(f"    - C2: ESCGO gate on R029 (most likely incremental improvement)")
        print(f"    - C3: HTF threshold on USDJPY/GBPUSD")
        print(f"    - C1: brick streak on GBPJPY 0.05 (different instrument)")
        print(f"    - C5: bigger brick study on EURUSD (0.0008/0.0010)")

    print(f"{'=' * 120}")
    print(f"\nNOTE: data ends 2026-03-25. April 7 trends NOT in this holdout.")

    # ── Save ─────────────────────────────────────────────────────────────────
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump({
            "config": {
                "renko_file": RENKO_FILE,
                "train_window":    [TRAIN_START, TRAIN_END],
                "validate_window": [VALIDATE_START, VALIDATE_END],
                "holdout_window":  [HOLDOUT_START, HOLDOUT_END],
                "r029_params": R029_PARAMS,
                "breakout_lookbacks": BREAKOUT_LOOKBACKS,
                "inter_cooldowns": INTER_COOLDOWNS,
                "rules": RULES,
            },
            "baseline": {"train": base_train, "validate": base_validate, "holdout": base_holdout},
            "overlap_diagnostic": {str(k): list(v) for k, v in overlap_data.items()},
            "results": results,
        }, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
