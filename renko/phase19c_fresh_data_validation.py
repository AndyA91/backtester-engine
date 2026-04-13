"""
Phase 19c — Fresh-Data Validation of R029 + Donchian Breakout Overlay

Purpose
-------
Phase 19b found that combining R029 with a Donchian breakout overlay
(lookback=60, inter_cooldown=0) increased HOLDOUT net profit by +78.7%
($69.88 → $124.85) with preserved per-trade economics ($1.17/trade for
breakout-added trades, matching R029's $1.18/trade) and zero standalone
overlap. Formally REJECTED by Phase 19b's locked WR ≥ 55% rule (combined
WR was 53.8%, 1.2pp under), but the rule was miscalibrated for an
additive overlay.

Phase 19c is the ONE clean test that resolves whether the +78.7% finding
is a real signal or a 4-sweep cherry-pick:

  - Refreshed data (must include April 7+ from the user's screenshot
    period that wasn't in Phase 17/18/19/19b's HOLDOUT)
  - NEW decision rules locked BEFORE viewing the new holdout
  - Test the 3 most promising configs from Phase 19b (the lookback=60
    cd=0 winner plus two reasonable neighbors)
  - Single test, no iteration

This is NOT iterating on the contaminated holdout. It's a fresh test
on a new data window with corrected rules, designed to either confirm
or definitively kill the trend overlay candidate.

Requires data refresh
---------------------
Before running, the user must refresh
    data/EURUSD/OANDA_EURUSD, 1S renko 0.0006.csv
to include data through April 7, 2026 or later (re-export from
TradingView). The current file ends 2026-03-25 and would NOT provide
a meaningful fresh-data test.

The script checks the data end date and refuses to run if it doesn't
include at least 2026-04-01.

Three-way splits (per R22)
--------------------------
  TRAIN     2023-01-02 -> 2025-09-30   (matches R029's IS boundary)
  VALIDATE  2025-10-01 -> 2025-12-31   (3 months — same as 17/18/19/19b)
  HOLDOUT   2026-01-01 -> 2026-03-25   (12 weeks — the contaminated old window)
  FRESH     2026-03-26 -> end of data  (NEW SEALED window — used for the decision)

Yes, this means we have FOUR splits. The HOLDOUT slot is reused for
context comparison only (it shows the Phase 19b numbers). The FRESH
window is what determines the deploy decision.

CORRECTED decision rules (fixed for additive overlays — NEW RULES)
------------------------------------------------------------------
Locked BEFORE viewing FRESH window:
  1. FRESH combined net >= R029 FRESH baseline net * 1.10  (>= +10%)
  2. FRESH combined PF >= R029 FRESH baseline PF * 0.85    (degradation <= 15%)
  3. Breakout entry fraction >= 30% of total entries
  4. FRESH combined trades >= R029 FRESH baseline trades   (no trade loss)
  5. NEW: FRESH combined max drawdown not worse than baseline by more than 50%

The WR rule is REMOVED because (per Phase 19b lesson) WR mechanically
degrades when adding lower-WR signals to a higher-WR base, which is the
arithmetic of weighted averages, not a quality signal.

Configs tested (3 only — no fishing)
------------------------------------
  1. lookback=60 inter_cd=0  (the Phase 19b near-miss winner)
  2. lookback=40 inter_cd=0  (slightly more aggressive — better PF preservation)
  3. lookback=60 inter_cd=5  (slightly more spaced — better WR in 19b but lower net)

3 configs × 4 windows = 12 backtests + R029 baseline 4 windows = 16 total.
Wallclock estimate: <5 seconds.

Usage
-----
    1. Refresh data file first (re-export EURUSD 0.0006 from TradingView)
    2. python renko/phase19c_fresh_data_validation.py
"""

import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

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

# ── Data splits ──────────────────────────────────────────────────────────────
TRAIN_START    = "2023-01-02"
TRAIN_END      = "2025-09-30"
VALIDATE_START = "2025-10-01"
VALIDATE_END   = "2025-12-31"
HOLDOUT_START  = "2026-01-01"
HOLDOUT_END    = "2026-03-25"   # the contaminated old window — for context
FRESH_START    = "2026-03-26"   # NEW sealed window — used for decision
FRESH_END      = "2099-12-31"   # whatever the refreshed data extends to

# Minimum end date required to consider data "refreshed"
MIN_FRESH_END = pd.Timestamp("2026-04-01")

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

OUTPUT_FILE = ROOT / "ai_context" / "phase19c_fresh_data_validation_results.json"

# ── R029 locked Pine params ─────────────────────────────────────────────────
R029_PARAMS = {
    "band_thresh": 0.20,
    "cooldown":    2,
    "rsi_thresh":  45,
    "adx_max":     25,
    "vol_max":     1.5,
}

# ── Breakout config locks (from Phase 19b winner config) ────────────────────
BREAKOUT_COOLDOWN_INTERNAL = 5  # locked from Phase 19b Angle B winner
BREAKOUT_USE_HIGH_LOW      = False
BREAKOUT_REQUIRE_BRICKDIR  = False

# ── 3 configs to test (no fishing — exactly the Phase 19b candidates) ──────
TEST_CONFIGS = [
    {"lookback": 60, "inter_cooldown": 0,  "label": "winner (Phase 19b best)"},
    {"lookback": 40, "inter_cooldown": 0,  "label": "more aggressive (more trades)"},
    {"lookback": 60, "inter_cooldown": 5,  "label": "more spaced (better WR in 19b)"},
]

# ── CORRECTED decision rules (locked BEFORE viewing FRESH) ──────────────────
RULES = {
    "min_net_multiplier":    1.10,   # FRESH net >= baseline net * 1.10
    "min_pf_multiplier":     0.85,   # FRESH PF >= baseline PF * 0.85 (15% degradation OK)
    "min_breakout_fraction": 0.30,   # breakout entries / total >= 30%
    "min_total_trade_ratio": 1.00,   # FRESH total trades >= baseline trades
    "max_dd_multiplier":     1.50,   # FRESH max DD not >50% worse than baseline
}


# ─── R029 standalone generator (same as Phase 19b) ─────────────────────────


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


# ─── Combined R029 + Donchian generator (same as Phase 19b) ────────────────


def gen_combined(df, breakout_lookback, inter_cooldown):
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
    entry_type  = np.zeros(n, dtype=np.int8)

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

        if pos == 1 and not up:
            long_exit[i] = True; pos = 0; last_trade_bar = i
        elif pos == -1 and up:
            short_exit[i] = True; pos = 0; last_trade_bar = i

        if pos != 0:
            continue
        if (i - last_trade_bar) < inter_cooldown:
            continue

        # R029 priority
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

        # Breakout fallback
        if (i - last_bo_bar) < BREAKOUT_COOLDOWN_INTERNAL:
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
    et = df_signals["_entry_type"].values
    n_r029 = int(np.sum(et == 1))
    n_breakout = int(np.sum(et == 2))
    return n_r029, n_breakout


def evaluate(combined, baseline, breakout_fraction):
    """Evaluate against CORRECTED decision rules."""
    failures = []
    base_net    = baseline["net_profit"]
    base_pf     = baseline["pf"]
    base_trades = baseline["trades"]
    base_dd     = baseline["max_dd"]

    if base_net <= 0:
        threshold_net = base_net + 5.0
    else:
        threshold_net = base_net * RULES["min_net_multiplier"]
    if combined["net_profit"] < threshold_net:
        failures.append(f"Net {combined['net_profit']:.2f} < {threshold_net:.2f} (baseline*1.10)")

    threshold_pf = base_pf * RULES["min_pf_multiplier"]
    if combined["pf"] < threshold_pf:
        failures.append(f"PF {combined['pf']:.2f} < {threshold_pf:.2f} (baseline*0.85)")

    if breakout_fraction < RULES["min_breakout_fraction"]:
        failures.append(f"Breakout fraction {breakout_fraction*100:.1f}% < {RULES['min_breakout_fraction']*100:.0f}%")

    if combined["trades"] < base_trades * RULES["min_total_trade_ratio"]:
        failures.append(f"Trades {combined['trades']} < {base_trades} (baseline)")

    # Max DD: baseline DD is negative; "1.5x worse" means DD <= base_dd * 1.5
    threshold_dd = base_dd * RULES["max_dd_multiplier"] if base_dd < 0 else base_dd - 50
    if combined["max_dd"] < threshold_dd:
        failures.append(f"MaxDD {combined['max_dd']:.2f} < {threshold_dd:.2f} (baseline*1.5 worse)")

    return (len(failures) == 0, failures)


def main():
    print(f"Loading {RENKO_FILE}...")
    df = load_renko_export(RENKO_FILE)
    end_date = df.index[-1]
    print(f"  {len(df):,} bricks loaded, end date: {end_date}")

    # ── Data freshness check ────────────────────────────────────────────────
    if end_date < MIN_FRESH_END:
        print(f"\n  ERROR: data end date ({end_date}) is BEFORE the minimum")
        print(f"  required fresh date ({MIN_FRESH_END.date()}). Phase 19c REQUIRES")
        print(f"  refreshed data through April 1+ to provide a meaningful fresh-window")
        print(f"  test that wasn't contaminated by Phase 17/18/19/19b sweeps.")
        print(f"\n  Action required:")
        print(f"    1. Re-export 'OANDA_EURUSD, 1S renko 0.0006' from TradingView")
        print(f"    2. Save to data/EURUSD/OANDA_EURUSD, 1S renko 0.0006.csv")
        print(f"    3. Re-run this script")
        print(f"\n  Aborting.")
        sys.exit(1)

    print("Computing indicators...")
    add_renko_indicators(df)

    df_train    = df.loc[TRAIN_START:TRAIN_END].copy()
    df_validate = df.loc[VALIDATE_START:VALIDATE_END].copy()
    df_holdout  = df.loc[HOLDOUT_START:HOLDOUT_END].copy()
    df_fresh    = df.loc[FRESH_START:FRESH_END].copy()

    print(f"  TRAIN:    {len(df_train):>6,} bricks  ({df_train.index[0]} -> {df_train.index[-1]})")
    print(f"  VALIDATE: {len(df_validate):>6,} bricks  ({df_validate.index[0]} -> {df_validate.index[-1]})")
    print(f"  HOLDOUT:  {len(df_holdout):>6,} bricks  ({df_holdout.index[0]} -> {df_holdout.index[-1]})  [contaminated old window]")
    print(f"  FRESH:    {len(df_fresh):>6,} bricks  ({df_fresh.index[0]} -> {df_fresh.index[-1]})  [SEALED — decides verdict]")

    if len(df_fresh) < 50:
        print(f"\n  ERROR: FRESH window has only {len(df_fresh)} bricks. Need at least 50 for")
        print(f"  a meaningful test. Re-export with data through April 7+ at minimum.")
        sys.exit(1)

    # ── R029 baseline on all 4 windows ───────────────────────────────────────
    print(f"\n{'=' * 120}")
    print(f"  R029 BASELINE (Python-native, locked Pine params)")
    print(f"{'=' * 120}")
    base_train    = run_window(gen_r029_baseline(df_train))
    base_validate = run_window(gen_r029_baseline(df_validate))
    base_holdout  = run_window(gen_r029_baseline(df_holdout))
    base_fresh    = run_window(gen_r029_baseline(df_fresh))
    for label, k in [("TRAIN", base_train), ("VALIDATE", base_validate),
                     ("HOLDOUT", base_holdout), ("FRESH", base_fresh)]:
        if k is None:
            print(f"  {label:<10} (too few trades)")
        else:
            print(f"  {label:<10} T={k['trades']:>4}  WR={k['wr']:>5.1f}%  PF={k['pf']:>6.2f}  "
                  f"Net=${k['net_profit']:>9.2f}  MaxDD=${k['max_dd']:>8.2f}  L/S={k['n_long']}/{k['n_short']}")

    if base_fresh is None or base_fresh["trades"] < 5:
        print(f"\n  ERROR: R029 baseline produced too few trades on FRESH window. Cannot evaluate.")
        sys.exit(1)

    # ── Run the 3 test configs on all 4 windows ──────────────────────────────
    print(f"\n{'=' * 120}")
    print(f"  COMBINED CONFIGS — TRAIN / VALIDATE / HOLDOUT / FRESH")
    print(f"{'=' * 120}")

    t0 = time.time()
    results = []
    for cfg in TEST_CONFIGS:
        lkbk = cfg["lookback"]
        intcd = cfg["inter_cooldown"]
        label = cfg["label"]

        train_df  = gen_combined(df_train,    lkbk, intcd)
        val_df    = gen_combined(df_validate, lkbk, intcd)
        hld_df    = gen_combined(df_holdout,  lkbk, intcd)
        fresh_df  = gen_combined(df_fresh,    lkbk, intcd)

        train_kpis = run_window(train_df)
        val_kpis   = run_window(val_df)
        hld_kpis   = run_window(hld_df)
        fresh_kpis = run_window(fresh_df)

        train_r029, train_bo = count_entry_types(train_df)
        val_r029,   val_bo   = count_entry_types(val_df)
        hld_r029,   hld_bo   = count_entry_types(hld_df)
        fresh_r029, fresh_bo = count_entry_types(fresh_df)

        results.append({
            "lookback": lkbk, "inter_cooldown": intcd, "label": label,
            "train": train_kpis, "validate": val_kpis,
            "holdout": hld_kpis, "fresh": fresh_kpis,
            "train_r029": train_r029, "train_bo": train_bo,
            "val_r029": val_r029, "val_bo": val_bo,
            "hld_r029": hld_r029, "hld_bo": hld_bo,
            "fresh_r029": fresh_r029, "fresh_bo": fresh_bo,
        })

    print(f"  Completed in {time.time()-t0:.1f}s")

    # ── Display HOLDOUT (contaminated, for context only) ─────────────────────
    print(f"\n{'=' * 120}")
    print(f"  CONTEXT — Old HOLDOUT (2026-01-01 to 2026-03-25)")
    print(f"  Baseline: T={base_holdout['trades']} WR={base_holdout['wr']:.1f}% "
          f"PF={base_holdout['pf']:.2f} Net=${base_holdout['net_profit']:.2f}")
    print(f"  (Already seen by sweeps 17/18/19/19b — shown only for continuity)")
    print(f"{'=' * 120}")
    for r in results:
        h = r["holdout"]
        if h is None: continue
        bo_frac = r["hld_bo"] / max(r["hld_r029"] + r["hld_bo"], 1)
        d_net = h["net_profit"] - base_holdout["net_profit"]
        d_pct = d_net / base_holdout["net_profit"] * 100 if base_holdout["net_profit"] != 0 else 0
        print(f"  lookback={r['lookback']:>2} inter_cd={r['inter_cooldown']:>2}: "
              f"T={h['trades']:>3} WR={h['wr']:>5.1f}% PF={h['pf']:>5.2f} "
              f"Net=${h['net_profit']:>7.2f} (dNet={d_net:+.2f}, {d_pct:+.1f}%) "
              f"BO%={bo_frac*100:>3.0f}%")

    # ── Display FRESH window — THE DECISION ──────────────────────────────────
    print(f"\n{'=' * 120}")
    print(f"  *** FRESH WINDOW *** ({df_fresh.index[0]} -> {df_fresh.index[-1]})")
    print(f"  Baseline: T={base_fresh['trades']} WR={base_fresh['wr']:.1f}% "
          f"PF={base_fresh['pf']:.2f} Net=${base_fresh['net_profit']:.2f} MaxDD=${base_fresh['max_dd']:.2f}")
    print(f"  THIS is what determines deploy decision.")
    print(f"{'=' * 120}")
    print(f"  Decision rules: Net>=baseline*1.10, PF>=baseline*0.85, BO%>=30%, T>=baseline, MaxDD>=baseline*1.5")
    print(f"")

    base_net   = base_fresh["net_profit"]
    base_pf    = base_fresh["pf"]
    base_t     = base_fresh["trades"]
    base_dd    = base_fresh["max_dd"]
    threshold_net = base_net * 1.10 if base_net > 0 else base_net + 5.0
    threshold_pf  = base_pf * 0.85
    threshold_dd  = base_dd * 1.5 if base_dd < 0 else base_dd - 50

    print(f"  Thresholds: Net>=${threshold_net:.2f}, PF>={threshold_pf:.2f}, "
          f"T>={base_t}, MaxDD>=${threshold_dd:.2f}")
    print(f"")

    passing = []
    for r in results:
        f = r["fresh"]
        if f is None:
            print(f"  REJECT [{r['label']}] lookback={r['lookback']} inter_cd={r['inter_cooldown']}")
            print(f"      x No trades in FRESH window")
            continue
        bo_frac = r["fresh_bo"] / max(r["fresh_r029"] + r["fresh_bo"], 1)
        d_net = f["net_profit"] - base_net
        d_pct = d_net / base_net * 100 if base_net != 0 else 0
        d_wr  = f["wr"] - base_fresh["wr"]
        d_pf  = f["pf"] - base_pf

        ok, failures = evaluate(f, base_fresh, bo_frac)
        decision = "PASS" if ok else "REJECT"

        print(f"  {decision} [{r['label']}] lookback={r['lookback']} inter_cd={r['inter_cooldown']}")
        print(f"      T={f['trades']:>3} WR={f['wr']:>5.1f}% PF={f['pf']:>5.2f} "
              f"Net=${f['net_profit']:>7.2f} MaxDD=${f['max_dd']:>7.2f} "
              f"R029={r['fresh_r029']} BO={r['fresh_bo']} BO%={bo_frac*100:>3.0f}%")
        print(f"      vs baseline: dNet={d_net:+.2f} ({d_pct:+.1f}%) dWR={d_wr:+.1f}pp "
              f"dPF={d_pf:+.2f}")
        if failures:
            for fmsg in failures:
                print(f"      x {fmsg}")
        if ok:
            passing.append(r)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 120}")
    print(f"  PHASE 19c SUMMARY — DEPLOY DECISION")
    print(f"{'=' * 120}")
    print(f"  FRESH-window passes: {len(passing)} / {len(TEST_CONFIGS)}")
    if passing:
        print(f"\n  REAL CANDIDATE FOUND on fresh window:")
        for w in passing:
            f = w["fresh"]
            d_net = f["net_profit"] - base_fresh["net_profit"]
            print(f"    lookback={w['lookback']} inter_cd={w['inter_cooldown']}: ")
            print(f"      FRESH PF={f['pf']:.2f} T={f['trades']} WR={f['wr']:.1f}% "
                  f"Net=${f['net_profit']:.2f} (+${d_net:.2f} vs baseline)")
            print(f"      Trade mix: R029={w['fresh_r029']} + Breakout={w['fresh_bo']}")
        print(f"\n  -> Next steps:")
        print(f"     1. Compute equity curve, examine drawdown clusters")
        print(f"     2. Write Pine for the combined strategy (R029 + Donchian breakout)")
        print(f"     3. Run Pine sanitization checklist (CLAUDE.md)")
        print(f"     4. TV-validate vs Python: <=5% trade count delta, <=10% net delta")
        print(f"     5. If TV matches, deploy as R029v2 with breakout overlay")
    else:
        print(f"\n  NO CONFIG PASSES on FRESH window.")
        print(f"  The Phase 19b +78.7% finding does NOT replicate on fresh data.")
        print(f"  Most likely interpretation: Phase 19b was a multi-comparison artifact")
        print(f"  from 4 sweeps on the same HOLDOUT window.")
        print(f"\n  R029 IS DEFINITIVELY DESIGN-LOCKED for EURUSD 0.0006.")
        print(f"  Pivot recommendations:")
        print(f"    - C2: ESCGO gate on R029 (most likely incremental improvement)")
        print(f"    - C3: HTF threshold sweep on USDJPY/GBPUSD")
        print(f"    - C1: brick streak on GBPJPY 0.05 (different instrument)")
        print(f"    - C5: bigger brick study on EURUSD (0.0008/0.0010)")
    print(f"{'=' * 120}")

    # ── Save ─────────────────────────────────────────────────────────────────
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump({
            "config": {
                "renko_file": RENKO_FILE,
                "data_end_date": str(end_date),
                "train_window":    [TRAIN_START, TRAIN_END],
                "validate_window": [VALIDATE_START, VALIDATE_END],
                "holdout_window":  [HOLDOUT_START, HOLDOUT_END],
                "fresh_window":    [FRESH_START, str(df_fresh.index[-1])],
                "r029_params": R029_PARAMS,
                "test_configs": TEST_CONFIGS,
                "rules": RULES,
            },
            "baseline": {
                "train": base_train, "validate": base_validate,
                "holdout": base_holdout, "fresh": base_fresh,
            },
            "results": results,
        }, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
