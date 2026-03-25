#!/usr/bin/env python3
"""
btc_phase6_exit_sweep.py -- BTC Phase 6: Exit Strategy Optimization (Long Only)

Entry fixed to BTC003 winner: R007 + FLIP_supertrend + BB_break
Gates fixed: psar_dir + ADX>=30 + vol<=1.5 + HTF ADX>=35
Params fixed: n=2, cd=30

Sweeps exit strategies:
  Brick-based:
    down_1        First down brick (baseline)
    down_2        2 consecutive down bricks
    down_3        3 consecutive down bricks
    trail_1of3    1 down brick out of last 3 triggers exit
    trail_2of3    2 down bricks out of last 3 triggers exit

  Indicator-based (exit when indicator flips bearish):
    psar_flip     PSAR flips bearish
    st_flip       Supertrend flips bearish
    ema_cross     EMA9 crosses below EMA21
    macd_neg      MACD histogram turns negative
    rsi_below_60  RSI drops below 60
    rsi_below_50  RSI drops below 50

  Conditional brick (down brick + indicator confirms):
    down_1+psar   Down brick AND PSAR bearish
    down_1+st     Down brick AND ST bearish
    down_1+macd   Down brick AND MACD hist negative
    down_1+ema    Down brick AND EMA9 < EMA21

  Time-based:
    max_10        Force exit after 10 bricks in position (fallback to down_1)
    max_20        Force exit after 20 bricks
    max_30        Force exit after 30 bricks

Uses ProcessPoolExecutor -- one worker per exit group.

Usage:
    python renko/btc_phase6_exit_sweep.py
    python renko/btc_phase6_exit_sweep.py --no-parallel
"""

import argparse
import contextlib
import io
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from renko.config import MAX_WORKERS
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# -- Instrument config ---------------------------------------------------------

LTF_FILE   = "OANDA_BTCUSD.SPOT.US, 1S renko 150.csv"
HTF_FILE   = "OANDA_BTCUSD.SPOT.US, 1S renko 300.csv"
IS_START   = "2024-06-04"
IS_END     = "2025-09-30"
OOS_START  = "2025-10-01"
OOS_END    = "2026-03-19"
COMMISSION = 0.0046
CAPITAL    = 1000.0
QTY_VALUE  = 20

# Fixed BTC003 entry params
N_BRICKS   = 2
COOLDOWN   = 30
VOL_MAX    = 1.5
ADX_THRESH = 30
HTF_ADX_THRESH = 35

# -- Exit strategies to test ---------------------------------------------------

EXIT_STRATEGIES = [
    # Brick-based
    "down_1", "down_2", "down_3",
    "trail_1of3", "trail_2of3",
    # Indicator-based
    "psar_flip", "st_flip", "ema_cross", "macd_neg",
    "rsi_below_60", "rsi_below_50",
    # Conditional brick
    "down_1+psar", "down_1+st", "down_1+macd", "down_1+ema",
    # Time-limited (with down_1 base)
    "max_10", "max_20", "max_30",
]


# -- Data loading --------------------------------------------------------------

def _load_ltf_data():
    sys.path.insert(0, str(ROOT))
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    from renko.phase6_enrichment import add_phase6_indicators
    df = load_renko_export(LTF_FILE)
    add_renko_indicators(df)
    add_phase6_indicators(df, include_mk=False)
    return df


def _load_htf_data():
    sys.path.insert(0, str(ROOT))
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    df = load_renko_export(HTF_FILE)
    add_renko_indicators(df)
    return df


# -- Gate computation (fixed BTC003 gates) ------------------------------------

def _compute_gates(df_ltf, df_htf):
    sys.path.insert(0, str(ROOT))
    from renko.phase6_sweep import _compute_gate_arrays

    n = len(df_ltf)
    gate = np.ones(n, dtype=bool)

    p6_long, _ = _compute_gate_arrays(df_ltf, "psar_dir")
    gate &= p6_long

    adx = df_ltf["adx"].values
    gate &= (np.isnan(adx) | (adx >= ADX_THRESH))

    vr = df_ltf["vol_ratio"].values
    gate &= (np.isnan(vr) | (vr <= VOL_MAX))

    # HTF ADX >= 35
    htf_adx = df_htf["adx"].values
    htf_gate = np.isnan(htf_adx) | (htf_adx >= HTF_ADX_THRESH)

    htf_frame = pd.DataFrame({
        "t": df_htf.index.values,
        "g": htf_gate.astype(float),
    }).sort_values("t")
    ltf_frame = pd.DataFrame({"t": df_ltf.index.values}).sort_values("t")
    merged = pd.merge_asof(ltf_frame, htf_frame, on="t", direction="backward")
    g = merged["g"].values
    htf_aligned = np.where(np.isnan(g), True, g > 0.5).astype(bool)
    gate &= htf_aligned

    return gate


# -- Precompute indicator arrays for exits ------------------------------------

def _precompute_exit_indicators(df):
    """Precompute all indicator arrays needed by exit strategies."""
    return {
        "brick_up": df["brick_up"].values,
        "psar_dir": df["psar_dir"].values,
        "st_dir":   df["st_dir"].values,
        "ema9":     df["ema9"].values,
        "ema21":    df["ema21"].values,
        "macd_hist": df["macd_hist"].values,
        "rsi":      df["rsi"].values,
    }


# -- Exit logic ----------------------------------------------------------------

def _should_exit(i, bars_in_pos, exit_name, ind, brick_history):
    """Return True if we should exit at bar i given the exit strategy."""
    up = bool(ind["brick_up"][i])

    if exit_name == "down_1":
        return not up

    elif exit_name == "down_2":
        if i < 1:
            return False
        return not up and not ind["brick_up"][i-1]

    elif exit_name == "down_3":
        if i < 2:
            return False
        return not up and not ind["brick_up"][i-1] and not ind["brick_up"][i-2]

    elif exit_name == "trail_1of3":
        if len(brick_history) < 3:
            return not up
        last3 = brick_history[-3:]
        down_count = sum(1 for b in last3 if not b)
        return down_count >= 1 and not up

    elif exit_name == "trail_2of3":
        if len(brick_history) < 3:
            return not up
        last3 = brick_history[-3:]
        down_count = sum(1 for b in last3 if not b)
        return down_count >= 2

    elif exit_name == "psar_flip":
        p = ind["psar_dir"][i]
        return not np.isnan(p) and p < 0

    elif exit_name == "st_flip":
        s = ind["st_dir"][i]
        return not np.isnan(s) and s < 0

    elif exit_name == "ema_cross":
        e9 = ind["ema9"][i]
        e21 = ind["ema21"][i]
        return (not np.isnan(e9) and not np.isnan(e21) and e9 < e21)

    elif exit_name == "macd_neg":
        mh = ind["macd_hist"][i]
        return not np.isnan(mh) and mh < 0

    elif exit_name == "rsi_below_60":
        r = ind["rsi"][i]
        return not np.isnan(r) and r < 60

    elif exit_name == "rsi_below_50":
        r = ind["rsi"][i]
        return not np.isnan(r) and r < 50

    elif exit_name == "down_1+psar":
        if up:
            return False
        p = ind["psar_dir"][i]
        return not np.isnan(p) and p < 0

    elif exit_name == "down_1+st":
        if up:
            return False
        s = ind["st_dir"][i]
        return not np.isnan(s) and s < 0

    elif exit_name == "down_1+macd":
        if up:
            return False
        mh = ind["macd_hist"][i]
        return not np.isnan(mh) and mh < 0

    elif exit_name == "down_1+ema":
        if up:
            return False
        e9 = ind["ema9"][i]
        e21 = ind["ema21"][i]
        return (not np.isnan(e9) and not np.isnan(e21) and e9 < e21)

    elif exit_name == "max_10":
        if bars_in_pos >= 10:
            return True
        return not up

    elif exit_name == "max_20":
        if bars_in_pos >= 20:
            return True
        return not up

    elif exit_name == "max_30":
        if bars_in_pos >= 30:
            return True
        return not up

    return not up  # fallback


# -- ST flip and BB break arrays (from Phase 5) --------------------------------

def _compute_st_flip(df):
    n = len(df)
    st = df["st_dir"].values
    flip = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not np.isnan(st[i]) and not np.isnan(st[i-1]):
            flip[i] = st[i] > 0 and st[i-1] <= 0
    return flip


def _compute_bb_break(df):
    n = len(df)
    close = df["Close"].values.astype(float)
    bb_u = df["bb_upper"].values
    sig = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not np.isnan(bb_u[i]) and not np.isnan(bb_u[i-1]):
            sig[i] = close[i] > bb_u[i] and close[i-1] <= bb_u[i-1]
    return sig


# -- Combined signal generator with variable exit -----------------------------

def _gen_signals(df, gate, st_flip, bb_break, exit_name, ind):
    """BTC003 entry + variable exit strategy."""
    n = len(df)
    brick_up = ind["brick_up"]
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)

    in_pos = False
    last_r001 = -999_999
    last_flip = -999_999
    last_bb = -999_999
    bars_in_pos = 0
    brick_history = []  # tracks brick directions while in position
    warmup = max(N_BRICKS + 1, 200)

    for i in range(warmup, n):
        up = bool(brick_up[i])

        if in_pos:
            bars_in_pos += 1
            brick_history.append(up)

            if _should_exit(i, bars_in_pos, exit_name, ind, brick_history):
                exit_[i] = True
                in_pos = False
                bars_in_pos = 0
                brick_history = []

        if in_pos:
            continue
        if not gate[i]:
            continue

        triggered = False

        # R002: reversal (no cooldown)
        prev = brick_up[i - N_BRICKS : i]
        prev_all_down = bool(not np.any(prev))
        if prev_all_down and up:
            triggered = True

        # R001: momentum (cooldown)
        if not triggered and (i - last_r001) >= COOLDOWN:
            window = brick_up[i - N_BRICKS + 1 : i + 1]
            if bool(np.all(window)):
                triggered = True
                last_r001 = i

        # FLIP: supertrend (own cooldown)
        if not triggered and up and st_flip[i] and (i - last_flip) >= COOLDOWN:
            triggered = True
            last_flip = i

        # BBRK: BB breakout (own cooldown)
        if not triggered and up and bb_break[i] and (i - last_bb) >= COOLDOWN:
            triggered = True
            last_bb = i

        if triggered:
            entry[i] = True
            in_pos = True
            bars_in_pos = 0
            brick_history = []

    return entry, exit_


# -- Backtest runner -----------------------------------------------------------

def _run_backtest(df, entry, exit_, start, end):
    sys.path.insert(0, str(ROOT))
    from engine import BacktestConfig, run_backtest

    df_sig = df.copy()
    df_sig["long_entry"] = entry
    df_sig["long_exit"] = exit_

    cfg = BacktestConfig(
        initial_capital=CAPITAL, commission_pct=COMMISSION, slippage_ticks=0,
        qty_type="cash", qty_value=QTY_VALUE, pyramiding=1,
        start_date=start, end_date=end,
        take_profit_pct=0.0, stop_loss_pct=0.0,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest(df_sig, cfg)

    pf = kpis.get("profit_factor", 0.0) or 0.0
    return {
        "pf":     float("inf") if math.isinf(pf) else float(pf),
        "net":    float(kpis.get("net_profit", 0.0) or 0.0),
        "trades": int(kpis.get("total_trades", 0) or 0),
        "wr":     float(kpis.get("win_rate", 0.0) or 0.0),
        "dd":     float(kpis.get("max_drawdown_pct", 0.0) or 0.0),
        "avg_bars": float(kpis.get("avg_bars_in_trade", 0.0) or 0.0),
    }


# -- Worker: sweep a group of exit strategies ----------------------------------

def _sweep_exit_group(exit_list, label):
    """Run all exit strategies in this group."""
    print(f"  [{label}] Loading data...", flush=True)

    df_ltf = _load_ltf_data()
    df_htf = _load_htf_data()
    gate = _compute_gates(df_ltf, df_htf)
    st_flip = _compute_st_flip(df_ltf)
    bb_break = _compute_bb_break(df_ltf)
    ind = _precompute_exit_indicators(df_ltf)

    print(f"  [{label}] Data ready, {len(exit_list)} exits to test", flush=True)

    results = []
    for exit_name in exit_list:
        e, x = _gen_signals(df_ltf, gate, st_flip, bb_break, exit_name, ind)

        is_r  = _run_backtest(df_ltf, e, x, IS_START, IS_END)
        oos_r = _run_backtest(df_ltf, e, x, OOS_START, OOS_END)

        is_pf = is_r["pf"]
        oos_pf = oos_r["pf"]
        decay = ((oos_pf - is_pf) / is_pf * 100) \
                if is_pf > 0 and not math.isinf(is_pf) else float("nan")

        results.append({
            "exit":       exit_name,
            "is_pf":      is_pf,
            "is_trades":  is_r["trades"],
            "is_wr":      is_r["wr"],
            "is_net":     is_r["net"],
            "oos_pf":     oos_pf,
            "oos_trades": oos_r["trades"],
            "oos_wr":     oos_r["wr"],
            "oos_net":    oos_r["net"],
            "oos_dd":     oos_r["dd"],
            "decay_pct":  decay,
        })

        pf_s = f"{oos_pf:.2f}" if not math.isinf(oos_pf) else "inf"
        print(f"  [{label}] {exit_name:<20} | "
              f"IS PF={is_pf:>7.2f} T={is_r['trades']:>4} WR={is_r['wr']:>5.1f}% | "
              f"OOS PF={pf_s:>8} T={oos_r['trades']:>4} WR={oos_r['wr']:>5.1f}%",
              flush=True)

    print(f"  [{label}] Done -- {len(results)} results", flush=True)
    return results


# -- Summary -------------------------------------------------------------------

def _summarize(all_results):
    print(f"\n{'='*100}")
    print("  BTC Phase 6 -- Exit Strategy Optimization (Long Only)")
    print(f"  Entry: R007 + FLIP_ST + BB_break | n=2, cd=30")
    print(f"  Gates: psar_dir + ADX>=30 + vol<=1.5 + HTF ADX>=35")
    print(f"{'='*100}")

    # Sort by OOS PF
    by_pf = sorted(all_results, key=lambda r: r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6, reverse=True)

    print(f"\n  {'Exit Strategy':<20} | {'IS PF':>7} {'T':>4} {'WR%':>6} {'Net':>8} | "
          f"{'OOS PF':>8} {'T':>4} {'WR%':>6} {'Net':>8} {'DD%':>7} {'Decay':>7}")
    print(f"  {'-'*100}")

    baseline_pf = None
    for r in by_pf:
        if r["exit"] == "down_1":
            baseline_pf = r["oos_pf"]

    for r in by_pf:
        pf_s = f"{r['oos_pf']:>8.2f}" if not math.isinf(r["oos_pf"]) else "     inf"
        is_pf_s = f"{r['is_pf']:>7.2f}" if not math.isinf(r["is_pf"]) else "    inf"
        dec_s = f"{r['decay_pct']:>+6.1f}%" if not math.isnan(r["decay_pct"]) else "    NaN"

        # Highlight if better than baseline
        vs_bl = ""
        if baseline_pf and not math.isinf(r["oos_pf"]) and r["exit"] != "down_1":
            delta = (r["oos_pf"] - baseline_pf) / baseline_pf * 100
            vs_bl = f" ({delta:>+.0f}%)"

        print(f"  {r['exit']:<20} | {is_pf_s} {r['is_trades']:>4} {r['is_wr']:>5.1f}% {r['is_net']:>8.2f} | "
              f"{pf_s} {r['oos_trades']:>4} {r['oos_wr']:>5.1f}% {r['oos_net']:>8.2f} {r['oos_dd']:>6.2f}% {dec_s}{vs_bl}")

    # Group analysis
    print(f"\n  --- Category Summary ---")
    categories = {
        "Brick-based":     ["down_1", "down_2", "down_3", "trail_1of3", "trail_2of3"],
        "Indicator-only":  ["psar_flip", "st_flip", "ema_cross", "macd_neg", "rsi_below_60", "rsi_below_50"],
        "Conditional":     ["down_1+psar", "down_1+st", "down_1+macd", "down_1+ema"],
        "Time-limited":    ["max_10", "max_20", "max_30"],
    }
    for cat_name, exits in categories.items():
        cat_results = [r for r in all_results if r["exit"] in exits and r["oos_trades"] > 0]
        if cat_results:
            finite = [r for r in cat_results if not math.isinf(r["oos_pf"])]
            avg_pf = np.mean([r["oos_pf"] for r in finite]) if finite else 0
            avg_t = np.mean([r["oos_trades"] for r in cat_results])
            best = max(cat_results, key=lambda r: r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6)
            best_s = f"{best['oos_pf']:.2f}" if not math.isinf(best["oos_pf"]) else "inf"
            print(f"    {cat_name:<18} avg PF={avg_pf:>7.2f}, avg T={avg_t:>5.1f} | "
                  f"best={best['exit']} PF={best_s} T={best['oos_trades']}")


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-parallel", action="store_true")
    args = parser.parse_args()

    out_path = ROOT / "ai_context" / "btc_phase6_exit_results.json"
    out_path.parent.mkdir(exist_ok=True)

    print("BTC Phase 6: Exit Strategy Optimization (Long Only)")
    print(f"  Entry     : R007 + FLIP_ST + BB_break (n={N_BRICKS}, cd={COOLDOWN})")
    print(f"  Gates     : psar_dir + ADX>={ADX_THRESH} + vol<={VOL_MAX} + HTF ADX>={HTF_ADX_THRESH}")
    print(f"  Exits     : {len(EXIT_STRATEGIES)} strategies")
    print(f"  IS period : {IS_START} -> {IS_END}")
    print(f"  OOS period: {OOS_START} -> {OOS_END}")
    print()

    # Split into 3 groups for parallel execution
    groups = [
        (EXIT_STRATEGIES[:6], "brick+trail"),
        (EXIT_STRATEGIES[6:12], "indicator"),
        (EXIT_STRATEGIES[12:], "conditional+time"),
    ]

    all_results = []

    if args.no_parallel:
        for exits, label in groups:
            all_results.extend(_sweep_exit_group(exits, label))
    else:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(_sweep_exit_group, exits, label): label
                for exits, label in groups
            }
            for future in as_completed(futures):
                label = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    print(f"  [{label}] finished -- {len(results)} records")
                except Exception as exc:
                    import traceback
                    print(f"  [{label}] FAILED: {exc}")
                    traceback.print_exc()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved {len(all_results)} results -> {out_path}")

    _summarize(all_results)


if __name__ == "__main__":
    main()
