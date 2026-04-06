#!/usr/bin/env python3
"""
btc_luxalgo_sweep.py -- BTC LuxAlgo Indicator Sweep (Long Only)

Tests 6 ported LuxAlgo indicators as entry signals and gates on BTC $150 Renko.

  Part A — 10 individual LuxAlgo entry signals (×3 cooldowns = 30 combos)
  Part B — LuxAlgo gates on BTC007 quartet (5 gates × {alone, +PSAR} × 2 cd = 20 combos)
  Part C — Best LuxAlgo + BTC007 signal combos (~24 combos)
  Part D — KNN Supertrend as primary system (4 entries × 3 cd = 12 combos)

Baseline: BTC007 optimized (MACD+KAMA+stoch+ST + PSAR + chop60, cd=2)
  Python OOS: PF=29.49, 102t (0.6/d), WR=69.6% (htf30)
  TV OOS:     PF=22.87, 194t (1.1/d), WR=65.5% (no htf)

Usage:
    python renko/btc_luxalgo_sweep.py
"""

import contextlib
import io
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from renko.config import MAX_WORKERS

# -- Config --------------------------------------------------------------------

LTF_FILE   = "OANDA_BTCUSD.SPOT.US, 1S renko 150.csv"
IS_START   = "2024-06-04"
IS_END     = "2025-09-30"
OOS_START  = "2025-10-01"
OOS_END    = "2026-03-19"
COMMISSION = 0.0046
CAPITAL    = 1000.0
QTY_VALUE  = 20
OOS_DAYS   = 170


# -- Data loading --------------------------------------------------------------

def _load_ltf():
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    from renko.phase6_enrichment import add_phase6_indicators
    from renko.luxalgo_indicators import add_luxalgo_indicators
    df = load_renko_export(LTF_FILE)
    add_renko_indicators(df)
    add_phase6_indicators(df)
    add_luxalgo_indicators(df, include_knn=True, svm_vol_weight=0.0)
    return df


def _run_bt(df, entry, exit_, start, end):
    from engine import BacktestConfig, run_backtest
    df2 = df.copy()
    df2["long_entry"] = entry
    df2["long_exit"] = exit_
    cfg = BacktestConfig(
        initial_capital=CAPITAL, commission_pct=COMMISSION, slippage_ticks=0,
        qty_type="cash", qty_value=QTY_VALUE, pyramiding=1,
        start_date=start, end_date=end,
        take_profit_pct=0.0, stop_loss_pct=0.0,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest(df2, cfg)
    pf = kpis.get("profit_factor", 0.0) or 0.0
    return {
        "pf":     float("inf") if math.isinf(pf) else float(pf),
        "net":    float(kpis.get("net_profit", 0.0) or 0.0),
        "trades": int(kpis.get("total_trades", 0) or 0),
        "wr":     float(kpis.get("win_rate", 0.0) or 0.0),
        "dd":     float(kpis.get("max_drawdown_pct", 0.0) or 0.0),
    }


# ==============================================================================
# Part A — 10 Individual LuxAlgo Entry Signals
# ==============================================================================

def _gen_luxalgo_single(df, gate, signal_name, cooldown):
    """Single LuxAlgo indicator entry signal."""
    n = len(df)
    brick_up = df["brick_up"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    # Pre-extract arrays based on signal
    ink = df["lux_inertial_k"].values
    ind = df["lux_inertial_d"].values
    rs_trend = df["lux_rollseg_trend"].values
    rs_bull = df["lux_rollseg_bull_rev"].values
    bp = df["lux_breakout_bull"].values
    svm_bt = df["lux_svm_break_type"].values
    svm_bb = df["lux_svm_break_bull"].values
    knn = df["lux_knn_bullish"].values.astype(float)
    streak_b = df["lux_streak_bull"].values.astype(float)
    streak_rp = df["lux_streak_rev"].values

    for i in range(60, n):
        up = bool(brick_up[i])

        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue

        if not gate[i] or not up or (i - last_bar) < cooldown:
            continue

        fired = False

        if signal_name == "inertial_cross_20":
            if not np.isnan(ink[i]) and not np.isnan(ink[i-1]):
                if ink[i] > 20 and ink[i-1] <= 20:
                    fired = True

        elif signal_name == "inertial_cross_30":
            if not np.isnan(ink[i]) and not np.isnan(ink[i-1]):
                if ink[i] > 30 and ink[i-1] <= 30:
                    fired = True

        elif signal_name == "inertial_kd_cross":
            if (not np.isnan(ink[i]) and not np.isnan(ind[i])
                    and not np.isnan(ink[i-1]) and not np.isnan(ind[i-1])):
                if ink[i] > ind[i] and ink[i-1] <= ind[i-1]:
                    fired = True

        elif signal_name == "rollseg_bull_rev":
            if not np.isnan(rs_bull[i]) and rs_bull[i]:
                fired = True

        elif signal_name == "rollseg_trend_flip":
            if not np.isnan(rs_trend[i]) and not np.isnan(rs_trend[i-1]):
                if rs_trend[i] > 0 and rs_trend[i-1] <= 0:
                    fired = True

        elif signal_name == "breakout_bull_40":
            if not np.isnan(bp[i]) and not np.isnan(bp[i-1]):
                if bp[i] > 40 and bp[i-1] <= 40:
                    fired = True

        elif signal_name == "breakout_bull_50":
            if not np.isnan(bp[i]) and not np.isnan(bp[i-1]):
                if bp[i] > 50 and bp[i-1] <= 50:
                    fired = True

        elif signal_name == "svm_bull_break":
            if not np.isnan(svm_bt[i]) and svm_bt[i] > 0:
                if not np.isnan(svm_bb[i]) and svm_bb[i]:
                    fired = True

        elif signal_name == "knn_bull_flip":
            if not np.isnan(knn[i]) and not np.isnan(knn[i-1]):
                if knn[i] > 0.5 and knn[i-1] <= 0.5:
                    fired = True

        elif signal_name == "streak_reversal":
            if (not np.isnan(streak_b[i]) and not np.isnan(streak_b[i-1])
                    and not np.isnan(streak_rp[i])):
                if streak_b[i] > 0.5 and streak_b[i-1] <= 0.5 and streak_rp[i] > 50:
                    fired = True

        if fired:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


# ==============================================================================
# Part B — LuxAlgo Gates on BTC007 Quartet
# ==============================================================================

def _gen_btc007_with_gate(df, gate, cooldown):
    """BTC007 quartet (MACD flip + KAMA turn + stoch cross + ST flip) with custom gate."""
    n = len(df)
    brick_up = df["brick_up"].values
    macd_h = df["macd_hist"].values
    kama_s = df["kama_slope"].values
    stoch_k = df["stoch_k"].values
    st_dir = df["st_dir"].values
    chop = df["chop"].values

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    for i in range(60, n):
        up = bool(brick_up[i])

        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue

        if not gate[i] or not up or (i - last_bar) < cooldown:
            continue

        # Chop gate (BTC007 best)
        if not np.isnan(chop[i]) and chop[i] > 60:
            continue

        fired = False

        # ST flip
        if not fired:
            if not np.isnan(st_dir[i]) and not np.isnan(st_dir[i-1]):
                if st_dir[i] > 0 and st_dir[i-1] <= 0:
                    fired = True

        # MACD flip
        if not fired:
            if not np.isnan(macd_h[i]) and not np.isnan(macd_h[i-1]):
                if macd_h[i] > 0 and macd_h[i-1] <= 0:
                    fired = True

        # KAMA turn
        if not fired:
            if not np.isnan(kama_s[i]) and not np.isnan(kama_s[i-1]):
                if kama_s[i] > 0 and kama_s[i-1] <= 0:
                    fired = True

        # Stoch cross (25)
        if not fired:
            if not np.isnan(stoch_k[i]) and not np.isnan(stoch_k[i-1]):
                if stoch_k[i] > 25 and stoch_k[i-1] <= 25:
                    fired = True

        if fired:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


# ==============================================================================
# Part C — Combined LuxAlgo + BTC007 Signals
# ==============================================================================

def _gen_combined_lux(df, gate, lux_signals, cooldown):
    """BTC007 quartet + selected LuxAlgo signals as additional entries."""
    n = len(df)
    brick_up = df["brick_up"].values
    macd_h = df["macd_hist"].values
    kama_s = df["kama_slope"].values
    stoch_k = df["stoch_k"].values
    st_dir = df["st_dir"].values
    chop = df["chop"].values

    # LuxAlgo arrays
    ink = df["lux_inertial_k"].values
    ind = df["lux_inertial_d"].values
    rs_trend = df["lux_rollseg_trend"].values
    rs_bull = df["lux_rollseg_bull_rev"].values
    svm_bt = df["lux_svm_break_type"].values
    svm_bb = df["lux_svm_break_bull"].values
    knn = df["lux_knn_bullish"].values.astype(float)

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    for i in range(60, n):
        up = bool(brick_up[i])

        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue

        if not gate[i] or not up or (i - last_bar) < cooldown:
            continue

        if not np.isnan(chop[i]) and chop[i] > 60:
            continue

        fired = False

        # BTC007 quartet
        if not fired:
            if not np.isnan(st_dir[i]) and not np.isnan(st_dir[i-1]):
                if st_dir[i] > 0 and st_dir[i-1] <= 0:
                    fired = True
        if not fired:
            if not np.isnan(macd_h[i]) and not np.isnan(macd_h[i-1]):
                if macd_h[i] > 0 and macd_h[i-1] <= 0:
                    fired = True
        if not fired:
            if not np.isnan(kama_s[i]) and not np.isnan(kama_s[i-1]):
                if kama_s[i] > 0 and kama_s[i-1] <= 0:
                    fired = True
        if not fired:
            if not np.isnan(stoch_k[i]) and not np.isnan(stoch_k[i-1]):
                if stoch_k[i] > 25 and stoch_k[i-1] <= 25:
                    fired = True

        # Additional LuxAlgo signals
        if not fired and "inertial_kd" in lux_signals:
            if (not np.isnan(ink[i]) and not np.isnan(ind[i])
                    and not np.isnan(ink[i-1]) and not np.isnan(ind[i-1])):
                if ink[i] > ind[i] and ink[i-1] <= ind[i-1]:
                    fired = True

        if not fired and "rollseg_rev" in lux_signals:
            if not np.isnan(rs_bull[i]) and rs_bull[i]:
                fired = True

        if not fired and "rollseg_flip" in lux_signals:
            if not np.isnan(rs_trend[i]) and not np.isnan(rs_trend[i-1]):
                if rs_trend[i] > 0 and rs_trend[i-1] <= 0:
                    fired = True

        if not fired and "svm_break" in lux_signals:
            if not np.isnan(svm_bt[i]) and svm_bt[i] > 0:
                if not np.isnan(svm_bb[i]) and svm_bb[i]:
                    fired = True

        if not fired and "knn_flip" in lux_signals:
            if not np.isnan(knn[i]) and not np.isnan(knn[i-1]):
                if knn[i] > 0.5 and knn[i-1] <= 0.5:
                    fired = True

        if fired:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


# ==============================================================================
# Part D — KNN Supertrend as Primary System
# ==============================================================================

def _gen_knn_system(df, gate, entry_type, cooldown):
    """KNN ml_bullish as trend filter + simple entry."""
    n = len(df)
    brick_up = df["brick_up"].values
    knn_bull = df["lux_knn_bullish"].values.astype(float)
    macd_h = df["macd_hist"].values
    stoch_k = df["stoch_k"].values

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    for i in range(60, n):
        up = bool(brick_up[i])

        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue

        if not gate[i] or not up or (i - last_bar) < cooldown:
            continue

        # KNN must be bullish
        if np.isnan(knn_bull[i]) or knn_bull[i] <= 0.5:
            continue

        fired = False

        if entry_type == "up_brick":
            fired = True  # Any up brick when KNN is bullish

        elif entry_type == "two_up":
            if brick_up[i-1]:
                fired = True

        elif entry_type == "macd_flip":
            if not np.isnan(macd_h[i]) and not np.isnan(macd_h[i-1]):
                if macd_h[i] > 0 and macd_h[i-1] <= 0:
                    fired = True

        elif entry_type == "stoch_cross":
            if not np.isnan(stoch_k[i]) and not np.isnan(stoch_k[i-1]):
                if stoch_k[i] > 25 and stoch_k[i-1] <= 25:
                    fired = True

        if fired:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


# ==============================================================================
# Combo Builders
# ==============================================================================

PART_A_SIGNALS = [
    "inertial_cross_20", "inertial_cross_30", "inertial_kd_cross",
    "rollseg_bull_rev", "rollseg_trend_flip",
    "breakout_bull_40", "breakout_bull_50",
    "svm_bull_break", "knn_bull_flip", "streak_reversal",
]

def _build_part_a():
    combos = []
    for sig in PART_A_SIGNALS:
        for cd in [3, 5, 7]:
            combos.append({
                "part": "A", "signal": sig, "cooldown": cd,
                "label": f"{sig}_cd{cd}",
            })
    return combos


def _build_part_b():
    combos = []
    gate_names = ["rollseg_bull", "knn_bull", "svm_bull", "inertial_bull", "breakout_squeeze"]
    for gname in gate_names:
        for use_psar in [False, True]:
            for cd in [2, 3]:
                psar_tag = "+psar" if use_psar else ""
                combos.append({
                    "part": "B", "gate": gname, "use_psar": use_psar, "cooldown": cd,
                    "label": f"q4_{gname}{psar_tag}_cd{cd}",
                })
    return combos


PART_C_SETS = {
    "q4+inertial":     ["inertial_kd"],
    "q4+rollseg":      ["rollseg_rev", "rollseg_flip"],
    "q4+svm":          ["svm_break"],
    "q4+knn":          ["knn_flip"],
    "q4+inertial+svm": ["inertial_kd", "svm_break"],
    "q4+rollseg+knn":  ["rollseg_flip", "knn_flip"],
    "q4+all_lux":      ["inertial_kd", "rollseg_flip", "svm_break", "knn_flip"],
    "q4+rs_svm":       ["rollseg_rev", "svm_break"],
}

def _build_part_c():
    combos = []
    for set_name, sigs in PART_C_SETS.items():
        for cd in [2, 3, 5]:
            combos.append({
                "part": "C", "set_name": set_name, "lux_signals": sigs,
                "cooldown": cd,
                "label": f"{set_name}_cd{cd}",
            })
    return combos


def _build_part_d():
    combos = []
    for etype in ["up_brick", "two_up", "macd_flip", "stoch_cross"]:
        for cd in [3, 5, 7]:
            combos.append({
                "part": "D", "entry_type": etype, "cooldown": cd,
                "label": f"knn_{etype}_cd{cd}",
            })
    return combos


# ==============================================================================
# Worker
# ==============================================================================

_w = {}

def _init_worker():
    if "df" not in _w:
        _w["df"] = _load_ltf()
        # PSAR gate
        psar = _w["df"]["psar_dir"].values
        _w["psar_gate"] = np.isnan(psar) | (psar > 0)
        # LuxAlgo gates
        _w["lux_gates"] = {}
        df = _w["df"]

        # Rolling segment trend bullish
        rs = df["lux_rollseg_trend"].values
        _w["lux_gates"]["rollseg_bull"] = np.isnan(rs) | (rs > 0)

        # KNN bullish
        knn = df["lux_knn_bullish"].values.astype(float)
        _w["lux_gates"]["knn_bull"] = np.isnan(knn) | (knn > 0.5)

        # SVM trend bullish
        svm = df["lux_svm_trend"].values
        _w["lux_gates"]["svm_bull"] = np.isnan(svm) | (svm > 0)

        # Inertial K > 50
        ink = df["lux_inertial_k"].values
        _w["lux_gates"]["inertial_bull"] = np.isnan(ink) | (ink > 50)

        # Breakout squeeze > 30
        sq = df["lux_breakout_squeeze"].values
        _w["lux_gates"]["breakout_squeeze"] = np.isnan(sq) | (sq > 30)


def _run_one(combo):
    _init_worker()
    df = _w["df"]

    if combo["part"] == "A":
        gate = _w["psar_gate"].copy()
        entry, exit_ = _gen_luxalgo_single(df, gate, combo["signal"], combo["cooldown"])

    elif combo["part"] == "B":
        gate = _w["lux_gates"][combo["gate"]].copy()
        if combo["use_psar"]:
            gate &= _w["psar_gate"]
        entry, exit_ = _gen_btc007_with_gate(df, gate, combo["cooldown"])

    elif combo["part"] == "C":
        gate = _w["psar_gate"].copy()
        entry, exit_ = _gen_combined_lux(df, gate, combo["lux_signals"], combo["cooldown"])

    elif combo["part"] == "D":
        gate = _w["psar_gate"].copy()
        entry, exit_ = _gen_knn_system(df, gate, combo["entry_type"], combo["cooldown"])

    is_r = _run_bt(df, entry, exit_, IS_START, IS_END)
    oos_r = _run_bt(df, entry, exit_, OOS_START, OOS_END)

    return combo, is_r, oos_r


# ==============================================================================
# Reporting
# ==============================================================================

def _print_header():
    print(f"  {'#':>3} {'Pt':>2} {'Label':<38} | "
          f"{'IS PF':>7} {'T':>5} {'WR%':>6} | "
          f"{'OOS PF':>8} {'T':>5} {'t/d':>5} {'WR%':>6} {'Net':>9} {'DD%':>7}")
    print("  " + "-" * 112)


def _print_row(r, rank):
    oos_td = r["oos_trades"] / OOS_DAYS
    is_pf = f"{r['is_pf']:.2f}" if r["is_pf"] < 9999 else "inf"
    oos_pf = f"{r['oos_pf']:.2f}" if r["oos_pf"] < 9999 else "inf"
    print(f"  {rank:>3} {r['part']:>2} {r['label']:<38} | "
          f"{is_pf:>7} {r['is_trades']:>5} {r['is_wr']:>5.1f}% | "
          f"{oos_pf:>8} {r['oos_trades']:>5} {oos_td:>5.1f} {r['oos_wr']:>5.1f}% "
          f"{r['oos_net']:>8.2f} {r['oos_dd']:>6.3f}%")


def _show_part(results, part, title, n=15):
    subset = [r for r in results if r["part"] == part]
    viable = [r for r in subset if r["oos_trades"] >= 10 and r["oos_net"] > 0]
    viable.sort(key=lambda r: (r["oos_wr"], r["oos_net"]), reverse=True)
    print(f"\n{'=' * 60}")
    print(f"  {title} ({len(viable)} viable / {len(subset)} total)")
    print(f"{'=' * 60}")
    if viable:
        _print_header()
        for i, r in enumerate(viable[:n]):
            _print_row(r, i + 1)
    else:
        print("  No viable configs.")


# ==============================================================================
# Main
# ==============================================================================

def main():
    combos = _build_part_a() + _build_part_b() + _build_part_c() + _build_part_d()
    total = len(combos)
    print(f"LuxAlgo Sweep: {total} combos ({total * 2} backtests) on {MAX_WORKERS} workers")

    all_results = []
    done = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run_one, c): c for c in combos}
        for fut in as_completed(futures):
            combo = futures[fut]
            try:
                combo_ret, is_r, oos_r = fut.result()
                row = {
                    "part":       combo_ret["part"],
                    "label":      combo_ret["label"],
                    "is_pf":      is_r["pf"],
                    "is_trades":  is_r["trades"],
                    "is_wr":      is_r["wr"],
                    "is_net":     is_r["net"],
                    "is_dd":      is_r["dd"],
                    "oos_pf":     oos_r["pf"],
                    "oos_trades": oos_r["trades"],
                    "oos_wr":     oos_r["wr"],
                    "oos_net":    oos_r["net"],
                    "oos_dd":     oos_r["dd"],
                }
                # Preserve extra keys
                for k in combo_ret:
                    if k not in row:
                        row[k] = combo_ret[k]
                all_results.append(row)
            except Exception as e:
                print(f"  ERROR: {combo.get('label', '?')}: {e}")

            done += 1
            if done % 20 == 0 or done == total:
                print(f"  [{done}/{total}]")

    # -- Show results per part --
    _show_part(all_results, "A", "Part A — Individual LuxAlgo Signals")
    _show_part(all_results, "B", "Part B — LuxAlgo Gates on BTC007 Quartet")
    _show_part(all_results, "C", "Part C — Combined LuxAlgo + BTC007 Signals")
    _show_part(all_results, "D", "Part D — KNN Supertrend System")

    # Global top 20
    viable = [r for r in all_results if r["oos_trades"] >= 10 and r["oos_net"] > 0]
    viable.sort(key=lambda r: (r["oos_wr"], r["oos_net"]), reverse=True)
    print(f"\n{'=' * 60}")
    print(f"  GLOBAL TOP 20 ({len(viable)} viable / {len(all_results)} total)")
    print(f"{'=' * 60}")
    if viable:
        _print_header()
        for i, r in enumerate(viable[:20]):
            _print_row(r, i + 1)

    # HF subset (1+ trade/day)
    hf = [r for r in viable if r["oos_trades"] >= OOS_DAYS]
    if hf:
        hf.sort(key=lambda r: (r["oos_wr"], r["oos_net"]), reverse=True)
        print(f"\n  HF SUBSET (1+/day, {len(hf)} configs)")
        _print_header()
        for i, r in enumerate(hf[:15]):
            _print_row(r, i + 1)

    # Save JSON
    out_path = ROOT / "ai_context" / "btc_luxalgo_sweep_results.json"
    serializable = []
    for r in all_results:
        sr = dict(r)
        for k in ("is_pf", "oos_pf"):
            if isinstance(sr[k], float) and math.isinf(sr[k]):
                sr[k] = "inf"
        serializable.append(sr)

    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved {len(all_results)} results -> {out_path}")


if __name__ == "__main__":
    main()
