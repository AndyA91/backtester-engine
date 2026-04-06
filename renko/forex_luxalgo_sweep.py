#!/usr/bin/env python3
"""
forex_luxalgo_sweep.py -- Forex LuxAlgo Indicator Sweep (Bidirectional)

Tests 6 ported LuxAlgo indicators as gates and entry signals across 3 live
forex instruments. Key advantage over BTC: forex has volume data, so SVM
Ranker's volume feature is active (svm_vol_weight=0.4).

  Part A — LuxAlgo as P6-style gates on R001+R002 entries (60 combos)
  Part B — LuxAlgo gates stacked with best P6 per instrument (72 combos)
  Part C — LuxAlgo standalone entry signals, bidirectional (60 combos)
  Part D — Best standalone + PSAR/ADX gates (81 combos)

Total: 273 combos (546 backtests)

Usage:
    python renko/forex_luxalgo_sweep.py
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

# -- Instrument configs --------------------------------------------------------

INSTRUMENTS = {
    "EURUSD": {
        "renko_file": "OANDA_EURUSD, 1S renko 0.0005.csv",
        "is_start":   "2022-05-18",
        "is_end":     "2025-09-30",
        "oos_start":  "2025-10-01",
        "oos_end":    "2026-03-19",
        "oos_days":   170,
        "commission":  0.0046,
        "capital":     1000.0,
        "include_mk":  False,
        "best_p6":    "stoch_cross",
    },
    "GBPJPY": {
        "renko_file": "OANDA_GBPJPY, 1S renko 0.05.csv",
        "is_start":   "2024-11-21",
        "is_end":     "2025-09-30",
        "oos_start":  "2025-10-01",
        "oos_end":    "2026-02-28",
        "oos_days":   151,
        "commission":  0.005,
        "capital":     150_000.0,
        "include_mk":  True,
        "best_p6":    "mk_regime",
    },
    "EURAUD": {
        "renko_file": "OANDA_EURAUD, 1S renko 0.0006.csv",
        "is_start":   "2023-07-20",
        "is_end":     "2025-09-30",
        "oos_start":  "2025-10-01",
        "oos_end":    "2026-03-17",
        "oos_days":   168,
        "commission":  0.009,
        "capital":     1000.0,
        "include_mk":  False,
        "best_p6":    "escgo_cross",
    },
}

LUX_GATE_NAMES = [
    "rollseg_trend", "inertial_cross", "svm_trend", "knn_trend", "breakout_bias",
]

STANDALONE_SIGNALS = [
    "rollseg_reversal", "rollseg_trend_flip",
    "svm_break", "svm_high_score",
    "knn_flip", "inertial_cross_20",
    "inertial_kd_cross", "breakout_prob_40",
    "breakout_prob_50", "streak_reversal",
]

PART_D_SIGNALS = ["rollseg_reversal", "svm_break", "knn_flip"]
PART_D_GATES = ["none", "psar", "adx25"]


# ==============================================================================
# Signal Generators
# ==============================================================================

def _gen_r001r002(df, n_bricks, cooldown, gate_long, gate_short):
    """R001+R002 entries with directional gates (bidirectional)."""
    n = len(df)
    brick_up = df["brick_up"].values

    long_entry = np.zeros(n, dtype=bool)
    long_exit = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit = np.zeros(n, dtype=bool)

    in_position = False
    trade_dir = 0
    last_r001_bar = -999_999
    warmup = max(n_bricks + 1, 200)

    for i in range(warmup, n):
        up = bool(brick_up[i])

        if in_position:
            is_opp = (trade_dir == 1 and not up) or (trade_dir == -1 and up)
            if is_opp:
                if trade_dir == 1:
                    long_exit[i] = True
                else:
                    short_exit[i] = True
                in_position = False
                trade_dir = 0

        if in_position:
            continue

        # R002: N same-dir + 1 opposing → contrarian entry
        prev = brick_up[i - n_bricks : i]
        prev_all_up = bool(np.all(prev))
        prev_all_down = bool(not np.any(prev))

        if prev_all_up and not up:
            cand = -1; is_r002 = True
        elif prev_all_down and up:
            cand = 1; is_r002 = True
        else:
            # R001: N consecutive including current → momentum entry
            if (i - last_r001_bar) < cooldown:
                continue
            window = brick_up[i - n_bricks + 1 : i + 1]
            all_up = bool(np.all(window))
            all_down = bool(not np.any(window))
            if all_up:
                cand = 1; is_r002 = False
            elif all_down:
                cand = -1; is_r002 = False
            else:
                continue

        # Apply directional gate
        if cand == 1 and not gate_long[i]:
            continue
        if cand == -1 and not gate_short[i]:
            continue

        if cand == 1:
            long_entry[i] = True
        else:
            short_entry[i] = True
        in_position = True
        trade_dir = cand
        if not is_r002:
            last_r001_bar = i

    return long_entry, long_exit, short_entry, short_exit


def _gen_standalone(df, signal_name, cooldown, gate_long, gate_short):
    """Standalone LuxAlgo signal — bidirectional entries with brick confirmation."""
    n = len(df)
    brick_up = df["brick_up"].values

    long_entry = np.zeros(n, dtype=bool)
    long_exit = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit = np.zeros(n, dtype=bool)

    in_position = False
    trade_dir = 0
    last_bar = -999_999

    # Pre-extract arrays
    ink = df["lux_inertial_k"].values
    ind = df["lux_inertial_d"].values
    rs_trend = df["lux_rollseg_trend"].values
    rs_bull = df["lux_rollseg_bull_rev"].values.astype(float)
    rs_bear = df["lux_rollseg_bear_rev"].values.astype(float)
    bp = df["lux_breakout_bull"].values
    bbr = df["lux_breakout_bear"].values
    svm_bt = df["lux_svm_break_type"].values
    svm_bb = df["lux_svm_break_bull"].values.astype(float)
    svm_score = df["lux_svm_score"].values
    knn = df["lux_knn_bullish"].values.astype(float)
    streak_b = df["lux_streak_bull"].values.astype(float)
    streak_rp = df["lux_streak_rev"].values

    for i in range(60, n):
        up = bool(brick_up[i])

        # Exit on opposing brick
        if in_position:
            is_opp = (trade_dir == 1 and not up) or (trade_dir == -1 and up)
            if is_opp:
                if trade_dir == 1:
                    long_exit[i] = True
                else:
                    short_exit[i] = True
                in_position = False
                trade_dir = 0
            continue

        if (i - last_bar) < cooldown:
            continue

        long_fire = False
        short_fire = False

        if signal_name == "rollseg_reversal":
            if not np.isnan(rs_bull[i]) and rs_bull[i] > 0.5:
                long_fire = True
            elif not np.isnan(rs_bear[i]) and rs_bear[i] > 0.5:
                short_fire = True

        elif signal_name == "rollseg_trend_flip":
            if not np.isnan(rs_trend[i]) and not np.isnan(rs_trend[i-1]):
                if rs_trend[i] > 0 and rs_trend[i-1] <= 0:
                    long_fire = True
                elif rs_trend[i] < 0 and rs_trend[i-1] >= 0:
                    short_fire = True

        elif signal_name == "svm_break":
            if not np.isnan(svm_bt[i]) and svm_bt[i] > 0:
                if not np.isnan(svm_bb[i]):
                    if svm_bb[i] > 0.5:
                        long_fire = True
                    else:
                        short_fire = True

        elif signal_name == "svm_high_score":
            if not np.isnan(svm_bt[i]) and svm_bt[i] > 0:
                if not np.isnan(svm_score[i]) and svm_score[i] > 60:
                    if not np.isnan(svm_bb[i]):
                        if svm_bb[i] > 0.5:
                            long_fire = True
                        else:
                            short_fire = True

        elif signal_name == "knn_flip":
            if not np.isnan(knn[i]) and not np.isnan(knn[i-1]):
                if knn[i] > 0.5 and knn[i-1] <= 0.5:
                    long_fire = True
                elif knn[i] <= 0.5 and knn[i-1] > 0.5:
                    short_fire = True

        elif signal_name == "inertial_cross_20":
            if not np.isnan(ink[i]) and not np.isnan(ink[i-1]):
                if ink[i] > 20 and ink[i-1] <= 20:
                    long_fire = True
                elif ink[i] < 80 and ink[i-1] >= 80:
                    short_fire = True

        elif signal_name == "inertial_kd_cross":
            if (not np.isnan(ink[i]) and not np.isnan(ind[i])
                    and not np.isnan(ink[i-1]) and not np.isnan(ind[i-1])):
                if ink[i] > ind[i] and ink[i-1] <= ind[i-1]:
                    long_fire = True
                elif ink[i] < ind[i] and ink[i-1] >= ind[i-1]:
                    short_fire = True

        elif signal_name == "breakout_prob_40":
            if not np.isnan(bp[i]) and not np.isnan(bp[i-1]):
                if bp[i] > 40 and bp[i-1] <= 40:
                    long_fire = True
            if not long_fire and not np.isnan(bbr[i]) and not np.isnan(bbr[i-1]):
                if bbr[i] > 40 and bbr[i-1] <= 40:
                    short_fire = True

        elif signal_name == "breakout_prob_50":
            if not np.isnan(bp[i]) and not np.isnan(bp[i-1]):
                if bp[i] > 50 and bp[i-1] <= 50:
                    long_fire = True
            if not long_fire and not np.isnan(bbr[i]) and not np.isnan(bbr[i-1]):
                if bbr[i] > 50 and bbr[i-1] <= 50:
                    short_fire = True

        elif signal_name == "streak_reversal":
            if (not np.isnan(streak_b[i]) and not np.isnan(streak_b[i-1])
                    and not np.isnan(streak_rp[i])):
                # Bearish streak ended (flipped to bullish) + high reversal prob → long
                if streak_b[i] > 0.5 and streak_b[i-1] <= 0.5 and streak_rp[i] > 50:
                    long_fire = True
                # Bullish streak ended (flipped to bearish) + high reversal prob → short
                elif streak_b[i] <= 0.5 and streak_b[i-1] > 0.5 and streak_rp[i] > 50:
                    short_fire = True

        # Apply gate + brick direction confirmation
        if long_fire and up and gate_long[i]:
            long_entry[i] = True
            in_position = True
            trade_dir = 1
            last_bar = i
        elif short_fire and not up and gate_short[i]:
            short_entry[i] = True
            in_position = True
            trade_dir = -1
            last_bar = i

    return long_entry, long_exit, short_entry, short_exit


# ==============================================================================
# Backtest Runner
# ==============================================================================

def _run_bt(df, le, lx, se, sx, start, end, commission, capital):
    from engine import BacktestConfig, run_backtest_long_short
    df2 = df.copy()
    df2["long_entry"] = le
    df2["long_exit"] = lx
    df2["short_entry"] = se
    df2["short_exit"] = sx
    cfg = BacktestConfig(
        initial_capital=capital, commission_pct=commission, slippage_ticks=0,
        qty_type="fixed", qty_value=1000.0, pyramiding=1,
        start_date=start, end_date=end,
        take_profit_pct=0.0, stop_loss_pct=0.0,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest_long_short(df2, cfg)
    pf = kpis.get("profit_factor", 0.0) or 0.0
    return {
        "pf":     float("inf") if math.isinf(pf) else float(pf),
        "net":    float(kpis.get("net_profit", 0.0) or 0.0),
        "trades": int(kpis.get("total_trades", 0) or 0),
        "wr":     float(kpis.get("win_rate", 0.0) or 0.0),
        "dd":     float(kpis.get("max_drawdown_pct", 0.0) or 0.0),
    }


# ==============================================================================
# Combo Builders
# ==============================================================================

def _build_part_a():
    """Part A: LuxAlgo gates on R001+R002 entries (5 gates × 2n × 2cd × 3 inst = 60)."""
    combos = []
    for inst in INSTRUMENTS:
        for gate in LUX_GATE_NAMES:
            for n in [3, 4]:
                for cd in [10, 20]:
                    combos.append({
                        "part": "A", "inst": inst, "gate": gate,
                        "n_bricks": n, "cooldown": cd,
                        "label": f"{inst}_{gate}_n{n}_cd{cd}",
                    })
    return combos


def _build_part_b():
    """Part B: Best P6 + LuxAlgo stacked (6 variants × 2n × 2cd × 3 inst = 72)."""
    combos = []
    for inst in INSTRUMENTS:
        for lux_gate in [None] + LUX_GATE_NAMES:
            for n in [3, 4]:
                for cd in [10, 20]:
                    lux_tag = lux_gate if lux_gate else "p6_only"
                    combos.append({
                        "part": "B", "inst": inst, "lux_gate": lux_gate,
                        "n_bricks": n, "cooldown": cd,
                        "label": f"{inst}_p6+{lux_tag}_n{n}_cd{cd}",
                    })
    return combos


def _build_part_c():
    """Part C: Standalone LuxAlgo signals (10 signals × 2cd × 3 inst = 60)."""
    combos = []
    for inst in INSTRUMENTS:
        for sig in STANDALONE_SIGNALS:
            for cd in [10, 20]:
                combos.append({
                    "part": "C", "inst": inst, "signal": sig,
                    "cooldown": cd,
                    "label": f"{inst}_{sig}_cd{cd}",
                })
    return combos


def _build_part_d():
    """Part D: Top standalone + PSAR/ADX gates (3 sig × 3 gates × 3cd × 3 inst = 81)."""
    combos = []
    for inst in INSTRUMENTS:
        for sig in PART_D_SIGNALS:
            for gate in PART_D_GATES:
                for cd in [10, 20, 30]:
                    combos.append({
                        "part": "D", "inst": inst, "signal": sig,
                        "gate": gate, "cooldown": cd,
                        "label": f"{inst}_{sig}_{gate}_cd{cd}",
                    })
    return combos


# ==============================================================================
# Worker (per-process caching)
# ==============================================================================

_w = {}


def _init_inst(inst_key):
    """Load and cache instrument data + pre-computed gates."""
    if inst_key in _w:
        return
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    from renko.phase6_enrichment import add_phase6_indicators
    from renko.luxalgo_indicators import add_luxalgo_indicators
    from renko.phase6_sweep import _compute_gate_arrays as _p6_gate

    cfg = INSTRUMENTS[inst_key]
    df = load_renko_export(cfg["renko_file"])
    add_renko_indicators(df)
    add_phase6_indicators(df, include_mk=cfg["include_mk"])
    add_luxalgo_indicators(df, include_knn=True, svm_vol_weight=0.4)

    n = len(df)

    # Best P6 gate per instrument
    p6_l, p6_s = _p6_gate(df, cfg["best_p6"])

    # PSAR directional gate
    psar = df["psar_dir"].values
    psar_nan = np.isnan(psar)
    psar_l = psar_nan | (psar > 0)
    psar_s = psar_nan | (psar < 0)

    # ADX >= 25 (symmetric)
    adx = df["adx"].values
    adx_nan = np.isnan(adx)
    adx25 = adx_nan | (adx >= 25)

    # LuxAlgo directional gates
    lux_g = {}

    rs = df["lux_rollseg_trend"].values
    rs_nan = np.isnan(rs)
    lux_g["rollseg_trend"] = (rs_nan | (rs > 0), rs_nan | (rs < 0))

    ink = df["lux_inertial_k"].values
    ind = df["lux_inertial_d"].values
    in_nan = np.isnan(ink) | np.isnan(ind)
    lux_g["inertial_cross"] = (in_nan | (ink > ind), in_nan | (ink < ind))

    svm = df["lux_svm_trend"].values
    svm_nan = np.isnan(svm)
    lux_g["svm_trend"] = (svm_nan | (svm > 0), svm_nan | (svm < 0))

    knn = df["lux_knn_bullish"].values.astype(float)
    knn_nan = np.isnan(knn)
    lux_g["knn_trend"] = (knn_nan | (knn > 0.5), knn_nan | (knn <= 0.5))

    bpb = df["lux_breakout_bull"].values
    bpr = df["lux_breakout_bear"].values
    bb_nan = np.isnan(bpb) | np.isnan(bpr)
    lux_g["breakout_bias"] = (bb_nan | (bpb > bpr), bb_nan | (bpr > bpb))

    _w[inst_key] = {
        "df": df,
        "p6": (p6_l, p6_s),
        "psar": (psar_l, psar_s),
        "adx25": (adx25, adx25),
        "lux_gates": lux_g,
    }


def _run_one(combo):
    inst = combo["inst"]
    _init_inst(inst)
    w = _w[inst]
    df = w["df"]
    cfg = INSTRUMENTS[inst]
    n = len(df)
    ones = np.ones(n, dtype=bool)

    if combo["part"] == "A":
        gl, gs = w["lux_gates"][combo["gate"]]
        le, lx, se, sx = _gen_r001r002(
            df, combo["n_bricks"], combo["cooldown"], gl, gs)

    elif combo["part"] == "B":
        p6_l, p6_s = w["p6"]
        if combo["lux_gate"] is not None:
            lux_l, lux_s = w["lux_gates"][combo["lux_gate"]]
            gl = p6_l & lux_l
            gs = p6_s & lux_s
        else:
            gl, gs = p6_l.copy(), p6_s.copy()
        le, lx, se, sx = _gen_r001r002(
            df, combo["n_bricks"], combo["cooldown"], gl, gs)

    elif combo["part"] == "C":
        le, lx, se, sx = _gen_standalone(
            df, combo["signal"], combo["cooldown"], ones, ones)

    elif combo["part"] == "D":
        if combo["gate"] == "psar":
            gl, gs = w["psar"]
        elif combo["gate"] == "adx25":
            gl, gs = w["adx25"]
        else:
            gl, gs = ones, ones
        le, lx, se, sx = _gen_standalone(
            df, combo["signal"], combo["cooldown"], gl, gs)

    is_r = _run_bt(df, le, lx, se, sx,
                   cfg["is_start"], cfg["is_end"], cfg["commission"], cfg["capital"])
    oos_r = _run_bt(df, le, lx, se, sx,
                    cfg["oos_start"], cfg["oos_end"], cfg["commission"], cfg["capital"])

    return combo, is_r, oos_r


# ==============================================================================
# Reporting
# ==============================================================================

def _print_header():
    print(f"  {'#':>3} {'Pt':>2} {'Label':<46} | "
          f"{'IS PF':>7} {'T':>5} {'WR%':>6} | "
          f"{'OOS PF':>8} {'T':>5} {'t/d':>5} {'WR%':>6} {'Net':>9} {'DD%':>7}")
    print("  " + "-" * 120)


def _print_row(r, rank):
    oos_td = r["oos_trades"] / r["oos_days"] if r["oos_days"] > 0 else 0
    is_pf = f"{r['is_pf']:.2f}" if r["is_pf"] < 9999 else "inf"
    oos_pf = f"{r['oos_pf']:.2f}" if r["oos_pf"] < 9999 else "inf"
    print(f"  {rank:>3} {r['part']:>2} {r['label']:<46} | "
          f"{is_pf:>7} {r['is_trades']:>5} {r['is_wr']:>5.1f}% | "
          f"{oos_pf:>8} {r['oos_trades']:>5} {oos_td:>5.1f} {r['oos_wr']:>5.1f}% "
          f"{r['oos_net']:>8.2f} {r['oos_dd']:>6.3f}%")


def _show_part(results, part, title, n=15):
    subset = [r for r in results if r["part"] == part]
    viable = [r for r in subset if r["oos_trades"] >= 5 and r["oos_net"] > 0]
    viable.sort(key=lambda r: (r["oos_wr"], r["oos_net"]), reverse=True)
    print(f"\n{'=' * 65}")
    print(f"  {title} ({len(viable)} viable / {len(subset)} total)")
    print(f"{'=' * 65}")
    if viable:
        _print_header()
        for i, r in enumerate(viable[:n]):
            _print_row(r, i + 1)
    else:
        print("  No viable configs.")


def _show_per_instrument(results, part, title, n=10):
    """Show top results per instrument for a given part."""
    subset = [r for r in results if r["part"] == part]
    for inst in INSTRUMENTS:
        inst_res = [r for r in subset if r["inst"] == inst
                    and r["oos_trades"] >= 5 and r["oos_net"] > 0]
        inst_res.sort(key=lambda r: (r["oos_wr"], r["oos_net"]), reverse=True)
        live_pf = {"EURUSD": 22.91, "GBPJPY": "inf", "EURAUD": 14.21,
                    "USDJPY": 35.36, "GBPUSD": 54.31}.get(inst, "??")
        print(f"\n  {inst} (live PF: {live_pf}) — {len(inst_res)} viable")
        if inst_res:
            _print_header()
            for i, r in enumerate(inst_res[:n]):
                _print_row(r, i + 1)


# ==============================================================================
# Main
# ==============================================================================

def main():
    combos = _build_part_a() + _build_part_b() + _build_part_c() + _build_part_d()
    total = len(combos)
    print(f"Forex LuxAlgo Sweep: {total} combos ({total * 2} backtests) on {MAX_WORKERS} workers")

    all_results = []
    done = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run_one, c): c for c in combos}
        for fut in as_completed(futures):
            combo = futures[fut]
            try:
                combo_ret, is_r, oos_r = fut.result()
                inst = combo_ret["inst"]
                row = {
                    "part":       combo_ret["part"],
                    "inst":       inst,
                    "label":      combo_ret["label"],
                    "oos_days":   INSTRUMENTS[inst]["oos_days"],
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
                for k in combo_ret:
                    if k not in row:
                        row[k] = combo_ret[k]
                all_results.append(row)
            except Exception as e:
                print(f"  ERROR: {combo.get('label', '?')}: {e}")

            done += 1
            if done % 25 == 0 or done == total:
                print(f"  [{done}/{total}]")

    # -- Per-part results --
    _show_part(all_results, "A", "Part A — LuxAlgo Gates on R001+R002")
    _show_per_instrument(all_results, "A", "Part A — Per Instrument")

    _show_part(all_results, "B", "Part B — P6 + LuxAlgo Stacked")
    _show_per_instrument(all_results, "B", "Part B — Per Instrument")

    _show_part(all_results, "C", "Part C — Standalone LuxAlgo Signals")
    _show_per_instrument(all_results, "C", "Part C — Per Instrument")

    _show_part(all_results, "D", "Part D — Standalone + PSAR/ADX Gates")
    _show_per_instrument(all_results, "D", "Part D — Per Instrument")

    # -- Global top 20 --
    viable = [r for r in all_results if r["oos_trades"] >= 5 and r["oos_net"] > 0]
    viable.sort(key=lambda r: (r["oos_wr"], r["oos_net"]), reverse=True)
    print(f"\n{'=' * 65}")
    print(f"  GLOBAL TOP 20 ({len(viable)} viable / {len(all_results)} total)")
    print(f"{'=' * 65}")
    if viable:
        _print_header()
        for i, r in enumerate(viable[:20]):
            _print_row(r, i + 1)

    # -- Save JSON --
    out_path = ROOT / "ai_context" / "forex_luxalgo_sweep_results.json"
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
