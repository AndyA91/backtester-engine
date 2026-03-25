#!/usr/bin/env python3
"""
mym_novel_signals_sweep.py — Novel Entry Signal Research for MYM Renko

Tests 4 fundamentally different entry signals that exploit Renko-specific data
(brick formation timing, velocity patterns, acceleration) to beat MYM001
(PF=232, 161t, 88.8% WR on TradingView, brick 14).

Signal types:
  BRE — Brick Burst Entry:     K fast consecutive same-direction bricks
  VIE — Velocity Impulse Entry: K bricks with monotonically decreasing formation time
  ACE — Acceleration Entry:     streak with accel < 1 (second half faster than first)
  MCE — Momentum Cascade:       N-of-M bricks same direction (relaxed R001)
  R007 — Baseline R001+R002:    standard MYM001 signal for comparison

All signals share:
  - Exit: first opposing brick
  - Forced close 15:45 ET, no entries after 15:30 ET
  - Commission 0.00475%, qty=0.50 (MYM point value)
  - IS/OOS split matching MYM001

Usage:
  python renko/mym_novel_signals_sweep.py
"""

import argparse
import contextlib
import io
import itertools
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "renko"))

from mym_sweep import _compute_et_hours, _generate_signal_arrays, _run_backtest
from mym_sweep_v4 import _load_renko_all_indicators_v4

# ── Constants ─────────────────────────────────────────────────────────────────

MYM_COMMISSION_PCT = 0.00475
MYM_CAPITAL = 1000.0
MYM_QTY = 0.50

INSTRUMENT = {
    "renko_file": "CBOT_MINI_MYM1!, 1S renko 14.csv",
    "is_start":   "2025-03-07",
    "is_end":     "2025-12-31",
    "oos_start":  "2026-01-01",
    "oos_end":    "2026-03-19",
    "label":      "MYM brick 14",
}

OUTPUT_FILE = ROOT / "ai_context" / "mym_novel_signals_results.json"
MIN_TRADES = 10

# ── Sweep grids ───────────────────────────────────────────────────────────────

ADX_THRESHOLDS = [0, 40, 50]     # 0 = no ADX gate
PSAR_OPTIONS   = [True, False]
COOLDOWNS      = [20, 30, 40, 50]

# BRE params
BRE_K          = [3, 4, 5, 6, 7]
BRE_VEL_THRESH = [0.4, 0.6, 0.8, 1.0]

# VIE params
VIE_K          = [3, 4, 5, 6]

# ACE params
ACE_K          = [3, 4, 5, 6, 7]

# MCE params (N, M)
MCE_NM         = [(6, 8), (7, 9), (8, 10), (7, 10), (8, 11)]

# Baseline R001+R002 (MYM001 reference)
BASELINE_N     = [9]


# ── Gate pre-computation ──────────────────────────────────────────────────────

def _precompute_gates(df):
    """Pre-compute all ADX × PSAR gate combinations."""
    n = len(df)
    gates = {}

    adx = df["adx"].values
    adx_nan = np.isnan(adx)
    gates["adx_0"] = np.ones(n, dtype=bool)  # no ADX filter
    for at in [40, 50]:
        gates[f"adx_{at}"] = adx_nan | (adx >= at)

    psar = df["psar_dir"].values
    psar_nan = np.isnan(psar)
    gates["psar_long"]  = psar_nan | (psar > 0)
    gates["psar_short"] = psar_nan | (psar < 0)

    return gates


def _combine_gates(gates, adx_thresh, use_psar):
    """Combine ADX + optional PSAR into (gate_long, gate_short) arrays."""
    adx_ok = gates[f"adx_{adx_thresh}"]
    if use_psar:
        gl = adx_ok & gates["psar_long"]
        gs = adx_ok & gates["psar_short"]
    else:
        gl = adx_ok.copy()
        gs = adx_ok.copy()
    return gl, gs


# ── Signal generators ─────────────────────────────────────────────────────────

def _generate_signals_bre(brick_up, et_hours, et_minutes,
                          gate_long_ok, gate_short_ok,
                          vel_ratio, K, vel_threshold, cooldown):
    """Brick Burst Entry: K consecutive same-direction FAST bricks."""
    n = len(brick_up)
    le = np.zeros(n, dtype=bool)
    lx = np.zeros(n, dtype=bool)
    se = np.zeros(n, dtype=bool)
    sx = np.zeros(n, dtype=bool)

    in_pos = False
    t_dir  = 0
    last_bar = -999_999
    warmup = max(K + 1, 200)

    for i in range(warmup, n):
        up   = bool(brick_up[i])
        h_et = et_hours[i]
        m_et = et_minutes[i]

        # Forced close 15:45 ET
        if in_pos and (h_et > 15 or (h_et == 15 and m_et >= 45)):
            lx[i] = (t_dir == 1)
            sx[i] = (t_dir == -1)
            in_pos = False; t_dir = 0
            continue

        # Normal exit: opposing brick
        if in_pos:
            is_opp = (t_dir == 1 and not up) or (t_dir == -1 and up)
            lx[i] = is_opp and t_dir == 1
            sx[i] = is_opp and t_dir == -1
            if is_opp:
                in_pos = False; t_dir = 0

        if in_pos:
            continue

        # No entry after 15:30 ET
        if h_et > 15 or (h_et == 15 and m_et >= 30):
            continue

        # Cooldown
        if (i - last_bar) < cooldown:
            continue

        # Check K consecutive same-direction bricks
        window = brick_up[i - K + 1 : i + 1]
        all_up   = bool(np.all(window))
        all_down = bool(not np.any(window))
        if not (all_up or all_down):
            continue

        # Check ALL K bricks have vel_ratio < threshold
        vel_win = vel_ratio[i - K + 1 : i + 1]
        if np.any(np.isnan(vel_win)):
            continue
        if not np.all(vel_win < vel_threshold):
            continue

        cand = 1 if all_up else -1

        # Gate check
        if cand == 1 and not gate_long_ok[i]:
            continue
        if cand == -1 and not gate_short_ok[i]:
            continue

        if cand == 1:
            le[i] = True
        else:
            se[i] = True
        in_pos = True; t_dir = cand; last_bar = i

    return le, lx, se, sx


def _generate_signals_vie(brick_up, et_hours, et_minutes,
                          gate_long_ok, gate_short_ok,
                          brick_td, K, cooldown):
    """Velocity Impulse Entry: K bricks with monotonically decreasing formation time."""
    n = len(brick_up)
    le = np.zeros(n, dtype=bool)
    lx = np.zeros(n, dtype=bool)
    se = np.zeros(n, dtype=bool)
    sx = np.zeros(n, dtype=bool)

    in_pos = False
    t_dir  = 0
    last_bar = -999_999
    warmup = max(K + 1, 200)

    for i in range(warmup, n):
        up   = bool(brick_up[i])
        h_et = et_hours[i]
        m_et = et_minutes[i]

        # Forced close 15:45 ET
        if in_pos and (h_et > 15 or (h_et == 15 and m_et >= 45)):
            lx[i] = (t_dir == 1)
            sx[i] = (t_dir == -1)
            in_pos = False; t_dir = 0
            continue

        # Normal exit: opposing brick
        if in_pos:
            is_opp = (t_dir == 1 and not up) or (t_dir == -1 and up)
            lx[i] = is_opp and t_dir == 1
            sx[i] = is_opp and t_dir == -1
            if is_opp:
                in_pos = False; t_dir = 0

        if in_pos:
            continue

        if h_et > 15 or (h_et == 15 and m_et >= 30):
            continue

        if (i - last_bar) < cooldown:
            continue

        # Check K consecutive same-direction bricks
        window = brick_up[i - K + 1 : i + 1]
        all_up   = bool(np.all(window))
        all_down = bool(not np.any(window))
        if not (all_up or all_down):
            continue

        # Check monotonically decreasing brick_td (each faster than previous)
        td_win = brick_td[i - K + 1 : i + 1]
        if np.any(np.isnan(td_win)):
            continue
        monotonic = True
        for j in range(1, len(td_win)):
            if td_win[j] > td_win[j - 1]:
                monotonic = False
                break
        if not monotonic:
            continue

        cand = 1 if all_up else -1

        if cand == 1 and not gate_long_ok[i]:
            continue
        if cand == -1 and not gate_short_ok[i]:
            continue

        if cand == 1:
            le[i] = True
        else:
            se[i] = True
        in_pos = True; t_dir = cand; last_bar = i

    return le, lx, se, sx


def _generate_signals_ace(brick_up, et_hours, et_minutes,
                          gate_long_ok, gate_short_ok,
                          streak_accel, streak_len, brick_td,
                          K, cooldown):
    """Acceleration Entry: streak K+ bricks with acceleration < 1.0."""
    n = len(brick_up)
    le = np.zeros(n, dtype=bool)
    lx = np.zeros(n, dtype=bool)
    se = np.zeros(n, dtype=bool)
    sx = np.zeros(n, dtype=bool)

    in_pos = False
    t_dir  = 0
    last_bar = -999_999
    warmup = max(K + 1, 200)

    for i in range(warmup, n):
        up   = bool(brick_up[i])
        h_et = et_hours[i]
        m_et = et_minutes[i]

        # Forced close 15:45 ET
        if in_pos and (h_et > 15 or (h_et == 15 and m_et >= 45)):
            lx[i] = (t_dir == 1)
            sx[i] = (t_dir == -1)
            in_pos = False; t_dir = 0
            continue

        # Normal exit: opposing brick
        if in_pos:
            is_opp = (t_dir == 1 and not up) or (t_dir == -1 and up)
            lx[i] = is_opp and t_dir == 1
            sx[i] = is_opp and t_dir == -1
            if is_opp:
                in_pos = False; t_dir = 0

        if in_pos:
            continue

        if h_et > 15 or (h_et == 15 and m_et >= 30):
            continue

        if (i - last_bar) < cooldown:
            continue

        # Check streak length >= K and direction matches current brick
        sl = streak_len[i]
        if np.isnan(sl):
            continue
        sl = int(sl)
        if abs(sl) < K:
            continue

        # Direction: positive streak_len = up streak
        cand = 1 if sl > 0 else -1
        # Current brick must continue the streak
        if (cand == 1 and not up) or (cand == -1 and up):
            continue

        # Check acceleration < 1.0 (second half faster than first half)
        sa = streak_accel[i]
        if not np.isnan(sa):
            if sa >= 1.0:
                continue
        else:
            # Fallback for short streaks: compare last vs first brick_td
            td_first = brick_td[i - abs(sl) + 1]
            td_last  = brick_td[i]
            if np.isnan(td_first) or np.isnan(td_last):
                continue
            if td_last >= td_first:
                continue

        if cand == 1 and not gate_long_ok[i]:
            continue
        if cand == -1 and not gate_short_ok[i]:
            continue

        if cand == 1:
            le[i] = True
        else:
            se[i] = True
        in_pos = True; t_dir = cand; last_bar = i

    return le, lx, se, sx


def _generate_signals_mce(brick_up, et_hours, et_minutes,
                          gate_long_ok, gate_short_ok,
                          N, M, cooldown):
    """Momentum Cascade Entry: N-of-M bricks same direction (relaxed R001)."""
    n = len(brick_up)
    le = np.zeros(n, dtype=bool)
    lx = np.zeros(n, dtype=bool)
    se = np.zeros(n, dtype=bool)
    sx = np.zeros(n, dtype=bool)

    in_pos = False
    t_dir  = 0
    last_bar = -999_999
    warmup = max(M + 1, 200)

    for i in range(warmup, n):
        up   = bool(brick_up[i])
        h_et = et_hours[i]
        m_et = et_minutes[i]

        # Forced close 15:45 ET
        if in_pos and (h_et > 15 or (h_et == 15 and m_et >= 45)):
            lx[i] = (t_dir == 1)
            sx[i] = (t_dir == -1)
            in_pos = False; t_dir = 0
            continue

        # Normal exit: opposing brick
        if in_pos:
            is_opp = (t_dir == 1 and not up) or (t_dir == -1 and up)
            lx[i] = is_opp and t_dir == 1
            sx[i] = is_opp and t_dir == -1
            if is_opp:
                in_pos = False; t_dir = 0

        if in_pos:
            continue

        if h_et > 15 or (h_et == 15 and m_et >= 30):
            continue

        if (i - last_bar) < cooldown:
            continue

        # Count direction in M-brick window ending at i
        window = brick_up[i - M + 1 : i + 1]
        n_up   = int(np.sum(window))
        n_down = M - n_up

        if n_up >= N:
            cand = 1
        elif n_down >= N:
            cand = -1
        else:
            continue

        if cand == 1 and not gate_long_ok[i]:
            continue
        if cand == -1 and not gate_short_ok[i]:
            continue

        if cand == 1:
            le[i] = True
        else:
            se[i] = True
        in_pos = True; t_dir = cand; last_bar = i

    return le, lx, se, sx


def _generate_signals_hybrid(brick_up, et_hours, et_minutes,
                             gate_long_ok, gate_short_ok,
                             vel_ratio,
                             n_bricks, bre_K, vel_threshold, cooldown):
    """
    Hybrid R001/R002 + BRE: dual entry triggers in one strategy.

    R001: N consecutive same-direction bricks (continuation) — uses cooldown
    R002: N opposite then reversal — exempt from cooldown
    BRE:  K consecutive fast bricks (vel_ratio < threshold) — uses cooldown

    Both share one position (no pyramiding). First opposing brick exits.
    """
    n = len(brick_up)
    le = np.zeros(n, dtype=bool)
    lx = np.zeros(n, dtype=bool)
    se = np.zeros(n, dtype=bool)
    sx = np.zeros(n, dtype=bool)

    in_pos = False
    t_dir  = 0
    last_entry_bar = -999_999
    warmup = max(n_bricks + 1, bre_K + 1, 200)

    for i in range(warmup, n):
        up   = bool(brick_up[i])
        h_et = et_hours[i]
        m_et = et_minutes[i]

        # Forced close 15:45 ET
        if in_pos and (h_et > 15 or (h_et == 15 and m_et >= 45)):
            lx[i] = (t_dir == 1)
            sx[i] = (t_dir == -1)
            in_pos = False; t_dir = 0
            continue

        # Normal exit: opposing brick
        if in_pos:
            is_opp = (t_dir == 1 and not up) or (t_dir == -1 and up)
            lx[i] = is_opp and t_dir == 1
            sx[i] = is_opp and t_dir == -1
            if is_opp:
                in_pos = False; t_dir = 0

        if in_pos:
            continue

        # No entry after 15:30 ET
        if h_et > 15 or (h_et == 15 and m_et >= 30):
            continue

        cand = 0
        is_r002 = False

        # ── R002: reversal after N consecutive bricks ──
        prev = brick_up[i - n_bricks : i]
        prev_all_up   = bool(np.all(prev))
        prev_all_down = bool(not np.any(prev))

        if prev_all_up and not up:
            cand = -1; is_r002 = True
        elif prev_all_down and up:
            cand = 1; is_r002 = True

        # ── R001 or BRE (both use cooldown) ──
        if cand == 0 and (i - last_entry_bar) >= cooldown:
            # Try R001 first
            window_r001 = brick_up[i - n_bricks + 1 : i + 1]
            all_up   = bool(np.all(window_r001))
            all_down = bool(not np.any(window_r001))
            if all_up:
                cand = 1
            elif all_down:
                cand = -1

            # If R001 didn't fire, try BRE
            if cand == 0:
                window_bre = brick_up[i - bre_K + 1 : i + 1]
                bre_all_up   = bool(np.all(window_bre))
                bre_all_down = bool(not np.any(window_bre))
                if bre_all_up or bre_all_down:
                    vel_win = vel_ratio[i - bre_K + 1 : i + 1]
                    if not np.any(np.isnan(vel_win)) and np.all(vel_win < vel_threshold):
                        cand = 1 if bre_all_up else -1

        if cand == 0:
            continue

        # Gate check
        if cand == 1 and not gate_long_ok[i]:
            continue
        if cand == -1 and not gate_short_ok[i]:
            continue

        if cand == 1:
            le[i] = True
        else:
            se[i] = True
        in_pos = True; t_dir = cand
        if not is_r002:
            last_entry_bar = i

    return le, lx, se, sx


# ── Hybrid sweep grid ────────────────────────────────────────────────────────

HYBRID_PARAMS = {
    "n_bricks":      [7, 8, 9, 10, 11],
    "bre_K":         [5, 6, 7],
    "vel_threshold": [0.3, 0.4, 0.5, 0.6],
    "cooldown":      [20, 30, 40, 50],
}
HYBRID_ADX  = [0, 40, 50]
HYBRID_PSAR = [True, False]


def run_hybrid_sweep():
    """Sweep hybrid R001/R002 + BRE signal."""
    print(f"Loading Renko + all indicators...", flush=True)
    df = _load_renko_all_indicators_v4(INSTRUMENT["renko_file"])
    print(f"Ready — {len(df)} bricks", flush=True)

    et_hours, et_minutes = _compute_et_hours(df.index)
    gates = _precompute_gates(df)
    brick_up  = df["brick_up"].values
    vel_ratio = df["vel_ratio"].values

    gate_combos = list(itertools.product(HYBRID_ADX, HYBRID_PSAR))
    keys = list(HYBRID_PARAMS.keys())
    param_combos = [dict(zip(keys, v)) for v in itertools.product(*HYBRID_PARAMS.values())]

    n_total = len(param_combos) * len(gate_combos)

    print(f"\n{'='*100}")
    print("  MYM Hybrid Signal Sweep — R001/R002 + BRE")
    print(f"{'='*100}")
    print(f"  n_bricks       : {HYBRID_PARAMS['n_bricks']}")
    print(f"  bre_K          : {HYBRID_PARAMS['bre_K']}")
    print(f"  vel_threshold  : {HYBRID_PARAMS['vel_threshold']}")
    print(f"  cooldown       : {HYBRID_PARAMS['cooldown']}")
    print(f"  ADX            : {HYBRID_ADX}")
    print(f"  PSAR           : {HYBRID_PSAR}")
    print(f"  Total          : {n_total} runs ({n_total * 2} IS+OOS backtests)")
    print()

    results = []
    done = 0

    for pc in param_combos:
        for adx_t, use_psar in gate_combos:
            gl, gs = _combine_gates(gates, adx_t, use_psar)
            le_, lx_, se_, sx_ = _generate_signals_hybrid(
                brick_up, et_hours, et_minutes, gl, gs,
                vel_ratio,
                pc["n_bricks"], pc["bre_K"], pc["vel_threshold"], pc["cooldown"])
            df["long_entry"] = le_; df["long_exit"] = lx_
            df["short_entry"] = se_; df["short_exit"] = sx_

            is_r  = _run_backtest(df, INSTRUMENT["is_start"], INSTRUMENT["is_end"])
            oos_r = _run_backtest(df, INSTRUMENT["oos_start"], INSTRUMENT["oos_end"])
            decay = ((oos_r["pf"] / is_r["pf"]) - 1.0) * 100 if is_r["pf"] > 0 and not math.isinf(is_r["pf"]) and not math.isinf(oos_r["pf"]) else float("nan")

            psar_str = "psar" if use_psar else "nopsar"
            stack = f"hyb_n{pc['n_bricks']}_bK{pc['bre_K']}_vt{pc['vel_threshold']}_cd{pc['cooldown']}_a{adx_t}_{psar_str}"

            results.append({
                "signal": "hybrid", "K": pc["bre_K"], "vel_threshold": pc["vel_threshold"],
                "N": None, "M": None, "n_bricks": pc["n_bricks"],
                "cooldown": pc["cooldown"], "adx": adx_t, "psar": use_psar,
                "stack": stack,
                "is_pf": is_r["pf"], "is_trades": is_r["trades"],
                "is_wr": is_r["wr"], "is_net": is_r["net"],
                "oos_pf": oos_r["pf"], "oos_trades": oos_r["trades"],
                "oos_wr": oos_r["wr"], "oos_net": oos_r["net"],
                "decay_pct": decay,
            })
            done += 1
            if done % 200 == 0:
                print(f"  {done:>5}/{n_total} | {stack:<55} | OOS PF={oos_r['pf']:>8.2f} T={oos_r['trades']:>4}", flush=True)

    print(f"  {done:>5}/{n_total} — Complete", flush=True)
    return results


def _summarize_hybrid(results):
    """Print hybrid sweep summary."""
    print(f"\n{'='*100}")
    print("  Hybrid R001/R002 + BRE — Results Summary")
    print(f"{'='*100}")

    viable = [r for r in results if r["oos_trades"] >= MIN_TRADES]
    viable.sort(key=lambda r: r["oos_pf"], reverse=True)

    # Top 25
    print(f"\n  Top 25 (OOS trades >= {MIN_TRADES}):")
    print(f"  {'Stack':<55} | {'IS PF':>7} {'T':>4} | {'OOS PF':>8} {'T':>4} {'WR%':>6} {'Net$':>9} {'Decay':>8}")
    print(f"  {'-'*110}")
    for r in viable[:25]:
        d = f"{r['decay_pct']:+.1f}%" if not math.isnan(r.get("decay_pct", float("nan"))) else "n/a"
        pf_str = "inf" if math.isinf(r["oos_pf"]) else f"{r['oos_pf']:.2f}"
        is_pf_str = "inf" if math.isinf(r["is_pf"]) else f"{r['is_pf']:.2f}"
        print(f"  {r['stack']:<55} | {is_pf_str:>7} {r['is_trades']:>4} | {pf_str:>8} {r['oos_trades']:>4} {r['oos_wr']:>5.1f}% {r['oos_net']:>9.2f} {d:>8}")

    # By n_bricks
    print(f"\n  By n_bricks (avg OOS PF):")
    for nb in HYBRID_PARAMS["n_bricks"]:
        nb_v = [r for r in viable if r["n_bricks"] == nb]
        if nb_v:
            avg_pf = np.mean([r["oos_pf"] for r in nb_v if not math.isinf(r["oos_pf"])])
            avg_t  = np.mean([r["oos_trades"] for r in nb_v])
            print(f"    n={nb}: avg PF={avg_pf:.2f}, avg T={avg_t:.0f}, N={len(nb_v)}")

    # By bre_K
    print(f"\n  By bre_K (avg OOS PF):")
    for bk in HYBRID_PARAMS["bre_K"]:
        bk_v = [r for r in viable if r["K"] == bk]
        if bk_v:
            avg_pf = np.mean([r["oos_pf"] for r in bk_v if not math.isinf(r["oos_pf"])])
            avg_t  = np.mean([r["oos_trades"] for r in bk_v])
            print(f"    K={bk}: avg PF={avg_pf:.2f}, avg T={avg_t:.0f}, N={len(bk_v)}")

    # By vel_threshold
    print(f"\n  By vel_threshold (avg OOS PF):")
    for vt in HYBRID_PARAMS["vel_threshold"]:
        vt_v = [r for r in viable if r["vel_threshold"] == vt]
        if vt_v:
            avg_pf = np.mean([r["oos_pf"] for r in vt_v if not math.isinf(r["oos_pf"])])
            avg_t  = np.mean([r["oos_trades"] for r in vt_v])
            print(f"    vt={vt}: avg PF={avg_pf:.2f}, avg T={avg_t:.0f}, N={len(vt_v)}")

    # By ADX
    print(f"\n  By ADX threshold:")
    for adx_t in HYBRID_ADX:
        adx_v = [r for r in viable if r["adx"] == adx_t]
        if adx_v:
            avg_pf = np.mean([r["oos_pf"] for r in adx_v if not math.isinf(r["oos_pf"])])
            avg_t  = np.mean([r["oos_trades"] for r in adx_v])
            print(f"    ADX>={adx_t}: avg PF={avg_pf:.2f}, avg T={avg_t:.0f}, N={len(adx_v)}")

    # Best overall
    if viable:
        best = viable[0]
        pf_str = "inf" if math.isinf(best["oos_pf"]) else f"{best['oos_pf']:.2f}"
        d = f"{best['decay_pct']:+.1f}%" if not math.isnan(best.get("decay_pct", float("nan"))) else "n/a"
        print(f"\n  {'='*60}")
        print(f"  MYM001 benchmark: PF=232, 161t, 88.8% WR, $22,493 net (TV)")
        print(f"  {'='*60}")
        print(f"  Best hybrid: PF={pf_str}, T={best['oos_trades']}, WR={best['oos_wr']:.1f}%, Net=${best['oos_net']:.2f}, Decay={d}")
        print(f"  Config: {best['stack']}")
        if not math.isinf(best["oos_pf"]):
            print(f"  PF delta vs MYM001: {((best['oos_pf'] / 232) - 1) * 100:+.1f}%")

    # Top by net profit
    net_sorted = sorted(viable, key=lambda r: r["oos_net"], reverse=True)
    print(f"\n  Top 10 by OOS Net Profit:")
    print(f"  {'Stack':<55} | {'OOS PF':>8} {'T':>4} {'WR%':>6} {'Net$':>9}")
    print(f"  {'-'*90}")
    for r in net_sorted[:10]:
        pf_str = "inf" if math.isinf(r["oos_pf"]) else f"{r['oos_pf']:.2f}"
        print(f"  {r['stack']:<55} | {pf_str:>8} {r['oos_trades']:>4} {r['oos_wr']:>5.1f}% {r['oos_net']:>9.2f}")


# ── Main sweep ────────────────────────────────────────────────────────────────

def run_sweep():
    print(f"Loading Renko + all indicators...", flush=True)
    df = _load_renko_all_indicators_v4(INSTRUMENT["renko_file"])
    print(f"Ready — {len(df)} bricks", flush=True)

    et_hours, et_minutes = _compute_et_hours(df.index)

    # Pre-compute gates
    gates = _precompute_gates(df)

    # Pre-extract data arrays
    brick_up     = df["brick_up"].values
    vel_ratio    = df["vel_ratio"].values
    brick_td     = df["brick_td"].values
    streak_accel = df["streak_accel"].values
    streak_len   = df["streak_len"].values

    # Gate combos
    gate_combos = list(itertools.product(ADX_THRESHOLDS, PSAR_OPTIONS))

    results = []
    total = 0

    # Count total runs
    n_bre = len(BRE_K) * len(BRE_VEL_THRESH) * len(COOLDOWNS) * len(gate_combos)
    n_vie = len(VIE_K) * len(COOLDOWNS) * len(gate_combos)
    n_ace = len(ACE_K) * len(COOLDOWNS) * len(gate_combos)
    n_mce = len(MCE_NM) * len(COOLDOWNS) * len(gate_combos)
    n_base = len(BASELINE_N) * len(COOLDOWNS) * len(gate_combos)
    n_total = n_bre + n_vie + n_ace + n_mce + n_base

    print(f"\n{'='*100}")
    print("  MYM Novel Entry Signals Research — Brick 14")
    print(f"{'='*100}")
    print(f"  BRE (Brick Burst)        : {n_bre:>4} runs  K={BRE_K} vel_thresh={BRE_VEL_THRESH}")
    print(f"  VIE (Velocity Impulse)   : {n_vie:>4} runs  K={VIE_K}")
    print(f"  ACE (Acceleration)       : {n_ace:>4} runs  K={ACE_K}")
    print(f"  MCE (Momentum Cascade)   : {n_mce:>4} runs  N/M={MCE_NM}")
    print(f"  R007 (Baseline R001+R002): {n_base:>4} runs  n={BASELINE_N}")
    print(f"  Total                    : {n_total:>4} runs ({n_total * 2} IS+OOS backtests)")
    print(f"  Gates: ADX={ADX_THRESHOLDS} × PSAR={PSAR_OPTIONS} = {len(gate_combos)} combos")
    print()

    done = 0

    def _backtest_pair(label):
        """Run IS + OOS backtests, return result dict."""
        is_r  = _run_backtest(df, INSTRUMENT["is_start"], INSTRUMENT["is_end"])
        oos_r = _run_backtest(df, INSTRUMENT["oos_start"], INSTRUMENT["oos_end"])
        decay = ((oos_r["pf"] / is_r["pf"]) - 1.0) * 100 if is_r["pf"] > 0 and not math.isinf(is_r["pf"]) and not math.isinf(oos_r["pf"]) else float("nan")
        return is_r, oos_r, decay

    # ── BRE ──
    for K in BRE_K:
        for vt in BRE_VEL_THRESH:
            for cd in COOLDOWNS:
                for adx_t, use_psar in gate_combos:
                    gl, gs = _combine_gates(gates, adx_t, use_psar)
                    le_, lx_, se_, sx_ = _generate_signals_bre(
                        brick_up, et_hours, et_minutes, gl, gs,
                        vel_ratio, K, vt, cd)
                    df["long_entry"] = le_; df["long_exit"] = lx_
                    df["short_entry"] = se_; df["short_exit"] = sx_
                    is_r, oos_r, decay = _backtest_pair(f"BRE K={K} vt={vt}")
                    psar_str = "psar" if use_psar else "nopsar"
                    results.append({
                        "signal": "bre", "K": K, "vel_threshold": vt,
                        "N": None, "M": None, "n_bricks": None,
                        "cooldown": cd, "adx": adx_t, "psar": use_psar,
                        "stack": f"bre_K{K}_vt{vt}_cd{cd}_a{adx_t}_{psar_str}",
                        "is_pf": is_r["pf"], "is_trades": is_r["trades"],
                        "is_wr": is_r["wr"], "is_net": is_r["net"],
                        "oos_pf": oos_r["pf"], "oos_trades": oos_r["trades"],
                        "oos_wr": oos_r["wr"], "oos_net": oos_r["net"],
                        "decay_pct": decay,
                    })
                    done += 1
                    if done % 100 == 0:
                        print(f"  {done:>4}/{n_total} | {results[-1]['stack']:<45} | OOS PF={oos_r['pf']:>8.2f} T={oos_r['trades']:>4}", flush=True)

    # ── VIE ──
    for K in VIE_K:
        for cd in COOLDOWNS:
            for adx_t, use_psar in gate_combos:
                gl, gs = _combine_gates(gates, adx_t, use_psar)
                le_, lx_, se_, sx_ = _generate_signals_vie(
                    brick_up, et_hours, et_minutes, gl, gs,
                    brick_td, K, cd)
                df["long_entry"] = le_; df["long_exit"] = lx_
                df["short_entry"] = se_; df["short_exit"] = sx_
                is_r, oos_r, decay = _backtest_pair(f"VIE K={K}")
                psar_str = "psar" if use_psar else "nopsar"
                results.append({
                    "signal": "vie", "K": K, "vel_threshold": None,
                    "N": None, "M": None, "n_bricks": None,
                    "cooldown": cd, "adx": adx_t, "psar": use_psar,
                    "stack": f"vie_K{K}_cd{cd}_a{adx_t}_{psar_str}",
                    "is_pf": is_r["pf"], "is_trades": is_r["trades"],
                    "is_wr": is_r["wr"], "is_net": is_r["net"],
                    "oos_pf": oos_r["pf"], "oos_trades": oos_r["trades"],
                    "oos_wr": oos_r["wr"], "oos_net": oos_r["net"],
                    "decay_pct": decay,
                })
                done += 1
                if done % 100 == 0:
                    print(f"  {done:>4}/{n_total} | {results[-1]['stack']:<45} | OOS PF={oos_r['pf']:>8.2f} T={oos_r['trades']:>4}", flush=True)

    # ── ACE ──
    for K in ACE_K:
        for cd in COOLDOWNS:
            for adx_t, use_psar in gate_combos:
                gl, gs = _combine_gates(gates, adx_t, use_psar)
                le_, lx_, se_, sx_ = _generate_signals_ace(
                    brick_up, et_hours, et_minutes, gl, gs,
                    streak_accel, streak_len, brick_td, K, cd)
                df["long_entry"] = le_; df["long_exit"] = lx_
                df["short_entry"] = se_; df["short_exit"] = sx_
                is_r, oos_r, decay = _backtest_pair(f"ACE K={K}")
                psar_str = "psar" if use_psar else "nopsar"
                results.append({
                    "signal": "ace", "K": K, "vel_threshold": None,
                    "N": None, "M": None, "n_bricks": None,
                    "cooldown": cd, "adx": adx_t, "psar": use_psar,
                    "stack": f"ace_K{K}_cd{cd}_a{adx_t}_{psar_str}",
                    "is_pf": is_r["pf"], "is_trades": is_r["trades"],
                    "is_wr": is_r["wr"], "is_net": is_r["net"],
                    "oos_pf": oos_r["pf"], "oos_trades": oos_r["trades"],
                    "oos_wr": oos_r["wr"], "oos_net": oos_r["net"],
                    "decay_pct": decay,
                })
                done += 1
                if done % 100 == 0:
                    print(f"  {done:>4}/{n_total} | {results[-1]['stack']:<45} | OOS PF={oos_r['pf']:>8.2f} T={oos_r['trades']:>4}", flush=True)

    # ── MCE ──
    for N_val, M_val in MCE_NM:
        for cd in COOLDOWNS:
            for adx_t, use_psar in gate_combos:
                gl, gs = _combine_gates(gates, adx_t, use_psar)
                le_, lx_, se_, sx_ = _generate_signals_mce(
                    brick_up, et_hours, et_minutes, gl, gs,
                    N_val, M_val, cd)
                df["long_entry"] = le_; df["long_exit"] = lx_
                df["short_entry"] = se_; df["short_exit"] = sx_
                is_r, oos_r, decay = _backtest_pair(f"MCE {N_val}/{M_val}")
                psar_str = "psar" if use_psar else "nopsar"
                results.append({
                    "signal": "mce", "K": None, "vel_threshold": None,
                    "N": N_val, "M": M_val, "n_bricks": None,
                    "cooldown": cd, "adx": adx_t, "psar": use_psar,
                    "stack": f"mce_{N_val}of{M_val}_cd{cd}_a{adx_t}_{psar_str}",
                    "is_pf": is_r["pf"], "is_trades": is_r["trades"],
                    "is_wr": is_r["wr"], "is_net": is_r["net"],
                    "oos_pf": oos_r["pf"], "oos_trades": oos_r["trades"],
                    "oos_wr": oos_r["wr"], "oos_net": oos_r["net"],
                    "decay_pct": decay,
                })
                done += 1
                if done % 100 == 0:
                    print(f"  {done:>4}/{n_total} | {results[-1]['stack']:<45} | OOS PF={oos_r['pf']:>8.2f} T={oos_r['trades']:>4}", flush=True)

    # ── Baseline R001+R002 ──
    for nb in BASELINE_N:
        for cd in COOLDOWNS:
            for adx_t, use_psar in gate_combos:
                gl, gs = _combine_gates(gates, adx_t, use_psar)
                le_, lx_, se_, sx_ = _generate_signal_arrays(
                    brick_up, nb, cd, gl, gs, et_hours, et_minutes)
                df["long_entry"] = le_; df["long_exit"] = lx_
                df["short_entry"] = se_; df["short_exit"] = sx_
                is_r, oos_r, decay = _backtest_pair(f"R007 n={nb}")
                psar_str = "psar" if use_psar else "nopsar"
                results.append({
                    "signal": "r007", "K": None, "vel_threshold": None,
                    "N": None, "M": None, "n_bricks": nb,
                    "cooldown": cd, "adx": adx_t, "psar": use_psar,
                    "stack": f"r007_n{nb}_cd{cd}_a{adx_t}_{psar_str}",
                    "is_pf": is_r["pf"], "is_trades": is_r["trades"],
                    "is_wr": is_r["wr"], "is_net": is_r["net"],
                    "oos_pf": oos_r["pf"], "oos_trades": oos_r["trades"],
                    "oos_wr": oos_r["wr"], "oos_net": oos_r["net"],
                    "decay_pct": decay,
                })
                done += 1

    print(f"  {done:>4}/{n_total} — Complete", flush=True)
    return results


# ── Summary ───────────────────────────────────────────────────────────────────

def _summarize(results):
    print(f"\n{'='*100}")
    print("  MYM Novel Signals — Results Summary")
    print(f"{'='*100}")

    # Filter viable (OOS trades >= MIN_TRADES)
    viable = [r for r in results if r["oos_trades"] >= MIN_TRADES]
    viable.sort(key=lambda r: r["oos_pf"], reverse=True)

    # ── Top 25 ──
    print(f"\n  Top 25 (OOS trades >= {MIN_TRADES}):")
    print(f"  {'Signal':<6} {'Stack':<45} | {'IS PF':>7} {'T':>4} | {'OOS PF':>8} {'T':>4} {'WR%':>6} {'Net$':>9} {'Decay':>8}")
    print(f"  {'-'*110}")
    for r in viable[:25]:
        d = f"{r['decay_pct']:+.1f}%" if not math.isnan(r.get("decay_pct", float("nan"))) else "n/a"
        pf_str = "inf" if math.isinf(r["oos_pf"]) else f"{r['oos_pf']:.2f}"
        is_pf_str = "inf" if math.isinf(r["is_pf"]) else f"{r['is_pf']:.2f}"
        print(f"  {r['signal']:<6} {r['stack']:<45} | {is_pf_str:>7} {r['is_trades']:>4} | {pf_str:>8} {r['oos_trades']:>4} {r['oos_wr']:>5.1f}% {r['oos_net']:>9.2f} {d:>8}")

    # ── By signal type ──
    print(f"\n  By Signal Type (OOS trades >= {MIN_TRADES}):")
    print(f"  {'Signal':<8} | {'Avg PF':>8} {'Avg Trades':>11} {'N Viable':>9} {'Best PF':>9} {'Best Trades':>12} {'Best Net$':>10}")
    print(f"  {'-'*85}")
    for sig in ["bre", "vie", "ace", "mce", "r007"]:
        sig_v = [r for r in viable if r["signal"] == sig]
        if sig_v:
            avg_pf = np.mean([r["oos_pf"] for r in sig_v if not math.isinf(r["oos_pf"])])
            avg_t  = np.mean([r["oos_trades"] for r in sig_v])
            best   = sig_v[0]
            best_pf = "inf" if math.isinf(best["oos_pf"]) else f"{best['oos_pf']:.2f}"
            print(f"  {sig:<8} | {avg_pf:>8.2f} {avg_t:>11.1f} {len(sig_v):>9} {best_pf:>9} {best['oos_trades']:>12} {best['oos_net']:>10.2f}")
        else:
            print(f"  {sig:<8} | {'(no viable results)':>50}")

    # ── By ADX threshold ──
    print(f"\n  By ADX Threshold:")
    print(f"  {'ADX':>6} | {'Avg PF':>8} {'N Viable':>9}")
    print(f"  {'-'*30}")
    for adx_t in ADX_THRESHOLDS:
        adx_v = [r for r in viable if r["adx"] == adx_t]
        if adx_v:
            avg_pf = np.mean([r["oos_pf"] for r in adx_v if not math.isinf(r["oos_pf"])])
            print(f"  {adx_t:>6} | {avg_pf:>8.2f} {len(adx_v):>9}")

    # ── PSAR value-add ──
    print(f"\n  PSAR Value-Add:")
    print(f"  {'PSAR':>6} | {'Avg PF':>8} {'N Viable':>9}")
    print(f"  {'-'*30}")
    for p in [True, False]:
        pv = [r for r in viable if r["psar"] == p]
        if pv:
            avg_pf = np.mean([r["oos_pf"] for r in pv if not math.isinf(r["oos_pf"])])
            print(f"  {'Yes' if p else 'No':>6} | {avg_pf:>8.2f} {len(pv):>9}")

    # ── Best per signal type ──
    print(f"\n  Best Config Per Signal Type:")
    print(f"  {'-'*110}")
    for sig in ["bre", "vie", "ace", "mce", "r007"]:
        sig_v = [r for r in viable if r["signal"] == sig]
        if sig_v:
            best = sig_v[0]
            pf_str = "inf" if math.isinf(best["oos_pf"]) else f"{best['oos_pf']:.2f}"
            d = f"{best['decay_pct']:+.1f}%" if not math.isnan(best.get("decay_pct", float("nan"))) else "n/a"
            print(f"  {sig:<6}: PF={pf_str} T={best['oos_trades']} WR={best['oos_wr']*100:.1f}% Net=${best['oos_net']:.2f} Decay={d}")
            print(f"         Config: {best['stack']}")

    # ── vs MYM001 ──
    print(f"\n  {'='*60}")
    print(f"  MYM001 benchmark: PF=232, 161 trades, 88.8% WR, $22,493 net")
    print(f"  {'='*60}")
    if viable:
        best = viable[0]
        pf_str = "inf" if math.isinf(best["oos_pf"]) else f"{best['oos_pf']:.2f}"
        print(f"  Best novel signal: [{best['signal']}] PF={pf_str}, T={best['oos_trades']}, WR={best['oos_wr']*100:.1f}%, Net=${best['oos_net']:.2f}")
        if not math.isinf(best["oos_pf"]):
            print(f"  PF delta: {((best['oos_pf'] / 232) - 1) * 100:+.1f}%")
        print(f"  Trade delta: {((best['oos_trades'] / 161) - 1) * 100:+.1f}%")
        print(f"  Net delta: {((best['oos_net'] / 22493) - 1) * 100:+.1f}%")

    # ── Top by net profit ──
    net_sorted = sorted(viable, key=lambda r: r["oos_net"], reverse=True)
    print(f"\n  Top 10 by OOS Net Profit:")
    print(f"  {'Signal':<6} {'Stack':<45} | {'OOS PF':>8} {'T':>4} {'WR%':>6} {'Net$':>9}")
    print(f"  {'-'*90}")
    for r in net_sorted[:10]:
        pf_str = "inf" if math.isinf(r["oos_pf"]) else f"{r['oos_pf']:.2f}"
        print(f"  {r['signal']:<6} {r['stack']:<45} | {pf_str:>8} {r['oos_trades']:>4} {r['oos_wr']*100:>5.1f}% {r['oos_net']:>9.2f}")


# ── Main ──────────────────────────────────────────────────────────────────────

HYBRID_OUTPUT_FILE = ROOT / "ai_context" / "mym_hybrid_results.json"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid", action="store_true", help="Run hybrid R001/R002+BRE sweep")
    args = parser.parse_args()

    if args.hybrid:
        results = run_hybrid_sweep()
        _summarize_hybrid(results)
        HYBRID_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HYBRID_OUTPUT_FILE, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved {len(results)} results -> {HYBRID_OUTPUT_FILE}")
    else:
        results = run_sweep()
        _summarize(results)
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved {len(results)} results -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
