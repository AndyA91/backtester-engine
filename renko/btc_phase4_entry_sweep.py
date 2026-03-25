#!/usr/bin/env python3
"""
btc_phase4_entry_sweep.py — BTC Phase 4: Hybrid Entry System Sweep (Long Only)

Tests three new entry triggers alongside R007 (brick patterns) within a single
strategy. Each entry type shares the same gates and exit logic.

Entry systems:
  R007     = existing brick pattern (R001 momentum + R002 reversal)
  CROSS    = indicator crossover on an up brick (no brick pattern needed)
  PULLBACK = 1-2 down bricks after established uptrend, re-enter on up brick
  FLIP     = regime flip (PSAR/supertrend flips bullish)

Each system is tested:
  1. Solo (just that entry type alone)
  2. Combined with R007 (R007 + that entry type)
  3. All combined (R007 + all three new entries)

Gate stack fixed to Phase 3 winner: psar_dir + ADX>=30 + vol<=1.5 + HTF ADX>=40

Uses ProcessPoolExecutor — one worker per entry system config.

Usage:
  python renko/btc_phase4_entry_sweep.py
  python renko/btc_phase4_entry_sweep.py --no-parallel
"""

import argparse
import contextlib
import io
import itertools
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from renko.config import MAX_WORKERS
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# ── Instrument config ──────────────────────────────────────────────────────────

LTF_FILE   = "OANDA_BTCUSD.SPOT.US, 1S renko 150.csv"
HTF_FILE   = "OANDA_BTCUSD.SPOT.US, 1S renko 300.csv"
IS_START   = "2024-06-04"
IS_END     = "2025-09-30"
OOS_START  = "2025-10-01"
OOS_END    = "2026-03-19"
COMMISSION = 0.0046
CAPITAL    = 1000.0
QTY_VALUE  = 20
VOL_MAX    = 1.5
ADX_THRESH = 30
HTF_ADX_THRESH = 40

# ── Param grid ─────────────────────────────────────────────────────────────────

PARAM_GRID = {
    "n_bricks": [2, 3, 4, 5],
    "cooldown": [10, 20, 30],
}

# Crossover indicator choices
CROSS_INDICATORS = ["escgo", "stoch", "macd_hist", "kama_slope"]

# Pullback depth choices
PULLBACK_DEPTHS = [1, 2, 3]

# Flip indicator choices
FLIP_INDICATORS = ["psar", "supertrend", "ema_cross"]


# ── Data loading ───────────────────────────────────────────────────────────────

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


# ── Gate computation (fixed Phase 3 winner) ────────────────────────────────────

def _compute_gates(df_ltf, df_htf):
    """Compute combined long gate: PSAR + ADX>=30 + vol<=1.5 + HTF ADX>=40."""
    sys.path.insert(0, str(ROOT))
    from renko.phase6_sweep import _compute_gate_arrays

    n = len(df_ltf)
    gate = np.ones(n, dtype=bool)

    # PSAR direction
    p6_long, _ = _compute_gate_arrays(df_ltf, "psar_dir")
    gate &= p6_long

    # LTF ADX
    adx = df_ltf["adx"].values
    adx_nan = np.isnan(adx)
    gate &= (adx_nan | (adx >= ADX_THRESH))

    # Vol ratio
    vr = df_ltf["vol_ratio"].values
    vr_nan = np.isnan(vr)
    gate &= (vr_nan | (vr <= VOL_MAX))

    # HTF ADX
    htf_adx = df_htf["adx"].values
    htf_adx_nan = np.isnan(htf_adx)
    htf_gate = htf_adx_nan | (htf_adx >= HTF_ADX_THRESH)

    # Align HTF to LTF
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


# ── Crossover detection arrays ─────────────────────────────────────────────────

def _compute_crossover_long(df, indicator):
    """Return bool array: True on bars where indicator crosses bullish."""
    n = len(df)
    cross = np.zeros(n, dtype=bool)

    if indicator == "escgo":
        fast = df["escgo_fast"].values
        slow = df["escgo_slow"].values
        for i in range(1, n):
            if (not np.isnan(fast[i]) and not np.isnan(slow[i]) and
                not np.isnan(fast[i-1]) and not np.isnan(slow[i-1])):
                cross[i] = fast[i] > slow[i] and fast[i-1] <= slow[i-1]

    elif indicator == "stoch":
        k = df["stoch_k"].values
        d = df["stoch_d"].values
        for i in range(1, n):
            if (not np.isnan(k[i]) and not np.isnan(d[i]) and
                not np.isnan(k[i-1]) and not np.isnan(d[i-1])):
                cross[i] = k[i] > d[i] and k[i-1] <= d[i-1]

    elif indicator == "macd_hist":
        mh = df["macd_hist"].values
        for i in range(1, n):
            if not np.isnan(mh[i]) and not np.isnan(mh[i-1]):
                cross[i] = mh[i] > 0 and mh[i-1] <= 0

    elif indicator == "kama_slope":
        ks = df["kama_slope"].values
        for i in range(1, n):
            if not np.isnan(ks[i]) and not np.isnan(ks[i-1]):
                cross[i] = ks[i] > 0 and ks[i-1] <= 0

    return cross


# ── Flip detection arrays ──────────────────────────────────────────────────────

def _compute_flip_long(df, indicator):
    """Return bool array: True on bars where indicator flips bullish."""
    n = len(df)
    flip = np.zeros(n, dtype=bool)

    if indicator == "psar":
        psar = df["psar_dir"].values
        for i in range(1, n):
            if not np.isnan(psar[i]) and not np.isnan(psar[i-1]):
                flip[i] = psar[i] > 0 and psar[i-1] <= 0

    elif indicator == "supertrend":
        st = df["st_dir"].values
        for i in range(1, n):
            if not np.isnan(st[i]) and not np.isnan(st[i-1]):
                flip[i] = st[i] > 0 and st[i-1] <= 0

    elif indicator == "ema_cross":
        e9 = df["ema9"].values
        e21 = df["ema21"].values
        for i in range(1, n):
            if (not np.isnan(e9[i]) and not np.isnan(e21[i]) and
                not np.isnan(e9[i-1]) and not np.isnan(e21[i-1])):
                flip[i] = e9[i] > e21[i] and e9[i-1] <= e21[i-1]

    return flip


# ── Signal generators ─────────────────────────────────────────────────────────

def _gen_r007_only(df, n_bricks, cooldown, gate):
    """Original R007 long-only signals."""
    n = len(df)
    brick_up = df["brick_up"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)

    in_pos = False
    last_r001 = -999_999
    warmup = max(n_bricks + 1, 200)

    for i in range(warmup, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
        if in_pos:
            continue

        prev = brick_up[i - n_bricks : i]
        prev_all_down = bool(not np.any(prev))
        if prev_all_down and up:
            if gate[i]:
                entry[i] = True
                in_pos = True
            continue

        if (i - last_r001) < cooldown:
            continue
        window = brick_up[i - n_bricks + 1 : i + 1]
        if bool(np.all(window)) and gate[i]:
            entry[i] = True
            in_pos = True
            last_r001 = i

    return entry, exit_


def _gen_cross_only(df, gate, cross_arr, cooldown):
    """Crossover entry: indicator crosses bullish on an up brick."""
    n = len(df)
    brick_up = df["brick_up"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)

    in_pos = False
    last_entry = -999_999
    warmup = 200

    for i in range(warmup, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
        if in_pos:
            continue

        if (i - last_entry) < cooldown:
            continue

        if up and cross_arr[i] and gate[i]:
            entry[i] = True
            in_pos = True
            last_entry = i

    return entry, exit_


def _gen_pullback_only(df, gate, pb_depth, cooldown):
    """Pullback entry: uptrend established, N down bricks, then up brick."""
    n = len(df)
    brick_up = df["brick_up"].values
    ema9 = df["ema9"].values
    ema21 = df["ema21"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)

    in_pos = False
    last_entry = -999_999
    warmup = 200

    for i in range(warmup, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
        if in_pos:
            continue

        if (i - last_entry) < cooldown:
            continue

        # Must be an up brick
        if not up:
            continue

        # Check uptrend: EMA9 > EMA21 (pre-shifted, use at [i])
        if np.isnan(ema9[i]) or np.isnan(ema21[i]) or ema9[i] <= ema21[i]:
            continue

        # Check pullback: exactly pb_depth down bricks before this up brick
        if i < pb_depth + 1:
            continue
        pullback_ok = True
        for j in range(1, pb_depth + 1):
            if brick_up[i - j]:  # not a down brick
                pullback_ok = False
                break
        if not pullback_ok:
            continue

        # Brick before pullback must be up (confirming prior trend)
        if i >= pb_depth + 1 and not brick_up[i - pb_depth - 1]:
            continue

        if gate[i]:
            entry[i] = True
            in_pos = True
            last_entry = i

    return entry, exit_


def _gen_flip_only(df, gate, flip_arr, cooldown):
    """Regime flip entry: indicator flips bullish on an up brick."""
    n = len(df)
    brick_up = df["brick_up"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)

    in_pos = False
    last_entry = -999_999
    warmup = 200

    for i in range(warmup, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
        if in_pos:
            continue

        if (i - last_entry) < cooldown:
            continue

        if up and flip_arr[i] and gate[i]:
            entry[i] = True
            in_pos = True
            last_entry = i

    return entry, exit_


def _gen_combined(df, n_bricks, cooldown, gate,
                  cross_arr=None, pb_depth=None, flip_arr=None,
                  use_r007=True, use_cross=False, use_pullback=False, use_flip=False):
    """Combined entry: multiple triggers share one position."""
    n = len(df)
    brick_up = df["brick_up"].values
    ema9 = df["ema9"].values
    ema21 = df["ema21"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)

    in_pos = False
    last_r001 = -999_999
    last_other = -999_999
    warmup = max(n_bricks + 1, 200)

    for i in range(warmup, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
        if in_pos:
            continue
        if not gate[i]:
            continue

        triggered = False

        # R007
        if use_r007 and not triggered:
            prev = brick_up[i - n_bricks : i]
            prev_all_down = bool(not np.any(prev))
            if prev_all_down and up:
                triggered = True
            elif (i - last_r001) >= cooldown:
                window = brick_up[i - n_bricks + 1 : i + 1]
                if bool(np.all(window)):
                    triggered = True
                    last_r001 = i

        # Crossover
        if use_cross and not triggered and cross_arr is not None:
            if up and cross_arr[i] and (i - last_other) >= cooldown:
                triggered = True
                last_other = i

        # Pullback
        if use_pullback and not triggered and pb_depth is not None:
            if (up and (i - last_other) >= cooldown and
                not np.isnan(ema9[i]) and not np.isnan(ema21[i]) and
                ema9[i] > ema21[i] and i >= pb_depth + 1):
                pb_ok = True
                for j in range(1, pb_depth + 1):
                    if brick_up[i - j]:
                        pb_ok = False
                        break
                if pb_ok and not brick_up[i - pb_depth - 1] == False:
                    # Prior brick before pullback was up
                    if i >= pb_depth + 1 and brick_up[i - pb_depth - 1]:
                        triggered = True
                        last_other = i

        # Flip
        if use_flip and not triggered and flip_arr is not None:
            if up and flip_arr[i] and (i - last_other) >= cooldown:
                triggered = True
                last_other = i

        if triggered:
            entry[i] = True
            in_pos = True

    return entry, exit_


# ── Backtest runner ────────────────────────────────────────────────────────────

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
    }


# ── Worker functions ──────────────────────────────────────────────────────────

def _sweep_solo_entries(label: str) -> list:
    """Sweep solo entry systems + R007 baseline."""
    print(f"  [{label}] Loading data...", flush=True)
    df_ltf = _load_ltf_data()
    df_htf = _load_htf_data()
    gate = _compute_gates(df_ltf, df_htf)
    print(f"  [{label}] Data ready", flush=True)

    keys = list(PARAM_GRID.keys())
    param_combos = [dict(zip(keys, v)) for v in itertools.product(*PARAM_GRID.values())]

    results = []
    done = 0

    # Precompute crossover arrays
    cross_arrays = {ind: _compute_crossover_long(df_ltf, ind) for ind in CROSS_INDICATORS}
    flip_arrays = {ind: _compute_flip_long(df_ltf, ind) for ind in FLIP_INDICATORS}

    configs = []

    # R007 baseline
    for pc in param_combos:
        configs.append(("r007_only", pc, None, None, None))

    # Solo crossovers
    for ind in CROSS_INDICATORS:
        for pc in param_combos:
            configs.append((f"cross_{ind}", pc, ind, None, None))

    # Solo pullbacks
    for depth in PULLBACK_DEPTHS:
        for pc in param_combos:
            configs.append((f"pullback_{depth}", pc, None, depth, None))

    # Solo flips
    for ind in FLIP_INDICATORS:
        for pc in param_combos:
            configs.append((f"flip_{ind}", pc, None, None, ind))

    total = len(configs)

    for entry_name, pc, cross_ind, pb_depth, flip_ind in configs:
        if entry_name == "r007_only":
            e, x = _gen_r007_only(df_ltf, pc["n_bricks"], pc["cooldown"], gate)
        elif entry_name.startswith("cross_"):
            e, x = _gen_cross_only(df_ltf, gate, cross_arrays[cross_ind], pc["cooldown"])
        elif entry_name.startswith("pullback_"):
            e, x = _gen_pullback_only(df_ltf, gate, pb_depth, pc["cooldown"])
        elif entry_name.startswith("flip_"):
            e, x = _gen_flip_only(df_ltf, gate, flip_arrays[flip_ind], pc["cooldown"])

        is_r  = _run_backtest(df_ltf, e, x, IS_START, IS_END)
        oos_r = _run_backtest(df_ltf, e, x, OOS_START, OOS_END)

        is_pf = is_r["pf"]
        oos_pf = oos_r["pf"]
        decay = ((oos_pf - is_pf) / is_pf * 100) \
                if is_pf > 0 and not math.isinf(is_pf) else float("nan")

        results.append({
            "config":     entry_name,
            "mode":       "solo",
            "n_bricks":   pc["n_bricks"],
            "cooldown":   pc["cooldown"],
            "is_pf":      is_pf,
            "is_trades":  is_r["trades"],
            "is_wr":      is_r["wr"],
            "oos_pf":     oos_pf,
            "oos_trades": oos_r["trades"],
            "oos_wr":     oos_r["wr"],
            "oos_net":    oos_r["net"],
            "decay_pct":  decay,
        })

        done += 1
        if done % 48 == 0 or done == total:
            print(f"  [{label}] {done:>3}/{total} | {entry_name:<20} "
                  f"n={pc['n_bricks']} cd={pc['cooldown']:>2} | "
                  f"OOS PF={oos_pf:>7.2f} T={oos_r['trades']:>4}", flush=True)

    print(f"  [{label}] Done -- {len(results)} results", flush=True)
    return results


def _sweep_combined_entries(label: str) -> list:
    """Sweep R007 + each new entry type, and R007 + all three."""
    print(f"  [{label}] Loading data...", flush=True)
    df_ltf = _load_ltf_data()
    df_htf = _load_htf_data()
    gate = _compute_gates(df_ltf, df_htf)
    print(f"  [{label}] Data ready", flush=True)

    keys = list(PARAM_GRID.keys())
    param_combos = [dict(zip(keys, v)) for v in itertools.product(*PARAM_GRID.values())]

    cross_arrays = {ind: _compute_crossover_long(df_ltf, ind) for ind in CROSS_INDICATORS}
    flip_arrays = {ind: _compute_flip_long(df_ltf, ind) for ind in FLIP_INDICATORS}

    results = []
    configs = []

    # R007 + each crossover
    for ind in CROSS_INDICATORS:
        for pc in param_combos:
            configs.append((f"r007+cross_{ind}", pc, ind, None, None))

    # R007 + each pullback depth
    for depth in PULLBACK_DEPTHS:
        for pc in param_combos:
            configs.append((f"r007+pb_{depth}", pc, None, depth, None))

    # R007 + each flip
    for ind in FLIP_INDICATORS:
        for pc in param_combos:
            configs.append((f"r007+flip_{ind}", pc, None, None, ind))

    # R007 + best of each (will use escgo cross + pb2 + psar flip as defaults)
    for cross_ind in CROSS_INDICATORS:
        for depth in PULLBACK_DEPTHS:
            for flip_ind in FLIP_INDICATORS:
                for pc in param_combos:
                    configs.append((f"r007+{cross_ind}+pb{depth}+{flip_ind}", pc,
                                    cross_ind, depth, flip_ind))

    total = len(configs)
    done = 0

    for entry_name, pc, cross_ind, pb_depth, flip_ind in configs:
        e, x = _gen_combined(
            df_ltf, pc["n_bricks"], pc["cooldown"], gate,
            cross_arr=cross_arrays.get(cross_ind),
            pb_depth=pb_depth,
            flip_arr=flip_arrays.get(flip_ind),
            use_r007=True,
            use_cross=cross_ind is not None,
            use_pullback=pb_depth is not None,
            use_flip=flip_ind is not None,
        )

        is_r  = _run_backtest(df_ltf, e, x, IS_START, IS_END)
        oos_r = _run_backtest(df_ltf, e, x, OOS_START, OOS_END)

        is_pf = is_r["pf"]
        oos_pf = oos_r["pf"]
        decay = ((oos_pf - is_pf) / is_pf * 100) \
                if is_pf > 0 and not math.isinf(is_pf) else float("nan")

        results.append({
            "config":     entry_name,
            "mode":       "combined",
            "n_bricks":   pc["n_bricks"],
            "cooldown":   pc["cooldown"],
            "is_pf":      is_pf,
            "is_trades":  is_r["trades"],
            "is_wr":      is_r["wr"],
            "oos_pf":     oos_pf,
            "oos_trades": oos_r["trades"],
            "oos_wr":     oos_r["wr"],
            "oos_net":    oos_r["net"],
            "decay_pct":  decay,
        })

        done += 1
        if done % 72 == 0 or done == total:
            print(f"  [{label}] {done:>4}/{total} | {entry_name:<35} "
                  f"n={pc['n_bricks']} cd={pc['cooldown']:>2} | "
                  f"OOS PF={oos_pf:>7.2f} T={oos_r['trades']:>4}", flush=True)

    print(f"  [{label}] Done -- {len(results)} results", flush=True)
    return results


# ── Summary ────────────────────────────────────────────────────────────────────

def _summarize(all_results):
    print(f"\n{'='*90}")
    print("  BTC Phase 4 -- Hybrid Entry System Sweep (Long Only)")
    print(f"{'='*90}")

    R007_BASELINE_PF = 55.73  # Phase 3 winner OOS PF

    # ── Solo entries: avg by entry type ──
    solo = [r for r in all_results if r["mode"] == "solo"]
    if solo:
        print(f"\n  --- Solo Entry Systems (each tested alone) ---")
        entry_types = sorted(set(r["config"] for r in solo))
        type_avgs = []
        for et in entry_types:
            viable = [r for r in solo if r["config"] == et and r["oos_trades"] >= 10]
            if viable:
                avg_pf = sum(r["oos_pf"] for r in viable if not math.isinf(r["oos_pf"])) / max(
                    len([r for r in viable if not math.isinf(r["oos_pf"])]), 1)
                avg_t = sum(r["oos_trades"] for r in viable) / len(viable)
                best = max(viable, key=lambda r: r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6)
                type_avgs.append((et, avg_pf, avg_t, best, len(viable)))

        type_avgs.sort(key=lambda x: x[1], reverse=True)
        print(f"\n  {'Entry Type':<25} {'Avg PF':>8} {'Avg T':>6} {'N':>3} | "
              f"{'Best PF':>8} {'T':>4} {'WR%':>6}")
        print(f"  {'-'*75}")
        for et, avg_pf, avg_t, best, n in type_avgs:
            print(f"  {et:<25} {avg_pf:>8.2f} {avg_t:>6.1f} {n:>3} | "
                  f"{best['oos_pf']:>8.2f} {best['oos_trades']:>4} "
                  f"{best['oos_wr']:>5.1f}%")

    # ── Combined: avg by combo type ──
    combined = [r for r in all_results if r["mode"] == "combined"]
    if combined:
        print(f"\n  --- Combined with R007 (R007 + new entry) ---")

        # Group by config prefix (r007+cross_X, r007+pb_X, r007+flip_X, r007+all)
        combo_groups = {}
        for r in combined:
            # Simplify: group multi-entry configs
            parts = r["config"].split("+")
            if len(parts) <= 2:
                group = r["config"]
            else:
                group = r["config"]  # keep full name for multi-combos
            combo_groups.setdefault(group, []).append(r)

        # Aggregate by simplified group
        simple_groups = {}
        for r in combined:
            parts = r["config"].split("+")
            if len(parts) == 2:
                simple_groups.setdefault(r["config"], []).append(r)
            else:
                simple_groups.setdefault("r007+multi", []).append(r)

        group_avgs = []
        for gname, gresults in simple_groups.items():
            viable = [r for r in gresults if r["oos_trades"] >= 10]
            if viable:
                avg_pf = sum(r["oos_pf"] for r in viable if not math.isinf(r["oos_pf"])) / max(
                    len([r for r in viable if not math.isinf(r["oos_pf"])]), 1)
                avg_t = sum(r["oos_trades"] for r in viable) / len(viable)
                best = max(viable, key=lambda r: r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6)
                group_avgs.append((gname, avg_pf, avg_t, best, len(viable)))

        group_avgs.sort(key=lambda x: x[1], reverse=True)
        print(f"\n  {'Combo':<30} {'Avg PF':>8} {'Avg T':>6} {'N':>3} | "
              f"{'Best PF':>8} {'T':>4} {'WR%':>6}")
        print(f"  {'-'*75}")
        for gn, avg_pf, avg_t, best, n in group_avgs:
            print(f"  {gn:<30} {avg_pf:>8.2f} {avg_t:>6.1f} {n:>3} | "
                  f"{best['oos_pf']:>8.2f} {best['oos_trades']:>4} "
                  f"{best['oos_wr']:>5.1f}%")

    # ── Overall top 15 ──
    all_viable = [r for r in all_results if r["oos_trades"] >= 10]
    all_viable.sort(key=lambda r: r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6, reverse=True)

    print(f"\n{'='*90}")
    print("  Overall Top 15 (OOS trades >= 10)")
    print(f"{'='*90}")
    print(f"  {'Config':<40} {'n':>2} {'cd':>3} | {'IS PF':>7} {'T':>4} | "
          f"{'OOS PF':>8} {'T':>4} {'WR%':>6} {'Decay':>7}")
    print(f"  {'-'*85}")
    for r in all_viable[:15]:
        dec_s = f"{r['decay_pct']:>+6.1f}%" if not math.isnan(r["decay_pct"]) else "   NaN"
        print(f"  {r['config']:<40} {r['n_bricks']:>2} {r['cooldown']:>3} | "
              f"{r['is_pf']:>7.2f} {r['is_trades']:>4} | "
              f"{r['oos_pf']:>8.2f} {r['oos_trades']:>4} "
              f"{r['oos_wr']:>5.1f}% {dec_s}")

    # ── Top 15 by trade count (more opportunities) ──
    high_trade = [r for r in all_results if r["oos_trades"] >= 30 and r["oos_pf"] >= 5]
    high_trade.sort(key=lambda r: (r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6), reverse=True)

    print(f"\n{'='*90}")
    print("  Top 15 High-Frequency (OOS trades >= 30, PF >= 5)")
    print(f"{'='*90}")
    print(f"  {'Config':<40} {'n':>2} {'cd':>3} | {'IS PF':>7} {'T':>4} | "
          f"{'OOS PF':>8} {'T':>4} {'WR%':>6} {'Decay':>7}")
    print(f"  {'-'*85}")
    for r in high_trade[:15]:
        dec_s = f"{r['decay_pct']:>+6.1f}%" if not math.isnan(r["decay_pct"]) else "   NaN"
        print(f"  {r['config']:<40} {r['n_bricks']:>2} {r['cooldown']:>3} | "
              f"{r['is_pf']:>7.2f} {r['is_trades']:>4} | "
              f"{r['oos_pf']:>8.2f} {r['oos_trades']:>4} "
              f"{r['oos_wr']:>5.1f}% {dec_s}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-parallel", action="store_true")
    args = parser.parse_args()

    out_path = ROOT / "ai_context" / "btc_phase4_results.json"
    out_path.parent.mkdir(exist_ok=True)

    print("BTC Phase 4: Hybrid Entry System Sweep (Long Only)")
    print(f"  Gate stack  : psar_dir + ADX>=30 + vol<=1.5 + HTF ADX>=40")
    print(f"  Crossovers  : {CROSS_INDICATORS}")
    print(f"  Pullbacks   : depth {PULLBACK_DEPTHS}")
    print(f"  Flips       : {FLIP_INDICATORS}")
    print(f"  IS period   : {IS_START} -> {IS_END}")
    print(f"  OOS period  : {OOS_START} -> {OOS_END}")
    print()

    all_results = []

    if args.no_parallel:
        all_results.extend(_sweep_solo_entries("solo"))
        all_results.extend(_sweep_combined_entries("combined"))
    else:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(_sweep_solo_entries, "solo"): "solo",
                pool.submit(_sweep_combined_entries, "combined"): "combined",
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    print(f"  [{name}] finished -- {len(results)} records")
                except Exception as exc:
                    import traceback
                    print(f"  [{name}] FAILED: {exc}")
                    traceback.print_exc()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved {len(all_results)} results -> {out_path}")

    _summarize(all_results)


if __name__ == "__main__":
    main()
