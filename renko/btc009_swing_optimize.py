#!/usr/bin/env python3
"""
btc009_swing_optimize.py — BTC Swing Bounce Optimization (Long Only)

Optimizes the swing support bounce strategy found in btc_sr_pivot_sweep.py.
Baseline: swBnc_zz10_p1.0_cd5 + PSAR = PF=20.09, 150t (0.9/d), WR=71.3%

Three stratified phases:
  Phase A — Core signal tuning: zigzag threshold × proximity × cooldown (PSAR only)
            -> find best zz/prox/cd combo
  Phase B — Gate optimization: add chop/ADX/RSI/stoch gates on Phase A best
            -> find best gate stack
  Phase C — Alternative swing detection: pivot-based (pv3/pv5/pv8) with Phase B best gates
            -> compare zigzag vs pivot-based

Target: 1+ trade/day with best possible WR and PF.

Usage:
    python renko/btc009_swing_optimize.py
"""

import contextlib
import io
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd  # noqa: F401 — used by load_renko_export internals

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
    df = load_renko_export(LTF_FILE)
    add_renko_indicators(df)
    add_phase6_indicators(df)
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
# S/R enrichment — add swing levels to the DataFrame
# ==============================================================================

def _online_zigzag_support(high, low, pct_threshold):
    """Compute zigzag support levels as they would be detected in real-time.

    Unlike calc_zigzag which marks swings at the actual swing bar, this
    function updates the support level only at the DETECTION bar (when
    the reversal threshold is first exceeded). This eliminates look-ahead.

    Returns pre-shifted array (value at i = support known through bar i-1).
    """
    n = len(high)
    threshold = pct_threshold / 100.0
    support = np.full(n, np.nan)

    direction = 0
    last_high = high[0]
    last_low = low[0]
    cur_sl = np.nan

    for i in range(1, n):
        if direction == 0:
            if high[i] > last_high:
                last_high = high[i]
                direction = 1
            elif low[i] < last_low:
                last_low = low[i]
                direction = -1
        elif direction == 1:
            if high[i] > last_high:
                last_high = high[i]
            elif last_high > 0 and (last_high - low[i]) / last_high >= threshold:
                # Swing HIGH confirmed -> start tracking new low
                last_low = low[i]
                direction = -1
        elif direction == -1:
            if low[i] < last_low:
                last_low = low[i]
            elif last_low > 0 and (high[i] - last_low) / last_low >= threshold:
                # Swing LOW confirmed NOW at bar i (not at the actual low bar)
                cur_sl = last_low
                last_high = high[i]
                direction = 1

        # Pre-shift: value at i+1 = what's known through bar i
        if i + 1 < n:
            support[i + 1] = cur_sl

    return support


def _enrich_sr(df):
    """Add zigzag swing levels and pivot-based swing levels.

    IMPORTANT: Both methods use online/causal detection to prevent look-ahead:
      - Zigzag: support level updated at the DETECTION bar (reversal threshold
        exceeded), not at the actual swing bar.
      - Pivot: shifted by (right + 1) bars — right bars for confirmation delay
        plus 1 for pre-shift convention.
    """
    from indicators.zigzag import calc_swing_points

    n = len(df)
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)

    # Zigzag swing levels — online detection (no look-ahead)
    for pct in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        tag = f"zz{str(pct).replace('.', '')}"
        df[f"{tag}_sl"] = _online_zigzag_support(high, low, pct)

    # Pivot-based swing lows — shifted by (right + 1) to match Pine's ta.pivotlow
    for lr in [3, 5, 8]:
        sp = calc_swing_points(df, left=lr, right=lr)
        tag = f"pv{lr}"

        last_pl = np.full(n, np.nan)
        cur_pl = np.nan
        for i in range(n):
            if sp["pivot_low"][i]:
                cur_pl = sp["pl_price"][i]
            last_pl[i] = cur_pl

        # Shift by (right + 1): right bars for confirmation + 1 for pre-shift
        shift = lr + 1
        df[f"{tag}_pl"] = np.roll(last_pl, shift)
        for j in range(min(shift, n)):
            df.iloc[j, df.columns.get_loc(f"{tag}_pl")] = np.nan

    return df


# ==============================================================================
# Signal generator — swing bounce with configurable gates
# ==============================================================================

def _gen_swing_bounce(brick_up, low, sl, proximity_pct, cooldown,
                      psar_gate, chop, adx, rsi, stoch_k,
                      chop_max, adx_min, rsi_floor, stoch_floor):
    """Swing bounce entry with optional gates.

    Gates (0 = disabled):
        chop_max:    Choppiness Index must be <= this (skip choppy markets)
        adx_min:     ADX must be >= this (require trending market)
        rsi_floor:   RSI must be >= this (skip deeply oversold traps)
        stoch_floor: Stoch %K must be >= this (skip deeply oversold)
    """
    n = len(brick_up)
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

        if not up or (i - last_bar) < cooldown:
            continue

        # PSAR gate (always on)
        if not psar_gate[i]:
            continue

        # Swing level proximity check
        if np.isnan(sl[i]):
            continue
        dist_pct = abs(low[i] - sl[i]) / sl[i] * 100 if sl[i] > 0 else 999
        if dist_pct > proximity_pct:
            continue

        # Optional gates
        if chop_max > 0 and not np.isnan(chop[i]) and chop[i] > chop_max:
            continue
        if adx_min > 0 and not np.isnan(adx[i]) and adx[i] < adx_min:
            continue
        if rsi_floor > 0 and not np.isnan(rsi[i]) and rsi[i] < rsi_floor:
            continue
        if stoch_floor > 0 and not np.isnan(stoch_k[i]) and stoch_k[i] < stoch_floor:
            continue

        entry[i] = True
        in_pos = True
        last_bar = i

    return entry, exit_


# ==============================================================================
# Combo builders
# ==============================================================================

def _build_phase_a():
    """Phase A: Core signal tuning — zz × proximity × cooldown, PSAR only."""
    combos = []
    for zz_pct in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        tag = f"zz{str(zz_pct).replace('.', '')}"
        for prox in [0.3, 0.5, 0.75, 1.0, 1.25, 1.5]:
            for cd in [2, 3, 4, 5, 7]:
                combos.append({
                    "phase": "A",
                    "sl_col": f"{tag}_sl",
                    "zz_pct": zz_pct,
                    "proximity": prox,
                    "cooldown": cd,
                    "chop_max": 0,
                    "adx_min": 0,
                    "rsi_floor": 0,
                    "stoch_floor": 0,
                    "label": f"zz{zz_pct}_p{prox}_cd{cd}",
                })
    return combos


def _build_phase_b(best_sl_col, best_prox, best_cd, best_zz_pct):
    """Phase B: Gate optimization on Phase A best."""
    combos = []
    for chop_max in [0, 50, 60, 70]:
        for adx_min in [0, 20, 25]:
            for rsi_floor in [0, 35, 45]:
                for stoch_floor in [0, 25]:
                    # Skip all-off (already tested in Phase A)
                    if chop_max == 0 and adx_min == 0 and rsi_floor == 0 and stoch_floor == 0:
                        continue
                    gates = []
                    if chop_max: gates.append(f"ch{chop_max}")
                    if adx_min: gates.append(f"a{adx_min}")
                    if rsi_floor: gates.append(f"r{rsi_floor}")
                    if stoch_floor: gates.append(f"s{stoch_floor}")
                    combos.append({
                        "phase": "B",
                        "sl_col": best_sl_col,
                        "zz_pct": best_zz_pct,
                        "proximity": best_prox,
                        "cooldown": best_cd,
                        "chop_max": chop_max,
                        "adx_min": adx_min,
                        "rsi_floor": rsi_floor,
                        "stoch_floor": stoch_floor,
                        "label": f"best+{'+'.join(gates)}",
                    })
    return combos


def _build_phase_c(best_gates):
    """Phase C: Pivot-based alternatives with best gates."""
    combos = []
    for lr in [3, 5, 8]:
        for prox in [0.3, 0.5, 0.75, 1.0, 1.25, 1.5]:
            for cd in [3, 4, 5, 7]:
                combos.append({
                    "phase": "C",
                    "sl_col": f"pv{lr}_pl",
                    "zz_pct": 0,
                    "proximity": prox,
                    "cooldown": cd,
                    "chop_max": best_gates["chop_max"],
                    "adx_min": best_gates["adx_min"],
                    "rsi_floor": best_gates["rsi_floor"],
                    "stoch_floor": best_gates["stoch_floor"],
                    "label": f"pv{lr}_p{prox}_cd{cd}",
                })
    return combos


# ==============================================================================
# Worker
# ==============================================================================

_w = {}

def _init_worker():
    if "df" not in _w:
        df = _load_ltf()
        _enrich_sr(df)
        _w["df"] = df
        # Pre-extract arrays
        psar = df["psar_dir"].values
        _w["psar_gate"] = np.isnan(psar) | (psar > 0)
        _w["brick_up"] = df["brick_up"].values
        _w["low"] = df["Low"].values.astype(float)
        _w["chop"] = df["chop"].values
        _w["adx"] = df["adx"].values
        _w["rsi"] = df["rsi"].values
        _w["stoch_k"] = df["stoch_k"].values


def _run_one(combo):
    _init_worker()
    df = _w["df"]
    w = _w

    sl = df[combo["sl_col"]].values

    ent, ext = _gen_swing_bounce(
        w["brick_up"], w["low"], sl,
        combo["proximity"], combo["cooldown"],
        w["psar_gate"], w["chop"], w["adx"], w["rsi"], w["stoch_k"],
        combo["chop_max"], combo["adx_min"],
        combo["rsi_floor"], combo["stoch_floor"],
    )

    is_r = _run_bt(df, ent, ext, IS_START, IS_END)
    oos_r = _run_bt(df, ent, ext, OOS_START, OOS_END)
    return combo, is_r, oos_r


# ==============================================================================
# Reporting
# ==============================================================================

def _print_header():
    print(f"  {'#':>3} {'Ph':>2} {'Label':<40} | "
          f"{'IS PF':>7} {'T':>5} {'WR%':>6} | "
          f"{'OOS PF':>8} {'T':>5} {'t/d':>5} {'WR%':>6} {'Net':>9} {'DD%':>7}")
    print(f"  {'-'*120}")


def _print_row(r, rank=None):
    pf_i = "INF" if math.isinf(r["is_pf"]) else f"{r['is_pf']:.2f}"
    pf_o = "INF" if math.isinf(r["oos_pf"]) else f"{r['oos_pf']:.2f}"
    tpd = r["oos_trades"] / OOS_DAYS if r["oos_trades"] > 0 else 0
    prefix = f"  {rank:>3}" if rank else "  "
    print(f"{prefix} {r['phase']:>2} {r['label']:<40} | "
          f"{pf_i:>7} {r['is_trades']:>5} {r['is_wr']:>5.1f}% | "
          f"{pf_o:>8} {r['oos_trades']:>5} {tpd:>4.1f} {r['oos_wr']:>5.1f}% "
          f"{r['oos_net']:>9.2f} {r['oos_dd']:>6.2f}%")


def _run_phase(combos, phase_name, all_results):
    total = len(combos)
    print(f"\n  Running Phase {phase_name}: {total} combos ({total*2} backtests)...")

    done = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run_one, c): c for c in combos}
        for fut in as_completed(futures):
            try:
                combo, is_r, oos_r = fut.result()
                row = {
                    "phase":      combo["phase"],
                    "label":      combo["label"],
                    "combo":      combo,
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
                all_results.append(row)
            except Exception as e:
                import traceback
                print(f"  ERROR: {combo.get('label', '???')}: {e}")
                traceback.print_exc()

            done += 1
            if done % 30 == 0 or done == total:
                print(f"    [{done:>4}/{total}]", flush=True)


def _show_results(results, phase, title, min_trades=30):
    subset = [r for r in results if r["phase"] == phase]
    viable = [r for r in subset if r["oos_trades"] >= min_trades and r["oos_net"] > 0]
    by_wr = sorted(viable, key=lambda r: (r["oos_wr"], r["oos_net"]), reverse=True)

    print(f"\n{'='*130}")
    print(f"  {title} — {len(viable)} viable / {len(subset)} total (T>={min_trades}, net>0)")
    print(f"{'='*130}")

    if by_wr:
        print(f"\n  Top 15 by WR:")
        _print_header()
        for i, r in enumerate(by_wr[:15]):
            _print_row(r, rank=i+1)

    # Also show top by net for frequency seekers
    by_net = sorted(viable, key=lambda r: r["oos_net"], reverse=True)
    if by_net:
        print(f"\n  Top 10 by Net:")
        _print_header()
        for i, r in enumerate(by_net[:10]):
            _print_row(r, rank=i+1)

    # Best at 1+/day
    freq = [r for r in viable if r["oos_trades"] >= OOS_DAYS]
    if freq:
        freq.sort(key=lambda r: (r["oos_wr"], r["oos_net"]), reverse=True)
        print(f"\n  Top 10 at 1+/day (T>={OOS_DAYS}):")
        _print_header()
        for i, r in enumerate(freq[:10]):
            _print_row(r, rank=i+1)

    return by_wr[0] if by_wr else None


def _pick_best_a(results):
    """Pick best Phase A config. Prefer 1+/day with highest WR, fallback to best WR."""
    viable_a = [r for r in results if r["phase"] == "A"
                and r["oos_trades"] >= 30 and r["oos_net"] > 0]

    # First: best at 1+/day
    freq = [r for r in viable_a if r["oos_trades"] >= OOS_DAYS]
    if freq:
        freq.sort(key=lambda r: (r["oos_wr"], r["oos_net"]), reverse=True)
        best = freq[0]
    else:
        viable_a.sort(key=lambda r: (r["oos_wr"], r["oos_net"]), reverse=True)
        best = viable_a[0]

    c = best["combo"]
    return c["sl_col"], c["proximity"], c["cooldown"], c["zz_pct"]


def _pick_best_b(results):
    """Pick best Phase B gates. Prefer highest WR at 1+/day."""
    viable_b = [r for r in results if r["phase"] == "B"
                and r["oos_trades"] >= 30 and r["oos_net"] > 0]

    if not viable_b:
        return {"chop_max": 0, "adx_min": 0, "rsi_floor": 0, "stoch_floor": 0}

    # Prefer 1+/day
    freq = [r for r in viable_b if r["oos_trades"] >= OOS_DAYS]
    pool = freq if freq else viable_b
    pool.sort(key=lambda r: (r["oos_wr"], r["oos_net"]), reverse=True)
    best = pool[0]

    # If best gate config is worse than no-gate Phase A best, return no gates
    best_a = [r for r in results if r["phase"] == "A"
              and r["oos_trades"] >= 30 and r["oos_net"] > 0]
    if best_a:
        best_a.sort(key=lambda r: (r["oos_wr"], r["oos_net"]), reverse=True)
        # Only use gates if they improve WR
        if best["oos_wr"] <= best_a[0]["oos_wr"] and best["oos_net"] <= best_a[0]["oos_net"]:
            print(f"  (Gates didn't improve on Phase A best — using no gates)")
            return {"chop_max": 0, "adx_min": 0, "rsi_floor": 0, "stoch_floor": 0}

    return {
        "chop_max": best["combo"]["chop_max"],
        "adx_min": best["combo"]["adx_min"],
        "rsi_floor": best["combo"]["rsi_floor"],
        "stoch_floor": best["combo"]["stoch_floor"],
    }


# ==============================================================================
# Main
# ==============================================================================

def main():
    print(f"\n{'='*70}")
    print(f"BTC009 Swing Bounce Optimization")
    print(f"  Baseline: swBnc_zz10_p1.0_cd5 + PSAR")
    print(f"    OOS: PF=20.09, 150t (0.9/d), WR=71.3%")
    print(f"  Workers: {MAX_WORKERS}")
    print(f"{'='*70}")

    all_results = []

    # -- Phase A ---------------------------------------------------------------
    combos_a = _build_phase_a()
    print(f"\n  Phase A: {len(combos_a)} combos — Core tuning (zz × prox × cd)")
    _run_phase(combos_a, "A", all_results)
    _show_results(all_results, "A", "Phase A — Core Signal Tuning")

    best_sl, best_prox, best_cd, best_zz = _pick_best_a(all_results)
    print(f"\n  -> Phase A best: sl_col={best_sl}, proximity={best_prox}, cd={best_cd}")

    # -- Phase B ---------------------------------------------------------------
    combos_b = _build_phase_b(best_sl, best_prox, best_cd, best_zz)
    print(f"\n  Phase B: {len(combos_b)} combos — Gate optimization")
    _run_phase(combos_b, "B", all_results)
    _show_results(all_results, "B", "Phase B — Gate Optimization")

    best_gates = _pick_best_b(all_results)
    gates_str = ", ".join(f"{k}={v}" for k, v in best_gates.items() if v > 0) or "none"
    print(f"\n  -> Phase B best gates: {gates_str}")

    # -- Phase C ---------------------------------------------------------------
    combos_c = _build_phase_c(best_gates)
    print(f"\n  Phase C: {len(combos_c)} combos — Pivot-based alternatives")
    _run_phase(combos_c, "C", all_results)
    _show_results(all_results, "C", "Phase C — Pivot-Based Swing Detection")

    # -- Global summary --------------------------------------------------------
    total = len(combos_a) + len(combos_b) + len(combos_c)
    viable_all = [r for r in all_results
                  if r["oos_trades"] >= 30 and r["oos_net"] > 0]
    viable_all.sort(key=lambda r: (r["oos_wr"], r["oos_net"]), reverse=True)

    print(f"\n{'='*130}")
    print(f"  GLOBAL TOP 20 by WR (T>=30, net>0): {len(viable_all)} / {total}")
    print(f"{'='*130}")
    _print_header()
    for i, r in enumerate(viable_all[:20]):
        _print_row(r, rank=i+1)

    # Best at 1+/day
    freq_all = [r for r in viable_all if r["oos_trades"] >= OOS_DAYS]
    if freq_all:
        freq_all.sort(key=lambda r: (r["oos_wr"], r["oos_net"]), reverse=True)
        print(f"\n  GLOBAL TOP 10 at 1+/day:")
        _print_header()
        for i, r in enumerate(freq_all[:10]):
            _print_row(r, rank=i+1)

    # IS/OOS consistency check
    consistent = [r for r in all_results
                  if r["oos_trades"] >= 30 and r["oos_net"] > 0
                  and r["is_trades"] >= 60 and r["is_net"] > 0]
    consistent.sort(key=lambda r: (r["oos_wr"], r["oos_net"]), reverse=True)

    print(f"\n  CONSISTENT IS/OOS (OOS T>=30 net>0, IS T>=60 net>0): {len(consistent)}")
    _print_header()
    for i, r in enumerate(consistent[:15]):
        _print_row(r, rank=i+1)

    # -- Save ------------------------------------------------------------------
    out_path = ROOT / "ai_context" / "btc009_optimize_results.json"
    out_path.parent.mkdir(exist_ok=True)
    serializable = []
    for r in all_results:
        sr = {k: v for k, v in r.items() if k != "combo"}
        sr["combo"] = r["combo"]
        for k in ("is_pf", "oos_pf"):
            if math.isinf(sr[k]):
                sr[k] = "inf"
        serializable.append(sr)
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved {len(all_results)} results -> {out_path}")
    print(f"Total combos: {total} ({total*2} backtests)")


if __name__ == "__main__":
    main()
