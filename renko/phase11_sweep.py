#!/usr/bin/env python3
"""
phase11_sweep.py -- Three-Workstream Optimization

Stage A: HTF ADX threshold refinement (existing 4 instruments)
         Sweep htf_adx {15,20,25,30,35,40,45} x 2 brick sizes x 12 params
         LTF gates fixed to Phase 8 winner.

Stage B: LTF re-optimization with HTF locked to Stage A winner
         Phase 8-style sweep: 3 P6 x 3 osc x 3 ADX x 3 sess x 12 params
         Goal: find if different LTF stack works better with HTF gate active

Stage C: New instruments (USDJPY, GBPUSD)
  C1: Phase 8-style LTF sweep (5 P6 x 3 osc x 3 ADX x 3 sess x 12 params)
  C2: HTF ADX sweep on C1 winner (7 thresh x 2 bricks x 12 params)

Usage:
  python renko/phase11_sweep.py
  python renko/phase11_sweep.py --stage a
  python renko/phase11_sweep.py --stage b
  python renko/phase11_sweep.py --stage c
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

# ── Shared constants ──────────────────────────────────────────────────────────

PARAM_GRID = {
    "n_bricks": [2, 3, 4, 5],
    "cooldown": [10, 20, 30],
}

VOL_MAX = 1.5
HTF_ADX_THRESHOLDS = [15, 20, 25, 30, 35, 40, 45]
ADX_THRESHOLDS = [20, 25, 30]
SESSION_STARTS = [12, 13, 14]
OSC_CHOICES = [None, "sto_tso", "macd_lc"]

# ── Existing instrument configs (Stages A + B) ──────────────────────────────

EXISTING_INSTRUMENTS = {
    "EURUSD_4": {
        "renko_file":  "OANDA_EURUSD, 1S renko 0.0004.csv",
        "htf_files":   ["OANDA_EURUSD, 1S renko 0.0008.csv",
                        "OANDA_EURUSD, 1S renko 0.0012.csv"],
        "htf_labels":  ["0.0008", "0.0012"],
        "is_start":    "2023-01-23",
        "is_end":      "2025-09-30",
        "oos_start":   "2025-10-01",
        "oos_end":     "2026-03-05",
        "commission":  0.0046,
        "capital":     1000.0,
        "include_mk":  False,
        # Phase 8 winner
        "fixed_sess":  12,
        "fixed_adx":   20,
        "fixed_p6":    "stoch_cross",
        "fixed_osc":   None,
        # Phase 8 sweep gates
        "p6_gates":    ["ema_cross", "ichi_cloud", "stoch_cross"],
        "label":       "EURUSD 0.0004",
    },
    "EURUSD_5": {
        "renko_file":  "OANDA_EURUSD, 1S renko 0.0005.csv",
        "htf_files":   ["OANDA_EURUSD, 1S renko 0.0008.csv",
                        "OANDA_EURUSD, 1S renko 0.0012.csv"],
        "htf_labels":  ["0.0008", "0.0012"],
        "is_start":    "2022-05-18",
        "is_end":      "2025-09-30",
        "oos_start":   "2025-10-01",
        "oos_end":     "2026-03-05",
        "commission":  0.0046,
        "capital":     1000.0,
        "include_mk":  False,
        "fixed_sess":  14,
        "fixed_adx":   30,
        "fixed_p6":    "ichi_cloud",
        "fixed_osc":   "sto_tso",
        "p6_gates":    ["ema_cross", "ichi_cloud", "stoch_cross"],
        "label":       "EURUSD 0.0005",
    },
    "GBPJPY": {
        "renko_file":  "OANDA_GBPJPY, 1S renko 0.05.csv",
        "htf_files":   ["OANDA_GBPJPY, 1S renko 0.1.csv",
                        "OANDA_GBPJPY, 1S renko 0.15.csv"],
        "htf_labels":  ["0.10", "0.15"],
        "is_start":    "2024-11-21",
        "is_end":      "2025-09-30",
        "oos_start":   "2025-10-01",
        "oos_end":     "2026-02-28",
        "commission":  0.005,
        "capital":     150_000.0,
        "include_mk":  True,
        "fixed_sess":  13,
        "fixed_adx":   30,
        "fixed_p6":    "psar_dir",
        "fixed_osc":   "macd_lc",
        "p6_gates":    ["mk_regime", "escgo_cross", "psar_dir"],
        "label":       "GBPJPY 0.05",
    },
    "EURAUD": {
        "renko_file":  "OANDA_EURAUD, 1S renko 0.0006.csv",
        "htf_files":   ["OANDA_EURAUD, 1S renko 0.0012.csv",
                        "OANDA_EURAUD, 1S renko 0.0018.csv"],
        "htf_labels":  ["0.0012", "0.0018"],
        "is_start":    "2023-07-20",
        "is_end":      "2025-09-30",
        "oos_start":   "2025-10-01",
        "oos_end":     "2026-03-17",
        "commission":  0.009,
        "capital":     1000.0,
        "include_mk":  False,
        "fixed_sess":  14,
        "fixed_adx":   30,
        "fixed_p6":    "ichi_cloud",
        "fixed_osc":   "sto_tso",
        "p6_gates":    ["ddl_dir", "ichi_cloud", "escgo_cross"],
        "label":       "EURAUD 0.0006",
    },
}

# ── New instrument configs (Stage C) ─────────────────────────────────────────

NEW_INSTRUMENTS = {
    "USDJPY": {
        "renko_file":  "OANDA_USDJPY, 1S renko 0.05.csv",
        "htf_files":   ["OANDA_USDJPY, 1S renko 0.1.csv",
                        "OANDA_USDJPY, 1S renko 0.15.csv"],
        "htf_labels":  ["0.10", "0.15"],
        "is_end":      "2025-09-30",
        "oos_start":   "2025-10-01",
        "oos_end":     "2026-03-17",
        "commission":  0.005,
        "capital":     1000.0,
        "include_mk":  True,
        "p6_gates":    ["stoch_cross", "ichi_cloud", "psar_dir",
                        "escgo_cross", "mk_regime"],
        "label":       "USDJPY 0.05",
    },
    "GBPUSD": {
        "renko_file":  "OANDA_GBPUSD, 1S renko 0.0004.csv",
        "htf_files":   ["OANDA_GBPUSD, 1S renko 0.0008.csv",
                        "OANDA_GBPUSD, 1S renko 0.0012.csv"],
        "htf_labels":  ["0.0008", "0.0012"],
        "is_end":      "2025-09-30",
        "oos_start":   "2025-10-01",
        "oos_end":     "2026-03-17",
        "commission":  0.005,
        "capital":     1000.0,
        "include_mk":  False,
        "p6_gates":    ["stoch_cross", "ichi_cloud", "psar_dir",
                        "escgo_cross", "ema_cross"],
        "label":       "GBPUSD 0.0004",
    },
}

# ── Phase 10 baselines (for comparison) ──────────────────────────────────────

P10_BASELINES = {
    "EURUSD_4": {"oos_pf": 27.72, "label": "R013 (no HTF improvement)"},
    "EURUSD_5": {"oos_pf": 22.03, "label": "R014 (no HTF improvement)"},
    "GBPJPY":   {"oos_pf": 85.94, "label": "GJ013 htf_adx30@0.10 (TV 71.01)"},
    "EURAUD":   {"oos_pf": 50.59, "label": "EA020 htf_adx30@0.0012 (TV 21.94)"},
}


# ══════════════════════════════════════════════════════════════════════════════
# Shared functions (copied from phase8/10 to avoid modifying stable scripts)
# ══════════════════════════════════════════════════════════════════════════════

def _load_renko_all_indicators(renko_file: str, include_mk: bool) -> pd.DataFrame:
    """Load Renko data + standard + Phase 6 + BC oscillator indicators."""
    sys.path.insert(0, str(ROOT))
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    from renko.phase6_enrichment import add_phase6_indicators
    from indicators.blackcat1402.bc_l1_swing_trade_oscillator import (
        calc_bc_swing_trade_oscillator,
    )
    from indicators.blackcat1402.bc_l1_trend_swing_oscillator import (
        calc_bc_trend_swing_oscillator,
    )
    from indicators.blackcat1402.blackcat_l3_macd_wave_signal_pro import (
        calc_bc_l3_macd_wave_signal_pro,
    )

    df = load_renko_export(renko_file)
    add_renko_indicators(df)
    add_phase6_indicators(df, include_mk=include_mk)

    # STO
    try:
        sto = calc_bc_swing_trade_oscillator(df)
        df["_bc_sto_mf"] = sto["bc_sto_main_force"].shift(1).values
        df["_bc_sto_ll"] = sto["bc_sto_life_line"].shift(1).values
    except Exception as e:
        print(f"  WARN: STO failed: {e}")
        df["_bc_sto_mf"] = np.nan
        df["_bc_sto_ll"] = np.nan

    # TSO
    try:
        tso = calc_bc_trend_swing_oscillator(df)
        df["_bc_tso_pink"] = tso["bc_tso_pink_hist"].shift(1).values
    except Exception as e:
        print(f"  WARN: TSO failed: {e}")
        df["_bc_tso_pink"] = np.nan

    # MACD Wave Signal Pro
    try:
        df_lc = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        macd_result = calc_bc_l3_macd_wave_signal_pro(df_lc)
        df["_bc_macd_state"] = macd_result["bc_macd_state"].shift(1).values
        df["_bc_lc"] = macd_result["bc_lc"].shift(1).values
    except Exception as e:
        print(f"  WARN: MACD Wave Signal Pro failed: {e}")
        df["_bc_macd_state"] = np.nan
        df["_bc_lc"] = np.nan

    return df


def _load_htf_data(htf_file: str) -> pd.DataFrame:
    """Load HTF Renko data + basic indicators."""
    sys.path.insert(0, str(ROOT))
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators

    df = load_renko_export(htf_file)
    add_renko_indicators(df)
    return df


def _compute_htf_adx_gate(df_htf: pd.DataFrame, threshold: int) -> tuple:
    """Compute HTF ADX gate with variable threshold (symmetric, NaN-pass)."""
    adx = df_htf["adx"].values  # already pre-shifted
    adx_nan = np.isnan(adx)
    ok = adx_nan | (adx >= threshold)
    return ok.copy(), ok.copy()


def _align_htf_gate_to_ltf(df_ltf: pd.DataFrame, df_htf: pd.DataFrame,
                           htf_gate_long: np.ndarray,
                           htf_gate_short: np.ndarray) -> tuple:
    """Backward-fill HTF gate onto LTF timestamps via merge_asof."""
    htf_frame = pd.DataFrame({
        "t": df_htf.index.values,
        "gl": htf_gate_long.astype(float),
        "gs": htf_gate_short.astype(float),
    }).sort_values("t")

    ltf_frame = pd.DataFrame({
        "t": df_ltf.index.values,
    }).sort_values("t")

    merged = pd.merge_asof(ltf_frame, htf_frame, on="t", direction="backward")

    gl = merged["gl"].values
    gs = merged["gs"].values
    aligned_long = np.where(np.isnan(gl), True, gl > 0.5)
    aligned_short = np.where(np.isnan(gs), True, gs > 0.5)

    return aligned_long.astype(bool), aligned_short.astype(bool)


def _compute_fixed_ltf_gates(df: pd.DataFrame, config: dict) -> tuple:
    """Compute AND-combined LTF gate for fixed Phase 8 winner config."""
    from renko.phase6_sweep import _compute_gate_arrays as _p6_gate

    n = len(df)
    cl = np.ones(n, dtype=bool)
    cs = np.ones(n, dtype=bool)

    hours = df.index.hour
    cl &= (hours >= config["fixed_sess"])
    cs &= (hours >= config["fixed_sess"])

    vr = df["vol_ratio"].values
    vol_ok = np.isnan(vr) | (vr <= VOL_MAX)
    cl &= vol_ok
    cs &= vol_ok

    adx = df["adx"].values
    adx_ok = np.isnan(adx) | (adx >= config["fixed_adx"])
    cl &= adx_ok
    cs &= adx_ok

    pl, ps = _p6_gate(df, config["fixed_p6"])
    cl &= pl
    cs &= ps

    osc = config["fixed_osc"]
    if osc == "sto_tso":
        sto_mf = df["_bc_sto_mf"].values
        sto_ll = df["_bc_sto_ll"].values
        sto_nan = np.isnan(sto_mf) | np.isnan(sto_ll)
        tso_pink = df["_bc_tso_pink"].values.astype(float)
        tso_nan = np.isnan(tso_pink)
        cl &= ((sto_nan | (sto_mf > sto_ll)) & (tso_nan | (tso_pink > 0.5)))
        cs &= ((sto_nan | (sto_mf < sto_ll)) & (tso_nan | (tso_pink < 0.5)))
    elif osc == "macd_lc":
        macd_st = df["_bc_macd_state"].values
        bc_lc = df["_bc_lc"].values
        ms_nan = np.isnan(macd_st)
        lc_nan = np.isnan(bc_lc)
        ms_int = np.where(ms_nan, -1, macd_st).astype(int)
        cl &= ((ms_nan | np.isin(ms_int, [0, 3])) & (lc_nan | (bc_lc > 0)))
        cs &= ((ms_nan | np.isin(ms_int, [1, 2])) & (lc_nan | (bc_lc < 0)))

    return cl, cs


def _compute_all_ltf_gates(df: pd.DataFrame, p6_gate_names: list) -> dict:
    """Pre-compute all LTF gate arrays (Phase 8-style)."""
    from renko.phase6_sweep import _compute_gate_arrays as _p6_gate

    gates = {}

    hours = df.index.hour
    for ss in SESSION_STARTS:
        ok = hours >= ss
        gates[f"sess_{ss}"] = (ok, ok)

    vr = df["vol_ratio"].values
    vol_ok = np.isnan(vr) | (vr <= VOL_MAX)
    gates["vol"] = (vol_ok, vol_ok)

    adx = df["adx"].values
    adx_nan = np.isnan(adx)
    for at in ADX_THRESHOLDS:
        ok = adx_nan | (adx >= at)
        gates[f"radx_{at}"] = (ok, ok)

    for gname in p6_gate_names:
        gates[f"p6:{gname}"] = _p6_gate(df, gname)

    # STO + TSO
    sto_mf = df["_bc_sto_mf"].values
    sto_ll = df["_bc_sto_ll"].values
    sto_nan = np.isnan(sto_mf) | np.isnan(sto_ll)
    tso_pink = df["_bc_tso_pink"].values.astype(float)
    tso_nan = np.isnan(tso_pink)
    gates["sto_tso"] = (
        (sto_nan | (sto_mf > sto_ll)) & (tso_nan | (tso_pink > 0.5)),
        (sto_nan | (sto_mf < sto_ll)) & (tso_nan | (tso_pink < 0.5)),
    )

    # MACD_LC
    macd_st = df["_bc_macd_state"].values
    bc_lc = df["_bc_lc"].values
    ms_nan = np.isnan(macd_st)
    lc_nan = np.isnan(bc_lc)
    ms_int = np.where(ms_nan, -1, macd_st).astype(int)
    gates["macd_lc"] = (
        (ms_nan | np.isin(ms_int, [0, 3])) & (lc_nan | (bc_lc > 0)),
        (ms_nan | np.isin(ms_int, [1, 2])) & (lc_nan | (bc_lc < 0)),
    )

    return gates


def _combine_gates(gates: dict, sess: int, adx_thresh: int,
                   p6_name: str, osc_name) -> tuple:
    """AND-combine selected gate arrays."""
    n = len(gates["vol"][0])
    cl = np.ones(n, dtype=bool)
    cs = np.ones(n, dtype=bool)

    sl, ss = gates[f"sess_{sess}"]
    cl &= sl; cs &= ss

    vl, vs = gates["vol"]
    cl &= vl; cs &= vs

    al, as_ = gates[f"radx_{adx_thresh}"]
    cl &= al; cs &= as_

    pl, ps = gates[f"p6:{p6_name}"]
    cl &= pl; cs &= ps

    if osc_name is not None:
        ol, os_ = gates[osc_name]
        cl &= ol; cs &= os_

    return cl, cs


def _generate_signals(df, n_bricks, cooldown, gate_long_ok, gate_short_ok):
    """R007 logic with pre-computed gate arrays."""
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
            long_exit[i] = is_opp and trade_dir == 1
            short_exit[i] = is_opp and trade_dir == -1
            if is_opp:
                in_position = False
                trade_dir = 0

        if in_position:
            continue

        prev = brick_up[i - n_bricks : i]
        prev_all_up = bool(np.all(prev))
        prev_all_down = bool(not np.any(prev))

        if prev_all_up and not up:
            cand = -1; is_r002 = True
        elif prev_all_down and up:
            cand = 1; is_r002 = True
        else:
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

        is_long = (cand == 1)

        if is_long and not gate_long_ok[i]:
            continue
        if not is_long and not gate_short_ok[i]:
            continue

        if is_long:
            long_entry[i] = True
        else:
            short_entry[i] = True
        in_position = True
        trade_dir = cand
        if not is_r002:
            last_r001_bar = i

    df["long_entry"] = long_entry
    df["long_exit"] = long_exit
    df["short_entry"] = short_entry
    df["short_exit"] = short_exit
    return df


def _run_backtest(df_sig, start, end, commission, capital):
    sys.path.insert(0, str(ROOT))
    from engine import BacktestConfig, run_backtest_long_short

    cfg = BacktestConfig(
        initial_capital=capital,
        commission_pct=commission,
        slippage_ticks=0,
        qty_type="fixed",
        qty_value=1000.0,
        pyramiding=1,
        start_date=start,
        end_date=end,
        take_profit_pct=0.0,
        stop_loss_pct=0.0,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest_long_short(df_sig, cfg)

    pf = kpis.get("profit_factor", 0.0) or 0.0
    return {
        "pf":     float("inf") if math.isinf(pf) else float(pf),
        "net":    float(kpis.get("net_profit", 0.0) or 0.0),
        "trades": int(kpis.get("total_trades", 0) or 0),
        "wr":     float(kpis.get("win_rate", 0.0) or 0.0),
        "dd":     float(kpis.get("max_drawdown_pct", 0.0) or 0.0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# STAGE A: HTF ADX Threshold Refinement
# ══════════════════════════════════════════════════════════════════════════════

def run_stage_a(name: str, config: dict) -> list:
    """Sweep HTF ADX thresholds on existing instruments with fixed LTF."""
    print(f"[A:{name}] Loading LTF data...", flush=True)
    df_ltf = _load_renko_all_indicators(config["renko_file"], config["include_mk"])
    print(f"[A:{name}] LTF ready -- {len(df_ltf)} bricks", flush=True)

    ltf_long, ltf_short = _compute_fixed_ltf_gates(df_ltf, config)

    keys = list(PARAM_GRID.keys())
    param_combos = [dict(zip(keys, v)) for v in itertools.product(*PARAM_GRID.values())]

    # Load + align HTF data
    htf_data = {}
    for htf_file, htf_label in zip(config["htf_files"], config["htf_labels"]):
        print(f"[A:{name}] Loading HTF {htf_label}...", flush=True)
        df_htf = _load_htf_data(htf_file)
        htf_data[htf_label] = df_htf
        print(f"[A:{name}] HTF {htf_label} -- {len(df_htf)} bricks", flush=True)

    # Sweep: threshold x brick x params
    total = len(HTF_ADX_THRESHOLDS) * len(config["htf_labels"]) * len(param_combos)
    done = 0
    results = []

    for htf_label, df_htf in htf_data.items():
        for thresh in HTF_ADX_THRESHOLDS:
            htf_gl, htf_gs = _compute_htf_adx_gate(df_htf, thresh)
            al, as_ = _align_htf_gate_to_ltf(df_ltf, df_htf, htf_gl, htf_gs)
            combined_long = ltf_long & al
            combined_short = ltf_short & as_

            for pc in param_combos:
                df_sig = _generate_signals(
                    df_ltf.copy(), pc["n_bricks"], pc["cooldown"],
                    combined_long, combined_short,
                )

                is_r = _run_backtest(df_sig, config["is_start"], config["is_end"],
                                     config["commission"], config["capital"])
                oos_r = _run_backtest(df_sig, config["oos_start"], config["oos_end"],
                                      config["commission"], config["capital"])

                is_pf = is_r["pf"]
                oos_pf = oos_r["pf"]
                decay = ((oos_pf - is_pf) / is_pf * 100) \
                         if is_pf > 0 and not math.isinf(is_pf) else float("nan")

                results.append({
                    "stage":        "A",
                    "instrument":   name,
                    "htf_brick":    htf_label,
                    "htf_adx_thresh": thresh,
                    "n_bricks":     pc["n_bricks"],
                    "cooldown":     pc["cooldown"],
                    "is_pf":        is_pf,
                    "is_trades":    is_r["trades"],
                    "is_wr":        is_r["wr"],
                    "oos_pf":       oos_pf,
                    "oos_trades":   oos_r["trades"],
                    "oos_wr":       oos_r["wr"],
                    "decay_pct":    decay,
                })

                done += 1
                if done % 42 == 0 or done == total:
                    print(
                        f"[A:{name}] {done:>4}/{total} | htf={htf_label} adx>={thresh} "
                        f"n={pc['n_bricks']} cd={pc['cooldown']:>2} | "
                        f"OOS PF={oos_pf:>7.2f} T={oos_r['trades']:>4}",
                        flush=True,
                    )

    print(f"[A:{name}] Complete -- {len(results)} results", flush=True)
    return results


def _find_best_htf_config(stage_a_results: list, name: str) -> dict:
    """Find best HTF config from Stage A results for an instrument."""
    inst_res = [r for r in stage_a_results if r["instrument"] == name
                and r["oos_trades"] >= 10]
    if not inst_res:
        return {"htf_brick": None, "htf_adx_thresh": 30}

    # Group by (htf_brick, htf_adx_thresh), compute avg OOS PF
    groups = {}
    for r in inst_res:
        key = (r["htf_brick"], r["htf_adx_thresh"])
        groups.setdefault(key, []).append(r["oos_pf"])

    best_key = max(groups.keys(),
                   key=lambda k: sum(groups[k]) / len(groups[k])
                   if not any(math.isinf(v) for v in groups[k])
                   else sum(min(v, 1e6) for v in groups[k]) / len(groups[k]))

    avg_pf = sum(min(v, 1e6) for v in groups[best_key]) / len(groups[best_key])
    print(f"  [{name}] Best HTF: brick={best_key[0]} adx>={best_key[1]} "
          f"(avg OOS PF={avg_pf:.2f}, N={len(groups[best_key])})", flush=True)

    return {"htf_brick": best_key[0], "htf_adx_thresh": best_key[1]}


# ══════════════════════════════════════════════════════════════════════════════
# STAGE B: LTF Re-optimization with HTF Locked
# ══════════════════════════════════════════════════════════════════════════════

def run_stage_b(name: str, config: dict, htf_best: dict) -> list:
    """Re-sweep LTF gates with HTF ADX locked to Stage A winner."""
    print(f"[B:{name}] Loading LTF data...", flush=True)
    df_ltf = _load_renko_all_indicators(config["renko_file"], config["include_mk"])
    print(f"[B:{name}] LTF ready -- {len(df_ltf)} bricks", flush=True)

    # Pre-compute all LTF gate arrays
    gates = _compute_all_ltf_gates(df_ltf, config["p6_gates"])

    # Load + align best HTF gate
    htf_brick_label = htf_best["htf_brick"]
    htf_adx_thresh = htf_best["htf_adx_thresh"]

    htf_aligned_long = np.ones(len(df_ltf), dtype=bool)
    htf_aligned_short = np.ones(len(df_ltf), dtype=bool)

    if htf_brick_label is not None:
        idx = config["htf_labels"].index(htf_brick_label)
        htf_file = config["htf_files"][idx]
        print(f"[B:{name}] Loading HTF {htf_brick_label} (adx>={htf_adx_thresh})...",
              flush=True)
        df_htf = _load_htf_data(htf_file)
        htf_gl, htf_gs = _compute_htf_adx_gate(df_htf, htf_adx_thresh)
        htf_aligned_long, htf_aligned_short = _align_htf_gate_to_ltf(
            df_ltf, df_htf, htf_gl, htf_gs)
        print(f"[B:{name}] HTF aligned -- {len(df_htf)} bricks", flush=True)

    keys = list(PARAM_GRID.keys())
    param_combos = [dict(zip(keys, v)) for v in itertools.product(*PARAM_GRID.values())]

    sweep_combos = list(itertools.product(
        config["p6_gates"], OSC_CHOICES, ADX_THRESHOLDS, SESSION_STARTS
    ))
    total = len(sweep_combos) * len(param_combos)
    done = 0
    results = []

    for p6_gate, osc, adx_t, sess in sweep_combos:
        gate_long, gate_short = _combine_gates(gates, sess, adx_t, p6_gate, osc)

        # AND with HTF gate
        combined_long = gate_long & htf_aligned_long
        combined_short = gate_short & htf_aligned_short

        osc_label = osc if osc else "none"
        stack_label = f"s{sess}_a{adx_t}_{p6_gate}_{osc_label}"

        for pc in param_combos:
            df_sig = _generate_signals(
                df_ltf.copy(), pc["n_bricks"], pc["cooldown"],
                combined_long, combined_short,
            )

            is_r = _run_backtest(df_sig, config["is_start"], config["is_end"],
                                 config["commission"], config["capital"])
            oos_r = _run_backtest(df_sig, config["oos_start"], config["oos_end"],
                                  config["commission"], config["capital"])

            is_pf = is_r["pf"]
            oos_pf = oos_r["pf"]
            decay = ((oos_pf - is_pf) / is_pf * 100) \
                     if is_pf > 0 and not math.isinf(is_pf) else float("nan")

            results.append({
                "stage":        "B",
                "instrument":   name,
                "stack":        stack_label,
                "htf_brick":    htf_brick_label or "none",
                "htf_adx_thresh": htf_adx_thresh,
                "p6_gate":      p6_gate,
                "osc":          osc_label,
                "adx_thresh":   adx_t,
                "sess_start":   sess,
                "n_bricks":     pc["n_bricks"],
                "cooldown":     pc["cooldown"],
                "is_pf":        is_pf,
                "is_trades":    is_r["trades"],
                "is_wr":        is_r["wr"],
                "oos_pf":       oos_pf,
                "oos_trades":   oos_r["trades"],
                "oos_wr":       oos_r["wr"],
                "decay_pct":    decay,
            })

            done += 1
            if done % 108 == 0 or done == total:
                print(
                    f"[B:{name}] {done:>4}/{total} | {stack_label:<35} "
                    f"n={pc['n_bricks']} cd={pc['cooldown']:>2} | "
                    f"OOS PF={oos_pf:>7.2f} T={oos_r['trades']:>4}",
                    flush=True,
                )

    print(f"[B:{name}] Complete -- {len(results)} results", flush=True)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# STAGE C: New Instruments
# ══════════════════════════════════════════════════════════════════════════════

def run_stage_c(name: str, config: dict) -> list:
    """Full pipeline for new instruments: Phase 8 sweep then HTF sweep."""
    ltf_path = ROOT / "data" / config["renko_file"]
    if not ltf_path.exists():
        print(f"[C:{name}] SKIP -- LTF data not found: {config['renko_file']}", flush=True)
        return []

    print(f"[C:{name}] Loading LTF data...", flush=True)
    df_ltf = _load_renko_all_indicators(config["renko_file"], config["include_mk"])
    print(f"[C:{name}] LTF ready -- {len(df_ltf)} bricks", flush=True)

    # Auto-detect IS start: 200 bars from first bar (warmup)
    first_date = df_ltf.index[200]
    is_start = first_date.strftime("%Y-%m-%d")
    config["is_start"] = is_start
    print(f"[C:{name}] IS range: {is_start} to {config['is_end']}", flush=True)

    # ── C1: Phase 8-style LTF sweep ──
    gates = _compute_all_ltf_gates(df_ltf, config["p6_gates"])

    keys = list(PARAM_GRID.keys())
    param_combos = [dict(zip(keys, v)) for v in itertools.product(*PARAM_GRID.values())]

    sweep_combos = list(itertools.product(
        config["p6_gates"], OSC_CHOICES, ADX_THRESHOLDS, SESSION_STARTS
    ))
    total_c1 = len(sweep_combos) * len(param_combos)
    done = 0
    c1_results = []

    print(f"[C1:{name}] Starting LTF sweep -- {total_c1} runs", flush=True)

    for p6_gate, osc, adx_t, sess in sweep_combos:
        gate_long, gate_short = _combine_gates(gates, sess, adx_t, p6_gate, osc)

        osc_label = osc if osc else "none"
        stack_label = f"s{sess}_a{adx_t}_{p6_gate}_{osc_label}"

        for pc in param_combos:
            df_sig = _generate_signals(
                df_ltf.copy(), pc["n_bricks"], pc["cooldown"],
                gate_long, gate_short,
            )

            is_r = _run_backtest(df_sig, is_start, config["is_end"],
                                 config["commission"], config["capital"])
            oos_r = _run_backtest(df_sig, config["oos_start"], config["oos_end"],
                                  config["commission"], config["capital"])

            is_pf = is_r["pf"]
            oos_pf = oos_r["pf"]
            decay = ((oos_pf - is_pf) / is_pf * 100) \
                     if is_pf > 0 and not math.isinf(is_pf) else float("nan")

            c1_results.append({
                "stage":        "C1",
                "instrument":   name,
                "stack":        stack_label,
                "p6_gate":      p6_gate,
                "osc":          osc_label,
                "adx_thresh":   adx_t,
                "sess_start":   sess,
                "n_bricks":     pc["n_bricks"],
                "cooldown":     pc["cooldown"],
                "is_pf":        is_pf,
                "is_trades":    is_r["trades"],
                "is_wr":        is_r["wr"],
                "oos_pf":       oos_pf,
                "oos_trades":   oos_r["trades"],
                "oos_wr":       oos_r["wr"],
                "decay_pct":    decay,
            })

            done += 1
            if done % 180 == 0 or done == total_c1:
                print(
                    f"[C1:{name}] {done:>4}/{total_c1} | {stack_label:<35} "
                    f"n={pc['n_bricks']} cd={pc['cooldown']:>2} | "
                    f"OOS PF={oos_pf:>7.2f} T={oos_r['trades']:>4}",
                    flush=True,
                )

    # Find C1 winner
    viable = [r for r in c1_results if r["oos_trades"] >= 15]
    if not viable:
        print(f"[C1:{name}] No viable results (OOS trades >= 15)", flush=True)
        return c1_results

    best = max(viable, key=lambda r: r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6)
    print(f"[C1:{name}] Winner: {best['stack']} n={best['n_bricks']} cd={best['cooldown']} "
          f"OOS PF={best['oos_pf']:.2f} T={best['oos_trades']} WR={best['oos_wr']:.1f}%",
          flush=True)

    # Build fixed config from C1 winner
    c1_winner = {
        "fixed_sess": best["sess_start"],
        "fixed_adx":  best["adx_thresh"],
        "fixed_p6":   best["p6_gate"],
        "fixed_osc":  best["osc"] if best["osc"] != "none" else None,
    }

    # ── C2: HTF ADX sweep on C1 winner ──
    ltf_long, ltf_short = _compute_fixed_ltf_gates(df_ltf, {**config, **c1_winner})

    c2_results = []
    htf_available = []
    for htf_file, htf_label in zip(config["htf_files"], config["htf_labels"]):
        htf_path = ROOT / "data" / htf_file
        if htf_path.exists():
            htf_available.append((htf_file, htf_label))
        else:
            print(f"[C2:{name}] SKIP HTF {htf_label} -- data not found", flush=True)

    if not htf_available:
        print(f"[C2:{name}] No HTF data available -- skipping", flush=True)
        return c1_results

    total_c2 = len(HTF_ADX_THRESHOLDS) * len(htf_available) * len(param_combos)
    done = 0

    print(f"[C2:{name}] Starting HTF sweep -- {total_c2} runs", flush=True)

    for htf_file, htf_label in htf_available:
        print(f"[C2:{name}] Loading HTF {htf_label}...", flush=True)
        df_htf = _load_htf_data(htf_file)

        for thresh in HTF_ADX_THRESHOLDS:
            htf_gl, htf_gs = _compute_htf_adx_gate(df_htf, thresh)
            al, as_ = _align_htf_gate_to_ltf(df_ltf, df_htf, htf_gl, htf_gs)
            combined_long = ltf_long & al
            combined_short = ltf_short & as_

            for pc in param_combos:
                df_sig = _generate_signals(
                    df_ltf.copy(), pc["n_bricks"], pc["cooldown"],
                    combined_long, combined_short,
                )

                is_r = _run_backtest(df_sig, is_start, config["is_end"],
                                     config["commission"], config["capital"])
                oos_r = _run_backtest(df_sig, config["oos_start"], config["oos_end"],
                                      config["commission"], config["capital"])

                is_pf = is_r["pf"]
                oos_pf = oos_r["pf"]
                decay = ((oos_pf - is_pf) / is_pf * 100) \
                         if is_pf > 0 and not math.isinf(is_pf) else float("nan")

                c2_results.append({
                    "stage":        "C2",
                    "instrument":   name,
                    "stack":        best["stack"],
                    "htf_brick":    htf_label,
                    "htf_adx_thresh": thresh,
                    "n_bricks":     pc["n_bricks"],
                    "cooldown":     pc["cooldown"],
                    "is_pf":        is_pf,
                    "is_trades":    is_r["trades"],
                    "is_wr":        is_r["wr"],
                    "oos_pf":       oos_pf,
                    "oos_trades":   oos_r["trades"],
                    "oos_wr":       oos_r["wr"],
                    "decay_pct":    decay,
                })

                done += 1
                if done % 42 == 0 or done == total_c2:
                    print(
                        f"[C2:{name}] {done:>4}/{total_c2} | htf={htf_label} adx>={thresh} "
                        f"n={pc['n_bricks']} cd={pc['cooldown']:>2} | "
                        f"OOS PF={oos_pf:>7.2f} T={oos_r['trades']:>4}",
                        flush=True,
                    )

    print(f"[C:{name}] Complete -- {len(c1_results) + len(c2_results)} total results",
          flush=True)
    return c1_results + c2_results


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════

def _summarize_stage_a(results: list) -> None:
    print(f"\n{'='*90}")
    print("  STAGE A: HTF ADX Threshold Refinement")
    print(f"{'='*90}")

    for inst in EXISTING_INSTRUMENTS:
        inst_res = [r for r in results if r["instrument"] == inst]
        if not inst_res:
            continue

        bench = P10_BASELINES[inst]
        cfg = EXISTING_INSTRUMENTS[inst]

        print(f"\n  {cfg['label']} | Phase 10 baseline: {bench['label']}")
        print(f"  {'-'*80}")

        # Group by (htf_brick, htf_adx_thresh) - avg OOS PF
        groups = {}
        for r in inst_res:
            if r["oos_trades"] < 10:
                continue
            key = (r["htf_brick"], r["htf_adx_thresh"])
            groups.setdefault(key, []).append(r)

        print(f"  {'HTF Brick':<10} {'ADX>=':<8} | {'Avg PF':>8} {'Avg T':>7} {'N':>4}")
        sorted_keys = sorted(groups.keys(), key=lambda k: (k[0], k[1]))
        for key in sorted_keys:
            recs = groups[key]
            avg_pf = sum(min(r["oos_pf"], 1e6) for r in recs) / len(recs)
            avg_t = sum(r["oos_trades"] for r in recs) / len(recs)
            marker = " <<" if avg_pf > bench["oos_pf"] else ""
            print(f"  {key[0]:<10} {key[1]:<8} | {avg_pf:>8.2f} {avg_t:>7.1f} {len(recs):>4}{marker}")


def _summarize_stage_b(results: list) -> None:
    print(f"\n{'='*90}")
    print("  STAGE B: LTF Re-optimization with HTF Locked")
    print(f"{'='*90}")

    for inst in EXISTING_INSTRUMENTS:
        inst_res = [r for r in results if r["instrument"] == inst]
        if not inst_res:
            continue

        bench = P10_BASELINES[inst]
        cfg = EXISTING_INSTRUMENTS[inst]

        print(f"\n  {cfg['label']} | HTF: {inst_res[0].get('htf_brick','?')} "
              f"adx>={inst_res[0].get('htf_adx_thresh','?')}")
        print(f"  Phase 10 baseline: {bench['label']} OOS PF {bench['oos_pf']}")
        print(f"  {'-'*80}")

        viable = [r for r in inst_res if r["oos_trades"] >= 15]
        viable.sort(key=lambda r: (r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6),
                    reverse=True)

        # Top 15
        print(f"\n  Top 15 (OOS trades >= 15):")
        print(f"  {'Stack':<35} {'n':>2} {'cd':>3} | {'OOS PF':>7} {'T':>5} {'WR%':>6}")
        print(f"  {'-'*70}")
        for r in viable[:15]:
            beat = " <<BEAT" if r["oos_pf"] > bench["oos_pf"] else ""
            print(f"  {r['stack']:<35} {r['n_bricks']:>2} {r['cooldown']:>3} | "
                  f"{r['oos_pf']:>7.2f} {r['oos_trades']:>5} {r['oos_wr']:>6.1f}%{beat}")

        # By P6 gate
        print(f"\n  By P6 gate (avg OOS PF, viable):")
        for pg in cfg["p6_gates"]:
            pv = [r for r in viable if r["p6_gate"] == pg]
            if pv:
                avg = sum(min(r["oos_pf"], 1e6) for r in pv) / len(pv)
                avg_t = sum(r["oos_trades"] for r in pv) / len(pv)
                print(f"    {pg:<20} avg PF={avg:>7.2f}  avg T={avg_t:>6.1f}  N={len(pv):>3}")

        # By oscillator
        print(f"\n  By oscillator (avg OOS PF, viable):")
        for osc in ["none", "sto_tso", "macd_lc"]:
            ov = [r for r in viable if r["osc"] == osc]
            if ov:
                avg = sum(min(r["oos_pf"], 1e6) for r in ov) / len(ov)
                avg_t = sum(r["oos_trades"] for r in ov) / len(ov)
                print(f"    {osc:<20} avg PF={avg:>7.2f}  avg T={avg_t:>6.1f}  N={len(ov):>3}")

        # By ADX
        print(f"\n  By ADX threshold (avg OOS PF, viable):")
        for at in ADX_THRESHOLDS:
            av = [r for r in viable if r["adx_thresh"] == at]
            if av:
                avg = sum(min(r["oos_pf"], 1e6) for r in av) / len(av)
                avg_t = sum(r["oos_trades"] for r in av) / len(av)
                print(f"    ADX>={at:<3}              avg PF={avg:>7.2f}  avg T={avg_t:>6.1f}  N={len(av):>3}")

        # By session
        print(f"\n  By session start (avg OOS PF, viable):")
        for ss in SESSION_STARTS:
            sv = [r for r in viable if r["sess_start"] == ss]
            if sv:
                avg = sum(min(r["oos_pf"], 1e6) for r in sv) / len(sv)
                avg_t = sum(r["oos_trades"] for r in sv) / len(sv)
                print(f"    sess>={ss}              avg PF={avg:>7.2f}  avg T={avg_t:>6.1f}  N={len(sv):>3}")


def _summarize_stage_c(results: list) -> None:
    print(f"\n{'='*90}")
    print("  STAGE C: New Instruments")
    print(f"{'='*90}")

    for inst in NEW_INSTRUMENTS:
        c1_res = [r for r in results if r["instrument"] == inst and r["stage"] == "C1"]
        c2_res = [r for r in results if r["instrument"] == inst and r["stage"] == "C2"]

        if not c1_res:
            print(f"\n  {NEW_INSTRUMENTS[inst]['label']}: No results (data missing?)")
            continue

        cfg = NEW_INSTRUMENTS[inst]
        print(f"\n  {cfg['label']}")
        print(f"  {'-'*80}")

        # C1 summary
        viable = [r for r in c1_res if r["oos_trades"] >= 15]
        viable.sort(key=lambda r: (r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6),
                    reverse=True)

        print(f"\n  C1: LTF Sweep -- Top 15 (OOS trades >= 15):")
        print(f"  {'Stack':<35} {'n':>2} {'cd':>3} | {'OOS PF':>7} {'T':>5} {'WR%':>6}")
        print(f"  {'-'*70}")
        for r in viable[:15]:
            print(f"  {r['stack']:<35} {r['n_bricks']:>2} {r['cooldown']:>3} | "
                  f"{r['oos_pf']:>7.2f} {r['oos_trades']:>5} {r['oos_wr']:>6.1f}%")

        # C1 by P6 gate
        print(f"\n  C1 by P6 gate (avg OOS PF, viable):")
        for pg in cfg["p6_gates"]:
            pv = [r for r in viable if r["p6_gate"] == pg]
            if pv:
                avg = sum(min(r["oos_pf"], 1e6) for r in pv) / len(pv)
                avg_t = sum(r["oos_trades"] for r in pv) / len(pv)
                print(f"    {pg:<20} avg PF={avg:>7.2f}  avg T={avg_t:>6.1f}  N={len(pv):>3}")

        # C2 summary
        if c2_res:
            c2_viable = [r for r in c2_res if r["oos_trades"] >= 10]
            c2_viable.sort(key=lambda r: (r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6),
                           reverse=True)

            print(f"\n  C2: HTF ADX Sweep -- Top 10:")
            print(f"  {'HTF Brick':<10} {'ADX>=':<8} {'n':>2} {'cd':>3} | {'OOS PF':>7} {'T':>5} {'WR%':>6}")
            print(f"  {'-'*60}")
            for r in c2_viable[:10]:
                print(f"  {r['htf_brick']:<10} {r['htf_adx_thresh']:<8} {r['n_bricks']:>2} {r['cooldown']:>3} | "
                      f"{r['oos_pf']:>7.2f} {r['oos_trades']:>5} {r['oos_wr']:>6.1f}%")

            # By HTF config
            groups = {}
            for r in c2_viable:
                key = (r["htf_brick"], r["htf_adx_thresh"])
                groups.setdefault(key, []).append(r)

            print(f"\n  C2 by HTF config (avg OOS PF):")
            for key in sorted(groups.keys()):
                recs = groups[key]
                avg_pf = sum(min(r["oos_pf"], 1e6) for r in recs) / len(recs)
                avg_t = sum(r["oos_trades"] for r in recs) / len(recs)
                print(f"    htf={key[0]} adx>={key[1]:<3}  avg PF={avg_pf:>7.2f}  avg T={avg_t:>6.1f}  N={len(recs):>3}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["a", "b", "c", "ab", "all"], default="all",
                        help="Which stage(s) to run")
    parser.add_argument("--no-parallel", action="store_true")
    args = parser.parse_args()

    out_path = ROOT / "ai_context" / "phase11_results.json"
    out_path.parent.mkdir(exist_ok=True)

    run_a = args.stage in ("a", "ab", "all")
    run_b = args.stage in ("b", "ab", "all")
    run_c = args.stage in ("c", "all")

    all_results = []

    # ── Stage A ──
    if run_a:
        n_a = len(HTF_ADX_THRESHOLDS) * 2 * 12 * len(EXISTING_INSTRUMENTS)
        print(f"\nStage A: HTF Threshold Refinement -- {n_a} runs")
        print(f"  Thresholds: {HTF_ADX_THRESHOLDS}")
        print()

        stage_a_results = []

        if args.no_parallel:
            for nm, cfg in EXISTING_INSTRUMENTS.items():
                stage_a_results.extend(run_stage_a(nm, cfg))
        else:
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
                futures = {
                    pool.submit(run_stage_a, nm, cfg): nm
                    for nm, cfg in EXISTING_INSTRUMENTS.items()
                }
                for future in as_completed(futures):
                    nm = futures[future]
                    try:
                        res = future.result()
                        stage_a_results.extend(res)
                        print(f"  [A:{nm}] finished -- {len(res)} records")
                    except Exception as exc:
                        import traceback
                        print(f"  [A:{nm}] FAILED: {exc}")
                        traceback.print_exc()

        all_results.extend(stage_a_results)
        _summarize_stage_a(stage_a_results)

        # Find best HTF per instrument for Stage B
        print(f"\n  Stage A winners:")
        htf_winners = {}
        for nm in EXISTING_INSTRUMENTS:
            htf_winners[nm] = _find_best_htf_config(stage_a_results, nm)

    # ── Stage B ──
    if run_b:
        if not run_a:
            # Load Stage A results from file if available
            if out_path.exists():
                with open(out_path) as f:
                    prev = json.load(f)
                stage_a_results = [r for r in prev if r.get("stage") == "A"]
                htf_winners = {}
                for nm in EXISTING_INSTRUMENTS:
                    htf_winners[nm] = _find_best_htf_config(stage_a_results, nm)
            else:
                print("ERROR: Stage A results not found. Run --stage a first.")
                return

        n_b = 81 * 12 * len(EXISTING_INSTRUMENTS)
        print(f"\nStage B: LTF Re-optimization with HTF Locked -- {n_b} runs")
        print()

        stage_b_results = []

        if args.no_parallel:
            for nm, cfg in EXISTING_INSTRUMENTS.items():
                stage_b_results.extend(run_stage_b(nm, cfg, htf_winners[nm]))
        else:
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
                futures = {
                    pool.submit(run_stage_b, nm, cfg, htf_winners[nm]): nm
                    for nm, cfg in EXISTING_INSTRUMENTS.items()
                }
                for future in as_completed(futures):
                    nm = futures[future]
                    try:
                        res = future.result()
                        stage_b_results.extend(res)
                        print(f"  [B:{nm}] finished -- {len(res)} records")
                    except Exception as exc:
                        import traceback
                        print(f"  [B:{nm}] FAILED: {exc}")
                        traceback.print_exc()

        all_results.extend(stage_b_results)
        _summarize_stage_b(stage_b_results)

    # ── Stage C ──
    if run_c:
        print(f"\nStage C: New Instruments (USDJPY, GBPUSD)")
        print()

        stage_c_results = []

        if args.no_parallel:
            for nm, cfg in NEW_INSTRUMENTS.items():
                stage_c_results.extend(run_stage_c(nm, cfg))
        else:
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
                futures = {
                    pool.submit(run_stage_c, nm, cfg): nm
                    for nm, cfg in NEW_INSTRUMENTS.items()
                }
                for future in as_completed(futures):
                    nm = futures[future]
                    try:
                        res = future.result()
                        stage_c_results.extend(res)
                        print(f"  [C:{nm}] finished -- {len(res)} records")
                    except Exception as exc:
                        import traceback
                        print(f"  [C:{nm}] FAILED: {exc}")
                        traceback.print_exc()

        all_results.extend(stage_c_results)
        if stage_c_results:
            _summarize_stage_c(stage_c_results)

    # ── Save ──
    # Merge with existing results if running individual stages
    if out_path.exists() and args.stage not in ("all",):
        with open(out_path) as f:
            prev = json.load(f)
        # Remove results for stages we just ran
        stages_ran = set(r["stage"] for r in all_results)
        prev = [r for r in prev if r.get("stage") not in stages_ran]
        all_results = prev + all_results

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved {len(all_results)} results to {out_path}")

    # ── Overall summary ──
    print(f"\n{'='*90}")
    print("  OVERALL BEST per instrument")
    print(f"{'='*90}")

    for inst in list(EXISTING_INSTRUMENTS.keys()) + list(NEW_INSTRUMENTS.keys()):
        viable = [r for r in all_results
                  if r["instrument"] == inst and r.get("oos_trades", 0) >= 15]
        if not viable:
            continue
        best = max(viable, key=lambda r: r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6)

        label = (EXISTING_INSTRUMENTS.get(inst, {}).get("label") or
                 NEW_INSTRUMENTS.get(inst, {}).get("label", inst))
        baseline_pf = P10_BASELINES.get(inst, {}).get("oos_pf", 0)
        beat = "BEATS P10" if best["oos_pf"] > baseline_pf and baseline_pf > 0 else ""

        extra = ""
        if "stack" in best:
            extra = f"stack={best['stack']} "
        if "htf_brick" in best:
            extra += f"htf={best.get('htf_brick','?')} "
        if "htf_adx_thresh" in best:
            extra += f"adx>={best.get('htf_adx_thresh','?')} "

        print(f"  {label:<16} [{best['stage']}] OOS PF={best['oos_pf']:>7.2f} "
              f"T={best['oos_trades']:>4} WR={best['oos_wr']:>5.1f}% "
              f"| n={best['n_bricks']} cd={best['cooldown']} {extra}"
              f"| {beat}")


if __name__ == "__main__":
    main()
