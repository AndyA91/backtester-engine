#!/usr/bin/env python3
"""
phase10_mtf_sweep.py — Multi-Timeframe Renko Sweep

Uses a larger Renko brick (HTF) as a regime filter for the trading-timeframe
(LTF) strategies.  HTF indicators are computed on the larger bricks, then
backward-filled onto LTF bar timestamps via merge_asof.  The HTF gate is
AND-combined with the fixed Phase 8 winning LTF entry stack.

LTF entry configs are fixed to Phase 8 TV-validated winners per instrument.
Sweep dimensions: HTF brick size × HTF gate type × n_bricks × cooldown.

Usage:
  python renko/phase10_mtf_sweep.py
  python renko/phase10_mtf_sweep.py --no-parallel
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

# ── Instrument configs ──────────────────────────────────────────────────────────

INSTRUMENTS = {
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
        # Phase 8 winner: s12_a20_stoch_cross_none
        "fixed_sess":  12,
        "fixed_adx":   20,
        "fixed_p6":    "stoch_cross",
        "fixed_osc":   None,
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
        # Phase 8 winner: s14_a30_ichi_cloud_sto_tso
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
        # Phase 8 winner: s13_a30_psar_dir_macd_lc
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
        # Phase 8 winner: s14_a30_ichi_cloud_sto_tso
        "fixed_sess":  14,
        "fixed_adx":   30,
        "fixed_p6":    "ichi_cloud",
        "fixed_osc":   "sto_tso",
        "p6_gates":    ["ddl_dir", "ichi_cloud", "escgo_cross"],
        "label":       "EURAUD 0.0006",
    },
}

PARAM_GRID = {
    "n_bricks": [2, 3, 4, 5],
    "cooldown": [10, 20, 30],
}

VOL_MAX = 1.5

HTF_GATE_NAMES = [
    "htf_brick_dir",
    "htf_n2_dir",
    "htf_n3_dir",
    "htf_adx30",
    "htf_ema_cross",
    "htf_psar_dir",
    "htf_macd_hist",
    "htf_stoch_cross",
]


# ── Data loading (reuse Phase 8) ──────────────────────────────────────────────

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
    """Load HTF Renko data + basic indicators (ADX, EMA, PSAR, MACD, Stoch)."""
    sys.path.insert(0, str(ROOT))
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators

    df = load_renko_export(htf_file)
    add_renko_indicators(df)
    return df


# ── HTF gate computation ──────────────────────────────────────────────────────

def _compute_htf_gates(df_htf: pd.DataFrame) -> dict:
    """
    Compute HTF gate arrays on the HTF DataFrame.

    Returns dict[gate_name -> (gate_long, gate_short)] where arrays are
    length = len(df_htf), indexed by HTF bar index.
    All use NaN-pass convention.
    """
    n = len(df_htf)
    brick_up = df_htf["brick_up"].values
    gates = {}

    # 1. htf_brick_dir: last brick direction
    # NaN-pass for first bar (no history)
    long_ok = np.ones(n, dtype=bool)
    short_ok = np.ones(n, dtype=bool)
    long_ok[1:] = brick_up[:-1]      # previous brick was up -> long ok
    short_ok[1:] = ~brick_up[:-1]    # previous brick was down -> short ok
    gates["htf_brick_dir"] = (long_ok.copy(), short_ok.copy())

    # 2. htf_n2_dir: last 2 bricks all same direction
    long_ok = np.ones(n, dtype=bool)
    short_ok = np.ones(n, dtype=bool)
    for i in range(2, n):
        long_ok[i] = brick_up[i-1] and brick_up[i-2]
        short_ok[i] = (not brick_up[i-1]) and (not brick_up[i-2])
    gates["htf_n2_dir"] = (long_ok, short_ok)

    # 3. htf_n3_dir: last 3 bricks all same direction
    long_ok = np.ones(n, dtype=bool)
    short_ok = np.ones(n, dtype=bool)
    for i in range(3, n):
        long_ok[i] = brick_up[i-1] and brick_up[i-2] and brick_up[i-3]
        short_ok[i] = (not brick_up[i-1]) and (not brick_up[i-2]) and (not brick_up[i-3])
    gates["htf_n3_dir"] = (long_ok, short_ok)

    # 4. htf_adx30: ADX >= 30 (symmetric, NaN-pass)
    adx = df_htf["adx"].values  # already pre-shifted
    adx_nan = np.isnan(adx)
    ok = adx_nan | (adx >= 30)
    gates["htf_adx30"] = (ok.copy(), ok.copy())

    # 5. htf_ema_cross: EMA9 > EMA21 (already pre-shifted)
    ema9 = df_htf["ema9"].values
    ema21 = df_htf["ema21"].values
    m = np.isnan(ema9) | np.isnan(ema21)
    gates["htf_ema_cross"] = (m | (ema9 > ema21), m | (ema9 < ema21))

    # 6. htf_psar_dir: PSAR direction (already pre-shifted, +1/-1)
    psar = df_htf["psar_dir"].values
    psar_nan = np.isnan(psar)
    gates["htf_psar_dir"] = (psar_nan | (psar > 0), psar_nan | (psar < 0))

    # 7. htf_macd_hist: MACD histogram direction (already pre-shifted)
    mh = df_htf["macd_hist"].values
    mh_nan = np.isnan(mh)
    gates["htf_macd_hist"] = (mh_nan | (mh >= 0), mh_nan | (mh < 0))

    # 8. htf_stoch_cross: %K > %D (already pre-shifted)
    sk = df_htf["stoch_k"].values
    sd = df_htf["stoch_d"].values
    sm = np.isnan(sk) | np.isnan(sd)
    gates["htf_stoch_cross"] = (sm | (sk > sd), sm | (sk < sd))

    return gates


# ── HTF -> LTF alignment ─────────────────────────────────────────────────────

def _align_htf_gate_to_ltf(df_ltf: pd.DataFrame, df_htf: pd.DataFrame,
                           htf_gate_long: np.ndarray,
                           htf_gate_short: np.ndarray) -> tuple:
    """
    Backward-fill HTF gate arrays onto LTF bar timestamps via merge_asof.

    Returns (ltf_aligned_long, ltf_aligned_short) as numpy bool arrays,
    length = len(df_ltf). NaN entries -> True (NaN-pass).
    """
    # Build HTF frame with timestamps + gate values
    htf_frame = pd.DataFrame({
        "t": df_htf.index.values,
        "gl": htf_gate_long.astype(float),
        "gs": htf_gate_short.astype(float),
    }).sort_values("t")

    ltf_frame = pd.DataFrame({
        "t": df_ltf.index.values,
    }).sort_values("t")

    merged = pd.merge_asof(
        ltf_frame, htf_frame,
        on="t", direction="backward",
    )

    # NaN -> True (NaN-pass for LTF bars before first HTF bar)
    gl = merged["gl"].values
    gs = merged["gs"].values
    aligned_long = np.where(np.isnan(gl), True, gl > 0.5)
    aligned_short = np.where(np.isnan(gs), True, gs > 0.5)

    return aligned_long.astype(bool), aligned_short.astype(bool)


# ── LTF gate computation (fixed Phase 8 winner) ──────────────────────────────

def _compute_fixed_ltf_gates(df: pd.DataFrame, config: dict) -> tuple:
    """
    Compute AND-combined LTF gate for the fixed Phase 8 winner config.

    Returns (combined_long, combined_short) numpy bool arrays.
    """
    from renko.phase6_sweep import _compute_gate_arrays as _p6_gate

    n = len(df)
    cl = np.ones(n, dtype=bool)
    cs = np.ones(n, dtype=bool)

    # Session
    hours = df.index.hour
    sess_ok = hours >= config["fixed_sess"]
    cl &= sess_ok
    cs &= sess_ok

    # Vol ratio (NaN-pass)
    vr = df["vol_ratio"].values
    vr_nan = np.isnan(vr)
    vol_ok = vr_nan | (vr <= VOL_MAX)
    cl &= vol_ok
    cs &= vol_ok

    # Renko ADX (NaN-pass)
    adx = df["adx"].values
    adx_nan = np.isnan(adx)
    adx_ok = adx_nan | (adx >= config["fixed_adx"])
    cl &= adx_ok
    cs &= adx_ok

    # P6 gate
    pl, ps = _p6_gate(df, config["fixed_p6"])
    cl &= pl
    cs &= ps

    # Oscillator
    osc = config["fixed_osc"]
    if osc == "sto_tso":
        sto_mf = df["_bc_sto_mf"].values
        sto_ll = df["_bc_sto_ll"].values
        sto_nan = np.isnan(sto_mf) | np.isnan(sto_ll)
        sto_long = sto_nan | (sto_mf > sto_ll)
        sto_short = sto_nan | (sto_mf < sto_ll)

        tso_pink = df["_bc_tso_pink"].values.astype(float)
        tso_nan = np.isnan(tso_pink)
        tso_long = tso_nan | (tso_pink > 0.5)
        tso_short = tso_nan | (tso_pink < 0.5)

        cl &= (sto_long & tso_long)
        cs &= (sto_short & tso_short)

    elif osc == "macd_lc":
        macd_st = df["_bc_macd_state"].values
        bc_lc = df["_bc_lc"].values
        ms_nan = np.isnan(macd_st)
        lc_nan = np.isnan(bc_lc)
        ms_int = np.where(ms_nan, -1, macd_st).astype(int)
        ms_long = ms_nan | np.isin(ms_int, [0, 3])
        ms_short = ms_nan | np.isin(ms_int, [1, 2])
        lc_long = lc_nan | (bc_lc > 0)
        lc_short = lc_nan | (bc_lc < 0)
        cl &= (ms_long & lc_long)
        cs &= (ms_short & lc_short)

    return cl, cs


# ── Signal generator (reuse Phase 8) ─────────────────────────────────────────

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


# ── Backtest runner ──────────────────────────────────────────────────────────

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


# ── Worker ───────────────────────────────────────────────────────────────────

def run_instrument_sweep(name: str, config: dict) -> list:
    print(f"[{name}] Loading LTF Renko + all indicators...", flush=True)
    df_ltf = _load_renko_all_indicators(config["renko_file"], config["include_mk"])
    print(f"[{name}] LTF ready -- {len(df_ltf)} bricks", flush=True)

    # Compute fixed LTF gates (Phase 8 winner)
    ltf_long, ltf_short = _compute_fixed_ltf_gates(df_ltf, config)

    # Build param combos
    keys = list(PARAM_GRID.keys())
    param_combos = [dict(zip(keys, v)) for v in itertools.product(*PARAM_GRID.values())]

    # Pre-compute all HTF aligned gates
    # htf_aligned[htf_label][gate_name] = (ltf_aligned_long, ltf_aligned_short)
    htf_aligned = {}
    for htf_file, htf_label in zip(config["htf_files"], config["htf_labels"]):
        print(f"[{name}] Loading HTF {htf_label}...", flush=True)
        df_htf = _load_htf_data(htf_file)
        print(f"[{name}] HTF {htf_label} ready -- {len(df_htf)} bricks", flush=True)

        htf_gates = _compute_htf_gates(df_htf)
        htf_aligned[htf_label] = {}
        for gname, (gl, gs) in htf_gates.items():
            al, as_ = _align_htf_gate_to_ltf(df_ltf, df_htf, gl, gs)
            htf_aligned[htf_label][gname] = (al, as_)

    # Build sweep configs: baseline + (htf_brick × htf_gate)
    sweep_configs = [("baseline", None, None)]  # (label, htf_label, gate_name)
    for htf_label in config["htf_labels"]:
        for gname in HTF_GATE_NAMES:
            sweep_configs.append((f"{htf_label}_{gname}", htf_label, gname))

    total = len(sweep_configs) * len(param_combos)
    done = 0
    results = []

    for sweep_label, htf_label, htf_gate_name in sweep_configs:
        # Combine LTF gates + optional HTF gate
        if htf_label is not None:
            htf_gl, htf_gs = htf_aligned[htf_label][htf_gate_name]
            combined_long = ltf_long & htf_gl
            combined_short = ltf_short & htf_gs
        else:
            combined_long = ltf_long
            combined_short = ltf_short

        for pc in param_combos:
            df_sig = _generate_signals(
                df_ltf.copy(),
                n_bricks=pc["n_bricks"],
                cooldown=pc["cooldown"],
                gate_long_ok=combined_long,
                gate_short_ok=combined_short,
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
                "instrument":   name,
                "htf_config":   sweep_label,
                "htf_brick":    htf_label or "none",
                "htf_gate":     htf_gate_name or "none",
                "n_bricks":     pc["n_bricks"],
                "cooldown":     pc["cooldown"],
                "is_pf":        is_pf,
                "is_trades":    is_r["trades"],
                "is_net":       is_r["net"],
                "is_wr":        is_r["wr"],
                "oos_pf":       oos_pf,
                "oos_trades":   oos_r["trades"],
                "oos_net":      oos_r["net"],
                "oos_wr":       oos_r["wr"],
                "decay_pct":    decay,
            })

            done += 1
            if done % 51 == 0 or done == total:
                print(
                    f"[{name}] {done:>4}/{total} | {sweep_label:<30} "
                    f"n={pc['n_bricks']} cd={pc['cooldown']:>2} | "
                    f"IS PF={is_pf:>6.2f} T={is_r['trades']:>4} | "
                    f"OOS PF={oos_pf:>6.2f} T={oos_r['trades']:>4}",
                    flush=True,
                )

    print(f"[{name}] Complete -- {len(results)} results", flush=True)
    return results


# ── Summary ──────────────────────────────────────────────────────────────────

P8_BASELINES = {
    "EURUSD_4": {"oos_pf": 27.72, "label": "R013 (stoch_cross)"},
    "EURUSD_5": {"oos_pf": 22.03, "label": "R014 (ichi+sto_tso)"},
    "GBPJPY":   {"oos_pf": 38.01, "label": "GJ012 (psar+macd_lc)"},
    "EURAUD":   {"oos_pf": 18.32, "label": "EA019 (ichi+sto_tso)"},
}


def _summarize(all_results: list) -> None:
    for inst in ["EURUSD_4", "EURUSD_5", "GBPJPY", "EURAUD"]:
        inst_res = [r for r in all_results if r["instrument"] == inst]
        if not inst_res:
            continue

        bench = P8_BASELINES[inst]
        cfg = INSTRUMENTS[inst]

        print(f"\n{'='*90}")
        print(f"  {cfg['label']}")
        print(f"  Phase 8 baseline: {bench['label']} OOS PF {bench['oos_pf']}")
        print(f"{'='*90}")

        viable = [r for r in inst_res if r["oos_trades"] >= 15]
        viable.sort(key=lambda r: (r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6),
                    reverse=True)

        # Top 20
        print(f"\n  Top 20 (OOS trades >= 15):")
        print(f"  {'HTF Config':<30} {'n':>2} {'cd':>3} | {'IS PF':>7} {'T':>5} | "
              f"{'OOS PF':>7} {'T':>5} {'WR%':>6} {'Decay':>7}")
        print(f"  {'-'*85}")
        for r in viable[:20]:
            beat = " <<BEAT" if r["oos_pf"] > bench["oos_pf"] else ""
            dec_s = f"{r['decay_pct']:>+6.1f}%" if not math.isnan(r["decay_pct"]) else "   NaN"
            print(f"  {r['htf_config']:<30} {r['n_bricks']:>2} {r['cooldown']:>3} | "
                  f"{r['is_pf']:>7.2f} {r['is_trades']:>5} | "
                  f"{r['oos_pf']:>7.2f} {r['oos_trades']:>5} {r['oos_wr']:>6.1f}% "
                  f"{dec_s}{beat}")

        # Average by HTF gate type
        print(f"\n  By HTF gate (avg OOS PF, viable):")
        for gname in ["none"] + HTF_GATE_NAMES:
            gv = [r for r in viable if r["htf_gate"] == gname]
            if gv:
                avg = sum(r["oos_pf"] for r in gv) / len(gv)
                avg_t = sum(r["oos_trades"] for r in gv) / len(gv)
                print(f"    {gname:<20} avg PF={avg:>7.2f}  avg T={avg_t:>6.1f}  N={len(gv):>3}")

        # Average by HTF brick size
        print(f"\n  By HTF brick size (avg OOS PF, viable):")
        for blabel in ["none"] + cfg["htf_labels"]:
            bv = [r for r in viable if r["htf_brick"] == blabel]
            if bv:
                avg = sum(r["oos_pf"] for r in bv) / len(bv)
                avg_t = sum(r["oos_trades"] for r in bv) / len(bv)
                print(f"    {blabel:<20} avg PF={avg:>7.2f}  avg T={avg_t:>6.1f}  N={len(bv):>3}")

        # Count how many beat baseline
        baseline_res = [r for r in inst_res if r["htf_gate"] == "none"]
        baseline_by_params = {}
        for r in baseline_res:
            baseline_by_params[(r["n_bricks"], r["cooldown"])] = r["oos_pf"]

        beat_count = 0
        total_non_baseline = 0
        for r in inst_res:
            if r["htf_gate"] != "none":
                total_non_baseline += 1
                bl_pf = baseline_by_params.get((r["n_bricks"], r["cooldown"]), 0)
                if r["oos_pf"] > bl_pf:
                    beat_count += 1
        pct = beat_count / total_non_baseline * 100 if total_non_baseline > 0 else 0
        print(f"\n  Beat baseline: {beat_count}/{total_non_baseline} ({pct:.0f}%)")

    # Overall best
    print(f"\n{'='*90}")
    print("  Overall best per instrument (single best OOS PF, trades >= 15)")
    print(f"{'='*90}")
    for inst in ["EURUSD_4", "EURUSD_5", "GBPJPY", "EURAUD"]:
        viable = [r for r in all_results
                  if r["instrument"] == inst and r["oos_trades"] >= 15]
        if not viable:
            continue
        best = max(viable, key=lambda r: r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6)
        bench = P8_BASELINES[inst]
        beat = "BEATS P8" if best["oos_pf"] > bench["oos_pf"] else ""
        print(f"  {INSTRUMENTS[inst]['label']:<16} OOS PF={best['oos_pf']:>7.2f} "
              f"T={best['oos_trades']:>4} WR={best['oos_wr']:>5.1f}% "
              f"| {best['htf_config']} n={best['n_bricks']} cd={best['cooldown']} "
              f"| {beat}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-parallel", action="store_true")
    args = parser.parse_args()

    out_path = ROOT / "ai_context" / "phase10_results.json"
    out_path.parent.mkdir(exist_ok=True)

    n_params = len(list(itertools.product(*PARAM_GRID.values())))
    n_htf_configs = 1 + 2 * len(HTF_GATE_NAMES)  # baseline + 2 bricks × 8 gates

    print("Phase 10: Multi-Timeframe Renko Sweep")
    print(f"  HTF gates      : {HTF_GATE_NAMES}")
    print(f"  HTF configs    : {n_htf_configs} (1 baseline + {2 * len(HTF_GATE_NAMES)} HTF)")
    print(f"  Param combos   : {n_params}")
    print()
    for nm, cfg in INSTRUMENTS.items():
        total = n_htf_configs * n_params
        print(f"  {cfg['label']:<16} HTF bricks: {cfg['htf_labels']}  "
              f"-> {total} runs")
    total_all = n_htf_configs * n_params * len(INSTRUMENTS)
    print(f"\n  Total runs     : {total_all} ({total_all * 2} IS+OOS backtests)")
    print(f"  Output         : {out_path}")
    print()

    all_results: list = []

    if args.no_parallel:
        for nm, config in INSTRUMENTS.items():
            all_results.extend(run_instrument_sweep(nm, config))
    else:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(run_instrument_sweep, nm, config): nm
                for nm, config in INSTRUMENTS.items()
            }
            for future in as_completed(futures):
                nm = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    print(f"  [{nm}] finished -- {len(results)} records")
                except Exception as exc:
                    import traceback
                    print(f"  [{nm}] FAILED: {exc}")
                    traceback.print_exc()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved {len(all_results)} results -> {out_path}")

    _summarize(all_results)


if __name__ == "__main__":
    main()
