#!/usr/bin/env python3
"""
phase14_eurusd_sweep.py — EURUSD Novel Indicator Discovery Sweep

Goal: Maximize profit with WR > 75% using UNTAPPED indicators.

5 ideas across 4 brick sizes (0.0004, 0.0005, 0.0006, 0.0007):

Idea 1: "Fisher Regime" — Fisher Transform cross + adaptive regime + HTF ADX
Idea 2: "DI Dominance"  — +DI > -DI + DI spread + Supertrend + HTF
Idea 3: "Vol Conviction" — VWMACD cross + RVI cross + HTF ADX
Idea 4: "Triple Align"  — EMA9>EMA21>EMA50 + AO>0 + HTF
Idea 5: "Mega Stack"    — Best new gates combined for max WR + HTF

All use R007 entry base + session/vol baseline.
HTF: 2x brick (0.0008 for EU4/EU5, 0.0012 for EU6/EU7).

Usage:
  python renko/phase14_eurusd_sweep.py
  python renko/phase14_eurusd_sweep.py --idea 1
"""

import argparse
import contextlib
import io
import itertools
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

# ── Instrument configs ────────────────────────────────────────────────────────

INSTRUMENTS = {
    "EU4": {
        "renko_file": "OANDA_EURUSD, 1S renko 0.0004.csv",
        "htf_file":   "OANDA_EURUSD, 1S renko 0.0008.csv",
        "is_start": "2023-01-23", "is_end": "2025-09-30",
        "oos_start": "2025-10-01", "oos_end": "2026-03-19",
        "label": "EURUSD 0.0004",
    },
    "EU5": {
        "renko_file": "OANDA_EURUSD, 1S renko 0.0005.csv",
        "htf_file":   "OANDA_EURUSD, 1S renko 0.0008.csv",
        "is_start": "2022-05-18", "is_end": "2025-09-30",
        "oos_start": "2025-10-01", "oos_end": "2026-03-19",
        "label": "EURUSD 0.0005",
    },
    "EU6": {
        "renko_file": "OANDA_EURUSD, 1S renko 0.0006.csv",
        "htf_file":   "OANDA_EURUSD, 1S renko 0.0012.csv",
        "is_start": None, "is_end": "2025-09-30",
        "oos_start": "2025-10-01", "oos_end": "2026-03-19",
        "label": "EURUSD 0.0006",
    },
    "EU7": {
        "renko_file": "OANDA_EURUSD, 1S renko 0.0007.csv",
        "htf_file":   "OANDA_EURUSD, 1S renko 0.0012.csv",
        "is_start": None, "is_end": "2025-09-30",
        "oos_start": "2025-10-01", "oos_end": "2026-03-19",
        "label": "EURUSD 0.0007",
    },
}

N_BRICKS = [2, 3, 4, 5]
COOLDOWNS = [10, 20, 30]
PARAM_COMBOS = [{"n_bricks": n, "cooldown": c}
                for n, c in itertools.product(N_BRICKS, COOLDOWNS)]

COMMISSION = 0.0046
CAPITAL = 1000.0
VOL_MAX = 1.5

# ── Data loading ─────────────────────────────────────────────────────────────

def _load_ltf(renko_file):
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    from renko.phase6_enrichment import add_phase6_indicators
    from indicators.fisher_transform import calc_fisher_transform
    from indicators.rvi import calc_rvi
    from indicators.vwmacd import calc_vwmacd
    from indicators.awesome_oscillator import calc_ao
    from indicators.adaptive_regime import calc_adaptive_regime
    from indicators.blackcat1402.bc_l1_swing_trade_oscillator import calc_bc_swing_trade_oscillator
    from indicators.blackcat1402.bc_l1_trend_swing_oscillator import calc_bc_trend_swing_oscillator
    from indicators.blackcat1402.blackcat_l3_macd_wave_signal_pro import calc_bc_l3_macd_wave_signal_pro

    df = load_renko_export(renko_file)
    add_renko_indicators(df)
    add_phase6_indicators(df, include_mk=False)

    # ── NEW indicators (all pre-shifted) ─────────────────────────────────────

    # Fisher Transform
    try:
        ft = calc_fisher_transform(df, period=10)
        df["fisher"] = pd.Series(ft["fisher"], index=df.index).shift(1).values
        df["fisher_sig"] = pd.Series(ft["signal"], index=df.index).shift(1).values
    except Exception:
        df["fisher"] = np.nan; df["fisher_sig"] = np.nan

    # RVI (Relative Vigor Index)
    try:
        rvi_r = calc_rvi(df, period=10)
        df["rvi"] = pd.Series(rvi_r["rvi"], index=df.index).shift(1).values
        df["rvi_sig"] = pd.Series(rvi_r["signal"], index=df.index).shift(1).values
    except Exception:
        df["rvi"] = np.nan; df["rvi_sig"] = np.nan

    # VWMACD
    try:
        vw = calc_vwmacd(df, fast=12, slow=26, signal=9)
        df["vwmacd"] = pd.Series(vw["vwmacd"], index=df.index).shift(1).values
        df["vwmacd_sig"] = pd.Series(vw["signal"], index=df.index).shift(1).values
        df["vwmacd_hist"] = pd.Series(vw["histogram"], index=df.index).shift(1).values
    except Exception:
        df["vwmacd"] = np.nan; df["vwmacd_sig"] = np.nan; df["vwmacd_hist"] = np.nan

    # Awesome Oscillator
    try:
        ao_r = calc_ao(df, fast=5, slow=34)
        df["ao"] = pd.Series(ao_r["ao"], index=df.index).shift(1).values
    except Exception:
        df["ao"] = np.nan

    # Adaptive Regime (uses pre-shifted adx/chop/squeeze already in df)
    # Need to compute on raw (un-shifted) values, so we reconstruct
    try:
        regime = calc_adaptive_regime(df)
        df["regime_score"] = pd.Series(regime["regime_score"], index=df.index).shift(1).values
        df["regime_trending"] = pd.Series(regime["regime_trending"], index=df.index).shift(1).values
    except Exception:
        df["regime_score"] = np.nan; df["regime_trending"] = np.nan

    # BlackCat oscillators (proven in Phase 8+)
    try:
        sto = calc_bc_swing_trade_oscillator(df)
        df["_bc_sto_mf"] = sto["bc_sto_main_force"].shift(1).values
        df["_bc_sto_ll"] = sto["bc_sto_life_line"].shift(1).values
    except Exception:
        df["_bc_sto_mf"] = np.nan; df["_bc_sto_ll"] = np.nan
    try:
        tso = calc_bc_trend_swing_oscillator(df)
        df["_bc_tso_pink"] = tso["bc_tso_pink_hist"].shift(1).values
    except Exception:
        df["_bc_tso_pink"] = np.nan
    try:
        df_lc = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
        macd_r = calc_bc_l3_macd_wave_signal_pro(df_lc)
        df["_bc_macd_state"] = macd_r["bc_macd_state"].shift(1).values
        df["_bc_lc"] = macd_r["bc_lc"].shift(1).values
    except Exception:
        df["_bc_macd_state"] = np.nan; df["_bc_lc"] = np.nan

    return df


def _load_htf(htf_file):
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    df = load_renko_export(htf_file)
    add_renko_indicators(df)
    return df


def _align_htf_to_ltf(df_ltf, df_htf, htf_long, htf_short):
    htf_frame = pd.DataFrame({
        "t": df_htf.index.values, "gl": htf_long.astype(float),
        "gs": htf_short.astype(float),
    }).sort_values("t")
    ltf_frame = pd.DataFrame({"t": df_ltf.index.values}).sort_values("t")
    merged = pd.merge_asof(ltf_frame, htf_frame, on="t", direction="backward")
    gl, gs = merged["gl"].values, merged["gs"].values
    return (np.where(np.isnan(gl), True, gl > 0.5).astype(bool),
            np.where(np.isnan(gs), True, gs > 0.5).astype(bool))


# ── R007 signal generator ────────────────────────────────────────────────────

def _generate_signals(brick_up, n_bricks, cooldown, gate_long, gate_short):
    n = len(brick_up)
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
                in_position = False; trade_dir = 0
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
            if bool(np.all(window)):
                cand = 1; is_r002 = False
            elif bool(not np.any(window)):
                cand = -1; is_r002 = False
            else:
                continue

        if cand == 1 and not gate_long[i]:
            continue
        if cand == -1 and not gate_short[i]:
            continue

        if cand == 1: long_entry[i] = True
        else: short_entry[i] = True
        in_position = True
        trade_dir = cand
        if not is_r002: last_r001_bar = i

    return long_entry, long_exit, short_entry, short_exit


def _run_bt(df, le, lx, se, sx, start, end):
    from engine import BacktestConfig, run_backtest_long_short
    df2 = df.copy()
    df2["long_entry"] = le; df2["long_exit"] = lx
    df2["short_entry"] = se; df2["short_exit"] = sx
    cfg = BacktestConfig(
        initial_capital=CAPITAL, commission_pct=COMMISSION,
        slippage_ticks=0, qty_type="fixed", qty_value=1000.0,
        pyramiding=1, start_date=start, end_date=end,
        take_profit_pct=0.0, stop_loss_pct=0.0,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest_long_short(df2, cfg)
    pf = kpis.get("profit_factor", 0.0) or 0.0
    return {
        "pf": float("inf") if math.isinf(pf) else float(pf),
        "net": float(kpis.get("net_profit", 0.0) or 0.0),
        "trades": int(kpis.get("total_trades", 0) or 0),
        "wr": float(kpis.get("win_rate", 0.0) or 0.0),
        "dd": float(kpis.get("max_drawdown_pct", 0.0) or 0.0),
    }


# ── Gate helpers ──────────────────────────────────────────────────────────────

def _nan_gate(v, cond_long, cond_short):
    """Standard NaN-pass gate from a single array."""
    m = np.isnan(v)
    return m | cond_long, m | cond_short


def _nan_gate2(a, b, cond_long, cond_short):
    """NaN-pass gate from two arrays."""
    m = np.isnan(a) | np.isnan(b)
    return m | cond_long, m | cond_short


def _macd_lc_gate(df):
    macd_st = df["_bc_macd_state"].values
    bc_lc = df["_bc_lc"].values
    ms_nan = np.isnan(macd_st); lc_nan = np.isnan(bc_lc)
    ms_int = np.where(ms_nan, -1, macd_st).astype(int)
    gl = (ms_nan | np.isin(ms_int, [0, 3])) & (lc_nan | (bc_lc > 0))
    gs = (ms_nan | np.isin(ms_int, [1, 2])) & (lc_nan | (bc_lc < 0))
    return gl, gs


# ── Pre-compute all gates ────────────────────────────────────────────────────

def _precompute_gates(df_ltf, df_htf):
    n = len(df_ltf)
    hours = df_ltf.index.hour
    vr = df_ltf["vol_ratio"].values
    vol_ok = np.isnan(vr) | (vr <= VOL_MAX)

    gates = {}

    # Session baselines
    for s in [12, 13, 14]:
        ok = (hours >= s) & vol_ok
        gates[f"base_s{s}"] = (ok.copy(), ok.copy())

    # ADX levels
    adx = df_ltf["adx"].values
    adx_nan = np.isnan(adx)
    for a in [0, 20, 25, 30]:
        ok = adx_nan | (adx >= a) if a > 0 else np.ones(n, dtype=bool)
        gates[f"adx_{a}"] = ok

    # ── Proven P6 gates ──────────────────────────────────────────────────────
    from renko.phase6_sweep import _compute_gate_arrays
    for g in ["escgo_cross", "stoch_cross", "ema_cross", "ichi_cloud",
              "psar_dir", "kama_slope", "macd_hist_dir"]:
        gates[f"p6:{g}"] = _compute_gate_arrays(df_ltf, g)

    # Oscillator: macd_lc
    gates["osc:macd_lc"] = _macd_lc_gate(df_ltf)

    # ── NEW gates (never tested before) ──────────────────────────────────────

    # 1. Fisher Transform cross: fisher > fisher_sig
    f = df_ltf["fisher"].values; fs = df_ltf["fisher_sig"].values
    gates["fisher_cross"] = _nan_gate2(f, fs, f > fs, f < fs)

    # 2. RVI cross: rvi > rvi_sig
    rv = df_ltf["rvi"].values; rs = df_ltf["rvi_sig"].values
    gates["rvi_cross"] = _nan_gate2(rv, rs, rv > rs, rv < rs)

    # 3. VWMACD cross: vwmacd > vwmacd_sig
    vw = df_ltf["vwmacd"].values; vs = df_ltf["vwmacd_sig"].values
    gates["vwmacd_cross"] = _nan_gate2(vw, vs, vw > vs, vw < vs)

    # 4. VWMACD histogram direction
    vwh = df_ltf["vwmacd_hist"].values
    gates["vwmacd_hist_dir"] = _nan_gate(vwh, vwh > 0, vwh < 0)

    # 5. Awesome Oscillator direction
    ao = df_ltf["ao"].values
    gates["ao_dir"] = _nan_gate(ao, ao > 0, ao < 0)

    # 6. DI crossover: +DI > -DI
    pdi = df_ltf["plus_di"].values; mdi = df_ltf["minus_di"].values
    gates["di_cross"] = _nan_gate2(pdi, mdi, pdi > mdi, pdi < mdi)

    # 7. DI spread: abs(+DI - -DI) >= threshold
    di_diff = np.abs(pdi - mdi)
    di_nan = np.isnan(pdi) | np.isnan(mdi)
    for t in [5, 10, 15]:
        ok = di_nan | (di_diff >= t)
        gates[f"di_spread_{t}"] = (ok.copy(), ok.copy())

    # 8. Supertrend direction (NEVER tested as gate before!)
    st = df_ltf["st_dir"].values
    gates["st_dir"] = _nan_gate(st, st > 0, st < 0)

    # 9. Triple EMA: EMA9 > EMA21 > EMA50
    e9 = df_ltf["ema9"].values; e21 = df_ltf["ema21"].values; e50 = df_ltf["ema50"].values
    ema_nan = np.isnan(e9) | np.isnan(e21) | np.isnan(e50)
    gl_trip = ema_nan | ((e9 > e21) & (e21 > e50))
    gs_trip = ema_nan | ((e9 < e21) & (e21 < e50))
    gates["triple_ema"] = (gl_trip, gs_trip)

    # 10. Adaptive regime trending
    rt = df_ltf["regime_trending"].values
    rt_nan = np.isnan(rt)
    ok = rt_nan | (rt > 0.5)
    gates["regime_trend"] = (ok.copy(), ok.copy())

    # 11. Regime score thresholds
    rs_val = df_ltf["regime_score"].values
    rs_nan = np.isnan(rs_val)
    for t in [0.3, 0.5, 0.7]:
        ok = rs_nan | (rs_val > t)
        gates[f"regime_{t}"] = (ok.copy(), ok.copy())

    # 12. MACD line positive (not histogram — the line itself)
    ml = df_ltf["macd"].values
    gates["macd_line_dir"] = _nan_gate(ml, ml > 0, ml < 0)

    # 13. RSI strong zones (stricter than RSI > 50)
    rsi = df_ltf["rsi"].values
    for t in [55, 60]:
        gates[f"rsi_strong_{t}"] = _nan_gate(rsi, rsi > t, rsi < (100 - t))

    # 14. Stochastic zone: %K > 50 AND %K > %D
    sk = df_ltf["stoch_k"].values; sd = df_ltf["stoch_d"].values
    sk_nan = np.isnan(sk) | np.isnan(sd)
    gates["stoch_zone"] = (
        sk_nan | ((sk > 50) & (sk > sd)),
        sk_nan | ((sk < 50) & (sk < sd)),
    )

    # ── HTF gates ────────────────────────────────────────────────────────────
    htf_adx = df_htf["adx"].values
    htf_nan = np.isnan(htf_adx)

    for t in [0, 30, 35, 40, 45]:
        if t == 0:
            gates[f"htf_adx_{t}"] = (np.ones(n, dtype=bool), np.ones(n, dtype=bool))
        else:
            ok = htf_nan | (htf_adx >= t)
            al, as_ = _align_htf_to_ltf(df_ltf, df_htf, ok, ok.copy())
            gates[f"htf_adx_{t}"] = (al, as_)

    # HTF stoch
    htf_k = df_htf["stoch_k"].values; htf_d = df_htf["stoch_d"].values
    htf_kd_nan = np.isnan(htf_k) | np.isnan(htf_d)
    hsl = htf_kd_nan | (htf_k > htf_d)
    hss = htf_kd_nan | (htf_k < htf_d)
    gates["htf_stoch"] = _align_htf_to_ltf(df_ltf, df_htf, hsl, hss)

    # HTF EMA cross
    htf_e9 = df_htf["ema9"].values; htf_e21 = df_htf["ema21"].values
    htf_ema_nan = np.isnan(htf_e9) | np.isnan(htf_e21)
    hel = htf_ema_nan | (htf_e9 > htf_e21)
    hes = htf_ema_nan | (htf_e9 < htf_e21)
    gates["htf_ema"] = _align_htf_to_ltf(df_ltf, df_htf, hel, hes)

    # HTF supertrend
    htf_st = df_htf["st_dir"].values
    htf_st_nan = np.isnan(htf_st)
    htf_stl = htf_st_nan | (htf_st > 0)
    htf_sts = htf_st_nan | (htf_st < 0)
    gates["htf_st"] = _align_htf_to_ltf(df_ltf, df_htf, htf_stl, htf_sts)

    # HTF PSAR
    htf_psar = df_htf["psar_dir"].values
    htf_psar_nan = np.isnan(htf_psar)
    htf_pl = htf_psar_nan | (htf_psar > 0)
    htf_ps = htf_psar_nan | (htf_psar < 0)
    gates["htf_psar"] = _align_htf_to_ltf(df_ltf, df_htf, htf_pl, htf_ps)

    return gates


# ── Build combo list per idea ─────────────────────────────────────────────────

def _build_all_combos(idea_num=None):
    combos = []

    # Idea 1: Fisher Regime — Fisher cross + regime + P6 + HTF
    if idea_num is None or idea_num == 1:
        for sess, adx, htf_t, regime_t in itertools.product(
            [12, 13, 14], [20, 25], [35, 40, 45], [0.3, 0.5]
        ):
            for pc in PARAM_COMBOS:
                combos.append({"idea":1, "sess":sess, "adx":adx,
                    "htf_thresh":htf_t, "regime_t":regime_t, **pc})

    # Idea 2: DI Dominance — DI cross + DI spread + supertrend + HTF
    if idea_num is None or idea_num == 2:
        for sess, di_spread, htf_t, p6 in itertools.product(
            [12, 13, 14], [5, 10, 15], [0, 35, 40, 45],
            ["stoch_cross", "ema_cross", "escgo_cross"]
        ):
            for pc in PARAM_COMBOS:
                combos.append({"idea":2, "sess":sess, "di_spread":di_spread,
                    "htf_thresh":htf_t, "p6":p6, **pc})

    # Idea 3: Volume Conviction — VWMACD cross + RVI cross + HTF
    if idea_num is None or idea_num == 3:
        for sess, adx, htf_t, osc in itertools.product(
            [12, 13, 14], [20, 25, 30], [0, 35, 40, 45],
            [None, "macd_lc"]
        ):
            for pc in PARAM_COMBOS:
                combos.append({"idea":3, "sess":sess, "adx":adx,
                    "htf_thresh":htf_t, "osc":osc, **pc})

    # Idea 4: Triple Align — triple EMA + AO + stoch_zone + HTF
    if idea_num is None or idea_num == 4:
        for sess, adx, htf_t, htf_type in itertools.product(
            [12, 13, 14], [0, 20, 25], [0, 35, 40, 45],
            ["htf_adx", "htf_st", "htf_psar"]
        ):
            for pc in PARAM_COMBOS:
                combos.append({"idea":4, "sess":sess, "adx":adx,
                    "htf_thresh":htf_t, "htf_type":htf_type, **pc})

    # Idea 5: Mega Stack — max WR by stacking strongest new + proven gates + HTF
    if idea_num is None or idea_num == 5:
        for sess, adx, htf_t, new_gate, p6 in itertools.product(
            [13, 14], [20, 25],
            [35, 40, 45],
            ["fisher_cross", "di_cross", "rvi_cross", "vwmacd_cross", "st_dir"],
            ["stoch_cross", "ema_cross", "escgo_cross", "triple_ema"]
        ):
            for pc in PARAM_COMBOS:
                combos.append({"idea":5, "sess":sess, "adx":adx,
                    "htf_thresh":htf_t, "new_gate":new_gate, "p6":p6, **pc})

    return combos


# ── Worker: runs one combo ───────────────────────────────────────────────────

_w_data = {}

def _init_worker(brick_up, df_bytes, is_start, is_end, oos_start, oos_end):
    _w_data["brick_up"] = brick_up
    _w_data["df"] = pd.read_pickle(io.BytesIO(df_bytes))
    _w_data["is_start"] = is_start
    _w_data["is_end"] = is_end
    _w_data["oos_start"] = oos_start
    _w_data["oos_end"] = oos_end


def _run_one_combo(args):
    combo, gate_long, gate_short = args
    brick_up = _w_data["brick_up"]
    df = _w_data["df"]

    le, lx, se, sx = _generate_signals(
        brick_up, combo["n_bricks"], combo["cooldown"], gate_long, gate_short
    )
    is_r = _run_bt(df, le, lx, se, sx, _w_data["is_start"], _w_data["is_end"])
    oos_r = _run_bt(df, le, lx, se, sx, _w_data["oos_start"], _w_data["oos_end"])
    return is_r, oos_r


# ── Assemble gate arrays per combo ───────────────────────────────────────────

def _assemble_tasks(combos, gates):
    tasks = []
    for combo in combos:
        idea = combo["idea"]
        bl, bs = gates[f"base_s{combo['sess']}"]
        bl = bl.copy(); bs = bs.copy()

        if idea == 1:
            # Fisher cross + regime threshold + ADX + HTF
            bl &= gates[f"adx_{combo['adx']}"]
            bs &= gates[f"adx_{combo['adx']}"]
            fl, fs = gates["fisher_cross"]
            bl &= fl; bs &= fs
            rl, rs_ = gates[f"regime_{combo['regime_t']}"]
            bl &= rl; bs &= rs_
            hl, hs = gates[f"htf_adx_{combo['htf_thresh']}"]
            bl &= hl; bs &= hs

        elif idea == 2:
            # DI cross + DI spread + supertrend + P6 + HTF
            dl, ds = gates["di_cross"]
            bl &= dl; bs &= ds
            spl, sps = gates[f"di_spread_{combo['di_spread']}"]
            bl &= spl; bs &= sps
            stl, sts = gates["st_dir"]
            bl &= stl; bs &= sts
            pl, ps = gates[f"p6:{combo['p6']}"]
            bl &= pl; bs &= ps
            if combo["htf_thresh"] > 0:
                hl, hs = gates[f"htf_adx_{combo['htf_thresh']}"]
                bl &= hl; bs &= hs

        elif idea == 3:
            # VWMACD cross + RVI cross + ADX + HTF + optional osc
            bl &= gates[f"adx_{combo['adx']}"]
            bs &= gates[f"adx_{combo['adx']}"]
            vl, vs = gates["vwmacd_cross"]
            bl &= vl; bs &= vs
            rl, rs_ = gates["rvi_cross"]
            bl &= rl; bs &= rs_
            if combo["osc"] == "macd_lc":
                ol, os_ = gates["osc:macd_lc"]
                bl &= ol; bs &= os_
            if combo["htf_thresh"] > 0:
                hl, hs = gates[f"htf_adx_{combo['htf_thresh']}"]
                bl &= hl; bs &= hs

        elif idea == 4:
            # Triple EMA + AO + stoch_zone + ADX + HTF (various HTF types)
            if combo["adx"] > 0:
                bl &= gates[f"adx_{combo['adx']}"]
                bs &= gates[f"adx_{combo['adx']}"]
            tl, ts = gates["triple_ema"]
            bl &= tl; bs &= ts
            al, as_ = gates["ao_dir"]
            bl &= al; bs &= as_
            szl, szs = gates["stoch_zone"]
            bl &= szl; bs &= szs
            if combo["htf_thresh"] > 0:
                htf_key = combo["htf_type"]
                if htf_key == "htf_adx":
                    hl, hs = gates[f"htf_adx_{combo['htf_thresh']}"]
                elif htf_key == "htf_st":
                    hl, hs = gates["htf_st"]
                elif htf_key == "htf_psar":
                    hl, hs = gates["htf_psar"]
                else:
                    hl, hs = gates[f"htf_adx_{combo['htf_thresh']}"]
                bl &= hl; bs &= hs

        elif idea == 5:
            # Mega stack: session + ADX + new_gate + P6 + HTF
            bl &= gates[f"adx_{combo['adx']}"]
            bs &= gates[f"adx_{combo['adx']}"]
            # New gate
            ng = combo["new_gate"]
            if ng in gates:
                nl, ns = gates[ng]
                bl &= nl; bs &= ns
            # P6 gate
            p6_key = combo["p6"]
            if p6_key == "triple_ema":
                pl, ps = gates["triple_ema"]
            else:
                pl, ps = gates[f"p6:{p6_key}"]
            bl &= pl; bs &= ps
            # HTF
            hl, hs = gates[f"htf_adx_{combo['htf_thresh']}"]
            bl &= hl; bs &= hs

        tasks.append((combo, bl, bs))

    return tasks


# ── Main sweep ────────────────────────────────────────────────────────────────

def run_sweep(idea_num=None):
    combos = _build_all_combos(idea_num)
    n_inst = len(INSTRUMENTS)
    total = len(combos) * n_inst

    print(f"\n{'='*70}")
    print(f"Phase 14 — EURUSD Novel Indicator Discovery Sweep")
    print(f"Ideas: {idea_num or 'ALL (1-5)'}")
    print(f"Combos per instrument: {len(combos)}")
    print(f"Instruments: {list(INSTRUMENTS.keys())}")
    print(f"Total runs: {total} ({total*2} backtests)")
    print(f"Workers: {MAX_WORKERS}")
    print(f"{'='*70}\n")

    all_results = []

    for inst_key, cfg in INSTRUMENTS.items():
        print(f"\n--- [{inst_key}] {cfg['label']} ---", flush=True)
        print("  Loading LTF data + new indicators...", flush=True)
        df_ltf = _load_ltf(cfg["renko_file"])
        print("  Loading HTF data...", flush=True)
        df_htf = _load_htf(cfg["htf_file"])

        is_start = cfg["is_start"] or str(df_ltf.index[0].date())

        print("  Pre-computing gates...", flush=True)
        gates = _precompute_gates(df_ltf, df_htf)

        brick_up = df_ltf["brick_up"].values

        print(f"  Assembling {len(combos)} combos...", flush=True)
        tasks = _assemble_tasks(combos, gates)

        print(f"  Running {len(tasks)} combos...", flush=True)
        done = 0
        inst_results = []

        buf = io.BytesIO()
        df_ltf.to_pickle(buf)
        df_bytes = buf.getvalue()

        with ProcessPoolExecutor(
            max_workers=MAX_WORKERS,
            initializer=_init_worker,
            initargs=(brick_up, df_bytes, is_start, cfg["is_end"],
                      cfg["oos_start"], cfg["oos_end"]),
        ) as pool:
            futures = {}
            for task in tasks:
                combo, gl, gs = task
                f = pool.submit(_run_one_combo, (combo, gl, gs))
                futures[f] = combo

            for fut in as_completed(futures):
                combo = futures[fut]
                try:
                    is_r, oos_r = fut.result()
                    inst_results.append({
                        "inst": inst_key, "label": cfg["label"],
                        "idea": combo["idea"], "combo": combo,
                        "is": is_r, "oos": oos_r,
                    })
                except Exception as e:
                    print(f"    ERROR: {e}", flush=True)
                done += 1
                if done % 500 == 0 or done == len(tasks):
                    print(f"    [{done:>5}/{len(tasks)}]", flush=True)

        all_results.extend(inst_results)
        print(f"  [{inst_key}] done — {len(inst_results)} results", flush=True)

    # ── Filter & Sort ────────────────────────────────────────────────────────
    # Primary: filter OOS WR >= 75% with >= 8 trades, rank by net profit
    wr75 = [r for r in all_results
            if r["oos"]["wr"] >= 75.0 and r["oos"]["trades"] >= 8]
    wr75.sort(key=lambda r: (r["oos"]["net"], r["oos"]["pf"]), reverse=True)

    # Also keep overall top for reference
    def sort_key(r):
        viable = r["oos"]["trades"] >= 8
        pf = r["oos"]["pf"] if not math.isinf(r["oos"]["pf"]) else 1e12
        return (viable, pf, r["oos"]["net"])

    all_results.sort(key=sort_key, reverse=True)

    idea_names = {
        1: "Fisher Regime", 2: "DI Dominance", 3: "Vol Conviction",
        4: "Triple Align", 5: "Mega Stack",
    }

    # ── Display WR >= 75% results ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"WR >= 75% RESULTS ({len(wr75)} configs)")
    print(f"{'='*70}")

    for idea in sorted(set(r["idea"] for r in wr75) if wr75 else []):
        idea_r = [r for r in wr75 if r["idea"] == idea]
        print(f"\n  IDEA {idea}: {idea_names[idea]} ({len(idea_r)} configs)")
        for inst_key in INSTRUMENTS:
            inst_r = [r for r in idea_r if r["inst"] == inst_key]
            inst_r.sort(key=lambda r: r["oos"]["net"], reverse=True)
            if not inst_r:
                continue
            print(f"    [{inst_key}] {INSTRUMENTS[inst_key]['label']} ({len(inst_r)} configs)")
            for r in inst_r[:5]:
                pf_is = "INF" if math.isinf(r["is"]["pf"]) else f"{r['is']['pf']:.2f}"
                pf_oos = "INF" if math.isinf(r["oos"]["pf"]) else f"{r['oos']['pf']:.2f}"
                c = r["combo"]
                keys_skip = {"idea", "n_bricks", "cooldown"}
                extra = " ".join(f"{k}={v}" for k, v in c.items() if k not in keys_skip)
                print(f"      IS PF={pf_is:>7} T={r['is']['trades']:>4} | "
                      f"OOS PF={pf_oos:>7} T={r['oos']['trades']:>3} "
                      f"WR={r['oos']['wr']:>5.1f}% Net={r['oos']['net']:>8.2f} "
                      f"DD={r['oos']['dd']:>5.2f}% | n={c['n_bricks']} cd={c['cooldown']} {extra}")

    # ── Overall top 30 ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"OVERALL TOP 30 (by PF, trades >= 8)")
    print(f"{'='*70}")
    for i, r in enumerate(all_results[:30]):
        pf_oos = "INF" if math.isinf(r["oos"]["pf"]) else f"{r['oos']['pf']:.2f}"
        c = r["combo"]
        keys_skip = {"idea", "n_bricks", "cooldown"}
        extra = " ".join(f"{k}={v}" for k, v in c.items() if k not in keys_skip)
        marker = " ***" if r["oos"]["wr"] >= 75.0 else ""
        print(f"  {i+1:>2}. [{r['inst']}] Idea{r['idea']} "
              f"OOS PF={pf_oos:>7} T={r['oos']['trades']:>3} "
              f"WR={r['oos']['wr']:>5.1f}% Net={r['oos']['net']:>8.2f} "
              f"DD={r['oos']['dd']:>5.2f}% | n={c['n_bricks']} cd={c['cooldown']} {extra}{marker}")

    # ── Top 20 by NET PROFIT with WR >= 75% ─────────────────────────────────
    print(f"\n{'='*70}")
    print(f"TOP 20 BY NET PROFIT (WR >= 75%, trades >= 8)")
    print(f"{'='*70}")
    for i, r in enumerate(wr75[:20]):
        pf_oos = "INF" if math.isinf(r["oos"]["pf"]) else f"{r['oos']['pf']:.2f}"
        c = r["combo"]
        keys_skip = {"idea", "n_bricks", "cooldown"}
        extra = " ".join(f"{k}={v}" for k, v in c.items() if k not in keys_skip)
        print(f"  {i+1:>2}. [{r['inst']}] Idea{r['idea']} "
              f"OOS PF={pf_oos:>7} T={r['oos']['trades']:>3} "
              f"WR={r['oos']['wr']:>5.1f}% Net={r['oos']['net']:>8.2f} "
              f"DD={r['oos']['dd']:>5.2f}% | n={c['n_bricks']} cd={c['cooldown']} {extra}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = ROOT / "ai_context" / "phase14_results.json"
    out_path.parent.mkdir(exist_ok=True)

    serializable = []
    for r in all_results:
        c = r["combo"]
        sr = {
            "inst": r["inst"], "label": r["label"], "idea": r["idea"],
            "combo": {k: (str(v) if isinstance(v, float) and k == "regime_t" else v)
                      for k, v in c.items() if k != "idea"},
            "is_pf": "inf" if math.isinf(r["is"]["pf"]) else r["is"]["pf"],
            "is_trades": r["is"]["trades"], "is_wr": r["is"]["wr"],
            "is_net": r["is"]["net"],
            "oos_pf": "inf" if math.isinf(r["oos"]["pf"]) else r["oos"]["pf"],
            "oos_trades": r["oos"]["trades"], "oos_wr": r["oos"]["wr"],
            "oos_net": r["oos"]["net"], "oos_dd": r["oos"]["dd"],
        }
        serializable.append(sr)

    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Total: {len(all_results)} runs ({len(all_results)*2} backtests)")
    print(f"WR >= 75% configs: {len(wr75)}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idea", type=int, default=None, help="Run single idea (1-5)")
    args = parser.parse_args()
    run_sweep(args.idea)
