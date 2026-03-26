#!/usr/bin/env python3
"""
phase13_eurusd_sweep.py — EURUSD Creative Strategy Sweep

4 ideas across 4 brick sizes (0.0004, 0.0005, 0.0006, 0.0007):

Idea 1: "Gap Filler" — escgo_cross + HTF ADX (never tested on EURUSD)
Idea 2: "Squeeze Breakout" — BB squeeze→expansion trigger
Idea 3: "Triple Cascade" — multi-TF momentum alignment (HTF stoch + HTF ema + LTF)
Idea 4: "ADX Momentum" — rising ADX gate (ADX slope, not just level)

All use R007 entry base + session/vol baseline.
HTF: 2x brick (0.0008 for EU4/EU5, 0.0012 for EU6/EU7).

Usage:
  python renko/phase13_eurusd_sweep.py
  python renko/phase13_eurusd_sweep.py --idea 1
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

# ── Data loading (called once per instrument in main process) ─────────────────

def _load_ltf(renko_file):
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    from renko.phase6_enrichment import add_phase6_indicators
    from indicators.blackcat1402.bc_l1_swing_trade_oscillator import calc_bc_swing_trade_oscillator
    from indicators.blackcat1402.bc_l1_trend_swing_oscillator import calc_bc_trend_swing_oscillator
    from indicators.blackcat1402.blackcat_l3_macd_wave_signal_pro import calc_bc_l3_macd_wave_signal_pro

    df = load_renko_export(renko_file)
    add_renko_indicators(df)
    add_phase6_indicators(df, include_mk=False)

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
    """R007 logic — returns (long_entry, long_exit, short_entry, short_exit)."""
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

def _p6_gate(df, name):
    from renko.phase6_sweep import _compute_gate_arrays
    return _compute_gate_arrays(df, name)


def _squeeze_gate_vectorized(sq_on_vals, bb_bw_vals, lookback):
    """Vectorized: True when squeeze was active within last `lookback` bars AND bb_bw expanding."""
    n = len(sq_on_vals)
    sq = np.where(np.isnan(sq_on_vals), 0, sq_on_vals > 0.5).astype(float)
    # Rolling max over lookback window: if any bar had squeeze, max > 0
    had_squeeze = np.zeros(n, dtype=bool)
    # Use a simple rolling window
    for i in range(n):
        if i >= lookback:
            # Check if any of [i-lookback, i) had squeeze
            had_squeeze[i] = np.any(sq[i - lookback : i] > 0.5)
        else:
            had_squeeze[i] = np.any(sq[:i] > 0.5) if i > 0 else False

    # BB expanding: bb_bw[i] > bb_bw[i-lookback]
    bw_prev = np.roll(bb_bw_vals, lookback)
    bw_prev[:lookback] = np.nan
    expanding = np.isnan(bb_bw_vals) | np.isnan(bw_prev) | (bb_bw_vals > bw_prev)

    gate = had_squeeze & expanding
    return gate, gate


def _adx_rising_gate_vectorized(adx_vals, adx_min, rise_lb):
    """Vectorized: ADX >= adx_min AND adx[i] > adx[i-rise_lb]."""
    adx_prev = np.roll(adx_vals, rise_lb)
    adx_prev[:rise_lb] = np.nan
    nan_cur = np.isnan(adx_vals)
    nan_prev = np.isnan(adx_prev)
    above_min = nan_cur | (adx_vals >= adx_min)
    rising = nan_cur | nan_prev | (adx_vals > adx_prev)
    gate = above_min & rising
    return gate, gate


def _macd_lc_gate(df):
    macd_st = df["_bc_macd_state"].values
    bc_lc = df["_bc_lc"].values
    ms_nan = np.isnan(macd_st); lc_nan = np.isnan(bc_lc)
    ms_int = np.where(ms_nan, -1, macd_st).astype(int)
    gl = (ms_nan | np.isin(ms_int, [0, 3])) & (lc_nan | (bc_lc > 0))
    gs = (ms_nan | np.isin(ms_int, [1, 2])) & (lc_nan | (bc_lc < 0))
    return gl, gs


# ── Pre-compute all gates for one instrument ──────────────────────────────────

def _precompute_gates(df_ltf, df_htf):
    """Pre-compute all gate arrays for all ideas. Returns dict of arrays."""
    n = len(df_ltf)
    hours = df_ltf.index.hour
    vr = df_ltf["vol_ratio"].values
    vol_ok = np.isnan(vr) | (vr <= VOL_MAX)

    gates = {}

    # Sessions
    for s in [12, 13, 14]:
        ok = (hours >= s) & vol_ok
        gates[f"base_s{s}"] = (ok.copy(), ok.copy())

    # ADX levels
    adx = df_ltf["adx"].values
    adx_nan = np.isnan(adx)
    for a in [0, 15, 20, 25, 30]:
        ok = adx_nan | (adx >= a) if a > 0 else np.ones(n, dtype=bool)
        gates[f"adx_{a}"] = ok

    # P6 gates
    for g in ["escgo_cross", "stoch_cross", "ema_cross", "ichi_cloud"]:
        gates[f"p6:{g}"] = _p6_gate(df_ltf, g)

    # Oscillator: macd_lc
    gates["osc:macd_lc"] = _macd_lc_gate(df_ltf)

    # Squeeze gates
    sq_on = df_ltf["sq_on"].values.astype(float)
    bb_bw = df_ltf["bb_bw"].values
    for lb in [3, 5, 8]:
        gates[f"squeeze_{lb}"] = _squeeze_gate_vectorized(sq_on, bb_bw, lb)

    # ADX rising gates
    for amin in [15, 20, 25]:
        for rlb in [3, 5, 8]:
            gates[f"adxrise_{amin}_{rlb}"] = _adx_rising_gate_vectorized(adx, amin, rlb)

    # HTF gates
    htf_adx = df_htf["adx"].values
    htf_nan = np.isnan(htf_adx)
    htf_k = df_htf["stoch_k"].values; htf_d = df_htf["stoch_d"].values
    htf_kd_nan = np.isnan(htf_k) | np.isnan(htf_d)
    htf_e9 = df_htf["ema9"].values; htf_e21 = df_htf["ema21"].values
    htf_ema_nan = np.isnan(htf_e9) | np.isnan(htf_e21)

    for t in [0, 30, 35, 40, 45]:
        if t == 0:
            gates[f"htf_adx_{t}"] = (np.ones(n, dtype=bool), np.ones(n, dtype=bool))
        else:
            ok = htf_nan | (htf_adx >= t)
            al, as_ = _align_htf_to_ltf(df_ltf, df_htf, ok, ok.copy())
            gates[f"htf_adx_{t}"] = (al, as_)

    # HTF stoch
    hsl = htf_kd_nan | (htf_k > htf_d)
    hss = htf_kd_nan | (htf_k < htf_d)
    gates["htf_stoch"] = _align_htf_to_ltf(df_ltf, df_htf, hsl, hss)

    # HTF ema
    hel = htf_ema_nan | (htf_e9 > htf_e21)
    hes = htf_ema_nan | (htf_e9 < htf_e21)
    gates["htf_ema"] = _align_htf_to_ltf(df_ltf, df_htf, hel, hes)

    return gates


# ── Build combo list per idea ─────────────────────────────────────────────────

def _build_all_combos(idea_num=None):
    combos = []

    if idea_num is None or idea_num == 1:
        for sess, adx, htf_t, osc in itertools.product(
            [12, 13, 14], [20, 25, 30], [30, 35, 40, 45], [None, "macd_lc"]
        ):
            for pc in PARAM_COMBOS:
                combos.append({"idea":1, "sess":sess, "adx":adx,
                    "htf_thresh":htf_t, "osc":osc, **pc})

    if idea_num is None or idea_num == 2:
        for sess, adx, lb, htf_t in itertools.product(
            [12, 13, 14], [0, 20, 25], [3, 5, 8], [0, 35, 40]
        ):
            for pc in PARAM_COMBOS:
                combos.append({"idea":2, "sess":sess, "adx":adx,
                    "squeeze_lb":lb, "htf_thresh":htf_t, **pc})

    if idea_num is None or idea_num == 3:
        for sess, adx, htf_t, ltf_p6 in itertools.product(
            [12, 13, 14], [20, 25], [30, 35, 40], ["stoch_cross", "ema_cross"]
        ):
            for pc in PARAM_COMBOS:
                combos.append({"idea":3, "sess":sess, "adx":adx,
                    "htf_thresh":htf_t, "ltf_p6":ltf_p6, **pc})

    if idea_num is None or idea_num == 4:
        for sess, adx_min, rise_lb, htf_t, p6 in itertools.product(
            [12, 13, 14], [15, 20, 25], [3, 5, 8], [0, 35, 40],
            ["stoch_cross", "ema_cross", "escgo_cross"]
        ):
            for pc in PARAM_COMBOS:
                combos.append({"idea":4, "sess":sess, "adx_min":adx_min,
                    "adx_rise_lb":rise_lb, "htf_thresh":htf_t, "p6":p6, **pc})

    return combos


# ── Worker: runs one combo using pre-loaded data from global ──────────────────

_w_data = {}  # populated by worker initializer

def _init_worker(brick_up, df_bytes, is_start, is_end, oos_start, oos_end):
    """Worker initializer — deserialize data once per process."""
    _w_data["brick_up"] = brick_up
    _w_data["df"] = pd.read_pickle(io.BytesIO(df_bytes))
    _w_data["is_start"] = is_start
    _w_data["is_end"] = is_end
    _w_data["oos_start"] = oos_start
    _w_data["oos_end"] = oos_end


def _run_one_combo(args):
    """Worker function — one combo, one IS+OOS run."""
    combo, gate_long, gate_short = args
    brick_up = _w_data["brick_up"]
    df = _w_data["df"]

    le, lx, se, sx = _generate_signals(
        brick_up, combo["n_bricks"], combo["cooldown"], gate_long, gate_short
    )
    is_r = _run_bt(df, le, lx, se, sx, _w_data["is_start"], _w_data["is_end"])
    oos_r = _run_bt(df, le, lx, se, sx, _w_data["oos_start"], _w_data["oos_end"])
    return is_r, oos_r


# ── Main sweep ────────────────────────────────────────────────────────────────

def run_sweep(idea_num=None):
    combos = _build_all_combos(idea_num)
    n_inst = len(INSTRUMENTS)
    total = len(combos) * n_inst

    print(f"\n{'='*70}")
    print(f"Phase 13 — EURUSD Creative Strategy Sweep")
    print(f"Ideas: {idea_num or 'ALL (1-4)'}")
    print(f"Combos per instrument: {len(combos)}")
    print(f"Instruments: {list(INSTRUMENTS.keys())}")
    print(f"Total runs: {total} ({total*2} backtests)")
    print(f"Workers: {MAX_WORKERS}")
    print(f"{'='*70}\n")

    all_results = []

    for inst_key, cfg in INSTRUMENTS.items():
        print(f"\n--- [{inst_key}] {cfg['label']} ---", flush=True)
        print("  Loading LTF data...", flush=True)
        df_ltf = _load_ltf(cfg["renko_file"])
        print("  Loading HTF data...", flush=True)
        df_htf = _load_htf(cfg["htf_file"])

        is_start = cfg["is_start"] or str(df_ltf.index[0].date())

        print("  Pre-computing gates...", flush=True)
        gates = _precompute_gates(df_ltf, df_htf)

        # Build gate arrays for each combo
        brick_up = df_ltf["brick_up"].values

        tasks = []
        for combo in combos:
            idea = combo["idea"]
            bl, bs = gates[f"base_s{combo['sess']}"]
            bl = bl.copy(); bs = bs.copy()

            if idea == 1:
                bl &= gates[f"adx_{combo['adx']}"]
                bs &= gates[f"adx_{combo['adx']}"]
                pl, ps = gates["p6:escgo_cross"]
                bl &= pl; bs &= ps
                if combo["osc"] == "macd_lc":
                    ol, os_ = gates["osc:macd_lc"]
                    bl &= ol; bs &= os_
                hl, hs = gates[f"htf_adx_{combo['htf_thresh']}"]
                bl &= hl; bs &= hs

            elif idea == 2:
                if combo["adx"] > 0:
                    bl &= gates[f"adx_{combo['adx']}"]
                    bs &= gates[f"adx_{combo['adx']}"]
                sql, sqs = gates[f"squeeze_{combo['squeeze_lb']}"]
                bl &= sql; bs &= sqs
                if combo["htf_thresh"] > 0:
                    hl, hs = gates[f"htf_adx_{combo['htf_thresh']}"]
                    bl &= hl; bs &= hs

            elif idea == 3:
                bl &= gates[f"adx_{combo['adx']}"]
                bs &= gates[f"adx_{combo['adx']}"]
                pl, ps = gates[f"p6:{combo['ltf_p6']}"]
                bl &= pl; bs &= ps
                hl, hs = gates[f"htf_adx_{combo['htf_thresh']}"]
                bl &= hl; bs &= hs
                tsl, tss = gates["htf_stoch"]
                bl &= tsl; bs &= tss
                tel, tes = gates["htf_ema"]
                bl &= tel; bs &= tes

            elif idea == 4:
                arl, ars = gates[f"adxrise_{combo['adx_min']}_{combo['adx_rise_lb']}"]
                bl &= arl; bs &= ars
                pl, ps = gates[f"p6:{combo['p6']}"]
                bl &= pl; bs &= ps
                if combo["htf_thresh"] > 0:
                    hl, hs = gates[f"htf_adx_{combo['htf_thresh']}"]
                    bl &= hl; bs &= hs

            tasks.append((combo, bl, bs))

        # Run all combos for this instrument in parallel
        print(f"  Running {len(tasks)} combos...", flush=True)
        done = 0
        inst_results = []

        # Serialize df for workers
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
                if done % 200 == 0 or done == len(tasks):
                    print(f"    [{done:>5}/{len(tasks)}]", flush=True)

        all_results.extend(inst_results)
        print(f"  [{inst_key}] done — {len(inst_results)} results", flush=True)

    # ── Sort & display ────────────────────────────────────────────────────────
    def sort_key(r):
        viable = r["oos"]["trades"] >= 10
        pf = r["oos"]["pf"] if not math.isinf(r["oos"]["pf"]) else 1e12
        return (viable, pf, r["oos"]["net"])

    all_results.sort(key=sort_key, reverse=True)

    idea_names = {1: "Gap Filler (escgo+HTF)", 2: "Squeeze Breakout",
                  3: "Triple Cascade", 4: "ADX Momentum"}

    for idea in sorted(set(r["idea"] for r in all_results)):
        idea_r = [r for r in all_results if r["idea"] == idea]
        print(f"\n{'='*70}")
        print(f"IDEA {idea}: {idea_names[idea]}")
        print(f"{'='*70}")

        for inst_key in INSTRUMENTS:
            inst_r = [r for r in idea_r if r["inst"] == inst_key]
            inst_r.sort(key=sort_key, reverse=True)
            viable = [r for r in inst_r if r["oos"]["trades"] >= 10]
            print(f"\n  [{inst_key}] {INSTRUMENTS[inst_key]['label']} — "
                  f"{len(inst_r)} runs, {len(viable)} viable")
            for r in inst_r[:5]:
                pf_is = "INF" if math.isinf(r["is"]["pf"]) else f"{r['is']['pf']:.2f}"
                pf_oos = "INF" if math.isinf(r["oos"]["pf"]) else f"{r['oos']['pf']:.2f}"
                c = r["combo"]
                # Compact combo display
                keys_skip = {"idea", "n_bricks", "cooldown"}
                extra = " ".join(f"{k}={v}" for k, v in c.items() if k not in keys_skip)
                print(f"    IS PF={pf_is:>7} T={r['is']['trades']:>4} | "
                      f"OOS PF={pf_oos:>7} T={r['oos']['trades']:>3} "
                      f"WR={r['oos']['wr']:>5.1f}% Net={r['oos']['net']:>7.2f} | "
                      f"n={c['n_bricks']} cd={c['cooldown']} {extra}")

    # ── Overall top 20 ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"OVERALL TOP 20 (all ideas, all instruments)")
    print(f"{'='*70}")
    for i, r in enumerate(all_results[:20]):
        pf_oos = "INF" if math.isinf(r["oos"]["pf"]) else f"{r['oos']['pf']:.2f}"
        c = r["combo"]
        keys_skip = {"idea", "n_bricks", "cooldown"}
        extra = " ".join(f"{k}={v}" for k, v in c.items() if k not in keys_skip)
        print(f"  {i+1:>2}. [{r['inst']}] Idea{r['idea']} "
              f"OOS PF={pf_oos:>7} T={r['oos']['trades']:>3} "
              f"WR={r['oos']['wr']:>5.1f}% Net={r['oos']['net']:>7.2f} "
              f"DD={r['oos']['dd']:>5.2f}% | n={c['n_bricks']} cd={c['cooldown']} {extra}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = ROOT / "ai_context" / "phase13_results.json"
    out_path.parent.mkdir(exist_ok=True)

    serializable = []
    for r in all_results:
        c = r["combo"]
        sr = {
            "inst": r["inst"], "label": r["label"], "idea": r["idea"],
            "combo": {k: v for k, v in c.items() if k != "idea"},
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

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idea", type=int, default=None, help="Run single idea (1-4)")
    args = parser.parse_args()
    run_sweep(args.idea)
