#!/usr/bin/env python3
"""
phase9_exit_sweep.py -- Exit Strategy Optimization

Fixes entry configs to Phase 8 winners and sweeps 15 exit strategy variants.
Current exit (first opposing brick) has been unchanged since R001.

Exit dimensions:
  n_opp:     {1, 2, 3}  -- consecutive opposing bricks needed to trigger exit
  min_hold:  {0, 2, 3, 5}  -- minimum bricks held before exit allowed
  max_hold:  {0, 20, 30}  -- forced exit after N bricks (0 = off)
  gate:      {none, p6_and, osc_and}  -- indicator-conditional exit

15 exit configs x 12 param combos x 4 instruments = 720 runs (1,440 IS+OOS backtests)

Usage:
  python renko/phase9_exit_sweep.py
  python renko/phase9_exit_sweep.py --no-parallel
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

# -- Instrument configs (Phase 8 winners as fixed entry stacks) ----------------

INSTRUMENTS = {
    "EURUSD_4": {
        "renko_file":  "OANDA_EURUSD, 1S renko 0.0004.csv",
        "is_start":    "2023-01-23",
        "is_end":      "2025-09-30",
        "oos_start":   "2025-10-01",
        "oos_end":     "2026-03-05",
        "commission":  0.0046,
        "capital":     1000.0,
        "include_mk":  False,
        "label":       "EURUSD 0.0004",
        # Fixed Phase 8 winner entry stack
        "p6_gate":     "stoch_cross",
        "osc":         None,
        "adx_thresh":  20,
        "sess_start":  12,
    },
    "EURUSD_5": {
        "renko_file":  "OANDA_EURUSD, 1S renko 0.0005.csv",
        "is_start":    "2022-05-18",
        "is_end":      "2025-09-30",
        "oos_start":   "2025-10-01",
        "oos_end":     "2026-03-05",
        "commission":  0.0046,
        "capital":     1000.0,
        "include_mk":  False,
        "label":       "EURUSD 0.0005",
        "p6_gate":     "ichi_cloud",
        "osc":         "sto_tso",
        "adx_thresh":  30,
        "sess_start":  14,
    },
    "GBPJPY": {
        "renko_file":  "OANDA_GBPJPY, 1S renko 0.05.csv",
        "is_start":    "2024-11-21",
        "is_end":      "2025-09-30",
        "oos_start":   "2025-10-01",
        "oos_end":     "2026-02-28",
        "commission":  0.005,
        "capital":     150_000.0,
        "include_mk":  True,
        "label":       "GBPJPY 0.05",
        "p6_gate":     "psar_dir",
        "osc":         "macd_lc",
        "adx_thresh":  30,
        "sess_start":  13,
    },
    "EURAUD": {
        "renko_file":  "OANDA_EURAUD, 1S renko 0.0006.csv",
        "is_start":    "2023-07-20",
        "is_end":      "2025-09-30",
        "oos_start":   "2025-10-01",
        "oos_end":     "2026-03-17",
        "commission":  0.009,
        "capital":     1000.0,
        "include_mk":  False,
        "label":       "EURAUD 0.0006",
        "p6_gate":     "ichi_cloud",
        "osc":         "sto_tso",
        "adx_thresh":  30,
        "sess_start":  14,
    },
}

# -- Sweep dimensions ----------------------------------------------------------

PARAM_GRID = {
    "n_bricks": [2, 3, 4, 5],
    "cooldown": [10, 20, 30],
}

VOL_MAX = 1.5

EXIT_CONFIGS = [
    {"name": "baseline",    "n_opp": 1, "min_hold": 0, "max_hold": 0,  "gate": "none"},
    {"name": "opp2",        "n_opp": 2, "min_hold": 0, "max_hold": 0,  "gate": "none"},
    {"name": "opp3",        "n_opp": 3, "min_hold": 0, "max_hold": 0,  "gate": "none"},
    {"name": "hold2",       "n_opp": 1, "min_hold": 2, "max_hold": 0,  "gate": "none"},
    {"name": "hold3",       "n_opp": 1, "min_hold": 3, "max_hold": 0,  "gate": "none"},
    {"name": "hold5",       "n_opp": 1, "min_hold": 5, "max_hold": 0,  "gate": "none"},
    {"name": "max20",       "n_opp": 1, "min_hold": 0, "max_hold": 20, "gate": "none"},
    {"name": "max30",       "n_opp": 1, "min_hold": 0, "max_hold": 30, "gate": "none"},
    {"name": "p6_exit",     "n_opp": 1, "min_hold": 0, "max_hold": 0,  "gate": "p6_and"},
    {"name": "osc_exit",    "n_opp": 1, "min_hold": 0, "max_hold": 0,  "gate": "osc_and"},
    {"name": "opp2_hold2",  "n_opp": 2, "min_hold": 2, "max_hold": 0,  "gate": "none"},
    {"name": "opp2_max30",  "n_opp": 2, "min_hold": 0, "max_hold": 30, "gate": "none"},
    {"name": "hold2_max30", "n_opp": 1, "min_hold": 2, "max_hold": 30, "gate": "none"},
    {"name": "opp2_p6",     "n_opp": 2, "min_hold": 0, "max_hold": 0,  "gate": "p6_and"},
    {"name": "hold2_p6",    "n_opp": 1, "min_hold": 2, "max_hold": 0,  "gate": "p6_and"},
]

# -- Phase 8 baselines (for comparison) ----------------------------------------

PHASE8_BASELINES = {
    "EURUSD_4": {"oos_pf": 27.72, "oos_trades": 68, "oos_wr": 72.1,
                 "best_n": 3, "best_cd": 20},
    "EURUSD_5": {"oos_pf": 22.03, "oos_trades": 20, "oos_wr": 65.0,
                 "best_n": 4, "best_cd": 20},
    "GBPJPY":   {"oos_pf": 38.01, "oos_trades": 48, "oos_wr": 77.1,
                 "best_n": 5, "best_cd": 30},
    "EURAUD":   {"oos_pf": 18.32, "oos_trades": 37, "oos_wr": 73.0,
                 "best_n": 5, "best_cd": 20},
}


# -- Data loading (copied from phase8_sweep.py) --------------------------------

def _load_renko_all_indicators(renko_file: str, include_mk: bool) -> pd.DataFrame:
    """Load Renko data, add standard + Phase 6 + BC L1 oscillator + BC L3 MACD indicators."""
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


# -- Gate pre-computation (copied from phase8_sweep.py) -------------------------

def _compute_all_gates(df: pd.DataFrame, p6_gate_name: str,
                       sess_start: int, adx_thresh: int) -> dict:
    """Pre-compute gate arrays for a single fixed entry config."""
    from renko.phase6_sweep import _compute_gate_arrays as _p6_gate

    gates = {}

    # Session filter
    hours = df.index.hour
    ok = hours >= sess_start
    gates["sess"] = (ok, ok)

    # Vol ratio (symmetric, NaN-pass)
    vr = df["vol_ratio"].values
    vr_nan = np.isnan(vr)
    vol_ok = vr_nan | (vr <= VOL_MAX)
    gates["vol"] = (vol_ok, vol_ok)

    # Renko ADX (symmetric, NaN-pass)
    adx = df["adx"].values
    adx_nan = np.isnan(adx)
    adx_ok = adx_nan | (adx >= adx_thresh)
    gates["radx"] = (adx_ok, adx_ok)

    # Phase 6 gate
    gates["p6"] = _p6_gate(df, p6_gate_name)

    # STO + TSO combined
    sto_mf = df["_bc_sto_mf"].values
    sto_ll = df["_bc_sto_ll"].values
    sto_nan = np.isnan(sto_mf) | np.isnan(sto_ll)
    sto_long  = sto_nan | (sto_mf > sto_ll)
    sto_short = sto_nan | (sto_mf < sto_ll)

    tso_pink = df["_bc_tso_pink"].values.astype(float)
    tso_nan = np.isnan(tso_pink)
    tso_long  = tso_nan | (tso_pink > 0.5)
    tso_short = tso_nan | (tso_pink < 0.5)

    gates["sto_tso"] = (sto_long & tso_long, sto_short & tso_short)

    # MACD_LC combined
    macd_st = df["_bc_macd_state"].values
    bc_lc   = df["_bc_lc"].values
    ms_nan = np.isnan(macd_st)
    lc_nan = np.isnan(bc_lc)
    ms_int  = np.where(ms_nan, -1, macd_st).astype(int)
    ms_long  = ms_nan | np.isin(ms_int, [0, 3])
    ms_short = ms_nan | np.isin(ms_int, [1, 2])
    lc_long  = lc_nan | (bc_lc > 0)
    lc_short = lc_nan | (bc_lc < 0)
    gates["macd_lc"] = (ms_long & lc_long, ms_short & lc_short)

    return gates


def _combine_entry_gates(gates: dict, osc_name) -> tuple:
    """AND-combine all entry gate layers. Returns (combined_long, combined_short)."""
    n = len(gates["vol"][0])
    cl = np.ones(n, dtype=bool)
    cs = np.ones(n, dtype=bool)

    for key in ["sess", "vol", "radx", "p6"]:
        gl, gs = gates[key]
        cl &= gl
        cs &= gs

    if osc_name is not None:
        ol, os_ = gates[osc_name]
        cl &= ol
        cs &= os_

    return cl, cs


# -- Signal generator (modified with exit params) ------------------------------

def _generate_signals(df, n_bricks, cooldown, gate_long_ok, gate_short_ok,
                      n_opp=1, min_hold=0, max_hold=0, exit_gate="none",
                      p6_long_ok=None, p6_short_ok=None,
                      osc_long_ok=None, osc_short_ok=None):
    """R007 logic with parameterized exit strategy."""
    n        = len(df)
    brick_up = df["brick_up"].values

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    in_position       = False
    trade_dir         = 0
    last_r001_bar     = -999_999
    bars_since_entry  = 0
    n_consecutive_opp = 0
    warmup            = max(n_bricks + 1, 200)

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # -- EXIT LOGIC (parameterized) ----------------------------------------
        if in_position:
            bars_since_entry += 1

            is_opp = (trade_dir == 1 and not up) or (trade_dir == -1 and up)

            # Track consecutive opposing bricks
            if is_opp:
                n_consecutive_opp += 1
            else:
                n_consecutive_opp = 0

            # Check forced exit by max_hold
            force_exit = (max_hold > 0) and (bars_since_entry >= max_hold)

            # Check opposing-brick exit
            opp_exit = False
            if n_consecutive_opp >= n_opp and bars_since_entry >= min_hold:
                if exit_gate == "none":
                    opp_exit = True
                elif exit_gate == "p6_and":
                    # P6 gate must have flipped against position
                    if trade_dir == 1 and p6_short_ok[i]:
                        opp_exit = True
                    elif trade_dir == -1 and p6_long_ok[i]:
                        opp_exit = True
                elif exit_gate == "osc_and":
                    if trade_dir == 1 and osc_short_ok[i]:
                        opp_exit = True
                    elif trade_dir == -1 and osc_long_ok[i]:
                        opp_exit = True

            do_exit = force_exit or opp_exit

            if do_exit:
                long_exit[i]  = (trade_dir == 1)
                short_exit[i] = (trade_dir == -1)
                in_position       = False
                trade_dir         = 0
                bars_since_entry  = 0
                n_consecutive_opp = 0

        if in_position:
            continue

        # -- ENTRY LOGIC (unchanged from Phase 8) ------------------------------
        prev          = brick_up[i - n_bricks : i]
        prev_all_up   = bool(np.all(prev))
        prev_all_down = bool(not np.any(prev))

        if prev_all_up and not up:
            cand = -1; is_r002 = True
        elif prev_all_down and up:
            cand = 1;  is_r002 = True
        else:
            if (i - last_r001_bar) < cooldown:
                continue
            window   = brick_up[i - n_bricks + 1 : i + 1]
            all_up   = bool(np.all(window))
            all_down = bool(not np.any(window))
            if all_up:
                cand = 1;  is_r002 = False
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
        in_position       = True
        trade_dir         = cand
        bars_since_entry  = 0
        n_consecutive_opp = 0
        if not is_r002:
            last_r001_bar = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df


# -- Backtest runner (copied from phase8_sweep.py) -----------------------------

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


# -- Worker --------------------------------------------------------------------

def run_instrument_sweep(name: str, config: dict) -> list:
    print(f"[{name}] Loading Renko + all indicators...", flush=True)
    df = _load_renko_all_indicators(config["renko_file"], config["include_mk"])
    print(f"[{name}] Ready -- {len(df)} bricks", flush=True)

    # Compute gates for fixed entry stack
    gates = _compute_all_gates(df, config["p6_gate"],
                               config["sess_start"], config["adx_thresh"])

    # Combined entry gate
    gate_long, gate_short = _combine_entry_gates(gates, config["osc"])

    # Raw P6 and osc gate arrays for exit checks
    p6_long_ok,  p6_short_ok  = gates["p6"]
    if config["osc"] is not None:
        osc_long_ok, osc_short_ok = gates[config["osc"]]
    else:
        n = len(df)
        osc_long_ok  = np.ones(n, dtype=bool)
        osc_short_ok = np.ones(n, dtype=bool)

    keys         = list(PARAM_GRID.keys())
    param_combos = [dict(zip(keys, v)) for v in itertools.product(*PARAM_GRID.values())]

    total = len(EXIT_CONFIGS) * len(param_combos)
    done  = 0
    results = []

    osc_label = config["osc"] if config["osc"] else "none"
    stack_label = (f"s{config['sess_start']}_a{config['adx_thresh']}_"
                   f"{config['p6_gate']}_{osc_label}")

    for exit_cfg in EXIT_CONFIGS:
        for pc in param_combos:
            df_sig = _generate_signals(
                df.copy(),
                n_bricks      = pc["n_bricks"],
                cooldown      = pc["cooldown"],
                gate_long_ok  = gate_long,
                gate_short_ok = gate_short,
                n_opp         = exit_cfg["n_opp"],
                min_hold      = exit_cfg["min_hold"],
                max_hold      = exit_cfg["max_hold"],
                exit_gate     = exit_cfg["gate"],
                p6_long_ok    = p6_long_ok,
                p6_short_ok   = p6_short_ok,
                osc_long_ok   = osc_long_ok,
                osc_short_ok  = osc_short_ok,
            )

            is_r  = _run_backtest(df_sig, config["is_start"],  config["is_end"],
                                  config["commission"], config["capital"])
            oos_r = _run_backtest(df_sig, config["oos_start"], config["oos_end"],
                                  config["commission"], config["capital"])

            is_pf  = is_r["pf"]
            oos_pf = oos_r["pf"]
            decay  = ((oos_pf - is_pf) / is_pf * 100) \
                     if is_pf > 0 and not math.isinf(is_pf) else float("nan")

            results.append({
                "instrument":   name,
                "entry_stack":  stack_label,
                "exit_config":  exit_cfg["name"],
                "n_opp":        exit_cfg["n_opp"],
                "min_hold":     exit_cfg["min_hold"],
                "max_hold":     exit_cfg["max_hold"],
                "exit_gate":    exit_cfg["gate"],
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
            if done % 60 == 0 or done == total:
                print(
                    f"[{name}] {done:>4}/{total} | {exit_cfg['name']:<14} "
                    f"n={pc['n_bricks']} cd={pc['cooldown']:>2} | "
                    f"IS PF={is_pf:>6.2f} T={is_r['trades']:>4} | "
                    f"OOS PF={oos_pf:>6.2f} T={oos_r['trades']:>4}",
                    flush=True,
                )

    print(f"[{name}] Complete -- {len(results)} results", flush=True)
    return results


# -- Summary -------------------------------------------------------------------

def _summarize(all_results: list) -> None:
    for inst in ["EURUSD_4", "EURUSD_5", "GBPJPY", "EURAUD"]:
        inst_res = [r for r in all_results if r["instrument"] == inst]
        if not inst_res:
            continue

        base = PHASE8_BASELINES[inst]
        cfg  = INSTRUMENTS[inst]

        print(f"\n{'='*90}")
        print(f"  {cfg['label']}  |  Entry: {inst_res[0]['entry_stack']}")
        print(f"  Phase 8 baseline: OOS PF {base['oos_pf']} "
              f"({base['oos_trades']}t, WR {base['oos_wr']}%) "
              f"@ n={base['best_n']},cd={base['best_cd']}")
        print(f"{'='*90}")

        viable = [r for r in inst_res if r["oos_trades"] >= 15]
        viable.sort(key=lambda r: (r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6),
                    reverse=True)

        # Top 20
        print(f"\n  Top 20 (OOS trades >= 15):")
        print(f"  {'Exit Config':<14} {'n':>2} {'cd':>3} | {'IS PF':>7} {'T':>5} | "
              f"{'OOS PF':>7} {'T':>5} {'WR%':>6} {'Decay':>7}")
        print(f"  {'-'*75}")
        for r in viable[:20]:
            beat = " <<P8" if r["oos_pf"] > base["oos_pf"] else ""
            dec_s = f"{r['decay_pct']:>+6.1f}%" if not math.isnan(r["decay_pct"]) else "   NaN"
            print(f"  {r['exit_config']:<14} {r['n_bricks']:>2} {r['cooldown']:>3} | "
                  f"{r['is_pf']:>7.2f} {r['is_trades']:>5} | "
                  f"{r['oos_pf']:>7.2f} {r['oos_trades']:>5} {r['oos_wr']:>6.1f}% "
                  f"{dec_s}{beat}")

        # Avg OOS PF by exit config
        print(f"\n  By exit config (avg OOS PF, viable):")
        exit_names = list(dict.fromkeys(r["exit_config"] for r in inst_res))
        for ename in exit_names:
            ev = [r for r in viable if r["exit_config"] == ename]
            if ev:
                avg = sum(r["oos_pf"] for r in ev if not math.isinf(r["oos_pf"])) / len(ev)
                avg_t = sum(r["oos_trades"] for r in ev) / len(ev)
                marker = " <<P8" if avg > base["oos_pf"] else ""
                print(f"    {ename:<14} avg PF={avg:>7.2f}  avg T={avg_t:>6.1f}  N={len(ev):>3}{marker}")

        # By exit dimension
        print(f"\n  By n_opp (avg OOS PF, viable):")
        for no in [1, 2, 3]:
            nv = [r for r in viable if r["n_opp"] == no]
            if nv:
                avg = sum(r["oos_pf"] for r in nv if not math.isinf(r["oos_pf"])) / len(nv)
                print(f"    n_opp={no}       avg PF={avg:>7.2f}  N={len(nv):>3}")

        print(f"\n  By min_hold (avg OOS PF, viable):")
        for mh in [0, 2, 3, 5]:
            mv = [r for r in viable if r["min_hold"] == mh]
            if mv:
                avg = sum(r["oos_pf"] for r in mv if not math.isinf(r["oos_pf"])) / len(mv)
                print(f"    min_hold={mh}    avg PF={avg:>7.2f}  N={len(mv):>3}")

        print(f"\n  By max_hold (avg OOS PF, viable):")
        for xh in [0, 20, 30]:
            xv = [r for r in viable if r["max_hold"] == xh]
            if xv:
                avg = sum(r["oos_pf"] for r in xv if not math.isinf(r["oos_pf"])) / len(xv)
                print(f"    max_hold={xh:<3}   avg PF={avg:>7.2f}  N={len(xv):>3}")

        print(f"\n  By exit_gate (avg OOS PF, viable):")
        for eg in ["none", "p6_and", "osc_and"]:
            gv = [r for r in viable if r["exit_gate"] == eg]
            if gv:
                avg = sum(r["oos_pf"] for r in gv if not math.isinf(r["oos_pf"])) / len(gv)
                print(f"    gate={eg:<8}  avg PF={avg:>7.2f}  N={len(gv):>3}")

    # Cross-instrument best
    print(f"\n{'='*90}")
    print("  Overall best per instrument (single best OOS PF, trades >= 15)")
    print(f"{'='*90}")
    for inst in ["EURUSD_4", "EURUSD_5", "GBPJPY", "EURAUD"]:
        viable = [r for r in all_results
                  if r["instrument"] == inst and r["oos_trades"] >= 15]
        if not viable:
            continue
        best = max(viable, key=lambda r: r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6)
        base = PHASE8_BASELINES[inst]
        beat = "BEATS P8" if best["oos_pf"] > base["oos_pf"] else ""
        print(f"  {INSTRUMENTS[inst]['label']:<16} OOS PF={best['oos_pf']:>7.2f} "
              f"T={best['oos_trades']:>4} WR={best['oos_wr']:>5.1f}% "
              f"| exit={best['exit_config']:<14} n={best['n_bricks']} cd={best['cooldown']} "
              f"| {beat}")


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-parallel", action="store_true")
    args = parser.parse_args()

    out_path = ROOT / "ai_context" / "phase9_results.json"
    out_path.parent.mkdir(exist_ok=True)

    n_params = len(list(itertools.product(*PARAM_GRID.values())))
    n_exits  = len(EXIT_CONFIGS)
    total_all = n_exits * n_params * len(INSTRUMENTS)

    print("Phase 9: Exit Strategy Optimization")
    print(f"  Mode           : Pure Renko (no candle data)")
    print(f"  Exit configs   : {n_exits}")
    print(f"  Param combos   : {n_params}")
    print(f"  Instruments    : {len(INSTRUMENTS)}")
    print(f"  Total runs     : {total_all} ({total_all * 2} IS+OOS backtests)")
    print(f"  Output         : {out_path}")
    print()
    for name, cfg in INSTRUMENTS.items():
        osc_label = cfg["osc"] if cfg["osc"] else "none"
        print(f"  {cfg['label']:<16} entry: s{cfg['sess_start']}_a{cfg['adx_thresh']}_"
              f"{cfg['p6_gate']}_{osc_label}")
    print()

    all_results: list = []

    if args.no_parallel:
        for name, config in INSTRUMENTS.items():
            all_results.extend(run_instrument_sweep(name, config))
    else:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(run_instrument_sweep, name, config): name
                for name, config in INSTRUMENTS.items()
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
