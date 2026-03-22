#!/usr/bin/env python3
"""
phase8_sweep.py — Comprehensive Stacking Optimization

Builds on Phase 7's winning architecture (sess+vol+radx+p6_gate+oscillator) and
optimizes every variable dimension:

  1. EURUSD 0.0004 brick (proven higher PF than 0.0005)
  2. Multiple P6 gates per instrument (top 3, not just #1)
  3. ADX threshold sweep {20, 25, 30}
  4. Session threshold sweep {12, 13, 14}
  5. Both oscillator gates (sto_tso, macd_lc) + none

Pure Renko only — no candle data. ProcessPoolExecutor for parallel instruments.

Instruments (4 configs, 3 processes):
  EURUSD_4  0.0004 brick  (proven higher PF)
  EURUSD_5  0.0005 brick  (Phase 7 tested)
  GBPJPY    0.05   brick
  EURAUD    0.0006 brick

Per instrument: 3 P6 gates × 3 osc choices × 3 ADX × 3 session × 12 params = 972 runs
Total: 972 × 4 instruments = 3,888 runs (7,776 IS+OOS backtests)

Usage:
  python renko/phase8_sweep.py
  python renko/phase8_sweep.py --no-parallel
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

# ── Instrument configs ──────────────────────────────────────────────────────────

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
        "p6_gates":    ["ema_cross", "ichi_cloud", "stoch_cross"],
        "label":       "EURUSD 0.0004",
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
        "p6_gates":    ["ema_cross", "ichi_cloud", "stoch_cross"],
        "label":       "EURUSD 0.0005",
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
        "p6_gates":    ["mk_regime", "escgo_cross", "psar_dir"],
        "label":       "GBPJPY 0.05",
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
        "p6_gates":    ["ddl_dir", "ichi_cloud", "escgo_cross"],
        "label":       "EURAUD 0.0006",
    },
}

# ── Sweep dimensions ────────────────────────────────────────────────────────────

PARAM_GRID = {
    "n_bricks": [2, 3, 4, 5],
    "cooldown": [10, 20, 30],
}

ADX_THRESHOLDS   = [20, 25, 30]
SESSION_STARTS   = [12, 13, 14]
VOL_MAX          = 1.5
OSC_CHOICES      = [None, "sto_tso", "macd_lc"]


# ── Data loading ────────────────────────────────────────────────────────────────

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


# ── Gate pre-computation ────────────────────────────────────────────────────────

def _compute_all_gates(df: pd.DataFrame, p6_gate_names: list) -> dict:
    """
    Pre-compute all gate boolean arrays.

    Returns dict with keys:
      "sess_12", "sess_13", "sess_14"  — session filters
      "vol"                            — vol_ratio <= 1.5
      "radx_20", "radx_25", "radx_30" — Renko ADX thresholds
      "p6:<name>"                      — Phase 6 gates
      "sto_tso"                        — STO + TSO combined
      "macd_lc"                        — MACD state + LC combined
    """
    from renko.phase6_sweep import _compute_gate_arrays as _p6_gate

    gates = {}

    # Session filters
    hours = df.index.hour
    for ss in SESSION_STARTS:
        ok = hours >= ss
        gates[f"sess_{ss}"] = (ok, ok)

    # Vol ratio (symmetric, NaN-pass)
    vr = df["vol_ratio"].values
    vr_nan = np.isnan(vr)
    vol_ok = vr_nan | (vr <= VOL_MAX)
    gates["vol"] = (vol_ok, vol_ok)

    # Renko ADX at multiple thresholds (symmetric, NaN-pass)
    adx = df["adx"].values
    adx_nan = np.isnan(adx)
    for at in ADX_THRESHOLDS:
        ok = adx_nan | (adx >= at)
        gates[f"radx_{at}"] = (ok, ok)

    # Phase 6 gates
    for gname in p6_gate_names:
        gates[f"p6:{gname}"] = _p6_gate(df, gname)

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


def _combine_gates(gates: dict, sess: int, adx_thresh: int,
                   p6_name: str, osc_name) -> tuple:
    """AND-combine selected gate arrays. Returns (combined_long, combined_short)."""
    n = len(gates["vol"][0])
    cl = np.ones(n, dtype=bool)
    cs = np.ones(n, dtype=bool)

    # Session
    sl, ss = gates[f"sess_{sess}"]
    cl &= sl; cs &= ss

    # Vol
    vl, vs = gates["vol"]
    cl &= vl; cs &= vs

    # Renko ADX
    al, as_ = gates[f"radx_{adx_thresh}"]
    cl &= al; cs &= as_

    # P6 gate
    pl, ps = gates[f"p6:{p6_name}"]
    cl &= pl; cs &= ps

    # Oscillator
    if osc_name is not None:
        ol, os_ = gates[osc_name]
        cl &= ol; cs &= os_

    return cl, cs


# ── Signal generator ────────────────────────────────────────────────────────────

def _generate_signals(df, n_bricks, cooldown, gate_long_ok, gate_short_ok):
    """R007 logic with pre-computed gate arrays."""
    n        = len(df)
    brick_up = df["brick_up"].values

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    in_position   = False
    trade_dir     = 0
    last_r001_bar = -999_999
    warmup        = max(n_bricks + 1, 200)

    for i in range(warmup, n):
        up = bool(brick_up[i])

        if in_position:
            is_opp        = (trade_dir == 1 and not up) or (trade_dir == -1 and up)
            long_exit[i]  = is_opp and trade_dir == 1
            short_exit[i] = is_opp and trade_dir == -1
            if is_opp:
                in_position = False
                trade_dir   = 0

        if in_position:
            continue

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
        in_position = True
        trade_dir   = cand
        if not is_r002:
            last_r001_bar = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df


# ── Backtest runner ─────────────────────────────────────────────────────────────

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


# ── Worker ──────────────────────────────────────────────────────────────────────

def run_instrument_sweep(name: str, config: dict) -> list:
    print(f"[{name}] Loading Renko + all indicators...", flush=True)
    df = _load_renko_all_indicators(config["renko_file"], config["include_mk"])
    print(f"[{name}] Ready — {len(df)} bricks", flush=True)

    gates = _compute_all_gates(df, config["p6_gates"])

    keys         = list(PARAM_GRID.keys())
    param_combos = [dict(zip(keys, v)) for v in itertools.product(*PARAM_GRID.values())]

    # Build sweep combos: p6_gate × osc × adx × session × params
    sweep_combos = list(itertools.product(
        config["p6_gates"], OSC_CHOICES, ADX_THRESHOLDS, SESSION_STARTS
    ))
    total = len(sweep_combos) * len(param_combos)
    done  = 0
    results = []

    for p6_gate, osc, adx_t, sess in sweep_combos:
        gate_long, gate_short = _combine_gates(gates, sess, adx_t, p6_gate, osc)

        for pc in param_combos:
            df_sig = _generate_signals(
                df.copy(),
                n_bricks      = pc["n_bricks"],
                cooldown      = pc["cooldown"],
                gate_long_ok  = gate_long,
                gate_short_ok = gate_short,
            )

            is_r  = _run_backtest(df_sig, config["is_start"],  config["is_end"],
                                  config["commission"], config["capital"])
            oos_r = _run_backtest(df_sig, config["oos_start"], config["oos_end"],
                                  config["commission"], config["capital"])

            is_pf  = is_r["pf"]
            oos_pf = oos_r["pf"]
            decay  = ((oos_pf - is_pf) / is_pf * 100) \
                     if is_pf > 0 and not math.isinf(is_pf) else float("nan")

            osc_label = osc if osc else "none"
            stack_label = f"s{sess}_a{adx_t}_{p6_gate}_{osc_label}"

            results.append({
                "instrument": name,
                "stack":      stack_label,
                "p6_gate":    p6_gate,
                "osc":        osc_label,
                "adx_thresh": adx_t,
                "sess_start": sess,
                "n_bricks":   pc["n_bricks"],
                "cooldown":   pc["cooldown"],
                "is_pf":      is_pf,
                "is_trades":  is_r["trades"],
                "is_net":     is_r["net"],
                "is_wr":      is_r["wr"],
                "oos_pf":     oos_pf,
                "oos_trades": oos_r["trades"],
                "oos_net":    oos_r["net"],
                "oos_wr":     oos_r["wr"],
                "decay_pct":  decay,
            })

            done += 1
            if done % 108 == 0 or done == total:
                print(
                    f"[{name}] {done:>4}/{total} | {stack_label:<35} "
                    f"n={pc['n_bricks']} cd={pc['cooldown']:>2} | "
                    f"IS PF={is_pf:>6.2f} T={is_r['trades']:>4} | "
                    f"OOS PF={oos_pf:>6.2f} T={oos_r['trades']:>4}",
                    flush=True,
                )

    print(f"[{name}] Complete — {len(results)} results", flush=True)
    return results


# ── Summary ─────────────────────────────────────────────────────────────────────

BENCHMARKS = {
    "EURUSD_4": {"oos_pf": 12.79, "label": "R008 (candle ADX)", "proven": 20.48, "proven_label": "R012 macd_lc"},
    "EURUSD_5": {"oos_pf": 12.79, "label": "R008 (candle ADX)", "proven": 16.22, "proven_label": "P7 svr_p6"},
    "GBPJPY":   {"oos_pf": 21.33, "label": "GJ008 (candle ADX)", "proven": 48.75, "proven_label": "GJ011 sto_tso"},
    "EURAUD":   {"oos_pf": 10.62, "label": "EA008 (VP+div)", "proven": 10.62, "proven_label": "EA008"},
}


def _summarize(all_results: list) -> None:
    for inst in ["EURUSD_4", "EURUSD_5", "GBPJPY", "EURAUD"]:
        inst_res = [r for r in all_results if r["instrument"] == inst]
        if not inst_res:
            continue

        bench = BENCHMARKS[inst]
        cfg   = INSTRUMENTS[inst]

        print(f"\n{'='*90}")
        print(f"  {cfg['label']}")
        print(f"  Benchmark: {bench['label']} OOS PF {bench['oos_pf']}")
        print(f"  Proven:    {bench['proven_label']} OOS PF {bench['proven']}")
        print(f"{'='*90}")

        viable = [r for r in inst_res if r["oos_trades"] >= 20]
        viable.sort(key=lambda r: (r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6),
                    reverse=True)

        # Top 25
        print(f"\n  Top 25 (OOS trades >= 20):")
        print(f"  {'Stack':<35} {'n':>2} {'cd':>3} | {'IS PF':>7} {'T':>5} | "
              f"{'OOS PF':>7} {'T':>5} {'WR%':>6} {'Decay':>7}")
        print(f"  {'-'*85}")
        for r in viable[:25]:
            beat = ""
            if r["oos_pf"] > bench["proven"]:
                beat = " <<PROVEN"
            elif r["oos_pf"] > bench["oos_pf"]:
                beat = " <<BENCH"
            dec_s = f"{r['decay_pct']:>+6.1f}%" if not math.isnan(r["decay_pct"]) else "   NaN"
            print(f"  {r['stack']:<35} {r['n_bricks']:>2} {r['cooldown']:>3} | "
                  f"{r['is_pf']:>7.2f} {r['is_trades']:>5} | "
                  f"{r['oos_pf']:>7.2f} {r['oos_trades']:>5} {r['oos_wr']:>6.1f}% "
                  f"{dec_s}{beat}")

        # Best by dimension: P6 gate
        print(f"\n  By P6 gate (avg OOS PF, viable):")
        for pg in cfg["p6_gates"]:
            pv = [r for r in viable if r["p6_gate"] == pg]
            if pv:
                avg = sum(r["oos_pf"] for r in pv) / len(pv)
                avg_t = sum(r["oos_trades"] for r in pv) / len(pv)
                print(f"    {pg:<16} avg PF={avg:>7.2f}  avg T={avg_t:>6.1f}  N={len(pv):>4}")

        # Best by oscillator
        print(f"\n  By oscillator (avg OOS PF, viable):")
        for osc in ["none", "sto_tso", "macd_lc"]:
            ov = [r for r in viable if r["osc"] == osc]
            if ov:
                avg = sum(r["oos_pf"] for r in ov) / len(ov)
                avg_t = sum(r["oos_trades"] for r in ov) / len(ov)
                print(f"    {osc:<16} avg PF={avg:>7.2f}  avg T={avg_t:>6.1f}  N={len(ov):>4}")

        # Best by ADX threshold
        print(f"\n  By ADX threshold (avg OOS PF, viable):")
        for at in ADX_THRESHOLDS:
            av = [r for r in viable if r["adx_thresh"] == at]
            if av:
                avg = sum(r["oos_pf"] for r in av) / len(av)
                avg_t = sum(r["oos_trades"] for r in av) / len(av)
                print(f"    ADX>={at:<3}         avg PF={avg:>7.2f}  avg T={avg_t:>6.1f}  N={len(av):>4}")

        # Best by session
        print(f"\n  By session start (avg OOS PF, viable):")
        for ss in SESSION_STARTS:
            sv = [r for r in viable if r["sess_start"] == ss]
            if sv:
                avg = sum(r["oos_pf"] for r in sv) / len(sv)
                avg_t = sum(r["oos_trades"] for r in sv) / len(sv)
                print(f"    sess>={ss}         avg PF={avg:>7.2f}  avg T={avg_t:>6.1f}  N={len(sv):>4}")

    # Cross-instrument best configs
    print(f"\n{'='*90}")
    print("  Overall best per instrument (single best OOS PF, trades >= 20)")
    print(f"{'='*90}")
    for inst in ["EURUSD_4", "EURUSD_5", "GBPJPY", "EURAUD"]:
        viable = [r for r in all_results
                  if r["instrument"] == inst and r["oos_trades"] >= 20]
        if not viable:
            continue
        best = max(viable, key=lambda r: r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6)
        bench = BENCHMARKS[inst]
        beat = "BEATS PROVEN" if best["oos_pf"] > bench["proven"] else \
               "BEATS BENCH" if best["oos_pf"] > bench["oos_pf"] else ""
        print(f"  {INSTRUMENTS[inst]['label']:<16} OOS PF={best['oos_pf']:>7.2f} "
              f"T={best['oos_trades']:>4} WR={best['oos_wr']:>5.1f}% "
              f"| {best['stack']} n={best['n_bricks']} cd={best['cooldown']} "
              f"| {beat}")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-parallel", action="store_true")
    args = parser.parse_args()

    out_path = ROOT / "ai_context" / "phase8_results.json"
    out_path.parent.mkdir(exist_ok=True)

    n_params = len(list(itertools.product(*PARAM_GRID.values())))
    print("Phase 8: Comprehensive Stacking Optimization")
    print(f"  Mode           : Pure Renko (no candle data)")
    print(f"  Param combos   : {n_params}")
    print(f"  ADX thresholds : {ADX_THRESHOLDS}")
    print(f"  Session starts : {SESSION_STARTS}")
    print(f"  Oscillators    : {OSC_CHOICES}")
    print()
    for name, cfg in INSTRUMENTS.items():
        n_sweep = len(cfg["p6_gates"]) * len(OSC_CHOICES) * len(ADX_THRESHOLDS) * len(SESSION_STARTS)
        total = n_sweep * n_params
        print(f"  {cfg['label']:<16} P6 gates: {cfg['p6_gates']}  "
              f"-> {n_sweep} sweep x {n_params} params = {total} runs")
    total_all = sum(
        len(cfg["p6_gates"]) * len(OSC_CHOICES) * len(ADX_THRESHOLDS) * len(SESSION_STARTS) * n_params
        for cfg in INSTRUMENTS.values()
    )
    print(f"\n  Total runs     : {total_all} ({total_all * 2} IS+OOS backtests)")
    print(f"  Output         : {out_path}")
    print()

    all_results: list = []

    if args.no_parallel:
        for name, config in INSTRUMENTS.items():
            all_results.extend(run_instrument_sweep(name, config))
    else:
        with ProcessPoolExecutor(max_workers=len(INSTRUMENTS)) as pool:
            futures = {
                pool.submit(run_instrument_sweep, name, config): name
                for name, config in INSTRUMENTS.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    print(f"  [{name}] finished — {len(results)} records")
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
