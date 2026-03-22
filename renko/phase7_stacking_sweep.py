#!/usr/bin/env python3
"""
phase7_stacking_sweep.py — Stacking Sweep: Base Filters + Phase 6 + Phase 3-5 Gates

Tests systematic stacking of Renko-native base filters (session, vol_ratio, Renko ADX)
with Phase 6 top indicator gates and Phase 3-5 oscillator gates (sto_tso, macd_lc).

Pure Renko only — no candle data. All indicators computed from Renko OHLCV.

Stacking layers:
  Base filters (from renko/indicators.py):
    sess  — hour >= 13 UTC
    vol   — vol_ratio <= 1.5
    radx  — Renko ADX >= 25

  Phase 6 top gate (instrument-specific):
    EURUSD: ema_cross   (avg OOS PF 6.45 in Phase 6)
    GBPJPY: mk_regime   (avg OOS PF 12.53 in Phase 6)
    EURAUD: ddl_dir      (avg OOS PF 5.75 in Phase 6)

  Phase 3-5 oscillator gates:
    sto_tso  — STO MainForce > LifeLine AND TSO pink=True (long) / opposite (short)
    macd_lc  — MACD state in {0,3} AND LC > 0 (long) / state in {1,2} AND LC < 0 (short)

14 stack configs × 12 param combos × 3 instruments = 504 runs (1008 IS+OOS backtests)

Usage:
  python renko/phase7_stacking_sweep.py
  python renko/phase7_stacking_sweep.py --no-parallel
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
    "EURUSD": {
        "renko_file":  "OANDA_EURUSD, 1S renko 0.0005.csv",
        "is_start":    "2022-05-18",
        "is_end":      "2025-09-30",
        "oos_start":   "2025-10-01",
        "oos_end":     "2026-03-05",
        "commission":  0.0046,
        "capital":     1000.0,
        "include_mk":  False,
        "p6_gate":     "ema_cross",
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
        "p6_gate":     "mk_regime",
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
        "p6_gate":     "ddl_dir",
    },
}

# ── Stack configurations ────────────────────────────────────────────────────────

STACK_CONFIGS = [
    # Baseline (no filters — same as Phase 6 baseline)
    {"name": "baseline",        "sess": False, "vol": False, "radx": False, "p6": False, "osc": None},

    # Individual base filters
    {"name": "sess",            "sess": True,  "vol": False, "radx": False, "p6": False, "osc": None},
    {"name": "sess_vol",        "sess": True,  "vol": True,  "radx": False, "p6": False, "osc": None},
    {"name": "sess_vol_radx",   "sess": True,  "vol": True,  "radx": True,  "p6": False, "osc": None},

    # Base (sess+vol) + oscillator gates
    {"name": "sv_sto_tso",      "sess": True,  "vol": True,  "radx": False, "p6": False, "osc": "sto_tso"},
    {"name": "sv_macd_lc",      "sess": True,  "vol": True,  "radx": False, "p6": False, "osc": "macd_lc"},
    {"name": "svr_sto_tso",     "sess": True,  "vol": True,  "radx": True,  "p6": False, "osc": "sto_tso"},
    {"name": "svr_macd_lc",     "sess": True,  "vol": True,  "radx": True,  "p6": False, "osc": "macd_lc"},

    # Base + Phase 6 top gate
    {"name": "sv_p6",           "sess": True,  "vol": True,  "radx": False, "p6": True,  "osc": None},
    {"name": "svr_p6",          "sess": True,  "vol": True,  "radx": True,  "p6": True,  "osc": None},

    # Full stacks (base + P6 + oscillator)
    {"name": "sv_p6_sto",       "sess": True,  "vol": True,  "radx": False, "p6": True,  "osc": "sto_tso"},
    {"name": "sv_p6_mlc",       "sess": True,  "vol": True,  "radx": False, "p6": True,  "osc": "macd_lc"},
    {"name": "svr_p6_sto",      "sess": True,  "vol": True,  "radx": True,  "p6": True,  "osc": "sto_tso"},
    {"name": "svr_p6_mlc",      "sess": True,  "vol": True,  "radx": True,  "p6": True,  "osc": "macd_lc"},
]

PARAM_GRID = {
    "n_bricks": [2, 3, 4, 5],
    "cooldown": [10, 20, 30],
}

SESSION_START = 13
VOL_MAX       = 1.5
RADX_THRESHOLD = 25


# ── Data loading ────────────────────────────────────────────────────────────────

def _load_renko_all_indicators(renko_file: str, include_mk: bool) -> pd.DataFrame:
    """Load Renko data, add standard + Phase 6 + BC L1 oscillator + BC L3 MACD indicators."""
    sys.path.insert(0, str(ROOT))
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    from renko.phase6_enrichment import add_phase6_indicators

    # BC L1 oscillator indicators (for sto_tso gate)
    from indicators.blackcat1402.bc_l1_swing_trade_oscillator import (
        calc_bc_swing_trade_oscillator,
    )
    from indicators.blackcat1402.bc_l1_trend_swing_oscillator import (
        calc_bc_trend_swing_oscillator,
    )
    # BC L3 MACD Wave Signal Pro (for macd_lc gate)
    from indicators.blackcat1402.blackcat_l3_macd_wave_signal_pro import (
        calc_bc_l3_macd_wave_signal_pro,
    )

    df = load_renko_export(renko_file)
    add_renko_indicators(df)
    add_phase6_indicators(df, include_mk=include_mk)

    # STO — uses High, Low, Close (capitalized)
    try:
        sto = calc_bc_swing_trade_oscillator(df)
        df["_bc_sto_mf"] = sto["bc_sto_main_force"].shift(1).values
        df["_bc_sto_ll"] = sto["bc_sto_life_line"].shift(1).values
    except Exception as e:
        print(f"  WARN: STO failed: {e}")
        df["_bc_sto_mf"] = np.nan
        df["_bc_sto_ll"] = np.nan

    # TSO — uses High, Low, Close, Open (capitalized)
    try:
        tso = calc_bc_trend_swing_oscillator(df)
        df["_bc_tso_pink"] = tso["bc_tso_pink_hist"].shift(1).values
    except Exception as e:
        print(f"  WARN: TSO failed: {e}")
        df["_bc_tso_pink"] = np.nan

    # MACD Wave Signal Pro — needs lowercase columns
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

def _compute_all_gate_arrays(df: pd.DataFrame, p6_gate_name: str) -> dict:
    """
    Pre-compute all gate boolean arrays.

    Returns dict with keys:
      "sess"     — session >= 13 UTC
      "vol"      — vol_ratio <= 1.5 (NaN-pass)
      "radx"     — Renko ADX >= 25 (NaN-pass)
      "p6"       — Phase 6 top gate (instrument-specific)
      "sto_tso"  — STO + TSO combined
      "macd_lc"  — MACD state + LC combined

    Each value is (gate_long_ok, gate_short_ok) — two bool arrays of len(df).
    """
    from renko.phase6_sweep import _compute_gate_arrays as _p6_gate

    gates = {}

    # ── Session filter (symmetric) ──────────────────────────────────────────
    hours = df.index.hour
    sess_ok = hours >= SESSION_START
    gates["sess"] = (sess_ok, sess_ok)

    # ── Vol ratio filter (symmetric, NaN-pass) ─────────────────────────────
    vr = df["vol_ratio"].values
    vr_nan = np.isnan(vr)
    vol_ok = vr_nan | (vr <= VOL_MAX)
    gates["vol"] = (vol_ok, vol_ok)

    # ── Renko ADX filter (symmetric, NaN-pass) ─────────────────────────────
    adx = df["adx"].values
    adx_nan = np.isnan(adx)
    radx_ok = adx_nan | (adx >= RADX_THRESHOLD)
    gates["radx"] = (radx_ok, radx_ok)

    # ── Phase 6 top gate (instrument-specific, directional) ────────────────
    gates["p6"] = _p6_gate(df, p6_gate_name)

    # ── STO + TSO combined gate ────────────────────────────────────────────
    sto_mf = df["_bc_sto_mf"].values
    sto_ll = df["_bc_sto_ll"].values
    sto_nan = np.isnan(sto_mf) | np.isnan(sto_ll)
    sto_long  = sto_nan | (sto_mf > sto_ll)
    sto_short = sto_nan | (sto_mf < sto_ll)

    tso_pink = df["_bc_tso_pink"].values.astype(float)
    tso_nan = np.isnan(tso_pink)
    tso_long  = tso_nan | (tso_pink > 0.5)  # True = bullish
    tso_short = tso_nan | (tso_pink < 0.5)  # False = bearish

    gates["sto_tso"] = (sto_long & tso_long, sto_short & tso_short)

    # ── MACD_LC combined gate ──────────────────────────────────────────────
    macd_st = df["_bc_macd_state"].values
    bc_lc   = df["_bc_lc"].values

    ms_nan = np.isnan(macd_st)
    lc_nan = np.isnan(bc_lc)

    # MACD state: long = {0, 3} (rising), short = {1, 2} (falling)
    ms_int  = np.where(ms_nan, -1, macd_st).astype(int)
    ms_long  = ms_nan | np.isin(ms_int, [0, 3])
    ms_short = ms_nan | np.isin(ms_int, [1, 2])

    # LC: long = > 0, short = < 0
    lc_long  = lc_nan | (bc_lc > 0)
    lc_short = lc_nan | (bc_lc < 0)

    gates["macd_lc"] = (ms_long & lc_long, ms_short & lc_short)

    return gates


def _combine_stack(gates: dict, stack_cfg: dict) -> tuple:
    """AND-combine gate arrays based on stack config. Returns (combined_long, combined_short)."""
    n = len(gates["sess"][0])
    combined_long  = np.ones(n, dtype=bool)
    combined_short = np.ones(n, dtype=bool)

    if stack_cfg["sess"]:
        combined_long  &= gates["sess"][0]
        combined_short &= gates["sess"][1]

    if stack_cfg["vol"]:
        combined_long  &= gates["vol"][0]
        combined_short &= gates["vol"][1]

    if stack_cfg["radx"]:
        combined_long  &= gates["radx"][0]
        combined_short &= gates["radx"][1]

    if stack_cfg["p6"]:
        combined_long  &= gates["p6"][0]
        combined_short &= gates["p6"][1]

    osc = stack_cfg["osc"]
    if osc is not None:
        combined_long  &= gates[osc][0]
        combined_short &= gates[osc][1]

    return combined_long, combined_short


# ── Signal generator ────────────────────────────────────────────────────────────

def _generate_signals(df, n_bricks, cooldown, gate_long_ok, gate_short_ok):
    """R007 logic (R001 + R002 combined) with pre-computed gate arrays."""
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


# ── Worker: one instrument per process ──────────────────────────────────────────

def run_instrument_sweep(name: str, config: dict) -> list:
    print(f"[{name}] Loading Renko + all indicators...", flush=True)
    df = _load_renko_all_indicators(config["renko_file"], config["include_mk"])
    print(f"[{name}] Ready — {len(df)} bricks | P6 gate: {config['p6_gate']}", flush=True)

    # Pre-compute all gate arrays once
    gates = _compute_all_gate_arrays(df, config["p6_gate"])

    keys         = list(PARAM_GRID.keys())
    param_combos = [dict(zip(keys, v)) for v in itertools.product(*PARAM_GRID.values())]
    total        = len(STACK_CONFIGS) * len(param_combos)
    done         = 0
    results      = []

    for stack_cfg in STACK_CONFIGS:
        stack_name = stack_cfg["name"]

        # Combine gate arrays for this stack config
        gate_long_ok, gate_short_ok = _combine_stack(gates, stack_cfg)

        for pc in param_combos:
            df_sig = _generate_signals(
                df.copy(),
                n_bricks      = pc["n_bricks"],
                cooldown      = pc["cooldown"],
                gate_long_ok  = gate_long_ok,
                gate_short_ok = gate_short_ok,
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
                "instrument": name,
                "stack":      stack_name,
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
                "p6_gate":    config["p6_gate"],
            })

            done += 1
            if done % 12 == 0 or done == total:
                print(
                    f"[{name}] {done:>3}/{total} | {stack_name:<16} "
                    f"n={pc['n_bricks']} cd={pc['cooldown']:>2} | "
                    f"IS PF={is_pf:>6.2f} T={is_r['trades']:>4} | "
                    f"OOS PF={oos_pf:>6.2f} T={oos_r['trades']:>4} "
                    f"decay={decay:>+6.1f}%",
                    flush=True,
                )

    print(f"[{name}] Complete — {len(results)} results", flush=True)
    return results


# ── Summary ─────────────────────────────────────────────────────────────────────

# Reference benchmarks (proven OOS PFs)
BENCHMARKS = {
    "EURUSD": {"oos_pf": 12.79, "label": "R008 (candle ADX+vol+sess)", "p6_best": 6.45},
    "GBPJPY": {"oos_pf": 21.33, "label": "GJ008 (candle ADX+vol+sess)", "p6_best": 12.53},
    "EURAUD": {"oos_pf": 10.62, "label": "EA008 (sess+VP+div)", "p6_best": 5.75},
}

PROVEN_BEST = {
    "EURUSD": {"strategy": "R012 macd_lc", "oos_pf": 20.48},
    "GBPJPY": {"strategy": "GJ011 sto_tso", "oos_pf": 48.75},
    "EURAUD": {"strategy": "EA008 vp+div+sess", "oos_pf": 10.62},
}


def _summarize(all_results: list) -> None:
    for inst in ["EURUSD", "GBPJPY", "EURAUD"]:
        inst_res = [r for r in all_results if r["instrument"] == inst]
        if not inst_res:
            continue

        bench   = BENCHMARKS[inst]
        proven  = PROVEN_BEST[inst]

        print(f"\n{'='*85}")
        print(f"  {inst}")
        print(f"  Benchmark:   {bench['label']} OOS PF {bench['oos_pf']}")
        print(f"  Proven best: {proven['strategy']} OOS PF {proven['oos_pf']}")
        print(f"  Phase 6 top: {inst_res[0]['p6_gate']} avg OOS PF {bench['p6_best']}")
        print(f"{'='*85}")

        viable = [r for r in inst_res if r["oos_trades"] >= 20]
        viable.sort(key=lambda r: (r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6),
                    reverse=True)

        # Top 20 combos
        print(f"\n  Top 20 (OOS trades >= 20):")
        print(f"  {'Stack':<16} {'n':>2} {'cd':>3} | {'IS PF':>7} {'T':>5} | "
              f"{'OOS PF':>7} {'T':>5} {'WR%':>6} {'Decay':>7}")
        print(f"  {'-'*74}")
        for r in viable[:20]:
            beat = ""
            if r["oos_pf"] > proven["oos_pf"]:
                beat = " <<PROVEN"
            elif r["oos_pf"] > bench["oos_pf"]:
                beat = " <<BENCH"
            dec_s = f"{r['decay_pct']:>+6.1f}%" if not math.isnan(r["decay_pct"]) else "   NaN"
            print(f"  {r['stack']:<16} {r['n_bricks']:>2} {r['cooldown']:>3} | "
                  f"{r['is_pf']:>7.2f} {r['is_trades']:>5} | "
                  f"{r['oos_pf']:>7.2f} {r['oos_trades']:>5} {r['oos_wr']:>6.1f}% "
                  f"{dec_s}{beat}")

        # Stack averages
        print(f"\n  Stack averages (OOS trades >= 20):")
        print(f"  {'Stack':<16} {'Avg PF':>8} {'Avg T':>7} {'N':>4}  "
              f"{'vs bench':>9} {'vs proven':>10} {'Avg Decay':>10}")

        stack_avgs = {}
        for sc in STACK_CONFIGS:
            sname = sc["name"]
            sv = [r for r in viable if r["stack"] == sname]
            if sv:
                avg_pf   = sum(r["oos_pf"] for r in sv) / len(sv)
                avg_t    = sum(r["oos_trades"] for r in sv) / len(sv)
                valid_dec = [r["decay_pct"] for r in sv if not math.isnan(r["decay_pct"])]
                avg_dec  = sum(valid_dec) / len(valid_dec) if valid_dec else float("nan")
                stack_avgs[sname] = (avg_pf, avg_t, len(sv), avg_dec)

        for sname, (avg_pf, avg_t, n_v, avg_dec) in sorted(
            stack_avgs.items(), key=lambda x: x[1][0], reverse=True
        ):
            vs_bench  = f"{avg_pf - bench['oos_pf']:>+8.2f}"
            vs_proven = f"{avg_pf - proven['oos_pf']:>+9.2f}"
            dec_s     = f"{avg_dec:>+9.1f}%" if not math.isnan(avg_dec) else "       NaN"
            marker    = " *" if avg_pf > bench["oos_pf"] else ""
            print(f"  {sname:<16} {avg_pf:>8.2f} {avg_t:>7.1f} {n_v:>4}  "
                  f"{vs_bench} {vs_proven} {dec_s}{marker}")

    # Cross-instrument summary
    print(f"\n{'='*85}")
    print("  Cross-instrument summary (avg OOS PF per stack, viable combos)")
    print(f"{'='*85}")
    print(f"  {'Stack':<16} {'EURUSD':>10} {'GBPJPY':>10} {'EURAUD':>10} {'BeatBench':>10}")
    print(f"  {'-'*58}")

    for sc in STACK_CONFIGS:
        sname = sc["name"]
        wins = 0
        row  = [f"  {sname:<16}"]
        for inst in ["EURUSD", "GBPJPY", "EURAUD"]:
            sv = [r for r in all_results
                  if r["instrument"] == inst and r["stack"] == sname and r["oos_trades"] >= 20]
            if sv:
                avg_pf = sum(r["oos_pf"] for r in sv) / len(sv)
                bmark  = BENCHMARKS[inst]["oos_pf"]
                marker = "+" if avg_pf > bmark else " "
                row.append(f"{avg_pf:>9.2f}{marker}")
                if avg_pf > bmark:
                    wins += 1
            else:
                row.append(f"{'N/A':>10}")
        row.append(f"{wins:>10}")
        print("".join(row))


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-parallel", action="store_true",
                        help="Run instruments sequentially (debug mode)")
    args = parser.parse_args()

    out_path = ROOT / "ai_context" / "phase7_results.json"
    out_path.parent.mkdir(exist_ok=True)

    n_combos = len(list(itertools.product(*PARAM_GRID.values())))
    total_runs = len(STACK_CONFIGS) * n_combos * len(INSTRUMENTS) * 2

    print("Phase 7: Stacking Sweep — Base Filters + Phase 6 + Phase 3-5 Gates")
    print(f"  Mode           : Pure Renko (no candle data)")
    print(f"  Stack configs  : {len(STACK_CONFIGS)}")
    print(f"  Param combos   : {n_combos}")
    print(f"  Instruments    : {list(INSTRUMENTS.keys())}")
    for name, cfg in INSTRUMENTS.items():
        print(f"    {name}: P6 gate = {cfg['p6_gate']}")
    print(f"  Total IS+OOS   : {total_runs}")
    print(f"  Output         : {out_path}")
    print()

    for sc in STACK_CONFIGS:
        layers = []
        if sc["sess"]:  layers.append("sess>=13")
        if sc["vol"]:   layers.append("vol<=1.5")
        if sc["radx"]:  layers.append("radx>=25")
        if sc["p6"]:    layers.append("P6_gate")
        if sc["osc"]:   layers.append(sc["osc"])
        desc = " + ".join(layers) if layers else "(no filters)"
        print(f"  {sc['name']:<16} = {desc}")
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
