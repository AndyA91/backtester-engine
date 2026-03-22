#!/usr/bin/env python3
"""
mym_sweep.py — MYM Futures Renko Strategy Sweep

Micro E-mini Dow (CBOT:MYM1!) on RTH Renko charts.
Adapts the R007 signal generator for futures with forced close before NY session end.

Key differences from FX sweeps:
  - Forced close at 15:45 ET (before 16:00 RTH close)
  - No new entries after 15:30 ET
  - Session gate uses ET hours (not UTC)
  - Commission: $1.90 RT ≈ 0.00475% at ~40k price
  - P&L via qty=0.50 (MYM point value) → output in dollars

Brick sizes: 11, 12, 13, 14, 15
ProcessPoolExecutor: 5 workers (one per brick size).

Usage:
  python renko/mym_sweep.py
  python renko/mym_sweep.py --no-parallel
"""

import argparse
import contextlib
import io
import itertools
import json
import math
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# ── Instrument configs ──────────────────────────────────────────────────────────

INSTRUMENTS = {
    "MYM_11": {
        "renko_file": "CBOT_MINI_MYM1!, 1S renko 11.csv",
        "is_start":   "2025-08-07",
        "is_end":     "2025-12-31",
        "oos_start":  "2026-01-01",
        "oos_end":    "2026-03-19",
        "label":      "MYM brick 11",
    },
    "MYM_12": {
        "renko_file": "CBOT_MINI_MYM1!, 1S renko 12.csv",
        "is_start":   "2025-05-19",
        "is_end":     "2025-12-31",
        "oos_start":  "2026-01-01",
        "oos_end":    "2026-03-19",
        "label":      "MYM brick 12",
    },
    "MYM_13": {
        "renko_file": "CBOT_MINI_MYM1!, 1S renko 13.csv",
        "is_start":   "2025-04-09",
        "is_end":     "2025-12-31",
        "oos_start":  "2026-01-01",
        "oos_end":    "2026-03-19",
        "label":      "MYM brick 13",
    },
    "MYM_14": {
        "renko_file": "CBOT_MINI_MYM1!, 1S renko 14.csv",
        "is_start":   "2025-03-07",
        "is_end":     "2025-12-31",
        "oos_start":  "2026-01-01",
        "oos_end":    "2026-03-19",
        "label":      "MYM brick 14",
    },
    "MYM_15": {
        "renko_file": "CBOT_MINI_MYM1!, 1S renko 15.csv",
        "is_start":   "2025-01-06",
        "is_end":     "2025-12-31",
        "oos_start":  "2026-01-01",
        "oos_end":    "2026-03-19",
        "label":      "MYM brick 15",
    },
}

# Shared config for MYM futures
MYM_COMMISSION_PCT = 0.00475   # ~$1.90 RT at ~40k price
MYM_CAPITAL = 1000.0
MYM_QTY = 0.50                 # Point value: $0.50/point → P&L in dollars

# ── Sweep dimensions ────────────────────────────────────────────────────────────

PARAM_GRID = {
    "n_bricks": [3, 5, 7],
    "cooldown": [10, 20, 30],
}

ADX_THRESHOLDS   = [20, 25, 30]
SESSION_STARTS_ET = [0, 10, 11]    # ET hours: 0=no filter, 10=skip open, 11=mid-morning
VOL_MAX          = 1.5
OSC_CHOICES      = [None, "sto_tso", "macd_lc"]

P6_GATES = ["none", "stoch_cross", "escgo_cross", "mk_regime", "ema_cross", "psar_dir"]


# ── DST-aware ET conversion ────────────────────────────────────────────────────

def _compute_et_hours(dt_index: pd.DatetimeIndex) -> tuple:
    """
    Convert UTC DatetimeIndex to ET hours and minutes.

    DST rules (US Eastern):
      EDT: 2nd Sunday of March → 1st Sunday of November (UTC-4)
      EST: 1st Sunday of November → 2nd Sunday of March  (UTC-5)

    Returns (et_hours, et_minutes) as numpy int arrays.
    """
    # Vectorized DST detection: check if each date falls in EDT
    utc_hours = dt_index.hour
    utc_minutes = dt_index.minute

    # Pre-compute DST boundaries for relevant years
    dst_ranges = {}
    for year in range(2024, 2028):
        # 2nd Sunday of March
        mar1 = datetime(year, 3, 1)
        mar_sun2 = mar1 + timedelta(days=(6 - mar1.weekday()) % 7 + 7)
        # 1st Sunday of November
        nov1 = datetime(year, 11, 1)
        nov_sun1 = nov1 + timedelta(days=(6 - nov1.weekday()) % 7)
        dst_ranges[year] = (mar_sun2.date(), nov_sun1.date())

    years = dt_index.year

    # Default EST (UTC-5), switch to EDT (UTC-4) during summer
    offsets = np.full(len(dt_index), -5, dtype=int)

    for i in range(len(dt_index)):
        y = years[i]
        if y in dst_ranges:
            edt_start, edt_end = dst_ranges[y]
            bar_date = dt_index[i].date()
            if edt_start <= bar_date < edt_end:
                offsets[i] = -4

    et_total_minutes = (utc_hours + offsets) * 60 + utc_minutes
    et_hours = (et_total_minutes // 60) % 24
    et_minutes = et_total_minutes % 60

    return et_hours.values.astype(int), et_minutes.values.astype(int)


# ── Data loading ────────────────────────────────────────────────────────────────

def _load_renko_all_indicators(renko_file: str) -> pd.DataFrame:
    """Load Renko data + standard + Phase 6 + oscillator indicators."""
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
    add_phase6_indicators(df, include_mk=True)

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

def _compute_all_gates(df: pd.DataFrame, et_hours: np.ndarray) -> dict:
    """
    Pre-compute all gate boolean arrays.

    Uses ET hours for session gates (unlike FX which uses UTC).
    """
    from renko.phase6_sweep import _compute_gate_arrays as _p6_gate

    gates = {}

    # Session filters (ET hours, not UTC)
    for ss in SESSION_STARTS_ET:
        if ss == 0:
            ok = np.ones(len(df), dtype=bool)
        else:
            ok = et_hours >= ss
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
    for gname in P6_GATES:
        if gname == "none":
            n = len(df)
            gates["p6:none"] = (np.ones(n, dtype=bool), np.ones(n, dtype=bool))
        else:
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

    # Session (ET)
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

def _generate_signal_arrays(brick_up, n_bricks, cooldown, gate_long_ok, gate_short_ok,
                            et_hours, et_minutes):
    """
    R007 logic adapted for MYM futures. Returns raw signal arrays (no df copy).

    Additions vs FX version:
      - Forced close at 15:45 ET (before 16:00 RTH end)
      - No new entries after 15:30 ET
    """
    n = len(brick_up)

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
        h_et = et_hours[i]
        m_et = et_minutes[i]

        # ── Forced close at 15:45 ET ──
        if in_position and (h_et > 15 or (h_et == 15 and m_et >= 45)):
            long_exit[i]  = (trade_dir == 1)
            short_exit[i] = (trade_dir == -1)
            in_position = False
            trade_dir   = 0
            continue

        # ── Normal exit: opposing brick ──
        if in_position:
            is_opp        = (trade_dir == 1 and not up) or (trade_dir == -1 and up)
            long_exit[i]  = is_opp and trade_dir == 1
            short_exit[i] = is_opp and trade_dir == -1
            if is_opp:
                in_position = False
                trade_dir   = 0

        if in_position:
            continue

        # ── No new entries after 15:30 ET ──
        if h_et > 15 or (h_et == 15 and m_et >= 30):
            continue

        # ── R002: reversal after N consecutive bricks ──
        prev          = brick_up[i - n_bricks : i]
        prev_all_up   = bool(np.all(prev))
        prev_all_down = bool(not np.any(prev))

        if prev_all_up and not up:
            cand = -1; is_r002 = True
        elif prev_all_down and up:
            cand = 1;  is_r002 = True
        else:
            # ── R001: continuation ──
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

        # ── Gate check ──
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

    return long_entry, long_exit, short_entry, short_exit


# ── Backtest runner ─────────────────────────────────────────────────────────────

def _run_backtest(df_sig, start, end):
    sys.path.insert(0, str(ROOT))
    from engine import BacktestConfig, run_backtest_long_short

    cfg = BacktestConfig(
        initial_capital=MYM_CAPITAL,
        commission_pct=MYM_COMMISSION_PCT,
        slippage_ticks=0,
        qty_type="fixed",
        qty_value=MYM_QTY,
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
    df = _load_renko_all_indicators(config["renko_file"])
    print(f"[{name}] Ready — {len(df)} bricks", flush=True)

    # Pre-compute ET hours (once per worker)
    et_hours, et_minutes = _compute_et_hours(df.index)
    print(f"[{name}] ET hours computed (EDT/EST aware)", flush=True)

    gates = _compute_all_gates(df, et_hours)

    keys         = list(PARAM_GRID.keys())
    param_combos = [dict(zip(keys, v)) for v in itertools.product(*PARAM_GRID.values())]

    sweep_combos = list(itertools.product(
        P6_GATES, OSC_CHOICES, ADX_THRESHOLDS, SESSION_STARTS_ET
    ))
    total = len(sweep_combos) * len(param_combos)
    done  = 0
    results = []

    # Pre-allocate signal columns once (avoid df.copy() per combo)
    brick_up = df["brick_up"].values
    df["long_entry"]  = False
    df["long_exit"]   = False
    df["short_entry"] = False
    df["short_exit"]  = False

    for p6_gate, osc, adx_t, sess in sweep_combos:
        gate_long, gate_short = _combine_gates(gates, sess, adx_t, p6_gate, osc)

        for pc in param_combos:
            le, lx, se, sx = _generate_signal_arrays(
                brick_up,
                n_bricks      = pc["n_bricks"],
                cooldown      = pc["cooldown"],
                gate_long_ok  = gate_long,
                gate_short_ok = gate_short,
                et_hours      = et_hours,
                et_minutes    = et_minutes,
            )
            df["long_entry"]  = le
            df["long_exit"]   = lx
            df["short_entry"] = se
            df["short_exit"]  = sx

            is_r  = _run_backtest(df, config["is_start"],  config["is_end"])
            oos_r = _run_backtest(df, config["oos_start"], config["oos_end"])

            is_pf  = is_r["pf"]
            oos_pf = oos_r["pf"]
            decay  = ((oos_pf - is_pf) / is_pf * 100) \
                     if is_pf > 0 and not math.isinf(is_pf) else float("nan")

            osc_label = osc if osc else "none"
            stack_label = f"s{sess}_a{adx_t}_{p6_gate}_{osc_label}"

            results.append({
                "instrument": name,
                "brick":      int(name.split("_")[1]),
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
            if done % 250 == 0 or done == total:
                print(
                    f"[{name}] {done:>5}/{total} | {stack_label:<35} "
                    f"n={pc['n_bricks']} cd={pc['cooldown']:>2} | "
                    f"IS PF={is_pf:>7.2f} T={is_r['trades']:>4} | "
                    f"OOS PF={oos_pf:>7.2f} T={oos_r['trades']:>4}",
                    flush=True,
                )

    print(f"[{name}] Complete — {len(results)} results", flush=True)
    return results


# ── Summary ─────────────────────────────────────────────────────────────────────

def _summarize(all_results: list) -> None:
    for inst in sorted(INSTRUMENTS.keys()):
        inst_res = [r for r in all_results if r["instrument"] == inst]
        if not inst_res:
            continue

        cfg = INSTRUMENTS[inst]

        print(f"\n{'='*90}")
        print(f"  {cfg['label']}")
        print(f"{'='*90}")

        # OOS trades >= 10 for MYM (less data than FX)
        viable = [r for r in inst_res if r["oos_trades"] >= 10]
        viable.sort(key=lambda r: (r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6),
                    reverse=True)

        # Top 25
        print(f"\n  Top 25 (OOS trades >= 10):")
        print(f"  {'Stack':<35} {'n':>2} {'cd':>3} | {'IS PF':>7} {'T':>5} | "
              f"{'OOS PF':>7} {'T':>5} {'WR%':>6} {'Net$':>8} {'Decay':>7}")
        print(f"  {'-'*95}")
        for r in viable[:25]:
            dec_s = f"{r['decay_pct']:>+6.1f}%" if not math.isnan(r["decay_pct"]) else "   NaN"
            print(f"  {r['stack']:<35} {r['n_bricks']:>2} {r['cooldown']:>3} | "
                  f"{r['is_pf']:>7.2f} {r['is_trades']:>5} | "
                  f"{r['oos_pf']:>7.2f} {r['oos_trades']:>5} {r['oos_wr']:>6.1f}% "
                  f"{r['oos_net']:>8.2f} {dec_s}")

        # Best by P6 gate
        print(f"\n  By P6 gate (avg OOS PF, viable):")
        for pg in P6_GATES:
            pv = [r for r in viable if r["p6_gate"] == pg]
            if pv:
                avg = sum(r["oos_pf"] for r in pv if not math.isinf(r["oos_pf"])) / max(len([r for r in pv if not math.isinf(r["oos_pf"])]), 1)
                avg_t = sum(r["oos_trades"] for r in pv) / len(pv)
                print(f"    {pg:<16} avg PF={avg:>7.2f}  avg T={avg_t:>6.1f}  N={len(pv):>4}")

        # Best by oscillator
        print(f"\n  By oscillator (avg OOS PF, viable):")
        for osc in ["none", "sto_tso", "macd_lc"]:
            ov = [r for r in viable if r["osc"] == osc]
            if ov:
                avg = sum(r["oos_pf"] for r in ov if not math.isinf(r["oos_pf"])) / max(len([r for r in ov if not math.isinf(r["oos_pf"])]), 1)
                avg_t = sum(r["oos_trades"] for r in ov) / len(ov)
                print(f"    {osc:<16} avg PF={avg:>7.2f}  avg T={avg_t:>6.1f}  N={len(ov):>4}")

        # Best by ADX threshold
        print(f"\n  By ADX threshold (avg OOS PF, viable):")
        for at in ADX_THRESHOLDS:
            av = [r for r in viable if r["adx_thresh"] == at]
            if av:
                avg = sum(r["oos_pf"] for r in av if not math.isinf(r["oos_pf"])) / max(len([r for r in av if not math.isinf(r["oos_pf"])]), 1)
                avg_t = sum(r["oos_trades"] for r in av) / len(av)
                print(f"    ADX>={at:<3}         avg PF={avg:>7.2f}  avg T={avg_t:>6.1f}  N={len(av):>4}")

        # Best by session
        print(f"\n  By session start (avg OOS PF, viable):")
        for ss in SESSION_STARTS_ET:
            sv = [r for r in viable if r["sess_start"] == ss]
            if sv:
                avg = sum(r["oos_pf"] for r in sv if not math.isinf(r["oos_pf"])) / max(len([r for r in sv if not math.isinf(r["oos_pf"])]), 1)
                avg_t = sum(r["oos_trades"] for r in sv) / len(sv)
                label = f"sess>={ss}ET" if ss > 0 else "no filter"
                print(f"    {label:<16} avg PF={avg:>7.2f}  avg T={avg_t:>6.1f}  N={len(sv):>4}")

    # Cross-brick best
    print(f"\n{'='*90}")
    print("  Overall best per brick size (single best OOS PF, trades >= 10)")
    print(f"{'='*90}")
    for inst in sorted(INSTRUMENTS.keys()):
        viable = [r for r in all_results
                  if r["instrument"] == inst and r["oos_trades"] >= 10]
        if not viable:
            print(f"  {INSTRUMENTS[inst]['label']:<16} — no viable results")
            continue
        best = max(viable, key=lambda r: r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6)
        print(f"  {INSTRUMENTS[inst]['label']:<16} OOS PF={best['oos_pf']:>7.2f} "
              f"T={best['oos_trades']:>4} WR={best['oos_wr']:>5.1f}% "
              f"Net=${best['oos_net']:>7.2f} "
              f"| {best['stack']} n={best['n_bricks']} cd={best['cooldown']}")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-parallel", action="store_true")
    args = parser.parse_args()

    out_path = ROOT / "ai_context" / "mym_sweep_results.json"
    out_path.parent.mkdir(exist_ok=True)

    n_params = len(list(itertools.product(*PARAM_GRID.values())))
    n_sweep = len(P6_GATES) * len(OSC_CHOICES) * len(ADX_THRESHOLDS) * len(SESSION_STARTS_ET)
    total_per_brick = n_sweep * n_params
    total_all = total_per_brick * len(INSTRUMENTS)

    print("MYM Futures Renko Strategy Sweep")
    print(f"  Instrument     : CBOT:MYM1! (Micro E-mini Dow)")
    print(f"  Point value    : $0.50/point")
    print(f"  Commission     : ~$1.90 RT ({MYM_COMMISSION_PCT}%)")
    print(f"  Brick sizes    : 11, 12, 13, 14, 15")
    print(f"  Param combos   : {n_params}")
    print(f"  Sweep combos   : {n_sweep}")
    print(f"  Per brick      : {total_per_brick} runs")
    print(f"  Total runs     : {total_all} ({total_all * 2} IS+OOS backtests)")
    print(f"  Workers        : {len(INSTRUMENTS)} (one per brick)")
    print(f"  Output         : {out_path}")
    print(f"  Forced close   : 15:45 ET | No entries after 15:30 ET")
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
                    print(f"  [{name}] FAILED: {exc}")
                    traceback.print_exc()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved {len(all_results)} results -> {out_path}")

    _summarize(all_results)


if __name__ == "__main__":
    main()
