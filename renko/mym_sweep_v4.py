#!/usr/bin/env python3
"""
mym_sweep_v4.py -- MYM Phase 4: Renko-Specific Indicator Sweep

Phase 3 established the baseline:
  - ADX>=50, PSAR direction gate, n_bricks=9-10, cooldown=45-50
  - OOS PF up to ~105 on brick 15

Phase 4 tests five NEW indicator families that exploit Renko-specific
properties no price-based indicator captures:

  1. Brick Velocity gates   — Only enter when bricks form fast/slow
  2. Streak Momentum gates  — Only enter early in streaks (not exhausted)
  3. Session Context gates  — Skip range-exhausted or opening/closing sessions
  4. Adaptive Regime gate   — Unified trend/chop regime filter
  5. Exhaustion gate        — Block entries when trend is exhausted

Strategy:
  - Lock proven baseline: ADX>=50, session=0 (locked in v3)
  - Test each new gate family stacked ON TOP of the v3 best combos
  - Also test new gates as REPLACEMENTS for existing P6 gates
  - Use v3 best n_bricks/cooldown ranges + slight expansion

Grid: baseline P6 gates × new P4 gates × param combos
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
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# ---- Instrument configs (same as v1-v3) ----------------------------------------

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

MYM_COMMISSION_PCT = 0.00475
MYM_CAPITAL = 1000.0
MYM_QTY = 0.50

# ---- Phase 4 sweep dimensions --------------------------------------------------

PARAM_GRID = {
    "n_bricks": [8, 9, 10, 11],       # v3 sweet spot ± 1
    "cooldown": [40, 45, 50, 55],      # v3 sweet spot + slight push
}

ADX_THRESHOLDS = [45, 50]             # v3 locked top 2

# Baseline P6 gates (v3 winners)
P6_GATES = ["none", "psar_dir", "ema_cross"]

# NEW Phase 4 gates
P4_GATES = [
    "none",               # no P4 gate (baseline comparison)
    "vel_fast",           # brick velocity < 1.0 (faster than average)
    "vel_very_fast",      # brick velocity < 0.7 (much faster than average)
    "no_exhaust",         # exhaustion score < 0.5 (not exhausted)
    "no_exhaust_strict",  # exhaustion score < 0.3 (definitely fresh)
    "regime_trend",       # adaptive regime says trending
    "streak_fresh",       # streak age < 70th percentile (not over-extended)
    "no_opening",         # skip first 30 min of RTH
    "no_closing",         # skip last 30 min before forced close
    "range_ok",           # session range < 1.2x average (not extended)
]

# Oscillators (lock to v3 best)
OSC_CHOICES = [None, "sto_tso"]

# ---- Data loader with new indicators -------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mym_sweep import (
    _compute_et_hours,
    _generate_signal_arrays,
    _run_backtest,
)


def _load_renko_all_indicators_v4(renko_file: str):
    """Load Renko data + standard + Phase 6 + Phase 4 MYM indicators."""
    sys.path.insert(0, str(ROOT))
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    from renko.phase6_enrichment import add_phase6_indicators
    from renko.mym_enrichment import add_mym_indicators
    from indicators.blackcat1402.bc_l1_swing_trade_oscillator import (
        calc_bc_swing_trade_oscillator,
    )
    from indicators.blackcat1402.bc_l1_trend_swing_oscillator import (
        calc_bc_trend_swing_oscillator,
    )

    df = load_renko_export(renko_file)
    add_renko_indicators(df)
    add_phase6_indicators(df, include_mk=True)
    add_mym_indicators(df)  # NEW Phase 4 indicators

    # STO (reuse from v1)
    try:
        sto = calc_bc_swing_trade_oscillator(df)
        df["_bc_sto_mf"] = sto["bc_sto_main_force"].shift(1).values
        df["_bc_sto_ll"] = sto["bc_sto_life_line"].shift(1).values
    except Exception as e:
        print(f"  WARN: STO failed: {e}")
        df["_bc_sto_mf"] = np.nan
        df["_bc_sto_ll"] = np.nan

    # TSO (reuse from v1)
    try:
        tso = calc_bc_trend_swing_oscillator(df)
        df["_bc_tso_pink"] = tso["bc_tso_pink_hist"].shift(1).values
    except Exception as e:
        print(f"  WARN: TSO failed: {e}")
        df["_bc_tso_pink"] = np.nan

    return df


# ---- Gate pre-computation -------------------------------------------------------

def _compute_all_gates(df, et_hours):
    from renko.phase6_sweep import _compute_gate_arrays as _p6_gate

    gates = {}
    n = len(df)

    # Vol ratio (from v1)
    vr = df["vol_ratio"].values
    vr_nan = np.isnan(vr)
    vol_ok = vr_nan | (vr <= 1.5)
    gates["vol"] = (vol_ok, vol_ok)

    # ADX gates
    adx = df["adx"].values
    adx_nan = np.isnan(adx)
    for at in ADX_THRESHOLDS:
        ok = adx_nan | (adx >= at)
        gates[f"radx_{at}"] = (ok, ok)

    # P6 gates (baseline)
    for gname in P6_GATES:
        if gname == "none":
            gates["p6:none"] = (np.ones(n, dtype=bool), np.ones(n, dtype=bool))
        else:
            gates[f"p6:{gname}"] = _p6_gate(df, gname)

    # ── NEW Phase 4 gates ──

    # Velocity gates
    vel_ratio = df["vel_ratio"].values
    vr_nan = np.isnan(vel_ratio)
    gates["p4:vel_fast"] = (
        vr_nan | (vel_ratio < 1.0),
        vr_nan | (vel_ratio < 1.0),
    )
    gates["p4:vel_very_fast"] = (
        vr_nan | (vel_ratio < 0.7),
        vr_nan | (vel_ratio < 0.7),
    )

    # Exhaustion gates
    exhaust = df["exhaust_score"].values
    ex_nan = np.isnan(exhaust)
    gates["p4:no_exhaust"] = (
        ex_nan | (exhaust < 0.5),
        ex_nan | (exhaust < 0.5),
    )
    gates["p4:no_exhaust_strict"] = (
        ex_nan | (exhaust < 0.3),
        ex_nan | (exhaust < 0.3),
    )

    # Regime gate
    regime_trending = df["regime_trending"].values
    rt_nan = np.isnan(regime_trending)
    gates["p4:regime_trend"] = (
        rt_nan | (regime_trending > 0.5),
        rt_nan | (regime_trending > 0.5),
    )

    # Streak freshness gate
    streak_age = df["streak_age_pct"].values
    sa_nan = np.isnan(streak_age)
    gates["p4:streak_fresh"] = (
        sa_nan | (streak_age < 70),
        sa_nan | (streak_age < 70),
    )

    # Session gates
    sess_opening = df["sess_is_opening"].values
    so_nan = np.isnan(sess_opening)
    gates["p4:no_opening"] = (
        so_nan | (sess_opening < 0.5),
        so_nan | (sess_opening < 0.5),
    )

    sess_closing = df["sess_is_closing"].values
    sc_nan = np.isnan(sess_closing)
    gates["p4:no_closing"] = (
        sc_nan | (sess_closing < 0.5),
        sc_nan | (sess_closing < 0.5),
    )

    # Range exhaustion gate
    range_used = df["sess_range_used"].values
    ru_nan = np.isnan(range_used)
    gates["p4:range_ok"] = (
        ru_nan | (range_used < 1.2),
        ru_nan | (range_used < 1.2),
    )

    # STO + TSO oscillator (from v1)
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

    # P4:none (no P4 gate)
    gates["p4:none"] = (np.ones(n, dtype=bool), np.ones(n, dtype=bool))

    return gates


def _combine_gates(gates, adx_thresh, p6_name, p4_name, osc_name):
    n = len(gates["vol"][0])
    cl = np.ones(n, dtype=bool)
    cs = np.ones(n, dtype=bool)

    # Vol
    vl, vs = gates["vol"]
    cl &= vl; cs &= vs

    # ADX
    al, as_ = gates[f"radx_{adx_thresh}"]
    cl &= al; cs &= as_

    # P6 baseline gate
    pl, ps = gates[f"p6:{p6_name}"]
    cl &= pl; cs &= ps

    # P4 new gate
    p4l, p4s = gates[f"p4:{p4_name}"]
    cl &= p4l; cs &= p4s

    # Oscillator
    if osc_name is not None:
        ol, os_ = gates[osc_name]
        cl &= ol; cs &= os_

    return cl, cs


# ---- Worker --------------------------------------------------------------------

def run_instrument_sweep(name, config):
    print(f"[{name}] Loading Renko + ALL indicators (v4)...", flush=True)
    df = _load_renko_all_indicators_v4(config["renko_file"])
    print(f"[{name}] Ready -- {len(df)} bricks", flush=True)

    et_hours, et_minutes = _compute_et_hours(df.index)
    gates = _compute_all_gates(df, et_hours)

    keys = list(PARAM_GRID.keys())
    param_combos = [dict(zip(keys, v)) for v in itertools.product(*PARAM_GRID.values())]

    sweep_combos = list(itertools.product(P6_GATES, P4_GATES, OSC_CHOICES, ADX_THRESHOLDS))
    total = len(sweep_combos) * len(param_combos)
    done = 0
    results = []

    brick_up = df["brick_up"].values
    df["long_entry"]  = False
    df["long_exit"]   = False
    df["short_entry"] = False
    df["short_exit"]  = False

    for p6_gate, p4_gate, osc, adx_t in sweep_combos:
        gate_long, gate_short = _combine_gates(gates, adx_t, p6_gate, p4_gate, osc)

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
            stack_label = f"a{adx_t}_{p6_gate}_{p4_gate}_{osc_label}"

            results.append({
                "instrument": name,
                "brick":      int(name.split("_")[1]),
                "stack":      stack_label,
                "p6_gate":    p6_gate,
                "p4_gate":    p4_gate,
                "osc":        osc_label,
                "adx_thresh": adx_t,
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
            if done % 500 == 0 or done == total:
                print(
                    f"[{name}] {done:>5}/{total} | {stack_label:<45} "
                    f"n={pc['n_bricks']:>2} cd={pc['cooldown']:>2} | "
                    f"IS PF={is_pf:>7.2f} T={is_r['trades']:>4} | "
                    f"OOS PF={oos_pf:>7.2f} T={oos_r['trades']:>4}",
                    flush=True,
                )

    print(f"[{name}] Complete -- {len(results)} results", flush=True)
    return results


# ---- Summary -------------------------------------------------------------------

def _summarize(all_results):
    MIN_OOS_TRADES = 10

    for inst in sorted(INSTRUMENTS.keys()):
        inst_res = [r for r in all_results if r["instrument"] == inst]
        if not inst_res:
            continue

        cfg = INSTRUMENTS[inst]
        print(f"\n{'='*100}")
        print(f"  {cfg['label']}  (Phase 4 -- Renko-Specific Indicators)")
        print(f"{'='*100}")

        viable = [r for r in inst_res if r["oos_trades"] >= MIN_OOS_TRADES]
        viable.sort(key=lambda r: (r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6),
                    reverse=True)

        print(f"\n  Top 25 (OOS trades >= {MIN_OOS_TRADES}):")
        print(f"  {'Stack':<45} {'n':>2} {'cd':>3} | {'IS PF':>7} {'T':>5} | "
              f"{'OOS PF':>7} {'T':>5} {'WR%':>6} {'Net$':>8} {'Decay':>7}")
        print(f"  {'-'*100}")
        for r in viable[:25]:
            dec_s = f"{r['decay_pct']:>+6.1f}%" if not math.isnan(r["decay_pct"]) else "   NaN"
            print(f"  {r['stack']:<45} {r['n_bricks']:>2} {r['cooldown']:>3} | "
                  f"{r['is_pf']:>7.2f} {r['is_trades']:>5} | "
                  f"{r['oos_pf']:>7.2f} {r['oos_trades']:>5} {r['oos_wr']:>6.1f}% "
                  f"{r['oos_net']:>8.2f} {dec_s}")

        # ── By P4 gate (the NEW indicators) ──
        print(f"\n  By P4 gate (avg OOS PF, viable) — THE NEW INDICATORS:")
        for pg in P4_GATES:
            pv = [r for r in viable if r["p4_gate"] == pg and not math.isinf(r["oos_pf"])]
            if pv:
                avg = sum(r["oos_pf"] for r in pv) / len(pv)
                avg_t = sum(r["oos_trades"] for r in pv) / len(pv)
                best = max(pv, key=lambda r: r["oos_pf"])
                print(f"    {pg:<22} avg PF={avg:>7.2f}  avg T={avg_t:>6.1f}  "
                      f"N={len(pv):>4}  best={best['oos_pf']:>7.2f}")

        # ── By P6 gate ──
        print(f"\n  By P6 gate (avg OOS PF, viable):")
        for pg in P6_GATES:
            pv = [r for r in viable if r["p6_gate"] == pg and not math.isinf(r["oos_pf"])]
            if pv:
                avg = sum(r["oos_pf"] for r in pv) / len(pv)
                avg_t = sum(r["oos_trades"] for r in pv) / len(pv)
                print(f"    {pg:<16} avg PF={avg:>7.2f}  avg T={avg_t:>6.1f}  N={len(pv):>4}")

        # ── By n_bricks ──
        print(f"\n  By n_bricks (avg OOS PF, viable):")
        for nb in sorted(PARAM_GRID["n_bricks"]):
            nv = [r for r in viable if r["n_bricks"] == nb and not math.isinf(r["oos_pf"])]
            if nv:
                avg = sum(r["oos_pf"] for r in nv) / len(nv)
                avg_t = sum(r["oos_trades"] for r in nv) / len(nv)
                print(f"    n={nb:>2}: avg PF={avg:>7.2f}  avg T={avg_t:>6.1f}  N={len(nv):>4}")

        # ── By cooldown ──
        print(f"\n  By cooldown (avg OOS PF, viable):")
        for cd in sorted(PARAM_GRID["cooldown"]):
            cv = [r for r in viable if r["cooldown"] == cd and not math.isinf(r["oos_pf"])]
            if cv:
                avg = sum(r["oos_pf"] for r in cv) / len(cv)
                avg_t = sum(r["oos_trades"] for r in cv) / len(cv)
                print(f"    cd={cd}: avg PF={avg:>7.2f}  avg T={avg_t:>6.1f}  N={len(cv):>4}")

        # ── By oscillator ──
        print(f"\n  By oscillator (avg OOS PF, viable):")
        for osc in ["none", "sto_tso"]:
            ov = [r for r in viable if r["osc"] == osc and not math.isinf(r["oos_pf"])]
            if ov:
                avg = sum(r["oos_pf"] for r in ov) / len(ov)
                avg_t = sum(r["oos_trades"] for r in ov) / len(ov)
                print(f"    {osc:<16} avg PF={avg:>7.2f}  avg T={avg_t:>6.1f}  N={len(ov):>4}")

        # ── Best P4 gate per P6 gate ──
        print(f"\n  Best P4 gate for each P6 gate (top OOS PF):")
        for p6 in P6_GATES:
            p6v = [r for r in viable if r["p6_gate"] == p6 and not math.isinf(r["oos_pf"])]
            if p6v:
                best = max(p6v, key=lambda r: r["oos_pf"])
                print(f"    P6={p6:<12} + P4={best['p4_gate']:<20} "
                      f"OOS PF={best['oos_pf']:>7.2f} T={best['oos_trades']:>4} "
                      f"n={best['n_bricks']} cd={best['cooldown']}")

    # Cross-brick best
    print(f"\n{'='*100}")
    print("  Overall best per brick size (OOS PF, trades >= 10)")
    print(f"{'='*100}")
    for inst in sorted(INSTRUMENTS.keys()):
        viable = [r for r in all_results
                  if r["instrument"] == inst and r["oos_trades"] >= MIN_OOS_TRADES
                  and not math.isinf(r["oos_pf"])]
        if not viable:
            print(f"  {INSTRUMENTS[inst]['label']:<16} -- no viable results")
            continue
        best = max(viable, key=lambda r: r["oos_pf"])
        print(f"  {INSTRUMENTS[inst]['label']:<16} OOS PF={best['oos_pf']:>7.2f} "
              f"T={best['oos_trades']:>4} WR={best['oos_wr']:>5.1f}% "
              f"Net=${best['oos_net']:>7.2f} "
              f"| {best['stack']} n={best['n_bricks']} cd={best['cooldown']}")

    # Compare with v3
    print(f"\n{'='*100}")
    print("  Phase 3 vs Phase 4 comparison")
    print(f"{'='*100}")
    try:
        with open(ROOT / "ai_context" / "mym_sweep_v3_results.json") as f:
            v3_data = json.load(f)
        for brick in [11, 12, 13, 14, 15]:
            v3_best = [r for r in v3_data if r["brick"] == brick
                       and r["oos_trades"] >= 10 and not math.isinf(r["oos_pf"])]
            v4_best = [r for r in all_results if r["brick"] == brick
                       and r["oos_trades"] >= 10 and not math.isinf(r["oos_pf"])]
            v3_top = max(v3_best, key=lambda r: r["oos_pf"]) if v3_best else None
            v4_top = max(v4_best, key=lambda r: r["oos_pf"]) if v4_best else None
            if v3_top and v4_top:
                delta = (v4_top["oos_pf"] - v3_top["oos_pf"]) / v3_top["oos_pf"] * 100
                print(f"  Brick {brick}: v3={v3_top['oos_pf']:>7.2f}  v4={v4_top['oos_pf']:>7.2f}  "
                      f"delta={delta:>+6.1f}%  "
                      f"| {v4_top['stack']} n={v4_top['n_bricks']} cd={v4_top['cooldown']}")
    except FileNotFoundError:
        print("  (v3 results not found)")

    # NEW: P4 gate value-add analysis
    print(f"\n{'='*100}")
    print("  P4 Gate Value-Add Analysis (does adding a P4 gate improve over P4:none?)")
    print(f"{'='*100}")
    for inst in sorted(INSTRUMENTS.keys()):
        viable = [r for r in all_results
                  if r["instrument"] == inst and r["oos_trades"] >= MIN_OOS_TRADES
                  and not math.isinf(r["oos_pf"])]
        if not viable:
            continue

        cfg = INSTRUMENTS[inst]
        print(f"\n  {cfg['label']}:")

        baseline = [r for r in viable if r["p4_gate"] == "none"]
        if not baseline:
            continue
        baseline_avg = sum(r["oos_pf"] for r in baseline) / len(baseline)

        for pg in P4_GATES:
            if pg == "none":
                continue
            gated = [r for r in viable if r["p4_gate"] == pg]
            if gated:
                gated_avg = sum(r["oos_pf"] for r in gated) / len(gated)
                delta = (gated_avg - baseline_avg) / baseline_avg * 100 if baseline_avg > 0 else 0
                symbol = "+" if delta > 0 else ""
                print(f"    {pg:<22} avg PF={gated_avg:>7.2f}  "
                      f"vs baseline={baseline_avg:>7.2f}  "
                      f"delta={symbol}{delta:.1f}%")


# ---- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-parallel", action="store_true")
    args = parser.parse_args()

    out_path = ROOT / "ai_context" / "mym_sweep_v4_results.json"
    out_path.parent.mkdir(exist_ok=True)

    n_params = len(list(itertools.product(*PARAM_GRID.values())))
    n_sweep  = len(P6_GATES) * len(P4_GATES) * len(OSC_CHOICES) * len(ADX_THRESHOLDS)
    total_per_brick = n_sweep * n_params
    total_all = total_per_brick * len(INSTRUMENTS)

    print("MYM Phase 4 Sweep -- Renko-Specific Indicators")
    print(f"  n_bricks       : {PARAM_GRID['n_bricks']}")
    print(f"  cooldown       : {PARAM_GRID['cooldown']}")
    print(f"  ADX thresholds : {ADX_THRESHOLDS}")
    print(f"  P6 gates       : {P6_GATES}")
    print(f"  P4 gates (NEW) : {P4_GATES}")
    print(f"  Oscillators    : {['none'] + [o for o in OSC_CHOICES if o]}")
    print(f"  Per brick      : {total_per_brick} runs")
    print(f"  Total runs     : {total_all} ({total_all * 2} IS+OOS backtests)")
    print(f"  Workers        : {len(INSTRUMENTS)}")
    print(f"  Output         : {out_path}")
    print()

    all_results = []

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
                    print(f"  [{name}] finished -- {len(results)} records")
                except Exception as exc:
                    print(f"  [{name}] FAILED: {exc}")
                    traceback.print_exc()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved {len(all_results)} results -> {out_path}")

    _summarize(all_results)


if __name__ == "__main__":
    main()
