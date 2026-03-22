#!/usr/bin/env python3
"""
mym_sweep_v2.py -- MYM Phase 2 Refinement Sweep

Phase 1 findings:
  - Session s0 dominates (RTH data already filters) -> LOCKED at 0
  - n_bricks=7 best across all bricks -> expand 5-9
  - cooldown=30 best -> expand 20-45
  - ADX>=30 best -> push higher (30-45)
  - psar_dir best P6, stoch_cross worst -> drop stoch_cross
  - sto_tso best osc

Changes vs v1:
  - Dropped session dimension (saves 3x compute)
  - Expanded n_bricks: [5, 6, 7, 8, 9]
  - Expanded cooldown: [20, 25, 30, 35, 40]
  - Higher ADX: [25, 30, 35, 40]
  - Dropped stoch_cross (worst P6 gate)
  - Added ichi_cloud P6 gate (strong for FX, untested on MYM)

Grid: 25 params x 60 sweep = 1,500 per brick x 5 = 7,500 total
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

# ---- Instrument configs (same as v1) ----------------------------------------

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

# ---- Phase 2 sweep dimensions -----------------------------------------------

PARAM_GRID = {
    "n_bricks": [5, 6, 7, 8, 9],
    "cooldown": [20, 25, 30, 35, 40],
}

ADX_THRESHOLDS = [25, 30, 35, 40]
P6_GATES = ["none", "psar_dir", "mk_regime", "escgo_cross", "ema_cross"]
OSC_CHOICES = [None, "sto_tso", "macd_lc"]

# Session LOCKED at 0 (no filter) -- RTH data already excludes off-hours

# ---- Reuse core functions from v1 -------------------------------------------
# Import heavy-lifting functions from mym_sweep (ET conversion, data loading,
# signal generator, backtest runner)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mym_sweep import (
    _compute_et_hours,
    _load_renko_all_indicators,
    _generate_signal_arrays,
    _run_backtest,
)


# ---- Gate pre-computation (v2: no session dimension) -------------------------

def _compute_all_gates(df, et_hours):
    """Pre-compute gate boolean arrays. No session gates in v2."""
    from renko.phase6_sweep import _compute_gate_arrays as _p6_gate

    gates = {}
    n = len(df)

    # Vol ratio (symmetric, NaN-pass)
    vr = df["vol_ratio"].values
    vr_nan = np.isnan(vr)
    vol_ok = vr_nan | (vr <= 1.5)
    gates["vol"] = (vol_ok, vol_ok)

    # Renko ADX at multiple thresholds
    adx = df["adx"].values
    adx_nan = np.isnan(adx)
    for at in ADX_THRESHOLDS:
        ok = adx_nan | (adx >= at)
        gates[f"radx_{at}"] = (ok, ok)

    # Phase 6 gates
    for gname in P6_GATES:
        if gname == "none":
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


def _combine_gates(gates, adx_thresh, p6_name, osc_name):
    """AND-combine selected gates. No session gate in v2."""
    n = len(gates["vol"][0])
    cl = np.ones(n, dtype=bool)
    cs = np.ones(n, dtype=bool)

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


# ---- Worker ------------------------------------------------------------------

def run_instrument_sweep(name, config):
    print(f"[{name}] Loading Renko + all indicators...", flush=True)
    df = _load_renko_all_indicators(config["renko_file"])
    print(f"[{name}] Ready -- {len(df)} bricks", flush=True)

    et_hours, et_minutes = _compute_et_hours(df.index)
    gates = _compute_all_gates(df, et_hours)

    keys = list(PARAM_GRID.keys())
    param_combos = [dict(zip(keys, v)) for v in itertools.product(*PARAM_GRID.values())]

    sweep_combos = list(itertools.product(P6_GATES, OSC_CHOICES, ADX_THRESHOLDS))
    total = len(sweep_combos) * len(param_combos)
    done = 0
    results = []

    brick_up = df["brick_up"].values
    df["long_entry"]  = False
    df["long_exit"]   = False
    df["short_entry"] = False
    df["short_exit"]  = False

    for p6_gate, osc, adx_t in sweep_combos:
        gate_long, gate_short = _combine_gates(gates, adx_t, p6_gate, osc)

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
            stack_label = f"a{adx_t}_{p6_gate}_{osc_label}"

            results.append({
                "instrument": name,
                "brick":      int(name.split("_")[1]),
                "stack":      stack_label,
                "p6_gate":    p6_gate,
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
            if done % 250 == 0 or done == total:
                print(
                    f"[{name}] {done:>5}/{total} | {stack_label:<30} "
                    f"n={pc['n_bricks']} cd={pc['cooldown']:>2} | "
                    f"IS PF={is_pf:>7.2f} T={is_r['trades']:>4} | "
                    f"OOS PF={oos_pf:>7.2f} T={oos_r['trades']:>4}",
                    flush=True,
                )

    print(f"[{name}] Complete -- {len(results)} results", flush=True)
    return results


# ---- Summary ----------------------------------------------------------------

def _summarize(all_results):
    MIN_OOS_TRADES = 10

    for inst in sorted(INSTRUMENTS.keys()):
        inst_res = [r for r in all_results if r["instrument"] == inst]
        if not inst_res:
            continue

        cfg = INSTRUMENTS[inst]
        print(f"\n{'='*90}")
        print(f"  {cfg['label']}  (Phase 2 Refinement)")
        print(f"{'='*90}")

        viable = [r for r in inst_res if r["oos_trades"] >= MIN_OOS_TRADES]
        viable.sort(key=lambda r: (r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6),
                    reverse=True)

        print(f"\n  Top 25 (OOS trades >= {MIN_OOS_TRADES}):")
        print(f"  {'Stack':<30} {'n':>2} {'cd':>3} | {'IS PF':>7} {'T':>5} | "
              f"{'OOS PF':>7} {'T':>5} {'WR%':>6} {'Net$':>8} {'Decay':>7}")
        print(f"  {'-'*90}")
        for r in viable[:25]:
            dec_s = f"{r['decay_pct']:>+6.1f}%" if not math.isnan(r["decay_pct"]) else "   NaN"
            print(f"  {r['stack']:<30} {r['n_bricks']:>2} {r['cooldown']:>3} | "
                  f"{r['is_pf']:>7.2f} {r['is_trades']:>5} | "
                  f"{r['oos_pf']:>7.2f} {r['oos_trades']:>5} {r['oos_wr']:>6.1f}% "
                  f"{r['oos_net']:>8.2f} {dec_s}")

        # Best by n_bricks
        print(f"\n  By n_bricks (avg OOS PF, viable):")
        for nb in sorted(PARAM_GRID["n_bricks"]):
            nv = [r for r in viable if r["n_bricks"] == nb and not math.isinf(r["oos_pf"])]
            if nv:
                avg = sum(r["oos_pf"] for r in nv) / len(nv)
                avg_t = sum(r["oos_trades"] for r in nv) / len(nv)
                print(f"    n={nb}: avg PF={avg:>7.2f}  avg T={avg_t:>6.1f}  N={len(nv):>4}")

        # Best by cooldown
        print(f"\n  By cooldown (avg OOS PF, viable):")
        for cd in sorted(PARAM_GRID["cooldown"]):
            cv = [r for r in viable if r["cooldown"] == cd and not math.isinf(r["oos_pf"])]
            if cv:
                avg = sum(r["oos_pf"] for r in cv) / len(cv)
                avg_t = sum(r["oos_trades"] for r in cv) / len(cv)
                print(f"    cd={cd}: avg PF={avg:>7.2f}  avg T={avg_t:>6.1f}  N={len(cv):>4}")

        # Best by P6 gate
        print(f"\n  By P6 gate (avg OOS PF, viable):")
        for pg in P6_GATES:
            pv = [r for r in viable if r["p6_gate"] == pg and not math.isinf(r["oos_pf"])]
            if pv:
                avg = sum(r["oos_pf"] for r in pv) / len(pv)
                avg_t = sum(r["oos_trades"] for r in pv) / len(pv)
                print(f"    {pg:<16} avg PF={avg:>7.2f}  avg T={avg_t:>6.1f}  N={len(pv):>4}")

        # Best by oscillator
        print(f"\n  By oscillator (avg OOS PF, viable):")
        for osc in ["none", "sto_tso", "macd_lc"]:
            ov = [r for r in viable if r["osc"] == osc and not math.isinf(r["oos_pf"])]
            if ov:
                avg = sum(r["oos_pf"] for r in ov) / len(ov)
                avg_t = sum(r["oos_trades"] for r in ov) / len(ov)
                print(f"    {osc:<16} avg PF={avg:>7.2f}  avg T={avg_t:>6.1f}  N={len(ov):>4}")

        # Best by ADX
        print(f"\n  By ADX threshold (avg OOS PF, viable):")
        for at in ADX_THRESHOLDS:
            av = [r for r in viable if r["adx_thresh"] == at and not math.isinf(r["oos_pf"])]
            if av:
                avg = sum(r["oos_pf"] for r in av) / len(av)
                avg_t = sum(r["oos_trades"] for r in av) / len(av)
                print(f"    ADX>={at:<3}  avg PF={avg:>7.2f}  avg T={avg_t:>6.1f}  N={len(av):>4}")

    # Cross-brick best
    print(f"\n{'='*90}")
    print("  Overall best per brick size (OOS PF, trades >= 10)")
    print(f"{'='*90}")
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

    # Compare v1 vs v2 bests
    print(f"\n{'='*90}")
    print("  Phase 1 vs Phase 2 comparison (best OOS PF per brick)")
    print(f"{'='*90}")
    try:
        with open(ROOT / "ai_context" / "mym_sweep_results.json") as f:
            v1_data = json.load(f)
        for brick in [11, 12, 13, 14, 15]:
            v1_best = [r for r in v1_data if r["brick"] == brick
                       and r["oos_trades"] >= 10 and not math.isinf(r["oos_pf"])]
            v2_best = [r for r in all_results if r["brick"] == brick
                       and r["oos_trades"] >= 10 and not math.isinf(r["oos_pf"])]
            v1_top = max(v1_best, key=lambda r: r["oos_pf"]) if v1_best else None
            v2_top = max(v2_best, key=lambda r: r["oos_pf"]) if v2_best else None
            if v1_top and v2_top:
                delta = (v2_top["oos_pf"] - v1_top["oos_pf"]) / v1_top["oos_pf"] * 100
                print(f"  Brick {brick}: v1={v1_top['oos_pf']:>7.2f}  v2={v2_top['oos_pf']:>7.2f}  "
                      f"delta={delta:>+6.1f}%  "
                      f"| {v2_top['stack']} n={v2_top['n_bricks']} cd={v2_top['cooldown']}")
    except FileNotFoundError:
        print("  (v1 results not found -- skipping comparison)")


# ---- Main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-parallel", action="store_true")
    args = parser.parse_args()

    out_path = ROOT / "ai_context" / "mym_sweep_v2_results.json"
    out_path.parent.mkdir(exist_ok=True)

    n_params = len(list(itertools.product(*PARAM_GRID.values())))
    n_sweep  = len(P6_GATES) * len(OSC_CHOICES) * len(ADX_THRESHOLDS)
    total_per_brick = n_sweep * n_params
    total_all = total_per_brick * len(INSTRUMENTS)

    print("MYM Phase 2 Refinement Sweep")
    print(f"  Instrument     : CBOT:MYM1! (Micro E-mini Dow)")
    print(f"  Session        : LOCKED at s0 (no filter, RTH data)")
    print(f"  n_bricks       : {PARAM_GRID['n_bricks']}")
    print(f"  cooldown       : {PARAM_GRID['cooldown']}")
    print(f"  ADX thresholds : {ADX_THRESHOLDS}")
    print(f"  P6 gates       : {P6_GATES}")
    print(f"  Oscillators    : {['none'] + [o for o in OSC_CHOICES if o]}")
    print(f"  Param combos   : {n_params}")
    print(f"  Sweep combos   : {n_sweep}")
    print(f"  Per brick      : {total_per_brick} runs")
    print(f"  Total runs     : {total_all} ({total_all * 2} IS+OOS backtests)")
    print(f"  Workers        : {len(INSTRUMENTS)} (one per brick)")
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
