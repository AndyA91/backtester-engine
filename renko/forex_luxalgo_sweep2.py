#!/usr/bin/env python3
"""
forex_luxalgo_sweep2.py -- LuxAlgo sweep on USDJPY + GBPUSD

Same structure as forex_luxalgo_sweep.py (Parts A-D) but for the 2 remaining
live forex instruments. Reuses signal generators from the original sweep.

Usage:
    python renko/forex_luxalgo_sweep2.py
"""

import contextlib
import io
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
from renko.forex_luxalgo_sweep import (
    LUX_GATE_NAMES, STANDALONE_SIGNALS, PART_D_SIGNALS, PART_D_GATES,
    _gen_r001r002, _gen_standalone, _run_bt,
    _print_header, _print_row, _show_part, _show_per_instrument,
)

# -- Instrument configs --------------------------------------------------------

INSTRUMENTS = {
    "USDJPY": {
        "renko_file": "OANDA_USDJPY, 1S renko 0.05.csv",
        "is_start":   "2024-03-14",
        "is_end":     "2025-09-30",
        "oos_start":  "2025-10-01",
        "oos_end":    "2026-03-17",
        "oos_days":   168,
        "commission":  0.005,
        "capital":     1000.0,
        "include_mk":  True,
        "best_p6":    "stoch_cross",  # live UJ001 uses stoch_cross
    },
    "GBPUSD": {
        "renko_file": "OANDA_GBPUSD, 1S renko 0.0004.csv",
        "is_start":   "2023-01-27",
        "is_end":     "2025-09-30",
        "oos_start":  "2025-10-01",
        "oos_end":    "2026-03-17",
        "oos_days":   168,
        "commission":  0.005,
        "capital":     1000.0,
        "include_mk":  False,
        "best_p6":    "escgo_cross",  # live GU001 uses escgo_cross
    },
}


# -- Combo builders (same structure, different instruments) --------------------

def _build_part_a():
    combos = []
    for inst in INSTRUMENTS:
        for gate in LUX_GATE_NAMES:
            for n in [3, 4]:
                for cd in [10, 20]:
                    combos.append({
                        "part": "A", "inst": inst, "gate": gate,
                        "n_bricks": n, "cooldown": cd,
                        "label": f"{inst}_{gate}_n{n}_cd{cd}",
                    })
    return combos


def _build_part_b():
    combos = []
    for inst in INSTRUMENTS:
        for lux_gate in [None] + LUX_GATE_NAMES:
            for n in [3, 4]:
                for cd in [10, 20]:
                    lux_tag = lux_gate if lux_gate else "p6_only"
                    combos.append({
                        "part": "B", "inst": inst, "lux_gate": lux_gate,
                        "n_bricks": n, "cooldown": cd,
                        "label": f"{inst}_p6+{lux_tag}_n{n}_cd{cd}",
                    })
    return combos


def _build_part_c():
    combos = []
    for inst in INSTRUMENTS:
        for sig in STANDALONE_SIGNALS:
            for cd in [10, 20]:
                combos.append({
                    "part": "C", "inst": inst, "signal": sig,
                    "cooldown": cd,
                    "label": f"{inst}_{sig}_cd{cd}",
                })
    return combos


def _build_part_d():
    combos = []
    for inst in INSTRUMENTS:
        for sig in PART_D_SIGNALS:
            for gate in PART_D_GATES:
                for cd in [10, 20, 30]:
                    combos.append({
                        "part": "D", "inst": inst, "signal": sig,
                        "gate": gate, "cooldown": cd,
                        "label": f"{inst}_{sig}_{gate}_cd{cd}",
                    })
    return combos


# -- Worker (per-process caching) ----------------------------------------------

_w = {}


def _init_inst(inst_key):
    if inst_key in _w:
        return
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    from renko.phase6_enrichment import add_phase6_indicators
    from renko.luxalgo_indicators import add_luxalgo_indicators
    from renko.phase6_sweep import _compute_gate_arrays as _p6_gate

    cfg = INSTRUMENTS[inst_key]
    df = load_renko_export(cfg["renko_file"])
    add_renko_indicators(df)
    add_phase6_indicators(df, include_mk=cfg["include_mk"])
    add_luxalgo_indicators(df, include_knn=True, svm_vol_weight=0.4)

    n = len(df)
    p6_l, p6_s = _p6_gate(df, cfg["best_p6"])

    psar = df["psar_dir"].values
    psar_nan = np.isnan(psar)
    psar_l = psar_nan | (psar > 0)
    psar_s = psar_nan | (psar < 0)

    adx = df["adx"].values
    adx_nan = np.isnan(adx)
    adx25 = adx_nan | (adx >= 25)

    lux_g = {}
    rs = df["lux_rollseg_trend"].values
    rs_nan = np.isnan(rs)
    lux_g["rollseg_trend"] = (rs_nan | (rs > 0), rs_nan | (rs < 0))

    ink = df["lux_inertial_k"].values
    ind = df["lux_inertial_d"].values
    in_nan = np.isnan(ink) | np.isnan(ind)
    lux_g["inertial_cross"] = (in_nan | (ink > ind), in_nan | (ink < ind))

    svm = df["lux_svm_trend"].values
    svm_nan = np.isnan(svm)
    lux_g["svm_trend"] = (svm_nan | (svm > 0), svm_nan | (svm < 0))

    knn = df["lux_knn_bullish"].values.astype(float)
    knn_nan = np.isnan(knn)
    lux_g["knn_trend"] = (knn_nan | (knn > 0.5), knn_nan | (knn <= 0.5))

    bpb = df["lux_breakout_bull"].values
    bpr = df["lux_breakout_bear"].values
    bb_nan = np.isnan(bpb) | np.isnan(bpr)
    lux_g["breakout_bias"] = (bb_nan | (bpb > bpr), bb_nan | (bpr > bpb))

    _w[inst_key] = {
        "df": df,
        "p6": (p6_l, p6_s),
        "psar": (psar_l, psar_s),
        "adx25": (adx25, adx25),
        "lux_gates": lux_g,
    }


def _run_one(combo):
    inst = combo["inst"]
    _init_inst(inst)
    w = _w[inst]
    df = w["df"]
    cfg = INSTRUMENTS[inst]
    n = len(df)
    ones = np.ones(n, dtype=bool)

    if combo["part"] == "A":
        gl, gs = w["lux_gates"][combo["gate"]]
        le, lx, se, sx = _gen_r001r002(
            df, combo["n_bricks"], combo["cooldown"], gl, gs)
    elif combo["part"] == "B":
        p6_l, p6_s = w["p6"]
        if combo["lux_gate"] is not None:
            lux_l, lux_s = w["lux_gates"][combo["lux_gate"]]
            gl = p6_l & lux_l
            gs = p6_s & lux_s
        else:
            gl, gs = p6_l.copy(), p6_s.copy()
        le, lx, se, sx = _gen_r001r002(
            df, combo["n_bricks"], combo["cooldown"], gl, gs)
    elif combo["part"] == "C":
        le, lx, se, sx = _gen_standalone(
            df, combo["signal"], combo["cooldown"], ones, ones)
    elif combo["part"] == "D":
        if combo["gate"] == "psar":
            gl, gs = w["psar"]
        elif combo["gate"] == "adx25":
            gl, gs = w["adx25"]
        else:
            gl, gs = ones, ones
        le, lx, se, sx = _gen_standalone(
            df, combo["signal"], combo["cooldown"], gl, gs)

    is_r = _run_bt(df, le, lx, se, sx,
                   cfg["is_start"], cfg["is_end"], cfg["commission"], cfg["capital"])
    oos_r = _run_bt(df, le, lx, se, sx,
                    cfg["oos_start"], cfg["oos_end"], cfg["commission"], cfg["capital"])
    return combo, is_r, oos_r


# -- Main ----------------------------------------------------------------------

def main():
    combos = _build_part_a() + _build_part_b() + _build_part_c() + _build_part_d()
    total = len(combos)
    print(f"Forex LuxAlgo Sweep 2 (USDJPY+GBPUSD): {total} combos ({total*2} backtests) on {MAX_WORKERS} workers")

    all_results = []
    done = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run_one, c): c for c in combos}
        for fut in as_completed(futures):
            combo = futures[fut]
            try:
                combo_ret, is_r, oos_r = fut.result()
                inst = combo_ret["inst"]
                row = {
                    "part":       combo_ret["part"],
                    "inst":       inst,
                    "label":      combo_ret["label"],
                    "oos_days":   INSTRUMENTS[inst]["oos_days"],
                    "is_pf":      is_r["pf"],
                    "is_trades":  is_r["trades"],
                    "is_wr":      is_r["wr"],
                    "is_net":     is_r["net"],
                    "is_dd":      is_r["dd"],
                    "oos_pf":     oos_r["pf"],
                    "oos_trades": oos_r["trades"],
                    "oos_wr":     oos_r["wr"],
                    "oos_net":    oos_r["net"],
                    "oos_dd":     oos_r["dd"],
                }
                for k in combo_ret:
                    if k not in row:
                        row[k] = combo_ret[k]
                all_results.append(row)
            except Exception as e:
                print(f"  ERROR: {combo.get('label', '?')}: {e}")
            done += 1
            if done % 25 == 0 or done == total:
                print(f"  [{done}/{total}]")

    # Monkey-patch INSTRUMENTS into the reporting module
    import renko.forex_luxalgo_sweep as fls
    orig = fls.INSTRUMENTS
    fls.INSTRUMENTS = INSTRUMENTS

    _show_part(all_results, "A", "Part A - LuxAlgo Gates on R001+R002")
    _show_per_instrument(all_results, "A", "Part A - Per Instrument")
    _show_part(all_results, "B", "Part B - P6 + LuxAlgo Stacked")
    _show_per_instrument(all_results, "B", "Part B - Per Instrument")
    _show_part(all_results, "C", "Part C - Standalone LuxAlgo Signals")
    _show_per_instrument(all_results, "C", "Part C - Per Instrument")
    _show_part(all_results, "D", "Part D - Standalone + PSAR/ADX Gates")
    _show_per_instrument(all_results, "D", "Part D - Per Instrument")

    # Global top 20
    viable = [r for r in all_results if r["oos_trades"] >= 5 and r["oos_net"] > 0]
    viable.sort(key=lambda r: (r["oos_wr"], r["oos_net"]), reverse=True)
    print(f"\n{'='*65}")
    print(f"  GLOBAL TOP 20 ({len(viable)} viable / {len(all_results)} total)")
    print(f"{'='*65}")
    if viable:
        _print_header()
        for i, r in enumerate(viable[:20]):
            _print_row(r, i + 1)

    # Combined IS+OOS top 15
    for r in all_results:
        r["total_trades"] = r["is_trades"] + r["oos_trades"]
        r["total_net"] = r["is_net"] + r["oos_net"]
        if r["total_trades"] > 0:
            is_w = r["is_trades"] * r["is_wr"] / 100
            oos_w = r["oos_trades"] * r["oos_wr"] / 100
            r["total_wr"] = (is_w + oos_w) / r["total_trades"] * 100
        else:
            r["total_wr"] = 0

    viable2 = [r for r in all_results if r["oos_trades"] >= 5 and r["oos_net"] > 0]
    viable2.sort(key=lambda r: (r["total_wr"], r["total_net"]), reverse=True)
    print(f"\n{'='*130}")
    print(f"  COMBINED IS+OOS TOP 15 + DECAY")
    print(f"{'='*130}")
    hdr = (f"  {'#':>3} {'Pt':>2} {'Label':<46} | "
           f"{'IS PF':>7} {'IS T':>5} {'IS WR':>6} | "
           f"{'OOS PF':>7} {'OOS T':>5} {'OOS WR':>6} {'WR chg':>7} | "
           f"{'ALL T':>6} {'ALL WR':>6} {'ALL Net':>10}")
    print(hdr)
    print("  " + "-" * 126)
    for i, r in enumerate(viable2[:15]):
        is_pf = f"{r['is_pf']:.2f}" if r['is_pf'] < 9999 else "inf"
        oos_pf = f"{r['oos_pf']:.2f}" if r['oos_pf'] < 9999 else "inf"
        wr_chg = r['oos_wr'] - r['is_wr']
        print(f"  {i+1:>3} {r['part']:>2} {r['label']:<46} | "
              f"{is_pf:>7} {r['is_trades']:>5} {r['is_wr']:>5.1f}% | "
              f"{oos_pf:>7} {r['oos_trades']:>5} {r['oos_wr']:>5.1f}% {wr_chg:>+6.1f}% | "
              f"{r['total_trades']:>6} {r['total_wr']:>5.1f}% {r['total_net']:>10.2f}")

    fls.INSTRUMENTS = orig  # restore

    # Save JSON
    out_path = ROOT / "ai_context" / "forex_luxalgo_sweep2_results.json"
    serializable = []
    for r in all_results:
        sr = {k: v for k, v in r.items() if k not in ("total_trades", "total_net", "total_wr")}
        for k in ("is_pf", "oos_pf"):
            if isinstance(sr[k], float) and math.isinf(sr[k]):
                sr[k] = "inf"
        serializable.append(sr)
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved {len(all_results)} results -> {out_path}")


if __name__ == "__main__":
    main()
