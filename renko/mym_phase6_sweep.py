#!/usr/bin/env python3
"""
mym_phase6_sweep.py — MYM Phase 6: Full P6 Gate + HTF + LTF Re-opt

Goal: Beat MYM001 (brick 14, ADX>=50, PSAR, n=9, cd=45 — PF=232, 161t, 88.8% WR).

Stage A: Full 20-gate P6 sweep + PSAR stacking on brick 14/13
         ~2,496 runs (no new data needed)
Stage B: HTF gate discovery using brick 28/42 data
         ~432 runs (needs TV exports)
Stage C: LTF re-optimization with best HTF config locked
         ~3,456 runs

Usage:
  python renko/mym_phase6_sweep.py --stage a
  python renko/mym_phase6_sweep.py --stage b
  python renko/mym_phase6_sweep.py --stage c
  python renko/mym_phase6_sweep.py --stage all
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

from renko.config import MAX_WORKERS
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# ── Constants ──────────────────────────────────────────────────────────────────

MYM_COMMISSION_PCT = 0.00475
MYM_CAPITAL = 1000.0
MYM_QTY = 0.50
VOL_MAX = 1.5
OUT_PATH = ROOT / "ai_context" / "mym_phase6_results.json"

# ── Imports from base sweeps ──────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mym_sweep import (
    _compute_et_hours,
    _generate_signal_arrays,
    _run_backtest,
)
from mym_sweep_v4 import _load_renko_all_indicators_v4

sys.path.insert(0, str(ROOT))
from renko.phase6_sweep import _compute_gate_arrays as _p6_gate

# ── Instrument configs ────────────────────────────────────────────────────────

INSTRUMENTS_A = {
    "MYM_14": {
        "renko_file": "CBOT_MINI_MYM1!, 1S renko 14.csv",
        "is_start": "2025-03-07", "is_end": "2025-12-31",
        "oos_start": "2026-01-01", "oos_end": "2026-03-19",
        "label": "MYM brick 14",
    },
    "MYM_13": {
        "renko_file": "CBOT_MINI_MYM1!, 1S renko 13.csv",
        "is_start": "2025-04-09", "is_end": "2025-12-31",
        "oos_start": "2026-01-01", "oos_end": "2026-03-19",
        "label": "MYM brick 13",
    },
}

INSTRUMENTS_B = {
    "MYM_14": {
        "renko_file": "CBOT_MINI_MYM1!, 1S renko 14.csv",
        "htf_files": [
            "CBOT_MINI_MYM1!, 1S renko 28.csv",
            "CBOT_MINI_MYM1!, 1S renko 42.csv",
        ],
        "htf_labels": ["28", "42"],
        "is_start": "2025-03-07", "is_end": "2025-12-31",
        "oos_start": "2026-01-01", "oos_end": "2026-03-19",
        "label": "MYM brick 14",
    },
}

# ── All 20 P6 gate names ─────────────────────────────────────────────────────

ALL_P6_GATES = [
    "baseline", "rsi_dir", "bb_pct_b", "chop_trend", "psar_dir",
    "kama_slope", "sq_mom", "stoch_cross", "cmf_dir", "mfi_dir",
    "obv_trend", "ema_cross", "macd_hist_dir", "cci_dir", "ichi_cloud",
    "wpr_dir", "donch_mid", "escgo_cross", "ddl_dir", "motn_dx",
    "mk_regime",
]

# HTF gate names (from phase10)
HTF_GATE_NAMES = [
    "htf_brick_dir", "htf_n2_dir", "htf_n3_dir", "htf_adx30",
    "htf_ema_cross", "htf_psar_dir", "htf_macd_hist", "htf_stoch_cross",
]

HTF_ADX_THRESHOLDS = [25, 30, 35, 40, 45]


# ==============================================================================
#  STAGE A: Full P6 Gate Sweep
# ==============================================================================

STAGE_A_PARAMS = {
    "n_bricks": [8, 9, 10, 11],
    "cooldown": [40, 45, 50, 55],
}
STAGE_A_ADX = 50
STAGE_A_OSC = [None, "sto_tso"]


def _compute_all_gates_a(df, et_hours):
    """Pre-compute vol, ADX, all 20 P6 gates, and oscillators."""
    n = len(df)
    gates = {}

    # Vol ratio
    vr = df["vol_ratio"].values
    vr_nan = np.isnan(vr)
    gates["vol"] = (vr_nan | (vr <= VOL_MAX), vr_nan | (vr <= VOL_MAX))

    # ADX
    adx = df["adx"].values
    adx_nan = np.isnan(adx)
    ok = adx_nan | (adx >= STAGE_A_ADX)
    gates["adx"] = (ok, ok)

    # All 20 P6 gates
    for gname in ALL_P6_GATES:
        gates[f"p6:{gname}"] = _p6_gate(df, gname)

    # STO + TSO oscillator
    sto_mf = df["_bc_sto_mf"].values
    sto_ll = df["_bc_sto_ll"].values
    sto_nan = np.isnan(sto_mf) | np.isnan(sto_ll)
    sto_long = sto_nan | (sto_mf > sto_ll)
    sto_short = sto_nan | (sto_mf < sto_ll)

    tso_pink = df["_bc_tso_pink"].values.astype(float)
    tso_nan = np.isnan(tso_pink)
    tso_long = tso_nan | (tso_pink > 0.5)
    tso_short = tso_nan | (tso_pink < 0.5)
    gates["sto_tso"] = (sto_long & tso_long, sto_short & tso_short)

    return gates


def _combine_gates_a(gates, p6_primary, p6_secondary, osc_name):
    """AND-combine: vol + ADX + P6 primary + optional P6 secondary + optional osc."""
    n = len(gates["vol"][0])
    cl = np.ones(n, dtype=bool)
    cs = np.ones(n, dtype=bool)

    vl, vs = gates["vol"]
    cl &= vl; cs &= vs

    al, as_ = gates["adx"]
    cl &= al; cs &= as_

    pl, ps = gates[f"p6:{p6_primary}"]
    cl &= pl; cs &= ps

    if p6_secondary is not None:
        sl, ss = gates[f"p6:{p6_secondary}"]
        cl &= sl; cs &= ss

    if osc_name is not None:
        ol, os_ = gates[osc_name]
        cl &= ol; cs &= os_

    return cl, cs


def run_stage_a_worker(name, config):
    """Run Stage A for one instrument."""
    print(f"[A:{name}] Loading Renko + all indicators...", flush=True)
    df = _load_renko_all_indicators_v4(config["renko_file"])
    print(f"[A:{name}] Ready — {len(df)} bricks", flush=True)

    et_hours, et_minutes = _compute_et_hours(df.index)
    gates = _compute_all_gates_a(df, et_hours)

    brick_up = df["brick_up"].values
    df["long_entry"] = False
    df["long_exit"] = False
    df["short_entry"] = False
    df["short_exit"] = False

    # Build sweep combos: solo gates + PSAR stacking
    sweep_combos = []
    for gname in ALL_P6_GATES:
        sweep_combos.append((gname, None))  # solo
    for gname in ALL_P6_GATES:
        if gname != "psar_dir" and gname != "baseline":
            sweep_combos.append((gname, "psar_dir"))  # PSAR + gate

    keys = list(STAGE_A_PARAMS.keys())
    param_combos = [dict(zip(keys, v)) for v in itertools.product(*STAGE_A_PARAMS.values())]

    total = len(sweep_combos) * len(param_combos) * len(STAGE_A_OSC)
    done = 0
    results = []

    for p6_primary, p6_secondary in sweep_combos:
        for osc in STAGE_A_OSC:
            gate_long, gate_short = _combine_gates_a(gates, p6_primary, p6_secondary, osc)

            for pc in param_combos:
                le, lx, se, sx = _generate_signal_arrays(
                    brick_up,
                    n_bricks=pc["n_bricks"],
                    cooldown=pc["cooldown"],
                    gate_long_ok=gate_long,
                    gate_short_ok=gate_short,
                    et_hours=et_hours,
                    et_minutes=et_minutes,
                )
                df["long_entry"] = le
                df["long_exit"] = lx
                df["short_entry"] = se
                df["short_exit"] = sx

                is_r = _run_backtest(df, config["is_start"], config["is_end"])
                oos_r = _run_backtest(df, config["oos_start"], config["oos_end"])

                is_pf = is_r["pf"]
                oos_pf = oos_r["pf"]
                decay = ((oos_pf - is_pf) / is_pf * 100) \
                    if is_pf > 0 and not math.isinf(is_pf) else float("nan")

                osc_label = osc if osc else "none"
                stack_label = f"{p6_primary}" if p6_secondary is None else f"psar+{p6_primary}"

                results.append({
                    "stage": "A",
                    "instrument": name,
                    "brick": int(name.split("_")[1]),
                    "p6_gate": p6_primary,
                    "p6_stack": stack_label,
                    "p6_secondary": p6_secondary or "none",
                    "osc": osc_label,
                    "adx_thresh": STAGE_A_ADX,
                    "n_bricks": pc["n_bricks"],
                    "cooldown": pc["cooldown"],
                    "is_pf": is_pf,
                    "is_trades": is_r["trades"],
                    "is_wr": is_r["wr"],
                    "is_net": is_r["net"],
                    "oos_pf": oos_pf,
                    "oos_trades": oos_r["trades"],
                    "oos_wr": oos_r["wr"],
                    "oos_net": oos_r["net"],
                    "decay_pct": decay,
                })

                done += 1
                if done % 200 == 0 or done == total:
                    print(f"[A:{name}] {done:>5}/{total} | {stack_label:<25} "
                          f"n={pc['n_bricks']:>2} cd={pc['cooldown']:>2} osc={osc_label:<7} | "
                          f"OOS PF={oos_pf:>7.2f} T={oos_r['trades']:>4}", flush=True)

    print(f"[A:{name}] Complete — {len(results)} results", flush=True)
    return results


def _summarize_stage_a(all_results):
    """Print Stage A summary."""
    MIN_T = 10

    for inst in sorted(INSTRUMENTS_A.keys()):
        inst_res = [r for r in all_results if r["instrument"] == inst and r["stage"] == "A"]
        if not inst_res:
            continue

        viable = [r for r in inst_res if r["oos_trades"] >= MIN_T and not math.isinf(r["oos_pf"])]
        viable.sort(key=lambda r: r["oos_pf"], reverse=True)

        cfg = INSTRUMENTS_A[inst]
        print(f"\n{'='*100}")
        print(f"  Stage A — {cfg['label']}")
        print(f"{'='*100}")

        # Top 25
        print(f"\n  Top 25 (OOS trades >= {MIN_T}):")
        print(f"  {'Stack':<25} {'osc':<7} {'n':>2} {'cd':>3} | "
              f"{'IS PF':>7} {'T':>4} | {'OOS PF':>7} {'T':>4} {'WR%':>6} {'Net$':>8} {'Decay':>7}")
        print(f"  {'-'*95}")
        for r in viable[:25]:
            dec_s = f"{r['decay_pct']:>+6.1f}%" if not math.isnan(r["decay_pct"]) else "   NaN"
            print(f"  {r['p6_stack']:<25} {r['osc']:<7} {r['n_bricks']:>2} {r['cooldown']:>3} | "
                  f"{r['is_pf']:>7.2f} {r['is_trades']:>4} | "
                  f"{r['oos_pf']:>7.2f} {r['oos_trades']:>4} {r['oos_wr']:>5.1f}% "
                  f"{r['oos_net']:>8.2f} {dec_s}")

        # By P6 gate (solo only)
        print(f"\n  By P6 gate — solo (avg OOS PF, viable):")
        gate_avgs = []
        for gname in ALL_P6_GATES:
            gv = [r for r in viable if r["p6_gate"] == gname and r["p6_secondary"] == "none"]
            if gv:
                avg = sum(r["oos_pf"] for r in gv) / len(gv)
                avg_t = sum(r["oos_trades"] for r in gv) / len(gv)
                best = max(gv, key=lambda r: r["oos_pf"])
                gate_avgs.append((gname, avg, avg_t, len(gv), best["oos_pf"]))
        gate_avgs.sort(key=lambda x: x[1], reverse=True)
        for gname, avg, avg_t, cnt, best_pf in gate_avgs:
            print(f"    {gname:<20} avg PF={avg:>7.2f}  avg T={avg_t:>6.1f}  "
                  f"N={cnt:>4}  best={best_pf:>7.2f}")

        # Stacking value: solo vs PSAR+gate
        print(f"\n  PSAR stacking value (avg PF: solo vs psar+gate):")
        for gname in ALL_P6_GATES:
            if gname in ("psar_dir", "baseline"):
                continue
            solo = [r for r in viable if r["p6_gate"] == gname and r["p6_secondary"] == "none"]
            stacked = [r for r in viable if r["p6_gate"] == gname and r["p6_secondary"] == "psar_dir"]
            if solo and stacked:
                s_avg = sum(r["oos_pf"] for r in solo) / len(solo)
                st_avg = sum(r["oos_pf"] for r in stacked) / len(stacked)
                delta = (st_avg - s_avg) / s_avg * 100 if s_avg > 0 else 0
                print(f"    {gname:<20} solo={s_avg:>7.2f}  psar+={st_avg:>7.2f}  "
                      f"delta={delta:>+6.1f}%")

        # By oscillator
        print(f"\n  By oscillator:")
        for osc in ["none", "sto_tso"]:
            ov = [r for r in viable if r["osc"] == osc]
            if ov:
                avg = sum(r["oos_pf"] for r in ov) / len(ov)
                avg_t = sum(r["oos_trades"] for r in ov) / len(ov)
                print(f"    {osc:<16} avg PF={avg:>7.2f}  avg T={avg_t:>6.1f}  N={len(ov):>4}")

    # Cross-brick best
    print(f"\n{'='*100}")
    print("  Stage A — Overall best per brick (OOS PF, trades >= 10)")
    print(f"{'='*100}")
    for inst in sorted(INSTRUMENTS_A.keys()):
        viable = [r for r in all_results
                  if r["instrument"] == inst and r["stage"] == "A"
                  and r["oos_trades"] >= MIN_T and not math.isinf(r["oos_pf"])]
        if not viable:
            print(f"  {inst:<16} — no viable results")
            continue
        best = max(viable, key=lambda r: r["oos_pf"])
        print(f"  {inst:<16} OOS PF={best['oos_pf']:>7.2f} T={best['oos_trades']:>4} "
              f"WR={best['oos_wr']:>5.1f}% Net=${best['oos_net']:>7.2f} | "
              f"{best['p6_stack']} osc={best['osc']} n={best['n_bricks']} cd={best['cooldown']}")


def run_stage_a():
    """Run Stage A: Full P6 gate sweep."""
    n_solo = len(ALL_P6_GATES)
    n_stacked = len([g for g in ALL_P6_GATES if g not in ("psar_dir", "baseline")])
    n_combos = n_solo + n_stacked
    n_params = len(list(itertools.product(*STAGE_A_PARAMS.values())))
    n_osc = len(STAGE_A_OSC)
    per_brick = n_combos * n_params * n_osc
    total = per_brick * len(INSTRUMENTS_A)

    print("=" * 100)
    print("  STAGE A: Full P6 Gate Sweep")
    print("=" * 100)
    print(f"  P6 gates (solo)   : {n_solo}")
    print(f"  P6 gates (stacked): {n_stacked} (PSAR + gate)")
    print(f"  Params             : n_bricks={STAGE_A_PARAMS['n_bricks']}  cooldown={STAGE_A_PARAMS['cooldown']}")
    print(f"  Oscillators        : {['none'] + [o for o in STAGE_A_OSC if o]}")
    print(f"  ADX                : {STAGE_A_ADX} (locked)")
    print(f"  Per brick          : {per_brick} runs")
    print(f"  Total runs         : {total} ({total * 2} IS+OOS backtests)")
    print()

    all_results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(run_stage_a_worker, name, config): name
            for name, config in INSTRUMENTS_A.items()
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

    _summarize_stage_a(all_results)
    return all_results


# ==============================================================================
#  STAGE B: HTF Gate Discovery
# ==============================================================================

STAGE_B_PARAMS = {
    "n_bricks": [8, 9, 10, 11],
    "cooldown": [40, 45, 50, 55],
}


def _load_htf_data(htf_file):
    """Load HTF Renko data + basic indicators."""
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators

    df = load_renko_export(htf_file)
    add_renko_indicators(df)
    return df


def _compute_htf_gates(df_htf):
    """Compute all 8 HTF gate types + variable ADX thresholds."""
    n = len(df_htf)
    brick_up = df_htf["brick_up"].values
    gates = {}

    # 1. htf_brick_dir
    long_ok = np.ones(n, dtype=bool)
    short_ok = np.ones(n, dtype=bool)
    long_ok[1:] = brick_up[:-1]
    short_ok[1:] = ~brick_up[:-1]
    gates["htf_brick_dir"] = (long_ok.copy(), short_ok.copy())

    # 2. htf_n2_dir
    long_ok = np.ones(n, dtype=bool)
    short_ok = np.ones(n, dtype=bool)
    for i in range(2, n):
        long_ok[i] = brick_up[i-1] and brick_up[i-2]
        short_ok[i] = (not brick_up[i-1]) and (not brick_up[i-2])
    gates["htf_n2_dir"] = (long_ok, short_ok)

    # 3. htf_n3_dir
    long_ok = np.ones(n, dtype=bool)
    short_ok = np.ones(n, dtype=bool)
    for i in range(3, n):
        long_ok[i] = brick_up[i-1] and brick_up[i-2] and brick_up[i-3]
        short_ok[i] = (not brick_up[i-1]) and (not brick_up[i-2]) and (not brick_up[i-3])
    gates["htf_n3_dir"] = (long_ok, short_ok)

    # 4. htf_adx30 (fixed threshold)
    adx = df_htf["adx"].values
    adx_nan = np.isnan(adx)
    ok = adx_nan | (adx >= 30)
    gates["htf_adx30"] = (ok.copy(), ok.copy())

    # Variable ADX thresholds
    for thresh in HTF_ADX_THRESHOLDS:
        ok = adx_nan | (adx >= thresh)
        gates[f"htf_adx{thresh}"] = (ok.copy(), ok.copy())

    # 5. htf_ema_cross
    ema9 = df_htf["ema9"].values
    ema21 = df_htf["ema21"].values
    m = np.isnan(ema9) | np.isnan(ema21)
    gates["htf_ema_cross"] = (m | (ema9 > ema21), m | (ema9 < ema21))

    # 6. htf_psar_dir
    psar = df_htf["psar_dir"].values
    psar_nan = np.isnan(psar)
    gates["htf_psar_dir"] = (psar_nan | (psar > 0), psar_nan | (psar < 0))

    # 7. htf_macd_hist
    mh = df_htf["macd_hist"].values
    mh_nan = np.isnan(mh)
    gates["htf_macd_hist"] = (mh_nan | (mh >= 0), mh_nan | (mh < 0))

    # 8. htf_stoch_cross
    sk = df_htf["stoch_k"].values
    sd = df_htf["stoch_d"].values
    sm = np.isnan(sk) | np.isnan(sd)
    gates["htf_stoch_cross"] = (sm | (sk > sd), sm | (sk < sd))

    return gates


def _align_htf_gate_to_ltf(df_ltf, df_htf, htf_gate_long, htf_gate_short):
    """Backward-fill HTF gate onto LTF timestamps via merge_asof."""
    htf_frame = pd.DataFrame({
        "t": df_htf.index.values,
        "gl": htf_gate_long.astype(float),
        "gs": htf_gate_short.astype(float),
    }).sort_values("t")

    ltf_frame = pd.DataFrame({
        "t": df_ltf.index.values,
    }).sort_values("t")

    merged = pd.merge_asof(ltf_frame, htf_frame, on="t", direction="backward")

    gl = merged["gl"].values
    gs = merged["gs"].values
    aligned_long = np.where(np.isnan(gl), True, gl > 0.5)
    aligned_short = np.where(np.isnan(gs), True, gs > 0.5)

    return aligned_long.astype(bool), aligned_short.astype(bool)


def _find_best_stage_a_config(stage_a_results, instrument):
    """Find best Stage A config for a given instrument. Falls back to MYM001."""
    viable = [r for r in stage_a_results
              if r["instrument"] == instrument and r["stage"] == "A"
              and r["oos_trades"] >= 10 and not math.isinf(r["oos_pf"])]
    if viable:
        best = max(viable, key=lambda r: r["oos_pf"])
        return {
            "p6_gate": best["p6_gate"],
            "p6_secondary": best["p6_secondary"],
            "osc": best["osc"] if best["osc"] != "none" else None,
        }
    # Fallback: MYM001 config
    return {"p6_gate": "psar_dir", "p6_secondary": "none", "osc": None}


def run_stage_b(existing_results=None):
    """Run Stage B: HTF gate discovery."""
    # Find best LTF config from Stage A
    stage_a = [r for r in (existing_results or []) if r.get("stage") == "A"]
    ltf_best = _find_best_stage_a_config(stage_a, "MYM_14")

    keys = list(STAGE_B_PARAMS.keys())
    param_combos = [dict(zip(keys, v)) for v in itertools.product(*STAGE_B_PARAMS.values())]

    # All HTF gates to test
    all_htf_gates = HTF_GATE_NAMES + [f"htf_adx{t}" for t in HTF_ADX_THRESHOLDS]

    total = len(param_combos) * (1 + len(INSTRUMENTS_B["MYM_14"]["htf_labels"]) * len(all_htf_gates))

    print(f"\n{'='*100}")
    print("  STAGE B: HTF Gate Discovery")
    print(f"{'='*100}")
    print(f"  LTF locked         : p6={ltf_best['p6_gate']} "
          f"p6_sec={ltf_best['p6_secondary']} osc={ltf_best['osc']}")
    print(f"  HTF bricks         : {INSTRUMENTS_B['MYM_14']['htf_labels']}")
    print(f"  HTF gates          : {len(all_htf_gates)}")
    print(f"  Params             : n_bricks={STAGE_B_PARAMS['n_bricks']}  cooldown={STAGE_B_PARAMS['cooldown']}")
    print(f"  Total runs         : {total} ({total * 2} IS+OOS backtests)")
    print()

    name = "MYM_14"
    config = INSTRUMENTS_B[name]

    print(f"[B:{name}] Loading LTF Renko + all indicators...", flush=True)
    df_ltf = _load_renko_all_indicators_v4(config["renko_file"])
    print(f"[B:{name}] LTF ready — {len(df_ltf)} bricks", flush=True)

    et_hours, et_minutes = _compute_et_hours(df_ltf.index)
    gates_ltf = _compute_all_gates_a(df_ltf, et_hours)

    # Compute fixed LTF gate
    p6_sec = ltf_best["p6_secondary"] if ltf_best["p6_secondary"] != "none" else None
    ltf_long, ltf_short = _combine_gates_a(gates_ltf, ltf_best["p6_gate"], p6_sec, ltf_best["osc"])

    brick_up = df_ltf["brick_up"].values
    df_ltf["long_entry"] = False
    df_ltf["long_exit"] = False
    df_ltf["short_entry"] = False
    df_ltf["short_exit"] = False

    # Pre-compute all HTF aligned gates
    htf_aligned = {}
    for htf_file, htf_label in zip(config["htf_files"], config["htf_labels"]):
        fpath = ROOT / "data" / htf_file
        if not fpath.exists():
            print(f"[B:{name}] SKIP HTF {htf_label} — file not found: {fpath}")
            continue
        print(f"[B:{name}] Loading HTF {htf_label}...", flush=True)
        df_htf = _load_htf_data(htf_file)
        print(f"[B:{name}] HTF {htf_label} ready — {len(df_htf)} bricks", flush=True)

        htf_gates = _compute_htf_gates(df_htf)
        htf_aligned[htf_label] = {}
        for gname, (gl, gs) in htf_gates.items():
            al, as_ = _align_htf_gate_to_ltf(df_ltf, df_htf, gl, gs)
            htf_aligned[htf_label][gname] = (al, as_)

    if not htf_aligned:
        print("[B] No HTF data found. Export brick 28 and 42 from TV first.")
        return []

    # Build sweep: baseline + HTF combos
    sweep_configs = [("baseline", None, None)]
    for htf_label in htf_aligned:
        for gname in all_htf_gates:
            if gname in htf_aligned[htf_label]:
                sweep_configs.append((f"{htf_label}_{gname}", htf_label, gname))

    results = []
    done = 0
    total_actual = len(sweep_configs) * len(param_combos)

    for sweep_label, htf_label, htf_gate_name in sweep_configs:
        if htf_label is not None:
            htf_gl, htf_gs = htf_aligned[htf_label][htf_gate_name]
            combined_long = ltf_long & htf_gl
            combined_short = ltf_short & htf_gs
        else:
            combined_long = ltf_long
            combined_short = ltf_short

        for pc in param_combos:
            le, lx, se, sx = _generate_signal_arrays(
                brick_up,
                n_bricks=pc["n_bricks"],
                cooldown=pc["cooldown"],
                gate_long_ok=combined_long,
                gate_short_ok=combined_short,
                et_hours=et_hours,
                et_minutes=et_minutes,
            )
            df_ltf["long_entry"] = le
            df_ltf["long_exit"] = lx
            df_ltf["short_entry"] = se
            df_ltf["short_exit"] = sx

            is_r = _run_backtest(df_ltf, config["is_start"], config["is_end"])
            oos_r = _run_backtest(df_ltf, config["oos_start"], config["oos_end"])

            is_pf = is_r["pf"]
            oos_pf = oos_r["pf"]
            decay = ((oos_pf - is_pf) / is_pf * 100) \
                if is_pf > 0 and not math.isinf(is_pf) else float("nan")

            results.append({
                "stage": "B",
                "instrument": name,
                "brick": 14,
                "htf_brick": htf_label or "none",
                "htf_gate": htf_gate_name or "none",
                "ltf_p6_gate": ltf_best["p6_gate"],
                "ltf_p6_secondary": ltf_best["p6_secondary"],
                "ltf_osc": ltf_best["osc"] or "none",
                "n_bricks": pc["n_bricks"],
                "cooldown": pc["cooldown"],
                "is_pf": is_pf,
                "is_trades": is_r["trades"],
                "is_wr": is_r["wr"],
                "is_net": is_r["net"],
                "oos_pf": oos_pf,
                "oos_trades": oos_r["trades"],
                "oos_wr": oos_r["wr"],
                "oos_net": oos_r["net"],
                "decay_pct": decay,
            })

            done += 1
            if done % 100 == 0 or done == total_actual:
                print(f"[B:{name}] {done:>5}/{total_actual} | {sweep_label:<30} "
                      f"n={pc['n_bricks']:>2} cd={pc['cooldown']:>2} | "
                      f"OOS PF={oos_pf:>7.2f} T={oos_r['trades']:>4}", flush=True)

    print(f"[B:{name}] Complete — {len(results)} results", flush=True)

    # Summarize
    _summarize_stage_b(results)
    return results


def _summarize_stage_b(results):
    """Print Stage B summary."""
    MIN_T = 10
    viable = [r for r in results if r["oos_trades"] >= MIN_T and not math.isinf(r["oos_pf"])]
    viable.sort(key=lambda r: r["oos_pf"], reverse=True)

    print(f"\n{'='*100}")
    print("  Stage B — HTF Gate Discovery Results")
    print(f"{'='*100}")

    # Top 20
    print(f"\n  Top 20:")
    print(f"  {'HTF':<30} {'n':>2} {'cd':>3} | {'IS PF':>7} {'T':>4} | "
          f"{'OOS PF':>7} {'T':>4} {'WR%':>6} {'Net$':>8} {'Decay':>7}")
    print(f"  {'-'*95}")
    for r in viable[:20]:
        htf_lbl = f"{r['htf_brick']}_{r['htf_gate']}" if r['htf_brick'] != "none" else "baseline"
        dec_s = f"{r['decay_pct']:>+6.1f}%" if not math.isnan(r["decay_pct"]) else "   NaN"
        print(f"  {htf_lbl:<30} {r['n_bricks']:>2} {r['cooldown']:>3} | "
              f"{r['is_pf']:>7.2f} {r['is_trades']:>4} | "
              f"{r['oos_pf']:>7.2f} {r['oos_trades']:>4} {r['oos_wr']:>5.1f}% "
              f"{r['oos_net']:>8.2f} {dec_s}")

    # By HTF gate type
    print(f"\n  By HTF gate (avg OOS PF):")
    baseline_res = [r for r in viable if r["htf_brick"] == "none"]
    baseline_avg = sum(r["oos_pf"] for r in baseline_res) / len(baseline_res) if baseline_res else 0
    print(f"    {'baseline':<25} avg PF={baseline_avg:>7.2f}  N={len(baseline_res):>4}")

    all_htf_gates = set(r["htf_gate"] for r in viable if r["htf_gate"] != "none")
    for gname in sorted(all_htf_gates):
        gv = [r for r in viable if r["htf_gate"] == gname]
        if gv:
            avg = sum(r["oos_pf"] for r in gv) / len(gv)
            delta = (avg - baseline_avg) / baseline_avg * 100 if baseline_avg > 0 else 0
            print(f"    {gname:<25} avg PF={avg:>7.2f}  N={len(gv):>4}  "
                  f"vs baseline: {delta:>+6.1f}%")

    # By HTF brick
    print(f"\n  By HTF brick size:")
    for htf_b in sorted(set(r["htf_brick"] for r in viable)):
        bv = [r for r in viable if r["htf_brick"] == htf_b]
        if bv:
            avg = sum(r["oos_pf"] for r in bv) / len(bv)
            print(f"    HTF {htf_b:<10} avg PF={avg:>7.2f}  N={len(bv):>4}")


# ==============================================================================
#  STAGE C: LTF Re-optimization with HTF Locked
# ==============================================================================

STAGE_C_P6_GATES = []  # dynamically set from Stage A results
STAGE_C_P4_GATES = ["none", "vel_fast", "range_ok", "no_exhaust"]
STAGE_C_OSC = [None, "sto_tso", "macd_lc"]
STAGE_C_ADX = [40, 45, 50]
STAGE_C_PARAMS = {
    "n_bricks": [8, 9, 10, 11],
    "cooldown": [40, 45, 50, 55],
}


def _find_best_htf_config(stage_b_results):
    """Find best HTF config from Stage B results."""
    viable = [r for r in stage_b_results
              if r.get("stage") == "B" and r["htf_brick"] != "none"
              and r["oos_trades"] >= 10 and not math.isinf(r["oos_pf"])]
    if viable:
        best = max(viable, key=lambda r: r["oos_pf"])
        return {
            "htf_brick": best["htf_brick"],
            "htf_gate": best["htf_gate"],
        }
    return None


def _find_top_p6_gates(stage_a_results, instrument, top_n=5):
    """Find top N P6 gates from Stage A by avg OOS PF on given instrument."""
    viable = [r for r in stage_a_results
              if r["instrument"] == instrument and r.get("stage") == "A"
              and r["oos_trades"] >= 10 and not math.isinf(r["oos_pf"])
              and r["p6_secondary"] == "none"]

    gate_avgs = {}
    for r in viable:
        g = r["p6_gate"]
        gate_avgs.setdefault(g, []).append(r["oos_pf"])

    ranked = sorted(gate_avgs.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True)
    return [g for g, _ in ranked[:top_n]]


def run_stage_c(existing_results=None):
    """Run Stage C: LTF re-optimization with best HTF locked."""
    all_existing = existing_results or []
    stage_a = [r for r in all_existing if r.get("stage") == "A"]
    stage_b = [r for r in all_existing if r.get("stage") == "B"]

    htf_best = _find_best_htf_config(stage_b)
    if htf_best is None:
        print("[C] No Stage B results found. Run Stage B first.")
        return []

    top_p6 = _find_top_p6_gates(stage_a, "MYM_14")
    if not top_p6:
        top_p6 = ["psar_dir", "ema_cross", "stoch_cross", "kama_slope", "ichi_cloud"]
    p6_gates = list(dict.fromkeys(["baseline"] + top_p6))  # dedupe, keep order

    name = "MYM_14"
    config = INSTRUMENTS_B[name]

    keys = list(STAGE_C_PARAMS.keys())
    param_combos = [dict(zip(keys, v)) for v in itertools.product(*STAGE_C_PARAMS.values())]

    total = len(p6_gates) * len(STAGE_C_P4_GATES) * len(STAGE_C_OSC) * len(STAGE_C_ADX) * len(param_combos)

    print(f"\n{'='*100}")
    print("  STAGE C: LTF Re-optimization with HTF Locked")
    print(f"{'='*100}")
    print(f"  HTF locked         : brick={htf_best['htf_brick']} gate={htf_best['htf_gate']}")
    print(f"  P6 gates           : {p6_gates}")
    print(f"  P4 gates           : {STAGE_C_P4_GATES}")
    print(f"  Oscillators        : {['none'] + [o for o in STAGE_C_OSC if o]}")
    print(f"  ADX thresholds     : {STAGE_C_ADX}")
    print(f"  Params             : n_bricks={STAGE_C_PARAMS['n_bricks']}  cooldown={STAGE_C_PARAMS['cooldown']}")
    print(f"  Total runs         : {total} ({total * 2} IS+OOS backtests)")
    print()

    # Load LTF
    print(f"[C:{name}] Loading LTF Renko + all indicators...", flush=True)
    df_ltf = _load_renko_all_indicators_v4(config["renko_file"])

    # Add MACD Wave Signal Pro (not included in v4 loader, needed for macd_lc osc)
    if "_bc_macd_state" not in df_ltf.columns:
        try:
            from indicators.blackcat1402.blackcat_l3_macd_wave_signal_pro import (
                calc_bc_l3_macd_wave_signal_pro,
            )
            df_lc = df_ltf.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            })
            macd_result = calc_bc_l3_macd_wave_signal_pro(df_lc)
            df_ltf["_bc_macd_state"] = macd_result["bc_macd_state"].shift(1).values
            df_ltf["_bc_lc"] = macd_result["bc_lc"].shift(1).values
        except Exception as e:
            print(f"  WARN: MACD Wave Signal Pro failed: {e}")
            df_ltf["_bc_macd_state"] = np.nan
            df_ltf["_bc_lc"] = np.nan

    print(f"[C:{name}] LTF ready — {len(df_ltf)} bricks", flush=True)

    et_hours, et_minutes = _compute_et_hours(df_ltf.index)

    # Pre-compute all LTF gates
    n = len(df_ltf)
    gates = {}

    # Vol
    vr = df_ltf["vol_ratio"].values
    vr_nan = np.isnan(vr)
    gates["vol"] = (vr_nan | (vr <= VOL_MAX), vr_nan | (vr <= VOL_MAX))

    # ADX at multiple thresholds
    adx = df_ltf["adx"].values
    adx_nan = np.isnan(adx)
    for at in STAGE_C_ADX:
        ok = adx_nan | (adx >= at)
        gates[f"adx_{at}"] = (ok, ok)

    # P6 gates
    for gname in p6_gates:
        gates[f"p6:{gname}"] = _p6_gate(df_ltf, gname)

    # P4 gates (reuse from v4)
    from mym_sweep_v4 import _compute_all_gates as _v4_gates
    v4_gates = _v4_gates(df_ltf, et_hours)
    for p4name in STAGE_C_P4_GATES:
        gates[f"p4:{p4name}"] = v4_gates.get(f"p4:{p4name}",
                                              (np.ones(n, dtype=bool), np.ones(n, dtype=bool)))

    # Oscillators
    sto_mf = df_ltf["_bc_sto_mf"].values
    sto_ll = df_ltf["_bc_sto_ll"].values
    sto_nan = np.isnan(sto_mf) | np.isnan(sto_ll)
    sto_long = sto_nan | (sto_mf > sto_ll)
    sto_short = sto_nan | (sto_mf < sto_ll)
    tso_pink = df_ltf["_bc_tso_pink"].values.astype(float)
    tso_nan = np.isnan(tso_pink)
    tso_long = tso_nan | (tso_pink > 0.5)
    tso_short = tso_nan | (tso_pink < 0.5)
    gates["sto_tso"] = (sto_long & tso_long, sto_short & tso_short)

    macd_st = df_ltf["_bc_macd_state"].values
    bc_lc = df_ltf["_bc_lc"].values
    ms_nan = np.isnan(macd_st)
    lc_nan = np.isnan(bc_lc)
    ms_int = np.where(ms_nan, -1, macd_st).astype(int)
    ms_long = ms_nan | np.isin(ms_int, [0, 3])
    ms_short = ms_nan | np.isin(ms_int, [1, 2])
    lc_long = lc_nan | (bc_lc > 0)
    lc_short = lc_nan | (bc_lc < 0)
    gates["macd_lc"] = (ms_long & lc_long, ms_short & lc_short)

    # Load HTF and align
    htf_file = f"CBOT_MINI_MYM1!, 1S renko {htf_best['htf_brick']}.csv"
    print(f"[C:{name}] Loading HTF brick {htf_best['htf_brick']}...", flush=True)
    df_htf = _load_htf_data(htf_file)
    print(f"[C:{name}] HTF ready — {len(df_htf)} bricks", flush=True)

    htf_gates = _compute_htf_gates(df_htf)
    htf_gl, htf_gs = htf_gates[htf_best["htf_gate"]]
    htf_long, htf_short = _align_htf_gate_to_ltf(df_ltf, df_htf, htf_gl, htf_gs)

    brick_up = df_ltf["brick_up"].values
    df_ltf["long_entry"] = False
    df_ltf["long_exit"] = False
    df_ltf["short_entry"] = False
    df_ltf["short_exit"] = False

    results = []
    done = 0

    for p6 in p6_gates:
        for p4 in STAGE_C_P4_GATES:
            for osc in STAGE_C_OSC:
                for adx_t in STAGE_C_ADX:
                    # Combine: vol + ADX + P6 + P4 + osc + HTF
                    cl = np.ones(n, dtype=bool)
                    cs = np.ones(n, dtype=bool)

                    vl, vs = gates["vol"]
                    cl &= vl; cs &= vs

                    al, as_ = gates[f"adx_{adx_t}"]
                    cl &= al; cs &= as_

                    pl, ps = gates[f"p6:{p6}"]
                    cl &= pl; cs &= ps

                    p4l, p4s = gates[f"p4:{p4}"]
                    cl &= p4l; cs &= p4s

                    if osc is not None:
                        ol, os_ = gates[osc]
                        cl &= ol; cs &= os_

                    # HTF gate
                    cl &= htf_long; cs &= htf_short

                    for pc in param_combos:
                        le, lx, se, sx = _generate_signal_arrays(
                            brick_up,
                            n_bricks=pc["n_bricks"],
                            cooldown=pc["cooldown"],
                            gate_long_ok=cl,
                            gate_short_ok=cs,
                            et_hours=et_hours,
                            et_minutes=et_minutes,
                        )
                        df_ltf["long_entry"] = le
                        df_ltf["long_exit"] = lx
                        df_ltf["short_entry"] = se
                        df_ltf["short_exit"] = sx

                        is_r = _run_backtest(df_ltf, config["is_start"], config["is_end"])
                        oos_r = _run_backtest(df_ltf, config["oos_start"], config["oos_end"])

                        is_pf = is_r["pf"]
                        oos_pf = oos_r["pf"]
                        decay = ((oos_pf - is_pf) / is_pf * 100) \
                            if is_pf > 0 and not math.isinf(is_pf) else float("nan")

                        osc_label = osc if osc else "none"

                        results.append({
                            "stage": "C",
                            "instrument": name,
                            "brick": 14,
                            "htf_brick": htf_best["htf_brick"],
                            "htf_gate": htf_best["htf_gate"],
                            "p6_gate": p6,
                            "p4_gate": p4,
                            "osc": osc_label,
                            "adx_thresh": adx_t,
                            "n_bricks": pc["n_bricks"],
                            "cooldown": pc["cooldown"],
                            "is_pf": is_pf,
                            "is_trades": is_r["trades"],
                            "is_wr": is_r["wr"],
                            "is_net": is_r["net"],
                            "oos_pf": oos_pf,
                            "oos_trades": oos_r["trades"],
                            "oos_wr": oos_r["wr"],
                            "oos_net": oos_r["net"],
                            "decay_pct": decay,
                        })

                        done += 1
                        if done % 500 == 0 or done == total:
                            print(f"[C:{name}] {done:>5}/{total} | "
                                  f"p6={p6:<15} p4={p4:<12} osc={osc_label:<7} adx={adx_t} "
                                  f"n={pc['n_bricks']:>2} cd={pc['cooldown']:>2} | "
                                  f"OOS PF={oos_pf:>7.2f} T={oos_r['trades']:>4}", flush=True)

    print(f"[C:{name}] Complete — {len(results)} results", flush=True)

    # Summarize
    _summarize_stage_c(results)
    return results


def _summarize_stage_c(results):
    """Print Stage C summary."""
    MIN_T = 10
    viable = [r for r in results if r["oos_trades"] >= MIN_T and not math.isinf(r["oos_pf"])]
    viable.sort(key=lambda r: r["oos_pf"], reverse=True)

    print(f"\n{'='*100}")
    print("  Stage C — LTF Re-optimization with HTF Locked")
    print(f"{'='*100}")

    # Top 25
    print(f"\n  Top 25:")
    print(f"  {'p6':<15} {'p4':<12} {'osc':<7} {'adx':>3} {'n':>2} {'cd':>3} | "
          f"{'IS PF':>7} {'T':>4} | {'OOS PF':>7} {'T':>4} {'WR%':>6} {'Net$':>8} {'Decay':>7}")
    print(f"  {'-'*105}")
    for r in viable[:25]:
        dec_s = f"{r['decay_pct']:>+6.1f}%" if not math.isnan(r["decay_pct"]) else "   NaN"
        print(f"  {r['p6_gate']:<15} {r['p4_gate']:<12} {r['osc']:<7} {r['adx_thresh']:>3} "
              f"{r['n_bricks']:>2} {r['cooldown']:>3} | "
              f"{r['is_pf']:>7.2f} {r['is_trades']:>4} | "
              f"{r['oos_pf']:>7.2f} {r['oos_trades']:>4} {r['oos_wr']:>5.1f}% "
              f"{r['oos_net']:>8.2f} {dec_s}")

    # By P6 gate
    print(f"\n  By P6 gate (avg OOS PF):")
    p6_gates = sorted(set(r["p6_gate"] for r in viable))
    for g in p6_gates:
        gv = [r for r in viable if r["p6_gate"] == g]
        if gv:
            avg = sum(r["oos_pf"] for r in gv) / len(gv)
            print(f"    {g:<20} avg PF={avg:>7.2f}  N={len(gv):>4}")

    # By P4 gate
    print(f"\n  By P4 gate (avg OOS PF):")
    for g in STAGE_C_P4_GATES:
        gv = [r for r in viable if r["p4_gate"] == g]
        if gv:
            avg = sum(r["oos_pf"] for r in gv) / len(gv)
            print(f"    {g:<20} avg PF={avg:>7.2f}  N={len(gv):>4}")

    # By oscillator
    print(f"\n  By oscillator:")
    for osc in ["none", "sto_tso", "macd_lc"]:
        ov = [r for r in viable if r["osc"] == osc]
        if ov:
            avg = sum(r["oos_pf"] for r in ov) / len(ov)
            print(f"    {osc:<20} avg PF={avg:>7.2f}  N={len(ov):>4}")

    # By ADX
    print(f"\n  By ADX threshold:")
    for at in STAGE_C_ADX:
        av = [r for r in viable if r["adx_thresh"] == at]
        if av:
            avg = sum(r["oos_pf"] for r in av) / len(av)
            print(f"    ADX>={at:<5}          avg PF={avg:>7.2f}  N={len(av):>4}")

    # Best overall
    if viable:
        best = viable[0]
        print(f"\n  BEST OVERALL:")
        print(f"    OOS PF={best['oos_pf']:.2f} T={best['oos_trades']} "
              f"WR={best['oos_wr']:.1f}% Net=${best['oos_net']:.2f}")
        print(f"    Config: p6={best['p6_gate']} p4={best['p4_gate']} osc={best['osc']} "
              f"adx={best['adx_thresh']} n={best['n_bricks']} cd={best['cooldown']}")
        print(f"    HTF: brick={best['htf_brick']} gate={best['htf_gate']}")
        print(f"\n  vs MYM001 (PF=232, 161t, 88.8% WR):")
        pf_delta = (best['oos_pf'] - 232) / 232 * 100
        print(f"    PF delta: {pf_delta:+.1f}%")


# ==============================================================================
#  Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="MYM Phase 6 Sweep")
    parser.add_argument("--stage", choices=["a", "b", "c", "ab", "bc", "all"],
                        default="a", help="Which stage(s) to run")
    parser.add_argument("--no-parallel", action="store_true")
    args = parser.parse_args()

    OUT_PATH.parent.mkdir(exist_ok=True)

    # Load existing results if continuing
    existing = []
    if OUT_PATH.exists() and args.stage not in ("a", "all"):
        with open(OUT_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing results from {OUT_PATH}")

    all_results = list(existing)

    run_a = args.stage in ("a", "ab", "all")
    run_b = args.stage in ("b", "ab", "bc", "all")
    run_c = args.stage in ("c", "bc", "all")

    if run_a:
        stage_a_results = run_stage_a()
        # Remove old Stage A results, add new
        all_results = [r for r in all_results if r.get("stage") != "A"]
        all_results.extend(stage_a_results)

        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nSaved {len(all_results)} results -> {OUT_PATH}")

    if run_b:
        stage_b_results = run_stage_b(existing_results=all_results)
        all_results = [r for r in all_results if r.get("stage") != "B"]
        all_results.extend(stage_b_results)

        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nSaved {len(all_results)} results -> {OUT_PATH}")

    if run_c:
        stage_c_results = run_stage_c(existing_results=all_results)
        all_results = [r for r in all_results if r.get("stage") != "C"]
        all_results.extend(stage_c_results)

        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nSaved {len(all_results)} results -> {OUT_PATH}")

    print(f"\nTotal results: {len(all_results)}")


if __name__ == "__main__":
    main()
