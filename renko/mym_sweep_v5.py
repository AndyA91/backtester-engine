#!/usr/bin/env python3
"""
mym_sweep_v5.py -- MYM Phase 5: Walk-Forward Validation + Gate Stacking

Two tests:

  TEST A: Walk-forward validation on brick 13
    Rolling 4-month IS / 1-month OOS windows across the full data range.
    Tests the top 5 v4 configs to see if the winner is consistent across folds.

  TEST B: Gate stacking — combine range_ok + vel_fast
    range_ok was the best P4 gate overall (+23% avg lift).
    vel_fast was best on brick 13 (+22%).
    Test requiring BOTH gates to pass, across all brick sizes.

Usage:
  python renko/mym_sweep_v5.py
  python renko/mym_sweep_v5.py --test wf       # walk-forward only
  python renko/mym_sweep_v5.py --test stack     # gate stacking only
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
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# ---- Shared constants ----------------------------------------------------------

MYM_COMMISSION_PCT = 0.00475
MYM_CAPITAL = 1000.0
MYM_QTY = 0.50

# ---- Imports from base sweep ---------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mym_sweep import (
    _compute_et_hours,
    _generate_signal_arrays,
    _run_backtest,
)
from mym_sweep_v4 import (
    _load_renko_all_indicators_v4,
    _compute_all_gates as _compute_all_gates_v4,
    _combine_gates as _combine_gates_v4,
)

# ====================================================================================
#  TEST A: Walk-Forward Validation on Brick 13
# ====================================================================================

# Top v4 configs for brick 13 to validate
WF_CONFIGS = [
    # (label, adx_thresh, p6_gate, p4_gate, osc, n_bricks, cooldown)
    ("psar_vel_fast",      50, "psar_dir",  "vel_fast",      None,     10, 50),
    ("ema_vel_fast",       50, "ema_cross", "vel_fast",      None,     10, 50),
    ("psar_vel_vfast",     50, "psar_dir",  "vel_very_fast", None,     10, 50),
    ("ema_vel_vfast",      50, "ema_cross", "vel_very_fast", None,     10, 50),
    ("psar_range_ok",      50, "psar_dir",  "range_ok",      None,     10, 40),
    ("ema_range_ok",       50, "ema_cross", "range_ok",      None,     10, 40),
    # Also test the baseline (no P4 gate) for comparison
    ("psar_none",          50, "psar_dir",  "none",          None,     10, 50),
    ("ema_none",           50, "ema_cross", "none",          None,     10, 50),
]

# Walk-forward windows: (is_start, is_end, oos_start, oos_end)
# Data range: ~2025-04-09 to 2026-03-19
# 4-month IS, 1-month OOS, rolling 1 month
WF_FOLDS = [
    ("2025-04-09", "2025-08-09", "2025-08-10", "2025-09-09"),   # Fold 1
    ("2025-05-10", "2025-09-09", "2025-09-10", "2025-10-09"),   # Fold 2
    ("2025-06-10", "2025-10-09", "2025-10-10", "2025-11-09"),   # Fold 3
    ("2025-07-10", "2025-11-09", "2025-11-10", "2025-12-09"),   # Fold 4
    ("2025-08-10", "2025-12-09", "2025-12-10", "2026-01-09"),   # Fold 5
    ("2025-09-10", "2026-01-09", "2026-01-10", "2026-02-09"),   # Fold 6
    ("2025-10-10", "2026-02-09", "2026-02-10", "2026-03-19"),   # Fold 7
]


def run_walk_forward():
    """Run walk-forward validation on brick 13."""
    print("=" * 100)
    print("  TEST A: Walk-Forward Validation — MYM Brick 13")
    print("=" * 100)
    print(f"  Configs to test : {len(WF_CONFIGS)}")
    print(f"  Folds           : {len(WF_FOLDS)}")
    print(f"  Total backtests : {len(WF_CONFIGS) * len(WF_FOLDS) * 2} (IS+OOS)")
    print()

    renko_file = "CBOT_MINI_MYM1!, 1S renko 13.csv"
    print("[WF] Loading Renko + ALL indicators (v4)...", flush=True)
    df = _load_renko_all_indicators_v4(renko_file)
    print(f"[WF] Ready — {len(df)} bricks", flush=True)

    et_hours, et_minutes = _compute_et_hours(df.index)
    gates = _compute_all_gates_v4(df, et_hours)

    brick_up = df["brick_up"].values
    df["long_entry"] = False
    df["long_exit"] = False
    df["short_entry"] = False
    df["short_exit"] = False

    results = []

    for cfg_label, adx_t, p6, p4, osc, nb, cd in WF_CONFIGS:
        gate_long, gate_short = _combine_gates_v4(gates, adx_t, p6, p4, osc)

        le, lx, se, sx = _generate_signal_arrays(
            brick_up,
            n_bricks=nb,
            cooldown=cd,
            gate_long_ok=gate_long,
            gate_short_ok=gate_short,
            et_hours=et_hours,
            et_minutes=et_minutes,
        )
        df["long_entry"] = le
        df["long_exit"] = lx
        df["short_entry"] = se
        df["short_exit"] = sx

        for fold_idx, (is_s, is_e, oos_s, oos_e) in enumerate(WF_FOLDS):
            is_r = _run_backtest(df, is_s, is_e)
            oos_r = _run_backtest(df, oos_s, oos_e)

            is_pf = is_r["pf"]
            oos_pf = oos_r["pf"]
            decay = ((oos_pf - is_pf) / is_pf * 100) \
                if is_pf > 0 and not math.isinf(is_pf) else float("nan")

            results.append({
                "config": cfg_label,
                "fold": fold_idx + 1,
                "is_period": f"{is_s} → {is_e}",
                "oos_period": f"{oos_s} → {oos_e}",
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

        # Progress
        fold_pfs = [r["oos_pf"] for r in results if r["config"] == cfg_label]
        avg_pf = sum(p for p in fold_pfs if not math.isinf(p)) / max(len([p for p in fold_pfs if not math.isinf(p)]), 1)
        print(f"  [{cfg_label:<20}] avg OOS PF={avg_pf:>7.2f} across {len(WF_FOLDS)} folds", flush=True)

    # ── Summarize walk-forward ──
    print(f"\n{'='*100}")
    print("  Walk-Forward Results — Per Config")
    print(f"{'='*100}")
    print(f"\n  {'Config':<22} | {'Avg PF':>7} {'Med PF':>7} {'Min PF':>7} {'Max PF':>7} | "
          f"{'Avg T':>5} {'Tot T':>5} {'Avg WR':>6} | {'Tot Net$':>9} {'Win Folds':>10}")
    print(f"  {'-'*105}")

    for cfg_label, *_ in WF_CONFIGS:
        fold_res = [r for r in results if r["config"] == cfg_label]
        oos_pfs = [r["oos_pf"] for r in fold_res]
        oos_pfs_finite = [p for p in oos_pfs if not math.isinf(p)]

        if not oos_pfs_finite:
            print(f"  {cfg_label:<22} | all inf")
            continue

        avg_pf = sum(oos_pfs_finite) / len(oos_pfs_finite)
        sorted_pfs = sorted(oos_pfs_finite)
        med_pf = sorted_pfs[len(sorted_pfs) // 2]
        min_pf = min(oos_pfs_finite)
        max_pf = max(oos_pfs_finite)
        avg_t = sum(r["oos_trades"] for r in fold_res) / len(fold_res)
        tot_t = sum(r["oos_trades"] for r in fold_res)
        tot_net = sum(r["oos_net"] for r in fold_res)
        wrs = [r["oos_wr"] for r in fold_res if r["oos_trades"] > 0]
        avg_wr = sum(wrs) / len(wrs) if wrs else 0

        # Count profitable folds (PF > 1.0 = profitable)
        win_folds = sum(1 for p in oos_pfs if p > 1.0)

        print(f"  {cfg_label:<22} | {avg_pf:>7.2f} {med_pf:>7.2f} {min_pf:>7.2f} {max_pf:>7.2f} | "
              f"{avg_t:>5.1f} {tot_t:>5} {avg_wr:>5.1f}% | "
              f"${tot_net:>8.2f} {win_folds:>4}/{len(WF_FOLDS)}")

    # Fold-by-fold detail for top config
    print(f"\n  Fold-by-fold detail:")
    print(f"  {'Config':<22} {'Fold':>4} | {'OOS Period':<27} | {'OOS PF':>7} {'T':>4} {'WR%':>6} {'Net$':>8}")
    print(f"  {'-'*95}")
    for r in results:
        pf_s = f"{r['oos_pf']:>7.2f}" if not math.isinf(r["oos_pf"]) else "    inf"
        print(f"  {r['config']:<22} {r['fold']:>4} | {r['oos_period']:<27} | "
              f"{pf_s} {r['oos_trades']:>4} {r['oos_wr']:>5.1f}% {r['oos_net']:>8.2f}")

    return results


# ====================================================================================
#  TEST B: Gate Stacking — range_ok + vel_fast combined
# ====================================================================================

INSTRUMENTS = {
    "MYM_11": {
        "renko_file": "CBOT_MINI_MYM1!, 1S renko 11.csv",
        "is_start": "2025-08-07", "is_end": "2025-12-31",
        "oos_start": "2026-01-01", "oos_end": "2026-03-19",
        "label": "MYM brick 11",
    },
    "MYM_12": {
        "renko_file": "CBOT_MINI_MYM1!, 1S renko 12.csv",
        "is_start": "2025-05-19", "is_end": "2025-12-31",
        "oos_start": "2026-01-01", "oos_end": "2026-03-19",
        "label": "MYM brick 12",
    },
    "MYM_13": {
        "renko_file": "CBOT_MINI_MYM1!, 1S renko 13.csv",
        "is_start": "2025-04-09", "is_end": "2025-12-31",
        "oos_start": "2026-01-01", "oos_end": "2026-03-19",
        "label": "MYM brick 13",
    },
    "MYM_14": {
        "renko_file": "CBOT_MINI_MYM1!, 1S renko 14.csv",
        "is_start": "2025-03-07", "is_end": "2025-12-31",
        "oos_start": "2026-01-01", "oos_end": "2026-03-19",
        "label": "MYM brick 14",
    },
    "MYM_15": {
        "renko_file": "CBOT_MINI_MYM1!, 1S renko 15.csv",
        "is_start": "2025-01-06", "is_end": "2025-12-31",
        "oos_start": "2026-01-01", "oos_end": "2026-03-19",
        "label": "MYM brick 15",
    },
}

# Gate stacking combos to test
STACK_COMBOS = [
    # (label, p4_gates_list) — all gates in list must pass
    ("vel_fast_only",       ["vel_fast"]),
    ("range_ok_only",       ["range_ok"]),
    ("vel_fast+range_ok",   ["vel_fast", "range_ok"]),
    ("vel_vfast+range_ok",  ["vel_very_fast", "range_ok"]),
    ("vel_fast+no_exhaust", ["vel_fast", "no_exhaust"]),
    ("range_ok+no_exhaust", ["range_ok", "no_exhaust"]),
    ("none",                ["none"]),
]

STACK_ADX = [45, 50]
STACK_P6 = ["psar_dir", "ema_cross"]
STACK_N_BRICKS = [8, 9, 10, 11]
STACK_COOLDOWN = [40, 45, 50, 55]


def _combine_gates_stacked(gates, adx_thresh, p6_name, p4_names, osc_name=None):
    """Like _combine_gates_v4 but supports multiple P4 gates ANDed together."""
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

    # Stack ALL P4 gates
    for p4_name in p4_names:
        p4l, p4s = gates[f"p4:{p4_name}"]
        cl &= p4l; cs &= p4s

    # Oscillator
    if osc_name is not None:
        ol, os_ = gates[osc_name]
        cl &= ol; cs &= os_

    return cl, cs


def run_gate_stacking_worker(name, config):
    """Run gate stacking sweep for one instrument."""
    print(f"[{name}] Loading Renko + ALL indicators (v4)...", flush=True)
    df = _load_renko_all_indicators_v4(config["renko_file"])
    print(f"[{name}] Ready — {len(df)} bricks", flush=True)

    et_hours, et_minutes = _compute_et_hours(df.index)
    gates = _compute_all_gates_v4(df, et_hours)

    brick_up = df["brick_up"].values
    df["long_entry"] = False
    df["long_exit"] = False
    df["short_entry"] = False
    df["short_exit"] = False

    results = []
    total = len(STACK_COMBOS) * len(STACK_ADX) * len(STACK_P6) * len(STACK_N_BRICKS) * len(STACK_COOLDOWN)
    done = 0

    for combo_label, p4_list in STACK_COMBOS:
        for adx_t in STACK_ADX:
            for p6 in STACK_P6:
                gate_long, gate_short = _combine_gates_stacked(gates, adx_t, p6, p4_list)

                for nb in STACK_N_BRICKS:
                    for cd in STACK_COOLDOWN:
                        le, lx, se, sx = _generate_signal_arrays(
                            brick_up,
                            n_bricks=nb,
                            cooldown=cd,
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

                        stack_label = f"a{adx_t}_{p6}_{combo_label}"
                        results.append({
                            "instrument": name,
                            "brick": int(name.split("_")[1]),
                            "stack": stack_label,
                            "p6_gate": p6,
                            "p4_combo": combo_label,
                            "adx_thresh": adx_t,
                            "n_bricks": nb,
                            "cooldown": cd,
                            "is_pf": is_pf,
                            "is_trades": is_r["trades"],
                            "is_net": is_r["net"],
                            "is_wr": is_r["wr"],
                            "oos_pf": oos_pf,
                            "oos_trades": oos_r["trades"],
                            "oos_net": oos_r["net"],
                            "oos_wr": oos_r["wr"],
                            "decay_pct": decay,
                        })

                        done += 1

    print(f"[{name}] Complete — {done} results", flush=True)
    return results


def run_gate_stacking():
    """Run gate stacking sweep across all brick sizes."""
    n_per_brick = len(STACK_COMBOS) * len(STACK_ADX) * len(STACK_P6) * len(STACK_N_BRICKS) * len(STACK_COOLDOWN)
    total = n_per_brick * len(INSTRUMENTS)

    print(f"\n{'='*100}")
    print("  TEST B: Gate Stacking — Combining Best P4 Gates")
    print(f"{'='*100}")
    print(f"  Stacking combos : {[c[0] for c in STACK_COMBOS]}")
    print(f"  P6 gates        : {STACK_P6}")
    print(f"  ADX thresholds  : {STACK_ADX}")
    print(f"  n_bricks        : {STACK_N_BRICKS}")
    print(f"  cooldown        : {STACK_COOLDOWN}")
    print(f"  Per brick       : {n_per_brick} runs")
    print(f"  Total runs      : {total} ({total * 2} IS+OOS backtests)")
    print(f"  Workers         : {len(INSTRUMENTS)}")
    print()

    all_results = []

    with ProcessPoolExecutor(max_workers=len(INSTRUMENTS)) as pool:
        futures = {
            pool.submit(run_gate_stacking_worker, name, config): name
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

    # ── Summarize gate stacking ──
    MIN_OOS_TRADES = 10

    print(f"\n{'='*100}")
    print("  Gate Stacking Results — By Combo (avg across all bricks)")
    print(f"{'='*100}")
    print(f"\n  {'Combo':<25} | {'Avg PF':>7} {'Avg T':>6} {'N':>5} | {'Best PF':>8} {'Best Brick':>10}")
    print(f"  {'-'*75}")

    for combo_label, _ in STACK_COMBOS:
        viable = [r for r in all_results
                  if r["p4_combo"] == combo_label
                  and r["oos_trades"] >= MIN_OOS_TRADES
                  and not math.isinf(r["oos_pf"])]
        if not viable:
            print(f"  {combo_label:<25} | no viable results")
            continue

        avg_pf = sum(r["oos_pf"] for r in viable) / len(viable)
        avg_t = sum(r["oos_trades"] for r in viable) / len(viable)
        best = max(viable, key=lambda r: r["oos_pf"])

        print(f"  {combo_label:<25} | {avg_pf:>7.2f} {avg_t:>6.1f} {len(viable):>5} | "
              f"{best['oos_pf']:>8.2f} brick {best['brick']:>2}")

    # Per-brick detail
    for inst in sorted(INSTRUMENTS.keys()):
        cfg = INSTRUMENTS[inst]
        print(f"\n{'='*100}")
        print(f"  {cfg['label']} — Gate Stacking")
        print(f"{'='*100}")

        inst_res = [r for r in all_results if r["instrument"] == inst]
        viable = [r for r in inst_res if r["oos_trades"] >= MIN_OOS_TRADES]
        viable.sort(key=lambda r: (r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6), reverse=True)

        # Top 15
        print(f"\n  Top 15 (OOS trades >= {MIN_OOS_TRADES}):")
        print(f"  {'Stack':<40} {'Combo':<25} {'n':>2} {'cd':>3} | {'IS PF':>7} {'T':>4} | "
              f"{'OOS PF':>7} {'T':>4} {'WR%':>6} {'Net$':>8}")
        print(f"  {'-'*110}")
        for r in viable[:15]:
            print(f"  {r['stack']:<40} {r['p4_combo']:<25} {r['n_bricks']:>2} {r['cooldown']:>3} | "
                  f"{r['is_pf']:>7.2f} {r['is_trades']:>4} | "
                  f"{r['oos_pf']:>7.2f} {r['oos_trades']:>4} {r['oos_wr']:>5.1f}% {r['oos_net']:>8.2f}")

        # By combo (avg for this brick)
        print(f"\n  By combo (avg OOS PF, viable):")
        for combo_label, _ in STACK_COMBOS:
            cv = [r for r in viable if r["p4_combo"] == combo_label and not math.isinf(r["oos_pf"])]
            if cv:
                avg = sum(r["oos_pf"] for r in cv) / len(cv)
                avg_t = sum(r["oos_trades"] for r in cv) / len(cv)
                best = max(cv, key=lambda r: r["oos_pf"])
                print(f"    {combo_label:<25} avg PF={avg:>7.2f}  avg T={avg_t:>6.1f}  "
                      f"N={len(cv):>4}  best={best['oos_pf']:>7.2f}")

    # Compare stacked vs individual
    print(f"\n{'='*100}")
    print("  Stacking Value-Add: Does combining gates beat individual gates?")
    print(f"{'='*100}")

    for inst in sorted(INSTRUMENTS.keys()):
        cfg = INSTRUMENTS[inst]
        viable = [r for r in all_results
                  if r["instrument"] == inst
                  and r["oos_trades"] >= MIN_OOS_TRADES
                  and not math.isinf(r["oos_pf"])]
        if not viable:
            continue

        print(f"\n  {cfg['label']}:")

        baselines = {}
        for combo_label, _ in STACK_COMBOS:
            cv = [r for r in viable if r["p4_combo"] == combo_label]
            if cv:
                baselines[combo_label] = sum(r["oos_pf"] for r in cv) / len(cv)

        none_avg = baselines.get("none", 0)
        for combo_label, _ in STACK_COMBOS:
            if combo_label == "none":
                continue
            avg = baselines.get(combo_label, 0)
            delta = (avg - none_avg) / none_avg * 100 if none_avg > 0 else 0
            symbol = "+" if delta > 0 else ""
            print(f"    {combo_label:<25} avg PF={avg:>7.2f}  vs none={none_avg:>7.2f}  "
                  f"delta={symbol}{delta:.1f}%")

    return all_results


# ====================================================================================
#  Main
# ====================================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["wf", "stack", "both"], default="both",
                        help="Which test to run: wf=walk-forward, stack=gate stacking, both=all")
    args = parser.parse_args()

    out_path = ROOT / "ai_context" / "mym_sweep_v5_results.json"
    out_path.parent.mkdir(exist_ok=True)

    combined = {}

    if args.test in ("wf", "both"):
        wf_results = run_walk_forward()
        combined["walk_forward"] = wf_results

    if args.test in ("stack", "both"):
        stack_results = run_gate_stacking()
        combined["gate_stacking"] = stack_results

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\nSaved results -> {out_path}")


if __name__ == "__main__":
    main()
