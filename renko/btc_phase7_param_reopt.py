#!/usr/bin/env python3
"""
btc_phase7_param_reopt.py -- BTC Phase 7: Entry Param Re-Optimization (Long Only)

BTC003 entry (R007 + FLIP_ST + BB_break) with HTF ADX>=35 is locked.
Re-sweep n_bricks and cooldown with finer granularity to find the optimal
combination now that the gate stack has changed from Phase 4.

Also tests whether FLIP and BB_break cooldowns should differ from R001 cooldown.

Sweep dimensions:
    n_bricks:     [2, 3, 4, 5, 6, 7]
    r001_cooldown: [15, 20, 25, 30, 35, 40]
    other_cooldown: [same, 15, 20, 30]  -- cooldown for FLIP and BB_break

Uses ProcessPoolExecutor -- one worker per n_bricks value.

Usage:
    python renko/btc_phase7_param_reopt.py
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

# -- Instrument config ---------------------------------------------------------

LTF_FILE   = "OANDA_BTCUSD.SPOT.US, 1S renko 150.csv"
HTF_FILE   = "OANDA_BTCUSD.SPOT.US, 1S renko 300.csv"
IS_START   = "2024-06-04"
IS_END     = "2025-09-30"
OOS_START  = "2025-10-01"
OOS_END    = "2026-03-19"
COMMISSION = 0.0046
CAPITAL    = 1000.0
QTY_VALUE  = 20
VOL_MAX    = 1.5
ADX_THRESH = 30
HTF_ADX_THRESH = 35

# -- Param grid ----------------------------------------------------------------

N_BRICKS_RANGE   = [2, 3, 4, 5, 6, 7]
R001_CD_RANGE    = [15, 20, 25, 30, 35, 40]
OTHER_CD_OPTIONS = ["same", 15, 20, 30]  # "same" = use r001_cd


# -- Data loading --------------------------------------------------------------

def _load_ltf_data():
    sys.path.insert(0, str(ROOT))
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    from renko.phase6_enrichment import add_phase6_indicators
    df = load_renko_export(LTF_FILE)
    add_renko_indicators(df)
    add_phase6_indicators(df, include_mk=False)
    return df


def _load_htf_data():
    sys.path.insert(0, str(ROOT))
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    df = load_renko_export(HTF_FILE)
    add_renko_indicators(df)
    return df


# -- Gate computation ----------------------------------------------------------

def _compute_gates(df_ltf, df_htf):
    sys.path.insert(0, str(ROOT))
    from renko.phase6_sweep import _compute_gate_arrays

    n = len(df_ltf)
    gate = np.ones(n, dtype=bool)

    p6_long, _ = _compute_gate_arrays(df_ltf, "psar_dir")
    gate &= p6_long

    adx = df_ltf["adx"].values
    gate &= (np.isnan(adx) | (adx >= ADX_THRESH))

    vr = df_ltf["vol_ratio"].values
    gate &= (np.isnan(vr) | (vr <= VOL_MAX))

    htf_adx = df_htf["adx"].values
    htf_gate = np.isnan(htf_adx) | (htf_adx >= HTF_ADX_THRESH)

    htf_frame = pd.DataFrame({
        "t": df_htf.index.values,
        "g": htf_gate.astype(float),
    }).sort_values("t")
    ltf_frame = pd.DataFrame({"t": df_ltf.index.values}).sort_values("t")
    merged = pd.merge_asof(ltf_frame, htf_frame, on="t", direction="backward")
    g = merged["g"].values
    htf_aligned = np.where(np.isnan(g), True, g > 0.5).astype(bool)
    gate &= htf_aligned

    return gate


# -- Trigger arrays ------------------------------------------------------------

def _compute_st_flip(df):
    n = len(df)
    st = df["st_dir"].values
    flip = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not np.isnan(st[i]) and not np.isnan(st[i-1]):
            flip[i] = st[i] > 0 and st[i-1] <= 0
    return flip


def _compute_bb_break(df):
    n = len(df)
    close = df["Close"].values.astype(float)
    bb_u = df["bb_upper"].values
    sig = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not np.isnan(bb_u[i]) and not np.isnan(bb_u[i-1]):
            sig[i] = close[i] > bb_u[i] and close[i-1] <= bb_u[i-1]
    return sig


# -- Signal generator ----------------------------------------------------------

def _gen_signals(df, gate, st_flip, bb_break, n_bricks, r001_cd, other_cd):
    """BTC003 entry with variable cooldowns. Exit on first down brick."""
    n = len(df)
    brick_up = df["brick_up"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)

    in_pos = False
    last_r001 = -999_999
    last_flip = -999_999
    last_bb = -999_999
    warmup = max(n_bricks + 1, 200)

    for i in range(warmup, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
        if in_pos:
            continue
        if not gate[i]:
            continue

        triggered = False

        # R002: reversal (no cooldown)
        prev = brick_up[i - n_bricks : i]
        prev_all_down = bool(not np.any(prev))
        if prev_all_down and up:
            triggered = True

        # R001: momentum (r001_cd)
        if not triggered and (i - last_r001) >= r001_cd:
            window = brick_up[i - n_bricks + 1 : i + 1]
            if bool(np.all(window)):
                triggered = True
                last_r001 = i

        # FLIP: supertrend (other_cd)
        if not triggered and up and st_flip[i] and (i - last_flip) >= other_cd:
            triggered = True
            last_flip = i

        # BBRK: BB breakout (other_cd)
        if not triggered and up and bb_break[i] and (i - last_bb) >= other_cd:
            triggered = True
            last_bb = i

        if triggered:
            entry[i] = True
            in_pos = True

    return entry, exit_


# -- Backtest runner -----------------------------------------------------------

def _run_backtest(df, entry, exit_, start, end):
    sys.path.insert(0, str(ROOT))
    from engine import BacktestConfig, run_backtest

    df_sig = df.copy()
    df_sig["long_entry"] = entry
    df_sig["long_exit"] = exit_

    cfg = BacktestConfig(
        initial_capital=CAPITAL, commission_pct=COMMISSION, slippage_ticks=0,
        qty_type="cash", qty_value=QTY_VALUE, pyramiding=1,
        start_date=start, end_date=end,
        take_profit_pct=0.0, stop_loss_pct=0.0,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest(df_sig, cfg)

    pf = kpis.get("profit_factor", 0.0) or 0.0
    return {
        "pf":     float("inf") if math.isinf(pf) else float(pf),
        "net":    float(kpis.get("net_profit", 0.0) or 0.0),
        "trades": int(kpis.get("total_trades", 0) or 0),
        "wr":     float(kpis.get("win_rate", 0.0) or 0.0),
        "dd":     float(kpis.get("max_drawdown_pct", 0.0) or 0.0),
    }


# -- Worker: sweep one n_bricks value -----------------------------------------

def _sweep_n_bricks(n_bricks):
    label = f"n={n_bricks}"
    print(f"  [{label}] Loading data...", flush=True)

    df_ltf = _load_ltf_data()
    df_htf = _load_htf_data()
    gate = _compute_gates(df_ltf, df_htf)
    st_flip = _compute_st_flip(df_ltf)
    bb_break = _compute_bb_break(df_ltf)

    configs = []
    for r001_cd in R001_CD_RANGE:
        for other_opt in OTHER_CD_OPTIONS:
            other_cd = r001_cd if other_opt == "same" else other_opt
            configs.append((r001_cd, other_cd, other_opt))

    total = len(configs)
    print(f"  [{label}] {total} configs to run", flush=True)

    results = []
    for idx, (r001_cd, other_cd, other_opt) in enumerate(configs):
        e, x = _gen_signals(df_ltf, gate, st_flip, bb_break,
                            n_bricks, r001_cd, other_cd)

        is_r  = _run_backtest(df_ltf, e, x, IS_START, IS_END)
        oos_r = _run_backtest(df_ltf, e, x, OOS_START, OOS_END)

        is_pf = is_r["pf"]
        oos_pf = oos_r["pf"]
        decay = ((oos_pf - is_pf) / is_pf * 100) \
                if is_pf > 0 and not math.isinf(is_pf) else float("nan")

        cd_label = f"same({r001_cd})" if other_opt == "same" else str(other_cd)

        results.append({
            "n_bricks":    n_bricks,
            "r001_cd":     r001_cd,
            "other_cd":    other_cd,
            "other_opt":   str(other_opt),
            "is_pf":       is_pf,
            "is_trades":   is_r["trades"],
            "is_wr":       is_r["wr"],
            "is_net":      is_r["net"],
            "oos_pf":      oos_pf,
            "oos_trades":  oos_r["trades"],
            "oos_wr":      oos_r["wr"],
            "oos_net":     oos_r["net"],
            "oos_dd":      oos_r["dd"],
            "decay_pct":   decay,
        })

        if (idx + 1) % 12 == 0 or idx + 1 == total:
            pf_s = f"{oos_pf:.2f}" if not math.isinf(oos_pf) else "inf"
            print(f"  [{label}] {idx+1:>3}/{total} | r001_cd={r001_cd:>2} other_cd={cd_label:<10} | "
                  f"OOS PF={pf_s:>8} T={oos_r['trades']:>4}", flush=True)

    print(f"  [{label}] Done -- {len(results)} results", flush=True)
    return results


# -- Summary -------------------------------------------------------------------

def _summarize(all_results):
    print(f"\n{'='*100}")
    print("  BTC Phase 7 -- Entry Param Re-Optimization (Long Only)")
    print(f"  Entry: R007 + FLIP_ST + BB_break | Gates: psar+ADX30+vol1.5+HTF35")
    print(f"{'='*100}")

    # Current BTC003 baseline
    bl = [r for r in all_results if r["n_bricks"] == 2 and r["r001_cd"] == 30
          and r["other_opt"] == "same"]
    if bl:
        b = bl[0]
        pf_s = f"{b['oos_pf']:.2f}" if not math.isinf(b['oos_pf']) else "inf"
        print(f"\n  BASELINE (n=2, cd=30, same): IS PF={b['is_pf']:.2f} T={b['is_trades']} | "
              f"OOS PF={pf_s} T={b['oos_trades']} WR={b['oos_wr']:.1f}% Net={b['oos_net']:.2f}")

    # Top 20 by OOS PF (trades >= 10)
    viable = [r for r in all_results if r["oos_trades"] >= 10]
    viable.sort(key=lambda r: r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6, reverse=True)

    print(f"\n  --- Top 20 by OOS PF (trades >= 10) ---")
    print(f"  {'n':>2} {'r001cd':>6} {'othcd':>6} | {'IS PF':>7} {'T':>4} {'WR%':>6} | "
          f"{'OOS PF':>8} {'T':>4} {'WR%':>6} {'Net':>8} {'Decay':>7}")
    print(f"  {'-'*80}")
    for r in viable[:20]:
        pf_s = f"{r['oos_pf']:>8.2f}" if not math.isinf(r["oos_pf"]) else "     inf"
        is_pf_s = f"{r['is_pf']:>7.2f}" if not math.isinf(r["is_pf"]) else "    inf"
        dec_s = f"{r['decay_pct']:>+6.1f}%" if not math.isnan(r["decay_pct"]) else "    NaN"
        print(f"  {r['n_bricks']:>2} {r['r001_cd']:>6} {r['other_cd']:>6} | "
              f"{is_pf_s} {r['is_trades']:>4} {r['is_wr']:>5.1f}% | "
              f"{pf_s} {r['oos_trades']:>4} {r['oos_wr']:>5.1f}% {r['oos_net']:>8.2f} {dec_s}")

    # Top 20 by trade count (PF >= 10)
    high_t = [r for r in all_results if r["oos_pf"] >= 10 and r["oos_trades"] >= 20]
    high_t.sort(key=lambda r: r["oos_trades"], reverse=True)

    print(f"\n  --- Top 20 by OOS Trades (PF >= 10) ---")
    print(f"  {'n':>2} {'r001cd':>6} {'othcd':>6} | {'IS PF':>7} {'T':>4} {'WR%':>6} | "
          f"{'OOS PF':>8} {'T':>4} {'WR%':>6} {'Net':>8}")
    print(f"  {'-'*80}")
    for r in high_t[:20]:
        pf_s = f"{r['oos_pf']:>8.2f}" if not math.isinf(r["oos_pf"]) else "     inf"
        is_pf_s = f"{r['is_pf']:>7.2f}" if not math.isinf(r["is_pf"]) else "    inf"
        print(f"  {r['n_bricks']:>2} {r['r001_cd']:>6} {r['other_cd']:>6} | "
              f"{is_pf_s} {r['is_trades']:>4} {r['is_wr']:>5.1f}% | "
              f"{pf_s} {r['oos_trades']:>4} {r['oos_wr']:>5.1f}% {r['oos_net']:>8.2f}")

    # Best per n_bricks
    print(f"\n  --- Best per n_bricks ---")
    for n in N_BRICKS_RANGE:
        n_results = [r for r in all_results if r["n_bricks"] == n and r["oos_trades"] >= 10]
        if n_results:
            best = max(n_results, key=lambda r: r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6)
            pf_s = f"{best['oos_pf']:.2f}" if not math.isinf(best['oos_pf']) else "inf"
            most_t = max(n_results, key=lambda r: r["oos_trades"])
            print(f"    n={n}: best PF={pf_s} T={best['oos_trades']} (r001cd={best['r001_cd']} othcd={best['other_cd']}) | "
                  f"most trades={most_t['oos_trades']} PF={most_t['oos_pf']:.2f}")

    # Same vs different cooldown analysis
    print(f"\n  --- Same vs Different Cooldown ---")
    same_cd = [r for r in all_results if r["other_opt"] == "same" and r["oos_trades"] >= 10]
    diff_cd = [r for r in all_results if r["other_opt"] != "same" and r["oos_trades"] >= 10]
    if same_cd:
        finite = [r["oos_pf"] for r in same_cd if not math.isinf(r["oos_pf"])]
        print(f"    Same cooldown:  avg PF={np.mean(finite):.2f}, avg T={np.mean([r['oos_trades'] for r in same_cd]):.1f} (n={len(same_cd)})")
    if diff_cd:
        finite = [r["oos_pf"] for r in diff_cd if not math.isinf(r["oos_pf"])]
        print(f"    Diff cooldown:  avg PF={np.mean(finite):.2f}, avg T={np.mean([r['oos_trades'] for r in diff_cd]):.1f} (n={len(diff_cd)})")


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-parallel", action="store_true")
    args = parser.parse_args()

    out_path = ROOT / "ai_context" / "btc_phase7_param_results.json"
    out_path.parent.mkdir(exist_ok=True)

    configs_per_n = len(R001_CD_RANGE) * len(OTHER_CD_OPTIONS)
    total = configs_per_n * len(N_BRICKS_RANGE)

    print("BTC Phase 7: Entry Param Re-Optimization (Long Only)")
    print(f"  Entry      : R007 + FLIP_ST + BB_break")
    print(f"  Gates      : psar_dir + ADX>={ADX_THRESH} + vol<={VOL_MAX} + HTF ADX>={HTF_ADX_THRESH}")
    print(f"  n_bricks   : {N_BRICKS_RANGE}")
    print(f"  r001_cd    : {R001_CD_RANGE}")
    print(f"  other_cd   : {OTHER_CD_OPTIONS}")
    print(f"  Total runs : {total} ({configs_per_n} per n x {len(N_BRICKS_RANGE)} n)")
    print(f"  IS period  : {IS_START} -> {IS_END}")
    print(f"  OOS period : {OOS_START} -> {OOS_END}")
    print()

    all_results = []

    if args.no_parallel:
        for n in N_BRICKS_RANGE:
            all_results.extend(_sweep_n_bricks(n))
    else:
        with ProcessPoolExecutor(max_workers=len(N_BRICKS_RANGE)) as pool:
            futures = {
                pool.submit(_sweep_n_bricks, n): n
                for n in N_BRICKS_RANGE
            }
            for future in as_completed(futures):
                n = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    print(f"  [n={n}] finished -- {len(results)} records")
                except Exception as exc:
                    import traceback
                    print(f"  [n={n}] FAILED: {exc}")
                    traceback.print_exc()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved {len(all_results)} results -> {out_path}")

    _summarize(all_results)


if __name__ == "__main__":
    main()
