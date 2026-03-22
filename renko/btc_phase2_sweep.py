#!/usr/bin/env python3
"""
btc_phase2_sweep.py — BTC Phase 2: ADX + P6 + Oscillator Stacking (Long Only)

Stacks top 3 P6 gates from Phase 1 with ADX thresholds and BC oscillators.
No session gate (24/7 market).

Dimensions:
  P6 gates:    psar_dir, escgo_cross, kama_slope (top 3 from Phase 1)
  ADX thresh:  {20, 25, 30}
  Oscillators: {none, sto_tso, macd_lc}
  n_bricks:    {2, 3, 4, 5}
  cooldown:    {10, 20, 30}

Per P6 gate: 3 ADX x 3 osc x 12 params = 108 runs
Total: 3 P6 x 108 = 324 runs (648 IS+OOS backtests)

Uses ProcessPoolExecutor — one worker per P6 gate.

Usage:
  python renko/btc_phase2_sweep.py
  python renko/btc_phase2_sweep.py --no-parallel
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

# ── Instrument config ──────────────────────────────────────────────────────────

RENKO_FILE = "OANDA_BTCUSD.SPOT.US, 1S renko 150.csv"
IS_START   = "2024-06-04"
IS_END     = "2025-09-30"
OOS_START  = "2025-10-01"
OOS_END    = "2026-03-19"
COMMISSION = 0.0046
CAPITAL    = 1000.0
QTY_VALUE  = 20

# ── Sweep dimensions ──────────────────────────────────────────────────────────

P6_GATES       = ["psar_dir", "escgo_cross", "kama_slope"]
ADX_THRESHOLDS = [20, 25, 30]
OSC_CHOICES    = [None, "sto_tso", "macd_lc"]
VOL_MAX        = 1.5

PARAM_GRID = {
    "n_bricks": [2, 3, 4, 5],
    "cooldown": [10, 20, 30],
}


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_data() -> pd.DataFrame:
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

    df = load_renko_export(RENKO_FILE)
    add_renko_indicators(df)
    add_phase6_indicators(df, include_mk=False)

    # BC STO
    try:
        sto = calc_bc_swing_trade_oscillator(df)
        df["_bc_sto_mf"] = sto["bc_sto_main_force"].shift(1).values
        df["_bc_sto_ll"] = sto["bc_sto_life_line"].shift(1).values
    except Exception as e:
        print(f"  WARN: STO failed: {e}")
        df["_bc_sto_mf"] = np.nan
        df["_bc_sto_ll"] = np.nan

    # BC TSO
    try:
        tso = calc_bc_trend_swing_oscillator(df)
        df["_bc_tso_pink"] = tso["bc_tso_pink_hist"].shift(1).values
    except Exception as e:
        print(f"  WARN: TSO failed: {e}")
        df["_bc_tso_pink"] = np.nan

    # BC MACD Wave Signal Pro
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


# ── Gate computation ───────────────────────────────────────────────────────────

def _compute_combined_gate(
    df: pd.DataFrame,
    p6_gate: str,
    adx_thresh: int,
    osc: str | None,
) -> np.ndarray:
    """Compute AND-combined long gate: P6 + ADX + vol_ratio + optional oscillator."""
    sys.path.insert(0, str(ROOT))
    from renko.phase6_sweep import _compute_gate_arrays

    n = len(df)
    gate = np.ones(n, dtype=bool)

    # P6 gate (long side only)
    p6_long, _ = _compute_gate_arrays(df, p6_gate)
    gate &= p6_long

    # ADX gate (NaN-pass)
    adx = df["adx"].values
    adx_nan = np.isnan(adx)
    gate &= (adx_nan | (adx >= adx_thresh))

    # Vol ratio gate (NaN-pass)
    vr = df["vol_ratio"].values
    vr_nan = np.isnan(vr)
    gate &= (vr_nan | (vr <= VOL_MAX))

    # Oscillator gate
    if osc == "sto_tso":
        sto_mf = df["_bc_sto_mf"].values
        sto_ll = df["_bc_sto_ll"].values
        sto_nan = np.isnan(sto_mf) | np.isnan(sto_ll)
        sto_long = sto_nan | (sto_mf > sto_ll)

        tso_pink = df["_bc_tso_pink"].values.astype(float)
        tso_nan = np.isnan(tso_pink)
        tso_long = tso_nan | (tso_pink > 0.5)

        gate &= (sto_long & tso_long)

    elif osc == "macd_lc":
        macd_st = df["_bc_macd_state"].values
        bc_lc = df["_bc_lc"].values
        ms_nan = np.isnan(macd_st)
        lc_nan = np.isnan(bc_lc)
        ms_int = np.where(ms_nan, -1, macd_st).astype(int)
        ms_long = ms_nan | np.isin(ms_int, [0, 3])
        lc_long = lc_nan | (bc_lc > 0)
        gate &= (ms_long & lc_long)

    return gate


# ── Signal generator (long only) ──────────────────────────────────────────────

def _generate_signals_long_only(
    df: pd.DataFrame,
    n_bricks: int,
    cooldown: int,
    gate_long_ok: np.ndarray,
) -> pd.DataFrame:
    """R007 logic — long entries only."""
    n = len(df)
    brick_up = df["brick_up"].values

    long_entry = np.zeros(n, dtype=bool)
    long_exit  = np.zeros(n, dtype=bool)

    in_position   = False
    last_r001_bar = -999_999
    warmup        = max(n_bricks + 1, 200)

    for i in range(warmup, n):
        up = bool(brick_up[i])

        if in_position:
            if not up:
                long_exit[i] = True
                in_position = False

        if in_position:
            continue

        # R002 long: n down then up (reversal)
        prev = brick_up[i - n_bricks : i]
        prev_all_down = bool(not np.any(prev))

        if prev_all_down and up:
            if gate_long_ok[i]:
                long_entry[i] = True
                in_position = True
            continue

        # R001 long: n consecutive up bricks (trend)
        if (i - last_r001_bar) < cooldown:
            continue

        window = brick_up[i - n_bricks + 1 : i + 1]
        all_up = bool(np.all(window))

        if all_up and gate_long_ok[i]:
            long_entry[i] = True
            in_position = True
            last_r001_bar = i

    df["long_entry"] = long_entry
    df["long_exit"]  = long_exit
    return df


# ── Backtest runner (long only) ────────────────────────────────────────────────

def _run_backtest(df_sig, start, end):
    sys.path.insert(0, str(ROOT))
    from engine import BacktestConfig, run_backtest

    cfg = BacktestConfig(
        initial_capital=CAPITAL,
        commission_pct=COMMISSION,
        slippage_ticks=0,
        qty_type="cash",
        qty_value=QTY_VALUE,
        pyramiding=1,
        start_date=start,
        end_date=end,
        take_profit_pct=0.0,
        stop_loss_pct=0.0,
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


# ── Worker: sweep one P6 gate across all ADX x osc x params ──────────────────

def _sweep_p6_gate(p6_gate: str) -> list:
    """Run all combos for one P6 gate. Called in a subprocess."""
    print(f"  [{p6_gate}] Loading data...", flush=True)
    df = _load_data()

    keys = list(PARAM_GRID.keys())
    param_combos = [dict(zip(keys, v)) for v in itertools.product(*PARAM_GRID.values())]

    results = []
    total = len(ADX_THRESHOLDS) * len(OSC_CHOICES) * len(param_combos)
    done = 0

    for adx_thresh in ADX_THRESHOLDS:
        for osc in OSC_CHOICES:
            gate_long = _compute_combined_gate(df, p6_gate, adx_thresh, osc)
            osc_label = osc or "none"

            for pc in param_combos:
                df_sig = _generate_signals_long_only(
                    df.copy(), pc["n_bricks"], pc["cooldown"], gate_long,
                )

                is_r  = _run_backtest(df_sig, IS_START, IS_END)
                oos_r = _run_backtest(df_sig, OOS_START, OOS_END)

                is_pf  = is_r["pf"]
                oos_pf = oos_r["pf"]
                decay  = ((oos_pf - is_pf) / is_pf * 100) \
                         if is_pf > 0 and not math.isinf(is_pf) else float("nan")

                label = f"{p6_gate}_a{adx_thresh}_{osc_label}"

                results.append({
                    "p6_gate":    p6_gate,
                    "adx_thresh": adx_thresh,
                    "osc":        osc_label,
                    "label":      label,
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
                if done % 36 == 0 or done == total:
                    print(
                        f"  [{p6_gate}] {done:>3}/{total} | {label:<35} "
                        f"n={pc['n_bricks']} cd={pc['cooldown']:>2} | "
                        f"IS PF={is_pf:>7.2f} T={is_r['trades']:>4} | "
                        f"OOS PF={oos_pf:>7.2f} T={oos_r['trades']:>4}",
                        flush=True,
                    )

    best = max(results, key=lambda r: r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6)
    print(f"  [{p6_gate}] Complete -- {len(results)} results | "
          f"Best: {best['label']} n={best['n_bricks']} cd={best['cooldown']} "
          f"OOS PF={best['oos_pf']:.2f} T={best['oos_trades']}",
          flush=True)
    return results


# ── Summary ────────────────────────────────────────────────────────────────────

def _summarize(all_results: list) -> None:
    print(f"\n{'='*90}")
    print("  BTC Phase 2 -- ADX + P6 + Oscillator Stacking (Long Only)")
    print(f"{'='*90}")

    # Phase 1 baseline for reference
    P1_BASELINE_PF = 14.58  # Phase 1 baseline best OOS PF
    P1_BEST = {
        "psar_dir":    21.79,
        "escgo_cross": 20.44,
        "kama_slope":  19.89,
    }

    # By P6 gate: avg and best
    for p6 in P6_GATES:
        p6_res = [r for r in all_results if r["p6_gate"] == p6]
        viable = [r for r in p6_res if r["oos_trades"] >= 10]
        if not viable:
            continue

        p1_best = P1_BEST.get(p6, 0)

        print(f"\n  --- {p6} (Phase 1 best: {p1_best:.2f}) ---")

        # By ADX threshold
        print(f"\n  By ADX threshold (avg OOS PF, trades >= 10):")
        for adx in ADX_THRESHOLDS:
            av = [r for r in viable if r["adx_thresh"] == adx]
            if av:
                avg = sum(r["oos_pf"] for r in av if not math.isinf(r["oos_pf"])) / max(
                    len([r for r in av if not math.isinf(r["oos_pf"])]), 1)
                avg_t = sum(r["oos_trades"] for r in av) / len(av)
                print(f"    ADX>={adx:<3}  avg PF={avg:>8.2f}  avg T={avg_t:>6.1f}  N={len(av):>3}")

        # By oscillator
        print(f"\n  By oscillator (avg OOS PF, trades >= 10):")
        for osc in ["none", "sto_tso", "macd_lc"]:
            ov = [r for r in viable if r["osc"] == osc]
            if ov:
                avg = sum(r["oos_pf"] for r in ov if not math.isinf(r["oos_pf"])) / max(
                    len([r for r in ov if not math.isinf(r["oos_pf"])]), 1)
                avg_t = sum(r["oos_trades"] for r in ov) / len(ov)
                print(f"    {osc:<10}  avg PF={avg:>8.2f}  avg T={avg_t:>6.1f}  N={len(ov):>3}")

        # Top 5 for this gate
        viable.sort(key=lambda r: r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6,
                     reverse=True)
        print(f"\n  Top 5:")
        print(f"  {'Config':<35} {'n':>2} {'cd':>3} | {'IS PF':>7} {'T':>4} | "
              f"{'OOS PF':>8} {'T':>4} {'WR%':>6} {'Decay':>7}")
        print(f"  {'-'*85}")
        for r in viable[:5]:
            dec_s = f"{r['decay_pct']:>+6.1f}%" if not math.isnan(r["decay_pct"]) else "   NaN"
            beat = " <<BEAT" if r["oos_pf"] > p1_best else ""
            print(f"  {r['label']:<35} {r['n_bricks']:>2} {r['cooldown']:>3} | "
                  f"{r['is_pf']:>7.2f} {r['is_trades']:>4} | "
                  f"{r['oos_pf']:>8.2f} {r['oos_trades']:>4} "
                  f"{r['oos_wr']:>5.1f}% {dec_s}{beat}")

    # Overall top 10
    all_viable = [r for r in all_results if r["oos_trades"] >= 10]
    all_viable.sort(key=lambda r: r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6,
                    reverse=True)

    print(f"\n{'='*90}")
    print("  Overall Top 10 (OOS trades >= 10)")
    print(f"{'='*90}")
    print(f"  {'Config':<35} {'n':>2} {'cd':>3} | {'IS PF':>7} {'T':>4} | "
          f"{'OOS PF':>8} {'T':>4} {'WR%':>6} {'Decay':>7}")
    print(f"  {'-'*85}")
    for r in all_viable[:10]:
        dec_s = f"{r['decay_pct']:>+6.1f}%" if not math.isnan(r["decay_pct"]) else "   NaN"
        print(f"  {r['label']:<35} {r['n_bricks']:>2} {r['cooldown']:>3} | "
              f"{r['is_pf']:>7.2f} {r['is_trades']:>4} | "
              f"{r['oos_pf']:>8.2f} {r['oos_trades']:>4} "
              f"{r['oos_wr']:>5.1f}% {dec_s}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-parallel", action="store_true")
    args = parser.parse_args()

    out_path = ROOT / "ai_context" / "btc_phase2_results.json"
    out_path.parent.mkdir(exist_ok=True)

    n_params = len(list(itertools.product(*PARAM_GRID.values())))
    per_gate = len(ADX_THRESHOLDS) * len(OSC_CHOICES) * n_params
    total = len(P6_GATES) * per_gate

    print("BTC Phase 2: ADX + P6 + Oscillator Stacking (Long Only)")
    print(f"  Instrument   : BTCUSD $150 Renko")
    print(f"  IS period    : {IS_START} -> {IS_END}")
    print(f"  OOS period   : {OOS_START} -> {OOS_END}")
    print(f"  P6 gates     : {P6_GATES}")
    print(f"  ADX thresholds: {ADX_THRESHOLDS}")
    print(f"  Oscillators  : {OSC_CHOICES}")
    print(f"  Param combos : {n_params}")
    print(f"  Per gate     : {per_gate} runs")
    print(f"  Total runs   : {total} ({total * 2} IS+OOS backtests)")
    print(f"  Output       : {out_path}")
    print()

    all_results = []

    if args.no_parallel:
        for p6 in P6_GATES:
            all_results.extend(_sweep_p6_gate(p6))
    else:
        with ProcessPoolExecutor(max_workers=len(P6_GATES)) as pool:
            futures = {pool.submit(_sweep_p6_gate, p6): p6 for p6 in P6_GATES}
            for future in as_completed(futures):
                p6 = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    print(f"  [{p6}] finished -- {len(results)} records")
                except Exception as exc:
                    import traceback
                    print(f"  [{p6}] FAILED: {exc}")
                    traceback.print_exc()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved {len(all_results)} results -> {out_path}")

    _summarize(all_results)


if __name__ == "__main__":
    main()
