#!/usr/bin/env python3
"""
btc_phase1_sweep.py — BTC Phase 1: P6 Gate Discovery (Long Only)

Sweeps all P6 indicator gates on R007 base logic (R001 + R002) for BTCUSD.
Long only — no short entries. 24/7 market — no session gate.

Gates (20): baseline + 19 indicator gates
Param grid: n_bricks={2,3,4,5} x cooldown={10,20,30} = 12 combos
Total: 20 gates x 12 params = 240 runs (480 IS+OOS backtests)

Uses ProcessPoolExecutor to parallelize gate groups across cores.

Usage:
  python renko/btc_phase1_sweep.py
  python renko/btc_phase1_sweep.py --no-parallel
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
QTY_VALUE  = 20  # $20 cash per trade

# ── Gate definitions ───────────────────────────────────────────────────────────

GATES = [
    "baseline",
    # Tier 1: Built-in
    "rsi_dir", "bb_pct_b", "chop_trend", "psar_dir", "kama_slope",
    "sq_mom", "stoch_cross", "cmf_dir", "mfi_dir", "obv_trend",
    "ema_cross", "macd_hist_dir",
    # Tier 2: Standalone
    "cci_dir", "ichi_cloud", "wpr_dir", "donch_mid",
    # Tier 3: Complex
    "escgo_cross", "ddl_dir", "motn_dx",
]

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

    df = load_renko_export(RENKO_FILE)
    add_renko_indicators(df)
    add_phase6_indicators(df, include_mk=False)
    return df


# ── Gate computation (reuse Phase 6 logic) ─────────────────────────────────────

def _compute_gate_long(df: pd.DataFrame, gate_name: str) -> np.ndarray:
    """Return boolean array: True where long entry is allowed by this gate."""
    sys.path.insert(0, str(ROOT))
    from renko.phase6_sweep import _compute_gate_arrays
    gate_long, _ = _compute_gate_arrays(df, gate_name)
    return gate_long


# ── Signal generator (long only) ──────────────────────────────────────────────

def _generate_signals_long_only(
    df: pd.DataFrame,
    n_bricks: int,
    cooldown: int,
    gate_long_ok: np.ndarray,
) -> pd.DataFrame:
    """
    R007 logic (R001 + R002) — long entries only.

    R001 long: n consecutive up bricks (trend continuation) → long entry
    R002 long: n consecutive down bricks then up (reversal) → long entry
    Exit: first down brick
    """
    n = len(df)
    brick_up = df["brick_up"].values

    long_entry = np.zeros(n, dtype=bool)
    long_exit  = np.zeros(n, dtype=bool)

    in_position   = False
    last_r001_bar = -999_999
    warmup        = max(n_bricks + 1, 200)

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # Exit: first down brick while in long position
        if in_position:
            if not up:
                long_exit[i] = True
                in_position = False

        if in_position:
            continue

        # R002 long: n down bricks followed by up → long (reversal)
        prev = brick_up[i - n_bricks : i]
        prev_all_down = bool(not np.any(prev))

        if prev_all_down and up:
            if gate_long_ok[i]:
                long_entry[i] = True
                in_position = True
            continue

        # R001 long: n consecutive up bricks (including current) → long (trend)
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


# ── Worker: sweep one gate across all params ──────────────────────────────────

def _sweep_gate(gate_name: str) -> list:
    """Run all param combos for one gate. Called in a subprocess."""
    print(f"  [{gate_name}] Loading data...", flush=True)
    df = _load_data()
    gate_long = _compute_gate_long(df, gate_name)

    keys = list(PARAM_GRID.keys())
    combos = [dict(zip(keys, v)) for v in itertools.product(*PARAM_GRID.values())]

    results = []
    for pc in combos:
        df_sig = _generate_signals_long_only(
            df.copy(), pc["n_bricks"], pc["cooldown"], gate_long,
        )

        is_r  = _run_backtest(df_sig, IS_START, IS_END)
        oos_r = _run_backtest(df_sig, OOS_START, OOS_END)

        is_pf  = is_r["pf"]
        oos_pf = oos_r["pf"]
        decay  = ((oos_pf - is_pf) / is_pf * 100) \
                 if is_pf > 0 and not math.isinf(is_pf) else float("nan")

        results.append({
            "gate":       gate_name,
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

    best = max(results, key=lambda r: r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6)
    print(f"  [{gate_name}] Done — best OOS PF={best['oos_pf']:.2f} "
          f"T={best['oos_trades']} WR={best['oos_wr']:.1f}%", flush=True)
    return results


# ── Summary ────────────────────────────────────────────────────────────────────

def _summarize(all_results: list) -> None:
    # Get baseline results for comparison
    baseline = [r for r in all_results if r["gate"] == "baseline"]
    baseline_by_params = {}
    for r in baseline:
        baseline_by_params[(r["n_bricks"], r["cooldown"])] = r["oos_pf"]

    # Average OOS PF per gate (trades >= 10)
    print(f"\n{'='*80}")
    print("  BTC Phase 1 — Gate Discovery (Long Only)")
    print(f"{'='*80}")

    gate_avgs = []
    for gate in GATES:
        viable = [r for r in all_results
                  if r["gate"] == gate and r["oos_trades"] >= 10]
        if viable:
            avg_pf = sum(r["oos_pf"] for r in viable
                         if not math.isinf(r["oos_pf"])) / max(
                len([r for r in viable if not math.isinf(r["oos_pf"])]), 1)
            avg_t = sum(r["oos_trades"] for r in viable) / len(viable)
            best = max(viable, key=lambda r: r["oos_pf"]
                       if not math.isinf(r["oos_pf"]) else 1e6)
            gate_avgs.append((gate, avg_pf, avg_t, best, len(viable)))

    gate_avgs.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  {'Gate':<20} {'Avg PF':>8} {'Avg T':>6} {'N':>3} | "
          f"{'Best PF':>8} {'T':>4} {'WR%':>6} {'Params'}")
    print(f"  {'-'*80}")

    bl_avg = next((g for g in gate_avgs if g[0] == "baseline"), None)
    bl_pf = bl_avg[1] if bl_avg else 0

    for gate, avg_pf, avg_t, best, n in gate_avgs:
        delta = f"{(avg_pf / bl_pf - 1) * 100:>+6.0f}%" if bl_pf > 0 and gate != "baseline" else "  BASE"
        params_str = f"n={best['n_bricks']} cd={best['cooldown']}"
        print(f"  {gate:<20} {avg_pf:>8.2f} {avg_t:>6.1f} {n:>3} | "
              f"{best['oos_pf']:>8.2f} {best['oos_trades']:>4} "
              f"{best['oos_wr']:>5.1f}% {params_str} {delta}")

    # Top 10 individual configs
    all_viable = [r for r in all_results if r["oos_trades"] >= 10]
    all_viable.sort(key=lambda r: r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6,
                    reverse=True)

    print(f"\n  Top 10 individual configs (OOS trades >= 10):")
    print(f"  {'Gate':<20} {'n':>2} {'cd':>3} | {'IS PF':>7} {'T':>4} | "
          f"{'OOS PF':>8} {'T':>4} {'WR%':>6} {'Decay':>7}")
    print(f"  {'-'*70}")
    for r in all_viable[:10]:
        dec_s = f"{r['decay_pct']:>+6.1f}%" if not math.isnan(r["decay_pct"]) else "   NaN"
        print(f"  {r['gate']:<20} {r['n_bricks']:>2} {r['cooldown']:>3} | "
              f"{r['is_pf']:>7.2f} {r['is_trades']:>4} | "
              f"{r['oos_pf']:>8.2f} {r['oos_trades']:>4} "
              f"{r['oos_wr']:>5.1f}% {dec_s}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-parallel", action="store_true")
    args = parser.parse_args()

    out_path = ROOT / "ai_context" / "btc_phase1_results.json"
    out_path.parent.mkdir(exist_ok=True)

    n_params = len(list(itertools.product(*PARAM_GRID.values())))
    total = len(GATES) * n_params

    print("BTC Phase 1: P6 Gate Discovery (Long Only)")
    print(f"  Instrument   : BTCUSD $150 Renko")
    print(f"  IS period    : {IS_START} -> {IS_END}")
    print(f"  OOS period   : {OOS_START} -> {OOS_END}")
    print(f"  Commission   : {COMMISSION}%")
    print(f"  Qty per trade: {QTY_VALUE} BTC")
    print(f"  Gates        : {len(GATES)}")
    print(f"  Param combos : {n_params}")
    print(f"  Total runs   : {total} ({total * 2} IS+OOS backtests)")
    print(f"  Output       : {out_path}")
    print()

    all_results = []

    if args.no_parallel:
        for gate in GATES:
            all_results.extend(_sweep_gate(gate))
    else:
        with ProcessPoolExecutor() as pool:
            futures = {pool.submit(_sweep_gate, gate): gate for gate in GATES}
            for future in as_completed(futures):
                gate = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as exc:
                    import traceback
                    print(f"  [{gate}] FAILED: {exc}")
                    traceback.print_exc()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved {len(all_results)} results -> {out_path}")

    _summarize(all_results)


if __name__ == "__main__":
    main()
