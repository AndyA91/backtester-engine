#!/usr/bin/env python3
"""
btc_phase3_htf_sweep.py — BTC Phase 3: HTF Gate Sweep (Long Only)

Uses $300 HTF Renko as regime filter on top of Phase 2 winners.
LTF gates fixed to Phase 2 top 3 configs (P6 + ADX, no oscillator).

Dimensions:
  LTF configs: escgo_cross_a30, psar_dir_a30, kama_slope_a30 (fixed)
  HTF gates:   8 gate types + baseline
  HTF ADX thresh: {20, 25, 30, 35, 40}  (for htf_adx gate)
  n_bricks:    {2, 3, 4, 5}
  cooldown:    {10, 20, 30}

Uses ProcessPoolExecutor — one worker per LTF config.

Usage:
  python renko/btc_phase3_htf_sweep.py
  python renko/btc_phase3_htf_sweep.py --no-parallel
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

# ── Phase 2 winners (fixed LTF configs) ───────────────────────────────────────

LTF_CONFIGS = {
    "escgo_cross_a30": {"p6_gate": "escgo_cross", "adx_thresh": 30},
    "psar_dir_a30":    {"p6_gate": "psar_dir",    "adx_thresh": 30},
    "kama_slope_a30":  {"p6_gate": "kama_slope",  "adx_thresh": 30},
}

# ── HTF gate definitions ──────────────────────────────────────────────────────

HTF_GATE_NAMES = [
    "htf_brick_dir",
    "htf_n2_dir",
    "htf_n3_dir",
    "htf_ema_cross",
    "htf_psar_dir",
    "htf_macd_hist",
    "htf_stoch_cross",
]

HTF_ADX_THRESHOLDS = [20, 25, 30, 35, 40]

PARAM_GRID = {
    "n_bricks": [2, 3, 4, 5],
    "cooldown": [10, 20, 30],
}


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_ltf_data() -> pd.DataFrame:
    sys.path.insert(0, str(ROOT))
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    from renko.phase6_enrichment import add_phase6_indicators

    df = load_renko_export(LTF_FILE)
    add_renko_indicators(df)
    add_phase6_indicators(df, include_mk=False)
    return df


def _load_htf_data() -> pd.DataFrame:
    sys.path.insert(0, str(ROOT))
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators

    df = load_renko_export(HTF_FILE)
    add_renko_indicators(df)
    return df


# ── LTF gate computation (fixed Phase 2 winner) ──────────────────────────────

def _compute_ltf_gate(df: pd.DataFrame, config: dict) -> np.ndarray:
    """Compute AND-combined long gate: P6 + ADX + vol_ratio."""
    sys.path.insert(0, str(ROOT))
    from renko.phase6_sweep import _compute_gate_arrays

    n = len(df)
    gate = np.ones(n, dtype=bool)

    # P6 gate
    p6_long, _ = _compute_gate_arrays(df, config["p6_gate"])
    gate &= p6_long

    # ADX
    adx = df["adx"].values
    adx_nan = np.isnan(adx)
    gate &= (adx_nan | (adx >= config["adx_thresh"]))

    # Vol ratio
    vr = df["vol_ratio"].values
    vr_nan = np.isnan(vr)
    gate &= (vr_nan | (vr <= VOL_MAX))

    return gate


# ── HTF gate computation ──────────────────────────────────────────────────────

def _compute_htf_gates(df_htf: pd.DataFrame) -> dict:
    """Compute all HTF gate arrays. Returns dict[name -> long_ok array]."""
    n = len(df_htf)
    brick_up = df_htf["brick_up"].values
    gates = {}

    # htf_brick_dir: previous brick direction
    g = np.ones(n, dtype=bool)
    g[1:] = brick_up[:-1]
    gates["htf_brick_dir"] = g.copy()

    # htf_n2_dir: last 2 bricks all up
    g = np.ones(n, dtype=bool)
    for i in range(2, n):
        g[i] = brick_up[i-1] and brick_up[i-2]
    gates["htf_n2_dir"] = g

    # htf_n3_dir: last 3 bricks all up
    g = np.ones(n, dtype=bool)
    for i in range(3, n):
        g[i] = brick_up[i-1] and brick_up[i-2] and brick_up[i-3]
    gates["htf_n3_dir"] = g

    # htf_ema_cross: EMA9 > EMA21
    ema9 = df_htf["ema9"].values
    ema21 = df_htf["ema21"].values
    m = np.isnan(ema9) | np.isnan(ema21)
    gates["htf_ema_cross"] = m | (ema9 > ema21)

    # htf_psar_dir: PSAR bullish
    psar = df_htf["psar_dir"].values
    psar_nan = np.isnan(psar)
    gates["htf_psar_dir"] = psar_nan | (psar > 0)

    # htf_macd_hist: MACD histogram >= 0
    mh = df_htf["macd_hist"].values
    mh_nan = np.isnan(mh)
    gates["htf_macd_hist"] = mh_nan | (mh >= 0)

    # htf_stoch_cross: %K > %D
    sk = df_htf["stoch_k"].values
    sd = df_htf["stoch_d"].values
    sm = np.isnan(sk) | np.isnan(sd)
    gates["htf_stoch_cross"] = sm | (sk > sd)

    # htf_adx variants
    adx = df_htf["adx"].values
    adx_nan = np.isnan(adx)
    for thresh in HTF_ADX_THRESHOLDS:
        gates[f"htf_adx{thresh}"] = adx_nan | (adx >= thresh)

    return gates


# ── HTF -> LTF alignment ──────────────────────────────────────────────────────

def _align_htf_to_ltf(df_ltf: pd.DataFrame, df_htf: pd.DataFrame,
                       htf_gate: np.ndarray) -> np.ndarray:
    """Backward-fill HTF gate onto LTF timestamps via merge_asof."""
    htf_frame = pd.DataFrame({
        "t": df_htf.index.values,
        "g": htf_gate.astype(float),
    }).sort_values("t")

    ltf_frame = pd.DataFrame({
        "t": df_ltf.index.values,
    }).sort_values("t")

    merged = pd.merge_asof(ltf_frame, htf_frame, on="t", direction="backward")
    g = merged["g"].values
    return np.where(np.isnan(g), True, g > 0.5).astype(bool)


# ── Signal generator (long only) ──────────────────────────────────────────────

def _generate_signals_long_only(
    df: pd.DataFrame,
    n_bricks: int,
    cooldown: int,
    gate_long_ok: np.ndarray,
) -> pd.DataFrame:
    """R007 logic -- long entries only."""
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

        # R002 long: n down then up
        prev = brick_up[i - n_bricks : i]
        prev_all_down = bool(not np.any(prev))

        if prev_all_down and up:
            if gate_long_ok[i]:
                long_entry[i] = True
                in_position = True
            continue

        # R001 long: n consecutive up
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


# ── Backtest runner ────────────────────────────────────────────────────────────

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


# ── Worker: sweep one LTF config across all HTF gates x params ────────────────

def _sweep_ltf_config(ltf_name: str, ltf_config: dict) -> list:
    """Run all HTF gate x param combos for one fixed LTF config."""
    print(f"  [{ltf_name}] Loading LTF data...", flush=True)
    df_ltf = _load_ltf_data()
    ltf_gate = _compute_ltf_gate(df_ltf, ltf_config)
    print(f"  [{ltf_name}] LTF ready -- {len(df_ltf)} bricks", flush=True)

    print(f"  [{ltf_name}] Loading HTF data...", flush=True)
    df_htf = _load_htf_data()
    print(f"  [{ltf_name}] HTF ready -- {len(df_htf)} bricks", flush=True)

    # Pre-compute and align all HTF gates
    htf_gates_raw = _compute_htf_gates(df_htf)
    htf_aligned = {}
    for gname, garr in htf_gates_raw.items():
        htf_aligned[gname] = _align_htf_to_ltf(df_ltf, df_htf, garr)

    # Build sweep configs: baseline + all HTF gates
    all_htf_names = ["baseline"] + HTF_GATE_NAMES + [f"htf_adx{t}" for t in HTF_ADX_THRESHOLDS]

    keys = list(PARAM_GRID.keys())
    param_combos = [dict(zip(keys, v)) for v in itertools.product(*PARAM_GRID.values())]

    total = len(all_htf_names) * len(param_combos)
    done = 0
    results = []

    for htf_name in all_htf_names:
        # Combine LTF + HTF gate
        if htf_name == "baseline":
            combined = ltf_gate.copy()
        else:
            combined = ltf_gate & htf_aligned[htf_name]

        for pc in param_combos:
            df_sig = _generate_signals_long_only(
                df_ltf.copy(), pc["n_bricks"], pc["cooldown"], combined,
            )

            is_r  = _run_backtest(df_sig, IS_START, IS_END)
            oos_r = _run_backtest(df_sig, OOS_START, OOS_END)

            is_pf  = is_r["pf"]
            oos_pf = oos_r["pf"]
            decay  = ((oos_pf - is_pf) / is_pf * 100) \
                     if is_pf > 0 and not math.isinf(is_pf) else float("nan")

            results.append({
                "ltf_config":  ltf_name,
                "htf_gate":    htf_name,
                "n_bricks":    pc["n_bricks"],
                "cooldown":    pc["cooldown"],
                "is_pf":       is_pf,
                "is_trades":   is_r["trades"],
                "is_net":      is_r["net"],
                "is_wr":       is_r["wr"],
                "oos_pf":      oos_pf,
                "oos_trades":  oos_r["trades"],
                "oos_net":     oos_r["net"],
                "oos_wr":      oos_r["wr"],
                "decay_pct":   decay,
            })

            done += 1
            if done % 48 == 0 or done == total:
                print(
                    f"  [{ltf_name}] {done:>3}/{total} | {htf_name:<20} "
                    f"n={pc['n_bricks']} cd={pc['cooldown']:>2} | "
                    f"IS PF={is_pf:>7.2f} T={is_r['trades']:>4} | "
                    f"OOS PF={oos_pf:>7.2f} T={oos_r['trades']:>4}",
                    flush=True,
                )

    best = max(results, key=lambda r: r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6)
    print(f"  [{ltf_name}] Complete -- {len(results)} results | "
          f"Best: {best['htf_gate']} n={best['n_bricks']} cd={best['cooldown']} "
          f"OOS PF={best['oos_pf']:.2f} T={best['oos_trades']}",
          flush=True)
    return results


# ── Summary ────────────────────────────────────────────────────────────────────

P2_BASELINES = {
    "escgo_cross_a30": 25.02,
    "psar_dir_a30":    23.97,
    "kama_slope_a30":  23.89,
}


def _summarize(all_results: list) -> None:
    print(f"\n{'='*90}")
    print("  BTC Phase 3 -- HTF Gate Sweep (Long Only)")
    print(f"{'='*90}")

    for ltf_name in LTF_CONFIGS:
        ltf_res = [r for r in all_results if r["ltf_config"] == ltf_name]
        viable = [r for r in ltf_res if r["oos_trades"] >= 10]
        if not viable:
            continue

        p2_best = P2_BASELINES.get(ltf_name, 0)

        print(f"\n  --- {ltf_name} (Phase 2 best: {p2_best:.2f}) ---")

        # By HTF gate (avg OOS PF)
        all_htf_names = ["baseline"] + HTF_GATE_NAMES + [f"htf_adx{t}" for t in HTF_ADX_THRESHOLDS]
        print(f"\n  By HTF gate (avg OOS PF, trades >= 10):")
        gate_avgs = []
        for gname in all_htf_names:
            gv = [r for r in viable if r["htf_gate"] == gname]
            if gv:
                avg = sum(r["oos_pf"] for r in gv if not math.isinf(r["oos_pf"])) / max(
                    len([r for r in gv if not math.isinf(r["oos_pf"])]), 1)
                avg_t = sum(r["oos_trades"] for r in gv) / len(gv)
                gate_avgs.append((gname, avg, avg_t, len(gv)))

        gate_avgs.sort(key=lambda x: x[1], reverse=True)
        bl_avg = next((g[1] for g in gate_avgs if g[0] == "baseline"), 0)

        for gname, avg, avg_t, n in gate_avgs:
            delta = f"{(avg / bl_avg - 1) * 100:>+6.0f}%" if bl_avg > 0 and gname != "baseline" else "  BASE"
            print(f"    {gname:<20} avg PF={avg:>8.2f}  avg T={avg_t:>6.1f}  N={n:>3} {delta}")

        # Top 5
        viable.sort(key=lambda r: r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6,
                     reverse=True)
        print(f"\n  Top 5:")
        print(f"  {'HTF Gate':<20} {'n':>2} {'cd':>3} | {'IS PF':>7} {'T':>4} | "
              f"{'OOS PF':>8} {'T':>4} {'WR%':>6} {'Decay':>7}")
        print(f"  {'-'*75}")
        for r in viable[:5]:
            dec_s = f"{r['decay_pct']:>+6.1f}%" if not math.isnan(r["decay_pct"]) else "   NaN"
            beat = " <<BEAT" if r["oos_pf"] > p2_best else ""
            print(f"  {r['htf_gate']:<20} {r['n_bricks']:>2} {r['cooldown']:>3} | "
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
    print(f"  {'LTF Config':<20} {'HTF Gate':<20} {'n':>2} {'cd':>3} | "
          f"{'IS PF':>7} {'T':>4} | {'OOS PF':>8} {'T':>4} {'WR%':>6} {'Decay':>7}")
    print(f"  {'-'*90}")
    for r in all_viable[:10]:
        dec_s = f"{r['decay_pct']:>+6.1f}%" if not math.isnan(r["decay_pct"]) else "   NaN"
        print(f"  {r['ltf_config']:<20} {r['htf_gate']:<20} "
              f"{r['n_bricks']:>2} {r['cooldown']:>3} | "
              f"{r['is_pf']:>7.2f} {r['is_trades']:>4} | "
              f"{r['oos_pf']:>8.2f} {r['oos_trades']:>4} "
              f"{r['oos_wr']:>5.1f}% {dec_s}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-parallel", action="store_true")
    args = parser.parse_args()

    out_path = ROOT / "ai_context" / "btc_phase3_results.json"
    out_path.parent.mkdir(exist_ok=True)

    n_params = len(list(itertools.product(*PARAM_GRID.values())))
    n_htf = 1 + len(HTF_GATE_NAMES) + len(HTF_ADX_THRESHOLDS)
    per_ltf = n_htf * n_params
    total = len(LTF_CONFIGS) * per_ltf

    print("BTC Phase 3: HTF Gate Sweep (Long Only)")
    print(f"  LTF brick    : $150")
    print(f"  HTF brick    : $300")
    print(f"  IS period    : {IS_START} -> {IS_END}")
    print(f"  OOS period   : {OOS_START} -> {OOS_END}")
    print(f"  LTF configs  : {list(LTF_CONFIGS.keys())}")
    print(f"  HTF gates    : {n_htf} (1 baseline + {len(HTF_GATE_NAMES)} direction + {len(HTF_ADX_THRESHOLDS)} ADX thresholds)")
    print(f"  Param combos : {n_params}")
    print(f"  Per LTF      : {per_ltf} runs")
    print(f"  Total runs   : {total} ({total * 2} IS+OOS backtests)")
    print(f"  Output       : {out_path}")
    print()

    all_results = []

    if args.no_parallel:
        for name, config in LTF_CONFIGS.items():
            all_results.extend(_sweep_ltf_config(name, config))
    else:
        with ProcessPoolExecutor(max_workers=len(LTF_CONFIGS)) as pool:
            futures = {
                pool.submit(_sweep_ltf_config, name, config): name
                for name, config in LTF_CONFIGS.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    print(f"  [{name}] finished -- {len(results)} records")
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
