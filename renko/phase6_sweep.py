#!/usr/bin/env python3
"""
phase6_sweep.py — Universal Untapped Indicator Gate Sweep

Sweeps ALL untapped indicators (Tier 1 built-in + Tier 2 standalone + Tier 3
complex) as gates on R007 base logic across 3 Forex instruments in parallel.

Pure Renko only — no candle data, no HTF merge, no hardcoded ADX/vol/session
gates. Each gate is tested clean against a bare baseline.

Instruments:
  EURUSD  0.0005 brick (NEW — first test on this brick size)
  GBPJPY  0.05   brick
  EURAUD  0.0006 brick

Gates (19 for EU/EA, 20 for GJ):
  Tier 1 (built-in): rsi_dir, bb_pct_b, chop_trend, psar_dir, kama_slope,
                      sq_mom, stoch_cross, cmf_dir, mfi_dir, obv_trend,
                      ema_cross, macd_hist_dir
  Tier 2 (standalone): cci_dir, ichi_cloud, wpr_dir, donch_mid
  Tier 3 (complex): escgo_cross, ddl_dir, motn_dx
  GBPJPY-only: mk_regime

Param grid: n_bricks={2,3,4,5} x cooldown={10,20,30} = 12 combos
Total: ~(20+20+21) gates x 12 params x 2 periods = ~1,464 backtests

Usage:
  python renko/phase6_sweep.py
  python renko/phase6_sweep.py --no-parallel
"""

import argparse
import contextlib
import io
import itertools
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from renko.config import MAX_WORKERS
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# ── Instrument configs ──────────────────────────────────────────────────────────

INSTRUMENTS = {
    "EURUSD": {
        "renko_file":  "OANDA_EURUSD, 1S renko 0.0005.csv",
        "is_start":    "2022-05-18",
        "is_end":      "2025-09-30",
        "oos_start":   "2025-10-01",
        "oos_end":     "2026-03-05",
        "commission":  0.0046,
        "capital":     1000.0,
        "include_mk":  False,
    },
    "GBPJPY": {
        "renko_file":  "OANDA_GBPJPY, 1S renko 0.05.csv",
        "is_start":    "2024-11-21",
        "is_end":      "2025-09-30",
        "oos_start":   "2025-10-01",
        "oos_end":     "2026-02-28",
        "commission":  0.005,
        "capital":     150_000.0,
        "include_mk":  True,
    },
    "EURAUD": {
        "renko_file":  "OANDA_EURAUD, 1S renko 0.0006.csv",
        "is_start":    "2023-07-20",
        "is_end":      "2025-09-30",
        "oos_start":   "2025-10-01",
        "oos_end":     "2026-03-17",
        "commission":  0.009,
        "capital":     1000.0,
        "include_mk":  False,
    },
}

# ── Gate definitions ─────────────────────────────────────────────────────────────

# All gates tested across all instruments
BASE_GATES = [
    "baseline",
    # Tier 1: Built-in (already in renko/indicators.py, pre-shifted)
    "rsi_dir", "bb_pct_b", "chop_trend", "psar_dir", "kama_slope",
    "sq_mom", "stoch_cross", "cmf_dir", "mfi_dir", "obv_trend",
    "ema_cross", "macd_hist_dir",
    # Tier 2: Standalone (from phase6_enrichment.py)
    "cci_dir", "ichi_cloud", "wpr_dir", "donch_mid",
    # Tier 3: Complex (from phase6_enrichment.py)
    "escgo_cross", "ddl_dir", "motn_dx",
]

# GBPJPY-only gate (MK is FLAT on small brick instruments)
GBPJPY_EXTRA_GATES = ["mk_regime"]

PARAM_GRID = {
    "n_bricks": [2, 3, 4, 5],
    "cooldown": [10, 20, 30],
}


# ── Gate array pre-computation ───────────────────────────────────────────────────

def _compute_gate_arrays(df: pd.DataFrame, gate_name: str):
    """
    Pre-compute boolean arrays (gate_long_ok, gate_short_ok) for each bar.

    NaN-pass convention: True when indicator is NaN (waive gate).
    Returns two numpy bool arrays of length len(df).
    """
    n = len(df)

    if gate_name == "baseline":
        return np.ones(n, dtype=bool), np.ones(n, dtype=bool)

    close = df["Close"].values

    # ── Tier 1: Built-in indicators ──────────────────────────────────────────

    if gate_name == "rsi_dir":
        v = df["rsi"].values
        m = np.isnan(v)
        return m | (v > 50), m | (v < 50)

    if gate_name == "bb_pct_b":
        v = df["bb_pct_b"].values
        m = np.isnan(v)
        return m | (v > 0.5), m | (v < 0.5)

    if gate_name == "chop_trend":
        v = df["chop"].values
        m = np.isnan(v)
        ok = m | (v < 38.2)
        return ok, ok  # symmetric — trending regime allows both directions

    if gate_name == "psar_dir":
        v = df["psar_dir"].values
        m = np.isnan(v)
        return m | (v > 0), m | (v < 0)

    if gate_name == "kama_slope":
        v = df["kama_slope"].values
        m = np.isnan(v)
        return m | (v > 0), m | (v < 0)

    if gate_name == "sq_mom":
        v = df["sq_momentum"].values
        m = np.isnan(v)
        return m | (v > 0), m | (v < 0)

    if gate_name == "stoch_cross":
        k = df["stoch_k"].values
        d = df["stoch_d"].values
        m = np.isnan(k) | np.isnan(d)
        return m | (k > d), m | (k < d)

    if gate_name == "cmf_dir":
        v = df["cmf"].values
        m = np.isnan(v)
        return m | (v > 0), m | (v < 0)

    if gate_name == "mfi_dir":
        v = df["mfi"].values
        m = np.isnan(v)
        return m | (v > 50), m | (v < 50)

    if gate_name == "obv_trend":
        o = df["obv"].values
        e = df["obv_ema"].values
        m = np.isnan(o) | np.isnan(e)
        return m | (o > e), m | (o < e)

    if gate_name == "ema_cross":
        e9 = df["ema9"].values
        e21 = df["ema21"].values
        m = np.isnan(e9) | np.isnan(e21)
        return m | (e9 > e21), m | (e9 < e21)

    if gate_name == "macd_hist_dir":
        v = df["macd_hist"].values
        m = np.isnan(v)
        return m | (v > 0), m | (v < 0)

    # ── Tier 2: Standalone indicators ────────────────────────────────────────

    if gate_name == "cci_dir":
        v = df["cci"].values
        m = np.isnan(v)
        return m | (v > 0), m | (v < 0)

    if gate_name == "ichi_cloud":
        v = df["ichi_pos"].values.astype(float)
        m = np.isnan(v)
        # +1 above cloud → long ok, -1 below → short ok, 0 inside → block both
        return m | (v > 0), m | (v < 0)

    if gate_name == "wpr_dir":
        v = df["wpr"].values
        m = np.isnan(v)
        return m | (v > -50), m | (v < -50)

    if gate_name == "donch_mid":
        mid = df["donch_mid"].values  # pre-shifted
        m = np.isnan(mid)
        # Compare current close to shifted midpoint (no future leak)
        return m | (close > mid), m | (close < mid)

    # ── Tier 3: Complex indicators ───────────────────────────────────────────

    if gate_name == "escgo_cross":
        f = df["escgo_fast"].values
        s = df["escgo_slow"].values
        m = np.isnan(f) | np.isnan(s)
        return m | (f > s), m | (f < s)

    if gate_name == "ddl_dir":
        v = df["ddl_diff"].values
        m = np.isnan(v)
        return m | (v > 0), m | (v < 0)

    if gate_name == "motn_dx":
        dx = df["motn_dx"].values
        zx = df["motn_zx"].values
        m = np.isnan(dx) | np.isnan(zx)
        return m | (dx > zx), m | (dx < zx)

    if gate_name == "mk_regime":
        if "mk_regime" not in df.columns:
            return np.ones(n, dtype=bool), np.ones(n, dtype=bool)
        v = df["mk_regime"].values.astype(float)
        m = np.isnan(v)
        return m | (v > 0), m | (v < 0)

    raise ValueError(f"Unknown gate: {gate_name}")


# ── Signal generator ─────────────────────────────────────────────────────────────

def _generate_signals(
    df:             pd.DataFrame,
    n_bricks:       int,
    cooldown:       int,
    gate_long_ok:   np.ndarray,
    gate_short_ok:  np.ndarray,
) -> pd.DataFrame:
    """
    R007 logic (R001 + R002 combined) with pre-computed gate arrays.

    No hardcoded ADX/vol/session gates — pure indicator gate test.
    """
    n        = len(df)
    brick_up = df["brick_up"].values

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

        # Exit: first opposing brick
        if in_position:
            is_opp        = (trade_dir == 1 and not up) or (trade_dir == -1 and up)
            long_exit[i]  = is_opp and trade_dir == 1
            short_exit[i] = is_opp and trade_dir == -1
            if is_opp:
                in_position = False
                trade_dir   = 0

        if in_position:
            continue

        # R007 logic: determine candidate direction
        prev          = brick_up[i - n_bricks : i]
        prev_all_up   = bool(np.all(prev))
        prev_all_down = bool(not np.any(prev))

        if prev_all_up and not up:
            cand = -1; is_r002 = True
        elif prev_all_down and up:
            cand = 1;  is_r002 = True
        else:
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

        # Gate check (pre-computed arrays)
        if is_long and not gate_long_ok[i]:
            continue
        if not is_long and not gate_short_ok[i]:
            continue

        # Enter
        if is_long:
            long_entry[i] = True
        else:
            short_entry[i] = True
        in_position = True
        trade_dir   = cand
        if not is_r002:
            last_r001_bar = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df


# ── Backtest runner ──────────────────────────────────────────────────────────────

def _run_backtest(df_sig, start, end, commission, capital):
    sys.path.insert(0, str(ROOT))
    from engine import BacktestConfig, run_backtest_long_short

    cfg = BacktestConfig(
        initial_capital=capital,
        commission_pct=commission,
        slippage_ticks=0,
        qty_type="fixed",
        qty_value=1000.0,
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


# ── Data loading ─────────────────────────────────────────────────────────────────

def _load_renko_enriched(renko_file: str, include_mk: bool) -> pd.DataFrame:
    """Load Renko data, add standard + Phase 6 indicators."""
    sys.path.insert(0, str(ROOT))
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    from renko.phase6_enrichment import add_phase6_indicators

    df = load_renko_export(renko_file)
    add_renko_indicators(df)
    add_phase6_indicators(df, include_mk=include_mk)
    return df


# ── Worker: one instrument per process ───────────────────────────────────────────

def run_instrument_sweep(name: str, config: dict) -> list:
    print(f"[{name}] Loading Renko + all indicators...", flush=True)
    df = _load_renko_enriched(config["renko_file"], config["include_mk"])
    print(f"[{name}] Ready — {len(df)} bricks", flush=True)

    # Determine gates for this instrument
    gates = list(BASE_GATES)
    if config["include_mk"]:
        gates.extend(GBPJPY_EXTRA_GATES)

    keys         = list(PARAM_GRID.keys())
    param_combos = [dict(zip(keys, v)) for v in itertools.product(*PARAM_GRID.values())]
    total        = len(gates) * len(param_combos)
    done         = 0
    results      = []

    for gate_name in gates:
        # Pre-compute gate arrays once per gate
        gate_long_ok, gate_short_ok = _compute_gate_arrays(df, gate_name)

        for pc in param_combos:
            df_sig = _generate_signals(
                df.copy(),
                n_bricks      = pc["n_bricks"],
                cooldown      = pc["cooldown"],
                gate_long_ok  = gate_long_ok,
                gate_short_ok = gate_short_ok,
            )

            is_r  = _run_backtest(df_sig, config["is_start"],  config["is_end"],
                                  config["commission"], config["capital"])
            oos_r = _run_backtest(df_sig, config["oos_start"], config["oos_end"],
                                  config["commission"], config["capital"])

            is_pf  = is_r["pf"]
            oos_pf = oos_r["pf"]
            decay  = ((oos_pf - is_pf) / is_pf * 100) \
                     if is_pf > 0 and not math.isinf(is_pf) else float("nan")

            results.append({
                "instrument": name,
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

            done += 1
            if done % 12 == 0 or done == total:
                print(
                    f"[{name}] {done:>3}/{total} | {gate_name:<16} "
                    f"n={pc['n_bricks']} cd={pc['cooldown']:>2} | "
                    f"IS PF={is_pf:>6.2f} T={is_r['trades']:>4} | "
                    f"OOS PF={oos_pf:>6.2f} T={oos_r['trades']:>4} "
                    f"decay={decay:>+6.1f}%",
                    flush=True,
                )

    print(f"[{name}] Complete — {len(results)} results", flush=True)
    return results


# ── Summary ──────────────────────────────────────────────────────────────────────

def _summarize(all_results: list) -> None:
    for inst in ["EURUSD", "GBPJPY", "EURAUD"]:
        inst_res = [r for r in all_results if r["instrument"] == inst]
        if not inst_res:
            continue

        # Get baseline avg for this instrument
        baseline_viable = [r for r in inst_res
                          if r["gate"] == "baseline" and r["oos_trades"] >= 20]
        baseline_avg = (sum(r["oos_pf"] for r in baseline_viable) / len(baseline_viable)
                       if baseline_viable else 0.0)

        print(f"\n{'='*80}")
        print(f"  {inst}  Baseline avg OOS PF: {baseline_avg:.2f}")
        print(f"{'='*80}")

        viable = [r for r in inst_res if r["oos_trades"] >= 20]
        viable.sort(key=lambda r: (r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6),
                    reverse=True)

        # Top 15 individual combos
        print(f"\n  Top 15 combos (OOS trades >= 20):")
        print(f"  {'Gate':<16} {'n':>2} {'cd':>3} | {'IS PF':>7} {'T':>5} | "
              f"{'OOS PF':>7} {'T':>5} {'WR%':>6} {'Decay':>7}")
        print(f"  {'-'*74}")
        for r in viable[:15]:
            beat  = " <<" if r["oos_pf"] > baseline_avg * 1.2 else ""
            dec_s = f"{r['decay_pct']:>+6.1f}%" if not math.isnan(r["decay_pct"]) else "   NaN"
            print(f"  {r['gate']:<16} {r['n_bricks']:>2} {r['cooldown']:>3} | "
                  f"{r['is_pf']:>7.2f} {r['is_trades']:>5} | "
                  f"{r['oos_pf']:>7.2f} {r['oos_trades']:>5} {r['oos_wr']:>6.1f}% "
                  f"{dec_s}{beat}")

        # Gate averages
        all_gates = BASE_GATES[:]
        if inst == "GBPJPY":
            all_gates.extend(GBPJPY_EXTRA_GATES)

        print(f"\n  Gate averages (OOS trades >= 20):")
        print(f"  {'Gate':<16} {'Avg OOS PF':>12} {'Avg T':>7} {'Avg Decay':>10} {'N':>4}  vs baseline")
        gate_avgs = {}
        for gate in all_gates:
            gv = [r for r in viable if r["gate"] == gate]
            if gv:
                avg_pf    = sum(r["oos_pf"] for r in gv) / len(gv)
                avg_t     = sum(r["oos_trades"] for r in gv) / len(gv)
                valid_dec = [r["decay_pct"] for r in gv if not math.isnan(r["decay_pct"])]
                avg_dec   = sum(valid_dec) / len(valid_dec) if valid_dec else float("nan")
                gate_avgs[gate] = (avg_pf, avg_t, avg_dec, len(gv))

        for gate, (avg_pf, avg_t, avg_dec, n_viable) in sorted(
            gate_avgs.items(), key=lambda x: x[1][0], reverse=True
        ):
            dec_s    = f"{avg_dec:>+9.1f}%" if not math.isnan(avg_dec) else "       NaN"
            vs_base  = f"{avg_pf - baseline_avg:>+7.2f}" if baseline_avg > 0 else "    N/A"
            marker   = " *" if avg_pf > baseline_avg else ""
            print(f"  {gate:<16} {avg_pf:>12.2f} {avg_t:>7.1f} {dec_s} {n_viable:>4}  {vs_base}{marker}")

    # Cross-instrument summary
    print(f"\n{'='*80}")
    print("  Cross-instrument summary (avg OOS PF per gate, viable combos)")
    print(f"{'='*80}")
    print(f"  {'Gate':<16} {'EURUSD':>12} {'GBPJPY':>12} {'EURAUD':>12} {'Wins':>6}")
    print(f"  {'-'*58}")

    all_gate_names = list(dict.fromkeys(BASE_GATES + GBPJPY_EXTRA_GATES))
    gate_cross = {}
    for gate in all_gate_names:
        row_vals = {}
        for inst in ["EURUSD", "GBPJPY", "EURAUD"]:
            gv = [r for r in all_results
                  if r["instrument"] == inst and r["gate"] == gate and r["oos_trades"] >= 20]
            if gv:
                row_vals[inst] = sum(r["oos_pf"] for r in gv) / len(gv)
        gate_cross[gate] = row_vals

    # Get baselines for comparison
    baselines = {}
    for inst in ["EURUSD", "GBPJPY", "EURAUD"]:
        bv = gate_cross.get("baseline", {}).get(inst, 0)
        baselines[inst] = bv

    for gate in all_gate_names:
        vals = gate_cross.get(gate, {})
        wins = 0
        row  = [f"  {gate:<16}"]
        for inst in ["EURUSD", "GBPJPY", "EURAUD"]:
            if inst in vals:
                avg_pf = vals[inst]
                marker = "+" if avg_pf > baselines.get(inst, 0) else " "
                row.append(f"{avg_pf:>11.2f}{marker}")
                if avg_pf > baselines.get(inst, 0):
                    wins += 1
            else:
                row.append(f"{'  N/A':>12}")
        row.append(f"{wins:>6}")
        print("".join(row))


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-parallel", action="store_true",
                        help="Run instruments sequentially (debug mode)")
    args = parser.parse_args()

    out_path = ROOT / "ai_context" / "phase6_results.json"
    out_path.parent.mkdir(exist_ok=True)

    n_combos = len(list(itertools.product(*PARAM_GRID.values())))
    total_gates = {
        name: len(BASE_GATES) + (len(GBPJPY_EXTRA_GATES) if cfg["include_mk"] else 0)
        for name, cfg in INSTRUMENTS.items()
    }
    total_runs = sum(g * n_combos * 2 for g in total_gates.values())

    print("Phase 6: Universal Untapped Indicator Gate Sweep")
    print(f"  Mode         : Pure Renko (no candle data)")
    print(f"  Base gates   : {len(BASE_GATES)} (incl. baseline)")
    print(f"  Extra (GJ)   : {len(GBPJPY_EXTRA_GATES)}")
    print(f"  Param combos : {n_combos}")
    print(f"  Instruments  : {list(INSTRUMENTS.keys())}")
    for name, n_gates in total_gates.items():
        print(f"    {name}: {n_gates} gates x {n_combos} combos = {n_gates * n_combos} runs")
    print(f"  Total IS+OOS : {total_runs}")
    print(f"  Output       : {out_path}")
    print()

    all_results: list = []

    if args.no_parallel:
        for name, config in INSTRUMENTS.items():
            all_results.extend(run_instrument_sweep(name, config))
    else:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
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
                    import traceback
                    print(f"  [{name}] FAILED: {exc}")
                    traceback.print_exc()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved {len(all_results)} results -> {out_path}")

    _summarize(all_results)


if __name__ == "__main__":
    main()
