#!/usr/bin/env python3
"""
phase6_combo_sweep.py — Pairwise Gate Combination Sweep

Takes the top 5 single gates per instrument from Phase 6 and tests all
pairwise AND combinations (C(5,2) = 10 combos per instrument).

Pure Renko only. Same base R007 logic and param grid as phase6_sweep.py.

Per-instrument top 5 gates (from phase6_results.json avg OOS PF):
  EURUSD: ema_cross, ichi_cloud, stoch_cross, mfi_dir, sq_mom
  GBPJPY: mk_regime, psar_dir, escgo_cross, macd_hist_dir, kama_slope
  EURAUD: ddl_dir, ichi_cloud, escgo_cross, motn_dx, cci_dir

10 combos x 12 params x 3 instruments = 360 runs

Usage:
  python renko/phase6_combo_sweep.py
  python renko/phase6_combo_sweep.py --no-parallel
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

# ── Instrument configs (same as phase6_sweep.py) ────────────────────────────────

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

# ── Top 5 gates per instrument (from Phase 6 avg OOS PF rankings) ────────────

TOP_GATES = {
    "EURUSD": ["ema_cross", "ichi_cloud", "stoch_cross", "mfi_dir", "sq_mom"],
    "GBPJPY": ["mk_regime", "psar_dir", "escgo_cross", "macd_hist_dir", "kama_slope"],
    "EURAUD": ["ddl_dir", "ichi_cloud", "escgo_cross", "motn_dx", "cci_dir"],
}

# Phase 6 single-gate baselines for comparison
SINGLE_GATE_AVGS = {
    "EURUSD": {"baseline": 5.25, "ema_cross": 6.45, "ichi_cloud": 6.38,
               "stoch_cross": 5.87, "mfi_dir": 5.76, "sq_mom": 5.69},
    "GBPJPY": {"baseline": 9.74, "mk_regime": 12.53, "psar_dir": 12.39,
               "escgo_cross": 12.22, "macd_hist_dir": 11.87, "kama_slope": 11.84},
    "EURAUD": {"baseline": 4.47, "ddl_dir": 5.75, "ichi_cloud": 5.55,
               "escgo_cross": 5.51, "motn_dx": 5.50, "cci_dir": 5.50},
}

PARAM_GRID = {
    "n_bricks": [2, 3, 4, 5],
    "cooldown": [10, 20, 30],
}


# ── Reuse gate array computation from phase6_sweep ──────────────────────────────

def _compute_gate_arrays(df, gate_name):
    """Imported logic from phase6_sweep._compute_gate_arrays."""
    sys.path.insert(0, str(ROOT))
    from renko.phase6_sweep import _compute_gate_arrays as _cga
    return _cga(df, gate_name)


# ── Signal generator (same as phase6_sweep) ──────────────────────────────────────

def _generate_signals(df, n_bricks, cooldown, gate_long_ok, gate_short_ok):
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

        if in_position:
            is_opp        = (trade_dir == 1 and not up) or (trade_dir == -1 and up)
            long_exit[i]  = is_opp and trade_dir == 1
            short_exit[i] = is_opp and trade_dir == -1
            if is_opp:
                in_position = False
                trade_dir   = 0

        if in_position:
            continue

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

        if is_long and not gate_long_ok[i]:
            continue
        if not is_long and not gate_short_ok[i]:
            continue

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

def _load_renko_enriched(renko_file, include_mk):
    sys.path.insert(0, str(ROOT))
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    from renko.phase6_enrichment import add_phase6_indicators

    df = load_renko_export(renko_file)
    add_renko_indicators(df)
    add_phase6_indicators(df, include_mk=include_mk)
    return df


# ── Worker ───────────────────────────────────────────────────────────────────────

def run_instrument_sweep(name, config):
    print(f"[{name}] Loading Renko + all indicators...", flush=True)
    df = _load_renko_enriched(config["renko_file"], config["include_mk"])
    print(f"[{name}] Ready — {len(df)} bricks", flush=True)

    gates = TOP_GATES[name]
    combos = list(itertools.combinations(gates, 2))

    # Pre-compute all single gate arrays
    gate_arrays = {}
    for g in gates:
        gate_arrays[g] = _compute_gate_arrays(df, g)

    keys         = list(PARAM_GRID.keys())
    param_combos = [dict(zip(keys, v)) for v in itertools.product(*PARAM_GRID.values())]
    total        = len(combos) * len(param_combos)
    done         = 0
    results      = []

    for g1, g2 in combos:
        combo_name = f"{g1}+{g2}"
        # AND the two gate arrays
        l1, s1 = gate_arrays[g1]
        l2, s2 = gate_arrays[g2]
        combo_long_ok  = l1 & l2
        combo_short_ok = s1 & s2

        # Overlap analysis: what % of baseline entries pass each gate?
        overlap_long  = np.sum(l1 & l2) / max(np.sum(l1), 1) * 100
        overlap_short = np.sum(s1 & s2) / max(np.sum(s1), 1) * 100

        for pc in param_combos:
            df_sig = _generate_signals(
                df.copy(),
                n_bricks      = pc["n_bricks"],
                cooldown      = pc["cooldown"],
                gate_long_ok  = combo_long_ok,
                gate_short_ok = combo_short_ok,
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
                "instrument":    name,
                "gate":          combo_name,
                "gate_a":        g1,
                "gate_b":        g2,
                "n_bricks":      pc["n_bricks"],
                "cooldown":      pc["cooldown"],
                "is_pf":         is_pf,
                "is_trades":     is_r["trades"],
                "is_net":        is_r["net"],
                "is_wr":         is_r["wr"],
                "oos_pf":        oos_pf,
                "oos_trades":    oos_r["trades"],
                "oos_net":       oos_r["net"],
                "oos_wr":        oos_r["wr"],
                "decay_pct":     decay,
                "overlap_long":  overlap_long,
                "overlap_short": overlap_short,
            })

            done += 1
            if done % 12 == 0 or done == total:
                print(
                    f"[{name}] {done:>3}/{total} | {combo_name:<28} "
                    f"n={pc['n_bricks']} cd={pc['cooldown']:>2} | "
                    f"IS PF={is_pf:>6.2f} T={is_r['trades']:>4} | "
                    f"OOS PF={oos_pf:>6.2f} T={oos_r['trades']:>4}",
                    flush=True,
                )

    print(f"[{name}] Complete — {len(results)} results", flush=True)
    return results


# ── Summary ──────────────────────────────────────────────────────────────────────

def _summarize(all_results):
    for inst in ["EURUSD", "GBPJPY", "EURAUD"]:
        inst_res = [r for r in all_results if r["instrument"] == inst]
        if not inst_res:
            continue

        singles = SINGLE_GATE_AVGS[inst]
        baseline = singles["baseline"]

        print(f"\n{'='*85}")
        print(f"  {inst}  Phase 6 baseline: {baseline:.2f}")
        print(f"  Single-gate refs: {', '.join(f'{k}={v:.2f}' for k, v in singles.items() if k != 'baseline')}")
        print(f"{'='*85}")

        viable = [r for r in inst_res if r["oos_trades"] >= 20]
        viable.sort(key=lambda r: (r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6),
                    reverse=True)

        # Top combos
        print(f"\n  Top 12 combos (OOS trades >= 20):")
        print(f"  {'Combo':<28} {'n':>2} {'cd':>3} | {'IS PF':>7} {'T':>5} | "
              f"{'OOS PF':>7} {'T':>5} {'WR%':>6} | {'Olap%':>5}")
        print(f"  {'-'*80}")
        for r in viable[:12]:
            best_single = max(singles.get(r["gate_a"], 0), singles.get(r["gate_b"], 0))
            beat = " <<" if r["oos_pf"] > best_single else ""
            olap = r.get("overlap_long", 0)
            print(f"  {r['gate']:<28} {r['n_bricks']:>2} {r['cooldown']:>3} | "
                  f"{r['is_pf']:>7.2f} {r['is_trades']:>5} | "
                  f"{r['oos_pf']:>7.2f} {r['oos_trades']:>5} {r['oos_wr']:>6.1f}% | "
                  f"{olap:>4.0f}%{beat}")

        # Combo averages
        all_combos = list(itertools.combinations(TOP_GATES[inst], 2))
        print(f"\n  Combo averages (OOS trades >= 20):")
        print(f"  {'Combo':<28} {'Avg PF':>8} {'Avg T':>7} {'N':>4}  "
              f"{'best_single':>12} {'delta':>7} {'Olap%':>6}")
        combo_avgs = {}
        for g1, g2 in all_combos:
            cname = f"{g1}+{g2}"
            cv = [r for r in viable if r["gate"] == cname]
            if cv:
                avg_pf = sum(r["oos_pf"] for r in cv) / len(cv)
                avg_t  = sum(r["oos_trades"] for r in cv) / len(cv)
                olap   = cv[0].get("overlap_long", 0)
                best_s = max(singles.get(g1, 0), singles.get(g2, 0))
                combo_avgs[cname] = (avg_pf, avg_t, len(cv), best_s, olap)

        for cname, (avg_pf, avg_t, n_v, best_s, olap) in sorted(
            combo_avgs.items(), key=lambda x: x[1][0], reverse=True
        ):
            delta = avg_pf - best_s
            marker = " *" if delta > 0 else ""
            print(f"  {cname:<28} {avg_pf:>8.2f} {avg_t:>7.1f} {n_v:>4}  "
                  f"{best_s:>12.2f} {delta:>+7.2f} {olap:>5.0f}%{marker}")


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-parallel", action="store_true")
    args = parser.parse_args()

    out_path = ROOT / "ai_context" / "phase6_combo_results.json"
    out_path.parent.mkdir(exist_ok=True)

    n_combos = len(list(itertools.product(*PARAM_GRID.values())))
    print("Phase 6 Combo Sweep: Pairwise Gate Combinations")
    for name in INSTRUMENTS:
        gates = TOP_GATES[name]
        pairs = list(itertools.combinations(gates, 2))
        print(f"  {name}: {len(pairs)} combos x {n_combos} params = {len(pairs) * n_combos} runs")
        for g1, g2 in pairs:
            print(f"    {g1} + {g2}")
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
