#!/usr/bin/env python3
"""
R024 KAMA Ribbon Pullback — multi-pair entry-mode comparison.

Runs fresh_only / resume_1 / resume across all 6 pairs (primary brick size).
One process per pair via ProcessPoolExecutor.

Usage:
  python renko/sweep_r024_all_pairs.py
"""

import contextlib
import io
import itertools
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent / "strategies"))

# ── Instrument configs ──────────────────────────────────────────────────────
# Primary brick size per pair, IS date range, commission, capital, qty

INSTRUMENTS = {
    "EURUSD": {
        "renko_file":  "OANDA_EURUSD, 1S renko 0.0004.csv",
        "is_start":    "2023-01-23",
        "is_end":      "2025-09-30",
        "commission":  0.0046,
        "capital":     1000.0,
        "qty_value":   1000.0,
    },
    "GBPJPY": {
        "renko_file":  "OANDA_GBPJPY, 1S renko 0.05.csv",
        "is_start":    "2024-11-22",
        "is_end":      "2025-09-30",
        "commission":  0.005,
        "capital":     150_000.0,
        "qty_value":   1000.0,
    },
    "EURAUD": {
        "renko_file":  "OANDA_EURAUD, 1S renko 0.0006.csv",
        "is_start":    "2023-07-20",
        "is_end":      "2025-09-30",
        "commission":  0.009,
        "capital":     1000.0,
        "qty_value":   1000.0,
    },
    "GBPUSD": {
        "renko_file":  "OANDA_GBPUSD, 1S renko 0.0004.csv",
        "is_start":    "2023-11-15",
        "is_end":      "2025-09-30",
        "commission":  0.0046,
        "capital":     1000.0,
        "qty_value":   1000.0,
    },
    "USDJPY": {
        "renko_file":  "OANDA_USDJPY, 1S renko 0.05.csv",
        "is_start":    "2024-07-18",
        "is_end":      "2025-09-30",
        "commission":  0.005,
        "capital":     150_000.0,
        "qty_value":   1000.0,
    },
    "BTCUSD": {
        "renko_file":  "OANDA_BTCUSD.SPOT.US, 1S renko 150.csv",
        "is_start":    "2024-06-04",
        "is_end":      "2025-09-30",
        "commission":  0.01,
        "capital":     10_000.0,
        "qty_value":   0.1,
    },
}

# ── Param grid (fixed ribbon/cooldown/adx/exit, vary entry_mode) ───────────
RIBBONS = ["3L_5_13_34", "3L_8_21_55"]
COOLDOWNS = [3, 5]
ADX_GATES = [0, 25]
EXIT_ON_BRICK = [True, False]
ENTRY_MODES = ["fresh_only", "resume_1", "resume"]


def run_instrument(name, cfg):
    """Run all param combos for one instrument. Called in subprocess."""
    import numpy as np
    import pandas as pd
    from engine import BacktestConfig, run_backtest_long_short
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    from renko.strategies.r024_kama_ribbon_pullback import generate_signals, _KAMA_CACHE

    df = load_renko_export(cfg["renko_file"])
    add_renko_indicators(df)

    bcfg = BacktestConfig(
        initial_capital=cfg["capital"],
        commission_pct=cfg["commission"],
        slippage_ticks=0,
        qty_type="fixed",
        qty_value=cfg["qty_value"],
        pyramiding=1,
        start_date=cfg["is_start"],
        end_date=cfg["is_end"],
        take_profit_pct=0.0,
        stop_loss_pct=0.0,
    )

    combos = list(itertools.product(RIBBONS, COOLDOWNS, ADX_GATES, ENTRY_MODES, EXIT_ON_BRICK))
    results = []

    for ribbon, cooldown, adx_gate, entry_mode, exit_on_brick in combos:
        _KAMA_CACHE.clear()
        params = dict(
            ribbon=ribbon, cooldown=cooldown, adx_gate=adx_gate,
            entry_mode=entry_mode, exit_on_brick=exit_on_brick,
        )
        df_sig = generate_signals(df.copy(), **params)
        with contextlib.redirect_stdout(io.StringIO()):
            kpis = run_backtest_long_short(df_sig, bcfg)

        pf = kpis.get("profit_factor", 0.0) or 0.0
        results.append({
            "pair":       name,
            "params":     params,
            "trades":     int(kpis.get("total_trades", 0) or 0),
            "win_rate":   float(kpis.get("win_rate", 0.0) or 0.0),
            "net":        float(kpis.get("net_profit", 0.0) or 0.0),
            "pf":         float("inf") if math.isinf(pf) else float(pf),
            "max_dd_pct": float(kpis.get("max_drawdown_pct", 0.0) or 0.0),
            "expectancy": float(kpis.get("avg_trade", 0.0) or 0.0),
        })

    return name, results


def main():
    print("R024 KAMA Ribbon Pullback — all pairs sweep")
    print(f"Pairs: {list(INSTRUMENTS.keys())}")
    combos_per = len(RIBBONS) * len(COOLDOWNS) * len(ADX_GATES) * len(ENTRY_MODES) * len(EXIT_ON_BRICK)
    print(f"Combos per pair: {combos_per}  |  Total: {combos_per * len(INSTRUMENTS)}")
    print()

    all_results = {}
    with ProcessPoolExecutor(max_workers=len(INSTRUMENTS)) as pool:
        futures = {
            pool.submit(run_instrument, name, cfg): name
            for name, cfg in INSTRUMENTS.items()
        }
        for fut in as_completed(futures):
            name = futures[fut]
            name, results = fut.result()
            all_results[name] = results
            print(f"  {name}: {len(results)} combos done")

    # ── Aggregate by entry_mode across all pairs ────────────────────────────
    print("\n" + "=" * 70)
    print("  AGGREGATE BY ENTRY MODE (across all pairs, all param combos)")
    print("=" * 70)

    for mode in ENTRY_MODES:
        mode_rows = []
        for pair, results in all_results.items():
            for r in results:
                if r["params"]["entry_mode"] == mode:
                    mode_rows.append(r)

        trades_all = [r["trades"] for r in mode_rows]
        nets_all = [r["net"] for r in mode_rows]
        pfs_all = [r["pf"] for r in mode_rows if not math.isinf(r["pf"]) and r["trades"] >= 30]
        wrs_all = [r["win_rate"] for r in mode_rows if r["trades"] >= 30]

        print(f"\n  {mode}:")
        print(f"    Combos         : {len(mode_rows)}")
        print(f"    Median trades  : {sorted(trades_all)[len(trades_all)//2]}")
        print(f"    Median net     : {sorted(nets_all)[len(nets_all)//2]:.2f}")
        if pfs_all:
            print(f"    Median PF      : {sorted(pfs_all)[len(pfs_all)//2]:.2f}")
        if wrs_all:
            print(f"    Median WR      : {sorted(wrs_all)[len(wrs_all)//2]:.1f}%")

    # ── Per-pair breakdown by entry mode ────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PER-PAIR BEST COMBO BY ENTRY MODE (min 30 trades, ranked by PF)")
    print("=" * 70)

    for pair in INSTRUMENTS:
        print(f"\n  {pair}:")
        for mode in ENTRY_MODES:
            candidates = [
                r for r in all_results[pair]
                if r["params"]["entry_mode"] == mode and r["trades"] >= 30
            ]
            if not candidates:
                print(f"    {mode:<12}  no qualifying combos (< 30 trades)")
                continue
            # rank by PF then net
            best = max(candidates, key=lambda r: (
                r["pf"] if not math.isinf(r["pf"]) else 1e12, r["net"]
            ))
            pf_s = f"{best['pf']:.2f}" if not math.isinf(best["pf"]) else "INF"
            p = best["params"]
            print(f"    {mode:<12}  PF={pf_s:>7}  Net={best['net']:>9.2f}  "
                  f"T={best['trades']:>4}  WR={best['win_rate']:>5.1f}%  "
                  f"DD={best['max_dd_pct']:>5.2f}%  "
                  f"| rb={p['ribbon']} cd={p['cooldown']} adx={p['adx_gate']} "
                  f"exit_brick={p['exit_on_brick']}")

    # ── Per-pair: same params, compare modes ────────────────────────────────
    print("\n" + "=" * 70)
    print("  APPLES-TO-APPLES: same params, vary entry_mode only")
    print("=" * 70)

    # For each pair, find the best fresh_only combo, then show resume_1 and resume with same params
    for pair in INSTRUMENTS:
        results = all_results[pair]
        fresh_candidates = [
            r for r in results
            if r["params"]["entry_mode"] == "fresh_only" and r["trades"] >= 30
        ]
        if not fresh_candidates:
            print(f"\n  {pair}: no qualifying fresh_only combos")
            continue

        best_fresh = max(fresh_candidates, key=lambda r: (
            r["pf"] if not math.isinf(r["pf"]) else 1e12, r["net"]
        ))
        base_params = {k: v for k, v in best_fresh["params"].items() if k != "entry_mode"}

        print(f"\n  {pair} (params: rb={base_params['ribbon']} cd={base_params['cooldown']} "
              f"adx={base_params['adx_gate']} exit_brick={base_params['exit_on_brick']}):")

        for mode in ENTRY_MODES:
            match = [
                r for r in results
                if r["params"]["entry_mode"] == mode
                and all(r["params"][k] == v for k, v in base_params.items())
            ]
            if match:
                r = match[0]
                pf_s = f"{r['pf']:.2f}" if not math.isinf(r["pf"]) else "INF"
                print(f"    {mode:<12}  PF={pf_s:>7}  Net={r['net']:>9.2f}  "
                      f"T={r['trades']:>4}  WR={r['win_rate']:>5.1f}%  "
                      f"DD={r['max_dd_pct']:>5.2f}%")


if __name__ == "__main__":
    main()
