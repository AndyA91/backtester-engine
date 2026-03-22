#!/usr/bin/env python3
"""
bc_sweep_analysis.py — Deep analysis of bc_master_sweep results.

Reads ai_context/bc_sweep_results.json and produces:
  1. Ranked tables per instrument
  2. Gate isolation: how much does each gate individually change OOS PF vs baseline?
  3. Cross-instrument consensus: gates that win on all 3 instruments
  4. Matched-pair decay analysis: IS→OOS PF decay per gate vs baseline

Usage:
  python ai_context/bc_sweep_analysis.py
  python ai_context/bc_sweep_analysis.py --min-trades 20 --save
"""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BENCHMARKS = {
    "EURUSD": {"oos_pf": 12.79, "oos_trades": 63,  "label": "R008  n=5 cd=30"},
    "GBPJPY": {"oos_pf": 21.33, "oos_trades": 92,  "label": "GJ008 n=5 cd=20"},
    "EURAUD": {"oos_pf": 10.62, "oos_trades": 72,  "label": "EA008 n=5 cd=30"},
}

GATE_ORDER = [
    "baseline", "mk_any", "mk_strong", "fsb_any", "fsb_strong",
    "macd_rising", "lc_diverging", "macd_lc", "motn_dx",
    "mk_macd", "fsb_macd", "mk_fsb", "mk_motn", "fsb_motn", "mk_fsb_macd",
]


def load_results(path: Path) -> list:
    with open(path) as f:
        return json.load(f)


def safe_pf(v):
    return v if not math.isinf(v) else 9999.0


def gate_isolation(all_results: list, instrument: str, min_trades: int = 20) -> None:
    """
    For each gate, compute:
    - avg OOS PF across all viable n/cd combos
    - avg OOS PF for matched pairs (same n/cd) vs baseline
    - avg IS→OOS decay
    """
    inst = [r for r in all_results if r["instrument"] == instrument]
    baseline = {(r["n_bricks"], r["cooldown"]): r for r in inst if r["gate"] == "baseline"}

    print(f"\n  Gate isolation — {instrument} (matched pairs vs baseline, OOS trades >= {min_trades}):")
    print(f"  {'Gate':<15} {'Avg OOS PF':>12} {'vs Baseline':>12} {'Avg T':>7} {'Avg Decay':>10} {'N pairs':>8}")
    print(f"  {'-'*72}")

    for gate in GATE_ORDER:
        gate_rows = [r for r in inst if r["gate"] == gate and r["oos_trades"] >= min_trades]
        if not gate_rows:
            print(f"  {gate:<15} {'—':>12} {'—':>12} {'—':>7} {'—':>10} {'0':>8}")
            continue

        # Matched pairs: same (n_bricks, cooldown) as baseline
        pairs = []
        for r in gate_rows:
            key = (r["n_bricks"], r["cooldown"])
            b   = baseline.get(key)
            if b and b["oos_trades"] >= min_trades:
                pairs.append((r, b))

        avg_pf  = sum(safe_pf(r["oos_pf"]) for r in gate_rows) / len(gate_rows)
        avg_t   = sum(r["oos_trades"] for r in gate_rows) / len(gate_rows)
        valid_d = [r["decay_pct"] for r in gate_rows if not math.isnan(r["decay_pct"])]
        avg_dec = sum(valid_d) / len(valid_d) if valid_d else float("nan")

        if pairs:
            delta_pf = sum(safe_pf(r["oos_pf"]) - safe_pf(b["oos_pf"]) for r, b in pairs) / len(pairs)
            vs_str   = f"{delta_pf:>+10.2f} ({len(pairs)}p)"
        else:
            vs_str   = "  no pairs"

        dec_s = f"{avg_dec:>+9.1f}%" if not math.isnan(avg_dec) else "       NaN"
        print(f"  {gate:<15} {avg_pf:>12.2f} {vs_str:>12} {avg_t:>7.1f} {dec_s} {len(gate_rows):>8}")


def top_configs(all_results: list, instrument: str, min_trades: int, n: int = 10) -> None:
    """Print top N configs by OOS PF for an instrument."""
    inst   = [r for r in all_results if r["instrument"] == instrument and r["oos_trades"] >= min_trades]
    inst.sort(key=lambda r: safe_pf(r["oos_pf"]), reverse=True)
    bench  = BENCHMARKS[instrument]

    print(f"\n  Top {n} configs — {instrument} (OOS trades >= {min_trades})")
    print(f"  Benchmark: OOS PF {bench['oos_pf']}  {bench['oos_trades']}t  [{bench['label']}]")
    print(f"  {'Gate':<15} {'n':>2} {'cd':>3} | {'IS PF':>7} {'T':>5} | {'OOS PF':>7} {'T':>5} {'WR%':>6} {'Decay':>7}")
    print(f"  {'-'*70}")
    for r in inst[:n]:
        beat  = " <<BEAT" if r["oos_pf"] > bench["oos_pf"] else ""
        dec_s = f"{r['decay_pct']:>+6.1f}%" if not math.isnan(r["decay_pct"]) else "   NaN"
        print(f"  {r['gate']:<15} {r['n_bricks']:>2} {r['cooldown']:>3} | "
              f"{r['is_pf']:>7.2f} {r['is_trades']:>5} | "
              f"{r['oos_pf']:>7.2f} {r['oos_trades']:>5} {r['oos_wr']:>6.1f}% "
              f"{dec_s}{beat}")


def cross_instrument_consensus(all_results: list, min_trades: int) -> None:
    """Gates that consistently beat/improve benchmark across all instruments."""
    print(f"\n{'='*76}")
    print("  Cross-instrument gate consensus")
    print(f"{'='*76}")
    print(f"  {'Gate':<15} {'EURUSD':>12} {'GBPJPY':>12} {'EURAUD':>12} {'Wins':>6}")
    print(f"  {'-'*60}")

    for gate in GATE_ORDER:
        row  = [f"  {gate:<15}"]
        wins = 0
        for inst in ["EURUSD", "GBPJPY", "EURAUD"]:
            gv = [r for r in all_results
                  if r["instrument"] == inst and r["gate"] == gate and r["oos_trades"] >= min_trades]
            if gv:
                avg_pf = sum(safe_pf(r["oos_pf"]) for r in gv) / len(gv)
                bmark  = BENCHMARKS[inst]["oos_pf"]
                marker = "+" if avg_pf > bmark else " "
                row.append(f"{avg_pf:>11.2f}{marker}")
                if avg_pf > bmark:
                    wins += 1
            else:
                row.append(f"{'  N/A':>12}")
        row.append(f"{wins:>6}")
        print("".join(row))


def decay_summary(all_results: list, instrument: str, min_trades: int) -> None:
    """For each gate, show IS→OOS decay vs baseline."""
    inst     = [r for r in all_results if r["instrument"] == instrument]
    baseline = [r for r in inst if r["gate"] == "baseline" and r["oos_trades"] >= min_trades]
    if not baseline:
        return
    base_avg_decay = sum(r["decay_pct"] for r in baseline if not math.isnan(r["decay_pct"]))
    base_avg_decay /= len(baseline)

    print(f"\n  IS→OOS decay vs baseline — {instrument} (viable, OOS trades >= {min_trades})")
    print(f"  Baseline avg decay: {base_avg_decay:+.1f}%")
    print(f"  {'Gate':<15} {'Avg IS PF':>10} {'Avg OOS PF':>12} {'Avg Decay':>10} {'Delta':>8}")
    print(f"  {'-'*60}")

    for gate in GATE_ORDER:
        if gate == "baseline":
            continue
        gv = [r for r in inst if r["gate"] == gate and r["oos_trades"] >= min_trades]
        if not gv:
            continue
        avg_is  = sum(safe_pf(r["is_pf"])  for r in gv) / len(gv)
        avg_oos = sum(safe_pf(r["oos_pf"]) for r in gv) / len(gv)
        valid_d = [r["decay_pct"] for r in gv if not math.isnan(r["decay_pct"])]
        avg_dec = sum(valid_d) / len(valid_d) if valid_d else float("nan")
        delta   = avg_dec - base_avg_decay if not math.isnan(avg_dec) else float("nan")
        dec_s   = f"{avg_dec:>+9.1f}%" if not math.isnan(avg_dec) else "       NaN"
        dlt_s   = f"{delta:>+7.1f}%" if not math.isnan(delta) else "     NaN"
        print(f"  {gate:<15} {avg_is:>10.2f} {avg_oos:>12.2f} {dec_s} {dlt_s}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-trades", type=int, default=20)
    parser.add_argument("--save", action="store_true", help="Save analysis to markdown file")
    args = parser.parse_args()

    results_path = ROOT / "ai_context" / "bc_sweep_results.json"
    if not results_path.exists():
        print(f"ERROR: {results_path} not found — run bc_master_sweep.py first")
        sys.exit(1)

    results = load_results(results_path)
    print(f"Loaded {len(results)} results from {results_path}")
    print(f"Min OOS trades: {args.min_trades}")

    for inst in ["EURUSD", "GBPJPY", "EURAUD"]:
        print(f"\n{'='*76}")
        print(f"  {inst}")
        print(f"{'='*76}")
        top_configs(results, inst, args.min_trades)
        gate_isolation(results, inst, args.min_trades)
        decay_summary(results, inst, args.min_trades)

    cross_instrument_consensus(results, args.min_trades)


if __name__ == "__main__":
    main()
