"""
Batch sweep: run R022/R023/R024 on a single instrument (all brick sizes).
Saves results to JSON for later aggregation.

Usage:
  cd renko
  python batch_sweep.py EURAUD
  python batch_sweep.py GBPJPY
"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent / "strategies"))

from runner import sweep, MIN_TRADES_FOR_RANK

DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "renko" / "sweep_results"
OUT_DIR.mkdir(exist_ok=True)

STRATEGIES = ["r022_ichimoku", "r023_williams_atr", "r024_keltner_breakout"]


def discover_files_for(instrument):
    """Return list of (brick_size, filename) for an instrument."""
    files = sorted(DATA_DIR.glob(f"OANDA_{instrument}*renko*.csv"))
    result = []
    for f in files:
        brick = f.name.split("renko ")[-1].replace(".csv", "")
        result.append((brick, f.name))
    return result


def main():
    instrument = sys.argv[1] if len(sys.argv) > 1 else None
    if not instrument:
        print("Usage: python batch_sweep.py <INSTRUMENT>")
        print("  e.g. python batch_sweep.py EURAUD")
        sys.exit(1)

    files = discover_files_for(instrument)
    if not files:
        print(f"No files found for {instrument}")
        sys.exit(1)

    print(f"\n{'#'*70}")
    print(f"# Batch sweep: {instrument} ({len(files)} files × {len(STRATEGIES)} strategies)")
    print(f"{'#'*70}")

    all_results = []

    for strat in STRATEGIES:
        for brick, filename in files:
            print(f"\n>>> {strat} | {instrument} brick={brick}")
            try:
                results, mod = sweep(strat, verbose=True, renko_file=filename)
                for r in results:
                    r["instrument"] = instrument
                    r["brick"] = brick
                    r["strategy"] = strat
                    # Convert inf for JSON
                    if math.isinf(r["pf"]):
                        r["pf"] = 999999.0
                all_results.extend(results)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Save to JSON
    out_file = OUT_DIR / f"{instrument}_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} results to {out_file}")

    # Print top 10 per strategy
    for strat in STRATEGIES:
        strat_results = [r for r in all_results if r["strategy"] == strat and r["trades"] >= MIN_TRADES_FOR_RANK]
        strat_results.sort(key=lambda r: (r["pf"], r["net"]), reverse=True)
        print(f"\n{'='*70}")
        print(f"TOP 5 — {strat} on {instrument}")
        print(f"{'='*70}")
        for i, r in enumerate(strat_results[:5], 1):
            print(f"  {i}. b={r['brick']:>6} PF={r['pf']:>8.4f} Net={r['net']:>8.2f} "
                  f"T={r['trades']:>4} WR={r['win_rate']:>5.1f}% DD={r['max_dd_pct']:>6.2f}% | {r['params']}")


if __name__ == "__main__":
    main()
