"""
Multi-instrument sweep: run one strategy across ALL Renko data files.

Usage:
  cd renko
  python multi_sweep.py r022_ichimoku
  python multi_sweep.py r023_williams_atr
  python multi_sweep.py r024_keltner_breakout
"""

import sys
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent / "strategies"))

from runner import sweep, MIN_TRADES_FOR_RANK

DATA_DIR = ROOT / "data"

# Commission & qty per instrument
INST_CFG = {
    "EURUSD": {"commission_pct": 0.0043, "qty": 1000},
    "GBPUSD": {"commission_pct": 0.0043, "qty": 1000},
    "GBPJPY": {"commission_pct": 0.0043, "qty": 1000},
    "USDJPY": {"commission_pct": 0.0043, "qty": 1000},
    "EURAUD": {"commission_pct": 0.0043, "qty": 1000},
    "BTCUSD": {"commission_pct": 0.10,   "qty": 0.01},
}


def discover_renko_files():
    """Return list of (instrument, brick_size, filename)."""
    files = sorted(DATA_DIR.glob("OANDA_*renko*.csv"))
    result = []
    for f in files:
        name = f.name
        # Extract instrument: OANDA_EURUSD -> EURUSD
        parts = name.split(",")[0].replace("OANDA_", "")
        # Handle BTCUSD.SPOT.US -> BTCUSD
        inst = parts.split(".")[0]
        # Extract brick size from "renko 0.0004.csv"
        brick = name.split("renko ")[-1].replace(".csv", "")
        result.append((inst, brick, name))
    return result


def main():
    strategy = sys.argv[1] if len(sys.argv) > 1 else None
    if not strategy:
        print("Usage: python multi_sweep.py <strategy_module>")
        sys.exit(1)

    renko_files = discover_renko_files()
    print(f"\n{'#'*70}")
    print(f"# Multi-instrument sweep: {strategy}")
    print(f"# Files: {len(renko_files)}")
    print(f"{'#'*70}")

    global_results = []

    for inst, brick, filename in renko_files:
        print(f"\n>>> {inst} brick={brick} ({filename})")
        try:
            results, mod = sweep(strategy, verbose=True, renko_file=filename)
            for r in results:
                r["instrument"] = inst
                r["brick"] = brick
            global_results.extend(results)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Global ranking
    global_results.sort(
        key=lambda r: (
            r["trades"] >= MIN_TRADES_FOR_RANK,
            r["pf"] if not math.isinf(r["pf"]) else 1e12,
            r["net"],
        ),
        reverse=True,
    )

    print(f"\n{'='*80}")
    print(f"GLOBAL TOP 20 — {strategy}")
    print(f"{'='*80}")
    for i, r in enumerate(global_results[:20], 1):
        tag = "OK" if r["trades"] >= MIN_TRADES_FOR_RANK else "LOW_T"
        pf = "INF" if math.isinf(r["pf"]) else f"{r['pf']:.4f}"
        print(
            f"  {i:>2}. [{tag}] {r['instrument']:>6} b={r['brick']:>6} "
            f"PF={pf:>8} Net={r['net']:>10.2f} T={r['trades']:>4} "
            f"WR={r['win_rate']:>5.1f}% DD={r['max_dd_pct']:>6.2f}% | {r['params']}"
        )

    # Per-instrument best
    instruments = sorted(set(r["instrument"] for r in global_results))
    print(f"\n{'='*80}")
    print(f"BEST PER INSTRUMENT — {strategy}")
    print(f"{'='*80}")
    for inst in instruments:
        inst_results = [r for r in global_results if r["instrument"] == inst and r["trades"] >= MIN_TRADES_FOR_RANK]
        if inst_results:
            r = inst_results[0]
            pf = "INF" if math.isinf(r["pf"]) else f"{r['pf']:.4f}"
            print(
                f"  {inst:>6} b={r['brick']:>6} "
                f"PF={pf:>8} Net={r['net']:>10.2f} T={r['trades']:>4} "
                f"WR={r['win_rate']:>5.1f}% DD={r['max_dd_pct']:>6.2f}% | {r['params']}"
            )


if __name__ == "__main__":
    main()
