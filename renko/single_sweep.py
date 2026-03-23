"""
Single-file sweep: run one strategy on one renko file.
Prints top 5 results. Fast enough to avoid timeouts.

Usage:
  cd renko
  python single_sweep.py r022_ichimoku "OANDA_GBPJPY, 1S renko 0.05.csv"
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent / "strategies"))

from runner import sweep


def main():
    strategy = sys.argv[1]
    renko_file = sys.argv[2]
    results, mod = sweep(strategy, verbose=True, renko_file=renko_file)


if __name__ == "__main__":
    main()
