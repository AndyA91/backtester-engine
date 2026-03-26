"""
Stock Renko data loader.

Thin wrapper around renko/data.py — same TV Renko CSV format.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from renko.data import load_renko_export
from renko.indicators import add_renko_indicators


def load_stock_renko(filename: str):
    """Load a stock Renko CSV and enrich with standard indicators."""
    df = load_renko_export(filename)
    add_renko_indicators(df)
    return df
