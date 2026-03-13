"""
MACD — Moving Average Convergence Divergence

Classic momentum/trend indicator. Shows the relationship between two EMAs.
Matches Pine's ta.macd(source, fast_length, slow_length, signal_smoothing).

Usage:
    from indicators.macd import calc_macd

    result = calc_macd(df)
    # result["macd"]      — MACD line: EMA(fast) - EMA(slow)
    # result["signal"]    — Signal line: EMA(MACD, signal_period)
    # result["histogram"] — MACD - Signal (positive = bullish, negative = bearish)

Interpretation:
    MACD crosses above signal  → bullish momentum (buy signal)
    MACD crosses below signal  → bearish momentum (sell signal)
    Both above zero            → uptrend context
    Both below zero            → downtrend context
    Histogram shrinking        → momentum weakening (potential reversal)
    Divergence (price vs MACD) → strong reversal signal
"""

import numpy as np
import pandas as pd
from indicators.ema import calc_ema


def calc_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    source_col: str = "Close",
) -> dict:
    """
    Parameters
    ----------
    df         : DataFrame with the source column (default 'Close')
    fast       : Fast EMA period (default 12)
    slow       : Slow EMA period (default 26)
    signal     : Signal line EMA period (default 9)
    source_col : Column to use as price source

    Returns
    -------
    dict with keys: macd, signal, histogram (all numpy arrays)
    """
    src = df[source_col]

    ema_fast = calc_ema(src, length=fast)
    ema_slow = calc_ema(src, length=slow)

    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, length=signal)
    histogram = macd_line - signal_line

    return {
        "macd":      macd_line.values,
        "signal":    signal_line.values,
        "histogram": histogram.values,
    }
