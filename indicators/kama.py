"""
KAMA — Kaufman's Adaptive Moving Average

Adapts smoothing speed based on market efficiency:
  - Trending (high ER) → tracks price closely
  - Choppy (low ER)    → smooths out noise

Usage:
    from indicators.kama import calc_kama

    kama = calc_kama(df["Close"])                    # default: ER=10, fast=2, slow=30
    kama = calc_kama(df["Close"], length=14, fast=3) # custom params
    kama_values = kama.values                        # numpy array
"""

import numpy as np
import pandas as pd


def calc_kama(
    series: pd.Series,
    length: int = 10,
    fast: int = 2,
    slow: int = 30,
) -> pd.Series:
    """
    Kaufman's Adaptive Moving Average, matching TradingView's ta.kama().

    Parameters
    ----------
    series : pd.Series
        Price series (typically Close).
    length : int
        Efficiency Ratio lookback period. Default: 10.
    fast : int
        Fastest smoothing period (trending). Default: 2.
    slow : int
        Slowest smoothing period (choppy). Default: 30.

    Returns
    -------
    pd.Series
        KAMA values with the same index as input.
    """
    src = series.values.astype(float)
    n = len(src)

    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)

    kama = np.full(n, np.nan)

    for i in range(n):
        if i < length:
            # Not enough bars for ER — seed with source
            kama[i] = src[i]
            continue

        # Efficiency Ratio
        direction = abs(src[i] - src[i - length])
        volatility = sum(abs(src[j] - src[j - 1]) for j in range(i - length + 1, i + 1))

        er = direction / volatility if volatility != 0 else 0.0

        # Smoothing Constant
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        # KAMA
        prev = kama[i - 1] if not np.isnan(kama[i - 1]) else src[i]
        kama[i] = prev + sc * (src[i] - prev)

    return pd.Series(kama, index=series.index, name="KAMA")
