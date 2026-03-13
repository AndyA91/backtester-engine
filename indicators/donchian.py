"""
Donchian Channel

Rolling highest-high / lowest-low envelope over N periods.
Matches Pine's ta.highest(high, length) / ta.lowest(low, length) exactly.

Usage:
    from indicators.donchian import calc_donchian
    result = calc_donchian(df, period=20)
    # result["upper"] — highest High over last N bars
    # result["lower"] — lowest Low over last N bars
    # result["mid"]   — (upper + lower) / 2

Interpretation:
    Price breaks above upper → bullish breakout (momentum entry)
    Price breaks below lower → bearish breakout (momentum entry)
    Price crosses mid from above/below → mean-reversion target or exit
    Channel width → volatility proxy (wide = trending, narrow = consolidating)
"""

import numpy as np
import pandas as pd


def calc_donchian(
    df: pd.DataFrame,
    period: int = 20,
    high_col: str = "High",
    low_col: str = "Low",
) -> dict:
    """
    Parameters
    ----------
    df       : DataFrame with High and Low columns
    period   : Rolling lookback for highest high / lowest low
    high_col : Column name for High prices
    low_col  : Column name for Low prices
    """
    high = pd.Series(df[high_col].values)
    low  = pd.Series(df[low_col].values)

    upper = high.rolling(period).max().values
    lower = low.rolling(period).min().values
    mid   = (upper + lower) / 2.0

    return {
        "upper": upper,
        "lower": lower,
        "mid":   mid,
    }
