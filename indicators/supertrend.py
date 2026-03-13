"""
Supertrend — ATR-Based Trend Following Indicator

Flips between bullish/bearish based on ATR bands around HL2.
Green = bullish (price above), Red = bearish (price below).

Usage:
    from indicators.supertrend import calc_supertrend
    result = calc_supertrend(df, period=10, multiplier=3.0)
    # result["supertrend"]  — Supertrend line values
    # result["direction"]   — +1 bullish, -1 bearish
    # result["upper_band"]  — Upper ATR band
    # result["lower_band"]  — Lower ATR band
"""

import numpy as np
import pandas as pd
from indicators.atr import calc_atr


def calc_supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
) -> dict:
    """
    Parameters
    ----------
    df : DataFrame with 'High', 'Low', 'Close'
    period : ATR period
    multiplier : ATR multiplier for band width
    """
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    n = len(close)

    atr_result = calc_atr(df, period=period)
    atr = atr_result["atr"]

    hl2 = (high + low) / 2.0

    # Basic bands
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    # Final bands with trailing logic
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.ones(n)  # +1 = bullish, -1 = bearish

    upper_band[0] = basic_upper[0]
    lower_band[0] = basic_lower[0]
    supertrend[0] = basic_lower[0]

    for i in range(1, n):
        # Lower band: only moves up (trailing support)
        if basic_lower[i] > lower_band[i-1] or close[i-1] < lower_band[i-1]:
            lower_band[i] = basic_lower[i]
        else:
            lower_band[i] = lower_band[i-1]

        # Upper band: only moves down (trailing resistance)
        if basic_upper[i] < upper_band[i-1] or close[i-1] > upper_band[i-1]:
            upper_band[i] = basic_upper[i]
        else:
            upper_band[i] = upper_band[i-1]

        # Direction flip
        if direction[i-1] == 1:  # was bullish
            if close[i] < lower_band[i]:
                direction[i] = -1  # flip bearish
                supertrend[i] = upper_band[i]
            else:
                direction[i] = 1
                supertrend[i] = lower_band[i]
        else:  # was bearish
            if close[i] > upper_band[i]:
                direction[i] = 1  # flip bullish
                supertrend[i] = lower_band[i]
            else:
                direction[i] = -1
                supertrend[i] = upper_band[i]

    return {
        "supertrend": supertrend,
        "direction": direction,
        "upper_band": upper_band,
        "lower_band": lower_band,
    }
