"""
Awesome Oscillator (AO) — Bill Williams

Simple momentum oscillator: difference between the 5-period and 34-period
SMA of the bar's midpoint (HL/2). No parameters to optimize, making it
robust against overfitting.

AO = SMA(HL/2, 5) - SMA(HL/2, 34)

Usage:
    from indicators.awesome_oscillator import calc_ao

    result = calc_ao(df, fast=5, slow=34)
    # result["ao"]    — Awesome Oscillator values
    # result["color"] — +1 (green, rising), -1 (red, falling)

Interpretation:
    AO > 0 → bullish momentum (5-bar midpoint above 34-bar midpoint)
    AO < 0 → bearish momentum
    AO crossing zero → momentum shift (trend change signal)
    "Saucer" pattern (2 consecutive same-color bars after opposite) → continuation
    Twin Peaks: two peaks on same side of zero, second smaller → reversal
    Simple, parameter-free complement to MACD
"""

import numpy as np
import pandas as pd


def calc_ao(
    df: pd.DataFrame,
    fast: int = 5,
    slow: int = 34,
) -> dict:
    """
    Parameters
    ----------
    df   : DataFrame with 'High', 'Low'
    fast : Fast SMA period (default 5)
    slow : Slow SMA period (default 34)

    Returns
    -------
    dict with keys: ao, color (both numpy arrays)
    """
    hl2 = (df["High"].values + df["Low"].values) / 2.0
    hl2_s = pd.Series(hl2)

    sma_fast = hl2_s.rolling(fast).mean().values
    sma_slow = hl2_s.rolling(slow).mean().values

    ao = sma_fast - sma_slow

    # Color: green (+1) if AO is rising, red (-1) if falling
    color = np.zeros(len(ao), dtype=int)
    for i in range(1, len(ao)):
        if not np.isnan(ao[i]) and not np.isnan(ao[i - 1]):
            color[i] = 1 if ao[i] > ao[i - 1] else -1

    return {
        "ao": ao,
        "color": color,
    }
