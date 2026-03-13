"""
Williams %R

Momentum oscillator, range -100 to 0. Measures where the close sits within
the period's high-low range, inverted (overbought near 0, oversold near -100).
Matches Pine's ta.wpr(length).

Usage:
    from indicators.williams_r import calc_williams_r

    result = calc_williams_r(df, period=14)
    # result["wpr"] — Williams %R values (-100 to 0)

Interpretation:
    %R > -20  → overbought (price near top of recent range)
    %R < -80  → oversold  (price near bottom of recent range)
    %R crosses above -50 from below → bullish momentum shift
    %R crosses below -50 from above → bearish momentum shift
    Best used with trend filter — fade extremes in range, follow extremes in trend
"""

import numpy as np
import pandas as pd


def calc_williams_r(
    df: pd.DataFrame,
    period: int = 14,
) -> dict:
    """
    %R = (Highest High - Close) / (Highest High - Lowest Low) * -100

    Parameters
    ----------
    df     : DataFrame with 'High', 'Low', 'Close'
    period : Lookback period (default 14)

    Returns
    -------
    dict with key: wpr (numpy array, values -100 to 0)
    """
    high  = pd.Series(df["High"].values)
    low   = pd.Series(df["Low"].values)
    close = df["Close"].values

    hh = high.rolling(period).max().values
    ll = low.rolling(period).min().values

    hl_range = hh - ll

    with np.errstate(invalid="ignore", divide="ignore"):
        wpr = np.where(hl_range > 0, (hh - close) / hl_range * -100.0, -50.0)

    # NaN out warmup bars
    wpr[:period - 1] = np.nan

    return {"wpr": wpr}
