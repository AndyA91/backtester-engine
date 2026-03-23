"""
Accumulation/Distribution Line (A/D Line)

Volume-based indicator that uses the close-location-value (CLV) to weight
volume, providing a more granular view than OBV's binary up/down approach.
Catches institutional accumulation/distribution before price moves.

CLV = ((Close - Low) - (High - Close)) / (High - Low)
A/D = cumsum(CLV * Volume)

Usage:
    from indicators.ad_line import calc_ad_line

    result = calc_ad_line(df, ema_period=21)
    # result["ad"]     — Accumulation/Distribution line
    # result["ad_ema"] — EMA of A/D line (signal line)

Interpretation:
    A/D rising while price flat/falling → accumulation (bullish divergence)
    A/D falling while price flat/rising → distribution (bearish divergence)
    A/D confirming price trend → trend is healthy
    A/D crosses above its EMA → buying pressure increasing
    Compare A/D divergences with RSI divergences for confluence
"""

import numpy as np
import pandas as pd
from indicators.ema import calc_ema


def calc_ad_line(
    df: pd.DataFrame,
    ema_period: int = 21,
) -> dict:
    """
    Parameters
    ----------
    df         : DataFrame with 'High', 'Low', 'Close', 'Volume'
    ema_period : EMA period for signal line (default 21)

    Returns
    -------
    dict with keys: ad, ad_ema (both numpy arrays)
    """
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    close = df["Close"].values.astype(float)
    volume = df["Volume"].values.astype(float)

    hl_range = high - low

    # Close Location Value: where the close sits within the bar's range
    with np.errstate(invalid="ignore", divide="ignore"):
        clv = np.where(
            hl_range > 0,
            ((close - low) - (high - close)) / hl_range,
            0.0,
        )

    # A/D = cumulative sum of CLV * Volume
    ad = np.cumsum(clv * volume)

    # Signal line
    ad_series = pd.Series(ad)
    ad_ema = calc_ema(ad_series, length=ema_period).values

    return {
        "ad": ad,
        "ad_ema": ad_ema,
    }
