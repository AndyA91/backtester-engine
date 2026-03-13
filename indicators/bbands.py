"""
Bollinger Bands

SMA-based price envelope at N standard deviations above/below the mean.
Matches Pine's ta.bb(source, length, mult) exactly.

Pine uses population std (biased, ddof=0) via ta.stdev(source, length, biased=true).

Usage:
    from indicators.bbands import calc_bbands
    result = calc_bbands(df, period=20, mult=2.0)
    # result["mid"]   — SMA(close, period)
    # result["upper"] — mid + mult * std
    # result["lower"] — mid - mult * std
    # result["bw"]    — bandwidth: (upper - lower) / mid
    # result["pct_b"] — %B: position of close within bands (0=lower, 1=upper)
    # result["std"]   — rolling population standard deviation

Interpretation:
    Price outside bands → extreme move (oversold/overbought)
    Price crosses back inside → mean-reversion signal
    Bandwidth expands → volatility expanding; contracts → squeeze building
    %B > 1 → above upper band; %B < 0 → below lower band
"""

import numpy as np
import pandas as pd


def calc_bbands(
    df: pd.DataFrame,
    period: int = 20,
    mult: float = 2.0,
    source_col: str = "Close",
) -> dict:
    """
    Parameters
    ----------
    df         : DataFrame with the source column (default 'Close')
    period     : SMA / std lookback period
    mult       : Standard deviation multiplier for the bands
    source_col : Column to use as price source
    """
    src = df[source_col].values
    n   = len(src)

    s   = pd.Series(src)
    mid = s.rolling(period).mean().values
    std = s.rolling(period).std(ddof=0).values   # population std — matches Pine ta.stdev

    upper = mid + mult * std
    lower = mid - mult * std

    with np.errstate(invalid="ignore", divide="ignore"):
        bw    = np.where(mid > 0, (upper - lower) / mid, np.nan)
        pct_b = np.where((upper - lower) > 0, (src - lower) / (upper - lower), np.nan)

    return {
        "mid":   mid,
        "upper": upper,
        "lower": lower,
        "bw":    bw,
        "pct_b": pct_b,
        "std":   std,
    }
