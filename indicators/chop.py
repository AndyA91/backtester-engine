"""
Choppiness Index (CHOP)

Measures whether the market is trending or ranging (choppy).
Ranges from ~18 (perfectly trending) to 100 (perfectly choppy).
A neutral/random market sits near 61.8 (the golden ratio level).

Matches the standard Choppiness Index formula used on TradingView.

Usage:
    from indicators.chop import calc_chop

    result = calc_chop(df, period=14)
    # result["chop"] — Choppiness Index values (18–100)

Interpretation:
    CHOP < 38.2  → strong trend (price is directional)
    CHOP > 61.8  → choppy / ranging market (avoid trend strategies)
    Falling CHOP → trend developing / gaining strength
    Rising CHOP  → market becoming more random/choppy
    Classic thresholds: 38.2 (trending) and 61.8 (choppy) — both are Fibonacci levels
"""

import numpy as np
import pandas as pd
from indicators.atr import calc_atr


def calc_chop(
    df: pd.DataFrame,
    period: int = 14,
    atr_period: int = 1,
) -> dict:
    """
    CHOP = 100 × log10(Σ ATR(1, period) / (Highest High - Lowest Low)) / log10(period)

    where:
        Σ ATR(1, period) = sum of True Range over 'period' bars (ATR with period=1 × N)
        Highest High     = rolling max of High over 'period' bars
        Lowest Low       = rolling min of Low over 'period' bars

    Parameters
    ----------
    df         : DataFrame with 'High', 'Low', 'Close'
    period     : Lookback period for CHOP (default 14)
    atr_period : Period for each individual ATR bar (default 1 = True Range)

    Returns
    -------
    dict with key: chop (numpy array, 18–100)
    """
    high  = pd.Series(df["High"].values)
    low   = pd.Series(df["Low"].values)

    # True Range (ATR period=1 matches ta.tr)
    atr_result = calc_atr(df, period=atr_period, method="sma")
    tr = atr_result["tr"]
    tr_s = pd.Series(tr)

    atr_sum = tr_s.rolling(period).sum().values
    hh      = high.rolling(period).max().values
    ll      = low.rolling(period).min().values
    hl_range = hh - ll

    log_period = np.log10(period)

    with np.errstate(invalid="ignore", divide="ignore"):
        chop = np.where(
            hl_range > 0,
            100.0 * np.log10(atr_sum / hl_range) / log_period,
            np.nan,
        )

    # NaN out warmup
    chop[:period - 1] = np.nan

    return {"chop": chop}
