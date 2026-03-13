"""
Stochastic Oscillator

Classic momentum oscillator. Measures where the close sits within the
high-low range over the lookback period.

Matches Pine's ta.stoch(close, high, low, k_period) for the raw %K, then
SMA smoothing for slow %K and %D.

  Fast %K = 100 * (close - lowest_low(k)) / (highest_high(k) - lowest_low(k))
  Slow %K = SMA(Fast %K, smooth_k)   — default 3 bars
  %D      = SMA(Slow %K, smooth_d)   — default 3 bars (signal line)

Usage:
    from indicators.stochastic import calc_stochastic
    result = calc_stochastic(df, k_period=14, smooth_k=3, smooth_d=3)
    # result["fast_k"] — raw stochastic (noisy, matches ta.stoch())
    # result["slow_k"] — smoothed %K (main line)
    # result["pct_d"]  — signal line (%D)

Interpretation:
    slow_k < 20  → oversold territory
    slow_k > 80  → overbought territory
    slow_k crosses pct_d from below in oversold zone → bullish signal
    slow_k crosses pct_d from above in overbought zone → bearish signal
    Faster than RSI — generates more extreme readings per session.
"""

import numpy as np
import pandas as pd


def calc_stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> dict:
    """
    Parameters
    ----------
    df       : DataFrame with 'High', 'Low', 'Close'
    k_period : Lookback for highest_high / lowest_low
    smooth_k : SMA period for slow %K (smoothed %K line)
    smooth_d : SMA period for %D signal line
    """
    high  = pd.Series(df["High"].values)
    low   = pd.Series(df["Low"].values)
    close = pd.Series(df["Close"].values)

    highest_high = high.rolling(k_period).max()
    lowest_low   = low.rolling(k_period).min()
    hl_range     = highest_high - lowest_low

    with np.errstate(invalid="ignore", divide="ignore"):
        fast_k = np.where(
            hl_range > 0,
            100.0 * (close - lowest_low) / hl_range,
            50.0,
        )
    # NaN out the warmup period
    fast_k = pd.Series(fast_k)
    fast_k.iloc[:k_period - 1] = np.nan

    slow_k = fast_k.rolling(smooth_k).mean()
    pct_d  = slow_k.rolling(smooth_d).mean()

    return {
        "fast_k": fast_k.values,
        "slow_k": slow_k.values,
        "pct_d":  pct_d.values,
    }
