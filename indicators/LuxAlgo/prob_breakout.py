"""
Probabilistic Breakout Forecaster [LuxAlgo]

Uses a geometric random walk model with normal CDF to estimate the
probability that price will break above/below its Donchian channel
within a forecast horizon.

Pine source: indicators/LuxAlgo/Probabilistic_Breakout_Forecaster__LuxAlgo_.pine

Outputs:
    bull_prob — probability of breaking above upper range (0-100)
    bear_prob — probability of breaking below lower range (0-100)
    squeeze   — volatility squeeze intensity (0-100), higher = more coiled

Usage:
    from indicators.luxalgo.prob_breakout import calc_prob_breakout
    result = calc_prob_breakout(df)
    # result["bull_prob"], result["bear_prob"], result["squeeze"]
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


def calc_prob_breakout(
    df: pd.DataFrame,
    length: int = 20,
    horizon: int = 10,
    vol_lookback: int = 50,
) -> dict:
    """
    Parameters
    ----------
    df           : DataFrame with 'High', 'Low', 'Close'
    length       : Donchian channel lookback (range boundaries)
    horizon      : Forecast horizon in bars
    vol_lookback : Lookback for log-return standard deviation
    """
    high = pd.Series(df["High"].values.astype(float))
    low = pd.Series(df["Low"].values.astype(float))
    close = pd.Series(df["Close"].values.astype(float))
    n = len(close)

    # Donchian channel boundaries
    upper_range = high.rolling(length, min_periods=length).max().values
    lower_range = low.rolling(length, min_periods=length).min().values

    # Log-return volatility
    log_returns = np.log(close / close.shift(1)).values
    sigma = pd.Series(log_returns).rolling(vol_lookback, min_periods=vol_lookback).std().values

    # Forecast
    horizon_sqrt = np.sqrt(horizon)
    close_arr = close.values

    bull_prob = np.zeros(n)
    bear_prob = np.zeros(n)

    for i in range(n):
        denom = sigma[i] * horizon_sqrt if not np.isnan(sigma[i]) else 0.0
        if denom > 0 and close_arr[i] > 0 and not np.isnan(upper_range[i]) and not np.isnan(lower_range[i]):
            z_upper = np.log(upper_range[i] / close_arr[i]) / denom
            z_lower = np.log(lower_range[i] / close_arr[i]) / denom
            bull_prob[i] = (1.0 - norm.cdf(z_upper)) * 100.0
            bear_prob[i] = norm.cdf(z_lower) * 100.0

    # Squeeze: compares current ATR(14) to SMA(100) of ATR
    tr = np.empty(n)
    tr[0] = high.values[0] - low.values[0]
    for i in range(1, n):
        tr[i] = max(
            high.values[i] - low.values[i],
            abs(high.values[i] - close_arr[i - 1]),
            abs(low.values[i] - close_arr[i - 1]),
        )
    current_atr = pd.Series(tr).rolling(14, min_periods=1).mean().values
    avg_atr = pd.Series(current_atr).rolling(100, min_periods=1).mean().values

    squeeze = np.zeros(n)
    for i in range(n):
        if avg_atr[i] > 0:
            squeeze[i] = max(0.0, 1.0 - current_atr[i] / avg_atr[i]) * 100.0

    return {
        "bull_prob": bull_prob,
        "bear_prob": bear_prob,
        "squeeze": squeeze,
    }
