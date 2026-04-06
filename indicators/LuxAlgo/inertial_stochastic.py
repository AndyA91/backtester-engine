"""
Inertial Stochastic [LuxAlgo]

Adaptive-length stochastic oscillator. For each bar, finds the lookback
length N (between min_len and max_len) that produces the stochastic value
closest to the previous bar's value — creating "inertia" that resists sudden
jumps and produces a smoother, less whipsaw-prone oscillator.

Pine source: indicators/LuxAlgo/Inertial_Stochastic__LuxAlgo_.pine

Outputs:
    k  — inertial stochastic %K (0-100), SMA-smoothed
    d  — %D signal line (SMA of %K)

Usage:
    from indicators.luxalgo.inertial_stochastic import calc_inertial_stochastic
    result = calc_inertial_stochastic(df, min_len=10, max_len=40, smooth_k=3, smooth_d=3)
    # result["k"], result["d"]
"""

import numpy as np
import pandas as pd


def calc_inertial_stochastic(
    df: pd.DataFrame,
    min_len: int = 10,
    max_len: int = 40,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> dict:
    """
    Parameters
    ----------
    df       : DataFrame with 'High', 'Low', 'Close'
    min_len  : Minimum stochastic lookback length
    max_len  : Maximum stochastic lookback length
    smooth_k : SMA period for %K smoothing
    smooth_d : SMA period for %D signal line
    """
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    close = df["Close"].values.astype(float)
    n = len(close)

    raw_stoch = np.full(n, np.nan)
    prev_stoch = 50.0  # Pine: var float lvStoch = 50.0

    for i in range(n):
        best_stoch = 50.0
        min_diff = 1e10
        hh = high[i]
        ll = low[i]

        for j in range(1, min(max_len, i + 1)):
            hh = max(hh, high[i - j])
            ll = min(ll, low[i - j])
            current_len = j + 1

            if current_len >= min_len:
                den = hh - ll
                stoch = 50.0 if den == 0 else 100.0 * (close[i] - ll) / den
                diff = abs(stoch - prev_stoch)

                if diff < min_diff:
                    min_diff = diff
                    best_stoch = stoch

        prev_stoch = best_stoch
        raw_stoch[i] = best_stoch

    # SMA smoothing (matches Pine ta.sma)
    k = pd.Series(raw_stoch).rolling(smooth_k, min_periods=smooth_k).mean().values
    d = pd.Series(k).rolling(smooth_d, min_periods=smooth_d).mean().values

    return {"k": k, "d": d}
