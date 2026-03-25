"""
Ehlers Fisher Transform

Converts prices into a Gaussian normal distribution, producing sharp turning
points that are much easier to identify than raw price action. Developed by
John Ehlers.

The transform normalizes price to a -1 to +1 range, then applies the inverse
Fisher function: Fisher = 0.5 * ln((1 + x) / (1 - x)). This creates an
oscillator with well-defined peaks and valleys.

Usage:
    from indicators.fisher_transform import calc_fisher_transform

    result = calc_fisher_transform(df, period=10)
    # result["fisher"]  — Fisher Transform line
    # result["signal"]  — Signal line (previous bar's Fisher value)

Interpretation:
    Fisher crosses above signal → bullish reversal signal
    Fisher crosses below signal → bearish reversal signal
    Extreme Fisher values (> 1.5 or < -1.5) → imminent reversal
    Sharp, well-defined peaks make turning points unambiguous
    Best for mean-reversion entries — pair with ADX regime filter
"""

import numpy as np
import pandas as pd


def calc_fisher_transform(
    df: pd.DataFrame,
    period: int = 10,
) -> dict:
    """
    Parameters
    ----------
    df     : DataFrame with 'High', 'Low'
    period : Lookback for normalization (default 10)

    Returns
    -------
    dict with keys: fisher, signal (both numpy arrays)
    """
    high = pd.Series(df["High"].values, dtype=float)
    low = pd.Series(df["Low"].values, dtype=float)
    n = len(high)

    hl2 = (high + low) / 2.0

    hh = high.rolling(period).max().values
    ll = low.rolling(period).min().values

    fisher = np.zeros(n)
    signal = np.zeros(n)
    value = np.zeros(n)

    for i in range(period - 1, n):
        hl_range = hh[i] - ll[i]
        if hl_range == 0:
            raw = 0.0
        else:
            raw = 2.0 * ((hl2.values[i] - ll[i]) / hl_range) - 1.0

        # Clamp to avoid log(0) — matches Pine Script behavior
        raw = max(-0.999, min(0.999, raw))

        # Smooth with previous value (EMA-like, alpha = 0.5)
        value[i] = 0.5 * raw + 0.5 * (value[i - 1] if i > 0 else 0.0)

        # Clamp again after smoothing
        v = max(-0.999, min(0.999, value[i]))

        # Inverse Fisher function
        fisher[i] = 0.5 * np.log((1.0 + v) / (1.0 - v))

        # Signal is previous bar's Fisher
        signal[i] = fisher[i - 1] if i > 0 else 0.0

    # NaN out warmup
    fisher[: period - 1] = np.nan
    signal[: period - 1] = np.nan

    return {
        "fisher": fisher,
        "signal": signal,
    }
