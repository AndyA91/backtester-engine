"""
OBV — On-Balance Volume

Cumulative volume indicator: adds volume on up days, subtracts on down days.
Shows whether volume is flowing into or out of a security.
Matches Pine's ta.obv.

Usage:
    from indicators.obv import calc_obv

    result = calc_obv(df)
    # result["obv"]     — cumulative OBV (raw units)
    # result["obv_ema"] — EMA of OBV (optional signal line, period=21)

Interpretation:
    OBV rising  + price rising  → volume confirms uptrend (healthy)
    OBV falling + price rising  → volume diverges (weakening uptrend — caution)
    OBV rising  + price falling → accumulation (potential reversal up)
    OBV falling + price falling → volume confirms downtrend
    OBV crossing its EMA → trend change signal
"""

import numpy as np
import pandas as pd
from indicators.ema import calc_ema


def calc_obv(
    df: pd.DataFrame,
    ema_period: int = 21,
) -> dict:
    """
    OBV(i) = OBV(i-1) + volume[i]  if close[i] > close[i-1]
           = OBV(i-1) - volume[i]  if close[i] < close[i-1]
           = OBV(i-1)              if close[i] == close[i-1]

    Parameters
    ----------
    df         : DataFrame with 'Close', 'Volume'
    ema_period : Period for the OBV signal EMA (default 21; set 0 to skip)

    Returns
    -------
    dict with keys: obv, obv_ema
    """
    close  = df["Close"].values
    volume = df["Volume"].values.astype(float)
    n      = len(close)

    obv = np.zeros(n)
    obv[0] = volume[0]

    for i in range(1, n):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]

    obv_series = pd.Series(obv, index=df.index)
    obv_ema = calc_ema(obv_series, length=ema_period).values if ema_period > 0 else np.full(n, np.nan)

    return {
        "obv":     obv,
        "obv_ema": obv_ema,
    }
