"""
CCI — Commodity Channel Index

Oscillator measuring how far price deviates from its average relative to
its typical volatility. Originally for commodities, widely used on FX/equities.
Matches Pine's ta.cci(source, length).

Usage:
    from indicators.cci import calc_cci

    result = calc_cci(df, period=20)
    # result["cci"] — CCI values (centred at 0, unbounded)

Interpretation:
    CCI > +100  → overbought / strong upward trend momentum
    CCI < -100  → oversold  / strong downward trend momentum
    CCI crossing above  0 → bullish signal
    CCI crossing below  0 → bearish signal
    Divergence between CCI and price → potential reversal
    Typical threshold for trend signals: ±100
"""

import numpy as np
import pandas as pd


def calc_cci(
    df: pd.DataFrame,
    period: int = 20,
    constant: float = 0.015,
) -> dict:
    """
    CCI = (TP - SMA(TP, period)) / (constant * MeanDev(TP, period))

    where TP = (High + Low + Close) / 3

    Parameters
    ----------
    df       : DataFrame with 'High', 'Low', 'Close'
    period   : Lookback period (default 20)
    constant : Scaling constant (default 0.015, the original Lambert value)

    Returns
    -------
    dict with key: cci (numpy array)
    """
    high  = df["High"].values
    low   = df["Low"].values
    close = df["Close"].values
    n     = len(close)

    tp  = (high + low + close) / 3.0
    tp_s = pd.Series(tp)

    sma_tp = tp_s.rolling(period).mean().values

    # Mean deviation — Pine uses mean absolute deviation (not std)
    mean_dev = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = tp[i - period + 1 : i + 1]
        mean_dev[i] = np.mean(np.abs(window - sma_tp[i]))

    with np.errstate(invalid="ignore", divide="ignore"):
        cci = np.where(mean_dev > 0, (tp - sma_tp) / (constant * mean_dev), 0.0)

    return {"cci": cci}
