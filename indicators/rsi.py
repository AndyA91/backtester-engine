"""
RSI — Relative Strength Index

Momentum oscillator (0-100). Overbought > 70, oversold < 30.
Uses Wilder's smoothing (RMA) to match TradingView.

Usage:
    from indicators.rsi import calc_rsi
    result = calc_rsi(df, period=14)
    # result["rsi"]  — RSI values (0-100)
"""

import numpy as np
import pandas as pd


def calc_rsi(
    df: pd.DataFrame,
    period: int = 14,
    source: str = "close",
) -> dict:
    """
    Parameters
    ----------
    df : DataFrame with 'Close' (or specified source column)
    period : RSI lookback period
    source : 'close', 'hlc3', etc.
    """
    if source == "hlc3":
        src = (df["High"].values + df["Low"].values + df["Close"].values) / 3.0
    else:
        src = df["Close"].values

    n = len(src)
    rsi = np.full(n, 50.0)

    # Price changes
    delta = np.diff(src, prepend=src[0])

    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    # Wilder's smoothing (RMA)
    alpha = 1.0 / period
    avg_gain = 0.0
    avg_loss = 0.0

    for i in range(1, n):
        if i <= period:
            avg_gain = np.mean(gains[1:i+1])
            avg_loss = np.mean(losses[1:i+1])
        else:
            avg_gain = avg_gain * (1 - alpha) + gains[i] * alpha
            avg_loss = avg_loss * (1 - alpha) + losses[i] * alpha

        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return {"rsi": rsi}
