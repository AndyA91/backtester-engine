"""
ATR — Average True Range

Measures volatility as a smoothed average of the true range.
Used for dynamic SL/TP, position sizing, and Supertrend.

Usage:
    from indicators.atr import calc_atr
    result = calc_atr(df, period=14)
    # result["atr"]  — ATR values
    # result["tr"]   — True Range values
"""

import numpy as np
import pandas as pd


def calc_atr(
    df: pd.DataFrame,
    period: int = 14,
    method: str = "rma",
) -> dict:
    """
    Parameters
    ----------
    df : DataFrame with 'High', 'Low', 'Close'
    period : ATR lookback
    method : 'rma' (Wilder's, TradingView default), 'sma', or 'ema'
    """
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    n = len(close)

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
    )

    atr = np.zeros(n)

    if method == "rma":
        # Wilder's smoothing (RMA) — matches TradingView ta.atr()
        alpha = 1.0 / period
        atr[0] = tr[0]
        for i in range(1, n):
            if i < period:
                atr[i] = np.mean(tr[:i+1])
            else:
                atr[i] = atr[i-1] * (1 - alpha) + tr[i] * alpha
    elif method == "sma":
        for i in range(n):
            start = max(0, i - period + 1)
            atr[i] = np.mean(tr[start:i+1])
    elif method == "ema":
        alpha = 2.0 / (period + 1)
        atr[0] = tr[0]
        for i in range(1, n):
            atr[i] = atr[i-1] * (1 - alpha) + tr[i] * alpha

    return {"atr": atr, "tr": tr}
