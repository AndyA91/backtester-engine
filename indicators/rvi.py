"""
Relative Vigor Index (RVI)

Measures the conviction of a price move by comparing the close-open range
(body) to the high-low range (full bar). The idea: in uptrends, closes tend
to be near highs (strong vigor); in downtrends, closes tend to be near lows.

Uses a symmetric weighted moving average (1,2,2,1)/6 for smoothing, matching
the standard TradingView implementation.

RVI = SWM(Close - Open, period) / SWM(High - Low, period)
Signal = SWM(RVI, 4) using (1,2,2,1)/6 weighting

Usage:
    from indicators.rvi import calc_rvi

    result = calc_rvi(df, period=10)
    # result["rvi"]    — RVI line
    # result["signal"] — Signal line

Interpretation:
    RVI crosses above signal → bullish (closes gaining conviction)
    RVI crosses below signal → bearish (closes losing conviction)
    RVI > 0 → bulls in control (closes near highs)
    RVI < 0 → bears in control (closes near lows)
    Use alongside ADX: ADX says "trending", RVI says "with conviction"
"""

import numpy as np
import pandas as pd


def _swma4(src: np.ndarray) -> np.ndarray:
    """Symmetric weighted moving average with weights (1,2,2,1)/6 over 4 bars."""
    n = len(src)
    result = np.full(n, np.nan)
    for i in range(3, n):
        result[i] = (src[i - 3] + 2.0 * src[i - 2] + 2.0 * src[i - 1] + src[i]) / 6.0
    return result


def calc_rvi(
    df: pd.DataFrame,
    period: int = 10,
) -> dict:
    """
    Parameters
    ----------
    df     : DataFrame with 'Open', 'High', 'Low', 'Close'
    period : Lookback period for SMA smoothing (default 10)

    Returns
    -------
    dict with keys: rvi, signal (both numpy arrays)
    """
    open_ = df["Open"].values.astype(float)
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    close = df["Close"].values.astype(float)
    n = len(close)

    # Body and range
    body = close - open_
    hl_range = high - low

    # Apply symmetric weighted MA (1,2,2,1)/6
    body_swma = _swma4(body)
    range_swma = _swma4(hl_range)

    # SMA of the SWMA values
    body_sma = pd.Series(body_swma).rolling(period).sum().values
    range_sma = pd.Series(range_swma).rolling(period).sum().values

    # RVI = smoothed body / smoothed range
    rvi = np.full(n, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        mask = (range_sma != 0) & ~np.isnan(range_sma)
        rvi[mask] = body_sma[mask] / range_sma[mask]

    # Signal line — SWMA of RVI
    signal = _swma4(np.nan_to_num(rvi, nan=0.0))

    # NaN out warmup
    warmup = period + 3
    rvi[:warmup] = np.nan
    signal[:warmup] = np.nan

    return {
        "rvi": rvi,
        "signal": signal,
    }
