"""
Volume-Weighted MACD (VWMACD)

Replaces the standard EMA with Volume-Weighted Moving Average (VWMA) in the
MACD calculation. This filters out low-volume noise bars that create false
crossovers in standard MACD.

VWMA(src, period) = sum(src * volume, period) / sum(volume, period)
VWMACD = VWMA(close, fast) - VWMA(close, slow)
Signal = EMA(VWMACD, signal_period)
Histogram = VWMACD - Signal

Usage:
    from indicators.vwmacd import calc_vwmacd

    result = calc_vwmacd(df, fast=12, slow=26, signal=9)
    # result["vwmacd"]    — VWMACD line
    # result["signal"]    — Signal line (EMA of VWMACD)
    # result["histogram"] — VWMACD - Signal

Interpretation:
    Same as standard MACD, but signals are volume-confirmed:
    VWMACD crosses above signal → bullish (volume-confirmed momentum)
    VWMACD crosses below signal → bearish
    Fewer false signals during low-volume chop vs standard MACD
    Divergences between price and VWMACD → strong reversal signals
"""

import numpy as np
import pandas as pd
from indicators.ema import calc_ema


def _calc_vwma(src: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
    """Volume-Weighted Moving Average — matches Pine's ta.vwma(src, length)."""
    n = len(src)
    vwma = np.full(n, np.nan)

    src_vol = pd.Series(src * volume)
    vol_sum = pd.Series(volume)

    numerator = src_vol.rolling(period).sum().values
    denominator = vol_sum.rolling(period).sum().values

    with np.errstate(invalid="ignore", divide="ignore"):
        mask = denominator > 0
        vwma[mask] = numerator[mask] / denominator[mask]

    return vwma


def calc_vwmacd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict:
    """
    Parameters
    ----------
    df     : DataFrame with 'Close', 'Volume'
    fast   : Fast VWMA period (default 12)
    slow   : Slow VWMA period (default 26)
    signal : Signal line EMA period (default 9)

    Returns
    -------
    dict with keys: vwmacd, signal, histogram (all numpy arrays)
    """
    close = df["Close"].values.astype(float)
    volume = df["Volume"].values.astype(float)

    vwma_fast = _calc_vwma(close, volume, fast)
    vwma_slow = _calc_vwma(close, volume, slow)

    vwmacd_line = vwma_fast - vwma_slow

    signal_line = calc_ema(pd.Series(vwmacd_line), length=signal).values
    histogram = vwmacd_line - signal_line

    return {
        "vwmacd": vwmacd_line,
        "signal": signal_line,
        "histogram": histogram,
    }
