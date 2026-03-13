"""
Moving Averages — SMA, EMA, WMA, HMA, DEMA, TEMA

Foundational building blocks used by MACD, Supertrend, and most strategies.
All implementations match TradingView's Pine Script ta.* functions exactly.

Usage:
    from indicators.ema import calc_ema, calc_sma, calc_wma, calc_hma, calc_dema, calc_tema

    ema9  = calc_ema(df["Close"], length=9)     # pd.Series
    sma20 = calc_sma(df["Close"], length=20)
    wma14 = calc_wma(df["Close"], length=14)
    hma55 = calc_hma(df["Close"], length=55)
    dema9 = calc_dema(df["Close"], length=9)
    tema9 = calc_tema(df["Close"], length=9)

Interpretation:
    EMA  — responds faster to recent prices than SMA (exponential weighting)
    SMA  — simple mean; slow but clean trend baseline
    WMA  — linear weighting (most recent = highest weight)
    HMA  — Hull: reduces lag while keeping smoothness; good for trend direction
    DEMA — Double EMA: further lag reduction via 2*EMA(src) - EMA(EMA(src))
    TEMA — Triple EMA: 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA)); very responsive
"""

import numpy as np
import pandas as pd


def calc_ema(series: pd.Series, length: int = 9) -> pd.Series:
    """
    Exponential Moving Average — matches Pine's ta.ema(src, length).

    alpha = 2 / (length + 1)
    EMA(i) = alpha * src(i) + (1 - alpha) * EMA(i-1)
    Seeds with the first non-NaN source value.

    Parameters
    ----------
    series : pd.Series  Price series.
    length : int        EMA period.

    Returns
    -------
    pd.Series with same index as input.
    """
    src = series.values.astype(float)
    n = len(src)
    alpha = 2.0 / (length + 1.0)

    ema = np.full(n, np.nan)

    # Seed: first non-NaN bar
    start = 0
    while start < n and np.isnan(src[start]):
        start += 1
    if start >= n:
        return pd.Series(ema, index=series.index, name="EMA")

    ema[start] = src[start]
    for i in range(start + 1, n):
        ema[i] = alpha * src[i] + (1.0 - alpha) * ema[i - 1]

    return pd.Series(ema, index=series.index, name=f"EMA_{length}")


def calc_sma(series: pd.Series, length: int = 20) -> pd.Series:
    """
    Simple Moving Average — matches Pine's ta.sma(src, length).

    Returns
    -------
    pd.Series with same index as input.
    """
    result = series.rolling(length).mean()
    result.name = f"SMA_{length}"
    return result


def calc_wma(series: pd.Series, length: int = 14) -> pd.Series:
    """
    Weighted Moving Average — matches Pine's ta.wma(src, length).

    Weights: 1, 2, ..., length (most recent bar gets highest weight).
    WMA = sum(src[i-j] * (length - j)) / sum(1..length)  for j in 0..length-1

    Returns
    -------
    pd.Series with same index as input.
    """
    src = series.values.astype(float)
    n = len(src)
    weights = np.arange(1, length + 1, dtype=float)
    denom = weights.sum()

    wma = np.full(n, np.nan)
    for i in range(length - 1, n):
        window = src[i - length + 1 : i + 1]
        if not np.any(np.isnan(window)):
            wma[i] = np.dot(window, weights) / denom

    return pd.Series(wma, index=series.index, name=f"WMA_{length}")


def calc_hma(series: pd.Series, length: int = 9) -> pd.Series:
    """
    Hull Moving Average — matches Pine's ta.hma(src, length).

    HMA = WMA(2 * WMA(src, length/2) - WMA(src, length), sqrt(length))

    Dramatically reduces lag while remaining smooth.

    Returns
    -------
    pd.Series with same index as input.
    """
    half = max(1, length // 2)
    sqrt_len = max(1, int(round(length ** 0.5)))

    wma_half = calc_wma(series, length=half)
    wma_full = calc_wma(series, length=length)

    raw = 2.0 * wma_half - wma_full
    hma = calc_wma(raw, length=sqrt_len)
    hma.name = f"HMA_{length}"
    return hma


def calc_dema(series: pd.Series, length: int = 9) -> pd.Series:
    """
    Double EMA — matches Pine's ta.dema(src, length).

    DEMA = 2 * EMA(src, length) - EMA(EMA(src, length), length)

    Reduces the lag of a standard EMA.

    Returns
    -------
    pd.Series with same index as input.
    """
    ema1 = calc_ema(series, length=length)
    ema2 = calc_ema(ema1, length=length)
    dema = 2.0 * ema1 - ema2
    dema.name = f"DEMA_{length}"
    return dema


def calc_tema(series: pd.Series, length: int = 9) -> pd.Series:
    """
    Triple EMA — matches Pine's ta.tema(src, length).

    TEMA = 3 * EMA1 - 3 * EMA(EMA1) + EMA(EMA(EMA1))

    Most responsive of the classic MAs; can be noisy on choppy markets.

    Returns
    -------
    pd.Series with same index as input.
    """
    ema1 = calc_ema(series, length=length)
    ema2 = calc_ema(ema1, length=length)
    ema3 = calc_ema(ema2, length=length)
    tema = 3.0 * ema1 - 3.0 * ema2 + ema3
    tema.name = f"TEMA_{length}"
    return tema
