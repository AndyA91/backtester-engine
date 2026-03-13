"""
TTM Squeeze — Squeeze Momentum Indicator (LazyBear / John Carter)

One of TradingView's most-used community indicators. Identifies:
  1. SQUEEZE: Bollinger Bands are inside Keltner Channels → low volatility / coiling
  2. MOMENTUM: Linear regression of price displacement from range midpoint

When the squeeze fires (bands expand beyond KC), momentum determines direction.

Usage:
    from indicators.squeeze import calc_squeeze

    result = calc_squeeze(df)
    # result["momentum"]    — momentum histogram (positive = bullish, negative = bearish)
    # result["squeeze_on"]  — bool array: True when BB inside KC (squeeze active)
    # result["squeeze_off"] — bool array: True when squeeze just released (bars to trade)
    # result["no_squeeze"]  — bool array: True when BB fully outside KC (momentum only)

Interpretation:
    squeeze_on  → consolidation, prepare for breakout
    squeeze_off → squeeze released; trade in direction of momentum bar
    momentum bars changing from negative→positive while still red → weakening bears
    momentum bars turning green (positive) → bullish breakout
    momentum bars turning red  (negative) → bearish breakout
"""

import numpy as np
import pandas as pd
from indicators.atr import calc_atr


def calc_squeeze(
    df: pd.DataFrame,
    bb_period: int = 20,
    bb_mult: float = 2.0,
    kc_period: int = 20,
    kc_mult: float = 1.5,
    use_true_range: bool = True,
) -> dict:
    """
    Squeeze Momentum matching LazyBear's indicator (the standard TV version).

    Squeeze condition: BB upper < KC upper AND BB lower > KC lower

    Momentum = LinearRegression(close - avg(avg(highest_high, lowest_low), SMA(close)), 1)
    where avg is over kc_period

    Parameters
    ----------
    df             : DataFrame with 'High', 'Low', 'Close'
    bb_period      : Bollinger Band period (default 20)
    bb_mult        : BB standard deviation multiplier (default 2.0)
    kc_period      : Keltner Channel period (default 20)
    kc_mult        : KC ATR multiplier (default 1.5)
    use_true_range : Use True Range for KC (True matches LazyBear; False uses HL range)

    Returns
    -------
    dict with keys: momentum, squeeze_on, squeeze_off, no_squeeze
    """
    high  = df["High"].values
    low   = df["Low"].values
    close = df["Close"].values
    n     = len(close)

    close_s = pd.Series(close)
    high_s  = pd.Series(high)
    low_s   = pd.Series(low)

    # --- Bollinger Bands ---
    bb_sma = close_s.rolling(bb_period).mean().values
    bb_std = close_s.rolling(bb_period).std(ddof=0).values
    bb_upper = bb_sma + bb_mult * bb_std
    bb_lower = bb_sma - bb_mult * bb_std

    # --- Keltner Channels ---
    kc_sma = close_s.rolling(kc_period).mean().values

    if use_true_range:
        atr_result = calc_atr(df, period=kc_period, method="sma")
        kc_atr = atr_result["atr"]
    else:
        kc_atr = (high_s - low_s).rolling(kc_period).mean().values

    kc_upper = kc_sma + kc_mult * kc_atr
    kc_lower = kc_sma - kc_mult * kc_atr

    # --- Squeeze states ---
    squeeze_on  = (bb_upper < kc_upper) & (bb_lower > kc_lower)
    no_squeeze  = (bb_upper > kc_upper) & (bb_lower < kc_lower)
    # squeeze_off: previous bar was squeeze_on, current is not
    squeeze_off = np.roll(squeeze_on, 1) & ~squeeze_on
    squeeze_off[0] = False

    # --- Momentum ---
    # val = close - avg(avg(highest(high, period), lowest(low, period)), sma(close, period))
    hh   = high_s.rolling(kc_period).max().values
    ll   = low_s.rolling(kc_period).min().values
    delta = close - (hh + ll) / 2.0 - kc_sma   # displacement from range midpoint

    # Linear regression of delta over 1 bar (LazyBear uses linreg(delta, period, 0))
    # linreg(src, length, offset=0) = least-squares slope × offset + intercept
    # For period=kc_period, we fit a line to the last kc_period values of delta
    momentum = np.full(n, np.nan)
    for i in range(kc_period - 1, n):
        y = delta[i - kc_period + 1 : i + 1]
        if np.any(np.isnan(y)):
            continue
        x = np.arange(kc_period, dtype=float)
        xm = x.mean()
        ym = y.mean()
        denom = np.sum((x - xm) ** 2)
        if denom == 0:
            momentum[i] = ym
        else:
            slope = np.sum((x - xm) * (y - ym)) / denom
            momentum[i] = slope * (kc_period - 1) + (ym - slope * xm)

    return {
        "momentum":    momentum,
        "squeeze_on":  squeeze_on,
        "squeeze_off": squeeze_off,
        "no_squeeze":  no_squeeze,
    }
