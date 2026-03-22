"""
Python translation of:
  Trend Line by HPotter
  https://www.tradingview.com/script/LUKaSZlC-Trend-Line/

Computes a smoothed mid-price trend line and indicates whether it is
acting as support (price above) or resistance (price below).

Pine logic:
  Points = ta.sma(hl2[1], Length)          — SMA of *previous* bar's hl2
  is_support = close[1] > Points           — prev close vs trend line

Python equivalence note:
  ta.sma(hl2[1], Length) shifts hl2 by 1 bar then takes a rolling mean.
  Result: Points[i] = mean( (H[i-1]+L[i-1])/2, ..., (H[i-L]+L[i-L])/2 )
  close[1] in Pine = previous bar's close = close.shift(1) in Pandas.

Output columns added to df
--------------------------
  hp_trend_line  — SMA of shifted hl2 (the actual trend line price level)
  hp_is_support  — bool: prev close > trend line (line acts as support)

Usage
-----
  from indicators.HPotter.trend_line import calc_hp_trend_line
  df = calc_hp_trend_line(df, length=25)
  # df["hp_trend_line"]  — trend line value
  # df["hp_is_support"]  — True = support, False = resistance
"""

import numpy as np
import pandas as pd


def calc_hp_trend_line(df: pd.DataFrame, length: int = 25) -> pd.DataFrame:
    """
    Python translation of Trend Line by HPotter.

    Parameters
    ----------
    df     : DataFrame with columns High, Low, Close
    length : SMA lookback period (default 25, matches Pine default)

    Returns
    -------
    df with new columns hp_trend_line, hp_is_support appended.
    """
    hl2 = (df["High"] + df["Low"]) / 2.0

    # ta.sma(hl2[1], length) → SMA applied to the 1-bar-shifted hl2 series
    hl2_prev = hl2.shift(1)
    trend_line = hl2_prev.rolling(window=length, min_periods=length).mean()

    # close[1] in Pine is the previous bar's close
    prev_close = df["Close"].shift(1)
    is_support = prev_close > trend_line

    df = df.copy()
    df["hp_trend_line"] = trend_line
    df["hp_is_support"] = is_support
    return df
