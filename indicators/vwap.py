"""
VWAP — Volume Weighted Average Price (Session-Anchored)

Resets at the start of each trading day.
VWAP = Σ(typical_price × volume) / Σ(volume)
Includes optional standard deviation bands.

Usage:
    from indicators.vwap import calc_vwap

    result = calc_vwap(df)
    # result["vwap"]   — VWAP line
    # result["upper1"] — +1σ band
    # result["lower1"] — -1σ band
    # result["upper2"] — +2σ band
    # result["lower2"] — -2σ band
    # result["stdev"]  — rolling session stdev

    # Or add columns directly:
    from indicators.vwap import add_vwap_columns
    df = add_vwap_columns(df)
    # Adds: vwap, vwap_upper1, vwap_lower1, vwap_upper2, vwap_lower2, vwap_stdev
"""

import numpy as np
import pandas as pd


def calc_vwap(
    df: pd.DataFrame,
    source: str = "hlc3",
    band_mult_1: float = 1.0,
    band_mult_2: float = 2.0,
) -> dict:
    """
    Session-anchored VWAP matching TradingView's built-in.

    Parameters
    ----------
    df : DataFrame
        Must contain 'High', 'Low', 'Close', 'Volume'. Index must be DatetimeIndex.
    source : str
        Price source: 'hlc3' (default), 'close', 'hl2', 'ohlc4'.
    band_mult_1 : float
        Inner band multiplier (default: 1.0 = ±1σ).
    band_mult_2 : float
        Outer band multiplier (default: 2.0 = ±2σ).

    Returns
    -------
    dict with keys: vwap, upper1, lower1, upper2, lower2, stdev
    """
    # Price source
    if source == "hlc3":
        src = (df["High"].values + df["Low"].values + df["Close"].values) / 3.0
    elif source == "hl2":
        src = (df["High"].values + df["Low"].values) / 2.0
    elif source == "ohlc4":
        src = (df["Open"].values + df["High"].values + df["Low"].values + df["Close"].values) / 4.0
    else:
        src = df["Close"].values

    volume = df["Volume"].values.astype(float)
    dates = df.index
    n = len(src)

    # Detect session boundaries (new day)
    day = pd.Series(dates).dt.date.values

    vwap = np.zeros(n)
    stdev_arr = np.zeros(n)

    cum_pv = 0.0
    cum_vol = 0.0
    cum_pv2 = 0.0

    for i in range(n):
        # Reset on new session
        if i == 0 or day[i] != day[i - 1]:
            cum_pv = 0.0
            cum_vol = 0.0
            cum_pv2 = 0.0

        cum_pv += src[i] * volume[i]
        cum_vol += volume[i]
        cum_pv2 += src[i] * src[i] * volume[i]

        if cum_vol > 0:
            vwap[i] = cum_pv / cum_vol
            variance = (cum_pv2 / cum_vol) - (vwap[i] ** 2)
            stdev_arr[i] = np.sqrt(max(variance, 0.0))
        else:
            vwap[i] = src[i]
            stdev_arr[i] = 0.0

    return {
        "vwap": vwap,
        "upper1": vwap + stdev_arr * band_mult_1,
        "lower1": vwap - stdev_arr * band_mult_1,
        "upper2": vwap + stdev_arr * band_mult_2,
        "lower2": vwap - stdev_arr * band_mult_2,
        "stdev": stdev_arr,
    }


def add_vwap_columns(
    df: pd.DataFrame, **kwargs
) -> pd.DataFrame:
    """
    Convenience wrapper: computes VWAP and adds columns to df.

    Adds: vwap, vwap_upper1, vwap_lower1, vwap_upper2, vwap_lower2, vwap_stdev

    All kwargs are forwarded to calc_vwap().
    Returns the modified DataFrame (copy).
    """
    df = df.copy()
    result = calc_vwap(df, **kwargs)
    df["vwap"] = result["vwap"]
    df["vwap_upper1"] = result["upper1"]
    df["vwap_lower1"] = result["lower1"]
    df["vwap_upper2"] = result["upper2"]
    df["vwap_lower2"] = result["lower2"]
    df["vwap_stdev"] = result["stdev"]
    return df
