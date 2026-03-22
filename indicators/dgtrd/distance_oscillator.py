"""
Python translation of:
  Distance Oscillator - Support and Resistance by DGT
  https://www.tradingview.com/script/lQU45Wmv-Distance-Oscillator-Support-and-Resistance-by-DGT/

Computes the Distance Oscillator (price % deviation from its moving average),
applies a Bollinger Band envelope to detect overbought/oversold extremes, and
tracks dynamic support/resistance lines at those extremes.

Output columns added to df
--------------------------
  do_value      Distance oscillator value  = (source / SMA(source, n) - 1)
                optionally EMA-smoothed (Pine: smooth=True, smooth_length=5)
  do_upper      Upper Bollinger Band on the oscillator
  do_lower      Lower Bollinger Band on the oscillator
  do_overbought bool — do_value > do_upper
  do_oversold   bool — do_value < do_lower
  do_sr_high    Highest high since current overbought run started (S/R line)
  do_sr_low     Lowest  low  since current oversold  run started (S/R line)

Pine-equivalent defaults
------------------------
  source        close
  ma_length     21    (Oscillator Length, Pine: i_maLength)
  smooth        True  (EMA smoothing toggle, Pine: i_smooth)
  smooth_length 5     (Pine: inline "pao" second input)
  bb_length     233   (Pine: Threshold Band Length)
  bb_mult       2.5   (Pine: Multiplier)

Usage
-----
  from indicators.dgtrd.distance_oscillator import distance_oscillator_sr

  df = distance_oscillator_sr(df, ma_length=21, bb_length=233, bb_mult=2.5)
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sma(series: np.ndarray, length: int) -> np.ndarray:
    out = np.full(len(series), np.nan)
    for i in range(length - 1, len(series)):
        out[i] = np.mean(series[i - length + 1 : i + 1])
    return out


def _ema(series: np.ndarray, length: int) -> np.ndarray:
    """EMA matching TradingView ta.ema() — seed = SMA of first `length` values."""
    out = np.full(len(series), np.nan)
    mult = 2.0 / (length + 1)
    valid = ~np.isnan(series)

    # find seed window
    count = 0
    start = -1
    for i in range(len(series)):
        if valid[i]:
            count += 1
            if count == length:
                start = i - length + 1
                break
        else:
            count = 0
    if start < 0:
        return out

    seed_idx = start + length - 1
    out[seed_idx] = np.mean(series[start : seed_idx + 1])
    for i in range(seed_idx + 1, len(series)):
        if np.isnan(series[i]):
            out[i] = out[i - 1]
        else:
            out[i] = series[i] * mult + out[i - 1] * (1 - mult)
    return out


def _bollinger(series: np.ndarray, length: int, mult: float):
    """Returns (mid, upper, lower) Bollinger Bands matching Pine's bb()."""
    mid   = _sma(series, length)
    upper = np.full(len(series), np.nan)
    lower = np.full(len(series), np.nan)

    for i in range(length - 1, len(series)):
        window = series[i - length + 1 : i + 1]
        if np.any(np.isnan(window)):
            continue
        std = np.std(window, ddof=1)   # Pine ta.stdev uses ddof=1
        upper[i] = mid[i] + mult * std
        lower[i] = mid[i] - mult * std

    return mid, upper, lower


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def distance_oscillator_sr(
    df: pd.DataFrame,
    source_col: str  = "Close",
    ma_length:   int   = 21,
    smooth:      bool  = True,
    smooth_length: int = 5,
    bb_length:   int   = 233,
    bb_mult:     float = 2.5,
) -> pd.DataFrame:
    """
    Compute Distance Oscillator S&R levels for every bar in *df*.

    Parameters
    ----------
    df            DataFrame with at least High, Low, and source_col columns.
    source_col    Price source (default 'Close').
    ma_length     SMA length for the base oscillator (Pine: i_maLength, default 21).
    smooth        Apply EMA smoothing to the raw oscillator (Pine: i_smooth, default True).
    smooth_length EMA length when smooth=True (Pine default 5).
    bb_length     Bollinger Band length for threshold bands (Pine default 233).
    bb_mult       Bollinger Band standard deviation multiplier (Pine default 2.5).

    Returns
    -------
    df copy with columns: do_value, do_upper, do_lower,
    do_overbought, do_oversold, do_sr_high, do_sr_low.
    """
    src  = df[source_col].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low  = df["Low"].to_numpy(dtype=float)
    n    = len(df)

    # --- Distance oscillator (Pine: pma) -----------------------------------
    sma_base = _sma(src, ma_length)
    raw_osc  = (src / sma_base - 1.0)                    # (source/SMA - 1)

    pma = _ema(raw_osc, smooth_length) if smooth else raw_osc

    # --- Bollinger Band thresholds -----------------------------------------
    _, upper, lower = _bollinger(pma, bb_length, bb_mult)

    overbought = pma > upper
    oversold   = pma < lower

    # --- S/R line tracking -------------------------------------------------
    # Pine: when overbought, track the running high; on new overbought start,
    #       reset to current high. Same logic for oversold/low.
    sr_high = np.full(n, np.nan)
    sr_low  = np.full(n, np.nan)
    cur_high = np.nan
    cur_low  = np.nan

    for i in range(n):
        ob = bool(overbought[i]) if not np.isnan(pma[i]) else False
        os = bool(oversold[i])   if not np.isnan(pma[i]) else False

        prev_ob = bool(overbought[i - 1]) if i > 0 and not np.isnan(pma[i - 1]) else False
        prev_os = bool(oversold[i - 1])   if i > 0 and not np.isnan(pma[i - 1]) else False

        if ob:
            if not prev_ob:         # new overbought run starts
                cur_high = high[i]
            elif high[i] > cur_high:
                cur_high = high[i]
            sr_high[i] = cur_high
        else:
            cur_high = np.nan

        if os:
            if not prev_os:         # new oversold run starts
                cur_low = low[i]
            elif low[i] < cur_low:
                cur_low = low[i]
            sr_low[i] = cur_low
        else:
            cur_low = np.nan

    df = df.copy()
    df["do_value"]      = pma
    df["do_upper"]      = upper
    df["do_lower"]      = lower
    df["do_overbought"] = overbought
    df["do_oversold"]   = oversold
    df["do_sr_high"]    = sr_high
    df["do_sr_low"]     = sr_low

    return df
