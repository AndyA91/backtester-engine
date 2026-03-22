"""
Python translation of:
  Raff Regression Channel by DGT
  https://www.tradingview.com/script/1QVTzi78-Raff-Regression-Channel-by-DGT/

Computes a rolling Raff Regression Channel and optional Standard Deviation
Channel over a configurable lookback window, plus a rolling Linear Regression
Curve (ta.linreg equivalent).

Pine computes its channel only at `barstate.islast` over a fixed date range.
This Python version produces a per-bar rolling equivalent, which is more
useful for backtesting signal generation.

Output columns added to df
--------------------------
  rrc_lrc       Linear Regression Curve value (ta.linreg equivalent, rolling)
  rrc_mid_now   Regression value at the CURRENT bar (= rrc_lrc)
  rrc_upper     Raff upper channel  = reg_line + max_deviation
  rrc_lower     Raff lower channel  = reg_line - max_deviation
  rrc_std_upper Standard Deviation upper channel (optional)
  rrc_std_lower Standard Deviation lower channel (optional)
  rrc_slope     Slope of the regression line (positive = uptrend in time)

Pine-equivalent defaults
------------------------
  length        50    (LRC rolling length, Pine: i_linregCurveL)
  raff_length   100   (Raff channel window, Pine: i_depth used for Auto mode)
  std_mult      2.0   (Pine: i_stdev)

Usage
-----
  from indicators.dgtrd.raff_regression import raff_regression_channel

  df = raff_regression_channel(df, length=50)
  # rrc_upper / rrc_lower are the Raff channel at each bar
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _linreg_window(src_window: np.ndarray) -> tuple[float, float]:
    """
    Least-squares linear regression matching Pine's f_calcSlope.

    Pine convention: index 0 = most recent bar (x=1), index n-1 = oldest (x=n).
    Returns (intercept, slope) where intercept = value at the CURRENT bar.

    The sign of slope follows Pine: positive slope = price was higher in the
    past (downward trend in time), negative = upward trend.
    """
    n = len(src_window)
    sumX = 0.0; sumY = 0.0; sumXSqr = 0.0; sumXY = 0.0

    for i in range(n):
        val = src_window[i]          # src_window[0] = most recent
        per = float(i + 1)           # x=1 for most recent, x=n for oldest
        sumX    += per
        sumY    += val
        sumXSqr += per * per
        sumXY   += val * per

    denom = n * sumXSqr - sumX * sumX
    if denom == 0.0:
        return src_window[0], 0.0

    slope     = (n * sumXY - sumX * sumY) / denom
    intercept = sumY / n - slope * sumX / n + slope
    return intercept, slope


def _raff_dev(src_window: np.ndarray,
              high_window: np.ndarray,
              low_window: np.ndarray,
              intercept: float,
              slope: float) -> float:
    """
    Max absolute deviation of high/low from the regression line.
    Matches Pine's f_calcDev.

    Pine iterates from i=0 (current bar, val=intercept) to i=n-1 (oldest bar):
      val starts at intercept and increments by slope each step.
    """
    furthest = 0.0
    val = intercept
    n = len(src_window)

    for i in range(n):
        h = high_window[i]
        l = low_window[i]
        dev = max(abs(h - val), abs(val - l))
        if dev > furthest:
            furthest = dev
        val += slope          # moves regression to next (older) bar

    return furthest


def _std_dev(src_window: np.ndarray, intercept: float, slope: float) -> float:
    """
    Standard deviation of close from the regression line.
    Matches Pine's f_calcDev2 (ddof = n-1 denominator, uses close only).
    """
    n = len(src_window)
    acc = 0.0
    val = intercept

    for i in range(n):
        diff = src_window[i] - val
        acc += diff * diff
        val += slope

    return np.sqrt(acc / max(n - 1, 1))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def raff_regression_channel(
    df: pd.DataFrame,
    source_col: str  = "Close",
    length:     int  = 50,
    raff_length: int = 100,
    std_mult:   float = 2.0,
    compute_std: bool = True,
) -> pd.DataFrame:
    """
    Compute rolling Raff Regression Channel and Linear Regression Curve.

    Two separate window lengths are supported:
    - `length`      : rolling window for the LRC curve (Pine: i_linregCurveL)
    - `raff_length` : rolling window for the Raff channel (Pine: channel range)

    Parameters
    ----------
    df           DataFrame with columns High, Low, Close (or source_col).
    source_col   Price source for the regression (default 'Close').
    length       LRC rolling window in bars (default 50).
    raff_length  Raff channel rolling window in bars (default 100).
    std_mult     Standard deviation multiplier (Pine: i_stdev, default 2.0).
    compute_std  Whether to compute the std-dev channel (default True).

    Returns
    -------
    df copy with columns: rrc_lrc, rrc_upper, rrc_lower, rrc_slope,
    and optionally rrc_std_upper, rrc_std_lower.
    """
    src  = df[source_col].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low  = df["Low"].to_numpy(dtype=float)
    n    = len(df)

    lrc       = np.full(n, np.nan)
    rrc_upper = np.full(n, np.nan)
    rrc_lower = np.full(n, np.nan)
    rrc_slope = np.full(n, np.nan)
    std_upper = np.full(n, np.nan)
    std_lower = np.full(n, np.nan)

    # --- Rolling LRC (ta.linreg equivalent) --------------------------------
    for i in range(length - 1, n):
        # Pine: source[0..length-1] with [0] = most recent
        window = src[i - length + 1 : i + 1][::-1]   # reverse so [0]=most recent
        if np.any(np.isnan(window)):
            continue
        intercept, _ = _linreg_window(window)
        lrc[i] = intercept                              # value at current bar

    # --- Rolling Raff Channel ----------------------------------------------
    for i in range(raff_length - 1, n):
        # Windows: [0]=most recent, [n-1]=oldest
        src_w  = src [i - raff_length + 1 : i + 1][::-1]
        high_w = high[i - raff_length + 1 : i + 1][::-1]
        low_w  = low [i - raff_length + 1 : i + 1][::-1]

        if np.any(np.isnan(src_w)):
            continue

        intercept, slope = _linreg_window(src_w)
        furthest = _raff_dev(src_w, high_w, low_w, intercept, slope)

        rrc_upper[i] = intercept + furthest
        rrc_lower[i] = intercept - furthest
        # Convert Pine slope sign to standard "positive = uptrend" convention:
        # Pine x=1 is most recent, x=n is oldest → positive slope = older > newer = downtrend
        # Flip sign for intuitive interpretation.
        rrc_slope[i] = -slope

        if compute_std:
            sd = _std_dev(src_w, intercept, slope)
            std_upper[i] = intercept + sd * std_mult
            std_lower[i] = intercept - sd * std_mult

    df = df.copy()
    df["rrc_lrc"]   = lrc
    df["rrc_upper"] = rrc_upper
    df["rrc_lower"] = rrc_lower
    df["rrc_slope"] = rrc_slope

    if compute_std:
        df["rrc_std_upper"] = std_upper
        df["rrc_std_lower"] = std_lower

    return df
