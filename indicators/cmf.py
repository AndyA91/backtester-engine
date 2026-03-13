"""
CMF — Chaikin Money Flow

Volume-based oscillator measuring buying/selling pressure over a rolling window.
Developed by Marc Chaikin. Matches the standard CMF formula used on TradingView.

Usage:
    from indicators.cmf import calc_cmf

    result = calc_cmf(df, period=20)
    # result["cmf"] — CMF values (-1 to +1)

Interpretation:
    CMF > 0    → net buying pressure (accumulation)
    CMF < 0    → net selling pressure (distribution)
    CMF > +0.1 → moderately bullish
    CMF < -0.1 → moderately bearish
    CMF > +0.25 → strong accumulation
    CMF < -0.25 → strong distribution
    Divergence between CMF and price → potential reversal signal
"""

import numpy as np
import pandas as pd


def calc_cmf(
    df: pd.DataFrame,
    period: int = 20,
) -> dict:
    """
    Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
    Money Flow Volume     = MFM * Volume
    CMF = Sum(MFV, period) / Sum(Volume, period)

    Parameters
    ----------
    df     : DataFrame with 'High', 'Low', 'Close', 'Volume'
    period : Rolling window (default 20)

    Returns
    -------
    dict with key: cmf (numpy array, -1 to +1)
    """
    high   = df["High"].values
    low    = df["Low"].values
    close  = df["Close"].values
    volume = df["Volume"].values.astype(float)

    hl_range = high - low

    with np.errstate(invalid="ignore", divide="ignore"):
        mfm = np.where(hl_range > 0, ((close - low) - (high - close)) / hl_range, 0.0)

    mfv = mfm * volume

    mfv_s = pd.Series(mfv)
    vol_s = pd.Series(volume)

    mfv_sum = mfv_s.rolling(period).sum().values
    vol_sum = vol_s.rolling(period).sum().values

    with np.errstate(invalid="ignore", divide="ignore"):
        cmf = np.where(vol_sum > 0, mfv_sum / vol_sum, 0.0)

    # NaN out warmup
    cmf[:period - 1] = np.nan

    return {"cmf": cmf}
