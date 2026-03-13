"""
MFI — Money Flow Index

Volume-weighted RSI. Oscillator 0-100 measuring buying/selling pressure
by combining price direction and volume.
Matches Pine's ta.mfi(source, length).

Usage:
    from indicators.mfi import calc_mfi

    result = calc_mfi(df, period=14)
    # result["mfi"] — MFI values (0-100)

Interpretation:
    MFI > 80  → overbought (strong buying pressure, potential reversal down)
    MFI < 20  → oversold  (strong selling pressure, potential reversal up)
    Divergence (price rises, MFI falls) → weakening trend, high-probability reversal
    Unlike pure price oscillators, MFI requires volume — spots accumulation/distribution
"""

import numpy as np
import pandas as pd


def calc_mfi(
    df: pd.DataFrame,
    period: int = 14,
) -> dict:
    """
    Typical Price = (High + Low + Close) / 3
    Raw Money Flow = Typical Price * Volume

    Positive MF = sum(RMF where TP > TP_prev, period)
    Negative MF = sum(RMF where TP < TP_prev, period)
    MFR = Positive MF / Negative MF
    MFI = 100 - 100 / (1 + MFR)

    Parameters
    ----------
    df     : DataFrame with 'High', 'Low', 'Close', 'Volume'
    period : Lookback period (default 14)

    Returns
    -------
    dict with key: mfi (numpy array, 0-100)
    """
    high   = df["High"].values
    low    = df["Low"].values
    close  = df["Close"].values
    volume = df["Volume"].values.astype(float)
    n      = len(close)

    tp  = (high + low + close) / 3.0
    rmf = tp * volume

    # Direction: positive / negative money flow per bar
    pos_mf = np.where(tp > np.roll(tp, 1), rmf, 0.0)
    neg_mf = np.where(tp < np.roll(tp, 1), rmf, 0.0)
    # First bar comparison is invalid — zero it out
    pos_mf[0] = 0.0
    neg_mf[0] = 0.0

    pos_mf_s = pd.Series(pos_mf)
    neg_mf_s = pd.Series(neg_mf)

    pos_sum = pos_mf_s.rolling(period).sum().values
    neg_sum = neg_mf_s.rolling(period).sum().values

    with np.errstate(invalid="ignore", divide="ignore"):
        mfr = np.where(neg_sum > 0, pos_sum / neg_sum, 100.0)
        mfi = np.where(neg_sum > 0, 100.0 - 100.0 / (1.0 + mfr), 100.0)

    # NaN out warmup
    mfi[:period] = np.nan

    return {"mfi": mfi}
