"""
ADX — Average Directional Index

Measures trend strength (0–100). Does NOT indicate direction.
Also returns +DI / -DI for directional bias.

Matches Pine's ta.dmi(dilen, adxlen) exactly.
Wilder's RMA smoothing throughout (same as ta.atr).

Usage:
    from indicators.adx import calc_adx
    result = calc_adx(df, di_period=14, adx_period=14)
    # result["adx"]      — ADX line (trend strength, 0-100)
    # result["plus_di"]  — +DI line (bullish directional pressure)
    # result["minus_di"] — -DI line (bearish directional pressure)
    # result["dx"]       — raw DX before final smoothing

Interpretation:
    ADX < 20  → no trend / ranging market
    ADX 20-25 → trend developing
    ADX > 25  → trending (stronger as it rises)
    ADX > 40  → strong trend
    +DI > -DI → bullish bias
    +DI < -DI → bearish bias
"""

import numpy as np
import pandas as pd


def calc_adx(
    df: pd.DataFrame,
    di_period: int = 14,
    adx_period: int = 14,
) -> dict:
    """
    Parameters
    ----------
    df        : DataFrame with 'High', 'Low', 'Close'
    di_period : Smoothing period for +DI / -DI (Wilder's RMA)
    adx_period: Smoothing period for ADX line (Wilder's RMA applied to DX)
    """
    high  = df["High"].values
    low   = df["Low"].values
    close = df["Close"].values
    n = len(close)

    prev_close = np.roll(close, 1); prev_close[0] = close[0]
    prev_high  = np.roll(high,  1); prev_high[0]  = high[0]
    prev_low   = np.roll(low,   1); prev_low[0]   = low[0]

    # True Range (matches ta.tr)
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )

    # Directional Movement
    up   = high - prev_high        # upward price movement
    down = prev_low - low          # downward price movement (positive = price fell)
    plus_dm  = np.where((up > down) & (up > 0),   up,   0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    # Wilder's RMA smoothing — seeds with cumulative mean, then RMA
    # Matches Pine: ta.rma seeds with SMA of first `length` bars
    alpha_di  = 1.0 / di_period
    alpha_adx = 1.0 / adx_period

    rma_tr    = np.zeros(n)
    rma_plus  = np.zeros(n)
    rma_minus = np.zeros(n)

    for i in range(n):
        if i == 0:
            rma_tr[i]    = tr[i]
            rma_plus[i]  = plus_dm[i]
            rma_minus[i] = minus_dm[i]
        elif i < di_period:
            rma_tr[i]    = np.mean(tr[:i+1])
            rma_plus[i]  = np.mean(plus_dm[:i+1])
            rma_minus[i] = np.mean(minus_dm[:i+1])
        else:
            rma_tr[i]    = rma_tr[i-1]    * (1 - alpha_di) + tr[i]       * alpha_di
            rma_plus[i]  = rma_plus[i-1]  * (1 - alpha_di) + plus_dm[i]  * alpha_di
            rma_minus[i] = rma_minus[i-1] * (1 - alpha_di) + minus_dm[i] * alpha_di

    # +DI / -DI (0–100 scale)
    with np.errstate(invalid="ignore", divide="ignore"):
        plus_di  = np.where(rma_tr > 0, 100.0 * rma_plus  / rma_tr, 0.0)
        minus_di = np.where(rma_tr > 0, 100.0 * rma_minus / rma_tr, 0.0)

    # DX = 100 * |+DI - -DI| / (+DI + -DI)
    di_sum = plus_di + minus_di
    with np.errstate(invalid="ignore", divide="ignore"):
        dx = np.where(di_sum > 0, 100.0 * np.abs(plus_di - minus_di) / di_sum, 0.0)

    # ADX = RMA(DX, adx_period)
    adx = np.zeros(n)
    for i in range(n):
        if i == 0:
            adx[i] = dx[i]
        elif i < adx_period:
            adx[i] = np.mean(dx[:i+1])
        else:
            adx[i] = adx[i-1] * (1 - alpha_adx) + dx[i] * alpha_adx

    return {
        "adx":      adx,
        "plus_di":  plus_di,
        "minus_di": minus_di,
        "dx":       dx,
    }
