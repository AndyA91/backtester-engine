"""
R006-Renko: Supertrend + ADX Regime Filter

Supertrend flip entries gated by ADX(14) trend strength.
adx_threshold=0 reproduces unfiltered R003 baseline.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from indicators.supertrend import calc_supertrend
from indicators.adx import calc_adx

DESCRIPTION = "Supertrend flip gated by ADX(14) regime filter (adx=0 = R003 baseline)"

HYPOTHESIS = (
    "R003 Supertrend flips produce many trades but low quality without a filter. "
    "ADX(14) was the #1 per-trade PnL predictor (r=+0.521). Gating Supertrend "
    "flips by ADX > threshold should strip out low-quality flips in choppy markets."
)

PARAM_GRID = {
    "atr_period":    [7, 10, 14],
    "multiplier":    [2.0, 3.0, 4.0],
    "adx_threshold": [0, 15, 20, 25],
    "cooldown":      [3, 6],
}


def generate_signals(df, atr_period=7, multiplier=4.0, adx_threshold=25, cooldown=6):
    close = df["Close"].values
    n = len(close)

    st_result = calc_supertrend(df, period=atr_period, multiplier=multiplier)
    direction = pd.Series(st_result["direction"]).shift(1).values

    adx_result = calc_adx(df, di_period=14, adx_period=14)
    adx = pd.Series(adx_result["adx"], index=df.index).shift(1).values

    le = np.zeros(n, dtype=bool)
    lx = np.zeros(n, dtype=bool)
    se = np.zeros(n, dtype=bool)
    sx = np.zeros(n, dtype=bool)

    last_trade_bar = -999_999
    warmup = max(atr_period, 14) + 2

    for i in range(warmup, n):
        if np.isnan(close[i]) or np.isnan(direction[i]):
            continue

        dir_now = direction[i]
        dir_prev = direction[i - 1]

        flip_bull = dir_now == 1 and dir_prev == -1
        flip_bear = dir_now == -1 and dir_prev == 1

        lx[i] = flip_bear
        sx[i] = flip_bull

        strong_trend = (adx_threshold == 0) or (not np.isnan(adx[i]) and adx[i] > adx_threshold)

        can_trade = (i - last_trade_bar) >= cooldown
        if can_trade and strong_trend:
            if flip_bull:
                le[i] = True
                last_trade_bar = i
            elif flip_bear:
                se[i] = True
                last_trade_bar = i

    df2 = df.copy()
    df2["long_entry"] = le
    df2["long_exit"] = lx | se
    df2["short_entry"] = se
    df2["short_exit"] = sx | le
    return df2
