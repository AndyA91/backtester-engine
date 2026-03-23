"""
R003-Renko: Supertrend Trend Follow

Entry when Renko Supertrend flips bullish/bearish.
Exit when Supertrend flips against position.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from indicators.supertrend import calc_supertrend

DESCRIPTION = "Supertrend flip entries on Renko, exit on flip"

HYPOTHESIS = (
    "Supertrend self-adapts its trailing stop via ATR, staying in trends longer "
    "than fixed channels. On Renko bars (which already filter noise), Supertrend "
    "flips should be higher quality than on time-based candles."
)

PARAM_GRID = {
    "atr_period":  [7, 10, 14],
    "multiplier":  [2.0, 3.0, 4.0],
    "cooldown":    [3, 6, 12],
}


def generate_signals(df, atr_period=10, multiplier=3.0, cooldown=6):
    close = df["Close"].values
    n = len(close)

    st_result = calc_supertrend(df, period=atr_period, multiplier=multiplier)
    direction = pd.Series(st_result["direction"]).shift(1).values

    le = np.zeros(n, dtype=bool)
    lx = np.zeros(n, dtype=bool)
    se = np.zeros(n, dtype=bool)
    sx = np.zeros(n, dtype=bool)

    last_trade_bar = -999_999
    warmup = atr_period + 2

    for i in range(warmup, n):
        if np.isnan(close[i]) or np.isnan(direction[i]):
            continue

        dir_now = direction[i]
        dir_prev = direction[i - 1]

        flip_bull = dir_now == 1 and dir_prev == -1
        flip_bear = dir_now == -1 and dir_prev == 1

        lx[i] = flip_bear
        sx[i] = flip_bull

        can_trade = (i - last_trade_bar) >= cooldown
        if can_trade:
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
