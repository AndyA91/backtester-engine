"""
R005-Renko: MACD Trend + Momentum

MACD line crosses signal line while both are on the same side of zero.
Exit when MACD crosses back through signal.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from indicators.macd import calc_macd

DESCRIPTION = "MACD cross while same-side-of-zero, exit on recross"

HYPOTHESIS = (
    "MACD captures momentum shifts. Requiring both lines above zero for longs "
    "(below zero for shorts) ensures we only trade with established momentum. "
    "Renko filters noise, so MACD crosses should be more meaningful."
)

PARAM_GRID = {
    "fast":     [8, 12],
    "slow":     [21, 26],
    "signal":   [7, 9],
    "cooldown": [3, 6, 12],
}


def generate_signals(df, fast=12, slow=26, signal=9, cooldown=6):
    close = df["Close"].values
    n = len(close)

    df_lag = df.copy()
    df_lag["Close"] = df["Close"].shift(1)
    macd_result = calc_macd(df_lag, fast=fast, slow=slow, signal=signal)
    macd_line = macd_result["macd"]
    signal_line = macd_result["signal"]

    le = np.zeros(n, dtype=bool)
    lx = np.zeros(n, dtype=bool)
    se = np.zeros(n, dtype=bool)
    sx = np.zeros(n, dtype=bool)

    last_trade_bar = -999_999
    warmup = slow + signal + 2

    for i in range(warmup, n):
        if np.isnan(close[i]) or np.isnan(macd_line[i]) or np.isnan(signal_line[i]):
            continue
        if np.isnan(macd_line[i - 1]) or np.isnan(signal_line[i - 1]):
            continue

        macd_now = macd_line[i]
        macd_prev = macd_line[i - 1]
        sig_now = signal_line[i]
        sig_prev = signal_line[i - 1]

        cross_up = macd_now > sig_now and macd_prev <= sig_prev
        cross_down = macd_now < sig_now and macd_prev >= sig_prev

        both_above = macd_now > 0 and sig_now > 0
        both_below = macd_now < 0 and sig_now < 0

        lx[i] = cross_down
        sx[i] = cross_up

        can_trade = (i - last_trade_bar) >= cooldown
        if can_trade:
            if cross_up and both_above:
                le[i] = True
                last_trade_bar = i
            elif cross_down and both_below:
                se[i] = True
                last_trade_bar = i

    df2 = df.copy()
    df2["long_entry"] = le
    df2["long_exit"] = lx | se
    df2["short_entry"] = se
    df2["short_exit"] = sx | le
    return df2
