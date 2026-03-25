"""
R001-Renko: Donchian Channel Breakout (no MTF, no session filter)

Classic turtle-style trend following on Renko bars.
Entry on N-bar high/low breakout, exit on M-bar opposite channel.
"""

import numpy as np
import pandas as pd

DESCRIPTION = "Donchian breakout (N-bar high/low), exit on M-bar opposite channel"

HYPOTHESIS = (
    "Donchian breakouts are a proven trend-following entry on time-based charts. "
    "On Renko bars (which already filter noise), breakouts should capture strong "
    "directional moves with less whipsaw than on raw candles."
)

PARAM_GRID = {
    "n_entry":  [10, 20, 40, 60],
    "n_exit":   [5, 10, 20],
    "cooldown": [3, 6, 12],
}


def generate_signals(df, n_entry=40, n_exit=20, cooldown=6):
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    n = len(close)

    high_s = pd.Series(high)
    low_s = pd.Series(low)
    don_high_entry = high_s.shift(1).rolling(n_entry).max().values
    don_low_entry = low_s.shift(1).rolling(n_entry).min().values
    don_high_exit = high_s.shift(1).rolling(n_exit).max().values
    don_low_exit = low_s.shift(1).rolling(n_exit).min().values

    le = np.zeros(n, dtype=bool)
    lx = np.zeros(n, dtype=bool)
    se = np.zeros(n, dtype=bool)
    sx = np.zeros(n, dtype=bool)

    last_trade_bar = -999_999

    for i in range(max(n_entry, n_exit) + 1, n):
        if np.isnan(close[i]) or np.isnan(don_high_entry[i]):
            continue

        brk_up = close[i] > don_high_entry[i]
        brk_dn = close[i] < don_low_entry[i]

        lx[i] = close[i] < don_low_exit[i]
        sx[i] = close[i] > don_high_exit[i]

        can_trade = (i - last_trade_bar) >= cooldown
        if can_trade:
            if brk_up:
                le[i] = True
                last_trade_bar = i
            elif brk_dn:
                se[i] = True
                last_trade_bar = i

    df2 = df.copy()
    df2["long_entry"] = le
    df2["long_exit"] = lx | se
    df2["short_entry"] = se
    df2["short_exit"] = sx | le
    return df2
