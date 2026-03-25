"""
R004-Renko: Bollinger Band Squeeze + Breakout

Entry when price closes outside BB after a squeeze (narrow bandwidth).
Exit when price crosses back inside the mid-band.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from indicators.bbands import calc_bbands

DESCRIPTION = "BB breakout after squeeze (bw < threshold), exit at mid-band"

HYPOTHESIS = (
    "BB squeezes precede explosive moves as volatility cycles low-to-high. "
    "Renko bars already filter noise, so squeezes on Renko should indicate "
    "genuine consolidation. Breakouts after squeeze signal new directional moves."
)

PARAM_GRID = {
    "bb_period":   [14, 20],
    "bb_std":      [1.5, 2.0, 2.5],
    "squeeze_pct": [0.002, 0.003, 0.005],
    "cooldown":    [3, 6],
}


def generate_signals(df, bb_period=20, bb_std=2.0, squeeze_pct=0.003, cooldown=6):
    close = df["Close"].values
    n = len(close)

    df_lag = df.copy()
    df_lag["Close"] = df["Close"].shift(1)
    df_lag["High"] = df["High"].shift(1)
    df_lag["Low"] = df["Low"].shift(1)
    bb = calc_bbands(df_lag, period=bb_period, mult=bb_std)

    mid = bb["mid"]
    upper = bb["upper"]
    lower = bb["lower"]
    bw = bb["bw"]

    le = np.zeros(n, dtype=bool)
    lx = np.zeros(n, dtype=bool)
    se = np.zeros(n, dtype=bool)
    sx = np.zeros(n, dtype=bool)

    last_trade_bar = -999_999
    warmup = bb_period + 2

    was_squeezed = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not np.isnan(bw[i - 1]):
            was_squeezed[i] = bw[i - 1] < squeeze_pct

    for i in range(warmup, n):
        if np.isnan(close[i]) or np.isnan(mid[i]) or np.isnan(upper[i]):
            continue

        brk_up = close[i] > upper[i]
        brk_dn = close[i] < lower[i]

        long_cond = brk_up and was_squeezed[i]
        short_cond = brk_dn and was_squeezed[i]

        lx[i] = close[i] < mid[i]
        sx[i] = close[i] > mid[i]

        can_trade = (i - last_trade_bar) >= cooldown
        if can_trade:
            if long_cond:
                le[i] = True
                last_trade_bar = i
            elif short_cond:
                se[i] = True
                last_trade_bar = i

    df2 = df.copy()
    df2["long_entry"] = le
    df2["long_exit"] = lx | se
    df2["short_entry"] = se
    df2["short_exit"] = sx | le
    return df2
