"""
R012-Renko: Regime-Gated Donchian (Choppiness / Squeeze Momentum)

R001 Donchian breakout gated by regime detection. Tests whether filtering
by Choppiness Index (trending vs choppy) or Squeeze Momentum (vol cycle)
improves signal quality.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from indicators.chop import calc_chop
from indicators.squeeze import calc_squeeze

DESCRIPTION = "Donchian breakout gated by Choppiness Index or Squeeze Momentum regime"

HYPOTHESIS = (
    "R001 Donchian (PF 3.8-8.1) trades all breakouts regardless of regime. "
    "Choppiness Index < threshold indicates trending regime (breakouts follow through). "
    "Squeeze Momentum release signals volatility expansion (breakouts have room to run). "
    "Either gate should improve PF by filtering breakouts in choppy/compressed markets."
)

PARAM_GRID = {
    "n_entry":        [20, 40],
    "n_exit":         [5, 10],
    "regime":         ["chop", "squeeze"],
    "chop_threshold": [50, 55, 61.8],  # below = trending (used with chop)
    "cooldown":       [6, 12],
}


def generate_signals(df, n_entry=40, n_exit=5, regime="chop",
                     chop_threshold=55, cooldown=6):
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    n = len(close)

    high_s = pd.Series(high)
    low_s = pd.Series(low)
    don_high_entry = high_s.shift(1).rolling(n_entry).max().values
    don_low_entry = low_s.shift(1).rolling(n_entry).min().values
    don_high_exit = high_s.shift(1).rolling(n_exit).max().values
    don_low_exit = low_s.shift(1).rolling(n_exit).min().values

    # Regime filter
    regime_ok = np.ones(n, dtype=bool)

    if regime == "chop":
        chop = calc_chop(df, period=14)["chop"]
        chop_s = np.empty(n)
        chop_s[0] = np.nan
        chop_s[1:] = chop[:-1]
        for i in range(n):
            if not np.isnan(chop_s[i]):
                regime_ok[i] = chop_s[i] < chop_threshold  # trending
            else:
                regime_ok[i] = False

    elif regime == "squeeze":
        sq = calc_squeeze(df)
        # Squeeze just released = volatility expansion starting
        sq_off = sq["squeeze_off"]
        momentum = sq["momentum"]
        # Allow trading when squeeze just released OR momentum is strong
        for i in range(1, n):
            if not np.isnan(momentum[i - 1]):
                regime_ok[i] = bool(sq_off[i - 1]) or abs(momentum[i - 1]) > 0
            else:
                regime_ok[i] = False

    le = np.zeros(n, dtype=bool)
    lx = np.zeros(n, dtype=bool)
    se = np.zeros(n, dtype=bool)
    sx = np.zeros(n, dtype=bool)

    last_trade_bar = -999_999
    warmup = max(n_entry, n_exit, 20) + 2

    for i in range(warmup, n):
        if np.isnan(close[i]) or np.isnan(don_high_entry[i]):
            continue

        brk_up = close[i] > don_high_entry[i]
        brk_dn = close[i] < don_low_entry[i]

        lx[i] = close[i] < don_low_exit[i]
        sx[i] = close[i] > don_high_exit[i]

        can_trade = (i - last_trade_bar) >= cooldown and regime_ok[i]
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
