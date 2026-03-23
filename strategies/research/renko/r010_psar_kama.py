"""
R010-Renko: Parabolic SAR + KAMA Adaptive Trend

PSAR direction flip as entry, confirmed by KAMA slope direction.
PSAR provides adaptive trailing stop; KAMA adapts smoothing to volatility.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from indicators.parabolic_sar import calc_psar
from indicators.kama import calc_kama

DESCRIPTION = "Parabolic SAR flip confirmed by KAMA slope direction"

HYPOTHESIS = (
    "PSAR is a natural trailing stop that accelerates as trend strengthens. "
    "On Renko (already noise-filtered), PSAR flips should be higher quality. "
    "KAMA adapts its smoothing speed to market efficiency — fast in trends, "
    "slow in chop. Requiring KAMA slope agreement filters PSAR flips in noise."
)

PARAM_GRID = {
    "sar_start":    [0.02],
    "sar_increment": [0.02],
    "sar_maximum":  [0.2, 0.3],
    "kama_length":  [10, 20],
    "kama_fast":    [2],
    "kama_slow":    [30, 60],
    "require_kama": [True, False],
    "cooldown":     [3, 6],
}


def generate_signals(df, sar_start=0.02, sar_increment=0.02, sar_maximum=0.2,
                     kama_length=10, kama_fast=2, kama_slow=30,
                     require_kama=True, cooldown=6):
    close = df["Close"].values
    n = len(close)

    psar = calc_psar(df, start=sar_start, increment=sar_increment, maximum=sar_maximum)
    direction = pd.Series(psar["direction"]).shift(1).values

    kama_vals = calc_kama(df["Close"], length=kama_length, fast=kama_fast, slow=kama_slow)
    kama_slope = kama_vals.diff().shift(1).values

    le = np.zeros(n, dtype=bool)
    lx = np.zeros(n, dtype=bool)
    se = np.zeros(n, dtype=bool)
    sx = np.zeros(n, dtype=bool)

    last_trade_bar = -999_999
    warmup = max(kama_length, 10) + 3

    for i in range(warmup, n):
        if np.isnan(close[i]) or np.isnan(direction[i]):
            continue
        if i < 2 or np.isnan(direction[i - 1]):
            continue

        dir_now = direction[i]
        dir_prev = direction[i - 1]

        flip_bull = dir_now == 1 and dir_prev == -1
        flip_bear = dir_now == -1 and dir_prev == 1

        # KAMA slope confirmation
        k_bull = (not require_kama) or (not np.isnan(kama_slope[i]) and kama_slope[i] > 0)
        k_bear = (not require_kama) or (not np.isnan(kama_slope[i]) and kama_slope[i] < 0)

        lx[i] = flip_bear
        sx[i] = flip_bull

        can_trade = (i - last_trade_bar) >= cooldown
        if can_trade:
            if flip_bull and k_bull:
                le[i] = True
                last_trade_bar = i
            elif flip_bear and k_bear:
                se[i] = True
                last_trade_bar = i

    df2 = df.copy()
    df2["long_entry"] = le
    df2["long_exit"] = lx | se
    df2["short_entry"] = se
    df2["short_exit"] = sx | le
    return df2
