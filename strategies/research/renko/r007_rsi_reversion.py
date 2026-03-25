"""
R007-Renko: RSI Mean-Reversion

Entry on RSI oversold/overbought reversals. Exit on RSI returning to neutral.
Renko bars already filter noise, so RSI extremes should mark genuine exhaustion.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from indicators.rsi import calc_rsi
from indicators.adx import calc_adx

DESCRIPTION = "RSI oversold/overbought reversal, ADX ranging gate, exit at neutral"

HYPOTHESIS = (
    "RSI mean-reversion is the opposite of our trend-following sweep. On Renko "
    "bars (noise-filtered), RSI extremes should mark genuine exhaustion points. "
    "ADX < threshold gates entries to ranging markets where mean-reversion works. "
    "This tests whether mean-reversion has edge on Renko alongside trend-following."
)

PARAM_GRID = {
    "rsi_period":    [7, 14],
    "os_level":      [25, 30],      # oversold
    "ob_level":      [70, 75],      # overbought
    "adx_ceiling":   [0, 25, 35],   # 0 = no ADX filter
    "cooldown":      [3, 6],
}


def generate_signals(df, rsi_period=14, os_level=30, ob_level=70, adx_ceiling=25, cooldown=6):
    close = df["Close"].values
    n = len(close)

    rsi = calc_rsi(df, period=rsi_period)["rsi"]
    rsi_shifted = np.empty(n)
    rsi_shifted[0] = np.nan
    rsi_shifted[1:] = rsi[:-1]

    adx_arr = None
    if adx_ceiling > 0:
        adx_result = calc_adx(df, di_period=14, adx_period=14)
        adx_arr = pd.Series(adx_result["adx"], index=df.index).shift(1).values

    le = np.zeros(n, dtype=bool)
    lx = np.zeros(n, dtype=bool)
    se = np.zeros(n, dtype=bool)
    sx = np.zeros(n, dtype=bool)

    last_trade_bar = -999_999
    warmup = max(rsi_period, 14) + 2

    for i in range(warmup, n):
        if np.isnan(close[i]) or np.isnan(rsi_shifted[i]):
            continue
        if i < 2 or np.isnan(rsi_shifted[i - 1]):
            continue

        r_now = rsi_shifted[i]
        r_prev = rsi_shifted[i - 1]

        # Cross up from oversold
        cross_up_os = r_prev <= os_level and r_now > os_level
        # Cross down from overbought
        cross_dn_ob = r_prev >= ob_level and r_now < ob_level

        # Exit at neutral zone
        lx[i] = r_now >= 50
        sx[i] = r_now <= 50

        # ADX ranging gate
        ranging = True
        if adx_ceiling > 0 and adx_arr is not None:
            if np.isnan(adx_arr[i]):
                ranging = False
            else:
                ranging = adx_arr[i] < adx_ceiling

        can_trade = (i - last_trade_bar) >= cooldown
        if can_trade and ranging:
            if cross_up_os:
                le[i] = True
                last_trade_bar = i
            elif cross_dn_ob:
                se[i] = True
                last_trade_bar = i

    df2 = df.copy()
    df2["long_entry"] = le
    df2["long_exit"] = lx | se
    df2["short_entry"] = se
    df2["short_exit"] = sx | le
    return df2
