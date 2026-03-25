"""
R002-Renko: EMA Crossover + ADX Trend Strength

Fast EMA crosses slow EMA on Renko bars, confirmed by ADX trend strength.
ADX was the #1 per-trade PnL predictor in prior analysis (r=+0.521).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from indicators.ema import calc_ema
from indicators.adx import calc_adx

DESCRIPTION = "Fast/slow EMA cross on Renko, filtered by ADX(14) strength"

HYPOTHESIS = (
    "EMA crosses whipsaw in ranging conditions. ADX(14) > threshold selects "
    "trending regimes where crosses have follow-through. Renko bars already "
    "filter noise, so the combination should yield cleaner signals."
)

PARAM_GRID = {
    "fast_ema":      [5, 9, 13],
    "slow_ema":      [21, 34, 55],
    "adx_threshold": [15, 20, 25, 30],
    "cooldown":      [3, 6],
}


def generate_signals(df, fast_ema=9, slow_ema=21, adx_threshold=25, cooldown=6):
    close = df["Close"].values
    n = len(close)

    close_s = df["Close"].shift(1)
    ema_fast = calc_ema(close_s, length=fast_ema).values
    ema_slow = calc_ema(close_s, length=slow_ema).values

    adx_result = calc_adx(df, di_period=14, adx_period=14)
    adx = pd.Series(adx_result["adx"], index=df.index).shift(1).values

    le = np.zeros(n, dtype=bool)
    lx = np.zeros(n, dtype=bool)
    se = np.zeros(n, dtype=bool)
    sx = np.zeros(n, dtype=bool)

    last_trade_bar = -999_999
    prev_fast_above = False
    warmup = slow_ema + 14 + 2

    for i in range(warmup, n):
        if np.isnan(close[i]) or np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or np.isnan(adx[i]):
            prev_fast_above = ema_fast[i] > ema_slow[i] if not np.isnan(ema_fast[i]) else prev_fast_above
            continue

        strong_trend = adx[i] > adx_threshold
        fast_above = ema_fast[i] > ema_slow[i]

        cross_up = fast_above and not prev_fast_above
        cross_down = not fast_above and prev_fast_above

        lx[i] = cross_down
        sx[i] = cross_up

        can_trade = (i - last_trade_bar) >= cooldown
        if can_trade and strong_trend:
            if cross_up:
                le[i] = True
                last_trade_bar = i
            elif cross_down:
                se[i] = True
                last_trade_bar = i

        prev_fast_above = fast_above

    df2 = df.copy()
    df2["long_entry"] = le
    df2["long_exit"] = lx | se
    df2["short_entry"] = se
    df2["short_exit"] = sx | le
    return df2
