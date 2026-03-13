"""
R005: MACD Trend + Momentum

MACD line crosses signal line while both are on the same side of zero.
Filtered by 1H KAMA slope for higher-timeframe trend alignment.
Exit when MACD crosses back through signal.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from indicators.macd import calc_macd
from indicators.kama import calc_kama

DESCRIPTION = "MACD cross above/below signal while same-side-of-zero, 1H KAMA slope filter, exit on recross"

HYPOTHESIS = (
    "MACD captures momentum shifts. Requiring both MACD and signal to be above zero for "
    "longs (and below zero for shorts) ensures we only trade with established momentum — "
    "avoiding counter-trend signals that cross zero frequently. 1H KAMA slope provides the "
    "trend direction anchor. Session filter keeps us in the liquid European/US overlap."
)

PARAM_GRID = {
    "fast":     [8, 12],
    "slow":     [21, 26],
    "signal":   [7, 9],
    "cooldown": [12, 24, 48],
}


def _align_1h_kama_slope(ltf_index: pd.DatetimeIndex, kama_1h: pd.Series) -> np.ndarray:
    htf = pd.DataFrame({
        "Date":  kama_1h.index,
        "slope": kama_1h.diff().shift(1).values,
    })
    ltf = pd.DataFrame({"Date": ltf_index})
    merged = pd.merge_asof(ltf.sort_values("Date"), htf.sort_values("Date"),
                           on="Date", direction="backward")
    return merged["slope"].values


def generate_signals(
    df_5m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_1d: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    cooldown: int = 24,
) -> pd.DataFrame:
    close = df_5m["Close"].values
    n = len(close)

    # MACD on shifted close to avoid lookahead
    df_5m_lag = df_5m.copy()
    df_5m_lag["Close"] = df_5m["Close"].shift(1)
    macd_result = calc_macd(df_5m_lag, fast=fast, slow=slow, signal=signal)
    macd_line   = macd_result["macd"]
    signal_line = macd_result["signal"]

    # 1H KAMA slope
    kama_1h  = calc_kama(df_1h["Close"], length=30, fast=2, slow=60)
    slope_1h = _align_1h_kama_slope(df_5m.index, kama_1h)

    hours = df_5m.index.hour.values

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    last_trade_bar = -999_999
    warmup = slow + signal + 2

    for i in range(warmup, n):
        if (np.isnan(close[i]) or np.isnan(macd_line[i]) or
                np.isnan(signal_line[i]) or np.isnan(macd_line[i - 1])):
            continue

        in_session = 7 <= hours[i] < 22
        slope_bull = slope_1h[i] > 0 if not np.isnan(slope_1h[i]) else False
        slope_bear = slope_1h[i] < 0 if not np.isnan(slope_1h[i]) else False

        macd_now  = macd_line[i]
        macd_prev = macd_line[i - 1]
        sig_now   = signal_line[i]
        sig_prev  = signal_line[i - 1]

        # Cross detection
        cross_up   = macd_now > sig_now   and macd_prev <= sig_prev
        cross_down = macd_now < sig_now   and macd_prev >= sig_prev

        # Both lines same side of zero
        both_above = macd_now > 0 and sig_now > 0
        both_below = macd_now < 0 and sig_now < 0

        # Exit: MACD crosses back through signal (no zero filter on exit)
        long_exit[i]  = cross_down
        short_exit[i] = cross_up

        can_trade = (i - last_trade_bar) >= cooldown
        if can_trade and in_session:
            if cross_up and both_above and slope_bull:
                long_entry[i]  = True
                last_trade_bar = i
            elif cross_down and both_below and slope_bear:
                short_entry[i] = True
                last_trade_bar = i

    df_5m["long_entry"]  = long_entry
    df_5m["long_exit"]   = long_exit
    df_5m["short_entry"] = short_entry
    df_5m["short_exit"]  = short_exit
    return df_5m
