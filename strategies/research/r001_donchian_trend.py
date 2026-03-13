"""
R001: Donchian Channel Breakout + 1H KAMA Slope Filter

Classic turtle-style trend following. Entry on N-bar high/low breakout,
filtered by 1H KAMA slope direction. Exit on M-bar opposite channel.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from indicators.kama import calc_kama

DESCRIPTION = "Donchian breakout (N-bar high/low) filtered by 1H KAMA slope, exit on M-bar opposite channel"

HYPOTHESIS = (
    "2024 was a trending year for EURUSD (USD strength). Donchian breakouts are a proven "
    "trend-following entry method. Filtering by 1H KAMA slope ensures we trade with the "
    "higher-timeframe trend rather than against it. Exit on opposite channel locks in profits "
    "while staying in strong trends. Session filter avoids illiquid hours."
)

PARAM_GRID = {
    "n_entry":  [20, 40, 60, 80],     # breakout lookback (5m bars)
    "n_exit":   [10, 20, 40],          # trailing exit channel (5m bars)
    "cooldown": [12, 24, 48],          # bars between trades (12=1h, 24=2h, 48=4h on 5m)
    "session":  [True],                # 07:00-22:00 UTC only
}


def _align_1h_slope(ltf_index: pd.DatetimeIndex, kama_1h: pd.Series) -> np.ndarray:
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
    n_entry: int = 40,
    n_exit:  int = 20,
    cooldown: int = 24,
    session: bool = True,
) -> pd.DataFrame:
    high  = df_5m["High"].values
    low   = df_5m["Low"].values
    close = df_5m["Close"].values
    n     = len(close)

    # Donchian channels — shift(1) so current bar doesn't see itself
    high_s = pd.Series(high)
    low_s  = pd.Series(low)
    don_high_entry = high_s.shift(1).rolling(n_entry).max().values
    don_low_entry  = low_s.shift(1).rolling(n_entry).min().values
    don_high_exit  = high_s.shift(1).rolling(n_exit).max().values
    don_low_exit   = low_s.shift(1).rolling(n_exit).min().values

    # 1H KAMA slope (aligned to 5m bars)
    kama_1h   = calc_kama(df_1h["Close"], length=30, fast=2, slow=60)
    slope_1h  = _align_1h_slope(df_5m.index, kama_1h)

    hours = df_5m.index.hour.values

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    last_trade_bar = -999_999

    for i in range(max(n_entry, n_exit) + 1, n):
        if np.isnan(close[i]) or np.isnan(don_high_entry[i]) or np.isnan(slope_1h[i]):
            continue

        in_session = (not session) or (7 <= hours[i] < 22)
        slope_bull = slope_1h[i] > 0
        slope_bear = slope_1h[i] < 0

        # Entry: close breaks above/below N-bar Donchian channel
        brk_up = close[i] > don_high_entry[i]
        brk_dn = close[i] < don_low_entry[i]

        long_cond  = brk_up and slope_bull and in_session
        short_cond = brk_dn and slope_bear and in_session

        # Exit: price touches opposite exit channel
        long_exit[i]  = close[i] < don_low_exit[i]
        short_exit[i] = close[i] > don_high_exit[i]

        can_trade = (i - last_trade_bar) >= cooldown
        if can_trade and long_cond:
            long_entry[i]  = True
            last_trade_bar = i
        elif can_trade and short_cond:
            short_entry[i] = True
            last_trade_bar = i

    df_5m["long_entry"]  = long_entry
    df_5m["long_exit"]   = long_exit
    df_5m["short_entry"] = short_entry
    df_5m["short_exit"]  = short_exit
    return df_5m
