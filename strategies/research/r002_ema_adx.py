"""
R002: EMA Crossover + ADX Trend Strength

Fast EMA crosses slow EMA on 5m, confirmed by ADX trend strength.
1H EMA slope used as higher-timeframe directional filter.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from indicators.ema import calc_ema
from indicators.adx import calc_adx

DESCRIPTION = "Fast/slow EMA cross on 5m, filtered by ADX(14) strength and 1H EMA slope direction"

HYPOTHESIS = (
    "EMA crosses are the most widely traded signal in retail FX. On their own they whipsaw "
    "badly in ranging conditions. ADX(14) > threshold selects trending regimes where crosses "
    "have follow-through. The 1H EMA slope adds a higher-timeframe directional bias filter. "
    "Session filter avoids illiquid Asian hours. This is a baseline trend-following benchmark."
)

PARAM_GRID = {
    "fast_ema":      [5, 9, 13],
    "slow_ema":      [21, 34, 55],
    "adx_threshold": [20, 25, 30],
    "cooldown":      [12, 24],
}


def _align_1h_ema(ltf_index: pd.DatetimeIndex, ema_1h: pd.Series) -> np.ndarray:
    """Align 1H EMA slope to 5m bars via merge_asof."""
    htf = pd.DataFrame({
        "Date":  ema_1h.index,
        "slope": ema_1h.diff().shift(1).values,
    })
    ltf = pd.DataFrame({"Date": ltf_index})
    merged = pd.merge_asof(ltf.sort_values("Date"), htf.sort_values("Date"),
                           on="Date", direction="backward")
    return merged["slope"].values


def generate_signals(
    df_5m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_1d: pd.DataFrame,
    fast_ema: int = 9,
    slow_ema: int = 21,
    adx_threshold: int = 25,
    cooldown: int = 24,
) -> pd.DataFrame:
    close = df_5m["Close"].values
    n = len(close)

    # 5m EMAs — shift(1) so current bar can't see itself in the window
    close_s = df_5m["Close"].shift(1)
    ema_fast = calc_ema(close_s, length=fast_ema).values
    ema_slow = calc_ema(close_s, length=slow_ema).values

    # 5m ADX: compute on raw OHLC, then shift output by 1 bar to avoid lookahead.
    adx_result = calc_adx(df_5m, di_period=14, adx_period=14)
    adx = pd.Series(adx_result["adx"], index=df_5m.index).shift(1).values

    # 1H EMA(21) slope
    ema_1h = calc_ema(df_1h["Close"], length=21)
    slope_1h = _align_1h_ema(df_5m.index, ema_1h)

    hours = df_5m.index.hour.values

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    last_trade_bar = -999_999
    prev_fast_above = ema_fast[0] > ema_slow[0] if not np.isnan(ema_fast[0]) else False

    warmup = slow_ema + 14 + 2

    for i in range(warmup, n):
        if np.isnan(close[i]) or np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or np.isnan(adx[i]):
            prev_fast_above = ema_fast[i] > ema_slow[i] if not np.isnan(ema_fast[i]) else prev_fast_above
            continue

        in_session = 7 <= hours[i] < 22
        strong_trend = adx[i] > adx_threshold
        slope_bull = slope_1h[i] > 0 if not np.isnan(slope_1h[i]) else False
        slope_bear = slope_1h[i] < 0 if not np.isnan(slope_1h[i]) else False

        fast_above = ema_fast[i] > ema_slow[i]

        # Cross detection: compare state vs previous bar
        cross_up   = fast_above and not prev_fast_above
        cross_down = not fast_above and prev_fast_above

        # Exit: EMA recross in opposite direction
        long_exit[i]  = cross_down
        short_exit[i] = cross_up

        can_trade = (i - last_trade_bar) >= cooldown
        if can_trade and in_session and strong_trend:
            if cross_up and slope_bull:
                long_entry[i]  = True
                last_trade_bar = i
            elif cross_down and slope_bear:
                short_entry[i] = True
                last_trade_bar = i

        prev_fast_above = fast_above

    df_5m["long_entry"]  = long_entry
    df_5m["long_exit"]   = long_exit
    df_5m["short_entry"] = short_entry
    df_5m["short_exit"]  = short_exit
    return df_5m
