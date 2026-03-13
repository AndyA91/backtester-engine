"""
R004: Bollinger Band Squeeze + Breakout

Entry when price closes outside the BB after a squeeze (narrow bandwidth).
1H EMA slope as higher-timeframe directional filter.
Exit when price crosses back inside the mid-band.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from indicators.bbands import calc_bbands
from indicators.ema import calc_ema

DESCRIPTION = "BB breakout after squeeze (bw < threshold), 1H EMA slope filter, exit at mid-band"

HYPOTHESIS = (
    "Bollinger Band squeezes (periods of contracting bandwidth) precede explosive moves as "
    "volatility cycles from low to high. Breaking out of the band after a squeeze signals "
    "the start of a new directional move. The 1H EMA slope adds a trend-direction filter "
    "so we only trade breakouts aligned with the higher-timeframe trend."
)

PARAM_GRID = {
    "bb_period":    [14, 20],
    "bb_std":       [1.5, 2.0, 2.5],
    "squeeze_pct":  [0.002, 0.003],   # bandwidth threshold (as fraction of price)
    "cooldown":     [12, 24],
}


def _align_1h_slope(ltf_index: pd.DatetimeIndex, ema_1h: pd.Series) -> np.ndarray:
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
    bb_period: int = 20,
    bb_std: float = 2.0,
    squeeze_pct: float = 0.003,
    cooldown: int = 24,
) -> pd.DataFrame:
    close = df_5m["Close"].values
    n = len(close)

    # BB on shifted close to avoid lookahead
    df_5m_lag = df_5m.copy()
    df_5m_lag["Close"] = df_5m["Close"].shift(1)
    df_5m_lag["High"]  = df_5m["High"].shift(1)
    df_5m_lag["Low"]   = df_5m["Low"].shift(1)
    bb = calc_bbands(df_5m_lag, period=bb_period, mult=bb_std)

    mid   = bb["mid"]
    upper = bb["upper"]
    lower = bb["lower"]
    bw    = bb["bw"]    # bandwidth = (upper - lower) / mid

    # 1H EMA(21) slope filter
    ema_1h   = calc_ema(df_1h["Close"], length=21)
    slope_1h = _align_1h_slope(df_5m.index, ema_1h)

    hours = df_5m.index.hour.values

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    last_trade_bar = -999_999
    warmup = bb_period + 2

    # Track whether a squeeze was active on the previous bar
    was_squeezed = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not np.isnan(bw[i - 1]):
            was_squeezed[i] = bw[i - 1] < squeeze_pct

    for i in range(warmup, n):
        if (np.isnan(close[i]) or np.isnan(mid[i]) or
                np.isnan(upper[i]) or np.isnan(bw[i])):
            continue

        in_session = 7 <= hours[i] < 22
        slope_bull = slope_1h[i] > 0 if not np.isnan(slope_1h[i]) else False
        slope_bear = slope_1h[i] < 0 if not np.isnan(slope_1h[i]) else False

        # Breakout: close outside band
        brk_up = close[i] > upper[i]
        brk_dn = close[i] < lower[i]

        # Entry only if previous bar had a squeeze
        long_cond  = brk_up and was_squeezed[i] and slope_bull
        short_cond = brk_dn and was_squeezed[i] and slope_bear

        # Exit: close crosses back inside mid-band
        long_exit[i]  = close[i] < mid[i]
        short_exit[i] = close[i] > mid[i]

        can_trade = (i - last_trade_bar) >= cooldown
        if can_trade and in_session:
            if long_cond:
                long_entry[i]  = True
                last_trade_bar = i
            elif short_cond:
                short_entry[i] = True
                last_trade_bar = i

    df_5m["long_entry"]  = long_entry
    df_5m["long_exit"]   = long_exit
    df_5m["short_entry"] = short_entry
    df_5m["short_exit"]  = short_exit
    return df_5m
