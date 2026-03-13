"""
R003: Supertrend Trend Follow

Entry when 5m Supertrend flips bullish/bearish.
Optional 1H Supertrend agreement filter.
Exit when 5m Supertrend flips against position.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from indicators.supertrend import calc_supertrend

DESCRIPTION = "5m Supertrend flip entries, optional 1H Supertrend directional filter, exit on flip"

HYPOTHESIS = (
    "Supertrend is one of the most popular trend-following indicators on TradingView. "
    "It self-adapts its trailing stop via ATR, so it stays in trends longer than a fixed "
    "channel exit. A 1H Supertrend filter should reduce false flips during intraday noise "
    "by requiring the higher-timeframe trend to agree. Session filter avoids thin Asian hours."
)

PARAM_GRID = {
    "atr_period":  [7, 10, 14],
    "multiplier":  [2.0, 3.0, 4.0],
    "htf_agree":   [True, False],
    "cooldown":    [12, 24],
}


def _align_1h_direction(ltf_index: pd.DatetimeIndex, direction_1h: np.ndarray, idx_1h: pd.DatetimeIndex) -> np.ndarray:
    """Align 1H Supertrend direction (+1/-1) to 5m bars via merge_asof."""
    htf = pd.DataFrame({
        "Date":      idx_1h,
        "direction": pd.Series(direction_1h, index=idx_1h).shift(1).values,
    })
    ltf = pd.DataFrame({"Date": ltf_index})
    merged = pd.merge_asof(ltf.sort_values("Date"), htf.sort_values("Date"),
                           on="Date", direction="backward")
    return merged["direction"].values


def generate_signals(
    df_5m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_1d: pd.DataFrame,
    atr_period: int = 10,
    multiplier: float = 3.0,
    htf_agree: bool = True,
    cooldown: int = 12,
) -> pd.DataFrame:
    close = df_5m["Close"].values
    n = len(close)

    # 5m Supertrend — compute on raw OHLC, shift direction output to avoid lookahead.
    # Shifting the full DataFrame (Pitfall #7) injects NaNs into H/L/C simultaneously,
    # collapsing ATR via RMA propagation → direction never flips → zero trades.
    st_result = calc_supertrend(df_5m, period=atr_period, multiplier=multiplier)
    direction_5m = pd.Series(st_result["direction"]).shift(1).values

    # 1H Supertrend (aligned to 5m)
    st_1h = calc_supertrend(df_1h, period=atr_period, multiplier=multiplier)
    direction_1h_aligned = _align_1h_direction(df_5m.index, st_1h["direction"], df_1h.index)

    hours = df_5m.index.hour.values

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    last_trade_bar = -999_999
    warmup = atr_period + 2

    for i in range(warmup, n):
        if np.isnan(close[i]) or np.isnan(direction_5m[i]):
            continue

        in_session = 7 <= hours[i] < 22

        dir_now  = direction_5m[i]
        dir_prev = direction_5m[i - 1]

        flip_bull = dir_now == 1  and dir_prev == -1
        flip_bear = dir_now == -1 and dir_prev == 1

        # Exit: Supertrend flips against us
        long_exit[i]  = flip_bear
        short_exit[i] = flip_bull

        htf_bull = (not htf_agree) or (direction_1h_aligned[i] == 1)
        htf_bear = (not htf_agree) or (direction_1h_aligned[i] == -1)

        can_trade = (i - last_trade_bar) >= cooldown
        if can_trade and in_session:
            if flip_bull and htf_bull:
                long_entry[i]  = True
                last_trade_bar = i
            elif flip_bear and htf_bear:
                short_entry[i] = True
                last_trade_bar = i

    df_5m["long_entry"]  = long_entry
    df_5m["long_exit"]   = long_exit
    df_5m["short_entry"] = short_entry
    df_5m["short_exit"]  = short_exit
    return df_5m
