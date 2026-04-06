"""
Consecutive Candle Streak Analysis [LuxAlgo]

Tracks consecutive bullish/bearish close-vs-previous-close streaks and
computes rolling continuation/reversal probabilities based on historical
streak data.

Pine source: indicators/LuxAlgo/Consecutive_Candle_Streak_Analysis__LuxAlgo_.pine

NOTE: Pine computes stats only on the last bar (barstate.islast). This port
computes rolling stats at every bar using completed streak history up to that
point (within lookback limit), making it usable in a backtest signal loop.

Outputs:
    streak_len     — current streak length (int, 0 = neutral)
    streak_bull    — True if current streak is bullish
    continue_prob  — probability of continuation given current length (0-100)
    reversal_prob  — probability of reversal given current length (0-100)
    avg_pct_move   — average absolute % move of same-direction streaks at this length

Usage:
    from indicators.luxalgo.candle_streak import calc_candle_streak
    result = calc_candle_streak(df, lookback=10000)
"""

import numpy as np
import pandas as pd
from collections import namedtuple


_Streak = namedtuple("_Streak", ["length", "pct_change", "is_bullish"])


def calc_candle_streak(
    df: pd.DataFrame,
    lookback: int = 10000,
) -> dict:
    """
    Parameters
    ----------
    df       : DataFrame with 'Close'
    lookback : Max historical completed streaks to use for stats
    """
    close = df["Close"].values.astype(float)
    n = len(close)

    streak_len_out = np.zeros(n, dtype=int)
    streak_bull_out = np.zeros(n, dtype=bool)
    continue_prob_out = np.zeros(n)
    reversal_prob_out = np.zeros(n)
    avg_pct_move_out = np.zeros(n)

    # State
    completed_streaks: list[_Streak] = []
    current_streak_len = 0
    streak_start_price = close[0] if n > 0 else 0.0
    prev_dir = 0  # +1 bull, -1 bear, 0 neutral

    for i in range(1, n):
        # Direction based on close vs previous close
        if close[i] > close[i - 1]:
            cur_dir = 1
        elif close[i] < close[i - 1]:
            cur_dir = -1
        else:
            cur_dir = 0

        changed = cur_dir != prev_dir

        if changed:
            # Record completed streak
            if current_streak_len > 0 and prev_dir != 0:
                pct = (close[i - 1] - streak_start_price) / streak_start_price * 100.0 if streak_start_price != 0 else 0.0
                completed_streaks.append(_Streak(
                    length=current_streak_len,
                    pct_change=pct,
                    is_bullish=prev_dir == 1,
                ))

            # Start new streak
            current_streak_len = 1 if cur_dir != 0 else 0
            streak_start_price = close[i - 1]
        elif cur_dir != 0:
            current_streak_len += 1

        prev_dir = cur_dir
        cur_len = current_streak_len
        cur_bull = cur_dir == 1

        streak_len_out[i] = cur_len
        streak_bull_out[i] = cur_bull

        # Compute stats from completed streak history
        if cur_len > 0 and len(completed_streaks) > 0:
            start_idx = max(0, len(completed_streaks) - lookback)
            count_reaching = 0
            count_continuing = 0
            sum_move = 0.0

            for j in range(start_idx, len(completed_streaks)):
                s = completed_streaks[j]
                if s.is_bullish == cur_bull and s.length >= cur_len:
                    count_reaching += 1
                    sum_move += abs(s.pct_change)
                    if s.length > cur_len:
                        count_continuing += 1

            if count_reaching > 0:
                continue_prob_out[i] = (count_continuing / count_reaching) * 100.0
                reversal_prob_out[i] = ((count_reaching - count_continuing) / count_reaching) * 100.0
                avg_pct_move_out[i] = sum_move / count_reaching

    return {
        "streak_len": streak_len_out,
        "streak_bull": streak_bull_out,
        "continue_prob": continue_prob_out,
        "reversal_prob": reversal_prob_out,
        "avg_pct_move": avg_pct_move_out,
    }
