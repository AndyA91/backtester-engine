"""
Streak Momentum Indicator — Enhanced brick streak analysis.

Goes beyond simple "N bricks in a row" by measuring how streaks
evolve over time: are bricks forming faster (acceleration) or
slower (deceleration) within a streak?

Columns produced:
    streak_len       Current streak length (signed: + for up, - for down)
    streak_vel_avg   Average brick velocity (seconds) within current streak
    streak_accel     Streak acceleration: ratio of recent vs early velocity
                     < 1 = accelerating (bricks getting faster)
                     > 1 = decelerating (bricks getting slower)
    streak_age_pct   Current streak length as percentile of historical streaks

All outputs are RAW (caller is responsible for .shift(1) if needed).
"""

import numpy as np
import pandas as pd


def calc_streak_momentum(
    df: pd.DataFrame,
    history_window: int = 200,
) -> dict:
    """
    Compute streak momentum metrics from a Renko DataFrame.

    Args:
        df: Renko DataFrame with 'brick_up' column and DatetimeIndex.
        history_window: Number of past streaks to use for percentile ranking.

    Returns:
        dict with keys: streak_len, streak_vel_avg, streak_accel, streak_age_pct
    """
    n = len(df)
    brick_up = df["brick_up"].values.astype(bool)

    # Time deltas in seconds
    timestamps = df.index.astype(np.int64) // 10**9
    td = np.full(n, np.nan, dtype=np.float64)
    td[1:] = np.diff(timestamps).astype(np.float64)

    # Cap outliers (overnight/weekend gaps)
    valid = td[~np.isnan(td)]
    if len(valid) > 0:
        cap = np.percentile(valid, 99)
        td = np.where(np.isnan(td), td, np.minimum(td, cap))

    streak_len = np.zeros(n, dtype=np.float64)
    streak_vel_avg = np.full(n, np.nan, dtype=np.float64)
    streak_accel = np.full(n, np.nan, dtype=np.float64)
    streak_age_pct = np.full(n, np.nan, dtype=np.float64)

    # Track streaks
    cur_len = 1
    cur_dir = brick_up[0]
    streak_start = 0
    completed_streaks = []  # list of streak lengths for percentile

    for i in range(1, n):
        if brick_up[i] == cur_dir:
            cur_len += 1
        else:
            # Streak ended — record it
            completed_streaks.append(cur_len)
            cur_len = 1
            cur_dir = brick_up[i]
            streak_start = i

        # Signed streak length
        sign = 1 if brick_up[i] else -1
        streak_len[i] = sign * cur_len

        # Average velocity within current streak
        if cur_len >= 2:
            streak_tds = td[streak_start + 1: i + 1]
            streak_tds = streak_tds[~np.isnan(streak_tds)]
            if len(streak_tds) > 0:
                streak_vel_avg[i] = np.mean(streak_tds)

                # Acceleration: compare second half vs first half of streak
                if len(streak_tds) >= 4:
                    mid = len(streak_tds) // 2
                    first_half = np.mean(streak_tds[:mid])
                    second_half = np.mean(streak_tds[mid:])
                    if first_half > 0:
                        streak_accel[i] = second_half / first_half

        # Percentile of current streak vs history
        recent = completed_streaks[-history_window:] if completed_streaks else []
        if len(recent) >= 10:
            streak_age_pct[i] = (
                np.sum(np.array(recent) <= abs(cur_len)) / len(recent) * 100
            )

    return {
        "streak_len":     streak_len,
        "streak_vel_avg": streak_vel_avg,
        "streak_accel":   streak_accel,
        "streak_age_pct": streak_age_pct,
    }
