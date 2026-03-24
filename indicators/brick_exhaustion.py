"""
Brick Exhaustion Indicator — Detects when a Renko trend is losing steam.

Combines three signals of exhaustion:
  1. Velocity deceleration within the current streak
  2. Streak length relative to historical norms
  3. ADX rolling off (trend weakening)

Columns produced:
    exhaust_score    0.0 (fresh trend) to 1.0 (fully exhausted)
    exhaust_alert    Boolean: exhaust_score > threshold (default 0.7)
    vel_decel        Velocity deceleration component (0 to 1)
    streak_overext   Streak over-extension component (0 to 1)
    adx_rolloff      ADX rolloff component (0 to 1)

All outputs are RAW (caller is responsible for .shift(1) if needed).
"""

import numpy as np
import pandas as pd


def calc_brick_exhaustion(
    df: pd.DataFrame,
    vel_lookback: int = 20,
    streak_pct_threshold: float = 80.0,
    adx_rolloff_bars: int = 5,
    alert_threshold: float = 0.7,
) -> dict:
    """
    Compute exhaustion score for Renko trends.

    Args:
        df: Renko DataFrame with 'brick_up', 'adx', and DatetimeIndex.
        vel_lookback: Window for velocity baseline.
        streak_pct_threshold: Percentile above which streak is "over-extended".
        adx_rolloff_bars: Bars to look back for ADX decline.
        alert_threshold: Threshold for exhaust_alert flag.

    Returns:
        dict with exhaustion metric arrays.
    """
    n = len(df)
    brick_up = df["brick_up"].values.astype(bool)

    # --- Time deltas ---
    timestamps = df.index.astype(np.int64) // 10**9
    td = np.full(n, np.nan, dtype=np.float64)
    td[1:] = np.diff(timestamps).astype(np.float64)
    valid = td[~np.isnan(td)]
    if len(valid) > 0:
        cap = np.percentile(valid, 99)
        td = np.where(np.isnan(td), td, np.minimum(td, cap))

    # --- Component 1: Velocity deceleration ---
    vel_decel = np.zeros(n, dtype=np.float64)

    cur_dir = brick_up[0]
    streak_start = 0

    for i in range(1, n):
        if brick_up[i] != cur_dir:
            cur_dir = brick_up[i]
            streak_start = i

        streak_len = i - streak_start + 1
        if streak_len >= 4:
            streak_tds = td[streak_start + 1: i + 1]
            streak_tds = streak_tds[~np.isnan(streak_tds)]
            if len(streak_tds) >= 4:
                mid = len(streak_tds) // 2
                first_half = np.mean(streak_tds[:mid])
                second_half = np.mean(streak_tds[mid:])
                if first_half > 0:
                    ratio = second_half / first_half
                    # ratio > 1 means slowing down
                    vel_decel[i] = np.clip((ratio - 1.0) / 2.0, 0, 1)

    # --- Component 2: Streak over-extension ---
    streak_overext = np.zeros(n, dtype=np.float64)
    completed_streaks = []
    cur_dir = brick_up[0]
    cur_len = 1

    for i in range(1, n):
        if brick_up[i] == cur_dir:
            cur_len += 1
        else:
            completed_streaks.append(cur_len)
            cur_dir = brick_up[i]
            cur_len = 1

        # Percentile of current streak
        recent = completed_streaks[-200:] if completed_streaks else []
        if len(recent) >= 10:
            pct = np.sum(np.array(recent) <= cur_len) / len(recent) * 100
            if pct >= streak_pct_threshold:
                streak_overext[i] = np.clip(
                    (pct - streak_pct_threshold) / (100 - streak_pct_threshold), 0, 1
                )

    # --- Component 3: ADX rolloff ---
    adx_rolloff = np.zeros(n, dtype=np.float64)
    adx = df["adx"].values.astype(np.float64) if "adx" in df.columns else np.full(n, np.nan)

    for i in range(adx_rolloff_bars, n):
        if np.isnan(adx[i]) or np.isnan(adx[i - adx_rolloff_bars]):
            continue
        adx_change = adx[i] - adx[i - adx_rolloff_bars]
        if adx_change < 0 and adx[i] > 20:
            # ADX was high but declining
            adx_rolloff[i] = np.clip(-adx_change / 15.0, 0, 1)

    # --- Combined exhaustion score ---
    # Weights: velocity 40%, streak extension 35%, ADX rolloff 25%
    exhaust_score = 0.40 * vel_decel + 0.35 * streak_overext + 0.25 * adx_rolloff
    exhaust_score = np.clip(exhaust_score, 0, 1)

    exhaust_alert = exhaust_score > alert_threshold

    return {
        "exhaust_score":  exhaust_score,
        "exhaust_alert":  exhaust_alert.astype(np.float64),
        "vel_decel":      vel_decel,
        "streak_overext": streak_overext,
        "adx_rolloff":    adx_rolloff,
    }
