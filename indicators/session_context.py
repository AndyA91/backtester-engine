"""
Session Context Indicator — Intraday positioning for futures.

MYM (Micro E-mini Dow) trades in defined sessions with forced close
at 15:45 ET.  This indicator tracks where we are within a session:
range exhaustion, brick count, directional bias, and time remaining.

Columns produced:
    sess_range_used    Fraction of session's developing range already used
                       (high - low) / rolling_avg_session_range.  > 1 = extended
    sess_brick_count   Bricks formed so far in current session
    sess_dir_bias      Net up/down bricks in session / total bricks (−1 to +1)
    sess_minutes_left  Minutes until forced close (15:45 ET).  Negative = after.
    sess_is_opening    Boolean: within first 30 min of RTH (09:30-10:00 ET)
    sess_is_closing    Boolean: within last 30 min before forced close (15:15-15:45 ET)

All outputs are RAW (caller is responsible for .shift(1) if needed).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def _compute_et_offset(dt):
    """Return UTC→ET offset (-4 for EDT, -5 for EST)."""
    year = dt.year
    # 2nd Sunday of March
    mar1 = datetime(year, 3, 1)
    mar_sun2 = mar1 + timedelta(days=(6 - mar1.weekday()) % 7 + 7)
    # 1st Sunday of November
    nov1 = datetime(year, 11, 1)
    nov_sun1 = nov1 + timedelta(days=(6 - nov1.weekday()) % 7)

    if mar_sun2.date() <= dt.date() < nov_sun1.date():
        return -4  # EDT
    return -5  # EST


def calc_session_context(
    df: pd.DataFrame,
    forced_close_h: int = 15,
    forced_close_m: int = 45,
    rth_open_h: int = 9,
    rth_open_m: int = 30,
    range_lookback: int = 20,
) -> dict:
    """
    Compute session context metrics for a futures Renko DataFrame.

    Args:
        df: Renko DataFrame with DatetimeIndex, 'brick_up', 'High', 'Low'.
        forced_close_h/m: Forced close time in ET (default 15:45).
        rth_open_h/m: RTH open time in ET (default 09:30).
        range_lookback: Sessions to average for typical range.

    Returns:
        dict with session context arrays.
    """
    n = len(df)
    idx = df.index

    # Convert to ET minutes-since-midnight
    et_total_min = np.zeros(n, dtype=np.int32)
    et_dates = []  # ET date string for session grouping

    for i in range(n):
        dt = idx[i].to_pydatetime()
        offset = _compute_et_offset(dt)
        et_dt = dt + timedelta(hours=offset)
        et_total_min[i] = et_dt.hour * 60 + et_dt.minute
        et_dates.append(et_dt.strftime("%Y-%m-%d"))

    et_dates = np.array(et_dates)
    forced_close_min = forced_close_h * 60 + forced_close_m
    rth_open_min = rth_open_h * 60 + rth_open_m

    # Minutes left until forced close
    sess_minutes_left = (forced_close_min - et_total_min).astype(np.float64)

    # Opening / closing windows
    sess_is_opening = (et_total_min >= rth_open_min) & (et_total_min < rth_open_min + 30)
    sess_is_closing = (et_total_min >= forced_close_min - 30) & (et_total_min < forced_close_min)

    # Per-session metrics
    highs = df["High"].values
    lows = df["Low"].values
    brick_up = df["brick_up"].values.astype(np.int8)  # 1 for up, 0 for down

    sess_range_used = np.full(n, np.nan, dtype=np.float64)
    sess_brick_count = np.zeros(n, dtype=np.float64)
    sess_dir_bias = np.full(n, np.nan, dtype=np.float64)

    # Track per-session state
    completed_ranges = []
    cur_date = et_dates[0]
    sess_high = highs[0]
    sess_low = lows[0]
    sess_up = 0
    sess_down = 0
    sess_count = 0

    for i in range(n):
        if et_dates[i] != cur_date:
            # New session — save completed session range
            sess_range = sess_high - sess_low
            if sess_range > 0:
                completed_ranges.append(sess_range)

            # Reset
            cur_date = et_dates[i]
            sess_high = highs[i]
            sess_low = lows[i]
            sess_up = 0
            sess_down = 0
            sess_count = 0

        # Update session stats
        sess_high = max(sess_high, highs[i])
        sess_low = min(sess_low, lows[i])
        sess_count += 1
        if brick_up[i]:
            sess_up += 1
        else:
            sess_down += 1

        sess_brick_count[i] = sess_count

        # Direction bias: -1 (all down) to +1 (all up)
        total = sess_up + sess_down
        if total > 0:
            sess_dir_bias[i] = (sess_up - sess_down) / total

        # Range used vs average
        cur_range = sess_high - sess_low
        recent_ranges = completed_ranges[-range_lookback:]
        if len(recent_ranges) >= 3 and cur_range > 0:
            avg_range = np.mean(recent_ranges)
            if avg_range > 0:
                sess_range_used[i] = cur_range / avg_range

    return {
        "sess_range_used":   sess_range_used,
        "sess_brick_count":  sess_brick_count,
        "sess_dir_bias":     sess_dir_bias,
        "sess_minutes_left": sess_minutes_left,
        "sess_is_opening":   sess_is_opening.astype(np.float64),
        "sess_is_closing":   sess_is_closing.astype(np.float64),
    }
