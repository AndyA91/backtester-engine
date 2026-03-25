"""
Brick Velocity Indicator — Renko-specific time-based momentum.

On Renko charts each brick takes a variable amount of time to form.
Fast brick formation signals strong momentum; slow formation signals
exhaustion or consolidation.  No price-based indicator captures this.

Columns produced:
    brick_td        Time-delta in seconds between consecutive bricks
    vel_sma         Rolling SMA of brick_td (smoothed baseline)
    vel_ratio       brick_td / vel_sma  (< 1 = fast bricks = momentum)
    vel_zscore      Z-score of brick_td vs rolling window (< 0 = fast)

All outputs are RAW (caller is responsible for .shift(1) if needed).
"""

import numpy as np
import pandas as pd


def calc_brick_velocity(
    df: pd.DataFrame,
    lookback: int = 20,
) -> dict:
    """
    Compute brick velocity metrics from a Renko DataFrame.

    Args:
        df: Renko DataFrame with a DatetimeIndex.
        lookback: Rolling window for SMA and z-score (default 20).

    Returns:
        dict with keys: brick_td, vel_sma, vel_ratio, vel_zscore
        Each value is a numpy array of length len(df).
    """
    n = len(df)

    # Time deltas in seconds between consecutive bricks
    timestamps = df.index.astype(np.int64) // 10**9  # unix seconds
    td = np.full(n, np.nan, dtype=np.float64)
    td[1:] = np.diff(timestamps).astype(np.float64)

    # Clip extreme outliers (overnight gaps, weekends) at 99th percentile
    valid = td[~np.isnan(td)]
    if len(valid) > 0:
        cap = np.percentile(valid, 99)
        td = np.where(np.isnan(td), td, np.minimum(td, cap))

    # Rolling SMA of time deltas
    td_series = pd.Series(td)
    vel_sma = td_series.rolling(lookback, min_periods=1).mean().values

    # Velocity ratio: current / average  (< 1 means faster than normal)
    with np.errstate(divide="ignore", invalid="ignore"):
        vel_ratio = np.where(vel_sma > 0, td / vel_sma, np.nan)

    # Z-score: how many stdev from the rolling mean
    vel_std = td_series.rolling(lookback, min_periods=2).std().values
    with np.errstate(divide="ignore", invalid="ignore"):
        vel_zscore = np.where(vel_std > 0, (td - vel_sma) / vel_std, 0.0)

    return {
        "brick_td":   td,
        "vel_sma":    vel_sma,
        "vel_ratio":  vel_ratio,
        "vel_zscore": vel_zscore,
    }
