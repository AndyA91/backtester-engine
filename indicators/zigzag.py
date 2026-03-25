"""
ZigZag + Swing Detection

Identifies significant swing highs and swing lows by filtering out noise
below a minimum percentage threshold. Connects the swings to form a zigzag
line that highlights the market's structural moves.

Also provides swing point arrays useful for:
  - Divergence detection (compare oscillator values at consecutive swings)
  - Support/resistance identification
  - Adaptive TP/SL sizing based on average swing magnitude

Usage:
    from indicators.zigzag import calc_zigzag

    result = calc_zigzag(df, pct_threshold=1.0)
    # result["zigzag"]      — zigzag line (NaN between pivots, price at pivots)
    # result["swing_high"]  — True at swing high bars
    # result["swing_low"]   — True at swing low bars
    # result["swing_type"]  — +1 at swing highs, -1 at swing lows, 0 elsewhere
    # result["swing_price"] — price at each swing point (NaN elsewhere)

    from indicators.zigzag import calc_swing_points

    result = calc_swing_points(df, left=5, right=5)
    # result["pivot_high"]  — True at pivot high bars
    # result["pivot_low"]   — True at pivot low bars
    # result["ph_price"]    — High price at pivot highs (NaN elsewhere)
    # result["pl_price"]    — Low price at pivot lows (NaN elsewhere)

Interpretation:
    Consecutive higher swing lows → uptrend structure
    Consecutive lower swing highs → downtrend structure
    Swing magnitude decreasing → momentum weakening
    Use pivot_high / pivot_low for divergence scanning
"""

import numpy as np
import pandas as pd


def calc_zigzag(
    df: pd.DataFrame,
    pct_threshold: float = 1.0,
) -> dict:
    """
    Percentage-based ZigZag.

    Parameters
    ----------
    df            : DataFrame with 'High', 'Low', 'Close'
    pct_threshold : Minimum % move to register a new swing (default 1.0%)

    Returns
    -------
    dict with keys: zigzag, swing_high, swing_low, swing_type, swing_price
    """
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    n = len(high)

    zigzag = np.full(n, np.nan)
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    swing_type = np.zeros(n, dtype=int)
    swing_price = np.full(n, np.nan)

    threshold = pct_threshold / 100.0

    # State: 1 = looking for swing high (trending up), -1 = looking for swing low
    direction = 0
    last_high = high[0]
    last_low = low[0]
    last_high_idx = 0
    last_low_idx = 0

    for i in range(1, n):
        if direction == 0:
            # Initialize direction
            if high[i] > last_high:
                last_high = high[i]
                last_high_idx = i
                direction = 1
            elif low[i] < last_low:
                last_low = low[i]
                last_low_idx = i
                direction = -1

        elif direction == 1:
            # Trending up — tracking the high
            if high[i] > last_high:
                last_high = high[i]
                last_high_idx = i
            elif last_high > 0 and (last_high - low[i]) / last_high >= threshold:
                # Reversal down — mark swing high
                swing_high[last_high_idx] = True
                swing_type[last_high_idx] = 1
                swing_price[last_high_idx] = last_high
                zigzag[last_high_idx] = last_high

                last_low = low[i]
                last_low_idx = i
                direction = -1

        elif direction == -1:
            # Trending down — tracking the low
            if low[i] < last_low:
                last_low = low[i]
                last_low_idx = i
            elif last_low > 0 and (high[i] - last_low) / last_low >= threshold:
                # Reversal up — mark swing low
                swing_low[last_low_idx] = True
                swing_type[last_low_idx] = -1
                swing_price[last_low_idx] = last_low
                zigzag[last_low_idx] = last_low

                last_high = high[i]
                last_high_idx = i
                direction = 1

    return {
        "zigzag": zigzag,
        "swing_high": swing_high,
        "swing_low": swing_low,
        "swing_type": swing_type,
        "swing_price": swing_price,
    }


def calc_swing_points(
    df: pd.DataFrame,
    left: int = 5,
    right: int = 5,
) -> dict:
    """
    Pivot-based swing detection — matches Pine's ta.pivothigh / ta.pivotlow.

    A pivot high is a bar whose High is the highest of [left + 1 + right] bars.
    A pivot low is a bar whose Low is the lowest of [left + 1 + right] bars.
    Pivots are confirmed 'right' bars after the fact.

    Parameters
    ----------
    df    : DataFrame with 'High', 'Low'
    left  : Bars to the left for pivot confirmation (default 5)
    right : Bars to the right for pivot confirmation (default 5)

    Returns
    -------
    dict with keys: pivot_high, pivot_low, ph_price, pl_price
    """
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    n = len(high)

    pivot_high = np.zeros(n, dtype=bool)
    pivot_low = np.zeros(n, dtype=bool)
    ph_price = np.full(n, np.nan)
    pl_price = np.full(n, np.nan)

    for i in range(left, n - right):
        # Check pivot high
        is_ph = True
        for j in range(i - left, i):
            if high[j] > high[i]:
                is_ph = False
                break
        if is_ph:
            for j in range(i + 1, i + right + 1):
                if high[j] >= high[i]:
                    is_ph = False
                    break
        if is_ph:
            pivot_high[i] = True
            ph_price[i] = high[i]

        # Check pivot low
        is_pl = True
        for j in range(i - left, i):
            if low[j] < low[i]:
                is_pl = False
                break
        if is_pl:
            for j in range(i + 1, i + right + 1):
                if low[j] <= low[i]:
                    is_pl = False
                    break
        if is_pl:
            pivot_low[i] = True
            pl_price[i] = low[i]

    return {
        "pivot_high": pivot_high,
        "pivot_low": pivot_low,
        "ph_price": ph_price,
        "pl_price": pl_price,
    }
