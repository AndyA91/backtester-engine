"""
Rolling Segment [LuxAlgo]

Linear trend-following overlay that moves at an ATR-derived slope.
Switches between fast and slow slope modes based on price distance from
the segment. Reverses direction when price exceeds a threshold.

Pine source: indicators/LuxAlgo/Rolling_Segment__LuxAlgo_.pine

Outputs:
    roll_seg      — segment price level (float)
    trend         — direction +1 (bullish) / -1 (bearish)
    bull_reversal — True on bullish reversal bar
    bear_reversal — True on bearish reversal bar

Usage:
    from indicators.luxalgo.rolling_segment import calc_rolling_segment
    result = calc_rolling_segment(df)
    # result["roll_seg"], result["trend"], result["bull_reversal"], result["bear_reversal"]
"""

import numpy as np
import pandas as pd


def calc_rolling_segment(
    df: pd.DataFrame,
    fast_length: int = 5,
    slow_length: int = 10,
    atr_length: int = 200,
    fast_threshold: float = 1.0,
    reverse_threshold: float = 2.0,
) -> dict:
    """
    Parameters
    ----------
    df                : DataFrame with 'High', 'Low', 'Close'
    fast_length       : Fast slope divisor (ATR / fast_length)
    slow_length       : Slow slope divisor (ATR / slow_length)
    atr_length        : ATR lookback period
    fast_threshold    : ATR multiplier to trigger fast mode
    reverse_threshold : ATR multiplier to trigger trend reversal
    """
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    close = df["Close"].values.astype(float)
    n = len(close)

    # Pre-compute ATR(atr_length) — True Range then rolling mean
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    atr_series = pd.Series(tr).rolling(atr_length, min_periods=1).mean().values

    # State variables (matching Pine var declarations)
    roll_seg_out = np.full(n, np.nan)
    trend_out = np.zeros(n, dtype=int)
    bull_rev_out = np.zeros(n, dtype=bool)
    bear_rev_out = np.zeros(n, dtype=bool)

    roll_seg = close[0]
    trend = 1
    active_slope = 0.0
    is_fast_mode = False

    for i in range(n):
        atr = atr_series[i]
        bull_reversal = False
        bear_reversal = False

        if not np.isnan(atr) and atr > 0:
            dist_from_seg = close[i] - roll_seg

            # Check for reversal
            bull_reversal = trend == -1 and dist_from_seg > reverse_threshold * atr
            bear_reversal = trend == 1 and -dist_from_seg > reverse_threshold * atr

            if bull_reversal:
                trend = 1
            elif bear_reversal:
                trend = -1

            # Check for speed change
            new_is_fast = abs(dist_from_seg) > fast_threshold * atr
            speed_changed = new_is_fast != is_fast_mode

            # Update slope on trend/speed changes
            if bull_reversal or bear_reversal or speed_changed or active_slope == 0.0:
                is_fast_mode = new_is_fast
                active_slope = atr / fast_length if is_fast_mode else atr / slow_length

            # Update segment with linear movement
            roll_seg = roll_seg + trend * active_slope
        else:
            roll_seg = close[i]

        roll_seg_out[i] = roll_seg
        trend_out[i] = trend
        bull_rev_out[i] = bull_reversal
        bear_rev_out[i] = bear_reversal

    return {
        "roll_seg": roll_seg_out,
        "trend": trend_out,
        "bull_reversal": bull_rev_out,
        "bear_reversal": bear_rev_out,
    }
