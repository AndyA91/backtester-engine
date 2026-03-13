"""
Parabolic SAR — Stop and Reverse

Trend-following indicator that places a trailing stop below price in an uptrend
and above price in a downtrend. Flips ("reverses") when price crosses the stop.

Matches Pine's ta.sar(start, increment, maximum) exactly.

Usage:
    from indicators.parabolic_sar import calc_psar

    result = calc_psar(df)
    # result["psar"]      — SAR value (the trailing stop level)
    # result["direction"] — +1 bullish (SAR below price), -1 bearish (SAR above price)
    # result["af"]        — current Acceleration Factor at each bar

Interpretation:
    Direction flips +1 → -1 when close crosses below the SAR → exit long / enter short
    Direction flips -1 → +1 when close crosses above the SAR → exit short / enter long
    AF starts at 'start' (0.02) and increases by 'increment' (0.02) each time the
    Extreme Point (EP) sets a new extreme, up to 'maximum' (0.2).
    The SAR accelerates towards EP as the trend extends.
"""

import numpy as np
import pandas as pd


def calc_psar(
    df: pd.DataFrame,
    start: float = 0.02,
    increment: float = 0.02,
    maximum: float = 0.2,
) -> dict:
    """
    Parameters
    ----------
    df        : DataFrame with 'High', 'Low', 'Close'
    start     : Initial acceleration factor (default 0.02)
    increment : AF step size each time EP is updated (default 0.02)
    maximum   : Maximum acceleration factor cap (default 0.2)

    Returns
    -------
    dict with keys: psar, direction, af  (all numpy arrays)
    """
    high  = df["High"].values
    low   = df["Low"].values
    close = df["Close"].values
    n     = len(close)

    psar      = np.full(n, np.nan)
    direction = np.zeros(n, dtype=int)
    af_arr    = np.full(n, np.nan)

    if n < 2:
        return {"psar": psar, "direction": direction, "af": af_arr}

    # Initialise: assume bullish start (SAR below price)
    bull     = True
    af       = start
    ep       = high[0]       # Extreme Point: highest high in current uptrend
    sar      = low[0]        # start SAR at first bar's low

    psar[0]      = sar
    direction[0] = 1
    af_arr[0]    = af

    for i in range(1, n):
        prev_sar = sar

        if bull:
            # Next SAR
            sar = prev_sar + af * (ep - prev_sar)
            # SAR cannot be above the two previous lows
            sar = min(sar, low[i - 1])
            if i >= 2:
                sar = min(sar, low[i - 2])

            # Check flip
            if low[i] < sar:
                bull  = False
                sar   = ep        # SAR jumps to EP (highest high)
                ep    = low[i]    # new EP = current low
                af    = start     # reset AF
            else:
                # Update EP and AF if new extreme
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + increment, maximum)
        else:
            # Bear SAR
            sar = prev_sar + af * (ep - prev_sar)
            # SAR cannot be below the two previous highs
            sar = max(sar, high[i - 1])
            if i >= 2:
                sar = max(sar, high[i - 2])

            # Check flip
            if high[i] > sar:
                bull  = True
                sar   = ep       # SAR jumps to EP (lowest low)
                ep    = high[i]  # new EP = current high
                af    = start    # reset AF
            else:
                # Update EP and AF if new extreme
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + increment, maximum)

        psar[i]      = sar
        direction[i] = 1 if bull else -1
        af_arr[i]    = af

    return {
        "psar":      psar,
        "direction": direction,
        "af":        af_arr,
    }
