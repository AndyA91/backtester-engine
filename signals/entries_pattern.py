"""
Pattern-Based Entry Signals

All functions return dict with "long_entry" and "short_entry" boolean arrays.
"""

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── 1. RSI Divergence ───────────────────────────────────────────────────────

def sig_rsi_divergence(df: pd.DataFrame, rsi_period: int = 14, pivot_left: int = 5,
                       pivot_right: int = 5) -> dict:
    """Bullish/bearish divergence between price and RSI."""
    from indicators.divergence import calc_divergence
    div = calc_divergence(df, oscillator="rsi", rsi_period=rsi_period,
                          pivot_left=pivot_left, pivot_right=pivot_right)
    return {
        "long_entry": div["bull_div"],
        "short_entry": div["bear_div"],
        "name": f"RSI Divergence {rsi_period}",
    }


# ── 2. MACD Divergence ──────────────────────────────────────────────────────

def sig_macd_divergence(df: pd.DataFrame, pivot_left: int = 5, pivot_right: int = 5) -> dict:
    """Bullish/bearish divergence between price and MACD histogram."""
    from indicators.divergence import calc_divergence
    div = calc_divergence(df, oscillator="macd", pivot_left=pivot_left,
                          pivot_right=pivot_right)
    return {
        "long_entry": div["bull_div"],
        "short_entry": div["bear_div"],
        "name": "MACD Divergence",
    }


# ── 3. Pivot Breakout ───────────────────────────────────────────────────────

def sig_pivot_breakout(df: pd.DataFrame, left: int = 10, right: int = 5) -> dict:
    """Price breaks above last pivot high / below last pivot low."""
    from indicators.zigzag import calc_swing_points
    swings = calc_swing_points(df, left=left, right=right)
    close = df["Close"].values
    n = len(close)

    # Track last known pivot high/low prices
    last_ph = np.nan
    last_pl = np.nan

    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if swings["pivot_high"][i]:
            last_ph = swings["ph_price"][i]
        if swings["pivot_low"][i]:
            last_pl = swings["pl_price"][i]

        if not np.isnan(last_ph) and close[i] > last_ph and close[i-1] <= last_ph:
            long_entry[i] = True
        if not np.isnan(last_pl) and close[i] < last_pl and close[i-1] >= last_pl:
            short_entry[i] = True

    return {"long_entry": long_entry, "short_entry": short_entry, "name": f"Pivot Breakout {left}/{right}"}


# ── 4. Inside Bar Breakout ──────────────────────────────────────────────────

def sig_inside_bar(df: pd.DataFrame) -> dict:
    """Inside bar (bar within previous bar's range) breakout."""
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    n = len(close)

    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)

    for i in range(2, n):
        # Bar i-1 is inside bar: its range fits within bar i-2's range
        is_inside = high[i-1] <= high[i-2] and low[i-1] >= low[i-2]
        if is_inside:
            # Bar i breaks out
            if close[i] > high[i-1]:
                long_entry[i] = True
            elif close[i] < low[i-1]:
                short_entry[i] = True

    return {"long_entry": long_entry, "short_entry": short_entry, "name": "Inside Bar Breakout"}


# ── 5. Engulfing Candle ─────────────────────────────────────────────────────

def sig_engulfing(df: pd.DataFrame) -> dict:
    """Bullish/bearish engulfing candle pattern."""
    open_ = df["Open"].values
    close = df["Close"].values
    n = len(close)

    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)

    for i in range(1, n):
        prev_body_bear = close[i-1] < open_[i-1]  # previous bar was bearish
        prev_body_bull = close[i-1] > open_[i-1]   # previous bar was bullish
        curr_body_bull = close[i] > open_[i]        # current bar is bullish
        curr_body_bear = close[i] < open_[i]        # current bar is bearish

        # Bullish engulfing: bearish bar followed by bullish bar that engulfs it
        if prev_body_bear and curr_body_bull:
            if open_[i] <= close[i-1] and close[i] >= open_[i-1]:
                long_entry[i] = True

        # Bearish engulfing: bullish bar followed by bearish bar that engulfs it
        if prev_body_bull and curr_body_bear:
            if open_[i] >= close[i-1] and close[i] <= open_[i-1]:
                short_entry[i] = True

    return {"long_entry": long_entry, "short_entry": short_entry, "name": "Engulfing"}


# ── 6. Parabolic SAR Flip ───────────────────────────────────────────────────

def sig_psar_flip(df: pd.DataFrame, start: float = 0.02, increment: float = 0.02,
                  maximum: float = 0.2) -> dict:
    """Parabolic SAR flips sides — trend direction change."""
    from indicators.parabolic_sar import calc_psar
    sar = calc_psar(df, start=start, increment=increment, maximum=maximum)
    direction = sar["direction"]
    n = len(direction)

    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)

    for i in range(1, n):
        long_entry[i] = direction[i] == 1 and direction[i-1] == -1
        short_entry[i] = direction[i] == -1 and direction[i-1] == 1

    return {"long_entry": long_entry, "short_entry": short_entry, "name": "PSAR Flip"}
