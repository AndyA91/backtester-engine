"""
Divergence Detection — RSI & MACD

Automated detection of bullish and bearish divergences between price and
oscillators. Divergences are among the highest win-rate reversal signals.

Bullish divergence: price makes a lower low, oscillator makes a higher low
Bearish divergence: price makes a higher high, oscillator makes a lower high

Uses pivot-based swing detection for robust identification.

Usage:
    from indicators.divergence import calc_divergence

    result = calc_divergence(df, oscillator="rsi", pivot_left=5, pivot_right=5)
    # result["bull_div"]     — True at bars where bullish divergence confirmed
    # result["bear_div"]     — True at bars where bearish divergence confirmed
    # result["osc"]          — oscillator values used for detection

    # Or pass your own oscillator array:
    result = calc_divergence(df, osc_values=my_oscillator, pivot_left=5, pivot_right=5)

Interpretation:
    Bullish divergence in oversold zone (RSI < 30) → strong long signal
    Bearish divergence in overbought zone (RSI > 70) → strong short signal
    Multiple consecutive divergences → stronger signal
    Best paired with ADX < 25 regime filter for mean-reversion contexts
"""

import numpy as np
import pandas as pd
from indicators.zigzag import calc_swing_points
from indicators.rsi import calc_rsi
from indicators.macd import calc_macd


def calc_divergence(
    df: pd.DataFrame,
    oscillator: str = "rsi",
    osc_values: np.ndarray = None,
    pivot_left: int = 5,
    pivot_right: int = 5,
    max_bars_between: int = 100,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
) -> dict:
    """
    Parameters
    ----------
    df                : DataFrame with 'High', 'Low', 'Close'
    oscillator        : 'rsi' or 'macd' (ignored if osc_values provided)
    osc_values        : Pre-computed oscillator array (overrides oscillator param)
    pivot_left        : Left bars for pivot detection (default 5)
    pivot_right       : Right bars for pivot detection (default 5)
    max_bars_between  : Max bars between two swing points for divergence (default 100)
    rsi_period        : RSI period if oscillator='rsi'
    macd_fast/slow/signal : MACD params if oscillator='macd'

    Returns
    -------
    dict with keys: bull_div, bear_div, osc (all numpy arrays)
    """
    n = len(df)

    # --- Get oscillator values ---
    if osc_values is not None:
        osc = osc_values.copy()
    elif oscillator == "rsi":
        osc = calc_rsi(df, period=rsi_period)["rsi"]
    elif oscillator == "macd":
        osc = calc_macd(df, fast=macd_fast, slow=macd_slow, signal=macd_signal)["histogram"]
    else:
        raise ValueError(f"Unknown oscillator: {oscillator}. Use 'rsi', 'macd', or provide osc_values.")

    # --- Find price swing points ---
    swings = calc_swing_points(df, left=pivot_left, right=pivot_right)

    bull_div = np.zeros(n, dtype=bool)
    bear_div = np.zeros(n, dtype=bool)

    # --- Collect pivot lows for bullish divergence ---
    pl_indices = np.where(swings["pivot_low"])[0]
    for k in range(1, len(pl_indices)):
        curr = pl_indices[k]
        prev = pl_indices[k - 1]

        if curr - prev > max_bars_between:
            continue

        # Bullish: price lower low, oscillator higher low
        price_lower = df["Low"].values[curr] < df["Low"].values[prev]
        osc_higher = osc[curr] > osc[prev]

        if price_lower and osc_higher:
            # Mark at the confirmation bar (right bars after the pivot)
            confirm_bar = min(curr + pivot_right, n - 1)
            bull_div[confirm_bar] = True

    # --- Collect pivot highs for bearish divergence ---
    ph_indices = np.where(swings["pivot_high"])[0]
    for k in range(1, len(ph_indices)):
        curr = ph_indices[k]
        prev = ph_indices[k - 1]

        if curr - prev > max_bars_between:
            continue

        # Bearish: price higher high, oscillator lower high
        price_higher = df["High"].values[curr] > df["High"].values[prev]
        osc_lower = osc[curr] < osc[prev]

        if price_higher and osc_lower:
            confirm_bar = min(curr + pivot_right, n - 1)
            bear_div[confirm_bar] = True

    return {
        "bull_div": bull_div,
        "bear_div": bear_div,
        "osc": osc,
    }
