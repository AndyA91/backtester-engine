"""
R021: KAMA Ribbon v2 — Enhanced with quality filters

Improvements over R020:
  1. ADX gate — skip entries when market is choppy (ADX below threshold)
  2. Ribbon spread filter — only enter when ribbon is fanning out (spread increasing)
  3. KAMA slope filter — fastest KAMA must slope in trade direction
  4. Softer exit — exit on price crossing mid-KAMA instead of full alignment break

Base: 3-line ribbon (proven best across all pairs in R020).
"""

import numpy as np
import pandas as pd

from indicators.kama import calc_kama

DESCRIPTION = "KAMA ribbon v2 — alignment + ADX/spread/slope quality filters"

HYPOTHESIS = (
    "R020 showed KAMA ribbon alignment is a strong trend signal across all pairs. "
    "Adding quality filters (ADX for trend strength, ribbon spread for momentum, "
    "KAMA slope for direction confirmation) should reduce false signals during "
    "low-quality alignment periods and improve already-high PF."
)

RIBBONS = {
    "3L_8_21_55":   (8, 21, 55),
    "3L_5_13_34":   (5, 13, 34),
}

PARAM_GRID = {
    "ribbon":           list(RIBBONS.keys()),
    "cooldown":         [5, 10],
    "adx_gate":         [0, 20, 25],
    "use_spread_filter": [True, False],
    "use_slope_filter":  [True, False],
    "exit_mode":         ["alignment_break", "mid_cross"],
}

_KAMA_CACHE = {}


def _get_kama(close_series: pd.Series, length: int) -> np.ndarray:
    if length not in _KAMA_CACHE:
        _KAMA_CACHE[length] = calc_kama(close_series, length=length).shift(1).values
    return _KAMA_CACHE[length]


def generate_signals(
    df: pd.DataFrame,
    ribbon: str = "3L_8_21_55",
    cooldown: int = 5,
    adx_gate: int = 0,
    use_spread_filter: bool = False,
    use_slope_filter: bool = True,
    exit_mode: str = "alignment_break",
) -> pd.DataFrame:
    n = len(df)
    lengths = RIBBONS[ribbon]

    brick_up = df["brick_up"].values
    close = df["Close"]
    adx = df["adx"].values  # pre-shifted by indicators.py

    # KAMA arrays (pre-shifted)
    kama_fast = _get_kama(close, lengths[0])
    kama_mid  = _get_kama(close, lengths[1])
    kama_slow = _get_kama(close, lengths[2])

    # Precompute alignment
    bull_align = (kama_fast > kama_mid) & (kama_mid > kama_slow)
    bear_align = (kama_fast < kama_mid) & (kama_mid < kama_slow)

    # NaN mask
    any_nan = (np.isnan(kama_fast) | np.isnan(kama_mid) | np.isnan(kama_slow))
    bull_align &= ~any_nan
    bear_align &= ~any_nan

    # Ribbon spread: distance between fastest and slowest KAMA
    spread = np.abs(kama_fast - kama_slow)

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    last_trade_bar = -999_999
    warmup = max(lengths) + 5

    for i in range(warmup, n):
        b_up = bool(brick_up[i])
        bull = bool(bull_align[i])
        bear = bool(bear_align[i])

        # ── Exits ───────────────────────────────────────────────────────
        if exit_mode == "mid_cross":
            # Softer exit: price crosses mid-KAMA against position
            if not np.isnan(kama_mid[i]):
                close_val = close.iloc[i]
                long_exit[i]  = close_val < kama_mid[i]
                short_exit[i] = close_val > kama_mid[i]
        else:
            # Original: alignment break OR opposing brick
            long_exit[i]  = not bull or not b_up
            short_exit[i] = not bear or b_up

        # ── Entries ─────────────────────────────────────────────────────
        can_trade = (i - last_trade_bar) >= cooldown
        if not can_trade:
            continue

        bull_prev = bool(bull_align[i - 1])
        bear_prev = bool(bear_align[i - 1])

        # New alignment trigger
        long_trigger  = bull and not bull_prev and b_up
        short_trigger = bear and not bear_prev and not b_up

        if not long_trigger and not short_trigger:
            continue

        # ── Quality gates ───────────────────────────────────────────────

        # ADX gate
        if adx_gate > 0 and not np.isnan(adx[i]):
            if adx[i] < adx_gate:
                continue

        # Spread expanding filter
        if use_spread_filter and i >= 2:
            if np.isnan(spread[i]) or np.isnan(spread[i - 1]):
                continue
            if spread[i] <= spread[i - 1]:
                continue  # ribbon contracting, skip

        # KAMA slope filter — fastest KAMA must slope in trade direction
        if use_slope_filter and i >= 1:
            if np.isnan(kama_fast[i]) or np.isnan(kama_fast[i - 1]):
                continue
            slope = kama_fast[i] - kama_fast[i - 1]
            if long_trigger and slope <= 0:
                continue
            if short_trigger and slope >= 0:
                continue

        if long_trigger:
            long_entry[i] = True
            last_trade_bar = i
        elif short_trigger:
            short_entry[i] = True
            last_trade_bar = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
