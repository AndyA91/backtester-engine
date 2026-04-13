"""
R030: KAMA Ribbon 3-Line (stripped) — exact match to kama_ribbon.pine

Mirrors the minimal Pine version built interactively:
  - Exactly 3 KAMA lines (fast / mid / slow)
  - Entry: first bar the ribbon enters full alignment (fast>mid>slow = long)
  - Exit:  ribbon loses full alignment ("gray exit"). NO opposing-brick exit,
           NO cooldown, NO brick-direction requirement.
  - All three KAMAs share the same fast_sc / slow_sc constants (default 2/30)

This is DIFFERENT from R020:
  R020 exits on alignment_break OR opposing_brick and has cooldown +
  require_brick_dir toggles. R030 is the pure "ribbon-only" baseline for
  isolating whether the KAMA alignment signal alone has edge on a new
  instrument (EURAUD 0.0006 is the first target).
"""

import numpy as np
import pandas as pd

from indicators.kama import calc_kama

DESCRIPTION = "3-line KAMA ribbon, flip-entry, gray-exit only (Pine-equivalent)"

HYPOTHESIS = (
    "A 3-line KAMA ribbon aligning fully (fast>mid>slow or vice versa) is a "
    "visual trend-confirmation that only fires in trending conditions. Exiting "
    "the moment the ribbon loses full alignment should capture the bulk of the "
    "trending move and step aside during chop, without the extra rigor of a "
    "brick-flip exit. Tested first on EURAUD 0.0006."
)

PARAM_GRID = {
    "fast_len": [5, 8, 10, 13],
    "mid_len":  [13, 20, 21, 30],
    "slow_len": [30, 34, 55, 80],
    "fast_sc":  [2],
    "slow_sc":  [30],
}

# Cache across sweep combos in the same worker
_KAMA_CACHE = {}


def _get_kama(close_series: pd.Series, length: int, fast: int, slow: int) -> np.ndarray:
    """Compute KAMA once per (length, fast, slow), cache, return pre-shifted array."""
    key = (length, fast, slow)
    if key not in _KAMA_CACHE:
        # Shift by 1 so value at [i] is through bar i-1 (no lookahead)
        _KAMA_CACHE[key] = calc_kama(close_series, length=length, fast=fast, slow=slow).shift(1).values
    return _KAMA_CACHE[key]


def generate_signals(
    df: pd.DataFrame,
    fast_len: int = 10,
    mid_len: int = 20,
    slow_len: int = 30,
    fast_sc: int = 2,
    slow_sc: int = 30,
) -> pd.DataFrame:
    """
    3-line KAMA ribbon — flip-entry, gray-exit only.

    Invalid grid points (where lengths are not strictly increasing) short-circuit
    to an empty signal set so the sweep runner can still rank them.
    """
    n = len(df)
    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    # Guard: require strictly fast < mid < slow
    if not (fast_len < mid_len < slow_len):
        df["long_entry"]  = long_entry
        df["long_exit"]   = long_exit
        df["short_entry"] = short_entry
        df["short_exit"]  = short_exit
        return df

    close = df["Close"]
    k_fast = _get_kama(close, fast_len, fast_sc, slow_sc)
    k_mid  = _get_kama(close, mid_len,  fast_sc, slow_sc)
    k_slow = _get_kama(close, slow_len, fast_sc, slow_sc)

    any_nan    = np.isnan(k_fast) | np.isnan(k_mid) | np.isnan(k_slow)
    bull_align = (k_fast > k_mid) & (k_mid > k_slow) & ~any_nan
    bear_align = (k_fast < k_mid) & (k_mid < k_slow) & ~any_nan

    # Stateless signal generation — let the engine manage position.
    # This satisfies L14 (no pos = 0/+1/-1 flag in the generator) and L2
    # (no state mutations to desync with bar_in_range).
    bull_prev = np.roll(bull_align, 1)
    bear_prev = np.roll(bear_align, 1)
    bull_prev[0] = False
    bear_prev[0] = False

    # Zero out warmup bars so KAMA seed region doesn't produce spurious flips.
    warmup = slow_len + 5
    mask = np.zeros(n, dtype=bool)
    mask[warmup:] = True

    long_entry  = (bull_align & ~bull_prev) & mask
    short_entry = (bear_align & ~bear_prev) & mask
    long_exit   = (~bull_align) & mask
    short_exit  = (~bear_align) & mask

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
