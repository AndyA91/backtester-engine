"""
R020: KAMA Ribbon — Multiple Kaufman Adaptive MAs forming a trend ribbon on Renko

Signal logic:
    N KAMA lines at increasing ER lengths (fast→slow).
    - Bullish alignment: KAMA_fast > KAMA_mid > ... > KAMA_slow (all ordered)
    - Bearish alignment: KAMA_fast < KAMA_mid < ... < KAMA_slow
    Entry: first brick where full alignment appears AND brick matches direction.
    Exit: alignment breaks OR opposing brick (whichever first).

The ribbon acts as an adaptive trend filter — in trending markets, all KAMAs
track price tightly and fan out in order. In choppy markets, KAMAs converge
and scramble, preventing entries.
"""

import numpy as np
import pandas as pd

from indicators.kama import calc_kama

DESCRIPTION = "KAMA ribbon alignment — multi-length adaptive MA trend filter"

HYPOTHESIS = (
    "Kaufman's Adaptive MA adjusts speed based on market efficiency. "
    "Multiple KAMAs at different lengths create a ribbon that fans out in "
    "trending conditions (high ER) and contracts in chop (low ER). "
    "Full ribbon alignment is a strong trend confirmation — on Renko, "
    "where noise is already reduced, this should produce high-quality entries."
)

# ── Ribbon definitions: each is a tuple of KAMA ER lengths (fast→slow) ────────
RIBBONS = {
    "3L_8_21_55":       (8, 21, 55),
    "3L_5_13_34":       (5, 13, 34),
    "4L_5_13_21_55":    (5, 13, 21, 55),
    "4L_8_13_21_34":    (8, 13, 21, 34),
    "5L_5_8_13_21_34":  (5, 8, 13, 21, 34),
    "3L_10_30_60":      (10, 30, 60),
}

PARAM_GRID = {
    "ribbon": list(RIBBONS.keys()),
    "cooldown": [3, 8, 15],
    "require_brick_dir": [True, False],
}

# ── Module-level KAMA cache (survives across param combos in same sweep) ──────
_KAMA_CACHE = {}


def _get_kama(close_series: pd.Series, length: int) -> np.ndarray:
    """Compute KAMA once per length, cache for reuse."""
    if length not in _KAMA_CACHE:
        # Shift by 1 so value at [i] is through bar i-1 (no lookahead)
        _KAMA_CACHE[length] = calc_kama(close_series, length=length).shift(1).values
    return _KAMA_CACHE[length]


def generate_signals(
    df: pd.DataFrame,
    ribbon: str = "3L_8_21_55",
    cooldown: int = 8,
    require_brick_dir: bool = True,
) -> pd.DataFrame:
    """
    KAMA Ribbon alignment on Renko bricks.

    Args:
        df: Renko DataFrame with brick_up + pre-shifted indicators.
        ribbon: Key into RIBBONS dict — defines the KAMA ER lengths.
        cooldown: Minimum bricks between trades.
        require_brick_dir: If True, long entry only on up brick, short on down brick.

    Returns:
        df with long_entry, long_exit, short_entry, short_exit (bool).
    """
    n = len(df)
    lengths = RIBBONS[ribbon]
    num_lines = len(lengths)

    brick_up = df["brick_up"].values
    close = df["Close"]

    # Get KAMA arrays (pre-shifted, no lookahead)
    kama_arrays = [_get_kama(close, l) for l in lengths]

    # Precompute alignment booleans
    bull_align = np.ones(n, dtype=bool)
    bear_align = np.ones(n, dtype=bool)
    any_nan = np.zeros(n, dtype=bool)

    for j in range(num_lines - 1):
        any_nan |= np.isnan(kama_arrays[j]) | np.isnan(kama_arrays[j + 1])
        bull_align &= (kama_arrays[j] > kama_arrays[j + 1])
        bear_align &= (kama_arrays[j] < kama_arrays[j + 1])

    # NaN bars can't have valid alignment
    bull_align &= ~any_nan
    bear_align &= ~any_nan

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

        # ── Exits: alignment breaks OR opposing brick ───────────────────────
        long_exit[i]  = not bull or not b_up
        short_exit[i] = not bear or b_up

        # ── Entries ─────────────────────────────────────────────────────────
        can_trade = (i - last_trade_bar) >= cooldown
        if not can_trade:
            continue

        # First bar of new alignment
        bull_prev = bool(bull_align[i - 1])
        bear_prev = bool(bear_align[i - 1])

        if bull and not bull_prev:
            if (not require_brick_dir) or b_up:
                long_entry[i] = True
                last_trade_bar = i
        elif bear and not bear_prev:
            if (not require_brick_dir) or (not b_up):
                short_entry[i] = True
                last_trade_bar = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
