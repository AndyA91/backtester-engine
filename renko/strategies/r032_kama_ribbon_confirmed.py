"""
R032: KAMA Ribbon 5/13/30 + Confirmation Delay (± ADX>=20 gate)

Follow-up to R031. R030 bars-held diagnostic showed:
  - Winners median hold = 17 bricks
  - Losers median hold = 9 bricks
  - Losers are "alignment wobbles in chop" — brief flips that reverse fast

Hypothesis: waiting N bricks of CONTINUED alignment before entering filters
out the chop-wobble losers (median ~2-3 bar lifespan) while barely denting
real-trend winners (median 17 bar lifespan). Asymmetric delay — entry is
gated, exit remains instantaneous (first non-aligned brick).

Params:
  - confirm_bars: N ∈ [1, 2, 3, 5, 8]
      N=1 = no delay (reproduces R031/R030 behavior)
      N=2 = require alignment to persist 1 extra brick
      N=8 = require alignment to persist 7 extra bricks (likely too strict)
  - use_adx_gate: bool — whether to also require adx[i] >= 20
      False × N=1 reproduces R030 no-gate
      True  × N=1 reproduces R031 adx>=20 (HOLDOUT PF=2.12 winner)

Ribbon LOCKED to R030 best: fast=5, mid=13, slow=30, fast_sc=2, slow_sc=30.
"""

import numpy as np
import pandas as pd

from indicators.kama import calc_kama

DESCRIPTION = "KAMA ribbon 5/13/30 + confirmation delay (± adx>=20)"

HYPOTHESIS = (
    "Chop losers die in ~2-3 bricks; real-trend winners live ~17 bricks. "
    "Requiring N bricks of continued alignment before entry should filter "
    "chop while preserving trends, provided N sits in the gap (2-5)."
)

FAST_LEN = 5
MID_LEN  = 13
SLOW_LEN = 30
FAST_SC  = 2
SLOW_SC  = 30

PARAM_GRID = {
    "confirm_bars": [1, 2, 3, 5, 8],
    "use_adx_gate": [False, True],
}

_KAMA_CACHE = {}


def _get_kama(close_series: pd.Series, length: int, fast: int, slow: int) -> np.ndarray:
    key = (length, fast, slow)
    if key not in _KAMA_CACHE:
        _KAMA_CACHE[key] = calc_kama(close_series, length=length, fast=fast, slow=slow).shift(1).values
    return _KAMA_CACHE[key]


def _consecutive_true_count(mask: np.ndarray) -> np.ndarray:
    """
    For each index i, return the number of consecutive True values in `mask`
    ending at i (inclusive). Resets to 0 on False. Causal, no look-ahead.
    """
    n = len(mask)
    out = np.zeros(n, dtype=int)
    run = 0
    for i in range(n):
        if mask[i]:
            run += 1
        else:
            run = 0
        out[i] = run
    return out


def generate_signals(
    df: pd.DataFrame,
    confirm_bars: int = 1,
    use_adx_gate: bool = True,
) -> pd.DataFrame:
    """
    R032 — KAMA ribbon with N-bar confirmation delay on entry.

    Entry fires on bar i when bull_align has been True for EXACTLY `confirm_bars`
    consecutive bars ending at i — i.e. bar i is the Nth bar of the alignment
    run. This means:
      - confirm_bars=1: fires on first aligned bar (same as R031)
      - confirm_bars=3: fires on 3rd aligned bar (delay of 2 bricks after flip)

    Stateless — engine manages position. Exits on first non-aligned bar.
    """
    n = len(df)
    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    close = df["Close"]
    k_fast = _get_kama(close, FAST_LEN, FAST_SC, SLOW_SC)
    k_mid  = _get_kama(close, MID_LEN,  FAST_SC, SLOW_SC)
    k_slow = _get_kama(close, SLOW_LEN, FAST_SC, SLOW_SC)

    any_nan    = np.isnan(k_fast) | np.isnan(k_mid) | np.isnan(k_slow)
    bull_align = (k_fast > k_mid) & (k_mid > k_slow) & ~any_nan
    bear_align = (k_fast < k_mid) & (k_mid < k_slow) & ~any_nan

    # Count consecutive-True runs ending at each bar
    bull_run = _consecutive_true_count(bull_align)
    bear_run = _consecutive_true_count(bear_align)

    # Entry fires on the EXACT Nth consecutive aligned bar (one-shot per run)
    raw_long_entry  = bull_run == confirm_bars
    raw_short_entry = bear_run == confirm_bars

    # Optional ADX gate (locked threshold from R031 winner)
    if use_adx_gate:
        adx = df["adx"].values
        adx_ok = (~np.isnan(adx)) & (adx >= 20.0)
    else:
        adx_ok = np.ones(n, dtype=bool)

    warmup = SLOW_LEN + 5
    mask = np.zeros(n, dtype=bool)
    mask[warmup:] = True

    long_entry  = raw_long_entry  & adx_ok & mask
    short_entry = raw_short_entry & adx_ok & mask
    long_exit   = (~bull_align) & mask
    short_exit  = (~bear_align) & mask

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
