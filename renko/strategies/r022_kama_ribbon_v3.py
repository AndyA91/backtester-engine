"""
R022: KAMA Ribbon v3 — Targeting the specific loss patterns from v1/v2

Loss analysis from v1/v2:
  1. Whipsaw exits: alignment forms for 1-2 bricks then breaks → entry fills, exit
     triggers immediately → small loss + commission drag
  2. Single opposing brick kills position: even in strong trends, one counter-brick
     triggers `not b_up` exit → premature exit during pullbacks
  3. No profit protection: rides winners until alignment fully breaks, giving back
     gains on the exit bricks

Improvements:
  1. Exit grace period — require N consecutive misaligned bricks before exiting
     (tolerates brief pullbacks within trends)
  2. ATR-based trailing stop — protects profits dynamically
  3. Brick streak entry filter — require N same-direction bricks before entry
     (avoids entries in alternating/choppy brick patterns)
  4. Choppiness Index gate — skip entries when CHOP > threshold (>61.8 = ranging)
  5. Volume confirmation — require vol_ratio > threshold on entry brick

Base: Best from v2 (3-line ribbon, ADX≥25, alignment_break exit).
"""

import numpy as np
import pandas as pd

from indicators.kama import calc_kama

DESCRIPTION = "KAMA ribbon v3 — delayed exit, ATR trailing, streak/chop filters"

HYPOTHESIS = (
    "V2 losses come from whipsaw entries (alignment flickers) and premature exits "
    "(single opposing brick during trends). Delayed exit tolerates brief pullbacks. "
    "ATR trailing stop locks in profits. Brick streak and chop filters prevent entries "
    "during alternating brick patterns that cause whipsaws."
)

RIBBONS = {
    "3L_8_21_55":   (8, 21, 55),
    "3L_5_13_34":   (5, 13, 34),
}

PARAM_GRID = {
    "ribbon":           list(RIBBONS.keys()),
    "cooldown":         [5],
    "adx_gate":         [25],
    "exit_grace":       [0, 2, 3],          # bricks of misalignment before exit
    "atr_trail_mult":   [0, 2.0, 3.0],      # 0 = no trail, N = N*ATR trailing stop
    "brick_streak":     [0, 2],              # require N same-dir bricks at entry
    "chop_gate":        [0, 55],             # 0 = off, N = skip if chop > N
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
    adx_gate: int = 25,
    exit_grace: int = 2,
    atr_trail_mult: float = 0,
    brick_streak: int = 0,
    chop_gate: int = 0,
) -> pd.DataFrame:
    n = len(df)
    lengths = RIBBONS[ribbon]

    brick_up = df["brick_up"].values
    close = df["Close"].values
    adx = df["adx"].values
    atr = df["atr"].values
    chop = df["chop"].values

    # KAMA arrays (pre-shifted)
    kama_fast = _get_kama(df["Close"], lengths[0])
    kama_mid  = _get_kama(df["Close"], lengths[1])
    kama_slow = _get_kama(df["Close"], lengths[2])

    # Precompute alignment
    any_nan = np.isnan(kama_fast) | np.isnan(kama_mid) | np.isnan(kama_slow)
    bull_align = (kama_fast > kama_mid) & (kama_mid > kama_slow) & ~any_nan
    bear_align = (kama_fast < kama_mid) & (kama_mid < kama_slow) & ~any_nan

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    # For ATR trailing stop
    sl_offset = np.zeros(n, dtype=np.float64)

    last_trade_bar = -999_999
    misalign_count_long = 0
    misalign_count_short = 0
    warmup = max(lengths) + 5

    for i in range(warmup, n):
        b_up = bool(brick_up[i])
        bull = bool(bull_align[i])
        bear = bool(bear_align[i])

        # ── Exits with grace period ─────────────────────────────────────
        # Long exit: track consecutive misaligned bricks
        if not bull or not b_up:
            misalign_count_long += 1
        else:
            misalign_count_long = 0

        if not bear or b_up:
            misalign_count_short += 1
        else:
            misalign_count_short = 0

        long_exit[i]  = misalign_count_long > exit_grace
        short_exit[i] = misalign_count_short > exit_grace

        # ── Entries ─────────────────────────────────────────────────────
        can_trade = (i - last_trade_bar) >= cooldown
        if not can_trade:
            continue

        bull_prev = bool(bull_align[i - 1])
        bear_prev = bool(bear_align[i - 1])

        # New alignment trigger + brick direction match
        long_trigger  = bull and not bull_prev and b_up
        short_trigger = bear and not bear_prev and not b_up

        if not long_trigger and not short_trigger:
            continue

        # ── Quality gates ───────────────────────────────────────────────

        # ADX gate
        if not np.isnan(adx[i]) and adx[i] < adx_gate:
            continue

        # Choppiness gate — skip if market is too choppy
        if chop_gate > 0 and not np.isnan(chop[i]):
            if chop[i] > chop_gate:
                continue

        # Brick streak — require N consecutive same-direction bricks
        if brick_streak > 0:
            streak_ok = True
            for k in range(1, brick_streak + 1):
                if i - k < 0:
                    streak_ok = False
                    break
                if long_trigger and not brick_up[i - k]:
                    streak_ok = False
                    break
                if short_trigger and brick_up[i - k]:
                    streak_ok = False
                    break
            if not streak_ok:
                continue

        if long_trigger:
            long_entry[i] = True
            last_trade_bar = i
            misalign_count_long = 0
            # ATR trailing stop
            if atr_trail_mult > 0 and not np.isnan(atr[i]):
                sl_offset[i] = atr[i] * atr_trail_mult
        elif short_trigger:
            short_entry[i] = True
            last_trade_bar = i
            misalign_count_short = 0
            if atr_trail_mult > 0 and not np.isnan(atr[i]):
                sl_offset[i] = atr[i] * atr_trail_mult

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit

    if atr_trail_mult > 0:
        df["sl_offset"] = sl_offset

    return df
