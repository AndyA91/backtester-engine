"""
R025: KAMA Ribbon resume_1 + selectivity gates

Builds on R024 resume_1 (one re-entry per trend) and adds the same
Renko-native gates that make the live portfolio strategies selective:

  1. Session filter — skip Asian session (low-momentum noise)
  2. Volume ratio cap — skip volume spikes (news/erratic bars)
  3. Supertrend confirmation — st_dir must agree with trade direction
  4. Choppiness filter — skip ranging markets (chop > threshold)

All indicators are already computed by add_renko_indicators() and
pre-shifted to prevent lookahead bias.
"""

import numpy as np
import pandas as pd

from indicators.kama import calc_kama

DESCRIPTION = "KAMA ribbon resume_1 + session/volume/supertrend/chop gates"

HYPOTHESIS = (
    "R024 resume_1 has edge (PF 5-14 OOS) but takes too many low-quality trades. "
    "The live portfolio strategies achieve 3-8x higher PF through gate stacking. "
    "Adding session, volume, supertrend, and choppiness filters should cut noise "
    "trades while preserving the core KAMA ribbon signal."
)

RIBBONS = {
    "3L_8_21_55":   (8, 21, 55),
    "3L_5_13_34":   (5, 13, 34),
}

PARAM_GRID = {
    "ribbon":        list(RIBBONS.keys()),
    "cooldown":      [3, 5],
    "adx_gate":      [0, 25],
    "session_start": [0, 13],          # 0=all hours, 13=London/NY only
    "vol_max":       [0, 1.5],         # 0=no filter, 1.5=skip spikes
    "st_gate":       [False, True],    # require supertrend agreement
    "chop_max":      [0, 50],          # 0=no filter, 50=skip choppy
}
# 2 × 2 × 2 × 2 × 2 × 2 × 2 = 128 combos

_KAMA_CACHE = {}


def _get_kama(close_series: pd.Series, length: int) -> np.ndarray:
    if length not in _KAMA_CACHE:
        _KAMA_CACHE[length] = calc_kama(close_series, length=length).shift(1).values
    return _KAMA_CACHE[length]


def generate_signals(
    df: pd.DataFrame,
    ribbon: str = "3L_5_13_34",
    cooldown: int = 3,
    adx_gate: int = 25,
    session_start: int = 13,
    vol_max: float = 1.5,
    st_gate: bool = True,
    chop_max: float = 0,
) -> pd.DataFrame:
    n = len(df)
    lengths = RIBBONS[ribbon]

    brick_up = df["brick_up"].values
    adx = df["adx"].values

    # Pre-extract gate arrays
    entry_hours = df.index.hour
    vol_ratio = df["vol_ratio"].values if "vol_ratio" in df.columns else np.zeros(n)
    st_dir = df["st_dir"].values if "st_dir" in df.columns else np.ones(n)
    chop = df["chop"].values if "chop" in df.columns else np.zeros(n)

    kama_fast = _get_kama(df["Close"], lengths[0])
    kama_mid  = _get_kama(df["Close"], lengths[1])
    kama_slow = _get_kama(df["Close"], lengths[2])

    any_nan = np.isnan(kama_fast) | np.isnan(kama_mid) | np.isnan(kama_slow)
    bull_align = (kama_fast > kama_mid) & (kama_mid > kama_slow) & ~any_nan
    bear_align = (kama_fast < kama_mid) & (kama_mid < kama_slow) & ~any_nan

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    last_trade_bar = -999_999
    warmup = max(lengths) + 5

    # Track resume count per trend (resume_1: max 1 re-entry)
    resume_used_long = 0
    resume_used_short = 0

    for i in range(warmup, n):
        b_up = bool(brick_up[i])
        bull = bool(bull_align[i])
        bear = bool(bear_align[i])

        # ── Track alignment breaks to reset resume counter ────────────
        if not bull:
            resume_used_long = 0
        if not bear:
            resume_used_short = 0

        # ── Exits: opposing brick OR alignment break ──────────────────
        long_exit[i]  = not bull or not b_up
        short_exit[i] = not bear or b_up

        # ── Cooldown ──────────────────────────────────────────────────
        can_trade = (i - last_trade_bar) >= cooldown
        if not can_trade:
            continue

        bull_prev = bool(bull_align[i - 1])
        bear_prev = bool(bear_align[i - 1])

        # Fresh alignment trigger
        fresh_long  = bull and not bull_prev and b_up
        fresh_short = bear and not bear_prev and not b_up

        # Resume trigger (max 1 per trend)
        resume_long  = bull and bull_prev and b_up and (i >= 1 and not brick_up[i - 1]) and resume_used_long < 1
        resume_short = bear and bear_prev and not b_up and (i >= 1 and brick_up[i - 1]) and resume_used_short < 1

        long_trigger  = fresh_long or resume_long
        short_trigger = fresh_short or resume_short

        if not long_trigger and not short_trigger:
            continue

        # ── Gate stack ────────────────────────────────────────────────

        # ADX gate
        if adx_gate > 0 and not np.isnan(adx[i]) and adx[i] < adx_gate:
            continue

        # Session filter
        if session_start > 0 and entry_hours[i] < session_start:
            continue

        # Volume spike filter
        if vol_max > 0 and not np.isnan(vol_ratio[i]) and vol_ratio[i] > vol_max:
            continue

        # Choppiness filter
        if chop_max > 0 and not np.isnan(chop[i]) and chop[i] > chop_max:
            continue

        # Supertrend direction gate
        if st_gate and not np.isnan(st_dir[i]):
            if long_trigger and st_dir[i] < 0:
                continue  # ST bearish, skip long
            if short_trigger and st_dir[i] > 0:
                continue  # ST bullish, skip short

        # ── Fire entry ────────────────────────────────────────────────
        if long_trigger:
            long_entry[i] = True
            last_trade_bar = i
            if not fresh_long:
                resume_used_long += 1
            else:
                resume_used_long = 0
        elif short_trigger:
            short_entry[i] = True
            last_trade_bar = i
            if not fresh_short:
                resume_used_short += 1
            else:
                resume_used_short = 0

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
