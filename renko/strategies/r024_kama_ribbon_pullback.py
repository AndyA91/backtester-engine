"""
R024: KAMA Ribbon + Pullback Re-entries

Problem with R020-R023: only enters on the FIRST brick of new alignment.
In a long trend, ribbon stays aligned for 100+ bricks but we only trade once.
After exiting on an opposing brick, we sit out the rest of the move.

Solution: Allow re-entries during existing alignment after a brief pullback.

Entry modes:
  1. "fresh" — original: first brick of new alignment only
  2. "resume" — also re-enter when brick direction resumes within existing alignment
     (e.g., aligned bull + down brick exits → next up brick re-enters if still aligned)
  3. "pullback_N" — re-enter after N opposing bricks followed by a confirming brick,
     while ribbon stays aligned (waits for pullback to develop before re-entering)

Exit: opposing brick (but NOT alignment break — stay in if ribbon holds through pullback)
      OR full alignment break (ribbon scrambles)

Key insight: separate "brick direction exit" from "alignment break exit":
  - Brick direction exit = temporary (pullback), re-entry possible
  - Alignment break = structural (trend over), no re-entry
"""

import numpy as np
import pandas as pd

from indicators.kama import calc_kama

DESCRIPTION = "KAMA ribbon with pullback re-entries — multiple bites per trend"

HYPOTHESIS = (
    "R020-R023 only trade the first alignment event per trend. Strong trends persist "
    "for many bricks with the ribbon staying aligned. By re-entering after brief "
    "pullbacks (opposing bricks) within an aligned ribbon, we capture more of the "
    "trend without increasing false signals — the ribbon acts as the quality gate."
)

RIBBONS = {
    "3L_8_21_55":   (8, 21, 55),
    "3L_5_13_34":   (5, 13, 34),
}

PARAM_GRID = {
    "ribbon":           list(RIBBONS.keys()),
    "cooldown":         [3, 5],
    "adx_gate":         [0, 25],
    "entry_mode":       ["fresh_only", "resume", "resume_1", "pullback_2"],
    "exit_on_brick":    [True, False],   # exit on opposing brick, or only on alignment break
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
    entry_mode: str = "resume",
    exit_on_brick: bool = True,
) -> pd.DataFrame:
    n = len(df)
    lengths = RIBBONS[ribbon]

    brick_up = df["brick_up"].values
    adx = df["adx"].values

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

    # Track opposing brick count for pullback mode
    opposing_count_long = 0
    opposing_count_short = 0

    # Track resume count per trend for resume_1 mode
    resume_used_long = 0
    resume_used_short = 0

    for i in range(warmup, n):
        b_up = bool(brick_up[i])
        bull = bool(bull_align[i])
        bear = bool(bear_align[i])

        # ── Track opposing bricks within alignment ──────────────────────
        if bull and not b_up:
            opposing_count_long += 1
        elif bull and b_up:
            # Brick resumed direction — check for re-entry below
            pass
        else:
            opposing_count_long = 0  # alignment broke, reset
            resume_used_long = 0

        if bear and b_up:
            opposing_count_short += 1
        elif bear and not b_up:
            pass
        else:
            opposing_count_short = 0
            resume_used_short = 0

        # ── Exits ───────────────────────────────────────────────────────
        if exit_on_brick:
            # Exit on opposing brick OR alignment break
            long_exit[i]  = not bull or not b_up
            short_exit[i] = not bear or b_up
        else:
            # Exit ONLY on alignment break (hold through pullbacks)
            long_exit[i]  = not bull
            short_exit[i] = not bear

        # ── Entries ─────────────────────────────────────────────────────
        can_trade = (i - last_trade_bar) >= cooldown
        if not can_trade:
            continue

        bull_prev = bool(bull_align[i - 1])
        bear_prev = bool(bear_align[i - 1])

        # Fresh alignment trigger (original)
        fresh_long  = bull and not bull_prev and b_up
        fresh_short = bear and not bear_prev and not b_up

        # Resume trigger — brick direction resumes within existing alignment
        resume_long  = bull and bull_prev and b_up and (i >= 1 and not brick_up[i - 1])
        resume_short = bear and bear_prev and not b_up and (i >= 1 and brick_up[i - 1])

        # Pullback trigger — after N opposing bricks, direction resumes
        pb_long  = bull and b_up and opposing_count_long == 0 and (i >= 1 and not brick_up[i - 1])
        pb_short = bear and not b_up and opposing_count_short == 0 and (i >= 1 and brick_up[i - 1])

        # Select entry mode
        long_trigger = False
        short_trigger = False

        if entry_mode == "fresh_only":
            long_trigger  = fresh_long
            short_trigger = fresh_short
        elif entry_mode == "resume":
            long_trigger  = fresh_long or resume_long
            short_trigger = fresh_short or resume_short
        elif entry_mode == "resume_1":
            long_trigger  = fresh_long or (resume_long and resume_used_long < 1)
            short_trigger = fresh_short or (resume_short and resume_used_short < 1)
        elif entry_mode == "pullback_2":
            # Re-enter after 2+ opposing bricks then resume
            pb2_long  = bull and b_up and (i >= 1 and not brick_up[i - 1]) and opposing_count_long == 0
            pb2_short = bear and not b_up and (i >= 1 and brick_up[i - 1]) and opposing_count_short == 0
            # Check that there were at least 2 opposing bricks in the pullback
            # We need to look back to count how many opposing bricks preceded this resume
            pb2_ok_long = False
            pb2_ok_short = False
            if pb2_long:
                cnt = 0
                for k in range(1, min(i + 1, 20)):
                    if not brick_up[i - k] and bool(bull_align[i - k]):
                        cnt += 1
                    else:
                        break
                pb2_ok_long = cnt >= 2
            if pb2_short:
                cnt = 0
                for k in range(1, min(i + 1, 20)):
                    if brick_up[i - k] and bool(bear_align[i - k]):
                        cnt += 1
                    else:
                        break
                pb2_ok_short = cnt >= 2

            long_trigger  = fresh_long or (pb2_ok_long)
            short_trigger = fresh_short or (pb2_ok_short)

        if not long_trigger and not short_trigger:
            continue

        # ADX gate
        if adx_gate > 0 and not np.isnan(adx[i]) and adx[i] < adx_gate:
            continue

        # Reset opposing count on entry; track resume usage
        if long_trigger:
            long_entry[i] = True
            last_trade_bar = i
            opposing_count_long = 0
            if fresh_long:
                resume_used_long = 0
            else:
                resume_used_long += 1
        elif short_trigger:
            short_entry[i] = True
            last_trade_bar = i
            opposing_count_short = 0
            if fresh_short:
                resume_used_short = 0
            else:
                resume_used_short += 1

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
