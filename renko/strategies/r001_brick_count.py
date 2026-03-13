"""
R001: N Consecutive Same-Direction Bricks

Pure Renko baseline — zero external indicators. N up bricks in a row triggers
a long entry; N down bricks in a row triggers a short entry. Exit on first
brick in the opposite direction. Cooldown in bricks (not time).

This establishes a baseline trade count and PF before adding filters (ADX, etc.).
"""

import numpy as np
import pandas as pd

DESCRIPTION = "N consecutive same-direction bricks -> entry, first opposing brick -> exit"

HYPOTHESIS = (
    "Consecutive same-direction bricks represent momentum confirmation on Renko. "
    "N bricks in a row means price has moved N × brick_size in one direction without "
    "a reversal, suggesting trend persistence. This is the purest possible Renko "
    "signal — no external indicators, just brick structure."
)

PARAM_GRID = {
    "n_bricks":  [2, 3, 4, 5],
    "cooldown":  [5, 10, 20, 30],
}


def generate_signals(
    df: pd.DataFrame,
    n_bricks: int = 3,
    cooldown: int = 10,
) -> pd.DataFrame:
    """
    Generate long/short entry and exit signals from consecutive brick counts.

    Entry: N consecutive same-direction bricks (long if up, short if down).
    Exit:  First brick in the opposite direction.
    Cooldown: minimum number of bricks between entries.

    Args:
        df: Renko DataFrame with brick_up bool column.
        n_bricks: Number of consecutive bricks required for entry.
        cooldown: Minimum bricks between entries.

    Returns:
        df with columns long_entry, long_exit, short_entry, short_exit (bool).
    """
    n = len(df)
    brick_up = df["brick_up"].values

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    last_trade_bar = -999_999
    warmup = n_bricks  # need at least n_bricks of history

    for i in range(warmup, n):
        # Count consecutive bricks in current direction ending at bar i
        # (look back n_bricks bars including current)
        window = brick_up[i - n_bricks + 1 : i + 1]  # length = n_bricks
        all_up   = bool(np.all(window))
        all_down = bool(np.not_equal(window, True).all())

        # Exit: first brick opposing the direction we entered
        # (unconditional — no cooldown on exits)
        long_exit[i]  = not brick_up[i]
        short_exit[i] = brick_up[i]

        can_trade = (i - last_trade_bar) >= cooldown
        if can_trade:
            if all_up:
                long_entry[i]  = True
                last_trade_bar = i
            elif all_down:
                short_entry[i] = True
                last_trade_bar = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
