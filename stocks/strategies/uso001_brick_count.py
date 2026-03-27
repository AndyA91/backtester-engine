"""
USO001: Brick Count (Long Only)

Enter long after N consecutive up bricks, exit on first down brick.
Base strategy for USO gate discovery — same logic as R001 but long-only.
"""

import numpy as np
import pandas as pd

DESCRIPTION = "N consecutive up bricks → long entry, first down brick → exit"

HYPOTHESIS = (
    "Consecutive same-direction bricks signal momentum continuation. "
    "On Renko, N up bricks in a row indicate strong bullish momentum "
    "likely to persist for at least one more brick."
)

PARAM_GRID = {
    "n": [2, 3, 4, 5, 6],
    "cooldown": [3, 5, 10, 15, 20, 30],
}

COMMISSION_PCT = 0.0
INITIAL_CAPITAL = 10000.0
RENKO_FILE = "BATS_USO, 1S renko 0.25.csv"


def generate_signals(
    df: pd.DataFrame,
    n: int = 3,
    cooldown: int = 10,
) -> pd.DataFrame:
    """
    Generate long-only brick count signals.

    Args:
        df: Renko DataFrame with brick_up column + pre-shifted indicators.
        n: Number of consecutive up bricks required for entry.
        cooldown: Minimum bricks between entries.

    Returns:
        df with columns long_entry, long_exit (bool).
    """
    num = len(df)
    brick_up = df["brick_up"].values

    long_entry = np.zeros(num, dtype=bool)
    long_exit  = np.zeros(num, dtype=bool)

    last_trade_bar = -999_999

    for i in range(n, num):
        # Exit: first down brick
        long_exit[i] = not brick_up[i]

        # Entry: N consecutive up bricks + cooldown
        can_trade = (i - last_trade_bar) >= cooldown
        if not can_trade:
            continue

        # Check N consecutive up bricks ending at i
        all_up = True
        for j in range(n):
            if not brick_up[i - j]:
                all_up = False
                break

        if all_up:
            long_entry[i] = True
            last_trade_bar = i

    df["long_entry"] = long_entry
    df["long_exit"]  = long_exit
    return df
