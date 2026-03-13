"""
R002: Brick Count Reversal

After N consecutive same-direction bricks, the FIRST opposing brick
triggers a counter-trend entry. Pure Renko reversal — zero external
indicators. Tests whether Renko exhaustion signals are predictive.

Compare directly with R001: R001 enters WITH the N-brick momentum;
R002 enters AGAINST it on the first reversal brick.
"""

import numpy as np
import pandas as pd

DESCRIPTION = "N consecutive bricks then first opposing brick -> counter-trend entry"

HYPOTHESIS = (
    "N consecutive same-direction bricks represent a local exhaustion point on Renko. "
    "The first opposing brick after N-in-a-row may signal a genuine reversal rather "
    "than a continuation. This is the counter-trend complement to R001 — if R001 edge "
    "comes from momentum, R002 tests whether the exhaustion/reversal is equally predictive."
)

PARAM_GRID = {
    "n_bricks": [2, 3, 4, 5],
    "cooldown": [5, 10, 20, 30],
}


def generate_signals(
    df: pd.DataFrame,
    n_bricks: int = 3,
    cooldown: int = 10,
) -> pd.DataFrame:
    """
    Generate reversal signals from brick count exhaustion.

    Entry: after N consecutive same-direction bricks, first opposing brick
           triggers a trade in the OPPOSITE direction (counter-trend).
    Exit:  First brick opposing the open position.
    Cooldown: minimum bricks between entries.

    Args:
        df: Renko DataFrame with brick_up bool column.
        n_bricks: Consecutive bricks required before reversal entry.
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
    # warmup: need n_bricks of history before current bar
    warmup = n_bricks

    for i in range(warmup, n):
        # Check the N bricks BEFORE the current bar
        prev_window   = brick_up[i - n_bricks : i]  # length = n_bricks, excludes bar i
        prev_all_up   = bool(np.all(prev_window))
        prev_all_down = bool(np.not_equal(prev_window, True).all())

        # Exit: first opposing brick — unconditional, no cooldown
        long_exit[i]  = not brick_up[i]
        short_exit[i] = brick_up[i]

        can_trade = (i - last_trade_bar) >= cooldown
        if can_trade:
            # N up bricks, then first DOWN brick → short (reversal)
            if prev_all_up and not brick_up[i]:
                short_entry[i] = True
                last_trade_bar = i
            # N down bricks, then first UP brick → long (reversal)
            elif prev_all_down and brick_up[i]:
                long_entry[i] = True
                last_trade_bar = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
