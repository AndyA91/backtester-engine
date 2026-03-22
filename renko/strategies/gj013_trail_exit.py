"""
GJ013: N-Brick Trailing Exit — GBPJPY

Variant of GJ001 brick-count with a trailing exit: instead of exiting on the
first opposing brick, require N consecutive opposing bricks before exiting.
This lets trades ride through 1-brick pullbacks (common on Renko).

Entry: N consecutive same-direction bricks (same as GJ001).
Exit:  N consecutive opposing bricks (exit_bricks=1 matches GJ001 baseline).

Dataset  : OANDA_GBPJPY, 1S renko 0.05.csv

GBPJPY engine calibration:
  Commission: 0.005% (same as gj001).
  Initial capital: 150,000 JPY.
"""

import numpy as np
import pandas as pd

RENKO_FILE      = "OANDA_GBPJPY, 1S renko 0.05.csv"
COMMISSION_PCT  = 0.005
INITIAL_CAPITAL = 150_000.0

DESCRIPTION = "Brick-count entry with N-brick trailing exit (GBPJPY)"

HYPOTHESIS = (
    "GJ001 exits on the first opposing brick, but Renko often prints a single "
    "opposing brick before resuming the trend. Requiring 2-3 opposing bricks "
    "before exit should let winning trades run longer, improving avg win size "
    "and expectancy at the cost of slightly worse max drawdown."
)

PARAM_GRID = {
    "n_bricks":    [2, 3, 4, 5],
    "cooldown":    [5, 10, 20, 30],
    "exit_bricks": [1, 2, 3],
}
# 4 x 4 x 3 = 48 combos


def generate_signals(
    df: pd.DataFrame,
    n_bricks: int = 3,
    cooldown: int = 10,
    exit_bricks: int = 1,
) -> pd.DataFrame:
    n = len(df)
    brick_up = df["brick_up"].values

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    in_position    = False
    trade_dir      = 0
    last_trade_bar = -999_999
    opposing_count = 0
    warmup         = n_bricks

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # --- Exit: count consecutive opposing bricks ---
        if in_position:
            is_opposing = (trade_dir == 1 and not up) or (trade_dir == -1 and up)
            if is_opposing:
                opposing_count += 1
            else:
                opposing_count = 0

            if opposing_count >= exit_bricks:
                if trade_dir == 1:
                    long_exit[i] = True
                else:
                    short_exit[i] = True
                in_position    = False
                trade_dir      = 0
                opposing_count = 0

        if in_position:
            continue

        # --- Entry: N consecutive same-direction bricks ---
        can_trade = (i - last_trade_bar) >= cooldown
        if not can_trade:
            continue

        window = brick_up[i - n_bricks + 1 : i + 1]
        all_up   = bool(np.all(window))
        all_down = bool(np.not_equal(window, True).all())

        if all_up:
            long_entry[i]  = True
            in_position    = True
            trade_dir      = 1
            last_trade_bar = i
            opposing_count = 0
        elif all_down:
            short_entry[i] = True
            in_position    = True
            trade_dir      = -1
            last_trade_bar = i
            opposing_count = 0

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
