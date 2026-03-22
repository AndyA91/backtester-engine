"""
EA001: EURAUD baseline port of R007 combined.

Direct port of renko/strategies/r007_combined.py with only module-level
constants changed for the EURAUD dataset and cost assumptions.
"""

import numpy as np
import pandas as pd

DESCRIPTION = "EURAUD R001+R002 combined baseline (no gates)"

HYPOTHESIS = (
    "R001+R002 edge proven on EURUSD/GBPJPY should transfer to EURAUD cross pair"
)

RENKO_FILE = "OANDA_EURAUD, 1S renko 0.0006.csv"
COMMISSION_PCT = 0.009
INITIAL_CAPITAL = 1000.0

PARAM_GRID = {
    "n_bricks": [2, 3, 4, 5],
    "cooldown": [10, 20, 30],
}


def generate_signals(
    df: pd.DataFrame,
    n_bricks: int = 3,
    cooldown: int = 10,
) -> pd.DataFrame:
    """
    Generate dual-entry signals from R001 + R002 with explicit position tracking.

    The generator tracks position state internally so:
      - Entries only fire when flat (no pyramiding, no signal-while-in-trade noise)
      - R001 cooldown is measured from last R001 entry only (R002 chains are exempt)
      - R002 always takes priority over R001 when both would apply

    Args:
        df:       Renko DataFrame with brick_up bool column.
        n_bricks: Consecutive bricks for R001 signal; also N for R002 lookback.
        cooldown: Minimum bricks between R001 entries. R002 entries are exempt.

    Returns:
        df with columns long_entry, long_exit, short_entry, short_exit (bool).
    """
    n = len(df)
    brick_up = df["brick_up"].values

    long_entry = np.zeros(n, dtype=bool)
    long_exit = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit = np.zeros(n, dtype=bool)

    in_position = False
    trade_dir = 0
    last_r001_bar = -999_999

    warmup = max(n_bricks + 1, 30)

    for i in range(warmup, n):
        up = bool(brick_up[i])

        long_exit[i] = not up
        short_exit[i] = up

        if in_position:
            if (trade_dir == 1 and not up) or (trade_dir == -1 and up):
                in_position = False
                trade_dir = 0

        if in_position:
            continue

        prev = brick_up[i - n_bricks : i]
        prev_all_up = bool(np.all(prev))
        prev_all_down = bool(not np.any(prev))

        if prev_all_up and not up:
            short_entry[i] = True
            in_position = True
            trade_dir = -1

        elif prev_all_down and up:
            long_entry[i] = True
            in_position = True
            trade_dir = 1

        elif (i - last_r001_bar) >= cooldown:
            window = brick_up[i - n_bricks + 1 : i + 1]
            all_up = bool(np.all(window))
            all_down = bool(not np.any(window))

            if all_up:
                long_entry[i] = True
                in_position = True
                trade_dir = 1
                last_r001_bar = i

            elif all_down:
                short_entry[i] = True
                in_position = True
                trade_dir = -1
                last_r001_bar = i

    df["long_entry"] = long_entry
    df["long_exit"] = long_exit
    df["short_entry"] = short_entry
    df["short_exit"] = short_exit
    return df
