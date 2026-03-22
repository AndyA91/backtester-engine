"""EA017: EURAUD R007 baseline on 0.0007 brick size.

Direct port of ea001_baseline.py onto the new 0.0007 brick CSV.

Key differences vs EA001 (0.0006):
  - Larger brick: 0.0007 (+16.7%) → ~14% fewer bricks/year, less micro-noise.
  - Extended IS: 2023-01-01 (vs EA001's 2023-07-20) — uses the full available
    history from the 0.0007 CSV which starts Dec 2022.

IS:  2023-01-01 → 2025-09-30
OOS: 2025-10-01 → 2026-03-18
"""

import numpy as np
import pandas as pd

DESCRIPTION = "EURAUD R001+R002 combined baseline on 0.0007 brick size (no gates)"

HYPOTHESIS = (
    "Larger 0.0007 brick size generates ~14% fewer bricks/year, filtering micro-noise. "
    "Extended IS from Jan 2023 adds ~8 months of training data vs EA001. "
    "Hypothesis: cleaner signal per brick improves OOS stability of the R007 base edge."
)

RENKO_FILE      = "OANDA_EURAUD, 1S renko 0.0007.csv"
COMMISSION_PCT  = 0.009
INITIAL_CAPITAL = 1000.0

PARAM_GRID = {
    "n_bricks": [2, 3, 4, 5],
    "cooldown":  [10, 20, 30],
}
# 4 × 3 = 12 combos


def generate_signals(
    df: pd.DataFrame,
    n_bricks: int = 3,
    cooldown: int = 10,
) -> pd.DataFrame:
    """
    Generate dual-entry signals from R001 + R002 with explicit position tracking.

    Args:
        df:       Renko DataFrame with brick_up bool column.
        n_bricks: Consecutive bricks for R001 signal; also N for R002 lookback.
        cooldown: Minimum bricks between R001 entries. R002 entries are exempt.

    Returns:
        df with columns long_entry, long_exit, short_entry, short_exit (bool).
    """
    n = len(df)
    brick_up = df["brick_up"].values

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    in_position   = False
    trade_dir     = 0
    last_r001_bar = -999_999

    warmup = max(n_bricks + 1, 30)

    for i in range(warmup, n):
        up = bool(brick_up[i])

        long_exit[i]  = not up
        short_exit[i] = up

        if in_position:
            if (trade_dir == 1 and not up) or (trade_dir == -1 and up):
                in_position = False
                trade_dir   = 0

        if in_position:
            continue

        prev          = brick_up[i - n_bricks : i]
        prev_all_up   = bool(np.all(prev))
        prev_all_down = bool(not np.any(prev))

        if prev_all_up and not up:
            short_entry[i] = True
            in_position    = True
            trade_dir      = -1

        elif prev_all_down and up:
            long_entry[i] = True
            in_position   = True
            trade_dir     = 1

        elif (i - last_r001_bar) >= cooldown:
            window   = brick_up[i - n_bricks + 1 : i + 1]
            all_up   = bool(np.all(window))
            all_down = bool(not np.any(window))

            if all_up:
                long_entry[i]  = True
                in_position    = True
                trade_dir      = 1
                last_r001_bar  = i

            elif all_down:
                short_entry[i] = True
                in_position    = True
                trade_dir      = -1
                last_r001_bar  = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
