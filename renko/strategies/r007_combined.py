"""
R007: R001 + R002 Combined System

R001 enters on the N-th consecutive same-direction brick (momentum continuation).
R002 enters on the first brick of a new run, immediately after a quality prior run
ends (run initiation). Together they stay positioned across both phases of each
directional move.

Entry logic (checked each bar when flat):
  1. R002 (priority): prev N bricks all UP, current brick DOWN → SHORT
                      prev N bricks all DOWN, current brick UP  → LONG
     No cooldown — triggered by structure, not time.
  2. R001 (fallback, cooldown gated): current N bricks all UP  → LONG
                                      current N bricks all DOWN → SHORT

Exit logic: first opposing brick (unconditional, same as R001/R002 standalone).

Position state is tracked explicitly in the generator so R001 and R002 never
conflict and the cooldown applies only to R001 re-entries, not R002 chains.

IS:  2023-01-23 -> 2025-09-30  (~2y 9m, same as R001/R002)
OOS: 2025-10-01 -> 2026-03-05  (sealed)
"""

import numpy as np
import pandas as pd

DESCRIPTION = "R001 momentum + R002 initiation — dual-entry combined, position-aware"

HYPOTHESIS = (
    "R001 (IS PF 12-15) and R002 (IS PF 11-14) capture the same Renko edge from "
    "complementary entry angles. R001 enters mid-run on the N-th brick; R002 enters "
    "on the first brick of the next run after N prior bricks. The combined system "
    "stays positioned through both phases, filling the gap between R001 exit and the "
    "next R001 entry. R002 chains naturally break during choppy markets (short runs "
    "< N bricks) — built-in chop filter without external data."
)

PARAM_GRID = {
    "n_bricks": [2, 3, 4, 5],
    "cooldown":  [10, 20, 30],
}
# 4 × 3 = 12 combinations
#
# Compare each combo against standalone baselines (same n, cd):
#   R001 IS: n=2 cd=10 PF 13.40 | n=3 cd=10 PF 12.99
#            n=2 cd=30 PF 15.70 | n=3 cd=30 PF 14.93
#   R002 IS: n=3 cd=10 PF 12.41 | n=3 cd=30 PF 13.23


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
    n        = len(df)
    brick_up = df["brick_up"].values

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    in_position   = False
    trade_dir     = 0          # +1 long, -1 short
    last_r001_bar = -999_999   # cooldown reference for R001 only

    warmup = max(n_bricks + 1, 30)  # n+1 ensures full R002 prev_window exists

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # ── Exit signals (unconditional — engine ignores when flat) ────────
        long_exit[i]  = not up
        short_exit[i] = up

        # ── Update internal position state from exit ───────────────────────
        if in_position:
            if (trade_dir == 1 and not up) or (trade_dir == -1 and up):
                in_position = False
                trade_dir   = 0

        # ── Entry signals (only when flat) ────────────────────────────────
        if in_position:
            continue

        # R002: N bricks immediately before bar i all same direction,
        #       bar i is the first opposing brick → enter in bar i's direction.
        prev = brick_up[i - n_bricks : i]        # length = n_bricks
        prev_all_up   = bool(np.all(prev))
        prev_all_down = bool(not np.any(prev))

        if prev_all_up and not up:
            # N UP bricks then first DOWN → SHORT (R002 initiation)
            short_entry[i] = True
            in_position    = True
            trade_dir      = -1

        elif prev_all_down and up:
            # N DOWN bricks then first UP → LONG (R002 initiation)
            long_entry[i] = True
            in_position   = True
            trade_dir     = 1

        # R001: N consecutive bricks including bar i → entry (cooldown gated).
        # Only reached if R002 condition was not met.
        elif (i - last_r001_bar) >= cooldown:
            window   = brick_up[i - n_bricks + 1 : i + 1]  # length = n_bricks
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
