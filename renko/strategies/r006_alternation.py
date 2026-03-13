"""
R006: R001 + Brick Alternation Filter

N consecutive same-direction Renko bricks -> momentum entry, but ONLY when
the market is NOT in a choppy/alternating state.

Choppiness is measured by counting direction changes (alternations) in the
last `alt_lookback` bricks ending at bar i. If alternations >= max_alternations,
the entry is skipped — the wider context reveals an oscillating market despite
the local N-brick alignment.

Compare with:
  R001 — same signal, no choppiness gate
  R004 — same signal, gated by external 5m candle ADX (R004 covers only 3.5m)
  R006 — same signal, gated by internal Renko alternation count (no external data)

Rationale:
  The trades circled in the TV chart occur during sideways periods where bricks
  flip up-down-up-down. R001's N-brick window is satisfied by coincidence even
  in a choppy market. Counting direction changes in a wider window (e.g. 12
  bricks) cleanly separates trending context (0-2 flips) from choppy context
  (5+ flips). This filter is purely Renko-native — it uses only brick_up history
  with no additional data source.

IS period:  2023-01-23 -> 2025-09-30  (same as R001)
OOS period: 2025-10-01 -> 2026-03-05  (sealed)
"""

import numpy as np
import pandas as pd

DESCRIPTION = "R001 momentum bricks + brick alternation choppiness filter"

HYPOTHESIS = (
    "In choppy markets Renko bricks alternate direction rapidly. Even when N "
    "consecutive same-direction bricks align, the surrounding context (alt_lookback "
    "bricks) shows many direction changes. Requiring alternations < max_alternations "
    "screens out false momentum signals caused by temporary alignment inside a range. "
    "This is Renko-native: zero external data, zero lookahead, works across all "
    "market regimes where Renko data exists."
)

PARAM_GRID = {
    "n_bricks":         [2, 3],
    "cooldown":         [10, 20, 30],
    "alt_lookback":     [8, 12, 16],
    "max_alternations": [2, 3, 4],
}
# 2 × 3 × 3 × 3 = 54 combinations
#
# Baseline reference (R001 IS results for comparison):
#   n=2, cd=10 -> PF 13.40, $2827, 1801 trades
#   n=3, cd=10 -> PF 12.99, $2005, 1275 trades
#   n=2, cd=30 -> PF 15.70,  $979,  560 trades
#   n=3, cd=30 -> PF 14.93,  $950,  557 trades
#
# Filter intuition:
#   alt_lookback=12, max_alt=2: in last 12 bricks allow at most 2 flips
#     -> very strict, only strong trending markets pass
#   alt_lookback=12, max_alt=4: allow 4 flips (33% chop) -> moderate gate
#   alt_lookback=8,  max_alt=3: shorter window, same 37.5% flip tolerance


def generate_signals(
    df: pd.DataFrame,
    n_bricks: int = 3,
    cooldown: int = 10,
    alt_lookback: int = 12,
    max_alternations: int = 3,
) -> pd.DataFrame:
    """
    Generate R001-style momentum entries gated by brick alternation count.

    Entry:  N consecutive same-direction bricks AND direction changes in the
            last alt_lookback bricks < max_alternations.
    Exit:   First opposing brick (unconditional — same as R001).
    Cooldown: minimum bricks between entries.

    Args:
        df:               Renko DataFrame with brick_up bool column.
        n_bricks:         Consecutive bricks required before entry signal.
        cooldown:         Minimum bricks between entries.
        alt_lookback:     Number of bricks to examine for alternations.
                          Must be >= n_bricks; if smaller, clamped automatically.
        max_alternations: Maximum allowed direction changes in lookback window.
                          Entry skipped when alternations >= this value.

    Returns:
        df with columns long_entry, long_exit, short_entry, short_exit (bool).
    """
    n        = len(df)
    brick_up = df["brick_up"].values

    # alt_lookback must cover at least the n-brick signal window
    alt_lookback = max(alt_lookback, n_bricks + 1)

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    last_trade_bar = -999_999
    warmup = max(n_bricks, alt_lookback, 30)

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # ── Exit: first opposing brick (unconditional, no cooldown) ────────
        long_exit[i]  = not up
        short_exit[i] = up

        # ── Cooldown gate ──────────────────────────────────────────────────
        if (i - last_trade_bar) < cooldown:
            continue

        # ── N consecutive same-direction bricks ending at bar i ────────────
        window   = brick_up[i - n_bricks + 1 : i + 1]
        all_up   = bool(np.all(window))
        all_down = bool(not np.any(window))

        if not (all_up or all_down):
            continue

        # ── Alternation filter ─────────────────────────────────────────────
        # Count direction changes in the last alt_lookback bricks.
        # The N-brick window is a subset of this wider context window.
        alt_window   = brick_up[i - alt_lookback + 1 : i + 1]
        alternations = int(np.sum(alt_window[1:] != alt_window[:-1]))

        if alternations >= max_alternations:
            continue   # choppy context — skip

        # ── Enter ──────────────────────────────────────────────────────────
        if all_up:
            long_entry[i] = True
        else:
            short_entry[i] = True
        last_trade_bar = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
