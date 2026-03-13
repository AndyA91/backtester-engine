"""
R005: Ultimate Hybrid Master

Combines R001 (momentum) and R002 (reversal) entry modes with two stacked
regime filters: ADX strength gate and a configurable brick trailing stop.

Momentum mode:  N consecutive same-direction bricks → entry in trend direction (R001).
Reversal mode:  N consecutive same-direction bricks → entry AGAINST trend direction (R002).
ADX gate:       Only enter when ADX(14) >= adx_threshold (0 = disabled).
Trailing exit:  Exit after trail_bricks consecutive opposing bricks
                (trail_bricks=1 replicates R001 first-opposing-brick exit).

SANITY CHECK: mode="momentum", n_bricks=2, cooldown=10, adx_threshold=0,
trail_bricks=1 → should produce similar trade count/PF to R001 n=2 cd=10
(minor difference expected: R005 suppresses entry signals while in position,
so cooldown timing differs slightly from R001's always-emit approach).
"""

import numpy as np
import pandas as pd

DESCRIPTION = "momentum/reversal bricks + ADX gate + brick trailing stop"

HYPOTHESIS = (
    "R001 confirmed brick momentum is a strong edge (all 16 combos PF>12 over 2.75yr IS). "
    "R005 tests whether ADX regime filtering can improve risk-adjusted PF for both momentum "
    "and reversal entries, and whether a looser trailing stop (trail_bricks=2) preserves "
    "more of the winning trades that R001's first-opposing-brick exit cuts short."
)

PARAM_GRID = {
    "n_bricks":      [2, 3, 4],
    "cooldown":      [10, 20, 30],
    "mode":          ["momentum", "reversal"],
    "adx_threshold": [0, 25],
    "trail_bricks":  [1, 2],
}
# 3 × 3 × 2 × 2 × 2 = 72 combinations


def generate_signals(
    df: pd.DataFrame,
    n_bricks: int = 2,
    cooldown: int = 10,
    mode: str = "momentum",
    adx_threshold: float = 0,
    trail_bricks: int = 1,
) -> pd.DataFrame:
    """
    Generate entry/exit signals for the Ultimate Hybrid Master strategy.

    Entry:  N consecutive same-direction bricks, optionally gated by ADX.
    Exit:   trail_bricks consecutive opposing bricks (1 = first opposing brick).
    Mode:   "momentum" → enter in brick direction; "reversal" → enter against direction.

    Position state is tracked internally so entries cannot fire while in a trade.
    Cooldown is only reset on actual entry bars (not suppressed signals).

    Args:
        df:             Renko DataFrame with brick_up bool + pre-shifted indicator columns.
        n_bricks:       Consecutive same-direction bricks required for entry signal.
        cooldown:       Minimum bricks between entries.
        mode:           "momentum" (trend) or "reversal" (counter-trend).
        adx_threshold:  Minimum ADX(14) to enter. 0 = disabled.
        trail_bricks:   Exit after this many consecutive opposing bricks.

    Returns:
        df with columns long_entry, long_exit, short_entry, short_exit (bool).
    """
    n        = len(df)
    brick_up = df["brick_up"].values
    adx      = df["adx"].values

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    last_trade_bar = -999_999
    warmup = max(n_bricks, 30)  # covers ADX(14) warmup

    in_long   = False
    in_short  = False
    opp_count = 0  # consecutive opposing bricks since last same-direction brick

    for i in range(warmup, n):
        # NaN guard (ADX needs ~14 bars to converge; first bars may be NaN)
        if np.isnan(adx[i]):
            continue

        up = bool(brick_up[i])

        # ── Exit: count consecutive opposing bricks ────────────────────────────
        if in_long:
            if up:
                opp_count = 0           # trend continuing — reset counter
            else:
                opp_count += 1
                if opp_count >= trail_bricks:
                    long_exit[i] = True
                    in_long      = False
                    opp_count    = 0
            continue  # no entry checks while in position

        if in_short:
            if not up:
                opp_count = 0           # trend continuing — reset counter
            else:
                opp_count += 1
                if opp_count >= trail_bricks:
                    short_exit[i] = True
                    in_short      = False
                    opp_count     = 0
            continue

        # ── Entry (only when flat) ─────────────────────────────────────────────
        # N consecutive same-direction bricks ending at bar i
        window   = brick_up[i - n_bricks + 1 : i + 1]
        all_up   = bool(np.all(window))
        all_down = bool(not np.any(window))

        if not (all_up or all_down):
            continue

        # Cooldown filter
        if (i - last_trade_bar) < cooldown:
            continue

        # ADX regime filter
        if adx_threshold > 0 and adx[i] < adx_threshold:
            continue

        # ── Determine entry direction based on mode ────────────────────────────
        if mode == "momentum":
            if all_up:
                long_entry[i]  = True
                in_long        = True
            else:
                short_entry[i] = True
                in_short       = True
        else:  # reversal
            if all_up:
                short_entry[i] = True
                in_short       = True
            else:
                long_entry[i]  = True
                in_long        = True

        last_trade_bar = i
        opp_count      = 0

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
