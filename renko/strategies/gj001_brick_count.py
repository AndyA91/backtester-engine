"""
GJ001: N Consecutive Same-Direction Bricks — GBPJPY Baseline

Identical logic to R001 (EURUSD baseline). Entry on N consecutive
same-direction bricks; exit on first opposing brick. Cooldown in bricks.

Purpose: Establish whether the pure Renko momentum edge (confirmed on
EURUSD R001 with IS PF 12–15, OOS PF 4.7–7.3) holds on a different asset
with a different volatility profile (GBPJPY, brick_size=0.05 JPY).

Dataset  : OANDA_GBPJPY, 1S renko 0.05.csv
Period   : 2024-11-20 → 2026-03-10 (~15.6 months, full IS period for GBPJPY)
Bricks   : ~19,785

NOTE on commission: runner.py hardcodes commission_pct=0.0046 (calibrated
for EURUSD, ~$0.10 round-trip per 1k units). GBPJPY typical OANDA spread
is ~1.8–2.5 pips — roughly 2x wider. PF figures here will therefore be
OPTIMISTIC vs real GBPJPY trading costs. Treat as a directional signal
check, not a precise performance estimate.
"""

import numpy as np
import pandas as pd

RENKO_FILE = "OANDA_GBPJPY, 1S renko 0.05.csv"

# GBPJPY engine calibration
# Commission: OANDA GBPJPY spread ~2 pips (0.02 JPY). Per-leg cost at ~195 JPY:
#   0.01 JPY / 195 = 0.00513% → rounded to 0.005% (slightly conservative vs EURUSD 0.0046%)
# Initial capital: 150,000 JPY ≈ $1,000 USD at ~150 JPY/USD.
#   Keeps leverage consistent with EURUSD strategies and makes DD% meaningful in JPY terms.
COMMISSION_PCT  = 0.005
INITIAL_CAPITAL = 150_000.0

DESCRIPTION = "N consecutive same-direction bricks -> entry, first opposing brick -> exit (GBPJPY)"

HYPOTHESIS = (
    "The pure Renko momentum edge (N consecutive bricks in one direction predict "
    "continuation) should be instrument-agnostic. GBPJPY has higher volatility and "
    "wider bricks than EURUSD, but the underlying mechanic — that renko runs cluster "
    "in time — should persist. This establishes a GBPJPY baseline for further filter "
    "development (session, ADX, etc.)."
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
