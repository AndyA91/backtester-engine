"""
BTC002: N Consecutive Same-Direction Bricks — BTCUSD Baseline

Port of R001/GJ001 brick-count momentum to BTCUSD. Entry on N consecutive
same-direction bricks; exit on first opposing brick. Cooldown in bricks.

Dataset  : OANDA_BTCUSD.SPOT.US, 1S renko 150.csv
Brick    : $150

BTCUSD engine calibration:
  Commission: 0.009% (same as btc001_fisher_adx).
  Initial capital: $1,000 USD.
"""

import numpy as np
import pandas as pd

RENKO_FILE      = "OANDA_BTCUSD.SPOT.US, 1S renko 150.csv"
COMMISSION_PCT  = 0.009
INITIAL_CAPITAL = 1000.0

DESCRIPTION = "N consecutive same-direction bricks -> entry, first opposing brick -> exit (BTCUSD)"

HYPOTHESIS = (
    "The pure Renko momentum edge (N consecutive bricks in one direction predict "
    "continuation) should be instrument-agnostic. BTCUSD $150 bricks filter out "
    "noise effectively. This establishes a pure brick-count baseline for BTCUSD "
    "to compare against the Fisher+ADX approach in btc001."
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
    n = len(df)
    brick_up = df["brick_up"].values

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    last_trade_bar = -999_999
    warmup = n_bricks

    for i in range(warmup, n):
        window = brick_up[i - n_bricks + 1 : i + 1]
        all_up   = bool(np.all(window))
        all_down = bool(np.not_equal(window, True).all())

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
