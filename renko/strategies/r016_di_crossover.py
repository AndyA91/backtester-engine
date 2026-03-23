"""
R016: DI Crossover — Directional Index Crossover Entry

Entry: +DI crosses above -DI (long) or -DI crosses above +DI (short).
This uses the DI *crossover* as the entry trigger, not just an ADX level gate.

Optional gates:
  - adx_threshold: minimum ADX at time of cross (0=off)
  - req_brick:     require brick direction to match cross direction
  - cooldown:      minimum bricks between entries

Exit: first opposing brick.

Key insight: existing strategies only use ADX as a *level gate* (ADX > 25).
Nobody has used the DI crossover itself as the primary entry signal on Renko.

Data: OANDA_EURUSD, 1S renko 0.0004.csv
"""

import numpy as np
import pandas as pd

DESCRIPTION = "DI crossover entry (+DI crosses -DI) with optional ADX gate and brick confirm"

HYPOTHESIS = (
    "The +DI/-DI crossover identifies the exact moment directional pressure "
    "shifts. On Renko, where each brick is a fixed price move, DI crosses "
    "should be cleaner than on time-based charts. No existing strategy uses "
    "DI cross as the primary entry trigger — only as a level gate."
)

RENKO_FILE      = "OANDA_EURUSD, 1S renko 0.0004.csv"
COMMISSION_PCT  = 0.0046
INITIAL_CAPITAL = 1000.0

PARAM_GRID = {
    "adx_threshold":     [0, 15, 20, 25],
    "req_brick":         [True, False],
    "cooldown":          [3, 5, 10, 20],
}
# 4 x 2 x 4 = 32 combos


def generate_signals(
    df:            pd.DataFrame,
    adx_threshold: int  = 20,
    req_brick:     bool = True,
    cooldown:      int  = 10,
) -> pd.DataFrame:
    n = len(df)
    brick_up = df["brick_up"].values
    adx_vals = df["adx"].values
    plus_di  = df["plus_di"].values
    minus_di = df["minus_di"].values

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    in_position    = False
    trade_dir      = 0
    last_trade_bar = -999_999
    warmup         = 30

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # --- Exit: first opposing brick ---
        if in_position:
            if trade_dir == 1 and not up:
                long_exit[i] = True
                in_position  = False
                trade_dir    = 0
            elif trade_dir == -1 and up:
                short_exit[i] = True
                in_position   = False
                trade_dir     = 0

        if in_position:
            continue

        # --- Cooldown ---
        if (i - last_trade_bar) < cooldown:
            continue

        # --- NaN guard ---
        if (np.isnan(plus_di[i]) or np.isnan(minus_di[i])
                or np.isnan(plus_di[i-1]) or np.isnan(minus_di[i-1])):
            continue

        # --- DI crossover detection (pre-shifted: values at [i] = computed thru bar i-1) ---
        bull_cross = plus_di[i] > minus_di[i] and plus_di[i-1] <= minus_di[i-1]
        bear_cross = minus_di[i] > plus_di[i] and minus_di[i-1] <= plus_di[i-1]

        if not bull_cross and not bear_cross:
            continue

        # --- ADX gate ---
        if adx_threshold > 0:
            adx_val = adx_vals[i]
            if np.isnan(adx_val) or adx_val < adx_threshold:
                continue

        # --- Entry ---
        if bull_cross:
            if req_brick and not up:
                continue
            long_entry[i]  = True
            in_position    = True
            trade_dir      = 1
            last_trade_bar = i

        elif bear_cross:
            if req_brick and up:
                continue
            short_entry[i] = True
            in_position    = True
            trade_dir      = -1
            last_trade_bar = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
