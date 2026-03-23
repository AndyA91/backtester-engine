"""
R020: Multi-Signal Voting Confluence

Entry: At least K out of N independent signals agree on direction.

Signals polled (each votes +1 for bullish, -1 for bearish, 0 for neutral):
  1. brick_up:     current brick direction
  2. macd_hist:    MACD histogram > 0 = bullish
  3. rsi:          RSI > 50 = bullish, < 50 = bearish
  4. st_dir:       Supertrend direction
  5. kama_slope:   KAMA slope positive = bullish

This is an ensemble approach — no single indicator dominates. The K-of-N
threshold lets us tune how much agreement is required.

Optional gates:
  - adx_threshold: minimum ADX strength
  - cooldown:      minimum bricks between entries

Exit: first opposing brick.

Data: OANDA_EURUSD, 1S renko 0.0004.csv
"""

import numpy as np
import pandas as pd

DESCRIPTION = "Multi-signal voting: enter when K of 5 signals agree on direction"

HYPOTHESIS = (
    "Individual indicators each capture a different market property: momentum "
    "(MACD), mean-reversion level (RSI), trend direction (Supertrend), "
    "adaptive momentum (KAMA slope), and raw Renko structure (brick direction). "
    "When multiple independent signals agree, the probability of a true trend "
    "is higher than any single signal alone. This ensemble approach should "
    "produce fewer but higher-quality trades."
)

RENKO_FILE      = "OANDA_EURUSD, 1S renko 0.0004.csv"
COMMISSION_PCT  = 0.0046
INITIAL_CAPITAL = 1000.0

PARAM_GRID = {
    "min_votes":     [3, 4, 5],
    "adx_threshold": [0, 20, 25],
    "cooldown":      [5, 10, 20],
}
# 3 x 3 x 3 = 27 combos


def generate_signals(
    df:            pd.DataFrame,
    min_votes:     int = 3,
    adx_threshold: int = 0,
    cooldown:      int = 10,
) -> pd.DataFrame:
    n = len(df)
    brick_up   = df["brick_up"].values
    macd_hist  = df["macd_hist"].values
    rsi_vals   = df["rsi"].values
    st_dir     = df["st_dir"].values
    kama_slope = df["kama_slope"].values
    adx_vals   = df["adx"].values

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    in_position    = False
    trade_dir      = 0
    last_trade_bar = -999_999
    warmup         = 50

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

        # --- Vote tally ---
        bull_votes = 0
        bear_votes = 0

        # Vote 1: brick direction
        if up:
            bull_votes += 1
        else:
            bear_votes += 1

        # Vote 2: MACD histogram
        if not np.isnan(macd_hist[i]):
            if macd_hist[i] > 0:
                bull_votes += 1
            elif macd_hist[i] < 0:
                bear_votes += 1

        # Vote 3: RSI > 50 / < 50
        if not np.isnan(rsi_vals[i]):
            if rsi_vals[i] > 50:
                bull_votes += 1
            elif rsi_vals[i] < 50:
                bear_votes += 1

        # Vote 4: Supertrend direction
        if not np.isnan(st_dir[i]):
            if st_dir[i] == 1:
                bull_votes += 1
            elif st_dir[i] == -1:
                bear_votes += 1

        # Vote 5: KAMA slope
        if not np.isnan(kama_slope[i]):
            if kama_slope[i] > 0:
                bull_votes += 1
            elif kama_slope[i] < 0:
                bear_votes += 1

        # --- ADX gate ---
        if adx_threshold > 0:
            adx_val = adx_vals[i]
            if np.isnan(adx_val) or adx_val < adx_threshold:
                continue

        # --- Entry: need min_votes agreement ---
        if bull_votes >= min_votes:
            long_entry[i]  = True
            in_position    = True
            trade_dir      = 1
            last_trade_bar = i

        elif bear_votes >= min_votes:
            short_entry[i] = True
            in_position    = True
            trade_dir      = -1
            last_trade_bar = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
