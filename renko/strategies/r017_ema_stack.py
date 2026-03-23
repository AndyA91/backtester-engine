"""
R017: EMA Stack Momentum — Aligned EMA Entry

Entry: All selected EMAs are perfectly aligned (EMA9 > EMA21 > EMA50 for longs,
       reverse for shorts) AND brick direction confirms.

This is a structural trend filter: when fast/medium/slow EMAs are all stacked
in order, the trend is established across multiple timeframes. On Renko, this
means price has been consistently moving in one direction across brick-scale
momentum windows.

Optional gates:
  - require_ema200: also require EMA200 alignment (stricter)
  - adx_threshold:  minimum ADX strength
  - cooldown:       minimum bricks between entries

Exit: first opposing brick.

Data: OANDA_EURUSD, 1S renko 0.0004.csv
"""

import numpy as np
import pandas as pd

DESCRIPTION = "EMA stack alignment entry (9>21>50 for longs) with brick confirmation"

HYPOTHESIS = (
    "Perfectly stacked EMAs (9>21>50 or 9<21<50) indicate established trend "
    "across multiple momentum windows. On Renko bricks, EMA alignment is more "
    "meaningful because each bar represents equal price movement. No existing "
    "strategy uses EMA alignment as an entry condition — only EMA cross (R002)."
)

RENKO_FILE      = "OANDA_EURUSD, 1S renko 0.0004.csv"
COMMISSION_PCT  = 0.0046
INITIAL_CAPITAL = 1000.0

PARAM_GRID = {
    "require_ema200": [False, True],
    "adx_threshold":  [0, 20, 25],
    "cooldown":       [5, 10, 20, 30],
}
# 2 x 3 x 4 = 24 combos


def generate_signals(
    df:             pd.DataFrame,
    require_ema200: bool = False,
    adx_threshold:  int  = 0,
    cooldown:       int  = 10,
) -> pd.DataFrame:
    n = len(df)
    brick_up = df["brick_up"].values
    ema9     = df["ema9"].values
    ema21    = df["ema21"].values
    ema50    = df["ema50"].values
    ema200   = df["ema200"].values
    adx_vals = df["adx"].values

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    in_position    = False
    trade_dir      = 0
    last_trade_bar = -999_999
    warmup         = 210 if require_ema200 else 55

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
        if np.isnan(ema9[i]) or np.isnan(ema21[i]) or np.isnan(ema50[i]):
            continue
        if require_ema200 and np.isnan(ema200[i]):
            continue

        # --- EMA stack detection ---
        bull_stack = ema9[i] > ema21[i] > ema50[i]
        bear_stack = ema9[i] < ema21[i] < ema50[i]

        if require_ema200:
            bull_stack = bull_stack and ema50[i] > ema200[i]
            bear_stack = bear_stack and ema50[i] < ema200[i]

        # --- ADX gate ---
        if adx_threshold > 0:
            adx_val = adx_vals[i]
            if np.isnan(adx_val) or adx_val < adx_threshold:
                continue

        # --- Entry: stack + brick confirmation ---
        if bull_stack and up:
            long_entry[i]  = True
            in_position    = True
            trade_dir      = 1
            last_trade_bar = i

        elif bear_stack and not up:
            short_entry[i] = True
            in_position    = True
            trade_dir      = -1
            last_trade_bar = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
