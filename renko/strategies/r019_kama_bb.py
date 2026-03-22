"""
R019: KAMA Slope + BB %B Confluence Entry

Entry: KAMA slope turns positive AND BB %B is above 0.5 (long), or
       KAMA slope turns negative AND BB %B is below 0.5 (short).

KAMA (Kaufman Adaptive MA) adapts its speed to market conditions — fast in
trends, slow in chop. Its slope changing sign means the adaptive average
has confirmed a direction change. Combining with BB %B (position within
Bollinger Bands) ensures price is in the correct half of the volatility
envelope.

Optional gates:
  - req_brick:     require brick direction to match
  - adx_threshold: minimum ADX strength
  - bb_pctb_level: BB %B threshold (default 0.5)

Exit: first opposing brick.

Data: OANDA_EURUSD, 1S renko 0.0004.csv
"""

import numpy as np
import pandas as pd

DESCRIPTION = "KAMA slope change + BB %B position confluence entry"

HYPOTHESIS = (
    "KAMA slope sign change = adaptive momentum shift. BB %B position = where "
    "price sits in the volatility envelope. Together they confirm: momentum "
    "is turning AND price is positioned correctly. On Renko, KAMA adapts to "
    "brick momentum — its slope change is a high-confidence signal."
)

RENKO_FILE      = "OANDA_EURUSD, 1S renko 0.0004.csv"
COMMISSION_PCT  = 0.0046
INITIAL_CAPITAL = 1000.0

PARAM_GRID = {
    "bb_pctb_level":  [0.4, 0.5, 0.6],
    "req_brick":      [True, False],
    "adx_threshold":  [0, 20, 25],
    "cooldown":       [5, 10, 20],
}
# 3 x 2 x 3 x 3 = 54 combos


def generate_signals(
    df:            pd.DataFrame,
    bb_pctb_level: float = 0.5,
    req_brick:     bool  = True,
    adx_threshold: int   = 0,
    cooldown:      int   = 10,
) -> pd.DataFrame:
    n = len(df)
    brick_up   = df["brick_up"].values
    kama_slope = df["kama_slope"].values
    bb_pct_b   = df["bb_pct_b"].values
    adx_vals   = df["adx"].values

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
        if (np.isnan(kama_slope[i]) or np.isnan(kama_slope[i-1])
                or np.isnan(bb_pct_b[i])):
            continue

        # --- KAMA slope sign change ---
        kama_turns_bull = kama_slope[i] > 0 and kama_slope[i-1] <= 0
        kama_turns_bear = kama_slope[i] < 0 and kama_slope[i-1] >= 0

        if not kama_turns_bull and not kama_turns_bear:
            continue

        # --- BB %B position ---
        if kama_turns_bull and bb_pct_b[i] < bb_pctb_level:
            continue
        if kama_turns_bear and bb_pct_b[i] > (1.0 - bb_pctb_level):
            continue

        # --- ADX gate ---
        if adx_threshold > 0:
            adx_val = adx_vals[i]
            if np.isnan(adx_val) or adx_val < adx_threshold:
                continue

        # --- Brick confirmation ---
        if req_brick:
            if kama_turns_bull and not up:
                continue
            if kama_turns_bear and up:
                continue

        # --- Entry ---
        if kama_turns_bull:
            long_entry[i]  = True
            in_position    = True
            trade_dir      = 1
            last_trade_bar = i

        elif kama_turns_bear:
            short_entry[i] = True
            in_position    = True
            trade_dir      = -1
            last_trade_bar = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
