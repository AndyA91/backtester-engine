"""
R021: Squeeze Release + Brick Count Hybrid

Combines two proven winners:
  - R015: TTM Squeeze release (volatility compression → expansion)
  - R001: N consecutive same-direction bricks (momentum confirmation)

Entry: Squeeze has released within the last `lookback` bricks AND N consecutive
       bricks confirm direction. This ensures we enter breakouts that have both
       volatility expansion AND price momentum.

The key difference from R015 alone: we don't enter immediately on squeeze
release — we wait for N bricks to confirm the breakout direction. This
filters false releases where momentum doesn't follow through.

Optional gates:
  - adx_threshold: minimum ADX strength
  - req_momentum:  require sq_momentum to match direction

Exit: first opposing brick.

Data: OANDA_EURUSD, 1S renko 0.0004.csv
"""

import numpy as np
import pandas as pd

DESCRIPTION = "Squeeze release + N consecutive bricks hybrid (R015 + R001 combined)"

HYPOTHESIS = (
    "R015 (Squeeze) identifies when volatility expands. R001 (brick count) "
    "identifies when momentum is established. Together: enter breakouts that "
    "have BOTH volatility expansion AND confirmed directional momentum. "
    "This should filter R015's false releases and R001's low-volatility entries."
)

RENKO_FILE      = "OANDA_EURUSD, 1S renko 0.0004.csv"
COMMISSION_PCT  = 0.0046
INITIAL_CAPITAL = 1000.0

PARAM_GRID = {
    "n_bricks":      [2, 3, 4],
    "lookback":      [3, 5, 10],
    "req_momentum":  [True, False],
    "adx_threshold": [0, 20],
    "cooldown":      [5, 10, 20],
}
# 3 x 3 x 2 x 2 x 3 = 108 combos


def generate_signals(
    df:            pd.DataFrame,
    n_bricks:      int  = 3,
    lookback:      int  = 5,
    req_momentum:  bool = True,
    adx_threshold: int  = 0,
    cooldown:      int  = 10,
) -> pd.DataFrame:
    n = len(df)
    brick_up = df["brick_up"].values
    sq_on    = df["sq_on"].values
    momentum = df["sq_momentum"].values
    adx_vals = df["adx"].values

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

        # --- Check for squeeze release within lookback window ---
        # sq_on[j] is pre-shifted: True = squeeze was active at bar j-1
        squeeze_released_recently = False
        for j in range(max(warmup, i - lookback), i + 1):
            if j < 2:
                continue
            prev_sq = sq_on[j - 1]
            curr_sq = sq_on[j]
            if np.isnan(prev_sq) or np.isnan(curr_sq):
                continue
            if bool(prev_sq) and not bool(curr_sq):
                squeeze_released_recently = True
                break

        if not squeeze_released_recently:
            continue

        # --- N consecutive bricks in same direction (R001 logic) ---
        window   = brick_up[i - n_bricks + 1 : i + 1]
        all_up   = bool(np.all(window))
        all_down = bool(np.not_equal(window, True).all())

        if not all_up and not all_down:
            continue

        # --- Momentum direction match ---
        if req_momentum:
            mom = momentum[i]
            if np.isnan(mom):
                continue
            if all_up and mom <= 0:
                continue
            if all_down and mom >= 0:
                continue

        # --- ADX gate ---
        if adx_threshold > 0:
            adx_val = adx_vals[i]
            if np.isnan(adx_val) or adx_val < adx_threshold:
                continue

        # --- Entry ---
        if all_up:
            long_entry[i]  = True
            in_position    = True
            trade_dir      = 1
            last_trade_bar = i

        elif all_down:
            short_entry[i] = True
            in_position    = True
            trade_dir      = -1
            last_trade_bar = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
