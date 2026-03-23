"""
R015: Squeeze Momentum Breakout — EURUSD Renko 0.0004

TTM Squeeze (LazyBear/John Carter) on Renko bricks. When Bollinger Bands
contract inside Keltner Channels, volatility is coiling. When the squeeze
releases (bands expand beyond KC), momentum determines trade direction.

Entry: Squeeze releases AND momentum confirms direction.
Exit:  N consecutive opposing bricks (exit_bricks=1 = standard).

Optional gates:
  - req_brick:     require brick direction to match momentum
  - adx_threshold: ADX regime gate (0=off)

Data: OANDA_EURUSD, 1S renko 0.0004.csv
"""

import numpy as np
import pandas as pd

RENKO_FILE      = "OANDA_EURUSD, 1S renko 0.0004.csv"
COMMISSION_PCT  = 0.0046
INITIAL_CAPITAL = 1000.0

DESCRIPTION = "Squeeze Momentum breakout — enter on squeeze release with momentum confirmation (EURUSD)"

HYPOTHESIS = (
    "TTM Squeeze identifies volatility compression → expansion cycles. On Renko "
    "bricks, each brick = fixed price move, so BB/KC relationship reflects true "
    "directional volatility. Squeeze release on Renko should signal high-probability "
    "breakouts with momentum confirmation filtering false releases."
)

PARAM_GRID = {
    "req_brick":      [True, False],
    "adx_threshold":  [0, 20, 25],
    "cooldown":       [5, 10, 20],
    "exit_bricks":    [1, 2],
}
# 2 x 3 x 3 x 2 = 36 combos


def generate_signals(
    df: pd.DataFrame,
    req_brick: bool = True,
    adx_threshold: int = 0,
    cooldown: int = 10,
    exit_bricks: int = 1,
) -> pd.DataFrame:
    n = len(df)
    brick_up   = df["brick_up"].values
    sq_on      = df["sq_on"].values       # pre-shifted: sq_on[i] = state at bar i-1
    momentum   = df["sq_momentum"].values # pre-shifted: momentum[i] = value at bar i-1
    adx_vals   = df["adx"].values         # pre-shifted

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    in_position    = False
    trade_dir      = 0
    last_trade_bar = -999_999
    opposing_count = 0
    warmup         = 50

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # --- Exit: count consecutive opposing bricks ---
        if in_position:
            is_opposing = (trade_dir == 1 and not up) or (trade_dir == -1 and up)
            if is_opposing:
                opposing_count += 1
            else:
                opposing_count = 0

            if opposing_count >= exit_bricks:
                if trade_dir == 1:
                    long_exit[i] = True
                else:
                    short_exit[i] = True
                in_position    = False
                trade_dir      = 0
                opposing_count = 0

        if in_position:
            continue

        # --- Cooldown ---
        if (i - last_trade_bar) < cooldown:
            continue

        # --- Squeeze release detection ---
        # sq_on is pre-shifted: sq_on[i] = state through bar i-1
        # sq_on[i-1] = state through bar i-2
        # Squeeze release: was on at i-2, off at i-1
        if i < 2:
            continue
        sq_prev = sq_on[i - 1]
        sq_curr = sq_on[i]
        if np.isnan(sq_prev) or np.isnan(sq_curr):
            continue

        squeeze_released = bool(sq_prev) and not bool(sq_curr)
        if not squeeze_released:
            continue

        # --- ADX gate ---
        if adx_threshold > 0:
            adx_val = adx_vals[i]
            if np.isnan(adx_val) or adx_val < adx_threshold:
                continue

        # --- Momentum direction ---
        mom = momentum[i]
        if np.isnan(mom):
            continue

        if mom > 0:
            if req_brick and not up:
                continue
            long_entry[i]  = True
            in_position    = True
            trade_dir      = 1
            last_trade_bar = i
            opposing_count = 0

        elif mom < 0:
            if req_brick and up:
                continue
            short_entry[i] = True
            in_position    = True
            trade_dir      = -1
            last_trade_bar = i
            opposing_count = 0

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
