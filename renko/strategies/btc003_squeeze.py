"""
BTC003: Squeeze Momentum Breakout — BTCUSD Renko 150

TTM Squeeze (LazyBear/John Carter) on Renko bricks. When Bollinger Bands
contract inside Keltner Channels, volatility is coiling. When the squeeze
releases (bands expand beyond KC), momentum determines trade direction.

Entry: Squeeze releases AND momentum confirms direction.
Exit:  N consecutive opposing bricks (exit_bricks=1 = standard).

Data: OANDA_BTCUSD.SPOT.US, 1S renko 150.csv

BTCUSD engine calibration:
  Commission: 0.009% (same as btc001).
  Initial capital: $1,000 USD.
"""

import numpy as np
import pandas as pd

RENKO_FILE      = "OANDA_BTCUSD.SPOT.US, 1S renko 150.csv"
COMMISSION_PCT  = 0.009
INITIAL_CAPITAL = 1000.0

DESCRIPTION = "Squeeze Momentum breakout — enter on squeeze release with momentum confirmation (BTCUSD)"

HYPOTHESIS = (
    "TTM Squeeze identifies volatility compression -> expansion cycles. BTCUSD "
    "$150 bricks have strong trending characteristics. Squeeze release should "
    "capture breakout moves that complement Fisher+ADX (btc001) entries."
)

PARAM_GRID = {
    "req_brick":      [True, False],
    "adx_threshold":  [0, 20, 25],
    "cooldown":       [5, 10, 20],
    "exit_bricks":    [1, 2],
}


def generate_signals(
    df: pd.DataFrame,
    req_brick: bool = True,
    adx_threshold: int = 0,
    cooldown: int = 10,
    exit_bricks: int = 1,
) -> pd.DataFrame:
    n = len(df)
    brick_up   = df["brick_up"].values
    sq_on      = df["sq_on"].values
    momentum   = df["sq_momentum"].values
    adx_vals   = df["adx"].values

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

        if (i - last_trade_bar) < cooldown:
            continue

        if i < 2:
            continue
        sq_prev = sq_on[i - 1]
        sq_curr = sq_on[i]
        if np.isnan(sq_prev) or np.isnan(sq_curr):
            continue

        squeeze_released = bool(sq_prev) and not bool(sq_curr)
        if not squeeze_released:
            continue

        if adx_threshold > 0:
            adx_val = adx_vals[i]
            if np.isnan(adx_val) or adx_val < adx_threshold:
                continue

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
