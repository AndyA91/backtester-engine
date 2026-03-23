"""
R023: Williams %R Momentum + ATR Trailing Exit

Entry: Williams %R crosses above -50 from below (bullish momentum shift) → long,
       Williams %R crosses below -50 from above (bearish momentum shift) → short.
       Confirmed by brick direction matching.

Exit: ATR-based trailing stop (N × ATR from peak/trough) instead of first
      opposing brick. This lets winners run longer in strong trends.

Williams %R on Renko is unexplored. Unlike RSI (mean-reversion, poor on Renko),
%R midline crosses detect momentum shifts — better fit for trending Renko data.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from indicators.williams_r import calc_williams_r

DESCRIPTION = "Williams %R midline cross + ATR trailing exit"

HYPOTHESIS = (
    "Williams %R midline (-50) crosses detect momentum shifts without the "
    "mean-reversion bias of RSI overbought/oversold zones. On Renko, momentum "
    "shifts should be cleaner. ATR trailing exit lets winners run — the R009 "
    "exit study showed 0-1h trades are 100% losers, so trailing helps."
)

PARAM_GRID = {
    "wpr_period":     [14, 21],
    "cooldown":       [10, 20, 30],
    "exit_mode":      [0, 1],       # 0=first opposing, 1=ATR trail
    "trail_atr_mult": [1.5, 2.0, 3.0],
    "adx_threshold":  [0, 20],
}
# 2 × 3 × 2 × 3 × 2 = 72 combinations


def generate_signals(
    df: pd.DataFrame,
    wpr_period: int = 14,
    cooldown: int = 20,
    exit_mode: int = 0,
    trail_atr_mult: float = 2.0,
    adx_threshold: int = 0,
) -> pd.DataFrame:
    n = len(df)
    brick_up = df["brick_up"].values
    closes = df["Close"].values
    adx_vals = df["adx"].values
    atr_vals = df["atr"].values

    # Compute Williams %R and shift by 1 to avoid lookahead
    wpr_result = calc_williams_r(df, period=wpr_period)
    wpr_raw = wpr_result["wpr"]
    wpr = np.empty(n)
    wpr[0] = np.nan
    wpr[1:] = wpr_raw[:-1]

    long_entry = np.zeros(n, dtype=bool)
    long_exit = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit = np.zeros(n, dtype=bool)

    in_position = False
    trade_dir = 0
    last_trade_bar = -999_999
    peak_close = 0.0
    trail_stop = 0.0
    warmup = max(wpr_period + 2, 30)

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # ── Exit logic ─────────────────────────────────────────────────
        if in_position:
            if exit_mode == 0:
                # First opposing brick
                do_exit = (trade_dir == 1 and not up) or (trade_dir == -1 and up)
            else:
                # ATR trailing stop
                atr_val = atr_vals[i]
                if np.isnan(atr_val):
                    atr_val = 0.0
                if trade_dir == 1:
                    if closes[i] > peak_close:
                        peak_close = closes[i]
                        trail_stop = peak_close - trail_atr_mult * atr_val
                    do_exit = closes[i] < trail_stop
                else:
                    if closes[i] < peak_close:
                        peak_close = closes[i]
                        trail_stop = peak_close + trail_atr_mult * atr_val
                    do_exit = closes[i] > trail_stop

            if do_exit:
                long_exit[i] = trade_dir == 1
                short_exit[i] = trade_dir == -1
                in_position = False
                trade_dir = 0

        if in_position:
            continue

        if (i - last_trade_bar) < cooldown:
            continue

        # ── ADX gate ────────────────────────────────────────────────────
        if adx_threshold > 0:
            av = adx_vals[i]
            if np.isnan(av) or av < adx_threshold:
                continue

        # ── Williams %R midline cross ───────────────────────────────────
        if np.isnan(wpr[i]) or np.isnan(wpr[i - 1]):
            continue

        cross_up = wpr[i - 1] < -50 and wpr[i] >= -50
        cross_down = wpr[i - 1] > -50 and wpr[i] <= -50

        if cross_up and up:
            long_entry[i] = True
            in_position = True
            trade_dir = 1
            last_trade_bar = i
            peak_close = closes[i]
            atr_val = atr_vals[i] if not np.isnan(atr_vals[i]) else 0.0
            trail_stop = closes[i] - trail_atr_mult * atr_val

        elif cross_down and not up:
            short_entry[i] = True
            in_position = True
            trade_dir = -1
            last_trade_bar = i
            peak_close = closes[i]
            atr_val = atr_vals[i] if not np.isnan(atr_vals[i]) else 0.0
            trail_stop = closes[i] + trail_atr_mult * atr_val

    df["long_entry"] = long_entry
    df["long_exit"] = long_exit
    df["short_entry"] = short_entry
    df["short_exit"] = short_exit
    return df
