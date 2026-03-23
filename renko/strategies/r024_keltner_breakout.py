"""
R024: Keltner Channel Breakout on Renko

Entry: Price closes above upper Keltner (long) or below lower Keltner (short).
       Keltner breakouts on Renko = volatility expansion in brick space.
       Optional ADX gate and Supertrend confirmation.

Exit: First opposing brick or price re-enters channel midline.

Unlike BB breakouts (already tested in R004), Keltner uses ATR instead of
std-dev — more robust to Renko's fixed-size bricks where std-dev is less
meaningful.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from indicators.keltner import calc_keltner

DESCRIPTION = "Keltner Channel breakout with ATR bands on Renko"

HYPOTHESIS = (
    "Keltner Channels use ATR (not std-dev like BB), making them better suited "
    "to Renko where brick sizes are fixed. A close beyond the channel means "
    "volatility is expanding — entering on breakout with trend confirmation "
    "should capture strong moves. ATR adapts to varying brick activity."
)

PARAM_GRID = {
    "kc_period":      [20],
    "kc_mult":        [1.5, 2.0, 2.5],
    "kc_atr_period":  [10, 14],
    "cooldown":       [10, 20, 30],
    "adx_threshold":  [0, 20],
    "require_st":     [True, False],
}
# 1 × 3 × 2 × 3 × 2 × 2 = 72 combinations


def generate_signals(
    df: pd.DataFrame,
    kc_period: int = 20,
    kc_mult: float = 2.0,
    kc_atr_period: int = 10,
    cooldown: int = 20,
    adx_threshold: int = 0,
    require_st: bool = False,
) -> pd.DataFrame:
    n = len(df)
    brick_up = df["brick_up"].values
    closes = df["Close"].values
    adx_vals = df["adx"].values
    st_dir = df["st_dir"].values

    # Compute Keltner and shift by 1
    kc = calc_keltner(df, period=kc_period, mult=kc_mult, atr_period=kc_atr_period)
    kc_upper = np.empty(n)
    kc_lower = np.empty(n)
    kc_mid = np.empty(n)
    kc_upper[0] = np.nan
    kc_lower[0] = np.nan
    kc_mid[0] = np.nan
    kc_upper[1:] = kc["upper"][:-1]
    kc_lower[1:] = kc["lower"][:-1]
    kc_mid[1:] = kc["mid"][:-1]

    long_entry = np.zeros(n, dtype=bool)
    long_exit = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit = np.zeros(n, dtype=bool)

    in_position = False
    trade_dir = 0
    last_trade_bar = -999_999
    warmup = max(kc_period + kc_atr_period, 50)

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # ── Exit: first opposing brick ──────────────────────────────────
        if in_position:
            if trade_dir == 1 and not up:
                long_exit[i] = True
                in_position = False
                trade_dir = 0
            elif trade_dir == -1 and up:
                short_exit[i] = True
                in_position = False
                trade_dir = 0

        if in_position:
            continue

        if (i - last_trade_bar) < cooldown:
            continue

        # ── NaN guard ───────────────────────────────────────────────────
        if np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]):
            continue

        # ── ADX gate ────────────────────────────────────────────────────
        if adx_threshold > 0:
            av = adx_vals[i]
            if np.isnan(av) or av < adx_threshold:
                continue

        # ── Supertrend confirmation ─────────────────────────────────────
        if require_st:
            sd = st_dir[i]
            if np.isnan(sd):
                continue

        # Use previous bar close vs Keltner bands (shifted already)
        prev_close = closes[i - 1] if i > 0 else closes[i]

        # ── Keltner breakout ────────────────────────────────────────────
        # Long: prev close above upper Keltner + up brick
        if prev_close > kc_upper[i] and up:
            if not require_st or st_dir[i] > 0:
                long_entry[i] = True
                in_position = True
                trade_dir = 1
                last_trade_bar = i

        # Short: prev close below lower Keltner + down brick
        elif prev_close < kc_lower[i] and not up:
            if not require_st or st_dir[i] < 0:
                short_entry[i] = True
                in_position = True
                trade_dir = -1
                last_trade_bar = i

    df["long_entry"] = long_entry
    df["long_exit"] = long_exit
    df["short_entry"] = short_entry
    df["short_exit"] = short_exit
    return df
