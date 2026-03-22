"""
R022: Ichimoku Cloud Breakout on Renko

Entry: Price breaks above cloud (long) or below cloud (short), confirmed by
Tenkan > Kijun (long) or Tenkan < Kijun (short). Optional ADX gate.

Exit: First opposing brick (baseline) or price re-enters cloud.

Ichimoku on Renko should work well — Renko removes noise, so cloud breakouts
should be cleaner than on time charts. The TK cross adds momentum confirmation.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from indicators.ichimoku import calc_ichimoku, price_vs_cloud

DESCRIPTION = "Ichimoku cloud breakout + TK cross on Renko bricks"

HYPOTHESIS = (
    "Renko filters noise, making Ichimoku cloud breaks more decisive. "
    "Price above cloud + Tenkan > Kijun = strong trend confirmation. "
    "The cloud itself acts as dynamic support/resistance on Renko."
)

PARAM_GRID = {
    "tenkan_period":  [9, 14],
    "kijun_period":   [26],
    "cooldown":       [10, 20, 30],
    "adx_threshold":  [0, 20, 25],
    "require_tk_cross": [True, False],
}
# 2 × 1 × 3 × 3 × 2 = 36 combinations


def generate_signals(
    df: pd.DataFrame,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    cooldown: int = 20,
    adx_threshold: int = 0,
    require_tk_cross: bool = True,
) -> pd.DataFrame:
    n = len(df)
    brick_up = df["brick_up"].values
    closes = df["Close"].values
    adx_vals = df["adx"].values

    # Compute Ichimoku on the df (not pre-shifted — we shift manually)
    ichi = calc_ichimoku(df, tenkan_period=tenkan_period, kijun_period=kijun_period)
    tenkan = ichi["tenkan"]
    kijun = ichi["kijun"]
    senkou_a = ichi["senkou_a"]
    senkou_b = ichi["senkou_b"]

    # Price vs cloud position: +1 above, -1 below, 0 inside
    cloud_pos = price_vs_cloud(df, ichi, kijun_period=kijun_period)

    long_entry = np.zeros(n, dtype=bool)
    long_exit = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit = np.zeros(n, dtype=bool)

    in_position = False
    trade_dir = 0
    last_trade_bar = -999_999
    warmup = max(kijun_period + 10, 60)

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # Use shifted cloud position (i-1) to avoid lookahead
        cp = cloud_pos[i - 1]

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

        # ── ADX gate ────────────────────────────────────────────────────
        if adx_threshold > 0:
            av = adx_vals[i]
            if np.isnan(av) or av < adx_threshold:
                continue

        # ── Ichimoku signals (use i-1 values to avoid lookahead) ────────
        t_val = tenkan[i - 1]
        k_val = kijun[i - 1]
        if np.isnan(t_val) or np.isnan(k_val):
            continue

        tk_bull = t_val > k_val
        tk_bear = t_val < k_val

        # Long: price above cloud + TK cross bullish + up brick
        if cp == 1 and up:
            if not require_tk_cross or tk_bull:
                long_entry[i] = True
                in_position = True
                trade_dir = 1
                last_trade_bar = i

        # Short: price below cloud + TK cross bearish + down brick
        elif cp == -1 and not up:
            if not require_tk_cross or tk_bear:
                short_entry[i] = True
                in_position = True
                trade_dir = -1
                last_trade_bar = i

    df["long_entry"] = long_entry
    df["long_exit"] = long_exit
    df["short_entry"] = short_entry
    df["short_exit"] = short_exit
    return df
