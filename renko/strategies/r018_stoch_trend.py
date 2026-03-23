"""
R018: Stochastic Trend Entry — Oscillator Extremes in Trend Direction

Entry: Stochastic %K crosses above %D from below oversold zone (long) or
       crosses below %D from above overbought zone (short), with trend
       confirmation from Supertrend or EMA direction.

This combines oscillator *timing* (buy at oversold in uptrend, sell at
overbought in downtrend) with trend *direction* confirmation. Unlike R007
(RSI mean-reversion, which failed), this trades WITH the trend — the
stochastic just times the entry at a pullback.

Optional gates:
  - trend_filter:   "st" (Supertrend), "ema" (EMA21 slope), "both"
  - adx_threshold:  minimum ADX strength
  - os/ob levels:   oversold/overbought thresholds

Exit: first opposing brick.

Data: OANDA_EURUSD, 1S renko 0.0004.csv
"""

import numpy as np
import pandas as pd

DESCRIPTION = "Stochastic pullback entry in trend direction (ST/EMA trend filter)"

HYPOTHESIS = (
    "R007 (RSI mean-reversion) failed because it fights the trend. This strategy "
    "uses Stochastic to time entries WITH the trend — buying pullbacks to oversold "
    "in an uptrend, selling rallies to overbought in a downtrend. On Renko, "
    "stochastic extremes represent genuine price exhaustion (each brick = fixed move)."
)

RENKO_FILE      = "OANDA_EURUSD, 1S renko 0.0004.csv"
COMMISSION_PCT  = 0.0046
INITIAL_CAPITAL = 1000.0

PARAM_GRID = {
    "os_level":      [20, 30],
    "ob_level":      [70, 80],
    "trend_filter":  ["st", "ema", "both"],
    "adx_threshold": [0, 20],
    "cooldown":      [5, 10, 20],
}
# 2 x 2 x 3 x 2 x 3 = 72 combos


def generate_signals(
    df:            pd.DataFrame,
    os_level:      int = 20,
    ob_level:      int = 80,
    trend_filter:  str = "st",
    adx_threshold: int = 0,
    cooldown:      int = 10,
) -> pd.DataFrame:
    n = len(df)
    brick_up = df["brick_up"].values
    stoch_k  = df["stoch_k"].values
    stoch_d  = df["stoch_d"].values
    st_dir   = df["st_dir"].values
    ema21    = df["ema21"].values
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

        # --- NaN guard ---
        if (np.isnan(stoch_k[i]) or np.isnan(stoch_d[i])
                or np.isnan(stoch_k[i-1]) or np.isnan(stoch_d[i-1])):
            continue

        # --- Stochastic cross from extreme ---
        bull_cross = (stoch_k[i] > stoch_d[i] and stoch_k[i-1] <= stoch_d[i-1]
                      and stoch_k[i-1] < os_level)
        bear_cross = (stoch_k[i] < stoch_d[i] and stoch_k[i-1] >= stoch_d[i-1]
                      and stoch_k[i-1] > ob_level)

        if not bull_cross and not bear_cross:
            continue

        # --- Trend filter ---
        if trend_filter in ("st", "both"):
            if np.isnan(st_dir[i]):
                continue
            if bull_cross and st_dir[i] != 1:
                continue
            if bear_cross and st_dir[i] != -1:
                continue

        if trend_filter in ("ema", "both"):
            if np.isnan(ema21[i]) or np.isnan(ema21[i-1]):
                continue
            ema_rising = ema21[i] > ema21[i-1]
            if bull_cross and not ema_rising:
                continue
            if bear_cross and ema_rising:
                continue

        # --- ADX gate ---
        if adx_threshold > 0:
            adx_val = adx_vals[i]
            if np.isnan(adx_val) or adx_val < adx_threshold:
                continue

        # --- Entry ---
        if bull_cross:
            long_entry[i]  = True
            in_position    = True
            trade_dir      = 1
            last_trade_bar = i

        elif bear_cross:
            short_entry[i] = True
            in_position    = True
            trade_dir      = -1
            last_trade_bar = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
