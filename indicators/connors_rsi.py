"""
Connors RSI (CRSI) — Composite Mean-Reversion Oscillator

Combines three components into a single 0-100 oscillator optimized for
mean-reversion entries:
  1. RSI(close, rsi_period)          — standard momentum
  2. RSI(streak, streak_period)      — RSI of the consecutive up/down streak length
  3. PercentRank(ROC(1), rank_period) — where today's 1-bar return sits historically

CRSI = (RSI + StreakRSI + PercentRank) / 3

Extreme readings (< 10 or > 90) have historically high reversal rates,
making this ideal for ADX-gated mean-reversion strategies.

Usage:
    from indicators.connors_rsi import calc_connors_rsi

    result = calc_connors_rsi(df, rsi_period=3, streak_period=2, rank_period=100)
    # result["crsi"]         — Connors RSI (0-100)
    # result["rsi"]          — standard RSI component
    # result["streak_rsi"]   — streak RSI component
    # result["pct_rank"]     — percent rank component (0-100)

Interpretation:
    CRSI < 10  → deeply oversold, high probability of bounce (long entry)
    CRSI > 90  → deeply overbought, high probability of pullback (short entry)
    Best paired with ADX < 25 regime filter for mean-reversion setups
    Faster than standard RSI — designed for short holding periods (1-5 bars)
"""

import numpy as np
import pandas as pd
from indicators.rsi import calc_rsi


def calc_connors_rsi(
    df: pd.DataFrame,
    rsi_period: int = 3,
    streak_period: int = 2,
    rank_period: int = 100,
) -> dict:
    """
    Parameters
    ----------
    df           : DataFrame with 'Close'
    rsi_period   : RSI period on close (default 3 — short for mean reversion)
    streak_period: RSI period on streak length (default 2)
    rank_period  : Lookback for percent rank of 1-bar ROC (default 100)

    Returns
    -------
    dict with keys: crsi, rsi, streak_rsi, pct_rank (all numpy arrays, 0-100)
    """
    close = df["Close"].values.astype(float)
    n = len(close)

    # --- Component 1: Standard RSI ---
    rsi_vals = calc_rsi(df, period=rsi_period)["rsi"]

    # --- Component 2: Streak RSI ---
    # Streak: consecutive up (+) or down (-) closes
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            streak[i] = streak[i - 1] + 1 if streak[i - 1] > 0 else 1
        elif close[i] < close[i - 1]:
            streak[i] = streak[i - 1] - 1 if streak[i - 1] < 0 else -1
        else:
            streak[i] = 0

    # RSI of the streak series
    streak_df = pd.DataFrame({"Close": streak})
    streak_rsi = calc_rsi(streak_df, period=streak_period)["rsi"]

    # --- Component 3: Percent Rank of 1-bar ROC ---
    roc = np.zeros(n)
    roc[1:] = (close[1:] - close[:-1]) / np.where(close[:-1] != 0, close[:-1], 1.0)

    pct_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = roc[i - rank_period + 1 : i + 1]
        pct_rank[i] = np.sum(window < roc[i]) / rank_period * 100.0

    # --- Composite ---
    crsi = (rsi_vals + streak_rsi + pct_rank) / 3.0

    return {
        "crsi": crsi,
        "rsi": rsi_vals,
        "streak_rsi": streak_rsi,
        "pct_rank": pct_rank,
    }
