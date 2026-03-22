"""
Python translation of:
  [blackcat] L2 Napoleon Mille-feuille by blackcat1402
  https://www.tradingview.com/script/AwSYwhYK-blackcat-L2-Napoleon-Mille-feuille/

A layered channel strength system (0–9 score) that counts how many of 9
band-vs-channel conditions are satisfied. Higher scores indicate the short-
term price bands are trading above the medium-term channel — a bullish
structure. Buy/Sell signals fire on score crossovers.

Pine equivalence notes
----------------------
Channel construction:
  midPrice25    = SMA25( (high+low)/2 )         — 25-bar mid-price MA
  upperChannel25 = midPrice25 * 1.15
  lowerChannel25 = midPrice25 * 0.95
  midChannel25   = (upper + lower) / 2  = midPrice25 * 1.05

  midPrice5   = (SMA5(close) + SMA5(open)) / 2
  upperBand5  = midPrice5 * 1.06
  lowerBand5  = midPrice5 * 0.98
  midBand5    = (upper + lower) / 2  = midPrice5 * 1.02

Strength score (0–9):
  cond1 = upperBand5 > lowerChannel25    (short upper > medium lower)
  cond2 = upperBand5 > midChannel25      (short upper > medium mid)
  cond3 = upperBand5 > upperChannel25    (short upper > medium upper)
  cond4 = midBand5   > lowerChannel25
  cond5 = midBand5   > midChannel25
  cond6 = midBand5   > upperChannel25
  cond7 = lowerBand5 > lowerChannel25    (short lower > medium lower)
  cond8 = lowerBand5 > midChannel25
  cond9 = lowerBand5 > upperChannel25    (most bullish — all bands above)
  strengthScore = sum of 9 booleans

Signals:
  sellLabelTrigger = crossover(strengthScore, sellLabelThreshold)  — score crosses UP above 4
  buyLabelTrigger  = crossunder(strengthScore, buyLabelThreshold)  — score crosses DOWN below 2

Output columns added to df
--------------------------
  bc_nap_strength_score  — integer 0–9 (count of satisfied band/channel conditions)
  bc_nap_buy_trigger     — bool: strength score crosses below buy threshold (2)
  bc_nap_sell_trigger    — bool: strength score crosses above sell threshold (4)
  bc_nap_trending_up     — bool: strength score > previous bar's score
  bc_nap_trending_down   — bool: strength score < previous bar's score

Usage
-----
  from indicators.blackcat1402.bc_l2_napoleon_mille_feuille import (
      calc_bc_napoleon_mille_feuille
  )
  df = calc_bc_napoleon_mille_feuille(df)
"""

import numpy as np
import pandas as pd


def calc_bc_napoleon_mille_feuille(
    df: pd.DataFrame,
    sell_threshold: int = 4,
    buy_threshold:  int = 2,
) -> pd.DataFrame:
    """
    Python translation of [blackcat] L2 Napoleon Mille-feuille.

    Parameters
    ----------
    df             : DataFrame with columns High, Low, Close, Open
    sell_threshold : strength score level for sell trigger crossover (default 4)
    buy_threshold  : strength score level for buy trigger crossunder (default 2)

    Returns
    -------
    df with new columns appended (see module docstring).
    """
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]
    open_ = df["Open"]

    # ── 25-period mid-price channels ──────────────────────────────────────────
    mid_price25     = ((low + high) / 2.0).rolling(25).mean()
    upper_channel25 = mid_price25 * 1.15
    lower_channel25 = mid_price25 * 0.95
    mid_channel25   = (upper_channel25 + lower_channel25) / 2.0  # = mid_price25 * 1.05

    # ── 5-period price bands ──────────────────────────────────────────────────
    ma_close5   = close.rolling(5).mean()
    ma_open5    = open_.rolling(5).mean()
    mid_price5  = (ma_close5 + ma_open5) / 2.0
    upper_band5 = mid_price5 * 1.06
    lower_band5 = mid_price5 * 0.98
    mid_band5   = (upper_band5 + lower_band5) / 2.0  # = mid_price5 * 1.02

    # ── 9 band/channel conditions ─────────────────────────────────────────────
    cond1 = upper_band5 > lower_channel25
    cond2 = upper_band5 > mid_channel25
    cond3 = upper_band5 > upper_channel25
    cond4 = mid_band5   > lower_channel25
    cond5 = mid_band5   > mid_channel25
    cond6 = mid_band5   > upper_channel25
    cond7 = lower_band5 > lower_channel25
    cond8 = lower_band5 > mid_channel25
    cond9 = lower_band5 > upper_channel25

    strength_score = (
        cond1.astype(int) + cond2.astype(int) + cond3.astype(int)
        + cond4.astype(int) + cond5.astype(int) + cond6.astype(int)
        + cond7.astype(int) + cond8.astype(int) + cond9.astype(int)
    )

    # ── Crossover/crossunder signals ──────────────────────────────────────────
    score_prev = strength_score.shift(1)

    sell_trigger = (score_prev <= sell_threshold) & (strength_score > sell_threshold)
    buy_trigger  = (score_prev >= buy_threshold)  & (strength_score < buy_threshold)

    trending_up   = strength_score > score_prev
    trending_down = strength_score < score_prev

    # ── Attach to DataFrame ───────────────────────────────────────────────────
    df = df.copy()
    df["bc_nap_strength_score"] = strength_score
    df["bc_nap_buy_trigger"]    = buy_trigger
    df["bc_nap_sell_trigger"]   = sell_trigger
    df["bc_nap_trending_up"]    = trending_up
    df["bc_nap_trending_down"]  = trending_down
    return df
