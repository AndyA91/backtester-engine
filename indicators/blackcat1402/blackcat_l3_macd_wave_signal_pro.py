"""
Python translation of:
  [blackcat] L3 MACD Wave Signal Pro by blackcat1402
  https://www.tradingview.com/script/mAEAwacE-blackcat-L3-MACD-Wave-Signal-Pro/

A MACD indicator enriched with a "multi-SMA fan" (40 SMAs of MID price,
periods 1–40) and a "Line Convergence" metric that measures how aligned
short-term SMAs are relative to the 40-bar SMA.

Core calculations
-----------------
  MID    = (7×close + low + open + high) / 10
  MA0    = SMA(MID, 40)           ← baseline (centre of the fan)
  MAk    = SMA(MID, k)  k=1..40
  diffk  = MAk − MA0              ← how far each short MA is from centre
  LC     = mean(diff6..diff20) / 15  ← "Line Convergence" of mid-range SMAs
  DIFF   = EMA(close, 12) − EMA(close, 26)
  DEA    = EMA(DIFF, 9)
  MACD   = 2 × (DIFF − DEA)      ← standard histogram (scaled × 1)

Line Convergence (LC) interpretation
-------------------------------------
  LC rising  (LC > LC[1] AND LC[1] < LC[2]): "lazy line divergence" — fan opening bullishly
  LC falling (LC < LC[1] AND LC[1] > LC[2]): "lazy line convergence" — fan compressing bearishly

Signals (same as Pine):
  Buy:  DIFF crosses above DEA (golden cross)
  Sell: DIFF crosses below DEA (death cross)

Vectorisation notes
-------------------
- MID, MACD: element-wise pandas operations.
- MA1..MA40: 40 pandas rolling().mean() calls.  Each is O(n) and the total
  cost is O(40n) ≈ 800 k ops for n=20 000 — trivially fast.
- Line Convergence: element-wise average of maDiff6..maDiff20 (15 series).
- No loops over the price series are needed.

Output columns (bc_ prefix)
----------------------------
  bc_mid           — MID price formula
  bc_ma0           — 40-bar SMA of MID (the fan centre)
  bc_diff          — MACD fast line (EMA12 − EMA26)
  bc_dea           — MACD slow line (EMA9 of DIFF)
  bc_macd_hist     — MACD histogram (2 × (DIFF − DEA))
  bc_lc            — Line Convergence (mean of maDiff6..maDiff20)
  bc_macd_state    — histogram colour-state integer:
                       0 = rising above zero  (magenta in Pine)
                       1 = falling above zero (yellow)
                       2 = falling below zero (yellow)
                       3 = rising below zero  (white)
  bc_lc_state      — LC direction:  1 = diverging (bullish fan), -1 = converging, 0 = flat
  bc_buy_signal    — DIFF crosses above DEA (bool)
  bc_sell_signal   — DIFF crosses below DEA (bool)

Usage
-----
  from indicators.blackcat1402.blackcat_l3_macd_wave_signal_pro import (
      calc_bc_l3_macd_wave_signal_pro
  )
  df = calc_bc_l3_macd_wave_signal_pro(df)
"""

import numpy as np
import pandas as pd


def calc_bc_l3_macd_wave_signal_pro(
    df: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    fan_centre: int = 40,
    lc_low: int = 6,
    lc_high: int = 20,
) -> pd.DataFrame:
    """
    Python translation of [blackcat] L3 MACD Wave Signal Pro by blackcat1402.
    - Input:  df with columns open, high, low, close, volume
    - Output: df with new bc_ prefixed columns appended

    Parameters
    ----------
    macd_fast   : fast EMA period for MACD (default 12)
    macd_slow   : slow EMA period for MACD (default 26)
    macd_signal : signal EMA period for MACD (default 9)
    fan_centre  : period of the baseline MA (fan centre, default 40)
    lc_low      : first MA period included in Line Convergence (default 6)
    lc_high     : last  MA period included in Line Convergence (default 20)
    """
    df = df.copy()
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    # -----------------------------------------------------------------------
    # 1. MID price
    #    MID = (7×close + low + open + high) / 10
    # -----------------------------------------------------------------------
    mid = (7.0 * c + l + o + h) / 10.0

    # -----------------------------------------------------------------------
    # 2. Fan centre: MA0 = SMA(MID, fan_centre)
    # -----------------------------------------------------------------------
    ma0 = mid.rolling(fan_centre).mean()

    # -----------------------------------------------------------------------
    # 3. Multi-SMA fan: MAk = SMA(MID, k) for k = 1 .. fan_centre
    #    diffk = MAk − MA0
    #    Line Convergence = mean(diff[lc_low]..diff[lc_high])
    #
    #    All 40 rolling means computed in a single list comprehension.
    #    Summing the LC range: sum 15 aligned Series, divide by 15.
    # -----------------------------------------------------------------------
    lc_diffs = [
        mid.rolling(k).mean() - ma0
        for k in range(lc_low, lc_high + 1)
    ]
    lc = sum(lc_diffs) / len(lc_diffs)

    # LC direction state
    lc_prev1 = lc.shift(1)
    lc_prev2 = lc.shift(2)
    lc_state = pd.Series(0, index=df.index)
    lc_state = lc_state.where(~((lc > lc_prev1) & (lc_prev1 < lc_prev2)),  1)  # diverging
    lc_state = lc_state.where(~((lc < lc_prev1) & (lc_prev1 > lc_prev2)), -1)  # converging

    # -----------------------------------------------------------------------
    # 4. MACD
    #    DIFF = EMA(close, fast) - EMA(close, slow)
    #    DEA  = EMA(DIFF, signal)
    #    MACD = 2 × (DIFF - DEA)    [Pine: SJ=1 scaling factor]
    # -----------------------------------------------------------------------
    ema_fast = c.ewm(span=macd_fast,   adjust=False).mean()
    ema_slow = c.ewm(span=macd_slow,   adjust=False).mean()
    diff_line = ema_fast - ema_slow
    dea_line  = diff_line.ewm(span=macd_signal, adjust=False).mean()
    macd_hist = 2.0 * (diff_line - dea_line)

    # -----------------------------------------------------------------------
    # 5. Histogram state
    #    0: rising  above zero  (macd >= 0 AND macd >= prev)
    #    1: falling above zero  (macd >= 0 AND macd <  prev)
    #    2: falling below zero  (macd <  0 AND macd <  prev)
    #    3: rising  below zero  (macd <  0 AND macd >= prev)
    # -----------------------------------------------------------------------
    macd_prev = macd_hist.shift(1)
    above = macd_hist >= 0
    rising = macd_hist >= macd_prev

    state = pd.Series(2, index=df.index)           # default: falling below zero
    state = state.where(~(above & rising),   0)    # rising above zero
    state = state.where(~(above & ~rising),  1)    # falling above zero
    state = state.where(~(~above & rising),  3)    # rising below zero
    # falling below zero stays at 2 (initial default)

    # -----------------------------------------------------------------------
    # 6. Buy / Sell signals (crossovers)
    # -----------------------------------------------------------------------
    buy_signal  = (diff_line > dea_line) & (diff_line.shift(1) <= dea_line.shift(1))
    sell_signal = (diff_line < dea_line) & (diff_line.shift(1) >= dea_line.shift(1))

    # -----------------------------------------------------------------------
    # 7. Attach columns
    # -----------------------------------------------------------------------
    df["bc_mid"]        = mid
    df["bc_ma0"]        = ma0
    df["bc_diff"]       = diff_line
    df["bc_dea"]        = dea_line
    df["bc_macd_hist"]  = macd_hist
    df["bc_lc"]         = lc
    df["bc_macd_state"] = state.astype(int)
    df["bc_lc_state"]   = lc_state.astype(int)
    df["bc_buy_signal"] = buy_signal.astype(bool)
    df["bc_sell_signal"] = sell_signal.astype(bool)

    return df
