"""
Python translation of:
  [blackcat] L2 Cyberpunk Value Trend Analyzer by blackcat1402
  https://www.tradingview.com/script/FZyAamU2-blackcat-L2-Cyberpunk-Value-Trend-Analyzer/

Composite value-trend oscillator that normalises close and open within a 75-bar
range using Wilder's RMA smoothing, then generates buy/sell signals based on
level crossings. Includes auxiliary RSI-like and stochastic sub-indicators.

Pine equivalence notes
----------------------
weightedSma(source, length, weight=1):
  alpha = weight / length  →  alpha = 1/length  (with weight=1)
  result = alpha * source + (1-alpha) * result[prev]
  This is Wilder's RMA: equivalent to ewm(alpha=1/length, adjust=False).mean()

valueTrend:
  closeNormSma = RMA20( (close - lowest75) / ((highest75-lowest75)/100) )
  valueTrendRaw = 3 * closeNormSma - 2 * RMA5(closeNormSma)
  valueTrend = valueTrendRaw * 1

RSI-like indicators:
  rsiLike_n = RMA_n(max(price_change,0)) / RMA_n(abs(price_change)) * 100
  Standard Wilder RSI formula using RMA instead of SMA for smoothing.

Williams %R-like:
  = (-200) * (highest(high,60) - close) / (highest(high,60) - lowest(low,60)) + 100
  Rescaled to roughly -100..+100 range (inverted Williams %R × 2).

Signals:
  buySignal  = valueTrend crosses above entry_level (default 30)
  sellSignal = valueTrend crosses below exit_level  (default 75)

Output columns added to df
--------------------------
  bc_vta_value_trend   — main Value Trend oscillator
  bc_vta_dev_index     — deviation index (100 - |pct dev from SMA13|)
  bc_vta_is_overbought — bool: valueTrend > deviationIndex
  bc_vta_rsi6          — RSI-like 6-period (Wilder-smoothed)
  bc_vta_rsi7          — RSI-like 7-period
  bc_vta_rsi13         — RSI-like 13-period
  bc_vta_williams      — Williams %R-like (-100 to +100 scale)
  bc_vta_stoch_k15     — Stochastic K 15-period
  bc_vta_stoch_d_sm    — Smoothed Stochastic D (from K15)
  bc_vta_buy_signal    — bool: valueTrend crosses above entry_level
  bc_vta_sell_signal   — bool: valueTrend crosses below exit_level

Usage
-----
  from indicators.blackcat1402.bc_l2_cyberpunk_value_trend_analyzer import (
      calc_bc_cyberpunk_value_trend_analyzer
  )
  df = calc_bc_cyberpunk_value_trend_analyzer(df)
"""

import numpy as np
import pandas as pd


# ── Internal helpers ──────────────────────────────────────────────────────────

def _rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder's RMA — matches Pine's ta.rma / weightedSma(src, len, weight=1)."""
    return series.ewm(alpha=1.0 / length, adjust=False).mean()


# ── Public function ───────────────────────────────────────────────────────────

def calc_bc_cyberpunk_value_trend_analyzer(
    df: pd.DataFrame,
    entry_level: int = 30,
    exit_level:  int = 75,
) -> pd.DataFrame:
    """
    Python translation of [blackcat] L2 Cyberpunk Value Trend Analyzer.

    Parameters
    ----------
    df          : DataFrame with columns High, Low, Close, Open
    entry_level : buy signal fires when valueTrend crosses above this (default 30)
    exit_level  : sell signal fires when valueTrend crosses below this (default 75)

    Returns
    -------
    df with new columns appended (see module docstring).
    """
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]
    open_ = df["Open"]

    # ── Deviation index ───────────────────────────────────────────────────────
    ma13          = close.rolling(13).mean()
    deviation_idx = 100.0 - (close - ma13).abs() / ma13 * 100.0

    # ── 75-bar range ──────────────────────────────────────────────────────────
    lowest75  = low.rolling(75).min()
    highest75 = high.rolling(75).max()
    range_div = (highest75 - lowest75) / 100.0   # rangeDivider

    # Normalised close/open — guard against zero range
    norm_close = np.where(range_div > 0, (close - lowest75) / range_div, 50.0)
    norm_open  = np.where(range_div > 0, (open_ - lowest75) / range_div, 50.0)
    norm_close = pd.Series(norm_close, index=df.index)
    norm_open  = pd.Series(norm_open,  index=df.index)

    # ── Value Trend ───────────────────────────────────────────────────────────
    close_norm_rma  = _rma(norm_close, 20)
    open_norm_rma   = _rma(norm_open,  20)

    value_trend_raw  = 3.0 * close_norm_rma - 2.0 * _rma(close_norm_rma, 5)
    value_trend      = value_trend_raw          # multiplier = 1

    open_trend_idx   = 3.0 * open_norm_rma - 2.0 * _rma(open_norm_rma, 5)
    # open_trend = 100 - open_trend_idx  (computed but not exposed as a signal column)

    is_overbought = value_trend > deviation_idx

    # ── RSI-like calculations ─────────────────────────────────────────────────
    price_change    = close.diff().fillna(0.0)
    pos_change      = price_change.clip(lower=0.0)
    abs_change      = price_change.abs()

    def _rsi_like(length: int) -> pd.Series:
        avg_up  = _rma(pos_change, length)
        avg_all = _rma(abs_change, length)
        return np.where(avg_all > 0, avg_up / avg_all * 100.0, 50.0)

    rsi6  = pd.Series(_rsi_like(6),  index=df.index)
    rsi7  = pd.Series(_rsi_like(7),  index=df.index)
    rsi13 = pd.Series(_rsi_like(13), index=df.index)

    # ── Williams %R-like ──────────────────────────────────────────────────────
    h60 = high.rolling(60).max()
    l60 = low.rolling(60).min()
    r60 = h60 - l60
    williams = np.where(r60 > 0, (-200.0) * (h60 - close) / r60 + 100.0, 0.0)
    williams = pd.Series(williams, index=df.index)

    # ── Stochastic K15 and smoothed D ────────────────────────────────────────
    h15 = high.rolling(15).max()
    l15 = low.rolling(15).min()
    r15 = h15 - l15
    stoch_k15   = np.where(r15 > 0, (close - l15) / r15 * 100.0, 50.0)
    stoch_k15   = pd.Series(stoch_k15, index=df.index)

    stoch_d4    = _rma(stoch_k15, 4)
    stoch_d_sm  = _rma((_rma(stoch_k15, 4) - 50.0) * 2.0, 3)

    # ── Signal generation ─────────────────────────────────────────────────────
    vt_prev = value_trend.shift(1)

    buy_signal  = (vt_prev <= entry_level) & (value_trend > entry_level)
    sell_signal = (vt_prev >= exit_level)  & (value_trend < exit_level)

    # ── Attach to DataFrame ───────────────────────────────────────────────────
    df = df.copy()
    df["bc_vta_value_trend"]   = value_trend
    df["bc_vta_dev_index"]     = deviation_idx
    df["bc_vta_is_overbought"] = is_overbought
    df["bc_vta_rsi6"]          = rsi6
    df["bc_vta_rsi7"]          = rsi7
    df["bc_vta_rsi13"]         = rsi13
    df["bc_vta_williams"]      = williams
    df["bc_vta_stoch_k15"]     = stoch_k15
    df["bc_vta_stoch_d_sm"]    = stoch_d_sm
    df["bc_vta_buy_signal"]    = buy_signal
    df["bc_vta_sell_signal"]   = sell_signal
    return df
