"""
Python translation of:
  [blackcat] L1 Trend Swing Oscillator by blackcat1402
  https://www.tradingview.com/script/jtgTVb4r-blackcat-L1-Trend-Swing-Oscillator/

Identifies swing buy/sell opportunities using a trend line derived from price
position within a 25-bar high / 11-bar low range, combined with a ratio-
difference measure for strong-buy detection.

Pine equivalence notes
----------------------
trendLine:
  trendRaw = (close - lowest(low,11)) / (highest(high,25) - lowest(low,11)) * 4
  trendLine = ta.ema(trendRaw, 8) * 10      — oscillates roughly 0..40

adaptiveMa:
  adaptiveMa = ma3 > ma21 ? ma21 : ma3    — min(SMA3, SMA21)
  yellowHistCond = adaptiveMa > adaptiveMa[1] and adaptiveMa == ma21
  This fires when the adaptive MA is rising AND currently equals SMA21
  (i.e., SMA3 >= SMA21, so adaptiveMa picked ma21, and ma21 is rising).

prevLowestClose / highestCloseRange:
  prevClose = close[1]
  prevLowestClose = ta.lowest(prevClose, 2)[1]   — 2-bar lowest of prev-close, shifted 1 more
  highestCloseRange = ta.highest(ta.highest(prevClose, 2), 20)
  These are computed in Pine but not used in the signal logic — omitted from output.

Signals:
  sell_signal      = rolling_max(trendLine, 5) > 34  AND price at 10-bar high
  buy_signal_dot   = rolling_min(trendLine, 5) < 3   AND price at 10-bar low
  strong_buy       = rolling_min(SMA5(ratioDiff), 5) < -45  AND price at 10-bar low
  buy_signal       = buy_signal_dot OR strong_buy

Output columns added to df
--------------------------
  bc_tso_trend_line       — main trend oscillator (roughly 0..40)
  bc_tso_pink_hist        — bool: weightedPrice > EMA10(weightedPrice)
  bc_tso_yellow_hist      — bool: adaptiveMa rising and equal to SMA21
  bc_tso_buy              — bool: buy_signal_dot OR strong_buy
  bc_tso_buy_dot          — bool: trend low < 3 and price at low
  bc_tso_strong_buy       — bool: ratio diff low < -45 and price at low
  bc_tso_sell             — bool: trend high > 34 and price at high

Usage
-----
  from indicators.blackcat1402.bc_l1_trend_swing_oscillator import calc_bc_trend_swing_oscillator
  df = calc_bc_trend_swing_oscillator(df)
"""

import numpy as np
import pandas as pd


def calc_bc_trend_swing_oscillator(
    df: pd.DataFrame,
    ema_length:          int   = 10,
    ma3_length:          int   = 3,
    ma21_length:         int   = 21,
    trend_ema_length:    int   = 8,
    lookback_high:       int   = 5,
    lookback_low:        int   = 5,
    price_lookback:      int   = 10,
    trend_high_thresh:   float = 34.0,
    trend_low_thresh:    float = 3.0,
    ratio_diff_thresh:   float = -45.0,
) -> pd.DataFrame:
    """
    Python translation of [blackcat] L1 Trend Swing Oscillator.

    Parameters
    ----------
    df                : DataFrame with columns High, Low, Close, Open
    ema_length        : EMA period for pink histogram condition (default 10)
    ma3_length        : short SMA period for adaptive MA (default 3)
    ma21_length       : long SMA period for adaptive MA (default 21)
    trend_ema_length  : EMA period for trend line smoothing (default 8)
    lookback_high     : rolling window for trendLine max check (default 5)
    lookback_low      : rolling window for trendLine min check (default 5)
    price_lookback    : window for price extreme detection (default 10)
    trend_high_thresh : sell threshold on rolling trendLine max (default 34.0)
    trend_low_thresh  : buy threshold on rolling trendLine min (default 3.0)
    ratio_diff_thresh : strong-buy threshold on rolling ratioDiffMa min (default -45.0)

    Returns
    -------
    df with new columns appended (see module docstring).
    """
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]
    open_ = df["Open"]

    # ── Weighted price ────────────────────────────────────────────────────────
    wp = (2.0 * close + high + low + open_) / 5.0

    # ── Pink histogram condition ──────────────────────────────────────────────
    wp_ema       = wp.ewm(span=ema_length, adjust=False).mean()
    pink_hist    = wp > wp_ema

    # ── Adaptive MA and yellow histogram ─────────────────────────────────────
    ma3         = close.rolling(ma3_length).mean()
    ma21        = close.rolling(ma21_length).mean()
    adaptive_ma = np.where(ma3 > ma21, ma21.values, ma3.values)
    adaptive_ma = pd.Series(adaptive_ma, index=df.index)

    yellow_hist = (adaptive_ma > adaptive_ma.shift(1)) & (adaptive_ma == ma21)

    # ── Ratio difference ──────────────────────────────────────────────────────
    h21 = high.rolling(21).max()
    l21 = low.rolling(21).min()
    rng21 = h21 - l21

    upper_ratio = np.where(rng21 > 0, (h21 - close) / rng21 * 100.0, 50.0)
    lower_ratio = np.where(rng21 > 0, (close - l21) / rng21 * 100.0, 50.0)
    upper_ratio = pd.Series(upper_ratio, index=df.index)
    lower_ratio = pd.Series(lower_ratio, index=df.index)
    ratio_diff  = lower_ratio - upper_ratio

    # ── Trend line ────────────────────────────────────────────────────────────
    h25 = high.rolling(25).max()
    l11 = low.rolling(11).min()
    rng_tl = h25 - l11

    trend_raw  = np.where(rng_tl > 0, (close - l11) / rng_tl * 4.0, 0.0)
    trend_raw  = pd.Series(trend_raw, index=df.index)
    trend_line = trend_raw.ewm(span=trend_ema_length, adjust=False).mean() * 10.0

    # ── Price extreme detection ───────────────────────────────────────────────
    h_close_lb = close.rolling(price_lookback).max()
    h_high_lb  = high.rolling(price_lookback).max()
    l_close_lb = close.rolling(price_lookback).min()
    l_low_lb   = low.rolling(price_lookback).min()

    is_at_high = (close == h_close_lb) | (high == h_high_lb)
    is_at_low  = (close == l_close_lb) | (low == l_low_lb)

    # ── Trend line extremes ───────────────────────────────────────────────────
    tl_max = trend_line.rolling(lookback_high).max()
    tl_min = trend_line.rolling(lookback_low).min()

    sell_signal    = (tl_max > trend_high_thresh) & is_at_high
    buy_signal_dot = (tl_min < trend_low_thresh)  & is_at_low

    # ── Strong buy (ratio diff) ───────────────────────────────────────────────
    ratio_diff_ma  = ratio_diff.rolling(5).mean()
    ratio_diff_min = ratio_diff_ma.rolling(lookback_low).min()
    strong_buy     = (ratio_diff_min < ratio_diff_thresh) & is_at_low

    buy_signal = buy_signal_dot | strong_buy

    # ── Attach to DataFrame ───────────────────────────────────────────────────
    df = df.copy()
    df["bc_tso_trend_line"] = trend_line
    df["bc_tso_pink_hist"]  = pink_hist
    df["bc_tso_yellow_hist"] = yellow_hist
    df["bc_tso_buy"]         = buy_signal
    df["bc_tso_buy_dot"]     = buy_signal_dot
    df["bc_tso_strong_buy"]  = strong_buy
    df["bc_tso_sell"]        = sell_signal
    return df
