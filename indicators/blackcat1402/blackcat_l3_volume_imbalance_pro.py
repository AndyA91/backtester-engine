"""
Python translation of:
  [blackcat] L3 Volume Imbalance Pro by blackcat1402
  https://www.tradingview.com/script/7fnpNcbE-blackcat-L3-Volume-Imbalance-Pro/

The original relies on TradingView's request.footprint() API (Premium plan).
Without tick-level bid/ask data the footprint quantities are approximated
from OHLCV:

  buy_volume  ≈ volume × (close − low)  / (high − low)   [Elder buying pressure]
  sell_volume ≈ volume × (high − close) / (high − low)
  imbalance_ratio = buy_volume / sell_volume  (or inverse for sell side)

  "Buy imbalance"  on a bar: buy_volume  > sell_volume × imbalance_threshold / 100
  "Sell imbalance" on a bar: sell_volume > buy_volume  × imbalance_threshold / 100
  "Stacked buy"    : rolling sum of buy-imbalance bars ≥ min_stacked_rows

Traditional indicator calculations (RSI, MACD, VWAP) are derived directly
from OHLCV — no approximation needed.

Composite strength score (−100 to +100) reproduces the Pine scoring logic:
  +20 per buy-imbalance condition, +20 stacked, +15 RSI oversold+imb,
  +15 MACD align, +15 VWAP+imb, +15 buy absorption
  (mirror negatives for sell side)

Vectorisation notes
-------------------
- All imbalance detection: vectorised pandas comparisons.
- Stacked imbalance: rolling().sum() ≥ threshold (no loop).
- RSI: pandas-ewm implementation matching Pine's ta.rsi convention.
- MACD: pandas ewm — exact match to Pine ta.macd.
- VWAP: daily-anchored using groupby-cumsum (requires DatetimeIndex;
  falls back to bar-by-bar cumsum when index is plain integer).
- Divergence: pivot-based detection via rolling max/min with right-offset
  confirmation (identical approach to Footprint Fusion Pro).
- Score: vectorised boolean-to-int arithmetic.

Output columns (bc_ prefix)
----------------------------
  bc_buy_vol              — estimated buy volume (Elder)
  bc_sell_vol             — estimated sell volume
  bc_buy_imbalance        — buy-imbalance flag per bar
  bc_sell_imbalance       — sell-imbalance flag per bar
  bc_max_buy_streak       — length of max consecutive buy-imbalance streak (rolling)
  bc_max_sell_streak      — length of max consecutive sell-imbalance streak (rolling)
  bc_stacked_buy          — stacked buy imbalance (streak ≥ min_stacked_rows)
  bc_stacked_sell         — stacked sell imbalance (streak ≥ min_stacked_rows)
  bc_rsi                  — RSI(rsi_length)
  bc_macd_line            — MACD line
  bc_macd_signal_line     — MACD signal line
  bc_macd_hist            — MACD histogram
  bc_vwap                 — VWAP
  bc_stacked_buy_signal   — stacked buy + bullish candle
  bc_stacked_sell_signal  — stacked sell + bearish candle
  bc_vwap_breakout_buy    — VWAP cross up + buy imbalance
  bc_vwap_breakdown_sell  — VWAP cross down + sell imbalance
  bc_rsi_extreme_buy      — RSI oversold + buy imbalance + RSI rising
  bc_rsi_extreme_sell     — RSI overbought + sell imbalance + RSI falling
  bc_macd_cross_buy       — MACD golden cross + buy imbalance
  bc_macd_cross_sell      — MACD death cross + sell imbalance
  bc_buy_absorption       — price down, buy imbalance, long lower wick
  bc_sell_absorption      — price up, sell imbalance, long upper wick
  bc_bearish_rsi_div      — RSI bearish divergence + sell imbalance
  bc_bullish_rsi_div      — RSI bullish divergence + buy imbalance
  bc_confirmed_bear_div   — confirmed bearish divergence
  bc_confirmed_bull_div   — confirmed bullish divergence
  bc_score                — composite imbalance strength score (−100..+100)

Usage
-----
  from indicators.blackcat1402.blackcat_l3_volume_imbalance_pro import (
      calc_bc_l3_volume_imbalance_pro
  )
  df = calc_bc_l3_volume_imbalance_pro(df)
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Pine-compatible RSI using Wilder's EMA (alpha = 1/length)."""
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _vwap(df: pd.DataFrame) -> pd.Series:
    """
    Daily-anchored VWAP.  Requires DatetimeIndex; falls back to cumulative
    VWAP from bar 0 when the index is not datetime.
    """
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3.0
    tpv  = hlc3 * df["volume"]

    if isinstance(df.index, pd.DatetimeIndex):
        dates    = df.index.date
        cum_tpv  = tpv.groupby(dates).transform("cumsum")
        cum_vol  = df["volume"].groupby(dates).transform("cumsum")
    else:
        cum_tpv  = tpv.cumsum()
        cum_vol  = df["volume"].cumsum()

    return cum_tpv / cum_vol.replace(0, np.nan)


def _pivot_high(series: np.ndarray, left: int, right: int) -> np.ndarray:
    n = len(series)
    result = np.full(n, np.nan)
    for i in range(left, n - right):
        v = series[i]
        if np.isnan(v):
            continue
        if all(series[i - j] <= v for j in range(1, left + 1)) and \
           all(series[i + j] <  v for j in range(1, right + 1)):
            result[i + right] = v
    return result


def _pivot_low(series: np.ndarray, left: int, right: int) -> np.ndarray:
    n = len(series)
    result = np.full(n, np.nan)
    for i in range(left, n - right):
        v = series[i]
        if np.isnan(v):
            continue
        if all(series[i - j] >= v for j in range(1, left + 1)) and \
           all(series[i + j] >  v for j in range(1, right + 1)):
            result[i + right] = v
    return result


def _rolling_max_streak(binary: np.ndarray, window: int) -> np.ndarray:
    """
    For each bar t return the maximum length of consecutive True runs
    ending anywhere within the last `window` bars.

    Equivalent to Pine's 'maxBuyStreak' which resets on each bar.

    Implemented with a running streak counter (O(n)) — fully vectorised-
    equivalent since it requires a single O(n) pass with no nested loops.
    """
    n = len(binary)
    streak = np.zeros(n, dtype=int)
    for i in range(n):
        streak[i] = (streak[i - 1] + 1) if (i > 0 and binary[i]) else int(binary[i])

    # Rolling max streak over last `window` bars
    streak_s = pd.Series(streak)
    return streak_s.rolling(window, min_periods=1).max().values.astype(int)


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def calc_bc_l3_volume_imbalance_pro(
    df: pd.DataFrame,
    imbalance_threshold: float = 300.0,
    min_stacked_rows: int = 3,
    rsi_length: int = 14,
    rsi_overbought: int = 70,
    rsi_oversold: int = 30,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal_period: int = 9,
    divergence_lookback: int = 5,
    streak_window: int = 20,
) -> pd.DataFrame:
    """
    Python translation of [blackcat] L3 Volume Imbalance Pro by blackcat1402.
    - Input:  df with columns open, high, low, close, volume
    - Output: df with new bc_ prefixed columns appended

    Parameters
    ----------
    imbalance_threshold : buy/sell ratio threshold (default 300 = 3×)
    min_stacked_rows    : consecutive imbalance bars to trigger stacked signal
    rsi_length          : RSI period
    rsi_overbought      : RSI overbought level
    rsi_oversold        : RSI oversold level
    macd_fast/slow/signal: MACD parameters
    divergence_lookback : left=right lookback for pivot detection
    streak_window       : rolling window to find max streak length
    """
    df = df.copy()
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    # -----------------------------------------------------------------------
    # 1. Buy / Sell volume approximation
    # -----------------------------------------------------------------------
    candle_range = h - l
    safe_range   = np.where(candle_range > 0, candle_range, 1.0)
    buy_vol  = v * (c - l) / safe_range
    sell_vol = v * (h - c) / safe_range

    # -----------------------------------------------------------------------
    # 2. Per-bar imbalance detection
    #    buy  imbalance: buy_vol  > sell_vol × (imbalance_threshold / 100)
    #    sell imbalance: sell_vol > buy_vol  × (imbalance_threshold / 100)
    # -----------------------------------------------------------------------
    ratio = imbalance_threshold / 100.0
    buy_imb  = buy_vol  > sell_vol * ratio
    sell_imb = sell_vol > buy_vol  * ratio

    # -----------------------------------------------------------------------
    # 3. Stacked imbalance (consecutive streaks)
    # -----------------------------------------------------------------------
    max_buy_streak  = pd.Series(
        _rolling_max_streak(buy_imb.values,  streak_window), index=df.index
    )
    max_sell_streak = pd.Series(
        _rolling_max_streak(sell_imb.values, streak_window), index=df.index
    )
    stacked_buy  = max_buy_streak  >= min_stacked_rows
    stacked_sell = max_sell_streak >= min_stacked_rows

    # -----------------------------------------------------------------------
    # 4. Traditional indicators
    # -----------------------------------------------------------------------
    rsi_val  = _rsi(c, rsi_length)
    vwap_val = _vwap(df)
    atr      = (h - l).rolling(14).mean()

    ema_fast = c.ewm(span=macd_fast,   adjust=False).mean()
    ema_slow = c.ewm(span=macd_slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_ln  = macd_line.ewm(span=macd_signal_period, adjust=False).mean()
    macd_hist  = macd_line - signal_ln

    # -----------------------------------------------------------------------
    # 5. Signals
    # -----------------------------------------------------------------------
    # Stacked imbalance entry
    stacked_buy_sig  = stacked_buy  & (c > o)
    stacked_sell_sig = stacked_sell & (c < o)

    # VWAP breakout / breakdown
    vwap_cross_up   = (c > vwap_val) & (c.shift(1) <= vwap_val.shift(1))
    vwap_cross_down = (c < vwap_val) & (c.shift(1) >= vwap_val.shift(1))
    vwap_breakout_buy   = vwap_cross_up   & buy_imb
    vwap_breakdown_sell = vwap_cross_down & sell_imb

    # RSI extreme + imbalance
    rsi_extreme_buy  = (rsi_val < rsi_oversold)   & buy_imb  & (rsi_val > rsi_val.shift(1))
    rsi_extreme_sell = (rsi_val > rsi_overbought) & sell_imb & (rsi_val < rsi_val.shift(1))

    # MACD cross + imbalance
    macd_cross_buy  = (macd_line > signal_ln) & (macd_line.shift(1) <= signal_ln.shift(1)) & buy_imb
    macd_cross_sell = (macd_line < signal_ln) & (macd_line.shift(1) >= signal_ln.shift(1)) & sell_imb

    # Absorption: price dropping but buy-imbalance (long lower wick)
    candle_body = (c - o).abs()
    lower_wick  = c - l   # length of lower wick (c > o bullish, always >= 0)
    upper_wick  = h - c
    buy_absorption  = (c < o) & buy_imb  & (lower_wick > upper_wick * 2)
    sell_absorption = (c > o) & sell_imb & (upper_wick > lower_wick * 2)

    # -----------------------------------------------------------------------
    # 6. Divergence detection (RSI / MACD + imbalance confirmation)
    # -----------------------------------------------------------------------
    lb = divergence_lookback
    close_arr = c.values.astype(float)
    rsi_arr   = rsi_val.values.astype(float)

    price_ph = _pivot_high(close_arr, lb, lb)
    price_pl = _pivot_low(close_arr,  lb, lb)

    bearish_rsi_div = np.zeros(len(df), dtype=bool)
    bullish_rsi_div = np.zeros(len(df), dtype=bool)

    prev_ph_price = prev_ph_rsi = np.nan
    for i in range(len(price_ph)):
        if not np.isnan(price_ph[i]):
            if (not np.isnan(prev_ph_price)
                    and price_ph[i] > prev_ph_price
                    and rsi_arr[i] < prev_ph_rsi
                    and rsi_arr[i] > rsi_overbought):
                bearish_rsi_div[i] = True
            prev_ph_price = price_ph[i]
            prev_ph_rsi   = rsi_arr[i]

    prev_pl_price = prev_pl_rsi = np.nan
    for i in range(len(price_pl)):
        if not np.isnan(price_pl[i]):
            if (not np.isnan(prev_pl_price)
                    and price_pl[i] < prev_pl_price
                    and rsi_arr[i] > prev_pl_rsi
                    and rsi_arr[i] < rsi_oversold):
                bullish_rsi_div[i] = True
            prev_pl_price = price_pl[i]
            prev_pl_rsi   = rsi_arr[i]

    sell_imb_arr = sell_imb.values
    buy_imb_arr  = buy_imb.values
    confirmed_bear_div = bearish_rsi_div & sell_imb_arr
    confirmed_bull_div = bullish_rsi_div & buy_imb_arr

    # -----------------------------------------------------------------------
    # 7. Composite score (−100..+100)
    # -----------------------------------------------------------------------
    has_buy  = buy_imb.astype(int)
    has_sell = sell_imb.astype(int)
    stb      = stacked_buy.astype(int)
    sts      = stacked_sell.astype(int)
    rsi_os   = ((rsi_val < rsi_oversold)   & buy_imb).astype(int)
    rsi_ob   = ((rsi_val > rsi_overbought) & sell_imb).astype(int)
    macd_aln_b = ((macd_hist > 0) & buy_imb).astype(int)
    macd_aln_s = ((macd_hist < 0) & sell_imb).astype(int)
    vwap_b   = ((c > vwap_val) & buy_imb).astype(int)
    vwap_s   = ((c < vwap_val) & sell_imb).astype(int)
    ba       = buy_absorption.astype(int)
    sa       = sell_absorption.astype(int)

    score = (
        has_buy * 20 + stb * 20 + rsi_os * 15 + macd_aln_b * 15 + vwap_b * 15 + ba * 15
        - has_sell * 20 - sts * 20 - rsi_ob * 15 - macd_aln_s * 15 - vwap_s * 15 - sa * 15
    )

    # -----------------------------------------------------------------------
    # 8. Attach columns
    # -----------------------------------------------------------------------
    df["bc_buy_vol"]              = buy_vol
    df["bc_sell_vol"]             = sell_vol
    df["bc_buy_imbalance"]        = buy_imb.astype(bool)
    df["bc_sell_imbalance"]       = sell_imb.astype(bool)
    df["bc_max_buy_streak"]       = max_buy_streak.astype(int)
    df["bc_max_sell_streak"]      = max_sell_streak.astype(int)
    df["bc_stacked_buy"]          = stacked_buy.astype(bool)
    df["bc_stacked_sell"]         = stacked_sell.astype(bool)
    df["bc_rsi"]                  = rsi_val
    df["bc_macd_line"]            = macd_line
    df["bc_macd_signal_line"]     = signal_ln
    df["bc_macd_hist"]            = macd_hist
    df["bc_vwap"]                 = vwap_val
    df["bc_stacked_buy_signal"]   = stacked_buy_sig.astype(bool)
    df["bc_stacked_sell_signal"]  = stacked_sell_sig.astype(bool)
    df["bc_vwap_breakout_buy"]    = vwap_breakout_buy.astype(bool)
    df["bc_vwap_breakdown_sell"]  = vwap_breakdown_sell.astype(bool)
    df["bc_rsi_extreme_buy"]      = rsi_extreme_buy.astype(bool)
    df["bc_rsi_extreme_sell"]     = rsi_extreme_sell.astype(bool)
    df["bc_macd_cross_buy"]       = macd_cross_buy.astype(bool)
    df["bc_macd_cross_sell"]      = macd_cross_sell.astype(bool)
    df["bc_buy_absorption"]       = buy_absorption.astype(bool)
    df["bc_sell_absorption"]      = sell_absorption.astype(bool)
    df["bc_bearish_rsi_div"]      = pd.Series(bearish_rsi_div, index=df.index)
    df["bc_bullish_rsi_div"]      = pd.Series(bullish_rsi_div, index=df.index)
    df["bc_confirmed_bear_div"]   = pd.Series(confirmed_bear_div, index=df.index)
    df["bc_confirmed_bull_div"]   = pd.Series(confirmed_bull_div, index=df.index)
    df["bc_score"]                = score.astype(int)

    return df
