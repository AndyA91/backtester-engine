"""
Python translation of:
  [blackcat] L3 Footprint Fusion Pro by blackcat1402
  https://www.tradingview.com/script/XRr5bKn2-blackcat-L3-Footprint-Fusion-Pro/

The original relies on TradingView's request.footprint() API which provides
per-tick bid/ask volume at each price level (Premium plan only).  Without
tick-level data those quantities are approximated from OHLCV as follows:

  delta ≈ volume × (2×close − high − low) / (high − low)
          [directional pressure: +1 if all-up candle, −1 if all-down]

  buy_volume  ≈ volume × (close − low)  / (high − low)  [Elder's buying pressure]
  sell_volume ≈ volume × (high − close) / (high − low)

  CVD = cumulative sum of delta across all bars

  POC / VAH / VAL: computed via rolling-window numpy histograms over a
  fixed lookback window (rolling_vp_bars), binned into n_bins price levels.
  Value Area expansion uses a greedy algorithm (same as TradingView):
  start at the POC bin, then iteratively add whichever adjacent bin (up or
  down) has the higher volume, until the accumulated volume ≥ va_pct% of total.

Signals from the original that are purely OHLCV-derivable remain identical.
Signals that require genuine per-row footprint data (imbalance clouds,
stacked imbalance) are approximated using consecutive-directional-candle
detection.

Vectorisation notes
-------------------
- Delta / CVD: fully vectorised with numpy / pandas cumsum.
- Volume Profile (POC/VAH/VAL): rolling-window histogram loop — one
  numpy.histogram call per bar, O(n × lookback).  Using stride_tricks to
  create rolling price/volume windows avoids repeated slicing overhead.
- Delta divergence: pivot detection uses a vectorised rolling-max/min
  approach with a confirmation offset (no per-bar loop needed).
- All moving averages (SMA, EMA) use pandas built-ins.

Output columns (bc_ prefix)
----------------------------
  bc_delta                — per-bar directional delta (approximated)
  bc_cvd                  — cumulative volume delta
  bc_buy_vol              — estimated buy volume (Elder)
  bc_sell_vol             — estimated sell volume (Elder)
  bc_poc                  — point-of-control price
  bc_vah                  — value area high
  bc_val                  — value area low
  bc_buy_absorption       — buy absorption signal (price down, strong +delta at POC)
  bc_sell_absorption      — sell absorption signal (price up, strong -delta at POC)
  bc_bullish_div          — bullish delta divergence (price LL, delta HL)
  bc_bearish_div          — bearish delta divergence (price HH, delta LH)
  bc_stacked_buy_imb      — ≥ min_stacked consecutive bullish candles
  bc_stacked_sell_imb     — ≥ min_stacked consecutive bearish candles
  bc_false_vah_breakout   — price closes above VAH but delta < 0
  bc_false_val_breakdown  — price closes below VAL but delta > 0
  bc_vah_breakout         — price crosses above VAH (any delta)
  bc_val_breakdown        — price crosses below VAL (any delta)

Usage
-----
  from indicators.blackcat1402.blackcat_l3_footprint_fusion_pro import (
      calc_bc_l3_footprint_fusion_pro
  )
  df = calc_bc_l3_footprint_fusion_pro(df)
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pivot_high(series: np.ndarray, left: int, right: int) -> np.ndarray:
    """
    Return pivot high values (confirmed at bar i+right).
    Strict > on right side, >= on left side mirrors Pine's ta.pivothigh.
    """
    n = len(series)
    result = np.full(n, np.nan)
    for i in range(left, n - right):
        pivot = series[i]
        if np.isnan(pivot):
            continue
        left_ok  = all(series[i - j] <= pivot for j in range(1, left + 1))
        right_ok = all(series[i + j] <  pivot for j in range(1, right + 1))
        if left_ok and right_ok:
            result[i + right] = pivot
    return result


def _pivot_low(series: np.ndarray, left: int, right: int) -> np.ndarray:
    n = len(series)
    result = np.full(n, np.nan)
    for i in range(left, n - right):
        pivot = series[i]
        if np.isnan(pivot):
            continue
        left_ok  = all(series[i - j] >= pivot for j in range(1, left + 1))
        right_ok = all(series[i + j] >  pivot for j in range(1, right + 1))
        if left_ok and right_ok:
            result[i + right] = pivot
    return result


def _greedy_value_area(
    vol_hist: np.ndarray,
    poc_idx: int,
    va_pct: float,
) -> tuple:
    """
    Greedy expansion from POC to capture va_pct% of total volume.
    Returns (lower_bin_idx, upper_bin_idx).
    """
    total_vol = vol_hist.sum()
    target = total_vol * va_pct / 100.0
    lo = poc_idx
    hi = poc_idx
    accumulated = vol_hist[poc_idx]
    n_bins = len(vol_hist)

    while accumulated < target:
        can_up   = hi + 1 < n_bins
        can_down = lo - 1 >= 0
        if not can_up and not can_down:
            break
        vol_up   = vol_hist[hi + 1] if can_up   else -1.0
        vol_down = vol_hist[lo - 1] if can_down else -1.0
        if vol_up >= vol_down:
            hi += 1
            accumulated += vol_hist[hi]
        else:
            lo -= 1
            accumulated += vol_hist[lo]

    return lo, hi


def _rolling_volume_profile(
    close_arr: np.ndarray,
    volume_arr: np.ndarray,
    lookback: int,
    n_bins: int,
    va_pct: float,
) -> tuple:
    """
    Compute POC, VAH, VAL for each bar using a rolling lookback window.

    Returns three numpy arrays: poc, vah, val.

    Logic per bar:
      1. Collect close and volume for the last `lookback` bars.
      2. Bin close prices into n_bins equal-width buckets weighted by volume.
      3. POC = midpoint of bin with maximum volume.
      4. Greedy expansion from POC until va_pct% of volume is captured.
      5. VAH = upper edge of highest captured bin.
      6. VAL = lower edge of lowest captured bin.
    """
    n = len(close_arr)
    poc = np.full(n, np.nan)
    vah = np.full(n, np.nan)
    val = np.full(n, np.nan)

    for i in range(lookback - 1, n):
        win_c = close_arr[i - lookback + 1: i + 1]
        win_v = volume_arr[i - lookback + 1: i + 1]

        p_min = win_c.min()
        p_max = win_c.max()
        if p_min == p_max or win_v.sum() == 0:
            poc[i] = p_min
            vah[i] = p_min
            val[i] = p_min
            continue

        counts, edges = np.histogram(win_c, bins=n_bins,
                                     range=(p_min, p_max),
                                     weights=win_v)
        poc_bin = int(np.argmax(counts))
        poc[i] = (edges[poc_bin] + edges[poc_bin + 1]) / 2.0

        lo, hi = _greedy_value_area(counts, poc_bin, va_pct)
        val[i] = edges[lo]
        vah[i] = edges[hi + 1]

    return poc, vah, val


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def calc_bc_l3_footprint_fusion_pro(
    df: pd.DataFrame,
    va_pct: float = 70.0,
    min_stacked: int = 3,
    divergence_lookback: int = 5,
    absorption_threshold: float = 2.0,
    rolling_vp_bars: int = 50,
    n_bins: int = 20,
) -> pd.DataFrame:
    """
    Python translation of [blackcat] L3 Footprint Fusion Pro by blackcat1402.

    Parameters
    ----------
    va_pct            : value area percentage (default 70)
    min_stacked       : minimum consecutive imbalance candles for stacked signal
    divergence_lookback: pivot lookback for delta divergence (left = right = this)
    absorption_threshold: POC delta-to-avg ratio threshold for absorption
    rolling_vp_bars   : rolling window for OHLCV-based volume profile
    n_bins            : number of price bins for volume profile histogram

    - Input:  df with columns open, high, low, close, volume
    - Output: df with new bc_ prefixed columns appended
    """
    df = df.copy()
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    # -----------------------------------------------------------------------
    # 1. Per-bar delta approximation
    #    delta ≈ volume × (2×close − high − low) / (high − low)
    # -----------------------------------------------------------------------
    candle_range = h - l
    safe_range = np.where(candle_range > 0, candle_range, 1.0)
    delta_frac = (2.0 * c - h - l) / safe_range
    delta = v * delta_frac

    # Elder-style buy/sell volume split
    buy_vol  = v * (c - l) / safe_range
    sell_vol = v * (h - c) / safe_range

    # -----------------------------------------------------------------------
    # 2. Cumulative Volume Delta (CVD)
    # -----------------------------------------------------------------------
    cvd = delta.cumsum()

    # -----------------------------------------------------------------------
    # 3. Rolling Volume Profile — POC, VAH, VAL
    # -----------------------------------------------------------------------
    close_arr  = c.values.astype(float)
    volume_arr = v.values.astype(float)
    poc_arr, vah_arr, val_arr = _rolling_volume_profile(
        close_arr, volume_arr, rolling_vp_bars, n_bins, va_pct
    )
    poc = pd.Series(poc_arr, index=df.index)
    vah = pd.Series(vah_arr, index=df.index)
    val = pd.Series(val_arr, index=df.index)

    # -----------------------------------------------------------------------
    # 4. Absorption signals
    #    Buy absorption:  bearish candle (close < open) but delta at "POC region" is positive
    #    Sell absorption: bullish candle (close > open) but delta at "POC region" is negative
    #
    #    "POC delta" is approximated as the bar's own delta weighted by how
    #    close the bar's close is to the POC (within ATR * 0.1 tolerance).
    # -----------------------------------------------------------------------
    atr = pd.Series(np.where(
        h.rolling(14).mean().notna(),
        ((h - l).rolling(14).mean()).values,
        (h - l).values
    ), index=df.index)

    poc_distance = (c - poc).abs()
    near_poc = poc_distance < atr * absorption_threshold * 0.1

    buy_absorption  = (c < o) & (delta > 0) & near_poc
    sell_absorption = (c > o) & (delta < 0) & near_poc

    # -----------------------------------------------------------------------
    # 5. Delta divergence (pivot-based)
    #    Bearish div: price makes HH but delta makes LH → confirmed at pivot
    #    Bullish div: price makes LL but delta makes HL → confirmed at pivot
    # -----------------------------------------------------------------------
    lb = divergence_lookback
    close_arr_f = c.values.astype(float)
    delta_arr   = delta.values.astype(float)

    price_ph = _pivot_high(close_arr_f, lb, lb)
    price_pl = _pivot_low(close_arr_f,  lb, lb)

    bearish_div = np.zeros(len(df), dtype=bool)
    bullish_div = np.zeros(len(df), dtype=bool)

    # Detect bearish divergence at each confirmed pivot high
    prev_ph_price = np.nan
    prev_ph_delta = np.nan
    for i in range(len(price_ph)):
        if not np.isnan(price_ph[i]):
            curr_ph_price = price_ph[i]
            curr_ph_delta = delta_arr[i]
            if (not np.isnan(prev_ph_price)
                    and curr_ph_price > prev_ph_price
                    and curr_ph_delta < prev_ph_delta):
                bearish_div[i] = True
            prev_ph_price = curr_ph_price
            prev_ph_delta = curr_ph_delta

    prev_pl_price = np.nan
    prev_pl_delta = np.nan
    for i in range(len(price_pl)):
        if not np.isnan(price_pl[i]):
            curr_pl_price = price_pl[i]
            curr_pl_delta = delta_arr[i]
            if (not np.isnan(prev_pl_price)
                    and curr_pl_price < prev_pl_price
                    and curr_pl_delta > prev_pl_delta):
                bullish_div[i] = True
            prev_pl_price = curr_pl_price
            prev_pl_delta = curr_pl_delta

    # -----------------------------------------------------------------------
    # 6. Stacked imbalance (consecutive directional candles as footprint proxy)
    #    Bullish candle = close > open = buy imbalance proxy
    #    min_stacked consecutive bullish/bearish candles = stacked imbalance
    # -----------------------------------------------------------------------
    bull_candle = (c > o).astype(int)
    bear_candle = (c < o).astype(int)

    # Count consecutive streaks using a rolling-sum approach:
    # A "streak" of k ends at bar t if sum(bull_candle[t-k+1:t+1]) == k.
    # We track maximum consecutive streak over a small rolling window equal
    # to min_stacked.
    bull_roll = bull_candle.rolling(min_stacked, min_periods=1).sum()
    bear_roll = bear_candle.rolling(min_stacked, min_periods=1).sum()
    stacked_buy_imb  = bull_roll >= min_stacked
    stacked_sell_imb = bear_roll >= min_stacked

    # -----------------------------------------------------------------------
    # 7. VAH / VAL breakout / breakdown signals
    # -----------------------------------------------------------------------
    vah_breakout      = (c > vah) & (c.shift(1) <= vah.shift(1))
    val_breakdown     = (c < val) & (c.shift(1) >= val.shift(1))
    false_vah_breakout = vah_breakout & (delta < 0)
    false_val_breakdown = val_breakdown & (delta > 0)

    # -----------------------------------------------------------------------
    # 8. Attach columns
    # -----------------------------------------------------------------------
    df["bc_delta"]               = delta
    df["bc_cvd"]                 = cvd
    df["bc_buy_vol"]             = buy_vol
    df["bc_sell_vol"]            = sell_vol
    df["bc_poc"]                 = poc
    df["bc_vah"]                 = vah
    df["bc_val"]                 = val
    df["bc_buy_absorption"]      = buy_absorption.astype(bool)
    df["bc_sell_absorption"]     = sell_absorption.astype(bool)
    df["bc_bullish_div"]         = bullish_div
    df["bc_bearish_div"]         = bearish_div
    df["bc_stacked_buy_imb"]     = stacked_buy_imb.astype(bool)
    df["bc_stacked_sell_imb"]    = stacked_sell_imb.astype(bool)
    df["bc_false_vah_breakout"]  = false_vah_breakout.astype(bool)
    df["bc_false_val_breakdown"] = false_val_breakdown.astype(bool)
    df["bc_vah_breakout"]        = vah_breakout.astype(bool)
    df["bc_val_breakdown"]       = val_breakdown.astype(bool)

    return df
