"""
Python translation of:
  [blackcat] L3 Volume Profile Pro by blackcat1402
  https://www.tradingview.com/script/T7pnn9NL-blackcat-L3-Volume-Profile-Pro/

The original relies on TradingView's request.footprint() API (Premium plan).
The Volume Profile (POC, VAH, VAL) is reconstructed from OHLCV using a
rolling-window histogram approach:

  For each bar t, take the last `rolling_vp_bars` close prices and their
  corresponding volumes.  Bin those prices into `n_bins` equal-width
  buckets weighted by volume.

  POC: midpoint of the bucket with the highest accumulated volume.
  VAH / VAL: greedy expansion from POC outward until va_pct% of total
             volume is captured.  VAH = upper edge of the highest included
             bucket, VAL = lower edge of the lowest included bucket.

Per-bar delta and CVD are approximated identically to the Footprint module:
  delta ≈ volume × (2×close − high − low) / (high − low)

POC Migration tracking is a pure bar-over-bar POC comparison — no
approximation needed.

Signals that use footprint rows (VAH/VAL interaction, POC touch, divergence,
accumulation/distribution) are reproduced faithfully from the original logic
using the approximated POC/VAH/VAL values.

Vectorisation notes
-------------------
- Delta / CVD: fully vectorised pandas cumsum.
- Rolling Volume Profile: one numpy.histogram call per bar inside a Python
  loop — O(n × lookback).  The greedy VA expansion is O(n_bins) per bar.
- POC migration: vectorised pandas shift comparison.
- Divergence: pivot-based rolling-max/min with right-offset confirmation
  (same helper as Footprint Fusion Pro).
- All moving averages and rolling statistics use pandas built-ins.

Output columns (bc_ prefix)
----------------------------
  bc_poc                — point-of-control price
  bc_vah                — value area high
  bc_val                — value area low
  bc_delta              — per-bar directional delta (approximated)
  bc_cvd                — cumulative volume delta
  bc_buy_sell_ratio     — buy volume fraction [0, 1]
  bc_poc_migration      — POC migration direction: 1=up, -1=down, 0=neutral
  bc_poc_bounce         — POC bounce signal (price near POC from above, +delta)
  bc_poc_rejection      — POC rejection signal (price near POC from below, -delta)
  bc_vah_breakout       — price closes above VAH with positive delta
  bc_vah_rejection      — price at or above VAH with bearish candle and neg delta
  bc_val_breakdown      — price closes below VAL with negative delta
  bc_val_support        — price at or below VAL with bullish candle and pos delta
  bc_bullish_div        — bullish delta divergence (price LL, delta HL)
  bc_bearish_div        — bearish delta divergence (price HH, delta LH)
  bc_accumulation       — POC migrating up + volume above average
  bc_distribution       — POC migrating down + volume above average

Usage
-----
  from indicators.blackcat1402.blackcat_l3_volume_profile_pro import (
      calc_bc_l3_volume_profile_pro
  )
  df = calc_bc_l3_volume_profile_pro(df)
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _greedy_value_area(
    vol_hist: np.ndarray,
    poc_idx: int,
    va_pct: float,
) -> tuple:
    """Expand from poc_idx outward until va_pct% of volume is captured."""
    total  = vol_hist.sum()
    target = total * va_pct / 100.0
    lo, hi = poc_idx, poc_idx
    accum  = vol_hist[poc_idx]
    nb     = len(vol_hist)

    while accum < target:
        can_up   = hi + 1 < nb
        can_down = lo - 1 >= 0
        if not can_up and not can_down:
            break
        v_up   = vol_hist[hi + 1] if can_up   else -1.0
        v_down = vol_hist[lo - 1] if can_down else -1.0
        if v_up >= v_down:
            hi    += 1
            accum += vol_hist[hi]
        else:
            lo    -= 1
            accum += vol_hist[lo]

    return lo, hi


def _rolling_volume_profile(
    close_arr: np.ndarray,
    volume_arr: np.ndarray,
    lookback: int,
    n_bins: int,
    va_pct: float,
) -> tuple:
    """
    Compute POC, VAH, VAL for every bar using a rolling lookback window.
    Returns three numpy arrays of length n.
    """
    n   = len(close_arr)
    poc = np.full(n, np.nan)
    vah = np.full(n, np.nan)
    val = np.full(n, np.nan)

    for i in range(lookback - 1, n):
        win_c = close_arr[i - lookback + 1: i + 1]
        win_v = volume_arr[i - lookback + 1: i + 1]

        p_min = win_c.min()
        p_max = win_c.max()
        if p_min == p_max or win_v.sum() == 0:
            poc[i] = vah[i] = val[i] = p_min
            continue

        counts, edges = np.histogram(
            win_c, bins=n_bins, range=(p_min, p_max), weights=win_v
        )
        poc_bin = int(np.argmax(counts))
        poc[i]  = (edges[poc_bin] + edges[poc_bin + 1]) / 2.0

        lo, hi = _greedy_value_area(counts, poc_bin, va_pct)
        val[i] = edges[lo]
        vah[i] = edges[hi + 1]

    return poc, vah, val


def _pivot_high(series: np.ndarray, left: int, right: int) -> np.ndarray:
    n = len(series)
    result = np.full(n, np.nan)
    for i in range(left, n - right):
        v = series[i]
        if np.isnan(v):
            continue
        if (all(series[i - j] <= v for j in range(1, left + 1)) and
                all(series[i + j] <  v for j in range(1, right + 1))):
            result[i + right] = v
    return result


def _pivot_low(series: np.ndarray, left: int, right: int) -> np.ndarray:
    n = len(series)
    result = np.full(n, np.nan)
    for i in range(left, n - right):
        v = series[i]
        if np.isnan(v):
            continue
        if (all(series[i - j] >= v for j in range(1, left + 1)) and
                all(series[i + j] >  v for j in range(1, right + 1))):
            result[i + right] = v
    return result


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def calc_bc_l3_volume_profile_pro(
    df: pd.DataFrame,
    va_pct: float = 70.0,
    rolling_vp_bars: int = 50,
    n_bins: int = 20,
    poc_touch_atr_mult: float = 0.1,
    divergence_lookback: int = 5,
) -> pd.DataFrame:
    """
    Python translation of [blackcat] L3 Volume Profile Pro by blackcat1402.
    - Input:  df with columns open, high, low, close, volume
    - Output: df with new bc_ prefixed columns appended

    Parameters
    ----------
    va_pct              : value area percentage (default 70)
    rolling_vp_bars     : rolling lookback for volume profile
    n_bins              : number of price buckets for histogram
    poc_touch_atr_mult  : ATR multiplier for POC-touch distance threshold
    divergence_lookback : left = right bars for pivot detection
    """
    df = df.copy()
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    # -----------------------------------------------------------------------
    # 1. Per-bar delta approximation and CVD
    # -----------------------------------------------------------------------
    candle_range = h - l
    safe_range   = np.where(candle_range > 0, candle_range, 1.0)
    delta        = v * (2.0 * c - h - l) / safe_range
    cvd          = delta.cumsum()

    buy_vol  = v * (c - l) / safe_range
    sell_vol = v * (h - c) / safe_range
    buy_sell_ratio = buy_vol / v.replace(0, np.nan)

    # -----------------------------------------------------------------------
    # 2. Rolling Volume Profile
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
    # 3. ATR
    # -----------------------------------------------------------------------
    atr = (h - l).rolling(14).mean()

    # -----------------------------------------------------------------------
    # 4. POC migration tracking
    #    Upward migration:   POC[t] > POC[t-1] > POC[t-2]   → 1
    #    Downward migration: POC[t] < POC[t-1] < POC[t-2]   → -1
    #    Otherwise: 0
    # -----------------------------------------------------------------------
    poc_prev1 = poc.shift(1)
    poc_prev2 = poc.shift(2)
    up_mig   = (poc > poc_prev1) & (poc_prev1 > poc_prev2)
    down_mig = (poc < poc_prev1) & (poc_prev1 < poc_prev2)
    poc_migration = pd.Series(0, index=df.index)
    poc_migration = poc_migration.where(~up_mig,    1)
    poc_migration = poc_migration.where(~down_mig, -1)

    # -----------------------------------------------------------------------
    # 5. Volume average for migration quality filter
    # -----------------------------------------------------------------------
    vol_sma = v.rolling(20).mean()

    # -----------------------------------------------------------------------
    # 6. Signal detection
    # -----------------------------------------------------------------------
    poc_distance = (c - poc).abs()
    poc_near     = poc_distance < atr * poc_touch_atr_mult

    # POC bounce: near POC from above, +delta
    poc_bounce    = poc_near & (c > poc) & (delta > 0)
    # POC rejection: near POC from below, -delta
    poc_rejection = poc_near & (c < poc) & (delta < 0)

    # VAH breakout: closes above VAH with +delta
    vah_cross_up   = (c > vah) & (c.shift(1) <= vah.shift(1))
    vah_breakout   = vah_cross_up & (delta > 0)

    # VAH rejection: high touches or exceeds VAH but close below, -delta
    vah_rejection  = (h >= vah) & (c < vah) & (delta < 0)

    # VAL breakdown: closes below VAL with -delta
    val_cross_down = (c < val) & (c.shift(1) >= val.shift(1))
    val_breakdown  = val_cross_down & (delta < 0)

    # VAL support: low touches or below VAL but close above, +delta
    val_support    = (l <= val) & (c > val) & (delta > 0)

    # Accumulation / Distribution (POC migration + above-avg volume)
    accumulation = (poc_migration == 1)  & (v > vol_sma)
    distribution = (poc_migration == -1) & (v > vol_sma)

    # -----------------------------------------------------------------------
    # 7. Delta divergence
    # -----------------------------------------------------------------------
    lb = divergence_lookback
    delta_arr = delta.values.astype(float)

    price_ph = _pivot_high(close_arr, lb, lb)
    price_pl = _pivot_low(close_arr,  lb, lb)

    bearish_div = np.zeros(len(df), dtype=bool)
    bullish_div = np.zeros(len(df), dtype=bool)

    prev_ph_p = prev_ph_d = np.nan
    for i in range(len(price_ph)):
        if not np.isnan(price_ph[i]):
            if (not np.isnan(prev_ph_p)
                    and price_ph[i] > prev_ph_p
                    and delta_arr[i] < prev_ph_d):
                bearish_div[i] = True
            prev_ph_p = price_ph[i]
            prev_ph_d = delta_arr[i]

    prev_pl_p = prev_pl_d = np.nan
    for i in range(len(price_pl)):
        if not np.isnan(price_pl[i]):
            if (not np.isnan(prev_pl_p)
                    and price_pl[i] < prev_pl_p
                    and delta_arr[i] > prev_pl_d):
                bullish_div[i] = True
            prev_pl_p = price_pl[i]
            prev_pl_d = delta_arr[i]

    # -----------------------------------------------------------------------
    # 8. Attach columns
    # -----------------------------------------------------------------------
    df["bc_poc"]              = poc
    df["bc_vah"]              = vah
    df["bc_val"]              = val
    df["bc_delta"]            = delta
    df["bc_cvd"]              = cvd
    df["bc_buy_sell_ratio"]   = buy_sell_ratio
    df["bc_poc_migration"]    = poc_migration.astype(int)
    df["bc_poc_bounce"]       = poc_bounce.astype(bool)
    df["bc_poc_rejection"]    = poc_rejection.astype(bool)
    df["bc_vah_breakout"]     = vah_breakout.astype(bool)
    df["bc_vah_rejection"]    = vah_rejection.astype(bool)
    df["bc_val_breakdown"]    = val_breakdown.astype(bool)
    df["bc_val_support"]      = val_support.astype(bool)
    df["bc_bullish_div"]      = pd.Series(bullish_div, index=df.index)
    df["bc_bearish_div"]      = pd.Series(bearish_div, index=df.index)
    df["bc_accumulation"]     = accumulation.astype(bool)
    df["bc_distribution"]     = distribution.astype(bool)

    return df
