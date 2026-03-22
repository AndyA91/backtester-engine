"""
Python translation of:
  Auto Fib Time Zones and Trend-Based Fib Time by DGT
  https://www.tradingview.com/script/l2SQEvfQ-Auto-Fib-Time-Zones-and-Trend-Based-Fib-Time-by-DGT/

Detects ZigZag swings (ATR-deviation-based) and projects two time-based
Fibonacci tools forward from each confirmed swing pair:

  1. Trend-Based Fib Time — projects at bar offsets of:
       iEndBase + round(referance × level)
     where referance = round(iMidPivot - iStartBase) and levels are the
     configurable Fib ratios (0, 0.382, 0.618, 1.0, 1.382, 1.618, 2, 2.382, 2.618, 3).

  2. Fib Time Zones — projects at:
       iMidPivot2 + round(referance2 × fib_number)
     where referance2 = round(iEndBase2 - iMidPivot2) and fib_numbers are
     the Fibonacci sequence (2, 3, 5, 8, 13, 21, 34, 55, 89).

Output columns added to df
--------------------------
  ft_zz_is_high      bool — confirmed ZigZag swing high at this bar
  ft_zz_is_low       bool — confirmed ZigZag swing low  at this bar
  ft_zz_price        price of the confirmed swing (NaN otherwise)

  ft_trend_ref_span  bar count of the A→B reference segment (Trend-Based Fib)
  ft_trend_end_bar   absolute bar index of point B (iEndBase)
  ft_trend_{level}   bool — current bar is at the Trend-Based Fib time level
                     (levels: 0, 0382, 0618, 1, 1382, 1618, 2, 2382, 2618, 3)

  ft_tz_ref_span     bar count of the B→C reference segment (Fib Time Zones)
  ft_tz_mid_bar      absolute bar index of point B (iMidPivot2)
  ft_tz_{n}          bool — current bar is at Fibonacci time zone n
                     (n: 2, 3, 5, 8, 13, 21, 34, 55, 89)

Pine-equivalent defaults
------------------------
  deviation_mult   3.0   (Pine: factor × ta.atr(10)/close*100)
  depth            11    (Pine: i_depth; half-length = depth//2)

Usage
-----
  from indicators.dgtrd.fib_time import fib_time_zones

  df = fib_time_zones(df, deviation_mult=3.0, depth=11)
  # ft_tz_8 == True marks bars at the Fibonacci x8 time zone
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TREND_LEVELS = [0.0, 0.382, 0.5, 0.618, 1.0, 1.382, 1.618, 2.0, 2.382, 2.618, 3.0]
_TZ_FIBS      = [2, 3, 5, 8, 13, 21, 34, 55, 89]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _calc_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
              n: int = 10) -> np.ndarray:
    """Wilder's ATR matching Pine ta.atr(n)."""
    tr = np.maximum(high - low,
         np.maximum(np.abs(high - np.roll(close, 1)),
                    np.abs(low  - np.roll(close, 1))))
    tr[0] = high[0] - low[0]   # no previous close for first bar

    alpha = 1.0 / n
    atr = np.full(len(close), np.nan)

    # seed = SMA of first n TR values
    atr[n - 1] = np.mean(tr[:n])
    for i in range(n, len(close)):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
    return atr


def _find_local_pivot(src: np.ndarray, idx: int, half: int,
                      is_high: bool) -> tuple[int, float] | tuple[None, None]:
    """
    Pine's pivots() function: checks window [idx-half, idx+half] centered
    on idx.  Any value strictly more extreme on EITHER side disqualifies.
    Returns (bar_index, value) or (None, None).
    """
    n = len(src)
    c = src[idx] if not np.isnan(src[idx]) else None
    if c is None:
        return None, None

    lo = max(0, idx - half)
    hi = min(n - 1, idx + half)
    if hi - lo < 2 * half:      # not enough bars around the candidate
        return None, None

    window = src[lo : hi + 1]
    if np.any(np.isnan(window)):
        return None, None

    if is_high:
        ok = all(src[j] <= c for j in range(lo, hi + 1) if j != idx)
    else:
        ok = all(src[j] >= c for j in range(lo, hi + 1) if j != idx)

    return (idx, c) if ok else (None, None)


# ---------------------------------------------------------------------------
# ZigZag state machine
# ---------------------------------------------------------------------------

def _build_zigzag(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    half: int,
    deviation_mult: float,
) -> list[tuple[int, float, bool]]:
    """
    ATR-deviation ZigZag matching Pine's pivotFound() logic.

    Returns list of (bar_index, price, is_high) for each CONFIRMED segment
    endpoint, in chronological order.  A segment endpoint is only appended
    when a new OPPOSING pivot is confirmed (not when the current tip updates).
    """
    n       = len(close)
    swings: list[tuple[int, float, bool]] = []   # confirmed endpoints

    # Running state
    i_last    = 0
    p_last    = close[0] if not np.isnan(close[0]) else 0.0
    is_h_last = False

    for i in range(half, n - half):
        ih, ph = _find_local_pivot(high, i, half, True)
        il, pl = _find_local_pivot(low,  i, half, False)

        for is_high, idx, price in [(True, ih, ph), (False, il, pl)]:
            if idx is None:
                continue

            # deviation threshold at this bar
            atr_val = atr[idx] if not np.isnan(atr[idx]) else 0.0
            dev = 100.0 * (price - p_last) / max(abs(price), 1e-10)

            if is_high == is_h_last:
                # same direction → update tip if more extreme
                if (is_high and price > p_last) or (not is_high and price < p_last):
                    if swings:
                        swings[-1] = (idx, price, is_high)   # update tip in-place
                    i_last    = idx
                    p_last    = price
            else:
                # opposite direction → confirm only if deviation > threshold
                thresh = atr_val / max(abs(close[idx]), 1e-10) * 100.0 * deviation_mult
                if abs(dev) > thresh:
                    swings.append((i_last, p_last, is_h_last))   # confirm previous tip
                    i_last    = idx
                    p_last    = price
                    is_h_last = is_high

    return swings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fib_time_zones(
    df: pd.DataFrame,
    deviation_mult: float = 3.0,
    depth:          int   = 11,
) -> pd.DataFrame:
    """
    Compute ZigZag-anchored Fibonacci time projections for every bar in *df*.

    Parameters
    ----------
    df              DataFrame with columns High, Low, Close (DatetimeIndex optional).
    deviation_mult  ATR deviation multiplier (Pine: Deviation input, default 3.0).
    depth           ZigZag depth; half-length = depth // 2 (Pine default 11).

    Returns
    -------
    df copy with columns described in module docstring.
    """
    high  = df["High"].to_numpy(dtype=float)
    low   = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    n     = len(df)
    half  = max(1, depth // 2)

    atr = _calc_atr(high, low, close, n=10)

    # Build confirmed ZigZag swings
    swings = _build_zigzag(high, low, close, atr, half, deviation_mult)

    # --- Output arrays -------------------------------------------------------
    zz_is_high = np.zeros(n, dtype=bool)
    zz_is_low  = np.zeros(n, dtype=bool)
    zz_price   = np.full(n, np.nan)

    # Trend-Based Fib Time: uses the 3 most recent confirmed swings (A, B, C)
    # A = swings[-3], B = swings[-2], C = swings[-1]
    # reference = round(B.bar - A.bar)  → project from C.bar + ref * level
    trend_ref_span  = np.full(n, np.nan)
    trend_end_bar   = np.full(n, np.nan)

    # Fib Time Zones: uses the last 2 confirmed swings (B, C)
    # reference = round(C.bar - B.bar)  → project from B.bar + ref * fib
    tz_ref_span = np.full(n, np.nan)
    tz_mid_bar  = np.full(n, np.nan)

    # Level hit flags — allocated per level/fib
    trend_hits: dict[str, np.ndarray] = {}
    for lv in _TREND_LEVELS:
        key = str(lv).replace(".", "").replace("-", "neg")
        trend_hits[key] = np.zeros(n, dtype=bool)

    tz_hits: dict[int, np.ndarray] = {f: np.zeros(n, dtype=bool) for f in _TZ_FIBS}

    # Mark confirmed swing points
    for bar, price, is_high in swings:
        if bar < n:
            if is_high:
                zz_is_high[bar] = True
            else:
                zz_is_low[bar] = True
            zz_price[bar] = price

    # Compute Fib projections from each confirmed swing triplet
    # Each time a 3rd+ swing is confirmed, emit time projections forward
    # We forward-fill the state from the detection bar onward.

    # We'll store the projection state and fill per-bar
    cur_trend_ref = np.nan
    cur_trend_end = np.nan
    cur_tz_ref    = np.nan
    cur_tz_mid    = np.nan

    # Track which bars are Fib time level hits (these are discrete bar events)
    for k in range(2, len(swings)):
        # A, B, C = swings[k-2], swings[k-1], swings[k]
        i_a, p_a, _ = swings[k - 2]
        i_b, p_b, _ = swings[k - 1]
        i_c, p_c, _ = swings[k]

        # Trend-Based Fib Time
        ref_ab = round(i_b - i_a)       # bar count of A→B leg
        for lv in _TREND_LEVELS:
            target_bar = round(i_c + ref_ab * lv)
            key = str(lv).replace(".", "").replace("-", "neg")
            if 0 <= target_bar < n:
                trend_hits[key][target_bar] = True

        # Fib Time Zones  (uses B and C as reference pair)
        ref_bc = round(i_c - i_b)       # bar count of B→C leg
        for fib in _TZ_FIBS:
            target_bar = round(i_b + ref_bc * fib)
            if 0 <= target_bar < n:
                tz_hits[fib][target_bar] = True

    # Fill the scalar state columns (forward-fill from each new swing confirmation)
    last_trend_ref = np.nan
    last_trend_end = np.nan
    last_tz_ref    = np.nan
    last_tz_mid    = np.nan

    swing_bars = {s[0]: (k, s) for k, s in enumerate(swings)}

    for i in range(n):
        if i in swing_bars:
            k, (i_c, _, _) = swing_bars[i]
            if k >= 2:
                i_a = swings[k - 2][0]
                i_b = swings[k - 1][0]
                last_trend_ref = float(round(i_b - i_a))
                last_trend_end = float(i_c)
                last_tz_ref    = float(round(i_c - i_b))
                last_tz_mid    = float(i_b)

        trend_ref_span[i] = last_trend_ref
        trend_end_bar[i]  = last_trend_end
        tz_ref_span[i]    = last_tz_ref
        tz_mid_bar[i]     = last_tz_mid

    # --- Attach to df --------------------------------------------------------
    df = df.copy()
    df["ft_zz_is_high"]      = zz_is_high
    df["ft_zz_is_low"]       = zz_is_low
    df["ft_zz_price"]        = zz_price
    df["ft_trend_ref_span"]  = trend_ref_span
    df["ft_trend_end_bar"]   = trend_end_bar
    df["ft_tz_ref_span"]     = tz_ref_span
    df["ft_tz_mid_bar"]      = tz_mid_bar

    for lv in _TREND_LEVELS:
        key  = str(lv).replace(".", "").replace("-", "neg")
        col  = f"ft_trend_{key}"
        df[col] = trend_hits[key]

    for fib in _TZ_FIBS:
        df[f"ft_tz_{fib}"] = tz_hits[fib]

    return df
