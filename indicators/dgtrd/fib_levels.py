"""
Python translation of:
  Fibonacci Extension / Retracement / Pivot Points by DGT
  https://www.tradingview.com/script/FWYQ4vTk-Fibonacci-Extension-Retracement-Pivot-Points-by-DGT/

Produces per-bar Fibonacci price levels (retracements and extensions from
the last ZigZag swing pair), HTF Pivot Points, Fibonacci Time Zones, and
volatility/volume add-on signals.

Output columns added to df
--------------------------
ZigZag swings (confirmed, no lookahead)
  fl_zz_is_high      bool — confirmed swing high at this bar
  fl_zz_is_low       bool — confirmed swing low  at this bar
  fl_zz_price        price of the confirmed swing (NaN otherwise)

Fibonacci price levels (forward-filled after each new swing; NaN until 2 swings confirmed)
  fl_ret_{level}     Retracement price for level (e.g. fl_ret_0618)
  fl_ext_{level}     Extension  price for level  (e.g. fl_ext_1618)
  Levels present: 0, 0236, 0382, 050, 0618, 0786, 1, 1618, 2618 (retr)
                  0, 0382, 0618, 1, 1382, 1618, 2618              (ext)

HTF Pivot Points (forward-filled from start of each new HTF period)
  fl_pp_pp           Pivot Point (H+L+C)/3
  fl_pp_r1 … fl_pp_r3  Resistance levels
  fl_pp_s1 … fl_pp_s3  Support levels
  Formula: PP ± (H-L) × {0.382, 0.618, 1.0}

Fibonacci Time Zones (discrete bar-hit booleans)
  fl_tz_{n}          bool — Fibonacci time zone hit for n in 2,3,5,8,13,21,34,55,89

Volatility / Volume add-ons
  fl_high_vol        bool — (high-low) > ATR(13) × 2.718
  fl_vol_spike       bool — volume > SMA(vol, 89) × 4.669

Pine-equivalent defaults
------------------------
  deviation_mult   3.0   (Pine: Deviation × ATR%)
  depth            11    (Pine: Depth; half = depth // 2)
  htf              'D'   (pandas resample offset for HTF Pivot Points)
  atr_mult         2.718 (Pine: High Volatility multiplier)
  atr_length       13    (Pine: ATR Length for high-vol detection)
  vol_spike_thresh 4.669 (Pine: Volume Spike Threshold)
  vol_sma_length   89    (Pine: Volume MA Length)

Usage
-----
  from indicators.dgtrd.fib_levels import fib_levels

  df = fib_levels(df, htf='D')   # requires DatetimeIndex for HTF PPs
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RET_LEVELS  = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, -0.618]
_EXT_LEVELS  = [0.0, 0.382, 0.618, 1.0, 1.382, 1.618, 2.618]
_TZ_FIBS     = [2, 3, 5, 8, 13, 21, 34, 55, 89]


def _level_key(v: float) -> str:
    """Convert float level to a safe column suffix: 0.618 → '0618'."""
    s = str(abs(v)).replace(".", "")
    return ("neg" + s) if v < 0 else s


# ---------------------------------------------------------------------------
# Helpers (shared with fib_time.py but kept local)
# ---------------------------------------------------------------------------

def _sma(s: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(s), np.nan)
    for i in range(n - 1, len(s)):
        out[i] = np.mean(s[i - n + 1 : i + 1])
    return out


def _calc_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
              n: int = 10) -> np.ndarray:
    """Wilder's ATR matching Pine ta.atr(n)."""
    tr = np.maximum(high - low,
         np.maximum(np.abs(high - np.roll(close, 1)),
                    np.abs(low  - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    alpha = 1.0 / n
    atr   = np.full(len(close), np.nan)
    atr[n - 1] = np.mean(tr[:n])
    for i in range(n, len(close)):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
    return atr


def _find_local_pivot(src: np.ndarray, idx: int, half: int,
                      is_high: bool) -> tuple[int, float] | tuple[None, None]:
    """
    Pine's pivots() function from the Fibonacci_Extension script.
    Non-strict: ties on either side disqualify.
    """
    n = len(src)
    if np.isnan(src[idx]):
        return None, None
    c  = src[idx]
    lo = max(0, idx - half)
    hi = min(n - 1, idx + half)
    if hi - lo < 2 * half:
        return None, None
    for j in range(lo, hi + 1):
        if j == idx or np.isnan(src[j]):
            continue
        if is_high and src[j] > c:
            return None, None
        if not is_high and src[j] < c:
            return None, None
    return idx, c


def _build_zigzag(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    half: int,
    deviation_mult: float,
) -> list[tuple[int, float, bool]]:
    """
    ATR-deviation ZigZag — same logic as fib_time.py.
    Returns list of confirmed (bar_index, price, is_high).
    """
    n      = len(close)
    swings: list[tuple[int, float, bool]] = []
    i_last    = 0
    p_last    = close[0] if not np.isnan(close[0]) else 0.0
    is_h_last = False

    for i in range(half, n - half):
        ih, ph = _find_local_pivot(high, i, half, True)
        il, pl = _find_local_pivot(low,  i, half, False)

        for is_high, idx, price in [(True, ih, ph), (False, il, pl)]:
            if idx is None:
                continue
            atr_v = atr[idx] if not np.isnan(atr[idx]) else 0.0
            dev   = 100.0 * (price - p_last) / max(abs(price), 1e-10)
            thresh = atr_v / max(abs(close[idx]), 1e-10) * 100.0 * deviation_mult

            if is_high == is_h_last:
                if (is_high and price > p_last) or (not is_high and price < p_last):
                    if swings:
                        swings[-1] = (idx, price, is_high)
                    i_last, p_last = idx, price
            else:
                if abs(dev) > thresh:
                    swings.append((i_last, p_last, is_h_last))
                    i_last, p_last, is_h_last = idx, price, is_high

    return swings


# ---------------------------------------------------------------------------
# Fibonacci price level formulas (matching Pine f_processLevelX)
# ---------------------------------------------------------------------------

def _fib_retracement(p_mid: float, p_end: float, level: float) -> float:
    """
    Retracement price at `level` within the B→C leg.

    Pine:
      pPivotDiff = abs(pMidPivot - pEndBase)
      price = pEndBase < pMidPivot ? pMidPivot - pPivotDiff*level
                                   : pMidPivot + pPivotDiff*level
    """
    diff = abs(p_mid - p_end)
    sign = -1.0 if p_end < p_mid else 1.0
    return p_mid + sign * diff * level


def _fib_extension(p_start: float, p_mid: float, p_end: float,
                   level: float) -> float:
    """
    Extension price at `level` projected from B using the A→B leg as offset.

    Pine:
      pPivotDiff = abs(pMidPivot - pEndBase)  # B→C height
      offset     = abs(pMidPivot - pStartBase) # A→B height
      price = pEndBase < pMidPivot
            ? pMidPivot - pPivotDiff + offset * level
            : pMidPivot + pPivotDiff - offset * level
    """
    diff_bc = abs(p_mid - p_end)
    offset  = abs(p_mid - p_start)
    if p_end < p_mid:
        return p_mid - diff_bc + offset * level
    else:
        return p_mid + diff_bc - offset * level


# ---------------------------------------------------------------------------
# HTF Pivot Points
# ---------------------------------------------------------------------------

def _compute_htf_pivots(
    df: pd.DataFrame,
    htf: str,
) -> pd.DataFrame:
    """
    Compute HTF Pivot Points using the PREVIOUS period's H, L, C.

    Requires df to have a DatetimeIndex.  htf is a pandas resample offset
    ('D' = daily, 'W' = weekly, 'ME' = month-end, etc.).

    Returns a DataFrame with columns pp_pp, pp_r1..r3, pp_s1..s3,
    aligned to the original index (forward-filled within each period).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame(index=df.index)

    htf_ohlc = df.resample(htf).agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    )

    prev_h = htf_ohlc["High"].shift(1)
    prev_l = htf_ohlc["Low"].shift(1)
    prev_c = htf_ohlc["Close"].shift(1)

    pp     = (prev_h + prev_l + prev_c) / 3.0
    rng    = prev_h - prev_l

    htf_pp = pd.DataFrame({
        "fl_pp_pp": pp,
        "fl_pp_r1": pp + rng * 0.382,
        "fl_pp_r2": pp + rng * 0.618,
        "fl_pp_r3": pp + rng * 1.000,
        "fl_pp_s1": pp - rng * 0.382,
        "fl_pp_s2": pp - rng * 0.618,
        "fl_pp_s3": pp - rng * 1.000,
    }, index=htf_ohlc.index)

    # Reindex to original index and forward-fill within each HTF period
    htf_pp = htf_pp.reindex(df.index, method="ffill")
    return htf_pp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fib_levels(
    df: pd.DataFrame,
    deviation_mult:   float = 3.0,
    depth:            int   = 11,
    htf:              str   = "D",
    atr_mult:         float = 2.718,
    atr_length:       int   = 13,
    vol_spike_thresh: float = 4.669,
    vol_sma_length:   int   = 89,
) -> pd.DataFrame:
    """
    Compute ZigZag-anchored Fibonacci levels, HTF Pivot Points, Fib Time
    Zones, and volatility/volume add-ons for every bar in *df*.

    Parameters
    ----------
    df                DataFrame with OHLCV columns (DatetimeIndex recommended).
    deviation_mult    ATR deviation multiplier for ZigZag (Pine default 3.0).
    depth             ZigZag depth; half = depth//2 (Pine default 11).
    htf               Pandas resample offset for HTF Pivot Points (default 'D').
    atr_mult          High-vol ATR multiplier (Pine: 2.718).
    atr_length        ATR period for high-vol detection (Pine: 13).
    vol_spike_thresh  Volume spike threshold × SMA (Pine: 4.669).
    vol_sma_length    Volume SMA period for spike detection (Pine: 89).

    Returns
    -------
    df copy with all columns described in module docstring.
    """
    high  = df["High"].to_numpy(dtype=float)
    low   = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    vol   = np.nan_to_num(df["Volume"].to_numpy(dtype=float), nan=0.0)
    n     = len(df)
    half  = max(1, depth // 2)

    atr10 = _calc_atr(high, low, close, n=10)    # for ZigZag deviation
    atr13 = _calc_atr(high, low, close, n=atr_length)   # for high-vol

    # --- ZigZag swings -------------------------------------------------------
    swings = _build_zigzag(high, low, close, atr10, half, deviation_mult)

    zz_is_high = np.zeros(n, dtype=bool)
    zz_is_low  = np.zeros(n, dtype=bool)
    zz_price   = np.full(n, np.nan)

    for bar, price, is_high in swings:
        if bar < n:
            (zz_is_high if is_high else zz_is_low)[bar] = True
            zz_price[bar] = price

    # --- Fibonacci price levels (forward-filled) ----------------------------
    ret_arrays: dict[str, np.ndarray] = {_level_key(lv): np.full(n, np.nan)
                                          for lv in _RET_LEVELS}
    ext_arrays: dict[str, np.ndarray] = {_level_key(lv): np.full(n, np.nan)
                                          for lv in _EXT_LEVELS}

    # Current active fib levels (updated at each new confirmed swing ≥ 2)
    cur_ret: dict[str, float] = {}
    cur_ext: dict[str, float] = {}

    swing_bar_to_k = {s[0]: k for k, s in enumerate(swings)}

    for i in range(n):
        k = swing_bar_to_k.get(i)
        if k is not None and k >= 1:
            i_b, p_b, _ = swings[k]      # most recent (B = "end of last leg")
            i_a, p_a, _ = swings[k - 1]  # one before  (A = "start of last leg" = pMidPivot)

            # Update retracements (B→A leg: pMidPivot=pLastPivot, pEndBase=pPrevPivot in Pine)
            # Pine: iMidPivot = line.get_x1(lineLast) = start of current segment = pMidPivot
            #       iEndBase  = line.get_x2(lineLast) = end of current segment = pEndBase
            # So for retracements, the "mid" is the start of the last leg (p_a) and
            # "end" is the tip of the last leg (p_b) when there's 1 prior leg.
            # With ≥2 prior legs we have 3 points; the "start" for ext is 2 legs back.
            p_mid  = p_a    # B (start of last segment / previous confirmed pivot)
            p_end  = p_b    # C (most recent confirmed pivot)

            for lv in _RET_LEVELS:
                cur_ret[_level_key(lv)] = _fib_retracement(p_mid, p_end, lv)

            if k >= 2:
                i_s, p_s, _ = swings[k - 2]  # A (two swings back)
                for lv in _EXT_LEVELS:
                    cur_ext[_level_key(lv)] = _fib_extension(p_s, p_mid, p_end, lv)

        # Forward-fill
        for lv in _RET_LEVELS:
            key = _level_key(lv)
            if key in cur_ret:
                ret_arrays[key][i] = cur_ret[key]

        for lv in _EXT_LEVELS:
            key = _level_key(lv)
            if key in cur_ext:
                ext_arrays[key][i] = cur_ext[key]

    # --- Fibonacci Time Zones ------------------------------------------------
    tz_hits: dict[int, np.ndarray] = {f: np.zeros(n, dtype=bool) for f in _TZ_FIBS}

    for k in range(1, len(swings)):
        i_b = swings[k - 1][0]
        i_c = swings[k][0]
        ref_bc = round(i_c - i_b)
        for fib in _TZ_FIBS:
            tgt = round(i_b + ref_bc * fib)
            if 0 <= tgt < n:
                tz_hits[fib][tgt] = True

    # --- HTF Pivot Points ----------------------------------------------------
    htf_pp_df = _compute_htf_pivots(df, htf)

    # --- Volatility & Volume add-ons ----------------------------------------
    bar_range = high - low
    high_vol  = (bar_range > atr13 * atr_mult) & ~np.isnan(atr13)

    vol_sma    = _sma(vol, vol_sma_length)
    vol_spike  = vol > vol_sma * vol_spike_thresh

    # --- Assemble output -----------------------------------------------------
    df = df.copy()

    df["fl_zz_is_high"] = zz_is_high
    df["fl_zz_is_low"]  = zz_is_low
    df["fl_zz_price"]   = zz_price

    for lv in _RET_LEVELS:
        df[f"fl_ret_{_level_key(lv)}"] = ret_arrays[_level_key(lv)]

    for lv in _EXT_LEVELS:
        df[f"fl_ext_{_level_key(lv)}"] = ext_arrays[_level_key(lv)]

    for fib in _TZ_FIBS:
        df[f"fl_tz_{fib}"] = tz_hits[fib]

    df["fl_high_vol"]  = high_vol
    df["fl_vol_spike"] = vol_spike

    # HTF Pivot Points (may be empty if no DatetimeIndex)
    for col in ["fl_pp_pp", "fl_pp_r1", "fl_pp_r2", "fl_pp_r3",
                "fl_pp_s1", "fl_pp_s2", "fl_pp_s3"]:
        df[col] = htf_pp_df[col] if col in htf_pp_df.columns else np.nan

    return df
