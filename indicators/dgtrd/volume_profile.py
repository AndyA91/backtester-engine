"""
Python translation of:
  Volume Profile, Pivot Anchored by DGT
  https://www.tradingview.com/script/utCRHZeP-Volume-Profile-Pivot-Anchored-by-DGT/

Produces per-bar scalar levels (POC, VAH, VAL) suitable for backtesting.
Visualization-only constructs (boxes, lines, labels, barcolor) are omitted.

Output columns added to df
--------------------------
  vp_poc          Point of Control price
  vp_vah          Value Area High price
  vp_val          Value Area Low price
  vp_profile_high Highest high in the profile window
  vp_profile_low  Lowest  low  in the profile window
  vp_in_va        bool — close is between VAL and VAH (inclusive)
  vp_above_poc    bool — close is strictly above POC

All levels are NaN until the second pivot is confirmed, then forward-filled
until the next profile is computed — matching Pine's draw-and-hold behaviour.

Pine-equivalent defaults
------------------------
  pvt_length  = 20   (input pvtLength)
  num_bins    = 25   (input profileLevels)
  va_pct      = 0.68 (input isValueArea, 68%)

Usage
-----
  from indicators.dgtrd.volume_profile import volume_profile_pivot_anchored

  df = pd.read_csv(...)   # must have columns: High, Low, Close, Volume
  df = volume_profile_pivot_anchored(df, pvt_length=20, num_bins=25, va_pct=0.68)
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _calc_pivots(
    high: np.ndarray,
    low: np.ndarray,
    length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Equivalent to ta.pivothigh(length, length) and ta.pivotlow(length, length).

    Returns (pivot_high, pivot_low) arrays where non-NaN values mark confirmed
    pivot bars.  Strict uniqueness check (MEMORY.md §6) prevents overcounting
    when multiple bars share the same high/low value in the window.

    Detection timing:  pivot at index i is confirmed at index i+length (same
    as Pine — the indicator fires pvtLength bars after the pivot bar).
    """
    n = len(high)
    pivot_high = np.full(n, np.nan)
    pivot_low  = np.full(n, np.nan)

    for i in range(length, n - length):
        window_h = high[i - length : i + length + 1]
        window_l = low [i - length : i + length + 1]

        if high[i] == np.max(window_h) and np.sum(window_h == high[i]) == 1:
            pivot_high[i] = high[i]

        if low[i] == np.min(window_l) and np.sum(window_l == low[i]) == 1:
            pivot_low[i] = low[i]

    return pivot_high, pivot_low


def _build_profile(
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    start_bar: int,   # prev_pivot_bar (inclusive in both price range and bins)
    end_bar: int,     # curr_pivot_bar (inclusive in price range, EXCLUDED from bins)
    num_bins: int,
) -> tuple[np.ndarray, float, float]:
    """
    Build a volume profile for bars [start_bar, end_bar).

    Price range is sized using [start_bar, end_bar] inclusive so bin boundaries
    match Pine's f_getHighLow call (which adds the curr_pivot bar to high/low
    but not to volume — see analysis notes).

    Returns
    -------
    bins        : float array of length num_bins, volume per bin
    price_low   : lowest  low  in [start_bar, end_bar]  (inclusive)
    price_high  : highest high in [start_bar, end_bar]  (inclusive)
    """
    # Bin boundaries use the inclusive range (both pivot bars)
    price_high = np.max(high [start_bar : end_bar + 1])
    price_low  = np.min(low  [start_bar : end_bar + 1])

    if price_high <= price_low:
        return np.zeros(num_bins), price_low, price_high

    bin_size  = (price_high - price_low) / num_bins
    bin_lows  = price_low + np.arange(num_bins) * bin_size   # shape (num_bins,)
    bin_highs = bin_lows + bin_size

    bins = np.zeros(num_bins)

    # Volume binning covers [start_bar, end_bar) — curr_pivot bar excluded
    # (matches Pine bin loop: barIndexx = 1 to profileLength, which starts one
    #  bar *before* curr_pivot and ends at prev_pivot)
    for b in range(start_bar, end_bar):
        h = high  [b]
        l = low   [b]
        v = volume[b]
        if v == 0:
            continue

        bar_range = h - l
        frac = 1.0 if bar_range == 0.0 else bin_size / bar_range

        # Uniform-distribution overlap: bar touches bin if high>=bin_low AND low<bin_high
        mask = (h >= bin_lows) & (l < bin_highs)
        bins[mask] += v * frac

    return bins, price_low, price_high


def _calc_value_area(
    bins: np.ndarray,
    va_pct: float,
) -> tuple[int, int, int]:
    """
    Greedy Value Area expansion matching Pine's while loop (lines 238-262).

    Ties between above/below go to 'above' (volumeAbovePoc >= volumeBelowPoc).

    Returns
    -------
    poc_idx   : bin index of the Point of Control
    above_idx : highest bin index inside the Value Area
    below_idx : lowest  bin index inside the Value Area
    """
    poc_idx = int(np.argmax(bins))
    target  = bins.sum() * va_pct
    va_vol  = bins[poc_idx]
    above   = poc_idx
    below   = poc_idx
    n       = len(bins)

    while va_vol < target:
        if below == 0 and above == n - 1:
            break

        vol_above = bins[above + 1] if above < n - 1 else 0.0
        vol_below = bins[below - 1] if below > 0     else 0.0

        if vol_above == 0.0 and vol_below == 0.0:
            break

        if vol_above >= vol_below:          # ties → above (Pine behaviour)
            va_vol += vol_above
            above  += 1
        else:
            va_vol += vol_below
            below  -= 1

    return poc_idx, above, below


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def volume_profile_pivot_anchored(
    df: pd.DataFrame,
    pvt_length: int  = 20,
    num_bins:   int  = 25,
    va_pct:     float = 0.68,
) -> pd.DataFrame:
    """
    Compute pivot-anchored Volume Profile levels for every bar in *df*.

    Parameters
    ----------
    df          DataFrame with columns High, Low, Close, Volume (any index).
    pvt_length  Pivot left/right lookback length (Pine: pvtLength, default 20).
    num_bins    Number of price bins (Pine: profileLevels, default 25).
    va_pct      Value Area percentage as a fraction (Pine: isValueArea/100,
                default 0.68 = 68%).

    Returns
    -------
    df with five new columns: vp_poc, vp_vah, vp_val,
    vp_profile_high, vp_profile_low, vp_in_va, vp_above_poc.
    Levels are NaN before the second pivot is confirmed, then forward-filled.
    """
    high   = df["High"].to_numpy(dtype=float)
    low    = df["Low"].to_numpy(dtype=float)
    close  = df["Close"].to_numpy(dtype=float)
    volume = df["Volume"].to_numpy(dtype=float)
    n      = len(df)

    # Output arrays (NaN until first profile is ready)
    out_poc  = np.full(n, np.nan)
    out_vah  = np.full(n, np.nan)
    out_val  = np.full(n, np.nan)
    out_phigh = np.full(n, np.nan)
    out_plow  = np.full(n, np.nan)

    # --- Step 1: find all pivot bars ------------------------------------------
    pivot_high_vals, pivot_low_vals = _calc_pivots(high, low, pvt_length)

    # Collect (pivot_bar_index, type) sorted by pivot bar (ascending)
    # A pivot at bar i is *detected* at bar i+pvt_length in Pine.
    pivot_events: list[tuple[int, str]] = []
    for i in range(n):
        if not np.isnan(pivot_high_vals[i]):
            pivot_events.append((i, "H"))
        if not np.isnan(pivot_low_vals[i]):
            pivot_events.append((i, "L"))

    # Sort by pivot bar index (they are already in order but be explicit)
    pivot_events.sort(key=lambda x: x[0])

    # --- Step 2: for each consecutive pivot pair, compute one profile ---------
    # We need at least two pivot events.
    for k in range(1, len(pivot_events)):
        prev_pbar, _ = pivot_events[k - 1]
        curr_pbar, _ = pivot_events[k]

        # Detection bar: curr_pbar + pvt_length (where Pine fires 'proceed')
        detect_bar = curr_pbar + pvt_length
        if detect_bar >= n:
            break                   # not enough future bars to confirm

        if curr_pbar <= prev_pbar:  # degenerate window
            continue

        bins, price_low, price_high = _build_profile(
            high, low, volume,
            start_bar = prev_pbar,
            end_bar   = curr_pbar,
            num_bins  = num_bins,
        )

        if bins.sum() == 0 or price_high <= price_low:
            continue

        bin_size                  = (price_high - price_low) / num_bins
        poc_idx, above_idx, below_idx = _calc_value_area(bins, va_pct)

        poc_price  = price_low + (poc_idx   + 0.5) * bin_size
        vah_price  = price_low + (above_idx + 1.0) * bin_size
        val_price  = price_low + (below_idx + 0.0) * bin_size

        # Assign at detection bar; forward-fill is done after the loop
        out_poc  [detect_bar] = poc_price
        out_vah  [detect_bar] = vah_price
        out_val  [detect_bar] = val_price
        out_phigh[detect_bar] = price_high
        out_plow [detect_bar] = price_low

    # --- Step 3: forward-fill all level columns --------------------------------
    def _ffill(arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        last = np.nan
        for i in range(len(out)):
            if not np.isnan(out[i]):
                last = out[i]
            elif not np.isnan(last):
                out[i] = last
        return out

    poc_ff  = _ffill(out_poc)
    vah_ff  = _ffill(out_vah)
    val_ff  = _ffill(out_val)
    phigh_ff = _ffill(out_phigh)
    plow_ff  = _ffill(out_plow)

    # --- Step 4: attach to df --------------------------------------------------
    df = df.copy()
    df["vp_poc"]          = poc_ff
    df["vp_vah"]          = vah_ff
    df["vp_val"]          = val_ff
    df["vp_profile_high"] = phigh_ff
    df["vp_profile_low"]  = plow_ff
    df["vp_in_va"]        = (close >= val_ff) & (close <= vah_ff)
    df["vp_above_poc"]    = close > poc_ff

    return df
