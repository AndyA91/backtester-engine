"""
Python translation of:
  [blackcat] L3 ESCGO by blackcat1402
  https://www.tradingview.com/script/L7zyU32y-blackcat-L3-ESCGO/

Fused Ehlers Stochastic Center-of-Gravity Oscillator + Banker Fund Flow.

Signal logic
------------
- Fast line (ESCGO): CG oscillator, stochastic-normalised, 4-3-2-1 smoothed, rescaled to [-1, +1].
- Slow line: ALMA of the fast line (trigger).
- Banker Fund Flow (BFF): double-WSA of a 7-bar stochastic of close, scaled to [0, 100].
- Bull Bear Line (BBL): 13-bar EMA of (typicalPrice − LL34) / (HH34 − LL34) × 100.
- Long entry:  ESCGO bullish cross AND (BFF cross-above BBL OR BFF > BBL).
- Short entry: ESCGO bearish cross AND (BFF cross-below BBL OR BFF < BBL).
- Long exit:   BFF < BBL  OR  BFF dropping > 5% from prior bar.
- Short exit:  BFF > BBL  OR  BFF rising  > 5% from prior bar.

Vectorisation notes
-------------------
- CG window sum: sliding_window_view gives O(n) at the cost of an n×W memory layout.
- ALMA: one rolling().apply() with pre-computed Gaussian weights (reversed for rolling order).
- WSA: inherently recursive (each bar depends on previous output) — implemented as a
  single O(n) Python loop; fast enough for any realistic price series.
- All pivot-less computations (rolling min/max, EMA, RSI) are fully vectorised pandas calls.

Output columns (bc_ prefix)
----------------------------
  bc_escgo_fast       — ESCGO oscillator [-1, +1]
  bc_escgo_slow       — ALMA trigger line
  bc_bff_trend        — Banker Fund Flow [0, 100]
  bc_bff_bbl          — Bull Bear Line   [0, 100]
  bc_bff_state        — Banker candle state integer:
                          0 = neutral, 1 = yellow (entry), 2 = green (increase),
                          3 = white (decrease),  4 = red (exit),  5 = blue (rebound)
  bc_long_entry       — fused long entry (bool)
  bc_short_entry      — fused short entry (bool)
  bc_long_exit        — long exit (bool)
  bc_short_exit       — short exit (bool)

Usage
-----
  from indicators.blackcat1402.blackcat_l3_escgo import calc_bc_l3_escgo
  df = calc_bc_l3_escgo(df)
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _wsa(src: pd.Series, length: int, weight: float = 1.0) -> pd.Series:
    """
    Pine's calculate_weighted_simple_average.

    Recurrence (initialised at 0.0, matching Pine's 'var float output = 0.0'):
        output[t] = (src[t] * weight + output[t-1] * (length - weight)) / length

    For weight=1 this is an EMA with alpha=1/length (different from standard
    pandas EMA which uses alpha=2/(length+1)).
    """
    alpha_num = weight
    alpha_den = float(length)
    src_vals = np.where(np.isnan(src.values), 0.0, src.values)
    result = np.empty(len(src_vals))
    prev = 0.0
    for i in range(len(src_vals)):
        curr = (src_vals[i] * alpha_num + prev * (alpha_den - alpha_num)) / alpha_den
        result[i] = curr
        prev = curr
    return pd.Series(result, index=src.index)


def _alma(series: pd.Series, length: int, offset: float, sigma: float) -> pd.Series:
    """
    Arnaud Legoux Moving Average.

    Pine weights w[i] = exp(-((i - m)^2) / (2*s^2)), i=0..length-1, i=0 = most recent.
    Rolling window: x[0] = oldest, x[-1] = most recent → reverse weights.
    """
    m = offset * (length - 1)
    s = length / sigma
    k = np.arange(length, dtype=float)
    w = np.exp(-((k - m) ** 2) / (2.0 * s * s))
    w /= w.sum()
    w_rev = w[::-1]  # align to rolling order (oldest first)
    return series.rolling(length).apply(lambda x: (x * w_rev).sum(), raw=True)


def _crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    """True on bars where a crosses above b (was below, now above)."""
    return (a > b) & (a.shift(1) <= b.shift(1))


def _crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    """True on bars where a crosses below b (was above, now below)."""
    return (a < b) & (a.shift(1) >= b.shift(1))


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def calc_bc_l3_escgo(
    df: pd.DataFrame,
    fast_length: int = 13,
    slow_length: int = 3,
    alma_offset: float = 0.85,
    alma_sigma: float = 6.0,
    mid_trend_period: int = 34,
    small_trend_period: int = 5,
    tiny_trend_period: int = 3,
    overbought: int = 80,
    oversold: int = 20,
) -> pd.DataFrame:
    """
    Python translation of [blackcat] L3 ESCGO by blackcat1402.
    - Input:  df with columns open, high, low, close, volume
    - Output: df with new bc_ prefixed columns appended
    """
    df = df.copy()
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    # -----------------------------------------------------------------------
    # 1. Price source: "Lao Xu 1949 Pivot Point"
    # -----------------------------------------------------------------------
    src = (h + l + 2.0 * c + o / 2.0) / 4.5

    # -----------------------------------------------------------------------
    # 2. Center of Gravity (CG)
    #
    #   cg = -Σ(i+1)*src[i] / Σsrc[i]  +  (fast_length+1)/2
    #   where i=0 is current bar (Pine convention).
    #
    # Vectorised with sliding_window_view:
    #   window[j] = [src[j], src[j+1], ..., src[j+fast_length-1]] (chronological)
    #   Pine's src[0..fast_length-1] at bar j+fast_length-1 maps to
    #       src[0]=window[-1], src[1]=window[-2], …
    #   so Pine's src[i] = window[fast_length-1-i]
    #   weights[i] = i+1  →  weights for window position k = fast_length - k
    # -----------------------------------------------------------------------
    src_arr = src.values.astype(float)
    n = len(src_arr)

    # Weights in sliding-window order (oldest = index 0 → weight fast_length; newest → weight 1)
    W = np.arange(fast_length, 0, -1, dtype=float)  # [fast_length, ..., 1]
    cg_arr = np.full(n, np.nan)

    if n >= fast_length:
        from numpy.lib.stride_tricks import sliding_window_view
        wins = sliding_window_view(src_arr, fast_length)  # shape (n-FL+1, FL)
        num_v = (wins * W).sum(axis=1)
        den_v = wins.sum(axis=1)
        # cg = -num/denom + (FL+1)/2
        safe = den_v != 0
        cg_vals = np.where(safe, -num_v / np.where(safe, den_v, 1.0) + (fast_length + 1) / 2.0, 0.0)
        cg_arr[fast_length - 1:] = cg_vals

    cg = pd.Series(cg_arr, index=df.index)

    # -----------------------------------------------------------------------
    # 3. Stochastic normalisation of CG
    # -----------------------------------------------------------------------
    maxc = cg.rolling(fast_length).max()
    minc = cg.rolling(fast_length).min()
    rng = maxc - minc
    v1 = np.where(rng != 0, (cg - minc) / rng, 0.0)
    v1 = pd.Series(v1, index=df.index)

    # -----------------------------------------------------------------------
    # 4. Weighted 4-3-2-1 smoothing, then rescale to [-1, +1]
    # -----------------------------------------------------------------------
    v2 = (4.0 * v1 + 3.0 * v1.shift(1) + 2.0 * v1.shift(2) + v1.shift(3)) / 10.0
    escgo_fast = 2.0 * (v2 - 0.5)

    # -----------------------------------------------------------------------
    # 5. Slow (trigger) line: ALMA
    # -----------------------------------------------------------------------
    escgo_slow = _alma(escgo_fast, slow_length, alma_offset, alma_sigma)

    # -----------------------------------------------------------------------
    # 6. Banker Fund Flow (BFF)
    #
    #   stoch_period = small_trend_period + tiny_trend_period - 1
    #   raw_stoch    = (close - LL(stoch_period)) / (HH(stoch_period) - LL(stoch_period)) * 100
    #   wsa1         = WSA(raw_stoch, small_trend_period, 1)
    #   wsa2         = WSA(wsa1, tiny_trend_period, 1)
    #   bff          = (3*wsa1 - 2*wsa2 - 50) * 1.032 + 50
    # -----------------------------------------------------------------------
    sp = small_trend_period + tiny_trend_period - 1
    ll_sp = l.rolling(sp).min()
    hh_sp = h.rolling(sp).max()
    rng_sp = hh_sp - ll_sp
    raw_stoch = np.where(rng_sp != 0, (c - ll_sp) / rng_sp * 100.0, 50.0)
    raw_stoch = pd.Series(raw_stoch, index=df.index)

    wsa1 = _wsa(raw_stoch, small_trend_period, 1.0)
    wsa2 = _wsa(wsa1, tiny_trend_period, 1.0)
    bff_trend = (3.0 * wsa1 - 2.0 * wsa2 - 50.0) * 1.032 + 50.0

    # -----------------------------------------------------------------------
    # 7. Bull Bear Line (BBL)
    #
    #   typical_price = (2*close + high + low + open) / 5
    #   bbl = EMA( (typical_price - LL34) / (HH34 - LL34) * 100 , 13 )
    # -----------------------------------------------------------------------
    tp = (2.0 * c + h + l + o) / 5.0
    ll34 = l.rolling(mid_trend_period).min()
    hh34 = h.rolling(mid_trend_period).max()
    rng34 = hh34 - ll34
    bbl_raw = np.where(rng34 != 0, (tp - ll34) / rng34 * 100.0, 50.0)
    bbl = pd.Series(bbl_raw, index=df.index).ewm(span=13, adjust=False).mean()

    # -----------------------------------------------------------------------
    # 8. Banker candle states
    # -----------------------------------------------------------------------
    bff_prev = bff_trend.shift(1).fillna(bff_trend)

    yellow = _crossover(bff_trend, bbl) & (bbl < oversold)       # entry
    green  = bff_trend > bbl                                       # increase position
    white  = bff_trend < bff_prev * 0.95                          # decrease (dropping >5%)
    red    = bff_trend < bbl                                       # exit
    blue   = red & ~white                                          # weak rebound

    # Encode state: priority — yellow > white > blue > green > red
    state = pd.Series(0, index=df.index)
    state = state.where(~red,    4)
    state = state.where(~green,  2)
    state = state.where(~blue,   5)
    state = state.where(~white,  3)
    state = state.where(~yellow, 1)

    # -----------------------------------------------------------------------
    # 9. Signal fusion
    # -----------------------------------------------------------------------
    escgo_bull_cross = _crossover(escgo_fast, escgo_slow)
    escgo_bear_cross = _crossunder(escgo_fast, escgo_slow)
    bff_bull_cross   = _crossover(bff_trend, bbl)
    bff_bear_cross   = _crossunder(bff_trend, bbl)

    long_entry  = escgo_bull_cross & (bff_bull_cross | green)
    short_entry = escgo_bear_cross & (bff_bear_cross | red)

    # Long exit:  BFF crosses below BBL OR BFF drops > 5%
    long_exit  = bff_bear_cross | white
    # Short exit: BFF crosses above BBL OR BFF rises > 5% from previous
    short_exit = bff_bull_cross | (bff_trend > bff_prev * 1.05)

    # -----------------------------------------------------------------------
    # 10. Attach columns
    # -----------------------------------------------------------------------
    df["bc_escgo_fast"]   = escgo_fast
    df["bc_escgo_slow"]   = escgo_slow
    df["bc_bff_trend"]    = bff_trend
    df["bc_bff_bbl"]      = bbl
    df["bc_bff_state"]    = state.astype(int)
    df["bc_long_entry"]   = long_entry.astype(bool)
    df["bc_short_entry"]  = short_entry.astype(bool)
    df["bc_long_exit"]    = long_exit.astype(bool)
    df["bc_short_exit"]   = short_exit.astype(bool)

    return df
