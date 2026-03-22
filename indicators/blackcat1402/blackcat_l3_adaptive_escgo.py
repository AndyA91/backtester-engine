"""
Python translation of:
  [blackcat] L3 Adaptive ESCGO by blackcat1402
  https://www.tradingview.com/script/z7LqbXtm-blackcat-L3-Adaptive-ESCGO/

Adaptive version of the ESCGO: instead of a fixed CG window the dominant
cycle length is estimated bar-by-bar using Ehlers' instantaneous-period
algorithm (bandpass filter → Hilbert transform approximation → median
smoothing → IIR period smoother).  The adaptive period drives the CG and
stochastic-normalisation window (fastlen).

The Banker Fund Flow component uses a hardcoded 27-bar stochastic (same
as the Adaptive ESCGO Pine source).

Signal logic — LOOKBACK FUSION + COOLDOWN
-----------------------------------------
- B signal: ESCGO golden cross within last N bars  AND
            BFF bullish cross within last N bars.
- S signal: ESCGO death  cross within last N bars  AND
            BFF bearish cross within last N bars.
- Mutual-exclusion cooldown: minimum signalCooldown bars between same-type signals.
- Exit signals (XL/XS) use BFF only (identical to L3 ESCGO).

Adaptive cycle algorithm — why it must be iterative
----------------------------------------------------
The cycle estimator is an IIR filter: each bar's values (c, ip, p) depend
on the two previous bars.  No closed-form vectorised solution exists.
We therefore pre-allocate numpy arrays and iterate over bars in Python.
This is O(n) with minimal per-bar work (~20 arithmetic ops) and remains
fast on tens of thousands of bars.

Once fastlen[] is computed for every bar, the subsequent CG calculation
IS vectorised bar-by-variable-width windows via a Python loop over bars
(also O(n × max_window), typically O(n × 15)).  The stochastic, smoothing,
and ALMA steps are fully vectorised pandas operations.

Output columns (bc_ prefix)
----------------------------
  bc_escgo_fast       — adaptive ESCGO oscillator [-1, +1]
  bc_escgo_slow       — ALMA trigger
  bc_escgo_cycle      — adaptive fastlen (integer)
  bc_bff_trend        — Banker Fund Flow [0, 100]
  bc_bff_bbl          — Bull Bear Line   [0, 100]
  bc_b_signal         — fused B (buy) signal after cooldown (bool)
  bc_s_signal         — fused S (sell) signal after cooldown (bool)
  bc_long_exit        — long exit signal  (bool)
  bc_short_exit       — short exit signal (bool)

Usage
-----
  from indicators.blackcat1402.blackcat_l3_adaptive_escgo import calc_bc_l3_adaptive_escgo
  df = calc_bc_l3_adaptive_escgo(df)
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers shared with L3 ESCGO (duplicated here for module independence)
# ---------------------------------------------------------------------------

def _wsa(src: pd.Series, length: int, weight: float = 1.0) -> pd.Series:
    """Pine's calculate_weighted_simple_average (recursive EMA-like filter)."""
    src_vals = np.where(np.isnan(src.values), 0.0, src.values)
    result = np.empty(len(src_vals))
    prev = 0.0
    for i in range(len(src_vals)):
        curr = (src_vals[i] * weight + prev * (float(length) - weight)) / float(length)
        result[i] = curr
        prev = curr
    return pd.Series(result, index=src.index)


def _alma(series: pd.Series, length: int, offset: float, sigma: float) -> pd.Series:
    m = offset * (length - 1)
    s = length / sigma
    w = np.exp(-((np.arange(length, dtype=float) - m) ** 2) / (2.0 * s * s))
    w /= w.sum()
    w_rev = w[::-1]
    return series.rolling(length).apply(lambda x: (x * w_rev).sum(), raw=True)


def _crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def _crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


def _med3(x: float, y: float, z: float) -> float:
    """Median of three values (Pine's med function)."""
    return x + y + z - min(x, min(y, z)) - max(x, max(y, z))


# ---------------------------------------------------------------------------
# Adaptive cycle detection (Ehlers instantaneous period)
# ---------------------------------------------------------------------------

def _adaptive_fastlen(src_arr: np.ndarray, acgo_alpha: float = 0.19) -> np.ndarray:
    """
    Compute the adaptive CG window length for every bar.

    Returns an integer array of length n.  Values before bar 6 are set to 2
    (the minimum).  Mirrors the Pine pseudocode exactly.
    """
    n = len(src_arr)
    # Smoothed source: s = (src + 2*src[1] + 2*src[2] + src[3]) / 6
    s   = np.zeros(n)
    c   = np.zeros(n)  # bandpass filter output
    ip  = np.zeros(n)  # instantaneous period
    p   = np.zeros(n)  # smoothed period
    q1  = np.zeros(n)
    I1  = np.zeros(n)
    dp  = np.zeros(n)

    a1  = 1.0 - 0.5 * acgo_alpha
    a2  = 1.0 - acgo_alpha

    for t in range(n):
        # Smoothed source (needs 3 previous bars)
        s[t] = (
            src_arr[t]
            + 2.0 * (src_arr[t - 1] if t >= 1 else src_arr[t])
            + 2.0 * (src_arr[t - 2] if t >= 2 else src_arr[t])
            + (src_arr[t - 3] if t >= 3 else src_arr[t])
        ) / 6.0

        # Bandpass / adaptive cycle filter
        if t < 7:
            s_1 = src_arr[t - 1] if t >= 1 else src_arr[t]
            s_2 = src_arr[t - 2] if t >= 2 else src_arr[t]
            c[t] = (src_arr[t] - 2.0 * s_1 + s_2) / 4.0
        else:
            c[t] = (
                a1 * a1 * (s[t] - 2.0 * s[t - 1] + s[t - 2])
                + 2.0 * a2 * c[t - 1]
                - a2 * a2 * c[t - 2]
            )

        # Quadrature and in-phase
        ip_prev = ip[t - 1] if t >= 1 else 0.0
        q1[t] = (
            (0.0962 * c[t]
             + 0.5769 * (c[t - 2] if t >= 2 else 0.0)
             - 0.5769 * (c[t - 4] if t >= 4 else 0.0)
             - 0.0962 * (c[t - 6] if t >= 6 else 0.0))
            * (0.5 + 0.08 * ip_prev)
        )
        I1[t] = c[t - 3] if t >= 3 else 0.0

        # Delta phase
        q1_prev = q1[t - 1] if t >= 1 else 0.0
        I1_prev = I1[t - 1] if t >= 1 else 0.0
        if q1[t] != 0.0 and q1_prev != 0.0:
            denom = 1.0 + I1[t] * I1_prev / (q1[t] * q1_prev)
            dp_ = (I1[t] / q1[t] - I1_prev / q1_prev) / (denom if denom != 0.0 else 1e-10)
        else:
            dp_ = 0.0
        dp[t] = max(0.1, min(1.1, dp_))

        # Median smoothing over dp (needs 4 prior bars)
        dp1 = dp[t - 1] if t >= 1 else dp[t]
        dp2 = dp[t - 2] if t >= 2 else dp[t]
        dp3 = dp[t - 3] if t >= 3 else dp[t]
        dp4 = dp[t - 4] if t >= 4 else dp[t]
        md = _med3(dp[t], dp1, _med3(dp2, dp3, dp4))

        dc = 15.0 if md == 0.0 else 6.28318 / md + 0.5
        ip[t] = 0.33 * dc + 0.67 * (ip[t - 1] if t >= 1 else 0.0)
        p[t]  = 0.15 * ip[t] + 0.85 * (p[t - 1] if t >= 1 else 0.0)

    # fastlen = max(2, round(p / 2))
    fastlen = np.maximum(2, np.round(p / 2.0).astype(int))
    return fastlen


# ---------------------------------------------------------------------------
# Adaptive CG calculation (variable window per bar)
# ---------------------------------------------------------------------------

def _adaptive_cg(src_arr: np.ndarray, fastlen_arr: np.ndarray) -> np.ndarray:
    """
    Compute CG for each bar using its adaptive window length.

    CG[t] = -Σ_{i=0}^{FL-1} (i+1)*src[t-i]  /  Σ_{i=0}^{FL-1} src[t-i]  +  (FL+1)/2

    Loop is O(n × mean_window) — typically 15 ops per bar.  Using a Python loop
    here is acceptable because the inner work is minimal and the window changes
    bar to bar, making stride tricks impractical.
    """
    n = len(src_arr)
    cg = np.full(n, np.nan)
    for t in range(n):
        fl = int(fastlen_arr[t])
        if t < fl - 1:
            continue
        window = src_arr[t - fl + 1: t + 1][::-1]  # window[0]=current, ..., window[fl-1]=oldest
        weights = np.arange(1, fl + 1, dtype=float)
        num = np.dot(weights, window)
        den = window.sum()
        if den != 0.0:
            cg[t] = -num / den + (fl + 1) / 2.0
    return cg


# ---------------------------------------------------------------------------
# Lookback-window signal check
# ---------------------------------------------------------------------------

def _recent_within(signal: np.ndarray, lookback: int) -> np.ndarray:
    """
    For each bar t, True if signal was True on any bar in [t-lookback+1, t].
    Equivalent to Pine's loop: for i = 0 to lookback-1: if signal[i] then recent = true
    Vectorised with a rolling max.
    """
    s = pd.Series(signal.astype(float))
    return s.rolling(lookback, min_periods=1).max().values.astype(bool)


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def calc_bc_l3_adaptive_escgo(
    df: pd.DataFrame,
    slow_length: int = 3,
    alma_offset: float = 0.85,
    alma_sigma: float = 6.0,
    acgo_alpha: float = 0.19,
    lookback_period: int = 5,
    signal_cooldown: int = 8,
) -> pd.DataFrame:
    """
    Python translation of [blackcat] L3 Adaptive ESCGO by blackcat1402.
    - Input:  df with columns open, high, low, close, volume
    - Output: df with new bc_ prefixed columns appended
    """
    df = df.copy()
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    # -----------------------------------------------------------------------
    # 1. Lao Xu 1949 Pivot Point price source
    # -----------------------------------------------------------------------
    src = (h + l + 2.0 * c + o / 2.0) / 4.5
    src_arr = src.values.astype(float)

    # -----------------------------------------------------------------------
    # 2. Adaptive cycle length (iterative Ehlers algorithm)
    # -----------------------------------------------------------------------
    fastlen_arr = _adaptive_fastlen(src_arr, acgo_alpha)

    # -----------------------------------------------------------------------
    # 3. Adaptive Center of Gravity
    # -----------------------------------------------------------------------
    cg_arr = _adaptive_cg(src_arr, fastlen_arr)
    cg = pd.Series(cg_arr, index=df.index)

    # -----------------------------------------------------------------------
    # 4. Stochastic normalisation using adaptive window
    # -----------------------------------------------------------------------
    # Rolling max/min with variable window — must loop (window changes per bar)
    n = len(src_arr)
    maxc = np.full(n, np.nan)
    minc = np.full(n, np.nan)
    for t in range(n):
        fl = int(fastlen_arr[t])
        if t < fl - 1 or np.isnan(cg_arr[t]):
            continue
        window = cg_arr[max(0, t - fl + 1): t + 1]
        maxc[t] = np.nanmax(window)
        minc[t] = np.nanmin(window)

    rng = maxc - minc
    safe = rng != 0
    v1_arr = np.where(safe, (cg_arr - minc) / np.where(safe, rng, 1.0), 0.0)
    v1 = pd.Series(v1_arr, index=df.index)

    # -----------------------------------------------------------------------
    # 5. 4-3-2-1 smoothing, rescale to [-1, +1]
    # -----------------------------------------------------------------------
    v2 = (4.0 * v1 + 3.0 * v1.shift(1) + 2.0 * v1.shift(2) + v1.shift(3)) / 10.0
    escgo_fast = 2.0 * (v2 - 0.5)

    # -----------------------------------------------------------------------
    # 6. ALMA slow line
    # -----------------------------------------------------------------------
    escgo_slow = _alma(escgo_fast, slow_length, alma_offset, alma_sigma)

    # -----------------------------------------------------------------------
    # 7. Banker Fund Flow (hardcoded 27-bar stochastic as in Adaptive ESCGO Pine)
    # -----------------------------------------------------------------------
    ll27 = l.rolling(27).min()
    hh27 = h.rolling(27).max()
    rng27 = hh27 - ll27
    raw_stoch27 = np.where(rng27 != 0, (c - ll27) / rng27 * 100.0, 50.0)
    raw_stoch27 = pd.Series(raw_stoch27, index=df.index)

    wsa1 = _wsa(raw_stoch27, 5, 1.0)
    wsa2 = _wsa(wsa1, 3, 1.0)
    bff_trend = (3.0 * wsa1 - 2.0 * wsa2 - 50.0) * 1.032 + 50.0

    # -----------------------------------------------------------------------
    # 8. Bull Bear Line
    # -----------------------------------------------------------------------
    tp = (2.0 * c + h + l + o) / 5.0
    ll34 = l.rolling(34).min()
    hh34 = h.rolling(34).max()
    rng34 = hh34 - ll34
    bbl_raw = np.where(rng34 != 0, (tp - ll34) / rng34 * 100.0, 50.0)
    bbl = pd.Series(bbl_raw, index=df.index).ewm(span=13, adjust=False).mean()

    # -----------------------------------------------------------------------
    # 9. ESCGO and BFF crossover signals
    # -----------------------------------------------------------------------
    escgo_golden = _crossover(escgo_fast, escgo_slow).values
    escgo_death  = _crossunder(escgo_fast, escgo_slow).values
    bff_bull_x   = _crossover(bff_trend, bbl).values
    bff_bear_x   = _crossunder(bff_trend, bbl).values

    # -----------------------------------------------------------------------
    # 10. Lookback fusion: both crosses must have occurred within N bars
    # -----------------------------------------------------------------------
    esc_recent_gold  = _recent_within(escgo_golden, lookback_period)
    esc_recent_death = _recent_within(escgo_death,  lookback_period)
    bff_recent_bull  = _recent_within(bff_bull_x,   lookback_period)
    bff_recent_bear  = _recent_within(bff_bear_x,   lookback_period)

    raw_b = esc_recent_gold & bff_recent_bull
    raw_s = esc_recent_death & bff_recent_bear

    # -----------------------------------------------------------------------
    # 11. Cooldown filter + mutual exclusion (iterative — state machine)
    # -----------------------------------------------------------------------
    b_signal = np.zeros(n, dtype=bool)
    s_signal = np.zeros(n, dtype=bool)
    bars_b = signal_cooldown + 1
    bars_s = signal_cooldown + 1
    escgo_vals = escgo_fast.values

    for t in range(n):
        bars_b += 1
        bars_s += 1

        cooled_b = raw_b[t] and bars_b >= signal_cooldown
        cooled_s = raw_s[t] and bars_s >= signal_cooldown

        # Mutual exclusion: if both fire, use ESCGO momentum
        if cooled_b and cooled_s:
            if escgo_vals[t] > 0:
                cooled_s = False
            else:
                cooled_b = False

        if cooled_b:
            b_signal[t] = True
            bars_b = 0
            bars_s = 0
        if cooled_s:
            s_signal[t] = True
            bars_s = 0
            bars_b = 0

    # -----------------------------------------------------------------------
    # 12. Exit signals (BFF only)
    # -----------------------------------------------------------------------
    bff_prev = bff_trend.shift(1).fillna(bff_trend)
    long_exit  = bff_bear_x | (bff_trend < bbl).values & (bff_trend < bff_prev * 0.95).values
    short_exit = bff_bull_x | (bff_trend > bbl).values & (bff_trend > bff_prev * 1.05).values

    # -----------------------------------------------------------------------
    # 13. Attach columns
    # -----------------------------------------------------------------------
    df["bc_escgo_fast"]   = escgo_fast
    df["bc_escgo_slow"]   = escgo_slow
    df["bc_escgo_cycle"]  = pd.Series(fastlen_arr.astype(int), index=df.index)
    df["bc_bff_trend"]    = bff_trend
    df["bc_bff_bbl"]      = bbl
    df["bc_b_signal"]     = b_signal
    df["bc_s_signal"]     = s_signal
    df["bc_long_exit"]    = long_exit.astype(bool)
    df["bc_short_exit"]   = short_exit.astype(bool)

    return df
