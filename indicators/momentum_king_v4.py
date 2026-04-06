"""
FS Momentum King v4.0 — Improved for Renko charts

Changes from v3.0:
  1. Neutral zone: ATR-proportional (not fixed price constant).
     v3 base_neutral=0.05 was dead on BTC (0.04% of momentum range).
     v4 uses neutral_atr_pct * ATR — scales correctly across all instruments.
  2. Adaptive smoothing: capped at max_smooth (default 10) for Renko.
     v3 averaged 32-bar smoothing on BTC Renko — too heavy on pre-denoised data.
     v4 caps it to keep lag under control (mean ~6 bars vs ~16).
  3. Normalization: percentile-rank over lookback window (not rolling-max).
     v3 rolling-max had "peak shadow" — after a spike, 200 bars read as weak.
     v4 percentile rank gives balanced 30/40/30 weak/mid/strong distribution.
  4. Signal line: EMA of smoothed momentum for crossover entries.
     v3 had zones only. v4 adds mk_signal + mk_cross_up/mk_cross_dn.

Usage:
    from indicators.momentum_king_v4 import calc_momentum_king_v4

    result = calc_momentum_king_v4(df)
    # result["smoothed_momentum"]  — histogram values
    # result["signal_line"]        — EMA of smoothed momentum
    # result["momentum_strength"]  — percentile-ranked 0–1
    # result["signed_strength"]    — signed (-1 to +1)
    # result["regime"]             — STRONG_UP / WEAK_UP / FLAT / WEAK_DOWN / STRONG_DOWN
    # result["neutral_zone_width"] — ATR-proportional neutral zone
    # result["cross_up"]           — bool: smoothed crossed above signal
    # result["cross_dn"]           — bool: smoothed crossed below signal
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from engine import calc_ema, calc_atr


def calc_momentum_king_v4(
    df: pd.DataFrame,
    ema_length: int = 14,
    smoothing_base: int = 3,
    max_smooth: int = 10,
    atr_length: int = 14,
    volatility_factor: float = 1.5,
    norm_lookback: int = 200,
    strength_threshold: float = 0.6,
    neutral_atr_pct: float = 0.3,
    signal_length: int = 9,
) -> dict:
    """
    Compute FS Momentum King v4.0 (Renko-optimized).

    Parameters
    ----------
    df : DataFrame
        Must contain 'High', 'Low', 'Close' columns.
    ema_length : int
        Period for the slow EMA. Fast EMA = ema_length // 2.
    smoothing_base : int
        Minimum smoothing period for adaptive EMA on raw momentum.
    max_smooth : int
        Maximum smoothing period (caps adaptive EMA for Renko).
    atr_length : int
        ATR period for adaptive smoothing and neutral zone.
    volatility_factor : float
        Multiplier on normalized ATR for adaptive smoothing.
    norm_lookback : int
        Bars used to percentile-rank momentum strength (0-1).
    strength_threshold : float
        Cutoff between Strong and Weak regimes (0-1).
    neutral_atr_pct : float
        Neutral zone width as fraction of ATR (0.3 = 30% of ATR).
    signal_length : int
        EMA period for the signal line.

    Returns
    -------
    dict with keys:
        smoothed_momentum, signal_line, momentum_strength,
        signed_strength, regime, neutral_zone_width,
        cross_up, cross_dn
    """
    close = df["Close"].values
    n = len(close)

    # --- ATR ---
    atr = calc_atr(df, atr_length).values

    # --- Adaptive smoothing alpha (capped for Renko) ---
    atr_normalized = np.where(close > 0, (atr / close) * 10000, 0.0)
    adaptive_smoothing = np.clip(
        atr_normalized * volatility_factor,
        smoothing_base,
        max_smooth,
    )
    alpha = 2.0 / (adaptive_smoothing + 1.0)

    # --- Momentum = fast EMA - slow EMA ---
    fast_len = max(1, ema_length // 2)
    fast_ema = calc_ema(df["Close"], fast_len).values
    slow_ema = calc_ema(df["Close"], ema_length).values
    momentum = fast_ema - slow_ema

    # --- Custom adaptive EMA smoothing (stateful, per-bar alpha) ---
    smoothed = np.zeros(n)
    smoothed[0] = momentum[0] if not np.isnan(momentum[0]) else 0.0
    for i in range(1, n):
        m = momentum[i] if not np.isnan(momentum[i]) else 0.0
        a = alpha[i] if not np.isnan(alpha[i]) else 2.0 / (smoothing_base + 1.0)
        smoothed[i] = a * m + (1 - a) * smoothed[i - 1]

    # --- Signal line (EMA of smoothed momentum) ---
    signal_line = calc_ema(pd.Series(smoothed, index=df.index), signal_length).values

    # --- Crossovers ---
    cross_up = np.zeros(n, dtype=bool)
    cross_dn = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if np.isnan(signal_line[i]) or np.isnan(signal_line[i - 1]):
            continue
        if smoothed[i] > signal_line[i] and smoothed[i - 1] <= signal_line[i - 1]:
            cross_up[i] = True
        elif smoothed[i] < signal_line[i] and smoothed[i - 1] >= signal_line[i - 1]:
            cross_dn[i] = True

    # --- Percentile-rank strength (0-1) ---
    abs_smoothed = np.abs(smoothed)
    momentum_strength = (
        pd.Series(abs_smoothed)
        .rolling(window=norm_lookback, min_periods=1)
        .rank(pct=True)
        .values
    )
    signed_strength = momentum_strength * np.sign(smoothed)

    # --- ATR-proportional neutral zone ---
    neutral_zone_width = neutral_atr_pct * atr

    # --- Regime classification (Flat first, mutually exclusive) ---
    regime = np.full(n, "FLAT", dtype=object)

    is_flat = (smoothed < neutral_zone_width) & (smoothed > -neutral_zone_width)
    is_strong_up = ~is_flat & (smoothed > 0) & (momentum_strength > strength_threshold)
    is_weak_up = ~is_flat & (smoothed > 0) & (momentum_strength <= strength_threshold)
    is_strong_down = ~is_flat & (smoothed < 0) & (momentum_strength > strength_threshold)
    is_weak_down = ~is_flat & (smoothed < 0) & (momentum_strength <= strength_threshold)

    regime[is_strong_up] = "STRONG_UP"
    regime[is_weak_up] = "WEAK_UP"
    regime[is_strong_down] = "STRONG_DOWN"
    regime[is_weak_down] = "WEAK_DOWN"

    return {
        "smoothed_momentum": smoothed,
        "signal_line": signal_line,
        "momentum_strength": momentum_strength,
        "signed_strength": signed_strength,
        "regime": regime,
        "neutral_zone_width": neutral_zone_width,
        "cross_up": cross_up,
        "cross_dn": cross_dn,
    }
