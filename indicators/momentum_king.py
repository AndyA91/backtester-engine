"""
FS Momentum King v3.0 — Python conversion matching FSmomentumking.pine

Momentum oscillator with adaptive smoothing, dynamic neutral zone,
and 5-regime classification (STRONG_UP, WEAK_UP, FLAT, WEAK_DOWN, STRONG_DOWN).

Usage:
    from indicators.momentum_king import calc_momentum_king

    result = calc_momentum_king(df)
    # result["smoothed_momentum"]  — histogram values
    # result["momentum_strength"]  — normalized 0–1
    # result["signed_strength"]    — signed normalized (-1 to +1)
    # result["regime"]             — string per bar: STRONG_UP / WEAK_UP / FLAT / ...
    # result["neutral_zone_width"] — dynamic neutral zone half-width

    # Or add columns directly to your DataFrame:
    df = add_momentum_king_columns(df)
    # Adds: mk_momentum, mk_strength, mk_regime, mk_neutral_zone
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from engine import calc_ema, calc_atr, calc_highest


def calc_momentum_king(
    df: pd.DataFrame,
    ema_length: int = 14,
    smoothing_base: int = 3,
    atr_length: int = 14,
    volatility_factor: float = 1.5,
    norm_lookback: int = 200,
    strength_threshold: float = 0.6,
    base_neutral: float = 0.05,
    neutral_vol_factor: float = 1.0,
) -> dict:
    """
    Compute FS Momentum King v3.0, matching the Pine Script indicator.

    Parameters
    ----------
    df : DataFrame
        Must contain 'High', 'Low', 'Close' columns.
    ema_length : int
        Period for the slow EMA. Fast EMA = ema_length // 2.
    smoothing_base : int
        Minimum smoothing period for adaptive EMA on raw momentum.
    atr_length : int
        ATR period for adaptive smoothing and neutral zone.
    volatility_factor : float
        Multiplier on normalized ATR for adaptive smoothing.
    norm_lookback : int
        Bars used to normalize momentum strength to 0–1.
    strength_threshold : float
        Cutoff between Strong and Weak regimes (0–1).
    base_neutral : float
        Base neutral zone half-width before ATR scaling.
    neutral_vol_factor : float
        How much ATR widens the neutral zone.

    Returns
    -------
    dict with keys:
        smoothed_momentum : np.ndarray  — histogram values
        momentum_strength : np.ndarray  — normalized 0–1
        signed_strength   : np.ndarray  — signed (-1 to +1)
        regime            : np.ndarray  — string labels per bar
        neutral_zone_width: np.ndarray  — dynamic neutral zone
    """
    close = df["Close"].values
    n = len(close)

    # --- ATR (matches ta.atr via engine's calc_atr) ---
    atr = calc_atr(df, atr_length).values

    # --- Adaptive smoothing alpha (per bar) ---
    # FIX #1: normalize ATR to pip-scale for FX
    atr_normalized = np.where(close > 0, (atr / close) * 10000, 0.0)
    adaptive_smoothing = np.maximum(smoothing_base, atr_normalized * volatility_factor)
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

    # --- Normalized strength (0–1) ---
    abs_smoothed = np.abs(smoothed)

    # Rolling max of |smoothed_momentum| over norm_lookback bars
    # Matches ta.highest(math.abs(smoothed_momentum), norm_lookback)
    abs_peak = pd.Series(abs_smoothed).rolling(window=norm_lookback, min_periods=1).max().values

    momentum_strength = np.where(abs_peak > 0, abs_smoothed / abs_peak, 0.0)
    signed_strength = momentum_strength * np.sign(smoothed)

    # --- Dynamic neutral zone ---
    vol_adj = np.where(close > 0, (atr / close) * 100, 0.0)
    neutral_zone_width = base_neutral * (1 + vol_adj * neutral_vol_factor)

    # --- Regime classification (mutually exclusive, Flat first) ---
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
        "momentum_strength": momentum_strength,
        "signed_strength": signed_strength,
        "regime": regime,
        "neutral_zone_width": neutral_zone_width,
    }


def add_momentum_king_columns(
    df: pd.DataFrame, **kwargs
) -> pd.DataFrame:
    """
    Convenience wrapper: computes Momentum King and adds columns to df.

    Adds: mk_momentum, mk_strength, mk_signed_strength, mk_regime, mk_neutral_zone

    All kwargs are forwarded to calc_momentum_king().
    Returns the modified DataFrame (copy).
    """
    df = df.copy()
    result = calc_momentum_king(df, **kwargs)
    df["mk_momentum"] = result["smoothed_momentum"]
    df["mk_strength"] = result["momentum_strength"]
    df["mk_signed_strength"] = result["signed_strength"]
    df["mk_regime"] = result["regime"]
    df["mk_neutral_zone"] = result["neutral_zone_width"]
    return df
