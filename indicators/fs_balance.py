"""
FS Balance v3.0 — Python conversion matching FSbalance.pine

Composite order-flow imbalance oscillator combining volume delta,
tick imbalance, and price imbalance into a single smoothed histogram
with 5-regime classification.

Requires Volume column in the DataFrame.

Usage:
    from indicators.fs_balance import calc_fs_balance, add_fs_balance_columns

    result = calc_fs_balance(df)
    # result["smoothed_imbalance"]  — histogram values
    # result["signal_line"]         — EMA of smoothed imbalance
    # result["imbalance_strength"]  — normalized 0–1
    # result["regime"]              — STRONG_BUY / WEAK_BUY / BALANCED / WEAK_SELL / STRONG_SELL
    # result["neutral_width"]       — dynamic neutral zone half-width

    # Or add columns directly:
    df = add_fs_balance_columns(df)
    # Adds: fb_imbalance, fb_signal, fb_strength, fb_regime, fb_neutral
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from engine import calc_ema, calc_atr, calc_sma


def calc_fs_balance(
    df: pd.DataFrame,
    vol_length: int = 10,
    tick_length: int = 5,
    price_length: int = 7,
    smoothing_base: int = 3,
    atr_length: int = 14,
    volatility_factor: float = 1.5,
    signal_length: int = 5,
    norm_lookback: int = 200,
    strength_threshold: float = 0.6,
    base_neutral: float = 0.05,
    neutral_vol_factor: float = 1.0,
    w_volume: float = 1.0,
    w_tick: float = 1.0,
    w_price: float = 1.0,
) -> dict:
    """
    Compute FS Balance v3.0, matching the Pine Script indicator.

    Parameters
    ----------
    df : DataFrame
        Must contain 'Open', 'High', 'Low', 'Close', 'Volume' columns.
    vol_length : int
        SMA period for normalizing volume delta.
    tick_length : int
        Rolling window for tick imbalance (up-ticks vs down-ticks).
    price_length : int
        ATR period for normalizing price imbalance.
    smoothing_base : int
        Minimum adaptive EMA smoothing period.
    atr_length : int
        ATR period for adaptive smoothing and neutral zone.
    volatility_factor : float
        Multiplier on normalized ATR for adaptive smoothing.
    signal_length : int
        EMA period for the signal line.
    norm_lookback : int
        Bars for strength normalization (0–1).
    strength_threshold : float
        Cutoff between Strong and Weak regimes.
    base_neutral : float
        Base neutral zone half-width.
    neutral_vol_factor : float
        ATR influence on neutral zone width.
    w_volume, w_tick, w_price : float
        Component weights in the composite sum.

    Returns
    -------
    dict with keys:
        smoothed_imbalance : np.ndarray
        signal_line        : np.ndarray
        imbalance_strength : np.ndarray  (0–1)
        regime             : np.ndarray  (string labels)
        neutral_width      : np.ndarray
    """
    close = df["Close"].values
    open_ = df["Open"].values
    volume = df["Volume"].values.astype(float)
    n = len(close)

    # --- ATR ---
    atr_main = calc_atr(df, atr_length).values
    atr_price = calc_atr(df, price_length).values

    # --- Adaptive smoothing (FIX #1: normalized ATR for FX) ---
    atr_normalized = np.where(close > 0, (atr_main / close) * 10000, 0.0)
    adaptive_smoothing = np.maximum(smoothing_base, atr_normalized * volatility_factor)
    alpha = 2.0 / (adaptive_smoothing + 1.0)

    # --- Volume Delta (FIX #2: close vs prev close, not close vs open) ---
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    up_vol = volume * (close > prev_close).astype(float)
    down_vol = volume * (close < prev_close).astype(float)
    volume_delta = up_vol - down_vol

    avg_vol = calc_sma(pd.Series(volume, index=df.index), vol_length).values
    denom_vol = np.maximum(avg_vol, 1e-10)
    normalized_volume_delta = volume_delta / denom_vol

    # --- Tick Imbalance (FIX #2: same doji fix) ---
    up_bool = (close > prev_close).astype(float)
    down_bool = (close < prev_close).astype(float)

    tick_up = calc_sma(pd.Series(up_bool, index=df.index), tick_length).values * tick_length
    tick_down = calc_sma(pd.Series(down_bool, index=df.index), tick_length).values * tick_length
    tick_imbalance = tick_up - tick_down

    # Normalize by rolling max
    abs_tick = np.abs(tick_imbalance)
    max_tick_window = pd.Series(abs_tick).rolling(window=100, min_periods=1).max().values
    denom_tick = np.maximum(max_tick_window, 1.0)
    normalized_tick_imbalance = tick_imbalance / denom_tick

    # --- Price Imbalance (body / ATR) ---
    price_change = (close - open_) / np.maximum(atr_price, 1e-10)
    normalized_price_imbalance = price_change

    # --- Composite (FIX #3: weighted sum) ---
    composite = (w_volume * normalized_volume_delta
                 + w_tick * normalized_tick_imbalance
                 + w_price * normalized_price_imbalance)

    # Handle NaNs from indicator warmup
    composite = np.nan_to_num(composite, nan=0.0)

    # --- Adaptive EMA smoothing (stateful, per-bar alpha) ---
    smoothed = np.zeros(n)
    smoothed[0] = composite[0]
    for i in range(1, n):
        a = alpha[i] if not np.isnan(alpha[i]) else 2.0 / (smoothing_base + 1.0)
        smoothed[i] = a * composite[i] + (1 - a) * smoothed[i - 1]

    # --- Signal line (EMA of smoothed) ---
    signal_line = calc_ema(pd.Series(smoothed, index=df.index), signal_length).values

    # --- Normalized strength (0–1) ---
    abs_smoothed = np.abs(smoothed)
    abs_peak = pd.Series(abs_smoothed).rolling(window=norm_lookback, min_periods=1).max().values
    imbalance_strength = np.where(abs_peak > 0, abs_smoothed / abs_peak, 0.0)

    # --- Dynamic neutral zone ---
    vol_adj = np.where(close > 0, (atr_main / close) * 100, 0.0)
    neutral_width = base_neutral * (1 + vol_adj * neutral_vol_factor)

    # --- Regime classification (5 regimes, mutually exclusive) ---
    regime = np.full(n, "BALANCED", dtype=object)

    is_flat = (smoothed < neutral_width) & (smoothed > -neutral_width)
    is_strong_buy = ~is_flat & (smoothed > 0) & (imbalance_strength > strength_threshold)
    is_weak_buy = ~is_flat & (smoothed > 0) & (imbalance_strength <= strength_threshold)
    is_strong_sell = ~is_flat & (smoothed < 0) & (imbalance_strength > strength_threshold)
    is_weak_sell = ~is_flat & (smoothed < 0) & (imbalance_strength <= strength_threshold)

    regime[is_strong_buy] = "STRONG_BUY"
    regime[is_weak_buy] = "WEAK_BUY"
    regime[is_strong_sell] = "STRONG_SELL"
    regime[is_weak_sell] = "WEAK_SELL"

    return {
        "smoothed_imbalance": smoothed,
        "signal_line": signal_line,
        "imbalance_strength": imbalance_strength,
        "regime": regime,
        "neutral_width": neutral_width,
    }


def add_fs_balance_columns(
    df: pd.DataFrame, **kwargs
) -> pd.DataFrame:
    """
    Convenience wrapper: computes FS Balance and adds columns to df.

    Adds: fb_imbalance, fb_signal, fb_strength, fb_regime, fb_neutral

    All kwargs are forwarded to calc_fs_balance().
    Returns the modified DataFrame (copy).
    """
    df = df.copy()
    result = calc_fs_balance(df, **kwargs)
    df["fb_imbalance"] = result["smoothed_imbalance"]
    df["fb_signal"] = result["signal_line"]
    df["fb_strength"] = result["imbalance_strength"]
    df["fb_regime"] = result["regime"]
    df["fb_neutral"] = result["neutral_width"]
    return df
