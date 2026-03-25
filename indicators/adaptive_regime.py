"""
Adaptive Regime Indicator — Unified trend/chop regime classifier.

Combines ADX (trend strength), Choppiness Index (trend quality),
and Squeeze Momentum (volatility compression) into a single
regime score.  Designed to answer: "Is NOW a good time for
trend-following entries?"

Columns produced:
    regime_score     −1.0 (choppy/compressed) to +1.0 (strong clean trend)
    regime_trending  Boolean: regime_score > threshold (default 0.3)
    regime_class     Categorical: -1 CHOPPY, 0 NEUTRAL, +1 TRENDING

All outputs are RAW (caller is responsible for .shift(1) if needed).
"""

import numpy as np
import pandas as pd


def calc_adaptive_regime(
    df: pd.DataFrame,
    adx_bull: float = 25.0,
    adx_strong: float = 40.0,
    chop_trend: float = 38.2,
    chop_chop: float = 61.8,
    trend_threshold: float = 0.3,
) -> dict:
    """
    Compute unified regime score from pre-computed indicators.

    Expects these columns already present (from add_renko_indicators):
        adx, chop, sq_momentum, sq_on

    Args:
        df: DataFrame with pre-computed indicator columns.
        adx_bull: ADX level where trend starts (default 25).
        adx_strong: ADX level for strong trend (default 40).
        chop_trend: Choppiness below this = trending (default 38.2).
        chop_chop: Choppiness above this = choppy (default 61.8).
        trend_threshold: regime_score cutoff for regime_trending flag.

    Returns:
        dict with keys: regime_score, regime_trending, regime_class
    """
    n = len(df)

    # --- ADX component: 0 (weak) to 1 (strong trend) ---
    adx = df["adx"].values.astype(np.float64)
    adx_score = np.clip((adx - adx_bull) / (adx_strong - adx_bull), 0, 1)
    adx_score = np.where(np.isnan(adx), 0.0, adx_score)

    # --- Choppiness component: -1 (choppy) to +1 (trending) ---
    chop = df["chop"].values.astype(np.float64)
    chop_score = np.full(n, 0.0, dtype=np.float64)
    # Below chop_trend → trending (+1)
    trending_mask = chop < chop_trend
    # Above chop_chop → choppy (-1)
    choppy_mask = chop > chop_chop
    # Between → linear interpolation
    mid_mask = ~trending_mask & ~choppy_mask & ~np.isnan(chop)

    chop_score[trending_mask] = 1.0
    chop_score[choppy_mask] = -1.0
    if np.any(mid_mask):
        chop_score[mid_mask] = 1.0 - 2.0 * (chop[mid_mask] - chop_trend) / (chop_chop - chop_trend)

    # --- Squeeze component: -0.5 (compressed) or +0.5 (released) ---
    sq_on = df["sq_on"].values if "sq_on" in df.columns else np.zeros(n, dtype=bool)
    sq_momentum = df["sq_momentum"].values if "sq_momentum" in df.columns else np.zeros(n)
    sq_score = np.where(sq_on, -0.5, 0.0)
    # Bonus for strong momentum after squeeze release
    sq_released = (~sq_on.astype(bool)) & (np.abs(sq_momentum) > 0)
    sq_score = np.where(sq_released, 0.5, sq_score)
    sq_score = np.where(np.isnan(sq_momentum), 0.0, sq_score)

    # --- Combined regime score ---
    # Weights: ADX 40%, Chop 40%, Squeeze 20%
    regime_score = 0.4 * adx_score + 0.4 * chop_score + 0.2 * sq_score
    regime_score = np.clip(regime_score, -1.0, 1.0)

    # --- Derived flags ---
    regime_trending = regime_score > trend_threshold
    regime_class = np.zeros(n, dtype=np.float64)
    regime_class[regime_score > trend_threshold] = 1.0
    regime_class[regime_score < -trend_threshold] = -1.0

    return {
        "regime_score":    regime_score,
        "regime_trending": regime_trending.astype(np.float64),
        "regime_class":    regime_class,
    }
