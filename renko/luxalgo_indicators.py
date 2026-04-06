"""
LuxAlgo indicator enrichment for Renko DataFrames.

Adds LuxAlgo indicator columns as pre-shifted values (same convention as
renko/indicators.py — value at row i = computed through bar i-1).

Separated from main indicators.py because:
  1. KNN Supertrend is computationally expensive (O(n * window_size))
  2. These are experimental indicators being evaluated for signal value
  3. Strategies opt-in by calling add_luxalgo_indicators()

Columns added:
    lux_inertial_k      Inertial Stochastic %K (0-100)
    lux_inertial_d      Inertial Stochastic %D (0-100)
    lux_rollseg         Rolling Segment price level
    lux_rollseg_trend   Rolling Segment direction +1/-1
    lux_rollseg_bull_rev Rolling Segment bullish reversal
    lux_rollseg_bear_rev Rolling Segment bearish reversal
    lux_breakout_bull   Probabilistic breakout bull probability (0-100)
    lux_breakout_bear   Probabilistic breakout bear probability (0-100)
    lux_breakout_squeeze Volatility squeeze intensity (0-100)
    lux_svm_trend       Structural SVM market structure trend +1/-1/0
    lux_svm_break_type  Break type: 0=none, 1=BOS, 2=CHoCH
    lux_svm_break_bull  True if break is bullish
    lux_svm_score       SVM quality score 0-100 on break bars
    lux_streak_len      Consecutive streak length
    lux_streak_bull     True if streak is bullish
    lux_streak_cont     Continuation probability (0-100)
    lux_streak_rev      Reversal probability (0-100)

Optional (if include_knn=True):
    lux_knn_bullish     KNN ML trend direction (bool)
    lux_knn_prob        KNN smoothed probability (0-100)
    lux_knn_st_val      KNN Supertrend value (price)
    lux_knn_st_dir      KNN Supertrend direction +1/-1

Usage:
    from renko.luxalgo_indicators import add_luxalgo_indicators
    df = add_luxalgo_indicators(df, include_knn=False)
"""

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indicators.LuxAlgo.inertial_stochastic import calc_inertial_stochastic
from indicators.LuxAlgo.rolling_segment import calc_rolling_segment
from indicators.LuxAlgo.prob_breakout import calc_prob_breakout
from indicators.LuxAlgo.structural_svm import calc_structural_svm
from indicators.LuxAlgo.knn_supertrend import calc_knn_supertrend
from indicators.LuxAlgo.candle_streak import calc_candle_streak


def add_luxalgo_indicators(
    df: pd.DataFrame,
    include_knn: bool = False,
    svm_vol_weight: float = 0.0,
) -> pd.DataFrame:
    """
    Compute and attach LuxAlgo indicators to a Renko DataFrame.

    All outputs are shifted by 1 bar (pre-shifted convention).

    Args:
        df: Renko DataFrame with OHLCV + brick_up columns.
        include_knn: If True, compute KNN Supertrend (slow, ~10s for 20k bars).
        svm_vol_weight: Volume weight for SVM. Set to 0 for BTC (no volume data).

    Returns:
        df with LuxAlgo indicator columns added in-place.
    """
    idx = df.index

    # ── Inertial Stochastic (adaptive-length oscillator) ─────────────────────
    inertial = calc_inertial_stochastic(df, min_len=10, max_len=40, smooth_k=3, smooth_d=3)
    df["lux_inertial_k"] = pd.Series(inertial["k"], index=idx).shift(1).values
    df["lux_inertial_d"] = pd.Series(inertial["d"], index=idx).shift(1).values

    # ── Rolling Segment (linear trend overlay) ───────────────────────────────
    rollseg = calc_rolling_segment(df)
    df["lux_rollseg"]          = pd.Series(rollseg["roll_seg"], index=idx).shift(1).values
    df["lux_rollseg_trend"]    = pd.Series(rollseg["trend"], index=idx).shift(1).values
    df["lux_rollseg_bull_rev"] = pd.Series(rollseg["bull_reversal"], index=idx).shift(1).values
    df["lux_rollseg_bear_rev"] = pd.Series(rollseg["bear_reversal"], index=idx).shift(1).values

    # ── Probabilistic Breakout Forecaster ────────────────────────────────────
    prob = calc_prob_breakout(df, length=20, horizon=10, vol_lookback=50)
    df["lux_breakout_bull"]    = pd.Series(prob["bull_prob"], index=idx).shift(1).values
    df["lux_breakout_bear"]    = pd.Series(prob["bear_prob"], index=idx).shift(1).values
    df["lux_breakout_squeeze"] = pd.Series(prob["squeeze"], index=idx).shift(1).values

    # ── Structural SVM Ranker ────────────────────────────────────────────────
    rsi_w = 0.3 + svm_vol_weight * 0.5 if svm_vol_weight == 0 else 0.3
    dist_w = 0.3 + svm_vol_weight * 0.5 if svm_vol_weight == 0 else 0.3
    svm = calc_structural_svm(
        df, pivot_len=5,
        vol_weight=svm_vol_weight,
        rsi_weight=rsi_w,
        dist_weight=dist_w,
        atr_len=14,
    )
    df["lux_svm_trend"]      = pd.Series(svm["trend"], index=idx).shift(1).values
    df["lux_svm_break_type"] = pd.Series(svm["break_type"], index=idx).shift(1).values
    df["lux_svm_break_bull"] = pd.Series(svm["break_bull"], index=idx).shift(1).values
    df["lux_svm_score"]      = pd.Series(svm["svm_score"], index=idx).shift(1).values

    # ── Consecutive Candle Streak Analysis ───────────────────────────────────
    streak = calc_candle_streak(df, lookback=5000)
    df["lux_streak_len"]  = pd.Series(streak["streak_len"], index=idx).shift(1).values
    df["lux_streak_bull"] = pd.Series(streak["streak_bull"], index=idx).shift(1).values
    df["lux_streak_cont"] = pd.Series(streak["continue_prob"], index=idx).shift(1).values
    df["lux_streak_rev"]  = pd.Series(streak["reversal_prob"], index=idx).shift(1).values

    # ── KNN Supertrend Horizon (optional, expensive) ─────────────────────────
    if include_knn:
        knn = calc_knn_supertrend(df)
        df["lux_knn_bullish"] = pd.Series(knn["ml_bullish"], index=idx).shift(1).values
        df["lux_knn_prob"]    = pd.Series(knn["smoothed_prob"], index=idx).shift(1).values
        df["lux_knn_st_val"]  = pd.Series(knn["st_val"], index=idx).shift(1).values
        df["lux_knn_st_dir"]  = pd.Series(knn["st_dir"], index=idx).shift(1).values

    return df
