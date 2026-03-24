"""
MYM-specific indicator enrichment — Phase 4 Renko indicators.

Adds five new indicator families on top of the standard renko
indicators and Phase 6 enrichment.  Designed specifically for
MYM (Micro E-mini Dow) futures on Renko charts.

New indicator families:
  1. Brick Velocity   — time-based momentum (unique to Renko)
  2. Streak Momentum  — enhanced streak analysis with acceleration
  3. Session Context  — intraday session positioning for futures
  4. Adaptive Regime  — unified trend/chop regime classifier
  5. Brick Exhaustion — trend exhaustion via velocity deceleration

All outputs are pre-shifted by .shift(1) following Pitfall #7.
Each indicator is wrapped in try/except — failures fill with NaN.

Columns added:
    brick_td          Seconds between consecutive bricks
    vel_sma           Rolling average brick time delta
    vel_ratio         Current / average velocity (< 1 = fast)
    vel_zscore        Z-score of brick velocity

    streak_len        Signed streak length (+ up, − down)
    streak_vel_avg    Average velocity within current streak
    streak_accel      Velocity acceleration within streak
    streak_age_pct    Streak length percentile vs history

    sess_range_used   Fraction of typical session range consumed
    sess_brick_count  Bricks formed in current session
    sess_dir_bias     Net directional bias within session (−1 to +1)
    sess_minutes_left Minutes until 15:45 ET forced close
    sess_is_opening   Boolean: within first 30 min of RTH
    sess_is_closing   Boolean: within last 30 min before close

    regime_score      Unified regime score (−1 choppy to +1 trending)
    regime_trending   Boolean: good for trend-following entries
    regime_class      Categorical: −1 CHOPPY / 0 NEUTRAL / +1 TRENDING

    exhaust_score     Trend exhaustion score (0 fresh to 1 exhausted)
    exhaust_alert     Boolean: exhaustion above threshold
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indicators.brick_velocity import calc_brick_velocity
from indicators.streak_momentum import calc_streak_momentum
from indicators.session_context import calc_session_context
from indicators.adaptive_regime import calc_adaptive_regime
from indicators.brick_exhaustion import calc_brick_exhaustion


def add_mym_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Phase 4 MYM-specific indicators to a Renko DataFrame.

    Requires:
        - Standard renko indicators already computed (add_renko_indicators)
        - Columns: brick_up, High, Low, Close, adx, chop, sq_momentum, sq_on

    All output columns are shifted by 1 bar (pre-shifted convention).

    Args:
        df: Renko DataFrame with standard + phase6 indicators.

    Returns:
        df with added columns (in-place).
    """

    # ── 1. Brick Velocity ─────────────────────────────────────────────────
    try:
        bv = calc_brick_velocity(df, lookback=20)
        for key in ["brick_td", "vel_sma", "vel_ratio", "vel_zscore"]:
            df[key] = pd.Series(bv[key], index=df.index).shift(1).values
    except Exception as e:
        print(f"  WARN: Brick Velocity failed: {e}")
        for key in ["brick_td", "vel_sma", "vel_ratio", "vel_zscore"]:
            df[key] = np.nan

    # ── 2. Streak Momentum ────────────────────────────────────────────────
    try:
        sm = calc_streak_momentum(df, history_window=200)
        for key in ["streak_len", "streak_vel_avg", "streak_accel", "streak_age_pct"]:
            df[key] = pd.Series(sm[key], index=df.index).shift(1).values
    except Exception as e:
        print(f"  WARN: Streak Momentum failed: {e}")
        for key in ["streak_len", "streak_vel_avg", "streak_accel", "streak_age_pct"]:
            df[key] = np.nan

    # ── 3. Session Context ────────────────────────────────────────────────
    try:
        sc = calc_session_context(df)
        for key in ["sess_range_used", "sess_brick_count", "sess_dir_bias",
                     "sess_minutes_left", "sess_is_opening", "sess_is_closing"]:
            df[key] = pd.Series(sc[key], index=df.index).shift(1).values
    except Exception as e:
        print(f"  WARN: Session Context failed: {e}")
        for key in ["sess_range_used", "sess_brick_count", "sess_dir_bias",
                     "sess_minutes_left", "sess_is_opening", "sess_is_closing"]:
            df[key] = np.nan

    # ── 4. Adaptive Regime ────────────────────────────────────────────────
    try:
        ar = calc_adaptive_regime(df)
        for key in ["regime_score", "regime_trending", "regime_class"]:
            df[key] = pd.Series(ar[key], index=df.index).shift(1).values
    except Exception as e:
        print(f"  WARN: Adaptive Regime failed: {e}")
        for key in ["regime_score", "regime_trending", "regime_class"]:
            df[key] = np.nan

    # ── 5. Brick Exhaustion ───────────────────────────────────────────────
    try:
        be = calc_brick_exhaustion(df)
        for key in ["exhaust_score", "exhaust_alert"]:
            df[key] = pd.Series(be[key], index=df.index).shift(1).values
    except Exception as e:
        print(f"  WARN: Brick Exhaustion failed: {e}")
        for key in ["exhaust_score", "exhaust_alert"]:
            df[key] = np.nan

    return df
