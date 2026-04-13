"""
R035: BTC Long-Only KAMA Ribbon 3L + MK v4 + TP/SL Bracket

BTC port of R033 (EURAUD KAMA + MK + bracket). Same mechanism, BTC conventions:
  - LONG ONLY (per OANDA BTC convention — no shorting)
  - Cash mode position sizing
  - First TV test (kama_ribbon_btc.pine, 412 trades) showed:
      Full PF=3.11, WR=56.6%, W/L=2.39, MaxDD=$1.86 (best KAMA-family on any
      instrument so far). 7 FRESH trades — too small to claim deploy.

Sweep dimensions (same shape as R033, but BTC-tuned):
  - tp_dist: dollar amounts (multiples of brick size)
  - sl_bricks: 1, 2, 3, 4
  - max_dist_bricks: 4, 8, 12, 1000(off)
  - mk_mode: off / direction / block_flat / strong

KAMA periods LOCKED at 5/13/60.
MK v4 internal params LOCKED at indicator defaults.

Per R20: BTC microstructure differs from EURAUD; cannot assume EURAUD Config #2
parameters transfer. The whole grid is swept fresh on BTC.
"""

import numpy as np
import pandas as pd

from indicators.kama import calc_kama
from indicators.momentum_king_v4 import calc_momentum_king_v4

DESCRIPTION = "BTC long-only 3-KAMA ribbon + MK v4 + distance filter + TP/SL bracket"

HYPOTHESIS = (
    "EURAUD R033 found Config #2 (KAMA 5/13/60 + MK block_flat + TP3b SL2b) "
    "TV-validated at FRESH PF=2.53. The first TV test of the same mechanism "
    "on BTC long-only showed Full PF=3.11 / W/L=2.39, the best KAMA-family "
    "result we've produced. This sweep tests whether the parameter surface "
    "supports a deploy candidate on BTC, and whether MK strict filtering "
    "(the dominant R033 finding) generalizes to BTC's microstructure."
)

# ── Locked params ─────────────────────────────────────────────────────────────
K1_LEN = 5
K2_LEN = 13
K3_LEN = 60
KAMA_FAST_SC = 2
KAMA_SLOW_SC = 30

# MK v4 at indicator defaults
MK_EMA_LEN          = 14
MK_SMOOTH_BASE      = 3
MK_MAX_SMOOTH       = 10
MK_ATR_LEN          = 14
MK_VOL_FACTOR       = 1.5
MK_NORM_LOOKBACK    = 200
MK_STRENGTH_THRESH  = 0.6
MK_NEUTRAL_ATR_PCT  = 0.3
MK_SIGNAL_LEN       = 9

BRICK_SIZE = 150.0  # BTC Renko 150

# ── Sweep grid ────────────────────────────────────────────────────────────────
PARAM_GRID = {
    # TP distance in dollars (multiples of 150 brick)
    "tp_dist":     [300, 450, 600, 900, 1200],   # 2-8 bricks
    # SL distance in bricks
    "sl_bricks":   [1, 2, 3, 4],
    # Distance filter: max bricks above K3 (1000 = effectively off)
    "max_dist_bricks": [4, 8, 12, 1000],
    # MK v4 mode
    "mk_mode":     ["off", "direction", "block_flat", "strong"],
}

_KAMA_CACHE = {}
_MK_CACHE   = {}


def _get_kama(close: pd.Series, length: int) -> np.ndarray:
    key = (length, KAMA_FAST_SC, KAMA_SLOW_SC)
    if key not in _KAMA_CACHE:
        _KAMA_CACHE[key] = calc_kama(
            close, length=length, fast=KAMA_FAST_SC, slow=KAMA_SLOW_SC
        ).shift(1).values
    return _KAMA_CACHE[key]


def _get_mk(df: pd.DataFrame) -> dict:
    key = "mk_v4_default"
    if key not in _MK_CACHE:
        result = calc_momentum_king_v4(
            df,
            ema_length=MK_EMA_LEN,
            smoothing_base=MK_SMOOTH_BASE,
            max_smooth=MK_MAX_SMOOTH,
            atr_length=MK_ATR_LEN,
            volatility_factor=MK_VOL_FACTOR,
            norm_lookback=MK_NORM_LOOKBACK,
            strength_threshold=MK_STRENGTH_THRESH,
            neutral_atr_pct=MK_NEUTRAL_ATR_PCT,
            signal_length=MK_SIGNAL_LEN,
        )
        smoothed = pd.Series(result["smoothed_momentum"]).shift(1).values
        regime   = pd.Series(result["regime"]).shift(1).values
        _MK_CACHE[key] = {"smoothed": smoothed, "regime": regime}
    return _MK_CACHE[key]


def generate_signals(
    df: pd.DataFrame,
    tp_dist: float = 450.0,
    sl_bricks: int = 2,
    max_dist_bricks: int = 1000,
    mk_mode: str = "block_flat",
) -> pd.DataFrame:
    """
    R035: BTC long-only KAMA + MK + bracket. Stateless. Long-only signals
    (no short_* columns since we use run_backtest, not run_backtest_long_short).
    """
    n = len(df)
    long_entry = np.zeros(n, dtype=bool)
    long_exit  = np.zeros(n, dtype=bool)

    if mk_mode not in ("off", "direction", "block_flat", "strong"):
        df = df.copy()
        df["long_entry"] = long_entry
        df["long_exit"]  = long_exit
        df["tp_offset"]  = 0.0
        df["sl_offset"]  = 0.0
        return df

    close = df["Close"]
    k1 = _get_kama(close, K1_LEN)
    k2 = _get_kama(close, K2_LEN)
    k3 = _get_kama(close, K3_LEN)

    any_nan    = np.isnan(k1) | np.isnan(k2) | np.isnan(k3)
    bull_align = (k1 > k2) & (k2 > k3) & ~any_nan

    bull_prev = np.roll(bull_align, 1); bull_prev[0] = False
    raw_long_entry = bull_align & ~bull_prev

    # Distance filter (longs only)
    close_arr   = close.values
    max_dist_pr = max_dist_bricks * BRICK_SIZE
    dist_above  = close_arr - k3
    long_dist_ok = dist_above <= max_dist_pr

    # MK filter
    if mk_mode == "off":
        mk_long_ok = np.ones(n, dtype=bool)
    else:
        mk = _get_mk(df)
        smoothed = mk["smoothed"]
        regime   = mk["regime"]
        mk_long_dir = ~np.isnan(smoothed) & (smoothed > 0)
        if mk_mode == "direction":
            mk_long_ok = mk_long_dir
        elif mk_mode == "block_flat":
            not_flat = regime != "FLAT"
            mk_long_ok = mk_long_dir & not_flat
        else:  # strong
            mk_long_ok = regime == "STRONG_UP"

    warmup = max(K3_LEN, MK_NORM_LOOKBACK) + 5
    mask = np.zeros(n, dtype=bool)
    mask[warmup:] = True

    long_entry = raw_long_entry & long_dist_ok & mk_long_ok & mask

    df = df.copy()
    df["long_entry"] = long_entry
    df["long_exit"]  = long_exit  # bracket handles exits
    df["tp_offset"]  = tp_dist
    df["sl_offset"]  = sl_bricks * BRICK_SIZE
    return df
