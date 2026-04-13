"""
R031: KAMA Ribbon + Trend-Regime Gate

Follow-up to R030 on EURAUD 0.0006. R030's postmortem showed:
  - Winners hold avg 19 bricks, losers hold avg 9 bricks (2x asymmetry)
  - Gray exit is NOT the problem (winners get plenty of room)
  - Problem is ENTRY: too many alignments fire during chop regimes

Hypothesis: gating entries with a trend-regime filter should kill the
chop-alignment losers while keeping the real-trend winners.

Ribbon is LOCKED to the best R030 finalist on EURAUD 0.0006: fast=5, mid=13,
slow=30. This sweep varies ONLY the regime gate so results isolate the gate
contribution.

Gates tested (one at a time — low-overlap R6 principle):
  - "none"   : baseline, should reproduce R030 config #3
  - "adx"    : adx[i] >= adx_min (classic trend-strength)
  - "chop"   : chop[i] <= chop_max (low-choppiness = trending)
  - "bb_bw"  : bb_bw[i] >= bb_bw_min (volatility expansion)
  - "streak" : N consecutive same-dir bricks before the alignment flip
               (R3: pure brick streak is a strong baseline — combine with ribbon)

Each gate is tested at 3 threshold values. 13 combos total.
"""

import numpy as np
import pandas as pd

from indicators.kama import calc_kama

DESCRIPTION = "KAMA ribbon 5/13/30 + single trend-regime gate (ADX / CHOP / BB_BW / STREAK)"

HYPOTHESIS = (
    "R030 showed 2x winner/loser hold asymmetry — the exit is fine, the "
    "entry is too permissive. A single low-overlap trend-regime gate should "
    "filter chop-alignments (the losers) while preserving real-trend "
    "alignments (the winners). Test 4 gate families one at a time, 3 "
    "thresholds each, on the locked best ribbon from R030 (5/13/30)."
)

# ── Locked ribbon params (from R030 best HOLDOUT finalist) ────────────────────
FAST_LEN = 5
MID_LEN  = 13
SLOW_LEN = 30
FAST_SC  = 2
SLOW_SC  = 30

# ── Sweep grid: (regime_mode, threshold) ──────────────────────────────────────
# Represented as two params so the runner's product grid works cleanly. Every
# combination of (mode, threshold) is a valid run, but the strategy only reads
# the threshold that matches the active mode. Duplicate (mode, threshold) combos
# that don't match are short-circuited to empty signals at run time so they
# produce no trades and get filtered out downstream.

PARAM_GRID = {
    "regime": ["none", "adx", "chop", "bb_bw", "streak"],
    "thresh": [0, 1, 2],   # index into THRESH_TABLE per regime
}

THRESH_TABLE = {
    "none":   [0.0, 0.0, 0.0],          # baseline — threshold ignored
    "adx":    [20.0, 25.0, 30.0],
    "chop":   [55.0, 45.0, 38.2],       # lower = more trending
    "bb_bw":  [0.001, 0.002, 0.003],    # higher = more volatile
    "streak": [3.0, 5.0, 8.0],
}

# Module-level cache
_KAMA_CACHE = {}


def _get_kama(close_series: pd.Series, length: int, fast: int, slow: int) -> np.ndarray:
    key = (length, fast, slow)
    if key not in _KAMA_CACHE:
        _KAMA_CACHE[key] = calc_kama(close_series, length=length, fast=fast, slow=slow).shift(1).values
    return _KAMA_CACHE[key]


def generate_signals(
    df: pd.DataFrame,
    regime: str = "none",
    thresh: int = 0,
) -> pd.DataFrame:
    """
    R031 — locked-ribbon + single regime gate. Stateless (engine manages pos).
    """
    n = len(df)
    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    if regime not in THRESH_TABLE:
        df["long_entry"]  = long_entry
        df["long_exit"]   = long_exit
        df["short_entry"] = short_entry
        df["short_exit"]  = short_exit
        return df

    threshold = THRESH_TABLE[regime][thresh]

    close = df["Close"]
    k_fast = _get_kama(close, FAST_LEN, FAST_SC, SLOW_SC)
    k_mid  = _get_kama(close, MID_LEN,  FAST_SC, SLOW_SC)
    k_slow = _get_kama(close, SLOW_LEN, FAST_SC, SLOW_SC)

    any_nan    = np.isnan(k_fast) | np.isnan(k_mid) | np.isnan(k_slow)
    bull_align = (k_fast > k_mid) & (k_mid > k_slow) & ~any_nan
    bear_align = (k_fast < k_mid) & (k_mid < k_slow) & ~any_nan

    bull_prev = np.roll(bull_align, 1); bull_prev[0] = False
    bear_prev = np.roll(bear_align, 1); bear_prev[0] = False

    raw_long_entry  = bull_align & ~bull_prev
    raw_short_entry = bear_align & ~bear_prev

    # ── Regime gate ──────────────────────────────────────────────────────────
    if regime == "none":
        gate_long  = np.ones(n, dtype=bool)
        gate_short = np.ones(n, dtype=bool)
    elif regime == "adx":
        adx = df["adx"].values
        gate = (~np.isnan(adx)) & (adx >= threshold)
        gate_long = gate_short = gate
    elif regime == "chop":
        chop = df["chop"].values
        gate = (~np.isnan(chop)) & (chop <= threshold)
        gate_long = gate_short = gate
    elif regime == "bb_bw":
        bb_bw = df["bb_bw"].values
        gate = (~np.isnan(bb_bw)) & (bb_bw >= threshold)
        gate_long = gate_short = gate
    elif regime == "streak":
        # Count consecutive same-direction bricks ending at [i-1] (pre-shifted
        # by construction since we look at i-1 not i). Causal, no look-ahead.
        brick_up = df["brick_up"].values
        up_streak   = np.zeros(n, dtype=int)
        down_streak = np.zeros(n, dtype=int)
        u = 0
        d = 0
        for i in range(n):
            if brick_up[i]:
                u += 1; d = 0
            else:
                d += 1; u = 0
            up_streak[i]   = u
            down_streak[i] = d
        # Gate at bar i reads the streak through bar i-1
        up_prev   = np.roll(up_streak,   1); up_prev[0]   = 0
        down_prev = np.roll(down_streak, 1); down_prev[0] = 0
        gate_long  = up_prev   >= threshold
        gate_short = down_prev >= threshold
    else:
        gate_long = gate_short = np.zeros(n, dtype=bool)

    warmup = SLOW_LEN + 5
    mask = np.zeros(n, dtype=bool)
    mask[warmup:] = True

    long_entry  = raw_long_entry  & gate_long  & mask
    short_entry = raw_short_entry & gate_short & mask
    long_exit   = (~bull_align) & mask
    short_exit  = (~bear_align) & mask

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
