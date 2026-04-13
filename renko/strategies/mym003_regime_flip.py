"""
MYM003: Regime Flip — ER Regime + Brick Direction + Streak Filter (MYM Renko 30)

Cross-instrument carryover of GJ001 to MYM brick 30. Per R20, re-optimize
on MYM data — do NOT import GBPJPY params.

Entry: ER regime flips from RANGE → TREND + brick direction.
       Optional min_streak filter: require N consecutive same-direction bricks
       at the flip bar to confirm direction commitment (R6: ER = volatility
       regime signal, brick_streak = momentum signal — structurally disjoint).
Exit:  First opposing brick (R1).

MYM conventions (R24):
  - qty_value=0.5 (workaround for $0.50/point multiplier → correct live P&L)
  - commission_pct=0.004 (≈$1.88 RT on ~$47K notional, close to $1.90 actual)
  - initial_capital=10000
"""

import numpy as np
import pandas as pd

DESCRIPTION = "ER regime flip + brick direction + streak filter, MYM Renko 30"

HYPOTHESIS = (
    "ER regime-flip catches early trend entries but fires on some false starts "
    "where direction hasn't committed. Adding a min brick streak at the flip "
    "bar filters out single-brick direction signals. ER (volatility regime) and "
    "brick streak (momentum) are structurally disjoint per R6 — low overlap "
    "should improve quality without crushing trade count."
)

BRICK_SIZE = 30  # MYM brick 30

# MYM-specific engine config (R24 workaround)
COMMISSION_PCT = 0.004
INITIAL_CAPITAL = 10000.0

PARAM_GRID = {
    "er_lookback":   [6, 8, 10, 14, 20],
    "er_smooth":     [1, 3, 5],
    "trend_thresh":  [0.35, 0.45, 0.55],
    "range_thresh":  [0.15, 0.25],
    "persistence":   [0, 2],
    "tp_bricks":     [0, 3, 5, 8],
}

# ── Caches ───────────────────────────────────────────────────────────────────
_ER_CACHE = {}


def _calc_er(close: np.ndarray, lookback: int, smooth: int) -> np.ndarray:
    """Compute Efficiency Ratio, optionally EMA-smoothed."""
    key = (lookback, smooth)
    if key in _ER_CACHE:
        return _ER_CACHE[key]

    n = len(close)
    raw_er = np.full(n, np.nan)

    for i in range(lookback, n):
        direction = abs(close[i] - close[i - lookback])
        volatility = 0.0
        for j in range(i - lookback + 1, i + 1):
            volatility += abs(close[j] - close[j - 1])
        raw_er[i] = direction / volatility if volatility != 0 else 0.0

    if smooth <= 1:
        er = raw_er
    else:
        er = np.full(n, np.nan)
        k = 2.0 / (smooth + 1)
        first_valid = lookback
        er[first_valid] = raw_er[first_valid]
        for i in range(first_valid + 1, n):
            if np.isnan(raw_er[i]):
                continue
            prev = er[i - 1]
            if np.isnan(prev):
                er[i] = raw_er[i]
            else:
                er[i] = raw_er[i] * k + prev * (1 - k)

    # Pre-shift by 1
    result = np.roll(er, 1)
    result[0] = np.nan

    _ER_CACHE[key] = result
    return result


def generate_signals(
    df: pd.DataFrame,
    er_lookback: int = 10,
    er_smooth: int = 5,
    trend_thresh: float = 0.45,
    range_thresh: float = 0.25,
    persistence: int = 0,
    tp_bricks: int = 0,
    min_streak: int = 1,
    reentry_streak: int = 0,
    reentry_cooldown: int = 0,
    max_reentries_per_regime: int = 999,
    min_reentry_er: float = 0.0,
    dir_filter: str = "none",
    dir_ema_len: int = 50,
    net_brick_lookback: int = 10,
) -> pd.DataFrame:
    """
    MYM003: ER regime flip + brick direction + re-entry within regime.

    Entry logic:
      1. Initial entry on RANGE→TREND flip (brick direction sets bias for
         the entire trending regime).
      2. Re-entry: while regime is TREND and flat, enter in the LOCKED bias
         direction after `reentry_streak` consecutive same-direction bricks
         and `reentry_cooldown` bars have passed since last exit.
      3. Bias resets when regime leaves TREND (back to RANGE or neutral).

    reentry_streak=0 disables re-entry (current behavior = baseline).

    Direction filters (dir_filter):
      "none"       — no filter (baseline)
      "kama"       — kama_slope sign must agree with entry direction
      "ema"        — close vs ema (dir_ema_len) must agree
      "di"         — plus_di vs minus_di must agree
      "macd"       — macd_hist sign must agree
      "st"         — supertrend direction must agree
      "net_bricks" — net brick count over net_brick_lookback must agree
    """
    n = len(df)
    close = df["Close"].values.astype(float)
    brick_up = df["brick_up"].values

    er = _calc_er(close, er_lookback, er_smooth)

    # ── Direction filter arrays (pre-shifted by indicators.py) ────────
    dir_long = np.ones(n, dtype=bool)   # default: all pass
    dir_short = np.ones(n, dtype=bool)

    if dir_filter == "kama" and "kama_slope" in df.columns:
        ks = df["kama_slope"].values
        dir_long  = ks > 0
        dir_short = ks < 0
    elif dir_filter == "ema":
        ema_col = f"ema{dir_ema_len}"
        if ema_col in df.columns:
            ema_vals = df[ema_col].values
            dir_long  = close > ema_vals
            dir_short = close < ema_vals
    elif dir_filter == "di" and "plus_di" in df.columns:
        pdi = df["plus_di"].values
        mdi = df["minus_di"].values
        dir_long  = pdi > mdi
        dir_short = mdi > pdi
    elif dir_filter == "macd" and "macd_hist" in df.columns:
        mh = df["macd_hist"].values
        dir_long  = mh > 0
        dir_short = mh < 0
    elif dir_filter == "st" and "st_dir" in df.columns:
        st = df["st_dir"].values
        dir_long  = st > 0
        dir_short = st < 0
    elif dir_filter == "net_bricks":
        # Net brick count: +1 for up brick, -1 for down, sum over lookback
        brick_sign = np.where(brick_up, 1, -1).astype(float)
        # Shift by 1 to avoid look-ahead (value at i uses bars through i-1)
        net = np.full(n, 0.0)
        for i in range(net_brick_lookback + 1, n):
            net[i] = brick_sign[i - net_brick_lookback:i].sum()
        dir_long  = net > 0
        dir_short = net < 0

    # Pre-compute running brick streak (consecutive same-direction bricks)
    streak = np.zeros(n, dtype=int)
    streak[0] = 1
    for i in range(1, n):
        if brick_up[i] == brick_up[i - 1]:
            streak[i] = streak[i - 1] + 1
        else:
            streak[i] = 1

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    regime = 0
    trend_count = 0
    range_count = 0
    prev_trending = False

    # Regime bias + position state (for re-entry logic)
    bias = 0            # +1 = long bias, -1 = short bias, 0 = no bias
    pos = 0             # +1 = in long, -1 = in short, 0 = flat
    last_exit_bar = -999_999
    reentries_this_regime = 0  # counter, reset when regime leaves TREND

    warmup = er_lookback + er_smooth + 5

    for i in range(warmup, n):
        if np.isnan(er[i]):
            continue

        raw_trending = er[i] >= trend_thresh
        raw_ranging  = er[i] <= range_thresh

        if raw_trending:
            trend_count += 1
            range_count = 0
        elif raw_ranging:
            range_count += 1
            trend_count = 0
        else:
            trend_count = 0
            range_count = 0

        if trend_count > persistence:
            regime = 1
        elif range_count > persistence:
            regime = -1

        is_trending = regime == 1

        # Reset bias AND re-entry counter when regime leaves TREND
        if not is_trending:
            bias = 0
            reentries_this_regime = 0

        # R1 exit: first opposing brick (unconditional; engine only acts if in pos)
        long_exit[i]  = not brick_up[i]
        short_exit[i] = brick_up[i]

        # Update internal position tracking to match engine's R1 exit behavior
        if pos == 1 and not brick_up[i]:
            pos = 0
            last_exit_bar = i
        elif pos == -1 and brick_up[i]:
            pos = 0
            last_exit_bar = i

        # ── Entry: initial flip OR re-entry within regime ──────────────
        regime_flip_to_trend = is_trending and not prev_trending

        if regime_flip_to_trend and streak[i] >= min_streak and pos == 0:
            if brick_up[i] and dir_long[i]:
                long_entry[i] = True
                pos = 1
                bias = 1
            elif not brick_up[i] and dir_short[i]:
                short_entry[i] = True
                pos = -1
                bias = -1

        elif (reentry_streak > 0 and is_trending and pos == 0 and bias != 0
              and reentries_this_regime < max_reentries_per_regime
              and (i - last_exit_bar) >= reentry_cooldown
              and er[i] >= min_reentry_er):
            # Re-entry within same regime, in locked bias direction
            # min_reentry_er gates: only re-enter when ER is still strong
            if bias == 1 and brick_up[i] and streak[i] >= reentry_streak:
                long_entry[i] = True
                pos = 1
                reentries_this_regime += 1
            elif bias == -1 and not brick_up[i] and streak[i] >= reentry_streak:
                short_entry[i] = True
                pos = -1
                reentries_this_regime += 1

        prev_trending = is_trending

    df = df.copy()
    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit

    if tp_bricks > 0:
        df["tp_offset"] = tp_bricks * BRICK_SIZE
    else:
        df["tp_offset"] = 0.0
    df["sl_offset"] = 1 * BRICK_SIZE

    return df
