"""
R026: SuperTrend Flip — enter on SuperTrend direction change

Primary entry signal: SuperTrend(10, 3.0) direction flip.
  LONG:  st_dir flips from -1 → +1  AND  current brick is UP
  SHORT: st_dir flips from +1 → -1  AND  current brick is DOWN

This is structurally different from every other strategy in the codebase,
which all use R001/R002 (brick count/reversal) as the base trigger.
SuperTrend captures trend regime shifts that brick counting misses —
it can fire mid-streak when volatility compresses enough to flip the
adaptive bands.

Optional gates:
  - ADX: require trend strength before entering
  - Session: London+NY overlap only
  - Volume: skip volume spikes
  - Squeeze: only enter when NOT in squeeze (expanded volatility)
  - RSI: skip overbought longs / oversold shorts
  - MACD: histogram sign must match direction

Exit: first opposing brick (standard).

All indicators pre-shifted in add_renko_indicators() — safe at index i.
"""

import numpy as np
import pandas as pd

DESCRIPTION = "SuperTrend direction flip entry with optional gates"

HYPOTHESIS = (
    "R001/R002 entries are blind to volatility regime — they fire purely on "
    "brick count. SuperTrend incorporates ATR into its bands, so a flip "
    "represents a genuine volatility-adjusted trend change. This should "
    "produce fewer but higher-quality entries, especially in transitions "
    "between choppy and trending regimes."
)

PARAM_GRID = {
    "cooldown":      [3, 5, 10, 20],
    "require_brick": [True, False],    # require brick dir to match ST flip
    "adx_gate":      [0, 25],
    "session_start": [0, 13],
    "vol_max":       [0, 1.5],
    "sq_filter":     [False, True],    # only enter when squeeze is OFF
    "rsi_gate":      [0, 70],
    "macd_gate":     [False, True],
}


def generate_signals(
    df: pd.DataFrame,
    cooldown: int = 5,
    require_brick: bool = True,
    adx_gate: int = 0,
    session_start: int = 0,
    vol_max: float = 0,
    sq_filter: bool = False,
    rsi_gate: int = 0,
    macd_gate: bool = False,
) -> pd.DataFrame:
    n = len(df)

    brick_up  = df["brick_up"].values
    st_dir    = df["st_dir"].values if "st_dir" in df.columns else np.ones(n)
    adx       = df["adx"].values if "adx" in df.columns else np.full(n, np.nan)
    vol_ratio = df["vol_ratio"].values if "vol_ratio" in df.columns else np.zeros(n)
    sq_on     = df["sq_on"].values if "sq_on" in df.columns else np.zeros(n, dtype=bool)
    rsi       = df["rsi"].values if "rsi" in df.columns else np.full(n, np.nan)
    macd_hist = df["macd_hist"].values if "macd_hist" in df.columns else np.full(n, np.nan)
    hours     = df.index.hour

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    in_position    = False
    trade_dir      = 0
    last_trade_bar = -999_999
    warmup         = 200

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # ── Exit: opposing brick ──
        if in_position:
            is_opp = (trade_dir == 1 and not up) or (trade_dir == -1 and up)
            long_exit[i]  = is_opp and trade_dir == 1
            short_exit[i] = is_opp and trade_dir == -1
            if is_opp:
                in_position = False
                trade_dir = 0

        if in_position:
            continue

        # ── Cooldown ──
        if (i - last_trade_bar) < cooldown:
            continue

        # ── Primary trigger: SuperTrend flip ──
        cur_st  = st_dir[i]
        prev_st = st_dir[i - 1]
        if np.isnan(cur_st) or np.isnan(prev_st):
            continue

        long_flip  = (prev_st < 0 and cur_st > 0)   # bearish → bullish
        short_flip = (prev_st > 0 and cur_st < 0)    # bullish → bearish

        if not long_flip and not short_flip:
            continue

        # ── Optionally require brick direction to match ──
        if require_brick:
            if long_flip and not up:
                continue
            if short_flip and up:
                continue

        is_long = long_flip

        # ── Gate stack ──
        if adx_gate > 0 and not np.isnan(adx[i]) and adx[i] < adx_gate:
            continue
        if session_start > 0 and hours[i] < session_start:
            continue
        if vol_max > 0 and not np.isnan(vol_ratio[i]) and vol_ratio[i] > vol_max:
            continue
        if sq_filter and not np.isnan(sq_on[i]) and bool(sq_on[i]):
            continue  # skip if in squeeze (want expanded vol)
        if rsi_gate > 0 and not np.isnan(rsi[i]):
            if is_long and rsi[i] > rsi_gate:
                continue
            if not is_long and rsi[i] < (100 - rsi_gate):
                continue
        if macd_gate and not np.isnan(macd_hist[i]):
            if is_long and macd_hist[i] < 0:
                continue
            if not is_long and macd_hist[i] > 0:
                continue

        # ── Fire entry ──
        if is_long:
            long_entry[i] = True
        else:
            short_entry[i] = True
        in_position = True
        trade_dir = 1 if is_long else -1
        last_trade_bar = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
