"""
R027: Volatility Squeeze Release — enter when low-volatility consolidation breaks

Standard TTM Squeeze (BB inside KC) almost never fires on Renko because
fixed-size bricks keep BB artificially wide. Instead, we use BB bandwidth
percentile as the consolidation detector:

  1. Compute rolling percentile rank of bb_bw over lookback window
  2. "Squeeze" = bb_bw is below the Nth percentile (low vol = coiling)
  3. "Release" = bb_bw exits the low-vol zone (crosses back above threshold)

Direction from MACD histogram sign at release:
  LONG:  low_vol was True → now False  AND  macd_hist > 0  AND  brick UP
  SHORT: low_vol was True → now False  AND  macd_hist < 0  AND  brick DOWN

Note: sq_momentum (LazyBear TTM) is unusable on Renko — its linear
regression basis makes it permanently negative on small brick sizes.
MACD histogram provides a balanced directional signal.

This catches explosive moves after volatility compression.
Structurally different from R001/R002 (brick patterns) and R026 (ST flip).

Optional gates: ADX, session, volume, SuperTrend, RSI, MACD.
Exit: first opposing brick (standard).
All indicators pre-shifted — safe at index i.
"""

import numpy as np
import pandas as pd

DESCRIPTION = "BB bandwidth squeeze release with momentum direction entry"

HYPOTHESIS = (
    "Renko bricks are fixed-size so BB bandwidth measures relative compression "
    "in brick-space. When BB_BW drops below its Nth percentile, the market is "
    "coiling. The breakout (BW rising above threshold) aligns with the start "
    "of a directional streak. Momentum direction at release predicts the move. "
    "This is orthogonal to brick-counting and trend-following entries."
)

PARAM_GRID = {
    "cooldown":        [3, 5, 10, 20],
    "bw_percentile":   [10, 20],         # BB_BW below this pctile = squeeze
    "bw_lookback":     [100, 200],       # rolling window for percentile calc
    "min_squeeze_len": [3, 5, 10],       # min consecutive low-vol bars
    "require_brick":   [True, False],
    "adx_gate":        [0, 25],
    "session_start":   [0, 13],
    "vol_max":         [0, 1.5],
    "st_gate":         [False, True],
    "rsi_gate":        [0, 70],
    "macd_gate":       [False, True],
}


def generate_signals(
    df: pd.DataFrame,
    cooldown: int = 5,
    bw_percentile: int = 20,
    bw_lookback: int = 200,
    min_squeeze_len: int = 3,
    require_brick: bool = True,
    adx_gate: int = 0,
    session_start: int = 0,
    vol_max: float = 0,
    st_gate: bool = False,
    rsi_gate: int = 0,
    macd_gate: bool = False,
) -> pd.DataFrame:
    n = len(df)

    brick_up  = df["brick_up"].values
    bb_bw     = df["bb_bw"].values if "bb_bw" in df.columns else np.full(n, np.nan)
    adx       = df["adx"].values if "adx" in df.columns else np.full(n, np.nan)
    vol_ratio = df["vol_ratio"].values if "vol_ratio" in df.columns else np.zeros(n)
    st_dir    = df["st_dir"].values if "st_dir" in df.columns else np.ones(n)
    rsi       = df["rsi"].values if "rsi" in df.columns else np.full(n, np.nan)
    macd_hist = df["macd_hist"].values if "macd_hist" in df.columns else np.full(n, np.nan)
    hours     = df.index.hour

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    # Pre-compute rolling percentile threshold for bb_bw
    # low_vol[i] = True when bb_bw[i] < rolling Nth percentile over lookback
    low_vol = np.zeros(n, dtype=bool)
    for i in range(bw_lookback, n):
        window = bb_bw[i - bw_lookback : i]
        valid = window[~np.isnan(window)]
        if len(valid) >= 20:
            threshold = np.percentile(valid, bw_percentile)
            if not np.isnan(bb_bw[i]):
                low_vol[i] = bb_bw[i] < threshold

    # Pre-compute consecutive low-vol length
    lv_len = np.zeros(n, dtype=int)
    for i in range(1, n):
        if low_vol[i]:
            lv_len[i] = lv_len[i - 1] + 1

    in_position    = False
    trade_dir      = 0
    last_trade_bar = -999_999
    warmup         = max(bw_lookback + 10, 200)

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

        # ── Primary trigger: low-vol zone exit ──
        if low_vol[i] or not low_vol[i - 1]:
            continue  # need: was in low-vol, now exited

        # Check minimum squeeze duration
        if lv_len[i - 1] < min_squeeze_len:
            continue

        # Direction from MACD histogram (sq_momentum is broken on Renko)
        mh = macd_hist[i]
        if np.isnan(mh) or mh == 0:
            continue

        is_long = (mh > 0)

        # ── Optionally require brick direction to match ──
        if require_brick:
            if is_long and not up:
                continue
            if not is_long and up:
                continue

        # ── Gate stack ──
        if adx_gate > 0 and not np.isnan(adx[i]) and adx[i] < adx_gate:
            continue
        if session_start > 0 and hours[i] < session_start:
            continue
        if vol_max > 0 and not np.isnan(vol_ratio[i]) and vol_ratio[i] > vol_max:
            continue
        if st_gate and not np.isnan(st_dir[i]):
            if is_long and st_dir[i] < 0:
                continue
            if not is_long and st_dir[i] > 0:
                continue
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
