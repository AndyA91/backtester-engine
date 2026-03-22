"""EA014: High-Confluence Alpha Sniper — Multi-L3 Simultaneous Confluence

Ultra-selective institutional strategy that only fires when 2 or 3 independent
L3 "market microstructure" events peak at the same brick:

  Signal 1 — Stacked Imbalance  : ≥ min_stacked consecutive buy/sell-imbalance
             bricks (bc_l3_volume_imbalance_pro → vi_stacked_buy/sell)
  Signal 2 — Absorption         : bearish candle + buy delta at POC, or bullish
             candle + sell delta at POC  (bc_l3_footprint_fusion_pro → ff_buy/sell_abs)
  Signal 3 — POC Migration      : point-of-control shifting directionally for ≥2
             consecutive bars  (bc_l3_volume_profile_pro → vp_poc_mig = ±1)

Entry condition:
  LONG:  brick_up AND (vi_stacked_buy + ff_buy_abs + vp_poc_up) >= min_signals
  SHORT: not brick_up AND (vi_stacked_sell + ff_sell_abs + vp_poc_down) >= min_signals
  (min_signals ∈ {2, 3} — 2 = any pair of signals, 3 = all three)

Exit: first opposing Renko brick (standard).

Hypothesis: three independent L3 microstructure signals confirming the same
direction simultaneously implies an unusually high-conviction institutional
event (imbalance + absorption + POC drive = institutional accumulation /
distribution in progress). The extreme rarity should produce very high Profit
Factors at the cost of low trade frequency.

EURAUD IS:  2023-07-20 → 2025-09-30
EURAUD OOS: 2025-10-01 → 2026-03-17
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from indicators.blackcat1402.blackcat_l3_volume_imbalance_pro import (
    calc_bc_l3_volume_imbalance_pro,
)
from indicators.blackcat1402.blackcat_l3_footprint_fusion_pro import (
    calc_bc_l3_footprint_fusion_pro,
)
from indicators.blackcat1402.blackcat_l3_volume_profile_pro import (
    calc_bc_l3_volume_profile_pro,
)
from renko.data import load_renko_export
from renko.indicators import add_renko_indicators

DESCRIPTION = "EURAUD Alpha Sniper — 2-of-3 / 3-of-3 L3 microstructure confluence"

HYPOTHESIS = (
    "When three independent L3 signals (stacked imbalance, absorption, POC "
    "migration) all point in the same direction at a single Renko brick, an "
    "institutional accumulation / distribution event is almost certainly in "
    "progress. The extreme rarity of this confluence should produce very high "
    "Profit Factors. min_signals=2 relaxes the confluence to any two of three "
    "signals for higher trade frequency while retaining edge."
)

RENKO_FILE      = "OANDA_EURAUD, 1S renko 0.0006.csv"
COMMISSION_PCT  = 0.009
INITIAL_CAPITAL = 1000.0

# ---------------------------------------------------------------------------
# PARAM_GRID
# ---------------------------------------------------------------------------
# min_stacked:    consecutive imbalance bars required for stacked signal
# vp_lookback:    rolling bars for Volume Profile Pro (POC / VAH / VAL)
# min_signals:    minimum signals that must fire simultaneously (2 or 3)
# cooldown:       minimum bricks between entries
# session_start:  UTC hour gate (0 = no gate, 13 = London+NY only)
#
# imb_threshold removed: on Renko bricks the Elder buy/sell volume approximation
# is always 100% buy (UP brick) or 100% sell (DOWN brick), so any imbalance
# ratio threshold is trivially satisfied by brick direction alone — the
# parameter had zero effect in Phase 2 (200/300/400 gave identical results).
# imbalance_threshold is hardcoded at 300.0 in the indicator call.
PARAM_GRID = {
    "min_stacked":    [2, 3],
    "vp_lookback":    [30, 50, 100],
    "min_signals":    [2, 3],
    "cooldown":       [5, 10, 20],
    "session_start":  [0, 13],
}
# Total: 2 × 3 × 2 × 3 × 2 = 72 combos
# Unique cache builds: 2 (stacked) × 3 (vp) = 6


# ---------------------------------------------------------------------------
# Indicator cache — keyed by (min_stacked, vp_lookback)
#
# Three indicators are computed on SEPARATE lowercase copies of the Renko df
# to avoid bc_ column name collisions between Volume Imbalance Pro,
# Footprint Fusion Pro, and Volume Profile Pro.
# Only the specific columns needed for entry logic are extracted and renamed.
# ---------------------------------------------------------------------------

_CACHE: dict = {}


def _get_or_build_cache(
    min_stacked: int,
    vp_lookback: int,
) -> pd.DataFrame:
    key = (min_stacked, vp_lookback)
    if key in _CACHE:
        return _CACHE[key]

    df = load_renko_export(RENKO_FILE)
    add_renko_indicators(df)  # adds pre-shifted standard indicators

    # Lowercase copy for all three blackcat indicators
    df_lc = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })

    # ── Signal 1: Stacked Imbalance — Volume Imbalance Pro ───────────────────
    vi = calc_bc_l3_volume_imbalance_pro(
        df_lc,
        imbalance_threshold=300.0,
        min_stacked_rows=min_stacked,
    )
    df["vi_stacked_buy"]  = vi["bc_stacked_buy"].shift(1).values
    df["vi_stacked_sell"] = vi["bc_stacked_sell"].shift(1).values

    # ── Signal 2: Absorption — Footprint Fusion Pro ───────────────────────────
    # Uses its own internal POC (rolling_vp_bars=50 default) for ATR-proximity
    # absorption detection. The output columns don't conflict with Signal 1.
    ff = calc_bc_l3_footprint_fusion_pro(df_lc)
    df["ff_buy_abs"]  = ff["bc_buy_absorption"].shift(1).values
    df["ff_sell_abs"] = ff["bc_sell_absorption"].shift(1).values

    # ── Signal 3: POC Migration — Volume Profile Pro ──────────────────────────
    # bc_poc_migration: +1 = upward migration, -1 = downward, 0 = neutral
    vp = calc_bc_l3_volume_profile_pro(df_lc, rolling_vp_bars=vp_lookback)
    df["vp_poc_mig"] = vp["bc_poc_migration"].shift(1).values

    _CACHE[key] = df
    return df


# ---------------------------------------------------------------------------
# Signal generator
# ---------------------------------------------------------------------------

def generate_signals(
    df: pd.DataFrame,
    min_stacked:    int   = 3,
    vp_lookback:    int   = 50,
    min_signals:    int   = 2,
    cooldown:       int   = 10,
    session_start:  int   = 0,
) -> pd.DataFrame:
    """
    Alpha Sniper: enter only when min_signals (2 or 3) of the three L3
    microstructure events peak simultaneously at the same Renko brick.

    Args:
        df:            Renko DataFrame with brick_up + standard indicators.
        min_stacked:   Minimum consecutive imbalance bars for stacked signal.
        vp_lookback:   Rolling bars for Volume Profile POC computation.
        min_signals:   Minimum simultaneous signals required (2 or 3).
        cooldown:      Minimum bricks between entries.
        session_start: UTC hour gate (0 = off, 13 = London+NY only).

    Returns:
        df with long_entry, long_exit, short_entry, short_exit bool columns.
    """
    # Warmup: Volume Profile Pro needs vp_lookback bars; add buffer
    warmup = max(vp_lookback + 5, 60)

    c = _get_or_build_cache(min_stacked, vp_lookback).reindex(df.index)

    brick_up  = df["brick_up"].values
    hours     = df.index.hour
    n         = len(df)

    vi_stk_buy  = c["vi_stacked_buy"].fillna(False).values.astype(bool)
    vi_stk_sell = c["vi_stacked_sell"].fillna(False).values.astype(bool)
    ff_buy_abs  = c["ff_buy_abs"].fillna(False).values.astype(bool)
    ff_sell_abs = c["ff_sell_abs"].fillna(False).values.astype(bool)
    vp_poc_mig  = c["vp_poc_mig"].fillna(0).values.astype(int)

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    in_position    = False
    trade_dir      = 0
    last_trade_bar = -999_999

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # ── Exit: first opposing brick ────────────────────────────────────────
        if in_position:
            if trade_dir == 1 and not up:
                long_exit[i]  = True
                in_position   = False
                trade_dir     = 0
            elif trade_dir == -1 and up:
                short_exit[i] = True
                in_position   = False
                trade_dir     = 0

        if in_position:
            continue

        # ── Session gate ──────────────────────────────────────────────────────
        if session_start > 0 and hours[i] < session_start:
            continue

        # ── Cooldown ──────────────────────────────────────────────────────────
        if (i - last_trade_bar) < cooldown:
            continue

        # ── Count active bullish signals ──────────────────────────────────────
        if up:
            bull_count = (
                int(vi_stk_buy[i])
                + int(ff_buy_abs[i])
                + int(vp_poc_mig[i] == 1)
            )
            if bull_count >= min_signals:
                long_entry[i]  = True
                in_position    = True
                trade_dir      = 1
                last_trade_bar = i

        # ── Count active bearish signals ──────────────────────────────────────
        else:
            bear_count = (
                int(vi_stk_sell[i])
                + int(ff_sell_abs[i])
                + int(vp_poc_mig[i] == -1)
            )
            if bear_count >= min_signals:
                short_entry[i] = True
                in_position    = True
                trade_dir      = -1
                last_trade_bar = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
