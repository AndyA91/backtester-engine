"""EA011: Advanced Auction Breakout — L3 Volume Profile Pro + CVD + POC Migration

Evolution of EA005 (Value Area Breakout). Upgrades the breakout confirmation
from simple price > VAH to three stacked auction-theory conditions:

  Condition 1 — Delta-Confirmed Breakout:
    bc_vah_breakout = True  (price crosses above VAH with positive delta)
    bc_val_breakdown = True (price crosses below VAL with negative delta)

  Condition 2 — CVD Trending in Breakout Direction:
    For longs:  bc_cvd[i] > bc_cvd[i - cvd_lookback]  (CVD rising)
    For shorts: bc_cvd[i] < bc_cvd[i - cvd_lookback]  (CVD falling)

  Condition 3 — POC Migration Support (optional):
    bc_poc_migration == +1 for longs  (POC drifting upward)
    bc_poc_migration == -1 for shorts (POC drifting downward)

Signal logic:
  LONG:  bc_vah_breakout AND cvd_rising AND (poc_mig == 1 OR not req_poc_mig)
         AND current brick is UP
  SHORT: bc_val_breakdown AND cvd_falling AND (poc_mig == -1 OR not req_poc_mig)
         AND current brick is DOWN

Exit: first opposing Renko brick (standard).

Hypothesis: requiring delta confirmation, CVD direction, and POC migration all
at once elevates the EA005 breakout signal from "price escaped the VA" to
"institutional order-flow is actively driving a structural repositioning."
The combination should dramatically reduce false breakouts vs plain price
threshold crossing.

EURAUD IS:  2023-07-20 → 2025-09-30
EURAUD OOS: 2025-10-01 → 2026-03-17
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from indicators.blackcat1402.blackcat_l3_volume_profile_pro import (
    calc_bc_l3_volume_profile_pro,
)
from renko.data import load_renko_export
from renko.indicators import add_renko_indicators

DESCRIPTION = "EURAUD Advanced Auction Breakout — delta + CVD + POC migration confluence"

HYPOTHESIS = (
    "EA005 fires on any VA escape. EA011 requires three concurrent auction-theory "
    "conditions: delta-confirmed VAH/VAL cross, CVD trending in breakout direction, "
    "and optional POC migration support. This triple gate targets only breakouts "
    "where institutional order flow is actively repositioning, dramatically "
    "reducing false breakouts at the cost of lower trade frequency."
)

RENKO_FILE      = "OANDA_EURAUD, 1S renko 0.0006.csv"
COMMISSION_PCT  = 0.009
INITIAL_CAPITAL = 1000.0

# ---------------------------------------------------------------------------
# PARAM_GRID
# ---------------------------------------------------------------------------
# vp_lookback:   rolling bars for Volume Profile Pro POC/VAH/VAL computation
# cvd_lookback:  bars over which CVD must be trending in breakout direction
# req_poc_mig:   if True, POC migration direction must align with entry
# cooldown:      minimum bricks between entries
# session_start: UTC hour gate (0 = no gate, 13 = London+NY only)
PARAM_GRID = {
    "vp_lookback":  [30, 50, 100],
    "cvd_lookback": [1, 3, 5],
    "req_poc_mig":  [True, False],
    "cooldown":     [5, 10, 20],
    "session_start":[0, 13],
}
# Total: 3 × 3 × 2 × 3 × 2 = 108 combos
# Unique cache builds: 3 (vp_lookback only)


# ---------------------------------------------------------------------------
# Indicator cache — keyed by vp_lookback
# ---------------------------------------------------------------------------

_CACHE: dict = {}


def _get_or_build_cache(vp_lookback: int) -> pd.DataFrame:
    key = vp_lookback
    if key in _CACHE:
        return _CACHE[key]

    df = load_renko_export(RENKO_FILE)
    add_renko_indicators(df)

    # Volume Profile Pro expects lowercase ohlcv
    df_lc = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    vp = calc_bc_l3_volume_profile_pro(df_lc, rolling_vp_bars=vp_lookback)

    # Shift all outputs to prevent lookahead
    df["vp_vah_breakout"]  = vp["bc_vah_breakout"].shift(1).values
    df["vp_val_breakdown"] = vp["bc_val_breakdown"].shift(1).values
    df["vp_poc_mig"]       = vp["bc_poc_migration"].shift(1).values
    df["vp_cvd"]           = vp["bc_cvd"].shift(1).values

    _CACHE[key] = df
    return df


# ---------------------------------------------------------------------------
# Signal generator
# ---------------------------------------------------------------------------

def generate_signals(
    df: pd.DataFrame,
    vp_lookback:  int  = 50,
    cvd_lookback: int  = 3,
    req_poc_mig:  bool = True,
    cooldown:     int  = 10,
    session_start:int  = 0,
) -> pd.DataFrame:
    """
    Advanced Auction Breakout: delta-confirmed VA escape + CVD trend + POC migration.

    Args:
        df:            Renko DataFrame with brick_up + standard indicators.
        vp_lookback:   Rolling bars for Volume Profile computation.
        cvd_lookback:  Bricks over which CVD must be trending in breakout direction.
        req_poc_mig:   Require POC migration to align with breakout direction.
        cooldown:      Minimum bricks between entries.
        session_start: UTC hour gate (0 = off, 13 = London+NY only).

    Returns:
        df with long_entry, long_exit, short_entry, short_exit bool columns.
    """
    warmup = max(vp_lookback + cvd_lookback + 5, 60)

    c = _get_or_build_cache(vp_lookback).reindex(df.index)

    brick_up  = df["brick_up"].values
    hours     = df.index.hour
    n         = len(df)

    vah_bo    = c["vp_vah_breakout"].fillna(False).values.astype(bool)
    val_bd    = c["vp_val_breakdown"].fillna(False).values.astype(bool)
    poc_mig   = c["vp_poc_mig"].fillna(0).values.astype(int)
    cvd_vals  = c["vp_cvd"].fillna(0.0).values.astype(float)

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

        # ── CVD trend ─────────────────────────────────────────────────────────
        cvd_ref_i = i - cvd_lookback
        if cvd_ref_i < 0:
            continue
        cvd_rising  = cvd_vals[i] > cvd_vals[cvd_ref_i]
        cvd_falling = cvd_vals[i] < cvd_vals[cvd_ref_i]

        # ── LONG: delta-confirmed VAH breakout + CVD up + POC up ──────────────
        if up and vah_bo[i] and cvd_rising:
            poc_ok = (not req_poc_mig) or (poc_mig[i] == 1)
            if poc_ok:
                long_entry[i]  = True
                in_position    = True
                trade_dir      = 1
                last_trade_bar = i

        # ── SHORT: delta-confirmed VAL breakdown + CVD down + POC down ────────
        elif not up and val_bd[i] and cvd_falling:
            poc_ok = (not req_poc_mig) or (poc_mig[i] == -1)
            if poc_ok:
                short_entry[i] = True
                in_position    = True
                trade_dir      = -1
                last_trade_bar = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
