"""EA010: Cyberpunk Momentum Flow — VTA Value Zone + Momentum Crossover Pro

Trend-continuation strategy. The L2 Cyberpunk Value Trend Analyzer defines
"value windows" — price regimes where the trend has momentum but is not yet
overextended. The L1 Momentum Crossover Pro then fires a precision entry
trigger within those windows.

Signal logic:
  LONG:  bc_vta_value_trend > vta_long_min   (in bullish value zone)
         AND NOT bc_vta_is_overbought        (not overextended)
         AND bc_mcp_buy_signal               (DIF crosses above White Out)
         AND current brick is UP

  SHORT: bc_vta_value_trend < vta_short_max  (in bearish / weak-trend zone)
         AND bc_mcp_sell_signal              (DIF crosses below White Out)
         AND current brick is DOWN

Exit: first opposing Renko brick (standard).

Hypothesis: the VTA value_trend oscillator identifies when price is in a
high-quality trend regime (above vta_long_min, below overbought). Inside
that window, the MCP crossover provides a precise, low-lag entry trigger.
Trend-continuation trades within confirmed value windows should have a
higher hit rate than unconstrained momentum entries.

EURAUD IS:  2023-07-20 → 2025-09-30
EURAUD OOS: 2025-10-01 → 2026-03-17
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from indicators.blackcat1402.bc_l2_cyberpunk_value_trend_analyzer import (
    calc_bc_cyberpunk_value_trend_analyzer,
)
from indicators.blackcat1402.bc_l1_momentum_crossover_pro import (
    calc_bc_momentum_crossover_pro,
)
from renko.data import load_renko_export
from renko.indicators import add_renko_indicators

DESCRIPTION = "EURAUD Cyberpunk Value Trend + Momentum Crossover Pro trend continuation"

HYPOTHESIS = (
    "The L2 Cyberpunk VTA measures how much price is 'in value' vs overextended. "
    "When value_trend is above the bullish gate but below the overbought zone, "
    "the L1 MCP DIF/WhiteOut crossover provides a low-lag precision entry. "
    "Combining a value-regime gate with a momentum trigger should yield higher "
    "conviction trend-continuation trades with reduced false positives."
)

RENKO_FILE      = "OANDA_EURAUD, 1S renko 0.0006.csv"
COMMISSION_PCT  = 0.009
INITIAL_CAPITAL = 1000.0

# ---------------------------------------------------------------------------
# PARAM_GRID
# ---------------------------------------------------------------------------
# vta_long_min:  minimum bc_vta_value_trend to allow long entries (value zone gate)
# vta_short_max: maximum bc_vta_value_trend to allow short entries (weak zone gate)
# req_no_overbought: if True, gate longs with NOT bc_vta_is_overbought
# cooldown:      minimum bricks between entries
# session_start: UTC hour gate (0 = no gate, 13 = London+NY only)
PARAM_GRID = {
    "vta_long_min":       [25, 35, 45],
    "vta_short_max":      [50, 60, 75],
    "req_no_overbought":  [True, False],
    "cooldown":           [5, 10, 20],
    "session_start":      [0, 13],
}
# Total: 3 × 3 × 2 × 3 × 2 = 108 combos
# Unique cache builds: 1 (both indicators use fixed defaults)


# ---------------------------------------------------------------------------
# Indicator cache — single build; VTA and MCP use fixed default parameters.
# The strategy-level thresholds (vta_long_min, vta_short_max) are applied in
# the signal loop against the raw bc_vta_value_trend column.
# ---------------------------------------------------------------------------

_CACHE: dict = {}


def _get_or_build_cache() -> pd.DataFrame:
    if "data" in _CACHE:
        return _CACHE["data"]

    df = load_renko_export(RENKO_FILE)
    add_renko_indicators(df)  # pre-shifted standard indicators

    # ── Cyberpunk Value Trend Analyzer (capitalized columns) ─────────────────
    # Both functions return df.copy() with new columns appended.
    df = calc_bc_cyberpunk_value_trend_analyzer(df)
    df["bc_vta_value_trend"]   = df["bc_vta_value_trend"].shift(1)
    df["bc_vta_is_overbought"] = df["bc_vta_is_overbought"].shift(1)

    # ── Momentum Crossover Pro (capitalized columns) ──────────────────────────
    df = calc_bc_momentum_crossover_pro(df)
    df["bc_mcp_buy_signal"]  = df["bc_mcp_buy_signal"].shift(1)
    df["bc_mcp_sell_signal"] = df["bc_mcp_sell_signal"].shift(1)
    df["bc_mcp_dif"]         = df["bc_mcp_dif"].shift(1)

    _CACHE["data"] = df
    return df


# ---------------------------------------------------------------------------
# Signal generator
# ---------------------------------------------------------------------------

def generate_signals(
    df: pd.DataFrame,
    vta_long_min:      float = 35.0,
    vta_short_max:     float = 60.0,
    req_no_overbought: bool  = True,
    cooldown:          int   = 10,
    session_start:     int   = 0,
) -> pd.DataFrame:
    """
    Cyberpunk Momentum Flow: enter within VTA value windows on MCP crossover.

    Args:
        df:               Renko DataFrame with brick_up + standard indicators.
        vta_long_min:     Minimum value_trend level to allow long entries.
        vta_short_max:    Maximum value_trend level to allow short entries.
        req_no_overbought: Gate longs with NOT bc_vta_is_overbought.
        cooldown:         Minimum bricks between entries.
        session_start:    UTC hour gate (0 = off, 13 = London+NY only).

    Returns:
        df with long_entry, long_exit, short_entry, short_exit bool columns.
    """
    # Warmup: VTA needs 75-bar range + RMA20 (~100 bars total) + MCP ~33 bars
    warmup = 120

    c = _get_or_build_cache().reindex(df.index)

    brick_up    = df["brick_up"].values
    hours       = df.index.hour
    n           = len(df)

    vta_vt     = c["bc_vta_value_trend"].fillna(50.0).values.astype(float)
    vta_ob     = c["bc_vta_is_overbought"].fillna(False).values.astype(bool)
    mcp_buy    = c["bc_mcp_buy_signal"].fillna(False).values.astype(bool)
    mcp_sell   = c["bc_mcp_sell_signal"].fillna(False).values.astype(bool)

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

        vt = vta_vt[i]

        # ── LONG: bullish value zone + MCP buy crossover + UP brick ───────────
        if up:
            in_value_zone = vt >= vta_long_min
            ob_ok         = (not req_no_overbought) or (not vta_ob[i])
            if in_value_zone and ob_ok and mcp_buy[i]:
                long_entry[i]  = True
                in_position    = True
                trade_dir      = 1
                last_trade_bar = i

        # ── SHORT: weak/bearish zone + MCP sell crossover + DOWN brick ────────
        elif not up:
            in_weak_zone = vt <= vta_short_max
            if in_weak_zone and mcp_sell[i]:
                short_entry[i] = True
                in_position    = True
                trade_dir      = -1
                last_trade_bar = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
