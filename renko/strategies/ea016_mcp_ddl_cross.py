"""EA016: MCP+DDL Dual Momentum — BC L1 Momentum Crossover Pro + Dynamic Defense Line

Dual-oscillator momentum strategy using two independently constructed BC L1
indicators from different lookback windows and normalization methods:

  MCP (Momentum Crossover Pro): DIF line (5-bar ZLEMA of 27-bar normalized price)
    crosses the WhiteOut line (8-bar ZLEMA of 33-bar normalized price) while
    PinkIn < 80 (not overbought) and DIF is rising. Signal: bc_mcp_buy_signal /
    bc_mcp_sell_signal.

  DDL (Dynamic Defense Line): Stochastic oscillator in a 34-bar range, compared
    to its own EMA. Regime: bc_buy_sell_diff > 0 = bullish momentum, < 0 = bearish.

Unlike the BC L3 MACD+LC gate pair (which failed EURAUD), these are both pure
price-position oscillators normalized to 0–100 within rolling high/low ranges —
a different family from the trend-following MACD and regime-based FSB gates.

Signal logic:
  LONG:  current brick is UP
         AND bc_mcp_buy_signal fired on previous bar (DIF crossed above WhiteOut,
             PinkIn < 80, DIF rising)
         AND (bc_buy_sell_diff > 0 OR not use_ddl_gate)

  SHORT: current brick is DOWN
         AND bc_mcp_sell_signal fired on previous bar (DIF crossed below WhiteOut,
             PinkIn > 20, DIF falling)
         AND (bc_buy_sell_diff < 0 OR not use_ddl_gate)

Exit: first opposing Renko brick (standard).

EURAUD IS:  2023-07-20 → 2025-09-30
EURAUD OOS: 2025-10-01 → 2026-03-17
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from indicators.blackcat1402.bc_l1_momentum_crossover_pro import (
    calc_bc_momentum_crossover_pro,
)
from indicators.blackcat1402.bc_l1_dynamic_defense_line import (
    calc_bc_dynamic_defense_line,
)
from renko.data import load_renko_export
from renko.indicators import add_renko_indicators

DESCRIPTION = "EURAUD MCP crossover entries with optional DDL regime confirmation"

HYPOTHESIS = (
    "The BC L1 MCP crossover fires when DIF (fast, 27-bar normalized) crosses "
    "the WhiteOut line (slow, 33-bar normalized) in non-overbought territory — "
    "a normalized momentum turn signal. The DDL stochastic regime (bc_buy_sell_diff) "
    "acts as a second confirmation from an independent 34-bar oscillator family. "
    "Both indicators normalize price position within their own rolling ranges, "
    "making them more stable on Renko's fixed-size bricks than MACD-style "
    "absolute-value indicators. The combination is structurally different from "
    "the MACD/FSB pair that was rejected on EURAUD."
)

RENKO_FILE      = "OANDA_EURAUD, 1S renko 0.0006.csv"
COMMISSION_PCT  = 0.009
INITIAL_CAPITAL = 1000.0

# ---------------------------------------------------------------------------
# PARAM_GRID
# ---------------------------------------------------------------------------
# use_ddl_gate:  if True, also require DDL regime to confirm direction
# cooldown:      minimum bricks between entries
# session_start: UTC hour gate (0 = no gate, 13 = London+NY only)
PARAM_GRID = {
    "use_ddl_gate":  [True, False],
    "cooldown":      [5, 10, 20],
    "session_start": [0, 13],
}
# Total: 2 × 3 × 2 = 12 combos
# Unique cache builds: 1 (all indicator params fixed at defaults)


# ---------------------------------------------------------------------------
# Indicator cache — single build
# ---------------------------------------------------------------------------

_CACHE: dict = {}


def _get_or_build_cache() -> pd.DataFrame:
    if "data" in _CACHE:
        return _CACHE["data"]

    df = load_renko_export(RENKO_FILE)
    add_renko_indicators(df)

    # ── BC L1 Momentum Crossover Pro (capitalized OHLC) ───────────────────────
    mcp = calc_bc_momentum_crossover_pro(df)
    df["mcp_buy"]  = mcp["bc_mcp_buy_signal"].shift(1).values
    df["mcp_sell"] = mcp["bc_mcp_sell_signal"].shift(1).values

    # ── BC L1 Dynamic Defense Line (capitalized OHLC including Open) ──────────
    ddl = calc_bc_dynamic_defense_line(df)
    df["ddl_diff"] = ddl["bc_buy_sell_diff"].shift(1).values   # >0 bullish, <0 bearish

    _CACHE["data"] = df
    return df


# ---------------------------------------------------------------------------
# Signal generator
# ---------------------------------------------------------------------------

def generate_signals(
    df:            pd.DataFrame,
    use_ddl_gate:  bool = True,
    cooldown:      int  = 10,
    session_start: int  = 0,
) -> pd.DataFrame:
    """
    MCP+DDL Dual Momentum: enter on MCP normalized crossover signals,
    optionally confirmed by DDL stochastic regime direction.

    Args:
        df:            Renko DataFrame with brick_up + standard indicators.
        use_ddl_gate:  Also require DDL bc_buy_sell_diff in correct direction.
        cooldown:      Minimum bricks between entries.
        session_start: UTC hour gate (0 = off, 13 = London+NY only).

    Returns:
        df with long_entry, long_exit, short_entry, short_exit bool columns.
    """
    # Warmup: MCP uses 33-bar lookback; DDL uses 34-bar lookback
    warmup = 60

    c = _get_or_build_cache().reindex(df.index)

    brick_up  = df["brick_up"].values
    hours     = df.index.hour
    n         = len(df)

    mcp_buy  = c["mcp_buy"].fillna(False).values.astype(bool)
    mcp_sell = c["mcp_sell"].fillna(False).values.astype(bool)
    ddl_diff = c["ddl_diff"].fillna(0.0).values.astype(float)

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

        ddl_val = ddl_diff[i]

        # ── LONG: MCP buy crossover + UP brick + optional DDL bullish ─────────
        if up and mcp_buy[i]:
            ddl_ok = (not use_ddl_gate) or (ddl_val > 0)
            if ddl_ok:
                long_entry[i]  = True
                in_position    = True
                trade_dir      = 1
                last_trade_bar = i

        # ── SHORT: MCP sell crossover + DOWN brick + optional DDL bearish ─────
        elif not up and mcp_sell[i]:
            ddl_ok = (not use_ddl_gate) or (ddl_val < 0)
            if ddl_ok:
                short_entry[i] = True
                in_position    = True
                trade_dir      = -1
                last_trade_bar = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
