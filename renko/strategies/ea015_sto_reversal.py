"""EA015: STO Swing Reversal — BC L1 Swing Trade Oscillator Oversold/Overbought Entries

Mean-reversion strategy using the BC L1 Swing Trade Oscillator (STO). The STO
measures price position within a short rolling range (n-bar low to 4-bar high)
and generates:
  - Buy signal:  MainForce crosses above LifeLine while MainForce < 40 (oversold)
  - Sell signal: LifeLine crosses above MainForce while MainForce > 90 (overbought)
                 OR previous MainForce > 80 (extended zone sell)

On Renko bricks, MainForce naturally drops after consecutive DOWN bricks (price
near range low) and rises after UP bricks. The buy signal fires at the first UP
brick after an oversold run — exactly capturing the Renko reversal while the
oscillator still confirms depressed momentum.

Signal logic:
  LONG:  current brick is UP
         AND bc_sto_buy fired on the previous bar (MF crossed above LL with MF < 40)
         AND (price above VP POC OR not req_vp)

  SHORT: current brick is DOWN
         AND (bc_sto_sell1 OR bc_sto_sell2) fired on previous bar
         AND (price below VP POC OR not req_vp)

Exit: first opposing Renko brick (standard).

EURAUD IS:  2023-07-20 → 2025-09-30
EURAUD OOS: 2025-10-01 → 2026-03-17
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from indicators.blackcat1402.bc_l1_swing_trade_oscillator import (
    calc_bc_swing_trade_oscillator,
)
from indicators.dgtrd.volume_profile import volume_profile_pivot_anchored
from renko.data import load_renko_export
from renko.indicators import add_renko_indicators

DESCRIPTION = "EURAUD STO oversold/overbought reversal entries with optional VP gate"

HYPOTHESIS = (
    "The BC L1 Swing Trade Oscillator's buy signal fires precisely when price "
    "has been in an oversold range and momentum crosses from bearish to bullish "
    "— on Renko bricks this maps to the first UP brick after a run of DOWN bricks "
    "while MainForce < 40. EURAUD's mean-reverting character may respond to this "
    "oscillator-confirmed reversal better than the trend-following BC/FS gates "
    "that failed in Phase 3."
)

RENKO_FILE      = "OANDA_EURAUD, 1S renko 0.0006.csv"
COMMISSION_PCT  = 0.009
INITIAL_CAPITAL = 1000.0

# ---------------------------------------------------------------------------
# PARAM_GRID
# ---------------------------------------------------------------------------
# sto_n:         STO lookback for n-bar lowest low (default 5; 3=faster, 8=slower)
# req_vp:        require price on correct side of VP POC (EA008-style gate)
# cooldown:      minimum bricks between entries
# session_start: UTC hour gate (0 = no gate, 13 = London+NY only)
PARAM_GRID = {
    "sto_n":         [3, 5, 8],
    "req_vp":        [True, False],
    "cooldown":      [5, 10, 20],
    "session_start": [0, 13],
}
# Total: 3 × 2 × 3 × 2 = 36 combos
# Unique cache builds: 3 (sto_n varies; VP always computed)


# ---------------------------------------------------------------------------
# Indicator cache — keyed by sto_n only
# VP is always computed (req_vp handled in signal loop)
# ---------------------------------------------------------------------------

_CACHE: dict = {}


def _get_or_build_cache(sto_n: int) -> pd.DataFrame:
    if sto_n in _CACHE:
        return _CACHE[sto_n]

    df = load_renko_export(RENKO_FILE)
    add_renko_indicators(df)

    # ── BC L1 Swing Trade Oscillator (capitalized OHLC) ───────────────────────
    sto = calc_bc_swing_trade_oscillator(df, n=sto_n)
    df["sto_buy"]   = sto["bc_sto_buy"].shift(1).values
    df["sto_sell1"] = sto["bc_sto_sell1"].shift(1).values
    df["sto_sell2"] = sto["bc_sto_sell2"].shift(1).values

    # ── Volume Profile — pivot-anchored POC side gate (EA008 style) ───────────
    df_vp = volume_profile_pivot_anchored(df, pvt_length=20, num_bins=25, va_pct=0.68)
    df["vp_above"] = df_vp["vp_above_poc"].shift(1).values

    _CACHE[sto_n] = df
    return df


# ---------------------------------------------------------------------------
# Signal generator
# ---------------------------------------------------------------------------

def generate_signals(
    df:            pd.DataFrame,
    sto_n:         int  = 5,
    req_vp:        bool = False,
    cooldown:      int  = 10,
    session_start: int  = 0,
) -> pd.DataFrame:
    """
    STO Swing Reversal: enter at oversold/overbought Renko reversals confirmed
    by the Swing Trade Oscillator's crossover signals.

    Args:
        df:            Renko DataFrame with brick_up + standard indicators.
        sto_n:         STO lookback for lowest-low range (default 5).
        req_vp:        Require VP POC side alignment for entries.
        cooldown:      Minimum bricks between entries.
        session_start: UTC hour gate (0 = off, 13 = London+NY only).

    Returns:
        df with long_entry, long_exit, short_entry, short_exit bool columns.
    """
    # Warmup: STO uses 34-bar rolling range; VP needs pivot lookback
    warmup = 100

    c = _get_or_build_cache(sto_n).reindex(df.index)

    brick_up  = df["brick_up"].values
    hours     = df.index.hour
    n         = len(df)

    sto_buy   = c["sto_buy"].fillna(False).values.astype(bool)
    sto_sell1 = c["sto_sell1"].fillna(False).values.astype(bool)
    sto_sell2 = c["sto_sell2"].fillna(False).values.astype(bool)
    vp_above  = c["vp_above"].fillna(True).values   # NaN-pass: True allows long

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

        vp_val = vp_above[i]
        vp_nan = np.isnan(float(vp_val)) if not isinstance(vp_val, bool) else False

        # ── LONG: STO oversold crossover + UP brick ───────────────────────────
        if up and sto_buy[i]:
            if not req_vp or vp_nan or bool(vp_val):
                long_entry[i]  = True
                in_position    = True
                trade_dir      = 1
                last_trade_bar = i

        # ── SHORT: STO overbought crossover + DOWN brick ──────────────────────
        elif not up and (sto_sell1[i] or sto_sell2[i]):
            if not req_vp or vp_nan or not bool(vp_val):
                short_entry[i] = True
                in_position    = True
                trade_dir      = -1
                last_trade_bar = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
