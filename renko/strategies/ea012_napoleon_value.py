"""EA012: Napoleon Value Layering — Multi-Layer Structural Mean Reversion

Multi-layer structural mean-reversion strategy. The L2 Napoleon Mille-feuille
scores market structure from 0–9 by counting how many of 9 short-term band /
medium-term channel conditions are satisfied. Extreme scores signal structural
over-extension; the L1 Undervalued Momentum Scanner confirms the reversal impulse.

Signal logic:
  LONG:  bc_nap_strength_score <= nap_buy_thr  (Napoleon gate — see note below)
         AND (bc_ums_buy OR not req_ums)       (UMS undervalued momentum fires)
         AND current brick is UP

  SHORT: bc_nap_strength_score >= nap_sell_thr (Napoleon gate — see note below)
         AND (bc_ums_sell OR not req_ums)      (UMS trend exhaustion fires)
         AND current brick is DOWN

Exit: first opposing Renko brick (standard).

DIAGNOSTIC NOTE (Phase 3): The Napoleon band/channel multipliers (±15%/±6%)
were designed for equities. On EURAUD Renko bricks (0.0006), 25 bricks span
only 0.9% of price so the score is locked at 4 on 99.9% of bars and never
exceeds 4. nap_buy_thr=4 and nap_sell_thr=4 are set to the actual score
maximum so the Napoleon gate is effectively always True — the strategy
becomes a clean UMS scanner. req_ums=True tests the UMS edge; req_ums=False
is the unconstrained baseline.

EURAUD IS:  2023-07-20 → 2025-09-30
EURAUD OOS: 2025-10-01 → 2026-03-17
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from indicators.blackcat1402.bc_l2_napoleon_mille_feuille import (
    calc_bc_napoleon_mille_feuille,
)
from indicators.blackcat1402.bc_l1_undervalued_momentum_scanner import (
    calc_bc_undervalued_momentum_scanner,
)
from renko.data import load_renko_export
from renko.indicators import add_renko_indicators

DESCRIPTION = "EURAUD Napoleon Mille-feuille layered structure + UMS mean-reversion entries"

HYPOTHESIS = (
    "The Napoleon strength score identifies when the short-term price band "
    "is deeply below (score ≤ 2) or far above (score ≥ 7) the medium-term "
    "channel — structural extremes that precede reversals. The UMS confirms "
    "that genuine undervalued momentum (or trend exhaustion) is present, "
    "distinguishing true reversals from continued trends through the extreme."
)

RENKO_FILE      = "OANDA_EURAUD, 1S renko 0.0006.csv"
COMMISSION_PCT  = 0.009
INITIAL_CAPITAL = 1000.0

# ---------------------------------------------------------------------------
# PARAM_GRID
# ---------------------------------------------------------------------------
# nap_buy_thr:   strength score <= this to qualify as "value floor" for longs
# nap_sell_thr:  strength score >= this to qualify as "overbought" for shorts
# req_ums:       if True, require UMS buy/sell confirmation signal
# cooldown:      minimum bricks between entries
# session_start: UTC hour gate (0 = no gate, 13 = London+NY only)
PARAM_GRID = {
    "nap_buy_thr":  [4],          # score always ≤ 4 → Napoleon gate always True
    "nap_sell_thr": [4],          # score always ≥ 4 → Napoleon gate always True
    "req_ums":      [True, False],
    "cooldown":     [5, 10, 20],
    "session_start":[0, 13],
}
# Total: 1 × 1 × 2 × 3 × 2 = 12 combos
# Unique cache builds: 1 (all indicator params fixed)
# Napoleon thresholds fixed at 4 (the actual score maximum on EURAUD Renko).
# This cleanly isolates the UMS signal as the sole entry driver.


# ---------------------------------------------------------------------------
# Indicator cache — single build; thresholds are signal-loop variables
# ---------------------------------------------------------------------------

_CACHE: dict = {}


def _get_or_build_cache() -> pd.DataFrame:
    if "data" in _CACHE:
        return _CACHE["data"]

    df = load_renko_export(RENKO_FILE)
    add_renko_indicators(df)

    # ── Napoleon Mille-feuille (capitalized High/Low/Close/Open) ─────────────
    # Use default thresholds; strategy-level thresholds applied in signal loop.
    df = calc_bc_napoleon_mille_feuille(df)
    df["nap_score"]       = df["bc_nap_strength_score"].shift(1)
    df["nap_trend_up"]    = df["bc_nap_trending_up"].shift(1)
    df["nap_trend_down"]  = df["bc_nap_trending_down"].shift(1)

    # ── Undervalued Momentum Scanner (capitalized High/Low/Close) ────────────
    df = calc_bc_undervalued_momentum_scanner(df)
    df["ums_buy"]  = df["bc_ums_buy"].shift(1)
    df["ums_sell"] = df["bc_ums_sell"].shift(1)

    _CACHE["data"] = df
    return df


# ---------------------------------------------------------------------------
# Signal generator
# ---------------------------------------------------------------------------

def generate_signals(
    df: pd.DataFrame,
    nap_buy_thr:  int  = 2,
    nap_sell_thr: int  = 7,
    req_ums:      bool = True,
    cooldown:     int  = 10,
    session_start:int  = 0,
) -> pd.DataFrame:
    """
    Napoleon Value Layering: enter at structural extremes with UMS confirmation.

    Args:
        df:            Renko DataFrame with brick_up + standard indicators.
        nap_buy_thr:   Napoleon score threshold for long setup (score <= this).
        nap_sell_thr:  Napoleon score threshold for short setup (score >= this).
        req_ums:       Require UMS buy/sell signal to confirm entry.
        cooldown:      Minimum bricks between entries.
        session_start: UTC hour gate (0 = off, 13 = London+NY only).

    Returns:
        df with long_entry, long_exit, short_entry, short_exit bool columns.
    """
    # Warmup: Napoleon needs 25 bars (SMA25); UMS needs 60 bars (position_length)
    warmup = 80

    c = _get_or_build_cache().reindex(df.index)

    brick_up  = df["brick_up"].values
    hours     = df.index.hour
    n         = len(df)

    nap_score   = c["nap_score"].fillna(4.0).values.astype(float)
    ums_buy     = c["ums_buy"].fillna(False).values.astype(bool)
    ums_sell    = c["ums_sell"].fillna(False).values.astype(bool)

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

        score = nap_score[i]

        # ── LONG: value floor (low score) + UMS buy ──────────────────────────
        if up and score <= nap_buy_thr:
            ums_ok = (not req_ums) or ums_buy[i]
            if ums_ok:
                long_entry[i]  = True
                in_position    = True
                trade_dir      = 1
                last_trade_bar = i

        # ── SHORT: structural rejection (high score) + UMS sell ───────────────
        elif not up and score >= nap_sell_thr:
            ums_ok = (not req_ums) or ums_sell[i]
            if ums_ok:
                short_entry[i] = True
                in_position    = True
                trade_dir      = -1
                last_trade_bar = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
