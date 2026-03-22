"""R013: R007 + Choppiness Index Gate — Universal Regime Filter Study

The Choppiness Index (CHOP) measures whether price is trending or ranging by
comparing the sum of True Range to the total price range over N bars:
  CHOP < 38.2  → trending (directional, tight path)
  CHOP > 61.8  → choppy (random, wide path relative to displacement)

CHOP has never been used as a gate in any strategy. This study tests it as
a universal regime filter on the R007 base (R001+R002 combined), comparing:
  1. CHOP gate alone (no ADX)
  2. ADX gate alone (R008 baseline comparison)
  3. CHOP + ADX together (dual regime filter — are they complementary?)

The hypothesis is that CHOP measures a fundamentally different thing than ADX:
  - ADX = trend STRENGTH (how directional is the current move)
  - CHOP = trend EFFICIENCY (how much of the total movement was directional)

A market can have high ADX (strong directional move) but high CHOP too (lots
of back-and-forth within that move). Combining both should filter scenarios
that either misses alone.

CHOP is computed on Renko bricks (via add_renko_indicators, pre-shifted).
ADX is also Renko-native (not 5m candle ADX like R008).

Data: OANDA_EURUSD, 1S renko 0.0004.csv
IS:  2024-01-01 → 2025-09-30
OOS: 2025-10-01 → 2026-03-05
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from renko.data import load_renko_export
from renko.indicators import add_renko_indicators

DESCRIPTION = "R007 combined + Choppiness Index regime gate (CHOP vs ADX vs both)"

HYPOTHESIS = (
    "The Choppiness Index has never been tested as a gate despite being computed "
    "on every Renko DataFrame. CHOP < 38.2 means price is trending efficiently — "
    "this is a fundamentally different measure than ADX (strength vs efficiency). "
    "Testing CHOP alone, ADX alone (Renko-native), and CHOP+ADX together reveals "
    "whether they are redundant or complementary regime filters."
)

RENKO_FILE = "OANDA_EURUSD, 1S renko 0.0004.csv"

# ---------------------------------------------------------------------------
# PARAM_GRID
# ---------------------------------------------------------------------------
# chop_max:       maximum CHOP to allow entry (0=off; 38.2=classic trending, 50=moderate)
# adx_threshold:  minimum Renko ADX to allow entry (0=off; 20/25=standard)
# n_bricks:       R001/R002 brick count
# cooldown:       R001 cooldown
# session_start:  UTC hour gate
# vol_max:        volume ratio gate
PARAM_GRID = {
    "n_bricks":      [3, 5],
    "cooldown":      [10, 30],
    "chop_max":      [0, 38.2, 50, 61.8],
    "adx_threshold": [0, 20, 25],
    "session_start": [0, 13],
    "vol_max":       [0, 1.5],
}
# 2 × 2 × 4 × 3 × 2 × 2 = 192 combinations
# Groups for analysis:
#   CHOP-only:    adx=0, chop>0     → pure CHOP gate
#   ADX-only:     adx>0, chop=0     → R008-style baseline
#   Both:         adx>0, chop>0     → dual gate
#   Neither:      adx=0, chop=0     → R007 baseline


# ---------------------------------------------------------------------------
# Data cache
# ---------------------------------------------------------------------------

_DF_CACHE = None


def _get_data() -> pd.DataFrame:
    global _DF_CACHE
    if _DF_CACHE is not None:
        return _DF_CACHE
    df = load_renko_export(RENKO_FILE)
    add_renko_indicators(df)
    _DF_CACHE = df
    return df


# ---------------------------------------------------------------------------
# Signal generator
# ---------------------------------------------------------------------------

def generate_signals(
    df:            pd.DataFrame,
    n_bricks:      int   = 3,
    cooldown:      int   = 10,
    chop_max:      float = 0.0,
    adx_threshold: int   = 0,
    session_start: int   = 0,
    vol_max:       float = 0.0,
) -> pd.DataFrame:
    """
    R007 signal logic with CHOP and/or ADX regime gates.

    Args:
        df:            Renko DataFrame with brick_up + standard indicators.
        n_bricks:      Consecutive bricks for R001; lookback N for R002.
        cooldown:      Minimum bricks between R001 entries.
        chop_max:      Maximum CHOP to allow entry (0=off). Classic: 38.2.
        adx_threshold: Minimum Renko ADX to allow entry (0=off).
        session_start: UTC hour gate (0=off, 13=London+NY only).
        vol_max:       Maximum vol_ratio to allow entry (0=off).

    Returns:
        df with long_entry, long_exit, short_entry, short_exit bool columns.
    """
    c = _get_data().reindex(df.index)

    n         = len(df)
    brick_up  = df["brick_up"].values
    hours     = df.index.hour

    # Pre-computed indicators (all pre-shifted by add_renko_indicators)
    chop_vals = c["chop"].values
    adx_vals  = c["adx"].values
    vol_ratio = c["vol_ratio"].values

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    in_position   = False
    trade_dir     = 0
    last_r001_bar = -999_999

    warmup = max(n_bricks + 1, 50)

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # ── Exit: first opposing brick ────────────────────────────────────
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

        # ── Session gate ──────────────────────────────────────────────────
        if session_start > 0 and hours[i] < session_start:
            continue

        # ── CHOP gate ─────────────────────────────────────────────────────
        if chop_max > 0:
            chop_val = chop_vals[i]
            if np.isnan(chop_val) or chop_val > chop_max:
                continue

        # ── ADX gate ──────────────────────────────────────────────────────
        if adx_threshold > 0:
            adx_val = adx_vals[i]
            if np.isnan(adx_val) or adx_val < adx_threshold:
                continue

        # ── Volume gate ───────────────────────────────────────────────────
        if vol_max > 0:
            vr = vol_ratio[i]
            if np.isnan(vr) or vr > vol_max:
                continue

        # ── R002: N bricks before bar i all same dir, bar i opposes ──────
        prev          = brick_up[i - n_bricks : i]
        prev_all_up   = bool(np.all(prev))
        prev_all_down = bool(not np.any(prev))

        if prev_all_up and not up:
            short_entry[i] = True
            in_position    = True
            trade_dir      = -1

        elif prev_all_down and up:
            long_entry[i] = True
            in_position   = True
            trade_dir     = 1

        # ── R001: N consecutive same-direction bricks → momentum entry ───
        elif (i - last_r001_bar) >= cooldown:
            window   = brick_up[i - n_bricks + 1 : i + 1]
            all_up   = bool(np.all(window))
            all_down = bool(not np.any(window))

            if all_up:
                long_entry[i]  = True
                in_position    = True
                trade_dir      = 1
                last_r001_bar  = i

            elif all_down:
                short_entry[i] = True
                in_position    = True
                trade_dir      = -1
                last_r001_bar  = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
