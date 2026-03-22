"""EA004: Band Runner — Raff Regression Mean-Reversion

Mean-reversion strategy triggered when Renko brick closes extend outside the
Raff Regression Channel bands and the price reversal begins (opposing brick).

Signal logic:
  LONG:  Last `lookback` brick closes were ALL below rrc_lower (oversold extension)
         AND current brick is UP (first reversal) → buy mean-reversion
  SHORT: Last `lookback` brick closes were ALL above rrc_upper (overbought extension)
         AND current brick is DOWN (first reversal) → sell mean-reversion

Exit: first opposing Renko brick (standard).

Hypothesis: Renko bricks that extend beyond the Raff channel bounds represent
exhaustion runs. The first opposing brick after a sustained band violation marks
the start of mean-reversion back to the regression midline.

Only the Raff Regression Channel indicator is required (no DO or VP warmup),
so this strategy has a short warmup (~100 bricks) and fast cache build.

EURAUD IS:  2023-07-20 → 2025-09-30
EURAUD OOS: 2025-10-01 → 2026-03-17
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from indicators.dgtrd.raff_regression import raff_regression_channel
from renko.data import load_renko_export
from renko.indicators import add_renko_indicators

DESCRIPTION = "EURAUD Raff Band Runner — mean-reversion from channel extremes"

HYPOTHESIS = (
    "Renko closes below rrc_lower / above rrc_upper mark exhaustion runs. "
    "The first opposing brick after N consecutive outside-band closes is the "
    "mean-reversion entry back toward the regression midline."
)

RENKO_FILE      = "OANDA_EURAUD, 1S renko 0.0006.csv"
COMMISSION_PCT  = 0.009
INITIAL_CAPITAL = 1000.0

# ---------------------------------------------------------------------------
# PARAM_GRID
# ---------------------------------------------------------------------------
# lookback:      consecutive bricks whose closes must be outside the channel band
# cooldown:      minimum bricks between entries (prevents rapid re-entry)
# session_start: UTC hour gate (0 = no gate, 13 = London+NY only)
PARAM_GRID = {
    "lookback":      [1, 2, 3],
    "cooldown":      [0, 10, 20, 30],
    "session_start": [0, 13],
}


# ---------------------------------------------------------------------------
# Indicator cache  (built once at import time)
# ---------------------------------------------------------------------------

def _build_indicator_cache() -> pd.DataFrame:
    df = load_renko_export(RENKO_FILE)
    add_renko_indicators(df)

    # Raff Regression Channel — length=50 LRC, raff_length=100 channel window
    df = raff_regression_channel(df, source_col="Close", length=50, raff_length=100)
    # Shift outputs so they're safe to read at [i] in the signal loop
    df["rrc_upper"] = df["rrc_upper"].shift(1)
    df["rrc_lower"] = df["rrc_lower"].shift(1)

    return df


_CACHE = _build_indicator_cache()

# raff_length=100 is the longest warmup; add buffer
_WARMUP = 130


# ---------------------------------------------------------------------------
# Signal generator
# ---------------------------------------------------------------------------

def generate_signals(
    df: pd.DataFrame,
    lookback:      int = 1,
    cooldown:      int = 10,
    session_start: int = 0,
) -> pd.DataFrame:
    """
    Mean-reversion entries from Raff Regression Channel band violations.

    Scans `lookback` bricks before current bar. If ALL had closes outside
    a channel band AND the current brick reverses direction → entry.

    Args:
        df:            Renko DataFrame with brick_up bool + OHLCV.
        lookback:      Number of consecutive outside-band bricks required.
        cooldown:      Minimum bars between entries.
        session_start: UTC hour gate (0 = disabled).

    Returns:
        df with long_entry, long_exit, short_entry, short_exit bool columns.
    """
    c = _CACHE.reindex(df.index)

    rrc_upper = c["rrc_upper"].values
    rrc_lower = c["rrc_lower"].values
    close     = df["Close"].values
    brick_up  = df["brick_up"].values
    hours     = df.index.hour
    n         = len(df)

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    in_position   = False
    trade_dir     = 0
    last_trade_bar = -999_999

    for i in range(_WARMUP, n):
        up = bool(brick_up[i])

        # ── Exit: first opposing brick ─────────────────────────────────────
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

        # ── Session gate ───────────────────────────────────────────────────
        if session_start > 0 and hours[i] < session_start:
            continue

        # ── Cooldown check ─────────────────────────────────────────────────
        if cooldown > 0 and (i - last_trade_bar) < cooldown:
            continue

        # ── Band violation check over last `lookback` bricks ──────────────
        if i < lookback:
            continue

        window_close = close    [i - lookback : i]
        window_upper = rrc_upper[i - lookback : i]
        window_lower = rrc_lower[i - lookback : i]

        # Skip if channel is not yet computed
        if np.any(np.isnan(window_upper)) or np.any(np.isnan(window_lower)):
            continue

        above_upper = bool(np.all(window_close > window_upper))
        below_lower = bool(np.all(window_close < window_lower))

        # ── Entry ──────────────────────────────────────────────────────────
        # LONG: all lookback closes were below lower band AND current brick is UP
        if below_lower and up:
            long_entry[i]  = True
            in_position    = True
            trade_dir      = 1
            last_trade_bar = i

        # SHORT: all lookback closes were above upper band AND current brick is DOWN
        elif above_upper and not up:
            short_entry[i] = True
            in_position    = True
            trade_dir      = -1
            last_trade_bar = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
