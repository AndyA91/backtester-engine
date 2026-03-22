"""EA013: Adaptive ESCGO Cycle Follower — Adaptive Cycle Signal + HP Trend Filter

Cycle-adaptive trend-following strategy. The L3 Adaptive ESCGO fuses two
independent crossover signals — the ESCGO golden/death cross and the Banker
Fund Flow bull/bear cross — via a lookback-window fusion gate and an internal
cooldown. The HP Trend Line acts as a structural filter: only take ESCGO
signals that agree with the trend-line regime.

Signal logic:
  LONG:  bc_b_signal = True (ESCGO + BFF fused buy signal)
         AND hp_is_support = True (price regime is above trend line = support)
         AND current brick is UP

  SHORT: bc_s_signal = True (ESCGO + BFF fused sell signal)
         AND hp_is_support = False (price regime is below trend line = resistance)
         AND current brick is DOWN

Exit:
  Primary:   first opposing Renko brick (standard)
  Secondary: if use_escgo_exit = True, ALSO exit on bc_long_exit / bc_short_exit
             (BFF trend reversal signal from the ESCGO module)

Hypothesis: the Adaptive ESCGO detects cycle-phase turning points using Ehlers'
instantaneous-period algorithm — it adapts the lookback to the current dominant
market cycle. Aligning these adaptive cycle signals with the HP Trend Line
structural regime eliminates counter-trend entries in strongly trending markets.
The ESCGO's own exit signals provide an additional early-exit mechanism.

EURAUD IS:  2023-07-20 → 2025-09-30
EURAUD OOS: 2025-10-01 → 2026-03-17
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from indicators.HPotter.trend_line import calc_hp_trend_line
from indicators.blackcat1402.blackcat_l3_adaptive_escgo import calc_bc_l3_adaptive_escgo
from renko.data import load_renko_export
from renko.indicators import add_renko_indicators

DESCRIPTION = "EURAUD Adaptive ESCGO cycle signals filtered by HPotter Trend Line regime"

HYPOTHESIS = (
    "The Adaptive ESCGO auto-tunes its cycle-gravity window to the dominant "
    "Ehlers cycle length, producing buy/sell signals that are phase-aligned "
    "to the current market rhythm. The HP Trend Line filter prevents entering "
    "against the structural trend direction, converting a pure cycle-oscillator "
    "system into a regime-filtered trend follower."
)

RENKO_FILE      = "OANDA_EURAUD, 1S renko 0.0006.csv"
COMMISSION_PCT  = 0.009
INITIAL_CAPITAL = 1000.0

# ---------------------------------------------------------------------------
# PARAM_GRID
# ---------------------------------------------------------------------------
# tl_length:       HP Trend Line SMA period
# escgo_lookback:  ESCGO fusion lookback (how many bars each cross remains active)
# escgo_cooldown:  ESCGO internal min bars between same-type signals
# use_escgo_exit:  also exit on ESCGO's built-in BFF exit signal
# session_start:   UTC hour gate (0 = no gate, 13 = London+NY only)
PARAM_GRID = {
    "tl_length":      [15, 25, 50],
    "escgo_lookback": [3, 5],
    "escgo_cooldown": [5, 8],
    "use_escgo_exit": [True, False],
    "session_start":  [0, 13],
}
# Total: 3 × 2 × 2 × 2 × 2 = 96 combos
# Unique cache builds: 3 (tl_length) × 2 (escgo_lookback) × 2 (escgo_cooldown) = 12


# ---------------------------------------------------------------------------
# Indicator cache — keyed by (tl_length, escgo_lookback, escgo_cooldown)
# ---------------------------------------------------------------------------

_CACHE: dict = {}


def _get_or_build_cache(
    tl_length: int,
    escgo_lookback: int,
    escgo_cooldown: int,
) -> pd.DataFrame:
    key = (tl_length, escgo_lookback, escgo_cooldown)
    if key in _CACHE:
        return _CACHE[key]

    df = load_renko_export(RENKO_FILE)
    add_renko_indicators(df)

    # ── HP Trend Line (capitalized columns) ───────────────────────────────────
    df = calc_hp_trend_line(df, length=tl_length)
    df["hp_trend_line"] = df["hp_trend_line"].shift(1)
    df["hp_is_support"] = df["hp_is_support"].shift(1)

    # ── Adaptive ESCGO (expects lowercase ohlcv) ──────────────────────────────
    df_lc = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    esc = calc_bc_l3_adaptive_escgo(
        df_lc,
        lookback_period=escgo_lookback,
        signal_cooldown=escgo_cooldown,
    )

    # Shift all ESCGO outputs to prevent lookahead
    df["esc_b_signal"]   = esc["bc_b_signal"].shift(1).values
    df["esc_s_signal"]   = esc["bc_s_signal"].shift(1).values
    df["esc_long_exit"]  = esc["bc_long_exit"].shift(1).values
    df["esc_short_exit"] = esc["bc_short_exit"].shift(1).values

    _CACHE[key] = df
    return df


# ---------------------------------------------------------------------------
# Signal generator
# ---------------------------------------------------------------------------

def generate_signals(
    df: pd.DataFrame,
    tl_length:      int  = 25,
    escgo_lookback: int  = 5,
    escgo_cooldown: int  = 8,
    use_escgo_exit: bool = True,
    session_start:  int  = 0,
) -> pd.DataFrame:
    """
    Adaptive ESCGO Cycle Follower: fused ESCGO+BFF signals filtered by HP trend regime.

    Args:
        df:             Renko DataFrame with brick_up + standard indicators.
        tl_length:      HP Trend Line SMA period.
        escgo_lookback: Bars over which each ESCGO/BFF cross remains active.
        escgo_cooldown: Minimum bars between same-type fused signals.
        use_escgo_exit: Also exit positions on ESCGO's BFF-based exit signal.
        session_start:  UTC hour gate (0 = off, 13 = London+NY only).

    Returns:
        df with long_entry, long_exit, short_entry, short_exit bool columns.
    """
    # Warmup: HP Trend Line needs tl_length bars; ESCGO needs ~50 bars for cycle
    warmup = max(tl_length * 2 + 5, 60)

    c = _get_or_build_cache(tl_length, escgo_lookback, escgo_cooldown).reindex(df.index)

    brick_up  = df["brick_up"].values
    hours     = df.index.hour
    n         = len(df)

    hp_is_sup    = c["hp_is_support"].fillna(False).values.astype(bool)
    esc_b        = c["esc_b_signal"].fillna(False).values.astype(bool)
    esc_s        = c["esc_s_signal"].fillna(False).values.astype(bool)
    esc_lx       = c["esc_long_exit"].fillna(False).values.astype(bool)
    esc_sx       = c["esc_short_exit"].fillna(False).values.astype(bool)

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    in_position    = False
    trade_dir      = 0
    last_trade_bar = -999_999

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # ── Exit ──────────────────────────────────────────────────────────────
        if in_position:
            exit_long  = (trade_dir == 1  and not up) or \
                         (use_escgo_exit and trade_dir == 1  and bool(esc_lx[i]))
            exit_short = (trade_dir == -1 and up) or \
                         (use_escgo_exit and trade_dir == -1 and bool(esc_sx[i]))

            if exit_long:
                long_exit[i]  = True
                in_position   = False
                trade_dir     = 0
            elif exit_short:
                short_exit[i] = True
                in_position   = False
                trade_dir     = 0

        if in_position:
            continue

        # ── Session gate ──────────────────────────────────────────────────────
        if session_start > 0 and hours[i] < session_start:
            continue

        # ── LONG: ESCGO buy + support regime + UP brick ───────────────────────
        if up and esc_b[i] and hp_is_sup[i]:
            long_entry[i]  = True
            in_position    = True
            trade_dir      = 1
            last_trade_bar = i

        # ── SHORT: ESCGO sell + resistance regime + DOWN brick ────────────────
        elif not up and esc_s[i] and not hp_is_sup[i]:
            short_entry[i] = True
            in_position    = True
            trade_dir      = -1
            last_trade_bar = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
