"""BTC001: Fisher Transform Cross + ADX Gate — BTCUSD Renko 150

Fisher Transform converts price into a Gaussian distribution with sharp turning
points. On Renko bricks, Fisher crosses are crisp because each brick is a fixed
price move — no noise from wicks/dojis. The signal line is simply the previous
bar's Fisher value, so a cross = one-bar directional shift.

Entry logic:
  LONG:  Fisher crosses above signal (bullish reversal)
         AND brick_up is True (current brick confirms direction)
         AND ADX > adx_threshold (trending regime, not chop)
         AND cooldown satisfied

  SHORT: Fisher crosses below signal (bearish reversal)
         AND brick_up is False (current brick confirms direction)
         AND ADX > adx_threshold
         AND cooldown satisfied

Exit: first opposing Renko brick (standard).

Optional gates:
  - session_start: UTC hour gate (0=off, 13=London+NY only)
  - vol_max: volume spike filter (0=off)
  - psar_gate: require PSAR direction to oppose trade (R010's finding:
    PSAR-opposing entries PF 21.51 vs aligned 15.99 on EURUSD)

Data: OANDA_BTCUSD.SPOT.US, 1S renko 150.csv
  20,017 bricks: 2024-06-03 → 2026-03-21

IS:  2024-06-03 → 2025-09-30
OOS: 2025-10-01 → 2026-03-21
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from indicators.fisher_transform import calc_fisher_transform
from renko.data import load_renko_export
from renko.indicators import add_renko_indicators

DESCRIPTION = "BTCUSD Fisher Transform cross entry with Renko ADX regime gate"

HYPOTHESIS = (
    "Fisher Transform was #1 on BTCUSD renko in our signal showdown (PF 2.02, "
    "-10% DD). On Renko bricks, Fisher crosses are exceptionally clean because "
    "each brick is a uniform $150 move — no wick noise. Adding an ADX gate "
    "(proven +101% OOS PF on EURUSD R008) should filter the choppy-regime false "
    "signals that hurt Fisher's raw performance. PSAR opposing gate (proven PF "
    "21.51 vs 15.99 on EURUSD R010) adds a counter-trend confirmation layer."
)

RENKO_FILE      = "OANDA_BTCUSD.SPOT.US, 1S renko 150.csv"
COMMISSION_PCT  = 0.009
INITIAL_CAPITAL = 1000.0

# ---------------------------------------------------------------------------
# PARAM_GRID
# ---------------------------------------------------------------------------
PARAM_GRID = {
    "fisher_period":  [8, 10, 13],
    "adx_threshold":  [0, 20, 25],
    "cooldown":       [5, 10, 20],
    "session_start":  [0],
    "vol_max":        [0],
    "psar_gate":      [False, True],
    "req_brick_confirm": [True, False],
}
# 3 × 3 × 3 × 1 × 1 × 2 × 2 = 108 combinations


# ---------------------------------------------------------------------------
# Indicator cache — keyed by fisher_period
# ---------------------------------------------------------------------------

_CACHE: dict = {}


def _get_or_build_cache(fisher_period: int) -> pd.DataFrame:
    if fisher_period in _CACHE:
        return _CACHE[fisher_period]

    df = load_renko_export(RENKO_FILE)
    add_renko_indicators(df)

    # ── Fisher Transform ────────────────────────────────────────────────────
    ft = calc_fisher_transform(df, period=fisher_period)
    fisher = ft["fisher"]
    signal = ft["signal"]

    # Detect crosses (shift by 1 for lookahead safety is handled by the
    # Fisher indicator itself — signal[i] = fisher[i-1])
    n = len(df)
    fisher_bull_cross = np.zeros(n, dtype=bool)
    fisher_bear_cross = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if np.isnan(fisher[i]) or np.isnan(signal[i]):
            continue
        if np.isnan(fisher[i-1]) or np.isnan(signal[i-1]):
            continue
        fisher_bull_cross[i] = fisher[i] > signal[i] and fisher[i-1] <= signal[i-1]
        fisher_bear_cross[i] = fisher[i] < signal[i] and fisher[i-1] >= signal[i-1]

    df["fisher_bull"] = fisher_bull_cross
    df["fisher_bear"] = fisher_bear_cross

    _CACHE[fisher_period] = df
    return df


# ---------------------------------------------------------------------------
# Signal generator
# ---------------------------------------------------------------------------

def generate_signals(
    df:                pd.DataFrame,
    fisher_period:     int  = 10,
    adx_threshold:     int  = 20,
    cooldown:          int  = 10,
    session_start:     int  = 0,
    vol_max:           float = 0.0,
    psar_gate:         bool = False,
    req_brick_confirm: bool = True,
) -> pd.DataFrame:
    """
    Fisher Transform cross entries with ADX regime gate.

    Args:
        df:                Renko DataFrame with brick_up + standard indicators.
        fisher_period:     Fisher Transform lookback (default 10).
        adx_threshold:     Minimum Renko ADX to allow entry (0=off).
        cooldown:          Minimum bricks between entries.
        session_start:     UTC hour gate (0=off, 13=London+NY only).
        vol_max:           Maximum vol_ratio for entry (0=off).
        psar_gate:         Require PSAR direction to oppose trade direction.
        req_brick_confirm: Require brick direction to match Fisher signal.

    Returns:
        df with long_entry, long_exit, short_entry, short_exit bool columns.
    """
    warmup = max(fisher_period + 5, 50)

    c = _get_or_build_cache(fisher_period).reindex(df.index)

    brick_up     = df["brick_up"].values
    hours        = df.index.hour
    n            = len(df)

    fisher_bull  = c["fisher_bull"].fillna(False).values.astype(bool)
    fisher_bear  = c["fisher_bear"].fillna(False).values.astype(bool)

    # Pre-computed indicators from add_renko_indicators (all pre-shifted)
    adx_vals     = c["adx"].values
    vol_ratio    = c["vol_ratio"].values
    psar_dir     = c["psar_dir"].values

    long_entry   = np.zeros(n, dtype=bool)
    long_exit    = np.zeros(n, dtype=bool)
    short_entry  = np.zeros(n, dtype=bool)
    short_exit   = np.zeros(n, dtype=bool)

    in_position    = False
    trade_dir      = 0
    last_trade_bar = -999_999

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

        # ── Cooldown ──────────────────────────────────────────────────────
        if (i - last_trade_bar) < cooldown:
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

        # ── LONG: Fisher bullish cross ────────────────────────────────────
        if fisher_bull[i]:
            if req_brick_confirm and not up:
                pass  # brick doesn't confirm — skip
            elif psar_gate and not np.isnan(psar_dir[i]) and psar_dir[i] != -1:
                pass  # PSAR not bearish (opposing) — skip
            else:
                long_entry[i]  = True
                in_position    = True
                trade_dir      = 1
                last_trade_bar = i

        # ── SHORT: Fisher bearish cross ───────────────────────────────────
        elif fisher_bear[i]:
            if req_brick_confirm and up:
                pass  # brick doesn't confirm — skip
            elif psar_gate and not np.isnan(psar_dir[i]) and psar_dir[i] != 1:
                pass  # PSAR not bullish (opposing) — skip
            else:
                short_entry[i] = True
                in_position    = True
                trade_dir      = -1
                last_trade_bar = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
