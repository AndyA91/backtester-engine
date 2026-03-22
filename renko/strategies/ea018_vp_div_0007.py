"""EA018: EURAUD EA008-Formula Gates on 0.0007 Brick Size.

Applies the proven EA008 gate stack (VP POC side + MACD Divergence + Session)
to the new 0.0007 brick Renko file with extended IS history.

Gate logic (same as EA008 / EA002 proven formula):
  req_vp:   Price must be on the correct side of the Volume Profile POC.
            Long entries require price above POC; short entries below POC.
            Applies to both R001 and R002 entries. NaN-pass.
  req_div:  MACD bullish divergence required for R002 long entries;
            bearish divergence required for R002 short entries.
            NOT applied to R001 (trend-follow) entries. NaN-pass.
  session_start: UTC hour gate (0 = off, 13 = London+NY only).

Key insight from EA002/EA008: div-alone inflates IS PF but collapses OOS.
VP+div together lower IS PF but raise OOS PF — genuine structural filter.

IS:  2023-01-01 → 2025-09-30  (extended vs EA008's 2023-07-20)
OOS: 2025-10-01 → 2026-03-18
Benchmark: EA008 OOS PF 10.62 (0.0006 data)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from indicators.dgtrd.oscillators import oscillators_overlay
from indicators.dgtrd.volume_profile import volume_profile_pivot_anchored
from renko.data import load_renko_export
from renko.indicators import add_renko_indicators

DESCRIPTION = "EURAUD EA008 gates (VP+div+session) on 0.0007 brick size"

HYPOTHESIS = (
    "The EA008 formula (VP POC side + MACD div + session=13) produced OOS PF 10.62 "
    "on 0.0006 bricks. The larger 0.0007 brick reduces noise per signal: each brick "
    "represents a bigger committed price move, potentially making VP and divergence "
    "gates more reliable. Extended IS (Jan 2023) provides more structural data."
)

RENKO_FILE      = "OANDA_EURAUD, 1S renko 0.0007.csv"
COMMISSION_PCT  = 0.009
INITIAL_CAPITAL = 1000.0

PARAM_GRID = {
    "n_bricks":      [2, 3, 4, 5],
    "cooldown":      [10, 20, 30],
    "session_start": [0, 13],
    "req_vp":        [True, False],
    "req_div":       [True, False],
}
# 4 × 3 × 2 × 2 × 2 = 96 combos
# Single cache build (all indicator params fixed)


# ---------------------------------------------------------------------------
# Indicator cache — single build
# ---------------------------------------------------------------------------

_CACHE: dict = {}


def _get_or_build_cache() -> pd.DataFrame:
    if "data" in _CACHE:
        return _CACHE["data"]

    df = load_renko_export(RENKO_FILE)
    add_renko_indicators(df)

    # ── Volume Profile POC side gate (same params as EA008/EA002) ─────────────
    df = volume_profile_pivot_anchored(df, pvt_length=20, num_bins=25, va_pct=0.68)
    df["vp_above"] = df["vp_above_poc"].shift(1).values

    # ── MACD Divergence gate (same params as EA008/EA002) ─────────────────────
    df = oscillators_overlay(
        df, osc_type="MACD",
        macd_fast=12, macd_slow=26, macd_signal=9,
        prefix="osc_",
    )
    df["bull_div"] = df["osc_bull_div"].shift(1).values
    df["bear_div"] = df["osc_bear_div"].shift(1).values

    _CACHE["data"] = df
    return df


# ---------------------------------------------------------------------------
# Signal generator
# ---------------------------------------------------------------------------

def generate_signals(
    df:            pd.DataFrame,
    n_bricks:      int  = 5,
    cooldown:      int  = 30,
    session_start: int  = 13,
    req_vp:        bool = True,
    req_div:       bool = True,
) -> pd.DataFrame:
    """
    EA008 formula on 0.0007 bricks: R001+R002 combined with VP POC + MACD div + session gates.

    Gate application:
      VP gate:  all entries (R001 and R002). NaN = pass.
      Div gate: R002 (reversal) entries only. NaN/False = block if req_div.
      Session:  all entries.

    Args:
        df:            Renko DataFrame with brick_up + standard indicators.
        n_bricks:      N-brick run length for R001/R002 signal.
        cooldown:      Minimum bricks between R001 entries (R002 exempt).
        session_start: UTC hour gate (0 = off, 13 = London+NY only).
        req_vp:        Require VP POC side alignment.
        req_div:       Require MACD divergence for R002 entries.

    Returns:
        df with long_entry, long_exit, short_entry, short_exit bool columns.
    """
    warmup = 160  # VP pivot lookback needs ~100 bars; add margin

    c = _get_or_build_cache().reindex(df.index)

    n         = len(df)
    brick_up  = df["brick_up"].values
    hours     = df.index.hour

    vp_above  = c["vp_above"].values
    bull_div  = c["bull_div"].fillna(False).values.astype(bool)
    bear_div  = c["bear_div"].fillna(False).values.astype(bool)

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    in_position   = False
    trade_dir     = 0
    last_r001_bar = -999_999

    def vp_ok(is_long: bool) -> bool:
        """VP gate — NaN passes (no POC yet = allow entry)."""
        if not req_vp:
            return True
        v = vp_above[i]
        try:
            if np.isnan(float(v)):
                return True  # NaN-pass
        except (TypeError, ValueError):
            return True
        return bool(v) if is_long else not bool(v)

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # ── Exit: first opposing brick ─────────────────────────────────────────
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

        # ── Session gate ───────────────────────────────────────────────────────
        if session_start > 0 and hours[i] < session_start:
            continue

        # ── Candidate window ───────────────────────────────────────────────────
        window_r2     = brick_up[i - n_bricks : i]
        prev_all_up   = len(window_r2) == n_bricks and bool(np.all(window_r2))
        prev_all_down = len(window_r2) == n_bricks and bool(not np.any(window_r2))

        # ── R002 long: N down-bricks reversed to UP ────────────────────────────
        if prev_all_down and up:
            if vp_ok(True):
                if req_div and not bull_div[i]:
                    pass  # div required but not present — skip
                else:
                    long_entry[i]  = True
                    in_position    = True
                    trade_dir      = 1

        # ── R002 short: N up-bricks reversed to DOWN ──────────────────────────
        elif prev_all_up and not up:
            if vp_ok(False):
                if req_div and not bear_div[i]:
                    pass  # div required but not present — skip
                else:
                    short_entry[i] = True
                    in_position    = True
                    trade_dir      = -1

        # ── R001 trend continuation (with cooldown) ────────────────────────────
        elif (i - last_r001_bar) >= cooldown:
            window_r1 = brick_up[i - n_bricks + 1 : i + 1]
            if len(window_r1) == n_bricks:
                if bool(np.all(window_r1)):
                    if vp_ok(True):
                        long_entry[i]  = True
                        in_position    = True
                        trade_dir      = 1
                        last_r001_bar  = i
                elif bool(not np.any(window_r1)):
                    if vp_ok(False):
                        short_entry[i] = True
                        in_position    = True
                        trade_dir      = -1
                        last_r001_bar  = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
