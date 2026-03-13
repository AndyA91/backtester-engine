"""
GJ009: GJ008 + Session/ADX Threshold Sweep (GBPJPY)

Motivation: GJ008 live-mode tag analysis revealed:
  - Asian session (00–07 UTC) PF 34.44 >> Lon+NY (13–22 UTC) PF 11.94
  - GBPJPY session structure is OPPOSITE to EURUSD R008
  - Current sess=13 gate is cutting the best session (live-mode OANDA ADX ground truth)
  - Higher ADX threshold buckets show stronger PF: 5A[50-70) PF 38.47 vs 5A[25-30) PF 21.60

Sweep: session_start=[0, 7, 13] × adx_threshold=[25, 35, 40] × n/cd grid
  Total: 4 × 3 × 3 × 3 = 108 combinations per period

Note: Python uses HISTDATA ADX; TV uses OANDA ADX. These diverge — session ranking
  in Python may differ from TV tag analysis. TV verification is required after sweep.

IS:    2024-11-21 → 2025-09-30
OOS:   2025-10-01 → 2026-02-28
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from indicators.adx import calc_adx

RENKO_FILE      = "OANDA_GBPJPY, 1S renko 0.05.csv"
COMMISSION_PCT  = 0.005
INITIAL_CAPITAL = 150_000.0

DESCRIPTION = "GJ008 session × ADX threshold sweep — testing Asian session gate and higher ADX threshold"

HYPOTHESIS = (
    "GJ008 live-mode tag analysis (OANDA ADX) shows Asian session (00–07 UTC) PF 34.44 "
    "vs Lon+NY (13–22 UTC) PF 11.94 — opposite to EURUSD R008. Current sess=13 gate "
    "cuts the best GBPJPY session. Sweep session_start=[0,7,13] × adx_threshold=[25,35,40] "
    "to find optimal combination. Also testing higher ADX threshold: 5A[50+) PF 38+ in tag data."
)

PARAM_GRID = {
    "n_bricks":      [2, 3, 4, 5],
    "cooldown":      [10, 20, 30],
    "adx_threshold": [25, 35, 40],
    "vol_max":       [1.5],
    "session_start": [0, 7, 13],
}
# 4 × 3 × 3 × 1 × 3 = 108 combinations


# ── Module-level: load 5m candle data + compute ADX(14) once ──────────────────

def _load_candle_adx() -> pd.Series:
    """
    Load HISTDATA GBPJPY 5m candles, compute ADX(14) shifted 1 bar.
    Returns a Series indexed by DatetimeIndex (UTC, tz-naive, datetime64[ns]).
    """
    data_path = ROOT / "data" / "HISTDATA_GBPJPY_5m.csv"
    df = pd.read_csv(data_path)

    if df["time"].max() < 2_000_000:
        df["time"] = df["time"] * 1000

    df.index = pd.to_datetime(df["time"], unit="s")
    df = df[["open", "high", "low", "close", "Volume"]]
    df.columns = ["Open", "High", "Low", "Close", "Volume"]

    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)

    adx_result = calc_adx(df, di_period=14, adx_period=14)
    adx = pd.Series(adx_result["adx"], index=df.index).shift(1)

    adx.index = adx.index.astype("datetime64[ns]")
    return adx.sort_index()


_CANDLE_ADX = _load_candle_adx()


def generate_signals(
    df: pd.DataFrame,
    n_bricks: int = 3,
    cooldown: int = 10,
    adx_threshold: int = 25,
    vol_max: float = 1.5,
    session_start: int = 0,
) -> pd.DataFrame:
    """
    GJ007 signal logic with ADX gate, volume ratio gate, and session gate.

    ADX gate:     skip entry if 5m candle ADX < adx_threshold (0 = off).
    Volume gate:  skip entry if brick vol_ratio > vol_max (0 = off).
    Session gate: skip entry if UTC hour < session_start (0 = all hours).
                  0 = all, 7 = skip Asian (00-06), 13 = Lon+NY only (13-23).
    """
    n            = len(df)
    brick_up     = df["brick_up"].values
    vol_ratio    = df["vol_ratio"].values
    entry_hours  = df.index.hour

    # ── Align 5m candle ADX to each Renko bar ─────────────────────────────────
    renko_times  = df.index.astype("datetime64[ns]")
    adx_frame    = pd.DataFrame({"t": renko_times})
    candle_frame = _CANDLE_ADX.reset_index()
    candle_frame.columns = ["t_candle", "adx_val"]

    merged = pd.merge_asof(
        adx_frame.sort_values("t"),
        candle_frame,
        left_on="t",
        right_on="t_candle",
        direction="backward",
    )
    merged = merged.sort_index()
    candle_adx_vals = merged["adx_val"].values

    # ── Signal arrays ──────────────────────────────────────────────────────────
    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    in_position   = False
    trade_dir     = 0
    last_r001_bar = -999_999
    opp_streak    = 0

    warmup = max(n_bricks + 1, 30)

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # ── Exit: first opposing brick ─────────────────────────────────────────
        is_opposing = (trade_dir == 1 and not up) or (trade_dir == -1 and up)
        if in_position and is_opposing:
            opp_streak += 1
        elif in_position:
            opp_streak = 0

        confirmed_exit = in_position and opp_streak >= 1
        long_exit[i]   = confirmed_exit and not up
        short_exit[i]  = confirmed_exit and up

        if confirmed_exit:
            in_position = False
            trade_dir   = 0
            opp_streak  = 0

        if in_position:
            continue

        # ── ADX gate ──────────────────────────────────────────────────────────
        if adx_threshold > 0:
            adx_val = candle_adx_vals[i]
            if np.isnan(adx_val) or adx_val < adx_threshold:
                continue

        # ── Volume gate ───────────────────────────────────────────────────────
        if vol_max > 0:
            vr = vol_ratio[i]
            if np.isnan(vr) or vr > vol_max:
                continue

        # ── Session gate ──────────────────────────────────────────────────────
        if session_start > 0 and entry_hours[i] < session_start:
            continue

        # ── R002: N same-dir bricks, current opposes → counter-entry ───────────
        prev          = brick_up[i - n_bricks : i]
        prev_all_up   = bool(np.all(prev))
        prev_all_down = bool(not np.any(prev))

        if prev_all_up and not up:
            short_entry[i] = True
            in_position    = True
            trade_dir      = -1
            opp_streak     = 0

        elif prev_all_down and up:
            long_entry[i] = True
            in_position   = True
            trade_dir     = 1
            opp_streak    = 0

        # ── R001: N consecutive same-dir bricks → momentum entry ───────────────
        elif (i - last_r001_bar) >= cooldown:
            window   = brick_up[i - n_bricks + 1 : i + 1]
            all_up   = bool(np.all(window))
            all_down = bool(not np.any(window))

            if all_up:
                long_entry[i]  = True
                in_position    = True
                trade_dir      = 1
                last_r001_bar  = i
                opp_streak     = 0

            elif all_down:
                short_entry[i] = True
                in_position    = True
                trade_dir      = -1
                last_r001_bar  = i
                opp_streak     = 0

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
