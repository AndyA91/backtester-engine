"""
R011: R008 logic optimized for 0.0005 brick size

The 0.0005 Renko file shows different characteristics vs 0.0004:
  - sess=7 dominates IS (not sess=13)
  - IS PF lower (~12 vs 15) with higher IS→OOS decay (~42% vs 15%)
  - ADX=25/vol=1.5 may not be optimal at this brick resolution

This sweep widens the gate search to find the best parameter set
for the 0.0005 brick size.

IS:  2024-01-01 → 2025-09-30  (constrained by HISTDATA 5m)
OOS: 2025-10-01 → 2026-02-28

Run with:
  python renko/runner.py r011_0005_optimize --start 2024-01-01 --end 2025-09-30 --renko "OANDA_EURUSD, 1S renko 0.0005.csv"
  python renko/runner.py r011_0005_optimize --start 2025-10-01 --end 2026-02-28 --renko "OANDA_EURUSD, 1S renko 0.0005.csv"
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from indicators.adx import calc_adx

DESCRIPTION = "R008 logic parameter sweep optimized for 0.0005 brick size"

HYPOTHESIS = (
    "0.0005 bricks are coarser than 0.0004 — fewer bricks per day, larger "
    "price moves per brick. The sess=13 gate that was optimal on 0.0004 may "
    "not hold. Sweeping adx_threshold [0,20,25,30], vol_max [1.0,1.5,2.0], "
    "and session_start [0,7,13] to find the best gate combination."
)

PARAM_GRID = {
    "n_bricks":      [2, 3, 4, 5],
    "cooldown":      [10, 20, 30],
    "adx_threshold": [0, 20, 25, 30],
    "vol_max":       [1.0, 1.5, 2.0],
    "session_start": [0, 7, 13],
}
# 4 × 3 × 4 × 3 × 3 = 432 combinations


# ── Module-level: load 5m candle ADX once ────────────────────────────────────
def _load_candle_adx() -> pd.Series:
    data_path = ROOT / "data" / "HISTDATA_EURUSD_5m.csv"
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
    session_start: int = 7,
) -> pd.DataFrame:
    n             = len(df)
    brick_up      = df["brick_up"].values
    vol_ratio     = df["vol_ratio"].values
    entry_hours   = df.index.hour

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
    ).sort_index()
    candle_adx_vals = merged["adx_val"].values

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    in_position   = False
    trade_dir     = 0
    last_r001_bar = -999_999

    warmup = max(n_bricks + 1, 30)

    for i in range(warmup, n):
        up = bool(brick_up[i])

        if in_position:
            if (trade_dir == 1 and not up) or (trade_dir == -1 and up):
                long_exit[i]  = trade_dir == 1
                short_exit[i] = trade_dir == -1
                in_position   = False
                trade_dir     = 0

        if in_position:
            continue

        if adx_threshold > 0:
            av = candle_adx_vals[i]
            if np.isnan(av) or av < adx_threshold:
                continue
        if vol_max > 0:
            vr = vol_ratio[i]
            if np.isnan(vr) or vr > vol_max:
                continue
        if session_start > 0 and entry_hours[i] < session_start:
            continue

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
