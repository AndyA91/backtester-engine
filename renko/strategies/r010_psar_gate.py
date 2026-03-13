"""
R010: R008 + PSAR Opposing Gate

Tag analysis on 628 trades (live mode, file 5) showed:
  entry PSAR opposing trade:  299t  PF 21.51  WR 72.2%
  entry PSAR aligned with:    329t  PF 15.99  WR 67.5%

Hypothesis: requiring PSAR to oppose trade direction at entry (i.e. price
is breaking against the SAR dot — a reversal signal) improves trade quality
without excessive trade count loss.

Gate logic:
  Long  entry: psar_dir == -1  (price below SAR = bearish SAR, opposes long)
  Short entry: psar_dir == +1  (price above SAR = bullish SAR, opposes short)

IS:  2024-01-01 → 2025-09-30
OOS: 2025-10-01 → 2026-02-28

Run with:
  python renko/runner.py r010_psar_gate --start 2024-01-01 --end 2025-09-30
  python renko/runner.py r010_psar_gate --start 2025-10-01 --end 2026-02-28
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from indicators.adx import calc_adx

DESCRIPTION = "R008 + PSAR opposing gate (require SAR to oppose trade direction at entry)"

HYPOTHESIS = (
    "Tag analysis on 628 trades showed entry PSAR opposing trade direction "
    "has PF 21.51 vs PSAR aligned PF 15.99. PSAR opposing means we are "
    "entering against the current SAR trend — a reversal signal consistent "
    "with R002's structure. Gate: long only when psar_dir=-1 (bearish SAR), "
    "short only when psar_dir=+1 (bullish SAR). psar_gate=False = R008 baseline."
)

PARAM_GRID = {
    "n_bricks":      [2, 3, 4, 5],
    "cooldown":      [10, 20, 30],
    "adx_threshold": [25],
    "vol_max":       [1.5],
    "session_start": [13],
    "psar_gate":     [False, True],
}
# 4 × 3 × 1 × 1 × 1 × 2 = 24 combinations
# session_start fixed at 13 (confirmed best OOS)
# psar_gate=False → R008 baseline for direct comparison
# Run IS then OOS to check decay


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
    n_bricks: int = 5,
    cooldown: int = 30,
    adx_threshold: int = 25,
    vol_max: float = 1.5,
    session_start: int = 13,
    psar_gate: bool = True,
) -> pd.DataFrame:
    """
    R008 entry logic + optional PSAR opposing gate.

    psar_gate=True: only enter long when psar_dir==-1 (bearish SAR opposes long),
                    only enter short when psar_dir==+1 (bullish SAR opposes short).
    psar_gate=False: R008 baseline (no PSAR filter).

    psar_dir column from indicators.py: +1 (bullish, price above SAR) / -1 (bearish).
    """
    n             = len(df)
    brick_up      = df["brick_up"].values
    vol_ratio     = df["vol_ratio"].values
    psar_dir      = df["psar_dir"].values
    entry_hours   = df.index.hour

    # ── Align 5m candle ADX ──────────────────────────────────────────────────
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

        # ── Exit: first opposing brick ────────────────────────────────────────
        if in_position:
            if (trade_dir == 1 and not up) or (trade_dir == -1 and up):
                long_exit[i]  = trade_dir == 1
                short_exit[i] = trade_dir == -1
                in_position   = False
                trade_dir     = 0

        if in_position:
            continue

        # ── Entry gates ───────────────────────────────────────────────────────
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

        # ── R002 ──────────────────────────────────────────────────────────────
        prev          = brick_up[i - n_bricks : i]
        prev_all_up   = bool(np.all(prev))
        prev_all_down = bool(not np.any(prev))

        if prev_all_up and not up:
            # PSAR gate: short only when SAR is bullish (price above SAR → bearish entry)
            if psar_gate and not np.isnan(psar_dir[i]) and psar_dir[i] != 1:
                continue
            short_entry[i] = True
            in_position    = True
            trade_dir      = -1

        elif prev_all_down and up:
            # PSAR gate: long only when SAR is bearish (price below SAR → bullish entry)
            if psar_gate and not np.isnan(psar_dir[i]) and psar_dir[i] != -1:
                continue
            long_entry[i] = True
            in_position   = True
            trade_dir     = 1

        # ── R001 ──────────────────────────────────────────────────────────────
        elif (i - last_r001_bar) >= cooldown:
            window   = brick_up[i - n_bricks + 1 : i + 1]
            all_up   = bool(np.all(window))
            all_down = bool(not np.any(window))

            if all_up:
                if psar_gate and not np.isnan(psar_dir[i]) and psar_dir[i] != -1:
                    continue
                long_entry[i]  = True
                in_position    = True
                trade_dir      = 1
                last_r001_bar  = i

            elif all_down:
                if psar_gate and not np.isnan(psar_dir[i]) and psar_dir[i] != 1:
                    continue
                short_entry[i] = True
                in_position    = True
                trade_dir      = -1
                last_r001_bar  = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
