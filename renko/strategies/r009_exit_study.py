"""
R009: R008 + Exit Strategy Study

R008 (ADX=25 + Vol=1.5 + Sess=13) uses first-opposing-brick exit.
Duration analysis of tagged IS trades showed:

  0–1h:   16t  WR   0%  PF  0.00  ← ALL losers (first-brick whipsaws)
  1–2h:   44t  WR  50%  PF  5.00
  2–4h:   66t  WR  67%  PF  8.53
  4–8h:   91t  WR  81%  PF 44.87
  8–16h:  75t  WR  84%  PF 44.43
  16h+:   79t  WR  90%  PF 302.99  ← winners average 14.6h hold

Hypothesis: Filtering fast exits (whipsaws) and letting winners run longer
improves OOS PF without curve-fitting.

Exit modes (exit_mode param):
  0 = first opposing brick (R008 baseline)
  1 = N-brick trailing stop: stay in until price reverses trail_n bricks
      from the furthest-favorable close reached since entry
  2 = min-hold: ignore exits until min_hold_bricks bricks have elapsed
      then revert to first-opposing-brick exit
  3 = combined: min_hold_bricks + trail_n trailing after that

IS/OOS same as R008: 2024-01-01→2025-09-30 / 2025-10-01→2026-02-28

Run with:
  python renko/runner.py r009_exit_study --start 2024-01-01 --end 2025-09-30
  python renko/runner.py r009_exit_study --start 2025-10-01 --end 2026-02-28
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from indicators.adx import calc_adx

DESCRIPTION = "R008 + exit strategy study (trail / min-hold)"

HYPOTHESIS = (
    "Duration analysis: 0-1h trades are 100% losers (whipsaws). "
    "Trades >4h have WR 81-90%. Filtering fast exits or requiring N "
    "opposing bricks before close should improve OOS PF."
)

PARAM_GRID = {
    "n_bricks":        [5],
    "cooldown":        [30],
    "adx_threshold":   [25],
    "vol_max":         [1.5],
    "session_start":   [13],
    "exit_mode":       [0, 1, 2, 3],
    "trail_n":         [2, 3],
    "min_hold_bricks": [3, 5, 8],
}
# 1×1×1×1×1×4×2×3 = 24 combinations
# exit_mode=0 ignores trail_n and min_hold_bricks (R008 baseline)
# exit_mode=1 uses trail_n only
# exit_mode=2 uses min_hold_bricks only
# exit_mode=3 uses both
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
    exit_mode: int = 0,
    trail_n: int = 2,
    min_hold_bricks: int = 5,
) -> pd.DataFrame:
    """
    R008 entry logic with swept exit strategies.

    exit_mode:
      0 = first opposing brick (R008 baseline)
      1 = N-brick trailing stop from furthest favorable close
      2 = min-hold: ignore exits for first min_hold_bricks bricks, then first-opposing
      3 = combined: min-hold then trailing
    """
    n             = len(df)
    brick_up      = df["brick_up"].values
    closes        = df["Close"].values
    vol_ratio     = df["vol_ratio"].values
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

    in_position    = False
    trade_dir      = 0
    last_r001_bar  = -999_999
    entry_bar      = -1          # bar index when trade was entered
    peak_close     = 0.0         # most favorable close since entry
    trail_count    = 0           # consecutive opposing bricks from peak

    warmup = max(n_bricks + 1, 30)

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # ── Exit logic ───────────────────────────────────────────────────────
        if in_position:
            bricks_held = i - entry_bar

            if exit_mode == 0:
                # R008 baseline: first opposing brick
                do_exit = (trade_dir == 1 and not up) or (trade_dir == -1 and up)

            elif exit_mode == 1:
                # N-brick trailing stop from furthest favorable close
                if trade_dir == 1:
                    if closes[i] > peak_close:
                        peak_close  = closes[i]
                        trail_count = 0
                    if not up:
                        trail_count += 1
                    else:
                        trail_count = 0
                    do_exit = trail_count >= trail_n
                else:
                    if closes[i] < peak_close:
                        peak_close  = closes[i]
                        trail_count = 0
                    if up:
                        trail_count += 1
                    else:
                        trail_count = 0
                    do_exit = trail_count >= trail_n

            elif exit_mode == 2:
                # Min-hold: ignore first opposing brick until min_hold_bricks elapsed
                if bricks_held < min_hold_bricks:
                    do_exit = False
                else:
                    do_exit = (trade_dir == 1 and not up) or (trade_dir == -1 and up)

            else:
                # exit_mode 3: min-hold then trailing
                if bricks_held < min_hold_bricks:
                    do_exit = False
                else:
                    if trade_dir == 1:
                        if closes[i] > peak_close:
                            peak_close  = closes[i]
                            trail_count = 0
                        if not up:
                            trail_count += 1
                        else:
                            trail_count = 0
                        do_exit = trail_count >= trail_n
                    else:
                        if closes[i] < peak_close:
                            peak_close  = closes[i]
                            trail_count = 0
                        if up:
                            trail_count += 1
                        else:
                            trail_count = 0
                        do_exit = trail_count >= trail_n

            if do_exit:
                long_exit[i]  = trade_dir == 1
                short_exit[i] = trade_dir == -1
                in_position   = False
                trade_dir     = 0

        if in_position:
            continue

        # ── Entry gates ──────────────────────────────────────────────────────
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

        # ── R002 ─────────────────────────────────────────────────────────────
        prev          = brick_up[i - n_bricks : i]
        prev_all_up   = bool(np.all(prev))
        prev_all_down = bool(not np.any(prev))

        if prev_all_up and not up:
            short_entry[i] = True
            in_position    = True
            trade_dir      = -1
            entry_bar      = i
            peak_close     = closes[i]
            trail_count    = 0

        elif prev_all_down and up:
            long_entry[i] = True
            in_position   = True
            trade_dir     = 1
            entry_bar     = i
            peak_close    = closes[i]
            trail_count   = 0

        # ── R001 ─────────────────────────────────────────────────────────────
        elif (i - last_r001_bar) >= cooldown:
            window   = brick_up[i - n_bricks + 1 : i + 1]
            all_up   = bool(np.all(window))
            all_down = bool(not np.any(window))

            if all_up:
                long_entry[i]  = True
                in_position    = True
                trade_dir      = 1
                last_r001_bar  = i
                entry_bar      = i
                peak_close     = closes[i]
                trail_count    = 0

            elif all_down:
                short_entry[i] = True
                in_position    = True
                trade_dir      = -1
                last_r001_bar  = i
                entry_bar      = i
                peak_close     = closes[i]
                trail_count    = 0

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
