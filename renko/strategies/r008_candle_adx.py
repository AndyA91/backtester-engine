"""
R008: R007 + 5m Candle ADX Gate

R007 (R001+R002 combined) confirmed by indicator analysis that all Renko-native
indicators have near-zero predictive power at entry (r < 0.08). The bad trades
are regime failures — choppy candle markets where Renko bricks happen to line up.

This strategy adds a 5m candle ADX(14) gate: only enter when the candle-space
ADX at entry is above adx_threshold. adx_threshold=0 = no gate (R007 baseline).

Data:
  Renko : OANDA_EURUSD, 1S renko 0.0004.csv  (always)
  Candle: HISTDATA_EURUSD_5m.csv             (Jan 2024 – Feb 2026)

IS:  2024-01-01 → 2025-09-30  (21 months — constrained by candle data start)
OOS: 2025-10-01 → 2026-02-28  (5 months — HISTDATA ends Feb 2026)

HISTDATA 5m timestamp fix:
  build_datasets.py stored timestamps in kiloseconds due to a pandas
  microsecond/nanosecond precision change. Detected and corrected here by
  multiplying by 1000 when max timestamp < 2_000_000.

Sanity check (adx_threshold=0, IS 2024-01-01→2025-09-30):
  Should reproduce R007 (n=3, cd=10) on same window within ±1%.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from indicators.adx import calc_adx

DESCRIPTION = "R007 combined (R001+R002) + 5m candle ADX(14) regime gate"

HYPOTHESIS = (
    "R007 indicator analysis showed all Renko-native indicators have r < 0.08 "
    "correlation with P&L at entry. Trade duration analysis showed bad trades "
    "(WR 30%, PF 1.7) are regime failures — choppy candle-space markets where "
    "Renko bricks align by noise. Candle ADX(14) measures real candle-space "
    "momentum; R004 (R001 + candle ADX) showed +39% PF improvement on a 3.5m "
    "window (IS only). R008 applies the same gate to R007 over the full 21-month "
    "period where HISTDATA 5m is available."
)

PARAM_GRID = {
    "n_bricks":      [2, 3, 4, 5],
    "cooldown":      [10, 20, 30],
    "adx_threshold": [25],
    "vol_max":       [1.5],
    "session_start": [0, 7, 13],
}
# 4 × 3 × 1 × 1 × 3 = 36 combinations
# adx_threshold=25, vol_max=1.5 fixed (both confirmed)
# session_start: 0=all hours, 7=skip Asian (00-07), 13=Lon+NY only (13-24)
# Exit modifications TESTED (2026-03-07) — both NEGATIVE, first-brick exit is optimal:
#   exit_vol_min:         n=5,cd=30 OOS: 0.0→PF 12.79 | 0.3→11.87 | 0.5→10.60 | 0.7→7.74
#   exit_confirm_bricks:  n=5,cd=30 OOS: 1→PF 12.79 | 2→PF 4.64 | 3→PF 3.92
#   Edge lives in duration asymmetry (winners 13.8h vs losers 6.5h) — holding through
#   initial opposition converts asymmetric wins to symmetric pain. Rejected.
# Run with:
#   python renko/runner.py r008_candle_adx --start 2024-01-01 --end 2025-09-30   (IS)
#   python renko/runner.py r008_candle_adx --start 2025-10-01 --end 2026-02-28   (OOS)


# ── Module-level: load 5m candle data + compute ADX(14) once ──────────────────
def _load_candle_adx() -> pd.Series:
    """
    Load HISTDATA 5m candles, fix broken timestamps, compute ADX(14) shifted 1 bar.
    Returns a Series indexed by DatetimeIndex (UTC, tz-naive, datetime64[ns]).
    """
    data_path = ROOT / "data" / "HISTDATA_EURUSD_5m.csv"
    df = pd.read_csv(data_path)

    # Timestamp fix: build_datasets.py stored kiloseconds (pandas µs/ns bug).
    # Real Unix seconds for Jan 2024 = ~1.704e9; file has ~1.704e6 → multiply by 1000.
    if df["time"].max() < 2_000_000:
        df["time"] = df["time"] * 1000

    df.index = pd.to_datetime(df["time"], unit="s")
    df = df[["open", "high", "low", "close", "Volume"]]
    df.columns = ["Open", "High", "Low", "Close", "Volume"]

    # Drop duplicate timestamps from the resampling quirk
    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)

    # ADX(14) — shift 1 bar to prevent lookahead (Pitfall #7 convention)
    adx_result = calc_adx(df, di_period=14, adx_period=14)
    adx = pd.Series(adx_result["adx"], index=df.index).shift(1)

    # Normalize to datetime64[ns] for merge_asof compatibility
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
    exit_vol_min: float = 0.0,
    exit_confirm_bricks: int = 1,
) -> pd.DataFrame:
    """
    R007 signal logic with ADX gate, volume ratio gate, session gate, and exit options.

    ADX gate:            skip entry if 5m candle ADX < adx_threshold (0 = off).
    Volume gate:         skip entry if brick vol_ratio > vol_max (0 = off).
                         vol_ratio = volume / EMA20(volume) — pre-shifted in df.
    Session gate:        skip entry if UTC hour < session_start (0 = all hours).
                         0 = all, 7 = skip Asian (00-06), 13 = Lon+NY only (13-23).
    Exit vol gate:       skip exit if opposing brick vol_ratio < exit_vol_min (0 = off).
                         TESTED 2026-03-07 — NEGATIVE at all thresholds. Rejected.
    Exit confirm bricks: require N consecutive opposing bricks before closing (1 = default).
                         2 or 3 filters single-brick noise reversals; stay in on first opp.
                         Counter resets to 0 on any same-direction brick while in position.

    Args:
        df:                   Renko DataFrame with brick_up bool column and vol_ratio.
        n_bricks:             Consecutive bricks for R001; lookback N for R002.
        cooldown:             Minimum bricks between R001 entries.
        adx_threshold:        Minimum 5m candle ADX to allow entry. 0 = no gate.
        vol_max:              Maximum vol_ratio to allow entry. 0 = no gate.
        session_start:        Minimum UTC hour to allow entry. 0 = no gate.
        exit_vol_min:         Minimum vol_ratio on opposing brick to trigger exit. 0 = no gate.
        exit_confirm_bricks:  Consecutive opposing bricks needed to exit. 1 = first brick exits.

    Returns:
        df with long_entry, long_exit, short_entry, short_exit (bool) columns.
    """
    n            = len(df)
    brick_up     = df["brick_up"].values
    vol_ratio    = df["vol_ratio"].values
    entry_hours  = df.index.hour

    # ── Align 5m candle ADX to each Renko bar ─────────────────────────────────
    renko_times = df.index.astype("datetime64[ns]")
    adx_frame = pd.DataFrame({"t": renko_times})
    candle_frame = _CANDLE_ADX.reset_index()
    candle_frame.columns = ["t_candle", "adx_val"]

    merged = pd.merge_asof(
        adx_frame.sort_values("t"),
        candle_frame,                      # already sorted (sorted at load)
        left_on="t",
        right_on="t_candle",
        direction="backward",
    )
    # Restore original Renko bar order (merge_asof sorts left)
    merged = merged.sort_index()
    candle_adx_vals = merged["adx_val"].values

    # ── Signal arrays ──────────────────────────────────────────────────────────
    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    in_position   = False
    trade_dir     = 0          # +1 long, -1 short
    last_r001_bar = -999_999
    opp_streak    = 0          # consecutive opposing bricks seen while in position

    warmup = max(n_bricks + 1, 30)

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # ── Exit signals (with N-brick confirmation) ───────────────────────────
        # exit_confirm_bricks=1 → first opposing brick exits (original behaviour).
        # exit_confirm_bricks=N → require N consecutive opposing bricks; streak
        # resets to 0 on any same-direction brick while in position.
        vr_exit  = vol_ratio[i]
        exit_ok  = exit_vol_min <= 0 or (not np.isnan(vr_exit) and vr_exit >= exit_vol_min)

        is_opposing = (trade_dir == 1 and not up) or (trade_dir == -1 and up)
        if in_position:
            if is_opposing and exit_ok:
                opp_streak += 1
            else:
                opp_streak = 0  # same-direction brick resets streak

        confirmed_exit = in_position and opp_streak >= exit_confirm_bricks
        long_exit[i]  = confirmed_exit and not up
        short_exit[i] = confirmed_exit and up

        # ── Update position state from exit ────────────────────────────────────
        if confirmed_exit:
            in_position = False
            trade_dir   = 0
            opp_streak  = 0

        if in_position:
            continue

        # ── ADX gate (applies to both R001 and R002) ───────────────────────────
        if adx_threshold > 0:
            adx_val = candle_adx_vals[i]
            if np.isnan(adx_val) or adx_val < adx_threshold:
                continue

        # ── Volume gate (applies to both R001 and R002) ────────────────────────
        if vol_max > 0:
            vr = vol_ratio[i]
            if np.isnan(vr) or vr > vol_max:
                continue

        # ── Session gate (applies to both R001 and R002) ───────────────────────
        if session_start > 0 and entry_hours[i] < session_start:
            continue

        # ── R002: N bricks before bar i all same dir, bar i opposes ───────────
        prev          = brick_up[i - n_bricks : i]
        prev_all_up   = bool(np.all(prev))
        prev_all_down = bool(not np.any(prev))

        if prev_all_up and not up:
            # N UP bricks then first DOWN → SHORT (R002 initiation)
            short_entry[i] = True
            in_position    = True
            trade_dir      = -1
            opp_streak     = 0

        elif prev_all_down and up:
            # N DOWN bricks then first UP → LONG (R002 initiation)
            long_entry[i] = True
            in_position   = True
            trade_dir     = 1
            opp_streak    = 0

        # ── R001: N consecutive same-direction bricks → momentum entry ─────────
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
