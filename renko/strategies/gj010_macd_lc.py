"""GJ010: GJ008 + BC/FS Gate Refinement — macd_lc and fsb_strong (GBPJPY)

GBPJPY port of R012. Applies the BC sweep's top consensus gates to the
GJ008 base system.

BC sweep results (phase 1) — GBPJPY top configs:
  macd_lc  n=5 cd=30  OOS PF 28.76 avg across all n/cd  (+14.37 vs baseline 14.39)
  fsb_strong            OOS PF 30.08 avg                  (+15.70 vs baseline 14.39)
  Headline single: st / fsb_any / st_fsb n=5 cd=30 → OOS PF 48.75 / 48t
    (WARNING: these three are functionally identical under this parameter set —
    gate overlap, not three independent edges. See bc_sweep_findings.md §3.)
  Benchmark: GJ008 OOS PF 21.33 (n=5, cd=20, ADX=25, vol=1.5, sess=13)

Architecture:
  Base:  GJ007 (R001+R002 combined) + ADX(25) + vol_max(1.5) + session=13
  Gate A (use_macd_lc):   BC L3 MACD Wave Signal Pro
    Long:  bc_macd_state in {0, 3}  (histogram rising)
           AND bc_lc > 0            (SMA fan bullish)
    Short: bc_macd_state in {1, 2}  (histogram falling)
           AND bc_lc < 0            (SMA fan bearish)
  Gate B (use_fsb_strong): FS Balance
    Long:  regime == "STRONG_BUY"
    Short: regime == "STRONG_SELL"
  NaN-pass convention: NaN from any indicator → gate is waived.

Data:
  Renko:  OANDA_GBPJPY, 1S renko 0.05.csv   (~19,785 bricks, Nov 2024–Mar 2026)
  Candle: HISTDATA_GBPJPY_5m.csv            (Jan 2023–Feb 2026)

IS:   2024-11-21 → 2025-09-30   (matches GJ008)
OOS:  2025-10-01 → 2026-02-28   (matches GJ008)

GBPJPY calibration:
  Commission: ~0.005% (OANDA GBPJPY spread ~2 pips)
  Capital:    150,000 JPY ≈ $1,000 USD at ~150 JPY/USD

Run with:
  python renko/runner.py gj010_macd_lc --start 2024-11-21 --end 2025-09-30  (IS)
  python renko/runner.py gj010_macd_lc --start 2025-10-01 --end 2026-02-28  (OOS)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from indicators.adx import calc_adx
from indicators.fs_balance import calc_fs_balance
from indicators.blackcat1402.blackcat_l3_macd_wave_signal_pro import (
    calc_bc_l3_macd_wave_signal_pro,
)

RENKO_FILE      = "OANDA_GBPJPY, 1S renko 0.05.csv"
COMMISSION_PCT  = 0.005
INITIAL_CAPITAL = 150_000.0

DESCRIPTION = "GJ008 + macd_lc / fsb_strong BC gate refinement — GBPJPY consensus champions"

HYPOTHESIS = (
    "BC sweep phase 1 identified macd_lc as the strongest cross-instrument gate "
    "with GBPJPY avg OOS PF 28.76 vs baseline 14.39 (+100%). fsb_strong averaged "
    "30.08 OOS PF (+109%). These are two-instrument consensus winners (also strong "
    "on EURUSD). GJ010 refines the cd grid (10/15/20/25/30) and tests each gate "
    "individually and combined on the GJ008 base (ADX=25, vol=1.5, sess=13 fixed). "
    "n=[2..5] covers the full range — GJ sweep showed n=5 dominates but n=3/4 "
    "may benefit differently from the gate combo."
)

# ---------------------------------------------------------------------------
# Fixed base-gate parameters (same as GJ008 champion config)
# ---------------------------------------------------------------------------
ADX_THRESHOLD = 25
VOL_MAX       = 1.5
SESSION_START = 13

# ---------------------------------------------------------------------------
# PARAM_GRID
# ---------------------------------------------------------------------------
PARAM_GRID = {
    "n_bricks":       [2, 3, 4, 5],
    "cooldown":       [10, 15, 20, 25, 30],
    "use_macd_lc":    [True, False],
    "use_fsb_strong": [True, False],
}
# 4 × 5 × 2 × 2 = 80 combos
# (False, False) = GJ008 baseline — in-grid reference

# ---------------------------------------------------------------------------
# Module-level lazy cache
# ---------------------------------------------------------------------------

_RENKO_DF: pd.DataFrame | None = None
_ADX_VALS: np.ndarray   | None = None


def _ensure_loaded() -> None:
    global _RENKO_DF, _ADX_VALS
    if _RENKO_DF is not None:
        return

    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators

    # ── Renko + standard indicators ─────────────────────────────────────────
    df = load_renko_export(RENKO_FILE)
    add_renko_indicators(df)

    # ── FS Balance ───────────────────────────────────────────────────────────
    fb = calc_fs_balance(df)
    df["_fb_regime"] = pd.Series(fb["regime"], index=df.index).shift(1)

    # ── BC L3 MACD Wave Signal Pro ───────────────────────────────────────────
    df_lc = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low", "Close": "close",
    })
    df_macd = calc_bc_l3_macd_wave_signal_pro(df_lc)
    df["_bc_macd_state"] = df_macd["bc_macd_state"].shift(1).values
    df["_bc_lc"]         = df_macd["bc_lc"].shift(1).values

    _RENKO_DF = df

    # ── HISTDATA GBPJPY 5m candle ADX ───────────────────────────────────────
    data_path = ROOT / "data" / "HISTDATA_GBPJPY_5m.csv"
    df_c = pd.read_csv(data_path)

    # Defensive kilosecond check (GBPJPY file is clean but check anyway)
    if df_c["time"].max() < 2_000_000:
        df_c["time"] = df_c["time"] * 1000

    df_c.index = pd.to_datetime(df_c["time"], unit="s")
    df_c = df_c[["open", "high", "low", "close", "Volume"]]
    df_c.columns = ["Open", "High", "Low", "Close", "Volume"]
    df_c = df_c[~df_c.index.duplicated(keep="first")].sort_index()

    adx_result = calc_adx(df_c, di_period=14, adx_period=14)
    adx = pd.Series(adx_result["adx"], index=df_c.index).shift(1)
    adx.index = adx.index.astype("datetime64[ns]")
    adx = adx.sort_index()

    # Align candle ADX onto every Renko bar via merge_asof (backward fill)
    renko_times  = _RENKO_DF.index.astype("datetime64[ns]")
    adx_frame    = pd.DataFrame({"t": renko_times})
    candle_frame = adx.reset_index()
    candle_frame.columns = ["t_candle", "adx_val"]

    merged = pd.merge_asof(
        adx_frame.sort_values("t"),
        candle_frame,
        left_on="t", right_on="t_candle",
        direction="backward",
    ).sort_index()

    _ADX_VALS = merged["adx_val"].values


# ---------------------------------------------------------------------------
# Signal generator
# ---------------------------------------------------------------------------

def generate_signals(
    df: pd.DataFrame,
    n_bricks:       int  = 5,
    cooldown:       int  = 20,
    use_macd_lc:    bool = True,
    use_fsb_strong: bool = False,
) -> pd.DataFrame:
    """
    GJ008 base (GJ007 + ADX + vol + sess) with optional BC/FS gates.

    Args:
        df:             Full Renko DataFrame (brick_up + pre-shifted standard indicators).
        n_bricks:       N-brick run length for R001/R002 signal detection.
        cooldown:       Minimum bricks between R001 entries (R002 exempt).
        use_macd_lc:    Require BC MACD histogram direction AND LC fan direction.
        use_fsb_strong: Require FS Balance STRONG_BUY / STRONG_SELL regime.

    Returns:
        df with long_entry, long_exit, short_entry, short_exit bool columns.
    """
    _ensure_loaded()

    c        = _RENKO_DF.reindex(df.index)
    adx_vals = _ADX_VALS[_RENKO_DF.index.get_indexer(df.index, method="nearest")]

    n           = len(df)
    brick_up    = df["brick_up"].values
    vol_ratio   = c["vol_ratio"].values
    hours       = df.index.hour
    fb_regime   = c["_fb_regime"].values
    macd_state  = c["_bc_macd_state"].values
    bc_lc       = c["_bc_lc"].values

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    in_position   = False
    trade_dir     = 0
    last_r001_bar = -999_999
    warmup        = max(n_bricks + 1, 200)

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # ── Exit: first opposing brick ─────────────────────────────────────
        if in_position:
            is_opp        = (trade_dir == 1 and not up) or (trade_dir == -1 and up)
            long_exit[i]  = is_opp and trade_dir == 1
            short_exit[i] = is_opp and trade_dir == -1
            if is_opp:
                in_position = False
                trade_dir   = 0

        if in_position:
            continue

        # ── Base gates: ADX + vol + session ────────────────────────────────
        av = adx_vals[i]
        if np.isnan(av) or av < ADX_THRESHOLD:
            continue

        vr = vol_ratio[i]
        if np.isnan(vr) or vr > VOL_MAX:
            continue

        if hours[i] < SESSION_START:
            continue

        # ── R002 / R001 candidate direction ────────────────────────────────
        prev          = brick_up[i - n_bricks : i]
        prev_all_up   = bool(np.all(prev))
        prev_all_down = bool(not np.any(prev))

        if prev_all_up and not up:
            cand = -1; is_r002 = True
        elif prev_all_down and up:
            cand = 1;  is_r002 = True
        else:
            if (i - last_r001_bar) < cooldown:
                continue
            window   = brick_up[i - n_bricks + 1 : i + 1]
            all_up   = bool(np.all(window))
            all_down = bool(not np.any(window))
            if all_up:
                cand = 1;  is_r002 = False
            elif all_down:
                cand = -1; is_r002 = False
            else:
                continue

        is_long = (cand == 1)

        # ── Gate A: macd_lc (NaN → pass) ───────────────────────────────────
        if use_macd_lc:
            ms = macd_state[i]
            if not np.isnan(ms):
                ms_int  = int(ms)
                macd_ok = (ms_int in (0, 3)) if is_long else (ms_int in (1, 2))
                if not macd_ok:
                    continue
            lc = bc_lc[i]
            if not np.isnan(lc):
                lc_ok = (lc > 0) if is_long else (lc < 0)
                if not lc_ok:
                    continue

        # ── Gate B: fsb_strong (NaN → pass) ────────────────────────────────
        if use_fsb_strong:
            reg = fb_regime[i]
            if not pd.isna(reg):
                fsb_ok = (reg == "STRONG_BUY") if is_long else (reg == "STRONG_SELL")
                if not fsb_ok:
                    continue

        # ── Enter ───────────────────────────────────────────────────────────
        if is_long:
            long_entry[i] = True
        else:
            short_entry[i] = True
        in_position = True
        trade_dir   = cand
        if not is_r002:
            last_r001_bar = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
