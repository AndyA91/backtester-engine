"""GJ011: GJ008 Base + Phase 4 STO/TSO Gate + Phase 3 MACD-LC Gate (GBPJPY)

Tests the Phase 4 bc_master_sweep_v2 top gate candidates on the proper GJ008
base (R007 + ADX(25) + vol(1.5) + session=13). Phase 4 swept these gates on
the simpler R007+session base; GJ011 confirms whether the edge survives on the
stronger GJ008 base and how it interacts with macd_lc.

Phase 4 results (simpler base, avg OOS PF):
  sto_tso:   32.82  (beat v1 macd_lc 28.76 and fsb_strong 30.08)
  tso_pink:  30.16
  sto_reg:   28.52
  Best single: sto_reg n=5,cd=30 OOS PF 48.96
Benchmark: GJ008 OOS PF 21.33 (n=5, cd=20, ADX=25, vol=1.5, sess=13)

Architecture:
  Base:  GJ007 (R001+R002) + ADX(25) + vol_max(1.5) + session=13 (fixed)
  Gate A (use_sto_tso):  BC L1 STO + TSO combined
    Long:  bc_sto_main_force > bc_sto_life_line  (STO bullish)
           AND bc_tso_pink_hist == True           (TSO bullish)
    Short: bc_sto_main_force < bc_sto_life_line  (STO bearish)
           AND bc_tso_pink_hist == False          (TSO bearish)
    NaN-pass on either indicator.
  Gate B (use_macd_lc):  BC L3 MACD Wave Signal Pro (same as GJ010)
    Long:  bc_macd_state in {0,3} AND bc_lc > 0
    Short: bc_macd_state in {1,2} AND bc_lc < 0
    NaN-pass.
  (False, False) = GJ008 baseline in-grid reference.

Data:
  Renko:  OANDA_GBPJPY, 1S renko 0.05.csv  (~19,785 bricks, Nov 2024-Mar 2026)
  Candle: HISTDATA_GBPJPY_5m.csv            (Jan 2023-Feb 2026)

IS:  2024-11-21 -> 2025-09-30
OOS: 2025-10-01 -> 2026-02-28
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
from indicators.blackcat1402.bc_l1_swing_trade_oscillator import (
    calc_bc_swing_trade_oscillator,
)
from indicators.blackcat1402.bc_l1_trend_swing_oscillator import (
    calc_bc_trend_swing_oscillator,
)

RENKO_FILE      = "OANDA_GBPJPY, 1S renko 0.05.csv"
COMMISSION_PCT  = 0.005
INITIAL_CAPITAL = 150_000.0

DESCRIPTION = "GJ008 + Phase4 sto_tso gate / Phase3 macd_lc gate — GBPJPY gate cross-validation"

HYPOTHESIS = (
    "Phase 4 bc_master_sweep_v2 showed sto_tso averaging OOS PF 32.82 on GBPJPY "
    "(vs benchmark 21.33) on the simple R007+session base. GJ011 tests whether this "
    "edge survives on the stronger GJ008 base (R007+ADX+vol+sess) and whether "
    "combining sto_tso with macd_lc (Phase 3 winner) compounds or conflicts. "
    "(False, False) = GJ008 baseline in-grid."
)

# ---------------------------------------------------------------------------
# Fixed base-gate parameters (same as GJ008/GJ010)
# ---------------------------------------------------------------------------
ADX_THRESHOLD = 25
VOL_MAX       = 1.5
SESSION_START = 13

# ---------------------------------------------------------------------------
# PARAM_GRID
# ---------------------------------------------------------------------------
PARAM_GRID = {
    "n_bricks":    [2, 3, 4, 5],
    "cooldown":    [10, 20, 30],
    "use_sto_tso": [True, False],   # Phase 4 winner: STO regime AND TSO pink
    "use_macd_lc": [True, False],   # Phase 3 winner: MACD rising AND LC positive
}
# 4 × 3 × 2 × 2 = 48 combos
# (False, False) = GJ008 baseline in-grid reference

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

    # ── BC L3 MACD Wave Signal Pro (same as GJ010) ───────────────────────────
    df_lc = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low", "Close": "close",
    })
    df_macd = calc_bc_l3_macd_wave_signal_pro(df_lc)
    df["_bc_macd_state"] = df_macd["bc_macd_state"].shift(1).values
    df["_bc_lc"]         = df_macd["bc_lc"].shift(1).values

    # ── BC L1 Swing Trade Oscillator ─────────────────────────────────────────
    sto = calc_bc_swing_trade_oscillator(df)
    df["_bc_sto_mf"] = sto["bc_sto_main_force"].shift(1).values
    df["_bc_sto_ll"] = sto["bc_sto_life_line"].shift(1).values

    # ── BC L1 Trend Swing Oscillator ─────────────────────────────────────────
    tso = calc_bc_trend_swing_oscillator(df)
    df["_bc_tso_pink"] = tso["bc_tso_pink_hist"].shift(1).values

    _RENKO_DF = df

    # ── HISTDATA GBPJPY 5m candle ADX ───────────────────────────────────────
    data_path = ROOT / "data" / "HISTDATA_GBPJPY_5m.csv"
    df_c = pd.read_csv(data_path)

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
    df:          pd.DataFrame,
    n_bricks:    int  = 5,
    cooldown:    int  = 20,
    use_sto_tso: bool = True,
    use_macd_lc: bool = False,
) -> pd.DataFrame:
    """
    GJ008 base (GJ007 + ADX + vol + sess) with optional sto_tso and macd_lc gates.

    Args:
        df:          Full Renko DataFrame (brick_up + pre-shifted standard indicators).
        n_bricks:    N-brick run length for R001/R002 signal detection.
        cooldown:    Minimum bricks between R001 entries (R002 exempt).
        use_sto_tso: Require STO bullish/bearish AND TSO pink/non-pink.
        use_macd_lc: Require BC MACD histogram direction AND LC fan direction.

    Returns:
        df with long_entry, long_exit, short_entry, short_exit bool columns.
    """
    _ensure_loaded()

    c        = _RENKO_DF.reindex(df.index)
    adx_vals = _ADX_VALS[_RENKO_DF.index.get_indexer(df.index, method="nearest")]

    n          = len(df)
    brick_up   = df["brick_up"].values
    vol_ratio  = c["vol_ratio"].values
    hours      = df.index.hour
    macd_state = c["_bc_macd_state"].values
    bc_lc      = c["_bc_lc"].values
    sto_mf     = c["_bc_sto_mf"].values
    sto_ll     = c["_bc_sto_ll"].values
    tso_pink   = c["_bc_tso_pink"].values

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

        # ── Gate A: sto_tso — STO regime AND TSO pink (NaN -> pass) ────────
        if use_sto_tso:
            mf_v = sto_mf[i]
            ll_v = sto_ll[i]
            if not (np.isnan(mf_v) or np.isnan(ll_v)):
                sto_ok = (mf_v > ll_v) if is_long else (mf_v < ll_v)
                if not sto_ok:
                    continue
            pk = tso_pink[i]
            if not (isinstance(pk, float) and np.isnan(pk)) and pk is not None:
                tso_ok = bool(pk) if is_long else not bool(pk)
                if not tso_ok:
                    continue

        # ── Gate B: macd_lc (NaN -> pass) ───────────────────────────────────
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
