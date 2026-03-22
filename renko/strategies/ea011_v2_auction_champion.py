"""EA011 v2: Auction Champion — Composite Score + Trend Line + Exhaustion Filter

Evolution of EA011 (Advanced Auction Breakout). Three structural layers added:

  Layer A — Imbalance Quality Gate (bc_score threshold):
    bc_score from L3 Volume Imbalance Pro is a composite -100..+100 score.
    For a typical Renko up-brick its sub-components are:
      +20 buy_imbalance (always True on up-brick, degenerate — see note)
      +20 stacked_buy   (N consecutive up-bricks — also degenerate)
      +15 MACD aligned  (macd_hist > 0 — meaningful, uses close prices)
      +15 above VWAP    (close > VWAP   — meaningful, uses price/volume)
      +15 RSI oversold  (RSI < 30       — meaningful, uses close prices)
    score_threshold=40 requires MACD+VWAP alignment beyond the degenerate base.
    score_threshold=0 disables the gate entirely.

    NOTE — Stacked Imbalance degeneracy on Renko:
    The Elder volume proxy (buy_vol = vol × (close−low)/(high−low)) reduces to
    buy_vol = vol and sell_vol = 0 on every up-brick, because Renko up-bricks
    always close exactly at the high (close = open + brick_size = high).
    bc_stacked_buy therefore becomes "N consecutive up-bricks" — a simple brick
    counter that adds no information beyond the CVD condition already in v1.
    The composite bc_score is used instead because its RSI/MACD/VWAP components
    are computed from close prices and remain non-degenerate on Renko data.

  Layer B — Trend Line Alignment (hp_is_support):
    HPotter Trend Line: Points = SMA(hl2[1], 25).  is_support = close[1] > Points.
    Longs:  require hp_is_support == True  (price above trend line → support)
    Shorts: require hp_is_support == False (price below → resistance)
    Ensures the breakout aligns with mid-term structural direction.

  Layer C — Exhaustion Skip (bc_bearish_div / bc_bullish_div from VP Pro):
    bc_bearish_div: price pivot-high HH but cumulative delta pivot-high LH.
    On Renko, delta ≈ directional volume per brick, so this detects rallies
    where volume at the pivot is contracting — a genuine exhaustion signal.
    Skip LONG if bc_bearish_div is present; skip SHORT if bc_bullish_div.

Core EA011 v1 conditions retained:
  Condition 1 — Delta-Confirmed Breakout (bc_vah_breakout / bc_val_breakdown)
  Condition 2 — CVD Trending in Breakout Direction (cvd_lookback)
  Condition 3 — POC Migration Support (optional via req_poc_mig)

Exit: first opposing Renko brick. Exit modifications were tested during the
2026-03-07 exit research and found catastrophically negative (exit_confirm_bricks=2
dropped OOS PF from 12.79 → 4.64). The duration asymmetry (winners 13.8h,
losers 6.5h) is the core edge — it is preserved only by the clean first-brick exit.

EURAUD IS:  2023-07-20 → 2025-09-30
EURAUD OOS: 2025-10-01 → 2026-03-17
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from indicators.blackcat1402.blackcat_l3_volume_profile_pro import (
    calc_bc_l3_volume_profile_pro,
)
from indicators.blackcat1402.blackcat_l3_volume_imbalance_pro import (
    calc_bc_l3_volume_imbalance_pro,
)
from indicators.HPotter.trend_line import calc_hp_trend_line
from renko.data import load_renko_export
from renko.indicators import add_renko_indicators

DESCRIPTION = (
    "EURAUD Auction Champion — VP breakout + imbalance score + "
    "trend line alignment + exhaustion skip"
)

HYPOTHESIS = (
    "EA011 v2 elevates the triple-gate auction breakout with three structural layers: "
    "(A) bc_score threshold from Imbalance Pro as a composite RSI/MACD/VWAP quality filter — "
    "requires real momentum alignment beyond the degenerate Elder-volume base on Renko; "
    "(B) HPotter trend-line alignment ensures the breakout direction matches mid-term structure; "
    "(C) VP Pro delta-divergence exhaustion skip prevents entries where volume is contracting "
    "at the breakout pivot, targeting the false-breakout class that drives IS→OOS decay."
)

RENKO_FILE      = "OANDA_EURAUD, 1S renko 0.0006.csv"
COMMISSION_PCT  = 0.009
INITIAL_CAPITAL = 1000.0

# ---------------------------------------------------------------------------
# PARAM_GRID
# ---------------------------------------------------------------------------
# vp_lookback:     rolling bars for Volume Profile Pro POC/VAH/VAL
# cvd_lookback:    bricks over which CVD must trend in breakout direction
# req_poc_mig:     require POC migration direction to match entry direction
# req_trendline:   require HP trend line to support the breakout direction
# req_no_exhaust:  skip if delta-divergence exhaustion detected on entry brick
# score_threshold: minimum |bc_score| for imbalance quality gate (0 = disabled)
# cooldown:        minimum bricks between entries
# session_start:   UTC hour gate (0 = no gate, 13 = London+NY only)
PARAM_GRID = {
    "vp_lookback":     [30, 50, 100],
    "cvd_lookback":    [3, 5],
    "req_poc_mig":     [True, False],
    "req_trendline":   [True, False],
    "req_no_exhaust":  [True, False],
    "score_threshold": [0, 40],
    "cooldown":        [5, 10, 20],
    "session_start":   [0, 13],
}
# Total: 3 × 2 × 2 × 2 × 2 × 2 × 3 × 2 = 384 combos
# Unique cache builds: 3 (vp_lookback is the only expensive axis)


# ---------------------------------------------------------------------------
# Indicator cache — keyed by vp_lookback
# Imbalance Pro and HP Trend Line use fixed params and are built once per key
# ---------------------------------------------------------------------------

_CACHE: dict = {}


def _get_or_build_cache(vp_lookback: int) -> pd.DataFrame:
    if vp_lookback in _CACHE:
        return _CACHE[vp_lookback]

    df = load_renko_export(RENKO_FILE)
    add_renko_indicators(df)

    # Both VP Pro and Imbalance Pro expect lowercase ohlcv
    df_lc = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })

    # ── Volume Profile Pro ──────────────────────────────────────────────────
    vp = calc_bc_l3_volume_profile_pro(df_lc, rolling_vp_bars=vp_lookback)

    # Shift 1 bar: signals computed from bar i's close, available at bar i+1
    df["vp_vah_breakout"]  = vp["bc_vah_breakout"].shift(1).values
    df["vp_val_breakdown"] = vp["bc_val_breakdown"].shift(1).values
    df["vp_poc_mig"]       = vp["bc_poc_migration"].shift(1).values
    df["vp_cvd"]           = vp["bc_cvd"].shift(1).values
    df["vp_bearish_div"]   = vp["bc_bearish_div"].shift(1).values
    df["vp_bullish_div"]   = vp["bc_bullish_div"].shift(1).values

    # ── Volume Imbalance Pro (composite score only) ─────────────────────────
    # bc_score uses RSI, MACD, VWAP — all non-degenerate on Renko close prices.
    # bc_stacked_buy/sell degenerate to a brick counter and are NOT used here.
    imb = calc_bc_l3_volume_imbalance_pro(df_lc)
    df["imb_score"] = imb["bc_score"].shift(1).values

    # ── HPotter Trend Line ──────────────────────────────────────────────────
    # calc_hp_trend_line expects uppercase High, Low, Close — matches our df.
    # The module internally uses close[1] and hl2[1], so hp_is_support[i] is
    # already based on bar i-1 data.  Shift 1 more for framework consistency
    # (same convention as VP signals): at bar i we use trend state from bar i-1.
    df_tl = calc_hp_trend_line(df, length=25)
    df["hp_is_support"] = df_tl["hp_is_support"].shift(1).values

    _CACHE[vp_lookback] = df
    return df


# ---------------------------------------------------------------------------
# Signal generator
# ---------------------------------------------------------------------------

def generate_signals(
    df: pd.DataFrame,
    vp_lookback:     int  = 50,
    cvd_lookback:    int  = 3,
    req_poc_mig:     bool = True,
    req_trendline:   bool = True,
    req_no_exhaust:  bool = True,
    score_threshold: int  = 40,
    cooldown:        int  = 10,
    session_start:   int  = 0,
) -> pd.DataFrame:
    """
    Auction Champion: delta-confirmed VA breakout + imbalance quality score
    + trend line alignment + exhaustion skip.

    Args:
        df:              Renko DataFrame with brick_up bool + standard indicators.
        vp_lookback:     Rolling bars for Volume Profile computation.
        cvd_lookback:    Bricks over which CVD must trend in breakout direction.
        req_poc_mig:     Require POC migration to align with breakout direction.
        req_trendline:   Require HP trend line to support breakout direction.
        req_no_exhaust:  Skip entry when VP Pro delta divergence is present.
        score_threshold: Minimum |bc_score| for Imbalance Pro quality gate (0=off).
        cooldown:        Minimum bricks between entries.
        session_start:   UTC hour gate (0 = off, 13 = London+NY only).

    Returns:
        df with long_entry, long_exit, short_entry, short_exit bool columns.
    """
    warmup = max(vp_lookback + cvd_lookback + 5, 60)

    c = _get_or_build_cache(vp_lookback).reindex(df.index)

    brick_up = df["brick_up"].values
    hours    = df.index.hour
    n        = len(df)

    vah_bo    = c["vp_vah_breakout"].fillna(False).values.astype(bool)
    val_bd    = c["vp_val_breakdown"].fillna(False).values.astype(bool)
    poc_mig   = c["vp_poc_mig"].fillna(0).values.astype(int)
    cvd_vals  = c["vp_cvd"].fillna(0.0).values.astype(float)
    bear_div  = c["vp_bearish_div"].fillna(False).values.astype(bool)
    bull_div  = c["vp_bullish_div"].fillna(False).values.astype(bool)
    imb_score = c["imb_score"].fillna(0.0).values.astype(float)
    # fillna(True): allow entries during warmup if trend line not yet computed
    is_support = c["hp_is_support"].fillna(True).values.astype(bool)

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    in_position    = False
    trade_dir      = 0
    last_trade_bar = -999_999

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # ── Exit: first opposing brick ─────────────────────────────────────
        if in_position:
            if trade_dir == 1 and not up:
                long_exit[i] = True
                in_position  = False
                trade_dir    = 0
            elif trade_dir == -1 and up:
                short_exit[i] = True
                in_position   = False
                trade_dir     = 0

        if in_position:
            continue

        # ── Session gate ───────────────────────────────────────────────────
        if session_start > 0 and hours[i] < session_start:
            continue

        # ── Cooldown ───────────────────────────────────────────────────────
        if (i - last_trade_bar) < cooldown:
            continue

        # ── CVD trend ──────────────────────────────────────────────────────
        cvd_ref_i = i - cvd_lookback
        if cvd_ref_i < 0:
            continue
        cvd_rising  = cvd_vals[i] > cvd_vals[cvd_ref_i]
        cvd_falling = cvd_vals[i] < cvd_vals[cvd_ref_i]

        # ── Layer A: Imbalance quality score (0 = disabled) ────────────────
        # Long: positive score ≥ threshold.  Short: negative score ≤ -threshold.
        if score_threshold > 0:
            score_ok_long  = imb_score[i] >= score_threshold
            score_ok_short = imb_score[i] <= -score_threshold
        else:
            score_ok_long  = True
            score_ok_short = True

        # ── Layer C: Exhaustion skip ────────────────────────────────────────
        # bear_div: price HH but delta LH → rally losing volume → skip long
        # bull_div: price LL but delta HL → selloff losing volume → skip short
        if req_no_exhaust:
            no_exhaust_long  = not bear_div[i]
            no_exhaust_short = not bull_div[i]
        else:
            no_exhaust_long  = True
            no_exhaust_short = True

        # ── LONG: VAH breakout + CVD up + all gates ─────────────────────────
        if up and vah_bo[i] and cvd_rising:
            poc_ok   = (not req_poc_mig)  or (poc_mig[i] == 1)
            trend_ok = (not req_trendline) or is_support[i]         # Layer B
            if poc_ok and trend_ok and score_ok_long and no_exhaust_long:
                long_entry[i]  = True
                in_position    = True
                trade_dir      = 1
                last_trade_bar = i

        # ── SHORT: VAL breakdown + CVD down + all gates ─────────────────────
        elif not up and val_bd[i] and cvd_falling:
            poc_ok   = (not req_poc_mig)  or (poc_mig[i] == -1)
            trend_ok = (not req_trendline) or (not is_support[i])   # Layer B
            if poc_ok and trend_ok and score_ok_short and no_exhaust_short:
                short_entry[i] = True
                in_position    = True
                trade_dir      = -1
                last_trade_bar = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
