"""EA009: Institutional Trend Reversal — HPotter S/R + Blackcat L3 Imbalance

Mean-reversion / institutional-turn strategy. The HPotter Trend Line identifies
persistent structural levels acting as support or resistance. Blackcat L3 Volume
Imbalance Pro detects when stacked volume imbalance or absorption events appear
at those levels — signalling institutional participation at the structural turn.

Signal logic:
  LONG:  hp_is_support = True  (price history above trend line → line = support)
         AND abs(close − hp_trend_line) < tl_dist_atr × ATR
         AND (vi_stacked_buy OR vi_buy_absorption)
         AND vi_score >= score_threshold  (0 = gate off)
         AND current brick is UP  (reversal direction confirmed)

  SHORT: hp_is_support = False (price history below trend line → line = resistance)
         AND abs(close − hp_trend_line) < tl_dist_atr × ATR
         AND (vi_stacked_sell OR vi_sell_absorption)
         AND vi_score <= −score_threshold
         AND current brick is DOWN

Exit: first opposing Renko brick (standard).

Hypothesis: institutional order-flow concentrates at persistent S/R levels. The
combination of HPotter Trend Line proximity + stacked volume imbalance targets
high-conviction institutional turns early in the reversal impulse.

EURAUD IS:  2023-07-20 → 2025-09-30
EURAUD OOS: 2025-10-01 → 2026-03-17
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from indicators.HPotter.trend_line import calc_hp_trend_line
from indicators.blackcat1402.blackcat_l3_volume_imbalance_pro import (
    calc_bc_l3_volume_imbalance_pro,
)
from renko.data import load_renko_export
from renko.indicators import add_renko_indicators

DESCRIPTION = "EURAUD Institutional Trend Reversal — HPotter S/R + Blackcat L3 Volume Imbalance"

HYPOTHESIS = (
    "The HPotter Trend Line identifies structural levels where institutional "
    "traders anchor orders. When Blackcat L3 Volume Imbalance Pro detects "
    "stacked buy (sell) imbalance or absorption at a trend-line support "
    "(resistance), it signals institutional participation in a turn. Entry on "
    "the confirming UP (DOWN) Renko brick captures the initial reversal impulse."
)

RENKO_FILE      = "OANDA_EURAUD, 1S renko 0.0006.csv"
COMMISSION_PCT  = 0.009
INITIAL_CAPITAL = 1000.0

# ---------------------------------------------------------------------------
# PARAM_GRID
# ---------------------------------------------------------------------------
# tl_length:       HP Trend Line SMA period — smaller = more responsive S/R
# tl_dist_atr:     ATR multiplier for "near trend line" proximity gate
# imb_threshold:   buy/sell ratio threshold for imbalance detection (× 100 internally)
# min_stacked:     consecutive imbalance bars required for stacked signal
# score_threshold: minimum absolute composite score; 0 = disable score gate
# cooldown:        minimum bricks between entries
# session_start:   UTC hour gate (0 = no gate, 13 = London+NY only)
PARAM_GRID = {
    "tl_length":       [15, 25, 50],
    "tl_dist_atr":     [1.0, 2.0, 3.0],
    "imb_threshold":   [200, 300, 400],
    "min_stacked":     [2, 3],
    "score_threshold": [0, 20, 40],
    "cooldown":        [10, 20],
    "session_start":   [0, 13],
}
# Total: 3 × 3 × 3 × 2 × 3 × 2 × 2 = 648 combos
# Unique cache builds: 3 (tl_length) × 3 (imb_threshold) × 2 (min_stacked) = 18


# ---------------------------------------------------------------------------
# Indicator cache — keyed by (tl_length, imb_threshold, min_stacked)
# ---------------------------------------------------------------------------

_CACHE: dict = {}


def _get_or_build_cache(
    tl_length: int,
    imb_threshold: float,
    min_stacked: int,
) -> pd.DataFrame:
    key = (tl_length, int(imb_threshold), min_stacked)
    if key in _CACHE:
        return _CACHE[key]

    df = load_renko_export(RENKO_FILE)
    add_renko_indicators(df)  # adds pre-shifted atr, adx, etc.

    # ── HP Trend Line (expects capitalized High/Low/Close) ────────────────────
    # calc_hp_trend_line returns a copy with hp_trend_line and hp_is_support.
    # The function applies hl2.shift(1) internally (Pine's hl2[1]), so
    # hp_trend_line[i] uses bars i-1..i-length only (no bar-i data).
    # We shift once more to match the project convention (indicator[i] = bar i-1).
    df = calc_hp_trend_line(df, length=tl_length)
    df["hp_trend_line"] = df["hp_trend_line"].shift(1)
    df["hp_is_support"] = df["hp_is_support"].shift(1)

    # ── Volume Imbalance Pro (expects lowercase ohlcv) ────────────────────────
    df_lc = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    vi = calc_bc_l3_volume_imbalance_pro(
        df_lc,
        imbalance_threshold=float(imb_threshold),
        min_stacked_rows=min_stacked,
    )

    # Shift all outputs by 1 to prevent lookahead in the signal loop
    df["vi_stacked_buy"]  = vi["bc_stacked_buy"].shift(1).values
    df["vi_stacked_sell"] = vi["bc_stacked_sell"].shift(1).values
    df["vi_buy_abs"]      = vi["bc_buy_absorption"].shift(1).values
    df["vi_sell_abs"]     = vi["bc_sell_absorption"].shift(1).values
    df["vi_score"]        = vi["bc_score"].shift(1).values

    _CACHE[key] = df
    return df


# ---------------------------------------------------------------------------
# Signal generator
# ---------------------------------------------------------------------------

def generate_signals(
    df: pd.DataFrame,
    tl_length:       int   = 25,
    tl_dist_atr:     float = 2.0,
    imb_threshold:   float = 300.0,
    min_stacked:     int   = 3,
    score_threshold: int   = 0,
    cooldown:        int   = 10,
    session_start:   int   = 0,
) -> pd.DataFrame:
    """
    Institutional Trend Reversal: enter at HP Trend Line S/R when volume
    imbalance confirms institutional participation at the level.

    Args:
        df:              Renko DataFrame with brick_up + standard indicators.
        tl_length:       HP Trend Line SMA period (15/25/50).
        tl_dist_atr:     ATR multiplier for proximity gate.
        imb_threshold:   Imbalance ratio threshold for stacked detection.
        min_stacked:     Minimum consecutive imbalance bars for stacked signal.
        score_threshold: Minimum abs composite score to allow entry (0 = off).
        cooldown:        Minimum bricks between entries.
        session_start:   UTC hour gate (0 = off, 13 = London+NY only).

    Returns:
        df with long_entry, long_exit, short_entry, short_exit bool columns.
    """
    # Warmup: HP Trend Line needs tl_length bars; add buffer for divergence detection
    warmup = max(tl_length * 2 + 5, 60)

    c = _get_or_build_cache(tl_length, imb_threshold, min_stacked).reindex(df.index)

    close     = df["Close"].values
    brick_up  = df["brick_up"].values
    atr_vals  = df["atr"].values        # pre-shifted in add_renko_indicators
    hours     = df.index.hour
    n         = len(df)

    hp_tl       = c["hp_trend_line"].values
    hp_is_sup   = c["hp_is_support"].fillna(False).values.astype(bool)
    vi_stk_buy  = c["vi_stacked_buy"].fillna(False).values.astype(bool)
    vi_stk_sell = c["vi_stacked_sell"].fillna(False).values.astype(bool)
    vi_buy_abs  = c["vi_buy_abs"].fillna(False).values.astype(bool)
    vi_sell_abs = c["vi_sell_abs"].fillna(False).values.astype(bool)
    vi_score    = c["vi_score"].fillna(0).values.astype(float)

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    in_position    = False
    trade_dir      = 0
    last_trade_bar = -999_999

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # ── Exit: first opposing brick ────────────────────────────────────────
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

        # ── Session gate ──────────────────────────────────────────────────────
        if session_start > 0 and hours[i] < session_start:
            continue

        # ── Cooldown ──────────────────────────────────────────────────────────
        if (i - last_trade_bar) < cooldown:
            continue

        # ── Level validity ────────────────────────────────────────────────────
        tl    = hp_tl[i]
        atr_v = atr_vals[i]
        if np.isnan(tl) or np.isnan(atr_v) or atr_v <= 0:
            continue

        # ── Near trend-line gate ──────────────────────────────────────────────
        if abs(close[i] - tl) >= tl_dist_atr * atr_v:
            continue

        is_sup = hp_is_sup[i]
        score  = float(vi_score[i])

        # ── LONG: at support, buy imbalance, UP brick ─────────────────────────
        if up and is_sup:
            has_signal = vi_stk_buy[i] or vi_buy_abs[i]
            score_ok   = score_threshold <= 0 or score >= score_threshold
            if has_signal and score_ok:
                long_entry[i]  = True
                in_position    = True
                trade_dir      = 1
                last_trade_bar = i

        # ── SHORT: at resistance, sell imbalance, DOWN brick ──────────────────
        elif not up and not is_sup:
            has_signal = vi_stk_sell[i] or vi_sell_abs[i]
            score_ok   = score_threshold <= 0 or score <= -score_threshold
            if has_signal and score_ok:
                short_entry[i] = True
                in_position    = True
                trade_dir      = -1
                last_trade_bar = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
