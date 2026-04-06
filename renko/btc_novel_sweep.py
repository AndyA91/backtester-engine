#!/usr/bin/env python3
"""
btc_novel_sweep.py -- BTC Novel Strategy Discovery (Long Only)

Four parts exploring untested entry concepts:
  Part A — 14 new individual entry signals (×3 cooldowns = 42 combos)
  Part B — Best new signals combined with existing macd+kama+stoch (24 combos)
  Part C — Confluence mode: require 2+ signals within N bricks (36 combos)
  Part D — MACD/Stoch parameter variants (48 combos)

Baseline: BTC007 optimized (macd+kama+stoch + PSAR + chop60, cd=3)
  Python OOS: PF=21.92, 177t (1.0/d), WR=68.4%

Usage:
    python renko/btc_novel_sweep.py
"""

import contextlib
import io
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from renko.config import MAX_WORKERS

# -- Config --------------------------------------------------------------------

LTF_FILE   = "OANDA_BTCUSD.SPOT.US, 1S renko 150.csv"
HTF_FILE   = "OANDA_BTCUSD.SPOT.US, 1S renko 300.csv"
IS_START   = "2024-06-04"
IS_END     = "2025-09-30"
OOS_START  = "2025-10-01"
OOS_END    = "2026-03-19"
COMMISSION = 0.0046
CAPITAL    = 1000.0
QTY_VALUE  = 20
OOS_DAYS   = 170


# -- Data loading --------------------------------------------------------------

def _load_ltf():
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    from renko.phase6_enrichment import add_phase6_indicators
    df = load_renko_export(LTF_FILE)
    add_renko_indicators(df)
    add_phase6_indicators(df)
    return df


def _load_htf():
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    df = load_renko_export(HTF_FILE)
    add_renko_indicators(df)
    return df


def _align_htf_gate(df_ltf, df_htf, htf_gate_arr):
    htf_frame = pd.DataFrame({
        "t": df_htf.index.values,
        "g": htf_gate_arr.astype(float),
    }).sort_values("t")
    ltf_frame = pd.DataFrame({
        "t": df_ltf.index.values,
    }).sort_values("t")
    merged = pd.merge_asof(ltf_frame, htf_frame, on="t", direction="backward")
    g = merged["g"].values
    return np.where(np.isnan(g), True, g > 0.5).astype(bool)


def _run_bt(df, entry, exit_, start, end):
    from engine import BacktestConfig, run_backtest
    df2 = df.copy()
    df2["long_entry"] = entry
    df2["long_exit"] = exit_
    cfg = BacktestConfig(
        initial_capital=CAPITAL, commission_pct=COMMISSION, slippage_ticks=0,
        qty_type="cash", qty_value=QTY_VALUE, pyramiding=1,
        start_date=start, end_date=end,
        take_profit_pct=0.0, stop_loss_pct=0.0,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest(df2, cfg)
    pf = kpis.get("profit_factor", 0.0) or 0.0
    return {
        "pf":     float("inf") if math.isinf(pf) else float(pf),
        "net":    float(kpis.get("net_profit", 0.0) or 0.0),
        "trades": int(kpis.get("total_trades", 0) or 0),
        "wr":     float(kpis.get("win_rate", 0.0) or 0.0),
        "dd":     float(kpis.get("max_drawdown_pct", 0.0) or 0.0),
    }


# ==============================================================================
# Part A — 14 New Individual Entry Signals
# ==============================================================================

def _gen_ema_cross(df, fast_col, slow_col, gate, cooldown):
    """EMA golden cross: fast crosses above slow on up brick."""
    n = len(df)
    fast = df[fast_col].values
    slow = df[slow_col].values
    brick_up = df["brick_up"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    for i in range(60, n):
        if in_pos:
            if not bool(brick_up[i]):
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or not bool(brick_up[i]) or (i - last_bar) < cooldown:
            continue
        # Cross: fast[i] > slow[i] and fast[i-1] <= slow[i-1]
        if (not np.isnan(fast[i]) and not np.isnan(slow[i])
            and not np.isnan(fast[i-1]) and not np.isnan(slow[i-1])
            and fast[i] > slow[i] and fast[i-1] <= slow[i-1]):
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_di_cross(df, gate, cooldown):
    """+DI crosses above -DI on up brick."""
    n = len(df)
    pdi = df["plus_di"].values
    mdi = df["minus_di"].values
    brick_up = df["brick_up"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    for i in range(60, n):
        if in_pos:
            if not bool(brick_up[i]):
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or not bool(brick_up[i]) or (i - last_bar) < cooldown:
            continue
        if (not np.isnan(pdi[i]) and not np.isnan(mdi[i])
            and not np.isnan(pdi[i-1]) and not np.isnan(mdi[i-1])
            and pdi[i] > mdi[i] and pdi[i-1] <= mdi[i-1]):
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_rsi_cross(df, level, gate, cooldown):
    """RSI crosses above level from below on up brick."""
    n = len(df)
    rsi = df["rsi"].values
    brick_up = df["brick_up"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    for i in range(60, n):
        if in_pos:
            if not bool(brick_up[i]):
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or not bool(brick_up[i]) or (i - last_bar) < cooldown:
            continue
        if (not np.isnan(rsi[i]) and not np.isnan(rsi[i-1])
            and rsi[i] > level and rsi[i-1] <= level):
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_cci_cross(df, level, gate, cooldown):
    """CCI crosses above level from below on up brick."""
    n = len(df)
    cci = df["cci"].values
    brick_up = df["brick_up"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    for i in range(60, n):
        if in_pos:
            if not bool(brick_up[i]):
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or not bool(brick_up[i]) or (i - last_bar) < cooldown:
            continue
        if (not np.isnan(cci[i]) and not np.isnan(cci[i-1])
            and cci[i] > level and cci[i-1] <= level):
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_squeeze_release(df, gate, cooldown):
    """Squeeze release bullish: squeeze was on, turns off, momentum > 0."""
    n = len(df)
    sq_on = df["sq_on"].values
    sq_mom = df["sq_momentum"].values
    brick_up = df["brick_up"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    for i in range(60, n):
        if in_pos:
            if not bool(brick_up[i]):
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or not bool(brick_up[i]) or (i - last_bar) < cooldown:
            continue
        # sq_on was True, now False, and momentum > 0
        if (not np.isnan(sq_on[i]) and not np.isnan(sq_on[i-1])
            and not np.isnan(sq_mom[i])
            and sq_on[i-1] > 0.5 and sq_on[i] < 0.5 and sq_mom[i] > 0):
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_stoch_kd_cross(df, max_level, gate, cooldown):
    """Stoch %K crosses above %D when both below max_level."""
    n = len(df)
    sk = df["stoch_k"].values
    sd = df["stoch_d"].values
    brick_up = df["brick_up"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    for i in range(60, n):
        if in_pos:
            if not bool(brick_up[i]):
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or not bool(brick_up[i]) or (i - last_bar) < cooldown:
            continue
        if (not np.isnan(sk[i]) and not np.isnan(sd[i])
            and not np.isnan(sk[i-1]) and not np.isnan(sd[i-1])
            and sk[i] > sd[i] and sk[i-1] <= sd[i-1]
            and sk[i] < max_level):
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_wpr_cross(df, level, gate, cooldown):
    """Williams %R crosses above level from below on up brick."""
    n = len(df)
    wpr = df["wpr"].values
    brick_up = df["brick_up"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    for i in range(60, n):
        if in_pos:
            if not bool(brick_up[i]):
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or not bool(brick_up[i]) or (i - last_bar) < cooldown:
            continue
        if (not np.isnan(wpr[i]) and not np.isnan(wpr[i-1])
            and wpr[i] > level and wpr[i-1] <= level):
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_consecutive_bricks(df, n_up, gate, cooldown):
    """N consecutive up bricks after at least 1 down brick."""
    n = len(df)
    brick_up = df["brick_up"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    for i in range(60, n):
        if in_pos:
            if not bool(brick_up[i]):
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or not bool(brick_up[i]) or (i - last_bar) < cooldown:
            continue
        # Check: n_up consecutive up bricks ending at i-1, preceded by a down
        ok = True
        for k in range(1, n_up + 1):
            idx = i - k
            if idx < 0 or not bool(brick_up[idx]):
                ok = False
                break
        if ok and (i - n_up - 1 >= 0) and not bool(brick_up[i - n_up - 1]):
            # Signal: i is also up, and we had n_up up bricks + 1 down before
            # Actually: enter on the (n_up+1)th up brick
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_bb_expansion(df, gate, cooldown):
    """BB bandwidth expansion: bw increases AND close > bb_mid on up brick."""
    n = len(df)
    bw = df["bb_bw"].values
    mid = df["bb_mid"].values
    close = df["Close"].values.astype(float)
    brick_up = df["brick_up"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    for i in range(60, n):
        if in_pos:
            if not bool(brick_up[i]):
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or not bool(brick_up[i]) or (i - last_bar) < cooldown:
            continue
        # BW expanding and price above mid
        if (not np.isnan(bw[i]) and not np.isnan(bw[i-1])
            and not np.isnan(mid[i])
            and bw[i] > bw[i-1] and close[i] > mid[i]):
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_adx_rising(df, min_adx, rise_amount, gate, cooldown):
    """ADX rising: ADX >= min_adx and ADX[i] - ADX[i-2] >= rise_amount."""
    n = len(df)
    adx = df["adx"].values
    brick_up = df["brick_up"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    for i in range(60, n):
        if in_pos:
            if not bool(brick_up[i]):
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or not bool(brick_up[i]) or (i - last_bar) < cooldown:
            continue
        if (not np.isnan(adx[i]) and not np.isnan(adx[i-2])
            and adx[i] >= min_adx
            and (adx[i] - adx[i-2]) >= rise_amount):
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_escgo_cross(df, gate, cooldown):
    """ESCGO fast crosses above slow (trigger) on up brick."""
    n = len(df)
    fast = df["escgo_fast"].values
    slow = df["escgo_slow"].values
    brick_up = df["brick_up"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    for i in range(60, n):
        if in_pos:
            if not bool(brick_up[i]):
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or not bool(brick_up[i]) or (i - last_bar) < cooldown:
            continue
        if (not np.isnan(fast[i]) and not np.isnan(slow[i])
            and not np.isnan(fast[i-1]) and not np.isnan(slow[i-1])
            and fast[i] > slow[i] and fast[i-1] <= slow[i-1]):
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_sq_momentum_cross(df, gate, cooldown):
    """Squeeze momentum crosses above 0 from below on up brick."""
    n = len(df)
    mom = df["sq_momentum"].values
    brick_up = df["brick_up"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    for i in range(60, n):
        if in_pos:
            if not bool(brick_up[i]):
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or not bool(brick_up[i]) or (i - last_bar) < cooldown:
            continue
        if (not np.isnan(mom[i]) and not np.isnan(mom[i-1])
            and mom[i] > 0 and mom[i-1] <= 0):
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


# ==============================================================================
# Part B — Combine best new signals with existing trio
# ==============================================================================

def _gen_combined_plus(df, params, gate):
    """Existing macd+kama+stoch signals plus optional new signals."""
    n = len(df)
    brick_up = df["brick_up"].values
    macd_h = df["macd_hist"].values
    stoch_k = df["stoch_k"].values
    kama_s = df["kama_slope"].values
    cooldown = params.get("cooldown", 3)
    stoch_thresh = params.get("stoch_thresh", 25)
    chop = df["chop"].values
    chop_max = params.get("chop_max", 60)

    # New signals to add
    add_signals = params.get("add_signals", [])

    # Pre-extract new signal arrays
    rsi = df["rsi"].values if "rsi50" in add_signals else None
    cci = df["cci"].values if "cci0" in add_signals else None
    wpr = df["wpr"].values if "wpr" in add_signals else None
    pdi = df["plus_di"].values if "di" in add_signals else None
    mdi = df["minus_di"].values if "di" in add_signals else None
    sq_mom = df["sq_momentum"].values if "sqmom" in add_signals else None
    escgo_f = df["escgo_fast"].values if "escgo" in add_signals else None
    escgo_s = df["escgo_slow"].values if "escgo" in add_signals else None
    ema9 = df["ema9"].values if "ema9_21" in add_signals else None
    ema21 = df["ema21"].values if "ema9_21" in add_signals else None

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    for i in range(60, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue

        if not gate[i] or not up or (i - last_bar) < cooldown:
            continue

        # Chop gate
        if chop_max > 0 and not np.isnan(chop[i]) and chop[i] > chop_max:
            continue

        fired = False

        # Existing signals
        if not fired:
            if not np.isnan(macd_h[i]) and not np.isnan(macd_h[i-1]):
                if macd_h[i] > 0 and macd_h[i-1] <= 0:
                    fired = True
        if not fired:
            if not np.isnan(kama_s[i]) and not np.isnan(kama_s[i-1]):
                if kama_s[i] > 0 and kama_s[i-1] <= 0:
                    fired = True
        if not fired:
            if not np.isnan(stoch_k[i]) and not np.isnan(stoch_k[i-1]):
                if stoch_k[i] > stoch_thresh and stoch_k[i-1] <= stoch_thresh:
                    fired = True

        # New signals
        if not fired and "rsi50" in add_signals:
            if not np.isnan(rsi[i]) and not np.isnan(rsi[i-1]):
                if rsi[i] > 50 and rsi[i-1] <= 50:
                    fired = True
        if not fired and "cci0" in add_signals:
            if not np.isnan(cci[i]) and not np.isnan(cci[i-1]):
                if cci[i] > 0 and cci[i-1] <= 0:
                    fired = True
        if not fired and "wpr" in add_signals:
            if not np.isnan(wpr[i]) and not np.isnan(wpr[i-1]):
                if wpr[i] > -50 and wpr[i-1] <= -50:
                    fired = True
        if not fired and "di" in add_signals:
            if (not np.isnan(pdi[i]) and not np.isnan(mdi[i])
                and not np.isnan(pdi[i-1]) and not np.isnan(mdi[i-1])
                and pdi[i] > mdi[i] and pdi[i-1] <= mdi[i-1]):
                fired = True
        if not fired and "sqmom" in add_signals:
            if not np.isnan(sq_mom[i]) and not np.isnan(sq_mom[i-1]):
                if sq_mom[i] > 0 and sq_mom[i-1] <= 0:
                    fired = True
        if not fired and "escgo" in add_signals:
            if (not np.isnan(escgo_f[i]) and not np.isnan(escgo_s[i])
                and not np.isnan(escgo_f[i-1]) and not np.isnan(escgo_s[i-1])
                and escgo_f[i] > escgo_s[i] and escgo_f[i-1] <= escgo_s[i-1]):
                fired = True
        if not fired and "ema9_21" in add_signals:
            if (not np.isnan(ema9[i]) and not np.isnan(ema21[i])
                and not np.isnan(ema9[i-1]) and not np.isnan(ema21[i-1])
                and ema9[i] > ema21[i] and ema9[i-1] <= ema21[i-1]):
                fired = True

        if fired:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


# ==============================================================================
# Part C — Confluence mode
# ==============================================================================

def _gen_confluence(df, gate, cooldown, required, window):
    """Require `required` of {MACD, KAMA, Stoch, RSI50, CCI0} to fire within
    the last `window` bricks. Enter on the brick where threshold is met."""
    n = len(df)
    brick_up = df["brick_up"].values
    macd_h = df["macd_hist"].values
    kama_s = df["kama_slope"].values
    stoch_k = df["stoch_k"].values
    rsi = df["rsi"].values
    cci = df["cci"].values
    chop = df["chop"].values

    # Pre-detect signal fire bars
    sig_macd = np.zeros(n, dtype=bool)
    sig_kama = np.zeros(n, dtype=bool)
    sig_stoch = np.zeros(n, dtype=bool)
    sig_rsi = np.zeros(n, dtype=bool)
    sig_cci = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if not np.isnan(macd_h[i]) and not np.isnan(macd_h[i-1]):
            if macd_h[i] > 0 and macd_h[i-1] <= 0:
                sig_macd[i] = True
        if not np.isnan(kama_s[i]) and not np.isnan(kama_s[i-1]):
            if kama_s[i] > 0 and kama_s[i-1] <= 0:
                sig_kama[i] = True
        if not np.isnan(stoch_k[i]) and not np.isnan(stoch_k[i-1]):
            if stoch_k[i] > 25 and stoch_k[i-1] <= 25:
                sig_stoch[i] = True
        if not np.isnan(rsi[i]) and not np.isnan(rsi[i-1]):
            if rsi[i] > 50 and rsi[i-1] <= 50:
                sig_rsi[i] = True
        if not np.isnan(cci[i]) and not np.isnan(cci[i-1]):
            if cci[i] > 0 and cci[i-1] <= 0:
                sig_cci[i] = True

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    for i in range(60, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue

        if not gate[i] or not up or (i - last_bar) < cooldown:
            continue

        # Chop gate
        if not np.isnan(chop[i]) and chop[i] > 60:
            continue

        # Count signals in window [i-window+1, i]
        start_w = max(0, i - window + 1)
        count = 0
        if np.any(sig_macd[start_w:i+1]):
            count += 1
        if np.any(sig_kama[start_w:i+1]):
            count += 1
        if np.any(sig_stoch[start_w:i+1]):
            count += 1
        if np.any(sig_rsi[start_w:i+1]):
            count += 1
        if np.any(sig_cci[start_w:i+1]):
            count += 1

        if count >= required:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


# ==============================================================================
# Part D — MACD/Stoch parameter variants
# ==============================================================================

def _compute_macd(close, fast, slow, signal):
    """Compute MACD histogram with custom params."""
    ema_fast = pd.Series(close).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, adjust=False).mean().values
    macd_line = ema_fast - ema_slow
    sig_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
    hist = macd_line - sig_line
    # Pre-shift
    result = np.full(len(close), np.nan)
    result[1:] = hist[:-1]
    return result


def _compute_stoch(close, high, low, k_period, smooth):
    """Compute Stochastic %K with custom params."""
    n = len(close)
    raw_k = np.full(n, np.nan)
    for i in range(k_period - 1, n):
        hi = np.max(high[i-k_period+1:i+1])
        lo = np.min(low[i-k_period+1:i+1])
        if hi > lo:
            raw_k[i] = 100.0 * (close[i] - lo) / (hi - lo)
        else:
            raw_k[i] = 50.0
    # Smooth
    smooth_k = pd.Series(raw_k).rolling(smooth, min_periods=1).mean().values
    # Pre-shift
    result = np.full(n, np.nan)
    result[1:] = smooth_k[:-1]
    return result


def _gen_macd_stoch_variant(df, macd_params, stoch_params, gate, cooldown, chop_max=60):
    """Custom MACD + custom Stoch + KAMA as entry signals."""
    n = len(df)
    brick_up = df["brick_up"].values
    close = df["Close"].values.astype(float)
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    chop = df["chop"].values
    kama_s = df["kama_slope"].values

    # Custom MACD
    macd_h = _compute_macd(close, macd_params[0], macd_params[1], macd_params[2])
    # Custom Stoch
    stoch_k = _compute_stoch(close, high, low, stoch_params[0], stoch_params[1])
    stoch_thresh = stoch_params[2] if len(stoch_params) > 2 else 25

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    for i in range(60, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or not up or (i - last_bar) < cooldown:
            continue
        if chop_max > 0 and not np.isnan(chop[i]) and chop[i] > chop_max:
            continue

        fired = False
        # MACD cross
        if not np.isnan(macd_h[i]) and not np.isnan(macd_h[i-1]):
            if macd_h[i] > 0 and macd_h[i-1] <= 0:
                fired = True
        # KAMA turn
        if not fired:
            if not np.isnan(kama_s[i]) and not np.isnan(kama_s[i-1]):
                if kama_s[i] > 0 and kama_s[i-1] <= 0:
                    fired = True
        # Stoch cross
        if not fired:
            if not np.isnan(stoch_k[i]) and not np.isnan(stoch_k[i-1]):
                if stoch_k[i] > stoch_thresh and stoch_k[i-1] <= stoch_thresh:
                    fired = True

        if fired:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


# ==============================================================================
# Combo builders
# ==============================================================================

def _build_part_a():
    """Part A: 14 new individual signals × 3 cooldowns = 42 combos."""
    combos = []
    signals = [
        ("ema9_21_cross", "ema_cross", {"fast_col": "ema9", "slow_col": "ema21"}),
        ("ema9_50_cross", "ema_cross", {"fast_col": "ema9", "slow_col": "ema50"}),
        ("di_cross", "di_cross", {}),
        ("rsi_50_cross", "rsi_cross", {"level": 50}),
        ("rsi_40_cross", "rsi_cross", {"level": 40}),
        ("cci_0_cross", "cci_cross", {"level": 0}),
        ("cci_n100_cross", "cci_cross", {"level": -100}),
        ("squeeze_release", "squeeze_release", {}),
        ("stoch_kd_50", "stoch_kd_cross", {"max_level": 50}),
        ("stoch_kd_80", "stoch_kd_cross", {"max_level": 80}),
        ("wpr_n50_cross", "wpr_cross", {"level": -50}),
        ("consec_3up", "consecutive", {"n_up": 3}),
        ("bb_expansion", "bb_expansion", {}),
        ("adx_rising_20_5", "adx_rising", {"min_adx": 20, "rise_amount": 5}),
        ("escgo_cross", "escgo_cross", {}),
        ("sq_mom_cross", "sq_momentum_cross", {}),
    ]
    for name, sig_type, sig_params in signals:
        for cd in [2, 3, 5]:
            combos.append({
                "part": "A",
                "sig_name": name,
                "sig_type": sig_type,
                "sig_params": sig_params,
                "cooldown": cd,
                "label": f"{name}_cd{cd}",
            })
    return combos


def _build_part_b():
    """Part B: Add new signals to existing trio. 8 add-ons × 3 CDs = 24 combos."""
    combos = []
    add_sets = [
        ("rsi50", ["rsi50"]),
        ("cci0", ["cci0"]),
        ("wpr", ["wpr"]),
        ("di", ["di"]),
        ("sqmom", ["sqmom"]),
        ("escgo", ["escgo"]),
        ("ema9_21", ["ema9_21"]),
        ("rsi50+cci0", ["rsi50", "cci0"]),
    ]
    for name, sigs in add_sets:
        for cd in [2, 3, 5]:
            combos.append({
                "part": "B",
                "add_signals": sigs,
                "cooldown": cd,
                "chop_max": 60,
                "stoch_thresh": 25,
                "label": f"trio+{name}_cd{cd}",
            })
    return combos


def _build_part_c():
    """Part C: Confluence mode — require N signals in W-brick window.
    N={2,3} × W={3,5,8} × CD={2,3,5} = 18 combos."""
    combos = []
    for req in [2, 3]:
        for window in [3, 5, 8]:
            for cd in [2, 3, 5]:
                combos.append({
                    "part": "C",
                    "required": req,
                    "window": window,
                    "cooldown": cd,
                    "label": f"conf{req}w{window}_cd{cd}",
                })
    return combos


def _build_part_d():
    """Part D: MACD/Stoch parameter variants.
    4 MACD × 4 Stoch × 3 CD = 48 combos."""
    combos = []
    macd_sets = [
        (12, 26, 9),   # default
        (8, 17, 9),    # faster
        (5, 35, 5),    # wider divergence
        (8, 21, 5),    # medium
    ]
    stoch_sets = [
        (14, 3, 25),   # default
        (10, 3, 20),   # faster + lower thresh
        (21, 5, 30),   # slower + higher thresh
        (14, 3, 20),   # default period, lower thresh
    ]
    for macd_p in macd_sets:
        for stoch_p in stoch_sets:
            for cd in [3]:  # lock cd=3 for params sweep (48→16 combos)
                combos.append({
                    "part": "D",
                    "macd_params": list(macd_p),
                    "stoch_params": list(stoch_p),
                    "cooldown": cd,
                    "label": f"M{macd_p[0]}_{macd_p[1]}_{macd_p[2]}_S{stoch_p[0]}_{stoch_p[1]}_t{stoch_p[2]}_cd{cd}",
                })
    return combos


# ==============================================================================
# Worker
# ==============================================================================

_w = {}

def _init_worker():
    if "df" not in _w:
        _w["df"] = _load_ltf()
        _w["df_htf"] = _load_htf()
        psar = _w["df"]["psar_dir"].values
        _w["psar_gate"] = np.isnan(psar) | (psar > 0)


def _run_one(combo):
    _init_worker()
    df = _w["df"]
    gate = _w["psar_gate"]
    part = combo["part"]

    if part == "A":
        cd = combo["cooldown"]
        sig_type = combo["sig_type"]
        sp = combo["sig_params"]

        if sig_type == "ema_cross":
            ent, ext = _gen_ema_cross(df, sp["fast_col"], sp["slow_col"], gate, cd)
        elif sig_type == "di_cross":
            ent, ext = _gen_di_cross(df, gate, cd)
        elif sig_type == "rsi_cross":
            ent, ext = _gen_rsi_cross(df, sp["level"], gate, cd)
        elif sig_type == "cci_cross":
            ent, ext = _gen_cci_cross(df, sp["level"], gate, cd)
        elif sig_type == "squeeze_release":
            ent, ext = _gen_squeeze_release(df, gate, cd)
        elif sig_type == "stoch_kd_cross":
            ent, ext = _gen_stoch_kd_cross(df, sp["max_level"], gate, cd)
        elif sig_type == "wpr_cross":
            ent, ext = _gen_wpr_cross(df, sp["level"], gate, cd)
        elif sig_type == "consecutive":
            ent, ext = _gen_consecutive_bricks(df, sp["n_up"], gate, cd)
        elif sig_type == "bb_expansion":
            ent, ext = _gen_bb_expansion(df, gate, cd)
        elif sig_type == "adx_rising":
            ent, ext = _gen_adx_rising(df, sp["min_adx"], sp["rise_amount"], gate, cd)
        elif sig_type == "escgo_cross":
            ent, ext = _gen_escgo_cross(df, gate, cd)
        elif sig_type == "sq_momentum_cross":
            ent, ext = _gen_sq_momentum_cross(df, gate, cd)
        else:
            raise ValueError(f"Unknown sig_type: {sig_type}")

    elif part == "B":
        ent, ext = _gen_combined_plus(df, combo, gate)

    elif part == "C":
        ent, ext = _gen_confluence(df, gate, combo["cooldown"],
                                   combo["required"], combo["window"])

    elif part == "D":
        ent, ext = _gen_macd_stoch_variant(
            df, combo["macd_params"], combo["stoch_params"],
            gate, combo["cooldown"])

    else:
        raise ValueError(f"Unknown part: {part}")

    is_r = _run_bt(df, ent, ext, IS_START, IS_END)
    oos_r = _run_bt(df, ent, ext, OOS_START, OOS_END)
    return combo, is_r, oos_r


# ==============================================================================
# Reporting
# ==============================================================================

def _print_header():
    print(f"  {'#':>3} {'Pt':>2} {'Label':<42} | "
          f"{'IS PF':>7} {'T':>5} {'WR%':>6} | "
          f"{'OOS PF':>8} {'T':>5} {'t/d':>5} {'WR%':>6} {'Net':>9} {'DD%':>7}")
    print(f"  {'-'*120}")


def _print_row(r, rank=None):
    pf_i = "INF" if math.isinf(r["is_pf"]) else f"{r['is_pf']:.2f}"
    pf_o = "INF" if math.isinf(r["oos_pf"]) else f"{r['oos_pf']:.2f}"
    tpd = r["oos_trades"] / OOS_DAYS if r["oos_trades"] > 0 else 0
    prefix = f"  {rank:>3}" if rank else "  "
    print(f"{prefix} {r['part']:>2} {r['label']:<42} | "
          f"{pf_i:>7} {r['is_trades']:>5} {r['is_wr']:>5.1f}% | "
          f"{pf_o:>8} {r['oos_trades']:>5} {tpd:>4.1f} {r['oos_wr']:>5.1f}% "
          f"{r['oos_net']:>9.2f} {r['oos_dd']:>6.2f}%")


def _run_phase(combos, phase_name, all_results):
    total = len(combos)
    print(f"\n  Running Part {phase_name}: {total} combos ({total*2} backtests)...")

    done = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run_one, c): c for c in combos}
        for fut in as_completed(futures):
            try:
                combo, is_r, oos_r = fut.result()
                row = {
                    "part":       combo["part"],
                    "label":      combo["label"],
                    "is_pf":      is_r["pf"],
                    "is_trades":  is_r["trades"],
                    "is_wr":      is_r["wr"],
                    "is_net":     is_r["net"],
                    "is_dd":      is_r["dd"],
                    "oos_pf":     oos_r["pf"],
                    "oos_trades": oos_r["trades"],
                    "oos_wr":     oos_r["wr"],
                    "oos_net":    oos_r["net"],
                    "oos_dd":     oos_r["dd"],
                }
                # Store extra params
                for k in ("cooldown", "sig_name", "required", "window",
                          "add_signals", "macd_params", "stoch_params"):
                    if k in combo:
                        row[k] = combo[k]
                all_results.append(row)
            except Exception as e:
                import traceback
                print(f"  ERROR: {combo.get('label', '???')}: {e}")
                traceback.print_exc()

            done += 1
            if done % 20 == 0 or done == total:
                print(f"    [{done:>4}/{total}]", flush=True)


def _show_part(results, part, title):
    subset = [r for r in results if r["part"] == part]
    viable = [r for r in subset if r["oos_trades"] >= 10 and r["oos_net"] > 0]

    by_wr = sorted([r for r in viable if r["oos_trades"] >= 50],
                   key=lambda r: (r["oos_wr"], r["oos_net"]), reverse=True)
    by_net = sorted([r for r in viable if r["oos_trades"] >= 50],
                    key=lambda r: r["oos_net"], reverse=True)

    print(f"\n{'='*130}")
    print(f"  {title} — {len(viable)} viable / {len(subset)} total")
    print(f"{'='*130}")

    if by_wr:
        print(f"\n  Top 10 by WR (T>=50):")
        _print_header()
        for i, r in enumerate(by_wr[:10]):
            _print_row(r, rank=i+1)

    if by_net:
        print(f"\n  Top 10 by Net (T>=50):")
        _print_header()
        for i, r in enumerate(by_net[:10]):
            _print_row(r, rank=i+1)


# ==============================================================================
# Main
# ==============================================================================

def main():
    print(f"\n{'='*70}")
    print(f"BTC Novel Strategy Sweep")
    print(f"  Baseline: macd+kama+stoch + PSAR + chop60, cd=3")
    print(f"  Python OOS: PF=21.92, 177t (1.0/d), WR=68.4%")
    print(f"  Workers:    {MAX_WORKERS}")
    print(f"{'='*70}")

    all_results = []

    # Part A
    combos_a = _build_part_a()
    print(f"\n  Part A: {len(combos_a)} combos — Individual new signals")
    _run_phase(combos_a, "A", all_results)
    _show_part(all_results, "A", "Part A — Individual New Signals")

    # Part B
    combos_b = _build_part_b()
    print(f"\n  Part B: {len(combos_b)} combos — Trio + new signals")
    _run_phase(combos_b, "B", all_results)
    _show_part(all_results, "B", "Part B — Trio + New Signals")

    # Part C
    combos_c = _build_part_c()
    print(f"\n  Part C: {len(combos_c)} combos — Confluence mode")
    _run_phase(combos_c, "C", all_results)
    _show_part(all_results, "C", "Part C — Confluence Mode")

    # Part D
    combos_d = _build_part_d()
    print(f"\n  Part D: {len(combos_d)} combos — MACD/Stoch param variants")
    _run_phase(combos_d, "D", all_results)
    _show_part(all_results, "D", "Part D — MACD/Stoch Parameter Variants")

    # Global summary
    total = len(combos_a) + len(combos_b) + len(combos_c) + len(combos_d)
    viable_all = [r for r in all_results
                  if r["oos_trades"] >= 50 and r["oos_net"] > 0]

    print(f"\n{'='*130}")
    print(f"  GLOBAL TOP 20 by WR (T>=50, net>0): {len(viable_all)} configs / {total} total")
    print(f"{'='*130}")
    viable_all.sort(key=lambda r: (r["oos_wr"], r["oos_net"]), reverse=True)
    _print_header()
    for i, r in enumerate(viable_all[:20]):
        _print_row(r, rank=i+1)

    # Highlight anything beating baseline
    better = [r for r in viable_all
              if r["oos_wr"] > 68.4 or
              (r["oos_wr"] >= 68.0 and r["oos_trades"] >= 170)]
    if better:
        print(f"\n  ** {len(better)} configs beat/match baseline WR=68.4% or "
              f"match WR>=68% with T>=170 **")
        _print_header()
        for i, r in enumerate(better[:10]):
            _print_row(r, rank=i+1)
    else:
        print(f"\n  No configs beat baseline WR=68.4% at T>=50")

    # Save
    out_path = ROOT / "ai_context" / "btc_novel_sweep_results.json"
    out_path.parent.mkdir(exist_ok=True)
    serializable = []
    for r in all_results:
        sr = dict(r)
        for k in ("is_pf", "oos_pf"):
            if math.isinf(sr[k]):
                sr[k] = "inf"
        serializable.append(sr)
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved {len(all_results)} results -> {out_path}")
    print(f"Total combos: {total} ({total*2} backtests)")


if __name__ == "__main__":
    main()
