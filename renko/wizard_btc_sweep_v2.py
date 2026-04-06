#!/usr/bin/env python3
"""
wizard_btc_sweep_v2.py -- Remaining 18 wizard strategies on BTC $150 Renko

Strategies ported (long-only, first-down-brick exit):
    THREE_COMMAS     MA crossover system (Bjorgum 3Commas Bot, 13.7k boosts)
    DOUBLE_TAP       Double bottom pattern (Bjorgum, 8.1k boosts)
    OPEN_CLOSE_CROSS MA(close) > MA(open) crossover (JustUncleL, 5.8k boosts)
    QQE_CROSS        QQE RSI-based threshold channel (JustUncleL, 3.0k boosts)
    DUAL_MA          EMA ribbon crossover (JustUncleL, 1.6k boosts)
    ELDER_RAY        Bull Power = High - EMA(close) (HPotter, 1.4k boosts)
    MADE_ATR         Displaced EMA envelope + ATR stop (HPotter, 1.3k boosts)
    DIVERGENCE       Bullish RSI divergence (Trendoscope, 1.1k boosts)
    INSIDE_BAR       Inside bar momentum (BacktestRookies, 0.8k boosts)
    OKX_MA           Low dips below SMA then recovers (HPotter, 0.6k boosts)
    HLHB             EMA5/10 cross + RSI50 cross (BacktestRookies, 0.4k boosts)
    COMBO_BULL_POWER EMA20-trend + Bull Power > threshold (HPotter, 0.4k)
    EMA_MA_CROSS     EMA/SMA crossover (HPotter, 0.4k boosts)
    COMBO_CCI        EMA20-trend + CCI fast/slow cross (HPotter, 0.4k boosts)
    DEMA_RSI         DEMA-RSI crossover (RicardoSantos, 0.3k boosts)
    COMBO_BBB        EMA20-trend + Bull/Bear Balance (HPotter, 0.25k boosts)
    COMBO_AO         EMA20-trend + Awesome Oscillator (HPotter, 0.23k boosts)
    COMBO_ATRR       EMA20-trend + ATR Trailing Stop flip (HPotter, 0.16k)
    COMBO_HLCH       EMA20-trend + HL/C Histogram (HPotter, 0.14k boosts)
    COMBO_BEAR_POWER EMA20-trend + Bear Power (HPotter, 0.13k boosts)

Skipped (incompatible): Backtest Adapter (jdehorty), Grid Like (alexgrover),
    Grid Martingale (alexgrover), Average Down (BacktestRookies), Tutorial (RicardoSantos)

Usage:
    python renko/wizard_btc_sweep_v2.py
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

# -- Instrument config ---------------------------------------------------------

LTF_FILE   = "OANDA_BTCUSD.SPOT.US, 1S renko 150.csv"
IS_START   = "2024-06-04"
IS_END     = "2025-09-30"
OOS_START  = "2025-10-01"
OOS_END    = "2026-03-19"
OOS_DAYS   = 170
COMMISSION = 0.0046
CAPITAL    = 1000.0
QTY_VALUE  = 20

COOLDOWNS  = [3, 5, 10, 20]
GATE_MODES = ["none", "psar", "adx25", "psar_adx25"]


# -- Data loading ---------------------------------------------------------------

def _load_data():
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    df = load_renko_export(LTF_FILE)
    add_renko_indicators(df)
    return df


# -- Gate computation -----------------------------------------------------------

def _compute_gate(df, gate_mode):
    n = len(df)
    gate = np.ones(n, dtype=bool)
    if "psar" in gate_mode:
        psar = df["psar_dir"].values
        gate &= (np.isnan(psar) | (psar > 0))
    if "adx25" in gate_mode:
        adx = df["adx"].values
        gate &= (np.isnan(adx) | (adx >= 25))
    return gate


# -- Backtest runner ------------------------------------------------------------

def _run_bt(df, entry, exit_, start, end):
    from engine import BacktestConfig, run_backtest
    df2 = df.copy()
    df2["long_entry"] = entry
    df2["long_exit"]  = exit_

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


# -- Shared helpers ------------------------------------------------------------

def _calc_ema(series, length):
    return pd.Series(series).ewm(span=length, adjust=False).mean().values


def _calc_sma(series, length):
    return pd.Series(series).rolling(length, min_periods=1).mean().values


def _ema20_pos(close, high, low, length=14):
    """
    HPotter EMA20 trend signal: +1 bullish, -1 bearish.
    Translation of Pine's EMA20() function used in all Combo strategies.
    Swing pivot (nXS) around EMA determines trend direction vs prev close.
    """
    n = len(close)
    ema_val = _calc_ema(close, length)

    pos = np.zeros(n)
    for i in range(1, n):
        nHH = max(high[i], high[i-1])
        nLL = min(low[i], low[i-1])
        # nXS: if LL > EMA or HH < EMA → use nLL, else nHH
        nXS = nLL if (nLL > ema_val[i] or nHH < ema_val[i]) else nHH
        if nXS < close[i-1]:
            pos[i] = 1.0
        elif nXS > close[i-1]:
            pos[i] = -1.0
        else:
            pos[i] = pos[i-1]
    return pos


def _make_entry_exit(n, brick_up, cooldown, gate, signal_pos, warmup=30):
    """
    State-machine entry/exit for a +1/-1 signal that doesn't need crossover.
    Enters when signal_pos becomes 1 (transition OR sustained), brick_up and gate pass.
    Exits on first down brick.
    Uses cooldown from last entry bar.
    """
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    for i in range(warmup, n):
        if in_pos:
            if not brick_up[i]:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        # Entry: signal is +1 (just turned or still +1) AND brick is up
        if signal_pos[i] == 1 and (signal_pos[i-1] != 1) and brick_up[i]:
            entry[i] = True
            in_pos = True
            last_bar = i
    return entry, exit_


def _make_entry_exit_cross(n, brick_up, cooldown, gate, cross_signal, warmup=30):
    """
    Entry/exit for a crossover/event signal (True/False per bar).
    Enters when cross_signal[i] is True AND brick_up AND gate AND cooldown.
    Exits on first down brick.
    """
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    for i in range(warmup, n):
        if in_pos:
            if not brick_up[i]:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        if cross_signal[i] and brick_up[i]:
            entry[i] = True
            in_pos = True
            last_bar = i
    return entry, exit_


# ==============================================================================
# SIGNAL GENERATORS
# ==============================================================================

def _gen_three_commas(df, cooldown, gate, fast=21, slow=50):
    """
    3Commas Bot by Bjorgum (13,718 boosts)
    Core: MA crossover. EMA(fast) crosses above EMA(slow) → long.
    Simplified from the full bot framework (which adds swing-based SL/TP).
    """
    n = len(df)
    close = df["Close"].values
    brick_up = df["brick_up"].values

    ema_fast = _calc_ema(close, fast)
    ema_slow = _calc_ema(close, slow)

    # Cross: fast crosses above slow
    cross = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]:
            cross[i] = True

    return _make_entry_exit_cross(n, brick_up, cooldown, gate, cross, warmup=slow+5)


def _gen_double_tap(df, cooldown, gate, pivot_len=5, tol_pct=0.5):
    """
    Bjorgum Double Tap (8,145 boosts)
    Double bottom: two pivot lows at similar price levels within tolerance.
    Entry when price bounces from the second low (brick_up).
    """
    n = len(df)
    close = df["Close"].values
    low = df["Low"].values
    brick_up = df["brick_up"].values

    # Find pivot lows: local minimum over pivot_len bars on each side
    pivot_lows = np.full(n, np.nan)
    for i in range(pivot_len, n - pivot_len):
        window = low[i - pivot_len: i + pivot_len + 1]
        if low[i] == np.min(window):
            pivot_lows[i] = low[i]

    # Detect double bottom: two non-nan pivot lows within last 50 bars,
    # similar price (within tol_pct % of their average height)
    cross = np.zeros(n, dtype=bool)
    for i in range(50, n):
        # Find pivot lows in window [i-50, i-1]
        prev_pivots = [(j, pivot_lows[j]) for j in range(max(0, i-50), i)
                       if not np.isnan(pivot_lows[j])]
        if len(prev_pivots) >= 2:
            # Check most recent two
            p1_val = prev_pivots[-2][1]
            p2_val = prev_pivots[-1][1]
            avg_price = (p1_val + p2_val) / 2.0
            if avg_price > 0:
                diff_pct = abs(p1_val - p2_val) / avg_price * 100.0
                if diff_pct <= tol_pct and not np.isnan(pivot_lows[i-1]):
                    # Just formed a second pivot low at similar level → entry
                    cross[i] = True

    return _make_entry_exit_cross(n, brick_up, cooldown, gate, cross,
                                   warmup=pivot_len * 2 + 10)


def _gen_open_close_cross(df, cooldown, gate, length=8):
    """
    Open Close Cross Strategy NoRepaint by JustUncleL (5,842 boosts)
    EMA(close, L) crosses above EMA(open, L) → long.
    """
    n = len(df)
    close = df["Close"].values
    open_ = df["Open"].values
    brick_up = df["brick_up"].values

    ema_close = _calc_ema(close, length)
    ema_open  = _calc_ema(open_, length)

    cross = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if ema_close[i] > ema_open[i] and ema_close[i-1] <= ema_open[i-1]:
            cross[i] = True

    return _make_entry_exit_cross(n, brick_up, cooldown, gate, cross, warmup=length+5)


def _gen_qqe_cross(df, cooldown, gate, rsi_len=14, sf=8, qqe_factor=5.0, threshold=10):
    """
    [Backtest]QQE Cross v6.0 by JustUncleL (3,012 boosts)
    RSI smoothed → QQE bands. Entry when smoothed RSI exits threshold channel
    from below (RSI > 50 + threshold) — confirmed bullish swing.
    """
    n = len(df)
    close = df["Close"].values
    brick_up = df["brick_up"].values

    # RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(com=rsi_len - 1, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(com=rsi_len - 1, adjust=False).mean().values
    rs = np.where(avg_loss == 0, 100.0, avg_gain / avg_loss)
    rsi = 100.0 - 100.0 / (1.0 + rs)

    # RSIndex = EMA(RSI, sf)
    rsi_index = _calc_ema(rsi, sf)

    # Wilder's ATR of RSI
    atr_rsi = np.abs(np.diff(rsi_index, prepend=rsi_index[0]))
    wilders_period = rsi_len * 2 - 1
    ma_atr_rsi = pd.Series(atr_rsi).ewm(com=wilders_period - 1, adjust=False).mean().values
    delta_fast = pd.Series(ma_atr_rsi).ewm(com=wilders_period - 1, adjust=False).mean().values * qqe_factor

    # QQE bands
    long_band  = np.zeros(n)
    short_band = np.zeros(n)
    trend      = np.ones(n, dtype=int)
    for i in range(1, n):
        new_lb = rsi_index[i] - delta_fast[i]
        new_sb = rsi_index[i] + delta_fast[i]
        long_band[i]  = max(long_band[i-1], new_lb)  if rsi_index[i-1] > long_band[i-1]  and rsi_index[i] > long_band[i-1]  else new_lb
        short_band[i] = min(short_band[i-1], new_sb) if rsi_index[i-1] < short_band[i-1] and rsi_index[i] < short_band[i-1] else new_sb
        if rsi_index[i] > short_band[i-1]:
            trend[i] = 1
        elif rsi_index[i] < long_band[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]

    # Threshold channel cross: RSI exits above 50+threshold
    # QQEclong: RSI > 50+threshold for first bar
    qqe_c_long = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if rsi_index[i] > (50 + threshold) and rsi_index[i-1] <= (50 + threshold):
            qqe_c_long[i] = True

    return _make_entry_exit_cross(n, brick_up, cooldown, gate, qqe_c_long,
                                   warmup=wilders_period + sf + 10)


def _gen_dual_ma(df, cooldown, gate, fast=21, slow=50):
    """
    [Autoview] Dual MA Ribbons R0.12 by JustUncleL (1,568 boosts)
    EMA ribbon: fast EMA crosses slow EMA from below → long.
    Similar to THREE_COMMAS but with default params from the ribbon strategy.
    """
    n = len(df)
    close = df["Close"].values
    brick_up = df["brick_up"].values

    ema_fast = _calc_ema(close, fast)
    ema_slow = _calc_ema(close, slow)

    # Cross: fast crosses above slow (and close above fast = momentum confirmation)
    cross = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1] and close[i] > ema_fast[i]:
            cross[i] = True

    return _make_entry_exit_cross(n, brick_up, cooldown, gate, cross, warmup=slow+5)


def _gen_elder_ray(df, cooldown, gate, ema_len=13, window=5):
    """
    Elder Ray (Bull Power) TP and SL by HPotter (1,425 boosts)
    Bull Power = rolling_max(High, window) - EMA(close, ema_len)
    Entry when Bull Power > 0 (price range above average = bullish energy).
    """
    n = len(df)
    close = df["Close"].values
    high  = df["High"].values
    brick_up = df["brick_up"].values

    ema_val = _calc_ema(close, ema_len)
    # Rolling high over window
    roll_high = pd.Series(high).rolling(window, min_periods=1).max().values

    bull_power = roll_high - ema_val

    # Signal: bull_power crosses above 0
    cross = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if bull_power[i] > 0 and bull_power[i-1] <= 0:
            cross[i] = True

    return _make_entry_exit_cross(n, brick_up, cooldown, gate, cross, warmup=ema_len+window+5)


def _gen_made_atr(df, cooldown, gate, period=9, disp=13, per_ab=0.5, atr_period=15, atr_mult=2.0):
    """
    Moving Average Displaced Envelope & ATRTS by HPotter (1,271 boosts)
    MADE: EMA displaced by 'disp' bars. Top = EMA[disp] * (1 + per_ab/100).
    ATR trailing stop as secondary filter.
    Entry when close breaks above top band AND above ATR trailing stop.
    """
    n = len(df)
    close = df["Close"].values
    high  = df["High"].values
    low   = df["Low"].values
    brick_up = df["brick_up"].values

    ema_val = _calc_ema(close, period)

    # Displaced top band: look back 'disp' bars
    top = np.full(n, np.nan)
    for i in range(disp, n):
        top[i] = ema_val[i - disp] * (1.0 + per_ab / 100.0)

    # ATR trailing stop
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = pd.Series(tr).rolling(atr_period, min_periods=1).mean().values
    n_loss = atr_mult * atr

    atr_stop = np.zeros(n)
    atr_stop[0] = close[0]
    for i in range(1, n):
        if close[i] > atr_stop[i-1] and close[i-1] > atr_stop[i-1]:
            atr_stop[i] = max(atr_stop[i-1], close[i] - n_loss[i])
        elif close[i] < atr_stop[i-1] and close[i-1] < atr_stop[i-1]:
            atr_stop[i] = min(atr_stop[i-1], close[i] + n_loss[i])
        elif close[i] > atr_stop[i-1]:
            atr_stop[i] = close[i] - n_loss[i]
        else:
            atr_stop[i] = close[i] + n_loss[i]

    # Signal: close breaks above displaced envelope AND above ATR stop
    cross = np.zeros(n, dtype=bool)
    for i in range(disp + period + 5, n):
        if np.isnan(top[i]) or np.isnan(top[i-1]):
            continue
        if close[i] > top[i] and close[i-1] <= top[i-1] and close[i] > atr_stop[i]:
            cross[i] = True

    return _make_entry_exit_cross(n, brick_up, cooldown, gate, cross,
                                   warmup=disp + period + atr_period + 5)


def _gen_divergence(df, cooldown, gate, pivot_len=5, rsi_len=14):
    """
    Divergence Strategy by Trendoscope (1,121 boosts)
    Bullish RSI divergence: price makes lower low but RSI makes higher low.
    Entry on brick_up following divergence detection.
    """
    n = len(df)
    close = df["Close"].values
    low   = df["Low"].values
    brick_up = df["brick_up"].values

    # Use pre-computed RSI (pre-shifted) — recompute causal RSI for divergence
    delta = np.diff(close, prepend=close[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(com=rsi_len - 1, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(com=rsi_len - 1, adjust=False).mean().values
    rs    = np.where(avg_l == 0, 100.0, avg_g / avg_l)
    rsi   = 100.0 - 100.0 / (1.0 + rs)

    # Pivot lows in price and RSI
    price_pivot = np.full(n, np.nan)
    rsi_pivot   = np.full(n, np.nan)
    for i in range(pivot_len, n - pivot_len):
        if low[i] == np.min(low[i - pivot_len: i + pivot_len + 1]):
            price_pivot[i] = low[i]
            rsi_pivot[i]   = rsi[i]

    # Divergence: last two pivot lows — price lower but RSI higher
    cross = np.zeros(n, dtype=bool)
    for i in range(pivot_len * 2 + 20, n):
        prev_pivots = [(j, price_pivot[j], rsi_pivot[j])
                       for j in range(max(0, i - 80), i)
                       if not np.isnan(price_pivot[j])]
        if len(prev_pivots) >= 2:
            j1, p1, r1 = prev_pivots[-2]
            j2, p2, r2 = prev_pivots[-1]
            # Bullish divergence: price lower low, RSI higher low
            if p2 < p1 and r2 > r1 and (i - j2) <= pivot_len + 3:
                cross[i] = True

    return _make_entry_exit_cross(n, brick_up, cooldown, gate, cross,
                                   warmup=pivot_len * 2 + rsi_len + 10)


def _gen_inside_bar(df, cooldown, gate):
    """
    Babypips: Inside Bar Momentum Strategy by BacktestRookies (761 boosts)
    Inside bar: high[1] > high[0] and low[1] < low[0].
    Bullish setup: previous bar was bullish (close[1] > open[1]).
    Entry on next brick_up after bullish inside bar.
    """
    n = len(df)
    close    = df["Close"].values
    open_    = df["Open"].values
    high     = df["High"].values
    low      = df["Low"].values
    brick_up = df["brick_up"].values

    inside_bar = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if high[i-1] > high[i] and low[i-1] < low[i]:
            inside_bar[i] = True

    # Bullish inside bar: parent bar was bullish
    bullish_ib = np.zeros(n, dtype=bool)
    for i in range(2, n):
        if inside_bar[i-1] and close[i-2] > open_[i-2]:
            bullish_ib[i] = True  # Bar after pattern forms

    return _make_entry_exit_cross(n, brick_up, cooldown, gate, bullish_ib, warmup=10)


def _gen_okx_ma(df, cooldown, gate, length=13):
    """
    OKX: MA Crossover by HPotter (644 boosts)
    Signal: low dips below SMA[1] (price touches MA from above → bounce entry).
    Entry when low[i] < SMA[i-1] transition (price visits SMA) on brick_up.
    """
    n = len(df)
    close    = df["Close"].values
    low      = df["Low"].values
    brick_up = df["brick_up"].values

    sma = _calc_sma(close, length)

    # doLong = low < SMA[1] — price visits MA from above
    # Entry when doLong transitions from False to True
    cross = np.zeros(n, dtype=bool)
    for i in range(2, n):
        do_long_prev2 = low[i-2] < sma[i-3] if i >= 3 else False
        do_long_now   = low[i-1] < sma[i-2]
        if do_long_now and not do_long_prev2:
            cross[i] = True

    return _make_entry_exit_cross(n, brick_up, cooldown, gate, cross, warmup=length+5)


def _gen_hlhb(df, cooldown, gate, ema_fast=5, ema_slow=10, rsi_len=10):
    """
    Hucklekiwi Pip HLHB Trend-Catcher System by BacktestRookies (416 boosts)
    EMA(5) crosses EMA(10) from below AND RSI(10) crosses 50 from below.
    Both must occur within 2 bars of each other.
    """
    n = len(df)
    close    = df["Close"].values
    brick_up = df["brick_up"].values

    ema5  = _calc_ema(close, ema_fast)
    ema10 = _calc_ema(close, ema_slow)

    # RSI
    delta = np.diff(close, prepend=close[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(com=rsi_len - 1, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(com=rsi_len - 1, adjust=False).mean().values
    rs    = np.where(avg_l == 0, 100.0, avg_g / avg_l)
    rsi   = 100.0 - 100.0 / (1.0 + rs)

    ema_cross = np.zeros(n, dtype=bool)
    rsi_cross = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if ema5[i] > ema10[i] and ema5[i-1] <= ema10[i-1]:
            ema_cross[i] = True
        if rsi[i] > 50 and rsi[i-1] <= 50:
            rsi_cross[i] = True

    # Both crosses within 3 bars of each other
    signal = np.zeros(n, dtype=bool)
    for i in range(3, n):
        ec_recent = any(ema_cross[max(0, i-2):i+1])
        rc_recent = any(rsi_cross[max(0, i-2):i+1])
        if ec_recent and rc_recent:
            signal[i] = True

    return _make_entry_exit_cross(n, brick_up, cooldown, gate, signal, warmup=ema_slow+rsi_len)


def _gen_combo_bull_power(df, cooldown, gate, ema_len=14, sell_level=-15.0):
    """
    Combo 2/20 EMA & Bull Power by HPotter (416 boosts)
    EMA20-trend == +1 AND Bull Power SMA > sell_level → long.
    Bull Power value from Vadim Gimelfarb's indicator (high/low vs prev).
    Simplified: SMA(bull_value, 15) > sell_level where bull_value = f(OHLC).
    """
    n = len(df)
    close    = df["Close"].values
    open_    = df["Open"].values
    high     = df["High"].values
    low      = df["Low"].values
    brick_up = df["brick_up"].values

    ema20_pos = _ema20_pos(close, high, low, ema_len)

    # Bull Power value (from Pine: complex OHLC measure)
    bp_val = np.zeros(n)
    for i in range(1, n):
        c, o, h, l, pc, po = close[i], open_[i], high[i], low[i], close[i-1], open_[i-1]
        if c < o:
            bp_val[i] = max(h - pc, c - l) if pc < o else max(h - o, c - l)
        elif c > o:
            bp_val[i] = h - l if pc > o else max(o - pc, h - l)
        else:
            if h - c > c - l:
                bp_val[i] = max(h - pc, c - l) if pc < o else h - o
            elif h - c < c - l:
                bp_val[i] = h - l if pc > o else max(o - c, h - l)
            else:
                bp_val[i] = max(h - o, c - l) if pc > o else (max(o - c, h - l) if pc < o else h - l)

    bp_sma = _calc_sma(bp_val, 15)

    # Signal: state transition where ema20 == +1 AND bp_sma > sell_level
    sig_pos = np.where((ema20_pos == 1) & (bp_sma > sell_level), 1.0, -1.0)

    return _make_entry_exit(n, brick_up, cooldown, gate, sig_pos, warmup=ema_len + 20)


def _gen_ema_ma_cross(df, cooldown, gate, ema_len=9, sma_len=21):
    """
    EMA & MA Crossover Strategy by HPotter (386 boosts)
    Basic EMA crosses above SMA → long.
    """
    n = len(df)
    close    = df["Close"].values
    brick_up = df["brick_up"].values

    ema_val = _calc_ema(close, ema_len)
    sma_val = _calc_sma(close, sma_len)

    cross = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if ema_val[i] > sma_val[i] and ema_val[i-1] <= sma_val[i-1]:
            cross[i] = True

    return _make_entry_exit_cross(n, brick_up, cooldown, gate, cross, warmup=sma_len+5)


def _gen_combo_cci(df, cooldown, gate, ema_len=14, cci_len=10, fast_ma=10, slow_ma=20):
    """
    Combo 2/20 EMA & CCI by HPotter (378 boosts)
    EMA20-trend == +1 AND CCI fast_ma > CCI slow_ma → long.
    CCI(close, cci_len): fast SMA > slow SMA of the CCI.
    """
    n = len(df)
    close    = df["Close"].values
    high     = df["High"].values
    low      = df["Low"].values
    brick_up = df["brick_up"].values

    ema20_pos = _ema20_pos(close, high, low, ema_len)

    # CCI
    typical = (high + low + close) / 3.0
    cci_sma  = _calc_sma(typical, cci_len)
    mad = pd.Series(typical).rolling(cci_len, min_periods=1).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci_val = np.where(mad == 0, 0.0, (typical - cci_sma) / (0.015 * mad))

    cci_fast = _calc_sma(cci_val, fast_ma)
    cci_slow = _calc_sma(cci_val, slow_ma)

    # Signal: ema20 +1 AND cci fast > slow (recent crossover)
    sig_pos = np.where((ema20_pos == 1) & (cci_fast > cci_slow), 1.0, -1.0)

    return _make_entry_exit(n, brick_up, cooldown, gate, sig_pos,
                             warmup=ema_len + cci_len + slow_ma + 5)


def _gen_dema_rsi(df, cooldown, gate, ma_len=21, rsi_len=4, smooth=4):
    """
    DemaRSI Strategy by RicardoSantos (290 boosts)
    DEMA = EMA(EMA(close, L), L). RSI(DEMA, rsi_len) smoothed with EMA(smooth).
    Entry when RSI_DEMA crosses above smooth EMA → bullish momentum.
    """
    n = len(df)
    close    = df["Close"].values
    brick_up = df["brick_up"].values

    # DEMA = 2*EMA - EMA(EMA)
    ema1  = _calc_ema(close, ma_len)
    ema2  = _calc_ema(ema1, ma_len)
    dema  = 2.0 * ema1 - ema2

    # RSI of DEMA
    delta = np.diff(dema, prepend=dema[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(com=rsi_len - 1, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(com=rsi_len - 1, adjust=False).mean().values
    rs    = np.where(avg_l == 0, 100.0, avg_g / avg_l)
    mar_si = 100.0 - 100.0 / (1.0 + rs)

    smoothed = _calc_ema(mar_si, smooth)

    # Entry: marsi crosses above smoothed
    cross = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if mar_si[i] > smoothed[i] and mar_si[i-1] <= smoothed[i-1]:
            cross[i] = True

    return _make_entry_exit_cross(n, brick_up, cooldown, gate, cross,
                                   warmup=ma_len * 2 + rsi_len + smooth + 5)


def _gen_combo_bbb(df, cooldown, gate, ema_len=14, sell_level=-15.0, buy_level=15.0):
    """
    Combo 2/20 EMA & Bull And Bear Balance by HPotter (250 boosts)
    EMA20-trend +1 AND Bull-Bear Balance (nBBB = value2 - value) >= buy_level.
    Bull-Bear Balance from Vadim Gimelfarb.
    """
    n = len(df)
    close    = df["Close"].values
    open_    = df["Open"].values
    high     = df["High"].values
    low      = df["Low"].values
    brick_up = df["brick_up"].values

    ema20_pos = _ema20_pos(close, high, low, ema_len)

    # Bear component (value)
    bear = np.zeros(n)
    bull = np.zeros(n)
    for i in range(1, n):
        c, o, h, l, pc, po = close[i], open_[i], high[i], low[i], close[i-1], open_[i-1]
        # Bear value
        if c < o:
            bear[i] = h - l if pc > o else max(c - o, h - l)
        elif c > o:
            bear[i] = max(h - pc, c - l) if pc > o else max(h - o, c - l)
        else:
            if h - c > c - l:
                bear[i] = max(h - pc, c - l) if pc < o else h - o
            elif h - c < c - l:
                bear[i] = max(c - l, h - c) if c > o else o - l
            else:
                bear[i] = max(h - o, c - l) if pc > o else (max(o - c, h - l) if pc < o else h - l)

        # Bull value (same logic as bp_val in bull power)
        if c < o:
            bull[i] = max(h - pc, c - l) if pc < o else max(h - o, c - l)
        elif c > o:
            bull[i] = h - l if pc > o else max(o - pc, h - l)
        else:
            if h - c > c - l:
                bull[i] = max(h - pc, c - l) if pc < o else h - o
            elif h - c < c - l:
                bull[i] = h - l if pc > o else max(o - c, h - l)
            else:
                bull[i] = max(h - o, c - l) if pc > o else (max(o - c, h - l) if pc < o else h - l)

    nBBB = bull - bear

    sig_pos = np.where((ema20_pos == 1) & (nBBB >= buy_level), 1.0, -1.0)

    return _make_entry_exit(n, brick_up, cooldown, gate, sig_pos, warmup=ema_len + 5)


def _gen_combo_ao(df, cooldown, gate, ema_len=14, slow=34, fast=5, wma_len=15):
    """
    Combo 2/20 EMA & Bill Awesome Oscillator (AC) by HPotter (232 boosts)
    EMA20-trend +1 AND AO_WMA crosses zero from below.
    AO = SMA(hl2, fast) - SMA(hl2, slow). AC = AO - WMA(AO, wma_len).
    Signal: AC WMA crosses zero (pos transition).
    """
    n = len(df)
    close    = df["Close"].values
    high     = df["High"].values
    low      = df["Low"].values
    brick_up = df["brick_up"].values

    ema20_pos = _ema20_pos(close, high, low, ema_len)

    hl2 = (high + low) / 2.0
    ao = _calc_sma(hl2, fast) - _calc_sma(hl2, slow)
    # AC = AO - SMA(AO, fast) — note HPotter uses xSMA1_SMA2 - xSMA(that, fast)
    ao_sma = _calc_sma(ao, fast)
    ac = ao - ao_sma
    ac_wma = pd.Series(ac).rolling(wma_len, min_periods=1).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True).values

    # AC signal: ac_wma crosses zero
    ac_sig_pos = np.zeros(n)
    for i in range(2, n):
        if ac_wma[i-1] < 0 and ac_wma[i-2] < 0:
            pass  # stays -1 from prev
        elif ac_wma[i-1] > 0 and ac_wma[i-2] < 0:
            ac_sig_pos[i] = 1.0
        elif ac_wma[i-1] < 0 and ac_wma[i-2] > 0:
            ac_sig_pos[i] = -1.0
        else:
            ac_sig_pos[i] = ac_sig_pos[i-1]

    # Recompute as sustained state
    ac_state = np.zeros(n)
    for i in range(1, n):
        if ac_wma[i] > 0:
            ac_state[i] = 1.0
        elif ac_wma[i] < 0:
            ac_state[i] = -1.0
        else:
            ac_state[i] = ac_state[i-1]

    sig_pos = np.where((ema20_pos == 1) & (ac_state == 1), 1.0, -1.0)

    return _make_entry_exit(n, brick_up, cooldown, gate, sig_pos,
                             warmup=ema_len + slow + wma_len + 5)


def _gen_combo_atrr(df, cooldown, gate, ema_len=14, atr_period=5, atr_mult=3.5):
    """
    Combo 2/20 EMA & ATR Reversed by HPotter (162 boosts)
    EMA20-trend +1 AND ATR trailing stop flips bullish (close crosses above ATR stop).
    ATR stop: ratcheting stop that follows price with ATR*mult distance.
    """
    n = len(df)
    close    = df["Close"].values
    high     = df["High"].values
    low      = df["Low"].values
    brick_up = df["brick_up"].values

    ema20_pos = _ema20_pos(close, high, low, ema_len)

    # ATR
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = pd.Series(tr).rolling(atr_period, min_periods=1).mean().values
    n_loss = atr_mult * atr

    atr_stop = np.zeros(n)
    atr_stop[0] = close[0] + n_loss[0]
    for i in range(1, n):
        if close[i] > atr_stop[i-1] and close[i-1] > atr_stop[i-1]:
            atr_stop[i] = max(atr_stop[i-1], close[i] - n_loss[i])
        elif close[i] < atr_stop[i-1] and close[i-1] < atr_stop[i-1]:
            atr_stop[i] = min(atr_stop[i-1], close[i] + n_loss[i])
        elif close[i] > atr_stop[i-1]:
            atr_stop[i] = close[i] - n_loss[i]
        else:
            atr_stop[i] = close[i] + n_loss[i]

    # ATR reversed position: +1 when close crosses above atr_stop
    atrr_pos = np.zeros(n)
    for i in range(1, n):
        if close[i-1] < atr_stop[i-1] and close[i] > atr_stop[i-1]:
            atrr_pos[i] = 1.0
        elif close[i-1] > atr_stop[i-1] and close[i] < atr_stop[i-1]:
            atrr_pos[i] = -1.0
        else:
            atrr_pos[i] = atrr_pos[i-1]

    sig_pos = np.where((ema20_pos == 1) & (atrr_pos == 1), 1.0, -1.0)

    return _make_entry_exit(n, brick_up, cooldown, gate, sig_pos,
                             warmup=ema_len + atr_period + 10)


def _gen_combo_hlch(df, cooldown, gate, ema_len=14, look_back=20, sma_len=16):
    """
    Combo 2/20 EMA & (H-L)/C Histogram by HPotter (136 boosts)
    EMA20-trend +1 AND current HL/C < SMA of HL/C (low volatility = inside bars).
    hlch = (high-low)/close. Signal +1 when SMA(|hlch|)[look_back] > |hlch|.
    """
    n = len(df)
    close    = df["Close"].values
    high     = df["High"].values
    low      = df["Low"].values
    brick_up = df["brick_up"].values

    ema20_pos = _ema20_pos(close, high, low, ema_len)

    hlch = np.where(close == 0, 0.0, (high - low) / close)
    hlch_abs_sma = _calc_sma(np.abs(hlch), sma_len)

    # Signal: historical SMA (look_back bars ago) > current abs value → contraction
    hlch_pos = np.zeros(n)
    for i in range(look_back, n):
        if hlch_abs_sma[i - look_back] > abs(hlch[i]):
            hlch_pos[i] = 1.0
        elif hlch_abs_sma[i - look_back] < abs(hlch[i]):
            hlch_pos[i] = -1.0
        else:
            hlch_pos[i] = hlch_pos[i-1]

    sig_pos = np.where((ema20_pos == 1) & (hlch_pos == 1), 1.0, -1.0)

    return _make_entry_exit(n, brick_up, cooldown, gate, sig_pos,
                             warmup=ema_len + look_back + sma_len + 5)


def _gen_combo_bear_power(df, cooldown, gate, ema_len=14, sell_level=10.0, buy_level=1.0):
    """
    Combo 2/20 EMA & Bear Power by HPotter (127 boosts)
    EMA20-trend +1 AND Bear Power <= buy_level (low bear pressure = bullish).
    Bear Power = complex OHLC formula, bullish when value <= buy_level.
    """
    n = len(df)
    close    = df["Close"].values
    open_    = df["Open"].values
    high     = df["High"].values
    low      = df["Low"].values
    brick_up = df["brick_up"].values

    ema20_pos = _ema20_pos(close, high, low, ema_len)

    # Bear Power value
    bp_val = np.zeros(n)
    for i in range(1, n):
        c, o, h, l, pc, po = close[i], open_[i], high[i], low[i], close[i-1], open_[i-1]
        if c < o:
            bp_val[i] = h - l if pc > o else max(c - o, h - l)
        elif c > o:
            bp_val[i] = max(h - pc, c - l) if pc > o else max(h - o, c - l)
        else:
            if h - c > c - l:
                bp_val[i] = max(h - pc, c - l) if pc < o else h - o
            elif h - c < c - l:
                bp_val[i] = max(c - l, h - c) if c > o else o - l
            else:
                bp_val[i] = max(h - o, c - l) if pc > o else (max(o - c, h - l) if pc < o else h - l)

    # Bear power signal: <= buy_level → long, > sell_level → short
    bp_pos = np.zeros(n)
    for i in range(1, n):
        if bp_val[i] <= buy_level:
            bp_pos[i] = 1.0
        elif bp_val[i] > sell_level:
            bp_pos[i] = -1.0
        else:
            bp_pos[i] = bp_pos[i-1]

    sig_pos = np.where((ema20_pos == 1) & (bp_pos == 1), 1.0, -1.0)

    return _make_entry_exit(n, brick_up, cooldown, gate, sig_pos, warmup=ema_len + 5)


# ==============================================================================
# STRATEGY REGISTRY
# ==============================================================================

STRATEGIES = {
    "THREE_COMMAS": {
        "fn": _gen_three_commas,
        "params": {"fast": [9, 21, 34], "slow": [21, 50, 89]},
        "desc": "3Commas Bot MA crossover (Bjorgum 13.7k)",
    },
    "DOUBLE_TAP": {
        "fn": _gen_double_tap,
        "params": {"pivot_len": [3, 5, 8], "tol_pct": [0.3, 0.5, 1.0, 2.0]},
        "desc": "Bjorgum Double Tap double-bottom (8.1k)",
    },
    "OPEN_CLOSE_CROSS": {
        "fn": _gen_open_close_cross,
        "params": {"length": [5, 8, 13, 21]},
        "desc": "Open Close Cross EMA(close)/EMA(open) (JustUncleL 5.8k)",
    },
    "QQE_CROSS": {
        "fn": _gen_qqe_cross,
        "params": {"rsi_len": [10, 14], "sf": [5, 8], "qqe_factor": [3.0, 5.0], "threshold": [5, 10]},
        "desc": "QQE Cross v6 RSI threshold channel (JustUncleL 3.0k)",
    },
    "DUAL_MA": {
        "fn": _gen_dual_ma,
        "params": {"fast": [9, 13, 21], "slow": [21, 34, 50]},
        "desc": "Dual MA Ribbons EMA cross+confirm (JustUncleL 1.6k)",
    },
    "ELDER_RAY": {
        "fn": _gen_elder_ray,
        "params": {"ema_len": [9, 13, 21], "window": [3, 5, 8]},
        "desc": "Elder Ray Bull Power (HPotter 1.4k)",
    },
    "MADE_ATR": {
        "fn": _gen_made_atr,
        "params": {"period": [7, 9, 13], "disp": [9, 13, 20], "per_ab": [0.3, 0.5, 1.0]},
        "desc": "Displaced EMA Envelope + ATR stop (HPotter 1.3k)",
    },
    "DIVERGENCE": {
        "fn": _gen_divergence,
        "params": {"pivot_len": [3, 5, 8], "rsi_len": [10, 14]},
        "desc": "Bullish RSI Divergence (Trendoscope 1.1k)",
    },
    "INSIDE_BAR": {
        "fn": _gen_inside_bar,
        "params": {},
        "desc": "Babypips Inside Bar Momentum (BacktestRookies 0.8k)",
    },
    "OKX_MA": {
        "fn": _gen_okx_ma,
        "params": {"length": [8, 13, 21, 34]},
        "desc": "OKX MA price-visits-SMA bounce (HPotter 0.6k)",
    },
    "HLHB": {
        "fn": _gen_hlhb,
        "params": {"ema_fast": [5, 8], "ema_slow": [10, 13, 21], "rsi_len": [10, 14]},
        "desc": "HLHB EMA+RSI50 cross (BacktestRookies 0.4k)",
    },
    "COMBO_BULL_POWER": {
        "fn": _gen_combo_bull_power,
        "params": {"ema_len": [10, 14, 20], "sell_level": [-20.0, -15.0, 0.0]},
        "desc": "Combo EMA20+Bull Power (HPotter 0.4k)",
    },
    "EMA_MA_CROSS": {
        "fn": _gen_ema_ma_cross,
        "params": {"ema_len": [5, 9, 13], "sma_len": [13, 21, 34]},
        "desc": "EMA & MA Crossover (HPotter 0.4k)",
    },
    "COMBO_CCI": {
        "fn": _gen_combo_cci,
        "params": {"ema_len": [10, 14], "cci_len": [10, 14], "fast_ma": [5, 10], "slow_ma": [15, 20]},
        "desc": "Combo EMA20+CCI cross (HPotter 0.4k)",
    },
    "DEMA_RSI": {
        "fn": _gen_dema_rsi,
        "params": {"ma_len": [14, 21], "rsi_len": [4, 7], "smooth": [4, 7]},
        "desc": "DemaRSI DEMA+RSI smoothing (RicardoSantos 0.3k)",
    },
    "COMBO_BBB": {
        "fn": _gen_combo_bbb,
        "params": {"ema_len": [10, 14, 20], "buy_level": [5.0, 15.0, 30.0]},
        "desc": "Combo EMA20+Bull-Bear Balance (HPotter 0.25k)",
    },
    "COMBO_AO": {
        "fn": _gen_combo_ao,
        "params": {"ema_len": [10, 14], "slow": [26, 34], "fast": [5, 8]},
        "desc": "Combo EMA20+Awesome Oscillator (HPotter 0.23k)",
    },
    "COMBO_ATRR": {
        "fn": _gen_combo_atrr,
        "params": {"ema_len": [10, 14], "atr_period": [5, 9, 14], "atr_mult": [2.5, 3.5]},
        "desc": "Combo EMA20+ATR Trailing Stop flip (HPotter 0.16k)",
    },
    "COMBO_HLCH": {
        "fn": _gen_combo_hlch,
        "params": {"ema_len": [10, 14], "look_back": [10, 20], "sma_len": [12, 16]},
        "desc": "Combo EMA20+(H-L)/C Histogram (HPotter 0.14k)",
    },
    "COMBO_BEAR_POWER": {
        "fn": _gen_combo_bear_power,
        "params": {"ema_len": [10, 14, 20], "sell_level": [5.0, 10.0, 20.0], "buy_level": [0.5, 1.0, 2.0]},
        "desc": "Combo EMA20+Bear Power (HPotter 0.13k)",
    },
}


# -- Worker -----------------------------------------------------------------------

_worker_cache = {}


def _worker_init():
    _worker_cache["df"] = _load_data()


def _run_single(task):
    df = _worker_cache["df"]
    strat_name = task["strategy"]
    strat = STRATEGIES[strat_name]
    fn = strat["fn"]
    cd = task["cooldown"]
    gate_mode = task["gate"]
    params = task.get("params", {})

    gate = _compute_gate(df, gate_mode)

    try:
        entry, exit_ = fn(df, cooldown=cd, gate=gate, **params)
    except Exception as e:
        return {**task, "error": str(e)}

    is_kpis  = _run_bt(df, entry, exit_, IS_START,  IS_END)
    oos_kpis = _run_bt(df, entry, exit_, OOS_START, OOS_END)
    oos_kpis["tpd"] = round(oos_kpis["trades"] / OOS_DAYS, 2)

    return {**task, "is": is_kpis, "oos": oos_kpis}


# -- Build tasks ------------------------------------------------------------------

def _build_tasks():
    import itertools
    tasks = []
    for strat_name, strat in STRATEGIES.items():
        grid = strat["params"]
        if grid:
            keys = list(grid.keys())
            combos = list(itertools.product(*[grid[k] for k in keys]))
        else:
            keys = []
            combos = [()]

        for combo in combos:
            params = dict(zip(keys, combo))
            for cd in COOLDOWNS:
                for gate in GATE_MODES:
                    tasks.append({
                        "strategy": strat_name,
                        "cooldown": cd,
                        "gate": gate,
                        "params": params,
                    })
    return tasks


# -- Main -------------------------------------------------------------------------

def main():
    tasks = _build_tasks()
    print(f"Wizard BTC Sweep v2: {len(tasks)} combos across {len(STRATEGIES)} strategies")
    print(f"Strategies: {', '.join(STRATEGIES.keys())}")
    print(f"Workers: {MAX_WORKERS}\n")

    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=_worker_init) as pool:
        futures = {pool.submit(_run_single, t): t for t in tasks}
        done = 0
        for f in as_completed(futures):
            done += 1
            r = f.result()
            results.append(r)
            if done % 100 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} complete...")

    valid = [r for r in results if "error" not in r and r["oos"]["trades"] >= 10]
    valid.sort(key=lambda x: x["oos"]["pf"], reverse=True)

    print(f"\n{'='*100}")
    print(f"TOP 30 BY OOS PF (min 10 trades)")
    print(f"{'='*100}")
    print(f"{'Strategy':<18} {'Params':<35} {'Gate':<12} {'CD':>3}  "
          f"{'IS_PF':>7} {'IS_T':>5} {'IS_WR':>6}  "
          f"{'OOS_PF':>7} {'OOS_T':>5} {'OOS_WR':>6} {'t/d':>5} {'OOS_Net':>9}")
    print("-" * 100)

    for r in valid[:30]:
        p = r["params"]
        param_str = " ".join(f"{k}={v}" for k, v in p.items()) if p else "-"
        pf_str = "INF" if math.isinf(r["oos"]["pf"]) else f"{r['oos']['pf']:.2f}"
        print(f"{r['strategy']:<18} {param_str:<35} {r['gate']:<12} {r['cooldown']:>3}  "
              f"{r['is']['pf']:>7.2f} {r['is']['trades']:>5} {r['is']['wr']:>5.1f}%  "
              f"{pf_str:>7} {r['oos']['trades']:>5} {r['oos']['wr']:>5.1f}% "
              f"{r['oos']['tpd']:>5.1f} {r['oos']['net']:>9.2f}")

    # Save results
    out_file = ROOT / "ai_context" / "wizard_btc_sweep_v2_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {len(results)} results to {out_file}")

    # Best per strategy
    print(f"\n{'='*80}")
    print("BEST CONFIG PER STRATEGY (by OOS PF, min 10 trades)")
    print(f"{'='*80}")
    for strat_name in STRATEGIES:
        strat_results = [r for r in valid if r["strategy"] == strat_name]
        if strat_results:
            best = strat_results[0]
            p = best["params"]
            param_str = " ".join(f"{k}={v}" for k, v in p.items()) if p else "-"
            pf_str = "INF" if math.isinf(best["oos"]["pf"]) else f"{best['oos']['pf']:.2f}"
            print(f"  {strat_name:<18} {param_str:<30} gate={best['gate']:<12} cd={best['cooldown']:<3} "
                  f"OOS: PF={pf_str} T={best['oos']['trades']} "
                  f"WR={best['oos']['wr']:.1f}% t/d={best['oos']['tpd']:.1f}")
        else:
            print(f"  {strat_name:<18} -- no qualifying results --")

    return results


if __name__ == "__main__":
    main()
