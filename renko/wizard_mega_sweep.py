#!/usr/bin/env python3
"""
wizard_mega_sweep.py -- All 27 Pine Wizard strategies × all 9 instruments

Combines all signal generators from wizard_btc_sweep.py (7 strategies)
and wizard_btc_sweep_v2.py (20 strategies) and runs them across every
instrument in the cross-asset universe.

Skipped (incompatible): Backtest Adapter, Grid Like, Grid Martingale,
    Average Down, Tutorial (same exclusions as BTC sweeps).

Architecture: per-process lazy instrument cache (same pattern as cross_asset_sweep.py).
All strategies are long-only with first-down-brick exit.
Gates: none | psar | adx25 | psar_adx25
Cooldowns: 3, 5, 10, 20

Output:
    ai_context/wizard_mega_sweep_results.json
    ai_context/wizard_mega_sweep_report.md

Usage:
    python renko/wizard_mega_sweep.py
"""

import contextlib
import io
import itertools
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

# ── Instrument configs (from cross_asset_sweep.py) ─────────────────────────

INSTRUMENTS = {
    "EURUSD_4": {
        "file": "OANDA_EURUSD, 1S renko 0.0004.csv",
        "is_start": "2023-01-23", "is_end": "2025-09-30",
        "oos_start": "2025-10-01", "oos_end": "2026-03-19",
        "oos_days": 170, "commission": 0.0046, "capital": 1000.0,
        "qty_type": "cash", "qty_value": 20,
    },
    "EURUSD_5": {
        "file": "OANDA_EURUSD, 1S renko 0.0005.csv",
        "is_start": "2022-05-18", "is_end": "2025-09-30",
        "oos_start": "2025-10-01", "oos_end": "2026-03-19",
        "oos_days": 170, "commission": 0.0046, "capital": 1000.0,
        "qty_type": "cash", "qty_value": 20,
    },
    "GBPJPY": {
        "file": "OANDA_GBPJPY, 1S renko 0.05.csv",
        "is_start": "2024-11-21", "is_end": "2025-09-30",
        "oos_start": "2025-10-01", "oos_end": "2026-03-11",
        "oos_days": 162, "commission": 0.005, "capital": 150000.0,
        "qty_type": "cash", "qty_value": 20,
    },
    "EURAUD": {
        "file": "OANDA_EURAUD, 1S renko 0.0006.csv",
        "is_start": "2023-07-20", "is_end": "2025-09-30",
        "oos_start": "2025-10-01", "oos_end": "2026-03-17",
        "oos_days": 168, "commission": 0.009, "capital": 1000.0,
        "qty_type": "cash", "qty_value": 20,
    },
    "GBPUSD": {
        "file": "OANDA_GBPUSD, 1S renko 0.0004.csv",
        "is_start": "2024-05-23", "is_end": "2025-09-30",
        "oos_start": "2025-10-01", "oos_end": "2026-03-17",
        "oos_days": 168, "commission": 0.005, "capital": 1000.0,
        "qty_type": "cash", "qty_value": 20,
    },
    "USDJPY": {
        "file": "OANDA_USDJPY, 1S renko 0.05.csv",
        "is_start": "2024-05-16", "is_end": "2025-09-30",
        "oos_start": "2025-10-01", "oos_end": "2026-03-17",
        "oos_days": 168, "commission": 0.005, "capital": 1000.0,
        "qty_type": "cash", "qty_value": 20,
    },
    "USDCHF": {
        "file": "OANDA_USDCHF, 1S renko 0.0005.csv",
        "is_start": "2024-01-01", "is_end": "2025-09-30",
        "oos_start": "2025-10-01", "oos_end": "2026-03-19",
        "oos_days": 170, "commission": 0.005, "capital": 1000.0,
        "qty_type": "cash", "qty_value": 20,
    },
    "BTCUSD": {
        "file": "OANDA_BTCUSD.SPOT.US, 1S renko 150.csv",
        "is_start": "2024-06-04", "is_end": "2025-09-30",
        "oos_start": "2025-10-01", "oos_end": "2026-03-19",
        "oos_days": 170, "commission": 0.0046, "capital": 1000.0,
        "qty_type": "cash", "qty_value": 20,
    },
    "MYM_14": {
        "file": "CBOT_MINI_MYM1!, 1S renko 14.csv",
        "is_start": "2025-03-07", "is_end": "2025-12-31",
        "oos_start": "2026-01-01", "oos_end": "2026-03-19",
        "oos_days": 78, "commission": 0.00475, "capital": 1000.0,
        "qty_type": "cash", "qty_value": 0.5,
    },
}

GATE_MODES = ["none", "psar", "adx25", "psar_adx25"]
COOLDOWNS  = [3, 5, 10, 20]

# ── Per-process lazy cache ──────────────────────────────────────────────────

_w = {}


def _load_inst(inst_key):
    cache_key = f"df_{inst_key}"
    if cache_key in _w:
        return
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    cfg = INSTRUMENTS[inst_key]
    df = load_renko_export(cfg["file"])
    add_renko_indicators(df)
    _w[cache_key] = df


# ── Backtest runner ─────────────────────────────────────────────────────────

def _run_bt(inst_key, entry, exit_, start, end):
    from engine import BacktestConfig, run_backtest
    cfg = INSTRUMENTS[inst_key]
    df2 = _w[f"df_{inst_key}"].copy()
    df2["long_entry"] = entry
    df2["long_exit"]  = exit_
    df2["short_entry"] = np.zeros(len(df2), dtype=bool)
    df2["short_exit"]  = np.zeros(len(df2), dtype=bool)
    bt_cfg = BacktestConfig(
        initial_capital=cfg["capital"], commission_pct=cfg["commission"],
        slippage_ticks=0, qty_type=cfg["qty_type"],
        qty_value=cfg["qty_value"], pyramiding=1,
        start_date=start, end_date=end,
        take_profit_pct=0.0, stop_loss_pct=0.0,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest(df2, bt_cfg)
    pf = kpis.get("profit_factor", 0.0) or 0.0
    return {
        "pf":     float("inf") if math.isinf(pf) else round(float(pf), 2),
        "net":    round(float(kpis.get("net_profit", 0.0) or 0.0), 4),
        "trades": int(kpis.get("total_trades", 0) or 0),
        "wr":     round(float(kpis.get("win_rate", 0.0) or 0.0), 1),
        "dd":     round(float(kpis.get("max_drawdown_pct", 0.0) or 0.0), 2),
    }


# ── Gate helper ─────────────────────────────────────────────────────────────

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


# ── Shared helpers (from v2) ────────────────────────────────────────────────

def _calc_ema(series, length):
    return pd.Series(series).ewm(span=length, adjust=False).mean().values


def _calc_sma(series, length):
    return pd.Series(series).rolling(length, min_periods=1).mean().values


def _ema20_pos(close, high, low, length=14):
    n = len(close)
    ema_val = _calc_ema(close, length)
    pos = np.zeros(n)
    for i in range(1, n):
        nHH = max(high[i], high[i-1])
        nLL = min(low[i], low[i-1])
        nXS = nLL if (nLL > ema_val[i] or nHH < ema_val[i]) else nHH
        if nXS < close[i-1]:
            pos[i] = 1.0
        elif nXS > close[i-1]:
            pos[i] = -1.0
        else:
            pos[i] = pos[i-1]
    return pos


def _make_entry_exit(n, brick_up, cooldown, gate, signal_pos, warmup=30):
    """State transition entry: signal goes from non-1 to 1 + brick_up."""
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
        if signal_pos[i] == 1 and (signal_pos[i-1] != 1) and brick_up[i]:
            entry[i] = True
            in_pos = True
            last_bar = i
    return entry, exit_


def _make_entry_exit_cross(n, brick_up, cooldown, gate, cross_signal, warmup=30):
    """Event-based entry: cross_signal fires once."""
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
# SIGNAL GENERATORS — v1 (from wizard_btc_sweep.py)
# ==============================================================================

def _gen_alpha_trend(df, cooldown, gate, period=14, coeff=1.0):
    n = len(df)
    brick_up = df["brick_up"].values
    high = df["High"].values
    low  = df["Low"].values
    close = df["Close"].values
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    atr_sma = pd.Series(tr).rolling(period, min_periods=1).mean().values
    rsi = df["rsi"].values
    alpha_trend = np.zeros(n)
    for i in range(1, n):
        upT   = low[i]  - atr_sma[i] * coeff
        downT = high[i] + atr_sma[i] * coeff
        rsi_v = rsi[i] if not np.isnan(rsi[i]) else 50.0
        alpha_trend[i] = max(upT, alpha_trend[i-1]) if rsi_v >= 50 else min(downT, alpha_trend[i-1])
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = max(period + 5, 30)
    for i in range(warmup, n):
        if in_pos:
            if not brick_up[i]: exit_[i] = True; in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown: continue
        if i >= 3 and alpha_trend[i] > alpha_trend[i-2] and alpha_trend[i-1] <= alpha_trend[i-3] and brick_up[i]:
            entry[i] = True; in_pos = True; last_bar = i
    return entry, exit_


def _gen_ssl_channel(df, cooldown, gate, period=13):
    n = len(df)
    brick_up = df["brick_up"].values
    high  = df["High"].values
    low   = df["Low"].values
    close = df["Close"].values
    sma_high = pd.Series(high).rolling(period, min_periods=1).mean().values
    sma_low  = pd.Series(low).rolling(period, min_periods=1).mean().values
    hlv = np.zeros(n)
    for i in range(1, n):
        if close[i] > sma_high[i]:   hlv[i] = 1
        elif close[i] < sma_low[i]:  hlv[i] = -1
        else:                         hlv[i] = hlv[i-1]
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = period + 5
    for i in range(warmup, n):
        if in_pos:
            if not brick_up[i]: exit_[i] = True; in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown: continue
        if hlv[i] == 1 and hlv[i-1] <= 0 and brick_up[i]:
            entry[i] = True; in_pos = True; last_bar = i
    return entry, exit_


def _gen_hhll(df, cooldown, gate, bb_len=29):
    n = len(df)
    brick_up = df["brick_up"].values
    high  = df["High"].values
    low   = df["Low"].values
    close = df["Close"].values
    hlc3 = (high + low + close) / 3.0
    s = pd.Series(hlc3)
    mid = s.rolling(bb_len, min_periods=1).mean().values
    std = s.rolling(bb_len, min_periods=1).std(ddof=0).values
    xHH = mid + 2.0 * std
    xLL = mid - 2.0 * std
    move = (xHH - xLL) / 2.0
    xLLM = xLL - move
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = bb_len + 5
    for i in range(warmup, n):
        if in_pos:
            if not brick_up[i]: exit_[i] = True; in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown: continue
        if low[i] < xLLM[i-1] and brick_up[i]:
            entry[i] = True; in_pos = True; last_bar = i
    return entry, exit_


def _gen_macd_reload(df, cooldown, gate):
    n = len(df)
    brick_up = df["brick_up"].values
    hist = df["macd_hist"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = 40
    for i in range(warmup, n):
        if in_pos:
            if not brick_up[i]: exit_[i] = True; in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown: continue
        if np.isnan(hist[i]) or np.isnan(hist[i-1]): continue
        if hist[i] > 0 and hist[i-1] <= 0 and brick_up[i]:
            entry[i] = True; in_pos = True; last_bar = i
    return entry, exit_


def _gen_wilder_vol(df, cooldown, gate, atr_len=9, arc_factor=1.8):
    n = len(df)
    brick_up = df["brick_up"].values
    close = df["Close"].values
    high  = df["High"].values
    low   = df["Low"].values
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    atr = pd.Series(tr).rolling(atr_len, min_periods=1).mean().values
    lowest_close = pd.Series(close).rolling(atr_len, min_periods=1).min().values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = atr_len + 5
    for i in range(warmup, n):
        if in_pos:
            if not brick_up[i]: exit_[i] = True; in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown: continue
        arc      = atr[i] * arc_factor
        sar_hi   = lowest_close[i-1] + arc
        arc_prev = atr[i-1] * arc_factor if not np.isnan(atr[i-1]) else arc
        sar_hi_p = lowest_close[max(0, i-2)] + arc_prev
        if close[i] > sar_hi and close[i-1] <= sar_hi_p and brick_up[i]:
            entry[i] = True; in_pos = True; last_bar = i
    return entry, exit_


def _gen_halftrend(df, cooldown, gate, amplitude=2):
    n = len(df)
    brick_up = df["brick_up"].values
    high  = df["High"].values
    low   = df["Low"].values
    close = df["Close"].values
    atr_val = df["atr"].values
    trend  = np.zeros(n, dtype=int)
    ht_line = np.zeros(n)
    for i in range(amplitude + 1, n):
        atr_i    = atr_val[i] if not np.isnan(atr_val[i]) else 0.0
        half_atr = atr_i / 2.0
        prev_trend = trend[i-1]
        if prev_trend == 0:
            max_low = np.max(low[max(0, i-amplitude):i+1])
            ht_line[i] = max(ht_line[i-1], max_low)
            if close[i] < ht_line[i] - half_atr:
                trend[i] = 1; ht_line[i] = np.min(high[max(0, i-amplitude):i+1])
            else:
                trend[i] = 0
        else:
            min_high = np.min(high[max(0, i-amplitude):i+1])
            ht_line[i] = min(ht_line[i-1], min_high)
            if close[i] > ht_line[i] + half_atr:
                trend[i] = 0; ht_line[i] = np.max(low[max(0, i-amplitude):i+1])
            else:
                trend[i] = 1
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = amplitude + 10
    for i in range(warmup, n):
        if in_pos:
            if not brick_up[i]: exit_[i] = True; in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown: continue
        if trend[i] == 0 and trend[i-1] == 1 and brick_up[i]:
            entry[i] = True; in_pos = True; last_bar = i
    return entry, exit_


def _gen_nrtr(df, cooldown, gate, period=40, mult=2.0):
    n = len(df)
    brick_up = df["brick_up"].values
    close = df["Close"].values
    nrtr_trend = np.zeros(n, dtype=int)
    hp = close[0]; lp = close[0]
    for i in range(1, n):
        hi_n = np.max(close[max(0, i-period+1):i+1])
        lo_n = np.min(close[max(0, i-period+1):i+1])
        channel = (hi_n - lo_n) * mult / 100.0 if hi_n > lo_n else 0.0
        prev = nrtr_trend[i-1]
        if prev >= 0:
            hp = max(hp, close[i])
            if close[i] <= hp - channel: nrtr_trend[i] = -1; lp = close[i]
            else:                         nrtr_trend[i] = 1
        else:
            lp = min(lp, close[i])
            if close[i] >= lp + channel: nrtr_trend[i] = 1; hp = close[i]
            else:                         nrtr_trend[i] = -1
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = period + 5
    for i in range(warmup, n):
        if in_pos:
            if not brick_up[i]: exit_[i] = True; in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown: continue
        if nrtr_trend[i] == 1 and nrtr_trend[i-1] == -1 and brick_up[i]:
            entry[i] = True; in_pos = True; last_bar = i
    return entry, exit_


# ==============================================================================
# SIGNAL GENERATORS — v2 (from wizard_btc_sweep_v2.py)
# ==============================================================================

def _gen_three_commas(df, cooldown, gate, fast=21, slow=50):
    n = len(df)
    close = df["Close"].values
    brick_up = df["brick_up"].values
    ema_fast = _calc_ema(close, fast)
    ema_slow = _calc_ema(close, slow)
    cross = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]:
            cross[i] = True
    return _make_entry_exit_cross(n, brick_up, cooldown, gate, cross, warmup=slow+5)


def _gen_double_tap(df, cooldown, gate, pivot_len=5, tol_pct=0.5):
    n = len(df)
    close = df["Close"].values
    low = df["Low"].values
    brick_up = df["brick_up"].values
    pivot_lows = np.full(n, np.nan)
    for i in range(pivot_len, n - pivot_len):
        window = low[i - pivot_len: i + pivot_len + 1]
        if low[i] == np.min(window):
            pivot_lows[i] = low[i]
    cross = np.zeros(n, dtype=bool)
    for i in range(50, n):
        prev_pivots = [(j, pivot_lows[j]) for j in range(max(0, i-50), i)
                       if not np.isnan(pivot_lows[j])]
        if len(prev_pivots) >= 2:
            p1_val = prev_pivots[-2][1]
            p2_val = prev_pivots[-1][1]
            avg_price = (p1_val + p2_val) / 2.0
            if avg_price > 0:
                diff_pct = abs(p1_val - p2_val) / avg_price * 100.0
                if diff_pct <= tol_pct and not np.isnan(pivot_lows[i-1]):
                    cross[i] = True
    return _make_entry_exit_cross(n, brick_up, cooldown, gate, cross, warmup=pivot_len*2+10)


def _gen_open_close_cross(df, cooldown, gate, length=8):
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
    n = len(df)
    close = df["Close"].values
    brick_up = df["brick_up"].values
    delta = np.diff(close, prepend=close[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(com=rsi_len-1, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(com=rsi_len-1, adjust=False).mean().values
    rs    = np.where(avg_l == 0, 100.0, avg_g / avg_l)
    rsi   = 100.0 - 100.0 / (1.0 + rs)
    rsi_index = _calc_ema(rsi, sf)
    atr_rsi = np.abs(np.diff(rsi_index, prepend=rsi_index[0]))
    wilders_period = rsi_len * 2 - 1
    ma_atr_rsi = pd.Series(atr_rsi).ewm(com=wilders_period-1, adjust=False).mean().values
    delta_fast = pd.Series(ma_atr_rsi).ewm(com=wilders_period-1, adjust=False).mean().values * qqe_factor
    long_band  = np.zeros(n)
    short_band = np.zeros(n)
    trend      = np.ones(n, dtype=int)
    for i in range(1, n):
        new_lb = rsi_index[i] - delta_fast[i]
        new_sb = rsi_index[i] + delta_fast[i]
        long_band[i]  = max(long_band[i-1], new_lb) if (rsi_index[i-1] > long_band[i-1]  and rsi_index[i] > long_band[i-1])  else new_lb
        short_band[i] = min(short_band[i-1], new_sb) if (rsi_index[i-1] < short_band[i-1] and rsi_index[i] < short_band[i-1]) else new_sb
        if rsi_index[i] > short_band[i-1]:   trend[i] = 1
        elif rsi_index[i] < long_band[i-1]:  trend[i] = -1
        else:                                  trend[i] = trend[i-1]
    qqe_c_long = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if rsi_index[i] > (50 + threshold) and rsi_index[i-1] <= (50 + threshold):
            qqe_c_long[i] = True
    return _make_entry_exit_cross(n, brick_up, cooldown, gate, qqe_c_long,
                                   warmup=wilders_period+sf+10)


def _gen_dual_ma(df, cooldown, gate, fast=21, slow=50):
    n = len(df)
    close = df["Close"].values
    brick_up = df["brick_up"].values
    ema_fast = _calc_ema(close, fast)
    ema_slow = _calc_ema(close, slow)
    cross = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1] and close[i] > ema_fast[i]:
            cross[i] = True
    return _make_entry_exit_cross(n, brick_up, cooldown, gate, cross, warmup=slow+5)


def _gen_elder_ray(df, cooldown, gate, ema_len=13, window=5):
    n = len(df)
    close = df["Close"].values
    high  = df["High"].values
    brick_up = df["brick_up"].values
    ema_val  = _calc_ema(close, ema_len)
    roll_high = pd.Series(high).rolling(window, min_periods=1).max().values
    bull_power = roll_high - ema_val
    cross = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if bull_power[i] > 0 and bull_power[i-1] <= 0:
            cross[i] = True
    return _make_entry_exit_cross(n, brick_up, cooldown, gate, cross, warmup=ema_len+window+5)


def _gen_made_atr(df, cooldown, gate, period=9, disp=13, per_ab=0.5, atr_period=15, atr_mult=2.0):
    n = len(df)
    close = df["Close"].values
    high  = df["High"].values
    low   = df["Low"].values
    brick_up = df["brick_up"].values
    ema_val = _calc_ema(close, period)
    top = np.full(n, np.nan)
    for i in range(disp, n):
        top[i] = ema_val[i - disp] * (1.0 + per_ab / 100.0)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    atr    = pd.Series(tr).rolling(atr_period, min_periods=1).mean().values
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
    cross = np.zeros(n, dtype=bool)
    for i in range(disp + period + 5, n):
        if np.isnan(top[i]) or np.isnan(top[i-1]): continue
        if close[i] > top[i] and close[i-1] <= top[i-1] and close[i] > atr_stop[i]:
            cross[i] = True
    return _make_entry_exit_cross(n, brick_up, cooldown, gate, cross,
                                   warmup=disp+period+atr_period+5)


def _gen_divergence(df, cooldown, gate, pivot_len=5, rsi_len=14):
    n = len(df)
    close = df["Close"].values
    low   = df["Low"].values
    brick_up = df["brick_up"].values
    delta = np.diff(close, prepend=close[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(com=rsi_len-1, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(com=rsi_len-1, adjust=False).mean().values
    rs    = np.where(avg_l == 0, 100.0, avg_g / avg_l)
    rsi   = 100.0 - 100.0 / (1.0 + rs)
    price_pivot = np.full(n, np.nan)
    rsi_pivot   = np.full(n, np.nan)
    for i in range(pivot_len, n - pivot_len):
        if low[i] == np.min(low[i-pivot_len:i+pivot_len+1]):
            price_pivot[i] = low[i]
            rsi_pivot[i]   = rsi[i]
    cross = np.zeros(n, dtype=bool)
    for i in range(pivot_len*2+20, n):
        prev_pivots = [(j, price_pivot[j], rsi_pivot[j])
                       for j in range(max(0, i-80), i)
                       if not np.isnan(price_pivot[j])]
        if len(prev_pivots) >= 2:
            j1, p1, r1 = prev_pivots[-2]
            j2, p2, r2 = prev_pivots[-1]
            if p2 < p1 and r2 > r1 and (i - j2) <= pivot_len + 3:
                cross[i] = True
    return _make_entry_exit_cross(n, brick_up, cooldown, gate, cross,
                                   warmup=pivot_len*2+rsi_len+10)


def _gen_inside_bar(df, cooldown, gate):
    n = len(df)
    close = df["Close"].values
    open_ = df["Open"].values
    high  = df["High"].values
    low   = df["Low"].values
    brick_up = df["brick_up"].values
    inside_bar = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if high[i-1] > high[i] and low[i-1] < low[i]:
            inside_bar[i] = True
    bullish_ib = np.zeros(n, dtype=bool)
    for i in range(2, n):
        if inside_bar[i-1] and close[i-2] > open_[i-2]:
            bullish_ib[i] = True
    return _make_entry_exit_cross(n, brick_up, cooldown, gate, bullish_ib, warmup=10)


def _gen_okx_ma(df, cooldown, gate, length=13):
    n = len(df)
    close = df["Close"].values
    low   = df["Low"].values
    brick_up = df["brick_up"].values
    sma = _calc_sma(close, length)
    cross = np.zeros(n, dtype=bool)
    for i in range(2, n):
        do_long_prev2 = low[i-2] < sma[i-3] if i >= 3 else False
        do_long_now   = low[i-1] < sma[i-2]
        if do_long_now and not do_long_prev2:
            cross[i] = True
    return _make_entry_exit_cross(n, brick_up, cooldown, gate, cross, warmup=length+5)


def _gen_hlhb(df, cooldown, gate, ema_fast=5, ema_slow=10, rsi_len=10):
    n = len(df)
    close = df["Close"].values
    brick_up = df["brick_up"].values
    ema5  = _calc_ema(close, ema_fast)
    ema10 = _calc_ema(close, ema_slow)
    delta = np.diff(close, prepend=close[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(com=rsi_len-1, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(com=rsi_len-1, adjust=False).mean().values
    rs    = np.where(avg_l == 0, 100.0, avg_g / avg_l)
    rsi   = 100.0 - 100.0 / (1.0 + rs)
    ema_cross = np.zeros(n, dtype=bool)
    rsi_cross = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if ema5[i] > ema10[i] and ema5[i-1] <= ema10[i-1]: ema_cross[i] = True
        if rsi[i] > 50 and rsi[i-1] <= 50:                  rsi_cross[i] = True
    signal = np.zeros(n, dtype=bool)
    for i in range(3, n):
        ec_recent = any(ema_cross[max(0,i-2):i+1])
        rc_recent = any(rsi_cross[max(0,i-2):i+1])
        if ec_recent and rc_recent:
            signal[i] = True
    return _make_entry_exit_cross(n, brick_up, cooldown, gate, signal, warmup=ema_slow+rsi_len)


def _gen_combo_bull_power(df, cooldown, gate, ema_len=14, sell_level=-15.0):
    n = len(df)
    close = df["Close"].values
    open_ = df["Open"].values
    high  = df["High"].values
    low   = df["Low"].values
    brick_up = df["brick_up"].values
    ema20_pos = _ema20_pos(close, high, low, ema_len)
    bp_val = np.zeros(n)
    for i in range(1, n):
        c, o, h, l, pc = close[i], open_[i], high[i], low[i], close[i-1]
        po = open_[i-1]
        if c < o:
            bp_val[i] = max(h-pc, c-l) if pc < o else max(h-o, c-l)
        elif c > o:
            bp_val[i] = h-l if pc > o else max(o-pc, h-l)
        else:
            if h-c > c-l:   bp_val[i] = max(h-pc, c-l) if pc < o else h-o
            elif h-c < c-l: bp_val[i] = h-l if pc > o else max(o-c, h-l)
            else:            bp_val[i] = max(h-o, c-l) if pc > o else (max(o-c, h-l) if pc < o else h-l)
    bp_sma  = _calc_sma(bp_val, 15)
    sig_pos = np.where((ema20_pos == 1) & (bp_sma > sell_level), 1.0, -1.0)
    return _make_entry_exit(n, brick_up, cooldown, gate, sig_pos, warmup=ema_len+20)


def _gen_ema_ma_cross(df, cooldown, gate, ema_len=9, sma_len=21):
    n = len(df)
    close = df["Close"].values
    brick_up = df["brick_up"].values
    ema_val = _calc_ema(close, ema_len)
    sma_val = _calc_sma(close, sma_len)
    cross = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if ema_val[i] > sma_val[i] and ema_val[i-1] <= sma_val[i-1]:
            cross[i] = True
    return _make_entry_exit_cross(n, brick_up, cooldown, gate, cross, warmup=sma_len+5)


def _gen_combo_cci(df, cooldown, gate, ema_len=14, cci_len=10, fast_ma=10, slow_ma=20):
    n = len(df)
    close = df["Close"].values
    high  = df["High"].values
    low   = df["Low"].values
    brick_up = df["brick_up"].values
    ema20_pos = _ema20_pos(close, high, low, ema_len)
    typical  = (high + low + close) / 3.0
    cci_sma  = _calc_sma(typical, cci_len)
    mad = pd.Series(typical).rolling(cci_len, min_periods=1).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci_val  = np.where(mad == 0, 0.0, (typical - cci_sma) / (0.015 * mad))
    cci_fast = _calc_sma(cci_val, fast_ma)
    cci_slow = _calc_sma(cci_val, slow_ma)
    sig_pos  = np.where((ema20_pos == 1) & (cci_fast > cci_slow), 1.0, -1.0)
    return _make_entry_exit(n, brick_up, cooldown, gate, sig_pos,
                             warmup=ema_len+cci_len+slow_ma+5)


def _gen_dema_rsi(df, cooldown, gate, ma_len=21, rsi_len=4, smooth=4):
    n = len(df)
    close = df["Close"].values
    brick_up = df["brick_up"].values
    ema1 = _calc_ema(close, ma_len)
    ema2 = _calc_ema(ema1, ma_len)
    dema = 2.0 * ema1 - ema2
    delta = np.diff(dema, prepend=dema[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(com=rsi_len-1, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(com=rsi_len-1, adjust=False).mean().values
    rs    = np.where(avg_l == 0, 100.0, avg_g / avg_l)
    mar_si    = 100.0 - 100.0 / (1.0 + rs)
    smoothed  = _calc_ema(mar_si, smooth)
    cross = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if mar_si[i] > smoothed[i] and mar_si[i-1] <= smoothed[i-1]:
            cross[i] = True
    return _make_entry_exit_cross(n, brick_up, cooldown, gate, cross,
                                   warmup=ma_len*2+rsi_len+smooth+5)


def _gen_combo_bbb(df, cooldown, gate, ema_len=14, buy_level=15.0):
    n = len(df)
    close = df["Close"].values
    open_ = df["Open"].values
    high  = df["High"].values
    low   = df["Low"].values
    brick_up = df["brick_up"].values
    ema20_pos = _ema20_pos(close, high, low, ema_len)
    bear = np.zeros(n)
    bull = np.zeros(n)
    for i in range(1, n):
        c, o, h, l, pc, po = close[i], open_[i], high[i], low[i], close[i-1], open_[i-1]
        if c < o:   bear[i] = h-l if pc > o else max(c-o, h-l)
        elif c > o: bear[i] = max(h-pc, c-l) if pc > o else max(h-o, c-l)
        else:
            if h-c > c-l:   bear[i] = max(h-pc, c-l) if pc < o else h-o
            elif h-c < c-l: bear[i] = max(c-l, h-c) if c > o else o-l
            else:            bear[i] = max(h-o, c-l) if pc > o else (max(o-c, h-l) if pc < o else h-l)
        if c < o:   bull[i] = max(h-pc, c-l) if pc < o else max(h-o, c-l)
        elif c > o: bull[i] = h-l if pc > o else max(o-pc, h-l)
        else:
            if h-c > c-l:   bull[i] = max(h-pc, c-l) if pc < o else h-o
            elif h-c < c-l: bull[i] = h-l if pc > o else max(o-c, h-l)
            else:            bull[i] = max(h-o, c-l) if pc > o else (max(o-c, h-l) if pc < o else h-l)
    nBBB    = bull - bear
    sig_pos = np.where((ema20_pos == 1) & (nBBB >= buy_level), 1.0, -1.0)
    return _make_entry_exit(n, brick_up, cooldown, gate, sig_pos, warmup=ema_len+5)


def _gen_combo_ao(df, cooldown, gate, ema_len=14, slow=34, fast=5, wma_len=15):
    n = len(df)
    close = df["Close"].values
    high  = df["High"].values
    low   = df["Low"].values
    brick_up = df["brick_up"].values
    ema20_pos = _ema20_pos(close, high, low, ema_len)
    hl2 = (high + low) / 2.0
    ao  = _calc_sma(hl2, fast) - _calc_sma(hl2, slow)
    ao_sma = _calc_sma(ao, fast)
    ac  = ao - ao_sma
    ac_wma = pd.Series(ac).rolling(wma_len, min_periods=1).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True).values
    ac_state = np.zeros(n)
    for i in range(1, n):
        if ac_wma[i] > 0:    ac_state[i] = 1.0
        elif ac_wma[i] < 0:  ac_state[i] = -1.0
        else:                 ac_state[i] = ac_state[i-1]
    sig_pos = np.where((ema20_pos == 1) & (ac_state == 1), 1.0, -1.0)
    return _make_entry_exit(n, brick_up, cooldown, gate, sig_pos,
                             warmup=ema_len+slow+wma_len+5)


def _gen_combo_atrr(df, cooldown, gate, ema_len=14, atr_period=5, atr_mult=3.5):
    n = len(df)
    close = df["Close"].values
    high  = df["High"].values
    low   = df["Low"].values
    brick_up = df["brick_up"].values
    ema20_pos = _ema20_pos(close, high, low, ema_len)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    atr    = pd.Series(tr).rolling(atr_period, min_periods=1).mean().values
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
    atrr_pos = np.zeros(n)
    for i in range(1, n):
        if close[i-1] < atr_stop[i-1] and close[i] > atr_stop[i-1]:    atrr_pos[i] = 1.0
        elif close[i-1] > atr_stop[i-1] and close[i] < atr_stop[i-1]:  atrr_pos[i] = -1.0
        else:                                                              atrr_pos[i] = atrr_pos[i-1]
    sig_pos = np.where((ema20_pos == 1) & (atrr_pos == 1), 1.0, -1.0)
    return _make_entry_exit(n, brick_up, cooldown, gate, sig_pos,
                             warmup=ema_len+atr_period+10)


def _gen_combo_hlch(df, cooldown, gate, ema_len=14, look_back=20, sma_len=16):
    n = len(df)
    close = df["Close"].values
    high  = df["High"].values
    low   = df["Low"].values
    brick_up = df["brick_up"].values
    ema20_pos = _ema20_pos(close, high, low, ema_len)
    hlch = np.where(close == 0, 0.0, (high - low) / close)
    hlch_abs_sma = _calc_sma(np.abs(hlch), sma_len)
    hlch_pos = np.zeros(n)
    for i in range(look_back, n):
        if hlch_abs_sma[i - look_back] > abs(hlch[i]):    hlch_pos[i] = 1.0
        elif hlch_abs_sma[i - look_back] < abs(hlch[i]):  hlch_pos[i] = -1.0
        else:                                               hlch_pos[i] = hlch_pos[i-1]
    sig_pos = np.where((ema20_pos == 1) & (hlch_pos == 1), 1.0, -1.0)
    return _make_entry_exit(n, brick_up, cooldown, gate, sig_pos,
                             warmup=ema_len+look_back+sma_len+5)


def _gen_combo_bear_power(df, cooldown, gate, ema_len=14, sell_level=10.0, buy_level=1.0):
    n = len(df)
    close = df["Close"].values
    open_ = df["Open"].values
    high  = df["High"].values
    low   = df["Low"].values
    brick_up = df["brick_up"].values
    ema20_pos = _ema20_pos(close, high, low, ema_len)
    bp_val = np.zeros(n)
    for i in range(1, n):
        c, o, h, l, pc, po = close[i], open_[i], high[i], low[i], close[i-1], open_[i-1]
        if c < o:   bp_val[i] = h-l if pc > o else max(c-o, h-l)
        elif c > o: bp_val[i] = max(h-pc, c-l) if pc > o else max(h-o, c-l)
        else:
            if h-c > c-l:   bp_val[i] = max(h-pc, c-l) if pc < o else h-o
            elif h-c < c-l: bp_val[i] = max(c-l, h-c) if c > o else o-l
            else:            bp_val[i] = max(h-o, c-l) if pc > o else (max(o-c, h-l) if pc < o else h-l)
    bp_pos = np.zeros(n)
    for i in range(1, n):
        if bp_val[i] <= buy_level:    bp_pos[i] = 1.0
        elif bp_val[i] > sell_level:  bp_pos[i] = -1.0
        else:                          bp_pos[i] = bp_pos[i-1]
    sig_pos = np.where((ema20_pos == 1) & (bp_pos == 1), 1.0, -1.0)
    return _make_entry_exit(n, brick_up, cooldown, gate, sig_pos, warmup=ema_len+5)


# ==============================================================================
# STRATEGY REGISTRY (all 27)
# ==============================================================================

STRATEGIES = {
    # v1 — 7 strategies
    "ALPHA_TREND": {
        "fn": _gen_alpha_trend,
        "params": {"period": [10, 14, 20], "coeff": [0.5, 1.0, 1.5, 2.0]},
        "desc": "AlphaTrend ATR+RSI adaptive band (KivancOzbilgic 6.7k)",
    },
    "SSL_CHANNEL": {
        "fn": _gen_ssl_channel,
        "params": {"period": [8, 13, 20, 30]},
        "desc": "SSL Channel SMA high/low flip (vdubus 3.9k)",
    },
    "HHLL": {
        "fn": _gen_hhll,
        "params": {"bb_len": [15, 20, 29, 40]},
        "desc": "HHLL BB offset bounce (HPotter 2.5k)",
    },
    "MACD_RELOAD": {
        "fn": _gen_macd_reload,
        "params": {},
        "desc": "MACD ReLoaded histogram zero cross (KivancOzbilgic 7.5k)",
    },
    "WILDER_VOL": {
        "fn": _gen_wilder_vol,
        "params": {"atr_len": [7, 9, 14], "arc_factor": [1.5, 1.8, 2.5, 3.0]},
        "desc": "Wilder Volatility System ATR SAR (LucF 0.9k)",
    },
    "HALFTREND": {
        "fn": _gen_halftrend,
        "params": {"amplitude": [2, 3, 5]},
        "desc": "HalfTrend adaptive trend band (everget 12.1k)",
    },
    "NRTR": {
        "fn": _gen_nrtr,
        "params": {"period": [20, 40, 60], "mult": [1.5, 2.0, 3.0]},
        "desc": "NRTR trailing reverse (everget 9.9k)",
    },
    # v2 — 20 strategies
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
        "desc": "Inside Bar Momentum (BacktestRookies 0.8k)",
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


# ── Task builder ────────────────────────────────────────────────────────────

def _build_tasks():
    tasks = []
    for inst in INSTRUMENTS:
        for strat_name, strat in STRATEGIES.items():
            grid = strat["params"]
            if grid:
                keys   = list(grid.keys())
                combos = list(itertools.product(*[grid[k] for k in keys]))
            else:
                keys   = []
                combos = [()]
            for combo in combos:
                params = dict(zip(keys, combo))
                for cd in COOLDOWNS:
                    for gate in GATE_MODES:
                        tasks.append({
                            "inst": inst,
                            "strategy": strat_name,
                            "params": params,
                            "cooldown": cd,
                            "gate": gate,
                        })
    return tasks


# ── Worker ──────────────────────────────────────────────────────────────────

def _run_one(task):
    inst       = task["inst"]
    strat_name = task["strategy"]
    params     = task["params"]
    cd         = task["cooldown"]
    gate_mode  = task["gate"]

    _load_inst(inst)
    df  = _w[f"df_{inst}"]
    cfg = INSTRUMENTS[inst]

    fn   = STRATEGIES[strat_name]["fn"]
    gate = _compute_gate(df, gate_mode)

    try:
        entry, exit_ = fn(df, cooldown=cd, gate=gate, **params)
    except Exception as e:
        return {**task, "error": str(e)}

    is_r  = _run_bt(inst, entry, exit_, cfg["is_start"],  cfg["is_end"])
    oos_r = _run_bt(inst, entry, exit_, cfg["oos_start"], cfg["oos_end"])
    oos_r["tpd"] = round(oos_r["trades"] / cfg["oos_days"], 2)

    return {**task, "is": is_r, "oos": oos_r}


# ── Report generator ────────────────────────────────────────────────────────

def _generate_report(results, out_path):
    valid = [r for r in results if "error" not in r and r["oos"]["trades"] >= 10]

    lines = []
    lines.append("# Wizard Mega Sweep — Cross-Asset Report")
    lines.append("")
    lines.append(f"**Date:** 2026-04-02  |  **Instruments:** {len(INSTRUMENTS)}  |  "
                 f"**Strategies:** {len(STRATEGIES)}  |  "
                 f"**Total runs:** {len(results)}  |  **Viable (≥10t OOS):** {len(valid)}")
    lines.append("")
    lines.append("**Methodology:** Long-only, first-down-brick exit, gates: none/psar/adx25/psar_adx25, "
                 "cooldowns: 3/5/10/20. IS/OOS split per instrument.")
    lines.append("")

    # ── Overall top 10 ─────────────────────────────────────────────────────
    top_by_pf = sorted(valid, key=lambda r: (r["oos"]["pf"] if r["oos"]["pf"] != float("inf") else 99999), reverse=True)
    top_by_wr = sorted(valid, key=lambda r: (r["oos"]["wr"], r["oos"]["pf"] if r["oos"]["pf"] != float("inf") else 99999), reverse=True)

    lines.append("## Overall Top 10 by OOS Profit Factor")
    lines.append("")
    lines.append("| Rank | Inst | Strategy | Params | Gate | CD | IS PF | IS T | OOS PF | OOS T | OOS WR | t/d |")
    lines.append("|------|------|----------|--------|------|-----|-------|------|--------|-------|--------|-----|")
    for i, r in enumerate(top_by_pf[:10], 1):
        p_str = " ".join(f"{k}={v}" for k, v in r["params"].items()) if r["params"] else "-"
        pf_s  = "inf" if r["oos"]["pf"] == float("inf") else f"{r['oos']['pf']:.1f}"
        ipf_s = "inf" if r["is"]["pf"]  == float("inf") else f"{r['is']['pf']:.1f}"
        lines.append(f"| {i} | {r['inst']} | **{r['strategy']}** | {p_str} | {r['gate']} | {r['cooldown']} | "
                     f"{ipf_s} | {r['is']['trades']} | {pf_s} | {r['oos']['trades']} | "
                     f"{r['oos']['wr']:.1f}% | {r['oos']['tpd']} |")
    lines.append("")

    lines.append("## Overall Top 10 by OOS Win Rate (min 20 trades)")
    lines.append("")
    valid20 = [r for r in valid if r["oos"]["trades"] >= 20]
    top_wr20 = sorted(valid20, key=lambda r: (r["oos"]["wr"], r["oos"]["pf"] if r["oos"]["pf"] != float("inf") else 99999), reverse=True)
    lines.append("| Rank | Inst | Strategy | Gate | CD | OOS PF | OOS T | OOS WR | t/d |")
    lines.append("|------|------|----------|------|-----|--------|-------|--------|-----|")
    for i, r in enumerate(top_wr20[:10], 1):
        pf_s = "inf" if r["oos"]["pf"] == float("inf") else f"{r['oos']['pf']:.1f}"
        p_str = " ".join(f"{k}={v}" for k, v in r["params"].items()) if r["params"] else "-"
        lines.append(f"| {i} | {r['inst']} | **{r['strategy']}** | {r['gate']} | {r['cooldown']} | "
                     f"{pf_s} | {r['oos']['trades']} | {r['oos']['wr']:.1f}% | {r['oos']['tpd']} |")
    lines.append("")

    # ── Best per strategy per instrument ────────────────────────────────────
    lines.append("## Best Config Per Strategy × Instrument")
    lines.append("")
    lines.append("_Best config = highest OOS PF (min 10 trades). `-` = no qualifying result._")
    lines.append("")

    # Header row
    inst_keys = list(INSTRUMENTS.keys())
    header = "| Strategy |" + "".join(f" {k} |" for k in inst_keys)
    sep    = "|---|" + "".join("---|" for _ in inst_keys)
    lines.append(header)
    lines.append(sep)

    strategy_inst_best = {}
    for strat in STRATEGIES:
        row = f"| {strat} |"
        strategy_inst_best[strat] = {}
        for inst in inst_keys:
            sv = [r for r in valid if r["strategy"] == strat and r["inst"] == inst]
            if sv:
                best = max(sv, key=lambda r: r["oos"]["pf"] if r["oos"]["pf"] != float("inf") else 99999)
                pf_s = "∞" if best["oos"]["pf"] == float("inf") else f"{best['oos']['pf']:.1f}"
                row += f" PF={pf_s} WR={best['oos']['wr']:.0f}% T={best['oos']['trades']} |"
                strategy_inst_best[strat][inst] = best
            else:
                row += " - |"
                strategy_inst_best[strat][inst] = None
        lines.append(row)
    lines.append("")

    # ── Multi-pair consistency ───────────────────────────────────────────────
    lines.append("## Strategy Consistency (# instruments with OOS PF≥5 and T≥10)")
    lines.append("")
    lines.append("| Strategy | Instruments Qualifying | Avg OOS PF | Avg OOS WR |")
    lines.append("|----------|------------------------|-----------|-----------|")

    consistency = []
    for strat in STRATEGIES:
        qualifying = []
        for inst in inst_keys:
            b = strategy_inst_best[strat].get(inst)
            if b and b["oos"]["pf"] >= 5.0 and b["oos"]["trades"] >= 10:
                qualifying.append(b)
        if qualifying:
            avg_pf = sum(r["oos"]["pf"] for r in qualifying if r["oos"]["pf"] != float("inf")) / max(1, len([r for r in qualifying if r["oos"]["pf"] != float("inf")]))
            avg_wr = sum(r["oos"]["wr"] for r in qualifying) / len(qualifying)
            consistency.append((strat, qualifying, avg_pf, avg_wr))
    consistency.sort(key=lambda x: (len(x[1]), x[2]), reverse=True)
    for strat, qualifying, avg_pf, avg_wr in consistency:
        inst_list = ", ".join(r["inst"] for r in qualifying)
        lines.append(f"| **{strat}** | {len(qualifying)}: {inst_list} | {avg_pf:.1f} | {avg_wr:.1f}% |")
    lines.append("")

    # ── Live candidates per instrument ──────────────────────────────────────
    lines.append("## Live Candidates Per Instrument")
    lines.append("")
    lines.append("_Criteria: OOS PF≥10, T≥20, WR≥60%, t/d≥0.2_")
    lines.append("")

    for inst in inst_keys:
        iv = [r for r in valid if r["inst"] == inst
              and r["oos"]["pf"] >= 10.0
              and r["oos"]["trades"] >= 20
              and r["oos"]["wr"] >= 60.0
              and r["oos"]["tpd"] >= 0.2]
        iv.sort(key=lambda r: (r["oos"]["wr"], r["oos"]["pf"] if r["oos"]["pf"] != float("inf") else 99999), reverse=True)

        lines.append(f"### {inst}")
        if iv:
            lines.append("| Strategy | Params | Gate | CD | IS PF | OOS PF | OOS T | OOS WR | t/d |")
            lines.append("|----------|--------|------|-----|-------|--------|-------|--------|-----|")
            for r in iv[:5]:
                p_str = " ".join(f"{k}={v}" for k, v in r["params"].items()) if r["params"] else "-"
                pf_s  = "∞" if r["oos"]["pf"] == float("inf") else f"{r['oos']['pf']:.1f}"
                ipf_s = "∞" if r["is"]["pf"]  == float("inf") else f"{r['is']['pf']:.1f}"
                lines.append(f"| **{r['strategy']}** | {p_str} | {r['gate']} | {r['cooldown']} | "
                              f"{ipf_s} | {pf_s} | {r['oos']['trades']} | {r['oos']['wr']:.1f}% | {r['oos']['tpd']} |")
        else:
            lines.append("_No candidates meeting all criteria._")
        lines.append("")

    # ── Best per instrument (overall) ────────────────────────────────────────
    lines.append("## Best Single Config Per Instrument (Top 3 by WR, min 20 trades)")
    lines.append("")
    for inst in inst_keys:
        iv = [r for r in valid if r["inst"] == inst and r["oos"]["trades"] >= 20]
        iv.sort(key=lambda r: (r["oos"]["wr"], r["oos"]["pf"] if r["oos"]["pf"] != float("inf") else 99999), reverse=True)
        lines.append(f"**{inst}**")
        if iv:
            for r in iv[:3]:
                p_str = " ".join(f"{k}={v}" for k, v in r["params"].items()) if r["params"] else "-"
                pf_s  = "∞" if r["oos"]["pf"] == float("inf") else f"{r['oos']['pf']:.2f}"
                lines.append(f"- {r['strategy']} | {p_str} | gate={r['gate']} cd={r['cooldown']} | "
                              f"OOS PF={pf_s} T={r['oos']['trades']} WR={r['oos']['wr']:.1f}% t/d={r['oos']['tpd']}")
        else:
            lines.append("- No results with ≥20 trades")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    import warnings
    warnings.filterwarnings("ignore")

    tasks = _build_tasks()
    total_param_combos = sum(
        max(1, len(list(itertools.product(*strat["params"].values()))) if strat["params"] else 1)
        for strat in STRATEGIES.values()
    )
    print(f"Wizard Mega Sweep")
    print(f"  Strategies: {len(STRATEGIES)} ({total_param_combos} param combos)")
    print(f"  Instruments: {len(INSTRUMENTS)}")
    print(f"  Gates: {len(GATE_MODES)}  |  Cooldowns: {len(COOLDOWNS)}")
    print(f"  Total combos: {len(tasks)}")
    print(f"  Workers: {MAX_WORKERS}")
    print()

    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(_run_one, t): t for t in tasks}
        done = 0
        errors = 0
        for f in as_completed(futs):
            done += 1
            r = f.result()
            if "error" in r:
                errors += 1
            results.append(r)
            if done % 2000 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} done ({errors} errors)...", flush=True)

    valid = [r for r in results if "error" not in r and r["oos"]["trades"] >= 10]
    valid_sorted = sorted(valid,
                          key=lambda r: r["oos"]["pf"] if r["oos"]["pf"] != float("inf") else 99999,
                          reverse=True)

    print(f"\nTotal: {len(results)} runs | {len(valid)} viable (≥10t OOS) | {errors} errors")

    # Console summary: top 20 overall
    print(f"\n{'='*110}")
    print(f"TOP 20 BY OOS PF (min 10 trades)")
    print(f"{'='*110}")
    print(f"  {'Inst':<12} {'Strategy':<18} {'Params':<30} {'Gate':<12} {'CD':>3}  "
          f"{'IS_PF':>7} {'OOS_PF':>7} {'OOS_T':>5} {'OOS_WR':>7} {'t/d':>5}")
    print("  " + "-" * 106)
    for r in valid_sorted[:20]:
        p_str = " ".join(f"{k}={v}" for k, v in r["params"].items()) if r["params"] else "-"
        pf_s  = "inf" if r["oos"]["pf"] == float("inf") else f"{r['oos']['pf']:.2f}"
        ipf_s = "inf" if r["is"]["pf"]  == float("inf") else f"{r['is']['pf']:.2f}"
        print(f"  {r['inst']:<12} {r['strategy']:<18} {p_str:<30} {r['gate']:<12} {r['cooldown']:>3}  "
              f"{ipf_s:>7} {pf_s:>7} {r['oos']['trades']:>5} {r['oos']['wr']:>6.1f}% {r['oos']['tpd']:>5.1f}")

    # Console: best per instrument
    print(f"\n{'='*80}")
    print("BEST PER INSTRUMENT (by OOS PF, min 10 trades)")
    print(f"{'='*80}")
    for inst in INSTRUMENTS:
        iv = [r for r in valid if r["inst"] == inst]
        if iv:
            best = max(iv, key=lambda r: r["oos"]["pf"] if r["oos"]["pf"] != float("inf") else 99999)
            p_str = " ".join(f"{k}={v}" for k, v in best["params"].items()) if best["params"] else "-"
            pf_s  = "inf" if best["oos"]["pf"] == float("inf") else f"{best['oos']['pf']:.2f}"
            print(f"  {inst:<12} {best['strategy']:<18} {p_str:<28} gate={best['gate']:<12} cd={best['cooldown']:<3} "
                  f"PF={pf_s} T={best['oos']['trades']} WR={best['oos']['wr']:.1f}% t/d={best['oos']['tpd']:.1f}")
        else:
            print(f"  {inst:<12} -- no results --")

    # Save full JSON
    json_out = ROOT / "ai_context" / "wizard_mega_sweep_results.json"
    with open(json_out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {len(results)} results -> {json_out}")

    # Generate markdown report
    report_out = ROOT / "ai_context" / "wizard_mega_sweep_report.md"
    _generate_report(results, report_out)
    print(f"Report generated -> {report_out}")


if __name__ == "__main__":
    main()
