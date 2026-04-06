#!/usr/bin/env python3
"""
btc_hf_sweep.py -- BTC High-Frequency Strategy Sweep (Long Only)

Target: 1+ trade/day on OANDA BTCUSD $150 Renko bricks.
All strategies are LONG ONLY with first-down-brick exit.

Strategy concepts:
    BAND_BOUNCE    BB %B oversold + up brick (mean reversion)
    RSI_OVERSOLD   RSI drops below threshold then up brick (MR)
    STOCH_CROSS    Stoch K crosses up from oversold zone (MR)
    MACD_FLIP      MACD histogram crosses from neg to pos (momentum)
    ST_FLIP        Supertrend flips bullish (trend)
    EMA_CROSS      EMA9 crosses above EMA21 (trend)
    KAMA_TURN      KAMA slope turns positive (adaptive)
    SQUEEZE_FIRE   Squeeze releases with positive momentum (breakout)
    R001_MOM       N consecutive up bricks (momentum)
    R002_REV       N down bricks then first up brick (reversal)
    DI_CROSS       +DI crosses above -DI (directional)
    RSI_REGIME     RSI crosses above 50 (regime change)
    BB_BREAK       Close breaks above BB upper (breakout)

Gate levels:
    none        No gates
    psar        PSAR bullish only
    adx20       ADX >= 20 only
    psar_adx20  PSAR bullish + ADX >= 20

Cooldowns: [3, 5, 8, 12]
Exit: first down brick

Note: BTC Renko data has NO volume column (Volume=0).
Volume-based indicators (CMF, MFI, OBV, vol_ratio) are unavailable.

Usage:
    python renko/btc_hf_sweep.py
"""

import argparse
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
HTF_FILE   = "OANDA_BTCUSD.SPOT.US, 1S renko 300.csv"
IS_START   = "2024-06-04"
IS_END     = "2025-09-30"
OOS_START  = "2025-10-01"
OOS_END    = "2026-03-19"
COMMISSION = 0.0046
CAPITAL    = 1000.0
QTY_VALUE  = 20   # $20 notional per trade (cash mode)

# -- Cooldown / gate grid -------------------------------------------------------

COOLDOWNS  = [3, 5, 8, 12]
GATE_MODES = ["none", "psar", "adx20", "psar_adx20"]


# -- Data loading ---------------------------------------------------------------

def _load_ltf_data():
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    df = load_renko_export(LTF_FILE)
    add_renko_indicators(df)
    return df


def _load_htf_data():
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    df = load_renko_export(HTF_FILE)
    add_renko_indicators(df)
    return df


# -- Gate computation -----------------------------------------------------------

def _compute_gate(df, gate_mode):
    """Compute long gate array for given mode."""
    n = len(df)
    gate = np.ones(n, dtype=bool)

    if "psar" in gate_mode:
        psar = df["psar_dir"].values
        gate &= (np.isnan(psar) | (psar > 0))  # +1 = bullish

    if "adx20" in gate_mode:
        adx = df["adx"].values
        gate &= (np.isnan(adx) | (adx >= 20))

    return gate


# -- Backtest runner ------------------------------------------------------------

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


# -- Signal generators ---------------------------------------------------------
# All long-only. Exit = first down brick. Gate applied externally.

def _gen_band_bounce(df, cooldown, gate, band_thresh=0.20):
    """BB %B drops below threshold then up brick = long entry."""
    n = len(df)
    brick_up = df["brick_up"].values
    pct_b = df["bb_pct_b"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = 30

    for i in range(warmup, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        if np.isnan(pct_b[i]):
            continue
        if up and pct_b[i] <= band_thresh:
            entry[i] = True
            in_pos = True
            last_bar = i
    return entry, exit_


def _gen_rsi_oversold(df, cooldown, gate, rsi_thresh=35):
    """RSI below threshold + up brick = oversold bounce long."""
    n = len(df)
    brick_up = df["brick_up"].values
    rsi = df["rsi"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = 20

    for i in range(warmup, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        if np.isnan(rsi[i]):
            continue
        if up and rsi[i] < rsi_thresh:
            entry[i] = True
            in_pos = True
            last_bar = i
    return entry, exit_


def _gen_stoch_cross(df, cooldown, gate, stoch_thresh=25):
    """Stoch K crosses up from below threshold on up brick."""
    n = len(df)
    brick_up = df["brick_up"].values
    stoch_k = df["stoch_k"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = 20

    for i in range(warmup, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        if np.isnan(stoch_k[i]) or np.isnan(stoch_k[i-1]):
            continue
        if up and stoch_k[i] > stoch_thresh and stoch_k[i-1] <= stoch_thresh:
            entry[i] = True
            in_pos = True
            last_bar = i
    return entry, exit_


def _gen_macd_flip(df, cooldown, gate):
    """MACD histogram crosses from negative to positive on up brick."""
    n = len(df)
    brick_up = df["brick_up"].values
    macd_h = df["macd_hist"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = 35

    for i in range(warmup, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        if np.isnan(macd_h[i]) or np.isnan(macd_h[i-1]):
            continue
        if up and macd_h[i] > 0 and macd_h[i-1] <= 0:
            entry[i] = True
            in_pos = True
            last_bar = i
    return entry, exit_


def _gen_st_flip(df, cooldown, gate):
    """Supertrend flips to bullish (+1) on up brick."""
    n = len(df)
    brick_up = df["brick_up"].values
    st_dir = df["st_dir"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = 15

    for i in range(warmup, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        if np.isnan(st_dir[i]) or np.isnan(st_dir[i-1]):
            continue
        if up and st_dir[i] > 0 and st_dir[i-1] <= 0:
            entry[i] = True
            in_pos = True
            last_bar = i
    return entry, exit_


def _gen_ema_cross(df, cooldown, gate):
    """EMA9 crosses above EMA21 on up brick."""
    n = len(df)
    brick_up = df["brick_up"].values
    ema9 = df["ema9"].values
    ema21 = df["ema21"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = 30

    for i in range(warmup, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        if np.isnan(ema9[i]) or np.isnan(ema21[i]) or np.isnan(ema9[i-1]) or np.isnan(ema21[i-1]):
            continue
        if up and ema9[i] > ema21[i] and ema9[i-1] <= ema21[i-1]:
            entry[i] = True
            in_pos = True
            last_bar = i
    return entry, exit_


def _gen_kama_turn(df, cooldown, gate):
    """KAMA slope turns from negative to positive on up brick."""
    n = len(df)
    brick_up = df["brick_up"].values
    kama_s = df["kama_slope"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = 15

    for i in range(warmup, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        if np.isnan(kama_s[i]) or np.isnan(kama_s[i-1]):
            continue
        if up and kama_s[i] > 0 and kama_s[i-1] <= 0:
            entry[i] = True
            in_pos = True
            last_bar = i
    return entry, exit_


def _gen_squeeze_fire(df, cooldown, gate):
    """Squeeze releases (was on, now off) with positive momentum on up brick."""
    n = len(df)
    brick_up = df["brick_up"].values
    sq_on = df["sq_on"].values
    sq_mom = df["sq_momentum"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = 30

    for i in range(warmup, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        if np.isnan(sq_on[i]) or np.isnan(sq_on[i-1]) or np.isnan(sq_mom[i]):
            continue
        if up and (not sq_on[i]) and sq_on[i-1] and sq_mom[i] > 0:
            entry[i] = True
            in_pos = True
            last_bar = i
    return entry, exit_


def _gen_r001(df, cooldown, gate, n_bricks=3):
    """N consecutive up bricks including current = momentum long."""
    n = len(df)
    brick_up = df["brick_up"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = max(n_bricks + 5, 20)

    for i in range(warmup, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        if not up:
            continue
        # Check N consecutive up bricks (including current)
        all_up = True
        for j in range(n_bricks):
            if not brick_up[i - j]:
                all_up = False
                break
        if all_up:
            entry[i] = True
            in_pos = True
            last_bar = i
    return entry, exit_


def _gen_r002(df, cooldown, gate, n_bricks=3):
    """N down bricks followed by first up brick = reversal long."""
    n = len(df)
    brick_up = df["brick_up"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = max(n_bricks + 5, 20)

    for i in range(warmup, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        if not up:
            continue
        # Check N previous bricks all down
        all_down = True
        for j in range(1, n_bricks + 1):
            if brick_up[i - j]:
                all_down = False
                break
        if all_down:
            entry[i] = True
            in_pos = True
            last_bar = i
    return entry, exit_


def _gen_di_cross(df, cooldown, gate):
    """+DI crosses above -DI on up brick."""
    n = len(df)
    brick_up = df["brick_up"].values
    plus_di = df["plus_di"].values
    minus_di = df["minus_di"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = 20

    for i in range(warmup, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        if np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or np.isnan(plus_di[i-1]) or np.isnan(minus_di[i-1]):
            continue
        if up and plus_di[i] > minus_di[i] and plus_di[i-1] <= minus_di[i-1]:
            entry[i] = True
            in_pos = True
            last_bar = i
    return entry, exit_


def _gen_rsi_regime(df, cooldown, gate):
    """RSI crosses above 50 from below on up brick."""
    n = len(df)
    brick_up = df["brick_up"].values
    rsi = df["rsi"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = 20

    for i in range(warmup, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        if np.isnan(rsi[i]) or np.isnan(rsi[i-1]):
            continue
        if up and rsi[i] > 50 and rsi[i-1] <= 50:
            entry[i] = True
            in_pos = True
            last_bar = i
    return entry, exit_


def _gen_bb_break(df, cooldown, gate):
    """Close breaks above BB upper band on up brick (breakout)."""
    n = len(df)
    brick_up = df["brick_up"].values
    close = df["Close"].values.astype(float)
    bb_upper = df["bb_upper"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = 25

    for i in range(warmup, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        if np.isnan(bb_upper[i]):
            continue
        if up and close[i] > bb_upper[i]:
            entry[i] = True
            in_pos = True
            last_bar = i
    return entry, exit_


# -- Combo builder --------------------------------------------------------------

STRATEGIES = {
    # name: (generator_fn, {extra_param: [values]} or None)
    "band_bounce":   (_gen_band_bounce,   {"band_thresh": [0.15, 0.20, 0.25, 0.30]}),
    "rsi_oversold":  (_gen_rsi_oversold,  {"rsi_thresh": [30, 35, 40, 45]}),
    "stoch_cross":   (_gen_stoch_cross,   {"stoch_thresh": [20, 25, 30, 35]}),
    "macd_flip":     (_gen_macd_flip,     None),
    "st_flip":       (_gen_st_flip,       None),
    "ema_cross":     (_gen_ema_cross,     None),
    "kama_turn":     (_gen_kama_turn,     None),
    "squeeze_fire":  (_gen_squeeze_fire,  None),
    "r001_mom":      (_gen_r001,          {"n_bricks": [2, 3, 4, 5]}),
    "r002_rev":      (_gen_r002,          {"n_bricks": [2, 3, 4, 5]}),
    "di_cross":      (_gen_di_cross,      None),
    "rsi_regime":    (_gen_rsi_regime,    None),
    "bb_break":      (_gen_bb_break,      None),
}


def _build_combos():
    """Build all (strategy, cooldown, gate, extra_params) combos."""
    combos = []
    for sname, (gen_fn, extra_grid) in STRATEGIES.items():
        if extra_grid:
            extra_key = list(extra_grid.keys())[0]
            extra_vals = extra_grid[extra_key]
        else:
            extra_key = None
            extra_vals = [None]

        for cd in COOLDOWNS:
            for gmode in GATE_MODES:
                for ev in extra_vals:
                    combo = {
                        "strategy": sname,
                        "cooldown": cd,
                        "gate_mode": gmode,
                    }
                    if extra_key:
                        combo[extra_key] = ev
                    combos.append(combo)
    return combos


# -- Worker (parallel execution) ------------------------------------------------

_w = {}  # per-process cache


def _run_one(combo):
    """Run IS + OOS backtest for one combo."""
    if "df" not in _w:
        _w["df"] = _load_ltf_data()
        # Precompute gate arrays
        _w["gates"] = {gm: _compute_gate(_w["df"], gm) for gm in GATE_MODES}

    df = _w["df"]
    sname = combo["strategy"]
    gen_fn = STRATEGIES[sname][0]
    gate = _w["gates"][combo["gate_mode"]]
    cd = combo["cooldown"]

    # Build kwargs for generator
    kwargs = {"cooldown": cd, "gate": gate}
    for k in ("band_thresh", "rsi_thresh", "stoch_thresh", "n_bricks"):
        if k in combo:
            kwargs[k] = combo[k]

    entry, exit_ = gen_fn(df, **kwargs)

    is_r = _run_bt(df, entry, exit_, IS_START, IS_END)
    oos_r = _run_bt(df, entry, exit_, OOS_START, OOS_END)

    return combo, is_r, oos_r


# -- Summary -------------------------------------------------------------------

def _summarize(all_results):
    # OOS period length in days
    oos_days = 170  # 2025-10-01 to 2026-03-19

    print(f"\n{'='*110}")
    print("  BTC High-Frequency Sweep — Long Only, 13 Strategy Concepts")
    print(f"  Target: 1+ trade/day ({oos_days}+ OOS trades)")
    print(f"{'='*110}")

    # -- Per strategy best (by net, require T >= oos_days) --
    print(f"\n  {'Strategy':<15} {'Gate':<12} {'cd':>3} {'Extra':>10} | "
          f"{'IS PF':>7} {'T':>5} {'WR%':>6} | "
          f"{'OOS PF':>8} {'T':>5} {'t/d':>5} {'WR%':>6} {'Net':>9} {'DD%':>7}")
    print(f"  {'-'*108}")

    for sname in STRATEGIES:
        subset = [r for r in all_results
                  if r["strategy"] == sname
                  and r["oos_trades"] >= 20
                  and r["oos_net"] > 0]
        if not subset:
            print(f"  {sname:<15} (no viable results)")
            continue

        best = max(subset, key=lambda r: (
            r["oos_net"],
            r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6,
        ))
        _print_row(best, oos_days)

    # -- HF viable: T >= oos_days and net > 0, sorted by net --
    hf = [r for r in all_results
          if r["oos_trades"] >= oos_days and r["oos_net"] > 0]
    hf.sort(key=lambda r: r["oos_net"], reverse=True)

    print(f"\n{'='*110}")
    print(f"  HIGH FREQ (T>={oos_days}, net>0): {len(hf)} configs — sorted by net")
    print(f"{'='*110}")
    print(f"  {'#':>3} {'Strategy':<15} {'Gate':<12} {'cd':>3} {'Extra':>10} | "
          f"{'IS PF':>7} {'T':>5} {'WR%':>6} | "
          f"{'OOS PF':>8} {'T':>5} {'t/d':>5} {'WR%':>6} {'Net':>9} {'DD%':>7}")
    print(f"  {'-'*112}")
    for i, r in enumerate(hf[:30]):
        _print_row(r, oos_days, rank=i+1)

    # -- Best WR (T >= 100) --
    hw = [r for r in all_results
          if r["oos_trades"] >= 100 and r["oos_net"] > 0]
    hw.sort(key=lambda r: (r["oos_wr"], r["oos_net"]), reverse=True)

    print(f"\n{'='*110}")
    print(f"  BEST WIN RATE (T>=100, net>0): {len(hw)} configs — sorted by WR")
    print(f"{'='*110}")
    print(f"  {'#':>3} {'Strategy':<15} {'Gate':<12} {'cd':>3} {'Extra':>10} | "
          f"{'IS PF':>7} {'T':>5} {'WR%':>6} | "
          f"{'OOS PF':>8} {'T':>5} {'t/d':>5} {'WR%':>6} {'Net':>9} {'DD%':>7}")
    print(f"  {'-'*112}")
    for i, r in enumerate(hw[:30]):
        _print_row(r, oos_days, rank=i+1)

    # -- Best PF (T >= 80) --
    hp = [r for r in all_results
          if r["oos_trades"] >= 80 and r["oos_net"] > 0]
    hp.sort(key=lambda r: (
        r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6,
        r["oos_net"],
    ), reverse=True)

    print(f"\n{'='*110}")
    print(f"  BEST PF (T>=80, net>0): {len(hp)} configs — sorted by PF")
    print(f"{'='*110}")
    print(f"  {'#':>3} {'Strategy':<15} {'Gate':<12} {'cd':>3} {'Extra':>10} | "
          f"{'IS PF':>7} {'T':>5} {'WR%':>6} | "
          f"{'OOS PF':>8} {'T':>5} {'t/d':>5} {'WR%':>6} {'Net':>9} {'DD%':>7}")
    print(f"  {'-'*112}")
    for i, r in enumerate(hp[:30]):
        _print_row(r, oos_days, rank=i+1)

    # -- IS/OOS consistency (both profitable, decent trade counts) --
    consistent = [r for r in all_results
                  if r["oos_trades"] >= 100 and r["oos_net"] > 0
                  and r["is_trades"] >= 100 and r["is_net"] > 0]
    consistent.sort(key=lambda r: (r["oos_net"], r["oos_wr"]), reverse=True)

    print(f"\n{'='*110}")
    print(f"  CONSISTENT IS/OOS (both net>0, T>=100): {len(consistent)} configs")
    print(f"{'='*110}")
    print(f"  {'#':>3} {'Strategy':<15} {'Gate':<12} {'cd':>3} {'Extra':>10} | "
          f"{'IS PF':>7} {'T':>5} {'WR%':>6} {'Net':>9} | "
          f"{'OOS PF':>8} {'T':>5} {'t/d':>5} {'WR%':>6} {'Net':>9}")
    print(f"  {'-'*118}")
    for i, r in enumerate(consistent[:30]):
        pf_i = "INF" if math.isinf(r["is_pf"]) else f"{r['is_pf']:.2f}"
        pf_o = "INF" if math.isinf(r["oos_pf"]) else f"{r['oos_pf']:.2f}"
        extra = _extra_str(r)
        tpd = r["oos_trades"] / oos_days
        print(f"  {i+1:>3} {r['strategy']:<15} {r['gate_mode']:<12} {r['cooldown']:>3} {extra:>10} | "
              f"{pf_i:>7} {r['is_trades']:>5} {r['is_wr']:>5.1f}% {r['is_net']:>9.2f} | "
              f"{pf_o:>8} {r['oos_trades']:>5} {tpd:>4.1f} {r['oos_wr']:>5.1f}% {r['oos_net']:>9.2f}")

    # -- Gate effectiveness --
    print(f"\n  --- Gate Effectiveness (avg OOS, trades >= 20) ---")
    for gm in GATE_MODES:
        rows = [r for r in all_results if r["gate_mode"] == gm and r["oos_trades"] >= 20]
        if rows:
            finite_pf = [r["oos_pf"] for r in rows if not math.isinf(r["oos_pf"]) and r["oos_pf"] > 0]
            avg_pf = np.mean(finite_pf) if finite_pf else 0
            avg_t = np.mean([r["oos_trades"] for r in rows])
            avg_wr = np.mean([r["oos_wr"] for r in rows])
            avg_net = np.mean([r["oos_net"] for r in rows])
            print(f"    {gm:<15} avg PF={avg_pf:>7.2f} | avg T={avg_t:>6.1f} | "
                  f"avg WR={avg_wr:>5.1f}% | avg Net={avg_net:>7.2f} | N={len(rows)}")


def _extra_str(r):
    """Format extra param value for display."""
    for k in ("band_thresh", "rsi_thresh", "stoch_thresh", "n_bricks"):
        if k in r:
            return f"{k.split('_')[0]}={r[k]}"
    return ""


def _print_row(r, oos_days, rank=None):
    pf_i = "INF" if math.isinf(r["is_pf"]) else f"{r['is_pf']:.2f}"
    pf_o = "INF" if math.isinf(r["oos_pf"]) else f"{r['oos_pf']:.2f}"
    extra = _extra_str(r)
    tpd = r["oos_trades"] / oos_days
    prefix = f"  {rank:>3}" if rank else "  "
    print(f"{prefix} {r['strategy']:<15} {r['gate_mode']:<12} {r['cooldown']:>3} {extra:>10} | "
          f"{pf_i:>7} {r['is_trades']:>5} {r['is_wr']:>5.1f}% | "
          f"{pf_o:>8} {r['oos_trades']:>5} {tpd:>4.1f} {r['oos_wr']:>5.1f}% {r['oos_net']:>9.2f} {r['oos_dd']:>6.2f}%")


# -- Main -----------------------------------------------------------------------

def main():
    combos = _build_combos()
    total = len(combos)

    print(f"\n{'='*70}")
    print(f"BTC High-Frequency Sweep — Long Only")
    print(f"  Strategies : {len(STRATEGIES)} concepts")
    print(f"  Cooldowns  : {COOLDOWNS}")
    print(f"  Gates      : {GATE_MODES}")
    print(f"  Total runs : {total} ({total*2} backtests)")
    print(f"  Workers    : {MAX_WORKERS}")
    print(f"  IS period  : {IS_START} -> {IS_END}")
    print(f"  OOS period : {OOS_START} -> {OOS_END}")
    print(f"{'='*70}\n")

    results = []
    done = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run_one, c): c for c in combos}
        for fut in as_completed(futures):
            try:
                combo, is_r, oos_r = fut.result()
                row = {
                    "strategy":   combo["strategy"],
                    "cooldown":   combo["cooldown"],
                    "gate_mode":  combo["gate_mode"],
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
                # Add extra params
                for k in ("band_thresh", "rsi_thresh", "stoch_thresh", "n_bricks"):
                    if k in combo:
                        row[k] = combo[k]
                results.append(row)
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()

            done += 1
            if done % 50 == 0 or done == total:
                print(f"  [{done:>4}/{total}]", flush=True)

    # Save results
    out_path = ROOT / "ai_context" / "btc_hf_sweep_results.json"
    out_path.parent.mkdir(exist_ok=True)

    serializable = []
    for r in results:
        sr = dict(r)
        for k in ("is_pf", "oos_pf"):
            if math.isinf(sr[k]):
                sr[k] = "inf"
        serializable.append(sr)

    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved {len(results)} results -> {out_path}")

    _summarize(results)


if __name__ == "__main__":
    main()
