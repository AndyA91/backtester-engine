#!/usr/bin/env python3
"""
btc_phase8_creative_sweep.py -- BTC Creative Strategy Exploration (Long Only)

Tests 10 fundamentally different strategy concepts, each independent of BTC003's
brick-counting approach. All are long-only with first-down-brick exit.

Strategy concepts:
    MEAN_REV       Stoch %K oversold bounce (crosses above 20) + price > EMA50
    EMA_CROSS      EMA9 crosses above EMA21 (classic MA crossover)
    MACD_FLIP      MACD histogram turns positive (momentum shift)
    SQUEEZE_FIRE   Squeeze releases with positive momentum (volatility breakout)
    KAMA_TURN      KAMA slope turns positive (adaptive momentum)
    DI_CROSS       +DI crosses above -DI (directional strength)
    RSI_REGIME     RSI crosses above 50 (momentum regime change)
    BB_BOUNCE      Price bounces off lower BB (mean reversion from support)
    TRIPLE_CONFIRM 3-of-5 indicators agree bullish (confluence)
    ESCGO_CROSS    ESCGO fast crosses above slow (Ehlers oscillator)

Each strategy tested:
    - With no gates vs full gates (PSAR + ADX>=30 + vol<=1.5)
    - With HTF ADX [0, 35]
    - With cooldown variations [15, 25, 40]

Uses ProcessPoolExecutor -- one worker per strategy.

Usage:
    python renko/btc_phase8_creative_sweep.py
    python renko/btc_phase8_creative_sweep.py --no-parallel
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

# -- Instrument config ---------------------------------------------------------

LTF_FILE   = "OANDA_BTCUSD.SPOT.US, 1S renko 150.csv"
HTF_FILE   = "OANDA_BTCUSD.SPOT.US, 1S renko 300.csv"
IS_START   = "2024-06-04"
IS_END     = "2025-09-30"
OOS_START  = "2025-10-01"
OOS_END    = "2026-03-19"
COMMISSION = 0.0046
CAPITAL    = 1000.0
QTY_VALUE  = 20
VOL_MAX    = 1.5
ADX_THRESH = 30

# -- Cooldown variations -------------------------------------------------------
COOLDOWNS = [15, 25, 40]

# -- Strategy definitions -------------------------------------------------------

STRATEGIES = [
    "mean_rev", "ema_cross", "macd_flip", "squeeze_fire", "kama_turn",
    "di_cross", "rsi_regime", "bb_bounce", "triple_confirm", "escgo_cross",
]


# -- Data loading --------------------------------------------------------------

def _load_ltf_data():
    sys.path.insert(0, str(ROOT))
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    from renko.phase6_enrichment import add_phase6_indicators
    df = load_renko_export(LTF_FILE)
    add_renko_indicators(df)
    add_phase6_indicators(df, include_mk=False)
    return df


def _load_htf_data():
    sys.path.insert(0, str(ROOT))
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    df = load_renko_export(HTF_FILE)
    add_renko_indicators(df)
    return df


# -- Gate computation ----------------------------------------------------------

def _compute_gates(df_ltf, df_htf, use_gates, htf_adx_thresh):
    """
    Compute combined long gate array.

    use_gates=True: PSAR bullish + ADX>=30 + vol<=1.5
    htf_adx_thresh>0: HTF ADX gate via merge_asof
    """
    n = len(df_ltf)
    gate = np.ones(n, dtype=bool)

    if use_gates:
        sys.path.insert(0, str(ROOT))
        from renko.phase6_sweep import _compute_gate_arrays

        # PSAR direction
        p6_long, _ = _compute_gate_arrays(df_ltf, "psar_dir")
        gate &= p6_long

        # LTF ADX
        adx = df_ltf["adx"].values
        gate &= (np.isnan(adx) | (adx >= ADX_THRESH))

        # Vol ratio
        vr = df_ltf["vol_ratio"].values
        gate &= (np.isnan(vr) | (vr <= VOL_MAX))

    # HTF ADX
    if htf_adx_thresh > 0:
        htf_adx = df_htf["adx"].values
        htf_gate = np.isnan(htf_adx) | (htf_adx >= htf_adx_thresh)

        htf_frame = pd.DataFrame({
            "t": df_htf.index.values,
            "g": htf_gate.astype(float),
        }).sort_values("t")
        ltf_frame = pd.DataFrame({"t": df_ltf.index.values}).sort_values("t")
        merged = pd.merge_asof(ltf_frame, htf_frame, on="t", direction="backward")
        g = merged["g"].values
        htf_aligned = np.where(np.isnan(g), True, g > 0.5).astype(bool)
        gate &= htf_aligned

    return gate


# -- Signal generators for each strategy concept --------------------------------

def _gen_mean_rev(df, cooldown, gate):
    """Stoch %K oversold bounce: crosses above 20 from below, price > EMA50."""
    n = len(df)
    brick_up = df["brick_up"].values
    stoch_k = df["stoch_k"].values
    ema50 = df["ema50"].values
    close = df["Close"].values.astype(float)

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = 60

    for i in range(warmup, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i]:
            continue
        if (i - last_bar) < cooldown:
            continue
        if np.isnan(stoch_k[i]) or np.isnan(stoch_k[i-1]) or np.isnan(ema50[i]):
            continue

        if up and stoch_k[i] > 20 and stoch_k[i-1] <= 20 and close[i] > ema50[i]:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_ema_cross(df, cooldown, gate):
    """EMA9 crosses above EMA21 on an up brick."""
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
        if not gate[i]:
            continue
        if (i - last_bar) < cooldown:
            continue
        if np.isnan(ema9[i]) or np.isnan(ema21[i]) or np.isnan(ema9[i-1]) or np.isnan(ema21[i-1]):
            continue

        if up and ema9[i] > ema21[i] and ema9[i-1] <= ema21[i-1]:
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
        if not gate[i]:
            continue
        if (i - last_bar) < cooldown:
            continue
        if np.isnan(macd_h[i]) or np.isnan(macd_h[i-1]):
            continue

        if up and macd_h[i] > 0 and macd_h[i-1] <= 0:
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
        if not gate[i]:
            continue
        if (i - last_bar) < cooldown:
            continue
        if np.isnan(sq_on[i]) or np.isnan(sq_on[i-1]) or np.isnan(sq_mom[i]):
            continue

        if up and (not sq_on[i]) and sq_on[i-1] and sq_mom[i] > 0:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_kama_turn(df, cooldown, gate):
    """KAMA slope turns positive (was negative) on up brick."""
    n = len(df)
    brick_up = df["brick_up"].values
    kama_s = df["kama_slope"].values

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
        if not gate[i]:
            continue
        if (i - last_bar) < cooldown:
            continue
        if np.isnan(kama_s[i]) or np.isnan(kama_s[i-1]):
            continue

        if up and kama_s[i] > 0 and kama_s[i-1] <= 0:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_di_cross(df, cooldown, gate):
    """+DI crosses above -DI on up brick (directional strength shift)."""
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
        if not gate[i]:
            continue
        if (i - last_bar) < cooldown:
            continue
        if np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or np.isnan(plus_di[i-1]) or np.isnan(minus_di[i-1]):
            continue

        if up and plus_di[i] > minus_di[i] and plus_di[i-1] <= minus_di[i-1]:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_rsi_regime(df, cooldown, gate):
    """RSI crosses above 50 from below on up brick (momentum regime change)."""
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
        if not gate[i]:
            continue
        if (i - last_bar) < cooldown:
            continue
        if np.isnan(rsi[i]) or np.isnan(rsi[i-1]):
            continue

        if up and rsi[i] > 50 and rsi[i-1] <= 50:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_bb_bounce(df, cooldown, gate):
    """Price bounces off lower BB: bb_pct_b crosses above 0.2 from below on up brick."""
    n = len(df)
    brick_up = df["brick_up"].values
    pct_b = df["bb_pct_b"].values

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
        if not gate[i]:
            continue
        if (i - last_bar) < cooldown:
            continue
        if np.isnan(pct_b[i]) or np.isnan(pct_b[i-1]):
            continue

        if up and pct_b[i] > 0.2 and pct_b[i-1] <= 0.2:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_triple_confirm(df, cooldown, gate):
    """
    3-of-5 confluence: enter when >= 3 of these are bullish on an up brick:
        1. RSI > 50
        2. MACD histogram > 0
        3. Stoch %K > 50
        4. KAMA slope > 0
        5. Supertrend bullish (st_dir > 0)
    """
    n = len(df)
    brick_up = df["brick_up"].values
    rsi = df["rsi"].values
    macd_h = df["macd_hist"].values
    stoch_k = df["stoch_k"].values
    kama_s = df["kama_slope"].values
    st_dir = df["st_dir"].values

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
        if not gate[i]:
            continue
        if (i - last_bar) < cooldown:
            continue

        # Count bullish confirmations
        confirms = 0
        valid = 0
        if not np.isnan(rsi[i]):
            valid += 1
            if rsi[i] > 50:
                confirms += 1
        if not np.isnan(macd_h[i]):
            valid += 1
            if macd_h[i] > 0:
                confirms += 1
        if not np.isnan(stoch_k[i]):
            valid += 1
            if stoch_k[i] > 50:
                confirms += 1
        if not np.isnan(kama_s[i]):
            valid += 1
            if kama_s[i] > 0:
                confirms += 1
        if not np.isnan(st_dir[i]):
            valid += 1
            if st_dir[i] > 0:
                confirms += 1

        if up and valid >= 4 and confirms >= 3:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_escgo_cross(df, cooldown, gate):
    """ESCGO fast crosses above slow on an up brick (Ehlers oscillator)."""
    n = len(df)
    brick_up = df["brick_up"].values
    escgo_f = df["escgo_fast"].values
    escgo_s = df["escgo_slow"].values

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
        if not gate[i]:
            continue
        if (i - last_bar) < cooldown:
            continue
        if np.isnan(escgo_f[i]) or np.isnan(escgo_s[i]) or np.isnan(escgo_f[i-1]) or np.isnan(escgo_s[i-1]):
            continue

        if up and escgo_f[i] > escgo_s[i] and escgo_f[i-1] <= escgo_s[i-1]:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


# -- Strategy dispatcher -------------------------------------------------------

GENERATORS = {
    "mean_rev":       _gen_mean_rev,
    "ema_cross":      _gen_ema_cross,
    "macd_flip":      _gen_macd_flip,
    "squeeze_fire":   _gen_squeeze_fire,
    "kama_turn":      _gen_kama_turn,
    "di_cross":       _gen_di_cross,
    "rsi_regime":     _gen_rsi_regime,
    "bb_bounce":      _gen_bb_bounce,
    "triple_confirm": _gen_triple_confirm,
    "escgo_cross":    _gen_escgo_cross,
}


# -- Backtest runner -----------------------------------------------------------

def _run_backtest(df, entry, exit_, start, end):
    sys.path.insert(0, str(ROOT))
    from engine import BacktestConfig, run_backtest

    df_sig = df.copy()
    df_sig["long_entry"] = entry
    df_sig["long_exit"] = exit_

    cfg = BacktestConfig(
        initial_capital=CAPITAL, commission_pct=COMMISSION, slippage_ticks=0,
        qty_type="cash", qty_value=QTY_VALUE, pyramiding=1,
        start_date=start, end_date=end,
        take_profit_pct=0.0, stop_loss_pct=0.0,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest(df_sig, cfg)

    pf = kpis.get("profit_factor", 0.0) or 0.0
    return {
        "pf":     float("inf") if math.isinf(pf) else float(pf),
        "net":    float(kpis.get("net_profit", 0.0) or 0.0),
        "trades": int(kpis.get("total_trades", 0) or 0),
        "wr":     float(kpis.get("win_rate", 0.0) or 0.0),
        "dd":     float(kpis.get("max_drawdown_pct", 0.0) or 0.0),
    }


# -- Worker: sweep one strategy ------------------------------------------------

def _sweep_strategy(strategy_name):
    """Run all gate/HTF/cooldown combos for a single strategy."""
    label = strategy_name.upper()
    print(f"  [{label}] Loading data...", flush=True)

    df_ltf = _load_ltf_data()
    df_htf = _load_htf_data()

    gen_fn = GENERATORS[strategy_name]

    # Precompute gate arrays for each (use_gates, htf_thresh) combo
    gate_combos = [
        (False, 0,  "no_gates"),
        (True,  0,  "gates_only"),
        (True,  35, "gates+htf35"),
    ]
    gates = {}
    for use_g, htf_t, gname in gate_combos:
        gates[gname] = _compute_gates(df_ltf, df_htf, use_g, htf_t)

    total = len(gate_combos) * len(COOLDOWNS)
    results = []
    done = 0

    for use_g, htf_t, gname in gate_combos:
        gate = gates[gname]
        for cd in COOLDOWNS:
            e, x = gen_fn(df_ltf, cd, gate)

            is_r  = _run_backtest(df_ltf, e, x, IS_START, IS_END)
            oos_r = _run_backtest(df_ltf, e, x, OOS_START, OOS_END)

            is_pf = is_r["pf"]
            oos_pf = oos_r["pf"]
            decay = ((oos_pf - is_pf) / is_pf * 100) \
                    if is_pf > 0 and not math.isinf(is_pf) else float("nan")

            results.append({
                "strategy":   strategy_name,
                "gates":      gname,
                "htf_thresh": htf_t,
                "cooldown":   cd,
                "is_pf":      is_pf,
                "is_trades":  is_r["trades"],
                "is_wr":      is_r["wr"],
                "is_net":     is_r["net"],
                "oos_pf":     oos_pf,
                "oos_trades": oos_r["trades"],
                "oos_wr":     oos_r["wr"],
                "oos_net":    oos_r["net"],
                "oos_dd":     oos_r["dd"],
                "decay_pct":  decay,
            })

            done += 1
            pf_s = f"{oos_pf:.2f}" if not math.isinf(oos_pf) else "inf"
            print(f"  [{label}] {done:>2}/{total} | {gname:<15} cd={cd:>2} | "
                  f"OOS PF={pf_s:>8} T={oos_r['trades']:>4} WR={oos_r['wr']:>5.1f}%",
                  flush=True)

    print(f"  [{label}] Done -- {len(results)} results", flush=True)
    return results


# -- Summary -------------------------------------------------------------------

def _summarize(all_results):
    print(f"\n{'='*100}")
    print("  BTC Creative Strategy Exploration — 10 New Concepts (Long Only)")
    print(f"{'='*100}")

    # -- Per strategy: best config --
    print(f"\n  {'Strategy':<18} {'Best Config':<20} {'cd':>3} | "
          f"{'IS PF':>7} {'T':>4} | {'OOS PF':>8} {'T':>4} {'WR%':>6} {'Net':>8} {'DD%':>7}")
    print(f"  {'-'*98}")

    for sname in STRATEGIES:
        subset = [r for r in all_results if r["strategy"] == sname and r["oos_trades"] >= 5]
        if not subset:
            print(f"  {sname:<18} {'(no viable results)':<20}")
            continue

        best = max(subset, key=lambda r: r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6)
        pf_s = f"{best['oos_pf']:.2f}" if not math.isinf(best['oos_pf']) else "inf"
        is_pf_s = f"{best['is_pf']:.2f}" if not math.isinf(best['is_pf']) else "inf"
        print(f"  {sname:<18} {best['gates']:<20} {best['cooldown']:>3} | "
              f"{is_pf_s:>7} {best['is_trades']:>4} | "
              f"{pf_s:>8} {best['oos_trades']:>4} {best['oos_wr']:>5.1f}% "
              f"{best['oos_net']:>8.2f} {best['oos_dd']:>6.2f}%")

    # -- Gate effectiveness --
    print(f"\n  --- Gate Effectiveness (avg across all strategies, OOS trades >= 5) ---")
    for gname in ["no_gates", "gates_only", "gates+htf35"]:
        rows = [r for r in all_results if r["gates"] == gname and r["oos_trades"] >= 5]
        if rows:
            finite = [r["oos_pf"] for r in rows if not math.isinf(r["oos_pf"])]
            avg_pf = np.mean(finite) if finite else 0
            avg_t = np.mean([r["oos_trades"] for r in rows])
            avg_wr = np.mean([r["oos_wr"] for r in rows])
            print(f"    {gname:<15} avg PF={avg_pf:>7.2f} | avg T={avg_t:>5.1f} | avg WR={avg_wr:>5.1f}% | N={len(rows)}")

    # -- Overall top 20 --
    all_viable = [r for r in all_results if r["oos_trades"] >= 5]
    all_viable.sort(key=lambda r: r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6, reverse=True)

    print(f"\n{'='*100}")
    print("  Overall Top 20 (OOS trades >= 5)")
    print(f"{'='*100}")
    print(f"  {'Strategy':<18} {'Gates':<15} {'cd':>3} | {'IS PF':>7} {'T':>4} | "
          f"{'OOS PF':>8} {'T':>4} {'WR%':>6} {'Net':>8} {'DD%':>7}")
    print(f"  {'-'*98}")
    for r in all_viable[:20]:
        pf_s = f"{r['oos_pf']:.2f}" if not math.isinf(r['oos_pf']) else "inf"
        is_pf_s = f"{r['is_pf']:.2f}" if not math.isinf(r['is_pf']) else "inf"
        print(f"  {r['strategy']:<18} {r['gates']:<15} {r['cooldown']:>3} | "
              f"{is_pf_s:>7} {r['is_trades']:>4} | "
              f"{pf_s:>8} {r['oos_trades']:>4} {r['oos_wr']:>5.1f}% "
              f"{r['oos_net']:>8.2f} {r['oos_dd']:>6.2f}%")

    # -- Top by trade count (PF >= 3) --
    high_trade = [r for r in all_results if r["oos_pf"] >= 3 and r["oos_trades"] >= 15]
    high_trade.sort(key=lambda r: r["oos_trades"], reverse=True)

    if high_trade:
        print(f"\n{'='*100}")
        print("  Top 20 by Trade Count (OOS PF >= 3, trades >= 15)")
        print(f"{'='*100}")
        print(f"  {'Strategy':<18} {'Gates':<15} {'cd':>3} | {'IS PF':>7} {'T':>4} | "
              f"{'OOS PF':>8} {'T':>4} {'WR%':>6} {'Net':>8} {'DD%':>7}")
        print(f"  {'-'*98}")
        for r in high_trade[:20]:
            pf_s = f"{r['oos_pf']:.2f}" if not math.isinf(r['oos_pf']) else "inf"
            is_pf_s = f"{r['is_pf']:.2f}" if not math.isinf(r['is_pf']) else "inf"
            print(f"  {r['strategy']:<18} {r['gates']:<15} {r['cooldown']:>3} | "
                  f"{is_pf_s:>7} {r['is_trades']:>4} | "
                  f"{pf_s:>8} {r['oos_trades']:>4} {r['oos_wr']:>5.1f}% "
                  f"{r['oos_net']:>8.2f} {r['oos_dd']:>6.2f}%")

    # -- BTC003 comparison --
    print(f"\n  --- Reference: BTC003 (R007+FLIP+BB, gates+htf35, cd=30) ---")
    print(f"  OOS: PF=49.27, 76t, WR=76.3%")


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-parallel", action="store_true")
    args = parser.parse_args()

    out_path = ROOT / "ai_context" / "btc_phase8_creative_results.json"
    out_path.parent.mkdir(exist_ok=True)

    total_runs = len(STRATEGIES) * 3 * len(COOLDOWNS)  # 3 gate combos
    print("BTC Creative Strategy Exploration (Long Only)")
    print(f"  Strategies   : {STRATEGIES}")
    print(f"  Gate combos  : no_gates, gates_only, gates+htf35")
    print(f"  Cooldowns    : {COOLDOWNS}")
    print(f"  Total runs   : {total_runs} ({len(STRATEGIES)} strategies x 3 gates x {len(COOLDOWNS)} cooldowns)")
    print(f"  IS period    : {IS_START} -> {IS_END}")
    print(f"  OOS period   : {OOS_START} -> {OOS_END}")
    print()

    all_results = []

    if args.no_parallel:
        for sname in STRATEGIES:
            all_results.extend(_sweep_strategy(sname))
    else:
        with ProcessPoolExecutor(max_workers=min(len(STRATEGIES), 6)) as pool:
            futures = {
                pool.submit(_sweep_strategy, sname): sname
                for sname in STRATEGIES
            }
            for future in as_completed(futures):
                sname = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    print(f"  [{sname.upper()}] finished -- {len(results)} records")
                except Exception as exc:
                    import traceback
                    print(f"  [{sname.upper()}] FAILED: {exc}")
                    traceback.print_exc()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved {len(all_results)} results -> {out_path}")

    _summarize(all_results)


if __name__ == "__main__":
    main()
