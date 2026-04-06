#!/usr/bin/env python3
"""
btc_mk_sweep.py -- BTC Momentum King v4 Strategy Sweep (Long Only)

Target: 1+ trade/day on OANDA BTCUSD $150 Renko bricks.
All strategies are LONG ONLY with first-down-brick exit.

Signal concepts using Momentum King v4 oscillator:
    MK_CROSS       Signal line crossover (smoothed crosses above signal)
    MK_STRONG      Enter on STRONG_UP regime start (momentum + strength)
    MK_FLAT_BREAK  Flat->positive breakout (momentum exits neutral zone upward)
    MK_ZERO_CROSS  Smoothed momentum crosses above zero
    MK_DIVERGENCE  Brick down but MK momentum rising (hidden bullish divergence)

Gate levels:
    none        No gates
    psar        PSAR bullish only
    chop60      Choppiness < 60
    psar_chop   PSAR + chop < 60
    adx25       ADX >= 25
    psar_adx25  PSAR + ADX >= 25

MK v4 params to sweep:
    ema_length:      [10, 14, 20]
    max_smooth:      [5, 8, 10, 15]
    neutral_atr_pct: [0.2, 0.3, 0.5]
    signal_length:   [5, 9, 14]

Cooldowns: [3, 5, 8]
Exit: first down brick

Usage:
    python renko/btc_mk_sweep.py
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
IS_START   = "2024-06-04"
IS_END     = "2025-09-30"
OOS_START  = "2025-10-01"
OOS_END    = "2026-03-19"
COMMISSION = 0.0046
CAPITAL    = 1000.0
QTY_VALUE  = 20   # $20 notional per trade (cash mode)

# -- Data loading ---------------------------------------------------------------

def _load_data():
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    df = load_renko_export(LTF_FILE)
    add_renko_indicators(df)
    return df


# -- MK v4 computation (per-worker, cached with params) -------------------------

def _compute_mk(df, ema_length, max_smooth, neutral_atr_pct, signal_length):
    """Compute MK v4 columns and return as dict of arrays (pre-shifted)."""
    from indicators.momentum_king_v4 import calc_momentum_king_v4
    result = calc_momentum_king_v4(
        df,
        ema_length=ema_length,
        max_smooth=max_smooth,
        neutral_atr_pct=neutral_atr_pct,
        signal_length=signal_length,
    )
    # Pre-shift all outputs (strategy reads at [i] directly)
    shift = lambda arr: np.concatenate([[np.nan if isinstance(arr[0], (float, np.floating)) else False], arr[:-1]])
    return {
        "momentum":  shift(result["smoothed_momentum"]),
        "signal":    shift(result["signal_line"]),
        "strength":  shift(result["momentum_strength"]),
        "cross_up":  shift(result["cross_up"]),
        "cross_dn":  shift(result["cross_dn"]),
        "nz":        shift(result["neutral_zone_width"]),
    }


# -- Gate computation -----------------------------------------------------------

def _compute_gate(df, gate_mode):
    """Compute long gate array for given mode."""
    n = len(df)
    gate = np.ones(n, dtype=bool)

    if "psar" in gate_mode:
        psar = df["psar_dir"].values
        gate &= (np.isnan(psar) | (psar > 0))

    if "chop" in gate_mode:
        chop = df["chop"].values
        gate &= (np.isnan(chop) | (chop < 60))

    if "adx25" in gate_mode:
        adx = df["adx"].values
        gate &= (np.isnan(adx) | (adx >= 25))

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

def _gen_mk_cross(df, mk, cooldown, gate):
    """MK signal line crossover: smoothed crosses above signal on up brick."""
    n = len(df)
    brick_up = df["brick_up"].values
    mom = mk["momentum"]
    sig = mk["signal"]
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
        if np.isnan(mom[i]) or np.isnan(sig[i]) or np.isnan(mom[i-1]) or np.isnan(sig[i-1]):
            continue
        # Crossover: momentum crosses above signal
        if up and mom[i] > sig[i] and mom[i-1] <= sig[i-1]:
            entry[i] = True
            in_pos = True
            last_bar = i
    return entry, exit_


def _gen_mk_strong(df, mk, cooldown, gate, strength_thresh=0.6):
    """Enter when MK transitions to STRONG_UP (momentum > 0, strength > thresh) on up brick."""
    n = len(df)
    brick_up = df["brick_up"].values
    mom = mk["momentum"]
    nz = mk["nz"]
    strength = mk["strength"]
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
        if np.isnan(mom[i]) or np.isnan(strength[i]) or np.isnan(nz[i]):
            continue
        # Strong up: above neutral zone, high strength
        is_strong = mom[i] > nz[i] and strength[i] > strength_thresh
        was_not_strong = np.isnan(mom[i-1]) or np.isnan(strength[i-1]) or mom[i-1] <= nz[i-1] or strength[i-1] <= strength_thresh
        if up and is_strong and was_not_strong:
            entry[i] = True
            in_pos = True
            last_bar = i
    return entry, exit_


def _gen_mk_flat_break(df, mk, cooldown, gate):
    """Flat zone breakout: momentum exits neutral zone upward on up brick."""
    n = len(df)
    brick_up = df["brick_up"].values
    mom = mk["momentum"]
    nz = mk["nz"]
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
        if np.isnan(mom[i]) or np.isnan(nz[i]) or np.isnan(mom[i-1]) or np.isnan(nz[i-1]):
            continue
        # Was in flat zone, now broke above
        was_flat = mom[i-1] < nz[i-1] and mom[i-1] > -nz[i-1]
        now_above = mom[i] > nz[i]
        if up and was_flat and now_above:
            entry[i] = True
            in_pos = True
            last_bar = i
    return entry, exit_


def _gen_mk_zero_cross(df, mk, cooldown, gate):
    """Smoothed momentum crosses above zero on up brick."""
    n = len(df)
    brick_up = df["brick_up"].values
    mom = mk["momentum"]
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
        if np.isnan(mom[i]) or np.isnan(mom[i-1]):
            continue
        if up and mom[i] > 0 and mom[i-1] <= 0:
            entry[i] = True
            in_pos = True
            last_bar = i
    return entry, exit_


def _gen_mk_divergence(df, mk, cooldown, gate, lookback=3):
    """Hidden bullish divergence: brick down but MK momentum rising over lookback bars, then up brick."""
    n = len(df)
    brick_up = df["brick_up"].values
    mom = mk["momentum"]
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
        if i < lookback + 1:
            continue
        if any(np.isnan(mom[i-j]) for j in range(lookback + 1)):
            continue
        # Previous bar was down, momentum has been rising over lookback
        if up and not brick_up[i-1] and mom[i] > mom[i-lookback]:
            entry[i] = True
            in_pos = True
            last_bar = i
    return entry, exit_


# -- Signal dispatch -----------------------------------------------------------

SIGNAL_TYPES = ["mk_cross", "mk_strong", "mk_flat_break", "mk_zero_cross", "mk_divergence"]

def _dispatch_signal(signal_type, df, mk, cooldown, gate):
    if signal_type == "mk_cross":
        return _gen_mk_cross(df, mk, cooldown, gate)
    elif signal_type == "mk_strong":
        return _gen_mk_strong(df, mk, cooldown, gate)
    elif signal_type == "mk_flat_break":
        return _gen_mk_flat_break(df, mk, cooldown, gate)
    elif signal_type == "mk_zero_cross":
        return _gen_mk_zero_cross(df, mk, cooldown, gate)
    elif signal_type == "mk_divergence":
        return _gen_mk_divergence(df, mk, cooldown, gate)
    else:
        raise ValueError(f"Unknown signal: {signal_type}")


# -- Sweep config ---------------------------------------------------------------

GATE_MODES = ["none", "psar", "chop60", "psar_chop", "adx25", "psar_adx25"]
COOLDOWNS  = [3, 5, 8]

MK_PARAMS = {
    "ema_length":      [10, 14, 20],
    "max_smooth":      [5, 8, 10, 15],
    "neutral_atr_pct": [0.2, 0.3, 0.5],
    "signal_length":   [5, 9, 14],
}


# -- Worker --------------------------------------------------------------------

_cache = {}

def _sweep_one(args):
    """Top-level picklable worker."""
    signal_type, gate_mode, cooldown, mk_params = args

    if "df" not in _cache:
        _cache["df"] = _load_data()

    df = _cache["df"]

    # Cache MK computation per unique param set
    mk_key = (mk_params["ema_length"], mk_params["max_smooth"],
              mk_params["neutral_atr_pct"], mk_params["signal_length"])
    if mk_key not in _cache:
        _cache[mk_key] = _compute_mk(
            df,
            ema_length=mk_params["ema_length"],
            max_smooth=mk_params["max_smooth"],
            neutral_atr_pct=mk_params["neutral_atr_pct"],
            signal_length=mk_params["signal_length"],
        )
    mk = _cache[mk_key]

    gate = _compute_gate(df, gate_mode)
    entry, exit_ = _dispatch_signal(signal_type, df, mk, cooldown, gate)

    # Count IS trading days for frequency calc
    is_mask = (df.index >= IS_START) & (df.index <= IS_END)
    is_days = (pd.Timestamp(IS_END) - pd.Timestamp(IS_START)).days

    is_result = _run_bt(df, entry, exit_, IS_START, IS_END)

    # OOS
    oos_result = _run_bt(df, entry, exit_, OOS_START, OOS_END)
    oos_days = (pd.Timestamp(OOS_END) - pd.Timestamp(OOS_START)).days

    return {
        "signal": signal_type,
        "gate": gate_mode,
        "cooldown": cooldown,
        "mk_params": mk_params,
        "is": is_result,
        "oos": oos_result,
        "is_tpd": is_result["trades"] / max(1, is_days),
        "oos_tpd": oos_result["trades"] / max(1, oos_days),
    }


# -- Main -----------------------------------------------------------------------

def build_combos():
    """Build all parameter combinations."""
    import itertools
    combos = []

    # MK param combos
    mk_keys = list(MK_PARAMS.keys())
    mk_combos = [dict(zip(mk_keys, v)) for v in itertools.product(*MK_PARAMS.values())]

    for signal_type in SIGNAL_TYPES:
        for gate in GATE_MODES:
            for cd in COOLDOWNS:
                for mk_p in mk_combos:
                    combos.append((signal_type, gate, cd, mk_p))
    return combos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal", default=None, help="Filter to one signal type")
    parser.add_argument("--top", type=int, default=20, help="Top N results to show")
    parser.add_argument("--output", default="ai_context/btc_mk_sweep_results.json")
    args = parser.parse_args()

    combos = build_combos()

    if args.signal:
        combos = [c for c in combos if c[0] == args.signal]

    print(f"BTC Momentum King v4 Sweep")
    print(f"Total combos: {len(combos)}")
    print(f"Signals: {SIGNAL_TYPES}")
    print(f"Gates: {GATE_MODES}")
    print(f"Cooldowns: {COOLDOWNS}")
    print(f"MK param combos: {len(combos) // (len(SIGNAL_TYPES) * len(GATE_MODES) * len(COOLDOWNS))}")
    print(f"Workers: {MAX_WORKERS}")
    print()

    results = []
    done = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_sweep_one, c): c for c in combos}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            done += 1
            oos = r["oos"]
            if done % 100 == 0 or done == len(combos):
                print(f"  [{done:>5}/{len(combos)}] {r['signal']:16s} {r['gate']:12s} "
                      f"cd={r['cooldown']:>2} OOS PF={oos['pf']:>8.2f} T={oos['trades']:>4} "
                      f"WR={oos['wr']:>5.1f}% t/d={r['oos_tpd']:.1f}")

    # Sort by OOS PF (with min trade filter)
    def sort_key(r):
        oos = r["oos"]
        has_trades = oos["trades"] >= 30
        pf = oos["pf"] if not math.isinf(oos["pf"]) else 1e12
        return (has_trades, pf, oos["net"])

    results.sort(key=sort_key, reverse=True)

    print(f"\n{'='*100}")
    print(f"TOP {args.top} BY OOS PF (min 30 trades)")
    print(f"{'='*100}")
    for r in results[:args.top]:
        oos = r["oos"]
        is_ = r["is"]
        mk = r["mk_params"]
        print(f"  {r['signal']:16s} {r['gate']:12s} cd={r['cooldown']:>2} "
              f"ema={mk['ema_length']:>2} smooth={mk['max_smooth']:>2} "
              f"nz={mk['neutral_atr_pct']:.1f} sig={mk['signal_length']:>2} | "
              f"IS: PF={is_['pf']:>7.2f} T={is_['trades']:>4} WR={is_['wr']:>5.1f}% | "
              f"OOS: PF={oos['pf']:>7.2f} T={oos['trades']:>4} WR={oos['wr']:>5.1f}% "
              f"t/d={r['oos_tpd']:.1f} net=${oos['net']:.2f}")

    # Per-signal-type best
    print(f"\n{'='*100}")
    print(f"BEST PER SIGNAL TYPE (OOS, min 30 trades)")
    print(f"{'='*100}")
    for sig in SIGNAL_TYPES:
        sig_results = [r for r in results if r["signal"] == sig and r["oos"]["trades"] >= 30]
        if sig_results:
            best = sig_results[0]
            oos = best["oos"]
            is_ = best["is"]
            mk = best["mk_params"]
            print(f"  {sig:16s} {best['gate']:12s} cd={best['cooldown']:>2} "
                  f"ema={mk['ema_length']:>2} smooth={mk['max_smooth']:>2} "
                  f"nz={mk['neutral_atr_pct']:.1f} sig={mk['signal_length']:>2} | "
                  f"IS: PF={is_['pf']:>7.2f} T={is_['trades']:>4} WR={is_['wr']:>5.1f}% | "
                  f"OOS: PF={oos['pf']:>7.2f} T={oos['trades']:>4} WR={oos['wr']:>5.1f}% "
                  f"t/d={r['oos_tpd']:.1f}")
        else:
            print(f"  {sig:16s} — no results with 30+ trades")

    # Save JSON
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results[:100], f, indent=2, default=str)
    print(f"\nSaved top 100 to {output_path}")


if __name__ == "__main__":
    main()
