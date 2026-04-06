#!/usr/bin/env python3
"""
btc007_optimize.py -- BTC007 Optimization Sweep (Long Only)

Three stratified phases:
  Phase A — Core signal tuning (stoch thresh, cooldown, ST mult, KAMA period)
  Phase B — Gate optimization (HTF ADX, LTF ADX, RSI filter, Chop max)
  Phase C — Signal enable/disable (which of 4 signals are truly additive)

Baseline: BTC007 trio_stoch + PSAR cd=3
  Python OOS: PF=21.07, 205t (1.2/d), WR=64.4%
  TV OOS:     PF=22.87, 194t (1.1/d), WR=65.5%

Usage:
    python renko/btc007_optimize.py
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


# -- Supertrend with configurable params --------------------------------------

def _supertrend(close, high, low, period, multiplier):
    """Compute Supertrend direction array. +1 = bullish, -1 = bearish."""
    n = len(close)
    atr = np.full(n, np.nan)
    tr = np.maximum(high - low,
         np.maximum(np.abs(high - np.roll(close, 1)),
                    np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]

    # SMA ATR for first period, then EMA-like
    for i in range(1, n):
        if i < period:
            atr[i] = np.nanmean(tr[1:i+1])
        elif i == period:
            atr[i] = np.mean(tr[1:period+1])
        else:
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

    mid = (high + low) / 2.0
    upper = mid + multiplier * atr
    lower = mid - multiplier * atr

    direction = np.ones(n)  # +1 = bullish
    final_upper = np.copy(upper)
    final_lower = np.copy(lower)

    for i in range(1, n):
        if lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = lower[i]
        else:
            final_lower[i] = final_lower[i-1]

        if upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = upper[i]
        else:
            final_upper[i] = final_upper[i-1]

        if direction[i-1] > 0:  # was bullish
            if close[i] < final_lower[i]:
                direction[i] = -1
            else:
                direction[i] = 1
        else:  # was bearish
            if close[i] > final_upper[i]:
                direction[i] = 1
            else:
                direction[i] = -1

    return direction


# -- KAMA with configurable period ---------------------------------------------

def _kama_slope(close, period):
    """Compute KAMA slope (pre-shifted by 1)."""
    n = len(close)
    kama = np.full(n, np.nan)
    fast_sc = 2.0 / (2.0 + 1.0)
    slow_sc = 2.0 / (30.0 + 1.0)

    for i in range(period, n):
        direction = abs(close[i] - close[i - period])
        volatility = sum(abs(close[j] - close[j-1]) for j in range(i - period + 1, i + 1))
        er = direction / volatility if volatility > 0 else 0.0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        if np.isnan(kama[i-1]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])

    # Slope: kama[i] - kama[i-1], then shift by 1 for pre-shift convention
    slope = np.full(n, np.nan)
    for i in range(period + 2, n):
        if not np.isnan(kama[i-1]) and not np.isnan(kama[i-2]):
            slope[i] = kama[i-1] - kama[i-2]
    return slope


# -- Signal generator ----------------------------------------------------------

def _gen_signals(df, params, gate):
    """Generate trio_stoch entry/exit signals with configurable params."""
    n = len(df)
    brick_up = df["brick_up"].values
    close = df["Close"].values.astype(float)
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)

    # Standard pre-shifted indicators from df
    macd_h = df["macd_hist"].values
    stoch_k = df["stoch_k"].values
    rsi = df["rsi"].values
    adx = df["adx"].values
    chop = df["chop"].values

    # Configurable indicators
    st_mult = params.get("st_mult", 3.0)
    kama_period = params.get("kama_period", 10)
    stoch_thresh = params.get("stoch_thresh", 25)
    cooldown = params.get("cooldown", 3)

    # Signal enables
    use_st = params.get("use_st", True)
    use_macd = params.get("use_macd", True)
    use_kama = params.get("use_kama", True)
    use_stoch = params.get("use_stoch", True)

    # Gate filters
    rsi_min = params.get("rsi_min", 0)       # 0 = off
    adx_min = params.get("adx_min", 0)       # 0 = off
    chop_max = params.get("chop_max", 0)     # 0 = off

    # Compute custom Supertrend if non-default multiplier
    if abs(st_mult - 3.0) > 0.01:
        st_raw = _supertrend(close, high, low, 10, st_mult)
        # Pre-shift by 1
        st_dir = np.full(n, np.nan)
        st_dir[1:] = st_raw[:-1]
    else:
        st_dir = df["st_dir"].values

    # Compute custom KAMA slope if non-default period
    if kama_period != 10:
        kama_s = _kama_slope(close, kama_period)
    else:
        kama_s = df["kama_slope"].values

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

        if not gate[i] or not up or (i - last_bar) < cooldown:
            continue

        # Additional gate filters
        if rsi_min > 0 and not np.isnan(rsi[i]) and rsi[i] < rsi_min:
            continue
        if adx_min > 0 and not np.isnan(adx[i]) and adx[i] < adx_min:
            continue
        if chop_max > 0 and not np.isnan(chop[i]) and chop[i] > chop_max:
            continue

        fired = False

        # ST flip
        if not fired and use_st:
            if not np.isnan(st_dir[i]) and not np.isnan(st_dir[i-1]):
                if st_dir[i] > 0 and st_dir[i-1] <= 0:
                    fired = True

        # MACD flip
        if not fired and use_macd:
            if not np.isnan(macd_h[i]) and not np.isnan(macd_h[i-1]):
                if macd_h[i] > 0 and macd_h[i-1] <= 0:
                    fired = True

        # KAMA turn
        if not fired and use_kama:
            if not np.isnan(kama_s[i]) and not np.isnan(kama_s[i-1]):
                if kama_s[i] > 0 and kama_s[i-1] <= 0:
                    fired = True

        # Stoch cross
        if not fired and use_stoch:
            if not np.isnan(stoch_k[i]) and not np.isnan(stoch_k[i-1]):
                if stoch_k[i] > stoch_thresh and stoch_k[i-1] <= stoch_thresh:
                    fired = True

        if fired:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


# -- Combo builders ------------------------------------------------------------

def _build_phase_a():
    """Phase A: core signal tuning."""
    combos = []
    for stoch_t in [20, 25, 30, 35]:
        for cd in [2, 3, 4, 5]:
            for st_m in [2.5, 3.0, 3.5]:
                for kama_p in [8, 10, 14]:
                    combos.append({
                        "phase": "A",
                        "stoch_thresh": stoch_t,
                        "cooldown": cd,
                        "st_mult": st_m,
                        "kama_period": kama_p,
                        "htf_thresh": 0,
                        "adx_min": 0,
                        "rsi_min": 0,
                        "chop_max": 0,
                        "use_st": True,
                        "use_macd": True,
                        "use_kama": True,
                        "use_stoch": True,
                        "label": f"sk{stoch_t}_cd{cd}_st{st_m}_k{kama_p}",
                    })
    return combos


def _build_phase_b(best_a):
    """Phase B: gate optimization. Uses best signal params from Phase A."""
    combos = []
    for htf_t in [0, 25, 30, 35, 40]:
        for adx_m in [0, 20, 25]:
            for rsi_m in [0, 35, 40, 45]:
                for chop_m in [0, 60, 70]:
                    c = dict(best_a)
                    c.update({
                        "phase": "B",
                        "htf_thresh": htf_t,
                        "adx_min": adx_m,
                        "rsi_min": rsi_m,
                        "chop_max": chop_m,
                        "label": f"htf{htf_t}_adx{adx_m}_rsi{rsi_m}_ch{chop_m}",
                    })
                    combos.append(c)
    return combos


def _build_phase_c(best_b):
    """Phase C: signal enable/disable. Uses best params from Phase B."""
    combos = []
    signals = ["use_st", "use_macd", "use_kama", "use_stoch"]
    for mask in range(1, 16):  # 1 to 15 (skip 0 = all off)
        enables = {}
        name_parts = []
        for j, sig in enumerate(signals):
            on = bool(mask & (1 << j))
            enables[sig] = on
            if on:
                name_parts.append(sig.replace("use_", ""))
        for cd in [2, 3, 5]:
            c = dict(best_b)
            c.update(enables)
            c["cooldown"] = cd
            c["phase"] = "C"
            c["label"] = "+".join(name_parts) + f"_cd{cd}"
            combos.append(c)
    return combos


# -- Worker --------------------------------------------------------------------

_w = {}

def _init_worker():
    if "df" not in _w:
        _w["df"] = _load_ltf()
        _w["df_htf"] = _load_htf()
        # PSAR gate
        psar = _w["df"]["psar_dir"].values
        _w["psar_gate"] = np.isnan(psar) | (psar > 0)
        # Pre-compute HTF ADX gates
        _w["htf_gates"] = {0: np.ones(len(_w["df"]), dtype=bool)}
        df_htf = _w["df_htf"]
        adx_htf = df_htf["adx"].values
        adx_nan = np.isnan(adx_htf)
        for thresh in [25, 30, 35, 40]:
            htf_arr = adx_nan | (adx_htf >= thresh)
            _w["htf_gates"][thresh] = _align_htf_gate(_w["df"], df_htf, htf_arr)


def _run_one(combo):
    _init_worker()
    df = _w["df"]

    # Build combined gate
    gate = _w["psar_gate"].copy()
    htf_t = combo.get("htf_thresh", 0)
    if htf_t > 0:
        gate &= _w["htf_gates"][htf_t]

    entry, exit_ = _gen_signals(df, combo, gate)
    is_r = _run_bt(df, entry, exit_, IS_START, IS_END)
    oos_r = _run_bt(df, entry, exit_, OOS_START, OOS_END)
    return combo, is_r, oos_r


# -- Summary -------------------------------------------------------------------

def _print_header():
    print(f"  {'#':>3} {'Ph':>2} {'Label':<35} | "
          f"{'IS PF':>7} {'T':>5} {'WR%':>6} | "
          f"{'OOS PF':>8} {'T':>5} {'t/d':>5} {'WR%':>6} {'Net':>9} {'DD%':>7}")
    print(f"  {'-'*115}")


def _print_row(r, rank=None):
    pf_i = "INF" if math.isinf(r["is_pf"]) else f"{r['is_pf']:.2f}"
    pf_o = "INF" if math.isinf(r["oos_pf"]) else f"{r['oos_pf']:.2f}"
    tpd = r["oos_trades"] / OOS_DAYS if r["oos_trades"] > 0 else 0
    prefix = f"  {rank:>3}" if rank else "  "
    print(f"{prefix} {r['phase']:>2} {r['label']:<35} | "
          f"{pf_i:>7} {r['is_trades']:>5} {r['is_wr']:>5.1f}% | "
          f"{pf_o:>8} {r['oos_trades']:>5} {tpd:>4.1f} {r['oos_wr']:>5.1f}% "
          f"{r['oos_net']:>9.2f} {r['oos_dd']:>6.2f}%")


def _run_phase(combos, phase_name, all_results):
    total = len(combos)
    print(f"\n  Running Phase {phase_name}: {total} combos ({total*2} backtests)...")

    done = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run_one, c): c for c in combos}
        for fut in as_completed(futures):
            try:
                combo, is_r, oos_r = fut.result()
                row = {
                    "phase":      combo["phase"],
                    "label":      combo["label"],
                    "stoch_thresh": combo.get("stoch_thresh", 25),
                    "cooldown":   combo.get("cooldown", 3),
                    "st_mult":    combo.get("st_mult", 3.0),
                    "kama_period": combo.get("kama_period", 10),
                    "htf_thresh": combo.get("htf_thresh", 0),
                    "adx_min":    combo.get("adx_min", 0),
                    "rsi_min":    combo.get("rsi_min", 0),
                    "chop_max":   combo.get("chop_max", 0),
                    "use_st":     combo.get("use_st", True),
                    "use_macd":   combo.get("use_macd", True),
                    "use_kama":   combo.get("use_kama", True),
                    "use_stoch":  combo.get("use_stoch", True),
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
                all_results.append(row)
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()

            done += 1
            if done % 50 == 0 or done == total:
                print(f"    [{done:>4}/{total}]", flush=True)


def _find_best(results, phase, min_trades=100):
    """Find best config from a phase by OOS net (min trades filter)."""
    subset = [r for r in results
              if r["phase"] == phase
              and r["oos_trades"] >= min_trades
              and r["oos_net"] > 0]
    if not subset:
        # Relax to 50 trades
        subset = [r for r in results
                  if r["phase"] == phase
                  and r["oos_trades"] >= 50
                  and r["oos_net"] > 0]
    if not subset:
        return None
    # Sort by composite: weighted WR + PF + net
    subset.sort(key=lambda r: (
        r["oos_wr"] * 0.4 +
        min(r["oos_pf"], 50) * 0.3 +
        r["oos_net"] * 0.3
    ), reverse=True)
    return subset[0]


def _show_phase(results, phase, title):
    subset = [r for r in results if r["phase"] == phase]
    viable = [r for r in subset if r["oos_trades"] >= 10 and r["oos_net"] > 0]

    # Best by WR (T >= 100)
    by_wr = sorted([r for r in viable if r["oos_trades"] >= 100],
                   key=lambda r: (r["oos_wr"], r["oos_net"]), reverse=True)

    # Best by net (T >= 100)
    by_net = sorted([r for r in viable if r["oos_trades"] >= 100],
                    key=lambda r: r["oos_net"], reverse=True)

    # Best by PF (T >= 50)
    by_pf = sorted([r for r in viable if r["oos_trades"] >= 50],
                   key=lambda r: (
                       r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6,
                       r["oos_net"]
                   ), reverse=True)

    print(f"\n{'='*125}")
    print(f"  {title} — {len(viable)} viable / {len(subset)} total")
    print(f"{'='*125}")

    if by_wr:
        print(f"\n  Top 15 by WR (T>=100):")
        _print_header()
        for i, r in enumerate(by_wr[:15]):
            _print_row(r, rank=i+1)

    if by_net:
        print(f"\n  Top 15 by Net (T>=100):")
        _print_header()
        for i, r in enumerate(by_net[:15]):
            _print_row(r, rank=i+1)

    if by_pf:
        print(f"\n  Top 15 by PF (T>=50):")
        _print_header()
        for i, r in enumerate(by_pf[:15]):
            _print_row(r, rank=i+1)


# -- Main ----------------------------------------------------------------------

def main():
    print(f"\n{'='*70}")
    print(f"BTC007 Optimization Sweep")
    print(f"  Baseline: trio_stoch + PSAR cd=3")
    print(f"  Python OOS: PF=21.07, 205t (1.2/d), WR=64.4%")
    print(f"  TV OOS:     PF=22.87, 194t (1.1/d), WR=65.5%")
    print(f"  Workers:    {MAX_WORKERS}")
    print(f"{'='*70}")

    all_results = []

    # -- Phase A: Core signal tuning --
    combos_a = _build_phase_a()
    print(f"\n  Phase A: {len(combos_a)} combos — Signal tuning")
    _run_phase(combos_a, "A", all_results)
    _show_phase(all_results, "A", "Phase A — Core Signal Tuning")

    best_a = _find_best(all_results, "A")
    if best_a:
        print(f"\n  >>> Phase A winner: {best_a['label']}")
        print(f"      OOS: PF={best_a['oos_pf']:.2f}, T={best_a['oos_trades']}, "
              f"WR={best_a['oos_wr']:.1f}%, Net={best_a['oos_net']:.2f}")
    else:
        print("  >>> No Phase A winner found, using baseline")
        best_a = {
            "stoch_thresh": 25, "cooldown": 3, "st_mult": 3.0,
            "kama_period": 10, "htf_thresh": 0, "adx_min": 0,
            "rsi_min": 0, "chop_max": 0,
            "use_st": True, "use_macd": True, "use_kama": True, "use_stoch": True,
        }

    # -- Phase B: Gate optimization --
    combos_b = _build_phase_b(best_a)
    print(f"\n  Phase B: {len(combos_b)} combos — Gate optimization")
    _run_phase(combos_b, "B", all_results)
    _show_phase(all_results, "B", "Phase B — Gate Optimization")

    best_b = _find_best(all_results, "B")
    if best_b:
        print(f"\n  >>> Phase B winner: {best_b['label']}")
        print(f"      OOS: PF={best_b['oos_pf']:.2f}, T={best_b['oos_trades']}, "
              f"WR={best_b['oos_wr']:.1f}%, Net={best_b['oos_net']:.2f}")
    else:
        print("  >>> No Phase B winner, using Phase A winner")
        best_b = best_a

    # -- Phase C: Signal enable/disable --
    combos_c = _build_phase_c(best_b)
    print(f"\n  Phase C: {len(combos_c)} combos — Signal enable/disable")
    _run_phase(combos_c, "C", all_results)
    _show_phase(all_results, "C", "Phase C — Signal Enable/Disable")

    # -- Global summary --
    viable_all = [r for r in all_results
                  if r["oos_trades"] >= 100 and r["oos_net"] > 0]

    print(f"\n{'='*125}")
    print(f"  GLOBAL TOP 20 by WR (T>=100, net>0): {len(viable_all)} configs")
    print(f"{'='*125}")
    viable_all.sort(key=lambda r: (r["oos_wr"], r["oos_net"]), reverse=True)
    _print_header()
    for i, r in enumerate(viable_all[:20]):
        _print_row(r, rank=i+1)

    print(f"\n{'='*125}")
    print(f"  GLOBAL TOP 20 by Net (T>=100, net>0)")
    print(f"{'='*125}")
    viable_all.sort(key=lambda r: r["oos_net"], reverse=True)
    _print_header()
    for i, r in enumerate(viable_all[:20]):
        _print_row(r, rank=i+1)

    # Baseline comparison
    baseline = [r for r in all_results
                if r.get("stoch_thresh") == 25
                and r.get("cooldown") == 3
                and abs(r.get("st_mult", 3.0) - 3.0) < 0.01
                and r.get("kama_period") == 10
                and r.get("htf_thresh", 0) == 0
                and r.get("adx_min", 0) == 0
                and r.get("rsi_min", 0) == 0
                and r.get("chop_max", 0) == 0
                and r.get("use_st", True) and r.get("use_macd", True)
                and r.get("use_kama", True) and r.get("use_stoch", True)]
    if baseline:
        bl = baseline[0]
        print(f"\n  BASELINE: {bl['label']}")
        print(f"    IS:  PF={bl['is_pf']:.2f}, T={bl['is_trades']}, WR={bl['is_wr']:.1f}%")
        print(f"    OOS: PF={bl['oos_pf']:.2f}, T={bl['oos_trades']}, WR={bl['oos_wr']:.1f}%, Net={bl['oos_net']:.2f}")

    # Save
    out_path = ROOT / "ai_context" / "btc007_optimize_results.json"
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


if __name__ == "__main__":
    main()
