#!/usr/bin/env python3
"""
combo_cci_optimize.py -- Fine-grained parameter sweep for COMBO_CCI on BTC $150 Renko.

Baseline winner: ema_len=14, cci_len=10, fast_ma=10, slow_ma=20, gate=adx25, cd=20
OOS PF=37.71, T=102, WR=72.5%, t/d=0.6

Sweep:
    ema_len:  [10, 12, 14, 16, 18, 20]
    cci_len:  [6, 8, 10, 12, 14, 16]
    fast_ma:  [5, 8, 10, 12, 15]
    slow_ma:  [15, 18, 20, 22, 25, 30]
    gates:    [none, psar, adx25, psar_adx25]
    cooldowns:[3, 5, 10, 15, 20, 25]

Total combos: 6*6*5*6*4*6 = 25,920

Usage:
    python renko/combo_cci_optimize.py
"""

import contextlib
import io
import json
import math
import sys
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from renko.config import MAX_WORKERS

# -- Config --------------------------------------------------------------------

LTF_FILE   = "OANDA_BTCUSD.SPOT.US, 1S renko 150.csv"
IS_START   = "2024-06-04"
IS_END     = "2025-09-30"
OOS_START  = "2025-10-01"
OOS_END    = "2026-03-19"
OOS_DAYS   = 170
COMMISSION = 0.0046
CAPITAL    = 1000.0
QTY_VALUE  = 20
MIN_OOS_TRADES = 20

BASELINE = {
    "ema_len": 14, "cci_len": 10, "fast_ma": 10, "slow_ma": 20,
    "gate": "adx25", "cooldown": 20,
    "oos_pf": 37.71, "oos_trades": 102, "oos_wr": 72.5, "oos_tpd": 0.6,
}

EMA_LENS  = [10, 12, 14, 16, 18, 20]
CCI_LENS  = [6, 8, 10, 12, 14, 16]
FAST_MAS  = [5, 8, 10, 12, 15]
SLOW_MAS  = [15, 18, 20, 22, 25, 30]
GATES     = ["none", "psar", "adx25", "psar_adx25"]
COOLDOWNS = [3, 5, 10, 15, 20, 25]


# -- Data + indicators ---------------------------------------------------------

def _load_data():
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    df = load_renko_export(LTF_FILE)
    add_renko_indicators(df)
    return df


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


# -- Signal generator ----------------------------------------------------------

def _calc_ema(series, length):
    return pd.Series(series).ewm(span=length, adjust=False).mean().values


def _calc_sma(series, length):
    return pd.Series(series).rolling(length, min_periods=1).mean().values


def _ema20_pos(close, high, low, length):
    """HPotter EMA20 trend: +1 bullish, -1 bearish."""
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


def _gen_combo_cci(df, cooldown, gate, ema_len, cci_len, fast_ma, slow_ma):
    """
    COMBO_CCI: HPotter EMA20-trend +1 AND CCI(fast_ma) > CCI(slow_ma).
    """
    n = len(df)
    close    = df["Close"].values
    high     = df["High"].values
    low      = df["Low"].values
    brick_up = df["brick_up"].values

    ema20_pos = _ema20_pos(close, high, low, ema_len)

    # CCI
    typical  = (high + low + close) / 3.0
    cci_sma  = _calc_sma(typical, cci_len)
    mad = pd.Series(typical).rolling(cci_len, min_periods=1).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    with np.errstate(divide="ignore", invalid="ignore"):
        cci_val = np.where(mad == 0, 0.0, (typical - cci_sma) / (0.015 * mad))

    cci_fast = _calc_sma(cci_val, fast_ma)
    cci_slow = _calc_sma(cci_val, slow_ma)

    # Signal: ema20 +1 AND cci_fast > cci_slow (sustained state, entry on transition)
    sig_pos = np.where((ema20_pos == 1) & (cci_fast > cci_slow), 1.0, -1.0)

    warmup = ema_len + cci_len + slow_ma + 5
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
        if sig_pos[i] == 1.0 and sig_pos[i-1] != 1.0 and brick_up[i]:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


# -- Worker --------------------------------------------------------------------

_worker_cache = {}


def _worker_init():
    _worker_cache["df"] = _load_data()


def _run_single(task):
    df = _worker_cache["df"]
    ema_len  = task["ema_len"]
    cci_len  = task["cci_len"]
    fast_ma  = task["fast_ma"]
    slow_ma  = task["slow_ma"]
    cd       = task["cooldown"]
    gate_mode = task["gate"]

    # Skip invalid: fast_ma must be < slow_ma
    if fast_ma >= slow_ma:
        return {**task, "skip": True}

    gate = _compute_gate(df, gate_mode)

    try:
        entry, exit_ = _gen_combo_cci(df, cd, gate, ema_len, cci_len, fast_ma, slow_ma)
    except Exception as e:
        return {**task, "error": str(e)}

    is_kpis  = _run_bt(df, entry, exit_, IS_START, IS_END)
    oos_kpis = _run_bt(df, entry, exit_, OOS_START, OOS_END)
    oos_kpis["tpd"] = round(oos_kpis["trades"] / OOS_DAYS, 2)

    return {**task, "is": is_kpis, "oos": oos_kpis}


# -- Main ----------------------------------------------------------------------

def main():
    # Build tasks
    tasks = []
    for ema_len, cci_len, fast_ma, slow_ma, gate, cd in itertools.product(
            EMA_LENS, CCI_LENS, FAST_MAS, SLOW_MAS, GATES, COOLDOWNS):
        if fast_ma >= slow_ma:
            continue   # skip invalid combos up-front
        tasks.append({
            "ema_len": ema_len, "cci_len": cci_len,
            "fast_ma": fast_ma, "slow_ma": slow_ma,
            "gate": gate, "cooldown": cd,
        })

    print(f"COMBO_CCI Optimization: {len(tasks)} valid combos (of 25,920 total)")
    print(f"Workers: {MAX_WORKERS}\n")

    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=_worker_init) as pool:
        futures = {pool.submit(_run_single, t): t for t in tasks}
        done = 0
        for f in as_completed(futures):
            done += 1
            r = f.result()
            results.append(r)
            if done % 500 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} complete...")

    # Filter and sort
    valid = [r for r in results
             if "error" not in r and "skip" not in r
             and r.get("oos", {}).get("trades", 0) >= MIN_OOS_TRADES]
    valid.sort(key=lambda x: x["oos"]["pf"], reverse=True)

    fmt = lambda pf: "INF" if math.isinf(pf) else f"{pf:.2f}"

    # Print top 30
    print(f"\n{'='*120}")
    print(f"TOP 30 BY OOS PF (min {MIN_OOS_TRADES} OOS trades)")
    print(f"{'='*120}")
    print(f"{'ema':>4} {'cci':>4} {'fast':>5} {'slow':>5} {'gate':<12} {'CD':>3}  "
          f"{'IS_PF':>7} {'IS_T':>5} {'IS_WR':>6}  "
          f"{'OOS_PF':>7} {'OOS_T':>5} {'OOS_WR':>6} {'t/d':>5} {'OOS_Net':>9}")
    print("-" * 120)

    for r in valid[:30]:
        pf_str = fmt(r["oos"]["pf"])
        bl = " <-- BASELINE" if (
            r["ema_len"] == BASELINE["ema_len"] and
            r["cci_len"] == BASELINE["cci_len"] and
            r["fast_ma"] == BASELINE["fast_ma"] and
            r["slow_ma"] == BASELINE["slow_ma"] and
            r["gate"]    == BASELINE["gate"] and
            r["cooldown"] == BASELINE["cooldown"]
        ) else ""
        print(f"{r['ema_len']:>4} {r['cci_len']:>4} {r['fast_ma']:>5} {r['slow_ma']:>5} "
              f"{r['gate']:<12} {r['cooldown']:>3}  "
              f"{r['is']['pf']:>7.2f} {r['is']['trades']:>5} {r['is']['wr']:>5.1f}%  "
              f"{pf_str:>7} {r['oos']['trades']:>5} {r['oos']['wr']:>5.1f}% "
              f"{r['oos']['tpd']:>5.1f} {r['oos']['net']:>9.2f}{bl}")

    # Compare baseline vs best
    best = valid[0] if valid else None
    print(f"\n{'='*80}")
    print("BASELINE vs BEST")
    print(f"{'='*80}")
    print(f"  Baseline:  ema={BASELINE['ema_len']} cci={BASELINE['cci_len']} "
          f"fast={BASELINE['fast_ma']} slow={BASELINE['slow_ma']} "
          f"gate={BASELINE['gate']} cd={BASELINE['cooldown']}")
    print(f"             OOS: PF={BASELINE['oos_pf']} T={BASELINE['oos_trades']} "
          f"WR={BASELINE['oos_wr']}% t/d={BASELINE['oos_tpd']}")
    if best:
        pf_str = fmt(best["oos"]["pf"])
        print(f"  Best opt:  ema={best['ema_len']} cci={best['cci_len']} "
              f"fast={best['fast_ma']} slow={best['slow_ma']} "
              f"gate={best['gate']} cd={best['cooldown']}")
        print(f"             OOS: PF={pf_str} T={best['oos']['trades']} "
              f"WR={best['oos']['wr']:.1f}% t/d={best['oos']['tpd']:.1f} "
              f"Net=${best['oos']['net']:.2f}")
        if not math.isinf(best["oos"]["pf"]):
            delta = best["oos"]["pf"] - BASELINE["oos_pf"]
            print(f"             Delta PF: {delta:+.2f} | WR: {best['oos']['wr'] - BASELINE['oos_wr']:+.1f}pp")

    # Save JSON
    out_json = ROOT / "ai_context" / "combo_cci_optimize_results.json"
    with open(out_json, "w") as f:
        json.dump({
            "baseline": BASELINE,
            "best": best,
            "top50": valid[:50],
            "all_valid": valid,
        }, f, indent=2, default=str)
    print(f"\nSaved {len(valid)} valid results to {out_json}")

    # Write markdown report
    _write_report(valid, best)
    return valid, best


def _write_report(valid, best):
    from datetime import date
    today = date.today()

    fmt = lambda pf: "INF" if math.isinf(pf) else f"{pf:.2f}"

    lines = []
    lines.append("# COMBO_CCI Optimization Report")
    lines.append(f"\n**Date:** {today}  |  **Instrument:** BTCUSD $150 Renko")
    lines.append(f"**IS:** 2024-06-04 to 2025-09-30  |  **OOS:** 2025-10-01 to 2026-03-19")
    lines.append(f"\n---")

    # Baseline
    lines.append(f"\n## Baseline (Wizard Sweep Winner)")
    lines.append(f"\n| Param | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| ema_len | {BASELINE['ema_len']} |")
    lines.append(f"| cci_len | {BASELINE['cci_len']} |")
    lines.append(f"| fast_ma | {BASELINE['fast_ma']} |")
    lines.append(f"| slow_ma | {BASELINE['slow_ma']} |")
    lines.append(f"| gate | {BASELINE['gate']} |")
    lines.append(f"| cooldown | {BASELINE['cooldown']} |")
    lines.append(f"\n**OOS:** PF={BASELINE['oos_pf']} | T={BASELINE['oos_trades']} | WR={BASELINE['oos_wr']}% | t/d={BASELINE['oos_tpd']}")

    lines.append(f"\n---")

    # Best
    if best:
        pf_str = fmt(best["oos"]["pf"])
        lines.append(f"\n## Best Optimized Config")
        lines.append(f"\n| Param | Baseline | Optimized | Change |")
        lines.append("|-------|----------|-----------|--------|")
        lines.append(f"| ema_len | {BASELINE['ema_len']} | {best['ema_len']} | {'same' if best['ema_len']==BASELINE['ema_len'] else str(best['ema_len']-BASELINE['ema_len'])} |")
        lines.append(f"| cci_len | {BASELINE['cci_len']} | {best['cci_len']} | {'same' if best['cci_len']==BASELINE['cci_len'] else str(best['cci_len']-BASELINE['cci_len'])} |")
        lines.append(f"| fast_ma | {BASELINE['fast_ma']} | {best['fast_ma']} | {'same' if best['fast_ma']==BASELINE['fast_ma'] else str(best['fast_ma']-BASELINE['fast_ma'])} |")
        lines.append(f"| slow_ma | {BASELINE['slow_ma']} | {best['slow_ma']} | {'same' if best['slow_ma']==BASELINE['slow_ma'] else str(best['slow_ma']-BASELINE['slow_ma'])} |")
        lines.append(f"| gate | {BASELINE['gate']} | {best['gate']} | {'same' if best['gate']==BASELINE['gate'] else 'changed'} |")
        lines.append(f"| cooldown | {BASELINE['cooldown']} | {best['cooldown']} | {'same' if best['cooldown']==BASELINE['cooldown'] else str(best['cooldown']-BASELINE['cooldown'])} |")
        lines.append(f"\n**IS:**  PF={best['is']['pf']:.2f} | T={best['is']['trades']} | WR={best['is']['wr']:.1f}%")
        lines.append(f"\n**OOS:** PF={pf_str} | T={best['oos']['trades']} | WR={best['oos']['wr']:.1f}% | t/d={best['oos']['tpd']:.1f} | Net=${best['oos']['net']:.2f}")
        if not math.isinf(best["oos"]["pf"]):
            delta_pf = best["oos"]["pf"] - BASELINE["oos_pf"]
            delta_wr = best["oos"]["wr"] - BASELINE["oos_wr"]
            lines.append(f"\n**Improvement vs baseline:** PF {delta_pf:+.2f} | WR {delta_wr:+.1f}pp")

    lines.append(f"\n---")

    # Top 20 table
    lines.append(f"\n## Top 20 Configs (OOS PF, min {MIN_OOS_TRADES} trades)")
    lines.append(f"\n| Rank | ema | cci | fast | slow | gate | cd | IS PF | IS T | OOS PF | OOS T | WR | t/d | Net |")
    lines.append("|------|-----|-----|------|------|------|----|-------|------|--------|-------|----|-----|-----|")

    for i, r in enumerate(valid[:20], 1):
        pf_str = fmt(r["oos"]["pf"])
        bl = " *" if (r["ema_len"]==BASELINE["ema_len"] and r["cci_len"]==BASELINE["cci_len"]
                      and r["fast_ma"]==BASELINE["fast_ma"] and r["slow_ma"]==BASELINE["slow_ma"]
                      and r["gate"]==BASELINE["gate"] and r["cooldown"]==BASELINE["cooldown"]) else ""
        lines.append(
            f"| {i}{bl} | {r['ema_len']} | {r['cci_len']} | {r['fast_ma']} | {r['slow_ma']} "
            f"| {r['gate']} | {r['cooldown']} "
            f"| {r['is']['pf']:.2f} | {r['is']['trades']} "
            f"| {pf_str} | {r['oos']['trades']} | {r['oos']['wr']:.1f}% "
            f"| {r['oos']['tpd']:.1f} | ${r['oos']['net']:.2f} |"
        )

    lines.append(f"\n_* = baseline config_")

    lines.append(f"\n---")
    lines.append(f"\n## Sweep Parameters")
    lines.append(f"\n- ema_len: {EMA_LENS}")
    lines.append(f"- cci_len: {CCI_LENS}")
    lines.append(f"- fast_ma: {FAST_MAS}")
    lines.append(f"- slow_ma: {SLOW_MAS}")
    lines.append(f"- gates: {GATES}")
    lines.append(f"- cooldowns: {COOLDOWNS}")
    lines.append(f"- Constraint: fast_ma < slow_ma")
    lines.append(f"- Min OOS trades: {MIN_OOS_TRADES}")

    out_md = ROOT / "ai_context" / "combo_cci_optimize_report.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved report to {out_md}")


if __name__ == "__main__":
    main()
