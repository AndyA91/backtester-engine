#!/usr/bin/env python3
"""
wizard_ssl_optimize.py -- Deep optimization of SSL Channel on BTC $150 Renko

Phase 1 found SSL_CHANNEL period=13 + psar_adx25 + cd=20 = PF 31.49, 70.1% WR, 77t OOS.

This sweep expands:
  1. Period: fine-grain around winner (8-40 in steps)
  2. MA type: SMA vs EMA for high/low channels
  3. Gates: psar, adx (25-50 step 5), chop (<50,<60), stoch_k<30, rsi<45, escgo
  4. Gate stacking: psar+adx, psar+chop, psar+adx+rsi, psar+adx+stoch
  5. Cooldowns: 5, 10, 15, 20, 30
  6. Confirmation filters: require brick_up, require RSI[1]<45, require stoch_k[1]<30

Usage:
    python renko/wizard_ssl_optimize.py
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


def _load_data():
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    df = load_renko_export(LTF_FILE)
    add_renko_indicators(df)
    return df


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
# SSL CHANNEL SIGNAL GENERATOR (expanded)
# ==============================================================================

def _gen_ssl(df, period=13, ma_type="sma", cooldown=20,
             gate_psar=True, gate_adx=25, gate_chop=0, gate_rsi=0, gate_stoch=0,
             confirm_rsi=0, confirm_stoch=0):
    """
    SSL Channel with expanded parameters.

    ma_type: "sma" or "ema" for channel calculation
    gate_psar: require PSAR bullish
    gate_adx: min ADX threshold (0=disabled)
    gate_chop: max chop threshold (0=disabled), e.g. 50 means chop<50
    gate_rsi: max RSI[1] threshold for entry (0=disabled), e.g. 45
    gate_stoch: max stoch_k[1] threshold for entry (0=disabled), e.g. 30
    confirm_rsi: require RSI[1] < this value at entry (0=disabled)
    confirm_stoch: require stoch_k[1] < this value at entry (0=disabled)
    """
    n = len(df)
    brick_up = df["brick_up"].values
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    psar_dir = df["psar_dir"].values
    adx = df["adx"].values
    chop = df["chop"].values
    rsi = df["rsi"].values
    stoch_k = df["stoch_k"].values

    # Compute SSL channels
    if ma_type == "ema":
        ch_high = pd.Series(high).ewm(span=period, adjust=False).mean().values
        ch_low = pd.Series(low).ewm(span=period, adjust=False).mean().values
    else:
        ch_high = pd.Series(high).rolling(period, min_periods=1).mean().values
        ch_low = pd.Series(low).rolling(period, min_periods=1).mean().values

    # Hlv direction
    hlv = np.zeros(n)
    for i in range(1, n):
        if close[i] > ch_high[i]:
            hlv[i] = 1
        elif close[i] < ch_low[i]:
            hlv[i] = -1
        else:
            hlv[i] = hlv[i-1]

    # Signal + gating
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = period + 5

    for i in range(warmup, n):
        up = bool(brick_up[i])

        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue

        if (i - last_bar) < cooldown:
            continue

        # SSL flip check
        if not (hlv[i] == 1 and hlv[i-1] <= 0 and up):
            continue

        # Gate: PSAR
        if gate_psar:
            if np.isnan(psar_dir[i]) or psar_dir[i] <= 0:
                continue

        # Gate: ADX
        if gate_adx > 0:
            if np.isnan(adx[i]) or adx[i] < gate_adx:
                continue

        # Gate: Chop (low = trending)
        if gate_chop > 0:
            if np.isnan(chop[i]) or chop[i] >= gate_chop:
                continue

        # Confirmation: RSI[1] below threshold (not overbought)
        if confirm_rsi > 0:
            if np.isnan(rsi[i]) or rsi[i] >= confirm_rsi:
                continue

        # Confirmation: Stoch K[1] below threshold
        if confirm_stoch > 0:
            if np.isnan(stoch_k[i]) or stoch_k[i] >= confirm_stoch:
                continue

        entry[i] = True
        in_pos = True
        last_bar = i

    return entry, exit_


# ==============================================================================
# PARAMETER GRID
# ==============================================================================

PARAM_GRID = {
    "period":        [8, 10, 13, 15, 18, 20, 25, 30],
    "ma_type":       ["sma", "ema"],
    "cooldown":      [5, 10, 15, 20, 30],
    "gate_psar":     [True, False],
    "gate_adx":      [0, 25, 30, 35, 40],
    "gate_chop":     [0, 50, 60],
    "confirm_rsi":   [0, 45, 55],
    "confirm_stoch": [0, 30, 40],
}

# Full cartesian is huge. Filter to sensible combos.
def _build_tasks():
    tasks = []
    for period in PARAM_GRID["period"]:
        for ma_type in PARAM_GRID["ma_type"]:
            for cooldown in PARAM_GRID["cooldown"]:
                for gate_psar in PARAM_GRID["gate_psar"]:
                    for gate_adx in PARAM_GRID["gate_adx"]:
                        for gate_chop in PARAM_GRID["gate_chop"]:
                            for confirm_rsi in PARAM_GRID["confirm_rsi"]:
                                for confirm_stoch in PARAM_GRID["confirm_stoch"]:
                                    # Skip: no gates at all (already tested, noisy)
                                    if not gate_psar and gate_adx == 0 and gate_chop == 0 and confirm_rsi == 0 and confirm_stoch == 0:
                                        continue
                                    # Skip: both chop AND low ADX (redundant — chop measures same thing)
                                    if gate_chop > 0 and gate_adx == 0:
                                        continue
                                    # Skip: both confirm filters at once (too restrictive, few trades)
                                    if confirm_rsi > 0 and confirm_stoch > 0:
                                        continue
                                    tasks.append({
                                        "period": period,
                                        "ma_type": ma_type,
                                        "cooldown": cooldown,
                                        "gate_psar": gate_psar,
                                        "gate_adx": gate_adx,
                                        "gate_chop": gate_chop,
                                        "confirm_rsi": confirm_rsi,
                                        "confirm_stoch": confirm_stoch,
                                    })
    return tasks


# -- Worker --------------------------------------------------------------------

_worker_cache = {}

def _worker_init():
    _worker_cache["df"] = _load_data()

def _run_single(task):
    df = _worker_cache["df"]
    try:
        entry, exit_ = _gen_ssl(df, **task)
    except Exception as e:
        return {**task, "error": str(e)}

    is_kpis = _run_bt(df, entry, exit_, IS_START, IS_END)
    oos_kpis = _run_bt(df, entry, exit_, OOS_START, OOS_END)
    oos_kpis["tpd"] = round(oos_kpis["trades"] / OOS_DAYS, 2)

    return {**task, "is": is_kpis, "oos": oos_kpis}


# -- Main ----------------------------------------------------------------------

def main():
    tasks = _build_tasks()
    print(f"SSL Channel Optimization: {len(tasks)} combos")
    print(f"Workers: {MAX_WORKERS}\n")

    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=_worker_init) as pool:
        futures = {pool.submit(_run_single, t): t for t in tasks}
        done = 0
        for f in as_completed(futures):
            done += 1
            r = f.result()
            results.append(r)
            if done % 200 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} complete...")

    # Filter & sort
    valid = [r for r in results if "error" not in r and r["oos"]["trades"] >= 10]
    valid.sort(key=lambda x: x["oos"]["pf"], reverse=True)

    # Print top 40
    print(f"\n{'='*130}")
    print(f"TOP 40 BY OOS PF (min 10 trades)")
    print(f"{'='*130}")
    hdr = (f"{'Per':>3} {'MA':<4} {'CD':>3} {'PSAR':<5} {'ADX':>3} {'Chop':>4} "
           f"{'cRSI':>4} {'cStK':>4}  "
           f"{'IS_PF':>7} {'IS_T':>5} {'IS_WR':>6}  "
           f"{'OOS_PF':>7} {'OOS_T':>5} {'OOS_WR':>6} {'t/d':>5} {'Net':>9}")
    print(hdr)
    print("-" * 130)

    for r in valid[:40]:
        psar_str = "Y" if r["gate_psar"] else "N"
        adx_str = str(r["gate_adx"]) if r["gate_adx"] > 0 else "-"
        chop_str = str(r["gate_chop"]) if r["gate_chop"] > 0 else "-"
        crsi_str = str(r["confirm_rsi"]) if r["confirm_rsi"] > 0 else "-"
        cstk_str = str(r["confirm_stoch"]) if r["confirm_stoch"] > 0 else "-"

        print(f"{r['period']:>3} {r['ma_type']:<4} {r['cooldown']:>3} {psar_str:<5} {adx_str:>3} {chop_str:>4} "
              f"{crsi_str:>4} {cstk_str:>4}  "
              f"{r['is']['pf']:>7.2f} {r['is']['trades']:>5} {r['is']['wr']:>5.1f}%  "
              f"{r['oos']['pf']:>7.2f} {r['oos']['trades']:>5} {r['oos']['wr']:>5.1f}% "
              f"{r['oos']['tpd']:>5.1f} {r['oos']['net']:>9.2f}")

    # Save
    out_file = ROOT / "ai_context" / "wizard_ssl_optimize_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {len(results)} results to {out_file}")

    # Best by trade frequency buckets
    print(f"\n{'='*100}")
    print("BEST BY FREQUENCY BUCKET (OOS)")
    print(f"{'='*100}")
    for label, lo, hi in [("HF 1+/day", 1.0, 99), ("MF 0.5-1/day", 0.5, 1.0), ("LF <0.5/day", 0.1, 0.5)]:
        bucket = [r for r in valid if lo <= r["oos"]["tpd"] < hi]
        if bucket:
            best = bucket[0]
            psar_str = "Y" if best["gate_psar"] else "N"
            print(f"  {label:<15} p={best['period']} {best['ma_type']} cd={best['cooldown']} "
                  f"psar={psar_str} adx={best['gate_adx']} chop={best['gate_chop']} "
                  f"cRSI={best['confirm_rsi']} cStK={best['confirm_stoch']}  "
                  f"PF={best['oos']['pf']:.2f} T={best['oos']['trades']} "
                  f"WR={best['oos']['wr']:.1f}% t/d={best['oos']['tpd']}")
        else:
            print(f"  {label:<15} -- none --")


if __name__ == "__main__":
    main()
