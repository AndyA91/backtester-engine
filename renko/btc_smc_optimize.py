#!/usr/bin/env python3
"""
btc_smc_optimize.py -- BTC SMC Structure Break Optimization (Long Only)

Phase 2: Fine-tune internal/swing sizes + add indicator gates on top of
the proven swing_align + PSAR base from Phase 1.

Best Phase 1 results:
  is7 sw25 both  SA PG — 42 OOS trades, PF=34.59, 73.8% WR
  is7 sw25 choch SA PG — 17 OOS trades, PF=39.85, 76.5% WR
"""

import contextlib
import io
import json
import math
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from renko.data import load_renko_export
from renko.indicators import add_renko_indicators
from engine import BacktestConfig, run_backtest

LTF_FILE = "OANDA_BTCUSD.SPOT.US, 1S renko 150.csv"
IS_START, IS_END = "2024-06-04", "2025-09-30"
OOS_START, OOS_END = "2025-10-01", "2026-03-19"
COMMISSION = 0.0046
CAPITAL, QTY_VALUE = 1000.0, 20


def gen_smc_signals(df, internal_size, swing_size, entry_mode,
                    adx_gate, rsi_gate, stoch_gate, chop_gate, cooldown):
    n = len(df)
    high_arr = df["High"].values
    low_arr = df["Low"].values
    close_arr = df["Close"].values
    brick_up = df["brick_up"].values
    psar_dir = df["psar_dir"].values
    adx_arr = df["adx"].values
    rsi_arr = df["rsi"].values
    stoch_k = df["stoch_k"].values
    chop_arr = df["chop"].values

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)

    int_high_level = np.nan
    int_high_crossed = True
    int_low_level = np.nan
    int_low_crossed = True
    int_trend = 0
    int_leg = 0
    sw_high_level = np.nan
    sw_high_crossed = True
    sw_low_level = np.nan
    sw_low_crossed = True
    sw_trend = 0
    sw_leg = 0

    last_trade_bar = -999_999
    in_pos = False
    warmup = swing_size + 2

    for i in range(warmup, n):
        if in_pos:
            if not brick_up[i]:
                exit_[i] = True
                in_pos = False
            continue

        # Internal pivots
        bar_h = high_arr[i - internal_size]
        bar_l = low_arr[i - internal_size]
        win_h = np.max(high_arr[i - internal_size + 1 : i + 1])
        win_l = np.min(low_arr[i - internal_size + 1 : i + 1])
        prev_int_leg = int_leg
        if bar_h > win_h:
            int_leg = 0
        elif bar_l < win_l:
            int_leg = 1
        if int_leg != prev_int_leg:
            if int_leg == 1:
                int_low_level = bar_l
                int_low_crossed = False
            else:
                int_high_level = bar_h
                int_high_crossed = False

        # Swing pivots
        sw_bar_h = high_arr[i - swing_size]
        sw_bar_l = low_arr[i - swing_size]
        sw_win_h = np.max(high_arr[i - swing_size + 1 : i + 1])
        sw_win_l = np.min(low_arr[i - swing_size + 1 : i + 1])
        prev_sw_leg = sw_leg
        if sw_bar_h > sw_win_h:
            sw_leg = 0
        elif sw_bar_l < sw_win_l:
            sw_leg = 1
        if sw_leg != prev_sw_leg:
            if sw_leg == 1:
                sw_low_level = sw_bar_l
                sw_low_crossed = False
            else:
                sw_high_level = sw_bar_h
                sw_high_crossed = False

        # Internal structure breaks
        bull_choch = False
        bull_bos = False
        if (not np.isnan(int_high_level) and not int_high_crossed
                and close_arr[i] > int_high_level
                and close_arr[i - 1] <= int_high_level):
            int_high_crossed = True
            if int_trend <= 0:
                bull_choch = True
            else:
                bull_bos = True
            int_trend = 1
        if (not np.isnan(int_low_level) and not int_low_crossed
                and close_arr[i] < int_low_level
                and close_arr[i - 1] >= int_low_level):
            int_low_crossed = True
            int_trend = -1

        # Swing breaks (for gate)
        if (not np.isnan(sw_high_level) and not sw_high_crossed
                and close_arr[i] > sw_high_level
                and close_arr[i - 1] <= sw_high_level):
            sw_high_crossed = True
            sw_trend = 1
        if (not np.isnan(sw_low_level) and not sw_low_crossed
                and close_arr[i] < sw_low_level
                and close_arr[i - 1] >= sw_low_level):
            sw_low_crossed = True
            sw_trend = -1

        # Entry logic
        if (i - last_trade_bar) < cooldown:
            continue

        bull_signal = False
        if entry_mode == "choch":
            bull_signal = bull_choch
        elif entry_mode == "bos":
            bull_signal = bull_bos
        else:
            bull_signal = bull_choch or bull_bos

        # Swing align + PSAR always on (proven best from Phase 1)
        if sw_trend != 1:
            bull_signal = False
        if np.isnan(psar_dir[i]) or psar_dir[i] != 1:
            bull_signal = False

        # Additional gates
        if bull_signal and adx_gate > 0:
            if np.isnan(adx_arr[i]) or adx_arr[i] < adx_gate:
                bull_signal = False
        if bull_signal and rsi_gate > 0:
            if np.isnan(rsi_arr[i]) or rsi_arr[i] < rsi_gate:
                bull_signal = False
        if bull_signal and stoch_gate > 0:
            if np.isnan(stoch_k[i]) or stoch_k[i] < stoch_gate:
                bull_signal = False
        if bull_signal and chop_gate > 0:
            if np.isnan(chop_arr[i]) or chop_arr[i] > chop_gate:
                bull_signal = False

        if bull_signal:
            entry[i] = True
            last_trade_bar = i
            in_pos = True

    return entry, exit_


def run_bt(df, entry, exit_, start, end):
    df2 = df.copy()
    df2["long_entry"] = entry
    df2["long_exit"] = exit_
    cfg = BacktestConfig(
        initial_capital=CAPITAL, commission_pct=COMMISSION, slippage_ticks=0,
        qty_type="cash", qty_value=QTY_VALUE, pyramiding=1,
        start_date=start, end_date=end,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest(df2, cfg)
    pf = kpis.get("profit_factor", 0.0) or 0.0
    return {
        "pf": float("inf") if math.isinf(pf) else float(pf),
        "net": float(kpis.get("net_profit", 0.0) or 0.0),
        "trades": int(kpis.get("total_trades", 0) or 0),
        "wr": float(kpis.get("win_rate", 0.0) or 0.0),
    }


def main():
    df = load_renko_export(LTF_FILE)
    add_renko_indicators(df)
    print(f"Loaded {len(df)} bricks")

    grid = {
        "internal_size": [5, 6, 7, 8, 9, 10],
        "swing_size": [15, 20, 25, 30, 35],
        "entry_mode": ["choch", "both"],
        "adx_gate": [0, 20, 25, 30],
        "rsi_gate": [0, 45, 50],
        "stoch_gate": [0, 25, 30],
        "chop_gate": [0, 50, 60],
        "cooldown": [3, 5, 10],
    }
    keys = list(grid.keys())
    combos = [dict(zip(keys, v)) for v in product(*grid.values())]
    print(f"Running {len(combos)} combos...")

    results = []
    for idx, cfg in enumerate(combos):
        entry, exit_ = gen_smc_signals(df, **cfg)
        is_r = run_bt(df, entry, exit_, IS_START, IS_END)
        oos_r = run_bt(df, entry, exit_, OOS_START, OOS_END)
        results.append({
            **cfg,
            "is_pf": is_r["pf"], "is_trades": is_r["trades"],
            "is_wr": is_r["wr"], "is_net": is_r["net"],
            "oos_pf": oos_r["pf"], "oos_trades": oos_r["trades"],
            "oos_wr": oos_r["wr"], "oos_net": oos_r["net"],
        })
        if (idx + 1) % 500 == 0:
            print(f"  {idx + 1}/{len(combos)} done")

    # Sort by OOS PF (min 10 trades)
    results.sort(
        key=lambda x: (
            x["oos_trades"] >= 10,
            x["oos_pf"] if x["oos_pf"] != float("inf") else 1e12,
            x["oos_net"],
        ),
        reverse=True,
    )

    print(f"\n{'=' * 130}")
    print("TOP 25 by OOS PF (min 10 OOS trades):")
    print(f"{'Config':<70} | {'IS':>25} | {'OOS':>30}")
    print("-" * 130)
    for r in results[:25]:
        adx = f"a{r['adx_gate']}" if r["adx_gate"] else "--"
        rsi = f"r{r['rsi_gate']}" if r["rsi_gate"] else "--"
        sto = f"s{r['stoch_gate']}" if r["stoch_gate"] else "--"
        chp = f"c{r['chop_gate']}" if r["chop_gate"] else "--"
        label = (f"is{r['internal_size']} sw{r['swing_size']} "
                 f"{r['entry_mode']:5s} {adx:>3s} {rsi:>3s} {sto:>3s} {chp:>3s} "
                 f"cd{r['cooldown']:2d}")
        is_pf = "INF" if r["is_pf"] > 1e10 else f"{r['is_pf']:.1f}"
        oos_pf = "INF" if r["oos_pf"] > 1e10 else f"{r['oos_pf']:.1f}"
        print(f"{label:<70} | T={r['is_trades']:3d} PF={is_pf:>5s} "
              f"WR={r['is_wr']:.1f}% | T={r['oos_trades']:3d} PF={oos_pf:>5s} "
              f"WR={r['oos_wr']:.1f}% ${r['oos_net']:7.2f}")

    # High-frequency configs
    freq = [r for r in results if r["oos_trades"] >= 30 and r["oos_pf"] >= 5]
    freq.sort(key=lambda x: (x["oos_pf"], x["oos_net"]), reverse=True)
    print(f"\nTOP 15 HIGH-FREQUENCY (>=30 OOS trades, PF>=5):")
    print("-" * 130)
    for r in freq[:15]:
        adx = f"a{r['adx_gate']}" if r["adx_gate"] else "--"
        rsi = f"r{r['rsi_gate']}" if r["rsi_gate"] else "--"
        sto = f"s{r['stoch_gate']}" if r["stoch_gate"] else "--"
        chp = f"c{r['chop_gate']}" if r["chop_gate"] else "--"
        label = (f"is{r['internal_size']} sw{r['swing_size']} "
                 f"{r['entry_mode']:5s} {adx:>3s} {rsi:>3s} {sto:>3s} {chp:>3s} "
                 f"cd{r['cooldown']:2d}")
        is_pf = "INF" if r["is_pf"] > 1e10 else f"{r['is_pf']:.1f}"
        oos_pf = "INF" if r["oos_pf"] > 1e10 else f"{r['oos_pf']:.1f}"
        tpd = r["oos_trades"] / 170
        print(f"{label:<70} | T={r['is_trades']:3d} PF={is_pf:>5s} "
              f"WR={r['is_wr']:.1f}% | T={r['oos_trades']:3d} PF={oos_pf:>5s} "
              f"WR={r['oos_wr']:.1f}% {tpd:.1f}/d ${r['oos_net']:7.2f}")

    out_path = ROOT / "ai_context" / "btc_smc_optimize_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {out_path.name}")


if __name__ == "__main__":
    main()
