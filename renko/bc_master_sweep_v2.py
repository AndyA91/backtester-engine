#!/usr/bin/env python3
"""
bc_master_sweep_v2.py — Extended Gate Discovery Sweep (4 New BC L1 Indicators)

Extends bc_master_sweep.py with 6 new gate configurations built from 4 untested
BC L1 indicators. Tests all 3 instruments with the same base strategy (R007 +
ADX(25) + vol(1.5) + session=13) and 12-combo n/cd parameter grid.

New indicators:
  DDL  — BC L1 Dynamic Defense Line: stochastic-based defense oscillator
          (34-bar range, continuous bc_buy_sell_diff: >0 bullish, <0 bearish)
  MCP  — BC L1 Momentum Crossover Pro: DIF-vs-WhiteOut normalized crossover
          (bc_mcp_dif > bc_mcp_white_out: bullish regime)
  STO  — BC L1 Swing Trade Oscillator: MainForce-vs-LifeLine oscillator pair
          (bc_sto_main_force > bc_sto_life_line: bullish regime)
  TSO  — BC L1 Trend Swing Oscillator: pink histogram (price > EMA10 of wt price)
          (bc_tso_pink_hist: True = bullish)

New gate configs (6):
  ddl_pos  — DDL stochastic momentum direction
  mcp_reg  — MCP DIF-vs-WhiteOut regime
  sto_reg  — STO MainForce-vs-LifeLine regime
  tso_pink — TSO pink histogram regime
  ddl_mcp  — DDL + MCP combined (both bullish/bearish)
  sto_tso  — STO + TSO combined (both bullish/bearish)

All 6 × 12 combos × 3 instruments = 216 runs.
Results compared against v1 baselines and macd_lc/fsb_strong benchmarks.

Benchmarks to beat:
  EURUSD  OOS PF 12.79  (R008 n=5 cd=30 adx=25 vol=1.5 ss=13)
  GBPJPY  OOS PF 21.33  (GJ008 n=5 cd=20 adx=25 vol=1.5 ss=13)
  EURAUD  OOS PF 10.62  (EA008 n=5 cd=30 ss=13 vp=T div=T)

V1 consensus gate averages (for comparison):
  EURUSD:  macd_lc 15.77 | fsb_strong 14.12 | baseline 11.44
  GBPJPY:  fsb_strong 30.08 | macd_lc 28.76 | baseline 14.39
  EURAUD:  baseline 4.63  (no gate beat benchmark in v1)

Usage:
  python renko/bc_master_sweep_v2.py
  python renko/bc_master_sweep_v2.py --no-parallel   # sequential (debug mode)
"""

import argparse
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

# ── Instrument configs ──────────────────────────────────────────────────────────

INSTRUMENTS = {
    "EURUSD": {
        "renko_file":  "OANDA_EURUSD, 1S renko 0.0004.csv",
        "candle_file": "HISTDATA_EURUSD_5m.csv",
        "is_start":    "2024-01-01",
        "is_end":      "2025-09-30",
        "oos_start":   "2025-10-01",
        "oos_end":     "2026-02-28",
        "commission":  0.0046,
        "capital":     1000.0,
        "adx_gate":    True,
    },
    "GBPJPY": {
        "renko_file":  "OANDA_GBPJPY, 1S renko 0.05.csv",
        "candle_file": "HISTDATA_GBPJPY_5m.csv",
        "is_start":    "2024-11-21",
        "is_end":      "2025-09-30",
        "oos_start":   "2025-10-01",
        "oos_end":     "2026-02-28",
        "commission":  0.005,
        "capital":     150_000.0,
        "adx_gate":    True,
    },
    "EURAUD": {
        "renko_file":  "OANDA_EURAUD, 1S renko 0.0006.csv",
        "candle_file": None,
        "is_start":    "2023-07-20",
        "is_end":      "2025-09-30",
        "oos_start":   "2025-10-01",
        "oos_end":     "2026-03-17",
        "commission":  0.009,
        "capital":     1000.0,
        "adx_gate":    False,
    },
}

# ── New gate configurations ─────────────────────────────────────────────────────
#
# Gate params (all boolean):
#   ddl_gate:  True -> DDL bc_buy_sell_diff > 0 (long) / < 0 (short)
#   mcp_gate:  True -> MCP bc_mcp_dif > bc_mcp_white_out (long) / < (short)
#   sto_gate:  True -> STO bc_sto_main_force > bc_sto_life_line (long) / < (short)
#   tso_gate:  True -> TSO bc_tso_pink_hist == True (long) / False (short)
#
# All gates use NaN-pass: if indicator returns NaN, gate is waived.

GATE_CONFIGS_V2 = {
    "ddl_pos":  dict(ddl_gate=True,  mcp_gate=False, sto_gate=False, tso_gate=False),
    "mcp_reg":  dict(ddl_gate=False, mcp_gate=True,  sto_gate=False, tso_gate=False),
    "sto_reg":  dict(ddl_gate=False, mcp_gate=False, sto_gate=True,  tso_gate=False),
    "tso_pink": dict(ddl_gate=False, mcp_gate=False, sto_gate=False, tso_gate=True),
    "ddl_mcp":  dict(ddl_gate=True,  mcp_gate=True,  sto_gate=False, tso_gate=False),
    "sto_tso":  dict(ddl_gate=False, mcp_gate=False, sto_gate=True,  tso_gate=True),
}

PARAM_GRID = {
    "n_bricks": [2, 3, 4, 5],
    "cooldown": [10, 20, 30],
}
# 4 × 3 = 12 param combos × 6 gates = 72 runs × IS+OOS = 144 per instrument
# Total: 144 × 3 instruments = 432 backtests (216 unique runs + IS periods)

ADX_THRESHOLD = 25
VOL_MAX       = 1.5
SESSION_START = 13


# ── Data loading ──────────────────────────────────────────────────────────────────

def _load_renko_with_new_indicators(root: Path, renko_file: str) -> pd.DataFrame:
    """
    Load Renko CSV, add standard renko indicators, then compute the 4 new BC L1
    indicator columns and pre-shift all by 1 bar.

    Columns added (all pre-shifted):
      _bc_ddl_diff   DDL bc_buy_sell_diff  (float: >0 bullish, <0 bearish)
      _bc_mcp_dif    MCP DIF line          (float, 0-100 normalized)
      _bc_mcp_wo     MCP WhiteOut line     (float, 0-100 normalized)
      _bc_sto_mf     STO MainForce         (float, 0-100)
      _bc_sto_ll     STO LifeLine          (float, 0-100)
      _bc_tso_pink   TSO pink histogram    (bool)
    """
    sys.path.insert(0, str(root))
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    from indicators.blackcat1402.bc_l1_dynamic_defense_line import (
        calc_bc_dynamic_defense_line,
    )
    from indicators.blackcat1402.bc_l1_momentum_crossover_pro import (
        calc_bc_momentum_crossover_pro,
    )
    from indicators.blackcat1402.bc_l1_swing_trade_oscillator import (
        calc_bc_swing_trade_oscillator,
    )
    from indicators.blackcat1402.bc_l1_trend_swing_oscillator import (
        calc_bc_trend_swing_oscillator,
    )

    df = load_renko_export(renko_file)
    add_renko_indicators(df)

    # DDL — uses High, Low, Close, Open (capitalized)
    ddl = calc_bc_dynamic_defense_line(df)
    df["_bc_ddl_diff"] = ddl["bc_buy_sell_diff"].shift(1).values

    # MCP — uses High, Low, Close (capitalized)
    mcp = calc_bc_momentum_crossover_pro(df)
    df["_bc_mcp_dif"] = mcp["bc_mcp_dif"].shift(1).values
    df["_bc_mcp_wo"]  = mcp["bc_mcp_white_out"].shift(1).values

    # STO — uses High, Low, Close (capitalized)
    sto = calc_bc_swing_trade_oscillator(df)
    df["_bc_sto_mf"] = sto["bc_sto_main_force"].shift(1).values
    df["_bc_sto_ll"] = sto["bc_sto_life_line"].shift(1).values

    # TSO — uses High, Low, Close, Open (capitalized)
    tso = calc_bc_trend_swing_oscillator(df)
    df["_bc_tso_pink"] = tso["bc_tso_pink_hist"].shift(1).values

    return df


def _load_candle_adx(root: Path, candle_file: str) -> pd.Series:
    """Load HISTDATA 5m candles, compute ADX(14) shifted 1 bar."""
    sys.path.insert(0, str(root))
    from indicators.adx import calc_adx

    path = root / "data" / candle_file
    df   = pd.read_csv(path)

    if df["time"].max() < 2_000_000:
        df["time"] = df["time"] * 1000

    df.index = pd.to_datetime(df["time"], unit="s")
    df       = df[["open", "high", "low", "close", "Volume"]]
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    df       = df[~df.index.duplicated(keep="first")].sort_index()

    adx_result = calc_adx(df, di_period=14, adx_period=14)
    adx        = pd.Series(adx_result["adx"], index=df.index).shift(1)
    adx.index  = adx.index.astype("datetime64[ns]")
    return adx.sort_index()


def _align_adx(df_renko: pd.DataFrame, adx_series: pd.Series) -> np.ndarray:
    """merge_asof: backward-fill candle ADX onto each Renko bar timestamp."""
    renko_times  = df_renko.index.astype("datetime64[ns]")
    adx_frame    = pd.DataFrame({"t": renko_times})
    candle_frame = adx_series.reset_index()
    candle_frame.columns = ["t_candle", "adx_val"]

    merged = pd.merge_asof(
        adx_frame.sort_values("t"),
        candle_frame,
        left_on="t", right_on="t_candle",
        direction="backward",
    ).sort_index()
    return merged["adx_val"].values


# ── Signal generator ──────────────────────────────────────────────────────────────

def _generate_signals(
    df:            pd.DataFrame,
    n_bricks:      int,
    cooldown:      int,
    adx_vals:      np.ndarray,
    adx_threshold: float,
    vol_max:       float,
    session_start: int,
    ddl_gate:      bool,
    mcp_gate:      bool,
    sto_gate:      bool,
    tso_gate:      bool,
) -> pd.DataFrame:
    """
    R007 logic (R001 + R002 combined) + R008 base gates + new BC L1 gates.

    NaN convention: NaN -> pass (no data = don't block).
    """
    n           = len(df)
    brick_up    = df["brick_up"].values
    vol_ratio   = df["vol_ratio"].values
    hours       = df.index.hour

    # New indicator arrays
    ddl_diff  = df["_bc_ddl_diff"].values
    mcp_dif   = df["_bc_mcp_dif"].values
    mcp_wo    = df["_bc_mcp_wo"].values
    sto_mf    = df["_bc_sto_mf"].values
    sto_ll    = df["_bc_sto_ll"].values
    tso_pink  = df["_bc_tso_pink"].values

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    in_position   = False
    trade_dir     = 0
    last_r001_bar = -999_999
    warmup        = max(n_bricks + 1, 200)

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # Exit: first opposing brick
        if in_position:
            is_opp        = (trade_dir == 1 and not up) or (trade_dir == -1 and up)
            long_exit[i]  = is_opp and trade_dir == 1
            short_exit[i] = is_opp and trade_dir == -1
            if is_opp:
                in_position = False
                trade_dir   = 0

        if in_position:
            continue

        # Base gates
        if adx_threshold > 0:
            av = adx_vals[i]
            if np.isnan(av) or av < adx_threshold:
                continue

        if vol_max > 0:
            vr = vol_ratio[i]
            if np.isnan(vr) or vr > vol_max:
                continue

        if session_start > 0 and hours[i] < session_start:
            continue

        # Candidate direction from R007 logic
        prev          = brick_up[i - n_bricks : i]
        prev_all_up   = bool(np.all(prev))
        prev_all_down = bool(not np.any(prev))

        if prev_all_up and not up:
            cand = -1; is_r002 = True
        elif prev_all_down and up:
            cand = 1;  is_r002 = True
        else:
            if (i - last_r001_bar) < cooldown:
                continue
            window   = brick_up[i - n_bricks + 1 : i + 1]
            all_up   = bool(np.all(window))
            all_down = bool(not np.any(window))
            if all_up:
                cand = 1;  is_r002 = False
            elif all_down:
                cand = -1; is_r002 = False
            else:
                continue

        is_long = (cand == 1)

        # ── DDL gate: bc_buy_sell_diff > 0 (long) / < 0 (short) ──────────────
        if ddl_gate:
            dv = ddl_diff[i]
            if not np.isnan(dv):
                ok = (dv > 0) if is_long else (dv < 0)
                if not ok:
                    continue

        # ── MCP gate: bc_mcp_dif > bc_mcp_white_out (long) / < (short) ───────
        if mcp_gate:
            dif_v = mcp_dif[i]
            wo_v  = mcp_wo[i]
            if not (np.isnan(dif_v) or np.isnan(wo_v)):
                ok = (dif_v > wo_v) if is_long else (dif_v < wo_v)
                if not ok:
                    continue

        # ── STO gate: main_force > life_line (long) / < (short) ──────────────
        if sto_gate:
            mf_v = sto_mf[i]
            ll_v = sto_ll[i]
            if not (np.isnan(mf_v) or np.isnan(ll_v)):
                ok = (mf_v > ll_v) if is_long else (mf_v < ll_v)
                if not ok:
                    continue

        # ── TSO gate: pink_hist True (long) / False (short) ───────────────────
        if tso_gate:
            pk = tso_pink[i]
            if not (isinstance(pk, float) and np.isnan(pk)) and pk is not None:
                ok = bool(pk) if is_long else not bool(pk)
                if not ok:
                    continue

        # Enter
        if is_long:
            long_entry[i] = True
        else:
            short_entry[i] = True
        in_position = True
        trade_dir   = cand
        if not is_r002:
            last_r001_bar = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df


# ── Backtest runner ───────────────────────────────────────────────────────────────

def _run_backtest(df_sig, start, end, commission, capital):
    sys.path.insert(0, str(ROOT))
    from engine import BacktestConfig, run_backtest_long_short

    cfg = BacktestConfig(
        initial_capital=capital,
        commission_pct=commission,
        slippage_ticks=0,
        qty_type="fixed",
        qty_value=1000.0,
        pyramiding=1,
        start_date=start,
        end_date=end,
        take_profit_pct=0.0,
        stop_loss_pct=0.0,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest_long_short(df_sig, cfg)

    pf = kpis.get("profit_factor", 0.0) or 0.0
    return {
        "pf":     float("inf") if math.isinf(pf) else float(pf),
        "net":    float(kpis.get("net_profit", 0.0) or 0.0),
        "trades": int(kpis.get("total_trades", 0) or 0),
        "wr":     float(kpis.get("win_rate", 0.0) or 0.0),
        "dd":     float(kpis.get("max_drawdown_pct", 0.0) or 0.0),
    }


# ── Worker: one instrument per process ───────────────────────────────────────────

def run_instrument_sweep(name: str, config: dict) -> list:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))

    print(f"[{name}] Loading Renko + new BC L1 indicators...", flush=True)
    df = _load_renko_with_new_indicators(root, config["renko_file"])

    adx_vals      = np.full(len(df), np.nan)
    adx_threshold = 0.0
    if config["adx_gate"] and config["candle_file"]:
        adx_series    = _load_candle_adx(root, config["candle_file"])
        adx_vals      = _align_adx(df, adx_series)
        adx_threshold = float(ADX_THRESHOLD)
    print(f"[{name}] Ready — {len(df)} bricks | ADX gate: {'ON' if adx_threshold > 0 else 'OFF'}", flush=True)

    keys         = list(PARAM_GRID.keys())
    param_combos = [dict(zip(keys, v)) for v in itertools.product(*PARAM_GRID.values())]
    total        = len(GATE_CONFIGS_V2) * len(param_combos)
    done         = 0
    results      = []

    for gate_name, gate_params in GATE_CONFIGS_V2.items():
        for pc in param_combos:
            df_sig = _generate_signals(
                df.copy(),
                n_bricks      = pc["n_bricks"],
                cooldown      = pc["cooldown"],
                adx_vals      = adx_vals,
                adx_threshold = adx_threshold,
                vol_max       = VOL_MAX,
                session_start = SESSION_START,
                **gate_params,
            )

            is_r  = _run_backtest(df_sig, config["is_start"],  config["is_end"],
                                  config["commission"], config["capital"])
            oos_r = _run_backtest(df_sig, config["oos_start"], config["oos_end"],
                                  config["commission"], config["capital"])

            is_pf  = is_r["pf"]
            oos_pf = oos_r["pf"]
            decay  = ((oos_pf - is_pf) / is_pf * 100) \
                     if is_pf > 0 and not math.isinf(is_pf) else float("nan")

            results.append({
                "instrument": name,
                "gate":       gate_name,
                "n_bricks":   pc["n_bricks"],
                "cooldown":   pc["cooldown"],
                "is_pf":      is_pf,
                "is_trades":  is_r["trades"],
                "is_net":     is_r["net"],
                "is_wr":      is_r["wr"],
                "oos_pf":     oos_pf,
                "oos_trades": oos_r["trades"],
                "oos_net":    oos_r["net"],
                "oos_wr":     oos_r["wr"],
                "decay_pct":  decay,
            })

            done += 1
            if done % 6 == 0 or done == total:
                print(
                    f"[{name}] {done:>3}/{total} | {gate_name:<10} "
                    f"n={pc['n_bricks']} cd={pc['cooldown']:>2} | "
                    f"IS PF={is_pf:>6.2f} T={is_r['trades']:>4} | "
                    f"OOS PF={oos_pf:>6.2f} T={oos_r['trades']:>4} "
                    f"decay={decay:>+6.1f}%",
                    flush=True,
                )

    print(f"[{name}] Complete — {len(results)} results", flush=True)
    return results


# ── Summary ───────────────────────────────────────────────────────────────────────

BENCHMARKS = {
    "EURUSD": {"oos_pf": 12.79, "label": "R008  n=5 cd=30 adx=25 vol=1.5 ss=13"},
    "GBPJPY": {"oos_pf": 21.33, "label": "GJ008 n=5 cd=20 adx=25 vol=1.5 ss=13"},
    "EURAUD": {"oos_pf": 10.62, "label": "EA008 n=5 cd=30 ss=13 vp=T div=T"},
}

# V1 gate averages for comparison (from bc_sweep_findings.md)
V1_GATE_AVGS = {
    "EURUSD": {"baseline": 11.44, "macd_lc": 15.77, "fsb_strong": 14.12},
    "GBPJPY": {"baseline": 14.39, "macd_lc": 28.76, "fsb_strong": 30.08},
    "EURAUD": {"baseline":  4.63, "macd_lc":  7.10, "fsb_strong":  5.55},
}


def _summarize(all_results: list) -> None:
    for inst in ["EURUSD", "GBPJPY", "EURAUD"]:
        inst_res = [r for r in all_results if r["instrument"] == inst]
        if not inst_res:
            continue

        bench = BENCHMARKS[inst]
        v1    = V1_GATE_AVGS[inst]
        print(f"\n{'='*76}")
        print(f"  {inst}  Benchmark: OOS PF {bench['oos_pf']}  [{bench['label']}]")
        print(f"  V1 reference: baseline={v1['baseline']:.2f}  "
              f"macd_lc={v1['macd_lc']:.2f}  fsb_strong={v1['fsb_strong']:.2f}")
        print(f"{'='*76}")

        viable = [r for r in inst_res if r["oos_trades"] >= 20]
        viable.sort(key=lambda r: (r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6), reverse=True)

        print(f"  {'Gate':<12} {'n':>2} {'cd':>3} | {'IS PF':>7} {'T':>5} | {'OOS PF':>7} {'T':>5} {'WR%':>6} {'Decay':>7}")
        print(f"  {'-'*70}")
        for r in viable[:12]:
            beat  = " <<BEAT" if r["oos_pf"] > bench["oos_pf"] else ""
            dec_s = f"{r['decay_pct']:>+6.1f}%" if not math.isnan(r["decay_pct"]) else "   NaN"
            print(f"  {r['gate']:<12} {r['n_bricks']:>2} {r['cooldown']:>3} | "
                  f"{r['is_pf']:>7.2f} {r['is_trades']:>5} | "
                  f"{r['oos_pf']:>7.2f} {r['oos_trades']:>5} {r['oos_wr']:>6.1f}% "
                  f"{dec_s}{beat}")

        print(f"\n  Gate averages (viable, OOS trades >= 20):")
        print(f"  {'Gate':<12} {'Avg OOS PF':>12} {'Avg OOS T':>10} {'Avg Decay':>10} {'N':>4}  vs v1_baseline  vs v1_macd_lc")
        gate_avgs = {}
        for gate in GATE_CONFIGS_V2:
            gv = [r for r in viable if r["gate"] == gate]
            if gv:
                avg_pf    = sum(r["oos_pf"] for r in gv) / len(gv)
                avg_t     = sum(r["oos_trades"] for r in gv) / len(gv)
                valid_dec = [r["decay_pct"] for r in gv if not math.isnan(r["decay_pct"])]
                avg_dec   = sum(valid_dec) / len(valid_dec) if valid_dec else float("nan")
                gate_avgs[gate] = (avg_pf, avg_t, avg_dec, len(gv))
        for gate, (avg_pf, avg_t, avg_dec, n_viable) in sorted(gate_avgs.items(),
                                                                 key=lambda x: x[1][0], reverse=True):
            dec_s    = f"{avg_dec:>+9.1f}%" if not math.isnan(avg_dec) else "       NaN"
            beat     = " *" if avg_pf > bench["oos_pf"] else ""
            vs_base  = f"{avg_pf - v1['baseline']:>+7.2f}"
            vs_macd  = f"{avg_pf - v1['macd_lc']:>+7.2f}"
            print(f"  {gate:<12} {avg_pf:>12.2f} {avg_t:>10.1f} {dec_s} {n_viable:>4}{beat}"
                  f"  {vs_base}  {vs_macd}")

    print(f"\n{'='*76}")
    print("  Cross-instrument summary (avg OOS PF per gate, viable combos)")
    print(f"{'='*76}")
    print(f"  {'Gate':<12} {'EURUSD':>12} {'GBPJPY':>12} {'EURAUD':>12} {'Wins':>6}")
    print(f"  {'-'*56}")
    for gate in GATE_CONFIGS_V2:
        wins = 0
        row  = [f"  {gate:<12}"]
        for inst in ["EURUSD", "GBPJPY", "EURAUD"]:
            gv = [r for r in all_results
                  if r["instrument"] == inst and r["gate"] == gate and r["oos_trades"] >= 20]
            if gv:
                avg_pf = sum(r["oos_pf"] for r in gv) / len(gv)
                bmark  = BENCHMARKS[inst]["oos_pf"]
                marker = "+" if avg_pf > bmark else " "
                row.append(f"{avg_pf:>11.2f}{marker}")
                if avg_pf > bmark:
                    wins += 1
            else:
                row.append(f"{'  N/A':>12}")
        row.append(f"{wins:>6}")
        print("".join(row))


# ── Main ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-parallel", action="store_true",
                        help="Run instruments sequentially (debug mode)")
    args = parser.parse_args()

    out_path = ROOT / "ai_context" / "bc_sweep_v2_results.json"
    out_path.parent.mkdir(exist_ok=True)

    n_combos = len(list(itertools.product(*PARAM_GRID.values())))
    print("BC/FS Gate Discovery Sweep v2 — New BC L1 Indicators")
    print(f"  New gates    : {list(GATE_CONFIGS_V2.keys())}")
    print(f"  Param combos : {n_combos}")
    print(f"  Instruments  : {list(INSTRUMENTS.keys())}")
    print(f"  Fixed gates  : ADX={ADX_THRESHOLD}  vol_max={VOL_MAX}  session>={SESSION_START}UTC")
    print(f"  Total runs   : {len(GATE_CONFIGS_V2) * n_combos * 2 * len(INSTRUMENTS)} IS+OOS")
    print(f"  Output       : {out_path}")
    print()

    all_results: list = []

    if args.no_parallel:
        for name, config in INSTRUMENTS.items():
            all_results.extend(run_instrument_sweep(name, config))
    else:
        with ProcessPoolExecutor(max_workers=len(INSTRUMENTS)) as pool:
            futures = {
                pool.submit(run_instrument_sweep, name, config): name
                for name, config in INSTRUMENTS.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    print(f"  [{name}] finished — {len(results)} records")
                except Exception as exc:
                    import traceback
                    print(f"  [{name}] FAILED: {exc}")
                    traceback.print_exc()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved {len(all_results)} results -> {out_path}")

    _summarize(all_results)


if __name__ == "__main__":
    main()
