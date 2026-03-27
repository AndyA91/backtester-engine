#!/usr/bin/env python3
"""
bc_master_sweep.py — BC/FS Indicator Gate Discovery Sweep

Tests FS Balance, BC L3 MACD Wave Signal Pro, BC L1 Multi-Oscillator
Trend Navigator, and Supertrend direction as gates stacked on top of the
R008 base (R007 + ADX(25) + vol(1.5) + session=13).

Runs all 3 instruments in parallel via ProcessPoolExecutor (one process per
instrument), then prints a ranked summary and saves results to JSON.

Why these gates (not MK or lc_state):
  MK (FS Momentum King): 100% FLAT on Renko data — brick size (~4 pips) is
    far below MK's neutral zone (52 pips). Momentum signal is always zero.
    Replaced with Supertrend direction (st_dir) — Renko-native, clean 50/50.
  bc_lc_state: Only 3.4% of bars are non-zero (requires local turning point
    in SMA fan). Replaced with continuous bc_lc > 0 direction filter.

Benchmarks to beat:
  EURUSD  OOS PF 12.79  63t  decay  15%  (R008 n=5 cd=30 adx=25 vol=1.5 ss=13)
  GBPJPY  OOS PF 21.33  92t  decay -19%  (GJ008 n=5 cd=20 adx=25 vol=1.5 ss=13)
  EURAUD  OOS PF 10.62  72t  decay  +4%  (EA008 n=5 cd=30 ss=13 vp=T div=T)

Gate configs: 15  |  Param combos: 12 (n x cd)  |  Runs per instrument: 360 IS+OOS
Total runs: ~1080 across 3 instruments (parallel -> instrument-time only)

Usage:
  python renko/bc_master_sweep.py
  python renko/bc_master_sweep.py --no-parallel   # sequential (easier to debug)
"""

import argparse
import contextlib
import io
import itertools
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from renko.config import MAX_WORKERS
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

# ── Gate configurations ─────────────────────────────────────────────────────────
#
# Base gates (fixed for all configs):
#   ADX(25) from 5m candles (where available), vol_max=1.5, session=13
#
# BC/FS gate layer:
#   st_gate:    True -> Supertrend direction (st_dir) must match trade direction
#   fsb_gate:   "off" | "any" (WEAK or STRONG) | "str" (STRONG only) — FS Balance
#   macd_gate:  True -> BC L3 MACD histogram rising in trade direction
#               (state 0 or 3 for longs; 1 or 2 for shorts)
#   lc_gate:    True -> BC L3 SMA fan (bc_lc) positive for longs, negative for shorts
#               continuous value: bc_lc > 0 = short SMAs above 40-bar center
#   motn_gate:  True -> BC L1 MOTN dx line direction matches trade direction

GATE_CONFIGS = {
    "baseline":    dict(st_gate=False, fsb_gate="off", macd_gate=False, lc_gate=False, motn_gate=False),
    "st":          dict(st_gate=True,  fsb_gate="off", macd_gate=False, lc_gate=False, motn_gate=False),
    "fsb_any":     dict(st_gate=False, fsb_gate="any", macd_gate=False, lc_gate=False, motn_gate=False),
    "fsb_strong":  dict(st_gate=False, fsb_gate="str", macd_gate=False, lc_gate=False, motn_gate=False),
    "macd_rising": dict(st_gate=False, fsb_gate="off", macd_gate=True,  lc_gate=False, motn_gate=False),
    "lc_positive": dict(st_gate=False, fsb_gate="off", macd_gate=False, lc_gate=True,  motn_gate=False),
    "macd_lc":     dict(st_gate=False, fsb_gate="off", macd_gate=True,  lc_gate=True,  motn_gate=False),
    "motn_dx":     dict(st_gate=False, fsb_gate="off", macd_gate=False, lc_gate=False, motn_gate=True),
    "st_macd":     dict(st_gate=True,  fsb_gate="off", macd_gate=True,  lc_gate=False, motn_gate=False),
    "st_fsb":      dict(st_gate=True,  fsb_gate="any", macd_gate=False, lc_gate=False, motn_gate=False),
    "fsb_macd":    dict(st_gate=False, fsb_gate="any", macd_gate=True,  lc_gate=False, motn_gate=False),
    "st_fsb_macd": dict(st_gate=True,  fsb_gate="any", macd_gate=True,  lc_gate=False, motn_gate=False),
    "motn_macd":   dict(st_gate=False, fsb_gate="off", macd_gate=True,  lc_gate=False, motn_gate=True),
    "motn_fsb":    dict(st_gate=False, fsb_gate="any", macd_gate=False, lc_gate=False, motn_gate=True),
    "st_motn":     dict(st_gate=True,  fsb_gate="off", macd_gate=False, lc_gate=False, motn_gate=True),
}

PARAM_GRID = {
    "n_bricks": [2, 3, 4, 5],
    "cooldown": [10, 20, 30],
}
# 4 x 3 = 12 param combos x 15 gates = 180 runs x IS+OOS = 360 per instrument

ADX_THRESHOLD = 25
VOL_MAX       = 1.5
SESSION_START = 13   # London + NY only (UTC >= 13)


# ── Data loading (inside worker process) ───────────────────────────────────────

def _load_renko_with_bc(root: Path, renko_file: str) -> pd.DataFrame:
    """
    Load Renko CSV, add standard renko indicators (st_dir, vol_ratio etc.),
    then compute BC/FS indicator columns and pre-shift all by 1 bar.

    Columns added (all pre-shifted):
      _fb_regime      FS Balance regime string
      _bc_macd_state  BC L3 MACD histogram state float (0/1/2/3 or NaN)
      _bc_lc          BC L3 SMA fan Line Convergence float (+ bullish, - bearish)
      _bc_motn_dx     BC L1 MOTN composite DX line
      _bc_motn_zx     BC L1 MOTN signal ZX line
    """
    sys.path.insert(0, str(root))
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    from indicators.fs_balance import calc_fs_balance
    from indicators.blackcat1402.blackcat_l3_macd_wave_signal_pro import (
        calc_bc_l3_macd_wave_signal_pro,
    )
    from indicators.blackcat1402.bc_l1_multi_oscillator_trend_navigator import (
        calc_bc_multi_oscillator_trend_navigator,
    )

    df = load_renko_export(renko_file)
    add_renko_indicators(df)   # adds st_dir, vol_ratio, kama_slope, etc.

    # FS Balance (renko-native — volume delta + tick + price imbalance)
    fb = calc_fs_balance(df)
    df["_fb_regime"] = pd.Series(fb["regime"], index=df.index).shift(1)

    # BC L3 MACD Wave Signal Pro (expects lowercase column names)
    df_lc   = df.rename(columns={"Open": "open", "High": "high",
                                  "Low": "low",  "Close": "close"})
    df_macd = calc_bc_l3_macd_wave_signal_pro(df_lc)
    df["_bc_macd_state"] = df_macd["bc_macd_state"].shift(1).values
    df["_bc_lc"]         = df_macd["bc_lc"].shift(1).values   # continuous fan value

    # BC L1 Multi-Oscillator Trend Navigator
    df_motn = calc_bc_multi_oscillator_trend_navigator(df)
    df["_bc_motn_dx"] = df_motn["bc_motn_dx"].shift(1).values
    df["_bc_motn_zx"] = df_motn["bc_motn_zx"].shift(1).values

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
    st_gate:       bool,
    fsb_gate:      str,
    macd_gate:     bool,
    lc_gate:       bool,
    motn_gate:     bool,
) -> pd.DataFrame:
    """
    R007 logic (R001 + R002 combined) + R008 base gates + optional BC/FS gates.

    R002 priority: N bricks before bar i all same dir, bar i opposes.
    R001 fallback: N bricks ending at bar i all same dir + cooldown.
    Exit: first opposing brick (R008 standard).
    Gate NaN convention: NaN -> pass (no data = don't block).
    """
    n           = len(df)
    brick_up    = df["brick_up"].values
    vol_ratio   = df["vol_ratio"].values
    hours       = df.index.hour
    st_dir      = df["st_dir"].values        # pre-shifted: +1 bullish / -1 bearish
    fb_regime   = df["_fb_regime"].values    # object array (strings / NaN)
    macd_state  = df["_bc_macd_state"].values  # float64
    bc_lc       = df["_bc_lc"].values          # float64
    motn_dx     = df["_bc_motn_dx"].values     # float64
    motn_zx     = df["_bc_motn_zx"].values     # float64

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

        # Candidate direction
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

        # BC/FS gates (NaN -> pass)

        # Supertrend direction (renko-native, pre-shifted)
        if st_gate:
            sd = st_dir[i]
            if not np.isnan(float(sd)):
                if (sd == 1) != is_long:
                    continue

        # FS Balance regime
        if fsb_gate != "off":
            reg = fb_regime[i]
            if not pd.isna(reg):
                if fsb_gate == "any":
                    ok = reg in ("STRONG_BUY", "WEAK_BUY") if is_long \
                         else reg in ("STRONG_SELL", "WEAK_SELL")
                else:
                    ok = (reg == "STRONG_BUY") if is_long else (reg == "STRONG_SELL")
                if not ok:
                    continue

        # BC L3 MACD histogram state
        # Long -> rising histogram: state 0 (above zero rising) or 3 (below zero rising)
        # Short -> falling histogram: state 1 (above zero falling) or 2 (below zero falling)
        if macd_gate:
            ms = macd_state[i]
            if not np.isnan(ms):
                ms_int = int(ms)
                ok = ms_int in (0, 3) if is_long else ms_int in (1, 2)
                if not ok:
                    continue

        # BC L3 SMA fan continuous direction
        if lc_gate:
            lc = bc_lc[i]
            if not np.isnan(lc):
                ok = (lc > 0) if is_long else (lc < 0)
                if not ok:
                    continue

        # BC L1 MOTN DX vs ZX direction
        if motn_gate:
            dx = motn_dx[i]
            zx = motn_zx[i]
            if not (np.isnan(dx) or np.isnan(zx)):
                ok = (dx > zx) if is_long else (dx < zx)
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

    print(f"[{name}] Loading Renko + BC/FS indicators...", flush=True)
    df = _load_renko_with_bc(root, config["renko_file"])

    adx_vals      = np.full(len(df), np.nan)
    adx_threshold = 0.0
    if config["adx_gate"] and config["candle_file"]:
        adx_series    = _load_candle_adx(root, config["candle_file"])
        adx_vals      = _align_adx(df, adx_series)
        adx_threshold = float(ADX_THRESHOLD)
    print(f"[{name}] Ready — {len(df)} bricks | ADX gate: {'ON' if adx_threshold > 0 else 'OFF'}", flush=True)

    keys         = list(PARAM_GRID.keys())
    param_combos = [dict(zip(keys, v)) for v in itertools.product(*PARAM_GRID.values())]
    total        = len(GATE_CONFIGS) * len(param_combos)
    done         = 0
    results      = []

    for gate_name, gate_params in GATE_CONFIGS.items():
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
            if done % 15 == 0 or done == total:
                print(
                    f"[{name}] {done:>3}/{total} | {gate_name:<13} "
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


def _summarize(all_results: list) -> None:
    for inst in ["EURUSD", "GBPJPY", "EURAUD"]:
        inst_res = [r for r in all_results if r["instrument"] == inst]
        if not inst_res:
            continue

        bench = BENCHMARKS[inst]
        print(f"\n{'='*76}")
        print(f"  {inst}  Benchmark: OOS PF {bench['oos_pf']}  [{bench['label']}]")
        print(f"{'='*76}")

        viable = [r for r in inst_res if r["oos_trades"] >= 20]
        viable.sort(key=lambda r: (r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6), reverse=True)

        print(f"  {'Gate':<14} {'n':>2} {'cd':>3} | {'IS PF':>7} {'T':>5} | {'OOS PF':>7} {'T':>5} {'WR%':>6} {'Decay':>7}")
        print(f"  {'-'*70}")
        for r in viable[:12]:
            beat  = " <<BEAT" if r["oos_pf"] > bench["oos_pf"] else ""
            dec_s = f"{r['decay_pct']:>+6.1f}%" if not math.isnan(r["decay_pct"]) else "   NaN"
            print(f"  {r['gate']:<14} {r['n_bricks']:>2} {r['cooldown']:>3} | "
                  f"{r['is_pf']:>7.2f} {r['is_trades']:>5} | "
                  f"{r['oos_pf']:>7.2f} {r['oos_trades']:>5} {r['oos_wr']:>6.1f}% "
                  f"{dec_s}{beat}")

        print(f"\n  Gate averages (viable, OOS trades >= 20):")
        print(f"  {'Gate':<14} {'Avg OOS PF':>12} {'Avg OOS T':>10} {'Avg Decay':>10} {'N':>4}")
        gate_avgs = {}
        for gate in GATE_CONFIGS:
            gv = [r for r in viable if r["gate"] == gate]
            if gv:
                avg_pf    = sum(r["oos_pf"] for r in gv) / len(gv)
                avg_t     = sum(r["oos_trades"] for r in gv) / len(gv)
                valid_dec = [r["decay_pct"] for r in gv if not math.isnan(r["decay_pct"])]
                avg_dec   = sum(valid_dec) / len(valid_dec) if valid_dec else float("nan")
                gate_avgs[gate] = (avg_pf, avg_t, avg_dec, len(gv))
        for gate, (avg_pf, avg_t, avg_dec, n) in sorted(gate_avgs.items(),
                                                          key=lambda x: x[1][0], reverse=True):
            dec_s = f"{avg_dec:>+9.1f}%" if not math.isnan(avg_dec) else "       NaN"
            beat  = " *" if avg_pf > bench["oos_pf"] else ""
            print(f"  {gate:<14} {avg_pf:>12.2f} {avg_t:>10.1f} {dec_s} {n:>4}{beat}")

    print(f"\n{'='*76}")
    print("  Cross-instrument — avg OOS PF per gate (+ = beats benchmark)")
    print(f"{'='*76}")
    print(f"  {'Gate':<14} {'EURUSD':>12} {'GBPJPY':>12} {'EURAUD':>12} {'Wins':>6}")
    print(f"  {'-'*56}")
    for gate in GATE_CONFIGS:
        wins = 0
        row  = [f"  {gate:<14}"]
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

    out_path = ROOT / "ai_context" / "bc_sweep_results.json"
    out_path.parent.mkdir(exist_ok=True)

    n_combos = len(list(itertools.product(*PARAM_GRID.values())))
    print("BC/FS Gate Discovery Sweep")
    print(f"  Gates        : {len(GATE_CONFIGS)}")
    print(f"  Param combos : {n_combos}")
    print(f"  Instruments  : {list(INSTRUMENTS.keys())}")
    print(f"  Fixed gates  : ADX={ADX_THRESHOLD}  vol_max={VOL_MAX}  session>={SESSION_START}UTC")
    print(f"  Total runs   : {len(GATE_CONFIGS) * n_combos * 2 * len(INSTRUMENTS)} IS+OOS")
    print(f"  Output       : {out_path}")
    print()

    all_results: list = []

    if args.no_parallel:
        for name, config in INSTRUMENTS.items():
            all_results.extend(run_instrument_sweep(name, config))
    else:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
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
