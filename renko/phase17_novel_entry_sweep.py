#!/usr/bin/env python3
"""
phase17_novel_entry_sweep.py — Novel Entry Signal Discovery

Tests two structurally new entry signals that do NOT use R001/R002 brick
counting as a base:

  R026: SuperTrend Flip — enter when st_dir changes sign
  R027: Squeeze Release — enter when TTM squeeze fires (ON→OFF)

Both tested across EURUSD (4 brick sizes) + GBPJPY (2 brick sizes)
with full gate sweeps and IS/OOS splits.

Baseline comparison: R007 (R001+R002 combined) with same gates, so we
can directly measure whether the new entry signal adds value.

Usage:
  python renko/phase17_novel_entry_sweep.py
  python renko/phase17_novel_entry_sweep.py --signal st_flip
  python renko/phase17_novel_entry_sweep.py --signal sq_release
  python renko/phase17_novel_entry_sweep.py --signal baseline
"""

import argparse
import contextlib
import io
import itertools
import json
import math
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from renko.config import MAX_WORKERS

# ── Instrument configs ────────────────────────────────────────────────────────

INSTRUMENTS = {
    "EU4": {
        "renko_file": "OANDA_EURUSD, 1S renko 0.0004.csv",
        "commission": 0.0046, "capital": 1000.0, "qty": 1000.0,
        "is_start": "2023-01-23", "is_end": "2025-09-30",
        "oos_start": "2025-10-01", "oos_end": "2026-03-19",
    },
    "EU5": {
        "renko_file": "OANDA_EURUSD, 1S renko 0.0005.csv",
        "commission": 0.0046, "capital": 1000.0, "qty": 1000.0,
        "is_start": "2022-05-18", "is_end": "2025-09-30",
        "oos_start": "2025-10-01", "oos_end": "2026-03-19",
    },
    "EU6": {
        "renko_file": "OANDA_EURUSD, 1S renko 0.0006.csv",
        "commission": 0.0046, "capital": 1000.0, "qty": 1000.0,
        "is_start": None, "is_end": "2025-09-30",
        "oos_start": "2025-10-01", "oos_end": "2026-03-19",
    },
    "EU7": {
        "renko_file": "OANDA_EURUSD, 1S renko 0.0007.csv",
        "commission": 0.0046, "capital": 1000.0, "qty": 1000.0,
        "is_start": None, "is_end": "2025-09-30",
        "oos_start": "2025-10-01", "oos_end": "2026-03-19",
    },
    "GJ5": {
        "renko_file": "OANDA_GBPJPY, 1S renko 0.05.csv",
        "commission": 0.008, "capital": 1000.0, "qty": 1000.0,
        "is_start": None, "is_end": "2025-09-30",
        "oos_start": "2025-10-01", "oos_end": "2026-03-19",
    },
    "GJ10": {
        "renko_file": "OANDA_GBPJPY, 1S renko 0.1.csv",
        "commission": 0.008, "capital": 1000.0, "qty": 1000.0,
        "is_start": None, "is_end": "2025-09-30",
        "oos_start": "2025-10-01", "oos_end": "2026-03-19",
    },
}

# ── Sweep parameters ──────────────────────────────────────────────────────────

# Shared gate grid (applied to all three signal types)
GATE_GRID = {
    "adx_gate":      [0, 25],
    "session_start": [0, 13],
    "vol_max":       [0, 1.5],
    "rsi_gate":      [0, 70],
    "macd_gate":     [False, True],
}

# R026 SuperTrend Flip specific
ST_FLIP_PARAMS = {
    "cooldown":      [3, 5, 10, 20],
    "require_brick": [True, False],
    "sq_filter":     [False, True],   # only enter when NOT in squeeze
}

# R027 Squeeze Release (BB bandwidth percentile)
SQ_RELEASE_PARAMS = {
    "cooldown":        [5, 10, 20],
    "bw_percentile":   [10, 20],         # BB_BW below this pctile = squeeze
    "bw_lookback":     [100, 200],       # rolling window for percentile
    "min_squeeze_len": [3, 5],           # min consecutive low-vol bars
    "require_brick":   [True, False],
    "st_gate":         [False, True],    # require SuperTrend agreement
}

# R007 baseline (for comparison)
BASELINE_PARAMS = {
    "n_bricks": [2, 3, 4, 5],
    "cooldown": [10, 20, 30],
}


def _build_combos(signal_type):
    """Build list of param dicts for the given signal type."""
    if signal_type == "st_flip":
        specific = ST_FLIP_PARAMS
    elif signal_type == "sq_release":
        specific = SQ_RELEASE_PARAMS
    elif signal_type == "baseline":
        # Baseline uses a simpler grid (gates only, no signal-specific)
        merged = {**BASELINE_PARAMS, **GATE_GRID}
        keys = list(merged.keys())
        return [dict(zip(keys, vals)) for vals in itertools.product(*merged.values())]
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")

    merged = {**specific, **GATE_GRID}
    keys = list(merged.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*merged.values())]


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_data(renko_file):
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    df = load_renko_export(renko_file)
    add_renko_indicators(df)
    return df


# ── Signal generators ─────────────────────────────────────────────────────────

def _gen_st_flip(brick_up, st_dir, sq_on, adx, vol_ratio, rsi, macd_hist, hours,
                 cooldown, require_brick, sq_filter, adx_gate, session_start,
                 vol_max, rsi_gate, macd_gate_flag):
    """R026: SuperTrend flip entry."""
    n = len(brick_up)
    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    in_position = False
    trade_dir = 0
    last_trade_bar = -999_999
    warmup = 200

    for i in range(warmup, n):
        up = bool(brick_up[i])
        if in_position:
            is_opp = (trade_dir == 1 and not up) or (trade_dir == -1 and up)
            long_exit[i]  = is_opp and trade_dir == 1
            short_exit[i] = is_opp and trade_dir == -1
            if is_opp:
                in_position = False; trade_dir = 0
        if in_position:
            continue
        if (i - last_trade_bar) < cooldown:
            continue

        cur_st = st_dir[i]; prev_st = st_dir[i - 1]
        if np.isnan(cur_st) or np.isnan(prev_st):
            continue
        long_flip  = (prev_st < 0 and cur_st > 0)
        short_flip = (prev_st > 0 and cur_st < 0)
        if not long_flip and not short_flip:
            continue
        if require_brick:
            if long_flip and not up:
                continue
            if short_flip and up:
                continue
        is_long = long_flip

        # Gates
        if adx_gate > 0 and not np.isnan(adx[i]) and adx[i] < adx_gate:
            continue
        if session_start > 0 and hours[i] < session_start:
            continue
        if vol_max > 0 and not np.isnan(vol_ratio[i]) and vol_ratio[i] > vol_max:
            continue
        if sq_filter and not np.isnan(sq_on[i]) and bool(sq_on[i]):
            continue
        if rsi_gate > 0 and not np.isnan(rsi[i]):
            if is_long and rsi[i] > rsi_gate:
                continue
            if not is_long and rsi[i] < (100 - rsi_gate):
                continue
        if macd_gate_flag and not np.isnan(macd_hist[i]):
            if is_long and macd_hist[i] < 0:
                continue
            if not is_long and macd_hist[i] > 0:
                continue

        if is_long:
            long_entry[i] = True
        else:
            short_entry[i] = True
        in_position = True
        trade_dir = 1 if is_long else -1
        last_trade_bar = i

    return long_entry, long_exit, short_entry, short_exit


def _gen_sq_release(brick_up, bb_bw, st_dir, adx, vol_ratio, rsi,
                    macd_hist, hours, cooldown, bw_percentile, bw_lookback,
                    min_squeeze_len, require_brick, st_gate_flag, adx_gate,
                    session_start, vol_max, rsi_gate, macd_gate_flag):
    """R027: BB bandwidth squeeze release entry."""
    n = len(brick_up)
    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    # Pre-compute low-vol zone using rolling BB bandwidth percentile
    low_vol = np.zeros(n, dtype=bool)
    for i in range(bw_lookback, n):
        window = bb_bw[i - bw_lookback : i]
        valid = window[~np.isnan(window)]
        if len(valid) >= 20:
            threshold = np.percentile(valid, bw_percentile)
            if not np.isnan(bb_bw[i]):
                low_vol[i] = bb_bw[i] < threshold

    # Pre-compute consecutive low-vol length
    lv_len = np.zeros(n, dtype=int)
    for i in range(1, n):
        if low_vol[i]:
            lv_len[i] = lv_len[i - 1] + 1

    in_position = False
    trade_dir = 0
    last_trade_bar = -999_999
    warmup = max(bw_lookback + 10, 200)

    for i in range(warmup, n):
        up = bool(brick_up[i])
        if in_position:
            is_opp = (trade_dir == 1 and not up) or (trade_dir == -1 and up)
            long_exit[i]  = is_opp and trade_dir == 1
            short_exit[i] = is_opp and trade_dir == -1
            if is_opp:
                in_position = False; trade_dir = 0
        if in_position:
            continue
        if (i - last_trade_bar) < cooldown:
            continue

        # Primary trigger: exit low-vol zone
        if low_vol[i] or not low_vol[i - 1]:
            continue  # need: was in low-vol, now exited

        if lv_len[i - 1] < min_squeeze_len:
            continue

        # Direction from MACD histogram (sq_momentum broken on Renko)
        mh = macd_hist[i]
        if np.isnan(mh) or mh == 0:
            continue
        is_long = (mh > 0)

        if require_brick:
            if is_long and not up:
                continue
            if not is_long and up:
                continue

        # Gates
        if adx_gate > 0 and not np.isnan(adx[i]) and adx[i] < adx_gate:
            continue
        if session_start > 0 and hours[i] < session_start:
            continue
        if vol_max > 0 and not np.isnan(vol_ratio[i]) and vol_ratio[i] > vol_max:
            continue
        if st_gate_flag and not np.isnan(st_dir[i]):
            if is_long and st_dir[i] < 0:
                continue
            if not is_long and st_dir[i] > 0:
                continue
        if rsi_gate > 0 and not np.isnan(rsi[i]):
            if is_long and rsi[i] > rsi_gate:
                continue
            if not is_long and rsi[i] < (100 - rsi_gate):
                continue
        if macd_gate_flag and not np.isnan(macd_hist[i]):
            if is_long and macd_hist[i] < 0:
                continue
            if not is_long and macd_hist[i] > 0:
                continue

        if is_long:
            long_entry[i] = True
        else:
            short_entry[i] = True
        in_position = True
        trade_dir = 1 if is_long else -1
        last_trade_bar = i

    return long_entry, long_exit, short_entry, short_exit


def _gen_baseline(brick_up, adx, vol_ratio, rsi, macd_hist, hours,
                  n_bricks, cooldown, adx_gate, session_start, vol_max,
                  rsi_gate, macd_gate_flag):
    """R007 baseline: R001+R002 combined for comparison."""
    n = len(brick_up)
    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    in_position = False
    trade_dir = 0
    last_r001_bar = -999_999
    warmup = max(n_bricks + 1, 200)

    for i in range(warmup, n):
        up = bool(brick_up[i])
        if in_position:
            is_opp = (trade_dir == 1 and not up) or (trade_dir == -1 and up)
            long_exit[i]  = is_opp and trade_dir == 1
            short_exit[i] = is_opp and trade_dir == -1
            if is_opp:
                in_position = False; trade_dir = 0
        if in_position:
            continue

        # R002: reversal
        prev = brick_up[i - n_bricks : i]
        prev_all_up = bool(np.all(prev))
        prev_all_down = bool(not np.any(prev))

        if prev_all_up and not up:
            cand = -1; is_r002 = True
        elif prev_all_down and up:
            cand = 1; is_r002 = True
        else:
            # R001: continuation
            if (i - last_r001_bar) < cooldown:
                continue
            window = brick_up[i - n_bricks + 1 : i + 1]
            if bool(np.all(window)):
                cand = 1; is_r002 = False
            elif bool(not np.any(window)):
                cand = -1; is_r002 = False
            else:
                continue

        is_long = (cand == 1)

        # Gates
        if adx_gate > 0 and not np.isnan(adx[i]) and adx[i] < adx_gate:
            continue
        if session_start > 0 and hours[i] < session_start:
            continue
        if vol_max > 0 and not np.isnan(vol_ratio[i]) and vol_ratio[i] > vol_max:
            continue
        if rsi_gate > 0 and not np.isnan(rsi[i]):
            if is_long and rsi[i] > rsi_gate:
                continue
            if not is_long and rsi[i] < (100 - rsi_gate):
                continue
        if macd_gate_flag and not np.isnan(macd_hist[i]):
            if is_long and macd_hist[i] < 0:
                continue
            if not is_long and macd_hist[i] > 0:
                continue

        if is_long:
            long_entry[i] = True
        else:
            short_entry[i] = True
        in_position = True
        trade_dir = cand
        if not is_r002:
            last_r001_bar = i

    return long_entry, long_exit, short_entry, short_exit


# ── Backtest runner ───────────────────────────────────────────────────────────

def _run_bt(df, le, lx, se, sx, start, end, commission, capital, qty):
    from engine import BacktestConfig, run_backtest_long_short
    df2 = df.copy()
    df2["long_entry"] = le; df2["long_exit"] = lx
    df2["short_entry"] = se; df2["short_exit"] = sx
    cfg = BacktestConfig(
        initial_capital=capital, commission_pct=commission,
        slippage_ticks=0, qty_type="fixed", qty_value=qty,
        pyramiding=1, start_date=start, end_date=end,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest_long_short(df2, cfg)
    pf = kpis.get("profit_factor", 0.0) or 0.0
    return {
        "pf":     float("inf") if math.isinf(pf) else float(pf),
        "net":    float(kpis.get("net_profit", 0.0) or 0.0),
        "trades": int(kpis.get("total_trades", 0) or 0),
        "wr":     float(kpis.get("win_rate", 0.0) or 0.0),
        "dd":     float(kpis.get("max_drawdown_pct", 0.0) or 0.0),
    }


# ── Worker per instrument ────────────────────────────────────────────────────

def _run_instrument(inst_name, inst_cfg, signal_type, combos):
    """Run all combos for one instrument. Returns list of result dicts."""
    print(f"  [{inst_name}] Loading data...", flush=True)
    df = _load_data(inst_cfg["renko_file"])
    print(f"  [{inst_name}] {len(df)} bricks loaded", flush=True)

    # Extract arrays once
    _n = len(df)
    brick_up  = df["brick_up"].values
    st_dir    = df["st_dir"].values if "st_dir" in df.columns else np.ones(_n)
    bb_bw     = df["bb_bw"].values if "bb_bw" in df.columns else np.full(_n, np.nan)
    sq_on     = df["sq_on"].values if "sq_on" in df.columns else np.zeros(_n, dtype=bool)
    sq_mom    = df["sq_momentum"].values if "sq_momentum" in df.columns else np.zeros(_n)
    adx       = df["adx"].values if "adx" in df.columns else np.full(_n, np.nan)
    vol_ratio = df["vol_ratio"].values if "vol_ratio" in df.columns else np.zeros(_n)
    rsi       = df["rsi"].values if "rsi" in df.columns else np.full(_n, np.nan)
    macd_hist = df["macd_hist"].values if "macd_hist" in df.columns else np.full(_n, np.nan)
    hours     = df.index.hour

    is_start  = inst_cfg.get("is_start") or str(df.index[0].date())
    is_end    = inst_cfg["is_end"]
    oos_start = inst_cfg["oos_start"]
    oos_end   = inst_cfg["oos_end"]
    comm      = inst_cfg["commission"]
    cap       = inst_cfg["capital"]
    qty       = inst_cfg["qty"]

    results = []

    for ci, params in enumerate(combos):
        try:
            if signal_type == "st_flip":
                le, lx, se, sx = _gen_st_flip(
                    brick_up, st_dir, sq_on, adx, vol_ratio, rsi, macd_hist, hours,
                    params["cooldown"], params["require_brick"], params["sq_filter"],
                    params["adx_gate"], params["session_start"], params["vol_max"],
                    params["rsi_gate"], params["macd_gate"],
                )
            elif signal_type == "sq_release":
                le, lx, se, sx = _gen_sq_release(
                    brick_up, bb_bw, st_dir, adx, vol_ratio, rsi, macd_hist,
                    hours, params["cooldown"], params["bw_percentile"],
                    params["bw_lookback"], params["min_squeeze_len"],
                    params["require_brick"], params["st_gate"],
                    params["adx_gate"], params["session_start"], params["vol_max"],
                    params["rsi_gate"], params["macd_gate"],
                )
            elif signal_type == "baseline":
                le, lx, se, sx = _gen_baseline(
                    brick_up, adx, vol_ratio, rsi, macd_hist, hours,
                    params["n_bricks"], params["cooldown"],
                    params["adx_gate"], params["session_start"], params["vol_max"],
                    params["rsi_gate"], params["macd_gate"],
                )
            else:
                continue

            is_kpi = _run_bt(df, le, lx, se, sx, is_start, is_end, comm, cap, qty)
            oos_kpi = _run_bt(df, le, lx, se, sx, oos_start, oos_end, comm, cap, qty)

            results.append({
                "instrument": inst_name,
                "signal": signal_type,
                "params": params,
                "is": is_kpi,
                "oos": oos_kpi,
            })

        except Exception as e:
            results.append({
                "instrument": inst_name,
                "signal": signal_type,
                "params": params,
                "error": str(e),
            })

        if (ci + 1) % 50 == 0:
            print(f"  [{inst_name}] {ci+1}/{len(combos)} done", flush=True)

    print(f"  [{inst_name}] Complete — {len(results)} results", flush=True)
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def run_sweep(signal_type):
    combos = _build_combos(signal_type)
    n_inst = len(INSTRUMENTS)
    n_combos = len(combos)
    total = n_inst * n_combos

    print("=" * 100)
    print(f"  Phase 17: Novel Entry Signal Sweep — {signal_type.upper()}")
    print("=" * 100)
    print(f"  Instruments    : {n_inst} ({', '.join(INSTRUMENTS.keys())})")
    print(f"  Param combos   : {n_combos}")
    print(f"  Total backtests: {total * 2} (IS+OOS)")
    print()

    all_results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_run_instrument, name, cfg, signal_type, combos): name
            for name, cfg in INSTRUMENTS.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                all_results.extend(future.result())
            except Exception as e:
                print(f"  [{name}] FAILED: {e}")
                traceback.print_exc()

    return all_results


def _summarize(results, signal_type):
    """Print top configs per instrument."""
    print()
    print("=" * 100)
    print(f"  RESULTS SUMMARY — {signal_type.upper()}")
    print("=" * 100)

    by_inst = {}
    for r in results:
        if "error" in r:
            continue
        inst = r["instrument"]
        by_inst.setdefault(inst, []).append(r)

    for inst in sorted(by_inst.keys()):
        rows = by_inst[inst]
        # Filter: OOS trades >= 10
        valid = [r for r in rows if r["oos"]["trades"] >= 10]
        if not valid:
            print(f"\n  [{inst}] No configs with >= 10 OOS trades")
            continue

        # Sort by OOS PF
        valid.sort(key=lambda r: r["oos"]["pf"], reverse=True)

        print(f"\n  [{inst}] Top 5 by OOS PF (min 10 OOS trades):")
        print(f"  {'Rank':<5} {'OOS PF':>8} {'OOS WR':>8} {'OOS Tr':>8} {'OOS Net':>10} "
              f"{'IS PF':>8} {'IS Tr':>8} {'Key Params'}")
        print(f"  {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*40}")

        for rank, r in enumerate(valid[:5], 1):
            p = r["params"]
            # Build key params string
            if signal_type == "st_flip":
                key = f"cd={p['cooldown']} brick={p['require_brick']} sq={p['sq_filter']}"
            elif signal_type == "sq_release":
                key = f"cd={p['cooldown']} bw%={p['bw_percentile']} lb={p['bw_lookback']} sqlen={p['min_squeeze_len']} brick={p['require_brick']} st={p['st_gate']}"
            else:
                key = f"nb={p['n_bricks']} cd={p['cooldown']}"
            gates = []
            if p.get("adx_gate", 0) > 0:
                gates.append(f"adx={p['adx_gate']}")
            if p.get("session_start", 0) > 0:
                gates.append(f"sess={p['session_start']}")
            if p.get("vol_max", 0) > 0:
                gates.append(f"vol={p['vol_max']}")
            if p.get("rsi_gate", 0) > 0:
                gates.append(f"rsi={p['rsi_gate']}")
            if p.get("macd_gate", False):
                gates.append("macd")
            if gates:
                key += " | " + " ".join(gates)

            oos = r["oos"]; is_ = r["is"]
            print(f"  {rank:<5} {oos['pf']:>8.2f} {oos['wr']:>7.1f}% {oos['trades']:>8d} "
                  f"{oos['net']:>10.2f} {is_['pf']:>8.2f} {is_['trades']:>8d} {key}")

    # Cross-instrument summary: median OOS PF of best config per instrument
    print(f"\n  {'─'*80}")
    print(f"  Cross-Instrument Best OOS PF:")
    for inst in sorted(by_inst.keys()):
        valid = [r for r in by_inst[inst] if r["oos"]["trades"] >= 10]
        if valid:
            best = max(valid, key=lambda r: r["oos"]["pf"])
            print(f"    {inst}: PF={best['oos']['pf']:.2f}  WR={best['oos']['wr']:.1f}%  "
                  f"Trades={best['oos']['trades']}  Net={best['oos']['net']:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal", choices=["st_flip", "sq_release", "baseline", "all"],
                        default="all")
    args = parser.parse_args()

    signals = ["st_flip", "sq_release", "baseline"] if args.signal == "all" else [args.signal]

    all_results = {}
    for sig in signals:
        results = run_sweep(sig)
        all_results[sig] = results
        _summarize(results, sig)

    # Save results
    out_path = ROOT / "ai_context" / "phase17_novel_entry_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    # Final comparison across signal types
    if len(signals) > 1:
        print("\n" + "=" * 100)
        print("  CROSS-SIGNAL COMPARISON")
        print("=" * 100)
        for sig in signals:
            results = all_results[sig]
            valid = [r for r in results if "error" not in r and r["oos"]["trades"] >= 10]
            if not valid:
                print(f"  {sig}: No valid results")
                continue
            oos_pfs = [r["oos"]["pf"] for r in valid]
            oos_wrs = [r["oos"]["wr"] for r in valid]
            oos_trades = [r["oos"]["trades"] for r in valid]
            print(f"\n  {sig.upper()} ({len(valid)} valid configs):")
            print(f"    OOS PF:     median={np.median(oos_pfs):.2f}  "
                  f"mean={np.mean(oos_pfs):.2f}  max={np.max(oos_pfs):.2f}")
            print(f"    OOS WR:     median={np.median(oos_wrs):.1f}%  "
                  f"mean={np.mean(oos_wrs):.1f}%  max={np.max(oos_wrs):.1f}%")
            print(f"    OOS Trades: median={np.median(oos_trades):.0f}  "
                  f"mean={np.mean(oos_trades):.0f}  max={np.max(oos_trades):.0f}")


if __name__ == "__main__":
    main()
