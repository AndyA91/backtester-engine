#!/usr/bin/env python3
"""
btc_sr_pivot_sweep.py -- BTC Support/Resistance & Pivot Sweep (Long Only)

Explores S/R-based entry concepts never tested on BTC Renko:
  Part A — Swing support bounce (zigzag-detected swing lows)
  Part B — Swing high breakout (price exceeds prior swing high)
  Part C — Daily pivot support entries (computed from Renko daily H/L/C)
  Part D — Combine best S/R signals with existing BTC007 quartet

Baseline: BTC007 v3 (macd+kama+stoch+rsi50 + PSAR+chop60, cd=3)
  TV OOS: PF=22.15, 201t (1.1/d), WR=64.7%

Usage:
    python renko/btc_sr_pivot_sweep.py
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
# S/R enrichment — add swing levels and daily pivots to the DataFrame
# ==============================================================================

def _enrich_sr(df):
    """Add swing levels and daily pivot columns to df."""
    from indicators.zigzag import calc_zigzag, calc_swing_points

    close = df["Close"].values.astype(float)
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    n = len(df)

    # -- Zigzag swing levels (multiple thresholds) --
    for pct in [0.5, 1.0, 2.0, 3.0]:
        zz = calc_zigzag(df, pct_threshold=pct)
        tag = f"zz{str(pct).replace('.', '')}"

        # Forward-fill last swing low/high prices (pre-shifted by 1)
        last_sl = np.full(n, np.nan)  # last swing low price
        last_sh = np.full(n, np.nan)  # last swing high price
        cur_sl = np.nan
        cur_sh = np.nan
        for i in range(n):
            if zz["swing_low"][i]:
                cur_sl = zz["swing_price"][i]
            if zz["swing_high"][i]:
                cur_sh = zz["swing_price"][i]
            last_sl[i] = cur_sl
            last_sh[i] = cur_sh

        # Pre-shift by 1 (value at i = swing level known through bar i-1)
        df[f"{tag}_sl"] = np.roll(last_sl, 1)
        df[f"{tag}_sl"].iloc[0] = np.nan
        df[f"{tag}_sh"] = np.roll(last_sh, 1)
        df[f"{tag}_sh"].iloc[0] = np.nan

    # -- Pivot-based swing points (left/right confirmation) --
    for lr in [3, 5, 8]:
        sp = calc_swing_points(df, left=lr, right=lr)
        tag = f"pv{lr}"

        last_pl = np.full(n, np.nan)
        last_ph = np.full(n, np.nan)
        cur_pl = np.nan
        cur_ph = np.nan
        for i in range(n):
            if sp["pivot_low"][i]:
                cur_pl = sp["pl_price"][i]
            if sp["pivot_high"][i]:
                cur_ph = sp["ph_price"][i]
            last_pl[i] = cur_pl
            last_ph[i] = cur_ph

        # Pivot is confirmed `right` bars later, so already delayed.
        # But we still need to pre-shift by 1 for our convention.
        df[f"{tag}_pl"] = np.roll(last_pl, 1)
        df[f"{tag}_pl"].iloc[0] = np.nan
        df[f"{tag}_ph"] = np.roll(last_ph, 1)
        df[f"{tag}_ph"].iloc[0] = np.nan

    # -- Daily pivots from Renko timestamps --
    # Group bricks by calendar date, compute daily H/L/C
    dates = df.index.normalize()
    unique_dates = dates.unique()

    pp_arr = np.full(n, np.nan)
    s1_arr = np.full(n, np.nan)
    s2_arr = np.full(n, np.nan)
    r1_arr = np.full(n, np.nan)
    r2_arr = np.full(n, np.nan)

    prev_h, prev_l, prev_c = np.nan, np.nan, np.nan
    for d in unique_dates:
        mask = dates == d
        day_idx = np.where(mask)[0]
        if len(day_idx) == 0:
            continue

        if not np.isnan(prev_h):
            pp = (prev_h + prev_l + prev_c) / 3.0
            r1 = 2 * pp - prev_l
            s1 = 2 * pp - prev_h
            r2 = pp + (prev_h - prev_l)
            s2 = pp - (prev_h - prev_l)

            pp_arr[day_idx] = pp
            s1_arr[day_idx] = s1
            s2_arr[day_idx] = s2
            r1_arr[day_idx] = r1
            r2_arr[day_idx] = r2

        prev_h = high[day_idx].max()
        prev_l = low[day_idx].min()
        prev_c = close[day_idx[-1]]

    df["dp_pp"] = pp_arr
    df["dp_s1"] = s1_arr
    df["dp_s2"] = s2_arr
    df["dp_r1"] = r1_arr
    df["dp_r2"] = r2_arr

    return df


# ==============================================================================
# Part A — Swing support bounce
# ==============================================================================

def _gen_swing_bounce(df, sl_col, proximity_pct, gate, cooldown):
    """Enter long when price is near a prior swing low and fires up brick.
    proximity_pct: how close price must be to swing low (e.g., 0.5 = within 0.5%)."""
    n = len(df)
    brick_up = df["brick_up"].values
    close = df["Close"].values.astype(float)
    low = df["Low"].values.astype(float)
    sl = df[sl_col].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    for i in range(60, n):
        if in_pos:
            if not bool(brick_up[i]):
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or not bool(brick_up[i]) or (i - last_bar) < cooldown:
            continue
        if np.isnan(sl[i]):
            continue
        # Check proximity: low of current or previous brick touched near swing low
        dist_pct = abs(low[i] - sl[i]) / sl[i] * 100 if sl[i] > 0 else 999
        if dist_pct <= proximity_pct:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


# ==============================================================================
# Part B — Swing high breakout
# ==============================================================================

def _gen_swing_breakout(df, sh_col, gate, cooldown):
    """Enter long when close breaks above prior swing high on up brick."""
    n = len(df)
    brick_up = df["brick_up"].values
    close = df["Close"].values.astype(float)
    sh = df[sh_col].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    for i in range(60, n):
        if in_pos:
            if not bool(brick_up[i]):
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or not bool(brick_up[i]) or (i - last_bar) < cooldown:
            continue
        if np.isnan(sh[i]):
            continue
        # Breakout: close exceeds prior swing high
        if close[i] > sh[i]:
            # Also require that previous bar was below (actual breakout, not continuation)
            if i > 0 and close[i-1] <= sh[i]:
                entry[i] = True
                in_pos = True
                last_bar = i

    return entry, exit_


# ==============================================================================
# Part C — Daily pivot support entries
# ==============================================================================

def _gen_pivot_bounce(df, level_col, proximity_pct, gate, cooldown):
    """Enter long when price bounces off a daily pivot support level."""
    n = len(df)
    brick_up = df["brick_up"].values
    low = df["Low"].values.astype(float)
    level = df[level_col].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    for i in range(60, n):
        if in_pos:
            if not bool(brick_up[i]):
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or not bool(brick_up[i]) or (i - last_bar) < cooldown:
            continue
        if np.isnan(level[i]):
            continue
        # Price touched near support and bounced (up brick)
        dist_pct = abs(low[i] - level[i]) / level[i] * 100 if level[i] > 0 else 999
        if dist_pct <= proximity_pct:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_pivot_breakout(df, level_col, gate, cooldown):
    """Enter long when close breaks above a daily pivot resistance level."""
    n = len(df)
    brick_up = df["brick_up"].values
    close = df["Close"].values.astype(float)
    level = df[level_col].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    for i in range(60, n):
        if in_pos:
            if not bool(brick_up[i]):
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or not bool(brick_up[i]) or (i - last_bar) < cooldown:
            continue
        if np.isnan(level[i]):
            continue
        if close[i] > level[i] and (i == 0 or close[i-1] <= level[i]):
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


# ==============================================================================
# Part D — Combine with existing BTC007 quartet
# ==============================================================================

def _gen_quartet_plus_sr(df, sr_type, sr_params, gate, cooldown, chop_max=60):
    """Existing MACD+KAMA+Stoch+RSI50 plus an S/R signal."""
    n = len(df)
    brick_up = df["brick_up"].values
    macd_h = df["macd_hist"].values
    kama_s = df["kama_slope"].values
    stoch_k = df["stoch_k"].values
    rsi = df["rsi"].values
    chop = df["chop"].values
    close = df["Close"].values.astype(float)
    low = df["Low"].values.astype(float)

    # S/R arrays
    if sr_type == "swing_bounce":
        sr_level = df[sr_params["col"]].values
        proximity = sr_params["proximity"]
    elif sr_type == "swing_breakout":
        sr_level = df[sr_params["col"]].values
    elif sr_type == "pivot_bounce":
        sr_level = df[sr_params["col"]].values
        proximity = sr_params["proximity"]
    elif sr_type == "pivot_breakout":
        sr_level = df[sr_params["col"]].values
    else:
        sr_level = None

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    for i in range(60, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or not up or (i - last_bar) < cooldown:
            continue
        if chop_max > 0 and not np.isnan(chop[i]) and chop[i] > chop_max:
            continue

        fired = False

        # Existing quartet
        if not fired:
            if not np.isnan(macd_h[i]) and not np.isnan(macd_h[i-1]):
                if macd_h[i] > 0 and macd_h[i-1] <= 0:
                    fired = True
        if not fired:
            if not np.isnan(kama_s[i]) and not np.isnan(kama_s[i-1]):
                if kama_s[i] > 0 and kama_s[i-1] <= 0:
                    fired = True
        if not fired:
            if not np.isnan(stoch_k[i]) and not np.isnan(stoch_k[i-1]):
                if stoch_k[i] > 25 and stoch_k[i-1] <= 25:
                    fired = True
        if not fired:
            if not np.isnan(rsi[i]) and not np.isnan(rsi[i-1]):
                if rsi[i] > 50 and rsi[i-1] <= 50:
                    fired = True

        # S/R signal
        if not fired and sr_level is not None and not np.isnan(sr_level[i]):
            if sr_type == "swing_bounce" or sr_type == "pivot_bounce":
                dist = abs(low[i] - sr_level[i]) / sr_level[i] * 100 if sr_level[i] > 0 else 999
                if dist <= proximity:
                    fired = True
            elif sr_type == "swing_breakout" or sr_type == "pivot_breakout":
                if close[i] > sr_level[i] and (i == 0 or close[i-1] <= sr_level[i]):
                    fired = True

        if fired:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


# ==============================================================================
# Combo builders
# ==============================================================================

def _build_part_a():
    """Swing support bounce: zigzag thresholds × proximity × cooldowns."""
    combos = []
    for zz_tag in ["zz05", "zz10", "zz20", "zz30"]:
        for prox in [0.3, 0.5, 1.0]:
            for cd in [2, 3, 5]:
                combos.append({
                    "part": "A",
                    "sig_type": "swing_bounce",
                    "sl_col": f"{zz_tag}_sl",
                    "proximity": prox,
                    "cooldown": cd,
                    "label": f"swBnc_{zz_tag}_p{prox}_cd{cd}",
                })
    # Also test pivot-based swing lows
    for lr in [3, 5, 8]:
        for prox in [0.3, 0.5, 1.0]:
            for cd in [3]:
                combos.append({
                    "part": "A",
                    "sig_type": "swing_bounce",
                    "sl_col": f"pv{lr}_pl",
                    "proximity": prox,
                    "cooldown": cd,
                    "label": f"swBnc_pv{lr}_p{prox}_cd{cd}",
                })
    return combos


def _build_part_b():
    """Swing high breakout: zigzag thresholds × cooldowns."""
    combos = []
    for zz_tag in ["zz05", "zz10", "zz20", "zz30"]:
        for cd in [2, 3, 5]:
            combos.append({
                "part": "B",
                "sig_type": "swing_breakout",
                "sh_col": f"{zz_tag}_sh",
                "cooldown": cd,
                "label": f"swBrk_{zz_tag}_cd{cd}",
            })
    for lr in [3, 5, 8]:
        for cd in [3]:
            combos.append({
                "part": "B",
                "sig_type": "swing_breakout",
                "sh_col": f"pv{lr}_ph",
                "cooldown": cd,
                "label": f"swBrk_pv{lr}_cd{cd}",
            })
    return combos


def _build_part_c():
    """Daily pivot entries: level × type × cooldown."""
    combos = []
    # Bounce off support levels
    for level in ["dp_s1", "dp_s2", "dp_pp"]:
        for prox in [0.3, 0.5, 1.0]:
            for cd in [2, 3]:
                combos.append({
                    "part": "C",
                    "sig_type": "pivot_bounce",
                    "level_col": level,
                    "proximity": prox,
                    "cooldown": cd,
                    "label": f"dpBnc_{level.split('_')[1]}_{prox}_cd{cd}",
                })
    # Breakout above resistance/pivot
    for level in ["dp_pp", "dp_r1"]:
        for cd in [2, 3]:
            combos.append({
                "part": "C",
                "sig_type": "pivot_breakout",
                "level_col": level,
                "cooldown": cd,
                "label": f"dpBrk_{level.split('_')[1]}_cd{cd}",
            })
    return combos


def _build_part_d(best_sr):
    """Combine best S/R signals with BTC007 quartet."""
    combos = []
    sr_signals = best_sr if best_sr else []
    for sr in sr_signals:
        for cd in [2, 3, 5]:
            combos.append({
                "part": "D",
                "sr_type": sr["type"],
                "sr_params": sr["params"],
                "cooldown": cd,
                "chop_max": 60,
                "label": f"q4+{sr['name']}_cd{cd}",
            })
    return combos


# ==============================================================================
# Worker
# ==============================================================================

_w = {}

def _init_worker():
    if "df" not in _w:
        df = _load_ltf()
        _enrich_sr(df)
        _w["df"] = df
        psar = df["psar_dir"].values
        _w["psar_gate"] = np.isnan(psar) | (psar > 0)


def _run_one(combo):
    _init_worker()
    df = _w["df"]
    gate = _w["psar_gate"]
    part = combo["part"]
    cd = combo["cooldown"]

    if part == "A":
        ent, ext = _gen_swing_bounce(df, combo["sl_col"], combo["proximity"], gate, cd)
    elif part == "B":
        ent, ext = _gen_swing_breakout(df, combo["sh_col"], gate, cd)
    elif part == "C":
        if combo["sig_type"] == "pivot_bounce":
            ent, ext = _gen_pivot_bounce(df, combo["level_col"], combo["proximity"], gate, cd)
        else:
            ent, ext = _gen_pivot_breakout(df, combo["level_col"], gate, cd)
    elif part == "D":
        ent, ext = _gen_quartet_plus_sr(
            df, combo["sr_type"], combo["sr_params"],
            gate, cd, combo.get("chop_max", 60))
    else:
        raise ValueError(f"Unknown part: {part}")

    is_r = _run_bt(df, ent, ext, IS_START, IS_END)
    oos_r = _run_bt(df, ent, ext, OOS_START, OOS_END)
    return combo, is_r, oos_r


# ==============================================================================
# Reporting
# ==============================================================================

def _print_header():
    print(f"  {'#':>3} {'Pt':>2} {'Label':<38} | "
          f"{'IS PF':>7} {'T':>5} {'WR%':>6} | "
          f"{'OOS PF':>8} {'T':>5} {'t/d':>5} {'WR%':>6} {'Net':>9} {'DD%':>7}")
    print(f"  {'-'*116}")


def _print_row(r, rank=None):
    pf_i = "INF" if math.isinf(r["is_pf"]) else f"{r['is_pf']:.2f}"
    pf_o = "INF" if math.isinf(r["oos_pf"]) else f"{r['oos_pf']:.2f}"
    tpd = r["oos_trades"] / OOS_DAYS if r["oos_trades"] > 0 else 0
    prefix = f"  {rank:>3}" if rank else "  "
    print(f"{prefix} {r['part']:>2} {r['label']:<38} | "
          f"{pf_i:>7} {r['is_trades']:>5} {r['is_wr']:>5.1f}% | "
          f"{pf_o:>8} {r['oos_trades']:>5} {tpd:>4.1f} {r['oos_wr']:>5.1f}% "
          f"{r['oos_net']:>9.2f} {r['oos_dd']:>6.2f}%")


def _run_phase(combos, phase_name, all_results):
    total = len(combos)
    print(f"\n  Running Part {phase_name}: {total} combos ({total*2} backtests)...")

    done = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run_one, c): c for c in combos}
        for fut in as_completed(futures):
            try:
                combo, is_r, oos_r = fut.result()
                row = {
                    "part":       combo["part"],
                    "label":      combo["label"],
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
                print(f"  ERROR: {combo.get('label', '???')}: {e}")
                traceback.print_exc()

            done += 1
            if done % 20 == 0 or done == total:
                print(f"    [{done:>4}/{total}]", flush=True)


def _show_part(results, part, title):
    subset = [r for r in results if r["part"] == part]
    viable = [r for r in subset if r["oos_trades"] >= 10 and r["oos_net"] > 0]

    by_wr = sorted([r for r in viable if r["oos_trades"] >= 30],
                   key=lambda r: (r["oos_wr"], r["oos_net"]), reverse=True)
    by_net = sorted([r for r in viable if r["oos_trades"] >= 30],
                    key=lambda r: r["oos_net"], reverse=True)

    print(f"\n{'='*126}")
    print(f"  {title} — {len(viable)} viable / {len(subset)} total")
    print(f"{'='*126}")

    if by_wr:
        print(f"\n  Top 10 by WR (T>=30):")
        _print_header()
        for i, r in enumerate(by_wr[:10]):
            _print_row(r, rank=i+1)

    if by_net:
        print(f"\n  Top 10 by Net (T>=30):")
        _print_header()
        for i, r in enumerate(by_net[:10]):
            _print_row(r, rank=i+1)


# ==============================================================================
# Main
# ==============================================================================

def main():
    print(f"\n{'='*70}")
    print(f"BTC S/R & Pivot Sweep")
    print(f"  Baseline: BTC007 v3 (TV OOS: PF=22.15, 201t, WR=64.7%)")
    print(f"  Workers:  {MAX_WORKERS}")
    print(f"{'='*70}")

    all_results = []

    # Part A — Swing bounce
    combos_a = _build_part_a()
    print(f"\n  Part A: {len(combos_a)} combos — Swing support bounce")
    _run_phase(combos_a, "A", all_results)
    _show_part(all_results, "A", "Part A — Swing Support Bounce")

    # Part B — Swing breakout
    combos_b = _build_part_b()
    print(f"\n  Part B: {len(combos_b)} combos — Swing high breakout")
    _run_phase(combos_b, "B", all_results)
    _show_part(all_results, "B", "Part B — Swing High Breakout")

    # Part C — Daily pivot entries
    combos_c = _build_part_c()
    print(f"\n  Part C: {len(combos_c)} combos — Daily pivot entries")
    _run_phase(combos_c, "C", all_results)
    _show_part(all_results, "C", "Part C — Daily Pivot Entries")

    # Find best standalone S/R signals for Part D
    viable_abc = [r for r in all_results
                  if r["oos_trades"] >= 30 and r["oos_net"] > 0]
    viable_abc.sort(key=lambda r: (r["oos_wr"], r["oos_net"]), reverse=True)

    best_sr = []
    # Pick top from each part
    for part_id in ["A", "B", "C"]:
        part_viable = [r for r in viable_abc if r["part"] == part_id]
        if part_viable:
            top = part_viable[0]
            label = top["label"]
            # Reconstruct params from label
            if part_id == "A":
                # e.g. "swBnc_zz10_p0.5_cd3"
                parts = label.split("_")
                col_name = parts[1] + "_sl"
                prox = float(parts[2].replace("p", ""))
                best_sr.append({"name": label.split("_cd")[0], "type": "swing_bounce",
                                "params": {"col": col_name, "proximity": prox}})
            elif part_id == "B":
                parts = label.split("_")
                col_name = parts[1] + "_sh"
                best_sr.append({"name": label.split("_cd")[0], "type": "swing_breakout",
                                "params": {"col": col_name}})
            elif part_id == "C":
                parts = label.split("_")
                if "Bnc" in label:
                    level_name = "dp_" + parts[1]
                    prox = float(parts[2])
                    best_sr.append({"name": label.split("_cd")[0], "type": "pivot_bounce",
                                    "params": {"col": level_name, "proximity": prox}})
                else:
                    level_name = "dp_" + parts[1]
                    best_sr.append({"name": label.split("_cd")[0], "type": "pivot_breakout",
                                    "params": {"col": level_name}})

    if best_sr:
        print(f"\n  Best S/R signals for Part D:")
        for sr in best_sr:
            print(f"    {sr['name']} ({sr['type']})")

    # Part D — Combine with quartet
    combos_d = _build_part_d(best_sr)
    if combos_d:
        print(f"\n  Part D: {len(combos_d)} combos — Quartet + S/R")
        _run_phase(combos_d, "D", all_results)
        _show_part(all_results, "D", "Part D — Quartet + S/R")

    # Global summary
    total = len(combos_a) + len(combos_b) + len(combos_c) + len(combos_d)
    viable_all = [r for r in all_results
                  if r["oos_trades"] >= 30 and r["oos_net"] > 0]

    print(f"\n{'='*126}")
    print(f"  GLOBAL TOP 20 by WR (T>=30, net>0): {len(viable_all)} / {total}")
    print(f"{'='*126}")
    viable_all.sort(key=lambda r: (r["oos_wr"], r["oos_net"]), reverse=True)
    _print_header()
    for i, r in enumerate(viable_all[:20]):
        _print_row(r, rank=i+1)

    # Save
    out_path = ROOT / "ai_context" / "btc_sr_pivot_results.json"
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
    print(f"Total combos: {total} ({total*2} backtests)")


if __name__ == "__main__":
    main()
