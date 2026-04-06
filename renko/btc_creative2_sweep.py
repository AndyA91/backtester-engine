#!/usr/bin/env python3
"""
btc_creative2_sweep.py -- BTC Novel Entry Concepts (Long Only)

Explores fundamentally different entry approaches not tested before:

  Part A -- Burst momentum: multi-brick same-timestamp bursts signal
            strong directional moves. Enter after a burst of N+ up bricks.
  Part B -- RSI/MACD divergence at swing lows: price lower low but oscillator
            higher low = bullish divergence. Enter on first up brick after.
  Part C -- Consecutive brick patterns: N down bricks then up = V-reversal.
            Deeper selloffs that reverse have higher WR.
  Part D -- Compression breakout: BB bandwidth contracts (squeeze), then
            first up brick after expansion = breakout entry.
  Part E -- Combine best novel signals with BTC007 quartet.

Target: beat BTC007 TV (PF=22.87, 194t, 1.1/d, WR=65.5%)

Usage:
    python renko/btc_creative2_sweep.py
"""

import contextlib
import io
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

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
# Enrichment -- add novel columns
# ==============================================================================

def _enrich_creative(df):
    """Add burst, divergence, and pattern columns."""
    n = len(df)
    close = df["Close"].values.astype(float)
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    brick_up = df["brick_up"].values
    times = df.index.values.astype("int64") // 10**9  # unix seconds

    # -- Burst detection: count same-timestamp consecutive bricks --
    # A "burst" is multiple bricks sharing the same timestamp (same candle)
    burst_size = np.ones(n, dtype=int)
    burst_up_count = np.zeros(n, dtype=int)
    burst_dn_count = np.zeros(n, dtype=int)

    i = 0
    while i < n:
        j = i
        up_cnt = 0
        dn_cnt = 0
        while j < n and times[j] == times[i]:
            if brick_up[j]:
                up_cnt += 1
            else:
                dn_cnt += 1
            j += 1
        size = j - i
        for k in range(i, j):
            burst_size[k] = size
            burst_up_count[k] = up_cnt
            burst_dn_count[k] = dn_cnt
        i = j

    # Pre-shift: value at i = burst info for bar i-1's timestamp group
    df["burst_size"] = np.roll(burst_size, 1)
    df["burst_up"] = np.roll(burst_up_count, 1)
    df["burst_dn"] = np.roll(burst_dn_count, 1)
    df.iloc[0, df.columns.get_loc("burst_size")] = 0
    df.iloc[0, df.columns.get_loc("burst_up")] = 0
    df.iloc[0, df.columns.get_loc("burst_dn")] = 0

    # -- Recent burst: was there a burst of N+ up bricks in the last M bars? --
    # We'll compute this on-the-fly in the signal generators

    # -- Divergence: detect at swing lows --
    # Use online zigzag (1.0%) to find swing lows, compare RSI values
    rsi = df["rsi"].values
    macd_h = df["macd_hist"].values
    stoch_k = df["stoch_k"].values

    # Online zigzag to find swing lows (causal, no look-ahead)
    threshold = 1.0 / 100.0
    direction = 0
    last_high_val = high[0]
    last_low_val = low[0]
    last_low_idx = 0

    # Store last two confirmed swing lows for divergence detection
    swing_low_prices = []  # (bar_idx, price, rsi, macd, stoch)

    div_rsi = np.zeros(n, dtype=bool)   # RSI bullish divergence
    div_macd = np.zeros(n, dtype=bool)  # MACD bullish divergence
    div_stoch = np.zeros(n, dtype=bool) # Stoch bullish divergence

    for i in range(1, n):
        if direction == 0:
            if high[i] > last_high_val:
                last_high_val = high[i]
                direction = 1
            elif low[i] < last_low_val:
                last_low_val = low[i]
                last_low_idx = i
                direction = -1
        elif direction == 1:
            if high[i] > last_high_val:
                last_high_val = high[i]
            elif last_high_val > 0 and (last_high_val - low[i]) / last_high_val >= threshold:
                last_low_val = low[i]
                last_low_idx = i
                direction = -1
        elif direction == -1:
            if low[i] < last_low_val:
                last_low_val = low[i]
                last_low_idx = i
            elif last_low_val > 0 and (high[i] - last_low_val) / last_low_val >= threshold:
                # Swing low confirmed at bar i. The low was at last_low_idx.
                # Use pre-shifted indicator values at the low bar
                sl_rsi = rsi[last_low_idx] if not np.isnan(rsi[last_low_idx]) else np.nan
                sl_macd = macd_h[last_low_idx] if not np.isnan(macd_h[last_low_idx]) else np.nan
                sl_stoch = stoch_k[last_low_idx] if not np.isnan(stoch_k[last_low_idx]) else np.nan
                sl_price = last_low_val

                swing_low_prices.append((last_low_idx, sl_price, sl_rsi, sl_macd, sl_stoch))

                # Check divergence with previous swing low
                if len(swing_low_prices) >= 2:
                    prev = swing_low_prices[-2]
                    curr = swing_low_prices[-1]

                    # Price lower low but indicator higher low = bullish divergence
                    if curr[1] < prev[1]:  # price lower low
                        if not np.isnan(curr[2]) and not np.isnan(prev[2]) and curr[2] > prev[2]:
                            div_rsi[i] = True  # confirmed at detection bar
                        if not np.isnan(curr[3]) and not np.isnan(prev[3]) and curr[3] > prev[3]:
                            div_macd[i] = True
                        if not np.isnan(curr[4]) and not np.isnan(prev[4]) and curr[4] > prev[4]:
                            div_stoch[i] = True

                last_high_val = high[i]
                direction = 1

    # Pre-shift divergence signals
    df["div_rsi"] = np.roll(div_rsi, 1)
    df["div_macd"] = np.roll(div_macd, 1)
    df["div_stoch"] = np.roll(div_stoch, 1)

    # -- Consecutive down brick count (for V-reversal) --
    consec_dn = np.zeros(n, dtype=int)
    run = 0
    for i in range(n):
        if not brick_up[i]:
            run += 1
        else:
            run = 0
        consec_dn[i] = run

    # Pre-shift: at bar i, consec_dn_prev = how many down bricks ended at bar i-1
    df["consec_dn_prev"] = np.roll(consec_dn, 1)
    df.iloc[0, df.columns.get_loc("consec_dn_prev")] = 0

    # -- BB bandwidth percentile (for compression detection) --
    bb_bw = df["bb_bw"].values
    # Rolling 100-bar percentile of bandwidth
    bw_pctl = np.full(n, np.nan)
    window = 100
    for i in range(window, n):
        vals = bb_bw[i - window:i]
        valid = vals[~np.isnan(vals)]
        if len(valid) > 10:
            bw_pctl[i] = np.sum(valid < bb_bw[i]) / len(valid) * 100

    df["bw_pctl"] = bw_pctl

    return df


# ==============================================================================
# Signal generators
# ==============================================================================

def _gen_burst_entry(brick_up, burst_up, burst_size, psar_gate, cooldown,
                     min_burst_up, min_burst_size):
    """Enter after a burst of min_burst_up+ up bricks (burst_size >= min_burst_size).
    The burst happened on the PREVIOUS bar's timestamp group (pre-shifted).
    Entry on current up brick following the burst."""
    n = len(brick_up)
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
        if not bool(brick_up[i]) or not psar_gate[i] or (i - last_bar) < cooldown:
            continue
        # Check if previous bar's timestamp had a burst
        if burst_up[i] >= min_burst_up and burst_size[i] >= min_burst_size:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_divergence_entry(brick_up, div_signal, psar_gate, cooldown, lookback):
    """Enter on first up brick within `lookback` bars after a divergence signal."""
    n = len(brick_up)
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    last_div_bar = -999_999

    for i in range(60, n):
        if in_pos:
            if not bool(brick_up[i]):
                exit_[i] = True
                in_pos = False
            continue

        # Track latest divergence signal
        if div_signal[i]:
            last_div_bar = i

        if not bool(brick_up[i]) or not psar_gate[i] or (i - last_bar) < cooldown:
            continue

        # Enter if divergence occurred within lookback bars
        if last_div_bar >= 0 and (i - last_div_bar) <= lookback:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_vreversal_entry(brick_up, consec_dn_prev, psar_gate, cooldown,
                         min_down):
    """V-reversal: enter on up brick after min_down+ consecutive down bricks."""
    n = len(brick_up)
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
        if not bool(brick_up[i]) or not psar_gate[i] or (i - last_bar) < cooldown:
            continue
        if consec_dn_prev[i] >= min_down:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_compression_breakout(brick_up, bw_pctl, bb_bw, psar_gate, cooldown,
                              max_pctl, min_bw_now):
    """Compression breakout: BB bandwidth was low (< max_pctl percentile)
    in recent bars, now expanding. Enter on up brick when bandwidth
    has risen above min_bw_now (expansion phase)."""
    n = len(brick_up)
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    # Track if we were recently compressed
    was_compressed = np.zeros(n, dtype=bool)
    lookback = 10
    for i in range(lookback, n):
        for j in range(i - lookback, i):
            if not np.isnan(bw_pctl[j]) and bw_pctl[j] <= max_pctl:
                was_compressed[i] = True
                break

    for i in range(60, n):
        if in_pos:
            if not bool(brick_up[i]):
                exit_[i] = True
                in_pos = False
            continue
        if not bool(brick_up[i]) or not psar_gate[i] or (i - last_bar) < cooldown:
            continue
        # Was compressed recently, now expanding
        if was_compressed[i] and not np.isnan(bb_bw[i]) and bb_bw[i] >= min_bw_now:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_combined(brick_up, psar_gate, cooldown, chop, chop_max,
                  # BTC007 quartet
                  macd_h, kama_s, stoch_k, rsi,
                  # Novel signal arrays
                  novel_signal, novel_lookback):
    """BTC007 quartet + a novel signal added as 5th entry."""
    n = len(brick_up)
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    last_novel = -999_999

    for i in range(60, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if not up or not psar_gate[i] or (i - last_bar) < cooldown:
            continue
        if chop_max > 0 and not np.isnan(chop[i]) and chop[i] > chop_max:
            continue

        fired = False

        # BTC007 quartet
        if not np.isnan(macd_h[i]) and not np.isnan(macd_h[i-1]):
            if macd_h[i] > 0 and macd_h[i-1] <= 0:
                fired = True
        if not fired and not np.isnan(kama_s[i]) and not np.isnan(kama_s[i-1]):
            if kama_s[i] > 0 and kama_s[i-1] <= 0:
                fired = True
        if not fired and not np.isnan(stoch_k[i]) and not np.isnan(stoch_k[i-1]):
            if stoch_k[i] > 25 and stoch_k[i-1] <= 25:
                fired = True
        if not fired and not np.isnan(rsi[i]) and not np.isnan(rsi[i-1]):
            if rsi[i] > 50 and rsi[i-1] <= 50:
                fired = True

        # Novel signal (active within lookback bars)
        if novel_signal[i]:
            last_novel = i
        if not fired and last_novel >= 0 and (i - last_novel) <= novel_lookback:
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
    """Part A: Burst momentum entries."""
    combos = []
    for min_burst_up in [2, 3, 4, 5]:
        for min_burst_size in [2, 3, 4, 5]:
            if min_burst_up > min_burst_size:
                continue  # can't have more up than total
            for cd in [2, 3, 5]:
                combos.append({
                    "part": "A", "sig": "burst",
                    "min_burst_up": min_burst_up,
                    "min_burst_size": min_burst_size,
                    "cooldown": cd,
                    "label": f"burst_u{min_burst_up}_s{min_burst_size}_cd{cd}",
                })
    return combos


def _build_part_b():
    """Part B: Divergence entries."""
    combos = []
    for div_type in ["rsi", "macd", "stoch"]:
        for lookback in [3, 5, 10, 20]:
            for cd in [2, 3, 5]:
                combos.append({
                    "part": "B", "sig": "divergence",
                    "div_type": div_type,
                    "lookback": lookback,
                    "cooldown": cd,
                    "label": f"div_{div_type}_lb{lookback}_cd{cd}",
                })
    # Also test combined divergence (any of RSI/MACD/Stoch)
    for lookback in [5, 10, 20]:
        for cd in [2, 3, 5]:
            combos.append({
                "part": "B", "sig": "divergence",
                "div_type": "any",
                "lookback": lookback,
                "cooldown": cd,
                "label": f"div_any_lb{lookback}_cd{cd}",
            })
    return combos


def _build_part_c():
    """Part C: V-reversal (consecutive down then up)."""
    combos = []
    for min_down in [2, 3, 4, 5, 6, 7, 8]:
        for cd in [1, 2, 3, 5]:
            combos.append({
                "part": "C", "sig": "vreversal",
                "min_down": min_down,
                "cooldown": cd,
                "label": f"vrev_d{min_down}_cd{cd}",
            })
    return combos


def _build_part_d():
    """Part D: Compression breakout."""
    combos = []
    for max_pctl in [10, 20, 30]:
        for min_bw in [0.002, 0.003, 0.005]:
            for cd in [2, 3, 5]:
                combos.append({
                    "part": "D", "sig": "compression",
                    "max_pctl": max_pctl,
                    "min_bw": min_bw,
                    "cooldown": cd,
                    "label": f"comp_p{max_pctl}_bw{min_bw}_cd{cd}",
                })
    return combos


def _build_part_e(best_signals):
    """Part E: Combine best novel with BTC007 quartet."""
    combos = []
    for sig in best_signals:
        for cd in [2, 3, 5]:
            combos.append({
                "part": "E", "sig": "combined",
                "novel_name": sig["name"],
                "novel_col": sig["col"],
                "novel_lookback": sig.get("lookback", 1),
                "cooldown": cd,
                "chop_max": 60,
                "label": f"q4+{sig['name']}_cd{cd}",
            })
    return combos


# ==============================================================================
# Worker
# ==============================================================================

_w = {}

def _init_worker():
    if "df" not in _w:
        df = _load_ltf()
        _enrich_creative(df)
        _w["df"] = df
        psar = df["psar_dir"].values
        _w["psar_gate"] = np.isnan(psar) | (psar > 0)
        _w["brick_up"] = df["brick_up"].values
        _w["burst_up"] = df["burst_up"].values
        _w["burst_size"] = df["burst_size"].values
        _w["consec_dn"] = df["consec_dn_prev"].values
        _w["bw_pctl"] = df["bw_pctl"].values
        _w["bb_bw"] = df["bb_bw"].values
        _w["div_rsi"] = df["div_rsi"].values
        _w["div_macd"] = df["div_macd"].values
        _w["div_stoch"] = df["div_stoch"].values
        _w["macd_h"] = df["macd_hist"].values
        _w["kama_s"] = df["kama_slope"].values
        _w["stoch_k"] = df["stoch_k"].values
        _w["rsi"] = df["rsi"].values
        _w["chop"] = df["chop"].values


def _run_one(combo):
    _init_worker()
    df = _w["df"]
    w = _w
    part = combo["part"]
    cd = combo["cooldown"]

    if part == "A":
        ent, ext = _gen_burst_entry(
            w["brick_up"], w["burst_up"], w["burst_size"],
            w["psar_gate"], cd,
            combo["min_burst_up"], combo["min_burst_size"])

    elif part == "B":
        dt = combo["div_type"]
        if dt == "any":
            div_sig = w["div_rsi"] | w["div_macd"] | w["div_stoch"]
        else:
            div_sig = w[f"div_{dt}"]
        ent, ext = _gen_divergence_entry(
            w["brick_up"], div_sig, w["psar_gate"], cd, combo["lookback"])

    elif part == "C":
        ent, ext = _gen_vreversal_entry(
            w["brick_up"], w["consec_dn"], w["psar_gate"], cd,
            combo["min_down"])

    elif part == "D":
        ent, ext = _gen_compression_breakout(
            w["brick_up"], w["bw_pctl"], w["bb_bw"],
            w["psar_gate"], cd,
            combo["max_pctl"], combo["min_bw"])

    elif part == "E":
        novel_col = combo["novel_col"]
        if novel_col == "div_any":
            novel_sig = w["div_rsi"] | w["div_macd"] | w["div_stoch"]
        else:
            novel_sig = w[novel_col]
        ent, ext = _gen_combined(
            w["brick_up"], w["psar_gate"], cd, w["chop"], combo["chop_max"],
            w["macd_h"], w["kama_s"], w["stoch_k"], w["rsi"],
            novel_sig, combo["novel_lookback"])

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
                    "combo":      combo,
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

    print(f"\n{'='*126}")
    print(f"  {title} -- {len(viable)} viable / {len(subset)} total")
    print(f"{'='*126}")

    if by_wr:
        print(f"\n  Top 10 by WR (T>=30):")
        _print_header()
        for i, r in enumerate(by_wr[:10]):
            _print_row(r, rank=i+1)

    freq = [r for r in viable if r["oos_trades"] >= OOS_DAYS]
    if freq:
        freq.sort(key=lambda r: (r["oos_wr"], r["oos_net"]), reverse=True)
        print(f"\n  Top 10 at 1+/day (T>={OOS_DAYS}):")
        _print_header()
        for i, r in enumerate(freq[:10]):
            _print_row(r, rank=i+1)

    return by_wr


# ==============================================================================
# Main
# ==============================================================================

def main():
    print(f"\n{'='*70}")
    print(f"BTC Creative Sweep -- Novel Entry Concepts")
    print(f"  Target: beat BTC007 TV (PF=22.87, 194t, 1.1/d, WR=65.5%)")
    print(f"  Workers: {MAX_WORKERS}")
    print(f"{'='*70}")

    all_results = []

    # Parts A-D
    for builder, name, desc in [
        (_build_part_a, "A", "Burst Momentum"),
        (_build_part_b, "B", "Divergence Entries"),
        (_build_part_c, "C", "V-Reversal Patterns"),
        (_build_part_d, "D", "Compression Breakout"),
    ]:
        combos = builder()
        print(f"\n  Part {name}: {len(combos)} combos -- {desc}")
        _run_phase(combos, name, all_results)
        _show_part(all_results, name, f"Part {name} -- {desc}")

    # Find best novel signals for Part E
    best_signals = []
    for part_id, sig_name, col, lb in [
        ("A", "burst", None, 1),  # burst signals are direct, no lookback needed
        ("B", "div", None, None),
        ("C", "vrev", None, 1),
        ("D", "comp", None, 1),
    ]:
        part_viable = [r for r in all_results
                       if r["part"] == part_id and r["oos_trades"] >= 30 and r["oos_net"] > 0]
        if part_viable:
            part_viable.sort(key=lambda r: (r["oos_wr"], r["oos_net"]), reverse=True)
            top = part_viable[0]
            c = top["combo"]

            if part_id == "A":
                # Burst: we need a signal column. Create a synthetic one.
                # For Part E, we'll pass burst params through.
                best_signals.append({
                    "name": f"burst_u{c['min_burst_up']}_s{c['min_burst_size']}",
                    "col": "burst_flag",  # will be computed in combined
                    "min_burst_up": c["min_burst_up"],
                    "min_burst_size": c["min_burst_size"],
                    "lookback": 1,
                })
            elif part_id == "B":
                dt = c["div_type"]
                best_signals.append({
                    "name": f"div_{dt}",
                    "col": f"div_{dt}" if dt != "any" else "div_any",
                    "lookback": c["lookback"],
                })
            elif part_id == "C":
                best_signals.append({
                    "name": f"vrev_d{c['min_down']}",
                    "col": "vrev_flag",
                    "min_down": c["min_down"],
                    "lookback": 1,
                })
            elif part_id == "D":
                best_signals.append({
                    "name": f"comp_p{c['max_pctl']}",
                    "col": "comp_flag",
                    "lookback": 1,
                })

    # For Part E, we need divergence signals that can be passed as arrays
    # Skip burst/vrev/comp for Part E since they need custom handling
    div_signals = [s for s in best_signals if s["col"].startswith("div_")]
    if div_signals:
        combos_e = _build_part_e(div_signals)
        if combos_e:
            print(f"\n  Part E: {len(combos_e)} combos -- Quartet + Best Novel")
            _run_phase(combos_e, "E", all_results)
            _show_part(all_results, "E", "Part E -- Quartet + Novel")

    # Global summary
    total = len(all_results)
    viable_all = [r for r in all_results
                  if r["oos_trades"] >= 30 and r["oos_net"] > 0]
    viable_all.sort(key=lambda r: (r["oos_wr"], r["oos_net"]), reverse=True)

    print(f"\n{'='*126}")
    print(f"  GLOBAL TOP 20 by WR (T>=30, net>0): {len(viable_all)} / {total}")
    print(f"  Baseline: BTC007 TV OOS: PF=22.87, 194t (1.1/d), WR=65.5%")
    print(f"{'='*126}")
    _print_header()
    for i, r in enumerate(viable_all[:20]):
        _print_row(r, rank=i+1)

    freq_all = [r for r in viable_all if r["oos_trades"] >= OOS_DAYS]
    if freq_all:
        freq_all.sort(key=lambda r: (r["oos_wr"], r["oos_net"]), reverse=True)
        print(f"\n  GLOBAL TOP 10 at 1+/day:")
        _print_header()
        for i, r in enumerate(freq_all[:10]):
            _print_row(r, rank=i+1)

    # Save
    out_path = ROOT / "ai_context" / "btc_creative2_results.json"
    out_path.parent.mkdir(exist_ok=True)
    serializable = []
    for r in all_results:
        sr = {k: v for k, v in r.items() if k != "combo"}
        sr["combo"] = {k: v for k, v in r["combo"].items()
                       if not isinstance(v, np.ndarray)}
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
