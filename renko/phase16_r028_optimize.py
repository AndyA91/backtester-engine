#!/usr/bin/env python3
"""
phase16_r028_optimize.py — R028 Band Bounce Optimization

Mean-reversion-specific gates (NOT directional trend gates which conflict
with counter-trend entries):

  1. RSI extreme confirmation: RSI < thresh for long, > (100-thresh) for short
  2. Stoch extreme confirmation: K < thresh for long, > (100-thresh) for short
  3. ADX CEILING: mean reversion fails in strong trends (ADX > max = skip)
  4. BB bandwidth minimum: need enough volatility for reversion to work
  5. MFI extreme confirmation: volume-weighted RSI at extreme
  6. MACD histogram turning: momentum exhausting = reversal more likely

Only EU4 (EURUSD 0.0004) — the validated instrument.
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
sys.path.insert(0, str(ROOT))

from renko.config import MAX_WORKERS

COMMISSION = 0.0046
CAPITAL = 1000.0
RENKO_FILE = "OANDA_EURUSD, 1S renko 0.0004.csv"
IS_START = "2023-01-23"
IS_END = "2025-09-30"
OOS_START = "2025-10-01"
OOS_END = "2026-03-19"


# ── Data loading ────────────────────────────────────────────────────────────


def _load_data():
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators

    df = load_renko_export(RENKO_FILE)
    add_renko_indicators(df)
    return df


# ── Backtest runner ─────────────────────────────────────────────────────────


def _run_bt(df, le, lx, se, sx, start, end):
    from engine import BacktestConfig, run_backtest_long_short
    df2 = df.copy()
    df2["long_entry"] = le; df2["long_exit"] = lx
    df2["short_entry"] = se; df2["short_exit"] = sx
    cfg = BacktestConfig(
        initial_capital=CAPITAL, commission_pct=COMMISSION,
        slippage_ticks=0, qty_type="fixed", qty_value=1000.0,
        pyramiding=1, start_date=start, end_date=end,
        take_profit_pct=0.0, stop_loss_pct=0.0,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest_long_short(df2, cfg)
    pf = kpis.get("profit_factor", 0.0) or 0.0
    return {
        "pf": float("inf") if math.isinf(pf) else float(pf),
        "net": float(kpis.get("net_profit", 0.0) or 0.0),
        "trades": int(kpis.get("total_trades", 0) or 0),
        "wr": float(kpis.get("win_rate", 0.0) or 0.0),
        "dd": float(kpis.get("max_drawdown_pct", 0.0) or 0.0),
    }


# ── Signal generator with MR-specific gates ────────────────────────────────


def _gen_band_bounce_opt(brick_up, bb_pct_b, rsi, stoch_k, adx, bb_bw,
                         mfi, macd_hist, vol_ratio,
                         cooldown, band_thresh, vol_max,
                         rsi_thresh, stoch_thresh, adx_max,
                         bw_min, mfi_thresh, macd_turn):
    """Band Bounce with mean-reversion-specific gates.

    Args:
        rsi_thresh: 0=off, else RSI < thresh for long, > (100-thresh) for short
        stoch_thresh: 0=off, else K < thresh for long, > (100-thresh) for short
        adx_max: 0=off, else only trade when ADX < adx_max (ranging market)
        bw_min: 0=off, else only trade when BB bandwidth >= bw_min
        mfi_thresh: 0=off, else MFI < thresh for long, > (100-thresh) for short
        macd_turn: False=off, True=require MACD hist turning toward zero
    """
    n = len(brick_up)
    long_entry = np.zeros(n, dtype=bool)
    long_exit = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit = np.zeros(n, dtype=bool)

    in_position = False
    trade_dir = 0
    last_trade_bar = -999_999
    warmup = 200

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # Exit: first opposing brick
        if in_position:
            is_opp = (trade_dir == 1 and not up) or (trade_dir == -1 and up)
            long_exit[i] = is_opp and trade_dir == 1
            short_exit[i] = is_opp and trade_dir == -1
            if is_opp:
                in_position = False
                trade_dir = 0
        if in_position:
            continue

        if (i - last_trade_bar) < cooldown:
            continue

        # Volume gate
        vr = vol_ratio[i]
        if not np.isnan(vr) and vol_max > 0 and vr > vol_max:
            continue

        pct = bb_pct_b[i]
        if np.isnan(pct):
            continue

        # Band extreme + reversal brick → candidate
        if pct <= band_thresh and up:
            cand = 1   # at lower band, brick up → MR long
        elif pct >= (1.0 - band_thresh) and not up:
            cand = -1  # at upper band, brick down → MR short
        else:
            continue

        # ── MR-specific gates ──────────────────────────────────────────

        # RSI extreme confirmation
        if rsi_thresh > 0:
            r = rsi[i]
            if np.isnan(r):
                pass  # NaN = pass
            elif cand == 1 and r >= rsi_thresh:
                continue  # long but RSI not oversold
            elif cand == -1 and r <= (100.0 - rsi_thresh):
                continue  # short but RSI not overbought

        # Stochastic extreme confirmation
        if stoch_thresh > 0:
            k = stoch_k[i]
            if np.isnan(k):
                pass
            elif cand == 1 and k >= stoch_thresh:
                continue
            elif cand == -1 and k <= (100.0 - stoch_thresh):
                continue

        # ADX ceiling (ranging market only)
        if adx_max > 0:
            a = adx[i]
            if not np.isnan(a) and a > adx_max:
                continue  # trend too strong for MR

        # BB bandwidth minimum
        if bw_min > 0:
            bw = bb_bw[i]
            if not np.isnan(bw) and bw < bw_min:
                continue  # bands too tight, no room for reversion

        # MFI extreme confirmation
        if mfi_thresh > 0:
            m = mfi[i]
            if np.isnan(m):
                pass
            elif cand == 1 and m >= mfi_thresh:
                continue  # long but MFI not oversold
            elif cand == -1 and m <= (100.0 - mfi_thresh):
                continue

        # MACD histogram turning toward zero
        if macd_turn and i >= 1:
            h_now = macd_hist[i]
            h_prev = macd_hist[i - 1]
            if not np.isnan(h_now) and not np.isnan(h_prev):
                if cand == 1 and h_now < h_prev:
                    continue  # hist still falling, momentum not exhausting
                elif cand == -1 and h_now > h_prev:
                    continue  # hist still rising

        # ── All gates passed ───────────────────────────────────────────
        if cand == 1:
            long_entry[i] = True
        else:
            short_entry[i] = True
        in_position = True
        trade_dir = cand
        last_trade_bar = i

    return long_entry, long_exit, short_entry, short_exit


# ── Build combos ────────────────────────────────────────────────────────────


def _build_combos():
    combos = []

    for (band_thresh, cooldown, vol_max,
         rsi_thresh, stoch_thresh, adx_max,
         bw_min, mfi_thresh, macd_turn) in itertools.product(
        [0.10, 0.15, 0.20, 0.25],     # band_thresh
        [2, 3, 5],                      # cooldown
        [1.5],                          # vol_max (keep fixed)
        [0, 40, 45],                    # rsi extreme: 0=off, 40=loose, 45=tight
        [0, 30, 40],                    # stoch extreme: 0=off, 30=tight, 40=loose
        [0, 25, 35],                    # adx ceiling: 0=off, 25=strict, 35=loose
        [0, 0.003],                     # bb bandwidth min: 0=off, 0.003=filter tight bands
        [0, 40],                        # mfi extreme: 0=off, 40=loose
        [False, True],                  # macd turning
    ):
        combos.append({
            "band_thresh": band_thresh,
            "cooldown": cooldown,
            "vol_max": vol_max,
            "rsi_thresh": rsi_thresh,
            "stoch_thresh": stoch_thresh,
            "adx_max": adx_max,
            "bw_min": bw_min,
            "mfi_thresh": mfi_thresh,
            "macd_turn": macd_turn,
        })

    return combos


# ── Worker ──────────────────────────────────────────────────────────────────
_w_data = {}


def _init_worker(df_bytes, arrays_bytes, is_start, is_end, oos_start, oos_end):
    import pickle
    _w_data["df"] = pd.read_pickle(io.BytesIO(df_bytes))
    _w_data["arrays"] = pickle.loads(arrays_bytes)
    _w_data["is_start"] = is_start
    _w_data["is_end"] = is_end
    _w_data["oos_start"] = oos_start
    _w_data["oos_end"] = oos_end


def _run_one_combo(combo):
    df = _w_data["df"]
    a = _w_data["arrays"]

    le, lx, se, sx = _gen_band_bounce_opt(
        a["brick_up"], a["bb_pct_b"], a["rsi"], a["stoch_k"],
        a["adx"], a["bb_bw"], a["mfi"], a["macd_hist"], a["vol_ratio"],
        combo["cooldown"], combo["band_thresh"], combo["vol_max"],
        combo["rsi_thresh"], combo["stoch_thresh"], combo["adx_max"],
        combo["bw_min"], combo["mfi_thresh"], combo["macd_turn"],
    )

    is_r = _run_bt(df, le, lx, se, sx, _w_data["is_start"], _w_data["is_end"])
    oos_r = _run_bt(df, le, lx, se, sx, _w_data["oos_start"], _w_data["oos_end"])
    return combo, is_r, oos_r


# ── Main ────────────────────────────────────────────────────────────────────


def run_sweep():
    import pickle

    combos = _build_combos()
    total = len(combos)

    print(f"\n{'='*70}")
    print(f"Phase 16 — R028 Band Bounce Optimization (MR-specific gates)")
    print(f"Instrument: EURUSD 0.0004")
    print(f"Total combos: {total} ({total*2} backtests)")
    print(f"Workers: {MAX_WORKERS}")
    print(f"{'='*70}\n")

    print("Loading data...", flush=True)
    df = _load_data()
    brick_up = df["brick_up"].values

    arrays = {
        "brick_up": brick_up,
        "bb_pct_b": df["bb_pct_b"].values,
        "rsi": df["rsi"].values,
        "stoch_k": df["stoch_k"].values,
        "adx": df["adx"].values,
        "bb_bw": df["bb_bw"].values,
        "mfi": df["mfi"].values,
        "macd_hist": df["macd_hist"].values,
        "vol_ratio": df["vol_ratio"].values,
    }

    buf = io.BytesIO()
    df.to_pickle(buf)
    df_bytes = buf.getvalue()
    arrays_bytes = pickle.dumps(arrays)

    print(f"Running {total} combos...", flush=True)
    results = []

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=_init_worker,
        initargs=(df_bytes, arrays_bytes,
                  IS_START, IS_END, OOS_START, OOS_END),
    ) as pool:
        futures = {pool.submit(_run_one_combo, c): c for c in combos}

        done = 0
        for fut in as_completed(futures):
            try:
                combo, is_r, oos_r = fut.result()
                results.append({
                    "combo": combo, "is": is_r, "oos": oos_r,
                })
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
            done += 1
            if done % 500 == 0 or done == total:
                print(f"  [{done:>5}/{total}]", flush=True)

    # ── Display ─────────────────────────────────────────────────────────────

    def _fmt(r, rank):
        c = r["combo"]
        o = r["oos"]
        pf = "INF" if math.isinf(o["pf"]) else f"{o['pf']:.2f}"
        tpd = o["trades"] / 170
        gates = []
        if c["rsi_thresh"]: gates.append(f"rsi<{c['rsi_thresh']}")
        if c["stoch_thresh"]: gates.append(f"sk<{c['stoch_thresh']}")
        if c["adx_max"]: gates.append(f"adx<{c['adx_max']}")
        if c["bw_min"]: gates.append(f"bw>{c['bw_min']}")
        if c["mfi_thresh"]: gates.append(f"mfi<{c['mfi_thresh']}")
        if c["macd_turn"]: gates.append("macd_turn")
        g_str = "+".join(gates) if gates else "none"
        return (f"  {rank:>3}. bt={c['band_thresh']:.2f} cd={c['cooldown']} "
                f"gates=[{g_str}] | "
                f"PF={pf:>7} T={o['trades']:>4} ({tpd:.1f}/d) "
                f"WR={o['wr']:>5.1f}% Net={o['net']:>8.2f} DD={o['dd']:>5.2f}%")

    # Best by net (T >= 100, ~0.6/day minimum)
    hf = [r for r in results if r["oos"]["trades"] >= 100 and r["oos"]["net"] > 0]
    hf.sort(key=lambda r: (r["oos"]["net"], r["oos"]["pf"]), reverse=True)

    print(f"\n{'='*70}")
    print(f"HIGH FREQ (T>=100, net>0): {len(hf)} configs")
    print(f"{'='*70}")
    for i, r in enumerate(hf[:25]):
        print(_fmt(r, i + 1))

    # Best by WR (T >= 80)
    hw = [r for r in results
          if r["oos"]["trades"] >= 80 and r["oos"]["net"] > 0]
    hw.sort(key=lambda r: (r["oos"]["wr"], r["oos"]["net"]), reverse=True)

    print(f"\n{'='*70}")
    print(f"HIGH WR (T>=80, net>0): {len(hw)} configs")
    print(f"{'='*70}")
    for i, r in enumerate(hw[:25]):
        print(_fmt(r, i + 1))

    # Best PF (T >= 80)
    hp = [r for r in results
          if r["oos"]["trades"] >= 80 and r["oos"]["net"] > 0]
    hp.sort(key=lambda r: (r["oos"]["pf"] if not math.isinf(r["oos"]["pf"]) else 999,
                           r["oos"]["net"]), reverse=True)

    print(f"\n{'='*70}")
    print(f"BEST PF (T>=80, net>0): {len(hp)} configs")
    print(f"{'='*70}")
    for i, r in enumerate(hp[:25]):
        print(_fmt(r, i + 1))

    # Best balance (high net AND decent WR)
    bal = [r for r in results
           if r["oos"]["trades"] >= 100 and r["oos"]["wr"] >= 53.0
           and r["oos"]["net"] > 0]
    bal.sort(key=lambda r: (r["oos"]["net"], r["oos"]["wr"]), reverse=True)

    print(f"\n{'='*70}")
    print(f"BALANCED (T>=100, WR>=53%, net>0): {len(bal)} configs")
    print(f"{'='*70}")
    for i, r in enumerate(bal[:25]):
        print(_fmt(r, i + 1))

    # ── IS/OOS consistency check ────────────────────────────────────────

    # Best configs where IS and OOS both profitable with similar PF direction
    consistent = [r for r in results
                  if r["oos"]["trades"] >= 80 and r["oos"]["net"] > 50
                  and r["is"]["net"] > 100 and r["is"]["trades"] >= 200]
    consistent.sort(key=lambda r: (r["oos"]["net"], r["oos"]["pf"]), reverse=True)

    print(f"\n{'='*70}")
    print(f"CONSISTENT IS/OOS (OOS T>=80 net>$50, IS T>=200 net>$100): {len(consistent)}")
    print(f"{'='*70}")
    for i, r in enumerate(consistent[:25]):
        c = r["combo"]
        o = r["oos"]; s = r["is"]
        pf_o = "INF" if math.isinf(o["pf"]) else f"{o['pf']:.2f}"
        pf_s = "INF" if math.isinf(s["pf"]) else f"{s['pf']:.2f}"
        gates = []
        if c["rsi_thresh"]: gates.append(f"rsi<{c['rsi_thresh']}")
        if c["stoch_thresh"]: gates.append(f"sk<{c['stoch_thresh']}")
        if c["adx_max"]: gates.append(f"adx<{c['adx_max']}")
        if c["bw_min"]: gates.append(f"bw>{c['bw_min']}")
        if c["mfi_thresh"]: gates.append(f"mfi<{c['mfi_thresh']}")
        if c["macd_turn"]: gates.append("macd_turn")
        g_str = "+".join(gates) if gates else "none"
        print(f"  {i+1:>3}. bt={c['band_thresh']:.2f} cd={c['cooldown']} "
              f"gates=[{g_str}] | "
              f"IS: PF={pf_s:>7} T={s['trades']:>4} WR={s['wr']:>5.1f}% Net={s['net']:>8.2f} | "
              f"OOS: PF={pf_o:>7} T={o['trades']:>4} WR={o['wr']:>5.1f}% Net={o['net']:>8.2f}")

    # ── Save ────────────────────────────────────────────────────────────
    out_path = ROOT / "ai_context" / "phase16_results.json"
    out_path.parent.mkdir(exist_ok=True)

    serializable = []
    for r in results:
        c = r["combo"]
        sr = {
            "combo": c,
            "is_pf": "inf" if math.isinf(r["is"]["pf"]) else r["is"]["pf"],
            "is_trades": r["is"]["trades"], "is_wr": r["is"]["wr"],
            "is_net": r["is"]["net"], "is_dd": r["is"]["dd"],
            "oos_pf": "inf" if math.isinf(r["oos"]["pf"]) else r["oos"]["pf"],
            "oos_trades": r["oos"]["trades"], "oos_wr": r["oos"]["wr"],
            "oos_net": r["oos"]["net"], "oos_dd": r["oos"]["dd"],
        }
        serializable.append(sr)

    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Total: {len(results)} runs ({len(results)*2} backtests)")

    return results


if __name__ == "__main__":
    run_sweep()
