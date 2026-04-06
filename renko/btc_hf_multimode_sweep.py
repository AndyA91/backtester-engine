#!/usr/bin/env python3
"""
btc_hf_multimode_sweep.py -- BTC HF Phase 3: Multi-Mode Entry System (Long Only)

Captures three distinct market behaviors with PER-MODE cooldowns:

  TREND (momentum/breakout):
    - R001: 2 consecutive up bricks
    - ST_FLIP: Supertrend flips bullish on up brick
    - MACD_X: MACD histogram crosses positive on up brick

  PULLBACK (dip-buy in uptrends):
    - EMA_DIP: In uptrend (EMA9 > EMA50), close dipped to EMA21 zone, now up brick
    - SHORT_RETRACE: After 3+ up bricks, exactly 1-2 down bricks, then up
    - RSI_RECOVERY: RSI was <45, now >=45, close > EMA50, up brick

  MEAN REVERSION (oversold bounce):
    - BB_BOUNCE: BB %B <= threshold on up brick
    - STOCH_OV: Stoch K crosses above threshold from below on up brick
    - RSI_OV: RSI < threshold on up brick (deep oversold)

Each mode has its own cooldown tracker — trend entry doesn't block pullback entry.
Only one position at a time (single long). Exit: first down brick.

Sweep dimensions:
  Mode combos: T, P, M, T+P, T+M, P+M, T+P+M = 7
  Gate: none, psar = 2
  Shared cd: 3, 5, 8 = 3
  Per-mode cd: (cd_t=3, cd_p=5, cd_mr=3) and (cd_t=5, cd_p=3, cd_mr=3) = 2
  MR bb threshold: 0.20, 0.25, 0.30 = 3
  Total: 7 modes × 2 gates × (3 shared + 2 per-mode) × 3 mr_thresh = 210 combos

Usage:
    python renko/btc_hf_multimode_sweep.py
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

# -- Instrument config ---------------------------------------------------------

LTF_FILE   = "OANDA_BTCUSD.SPOT.US, 1S renko 150.csv"
IS_START   = "2024-06-04"
IS_END     = "2025-09-30"
OOS_START  = "2025-10-01"
OOS_END    = "2026-03-19"
COMMISSION = 0.0046
CAPITAL    = 1000.0
QTY_VALUE  = 20

# -- Data loading ---------------------------------------------------------------

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
    if "adx20" in gate_mode:
        adx = df["adx"].values
        gate &= (np.isnan(adx) | (adx >= 20))
    return gate

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


# -- Multi-mode signal generator ------------------------------------------------

def _gen_multimode(df, gate, modes, cd_trend, cd_pullback, cd_mr, mr_bb_thresh):
    """
    Generate combined entry signals from multiple modes.
    Each mode has its own cooldown tracker.

    modes: set of "trend", "pullback", "mr"
    """
    n = len(df)
    brick_up = df["brick_up"].values
    close = df["Close"].values.astype(float)

    # Indicator arrays (all pre-shifted)
    st_dir    = df["st_dir"].values
    macd_h    = df["macd_hist"].values
    kama_s    = df["kama_slope"].values
    bb_pct    = df["bb_pct_b"].values
    rsi       = df["rsi"].values
    stoch_k   = df["stoch_k"].values
    ema9      = df["ema9"].values
    ema21     = df["ema21"].values
    ema50     = df["ema50"].values
    psar_dir  = df["psar_dir"].values

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False

    # Per-mode cooldown trackers
    last_trend_bar = -999_999
    last_pullback_bar = -999_999
    last_mr_bar = -999_999

    warmup = 50

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # Exit: first down brick
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue

        # Global gate
        if not gate[i] or not up:
            continue

        fired = False

        # ── TREND MODE ────────────────────────────────────────────────────
        if "trend" in modes and (i - last_trend_bar) >= cd_trend:
            # R001: 2 consecutive up bricks
            if brick_up[i - 1]:
                fired = True
                last_trend_bar = i

            # ST flip: supertrend flips bullish
            if not fired:
                if not np.isnan(st_dir[i]) and not np.isnan(st_dir[i-1]):
                    if st_dir[i] > 0 and st_dir[i-1] <= 0:
                        fired = True
                        last_trend_bar = i

            # MACD histogram crosses positive
            if not fired:
                if not np.isnan(macd_h[i]) and not np.isnan(macd_h[i-1]):
                    if macd_h[i] > 0 and macd_h[i-1] <= 0:
                        fired = True
                        last_trend_bar = i

        # ── PULLBACK MODE ─────────────────────────────────────────────────
        if not fired and "pullback" in modes and (i - last_pullback_bar) >= cd_pullback:
            # EMA dip: uptrend (EMA9 > EMA50), close was near/below EMA21, now up
            if not np.isnan(ema9[i]) and not np.isnan(ema21[i]) and not np.isnan(ema50[i]):
                in_uptrend = ema9[i] > ema50[i]
                # Previous close was at or below EMA21 (within 1 brick)
                if in_uptrend and i >= 2:
                    prev_close = close[i - 1]
                    ema21_val = ema21[i]
                    # Dipped to EMA21 zone: prev close within 1 brick of EMA21 or below
                    brick_size = abs(close[i] - close[i-1]) if close[i] != close[i-1] else 150.0
                    if prev_close <= ema21_val + brick_size * 0.5:
                        fired = True
                        last_pullback_bar = i

            # Short retrace: 3+ up bricks, then 1-2 down, then up
            if not fired and i >= 6:
                # Current is up. Check 1-2 bricks before are down.
                if not brick_up[i-1]:  # prev was down
                    # Count consecutive down bricks before current
                    down_count = 0
                    j = i - 1
                    while j >= i - 2 and not brick_up[j]:
                        down_count += 1
                        j -= 1
                    # Before the down bricks, need 3+ up bricks
                    if 1 <= down_count <= 2:
                        up_start = i - 1 - down_count
                        up_count = 0
                        for k in range(up_start, max(up_start - 5, warmup - 1), -1):
                            if brick_up[k]:
                                up_count += 1
                            else:
                                break
                        if up_count >= 3:
                            fired = True
                            last_pullback_bar = i

            # RSI recovery: RSI was <45, now >=45, close > EMA50
            if not fired:
                if not np.isnan(rsi[i]) and not np.isnan(rsi[i-1]) and not np.isnan(ema50[i]):
                    if rsi[i] >= 45 and rsi[i-1] < 45 and close[i] > ema50[i]:
                        fired = True
                        last_pullback_bar = i

        # ── MEAN REVERSION MODE ───────────────────────────────────────────
        if not fired and "mr" in modes and (i - last_mr_bar) >= cd_mr:
            # BB bounce: %B at lower extreme
            if not np.isnan(bb_pct[i]):
                if bb_pct[i] <= mr_bb_thresh:
                    fired = True
                    last_mr_bar = i

            # Stoch oversold cross: K crosses above 25 from below
            if not fired:
                if not np.isnan(stoch_k[i]) and not np.isnan(stoch_k[i-1]):
                    if stoch_k[i] > 25 and stoch_k[i-1] <= 25:
                        fired = True
                        last_mr_bar = i

            # RSI deep oversold: RSI < 30
            if not fired:
                if not np.isnan(rsi[i]):
                    if rsi[i] < 30:
                        fired = True
                        last_mr_bar = i

        if fired:
            entry[i] = True
            in_pos = True

    return entry, exit_


# -- Combo builder --------------------------------------------------------------

MODE_COMBOS = {
    "T":     {"trend"},
    "P":     {"pullback"},
    "M":     {"mr"},
    "T+P":   {"trend", "pullback"},
    "T+M":   {"trend", "mr"},
    "P+M":   {"pullback", "mr"},
    "T+P+M": {"trend", "pullback", "mr"},
}

GATE_MODES = ["none", "psar"]
MR_BB_THRESHOLDS = [0.20, 0.25, 0.30]

# Cooldown configs: (cd_trend, cd_pullback, cd_mr, label)
CD_CONFIGS = [
    # Shared cooldowns
    (3, 3, 3, "shared_3"),
    (5, 5, 5, "shared_5"),
    (8, 8, 8, "shared_8"),
    # Per-mode (trend fast, pullback slower)
    (3, 5, 3, "permode_3_5_3"),
    (3, 8, 3, "permode_3_8_3"),
    # Per-mode (all fast)
    (3, 3, 5, "permode_3_3_5"),
]


def _build_combos():
    combos = []
    for mode_name, modes in MODE_COMBOS.items():
        for gm in GATE_MODES:
            for cd_t, cd_p, cd_mr, cd_label in CD_CONFIGS:
                for mr_t in MR_BB_THRESHOLDS:
                    combos.append({
                        "mode_name": mode_name,
                        "modes": modes,
                        "gate_mode": gm,
                        "cd_trend": cd_t,
                        "cd_pullback": cd_p,
                        "cd_mr": cd_mr,
                        "cd_label": cd_label,
                        "mr_bb_thresh": mr_t,
                    })
    return combos


# -- Worker ---------------------------------------------------------------------

_w = {}

def _run_one(combo):
    if "df" not in _w:
        _w["df"] = _load_data()
        _w["gates"] = {gm: _compute_gate(_w["df"], gm) for gm in GATE_MODES}

    df = _w["df"]
    gate = _w["gates"][combo["gate_mode"]]

    entry, exit_ = _gen_multimode(
        df, gate, combo["modes"],
        combo["cd_trend"], combo["cd_pullback"], combo["cd_mr"],
        combo["mr_bb_thresh"],
    )

    is_r = _run_bt(df, entry, exit_, IS_START, IS_END)
    oos_r = _run_bt(df, entry, exit_, OOS_START, OOS_END)

    return combo, is_r, oos_r


# -- Summary -------------------------------------------------------------------

def _summarize(all_results):
    oos_days = 170

    print(f"\n{'='*130}")
    print("  BTC HF Phase 3 — Multi-Mode Entry System (Long Only)")
    print(f"  TREND (R001+ST_flip+MACD_x) | PULLBACK (EMA_dip+retrace+RSI_recovery) | MR (BB_bounce+Stoch_ov+RSI_ov)")
    print(f"  Per-mode cooldowns | Target: 1+ trade/day")
    print(f"{'='*130}")

    # -- Per mode-combo best (T >= 100) --
    print(f"\n  {'Modes':<8} {'Gate':<6} {'CD Config':<16} {'MR_BB':>5} | "
          f"{'IS PF':>7} {'T':>5} {'WR%':>6} | "
          f"{'OOS PF':>8} {'T':>5} {'t/d':>5} {'WR%':>6} {'Net':>9} {'DD%':>7}")
    print(f"  {'-'*125}")

    for mode_name in MODE_COMBOS:
        subset = [r for r in all_results
                  if r["mode_name"] == mode_name
                  and r["oos_trades"] >= 50 and r["oos_net"] > 0]
        if not subset:
            print(f"  {mode_name:<8} (no viable results)")
            continue
        best = max(subset, key=lambda r: r["oos_net"])
        _print_row_full(best, oos_days)

    # -- HF targets (T >= 170) sorted by net --
    hf = [r for r in all_results
          if r["oos_trades"] >= oos_days and r["oos_net"] > 0]
    hf.sort(key=lambda r: r["oos_net"], reverse=True)

    print(f"\n{'='*130}")
    print(f"  HIGH FREQ (T>={oos_days}, net>0): {len(hf)} configs")
    print(f"{'='*130}")
    _print_header_full()
    for i, r in enumerate(hf[:30]):
        _print_row_full(r, oos_days, rank=i+1)

    # -- Best WR (T >= 150) --
    hw = [r for r in all_results
          if r["oos_trades"] >= 150 and r["oos_net"] > 0]
    hw.sort(key=lambda r: (r["oos_wr"], r["oos_net"]), reverse=True)

    print(f"\n{'='*130}")
    print(f"  BEST WR (T>=150, net>0): {len(hw)} configs")
    print(f"{'='*130}")
    _print_header_full()
    for i, r in enumerate(hw[:30]):
        _print_row_full(r, oos_days, rank=i+1)

    # -- Best PF (T >= 150) --
    hp = [r for r in all_results
          if r["oos_trades"] >= 150 and r["oos_net"] > 0]
    hp.sort(key=lambda r: (
        r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6,
        r["oos_net"],
    ), reverse=True)

    print(f"\n{'='*130}")
    print(f"  BEST PF (T>=150, net>0): {len(hp)} configs")
    print(f"{'='*130}")
    _print_header_full()
    for i, r in enumerate(hp[:30]):
        _print_row_full(r, oos_days, rank=i+1)

    # -- Balanced: T >= 200, WR >= 55% --
    bal = [r for r in all_results
           if r["oos_trades"] >= 200 and r["oos_wr"] >= 55.0
           and r["oos_net"] > 0]
    bal.sort(key=lambda r: (r["oos_net"], r["oos_wr"]), reverse=True)

    print(f"\n{'='*130}")
    print(f"  BALANCED (T>=200, WR>=55%, net>0): {len(bal)} configs")
    print(f"{'='*130}")
    _print_header_full()
    for i, r in enumerate(bal[:30]):
        _print_row_full(r, oos_days, rank=i+1)

    # -- Mode contribution analysis --
    print(f"\n  --- Mode Contribution (avg OOS, psar gate, shared_3 cd, mr_bb=0.25) ---")
    for mode_name in MODE_COMBOS:
        rows = [r for r in all_results
                if r["mode_name"] == mode_name
                and r["gate_mode"] == "psar"
                and r["cd_label"] == "shared_3"
                and abs(r["mr_bb_thresh"] - 0.25) < 0.01
                and r["oos_trades"] >= 10]
        if rows:
            r = rows[0]
            tpd = r["oos_trades"] / oos_days
            pf_s = f"{r['oos_pf']:.2f}" if not math.isinf(r["oos_pf"]) else "INF"
            print(f"    {mode_name:<8} | OOS PF={pf_s:>7} T={r['oos_trades']:>5} ({tpd:.1f}/d) "
                  f"WR={r['oos_wr']:>5.1f}% Net={r['oos_net']:>9.2f}")

    # -- Per-mode vs shared cooldown --
    print(f"\n  --- Shared vs Per-Mode Cooldown (T+P+M, psar gate, mr_bb=0.25) ---")
    for cd_label in ["shared_3", "shared_5", "permode_3_5_3", "permode_3_8_3", "permode_3_3_5"]:
        rows = [r for r in all_results
                if r["mode_name"] == "T+P+M"
                and r["gate_mode"] == "psar"
                and r["cd_label"] == cd_label
                and abs(r["mr_bb_thresh"] - 0.25) < 0.01]
        if rows:
            r = rows[0]
            tpd = r["oos_trades"] / oos_days
            pf_s = f"{r['oos_pf']:.2f}" if not math.isinf(r["oos_pf"]) else "INF"
            print(f"    {cd_label:<16} | OOS PF={pf_s:>7} T={r['oos_trades']:>5} ({tpd:.1f}/d) "
                  f"WR={r['oos_wr']:>5.1f}% Net={r['oos_net']:>9.2f}")


def _print_header_full():
    print(f"  {'#':>3} {'Modes':<8} {'Gate':<6} {'CD Config':<16} {'MR_BB':>5} | "
          f"{'IS PF':>7} {'T':>5} {'WR%':>6} | "
          f"{'OOS PF':>8} {'T':>5} {'t/d':>5} {'WR%':>6} {'Net':>9} {'DD%':>7}")
    print(f"  {'-'*128}")


def _print_row_full(r, oos_days, rank=None):
    pf_i = "INF" if math.isinf(r["is_pf"]) else f"{r['is_pf']:.2f}"
    pf_o = "INF" if math.isinf(r["oos_pf"]) else f"{r['oos_pf']:.2f}"
    tpd = r["oos_trades"] / oos_days
    prefix = f"  {rank:>3}" if rank else "  "
    print(f"{prefix} {r['mode_name']:<8} {r['gate_mode']:<6} {r['cd_label']:<16} {r['mr_bb_thresh']:>4.2f} | "
          f"{pf_i:>7} {r['is_trades']:>5} {r['is_wr']:>5.1f}% | "
          f"{pf_o:>8} {r['oos_trades']:>5} {tpd:>4.1f} {r['oos_wr']:>5.1f}% "
          f"{r['oos_net']:>9.2f} {r['oos_dd']:>6.2f}%")


# -- Main -----------------------------------------------------------------------

def main():
    combos = _build_combos()
    total = len(combos)

    print(f"\n{'='*70}")
    print(f"BTC HF Phase 3 — Multi-Mode Entry System (Long Only)")
    print(f"  Mode combos : {list(MODE_COMBOS.keys())}")
    print(f"  Gates       : {GATE_MODES}")
    print(f"  CD configs  : {len(CD_CONFIGS)}")
    print(f"  MR thresholds: {MR_BB_THRESHOLDS}")
    print(f"  Total runs  : {total} ({total*2} backtests)")
    print(f"  Workers     : {MAX_WORKERS}")
    print(f"  IS period   : {IS_START} -> {IS_END}")
    print(f"  OOS period  : {OOS_START} -> {OOS_END}")
    print(f"{'='*70}\n")

    results = []
    done = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run_one, c): c for c in combos}
        for fut in as_completed(futures):
            try:
                combo, is_r, oos_r = fut.result()
                row = {
                    "mode_name":    combo["mode_name"],
                    "gate_mode":    combo["gate_mode"],
                    "cd_trend":     combo["cd_trend"],
                    "cd_pullback":  combo["cd_pullback"],
                    "cd_mr":        combo["cd_mr"],
                    "cd_label":     combo["cd_label"],
                    "mr_bb_thresh": combo["mr_bb_thresh"],
                    "is_pf":        is_r["pf"],
                    "is_trades":    is_r["trades"],
                    "is_wr":        is_r["wr"],
                    "is_net":       is_r["net"],
                    "is_dd":        is_r["dd"],
                    "oos_pf":       oos_r["pf"],
                    "oos_trades":   oos_r["trades"],
                    "oos_wr":       oos_r["wr"],
                    "oos_net":      oos_r["net"],
                    "oos_dd":       oos_r["dd"],
                }
                results.append(row)
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()

            done += 1
            if done % 50 == 0 or done == total:
                print(f"  [{done:>4}/{total}]", flush=True)

    # Save
    out_path = ROOT / "ai_context" / "btc_hf_multimode_results.json"
    out_path.parent.mkdir(exist_ok=True)
    serializable = []
    for r in results:
        sr = dict(r)
        for k in ("is_pf", "oos_pf"):
            if math.isinf(sr[k]):
                sr[k] = "inf"
        serializable.append(sr)
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved {len(results)} results -> {out_path}")

    _summarize(results)


if __name__ == "__main__":
    main()
