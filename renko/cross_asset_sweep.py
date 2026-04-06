#!/usr/bin/env python3
"""
cross_asset_sweep.py -- Universal cross-asset × cross-strategy matrix

Runs every portable signal type against every instrument/brick combo.
Output: a single JSON matrix showing IS/OOS performance for each cell.

Signal types (entries):
  - R001: N consecutive same-direction bricks (momentum)
  - R002: N opposite then reversal brick (counter-trend)
  - R007: R001 + R002 combined
  - PSAR_flip: PSAR direction change
  - stoch_cross: stochastic K crosses below threshold
  - ema_cross: EMA9 crosses above/below EMA21
  - escgo_cross: ESCGO fast > slow
  - mk_regime: Momentum King bull/bear regime

Gate combos (applied on top of each signal):
  - none: signal only
  - psar: + PSAR direction gate
  - adx25/30/40/50: + ADX threshold gate
  - psar+adx: combined

Exit: first opposing brick (universal).

Cooldowns: 3, 10, 20, 30
N-bricks (R001/R002/R007): 3, 5, 9

Usage:
    python renko/cross_asset_sweep.py
"""

import contextlib
import io
import json
import math
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from renko.config import MAX_WORKERS

warnings.filterwarnings("ignore")

# ── Instrument configs ──────────────────────────────────────────────────────

INSTRUMENTS = {
    "EURUSD_4": {
        "file": "OANDA_EURUSD, 1S renko 0.0004.csv",
        "is_start": "2023-01-23", "is_end": "2025-09-30",
        "oos_start": "2025-10-01", "oos_end": "2026-03-19",
        "oos_days": 170, "commission": 0.0046, "capital": 1000.0,
        "qty_type": "cash", "qty_value": 20,
    },
    "EURUSD_5": {
        "file": "OANDA_EURUSD, 1S renko 0.0005.csv",
        "is_start": "2022-05-18", "is_end": "2025-09-30",
        "oos_start": "2025-10-01", "oos_end": "2026-03-19",
        "oos_days": 170, "commission": 0.0046, "capital": 1000.0,
        "qty_type": "cash", "qty_value": 20,
    },
    "GBPJPY": {
        "file": "OANDA_GBPJPY, 1S renko 0.05.csv",
        "is_start": "2024-11-21", "is_end": "2025-09-30",
        "oos_start": "2025-10-01", "oos_end": "2026-03-11",
        "oos_days": 162, "commission": 0.005, "capital": 150000.0,
        "qty_type": "cash", "qty_value": 20,
    },
    "EURAUD": {
        "file": "OANDA_EURAUD, 1S renko 0.0006.csv",
        "is_start": "2023-07-20", "is_end": "2025-09-30",
        "oos_start": "2025-10-01", "oos_end": "2026-03-17",
        "oos_days": 168, "commission": 0.009, "capital": 1000.0,
        "qty_type": "cash", "qty_value": 20,
    },
    "GBPUSD": {
        "file": "OANDA_GBPUSD, 1S renko 0.0004.csv",
        "is_start": "2024-05-23", "is_end": "2025-09-30",
        "oos_start": "2025-10-01", "oos_end": "2026-03-17",
        "oos_days": 168, "commission": 0.005, "capital": 1000.0,
        "qty_type": "cash", "qty_value": 20,
    },
    "USDJPY": {
        "file": "OANDA_USDJPY, 1S renko 0.05.csv",
        "is_start": "2024-05-16", "is_end": "2025-09-30",
        "oos_start": "2025-10-01", "oos_end": "2026-03-17",
        "oos_days": 168, "commission": 0.005, "capital": 1000.0,
        "qty_type": "cash", "qty_value": 20,
    },
    "USDCHF": {
        "file": "OANDA_USDCHF, 1S renko 0.0005.csv",
        "is_start": "2024-01-01", "is_end": "2025-09-30",
        "oos_start": "2025-10-01", "oos_end": "2026-03-19",
        "oos_days": 170, "commission": 0.005, "capital": 1000.0,
        "qty_type": "cash", "qty_value": 20,
    },
    "BTCUSD": {
        "file": "OANDA_BTCUSD.SPOT.US, 1S renko 150.csv",
        "is_start": "2024-06-04", "is_end": "2025-09-30",
        "oos_start": "2025-10-01", "oos_end": "2026-03-19",
        "oos_days": 170, "commission": 0.0046, "capital": 1000.0,
        "qty_type": "cash", "qty_value": 20,
    },
    "MYM_14": {
        "file": "CBOT_MINI_MYM1!, 1S renko 14.csv",
        "is_start": "2025-03-07", "is_end": "2025-12-31",
        "oos_start": "2026-01-01", "oos_end": "2026-03-19",
        "oos_days": 78, "commission": 0.00475, "capital": 1000.0,
        "qty_type": "cash", "qty_value": 0.5,
    },
}

# ── Signal types ────────────────────────────────────────────────────────────

SIGNAL_TYPES = ["R001", "R002", "R007", "psar_flip", "stoch_cross",
                "ema_cross", "escgo_cross", "mk_regime"]

GATE_COMBOS = ["none", "psar", "adx25", "adx30", "adx40", "adx50",
               "psar+adx30", "psar+adx50", "chop60"]

N_BRICKS_OPTIONS = [3, 5, 9]
COOLDOWNS = [3, 10, 20, 30]

# ── Per-process cache ───────────────────────────────────────────────────────

_w = {}


def _init(inst_key):
    cache_key = f"df_{inst_key}"
    if cache_key in _w:
        return
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    from renko.phase6_enrichment import add_phase6_indicators
    from indicators.momentum_king import calc_momentum_king

    cfg = INSTRUMENTS[inst_key]
    df = load_renko_export(cfg["file"])
    add_renko_indicators(df)

    # Phase 6 indicators (ESCGO, ichi, etc.)
    try:
        add_phase6_indicators(df, include_mk=True)
    except Exception:
        pass

    # MK
    try:
        mk = calc_momentum_king(df)
        mk_sm = mk["smoothed_momentum"]
        mk_nz = mk["neutral_zone_width"]
        n = len(df)
        mk_bull = np.zeros(n, dtype=bool)
        mk_bear = np.zeros(n, dtype=bool)
        for i in range(1, n):
            mk_bull[i] = mk_sm[i - 1] > mk_nz[i - 1]
            mk_bear[i] = mk_sm[i - 1] < -mk_nz[i - 1]
        df["mk_bull"] = mk_bull
        df["mk_bear"] = mk_bear
    except Exception:
        df["mk_bull"] = False
        df["mk_bear"] = False

    _w[cache_key] = df
    _w[f"n_{inst_key}"] = len(df)


def _run_bt(inst_key, long_entry, long_exit, short_entry, short_exit, start, end):
    from engine import BacktestConfig, run_backtest

    cfg = INSTRUMENTS[inst_key]
    df2 = _w[f"df_{inst_key}"].copy()
    df2["long_entry"] = long_entry
    df2["long_exit"] = long_exit
    df2["short_entry"] = short_entry
    df2["short_exit"] = short_exit
    bt_cfg = BacktestConfig(
        initial_capital=cfg["capital"], commission_pct=cfg["commission"],
        slippage_ticks=0, qty_type=cfg["qty_type"],
        qty_value=cfg["qty_value"], pyramiding=1,
        start_date=start, end_date=end,
        take_profit_pct=0.0, stop_loss_pct=0.0,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest(df2, bt_cfg)
    pf = kpis.get("profit_factor", 0.0) or 0.0
    return {
        "pf": float("inf") if math.isinf(pf) else round(float(pf), 2),
        "trades": int(kpis.get("total_trades", 0) or 0),
        "wr": round(float(kpis.get("win_rate", 0.0) or 0.0), 1),
        "net": round(float(kpis.get("net_profit", 0.0) or 0.0), 4),
    }


def _gen_signal(df, signal_type, n_bricks):
    """Generate raw entry signals (before gates/cooldown)."""
    n = len(df)
    brick_up = df["brick_up"].values
    long_sig = np.zeros(n, dtype=bool)
    short_sig = np.zeros(n, dtype=bool)

    if signal_type == "R001":
        for i in range(n_bricks, n):
            all_up = all(brick_up[i - j] for j in range(n_bricks))
            all_dn = all(not brick_up[i - j] for j in range(n_bricks))
            if all_up:
                long_sig[i] = True
            if all_dn:
                short_sig[i] = True

    elif signal_type == "R002":
        for i in range(n_bricks + 1, n):
            prev_all_dn = all(not brick_up[i - 1 - j] for j in range(n_bricks))
            prev_all_up = all(brick_up[i - 1 - j] for j in range(n_bricks))
            if prev_all_dn and brick_up[i]:
                long_sig[i] = True
            if prev_all_up and not brick_up[i]:
                short_sig[i] = True

    elif signal_type == "R007":
        for i in range(n_bricks + 1, n):
            # R001
            all_up = all(brick_up[i - j] for j in range(n_bricks))
            all_dn = all(not brick_up[i - j] for j in range(n_bricks))
            # R002
            prev_all_dn = all(not brick_up[i - 1 - j] for j in range(n_bricks))
            prev_all_up = all(brick_up[i - 1 - j] for j in range(n_bricks))
            if all_up or (prev_all_dn and brick_up[i]):
                long_sig[i] = True
            if all_dn or (prev_all_up and not brick_up[i]):
                short_sig[i] = True

    elif signal_type == "psar_flip":
        psar_dir = df["psar_dir"].values
        for i in range(1, n):
            if brick_up[i] and (np.isnan(psar_dir[i]) or psar_dir[i] > 0):
                long_sig[i] = True
            if not brick_up[i] and (np.isnan(psar_dir[i]) or psar_dir[i] < 0):
                short_sig[i] = True

    elif signal_type == "stoch_cross":
        stk = df["stoch_k"].values
        for i in range(1, n):
            if brick_up[i] and not np.isnan(stk[i]) and stk[i] < 25:
                long_sig[i] = True
            if not brick_up[i] and not np.isnan(stk[i]) and stk[i] > 75:
                short_sig[i] = True

    elif signal_type == "ema_cross":
        ema9 = df["ema9"].values if "ema9" in df.columns else np.full(n, np.nan)
        ema21 = df["ema21"].values if "ema21" in df.columns else np.full(n, np.nan)
        for i in range(1, n):
            if np.isnan(ema9[i]) or np.isnan(ema21[i]):
                continue
            if brick_up[i] and ema9[i] > ema21[i]:
                long_sig[i] = True
            if not brick_up[i] and ema9[i] < ema21[i]:
                short_sig[i] = True

    elif signal_type == "escgo_cross":
        ef = df["escgo_fast"].values if "escgo_fast" in df.columns else np.full(n, np.nan)
        es = df["escgo_slow"].values if "escgo_slow" in df.columns else np.full(n, np.nan)
        for i in range(1, n):
            if np.isnan(ef[i]) or np.isnan(es[i]):
                continue
            if brick_up[i] and ef[i] > es[i]:
                long_sig[i] = True
            if not brick_up[i] and ef[i] < es[i]:
                short_sig[i] = True

    elif signal_type == "mk_regime":
        mk_b = df["mk_bull"].values
        mk_br = df["mk_bear"].values
        for i in range(1, n):
            if brick_up[i] and mk_b[i]:
                long_sig[i] = True
            if not brick_up[i] and mk_br[i]:
                short_sig[i] = True

    return long_sig, short_sig


def _apply_gate(df, long_sig, short_sig, gate_name):
    """Apply gate filter to raw signals."""
    n = len(df)

    if gate_name == "none":
        return long_sig.copy(), short_sig.copy()

    if gate_name == "psar":
        psar_dir = df["psar_dir"].values
        pl = np.array([np.isnan(psar_dir[i]) or psar_dir[i] > 0 for i in range(n)])
        ps = np.array([np.isnan(psar_dir[i]) or psar_dir[i] < 0 for i in range(n)])
        return long_sig & pl, short_sig & ps

    if gate_name.startswith("adx"):
        thresh = int(gate_name.replace("adx", ""))
        adx = df["adx"].values
        ok = np.array([np.isnan(adx[i]) or adx[i] >= thresh for i in range(n)])
        return long_sig & ok, short_sig & ok

    if gate_name == "psar+adx30":
        l1, s1 = _apply_gate(df, long_sig, short_sig, "psar")
        return _apply_gate(df, l1, s1, "adx30")

    if gate_name == "psar+adx50":
        l1, s1 = _apply_gate(df, long_sig, short_sig, "psar")
        return _apply_gate(df, l1, s1, "adx50")

    if gate_name == "chop60":
        chop = df["chop"].values
        ok = np.array([np.isnan(chop[i]) or chop[i] <= 60 for i in range(n)])
        return long_sig & ok, short_sig & ok

    return long_sig.copy(), short_sig.copy()


def _apply_cooldown_and_exit(df, long_sig, short_sig, cd):
    """Apply cooldown + first-opposing-brick exit."""
    n = len(df)
    brick_up = df["brick_up"].values

    long_entry = np.zeros(n, dtype=bool)
    long_exit = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit = np.zeros(n, dtype=bool)

    in_pos = False
    pos_dir = 0
    last_entry = -999

    for i in range(1, n):
        up = bool(brick_up[i])
        if in_pos:
            if (pos_dir == 1 and not up) or (pos_dir == -1 and up):
                if pos_dir == 1:
                    long_exit[i] = True
                else:
                    short_exit[i] = True
                in_pos = False
            continue
        if i - last_entry < cd:
            continue
        if long_sig[i] and up:
            long_entry[i] = True
            in_pos = True
            pos_dir = 1
            last_entry = i
        elif short_sig[i] and not up:
            short_entry[i] = True
            in_pos = True
            pos_dir = -1
            last_entry = i

    return long_entry, long_exit, short_entry, short_exit


def _run_one(combo):
    """Run a single instrument × signal × gate × params combo."""
    inst = combo["inst"]
    _init(inst)
    df = _w[f"df_{inst}"]
    cfg = INSTRUMENTS[inst]

    sig_type = combo["signal"]
    n_bricks = combo.get("n_bricks", 3)
    gate = combo["gate"]
    cd = combo["cd"]

    # Generate signals
    long_sig, short_sig = _gen_signal(df, sig_type, n_bricks)

    # Apply gate
    long_sig, short_sig = _apply_gate(df, long_sig, short_sig, gate)

    # Apply cooldown + exit
    le, lx, se, sx = _apply_cooldown_and_exit(df, long_sig, short_sig, cd)

    # Run backtests
    is_r = _run_bt(inst, le, lx, se, sx, cfg["is_start"], cfg["is_end"])
    oos_r = _run_bt(inst, le, lx, se, sx, cfg["oos_start"], cfg["oos_end"])

    label_parts = [inst, sig_type]
    if sig_type in ("R001", "R002", "R007"):
        label_parts.append(f"n{n_bricks}")
    label_parts.extend([gate, f"cd{cd}"])
    label = "_".join(label_parts)

    return {
        "label": label,
        "inst": inst,
        "signal": sig_type,
        "n_bricks": n_bricks,
        "gate": gate,
        "cd": cd,
        "is": is_r,
        "oos": oos_r,
    }


def _build_combos():
    combos = []
    for inst in INSTRUMENTS:
        for sig in SIGNAL_TYPES:
            if sig in ("R001", "R002", "R007"):
                n_opts = N_BRICKS_OPTIONS
            else:
                n_opts = [3]  # n_bricks irrelevant for non-brick signals

            for n_b in n_opts:
                for gate in GATE_COMBOS:
                    for cd in COOLDOWNS:
                        combos.append({
                            "inst": inst,
                            "signal": sig,
                            "n_bricks": n_b,
                            "gate": gate,
                            "cd": cd,
                        })
    return combos


if __name__ == "__main__":
    combos = _build_combos()
    print(f"Cross-Asset Sweep: {len(combos)} combos across "
          f"{len(INSTRUMENTS)} instruments × {len(SIGNAL_TYPES)} signals "
          f"on {MAX_WORKERS} workers", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(_run_one, c): c for c in combos}
        done = 0
        for f in as_completed(futs):
            done += 1
            if done % 1000 == 0:
                print(f"  {done}/{len(combos)}...", flush=True)
            try:
                results.append(f.result())
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)

    # ── Analysis ────────────────────────────────────────────────────────────

    # Filter viable: OOS >= 10 trades
    viable = [r for r in results if r["oos"]["trades"] >= 10]

    # Best per instrument (by OOS WR)
    print(f"\n{'='*100}")
    print(f"BEST PER INSTRUMENT (OOS trades >= 10, sorted by WR then PF)")
    print("=" * 100)

    for inst in INSTRUMENTS:
        iv = [r for r in viable if r["inst"] == inst]
        if not iv:
            print(f"\n{inst}: no viable results")
            continue
        iv.sort(key=lambda r: (r["oos"]["wr"], r["oos"]["pf"]), reverse=True)
        oos_days = INSTRUMENTS[inst]["oos_days"]
        print(f"\n-- {inst} (top 10) --")
        hdr = f"  {'Config':<55} {'IS PF':>7} {'IS WR':>6} {'OOS PF':>7} {'OOS T':>5} {'t/d':>5} {'OOS WR':>7}"
        print(hdr)
        for r in iv[:10]:
            tpd = r["oos"]["trades"] / oos_days
            pf_s = "inf" if r["oos"]["pf"] == float("inf") else f"{r['oos']['pf']:.1f}"
            is_pf = "inf" if r["is"]["pf"] == float("inf") else f"{r['is']['pf']:.1f}"
            print(f"  {r['label']:<55} {is_pf:>7} {r['is']['wr']:>5.1f}% "
                  f"{pf_s:>7} {r['oos']['trades']:>5} {tpd:>5.1f} {r['oos']['wr']:>6.1f}%")

    # Best signal per instrument (collapse gates/params)
    print(f"\n{'='*100}")
    print(f"BEST SIGNAL TYPE PER INSTRUMENT (best gate/cd combo for each)")
    print("=" * 100)
    hdr2 = f"  {'Instrument':<15} {'Signal':<15} {'Gate':<15} {'CD':>3} {'OOS PF':>7} {'OOS T':>5} {'OOS WR':>7}"
    print(hdr2)
    print("  " + "-" * 80)
    for inst in INSTRUMENTS:
        for sig in SIGNAL_TYPES:
            sv = [r for r in viable if r["inst"] == inst and r["signal"] == sig]
            if not sv:
                continue
            best = max(sv, key=lambda r: (r["oos"]["wr"], r["oos"]["pf"]))
            pf_s = "inf" if best["oos"]["pf"] == float("inf") else f"{best['oos']['pf']:.1f}"
            nb = f"(n={best['n_bricks']})" if sig in ("R001", "R002", "R007") else ""
            print(f"  {inst:<15} {sig+nb:<15} {best['gate']:<15} {best['cd']:>3} "
                  f"{pf_s:>7} {best['oos']['trades']:>5} {best['oos']['wr']:>6.1f}%")
        print("  " + "-" * 80)

    # Global top 50
    viable.sort(key=lambda r: (r["oos"]["wr"], r["oos"]["pf"]), reverse=True)
    print(f"\n{'='*100}")
    print(f"GLOBAL TOP 50 (OOS trades >= 10)")
    print("=" * 100)
    hdr3 = f"{'Config':<60} {'IS PF':>7} {'IS WR':>6} {'OOS PF':>7} {'OOS T':>5} {'OOS WR':>7}"
    print(hdr3)
    for r in viable[:50]:
        pf_s = "inf" if r["oos"]["pf"] == float("inf") else f"{r['oos']['pf']:.1f}"
        is_pf = "inf" if r["is"]["pf"] == float("inf") else f"{r['is']['pf']:.1f}"
        print(f"{r['label']:<60} {is_pf:>7} {r['is']['wr']:>5.1f}% "
              f"{pf_s:>7} {r['oos']['trades']:>5} {r['oos']['wr']:>6.1f}%")

    # Summary stats
    print(f"\nTotal: {len(results)} runs, {len(viable)} viable (>=10t OOS)")

    # Save
    out = ROOT / "ai_context" / "cross_asset_matrix.json"
    with open(out, "w") as f:
        json.dump({
            "total": len(results),
            "viable": len(viable),
            "top50": viable[:50],
            "best_per_inst": {
                inst: max(
                    [r for r in viable if r["inst"] == inst],
                    key=lambda r: (r["oos"]["wr"], r["oos"]["pf"]),
                    default=None,
                )
                for inst in INSTRUMENTS
            },
            "all": results,
        }, f, indent=2)
    print(f"Saved to {out}")
