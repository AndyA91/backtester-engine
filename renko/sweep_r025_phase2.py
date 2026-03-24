#!/usr/bin/env python3
"""
R025 Phase 2 — momentum gate sweep (stoch, MACD, RSI).

Locks ADX=25 (phase 1 winner, +19% PF). Sweeps:
  - ribbon × cooldown × session × vol × ST (phase 1 params)
  - stoch_gate × macd_gate × rsi_gate (new phase 2 params)

128 combos × 5 pairs × 2 periods (IS + OOS).
"""

import contextlib
import io
import itertools
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent / "strategies"))

IS_END    = "2025-09-30"
OOS_START = "2025-10-01"
OOS_END   = "2026-03-19"

INSTRUMENTS = {
    "EURUSD": {
        "renko_file": "OANDA_EURUSD, 1S renko 0.0004.csv",
        "is_start":   "2023-01-23",
        "commission":  0.0046,
        "capital":     1000.0,
        "qty_value":   1000.0,
    },
    "GBPJPY": {
        "renko_file": "OANDA_GBPJPY, 1S renko 0.05.csv",
        "is_start":   "2024-11-22",
        "commission":  0.005,
        "capital":     150_000.0,
        "qty_value":   1000.0,
    },
    "EURAUD": {
        "renko_file": "OANDA_EURAUD, 1S renko 0.0006.csv",
        "is_start":   "2023-07-20",
        "commission":  0.009,
        "capital":     1000.0,
        "qty_value":   1000.0,
    },
    "GBPUSD": {
        "renko_file": "OANDA_GBPUSD, 1S renko 0.0004.csv",
        "is_start":   "2023-11-15",
        "commission":  0.0046,
        "capital":     1000.0,
        "qty_value":   1000.0,
    },
    "USDJPY": {
        "renko_file": "OANDA_USDJPY, 1S renko 0.05.csv",
        "is_start":   "2024-07-18",
        "commission":  0.005,
        "capital":     150_000.0,
        "qty_value":   1000.0,
    },
}

LIVE = {
    "EURUSD": {"strategy": "R016",  "pf": 22.91, "trades": 26, "wr": 69.2},
    "GBPJPY": {"strategy": "GJ014", "pf": float("inf"), "trades": 17, "wr": 100.0},
    "EURAUD": {"strategy": "EA021", "pf": 14.21, "trades": 19, "wr": 68.4},
    "GBPUSD": {"strategy": "GU001", "pf": 54.31, "trades": 13, "wr": 84.6},
    "USDJPY": {"strategy": "UJ001", "pf": 35.36, "trades": 19, "wr": 84.2},
}

# ── Param grid: lock ADX=25, sweep everything else ─────────────────────────
GRID = {
    "ribbon":        ["3L_5_13_34", "3L_8_21_55"],
    "cooldown":      [3, 5],
    "adx_gate":      [25],            # locked
    "session_start": [0, 13],
    "vol_max":       [0, 1.5],
    "st_gate":       [False, True],
    "stoch_gate":    [False, True],   # NEW
    "macd_gate":     [False, True],   # NEW
    "rsi_gate":      [0, 70],         # NEW
}

GRID_KEYS = list(GRID.keys())
COMBOS = [dict(zip(GRID_KEYS, v)) for v in itertools.product(*GRID.values())]
MIN_TRADES = 20


def run_instrument(name, cfg):
    from engine import BacktestConfig, run_backtest_long_short
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    from renko.strategies.r025_kama_ribbon_gated import generate_signals, _KAMA_CACHE

    df = load_renko_export(cfg["renko_file"])
    add_renko_indicators(df)

    def make_cfg(start, end):
        return BacktestConfig(
            initial_capital=cfg["capital"],
            commission_pct=cfg["commission"],
            slippage_ticks=0,
            qty_type="fixed",
            qty_value=cfg["qty_value"],
            pyramiding=1,
            start_date=start,
            end_date=end,
            take_profit_pct=0.0,
            stop_loss_pct=0.0,
        )

    bcfg_is  = make_cfg(cfg["is_start"], IS_END)
    bcfg_oos = make_cfg(OOS_START, OOS_END)

    results = []
    for params in COMBOS:
        _KAMA_CACHE.clear()
        df_sig = generate_signals(df.copy(), **params)
        row = {"pair": name, "params": params}
        for period, bcfg in [("is", bcfg_is), ("oos", bcfg_oos)]:
            with contextlib.redirect_stdout(io.StringIO()):
                kpis = run_backtest_long_short(df_sig, bcfg)
            pf = kpis.get("profit_factor", 0.0) or 0.0
            row[f"{period}_trades"] = int(kpis.get("total_trades", 0) or 0)
            row[f"{period}_wr"]     = float(kpis.get("win_rate", 0.0) or 0.0)
            row[f"{period}_net"]    = float(kpis.get("net_profit", 0.0) or 0.0)
            row[f"{period}_pf"]     = float("inf") if math.isinf(pf) else float(pf)
            row[f"{period}_dd"]     = float(kpis.get("max_drawdown_pct", 0.0) or 0.0)
        results.append(row)
    return name, results


def fmt_pf(pf):
    return "INF" if math.isinf(pf) else f"{pf:.2f}"


def main():
    print(f"R025 Phase 2 — Momentum Gate Sweep (stoch + MACD + RSI)")
    print(f"Combos per pair: {len(COMBOS)}  |  Total: {len(COMBOS) * len(INSTRUMENTS)}")
    print()

    all_results = {}
    with ProcessPoolExecutor(max_workers=len(INSTRUMENTS)) as pool:
        futures = {pool.submit(run_instrument, n, c): n for n, c in INSTRUMENTS.items()}
        for fut in as_completed(futures):
            name, results = fut.result()
            all_results[name] = results
            print(f"  {name}: {len(results)} combos done")

    # ── Per-pair: top 5 OOS combos ──────────────────────────────────────────
    print("\n" + "=" * 110)
    print("  TOP 5 OOS COMBOS PER PAIR (min 20 trades, ranked by PF)")
    print("=" * 110)

    for pair in INSTRUMENTS:
        results = all_results[pair]
        qualified = [r for r in results if r["oos_trades"] >= MIN_TRADES]
        qualified.sort(key=lambda r: (r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e12, r["oos_net"]), reverse=True)

        lv = LIVE[pair]
        print(f"\n  {pair}  (live: {lv['strategy']} PF={fmt_pf(lv['pf'])} T={lv['trades']} WR={lv['wr']}%)")
        print(f"  {'':4}{'OOS PF':>8} {'OOS T':>6} {'OOS WR':>7} {'OOS Net':>11} {'OOS DD':>7}"
              f"  |  {'IS PF':>7} {'IS T':>5} {'IS WR':>6}  | Gates")
        print(f"  {'-'*105}")

        for r in qualified[:5]:
            p = r["params"]
            gates = []
            if p["session_start"] > 0:  gates.append(f"s{p['session_start']}")
            if p["vol_max"] > 0:        gates.append(f"vol{p['vol_max']}")
            if p["st_gate"]:             gates.append("ST")
            if p["stoch_gate"]:          gates.append("stoch")
            if p["macd_gate"]:           gates.append("MACD")
            if p["rsi_gate"] > 0:        gates.append(f"rsi{p['rsi_gate']}")
            gate_str = "+".join(gates) if gates else "none"

            print(f"  {'':4}{fmt_pf(r['oos_pf']):>8} {r['oos_trades']:>6} {r['oos_wr']:>6.1f}% "
                  f"{r['oos_net']:>11.2f} {r['oos_dd']:>6.2f}%"
                  f"  |  {fmt_pf(r['is_pf']):>7} {r['is_trades']:>5} {r['is_wr']:>5.1f}%"
                  f"  | rb={p['ribbon']} cd={p['cooldown']} {gate_str}")

        if not qualified:
            print(f"  {'':4}No combos with >= {MIN_TRADES} OOS trades")

    # ── Gate impact analysis ────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("  GATE IMPACT (median OOS PF by gate on/off, all pairs, min 20 OOS trades)")
    print("=" * 110)

    all_rows = []
    for pair in INSTRUMENTS:
        all_rows.extend(all_results[pair])
    qualified_all = [r for r in all_rows if r["oos_trades"] >= MIN_TRADES]

    for gate_name, param_key, on_val, off_val in [
        ("Session s13",  "session_start", 13, 0),
        ("Vol max 1.5",  "vol_max",       1.5, 0),
        ("Supertrend",   "st_gate",       True, False),
        ("Stoch cross",  "stoch_gate",    True, False),
        ("MACD hist",    "macd_gate",     True, False),
        ("RSI 70/30",    "rsi_gate",      70, 0),
    ]:
        on_pfs  = sorted([r["oos_pf"] for r in qualified_all if r["params"][param_key] == on_val and not math.isinf(r["oos_pf"])])
        off_pfs = sorted([r["oos_pf"] for r in qualified_all if r["params"][param_key] == off_val and not math.isinf(r["oos_pf"])])
        on_med  = on_pfs[len(on_pfs)//2] if on_pfs else 0
        off_med = off_pfs[len(off_pfs)//2] if off_pfs else 0
        on_t  = sorted([r["oos_trades"] for r in qualified_all if r["params"][param_key] == on_val])
        off_t = sorted([r["oos_trades"] for r in qualified_all if r["params"][param_key] == off_val])
        on_med_t  = on_t[len(on_t)//2] if on_t else 0
        off_med_t = off_t[len(off_t)//2] if off_t else 0
        delta = ((on_med - off_med) / off_med * 100) if off_med > 0 else 0
        print(f"  {gate_name:<16}  OFF: PF {off_med:>6.2f} T={off_med_t:>4}  |  ON: PF {on_med:>6.2f} T={on_med_t:>4}  |  PF delta: {delta:>+.0f}%")

    # ── Per-pair gate impact ────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("  PER-PAIR GATE IMPACT (median OOS PF, min 20 trades)")
    print("=" * 110)

    for gate_name, param_key, on_val, off_val in [
        ("Stoch cross",  "stoch_gate",    True, False),
        ("MACD hist",    "macd_gate",     True, False),
        ("RSI 70/30",    "rsi_gate",      70, 0),
    ]:
        print(f"\n  {gate_name}:")
        for pair in INSTRUMENTS:
            rows = [r for r in all_results[pair] if r["oos_trades"] >= MIN_TRADES]
            on_pfs  = sorted([r["oos_pf"] for r in rows if r["params"][param_key] == on_val and not math.isinf(r["oos_pf"])])
            off_pfs = sorted([r["oos_pf"] for r in rows if r["params"][param_key] == off_val and not math.isinf(r["oos_pf"])])
            on_med  = on_pfs[len(on_pfs)//2] if on_pfs else 0
            off_med = off_pfs[len(off_pfs)//2] if off_pfs else 0
            delta = ((on_med - off_med) / off_med * 100) if off_med > 0 else 0
            on_t = sorted([r["oos_trades"] for r in rows if r["params"][param_key] == on_val])
            off_t = sorted([r["oos_trades"] for r in rows if r["params"][param_key] == off_val])
            on_med_t  = on_t[len(on_t)//2] if on_t else 0
            off_med_t = off_t[len(off_t)//2] if off_t else 0
            print(f"    {pair:<8}  OFF: PF {off_med:>6.2f} T={off_med_t:>3}  |  ON: PF {on_med:>6.2f} T={on_med_t:>3}  |  {delta:>+.0f}%")

    # ── IS/OOS decay for top combo per pair ─────────────────────────────────
    print("\n" + "=" * 110)
    print("  IS → OOS DECAY (best OOS PF combo per pair)")
    print("=" * 110)
    print(f"  {'Pair':<8} {'IS PF':>7} {'OOS PF':>8} {'Decay':>8} {'IS T':>5} {'OOS T':>6} {'Live PF':>8}")
    print(f"  {'-'*55}")

    for pair in INSTRUMENTS:
        qualified = [r for r in all_results[pair] if r["oos_trades"] >= MIN_TRADES]
        if not qualified:
            continue
        best = max(qualified, key=lambda r: (r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e12, r["oos_net"]))
        is_pf = best["is_pf"] if not math.isinf(best["is_pf"]) else 999
        oos_pf = best["oos_pf"] if not math.isinf(best["oos_pf"]) else 999
        decay = ((oos_pf - is_pf) / is_pf * 100) if is_pf > 0 else 0
        lv = LIVE[pair]
        print(f"  {pair:<8} {fmt_pf(best['is_pf']):>7} {fmt_pf(best['oos_pf']):>8} {decay:>+7.0f}% "
              f"{best['is_trades']:>5} {best['oos_trades']:>6} {fmt_pf(lv['pf']):>8}")


if __name__ == "__main__":
    main()
