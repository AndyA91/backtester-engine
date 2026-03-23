#!/usr/bin/env python3
"""
R024 resume_1 OOS head-to-head vs live portfolio.

Uses the best IS params per pair from the sweep, runs them on the sealed
OOS period (2025-10-01 → 2026-03-19), and compares to live portfolio KPIs.
"""

import contextlib
import io
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent / "strategies"))

# ── OOS period (matches live portfolio validation window) ───────────────────
OOS_START = "2025-10-01"
OOS_END   = "2026-03-19"

# ── Live portfolio benchmarks (from live_portfolio.json) ────────────────────
LIVE = {
    "EURUSD": {"strategy": "R016",  "pf": 22.91, "trades": 26, "wr": 69.2},
    "GBPJPY": {"strategy": "GJ014", "pf": float("inf"), "trades": 17, "wr": 100.0},
    "EURAUD": {"strategy": "EA021", "pf": 14.21, "trades": 19, "wr": 68.4},
    "GBPUSD": {"strategy": "GU001", "pf": 54.31, "trades": 13, "wr": 84.6},
    "USDJPY": {"strategy": "UJ001", "pf": 35.36, "trades": 19, "wr": 84.2},
}

# ── Best IS params per pair (from sweep results) + instrument config ────────
# For each pair: the best fresh_only params, then we run all 3 entry modes
PAIRS = {
    "EURUSD": {
        "renko_file": "OANDA_EURUSD, 1S renko 0.0004.csv",
        "commission":  0.0046,
        "capital":     1000.0,
        "qty_value":   1000.0,
        # Best IS fresh_only: rb=3L_8_21_55 cd=3 adx=25
        "base_params": {"ribbon": "3L_8_21_55", "cooldown": 3, "adx_gate": 25, "exit_on_brick": True},
    },
    "GBPJPY": {
        "renko_file": "OANDA_GBPJPY, 1S renko 0.05.csv",
        "commission":  0.005,
        "capital":     150_000.0,
        "qty_value":   1000.0,
        # Best IS fresh_only: rb=3L_8_21_55 cd=5 adx=0
        "base_params": {"ribbon": "3L_8_21_55", "cooldown": 5, "adx_gate": 0, "exit_on_brick": True},
    },
    "EURAUD": {
        "renko_file": "OANDA_EURAUD, 1S renko 0.0006.csv",
        "commission":  0.009,
        "capital":     1000.0,
        "qty_value":   1000.0,
        # Best IS fresh_only: rb=3L_5_13_34 cd=5 adx=25
        "base_params": {"ribbon": "3L_5_13_34", "cooldown": 5, "adx_gate": 25, "exit_on_brick": True},
    },
    "GBPUSD": {
        "renko_file": "OANDA_GBPUSD, 1S renko 0.0004.csv",
        "commission":  0.0046,
        "capital":     1000.0,
        "qty_value":   1000.0,
        # Best IS fresh_only: rb=3L_5_13_34 cd=3 adx=25
        "base_params": {"ribbon": "3L_5_13_34", "cooldown": 3, "adx_gate": 25, "exit_on_brick": True},
    },
    "USDJPY": {
        "renko_file": "OANDA_USDJPY, 1S renko 0.05.csv",
        "commission":  0.005,
        "capital":     150_000.0,
        "qty_value":   1000.0,
        # Best IS fresh_only: rb=3L_5_13_34 cd=3 adx=25
        "base_params": {"ribbon": "3L_5_13_34", "cooldown": 3, "adx_gate": 25, "exit_on_brick": True},
    },
}

ENTRY_MODES = ["fresh_only", "resume_1", "resume"]


def run_pair(name, cfg):
    """Run all 3 entry modes for one pair on OOS. Called in subprocess."""
    from engine import BacktestConfig, run_backtest_long_short
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    from renko.strategies.r024_kama_ribbon_pullback import generate_signals, _KAMA_CACHE

    df = load_renko_export(cfg["renko_file"])
    add_renko_indicators(df)

    bcfg = BacktestConfig(
        initial_capital=cfg["capital"],
        commission_pct=cfg["commission"],
        slippage_ticks=0,
        qty_type="fixed",
        qty_value=cfg["qty_value"],
        pyramiding=1,
        start_date=OOS_START,
        end_date=OOS_END,
        take_profit_pct=0.0,
        stop_loss_pct=0.0,
    )

    results = {}
    for mode in ENTRY_MODES:
        _KAMA_CACHE.clear()
        params = {**cfg["base_params"], "entry_mode": mode}
        df_sig = generate_signals(df.copy(), **params)
        with contextlib.redirect_stdout(io.StringIO()):
            kpis = run_backtest_long_short(df_sig, bcfg)

        pf = kpis.get("profit_factor", 0.0) or 0.0
        trades = [t for t in kpis.get("trades", []) if t.exit_price is not None]
        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total_win = sum(wins) if wins else 0
        total_loss = abs(sum(losses)) if losses else 0

        results[mode] = {
            "trades":   int(kpis.get("total_trades", 0) or 0),
            "win_rate": float(kpis.get("win_rate", 0.0) or 0.0),
            "net":      float(kpis.get("net_profit", 0.0) or 0.0),
            "pf":       float("inf") if (math.isinf(pf) or total_loss == 0) else float(pf),
            "max_dd_pct": float(kpis.get("max_drawdown_pct", 0.0) or 0.0),
        }

    return name, results


def fmt_pf(pf):
    return "INF" if math.isinf(pf) else f"{pf:.2f}"


def main():
    print(f"R024 OOS Head-to-Head vs Live Portfolio")
    print(f"OOS Period: {OOS_START} to {OOS_END}")
    print()

    all_results = {}
    with ProcessPoolExecutor(max_workers=len(PAIRS)) as pool:
        futures = {pool.submit(run_pair, name, cfg): name for name, cfg in PAIRS.items()}
        for fut in as_completed(futures):
            name, results = fut.result()
            all_results[name] = results

    # ── Per-pair comparison ─────────────────────────────────────────────────
    print("=" * 90)
    print(f"  {'Pair':<8} {'Source':<13} {'PF':>7} {'Trades':>7} {'WR%':>7} {'Net':>12} {'DD%':>7}")
    print("=" * 90)

    portfolio_net = 0.0
    r024_net = 0.0

    for pair in PAIRS:
        # Live portfolio row
        lv = LIVE[pair]
        print(f"  {pair:<8} {lv['strategy']:<13} {fmt_pf(lv['pf']):>7} {lv['trades']:>7} "
              f"{lv['wr']:>6.1f}%  {'n/a':>12} {'n/a':>7}")

        # R024 rows
        for mode in ENTRY_MODES:
            r = all_results[pair][mode]
            tag = f"R024-{mode}"
            marker = " <--" if mode == "resume_1" else ""
            print(f"  {'':8} {tag:<13} {fmt_pf(r['pf']):>7} {r['trades']:>7} "
                  f"{r['win_rate']:>6.1f}% {r['net']:>12.2f} {r['max_dd_pct']:>6.2f}%{marker}")

        if pair in all_results:
            r024_net += all_results[pair]["resume_1"]["net"]
        print()

    # ── Portfolio-level summary ─────────────────────────────────────────────
    print("=" * 90)
    print("  PORTFOLIO SUMMARY (resume_1 across all 5 pairs)")
    print("=" * 90)

    total_trades = sum(all_results[p]["resume_1"]["trades"] for p in PAIRS)
    total_net = sum(all_results[p]["resume_1"]["net"] for p in PAIRS)

    # Weighted avg PF
    all_pfs = []
    for p in PAIRS:
        r = all_results[p]["resume_1"]
        if not math.isinf(r["pf"]) and r["trades"] > 0:
            all_pfs.append((r["pf"], r["trades"]))

    weighted_pf = sum(pf * t for pf, t in all_pfs) / sum(t for _, t in all_pfs) if all_pfs else 0
    avg_wr = sum(all_results[p]["resume_1"]["win_rate"] * all_results[p]["resume_1"]["trades"]
                 for p in PAIRS) / total_trades if total_trades else 0

    print(f"  R024 resume_1:  Trades={total_trades}  Net={total_net:.2f}  "
          f"Wtd-PF={weighted_pf:.2f}  Wtd-WR={avg_wr:.1f}%")
    print(f"  Live portfolio: Trades=94  Net=$175.21  PF=51.35  WR=85.7% (win-day rate)")
    print()

    # Note about capital normalization
    print("  NOTE: Net P&L not directly comparable — live uses $1K per pair (FX),")
    print("  R024 uses $1K (FX) / $150K (JPY) / fixed 1000 units. Compare PF and WR.")


if __name__ == "__main__":
    main()
