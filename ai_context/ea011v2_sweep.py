"""
EA011 v2 Auction Champion — Parallel Parameter Sweep (24-core)

Parallelisation strategy:
  • 384 combos split by vp_lookback → 3 groups of 128
  • Each group split across (N_CORES // 3) workers = 8 workers per group
  • Each worker builds exactly 1 VP cache then processes its ~16 combos
  • Signals generated ONCE per combo; IS and OOS backtests share the same df_sig

Expected runtime: ~2-4 min (vs ~25 min sequential)

Run from project root:
  python ai_context/ea011v2_sweep.py
"""

import contextlib
import io
import itertools
import json
import math
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "renko" / "strategies"))

IS_START  = "2023-07-20"
IS_END    = "2025-09-30"
OOS_START = "2025-10-01"
OOS_END   = "2026-03-17"

OUT_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Worker — must be top-level for Windows multiprocessing (spawn method)
# ---------------------------------------------------------------------------

def _run_batch(args):
    """
    Worker process: build VP cache once for this vp_lookback group, then run
    all assigned combos.  Signals are generated once per combo; IS and OOS
    backtests both use the same df_sig to halve signal-generation work.
    Returns {"is": [...], "oos": [...]} result dicts.
    """
    combo_batch, is_start, is_end, oos_start, oos_end = args

    # Re-insert paths — necessary for Windows spawn (fresh interpreter per worker)
    import sys
    from pathlib import Path as _P
    _root = _P(__file__).resolve().parent.parent
    for p in [str(_root), str(_root / "renko" / "strategies")]:
        if p not in sys.path:
            sys.path.insert(0, p)

    import contextlib, io, math
    import ea011_v2_auction_champion as mod
    from engine import BacktestConfig, run_backtest_long_short

    # Build cache once for this batch's vp_lookback — all combos share it
    vp_lookback = combo_batch[0]["vp_lookback"]
    cached_df   = mod._get_or_build_cache(vp_lookback)

    comm  = mod.COMMISSION_PCT
    cap   = mod.INITIAL_CAPITAL

    def _backtest(df_sig, start, end):
        cfg = BacktestConfig(
            initial_capital=cap,
            commission_pct=comm,
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
            "pf":         float("inf") if math.isinf(pf) else float(pf),
            "net":        float(kpis.get("net_profit",       0.0) or 0.0),
            "trades":     int(  kpis.get("total_trades",      0)   or 0),
            "win_rate":   float(kpis.get("win_rate",          0.0) or 0.0),
            "max_dd_pct": float(kpis.get("max_drawdown_pct",  0.0) or 0.0),
            "expectancy": float(kpis.get("avg_trade",         0.0) or 0.0),
            "avg_wl":     float(kpis.get("avg_win_loss_ratio",0.0) or 0.0),
        }

    is_results  = []
    oos_results = []

    for params in combo_batch:
        # Generate signals once, reuse for both periods
        df_sig = mod.generate_signals(cached_df.copy(), **params)

        r_is  = _backtest(df_sig, is_start,  is_end)
        r_oos = _backtest(df_sig, oos_start, oos_end)

        r_is["params"]  = params
        r_oos["params"] = params

        is_results.append(r_is)
        oos_results.append(r_oos)

    return {"is": is_results, "oos": oos_results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import ea011_v2_auction_champion as mod

    grid       = mod.PARAM_GRID
    keys       = list(grid.keys())
    all_combos = [dict(zip(keys, v)) for v in itertools.product(*grid.values())]

    # Group combos by vp_lookback so each worker builds exactly 1 VP cache
    by_vp: dict = {}
    for combo in all_combos:
        by_vp.setdefault(combo["vp_lookback"], []).append(combo)

    # Split each vp group across (N_CORES // N_VP_GROUPS) workers
    N_CORES     = cpu_count()       # 24
    N_VP        = len(by_vp)        # 3
    workers_per_group = max(1, N_CORES // N_VP)  # 8

    batches = []
    for vp, combos in by_vp.items():
        chunk = max(1, math.ceil(len(combos) / workers_per_group))
        for i in range(0, len(combos), chunk):
            batches.append(combos[i : i + chunk])

    n_workers = min(len(batches), N_CORES)
    print(
        f"EA011 v2 Parallel Sweep\n"
        f"  Combos : {len(all_combos)} | Batches : {len(batches)} | Workers : {n_workers}\n"
        f"  VP builds per worker: 1  |  Signal gen per combo: 1 (IS+OOS reuse)\n"
    )

    args_list = [
        (batch, IS_START, IS_END, OOS_START, OOS_END)
        for batch in batches
    ]

    with Pool(processes=n_workers) as pool:
        batch_results = pool.map(_run_batch, args_list)

    # Merge results
    is_results  = []
    oos_results = []
    for br in batch_results:
        is_results.extend(br["is"])
        oos_results.extend(br["oos"])

    print(f"Completed {len(is_results)} IS and {len(oos_results)} OOS runs.")

    # Save JSON
    with open(OUT_DIR / "ea011v2_is_results.json", "w") as f:
        json.dump(is_results, f, indent=2)
    with open(OUT_DIR / "ea011v2_oos_results.json", "w") as f:
        json.dump(oos_results, f, indent=2)

    print(f"Saved: {OUT_DIR / 'ea011v2_is_results.json'}")
    print(f"Saved: {OUT_DIR / 'ea011v2_oos_results.json'}")

    # ---------------------------------------------------------------------------
    # Analysis
    # ---------------------------------------------------------------------------

    is_by_params = {str(r["params"]): r for r in is_results}

    # ── Density winners ───────────────────────────────────────────────────────
    print("\n=== OOS Density Winners (trades>=20, PF>=8.0, decay<30%) ===")
    density_winners = []
    for r in oos_results:
        if r["trades"] < 20 or r["pf"] < 8.0:
            continue
        key = str(r["params"])
        if key in is_by_params:
            is_pf = is_by_params[key]["pf"]
            decay = (r["pf"] - is_pf) / is_pf * 100
            if decay > -30.0:
                r["_is_pf"] = is_pf
                r["_decay"] = decay
                density_winners.append(r)

    density_winners.sort(key=lambda r: (-r["pf"], -r["trades"]))

    if density_winners:
        for r in density_winners[:15]:
            print(
                f"  OOS PF={r['pf']:.2f} T={r['trades']:>3} WR={r['win_rate']:>5.1f}% "
                f"Net={r['net']:>7.2f} | IS PF={r['_is_pf']:.2f} decay={r['_decay']:+.1f}% "
                f"| {r['params']}"
            )
    else:
        print("  None found — fallback: PF>=6.0, trades>=15")
        fallback = []
        for r in oos_results:
            if r["trades"] < 15 or r["pf"] < 6.0:
                continue
            key = str(r["params"])
            if key in is_by_params:
                r["_is_pf"] = is_by_params[key]["pf"]
                r["_decay"] = (r["pf"] - r["_is_pf"]) / r["_is_pf"] * 100
                fallback.append(r)
        fallback.sort(key=lambda r: (-r["pf"], -r["trades"]))
        for r in fallback[:10]:
            print(
                f"  OOS PF={r['pf']:.2f} T={r['trades']:>3} WR={r['win_rate']:>5.1f}% "
                f"Net={r['net']:>7.2f} | IS PF={r['_is_pf']:.2f} decay={r['_decay']:+.1f}% "
                f"| {r['params']}"
            )

    # ── Top 10 OOS overall ────────────────────────────────────────────────────
    print("\n=== Top 10 OOS by PF (all trade counts) ===")
    for r in sorted(oos_results, key=lambda r: (-r["pf"], -r["net"]))[:10]:
        key   = str(r["params"])
        is_pf = is_by_params[key]["pf"] if key in is_by_params else float("nan")
        decay = (r["pf"] - is_pf) / is_pf * 100
        print(
            f"  OOS PF={r['pf']:.2f} T={r['trades']:>3} WR={r['win_rate']:>5.1f}% "
            f"Net={r['net']:>7.2f} | IS PF={is_pf:.2f} decay={decay:+.1f}% "
            f"| {r['params']}"
        )

    # ── Gate isolation ────────────────────────────────────────────────────────
    print("\n=== Gate Isolation: avg OOS PF by flag combo (trades>=10) ===")
    for combo in itertools.product([True, False], [True, False], [True, False]):
        rows = [
            r for r in oos_results
            if r["params"]["req_poc_mig"]    == combo[0]
            and r["params"]["req_trendline"]  == combo[1]
            and r["params"]["req_no_exhaust"] == combo[2]
            and r["trades"] >= 10
        ]
        if not rows:
            continue
        avg_pf = sum(r["pf"] for r in rows) / len(rows)
        avg_t  = sum(r["trades"] for r in rows) / len(rows)
        print(
            f"  poc={str(combo[0])[0]} tl={str(combo[1])[0]} exhaust={str(combo[2])[0]}"
            f"  → avg OOS PF={avg_pf:.2f}  avg T={avg_t:.0f}  n={len(rows)}"
        )

    # ── score_threshold effect ────────────────────────────────────────────────
    print("\n=== score_threshold effect (avg OOS PF, trades>=10) ===")
    for thresh in [0, 40]:
        rows = [
            r for r in oos_results
            if r["params"]["score_threshold"] == thresh and r["trades"] >= 10
        ]
        if rows:
            avg_pf = sum(r["pf"] for r in rows) / len(rows)
            avg_t  = sum(r["trades"] for r in rows) / len(rows)
            print(f"  score_threshold={thresh:>2}  avg PF={avg_pf:.2f}  avg T={avg_t:.0f}  n={len(rows)}")
