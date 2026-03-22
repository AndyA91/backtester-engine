"""
Master sweep — run ALL Renko strategies on their native instruments.
Outputs a ranked leaderboard sorted by profit factor (min 60 trades).

Usage:
    cd renko
    python master_sweep.py
"""

import importlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent / "strategies"))

from renko.runner import sweep, IS_START, IS_END

# All strategy modules grouped by instrument
STRATEGIES = [
    # --- EURUSD (default 0.0004 brick) ---
    "r001_brick_count",
    "r002_reversal",
    "r004_candle_adx",
    "r005_master",
    "r006_alternation",
    "r007_combined",
    "r008_candle_adx",
    "r009_exit_study",
    "r010_psar_gate",
    "r011_0005_optimize",
    "r012_macd_lc",
    "r013_chop_gate",
    "r014_fisher_adx",
    "r015_squeeze",
    # --- EURAUD (0.0006 / 0.0007 brick) ---
    "ea001_baseline",
    "ea002_gate_sweep",
    "ea003_confluence_master",
    "ea003r_combined_confluence",
    "ea004_band_runner",
    "ea005_va_breakout",
    "ea006_distance_divergence",
    "ea009_institutional_reversal",
    "ea010_cyberpunk_momentum",
    "ea011_auction_breakout_pro",
    "ea011_v2_auction_champion",
    "ea012_napoleon_value",
    "ea013_adaptive_escgo",
    "ea014_alpha_sniper",
    "ea015_sto_reversal",
    "ea016_mcp_ddl_cross",
    "ea017_baseline_0007",
    "ea018_vp_div_0007",
    "ea019_fisher_adx",
    # --- GBPJPY (0.05 brick) ---
    "gj001_brick_count",
    "gj007_combined",
    "gj008_candle_adx",
    "gj009_session_adx",
    "gj010_macd_lc",
    "gj011_sto_tso",
    "gj012_fisher_adx",
    "gj013_trail_exit",
    "gj014_squeeze",
    # --- GBPUSD (0.0004 brick) ---
    "gu001_brick_count",
    # --- USDJPY (0.05 brick) ---
    "uj001_brick_count",
    # --- BTCUSD (150 brick) ---
    "btc001_fisher_adx",
    "btc002_brick_count",
    "btc003_squeeze",
]


def main():
    leaderboard = []
    failed = []

    print(f"\n{'#'*70}")
    print(f"  RENKO MASTER SWEEP — {len(STRATEGIES)} strategies")
    print(f"  IS Period: {IS_START} to {IS_END}")
    print(f"{'#'*70}\n")

    for i, name in enumerate(STRATEGIES, 1):
        print(f"\n[{i}/{len(STRATEGIES)}] Running {name} ...")
        t0 = time.time()
        try:
            results, mod = sweep(name, IS_START, IS_END, verbose=False)
            elapsed = time.time() - t0

            if not results:
                print(f"  -> No results (empty grid?)")
                failed.append((name, "no results"))
                continue

            best = results[0]  # already sorted by rank_key
            desc = getattr(mod, "DESCRIPTION", "")
            renko_file = getattr(mod, "RENKO_FILE", "EURUSD 0.0004")

            # Extract instrument from RENKO_FILE
            if "BTCUSD" in str(renko_file):
                instrument = "BTCUSD"
            elif "EURAUD" in str(renko_file):
                instrument = "EURAUD"
            elif "GBPJPY" in str(renko_file):
                instrument = "GBPJPY"
            elif "GBPUSD" in str(renko_file):
                instrument = "GBPUSD"
            elif "USDJPY" in str(renko_file):
                instrument = "USDJPY"
            else:
                instrument = "EURUSD"

            entry = {
                "strategy": name,
                "instrument": instrument,
                "pf": best["pf"],
                "net": best["net"],
                "trades": best["trades"],
                "win_rate": best["win_rate"],
                "max_dd_pct": best["max_dd_pct"],
                "expectancy": best["expectancy"],
                "avg_wl": best["avg_wl"],
                "params": best["params"],
                "combos": len(results),
                "description": desc,
                "elapsed": elapsed,
            }
            leaderboard.append(entry)

            tag = "OK" if best["trades"] >= 60 else "LOW_T"
            pf_str = "INF" if best["pf"] > 1e10 else f"{best['pf']:.3f}"
            print(f"  -> [{tag}] PF={pf_str} Net={best['net']:.2f} "
                  f"T={best['trades']} WR={best['win_rate']:.1f}% "
                  f"DD={best['max_dd_pct']:.2f}% ({elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  -> FAILED: {e} ({elapsed:.1f}s)")
            failed.append((name, str(e)))

    # --- LEADERBOARD ---
    # Sort: qualified first, then by PF descending
    qualified = [e for e in leaderboard if e["trades"] >= 60]
    unqualified = [e for e in leaderboard if e["trades"] < 60]
    qualified.sort(key=lambda e: (e["pf"] if e["pf"] < 1e10 else 1e10, e["net"]), reverse=True)
    unqualified.sort(key=lambda e: (e["pf"] if e["pf"] < 1e10 else 1e10, e["net"]), reverse=True)

    print(f"\n\n{'='*90}")
    print(f"  RENKO LEADERBOARD — IS Period ({IS_START} to {IS_END})")
    print(f"  Qualified: {len(qualified)} | Unqualified (<60 trades): {len(unqualified)} | Failed: {len(failed)}")
    print(f"{'='*90}")

    header = (f"{'Rank':<5} {'Strategy':<35} {'Instr':<8} {'PF':>7} {'Net':>10} "
              f"{'Trades':>7} {'WR%':>6} {'DD%':>7} {'Exp':>8} {'AvgW/L':>7}")
    print(f"\n{header}")
    print("-" * len(header))

    for rank, e in enumerate(qualified, 1):
        pf_str = "INF" if e["pf"] > 1e10 else f"{e['pf']:.3f}"
        print(f"{rank:<5} {e['strategy']:<35} {e['instrument']:<8} {pf_str:>7} "
              f"{e['net']:>10.2f} {e['trades']:>7} {e['win_rate']:>5.1f}% "
              f"{e['max_dd_pct']:>6.2f}% {e['expectancy']:>8.3f} {e['avg_wl']:>7.3f}")

    if unqualified:
        print(f"\n--- Unqualified (<60 trades) ---")
        for e in unqualified:
            pf_str = "INF" if e["pf"] > 1e10 else f"{e['pf']:.3f}"
            print(f"  {e['strategy']:<35} {e['instrument']:<8} {pf_str:>7} "
                  f"{e['net']:>10.2f} {e['trades']:>7} {e['win_rate']:>5.1f}% "
                  f"{e['max_dd_pct']:>6.2f}%")

    if failed:
        print(f"\n--- Failed ---")
        for name, err in failed:
            print(f"  {name}: {err}")

    # --- Write markdown leaderboard ---
    md_path = ROOT / "RENKO_LEADERBOARD.md"
    with open(md_path, "w") as f:
        f.write(f"# Renko Strategy Leaderboard\n\n")
        f.write(f"**IS Period:** {IS_START} to {IS_END}  \n")
        f.write(f"**Min trades:** 60  \n")
        f.write(f"**Strategies tested:** {len(STRATEGIES)}  \n")
        f.write(f"**Qualified:** {len(qualified)} | **Unqualified:** {len(unqualified)} | **Failed:** {len(failed)}\n\n")

        f.write(f"## Qualified (>= 60 trades)\n\n")
        f.write(f"| Rank | Strategy | Instrument | PF | Net | Trades | WR% | MaxDD% | Expectancy | AvgW/L | Best Params |\n")
        f.write(f"|------|----------|------------|---:|----:|-------:|----:|-------:|-----------:|-------:|-------------|\n")
        for rank, e in enumerate(qualified, 1):
            pf_str = "INF" if e["pf"] > 1e10 else f"{e['pf']:.3f}"
            params_str = ", ".join(f"{k}={v}" for k, v in e["params"].items())
            f.write(f"| {rank} | {e['strategy']} | {e['instrument']} | {pf_str} | "
                    f"{e['net']:.2f} | {e['trades']} | {e['win_rate']:.1f} | "
                    f"{e['max_dd_pct']:.2f} | {e['expectancy']:.3f} | {e['avg_wl']:.3f} | "
                    f"{params_str} |\n")

        if unqualified:
            f.write(f"\n## Unqualified (< 60 trades)\n\n")
            f.write(f"| Strategy | Instrument | PF | Net | Trades | WR% | MaxDD% |\n")
            f.write(f"|----------|------------|---:|----:|-------:|----:|-------:|\n")
            for e in unqualified:
                pf_str = "INF" if e["pf"] > 1e10 else f"{e['pf']:.3f}"
                f.write(f"| {e['strategy']} | {e['instrument']} | {pf_str} | "
                        f"{e['net']:.2f} | {e['trades']} | {e['win_rate']:.1f} | "
                        f"{e['max_dd_pct']:.2f} |\n")

        if failed:
            f.write(f"\n## Failed\n\n")
            for name, err in failed:
                f.write(f"- **{name}**: {err}\n")

    print(f"\nLeaderboard written to {md_path}")


if __name__ == "__main__":
    main()
