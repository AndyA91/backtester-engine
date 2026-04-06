#!/usr/bin/env python3
"""
wizard_btc_combine.py -- Combine v1 (7 strategies) + v2 (20 strategies) results,
rank by OOS performance, save final JSON + markdown winner report.

Output:
    ai_context/wizard_btc_full_sweep_results.json   -- all results merged & sorted
    ai_context/wizard_btc_winner_report.md          -- human-readable summary

Usage:
    python renko/wizard_btc_combine.py
"""

import json
import math
from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parent.parent

V1_FILE   = ROOT / "ai_context" / "wizard_btc_sweep_results.json"
V2_FILE   = ROOT / "ai_context" / "wizard_btc_sweep_v2_results.json"
OUT_JSON  = ROOT / "ai_context" / "wizard_btc_full_sweep_results.json"
OUT_MD    = ROOT / "ai_context" / "wizard_btc_winner_report.md"

MIN_OOS_TRADES = 10
LIVE_MIN_TRADES = 20          # for live recommendation
LIVE_MIN_TPD    = 0.3         # trades/day minimum for live
LIVE_MIN_WR     = 60.0        # win rate minimum

# Boosts lookup for each strategy name
BOOSTS = {
    # v1
    "ALPHA_TREND":      6722,
    "SSL_CHANNEL":      3939,
    "HHLL":             2494,
    "MACD_RELOAD":      7478,
    "WILDER_VOL":        895,
    "HALFTREND":       12106,
    "NRTR":             9887,
    # v2
    "THREE_COMMAS":    13718,
    "DOUBLE_TAP":       8145,
    "OPEN_CLOSE_CROSS": 5842,
    "QQE_CROSS":        3012,
    "DUAL_MA":          1568,
    "ELDER_RAY":        1425,
    "MADE_ATR":         1271,
    "DIVERGENCE":       1121,
    "INSIDE_BAR":        761,
    "OKX_MA":            644,
    "HLHB":              416,
    "COMBO_BULL_POWER":  416,
    "EMA_MA_CROSS":      386,
    "COMBO_CCI":         378,
    "DEMA_RSI":          290,
    "COMBO_BBB":         250,
    "COMBO_AO":          232,
    "COMBO_ATRR":        162,
    "COMBO_HLCH":        136,
    "COMBO_BEAR_POWER":  127,
}

DESCS = {
    "ALPHA_TREND":      "ATR+RSI adaptive band crossover (KivancOzbilgic)",
    "SSL_CHANNEL":      "SMA(high) vs SMA(low) direction flip (vdubus)",
    "HHLL":             "BB offset breakout mean-reversion (HPotter)",
    "MACD_RELOAD":      "MACD histogram zero cross (KivancOzbilgic)",
    "WILDER_VOL":       "ATR-based SAR system (LucF/Wilder)",
    "HALFTREND":        "Adaptive trend band ATR channels (everget)",
    "NRTR":             "Nick Rypock Trailing Reverse (everget)",
    "THREE_COMMAS":     "MA crossover bot signal (Bjorgum)",
    "DOUBLE_TAP":       "Double bottom pattern detection (Bjorgum)",
    "OPEN_CLOSE_CROSS": "EMA(close) > EMA(open) crossover (JustUncleL)",
    "QQE_CROSS":        "QQE RSI threshold channel exit (JustUncleL)",
    "DUAL_MA":          "EMA ribbon cross + close confirm (JustUncleL)",
    "ELDER_RAY":        "Bull Power = rolling_high - EMA (HPotter)",
    "MADE_ATR":         "Displaced EMA envelope + ATR stop (HPotter)",
    "DIVERGENCE":       "Bullish RSI divergence on pivot lows (Trendoscope)",
    "INSIDE_BAR":       "Inside bar momentum pattern (BacktestRookies)",
    "OKX_MA":           "Price dips below SMA then bounces (HPotter)",
    "HLHB":             "EMA5/10 cross + RSI50 cross (BacktestRookies)",
    "COMBO_BULL_POWER": "EMA20-trend + Bull Power oscillator (HPotter)",
    "EMA_MA_CROSS":     "EMA crosses above SMA (HPotter)",
    "COMBO_CCI":        "EMA20-trend + CCI fast/slow cross (HPotter)",
    "DEMA_RSI":         "DEMA + RSI smoothing crossover (RicardoSantos)",
    "COMBO_BBB":        "EMA20-trend + Bull-Bear Balance (HPotter)",
    "COMBO_AO":         "EMA20-trend + Awesome Oscillator (HPotter)",
    "COMBO_ATRR":       "EMA20-trend + ATR trailing stop flip (HPotter)",
    "COMBO_HLCH":       "EMA20-trend + HL/C histogram (HPotter)",
    "COMBO_BEAR_POWER": "EMA20-trend + Bear Power low (HPotter)",
}


def fmt_pf(pf):
    return "INF" if math.isinf(pf) else f"{pf:.2f}"


def load_results(path):
    with open(path) as f:
        return json.load(f)


def best_per_strategy(all_results):
    """Return best result per strategy (by OOS PF, min MIN_OOS_TRADES)."""
    valid = [r for r in all_results
             if "error" not in r and r.get("oos", {}).get("trades", 0) >= MIN_OOS_TRADES]
    by_strat = {}
    for r in valid:
        s = r["strategy"]
        pf = r["oos"]["pf"]
        prev = by_strat.get(s)
        if prev is None or pf > prev["oos"]["pf"]:
            by_strat[s] = r
    return by_strat


def main():
    print("Loading v1 results...")
    v1 = load_results(V1_FILE)
    print(f"  {len(v1)} entries")

    print("Loading v2 results...")
    v2 = load_results(V2_FILE)
    print(f"  {len(v2)} entries")

    all_results = v1 + v2
    print(f"  Total: {len(all_results)} entries\n")

    # Add strategy boosts to each result for reference
    for r in all_results:
        r["boosts"] = BOOSTS.get(r.get("strategy", ""), 0)

    # Sort all valid by OOS PF
    valid = [r for r in all_results
             if "error" not in r and r.get("oos", {}).get("trades", 0) >= MIN_OOS_TRADES]
    valid.sort(key=lambda x: x["oos"]["pf"], reverse=True)

    print(f"Valid configs (OOS >= {MIN_OOS_TRADES} trades): {len(valid)}")

    # Best per strategy
    best = best_per_strategy(all_results)
    # Sort best by OOS PF
    ranked_strategies = sorted(best.values(), key=lambda x: x["oos"]["pf"], reverse=True)

    # Print ranked summary
    print(f"\n{'='*110}")
    print(f"ALL 27 STRATEGIES RANKED BY BEST OOS PF")
    print(f"{'='*110}")
    print(f"{'Rank':<5} {'Strategy':<20} {'Boosts':>7}  {'Params':<30} {'Gate':<12} {'CD':>3}  "
          f"{'IS_PF':>7} {'IS_T':>5}  {'OOS_PF':>7} {'OOS_T':>5} {'OOS_WR':>6} {'t/d':>5} {'OOS_Net':>9}")
    print("-" * 110)

    for rank, r in enumerate(ranked_strategies, 1):
        s = r["strategy"]
        p = r.get("params", {})
        param_str = " ".join(f"{k}={v}" for k, v in p.items()) if p else "-"
        if len(param_str) > 28:
            param_str = param_str[:25] + "..."
        boosts = BOOSTS.get(s, 0)
        pf_str = fmt_pf(r["oos"]["pf"])
        print(f"  {rank:<3} {s:<20} {boosts:>7}  {param_str:<30} {r['gate']:<12} {r['cooldown']:>3}  "
              f"{r['is']['pf']:>7.2f} {r['is']['trades']:>5}  "
              f"{pf_str:>7} {r['oos']['trades']:>5} {r['oos']['wr']:>5.1f}% "
              f"{r['oos']['tpd']:>5.1f} {r['oos']['net']:>9.2f}")

    # Live candidates
    live_candidates = [r for r in ranked_strategies
                       if r["oos"]["trades"] >= LIVE_MIN_TRADES
                       and r["oos"]["tpd"] >= LIVE_MIN_TPD
                       and r["oos"]["wr"] >= LIVE_MIN_WR]

    print(f"\n{'='*80}")
    print(f"LIVE CANDIDATES (OOS: T>={LIVE_MIN_TRADES}, t/d>={LIVE_MIN_TPD}, WR>={LIVE_MIN_WR}%)")
    print(f"{'='*80}")
    if live_candidates:
        for i, r in enumerate(live_candidates, 1):
            s = r["strategy"]
            p = r.get("params", {})
            param_str = " ".join(f"{k}={v}" for k, v in p.items()) if p else "-"
            boosts = BOOSTS.get(s, 0)
            pf_str = fmt_pf(r["oos"]["pf"])
            print(f"  {i}. {s} (boosts={boosts})")
            print(f"     Params: {param_str}")
            print(f"     Gate={r['gate']} CD={r['cooldown']}")
            print(f"     IS:  PF={r['is']['pf']:.2f} T={r['is']['trades']} WR={r['is']['wr']:.1f}%")
            print(f"     OOS: PF={pf_str} T={r['oos']['trades']} WR={r['oos']['wr']:.1f}% "
                  f"t/d={r['oos']['tpd']:.1f} Net=${r['oos']['net']:.2f}")
            print()
    else:
        print("  No strategies meet all live criteria.")

    # Save merged JSON
    out_data = {
        "meta": {
            "date": str(date.today()),
            "v1_count": len(v1),
            "v2_count": len(v2),
            "total_combos": len(all_results),
            "valid_combos": len(valid),
            "strategies_tested": len(best),
        },
        "ranked_best_per_strategy": ranked_strategies,
        "top50_all_configs": valid[:50],
        "all_results": all_results,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out_data, f, indent=2, default=str)
    print(f"\nSaved full results to {OUT_JSON}")

    # Write markdown winner report
    _write_md_report(ranked_strategies, live_candidates, all_results, valid)
    print(f"Saved winner report to {OUT_MD}")

    return ranked_strategies, live_candidates


def _write_md_report(ranked, live_candidates, all_results, valid):
    today = date.today()
    winner = live_candidates[0] if live_candidates else ranked[0] if ranked else None

    lines = []
    lines.append("# Wizard BTC Sweep — Full Results & Winner Report")
    lines.append(f"\n**Date:** {today}  |  **Data:** BTCUSD $150 Renko  |  **IS:** 2024-06-04 to 2025-09-30  |  **OOS:** 2025-10-01 to 2026-03-19 (sealed)")
    lines.append(f"\n**Scope:** 30 scraped Pine Wizard strategies. 7 ported in v1, 20 in v2 (5 skipped as incompatible with long-only Renko framework).")
    lines.append(f"\n**Total combos tested:** {len(all_results)} (27 strategies × params × 4 gates × 4 cooldowns)")
    lines.append(f"\n---")

    # Winner box
    if winner:
        s = winner["strategy"]
        p = winner.get("params", {})
        param_str = " ".join(f"{k}={v}" for k, v in p.items()) if p else "-"
        pf_str = fmt_pf(winner["oos"]["pf"])
        lines.append(f"\n## WINNER: {s}")
        lines.append(f"\n> {DESCS.get(s, '')} — {BOOSTS.get(s, 0):,} boosts")
        lines.append(f"\n| Metric | IS | OOS |")
        lines.append("|--------|-----|-----|")
        lines.append(f"| Profit Factor | {winner['is']['pf']:.2f} | {pf_str} |")
        lines.append(f"| Trades | {winner['is']['trades']} | {winner['oos']['trades']} |")
        lines.append(f"| Win Rate | {winner['is']['wr']:.1f}% | {winner['oos']['wr']:.1f}% |")
        lines.append(f"| Net Profit ($20/trade) | ${winner['is']['net']:.2f} | ${winner['oos']['net']:.2f} |")
        lines.append(f"| Trades/Day (OOS) | — | {winner['oos']['tpd']:.1f} |")
        lines.append(f"\n**Config:** Params={param_str} | Gate={winner['gate']} | CD={winner['cooldown']}")
        if not live_candidates:
            lines.append(f"\n> ⚠️ No strategy met all live criteria (T≥{LIVE_MIN_TRADES}, t/d≥{LIVE_MIN_TPD}, WR≥{LIVE_MIN_WR}%). Best available shown above.")
        lines.append(f"\n---")

    # All strategies ranked
    lines.append(f"\n## All 27 Strategies Ranked by OOS Profit Factor")
    lines.append(f"\n_Best config per strategy (min {MIN_OOS_TRADES} OOS trades). Long-only, first-down-brick exit._")
    lines.append(f"\n| Rank | Strategy | Boosts | Gate | CD | IS PF | IS T | OOS PF | OOS T | OOS WR | t/d | OOS Net |")
    lines.append("|------|----------|--------|------|-----|-------|------|--------|-------|--------|-----|---------|")

    for rank, r in enumerate(ranked, 1):
        s = r["strategy"]
        boosts = BOOSTS.get(s, 0)
        pf_str = fmt_pf(r["oos"]["pf"])
        badge = " 🏆" if rank == 1 and live_candidates else (" ⭐" if r in live_candidates else "")
        lines.append(
            f"| {rank} | **{s}**{badge} | {boosts:,} | {r['gate']} | {r['cooldown']} "
            f"| {r['is']['pf']:.2f} | {r['is']['trades']} "
            f"| {pf_str} | {r['oos']['trades']} | {r['oos']['wr']:.1f}% "
            f"| {r['oos']['tpd']:.1f} | ${r['oos']['net']:.2f} |"
        )

    lines.append(f"\n---")

    # Live candidates detail
    lines.append(f"\n## Live Candidates (T≥{LIVE_MIN_TRADES}, t/d≥{LIVE_MIN_TPD}, WR≥{LIVE_MIN_WR}%)")
    if live_candidates:
        for i, r in enumerate(live_candidates, 1):
            s = r["strategy"]
            p = r.get("params", {})
            param_str = " ".join(f"{k}={v}" for k, v in p.items()) if p else "-"
            boosts = BOOSTS.get(s, 0)
            pf_str = fmt_pf(r["oos"]["pf"])
            lines.append(f"\n### #{i} {s} ({boosts:,} boosts)")
            lines.append(f"\n{DESCS.get(s, '')}")
            lines.append(f"\n- **Params:** {param_str}")
            lines.append(f"- **Gate:** {r['gate']} | **Cooldown:** {r['cooldown']}")
            lines.append(f"- **IS:** PF={r['is']['pf']:.2f}, T={r['is']['trades']}, WR={r['is']['wr']:.1f}%")
            lines.append(f"- **OOS:** PF={pf_str}, T={r['oos']['trades']}, WR={r['oos']['wr']:.1f}%, t/d={r['oos']['tpd']:.1f}, Net=${r['oos']['net']:.2f}")
    else:
        lines.append(f"\nNo strategies met all strict live criteria. Relax WR/t/d threshold or use best available.")

    lines.append(f"\n---")

    # Skipped strategies note
    lines.append(f"\n## Skipped Strategies (5)")
    lines.append(f"\n| Strategy | Author | Reason |")
    lines.append("|----------|--------|--------|")
    lines.append("| Backtest Adapter | jdehorty | Framework helper, not a trading signal |")
    lines.append("| Grid Like Strategy | alexgrover | Grid system, incompatible with Renko long-only |")
    lines.append("| Grid System w/ Fake Martingale | alexgrover | Grid/martingale, incompatible |")
    lines.append("| Average Down | BacktestRookies | Pyramiding strategy, incompatible |")
    lines.append("| Tutorial - Adding Sessions | RicardoSantos | Educational only, no signal logic |")

    lines.append(f"\n---")
    lines.append(f"\n## Methodology")
    lines.append(f"\n- **Long-only** (BTC bias): no shorts")
    lines.append(f"- **Exit:** first down-brick (universal — proven optimal in BTC research)")
    lines.append(f"- **Gates:** none | PSAR | ADX≥25 | PSAR+ADX25 (applied externally)")
    lines.append(f"- **Cooldowns:** 3, 5, 10, 20 bars")
    lines.append(f"- **Commission:** 0.0046% per side")
    lines.append(f"- **Position size:** $20/trade cash on $1,000 capital")
    lines.append(f"- **IS/OOS split:** Jun 2024–Sep 2025 / Oct 2025–Mar 2026")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
