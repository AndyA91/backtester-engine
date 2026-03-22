import argparse
import importlib
import itertools
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AI_CONTEXT = ROOT / "ai_context"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "renko" / "strategies"))

from renko import runner

IS_START = "2023-07-20"
IS_END = "2025-09-30"
OOS_START = "2025-10-01"
OOS_END = "2026-03-17"

EA008_REF = {
    "is_pf": 15.06,
    "is_trades": 269,
    "oos_pf": 12.79,
    "oos_trades": 63,
    "decay_pct": 15.0,
    "verdict": "LIVE",
}

STRATEGIES = [
    {
        "label": "EA003R",
        "module": "ea003r_combined_confluence",
        "is_json": AI_CONTEXT / "ea003r_is_results.json",
        "oos_json": AI_CONTEXT / "ea003r_oos_results.json",
    },
    {
        "label": "EA004",
        "module": "ea004_band_runner",
        "is_json": AI_CONTEXT / "ea004_is_results.json",
        "oos_json": AI_CONTEXT / "ea004_oos_results.json",
    },
    {
        "label": "EA005",
        "module": "ea005_va_breakout",
        "is_json": AI_CONTEXT / "ea005_is_results.json",
        "oos_json": AI_CONTEXT / "ea005_oos_results.json",
    },
    {
        "label": "EA006",
        "module": "ea006_distance_divergence",
        "is_json": AI_CONTEXT / "ea006_is_results.json",
        "oos_json": AI_CONTEXT / "ea006_oos_results.json",
    },
]


def pf_to_json(value):
    if value is None:
        return 0.0
    if math.isinf(value):
        return 999999
    return float(value)


def num_to_json(value):
    if value is None:
        return 0.0
    if isinstance(value, float) and math.isinf(value):
        return 999999
    return float(value)


def params_product(grid):
    keys = list(grid.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*grid.values())]


def result_sort_key(result):
    return (pf_to_json(result["pf"]), result["net"], result["trades"])


def serialize_result(result):
    return {
        "pf": pf_to_json(result["pf"]),
        "net": float(result["net"]),
        "trades": int(result["trades"]),
        "win_rate": float(result["win_rate"]),
        "max_dd_pct": float(result["max_dd_pct"]),
        "expectancy": float(result["expectancy"]),
        "avg_wl": num_to_json(result["avg_wl"]),
        "params": result["params"],
    }


def run_period(df, mod, combos, start, end):
    commission_pct = getattr(mod, "COMMISSION_PCT", 0.0046)
    initial_capital = getattr(mod, "INITIAL_CAPITAL", 1000.0)
    results = []
    for idx, params in enumerate(combos, start=1):
        result = runner.run_single(
            df,
            mod.generate_signals,
            params,
            start,
            end,
            commission_pct=commission_pct,
            initial_capital=initial_capital,
        )
        results.append(result)
        print(
            f"  [{idx:>3}/{len(combos)}] "
            f"PF={pf_to_json(result['pf']):>8.4f} "
            f"Net={result['net']:>8.2f} "
            f"T={result['trades']:>4} "
            f"WR={result['win_rate']:>5.1f}% "
            f"| {params}"
        )
    results.sort(key=result_sort_key, reverse=True)
    return results


def save_results(path, results):
    data = [serialize_result(result) for result in results]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_results(path):
    return json.loads(path.read_text(encoding="utf-8"))


def format_params(params):
    return ", ".join(f"{key}={value}" for key, value in params.items())


def filter_rankable(results, min_trades):
    return [row for row in results if row["trades"] >= min_trades]


def top_rows(results, min_trades, n=5):
    ranked = sorted(filter_rankable(results, min_trades), key=result_sort_key, reverse=True)
    return ranked[:n]


def match_by_params(results, params):
    for row in results:
        if row["params"] == params:
            return row
    return None


def decay_pct(is_pf, oos_pf):
    if not is_pf:
        return 0.0
    return (is_pf - oos_pf) / is_pf * 100.0


def is_practical_pf(value):
    return value not in (None, 999999) and value > 0


def group_summary(results, key_name):
    groups = {}
    group_count = {}
    for row in results:
        key = row["params"][key_name]
        groups.setdefault(key, []).append(row)
        group_count[key] = group_count.get(key, 0) + 1

    summaries = []
    for key in sorted(groups):
        rows = groups[key]
        pfs = [row["pf"] for row in rows if is_practical_pf(row["pf"])]
        active = sum(1 for row in rows if row["trades"] > 0)
        summaries.append(
            {
                "key": key,
                "avg_trades": sum(row["trades"] for row in rows) / len(rows),
                "avg_pf": (sum(pfs) / len(pfs)) if pfs else 0.0,
                "best_pf": max(pfs) if pfs else 0.0,
                "active": active,
                "total": group_count[key],
            }
        )
    return summaries


def pairwise_bool_effect(results, bool_key):
    grouped = {}
    for row in results:
        params = dict(row["params"])
        flag = params.pop(bool_key)
        grouped.setdefault(tuple(params.items()), {})[flag] = row

    deltas = []
    for pair in grouped.values():
        if False in pair and True in pair:
            delta = pair[True]["pf"] - pair[False]["pf"]
            deltas.append((pair[False], pair[True], delta))
    return deltas


def render_top_table(rows):
    lines = [
        "| Rank | PF | Net | Trades | Win Rate | Params |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    if not rows:
        lines.append("| - | - | - | - | - | No combos met the trade threshold |")
        return "\n".join(lines)
    for idx, row in enumerate(rows, start=1):
        lines.append(
            f"| {idx} | {row['pf']:.4f} | {row['net']:.2f} | {row['trades']} | "
            f"{row['win_rate']:.1f}% | {format_params(row['params'])} |"
        )
    return "\n".join(lines)


def render_decay_table(is_rows, oos_results):
    lines = [
        "| Params | IS PF | OOS PF | Decay | OOS Trades |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    if not is_rows:
        lines.append("| No qualifying IS combos | - | - | - | - |")
        return "\n".join(lines)
    for row in is_rows:
        oos_row = match_by_params(oos_results, row["params"])
        oos_pf = oos_row["pf"] if oos_row else 0.0
        oos_trades = oos_row["trades"] if oos_row else 0
        lines.append(
            f"| {format_params(row['params'])} | {row['pf']:.4f} | {oos_pf:.4f} | "
            f"{decay_pct(row['pf'], oos_pf):.1f}% | {oos_trades} |"
        )
    return "\n".join(lines)


def render_group_table(title_key, rows):
    lines = [
        f"| {title_key} | Avg Trades | Avg PF | Best PF | Active Combos |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['key']} | {row['avg_trades']:.1f} | {row['avg_pf']:.4f} | "
            f"{row['best_pf']:.4f} | {row['active']}/{row['total']} |"
        )
    return "\n".join(lines)


def best_qualifying(results, min_trades):
    ranked = top_rows(results, min_trades, n=1)
    return ranked[0] if ranked else None


def summarize_cooldown(results):
    return render_group_table("cooldown", group_summary(results, "cooldown"))


def summarize_lookback(results):
    return render_group_table("lookback", group_summary(results, "lookback"))


def summarize_n_inside(results):
    return render_group_table("n_inside", group_summary(results, "n_inside"))


def summarize_n_bricks(results):
    return render_group_table("n_bricks", group_summary(results, "n_bricks"))


def summarize_require_both(results):
    rows = group_summary(results, "require_both_divs")
    lines = [
        "| require_both_divs | Avg Trades | Avg PF | Best PF | Active Combos |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['key']} | {row['avg_trades']:.1f} | {row['avg_pf']:.4f} | "
            f"{row['best_pf']:.4f} | {row['active']}/{row['total']} |"
        )
    return "\n".join(lines)


def verdict_line(best_oos):
    if not best_oos:
        return "No viable OOS candidate"
    if best_oos["pf"] >= EA008_REF["oos_pf"] and best_oos["trades"] >= 20:
        return "Beats EA008 benchmark"
    if best_oos["pf"] >= 8.0 and best_oos["trades"] >= 20:
        return "Viable challenger"
    if best_oos["pf"] >= 5.0 and best_oos["trades"] >= 20:
        return "Promising but below champion"
    return "Reject for now"


def build_ea003r_section(all_results):
    is_results = all_results["EA003R"]["is"]
    oos_results = all_results["EA003R"]["oos"]
    baseline_is = load_results(AI_CONTEXT / "ea003_is_results.json")
    baseline_oos = load_results(AI_CONTEXT / "ea003_oos_results.json")

    is_top = top_rows(is_results, 30)
    oos_top = top_rows(oos_results, 20)
    baseline_best_is = best_qualifying(baseline_is, 30)
    baseline_best_oos = best_qualifying(baseline_oos, 20)
    ea003r_best_is = best_qualifying(is_results, 30)
    ea003r_best_oos = best_qualifying(oos_results, 20)

    compare_lines = []
    if baseline_best_is and ea003r_best_is:
        compare_lines.append(
            f"EA003 baseline best qualifying IS PF was {baseline_best_is['pf']:.4f} "
            f"on {baseline_best_is['trades']} trades; EA003R reached "
            f"{ea003r_best_is['pf']:.4f} on {ea003r_best_is['trades']} trades."
        )
    if baseline_best_oos and ea003r_best_oos:
        compare_lines.append(
            f"EA003 OOS benchmark from the saved baseline file is PF {baseline_best_oos['pf']:.4f} "
            f"on {baseline_best_oos['trades']} trades. EA003R best qualifying OOS result is "
            f"PF {ea003r_best_oos['pf']:.4f} on {ea003r_best_oos['trades']} trades, "
            f"which is {'better' if ea003r_best_oos['pf'] > baseline_best_oos['pf'] else 'worse'} "
            f"by {abs(ea003r_best_oos['pf'] - baseline_best_oos['pf']):.4f} PF points."
        )
        compare_lines.append(
            f"Against the EA008 OOS benchmark PF {EA008_REF['oos_pf']:.2f}, "
            f"EA003R trails by {EA008_REF['oos_pf'] - ea003r_best_oos['pf']:.4f} PF points."
        )

    return "\n".join(
        [
            "## 1. EA003R - Combined Confluence",
            "",
            "### IS top 5 by PF (trades >= 30)",
            render_top_table(is_top),
            "",
            "### OOS top 5 by PF (trades >= 20)",
            render_top_table(oos_top),
            "",
            "### IS -> OOS PF decay for IS top 5",
            render_decay_table(is_top, oos_results),
            "",
            "### Effect of min_confluence on count and PF",
            render_group_table("min_confluence", group_summary(oos_results, "min_confluence")),
            "",
            "### EA003R vs EA003 / EA008",
            *compare_lines,
        ]
    )


def build_ea004_section(all_results):
    is_results = all_results["EA004"]["is"]
    oos_results = all_results["EA004"]["oos"]
    is_top = top_rows(is_results, 30)
    oos_top = top_rows(oos_results, 20)
    best_oos = best_qualifying(oos_results, 20)

    verdict = (
        "Raff-band mean-reversion shows a usable structural edge."
        if best_oos and best_oos["pf"] >= 8.0 and best_oos["trades"] >= 20
        else "Raff-band mean-reversion does not clear the minimum viable OOS bar."
    )

    return "\n".join(
        [
            "## 2. EA004 - Band Runner",
            "",
            "### IS top 5 by PF (trades >= 30)",
            render_top_table(is_top),
            "",
            "### OOS top 5 by PF (trades >= 20)",
            render_top_table(oos_top),
            "",
            "### IS -> OOS PF decay for IS top 5",
            render_decay_table(is_top, oos_results),
            "",
            "### Effect of lookback on count and PF",
            summarize_lookback(oos_results),
            "",
            "### Effect of cooldown on OOS performance",
            summarize_cooldown(oos_results),
            "",
            "### Verdict",
            verdict,
        ]
    )


def build_ea005_section(all_results):
    is_results = all_results["EA005"]["is"]
    oos_results = all_results["EA005"]["oos"]
    is_top = top_rows(is_results, 30)
    oos_top = top_rows(oos_results, 20)
    best_oos = best_qualifying(oos_results, 20)

    verdict = (
        "Value-area acceptance plus breakout captures a viable momentum edge."
        if best_oos and best_oos["pf"] >= 8.0 and best_oos["trades"] >= 20
        else "Value-area breakout does not show a strong enough OOS momentum edge yet."
    )

    return "\n".join(
        [
            "## 3. EA005 - Value Area Breakout",
            "",
            "### IS top 5 by PF (trades >= 30)",
            render_top_table(is_top),
            "",
            "### OOS top 5 by PF (trades >= 20)",
            render_top_table(oos_top),
            "",
            "### IS -> OOS PF decay for IS top 5",
            render_decay_table(is_top, oos_results),
            "",
            "### Effect of n_inside on count and PF",
            summarize_n_inside(oos_results),
            "",
            "### Verdict",
            verdict,
        ]
    )


def build_ea006_section(all_results):
    is_results = all_results["EA006"]["is"]
    oos_results = all_results["EA006"]["oos"]
    is_top = top_rows(is_results, 30)
    oos_top = top_rows(oos_results, 20)
    best_oos = best_qualifying(oos_results, 20)

    effect_rows = pairwise_bool_effect(oos_results, "require_both_divs")
    helps = sum(1 for _, _, delta in effect_rows if delta > 1e-9)
    hurts = sum(1 for _, _, delta in effect_rows if delta < -1e-9)
    ties = sum(1 for _, _, delta in effect_rows if abs(delta) <= 1e-9)
    avg_delta = sum(delta for _, _, delta in effect_rows) / len(effect_rows) if effect_rows else 0.0

    verdict = (
        "The three-factor gate produces a stable edge."
        if best_oos and best_oos["pf"] >= 8.0 and best_oos["trades"] >= 20
        else "The three-factor gate is not stable enough OOS."
    )

    return "\n".join(
        [
            "## 4. EA006 - Distance Divergence",
            "",
            "### IS top 5 by PF (trades >= 30)",
            render_top_table(is_top),
            "",
            "### OOS top 5 by PF (trades >= 20)",
            render_top_table(oos_top),
            "",
            "### IS -> OOS PF decay for IS top 5",
            render_decay_table(is_top, oos_results),
            "",
            "### Effect of require_both_divs on OOS PF and trade count",
            summarize_require_both(oos_results),
            "",
            f"`require_both_divs=True` helps in {helps}/{len(effect_rows)} matched pairs, "
            f"hurts in {hurts}/{len(effect_rows)}, ties in {ties}/{len(effect_rows)}. "
            f"Average PF delta vs `False`: {avg_delta:+.4f}.",
            "",
            "### Effect of n_bricks on OOS PF",
            summarize_n_bricks(oos_results),
            "",
            "### Verdict",
            verdict,
        ]
    )


def build_summary_table(all_results):
    lines = [
        "| Strategy | IS PF (best) | IS T (best) | OOS PF (best) | OOS T (best) | IS->OOS Decay | Verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for label in ("EA003R", "EA004", "EA005", "EA006"):
        is_best = best_qualifying(all_results[label]["is"], 30)
        oos_best = best_qualifying(all_results[label]["oos"], 20)
        if is_best:
            matched_oos = match_by_params(all_results[label]["oos"], is_best["params"])
            decay = decay_pct(is_best["pf"], matched_oos["pf"] if matched_oos else 0.0)
            is_pf = f"{is_best['pf']:.4f}"
            is_trades = str(is_best["trades"])
        else:
            decay = None
            is_pf = "n/a"
            is_trades = "n/a"
        if oos_best:
            oos_pf = f"{oos_best['pf']:.4f}"
            oos_trades = str(oos_best["trades"])
        else:
            oos_pf = "n/a"
            oos_trades = "n/a"
        decay_text = f"{decay:.1f}%" if decay is not None else "n/a"
        lines.append(
            f"| {label} | {is_pf} | {is_trades} | {oos_pf} | "
            f"{oos_trades} | {decay_text} | {verdict_line(oos_best)} |"
        )
    lines.append(
        f"| EA008 ref | {EA008_REF['is_pf']:.2f} | {EA008_REF['is_trades']} | "
        f"{EA008_REF['oos_pf']:.2f} | {EA008_REF['oos_trades']} | "
        f"{EA008_REF['decay_pct']:.0f}% | {EA008_REF['verdict']} |"
    )
    return "\n".join(lines)


def build_recommendations(all_results):
    oos_best = {
        label: best_qualifying(all_results[label]["oos"], 20)
        for label in ("EA003R", "EA004", "EA005", "EA006")
    }

    beat_ea008 = [
        f"{label} (PF {row['pf']:.4f}, {row['trades']} trades)"
        for label, row in oos_best.items()
        if row and row["pf"] >= EA008_REF["oos_pf"]
    ]
    viable = [
        f"{label} (PF {row['pf']:.4f}, {row['trades']} trades)"
        for label, row in oos_best.items()
        if row and row["pf"] >= 8.0 and row["trades"] >= 20
    ]
    investigate = [
        f"{label} (PF {row['pf']:.4f}, {row['trades']} trades)"
        for label, row in sorted(
            oos_best.items(),
            key=lambda item: (item[1]["pf"], item[1]["trades"]) if item[1] else (0, 0),
            reverse=True,
        )
        if row and row["pf"] >= 5.0 and row["trades"] >= 20
    ]
    reject = []
    for label, row in oos_best.items():
        if row and row["pf"] >= 5.0 and row["trades"] >= 20:
            continue
        if row:
            reject.append(f"{label} (PF {row['pf']:.4f}, {row['trades']} trades)")
        else:
            reject.append(f"{label} (no qualifying OOS combo)")

    return "\n".join(
        [
            "## 6. Recommendations",
            "",
            f"- Strategies beating the EA008 OOS PF {EA008_REF['oos_pf']:.2f} benchmark: "
            f"{', '.join(beat_ea008) if beat_ea008 else 'none.'}",
            f"- Strategies meeting the minimum viable bar (OOS PF >= 8.0 and >= 20 trades): "
            f"{', '.join(viable) if viable else 'none.'}",
            f"- Strategies to combine or investigate further: "
            f"{', '.join(investigate[:3]) if investigate else 'none beyond baseline research.'}",
            f"- Strategies to reject for now: {', '.join(reject) if reject else 'none.'}",
            "",
            "Note: pseudo-infinite PF rows from zero-loss micro-samples were preserved in JSON as `999999` "
            "per handoff, but excluded from group PF averages and thresholded rankings.",
        ]
    )


def write_analysis(all_results):
    sections = [
        "# Big Batch Analysis",
        "",
        build_ea003r_section(all_results),
        "",
        build_ea004_section(all_results),
        "",
        build_ea005_section(all_results),
        "",
        build_ea006_section(all_results),
        "",
        "## 5. Cross-Strategy Summary Table",
        build_summary_table(all_results),
        "",
        build_recommendations(all_results),
    ]
    (AI_CONTEXT / "big_batch_analysis.md").write_text("\n".join(sections), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()

    all_results = {}

    for spec in STRATEGIES:
        label = spec["label"]
        if not args.analysis_only:
            print(f"\n=== {label}: loading module {spec['module']} ===")
            mod = importlib.import_module(spec["module"])
            combos = params_product(mod.PARAM_GRID)
            df = runner.load_data(getattr(mod, "RENKO_FILE", None))
            print(f"{label}: running IS sweep with {len(combos)} combos")
            is_results = run_period(df, mod, combos, IS_START, IS_END)
            save_results(spec["is_json"], is_results)
            print(f"{label}: wrote {spec['is_json'].name}")

            print(f"{label}: running OOS sweep with {len(combos)} combos")
            oos_results = run_period(df, mod, combos, OOS_START, OOS_END)
            save_results(spec["oos_json"], oos_results)
            print(f"{label}: wrote {spec['oos_json'].name}")

        all_results[label] = {
            "is": load_results(spec["is_json"]),
            "oos": load_results(spec["oos_json"]),
        }

    write_analysis(all_results)
    print("\nWrote ai_context/big_batch_analysis.md")


if __name__ == "__main__":
    main()
