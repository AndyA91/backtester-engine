"""Phase 5 backtest sweep runner — EA017 + EA018 + GJ011.

Strategies:
  EA017 — EURAUD R007 baseline on 0.0007 brick size (12 combos)
  EA018 — EURAUD EA008 gates (VP+div+session) on 0.0007 brick size (96 combos)
  GJ011 — GBPJPY GJ008 base + sto_tso + macd_lc optional gates (48 combos)

Benchmarks:
  EURAUD: EA008 OOS PF 10.62 (reference; note: different brick size / IS dates for EA017)
  GBPJPY: GJ008 OOS PF 21.33

Outputs:
  ai_context/ea_phase5_results.json
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import itertools
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent

for path in (ROOT, ROOT / "renko" / "strategies"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from engine import BacktestConfig, run_backtest_long_short
from renko.data import load_renko_export
from renko.indicators import add_renko_indicators

FIXED_QTY_VALUE = 1000.0

STRATEGY_SPECS: dict[str, dict[str, Any]] = {
    "ea017_baseline_0007": {
        "label":           "EA017",
        "is_start":        "2023-01-01",
        "is_end":          "2025-09-30",
        "oos_start":       "2025-10-01",
        "oos_end":         "2026-03-18",
        "benchmark_pf":    None,
        "benchmark_label": "no benchmark (new brick size)",
        "summary":         "creative",
    },
    "ea018_vp_div_0007": {
        "label":           "EA018",
        "is_start":        "2023-01-01",
        "is_end":          "2025-09-30",
        "oos_start":       "2025-10-01",
        "oos_end":         "2026-03-18",
        "benchmark_pf":    10.62,
        "benchmark_label": "EA008 n=5 cd=30 (0.0006 data)",
        "summary":         "creative",
    },
    "gj011_sto_tso": {
        "label":           "GJ011",
        "is_start":        "2024-11-21",
        "is_end":          "2025-09-30",
        "oos_start":       "2025-10-01",
        "oos_end":         "2026-02-28",
        "benchmark_pf":    21.33,
        "benchmark_label": "GJ008 n=5 cd=20",
        "summary":         "gate",
    },
}

STRATEGY_NAMES = ("ea017_baseline_0007", "ea018_vp_div_0007", "gj011_sto_tso")

DATA_CACHE: dict[str, Any] = {}


def load_base_df(renko_file: str):
    if renko_file in DATA_CACHE:
        return DATA_CACHE[renko_file]
    df = load_renko_export(renko_file)
    add_renko_indicators(df)
    DATA_CACHE[renko_file] = df
    return df


def run_period_backtest(
    df_sig,
    start: str,
    end: str,
    commission_pct: float,
    initial_capital: float,
) -> dict[str, Any]:
    cfg = BacktestConfig(
        initial_capital=initial_capital,
        commission_pct=commission_pct,
        slippage_ticks=0,
        qty_type="fixed",
        qty_value=FIXED_QTY_VALUE,
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
        "pf": float("inf") if math.isinf(pf) else float(pf),
        "net_profit": float(kpis.get("net_profit", 0.0) or 0.0),
        "total_trades": int(kpis.get("total_trades", 0) or 0),
        "win_rate": float(kpis.get("win_rate", 0.0) or 0.0),
        "max_drawdown_pct": float(kpis.get("max_drawdown_pct", 0.0) or 0.0),
        "avg_trade": float(kpis.get("avg_trade", 0.0) or 0.0),
        "avg_win_loss_ratio": float(kpis.get("avg_win_loss_ratio", 0.0) or 0.0),
    }


def compute_decay_pct(is_pf: float, oos_pf: float) -> float:
    if math.isinf(is_pf) and math.isinf(oos_pf):
        return 0.0
    if math.isinf(is_pf):
        return -100.0
    if math.isinf(oos_pf):
        return float("inf")
    if is_pf == 0.0:
        return float("inf") if oos_pf > 0.0 else 0.0
    return ((oos_pf - is_pf) / is_pf) * 100.0


def pf_sort_value(pf: float) -> float:
    return 1e12 if math.isinf(pf) else pf


def row_sort_key(row: dict[str, Any]) -> tuple[float, int, float, float]:
    return (
        pf_sort_value(row["oos_pf"]),
        row["oos_trades"],
        row["oos_net_profit"],
        pf_sort_value(row["is_pf"]),
    )


def json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: json_ready(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [json_ready(value) for value in obj]
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return "INF" if obj > 0 else "-INF"
    return obj


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")


def format_pf(pf: float, width: int = 6) -> str:
    return f"{'INF':>{width}}" if math.isinf(pf) else f"{pf:>{width}.2f}"


def format_pct(value: float, width: int = 7) -> str:
    return f"{'+INF%':>{width}}" if math.isinf(value) else f"{value:+{width - 1}.1f}%"


def ordered_combos(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid.keys())
    return [dict(zip(keys, values)) for values in itertools.product(*grid.values())]


def build_row(
    strategy_name: str,
    label: str,
    params: dict[str, Any],
    is_metrics: dict[str, Any],
    oos_metrics: dict[str, Any],
    benchmark_pf: float | None,
) -> dict[str, Any]:
    row = {
        "strategy": strategy_name,
        "label": label,
        "params": params,
        "is_pf": is_metrics["pf"],
        "is_net_profit": is_metrics["net_profit"],
        "is_trades": is_metrics["total_trades"],
        "is_win_rate": is_metrics["win_rate"],
        "is_max_drawdown_pct": is_metrics["max_drawdown_pct"],
        "is_avg_trade": is_metrics["avg_trade"],
        "is_avg_win_loss_ratio": is_metrics["avg_win_loss_ratio"],
        "oos_pf": oos_metrics["pf"],
        "oos_net_profit": oos_metrics["net_profit"],
        "oos_trades": oos_metrics["total_trades"],
        "oos_win_rate": oos_metrics["win_rate"],
        "oos_max_drawdown_pct": oos_metrics["max_drawdown_pct"],
        "oos_avg_trade": oos_metrics["avg_trade"],
        "oos_avg_win_loss_ratio": oos_metrics["avg_win_loss_ratio"],
    }
    row["decay_pct"] = compute_decay_pct(row["is_pf"], row["oos_pf"])
    if benchmark_pf is not None:
        row["benchmark_pf"] = benchmark_pf
        row["beats_benchmark"] = pf_sort_value(row["oos_pf"]) > pf_sort_value(benchmark_pf)
    return row


def strategy_payload(
    strategy_name: str,
    spec: dict[str, Any],
    rows: list[dict[str, Any]],
    combos_total: int,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    payload = {
        "strategy": spec["label"],
        "module": strategy_name,
        "status": status,
        "completed_combos": len(rows),
        "total_combos": combos_total,
        "is_period": {"start": spec["is_start"], "end": spec["is_end"]},
        "oos_period": {"start": spec["oos_start"], "end": spec["oos_end"]},
        "results": rows,
    }
    if spec.get("benchmark_pf") is not None:
        payload["benchmark_pf"] = spec["benchmark_pf"]
        payload["benchmark_label"] = spec["benchmark_label"]
    if error:
        payload["error"] = error
    return payload


def print_summary(
    label: str,
    rows: list[dict[str, Any]],
    spec: dict[str, Any],
) -> None:
    benchmark_pf = spec.get("benchmark_pf")
    bench_str = f"  [benchmark OOS PF {benchmark_pf:.2f}]" if benchmark_pf else ""
    print(f"{'=' * 60} {label} Top 10 OOS{bench_str}")
    print("params |  IS PF   T | OOS PF   T   WR%   Decay  Beat?")
    for row in sorted(rows, key=row_sort_key, reverse=True)[:10]:
        beat = "  YES" if row.get("beats_benchmark") else "  ---"
        print(
            f"{row['params']} | "
            f"{format_pf(row['is_pf'], 6)} {row['is_trades']:>4} | "
            f"{format_pf(row['oos_pf'], 6)} {row['oos_trades']:>4} "
            f"{row['oos_win_rate']:>5.1f} {format_pct(row['decay_pct'], 8)}{beat}"
        )
    zero  = sum(1 for r in rows if r["oos_trades"] == 0)
    total = len(rows)
    beats = sum(1 for r in rows if r.get("beats_benchmark"))
    bench_note = f" | {beats}/{total} beat benchmark" if benchmark_pf else ""
    print(f"  ({total - zero}/{total} combos produced OOS trades{bench_note})")
    print()


def run_strategy(strategy_name: str) -> tuple[dict[str, Any], list[dict[str, Any]], str | None]:
    spec    = STRATEGY_SPECS[strategy_name]
    mod     = importlib.import_module(strategy_name)
    base_df = load_base_df(mod.RENKO_FILE)
    combos  = ordered_combos(mod.PARAM_GRID)

    commission_pct  = getattr(mod, "COMMISSION_PCT", 0.009)
    initial_capital = getattr(mod, "INITIAL_CAPITAL", 1000.0)
    started         = time.time()
    error_text      = None
    rows: list[dict[str, Any]] = []

    print(
        f"Running {spec['label']} ({strategy_name}) - "
        f"{len(combos)} combos | IS {spec['is_start']} to {spec['is_end']} | "
        f"OOS {spec['oos_start']} to {spec['oos_end']}"
    )

    try:
        for idx, params in enumerate(combos, start=1):
            df_sig = mod.generate_signals(base_df.copy(), **params)
            is_metrics = run_period_backtest(
                df_sig, spec["is_start"], spec["is_end"],
                commission_pct, initial_capital,
            )
            oos_metrics = run_period_backtest(
                df_sig, spec["oos_start"], spec["oos_end"],
                commission_pct, initial_capital,
            )
            rows.append(build_row(
                strategy_name, spec["label"], params,
                is_metrics, oos_metrics, spec.get("benchmark_pf"),
            ))

            if idx % 6 == 0 or idx == len(combos):
                elapsed = time.time() - started
                best    = max(rows, key=row_sort_key)
                print(
                    f"  [{idx:>3}/{len(combos)}] {elapsed:>6.1f}s "
                    f"best OOS PF {format_pf(best['oos_pf'], 6)} ({best['oos_trades']}t)"
                )
    except Exception:
        error_text = traceback.format_exc()
        print(f"\n*** ERROR in {spec['label']} after {len(rows)} combos ***")
        print(error_text)

    return spec, rows, error_text


def save_payload(
    all_rows:    dict[str, list[dict[str, Any]]],
    errors:      dict[str, str],
    total_combos: dict[str, int],
    status:      str,
) -> None:
    payload = {"status": status, "strategies": {}}
    for strategy_name in STRATEGY_NAMES:
        spec  = STRATEGY_SPECS[strategy_name]
        entry = strategy_payload(
            strategy_name,
            spec,
            sorted(all_rows.get(strategy_name, []), key=row_sort_key, reverse=True),
            total_combos.get(strategy_name, 0),
            "failed" if strategy_name in errors else status,
            errors.get(strategy_name),
        )
        payload["strategies"][spec["label"]] = entry
    save_json(OUT_DIR / "ea_phase5_results.json", payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5 backtest sweep — EA017 + EA018 + GJ011")
    parser.add_argument(
        "--strategies",
        nargs="*",
        default=list(STRATEGY_SPECS.keys()),
        help="Subset of strategy modules to run.",
    )
    return parser.parse_args()


def main() -> int:
    args     = parse_args()
    selected = args.strategies
    invalid  = [name for name in selected if name not in STRATEGY_SPECS]
    if invalid:
        print(f"Unknown strategy names: {', '.join(invalid)}", file=sys.stderr)
        return 2

    all_rows: dict[str, list[dict[str, Any]]]    = {}
    all_errors: dict[str, str]                   = {}
    total_combos: dict[str, int] = {
        name: len(list(itertools.product(
            *importlib.import_module(name).PARAM_GRID.values()
        )))
        for name in STRATEGY_NAMES
    }

    overall_errors: list[tuple[str, str]] = []

    for strategy_name in selected:
        spec, rows, error_text = run_strategy(strategy_name)

        all_rows[strategy_name] = rows
        if error_text:
            all_errors[strategy_name] = error_text
            overall_errors.append((spec["label"], error_text))
            print(f"{spec['label']} failed after {len(rows)} completed combos.")

        save_payload(all_rows, all_errors, total_combos,
                     "completed" if not all_errors else "partial")
        print_summary(spec["label"], rows, spec)

    save_payload(all_rows, all_errors, total_combos,
                 "completed" if not all_errors else "partial")

    if overall_errors:
        print("\nSweep completed with failures:")
        for label, error_text in overall_errors:
            first_line = error_text.strip().splitlines()[-1] if error_text.strip() else "Unknown error"
            print(f"  - {label}: {first_line}")
        return 1

    print("\nAll requested sweeps completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
