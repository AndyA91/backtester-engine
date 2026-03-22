"""Phase 2 backtest sweep runner.

Runs the requested IS+OOS sweeps for:
  - r012_macd_lc
  - gj010_macd_lc
  - ea011_v2_auction_champion
  - ea012_napoleon_value
  - ea013_adaptive_escgo
  - ea014_alpha_sniper

Outputs:
  - ai_context/r012_results.json
  - ai_context/gj010_results.json
  - ai_context/ea011v2_results.json
  - ai_context/ea_creative_results.json
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

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent

for path in (ROOT, ROOT / "renko" / "strategies"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from engine import BacktestConfig, run_backtest_long_short
from renko.data import load_renko_export
from renko.indicators import add_renko_indicators

FIXED_QTY_VALUE = 1000.0

R012_BENCHMARK = 12.79
GJ010_BENCHMARK = 21.33
EA011_BENCHMARK = 10.62

STRATEGY_SPECS: dict[str, dict[str, Any]] = {
    "r012_macd_lc": {
        "label": "R012",
        "is_start": "2024-01-01",
        "is_end": "2025-09-30",
        "oos_start": "2025-10-01",
        "oos_end": "2026-02-28",
        "benchmark_pf": R012_BENCHMARK,
        "benchmark_label": "R008 n=5 cd=30",
        "json_path": OUT_DIR / "r012_results.json",
        "top_n": 10,
        "save_mode": "top",
        "summary": "gate",
    },
    "gj010_macd_lc": {
        "label": "GJ010",
        "is_start": "2024-11-21",
        "is_end": "2025-09-30",
        "oos_start": "2025-10-01",
        "oos_end": "2026-02-28",
        "benchmark_pf": GJ010_BENCHMARK,
        "benchmark_label": "GJ008 n=5 cd=20",
        "json_path": OUT_DIR / "gj010_results.json",
        "top_n": 10,
        "save_mode": "top",
        "summary": "gate",
    },
    "ea011_v2_auction_champion": {
        "label": "EA011 v2",
        "is_start": "2023-07-20",
        "is_end": "2025-09-30",
        "oos_start": "2025-10-01",
        "oos_end": "2026-03-17",
        "benchmark_pf": EA011_BENCHMARK,
        "benchmark_label": "EA008 n=5 cd=30",
        "json_path": OUT_DIR / "ea011v2_results.json",
        "top_n": 10,
        "save_mode": "top",
        "summary": "ea011",
        "min_oos_trades_for_report": 15,
    },
    "ea012_napoleon_value": {
        "label": "EA012",
        "is_start": "2023-07-20",
        "is_end": "2025-09-30",
        "oos_start": "2025-10-01",
        "oos_end": "2026-03-17",
        "summary": "creative",
    },
    "ea013_adaptive_escgo": {
        "label": "EA013",
        "is_start": "2023-07-20",
        "is_end": "2025-09-30",
        "oos_start": "2025-10-01",
        "oos_end": "2026-03-17",
        "summary": "creative",
    },
    "ea014_alpha_sniper": {
        "label": "EA014",
        "is_start": "2023-07-20",
        "is_end": "2025-09-30",
        "oos_start": "2025-10-01",
        "oos_end": "2026-03-17",
        "summary": "creative",
    },
}

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


def gate_combo_name(params: dict[str, Any]) -> str:
    use_macd = bool(params.get("use_macd_lc"))
    use_fsb = bool(params.get("use_fsb_strong"))
    if use_macd and use_fsb:
        return "both gates"
    if use_macd:
        return "macd_lc only"
    if use_fsb:
        return "fsb_strong only"
    return "baseline"


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


def ordered_combos(strategy_name: str, grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid.keys())
    combos = [dict(zip(keys, values)) for values in itertools.product(*grid.values())]

    if strategy_name == "ea011_v2_auction_champion":
        combos.sort(
            key=lambda p: (
                p["vp_lookback"],
                p["cvd_lookback"],
                p["req_poc_mig"],
                p["req_trendline"],
                p["req_no_exhaust"],
                p["score_threshold"],
                p["cooldown"],
                p["session_start"],
            )
        )
    elif strategy_name == "ea013_adaptive_escgo":
        combos.sort(
            key=lambda p: (
                p["tl_length"],
                p["escgo_lookback"],
                p["escgo_cooldown"],
                p["use_escgo_exit"],
                p["session_start"],
            )
        )
    elif strategy_name == "ea014_alpha_sniper":
        combos.sort(
            key=lambda p: (
                1 if p["vp_lookback"] == 100 else 0,
                p["vp_lookback"],
                p["imb_threshold"],
                p["min_stacked"],
                p["min_signals"],
                p["cooldown"],
                p["session_start"],
            )
        )
    else:
        combos.sort(key=lambda p: tuple(str(p[key]) for key in keys))
    return combos


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
    if strategy_name in {"r012_macd_lc", "gj010_macd_lc"}:
        row["gate_combo"] = gate_combo_name(params)
    if benchmark_pf is not None:
        row["benchmark_pf"] = benchmark_pf
        row["beats_benchmark"] = pf_sort_value(row["oos_pf"]) > pf_sort_value(benchmark_pf)
    return row


def strategy_payload(
    spec: dict[str, Any],
    rows: list[dict[str, Any]],
    combos_total: int,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    payload = {
        "strategy": spec["label"],
        "module": next(
            name for name, value in STRATEGY_SPECS.items() if value is spec
        ),
        "status": status,
        "completed_combos": len(rows),
        "total_combos": combos_total,
        "is_period": {"start": spec["is_start"], "end": spec["is_end"]},
        "oos_period": {"start": spec["oos_start"], "end": spec["oos_end"]},
        "results": rows,
    }
    if "benchmark_pf" in spec:
        payload["benchmark_pf"] = spec["benchmark_pf"]
        payload["benchmark_label"] = spec["benchmark_label"]
    if error:
        payload["error"] = error
    return payload


def top_rows_for_save(spec: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered = rows
    min_trades = spec.get("min_oos_trades_for_report")
    if min_trades is not None:
        filtered = [row for row in rows if row["oos_trades"] >= min_trades]
    filtered = sorted(filtered, key=row_sort_key, reverse=True)
    return filtered[: spec["top_n"]]


def print_gate_summary(spec: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    header = (
        f"{'=' * 60} {spec['label']} Benchmark: OOS PF {spec['benchmark_pf']:.2f} "
        f"[{spec['benchmark_label']}]"
    )
    print(header)
    print("gate_combo        n cd |  IS PF   T | OOS PF   T   WR%   Decay")

    for row in sorted(rows, key=row_sort_key, reverse=True)[: spec["top_n"]]:
        beat = "  <<BEAT" if row.get("beats_benchmark") else ""
        print(
            f"{row['gate_combo']:<16} "
            f"{row['params']['n_bricks']:>1} {row['params']['cooldown']:>2} | "
            f"{format_pf(row['is_pf'], 6)} {row['is_trades']:>4} | "
            f"{format_pf(row['oos_pf'], 6)} {row['oos_trades']:>4} "
            f"{row['oos_win_rate']:>5.1f} {format_pct(row['decay_pct'], 8)}{beat}"
        )

    print()
    print("Gate averages (OOS trades >= 20):")
    order = ["macd_lc only", "fsb_strong only", "both gates", "baseline"]
    eligible = [row for row in rows if row["oos_trades"] >= 20]
    for gate_combo in order:
        subset = [row for row in eligible if row["gate_combo"] == gate_combo]
        if not subset:
            print(f"{gate_combo:<16} avg OOS PF: n/a    avg T: n/a")
            continue
        avg_pf = sum(pf_sort_value(row["oos_pf"]) for row in subset) / len(subset)
        avg_t = sum(row["oos_trades"] for row in subset) / len(subset)
        print(f"{gate_combo:<16} avg OOS PF: {avg_pf:>5.2f}   avg T: {avg_t:>4.0f}")
    print()


def print_ea011_summary(spec: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    print(
        f"{'=' * 60} {spec['label']} Benchmark: OOS PF {spec['benchmark_pf']:.2f} "
        f"[{spec['benchmark_label']}]"
    )
    print("vp cvd poc tl exh score cd ss |  IS PF   T | OOS PF   T   WR%   Decay")

    filtered = [row for row in rows if row["oos_trades"] >= spec["min_oos_trades_for_report"]]
    for row in sorted(filtered, key=row_sort_key, reverse=True)[: spec["top_n"]]:
        params = row["params"]
        beat = "  <<BEAT" if row.get("beats_benchmark") else ""
        print(
            f"{params['vp_lookback']:>3} {params['cvd_lookback']:>3} "
            f"{int(params['req_poc_mig']):>3} {int(params['req_trendline']):>2} "
            f"{int(params['req_no_exhaust']):>3} {params['score_threshold']:>5} "
            f"{params['cooldown']:>2} {params['session_start']:>2} | "
            f"{format_pf(row['is_pf'], 6)} {row['is_trades']:>4} | "
            f"{format_pf(row['oos_pf'], 6)} {row['oos_trades']:>4} "
            f"{row['oos_win_rate']:>5.1f} {format_pct(row['decay_pct'], 8)}{beat}"
        )
    print()


def print_creative_summary(label: str, rows: list[dict[str, Any]]) -> None:
    print(f"{'=' * 60} {label} Top 5 OOS")
    print("params |  IS PF   T | OOS PF   T   WR%   Decay")
    for row in sorted(rows, key=row_sort_key, reverse=True)[:5]:
        print(
            f"{row['params']} | "
            f"{format_pf(row['is_pf'], 6)} {row['is_trades']:>4} | "
            f"{format_pf(row['oos_pf'], 6)} {row['oos_trades']:>4} "
            f"{row['oos_win_rate']:>5.1f} {format_pct(row['decay_pct'], 8)}"
        )
    print()


def run_strategy(strategy_name: str) -> tuple[dict[str, Any], list[dict[str, Any]], str | None]:
    spec = STRATEGY_SPECS[strategy_name]
    mod = importlib.import_module(strategy_name)
    base_df = load_base_df(mod.RENKO_FILE)
    combos = ordered_combos(strategy_name, mod.PARAM_GRID)
    rows: list[dict[str, Any]] = []

    commission_pct = getattr(mod, "COMMISSION_PCT", 0.0046)
    initial_capital = getattr(mod, "INITIAL_CAPITAL", 1000.0)
    started = time.time()
    error_text = None

    print(
        f"Running {spec['label']} ({strategy_name}) - "
        f"{len(combos)} combos, IS {spec['is_start']} to {spec['is_end']}, "
        f"OOS {spec['oos_start']} to {spec['oos_end']}"
    )

    try:
        for idx, params in enumerate(combos, start=1):
            df_sig = mod.generate_signals(base_df.copy(), **params)
            is_metrics = run_period_backtest(
                df_sig,
                spec["is_start"],
                spec["is_end"],
                commission_pct,
                initial_capital,
            )
            oos_metrics = run_period_backtest(
                df_sig,
                spec["oos_start"],
                spec["oos_end"],
                commission_pct,
                initial_capital,
            )
            rows.append(
                build_row(
                    strategy_name,
                    spec["label"],
                    params,
                    is_metrics,
                    oos_metrics,
                    spec.get("benchmark_pf"),
                )
            )

            if idx % 10 == 0 or idx == len(combos):
                elapsed = time.time() - started
                print(
                    f"  [{idx:>3}/{len(combos)}] elapsed {elapsed:>7.1f}s "
                    f"best OOS PF {format_pf(max(rows, key=row_sort_key)['oos_pf'], 6)}"
                )

                if spec.get("save_mode") == "top":
                    save_json(
                        spec["json_path"],
                        strategy_payload(
                            spec,
                            top_rows_for_save(spec, rows),
                            len(combos),
                            "running" if idx < len(combos) else "completed",
                        ),
                    )
    except Exception:
        error_text = traceback.format_exc()
        if spec.get("save_mode") == "top":
            save_json(
                spec["json_path"],
                strategy_payload(
                    spec,
                    top_rows_for_save(spec, rows),
                    len(combos),
                    "failed",
                    error_text,
                ),
            )

    if spec.get("save_mode") == "top" and error_text is None:
        save_json(
            spec["json_path"],
            strategy_payload(
                spec,
                top_rows_for_save(spec, rows),
                len(combos),
                "completed",
            ),
        )

    return spec, rows, error_text


def save_creative_payload(
    creative_rows: dict[str, list[dict[str, Any]]],
    errors: dict[str, str],
    creative_total_combos: dict[str, int],
    status: str,
) -> None:
    payload = {
        "status": status,
        "strategies": {},
    }
    for strategy_name in ("ea012_napoleon_value", "ea013_adaptive_escgo", "ea014_alpha_sniper"):
        spec = STRATEGY_SPECS[strategy_name]
        entry = strategy_payload(
            spec,
            sorted(creative_rows.get(strategy_name, []), key=row_sort_key, reverse=True),
            creative_total_combos[strategy_name],
            "failed" if strategy_name in errors else status,
            errors.get(strategy_name),
        )
        payload["strategies"][spec["label"]] = entry
    save_json(OUT_DIR / "ea_creative_results.json", payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategies",
        nargs="*",
        default=[
            "r012_macd_lc",
            "gj010_macd_lc",
            "ea011_v2_auction_champion",
            "ea012_napoleon_value",
            "ea013_adaptive_escgo",
            "ea014_alpha_sniper",
        ],
        help="Subset of strategy modules to run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = args.strategies
    invalid = [name for name in selected if name not in STRATEGY_SPECS]
    if invalid:
        print(f"Unknown strategy names: {', '.join(invalid)}", file=sys.stderr)
        return 2

    creative_rows: dict[str, list[dict[str, Any]]] = {}
    creative_errors: dict[str, str] = {}
    creative_total_combos: dict[str, int] = {
        name: len(ordered_combos(name, importlib.import_module(name).PARAM_GRID))
        for name in ("ea012_napoleon_value", "ea013_adaptive_escgo", "ea014_alpha_sniper")
    }

    overall_errors: list[tuple[str, str]] = []

    for strategy_name in selected:
        spec, rows, error_text = run_strategy(strategy_name)

        if spec["summary"] == "gate":
            print_gate_summary(spec, rows)
        elif spec["summary"] == "ea011":
            print_ea011_summary(spec, rows)
        elif spec["summary"] == "creative":
            creative_rows[strategy_name] = rows
            if error_text:
                creative_errors[strategy_name] = error_text
            save_creative_payload(
                creative_rows,
                creative_errors,
                creative_total_combos,
                "completed" if not creative_errors else "partial",
            )
            print_creative_summary(spec["label"], rows)

        if error_text:
            overall_errors.append((spec["label"], error_text))
            print(f"{spec['label']} failed after {len(rows)} completed combos.")

    if selected and any(name in creative_rows or name in creative_errors for name in creative_total_combos):
        save_creative_payload(
            creative_rows,
            creative_errors,
            creative_total_combos,
            "completed" if not creative_errors else "partial",
        )

    if overall_errors:
        print("Sweep completed with failures:")
        for label, error_text in overall_errors:
            first_line = error_text.strip().splitlines()[-1] if error_text.strip() else "Unknown error"
            print(f"  - {label}: {first_line}")
        return 1

    print("All requested sweeps completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
