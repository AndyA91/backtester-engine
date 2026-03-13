"""
Pivot Breakout v1 optimizer.

Clean fork of validated mtf_kama_dual_v4.py, replacing chart-TF KAMA crossover
entries with strict confirmed pivot breakout logic.

Sweep size: 12 combinations.
"""

import concurrent.futures
import contextlib
import io
import itertools
import math
import multiprocessing
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import BacktestConfig, load_tv_export, run_backtest_long_short
from indicators.adx import calc_adx
from indicators.kama import calc_kama


GLOBAL_CONTEXT = None


def _calc_pivots(high: np.ndarray, low: np.ndarray, pivot_len: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Non-repainting pivot highs/lows matching Pine's ta.pivothigh/ta.pivotlow.
    At bar T, confirms if high[T - pivot_len] is the STRICT max of high[T - 2*pivot_len : T + 1].
    Strict = no ties allowed (matches Pine behavior exactly).
    pivot_high[T] = high[T - pivot_len] if confirmed, else NaN. Same for pivot_low.
    """
    n = len(high)
    pivot_high = np.full(n, np.nan)
    pivot_low = np.full(n, np.nan)
    for T in range(2 * pivot_len, n):
        pb = T - pivot_len
        window_h = high[T - 2 * pivot_len : T + 1]
        window_l = low[T - 2 * pivot_len : T + 1]
        if high[pb] == np.max(window_h) and np.sum(window_h == high[pb]) == 1:
            pivot_high[T] = high[pb]
        if low[pb] == np.min(window_l) and np.sum(window_l == low[pb]) == 1:
            pivot_low[T] = low[pb]
    return pivot_high, pivot_low


def _resample_60m_to_1d(df_60m: pd.DataFrame) -> pd.DataFrame:
    daily = df_60m.resample("1D").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )
    return daily.dropna(subset=["Open", "High", "Low", "Close"])


def _align_htf_to_ltf(ltf_index: pd.DatetimeIndex, htf_kama: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    htf_frame = pd.DataFrame(
        {
            "Date": htf_kama.index,
            "kama": htf_kama.shift(1).values,
            "slope": htf_kama.diff().shift(1).values,
        }
    )
    ltf_frame = pd.DataFrame({"Date": ltf_index})
    merged = pd.merge_asof(
        ltf_frame.sort_values("Date"),
        htf_frame.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    return merged["kama"].values, merged["slope"].values


def _build_signals(
    close: np.ndarray,
    high: np.ndarray,
    pivot_high: np.ndarray,
    pivot_low: np.ndarray,
    slope_tf1: np.ndarray,
    slope_tf2: np.ndarray,
    adx_vals: np.ndarray,
    hours: np.ndarray,
    cooldown: int,
    dates: np.ndarray,
    start_date: str,
    end_date: str,
    use_session_filter: bool,
    adx_threshold: int,
    use_kama_slope_filter: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _ = high
    _ = use_kama_slope_filter
    ts_start = np.datetime64(start_date, "ns")
    ts_end = np.datetime64(end_date, "ns")

    n = len(close)
    long_entry = np.zeros(n, dtype=bool)
    long_exit = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit = np.zeros(n, dtype=bool)

    last_trade_bar = -999_999
    last_ph = np.nan
    last_pl = np.nan

    for i in range(1, n):
        if not np.isnan(pivot_high[i]):
            last_ph = pivot_high[i]
        if not np.isnan(pivot_low[i]):
            last_pl = pivot_low[i]

        if (
            np.isnan(close[i])
            or np.isnan(close[i - 1])
            or np.isnan(slope_tf1[i])
            or np.isnan(slope_tf2[i])
            or np.isnan(adx_vals[i])
        ):
            continue

        bar_in_range = ts_start <= dates[i] <= ts_end

        tf1_bull = slope_tf1[i] > 0
        tf1_bear = slope_tf1[i] < 0
        tf2_bull = slope_tf2[i] > 0
        tf2_bear = slope_tf2[i] < 0

        adx_ok = adx_threshold <= 0 or adx_vals[i] > adx_threshold
        in_session = (7 <= hours[i] < 22) if use_session_filter else True

        breakout_long = (not np.isnan(last_ph)) and close[i] > last_ph and close[i - 1] <= last_ph
        breakout_short = (not np.isnan(last_pl)) and close[i] < last_pl and close[i - 1] >= last_pl

        long_cond = tf1_bull and tf2_bull and breakout_long and in_session and adx_ok
        short_cond = tf1_bear and tf2_bear and breakout_short and in_session and adx_ok

        long_exit_cond = (not tf1_bull) or (not tf2_bull) or (
            not np.isnan(last_pl) and close[i] < last_pl
        )
        short_exit_cond = (not tf1_bear) or (not tf2_bear) or (
            not np.isnan(last_ph) and close[i] > last_ph
        )
        long_exit[i] = long_exit_cond
        short_exit[i] = short_exit_cond

        can_trade = (i - last_trade_bar) >= cooldown
        if bar_in_range and can_trade and long_cond:
            long_entry[i] = True
            last_trade_bar = i
        if bar_in_range and can_trade and short_cond:
            short_entry[i] = True
            last_trade_bar = i

    return long_entry, long_exit, short_entry, short_exit


def _run_backtest(df_sig: pd.DataFrame, start_date: str, end_date: str) -> dict:
    cfg = BacktestConfig(
        initial_capital=1000.0,
        commission_pct=0.0043,
        slippage_ticks=0,
        qty_type="fixed",
        qty_value=1000.0,
        pyramiding=1,
        start_date=start_date,
        end_date=end_date,
        take_profit_pct=0.0,
        stop_loss_pct=0.0,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return run_backtest_long_short(df_sig, cfg)


def _extract_metrics(kpis: dict) -> dict:
    pf = kpis.get("profit_factor", 0.0)
    return {
        "pf": float(pf) if pf is not None else 0.0,
        "net_profit": float(kpis.get("net_profit", 0.0) or 0.0),
        "max_dd_pct": float(kpis.get("max_drawdown_pct", 0.0) or 0.0),
        "win_rate": float(kpis.get("win_rate", 0.0) or 0.0),
        "trades": int(kpis.get("total_trades", 0) or 0),
    }


def _pool_init(context: dict) -> None:
    global GLOBAL_CONTEXT
    GLOBAL_CONTEXT = context


def _worker(params: tuple[int, bool, int, int, bool]) -> dict:
    global GLOBAL_CONTEXT
    cooldown, use_session_filter, pivot_len, adx_threshold, use_kama_slope_filter = params
    pivot_high, pivot_low = GLOBAL_CONTEXT["pivot_cache"][pivot_len]

    le, lx, se, sx = _build_signals(
        close=GLOBAL_CONTEXT["close"],
        high=GLOBAL_CONTEXT["high"],
        pivot_high=pivot_high,
        pivot_low=pivot_low,
        slope_tf1=GLOBAL_CONTEXT["slope_tf1"],
        slope_tf2=GLOBAL_CONTEXT["slope_tf2"],
        adx_vals=GLOBAL_CONTEXT["adx"],
        hours=GLOBAL_CONTEXT["hours"],
        cooldown=cooldown,
        dates=GLOBAL_CONTEXT["dates"],
        start_date=GLOBAL_CONTEXT["start_date"],
        end_date=GLOBAL_CONTEXT["end_date"],
        use_session_filter=use_session_filter,
        adx_threshold=adx_threshold,
        use_kama_slope_filter=use_kama_slope_filter,
    )

    df_sig = GLOBAL_CONTEXT["base_df"].copy()
    df_sig["long_entry"] = le
    df_sig["long_exit"] = lx
    df_sig["short_entry"] = se
    df_sig["short_exit"] = sx

    metrics = _extract_metrics(
        _run_backtest(df_sig, GLOBAL_CONTEXT["start_date"], GLOBAL_CONTEXT["end_date"])
    )
    metrics["params"] = {
        "kama_len": 30,
        "kama_fast": 2,
        "kama_slow": 60,
        "tf1": 60,
        "tf2": 1440,
        "cooldown": cooldown,
        "use_session_filter": use_session_filter,
        "pivot_len": pivot_len,
        "adx_threshold": adx_threshold,
        "use_kama_slope_filter": use_kama_slope_filter,
    }
    return metrics


def _pf_sort_key(pf: float) -> float:
    if math.isinf(pf):
        return 1e12
    if math.isnan(pf):
        return -1e12
    return pf


def _fmt_pf(pf: float) -> str:
    if math.isinf(pf):
        return "INF"
    if math.isnan(pf):
        return "NaN"
    return f"{pf:.4f}"


def main() -> None:
    here = Path(__file__).resolve().parent
    out_path = here / "pivot_breakout_v1_results.txt"

    start_date = "2025-11-24"
    end_date = "2069-12-31"

    print("Loading data...")
    df_5m = load_tv_export("OANDA_EURUSD, 5.csv")
    df_60m = load_tv_export("OANDA_EURUSD, 60.csv")

    for daily_name in ("OANDA_EURUSD, 1440.csv", "OANDA_EURUSD, 1D.csv"):
        try:
            df_1d = load_tv_export(daily_name)
            print(f"Using native daily CSV ({daily_name}).")
            break
        except FileNotFoundError:
            continue
    else:
        df_1d = _resample_60m_to_1d(df_60m)
        print("WARNING: Daily CSV missing; fell back to 60m->1D resample (not TV-exact for OANDA).")

    print("Precomputing indicator caches...")
    kama_1h = calc_kama(df_60m["Close"], length=30, fast=2, slow=60)
    kama_1d = calc_kama(df_1d["Close"], length=30, fast=2, slow=60)
    _, slope_tf1 = _align_htf_to_ltf(df_5m.index, kama_1h)
    _, slope_tf2 = _align_htf_to_ltf(df_5m.index, kama_1d)
    adx_vals = calc_adx(df_5m, di_period=14, adx_period=14)["adx"]

    cooldown_vals = [90]
    session_vals = [True]
    pivot_len_vals = [3, 5, 8, 13]
    adx_threshold_vals = [0, 25, 30]
    kama_slope_vals = [False]

    pivot_cache = {
        pl: _calc_pivots(df_5m["High"].values, df_5m["Low"].values, pl)
        for pl in pivot_len_vals
    }

    all_params = list(
        itertools.product(
            cooldown_vals,
            session_vals,
            pivot_len_vals,
            adx_threshold_vals,
            kama_slope_vals,
        )
    )
    print(f"Total combinations: {len(all_params)}")

    context = {
        "base_df": df_5m[["Open", "High", "Low", "Close", "Volume"]].copy(),
        "close": df_5m["Close"].values,
        "high": df_5m["High"].values,
        "slope_tf1": slope_tf1,
        "slope_tf2": slope_tf2,
        "adx": adx_vals,
        "hours": df_5m.index.hour.values,
        "dates": df_5m.index.to_numpy(dtype="datetime64[ns]"),
        "start_date": start_date,
        "end_date": end_date,
        "pivot_cache": pivot_cache,
    }
    _pool_init(context)

    sanity_result = _worker((90, True, 5, 0, False))
    print(
        "Sanity check (pivot_len=5, adx_threshold=0): "
        f"PF={_fmt_pf(sanity_result['pf'])}, "
        f"Net={sanity_result['net_profit']:.2f}, "
        f"Trades={sanity_result['trades']}"
    )
    if sanity_result["trades"] <= 0:
        raise RuntimeError("Sanity check failed: expected > 0 trades for pivot_len=5, adx_threshold=0.")

    max_workers = max(1, (os.cpu_count() or 1) - 1)
    print(f"Running optimization with {max_workers} workers...")

    results: list[dict] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_pool_init,
        initargs=(context,),
    ) as pool:
        for item in pool.map(_worker, all_params):
            results.append(item)

    results.sort(key=lambda r: (_pf_sort_key(r["pf"]), r["net_profit"]), reverse=True)
    top3 = results[:3]
    zero_trade = [r for r in results if r["trades"] == 0]

    with out_path.open("w", encoding="utf-8") as f:
        f.write("Pivot Breakout v1 Optimization Results\n")
        f.write("Goal: Max PF with fixed v4-A baseline filters and pivot-breakout entries (full-period, no OOS)\n")
        f.write("Fixed baseline: kama_len=30, kama_fast=2, kama_slow=60, tf1=60, tf2=1440, cooldown=90, use_session_filter=True, use_kama_slope_filter=False\n")
        f.write("Sweep: pivot_len in [3, 5, 8, 13], adx_threshold in [0, 25, 30]\n\n")
        f.write("# Header Notes\n")
        f.write(
            f"- Sanity check (pivot_len=5, adx_threshold=0): PF={_fmt_pf(sanity_result['pf'])}, Net={sanity_result['net_profit']:.2f}, Trades={sanity_result['trades']}\n"
        )
        f.write("- Top 3 candidates:\n")
        for idx, row in enumerate(top3, start=1):
            f.write(
                f"  {idx}) PF={_fmt_pf(row['pf'])}, Net={row['net_profit']:.2f}, Trades={row['trades']}, Params={row['params']}\n"
            )
        if zero_trade:
            f.write("- trades=0 combinations:\n")
            for row in zero_trade:
                f.write(f"  - {row['params']}\n")
        else:
            f.write("- trades=0 combinations: none\n")

        f.write("\n| PF | Net Profit | Max DD % | Win Rate % | Trades | Params |\n")
        f.write("|---:|-----------:|---------:|-----------:|-------:|:-------|\n")
        for row in results:
            f.write(
                f"| {_fmt_pf(row['pf'])} | "
                f"{row['net_profit']:.2f} | "
                f"{row['max_dd_pct']:.2f} | "
                f"{row['win_rate']:.2f} | "
                f"{row['trades']} | "
                f"{row['params']} |\n"
            )

    if results:
        best = results[0]
        print(
            "Best: "
            f"PF={_fmt_pf(best['pf'])}, Net={best['net_profit']:.2f}, "
            f"DD%={best['max_dd_pct']:.2f}, WR%={best['win_rate']:.2f}, "
            f"Trades={best['trades']}, Params={best['params']}"
        )
    print(f"Wrote leaderboard: {out_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
