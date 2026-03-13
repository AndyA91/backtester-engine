"""
MTF KAMA Dual v1 optimizer.

Implements and optimizes the strategy logic from strategies/mtf_kama_dual_v1.pine:
- 5m execution timeframe
- HTF directional filters (tf1/tf2)
- KAMA cross entries with cooldown
- Alignment-break exits
- Optional TP/SL

Optimization target:
- Highest Profit Factor among configurations with at least 30 trades.
- Writes top 20 rows to strategies/mtf_kama_dual_results.txt
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
from indicators.kama import calc_kama


GLOBAL_CONTEXT = None


def _resample_60m_to_1d(df_60m: pd.DataFrame) -> pd.DataFrame:
    """Build 1D OHLC bars from 60m data if no native 1440 CSV is present."""
    daily = df_60m.resample("1D").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )
    daily = daily.dropna(subset=["Open", "High", "Low", "Close"])
    return daily


def _align_htf_to_ltf(ltf_index: pd.DatetimeIndex, htf_kama: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Align HTF KAMA and its slope to LTF bars via backward merge_asof."""
    htf_frame = pd.DataFrame(
        {
            "Date": htf_kama.index,
            "kama": htf_kama.shift(1).values,
            "slope": htf_kama.diff().shift(1).values,  # Pine lookahead_on + [1]/[2]: slope = kama[i-1] - kama[i-2]
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
    kama_chart: np.ndarray,
    slope_tf1: np.ndarray,
    slope_tf2: np.ndarray,
    cooldown: int,
    dates: np.ndarray,
    start_date: str = "2025-11-24",
    end_date: str = "2069-12-31",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pine mapping:
    - Entries: tf1/tf2 bias + crossover/crossunder + cooldown gate.
    - Exits: bias break OR close recross of chart KAMA.
    - start_date/end_date gate matches Pine's timeCondition. Entries and the
      cooldown counter are gated so phantom pre-range signals cannot desync state.
    """
    ts_start = np.datetime64(start_date, "ns")
    ts_end = np.datetime64(end_date, "ns")

    n = len(close)
    long_entry = np.zeros(n, dtype=bool)
    long_exit = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit = np.zeros(n, dtype=bool)

    last_trade_bar = -999_999

    for i in range(1, n):
        if (
            np.isnan(close[i])
            or np.isnan(close[i - 1])
            or np.isnan(kama_chart[i])
            or np.isnan(kama_chart[i - 1])
            or np.isnan(slope_tf1[i])
            or np.isnan(slope_tf2[i])
        ):
            continue

        bar_in_range = ts_start <= dates[i] <= ts_end

        tf1_bull = slope_tf1[i] > 0
        tf1_bear = slope_tf1[i] < 0
        tf2_bull = slope_tf2[i] > 0
        tf2_bear = slope_tf2[i] < 0

        cross_up = close[i] > kama_chart[i] and close[i - 1] <= kama_chart[i - 1]
        cross_dn = close[i] < kama_chart[i] and close[i - 1] >= kama_chart[i - 1]

        long_cond = tf1_bull and tf2_bull and cross_up
        short_cond = tf1_bear and tf2_bear and cross_dn

        long_exit[i] = (not tf1_bull) or (not tf2_bull) or (close[i] < kama_chart[i])
        short_exit[i] = (not tf1_bear) or (not tf2_bear) or (close[i] > kama_chart[i])

        # Gate entries and cooldown counter on bar_in_range (matches Pine's timeCondition)
        can_trade = (i - last_trade_bar) >= cooldown
        if bar_in_range and can_trade and long_cond:
            long_entry[i] = True
            last_trade_bar = i
        if bar_in_range and can_trade and short_cond:
            short_entry[i] = True
            last_trade_bar = i

    return long_entry, long_exit, short_entry, short_exit


def _run_backtest(df_with_signals: pd.DataFrame, tp_pct: float, sl_pct: float, start_date: str = "2025-11-24") -> dict:
    cfg = BacktestConfig(
        initial_capital=1000.0,
        commission_pct=0.0043,
        slippage_ticks=0,
        qty_type="fixed",
        qty_value=1000.0,
        pyramiding=1,
        take_profit_pct=tp_pct,
        stop_loss_pct=sl_pct,
        start_date=start_date,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return run_backtest_long_short(df_with_signals, cfg)


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


def _worker(params: tuple[int, int, int, int, int, float, float]) -> dict:
    global GLOBAL_CONTEXT

    kama_len, kama_fast, kama_slow, tf1, tf2, cooldown, tp_pct, sl_pct = params
    close = GLOBAL_CONTEXT["close"]
    dates = GLOBAL_CONTEXT["dates"]
    start_date = GLOBAL_CONTEXT["start_date"]
    base_df = GLOBAL_CONTEXT["base_df"]

    kama_chart = GLOBAL_CONTEXT["kama_5m_cache"][(kama_len, kama_fast, kama_slow)]
    slope_tf1 = GLOBAL_CONTEXT["htf_slope_cache"][(tf1, kama_len, kama_fast, kama_slow)]
    slope_tf2 = GLOBAL_CONTEXT["htf_slope_cache"][(tf2, kama_len, kama_fast, kama_slow)]

    le, lx, se, sx = _build_signals(close, kama_chart, slope_tf1, slope_tf2, cooldown, dates, start_date)

    df_sig = base_df.copy()
    df_sig["long_entry"] = le
    df_sig["long_exit"] = lx
    df_sig["short_entry"] = se
    df_sig["short_exit"] = sx

    metrics = _extract_metrics(_run_backtest(df_sig, tp_pct, sl_pct, start_date))
    metrics["params"] = {
        "kama_len": kama_len,
        "kama_fast": kama_fast,
        "kama_slow": kama_slow,
        "tf1": tf1,
        "tf2": tf2,
        "cooldown": cooldown,
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
    }
    return metrics


def _pf_sort_key(pf: float) -> float:
    if math.isinf(pf):
        return 1e12
    if math.isnan(pf):
        return -1e12
    return pf


def _format_pf(pf: float) -> str:
    if math.isinf(pf):
        return "INF"
    if math.isnan(pf):
        return "NaN"
    return f"{pf:.4f}"


def main() -> None:
    here = Path(__file__).resolve().parent
    output_path = here / "mtf_kama_dual_results.txt"

    # Must match Pine's timeCondition start_date for TV matching.
    # OANDA daily bars open at 22:00 UTC — a native 1440m CSV is REQUIRED for
    # correct slope timing. Resampling from 60m uses midnight UTC boundaries
    # which diverge from TV's session and cause different entry signals.
    START_DATE = "2025-11-24"

    print("Loading data...")
    df_5m = load_tv_export("OANDA_EURUSD, 5.csv")
    df_60m = load_tv_export("OANDA_EURUSD, 60.csv")
    df_240m = load_tv_export("OANDA_EURUSD, 240.csv")

    for _daily_name in ("OANDA_EURUSD, 1440.csv", "OANDA_EURUSD, 1D.csv"):
        try:
            df_1440m = load_tv_export(_daily_name)
            print(f"Using native daily CSV ({_daily_name}) for tf2=1440.")
            break
        except FileNotFoundError:
            continue
    else:
        df_1440m = _resample_60m_to_1d(df_60m)
        print(
            "WARNING: No native daily CSV found (tried OANDA_EURUSD, 1440.csv and 1D.csv).\n"
            "  Built tf=1440 from 60m resample. OANDA daily bars open at 22:00 UTC, not\n"
            "  midnight — slope timing will diverge from TradingView.\n"
            "  Export a daily CSV from TV and place it in data/ for exact matching."
        )

    htf_by_tf = {
        60: df_60m,
        240: df_240m,
        1440: df_1440m,
    }

    kama_len_vals = [10, 14, 21, 30]
    kama_fast_vals = [2, 3, 5]
    kama_slow_vals = [30, 60, 100]
    tf1_vals = [60, 240]
    tf2_vals = [240, 1440]
    cooldown_vals = [30, 60, 120]
    tp_vals = [0.0, 0.1, 0.2, 0.3]
    sl_vals = [0.0, 0.1, 0.2, 0.3]

    print("Precomputing KAMA caches...")
    kama_5m_cache: dict[tuple[int, int, int], np.ndarray] = {}
    htf_slope_cache: dict[tuple[int, int, int, int], np.ndarray] = {}

    for kama_len, kama_fast, kama_slow in itertools.product(
        kama_len_vals, kama_fast_vals, kama_slow_vals
    ):
        kama_5m = calc_kama(
            df_5m["Close"],
            length=kama_len,
            fast=kama_fast,
            slow=kama_slow,
        )
        kama_5m_cache[(kama_len, kama_fast, kama_slow)] = kama_5m.values

        for tf in sorted(set(tf1_vals + tf2_vals)):
            htf_kama = calc_kama(
                htf_by_tf[tf]["Close"],
                length=kama_len,
                fast=kama_fast,
                slow=kama_slow,
            )
            _, aligned_slope = _align_htf_to_ltf(df_5m.index, htf_kama)
            htf_slope_cache[(tf, kama_len, kama_fast, kama_slow)] = aligned_slope

    all_params = list(
        itertools.product(
            kama_len_vals,
            kama_fast_vals,
            kama_slow_vals,
            tf1_vals,
            tf2_vals,
            cooldown_vals,
            tp_vals,
            sl_vals,
        )
    )
    print(f"Total combinations: {len(all_params)}")

    context = {
        "base_df": df_5m[["Open", "High", "Low", "Close", "Volume"]].copy(),
        "close": df_5m["Close"].values,
        "dates": df_5m.index.to_numpy(dtype="datetime64[ns]"),
        "start_date": START_DATE,
        "kama_5m_cache": kama_5m_cache,
        "htf_slope_cache": htf_slope_cache,
    }

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

    eligible = [r for r in results if r["trades"] >= 30]
    eligible.sort(
        key=lambda r: (_pf_sort_key(r["pf"]), r["net_profit"]),
        reverse=True,
    )

    top20 = eligible[:20]

    with output_path.open("w", encoding="utf-8") as f:
        f.write("MTF KAMA Dual v1 Optimization Results\n")
        f.write("Goal: Max Profit Factor with trades >= 30 (full-period, no OOS)\n\n")
        f.write("| PF | Net Profit | Max DD % | Win Rate % | Trades | Params |\n")
        f.write("|---:|-----------:|---------:|-----------:|-------:|:-------|\n")
        for row in top20:
            f.write(
                f"| {_format_pf(row['pf'])} | "
                f"{row['net_profit']:.2f} | "
                f"{row['max_dd_pct']:.2f} | "
                f"{row['win_rate']:.2f} | "
                f"{row['trades']} | "
                f"{row['params']} |\n"
            )

    print(f"Eligible results (trades >= 30): {len(eligible)}")
    if top20:
        best = top20[0]
        print(
            "Best: "
            f"PF={_format_pf(best['pf'])}, "
            f"Net={best['net_profit']:.2f}, "
            f"DD%={best['max_dd_pct']:.2f}, "
            f"WR%={best['win_rate']:.2f}, "
            f"Trades={best['trades']}, "
            f"Params={best['params']}"
        )
    else:
        print("No parameter set met trades >= 30.")
    print(f"Wrote leaderboard: {output_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
