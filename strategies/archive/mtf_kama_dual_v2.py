"""
MTF KAMA Dual v2 optimizer.

Implements and optimizes the strategy logic from strategies/mtf_kama_dual_v2.pine:
- 5m execution timeframe
- HTF directional filters (tf1/tf2)
- KAMA cross entries with cooldown
- Volume SMA filter (>20 period avg)
- RSI Pullback filter (RSI < 30 for long)
- ATR-based dynamic risk management (TP/SL)
- Friday End-Of-Week exit
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
from indicators.rsi import calc_rsi
from indicators.atr import calc_atr

GLOBAL_CONTEXT = None


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
    daily = daily.dropna(subset=["Open", "High", "Low", "Close"])
    return daily


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
    low: np.ndarray,
    volume: np.ndarray,
    kama_chart: np.ndarray,
    slope_tf1: np.ndarray,
    slope_tf2: np.ndarray,
    rsi_5m: np.ndarray,
    atr_5m: np.ndarray,
    vol_sma_5m: np.ndarray,
    cooldown: int,
    rsi_bull_thresh: float,
    rsi_bear_thresh: float,
    dates: np.ndarray,
    start_date: str = "2025-11-24",
    end_date: str = "2069-12-31",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Series, pd.Series]:
    
    ts_start = np.datetime64(start_date, "ns")
    ts_end = np.datetime64(end_date, "ns")

    n = len(close)
    long_entry = np.zeros(n, dtype=bool)
    long_exit = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit = np.zeros(n, dtype=bool)
    
    # Store exact ATR values at signal time for TP/SL calculation matching TV
    entry_atr = pd.Series(0.0, index=range(n))

    last_trade_bar = -999_999

    for i in range(1, n):
        if (
            np.isnan(close[i])
            or np.isnan(close[i - 1])
            or np.isnan(kama_chart[i])
            or np.isnan(kama_chart[i - 1])
            or np.isnan(slope_tf1[i])
            or np.isnan(slope_tf2[i])
            or np.isnan(rsi_5m[i])
        ):
            continue

        bar_in_range = ts_start <= dates[i] <= ts_end

        # Friday Close Logic (close all going into weekend)
        timestamp = pd.Timestamp(dates[i])
        if timestamp.dayofweek == 4 and timestamp.hour >= 16 and timestamp.minute >= 50:
            long_exit[i] = True
            short_exit[i] = True
            continue

        tf1_bull = slope_tf1[i] > 0
        tf1_bear = slope_tf1[i] < 0
        tf2_bull = slope_tf2[i] > 0
        tf2_bear = slope_tf2[i] < 0

        cross_up = close[i] > kama_chart[i] and close[i - 1] <= kama_chart[i - 1]
        cross_dn = close[i] < kama_chart[i] and close[i - 1] >= kama_chart[i - 1]

        # v2 Filters
        vol_ok = volume[i] > vol_sma_5m[i]
        rsi_bull_ok = rsi_5m[i] < rsi_bull_thresh
        rsi_bear_ok = rsi_5m[i] > rsi_bear_thresh

        long_cond = tf1_bull and tf2_bull and cross_up and vol_ok and rsi_bull_ok
        short_cond = tf1_bear and tf2_bear and cross_dn and vol_ok and rsi_bear_ok

        long_exit[i] = (not tf1_bull) or (not tf2_bull) or (close[i] < kama_chart[i])
        short_exit[i] = (not tf1_bear) or (not tf2_bear) or (close[i] > kama_chart[i])

        can_trade = (i - last_trade_bar) >= cooldown
        
        if bar_in_range and can_trade and long_cond:
            long_entry[i] = True
            entry_atr[i] = atr_5m[i] # Store ATR for exit levels
            last_trade_bar = i
            
        if bar_in_range and can_trade and short_cond:
            short_entry[i] = True
            entry_atr[i] = atr_5m[i] # Store ATR for exit levels
            last_trade_bar = i

    return long_entry, long_exit, short_entry, short_exit, entry_atr


def _run_backtest(df_with_signals: pd.DataFrame, tp_atr_mult: float, sl_atr_mult: float, start_date: str = "2025-11-24") -> dict:
    # Instead of config percentages, we compute offset limits for each trade row
    
    # Forward fill entry ATR so while a trade is open it checks levels against its entry ATR
    df_with_signals['tp_offset'] = df_with_signals['entry_atr'].replace(0.0, np.nan).ffill() * tp_atr_mult
    df_with_signals['sl_offset'] = df_with_signals['entry_atr'].replace(0.0, np.nan).ffill() * sl_atr_mult

    cfg = BacktestConfig(
        initial_capital=1000.0,
        commission_pct=0.0043,
        slippage_ticks=0,
        qty_type="fixed",
        qty_value=1000.0, # 1000 units is appropriate for FX base standard
        pyramiding=1,
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


def _worker(params: tuple) -> dict:
    global GLOBAL_CONTEXT

    # Unpack params
    kama_len, kama_fast, kama_slow, tf1, tf2, cooldown, rsi_bull, rsi_bear, tp_atr, sl_atr = params
    
    close = GLOBAL_CONTEXT["close"]
    high = GLOBAL_CONTEXT["high"]
    low = GLOBAL_CONTEXT["low"]
    volume = GLOBAL_CONTEXT["volume"]
    rsi = GLOBAL_CONTEXT["rsi"]
    atr = GLOBAL_CONTEXT["atr"]
    vol_sma = GLOBAL_CONTEXT["vol_sma"]
    dates = GLOBAL_CONTEXT["dates"]
    start_date = GLOBAL_CONTEXT["start_date"]
    base_df = GLOBAL_CONTEXT["base_df"]

    kama_chart = GLOBAL_CONTEXT["kama_5m_cache"][(kama_len, kama_fast, kama_slow)]
    slope_tf1 = GLOBAL_CONTEXT["htf_slope_cache"][(tf1, kama_len, kama_fast, kama_slow)]
    slope_tf2 = GLOBAL_CONTEXT["htf_slope_cache"][(tf2, kama_len, kama_fast, kama_slow)]

    le, lx, se, sx, entry_atr = _build_signals(
        close, high, low, volume, kama_chart, slope_tf1, slope_tf2, 
        rsi, atr, vol_sma, cooldown, rsi_bull, rsi_bear, dates, start_date
    )

    df_sig = base_df.copy()
    df_sig["long_entry"] = le
    df_sig["long_exit"] = lx
    df_sig["short_entry"] = se
    df_sig["short_exit"] = sx
    df_sig["entry_atr"] = entry_atr.values

    metrics = _extract_metrics(_run_backtest(df_sig, tp_atr, sl_atr, start_date))
    metrics["params"] = {
        "kama": f"{kama_len}/{kama_fast}/{kama_slow}",
        "tfs": f"{tf1}/{tf2}",
        "cd": cooldown,
        "rsi_b_b": f"{rsi_bull}/{rsi_bear}",
        "tp_sl_atr": f"{tp_atr}/{sl_atr}",
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
    output_path = here / "mtf_kama_dual_results_v2.txt"

    START_DATE = "2025-11-24"

    print("Loading data...")
    df_5m = load_tv_export("OANDA_EURUSD, 5.csv")
    df_60m = load_tv_export("OANDA_EURUSD, 60.csv")
    df_240m = load_tv_export("OANDA_EURUSD, 240.csv")

    try:
        df_1440m = load_tv_export("OANDA_EURUSD, 1D.csv")
    except FileNotFoundError:
        df_1440m = _resample_60m_to_1d(df_60m)

    htf_by_tf = {
        60: df_60m,
        240: df_240m,
        1440: df_1440m,
    }

    # Base parameters narrowed down based on v1 findings
    kama_len_vals = [30]
    kama_fast_vals = [2]
    kama_slow_vals = [60]
    tf1_vals = [60]
    tf2_vals = [1440]
    cooldown_vals = [60, 120]
    
    # New v2 params
    rsi_bull_thresh_vals = [30, 40, 50]
    rsi_bear_thresh_vals = [70, 60, 50]
    tp_atr_vals = [1.0, 2.0, 3.0, 0.0] # 0 = disabled
    sl_atr_vals = [1.0, 1.5, 2.0, 0.0]

    print("Precomputing indicators...")
    
    # Shared V2 indicators
    rsi_res = calc_rsi(df_5m, period=14)
    atr_res = calc_atr(df_5m, period=14)
    vol_sma = df_5m["Volume"].rolling(20).mean().bfill()
    
    kama_5m_cache = {}
    htf_slope_cache = {}

    for k_len, k_fast, k_slow in itertools.product(kama_len_vals, kama_fast_vals, kama_slow_vals):
        kama_5m = calc_kama(df_5m["Close"], length=k_len, fast=k_fast, slow=k_slow)
        kama_5m_cache[(k_len, k_fast, k_slow)] = kama_5m.values

        for tf in sorted(set(tf1_vals + tf2_vals)):
            htf_kama = calc_kama(htf_by_tf[tf]["Close"], length=k_len, fast=k_fast, slow=k_slow)
            _, aligned_slope = _align_htf_to_ltf(df_5m.index, htf_kama)
            htf_slope_cache[(tf, k_len, k_fast, k_slow)] = aligned_slope

    all_params = list(
        itertools.product(
            kama_len_vals,
            kama_fast_vals,
            kama_slow_vals,
            tf1_vals,
            tf2_vals,
            cooldown_vals,
            rsi_bull_thresh_vals,
            rsi_bear_thresh_vals,
            tp_atr_vals,
            sl_atr_vals,
        )
    )
    print(f"Total combinations: {len(all_params)}")

    context = {
        "base_df": df_5m[["Open", "High", "Low", "Close", "Volume"]].copy(),
        "close": df_5m["Close"].values,
        "high": df_5m["High"].values,
        "low": df_5m["Low"].values,
        "volume": df_5m["Volume"].values,
        "rsi": rsi_res["rsi"],
        "atr": atr_res["atr"],
        "vol_sma": vol_sma.values,
        "dates": df_5m.index.to_numpy(dtype="datetime64[ns]"),
        "start_date": START_DATE,
        "kama_5m_cache": kama_5m_cache,
        "htf_slope_cache": htf_slope_cache,
    }

    max_workers = max(1, (os.cpu_count() or 1) - 1)
    
    results: list[dict] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_pool_init,
        initargs=(context,),
    ) as pool:
        for item in pool.map(_worker, all_params):
            results.append(item)

    eligible = [r for r in results if r["trades"] >= 20] # Lowered trade cap slightly due to strict filters
    eligible.sort(
        key=lambda r: (_pf_sort_key(r["pf"]), r["net_profit"]),
        reverse=True,
    )

    top20 = eligible[:20]

    with output_path.open("w", encoding="utf-8") as f:
        f.write("MTF KAMA Dual v2 Optimization Results\n")
        f.write("Goal: Max Profit Factor with trades >= 20 (RSI/ATR Filters Active)\n\n")
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

    print(f"Eligible results (trades >= 20): {len(eligible)}")
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
        print("No parameter set met minimum trades threshold.")
    print(f"Wrote leaderboard: {output_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
