"""
Pivot Breakout v2 optimizer.

Fork of pivot_breakout_v1.py with three new entry-quality filters:
- 5m Efficiency Ratio (ER)
- 5m Bollinger bandwidth squeeze+expansion
- 1H Supertrend direction agreement

Sweep size: 36 combinations.
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
from indicators.bbands import calc_bbands
from indicators.kama import calc_kama
from indicators.supertrend import calc_supertrend


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


def _calc_er(series: pd.Series, length: int = 30) -> np.ndarray:
    """
    Kaufman Efficiency Ratio on a price series.
    ER = |price[i] - price[i-length]| / sum(|price[j] - price[j-1]|, length bars)
    Range [0, 1]. High ER = trending, low ER = choppy.
    Returns array of same length as input; first `length` bars are 0.
    """
    src = series.values.astype(float)
    n = len(src)
    er = np.zeros(n)
    for i in range(length, n):
        direction = abs(src[i] - src[i - length])
        volatility = sum(abs(src[j] - src[j - 1]) for j in range(i - length + 1, i + 1))
        er[i] = direction / volatility if volatility != 0 else 0.0
    return er


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


def _align_htf_st_to_ltf(ltf_index: pd.DatetimeIndex, htf_direction: pd.Series) -> np.ndarray:
    """
    Align 1H Supertrend direction to 5m bars.
    Uses .shift(1) to avoid lookahead (same pattern as _align_htf_to_ltf).
    """
    htf_frame = pd.DataFrame(
        {
            "Date": htf_direction.index,
            "direction": htf_direction.shift(1).values,
        }
    )
    ltf_frame = pd.DataFrame({"Date": ltf_index})
    merged = pd.merge_asof(
        ltf_frame.sort_values("Date"),
        htf_frame.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    return merged["direction"].fillna(0).values


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
    er_vals: np.ndarray,
    er_threshold: float,
    bw_vals: np.ndarray,
    bw_sma_vals: np.ndarray,
    bb_squeeze_lookback: int,
    htf_st_direction: np.ndarray,
    htf_st_multiplier: float,
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

        if er_threshold > 0.0 and np.isnan(er_vals[i]):
            continue
        if bb_squeeze_lookback > 0 and (
            np.isnan(bw_vals[i]) or np.isnan(bw_vals[i - 1]) or np.isnan(bw_sma_vals[i - 1])
        ):
            continue
        if htf_st_multiplier > 0.0 and np.isnan(htf_st_direction[i]):
            continue

        bar_in_range = ts_start <= dates[i] <= ts_end

        tf1_bull = slope_tf1[i] > 0
        tf1_bear = slope_tf1[i] < 0
        tf2_bull = slope_tf2[i] > 0
        tf2_bear = slope_tf2[i] < 0

        adx_ok = adx_threshold <= 0 or adx_vals[i] > adx_threshold
        in_session = (7 <= hours[i] < 22) if use_session_filter else True

        # Efficiency Ratio filter
        er_ok = er_threshold <= 0.0 or er_vals[i] >= er_threshold

        # BB Bandwidth Squeeze filter
        # Require: bw was below its N-bar SMA (coiling), and is now expanding
        if bb_squeeze_lookback > 0 and i >= 1:
            squeeze_ok = (bw_vals[i - 1] < bw_sma_vals[i - 1]) and (bw_vals[i] > bw_vals[i - 1])
        else:
            squeeze_ok = True

        # HTF Supertrend filter (direction: +1 = bullish, -1 = bearish)
        st_long_ok = htf_st_multiplier <= 0.0 or htf_st_direction[i] == 1
        st_short_ok = htf_st_multiplier <= 0.0 or htf_st_direction[i] == -1

        breakout_long = (not np.isnan(last_ph)) and close[i] > last_ph and close[i - 1] <= last_ph
        breakout_short = (not np.isnan(last_pl)) and close[i] < last_pl and close[i - 1] >= last_pl

        long_cond = (
            tf1_bull
            and tf2_bull
            and breakout_long
            and in_session
            and adx_ok
            and er_ok
            and squeeze_ok
            and st_long_ok
        )
        short_cond = (
            tf1_bear
            and tf2_bear
            and breakout_short
            and in_session
            and adx_ok
            and er_ok
            and squeeze_ok
            and st_short_ok
        )

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


def _worker(params: tuple) -> dict:
    global GLOBAL_CONTEXT
    (
        cooldown,
        use_session_filter,
        pivot_len,
        adx_threshold,
        use_kama_slope_filter,
        er_threshold,
        bb_squeeze_lookback,
        htf_st_multiplier,
    ) = params

    pivot_high, pivot_low = GLOBAL_CONTEXT["pivot_cache"][pivot_len]
    er_vals = GLOBAL_CONTEXT["er_5m"]
    bw_vals = GLOBAL_CONTEXT["bw_5m"]
    bw_sma_vals = GLOBAL_CONTEXT["bw_sma_cache"].get(bb_squeeze_lookback, np.zeros_like(bw_vals))
    htf_st_dir = GLOBAL_CONTEXT["htf_st_dir_cache"].get(htf_st_multiplier, np.zeros_like(bw_vals))

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
        start_date=GLOBAL_CONTEXT["full_start"],
        end_date=GLOBAL_CONTEXT["full_end"],
        use_session_filter=use_session_filter,
        adx_threshold=adx_threshold,
        use_kama_slope_filter=use_kama_slope_filter,
        er_vals=er_vals,
        er_threshold=er_threshold,
        bw_vals=bw_vals,
        bw_sma_vals=bw_sma_vals,
        bb_squeeze_lookback=bb_squeeze_lookback,
        htf_st_direction=htf_st_dir,
        htf_st_multiplier=htf_st_multiplier,
    )

    df_sig = GLOBAL_CONTEXT["base_df"].copy()
    df_sig["long_entry"] = le
    df_sig["long_exit"] = lx
    df_sig["short_entry"] = se
    df_sig["short_exit"] = sx

    is_metrics = _extract_metrics(
        _run_backtest(df_sig, GLOBAL_CONTEXT["is_start"], GLOBAL_CONTEXT["is_end"])
    )
    oos_metrics = _extract_metrics(
        _run_backtest(df_sig, GLOBAL_CONTEXT["oos_start"], GLOBAL_CONTEXT["oos_end"])
    )

    return {
        "is": is_metrics,
        "oos": oos_metrics,
        "params": {
            "pivot_len": pivot_len,
            "adx_threshold": adx_threshold,
            "er_threshold": er_threshold,
            "bb_squeeze_lookback": bb_squeeze_lookback,
            "htf_st_multiplier": htf_st_multiplier,
        },
    }


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
    out_path = here / "pivot_breakout_v2_results.txt"

    FULL_START = "2025-11-24"
    FULL_END = "2026-03-03 23:59:59"
    IS_START = "2025-11-24"
    IS_END = "2026-01-31"
    OOS_START = "2026-02-01"
    OOS_END = "2026-03-03 23:59:59"

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

    print("Precomputing baseline indicator caches...")
    kama_1h = calc_kama(df_60m["Close"], length=30, fast=2, slow=60)
    kama_1d = calc_kama(df_1d["Close"], length=30, fast=2, slow=60)
    _, slope_tf1 = _align_htf_to_ltf(df_5m.index, kama_1h)
    _, slope_tf2 = _align_htf_to_ltf(df_5m.index, kama_1d)
    adx_vals = calc_adx(df_5m, di_period=14, adx_period=14)["adx"]

    print("Computing ER(30) on 5m...")
    er_5m = _calc_er(df_5m["Close"], length=30)

    print("Computing BB(20) bandwidth on 5m...")
    bb_result = calc_bbands(df_5m, period=20)
    bw_5m = bb_result["bw"]

    bw_sma_cache = {}
    for lb in [10, 20]:
        bw_sma_cache[lb] = pd.Series(bw_5m).rolling(lb).mean().values

    print("Computing HTF Supertrend variants on 1H...")
    htf_st_dir_cache = {}
    for mult in [2.0, 3.0]:
        st_result = calc_supertrend(df_60m, period=10, multiplier=mult)
        st_direction_series = pd.Series(st_result["direction"], index=df_60m.index)
        htf_st_dir_cache[mult] = _align_htf_st_to_ltf(df_5m.index, st_direction_series)

    cooldown_vals = [90]
    session_vals = [True]
    pivot_len_vals = [8]
    adx_threshold_vals = [30]
    kama_slope_vals = [False]
    er_threshold_vals = [0.0, 0.15, 0.25, 0.35]
    bb_squeeze_lookback_vals = [0, 10, 20]
    htf_st_multiplier_vals = [0.0, 2.0, 3.0]

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
            er_threshold_vals,
            bb_squeeze_lookback_vals,
            htf_st_multiplier_vals,
        )
    )
    print(f"Total combinations: {len(all_params)}")

    base_df = df_5m[["Open", "High", "Low", "Close", "Volume"]].copy()
    context = {
        "base_df": base_df,
        "close": df_5m["Close"].values,
        "high": df_5m["High"].values,
        "slope_tf1": slope_tf1,
        "slope_tf2": slope_tf2,
        "adx": adx_vals,
        "hours": df_5m.index.hour.values,
        "dates": df_5m.index.to_numpy(dtype="datetime64[ns]"),
        "pivot_cache": pivot_cache,
        "er_5m": er_5m,
        "bw_5m": bw_5m,
        "bw_sma_cache": bw_sma_cache,
        "htf_st_dir_cache": htf_st_dir_cache,
        "full_start": FULL_START,
        "full_end": FULL_END,
        "is_start": IS_START,
        "is_end": IS_END,
        "oos_start": OOS_START,
        "oos_end": OOS_END,
    }
    _pool_init(context)

    # Control worker (IS/OOS split metrics)
    sanity_result = _worker((90, True, 8, 30, False, 0.0, 0, 0.0))
    print(
        "Control (IS/OOS split) => "
        f"IS PF={_fmt_pf(sanity_result['is']['pf'])}, OOS PF={_fmt_pf(sanity_result['oos']['pf'])}"
    )

    # Sanity: build signals + run on full period to compare against v1
    pivot_high, pivot_low = pivot_cache[8]
    le, lx, se, sx = _build_signals(
        close=context["close"],
        high=context["high"],
        pivot_high=pivot_high,
        pivot_low=pivot_low,
        slope_tf1=context["slope_tf1"],
        slope_tf2=context["slope_tf2"],
        adx_vals=context["adx"],
        hours=context["hours"],
        cooldown=90,
        dates=context["dates"],
        start_date=FULL_START,
        end_date=FULL_END,
        use_session_filter=True,
        adx_threshold=30,
        use_kama_slope_filter=False,
        er_vals=context["er_5m"],
        er_threshold=0.0,
        bw_vals=context["bw_5m"],
        bw_sma_vals=np.zeros_like(context["bw_5m"]),
        bb_squeeze_lookback=0,
        htf_st_direction=np.zeros_like(context["bw_5m"]),
        htf_st_multiplier=0.0,
    )
    df_sanity = base_df.copy()
    df_sanity["long_entry"] = le
    df_sanity["long_exit"] = lx
    df_sanity["short_entry"] = se
    df_sanity["short_exit"] = sx
    sanity_kpis = _extract_metrics(_run_backtest(df_sanity, FULL_START, FULL_END))

    sanity_pass = (
        abs(sanity_kpis["pf"] - 3.3562) <= 0.001
        and abs(sanity_kpis["net_profit"] - 22.81) <= 0.10
        and sanity_kpis["trades"] == 17
    )
    print(
        "Sanity full-period (expect v1 best): "
        f"PF={_fmt_pf(sanity_kpis['pf'])}, Net={sanity_kpis['net_profit']:.2f}, Trades={sanity_kpis['trades']}, "
        f"PASS={sanity_pass}"
    )
    if not sanity_pass:
        raise RuntimeError(
            "Sanity check failed. Expected PF=3.3562 (+/-0.001), Net=22.81 (+/-0.10), Trades=17."
        )

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

    results.sort(
        key=lambda r: (
            r["oos"]["trades"] > 0,
            _pf_sort_key(r["oos"]["pf"]),
            r["oos"]["net_profit"],
        ),
        reverse=True,
    )
    top5 = results[:5]
    zero_oos_trade = [r for r in results if r["oos"]["trades"] == 0]

    with out_path.open("w", encoding="utf-8") as f:
        f.write("Pivot Breakout v2 Optimization Results\n")
        f.write("Goal: Improve breakout quality via ER + BB squeeze + HTF Supertrend filters\n")
        f.write(
            "Fixed baseline: kama_len=30, kama_fast=2, kama_slow=60, tf1=60, tf2=1440, pivot_len=8, adx_threshold=30, cooldown=90, use_session_filter=True, use_kama_slope_filter=False\n"
        )
        f.write(
            "Sweep: er_threshold in [0.0, 0.15, 0.25, 0.35], bb_squeeze_lookback in [0, 10, 20], htf_st_multiplier in [0.0, 2.0, 3.0]\n\n"
        )
        f.write("# Header Notes\n")
        f.write(
            f"- IS/OOS split: IS={IS_START}..{IS_END}, OOS={OOS_START}..{OOS_END}, FULL={FULL_START}..{FULL_END}\n"
        )
        f.write(
            "- Sanity check vs v1 full-period control (expect PF=3.3562, Net=22.81, Trades=17): "
            f"PF={_fmt_pf(sanity_kpis['pf'])}, Net={sanity_kpis['net_profit']:.2f}, Trades={sanity_kpis['trades']}, "
            f"{'PASS' if sanity_pass else 'FAIL'}\n"
        )
        f.write("- Top 5 candidates by OOS PF:\n")
        for idx, row in enumerate(top5, start=1):
            f.write(
                f"  {idx}) OOS PF={_fmt_pf(row['oos']['pf'])}, OOS Net={row['oos']['net_profit']:.2f}, OOS Trades={row['oos']['trades']}, "
                f"IS PF={_fmt_pf(row['is']['pf'])}, IS Net={row['is']['net_profit']:.2f}, IS Trades={row['is']['trades']}, "
                f"Params={row['params']}\n"
            )
        if zero_oos_trade:
            f.write("- OOS trades=0 combinations (overfitting warning):\n")
            for row in zero_oos_trade:
                f.write(f"  - {row['params']}\n")
        else:
            f.write("- OOS trades=0 combinations (overfitting warning): none\n")

        f.write("\n| IS PF | IS Net | IS Trades | OOS PF | OOS Net | OOS Trades | Params |\n")
        f.write("|---:|---:|---:|---:|---:|---:|:---|\n")
        for row in results:
            f.write(
                f"| {_fmt_pf(row['is']['pf'])} | "
                f"{row['is']['net_profit']:.2f} | "
                f"{row['is']['trades']} | "
                f"{_fmt_pf(row['oos']['pf'])} | "
                f"{row['oos']['net_profit']:.2f} | "
                f"{row['oos']['trades']} | "
                f"{row['params']} |\n"
            )

    if results:
        best = results[0]
        print(
            "Best by OOS PF: "
            f"OOS PF={_fmt_pf(best['oos']['pf'])}, OOS Net={best['oos']['net_profit']:.2f}, OOS Trades={best['oos']['trades']}, "
            f"IS PF={_fmt_pf(best['is']['pf'])}, IS Net={best['is']['net_profit']:.2f}, IS Trades={best['is']['trades']}, "
            f"Params={best['params']}"
        )
    print(f"Wrote leaderboard: {out_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
