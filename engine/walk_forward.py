"""
Walk-Forward Optimization (WFO) for the backtesting engine.

Splits historical data into rolling train/test windows, optimizes strategy
parameters on each training period, then evaluates on the out-of-sample test
period. This detects overfitting and measures how well a strategy generalises.

Usage:
    from engine import run_walk_forward

    def my_signal_fn(df, fast=9, slow=21):
        df = df.copy()
        df["fast_ema"] = calc_ema(df["Close"], fast)
        df["slow_ema"] = calc_ema(df["Close"], slow)
        df["long_entry"] = detect_crossover(df["fast_ema"], df["slow_ema"])
        df["long_exit"] = detect_crossunder(df["fast_ema"], df["slow_ema"])
        return df

    param_grid = {"fast": [5, 9, 12], "slow": [18, 21, 26]}

    results = run_walk_forward(
        df, my_signal_fn, param_grid,
        config=BacktestConfig(),
        train_months=12, test_months=3,
    )
    print_wfo_results(results)
"""

import pandas as pd
import numpy as np
from itertools import product
from dataclasses import dataclass
from typing import Callable

from engine.engine import BacktestConfig, run_backtest, run_backtest_long_short


@dataclass
class WFOWindow:
    """Results for a single walk-forward window."""
    window_num: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict
    train_net_profit_pct: float
    train_profit_factor: float
    train_max_dd_pct: float
    train_trades: int
    test_net_profit_pct: float
    test_profit_factor: float
    test_max_dd_pct: float
    test_trades: int


def _build_param_combos(param_grid: dict) -> list[dict]:
    """Expand a param grid into a list of param dicts."""
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def _run_single(
    df: pd.DataFrame,
    signal_fn: Callable,
    params: dict,
    config: BacktestConfig,
    long_short: bool,
) -> dict:
    """Run a single backtest with given params and return KPIs."""
    df_sig = signal_fn(df, **params)
    if long_short:
        return run_backtest_long_short(df_sig, config)
    return run_backtest(df_sig, config)


def _score(kpis: dict, objective: str) -> float:
    """Extract the optimization objective from KPIs. Higher is better."""
    if "error" in kpis:
        return float("-inf")
    if objective == "net_profit_pct":
        return kpis.get("net_profit_pct", float("-inf"))
    elif objective == "profit_factor":
        pf = kpis.get("profit_factor", 0)
        return pf if pf != float("inf") else 0
    elif objective == "sharpe":
        # Approximate Sharpe from trade returns
        trades = [t for t in kpis.get("trades", []) if t.exit_date is not None]
        if len(trades) < 2:
            return float("-inf")
        returns = [t.pnl_pct for t in trades]
        mean_r = np.mean(returns)
        std_r = np.std(returns, ddof=1)
        return mean_r / std_r if std_r > 0 else 0
    elif objective == "calmar":
        # Net profit % / max drawdown %
        net_pct = kpis.get("net_profit_pct", 0)
        dd_pct = abs(kpis.get("max_drawdown_pct", 100))
        return net_pct / dd_pct if dd_pct > 0 else 0
    else:
        return kpis.get(objective, float("-inf"))


def run_walk_forward(
    df: pd.DataFrame,
    signal_fn: Callable,
    param_grid: dict,
    config: BacktestConfig = None,
    train_months: int = 12,
    test_months: int = 3,
    objective: str = "net_profit_pct",
    min_trades: int = 3,
    long_short: bool = False,
    warmup_bars: int = 100,
    anchored: bool = False,
) -> dict:
    """
    Run walk-forward optimization.

    Args:
        df: Full OHLCV DataFrame (with DatetimeIndex).
        signal_fn: Function(df, **params) -> df with signal columns.
        param_grid: Dict of param_name -> list of values to test.
            Example: {"fast": [5, 9, 12], "slow": [18, 21, 26]}
        config: BacktestConfig (start_date/end_date are overridden per window).
        train_months: Length of each training window in months.
        test_months: Length of each test window in months.
        objective: Metric to optimise. One of:
            "net_profit_pct", "profit_factor", "sharpe", "calmar"
        min_trades: Skip param combos with fewer trades than this.
        long_short: Use run_backtest_long_short instead of run_backtest.
        warmup_bars: Number of bars before each train window for indicator warmup.
        anchored: If True, training window always starts from the beginning of
            data (expanding window). If False, rolling window.

    Returns:
        Dict with:
            - "windows": list of WFOWindow results
            - "oos_equity_curve": combined out-of-sample equity curve
            - "oos_net_profit_pct": combined OOS net profit %
            - "oos_profit_factor": combined OOS profit factor
            - "oos_max_dd_pct": combined OOS max drawdown %
            - "oos_total_trades": combined OOS trade count
            - "efficiency_ratio": OOS net profit / avg IS net profit
            - "param_combos_tested": number of param combinations
    """
    if config is None:
        config = BacktestConfig()

    combos = _build_param_combos(param_grid)
    n_combos = len(combos)

    # Determine date boundaries
    data_start = df.index[0]
    data_end = df.index[-1]

    # Build windows
    windows_spec = []
    anchor_start = data_start + pd.DateOffset(months=0)

    # First train starts after enough warmup data
    first_train_start = df.index[min(warmup_bars, len(df) - 1)]
    current_train_start = first_train_start

    window_num = 0
    while True:
        train_end = current_train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)

        # Check we have enough data for the test window
        if test_start >= data_end:
            break

        # Clamp test_end to data end
        if test_end > data_end:
            test_end = data_end

        actual_train_start = anchor_start if anchored else current_train_start

        windows_spec.append({
            "num": window_num + 1,
            "warmup_start": actual_train_start - pd.DateOffset(days=warmup_bars * 2),
            "train_start": actual_train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        })
        window_num += 1
        current_train_start = test_start  # next train starts where test ended

    if not windows_spec:
        raise ValueError(
            f"Not enough data for walk-forward. Data spans "
            f"{data_start.date()} to {data_end.date()}, need at least "
            f"{train_months + test_months} months."
        )

    print(f"\n{'='*65}")
    print(f"  WALK-FORWARD OPTIMIZATION")
    print(f"{'='*65}")
    print(f"  Data:        {data_start.date()} to {data_end.date()}")
    print(f"  Train:       {train_months} months  |  Test: {test_months} months")
    print(f"  Windows:     {len(windows_spec)}")
    print(f"  Params:      {n_combos} combinations")
    print(f"  Objective:   {objective}")
    print(f"  Mode:        {'anchored (expanding)' if anchored else 'rolling'}")
    print(f"{'='*65}")

    wfo_windows: list[WFOWindow] = []
    oos_trades_all = []
    oos_returns = []  # per-window OOS return multiplier

    for w in windows_spec:
        print(f"\n  Window {w['num']}/{len(windows_spec)}: "
              f"Train {w['train_start'].date()}..{w['train_end'].date()} | "
              f"Test {w['test_start'].date()}..{w['test_end'].date()}")

        # Slice data for training (include warmup bars before train_start)
        df_train_full = df[df.index <= w["train_end"]].copy()
        if len(df_train_full) < warmup_bars + 10:
            print(f"    Skipping — not enough training data")
            continue

        # -- OPTIMISE on training window --
        best_score = float("-inf")
        best_params = combos[0]
        best_train_kpis = None

        train_config = BacktestConfig(
            initial_capital=config.initial_capital,
            commission_pct=config.commission_pct,
            slippage_ticks=config.slippage_ticks,
            qty_type=config.qty_type,
            qty_value=config.qty_value,
            pyramiding=config.pyramiding,
            start_date=str(w["train_start"].date()),
            end_date=str(w["train_end"].date()),
            process_orders_on_close=config.process_orders_on_close,
        )

        for params in combos:
            kpis = _run_single(df_train_full, signal_fn, params, train_config, long_short)
            if "error" in kpis:
                continue
            if kpis.get("total_trades", 0) < min_trades:
                continue
            s = _score(kpis, objective)
            if s > best_score:
                best_score = s
                best_params = params
                best_train_kpis = kpis

        if best_train_kpis is None or "error" in (best_train_kpis or {}):
            print(f"    No valid params found in training — skipping window")
            continue

        # -- EVALUATE on test window with best params --
        df_test_full = df[df.index <= w["test_end"]].copy()

        test_config = BacktestConfig(
            initial_capital=config.initial_capital,
            commission_pct=config.commission_pct,
            slippage_ticks=config.slippage_ticks,
            qty_type=config.qty_type,
            qty_value=config.qty_value,
            pyramiding=config.pyramiding,
            start_date=str(w["test_start"].date()),
            end_date=str(w["test_end"].date()),
            process_orders_on_close=config.process_orders_on_close,
        )

        test_kpis = _run_single(df_test_full, signal_fn, best_params, test_config, long_short)

        if "error" in test_kpis:
            test_net_pct = 0.0
            test_pf = 0.0
            test_dd_pct = 0.0
            test_trades_n = 0
        else:
            test_net_pct = test_kpis.get("net_profit_pct", 0)
            test_pf = test_kpis.get("profit_factor", 0)
            test_dd_pct = test_kpis.get("max_drawdown_pct", 0)
            test_trades_n = test_kpis.get("total_trades", 0)
            closed = [t for t in test_kpis.get("trades", []) if t.exit_date is not None]
            oos_trades_all.extend(closed)

        oos_returns.append(1 + test_net_pct / 100)

        wfo_result = WFOWindow(
            window_num=w["num"],
            train_start=w["train_start"],
            train_end=w["train_end"],
            test_start=w["test_start"],
            test_end=w["test_end"],
            best_params=best_params,
            train_net_profit_pct=best_train_kpis.get("net_profit_pct", 0),
            train_profit_factor=best_train_kpis.get("profit_factor", 0),
            train_max_dd_pct=best_train_kpis.get("max_drawdown_pct", 0),
            train_trades=best_train_kpis.get("total_trades", 0),
            test_net_profit_pct=test_net_pct,
            test_profit_factor=test_pf if test_pf != float("inf") else 0,
            test_max_dd_pct=test_dd_pct,
            test_trades=test_trades_n,
        )
        wfo_windows.append(wfo_result)

        params_str = ", ".join(f"{k}={v}" for k, v in best_params.items())
        print(f"    Best: {{{params_str}}}")
        print(f"    Train: {best_train_kpis['net_profit_pct']:+.1f}%  "
              f"PF={best_train_kpis['profit_factor']:.2f}  "
              f"DD={best_train_kpis['max_drawdown_pct']:.1f}%  "
              f"({best_train_kpis['total_trades']} trades)")
        print(f"    Test:  {test_net_pct:+.1f}%  "
              f"PF={test_pf:.2f}  "
              f"DD={test_dd_pct:.1f}%  "
              f"({test_trades_n} trades)")

    # -- Aggregate OOS results --
    if not wfo_windows:
        return {"error": "No valid walk-forward windows produced results"}

    # Combined OOS equity = product of per-window returns
    oos_equity = config.initial_capital
    for r in oos_returns:
        oos_equity *= r
    oos_net_profit_pct = (oos_equity / config.initial_capital - 1) * 100

    # OOS profit factor from all OOS trades
    oos_gross_profit = sum(t.pnl for t in oos_trades_all if t.pnl > 0)
    oos_gross_loss = sum(t.pnl for t in oos_trades_all if t.pnl <= 0)
    oos_pf = abs(oos_gross_profit / oos_gross_loss) if oos_gross_loss != 0 else float("inf")

    # OOS max drawdown (worst single window)
    oos_max_dd = min(w.test_max_dd_pct for w in wfo_windows)

    # Efficiency ratio: OOS performance / avg IS performance
    avg_is_return = np.mean([w.train_net_profit_pct for w in wfo_windows])
    avg_oos_return = np.mean([w.test_net_profit_pct for w in wfo_windows])
    efficiency = avg_oos_return / avg_is_return if avg_is_return != 0 else 0

    # Win rate across windows
    profitable_windows = sum(1 for w in wfo_windows if w.test_net_profit_pct > 0)

    return {
        "windows": wfo_windows,
        "oos_net_profit_pct": oos_net_profit_pct,
        "oos_profit_factor": oos_pf,
        "oos_max_dd_pct": oos_max_dd,
        "oos_total_trades": len(oos_trades_all),
        "oos_final_equity": oos_equity,
        "efficiency_ratio": efficiency,
        "avg_is_return": avg_is_return,
        "avg_oos_return": avg_oos_return,
        "profitable_windows": profitable_windows,
        "total_windows": len(wfo_windows),
        "param_combos_tested": n_combos,
    }


def print_wfo_results(results: dict):
    """Pretty-print walk-forward optimization results."""
    if "error" in results:
        print(f"\n  WFO Error: {results['error']}")
        return

    windows = results["windows"]

    print(f"\n{'='*75}")
    print(f"  WALK-FORWARD OPTIMIZATION RESULTS")
    print(f"{'='*75}")

    # Per-window table
    print(f"\n  {'Win':>3s}  {'Train Period':<25s}  {'Test Period':<25s}  "
          f"{'IS %':>7s}  {'OOS %':>7s}  {'OOS PF':>6s}  {'Params'}")
    print(f"  {'-'*100}")

    for w in windows:
        params_str = ", ".join(f"{k}={v}" for k, v in w.best_params.items())
        oos_marker = "+" if w.test_net_profit_pct > 0 else " "
        print(f"  {w.window_num:>3d}  "
              f"{w.train_start.date()} - {w.train_end.date()}  "
              f"{w.test_start.date()} - {w.test_end.date()}  "
              f"{w.train_net_profit_pct:>+6.1f}%  "
              f"{oos_marker}{w.test_net_profit_pct:>+5.1f}%  "
              f"{w.test_profit_factor:>6.2f}  "
              f"{params_str}")

    # Summary
    print(f"\n  {'─'*60}")
    print(f"  COMBINED OUT-OF-SAMPLE PERFORMANCE")
    print(f"  {'─'*60}")
    print(f"  OOS Net Profit:      {results['oos_net_profit_pct']:>+8.2f}%  "
          f"(${results['oos_final_equity']:,.2f} from ${windows[0].train_start.date()})")
    print(f"  OOS Profit Factor:   {results['oos_profit_factor']:>8.2f}")
    print(f"  OOS Max DD (window): {results['oos_max_dd_pct']:>8.2f}%")
    print(f"  OOS Total Trades:    {results['oos_total_trades']:>8d}")
    print(f"  Profitable Windows:  {results['profitable_windows']}/{results['total_windows']}")

    print(f"\n  OVERFITTING ANALYSIS")
    print(f"  {'─'*60}")
    print(f"  Avg In-Sample Return:      {results['avg_is_return']:>+8.2f}%")
    print(f"  Avg Out-of-Sample Return:  {results['avg_oos_return']:>+8.2f}%")
    print(f"  Efficiency Ratio:          {results['efficiency_ratio']:>8.2f}")
    print(f"  Param Combos Tested:       {results['param_combos_tested']:>8d}")

    # Interpretation
    eff = results["efficiency_ratio"]
    if eff >= 0.5:
        verdict = "GOOD — strategy generalises well (eff >= 0.50)"
    elif eff >= 0.25:
        verdict = "FAIR — moderate overfitting detected (0.25 <= eff < 0.50)"
    elif eff > 0:
        verdict = "WEAK — significant overfitting (0 < eff < 0.25)"
    else:
        verdict = "FAIL — strategy does not generalise (eff <= 0)"

    pwin = results["profitable_windows"] / results["total_windows"] if results["total_windows"] > 0 else 0
    if pwin >= 0.6:
        consistency = "Consistent — profitable in majority of windows"
    elif pwin >= 0.4:
        consistency = "Mixed — roughly half the windows profitable"
    else:
        consistency = "Inconsistent — most windows unprofitable"

    print(f"\n  Verdict:      {verdict}")
    print(f"  Consistency:  {consistency}")
    print(f"{'='*75}")
