"""
Portfolio Analysis — 5 Renko Strategies

Loads TV trade exports for each strategy, builds daily P&L series,
and produces:
  1. Correlation matrix of daily returns
  2. Trade overlap matrix
  3. Combined portfolio metrics (equal-weight)
  4. Optimal subset selection (3- and 4-strategy portfolios)
  5. Capital allocation (inverse-volatility)

Run from repo root:
  python renko/portfolio_analysis.py
"""

from pathlib import Path
from itertools import combinations
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

OOS_START = "2025-10-01"
OOS_END = "2026-03-19"
INITIAL_CAPITAL = 1000.0

STRATEGIES = {
    "GJ014": {
        "pair": "GBPJPY",
        "brick": 0.05,
        "csv": ROOT / "tvresults" / "GJ014_MK_Regime_+_HTF_ADX_[Renko_GBPJPY]_OANDA_GBPJPY_2026-03-20.csv",
    },
    "GU001": {
        "pair": "GBPUSD",
        "brick": 0.0004,
        "csv": ROOT / "tvresults" / "GU001_ESCGO_+_HTF_ADX_[Renko_GBPUSD]_OANDA_GBPUSD_2026-03-20.csv",
    },
    "EA021": {
        "pair": "EURAUD",
        "brick": 0.0006,
        "csv": ROOT / "renko" / "strategies" / "EA021_ESCGO_+_MACD_LC_+_HTF_ADX_[Renko_EURAUD]_OANDA_EURAUD_2026-03-20.csv",
    },
    "UJ001": {
        "pair": "USDJPY",
        "brick": 0.05,
        "csv": ROOT / "tvresults" / "UJ001_Stoch_+_MACD_LC_[Renko_USDJPY]_OANDA_USDJPY_2026-03-20.csv",
    },
    "R016": {
        "pair": "EURUSD",
        "brick": 0.0005,
        "csv": ROOT / "tvresults" / "R016_Stoch_Cross_+_HTF_ADX_[Renko_EURUSD_0.0005]_OANDA_EURUSD_2026-03-20.csv",
    },
}


# ---------------------------------------------------------------------------
# CSV Loading
# ---------------------------------------------------------------------------

def load_trades(path: Path) -> pd.DataFrame:
    """Load TV CSV, merge entry/exit rows, return trade-level DataFrame."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    entry_rows = df[df["Type"].str.contains("Entry", case=False)].copy()
    exit_rows = df[df["Type"].str.contains("Exit", case=False)].copy()

    entry_rows = entry_rows.rename(columns={
        "Date and time": "entry_time",
        "Type": "entry_type",
    })
    exit_rows = exit_rows.rename(columns={
        "Date and time": "exit_time",
        "Type": "exit_type",
        "Net P&L USD": "pnl",
    })

    trades = entry_rows[["Trade #", "entry_time", "entry_type"]].merge(
        exit_rows[["Trade #", "exit_time", "exit_type", "pnl"]],
        on="Trade #",
    )

    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    trades["exit_time"] = pd.to_datetime(trades["exit_time"])
    trades["is_long"] = trades["entry_type"].str.contains("long", case=False)
    trades["direction"] = trades["is_long"].map({True: "long", False: "short"})

    return trades


def load_oos_trades(name: str, cfg: dict) -> pd.DataFrame:
    """Load trades and filter to OOS period."""
    trades = load_trades(cfg["csv"])
    oos = trades[trades["entry_time"] >= OOS_START].copy()
    oos["strategy"] = name
    oos["pair"] = cfg["pair"]
    return oos


# ---------------------------------------------------------------------------
# Daily P&L Construction
# ---------------------------------------------------------------------------

def build_daily_pnl(trades: pd.DataFrame, name: str) -> pd.Series:
    """Build daily P&L series. P&L attributed to exit date (realized)."""
    trades = trades.copy()
    trades["exit_date"] = trades["exit_time"].dt.normalize()
    daily = trades.groupby("exit_date")["pnl"].sum()

    full_range = pd.date_range(OOS_START, OOS_END, freq="B")
    daily = daily.reindex(full_range, fill_value=0.0)
    daily.name = name
    return daily


def build_daily_pnl_matrix(all_oos: dict) -> pd.DataFrame:
    """Build NxM matrix: N business days x M strategies."""
    series = {}
    for name, trades in all_oos.items():
        series[name] = build_daily_pnl(trades, name)
    return pd.DataFrame(series)


# ---------------------------------------------------------------------------
# 1. Correlation Matrix
# ---------------------------------------------------------------------------

def correlation_analysis(daily_returns: pd.DataFrame) -> pd.DataFrame:
    return daily_returns.corr()


# ---------------------------------------------------------------------------
# 2. Trade Overlap Matrix
# ---------------------------------------------------------------------------

def build_position_calendar(trades: pd.DataFrame) -> pd.Series:
    """Mark each calendar day as 1 if strategy has a position open."""
    full_range = pd.date_range(OOS_START, OOS_END, freq="D")
    calendar = pd.Series(0, index=full_range)

    for _, t in trades.iterrows():
        entry_d = t["entry_time"].normalize()
        exit_d = t["exit_time"].normalize()
        mask = (calendar.index >= entry_d) & (calendar.index <= exit_d)
        calendar[mask] = 1

    return calendar


def trade_overlap_matrix(all_oos: dict) -> pd.DataFrame:
    """Overlap(A,B) = days_both_open / max(days_A_open, days_B_open)."""
    calendars = {name: build_position_calendar(trades) for name, trades in all_oos.items()}
    names = list(all_oos.keys())
    n = len(names)
    overlap = pd.DataFrame(0.0, index=names, columns=names)

    for i in range(n):
        for j in range(n):
            a = calendars[names[i]]
            b = calendars[names[j]]
            both_open = ((a == 1) & (b == 1)).sum()
            max_open = max(a.sum(), b.sum())
            overlap.iloc[i, j] = both_open / max_open if max_open > 0 else 0.0

    return overlap


# ---------------------------------------------------------------------------
# 3. Combined Portfolio Metrics
# ---------------------------------------------------------------------------

def _pf_from_series(pnl: pd.Series) -> float:
    gp = pnl[pnl > 0].sum()
    gl = abs(pnl[pnl < 0].sum())
    return gp / gl if gl > 0 else float("inf")


def portfolio_metrics(daily_pnl: pd.DataFrame) -> dict:
    """Equal-weight portfolio metrics."""
    port_daily = daily_pnl.sum(axis=1)
    total_capital = INITIAL_CAPITAL * len(daily_pnl.columns)

    equity = total_capital + port_daily.cumsum()
    pf = _pf_from_series(port_daily)

    daily_ret = port_daily / total_capital
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0.0

    downside = daily_ret[daily_ret < 0]
    downside_std = downside.std() if len(downside) > 1 else 0.0
    sortino = (daily_ret.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0.0

    running_max = equity.cummax()
    drawdown = equity - running_max
    max_dd = drawdown.min()
    max_dd_pct = (drawdown / running_max).min() * 100

    total_return = port_daily.sum()
    total_return_pct = (total_return / total_capital) * 100

    active_days = (port_daily != 0).sum()
    win_days = (port_daily > 0).sum()
    win_rate = win_days / active_days * 100 if active_days > 0 else 0

    return {
        "total_capital": total_capital,
        "total_return_usd": round(float(total_return), 2),
        "total_return_pct": round(float(total_return_pct), 2),
        "profit_factor": round(pf, 2) if pf != float("inf") else "inf",
        "sharpe": round(float(sharpe), 2),
        "sortino": round(float(sortino), 2),
        "max_drawdown_usd": round(float(max_dd), 2),
        "max_drawdown_pct": round(float(max_dd_pct), 2),
        "active_days": int(active_days),
        "win_day_rate": round(float(win_rate), 1),
    }


# ---------------------------------------------------------------------------
# 4. Optimal Subset Selection
# ---------------------------------------------------------------------------

def optimal_subset(daily_pnl: pd.DataFrame, sizes=None) -> dict:
    """Evaluate all subsets, score = Sharpe * (1 - avg_corr)."""
    if sizes is None:
        sizes = [3, 4]
    names = list(daily_pnl.columns)
    daily_ret = daily_pnl / INITIAL_CAPITAL
    corr_matrix = daily_ret.corr()

    results = {}
    for n in sizes:
        all_combos = []
        for combo in combinations(names, n):
            sub = daily_pnl[list(combo)]
            port = sub.sum(axis=1)
            cap = INITIAL_CAPITAL * n

            ret = port / cap
            sharpe = (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() > 0 else 0

            sub_corr = corr_matrix.loc[list(combo), list(combo)]
            mask = np.triu(np.ones(sub_corr.shape, dtype=bool), k=1)
            avg_corr = sub_corr.where(mask).stack().mean()

            pf = _pf_from_series(port)
            total_ret = port.sum()

            score = sharpe * (1 - avg_corr)

            all_combos.append({
                "strategies": list(combo),
                "sharpe": round(float(sharpe), 3),
                "avg_corr": round(float(avg_corr), 4),
                "pf": round(pf, 2) if pf != float("inf") else "inf",
                "total_return_usd": round(float(total_ret), 2),
                "score": round(float(score), 3),
            })

        all_combos.sort(key=lambda x: x["score"], reverse=True)
        results[f"best_{n}"] = {
            "best": all_combos[0] if all_combos else None,
            "all": all_combos,
        }

    return results


# ---------------------------------------------------------------------------
# 5. Capital Allocation
# ---------------------------------------------------------------------------

def capital_allocation(daily_pnl: pd.DataFrame) -> dict:
    """Inverse-volatility weighting."""
    daily_ret = daily_pnl / INITIAL_CAPITAL
    vols = daily_ret.std()

    min_nonzero = vols[vols > 0].min() if (vols > 0).any() else 1e-6
    vols_safe = vols.replace(0, min_nonzero)

    inv_vol = 1.0 / vols_safe
    weights = inv_vol / inv_vol.sum()

    n = len(daily_pnl.columns)
    total_cap = INITIAL_CAPITAL * n

    alloc = {}
    for name in daily_pnl.columns:
        alloc[name] = {
            "daily_vol": round(float(vols[name]), 6),
            "inv_vol_weight": round(float(weights[name]), 4),
            "inv_vol_capital": round(float(weights[name]) * total_cap, 2),
            "equal_weight": round(1.0 / n, 4),
            "equal_capital": round(total_cap / n, 2),
        }

    return alloc


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_header():
    print("=" * 70)
    print("  PORTFOLIO ANALYSIS — 5 Renko Strategies (OOS)")
    print(f"  Period: {OOS_START} to {OOS_END}")
    print("=" * 70)


def print_strategy_summary(all_oos: dict):
    print("\n--- Strategy Summary ---")
    print(f"{'Strategy':<10}{'Pair':<10}{'Trades':>7}{'Wins':>6}{'WR':>8}{'Net P&L':>10}{'PF':>8}")
    for name, trades in all_oos.items():
        n_trades = len(trades)
        wins = (trades["pnl"] > 0).sum()
        wr = wins / n_trades * 100 if n_trades > 0 else 0
        net = trades["pnl"].sum()
        pf = _pf_from_series(trades["pnl"])
        pf_str = "inf" if pf == float("inf") else f"{pf:.2f}"
        print(f"{name:<10}{STRATEGIES[name]['pair']:<10}{n_trades:>7}{wins:>6}{wr:>7.1f}%{net:>9.2f}{pf_str:>8}")


def print_matrix(title: str, mat: pd.DataFrame, fmt: str = ".3f"):
    print(f"\n--- {title} ---")
    names = list(mat.columns)
    header = f"{'':>8}" + "".join(f"{n:>8}" for n in names)
    print(header)
    for name in names:
        row = f"{name:>8}"
        for col in names:
            row += f"{mat.loc[name, col]:>8{fmt}}"
        print(row)


def print_portfolio(port: dict):
    print("\n--- Combined Portfolio (Equal-Weight) ---")
    print(f"  Total capital:    ${port['total_capital']:.0f}")
    print(f"  Total return:     ${port['total_return_usd']:.2f} ({port['total_return_pct']:.2f}%)")
    print(f"  Profit Factor:    {port['profit_factor']}")
    print(f"  Sharpe:           {port['sharpe']:.2f}")
    print(f"  Sortino:          {port['sortino']:.2f}")
    print(f"  Max Drawdown:     ${port['max_drawdown_usd']:.2f} ({port['max_drawdown_pct']:.2f}%)")
    print(f"  Active days:      {port['active_days']}")
    print(f"  Win-day rate:     {port['win_day_rate']:.1f}%")


def print_subsets(subsets: dict):
    print("\n--- Optimal Subsets ---")
    for key, data in subsets.items():
        best = data["best"]
        if best is None:
            continue
        size = key.split("_")[1]
        strats = ", ".join(best["strategies"])
        print(f"  Best {size}-strategy: [{strats}]")
        print(f"    Sharpe={best['sharpe']:.3f}  Avg Corr={best['avg_corr']:.4f}  "
              f"PF={best['pf']}  Return=${best['total_return_usd']:.2f}  "
              f"Score={best['score']:.3f}")

        print(f"  All {size}-strategy combos:")
        for c in data["all"]:
            s = ", ".join(c["strategies"])
            print(f"    [{s}]  Sharpe={c['sharpe']:.3f}  Corr={c['avg_corr']:.4f}  "
                  f"PF={c['pf']}  Score={c['score']:.3f}")


def print_allocation(alloc: dict):
    print("\n--- Capital Allocation ---")
    print(f"{'Strategy':<10}{'Daily Vol':>12}{'Inv-Vol Wt':>12}{'Inv-Vol $':>10}{'Equal $':>10}")
    for name, a in alloc.items():
        print(f"{name:<10}{a['daily_vol']:>12.6f}{a['inv_vol_weight']:>11.1%}"
              f"{a['inv_vol_capital']:>10.0f}{a['equal_capital']:>10.0f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print_header()

    # Load OOS trades
    all_oos = {}
    for name, cfg in STRATEGIES.items():
        all_oos[name] = load_oos_trades(name, cfg)

    # Strategy summary
    print_strategy_summary(all_oos)

    # Daily P&L matrix
    daily_pnl = build_daily_pnl_matrix(all_oos)
    daily_ret = daily_pnl / INITIAL_CAPITAL

    # 1. Correlation
    corr = correlation_analysis(daily_ret)
    print_matrix("Daily Return Correlation Matrix", corr)

    # 2. Trade overlap
    overlap = trade_overlap_matrix(all_oos)
    print_matrix("Trade Overlap Matrix (fraction of max open days)", overlap)

    # 3. Combined portfolio
    port = portfolio_metrics(daily_pnl)
    print_portfolio(port)

    # Per-strategy metrics for comparison
    print("\n--- Per-Strategy Metrics ---")
    print(f"{'Strategy':<10}{'Sharpe':>8}{'Sortino':>8}{'Max DD%':>8}")
    for name in daily_pnl.columns:
        s_daily = daily_pnl[name]
        s_ret = s_daily / INITIAL_CAPITAL
        sharpe = (s_ret.mean() / s_ret.std()) * np.sqrt(252) if s_ret.std() > 0 else 0
        downside = s_ret[s_ret < 0]
        ds_std = downside.std() if len(downside) > 1 else 0
        sortino = (s_ret.mean() / ds_std) * np.sqrt(252) if ds_std > 0 else 0
        eq = INITIAL_CAPITAL + s_daily.cumsum()
        dd_pct = ((eq - eq.cummax()) / eq.cummax()).min() * 100
        print(f"{name:<10}{sharpe:>8.2f}{sortino:>8.2f}{dd_pct:>7.2f}%")

    # 4. Optimal subsets
    subsets = optimal_subset(daily_pnl)
    print_subsets(subsets)

    # 5. Capital allocation
    alloc = capital_allocation(daily_pnl)
    print_allocation(alloc)

    # Save JSON
    results = {
        "oos_period": {"start": OOS_START, "end": OOS_END},
        "strategies": {k: {"pair": v["pair"], "brick": v["brick"]} for k, v in STRATEGIES.items()},
        "strategy_summary": {},
        "correlation_matrix": corr.round(4).to_dict(),
        "trade_overlap_matrix": overlap.round(4).to_dict(),
        "portfolio_metrics": port,
        "optimal_subsets": subsets,
        "capital_allocation": alloc,
    }
    for name, trades in all_oos.items():
        n_t = len(trades)
        wins = int((trades["pnl"] > 0).sum())
        pf = _pf_from_series(trades["pnl"])
        results["strategy_summary"][name] = {
            "trades": n_t,
            "wins": wins,
            "win_rate": round(wins / n_t * 100, 1) if n_t > 0 else 0,
            "net_pnl": round(float(trades["pnl"].sum()), 2),
            "pf": round(pf, 2) if pf != float("inf") else "inf",
        }

    out_path = ROOT / "ai_context" / "portfolio_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[Saved -> {out_path}]")


if __name__ == "__main__":
    main()
