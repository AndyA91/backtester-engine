"""
Generate synthetic forex data for testing signal libraries.

Creates realistic OHLCV data with trending/ranging regimes, session volume
patterns, and proper forex pip spreads. Not for strategy validation — use
TradingView exports for that. This is purely for signal development/testing.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_forex_daily(
    pair: str = "EURUSD",
    start: str = "2015-01-01",
    end: str = "2025-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic daily OHLCV data that behaves like a forex pair.

    Uses regime-switching (trending ↔ ranging) with realistic:
    - Daily volatility (~0.5-1.0% for majors)
    - Volume clustering (higher Mon-Wed, lower Fri)
    - Mean-reversion within ranges, momentum in trends
    """
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range(start=start, end=end, freq="B")  # business days
    n = len(dates)

    # Starting price
    if "JPY" in pair:
        price = 110.0
        pip = 0.01
    else:
        price = 1.1000
        pip = 0.0001

    daily_vol = 0.006  # ~0.6% daily volatility

    # Regime: 0 = ranging, 1 = trending up, -1 = trending down
    regime = np.zeros(n, dtype=int)
    regime[0] = 0
    for i in range(1, n):
        if rng.random() < 0.03:  # 3% chance of regime change per day
            regime[i] = rng.choice([-1, 0, 1])
        else:
            regime[i] = regime[i - 1]

    opens = np.zeros(n)
    highs = np.zeros(n)
    lows = np.zeros(n)
    closes = np.zeros(n)
    volumes = np.zeros(n)

    opens[0] = price
    for i in range(n):
        if i > 0:
            # Gap from previous close (small for forex)
            gap = rng.normal(0, daily_vol * 0.1) * closes[i - 1]
            opens[i] = closes[i - 1] + gap

        o = opens[i]

        # Daily return with regime drift
        drift = regime[i] * daily_vol * 0.3  # trend adds ~30% of vol as drift
        ret = rng.normal(drift, daily_vol)

        c = o * (1 + ret)

        # Intraday range (high/low)
        intraday_range = abs(ret) + rng.exponential(daily_vol * 0.5)
        if c >= o:
            highs[i] = o + abs(c - o) + rng.exponential(daily_vol * 0.3) * o
            lows[i] = min(o, c) - rng.exponential(daily_vol * 0.2) * o
        else:
            highs[i] = max(o, c) + rng.exponential(daily_vol * 0.2) * o
            lows[i] = o - abs(o - c) - rng.exponential(daily_vol * 0.3) * o

        closes[i] = c

        # Volume: higher mid-week, random clustering
        dow = dates[i].dayofweek  # 0=Mon, 4=Fri
        vol_mult = [1.0, 1.1, 1.15, 1.05, 0.85][dow]
        volumes[i] = max(1000, rng.lognormal(10, 0.5) * vol_mult)

    # Round to pip precision
    decimals = 2 if "JPY" in pair else 4
    opens = np.round(opens, decimals)
    highs = np.round(highs, decimals)
    lows = np.round(lows, decimals)
    closes = np.round(closes, decimals)
    volumes = np.round(volumes, 0)

    # Ensure OHLC consistency
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))

    df = pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes,
    }, index=dates)
    df.index.name = "Date"

    return df


def save_as_tv_export(df: pd.DataFrame, filename: str):
    """Save DataFrame in TradingView CSV export format (unix timestamps)."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    export = pd.DataFrame({
        "time": (df.index.astype(np.int64) // 10**9).astype(int),
        "open": df["Open"].values,
        "high": df["High"].values,
        "low": df["Low"].values,
        "close": df["Close"].values,
    })

    if "Volume" in df.columns:
        export["Volume"] = df["Volume"].values

    filepath = data_dir / filename
    export.to_csv(filepath, index=False)
    print(f"Saved synthetic data: {filepath} ({len(df)} bars)")
    return filepath


if __name__ == "__main__":
    # Generate and save test data
    for pair, seed in [("EURUSD", 42), ("GBPUSD", 43), ("USDJPY", 44)]:
        df = generate_forex_daily(pair=pair, seed=seed)
        save_as_tv_export(df, f"SYNTH_{pair}, 1D.csv")
        print(f"  {pair}: {df.index[0].date()} to {df.index[-1].date()}, "
              f"range {df['Close'].min():.4f} - {df['Close'].max():.4f}")
