"""
Candle data loader for 1-min (and other timeframe) CSV exports from TradingView.

Format: time,open,high,low,close  (unix epoch seconds, no volume)
"""

import pandas as pd
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_DIR / "data"


def load_candle_csv(filename: str, instrument_dir: str = "MYM") -> pd.DataFrame:
    """
    Load a TradingView candle CSV export.

    Args:
        filename: CSV filename (e.g. "CBOT_MINI_MYM1!, 1.csv")
        instrument_dir: subdirectory under data/ (e.g. "MYM")

    Returns:
        DataFrame with DatetimeIndex and columns: Open, High, Low, Close
    """
    path = _DATA_DIR / instrument_dir / filename
    df = pd.read_csv(path, encoding="utf-8-sig")

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    df["datetime"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("datetime").sort_index()

    # Rename to engine convention (capitalized)
    df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
    })

    # Keep only OHLC + any extra columns (volume if present)
    keep = [c for c in ["Open", "High", "Low", "Close", "volume"] if c in df.columns]
    df = df[keep]

    # Drop last bar (incomplete candle)
    df = df.iloc[:-1]

    return df


def resample_ohlc(df: pd.DataFrame, tf_minutes: int) -> pd.DataFrame:
    """
    Resample 1-min OHLC to a higher timeframe.

    Args:
        df: 1-min DataFrame with Open, High, Low, Close columns.
        tf_minutes: Target timeframe in minutes (e.g. 5, 15, 30).

    Returns:
        Resampled DataFrame with same column structure.
    """
    if tf_minutes <= 1:
        return df

    rule = f"{tf_minutes}min"
    resampled = df.resample(rule).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
    }).dropna()

    return resampled
