"""
Renko data loader.

TV Renko exports use fractional Unix timestamps (e.g. 1674424800.001,
1674424800.002) to distinguish multiple bricks that formed within the same
second. The standard load_tv_export() does astype("int64") which truncates
all fractional parts — causing 10,827 of 20,003 bricks to get duplicate
DatetimeIndex entries. This module fixes that by passing the float column
directly to pd.to_datetime(..., unit="s").
"""

import pandas as pd
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_renko_export(filename: str) -> pd.DataFrame:
    """
    Load a TradingView Renko export CSV.

    Preserves fractional timestamps (multiple bricks per second) as
    millisecond-precision DatetimeIndex entries so every brick is unique.
    Adds a `brick_up` boolean column (True = up brick, False = down brick).
    Drops the last brick (may be unfinished).

    Args:
        filename: CSV filename inside the data/ directory.

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume, brick_up
        Index: DatetimeIndex (timezone-naive, named 'Date', millisecond precision)
    """
    filepath = _DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Renko export not found: {filepath}\n"
            f"Place CSV files in the data/ directory."
        )

    df = pd.read_csv(filepath)

    # KEY FIX: pass float directly — pd.to_datetime preserves sub-second offsets
    # as millisecond precision. astype("int64") would truncate them, causing
    # 10,827 duplicate index entries in the EURUSD 0.0004 dataset.
    dt = pd.to_datetime(df["time"], unit="s")
    if len(dt) > 0 and dt.max().year < 2000:
        # Scaled epoch fallback (e.g. HistData-style ms timestamps)
        dt = pd.to_datetime(df["time"].astype("int64") * 1000, unit="ms")

    df["Date"] = dt
    df = df.set_index("Date")
    df.index.name = "Date"

    df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
    })

    # Keep standard OHLCV columns; add dummy Volume if absent
    if "Volume" not in df.columns:
        df["Volume"] = 0
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    df = df.sort_index()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    # Drop last brick — may be unfinished
    if len(df) < 2:
        raise ValueError(
            f"Renko export has only {len(df)} brick(s) after filtering."
        )
    df = df.iloc[:-1]

    # Primary signal atom: direction of each brick
    df["brick_up"] = df["Close"] > df["Open"]

    print(f"Loaded Renko export: {len(df)} bricks from "
          f"{df.index[0]} to {df.index[-1]}")
    up = df["brick_up"].sum()
    dn = len(df) - up
    print(f"  Up bricks: {up} ({100*up/len(df):.1f}%), "
          f"Down bricks: {dn} ({100*dn/len(df):.1f}%)")
    dups = df.index.duplicated().sum()
    if dups:
        print(f"  WARNING: {dups} duplicate index entries detected!")
    return df
