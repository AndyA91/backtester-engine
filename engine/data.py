"""
Data loading utilities for backtesting.

Supports:
- TradingView CSV exports (recommended for exact OHLC match)
- Bitstamp API (free, no auth, fallback for BTC/USD daily)
- Multi-exchange crypto fetcher via ccxt (any pair, any timeframe)

Important: The last bar in live data is always dropped — it represents
an unfinished candle that would produce unreliable signals.
"""

import requests
import time
import pandas as pd
import numpy as np
from pathlib import Path

_ENGINE_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _ENGINE_DIR.parent          # dev/ or ship/
_DATA_DIR = _PROJECT_DIR / "data"
CACHE_DIR = _DATA_DIR / "cache"


def _parse_date_range(raw: str) -> tuple[str, str]:
    """Parse a date range string like 'Jan 02, 2018 — Feb 17, 2026'
    into two ISO-formatted date strings."""
    parts = raw.split("\u2014")            # split on em dash '—'
    if len(parts) != 2:
        parts = raw.split("-")             # fallback: hyphen
    start_raw, end_raw = parts[0].strip(), parts[1].strip()

    start_ts = pd.to_datetime(start_raw)
    end_ts = pd.to_datetime(end_raw)

    # Use date-only format for daily bars, datetime for intraday
    fmt = "%Y-%m-%d %H:%M" if (start_ts.hour or start_ts.minute
                                or end_ts.hour or end_ts.minute) else "%Y-%m-%d"
    return start_ts.strftime(fmt), end_ts.strftime(fmt)


def _fmt_date(ts: pd.Timestamp) -> str:
    """Format a timestamp as date-only or datetime depending on time component."""
    if ts.hour or ts.minute:
        return ts.strftime("%Y-%m-%d %H:%M")
    return ts.strftime("%Y-%m-%d")


def read_tv_xlsx_dates(xlsx_path: str | Path) -> dict:
    """
    Read all date parameters from a TradingView XLSX export.

    Reads the Properties sheet and returns a dict with:

    - ``pine_start``, ``pine_end``: The Pine script's Start Date / End Date
      input parameters.  These are what the script uses for its
      ``timeCondition`` (``time >= startDate and time <= endDate``).
    - ``range_start``, ``range_end``: The observed Trading Range — the
      actual date span over which trades occurred in the TV export.

    For most strategies, use ``pine_start``/``pine_end`` as the signal
    generator's ``start_date``/``end_date``.  Use ``range_end`` as
    ``BacktestConfig.end_date`` to clip the engine's bar range to match
    the TV export (important when CSV data extends beyond the XLSX
    export date).

    Parameters
    ----------
    xlsx_path : str or Path
        Path to a TradingView ``.xlsx`` export file.

    Returns
    -------
    dict
        ``{"pine_start": str, "pine_end": str,
          "range_start": str, "range_end": str}``
    """
    props = pd.read_excel(xlsx_path, sheet_name="Properties")

    def _prop(name):
        row = props[props["name"] == name]
        if row.empty:
            return None
        return row["value"].values[0]

    # Pine script date parameters — TV uses different property names
    # depending on the Pine version / export:
    #   "Start Date" / "End Date"
    #   "Date Start" / "Date End"
    #   "Backtest Start Date" / "Backtest End Date"
    pine_start_raw = (_prop("Start Date")
                      or _prop("Date Start")
                      or _prop("Backtest Start Date"))
    pine_end_raw = (_prop("End Date")
                    or _prop("Date End")
                    or _prop("Backtest End Date"))
    if pine_start_raw is None:
        raise ValueError(
            f"Missing Pine start date in {xlsx_path} Properties. "
            f"Looked for 'Start Date', 'Date Start', 'Backtest Start Date'."
        )
    pine_start = _fmt_date(pd.to_datetime(pine_start_raw))
    # Some Pine scripts have no explicit end date — default to far future
    pine_end = (_fmt_date(pd.to_datetime(pine_end_raw))
                if pine_end_raw is not None else "2100-12-31")

    # Observed trading range
    range_raw = _prop("Trading range")
    if range_raw is None:
        raise ValueError(f"No 'Trading range' row in {xlsx_path} Properties sheet")
    range_start, range_end = _parse_date_range(range_raw)

    return {
        "pine_start": pine_start,
        "pine_end": pine_end,
        "range_start": range_start,
        "range_end": range_end,
    }


def load_tv_export(filename: str = "INDEX_BTCUSD, 1D.csv") -> pd.DataFrame:
    """
    Load a TradingView-exported CSV file.

    TV export format: time (unix timestamp), open, high, low, close.
    The last bar is dropped because it is an unfinished (still-printing) candle.

    Args:
        filename: Name of the CSV file in the ``data/`` directory.

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
        Index: DatetimeIndex (timezone-naive, named 'Date')
    """
    filepath = _DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"TV export not found: {filepath}\n"
            f"Place CSV files in the data/ directory."
        )

    df = pd.read_csv(filepath)

    # Convert unix timestamps to datetime index.
    # Some HistData-built files store scaled epochs (e.g. 1704146 instead of
    # 1704146400). If that parses into 1970-era dates, upscale by 1000.
    raw_time = df["time"].astype("int64")
    dt = pd.to_datetime(raw_time, unit="s")
    if len(dt) > 0 and dt.max().year < 2000:
        dt = pd.to_datetime(raw_time * 1000, unit="s")
    df["Date"] = dt
    df = df.set_index("Date")

    # Rename to standard column names
    df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
    })

    # Keep only OHLC(V) columns + optional OnBalanceVolume
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume", "OnBalanceVolume"]
            if c in df.columns]
    df = df[cols]

    # Add dummy volume if not present (TV exports don't include volume)
    if "Volume" not in df.columns:
        df["Volume"] = 0

    df = df.sort_index()

    # Only drop rows where OHLC data is missing — auxiliary columns like
    # OnBalanceVolume may legitimately be NaN on the first bar.
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    # Drop the last bar — it is an unfinished candle
    if len(df) < 2:
        raise ValueError(
            f"TV export has only {len(df)} bar(s) after filtering — "
            f"need at least 2 (1 bar is dropped as unfinished candle)."
        )
    dropped_date = df.index[-1]
    df = df.iloc[:-1]

    print(f"Loaded TV export: {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Dropped last bar (unfinished candle): {dropped_date.date()}")
    return df


# ---------------------------------------------------------------------------
# Bitstamp API (fallback — prices differ slightly from INDEX:BTCUSD)
# ---------------------------------------------------------------------------

BITSTAMP_OHLC_URL = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"


def _fetch_bitstamp_chunk(end_ts: int, step: int = 86400, limit: int = 1000) -> list[dict]:
    """Fetch up to `limit` daily candles ending before `end_ts` from Bitstamp."""
    params = {"step": step, "limit": limit, "end": end_ts}
    resp = requests.get(BITSTAMP_OHLC_URL, params=params)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", {}).get("ohlc", [])


def fetch_btc_daily(
    start: str = "2017-01-01",
    end: str = "2026-12-31",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch BTC/USD daily OHLCV data from Bitstamp.

    The last bar is dropped because it is an unfinished (still-printing) candle.

    Args:
        start: Start date for data fetch (include warmup period).
        end:   End date for data fetch.
        use_cache: Cache data locally as CSV.

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
        Index: DatetimeIndex (timezone-naive, named 'Date')
    """
    cache_file = CACHE_DIR / f"BITSTAMP-BTCUSD_{start}_{end}_1d.csv"

    if use_cache and cache_file.exists():
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        print(f"Loaded cached data: {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")
        return df

    print(f"Fetching BITSTAMP:BTCUSD daily data from {start} to {end}...")

    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int(pd.Timestamp(end).timestamp())

    all_candles: list[dict] = []
    cursor = end_ts

    while cursor > start_ts:
        chunk = _fetch_bitstamp_chunk(end_ts=cursor, step=86400, limit=1000)
        if not chunk:
            break

        all_candles.extend(chunk)

        first_ts = int(chunk[0]["timestamp"])
        if first_ts >= cursor:
            break
        cursor = first_ts
        print(f"  Fetched {len(all_candles)} candles so far "
              f"(back to {pd.Timestamp(first_ts, unit='s').date()})...")
        time.sleep(0.3)

    if not all_candles:
        raise ValueError("No data returned from Bitstamp API")

    # Build DataFrame
    df = pd.DataFrame(all_candles)
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    df["Date"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.set_index("Date")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[df["Volume"] > 0]

    # Drop the last bar — it is an unfinished candle
    dropped_date = df.index[-1]
    df = df.iloc[:-1]

    print(f"Fetched {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Dropped last bar (unfinished candle): {dropped_date.date()}")

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_file)
        print(f"  Cached to {cache_file}")

    return df


# ---------------------------------------------------------------------------
# Multi-exchange crypto fetcher via ccxt
# ---------------------------------------------------------------------------

# TradingView timeframe notation → ccxt timeframe
_TV_TF_MAP = {
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "360": "6h",
    "480": "8h", "720": "12h",
    "D": "1d", "1D": "1d",
    "W": "1w", "1W": "1w",
    "M": "1M", "1M": "1M",
}

# Reverse: ccxt timeframe → milliseconds per candle (for pagination)
_TF_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000,
    "4h": 14_400_000, "6h": 21_600_000, "8h": 28_800_000,
    "12h": 43_200_000, "1d": 86_400_000, "1w": 604_800_000,
    "1M": 2_592_000_000,  # ~30 days, approximate
}

# Exchange fallback priority (most liquid first)
_EXCHANGE_FALLBACK = [
    "binance", "coinbase", "kraken", "bybit",
    "kucoin", "okx", "gate", "bitget", "mexc",
]

# Common quote currencies to try when resolving a base-only symbol
_QUOTE_FALLBACK = ["USDT", "USD", "BUSD", "USDC"]


def _normalize_tf(timeframe: str) -> str:
    """Convert TV-style timeframe to ccxt format.

    Accepts TV notation (e.g. '240', '1D', 'W') or ccxt notation
    (e.g. '4h', '1d', '1w'). Returns ccxt format.
    """
    if timeframe in _TV_TF_MAP:
        return _TV_TF_MAP[timeframe]
    # Already ccxt format?
    if timeframe in _TF_MS:
        return timeframe
    raise ValueError(
        f"Unknown timeframe '{timeframe}'. "
        f"TV notation: {list(_TV_TF_MAP.keys())} | "
        f"ccxt notation: {list(_TF_MS.keys())}"
    )


def _parse_symbol(symbol: str) -> tuple[str | None, str, str | None]:
    """Parse a flexible symbol input into (exchange, base, quote).

    Supports:
      - "BINANCE:BTCUSDT"  → ("binance", "BTC", "USDT")
      - "BTC/USDT"         → (None, "BTC", "USDT")
      - "BTCUSDT"          → (None, "BTC", "USDT")
      - "BTCUSD"           → (None, "BTC", "USD")
      - "BTC"              → (None, "BTC", None)  # quote resolved later
      - "SOL"              → (None, "SOL", None)

    Returns (exchange_id or None, base, quote or None).
    """
    exchange_id = None

    # Strip "EXCHANGE:" prefix
    if ":" in symbol:
        exch_part, symbol = symbol.split(":", 1)
        exchange_id = exch_part.strip().lower()
        # Map TV exchange names to ccxt IDs
        exch_map = {
            "binance": "binance", "coinbase": "coinbase",
            "kraken": "kraken", "bybit": "bybit",
            "kucoin": "kucoin", "okx": "okx",
            "gate": "gate", "bitget": "bitget", "mexc": "mexc",
        }
        exchange_id = exch_map.get(exchange_id, exchange_id)

    symbol = symbol.strip().upper()

    # "BTC/USDT" format
    if "/" in symbol:
        base, quote = symbol.split("/", 1)
        return exchange_id, base.strip(), quote.strip()

    # Try to split concatenated pairs: "BTCUSDT", "BTCUSD", "SOLUSD"
    for q in _QUOTE_FALLBACK:
        if symbol.endswith(q) and len(symbol) > len(q):
            base = symbol[: -len(q)]
            return exchange_id, base, q

    # Just a base symbol: "BTC", "SOL", "ETH"
    return exchange_id, symbol, None


def _resolve_ccxt_symbol(exchange, base: str, quote: str | None) -> str | None:
    """Find a valid ccxt symbol for base/quote on the given exchange.

    Tries quote currencies in _QUOTE_FALLBACK order if quote is None.
    Returns the ccxt symbol string (e.g. 'BTC/USDT') or None.
    """
    markets = exchange.markets
    if not markets:
        exchange.load_markets()
        markets = exchange.markets

    quotes_to_try = [quote] if quote else _QUOTE_FALLBACK
    for q in quotes_to_try:
        sym = f"{base}/{q}"
        if sym in markets:
            return sym
    return None


def _discover_via_coingecko(base: str) -> str | None:
    """Use CoinGecko free API to find the CoinGecko ID for a ticker symbol.

    Returns the coin ID (e.g. 'bitcoin') or None if not found.
    Used for logging/diagnostics; the actual trading pair resolution
    is done via ccxt load_markets().
    """
    try:
        url = "https://api.coingecko.com/api/v3/coins/list"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        coins = resp.json()
        base_lower = base.lower()
        for coin in coins:
            if coin.get("symbol", "").lower() == base_lower:
                return coin["id"]
    except Exception:
        pass
    return None


def fetch_crypto(
    symbol: str,
    timeframe: str = "1D",
    start: str = "2017-01-01",
    end: str = "2069-12-31",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch crypto OHLCV data from exchanges via ccxt.

    Uses a fallback chain of exchanges (binance → coinbase → kraken →
    bybit → kucoin → okx → gate → bitget → mexc) and returns the first
    one that has ≥50 candles for the requested pair and timeframe.

    The last bar is dropped (unfinished candle).

    Args:
        symbol: Flexible symbol input. Accepted formats:
            - ``"BTC"`` — auto-resolves quote currency (USDT→USD→BUSD→USDC)
            - ``"BTCUSDT"`` or ``"BTC/USDT"`` — explicit pair
            - ``"BINANCE:BTCUSDT"`` — specific exchange + pair
        timeframe: Bar period. Accepts TV notation (``"240"``, ``"1D"``,
            ``"W"``) or ccxt notation (``"4h"``, ``"1d"``, ``"1w"``).
        start: Start date for data (ISO format, e.g. ``"2017-01-01"``).
            Include warmup period for moving averages.
        end: End date for data (ISO format). Defaults to far future.
        use_cache: Cache fetched data as CSV in ``data/cache/``.

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
        Index: DatetimeIndex (timezone-naive, named 'Date')

    Raises:
        ImportError: If ccxt is not installed.
        ValueError: If no exchange has data for the requested pair.

    Examples:
        >>> df = fetch_crypto("BTC", "1D", start="2020-01-01")
        >>> df = fetch_crypto("SOL", "4h", start="2023-01-01")
        >>> df = fetch_crypto("BINANCE:ETHUSDT", "240", start="2021-01-01")
    """
    try:
        import ccxt
    except ImportError:
        raise ImportError(
            "ccxt is required for fetch_crypto(). "
            "Install it with: pip install ccxt"
        )

    tf = _normalize_tf(timeframe)
    tf_ms = _TF_MS[tf]
    exch_hint, base, quote = _parse_symbol(symbol)

    start_ts = int(pd.Timestamp(start).timestamp() * 1000)  # ms
    end_ts = int(pd.Timestamp(end).timestamp() * 1000)
    # Clamp end to now (can't fetch future candles)
    now_ms = int(time.time() * 1000)
    end_ts = min(end_ts, now_ms)

    # Build cache filename
    quote_label = quote or "AUTO"
    tf_label = timeframe.upper() if timeframe in _TV_TF_MAP else tf
    cache_name = f"ccxt_{base}{quote_label}_{tf_label}_{start}_{end}.csv"
    cache_file = CACHE_DIR / cache_name

    if use_cache and cache_file.exists():
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        print(f"Loaded cached data: {len(df)} bars from "
              f"{df.index[0]} to {df.index[-1]}")
        return df

    # Determine exchange order
    if exch_hint:
        exchanges_to_try = [exch_hint]
    else:
        exchanges_to_try = list(_EXCHANGE_FALLBACK)

    resolved_exchange = None
    resolved_symbol = None
    all_candles = []

    for exch_id in exchanges_to_try:
        try:
            exchange_class = getattr(ccxt, exch_id, None)
            if exchange_class is None:
                continue
            exchange = exchange_class({"enableRateLimit": True})
            exchange.load_markets()
        except Exception as e:
            print(f"  ⚠ {exch_id}: failed to load markets — {e}")
            continue

        ccxt_sym = _resolve_ccxt_symbol(exchange, base, quote)
        if ccxt_sym is None:
            print(f"  ⚠ {exch_id}: {base}/{'?' if not quote else quote} not listed")
            continue

        print(f"  Trying {exch_id} → {ccxt_sym} ({tf})...")

        # Paginated fetch
        candles = []
        cursor = start_ts
        page_limit = 1000  # most exchanges support 1000
        retries = 0

        while cursor < end_ts:
            try:
                chunk = exchange.fetch_ohlcv(
                    ccxt_sym, tf, since=cursor, limit=page_limit
                )
            except Exception as e:
                retries += 1
                if retries > 3:
                    print(f"  ⚠ {exch_id}: too many errors — {e}")
                    break
                time.sleep(1)
                continue

            if not chunk:
                break

            candles.extend(chunk)
            last_ts = chunk[-1][0]
            if last_ts <= cursor:
                break  # no progress
            cursor = last_ts + tf_ms

            if len(candles) % 5000 == 0:
                dt = pd.Timestamp(last_ts, unit="ms")
                print(f"    {len(candles)} candles so far (up to {dt})...")

            time.sleep(exchange.rateLimit / 1000)  # respect rate limit

        if len(candles) >= 50:
            all_candles = candles
            resolved_exchange = exch_id
            resolved_symbol = ccxt_sym
            break
        else:
            print(f"  ⚠ {exch_id}: only {len(candles)} candles — skipping")

    if not all_candles:
        # Try CoinGecko discovery for a better error message
        cg_id = _discover_via_coingecko(base)
        hint = f" (CoinGecko knows this as '{cg_id}')" if cg_id else ""
        raise ValueError(
            f"No exchange returned ≥50 candles for {base}/{quote or 'USDT'} "
            f"on timeframe {tf}{hint}.\n"
            f"Tried: {exchanges_to_try}\n"
            f"Suggestions:\n"
            f"  1. Check the symbol spelling\n"
            f"  2. Try a different timeframe (some pairs have limited history)\n"
            f"  3. Export data from TradingView instead"
        )

    # Build DataFrame
    df = pd.DataFrame(
        all_candles,
        columns=["timestamp", "Open", "High", "Low", "Close", "Volume"],
    )
    df["Date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("Date")
    df = df.drop(columns=["timestamp"])

    # Clean up
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    # Filter to requested date range
    df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    # Drop zero-volume bars (exchange artifacts)
    if (df["Volume"] == 0).sum() < len(df) * 0.5:
        # Only filter if most bars have volume — some indices don't report volume
        df = df[df["Volume"] > 0]

    if len(df) == 0:
        raise ValueError(
            f"No valid candles after filtering for {resolved_symbol} "
            f"on {resolved_exchange} ({tf}), date range {start} to {end}."
        )

    # Drop the last bar — it is an unfinished candle
    dropped_date = df.index[-1]
    df = df.iloc[:-1]

    print(f"Fetched {len(df)} bars of {resolved_symbol} ({tf}) from "
          f"{resolved_exchange}")
    print(f"  Range: {df.index[0]} to {df.index[-1]}")
    print(f"  Dropped last bar (unfinished candle): {dropped_date}")

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_file)
        print(f"  Cached to {cache_file}")

    return df


# ---------------------------------------------------------------------------
# Data-coverage check
# ---------------------------------------------------------------------------

def ensure_data_coverage(
    df: pd.DataFrame,
    xlsx_path: str | Path,
    timeframe: str,
    symbol: str | None = None,
    warmup_bars: int = 500,
) -> pd.DataFrame:
    """Check that *df* covers the XLSX trading range (+ warmup) and fetch if not.

    Indicators need history before the strategy's start date to produce
    correct values.  This function ensures the data extends far enough
    back (``pine_start - warmup_bars``) and far enough forward
    (``range_end``) to match the XLSX.

    1. Reads ``range_start`` and ``range_end`` from the XLSX via
       :func:`read_tv_xlsx_dates`.
    2. Computes the required window: ``range_start - warmup`` to ``range_end``.
    3. If *df* already covers that window → returns *df* unchanged.
    4. If not, and *symbol* is provided, calls :func:`fetch_crypto` with
       ``use_cache=False`` to fetch fresh data.
    5. If the fresh data still doesn't cover the window → raises
       ``ValueError`` with a clear message.

    Parameters
    ----------
    df : DataFrame
        Current OHLCV data (DatetimeIndex).
    xlsx_path : str or Path
        TradingView ``.xlsx`` export to compare against.
    timeframe : str
        Timeframe in TV or ccxt notation (e.g. ``"1D"``, ``"4h"``).
        Used to convert *warmup_bars* to a calendar offset and for
        fetching if needed.
    symbol : str, optional
        Symbol for :func:`fetch_crypto` (e.g. ``"BTC"``, ``"BTCUSDT"``).
        Required if auto-fetching is desired.
    warmup_bars : int
        Number of bars before ``pine_start`` needed for indicator warmup.
        Defaults to 500 — generous enough for most indicators (200-EMA,
        Ichimoku 52, Gaussian 25×4 poles, etc.).

    Returns
    -------
    DataFrame
        Either *df* unchanged (if coverage OK) or a freshly fetched DataFrame.

    Raises
    ------
    ValueError
        If coverage is insufficient and cannot be resolved by fetching.
    """
    dates = read_tv_xlsx_dates(xlsx_path)
    required_end = pd.Timestamp(dates["range_end"])

    tf = _normalize_tf(timeframe)
    bar_ms = _TF_MS[tf]
    warmup_td = pd.Timedelta(milliseconds=warmup_bars * bar_ms)

    # Use range_start (actual first trade) for warmup, not pine_start.
    # Coins may not have data back to the Pine's startDate (e.g. SUI
    # only exists since 2023, but Pine startDate may be 2018).
    range_start = pd.Timestamp(dates["range_start"])
    required_start = range_start - warmup_td

    def _check(data: pd.DataFrame) -> tuple[bool, bool]:
        """Return (start_ok, end_ok)."""
        if data.empty:
            return False, False
        return data.index[0] <= required_start, data.index[-1] >= required_end

    start_ok, end_ok = _check(df)
    if start_ok and end_ok:
        return df

    # --- Attempt fresh fetch ---------------------------------------------------
    if symbol is None:
        _raise_coverage_error(df, required_start, required_end, start_ok, end_ok)

    print(f"\nData coverage insufficient for XLSX range.")
    print(f"  Need:  {_fmt_date(required_start)} to {_fmt_date(required_end)}")
    if df.empty:
        print(f"  Have:  (empty DataFrame)")
    else:
        print(f"  Have:  {_fmt_date(df.index[0])} to {_fmt_date(df.index[-1])}")
    print(f"  Fetching fresh data via fetch_crypto({symbol!r}, {timeframe!r}) ...")

    # Pad end by 2 bars — fetch_crypto drops the last bar (unfinished candle)
    fetch_end = required_end + pd.Timedelta(milliseconds=2 * bar_ms)

    fresh_df = fetch_crypto(
        symbol, timeframe,
        start=_fmt_date(required_start),
        end=_fmt_date(fetch_end),
        use_cache=False,
    )

    start_ok, end_ok = _check(fresh_df)
    if start_ok and end_ok:
        print(f"  Fresh data covers required range.")
        return fresh_df

    # Soft-fail on start: if end is covered but the asset simply doesn't
    # have data going back far enough (e.g. recently listed coin), accept
    # with a warning.  Indicators will produce NaN for early bars, but the
    # strategy handles that by not trading until they warm up.
    if end_ok and not start_ok:
        avail = len(fresh_df.loc[:range_start])
        print(f"  ⚠  Asset history starts at {_fmt_date(fresh_df.index[0])}.")
        print(f"     Wanted {warmup_bars} warmup bars, have {avail}.")
        print(f"     Proceeding — indicators will warm up with available data.")
        return fresh_df

    _raise_coverage_error(fresh_df, required_start, required_end, start_ok, end_ok)


def _raise_coverage_error(
    df: pd.DataFrame,
    required_start: pd.Timestamp,
    required_end: pd.Timestamp,
    start_ok: bool,
    end_ok: bool,
) -> None:
    """Raise a clear ValueError describing the coverage gap."""
    if df.empty:
        raise ValueError(
            "Data does not cover the XLSX trading range.\n"
            f"  DataFrame is empty — no data available.\n"
            f"  Need: {_fmt_date(required_start)} to {_fmt_date(required_end)}"
        )
    parts = []
    if not start_ok:
        gap = df.index[0] - required_start
        parts.append(
            f"Data starts at {_fmt_date(df.index[0])} but need "
            f"{_fmt_date(required_start)} ({gap.days} days short)"
        )
    if not end_ok:
        gap = required_end - df.index[-1]
        parts.append(
            f"Data ends at {_fmt_date(df.index[-1])} but need "
            f"{_fmt_date(required_end)} ({gap.days} days short)"
        )
    raise ValueError(
        "Data does not cover the XLSX trading range.\n  " + "\n  ".join(parts)
    )
