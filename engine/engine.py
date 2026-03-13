"""
Backtesting engine that matches TradingView's PineScript strategy behavior.

Replicates these TradingView settings:
- calc_on_every_tick = false   → signals on bar close, orders fill on next bar open
- fill_orders_on_standard_ohlc = true  → fills at the Open price
- process_orders_on_close = false/true  → fill at bar Close instead of next bar Open
- margin_long = 0, margin_short = 0    → no margin calls
- commission_type = percent    → applied on both entry and exit

Usage:
    from engine import BacktestConfig, run_backtest
    from engine import calc_ema, detect_crossover, detect_crossunder, ema_cross_signals

    df = load_your_data()                        # DataFrame with Open, High, Low, Close
    df = ema_cross_signals(df, fast_len=9, slow_len=21)  # adds long_entry / long_exit columns
    kpis = run_backtest(df, BacktestConfig())    # run the backtest
    print_kpis(kpis)
"""

__version__ = "21.0.0"

import math
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """Backtest settings matching TradingView's strategy() properties."""
    initial_capital: float = 1000.0
    commission_pct: float = 0.1       # e.g. 0.1 = 0.1%
    slippage_ticks: int = 0
    qty_type: str = "percent_of_equity"
    qty_value: float = 100.0          # 100 = 100% of equity
    pyramiding: int = 1
    start_date: str = "2018-01-01"
    end_date: str = "2069-12-31"
    take_profit_pct: float = 0.0  # 0.0 = disabled; e.g. 5.0 = exit at +5% from entry
    stop_loss_pct: float = 0.0    # 0.0 = disabled; e.g. 3.0 = exit at -3% from entry
    process_orders_on_close: bool = False  # True = fill at bar Close (not next bar Open)
    show_progress: bool = False   # True = show tqdm progress bar


@dataclass
class Trade:
    """A single completed (or open) trade."""
    entry_date: pd.Timestamp
    entry_price: float
    entry_qty: float                  # asset units (e.g. BTC, fractional)
    direction: str = "long"           # "long" or "short"
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    entry_commission: float = 0.0
    exit_commission: float = 0.0


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------

def calc_ema(series: pd.Series, length: int) -> pd.Series:
    """
    EMA matching TradingView's ``ta.ema()``.

    - Multiplier: ``2 / (length + 1)``
    - Seed: SMA of the first *length* **valid** values

    Handles NaN-leading input (e.g. when chaining indicators).
    """
    multiplier = 2.0 / (length + 1)
    ema = pd.Series(np.nan, index=series.index, dtype=float)
    vals = series.values

    # Find the first index with `length` consecutive non-NaN values for the seed
    valid = ~np.isnan(vals)
    start = -1
    count = 0
    for i in range(len(vals)):
        if valid[i]:
            count += 1
            if count == length:
                start = i - length + 1
                break
        else:
            count = 0

    if start < 0:
        return ema  # not enough data

    seed_idx = start + length - 1
    ema.iloc[seed_idx] = np.mean(vals[start:start + length])
    for i in range(seed_idx + 1, len(vals)):
        if np.isnan(vals[i]):
            ema.iloc[i] = ema.iloc[i - 1]  # carry forward (matches Pine behavior)
            continue
        ema.iloc[i] = vals[i] * multiplier + ema.iloc[i - 1] * (1 - multiplier)

    return ema


def detect_crossover(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """True on bars where *fast* crosses **above** *slow*."""
    return ((fast.shift(1) <= slow.shift(1)) & (fast > slow)).fillna(False)


def detect_crossunder(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """True on bars where *fast* crosses **below** *slow*."""
    return ((fast.shift(1) >= slow.shift(1)) & (fast < slow)).fillna(False)


def calc_smma(series: pd.Series, length: int) -> pd.Series:
    """
    Smoothed Moving Average matching TradingView's ``ta.rma()``.

    Also known as RMA / Wilder's smoothing.
    Formula: ``smma[i] = (smma[i-1] * (length - 1) + src[i]) / length``
    Seed: SMA of the first *length* **valid** values.
    """
    smma = pd.Series(np.nan, index=series.index)
    vals = series.values

    # Find seed from first `length` consecutive non-NaN values
    valid = ~np.isnan(vals)
    start = -1
    count = 0
    for i in range(len(vals)):
        if valid[i]:
            count += 1
            if count == length:
                start = i - length + 1
                break
        else:
            count = 0

    if start < 0:
        return smma

    seed_idx = start + length - 1
    smma.iloc[seed_idx] = np.mean(vals[start:start + length])
    for i in range(seed_idx + 1, len(vals)):
        if np.isnan(vals[i]):
            smma.iloc[i] = smma.iloc[i - 1]  # carry forward (matches Pine behavior)
            continue
        smma.iloc[i] = (smma.iloc[i - 1] * (length - 1) + vals[i]) / length

    return smma


def calc_wma(series: pd.Series, length: int) -> pd.Series:
    """
    Weighted Moving Average matching TradingView's ``ta.wma()``.

    Weights increase linearly: ``[1, 2, 3, ..., length]``.
    """
    weights = np.arange(1, length + 1, dtype=float)
    weight_sum = weights.sum()

    def _weighted_avg(window):
        return np.dot(window, weights) / weight_sum

    return series.rolling(window=length, min_periods=length).apply(_weighted_avg, raw=True)


def calc_hma(series: pd.Series, length: int) -> pd.Series:
    """
    Hull Moving Average (HMA) by Alan Hull.

    ``HMA = WMA( 2·WMA(n/2) − WMA(n) , √n )``
    """
    half = length // 2
    sqrt = round(math.sqrt(length))
    diff = 2 * calc_wma(series, half) - calc_wma(series, length)
    return calc_wma(diff, sqrt)


def calc_ehma(series: pd.Series, length: int) -> pd.Series:
    """
    Exponential Hull Moving Average (EHMA).

    Same structure as HMA but uses EMA instead of WMA.
    """
    half = length // 2
    sqrt = round(math.sqrt(length))
    diff = 2 * calc_ema(series, half) - calc_ema(series, length)
    return calc_ema(diff, sqrt)


def calc_thma(series: pd.Series, length: int) -> pd.Series:
    """
    Triple Hull Moving Average (THMA).

    ``THMA = WMA( 3·WMA(n/3) − WMA(n/2) − WMA(n) , n )``
    """
    len3 = length // 3
    half = length // 2
    inner = 3 * calc_wma(series, len3) - calc_wma(series, half) - calc_wma(series, length)
    return calc_wma(inner, length)


def calc_gaussian(series: pd.Series, length: int, poles: int = 1) -> pd.Series:
    """
    Gaussian filter approximated by cascading EMAs.

    *poles* (1–4) controls smoothness: more poles → smoother curve.
    """
    poles = max(1, min(poles, 4))
    result = calc_ema(series, length)
    for _ in range(poles - 1):
        result = calc_ema(result, length)
    return result


def calc_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    Average True Range matching TradingView's ``ta.atr()``.

    ``ATR = ta.rma(ta.tr(), length)``

    Requires ``High``, ``Low``, ``Close`` columns in *df*.
    """
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"] - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return calc_smma(tr, length)


def calc_sma(series: pd.Series, length: int) -> pd.Series:
    """
    Simple Moving Average matching TradingView's ``ta.sma()``.
    """
    return series.rolling(window=length, min_periods=length).mean()


def calc_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """
    RSI matching TradingView's ``ta.rsi()``.

    Uses ``ta.rma()`` (SMMA / Wilder's smoothing) for the gain and loss
    averages — NOT a simple rolling mean.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = calc_smma(gain, length)
    avg_loss = calc_smma(loss, length)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(
    series: pd.Series,
    fast_len: int = 12,
    slow_len: int = 26,
    signal_len: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD matching TradingView's ``ta.macd()``.

    Returns
    -------
    (macd_line, signal_line, histogram)
        - macd_line   = EMA(fast) - EMA(slow)
        - signal_line = EMA(macd_line, signal_len)
        - histogram   = macd_line - signal_line
    """
    fast_ema = calc_ema(series, fast_len)
    slow_ema = calc_ema(series, slow_len)
    macd_line = fast_ema - slow_ema
    signal_line = calc_ema(macd_line, signal_len)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ---------------------------------------------------------------------------
# Highest / Lowest / Donchian / OBV / Ichimoku
# ---------------------------------------------------------------------------

def calc_highest(series: pd.Series, length: int) -> pd.Series:
    """
    Highest value over *length* bars, matching TradingView's ``ta.highest()``.
    """
    return series.rolling(window=length, min_periods=length).max()


def calc_lowest(series: pd.Series, length: int) -> pd.Series:
    """
    Lowest value over *length* bars, matching TradingView's ``ta.lowest()``.
    """
    return series.rolling(window=length, min_periods=length).min()


def calc_donchian(
    high: pd.Series,
    low: pd.Series,
    length: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Donchian channel matching PineScript's ``ta.highest/ta.lowest`` pair.

    Returns
    -------
    (upper, lower, mid)
        - upper = ta.highest(high, length)
        - lower = ta.lowest(low, length)
        - mid   = (upper + lower) / 2   (same as Ichimoku's ``donchian()`` helper)
    """
    upper = calc_highest(high, length)
    lower = calc_lowest(low, length)
    mid = (upper + lower) / 2.0
    return upper, lower, mid


def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume matching TradingView's ``ta.obv``.

    OBV accumulates volume: +volume when close > prev close,
    −volume when close < prev close, unchanged otherwise.
    """
    close_v = close.values
    vol_v = volume.values
    n = len(close_v)
    obv = np.zeros(n)
    for i in range(1, n):
        if close_v[i] > close_v[i - 1]:
            obv[i] = obv[i - 1] + vol_v[i]
        elif close_v[i] < close_v[i - 1]:
            obv[i] = obv[i - 1] - vol_v[i]
        else:
            obv[i] = obv[i - 1]
    return pd.Series(obv, index=close.index)


def calc_ichimoku(
    high: pd.Series,
    low: pd.Series,
    conversion_periods: int = 9,
    base_periods: int = 26,
    lagging_span2_periods: int = 52,
    displacement: int = 26,
) -> dict[str, pd.Series]:
    """
    Ichimoku Cloud components matching TradingView's built-in Ichimoku.

    Returns a dict with keys:
        ``conversion``, ``base``, ``lead_a``, ``lead_b``,
        ``displaced_lead_a``, ``displaced_lead_b``

    The displaced variants are shifted *forward* by *displacement* bars
    (``series.shift(displacement)``), matching PineScript's
    ``leadLine1[displacement]`` which reads the value from N bars ago.
    """
    _, _, conversion = calc_donchian(high, low, conversion_periods)
    _, _, base = calc_donchian(high, low, base_periods)
    lead_a = (conversion + base) / 2.0
    _, _, lead_b = calc_donchian(high, low, lagging_span2_periods)

    return {
        "conversion": conversion,
        "base": base,
        "lead_a": lead_a,
        "lead_b": lead_b,
        "displaced_lead_a": lead_a.shift(displacement),
        "displaced_lead_b": lead_b.shift(displacement),
    }


# ---------------------------------------------------------------------------
# Source selector (matches PineScript input.source())
# ---------------------------------------------------------------------------

def get_source(df: pd.DataFrame, source: str = "Close") -> pd.Series:
    """
    Return a price series from *df* matching a PineScript source string.

    Accepted values (case-insensitive):
        ``close``, ``open``, ``high``, ``low``,
        ``hl2``, ``hlc3``, ``ohlc4``
    """
    key = source.strip().lower()
    if key == "close":
        return df["Close"]
    if key == "open":
        return df["Open"]
    if key == "high":
        return df["High"]
    if key == "low":
        return df["Low"]
    if key == "hl2":
        return (df["High"] + df["Low"]) / 2
    if key == "hlc3":
        return (df["High"] + df["Low"] + df["Close"]) / 3
    if key == "ohlc4":
        return (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    raise ValueError(
        f"Unknown source '{source}'. "
        "Use: close, open, high, low, hl2, hlc3, ohlc4"
    )


# ---------------------------------------------------------------------------
# Strategy signal generators (add more here for new strategies)
# ---------------------------------------------------------------------------

def ema_cross_signals(df: pd.DataFrame, fast_len: int = 9, slow_len: int = 21) -> pd.DataFrame:
    """
    Add EMA-crossover entry/exit signals to *df* (in-place + returned).

    Columns added:
        fast_ema, slow_ema, long_entry, long_exit
    """
    df = df.copy()
    df["fast_ema"] = calc_ema(df["Close"], fast_len)
    df["slow_ema"] = calc_ema(df["Close"], slow_len)
    df["long_entry"] = detect_crossover(df["fast_ema"], df["slow_ema"])
    df["long_exit"] = detect_crossunder(df["fast_ema"], df["slow_ema"])
    return df


# ---------------------------------------------------------------------------
# TP/SL helper
# ---------------------------------------------------------------------------

def _check_tpsl_fill(
    bar_open: float,
    bar_high: float,
    bar_low: float,
    entry_price: float,
    position_side: str,
    tp_pct: float,
    sl_pct: float,
    tp_price: float = 0.0,
    sl_price: float = 0.0,
    tp_offset: float = 0.0,
    sl_offset: float = 0.0,
) -> tuple:
    """
    Check if a Take-Profit or Stop-Loss level is hit on this bar.

    Matches TradingView behaviour *without* Bar Magnifier:
      1. Gap-through: bar Open already past TP or SL → fill at Open.
      2. Both TP and SL hit intrabar: use TV heuristic — Open closer to
         the favourable extreme → that side hit first.
      3. Single level hit: fill at exact TP or SL price.
      4. No hit: return (None, "").

    Price levels are determined by priority (first match wins):
      1. Absolute price columns (``tp_price`` / ``sl_price``) — if > 0
      2. Offset from entry (``tp_offset`` / ``sl_offset``) — if > 0,
         added/subtracted from entry_price based on position side.
         Matches PineScript ``strategy.exit(limit=entry ± offset, ...)``.
      3. Percentage from entry (``tp_pct`` / ``sl_pct``) — if > 0
      4. Disabled — no TP/SL check

    Returns
    -------
    (fill_price, fill_type)
        fill_type is ``"tp"``, ``"sl"``, or ``""`` (no fill).
    """
    # --- price levels (absolute > offset > percentage) ---
    tp_level = None
    sl_level = None

    if tp_price > 0:
        tp_level = tp_price
    elif tp_offset > 0:
        if position_side == "long":
            tp_level = entry_price + tp_offset
        else:
            tp_level = entry_price - tp_offset
    elif tp_pct > 0:
        if position_side == "long":
            tp_level = entry_price * (1 + tp_pct / 100)
        else:
            tp_level = entry_price * (1 - tp_pct / 100)

    if sl_price > 0:
        sl_level = sl_price
    elif sl_offset > 0:
        if position_side == "long":
            sl_level = entry_price - sl_offset
        else:
            sl_level = entry_price + sl_offset
    elif sl_pct > 0:
        if position_side == "long":
            sl_level = entry_price * (1 - sl_pct / 100)
        else:
            sl_level = entry_price * (1 + sl_pct / 100)

    # --- 1) gap-through at Open ---
    if position_side == "long":
        if tp_level is not None and bar_open >= tp_level:
            return (bar_open, "tp")
        if sl_level is not None and bar_open <= sl_level:
            return (bar_open, "sl")
    else:
        if tp_level is not None and bar_open <= tp_level:
            return (bar_open, "tp")
        if sl_level is not None and bar_open >= sl_level:
            return (bar_open, "sl")

    # --- 2) intrabar hit detection ---
    if position_side == "long":
        tp_hit = tp_level is not None and bar_high >= tp_level
        sl_hit = sl_level is not None and bar_low <= sl_level
    else:
        tp_hit = tp_level is not None and bar_low <= tp_level
        sl_hit = sl_level is not None and bar_high >= sl_level

    if tp_hit and sl_hit:
        # Both hit — TV heuristic: Open closer to favourable extreme → hit first
        if position_side == "long":
            # Open closer to High → went up first → TP
            if (bar_high - bar_open) <= (bar_open - bar_low):
                return (tp_level, "tp")
            else:
                return (sl_level, "sl")
        else:
            # Open closer to Low → went down first → TP (short profits from drop)
            if (bar_open - bar_low) <= (bar_high - bar_open):
                return (tp_level, "tp")
            else:
                return (sl_level, "sl")

    if tp_hit:
        return (tp_level, "tp")
    if sl_hit:
        return (sl_level, "sl")

    return (None, "")


# ---------------------------------------------------------------------------
# Core backtest engine
# ---------------------------------------------------------------------------

def run_backtest(df: pd.DataFrame, config: BacktestConfig) -> dict:
    """
    Run a long-only backtest matching TradingView behaviour.

    **Required columns** in *df*:
        ``Open``, ``High``, ``Low``, ``Close``,
        ``long_entry`` (bool), ``long_exit`` (bool)

    **Optional columns**:
        ``entry_qty`` (float) — per-bar entry size in asset units.
            When present, the engine uses this qty instead of computing from
            equity.  Matches PineScript's ``strategy.entry(qty=...)`` pattern.
            The value on the signal bar (where ``long_entry=True``) is used
            for the fill on the next bar's Open.

    **Position sizing** (``config.qty_type`` / ``config.qty_value``):
        - ``"percent_of_equity"`` (default, 100.0) — invest N% of equity.
          At 100% = full equity, commission-adjusted at fill time.
          At <100% = partial equity, qty computed at signal time using bar Close.
        - ``"cash"`` — invest a fixed dollar amount per trade.
          ``qty = cash_value / close`` computed at signal time, fills at Open.
        - ``"fixed"`` — invest a fixed number of asset units per trade.
          ``qty = config.qty_value`` (e.g. 0.1 = 0.1 BTC). Matches
          TradingView's ``strategy.fixed`` qty type.

    **Pyramiding** (``config.pyramiding > 1``):
        Allows multiple simultaneous long sub-positions.  Each sub-position
        is tracked as a separate ``Trade``.  ``long_exit=True`` triggers
        ``strategy.close_all()`` — closing every open sub-position at the
        next bar's Open.

    The DataFrame should include warmup bars **before** ``config.start_date``
    so that indicator values are accurate when the trading window begins.

    Returns a dict of KPIs (see ``compute_kpis``) including a ``trades`` list.
    """
    # --- input validation ---------------------------------------------------
    required = {"Open", "High", "Low", "Close", "long_entry", "long_exit"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {missing}. "
            "Add signal columns (e.g. via ema_cross_signals()) before calling run_backtest()."
        )

    df = df.copy()
    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.end_date)

    # --- start-date safeguard ------------------------------------------------
    # If the data doesn't go back far enough, adjust start_date to the first
    # available bar.  This prevents mismatched KPIs when the user's CSV export
    # begins later than expected (TV always trades from the first bar).
    data_first = df.index[0]
    if data_first > start:
        print(f"\n  (!)  DATA STARTS AFTER start_date!")
        print(f"     start_date was:  {config.start_date}")
        print(f"     data starts at:  {data_first.date()}")
        print(f"     -> Adjusted start_date to {data_first.date()}")
        print(f"     To match TradingView, set the same start date in your")
        print(f"     TV strategy properties (Date Range -> Start Date).\n")
        start = data_first

    # --- state ---------------------------------------------------------------
    equity = config.initial_capital
    cash = config.initial_capital
    open_positions: list[Trade] = []   # sub-positions (pyramiding support)
    position_qty = 0.0                 # aggregate qty across all open positions
    trades: list[Trade] = []

    pending_entry = False
    pending_entry_qty = 0.0            # qty from entry_qty column or qty_type sizing
    pending_exit = False
    entry_bar_idx = -1  # bar index where position was entered; TP/SL skipped on this bar

    equity_curve: list[dict] = []
    commission_rate = config.commission_pct / 100.0

    # Optional per-bar entry quantity column (for variable-size / pyramiding strategies).
    # When present, the engine uses this qty instead of computing from equity.
    has_entry_qty = "entry_qty" in df.columns

    # TP/SL can come from config (percentages), DataFrame columns (absolute prices),
    # or DataFrame offset columns (distance from entry). Priority: price > offset > pct.
    has_tp_col = "tp_price" in df.columns
    has_sl_col = "sl_price" in df.columns
    has_tp_off = "tp_offset" in df.columns
    has_sl_off = "sl_offset" in df.columns
    tp_sl_active = (config.take_profit_pct > 0 or config.stop_loss_pct > 0
                    or has_tp_col or has_sl_col or has_tp_off or has_sl_off)

    # --- intrabar drawdown tracking (TV methodology) -------------------------
    # Peak equity only updates when flat (no open position).
    # TV defines "max_equity" as peak from initial capital + closed trades.
    # Intrabar trough uses bar["Low"] for open long positions.
    # Max DD ($) and max DD (%) are tracked independently — they may occur
    # at different points in time, matching TV's reporting.
    peak_equity = config.initial_capital
    max_intrabar_dd = 0.0       # worst absolute drawdown (negative or zero)
    max_intrabar_dd_pct = 0.0   # worst percentage drawdown (negative or zero)

    # --- bar-by-bar loop (matches TV execution order) ------------------------
    main_loop = range(len(df))
    if config.show_progress:
        main_loop = tqdm(main_loop, desc="Backtesting Bars")

    for i in main_loop:
        bar = df.iloc[i]
        bar_date = df.index[i]
        bar_in_range = start <= bar_date <= end

        # 1) FILL pending orders at this bar's Open
        #    Exit fills BEFORE entry fills (close_all then new position).

        if pending_exit and position_qty > 0:
            fill_price = bar["Open"]
            for pos in open_positions:
                trade_value = pos.entry_qty * fill_price
                exit_commission = trade_value * commission_rate
                gross_pnl = pos.entry_qty * (fill_price - pos.entry_price)
                net_pnl = gross_pnl - pos.entry_commission - exit_commission

                cash += trade_value - exit_commission

                pos.exit_date = bar_date
                pos.exit_price = fill_price
                pos.pnl = net_pnl
                entry_value = pos.entry_qty * pos.entry_price
                pos.pnl_pct = (net_pnl / entry_value) * 100
                pos.exit_commission = exit_commission
                trades.append(pos)

            equity = cash
            open_positions = []
            position_qty = 0.0
            pending_exit = False

        if pending_entry:
            fill_price = bar["Open"]
            if pending_entry_qty > 0:
                # Signal-time sized: entry_qty column, cash, or partial pct_equity
                qty = pending_entry_qty
                trade_value = qty * fill_price
            else:
                # Default 100% equity at fill time (backward compatible)
                trade_value = equity / (1 + commission_rate)
                qty = trade_value / fill_price
            entry_commission = trade_value * commission_rate

            position_qty += qty
            cash -= (trade_value + entry_commission)

            new_trade = Trade(
                entry_date=bar_date,
                entry_price=fill_price,
                entry_qty=qty,
                direction="long",
                entry_commission=entry_commission,
            )
            open_positions.append(new_trade)
            pending_entry = False
            entry_bar_idx = i

        # 1.5) TP/SL intrabar fill — fills at exact TP/SL price on this bar
        #      Uses first position's entry price for level check (correct for
        #      pyramiding=1; for pyramiding>1 the first position is representative).
        tpsl_filled = False
        if tp_sl_active and bar_in_range and position_qty > 0 and i > entry_bar_idx:
            tpsl_entry_price = open_positions[0].entry_price
            bar_tp = bar["tp_price"] if has_tp_col else 0.0
            bar_sl = bar["sl_price"] if has_sl_col else 0.0
            bar_tp_off = bar["tp_offset"] if has_tp_off else 0.0
            bar_sl_off = bar["sl_offset"] if has_sl_off else 0.0
            fill_price, fill_type = _check_tpsl_fill(
                bar_open=bar["Open"], bar_high=bar["High"], bar_low=bar["Low"],
                entry_price=tpsl_entry_price, position_side="long",
                tp_pct=config.take_profit_pct, sl_pct=config.stop_loss_pct,
                tp_price=bar_tp, sl_price=bar_sl,
                tp_offset=bar_tp_off, sl_offset=bar_sl_off,
            )
            if fill_price is not None:
                tpsl_filled = True

                # Drawdown while still holding (before exit settles)
                if fill_type == "sl":
                    worst_price = fill_price       # exited at SL
                elif fill_price == bar["Open"]:
                    worst_price = fill_price       # gap-through, exited at Open
                else:
                    worst_price = bar["Low"]       # exact TP: held through bar range
                equity_at_worst = cash + position_qty * worst_price
                dd = equity_at_worst - peak_equity
                dd_pct = (dd / peak_equity) * 100 if peak_equity != 0 else 0.0
                if dd < max_intrabar_dd:
                    max_intrabar_dd = dd
                if dd_pct < max_intrabar_dd_pct:
                    max_intrabar_dd_pct = dd_pct

                # Close ALL open positions at TP/SL price
                for pos in open_positions:
                    tv = pos.entry_qty * fill_price
                    ec = tv * commission_rate
                    gpnl = pos.entry_qty * (fill_price - pos.entry_price)
                    npnl = gpnl - pos.entry_commission - ec

                    cash += tv - ec

                    pos.exit_date = bar_date
                    pos.exit_price = fill_price
                    pos.pnl = npnl
                    ev = pos.entry_qty * pos.entry_price
                    pos.pnl_pct = (npnl / ev) * 100
                    pos.exit_commission = ec
                    trades.append(pos)

                equity = cash
                open_positions = []
                position_qty = 0.0

                # Update peak equity (now flat after TP/SL exit)
                if equity > peak_equity:
                    peak_equity = equity

        # 2a) Intrabar drawdown check (only while holding; skip if TP/SL handled it)
        if bar_in_range and position_qty > 0 and not tpsl_filled:
            equity_at_low = cash + position_qty * bar["Low"]
            dd = equity_at_low - peak_equity
            dd_pct = (dd / peak_equity) * 100 if peak_equity != 0 else 0.0
            if dd < max_intrabar_dd:
                max_intrabar_dd = dd
            if dd_pct < max_intrabar_dd_pct:
                max_intrabar_dd_pct = dd_pct

        # 2b) Mark-to-market equity at Close
        if position_qty > 0:
            equity = cash + position_qty * bar["Close"]
        else:
            equity = cash

        if bar_in_range:
            equity_curve.append({"date": bar_date, "equity": equity})

        # 2c) Update peak equity — ONLY when flat (no open position).
        # TV's "max_equity" = peak from initial capital + all closed trades.
        # Unrealised mark-to-market equity does NOT update the peak.
        if bar_in_range and position_qty == 0 and equity > peak_equity:
            peak_equity = equity

        # 3) Detect signals at Close (only inside trading window)
        pending_entry = False
        pending_exit = False
        pending_entry_qty = 0.0

        if bar_in_range:
            if bar["long_entry"] and len(open_positions) < config.pyramiding:
                pending_entry = True
                if has_entry_qty:
                    pending_entry_qty = bar["entry_qty"]
                elif config.qty_type == "cash":
                    # TV computes qty at signal time: qty = cash_value / close
                    pending_entry_qty = config.qty_value / bar["Close"]
                elif config.qty_type == "fixed":
                    # Fixed asset units (e.g. 0.1 BTC) — matches TV strategy.fixed
                    pending_entry_qty = config.qty_value
                elif config.qty_type == "percent_of_equity" and config.qty_value < 100.0:
                    # TV computes qty at signal time for partial equity sizing
                    target = equity * config.qty_value / 100.0
                    pending_entry_qty = target / ((1 + commission_rate) * bar["Close"])
            if bar["long_exit"] and position_qty > 0:
                pending_exit = True

        # 3.5) process_orders_on_close: fill at THIS bar's Close
        if config.process_orders_on_close:

            # --- Exit at Close ---
            if pending_exit and position_qty > 0:
                fill_price = bar["Close"]
                for pos in open_positions:
                    trade_value = pos.entry_qty * fill_price
                    exit_commission = trade_value * commission_rate
                    gross_pnl = pos.entry_qty * (fill_price - pos.entry_price)
                    net_pnl = gross_pnl - pos.entry_commission - exit_commission

                    cash += trade_value - exit_commission

                    pos.exit_date = bar_date
                    pos.exit_price = fill_price
                    pos.pnl = net_pnl
                    entry_value = pos.entry_qty * pos.entry_price
                    pos.pnl_pct = (net_pnl / entry_value) * 100
                    pos.exit_commission = exit_commission
                    trades.append(pos)

                equity = cash
                open_positions = []
                position_qty = 0.0
                pending_exit = False

                # Update peak equity (now flat) — for reversal on same bar
                if bar_in_range and equity > peak_equity:
                    peak_equity = equity

            # --- Entry at Close ---
            if pending_entry and len(open_positions) < config.pyramiding:
                fill_price = bar["Close"]
                if pending_entry_qty > 0:
                    qty = pending_entry_qty
                    trade_value = qty * fill_price
                else:
                    # 100% equity at fill time
                    trade_value = equity / (1 + commission_rate)
                    qty = trade_value / fill_price
                entry_commission = trade_value * commission_rate

                position_qty += qty
                cash -= (trade_value + entry_commission)

                new_trade = Trade(
                    entry_date=bar_date,
                    entry_price=fill_price,
                    entry_qty=qty,
                    direction="long",
                    entry_commission=entry_commission,
                )
                open_positions.append(new_trade)
                pending_entry = False
                entry_bar_idx = i

            # Re-mark equity after Close fills
            if position_qty > 0:
                equity = cash + position_qty * bar["Close"]
            else:
                equity = cash

            # Overwrite equity curve entry for this bar
            if bar_in_range and equity_curve and equity_curve[-1]["date"] == bar_date:
                equity_curve[-1]["equity"] = equity

            # Update peak equity if flat after Close fills
            if bar_in_range and position_qty == 0 and equity > peak_equity:
                peak_equity = equity

    # Record any open positions at end of data
    for pos in open_positions:
        trades.append(pos)

    equity_df = pd.DataFrame(equity_curve)
    kpis = compute_kpis(trades, equity_df, config,
                        max_intrabar_dd, max_intrabar_dd_pct)
    # Include the actual start/end dates used (may differ from config if adjusted)
    kpis["actual_start_date"] = str(start.date())
    kpis["actual_end_date"] = str(end.date())
    return kpis


def run_backtest_long_short(df: pd.DataFrame, config: BacktestConfig) -> dict:
    """
    Run a long+short backtest matching TradingView behaviour.

    Supports separate long and short positions (never simultaneously).
    Long and short signals are completely independent — a long_exit does
    NOT open a short, and vice versa.

    **Pyramiding** (``config.pyramiding > 1``):
        Allows multiple simultaneous sub-positions in the same direction.
        Each sub-position is tracked as a separate ``Trade``.  Exit signals
        (``long_exit`` / ``short_exit``) close ALL open sub-positions
        (``strategy.close_all()`` equivalent).  Reversals close all positions
        in the current direction before opening the new one.

    **Required columns** in *df*:
        ``Open``, ``High``, ``Low``, ``Close``,
        ``long_entry`` (bool), ``long_exit`` (bool),
        ``short_entry`` (bool), ``short_exit`` (bool)

    **Position sizing** (``config.qty_type`` / ``config.qty_value``):
        - ``"percent_of_equity"`` (default, 100.0) — invest N% of equity.
        - ``"cash"`` — invest a fixed dollar amount per trade.
        - ``"fixed"`` — invest a fixed number of asset units per trade.

    TradingView matching notes:
    - Short PnL: qty * (entry_price - exit_price) - commissions
    - Intrabar DD for shorts uses bar High (worst case for short = price spike up)
    - Signals on bar close, fill on next bar open (calc_on_every_tick=false)
    - margin_long = 0, margin_short = 0

    Returns a dict of KPIs (see ``compute_kpis``) including a ``trades`` list.
    """
    # --- input validation ---------------------------------------------------
    required = {"Open", "High", "Low", "Close",
                "long_entry", "long_exit", "short_entry", "short_exit"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {missing}. "
            "Add all signal columns before calling run_backtest_long_short()."
        )

    # --- process_orders_on_close signal conflict check -------------------------
    if config.process_orders_on_close:
        # Detect reversal + same-bar management exit — a common signal generator
        # bug when position state is updated immediately (post-fill) instead of
        # using pre-fill state.  With process_orders_on_close, Pine evaluates
        # strategy.position_size BEFORE fills, so the management block for the
        # OLD position should NOT fire on the reversal bar.
        conflict_long = (
            df["short_exit"] & df["long_entry"] & df["long_exit"]
        ).sum()
        conflict_short = (
            df["long_exit"] & df["short_entry"] & df["short_exit"]
        ).sum()
        if conflict_long + conflict_short > 0:
            import warnings
            warnings.warn(
                f"\n  (!)  PROCESS_ORDERS_ON_CLOSE: {conflict_long + conflict_short} bars have "
                f"reversal entry + management exit on the same bar.\n"
                f"  This usually means the signal generator updates position state immediately\n"
                f"  (post-fill) instead of using the pre-fill state.  With process_orders_on_close,\n"
                f"  Pine's strategy.position_size is pre-fill during calc — the management block\n"
                f"  for the old position should NOT fire on the reversal bar.\n"
                f"  Fix: use the position state at bar start for ALL condition checks in your\n"
                f"  signal generator (snapshot `pos = position` before signals, check `pos` not\n"
                f"  the live `position` variable).  See BACKTESTING.md for details.",
                stacklevel=2,
            )

    df = df.copy()
    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.end_date)

    # --- start-date safeguard ------------------------------------------------
    data_first = df.index[0]
    if data_first > start:
        print(f"\n  (!)  DATA STARTS AFTER start_date!")
        print(f"     start_date was:  {config.start_date}")
        print(f"     data starts at:  {data_first.date()}")
        print(f"     -> Adjusted start_date to {data_first.date()}")
        print(f"     To match TradingView, set the same start date in your")
        print(f"     TV strategy properties (Date Range -> Start Date).\n")
        start = data_first

    # --- state ---------------------------------------------------------------
    equity = config.initial_capital
    cash = config.initial_capital
    position_qty = 0.0          # positive = long, negative = short
    position_entry_price = 0.0  # weighted average for TP/SL (pyramiding)
    position_side = ""          # "long" or "short" or ""
    trades: list[Trade] = []
    open_positions: list[Trade] = []  # sub-positions (pyramiding support)

    pending_long_entry = False
    pending_long_exit = False
    pending_short_entry = False
    pending_short_exit = False
    pending_entry_qty = 0.0  # qty from qty_type sizing (signal time)
    entry_bar_idx = -1  # bar index where position was entered; TP/SL skipped on this bar

    equity_curve: list[dict] = []
    commission_rate = config.commission_pct / 100.0

    # TP/SL can come from config (percentages), DataFrame columns (absolute prices),
    # or DataFrame offset columns (distance from entry). Priority: price > offset > pct.
    has_tp_col = "tp_price" in df.columns
    has_sl_col = "sl_price" in df.columns
    has_tp_off = "tp_offset" in df.columns
    has_sl_off = "sl_offset" in df.columns
    tp_sl_active = (config.take_profit_pct > 0 or config.stop_loss_pct > 0
                    or has_tp_col or has_sl_col or has_tp_off or has_sl_off)

    # --- intrabar drawdown tracking ------------------------------------------
    peak_equity = config.initial_capital
    max_intrabar_dd = 0.0
    max_intrabar_dd_pct = 0.0

    # --- bar-by-bar loop -----------------------------------------------------
    main_loop = range(len(df))
    if config.show_progress:
        main_loop = tqdm(main_loop, desc="Backtesting Bars (LS)")

    for i in main_loop:
        bar = df.iloc[i]
        bar_date = df.index[i]
        bar_in_range = start <= bar_date <= end

        # 1) FILL pending orders at this bar's Open

        # --- Close existing positions first ---
        if pending_long_exit and position_side == "long" and position_qty > 0:
            fill_price = bar["Open"]
            for pos in open_positions:
                trade_value = pos.entry_qty * fill_price
                exit_commission = trade_value * commission_rate
                gross_pnl = pos.entry_qty * (fill_price - pos.entry_price)
                net_pnl = gross_pnl - pos.entry_commission - exit_commission
                cash += trade_value - exit_commission
                pos.exit_date = bar_date
                pos.exit_price = fill_price
                pos.pnl = net_pnl
                entry_value = pos.entry_qty * pos.entry_price
                pos.pnl_pct = (net_pnl / entry_value) * 100
                pos.exit_commission = exit_commission
                trades.append(pos)
            equity = cash
            open_positions = []
            position_qty = 0.0
            position_entry_price = 0.0
            position_side = ""
            pending_long_exit = False

            # Update peak equity while flat — critical for reversal strategies
            # where the next entry fills immediately after on the same bar.
            if bar_in_range and equity > peak_equity:
                peak_equity = equity

        if pending_short_exit and position_side == "short" and position_qty < 0:
            fill_price = bar["Open"]
            for pos in open_positions:
                abs_qty = pos.entry_qty
                trade_value = abs_qty * fill_price
                exit_commission = trade_value * commission_rate
                gross_pnl = abs_qty * (pos.entry_price - fill_price)
                net_pnl = gross_pnl - pos.entry_commission - exit_commission
                cash = cash + gross_pnl - exit_commission
                pos.exit_date = bar_date
                pos.exit_price = fill_price
                pos.pnl = net_pnl
                entry_value = abs_qty * pos.entry_price
                pos.pnl_pct = (net_pnl / entry_value) * 100
                pos.exit_commission = exit_commission
                trades.append(pos)
            equity = cash
            open_positions = []
            position_qty = 0.0
            position_entry_price = 0.0
            position_side = ""
            pending_short_exit = False

            # Update peak equity while flat — critical for reversal strategies
            # where the next entry fills immediately after on the same bar.
            if bar_in_range and equity > peak_equity:
                peak_equity = equity

        # --- Open new positions ---
        # TV sizes positions so that trade_value + entry_commission = equity,
        # i.e. trade_value = equity / (1 + commission_rate).  This ensures
        # the total outlay never exceeds available equity.

        # Implicit reversal at fill time: if an entry for the opposite
        # direction is pending while already in a position, close all
        # current sub-positions first.  Matches TV's strategy.entry()
        # which automatically reverses when called for the opposite side.
        # This handles "flash" trades where a pyramid add and a reversal
        # both fire on the same bar — the add fills first, then the
        # reversal closes everything (including the add) and opens new.
        if pending_long_entry and position_side == "short" and position_qty < 0:
            fill_price = bar["Open"]
            for pos in open_positions:
                abs_qty = pos.entry_qty
                trade_value = abs_qty * fill_price
                exit_commission = trade_value * commission_rate
                gross_pnl = abs_qty * (pos.entry_price - fill_price)
                net_pnl = gross_pnl - pos.entry_commission - exit_commission
                cash = cash + gross_pnl - exit_commission
                pos.exit_date = bar_date
                pos.exit_price = fill_price
                pos.pnl = net_pnl
                entry_value = abs_qty * pos.entry_price
                pos.pnl_pct = (net_pnl / entry_value) * 100
                pos.exit_commission = exit_commission
                trades.append(pos)
            equity = cash
            open_positions = []
            position_qty = 0.0
            position_entry_price = 0.0
            position_side = ""
            if bar_in_range and equity > peak_equity:
                peak_equity = equity

        n_open = len(open_positions)
        if pending_long_entry and (position_qty == 0 or (position_side == "long" and n_open < config.pyramiding)):
            fill_price = bar["Open"]
            if pending_entry_qty > 0:
                qty = pending_entry_qty
                trade_value = qty * fill_price
            else:
                trade_value = equity / (1 + commission_rate)
                qty = trade_value / fill_price
            entry_commission = trade_value * commission_rate
            position_qty += qty
            position_side = "long"
            cash -= (trade_value + entry_commission)
            new_trade = Trade(
                entry_date=bar_date, entry_price=fill_price,
                entry_qty=qty, direction="long",
                entry_commission=entry_commission,
            )
            open_positions.append(new_trade)
            # Weighted average entry price for TP/SL
            total_qty = sum(p.entry_qty for p in open_positions)
            position_entry_price = sum(p.entry_qty * p.entry_price for p in open_positions) / total_qty
            pending_long_entry = False
            entry_bar_idx = i

        if pending_short_entry and position_side == "long" and position_qty > 0:
            fill_price = bar["Open"]
            for pos in open_positions:
                trade_value = pos.entry_qty * fill_price
                exit_commission = trade_value * commission_rate
                gross_pnl = pos.entry_qty * (fill_price - pos.entry_price)
                net_pnl = gross_pnl - pos.entry_commission - exit_commission
                cash += trade_value - exit_commission
                pos.exit_date = bar_date
                pos.exit_price = fill_price
                pos.pnl = net_pnl
                entry_value = pos.entry_qty * pos.entry_price
                pos.pnl_pct = (net_pnl / entry_value) * 100
                pos.exit_commission = exit_commission
                trades.append(pos)
            equity = cash
            open_positions = []
            position_qty = 0.0
            position_entry_price = 0.0
            position_side = ""
            if bar_in_range and equity > peak_equity:
                peak_equity = equity

        n_open = len(open_positions)
        if pending_short_entry and (position_qty == 0 or (position_side == "short" and n_open < config.pyramiding)):
            fill_price = bar["Open"]
            if pending_entry_qty > 0:
                abs_qty = pending_entry_qty
                trade_value = abs_qty * fill_price
            else:
                trade_value = equity / (1 + commission_rate)
                abs_qty = trade_value / fill_price
            entry_commission = trade_value * commission_rate
            position_qty -= abs_qty  # more negative = larger short
            position_side = "short"
            cash -= entry_commission
            new_trade = Trade(
                entry_date=bar_date, entry_price=fill_price,
                entry_qty=abs_qty, direction="short",
                entry_commission=entry_commission,
            )
            open_positions.append(new_trade)
            # Weighted average entry price for TP/SL
            total_qty = sum(p.entry_qty for p in open_positions)
            position_entry_price = sum(p.entry_qty * p.entry_price for p in open_positions) / total_qty
            pending_short_entry = False
            entry_bar_idx = i

        # 1.5) TP/SL intrabar fill — fills at exact TP/SL price on this bar
        tpsl_filled = False
        if tp_sl_active and bar_in_range and position_qty != 0 and i > entry_bar_idx:
            bar_tp = bar["tp_price"] if has_tp_col else 0.0
            bar_sl = bar["sl_price"] if has_sl_col else 0.0
            bar_tp_off = bar["tp_offset"] if has_tp_off else 0.0
            bar_sl_off = bar["sl_offset"] if has_sl_off else 0.0
            fill_price, fill_type = _check_tpsl_fill(
                bar_open=bar["Open"], bar_high=bar["High"], bar_low=bar["Low"],
                entry_price=position_entry_price, position_side=position_side,
                tp_pct=config.take_profit_pct, sl_pct=config.stop_loss_pct,
                tp_price=bar_tp, sl_price=bar_sl,
                tp_offset=bar_tp_off, sl_offset=bar_sl_off,
            )
            if fill_price is not None:
                tpsl_filled = True

                if position_side == "long":
                    # Drawdown while still holding (aggregate)
                    if fill_type == "sl":
                        worst_price = fill_price
                    elif fill_price == bar["Open"]:
                        worst_price = fill_price
                    else:
                        worst_price = bar["Low"]
                    equity_at_worst = cash + position_qty * worst_price

                    # Settle each long sub-position at TP/SL price
                    for pos in open_positions:
                        trade_value = pos.entry_qty * fill_price
                        exit_commission = trade_value * commission_rate
                        gross_pnl = pos.entry_qty * (fill_price - pos.entry_price)
                        net_pnl = gross_pnl - pos.entry_commission - exit_commission
                        cash += trade_value - exit_commission
                        pos.exit_date = bar_date
                        pos.exit_price = fill_price
                        pos.pnl = net_pnl
                        entry_value = pos.entry_qty * pos.entry_price
                        pos.pnl_pct = (net_pnl / entry_value) * 100
                        pos.exit_commission = exit_commission
                        trades.append(pos)
                    equity = cash

                else:  # short
                    abs_qty = abs(position_qty)
                    # Drawdown while still holding (aggregate)
                    if fill_type == "sl":
                        worst_price = fill_price
                    elif fill_price == bar["Open"]:
                        worst_price = fill_price
                    else:
                        worst_price = bar["High"]
                    unrealised_pnl_worst = abs_qty * (position_entry_price - worst_price)
                    equity_at_worst = cash + unrealised_pnl_worst

                    # Settle each short sub-position at TP/SL price
                    for pos in open_positions:
                        p_qty = pos.entry_qty
                        trade_value = p_qty * fill_price
                        exit_commission = trade_value * commission_rate
                        gross_pnl = p_qty * (pos.entry_price - fill_price)
                        net_pnl = gross_pnl - pos.entry_commission - exit_commission
                        cash = cash + gross_pnl - exit_commission
                        pos.exit_date = bar_date
                        pos.exit_price = fill_price
                        pos.pnl = net_pnl
                        entry_value = p_qty * pos.entry_price
                        pos.pnl_pct = (net_pnl / entry_value) * 100
                        pos.exit_commission = exit_commission
                        trades.append(pos)
                    equity = cash

                # Drawdown update (common)
                dd = equity_at_worst - peak_equity
                dd_pct = (dd / peak_equity) * 100 if peak_equity != 0 else 0.0
                if dd < max_intrabar_dd:
                    max_intrabar_dd = dd
                if dd_pct < max_intrabar_dd_pct:
                    max_intrabar_dd_pct = dd_pct

                open_positions = []
                position_qty = 0.0
                position_entry_price = 0.0
                position_side = ""

                # Update peak equity (now flat after TP/SL exit)
                if bar_in_range and equity > peak_equity:
                    peak_equity = equity

        # 2a) Intrabar drawdown check (skip if TP/SL already handled it)
        if bar_in_range and position_qty != 0 and not tpsl_filled:
            if position_side == "long":
                # Worst case for long: price drops to bar Low
                equity_at_worst = cash + position_qty * bar["Low"]
            else:
                # Worst case for short: price spikes to bar High
                abs_qty = abs(position_qty)
                unrealised_pnl = abs_qty * (position_entry_price - bar["High"])
                equity_at_worst = cash + unrealised_pnl

            dd = equity_at_worst - peak_equity
            dd_pct = (dd / peak_equity) * 100 if peak_equity != 0 else 0.0
            if dd < max_intrabar_dd:
                max_intrabar_dd = dd
            if dd_pct < max_intrabar_dd_pct:
                max_intrabar_dd_pct = dd_pct

        # 2b) Mark-to-market equity at Close
        if position_side == "long" and position_qty > 0:
            equity = cash + position_qty * bar["Close"]
        elif position_side == "short" and position_qty < 0:
            abs_qty = abs(position_qty)
            unrealised_pnl = abs_qty * (position_entry_price - bar["Close"])
            equity = cash + unrealised_pnl
        else:
            equity = cash

        if bar_in_range:
            equity_curve.append({"date": bar_date, "equity": equity})

        # 2c) Update peak equity — ONLY when flat
        if bar_in_range and position_qty == 0 and equity > peak_equity:
            peak_equity = equity

        # 3) Detect signals at Close (only inside trading window)
        pending_long_entry = False
        pending_long_exit = False
        pending_short_entry = False
        pending_short_exit = False
        pending_entry_qty = 0.0

        if bar_in_range:
            n_open = len(open_positions)
            # Long signals
            if bar["long_entry"] and (position_qty == 0 or (position_side == "long" and n_open < config.pyramiding)):
                pending_long_entry = True
            if bar["long_exit"] and position_side == "long" and position_qty > 0:
                pending_long_exit = True
            # Short signals (independent from long)
            if bar["short_entry"] and (position_qty == 0 or (position_side == "short" and n_open < config.pyramiding)):
                pending_short_entry = True
            if bar["short_exit"] and position_side == "short" and position_qty < 0:
                pending_short_exit = True

            # Reversal: if exiting one side and entering the other on the
            # same bar, allow both to queue.  The exit fills first on the
            # next bar's Open (setting position_qty=0), then the entry fills
            # immediately after — matching TV's strategy.entry() reversal.
            if pending_long_exit and bar["short_entry"]:
                pending_short_entry = True
            if pending_short_exit and bar["long_entry"]:
                pending_long_entry = True

            # Signal-time sizing for non-default qty_type.
            # MUST run after reversal logic so reversal entries also get sized.
            if pending_long_entry or pending_short_entry:
                if config.qty_type == "cash":
                    pending_entry_qty = config.qty_value / bar["Close"]
                elif config.qty_type == "fixed":
                    pending_entry_qty = config.qty_value
                elif config.qty_type == "percent_of_equity" and config.qty_value < 100.0:
                    target = equity * config.qty_value / 100.0
                    pending_entry_qty = target / ((1 + commission_rate) * bar["Close"])

        # 3.5) process_orders_on_close: fill at THIS bar's Close
        if config.process_orders_on_close:

            # --- Long exit at Close ---
            if pending_long_exit and position_side == "long" and position_qty > 0:
                fill_price = bar["Close"]
                for pos in open_positions:
                    trade_value = pos.entry_qty * fill_price
                    exit_commission = trade_value * commission_rate
                    gross_pnl = pos.entry_qty * (fill_price - pos.entry_price)
                    net_pnl = gross_pnl - pos.entry_commission - exit_commission
                    cash += trade_value - exit_commission
                    pos.exit_date = bar_date
                    pos.exit_price = fill_price
                    pos.pnl = net_pnl
                    entry_value = pos.entry_qty * pos.entry_price
                    pos.pnl_pct = (net_pnl / entry_value) * 100
                    pos.exit_commission = exit_commission
                    trades.append(pos)
                equity = cash
                open_positions = []
                position_qty = 0.0
                position_entry_price = 0.0
                position_side = ""
                pending_long_exit = False

                # Peak equity while flat — critical for reversal on same bar
                if bar_in_range and equity > peak_equity:
                    peak_equity = equity

            # --- Short exit at Close ---
            if pending_short_exit and position_side == "short" and position_qty < 0:
                fill_price = bar["Close"]
                for pos in open_positions:
                    abs_qty = pos.entry_qty
                    trade_value = abs_qty * fill_price
                    exit_commission = trade_value * commission_rate
                    gross_pnl = abs_qty * (pos.entry_price - fill_price)
                    net_pnl = gross_pnl - pos.entry_commission - exit_commission
                    cash = cash + gross_pnl - exit_commission
                    pos.exit_date = bar_date
                    pos.exit_price = fill_price
                    pos.pnl = net_pnl
                    entry_value = abs_qty * pos.entry_price
                    pos.pnl_pct = (net_pnl / entry_value) * 100
                    pos.exit_commission = exit_commission
                    trades.append(pos)
                equity = cash
                open_positions = []
                position_qty = 0.0
                position_entry_price = 0.0
                position_side = ""
                pending_short_exit = False

                # Peak equity while flat — critical for reversal on same bar
                if bar_in_range and equity > peak_equity:
                    peak_equity = equity

            # --- Implicit reversal at Close (same logic as Open fills) ---
            if pending_long_entry and position_side == "short" and position_qty < 0:
                fill_price = bar["Close"]
                for pos in open_positions:
                    abs_qty = pos.entry_qty
                    trade_value = abs_qty * fill_price
                    exit_commission = trade_value * commission_rate
                    gross_pnl = abs_qty * (pos.entry_price - fill_price)
                    net_pnl = gross_pnl - pos.entry_commission - exit_commission
                    cash = cash + gross_pnl - exit_commission
                    pos.exit_date = bar_date
                    pos.exit_price = fill_price
                    pos.pnl = net_pnl
                    entry_value = abs_qty * pos.entry_price
                    pos.pnl_pct = (net_pnl / entry_value) * 100
                    pos.exit_commission = exit_commission
                    trades.append(pos)
                equity = cash
                open_positions = []
                position_qty = 0.0
                position_entry_price = 0.0
                position_side = ""
                if bar_in_range and equity > peak_equity:
                    peak_equity = equity

            # --- Long entry at Close ---
            n_open_close = len(open_positions)
            if pending_long_entry and (position_qty == 0 or (position_side == "long" and n_open_close < config.pyramiding)):
                fill_price = bar["Close"]
                if pending_entry_qty > 0:
                    qty = pending_entry_qty
                    trade_value = qty * fill_price
                else:
                    trade_value = equity / (1 + commission_rate)
                    qty = trade_value / fill_price
                entry_commission = trade_value * commission_rate
                position_qty += qty
                position_side = "long"
                cash -= (trade_value + entry_commission)
                new_trade = Trade(
                    entry_date=bar_date, entry_price=fill_price,
                    entry_qty=qty, direction="long",
                    entry_commission=entry_commission,
                )
                open_positions.append(new_trade)
                total_qty = sum(p.entry_qty for p in open_positions)
                position_entry_price = sum(p.entry_qty * p.entry_price for p in open_positions) / total_qty
                pending_long_entry = False
                entry_bar_idx = i

            # --- Implicit reversal before short entry at Close ---
            if pending_short_entry and position_side == "long" and position_qty > 0:
                fill_price = bar["Close"]
                for pos in open_positions:
                    trade_value = pos.entry_qty * fill_price
                    exit_commission = trade_value * commission_rate
                    gross_pnl = pos.entry_qty * (fill_price - pos.entry_price)
                    net_pnl = gross_pnl - pos.entry_commission - exit_commission
                    cash += trade_value - exit_commission
                    pos.exit_date = bar_date
                    pos.exit_price = fill_price
                    pos.pnl = net_pnl
                    entry_value = pos.entry_qty * pos.entry_price
                    pos.pnl_pct = (net_pnl / entry_value) * 100
                    pos.exit_commission = exit_commission
                    trades.append(pos)
                equity = cash
                open_positions = []
                position_qty = 0.0
                position_entry_price = 0.0
                position_side = ""
                if bar_in_range and equity > peak_equity:
                    peak_equity = equity

            # --- Short entry at Close ---
            if pending_short_entry and (position_qty == 0 or (position_side == "short" and len(open_positions) < config.pyramiding)):
                fill_price = bar["Close"]
                if pending_entry_qty > 0:
                    abs_qty = pending_entry_qty
                    trade_value = abs_qty * fill_price
                else:
                    trade_value = equity / (1 + commission_rate)
                    abs_qty = trade_value / fill_price
                entry_commission = trade_value * commission_rate
                position_qty -= abs_qty
                position_side = "short"
                cash -= entry_commission
                new_trade = Trade(
                    entry_date=bar_date, entry_price=fill_price,
                    entry_qty=abs_qty, direction="short",
                    entry_commission=entry_commission,
                )
                open_positions.append(new_trade)
                total_qty = sum(p.entry_qty for p in open_positions)
                position_entry_price = sum(p.entry_qty * p.entry_price for p in open_positions) / total_qty
                pending_short_entry = False
                entry_bar_idx = i

            # Re-mark equity after Close fills
            if position_side == "long" and position_qty > 0:
                equity = cash + position_qty * bar["Close"]
            elif position_side == "short" and position_qty < 0:
                abs_qty = abs(position_qty)
                unrealised_pnl = abs_qty * (position_entry_price - bar["Close"])
                equity = cash + unrealised_pnl
            else:
                equity = cash

            # Overwrite equity curve entry for this bar
            if bar_in_range and equity_curve and equity_curve[-1]["date"] == bar_date:
                equity_curve[-1]["equity"] = equity

            # Update peak equity if flat after Close fills
            if bar_in_range and position_qty == 0 and equity > peak_equity:
                peak_equity = equity

    # Record any open positions at end of data
    for pos in open_positions:
        trades.append(pos)

    equity_df = pd.DataFrame(equity_curve)
    kpis = compute_kpis(trades, equity_df, config,
                        max_intrabar_dd, max_intrabar_dd_pct)
    kpis["actual_start_date"] = str(start.date())
    kpis["actual_end_date"] = str(end.date())
    return kpis


# ---------------------------------------------------------------------------
# KPI computation
# ---------------------------------------------------------------------------

def compute_kpis(
    trades: list[Trade],
    equity_df: pd.DataFrame,
    config: BacktestConfig,
    max_intrabar_dd: float = 0.0,
    max_intrabar_dd_pct: float = 0.0,
) -> dict:
    """Compute performance KPIs matching TradingView's Strategy Tester.

    Max drawdown uses TradingView's intrabar methodology:
    - Uses bar Low prices for worst-case equity during open long positions
    - Peak equity only updates when flat (from closed-trade equity)
    - Max DD ($) and max DD (%) are independent maximums
    """

    initial_capital = config.initial_capital

    if not trades:
        return {"error": "No trades executed"}

    final_equity = equity_df["equity"].iloc[-1] if len(equity_df) > 0 else initial_capital

    # Separate closed vs open trades
    closed_trades = [t for t in trades if t.exit_date is not None]
    open_trades = [t for t in trades if t.exit_date is None]

    # Profit / loss — net_profit based on CLOSED trades only (matches TV)
    winning_trades = [t for t in closed_trades if t.pnl > 0]
    losing_trades = [t for t in closed_trades if t.pnl <= 0]

    gross_profit = sum(t.pnl for t in winning_trades)
    gross_loss = sum(t.pnl for t in losing_trades)
    net_profit = gross_profit + gross_loss
    net_profit_pct = (net_profit / initial_capital) * 100

    # Open P&L = equity beyond what closed trades account for
    open_profit = (final_equity - initial_capital) - net_profit

    # Total P&L = closed + open (matches TV Overview "Total P&L")
    total_pnl = net_profit + open_profit
    total_pnl_pct = (total_pnl / initial_capital) * 100

    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float("inf")

    # Drawdown — intrabar methodology matching TradingView
    max_drawdown = max_intrabar_dd
    max_drawdown_pct = max_intrabar_dd_pct

    # Trade statistics (closed trades only — matching TV)
    total_trades = len(closed_trades)
    num_winning = len(winning_trades)
    num_losing = len(losing_trades)
    win_rate = (num_winning / total_trades) * 100 if total_trades > 0 else 0

    avg_trade = net_profit / total_trades if total_trades > 0 else 0
    avg_trade_pct = sum(t.pnl_pct for t in closed_trades) / total_trades if total_trades > 0 else 0
    avg_winning = gross_profit / num_winning if num_winning > 0 else 0
    avg_losing = gross_loss / num_losing if num_losing > 0 else 0
    avg_win_loss_ratio = abs(avg_winning / avg_losing) if avg_losing != 0 else float("inf")

    largest_winning = max((t.pnl for t in winning_trades), default=0)
    largest_losing = min((t.pnl for t in losing_trades), default=0)

    # Consecutive wins / losses
    max_consec_wins = max_consec_losses = 0
    cur_w = cur_l = 0
    for t in closed_trades:
        if t.pnl > 0:
            cur_w += 1; cur_l = 0
            max_consec_wins = max(max_consec_wins, cur_w)
        else:
            cur_l += 1; cur_w = 0
            max_consec_losses = max(max_consec_losses, cur_l)

    total_commission = sum(t.entry_commission + t.exit_commission for t in closed_trades)

    # First/last order dates (for TV verification)
    # First order = earliest entry; last order = latest entry or exit
    first_order_date = trades[0].entry_date if trades else None
    all_dates = [t.entry_date for t in trades]
    all_dates += [t.exit_date for t in trades if t.exit_date is not None]
    last_order_date = max(all_dates) if all_dates else None

    return {
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl_pct,
        "net_profit": net_profit,
        "net_profit_pct": net_profit_pct,
        "open_profit": open_profit,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "total_trades": total_trades,
        "num_winning": num_winning,
        "num_losing": num_losing,
        "win_rate": win_rate,
        "avg_trade": avg_trade,
        "avg_trade_pct": avg_trade_pct,
        "avg_winning": avg_winning,
        "avg_losing": avg_losing,
        "avg_win_loss_ratio": avg_win_loss_ratio,
        "largest_winning": largest_winning,
        "largest_losing": largest_losing,
        "max_consec_wins": max_consec_wins,
        "max_consec_losses": max_consec_losses,
        "total_commission": total_commission,
        "final_equity": final_equity,
        "initial_capital": initial_capital,
        "first_order_date": first_order_date,
        "last_order_date": last_order_date,
        "trades": trades,
    }


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def print_kpis(kpis: dict):
    """Print KPIs in a format similar to TradingView's Strategy Tester.

    Display order matches what users see in TV:
    - Total P&L (incl. open) shown first — matches TV Overview headline
    - Net Profit (closed only) shown second — matches TV Excel export
    - % always shown alongside $ for Net Profit and Max Drawdown
    """
    print("=" * 60)
    print("  STRATEGY PERFORMANCE SUMMARY")
    print("=" * 60)
    print()
    has_open = abs(kpis.get('open_profit', 0)) > 0.005
    if has_open:
        print(f"  Total P&L (incl. open): ${kpis['total_pnl']:>10,.2f}  ({kpis['total_pnl_pct']:>8.2f}%)")
    print(f"  Net Profit (closed):    ${kpis['net_profit']:>10,.2f}  ({kpis['net_profit_pct']:>8.2f}%)")
    if has_open:
        open_pct = (kpis['open_profit'] / kpis['initial_capital']) * 100
        print(f"  Open P&L:               ${kpis['open_profit']:>10,.2f}  ({open_pct:>8.2f}%)")
    print(f"  Gross Profit:           ${kpis['gross_profit']:>10,.2f}")
    print(f"  Gross Loss:             ${kpis['gross_loss']:>10,.2f}")
    print()
    print(f"  Profit Factor:         {kpis['profit_factor']:>12.3f}")
    print(f"  Max Drawdown:         ${kpis['max_drawdown']:>12,.2f}  ({kpis['max_drawdown_pct']:>8.2f}%)")
    print()
    print(f"  Total Trades:          {kpis['total_trades']:>12d}")
    print(f"  Winning Trades:        {kpis['num_winning']:>12d}  ({kpis['win_rate']:>6.2f}%)")
    print(f"  Losing Trades:         {kpis['num_losing']:>12d}")
    print()
    print(f"  Avg Trade:            ${kpis['avg_trade']:>12,.2f}  ({kpis['avg_trade_pct']:>8.2f}%)")
    print(f"  Avg Winning Trade:    ${kpis['avg_winning']:>12,.2f}")
    print(f"  Avg Losing Trade:     ${kpis['avg_losing']:>12,.2f}")
    print(f"  Avg Win/Loss Ratio:    {kpis['avg_win_loss_ratio']:>12.3f}")
    print()
    print(f"  Largest Win:          ${kpis['largest_winning']:>12,.2f}")
    print(f"  Largest Loss:         ${kpis['largest_losing']:>12,.2f}")
    print()
    print(f"  Max Consec. Wins:      {kpis['max_consec_wins']:>12d}")
    print(f"  Max Consec. Losses:    {kpis['max_consec_losses']:>12d}")
    print()
    print(f"  Total Commission:     ${kpis['total_commission']:>12,.2f}")
    print(f"  Initial Capital:      ${kpis['initial_capital']:>12,.2f}")
    print(f"  Final Equity:         ${kpis['final_equity']:>12,.2f}")
    print()
    first_dt = kpis.get("first_order_date")
    last_dt = kpis.get("last_order_date")
    if first_dt and last_dt:
        print(f"  First Order:           {first_dt.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Last Order:            {last_dt.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)


def print_trades(trades: list[Trade], max_trades: int = 0):
    """Print trade list.

    Automatically shows a 'Dir' column when short trades are present.
    """
    print()
    has_shorts = any(t.direction == "short" for t in trades)

    if has_shorts:
        header = (f"  {'#':>3}  {'Dir':>5}  {'Entry Date':>12}  {'Entry $':>10}  {'Exit Date':>12}  "
                  f"{'Exit $':>10}  {'Qty':>12}  {'PnL $':>12}  {'PnL %':>8}")
        print(header)
        print("  " + "-" * 103)
    else:
        header = (f"  {'#':>3}  {'Entry Date':>12}  {'Entry $':>10}  {'Exit Date':>12}  "
                  f"{'Exit $':>10}  {'Qty':>12}  {'PnL $':>12}  {'PnL %':>8}")
        print(header)
        print("  " + "-" * 95)

    display = trades if max_trades == 0 else trades[:max_trades]
    for i, t in enumerate(display, 1):
        exit_date = t.exit_date.strftime("%Y-%m-%d") if t.exit_date else "OPEN"
        exit_price = f"{t.exit_price:>10,.2f}" if t.exit_price else "      OPEN"
        pnl = f"{t.pnl:>12,.2f}" if t.pnl is not None else "        N/A"
        pnl_pct = f"{t.pnl_pct:>8.2f}" if t.pnl_pct is not None else "     N/A"
        direction = t.direction.upper()[:5]

        if has_shorts:
            print(f"  {i:>3}  {direction:>5}  {t.entry_date.strftime('%Y-%m-%d'):>12}  {t.entry_price:>10,.2f}  "
                  f"{exit_date:>12}  {exit_price}  {t.entry_qty:>12.6f}  {pnl}  {pnl_pct}")
        else:
            print(f"  {i:>3}  {t.entry_date.strftime('%Y-%m-%d'):>12}  {t.entry_price:>10,.2f}  "
                  f"{exit_date:>12}  {exit_price}  {t.entry_qty:>12.6f}  {pnl}  {pnl_pct}")

    if max_trades and len(trades) > max_trades:
        print(f"  ... ({len(trades) - max_trades} more trades)")
