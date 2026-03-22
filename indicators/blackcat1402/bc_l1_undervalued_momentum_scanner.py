"""
Python translation of:
  [blackcat] L1 Undervalued Momentum Scanner by blackcat1402
  https://www.tradingview.com/script/9GL7ladc-blackcat-L1-Undervalued-Momentum-Scanner/

Scans for undervalued momentum conditions using a "Horse Line" (price
volatility oscillator), a "Trend Line" (KDJ-based), and several named
signal patterns (Dark Horse, Big Dark Horse, Run, Bottom, Opportunity).

Pine equivalence notes
----------------------
priceOscillator (A1) — NOTE: the Pine script computes:
  3 * ta.sma(priceVolatility, emaLength) - 2 * ta.sma(priceVolatility, emaLength)
  Both calls use the SAME series and period — this reduces to SMA(priceVolatility, 5).
  horseLine = ta.ema(priceOscillator, emaLength) = EMA5 of SMA5(priceVolatility)

forwardSet(condition, bars):
  Pine: when condition is True, set a counter to `bars`; decrement each bar; return counter > 0.
  Equivalent to: condition fired within the last `bars` bars (inclusive).
  Python: condition.rolling(bars, min_periods=1).max().astype(bool)

Opportunity condition:
  Pine: (sma5 - close) / close > 0.04 AND (sma10 - sma5) / sma5 > 0.04
  Uses the *previous* bar's condition: opportunityComing = opportunityCondition[1] ? 30 : 0
  Python: opportunityCondition.shift(1) * 30

Output columns added to df
--------------------------
  bc_ums_horse_line       — Horse Line (primary oscillator, 0–100)
  bc_ums_reverse_line     — 100 - Horse Line
  bc_ums_trend_line       — KDJ-based Trend Line
  bc_ums_momentum_change  — % change in horse line (A2)
  bc_ums_run              — bool: run signal (horse > 90, momentum declining, 7-bar hold)
  bc_ums_dark_horse       — bool: trend line crosses above horse line
  bc_ums_big_dark_horse   — bool: trend line crosses above 0 with horse < 40
  bc_ums_bottom           — bool: momentum bounce from deep negative
  bc_ums_opportunity      — bool: MA divergence signal (uses prev bar)
  bc_ums_trend_exhausted  — bool: horse crosses oscillator with overbought conditions
  bc_ums_buy              — bool: any buy condition
  bc_ums_sell             — bool: any sell condition

Usage
-----
  from indicators.blackcat1402.bc_l1_undervalued_momentum_scanner import (
      calc_bc_undervalued_momentum_scanner
  )
  df = calc_bc_undervalued_momentum_scanner(df)
"""

import numpy as np
import pandas as pd


def calc_bc_undervalued_momentum_scanner(
    df: pd.DataFrame,
    volatility_length: int = 55,
    kdj_length:        int = 21,
    position_length:   int = 60,
    ema_length:        int = 5,
) -> pd.DataFrame:
    """
    Python translation of [blackcat] L1 Undervalued Momentum Scanner.

    Parameters
    ----------
    df               : DataFrame with columns High, Low, Close
    volatility_length: range lookback for Horse Line (default 55)
    kdj_length       : KDJ lookback for Trend Line (default 21)
    position_length  : range lookback for inverse position (default 60)
    ema_length       : EMA/SMA period for smoothing (default 5)

    Returns
    -------
    df with new columns appended (see module docstring).
    """
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]

    # ── Horse Line ────────────────────────────────────────────────────────────
    h_vol = high.rolling(volatility_length).max()
    l_vol = low.rolling(volatility_length).min()
    r_vol = h_vol - l_vol

    price_volatility = np.where(r_vol > 0, (close - l_vol) / r_vol * 100.0, 50.0)
    price_volatility = pd.Series(price_volatility, index=df.index)

    # priceOscillator = 3*SMA5 - 2*SMA5 = SMA5 (identity — see docstring)
    price_oscillator = price_volatility.rolling(ema_length).mean()
    horse_line       = price_oscillator.ewm(span=ema_length, adjust=False).mean()
    reverse_line     = 100.0 - horse_line

    # ── Momentum change (A2) ──────────────────────────────────────────────────
    prev_horse      = horse_line.shift(1)
    momentum_change = np.where(
        prev_horse != 0,
        (horse_line - prev_horse) / prev_horse.abs() * 100.0,
        0.0,
    )
    momentum_change = pd.Series(momentum_change, index=df.index)

    # ── KDJ-based Trend Line ──────────────────────────────────────────────────
    h_kdj = high.rolling(kdj_length).max()
    l_kdj = low.rolling(kdj_length).min()
    r_kdj = h_kdj - l_kdj

    kdj_raw = np.where(r_kdj > 0, (close - l_kdj) / r_kdj * 100.0, 50.0)
    kdj_raw = pd.Series(kdj_raw, index=df.index)

    trend_comp1      = 4.0 * kdj_raw.rolling(6).mean()
    trend_comp2      = 3.0 * kdj_raw.rolling(5).mean()
    trend_inter      = trend_comp2.rolling(5).mean()
    trend_line       = (trend_comp1 - trend_inter).ewm(span=2, adjust=False).mean()

    # ── Inverse Position (A5) ─────────────────────────────────────────────────
    h_pos = high.rolling(position_length).max()
    l_pos = low.rolling(position_length).min()
    r_pos = h_pos - l_pos

    inverse_position = np.where(r_pos > 0, (h_pos - close) / r_pos * 100.0, 50.0)
    inverse_position = pd.Series(inverse_position, index=df.index)

    # ── Signal: Run ───────────────────────────────────────────────────────────
    # Condition: prev horse > 90 AND momentum_change < 0
    # Held for 7 bars via forward-set pattern
    run_cond   = (prev_horse > 90) & (momentum_change < 0)
    run_signal = run_cond.rolling(7, min_periods=1).max().astype(bool)

    # ── Signal: Dark Horse ────────────────────────────────────────────────────
    # trendLine crosses above horseLine AND horse > 20 AND horse < inversePosition
    tl_prev = trend_line.shift(1)
    hl_prev = horse_line.shift(1)
    dark_horse = (
        (tl_prev <= hl_prev)
        & (trend_line > horse_line)
        & (horse_line > 20)
        & (horse_line < inverse_position)
    )

    # ── Signal: Big Dark Horse ────────────────────────────────────────────────
    # trendLine crosses above 0 AND horse < 40
    big_dark_horse = (
        (tl_prev <= 0) & (trend_line > 0) & (horse_line < 40)
    )

    # ── Signal: Bottom ────────────────────────────────────────────────────────
    # momentumChange < -10 AND momentum rising from prev bar
    prev_mc    = momentum_change.shift(1)
    bottom_sig = (momentum_change < -10) & (momentum_change > prev_mc)

    # ── Signal: Opportunity ───────────────────────────────────────────────────
    sma5  = close.rolling(5).mean()
    sma10 = close.rolling(10).mean()
    opp_cond   = ((sma5 - close) / close > 0.04) & ((sma10 - sma5) / sma5 > 0.04)
    opp_signal = opp_cond.shift(1).fillna(False).astype(bool)   # uses prev bar's condition

    # ── Signal: Trend Exhausted ───────────────────────────────────────────────
    # horseLine crosses above priceOscillator AND horse > 70 AND trendLine > 80
    po_prev   = price_oscillator.shift(1)
    hl_prev2  = horse_line.shift(1)
    trend_exhausted = (
        (hl_prev2 <= po_prev)
        & (horse_line > price_oscillator)
        & (horse_line > 70)
        & (trend_line > 80)
    )

    # ── Composite buy/sell ────────────────────────────────────────────────────
    should_buy  = dark_horse | big_dark_horse | bottom_sig | opp_signal
    should_sell = run_signal | trend_exhausted

    # ── Attach to DataFrame ───────────────────────────────────────────────────
    df = df.copy()
    df["bc_ums_horse_line"]      = horse_line
    df["bc_ums_reverse_line"]    = reverse_line
    df["bc_ums_trend_line"]      = trend_line
    df["bc_ums_momentum_change"] = momentum_change
    df["bc_ums_run"]             = run_signal
    df["bc_ums_dark_horse"]      = dark_horse
    df["bc_ums_big_dark_horse"]  = big_dark_horse
    df["bc_ums_bottom"]          = bottom_sig
    df["bc_ums_opportunity"]     = opp_signal
    df["bc_ums_trend_exhausted"] = trend_exhausted
    df["bc_ums_buy"]             = should_buy
    df["bc_ums_sell"]            = should_sell
    return df
