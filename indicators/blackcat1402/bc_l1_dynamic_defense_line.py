"""
Python translation of:
  [blackcat] L1 Dynamic Defense Line by blackcat1402
  https://www.tradingview.com/script/EfcQST4F-blackcat-L1-Dynamic-Defense-Line/

Identifies trend reversals via a stochastic-based defense oscillator.
The "Defense Line B" tracks price position within a 34-bar range;
its difference from a short EMA generates buy/sell signals.

Pine equivalence notes
----------------------
  stochasticValue = (typicalPrice - lowest34) / (highest34 - lowest34) * 100
    where typicalPrice = (2*close + high + low) / 4
  stochasticEma   = ta.ema(stochasticValue, 8)         — EMA-8 (always called)
  defenseLineB    = highest34 != lowest34 ? stochasticEma : 50
    Pine calls the EMA unconditionally for continuity, then picks 50 as fallback.
    Python equivalent: compute EMA on stochasticValue (0 when range=0), then overwrite
    with 50 where highest34 == lowest34.
  defenseLineB1   = ta.ema(defenseLineB, 5)
  buySellDiff     = defenseLineB - defenseLineB1
  sellSignal      = (buySellDiff[1] > 0 and buySellDiff <= 0)
                    OR (0 <= buySellDiff < 1 and close < open)
  buySignal       = not sellSignal AND buySellDiff[1] < 0 AND buySellDiff > 0

All state-based (recursive EMA) calculations use pandas ewm(adjust=False),
which seeds at the first value — matching Pine's ta.ema seeding behaviour.

Output columns added to df
--------------------------
  bc_defense_b        — Defense Line B (primary trend oscillator)
  bc_defense_b1       — EMA-5 of Defense Line B (signal line)
  bc_buy_sell_diff    — defense_b - defense_b1 (positive = bullish momentum)
  bc_short_line       — negative of bc_buy_sell_diff
  bc_buy_signal       — bool: bullish crossover signal
  bc_sell_signal      — bool: bearish crossover / bearish candle signal

Usage
-----
  from indicators.blackcat1402.bc_l1_dynamic_defense_line import calc_bc_dynamic_defense_line
  df = calc_bc_dynamic_defense_line(df)
"""

import numpy as np
import pandas as pd


def calc_bc_dynamic_defense_line(df: pd.DataFrame) -> pd.DataFrame:
    """
    Python translation of [blackcat] L1 Dynamic Defense Line.

    Parameters
    ----------
    df : DataFrame with columns High, Low, Close, Open

    Returns
    -------
    df with new columns appended (see module docstring).
    """
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]
    open_ = df["Open"]

    # ── 9-period range components ─────────────────────────────────────────────
    highest9 = high.rolling(9).max()
    lowest9  = low.rolling(9).min()
    nine_period_range = highest9 - lowest9

    # ── Stochastic-like defense line computation ──────────────────────────────
    highest34 = high.rolling(34).max()
    lowest34  = low.rolling(34).min()
    h34_range = highest34 - lowest34

    typical_price = (2.0 * close + high + low) / 4.0

    # Compute stochasticValue; set 0 where range == 0 (handled by Pine's fallback)
    stoch_value = np.where(
        h34_range != 0,
        (typical_price - lowest34) / h34_range * 100.0,
        0.0,
    )
    stoch_series = pd.Series(stoch_value, index=df.index)

    # EMA-8 — always computed (matches Pine calling EMA unconditionally)
    stoch_ema = stoch_series.ewm(span=8, adjust=False).mean()

    # defenseLineB: use EMA when range > 0, else 50
    defense_b_vals = np.where(h34_range.values != 0, stoch_ema.values, 50.0)
    defense_b = pd.Series(defense_b_vals, index=df.index)

    # EMA-5 of defenseLineB
    defense_b1 = defense_b.ewm(span=5, adjust=False).mean()

    buy_sell_diff = defense_b - defense_b1
    short_line    = -buy_sell_diff

    # ── Signal generation ─────────────────────────────────────────────────────
    bsd_prev = buy_sell_diff.shift(1)

    sell_signal = (
        ((bsd_prev > 0) & (buy_sell_diff <= 0))
        | ((buy_sell_diff >= 0) & (buy_sell_diff < 1) & (close < open_))
    )

    buy_signal = (~sell_signal) & (bsd_prev < 0) & (buy_sell_diff > 0)

    # ── Attach to DataFrame ───────────────────────────────────────────────────
    df = df.copy()
    df["bc_defense_b"]     = defense_b
    df["bc_defense_b1"]    = defense_b1
    df["bc_buy_sell_diff"] = buy_sell_diff
    df["bc_short_line"]    = short_line
    df["bc_buy_signal"]    = buy_signal
    df["bc_sell_signal"]   = sell_signal
    return df
