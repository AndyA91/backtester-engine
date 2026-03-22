"""
Python translation of:
  [blackcat] L1 Swing Trade Oscillator by blackcat1402
  https://www.tradingview.com/script/4BHvDmmc-blackcat-L1-Swing-Trade-Oscillator/

Identifies swing entries using a Main Force oscillator and a Life Line signal.
Main Force measures price position within a recent range; Life Line is a
smoothed version used for crossover-based entries and exits.

Pine equivalence notes
----------------------
Main Force:
  var2      = (2*close + high + low) / 4       — weighted typical price
  var4      = ta.lowest(low, n)                — n-bar low (default n=5)
  var5      = ta.highest(high, 4)              — 4-bar high (fixed)
  mainForce = ta.ema(((var2 - var4)/(var5-var4)) * 100, 4) * var1  (var1=multiplier, default 1)

Life Line (stateful — references previous mainForce bar):
  lifeLine = ta.ema((0.667 * mainForce[1] + 0.333 * mainForce), 2) * var1
  "mainForce[1]" is the *previous bar's* mainForce (not recursive self-reference).
  Python: blend mainForce.shift(1) and mainForce, then apply ewm(span=2).

Signals:
  buy_signal    = crossover(mainForce, lifeLine) AND mainForce < 40
  sell_signal1  = crossover(lifeLine, mainForce) AND mainForce > 90
  buy_signal2   = crossover(mainForce, lifeLine) AND mainForce[1] < 20
  sell_signal2  = crossover(lifeLine, mainForce) AND mainForce[1] > 80

Output columns added to df
--------------------------
  bc_sto_main_force    — Main Force oscillator (0–100 scale)
  bc_sto_life_line     — Life Line signal (0–100 scale)
  bc_sto_buy           — bool: main crossover above life with mainForce < 40
  bc_sto_sell1         — bool: life crossover above main with mainForce > 90
  bc_sto_buy2          — bool: crossover above life with prev mainForce < 20
  bc_sto_sell2         — bool: life crossover above main with prev mainForce > 80

Usage
-----
  from indicators.blackcat1402.bc_l1_swing_trade_oscillator import calc_bc_swing_trade_oscillator
  df = calc_bc_swing_trade_oscillator(df)
"""

import numpy as np
import pandas as pd


def calc_bc_swing_trade_oscillator(
    df: pd.DataFrame,
    n:    int   = 5,
    var1: float = 1.0,
) -> pd.DataFrame:
    """
    Python translation of [blackcat] L1 Swing Trade Oscillator.

    Parameters
    ----------
    df   : DataFrame with columns High, Low, Close
    n    : lookback for the lowest low (default 5, matches Pine default)
    var1 : output multiplier (default 1.0, matches Pine default)

    Returns
    -------
    df with new columns appended (see module docstring).
    """
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]

    # ── Core oscillator ───────────────────────────────────────────────────────
    var2 = (2.0 * close + high + low) / 4.0       # weighted price
    var4 = low.rolling(n).min()                    # n-bar low
    var5 = high.rolling(4).max()                   # 4-bar high (fixed)

    rng  = var5 - var4
    raw  = np.where(rng > 0, (var2 - var4) / rng * 100.0, 50.0)
    raw  = pd.Series(raw, index=df.index)

    main_force = raw.ewm(span=4, adjust=False).mean() * var1

    # ── Life Line ─────────────────────────────────────────────────────────────
    # Source = 0.667 * mainForce[1] + 0.333 * mainForce  (no self-reference)
    mf_prev    = main_force.shift(1)
    life_src   = 0.667 * mf_prev + 0.333 * main_force
    life_line  = life_src.ewm(span=2, adjust=False).mean() * var1

    # ── Signal detection ──────────────────────────────────────────────────────
    mf_prev_bar = main_force.shift(1)
    ll_prev_bar = life_line.shift(1)

    cross_up   = (mf_prev_bar <= ll_prev_bar) & (main_force > life_line)
    cross_down = (mf_prev_bar >= ll_prev_bar) & (main_force < life_line)

    buy_signal  = cross_up   & (main_force < 40)
    sell_signal1 = cross_down & (main_force > 90)
    buy_signal2  = cross_up   & (mf_prev_bar < 20)
    sell_signal2 = cross_down & (mf_prev_bar > 80)

    # ── Attach to DataFrame ───────────────────────────────────────────────────
    df = df.copy()
    df["bc_sto_main_force"] = main_force
    df["bc_sto_life_line"]  = life_line
    df["bc_sto_buy"]        = buy_signal
    df["bc_sto_sell1"]      = sell_signal1
    df["bc_sto_buy2"]       = buy_signal2
    df["bc_sto_sell2"]      = sell_signal2
    return df
