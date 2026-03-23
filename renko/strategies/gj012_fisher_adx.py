"""
GJ012: Fisher Transform Cross + ADX Gate — GBPJPY Renko 0.05

Port of BTC001 Fisher+ADX strategy to GBPJPY. Fisher Transform converts price
into a Gaussian distribution with sharp turning points. On Renko bricks, Fisher
crosses are crisp because each brick is a fixed price move.

Entry logic:
  LONG:  Fisher crosses above signal + optional brick confirm + optional ADX gate
  SHORT: Fisher crosses below signal + optional brick confirm + optional ADX gate

Exit: first opposing Renko brick.

Data: OANDA_GBPJPY, 1S renko 0.05.csv

GBPJPY engine calibration:
  Commission: 0.005% (same as gj001).
  Initial capital: 150,000 JPY.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from indicators.fisher_transform import calc_fisher_transform
from renko.data import load_renko_export
from renko.indicators import add_renko_indicators

DESCRIPTION = "GBPJPY Fisher Transform cross entry with Renko ADX regime gate"

HYPOTHESIS = (
    "Fisher Transform was proven on BTCUSD renko (PF 18.67). GBPJPY has the "
    "highest baseline PF (20.87 with simple brick-count). Fisher+ADX may "
    "capture different entry points from momentum plays, adding diversification "
    "to the GBPJPY strategy portfolio."
)

RENKO_FILE      = "OANDA_GBPJPY, 1S renko 0.05.csv"
COMMISSION_PCT  = 0.005
INITIAL_CAPITAL = 150_000.0

PARAM_GRID = {
    "fisher_period":     [8, 10, 13],
    "adx_threshold":     [0, 20, 25],
    "cooldown":          [5, 10, 20],
    "session_start":     [0],
    "vol_max":           [0],
    "psar_gate":         [False, True],
    "req_brick_confirm": [True, False],
}

_CACHE: dict = {}


def _get_or_build_cache(fisher_period: int) -> pd.DataFrame:
    if fisher_period in _CACHE:
        return _CACHE[fisher_period]

    df = load_renko_export(RENKO_FILE)
    add_renko_indicators(df)

    ft = calc_fisher_transform(df, period=fisher_period)
    fisher = ft["fisher"]
    signal = ft["signal"]

    n = len(df)
    fisher_bull_cross = np.zeros(n, dtype=bool)
    fisher_bear_cross = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if np.isnan(fisher[i]) or np.isnan(signal[i]):
            continue
        if np.isnan(fisher[i-1]) or np.isnan(signal[i-1]):
            continue
        fisher_bull_cross[i] = fisher[i] > signal[i] and fisher[i-1] <= signal[i-1]
        fisher_bear_cross[i] = fisher[i] < signal[i] and fisher[i-1] >= signal[i-1]

    df["fisher_bull"] = fisher_bull_cross
    df["fisher_bear"] = fisher_bear_cross

    _CACHE[fisher_period] = df
    return df


def generate_signals(
    df:                pd.DataFrame,
    fisher_period:     int  = 10,
    adx_threshold:     int  = 20,
    cooldown:          int  = 10,
    session_start:     int  = 0,
    vol_max:           float = 0.0,
    psar_gate:         bool = False,
    req_brick_confirm: bool = True,
) -> pd.DataFrame:
    warmup = max(fisher_period + 5, 50)

    c = _get_or_build_cache(fisher_period).reindex(df.index)

    brick_up     = df["brick_up"].values
    hours        = df.index.hour
    n            = len(df)

    fisher_bull  = c["fisher_bull"].fillna(False).values.astype(bool)
    fisher_bear  = c["fisher_bear"].fillna(False).values.astype(bool)

    adx_vals     = c["adx"].values
    vol_ratio    = c["vol_ratio"].values
    psar_dir     = c["psar_dir"].values

    long_entry   = np.zeros(n, dtype=bool)
    long_exit    = np.zeros(n, dtype=bool)
    short_entry  = np.zeros(n, dtype=bool)
    short_exit   = np.zeros(n, dtype=bool)

    in_position    = False
    trade_dir      = 0
    last_trade_bar = -999_999

    for i in range(warmup, n):
        up = bool(brick_up[i])

        if in_position:
            if trade_dir == 1 and not up:
                long_exit[i]  = True
                in_position   = False
                trade_dir     = 0
            elif trade_dir == -1 and up:
                short_exit[i] = True
                in_position   = False
                trade_dir     = 0

        if in_position:
            continue

        if session_start > 0 and hours[i] < session_start:
            continue

        if (i - last_trade_bar) < cooldown:
            continue

        if adx_threshold > 0:
            adx_val = adx_vals[i]
            if np.isnan(adx_val) or adx_val < adx_threshold:
                continue

        if vol_max > 0:
            vr = vol_ratio[i]
            if np.isnan(vr) or vr > vol_max:
                continue

        if fisher_bull[i]:
            if req_brick_confirm and not up:
                pass
            elif psar_gate and not np.isnan(psar_dir[i]) and psar_dir[i] != -1:
                pass
            else:
                long_entry[i]  = True
                in_position    = True
                trade_dir      = 1
                last_trade_bar = i

        elif fisher_bear[i]:
            if req_brick_confirm and up:
                pass
            elif psar_gate and not np.isnan(psar_dir[i]) and psar_dir[i] != 1:
                pass
            else:
                short_entry[i] = True
                in_position    = True
                trade_dir      = -1
                last_trade_bar = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
