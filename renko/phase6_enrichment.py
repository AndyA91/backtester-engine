"""
Phase 6 extended enrichment — Tier 2+3 untapped indicators.

Adds indicator columns on top of the standard renko indicators
(which must already be present from add_renko_indicators()).

All outputs are pre-shifted by .shift(1) following Pitfall #7.
Each indicator is wrapped in try/except — failures fill with NaN.

Columns added:
    cci          CCI(20)                    — Commodity Channel Index
    ichi_pos     Ichimoku cloud position    — +1 above / -1 below / 0 inside
    wpr          Williams %R(14)            — range -100 to 0
    donch_mid    Donchian(20) midpoint      — channel midline
    escgo_fast   ESCGO fast line            — Ehlers Stochastic CG Oscillator
    escgo_slow   ESCGO slow (trigger) line  — ALMA of ESCGO
    ddl_diff     DDL buy-sell difference    — >0 bullish, <0 bearish
    motn_dx      MOTN DX composite         — DX oscillator line
    motn_zx      MOTN ZX composite         — ZX oscillator line
    mk_regime    Momentum King regime       — +1 UP, -1 DOWN, 0 FLAT (opt-in)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Tier 2
from indicators.cci import calc_cci
from indicators.ichimoku import calc_ichimoku, price_vs_cloud
from indicators.williams_r import calc_williams_r
from indicators.donchian import calc_donchian

# Tier 3
from indicators.blackcat1402.blackcat_l3_escgo import calc_bc_l3_escgo
from indicators.blackcat1402.bc_l1_dynamic_defense_line import (
    calc_bc_dynamic_defense_line,
)
from indicators.blackcat1402.bc_l1_multi_oscillator_trend_navigator import (
    calc_bc_multi_oscillator_trend_navigator,
)
from indicators.momentum_king import calc_momentum_king


def add_phase6_indicators(
    df: pd.DataFrame,
    include_mk: bool = False,
) -> pd.DataFrame:
    """
    Add Tier 2+3 indicator columns to a Renko DataFrame.

    All output columns are shifted by 1 bar (pre-shifted convention).

    Args:
        df: Renko DataFrame with standard indicators from add_renko_indicators().
        include_mk: Include Momentum King (only useful on GBPJPY with large bricks).

    Returns:
        df with added columns (in-place).
    """

    # ── Tier 2: Standalone indicators ────────────────────────────────────────

    # CCI(20) — Commodity Channel Index
    try:
        cci_result = calc_cci(df, period=20)
        df["cci"] = pd.Series(cci_result["cci"], index=df.index).shift(1).values
    except Exception as e:
        print(f"  WARN: CCI failed: {e}")
        df["cci"] = np.nan

    # Ichimoku cloud position (+1 above, -1 below, 0 inside)
    try:
        ichi = calc_ichimoku(df)
        pos = price_vs_cloud(df, ichi, kijun_period=26)
        df["ichi_pos"] = pd.Series(pos, index=df.index).shift(1).values
    except Exception as e:
        print(f"  WARN: Ichimoku failed: {e}")
        df["ichi_pos"] = np.nan

    # Williams %R(14)
    try:
        wpr_result = calc_williams_r(df, period=14)
        df["wpr"] = pd.Series(wpr_result["wpr"], index=df.index).shift(1).values
    except Exception as e:
        print(f"  WARN: Williams %R failed: {e}")
        df["wpr"] = np.nan

    # Donchian(20) midpoint
    try:
        donch = calc_donchian(df, period=20)
        df["donch_mid"] = pd.Series(donch["mid"], index=df.index).shift(1).values
    except Exception as e:
        print(f"  WARN: Donchian failed: {e}")
        df["donch_mid"] = np.nan

    # ── Tier 3: Complex indicators ───────────────────────────────────────────

    # ESCGO — Ehlers Stochastic Center-of-Gravity Oscillator (needs lowercase cols)
    try:
        df_lc = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        escgo_out = calc_bc_l3_escgo(df_lc)
        df["escgo_fast"] = escgo_out["bc_escgo_fast"].shift(1).values
        df["escgo_slow"] = escgo_out["bc_escgo_slow"].shift(1).values
    except Exception as e:
        print(f"  WARN: ESCGO failed: {e}")
        df["escgo_fast"] = np.nan
        df["escgo_slow"] = np.nan

    # Dynamic Defense Line — stochastic-based defense oscillator
    try:
        ddl = calc_bc_dynamic_defense_line(df)
        df["ddl_diff"] = ddl["bc_buy_sell_diff"].shift(1).values
    except Exception as e:
        print(f"  WARN: DDL failed: {e}")
        df["ddl_diff"] = np.nan

    # Multi-Oscillator Trend Navigator — DX/ZX composite
    try:
        motn = calc_bc_multi_oscillator_trend_navigator(df)
        df["motn_dx"] = motn["bc_motn_dx"].shift(1).values
        df["motn_zx"] = motn["bc_motn_zx"].shift(1).values
    except Exception as e:
        print(f"  WARN: MOTN failed: {e}")
        df["motn_dx"] = np.nan
        df["motn_zx"] = np.nan

    # Momentum King — adaptive momentum regime (GBPJPY only)
    if include_mk:
        try:
            mk = calc_momentum_king(df)
            regime_map = {
                "STRONG_UP": 1, "WEAK_UP": 1,
                "FLAT": 0,
                "WEAK_DOWN": -1, "STRONG_DOWN": -1,
            }
            regime_arr = np.array(
                [regime_map.get(str(r), 0) for r in mk["regime"]]
            )
            df["mk_regime"] = pd.Series(
                regime_arr, index=df.index
            ).shift(1).values
        except Exception as e:
            print(f"  WARN: Momentum King failed: {e}")
            df["mk_regime"] = np.nan

    return df
