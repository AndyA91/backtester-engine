"""
GJ007: R001 + R002 Combined Baseline (GBPJPY)

Port of EURUSD R007 to GBPJPY. Tests whether the combined brick momentum edge
transfers to GBPJPY without any candle-space filters.

Entry logic:
  1. R002 (priority): prev N bricks all same dir, current brick opposes → enter.
     No cooldown — triggered by structure.
  2. R001 (fallback, cooldown gated): current N bricks all same dir → enter.

Exit: first opposing brick (unconditional).

Data:  OANDA_GBPJPY, 1S renko 0.05.csv  (TV export, Nov 2024–Mar 2026)
IS:    2024-11-21 → 2025-09-30  (~10 months)
OOS:   2025-10-01 → 2026-02-28  (5 months)

GBPJPY engine calibration:
  Commission: OANDA GBPJPY spread ~2 pips (0.02 JPY). Per-leg cost at ~195 JPY:
    0.01 JPY / 195 = 0.00513% → rounded to 0.005%
  Initial capital: 150,000 JPY ≈ $1,000 USD at ~150 JPY/USD.
    Keeps leverage consistent with EURUSD strategies; makes DD% meaningful in JPY.

Run with:
  python renko/runner.py gj007_combined --start 2024-11-21 --end 2025-09-30   (IS)
  python renko/runner.py gj007_combined --start 2025-10-01 --end 2026-02-28   (OOS)
"""

import numpy as np
import pandas as pd

RENKO_FILE      = "OANDA_GBPJPY, 1S renko 0.05.csv"
COMMISSION_PCT  = 0.005
INITIAL_CAPITAL = 150_000.0

DESCRIPTION = "R001 momentum + R002 initiation — dual-entry combined (GBPJPY baseline)"

HYPOTHESIS = (
    "EURUSD R007 achieved PF 11.96–13.00 IS / PF 5.32–6.12 OOS, with +78–481% net "
    "improvement vs standalone R001. GBPJPY has higher volatility (ATR ~1 JPY/day vs "
    "EURUSD ~0.0050) and a wider brick relative to pip size, but the underlying edge — "
    "Renko runs cluster directionally — should be instrument-agnostic. GJ007 establishes "
    "the brick-space baseline for GJ008 (ADX + vol + session gates)."
)

PARAM_GRID = {
    "n_bricks": [2, 3, 4, 5],
    "cooldown":  [10, 20, 30],
}
# 4 × 3 = 12 combinations


def generate_signals(
    df: pd.DataFrame,
    n_bricks: int = 3,
    cooldown: int = 10,
) -> pd.DataFrame:
    """
    R002 (reversal initiation) has priority; R001 (momentum continuation) is fallback.
    Exit on first opposing brick. Only R001 entries update last_r001_bar.
    """
    n        = len(df)
    brick_up = df["brick_up"].values

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    in_position   = False
    trade_dir     = 0
    last_r001_bar = -999_999

    warmup = max(n_bricks + 1, 30)

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # ── Exit: first opposing brick ─────────────────────────────────────────
        long_exit[i]  = not up
        short_exit[i] = up

        if in_position:
            if (trade_dir == 1 and not up) or (trade_dir == -1 and up):
                in_position = False
                trade_dir   = 0

        if in_position:
            continue

        # ── R002: N same-dir bricks, current opposes → counter-entry ───────────
        prev          = brick_up[i - n_bricks : i]
        prev_all_up   = bool(np.all(prev))
        prev_all_down = bool(not np.any(prev))

        if prev_all_up and not up:
            short_entry[i] = True
            in_position    = True
            trade_dir      = -1

        elif prev_all_down and up:
            long_entry[i] = True
            in_position   = True
            trade_dir     = 1

        # ── R001: N consecutive same-dir bricks → momentum entry ───────────────
        elif (i - last_r001_bar) >= cooldown:
            window   = brick_up[i - n_bricks + 1 : i + 1]
            all_up   = bool(np.all(window))
            all_down = bool(not np.any(window))

            if all_up:
                long_entry[i]  = True
                in_position    = True
                trade_dir      = 1
                last_r001_bar  = i

            elif all_down:
                short_entry[i] = True
                in_position    = True
                trade_dir      = -1
                last_r001_bar  = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
