"""
MYM Dual Smoothed Heikin Ashi v3 — Multi-Timeframe Sweep

Fixes from v2:
- ATR uses RMA (Wilder's) to match Pine's ta.atr() exactly
- Supports resampled timeframes (5m, 15m, 30m)
- All gates from v2: slow streak, session filter, ATR stop/target
"""

import numpy as np
import pandas as pd

DESCRIPTION = "Dual SHA v3 — RMA ATR, multi-timeframe"


def _ema(src: np.ndarray, length: int) -> np.ndarray:
    """EMA matching Pine's ta.ema()."""
    out = np.empty_like(src, dtype=float)
    out[0] = src[0]
    k = 2.0 / (length + 1)
    for i in range(1, len(src)):
        out[i] = src[i] * k + out[i - 1] * (1 - k)
    return out


def _rma(src: np.ndarray, length: int) -> np.ndarray:
    """RMA (Wilder's smoothing) matching Pine's ta.rma()."""
    out = np.empty_like(src, dtype=float)
    # Pine initializes RMA with SMA of first `length` values
    out[:length] = np.nan
    out[length - 1] = np.mean(src[:length])
    k = 1.0 / length
    for i in range(length, len(src)):
        out[i] = src[i] * k + out[i - 1] * (1 - k)
    return out


def _atr_rma(h, l, c, length):
    """ATR using RMA — matches Pine's ta.atr() exactly."""
    n = len(h)
    tr = np.empty(n, dtype=float)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    return _rma(tr, length)


def _smoothed_ha(o, h, l, c, length):
    """Compute smoothed HA. Returns (ha_open, ha_close, is_bull)."""
    s_o = _ema(o, length)
    s_h = _ema(h, length)
    s_l = _ema(l, length)
    s_c = _ema(c, length)

    n = len(o)
    ha_open = np.empty(n, dtype=float)
    ha_close = (s_o + s_h + s_l + s_c) / 4.0

    ha_open[0] = (s_o[0] + s_c[0]) / 2.0
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

    is_bull = ha_close >= ha_open
    return ha_open, ha_close, is_bull


def generate_signals(
    df: pd.DataFrame,
    fast_len: int = 3,
    slow_len: int = 22,
    cooldown: int = 30,
    min_slow_streak: int = 12,
    session_mode: str = "rth",
    exit_mode: str = "atr_only",
    atr_sl_mult: float = 1.5,
    atr_tp_mult: float = 4.0,
    atr_len: int = 14,
) -> pd.DataFrame:
    """Generate dual SHA v3 signals with RMA-based ATR."""
    o = df["Open"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    c = df["Close"].values.astype(float)
    n = len(df)

    _, _, f_bull = _smoothed_ha(o, h, l, c, fast_len)
    _, _, s_bull = _smoothed_ha(o, h, l, c, slow_len)
    atr_vals = _atr_rma(h, l, c, atr_len)

    # Slow streak counts
    slow_streaks = np.zeros(n, dtype=int)
    count = 1
    for i in range(1, n):
        if s_bull[i] == s_bull[i - 1]:
            count += 1
        else:
            count = 1
        slow_streaks[i] = count

    # Session filter (UTC-4 for EDT, March-April)
    use_session = session_mode != "none"
    in_session = np.ones(n, dtype=bool)
    if use_session:
        idx = df.index
        et_hour = np.array([(t.hour - 4) % 24 for t in idx])
        et_min = np.array([t.minute for t in idx])
        hhmm = et_hour * 100 + et_min

        if session_mode == "rth":
            in_session = (hhmm >= 930) & (hhmm < 1545)
        elif session_mode == "rth_skip15":
            in_session = (hhmm >= 945) & (hhmm < 1545)

    long_entry = np.zeros(n, dtype=bool)
    long_exit = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit = np.zeros(n, dtype=bool)

    warmup = max(fast_len, slow_len, atr_len) + 2
    last_exit_bar = -999_999
    pos = 0
    entry_price = 0.0
    entry_atr = 0.0

    for i in range(warmup, n):
        if np.isnan(atr_vals[i]):
            continue

        fast_bull_flip = f_bull[i] and not f_bull[i - 1]
        fast_bear_flip = not f_bull[i] and f_bull[i - 1]

        # ── Exits ──
        if pos == 1:
            atr_exit = False
            if entry_atr > 0:
                sl = entry_price - atr_sl_mult * entry_atr if atr_sl_mult > 0 else -1e18
                tp = entry_price + atr_tp_mult * entry_atr if atr_tp_mult > 0 else 1e18
                atr_exit = (l[i] <= sl) or (h[i] >= tp)

            if atr_exit:
                long_exit[i] = True
                pos = 0
                last_exit_bar = i

        elif pos == -1:
            atr_exit = False
            if entry_atr > 0:
                sl = entry_price + atr_sl_mult * entry_atr if atr_sl_mult > 0 else 1e18
                tp = entry_price - atr_tp_mult * entry_atr if atr_tp_mult > 0 else -1e18
                atr_exit = (h[i] >= sl) or (l[i] <= tp)

            if atr_exit:
                short_exit[i] = True
                pos = 0
                last_exit_bar = i

        # Session close
        if use_session and pos != 0 and not in_session[i]:
            if pos == 1:
                long_exit[i] = True
            elif pos == -1:
                short_exit[i] = True
            pos = 0
            last_exit_bar = i

        # ── Entries ──
        if pos == 0 and (i - last_exit_bar) >= cooldown and in_session[i]:
            streak_ok = (min_slow_streak == 0) or (slow_streaks[i] >= min_slow_streak)

            if streak_ok and fast_bull_flip and s_bull[i]:
                long_entry[i] = True
                pos = 1
                # Use next bar open as approximate entry price (matches engine)
                entry_price = o[i + 1] if i + 1 < n else c[i]
                entry_atr = atr_vals[i]
            elif streak_ok and fast_bear_flip and not s_bull[i]:
                short_entry[i] = True
                pos = -1
                entry_price = o[i + 1] if i + 1 < n else c[i]
                entry_atr = atr_vals[i]

    df["long_entry"] = long_entry
    df["long_exit"] = long_exit
    df["short_entry"] = short_entry
    df["short_exit"] = short_exit
    return df
