"""
Signal Library — Reusable entry/exit signal generators.

Each function takes a DataFrame (with OHLCV) and returns a dict of boolean
arrays that can be assigned directly to df["long_entry"], df["long_exit"], etc.

Categories:
    entries_trend      — trend-following entries (EMA cross, Supertrend, etc.)
    entries_meanrev    — mean-reversion entries (RSI extreme, BB bounce, etc.)
    entries_momentum   — momentum entries (MACD cross, AO, etc.)
    entries_volume     — volume-based entries (OBV breakout, A/D, etc.)
    entries_pattern    — pattern entries (divergence, Fisher, pivots)
    exits              — exit signal generators (trailing, SAR, time-based, etc.)
"""
