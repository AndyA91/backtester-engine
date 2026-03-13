"""
Indicator library — Python conversions of TradingView Pine Script indicators.

Usage:
    from indicators.supertrend import calc_supertrend
    from indicators.bollinger import calc_bollinger_bands

Workflow:
    1. Drop the Pine Script source into indicators/  (e.g. supertrend.pine)
    2. Convert to a matching Python function  (e.g. supertrend.py)
    3. Import from any strategy in strategies/

Each .pine file is the TradingView source of truth.
Each .py file is the exact Python conversion, matched to the Pine math.
"""
