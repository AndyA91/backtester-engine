"""Utility functions for indicators"""

import numpy as np
from typing import List, Union


def moving_average(prices: List[float], period: int) -> List[float]:
    """Calculate simple moving average"""
    if len(prices) < period:
        return []
    
    return [np.mean(prices[i-period+1:i+1]) if i >= period-1 else np.nan for i in range(len(prices))]

def exponential_moving_average(prices: List[float], period: int) -> List[float]:
    """Calculate exponential moving average"""
    if len(prices) < period:
        return []
    
    k = 2 / (period + 1)
    ema = [np.nan] * (period - 1) + [np.mean(prices[:period])]
    
    for i in range(period, len(prices)):
        ema.append(prices[i] * k + ema[i-1] * (1 - k))
    
    return ema

def standard_deviation(prices: List[float], period: int) -> List[float]:
    """Calculate standard deviation"""
    if len(prices) < period:
        return []
    
    return [np.std(prices[i-period+1:i+1]) if i >= period-1 else np.nan for i in range(len(prices))]

def smooth(values: List[float], period: int) -> List[float]:
    """Smooth values using simple moving average"""
    if len(values) < period:
        return [np.nan] * len(values)
    
    return [np.mean(values[i-period+1:i+1]) if i >= period-1 else np.nan for i in range(len(values))]