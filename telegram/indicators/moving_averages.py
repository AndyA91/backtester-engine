"""Moving Average Indicators"""

import numpy as np
from typing import List, Union

class SMA:
    """Simple Moving Average"""
    
    def __init__(self, period: int):
        self.period = period
        self.values = []
        
    def calculate(self, prices: List[float]) -> List[float]:
        """Calculate SMA for given prices"""
        if len(prices) < self.period:
            return []
            
        sma = []
        for i in range(len(prices)):
            if i < self.period - 1:
                sma.append(np.nan)
            else:
                window = prices[i-self.period+1:i+1]
                sma.append(np.mean(window))
        return sma

class EMA:
    """Exponential Moving Average"""
    
    def __init__(self, period: int):
        self.period = period
        self.values = []
        
    def calculate(self, prices: List[float]) -> List[float]:
        """Calculate EMA for given prices"""
        if len(prices) < self.period:
            return []
            
        ema = []
        k = 2 / (self.period + 1)
        
        for i in range(len(prices)):
            if i < self.period - 1:
                ema.append(np.nan)
            elif i == self.period - 1:
                ema.append(np.mean(prices[:self.period]))
            else:
                ema.append(prices[i] * k + ema[i-1] * (1 - k))
        
        return ema

class WMA:
    """Weighted Moving Average"""
    
    def __init__(self, period: int):
        self.period = period
        self.values = []
        
    def calculate(self, prices: List[float]) -> List[float]:
        """Calculate WMA for given prices"""
        if len(prices) < self.period:
            return []
            
        wma = []
        for i in range(len(prices)):
            if i < self.period - 1:
                wma.append(np.nan)
            else:
                window = prices[i-self.period+1:i+1]
                weights = np.arange(1, self.period + 1)
                weighted_sum = np.sum(np.array(window) * weights)
                wma.append(weighted_sum / np.sum(weights))
        
        return wma