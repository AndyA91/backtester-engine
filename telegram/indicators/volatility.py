"""Volatility Indicators"""

import numpy as np
from typing import List, Union, Dict, Any

class BollingerBands:
    """Bollinger Bands"""
    
    def __init__(self, period: int = 20, std_dev: int = 2):
        self.period = period
        self.std_dev = std_dev
        self.values = []
        
    def calculate(self, prices: List[float]) -> Dict[str, List[float]]:
        """Calculate Bollinger Bands"""
        if len(prices) < self.period:
            return {'Middle': [], 'Upper': [], 'Lower': []}
            
        middle = []
        upper = []
        lower = []
        
        for i in range(len(prices)):
            if i < self.period - 1:
                middle.append(np.nan)
                upper.append(np.nan)
                lower.append(np.nan)
            else:
                window = prices[i-self.period+1:i+1]
                mean = np.mean(window)
                std = np.std(window)
                
                middle.append(mean)
                upper.append(mean + self.std_dev * std)
                lower.append(mean - self.std_dev * std)
        
        return {'Middle': middle, 'Upper': upper, 'Lower': lower}

class ATR:
    """Average True Range"""
    
    def __init__(self, period: int = 14):
        self.period = period
        self.values = []
        
    def calculate(self, highs: List[float], lows: List[float], closes: List[float]) -> List[float]:
        """Calculate ATR"""
        if len(highs) < self.period + 1:
            return []
            
        true_ranges = []
        
        for i in range(1, len(highs)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            true_ranges.append(max(tr1, tr2, tr3))
        
        atr = []
        
        for i in range(len(true_ranges)):
            if i < self.period - 1:
                atr.append(np.nan)
            elif i == self.period - 1:
                atr.append(np.mean(true_ranges[:self.period]))
            else:
                atr.append((atr[i-1] * (self.period - 1) + true_ranges[i]) / self.period)
        
        return atr