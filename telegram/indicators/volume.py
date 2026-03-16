"""Volume Indicators"""

import numpy as np
from typing import List, Union, Dict, Any

class OBV:
    """On-Balance Volume"""
    
    def __init__(self):
        self.values = []
        
    def calculate(self, prices: List[float], volumes: List[float]) -> List[float]:
        """Calculate OBV"""
        if len(prices) < 2:
            return []
            
        obv = [volumes[0]]
        
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                obv.append(obv[i-1] + volumes[i])
            elif prices[i] < prices[i-1]:
                obv.append(obv[i-1] - volumes[i])
            else:
                obv.append(obv[i-1])
        
        return obv

class VWAP:
    """Volume Weighted Average Price"""
    
    def __init__(self):
        self.values = []
        
    def calculate(self, prices: List[float], volumes: List[float]) -> List[float]:
        """Calculate VWAP"""
        if len(prices) < 1:
            return []
            
        vwap = []
        cumulative_price_volume = 0
        cumulative_volume = 0
        
        for i in range(len(prices)):
            typical_price = (prices[i] + prices[i] + prices[i]) / 3
            cumulative_price_volume += typical_price * volumes[i]
            cumulative_volume += volumes[i]
            
            if cumulative_volume == 0:
                vwap.append(np.nan)
            else:
                vwap.append(cumulative_price_volume / cumulative_volume)
        
        return vwap