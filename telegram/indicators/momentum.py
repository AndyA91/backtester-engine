"""Momentum Indicators"""

import numpy as np
from typing import List, Union

class RSI:
    """Relative Strength Index"""
    
    def __init__(self, period: int = 14):
        self.period = period
        self.values = []
        
    def calculate(self, prices: List[float]) -> List[float]:
        """Calculate RSI for given prices"""
        if len(prices) < self.period + 1:
            return []
            
        rsi = []
        
        # Calculate price changes
        deltas = np.diff(prices)
        
        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate initial averages
        avg_gain = np.mean(gains[:self.period])
        avg_loss = np.mean(losses[:self.period])
        
        for i in range(len(prices)):
            if i < self.period:
                rsi.append(np.nan)
            elif i == self.period:
                rs = avg_gain / avg_loss if avg_loss != 0 else 100
                rsi.append(100 - (100 / (1 + rs)))
            else:
                avg_gain = (avg_gain * (self.period - 1) + gains[i-1]) / self.period
                avg_loss = (avg_loss * (self.period - 1) + losses[i-1]) / self.period
                rs = avg_gain / avg_loss if avg_loss != 0 else 100
                rsi.append(100 - (100 / (1 + rs)))
        
        return rsi

class Stochastic:
    """Stochastic Oscillator"""
    
    def __init__(self, period: int = 14, smooth_k: int = 3, smooth_d: int = 3):
        self.period = period
        self.smooth_k = smooth_k
        self.smooth_d = smooth_d
        self.values = []
        
    def calculate(self, prices: List[float], highs: List[float], lows: List[float]) -> dict:
        """Calculate Stochastic Oscillator"""
        if len(prices) < self.period:
            return {'%K': [], '%D': []}
            
        %K = []
        
        for i in range(len(prices)):
            if i < self.period - 1:
                %K.append(np.nan)
            else:
                high = np.max(highs[i-self.period+1:i+1])
                low = np.min(lows[i-self.period+1:i+1])
                if high == low:
                    %K.append(0)
                else:
                    %K.append(100 * (prices[i] - low) / (high - low))
        
        # Smooth %K
        if self.smooth_k > 1:
            smoothed_K = []
            for i in range(len(%K)):
                if i < self.smooth_k - 1:
                    smoothed_K.append(np.nan)
                else:
                    window = %K[i-self.smooth_k+1:i+1]
                    smoothed_K.append(np.mean(window))
            %K = smoothed_K
        
        # Calculate %D
        %D = []
        for i in range(len(%K)):
            if i < self.smooth_d - 1:
                %D.append(np.nan)
            else:
                window = %K[i-self.smooth_d+1:i+1]
                %D.append(np.mean(window))
        
        return {'%K': %K, '%D': %D}