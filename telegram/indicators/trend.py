"""Trend Indicators"""

import numpy as np
from typing import List, Union, Dict, Any

class MACD:
    """Moving Average Convergence Divergence"""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.values = []
        
    def calculate(self, prices: List[float]) -> Dict[str, List[float]]:
        """Calculate MACD and signal line"""
        if len(prices) < self.slow_period:
            return {'MACD': [], 'Signal': [], 'Histogram': []}
            
        # Calculate EMAs
        fast_ema = self._ema(prices, self.fast_period)
        slow_ema = self._ema(prices, self.slow_period)
        
        # Calculate MACD line
        macd = []
        for i in range(len(prices)):
            if np.isnan(fast_ema[i]) or np.isnan(slow_ema[i]):
                macd.append(np.nan)
            else:
                macd.append(fast_ema[i] - slow_ema[i])
        
        # Calculate signal line
        signal = self._ema(macd, self.signal_period)
        
        # Calculate histogram
        histogram = []
        for i in range(len(macd)):
            if np.isnan(macd[i]) or np.isnan(signal[i]):
                histogram.append(np.nan)
            else:
                histogram.append(macd[i] - signal[i])
        
        return {'MACD': macd, 'Signal': signal, 'Histogram': histogram}
        
    def _ema(self, prices: List[float], period: int) -> List[float]:
        """Helper function to calculate EMA"""
        if len(prices) < period:
            return [np.nan] * len(prices)
            
        ema = []
        k = 2 / (period + 1)
        
        for i in range(len(prices)):
            if i < period - 1:
                ema.append(np.nan)
            elif i == period - 1:
                ema.append(np.mean(prices[:period]))
            else:
                ema.append(prices[i] * k + ema[i-1] * (1 - k))
        
        return ema

class ADX:
    """Average Directional Index"""
    
    def __init__(self, period: int = 14):
        self.period = period
        self.values = []
        
    def calculate(self, highs: List[float], lows: List[float], closes: List[float]) -> Dict[str, List[float]]:
        """Calculate ADX and directional movement indicators"""
        if len(highs) < self.period + 1:
            return {'ADX': [], '+DI': [], '-DI': []}
            
        # Calculate directional movement
        plus_dm = []
        minus_dm = []
        true_ranges = []
        
        for i in range(1, len(highs)):
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]
            
            plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0)
            minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0)
            
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            true_ranges.append(max(tr1, tr2, tr3))
        
        # Smooth directional movement and true range
        smoothed_plus_dm = self._smooth(plus_dm, self.period)
        smoothed_minus_dm = self._smooth(minus_dm, self.period)
        smoothed_tr = self._smooth(true_ranges, self.period)
        
        # Calculate directional indicators
        plus_di = []
        minus_di = []
        for i in range(len(smoothed_tr)):
            if smoothed_tr[i] == 0:
                plus_di.append(0)
                minus_di.append(0)
            else:
                plus_di.append(100 * smoothed_plus_dm[i] / smoothed_tr[i])
                minus_di.append(100 * smoothed_minus_dm[i] / smoothed_tr[i])
        
        # Calculate directional movement index
        dx = []
        for i in range(len(plus_di)):
            if plus_di[i] + minus_di[i] == 0:
                dx.append(0)
            else:
                dx.append(100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]))
        
        # Calculate ADX
        adx = self._smooth(dx, self.period)
        
        return {'ADX': adx, '+DI': plus_di, '-DI': minus_di}
        
    def _smooth(self, values: List[float], period: int) -> List[float]:
        """Helper function to smooth values"""
        if len(values) < period:
            return [np.nan] * len(values)
            
        smoothed = []
        
        for i in range(len(values)):
            if i < period - 1:
                smoothed.append(np.nan)
            elif i == period - 1:
                smoothed.append(np.mean(values[:period]))
            else:
                smoothed.append((smoothed[i-1] * (period - 1) + values[i]) / period)
        
        return smoothed

class ParabolicSAR:
    """Parabolic Stop and Reverse"""
    
    def __init__(self, acceleration_factor: float = 0.02, max_acceleration: float = 0.2):
        self.acceleration_factor = acceleration_factor
        self.max_acceleration = max_acceleration
        self.values = []
        
    def calculate(self, highs: List[float], lows: List[float]) -> List[float]:
        """Calculate Parabolic SAR"""
        if len(highs) < 2:
            return []
            
        sar = []
        
        # Initialize
        if highs[1] > highs[0]:
            trend_up = True
            sar.append(lows[0])
            extreme_point = highs[0]
            current_acceleration = self.acceleration_factor
        else:
            trend_up = False
            sar.append(highs[0])
            extreme_point = lows[0]
            current_acceleration = self.acceleration_factor
        
        for i in range(1, len(highs)):
            prev_sar = sar[-1]
            
            # Calculate new SAR
            if trend_up:
                new_sar = prev_sar + current_acceleration * (extreme_point - prev_sar)
                new_sar = min(new_sar, lows[i-1], lows[i]) if i > 1 else new_sar
            else:
                new_sar = prev_sar + current_acceleration * (extreme_point - prev_sar)
                new_sar = max(new_sar, highs[i-1], highs[i]) if i > 1 else new_sar
            
            # Check for trend reversal
            if trend_up:
                if new_sar > lows[i]:
                    # Reverse to downtrend
                    trend_up = False
                    new_sar = extreme_point
                    extreme_point = lows[i]
                    current_acceleration = self.acceleration_factor
                else:
                    if highs[i] > extreme_point:
                        extreme_point = highs[i]
                        current_acceleration = min(current_acceleration + self.acceleration_factor, self.max_acceleration)
            else:
                if new_sar < highs[i]:
                    # Reverse to uptrend
                    trend_up = True
                    new_sar = extreme_point
                    extreme_point = lows[i]
                    current_acceleration = self.acceleration_factor
                else:
                    if lows[i] < extreme_point:
                        extreme_point = lows[i]
                        current_acceleration = min(current_acceleration + self.acceleration_factor, self.max_acceleration)
            
            sar.append(new_sar)
        
        return sar