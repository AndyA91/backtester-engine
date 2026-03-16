# Telegram Indicators Library

This folder contains a comprehensive library of technical indicators for trading strategy development.

## Structure

```
telegram/
├── indicators/          # Technical indicators
│   ├── __init__.py
│   ├── moving_averages.py
│   ├── momentum.py
│   ├── trend.py
│   ├── volatility.py
│   ├── volume.py
│   └── utils.py
├── strategies/          # Trading strategies
├── backtests/          # Backtesting results
├── data/              # Market data
├── docs/              # Documentation
└── README.md
```

## Available Indicators

### Moving Averages
- **SMA** - Simple Moving Average
- **EMA** - Exponential Moving Average  
- **WMA** - Weighted Moving Average

### Momentum
- **RSI** - Relative Strength Index
- **Stochastic** - Stochastic Oscillator

### Trend
- **MACD** - Moving Average Convergence Divergence
- **ADX** - Average Directional Index
- **ParabolicSAR** - Parabolic Stop and Reverse

### Volatility
- **BollingerBands** - Bollinger Bands
- **ATR** - Average True Range

### Volume
- **OBV** - On-Balance Volume
- **VWAP** - Volume Weighted Average Price

## Usage Example

```python
from telegram.indicators import RSI, BollingerBands

prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
rsi = RSI().calculate(prices)
bbands = BollingerBands().calculate(prices)
```

## Dependencies

- numpy
- typing