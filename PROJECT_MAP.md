# Project Map: Backtester Engine

This project is a high-fidelity Python backtesting engine designed to match TradingView's (TV) strategy execution logic exactly. It allows for rapid strategy development, optimization, and validation before deploying Pine Script to TradingView.

## 📁 Directory Structure

- [engine/](file:///c:/Users/float/Desktop/VS%20code/backtester-engine/engine): **Core Engine Logic**
    - [engine.py](file:///c:/Users/float/Desktop/VS%20code/backtester-engine/engine/engine.py): Main backtesting execution logic, indicator library, and KPI calculations.
    - [data.py](file:///c:/Users/float/Desktop/VS%20code/backtester-engine/engine/data.py): Data loading utilities (TV CSV exports, Bitstamp API, CCXT exchange fetcher).
- [indicators/](file:///c:/Users/float/Desktop/VS%20code/backtester-engine/indicators): **Indicator Implementations**
    - Contains both Python (`.py`) and Pine Script (`.pine`) implementations of various technical indicators to ensure they match.
- [strategies/](file:///c:/Users/float/Desktop/VS%20code/backtester-engine/strategies): **Trading Strategies & Research**
    - Implementation of specific trading strategies.
    - Test scripts (`*_test.py`), validation scripts (`*_validate.py`), and result files (`*_results.txt`).
    - Pine Script versions of strategies for TV deployment.
- [data/](file:///c:/Users/float/Desktop/VS%20code/backtester-engine/data): **Market Data**
    - CSV files exported from TradingView used for backtesting.
- [tvresults/](file:///c:/Users/float/Desktop/VS%20code/backtester-engine/tvresults): **Baseline Results**
    - Exported results from TradingView (often XLSX) used as the ground truth for engine validation.

## 📖 Key Documentation

- [MEMORY.md](file:///c:/Users/float/Desktop/VS%20code/backtester-engine/MEMORY.md): **The Rulebook (WHAT to do)**
    - Coding standards, mandatory sanitization rules, available indicators, and strategy templates. **Read this first.**
- [BACKTESTING.md](file:///c:/Users/float/Desktop/VS%20code/backtester-engine/BACKTESTING.md): **The Mechanics (HOW it works)**
    - Internal formulas, fill mechanics, order of operations, and TradingView quirk handling.
- [CLAUDE.md](file:///c:/Users/float/Desktop/VS%20code/backtester-engine/CLAUDE.md): Quick-start instructions for AI assistants.

## 🛠️ Core Engine Components

### Configuration (`BacktestConfig`)
Located in `engine/engine.py`. Controls all parameters of the backtest:
- `initial_capital`, `commission_pct`, `slippage_ticks`
- `qty_type` ("percent_of_equity", "cash", "fixed")
- `pyramiding` (max simultaneous positions)
- `process_orders_on_close` (match TV's same-bar fill setting)

### Execution Functions
- `run_backtest(df, config)`: Standard long-only backtesting.
- `run_backtest_long_short(df, config)`: Backtesting supporting both Long and Short positions with reversal logic.

### Data Loading
- `load_tv_export(filename)`: Loads CSVs exported from TV, preserving OHLC and auxiliary columns.
- `fetch_crypto(...)`: Fetches historical data from exchanges via CCXT.

## 🔄 Core Workflow

1.  **Sanitize Pine Script**: Before converting a TV strategy, audit its settings (commission, margin, date range) against the rules in `MEMORY.md`.
2.  **Generate Signals**: Implement a Python function that adds `long_entry`, `long_exit`, `short_entry`, etc., to a pandas DataFrame.
3.  **Run Backtest**: Use `run_backtest_long_short` with a `BacktestConfig` to execute the strategy.
4.  **Validate**: Compare trade counts, net profit, and max drawdown against TradingView's "Strategy Tester" output to ensure a 1:1 match.

## 📊 Available Indicators

The `engine.py` library includes many built-in indicators matching `ta.*` in Pine:
- Moving Averages: `ema`, `sma`, `smma` (RMA), `wma`, `hma`, `ehma`, `thma`
- Oscillators: `rsi`, `macd`, `stoch`
- Channels/Volatility: `atr`, `bbands`, `donchian`, `highest/lowest`
- Specialty: `ichimoku`, `gaussian` (cascaded EMA), `gaussian_npole_iir` (recursive filter)
