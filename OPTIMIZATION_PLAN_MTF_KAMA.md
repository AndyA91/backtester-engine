# Optimization Instructions: MTF KAMA Dual Strategy

**Target**: Full-period optimization (No Out-of-Sample)
**Strategy Base**: `mtf_kama_dual_v1.pine`
**Asset**: EURUSD (5-min execution)

## 1. Strategy Logic (Mapping)
Port the following logic from `mtf_kama_dual_v1.pine` to a new Python script `strategies/mtf_kama_dual_v1.py`:

- **Low Timeframe (5-min)**:
  - Base KAMA: `kama_chart = calc_kama(close, length=kama_len, fast=kama_fast, slow=kama_slow)`
  - Signal: `ta.crossover(close, kama_chart)` (Long) or `ta.crossunder(close, kama_chart)` (Short).
- **Higher Timeframes (1H / 4H)**:
  - Bias calculation: `diff = kama_htf.diff()`
  - Bullish: `diff > 0`
  - Bearish: `diff < 0`
- **Entry Requirements**:
  - Long: `TF1_bull` AND `TF2_bull` AND `5m_cross_up`.
  - Short: `TF1_bear` AND `TF2_bear` AND `5m_cross_dn`.
  - Cooldown: `bars_since_last_trade >= cooldown`.
- **Exit Requirements**:
  - Long Exit: `NOT TF1_bull` OR `NOT TF2_bull` OR `close < kama_chart`.
  - Short Exit: `NOT TF1_bear` OR `NOT TF2_bear` OR `close > kama_chart`.
  - Optional: TP/SL percentages.

## 2. Parameter Grid
Execute an exhaustive grid search for the following parameters:

| Parameter | Values |
|-----------|--------|
| `kama_len` | 10, 14, 21, 30 |
| `kama_fast` | 2, 3, 5 |
| `kama_slow` | 30, 60, 100 |
| `tf1` | 60 (1H), 240 (4H) |
| `tf2` | 240 (4H), 1440 (1D) |
| `cooldown` | 30, 60, 120 (bars) |
| `tp_pct` | 0.0, 0.1, 0.2, 0.3 |
| `sl_pct` | 0.0, 0.1, 0.2, 0.3 |

## 3. Data & Setup
- **Core Engine**: `engine/engine.py` and `engine/data.py`.
- **Indicators**: `indicators/kama.py`.
- **Data Files**: 
  - `data/OANDA_EURUSD, 5.csv`
  - `data/OANDA_EURUSD, 60.csv`
  - `data/OANDA_EURUSD, 240.csv`
- **Timeframe Alignment**: Use `pd.merge_asof` with `direction='backward'` to align HTF KAMA values to the 5-min index.

## 4. Execution Workflow
1.  **Script Creation**: Create `mtf_kama_dual_v1.py`.
2.  **Precompute Caches**: Precalculate all KAMA and Slope values for all lengths and timeframes before starting the grid search.
3.  **Multiprocessing**: Use `concurrent.futures.ProcessPoolExecutor` to parallelize the search across available CPU cores.
4.  **Reporting**:
    - **Optimization Goal**: Identify the combination with the highest **Profit Factor** that also has at least 30 trades.
    - **Output**: Write the top 20 results to `strategies/mtf_kama_dual_results.txt`.
    - **Learderboard Schema**: `| PF | Net Profit | Max DD % | Win Rate % | Trades | Params |`

## 5. Constraints
- **Strictly Optimization Only**: Do not perform any OOS (Out-of-Sample) splits. Use all available data.
- **Matching**: Ensure the trade count matches TradingView's expected behavior (0% margin, commission 0.0043%).
