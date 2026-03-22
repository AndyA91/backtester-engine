# Renko Strategy Leaderboard

**IS Period:** 2023-01-23 to 2025-09-30  
**Min trades:** 60  
**Strategies tested:** 37  
**Qualified:** 26 | **Unqualified:** 1 | **Failed:** 10

## Qualified (>= 60 trades)

| Rank | Strategy | Instrument | PF | Net | Trades | WR% | MaxDD% | Expectancy | AvgW/L | Best Params |
|------|----------|------------|---:|----:|-------:|----:|-------:|-----------:|-------:|-------------|
| 1 | gj001_brick_count | GBPJPY | 20.868 | 120606.60 | 395 | 71.6 | -0.20 | 305.333 | 8.259 | n_bricks=3, cooldown=30 |
| 2 | btc001_fisher_adx | BTCUSD | 18.672 | 643826106.00 | 963 | 64.3 | -25575.05 | 668562.935 | 10.377 | fisher_period=10, adx_threshold=25, cooldown=5, session_start=0, vol_max=0, psar_gate=False, req_brick_confirm=True |
| 3 | gj007_combined | GBPJPY | 17.841 | 326898.68 | 1218 | 68.1 | -0.19 | 268.390 | 8.372 | n_bricks=4, cooldown=30 |
| 4 | r013_chop_gate | EURUSD | 17.721 | 1355.24 | 811 | 69.3 | -0.16 | 1.671 | 7.851 | n_bricks=5, cooldown=30, chop_max=0, adx_threshold=20, session_start=13, vol_max=1.5 |
| 5 | r001_brick_count | EURUSD | 15.704 | 978.83 | 560 | 66.2 | -0.16 | 1.748 | 8.000 | n_bricks=2, cooldown=30 |
| 6 | r006_alternation | EURUSD | 15.628 | 877.65 | 540 | 66.1 | -0.15 | 1.625 | 8.011 | n_bricks=2, cooldown=30, alt_lookback=8, max_alternations=2 |
| 7 | r005_master | EURUSD | 14.951 | 1379.95 | 831 | 65.3 | -0.22 | 1.661 | 7.930 | n_bricks=2, cooldown=20, mode=momentum, adx_threshold=0, trail_bricks=1 |
| 8 | r002_reversal | EURUSD | 13.975 | 742.10 | 460 | 62.0 | -0.18 | 1.613 | 8.581 | n_bricks=4, cooldown=30 |
| 9 | ea013_adaptive_escgo | EURAUD | 13.973 | 484.80 | 215 | 71.2 | -0.25 | 2.255 | 5.662 | tl_length=50, escgo_lookback=5, escgo_cooldown=8, use_escgo_exit=True, session_start=0 |
| 10 | r007_combined | EURUSD | 13.005 | 2598.96 | 1687 | 63.7 | -0.14 | 1.541 | 7.423 | n_bricks=5, cooldown=30 |
| 11 | ea010_cyberpunk_momentum | EURAUD | 12.470 | 567.49 | 235 | 68.5 | -0.31 | 2.415 | 5.731 | vta_long_min=45, vta_short_max=50, req_no_overbought=True, cooldown=20, session_start=0 |
| 12 | ea005_va_breakout | EURAUD | 12.078 | 412.99 | 174 | 66.1 | -0.24 | 2.374 | 6.197 | pvt_length=10, va_pct=0.7, n_inside=3, cooldown=5, session_start=13 |
| 13 | ea011_v2_auction_champion | EURAUD | 11.941 | 475.83 | 191 | 64.4 | -0.28 | 2.491 | 6.601 | vp_lookback=30, cvd_lookback=3, req_poc_mig=False, req_trendline=True, req_no_exhaust=False, score_threshold=40, cooldown=20, session_start=13 |
| 14 | ea002_gate_sweep | EURAUD | 11.783 | 1163.99 | 469 | 66.3 | -0.23 | 2.482 | 5.986 | n_bricks=2, cooldown=30, session_start=7, raff_gate=False, vp_gate=False, div_gate=True, do_gate=False |
| 15 | ea014_alpha_sniper | EURAUD | 11.283 | 780.26 | 341 | 65.4 | -0.48 | 2.288 | 5.970 | min_stacked=3, vp_lookback=100, min_signals=2, cooldown=10, session_start=0 |
| 16 | ea011_auction_breakout_pro | EURAUD | 10.916 | 1119.55 | 465 | 63.7 | -0.38 | 2.408 | 6.233 | vp_lookback=30, cvd_lookback=3, req_poc_mig=False, cooldown=20, session_start=0 |
| 17 | ea012_napoleon_value | EURAUD | 10.659 | 1262.36 | 549 | 65.8 | -0.34 | 2.299 | 5.551 | nap_buy_thr=4, nap_sell_thr=4, req_ums=False, cooldown=20, session_start=13 |
| 18 | ea015_sto_reversal | EURAUD | 10.600 | 325.53 | 154 | 62.3 | -0.28 | 2.114 | 6.404 | sto_n=8, req_vp=True, cooldown=20, session_start=13 |
| 19 | ea009_institutional_reversal | EURAUD | 10.231 | 222.73 | 92 | 60.9 | -0.23 | 2.421 | 6.577 | tl_length=25, tl_dist_atr=2.0, imb_threshold=200, min_stacked=2, score_threshold=20, cooldown=10, session_start=13 |
| 20 | ea016_mcp_ddl_cross | EURAUD | 9.950 | 401.89 | 168 | 62.5 | -0.32 | 2.392 | 5.970 | use_ddl_gate=True, cooldown=20, session_start=0 |
| 21 | ea018_vp_div_0007 | EURAUD | 9.834 | 1379.66 | 569 | 62.4 | -0.34 | 2.425 | 5.928 | n_bricks=5, cooldown=20, session_start=0, req_vp=True, req_div=True |
| 22 | ea003r_combined_confluence | EURAUD | 9.772 | 1559.41 | 719 | 63.6 | -0.33 | 2.169 | 5.602 | n_bricks=5, cooldown=20, session_start=0, min_confluence=3 |
| 23 | ea004_band_runner | EURAUD | 9.336 | 464.71 | 200 | 59.5 | -0.36 | 2.324 | 6.355 | lookback=3, cooldown=10, session_start=0 |
| 24 | ea003_confluence_master | EURAUD | 9.286 | 2568.87 | 1181 | 61.3 | -0.40 | 2.175 | 5.861 | n_bricks=5, session_start=0, min_confluence=1, strong_confirm=False |
| 25 | ea001_baseline | EURAUD | 9.170 | 3665.02 | 1736 | 61.5 | -0.34 | 2.111 | 5.750 | n_bricks=4, cooldown=30 |
| 26 | ea017_baseline_0007 | EURAUD | 8.854 | 3602.94 | 1598 | 59.3 | -0.42 | 2.255 | 6.071 | n_bricks=5, cooldown=30 |

## Unqualified (< 60 trades)

| Strategy | Instrument | PF | Net | Trades | WR% | MaxDD% |
|----------|------------|---:|----:|-------:|----:|-------:|
| ea006_distance_divergence | EURAUD | 0.000 | 0.00 | 0 | 0.0 | 0.00 |

## Failed

- **r004_candle_adx**: TV export not found: /home/user/backtester-engine/data/OANDA_EURUSD, 5.csv
Place CSV files in the data/ directory.
- **r008_candle_adx**: [Errno 2] No such file or directory: '/home/user/backtester-engine/data/HISTDATA_EURUSD_5m.csv'
- **r009_exit_study**: [Errno 2] No such file or directory: '/home/user/backtester-engine/data/HISTDATA_EURUSD_5m.csv'
- **r010_psar_gate**: [Errno 2] No such file or directory: '/home/user/backtester-engine/data/HISTDATA_EURUSD_5m.csv'
- **r011_0005_optimize**: [Errno 2] No such file or directory: '/home/user/backtester-engine/data/HISTDATA_EURUSD_5m.csv'
- **r012_macd_lc**: [Errno 2] No such file or directory: '/home/user/backtester-engine/data/HISTDATA_EURUSD_5m.csv'
- **gj008_candle_adx**: [Errno 2] No such file or directory: '/home/user/backtester-engine/data/HISTDATA_GBPJPY_5m.csv'
- **gj009_session_adx**: [Errno 2] No such file or directory: '/home/user/backtester-engine/data/HISTDATA_GBPJPY_5m.csv'
- **gj010_macd_lc**: [Errno 2] No such file or directory: '/home/user/backtester-engine/data/HISTDATA_GBPJPY_5m.csv'
- **gj011_sto_tso**: [Errno 2] No such file or directory: '/home/user/backtester-engine/data/HISTDATA_GBPJPY_5m.csv'
