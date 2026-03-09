# Ridge Position Sizer Metrics Report

**Run ID:** 20260214_190000

## Direction: LONG

### Bucket: long_tf
- **Race Winner**: `Unknown`
- **Dynamic Threshold (top_k_pct)**: `30.00%`
- **Position Sizing Formula**: `linear`
- **Base Size**: `5.00%`
- **Rank Multiplier**: `10.00%`
- **Squash Function**: `tanh`
- **Squash k**: `1.00`
- **CV Performance (Best Trial)**:
  - **PnL/Day**: 0.00%
  - **Trades/Day**: 0.0000
  - **Sortino**: 0.0000
  - **MaxDD**: 0.00%
  - **WinRate**: 0.00%
  - **Profit Factor**: 0.00
  - **Avg Win/Loss**: 0.00% / 0.00%
  - **Ulcer Index**: 0.0000
  - **Time Under Water**: 0.00%
- **OOF Calibration (Top Rank Diagnostics)**:

### Bucket: long_mr
- **Race Winner**: `ridge+ridge`
- **Dynamic Threshold (top_k_pct)**: `30.00%`
- **Position Sizing Formula**: `concave`
- **Base Size**: `11.00%`
- **Rank Multiplier**: `14.00%`
- **Squash Function**: `tanh`
- **Squash k**: `2.25`
- **CV Performance (Best Trial)**:
  - **PnL/Day**: 0.19%
  - **Trades/Day**: 6.3563
  - **Sortino**: 5.8160
  - **MaxDD**: 0.00%
  - **WinRate**: 83.42%
  - **Profit Factor**: 285.98
  - **Avg Win/Loss**: 0.04% / -0.00%
  - **Ulcer Index**: 0.0004
  - **Time Under Water**: 17.11%
- **Position Sizing (CV Best)**:
  - **Formula**: concave, Squash: tanh, Squash k: 2.25
  - **Base Size**: 11.00%
  - **Rank Multiplier**: 14.00%
  - **Max Position**: 20.00% (capped at 20.00%)
  - **Average Size**: 19.01%
  - **Median Size**: 19.47%
  - **Size Range**: [11.00%, 20.00%]
  - **Std Dev**: 1.21%
  - **Zero Positions**: 0
  - **Max Positions**: 1234
- **OOF Calibration (Top Rank Diagnostics)**:
  - **Top 30%**: PnL/Day: -0.37%, Trades/Day: 6.65, Sortino: -6.2887, MaxDD: 42.32%, WinRate: 43.3%, PF: 0.33, Avg Win/Loss: 0.11% / -0.11%, Ulcer: 25.9138, TUW: 99.90%, N: 984
  - **Top 20%**: PnL/Day: -0.24%, Trades/Day: 5.34, Sortino: -5.2959, MaxDD: 30.15%, WinRate: 49.2%, PF: 0.41, Avg Win/Loss: 0.10% / -0.11%, Ulcer: 18.0412, TUW: 99.87%, N: 790
  - **Top 10%**: PnL/Day: -0.07%, Trades/Day: 3.26, Sortino: -2.7280, MaxDD: 10.16%, WinRate: 58.0%, PF: 0.66, Avg Win/Loss: 0.11% / -0.10%, Ulcer: 5.3704, TUW: 99.79%, N: 483
- **Walk-Forward Validation (Out-of-Sample)**:
  - **PnL/Day**: -0.08%
  - **Trades/Day**: 3.2548
  - **N_selected**: 1710
  - **Sortino**: -3.4269
  - **MaxDD**: 35.07%
  - **WinRate**: 35.91%
  - **Profit Factor**: 0.60
  - **Avg Win/Loss**: 0.10% / -0.10%
  - **Ulcer Index**: 21.4535
  - **Time Under Water**: 99.88%
- **OOS Per-Decile Diagnostics**:
  - **Top 30%**: PnL/Day: -0.64%, Trades/Day: 9.76, Sortino: -7.1387, MaxDD: 96.61%, WinRate: 43.4%, PF: 0.26, Avg Win/Loss: 0.10% / -0.12%, Ulcer: 75.6527, TUW: 99.98%, N: 5129
  - **Top 20%**: PnL/Day: -0.27%, Trades/Day: 6.51, Sortino: -5.4450, MaxDD: 75.88%, WinRate: 49.7%, PF: 0.41, Avg Win/Loss: 0.10% / -0.10%, Ulcer: 52.4854, TUW: 99.97%, N: 3420
  - **Top 10%**: PnL/Day: -0.08%, Trades/Day: 3.25, Sortino: -3.4269, MaxDD: 35.07%, WinRate: 57.7%, PF: 0.60, Avg Win/Loss: 0.10% / -0.10%, Ulcer: 21.4535, TUW: 99.88%, N: 1710
- **Feature Selection (Ridge)**: Kept 10/10 features.
- **Feature Selection (Tree)**: Kept 30 features.
- **Label Stability (Sensitivity Analysis)**:
  - **Selected Policy**: N/A
  - **J_stable**: 0.0000
  - **TP Sweep Result**: N/A

## Direction: SHORT

### Bucket: short_tf
- **Race Winner**: `ridge+ridge`
- **Dynamic Threshold (top_k_pct)**: `30.00%`
- **Position Sizing Formula**: `concave`
- **Base Size**: `12.00%`
- **Rank Multiplier**: `20.00%`
- **Squash Function**: `tanh`
- **Squash k**: `2.25`
- **CV Performance (Best Trial)**:
  - **PnL/Day**: 0.26%
  - **Trades/Day**: 6.1354
  - **Sortino**: 7.9484
  - **MaxDD**: 0.00%
  - **WinRate**: 86.87%
  - **Profit Factor**: 519.34
  - **Avg Win/Loss**: 0.05% / -0.00%
  - **Ulcer Index**: 0.0004
  - **Time Under Water**: 12.69%
- **Position Sizing (CV Best)**:
  - **Formula**: concave, Squash: tanh, Squash k: 2.25
  - **Base Size**: 12.00%
  - **Rank Multiplier**: 20.00%
  - **Max Position**: 20.00% (capped at 20.00%)
  - **Average Size**: 19.86%
  - **Median Size**: 20.00%
  - **Size Range**: [0.00%, 20.00%]
  - **Std Dev**: 1.60%
  - **Zero Positions**: 18
  - **Max Positions**: 2710
- **OOF Calibration (Top Rank Diagnostics)**:
  - **Top 30%**: PnL/Day: -0.27%, Trades/Day: 8.34, Sortino: -6.2846, MaxDD: 33.51%, WinRate: 49.6%, PF: 0.51, Avg Win/Loss: 0.11% / -0.10%, Ulcer: 19.8380, TUW: 99.76%, N: 1231
  - **Top 20%**: PnL/Day: -0.14%, Trades/Day: 6.39, Sortino: -4.3214, MaxDD: 19.38%, WinRate: 53.4%, PF: 0.63, Avg Win/Loss: 0.11% / -0.09%, Ulcer: 10.6951, TUW: 99.68%, N: 943
  - **Top 10%**: PnL/Day: -0.02%, Trades/Day: 3.83, Sortino: -0.9612, MaxDD: 5.46%, WinRate: 59.1%, PF: 0.90, Avg Win/Loss: 0.11% / -0.09%, Ulcer: 2.6067, TUW: 93.98%, N: 565
- **Walk-Forward Validation (Out-of-Sample)**:
  - **PnL/Day**: -0.04%
  - **Trades/Day**: 3.2604
  - **N_selected**: 1714
  - **Sortino**: -2.0086
  - **MaxDD**: 23.98%
  - **WinRate**: 37.51%
  - **Profit Factor**: 0.78
  - **Avg Win/Loss**: 0.13% / -0.10%
  - **Ulcer Index**: 14.7690
  - **Time Under Water**: 99.88%
- **OOS Per-Decile Diagnostics**:
  - **Top 30%**: PnL/Day: -0.33%, Trades/Day: 9.78, Sortino: -5.1093, MaxDD: 82.20%, WinRate: 53.0%, PF: 0.52, Avg Win/Loss: 0.12% / -0.10%, Ulcer: 59.0575, TUW: 99.96%, N: 5140
  - **Top 20%**: PnL/Day: -0.17%, Trades/Day: 6.52, Sortino: -3.9329, MaxDD: 59.83%, WinRate: 57.0%, PF: 0.60, Avg Win/Loss: 0.12% / -0.10%, Ulcer: 40.2775, TUW: 99.97%, N: 3427
  - **Top 10%**: PnL/Day: -0.04%, Trades/Day: 3.26, Sortino: -2.0086, MaxDD: 23.98%, WinRate: 61.3%, PF: 0.78, Avg Win/Loss: 0.13% / -0.10%, Ulcer: 14.7690, TUW: 99.88%, N: 1714
- **Feature Selection (Ridge)**: Kept 10/10 features.
- **Feature Selection (Tree)**: Kept 30 features.
- **Label Stability (Sensitivity Analysis)**:
  - **Selected Policy**: N/A
  - **J_stable**: 0.0000
  - **TP Sweep Result**: N/A

### Bucket: short_mr
- **Race Winner**: `Unknown`
- **Dynamic Threshold (top_k_pct)**: `30.00%`
- **Position Sizing Formula**: `linear`
- **Base Size**: `5.00%`
- **Rank Multiplier**: `10.00%`
- **Squash Function**: `tanh`
- **Squash k**: `1.00`
- **CV Performance (Best Trial)**:
  - **PnL/Day**: 0.00%
  - **Trades/Day**: 0.0000
  - **Sortino**: 0.0000
  - **MaxDD**: 0.00%
  - **WinRate**: 0.00%
  - **Profit Factor**: 0.00
  - **Avg Win/Loss**: 0.00% / 0.00%
  - **Ulcer Index**: 0.0000
  - **Time Under Water**: 0.00%
- **OOF Calibration (Top Rank Diagnostics)**:


---
*Report generated with Bias Mitigation (2-Step CV Gating & 48h Purging + Walk-Forward OOS)*
