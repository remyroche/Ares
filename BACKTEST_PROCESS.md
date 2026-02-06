# Backtest Process Explanation

The `backtest` mode in `extreme_price_movements/run_pipeline.py` validates the performance of trained models on an **Out-Of-Sample (OOS)** period. It simulates realistic trading conditions, including dynamic risk management (Triple Barrier Method).

## 1. Prerequisites & Initialization
*   **State Verification:** The system checks for `trained_state.pkl` in `artifacts/{run_id}/models/`. This file contains the trained model bundle and optimized risk parameters (`tp_mult`, `sl_mult`).
*   **Timestamp Selection:** It determines the target timestamp (`ts_sig`), usually corresponding to the latest feature generation run.

## 2. Data Preparation
*   **Universe Selection:** Calls `get_training_universe` to filter assets based on volatility and variance criteria (top `M` symbols).
*   **Data Loading:** Loads OHLCV data for the universe from the `PartitionedOHLCVStore`.
*   **Feature Alignment:**
    *   Loads pre-computed feature sets (`feats`).
    *   Computes **Market Features** (`mkt_df`) and **Regime Gates** (`mkt_gates`) on the fly.
    *   Loads or regenerates the **Exhaustion History** (`p_exh_hist`) to track trend exhaustion probabilities.

## 3. The Backtest Loop
The core process (`run_backtest_step`) iterates hourly through the OOS holdout window (default: last 30 days).

### A. Signal Generation (`generate_hourly_signals`)
For each hour `t`:
1.  **Candidate Selection:** Scans lookback offsets (`t, t-4, t-8...`) to find assets meeting extreme criteria (e.g., top/bottom 5% deviations).
2.  **Model Inference:**
    *   **Spike Model:** Detects market regimes (e.g., "Grind" vs. "Spike").
    *   **Alpha Models:** Generates predictions for **Long** and **Short** sides using both **Mean Reversion (MR)** and **Trend Following (TF)** strategies.
    *   **Meta Model:** Combines MR and TF predictions to output a final `score` and determines dominance (`dom` - which strategy leads).
3.  **Filtering & Ranking:**
    *   Separates Long and Short signals.
    *   Ranks by absolute score magnitude.
    *   Selects top `k` candidates (default: 10).
4.  **Risk Injection:** Attaches optimized risk parameters (e.g., specific `tp_mult`/`sl_mult` for "Long MR") to the order.

### B. Execution Simulation (`simulate_trade_hourly`)
For each order, the system simulates the trade outcome:
1.  **Entry:** Assumes entry at the **next hour's open** (`t + 1h`).
2.  **Dynamic Triple Barrier:**
    *   **Barrier Width:** Calculated via `scaled_atr_pct`. It uses the volatility Z-score to expand the barrier during shocks (clamped between 3% and 6%).
    *   **Levels:**
        *   `TP Distance = tp_mult * barrier_pct`
        *   `SL Distance = sl_mult * barrier_pct`
3.  **Path Simulation:**
    *   Iterates through future hourly bars up to `max_hold_hours` (default: 24h).
    *   **Optimistic Execution:** For Longs, checks `High >= TP` *before* `Low <= SL`.
    *   **Time Exit:** If no barrier is hit, exits at the close of the final bar.

## 4. Results
*   **Aggregation:** Collects all trades into a DataFrame.
*   **Metrics:** Calculates Win Rate, Average Return, and Total Return.
*   **Artifacts:** Saves detailed trade logs to `artifacts/{run_id}/backtest_results.csv`.
