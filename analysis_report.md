# Analysis Report: Optimization of Labeling Density and Economic Quality

## Executive Summary

This report analyzes the current meta-labeling pipeline to identify opportunities for increasing trade frequency (`trades_per_day`) without degrading the model's predictive power (AUC) or economic edge. The current filtering process involves three stages:
1.  **Triple-Barrier Event Generation**: Defines potential trade opportunities.
2.  **Economic Floor Filtering**: Removes economically trivial returns based on transaction costs.
3.  **Quantile Labeling**: Assigns binary labels (0/1) to the tails of the return distribution, dropping the noisy middle.

Our analysis suggests that **hierarchical optimization of labeling parameters**—specifically `econ_min_return_multiple`, `label_low_q`, and `label_high_q`—can unlock a significant number of "medium-quality" trades that are currently discarded. By relaxing these filters *after* locking structural parameters (like horizon and stop-loss ratios), we can likely increase trade volume while maintaining a positive expectancy.

## Pipeline Analysis: Where are targets dropped?

### 1. Triple-Barrier / TPSL (Structural Definition)
*   **Input**: All primary signals (e.g., consensus signals from technical indicators).
*   **Constraint**: `min_event_spacing` prevents overlapping trades. `horizon`, `profit_threshold`, and `stop_threshold` define the outcome.
*   **Drop Mechanism**: Does not "drop" events per se, but defines the *universe* of resolvable events.
*   **Metric**: `n_raw_events`.

### 2. Volatility Scaling & Economic Floor (Economic Filter)
*   **Function**: `compute_vol_scaled_returns_for_events`
*   **Logic**:
    ```python
    econ_floor = transaction_cost * econ_min_return_multiple
    small_mask = realized_returns.abs() < econ_floor
    vol_scaled[small_mask] = np.nan  # Dropped
    ```
*   **Impact**: Events with returns smaller than a multiple of the transaction cost (default 2.0x) are discarded as "noise".
*   **Optimization Opportunity**: Reducing `econ_min_return_multiple` (e.g., from 2.0 to 1.0 or 1.5) allows smaller wins/losses to enter the dataset. If the meta-model can predict these correctly, PnL increases through volume.

### 3. Quantile Labeling (Signal-to-Noise Filter)
*   **Function**: `create_quantile_labels_from_vol_scaled_returns`
*   **Logic**:
    ```python
    low_val = quantile(low_q)   # e.g., 30th percentile
    high_val = quantile(high_q) # e.g., 80th percentile
    labels[vol_scaled >= high_val] = 1.0
    labels[vol_scaled <= low_val] = 0.0
    # Everything between low_val and high_val is NaN (Dropped)
    ```
*   **Impact**: The "middle" of the distribution is discarded to create a cleaner separation between classes (0 vs 1).
*   **Optimization Opportunity**:
    *   **Widening the band**: Moving `high_q` lower (e.g., 0.80 -> 0.65) and `low_q` higher (e.g., 0.30 -> 0.35) includes more events.
    *   **Trade-off**: Including the middle decreases the distinctness of the classes, likely lowering AUC. However, if the AUC remains above 0.55-0.60, the increased number of trades may result in higher total Edge/PnL.

## Proposed Strategy: Hierarchical HPO "Stage 4"

To safely explore this trade-off without breaking the core event structure found in previous stages, we propose adding a **Stage 4** to the HPO process.

**Stage 1-3 (Existing)**: Optimize structural parameters (Horizon, TPSL ratios, Smoothing) to find the best "shape" of events.

**Stage 4 (New: Labeling Refinement)**:
1.  **Lock Structural Parameters**: Fix `horizon_bars`, `profit_thr_base`, `stop_to_profit_ratio`, etc., to the best values from Stage 3.
2.  **Optimize Filtering Parameters**:
    *   `econ_min_return_multiple`: Range [0.5, 2.5]. Lower values allow smaller trades.
    *   `label_high_q`: Range [0.55, 0.95]. Lower values include more "modest winners".
    *   `label_low_q`: Range [0.05, 0.45]. Higher values include more "modest losers".
    *   `target_clip_high_q`: Range [0.85, 0.99]. Controls outlier influence on regression targets.
3.  **Objective**: Maximize **Edge** (`(Mean_Ret - Cost) * (2*AUC - 1)`). This metric naturally balances quality (AUC/Mean_Ret) and quantity (implied by stability of AUC, though we should explicitly monitor `trades_per_day`).

## Metrics for Monitoring

We will enhance the HPO reporting to track the "funnel" of events:
*   `n_raw_events`: Potential trades from TPSL engine.
*   `n_vol_scaled_events`: Events remaining after economic floor.
*   `n_final_events`: Final labeled examples passed to ML.

This will allow us to see exactly which stage is the bottleneck for trade frequency and tune accordingly.
