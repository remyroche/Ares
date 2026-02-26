# Candidate Selection Thresholds & Criteria

Based on analysis of `compare_candidate_thresholds.py`:

## 1. Selection Metrics
Candidates are selected cross-sectionally based on one of the following metrics, all utilizing a **12-hour** observation window:
*   **Fixed**: Signed 12h Range `((High12h - Low12h) / Low12h * Sign(Ret12h))`.
*   **ATR-Normalized**: Signed 12h Range scaled by robust ATR `(SignedRange12h / atr_robust)`.
*   **ATR-Vol-Weighted**: ATR-normalized 12h Range tilted by volume z-scores (`rvol_z`, `volu_z`).
*   **CUSUM**: Cumulative Sum strength of hourly returns (`ret1h`) with parameters tuned for identifying significant trends.

## 2. Selection Thresholds
*   **Percentile (`pct`)**: The top and bottom **5%**, **6%**, or **7%** of symbols are selected based on the chosen metric.
*   **Minimum Range (`min_range_pct`)**: Candidates must have a 12-hour range of at least **7%** (default). The optimization grid tests 6%, 7%, 8%, and 9%.
*   **Minimum Volatility (`min_vol_zscore`)**: Candidates must have a volatility Z-score of at least **1.5** (hardcoded override in the script, defaulting from 1.6). The grid tests 1.4 to 1.8.
*   **Sign Consistency**: Explicitly disabled (`None`) in this optimization script.

## 3. CUSUM Specifics
*   **H Threshold**: **6.0** (standard deviations).
*   **Z-Gate**: **0.5** (Z-score magnitude required to trigger CUSUM accumulation).

## 4. Tail Filters
*   **Mode**: `vol_or_entropy_top20` (Default in "FULL" configurations).
*   **Criterion**: Candidates falling into the top **20%** of **Volatility** (24h Z-score) or **Entropy** (Spectral or Shannon) distributions are explicitly *excluded* to avoid unpredictable "tail" events.
*   **Threshold Logging**: The actual cutoff values for Volatility and Entropy are computed and logged per run.

## 5. Expansion Window Logic
*   **Standard Expansion**: Uses **binary dilation** to include future timestamps (e.g., `t+2`, `t+4`) for every selected candidate `t`, allowing the model to learn from late entries.
*   **Conditional Expansion (CUSUM)**: Expands only if the future timestamp maintains specific conditions:
    *   **Z-Gate**: `|Z| >= 1.0`
    *   **Sign Consistency**: Direction matches original candidate (> 60%).
    *   **Vol Ratio**: Local volatility is elevated (> 1.2x median).
