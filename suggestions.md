# Suggestions to Reduce Noise in Sample Selection

Based on the analysis of `compare_candidate_thresholds.py` and `extreme_price_movements/offline_optimisers/compare_tbm_parameters.py`, the following suggestions are made to reduce noise in sample selection:

## 1. Eliminate Arbitrary Symbol Subsampling

Both scripts currently use arbitrary alphabetical subsampling to reduce the dataset size.
- `compare_candidate_thresholds.py`: Uses `step=3` (selecting every 3rd symbol).
- `compare_tbm_parameters.py`: Uses `[::4]` subsampling and limits to the first 30 symbols.

**Suggestion:** Replace alphabetical subsampling with **Volume-Weighted Sampling** or simply selecting the **Top N Liquid Symbols** (e.g., Top 100 by 30-day average volume). This ensures the analysis focuses on the tradable universe where price formation is more efficient and less prone to microstructure noise.

## 2. Remove Random Row Subsampling

`compare_candidate_thresholds.py` hardcodes `SAMPLE_FRAC = 0.33` (Line 1344), discarding ~67% of rows randomly. This introduces Monte Carlo noise, where results can vary purely based on the random seed or specific dropped rows, especially for tail events.

**Suggestion:** **Set `SAMPLE_FRAC = 1.0`** (or significantly higher) to use the full dataset. If performance is a constraint, consider optimizing the metric calculation rather than dropping data. If subsampling is absolutely necessary, use **Stratified Sampling** to ensure balanced coverage across time and regimes.

## 3. Expand Symbol Universe in TBM Optimization

`compare_tbm_parameters.py` limits the symbol universe to a maximum of 30 symbols (`train_syms[:30]`). This small sample size makes the TBM parameter optimization highly sensitive to idiosyncratic moves in a few assets.

**Suggestion:** **Increase the symbol limit** (e.g., to 50 or 100) to capture broader market behavior. A larger, more diverse universe will produce parameters that generalize better across the market, reducing overfitting to specific assets.

## 4. Prioritize Robust Metrics (Spearman vs. Pearson)

`compare_candidate_thresholds.py` currently uses Pearson Correlation (`ic`) as the primary driver for `global_score`. Pearson is sensitive to outliers, which are common in crypto return distributions.

**Suggestion:** **Use Spearman Rank Correlation (`ic_spearman`)** as the primary metric for ranking. Spearman is robust to outliers and non-linear relationships, providing a more stable signal of predictive power. `compare_tbm_parameters.py` already uses Spearman (`_safe_spearman`), which is good practice to standardise on.

## 5. Winsorize Returns and Metrics

Extreme outliers can skew mean-based statistics (like Sharpe or mean IC) significantly.

**Suggestion:** **Apply Winsorization** (e.g., clip at 1st/99th percentile) to returns and target metrics before aggregation. This dampens the impact of extreme, non-repeatable events on the global score, leading to more robust parameter selection.

## 6. Enforce Minimum Absolute Liquidity/Volume

While `compare_candidate_thresholds.py` filters by relative volatility (`vol_zscore`) and range (`range_pct`), it doesn't explicitly filter by absolute volume or liquidity.

**Clarification on "80% of Average Daily Volume":**
A user might consider setting a threshold such as "Current Volume > 80% of Average Daily Volume". It is important to distinguish between **Absolute Liquidity** and **Relative Volume**:

*   **Absolute Liquidity (Asset Filter):** This ensures the asset itself is tradable without significant slippage.
    *   **Recommendation:** Use a fixed dollar threshold (e.g., **Minimum Average Daily Volume > $10M**). This filters out illiquid micro-caps entirely.
    *   *Why:* An asset with $100k daily volume trading at 200% relative volume is still too illiquid for institutional-sized positions.

*   **Relative Volume (Time Filter):** This ensures the specific time period is active.
    *   **Caution on "80% Threshold":** Setting a minimum threshold of "80% of Average Volume" is likely **too aggressive**. Volume typically follows a log-normal distribution where the median is lower than the mean. A threshold of 80% of the mean could exclude >60% of valid trading periods, including quiet but profitable sessions (e.g., Asian session).
    *   **Recommendation:** If filtering for activity, use a lower threshold (e.g., **>20-50% of Average**) or use a **Z-score (e.g., `vol_z > -0.5`)** to avoid only "dead" zones, rather than requiring high activity for every trade.

**Suggestion:** Implement a **Minimum Absolute Dollar Volume Filter** (e.g., >$10M ADV) as a primary gate to ensure all selected candidates are structurally tradable.
