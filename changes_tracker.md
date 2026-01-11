# Changes Tracker

## Session: Running Ares Launcher (meta_labeling_hpo_sample_weighted)
**Date:** 2026-01-10

| Time | File Modified | Function/Component | Change Description |
|------|---------------|-------------------|--------------------|
| 14:30 | src/training/steps/labeling/causal_uncertainty_quantification.py | discover_with_uncertainty | Added `gc.collect()` and explicit cleanup of bootstrap data to resolve OOM issues |
| 14:30 | src/training/steps/labeling/label_based_layer_2.py | __init__ | Retained default `discovery_bootstrap_samples=25` (reverted temporary change to 15) |
| 18:55 | src/training/steps/labeling/causal_discovery.py | lingam_orientation | Optimized LiNGAM to use PC skeleton constraints (sparse regression) instead of dense O(N^3) |
| 18:55 | src/training/steps/labeling/causal_surprise_events.py | Multiple | Fixed `sigma_floor` bug (1.0 -> 1e-6) to improve recall for small-scale metrics |
| 19:30 | src/training/steps/labeling/causal_surprise_events.py, src/training/steps/labeling/label_based_layer_2.py | Multiple | Implemented comprehensive Surprise Detector enhancements: Adaptive Thresholding, Event Clustering, Regime Break Retention, Directional Splits, and Volatility/Entropy/Liquidity Normalization. Modified `label_based_layer_2.py` to pass market data for normalization. |
| 20:00 | src/utils/data/klines_parquet.py | update_data | Refactored `update_data` to use partition-based monthly processing instead of loading full history. Fixed `sort_index` crash by normalizing timezone-aware/naive indexes. |
| 20:00 | src/training/steps/data_collection/enhanced_klines_processing_pipeline.py | process_klines_data | Verified Steps 6 & 7 execution flow. Removed temporary debug prints. |
| 22:48 | src/training/steps/labeling/label_based_layer_2.py | __init__, _run_causal_surprise_events, _run_model_race | (User Edits) Reverted bootstrap samples to 25. Added Hunter Mode (Focal Gamma=2.0). Passed `market_data` to surprise aggregation. Replaced LGBM probe with Ridge+Calibration for collinearity handling. |
| 23:23 | src/utils/data/quality/comprehensive_duplicate_analyzer.py | _group_duplicates_by_timestamp | Optimized duplicate grouping from O(N*M) to O(N) using `isin()` and `groupby()`. Fixed major bottleneck in multi-asset resampling for large datasets. |
| 23:35 | src/training/steps/data_collection/enhanced_klines_processing_pipeline.py | _download_data | Implemented gap consolidation (12h threshold) to prevent O(N^2) disk I/O when filling massive fragmentation (e.g., 315K gaps for LINK). |
