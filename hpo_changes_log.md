# Changes Log - HPO Run 2026-01-07

| Script | Function | Change Description |
|--------|----------|--------------------|
| `ares_launcher.py` | `create_cli_parser` | Added `--enable-labeling-hpo` alias for user command compatibility. |
| `ares_launcher.py` | `main` | Cleaned up redundant `enable_labeling_hpo_params` config blocks. |
| `numba_funcs.py` | `aggregate_causal_surprise_scores` | Optimized surprise aggregation with Numba (minutes → seconds). |
| `adaptive_hunter_router.py` | `_map_regimes_to_labels` | Added fallback to ensure "Chaos" regime is always assigned. |
| `causal_targets.py` | `compute_dml_causal_effects` | Fixed `IndexError` in stratified subsampling by correcting raveled indexing. |
| `label_based_layer_2.py` | `_create_irm_environments` | Optimized `price_trend` calculation using Numba `_numba_rolling_slope`. |
| `mtf_feature_generation.py` | `KalmanFilter1D` | Optimized `filter_series` using Numba JIT to eliminate Python loops. |
| `numba_funcs.py` | `_numba_rolling_slope` | Added JIT-compiled linear regression slope for fast rolling calculations. |
| `adaptive_hunter_router.py` | `predict_batch` | Fixed unreachable code for probability column remapping. |
| `adaptive_hunter_router.py` | Multiple | Added `tprint` instrumentation for GMM fitting and batch prediction. |
| `orthogonal_label_generation.py` | `orthogonal_label_generation` | Added `tprint` instrumentation for pruning and main sweep phases. |
| `layer3_specific_features.py` | `generate_layer3_features` | Integrated `regime_label` and probabilities from Layer 2. |
| `layer3_specific_features.py` | `_compute_anchor_and_drift_features` | Removed dead/duplicate GMM code. |
