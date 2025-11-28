# Specialist Feature Alignment Issue (Analyst Base Unified Training)

## Symptoms

- In unified analyst base training logs (e.g. `logs/unified_20251128_000120.log`), all specialist blocks report zero non-null values after alignment:
  - ML Risk:
    - `non_null_before=105087, non_null_after=0`
    - `⚠️ ML Risk block aligned to training_index is all-NaN. Check risk artifacts and label index overlap.`
  - Liquidity:
    - `non_null_before=3600, non_null_after=0`
    - `⚠️ Liquidity block aligned to training_index is all-NaN. Check liquidity_probs timestamps and values.`
  - Path, SMC, Mean-reversion show the same pattern: successfully loaded, then all-NaN after alignment.
- The unified training step logs a highly suspicious training index range:
  - `🎯 Training index range: 1970-01-01 00:00:00 → 1970-01-01 00:00:00.000001203 (n=1204)`
- Specialist artifacts themselves have *real* market-time DatetimeIndex ranges, e.g.:
  - ML Risk: `2022-09-14 17:59:35.168000 → 2025-09-13 10:32:17.152000`
  - Liquidity: `2025-05-31 03:00:00 → 2025-10-31 23:00:00`
  - Mean-reversion / Breakout / Path / SMC: similar 2022–2025 ranges.

## Impact

- All specialist features are effectively empty (NaN) in the final training matrix, even though their artifacts exist and contain valid values.
- The analyst base model trains only on the base selected features (e.g. 4 engineered columns), without risk, liquidity, SMC, mean-reversion or path specialist signals.
- Diagnostics show `non_null_before > 0` for each specialist artifact, so the issue is **not** missing data but alignment.

## Likely Root Cause

### 1. Training index corruption (epoch microseconds)

- Unified training uses a `training_index` derived from the feature artifact `selected_feature_dataframe_40`.
- In practice, this index is **not** a 15m DatetimeIndex. It appears to be a numeric / RangeIndex that later gets coerced to timestamps via `pd.to_datetime(..., errors="coerce")`.
- For integer indices 0..1203, `pd.to_datetime` interprets them as **nanoseconds since UNIX epoch**, producing timestamps like:
  - `1970-01-01 00:00:00` to `1970-01-01 00:00:00.000001203`
- When all specialist artifacts (with indices in 2022–2025) are reindexed to this 1970-based `training_index`, there is *no overlap*, so after reindex/shift/ffill all values become NaN.

### 2. Where the index should come from

- `selected_feature_dataframe_{size}` is created in `feature_generation_final_feature_selection_step.py`, in `_perform_multi_size_selection`:

```python
# Create dataframe with available features + targets
all_cols_to_include = available_features + target_cols
selected_dataframe = features_df[all_cols_to_include].copy()

feature_sets[f'selected_feature_dataframe_{size}'] = selected_dataframe
```

- and also in `_generate_artifacts`:

```python
selected_data = combined_features_df[all_cols].copy()
artifacts[dataframe_name] = selected_data
```

- In both cases, the index of `selected_dataframe` / `selected_data` is inherited from `features_df` / `combined_features_df`.
- Those upstream dataframes are built from the meta-labeled / feature-generation outputs, where a proper 15m DatetimeIndex (or at least a `timestamp` column) is expected.
- Somewhere *before saving* `selected_feature_dataframe_40`, or *when reloading* it in `unified_models_training_step`, the DatetimeIndex is lost or replaced with a plain RangeIndex.

### 3. Alignment logic in unified training

- Unified training retrieves the feature artifact via:

```python
training_data = self._get_artifact('selected_feature_dataframe_40', 'data')
```

- It then derives a `training_index` from `training_data.index` and passes this to `get_specialist_models_outputs`, which:
  - Loads each specialist artifact (with its own DatetimeIndex).
  - Aligns via `reindex` onto `training_index` (after optional coercion to DatetimeIndex).
- If `training_index` is in 1970 microseconds and the specialist indices are in 2022–2025, the intersection is empty → all-NaN aligned blocks.

## Confirmed Observations

- Specialist artifacts are **present and non-empty** (non_null_before >> 0).
- Unified training always logs `non_null_after=0` for these blocks.
- Training index is logged in 1970 with microseconds and only 1204 unique timestamps.
- `selected_feature_dataframe_40` currently has shape `(1204, 4)` when loaded in unified training.
- `feature_generation_final_feature_selection_step` logs a reasonable time range when creating selected feature dataframes, suggesting it *expected* a DatetimeIndex at that point.

## Plausible Failure Modes

1. **Index reset / loss before saving**
   - At some point in the pre-training pipeline, the DatetimeIndex may have been replaced with a RangeIndex (e.g. via `.reset_index(drop=True)` or concatenation that drops the original index) before being passed into final feature selection.
   - As a result, `combined_features_df.index` (and therefore `selected_feature_dataframe_40.index`) is no longer the true time index.

2. **Index mangling on load**
   - HDF5 / versioned artifact loading may be returning the data with a numeric index, especially if the original was saved without an explicit DatetimeIndex or timestamp column, or if there were prior coercions.
   - Unified training may be coercing a non-datetime index using `pd.to_datetime`, ending up in 1970.

3. **Mixed representation of time**
   - Upstream steps may be using both a `timestamp` column and a DatetimeIndex inconsistently:
     - Some steps treat `timestamp` as a normal column and drop it in the selection stage.
     - Others rely on the index being datetime-like for resampling and alignment.
   - If final selection excludes the `timestamp` column and also loses the DatetimeIndex, there is no canonical time axis left for alignment.

## Direction of Fix

To make specialist blocks usable again:

1. **Preserve a real 15m DatetimeIndex (or explicit `timestamp` column) in `selected_feature_dataframe_{size}`**
   - Ensure that `combined_features_df` entering `_perform_multi_size_selection` and `_generate_artifacts` carries a true DatetimeIndex aligned with all upstream artifacts.
   - If necessary, reconstruct the index from a `timestamp` column before slicing features:
     - e.g. `features_df = features_df.set_index('timestamp')` with proper dtype.

2. **Update unified training to respect that index**
   - When loading `selected_feature_dataframe_40`, use its DatetimeIndex as the canonical `training_index`.
   - Pass this DatetimeIndex unchanged into `get_specialist_models_outputs`.
   - Avoid coercing numeric indices to datetime unless we are explicitly reconstructing from epoch timestamps.

3. **Sanity checks and logging**
   - In both the final selection step and unified training:
     - Log index type and min/max when saving and when loading `selected_feature_dataframe_40`.
     - Assert the index is monotonic and covers the expected 15m date range.
   - In `get_specialist_models_outputs`, log the overlap size between `training_index` and each specialist index, and fail loudly if overlap is zero when artifacts exist.

Once these changes are in place, the expectation is:

- `training_index` reflects actual ETHUSDT/binance 15m timestamps over the training window (e.g. 2021–2025).
- Specialist artifacts (ML Risk, Liquidity, SMC, Mean-reversion, Path, Breakout/Bounce) can be reindexed to that same DatetimeIndex with substantial overlap.
- `non_null_after` in the logs becomes >> 0 for these blocks, and models can finally consume these specialist features.
