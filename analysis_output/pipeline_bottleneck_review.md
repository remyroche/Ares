# Pipeline bottleneck review for meta_labeling_hpo_sample_weighted (ETHUSDT)

## Scope & sources
- Reviewed MTF feature generation and meta-feature selection code paths that can explain the `create_meta_features produced 238 columns` log line and potential stalls.
- Reviewed merge alignment logic (`merge_asof`) in Layer2 label pipeline as requested.
- Checked available logs in this repo (notably `specialist_training_run.log` and `debug_specialist.log`) for errors/warnings that may indicate other bottlenecks or data issues.

## 1) Why 238 columns per candidate despite feature selection
- The `create_meta_features` implementation in `src/training/steps/labeling/mtf_feature_generation.py` constructs a large, fixed set of MTF indicators for multiple windows (default `windows=[10, 20, 50, 100, 150, 200]`). It logs the **pre-filter** column count (`[MTF] create_meta_features produced ... columns prior to Layer2 filtering`). That means the 238-column number is explicitly **before** any downstream selection or pruning happens.【F:src/training/steps/labeling/mtf_feature_generation.py†L804-L864】【F:src/training/steps/labeling/mtf_feature_generation.py†L1551-L1562】
- The feature selection pipeline that reduces features runs later in `weighted_meta_labeling_step.py` (HPO feature set application, De Prado feature selection, and optional MDA/SHAP selection). These steps are applied **after** `build_meta_features_for_model` returns a full feature matrix and after optional Kalman features are appended.【F:src/training/steps/labeling/weighted_meta_labeling_step.py†L1893-L2140】
- Therefore, a log line stating `create_meta_features produced 238 columns` is expected even when feature selection is enabled: it is emitted **before** the selection steps in the weighted pipeline and before any Layer2 pruning logic takes place.【F:src/training/steps/labeling/mtf_feature_generation.py†L1551-L1562】【F:src/training/steps/labeling/weighted_meta_labeling_step.py†L1893-L2140】

### Suggested improvements (non-implementation) to reduce pre-filter column volume
- **Early feature gating by horizon**: skip or simplify feature families (interactions, cross-timeframe ratios, tail-event flags) when horizon ≥ 720m; compute only core trend/volatility blocks and add optional families only when memory headroom allows.【F:src/training/steps/labeling/mtf_feature_generation.py†L1213-L1562】
- **Downsample before MTF generation**: resample the raw series to 1h bars when horizon ≥ 720m and only upsample after selection (or keep 1h for long-horizon labels), reducing the number of rows and rolling window cost.
- **Chunked feature generation**: compute features in chunks (time windows or feature groups) and write to disk (parquet) to avoid holding all features in memory before selection.
- **Streaming selection**: compute feature quality metrics in streaming mode and keep only the top-N as each feature family is produced (online selection), instead of generating the entire feature matrix first.
- **JIT/Numba targeting**: rework the heaviest rolling ops into Numba-compiled kernels, especially for repeated rolling stats across windows (mean/std/ewm/rolling corr), and avoid Python-level loops in per-window feature blocks.【F:src/training/steps/labeling/mtf_feature_generation.py†L816-L833】
- **Feature proxies**: replace expensive features with cheaper proxies when memory/CPU pressure is detected (e.g., use 1-2 volatility windows instead of full window grid; use approximate rolling stats like Welford or exponentially weighted metrics).
- **Precision downcast**: cast intermediate arrays to float32 (or even float16 for bounded signals) immediately after computation, before concatenation, to reduce memory footprint.

## 2) Other reasons the pipeline may have gotten stuck
- **High-cost MTF feature generation**: `create_meta_features` executes dozens of rolling/window operations for every window and includes multiple feature families (momentum, volatility, regime, interactions, cross-timeframe ratios, etc.). With large horizons (720m) and multiple candidates, this quickly becomes CPU- and memory-intensive and can trigger GC pressure (explicit `gc.collect()` runs at the end).【F:src/training/steps/labeling/mtf_feature_generation.py†L799-L864】【F:src/training/steps/labeling/mtf_feature_generation.py†L1480-L1562】
- **Post-generation feature expansion**: the weighted pipeline optionally expands multi-horizon features and cross-features before applying HPO-selected feature sets. This can significantly grow intermediate matrices before any pruning is applied, increasing memory pressure when combined with large horizons and candidate counts.【F:src/training/steps/labeling/weighted_meta_labeling_step.py†L1970-L2140】
- **Merge-alignment overhead**: Layer2 uses `merge_asof` alignment for event features in `_align_features_efficiently`, which sorts indices, performs a time-based join, fills NaNs, and then runs memory optimization. For large feature frames, this path can spike memory use (multiple copies created via `sort_index`, `merge_asof`, and `fillna`). If this runs per candidate/event batch, it can compound memory usage and stall under heavy load.【F:src/training/steps/labeling/label_based_layer_2.py†L10507-L10620】
- **Huber Teacher failures can expand work**: In Layer2, Huber-based pruning and constraints are attempted before model races. If Huber fails, the pipeline falls back to using **all** features (`X_train_final = X_train`), which increases compute/memory in downstream model races and could slow or stall runs under heavy feature loads.【F:src/training/steps/labeling/label_based_layer_2.py†L6210-L6252】

### Optimization ideas for these stalls (non-implementation)
- **Rolling feature reuse**: cache rolling stats per window (or leverage `numba`-accelerated rolling kernels) and reuse across feature families to avoid recomputing rolling mean/std multiple times for the same window.
- **Limit pre-selection expansion**: in weighted pipeline, only expand multi-horizon/cross features for a capped subset of base features, or only for features that survive a coarse pre-filter (variance/correlation).
- **Event alignment batching**: group events into time-batched chunks and align features per chunk to avoid a single massive `merge_asof` allocation.
- **Parallelize per-window blocks**: use joblib or multiprocessing for independent feature families when memory allows, but cap worker count to avoid multiplying memory use.
- **Cache-aware pruning**: if feature caches grow beyond a threshold, force eviction before entering feature expansion/merge steps to reduce peak memory.

## 3) Other errors or warnings in logs
- `specialist_training_run.log` shows a `NotOpenSSLWarning` from `urllib3` (LibreSSL in use) and notes `TA-Lib not available`, which can degrade indicator performance or change code paths to slower pandas-based fallbacks.【F:specialist_training_run.log†L11-L12】【F:specialist_training_run.log†L199-L200】
- `debug_specialist.log` reports repeated missing artifact errors (e.g., `klines_data`, `market_data`, `ohlcv_data` not found), large invalid timestamp counts, and a failure to train due to an empty dataset (`anchor_df (dollar) is empty`). These conditions can destabilize runs, trigger retries, and cause unexpected stalls or early exits in dependent pipelines.【F:debug_specialist.log†L345-L2448】

### Suggested remediations for warnings/errors (non-implementation)
- **OpenSSL warning**: upgrade Python/OpenSSL to a supported version (OpenSSL 1.1.1+) or pin `urllib3<2` for the environment if upgrading is infeasible.【F:specialist_training_run.log†L11-L12】
- **TA-Lib missing**: install TA-Lib or explicitly disable TA-Lib-dependent features to avoid slow fallbacks and mismatched behavior across environments.【F:specialist_training_run.log†L199-L200】
- **Missing artifacts**: add preflight checks for required artifacts (klines/ohlcv/market_data) and fast-fail with a clear error before attempting specialist steps, rather than repeatedly retrying fallbacks.【F:debug_specialist.log†L345-L2448】
- **Invalid timestamps**: validate and sanitize timestamps earlier in ingestion, and log the first N invalid rows to make dataset issues actionable before training starts.【F:debug_specialist.log†L2403-L2448】
- **Empty datasets**: enforce a minimum sample count for each specialist step and skip/abort training when below threshold to avoid wasted compute and confusing errors.【F:debug_specialist.log†L2422-L2448】

## 4) Suggestions to make MTF + 720m horizon more efficient (non-implementation)
- **Reduce intermediate feature volume**: Consider pruning inside `create_meta_features` (e.g., disable interaction/cross-timeframe ratios or restrict `windows`) when the horizon is large, rather than after the fact. This removes high-memory features before they are materialized.【F:src/training/steps/labeling/mtf_feature_generation.py†L804-L864】
- **Tighten row limits**: `MAX_FEATURE_ROWS` already caps feature generation at 20,000 rows; for 720m horizons, reducing this further or scaling it with horizon length would cut memory footprint at the source.【F:src/training/steps/labeling/mtf_feature_generation.py†L839-L864】
- **Delay/limit feature expansion**: In the weighted pipeline, consider limiting multi-horizon and cross-feature expansion before HPO application, or applying HPO selection directly to the base feature set to avoid large temporary matrices.【F:src/training/steps/labeling/weighted_meta_labeling_step.py†L1970-L2140】
- **Batch feature generation**: If candidates are processed in batches, reduce batch size so memory spikes are bounded by batch rather than full candidate set.
- **Use float32 and categorical compression earlier**: Where safe, cast feature matrices to `float32` (or `float16` for low-variance features) before big joins/expansions to cut memory usage by ~50%.

## 5) Unified price layer2 merge_asof memory usage
- `unified_price_layer2.py` does **not** itself use `merge_asof`. The relevant merge path is `_align_features_efficiently` in `label_based_layer_2.py`, which uses `merge_asof` and `fillna` on full feature frames; this is likely the alignment stage most prone to memory pressure for large, dense feature sets.【F:src/training/steps/labeling/unified_price_layer2.py†L1-L220】【F:src/training/steps/labeling/label_based_layer_2.py†L10507-L10620】
- The code performs sorting, a time-based join, NaN fill, and memory optimization. Each step can allocate new arrays; for large frames, this can lead to temporary memory spikes and swapping, especially on constrained systems like M1 laptops.【F:src/training/steps/labeling/label_based_layer_2.py†L10507-L10620】

### Merge alignment optimizations (non-implementation)
- **Pre-sort once**: ensure `X_all` is sorted and cached in sorted order, and short-circuit sorting in `_align_features_efficiently` when already monotonic.
- **Reduce temp copies**: avoid `fillna` on the full frame until after downstream pruning, or fill only the required columns to reduce temporary allocations.
- **Smaller tolerance windows**: tighten `merge_asof` tolerance when features are dense to reduce matched rows and limit alignment size.【F:src/training/steps/labeling/label_based_layer_2.py†L10507-L10620】
- **Use categorical-aware fills**: convert categorical columns before merge or drop them for alignment to avoid coercion overhead post-merge.
- **Batch alignment**: align a subset of features per candidate or per event batch (e.g., two-phase alignment: core features first, add long-tail features only for survivors).

## 6) Huber regression integration review
- Layer2 uses a **Huber Teacher** pipeline to prune features, generate monotonic constraints, and warm-start model races. This occurs in `_run_model_race` and directly influences which features are fed into LGBM/XGB/CatBoost. If Huber succeeds, it prunes feature count; if it fails, the system falls back to all features (higher compute and memory).【F:src/training/steps/labeling/label_based_layer_2.py†L6210-L6252】
- Model factory exposes a `HuberRegressor` option, indicating that Huber is a first-class model type within the shared ML utilities (used across other pipelines).【F:src/utils/ml_common/models/model_factory.py†L2008-L2025】

### Suggested change (non-implementation)
- **Fast-fail on Huber Teacher failure**: instead of falling back to all features, abort the candidate/model race when Huber pruning fails, or fall back to a minimal safe subset (e.g., top-N variance features) to keep memory bounded.【F:src/training/steps/labeling/label_based_layer_2.py†L6210-L6252】

## 7) Potential blind spots / additional risks
- **Feature cache scope**: The weighted pipeline caches static meta-features keyed by configuration. If the cache grows across candidates or runs, it can retain large DataFrames in memory longer than needed (making memory pressure worse under long horizons).【F:src/training/steps/labeling/label_based_pipeline.py†L5169-L5280】
- **Artifact/data availability**: Missing artifacts and invalid timestamps (from `debug_specialist.log`) suggest the pipeline can enter error paths or retry loops that are unrelated to feature generation but still stall overall runs.【F:debug_specialist.log†L345-L2448】
- **Numba availability**: If Numba is unavailable, MTF uses pandas fallback, which is slower and more memory-intensive. Ensuring Numba is installed and used for your target environment can materially change runtime behavior.【F:src/training/steps/labeling/mtf_feature_generation.py†L816-L833】

### Potential fixes (non-implementation)
- **Cache limits**: cap `_static_meta_cache` and feature caches by size/age and evict aggressively before long-horizon runs to avoid memory retention across candidates.【F:src/training/steps/labeling/label_based_pipeline.py†L5169-L5280】
- **Artifact preflight**: assert required artifacts exist and contain data (non-empty, valid timestamps) before starting pipeline steps; abort early with a clear reason to avoid silent stalls.【F:debug_specialist.log†L345-L2448】
- **Numba check**: add a startup check that logs whether Numba is active and warns if disabled for long-horizon runs, optionally forcing reduced windows when Numba is not available.【F:src/training/steps/labeling/mtf_feature_generation.py†L816-L833】

## Commands used (for traceability)
- `rg -n "create_meta_features|meta_features|MTF" src`
- `sed -n '760,980p' src/training/steps/labeling/mtf_feature_generation.py`
- `sed -n '1480,1665p' src/training/steps/labeling/mtf_feature_generation.py`
- `sed -n '1780,2140p' src/training/steps/labeling/weighted_meta_labeling_step.py`
- `sed -n '10480,10640p' src/training/steps/labeling/label_based_layer_2.py`
- `sed -n '6160,6325p' src/training/steps/labeling/label_based_layer_2.py`
- `sed -n '1,240p' src/training/steps/labeling/unified_price_layer2.py`
- `rg -n "ERROR|WARNING" debug_specialist.log`
- `sed -n '1,200p' specialist_training_run.log`
