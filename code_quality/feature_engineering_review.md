# Feature Engineering Module Review

## Assembly Pipeline

- `TransformRouter.fit_transform` returns both train and validation frames, but `AssemblyDAG.assemble` only concatenates the `'train'` slices. As written the pipeline discards ~20% of the rows (`val_features`) so the assembled matrix no longer aligns with the original inputs. 【F:src/feature_engineering/assembly_dag.py†L195-L215】【F:src/feature_engineering/transforms.py†L322-L351】
- Interaction construction fails because the registry checks for column prefixes that never exist. Transform columns are emitted as `t/<parent>/<transform>` (for example `t/p/mom5/ewz`), but the interaction config looks for `t/mom5`, `t/rv_short_3`, etc., so every interaction bails out with "missing fields" warnings and the feature set never expands beyond the transformed parents. The same mismatch prevents regime quantiles from being learned. 【F:src/feature_engineering/transforms.py†L338-L344】【F:src/feature_engineering/interactions.py†L49-L218】
- `AssemblyDAG` catches every exception, emits a warning, and returns empty results. That swallows root causes (e.g., data schema problems) and makes debugging the pipeline extremely hard. Consider logging and re-raising or propagating structured errors. 【F:src/feature_engineering/assembly_dag.py†L288-L299】

## Lookback Selection

- `_evaluate_lookback` ignores the `lookback` argument entirely; it just reuses the existing feature columns. That means the selector cannot differentiate among menu choices and every score collapses to the same calculation. 【F:src/feature_engineering/lookback_selection.py†L122-L171】
- The hysteresis logic stores lists in `self.history[family]` but reads them back as a scalar (`current_selection = self.history.get(...)`). Once a family has history, `current_selection` becomes a list, so the lookup `next((score for l, score in choices if l == current_selection), ...)` never succeeds and hysteresis is effectively disabled. 【F:src/feature_engineering/lookback_selection.py†L181-L216】

## Transforms

- The online EW-Z transformer updates the mean before computing variance, so `delta` uses the already-updated mean. Welford-style updates need the previous mean; otherwise variance is systematically understated and z-scores drift. 【F:src/feature_engineering/transforms.py†L56-L67】

## Disagreement & Ensemble Meta-Features

- The disagreement calculators collapse the entire time axis into scalars (`np.var(...).mean()`, `np.sort(...)`), so the resulting columns are constant across the index and cannot inform per-bar decisions. 【F:src/feature_engineering/disagreement_meta_features.py†L119-L212】
- Binary direction conflict assumes "long" means `prediction > 0.5`, which is incompatible with signed return forecasts. That mislabels most models when predictions are centered near zero. 【F:src/feature_engineering/disagreement_meta_features.py†L153-L172】
- Jensen–Shannon/KL divergences are computed on raw probability arrays without normalising them; if the inputs are logits or unnormalised scores the distances become meaningless. 【F:src/feature_engineering/disagreement_meta_features.py†L326-L354】
- `calculate_disagreement_features_for_ensemble` wraps each prediction in `np.array([value])`, which erases the sample axis (and for arrays introduces an extra dimension). Downstream metrics then work on 1-element vectors instead of the original series. 【F:src/feature_engineering/disagreement_meta_features.py†L437-L456】
- `EnsembleMetaFeatureGenerator.get_base_model_predictions` only ever uses the first row of each model's output (`predict_proba(...)[0]` / `predict(...)[0]`), so disagreement features ignore the rest of the dataset. 【F:src/feature_engineering/ensemble_meta_features.py†L309-L341】

## Miscellaneous

- Calendar session flags (`open30`, `last30`) and several parent/context features are still placeholders that always return zero. They should either be implemented or explicitly flagged so downstream components do not treat them as informative signals. 【F:src/feature_engineering/assembly_dag.py†L47-L111】【F:src/feature_engineering/feature_registry.py†L242-L259】
- Multiple modules import `Path`/`warnings` but do not use them in normal execution paths, suggesting either unfinished work or dead code. For example, `Path` is unused in `assembly_dag.py`. 【F:src/feature_engineering/assembly_dag.py†L13-L25】

These issues should be prioritised before relying on the new feature-engineering stack in production.
