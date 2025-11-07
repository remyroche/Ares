# Regime Training Pipeline Modifications

## Overview
This document outlines the required modifications to ensure proper artifact chaining between regime training steps and model training steps.

## Data Flow

```
rolling_hmm_clustering
  ↓ (saves) rolling_hmm_regime_probabilities (1h timeframe)

regime_models_training
  ← (loads) rolling_hmm_regime_probabilities
  ↓ (resample to 15m, forward-fill)
  ↓ (train models)
  ↓ (saves) regime_models_predictions_hdf5 (15m timeframe)

regime_ensemble_training
  ← (loads) regime_models_predictions_hdf5
  ← (adds) disagreement features
  ↓ (train ensemble)
  ↓ (saves/updates) regime_ensemble_predictions_hdf5 (15m timeframe)

analyst/tactician training
  ← (loads) regime_ensemble_predictions_hdf5
  ← (loads) feature_generation_final_feature_selection_step features
  ↓ (train models)
```

## 1. Modifications to `regime_models_training`

### Location: `src/training/steps/market_analysis/components/regime_models_training.py`

#### A. Add method to load rolling_hmm artifact and resample to 15m

```python
async def _load_and_resample_regime_probabilities(
    self,
    base_step: BaseStep
) -> Optional[pd.DataFrame]:
    """
    Load rolling_hmm_regime_probabilities and resample to 15m.

    Args:
        base_step: BaseStep instance for artifact loading

    Returns:
        DataFrame with regime probabilities at 15m timeframe
    """
    try:
        tprint("📥 [REGIME_MODELS] Loading rolling_hmm_regime_probabilities artifact", color="cyan")

        # Load 1h regime probabilities
        regime_probs_1h = base_step._get_artifact(
            'rolling_hmm_regime_probabilities',
            artifact_type='data'
        )

        if regime_probs_1h is None:
            tprint("⚠️ [REGIME_MODELS] No rolling_hmm_regime_probabilities found", color="yellow")
            return None

        tprint(f"✅ [REGIME_MODELS] Loaded regime probabilities: {regime_probs_1h.shape}", color="green")
        tprint(f"📊 [REGIME_MODELS] Columns: {list(regime_probs_1h.columns)}", color="blue")

        # Ensure datetime index
        if not isinstance(regime_probs_1h.index, pd.DatetimeIndex):
            regime_probs_1h.index = pd.to_datetime(regime_probs_1h.index)

        # Resample from 1h to 15m using forward-fill
        tprint("🔄 [REGIME_MODELS] Resampling from 1h to 15m (forward-fill)", color="cyan")
        regime_probs_15m = regime_probs_1h.resample('15T').ffill()

        tprint(f"✅ [REGIME_MODELS] Resampled to 15m: {regime_probs_15m.shape}", color="green")

        return regime_probs_15m

    except Exception as e:
        tprint(f"❌ [REGIME_MODELS] Failed to load/resample regime probabilities: {e}", color="red")
        self.logger.error(f"Failed to load/resample regime probabilities: {e}", exc_info=True)
        return None
```

#### B. Add method to save predictions to HDF5 with column management

```python
async def _save_predictions_to_hdf5(
    self,
    predictions: pd.DataFrame,
    base_step: BaseStep,
    artifact_name: str = 'regime_models_predictions'
) -> None:
    """
    Save model predictions to HDF5 file at 15m timeframe.
    Handles column cleanup for disappeared regimes.

    Args:
        predictions: DataFrame with model predictions (columns = regime probabilities)
        base_step: BaseStep instance for artifact saving
        artifact_name: Name for the HDF5 artifact
    """
    try:
        tprint(f"💾 [REGIME_MODELS] Saving predictions to HDF5: {artifact_name}", color="cyan")

        # Ensure datetime index
        if not isinstance(predictions.index, pd.DatetimeIndex):
            predictions.index = pd.to_datetime(predictions.index)

        # Try to load existing HDF5 to compare columns
        try:
            existing_data = base_step._get_artifact(artifact_name, artifact_type='data')

            if existing_data is not None:
                # Compare columns - find disappeared regimes
                existing_cols = set(existing_data.columns)
                new_cols = set(predictions.columns)

                disappeared_cols = existing_cols - new_cols

                if disappeared_cols:
                    tprint(f"🗑️  [REGIME_MODELS] Removing disappeared regime columns: {disappeared_cols}", color="yellow")
                    # Drop disappeared columns
                    existing_data = existing_data.drop(columns=list(disappeared_cols))

                # Merge with existing data (update overlapping, add new)
                merged_data = pd.concat([existing_data, predictions], axis=0)
                merged_data = merged_data[~merged_data.index.duplicated(keep='last')]
                merged_data = merged_data.sort_index()

                predictions = merged_data

                tprint(f"✅ [REGIME_MODELS] Merged with existing data: {predictions.shape}", color="green")

        except Exception as e:
            tprint(f"ℹ️ [REGIME_MODELS] No existing HDF5 found, creating new: {e}", color="blue")

        # Ensure 15m timeframe
        if predictions.index.freq != '15T':
            predictions = predictions.resample('15T').ffill()

        # Save to HDF5
        base_step._save_artifact(
            data=predictions,
            artifact_name=artifact_name,
            artifact_type='data',
            compression='auto',
            metadata={
                'timeframe': '15m',
                'n_regimes': len([c for c in predictions.columns if 'regime' in c.lower()]),
                'columns': list(predictions.columns),
                'shape': predictions.shape,
                'timestamp': datetime.now().isoformat()
            }
        )

        tprint(f"✅ [REGIME_MODELS] Saved predictions to HDF5: {predictions.shape}", color="green")

    except Exception as e:
        tprint(f"❌ [REGIME_MODELS] Failed to save predictions to HDF5: {e}", color="red")
        self.logger.error(f"Failed to save predictions to HDF5: {e}", exc_info=True)
```

#### C. Modify execute method to use new methods

Insert after line 830 (after regime labels extraction):

```python
# Load and resample rolling_hmm regime probabilities as base features
tprint("📥 [REGIME_MODELS] Loading rolling_hmm artifacts", color="cyan")
from src.training.steps.base_step import BaseStep
base_step_inst = BaseStep("regime_models_training_loader")
base_step_inst._current_context = {
    'symbol': self.config.symbol,
    'exchange': self.config.exchange,
    'timeframe': self.config.timeframe,
    'direction': 'long',
    'model': 'regime'
}

regime_probs_15m = await self._load_and_resample_regime_probabilities(base_step_inst)

if regime_probs_15m is not None:
    tprint(f"✅ [REGIME_MODELS] Using rolling_hmm regime probabilities as features: {regime_probs_15m.shape}", color="green")
    # Add regime probabilities to protected_data
    protected_data = protected_data.join(regime_probs_15m, how='left')
    tprint(f"📊 [REGIME_MODELS] Enhanced data shape: {protected_data.shape}", color="blue")
```

Insert after line 883 (after model evaluation):

```python
# Generate predictions on full dataset
tprint("🎯 [REGIME_MODELS] Generating predictions for HDF5 storage", color="cyan")
model_predictions = {}

for model_name, model in trained_models.items():
    try:
        if hasattr(model, 'predict_proba'):
            pred_probs = model.predict_proba(X)
            # Create columns for each regime
            for regime_idx in range(pred_probs.shape[1]):
                col_name = f'{model_name}_regime_{regime_idx}_prob'
                model_predictions[col_name] = pred_probs[:, regime_idx]
        tprint(f"✅ [REGIME_MODELS] Generated predictions for {model_name}", color="green")
    except Exception as e:
        tprint(f"⚠️ [REGIME_MODELS] Failed to generate predictions for {model_name}: {e}", color="yellow")

if model_predictions:
    predictions_df = pd.DataFrame(model_predictions, index=protected_data.index)
    # Save to HDF5
    await self._save_predictions_to_hdf5(predictions_df, base_step_inst, 'regime_models_predictions')
else:
    tprint("⚠️ [REGIME_MODELS] No model predictions generated", color="yellow")
```

## 2. Modifications to `regime_ensemble_training`

### Location: `src/training/steps/market_analysis/components/regime_ensemble_training.py`

#### A. Add method to load regime_models predictions

```python
async def _load_regime_models_predictions(
    self,
    base_step: BaseStep
) -> Optional[pd.DataFrame]:
    """
    Load regime_models_predictions from HDF5.

    Args:
        base_step: BaseStep instance for artifact loading

    Returns:
        DataFrame with regime model predictions
    """
    try:
        tprint("📥 [REGIME_ENSEMBLE] Loading regime_models_predictions", color="cyan")

        predictions = base_step._get_artifact(
            'regime_models_predictions',
            artifact_type='data'
        )

        if predictions is None:
            tprint("⚠️ [REGIME_ENSEMBLE] No regime_models_predictions found", color="yellow")
            return None

        tprint(f"✅ [REGIME_ENSEMBLE] Loaded predictions: {predictions.shape}", color="green")
        tprint(f"📊 [REGIME_ENSEMBLE] Columns: {list(predictions.columns)}", color="blue")

        return predictions

    except Exception as e:
        tprint(f"❌ [REGIME_ENSEMBLE] Failed to load predictions: {e}", color="red")
        self.logger.error(f"Failed to load predictions: {e}", exc_info=True)
        return None
```

#### B. Add method to calculate disagreement features

```python
def _calculate_disagreement_features(
    self,
    predictions: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate disagreement features from base model predictions.

    Args:
        predictions: DataFrame with base model predictions

    Returns:
        DataFrame with disagreement features
    """
    try:
        tprint("🔢 [REGIME_ENSEMBLE] Calculating disagreement features", color="cyan")

        disagreement_features = pd.DataFrame(index=predictions.index)

        # Group by regime (e.g., all *_regime_0_prob columns)
        regime_groups = {}
        for col in predictions.columns:
            # Extract regime number from column name
            if '_regime_' in col and '_prob' in col:
                regime_num = col.split('_regime_')[1].split('_')[0]
                if regime_num not in regime_groups:
                    regime_groups[regime_num] = []
                regime_groups[regime_num].append(col)

        # Calculate disagreement features for each regime
        for regime_num, cols in regime_groups.items():
            if len(cols) < 2:
                continue

            regime_preds = predictions[cols]

            # 1. Standard deviation
            disagreement_features[f'regime_{regime_num}_std'] = regime_preds.std(axis=1)

            # 2. Range
            disagreement_features[f'regime_{regime_num}_range'] = regime_preds.max(axis=1) - regime_preds.min(axis=1)

            # 3. Coefficient of variation
            mean_pred = regime_preds.mean(axis=1)
            disagreement_features[f'regime_{regime_num}_cv'] = disagreement_features[f'regime_{regime_num}_std'] / (mean_pred + 1e-8)

            # 4. Median absolute deviation
            median_pred = regime_preds.median(axis=1)
            mad = (regime_preds.sub(median_pred, axis=0).abs()).median(axis=1)
            disagreement_features[f'regime_{regime_num}_mad'] = mad

        tprint(f"✅ [REGIME_ENSEMBLE] Calculated {len(disagreement_features.columns)} disagreement features", color="green")

        return disagreement_features

    except Exception as e:
        tprint(f"❌ [REGIME_ENSEMBLE] Failed to calculate disagreement features: {e}", color="red")
        self.logger.error(f"Failed to calculate disagreement features: {e}", exc_info=True)
        return pd.DataFrame(index=predictions.index)
```

#### C. Add method to save ensemble predictions to HDF5

```python
async def _save_ensemble_predictions_to_hdf5(
    self,
    predictions: pd.DataFrame,
    base_step: BaseStep,
    artifact_name: str = 'regime_ensemble_predictions'
) -> None:
    """
    Save ensemble predictions to HDF5 file.

    Args:
        predictions: DataFrame with ensemble predictions
        base_step: BaseStep instance for artifact saving
        artifact_name: Name for the HDF5 artifact
    """
    try:
        tprint(f"💾 [REGIME_ENSEMBLE] Saving ensemble predictions to HDF5: {artifact_name}", color="cyan")

        # Ensure datetime index and 15m timeframe
        if not isinstance(predictions.index, pd.DatetimeIndex):
            predictions.index = pd.to_datetime(predictions.index)

        if predictions.index.freq != '15T':
            predictions = predictions.resample('15T').ffill()

        # Save to HDF5
        base_step._save_artifact(
            data=predictions,
            artifact_name=artifact_name,
            artifact_type='data',
            compression='auto',
            metadata={
                'timeframe': '15m',
                'ensemble_type': 'stacker_lgbm_calibrated',
                'n_regimes': len([c for c in predictions.columns if 'regime' in c.lower()]),
                'columns': list(predictions.columns),
                'shape': predictions.shape,
                'timestamp': datetime.now().isoformat()
            }
        )

        tprint(f"✅ [REGIME_ENSEMBLE] Saved ensemble predictions to HDF5: {predictions.shape}", color="green")

    except Exception as e:
        tprint(f"❌ [REGIME_ENSEMBLE] Failed to save ensemble predictions to HDF5: {e}", color="red")
        self.logger.error(f"Failed to save ensemble predictions to HDF5: {e}", exc_info=True)
```

## 3. Modifications to `unified_models_training_step.py`

### Location: `src/training/steps/model_training/unified_models_training_step.py`

#### Modify `_get_additional_model_outputs` method (around line 1367)

Update to load regime ensemble predictions:

```python
# Load regime ensemble predictions
try:
    regime_ensemble_preds = self._get_artifact('regime_ensemble_predictions', 'data')
    if regime_ensemble_preds is not None:
        tprint_info(f"   ↪ Retrieved regime_ensemble_predictions: shape={regime_ensemble_preds.shape}")
        # Resample/reindex to match training data
        if not regime_ensemble_preds.index.equals(training_data.index):
            tprint_warning(f"Aligning 'regime_ensemble_predictions' index to training data")
            regime_ensemble_preds = regime_ensemble_preds.reindex(training_data.index, method='ffill').fillna(method='bfill')
            tprint_info(f"   ↪ Resampled regime_ensemble_predictions -> shape={regime_ensemble_preds.shape}")
        additional_features_list.append(regime_ensemble_preds)
        tprint_success(f"✅ Added regime ensemble predictions: {regime_ensemble_preds.shape}")
    else:
        tprint_warning("⚠️ No regime_ensemble_predictions found")
except Exception as e:
    tprint_warning(f"⚠️ Could not load regime ensemble predictions: {e}")
```

#### Modify `_retrieve_training_data` method (around line 350)

Update to load full feature set from feature_generation:

```python
# Load full feature set from feature_generation_final_feature_selection_step
try:
    tprint_info("Loading full feature set from feature_generation_final_feature_selection_step")
    final_features = self._get_artifact('final_selected_features', 'data')

    if final_features is None:
        # Try alternative artifact names
        final_features = self._get_artifact('feature_generation_final_features', 'data')

    if final_features is not None:
        tprint_info(f"   ↪ Loaded final features: shape={final_features.shape}")
        training_data = final_features
    else:
        tprint_warning("⚠️ Could not load feature_generation features, using pipeline data")

except Exception as e:
    tprint_warning(f"⚠️ Failed to load feature_generation features: {e}")
```

## 4. Testing the Implementation

### Test Sequence

```bash
# 1. Run rolling_hmm_clustering
python src/launcher/ares_launcher.py --rolling-hmm-regime-discovery --symbol ETHUSDT --execution-mode light

# 2. Run regime_models_training (should load rolling_hmm artifact)
python src/launcher/ares_launcher.py --regime-models-training --symbol ETHUSDT --execution-mode light

# 3. Run regime_ensemble_training (should load regime_models predictions)
python src/launcher/ares_launcher.py --regime-ensemble-training --symbol ETHUSDT --execution-mode light

# 4. Run analyst training (should load regime_ensemble predictions + full features)
python src/launcher/ares_launcher.py --train-analyst-base --symbol ETHUSDT --timeframe 15m --direction long

# 5. Verify artifacts
python -c "
from src.training.steps.base_step import BaseStep
base = BaseStep('test')
base._current_context = {'symbol': 'ETHUSDT', 'exchange': 'binance', 'timeframe': '15m', 'direction': 'long', 'model': 'regime'}

# Check regime_models_predictions
regime_models = base._get_artifact('regime_models_predictions', 'data')
print(f'regime_models_predictions: {regime_models.shape if regime_models is not None else None}')

# Check regime_ensemble_predictions
regime_ensemble = base._get_artifact('regime_ensemble_predictions', 'data')
print(f'regime_ensemble_predictions: {regime_ensemble.shape if regime_ensemble is not None else None}')
"
```

## 5. Key Points

1. **Timeframe Management**: All regime artifacts are resampled to 15m (forward-fill) for consistency with model training
2. **Column Cleanup**: When regimes disappear between runs, old columns are automatically removed from HDF5
3. **Disagreement Features**: Ensemble training adds 4 disagreement metrics per regime
4. **Feature Integration**: Model training loads both regime features AND full feature set
5. **Artifact Chaining**: Each step properly loads from previous step and saves for next step

## 6. Expected Artifacts

After full pipeline:
- `rolling_hmm_regime_probabilities` (1h timeframe, from rolling_hmm)
- `regime_models_predictions` (15m timeframe, from regime_models_training)
- `regime_ensemble_predictions` (15m timeframe, from regime_ensemble_training)
- `analyst_base_outputs` (15m, from analyst training)
- `analyst_ensemble_outputs` (15m, from analyst ensemble)
- `tactician_base_outputs` (15m, from tactician training)
- `tactician_ensemble_outputs` (15m, from tactician ensemble)

All stored in versioned HDF5 artifacts through BaseStep._save_artifact().
