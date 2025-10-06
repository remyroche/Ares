# Complete Wiring Implementation Guide

## Overview
This document provides the complete implementation for wiring NAS/TAS models, adding MultiHorizon N-BEATS, ensuring short/long separation, implementing per-regime training for Analyst, and feeding regime model outputs to both pipelines.

## Requirement 1 & 4 & 5: Model Type Integration

### Analyst Model Types (15m timeframe, per-regime)

```python
# File: src/training/steps/models_training/analyst_models_training.py

class AnalystModelType(Enum):
    """Analyst model types."""
    # Base models
    ELASTIC_NET = "ELASTIC_NET"
    RANDOM_FOREST = "RANDOM_FOREST"
    
    # Advanced models
    NAS = "NAS"  # Neural Architecture Search
    TAS = "TAS"  # Tree-based Architecture Search
    MULTISCALE_NBEATS = "MULTISCALE_NBEATS"  # MultiHorizon N-BEATS


@dataclass
class AnalystModelsTrainingConfig:
    """Configuration for Analyst base models training."""
    # Default model types for Analyst (IF we trade - 15m)
    model_types: List[AnalystModelType] = field(default_factory=lambda: [
        AnalystModelType.ELASTIC_NET,
        AnalystModelType.RANDOM_FOREST,
        AnalystModelType.NAS,
        AnalystModelType.TAS,
        AnalystModelType.MULTISCALE_NBEATS
    ])
    
    # Training parameters
    enable_per_regime_training: bool = True  # ANALYST TRAINS PER-REGIME
    per_regime_training_config: Optional[Dict[str, Any]] = None
    
    # NAS/TAS configuration
    enable_nas_tas_training: bool = True
    nas_tas_config: Optional[Dict[str, Any]] = None
    
    # Regime feature integration
    include_regime_features: bool = True  # Include top 3 regime probabilities
    regime_feature_names: List[str] = field(default_factory=lambda: [
        'regime_prob_1', 'regime_prob_2', 'regime_prob_3',
        'regime_1_id', 'regime_2_id', 'regime_3_id',
        'regime_confidence'
    ])
    
    # Short/Long separation
    enable_directional_training: bool = True  # Separate short & long models
    direction_mode: str = "separate"  # "both", "long_only", "short_only", "separate"
```

### Tactician Model Types (5m timeframe, unified)

```python
# File: src/training/steps/models_training/tactician_models_training.py

class TacticianModelType(Enum):
    """Tactician model types."""
    # Base models
    RANDOM_SURVIVAL_FOREST = "RANDOM_SURVIVAL_FOREST"
    XGBOOST = "XGBOOST"
    
    # Advanced models
    NAS = "NAS"  # Neural Architecture Search
    TAS = "TAS"  # Tree-based Architecture Search


@dataclass
class TacticianModelsTrainingConfig:
    """Configuration for Tactician base models training."""
    # Default model types for Tactician (WHEN we trade - 5m)
    model_types: List[TacticianModelType] = field(default_factory=lambda: [
        TacticianModelType.RANDOM_SURVIVAL_FOREST,
        TacticianModelType.XGBOOST,
        TacticianModelType.NAS,
        TacticianModelType.TAS
    ])
    
    # Training parameters
    enable_per_regime_training: bool = False  # TACTICIAN TRAINS UNIFIED (NOT PER-REGIME)
    unified_training_config: Optional[Dict[str, Any]] = None
    
    # NAS/TAS configuration
    enable_nas_tas_training: bool = True
    nas_tas_config: Optional[Dict[str, Any]] = None
    
    # Regime feature integration
    include_regime_features: bool = True  # Include top 3 regime probabilities
    regime_feature_names: List[str] = field(default_factory=lambda: [
        'regime_prob_1', 'regime_prob_2', 'regime_prob_3',
        'regime_1_id', 'regime_2_id', 'regime_3_id',
        'regime_confidence'
    ])
    
    # Analyst features
    include_analyst_features: bool = True  # Include Analyst outputs
    analyst_feature_names: List[str] = field(default_factory=lambda: [
        'analyst_prediction_long', 'analyst_prediction_short',
        'analyst_confidence_long', 'analyst_confidence_short',
        'analyst_ensemble_score'
    ])
    
    # Short/Long separation
    enable_directional_training: bool = True  # Separate short & long models
    direction_mode: str = "separate"  # "both", "long_only", "short_only", "separate"
```

---

## Requirement 2: Short/Long Separation (Already Implemented)

### DirectionMode Configuration

```python
# Both Analyst and Tactician use the same DirectionMode enum
from src.training.steps.models_training.nas_tas.regime_aware_trainer import DirectionMode

# Configuration options:
config.direction_mode = DirectionMode.SEPARATE  # Train separate models for long/short
config.direction_mode = DirectionMode.BOTH      # Train combined model
config.direction_mode = DirectionMode.LONG_ONLY # Train only long models
config.direction_mode = DirectionMode.SHORT_ONLY # Train only short models

# Separate directional features
config.separate_directional_features = True
config.directional_feature_prefixes = {
    'long': 'long_',
    'short': 'short_'
}
config.min_directional_samples = 50
```

**Status**: ✅ ALREADY IMPLEMENTED in `regime_aware_trainer.py`

---

## Requirement 3: Per-Regime vs Unified Training

### Analyst: Per-Regime Training (Enabled)

```python
# File: src/training/steps/models_training/analyst_models_training.py

class AnalystModelsTrainingStep:
    """Analyst models training with PER-REGIME training."""
    
    async def train_analyst_models(
        self,
        training_data: pd.DataFrame,
        feature_columns: List[str],
        target_columns: List[str],
        regime_assignments: pd.DataFrame,  # Regime assignments from market_analysis
        **kwargs
    ) -> AnalystModelsTrainingResult:
        """
        Train Analyst models with PER-REGIME optimization.
        
        Key Differences from Tactician:
        1. Trains separate models for each regime
        2. Uses 15m timeframe data
        3. Trains on ALL data (not filtered)
        4. Optimizes per-regime hyperparameters
        """
        
        # Split data by regime
        regime_data = self._split_by_regime(training_data, regime_assignments)
        
        # Train models for EACH regime separately
        regime_models = {}
        for regime_id, regime_df in regime_data.items():
            tprint(f"📊 Training Analyst models for Regime {regime_id} ({len(regime_df)} samples)")
            
            # Train all model types for this regime
            regime_models[regime_id] = await self._train_regime_models(
                data=regime_df,
                feature_columns=feature_columns,
                target_columns=target_columns,
                regime_id=regime_id
            )
        
        return AnalystModelsTrainingResult(
            models=regime_models,  # Dict[regime_id, Dict[model_type, model]]
            per_regime=True
        )
```

### Tactician: Unified Training (Disabled)

```python
# File: src/training/steps/models_training/tactician_models_training.py

class TacticianModelsTrainingStep:
    """Tactician models training with UNIFIED training (NOT per-regime)."""
    
    async def train_tactician_models(
        self,
        training_data: pd.DataFrame,
        feature_columns: List[str],
        target_columns: List[str],
        **kwargs
    ) -> TacticianModelsTrainingResult:
        """
        Train Tactician models with UNIFIED training.
        
        Key Differences from Analyst:
        1. Trains single unified model (not per-regime)
        2. Uses 5m timeframe data
        3. Trains on FILTERED data (Analyst signals >0.4%)
        4. Uses regime features as inputs (not splits)
        """
        
        # Train unified models on ALL data (no regime splitting)
        tprint(f"📊 Training Tactician models UNIFIED ({len(training_data)} samples)")
        
        # Regime features are included AS INPUTS, not for splitting
        # This allows the model to learn regime relationships itself
        
        # Train all model types on complete dataset
        unified_models = await self._train_unified_models(
            data=training_data,  # Complete dataset
            feature_columns=feature_columns + self.config.regime_feature_names,  # Include regime features
            target_columns=target_columns
        )
        
        return TacticianModelsTrainingResult(
            models=unified_models,  # Dict[model_type, model]
            per_regime=False
        )
```

---

## Requirement 6: Regime Feature Integration

### Loading Regime Model Outputs

```python
# File: src/training/steps/models_training/analyst_pre_ml_orchestration.py
# File: src/training/steps/models_training/tactician_pre_ml_orchestration.py

def _add_regime_features(
    self,
    training_data: pd.DataFrame,
    regime_predictions: pd.DataFrame  # From market_analysis/regime_ensemble_training
) -> pd.DataFrame:
    """
    Add regime model outputs as features (top 3 most likely regimes).
    
    Args:
        training_data: Input training data
        regime_predictions: DataFrame with columns:
            - regime_prob_0, regime_prob_1, ..., regime_prob_7 (probabilities for each regime)
            - regime_id (most likely regime)
            
    Returns:
        DataFrame with added regime features
    """
    # Extract top 3 regime probabilities for each sample
    regime_probs = regime_predictions[[f'regime_prob_{i}' for i in range(8)]]
    
    # Get top 3 regime indices and probabilities for each row
    top_3_regimes = []
    top_3_probs = []
    
    for idx, row in regime_probs.iterrows():
        # Get top 3 regimes by probability
        top_indices = row.argsort()[-3:][::-1]  # Descending order
        top_values = row.iloc[top_indices].values
        
        top_3_regimes.append(top_indices)
        top_3_probs.append(top_values)
    
    # Add as features to training data
    enriched_data = training_data.copy()
    
    # Top 3 regime probabilities
    enriched_data['regime_prob_1'] = [probs[0] for probs in top_3_probs]
    enriched_data['regime_prob_2'] = [probs[1] for probs in top_3_probs]
    enriched_data['regime_prob_3'] = [probs[2] for probs in top_3_probs]
    
    # Top 3 regime IDs
    enriched_data['regime_1_id'] = [regimes[0] for regimes in top_3_regimes]
    enriched_data['regime_2_id'] = [regimes[1] for regimes in top_3_regimes]
    enriched_data['regime_3_id'] = [regimes[2] for regimes in top_3_regimes]
    
    # Confidence score (probability of top regime)
    enriched_data['regime_confidence'] = enriched_data['regime_prob_1']
    
    tprint_info(f"✅ Added 7 regime features (top 3 regimes + confidence)")
    
    return enriched_data
```

### Integration in Pre-ML Orchestration

```python
# Both Analyst and Tactician pre-ML orchestration steps

async def orchestrate(
    self,
    training_data: pd.DataFrame,
    regime_predictions: Optional[pd.DataFrame] = None,  # NEW: regime outputs
    **kwargs
) -> PreMLResult:
    """Execute pre-ML orchestration with regime features."""
    
    # Step 0: Add regime features (if available)
    if regime_predictions is not None:
        tprint_info("📊 Step 0: Adding regime features...")
        training_data = self._add_regime_features(training_data, regime_predictions)
    
    # Step 1: Multi-Horizon Profit Labeling
    tprint_info("📈 Step 1: Multi-Horizon Profit Labeling...")
    horizon_result = await self._execute_multi_horizon_profit_labeler(config)
    
    # Step 2: Feature Lookback Optimization
    tprint_info("⚙️ Step 2: Feature Lookback Optimization...")
    lookback_result = await self._execute_feature_lookback_optimization(config)
    
    # Step 3: PID-Based Feature Generation
    tprint_info("🔧 Step 3: PID-Based Feature Generation...")
    pid_result = await self._execute_pid_based_feature_generation(config)
    
    # Step 4: Final Feature Selection
    # NOTE: Regime features are preserved in selection
    tprint_info("🎯 Step 4: Final Feature Selection...")
    selection_result = await self._execute_final_feature_selection(config)
    
    return result
```

---

## Complete Pipeline Flow

### Market Analysis Stage → Model Training Stage

```python
# File: src/training/steps/model_training/sub_pipeline.py

async def execute_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
    """Execute complete model training pipeline."""
    
    # Load regime predictions from market_analysis stage
    regime_predictions = self._load_regime_predictions()
    # Expected columns:
    # - regime_prob_0, regime_prob_1, ..., regime_prob_7
    # - regime_id (most likely regime)
    # - regime_confidence
    
    # ==================== ANALYST PIPELINE (15m, per-regime) ====================
    if config.train_analyst:
        # Step 1: Analyst Pre-ML (15m + regime features)
        analyst_pre_ml_result = await self.analyst_pre_ml.orchestrate(
            training_data=analyst_data_15m,  # 15m timeframe
            regime_predictions=regime_predictions,  # Add regime features
        )
        
        # Step 2: Analyst Models Training (PER-REGIME)
        analyst_models_result = await self.analyst_training.train_analyst_models(
            training_data=analyst_pre_ml_result.final_features,
            feature_columns=analyst_pre_ml_result.selected_feature_names,
            target_columns=['target_long', 'target_short'],
            regime_assignments=regime_predictions  # For per-regime splitting
        )
        
        # Step 3: Analyst Ensemble Training
        analyst_ensemble_result = await self.analyst_training.train_analyst_ensemble(
            base_models=analyst_models_result.models,
            # ... ensemble training
        )
    
    # ==================== TACTICIAN PIPELINE (5m, unified) ====================
    if config.train_tactician:
        # Get Analyst predictions for filtering
        analyst_predictions = analyst_ensemble_result.predictions
        
        # Step 4: Tactician Pre-ML (5m + regime features + filtered)
        tactician_pre_ml_result = await self.tactician_pre_ml.orchestrate(
            training_data=tactician_data_5m,  # 5m timeframe
            analyst_predictions=analyst_predictions,  # For filtering (>0.4%)
            regime_predictions=regime_predictions,  # Add regime features
        )
        
        # Step 5: Tactician Models Training (UNIFIED - NOT per-regime)
        tactician_models_result = await self.tactician_training.train_tactician_models(
            training_data=tactician_pre_ml_result.final_features,
            feature_columns=tactician_pre_ml_result.selected_feature_names,
            target_columns=['target_long', 'target_short'],
            # NO regime_assignments parameter - unified training
        )
        
        # Step 6: Tactician Ensemble Training
        tactician_ensemble_result = await self.tactician_training.train_tactician_ensemble(
            base_models=tactician_models_result.models,
            # ... ensemble training
        )
```

---

## NAS/TAS Model Training Integration

### Using Training Orchestrator

```python
# File: src/training/steps/models_training/analyst_models_training.py
# File: src/training/steps/models_training/tactician_models_training.py

from src.training.steps.models_training.nas_tas.training_orchestrator import (
    TrainingOrchestrator, OrchestratorConfig, OrchestrationMode
)

class ModelsTrainingStep:
    """Base class for model training with NAS/TAS support."""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize NAS/TAS orchestrator
        if config.enable_nas_tas_training:
            nas_tas_config = OrchestratorConfig(
                mode=OrchestrationMode.TRAINING_ONLY,
                enable_regime_detection=False,  # We already have regime assignments
                enable_model_training=True,
                enable_model_selection=True,
                direction_mode=config.direction_mode,  # "separate" for long/short
                separate_directional_features=config.enable_directional_training
            )
            self.nas_tas_orchestrator = TrainingOrchestrator(nas_tas_config)
        else:
            self.nas_tas_orchestrator = None
    
    async def _train_nas_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Train NAS (Neural Architecture Search) model."""
        if not self.nas_tas_orchestrator:
            raise RuntimeError("NAS/TAS orchestrator not initialized")
        
        # Use orchestrator to train NAS model
        result = await self.nas_tas_orchestrator.orchestrate_async(
            market_data=pd.DataFrame(X),
            target_variable='target',
            feature_columns=list(range(X.shape[1]))
        )
        
        return {
            'model': result.training_result.models_trained.get('NAS'),
            'metrics': result.training_result.regime_performance,
            'training_time': result.execution_time
        }
    
    async def _train_tas_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Train TAS (Tree-based Architecture Search) model."""
        if not self.nas_tas_orchestrator:
            raise RuntimeError("NAS/TAS orchestrator not initialized")
        
        # Use orchestrator to train TAS model
        result = await self.nas_tas_orchestrator.orchestrate_async(
            market_data=pd.DataFrame(X),
            target_variable='target',
            feature_columns=list(range(X.shape[1]))
        )
        
        return {
            'model': result.training_result.models_trained.get('TAS'),
            'metrics': result.training_result.regime_performance,
            'training_time': result.execution_time
        }
```

---

## Summary of Changes

### ✅ Requirement 1: NAS & TAS Wiring
- Import `TrainingOrchestrator` from `nas_tas/`
- Add NAS and TAS to Analyst model types
- Add NAS and TAS to Tactician model types
- Implement `_train_nas_model()` and `_train_tas_model()` methods

### ✅ Requirement 2: Short/Long Separation
- Already implemented via `DirectionMode.SEPARATE`
- Already implemented via `separate_directional_features=True`
- No additional changes needed

### ✅ Requirement 3: Per-Regime vs Unified
- Analyst: `enable_per_regime_training=True` → trains separate models per regime
- Tactician: `enable_per_regime_training=False` → trains unified model
- Analyst uses `regime_assignments` for splitting
- Tactician uses regime features as inputs

### ✅ Requirement 4: MultiHorizon N-BEATS
- Add `MULTISCALE_NBEATS` to Analyst model types
- Import from `src.utils.ml_common.models.multiscale_nbeats`
- Configure for 15m timeframe

### ✅ Requirement 5: RandomSurvivalForest & XGBoost
- Already present in Tactician model types
- No changes needed
- Add NAS & TAS to complete the model set

### ✅ Requirement 6: Regime Features
- Implement `_add_regime_features()` method
- Extract top 3 regime probabilities + IDs
- Add to both Analyst and Tactician pre-ML orchestration
- Preserve regime features through feature selection

---

## Testing Commands

```bash
# Test Analyst pipeline (15m, per-regime)
python src/launcher/ares_launcher.py --mode sub_pipeline \
  --sub_pipeline analyst_models_training \
  --execution-mode full \
  --timeframe 15m \
  --symbol ETHUSDT

# Test Tactician pipeline (5m, unified)
python src/launcher/ares_launcher.py --mode sub_pipeline \
  --sub_pipeline tactician_models_training \
  --execution-mode full \
  --timeframe 5m \
  --symbol ETHUSDT

# Test complete model_training stage
python src/launcher/ares_launcher.py --mode stage \
  --stage model_training \
  --execution-mode full \
  --symbol ETHUSDT
```

---

## Expected Output Structure

```python
# Analyst Models (per-regime)
analyst_models = {
    'regime_0': {
        'TCN': {...},
        'LIGHTGBM': {...},
        'RIDGE': {...},
        'ELASTIC_NET': {...},
        'RANDOM_FOREST': {...},
        'NAS': {...},
        'TAS': {...},
        'MULTISCALE_NBEATS': {...}
    },
    'regime_1': { ... },
    # ... per regime
}

# Tactician Models (unified)
tactician_models = {
    'RANDOM_SURVIVAL_FOREST': {...},
    'XGBOOST': {...},
    'ELASTIC_NET_CV': {...},
    'NAS': {...},
    'TAS': {...}
}
```

---

## Implementation Status

- ✅ Requirements documented
- ✅ Architecture designed
- ✅ Code structure defined
- ⏳ Implementation in progress
- ⏳ Testing pending
- ⏳ Integration validation pending

All requirements are now properly documented and ready for implementation!
