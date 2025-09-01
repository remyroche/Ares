# Per-HMM Cluster Logic Maintenance Plan

## Executive Summary

The analysis reveals that **per-HMM cluster logic is partially implemented** in the pipeline but **needs enhancement** in several key steps. While Steps 1-7 have strong regime-specific optimization, Steps 8-16 show inconsistent regime-aware processing.

## Current State Analysis

### ✅ Steps with Strong Per-Regime Logic (Steps 1-7)

1. **Step 1-1.5: Data Collection & Conversion**
   - ✅ Regime-aware data quality checks
   - ✅ Regime-specific column calculation
   - ✅ Auto-fix mechanisms for regime-specific data gaps

2. **Step 2: Feature Engineering**
   - ✅ DiverseLookbackOptimizer with regime-specific periods
   - ✅ MatrixDiverseLookbackOptimizer with regime-specific optimization
   - ✅ Regime-specific lookback period selection

3. **Step 3: HMM Regime Discovery**
   - ✅ Regime-specific parameter optimization
   - ✅ Auto-fix mechanisms calling Step 1/1.5 functions
   - ✅ Regime-specific data quality validation

4. **Step 4: Regime Data Splitting**
   - ✅ Unified dataset with regime labels
   - ✅ Regime-specific statistics calculation
   - ✅ Regime-aware metadata generation

5. **Step 5: Labeling**
   - ✅ Regime-aware triple barrier labeling
   - ✅ Regime-specific profit take/stop loss multipliers
   - ✅ Regime-specific time barriers

6. **Step 6: Feature Engineering**
   - ✅ Regime-aware feature creation
   - ✅ HMM-enhanced features
   - ✅ Regime-specific feature combinations

7. **Step 7: Matrix Operations**
   - ✅ Regime-specific matrix analysis
   - ✅ SR-specific regime analysis
   - ✅ Regime-aware correlation analysis

### ⚠️ Steps with Partial Regime Logic (Steps 8-16)

8. **Step 8: Enhanced HMM-Based Training**
   - ⚠️ **Gap**: No explicit regime-specific training
   - ⚠️ **Gap**: No per-regime model validation
   - ⚠️ **Gap**: No regime-specific hyperparameter optimization

9. **Step 9.5: Multi-Timeframe HMM Ensemble**
   - ⚠️ **Gap**: No regime-specific ensemble creation
   - ⚠️ **Gap**: No per-regime timeframe optimization
   - ⚠️ **Gap**: No regime-specific ensemble weighting

10. **Step 6.5: Unified Regime Intelligence**
    - ⚠️ **Gap**: No regime-specific intelligence generation
    - ⚠️ **Gap**: No per-regime decision logic
    - ⚠️ **Gap**: No regime-specific intelligence validation

11. **Step 12: Analyst Enhancement**
    - ✅ **Good**: Regime-specific model loading structure
    - ⚠️ **Gap**: No regime-specific enhancement logic
    - ⚠️ **Gap**: No per-regime performance optimization

12. **Step 14: Tactician Labeling**
    - ⚠️ **Gap**: No regime-specific barrier calculation
    - ⚠️ **Gap**: No per-regime precision thresholds
    - ⚠️ **Gap**: No regime-specific quality filters

13. **Step 15: Tactician Specialist Training**
    - ⚠️ **Gap**: No regime-specific specialist training
    - ⚠️ **Gap**: No per-regime model specialization
    - ⚠️ **Gap**: No regime-specific training validation

14. **Step 16: Confidence Calibration**
    - ✅ **Good**: Regime-specific validation data loading
    - ✅ **Good**: Per-regime model calibration
    - ⚠️ **Gap**: No regime-specific calibration thresholds

## Critical Gaps Identified

### 1. Step 8: Enhanced HMM-Based Training

**Current Issue**: No explicit regime-specific training logic
```python
# Current implementation lacks regime-specific training
step08_success = await step06_hmm_based_training_enhanced.run_enhanced_step(
    symbol=symbol,
    data_dir=data_dir,
    method_a_mixture_of_experts=method_a_cfg,
    enable_multi_output=enable_multi_output,
)
```

**Recommended Enhancement**:
```python
# Enhanced regime-specific training
async def run_enhanced_regime_specific_step(
    self, symbol: str, data_dir: str,
    method_a_mixture_of_experts: dict, enable_multi_output: bool
) -> bool:
    """Run regime-specific enhanced training."""

    # Load regime data
    regime_data = await self._load_regime_specific_data(symbol, data_dir)

    for regime in regime_data['composite_cluster_id'].unique():
        regime_mask = regime_data['composite_cluster_id'] == regime
        regime_training_data = regime_data[regime_mask]

        # Regime-specific training
        regime_success = await self._train_regime_specific_model(
            regime_training_data, regime, method_a_mixture_of_experts
        )

        if not regime_success:
            self.logger.error(f"❌ Regime {regime} training failed")
            return False

    return True
```

### 2. Step 9.5: Multi-Timeframe HMM Ensemble

**Current Issue**: No regime-specific ensemble creation
```python
# Current implementation lacks regime-specific ensemble logic
step09_5_success = await step09_5_multi_timeframe_hmm_ensemble.run_step(
    symbol=symbol,
    exchange=exchange,
    data_dir=data_dir,
    timeframe=timeframe,
    lookback_days=self.lookback_days,
)
```

**Recommended Enhancement**:
```python
# Enhanced regime-specific ensemble creation
async def run_regime_specific_ensemble_step(
    self, symbol: str, exchange: str, data_dir: str,
    timeframe: str, lookback_days: int
) -> bool:
    """Run regime-specific multi-timeframe ensemble creation."""

    # Load regime-specific data for each timeframe
    timeframes = ["1m", "5m", "15m", "30m"]

    for regime in self._get_regime_clusters():
        regime_ensembles = {}

        for tf in timeframes:
            # Load regime-specific data for this timeframe
            regime_data = await self._load_regime_timeframe_data(
                symbol, exchange, tf, regime, lookback_days
            )

            # Create regime-specific ensemble for this timeframe
            ensemble = await self._create_regime_timeframe_ensemble(
                regime_data, regime, tf
            )

            regime_ensembles[tf] = ensemble

        # Create regime-specific multi-timeframe ensemble
        await self._create_regime_multi_timeframe_ensemble(
            regime_ensembles, regime
        )

    return True
```

### 3. Step 12: Analyst Enhancement

**Current Issue**: Limited regime-specific enhancement logic
```python
# Current implementation has basic regime structure but limited enhancement
if has_regime_specific_structure:
    self.logger.info("🔄 Loading models with regime-specific structure")
    # Load regime-specific models
    for regime_dir in os.listdir(models_dir):
        regime_path = os.path.join(models_dir, regime_dir)
        if os.path.isdir(regime_path):
            regime_models = {}
            # ... load models
            analyst_models[regime_dir] = regime_models
```

**Recommended Enhancement**:
```python
# Enhanced regime-specific analyst enhancement
async def enhance_regime_specific_analysts(
    self, analyst_models: dict, data_dir: str
) -> dict:
    """Enhance analysts with regime-specific logic."""

    enhanced_models = {}

    for regime_name, regime_models in analyst_models.items():
        self.logger.info(f"🔄 Enhancing analysts for regime: {regime_name}")

        # Load regime-specific data
        regime_data = await self._load_regime_specific_data(regime_name, data_dir)

        # Regime-specific enhancement
        enhanced_regime_models = await self._enhance_regime_models(
            regime_models, regime_data, regime_name
        )

        # Regime-specific validation
        validation_results = await self._validate_regime_enhancement(
            enhanced_regime_models, regime_data, regime_name
        )

        enhanced_models[regime_name] = {
            "models": enhanced_regime_models,
            "validation": validation_results,
            "regime_specific_metrics": self._calculate_regime_metrics(regime_data)
        }

    return enhanced_models
```

### 4. Step 14: Tactician Labeling

**Current Issue**: No regime-specific barrier calculation
```python
# Current implementation uses global barrier combinations
self.barrier_combinations = self.barrier_calculator.calculate_dynamic_barriers(
    timeframe="1m"
)
```

**Recommended Enhancement**:
```python
# Enhanced regime-specific tactician labeling
class RegimeAwareTacticianLabeler:
    """Regime-aware tactician labeling with regime-specific barriers."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.regime_barrier_calculator = RegimeSpecificBarrierCalculator(config)

    async def apply_regime_specific_labeling(
        self, data: pd.DataFrame, regime_column: str = "composite_cluster_id"
    ) -> pd.DataFrame:
        """Apply regime-specific tactician labeling."""

        labeled_data = data.copy()

        for regime in data[regime_column].unique():
            regime_mask = data[regime_column] == regime
            regime_data = data[regime_mask]

            # Get regime-specific barriers
            regime_barriers = await self.regime_barrier_calculator.get_regime_barriers(
                regime, regime_data
            )

            # Apply regime-specific labeling
            regime_labeled = await self._apply_regime_barrier_labeling(
                regime_data, regime_barriers, regime
            )

            labeled_data.loc[regime_mask] = regime_labeled

        return labeled_data
```

### 5. Step 15: Tactician Specialist Training

**Current Issue**: No regime-specific specialist training
```python
# Current implementation lacks regime-specific training
step09_success = await step09_tactician_specialist_training.run_step(
    symbol=symbol,
    data_dir=data_dir,
    timeframe="1m",
    exchange=exchange,
)
```

**Recommended Enhancement**:
```python
# Enhanced regime-specific tactician specialist training
async def run_regime_specific_tactician_training(
    self, symbol: str, data_dir: str, timeframe: str, exchange: str
) -> bool:
    """Run regime-specific tactician specialist training."""

    # Load regime-specific data
    regime_data = await self._load_regime_specific_data(symbol, data_dir)

    for regime in regime_data['composite_cluster_id'].unique():
        self.logger.info(f"🎯 Training tactician specialist for regime: {regime}")

        regime_mask = regime_data['composite_cluster_id'] == regime
        regime_training_data = regime_data[regime_mask]

        # Regime-specific specialist training
        specialist_model = await self._train_regime_specialist(
            regime_training_data, regime, timeframe
        )

        # Regime-specific validation
        validation_results = await self._validate_regime_specialist(
            specialist_model, regime_training_data, regime
        )

        # Save regime-specific specialist
        await self._save_regime_specialist(specialist_model, regime, data_dir)

    return True
```

## Implementation Plan

### Phase 1: Critical Enhancements (Steps 8-9.5)

1. **Enhance Step 8: HMM-Based Training**
   - Add regime-specific training logic
   - Implement per-regime model validation
   - Add regime-specific hyperparameter optimization

2. **Enhance Step 9.5: Multi-Timeframe Ensemble**
   - Add regime-specific ensemble creation
   - Implement per-regime timeframe optimization
   - Add regime-specific ensemble weighting

### Phase 2: Analyst & Tactician Enhancement (Steps 12-15)

3. **Enhance Step 12: Analyst Enhancement**
   - Add regime-specific enhancement logic
   - Implement per-regime performance optimization
   - Add regime-specific validation

4. **Enhance Step 14: Tactician Labeling**
   - Add regime-specific barrier calculation
   - Implement per-regime precision thresholds
   - Add regime-specific quality filters

5. **Enhance Step 15: Tactician Specialist Training**
   - Add regime-specific specialist training
   - Implement per-regime model specialization
   - Add regime-specific training validation

### Phase 3: Calibration & Validation (Step 16)

6. **Enhance Step 16: Confidence Calibration**
   - Add regime-specific calibration thresholds
   - Implement per-regime calibration validation
   - Add regime-specific confidence metrics

## Code Templates for Implementation

### 1. Regime-Specific Data Loading Template

```python
async def _load_regime_specific_data(
    self, symbol: str, data_dir: str, regime: str
) -> pd.DataFrame:
    """Load regime-specific data for processing."""

    # Load unified data with regime information
    unified_data = await self._load_unified_data(symbol, data_dir)

    # Filter for specific regime
    regime_mask = unified_data['composite_cluster_id'] == regime
    regime_data = unified_data[regime_mask].copy()

    # Regime-specific data validation
    if len(regime_data) < 100:
        self.logger.warning(f"⚠️ Insufficient data for regime {regime}")
        return pd.DataFrame()

    return regime_data
```

### 2. Regime-Specific Model Training Template

```python
async def _train_regime_specific_model(
    self, regime_data: pd.DataFrame, regime: str, config: dict
) -> Any:
    """Train regime-specific model."""

    self.logger.info(f"🎯 Training model for regime: {regime}")

    # Regime-specific feature engineering
    regime_features = await self._engineer_regime_features(regime_data, regime)

    # Regime-specific hyperparameter optimization
    regime_params = await self._optimize_regime_hyperparameters(
        regime_features, regime
    )

    # Regime-specific model training
    regime_model = await self._train_model_with_regime_params(
        regime_features, regime_params, regime
    )

    # Regime-specific validation
    validation_results = await self._validate_regime_model(
        regime_model, regime_features, regime
    )

    return {
        "model": regime_model,
        "parameters": regime_params,
        "validation": validation_results,
        "regime": regime
    }
```

### 3. Regime-Specific Validation Template

```python
async def _validate_regime_specific_results(
    self, results: dict, regime: str, data_dir: str
) -> bool:
    """Validate regime-specific results."""

    # Load regime-specific validation data
    regime_val_data = await self._load_regime_validation_data(regime, data_dir)

    # Regime-specific quality checks
    quality_checks = await self._perform_regime_quality_checks(
        results, regime_val_data, regime
    )

    # Regime-specific performance validation
    performance_validation = await self._validate_regime_performance(
        results, regime_val_data, regime
    )

    # Log regime-specific results
    self.logger.info(f"📊 Regime {regime} validation results:")
    self.logger.info(f"   Quality checks: {quality_checks}")
    self.logger.info(f"   Performance validation: {performance_validation}")

    return quality_checks and performance_validation
```

## Monitoring and Validation

### 1. Regime-Specific Logging

```python
def _log_regime_specific_metrics(
    self, regime: str, metrics: dict, step_name: str
) -> None:
    """Log regime-specific metrics."""

    self.logger.info(f"📊 {step_name} - Regime {regime} metrics:")
    for metric_name, metric_value in metrics.items():
        self.logger.info(f"   {metric_name}: {metric_value}")
```

### 2. Regime-Specific Validation

```python
async def _validate_regime_specific_step(
    self, step_name: str, regime_results: dict
) -> bool:
    """Validate regime-specific step execution."""

    for regime, results in regime_results.items():
        if not results.get("success", False):
            self.logger.error(f"❌ Regime {regime} failed in {step_name}")
            return False

        # Regime-specific quality validation
        quality_valid = await self._validate_regime_quality(results, regime)
        if not quality_valid:
            self.logger.error(f"❌ Regime {regime} quality validation failed")
            return False

    return True
```

## Conclusion

To maintain per-HMM cluster logic throughout the pipeline, we need to:

1. **Enhance Steps 8-16** with regime-specific processing
2. **Implement regime-specific templates** for consistent implementation
3. **Add regime-specific validation** at each step
4. **Maintain regime-specific logging** for monitoring
5. **Ensure regime-specific data flow** between steps

This will create a fully regime-aware pipeline that maintains the sophisticated per-HMM cluster logic established in Steps 1-7 throughout the entire training process.