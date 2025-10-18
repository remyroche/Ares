# CMI Complementarity Integration Guide

## Overview

This guide provides comprehensive documentation for the Conditional Mutual Information (CMI) complementarity integration in the Tactician labeler. The CMI complementarity system maximizes feature-target MI while minimizing redundancy with Analyst outputs through adaptive estimators and hardware optimizations.

## Key Features

- **Three-tier CMI estimation**: KSG (high accuracy), GCMI (balanced), Binned (fallback)
- **Adaptive estimator selection**: Based on data characteristics and performance requirements
- **Hardware optimizations**: M1 GPU/CPU optimizations and VectorBT integration
- **Complete separation**: Analyst mode remains completely unchanged
- **Performance monitoring**: Comprehensive benchmarks and auto-fallback mechanisms

## Architecture

### Core Components

1. **CMI Estimators** (`src/training/steps/pre_training/unified_data_driven_pipeline/utils/cmi_estimators.py`)
2. **Analyst Side Information Handler** (`src/training/steps/pre_training/unified_data_driven_pipeline/utils/analyst_side_info.py`)
3. **CMI Complementarity Scorer** (`src/training/steps/pre_training/unified_data_driven_pipeline/utils/cmi_complementarity.py`)

### Integration Points

- **Feature Generation**: Lookback optimization, interaction generation, period optimization
- **Feature Selection**: Upstream prefiltering with CMI complementarity
- **Tactician Labeler**: Analyst side information emission and CMI diagnostics

## Configuration

### Default Configuration

```python
cmi_config = {
    'enable_cmi_complementarity': True,  # Master switch (Tactician only)
    'estimator_tier': 'adaptive',  # 'ksg', 'gcmi', 'binned', 'adaptive'
    'alpha_candidates': [0.3, 0.5, 0.7],  # Redundancy penalty weights (CV-tuned)
    'cv_folds': 5,  # Purged K-fold splits
    'embargo_windows': 1,  # Time embargo for CV (1-3 windows)
    'per_family_budget': (5, 15),  # Min/max features per family
    'upstream_multiplier': 3,  # Total budget to RFE = 3× per-family
    'noise_floor_permutations': 150,  # Label shuffles for noise floor
    'delta_perf_permutations': 25,  # Null permutations for ΔPerf threshold
    'noise_floor_percentile': 95,  # Threshold percentile
    'weak_A_threshold': 0.005,  # AUC-equivalent; degrade to unconditional MI
    'interaction_gain_percentile': 75,  # Accept if > 75th percentile of null
    'ksg_neighbors': 5,  # k for KSG estimator
    'gcmi_bins': 10,  # Bins for GCMI
    'binned_quantiles': 10,  # Quantiles for fallback
    'min_samples_per_bin': 100,  # For isotonic calibration
    'max_A_dims': 2,  # Reduce A to ≤2 dims for CMI efficiency
    'compute_timeout_seconds': 300,  # 5 min hard limit per stage
}
```

### Adaptive Estimator Selection

The system automatically selects the appropriate estimator based on data characteristics:

```python
def select_estimator(n_features, n_rows, stage):
    """Select estimator based on data characteristics."""
    if n_features > 800 or n_rows < 1500:
        if stage == 'prefilter':
            return 'binned'
        elif stage == 'shortlist':  # Top 3×k
            return 'gcmi'
        else:  # Final k
            return 'ksg'
    elif n_features <= 600 and n_rows >= 2000:
        if stage == 'prefilter':
            return 'gcmi'
        else:  # Final
            return 'ksg'
    else:
        return 'gcmi'  # Balanced default
```

## Usage Examples

### Basic Usage

```python
from src.training.steps.pre_training.unified_data_driven_pipeline.utils.cmi_complementarity import CMIComplementarityScorer
from src.training.steps.pre_training.unified_data_driven_pipeline.utils.analyst_side_info import AnalystSideInfoHandler

# Initialize CMI complementarity scorer
cmi_scorer = CMIComplementarityScorer()

# Initialize Analyst side information handler
analyst_handler = AnalystSideInfoHandler()

# Extract Analyst side information
analyst_result = analyst_handler.extract_side_info(
    pipeline_state, targets, data_index
)

# Score features with CMI complementarity
result = cmi_scorer.score_features(
    X_df, y_series, analyst_result.A,
    family_tags=family_tags,
    cv_splits=cv_splits,
    pipeline_state=pipeline_state
)

# Use selected features
selected_features = result.selected_features
```

### Feature Generation Integration

```python
# In feature generation step
if pipeline_state.get('tactician_mode', False):
    # Apply CMI complementarity filtering
    analyst_result = self.analyst_handler.extract_side_info(
        pipeline_state, targets, generated_features_df.index
    )
    
    if analyst_result.is_valid:
        cmi_result = self.cmi_scorer.score_features(
            generated_features_df, targets, analyst_result.A,
            pipeline_state=pipeline_state
        )
        
        if cmi_result.is_valid:
            generated_features_df = generated_features_df[cmi_result.selected_features]
```

### Feature Selection Integration

```python
# In feature selection step
if pipeline_state.get('tactician_mode', False):
    # Apply CMI prefiltering
    prefilter_mask = self.prefilter_by_cmi(
        X, y, analyst_side_info, family_tags, cv_splits, pipeline_state
    )
    
    # Apply prefilter mask
    if prefilter_mask is not None:
        features = features.loc[:, prefilter_mask]
```

### Tactician Labeler Integration

```python
# In tactician_entry_labeler.py
if self.config.enable_cmi_complementarity:
    # Emit Analyst side information
    analyst_side_info_result = self.emit_analyst_side_info(
        pipeline_state, targets=None, data_index=data.index
    )
    
    if analyst_side_info_result.get('cmi_enabled', False):
        # Store in pipeline state for downstream use
        pipeline_state['analyst_side_info'] = analyst_side_info_result
        # CRITICAL: Set tactician_mode flag
        pipeline_state['tactician_mode'] = True
```

## Hardware Optimizations

### M1 Chip Optimizations

The system automatically detects and uses M1-specific optimizations:

```python
# GPU optimizations
if HARDWARE_OPTIMIZATIONS_AVAILABLE:
    self.gpu_optimizer = M1GPUOptimizer()
    self.memory_optimizer = M1MemoryOptimizer()
    self.cpu_optimizer = M1CPUOptimizer()
```

### VectorBT Integration

For efficient rolling computations:

```python
# VectorBT optimizations
if VECTORBT_OPTIMIZATIONS_AVAILABLE:
    self.vectorbt_optimizer = VectorBTRollingOptimizer()
    self.vectorization_manager = UnifiedVectorizationManager()
```

### ML Utilities

For cross-validation and data leakage detection:

```python
# ML utilities
if ML_UTILITIES_AVAILABLE:
    self.purged_kfold = PurgedKFold
    self.data_leakage_detector = DataLeakageDetector()
    self.lookahead_validator = LookaheadValidator()
    self.bayesian_optimizer = BayesianTPEOptimizer()
```

## Performance Monitoring

### Time per Feature Dashboard

```python
# Monitor time per feature across different estimators
time_per_feature = {
    'ksg': 0.01,      # 10ms per feature
    'gcmi': 0.005,    # 5ms per feature
    'binned': 0.001   # 1ms per feature
}
```

### Memory Usage Dashboard

```python
# Monitor memory usage
memory_usage = {
    'base_memory': 100,    # MB
    'cmi_overhead': 20,    # MB (20% overhead)
    'total_memory': 120    # MB
}
```

### Estimator Breakdown Dashboard

```python
# Monitor estimator usage
estimator_breakdown = {
    'ksg': 0.2,      # 20% of computations
    'gcmi': 0.6,     # 60% of computations
    'binned': 0.2    # 20% of computations
}
```

## Auto-Fallback Mechanisms

### Timeout Fallback

```python
# Automatic fallback when timeout is exceeded
if computation_time > timeout_threshold:
    fallback_estimator = 'binned'
    tprint_warning(f"⚠️ Timeout fallback: {fallback_estimator}")
```

### Memory Fallback

```python
# Automatic fallback when memory usage is high
if memory_usage > memory_threshold:
    fallback_estimator = 'binned'
    tprint_warning(f"⚠️ Memory fallback: {fallback_estimator}")
```

### Accuracy Fallback

```python
# Automatic fallback when accuracy is low
if n_samples < accuracy_threshold:
    fallback_estimator = 'binned'
    tprint_warning(f"⚠️ Accuracy fallback: {fallback_estimator}")
```

## Critical Separation Requirements

### Analyst Mode Protection

**CRITICAL**: All CMI modifications are gated on `tactician_mode=True`. Analyst mode remains completely unchanged.

```python
# In all CMI integration points
if not pipeline_state.get('tactician_mode', False):
    # Skip all CMI logic, use standard behavior
    return standard_result

# Proceed with CMI-enhanced behavior only if tactician_mode=True
```

### Mode Detection

```python
# Check mode in pipeline state
if pipeline_state.get('tactician_mode', False):
    tprint_info("🔧 Tactician mode detected - CMI complementarity enabled")
else:
    tprint_info("🔧 Analyst mode detected - CMI complementarity disabled")
```

## Testing

### Unit Tests

```bash
# Run CMI estimator tests
pytest tests/training/test_cmi_estimators.py -v

# Run CMI complementarity tests
pytest tests/training/test_cmi_complementarity.py -v
```

### Integration Tests

```bash
# Run integration tests
pytest tests/training/test_cmi_complementarity.py -v

# Run Analyst mode protection tests
pytest tests/training/test_analyst_mode_protection.py -v
```

### Performance Tests

```bash
# Run performance benchmarks
pytest tests/training/test_cmi_performance_benchmarks.py -v
```

## Diagnostics and Monitoring

### CMI Diagnostics

The system provides comprehensive diagnostics:

```python
cmi_diagnostics = {
    'cmi_enabled': True,
    'original_features': 100,
    'filtered_features': 50,
    'noise_floor': 0.001,
    'delta_perf_threshold': 0.002,
    'analyst_source': 'oof_confidence',
    'analyst_dims': 1,
    'I_Y_A': 0.05,
    'degraded_to_unconditional': False
}
```

### Performance Metrics

```python
performance_metrics = {
    'computation_time': 2.5,
    'features_processed': 100,
    'folds_processed': 5,
    'memory_usage': 120.5,
    'cache_hits': 75,
    'timeout_events': 0,
    'estimator_breakdown': {'ksg': 0.2, 'gcmi': 0.6, 'binned': 0.2},
    'success_rate': 0.95
}
```

## Troubleshooting

### Common Issues

1. **CMI not activating**: Check that `tactician_mode=True` in pipeline state
2. **Performance issues**: Check auto-fallback mechanisms and estimator selection
3. **Memory issues**: Monitor memory usage and enable memory fallback
4. **Timeout issues**: Check timeout thresholds and enable timeout fallback

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger('cmi_complementarity').setLevel(logging.DEBUG)
```

### Performance Profiling

```python
# Profile CMI computation
import cProfile
cProfile.run('cmi_scorer.score_features(X, y, A)')
```

## Best Practices

1. **Always check mode**: Verify `tactician_mode` before applying CMI
2. **Monitor performance**: Use performance dashboards to track efficiency
3. **Use auto-fallback**: Enable automatic fallback mechanisms for robustness
4. **Test thoroughly**: Run comprehensive tests including Analyst mode protection
5. **Monitor diagnostics**: Use CMI diagnostics to understand system behavior

## Conclusion

The CMI complementarity integration provides a powerful and efficient system for maximizing feature-target MI while minimizing redundancy with Analyst outputs. The system is designed with complete separation between Analyst and Tactician modes, ensuring that Analyst mode remains completely unchanged while providing enhanced capabilities for Tactician mode.

The comprehensive testing suite, performance monitoring, and auto-fallback mechanisms ensure robust operation across different data sizes and scenarios, while the hardware optimizations provide efficient computation on M1 chips.
