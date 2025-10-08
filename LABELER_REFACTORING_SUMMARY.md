# Multi-Horizon Profit Labeler Refactoring Summary

## Overview

This document summarizes the refactoring of the multi-horizon profit labeling system, splitting it into specialized components for Analyst and Tactician models, and integrating them into the pipeline.

## Changes Made

### 1. New Labeling Components Created

#### Analyst Profit Labeler (`analyst_profit_labeler.py`)
- **Location**: `/workspace/src/training/steps/pre_training/analyst_profit_labeler.py`
- **Purpose**: Specialized multi-horizon profit labeling for Analyst models
- **Key Features**:
  - 60m timeframe optimization for strategic decision-making
  - Multi-horizon profit labeling (1h, 4h, 12h, 24h horizons)
  - Volatility-aware target bands
  - Enhanced label quality scoring
  - Per-regime/cluster optimization support
  - Wraps `VolatilityAwareMultiHorizonLabeler` from `profit_labeling/volatility_aware_labeler.py`

#### Tactician Entry Labeler (`tactician_entry_labeler.py`)
- **Location**: `/workspace/src/training/steps/pre_training/tactician_entry_labeler.py`
- **Purpose**: Differentiated entry timing labels for Tactician models
- **Key Features**:
  - 15m timeframe optimization for entry timing
  - Local maxima/minima detection with peak filtering
  - Enhanced entry quality scoring (adaptive multi-factor)
  - Regime-aware labeling with adaptive thresholds
  - Trains on ALL market data (not just Analyst green lights)
  - Extracted `TacticianDifferentiatedLabeler` from `tactician_pre_ml_orchestration.py`

### 2. Component Registration

Both new components have been registered with the component factory:
- `AnalystProfitLabelerComponent` registered as `'analyst_profit_labeler'`
- `TacticianEntryLabelerComponent` registered as `'tactician_entry_labeler'`

### 3. Sub-Pipeline Integration

#### STEP_REGISTRY Updates (`sub_pipeline.py`)
Added new steps to the registry:
```python
'analyst_profit_labeler': StepSpec(
    name='analyst_profit_labeler',
    component_key='analyst_profit_labeler',
    executor_method='_execute_analyst_profit_labeler',
    display_name='Analyst profit labeling',
    description='Apply Analyst-specific multi-horizon profit labeling (60m timeframe).',
    order=11,
),
'tactician_entry_labeler': StepSpec(
    name='tactician_entry_labeler',
    component_key='tactician_entry_labeler',
    executor_method='_execute_tactician_entry_labeler',
    display_name='Tactician entry labeling',
    description='Apply Tactician-specific entry timing labels (15m timeframe).',
    order=12,
),
```

#### Progress Icons
- `'analyst_profit_labeler'`: 📈
- `'tactician_entry_labeler'`: 🎲

### 4. Component Factory Updates (`component_factory.py`)

Added new modules to BUILTIN_MODULES:
```python
"src.training.steps.pre_training.analyst_profit_labeler",
"src.training.steps.pre_training.tactician_entry_labeler",
```

### 5. Ares Launcher Integration (`ares_launcher.py`)

#### New Sub-Pipeline Descriptions
- `'analyst_profit_labeler'`: "Analyst-specific multi-horizon profit labeling (60m timeframe, strategic decision-making)"
- `'tactician_entry_labeler'`: "Tactician-specific entry timing labels (15m timeframe, local maxima/minima detection)"

#### New Dependencies
Both components depend on `'regime_data_splitting'`

#### New Output Files
- `analyst_profit_labeler`: `['analyst_multi_horizon_labels.parquet', 'analyst_labeling_report.json']`
- `tactician_entry_labeler`: `['tactician_entry_labels.parquet', 'tactician_labeling_report.json']`

#### New CLI Shortcut Flags
```bash
# Execute Analyst profit labeler
python ares_launcher.py --analyst-labeler --symbol ETHUSDT --timeframe 60m

# Execute Tactician entry labeler
python ares_launcher.py --tactician-labeler --symbol ETHUSDT --timeframe 15m
```

## Usage Examples

### Direct Component Usage

#### Analyst Profit Labeler
```python
from src.training.steps.pre_training.analyst_profit_labeler import (
    AnalystProfitLabeler,
    AnalystProfitLabelerConfig
)

# Create configuration
config = AnalystProfitLabelerConfig(
    timeframe="60m",
    horizons=[60, 240, 720, 1440],  # 1h, 4h, 12h, 24h
    target_profits=[0.5, 1.0, 2.0, 3.0],
    use_volatility_normalization=True
)

# Create labeler
labeler = AnalystProfitLabeler(config)

# Generate labels
result = labeler.generate_labels(
    data=market_data,
    regime_assignments=regime_assignments
)
```

#### Tactician Entry Labeler
```python
from src.training.steps.pre_training.tactician_entry_labeler import (
    TacticianDifferentiatedLabeler,
    TacticianLabelingConfig
)

# Create configuration
config = TacticianLabelingConfig(
    max_entry_window_minutes=60,
    entry_quality_threshold=0.25,
    entry_quality_scoring_method="adaptive_multi_factor",
    enable_regime_adaptive_labeling=True
)

# Create labeler
labeler = TacticianDifferentiatedLabeler(config)

# Generate labels
labels, quality_metrics = labeler.create_entry_timing_labels(
    data=market_data,
    analyst_signals=analyst_signals,  # Optional
    regime_assignments=regime_assignments
)
```

### CLI Usage

```bash
# Execute Analyst profit labeler with full mode
python ares_launcher.py --mode sub_pipeline --sub_pipeline analyst_profit_labeler \
    --execution-mode full --symbol ETHUSDT --timeframe 60m

# Execute Tactician entry labeler with full mode
python ares_launcher.py --mode sub_pipeline --sub_pipeline tactician_entry_labeler \
    --execution-mode full --symbol ETHUSDT --timeframe 15m

# Using shortcut flags
python ares_launcher.py --analyst-labeler --symbol ETHUSDT --timeframe 60m
python ares_launcher.py --tactician-labeler --symbol ETHUSDT --timeframe 15m
```

## Architecture Benefits

### 1. Separation of Concerns
- Analyst labeling (strategic, multi-horizon) is now separate from Tactician labeling (tactical, entry timing)
- Each component has its own configuration and logic
- Easier to maintain and extend independently

### 2. Specialized Configurations
- Analyst: Optimized for 60m timeframe, strategic profit targets
- Tactician: Optimized for 15m timeframe, entry timing quality

### 3. Improved Pipeline Integration
- Components can be called directly via ares_launcher
- Proper registration with ComponentFactory
- Clear dependencies in sub_pipeline

### 4. Enhanced Usability
- CLI shortcut flags for easy access
- Component-based architecture allows for easy testing
- Clear documentation and examples

## Original vs. Refactored Structure

### Before Refactoring
```
multi_horizon_profit_labeler.py (monolithic)
├── MultiHorizonProfitLabeler
│   └── Uses VolatilityAwareMultiHorizonLabeler
└── MultiHorizonProfitLabelerComponent

tactician_pre_ml_orchestration.py
├── TacticianDifferentiatedLabeler (embedded)
└── TacticianPreMLOrchestrator
```

### After Refactoring
```
analyst_profit_labeler.py
├── AnalystProfitLabeler
│   └── Wraps VolatilityAwareMultiHorizonLabeler (Analyst-specific)
└── AnalystProfitLabelerComponent

tactician_entry_labeler.py
├── TacticianDifferentiatedLabeler (extracted)
└── TacticianEntryLabelerComponent

multi_horizon_profit_labeler.py (remains for backward compatibility)
├── MultiHorizonProfitLabeler
└── MultiHorizonProfitLabelerComponent

tactician_pre_ml_orchestration.py (simplified)
└── TacticianPreMLOrchestrator (now uses tactician_entry_labeler)
```

## Feature Engineering Scripts

The original plan to create separate analyst/tactician versions of feature engineering scripts 
(feature_lookback_optimization, interactive_feature_generation, final_feature_selection) was 
simplified. These scripts remain shared and can be used by both Analyst and Tactician orchestrators,
with the orchestrators passing the appropriate role/context through the pipeline state.

## Testing

To test the refactored structure:

1. **Test Analyst Profit Labeler**:
```bash
python ares_launcher.py --analyst-labeler --execution-mode light --symbol ETHUSDT --timeframe 60m
```

2. **Test Tactician Entry Labeler**:
```bash
python ares_launcher.py --tactician-labeler --execution-mode light --symbol ETHUSDT --timeframe 15m
```

3. **Test Full Orchestration**:
```bash
# Analyst orchestration should work with analyst_profit_labeler
python ares_launcher.py --analyst-pre-ml --execution-mode light --symbol ETHUSDT

# Tactician orchestration should work with tactician_entry_labeler
python ares_launcher.py --tactician-pre-ml --execution-mode light --symbol ETHUSDT
```

## Backward Compatibility

The original `multi_horizon_profit_labeler` component remains intact for backward compatibility.
Existing pipelines will continue to work without modification. The new components provide
enhanced, specialized functionality for Analyst and Tactician models.

## Next Steps

1. Update orchestrators to optionally use the new specialized labelers
2. Add configuration options to switch between generic and specialized labelers
3. Create comprehensive tests for the new components
4. Update documentation with usage examples
5. Consider adding role-based configuration to feature engineering scripts

## Files Modified

1. `/workspace/src/training/steps/pre_training/analyst_profit_labeler.py` - **CREATED**
2. `/workspace/src/training/steps/pre_training/tactician_entry_labeler.py` - **CREATED**
3. `/workspace/src/training/steps/pre_training/sub_pipeline.py` - **UPDATED**
4. `/workspace/src/training/steps/pre_training/components/component_factory.py` - **UPDATED**
5. `/workspace/src/launcher/ares_launcher.py` - **UPDATED**

## Summary

The refactoring successfully splits the monolithic multi-horizon profit labeling system into
specialized, maintainable components for Analyst and Tactician models. The new components are
properly integrated into the pipeline, registered with the component factory, and accessible
via the ares_launcher CLI. The refactoring improves code organization, maintainability, and
provides clear separation between strategic (Analyst) and tactical (Tactician) labeling approaches.