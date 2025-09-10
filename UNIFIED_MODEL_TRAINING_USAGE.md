# Usage Examples for New Unified Model Training System

## 1. Basic Usage

```python
from src.training.steps.consolidated_model_training import (
    ConsolidatedHMMBasedTraining,
    ConsolidatedAnalystEnhancement,
    ConsolidatedTacticianSpecialistTraining,
    ConsolidatedUnifiedRegimeIntelligence
)

# Create Analyst
analyst = ConsolidatedAnalystEnhancement(config)
analyst_result = await analyst.execute(features, targets)

# Create Tactician
tactician = ConsolidatedTacticianSpecialistTraining(config)
tactician_result = await tactician.execute(features, targets)

# Create HMM-based model
hmm_model = ConsolidatedHMMBasedTraining(config)
hmm_result = await hmm_model.execute(features, targets)

# Create unified regime intelligence
regime_intel = ConsolidatedUnifiedRegimeIntelligence(config)
regime_result = await regime_intel.execute(features, targets)
```

## 2. Through Unified Model Training

```python
from src.training.steps.unified_model_training import comprehensive_model_training

# Create Analyst
analyst_result = await comprehensive_model_training(
    config, 
    pipeline_state, 
    model_name='analyst_enhancement_model'
)

# Create Tactician
tactician_result = await comprehensive_model_training(
    config, 
    pipeline_state, 
    model_name='tactician_specialist_model'
)
```

## 3. Through Pipeline (Recommended)

```python
from src.training.steps.example_simplified_pipeline import ExampleSimplifiedPipeline

# The pipeline automatically creates both Analyst and Tactician
pipeline = ExampleSimplifiedPipeline(config)
result = await pipeline.execute_pipeline()
```

## 4. Configuration

```python
config = {
    'symbol': 'BTCUSDT',
    'exchange': 'binance',
    'timeframe': '1m',
    'model_training_config': {
        'enable_confidence_metrics': True,
        'enable_calibration_assessment': True,
        'enable_feature_importance': True,
        'enable_cross_validation': True,
        'enable_model_explanations': True,
        'enable_post_training_hpo': True,
        'cv_folds': 5
    }
}
```

## Core Principles Preserved

- ✅ **per-HMM regime training**: Models are trained specifically for different HMM-identified market regimes
- ✅ **Analyst/Tactician separation**: Distinct roles and models for Analyst and Tactician components
- ✅ **Tactician creation**: ConsolidatedTacticianSpecialistTraining handles tactician model creation
- ✅ **General model (Step 10)**: ConsolidatedUnifiedRegimeIntelligence handles the unified regime intelligence model
- ✅ **Tactician labels based on Analyst predictions**: Logic preserved in unified training and labeling
