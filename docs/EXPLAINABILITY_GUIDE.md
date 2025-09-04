# Explainability System Guide

## Overview

The explainability system provides comprehensive SHAP/LIME explanations for all ML models in the trading system and enables traceability of trade decisions back to individual factors. This guide covers installation, configuration, usage, and best practices.

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Basic Usage](#basic-usage)
4. [Model-Specific Explainers](#model-specific-explainers)
5. [Decision Tracing](#decision-tracing)
6. [Visualization](#visualization)
7. [Integration](#integration)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Installation

### Required Dependencies

```bash
pip install numpy pandas scikit-learn
```

### Optional Dependencies

For full functionality, install these optional packages:

```bash
# SHAP explanations
pip install shap

# LIME explanations
pip install lime

# Visualization
pip install matplotlib plotly seaborn

# Configuration
pip install pyyaml
```

### Verify Installation

```python
from src.explainability import ExplainabilityOrchestrator
print("✅ Explainability system installed successfully!")
```

## Configuration

### Basic Configuration

Create a configuration file `config/explainability_config.yaml`:

```yaml
explainability:
  enable_explanations: true
  enable_decision_tracing: true
  explanation_timeout: 30
  
  storage_path: "data/explanations"
  traces_storage_path: "data/decision_traces"
  
  enable_shap: true
  enable_lime: true
  max_features: 20
```

### Model-Specific Configuration

```yaml
explainability:
  tactician:
    explain_scenario_predictor: true
    explain_position_sizer: true
    explain_leverage_sizer: true
    
  hmm:
    explain_regime_classifier: true
    explain_transition_predictor: true
    explain_regime_probability: true
    
  sr:
    explain_level_detection: true
    explain_breakout_prediction: true
    explain_level_quality: true
    explain_strength_calculation: true
    
  analyst:
    explain_regime_classifier: true
    explain_location_classifier: true
    explain_ensemble_prediction: true
    explain_confidence_prediction: true
```

### Visualization Configuration

```yaml
explainability:
  visualization:
    output_path: "data/visualizations"
    enable_matplotlib: true
    enable_plotly: true
    
    style:
      figure_size: [12, 8]
      dpi: 300
      colors:
        positive: "#2E8B57"
        negative: "#DC143C"
        neutral: "#4682B4"
```

## Basic Usage

### 1. Initialize the Orchestrator

```python
from src.explainability import ExplainabilityOrchestrator

config = {
    "explainability": {
        "enable_explanations": True,
        "enable_decision_tracing": True,
        "storage_path": "data/explanations",
        "traces_storage_path": "data/decision_traces"
    }
}

orchestrator = ExplainabilityOrchestrator(config)
```

### 2. Register Models

```python
import pandas as pd
from your_models import TacticianModel, HMMModel

# Create your models
tactician_model = TacticianModel()
hmm_model = HMMModel()

# Prepare training data
training_data = pd.read_csv("data/training_data.csv")

# Register models
await orchestrator.register_model(
    'tactician', 'main', tactician_model, training_data
)
await orchestrator.register_model(
    'hmm', 'main', hmm_model, training_data
)
```

### 3. Generate Explanations

```python
import numpy as np

# Prepare features for prediction
features = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
feature_names = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']

# Generate explanation
explanation = await orchestrator.explain_model_prediction(
    'tactician', 'main', features, feature_names
)

if explanation:
    print(f"Model: {explanation.model_name}")
    print(f"Prediction: {explanation.prediction}")
    print(f"Confidence: {explanation.confidence}")
    print(f"SHAP values available: {explanation.shap_values is not None}")
    print(f"LIME explanation available: {explanation.lime_explanation is not None}")
```

## Model-Specific Explainers

### Tactician Explainer

```python
from src.explainability import TacticianExplainer

explainer = TacticianExplainer(config)

# Explain scenario prediction
scenario_explanation = await explainer.explain_scenario_prediction(
    scenario_predictor, market_data, features, feature_names
)

# Explain position sizing
position_explanation = await explainer.explain_position_sizing(
    position_sizer, market_data, features, feature_names
)

# Explain leverage decision
leverage_explanation = await explainer.explain_leverage_decision(
    leverage_sizer, market_data, features, feature_names
)
```

### HMM Explainer

```python
from src.explainability import HMMExplainer

explainer = HMMExplainer(config)

# Explain regime classification
regime_explanation = await explainer.explain_regime_classification(
    hmm_model, market_data, features, feature_names
)

# Explain regime probabilities
prob_explanation = await explainer.explain_regime_probabilities(
    hmm_model, market_data, features, feature_names
)

# Explain transition prediction
transition_explanation = await explainer.explain_transition_prediction(
    hmm_model, market_data, features, feature_names
)
```

### SR Explainer

```python
from src.explainability import SRExplainer

explainer = SRExplainer(config)

# Explain level detection
detection_explanation = await explainer.explain_level_detection(
    sr_model, market_data, features, feature_names
)

# Explain breakout prediction
breakout_explanation = await explainer.explain_breakout_prediction(
    sr_model, market_data, features, feature_names
)

# Explain level quality
quality_explanation = await explainer.explain_level_quality(
    sr_model, market_data, features, feature_names
)
```

### Analyst Explainer

```python
from src.explainability import AnalystExplainer

explainer = AnalystExplainer(config)

# Explain regime classification
regime_explanation = await explainer.explain_regime_classification(
    analyst_model, market_data, features, feature_names
)

# Explain location classification
location_explanation = await explainer.explain_location_classification(
    analyst_model, market_data, features, feature_names
)

# Explain ensemble prediction
ensemble_explanation = await explainer.explain_ensemble_prediction(
    analyst_model, market_data, features, feature_names
)
```

## Decision Tracing

### Start a Decision Trace

```python
# Start tracing a trade decision
decision_id = "trade_001"
trace = await orchestrator.start_trade_decision_trace(
    decision_id, "entry", market_conditions
)
```

### Add Model Explanations

```python
# Add explanations from different models
if tactician_explanation:
    await orchestrator.add_explanation_to_trace(
        decision_id, 'tactician', tactician_explanation
    )

if hmm_explanation:
    await orchestrator.add_explanation_to_trace(
        decision_id, 'hmm', hmm_explanation
    )
```

### Finalize the Trace

```python
# Finalize with the final decision
final_trace = await orchestrator.finalize_trade_decision_trace(
    decision_id, "BUY", 0.8
)

if final_trace:
    print(f"Decision: {final_trace.final_decision}")
    print(f"Confidence: {final_trace.confidence}")
    print(f"Top factors: {len(final_trace.top_contributing_factors)}")
    print(f"Risk factors: {len(final_trace.risk_factors)}")
    print(f"Opportunity factors: {len(final_trace.opportunity_factors)}")
```

### Complete Trading Decision

```python
# Explain a complete trading decision across all models
trace = await orchestrator.explain_complete_trading_decision(
    decision_id="complete_001",
    decision_type="entry",
    market_data=market_data,
    tactician_features=(features, feature_names),
    hmm_features=(features, feature_names),
    sr_features=(features, feature_names),
    analyst_features=(features, feature_names),
    final_decision="BUY",
    confidence=0.8
)
```

## Visualization

### Explanation Visualizations

```python
from src.explainability import ExplanationVisualizer

visualizer = ExplanationVisualizer(config)

# Visualize SHAP values
shap_path = visualizer.visualize_shap_values(explanation)

# Visualize LIME explanation
lime_path = visualizer.visualize_lime_explanation(explanation)

# Visualize feature importance
importance_path = visualizer.visualize_feature_importance(explanation)

# Create comprehensive dashboard
dashboard_path = visualizer.create_explanation_dashboard(explanation)
```

### Decision Trace Visualizations

```python
from src.explainability import DecisionTraceVisualizer

trace_visualizer = DecisionTraceVisualizer(config)

# Visualize complete decision trace
trace_path = trace_visualizer.visualize_decision_trace(final_trace)
```

## Integration

### Using Decorators

```python
from src.explainability import (
    explainable_tactician_prediction,
    explainable_hmm_prediction,
    explainable_trading_decision,
    FeatureExtractor
)

# Decorate prediction functions
@explainable_tactician_prediction(
    model_name="main",
    feature_extractor=FeatureExtractor.from_dataframe(['close', 'volume', 'rsi'])
)
async def predict_trading_decision(market_data):
    # Your prediction logic here
    return {"decision": "BUY", "confidence": 0.8}

# Decorate trading decisions
@explainable_trading_decision(
    decision_type="entry",
    model_types=['tactician', 'hmm'],
    feature_extractors={
        'tactician': FeatureExtractor.from_dataframe(['close', 'volume', 'rsi']),
        'hmm': FeatureExtractor.from_dataframe(['volatility_20', 'atr', 'adx'])
    }
)
async def make_trading_decision(market_data):
    # Your decision logic here
    return {"action": "BUY", "confidence": 0.8}
```

### Feature Extractors

```python
from src.explainability import FeatureExtractor

# From DataFrame columns
extractor1 = FeatureExtractor.from_dataframe(['close', 'volume', 'rsi'])

# From dictionary mapping
extractor2 = FeatureExtractor.from_dict({
    'price_feature': 'close',
    'volume_feature': 'volume',
    'volatility_feature': 'volatility_20'
})

# Custom extractor
async def custom_extractor(market_data):
    features = np.array([
        market_data['close'].iloc[-1],
        market_data['volume'].iloc[-1],
        market_data['rsi'].iloc[-1]
    ])
    feature_names = ['close', 'volume', 'rsi']
    return features, feature_names

extractor3 = FeatureExtractor.custom(custom_extractor)
```

## Best Practices

### 1. Model Registration

- Always register models with training data for better explanations
- Use descriptive model names for easier identification
- Initialize explainers after model training

### 2. Feature Engineering

- Use consistent feature names across models
- Normalize features before explanation
- Include feature groups for better interpretation

### 3. Decision Tracing

- Start traces early in the decision process
- Include all relevant model explanations
- Use meaningful decision IDs for tracking

### 4. Performance

- Set appropriate timeouts for explanation generation
- Use feature selection to reduce explanation time
- Cache explanations for repeated predictions

### 5. Storage Management

- Regularly clean up old explanation files
- Use compressed storage for large datasets
- Monitor storage usage

## Troubleshooting

### Common Issues

#### 1. SHAP/LIME Not Available

**Error**: `Warning: SHAP not available, SHAP explanations disabled`

**Solution**: Install SHAP and LIME
```bash
pip install shap lime
```

#### 2. Visualization Not Working

**Error**: `Warning: matplotlib not available, visualization features disabled`

**Solution**: Install visualization packages
```bash
pip install matplotlib plotly seaborn
```

#### 3. Explanation Timeout

**Error**: `Explanation timeout for model_name`

**Solution**: Increase timeout or reduce features
```yaml
explainability:
  explanation_timeout: 60  # Increase timeout
  max_features: 10        # Reduce features
```

#### 4. Memory Issues

**Error**: Out of memory during explanation generation

**Solution**: 
- Reduce batch size
- Use feature selection
- Process explanations in smaller chunks

#### 5. Model Not Registered

**Error**: `Model model_name not registered`

**Solution**: Register the model first
```python
await orchestrator.register_model(
    'model_type', 'model_name', model, training_data
)
```

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Monitoring

Monitor explanation performance:

```python
import time

start_time = time.time()
explanation = await orchestrator.explain_model_prediction(...)
end_time = time.time()

print(f"Explanation time: {end_time - start_time:.2f} seconds")
```

## Examples

See the `examples/explainability_example.py` file for comprehensive examples of:

- Basic model explanations
- Decision tracing
- Complete trading decision explanation
- Visualization
- Integration decorators

## API Reference

### Core Classes

- `ExplainabilityOrchestrator`: Main orchestrator for managing explanations
- `BaseExplainer`: Base class for model explainers
- `TradeDecisionTracer`: Tracer for trade decisions
- `ExplanationResult`: Result of model explanation
- `TradeDecisionTrace`: Trace of a trade decision

### Model Explainers

- `TacticianExplainer`: Explainer for Tactician models
- `HMMExplainer`: Explainer for HMM models
- `SRExplainer`: Explainer for SR models
- `AnalystExplainer`: Explainer for Analyst models

### Visualization

- `ExplanationVisualizer`: Visualizer for model explanations
- `DecisionTraceVisualizer`: Visualizer for decision traces

### Integration

- `ExplainabilityIntegration`: Integration helper
- `FeatureExtractor`: Feature extraction utilities
- Decorators: `explainable_*_prediction`, `explainable_trading_decision`

## Contributing

When contributing to the explainability system:

1. Follow the existing code structure
2. Add comprehensive tests
3. Update documentation
4. Ensure backward compatibility
5. Add type hints and docstrings

## License

This explainability system is part of the trading system and follows the same license terms.