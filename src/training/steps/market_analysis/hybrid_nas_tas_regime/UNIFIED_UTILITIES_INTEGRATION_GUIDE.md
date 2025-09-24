# Unified Utilities Integration Guide

This guide explains how to use the newly implemented unified utilities that consolidate common functionality between TAS and NAS regime detection systems.

## Overview

The unified utilities provide a comprehensive framework for:
- **Configuration Management**: Unified configuration system for TAS, NAS, and hybrid architectures
- **Performance Monitoring**: Real-time performance tracking and regime-specific analysis
- **Meta-Learning**: Rapid adaptation and few-shot learning capabilities
- **Hardware Optimization**: GPU acceleration, memory optimization, and adaptive resource allocation
- **Evaluation Framework**: Comprehensive evaluation including economic significance and trading viability

## Quick Start

### 1. Configuration Management

```python
from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils import (
    create_tas_config, create_nas_config, create_hybrid_config,
    ArchitectureType, SearchStrategy, OptimizationObjective
)

# Create TAS configuration
tas_config = create_tas_config(
    search_strategy=SearchStrategy.BAYESIAN,
    enable_micro_regime_detection=True,
    economic_significance_threshold=0.8
)

# Create NAS configuration
nas_config = create_nas_config(
    search_strategy=SearchStrategy.EVOLUTIONARY,
    enable_meta_learning=True,
    meta_learning_rate=1e-3
)

# Save configurations
tas_config.save_config("tas_config.json")
nas_config.save_config("nas_config.json")
```

### 2. Performance Monitoring

```python
from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils import (
    create_performance_monitor, MonitoringLevel, PerformanceMetric
)

# Create performance monitor
monitor = create_performance_monitor(
    architecture_type=ArchitectureType.TAS,
    monitoring_level=MonitoringLevel.STANDARD,
    enable_real_time=True
)

# Start monitoring
monitor.start_monitoring()

# Record performance
metrics = {
    PerformanceMetric.ACCURACY: 0.85,
    PerformanceMetric.SHARPE_RATIO: 1.6,
    PerformanceMetric.ECONOMIC_SIGNIFICANCE: 0.75
}
monitor.record_performance(metrics, iteration=1, regime_id="regime_1")

# Get summary
summary = monitor.get_performance_summary()
print(f"Performance summary: {summary}")
```

### 3. Meta-Learning

```python
from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils import (
    create_meta_learner, MetaLearningMethod, AdaptationType
)

# Create meta-learner
meta_learner = create_meta_learner(
    architecture_type=ArchitectureType.NAS,
    method=MetaLearningMethod.MAML,
    adaptation_type=AdaptationType.REGIME_ADAPTATION
)

# Create meta-tasks from data
tasks = meta_learner.create_meta_tasks(
    data={'regime_1': (X_data, y_data)},
    regime_labels=regime_labels
)

# Perform meta-training
result = meta_learner.meta_train(tasks)
print(f"Meta-training result: {result}")
```

### 4. Hardware Optimization

```python
from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils import (
    create_hardware_manager, OptimizationLevel, WorkloadType
)

# Create hardware manager
hardware_manager = create_hardware_manager(
    architecture_type=ArchitectureType.NAS,
    optimization_level=OptimizationLevel.AGGRESSIVE,
    enable_gpu_acceleration=True
)

# Optimize for specific workload
optimization_result = hardware_manager.optimize_for_workload(
    WorkloadType.NEURAL_TRAINING,
    parameters={'batch_size': 64, 'learning_rate': 0.001}
)

# Get hardware status
status = hardware_manager.get_hardware_status()
print(f"Hardware status: {status}")
```

### 5. Evaluation Framework

```python
from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils import (
    create_evaluation_framework, EvaluationType, EvaluationMetric
)

# Create evaluation framework
evaluator = create_evaluation_framework(
    architecture_type=ArchitectureType.TAS,
    evaluation_type=EvaluationType.COMPREHENSIVE
)

# Evaluate model
result = evaluator.evaluate_model(
    model=model,
    X_test=X_test,
    y_test=y_test,
    market_data=market_data,
    regime_labels=regime_labels,
    model_name="My_Model"
)

# Get evaluation summary
summary = evaluator.get_evaluation_summary()
print(f"Evaluation summary: {summary}")
```

## Advanced Usage

### Configuration Customization

The unified configuration system supports extensive customization:

```python
# Create custom TAS configuration
tas_config = TASArchitectureConfig(
    # Basic settings
    architecture_type=ArchitectureType.TAS,
    search_strategy=SearchStrategy.HYBRID,
    
    # TAS-specific settings
    min_trees=50,
    max_trees=500,
    enable_cvlSA_architecture=True,
    cvlSA_cascade_depth=3,
    
    # Performance thresholds
    economic_significance_threshold=0.8,
    trading_viability_threshold=0.7,
    
    # Advanced features
    enable_meta_learning=True,
    enable_uncertainty_estimation=True,
    enable_real_time_adaptation=True
)

# Save and load configurations
tas_config.save_config("custom_tas_config.json")
loaded_config = TASArchitectureConfig.load_config("custom_tas_config.json")
```

### Performance Monitoring with Alerts

```python
# Create monitor with custom alert callback
def alert_callback(alert):
    print(f"ALERT: {alert['message']} - Data: {alert['data']}")

monitor = create_performance_monitor(
    architecture_type=ArchitectureType.NAS,
    monitoring_level=MonitoringLevel.REAL_TIME
)

# Register alert callback
monitor.register_alert_callback(alert_callback)

# Start monitoring
monitor.start_monitoring()

# The monitor will automatically trigger alerts for performance issues
```

### Meta-Learning with Continual Learning

```python
# Create continual learning meta-learner
meta_learner = create_continual_learner(
    architecture_type=ArchitectureType.HYBRID
)

# Initialize with base model
meta_learner.initialize_meta_model(base_model)

# Perform continual learning updates
for new_data_batch in data_stream:
    meta_learner.continual_learning_update(
        new_data=new_data_batch,
        regime_id=f"regime_{batch_id}"
    )

# Get adaptation statistics
stats = meta_learner.get_adaptation_statistics()
print(f"Adaptation statistics: {stats}")
```

### Hardware Optimization for Specific Workloads

```python
# Create aggressive hardware manager
hardware_manager = create_aggressive_hardware_manager(
    architecture_type=ArchitectureType.NAS
)

# Optimize for different workloads
training_optimization = hardware_manager.optimize_for_workload(
    WorkloadType.NEURAL_TRAINING,
    parameters={
        'batch_size': 128,
        'mixed_precision': True,
        'gradient_accumulation': 4
    }
)

inference_optimization = hardware_manager.optimize_for_workload(
    WorkloadType.INFERENCE,
    parameters={
        'batch_size': 1,
        'enable_quantization': True
    }
)

# Get optimization recommendations
recommendations = hardware_manager.get_optimization_recommendations()
print(f"Recommendations: {recommendations}")
```

### Comprehensive Model Evaluation

```python
# Create comprehensive evaluator
evaluator = create_evaluation_framework(
    architecture_type=ArchitectureType.HYBRID,
    evaluation_type=EvaluationType.COMPREHENSIVE
)

# Evaluate multiple models
models = [
    (tas_model, "TAS_Model"),
    (nas_model, "NAS_Model"),
    (hybrid_model, "Hybrid_Model")
]

results = evaluator.compare_models(
    models=models,
    X_test=X_test,
    y_test=y_test,
    market_data=market_data,
    regime_labels=regime_labels
)

# Results are automatically sorted by performance
best_model = results[0]
print(f"Best model: {best_model.model_name}")
```

## Integration with Existing Systems

### Updating TAS Engine

```python
# In your existing TAS engine, replace old configuration with unified config
from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils import (
    TASArchitectureConfig, create_performance_monitor, create_hardware_manager
)

class UpdatedTASEngine:
    def __init__(self, config_dict=None):
        # Use unified configuration
        self.config = TASArchitectureConfig.from_dict(config_dict) if config_dict else TASArchitectureConfig()
        
        # Use unified performance monitoring
        self.performance_monitor = create_performance_monitor(
            architecture_type=ArchitectureType.TAS,
            monitoring_level=MonitoringLevel.STANDARD
        )
        
        # Use unified hardware management
        self.hardware_manager = create_hardware_manager(
            architecture_type=ArchitectureType.TAS,
            optimization_level=OptimizationLevel.BALANCED
        )
    
    def search(self, train_data, validation_data):
        # Start performance monitoring
        self.performance_monitor.start_monitoring()
        
        # Optimize hardware for tree training
        self.hardware_manager.optimize_for_workload(
            WorkloadType.TREE_TRAINING,
            parameters={'tree_count': self.config.max_trees}
        )
        
        # Your existing search logic here...
        
        # Record performance
        self.performance_monitor.record_performance(
            metrics={'accuracy': accuracy, 'sharpe_ratio': sharpe},
            iteration=iteration,
            regime_id=regime_id
        )
```

### Updating NAS Engine

```python
# In your existing NAS engine, integrate unified utilities
from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils import (
    NASArchitectureConfig, create_meta_learner, create_evaluation_framework
)

class UpdatedNASEngine:
    def __init__(self, config_dict=None):
        # Use unified configuration
        self.config = NASArchitectureConfig.from_dict(config_dict) if config_dict else NASArchitectureConfig()
        
        # Use unified meta-learning
        self.meta_learner = create_meta_learner(
            architecture_type=ArchitectureType.NAS,
            method=MetaLearningMethod.MAML
        )
        
        # Use unified evaluation
        self.evaluator = create_evaluation_framework(
            architecture_type=ArchitectureType.NAS,
            evaluation_type=EvaluationType.COMPREHENSIVE
        )
    
    def search(self, train_data, validation_data):
        # Initialize meta-model
        self.meta_learner.initialize_meta_model(base_model)
        
        # Create meta-tasks
        tasks = self.meta_learner.create_meta_tasks(train_data)
        
        # Perform meta-training
        meta_result = self.meta_learner.meta_train(tasks)
        
        # Your existing NAS search logic here...
        
        # Evaluate results
        evaluation_result = self.evaluator.evaluate_model(
            model=best_model,
            X_test=test_data,
            y_test=test_labels,
            model_name="NAS_Best_Model"
        )
```

## Best Practices

### 1. Configuration Management
- Use the appropriate configuration class for your architecture type
- Save configurations for reproducibility
- Validate configurations before use
- Use configuration presets for common scenarios

### 2. Performance Monitoring
- Start monitoring early in your training process
- Use appropriate monitoring levels (basic for simple cases, comprehensive for production)
- Register alert callbacks for important performance thresholds
- Export performance data for analysis

### 3. Meta-Learning
- Initialize meta-models with your base architecture
- Create diverse meta-tasks for better adaptation
- Use continual learning for streaming data scenarios
- Monitor adaptation statistics for optimization

### 4. Hardware Optimization
- Choose optimization levels appropriate for your use case
- Optimize for specific workloads when possible
- Monitor hardware performance and adjust as needed
- Use aggressive optimization only when necessary

### 5. Evaluation
- Use comprehensive evaluation for production models
- Compare multiple models using the unified framework
- Include economic significance and trading viability metrics
- Export evaluation results for documentation

## Migration Guide

### From Old TAS Configuration

```python
# Old way
from src.training.steps.market_analysis.tas_regime.core.tas_config import TASConfig
config = TASConfig()

# New way
from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils import TASArchitectureConfig
config = TASArchitectureConfig()
```

### From Old NAS Configuration

```python
# Old way
from src.training.steps.market_analysis.nas_regime.core.perfect_nas_config import PerfectNASConfig
config = PerfectNASConfig()

# New way
from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils import NASArchitectureConfig
config = NASArchitectureConfig()
```

### From Old Evaluation Systems

```python
# Old way
from src.training.steps.market_analysis.tas_regime.evaluation.tas_evaluator import TASEvaluator
evaluator = TASEvaluator(config)

# New way
from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils import create_evaluation_framework
evaluator = create_evaluation_framework(
    architecture_type=ArchitectureType.TAS,
    evaluation_type=EvaluationType.COMPREHENSIVE
)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're importing from the correct unified utilities module
2. **Configuration Errors**: Validate your configuration parameters before use
3. **Performance Monitoring**: Ensure monitoring is started before recording metrics
4. **Meta-Learning**: Initialize meta-models before creating meta-tasks
5. **Hardware Optimization**: Check device availability before enabling GPU acceleration

### Getting Help

- Check the example scripts in `examples/unified_utilities_example.py`
- Review the configuration classes for available parameters
- Use the convenience functions for common use cases
- Check logs for detailed error messages

## Future Enhancements

The unified utilities framework is designed to be extensible. Future enhancements may include:

- Additional meta-learning methods
- More sophisticated hardware optimization strategies
- Enhanced evaluation metrics
- Integration with more ML frameworks
- Automated hyperparameter optimization
- Distributed training support

## Conclusion

The unified utilities provide a comprehensive, consolidated framework for managing TAS and NAS architectures. By using these utilities, you can:

- Reduce code duplication between TAS and NAS systems
- Ensure consistency across different architectures
- Leverage advanced features like meta-learning and hardware optimization
- Perform comprehensive evaluations with economic significance
- Monitor performance in real-time with regime-specific analysis

Start with the basic examples and gradually incorporate more advanced features as needed for your specific use case.