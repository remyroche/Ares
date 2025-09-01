# Probabilistic Bayesian Optimization for Tactician and Analyst Models

This framework provides advanced Bayesian optimization specifically designed for probabilistic models that output probability distributions, confidence intervals, and uncertainty estimates. It's tailored to optimize your Tactician and Analyst models for better probabilistic outputs and uncertainty quantification.

## 🎯 Key Features

### **Probabilistic Model Optimization**
- **Calibration**: Ensures predicted probabilities match observed frequencies
- **Sharpness**: Makes predictions as precise and confident as possible
- **Discrimination**: Maximizes the difference between positive and negative predictions
- **Uncertainty Quantification**: Optimizes confidence intervals and uncertainty estimates

### **Model-Specific Optimization**
- **Tactician Models**: Optimizes barrier systems, confidence thresholds, and precision parameters
- **Analyst Models**: Optimizes regime detection, ensemble methods, and multi-output predictions
- **Customizable**: Easy to extend for new model types and objectives

### **Advanced Optimization Algorithms**
- **Multi-objective optimization** using NSGA-II, MOEA/D, and SPEA2
- **Efficient sampling** with TPE, CMA-ES, and Random strategies
- **Early stopping** and pruning for faster convergence
- **Parallel execution** for scalable optimization

## 🚀 Quick Start

### 1. Installation

Ensure you have the required dependencies:

```bash
pip install optuna scikit-learn numpy pandas matplotlib
```

### 2. Basic Usage

```python
from src.training.probabilistic_bayesian_optimizer import (
    ProbabilisticBayesianOptimizer,
    ProbabilisticOptimizationConfig
)

# Configure optimization
config = ProbabilisticOptimizationConfig(
    objectives=['calibration', 'sharpness', 'discrimination'],
    n_trials=100,
    n_jobs=1
)

# Create optimizer for Tactician
tactician_optimizer = ProbabilisticBayesianOptimizer(
    config=config,
    model_type="tactician"
)

# Run optimization
results = tactician_optimizer.optimize(
    X=your_features,
    y=your_targets,
    model_factory=your_model_factory
)
```

### 3. Using the Model Integrator

```python
from src.training.probabilistic_model_integration import ProbabilisticModelIntegrator

# Configuration
config = {
    "optimization": {
        "n_trials": 100,
        "n_jobs": 1,
        "early_stopping_patience": 10
    }
}

# Create integrator
integrator = ProbabilisticModelIntegrator(config)

# Run comprehensive optimization
results = await integrator.run_comprehensive_optimization(
    market_data=your_market_data,
    historical_predictions=your_predictions
)
```

## 📊 Optimization Objectives

### **Calibration**
- **Goal**: Ensure predicted probabilities match actual outcomes
- **Metric**: Brier Score (lower is better)
- **Example**: If model predicts 80% probability, actual outcome should occur ~80% of the time

### **Sharpness**
- **Goal**: Make predictions as confident as possible
- **Metric**: Negative entropy of predictions (higher is better)
- **Example**: Prefer 90% vs 10% over 60% vs 40%

### **Discrimination**
- **Goal**: Maximize separation between positive and negative predictions
- **Metric**: ROC AUC (higher is better)
- **Example**: Clear distinction between winning and losing trades

### **Uncertainty Quality**
- **Goal**: Ensure confidence intervals are accurate
- **Metric**: Coverage rate (should match confidence level)
- **Example**: 95% confidence intervals should contain true values 95% of the time

## 🔧 Configuration

The framework uses a comprehensive YAML configuration file (`config/probabilistic_optimization.yaml`) that allows you to customize:

- **Optimization parameters**: Number of trials, sampling strategy, early stopping
- **Model-specific settings**: Hyperparameter search spaces, objective weights
- **Data preparation**: Feature engineering, target creation, validation
- **Evaluation metrics**: Cross-validation, performance thresholds
- **Deployment settings**: Model versioning, monitoring, A/B testing

### Example Configuration

```yaml
models:
  tactician:
    objectives: ["calibration", "sharpness", "discrimination"]
    objective_weights:
      calibration: 0.4
      sharpness: 0.3
      discrimination: 0.3
    hyperparameters:
      barrier_system:
        upper_barrier_multiplier: [0.3, 0.8]
        confidence_threshold: [0.6, 0.9]
```

## 📈 Advanced Usage

### **Multi-Objective Optimization**

```python
# Configure for multiple objectives
config = ProbabilisticOptimizationConfig(
    objectives=['calibration', 'sharpness', 'discrimination'],
    n_trials=200
)

# Get Pareto-optimal solutions
results = optimizer.optimize(X, y, model_factory)
pareto_front = results['pareto_front']

# Choose solution based on your preferences
best_solution = optimizer.get_recommended_hyperparameters(
    objective_weights={'calibration': 0.5, 'sharpness': 0.3, 'discrimination': 0.2}
)
```

### **Custom Model Factories**

```python
def create_custom_tactician_model(params):
    """Create a custom Tactician model with given parameters."""
    from your_tactician_module import TacticianModel
    
    model = TacticianModel(
        upper_barrier_multiplier=params['upper_barrier_multiplier'],
        confidence_threshold=params['confidence_threshold'],
        calibration_method=params['calibration_method']
    )
    
    return model

# Use custom factory
results = optimizer.optimize(
    X=X, y=y,
    model_factory=create_custom_tactician_model
)
```

### **Uncertainty Estimation Methods**

```python
# Configure uncertainty estimation
config = ProbabilisticOptimizationConfig(
    objectives=['calibration', 'sharpness', 'discrimination'],
    uncertainty_methods=['ensemble', 'gaussian', 'conformal']
)

# The optimizer will automatically test different uncertainty methods
# and select the best performing one
```

## 📊 Monitoring and Analysis

### **Optimization Progress**

```python
# Get current status
status = integrator.get_optimization_status()
print(f"Optimizers created: {status['optimizers_created']}")
print(f"Recommendations: {status['recommendations']}")

# Plot optimization results
integrator.plot_optimization_results("tactician", save_path="tactician_optimization.png")
```

### **Performance Tracking**

```python
# Track model performance over time
performance_history = integrator.model_performance

# Get optimization recommendations
recommendations = integrator.get_optimization_status()['recommendations']
for rec in recommendations:
    print(f"• {rec}")
```

## 🧪 Testing

Run the comprehensive test suite to verify everything works:

```bash
python test_probabilistic_bayesian_optimization.py
```

This will test:
1. **Direct optimizer testing** - Basic optimization functionality
2. **Model integrator testing** - End-to-end optimization workflows
3. **Uncertainty quantification** - Different uncertainty estimation methods

## 🔍 Integration with Existing Models

### **Tactician Integration**

```python
# Your existing Tactician model
from src.tactician.enhanced_prediction_integrator import TacticianEnhancedPredictionIntegrator

# Create factory function
def tactician_factory(params):
    tactician = TacticianEnhancedPredictionIntegrator()
    
    # Apply optimized parameters
    tactician.config['barrier_system'].update({
        'upper_barrier_multiplier': params['upper_barrier_multiplier'],
        'confidence_threshold': params['confidence_threshold']
    })
    
    return tactician

# Optimize
results = await integrator.optimize_tactician_model(
    market_data, historical_predictions
)
```

### **Analyst Integration**

```python
# Your existing Analyst models
from src.analyst.ml_confidence_predictor import MLConfidencePredictor
from src.analyst.regime_predictor import RegimePredictor

# Create factory function
def analyst_factory(params):
    # Create ensemble of Analyst models
    models = {
        'confidence': MLConfidencePredictor(),
        'regime': RegimePredictor()
    }
    
    # Apply optimized parameters
    for model in models.values():
        if hasattr(model, 'set_hyperparameters'):
            model.set_hyperparameters(params)
    
    return models

# Optimize
results = await integrator.optimize_analyst_model(
    market_data, historical_predictions
)
```

## 📚 Best Practices

### **1. Data Quality**
- Ensure sufficient historical data (minimum 1000 samples)
- Validate feature quality and remove outliers
- Use time-series cross-validation for financial data

### **2. Optimization Strategy**
- Start with fewer trials (20-50) for testing
- Use early stopping to prevent overfitting
- Monitor convergence and adjust parameters

### **3. Model Selection**
- Balance multiple objectives based on your priorities
- Consider ensemble methods for better uncertainty estimation
- Validate on out-of-sample data

### **4. Continuous Improvement**
- Re-optimize models periodically (weekly/monthly)
- Monitor performance degradation
- A/B test new optimizations

## 🚨 Troubleshooting

### **Common Issues**

1. **Import Errors**
   ```bash
   # Ensure modules are in Python path
   export PYTHONPATH="${PYTHONPATH}:/path/to/your/project"
   ```

2. **Memory Issues**
   ```python
   # Reduce data size for testing
   config = ProbabilisticOptimizationConfig(n_trials=20)
   ```

3. **Slow Optimization**
   ```python
   # Enable parallel processing
   config = ProbabilisticOptimizationConfig(n_jobs=4)
   
   # Use early stopping
   config.early_stopping_patience = 10
   ```

4. **Poor Results**
   ```python
   # Check data quality
   print(f"Data shape: {X.shape}")
   print(f"Target distribution: {np.bincount(y)}")
   
   # Adjust objective weights
   config.objective_weights = {'calibration': 0.6, 'sharpness': 0.2, 'discrimination': 0.2}
   ```

## 🔮 Future Enhancements

- **Deep Learning Integration**: Support for neural network uncertainty estimation
- **Online Learning**: Continuous optimization during live trading
- **Market Regime Adaptation**: Dynamic optimization based on market conditions
- **Federated Learning**: Collaborative optimization across multiple models
- **Explainable AI**: Interpretable optimization results and recommendations

## 📖 References

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Probabilistic Machine Learning](https://probml.github.io/pml-book/)
- [Bayesian Optimization for Machine Learning](https://arxiv.org/abs/1807.02811)
- [Uncertainty Quantification in Deep Learning](https://arxiv.org/abs/2006.07520)

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Happy Optimizing! 🚀**

For questions and support, please open an issue in the repository.