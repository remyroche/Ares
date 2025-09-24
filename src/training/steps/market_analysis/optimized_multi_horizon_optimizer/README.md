# Optimized Multi-Horizon Optimizer

## 🎯 **Overview**

The Optimized Multi-Horizon Optimizer is a comprehensive timeframe optimization system that leverages **ml_commons utilities extensively** for grid search, Bayesian TPE optimization, cross-validation, and comprehensive validation.

## 🚀 **Key Features**

### **Optimization Methods**
- ✅ **Grid Search (Coarse + Fine)**: Uses `ml_commons.optimization.grid_utils`
- ✅ **Bayesian TPE Optimization**: Uses `ml_commons.utils.hpo_utils`
- ✅ **Grid + Bayesian Hybrid**: Best of both worlds

### **Validation Framework**
- ✅ **Statistical Validation**: IC, SNR, Hit Rate, Significance
- ✅ **Economic Validation**: Transaction costs, Sharpe ratio, Drawdown
- ✅ **Market Microstructure**: Liquidity, Volatility, Depth
- ✅ **Cross-Validation**: Time series CV, Stability analysis
- ✅ **Stability Validation**: Temporal, Regime, Parameter stability

### **ml_commons Integration**
- ✅ **Grid Utilities**: `build_coarse_grid_from_search_space`, `build_fine_grid_around_best`
- ✅ **HPO Utilities**: `HyperparameterOptimization` with TPE sampling
- ✅ **Validation System**: `UnifiedValidationSystem` for comprehensive validation
- ✅ **Cross-Validation**: `TemporalCrossValidator`, `CrossValidationUtilities`
- ✅ **Overfitting Detection**: `EnhancedOverfittingDetector`
- ✅ **Data Leakage Prevention**: `DataLeakagePrevention`
- ✅ **Stability Validation**: `StabilityValidator`
- ✅ **Memory Optimization**: `MemoryOptimizer`
- ✅ **Lookahead Protection**: `LookaheadProtection`

## 📁 **Directory Structure**

```
optimized_multi_horizon_optimizer/
├── __init__.py                          # Module exports
├── optimization_config.py               # Configuration classes and enums
├── grid_bayesian_optimizer.py          # Grid + Bayesian TPE optimizer
├── enhanced_validation.py              # Enhanced validation framework
├── optimized_timeframe_optimizer.py    # Main optimizer class
├── integration_example.py              # Usage examples
└── README.md                           # This documentation
```

## 🔧 **Usage**

### **Basic Usage**

```python
from src.training.steps.market_analysis.optimized_multi_horizon_optimizer import (
    OptimizedTimeframeOptimizer,
    OptimizationConfig,
    ModelType,
    OptimizationMethod
)

# Create configuration
config = OptimizationConfig(
    model_type=ModelType.BOTH,
    optimization_method=OptimizationMethod.GRID_BAYESIAN,
    base_timeframe_analyst=15,  # 15 minutes
    base_timeframe_tactician=5,  # 5 minutes
    horizon_range=(1, 16)  # 1-16 periods
)

# Initialize optimizer
optimizer = OptimizedTimeframeOptimizer(config)

# Run optimization
results = optimizer.get_optimal_timeframes_for_models(
    market_data=market_data,
    model_type=ModelType.BOTH
)

# Access results
for model_type, result in results.items():
    print(f"{model_type}: {result.optimal_horizons}")
    print(f"Score: {result.optimization_score:.3f}")
```

### **Advanced Usage**

```python
# Custom configuration
config = OptimizationConfig(
    model_type=ModelType.ANALYST,
    optimization_method=OptimizationMethod.GRID_BAYESIAN,
    
    # Grid search settings
    grid_search_config=GridSearchConfig(
        coarse_grid_points=10,
        fine_grid_points=20,
        enable_parallel=True,
        max_workers=8
    ),
    
    # Bayesian TPE settings
    bayesian_tpe_config=BayesianTPEConfig(
        n_trials=100,
        enable_pruning=True,
        pruning_patience=10
    ),
    
    # Validation settings
    validation_config=ValidationConfig(
        validation_level=ValidationLevel.COMPREHENSIVE,
        cv_folds=10,
        enable_statistical_validation=True,
        enable_economic_validation=True,
        enable_microstructure_validation=True
    )
)

# Initialize with custom config
optimizer = OptimizedTimeframeOptimizer(config)

# Run optimization with monitoring
results = optimizer.optimize_for_model(
    model_type=ModelType.ANALYST,
    market_data=market_data,
    force_optimization=True
)
```

## 🎯 **Optimization Process**

### **Stage 1: Coarse Grid Search**
1. **Generate Coarse Grid**: Uses `build_coarse_grid_from_search_space`
2. **Evaluate Points**: Tests each grid point for performance
3. **Select Best**: Identifies promising regions

### **Stage 2: Fine Grid Search**
1. **Generate Fine Grid**: Uses `build_fine_grid_around_best`
2. **Refine Search**: Focuses on best coarse results
3. **Improve Precision**: Higher resolution around optimal regions

### **Stage 3: Bayesian TPE Optimization**
1. **Initialize TPE**: Uses `HyperparameterOptimization` from ml_commons
2. **Sample Parameters**: TPE sampling for efficient exploration
3. **Prune Poor Trials**: Early stopping for efficiency
4. **Converge to Optimum**: Bayesian optimization for final refinement

### **Stage 4: Comprehensive Validation**
1. **Statistical Validation**: IC, SNR, Hit Rate, Significance
2. **Economic Validation**: Transaction costs, Sharpe ratio, Drawdown
3. **Microstructure Validation**: Liquidity, Volatility, Depth
4. **Cross-Validation**: Time series CV, Stability analysis
5. **Stability Validation**: Temporal, Regime, Parameter stability

## 📊 **Configuration Options**

### **Model Types**
- `ModelType.ANALYST`: 15m base timeframe, 1-16 periods (15m-240m)
- `ModelType.TACTICIAN`: 5m base timeframe, 1-16 periods (5m-80m)
- `ModelType.BOTH`: Optimize for both models

### **Optimization Methods**
- `OptimizationMethod.GRID_SEARCH`: Grid search only
- `OptimizationMethod.BAYESIAN_TPE`: Bayesian TPE only
- `OptimizationMethod.GRID_BAYESIAN`: Grid + Bayesian TPE (recommended)

### **Validation Levels**
- `ValidationLevel.BASIC`: Basic validation only
- `ValidationLevel.INTERMEDIATE`: Statistical + Economic validation
- `ValidationLevel.ADVANCED`: + Microstructure validation
- `ValidationLevel.COMPREHENSIVE`: All validation types (recommended)

## 🔍 **ml_commons Integration Details**

### **Grid Search Integration**
```python
# Uses ml_commons grid utilities
from src.utils.ml_common.optimization.grid_utils import (
    build_coarse_grid_from_search_space,
    build_fine_grid_around_best
)

# Coarse grid generation
coarse_grid = build_coarse_grid_from_search_space(
    search_space, 
    config.coarse_grid_points
)

# Fine grid around best result
fine_grid = build_fine_grid_around_best(
    search_space,
    best_coarse_params,
    config.fine_grid_points
)
```

### **Bayesian TPE Integration**
```python
# Uses ml_commons HPO utilities
from src.utils.ml_common.utils.hpo_utils import HyperparameterOptimization

# Initialize HPO optimizer
hpo_optimizer = HyperparameterOptimization({
    'enable_parallel': True,
    'max_workers': 4,
    'use_nonlinear_optimization': True
})

# Run Bayesian optimization
result = hpo_optimizer.optimize_hyperparameters(
    objective=objective_function,
    search_space=search_space,
    n_trials=config.n_trials
)
```

### **Validation Integration**
```python
# Uses ml_commons validation utilities
from src.utils.ml_common.validation.unified_validation_system import UnifiedValidationSystem
from src.utils.ml_common.validation.temporal_cross_validation import TemporalCrossValidator
from src.utils.ml_common.validation.cv_utils import CrossValidationUtilities

# Initialize validation system
validation_system = UnifiedValidationSystem()
temporal_cv = TemporalCrossValidator(n_splits=5, gap=1)
cv_utilities = CrossValidationUtilities()

# Comprehensive validation
validation_result = validation_system.validate_model_performance(
    model=None,
    X=market_data,
    y=labeled_data,
    validation_type='comprehensive_validation'
)
```

## 📈 **Performance Features**

### **Memory Optimization**
- Uses `ml_commons.utils.memory_optimization.MemoryOptimizer`
- Automatic memory management for large datasets
- Efficient data processing and caching

### **Lookahead Protection**
- Uses `ml_commons.utils.lookahead_protection.LookaheadProtection`
- Prevents data leakage in time series optimization
- Ensures temporal integrity

### **Parallel Processing**
- Grid search parallelization
- Bayesian TPE parallel trials
- Cross-validation parallel execution

### **Caching System**
- Result caching with TTL
- Performance metrics caching
- Optimization history tracking

## 🚨 **Fast Fail System**

### **Optimization Failure**
```python
if optimization_score < min_optimization_score:
    raise RuntimeError(
        "❌ FAST FAIL: Optimization score below threshold. "
        "Cannot proceed without optimal timeframe discovery. "
        "Training pipeline will terminate."
    )
```

### **Validation Failure**
```python
if validation_score < min_validation_score:
    raise RuntimeError(
        "❌ FAST FAIL: Validation score below threshold. "
        "Cannot proceed with invalid timeframe configuration. "
        "Training pipeline will terminate."
    )
```

## 📊 **Output Format**

### **OptimizationResult**
```python
@dataclass
class OptimizationResult:
    optimal_horizons: Dict[str, int]      # {'immediate': 2, 'short': 8}
    optimal_targets: Dict[str, float] # {'micro': 0.003, 'small': 0.005, ...}
    optimization_score: float             # Overall optimization score
    validation_score: float              # Validation score
    performance_metrics: Dict[str, float] # Detailed performance metrics
    optimization_time: float             # Time taken for optimization
    optimization_method: str             # Method used (grid_bayesian, etc.)
    n_trials: int                        # Number of trials
    convergence_info: Dict[str, Any]     # Convergence information
```

### **ValidationResult**
```python
@dataclass
class ValidationResult:
    validation_score: float               # Overall validation score
    statistical_metrics: Dict[str, float] # Statistical validation metrics
    economic_metrics: Dict[str, float]    # Economic validation metrics
    microstructure_metrics: Dict[str, float] # Microstructure metrics
    cross_validation_metrics: Dict[str, float] # CV metrics
    stability_metrics: Dict[str, float]   # Stability metrics
    overall_quality: str                  # Quality assessment
    recommendations: List[str]            # Improvement recommendations
```

## 🧪 **Testing and Examples**

### **Run Integration Example**
```python
from src.training.steps.market_analysis.optimized_multi_horizon_optimizer.integration_example import (
    run_optimization_example,
    run_analyst_optimization_example,
    run_tactician_optimization_example,
    demonstrate_ml_commons_integration
)

# Run complete example
run_optimization_example()

# Run individual model examples
run_analyst_optimization_example()
run_tactician_optimization_example()

# Demonstrate ml_commons integration
demonstrate_ml_commons_integration()
```

## 🔧 **Integration with Training Pipeline**

### **Replace Existing Optimizer**
```python
# Old approach
from src.training.steps.market_analysis.automatic_timeframe_optimizer import AutomaticTimeframeOptimizer

# New approach
from src.training.steps.market_analysis.optimized_multi_horizon_optimizer import (
    OptimizedTimeframeOptimizer,
    OptimizationConfig,
    ModelType
)

# Initialize with ml_commons integration
config = OptimizationConfig(
    model_type=ModelType.BOTH,
    optimization_method=OptimizationMethod.GRID_BAYESIAN
)
optimizer = OptimizedTimeframeOptimizer(config)
```

### **Pipeline Integration**
```python
# In training pipeline
def execute_multi_horizon_labeling_step(self, data, config, model_type="both"):
    # Initialize optimized optimizer
    optimization_config = OptimizationConfig(
        model_type=ModelType.BOTH if model_type == "both" else ModelType.ANALYST if model_type == "analyst" else ModelType.TACTICIAN,
        optimization_method=OptimizationMethod.GRID_BAYESIAN
    )
    
    optimizer = OptimizedTimeframeOptimizer(optimization_config)
    
    # Run optimization
    results = optimizer.get_optimal_timeframes_for_models(
        market_data=data,
        model_type=optimization_config.model_type
    )
    
    # Use optimized configuration
    for model_type, result in results.items():
        optimized_config = MultiHorizonConfig()
        optimized_config.time_horizons = result.optimal_horizons
        optimized_config.profit_targets = result.optimal_targets
        
        # Continue with labeling using optimized config
        # ...
```

## 📊 **Performance Metrics**

### **Optimization Metrics**
- **Optimization Score**: Overall optimization performance
- **Validation Score**: Comprehensive validation score
- **Convergence Info**: Optimization convergence details
- **Trial Count**: Number of optimization trials

### **Validation Metrics**
- **Statistical**: IC, SNR, Hit Rate, Significance
- **Economic**: Transaction costs, Sharpe ratio, Drawdown
- **Microstructure**: Liquidity, Volatility, Depth
- **Cross-Validation**: CV score, Stability
- **Stability**: Temporal, Regime, Parameter stability

## 🎯 **Best Practices**

### **Configuration**
1. **Use GRID_BAYESIAN**: Best combination of grid search + Bayesian TPE
2. **Set Appropriate Ranges**: 1-16 periods for both Analyst and Tactician
3. **Enable Comprehensive Validation**: Use ValidationLevel.COMPREHENSIVE
4. **Enable Parallel Processing**: Set max_workers appropriately

### **Performance**
1. **Use Caching**: Enable result caching for repeated optimizations
2. **Monitor Memory**: Use memory optimization for large datasets
3. **Enable Fast Fail**: Prevent suboptimal training with fast fail
4. **Export Results**: Save optimization results for analysis

### **Integration**
1. **Replace Old Optimizer**: Use OptimizedTimeframeOptimizer instead of AutomaticTimeframeOptimizer
2. **Leverage ml_commons**: Take advantage of extensive ml_commons integration
3. **Monitor Performance**: Use optimization monitoring and reporting
4. **Handle Failures**: Implement proper error handling and fast fail

## 🚀 **Future Enhancements**

### **Planned Features**
- [ ] **Adaptive Optimization**: Continuous optimization based on performance
- [ ] **Multi-Objective Optimization**: Pareto frontier optimization
- [ ] **Ensemble Methods**: Combine multiple optimization strategies
- [ ] **Distributed Optimization**: Scale across multiple machines
- [ ] **Advanced Analytics**: Comprehensive optimization analytics

### **ml_commons Integration**
- [ ] **Additional Validation**: More validation utilities from ml_commons
- [ ] **Advanced HPO**: More sophisticated HPO strategies
- [ ] **Performance Monitoring**: Real-time optimization monitoring
- [ ] **Result Visualization**: Optimization result visualization

## 📝 **Conclusion**

The Optimized Multi-Horizon Optimizer provides a comprehensive, production-ready solution for timeframe optimization that leverages ml_commons utilities extensively. It offers:

1. **Advanced Optimization**: Grid search + Bayesian TPE optimization
2. **Comprehensive Validation**: Statistical, economic, microstructure, and stability validation
3. **ml_commons Integration**: Extensive use of existing ml_commons utilities
4. **Production Ready**: Fast fail, caching, monitoring, and error handling
5. **Easy Integration**: Simple API for integration with training pipeline

This system represents a significant improvement over the previous optimization approach, providing better performance, more comprehensive validation, and extensive integration with the existing ml_commons infrastructure.
