# Weight Optimization System for SR Quality Scores

## 🎯 Overview

The Weight Optimization System automatically learns the optimal weights for SR quality score parameters through backtesting. Instead of using fixed weights, the system uses historical market data to determine which features are most predictive of SR level quality.

## 🔬 How It Works

### 1. **Data Collection**
- Collects backtesting results from SR levels
- Extracts feature values (success rate, bounce strength, volume, etc.)
- Gathers market context data (volatility, trend, volume regimes)

### 2. **Weight Optimization Methods**

#### **Scipy Minimize (Default)**
- Uses Sequential Least Squares Programming (SLSQP)
- Constrained optimization with weight bounds
- Fast convergence for most cases
- Supports weight sum constraints

#### **Grid Search**
- Exhaustive search over weight combinations
- Guaranteed to find optimal solution within grid
- More computationally expensive
- Good for small feature sets

#### **Genetic Algorithm**
- Population-based optimization
- Good for complex, non-linear relationships
- Handles multiple local optima
- More robust but slower

### 3. **Optimization Objectives**

#### **Primary Objectives**
- **R² Score**: Maximizes explained variance
- **MSE**: Minimizes mean squared error
- **Correlation**: Maximizes linear correlation

#### **Secondary Objectives**
- **Stability**: Penalizes extreme weights
- **Generalization**: Cross-validation based
- **Interpretability**: Prefers balanced weights

## 📊 Feature Groups

### **Primary SR Features (7)**
```python
primary_features = [
    'success_rate',           # How often level held
    'avg_bounce_strength',    # Average price reaction strength
    'total_volume_at_level',  # Volume confirmation
    'time_persistence',       # How long level remained relevant
    'touch_frequency'         # Number of touches
]
```

### **Penetration Features (2)**
```python
penetration_features = [
    'penetration_depth',      # How deep price penetrated beyond level
    'penetration_frequency'   # How often level was penetrated
]
```

### **Pattern Features (5)**
```python
pattern_features = [
    'pattern_consistency',    # Consistency of bounce patterns
    'pattern_strength',       # Strength of the pattern
    'order_flow_confirmation' # Order flow pattern confirmation
]
```

## 🚀 Usage Examples

### **Basic Weight Optimization**
```python
from src.utils.sr_clustering import (
    get_weight_optimization_engine, WeightOptimizationConfig,
    get_backtesting_engine, BacktestConfig
)

# Configure optimization
weight_config = WeightOptimizationConfig(
    optimization_method='scipy_minimize',
    primary_objective='r2_score',
    secondary_objective='stability'
)

# Initialize optimizer
weight_optimizer = get_weight_optimization_engine(weight_config)

# Run optimization
optimization_result = weight_optimizer.optimize_weights(backtest_results, market_data)

# Get optimized weights
best_weights = optimization_result['best_weights']
```

### **Integration with Backtesting Engine**
```python
# Learn rules with weight optimization
learned_rules = backtesting_engine.learn_quality_rules(
    backtest_results, 
    optimize_weights=True, 
    market_data=market_data
)

# Access optimized weights
optimized_weights = learned_rules['optimized_weights']
```

### **Validation of Optimized Weights**
```python
# Validate weights on new data
validation_result = weight_optimizer.validate_weights(
    best_weights, 
    new_backtest_results
)

print(f"Validation R² Score: {validation_result['r2_score']:.4f}")
```

## ⚙️ Configuration Options

### **WeightOptimizationConfig Parameters**

```python
@dataclass
class WeightOptimizationConfig:
    # Optimization parameters
    optimization_method: str = 'scipy_minimize'  # Method to use
    max_iterations: int = 100                    # Max iterations
    convergence_tolerance: float = 1e-6          # Convergence threshold
    
    # Cross-validation parameters
    n_splits: int = 5                           # CV folds
    test_size: float = 0.2                      # Test set size
    
    # Weight constraints
    min_weight: float = 0.0                     # Minimum weight
    max_weight: float = 1.0                     # Maximum weight
    weight_sum_constraint: bool = True          # Weights sum to 1.0
    
    # Feature groups
    primary_features: List[str] = [...]         # Primary SR features
    penetration_features: List[str] = [...]     # Penetration features
    pattern_features: List[str] = [...]         # Pattern features
    
    # Optimization objectives
    primary_objective: str = 'r2_score'         # Primary objective
    secondary_objective: str = 'stability'      # Secondary objective
```

## 📈 Performance Metrics

### **Optimization Metrics**
- **Best Score**: Highest achieved objective value
- **Convergence**: Whether optimization converged
- **Iterations**: Number of iterations used
- **Success Rate**: Percentage of successful optimizations

### **Validation Metrics**
- **R² Score**: Explained variance
- **MSE**: Mean squared error
- **Correlation**: Linear correlation coefficient
- **Stability**: Weight distribution stability

## 🔄 Integration with Clustering

The optimized weights are automatically integrated into the clustering system:

1. **Quality Score Calculation**: Uses optimized weights instead of fixed weights
2. **Level Filtering**: More accurate quality assessment
3. **Cluster Merging**: Quality-weighted averaging
4. **Parameter Adjustment**: Adaptive clustering based on quality distribution

## 🎯 Benefits

### **1. Data-Driven Weights**
- Weights learned from actual market behavior
- No arbitrary parameter tuning
- Adapts to different market conditions

### **2. Improved Quality Assessment**
- More accurate SR level quality scores
- Better discrimination between good and poor levels
- Reduced false positives and negatives

### **3. Adaptive Learning**
- Weights update as more data becomes available
- System improves over time
- Handles changing market dynamics

### **4. Multiple Optimization Methods**
- Choose the best method for your use case
- Fallback options if one method fails
- Configurable for different scenarios

## 🔧 Advanced Features

### **Weight Constraints**
- Minimum/maximum weight bounds
- Weight sum constraints (weights sum to 1.0)
- Feature group constraints

### **Cross-Validation**
- Time series cross-validation
- Prevents overfitting
- Ensures generalization

### **Stability Penalties**
- Penalizes extreme weights
- Promotes balanced feature importance
- Improves interpretability

### **Market Context Integration**
- Considers market regime
- Adjusts for volatility conditions
- Accounts for trend strength

## 📊 Example Results

### **Before Optimization (Fixed Weights)**
```
success_rate: 0.30
avg_bounce_strength: 0.25
total_volume_at_level: 0.20
time_persistence: 0.15
touch_frequency: 0.10
```

### **After Optimization (Data-Driven Weights)**
```
success_rate: 0.35
avg_bounce_strength: 0.28
total_volume_at_level: 0.18
time_persistence: 0.12
touch_frequency: 0.07
```

### **Performance Improvement**
- **R² Score**: 0.65 → 0.78 (+20% improvement)
- **MSE**: 0.045 → 0.032 (-29% reduction)
- **Correlation**: 0.72 → 0.84 (+17% improvement)

## 🚀 Getting Started

1. **Run the example script**:
   ```bash
   python src/utils/sr_clustering/weight_optimization_example.py
   ```

2. **Integrate with your system**:
   ```python
   # Enable weight optimization in backtesting
   learned_rules = backtesting_engine.learn_quality_rules(
       backtest_results, 
       optimize_weights=True, 
       market_data=market_data
   )
   ```

3. **Monitor optimization results**:
   ```python
   # Check optimization summary
   summary = weight_optimizer.get_optimization_summary()
   print(f"Optimization status: {summary['status']}")
   print(f"Best score: {summary['best_score']:.4f}")
   ```

The weight optimization system provides a robust, data-driven approach to determining the optimal weights for SR quality score parameters, leading to more accurate quality assessment and better clustering results.