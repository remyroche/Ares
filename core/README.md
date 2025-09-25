# Core Tree Architecture Search & Search Space Implementation

This directory contains comprehensive implementations of Tree Architecture Search (TAS) and Neural Architecture Search (NAS) with full integration to shared utilities from `src/utils/`.

## 📁 Files Completed

### 1. `tree_architecture_search.py` ✅
- **Purpose**: Advanced Tree Architecture Search for tree-based models (Random Forest, XGBoost, LightGBM, etc.)
- **Integration**: Uses shared utilities from `src/utils/`
- **Features**:
  - Grid + TPE optimization strategy
  - M1 hardware optimization support
  - Evolutionary and Bayesian search methods
  - Comprehensive evaluation metrics (accuracy, efficiency, interpretability)
  - Result saving and serialization
  - Memory optimization

### 2. `search_space.py` ✅
- **Purpose**: Comprehensive search space management for NAS/TAS
- **Integration**: Uses shared utilities from `src/utils/`
- **Features**:
  - Multiple optimization strategies (Grid, TPE, Grid+TPE, Random, Hierarchical)
  - Parameter validation and constraint checking
  - Hardware acceleration (M1 GPU, Memory, CPU optimization)
  - Matrix operations optimization
  - Comprehensive result tracking and serialization
  - Memory management and parallel processing

## 🔧 Shared Utilities Integration

Both modules integrate extensively with shared utilities:

### Core Operations (`src/utils/common_operations.py`)
- `safe_json_dump()`, `safe_json_load()`: Safe file I/O
- `ensure_directory()`: Directory creation
- `safe_divide()`, `safe_log()`, `safe_sqrt()`: Safe mathematical operations
- `validate_finite()`, `validate_positive()`: Data validation
- `get_current_datetime()`: Timestamp utilities
- `optimize_dataframe_dtypes()`: DataFrame optimization

### Enhanced Printing (`src/utils/tprint.py`)
- `tprint_info()`, `tprint_success()`, `tprint_warning()`, `tprint_error()`: Structured logging
- `tprint_performance()`: Performance timing
- `tprint_structured()`: Structured data display

### Mathematical Validation (`src/utils/math_validation.py`)
- `safe_correlation()`, `safe_mean()`, `safe_std()`: Safe statistical operations
- `safe_percentile()`: Safe percentile calculations
- `validate_correlation_matrix()`: Matrix validation

### Serialization (`src/utils/serialization_utils.py`)
- `JSONSerializer`: JSON serialization
- `UniversalSerializer`: Multi-format serialization

### Hardware Optimization (`src/utils/hardware/`)
- **M1 GPU Utils**: GPU acceleration for M1 systems
- **M1 Memory Optimizer**: Memory optimization
- **M1 CPU Optimizer**: CPU optimization
- **Matrix Operations**: Optimized matrix computations

### ML Optimization (`src/utils/ml_common/optimization/`)
- **Bayesian TPE Optimizer**: Tree-structured Parzen Estimator optimization
- **Grid Utils**: Grid search utilities
- **Cross-validation**: Unified cross-validation framework

## 🚀 Key Features

### Tree Architecture Search
1. **Multi-Strategy Optimization**:
   - Grid Search: Systematic exploration
   - TPE (Tree-structured Parzen Estimator): Bayesian optimization
   - Grid + TPE: Combined strategy (30% grid, 70% TPE)
   - Random: Baseline comparison
   - Evolutionary: Genetic algorithm approach

2. **Hardware Integration**:
   - M1 GPU acceleration when available
   - Memory optimization for large datasets
   - CPU optimization for training

3. **Comprehensive Evaluation**:
   - Accuracy scoring
   - Efficiency metrics (training time, model size)
   - Interpretability scoring
   - Multi-objective optimization

### Search Space Management
1. **Parameter Types**:
   - Continuous parameters with optional log scaling
   - Discrete parameters with step sizes
   - Categorical parameters with choices
   - Mixed parameter spaces

2. **Optimization Strategies**:
   - Grid + TPE: Best of both worlds
   - Pure Grid: Systematic exploration
   - Pure TPE: Bayesian optimization
   - Random: Baseline
   - Hierarchical: Advanced search structures

3. **Validation & Constraints**:
   - Parameter validation with detailed error reporting
   - Constraint checking
   - Hardware-optimized parameter sampling

## 📊 Usage Examples

### Basic Tree Architecture Search
```python
from core.tree_architecture_search import TreeArchitectureSearch, TreeArchitectureConfig
import numpy as np

# Configure search
config = TreeArchitectureConfig(
    n_trials=50,
    optimization_strategy="grid_tpe",
    enable_m1_optimization=True
)

# Create search instance
search = TreeArchitectureSearch(config)

# Run search
X_train = np.random.randn(1000, 20)
y_train = np.random.randn(1000)
best_candidate = search.search(X_train, y_train)

print(f"Best score: {best_candidate.overall_score:.4f}")
print(f"Best parameters: n_trees={best_candidate.n_trees}, depth={best_candidate.max_depth}")
```

### Advanced Search Space
```python
from core.search_space import SearchSpace, SearchSpaceConfig, OptimizationStrategy

# Configure search space
config = SearchSpaceConfig(
    name="custom_nas_space",
    optimization_strategy=OptimizationStrategy.GRID_TPE,
    enable_m1_optimization=True,
    grid_phase_ratio=0.3,
    tpe_phase_ratio=0.7
)

# Create search space
search_space = SearchSpace(config)

# Add parameters
search_space.add_continuous_parameter("learning_rate", 1e-5, 1e-1, log_scale=True)
search_space.add_discrete_parameter("hidden_layers", 1, 10)
search_space.add_categorical_parameter("activation", ["relu", "tanh", "gelu"])

# Define objective function
def objective(params):
    # Your model training and evaluation logic here
    return score

# Run optimization
results = search_space.optimize(objective)
print(f"Best score: {results['best_score']}")
```

### Hardware-Optimized NAS
```python
from core.search_space import create_default_nas_search_space

# Create default NAS space with M1 optimization
search_space = create_default_nas_search_space()

# The search space automatically:
# - Enables M1 GPU acceleration
# - Optimizes memory usage
# - Uses Grid + TPE strategy
# - Saves comprehensive results

# Summary shows hardware status
summary = search_space.get_summary()
print(summary['hardware_status'])
```

## 🎯 Optimization Strategies

### Grid + TPE (Recommended)
- **Phase 1 (30%)**: Grid search for systematic exploration
- **Phase 2 (70%)**: TPE for exploitation around promising regions
- **Best of both worlds**: Exploration + exploitation

### Grid Search
- Systematic exploration of parameter space
- Guaranteed coverage of search space
- Good for understanding parameter relationships

### TPE (Tree-structured Parzen Estimator)
- Bayesian optimization
- Efficient for expensive evaluations
- Adaptive sampling based on previous results

### Hierarchical
- Advanced search for complex architectures
- Multi-level optimization
- Future expansion capability

## 🔧 Dependencies

### Required
- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `scikit-learn`: Machine learning models

### Optional (for enhanced features)
- `optuna`: Advanced TPE optimization
- `xgboost`: XGBoost models
- `lightgbm`: LightGBM models

### Hardware Optimization (M1 Systems)
- Apple Silicon optimizations
- Metal Performance Shaders (MPS)
- Unified memory optimization

## 💾 Results and Serialization

Both modules provide comprehensive result saving:

```python
# Results are automatically saved to:
# - tree_search_results/ (Tree Architecture Search)
# - search_space_results/ (Search Space)

# Results include:
# - Configuration parameters
# - Search space definition
# - Optimization history
# - Best results
# - Hardware information
# - Timestamps and metadata
```

## 🧪 Testing

Run the demonstration:
```bash
python3 core/demo_usage.py
```

The modules handle missing dependencies gracefully and provide informative error messages.

## 🔄 Integration Points

The modules are designed to integrate with:

1. **Existing NAS/TAS systems** in the codebase
2. **Trading algorithms** that need architecture optimization
3. **Model training pipelines** requiring hyperparameter tuning
4. **Cross-validation frameworks** in `src/utils/ml_common/`
5. **Hardware acceleration** systems

## 📈 Performance Features

- **Memory optimization**: Efficient handling of large parameter spaces
- **Parallel processing**: Multi-core optimization when available
- **Hardware acceleration**: M1 GPU/CPU optimization
- **Early stopping**: Prevent unnecessary computation
- **Incremental saving**: Results saved during optimization
- **Graceful error handling**: Robust to evaluation failures

## 🎉 Summary

The completed modules provide:

✅ **Comprehensive TAS/NAS implementations**  
✅ **Full shared utilities integration**  
✅ **Grid + Bayesian TPE optimization**  
✅ **M1 hardware optimization**  
✅ **Robust error handling**  
✅ **Extensive documentation**  
✅ **Production-ready code**  

These implementations are ready for production use and provide a solid foundation for architecture search in the hybrid NAS/TAS system.