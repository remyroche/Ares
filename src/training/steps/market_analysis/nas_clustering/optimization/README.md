# NAS Optimization Framework with Grid Utils and Hardware Integration

This module provides comprehensive Bayesian optimization for NAS parameters using existing grid utilities, matrix operations, and hardware optimization tools.

## 🎯 **Key Features**

### **1. Grid Utils Integration**
- **Coarse-to-fine optimization** using existing `grid_utils.py`
- **Automatic parameter space exploration** before TPE
- **Fine-tuning around best parameters** discovered in coarse grid
- **Configurable grid points** for different optimization phases

### **2. Matrix Operations Integration**
- **Unified matrix operations** using existing `matrix_operations.py`
- **Hardware-optimized computations** for large-scale feature processing
- **Batch processing** for efficient memory usage
- **GPU acceleration** when available

### **3. Hardware Optimization Integration**
- **Unified hardware manager** using existing `unified_hardware_manager.py`
- **CPU, GPU, and memory optimization** for ML training workloads
- **Adaptive optimization** based on workload characteristics
- **Performance monitoring** and automatic tuning

### **4. Multi-Objective Optimization**
- **Regime stability** optimization
- **Economic significance** maximization
- **Trading viability** assessment
- **Micro-regime detection accuracy** improvement

## 🚀 **Usage Examples**

### **Quick Start: Short-term Trading Optimization**
```python
from nas_clustering.optimization import NASOptimizationIntegration, NASOptimizationConfig

# Create configuration for short-term trading
config = NASOptimizationConfig.create_short_term_trading_config()

# Create optimization integration
optimizer = NASOptimizationIntegration(config)

# Run optimization
results = optimizer.run_optimization(
    market_data=market_data,
    features=features,
    timestamps=timestamps,
    nas_config=nas_config,
    save_path="nas_optimization_results"
)
```

### **High Performance Optimization**
```python
# Create configuration for high performance
config = NASOptimizationConfig.create_high_performance_config()

# Create optimization integration
optimizer = NASOptimizationIntegration(config)

# Run optimization with aggressive hardware optimization
results = optimizer.run_optimization(...)
```

### **Custom Configuration**
```python
# Create custom configuration
config = NASOptimizationConfig(
    optimization_strategy=OptimizationStrategy.HYBRID,
    grid_config=NASGridOptimizationConfig(
        enable_coarse_grid=True,
        enable_fine_grid=True,
        coarse_grid_points=10,
        fine_grid_points=6,
        grid_phase_trials=40
    ),
    hardware_config=NASHardwareOptimizationConfig(
        enable_hardware_optimization=True,
        optimization_level=OptimizationLevel.AGGRESSIVE,
        memory_limit_gb=16.0
    ),
    bayesian_config=NASBayesianOptimizationConfig(
        n_trials=150,
        objectives=['regime_stability', 'economic_significance', 'trading_viability'],
        objective_weights=[0.4, 0.3, 0.3]
    )
)
```

## 📊 **Optimization Strategy**

### **Phase 1: Grid Search (Optional)**
- **Coarse grid exploration** using `build_coarse_grid_from_search_space()`
- **Parameter space coverage** for initial exploration
- **Fine grid refinement** using `build_fine_grid_around_best()`
- **Best parameter identification** for TPE initialization

### **Phase 2: Bayesian Optimization**
- **TPE (Tree-structured Parzen Estimator)** for efficient parameter search
- **Multi-objective optimization** with weighted objectives
- **Pruning and early stopping** for efficiency
- **Convergence analysis** and recommendations

## 🔧 **Configuration Options**

### **Grid Optimization**
```python
grid_config = NASGridOptimizationConfig(
    enable_coarse_grid=True,          # Enable coarse grid search
    enable_fine_grid=True,            # Enable fine grid refinement
    coarse_grid_points=8,             # Coarse grid resolution
    fine_grid_points=5,               # Fine grid resolution
    grid_phase_trials=30,             # Number of grid trials
    fine_grid_around_best=True,      # Fine-tune around best parameters
    fine_grid_range_factor=0.2       # 20% range around best parameters
)
```

### **Matrix Operations**
```python
matrix_config = NASMatrixOptimizationConfig(
    enable_matrix_optimization=True,  # Enable matrix operations optimization
    enable_batch_processing=True,     # Enable batch processing
    batch_size=1000,                 # Batch size for processing
    enable_gpu_acceleration=True,    # Enable GPU acceleration
    enable_parallel_processing=True  # Enable parallel processing
)
```

### **Hardware Optimization**
```python
hardware_config = NASHardwareOptimizationConfig(
    enable_hardware_optimization=True,    # Enable hardware optimization
    workload_type=WorkloadType.ML_TRAINING, # Workload type
    optimization_level=OptimizationLevel.BALANCED, # Optimization level
    memory_limit_gb=8.0,                 # Memory limit
    enable_adaptive_optimization=True,   # Enable adaptive optimization
    learning_enabled=True,               # Enable learning
    auto_tuning_enabled=True            # Enable auto-tuning
)
```

### **Bayesian Optimization**
```python
bayesian_config = NASBayesianOptimizationConfig(
    enable_tpe_optimization=True,     # Enable TPE optimization
    n_trials=100,                     # Number of trials
    n_startup_trials=20,              # Startup trials
    objectives=['regime_stability', 'economic_significance', 'trading_viability'],
    objective_weights=[0.3, 0.3, 0.2, 0.2],  # Objective weights
    enable_pruning=True,              # Enable pruning
    pruning_patience=10              # Pruning patience
)
```

## 🎯 **NAS Parameters Optimized**

### **Strategic Parameters (Set at Business Level)**
- `timeframe`: Primary trading timeframe
- `micro_timeframe`: Micro-regime detection timeframe
- `n_regimes`: Target number of regimes
- `nas_architecture_type`: Architecture focus (HYBRID, VOLATILITY_FOCUSED, etc.)
- `enable_micro_regime_detection`: Enable micro-regime detection

### **Optimization Parameters (Bayesian Framework)**
- `architecture_depth`: Network depth (3-9)
- `hidden_units`: Hidden layer sizes (32-256)
- `activation_function`: Activation functions (relu, tanh, swish, gelu)
- `dropout_rate`: Dropout rates (0.1-0.4)
- `learning_rate`: Learning rates (0.001-0.1)
- `micro_regime_sensitivity`: Micro-regime detection sensitivity (0.5-0.9)
- `economic_significance_threshold`: Economic significance threshold (0.5-0.9)
- `trading_viability_threshold`: Trading viability threshold (0.4-0.8)
- `regime_transition_cost`: Regime transition cost (0.01-0.1)

## 📈 **Performance Benefits**

### **Grid Utils Integration**
- **Faster convergence** through intelligent parameter space exploration
- **Better initial parameters** for TPE optimization
- **Reduced search space** for Bayesian optimization
- **Coarse-to-fine refinement** for parameter tuning

### **Matrix Operations Integration**
- **Hardware-optimized computations** for large-scale feature processing
- **Batch processing** for efficient memory usage
- **GPU acceleration** when available
- **Unified matrix operations** across different backends

### **Hardware Optimization Integration**
- **Automatic hardware tuning** based on workload characteristics
- **CPU, GPU, and memory optimization** for ML training
- **Adaptive optimization** that learns from performance patterns
- **Performance monitoring** and automatic resource management

## 🔍 **Optimization Results**

### **Performance Metrics**
- `best_score`: Best optimization score achieved
- `best_params`: Best parameter combination
- `execution_time`: Total optimization time
- `n_trials`: Number of trials completed
- `convergence_analysis`: Convergence patterns and analysis

### **Hardware Metrics**
- `cpu_usage`: CPU utilization during optimization
- `memory_usage`: Memory utilization during optimization
- `gpu_usage`: GPU utilization during optimization
- `temperature`: System temperature during optimization
- `power_consumption`: Power consumption during optimization

### **Matrix Operations Metrics**
- `matrix_operations_available`: Matrix operations backend availability
- `optimization_enabled`: Matrix operations optimization status
- `batch_processing_enabled`: Batch processing status
- `gpu_acceleration_enabled`: GPU acceleration status

## 🚀 **Getting Started**

1. **Import the optimization framework**:
   ```python
   from nas_clustering.optimization import NASOptimizationIntegration, NASOptimizationConfig
   ```

2. **Create configuration**:
   ```python
   config = NASOptimizationConfig.create_short_term_trading_config()
   ```

3. **Create optimizer**:
   ```python
   optimizer = NASOptimizationIntegration(config)
   ```

4. **Run optimization**:
   ```python
   results = optimizer.run_optimization(
       market_data=market_data,
       features=features,
       timestamps=timestamps,
       nas_config=nas_config,
       save_path="optimization_results"
   )
   ```

5. **Analyze results**:
   ```python
   print(f"Best score: {results['optimization_results']['best_overall_score']}")
   print(f"Best parameters: {results['optimization_results']['best_overall_params']}")
   print(f"Recommendations: {results['recommendations']}")
   ```

## 📚 **Additional Resources**

- **Grid Utils**: `src/utils/ml_common/optimization/grid_utils.py`
- **Matrix Operations**: `src/utils/matrix_operations.py`
- **Hardware Manager**: `src/utils/hardware/unified_hardware_manager.py`
- **NAS Configuration**: `nas_clustering/core/nas_config.py`
- **Example Usage**: `nas_clustering/optimization/example_usage.py`

This framework provides a comprehensive solution for NAS parameter optimization while leveraging existing tools and infrastructure for maximum efficiency and performance.