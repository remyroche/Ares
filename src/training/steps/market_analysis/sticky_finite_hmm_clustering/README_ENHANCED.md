# Enhanced Sticky Finite HMM Clustering

This document describes the enhanced capabilities added to the Sticky Finite HMM clustering system.

## 🚀 New Enhanced Features

### 1. Enhanced Standalone Runner (`enhanced_standalone_runner.py`)

**2-Stage Auto-Tuning:**
- **Stage 1**: Coarse grid search across broad parameter space
- **Stage 2**: Fine grid search around best parameters from Stage 1
- Configurable number of trials per stage
- Automatic parameter space refinement

**Multi-Objective Optimization:**
- Pareto front analysis for multiple objectives
- Knee point selection for optimal trade-offs
- Support for composite scoring, silhouette score, transition persistence
- Configurable objective directions (maximize/minimize)

**Quality Assessor Integration:**
- Full integration with `ClusterQualityAssessor`
- Comprehensive quality metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)
- Composite scoring with weighted objectives
- Quality-based parameter selection

**KPI Tracking:**
- Performance metrics monitoring
- Success rate tracking
- Trials per second measurement
- Optimization time analysis
- Stage-by-stage performance breakdown

### 2. Enhanced Main Clusterer (`sticky_finite_hmm_clusterer.py`)

**Natural Gradient Updates:**
- Fisher information matrix preconditioning
- Reduced variance in gradient updates
- Configurable natural gradient learning rate
- Frequency-based application (every N iterations)

**Rao-Blackwellization:**
- Exact sufficient statistics computation
- Analytical integration of transition matrix parameters
- Reduced sampling variance
- Quality monitoring and diagnostics

**Vectorized Computations:**
- GPU/CPU parallel processing optimization
- Mac M1 MPS support
- CUDA backend optimization
- Memory management improvements

**Enhanced Configuration:**
```python
@dataclass
class StickyFiniteHMMConfig:
    # ... existing parameters ...
    
    # Enhanced SVI Features
    enable_natural_gradients: bool = True
    enable_rao_blackwellization: bool = True
    enable_vectorization: bool = True
    natural_gradient_lr: float = 0.5
    rao_blackwell_samples: int = 100
    natural_gradient_frequency: int = 5
```

## 📊 Usage Examples

### Basic Enhanced Auto-Tuning
```python
from src.training.steps.market_analysis.sticky_finite_hmm_clustering.enhanced_standalone_runner import (
    run_sticky_finite_hmm_with_auto_tuning
)

result = run_sticky_finite_hmm_with_auto_tuning(
    market_data=your_data,
    symbol="ETHUSDT",
    optimization_stages=2,  # grid -> fine grid
    use_multi_objective=False,
    max_trials_per_stage=50,
    enable_kpi_tracking=True
)
```

### Multi-Objective Optimization
```python
result = run_sticky_finite_hmm_with_auto_tuning(
    market_data=your_data,
    symbol="ETHUSDT",
    optimization_stages=2,
    use_multi_objective=True,
    objectives=["composite_score", "silhouette_score", "transition_persistence"],
    max_trials_per_stage=30
)

# Access Pareto solutions
for solution in result.pareto_solutions:
    print(f"Score: {solution.score:.4f}, Objectives: {solution.objectives}")
```

### Enhanced SVI Features
```python
from src.training.steps.market_analysis.sticky_finite_hmm_clustering.sticky_finite_hmm_clusterer import (
    StickyFiniteHMMClusterer,
    StickyFiniteHMMConfig
)

config = StickyFiniteHMMConfig(
    K=5,
    base_alpha=0.5,
    kappa=10.0,
    
    # Enhanced SVI Features
    enable_natural_gradients=True,
    enable_rao_blackwellization=True,
    enable_vectorization=True,
    natural_gradient_lr=0.5,
    natural_gradient_frequency=5
)

clusterer = StickyFiniteHMMClusterer(config)
result = clusterer.fit_predict(market_data)
```

## 🔧 Configuration Options

### AutoTuningConfig
```python
@dataclass
class AutoTuningConfig:
    optimization_stages: int = 2
    use_multi_objective: bool = False
    objectives: List[str] = ["composite_score"]
    max_trials_per_stage: int = 50
    timeout_seconds: int = 1800
    enable_kpi_tracking: bool = True
    save_results: bool = True
    grid_resolution: int = 3
    fine_grid_factor: float = 0.2
```

### Enhanced StickyFiniteHMMConfig
```python
# Enhanced SVI Features
enable_natural_gradients: bool = True
enable_rao_blackwellization: bool = True
enable_vectorization: bool = True
natural_gradient_lr: float = 0.5
rao_blackwell_samples: int = 100
natural_gradient_frequency: int = 5
```

## 📈 Performance Benefits

### Natural Gradients
- **Reduced Variance**: More stable gradient updates
- **Faster Convergence**: Fewer iterations needed
- **Better Local Optima**: Improved parameter discovery

### Rao-Blackwellization
- **Exact Statistics**: Analytical integration where possible
- **Reduced Sampling Error**: Lower variance in estimates
- **Better Posterior Estimates**: More accurate inference

### Vectorization
- **GPU Acceleration**: Leverage modern hardware
- **Parallel Processing**: Handle larger datasets efficiently
- **Memory Optimization**: Better resource utilization

### 2-Stage Auto-Tuning
- **Broad Exploration**: Coarse grid finds promising regions
- **Fine Refinement**: Detailed search around best parameters
- **Efficient Resource Use**: Optimal trial allocation

## 🧪 Testing and Validation

Run the test suite to verify enhanced features:

```bash
# Test enhanced features availability
python src/training/steps/market_analysis/sticky_finite_hmm_clustering/examples/test_enhanced_features.py

# Test enhanced auto-tuning (with sample data)
python src/training/steps/market_analysis/sticky_finite_hmm_clustering/examples/simple_enhanced_demo.py
```

## 📊 Results and Metrics

### KPI Metrics Available
- **Success Rate**: Percentage of successful trials
- **Trials per Second**: Optimization throughput
- **Optimization Time**: Total time spent
- **Best Score**: Highest objective value found
- **Convergence Rate**: Speed of improvement

### Quality Metrics
- **Silhouette Score**: Cluster separation quality
- **Davies-Bouldin Index**: Cluster compactness
- **Calinski-Harabasz Index**: Cluster dispersion
- **Transition Persistence**: Temporal stability
- **Composite Score**: Weighted combination

## 🔍 Integration Details

### Dependencies
- Pyro for probabilistic programming
- PyTorch for tensor operations
- Scikit-learn for quality metrics
- NumPy for numerical operations
- Pandas for data handling

### Backward Compatibility
- All existing functionality preserved
- Enhanced features are opt-in via configuration
- Original API remains unchanged
- Graceful fallbacks for missing dependencies

## 🚀 Future Enhancements

### Planned Features
- **Advanced Natural Gradients**: Full Fisher matrix implementation
- **Hierarchical Rao-Blackwellization**: Multi-level integration
- **Adaptive Vectorization**: Dynamic resource allocation
- **Bayesian Optimization**: TPE integration for fine-tuning
- **Distributed Computing**: Multi-GPU and cluster support

### Performance Optimizations
- **Memory Mapping**: Large dataset handling
- **Incremental Learning**: Online parameter updates
- **Caching**: Smart result caching
- **Early Stopping**: Intelligent convergence detection

## 📝 Examples and Documentation

See the `examples/` directory for complete working examples:
- `test_enhanced_features.py` - Feature availability testing
- `simple_enhanced_demo.py` - Basic enhanced features demo
- `enhanced_auto_tuning_example.py` - Comprehensive auto-tuning demo

## 🎯 Best Practices

1. **Start with Default Settings**: Enhanced features work well out-of-the-box
2. **Monitor KPI Metrics**: Use built-in tracking for optimization insights
3. **Compare Results**: Run both standard and enhanced configurations
4. **Adjust Frequency**: Tune natural gradient frequency for your data size
5. **Use Multi-Objective**: Balance multiple quality metrics for robustness

## 📞 Support

For questions or issues with enhanced features:
1. Check the test outputs for feature availability
2. Review configuration options in the code
3. Monitor KPI metrics during optimization
4. Compare with baseline results for validation
