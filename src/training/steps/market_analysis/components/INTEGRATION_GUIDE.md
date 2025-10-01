# Integration Guide for Refactored Modules

## Overview

This guide shows how to integrate the new refactored modules with existing code. The refactoring provides focused, maintainable modules while maintaining backward compatibility.

## New Modules Created

### 1. Configuration Module (`clustering_config.py`)
Centralized configuration management with validation.

### 2. Memory Manager (`memory_manager.py`)
Advanced memory management with monitoring and cleanup.

### 3. Clustering Algorithms (`clustering_algorithms.py`)
Specialized clustering algorithms with consistent interface.

### 4. Import Manager (`imports.py`)
Centralized import management with fallback mechanisms.

## Modified Files

### 1. Label Fusion (`regime_analysis/label_fusion.py`)
Completely refactored with better separation of concerns:
- `LabelMappingService` - Handles label mapping
- `DawidSkeneService` - Implements EM algorithm
- `LabelFusionService` - Main orchestration

### 2. Service (`regime_analysis/service.py`)
Fixed missing `os` import.

## Quick Start Examples

### Using Configuration Module

```python
from src.training.steps.market_analysis.components.clustering_config import (
    NASTASClusteringConfig,
    ConfigurationManager
)

# Create configuration with validation
config = NASTASClusteringConfig(
    n_regimes=8,
    algorithm_type='adaptive_clustering',
    regime_search_min=5,
    regime_search_max=15,
    enable_m1_optimization=True,
    memory_limit_mb=2048
)

# Save configuration
config.save_to_file('config/clustering_config.json')

# Load configuration
config = NASTASClusteringConfig.load_from_file('config/clustering_config.json')
```

### Using Memory Manager

```python
from src.training.steps.market_analysis.components.memory_manager import (
    MemoryManager,
    memory_checkpoint,
    monitor_memory_usage
)

# Create memory manager
memory_manager = MemoryManager(
    memory_limit_mb=2048,
    enable_m1_optimization=True
)

# Use with context manager
with memory_checkpoint("clustering_operation", memory_manager):
    # Your clustering code here
    result = perform_clustering(data)

# Get memory report
report = memory_manager.get_memory_report()
print(f"Peak memory: {report['peak_memory_mb']:.2f} MB")

# Decorate functions for automatic monitoring
@monitor_memory_usage
def process_large_dataset(data):
    return data.apply(some_operation)
```

### Using Clustering Algorithms

```python
from src.training.steps.market_analysis.components.clustering_algorithms import (
    ClusteringAlgorithmFactory,
    GaussianMixtureClustering,
    AdaptiveClusteringAlgorithm
)
from src.training.steps.market_analysis.components.clustering_config import ClusteringConfig

# Create configuration
config = ClusteringConfig(
    n_regimes=8,
    algorithm_type='adaptive_clustering'
)

# Create algorithm using factory
algorithm = ClusteringAlgorithmFactory.create_algorithm(
    'adaptive_clustering',
    config,
    memory_manager
)

# Run clustering
result = algorithm.fit_predict(features)

# Access results
print(f"Number of clusters: {result.n_clusters}")
print(f"Silhouette score: {result.metrics['silhouette_score']:.3f}")
print(f"Execution time: {result.execution_time:.2f}s")
```

### Using Refactored Label Fusion

```python
from src.training.steps.market_analysis.regime_analysis.label_fusion import (
    LabelFusionService,
    LabelMappingService,
    DawidSkeneService
)

# Create label fusion service
fusion_service = LabelFusionService()

# Run Dawid-Skene fusion
result = fusion_service.run_dawid_skene(
    tas_assignments=tas_labels,
    nas_assignments=nas_labels,
    target_k=8,
    features=feature_matrix,
    max_iterations=50,
    tolerance=1e-6
)

# Access fused labels
fused_labels = result.assignments
metadata = result.metadata

print(f"Converged: {metadata['converged']}")
print(f"Iterations: {metadata['iterations']}")
```

### Using Import Manager

```python
from src.training.steps.market_analysis.components.imports import (
    get_import_manager,
    check_dependencies,
    log_import_status
)

# Check available dependencies
manager = get_import_manager()

if manager.is_available('sklearn'):
    from sklearn.cluster import KMeans
    # Use sklearn
else:
    # Use fallback implementation
    pass

# Get dependency report
report = check_dependencies()
print(f"Available modules: {report['available_modules']}")
print(f"Missing modules: {report['missing_modules']}")

# Log import status
log_import_status()
```

## Integration with Existing Code

### Example: Updating Existing Clustering Component

```python
# OLD CODE
class MyClusteringComponent:
    def __init__(self, config_dict):
        self.config = config_dict
        # Manual configuration validation
        # No memory management
        # Manual import handling
    
    def run_clustering(self, data):
        # Direct clustering without memory management
        result = self._cluster(data)
        return result

# NEW CODE WITH REFACTORED MODULES
from .clustering_config import NASTASClusteringConfig, ConfigurationManager
from .memory_manager import MemoryManager, memory_checkpoint
from .clustering_algorithms import ClusteringAlgorithmFactory

class MyClusteringComponent:
    def __init__(self, config: NASTASClusteringConfig = None):
        # Use configuration manager
        self.config_manager = ConfigurationManager()
        self.config = config or self.config_manager.create_config('nas_tas')
        
        # Initialize memory manager
        self.memory_manager = MemoryManager(
            memory_limit_mb=getattr(config, 'memory_limit_mb', None),
            enable_m1_optimization=True
        )
        
        # Initialize clustering algorithm
        self.clustering_algorithm = ClusteringAlgorithmFactory.create_algorithm(
            self.config.algorithm_type,
            self.config,
            self.memory_manager
        )
    
    def run_clustering(self, data):
        # Use memory checkpoint
        with memory_checkpoint("clustering_execution", self.memory_manager):
            # Run clustering with algorithm
            result = self.clustering_algorithm.fit_predict(data)
            
            # Get memory report
            memory_report = self.memory_manager.get_memory_report()
            result.metadata['memory_report'] = memory_report
            
            return result
```

## Best Practices

### 1. Configuration Management
- Always use `NASTASClusteringConfig` for type safety
- Validate configuration before use
- Save configurations for reproducibility

### 2. Memory Management
- Use `memory_checkpoint` for all large operations
- Set appropriate `memory_limit_mb` based on available resources
- Monitor peak memory usage for optimization

### 3. Clustering Algorithms
- Use `AdaptiveClusteringAlgorithm` for automatic algorithm selection
- Access metrics through `ClusteringResult.metrics`
- Log execution times for performance analysis

### 4. Label Fusion
- Use the refactored service for cleaner code
- Check convergence status in metadata
- Handle non-convergence cases appropriately

### 5. Import Management
- Check dependencies before use with `ImportManager`
- Use fallback implementations when modules unavailable
- Log import status for debugging

## Performance Tips

### Memory Optimization
```python
# Optimize large arrays
optimized_array = memory_manager.optimize_memory_usage(large_array)

# Add cleanup callbacks
memory_manager.add_cleanup_callback(lambda: del temporary_data)

# Force cleanup when needed
memory_manager.force_cleanup()
```

### Algorithm Selection
```python
# Let adaptive algorithm choose based on data
adaptive_algo = ClusteringAlgorithmFactory.create_algorithm(
    'adaptive_clustering',
    config,
    memory_manager
)

# Or specify algorithm for specific use case
gmm_algo = ClusteringAlgorithmFactory.create_algorithm(
    'gaussian_mixture',
    config,
    memory_manager
)
```

## Troubleshooting

### Memory Issues
```python
# Check memory usage
stats = memory_manager.get_memory_stats()
if stats.memory_percentage > 80:
    memory_manager.force_cleanup()
```

### Import Errors
```python
# Check what's available
manager = get_import_manager()
report = manager.get_availability_report()
for module, available in report.items():
    if not available:
        print(f"Missing: {module}")
```

### Configuration Errors
```python
# Validate configuration
config_manager = ConfigurationManager()
is_valid = config_manager.validate_config(config)
if not is_valid:
    print("Configuration validation failed")
```

## Migration Path

1. **Phase 1: Use New Support Modules**
   - Start using configuration module for new code
   - Integrate memory manager in critical paths
   - Replace manual clustering with algorithm factory

2. **Phase 2: Refactor Existing Code**
   - Update existing components to use new modules
   - Replace old configuration dictionaries
   - Add memory management to existing operations

3. **Phase 3: Complete Integration**
   - Remove redundant code
   - Consolidate all configuration management
   - Optimize memory usage across all components

## Support and Issues

For issues or questions about the refactored modules:
1. Check this integration guide
2. Review the `REFACTORING_SUMMARY.md`
3. Examine the module docstrings
4. Check the example code in this guide