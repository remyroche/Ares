# Hardware Module Migration Guide

This guide helps you migrate from the separate `hardware.py` and `hardware_accelerator.py` modules to the new unified `unified_hardware.py` module.

## Overview

The unified hardware module combines the functionality of both `hardware.py` and `hardware_accelerator.py` into a single, comprehensive hardware management system.

## Key Changes

### 1. Unified Hardware Manager

**Old approach:**
```python
from src.utils.nas_tas.hardware import HardwareOptimizer
from src.utils.nas_tas.hardware_accelerator import NASHardwareAccelerator, TASHardwareAccelerator

# Separate managers for different purposes
hardware_optimizer = HardwareOptimizer()
nas_accelerator = NASHardwareAccelerator()
tas_accelerator = TASHardwareAccelerator()
```

**New approach:**
```python
from src.utils.nas_tas import UnifiedHardwareManager, HardwareAccelerationConfig

# Single unified manager
config = HardwareAccelerationConfig(
    enable_gpu_acceleration=True,
    enable_m1_optimization=True,
    workload_type=WorkloadType.NAS_SEARCH
)
hardware_manager = UnifiedHardwareManager(config)
```

### 2. Workload-Specific Optimization

**Old approach:**
```python
# Manual optimization for different workloads
if workload == "nas":
    nas_accelerator.optimize_model(model)
elif workload == "tas":
    tas_accelerator.optimize_model(model)
else:
    hardware_optimizer.optimize_model(model)
```

**New approach:**
```python
# Automatic workload-specific optimization
optimized_data = hardware_manager.optimize_for_workload(
    workload_type=WorkloadType.NAS_SEARCH,
    data=model_data
)
```

### 3. Configuration Management

**Old approach:**
```python
# Multiple configuration objects
nas_config = NASConfig()
tas_config = TASConfig()
hardware_config = HardwareConfig()
```

**New approach:**
```python
# Single unified configuration
config = HardwareAccelerationConfig(
    enable_gpu_acceleration=True,
    enable_xla_compilation=True,
    enable_memory_optimization=True,
    enable_m1_optimization=True,
    workload_type=WorkloadType.NAS_SEARCH,
    optimization_level=OptimizationLevel.BALANCED
)
```

## Migration Steps

### Step 1: Update Imports

Replace:
```python
from src.utils.nas_tas.hardware import HardwareOptimizer
from src.utils.nas_tas.hardware_accelerator import NASHardwareAccelerator, TASHardwareAccelerator
```

With:
```python
from src.utils.nas_tas import (
    UnifiedHardwareManager, 
    HardwareAccelerationConfig,
    WorkloadType,
    OptimizationLevel
)
```

### Step 2: Update Hardware Manager Creation

Replace:
```python
hardware_optimizer = HardwareOptimizer()
nas_accelerator = NASHardwareAccelerator()
tas_accelerator = TASHardwareAccelerator()
```

With:
```python
config = HardwareAccelerationConfig(
    enable_gpu_acceleration=True,
    enable_m1_optimization=True,
    workload_type=WorkloadType.NAS_SEARCH
)
hardware_manager = UnifiedHardwareManager(config)
```

### Step 3: Update Optimization Calls

Replace:
```python
# Old approach
if workload == "nas":
    optimized_data = nas_accelerator.optimize_model(model)
elif workload == "tas":
    optimized_data = tas_accelerator.optimize_model(model)
else:
    optimized_data = hardware_optimizer.optimize_model(model)
```

With:
```python
# New approach
optimized_data = hardware_manager.optimize_for_workload(
    workload_type=WorkloadType.NAS_SEARCH,  # or TAS_SEARCH, ML_TRAINING, etc.
    data=model_data
)
```

### Step 4: Update Performance Monitoring

Replace:
```python
# Old approach
metrics = hardware_optimizer.get_performance_metrics()
status = nas_accelerator.get_hardware_status()
```

With:
```python
# New approach
metrics = hardware_manager.get_performance_metrics()
status = hardware_manager.get_hardware_status()
```

## Workload Types

The unified hardware manager supports the following workload types:

- `WorkloadType.NAS_SEARCH` - Neural Architecture Search
- `WorkloadType.TAS_SEARCH` - Tree-based Architecture Search
- `WorkloadType.ML_TRAINING` - Machine Learning Training
- `WorkloadType.BACKTESTING` - Backtesting
- `WorkloadType.DATA_PROCESSING` - Data Processing
- `WorkloadType.MONTE_CARLO` - Monte Carlo Simulation
- `WorkloadType.FEATURE_ENGINEERING` - Feature Engineering
- `WorkloadType.GENERAL` - General purpose

## Optimization Levels

Choose the appropriate optimization level:

- `OptimizationLevel.MINIMAL` - Minimal optimization
- `OptimizationLevel.BALANCED` - Balanced optimization (recommended)
- `OptimizationLevel.AGGRESSIVE` - Aggressive optimization
- `OptimizationLevel.MAXIMUM` - Maximum optimization

## Benefits of Migration

1. **Unified Interface**: Single interface for all hardware optimization
2. **Workload-Specific Optimization**: Automatic optimization based on workload type
3. **Better Performance**: Combined optimizations from both modules
4. **Easier Maintenance**: Single module to maintain instead of two
5. **Comprehensive Monitoring**: Unified performance monitoring
6. **Flexible Configuration**: Single configuration object for all settings

## Backward Compatibility

The old modules (`hardware.py` and `hardware_accelerator.py`) are still available for backward compatibility, but they are deprecated. It's recommended to migrate to the unified hardware manager for new code.

## Example Migration

Here's a complete example of migrating from the old approach to the new unified approach:

```python
# Old approach
from src.utils.nas_tas.hardware import HardwareOptimizer
from src.utils.nas_tas.hardware_accelerator import NASHardwareAccelerator

hardware_optimizer = HardwareOptimizer()
nas_accelerator = NASHardwareAccelerator()

# Optimize for NAS
optimized_model = nas_accelerator.optimize_model(model)
metrics = hardware_optimizer.get_performance_metrics()

# New approach
from src.utils.nas_tas import (
    UnifiedHardwareManager, 
    HardwareAccelerationConfig,
    WorkloadType
)

config = HardwareAccelerationConfig(
    enable_gpu_acceleration=True,
    enable_m1_optimization=True,
    workload_type=WorkloadType.NAS_SEARCH
)
hardware_manager = UnifiedHardwareManager(config)

# Optimize for NAS
optimized_data = hardware_manager.optimize_for_workload(
    workload_type=WorkloadType.NAS_SEARCH,
    data=model_data
)
metrics = hardware_manager.get_performance_metrics()
```

## Support

If you encounter any issues during migration, please check the unified hardware module documentation or contact the development team.