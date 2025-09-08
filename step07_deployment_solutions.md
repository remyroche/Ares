# Step07 Dependency and Import Fix - Deployment Solutions

## Problem Summary

The Step07 audit identified two critical issues:
1. **❌ Missing Dependencies** - System is non-functional without numpy, pandas, sklearn, torch, numba, psutil
2. **❌ Import Failures** - Complex import chains causing runtime instability

## Environment Constraints

The current environment has the following limitations:
- **Externally Managed Environment** - Cannot install packages system-wide
- **No Virtual Environment Support** - python3-venv package not available
- **No Conda** - Conda package manager not installed
- **Restricted Permissions** - Cannot modify system Python packages

## Solution Options

### Option 1: Docker Deployment (Recommended)

**Advantages:**
- Complete isolation from host environment
- All dependencies included
- Reproducible builds
- No system modifications required

**Implementation:**
```bash
# Build the Docker image
docker build -f Dockerfile.step07 -t step07-matrix-ops .

# Run Step07 in container
docker run -v $(pwd):/workspace step07-matrix-ops

# Interactive mode for development
docker run -it -v $(pwd):/workspace step07-matrix-ops bash
```

### Option 2: Simplified Step07 (Immediate Fix)

**Advantages:**
- Works with current environment
- No external dependencies required
- Maintains core functionality
- Can be deployed immediately

**Implementation:**
```python
# Use the simplified version
from src.training.steps.model_training.step07_simplified_fixed import create_step07_step

config = {'matrix_operations_config': {'use_gpu': False, 'use_numba': False}}
step = create_step07_step(config)
```

### Option 3: System Package Installation

**Advantages:**
- Uses system package manager
- No virtual environment needed
- Stable and tested packages

**Implementation:**
```bash
# Install system packages (requires sudo)
sudo apt update
sudo apt install python3-numpy python3-pandas python3-sklearn python3-scipy python3-psutil

# For optional packages
sudo apt install python3-torch python3-numba python3-lightgbm
```

### Option 4: User Installation with --user flag

**Advantages:**
- No system modifications
- Installs to user directory
- Works with externally managed environments

**Implementation:**
```bash
# Install to user directory
pip3 install --user numpy pandas scikit-learn scipy psutil

# Optional packages
pip3 install --user torch numba lightgbm
```

## Import Chain Fixes

### Problem Analysis

The original Step07 has complex import chains:
```python
# Complex import chain causing issues
from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from src.training.base_step import BaseStep
from src.training.steps.model_training.matrix_components import MatrixProcessor
```

### Solution: Safe Import Module

Created `src/utils/step07_import_fix.py` with:
- **Safe Import Utility** - Handles missing modules gracefully
- **Fallback Implementations** - Provides basic functionality when modules missing
- **Import Caching** - Prevents repeated import attempts
- **Status Monitoring** - Tracks which modules are available

### Usage Example

```python
from src.utils.step07_import_fix import (
    numpy as np, pandas as pd, torch, numba, psutil,
    system_logger, handles_errors, BaseStep, check_dependencies
)

# Check if all dependencies are available
if check_dependencies():
    print("✅ All dependencies available")
else:
    print("⚠️ Some dependencies missing, using fallbacks")
```

## Immediate Action Plan

### Step 1: Use Simplified Version (Immediate)

```python
# Replace the complex Step07 with simplified version
from src.training.steps.model_training.step07_simplified_fixed import SimplifiedMatrixOperationsStep

# This version:
# - Works without external dependencies
# - Has fixed import chains
# - Maintains core matrix operations
# - Provides fallback implementations
```

### Step 2: Test Current Functionality

```bash
# Test the simplified version
python3 -c "
from src.training.steps.model_training.step07_simplified_fixed import create_step07_step
config = {'matrix_operations_config': {'use_gpu': False}}
step = create_step07_step(config)
print('✅ Simplified Step07 created successfully')
"
```

### Step 3: Deploy with Docker (Recommended)

```bash
# Build and run with Docker
docker build -f Dockerfile.step07 -t step07 .
docker run -v $(pwd):/workspace step07 python3 step07_import_verification.py
```

## Configuration Updates

### Update Step07 Configuration

```yaml
# config/training_config.json
{
  "step07_enhanced_matrix_operations": {
    "output_dir": "data/matrix_operations",
    "use_simplified_version": true,
    "fallback_mode": true,
    "disable_gpu": true,
    "disable_numba": true,
    "target_features": 200,
    "removal_fraction": 0.33
  }
}
```

### Update Pipeline Integration

```python
# In your pipeline code
try:
    from src.training.steps.model_training.step07_enhanced_matrix_operations import EnhancedMatrixOperationsStep
    step_class = EnhancedMatrixOperationsStep
except ImportError:
    from src.training.steps.model_training.step07_simplified_fixed import SimplifiedMatrixOperationsStep
    step_class = SimplifiedMatrixOperationsStep

# Create step instance
step = step_class(config)
```

## Testing and Validation

### Test Script

```python
#!/usr/bin/env python3
"""Test Step07 functionality"""

def test_step07():
    try:
        # Test simplified version
        from src.training.steps.model_training.step07_simplified_fixed import create_step07_step
        
        config = {
            'matrix_operations_config': {
                'use_gpu': False,
                'use_numba': False,
                'batch_size': 1000
            }
        }
        
        step = create_step07_step(config)
        print("✅ Step07 simplified version works")
        
        # Test import fix module
        from src.utils.step07_import_fix import check_dependencies, get_import_summary
        check_dependencies()
        get_import_summary()
        
        return True
        
    except Exception as e:
        print(f"❌ Step07 test failed: {e}")
        return False

if __name__ == "__main__":
    test_step07()
```

## Monitoring and Maintenance

### Dependency Status Monitoring

```python
# Add to your monitoring system
from src.utils.step07_import_fix import get_import_summary

def check_step07_health():
    status = get_import_summary()
    missing = [module for module, available in status.items() if not available]
    
    if missing:
        logger.warning(f"Step07 missing dependencies: {missing}")
        return False
    else:
        logger.info("Step07 all dependencies available")
        return True
```

### Performance Monitoring

```python
# Monitor Step07 performance
def monitor_step07_performance():
    # Check if simplified version is being used
    if hasattr(step, 'simplified_mode'):
        logger.info("Step07 running in simplified mode")
    
    # Monitor matrix computation time
    start_time = time.time()
    # ... matrix operations ...
    duration = time.time() - start_time
    
    if duration > 300:  # 5 minutes
        logger.warning(f"Step07 taking too long: {duration:.2f}s")
```

## Conclusion

The Step07 dependency and import issues can be resolved using multiple approaches:

1. **Immediate Fix**: Use the simplified version that works without external dependencies
2. **Production Fix**: Deploy with Docker for complete dependency isolation
3. **Development Fix**: Use system packages or user installation

The simplified version maintains core functionality while providing a stable foundation for further development. The Docker approach provides the most robust solution for production deployment.

## Files Created

- `requirements_step07.txt` - Python dependencies
- `environment_step07.yml` - Conda environment
- `Dockerfile.step07` - Docker container
- `install_step07_dependencies.sh` - Installation script
- `src/utils/step07_import_fix.py` - Safe import module
- `src/training/steps/model_training/step07_simplified_fixed.py` - Simplified Step07
- `step07_deployment_solutions.md` - This documentation

All files are ready for immediate use and provide multiple deployment options based on your environment constraints.