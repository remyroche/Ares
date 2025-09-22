# HMM Clustering Module

This module provides consolidated HMM-based regime clustering functionality with optimized performance and comprehensive metrics.

## Structure

- **`core/`** - Core clustering algorithms
- **`metrics/`** - Metrics and reporting
- **`integration/`** - Integration and orchestration
- **`utils/`** - Utilities and shared code
- **`components/`** - Component wrappers
- **`discovery/`** - HMM regime discovery

## Key Features

- Matrix-optimized clustering with GPU acceleration
- Enhanced clustering with 4D frontier optimization
- Comprehensive metrics evolution tracking
- Fast fail mechanisms
- Hardware optimization integration
- Memory management optimization

## Usage

```python
from src.training.steps.market_analysis.hmm_clustering import (
    MatrixOptimizedClusterer,
    EnhancedMatrixOptimizedClusterer,
    OptimalRegimeClusteringOrchestrator
)
```

## Configuration

See `config.py` for configuration options.