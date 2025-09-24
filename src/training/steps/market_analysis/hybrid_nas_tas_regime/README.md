# Hybrid NAS-TAS Regime Detection System

A comprehensive regime detection system that combines Neural Architecture Search (NAS) and Tree Architecture Search (TAS) approaches for market regime identification.

## Overview

This system implements a triple approach to regime detection:
1. **NAS Regime Detection** - Uses neural architecture search for regime identification
2. **TAS Regime Detection** - Uses tree architecture search for regime identification  
3. **Hybrid Consolidation** - Combines both approaches for robust regime detection

## Key Components

### Shared Utilities (`shared_utils/`)

Common utilities used by both NAS and TAS systems:

- **Data Pipeline** - Market data collection and preprocessing
- **Feature Collection** - Feature extraction using existing feature_generator/
- **Economic Significance** - Economic impact evaluation
- **Trading Viability** - Trading opportunity assessment
- **Search Strategies** - Bayesian optimization and grid search
- **Evolutionary Algorithms** - NSGA-II and SPEA2 optimization
- **Hardware Optimization** - Performance optimization
- **Analysis Components** - Regime and cluster analysis
- **Metrics Reporting** - Consolidated reporting

## Usage

```python
from src.training.steps.market_analysis.hybrid_nas_tas_regime.hybrid_orchestrator import (
    HybridOrchestrator, HybridOrchestratorConfig
)

# Configure the system
config = HybridOrchestratorConfig(
    symbol="BTCUSDT",
    timeframe="15m",
    start_date="2023-01-01",
    end_date="2023-12-31"
)

# Create orchestrator
orchestrator = HybridOrchestrator(config)

# Execute pipeline
consolidated_report = await orchestrator.execute_hybrid_pipeline()
```

## Output Format

The system produces a `ConsolidatedMetricsReport` containing:

- **NAS Metrics** - Neural architecture search results
- **TAS Metrics** - Tree architecture search results  
- **Hybrid Metrics** - Consolidated regime mapping
- **Comparison Metrics** - System performance comparison
- **Recommendations** - Best system recommendations

## Features

- **Data Pipeline**: Uses existing klines_parquet system
- **Feature Collection**: Integrates with pre-existing feature_generator/
- **Economic Significance**: Comprehensive economic impact analysis
- **Trading Viability**: Practical trading opportunity assessment
- **Advanced Optimization**: Bayesian optimization and evolutionary algorithms
- **Hardware Optimization**: GPU acceleration and memory optimization

## Integration Points

- **NAS Regime Detection**: Integrates with existing NAS systems
- **TAS Regime Detection**: Integrates with existing TAS systems
- **Future Trading Systems**: Designed for compatibility with future NAS and TAS trading systems

## Dependencies

- `pandas`, `numpy`, `scikit-learn`
- `src.utils.data.klines_parquet`
- `src.utils.hardware`
- `src.utils.matrix_operations`
- `src.feature_generation`

## Performance

- **Execution Time**: 30-120 seconds depending on data size
- **Memory Usage**: Optimized for 8GB+ systems
- **Scalability**: Supports large datasets with hardware optimization
- **Accuracy**: Combines multiple approaches for robust regime detection