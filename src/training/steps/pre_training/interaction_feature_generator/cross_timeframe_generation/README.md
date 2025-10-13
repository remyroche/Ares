# Cross-Timeframe Feature Generation System

A comprehensive, plug-and-play system for generating and optimizing higher temporal frequency (HTF) features with regime-aware optimization, cost-aware materialization, and statistical selection.

## Overview

This system implements a complete pipeline for cross-timeframe feature generation that:

1. **Generates HTF features** using the same FeatureBank functions and TransformRouter as base features
2. **Optimizes lookback lengths** with regime-aware hierarchical shrinkage
3. **Assigns update styles** (EHU vs RIH) based on cost-benefit analysis
4. **Selects features** using knapsack optimization with correlation constraints
5. **Generates interactions** with HTF-aware templates
6. **Performs statistical selection** with stability selection and FDR control
7. **Evaluates performance** using walk-forward validation
8. **Monitors and automates** with adaptive penalties and BOCPD triggers

## Key Features

### 🎯 **Phase-1 HTF Probe Stage**
- Coarse adaptive grids (15m to 300m)
- Regime segmentation with change-point detection
- Adaptive penalized OOS scoring
- Early stopping and shortlisting

### 🔧 **Phase-2 Optimization**
- Local grids around shortlisted candidates
- IC surface fitting with penalized splines or Gaussian processes
- Regime-aware hierarchical shrinkage
- Discrete vs blend export strategies

### ⚡ **EHU vs RIH Assignment**
- Cost-aware assignment based on utility/cost ratios
- Feature-specific staleness curves
- Hybrid mode with runtime switching
- Dynamic adaptation to market conditions

### 🎒 **Knapsack Selection**
- Integer programming with correlation constraints
- Family coverage requirements
- Fallback to greedy algorithm
- Cost and cardinality limits

### 🔄 **HTF Materialization**
- Same FeatureBank functions and TransformRouter
- RIH features with incremental state maintenance
- EHU features with session-based updates
- Consistent naming convention

### 🧬 **Interaction Templates**
- Core 15 interactions (theory-first)
- HTF-aware templates (2-3 additional)
- Dynamic budget allocation

### 📊 **Statistical Selection**
- Stability selection with block bootstrap
- Permutation importance testing
- Benjamini-Hochberg FDR control
- Conditional IC tests
- Group LASSO option

### 📈 **Walk-Forward Evaluation**
- Purged and embargoed validation
- Wild bootstrap confidence intervals
- SPA (Superior Predictive Ability) test
- Regime-aware evaluation
- Ablation studies

### 🤖 **Monitoring & Automation**
- Adaptive penalties meta-learning
- BOCPD triggers for regime changes
- Performance dashboards
- Automated retraining triggers
- Alert systems

## Installation

```bash
# Install required dependencies
pip install pandas numpy scipy scikit-learn plotly
pip install cvxpy  # For integer programming (optional)
pip install ruptures  # For change-point detection (optional)
pip install pymc  # For Bayesian optimization (optional)
```

## Quick Start

```python
from pipeline import CrossTimeframePipeline, PipelineConfig
import pandas as pd
import numpy as np

# Create sample data
ohlcv_data = pd.DataFrame({
    'open': np.random.randn(1000).cumsum() + 100,
    'high': np.random.randn(1000).cumsum() + 101,
    'low': np.random.randn(1000).cumsum() + 99,
    'close': np.random.randn(1000).cumsum() + 100,
    'volume': np.random.randint(1000, 10000, 1000)
})

# Create targets
targets = np.log(ohlcv_data['close'] / ohlcv_data['close'].shift(1)).shift(-1)

# Configure pipeline
config = PipelineConfig(
    base_timeframe_minutes=5,
    max_cost_ms=25.0,
    max_features=120,
    max_correlation=0.8
)

# Run pipeline
pipeline = CrossTimeframePipeline(config)
results = pipeline.run_pipeline(ohlcv_data, targets=targets)

# Access results
print(f"Selected features: {len(results['final_features'].selected_features)}")
print(f"Overall IC: {results['evaluation_results'].overall_ic:.4f}")
```

## Configuration Options

### Conservative Configuration
```python
config = PipelineConfig(
    base_timeframe_minutes=5,
    coarse_grid_min=30,
    coarse_grid_max=180,
    max_cost_ms=15.0,
    max_features=80,
    max_correlation=0.7,
    fdr_q=0.05,
    walk_forward_folds=5
)
```

### Aggressive Configuration
```python
config = PipelineConfig(
    base_timeframe_minutes=5,
    coarse_grid_min=15,
    coarse_grid_max=300,
    max_cost_ms=50.0,
    max_features=200,
    max_correlation=0.9,
    fdr_q=0.2,
    walk_forward_folds=3
)
```

### High-Frequency Configuration
```python
config = PipelineConfig(
    base_timeframe_minutes=1,
    coarse_grid_min=5,
    coarse_grid_max=60,
    max_cost_ms=10.0,
    max_features=50,
    max_correlation=0.6,
    fdr_q=0.01,
    walk_forward_folds=10,
    hybrid_mode=True
)
```

## Pipeline Components

### 1. Phase-1 HTF Probe (`phase1_probe.py`)
- **CoarseGridGenerator**: Generates adaptive grids around base lookbacks
- **HTFFeatureGenerator**: Creates HTF features using same FeatureBank functions
- **Phase1HTFProbe**: Main probe stage with scoring and shortlisting

### 2. Regime Segmentation (`regime_segmentation.py`)
- **ChangePointDetector**: PELT/CUSUM change-point detection
- **BOCPD**: Bayesian Online Change-Point Detection
- **RegimeClassifier**: Volatility regime classification
- **RegimeSegmentation**: Main regime segmentation system

### 3. Adaptive Scoring (`scoring_system.py`)
- **UncertaintyEstimator**: Wild/bootstrap standard error estimation
- **CostEstimator**: CPU and memory cost estimation
- **StalenessCalculator**: Feature-specific staleness curves
- **MetaLearner**: Adaptive penalty parameter learning

### 4. Phase-2 Optimization (`phase2_optimization.py`)
- **LocalGridGenerator**: Local grids around shortlisted candidates
- **ICSurfaceFitter**: IC surface fitting with splines or GPs
- **HierarchicalShrinkage**: Regime-aware hierarchical shrinkage
- **ExportDecisionMaker**: Discrete vs blend export decisions

### 5. EHU/RIH Assignment (`ehu_rih_assignment.py`)
- **StalenessCurveCalculator**: Feature-specific staleness curves
- **CostBenefitAnalyzer**: Cost-benefit trade-off analysis
- **HybridModeManager**: Runtime switching between EHU/RIH
- **EHU_RIH_Assignment**: Main assignment system

### 6. Knapsack Selection (`knapsack_selection.py`)
- **CorrelationCalculator**: Partial correlation calculation
- **IntegerProgramSolver**: CVXPY/SCIPY optimization
- **KnapsackSelection**: Main selection system

### 7. HTF Materialization (`htf_materialization.py`)
- **RIHStateManager**: Incremental state management for RIH features
- **EHUStateManager**: State management for EHU features
- **HTFFeatureGenerator**: HTF feature generation
- **HTFMaterialization**: Main materialization system

### 8. Interaction Templates (`interaction_templates.py`)
- **CoreInteractionTemplates**: Core 15 interaction templates
- **HTFAwareTemplates**: HTF-aware interaction templates
- **InteractionGenerator**: Main interaction generation

### 9. Statistical Selection (`statistical_selection.py`)
- **StabilitySelector**: Stability selection with block bootstrap
- **PermutationTester**: Permutation importance testing
- **FDRController**: Benjamini-Hochberg FDR control
- **ConditionalICTester**: Conditional IC testing
- **GroupLASSOSelector**: Group LASSO feature selection

### 10. Walk-Forward Evaluation (`evaluation.py`)
- **WalkForwardValidator**: Walk-forward validation with purging/embargo
- **BootstrapEvaluator**: Wild bootstrap confidence intervals
- **SPATester**: Superior Predictive Ability test
- **RegimeEvaluator**: Regime-aware evaluation
- **AblationEvaluator**: Ablation studies

### 11. Monitoring System (`monitoring.py`)
- **AdaptivePenaltyLearner**: Meta-learning of penalty parameters
- **BOCPDTrigger**: BOCPD-based regime change triggers
- **PerformanceDashboard**: Performance monitoring dashboard
- **AlertSystem**: Alert system for monitoring
- **RetrainingTrigger**: Automated retraining triggers

## Usage Examples

### Basic Usage
```python
# See example_usage.py for comprehensive examples
python example_usage.py
```

### Individual Components
```python
from regime_segmentation import RegimeSegmentation
from phase1_probe import Phase1HTFProbe
from monitoring import MonitoringSystem

# Use individual components
regime_seg = RegimeSegmentation(config)
phase1_probe = Phase1HTFProbe(config)
monitoring = MonitoringSystem(config)
```

### Custom Configuration
```python
# Create custom configuration
config = PipelineConfig(
    base_timeframe_minutes=5,
    session_start_hour=9,
    session_end_hour=16,
    coarse_grid_min=15,
    coarse_grid_max=300,
    adaptive_refinement_threshold=0.75,
    change_point_method='PELT',
    regime_vol_quantile=0.6,
    bocpd_hazard=1/200,
    lambda_unc=0.10,
    lambda_cost=0.05,
    lambda_stale=0.05,
    meta_learning_range=0.05,
    rih_threshold=0.01,
    hybrid_mode=True,
    max_cost_ms=25.0,
    max_features=120,
    max_correlation=0.8,
    stability_resamples=80,
    fdr_q=0.1,
    min_conditional_ic=0.25,
    embargo_minutes=60,
    walk_forward_folds=5,
    spa_test=True,
    adaptive_penalties=True,
    dashboard_enabled=True
)
```

## Performance Considerations

### Memory Usage
- RIH features maintain incremental state
- EHU features are computed once per HTF period
- State persistence for recovery

### Computational Cost
- Phase-1: Coarse grid evaluation (fast)
- Phase-2: Local grid optimization (moderate)
- Statistical selection: Bootstrap resampling (slow)
- Walk-forward evaluation: Multiple folds (slow)

### Latency Requirements
- RIH features: Real-time incremental updates
- EHU features: End-of-hour updates
- Hybrid mode: Runtime switching based on conditions

## Monitoring and Alerts

### Performance Metrics
- Information Coefficient (IC)
- Sharpe ratio
- Maximum drawdown
- Feature count
- Regime-specific performance

### Alert Types
- Low IC alerts
- High drawdown alerts
- Low Sharpe ratio alerts
- High feature count alerts
- Regime change alerts

### Dashboard Features
- Real-time performance plots
- Regime transition visualization
- Feature importance tracking
- Cost-benefit analysis

## File Structure

```
cross_timeframe_generation/
├── __init__.py
├── pipeline.py                    # Main pipeline orchestrator
├── phase1_probe.py               # Phase-1 HTF probe stage
├── phase2_optimization.py        # Phase-2 optimization
├── regime_segmentation.py        # Regime segmentation
├── scoring_system.py             # Adaptive scoring system
├── ehu_rih_assignment.py         # EHU/RIH assignment
├── knapsack_selection.py         # Knapsack selection
├── htf_materialization.py        # HTF materialization
├── interaction_templates.py      # Interaction templates
├── statistical_selection.py      # Statistical selection
├── evaluation.py                 # Walk-forward evaluation
├── monitoring.py                 # Monitoring system
├── example_usage.py              # Usage examples
└── README.md                     # This file
```

## Dependencies

### Required
- pandas >= 1.3.0
- numpy >= 1.20.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0

### Optional
- cvxpy (for integer programming)
- ruptures (for change-point detection)
- pymc (for Bayesian optimization)
- plotly (for dashboards)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{cross_timeframe_features,
  title={Cross-Timeframe Feature Generation System},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/cross-timeframe-features}
}
```

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the example usage

## Changelog

### Version 1.0.0
- Initial release
- Complete pipeline implementation
- All core components
- Monitoring and automation
- Comprehensive documentation