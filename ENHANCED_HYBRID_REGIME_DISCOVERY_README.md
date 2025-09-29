# Enhanced Hybrid NAS-TAS Regime Discovery System

## Overview

This enhanced hybrid regime discovery system combines Neural Architecture Search (NAS) and Tree-driven Advanced Statistics (TAS) methods with advanced multi-objective optimization to achieve the most statistically and economically relevant regime clustering.

## Key Features

### 🎯 Multi-Objective Optimization
- **CV Optimization**: Minimizes coefficient of variation for volatility, returns, and volume
- **Accessory CV**: Includes momentum and entropy CV with lower weights for uncertain regimes
- **Cluster Distribution Targets**: Ensures 6-15 clusters with no cluster above 25% or below 3% distribution
- **Pareto Frontier Analysis**: Finds optimal solutions balancing multiple objectives

### 🔄 Advanced Regime Alignment
- **Optimal Transport**: Uses Hungarian algorithm for optimal regime alignment
- **Feature-Based Alignment**: Aligns regimes based on market characteristics
- **Soft Alignment**: Handles cases where methods disagree on regime boundaries

### 💰 Enhanced Economic Evaluation
- **Economic Significance Scoring**: Prioritizes regimes with higher economic impact
- **Trading Viability Assessment**: Ensures regimes are actually tradeable
- **CV-Based Optimization**: Optimizes coefficient of variation across multiple metrics

### 🔍 Consensus Validation
- **Multi-Criteria Validation**: Combines statistical, economic, and temporal criteria
- **Quality Assurance**: Ensures consensus results are better than individual methods
- **Temporal Consistency**: Validates regime transition smoothness

## Architecture

```
Enhanced Hybrid System
├── RegimeAlignmentManager
│   ├── Hungarian Algorithm
│   ├── Optimal Transport
│   └── Feature-Based Alignment
├── EnhancedEconomicEvaluator
│   ├── CV Optimization (Volatility, Returns, Volume)
│   ├── Accessory CV (Momentum, Entropy)
│   └── Distribution Target Validation
├── ConsensusValidator
│   ├── Statistical Validation
│   ├── Economic Validation
│   └── Temporal Validation
├── MultiObjectiveOptimizer
│   ├── Pareto Frontier Analysis
│   ├── Genetic Algorithm Evolution
│   └── Solution Selection
└── HybridOrchestrator
    ├── Component Coordination
    ├── Result Integration
    └── Quality Assurance
```

## Implementation Details

### 1. Regime Alignment Manager

**Purpose**: Aligns regimes between NAS and TAS methods before combination.

**Key Methods**:
- `align_regimes()`: Main alignment function using Hungarian algorithm
- `_hungarian_alignment()`: Optimal assignment using Hungarian algorithm
- `_optimal_transport_alignment()`: Alternative alignment using optimal transport
- `_feature_based_alignment()`: Alignment based on market characteristics

**Configuration**:
```python
alignment_config = AlignmentConfig(
    method='hungarian',  # 'hungarian', 'optimal_transport', 'greedy'
    min_overlap_threshold=0.1,
    max_regime_distance=0.5,
    enable_soft_alignment=True,
    alignment_confidence_threshold=0.3
)
```

### 2. Enhanced Economic Evaluator

**Purpose**: Evaluates regime clustering with CV optimization and distribution targets.

**Key Features**:
- **CV Optimization**: Minimizes coefficient of variation for volatility, returns, volume
- **Accessory CV**: Includes momentum and entropy CV with lower weights
- **Distribution Validation**: Ensures cluster count and distribution targets are met

**Configuration**:
```python
economic_config = EconomicEvaluationConfig(
    target_cluster_count_min=6,
    target_cluster_count_max=15,
    max_cluster_distribution=0.25,  # 25% max
    min_cluster_distribution=0.03,  # 3% min
    volatility_cv_weight=0.4,
    returns_cv_weight=0.3,
    volume_cv_weight=0.3,
    momentum_cv_weight=0.1,  # Lower weight
    entropy_cv_weight=0.1    # Lower weight
)
```

### 3. Consensus Validator

**Purpose**: Validates consensus predictions using multi-objective criteria.

**Validation Criteria**:
- **Statistical**: Silhouette score, Calinski-Harabasz score, Davies-Bouldin score
- **Economic**: Economic significance, trading viability, regime stability
- **Temporal**: Temporal smoothness, regime duration, transition consistency

**Configuration**:
```python
consensus_config = ConsensusValidationConfig(
    silhouette_weight=0.25,
    calinski_harabasz_weight=0.20,
    davies_bouldin_weight=0.20,
    economic_significance_weight=0.30,
    trading_viability_weight=0.25,
    temporal_smoothness_weight=0.30,
    min_consensus_quality=0.6,
    enable_multi_objective=True
)
```

### 4. Multi-Objective Optimizer

**Purpose**: Optimizes regime clustering using multiple objectives simultaneously.

**Optimization Objectives**:
1. **Statistical Quality**: Silhouette score, Calinski-Harabasz score, Davies-Bouldin score
2. **Economic Significance**: Economic impact, trading viability, regime stability
3. **CV Optimization**: Coefficient of variation minimization
4. **Distribution Quality**: Cluster count and distribution targets

**Configuration**:
```python
multi_objective_config = MultiObjectiveConfig(
    target_cluster_count_min=6,
    target_cluster_count_max=15,
    max_cluster_distribution=0.25,
    min_cluster_distribution=0.03,
    volatility_cv_weight=0.4,
    returns_cv_weight=0.3,
    volume_cv_weight=0.3,
    momentum_cv_weight=0.1,
    entropy_cv_weight=0.1,
    statistical_weight=0.25,
    economic_weight=0.30,
    temporal_weight=0.20,
    cv_optimization_weight=0.25,
    max_iterations=100,
    population_size=50,
    enable_pareto_frontier=True
)
```

## Usage Example

```python
import asyncio
from src.training.steps.market_analysis.hybrid_nas_tas_regime.hybrid_orchestrator import (
    HybridOrchestrator, HybridOrchestratorConfig
)

async def run_enhanced_hybrid_discovery():
    # Configure enhanced hybrid orchestrator
    config = HybridOrchestratorConfig(
        symbol="BTCUSDT",
        timeframe="15m",
        population_size=50,
        max_generations=100,
        use_gpu_acceleration=True,
        include_detailed_metrics=True
    )
    
    # Initialize orchestrator
    orchestrator = HybridOrchestrator(config)
    
    # Run enhanced hybrid analysis
    results = orchestrator.orchestrate_tas_nas_detection(
        market_data, 
        timeframes=['15m']
    )
    
    # Analyze results
    hybrid_analysis = results.get('hybrid_analysis', {})
    if hybrid_analysis.get('success', False):
        hybrid_labels = hybrid_analysis.get('hybrid_labels', [])
        print(f"Found {len(np.unique(hybrid_labels))} regimes")
        
        # Display optimization results
        optimization_result = hybrid_analysis.get('optimization_result', {})
        best_solution = optimization_result.get('best_solution', {})
        objectives = best_solution.get('objectives', {})
        
        print(f"Statistical Score: {objectives.get('silhouette_score', 0.0):.3f}")
        print(f"Economic Score: {objectives.get('economic_significance', 0.0):.3f}")
        print(f"CV Optimization: {objectives.get('cv_optimization', 0.0):.3f}")

# Run the example
asyncio.run(run_enhanced_hybrid_discovery())
```

## Key Benefits

### 1. **Statistically Robust**
- Multi-objective optimization ensures high clustering quality
- CV optimization minimizes within-cluster variation
- Pareto frontier analysis finds optimal trade-offs

### 2. **Economically Relevant**
- Economic significance scoring prioritizes meaningful regimes
- Trading viability assessment ensures practical applicability
- Regime stability validation prevents excessive switching

### 3. **Temporally Consistent**
- Temporal smoothness validation ensures stable regimes
- Regime duration optimization prevents over-fragmentation
- Transition consistency validation maintains logical regime changes

### 4. **Distribution Optimized**
- Cluster count targets (6-15 clusters) ensure appropriate granularity
- Distribution limits prevent dominance by single regimes
- Minimum cluster size ensures statistical significance

## Performance Metrics

The enhanced system provides comprehensive metrics:

### Statistical Metrics
- **Silhouette Score**: Measures cluster separation quality
- **Calinski-Harabasz Score**: Measures cluster compactness
- **Davies-Bouldin Score**: Measures cluster separation (lower is better)

### Economic Metrics
- **Economic Significance Score**: Measures economic impact of regimes
- **Trading Viability Score**: Measures practical tradeability
- **CV Optimization Score**: Measures coefficient of variation optimization

### Temporal Metrics
- **Temporal Smoothness**: Measures regime transition stability
- **Regime Duration**: Measures average regime persistence
- **Transition Consistency**: Measures logical regime changes

### Distribution Metrics
- **Cluster Count**: Number of regimes (target: 6-15)
- **Distribution Balance**: Evenness of regime distribution
- **Target Compliance**: Adherence to distribution limits

## Configuration Options

### Regime Alignment
- **Method**: Hungarian, Optimal Transport, or Greedy
- **Overlap Threshold**: Minimum overlap for alignment
- **Confidence Threshold**: Minimum confidence for alignment

### Economic Evaluation
- **CV Weights**: Weights for volatility, returns, volume CV
- **Accessory Weights**: Lower weights for momentum and entropy
- **Distribution Targets**: Cluster count and distribution limits

### Multi-Objective Optimization
- **Objective Weights**: Weights for statistical, economic, temporal objectives
- **Population Size**: Number of solutions in genetic algorithm
- **Max Iterations**: Maximum optimization iterations
- **Pareto Frontier**: Enable/disable Pareto frontier analysis

## Error Handling

The system includes comprehensive error handling:

1. **Graceful Degradation**: Falls back to simpler methods if enhanced components fail
2. **Validation Checks**: Ensures all components are properly initialized
3. **Quality Assurance**: Validates results before returning
4. **Detailed Logging**: Provides comprehensive logging for debugging

## Future Enhancements

1. **Dynamic Weight Adjustment**: Automatically adjust weights based on market conditions
2. **Online Learning**: Update models with new data in real-time
3. **Ensemble Methods**: Combine multiple optimization strategies
4. **Advanced Metrics**: Include additional economic and statistical metrics

## Conclusion

The enhanced hybrid NAS-TAS regime discovery system provides a comprehensive solution for regime detection that balances statistical rigor with economic relevance. By combining multiple optimization objectives and advanced validation techniques, it ensures the discovery of regimes that are both statistically sound and economically meaningful.

The system's modular architecture allows for easy customization and extension, while its comprehensive validation framework ensures high-quality results. The multi-objective optimization approach ensures that all important criteria are considered simultaneously, leading to superior regime detection performance.