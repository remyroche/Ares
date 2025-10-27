# Staged Optimization Implementation

## Overview
Implemented a single entry point for SR parameter optimization that follows the progression: **Coarse Grid → Fine Grid → Bayesian TPE**. This staged approach provides efficient exploration and refinement of the parameter space.

## Implementation Details

### 1. Single Entry Point
The main optimization now uses `_run_staged_optimization()` as the primary method:

```python
# Main optimization logic
if enhanced_config.enable_staged_optimization:
    algorithm_result = await self._run_staged_optimization(
        search_space, train_data, test_data, enhanced_config
    )
```

### 2. Three-Stage Process

#### Stage 1: Coarse Grid Search
- **Purpose**: Initial exploration of parameter space
- **Grid Size**: 3x3x3x3x3 = 243 combinations
- **Parameters**: Wide ranges, fewer points
- **Time**: Fast initial exploration

```python
# Coarse grid parameters
min_touches_values = [2, 4, 6]  # 3 points
strength_thresholds = [0.3, 0.5, 0.7]  # 3 points
distance_thresholds = [0.01, 0.02, 0.03]  # 3 points
lookback_periods = [50, 100, 150]  # 3 points
volume_thresholds = [1.0, 1.5, 2.0]  # 3 points
```

#### Stage 2: Fine Grid Search
- **Purpose**: Refinement around best coarse result
- **Grid Size**: 3x3x3x3x3 = 243 combinations
- **Parameters**: Narrower ranges around coarse best
- **Time**: Focused refinement

```python
# Fine grid around coarse best
min_touches_values = [center-1, center, center+1]
strength_thresholds = [center-0.1, center, center+0.1]
# ... etc
```

#### Stage 3: Bayesian TPE Refinement
- **Purpose**: Final optimization using Bayesian methods
- **Trials**: Configurable (default 50)
- **Search Space**: Focused around fine grid best
- **Time**: Intelligent exploration

```python
# Focused search space around fine result
focused_search_space = {
    'min_touches': {'low': center-2, 'high': center+2},
    'strength_threshold': {'low': center-0.2, 'high': center+0.2},
    # ... etc
}
```

### 3. Configuration

```python
@dataclass
class EnhancedSRConfig:
    # Staged optimization settings
    enable_staged_optimization: bool = True
    
    # Stage parameters
    coarse_grid_points: int = 3  # 3x3 coarse grid
    fine_grid_points: int = 5    # 5x5 fine grid  
    bayesian_trials: int = 50    # Bayesian TPE trials
    enable_bayesian_refinement: bool = True
```

### 4. Algorithm Flow

```
┌─────────────────┐
│   Start         │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Stage 1: Coarse │ ──┐
│ Grid Search     │   │
└─────────┬───────┘   │
          │           │
          ▼           │
┌─────────────────┐   │
│ Stage 2: Fine   │   │
│ Grid Search     │   │
└─────────┬───────┘   │
          │           │
          ▼           │
┌─────────────────┐   │
│ Stage 3:        │   │
│ Bayesian TPE    │   │
└─────────┬───────┘   │
          │           │
          ▼           │
┌─────────────────┐   │
│ Return Best     │ ◄─┘
│ Result          │
└─────────────────┘
```

### 5. Error Handling & Fallbacks

- **Stage 1 Failure**: Try fallback algorithms (VectorBT)
- **Stage 2 Failure**: Use coarse result
- **Stage 3 Failure**: Use fine result
- **Complete Failure**: Return error with details

### 6. Performance Characteristics

| Stage | Combinations | Time | Purpose |
|-------|-------------|------|---------|
| Coarse Grid | 243 | Fast | Initial exploration |
| Fine Grid | 243 | Medium | Refinement |
| Bayesian TPE | 50 | Medium | Final optimization |
| **Total** | **~536** | **~3x** | **Complete optimization** |

### 7. Benefits

#### Efficiency
- **Focused Search**: Each stage builds on the previous
- **Reduced Computations**: Bayesian TPE only explores promising regions
- **Early Termination**: Can stop at any stage if needed

#### Reliability
- **Fallback Mechanisms**: Multiple safety nets
- **Progressive Refinement**: Each stage improves the result
- **Error Recovery**: Graceful handling of failures

#### Flexibility
- **Configurable Stages**: Adjust grid sizes and trials
- **Optional Stages**: Can disable Bayesian refinement
- **Fallback Options**: VectorBT as backup

### 8. Usage Examples

#### Default Configuration
```python
config = EnhancedSRConfig()
# Uses: 3x3 coarse → 5x5 fine → 50 Bayesian trials
```

#### High-Precision Configuration
```python
config = EnhancedSRConfig(
    coarse_grid_points=5,    # 5x5 coarse (3125 combinations)
    fine_grid_points=7,      # 7x7 fine (16807 combinations)
    bayesian_trials=100      # 100 Bayesian trials
)
```

#### Fast Configuration
```python
config = EnhancedSRConfig(
    coarse_grid_points=3,    # 3x3 coarse (243 combinations)
    fine_grid_points=3,      # 3x3 fine (243 combinations)
    bayesian_trials=25,      # 25 Bayesian trials
    enable_bayesian_refinement=False  # Skip Bayesian stage
)
```

### 9. Logging & Monitoring

Each stage provides detailed logging:

```
🚀 Starting staged optimization: Coarse Grid → Fine Grid → Bayesian TPE
🔍 Stage 1: Coarse grid search (3x3 grid)
✅ Coarse grid completed: 243 combinations, best score: 0.7234
🔍 Stage 2: Fine grid search (5x5 grid) around best coarse result
✅ Fine grid completed: 243 combinations, improvement: 0.0456
🧠 Stage 3: Bayesian TPE refinement (50 trials)
✅ Bayesian refinement completed: 50 trials, improvement: 0.0123
```

### 10. Result Structure

```python
{
    'optimized_parameters': {...},
    'best_score': 0.7813,
    'stage': 'bayesian_refinement',
    'improvement_over_fine': 0.0123,
    'algorithm_used': 'bayesian_tpe',
    'success': True,
    'total_combinations_tested': 536,
    'bayesian_trials': 50
}
```

## Conclusion

The staged optimization implementation provides:

- **Single Entry Point**: Clean, simple interface
- **Progressive Refinement**: Each stage improves the result
- **Efficient Exploration**: Focused search in promising regions
- **Robust Fallbacks**: Multiple safety mechanisms
- **Configurable Performance**: Adjustable for speed vs accuracy

This approach combines the reliability of grid search with the intelligence of Bayesian optimization, providing an optimal balance of exploration and exploitation for SR parameter optimization.