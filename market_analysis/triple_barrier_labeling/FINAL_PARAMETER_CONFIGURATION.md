# Final Triple Barrier Parameter Configuration

## 🎯 **Three-Stage Optimization with 7×7×7 Grids**

Updated configuration using 3 stages of 7×7×7 grids for optimal balance between exploration and efficiency.

---

## 📊 **Three-Stage Grid Configuration**

### **Stage 1: Coarse Grid Search**
- **Grid Size**: 7 × 7 × 7 = **343 combinations**
- **Purpose**: Initial exploration of parameter space
- **Selection**: Top 8 candidates

### **Stage 2: Fine Grid Search**  
- **Grid Size**: 7 × 7 × 7 = **343 combinations**
- **Purpose**: Refined search around promising regions
- **Selection**: Top 5 candidates

### **Stage 3: Bayesian Optimization**
- **Trials**: 100 trials
- **Purpose**: Fine-tune in optimal parameter space
- **Selection**: Best single parameter set

---

## 🔍 **Parameter Details**

### **1. `pt_mult` (Profit Target Multiplier)**
- **Range**: `0.005` to `0.02` (0.5% to 2.0%)
- **Grid Points**: 7 evenly spaced values
- **Example Values**: 
  ```
  [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02]
  ```

### **2. `sl_mult` (Stop Loss Multiplier)**
- **Range**: `0.002` to `0.01` (0.2% to 1.0%)
- **Grid Points**: 7 evenly spaced values
- **Example Values**:
  ```
  [0.002, 0.0033, 0.0047, 0.006, 0.0073, 0.0087, 0.01]
  ```

### **3. `time_barrier` (Time Barrier)**
- **Range**: `20` to `90` minutes
- **Grid Points**: 7 evenly spaced values
- **Example Values**:
  ```
  [20, 31, 43, 55, 66, 78, 90]
  ```

---

## 🔄 **Optimization Process**

### **Stage 1: Coarse Grid (343 combinations)**
```
pt_mult:    7 points
sl_mult:    7 points
time_barrier: 7 points
Total: 7 × 7 × 7 = 343 combinations
```

### **Stage 2: Fine Grid (343 combinations)**
- Refines around top 8 candidates from Stage 1
- Uses 30% refinement factor
- Tests 343 combinations in narrowed parameter space

### **Stage 3: Bayesian (100 trials)**
- Fine-tunes around top 5 candidates from Stage 2
- Uses Optuna TPE sampler
- Continuous parameter optimization

---

## 📈 **Search Space Progression**

```
Stage 1: 343 combinations (7³)
    ↓ (Select top 8)
Stage 2: 343 combinations (7³) in refined space
    ↓ (Select top 5)
Stage 3: 100 trials in ultra-refined space
    ↓ (Select best 1)
Final: Optimal parameters
```

---

## 🎯 **Why 7×7×7 Grids?**

### **Advantages**
- **Balanced Exploration**: 7 points per dimension provides good coverage
- **Efficient**: 343 combinations per stage (manageable computation)
- **Progressive Refinement**: Each stage focuses on promising regions
- **Optimal Granularity**: Not too coarse, not too fine

### **Comparison**
| Grid Size | Combinations | Exploration | Efficiency |
|-----------|-------------|-------------|------------|
| 5×5×5     | 125         | Too coarse  | Very fast  |
| 7×7×7     | 343         | **Optimal** | **Fast**   |
| 10×10×10  | 1,000       | Good        | Slower     |
| 15×15×15  | 3,375       | Very fine   | Slow       |

---

## 🔧 **Configuration Example**

```python
from market_analysis.triple_barrier_labeling import (
    EnhancedOptimizedTripleBarrierLabeler,
    CoarseGridConfig,
    FineGridConfig,
    BayesianConfig
)

# Three-stage configuration with 7×7×7 grids
coarse_config = CoarseGridConfig(
    pt_mult_range=(0.005, 0.02),      # 0.5% to 2.0%
    sl_mult_range=(0.002, 0.01),      # 0.2% to 1.0%
    time_barrier_range=(20, 90),      # 20 to 90 minutes
    grid_size=7,                      # 7³ = 343 combinations
    top_k_candidates=8                # Top 8 candidates
)

fine_config = FineGridConfig(
    refinement_factor=0.3,             # 30% of original range
    grid_size=7,                      # 7³ = 343 combinations
    top_k_candidates=5,               # Top 5 candidates
    min_range_size=0.001              # Minimum range size
)

bayesian_config = BayesianConfig(
    n_trials=100,                     # 100 optimization trials
    objective_function="combined",     # Combined objective
    acquisition_function="EI",        # Expected Improvement
    early_stopping_patience=20        # Early stopping patience
)

# Initialize labeler
labeler = EnhancedOptimizedTripleBarrierLabeler({
    'coarse_grid_config': coarse_config,
    'fine_grid_config': fine_config,
    'bayesian_config': bayesian_config
})
```

---

## 📊 **Performance Characteristics**

### **Total Combinations**
- **Stage 1**: 343 combinations
- **Stage 2**: 343 combinations  
- **Stage 3**: 100 trials
- **Total**: 786 evaluations

### **Time Distribution**
- **Stage 1**: ~40% of total time (exploration)
- **Stage 2**: ~40% of total time (refinement)
- **Stage 3**: ~20% of total time (fine-tuning)

### **Efficiency**
- **Faster than 10×10×10**: 3x fewer combinations per stage
- **More thorough than 5×5×5**: Better parameter coverage
- **Optimal balance**: Good exploration with reasonable speed

---

## 🎯 **Example Parameter Progression**

### **Stage 1: Coarse Grid**
```
Best candidate: pt_mult=0.01, sl_mult=0.005, time_barrier=60
Score: 0.75
```

### **Stage 2: Fine Grid (around best candidate)**
```
Refined ranges:
- pt_mult: 0.0085 to 0.0115 (30% of original range)
- sl_mult: 0.0035 to 0.0065 (30% of original range)  
- time_barrier: 45 to 75 (30% of original range)

Best candidate: pt_mult=0.0095, sl_mult=0.0048, time_barrier=55
Score: 0.82
```

### **Stage 3: Bayesian (ultra-refined)**
```
Ultra-refined ranges:
- pt_mult: 0.009 to 0.010 (15% of original range)
- sl_mult: 0.0045 to 0.0051 (15% of original range)
- time_barrier: 50 to 60 (15% of original range)

Final result: pt_mult=0.0097, sl_mult=0.0049, time_barrier=57
Score: 0.85
```

---

## ✅ **Benefits of 7×7×7 Configuration**

### **Efficiency**
- **3x faster** than 10×10×10 grids
- **Manageable computation** (343 combinations per stage)
- **Good parallelization** (343 combinations can be processed efficiently)

### **Quality**
- **Better exploration** than 5×5×5 grids
- **Progressive refinement** through three stages
- **Optimal parameter discovery** with reasonable computation

### **Practical**
- **Fast enough** for real-time optimization
- **Thorough enough** for production use
- **Scalable** to different parameter ranges

---

## 🚀 **Usage**

```python
# Initialize with 7×7×7 configuration
labeler = EnhancedOptimizedTripleBarrierLabeler()

# Run three-stage optimization
results = labeler.optimize_regime_parameters(data, regime_data)

# Create optimized labels
labels = labeler.create_optimized_labels(data, regime_data)

# Print results
labeler.print_optimization_report()
```

The 7×7×7 configuration provides the optimal balance between exploration quality and computational efficiency for triple barrier parameter optimization!