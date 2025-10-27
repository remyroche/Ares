# Quick Tuning Guide - Improve Your Clustering Metrics

## Current State ✅
You have **8 clusters** working! But metrics can be improved.

## Your Current Metrics
```
CV Score:            1.19  ✅ Good
Silhouette Score:   -0.03  ❌ Needs improvement  
DBI Score:           3.2   ❌ Needs improvement
Balance:             0.63  ✅ Moderate
Temporal Smoothness: 0.987 ✅ Excellent
```

## 🚀 Quick Commands

### Option 1: Quick Tuning (15 minutes)
```bash
python3 src/training/steps/market_analysis/clusters/run_iterative_opt_tuning.py \
    --symbol ETHUSDT \
    --n-trials 15 \
    --method bayesian
```

### Option 2: Best Results (30 minutes)
```bash
python3 src/training/steps/market_analysis/clusters/run_iterative_opt_tuning.py \
    --symbol ETHUSDT \
    --n-trials 30 \
    --method bayesian
```

### Option 3: Comprehensive (60 minutes)
```bash
python3 src/training/steps/market_analysis/clusters/run_iterative_opt_tuning.py \
    --symbol ETHUSDT \
    --n-trials 50 \
    --method multiobjective
```

## 📊 Expected After Tuning

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Silhouette | -0.03 | **0.15-0.30** | 🚀 Much better |
| DBI Score | 3.2 | **1.5-2.2** | ⬇️ 30-50% lower |
| CV Score | 1.19 | **1.3-1.6** | ⬆️ 10-35% higher |
| Balance | 0.63 | **0.65-0.75** | ➡️ Maintained |
| Temporal | 0.987 | **0.95-0.99** | ➡️ Maintained |

## 📂 Where to Find Results

After tuning completes, check:
```bash
# View the report
cat artifacts/hyperparameter_tuning/optimization_report_*.md

# View the JSON results
cat artifacts/hyperparameter_tuning/optimization_results_*.json
```

## ⚙️ Applying the Results

### Step 1: Get Best Parameters
```bash
# Find the most recent results file
ls -t artifacts/hyperparameter_tuning/optimization_results_*.json | head -1

# View it
cat artifacts/hyperparameter_tuning/optimization_results_*.json | jq '.best_params'
```

### Step 2: Update OptConfig

Edit `src/training/steps/market_analysis/clusters/iterative_optimization.py` (lines 2489-2562):

```python
@dataclass
class OptConfig:
    # Copy values from best_params
    K_MIN: int = 6  # From best_params['K_MIN']
    K_MAX: int = 10  # From best_params['K_MAX']
    MIN_FRAC: float = 0.03  # From best_params['MIN_FRAC']
    MAX_FRAC: float = 0.20  # From best_params['MAX_FRAC']
    
    # Objective weights
    w_cv: float = 0.70  # From best_params['w_cv']
    w_temp: float = 0.20  # From best_params['w_temp']
    w_sil: float = 0.10  # From best_params['w_sil']
    w_bal: float = 0.05  # From best_params['w_bal']
    
    # Optimization parameters
    max_rounds: int = 40  # From best_params['max_rounds']
    eps_std_step1: float = -0.20  # From best_params['eps_std_step1']
    sil_guard: float = -0.08  # From best_params['sil_guard']
    temporal_bonus: float = 0.25  # From best_params['temporal_bonus']
    
    # ... continue for other parameters
```

### Step 3: Test the Improvement

```bash
python3 src/launcher/ares_launcher.py --step regime_clustering --symbol ETHUSDT --execution-mode light
```

Check if metrics improved!

## 🎯 What Each Parameter Does

### Quick Reference
| Parameter | What it does | Higher = | Lower = |
|-----------|--------------|----------|---------|
| `w_cv` | Focus on variance separation | More distinct regimes | More compact clusters |
| `w_sil` | Focus on cluster cohesion | Better internal consistency | Allows looser clusters |
| `w_temp` | Focus on time stability | More stable regimes | More responsive to changes |
| `eps_std_step1` | Acceptance threshold | More aggressive (-0.3) | More conservative (-0.1) |
| `K_MIN` | Minimum clusters | More regimes | Fewer regimes |
| `max_rounds` | Optimization iterations | More refinement | Faster but less optimal |

## 🔍 Monitoring Progress

During tuning, you'll see output like:
```
✅ Trial 5: CV=1.32, Sil=0.18, DBI=2.1, Balance=0.68, Temporal=0.94, K=7, Score=0.6543
✅ Trial 10: CV=1.45, Sil=0.25, DBI=1.9, Balance=0.72, Temporal=0.96, K=8, Score=0.7234
```

The `Score` is the composite weighted score - higher is better!

## 📈 Interpreting Results

### Good Signs
- ✅ Silhouette > 0.15
- ✅ DBI < 2.5
- ✅ CV > 1.2
- ✅ Balance > 0.6
- ✅ Temporal > 0.90

### Warning Signs
- ⚠️ Silhouette < 0
- ⚠️ DBI > 4.0
- ⚠️ Balance < 0.5
- ⚠️ Temporal < 0.85
- ⚠️ Clusters outside 6-8 range

## 🛠️ Troubleshooting

### Tuning is slow
- Reduce `--n-trials` to 10-15
- Use faster machine
- Run overnight for comprehensive tuning

### All trials fail constraints
Edit `iterative_optimization_tuner.py` line 67:
```python
def meets_constraints(self,
                     min_balance: float = 0.45,  # Lowered from 0.5
                     min_temporal: float = 0.80,  # Lowered from 0.85
                     target_clusters: Tuple[int, int] = (5, 9)):  # Wider range
```

### Metrics don't improve
- Try `--method multiobjective` for Pareto analysis
- Increase trials to 50-100
- Review and adjust parameter ranges in `OptimizationParameterSpace`

## 📚 Full Documentation

For complete details, see:
- `ITERATIVE_OPTIMIZATION_SUCCESS_SUMMARY.md` - Full implementation summary
- `ITERATIVE_OPT_TUNING_README.md` - Complete tuning documentation
- `src/training/steps/market_analysis/clusters/iterative_optimization_tuner.py` - Source code with docstrings

## ✨ Success Criteria

You know tuning worked when:
1. ✅ Silhouette score becomes positive (>0.1)
2. ✅ DBI score drops below 2.5
3. ✅ CV score increases (>1.3)
4. ✅ Balance and Temporal stay high
5. ✅ Still get 6-8 clusters

## 🎓 Pro Tips

1. **Start with bayesian method** - Usually finds good solutions faster
2. **Use multiobjective for exploration** - Shows trade-offs between metrics
3. **Run tuning overnight** - 50-100 trials gives best results
4. **Save baseline first** - Note current metrics before changing parameters
5. **Test incrementally** - Apply and test parameters one section at a time

Good luck improving your clustering quality! 🚀

