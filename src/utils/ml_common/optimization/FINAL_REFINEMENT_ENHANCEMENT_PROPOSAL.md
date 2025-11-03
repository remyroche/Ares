# Final Refinement Enhancement Proposal

## Current State Analysis

### What Final Refinement Does
After multiple rounds of group-by-group optimization, final refinement:
1. Combines ALL parameter groups into one
2. Creates a narrowed search space (±10% of original range)
3. Runs TPE optimization (50 trials default)
4. Captures interaction effects between parameter groups

### Current Limitations

**Problem 1: Uniform Narrowing**
```python
# Current approach: 10% of range for ALL parameters
narrow_factor = 0.1  # Fixed for all parameters

# Example:
n_estimators: [50, 500] → best=200 → narrowed to [155, 245]  # ±45 units
learning_rate: [0.01, 0.3] → best=0.1 → narrowed to [0.071, 0.129]  # ±0.029 units

# Issue: Different parameters have different sensitivities!
# learning_rate might need tighter focus (more sensitive)
# n_estimators might benefit from wider range (less sensitive)
```

**Problem 2: No Log-Space Narrowing**
```python
# For log-scale parameters (learning_rate has log=True):
learning_rate: [0.01, 0.3]
# Linear narrowing: ±10% of 0.29 = ±0.029
# But in log space: log(0.01)=-4.6, log(0.3)=-1.2, range=3.4
# Should narrow in log space for proper scaling!
```

**Problem 3: No Parameter Importance**
- All parameters treated equally
- Critical parameters should get more attention
- Insensitive parameters should get less narrowing

---

## 🚀 Enhancement Options

### Option 1: Log-Space Narrowing (Easy Win) ✅

**Implementation:**
```python
def _create_narrowed_search_space_enhanced(
    self,
    search_space: Dict[str, Dict[str, Any]],
    best_params: Dict[str, Any],
    narrow_factor: float = 0.1,
    use_log_space: bool = True  # NEW
) -> Dict[str, Dict[str, Any]]:
    """Enhanced narrowing with log-space support."""
    
    narrowed = {}
    
    for param_name, param_config in search_space.items():
        if param_name not in best_params:
            narrowed[param_name] = param_config.copy()
            continue
        
        best_value = best_params[param_name]
        narrowed_config = param_config.copy()
        
        if param_config['type'] == 'float':
            low, high = param_config['low'], param_config['high']
            
            # Check if this is a log-scale parameter
            if use_log_space and param_config.get('log', False):
                # Narrow in log space
                log_low = np.log(max(low, 1e-10))
                log_high = np.log(max(high, 1e-10))
                log_best = np.log(max(best_value, 1e-10))
                log_range = log_high - log_low
                
                narrow_log_range = log_range * narrow_factor
                narrowed_log_low = max(log_low, log_best - narrow_log_range)
                narrowed_log_high = min(log_high, log_best + narrow_log_range)
                
                # Convert back to linear space
                narrowed_config['low'] = max(low, np.exp(narrowed_log_low))
                narrowed_config['high'] = min(high, np.exp(narrowed_log_high))
            else:
                # Linear narrowing (current approach)
                range_size = high - low
                narrow_range = range_size * narrow_factor
                
                narrowed_config['low'] = max(low, best_value - narrow_range)
                narrowed_config['high'] = min(high, best_value + narrow_range)
        
        elif param_config['type'] == 'int':
            # Integer narrowing (unchanged)
            low, high = param_config['low'], param_config['high']
            narrow_amount = max(1, int((high - low) * narrow_factor))
            
            narrowed_config['low'] = max(low, best_value - narrow_amount)
            narrowed_config['high'] = min(high, best_value + narrow_amount)
        
        narrowed[param_name] = narrowed_config
    
    return narrowed
```

**Benefits:**
- ✅ Easy to implement (10 lines of code)
- ✅ Proper handling of log-scale parameters
- ✅ No dependencies on other code
- ✅ Immediate improvement

**Example Impact:**
```python
# learning_rate: [0.01, 0.3], log=True, best=0.1

# Before (linear narrowing):
narrowed: [0.071, 0.129]  # ±29%

# After (log-space narrowing):
log_space: [log(0.01), log(0.3)] = [-4.6, -1.2]
best_log: log(0.1) = -2.3
narrowed_log: [-2.64, -1.96]  # ±10% of log range
narrowed: [0.07, 0.14]  # ±40% in linear space (more appropriate!)
```

---

### Option 2: Adaptive Narrowing with Trial History (Better) ✅✅

**Implementation:**
```python
def _calculate_parameter_importance(self) -> Dict[str, float]:
    """
    Calculate parameter importance from optimization history.
    Parameters with more impact on score get higher importance.
    """
    if not self.group_results:
        return {}
    
    # Analyze all trials to determine parameter sensitivity
    importance = {}
    
    for group_result in self.group_results:
        for trial in group_result.all_trials:
            params = trial['params']
            score = trial['score']
            
            # Calculate sensitivity for each parameter
            for param_name, param_value in params.items():
                if param_name not in importance:
                    importance[param_name] = []
                
                importance[param_name].append({
                    'value': param_value,
                    'score': score
                })
    
    # Calculate sensitivity scores
    sensitivity = {}
    for param_name, data in importance.items():
        if len(data) < 2:
            sensitivity[param_name] = 0.5
            continue
        
        # Calculate correlation between parameter value and score
        values = [d['value'] for d in data]
        scores = [d['score'] for d in data]
        
        try:
            # Higher correlation = more important parameter
            corr = abs(np.corrcoef(values, scores)[0, 1])
            sensitivity[param_name] = corr if not np.isnan(corr) else 0.5
        except:
            sensitivity[param_name] = 0.5
    
    return sensitivity

def _create_adaptive_narrowed_space(
    self,
    search_space: Dict[str, Dict[str, Any]],
    best_params: Dict[str, Any],
    importance_weights: Dict[str, float],
    base_narrow_factor: float = 0.1
) -> Dict[str, Dict[str, Any]]:
    """
    Adaptive narrowing: important parameters get more focus.
    """
    narrowed = {}
    
    for param_name, param_config in search_space.items():
        # Get importance (default to 0.5 if not calculated)
        importance = importance_weights.get(param_name, 0.5)
        
        # Adaptive narrow factor:
        # High importance → narrow MORE (focus optimization)
        # Low importance → narrow LESS (allow exploration)
        adaptive_factor = base_narrow_factor * (0.5 + importance)
        
        narrowed[param_name] = self._narrow_param_with_factor(
            param_config,
            best_params.get(param_name),
            adaptive_factor
        )
    
    return narrowed
```

**Benefits:**
- ✅ Focuses on important parameters
- ✅ Data-driven narrowing
- ✅ Better final convergence

---

### Option 3: Pareto-Based Multi-Objective Refinement (Advanced) ✅✅✅

**Leverage `pareto.py` for final refinement:**

```python
def _final_refinement_with_pareto(
    self,
    X_train, y_train, X_val, y_val, model,
    current_best_params: Dict[str, Any]
) -> OptimizationResult:
    """
    Enhanced final refinement using Pareto optimization.
    
    Instead of single-objective TPE, use multi-objective Pareto approach:
    1. Optimize for financial performance separately
    2. Optimize for statistical accuracy separately  
    3. Find Pareto-optimal solutions
    4. Select knee point as final solution
    """
    from .pareto import ParetoFront, Solution, select_knee_point
    
    logger.info(f"    Running Pareto-enhanced final refinement ({self.final_refinement_trials} trials)")
    
    # Combine all parameter groups
    all_params = {}
    for group in self.param_groups:
        all_params.update(group.params)
    
    # Create narrowed search space (with log-space support)
    narrow_space = self._create_narrowed_search_space_enhanced(
        all_params, 
        current_best_params,
        use_log_space=True  # ← Use Option 1
    )
    
    # Multi-objective optimization
    solutions = []
    
    for trial_num in range(self.final_refinement_trials):
        # Sample parameters
        params = self._sample_random_params(narrow_space)
        
        # Evaluate
        full_params = params
        score = self._evaluate_params(full_params, X_train, y_train, X_val, y_val, model)
        
        # Get component scores if using custom_balanced_score
        if self.scoring_metric == 'custom_balanced_score':
            # Extract financial and statistical components
            financial_score, statistical_score = self._get_component_scores(
                full_params, X_train, y_train, X_val, y_val, model
            )
            
            solutions.append(Solution(
                metrics={
                    'overall': score,
                    'financial': financial_score,
                    'statistical': statistical_score
                },
                params=params
            ))
        else:
            solutions.append(Solution(
                metrics={'score': score},
                params=params
            ))
    
    # Compute Pareto front
    pareto_front_obj = ParetoFront()
    
    if len(solutions) > 0 and 'financial' in solutions[0].metrics:
        # Multi-objective: find Pareto optimal solutions
        objectives = {'financial': 'max', 'statistical': 'max'}
        pareto_solutions = pareto_front_obj.compute_pareto_front_gpu(
            solutions, 
            objectives,
            use_gpu=False  # Use CPU for small final refinement
        )
        
        # Select knee point (best trade-off)
        best_solution = select_knee_point(
            pareto_solutions,
            objectives,
            weights={'financial': 0.6, 'statistical': 0.4}  # Match our scoring
        )
        
        best_params = best_solution.params
        best_score = best_solution.metrics['overall']
        
        logger.info(f"    Found {len(pareto_solutions)} Pareto-optimal solutions")
        logger.info(f"    Selected knee point with score: {best_score:.6f}")
    else:
        # Single-objective: select best score
        best_solution = max(solutions, key=lambda s: s.metrics.get('score', s.metrics.get('overall', 0)))
        best_params = best_solution.params
        best_score = best_solution.metrics.get('score', best_solution.metrics.get('overall', 0))
    
    return OptimizationResult(
        group_name="final_refinement",
        stage=OptimizationStage.TPE,
        best_params=best_params,
        best_score=best_score,
        n_trials=len(solutions),
        optimization_time=0.0,
        all_trials=[
            {'params': s.params, 'score': s.metrics.get('overall', s.metrics.get('score', 0)), 'trial_number': i}
            for i, s in enumerate(solutions)
        ]
    )
```

**Benefits:**
- ✅ Finds Pareto-optimal solutions
- ✅ Better trade-offs between financial/statistical
- ✅ Uses existing Pareto utilities
- ✅ More robust final convergence

---

## 💡 Recommendation

**Implement Option 1 + Option 2 hybrid:**

1. **Quick Win**: Add log-space narrowing (Option 1)
   - Simple, immediate improvement
   - Proper handling of log-scale parameters
   - ~20 lines of code

2. **Future Enhancement**: Add adaptive narrowing (Option 2)
   - Use trial history to calculate importance
   - Adaptive factors based on sensitivity
   - Better final convergence

3. **Advanced (Optional)**: Pareto-based refinement (Option 3)
   - For multi-objective scenarios
   - When you want explicit financial vs statistical trade-offs

---

## 🎯 Recommended Implementation

**Step 1: Enhance `_create_narrowed_search_space` with log-space support**

```python
def _create_narrowed_search_space(
    self,
    search_space: Dict[str, Dict[str, Any]],
    best_params: Dict[str, Any],
    narrow_factor: float = 0.1,
    use_log_space_narrowing: bool = True  # NEW - enable by default
) -> Dict[str, Dict[str, Any]]:
    """
    Create a narrowed search space around best parameters.
    
    Enhanced with:
    - Log-space narrowing for log-scale parameters
    - Proper handling of different parameter scales
    
    Args:
        search_space: Original search space
        best_params: Best parameters found so far
        narrow_factor: Factor to narrow range (0.1 = ±10% of original range)
        use_log_space_narrowing: If True, narrow log-scale params in log space
    
    Returns:
        Narrowed search space with proper scaling
    """
    narrowed = {}
    
    for param_name, param_config in search_space.items():
        if param_name not in best_params:
            narrowed[param_name] = param_config.copy()
            continue
        
        best_value = best_params[param_name]
        narrowed_config = param_config.copy()
        
        if param_config['type'] == 'float':
            low, high = param_config['low'], param_config['high']
            
            # Enhanced: narrow in log space for log-scale parameters
            if use_log_space_narrowing and param_config.get('log', False):
                # Log-space narrowing
                log_low = np.log(max(low, 1e-10))
                log_high = np.log(max(high, 1e-10))
                log_best = np.log(max(best_value, 1e-10))
                log_range = log_high - log_low
                
                narrow_log_range = log_range * narrow_factor
                narrowed_log_low = max(log_low, log_best - narrow_log_range)
                narrowed_log_high = min(log_high, log_best + narrow_log_range)
                
                narrowed_config['low'] = max(low, np.exp(narrowed_log_low))
                narrowed_config['high'] = min(high, np.exp(narrowed_log_high))
                
                logger.debug(
                    f"      {param_name} (log-scale): "
                    f"[{low:.4f}, {high:.4f}] → [{narrowed_config['low']:.4f}, {narrowed_config['high']:.4f}]"
                )
            else:
                # Linear narrowing (original approach)
                range_size = high - low
                narrow_range = range_size * narrow_factor
                
                narrowed_config['low'] = max(low, best_value - narrow_range)
                narrowed_config['high'] = min(high, best_value + narrow_range)
                
                logger.debug(
                    f"      {param_name} (linear): "
                    f"[{low:.4f}, {high:.4f}] → [{narrowed_config['low']:.4f}, {narrowed_config['high']:.4f}]"
                )
        
        elif param_config['type'] == 'int':
            # Integer narrowing (unchanged but add logging)
            low, high = param_config['low'], param_config['high']
            narrow_amount = max(1, int((high - low) * narrow_factor))
            
            narrowed_config['low'] = max(low, best_value - narrow_amount)
            narrowed_config['high'] = min(high, best_value + narrow_amount)
            
            logger.debug(
                f"      {param_name} (int): "
                f"[{low}, {high}] → [{narrowed_config['low']}, {narrowed_config['high']}]"
            )
        
        # Categorical parameters stay the same
        narrowed[param_name] = narrowed_config
    
    return narrowed
```

**Impact:**
```python
# Example: learning_rate with log=True
Original: [0.01, 0.3]
Best: 0.1

# Before (linear):
Narrowed: [0.071, 0.129]  # ±29%

# After (log-space):
Log space: [-4.6, -1.2], best_log=-2.3
Narrowed log: [-2.64, -1.96]  # ±10% of log range
Narrowed linear: [0.07, 0.14]  # ±40% in linear space
# More appropriate for log-scale parameter!
```

---

## 📊 Expected Benefits

### 1. Better Convergence
- Log-scale parameters narrowed appropriately
- Proper scaling reduces wasted trials
- More efficient exploration in final refinement

### 2. More Robust
- Handles different parameter scales correctly
- Respects parameter type (linear vs log)
- Better final optimization

### 3. Minimal Code Change
- ~30 lines of code
- No breaking changes
- Backward compatible (controlled by flag)

---

## 🎯 Next Steps (Optional Advanced Enhancements)

### After Option 1 is working:

**Add Parameter Importance Analysis:**
```python
# Calculate from trial history
importance = self._analyze_parameter_importance(self.group_results)

# Use in final refinement
adaptive_narrow_factors = {
    param: base_factor * (0.5 + importance[param])
    for param in all_params
}
```

**Add Pareto Multi-Objective:**
```python
# For custom_balanced_score users
if self.scoring_metric == 'custom_balanced_score':
    # Optimize financial and statistical separately
    pareto_solutions = self._pareto_final_refinement(...)
    best = select_knee_point(pareto_solutions)
```

---

## ✅ Recommendation

**Implement Option 1 (Log-Space Narrowing) NOW:**
- Simple enhancement
- Immediate benefit
- No dependencies
- ~30 lines of code
- Proper handling of parameter scales

**Would you like me to implement this?**

It will make final refinement:
- ✅ Respect log-scale parameters
- ✅ More efficient (better narrowing)
- ✅ Better final convergence
- ✅ Leverage proper scaling

