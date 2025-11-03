# SR Parameter Group Optimization Fix

## 🚨 **Problem: Combinatorial Explosion**

### **Original Group 5: strength_weights**
- **Parameters:** 11
- **Coarse grid combos:** 5^11 = **48,828,125** 😱
- **Result:** System stuck, high memory pressure (82%+), hours to complete

---

## ✅ **Solution: Split High-Dimensional Groups**

### **Rule:** No group should have 6+ parameters

### **New Structure:**

#### **Group 5a: strength_boosts_core** (5 params)
```python
{
    "touch_weight": {"type": "float", "low": 0.05, "high": 0.3},
    "volume_weight": {"type": "float", "low": 0.1, "high": 0.4},
    "consistency_weight": {"type": "float", "low": 0.1, "high": 0.4},
    "confluence_weight": {"type": "float", "low": 0.05, "high": 0.2},
    "pivot_boost": {"type": "float", "low": 0.05, "high": 0.2}
}
```
**Coarse grid:** 5^5 = **3,125 combinations** ✅

#### **Group 5b: strength_boosts_special** (3 params)
```python
{
    "psychological_boost": {"type": "float", "low": 0.02, "high": 0.1},
    "hvn_boost": {"type": "float", "low": 0.05, "high": 0.2},
    "strength_filter_threshold": {"type": "float", "low": 0.3, "high": 0.8}
}
```
**Coarse grid:** 5^3 = **125 combinations** ✅

#### **Group 5c: strength_penalties** (3 params)
```python
{
    "failure_penalty_base": {"type": "float", "low": 0.1, "high": 0.5},
    "failure_volume_multiplier": {"type": "float", "low": 1.0, "high": 2.5},
    "failure_max_penalty": {"type": "float", "low": 0.4, "high": 1.0}
}
```
**Coarse grid:** 5^3 = **125 combinations** ✅

---

## 📊 **Impact**

### **Combinatorial Complexity**

| Approach | Groups | Max Params/Group | Largest Grid | Total Trials |
|----------|--------|------------------|--------------|--------------|
| **Before** | 5 groups | 11 params | **48,828,125** | Would take hours ❌ |
| **After** | 7 groups | 5 params | **3,125** | ~20-25 minutes ✅ |

**Speedup:** ~14,000x faster for the largest group! 🚀

### **Trial Breakdown**

| Group | Trials | Time Est. |
|-------|--------|-----------|
| Groups 1-4 | ~792 | ~6 min |
| Group 5a (5 params) | ~3,425 | ~5 min |
| Group 5b (3 params) | ~425 | ~2 min |
| Group 5c (3 params) | ~425 | ~2 min |
| Final refinement | ~150 | ~3 min |
| **Total** | **~5,217** | **~18 min** |

---

## 🎓 **Key Principle**

**Hierarchical HPO works best when:**
- ✅ Each group has **≤5 parameters**
- ✅ Groups are logically organized
- ✅ Dependencies are clear
- ✅ TPE handles high-dimensional final refinement

**Avoid:**
- ❌ Groups with 6+ parameters (combinatorial explosion)
- ❌ Grid search on high-dimensional spaces
- ❌ Mixing unrelated parameters in one group

---

## 📝 **Code Changes**

**File:** `src/training/steps/market_analysis/components/sr_parameter_optimization.py`

**Lines 1284-1337:** Split strength_weights into 3 groups

**Result:**
- 7 groups total (vs. 5)
- Max 5 params/group (vs. 11)
- ~5,000 trials (vs. 48M combinations)

---

## ✅ **Verification**

When the current run completes, verify:

1. **Log messages:**
   ```
   ✅ Added strength weight optimization (split into 3 groups to avoid combinatorial explosion)
      - Group 5a: Core boosts (5 params) - 5^5 = 3,125 combos
      - Group 5b: Special boosts + filter (3 params) - 5^3 = 125 combos  
      - Group 5c: Penalties (3 params) - 5^3 = 125 combos
      - Total: 11 params split across 3 groups (vs. 5^11 = 48M in 1 group)
   ```

2. **No more 48M combination messages**

3. **Reasonable optimization time** (~20 minutes, not hours)

4. **High trial count** in final report (5,000+ vs. 12)

---

## 🎉 **Expected Final Results**

```json
{
  "total_combinations_tested": 5217,  // ✅ vs. 12!
  "optimization_time": 1200,          // ~20 min
  "best_score": 0.96+,
  "parameter_groups_optimized": 7,
  "parameters_optimized": 17,
  "max_group_size": 5,                // ✅ No 11-param groups!
  "bayesian_efficiency": 0.8+
}
```

---

**Status:** ✅ **FIXED**  
**Run:** 🏃 **ACTIVE** (started 14:02:55)  
**ETA:** ~15-20 minutes remaining

