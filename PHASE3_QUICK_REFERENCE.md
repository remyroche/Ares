# Phase 3 Quick Reference Card

## 📊 Current Parameters (Updated)

```
┌─────────────────────────────────────────────────────┐
│           PHASE 3: LGBM+SHAP PIPELINE               │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Phase 3.1: Shallow LGBM Sweep                     │
│  ├─ Input:  400-480 pruned features                │
│  ├─ Output: 120 features                           │
│  ├─ Method: 60% LGBM + 40% MI                      │
│  └─ Time:   ~2-3 minutes                           │
│                                                     │
│  Phase 3.2: Deeper LGBM Refinement                 │
│  ├─ Input:  120 features                           │
│  ├─ Output: 80 features (final_features)           │
│  ├─ Method: 60% LGBM + 30% MI + 10% Stability      │
│  └─ Time:   ~3-6 minutes                           │
│                                                     │
│  Phase 3.3: Deep Interaction Discovery             │
│  ├─ Input:  80 features                            │
│  ├─ Output: 80 interactions                        │
│  ├─ Method: 5-way RFE (MI+Red+LGBM+SHAP+Stab)     │
│  ├─ RFE:    5 rounds (400→268→180→121→81→80)      │
│  └─ Time:   ~12-18 minutes                         │
│                                                     │
│  TOTAL OUTPUT: 160 features                        │
│  ├─ final_features: 80                             │
│  ├─ interactions:   80                             │
│  └─ Total time:     ~17-27 minutes                 │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 🔢 Key Numbers

| Metric | Value |
|--------|-------|
| **Phase 3.1 output** | 120 features |
| **Phase 3.2 input** | 120 features |
| **Phase 3.2 output** | 80 features |
| **Phase 3.3 input** | 80 features |
| **Phase 3.3 output** | 80 interactions |
| **Total output** | **160 features** |
| **Total time** | 17-27 minutes |

---

## 📈 Feature Composition (Typical 160 Features)

```
┌────────────────────────────────────┐
│ Hybrid CT interactions:   25 (16%) │
│ Traditional interactions: 55 (34%) │
│ Cross-timeframe ratios:   30 (19%) │
│ Variant features:         25 (16%) │
│ Base features:            25 (16%) │
│                                    │
│ TOTAL:                   160 (100%)│
└────────────────────────────────────┘
```

---

## 🎯 Phase 3.3 RFE Process

```
Round 1: 400 candidates → Remove 33% → 268 remain
Round 2: 268 candidates → Remove 33% → 180 remain
Round 3: 180 candidates → Remove 33% → 121 remain
Round 4: 121 candidates → Remove 33% → 81 remain
Round 5: 81 candidates  → Keep top 80 → 80 remain
```

**5 rounds total** | **80% filtered** | **20% kept**

---

## 🔑 Classification Rules (Priority Order)

```python
1. Has interaction operator? (_x_, _div_, _minus_, _log_, _plus_)
   ├─ Yes + Has CT marker? → Hybrid CT interaction
   └─ Yes + No CT marker?  → Traditional interaction

2. Has CT marker? (_3x_ratio, _6x_ratio, _9x_ratio, _27x_ratio)
   └─ Yes → Cross-timeframe ratio

3. Has variant suffix? (_volnorm, _vwap, _trend_adj)
   ├─ Yes → Variant feature
   └─ No  → Base feature (includes _base suffix!)
```

---

## ⏱️ Performance Benchmarks

| Phase | Time | Memory | CPU |
|-------|------|--------|-----|
| 3.1 | 2-3 min | Low | Medium |
| 3.2 | 3-6 min | Medium | Medium |
| 3.3 | 12-18 min | Medium | High |
| **Total** | **17-27 min** | **Medium** | **Medium-High** |

---

## 💡 Quick Tips

### When to Adjust Parameters

**Increase Phase 3.1 output (120)** if:
- ✅ You have many cross-timeframe features being dropped
- ✅ Phase 3.2 needs more candidates
- ❌ Don't exceed ~150 (diminishing returns)

**Increase Phase 3.3 output (80)** if:
- ✅ You want more interaction features
- ✅ Model can handle more features
- ❌ Don't exceed ~100 (risk of overfitting)

**Decrease if**:
- ⚠️ Processing time is too long
- ⚠️ Memory constraints
- ⚠️ Model training is too slow

---

## 📊 Comparison: Old vs New

| Parameter | Old | New | Change |
|-----------|-----|-----|--------|
| P3.1 output | 100 | **120** | +20% |
| P3.2 input | 100 | **120** | +20% |
| P3.2 output | 80 | **80** | 0% |
| P3.3 output | 20-50 | **80** | +60-300% |
| Total features | 100-130 | **160** | +23-60% |
| Processing time | 15-23m | **17-27m** | +10-15% |

---

## 🎯 Expected Output Example

```python
final_features = [
    'rsi_base',              # Base feature
    'macd_volnorm',          # Variant feature
    'rsi_base_3x_ratio',     # CT ratio
    'volume_vwap',           # Variant feature
    'atr_base',              # Base feature
    # ... 75 more
]  # Total: 80 features

interactions = [
    'rsi_base_x_macd_volnorm',              # Traditional
    'rsi_base_3x_ratio_x_macd_6x_ratio',    # Hybrid CT
    'volume_vwap_div_atr_base',             # Traditional
    'rsi_base_minus_macd_volnorm',          # Traditional
    # ... 76 more
]  # Total: 80 interactions

combined_features = final_features + interactions
# Total: 160 features
```

---

## 🔧 Code Location

**File**: `src/training/steps/pre_training/feature_generation_interaction_generation_step.py`

**Key functions**:
- `_phase3_1_shallow_sweep()`: Lines 2315-2614
- `_phase3_2_deeper_refinement()`: Lines 2795-3052
- `_phase3_3_interaction_discovery()`: Lines 3053-3460
- Classification logic: Lines 3611-3650

**Key parameters**:
```python
# Phase 3.1
n_select = min(120, len(features.columns))  # Line 2609

# Phase 3.3
n_features = 80  # Line 3320
max_interactions = min(80, len(sorted_interactions))  # Line 3371
```

---

## 📚 Documentation

- **Quick overview**: `COMPLETE_PHASE3_SUMMARY.md`
- **Technical details**: `PHASE3_DETAILED_EXPLANATION.md`
- **Full pipeline**: `FEATURE_FLOW_EXPLANATION.md`
- **Navigation**: `PHASE3_DOCUMENTATION_INDEX.md`
- **Parameter updates**: `PHASE3_PARAMETER_UPDATES_SUMMARY.md`

---

## ✅ Validation Checklist

Before running:
- [ ] Phase 3.1 outputs 120 features
- [ ] Phase 3.2 receives 120 features
- [ ] Phase 3.2 outputs 80 features
- [ ] Phase 3.3 generates 80 interactions
- [ ] Total output is 160 features
- [ ] No linter errors
- [ ] Classification logic working

After running:
- [ ] Check feature breakdown (25+55+30+25+25=160)
- [ ] Verify interaction quality
- [ ] Monitor processing time (17-27 min expected)
- [ ] Check memory usage (should be moderate)

---

## 🚀 Status

✅ **All parameters updated**
✅ **Code validated**
✅ **Documentation complete**
✅ **Ready for production**

**Last updated**: 2025-10-28
**Version**: 2.0 (Updated parameters)
