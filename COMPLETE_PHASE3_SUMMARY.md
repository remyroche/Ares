# Phase 3: LGBM+SHAP Pipeline - Quick Summary

## 🎯 What Phase 3 Does

Takes **~400-480 pruned features** and produces:
- ✅ **80 best features** (`final_features`) 
- ✅ **20-50 interaction features** (`interactions`)

---

## 📊 The Three Steps

```
Input: 400-480 pruned features
    ↓
┌─────────────────────────────────────────┐
│ Phase 3.1: Shallow LGBM Sweep           │
│ • Fast filtering                         │
│ • 60% LGBM importance + 40% MI          │
│ • Duration: ~2-3 minutes                │
└─────────────────────────────────────────┘
    ↓
100 features
    ↓
┌─────────────────────────────────────────┐
│ Phase 3.2: Deeper LGBM Refinement       │
│ • Accurate scoring                       │
│ • 60% LGBM + 30% MI + 10% Stability     │
│ • Duration: ~3-5 minutes                │
└─────────────────────────────────────────┘
    ↓
80 features (final_features)
    ↓
┌─────────────────────────────────────────┐
│ Phase 3.3: Deep Interaction Discovery   │
│ • Extract 80 feature pairs from trees   │
│ • Generate 400 candidates (80×5 ops)    │
│ • RFE with 5-way scoring (20% each):    │
│   1. Mutual Information                 │
│   2. Redundancy                         │
│   3. LGBM Importance                    │
│   4. SHAP Values                        │
│   5. Stability                          │
│ • Duration: ~10-15 minutes              │
└─────────────────────────────────────────┘
    ↓
20-50 interactions
```

**Total Duration**: ~15-23 minutes

---

## 🔑 Key Concepts

### final_features (80 features)
**What**: The best individual features that survived all selection rounds

**Contains**:
- Base features (e.g., `rsi_base`, `atr`)
- Variant features (e.g., `macd_volnorm`, `volume_vwap`)
- Cross-timeframe ratios (e.g., `rsi_base_3x_ratio`)

**Think of them as**: "Solo performers" - individually strong features

---

### interactions (20-50 features)
**What**: Newly discovered synergistic feature combinations

**Created from**: 80 final_features analyzed for co-occurrence in decision trees

**5 Operation Types**:
1. **Multiplication** (`_x_`): `rsi_base_x_macd_volnorm`
2. **Division** (`_div_`): `volume_vwap_div_atr_base`
3. **Subtraction** (`_minus_`): `macd_base_minus_rsi_volnorm`
4. **Log ratio** (`_log_`): `rsi_base_log_volume_vwap`
5. **Log of ratio** (`_log_ratio_`): `macd_base_log_ratio_atr_base`

**Special Case - Hybrid CT Interactions**:
```
rsi_base_3x_ratio_x_macd_6x_ratio
```
This is an interaction BETWEEN cross-timeframe features!

**Think of them as**: "Duets" - two features working together

---

## 🎭 The Magic of Phase 3.3

### How Interaction Discovery Works

**Step 1**: Train LGBM on the 80 final features
```
Model learns: "When RSI is high AND volume is low → price drops"
```

**Step 2**: Analyze decision trees to find co-occurring features
```
Trees show: rsi_base and volume_vwap often split near each other
→ They have a relationship!
```

**Step 3**: Generate interaction candidates
```
Create: rsi_base_x_volume_vwap
       rsi_base_div_volume_vwap
       rsi_base_minus_volume_vwap
       etc.
```

**Step 4**: Score with 5 different methods
```
Each interaction gets 5 scores:
1. How informative? (MI)
2. How unique? (Redundancy)
3. How useful in trees? (LGBM)
4. What's the true impact? (SHAP)
5. How stable? (Stability)

Combined score = average of all 5
```

**Step 5**: RFE (Recursive Feature Elimination)
```
Round 1: Score 400 → Remove worst 33% → 268 remain
Round 2: Score 268 → Remove worst 33% → 180 remain
Round 3: Score 180 → Remove worst 33% → 121 remain
Round 4: Score 121 → Remove worst 33% → 81 remain
Round 5: Score 81  → Remove worst 33% → 54 remain
Round 6: Score 54  → Keep best 50   → 50 remain
```

**Why RFE?** Interactions affect each other. A feature might look good with 400 candidates but redundant with 50. RFE re-evaluates at each stage.

---

## 💡 Real-World Example

### Input to Phase 3: 480 features
```
rsi_base
rsi_volnorm
rsi_vwap
rsi_trend_adj
rsi_base_3x_ratio
rsi_base_6x_ratio
... (474 more)
```

### After Phase 3.1: 100 features
```
rsi_base              ← Selected
rsi_volnorm           ← Selected
rsi_vwap              ← Removed (less important)
rsi_trend_adj         ← Removed
rsi_base_3x_ratio     ← Selected
macd_base             ← Selected
... (94 more)
```

### After Phase 3.2: 80 features (final_features)
```
rsi_base              ← Best solo performer
rsi_volnorm           ← Still useful
rsi_base_3x_ratio     ← CT ratio survived
macd_base             ← Essential momentum
volume_vwap           ← Volume signal
atr_base              ← Volatility signal
... (74 more)
```

### After Phase 3.3: 50 interactions
```
rsi_base_x_macd_base                    ← Traditional interaction
rsi_base_3x_ratio_x_macd_base           ← Hybrid! (CT + base)
rsi_base_3x_ratio_x_macd_6x_ratio       ← Hybrid! (CT + CT)
volume_vwap_div_atr_base                ← Traditional interaction
macd_base_minus_rsi_volnorm             ← Traditional interaction
... (45 more)
```

### Combined Output: 130 features
```
80 final_features + 50 interactions = 130 total features
```

---

## 🎯 Why Both final_features AND interactions?

### final_features (80)
- **Role**: Foundation features
- **Strength**: Individually strong, stable, interpretable
- **Use case**: Direct signals like "RSI is oversold" or "Volume is spiking"

### interactions (20-50)
- **Role**: Context-dependent features  
- **Strength**: Capture non-linear relationships, synergies
- **Use case**: Complex signals like "RSI is oversold AND volume is low" (contrarian opportunity)

### Why combine them?
- **final_features**: Handle simple patterns
- **interactions**: Handle complex patterns
- **Together**: Comprehensive feature set for model training

---

## 📈 Feature Type Breakdown (Typical)

From the final 130 features:

```
📊 Feature Classification:
  - Hybrid CT interactions: 15    (e.g., rsi_3x_ratio_x_macd_6x_ratio)
  - Traditional interactions: 35   (e.g., rsi_x_macd)
  - Cross-timeframe ratios: 30     (e.g., rsi_base_3x_ratio)
  - Variant features: 25           (e.g., macd_volnorm)
  - Base features: 25              (e.g., rsi_base, atr)
  Total: 130 features
```

### Hybrid CT Interactions (NEW!)
The most sophisticated features:
```
rsi_base_3x_ratio_x_macd_volnorm_6x_ratio
```
This combines:
- RSI's 3x timeframe ratio (how RSI changed over 3× lookback)
- WITH MACD's volume-normalized 6x timeframe ratio
- Creating a cross-timeframe, multi-variant interaction!

These capture:
- Multi-timeframe dynamics
- Multiple transformations (variant + CT)
- Feature synergies
- **Extremely powerful but also most complex**

---

## 🔑 Key Takeaways

1. **Phase 3 is expensive but critical**
   - Takes 15-23 minutes (~70% of pipeline time)
   - But produces the highest quality features
   - Investment pays off in model performance

2. **Progressive refinement prevents overfitting**
   - 3.1: Fast, aggressive filtering (400 → 100)
   - 3.2: Accurate, careful selection (100 → 80)
   - 3.3: Sophisticated interaction discovery

3. **Multiple scoring methods = robustness**
   - No single metric is perfect
   - Combining 5 methods (MI, Redundancy, LGBM, SHAP, Stability)
   - Equal weights (20% each) prevents bias

4. **RFE is key to interaction quality**
   - One-shot selection misses redundancies
   - Progressive removal with re-scoring finds true value
   - More expensive but much more accurate

5. **final_features + interactions = complete picture**
   - final_features: Solo performers
   - interactions: Duets
   - Together: Comprehensive feature space

---

## 📚 Documentation Reference

For more details:
- **Complete pipeline flow**: See `FEATURE_FLOW_EXPLANATION.md`
- **Detailed Phase 3 explanation**: See `PHASE3_DETAILED_EXPLANATION.md`
- **Classification logic**: See `FINAL_CLASSIFICATION_SUMMARY.md`

---

## ✅ Summary in One Sentence

**Phase 3 uses 3 stages of LGBM models with composite scoring to select 80 best individual features, then uses tree-based analysis with 5-way RFE scoring to discover 20-50 synergistic interaction features.**
