# Phase 3: LGBM+SHAP Pipeline - Detailed Explanation

## 🎯 Overview

Phase 3 is the **most important phase** where feature selection and interaction discovery happen. It takes ~400-480 pruned features and produces:
- **80 best features** (`final_features`)
- **80 interaction features** (`interactions`)

The phase uses a **3-stage progressive refinement** approach with machine learning models.

---

## 📊 The Three Sub-Phases

```
Phase 3: LGBM+SHAP Pipeline
├── Phase 3.1: Shallow LGBM Sweep (400 → 120 features)
├── Phase 3.2: Deeper LGBM Refinement (100 → 80 features)  
└── Phase 3.3: Deep Interaction Discovery (80 features → 20-80 interactions)
```

---

## 🔍 Phase 3.1: Shallow LGBM Sweep

### Goal
Rapidly filter from ~400-480 pruned features down to **top 120 features**

### Method: Fast Proxy Selection

**1. Train Shallow LGBM Model**
```python
lgbm_params = {
    'max_depth': 3,              # Shallow trees
    'num_leaves': 10,            # Limited complexity
    'n_estimators': 80,          # Moderate number of trees
    'learning_rate': 0.05,       # Conservative learning
    'reg_alpha': 0.2,            # L1 regularization
    'reg_lambda': 0.2,           # L2 regularization
    'min_child_samples': 80,     # Prevent overfitting
    'subsample': 0.6,            # Row sampling
    'colsample_bytree': 0.6      # Column sampling
}
```

**2. Extract Two Scores**

a) **LGBM Feature Importance** (60% weight)
   - Direct from LGBM: `model.feature_importances_`
   - Measures how often and how effectively each feature is used in splits
   - Fast to compute, no SHAP needed

b) **Mutual Information (MI)** (40% weight)
   - Fast proxy using binning (5 bins)
   - Measures statistical dependency between feature and target
   - Computation: `_fast_mi_proxy(feature, target, n_bins=5)`

**3. Combine Scores**
```python
# Normalize both scores to [0, 1]
importance_normalized = (importance - min) / (max - min)
mi_normalized = (mi - min) / (max - min)

# Weighted combination
combined_score = 0.6 × importance_normalized + 0.4 × mi_normalized
```

**4. Select Top 120**
```python
top_100_features = features[top_100_indices]
```

### Why This Works
- **Fast**: No expensive SHAP calculations
- **Effective**: Combines model-based and statistical approaches
- **Balanced**: LGBM captures non-linear patterns, MI captures linear relationships
- **Aggressive**: Removes 75% of features (400 → 100) quickly

### Output
- **120 features** with highest combined scores
- Typically includes mix of base, variants, CT ratios

---

## 🎯 Phase 3.2: Deeper LGBM Refinement

### Goal
Refine from 120 features down to **top 80 features** with more accurate scoring

### Method: Multi-Criteria Selection

**1. Train Deeper LGBM Model**
```python
lgbm_params = {
    'max_depth': 4,              # Slightly deeper (was 3)
    'num_leaves': 15,            # More flexibility (was 10)
    'n_estimators': 100,         # More trees (was 80)
    'learning_rate': 0.05,       # Same conservative rate
    'reg_alpha': 0.2,            # L1 regularization
    'reg_lambda': 0.2,           # L2 regularization
    'min_child_samples': 80,     # Prevent overfitting
    'subsample': 0.6,            # Row sampling
    'colsample_bytree': 0.6      # Column sampling
}
```

**2. Calculate Three Scores**

a) **LGBM Feature Importance** (60% weight)
   - More accurate with deeper trees
   - Captures more complex interactions

b) **Mutual Information** (30% weight)
   - Same fast proxy method
   - Lower weight as we trust LGBM more now

c) **Stability Score** (10% weight)
   - Measures consistency across different data samples
   - Computed via `_calculate_importance_consistency()`
   - Helps prevent overfitting by favoring stable features

**3. Combine Scores**
```python
combined_score = 0.6 × importance + 0.3 × mi + 0.1 × stability
```

**4. Select Top 80**
```python
final_features = features[top_80_indices]
```

### Why This Works
- **More Accurate**: Deeper trees capture more patterns
- **Stability Check**: Prevents selecting noisy/overfitted features
- **Smaller Reduction**: Only 20% removed (100 → 80), more careful selection

### Output
- **80 best features** → This becomes `final_features`
- These are the "winners" from all previous phases

---

## 🌳 Phase 3.3: Deep Interaction Discovery

### Goal
Discover synergistic feature pairs and generate **80 interaction features**

This is the **most complex phase** with multiple steps:

---

### Step 1: Extract Cross-Timeframe Interactions

**What**: Identify existing cross-timeframe features that can be combined
```python
cross_timeframe_interactions = _extract_cross_timeframe_interactions(features)
# Merges them back into features for analysis
```

Example: If you have `rsi_base_3x_ratio` and `macd_6x_ratio`, these might already interact.

---

### Step 2: Train Deep LGBM for Tree Analysis

**Purpose**: Use decision trees to find which features naturally split together

```python
lgbm_params = {
    'max_depth': 3,              # Shallow to prevent overfitting
    'num_leaves': 10,            # Conservative
    'n_estimators': 80,          # Moderate trees
    'reg_alpha': 0.25,           # Higher regularization
    'reg_lambda': 0.25,          # Higher regularization
    'min_child_samples': 100     # Very conservative
}

model = lgb.LGBMRegressor(**lgbm_params)
model.fit(features, targets)
```

**Why Deep Model?** 
- Despite the name "deep", the parameters are actually conservative
- Focus is on tree structure analysis, not pure prediction
- We want to see which features "work together" in decision paths

---

### Step 3: Extract Feature Pairs from Tree Splits

**Method**: `_extract_tree_splitting_pairs(model)`

**How it works**:
1. Analyzes all trees in the LGBM model
2. Finds features that frequently appear in the same decision path
3. Counts "co-occurrence" - how often f1 and f2 split near each other
4. Returns top 80 feature pairs ranked by co-occurrence

**Example Output**:
```python
feature_pairs = [
    ('rsi_base', 'macd_volnorm', 45),      # co-occurred 45 times
    ('volume_vwap', 'price_3x_ratio', 38), # co-occurred 38 times
    ('atr_base', 'volatility_spike', 32),  # co-occurred 32 times
    ...
]
```

---

### Step 4: Generate Interaction Candidates

For each of the **top 80 feature pairs**, create **5 operations**:

```python
operations = [
    # 1. Multiplication
    f"{f1}_x_{f2}":        f1 × f2
    
    # 2. Division  
    f"{f1}_div_{f2}":      f1 / (f2 + ε)
    
    # 3. Subtraction
    f"{f1}_minus_{f2}":    f1 - f2
    
    # 4. Log ratio
    f"{f1}_log_{f2}":      log(|f1| + ε) / (log(|f2| + ε) + ε)
    
    # 5. Log of ratio
    f"{f1}_log_ratio_{f2}": log(|f1 / (f2 + ε)| + ε)
]
```

**Total Candidates**: 80 pairs × 5 operations = **400 interaction candidates**

---

### Step 5: Composite Scoring with RFE

**The Most Sophisticated Part!**

Uses **CompositeFeatureScorer** with 5 scoring methods:

#### 5-Way Composite Score (Equal 20% weights):

**1. Mutual Information (20%)**
```python
mi_score = mutual_info_regression(interaction, target)
```
- Statistical dependency between interaction and target
- Captures linear + some non-linear relationships

**2. Redundancy Score (20%)**
```python
redundancy = 1 - max_correlation_with_existing_features
```
- Measures how unique the interaction is
- Prevents adding features that duplicate existing info

**3. LGBM Importance (20%)**
```python
lgbm_model.fit(interactions, target)
importance = lgbm_model.feature_importances_
```
- How useful the interaction is in gradient boosting
- Captures non-linear predictive power

**4. SHAP Values (20%)**
```python
shap_explainer = shap.TreeExplainer(lgbm_model)
shap_values = shap_explainer.shap_values(interactions)
importance = mean(|shap_values|)
```
- Model-agnostic feature importance
- More reliable than raw LGBM importance
- Shows true marginal contribution

**5. Stability Score (20%)**
```python
stability = consistency_across_bootstrap_samples
```
- Tests if interaction is consistently important across different data samples
- Prevents selection of noise/overfitted interactions

#### RFE (Recursive Feature Elimination)

**Process**:
```
Round 1: 400 candidates → Score all → Remove bottom 33% → 268 remain
Round 2: 268 candidates → Score all → Remove bottom 33% → 180 remain
Round 3: 180 candidates → Score all → Remove bottom 33% → 121 remain
Round 4: 121 candidates → Score all → Remove bottom 33% → 81 remain
Round 5: 81 candidates  → Score all → Remove bottom 33% → 54 remain
Round 6: 54 candidates  → Score all → Keep top 80     → 80 remain
```

**Why RFE?**
- Scores become more accurate as you remove noise
- Features that seem important alone may be redundant together
- Progressive refinement prevents local optima

---

### Step 6: Apply Overfitting Prevention

**Complexity Filtering**:
```python
max_complexity = 3  # Limit to 3-way interactions
```
- Rejects interactions like `f1_x_f2_x_f3_x_f4` (too complex)
- Keeps simpler interactions that generalize better

**Final Selection**:
```python
top_interactions = sorted_interactions[:50]  # Max 50
```

---

### Step 7: Create Interaction Features

**Build the DataFrame**:
```python
interaction_features = {}
for name, score in top_interactions:
    # Re-compute the interaction
    interaction_features[name] = computed_interaction

interaction_df = pd.DataFrame(interaction_features)
```

**Apply Causality Shift**:
```python
interaction_df = interaction_df.shift(1)
```
- Critical for time-series: Prevents look-ahead bias
- Interaction at time `t` uses data from time `t-1`

**Apply RobustScaler**:
```python
scaler = RobustScaler()
interaction_df = scaler.fit_transform(interaction_df)
```
- Normalizes interactions to similar scale
- Uses robust method (median/IQR) to handle outliers

---

### Output
- **80 interaction features** → This becomes `interactions`
- Each interaction has passed rigorous multi-criteria evaluation
- Names clearly show operation: `rsi_base_x_macd_volnorm`, `volume_div_price`, etc.

---

## 📈 Complete Phase 3 Flow Diagram

```
Phase 3 Input: 400-480 pruned features
│
├─ Phase 3.1: Shallow LGBM Sweep ──────────────────────┐
│  │                                                     │
│  ├─ Train shallow LGBM (max_depth=3)                 │
│  ├─ Extract feature importance                       │
│  ├─ Calculate MI (fast proxy)                        │
│  ├─ Combine: 60% importance + 40% MI                 │
│  └─ Select top 120 features                          │
│                                                        │
│  Output: 120 features ────────────────────────────────┤
│                                                        │
├─ Phase 3.2: Deeper LGBM Refinement ──────────────────┤
│  │                                                     │
│  ├─ Train deeper LGBM (max_depth=4)                  │
│  ├─ Extract feature importance                       │
│  ├─ Calculate MI                                      │
│  ├─ Calculate stability score                        │
│  ├─ Combine: 60% importance + 30% MI + 10% stability │
│  └─ Select top 80 features                           │
│                                                        │
│  Output: 80 features (final_features) ────────────────┤
│                                                        │
├─ Phase 3.3: Deep Interaction Discovery ──────────────┤
│  │                                                     │
│  ├─ Extract cross-timeframe interactions             │
│  ├─ Train deep LGBM (max_depth=3, conservative)     │
│  ├─ Extract 80 feature pairs from tree splits       │
│  ├─ Generate 400 interaction candidates (80×5)       │
│  │  ├─ Multiplication (_x_)                          │
│  │  ├─ Division (_div_)                              │
│  │  ├─ Subtraction (_minus_)                         │
│  │  ├─ Log ratio (_log_)                             │
│  │  └─ Log of ratio (_log_ratio_)                    │
│  │                                                     │
│  ├─ Composite Scoring with RFE:                      │
│  │  ├─ Round 1: 400 → 268 (remove 33%)              │
│  │  ├─ Round 2: 268 → 180 (remove 33%)              │
│  │  ├─ Round 3: 180 → 121 (remove 33%)              │
│  │  ├─ Round 4: 121 → 81 (remove 33%)               │
│  │  ├─ Round 5: 81 → 54 (remove 33%)                │
│  │  └─ Round 6: 54 → 50 (keep top)                  │
│  │                                                     │
│  │  Scoring (20% each):                              │
│  │  ├─ Mutual Information                            │
│  │  ├─ Redundancy                                     │
│  │  ├─ LGBM Importance                               │
│  │  ├─ SHAP Values                                    │
│  │  └─ Stability                                      │
│  │                                                     │
│  ├─ Filter by complexity (max 3-way)                 │
│  ├─ Apply causality shift (t-1)                      │
│  └─ Apply RobustScaler normalization                 │
│                                                        │
│  Output: 20-80 interactions ──────────────────────────┤
│                                                        │
└─ Phase 3 Output: ────────────────────────────────────┘
   ├─ final_features: 80 best features
   └─ interactions: 80 interaction features
```

---

## 🎯 Key Design Decisions

### Why Progressive Refinement?
- **Phase 3.1**: Fast filtering removes obvious noise
- **Phase 3.2**: More accurate scoring on quality candidates  
- **Phase 3.3**: Interaction discovery on proven features

### Why Not Just Use SHAP Everywhere?
- **Speed**: SHAP is expensive (10-100x slower than alternatives)
- **Phase 3.1**: Fast proxy gets 80% accuracy at 1% cost
- **Phase 3.2**: Still uses proxy but more accurate models
- **Phase 3.3**: SHAP used here as 20% of composite score where it matters most

### Why 5 Different Scoring Methods?
Each captures something unique:
1. **MI**: Statistical dependency
2. **Redundancy**: Uniqueness
3. **LGBM**: Tree-based importance
4. **SHAP**: True marginal contribution
5. **Stability**: Robustness across samples

No single metric is perfect. Equal weighting (20% each) prevents bias.

### Why RFE Instead of One-Shot Selection?
- Features interact with each other
- A feature may seem important with 400 candidates but redundant with 50
- RFE re-evaluates at each stage
- More expensive but much more accurate

---

## 💡 Real Example

**Starting Point**: 480 pruned features

**After Phase 3.1** (120 features):
```
- rsi_base
- rsi_volnorm  
- rsi_vwap
- rsi_base_3x_ratio
- macd_base
- macd_volnorm
- volume_vwap
- atr_base
... (92 more)
```

**After Phase 3.2** (80 features = final_features):
```
- rsi_base
- rsi_volnorm
- rsi_base_3x_ratio
- macd_base
- macd_volnorm_6x_ratio
- volume_vwap
- atr_base
... (73 more)
```

**After Phase 3.3** (80 interactions):
```
- rsi_base_x_macd_volnorm
- rsi_base_3x_ratio_x_volume_vwap       ← Hybrid CT interaction!
- volume_vwap_div_atr_base
- macd_base_minus_rsi_volnorm
- rsi_base_log_volume_vwap
... (45 more)
```

**Final Phase 3 Output**: 80 + 50 = **160 features total**

---

## 🔑 Critical Insights

1. **final_features** are the 80 "solo performers"
   - Best individual features
   - Can include base, variants, and CT ratios

2. **interactions** are the "duets"  
   - Synergistic combinations
   - Often more predictive than individual features
   - Can create hybrid CT interactions (combining CT features)

3. **Why both matter**:
   - final_features: Stable, interpretable, individual signals
   - interactions: Capture non-linear relationships, context-dependent patterns

4. **Phase 3 is expensive but critical**:
   - Takes 60-80% of total pipeline time
   - But produces the highest-quality feature set
   - Investment pays off in model performance

---

## 📊 Performance Stats

Typical Phase 3 timing (for 480 input features):
- Phase 3.1: 2-3 minutes
- Phase 3.2: 3-5 minutes  
- Phase 3.3: 10-15 minutes (most expensive due to RFE)
- **Total: ~15-23 minutes**

This is why earlier phases do aggressive pruning!
