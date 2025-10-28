# Feature Generation Pipeline - Complete Flow Explanation

## 🔄 Pipeline Overview

The feature generation pipeline transforms features through 5 phases:

```
Phase 0: Load & Select → Phase 1: Variants & CT → Phase 2: Pruning → Phase 3: LGBM+SHAP → Phase 4: Combine & Save
```

## 📊 Detailed Phase Breakdown

### **Phase 0: Load Artifacts and Select Top Features**

**Input**: Feature bank (lookback_optimization)
**Output**: `top_features_by_category` (Dict)

- Loads artifact from feature bank
- Selects top N features per category (e.g., top 4)
- Uses either MI (Analyst) or CMI (Tactician) scoring

**Example Output**:
```python
top_features_by_category = {
    'volatility': ['atr', 'bollinger_width', ...],
    'momentum': ['rsi', 'macd', ...],
    'volume': ['volume_sma', 'obv', ...],
    ...
}
```

---

### **Phase 1: Generate Variants + Cross-Timeframe Features**

**Input**: `top_features_by_category` (selected features)
**Output**: `variant_features` (DataFrame with ~5x expansion)

#### Step 1.1: Generate Variants
For each selected feature, generate 4 versions:

1. **Base version** (`_base`): Original feature with robust scaling
   - Example: `rsi_base`
   
2. **Volume-normalized** (`_volnorm`): Normalized by volatility
   - Example: `rsi_volnorm`
   - Skipped for volatility category features
   
3. **VWAP-weighted** (`_vwap`): Weighted by volume
   - Example: `rsi_vwap`
   - Skipped for volume category features
   
4. **Trend-adjusted** (`_trend_adj`): Detrended version
   - Example: `rsi_trend_adj`

#### Step 1.2: Generate Cross-Timeframe Ratios
For each variant, create 4 timeframe ratios:

- **3x ratio**: `feature / feature.shift(3 × lookback)`
  - Example: `rsi_base_3x_ratio`
  
- **6x ratio**: `feature / feature.shift(6 × lookback)`
  - Example: `rsi_base_6x_ratio`
  
- **9x ratio**: `feature / feature.shift(9 × lookback)`
  - Example: `rsi_base_9x_ratio`
  
- **27x ratio**: `feature / feature.shift(27 × lookback)`
  - Example: `rsi_base_27x_ratio`

#### Total Expansion
- 1 original feature → 4 variants
- 4 variants → 4 × 5 = 20 features (4 base + 16 cross-timeframe)
- If you start with 40 features → ~800 features after Phase 1

---

### **Phase 2: Cheap Pruning**

**Input**: `variant_features` (all variants + CT features)
**Output**: `pruned_features` (40-50% reduction)

- Uses fast statistical pruning
- Removes low-variance, highly correlated, or unstable features
- Protects cross-timeframe features (guaranteed minimum per category)
- Reduces from ~800 → ~400-480 features

**Pruning Criteria**:
- Variance threshold
- Correlation threshold
- Stability score
- Category-based protection

---

### **Phase 3: LGBM+SHAP Pipeline**

**Input**: `pruned_features` (400-480 features)
**Output**: `final_features` + `interactions` + `shap_metadata`

This is the **most important phase** where the magic happens!

#### Phase 3.1: Shallow LGBM Sweep
- Trains shallow LGBM models (max_depth=3)
- Ranks features by SHAP importance
- Selects **top 100 features**
- Fast first pass to remove noise

#### Phase 3.2: Deep LGBM Refinement
- Trains deeper LGBM models (max_depth=5)
- More accurate importance scores
- Selects **top 80 features** → **`final_features`**

#### Phase 3.3: Interaction Discovery
- Analyzes feature interactions using SHAP interaction values
- Discovers synergistic feature pairs
- Generates interaction features:
  - Multiplication: `feature1_x_feature2`
  - Division: `feature1_div_feature2`
  - Subtraction: `feature1_minus_feature2`
  - Log ratio: `feature1_log_feature2`
  - Addition: `feature1_plus_feature2`
- Output: **`interactions`** DataFrame

**Key Point**: 
- `final_features` = 80 best features (base, variants, CT ratios)
- `interactions` = Newly discovered interaction features

---

### **Phase 4: Combine and Save**

**Input**: `final_features` + `interactions`
**Output**: `combined_features` (saved as artifact)

```python
# Line 3568 in the code:
combined_features = pd.concat([final_features, interactions], axis=1)
```

This is where the **final feature classification happens**!

#### What's in `combined_features`?

The combined features contain ALL of these:

1. **Base features with `_base` suffix**
   - Example: `rsi_base`, `macd_base`
   - Original features that survived selection
   
2. **Variant features** (`_volnorm`, `_vwap`, `_trend_adj`)
   - Example: `rsi_volnorm`, `volume_sma_trend_adj`
   - Transformed versions that survived
   
3. **Cross-timeframe ratios** (CT only)
   - Example: `rsi_base_3x_ratio`, `macd_volnorm_6x_ratio`
   - Pure ratio features without interactions
   
4. **Traditional interactions** (no CT)
   - Example: `rsi_x_macd`, `volume_div_price`
   - Interactions between features (same timeframe)
   
5. **Hybrid CT interactions** (interaction + CT)
   - Example: `rsi_base_3x_ratio_x_macd_6x_ratio`
   - Interactions between cross-timeframe features
   - Most complex features

---

## 🎯 Classification Logic (Lines 3611-3650)

### Priority Order

The classification checks features in this **specific order**:

```python
for col in combined_features.columns:
    # 1. Check for interaction operators FIRST
    if any(op in col for op in ['_x_', '_div_', '_minus_', '_log_', '_plus_']):
        if any(ct in col for ct in ['_3x_ratio', '_6x_ratio', '_9x_ratio', '_27x_ratio']):
            → Hybrid CT interaction
        else:
            → Traditional interaction
    
    # 2. Check for CT markers (if no interaction)
    elif any(ct in col for ct in ['_3x_ratio', '_6x_ratio', '_9x_ratio', '_27x_ratio']):
        → Cross-timeframe ratio
    
    # 3. Check for variant suffixes (if no interaction/CT)
    else:
        if col.endswith('_volnorm') or col.endswith('_vwap') or col.endswith('_trend_adj'):
            → Variant feature
        else:
            → Base feature (includes _base suffix!)
```

### Why This Order Matters

**Example 1**: `rsi_base_3x_ratio_x_macd_6x_ratio`
- Contains `_x_` → check CT markers
- Contains `_3x_ratio` → **Hybrid CT interaction** ✅
- Not classified as CT ratio (even though it has `_3x_ratio`)

**Example 2**: `rsi_base_3x_ratio`
- No interaction operators → check CT markers
- Contains `_3x_ratio` → **Cross-timeframe ratio** ✅
- Not classified as base (even though it has `_base`)

**Example 3**: `rsi_base`
- No interaction operators → check CT markers
- No CT markers → check variant suffixes
- No variant suffixes → **Base feature** ✅

**Example 4**: `rsi_volnorm`
- No interaction operators → check CT markers
- No CT markers → check variant suffixes
- Ends with `_volnorm` → **Variant feature** ✅

---

## 📈 Typical Feature Counts

For a typical run with 40 initial features:

| Phase | Feature Count | Notes |
|-------|--------------|-------|
| Phase 0 | 40 | Top features selected |
| Phase 1 | 800 | 40 × 4 variants × 5 (base + 4 CT) |
| Phase 2 | 400-480 | 40-50% pruning |
| Phase 3 final | 80 | Top 80 features |
| Phase 3 interactions | 20-50 | Discovered interactions |
| Phase 4 combined | **100-130** | Final output |

### Final Breakdown Example:
```
📊 Feature breakdown before save:
  - Hybrid CT interactions: 15
  - Traditional interactions: 20
  - Cross-timeframe ratios: 30
  - Variant features: 25
  - Base features: 40
  Total: 130 features
```

---

## 🔑 Key Insights

1. **`_base` is NOT a variant** - it's the base/original feature
2. **Variants** are transformations: `_volnorm`, `_vwap`, `_trend_adj`
3. **Final features** = Selected best features from Phase 3.2
4. **Interactions** = Newly discovered from Phase 3.3
5. **Combined features** = Final + Interactions (what gets saved)
6. **Classification happens AFTER combination** (Phase 4)

---

## 💾 What Gets Saved

The pipeline saves `combined_features` as the artifact, which contains:
- All 5 feature types properly labeled
- Enhanced metadata with category coverage
- Performance statistics
- SHAP importance scores

This artifact is then used by downstream training steps!
