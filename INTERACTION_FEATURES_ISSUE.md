# Interaction Features Issue - Root Cause Found

## Date: 2025-11-11 21:26

## Problem

Interaction features (160 columns) are being **loaded** but **not making it** to the feature selection component.

## Evidence

### 1. Interaction Features ARE Loaded ✅
```
✅ Loaded versioned artifact: analyst_interaction_features (14023 rows × 160 cols)
```

### 2. But NOT in Feature Selection Pool ❌
```
⚠️ No interaction features found in feature pool (checked for 'interaction' or '_x_' in names)
```

### 3. Feature Selection Only Sees 294 Features
The selection component is working with 294 features total, which suggests the 160 interaction features are being filtered out somewhere in `_combine_features()`.

## Root Cause

The interaction features are loaded into `features_data['analyst_interactions']` but are being filtered out during the `_combine_features()` process. Possible reasons:

1. **Column name mismatch**: Interaction features might not have 'interaction' or '_x_' in their names after combination
2. **Duplicate column filtering**: They might be getting removed as duplicates
3. **Index alignment issue**: They might be dropped during index alignment
4. **Numeric column filtering**: They might not pass the numeric column filter

## Next Steps

1. Add logging in `_combine_features()` to show:
   - How many interaction features are in the input
   - How many make it through each filtering step
   - Sample of interaction feature names at each step

2. Check if interaction feature names are being modified during combination

3. Verify the interaction features are actually in the combined dataframe before it goes to selection

## Expected Behavior

The combined feature matrix should have:
- ~294 base features
- ~160 interaction features  
- **Total: ~454 features** for selection

Currently only seeing 294 features, confirming the 160 interaction features are being lost.
