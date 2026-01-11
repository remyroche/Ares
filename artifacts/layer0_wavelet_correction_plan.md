# Layer 0 Wavelet-First Correction Plan

Correct Layer 0 to prioritize Wavelet denoising over Kalman/VWAP as the primary signal processing method.

## Current Issue Analysis

**Problem**: Layer 0 is currently described as "Kalman Filter & VWAP optimization" but should prioritize Wavelet denoising as the primary method.

**Current Implementation Status**:
- ✅ Wavelet denoising IS already implemented (lines 245-293 in label_based_layer_0.py)
- ✅ Wavelet runs FIRST before Kalman optimization 
- ✅ Config flag `use_wavelets` defaults to `True`
- ✅ Wavelet output stored as `wavelet_close` and `wavelet_noise`
- ❌ **Documentation/labeling issue**: Layer 0 described incorrectly as "Kalman Filter & VWAP"

## Root Cause

The issue is primarily in **description and perception**, not implementation:
1. Layer 0 logging describes it as "Kalman Filter & VWAP" 
2. Comments focus on Kalman optimization
3. Wavelet integration exists but is under-emphasized

## Implementation Details

### Phase 1: Documentation & Logging Updates

**Files to modify:**
1. **label_based_layer_0.py**:
   - Update function docstring: "Wavelet Denoising + Kalman Enhancement"
   - Change logging: "🌊 Wavelet Denoising (Primary) + Kalman Enhancement"
   - Add prominent Wavelet metrics in reports

2. **meta_labeling_hpo_experiment_step.py**:
   - Line 124: Change "🔹 Running Layer 0: Kalman Filter & VWAP..."
   - Update to: "🔹 Running Layer 0: Wavelet Denoising + Kalman Enhancement..."

3. **Layer 0 reports**:
   - Reorder sections: Wavelet diagnostics first, Kalman secondary
   - Add Wavelet quality metrics section

### Phase 2: Implementation Refinements

**Signal Priority Logic:**
```python
# In unified price computation:
if 'wavelet_close' in market_data.columns:
    unified_price = market_data['wavelet_close']  # Primary
elif 'kalman_price' in market_data.columns:
    unified_price = market_data['kalman_price']   # Fallback
else:
    unified_price = market_data['close']          # Raw
```

**Wavelet Quality Diagnostics:**
- Add SNR improvement metrics
- Noise reduction percentages
- Frequency domain analysis
- Wavelet vs Kalman comparison

### Phase 3: Validation Steps

1. **Current run monitoring**: Watch meta_labeling_hpo_sample_weighted execution
2. **Performance comparison**: Wavelet-first vs Kalman-first metrics
3. **Downstream validation**: Ensure no regression in Layers 1-5

## Specific Code Changes Required

### 1. label_based_layer_0.py
- Line 228: Update function docstring
- Line 248: Update tprint_info message
- Lines 498-549: Reorder report sections (Wavelet first)
- Add Wavelet quality metrics section

### 2. meta_labeling_hpo_experiment_step.py  
- Line 124: Update Layer 0 description
- Ensure wavelet_close is used as primary signal where applicable

### 3. Report Generation
- Promote Wavelet diagnostics to primary position
- Add Wavelet vs Kalman comparison section
- Include Wavelet quality scores

## Success Criteria

- Layer 0 clearly described as "Wavelet Denoising + Kalman Enhancement"
- Wavelet metrics prominently displayed in reports
- Wavelet output used as primary signal where appropriate
- No functional regression in pipeline performance
