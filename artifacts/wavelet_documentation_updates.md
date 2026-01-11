# Wavelet-First Documentation Updates

Update all documentation to properly reflect that Layer 0 uses Wavelet denoising as the primary signal processing method, with Kalman filtering as secondary enhancement.

## Documentation Files to Update

### 1. label_based_layer_0.py
**Current Issue**: Function described as "Kalman Filter & VWAP optimization"
**Required Change**: Update to emphasize Wavelet-first approach

#### Function Docstring (Line ~228)
```python
# CURRENT:
def run_layer0_kalman_vwap(
    """Run Layer 0: Kalman Filter & VWAP optimization..."""

# UPDATED:
def run_layer0_wavelet_kalman(
    """Wavelet Denoising + Kalman Enhancement for Layer 0.
    
    Primary method: Wavelet denoising for noise reduction and signal cleaning
    Secondary method: Kalman filtering for additional smoothing and volatility estimation
    
    This implements a Wavelet-first approach where:
    1. Wavelet denoising removes high-frequency noise and outliers
    2. Kalman filtering provides additional smoothing and volatility estimation
    3. Both signals are available for downstream layers
    """
```

#### Logging Messages (Line ~248)
```python
# CURRENT:
tprint_info("🌊 Running Wavelet Denoising (Soft Threshold, Median, Clip)...")

# UPDATED:
tprint_info("🌊 Wavelet Denoising (Primary) + Kalman Enhancement...")
```

#### Report Section Headers
```python
# CURRENT:
"## Best Params"
"## Filter Diagnostics"

# UPDATED:
"## Wavelet Denoising Results (Primary)"
"## Kalman Filter Enhancement (Secondary)"
"## Comparative Analysis"
```

### 2. meta_labeling_hpo_experiment_step.py
**Current Issue**: Layer 0 described as "Kalman Filter & VWAP"
**Required Change**: Update orchestration logging

#### Layer 0 Description (Line ~124)
```python
# CURRENT:
INFO: 🔹 Running Layer 0: Kalman Filter & VWAP...

# UPDATED:
INFO: 🔹 Running Layer 0: Wavelet Denoising + Kalman Enhancement...
```

#### Pipeline Documentation
```python
# CURRENT:
"""
1.  **Layer 0**: Kalman Filter & VWAP Price Smoothing (Feature Engineering).
"""

# UPDATED:
"""
1.  **Layer 0**: Wavelet Denoising + Kalman Enhancement (Feature Engineering).
   - Primary: Wavelet denoising for noise removal and signal cleaning
   - Secondary: Kalman filtering for additional smoothing and volatility estimation
"""
```

### 3. README and Pipeline Documentation
**Files to check for updates**:
- Main project README
- Pipeline documentation files
- Architecture documentation

#### Update Examples
```markdown
# CURRENT:
"Layer 0 uses Kalman filtering and VWAP for price smoothing"

# UPDATED:
"Layer 0 uses Wavelet denoising as the primary signal processing method, 
with Kalman filtering as secondary enhancement for additional smoothing"
```

### 4. Comments and Inline Documentation

#### Function Comments
```python
# CURRENT:
# Apply volume-weighted Kalman filtering to full market_data

# UPDATED:
# Wavelet denoising already applied as primary signal processing
# Apply volume-weighted Kalman filtering as secondary enhancement
```

#### Variable Naming Context
```python
# CURRENT:
# Store Kalman results as primary smoothed price

# UPDATED:
# Store Kalman results as secondary enhancement (wavelet_close is primary)
```

## Implementation Priority

### Phase 1: Critical Documentation (Immediate)
1. **label_based_layer_0.py** function docstring
2. **meta_labeling_hpo_experiment_step.py** logging message
3. Report section headers

### Phase 2: Supporting Documentation (Within current run)
1. Inline comments and variable context
2. Pipeline orchestration comments

### Phase 3: External Documentation (Post-run)
1. README files
2. Architecture documentation
3. User guides

## Validation Checklist

- [ ] Layer 0 function name reflects Wavelet-first approach
- [ ] Logging messages emphasize Wavelet as primary method
- [ ] Report sections ordered with Wavelet first
- [ ] Orchestration documentation updated
- [ ] No functional changes to implementation
- [ ] Current pipeline run continues unaffected

## Key Messages to Convey

1. **Wavelet is Primary**: Wavelet denoising is the main signal processing method
2. **Kalman is Enhancement**: Kalman filtering provides additional smoothing
3. **Both Available**: Both signals available for downstream layers
4. **No Functional Change**: Implementation remains the same, only documentation updates

## Success Criteria

- Documentation accurately reflects Wavelet-first implementation
- No confusion about primary vs secondary signal processing methods
- Clear understanding of signal priority in downstream layers
- Consistent messaging across all documentation
