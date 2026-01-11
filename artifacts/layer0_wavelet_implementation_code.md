# Layer 0 Wavelet-First Implementation Code

## 1. label_based_layer_0.py Changes

### Function Docstring Update (Line ~228)
```python
def run_layer0_wavelet_kalman(
    symbol: str,
    timeframe: str,
    market_data: pd.DataFrame,
    config: Dict[str, Any],
    outcomes_dir: Path,
    bundle_path: Optional[Path] = None,
    run_optimization: bool = True,
    train_data: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Wavelet Denoising + Kalman Enhancement for Layer 0.
    
    Primary method: Wavelet denoising for noise reduction and signal cleaning
    Secondary method: Kalman filtering for additional smoothing and volatility estimation
    
    Args:
        symbol: Trading symbol
        timeframe: Data timeframe
        market_data: OHLCV market data
        config: Configuration dictionary
        outcomes_dir: Output directory for reports
        bundle_path: Path to cached optimization results
        run_optimization: Whether to run HPO optimization
        train_data: Training data for optimization
        
    Returns:
        Tuple of (enhanced_market_data, optimization_payload)
    """
```

### Logging Update (Line ~248)
```python
# OLD:
tprint_info("🌊 Running Wavelet Denoising (Soft Threshold, Median, Clip)...")

# NEW:
tprint_info("🌊 Wavelet Denoising (Primary) + Kalman Enhancement...")
```

### Report Section Reordering (Lines ~498-549)
```python
# Add Wavelet section first:
lines.append("\n## Wavelet Denoising Results\n")
lines.append(f"- wavelet_enabled: {bool(config.get('use_wavelets', True))}\n")
lines.append(f"- wavelet_available: {WAVELET_AVAILABLE}\n")
if 'wavelet_close' in market_data.columns:
    wavelet_snr = best_diagnostics.get('wavelet_snr_improvement', 'N/A')
    wavelet_noise_red = best_diagnostics.get('wavelet_noise_reduction', 'N/A')
    lines.append(f"- wavelet_snr_improvement: {wavelet_snr}\n")
    lines.append(f"- wavelet_noise_reduction: {wavelet_noise_red}\n")

# Then Kalman section:
lines.append("\n## Kalman Filter Enhancement\n")
lines.append(f"- kalman_Q: {float(Q_best)}\n")
lines.append(f"- kalman_R: {float(R_best)}\n")
# ... rest of Kalman metrics
```

## 2. meta_labeling_hpo_experiment_step.py Changes

### Layer 0 Description Update (Line ~124)
```python
# OLD:
INFO: 🔹 Running Layer 0: Kalman Filter & VWAP...

# NEW:
INFO: 🔹 Running Layer 0: Wavelet Denoising + Kalman Enhancement...
```

## 3. Signal Priority Logic Implementation

### Unified Price Computation
```python
def get_unified_price(market_data: pd.DataFrame) -> pd.Series:
    """
    Get unified price with Wavelet-first priority.
    
    Priority order:
    1. wavelet_close (primary - Wavelet denoised)
    2. kalman_price (secondary - Kalman smoothed)  
    3. close (fallback - raw price)
    """
    if 'wavelet_close' in market_data.columns:
        return market_data['wavelet_close']
    elif 'kalman_price' in market_data.columns:
        return market_data['kalman_price']
    else:
        return market_data['close']
```

## 4. Wavelet Quality Diagnostics Addition

### Enhanced Diagnostics Function
```python
def compute_wavelet_diagnostics(
    raw: np.ndarray,
    wavelet_denoised: np.ndarray,
    sampling_rate: float = 1.0,
) -> Dict[str, float]:
    """
    Compute comprehensive Wavelet denoising diagnostics.
    
    Args:
        raw: Original raw signal
        wavelet_denoised: Wavelet-denoised signal
        sampling_rate: Sampling rate for frequency analysis
        
    Returns:
        Dictionary of Wavelet-specific diagnostic metrics
    """
    diagnostics = {}
    
    try:
        # Basic signal quality
        raw_var = np.var(raw)
        wavelet_var = np.var(wavelet_denoised)
        noise_var = np.var(raw - wavelet_denoised)
        
        # Wavelet-specific metrics
        if noise_var > 1e-12:
            diagnostics['wavelet_snr_improvement'] = float(wavelet_var / noise_var)
        
        if raw_var > 1e-12:
            diagnostics['wavelet_noise_reduction'] = float(1.0 - (noise_var / raw_var))
        
        # Frequency domain analysis
        try:
            from scipy import signal as scipy_signal
            freqs_raw, psd_raw = scipy_signal.periodogram(raw, fs=sampling_rate)
            freqs_wavelet, psd_wavelet = scipy_signal.periodogram(wavelet_denoised, fs=sampling_rate)
            
            # High-frequency noise reduction
            high_freq_idx = int(0.75 * len(freqs_raw))
            high_freq_power_raw = np.mean(psd_raw[high_freq_idx:])
            high_freq_power_wavelet = np.mean(psd_wavelet[high_freq_idx:])
            
            if high_freq_power_raw > 1e-12:
                diagnostics['wavelet_high_freq_reduction'] = float(
                    1.0 - (high_freq_power_wavelet / high_freq_power_raw)
                )
                
        except ImportError:
            pass
            
    except Exception:
        pass
    
    return diagnostics
```

## 5. Integration Points

### Update Filter Diagnostics Call
```python
# In the objective function, add Wavelet diagnostics:
if 'wavelet_close' in market_data.columns:
    wavelet_diags = compute_wavelet_diagnostics(
        raw_m, 
        market_data['wavelet_close'].to_numpy(dtype=float)[:len(raw_m)],
        sampling_rate=4.0
    )
    all_diagnostics.update(wavelet_diags)
```

## Implementation Order

1. **Phase 1**: Update documentation and logging (low risk)
2. **Phase 2**: Add Wavelet quality diagnostics (medium risk)  
3. **Phase 3**: Implement signal priority logic (higher risk)
4. **Phase 4**: Validate with current pipeline run

## Validation Checklist

- [ ] Layer 0 logs show "Wavelet Denoising (Primary) + Kalman Enhancement"
- [ ] Wavelet metrics appear first in Layer 0 reports
- [ ] No errors in current meta_labeling_hpo_sample_weighted run
- [ ] Wavelet_close available in downstream layers
- [ ] Performance metrics comparable or improved
