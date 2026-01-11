# Documentation Update Code Implementation

## Exact Code Changes Required

### 1. label_based_layer_0.py Updates

#### Function Name and Docstring (Line ~228)
```python
# CURRENT CODE:
def run_layer0_kalman_vwap(
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
    Run Layer 0: Kalman Filter & VWAP optimization.

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

# REPLACEMENT CODE:
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
    
    This implements a Wavelet-first approach where:
    1. Wavelet denoising removes high-frequency noise and outliers
    2. Kalman filtering provides additional smoothing and volatility estimation
    3. Both signals are available for downstream layers

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

#### Logging Message Update (Line ~248)
```python
# CURRENT CODE:
            tprint_info("🌊 Running Wavelet Denoising (Soft Threshold, Median, Clip)...")

# REPLACEMENT CODE:
            tprint_info("🌊 Wavelet Denoising (Primary) + Kalman Enhancement...")
```

#### Report Section Updates (Lines ~498-550)
```python
# CURRENT CODE:
        lines = [
            "# Layer0 Report\n",
            f"- timestamp: {ts}\n",
            f"- symbol: {symbol}\n",
            f"- timeframe: {timeframe}\n",
            f"- run_optimization: {bool(run_optimization)}\n",
            f"- bundle_path: {str(bundle_path)}\n",
            f"- loaded_from: {str(loaded_from) if loaded_from else ''}\n",
            f"- n_bars: {int(len(market_data))}\n",
            f"- date_range: {start_ts} -> {end_ts}\n",
            "\n## Best Params\n",
            f"- kalman_Q: {float(Q_best)}\n",
            f"- kalman_R: {float(R_best)}\n",
            f"- volume_weight: {float(volume_weight)}\n",
            f"- volume_adaptive: {bool(volume_adaptive)}\n",
            "\n## Loss Components\n",
            # ... rest of current structure
        ]

# REPLACEMENT CODE:
        lines = [
            "# Layer0 Report\n",
            f"- timestamp: {ts}\n",
            f"- symbol: {symbol}\n",
            f"- timeframe: {timeframe}\n",
            f"- run_optimization: {bool(run_optimization)}\n",
            f"- bundle_path: {str(bundle_path)}\n",
            f"- loaded_from: {str(loaded_from) if loaded_from else ''}\n",
            f"- n_bars: {int(len(market_data))}\n",
            f"- date_range: {start_ts} -> {end_ts}\n",
            "\n## Wavelet Denoising Results (Primary)\n",
            f"- wavelet_enabled: {bool(config.get('use_wavelets', True))}\n",
            f"- wavelet_available: {WAVELET_AVAILABLE}\n",
        ]
        
        # Add Wavelet metrics if available
        if 'wavelet_close' in market_data.columns:
            lines.append(f"- wavelet_processed: True\n")
            lines.append(f"- wavelet_noise_removed: {len(market_data) - market_data['wavelet_noise'].isna().sum()}\n")
            if best_diagnostics:
                wavelet_snr = best_diagnostics.get('wavelet_snr_improvement', 'N/A')
                wavelet_noise_red = best_diagnostics.get('wavelet_noise_reduction', 'N/A')
                lines.append(f"- wavelet_snr_improvement: {wavelet_snr}\n")
                lines.append(f"- wavelet_noise_reduction: {wavelet_noise_red}\n")
        else:
            lines.append(f"- wavelet_processed: False\n")
        
        lines.extend([
            "\n## Kalman Filter Enhancement (Secondary)\n",
            f"- kalman_Q: {float(Q_best)}\n",
            f"- kalman_R: {float(R_best)}\n",
            f"- volume_weight: {float(volume_weight)}\n",
            f"- volume_adaptive: {bool(volume_adaptive)}\n",
            "\n## Loss Components\n",
            # ... rest of existing structure
        ])
```

### 2. meta_labeling_hpo_experiment_step.py Updates

#### Layer 0 Description (Line ~124)
```python
# CURRENT CODE:
        tprint_info("🔹 Running Layer 0: Kalman Filter & VWAP...")

# REPLACEMENT CODE:
        tprint_info("🔹 Running Layer 0: Wavelet Denoising + Kalman Enhancement...")
```

#### Pipeline Documentation (Lines ~7-16)
```python
# CURRENT CODE:
"""
This step acts as an Orchestrator for the full Label-Based Pipeline (Layers 0-5),
integrating the proper De Prado Causal Framework (Layer 2) and subsequent
Meta-Labeling (Layer 3) and Position Sizing (Layers 4-5) stages.

It replaces the legacy inline HPO logic with a sequential execution of:
1.  **Layer 0**: Kalman Filter & VWAP Price Smoothing (Feature Engineering).
2.  **Layer 1**: Sample Weighting Optimization.
3.  **Layer 2**: Causal Event Generation & Triple Barrier Labeling (Primary Model).
4.  **Layer 3**: Multi-Geometry Meta-Model Training (Analyst).
5.  **Layer 4**: ExtraTrees Position Sizing (PnL Optimization).
6.  **Layer 5**: Portfolio Construction & Backtesting.
"""

# REPLACEMENT CODE:
"""
This step acts as an Orchestrator for the full Label-Based Pipeline (Layers 0-5),
integrating the proper De Prado Causal Framework (Layer 2) and subsequent
Meta-Labeling (Layer 3) and Position Sizing (Layers 4-5) stages.

It replaces the legacy inline HPO logic with a sequential execution of:
1.  **Layer 0**: Wavelet Denoising + Kalman Enhancement (Feature Engineering).
    - Primary: Wavelet denoising for noise removal and signal cleaning
    - Secondary: Kalman filtering for additional smoothing and volatility estimation
2.  **Layer 1**: Sample Weighting Optimization.
3.  **Layer 2**: Causal Event Generation & Triple Barrier Labeling (Primary Model).
4.  **Layer 3**: Multi-Geometry Meta-Model Training (Analyst).
5.  **Layer 4**: ExtraTrees Position Sizing (PnL Optimization).
6.  **Layer 5**: Portfolio Construction & Backtesting.
"""
```

### 3. Additional Context Updates

#### Comment Updates (Line ~432)
```python
# CURRENT CODE:
    # Apply volume-weighted Kalman filtering to full market_data

# REPLACEMENT CODE:
    # Wavelet denoising already applied as primary signal processing
    # Apply volume-weighted Kalman filtering as secondary enhancement
```

## Implementation Steps

### Step 1: Safe Updates (Can be done during run)
1. Update logging messages
2. Update inline comments
3. Update report section headers

### Step 2: Function Name Update (After run completion)
1. Rename function from `run_layer0_kalman_vwap` to `run_layer0_wavelet_kalman`
2. Update any imports/references to the function name
3. Update docstring

### Step 3: Documentation Updates (Post-run)
1. Update pipeline orchestration documentation
2. Update any README files
3. Update architecture documentation

## Validation

After implementation, verify:
- [ ] Pipeline runs without errors
- [ ] Wavelet metrics appear prominently in reports
- [ ] Logging messages reflect Wavelet-first approach
- [ ] No functional changes to signal processing
- [ ] Both wavelet_close and kalman_price available downstream
