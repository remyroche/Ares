"""
Optimized Wavelet Decomposition for Adaptive Event-Driven Labeling (AEDL)

High-performance implementation with:
- Vectorized multi-scale decomposition
- In-place array operations
- Parallel processing support
- Memory-efficient operations
- Strictly Causal Mode (EMA Cascade) to prevent data leakage

Key Features:
- O(N) multi-scale decomposition
- Causal decomposition using EMA cascade (approximating dyadic filter bank)
- Parallel specialist processing
- GPU-ready architecture
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy import signal
from scipy.stats import pearsonr
import pywt
from concurrent.futures import ThreadPoolExecutor
import warnings

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    # Fallback print functions
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")

from src.utils.numba_funcs import _numba_ewma

class OptimizedWaveletDecomposition:
    """
    High-performance wavelet decomposition engine.
    
    Optimizations:
    1. Vectorized multi-scale decomposition
    2. Causal decomposition mode (EMA Cascade) using Numba
    3. Parallel processing
    4. Memory-efficient operations
    """
    
    def __init__(
        self,
        wavelet: str = 'db4',
        scales: List[str] = ['d1', 'd2', 'd3', 'd4', 's4'],
        max_level: int = 4,
        enable_parallel: bool = True,
        causal: bool = True, # Enforce causality by default
        verbose: bool = True
    ):
        """
        Initialize Optimized Wavelet Decomposition Engine.
        
        Args:
            wavelet: Wavelet family for decomposition (used in non-causal mode)
            scales: List of scale names
            max_level: Maximum decomposition level
            enable_parallel: Enable parallel processing
            causal: If True, uses Causal EMA Cascade instead of Wavelets to prevent leakage.
            verbose: Whether to print progress information
        """
        self.verbose = verbose
        self.wavelet = wavelet
        self.scales = scales
        self.max_level = max_level
        self.enable_parallel = enable_parallel
        self.causal = causal
        self._modwt_available = hasattr(pywt, "modwt")
        
        # Scale definitions optimized for 2-4h trades
        self.scale_definitions = {
            'd1': {'name': 'Micro-Shock', 'timeframe': '5m-15m', 'description': 'High-frequency order flow and HFT noise'},
            'd2': {'name': 'Dealer Flow', 'timeframe': '15m-1h', 'description': 'Core inventory cycle for trade duration'},
            'd3': {'name': 'Session Trend', 'timeframe': '1h-4h', 'description': 'Parent move of specific trade'},
            'd4': {'name': 'Causal Baseline', 'timeframe': '4h-12h', 'description': 'Structural gravity of the day'},
            's4': {'name': 'Macro Ground', 'timeframe': '12h+', 'description': 'Low-frequency regime (Anchored state)'}
        }
        
        # Pre-allocate scale order for efficiency
        self.scale_order = ['d1', 'd2', 'd3', 'd4', 's4']
        
        if self.verbose:
            tprint_info("🚀 Optimized Wavelet Decomposition: Initializing...")
            tprint_info(f"   ⚙️ Wavelet family: {wavelet}")
            tprint_info(f"   ⚙️ Decomposition levels: {max_level}")
            tprint_info(f"   ⚙️ Scales: {', '.join(scales)}")
            tprint_info(f"   ⚙️ Parallel processing: {enable_parallel}")
            tprint_info(f"   ⚙️ Causal Mode: {self.causal}")
            if not self.causal and not self._modwt_available:
                tprint_warning("   ⚠️ pywt.modwt unavailable; using SWT fallback by default")
            tprint_success("   ✅ Optimized Wavelet Decomposition: Initialization complete")
    
    def _clean_signal_inplace(self, signal: np.ndarray) -> np.ndarray:
        """
        Clean signal in-place for efficiency.
        
        Args:
            signal: Input signal to clean
            
        Returns:
            Cleaned signal
        """
        # In-place forward fill
        mask = ~np.isnan(signal)
        if not np.any(mask):
            return np.zeros_like(signal)
        
        # Find first valid index
        first_valid = np.argmax(mask)
        if first_valid > 0:
            signal[:first_valid] = 0.0
        
        # In-place forward fill
        # Using pandas ffill logic is faster than python loop for large arrays
        # if signal is large, convert to series
        if len(signal) > 1000:
            # We can use pd.Series(signal).ffill().values but it copies
            # Use manual loop only for NaNs
            nans = np.isnan(signal)
            if np.any(nans):
                # Basic forward fill in numpy
                idx = np.where(~nans, np.arange(len(signal)), 0)
                np.maximum.accumulate(idx, out=idx)
                signal[:] = signal[idx]
        else:
            # Small loop
            for i in range(first_valid + 1, len(signal)):
                if np.isnan(signal[i]):
                    signal[i] = signal[i-1]
        
        return signal
    
    def _decompose_causal_ema(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform strictly causal decomposition using a cascade of EMAs.
        Approximates dyadic wavelet scales without lookahead bias.
        Optimized using Numba EWMA.

        Scales:
        - d1: signal - EMA(2)
        - d2: EMA(2) - EMA(4)
        - d3: EMA(4) - EMA(8)
        - d4: EMA(8) - EMA(16)
        - s4: EMA(16)
        """
        decomposition = {}

        # Calculate alphas for different spans
        # span -> alpha = 2 / (span + 1)
        # Numba function takes alpha directly

        alpha2 = 2.0 / (2.0 + 1.0)
        alpha4 = 2.0 / (4.0 + 1.0)
        alpha8 = 2.0 / (8.0 + 1.0)
        alpha16 = 2.0 / (16.0 + 1.0)

        signal_float = signal.astype(np.float64)

        ema2 = _numba_ewma(signal_float, alpha2, adjust=False)
        ema4 = _numba_ewma(signal_float, alpha4, adjust=False)
        ema8 = _numba_ewma(signal_float, alpha8, adjust=False)
        ema16 = _numba_ewma(signal_float, alpha16, adjust=False)

        # Extract scales
        # d1: Highest frequency (Noise/Micro-shock)
        if 'd1' in self.scales:
            decomposition['d1'] = signal_float - ema2

        # d2: High-Medium frequency
        if 'd2' in self.scales:
            decomposition['d2'] = ema2 - ema4

        # d3: Medium frequency
        if 'd3' in self.scales:
            decomposition['d3'] = ema4 - ema8

        # d4: Medium-Low frequency
        if 'd4' in self.scales:
            decomposition['d4'] = ema8 - ema16

        # s4: Trend / Low frequency
        if 's4' in self.scales:
            decomposition['s4'] = ema16

        return decomposition

    def decompose_signal_vectorized(
        self, 
        signal: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> Dict[str, np.ndarray]:
        """
        Vectorized signal decomposition.
        
        Args:
            signal: Input signal to decompose
            timestamps: Timestamps for the signal (optional)
            
        Returns:
            Dictionary with decomposed scales
        """
        try:
            if self.verbose:
                tprint_info("📊 Vectorized signal decomposition...")
            
            # Validate input
            if len(signal) < 32:
                if self.verbose:
                    tprint_warning("   ⚠️ Signal too short for decomposition")
                return self._create_dummy_decomposition(signal)
            
            # In-place cleaning
            signal_clean = self._clean_signal_inplace(signal.copy())
            
            # CAUSAL MODE
            if self.causal:
                if self.verbose:
                    tprint_info("   🌊 Using Strict Causal Decomposition (EMA Cascade - Numba Optimized)")
                return self._decompose_causal_ema(signal_clean)

            # NON-CAUSAL (Wavelet) MODE
            # Vectorized MODWT decomposition
            if self._modwt_available:
                try:
                    # Single MODWT call for all scales
                    coeffs = pywt.modwt(signal_clean, self.wavelet, level=self.max_level)
                    
                    # Vectorized scale extraction
                    decomposition = {}
                    
                    # Extract all scales efficiently
                    n_coeffs = len(coeffs)
                    for i, scale_name in enumerate(self.scale_order):
                        if scale_name in self.scales and i < n_coeffs:
                            if scale_name == 's4':
                                # Scaling coefficients
                                decomposition[scale_name] = coeffs[-1].copy()
                            else:
                                # Detail coefficients
                                idx = i if i < self.max_level else self.max_level - 1
                                if idx < n_coeffs - 1:
                                    decomposition[scale_name] = coeffs[idx].copy()
                                else:
                                    decomposition[scale_name] = np.zeros_like(signal_clean)
                    
                    # Ensure all requested scales are present
                    for scale in self.scales:
                        if scale not in decomposition:
                            decomposition[scale] = np.zeros_like(signal_clean)
                    
                    if self.verbose:
                        tprint_success(f"   ✅ Vectorized decomposition: {len(decomposition)} scales")
                    
                    return decomposition
                    
                except Exception as e:
                    if self.verbose:
                        tprint_warning(f"   ⚠️ Vectorized MODWT failed: {e}")
                        tprint_info("   🔄 Using fallback decomposition...")
                    
                    return self._fallback_decomposition_vectorized(signal_clean)
            else:
                # Deterministic fallback when MODWT is unavailable
                return self._fallback_decomposition_vectorized(signal_clean)
                
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Vectorized signal decomposition failed: {e}")
            return self._create_dummy_decomposition(signal)
    
    # Backwards-compatible aliases
    def decompose_signal(
        self,
        signal: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> Dict[str, np.ndarray]:
        return self.decompose_signal_vectorized(signal, timestamps)

    def decompose_specialist_vectorized(
        self,
        specialist_data: pd.Series,
        specialist_name: str
    ) -> Dict[str, np.ndarray]:
        """Vectorized specialist decomposition."""
        try:
            if self.verbose:
                tprint_info(f"🎯 Vectorized {specialist_name} decomposition...")
            
            # Vectorized decomposition
            decomposition = self.decompose_signal_vectorized(
                specialist_data.values, 
                specialist_data.index if hasattr(specialist_data, 'index') else None
            )
            
            # Add specialist prefix efficiently
            spectral_components = {}
            for scale, coeffs in decomposition.items():
                spectral_components[f'{specialist_name}_{scale}'] = coeffs
            
            if self.verbose:
                tprint_success(f"   ✅ {specialist_name}: {len(spectral_components)} spectral components")
            
            return spectral_components
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Vectorized specialist decomposition failed: {e}")
            return {}
    
    def decompose_specialist(
        self,
        specialist_data: pd.Series,
        specialist_name: str
    ) -> Dict[str, np.ndarray]:
        return self.decompose_specialist_vectorized(specialist_data, specialist_name)

    def decompose_all_specialists_vectorized(
        self,
        specialists: Dict[str, pd.Series]
    ) -> Dict[str, np.ndarray]:
        """Vectorized decomposition of all specialists."""
        try:
            if self.verbose:
                tprint_info("🚀 Vectorized multi-specialist decomposition...")
            
            all_spectral_components = {}
            
            # Parallel or sequential processing
            if self.enable_parallel and len(specialists) > 1:
                # Parallel processing
                with ThreadPoolExecutor(max_workers=min(4, len(specialists))) as executor:
                    futures = [
                        executor.submit(
                            self.decompose_specialist_vectorized,
                            specialist_data, specialist_name
                        )
                        for specialist_name, specialist_data in specialists.items()
                    ]
                    
                    # Collect results
                    for future in futures:
                        try:
                            spectral_components = future.result(timeout=30)
                            all_spectral_components.update(spectral_components)
                        except Exception as e:
                            if self.verbose:
                                tprint_warning(f"      ⚠️ Parallel decomposition failed: {e}")
            else:
                # Sequential processing
                for specialist_name, specialist_data in specialists.items():
                    spectral_components = self.decompose_specialist_vectorized(
                        specialist_data, specialist_name
                    )
                    all_spectral_components.update(spectral_components)
            
            if self.verbose:
                tprint_success(f"✅ Vectorized decomposition complete:")
                tprint_info(f"   - Specialists processed: {len(specialists)}")
                tprint_info(f"   - Spectral components: {len(all_spectral_components)}")
                tprint_info(f"   - Scales per specialist: {len(self.scales)}")
                tprint_info(f"   - Processing method: {'Parallel' if self.enable_parallel else 'Sequential'}")
            
            return all_spectral_components
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Vectorized multi-specialist decomposition failed: {e}")
            return {}
    
    def decompose_all_specialists(
        self,
        specialists: Dict[str, pd.Series]
    ) -> Dict[str, np.ndarray]:
        return self.decompose_all_specialists_vectorized(specialists)

    def _fallback_decomposition_vectorized(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """Vectorized fallback decomposition using standard DWT."""
        try:
            # Use standard DWT as fallback
            coeffs = pywt.wavedec(signal, self.wavelet, level=self.max_level)
            
            decomposition = {}
            
            # Vectorized reconstruction of individual scales
            for i, scale_name in enumerate(self.scale_order):
                if scale_name in self.scales and i < len(coeffs) - 1:
                    # Create dummy coefficients WITH CORRECT SHAPES
                    dummy_coeffs = [np.zeros_like(c) for c in coeffs]
                    dummy_coeffs[i] = coeffs[i]
                    
                    try:
                        reconstruction = pywt.waverec(dummy_coeffs, self.wavelet)
                        
                        # Efficient length handling
                        if len(reconstruction) > len(signal):
                            reconstruction = reconstruction[:len(signal)]
                        elif len(reconstruction) < len(signal):
                            padded = np.zeros(len(signal))
                            padded[:len(reconstruction)] = reconstruction
                            reconstruction = padded
                        
                        decomposition[scale_name] = reconstruction
                        
                    except Exception:
                        decomposition[scale_name] = np.zeros_like(signal)
            
            # Scaling coefficients
            if len(coeffs) > 0:
                scaling_coeffs = coeffs[-1]
                if len(scaling_coeffs) > 0:
                    s4_signal = np.full_like(signal, np.mean(scaling_coeffs))
                    decomposition['s4'] = s4_signal
                else:
                    decomposition['s4'] = np.zeros_like(signal)
            
            return decomposition
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"   ⚠️ Vectorized fallback decomposition failed: {e}")
            return self._create_dummy_decomposition(signal)
    
    def _create_dummy_decomposition(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """Create dummy decomposition when all methods fail."""
        decomposition = {}
        for scale in self.scales:
            decomposition[scale] = np.zeros_like(signal)
        return decomposition
    
    def get_scale_info(self) -> Dict[str, Dict[str, str]]:
        """Get information about all scales."""
        return self.scale_definitions
    
    def validate_decomposition_vectorized(
        self,
        original_signal: np.ndarray,
        decomposition: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Vectorized decomposition validation."""
        try:
            # Vectorized signal reconstruction
            reconstructed = np.zeros_like(original_signal)
            for coeffs in decomposition.values():
                if len(coeffs) == len(original_signal):
                    reconstructed += coeffs
            
            mse = np.mean((original_signal - reconstructed) ** 2)
            
            if np.std(original_signal) > 0 and np.std(reconstructed) > 0:
                correlation = np.corrcoef(original_signal, reconstructed)[0, 1]
            else:
                correlation = 0.0
            
            validation_metrics = {
                'reconstruction_mse': mse,
                'reconstruction_correlation': correlation if not np.isnan(correlation) else 0.0,
                'signal_energy': np.var(original_signal),
                'reconstruction_energy': np.var(reconstructed),
                'validation_method': 'vectorized'
            }
            
            return validation_metrics
        except Exception as e:
            if self.verbose:
                tprint_warning(f"   ⚠️ Vectorized decomposition validation failed: {e}")
            return {'reconstruction_mse': float('inf'), 'reconstruction_correlation': 0.0}

    def denoise_signal_vectorized(
        self,
        signal: np.ndarray,
        threshold_method: str = 'visushrink',
        threshold_mode: str = 'soft',
        sigma_est: str = 'mad'
    ) -> np.ndarray:
        """
        Denoise signal.
        If causal mode is enabled, uses strict causal EWMA smoothing.
        If not, uses Wavelet thresholding (non-causal).
        
        Args:
            signal: Input signal
            
        Returns:
            Denoised signal
        """
        # Causal Mode (EWMA Smoothing)
        if self.causal:
            if self.verbose:
                tprint_info("   🌊 Using Strictly Causal Denoising (EMA Cascade - Numba Optimized)")
            # Use span=10 as standard causal smoother
            # span=10 -> alpha = 2/11
            alpha = 2.0 / 11.0
            return _numba_ewma(signal.astype(np.float64), alpha, adjust=False)

        # Non-Causal Mode (Wavelet)
        try:
            coeffs = pywt.wavedec(signal, self.wavelet, level=self.max_level)
            d1 = coeffs[-1]
            if sigma_est == 'mad':
                sigma = np.median(np.abs(d1 - np.median(d1))) / 0.6745
            else:
                sigma = np.std(d1)
                
            if threshold_method == 'visushrink':
                thresh = sigma * np.sqrt(2 * np.log(len(signal)))
            else:
                thresh = sigma
                
            new_coeffs = [coeffs[0]]
            for i in range(1, len(coeffs)):
                new_coeffs.append(pywt.threshold(coeffs[i], thresh, mode=threshold_mode))
                
            denoised = pywt.waverec(new_coeffs, self.wavelet)
            
            if len(denoised) > len(signal):
                denoised = denoised[:len(signal)]
            elif len(denoised) < len(signal):
                padded = np.zeros_like(signal)
                padded[:len(denoised)] = denoised
                padded[len(denoised):] = denoised[-1] 
                denoised = padded
                
            return denoised
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"   ⚠️ Wavelet Denoising failed: {e}")
            return signal


# Convenience functions for quick usage
def quick_vectorized_wavelet_decompose(
    specialist_data: pd.Series,
    specialist_name: str,
    enable_parallel: bool = True,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """Quick vectorized wavelet decomposition for a single specialist (Causal by default)."""
    engine = OptimizedWaveletDecomposition(
        enable_parallel=enable_parallel,
        verbose=verbose,
        causal=True
    )
    return engine.decompose_specialist_vectorized(specialist_data, specialist_name)


def quick_vectorized_multi_specialist_decompose(
    specialists: Dict[str, pd.Series],
    enable_parallel: bool = True,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """Quick vectorized wavelet decomposition for multiple specialists (Causal by default)."""
    engine = OptimizedWaveletDecomposition(
        enable_parallel=enable_parallel,
        verbose=verbose,
        causal=True
    )
    return engine.decompose_all_specialists_vectorized(specialists)


if __name__ == "__main__":
    # Example usage
    print("Optimized Wavelet Decomposition Engine for AEDL (Strictly Causal)")
    print("Use quick_vectorized_wavelet_decompose() or quick_vectorized_multi_specialist_decompose() for quick usage")
    
    engine = OptimizedWaveletDecomposition(causal=True)
    print("\nScale Definitions (Approximated by Causal Filters):")
    for scale, info in engine.get_scale_info().items():
        print(f"  {scale}: {info['name']} ({info['timeframe']}) - {info['description']}")
