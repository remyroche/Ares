"""
Wavelet Decomposition Engine for Adaptive Event-Driven Labeling (AEDL)

This module implements Maximal Overlap Discrete Wavelet Transform (MODWT)
with 5-scale decomposition optimized for 2-4 hour trading strategies.

Key Features:
- 5-scale MODWT decomposition (d1, d2, d3, d4, s4)
- Multiresolution Analysis (MRA) for spectral specialists
- Phase synchronization analysis
- Optimized scales for 2-4h holding periods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy import signal
from scipy.stats import pearsonr
import pywt
from concurrent.futures import ThreadPoolExecutor
import hashlib

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    # Fallback print functions
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")


class WaveletDecomposition:
    """
    Maximal Overlap Discrete Wavelet Transform (MODWT) engine.
    
    Implements 5-scale decomposition optimized for 2-4 hour trading:
    - d1: 5m-15m (Micro-Shock)
    - d2: 15m-1h (Dealer Flow)
    - d3: 1h-4h (Session Trend)
    - d4: 4h-12h (Causal Baseline)
    - s4: 12h+ (Macro Ground)
    """
    
    def __init__(
        self,
        wavelet: str = 'db4',
        scales: List[str] = ['d1', 'd2', 'd3', 'd4', 's4'],
        max_level: int = 4,
        enable_parallel: bool = True,
        cache_size: int = 64,
        verbose: bool = True
    ):
        """
        Initialize Wavelet Decomposition Engine.
        
        Args:
            wavelet: Wavelet family for decomposition
            scales: List of scale names
            max_level: Maximum decomposition level
            enable_parallel: Enable parallel processing
            cache_size: Size of decomposition cache
            verbose: Whether to print progress information
        """
        self.verbose = verbose
        self.wavelet = wavelet
        self.scales = scales
        self.max_level = max_level
        self.enable_parallel = enable_parallel
        self.cache_size = cache_size
        
        # Cache for decomposition results
        self._decomposition_cache = {}
        
        # Pre-allocate scale order for efficiency
        self.scale_order = ['d1', 'd2', 'd3', 'd4', 's4']
        
        # Scale definitions optimized for 2-4h trades
        self.scale_definitions = {
            'd1': {'name': 'Micro-Shock', 'timeframe': '5m-15m', 'description': 'High-frequency order flow and HFT noise'},
            'd2': {'name': 'Dealer Flow', 'timeframe': '15m-1h', 'description': 'Core inventory cycle for trade duration'},
            'd3': {'name': 'Session Trend', 'timeframe': '1h-4h', 'description': 'Parent move of specific trade'},
            'd4': {'name': 'Causal Baseline', 'timeframe': '4h-12h', 'description': 'Structural gravity of the day'},
            's4': {'name': 'Macro Ground', 'timeframe': '12h+', 'description': 'Low-frequency regime (Anchored state)'}
        }
        
        if self.verbose:
            tprint_info("🔬 Wavelet Decomposition Engine: Initializing...")
            tprint_info(f"   ⚙️ Wavelet family: {wavelet}")
            tprint_info(f"   ⚙️ Decomposition levels: {max_level}")
            tprint_info(f"   ⚙️ Scales: {', '.join(scales)}")
            tprint_info(f"   ⚙️ Parallel processing: {enable_parallel}")
            tprint_info(f"   ⚙️ Cache size: {cache_size}")
            tprint_success("   ✅ Wavelet Decomposition Engine: Initialization complete")
    
    def _get_cache_key(self, signal: np.ndarray) -> str:
        """Generate cache key for signal array."""
        return hashlib.md5(signal.tobytes()).hexdigest()[:16]
    
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
        for i in range(first_valid + 1, len(signal)):
            if np.isnan(signal[i]):
                signal[i] = signal[i-1]
        
        return signal
    
    def decompose_signal(
        self, 
        signal: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> Dict[str, np.ndarray]:
        """
        Optimized signal decomposition into 5 wavelet scales.
        
        Args:
            signal: Input signal to decompose
            timestamps: Timestamps for the signal (optional)
            
        Returns:
            Dictionary with decomposed scales
        """
        try:
            if self.verbose:
                tprint_info("📊 Optimized signal decomposition...")
            
            # Check cache first
            cache_key = self._get_cache_key(signal)
            if cache_key in self._decomposition_cache:
                if self.verbose:
                    tprint_info("   🎯 Cache hit for signal decomposition")
                return self._decomposition_cache[cache_key]
            
            # Validate input
            if len(signal) < 32:
                if self.verbose:
                    tprint_warning("   ⚠️ Signal too short for wavelet decomposition")
                return self._create_dummy_decomposition(signal)
            
            # In-place cleaning
            signal_clean = self._clean_signal_inplace(signal.copy())
            
            # Vectorized MODWT decomposition
            try:
                # Stationary Wavelet Transform (SWT)
                # Returns list of tuples [(cA_n, cD_n), ..., (cA_1, cD_1)]
                coeffs = pywt.swt(signal_clean, self.wavelet, level=self.max_level)
                
                # Vectorized scale extraction
                decomposition = {}
                
                # Extract all scales efficiently
                # coeffs items are (approx, detail) for each level, starting from max_level down to 1
                # d1 corresponds to coeffs[-1][1], d2 to coeffs[-2][1], etc.
                for i, scale_name in enumerate(self.scale_order):
                    if scale_name in self.scales:
                        if scale_name == 's4':
                            # Scaling coefficients: Approximation at max level -> coeffs[0][0]
                            decomposition[scale_name] = coeffs[0][0].copy()
                        else:
                            # Detail coefficients: d_i -> coeffs[-(i+1)][1]
                            # i=0 (d1) -> index -1 -> coeffs[-1][1]
                            # i=1 (d2) -> index -2 -> coeffs[-2][1]
                            # etc.
                            # Ensure we don't go out of bounds if i >= max_level (shouldn't happen with correct config)
                            idx = -(i + 1)
                            if abs(idx) <= len(coeffs):
                                decomposition[scale_name] = coeffs[idx][1].copy()
                            else:
                                decomposition[scale_name] = np.zeros_like(signal_clean)
                
                # Ensure all requested scales are present
                for scale in self.scales:
                    if scale not in decomposition:
                        decomposition[scale] = np.zeros_like(signal_clean)
                
                # Cache result
                if len(self._decomposition_cache) < self.cache_size:
                    self._decomposition_cache[cache_key] = decomposition
                
                if self.verbose:
                    tprint_success(f"   ✅ Optimized decomposition: {len(decomposition)} scales")
                
                return decomposition
                
            except Exception as e:
                if self.verbose:
                    tprint_warning(f"   ⚠️ Vectorized MODWT failed: {e}")
                    tprint_info("   🔄 Using fallback decomposition...")
                
                return self._fallback_decomposition_vectorized(signal_clean)
                
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Optimized signal decomposition failed: {e}")
            return self._create_dummy_decomposition(signal)
    
    def decompose_specialist(
        self,
        specialist_data: pd.Series,
        specialist_name: str
    ) -> Dict[str, np.ndarray]:
        """
        Decompose specialist data into spectral components.
        
        Args:
            specialist_data: Time series data for specialist
            specialist_name: Name of the specialist
            
        Returns:
            Dictionary with spectral components
        """
        try:
            if self.verbose:
                tprint_info(f"🎯 Decomposing {specialist_name} specialist...")
            
            # Decompose the signal
            decomposition = self.decompose_signal(
                specialist_data.values, 
                specialist_data.index if hasattr(specialist_data, 'index') else None
            )
            
            # Add specialist prefix to scale names
            spectral_components = {}
            for scale, coeffs in decomposition.items():
                spectral_components[f'{specialist_name}_{scale}'] = coeffs
            
            if self.verbose:
                tprint_success(f"   ✅ {specialist_name}: {len(spectral_components)} spectral components")
            
            return spectral_components
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Specialist decomposition failed: {e}")
            return {}
    
    def decompose_all_specialists(
        self,
        specialists: Dict[str, pd.Series]
    ) -> Dict[str, np.ndarray]:
        """
        Optimized decomposition of all specialists with parallel processing.
        
        Args:
            specialists: Dictionary of specialist time series
            
        Returns:
            Dictionary with all spectral components
        """
        try:
            if self.verbose:
                tprint_info("🚀 Optimized multi-specialist decomposition...")
            
            all_spectral_components = {}
            
            # Parallel or sequential processing
            if self.enable_parallel and len(specialists) > 1:
                # Parallel processing
                with ThreadPoolExecutor(max_workers=min(4, len(specialists))) as executor:
                    futures = [
                        executor.submit(
                            self.decompose_specialist,
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
                    spectral_components = self.decompose_specialist(specialist_data, specialist_name)
                    all_spectral_components.update(spectral_components)
            
            if self.verbose:
                tprint_success(f"✅ Optimized decomposition complete:")
                tprint_info(f"   - Specialists processed: {len(specialists)}")
                tprint_info(f"   - Spectral components: {len(all_spectral_components)}")
                tprint_info(f"   - Scales per specialist: {len(self.scales)}")
                tprint_info(f"   - Processing method: {'Parallel' if self.enable_parallel else 'Sequential'}")
                tprint_info(f"   - Cache size: {len(self._decomposition_cache)}")
            
            return all_spectral_components
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Optimized multi-specialist decomposition failed: {e}")
            return {}
    
    def _fallback_decomposition_vectorized(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """Vectorized fallback decomposition using standard DWT."""
        try:
            # Use standard DWT as fallback
            coeffs = pywt.wavedec(signal, self.wavelet, level=self.max_level)
            
            decomposition = {}
            
            # Vectorized reconstruction of individual scales
            for i, scale_name in enumerate(self.scale_order):
                if scale_name in self.scales and i < len(coeffs) - 1:
                    # Create dummy coefficients for reconstruction
                    dummy_coeffs = [np.zeros_like(signal) for _ in range(len(coeffs))]
                    dummy_coeffs[i] = coeffs[i]
                    
                    try:
                        reconstruction = pywt.waverec(dummy_coeffs, self.wavelet)
                        
                        # Efficient length handling
                        if len(reconstruction) > len(signal):
                            reconstruction = reconstruction[:len(signal)]
                        elif len(reconstruction) < len(signal):
                            # In-place padding
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
                    # Vectorized scaling coefficient creation
                    s4_signal = np.full_like(signal, np.mean(scaling_coeffs))
                    decomposition['s4'] = s4_signal
                else:
                    decomposition['s4'] = np.zeros_like(signal)
            
            return decomposition
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"   ⚠️ Vectorized fallback decomposition failed: {e}")
            return self._create_dummy_decomposition(signal)
    
    def clear_cache(self):
        """Clear decomposition cache to free memory."""
        self._decomposition_cache.clear()
        if self.verbose:
            tprint_info("🧹 Cleared wavelet decomposition cache")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'decomposition_cache_size': len(self._decomposition_cache),
            'cache_limit': self.cache_size
        }
    
    def _fallback_decomposition(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """Fallback decomposition using standard DWT."""
        try:
            # Use standard DWT as fallback
            coeffs = pywt.wavedec(signal, self.wavelet, level=self.max_level)
            
            decomposition = {}
            
            # Reconstruct individual scales
            for i in range(len(coeffs) - 1):
                scale_name = f'd{i+1}'
                if scale_name in self.scales:
                    # Create dummy coefficients for this level
                    dummy_coeffs = [np.zeros_like(signal) for _ in range(len(coeffs))]
                    dummy_coeffs[i] = coeffs[i]
                    reconstruction = pywt.waverec(dummy_coeffs, self.wavelet)
                    
                    # Pad or truncate to match original length
                    if len(reconstruction) > len(signal):
                        reconstruction = reconstruction[:len(signal)]
                    elif len(reconstruction) < len(signal):
                        reconstruction = np.pad(reconstruction, (0, len(signal) - len(reconstruction)))
                    
                    decomposition[scale_name] = reconstruction
            
            # Scaling coefficients
            if len(coeffs) > 0:
                decomposition['s4'] = np.full_like(signal, coeffs[-1].mean())
            
            return decomposition
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"   ⚠️ Fallback decomposition also failed: {e}")
            return self._create_dummy_decomposition(signal)
    
    def _create_dummy_decomposition(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """Create dummy decomposition when all methods fail."""
        if self.verbose:
            tprint_info("📝 Creating dummy decomposition")
        decomposition = {}
        for scale in self.scales:
            decomposition[scale] = np.zeros_like(signal)
        return decomposition
    
    def get_scale_info(self) -> Dict[str, Dict[str, str]]:
        """Get information about all scales."""
        if self.verbose:
            tprint_info("📋 Retrieving scale information")
        return self.scale_definitions
    
    def validate_decomposition(
        self,
        original_signal: np.ndarray,
        decomposition: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Validate decomposition by reconstructing signal.
        
        Args:
            original_signal: Original signal
            decomposition: Decomposed scales
            
        Returns:
            Validation metrics
        """
        try:
            # Simple reconstruction (sum of all scales)
            reconstructed = np.zeros_like(original_signal)
            for coeffs in decomposition.values():
                if len(coeffs) == len(original_signal):
                    reconstructed += coeffs
            
            # Calculate reconstruction error
            mse = np.mean((original_signal - reconstructed) ** 2)
            correlation = np.corrcoef(original_signal, reconstructed)[0, 1]
            
            validation_metrics = {
                'reconstruction_mse': mse,
                'reconstruction_correlation': correlation if not np.isnan(correlation) else 0.0,
                'signal_energy': np.var(original_signal),
                'reconstruction_energy': np.var(reconstructed)
            }
            
            return validation_metrics
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"   ⚠️ Decomposition validation failed: {e}")
            return {'reconstruction_mse': float('inf'), 'reconstruction_correlation': 0.0}


# Convenience functions for quick usage
def quick_wavelet_decompose(
    specialist_data: pd.Series,
    specialist_name: str,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """Quick wavelet decomposition for a single specialist."""
    if verbose:
        tprint_info(f"🚀 Quick decomposing specialist: {specialist_name}")
    engine = WaveletDecomposition(verbose=verbose)
    return engine.decompose_specialist(specialist_data, specialist_name)


def quick_multi_specialist_decompose(
    specialists: Dict[str, pd.Series],
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """Quick wavelet decomposition for multiple specialists."""
    if verbose:
        tprint_info(f"🚀 Quick decomposing {len(specialists)} specialists")
    engine = WaveletDecomposition(verbose=verbose)
    return engine.decompose_all_specialists(specialists)


if __name__ == "__main__":
    # Example usage
    print("Wavelet Decomposition Engine for AEDL")
    print("Use quick_wavelet_decompose() or quick_multi_specialist_decompose() for quick usage")
    
    # Display scale information
    engine = WaveletDecomposition()
    print("\nScale Definitions:")
    for scale, info in engine.get_scale_info().items():
        print(f"  {scale}: {info['name']} ({info['timeframe']}) - {info['description']}")
