"""
Resonance Detector for Adaptive Event-Driven Labeling (AEDL)

This module implements cross-scale resonance detection with phase synchronization
to identify harmonic entries and structural breakouts in the market.

Key Features:
- Wavelet coherence calculation between scales
- Phase lead/lag detection for breakout vs reversion
- Resonance State Vector (RSV) calculation
- Cross-scale resonance analysis for 4 specialists
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy import signal
from scipy.stats import pearsonr
import warnings
from functools import lru_cache
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


class ResonanceDetector:
    """
    Cross-scale resonance detector with phase synchronization.
    
    Identifies harmonic entries by analyzing coherence and phase relationships
    between different wavelet scales (micro vs macro).
    """
    
    def __init__(
        self,
        coherence_threshold: float = 0.7,
        phase_threshold: float = 0.1,
        cache_size: int = 128,
        enable_parallel: bool = True,
        verbose: bool = True
    ):
        """
        Initialize Optimized Resonance Detector.
        
        Args:
            coherence_threshold: Minimum coherence for resonance
            phase_threshold: Phase difference threshold for lead/lag
            cache_size: Size of LRU cache for resonance calculations
            enable_parallel: Enable parallel processing
            verbose: Whether to print progress information
        """
        self.verbose = verbose
        self.coherence_threshold = coherence_threshold
        self.phase_threshold = phase_threshold
        # Fix user-reported cache size = 0 issue
        self.cache_size = cache_size if cache_size > 0 else 128
        self.enable_parallel = enable_parallel
        
        # Cache for resonance calculations
        self._coherence_cache: Dict[str, Tuple[float, np.ndarray]] = {}
        self._phase_cache: Dict[str, Tuple[float, np.ndarray]] = {}
        self._resonance_cache: Dict[str, Dict[str, Any]] = {}
        
        # Scale pairs for micro-macro analysis
        self.micro_macro_pairs = [
            ('d1', 'd3'),  # Micro-shock vs Session Trend
            ('d2', 'd4'),  # Dealer Flow vs Causal Baseline
            ('d1', 'd4'),  # Micro-shock vs Causal Baseline
            ('d2', 'd3'),  # Dealer Flow vs Session Trend
        ]
        
        if self.verbose:
            tprint_info("🚀 Optimized Resonance Detector: Initializing...")
            tprint_info(f"   ⚙️ Coherence threshold: {coherence_threshold}")
            tprint_info(f"   ⚙️ Phase threshold: {phase_threshold}")
            tprint_info(f"   ⚙️ Cache size: {cache_size}")
            tprint_info(f"   ⚙️ Parallel processing: {enable_parallel}")
            tprint_info(f"   ⚙️ Micro-macro pairs: {len(self.micro_macro_pairs)}")
            tprint_success("   ✅ Optimized Resonance Detector: Initialization complete")
    
    def _get_cache_key(self, data1: np.ndarray, data2: np.ndarray, operation: str) -> str:
        """Generate cache key for data arrays."""
        # Use hash of data for caching
        data1_hash = hashlib.md5(data1.tobytes()).hexdigest()[:16]
        data2_hash = hashlib.md5(data2.tobytes()).hexdigest()[:16]
        return f"{operation}_{data1_hash}_{data2_hash}"
    
    def _clean_data_inplace(self, fast: np.ndarray, slow: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Clean data in-place for efficiency.
        
        Args:
            fast: Fast scale coefficients
            slow: Slow scale coefficients
            
        Returns:
            Tuple of cleaned arrays
        """
        # Ensure equal length
        min_len = min(len(fast), len(slow))
        if len(fast) != min_len:
            fast = fast[:min_len]
        if len(slow) != min_len:
            slow = slow[:min_len]
        
        # Remove NaN values in-place
        valid_mask = ~(np.isnan(fast) | np.isnan(slow))
        
        if not np.any(valid_mask):
            return np.array([0.0]), np.array([0.0])
        
        return fast[valid_mask], slow[valid_mask]
    
    def calculate_wavelet_coherence(
        self,
        coeffs_fast: np.ndarray,
        coeffs_slow: np.ndarray,
        nperseg: int = 64
    ) -> Tuple[float, np.ndarray]:
        """
        Optimized coherence calculation with caching.
        
        Args:
            coeffs_fast: Fast scale coefficients (micro)
            coeffs_slow: Slow scale coefficients (macro)
            nperseg: Segment length for coherence calculation
            
        Returns:
            Tuple of (squared coherence summary, per-bar coherence series)
        """
        try:
            # Clean data in-place
            fast_clean, slow_clean = self._clean_data_inplace(coeffs_fast, coeffs_slow)
            
            if len(fast_clean) < nperseg:
                return 0.0, np.zeros_like(fast_clean)
            
            # Robustness check: fail fast on constant signals (variance ~ 0)
            if np.std(fast_clean) < 1e-9 or np.std(slow_clean) < 1e-9:
                return 0.0, np.zeros_like(fast_clean)

            # Check cache first
            cache_key = self._get_cache_key(fast_clean, slow_clean, 'coherence')
            if cache_key in self._coherence_cache:
                cached_summary, cached_series = self._coherence_cache[cache_key]
                return cached_summary, cached_series.copy()
            
            # Adaptive nperseg based on data length
            adaptive_nperseg = min(nperseg, len(fast_clean)//4)
            if adaptive_nperseg < 16:
                adaptive_nperseg = 16
            
            # Vectorized coherence calculation
            try:
                freqs, coherence = signal.coherence(
                    fast_clean, slow_clean,
                    fs=1.0, nperseg=adaptive_nperseg
                )
                
                # Vectorized frequency filtering
                relevant_freq_mask = freqs <= 0.5
                if np.any(relevant_freq_mask):
                    # Vectorized mean calculation
                    avg_coherence = np.mean(coherence[relevant_freq_mask])
                else:
                    avg_coherence = np.mean(coherence)
                
                coherence_summary = avg_coherence ** 2
                coherence_series = self._compute_coherence_series(fast_clean, slow_clean)
                
                # Cache result
                if len(self._coherence_cache) < self.cache_size:
                    self._coherence_cache[cache_key] = (coherence_summary, coherence_series.copy())
                
                return coherence_summary, coherence_series
                
            except Exception as e:
                if self.verbose:
                    tprint_warning(f"      ⚠️ Vectorized coherence failed: {e}")
                return 0.0, np.zeros_like(fast_clean)
                
        except Exception as e:
            if self.verbose:
                tprint_warning(f"      ⚠️ Coherence calculation failed: {e}")
            return 0.0, np.zeros_like(coeffs_fast)
    
    def _compute_coherence_series(
        self,
        fast: np.ndarray,
        slow: np.ndarray,
        window: int = 32
    ) -> np.ndarray:
        """
        Compute rolling coherence series using correlation.
        
        Args:
            fast: Fast scale signal
            slow: Slow scale signal
            window: Rolling window size
            
        Returns:
            Coherence series (0-1)
        """
        try:
            # Use pandas for efficient rolling correlation
            s1 = pd.Series(fast)
            s2 = pd.Series(slow)
            
            # Rolling correlation
            corr = s1.rolling(window=window, min_periods=window//2).corr(s2)
            
            # Convert to coherence (squared correlation) to match magnitude
            coherence = corr.pow(2).fillna(0.0).values
            
            return coherence
            
        except Exception:
            return np.zeros_like(fast)
            
    def calculate_phase_lead_lag(
        self,
        coeffs_fast: np.ndarray,
        coeffs_slow: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Vectorized phase difference calculation with caching.
        
        Args:
            coeffs_fast: Fast scale coefficients (micro)
            coeffs_slow: Slow scale coefficients (macro)
            
        Returns:
            Tuple of (phase summary, per-bar phase difference series)
        """
        try:
            # Clean data in-place
            fast_clean, slow_clean = self._clean_data_inplace(coeffs_fast, coeffs_slow)
            
            if len(fast_clean) < 32:
                return 0.0, np.zeros_like(fast_clean)
            
            # Robustness check: fail fast on constant signals
            if np.std(fast_clean) < 1e-9 or np.std(slow_clean) < 1e-9:
                return 0.0, np.zeros_like(fast_clean)

            # Check cache first
            cache_key = self._get_cache_key(fast_clean, slow_clean, 'phase')
            if cache_key in self._phase_cache:
                cached_phase, cached_series = self._phase_cache[cache_key]
                return cached_phase, cached_series.copy()
            
            # Zero-mean in-place before Hilbert transform
            fast_clean = fast_clean - np.mean(fast_clean)
            slow_clean = slow_clean - np.mean(slow_clean)
            
            # Instantaneous phase via Hilbert transform
            analytic_fast = signal.hilbert(fast_clean)
            analytic_slow = signal.hilbert(slow_clean)
            phase_fast = np.unwrap(np.angle(analytic_fast))
            phase_slow = np.unwrap(np.angle(analytic_slow))
            
            # Normalize phase difference to [-1, 1]
            raw_phase = phase_fast - phase_slow
            phase_series = np.clip(raw_phase / np.pi, -1.0, 1.0)
            
            phase_summary = float(np.clip(np.mean(phase_series), -1.0, 1.0))
            
            # Cache result
            if len(self._phase_cache) < self.cache_size:
                self._phase_cache[cache_key] = (phase_summary, phase_series.copy())
            
            return phase_summary, phase_series
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"      ⚠️ Vectorized phase calculation failed: {e}")
            zero_series = np.zeros_like(coeffs_fast)
            return 0.0, zero_series
    
    def calculate_resonance_score(
        self,
        coeffs_fast: np.ndarray,
        coeffs_slow: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate enhanced resonance score with phase synchronization.
        
        Args:
            coeffs_fast: Fast scale coefficients (micro)
            coeffs_slow: Slow scale coefficients (macro)
            
        Returns:
            Dictionary with scalar summary and per-bar resonance series
        """
        # Removed inner verbose logging to prevent "Retry loops" spam
        # if self.verbose:
        #    tprint_info("⭐ Calculating resonance score")
        try:
            # 1. Squared Wavelet Coherence (Strength)
            coherence_summary, coherence_series = self.calculate_wavelet_coherence(
                coeffs_fast, coeffs_slow
            )
            
            # 2. Phase Difference (Direction)
            phase_summary, phase_series = self.calculate_phase_lead_lag(
                coeffs_fast, coeffs_slow
            )
            
            resonance_series = self._combine_resonance_series(coherence_series, phase_series)
            
            # 3. Structural Resonance Score
            # High coherence + Micro leading Macro = Structural Breakout
            is_leading = 1.0 if phase_summary > self.phase_threshold else 0.5
            resonance_score = coherence_summary * is_leading
            resonance_score = float(np.clip(resonance_score, 0.0, 1.0))
            
            return {
                'summary': resonance_score,
                'series': resonance_series,
                'coherence_summary': float(coherence_summary),
                'coherence_series': coherence_series,
                'phase_summary': float(phase_summary),
                'phase_series': phase_series
            }
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"      ⚠️ Resonance score calculation failed: {e}")
            return {
                'summary': 0.0,
                'series': np.zeros_like(coeffs_fast),
                'coherence_summary': 0.0,
                'coherence_series': np.zeros_like(coeffs_fast),
                'phase_summary': 0.0,
                'phase_series': np.zeros_like(coeffs_fast)
            }
    
    def _log_resonance_sanity(
        self,
        resonance_summary: float,
        coherence_summary: float,
        phase_summary: float
    ):
        """Log resonance sanity check metrics."""
        if resonance_summary > 0.8:
            tprint_info(f"      ⭐ High resonance detected: {resonance_summary:.3f}")
        elif resonance_summary < 0.1:
            pass # functional equivalence to low resonance logging removal
            
    def _combine_resonance_series(
        self,
        coherence_series: np.ndarray,
        phase_series: np.ndarray
    ) -> np.ndarray:
        """
        Combine coherence and phase series into resonance series.
        
        Args:
            coherence_series: Coherence time series
            phase_series: Phase difference time series
            
        Returns:
            Resonance score time series
        """
        # Resonance = Coherence * (1 - |Phase|)
        # Higher coherence and lower phase difference = higher resonance
        resonance = coherence_series * (1.0 - np.abs(phase_series))
        return np.clip(resonance, 0.0, 1.0)
        
    def compute_specialist_resonance(
        self,
        spectral_components: Dict[str, np.ndarray],
        specialist_name: str
    ) -> Dict[str, float]:
        """
        Compute resonance scores for a single specialist.
        
        Args:
            spectral_components: Spectral components for all specialists
            specialist_name: Name of the specialist
            
        Returns:
            Dictionary of resonance scores for this specialist
        """
        # Removed inner verbose logging to reduce noise
        # if self.verbose:
        #    tprint_info(f"🎯 Computing resonance for {specialist_name}")
        try:
            specialist_resonance = {}
            
            for fast_scale, slow_scale in self.micro_macro_pairs:
                fast_key = f'{specialist_name}_{fast_scale}'
                slow_key = f'{specialist_name}_{slow_scale}'
                
                if fast_key in spectral_components and slow_key in spectral_components:
                    resonance_key = f'{specialist_name}_{fast_scale}_{slow_scale}_resonance'
                    resonance_data = self.calculate_resonance_score(
                        spectral_components[fast_key],
                        spectral_components[slow_key]
                    )
                    specialist_resonance[resonance_key] = resonance_data
            
            return specialist_resonance
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"   ⚠️ Specialist resonance calculation failed: {e}")
            return {}
    
    def compute_all_resonances(
        self,
        spectral_components: Dict[str, np.ndarray],
        specialist_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Optimized computation of all resonance scores with parallel processing.
        
        Args:
            spectral_components: Spectral components for all specialists
            specialist_names: Explicit specialist ordering (optional)
            
        Returns:
            Dictionary of all resonance scores
        """
        try:
            if self.verbose:
                tprint_info("🚀 Computing optimized cross-scale resonance scores...")
            
            all_resonances: Dict[str, Dict[str, Any]] = {}
            
            # Determine specialist names
            if specialist_names is None:
                specialist_names = self._infer_specialist_names(spectral_components)
            else:
                # Preserve order and drop duplicates
                seen = set()
                specialist_names = [name for name in specialist_names if not (name in seen or seen.add(name))]
            
            if not specialist_names:
                if self.verbose:
                    tprint_warning("   ⚠️ No specialist names available for resonance computation")
                return {}
            
            # Parallel or sequential processing
            if self.enable_parallel and len(specialist_names) > 1:
                # Parallel processing
                with ThreadPoolExecutor(max_workers=min(4, len(specialist_names))) as executor:
                    futures = [
                        executor.submit(
                            self.compute_specialist_resonance,
                            spectral_components,
                            specialist_name
                        )
                        for specialist_name in specialist_names
                    ]
                    
                    # Collect results
                    for future in futures:
                        try:
                            specialist_resonance = future.result(timeout=30)
                            all_resonances.update(specialist_resonance)
                        except Exception as e:
                            if self.verbose:
                                tprint_warning(f"      ⚠️ Parallel resonance computation failed: {e}")
            else:
                # Sequential processing
                for specialist_name in specialist_names:
                    specialist_resonance = self.compute_specialist_resonance(
                        spectral_components, specialist_name
                    )
                    all_resonances.update(specialist_resonance)
            
            if self.verbose:
                high_resonance = sum(
                    1 for data in all_resonances.values()
                    if data.get('summary', 0.0) > self.coherence_threshold
                )
                avg_resonance = np.mean([
                    data.get('summary', 0.0)
                    for data in all_resonances.values()
                ]) if all_resonances else 0.0
                tprint_success(f"✅ Optimized resonance analysis complete:")
                tprint_info(f"   - Total resonance scores: {len(all_resonances)}")
                tprint_info(f"   - High resonance scores: {high_resonance}")
                tprint_info(f"   - Average resonance: {avg_resonance:.3f}")
                tprint_info(f"   - Cache hits: Coherence={len(self._coherence_cache)}, Phase={len(self._phase_cache)}")
                tprint_info(f"   - Processing method: {'Parallel' if self.enable_parallel else 'Sequential'}")
                
                # Compute coherence and phase summaries for diagnostics
                coherence_summary = {'mean_coherence': np.mean([r.get('coherence', 0) for r in all_resonances.values()])} if all_resonances else {}
                phase_summary = {'mean_phase': np.mean([r.get('phase', 0) for r in all_resonances.values()])} if all_resonances else {}
                
                self._log_resonance_sanity(all_resonances, coherence_summary, phase_summary)
            
            return all_resonances
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Optimized resonance calculation failed: {e}")
            return {}
    
    @staticmethod
    def _infer_specialist_names(spectral_components: Dict[str, np.ndarray]) -> List[str]:
        """Infer specialist names from flattened spectral component keys."""
        specialist_names = []
        seen = set()
        for key in spectral_components.keys():
            parts = key.split('_')
            if len(parts) < 2:
                continue
            name = '_'.join(parts[:-1])
            if name not in seen:
                seen.add(name)
                specialist_names.append(name)
        return specialist_names
    
    def _log_resonance_sanity(self, all_resonances: Dict[str, float], 
                          coherence_summary: Optional[Dict] = None, 
                          phase_summary: Optional[Dict] = None):
        """
        Log comprehensive resonance diagnostics for debugging zero resonance issues.
        
        Args:
            all_resonances: Dictionary of resonance scores
            coherence_summary: Optional coherence statistics
            phase_summary: Optional phase statistics
        """
        if not all_resonances:
            tprint_error("❌ CRITICAL: No resonance scores computed")
            return
            
        # Basic statistics
        resonance_values = list(all_resonances.values())
        mean_resonance = np.mean(resonance_values)
        std_resonance = np.std(resonance_values)
        max_resonance = np.max(resonance_values)
        min_resonance = np.min(resonance_values)
        
        tprint_info(f"📊 Resonance Diagnostics:")
        tprint_info(f"   - Total pairs: {len(all_resonances)}")
        tprint_info(f"   - Mean resonance: {mean_resonance:.6f}")
        tprint_info(f"   - Std resonance: {std_resonance:.6f}")
        tprint_info(f"   - Max resonance: {max_resonance:.6f}")
        tprint_info(f"   - Min resonance: {min_resonance:.6f}")
        tprint_info(f"   - Zero resonance count: {sum(1 for v in resonance_values if abs(v) < 1e-10)}")
        
        # Check for degenerate case
        if max_resonance < 1e-6:
            tprint_error("❌ CRITICAL: All resonance values effectively zero!")
            tprint_error("   Possible causes:")
            tprint_error("   1. Spectral components are constant/flat")
            tprint_error("   2. No variance in specialist signals")
            tprint_error("   3. Coherence calculation failed")
            tprint_error("   4. Phase alignment issues")
            
            # Log sample spectral component stats if available
            if hasattr(self, '_last_spectral_components'):
                tprint_error("   Spectral component statistics:")
                for name, component in list(self._last_spectral_components.items())[:3]:
                    tprint_error(f"     {name}: mean={np.mean(component):.6f}, std={np.std(component):.6f}")
        
        # Log coherence and phase summaries if provided
        if coherence_summary:
            tprint_info(f"📈 Coherence Summary:")
            for key, value in coherence_summary.items():
                tprint_info(f"   - {key}: {value:.4f}")
                
        if phase_summary:
            tprint_info(f"🔄 Phase Summary:")
            for key, value in phase_summary.items():
                tprint_info(f"   - {key}: {value:.4f}")
        
        # Alert on zero resonance
        if mean_resonance < 1e-6:
            tprint_warning("⚠️  Resonance effectively zero - forcing NEUTRAL_REGIME")
            tprint_warning("   This indicates specialists are not providing meaningful signals")
            tprint_warning("   Consider:")
            tprint_warning("   - Checking specialist feature engineering")
            tprint_warning("   - Verifying input data quality")
            tprint_warning("   - Adjusting spectral decomposition parameters")
    
    def clear_cache(self):
        """Clear all caches to free memory."""
        self._coherence_cache.clear()
        self._phase_cache.clear()
        self._resonance_cache.clear()
        if self.verbose:
            tprint_info("🧹 Cleared all resonance caches")

    @staticmethod
    def _align_lengths(*arrays: np.ndarray) -> List[np.ndarray]:
        """Trim all arrays to the minimum common length (preserving tail alignment)."""
        lengths = [len(arr) for arr in arrays if arr is not None]
        if not lengths:
            return [np.array([]) for _ in arrays]
        min_len = min(lengths)
        if min_len <= 0:
            return [np.array([]) for _ in arrays]
        aligned = []
        for arr in arrays:
            if arr is None or len(arr) < min_len:
                aligned.append(np.array([]))
            else:
                aligned.append(arr[-min_len:])
        return aligned

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'coherence_cache_size': len(self._coherence_cache),
            'phase_cache_size': len(self._phase_cache),
            'resonance_cache_size': len(self._resonance_cache),
            'total_cache_size': len(self._coherence_cache) + len(self._phase_cache) + len(self._resonance_cache)
        }
    
    def compute_rsv_eigenvalue(
        self,
        spectral_components: Dict[str, np.ndarray],
        specialist_names: List[str]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute Resonance State Vector (RSV) eigenvalue.
        
        Args:
            spectral_components: Spectral components for all specialists
            specialist_names: List of specialist names
            
        Returns:
            Tuple of (eigenvalue, resonance_state_info)
        """
        try:
            if self.verbose:
                tprint_info("🔢 Computing Resonance State Vector (RSV) eigenvalue...")
            
            n_specialists = len(specialist_names)
            if n_specialists < 2:
                return 0.0, {'error': 'Insufficient specialists for RSV'}
            
            # Create RSV correlation matrix
            rsv_matrix = np.zeros((n_specialists, n_specialists))
            
            # Calculate pairwise resonance
            total_resonance = 0.0
            pair_count = 0
            
            for i, specialist_i in enumerate(specialist_names):
                for j, specialist_j in enumerate(specialist_names):
                    if i != j:
                        # Use d2-d4 resonance as representative
                        resonance_key_i = f'{specialist_i}_d2_d4_resonance'
                        resonance_key_j = f'{specialist_j}_d2_d4_resonance'
                        
                        # Extract resonance scores
                        resonance_i = self._extract_resonance_component(
                            spectral_components, specialist_i, 'd2', 'd4'
                        )
                        resonance_j = self._extract_resonance_component(
                            spectral_components, specialist_j, 'd2', 'd4'
                        )

                        resonance_i, resonance_j = self._align_lengths(resonance_i, resonance_j)
                        if len(resonance_i) == 0 or len(resonance_j) == 0:
                            continue
                        
                        # Use correlation as proxy for resonance alignment
                        try:
                            # Robust correlation with constant signal check
                            if np.std(resonance_i) < 1e-9 or np.std(resonance_j) < 1e-9:
                                corr = 0.0
                            else:
                                corr, _ = pearsonr(resonance_i, resonance_j)
                            
                            rsv_matrix[i, j] = abs(corr)
                            # Track resonance magnitude (using mean of both signals)
                            avg_res = (np.mean(resonance_i) + np.mean(resonance_j)) / 2.0
                            total_resonance += avg_res
                            pair_count += 1
                        except Exception:
                            rsv_matrix[i, j] = 0.0

            rsv_matrix[np.diag_indices_from(rsv_matrix)] = 1.0
            
            # --- SAFETY GATE: Check for resonance validity ---
            mean_resonance_magnitude = total_resonance / pair_count if pair_count > 0 else 0.0
            
            if self.verbose:
                tprint_info("      - Mean resonance magnitude: {:.4f}".format(mean_resonance_magnitude))
             
            # SAFETY GATE: If resonance is effectively zero or broken, force NEUTRAL REGIME
            # This prevents computing meaningless eigenvalues from noise
            if mean_resonance_magnitude < 1e-6:
                if self.verbose:
                    tprint_warning("      ⚠️ CRITICAL: Resonance effectively zero. Aborting RSV computation.")
                    tprint_warning("      ⚠️ Forcing NEUTRAL_REGIME (Size=0) to prevent unsafe exposure.")
                
                # Return forced NEUTRAL state
                return 0.0, { 
                    'rsv_matrix': np.zeros((len(specialist_names), len(specialist_names))).tolist(),
                    'regime': 'NEUTRAL_REGIME',
                    'mean_resonance': 0.0,
                    'forced_safety': True,
                    'eigenvalues': [],
                    'specialist_names': specialist_names
                }
            
            # Original logic continues if safe...
            if mean_resonance_magnitude < 0.05:
                if self.verbose:
                    tprint_warning("      ⚠️ CRITICAL: Resonance near zero. Forcing NOISE REGIME to prevent unsafe exposure.")
                # Return low eigenvalue (< 1.0) to trigger NOISE_REGIME
                # We return a dummy structure that mimics the expected return format but forces safety
                return 0.1, {
                    'rsv_matrix': rsv_matrix.tolist(),
                    'regime': 'NOISE_REGIME',
                    'mean_resonance': mean_resonance_magnitude,
                    'forced_safety': True,
                    'eigenvalues': [0.1] * len(specialist_names), # Dummy eigenvalues
                    'specialist_names': specialist_names
                }
            # -------------------------------------------------
            
            # Calculate eigenvalues
            eigenvalues = np.linalg.eigvals(rsv_matrix)
            max_eigenvalue = np.max(eigenvalues)
            
            # Classify resonance regime
            resonance_regime = self._classify_resonance_regime(max_eigenvalue)
            
            resonance_state_info = {
                'eigenvalue': max_eigenvalue,
                'eigenvalues': eigenvalues.tolist(),
                'rsv_matrix': rsv_matrix.tolist(),
                'resonance_regime': resonance_regime,
                'specialist_names': specialist_names,
                 'mean_resonance': mean_resonance_magnitude,
                'forced_safety': False
            }
            
            if self.verbose:
                tprint_success(f"   ✅ RSV eigenvalue: {max_eigenvalue:.3f}")
                tprint_info(f"      - Resonance regime: {resonance_regime}")
                tprint_info(f"      - Matrix condition: {np.linalg.cond(rsv_matrix):.2f}")
            
            return max_eigenvalue, resonance_state_info
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ RSV eigenvalue calculation failed: {e}")
            return 0.0, {'error': str(e)}
    
    def _extract_resonance_component(
        self,
        spectral_components: Dict[str, np.ndarray],
        specialist_name: str,
        fast_scale: str,
        slow_scale: str
    ) -> np.ndarray:
        """Extract resonance component for RSV calculation."""
        if self.verbose:
            tprint_info("📊 Extracting resonance component")
        try:
            fast_key = f'{specialist_name}_{fast_scale}'
            slow_key = f'{specialist_name}_{slow_scale}'
            
            if fast_key in spectral_components and slow_key in spectral_components:
                # Use normalized coefficients as resonance component
                fast_coeffs = spectral_components[fast_key]
                slow_coeffs = spectral_components[slow_key]
                fast_coeffs, slow_coeffs = self._align_lengths(fast_coeffs, slow_coeffs)
                if len(fast_coeffs) == 0 or len(slow_coeffs) == 0:
                    return np.array([])
                
                # Simple resonance component: product of normalized coefficients
                fast_norm = (fast_coeffs - np.mean(fast_coeffs)) / (np.std(fast_coeffs) + 1e-9)
                slow_norm = (slow_coeffs - np.mean(slow_coeffs)) / (np.std(slow_coeffs) + 1e-9)
                
                resonance_component = fast_norm * slow_norm
                return resonance_component
            
            return np.array([])
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"      ⚠️ Resonance component extraction failed: {e}")
            return np.array([])
    
    def _classify_resonance_regime(self, eigenvalue: float) -> str:
        """Classify resonance regime based on eigenvalue."""
        if self.verbose:
            tprint_info("🏷️ Classifying resonance regime")
        if eigenvalue > 0.8:
            return "HIGH_GLOBAL_RESONANCE"
        elif eigenvalue > 0.5:
            return "MODERATE_RESONANCE"
        elif eigenvalue > 0.3:
            return "LOW_RESONANCE"
        else:
            return "NOISE_REGIME"
    
    def get_position_sizing_guidance(self, eigenvalue: float) -> Dict[str, Any]:
        """
        Get position sizing guidance based on RSV eigenvalue.
        
        Args:
            eigenvalue: RSV eigenvalue
            
        Returns:
            Position sizing guidance
        """
        if self.verbose:
            tprint_info("📏 Getting position sizing guidance")
        regime = self._classify_resonance_regime(eigenvalue)
        
        guidance = {
            'resonance_regime': regime,
            'eigenvalue': eigenvalue,
            'recommended_position_size': 0.10,  # Default 10%
            'leverage_multiplier': 1.0,        # Default 1x
            'confidence_level': 0.5
        }
        
        if regime == "HIGH_GLOBAL_RESONANCE":
            guidance.update({
                'recommended_position_size': 0.20,  # 20% (full 10x leverage)
                'leverage_multiplier': 2.0,
                'confidence_level': 0.8,
                'rationale': 'All specialists vibrating in sync - structural regime shift'
            })
        elif regime == "MODERATE_RESONANCE":
            guidance.update({
                'recommended_position_size': 0.15,  # 15% (7.5x leverage)
                'leverage_multiplier': 1.5,
                'confidence_level': 0.65,
                'rationale': 'Moderate resonance across specialists'
            })
        elif regime == "LOW_RESONANCE":
            guidance.update({
                'recommended_position_size': 0.10,  # 10% (5x leverage)
                'leverage_multiplier': 1.0,
                'confidence_level': 0.5,
                'rationale': 'Low resonance - normal market conditions'
            })
        else:  # NOISE_REGIME
            guidance.update({
                'recommended_position_size': 0.05,  # 5% (2.5x leverage)
                'leverage_multiplier': 0.5,
                'confidence_level': 0.3,
                'rationale': 'Specialists discordant - noise regime, reduce position'
            })
        
        return guidance


# Convenience functions for quick usage
def quick_resonance_analysis(
    spectral_components: Dict[str, np.ndarray],
    specialist_names: List[str],
    verbose: bool = True
) -> Dict[str, Any]:
    """Quick resonance analysis for spectral components."""
    if verbose:
        tprint_info("🚀 Quick resonance analysis")
    detector = ResonanceDetector(verbose=verbose)
    
    # Compute all resonances
    all_resonances = detector.compute_all_resonances(spectral_components)
    
    # Compute RSV eigenvalue
    eigenvalue, rsv_info = detector.compute_rsv_eigenvalue(spectral_components, specialist_names)
    
    # Get position sizing guidance
    guidance = detector.get_position_sizing_guidance(eigenvalue)
    
    return {
        'resonance_scores': all_resonances,
        'rsv_eigenvalue': eigenvalue,
        'rsv_info': rsv_info,
        'position_sizing_guidance': guidance
    }


if __name__ == "__main__":
    # Example usage
    print("Resonance Detector for AEDL")
    print("Use quick_resonance_analysis() for quick usage")
    
    # Display micro-macro pairs
    detector = ResonanceDetector()
    print("\nMicro-Macro Scale Pairs:")
    for fast, slow in detector.micro_macro_pairs:
        print(f"  {fast} <-> {slow}")
