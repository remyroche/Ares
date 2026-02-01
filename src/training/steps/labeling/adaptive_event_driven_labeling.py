"""
Adaptive Event-Driven Labeling (AEDL) Framework

Core framework implementation for frequency-dependent labeling that replaces
static Triple Barrier Method with dynamic wavelet-based analysis.

Key Features:
- Frequency-dependent labeling instead of static barriers
- Cross-scale resonance detection for harmonic entries
- Structural breakout identification through phase synchronization
- Integration with modern De Prado causal framework
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Import AEDL components
from .wavelet_decomposition import WaveletDecomposition
from .optimized_wavelet_decomposition import OptimizedWaveletDecomposition
from .spectral_specialists import SpectralSpecialists
from .resonance_detector import ResonanceDetector
from .causal_compression import CausalCompression

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    # Fallback print functions
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")


class AdaptiveEventDrivenLabeling:
    """
    Adaptive Event-Driven Labeling (AEDL) framework.
    
    Replaces static Triple Barrier Method with frequency-dependent analysis
    using wavelet decomposition and cross-scale resonance detection.
    """
    
    # Class-level configuration constants
    DEFAULT_COVERAGE_PERCENT = 3.0  # Default coverage percentage for structural breakouts
    
    def __init__(
        self,
        causal_graph: Dict[str, List[str]] = None,
        wavelet_params: Dict[str, Any] = None,
        resonance_params: Dict[str, Any] = None,
        compression_params: Dict[str, Any] = None,
        verbose: bool = True
    ):
        """
        Initialize AEDL framework.
        
        Args:
            causal_graph: DAG for causal parent filtering
            wavelet_params: Parameters for wavelet decomposition
            resonance_params: Parameters for resonance detection
            compression_params: Parameters for causal compression
            verbose: Whether to print progress information
        """
        self.verbose = verbose
        self.causal_graph = causal_graph or {}
        
        # Initialize components
        self.wavelet_engine = OptimizedWaveletDecomposition(
            **(wavelet_params or {}),
            verbose=verbose
        )
        
        self.spectral_specialists = SpectralSpecialists(
            verbose=verbose
        )
        
        self.resonance_detector = ResonanceDetector(
            **(resonance_params or {}),
            verbose=verbose
        )
        
        self.compression_engine = CausalCompression(
            causal_graph=self.causal_graph,
            **(compression_params or {}),
            verbose=verbose
        )
        
        # Results storage
        self.aedl_results = {}
        self.spectral_components = {}
        self.resonance_analysis = {}
        self.compression_metrics = {}
        
        if self.verbose:
            tprint_info("🚀 Adaptive Event-Driven Labeling: Initializing...")
            tprint_info(f"   ⚙️ Wavelet engine: MODWT with 5 scales")
            tprint_info(f"   ⚙️ Spectral specialists: 4 priority specialists")
            tprint_info(f"   ⚙️ Resonance detector: Phase synchronization enabled")
            tprint_info(f"   ⚙️ Causal compression: 20 → 4 features")
            tprint_success("   ✅ AEDL Framework: Initialization complete")
    
    def process_market_data(
        self,
        df: pd.DataFrame,
        causal_anchor_predictions: np.ndarray = None,
        specialist_configs: Dict[str, Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process market data through complete AEDL pipeline.
        
        Args:
            df: Market data with OHLCV and derived features
            causal_anchor_predictions: Causal anchor model predictions
            specialist_configs: Configuration for specialist extraction
            
        Returns:
            Dictionary with AEDL results
        """
        try:
            if self.verbose:
                tprint_info("🎯 AEDL Pipeline: Starting frequency-dependent analysis...")
            
            pipeline_start_time = pd.Timestamp.now()
            
            # Step 1: Extract specialist signals
            if self.verbose:
                tprint_info("   📊 Step 1: Extracting specialist signals...")
            
            specialist_signals = self.spectral_specialists.extract_specialist_signals(
                df, specialist_configs
            )
            
            if not specialist_signals:
                raise ValueError("No specialist signals extracted")
            
            # Validate specialist signals
            validation_results = self.spectral_specialists.validate_specialist_signals(
                specialist_signals
            )
            
            # Step 2: Wavelet decomposition
            if self.verbose:
                tprint_info("   🌊 Step 2: Wavelet decomposition (5 scales)...")
            
            self.spectral_components = self.spectral_specialists.transform_to_spectral(
                specialist_signals, self.wavelet_engine
            )
            
            if not self.spectral_components:
                raise ValueError("Wavelet decomposition failed")
            
            # Step 3: Resonance analysis
            if self.verbose:
                tprint_info("   🔍 Step 3: Cross-scale resonance detection...")
            
            self.resonance_analysis = self.resonance_detector.compute_all_resonances(
                self.spectral_components
            )
            
            # Calculate RSV eigenvalue
            specialist_names = list(specialist_signals.keys())
            rsv_eigenvalue, rsv_info = self.resonance_detector.compute_rsv_eigenvalue(
                self.spectral_components, specialist_names
            )
            
            # Get position sizing guidance
            position_guidance = self.resonance_detector.get_position_sizing_guidance(
                rsv_eigenvalue
            )
            
            # Step 4: Causal compression (if causal anchor available)
            alpha_features = {}
            if causal_anchor_predictions is not None:
                if self.verbose:
                    tprint_info("   🔬 Step 4: Causal compression (20 → 4 alpha features)...")
                
                # ALIGNMENT FIX: Ensure causal_anchor_predictions matches spectral components shape
                # 1. Get reference index from spectral components
                ref_indices = [comp.index for comp in self.spectral_components.values() if hasattr(comp, 'index')]
                if not ref_indices:
                     # Fallback to signals
                    ref_indices = [sig.index for sig in specialist_signals.values() if hasattr(sig, 'index')]
                
                if ref_indices:
                    # Use the first valid index as reference (all should be aligned)
                    target_index = ref_indices[0]
                    
                    # 2. Check alignment
                    if len(causal_anchor_predictions) != len(target_index):
                         if self.verbose:
                             tprint_warning(f"   ⚠️ Aligning anchor ({len(causal_anchor_predictions)}) to specialists ({len(target_index)})")
                         
                         # Create Series with ORIGINAL df index to perform safe reindexing
                         # assumption: causal_anchor_predictions aligns with input df
                         if len(causal_anchor_predictions) == len(df):
                             anchor_series = pd.Series(causal_anchor_predictions, index=df.index)
                             
                             # Ensure index is unique for reindex to succeed
                             # If we have multiple assets, we should have already handled this at a higher level,
                             # but here we must be safe.
                             if anchor_series.index.has_duplicates:
                                 anchor_series = anchor_series.loc[~anchor_series.index.duplicated(keep='last')]
                             
                             safe_target = target_index
                             if target_index.has_duplicates:
                                 safe_target = target_index.drop_duplicates(keep='last')
                                 
                             aligned_anchor = anchor_series.reindex(safe_target).fillna(0).values
                             causal_anchor_predictions = aligned_anchor
                         else:
                             # Fallback: simple truncation/padding if we can't match index
                             # This happens if lengths are totally mismatched
                             tprint_warning("   ⚠️ Index mismatch prevents safe alignment. Using truncation/padding.")
                             if len(causal_anchor_predictions) > len(target_index):
                                 causal_anchor_predictions = causal_anchor_predictions[-len(target_index):]
                             else:
                                 new_anc = np.zeros(len(target_index))
                                 new_anc[:len(causal_anchor_predictions)] = causal_anchor_predictions
                                 causal_anchor_predictions = new_anc

                alpha_features, self.compression_metrics = \
                    self.compression_engine.compress_spectral_features(
                        self.spectral_components, causal_anchor_predictions
                    )
            else:
                if self.verbose:
                    tprint_warning("   ⚠️ Step 4: Skipping compression (no causal anchor predictions)")
            
            # Compile results
            pipeline_time = (pd.Timestamp.now() - pipeline_start_time).total_seconds()
            
            self.aedl_results = {
                'specialist_signals': specialist_signals,
                'validation_results': validation_results,
                'spectral_components': self.spectral_components,
                'resonance_scores': self.resonance_analysis,
                'rsv_eigenvalue': rsv_eigenvalue,
                'rsv_info': rsv_info,
                'position_sizing_guidance': position_guidance,
                'alpha_features': alpha_features,
                'compression_metrics': self.compression_metrics,
                'pipeline_metrics': {
                    'total_time': pipeline_time,
                    'specialists_processed': len(specialist_signals),
                    'spectral_components': len(self.spectral_components),
                    'resonance_scores': len(self.resonance_analysis),
                    'alpha_features': len(alpha_features)
                }
            }
            
            if self.verbose:
                tprint_success("✅ AEDL Pipeline Complete:")
                tprint_info(f"   - Specialists: {len(specialist_signals)}")
                tprint_info(f"   - Spectral components: {len(self.spectral_components)}")
                tprint_info(f"   - Resonance scores: {len(self.resonance_analysis)}")
                tprint_info(f"   - RSV eigenvalue: {rsv_eigenvalue:.3f}")
                tprint_info(f"   - Alpha features: {len(alpha_features)}")
                tprint_info(f"   - Position regime: {position_guidance['resonance_regime']}")
                tprint_info(f"   - Pipeline time: {pipeline_time:.3f}s")
            
            return self.aedl_results
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ AEDL pipeline failed: {e}")
            return {'error': str(e)}
    
    def get_harmonic_entries(
        self,
        resonance_threshold: float = 0.7,
        min_rsv_eigenvalue: float = 0.5
    ) -> Dict[str, Any]:
        """
        Identify harmonic entry points based on resonance analysis.
        
        Args:
            resonance_threshold: Minimum resonance score for entry
            min_rsv_eigenvalue: Minimum RSV eigenvalue for entry
            
        Returns:
            Dictionary with harmonic entry signals
        """
        try:
            if not self.resonance_analysis:
                if self.verbose:
                    tprint_warning("   ⚠️ No resonance analysis available for harmonic entries")
                return {'error': 'No resonance analysis available', 'entry_quality': 0.0}
            
            if self.verbose:
                tprint_info("🎯 Identifying harmonic entries...")
            
            # Find high resonance periods
            high_resonance_periods = {}
            
            for resonance_name, resonance_data in self.resonance_analysis.items():
                # Extract series from the dictionary if necessary
                resonance_scores = None
                if isinstance(resonance_data, dict) and 'series' in resonance_data:
                    resonance_scores = resonance_data['series']
                elif isinstance(resonance_data, np.ndarray):
                    resonance_scores = resonance_data

                if isinstance(resonance_scores, np.ndarray):
                    # Identify periods where resonance > threshold
                    high_resonance_mask = resonance_scores > resonance_threshold
                    high_resonance_periods[resonance_name] = high_resonance_mask
            
            # Check RSV eigenvalue threshold
            rsv_eigenvalue = self.aedl_results.get('rsv_eigenvalue', 0.0)
            rsv_valid = rsv_eigenvalue >= min_rsv_eigenvalue
            
            # Combine signals
            harmonic_entries = {
                'high_resonance_periods': high_resonance_periods,
                'rsv_eigenvalue': rsv_eigenvalue,
                'rsv_valid': rsv_valid,
                'entry_signal': rsv_valid and len(high_resonance_periods) > 0,
                'position_sizing_guidance': self.aedl_results.get('position_sizing_guidance', {}),
                'entry_quality': self._calculate_entry_quality(high_resonance_periods, rsv_eigenvalue)
            }
            
            if self.verbose:
                tprint_success(f"   ✅ Harmonic entries identified:")
                tprint_info(f"      - High resonance periods: {len(high_resonance_periods)}")
                tprint_info(f"      - RSV eigenvalue: {rsv_eigenvalue:.3f}")
                tprint_info(f"      - Entry signal: {harmonic_entries['entry_signal']}")
                tprint_info(f"      - Entry quality: {harmonic_entries['entry_quality']:.3f}")
            
            return harmonic_entries
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Harmonic entry identification failed: {e}")
            return {'error': str(e)}
    
    def _calculate_entry_quality(
        self,
        high_resonance_periods: Dict[str, np.ndarray],
        rsv_eigenvalue: float
    ) -> float:
        """Calculate overall entry quality score."""
        try:
            # Removed early return so RSV eigenvalue can contribute even if no high resonance periods
            # if not high_resonance_periods:
            #     return 0.0
            
            # Average resonance strength
            resonance_strengths = []
            for resonance_name, resonance_mask in high_resonance_periods.items():
                if isinstance(resonance_mask, np.ndarray):
                    # Extract series from the dictionary if necessary
                    resonance_data = self.resonance_analysis.get(resonance_name)
                    resonance_scores = None

                    if isinstance(resonance_data, dict) and 'series' in resonance_data:
                        resonance_scores = resonance_data['series']
                    elif isinstance(resonance_data, np.ndarray):
                        resonance_scores = resonance_data

                    if isinstance(resonance_scores, np.ndarray) and len(resonance_scores) > 0:
                        high_scores = resonance_scores[resonance_mask]
                        if len(high_scores) > 0:
                            resonance_strengths.append(np.mean(high_scores))
            
            avg_resonance = np.mean(resonance_strengths) if resonance_strengths else 0.0
            
            # Combine resonance strength with RSV eigenvalue
            entry_quality = 0.6 * avg_resonance + 0.4 * rsv_eigenvalue
            
            return entry_quality
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"      ⚠️ Entry quality calculation failed: {e}")
            return 0.0
    
    def get_structural_breakouts(
        self,
        phase_threshold: float = None,
        use_quantile_approach: bool = True,
        min_coverage_percent: float = None,
        volatility_window: int = 500
    ) -> Dict[str, Any]:
        """
        Identify structural breakouts based on phase lead-lag analysis.
        
        Args:
            phase_threshold: Phase difference threshold for breakout detection (ignored if use_quantile_approach=True)
            use_quantile_approach: Whether to use 2% quantile approach for guaranteed coverage
            min_coverage_percent: Minimum percentage of time periods to identify as breakouts
            volatility_window: Window size for rolling volatility barrier (default: 500)
            
        Returns:
            Dictionary with structural breakout signals
        """
        try:
            if not self.spectral_components:
                return {'error': 'No spectral components available'}
            
            # Use default coverage if not specified
            if min_coverage_percent is None:
                min_coverage_percent = self.DEFAULT_COVERAGE_PERCENT
            
            if self.verbose:
                tprint_info("💥 Identifying structural breakouts...")
            
            # Analyze phase lead-lag relationships
            breakout_signals = {}
            diagnostics: Dict[str, Dict[str, Any]] = {}
            specialist_breakout_masks: Dict[str, np.ndarray] = {}
            
            # Collect all z-score series for global diagnostics
            all_z_values = []
            z_scores_by_specialist = {}
            
            # Get all available specialist names from spectral components
            specialist_names = []
            for key in self.spectral_components.keys():
                if key.endswith('_d1'):  # Extract base name from d1 component
                    base_name = key[:-3]  # Remove '_d1' suffix
                    if base_name not in specialist_names:
                        specialist_names.append(base_name)
            
            for specialist_name in specialist_names:
                # Check d1-d3 phase relationship (micro leading macro = breakout)
                d1_key = f'{specialist_name}_d1'
                d3_key = f'{specialist_name}_d3'
                
                if d1_key in self.spectral_components and d3_key in self.spectral_components:
                    d1_coeffs = self.spectral_components[d1_key]
                    d3_coeffs = self.spectral_components[d3_key]
                    var_d1 = float(np.var(d1_coeffs)) if len(d1_coeffs) else 0.0
                    var_d3 = float(np.var(d3_coeffs)) if len(d3_coeffs) else 0.0
                    
                    phase_output = self.resonance_detector.calculate_phase_lead_lag(
                        d1_coeffs,
                        d3_coeffs
                    )
                    
                    # Support both (summary, series) tuples and legacy scalar responses
                    phase_summary: float
                    phase_series: Optional[np.ndarray]
                    
                    if isinstance(phase_output, tuple):
                        phase_summary, phase_series = phase_output
                    elif isinstance(phase_output, np.ndarray):
                        phase_series = phase_output
                        phase_summary = float(np.nanmean(phase_series))
                    else:
                        phase_summary = float(phase_output)
                        phase_series = None
                    
                    if phase_series is None or not isinstance(phase_series, np.ndarray):
                        phase_series = np.full_like(self.spectral_components[d1_key], phase_summary, dtype=float)
                    
                    # Calculate Rolling Z-Score of Phase
                    # Z = (Phase - RollingMean) / RollingStd
                    phase_s = pd.Series(phase_series)
                    rolling_mean = phase_s.rolling(window=volatility_window, min_periods=1).mean()
                    rolling_std = phase_s.rolling(window=volatility_window, min_periods=1).std().fillna(1e-9) # Avoid div/0

                    # Apply small epsilon to std to avoid division by zero
                    rolling_std = rolling_std.replace(0, 1e-9)

                    z_series = (phase_s - rolling_mean) / rolling_std
                    z_series_values = z_series.values

                    # Store z-series for thresholding
                    z_scores_by_specialist[specialist_name] = z_series_values
                    all_z_values.extend(z_series_values[~np.isnan(z_series_values)].tolist())

            # Calculate global quantile threshold for Z-scores if enabled
            # Note: We use per-specialist quantiles below, but can compute global stats here
            global_z_threshold = 2.0 # Default
            if use_quantile_approach and all_z_values:
                global_z_threshold = np.percentile(all_z_values, 100 - min_coverage_percent)
                
                if self.verbose:
                    tprint_info(f"   🎯 Global Z-Score Analysis ({min_coverage_percent}% target)")
                    tprint_info(f"      - Total values: {len(all_z_values):,}")
                    tprint_info(f"      - Global Z-Threshold: {global_z_threshold:.4f}")
                    tprint_info(f"      - Z-Score Range: [{np.min(all_z_values):.2f}, {np.max(all_z_values):.2f}]")

            else:
                # Fallback fixed Z-threshold (approx 2 sigma)
                global_z_threshold = 2.0
                if self.verbose:
                    tprint_info(f"   📐 Using fixed Z-threshold: {global_z_threshold:.2f}")
            
            # Process each specialist
            for specialist_name in specialist_names:
                if specialist_name not in z_scores_by_specialist:
                    continue
                    
                z_series = z_scores_by_specialist[specialist_name]

                # Retrieve original phase components for diagnostics
                d1_key = f'{specialist_name}_d1'
                d3_key = f'{specialist_name}_d3'

                # Calculate basic phase statistics for logging
                # We need to re-derive phase summary since we are in a new loop
                # This ensures we don't use stale variables from the previous loop
                phase_summary = 0.0
                if d1_key in self.spectral_components and d3_key in self.spectral_components:
                    d1_coeffs = self.spectral_components[d1_key]
                    d3_coeffs = self.spectral_components[d3_key]

                    # Quick phase check (re-using scalar calculation for speed if possible)
                    # For logging purposes, simple mean of Z-score is not phase summary.
                    # We can use the mean of the phase series if we stored it, or re-calc.
                    # Given Z-scores were derived from phase series, we can try to recover or just skip.
                    # Let's re-calculate simple phase summary using the detector if needed,
                    # OR just report Z-score summary. Reporting Z-score mean is more relevant here.
                    z_mean = float(np.nanmean(z_series))
                    phase_summary = z_mean # Use Z-mean as the summary metric in this context

                # Per-specialist Z-score threshold
                specialist_z_threshold = global_z_threshold
                if use_quantile_approach:
                    z_values = z_series[~np.isnan(z_series)]
                    if len(z_values) > 0:
                        specialist_z_threshold = np.percentile(
                            z_values,
                            100 - min_coverage_percent
                        )

                # Breakout signal: Z-score > Threshold
                valid_mask = ~np.isnan(z_series)
                breakout_mask = np.zeros_like(z_series, dtype=bool)

                if np.any(valid_mask):
                    breakout_mask[valid_mask] = z_series[valid_mask] >= specialist_z_threshold

                specialist_breakout_masks[specialist_name] = breakout_mask

                if np.any(valid_mask):
                    phase_coverage = float(np.mean(breakout_mask[valid_mask].astype(float)))
                else:
                    phase_coverage = 0.0
                
                # Debug per-specialist analysis
                if use_quantile_approach and self.verbose:
                    z_valid = z_series[~np.isnan(z_series)]
                    if len(z_valid) > 0:
                        z_above = np.sum(z_valid >= specialist_z_threshold)

                        tprint_info(f"      🔍 {specialist_name} Analysis (Z-Score):")
                        tprint_info(f"         - Z-Threshold: {specialist_z_threshold:.4f}")
                        tprint_info(f"         - Breakouts: {z_above} ({phase_coverage:.2%})")
                        tprint_info(f"         - Z-Range: [{np.min(z_valid):.2f}, {np.max(z_valid):.2f}]")
                
                # Get variance info for diagnostics
                var_d1 = float(np.var(self.spectral_components.get(d1_key, [])))
                var_d3 = float(np.var(self.spectral_components.get(d3_key, [])))
                
                diagnostics[specialist_name] = {
                    "var_d1": var_d1,
                    "var_d3": var_d3,
                    "z_mean": float(phase_summary),
                    "z_threshold": float(specialist_z_threshold),
                    "phase_coverage": phase_coverage,
                    "sample_count": int(len(z_series)),
                    "quantile_used": use_quantile_approach,
                    "min_coverage_percent": min_coverage_percent
                }
                
                if self.verbose:
                    method_str = f"{min_coverage_percent}% quantile" if use_quantile_approach else f"fixed"
                    tprint_info(
                        f"      • {specialist_name}: var(d1)={var_d1:.3e}, var(d3)={var_d3:.3e}, "
                        f"z_mean={phase_summary:.3f}, coverage>{specialist_z_threshold:.3f}={phase_coverage:.2%} ({method_str})"
                    )
                
                # Only register actual breakouts
                if np.any(breakout_mask):
                    breakout_signals[f'{specialist_name}_d1_d3_breakout'] = breakout_mask
            
            # Verify individual specialist coverage
            if self.verbose and use_quantile_approach:
                tprint_info(f"      📈 Individual Specialist Coverage:")
                for specialist_name in specialist_names:
                    if specialist_name in z_scores_by_specialist:
                        z_series = z_scores_by_specialist[specialist_name]
                        breakout_mask = specialist_breakout_masks.get(specialist_name)
                        if breakout_mask is None:
                            valid_mask = ~np.isnan(z_series)
                            if np.any(valid_mask):
                                individual_coverage = float(
                                    np.mean(z_series[valid_mask] >= np.percentile(
                                        z_series[valid_mask],
                                        100 - min_coverage_percent
                                    ))
                                )
                            else:
                                individual_coverage = 0.0
                        else:
                            valid_mask = ~np.isnan(z_series)
                            if np.any(valid_mask):
                                individual_coverage = float(
                                    np.mean(breakout_mask[valid_mask].astype(float))
                                )
                            else:
                                individual_coverage = 0.0
                        tprint_info(f"         - {specialist_name}: {individual_coverage:.2%} coverage")
            
            # Combine breakout signals
            structural_breakouts = {
                'breakout_signals': breakout_signals,
                'breakout_periods': len(breakout_signals),
                'dominant_breakout_specialist': self._find_dominant_breakout(breakout_signals),
                'breakout_strength': self._calculate_breakout_strength(breakout_signals),
                'diagnostics': diagnostics
            }
            self.structural_breakout_diagnostics = diagnostics
            
            if self.verbose:
                method_desc = f"{min_coverage_percent}% quantile" if use_quantile_approach else f"fixed threshold ({quantile_threshold:.3f})"
                tprint_success(f"   ✅ Structural breakouts identified ({method_desc}):")
                tprint_info(f"      - Breakout signals: {len(breakout_signals)}")
                tprint_info(f"      - Dominant specialist: {structural_breakouts['dominant_breakout_specialist']}")
                tprint_info(f"      - Breakout strength: {structural_breakouts['breakout_strength']:.3f}")
                if use_quantile_approach:
                    total_periods = sum(len(mask) for mask in breakout_signals.values())
                    total_breakouts = sum(np.sum(mask) for mask in breakout_signals.values())
                    actual_coverage = (total_breakouts / total_periods * 100) if total_periods > 0 else 0
                    tprint_info(f"      - Actual coverage: {actual_coverage:.1f}% (target: {min_coverage_percent}%)")
            
            return structural_breakouts
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Structural breakout identification failed: {e}")
            return {'error': str(e)}
    
    def get_structural_breakouts_3percent(self) -> Dict[str, Any]:
        """
        Convenience method to get structural breakouts with guaranteed 3% coverage.
        
        Returns:
            Dictionary with structural breakout signals using 3% quantile approach
        """
        return self.get_structural_breakouts(
            use_quantile_approach=True,
            min_coverage_percent=self.DEFAULT_COVERAGE_PERCENT
        )
    
    def _find_dominant_breakout(self, breakout_signals: Dict[str, np.ndarray]) -> str:
        """Find the specialist with strongest breakout signal."""
        if self.verbose:
            tprint_info("🔥 Finding dominant breakout")
        try:
            breakout_strengths = {}
            
            for signal_name, breakout_mask in breakout_signals.items():
                if isinstance(breakout_mask, np.ndarray):
                    strength = np.mean(breakout_mask.astype(float))
                    breakout_strengths[signal_name] = strength
            
            if breakout_strengths:
                return max(breakout_strengths, key=breakout_strengths.get)
            
            return "None"
            
        except Exception:
            return "None"
    
    def _calculate_breakout_strength(self, breakout_signals: Dict[str, np.ndarray]) -> float:
        """Calculate overall breakout strength."""
        if self.verbose:
            tprint_info("💪 Calculating breakout strength")
        try:
            if not breakout_signals:
                return 0.0
            
            strengths = []
            for breakout_mask in breakout_signals.values():
                if isinstance(breakout_mask, np.ndarray):
                    strength = np.mean(breakout_mask.astype(float))
                    strengths.append(strength)
            
            return np.mean(strengths) if strengths else 0.0
            
        except Exception:
            return 0.0
    
    def generate_aedl_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive AEDL analysis report.
        
        Returns:
            Dictionary with complete AEDL analysis
        """
        try:
            if not self.aedl_results:
                return {'error': 'No AEDL results available'}
            
            report = {
                'aedl_summary': {
                    'pipeline_completed': True,
                    'specialists_processed': len(self.aedl_results.get('specialist_signals', {})),
                    'spectral_components': len(self.spectral_components),
                    'resonance_scores': len(self.resonance_analysis),
                    'rsv_eigenvalue': self.aedl_results.get('rsv_eigenvalue', 0.0),
                    'alpha_features': len(self.aedl_results.get('alpha_features', {})),
                    'pipeline_time': self.aedl_results.get('pipeline_metrics', {}).get('total_time', 0.0)
                },
                'specialist_analysis': self.aedl_results.get('validation_results', {}),
                'resonance_analysis': {
                    'scores': self.resonance_analysis,
                    'rsv_info': self.aedl_results.get('rsv_info', {}),
                    'position_guidance': self.aedl_results.get('position_sizing_guidance', {})
                },
                'compression_analysis': self.compression_metrics,
                'harmonic_entries': self.get_harmonic_entries(),
                'structural_breakouts': self.get_structural_breakouts(),
                'recommendations': self._generate_recommendations()
            }
            
            return report
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Report generation failed: {e}")
            return {'error': str(e)}
    
    def _generate_recommendations(self) -> List[str]:
        """Generate trading recommendations based on AEDL analysis."""
        if self.verbose:
            tprint_info("💡 Generating recommendations")
        try:
            recommendations = []
            
            rsv_eigenvalue = self.aedl_results.get('rsv_eigenvalue', 0.0)
            position_guidance = self.aedl_results.get('position_sizing_guidance', {})
            
            # Position sizing recommendation
            if position_guidance:
                regime = position_guidance.get('resonance_regime', 'UNKNOWN')
                position_size = position_guidance.get('recommended_position_size', 0.10)
                
                if regime == "HIGH_GLOBAL_RESONANCE":
                    recommendations.append(
                        f"STRONG BUY: High global resonance detected. "
                        f"Recommended position size: {position_size:.1%} (full leverage)."
                    )
                elif regime == "MODERATE_RESONANCE":
                    recommendations.append(
                        f"MODERATE BUY: Moderate resonance detected. "
                        f"Recommended position size: {position_size:.1%}."
                    )
                elif regime == "NOISE_REGIME":
                    recommendations.append(
                        f"REDUCE POSITION: Noise regime detected. "
                        f"Recommended position size: {position_size:.1%} (risk reduction)."
                    )
            
            # RSV-based recommendation
            if rsv_eigenvalue > 0.8:
                recommendations.append(
                    f"STRUCTURAL SHIFT: RSV eigenvalue {rsv_eigenvalue:.3f} indicates structural regime shift."
                )
            elif rsv_eigenvalue < 0.3:
                recommendations.append(
                    f"MARKET NOISE: RSV eigenvalue {rsv_eigenvalue:.3f} indicates noise regime."
                )
            
            # Compression quality recommendation
            compression_metrics = self.compression_metrics
            if compression_metrics:
                total_compression = compression_metrics.get('total_compression_ratio', 1.0)
                if total_compression > 4.0:
                    recommendations.append(
                        f"EXCELLENT COMPRESSION: {total_compression:.1f}x compression achieved with minimal signal loss."
                    )
            
            return recommendations
            
        except Exception:
            return ["Unable to generate recommendations due to analysis errors."]


# Convenience functions for quick usage
def quick_aedl_analysis(
    df: pd.DataFrame,
    causal_anchor_predictions: np.ndarray = None,
    causal_graph: Dict[str, List[str]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Quick AEDL analysis for market data."""
    if verbose:
        tprint_info("🚀 Quick AEDL analysis")
    aedl = AdaptiveEventDrivenLabeling(
        causal_graph=causal_graph,
        verbose=verbose
    )
    
    return aedl.process_market_data(df, causal_anchor_predictions)


if __name__ == "__main__":
    # Example usage
    print("Adaptive Event-Driven Labeling (AEDL) Framework")
    print("Use quick_aedl_analysis() for quick usage")
    
    print("\nAEDL Pipeline:")
    print("1. Extract specialist signals (4 priority specialists)")
    print("2. Wavelet decomposition (5 scales per specialist)")
    print("3. Cross-scale resonance detection")
    print("4. Causal compression (20 → 4 alpha features)")
    print("5. Harmonic entry and structural breakout identification")
