"""
Causal Surprise Events Module

Implements causal surprise event detection based on specialist prediction errors
and structural breaks in causal relationships.

Key Features:
1. Causal surprise detection from specialist prediction errors
2. Structural break detection in causal relationships
3. Event generation and scoring
4. Integration with existing event systems
"""

import numpy as np
import pandas as pd
from src.utils.numba_funcs import _numba_rolling_mad
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import stats
from sklearn.preprocessing import StandardScaler
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

class CausalSurpriseDetector:
    """
    Detects causal surprise events based on specialist prediction errors.
    
    Causal surprise occurs when specialist models make large prediction errors,
    indicating potential mechanism breaks or regime changes.
    """
    
    def __init__(
        self,
        surprise_threshold: float = 1.5,
        rolling_window: int = 100,
        min_specialists: int = 2,
        structural_break_window: int = 50,
        verbose: bool = True,
        zone_score_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Causal Surprise Detector.
        
        Args:
            surprise_threshold: Z-score threshold for surprise detection (default 1.8)
            rolling_window: Window for rolling statistics (Adaptive Volatility Filter, default 500)
            min_specialists: Minimum number of specialists required
            structural_break_window: Window for structural break detection
            verbose: Whether to print progress information
        """
        self.surprise_threshold = surprise_threshold
        self.rolling_window = rolling_window
        self.min_specialists = min_specialists
        self.structural_break_window = structural_break_window
        self.verbose = verbose
        
        self.specialist_predictions_ = {}
        self.specialist_errors_ = {}
        self.specialist_metadata_ = {}
        self.surprise_events_ = {}
        self.structural_breaks_ = {}
        self.surprise_density_ = 0.0

        # Zone score configuration and storage
        self.zone_score_config = zone_score_config or {}
        self.zone_score_power = max(1.0, float(self.zone_score_config.get('power', 2.0)))
        self.zone_score_cap = float(self.zone_score_config.get('cap', 0.99))
        self.zone3_floor = float(self.zone_score_config.get('zone3_floor', 0.85))
        self.zone3_ratio_boost = float(self.zone_score_config.get('zone3_ratio_boost', 0.5))
        self.zone2_ratio_boost = float(self.zone_score_config.get('zone2_ratio_boost', 0.2))

        self.specialist_zone_scores_: pd.DataFrame = pd.DataFrame()
        self.specialist_zone_levels_: pd.DataFrame = pd.DataFrame()
        self.surprise_aggregates_df_: pd.DataFrame = pd.DataFrame()
        self.specialist_reliability_: Dict[str, Dict[str, float]] = {}
        self.detector_reliability_: Dict[str, float] = {}
        
    def register_specialist(
        self,
        specialist_name: str,
        predictions: pd.Series,
        targets: pd.Series
    ) -> None:
        """
        Register a specialist model with predictions and targets.
        
        Args:
            specialist_name: Name of the specialist
            predictions: Specialist predictions
            targets: True targets
        """
        try:
            if len(predictions) != len(targets):
                raise ValueError("Predictions and targets must have same length")
            
            # Compute prediction errors
            errors = targets - predictions
            
            # Compute Global MAD for this specialist (Robust baseline)
            median_error = np.median(errors)
            global_mad = np.median(np.abs(errors - median_error))
            
            # Store specialist data
            self.specialist_predictions_[specialist_name] = predictions
            self.specialist_errors_[specialist_name] = errors
            self.specialist_metadata_[specialist_name] = {
                'global_mad': global_mad,
                'mean_error': errors.mean(),
                'std_error': errors.std()
            }
            
            if self.verbose:
                tprint_info(f"📝 Registered specialist: {specialist_name}")
                tprint_info(f"   - Samples: {len(predictions)}")
                tprint_info(f"   - Global MAD: {global_mad:.6f}")
                tprint_info(f"   - Mean error: {errors.mean():.6f}")
                tprint_info(f"   - Error std: {errors.std():.6f}")
                
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Failed to register specialist {specialist_name}: {e}")
            raise

    def adaptive_calibration(self, target_density: float, duration_days: float) -> float:
        """
        Adjust surprise_threshold to target a specific event density.
        
        Args:
            target_density: Target number of events per day
            duration_days: Total duration of the dataset in days
            
        Returns:
            The newly calibrated threshold
        """
        if self.surprise_events_ is None or len(self.surprise_events_) == 0:
            if self.verbose:
                tprint_warning("   ⚠️ Adaptive Calibration: No surprise events aggregated yet.")
            return self.surprise_threshold
            
        # Extract zone_scores if available
        if 'zone_score' in self.surprise_events_.columns:
            scores = self.surprise_events_['zone_score'].values
        else:
            # Fallback to max_surprise if zone_score not yet aggregated
            scores = self.surprise_events_['max_surprise'].values
            
        target_count = max(1, int(target_density * duration_days))
        
        if len(scores) <= target_count:
            self.surprise_threshold = 0.01 # Very loose if we have very little data
        else:
            # Find threshold that gives target_count events
            threshold = np.partition(scores, -target_count)[-target_count]
            self.surprise_threshold = float(threshold)
            
        if self.verbose:
            tprint_info(f"🎯 Adaptive Calibration: Target {target_density:.2f} events/day ({target_count} total)")
            tprint_info(f"   - New threshold: {self.surprise_threshold:.4f}")
            
        return self.surprise_threshold
    
    def compute_soft_surprise(
        self,
        errors: pd.DataFrame = None,
        q: float = 0.975
    ) -> pd.DataFrame:
        """
        Compute continuous surprise scores per specialist in (0,1) range
        using a logistic sigmoid mapping.
        """
        if errors is None:
            errors = self._build_error_frame()
            
        if errors.empty:
            return pd.DataFrame()
            
        # Extract Global MADs for scaling
        mads = pd.Series({
            k: v.get('global_mad', 1.0) 
            for k, v in self.specialist_metadata_.items()
        }).reindex(errors.columns).fillna(1.0)
        
        # Normalize errors by global MAD
        norm_error = errors.abs().divide(mads, axis=1)
        
        # Calculate sigmoid parameters (alpha, mu) based on data distribution or fixed targets
        # User reference: mu = norm_error.quantile(q), alpha = 1.0 / sigma
        mu = norm_error.quantile(q)
        sigma = norm_error.sub(mu).std().replace(0, 1.0)
        alpha = 1.0 / sigma
        
        # Sigmoid mapping: 1 / (1 + exp(-alpha * (x - mu)))
        soft_surprise = 1 / (1 + np.exp(-alpha * (norm_error - mu)))
        return soft_surprise.clip(0.0, 1.0)

    def compute_zone_score(
        self,
        soft_surprise: pd.DataFrame,
        chaos_emphasis: float = 2.0
    ) -> pd.Series:
        """
        Compute a global ZoneScore (0-1 gradient) using reliability-weighted
        aggregation of soft surprise scores.
        """
        if soft_surprise.empty:
            return pd.Series(0.0, index=soft_surprise.index)
            
        # Extract reliability weights
        # If not computed yet, use uniform weights
        reliability = pd.Series({
            k: v.get('composite_reliability', 1.0) 
            for k, v in self.specialist_reliability_.items()
        }).reindex(soft_surprise.columns).fillna(1.0)
        
        weights = reliability / reliability.sum()
        
        # Power weighting for chaos emphasis
        weighted = soft_surprise.pow(chaos_emphasis).mul(weights, axis=1)
        zone_score = weighted.sum(axis=1)
        
        return zone_score.clip(0.0, 1.0)

    def compute_specialist_surprise(
        self,
        specialist_name: str,
        method: str = "zscore"
    ) -> pd.Series:
        """
        Compute surprise scores for a specialist.
        
        Args:
            specialist_name: Name of the specialist
            method: Method for surprise computation ("zscore", "magnitude", "combined")
            
        Returns:
            Surprise scores
        """
        try:
            if specialist_name not in self.specialist_errors_:
                raise ValueError(f"Specialist {specialist_name} not registered")
            
            errors = self.specialist_errors_[specialist_name]
            
            if method == "zscore" or method == "robust_zscore":
                # Robust Z-score based surprise using Median Absolute Deviation (MAD)
                # Surprise = |Y_actual - Y_specialist| / sigma_residual
                
                # Use MAD for robustness as requested
                def get_mad(x):
                    median = np.median(x)
                    return np.median(np.abs(x - median))
                
                rolling_median = errors.rolling(self.rolling_window, min_periods=min(self.rolling_window, 20)).median()
                rolling_mad = errors.rolling(self.rolling_window, min_periods=min(self.rolling_window, 20)).apply(get_mad)
                
                # Standardize: current error / Rolling MAD (robust sigma)
                # Apply floor of Global MAD + absolute unit floor (1.0 for price)
                global_mad = self.specialist_metadata_.get(specialist_name, {}).get('global_mad', 1.0)
                sigma_floor = np.maximum(global_mad, 1.0)
                surprise_scores = np.abs(errors - rolling_median) / (np.maximum(rolling_mad, sigma_floor))
                
            elif method == "magnitude":
                # Magnitude-based surprise
                rolling_mad = errors.rolling(self.rolling_window, min_periods=min(self.rolling_window, 20)).apply(
                    lambda x: np.median(np.abs(x - np.median(x)))
                )
                global_mad = self.specialist_metadata_.get(specialist_name, {}).get('global_mad', 1.0)
                sigma_floor = np.maximum(global_mad, 1.0)
                surprise_scores = np.abs(errors) / (np.maximum(rolling_mad, sigma_floor))
                
            elif method == "combined":
                # Combined robust z-score and magnitude
                rolling_median = errors.rolling(self.rolling_window, min_periods=min(self.rolling_window, 20)).median()
                rolling_mad = errors.rolling(self.rolling_window, min_periods=min(self.rolling_window, 20)).apply(
                    lambda x: np.median(np.abs(x - np.median(x)))
                )
                
                global_mad = self.specialist_metadata_.get(specialist_name, {}).get('global_mad', 1.0)
                sigma_floor = np.maximum(global_mad, 1.0)
                
                zscore_surprise = np.abs(errors - rolling_median) / (np.maximum(rolling_mad, sigma_floor))
                magnitude_surprise = np.abs(errors) / (np.maximum(rolling_mad, sigma_floor))
                surprise_scores = 0.6 * zscore_surprise + 0.4 * magnitude_surprise
                
            else:
                raise ValueError(f"Unknown surprise method: {method}")
            
            return surprise_scores.fillna(0)
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Surprise computation failed for {specialist_name}: {e}")
            return pd.Series(0, index=self.specialist_errors_[specialist_name].index)
    
    def detect_structural_breaks(
        self,
        specialist_name: str,
        method: str = "chow"
    ) -> pd.Series:
        """
        Detect structural breaks in specialist prediction errors.
        
        Args:
            specialist_name: Name of the specialist
            method: Method for break detection ("chow", "cusum", "variance")
            
        Returns:
            Structural break indicators
        """
        try:
            if specialist_name not in self.specialist_errors_:
                raise ValueError(f"Specialist {specialist_name} not registered")
            
            errors = self.specialist_errors_[specialist_name].values
            n_samples = len(errors)
            break_indicators = np.zeros(n_samples)
            
            if method == "chow":
                # Simplified Chow test for structural breaks
                for i in range(self.structural_break_window, n_samples - self.structural_break_window):
                    # Split data at potential break point
                    errors_before = errors[i - self.structural_break_window:i]
                    errors_after = errors[i:i + self.structural_break_window]
                    
                    if len(errors_before) > 10 and len(errors_after) > 10:
                        # Compare means and variances
                        mean_before, var_before = np.mean(errors_before), np.var(errors_before)
                        mean_after, var_after = np.mean(errors_after), np.var(errors_after)
                        
                        # Simple break test statistic
                        mean_diff = abs(mean_before - mean_after)
                        var_ratio = max(var_before, var_after) / (min(var_before, var_after) + 1e-8)
                        
                        # Break if significant difference
                        if mean_diff > 2 * np.sqrt(var_before) or var_ratio > 3:
                            break_indicators[i] = 1
            
            elif method == "cusum":
                # CUSUM-based break detection
                cumulative_errors = np.cumsum(errors - np.mean(errors))
                std_cumulative = np.std(cumulative_errors)
                
                # Break if cumulative error exceeds threshold
                threshold = 3 * std_cumulative
                break_indicators[np.abs(cumulative_errors) > threshold] = 1
            
            elif method == "variance":
                # Variance-based break detection
                rolling_var = pd.Series(errors).rolling(self.structural_break_window).var()
                var_threshold = rolling_var.quantile(0.95)
                break_indicators[rolling_var > var_threshold] = 1
            
            return pd.Series(break_indicators, index=self.specialist_errors_[specialist_name].index)
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Structural break detection failed for {specialist_name}: {e}")
            return pd.Series(0, index=self.specialist_errors_[specialist_name].index)
    
    def set_zone_score_weights(
        self,
        zone3_boost: float = 0.5,
        zone2_boost: float = 0.2,
        exposure_scalar: float = 1.0
    ) -> None:
        """Update zone score emphasis parameters on the fly."""
        self.zone3_ratio_boost = float(zone3_boost)
        self.zone2_ratio_boost = float(zone2_boost)
        self.zone_score_exposure = float(exposure_scalar)
        if self.verbose:
            tprint_info(
                f"🎚️ Zone score weights updated: "
                f"zone3={self.zone3_ratio_boost:.2f}, "
                f"zone2={self.zone2_ratio_boost:.2f}, "
                f"exposure={self.zone_score_exposure:.2f}"
            )

    def aggregate_specialist_surprise(
        self,
        spectral_reliability: Optional[Dict[str, Dict[str, Any]]] = None,
        exposure_scalar: float = 1.0,
        regime_vol: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Aggregate surprise scores across all specialists.
        
        Args:
            spectral_reliability: Optional spectral reliability metrics
            exposure_scalar: Exposure scaling factor
            regime_vol: Optional regime volatility series for enhanced weighting
            
        Returns:
            DataFrame with aggregated surprise metrics
            
        Raises:
            ValueError: If insufficient specialists or empty data
        """
        try:
            if self.verbose:
                tprint_info("🔄 Aggregating Specialist Surprise (Continuous Framework)...")
            
            if len(self.specialist_errors_) < self.min_specialists:
                if self.verbose:
                    tprint_error(f"❌ FAIL FAST: Insufficient specialists: {len(self.specialist_errors_)} < {self.min_specialists}")
                raise ValueError(f"Insufficient specialists for surprise aggregation: {len(self.specialist_errors_)} < {self.min_specialists}")
            
            errors_df = self._build_error_frame()
            if errors_df.empty:
                if self.verbose:
                    tprint_error("❌ FAIL FAST: Empty error frame - no specialist data available")
                raise ValueError("Empty error frame - no specialist data available")
            
            # Step 1: Compute Soft Surprises (Logistic Sigmoid Mapping)
            soft_surprise_df = self.compute_soft_surprise(errors_df)
            self.specialist_soft_surprise_ = soft_surprise_df
            
            # Step 2: Compute Global ZoneScore (Continuous Gradient)
            zone_score = self.compute_zone_score(soft_surprise_df)
            
            # Step 3: Compute Legacy/Discrete metrics for backward compatibility
            surprise_df = self._compute_batch_surprises(errors_df, method="combined")
            self.specialist_surprises_ = surprise_df  # Required for generate_causal_events
            
            # Step 4: Integrate regime_vol if provided for enhanced weighting
            if regime_vol is not None and not regime_vol.empty:
                # Align regime_vol with surprise_df index
                regime_vol_aligned = regime_vol.reindex(surprise_df.index).fillna(0.0)
                # Use regime volatility to modulate surprise intensity
                regime_weight = 1.0 + 0.5 * np.tanh(regime_vol_aligned)  # Scale: [0.5, 1.5]
                surprise_df = surprise_df.multiply(regime_weight, axis=0)
                if self.verbose:
                    tprint_info(f"📊 Integrated regime volatility weighting (mean factor: {regime_weight.mean():.3f})")
            
            # Build reliability weights by blending Spectral + detector scores
            spectral_scores = pd.Series()
            if spectral_reliability:
                spectral_scores = pd.Series({
                    spec: float(metrics.get("composite_reliability", np.nan))
                    for spec, metrics in spectral_reliability.items()
                })
            detector_scores = pd.Series({
                spec: metrics.get('composite_reliability', np.nan)
                for spec, metrics in (self.specialist_reliability_ or {}).items()
            })
            reliability = pd.concat([spectral_scores, detector_scores], axis=1)
            if reliability.empty:
                weight_series = pd.Series(1.0, index=surprise_df.columns)
            else:
                reliability.columns = ['spectral', 'detector']
                reliability['spectral'] = reliability['spectral'].clip(lower=0, upper=1)
                reliability['detector'] = reliability['detector'].clip(lower=0, upper=1)
                reliability = reliability.reindex(surprise_df.columns).fillna(0.0)
                reliability['blended'] = 0.6 * reliability['spectral'] + 0.4 * reliability['detector']
                reliability['blended'] = reliability['blended'].replace(0.0, np.nan)
                reliability['blended'] = reliability['blended'].fillna(
                    reliability[['spectral', 'detector']].max(axis=1)
                )
                reliability['blended'] = reliability['blended'].replace(0.0, 0.1)
                weight_series = reliability['blended']
            weight_series = weight_series.reindex(surprise_df.columns).fillna(0.1)
            weight_series = weight_series / weight_series.sum()

            # Aggregation logic with reliability-weighted statistics
            aggregated = pd.DataFrame(index=surprise_df.index)
            aggregated['max_surprise'] = (surprise_df * weight_series).max(axis=1)
            aggregated['mean_surprise'] = (surprise_df * weight_series).sum(axis=1)
            weighted_consensus = (surprise_df > self.surprise_threshold).mul(weight_series, axis=1).sum(axis=1)
            aggregated['surprise_consensus'] = weighted_consensus * float(len(weight_series))
            
            # Define break_df for total breaks calculation
            break_df = (surprise_df > self.surprise_threshold).astype(int)
            aggregated['total_breaks'] = break_df.sum(axis=1)
            aggregated['has_break'] = (aggregated['total_breaks'] > 0).astype(int)

            zone_levels = self._compute_specialist_zone_levels(surprise_df)
            zone_scores = self._compute_specialist_zone_scores(zone_levels, surprise_df)
            combined_zone_score = self._compute_combined_zone_score(zone_scores, zone_levels)
            adjusted_zone_score = combined_zone_score.reindex(aggregated.index).fillna(0.0)
            adjusted_zone_score *= float(getattr(self, "zone_score_exposure", 1.0)) * float(exposure_scalar)
            aggregated['zone_score'] = adjusted_zone_score.clip(0.0, self.zone_score_cap)
            denom = float(zone_levels.shape[1]) if zone_levels.shape[1] > 0 else 1.0
            aggregated['zone3_ratio'] = (zone_levels == 3.0).sum(axis=1) / denom
            aggregated['zone2_ratio'] = (zone_levels == 2.0).sum(axis=1) / denom
            
            # Map ZoneScore to Discrete Zones for reporting/filtering
            # Zone 1: [0, 0.33], Zone 2: [0.33, 0.66], Zone 3: [0.66, 1.0]
            aggregated['surprise_zone'] = pd.cut(
                zone_score,
                bins=[-np.inf, 0.33, 0.66, np.inf],
                labels=[1, 2, 3]
            ).astype(float).fillna(1)
            
            # Surprise Density Monitoring
            n_samples = len(aggregated)
            if n_samples > 0:
                # Use a soft threshold for density reporting
                surprise_density = (zone_score > 0.33).mean()
                self.surprise_density_ = surprise_density
                if self.verbose:
                    tprint_info(f"📊 Surprise Density (Soft): {surprise_density:.2%} (Target: < 15%)")
            
            # Combined surprise indicator (True if outside Zone 1)
            aggregated['causal_surprise'] = (zone_score > 0.33).astype(int)
            
            # Store results
            self.surprise_events_ = aggregated
            
            if self.verbose:
                tprint_success(f"✅ Continuous surprise aggregation complete:")
                tprint_info(f"   - ZoneScore Mean: {zone_score.mean():.4f}")
            
            return aggregated
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ FAIL FAST: Surprise aggregation failed: {e}")
                tprint_error(f"   Specialist count: {len(self.specialist_errors_)}")
                tprint_error(f"   Error frame shape: {errors_df.shape if 'errors_df' in locals() else 'N/A'}")
            raise ValueError(f"Causal surprise aggregation failed: {e}") from e
            
    def _build_error_frame(self) -> pd.DataFrame:
        """
        Build a time-aligned DataFrame of specialist errors.
        """
        if not self.specialist_errors_:
            return pd.DataFrame()
        
        errors_df = pd.DataFrame(self.specialist_errors_)
        errors_df = errors_df.sort_index()
        
        # Drop rows that are completely empty
        errors_df = errors_df.dropna(how='all')
        return errors_df
    
    def _compute_batch_surprises(
        self,
        errors_df: pd.DataFrame,
        method: str = "combined"
    ) -> pd.DataFrame:
        """
        Compute surprise scores for all specialists using vectorized rolling stats.
        """
        window = self.rolling_window
        min_periods = min(window, 20)
        
        rolling_median = errors_df.rolling(window=window, min_periods=min_periods).median()
        
        
        # 2026 Optimization: Use Numba for rolling MAD (100x speedup)
        # Replaces slow rolling().apply(mad_func) which is Python-loop bound
        try:
            mads_dict = {}
            for col in errors_df.columns:
                # Numba requires dense float arrays
                # We assume fillna(0) is safe for error/diff arrays
                values = errors_df[col].fillna(0).astype(np.float64).values
                mads = _numba_rolling_mad(values, window)
                mads_dict[col] = mads
            
            rolling_mad = pd.DataFrame(mads_dict, index=errors_df.index)
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"⚠️ Numba MAD failed ({e}), falling back to slow pandas...")
            
            def mad_func(values: np.ndarray) -> float:
                median = np.median(values)
                return np.median(np.abs(values - median))
            
            rolling_mad = errors_df.rolling(window=window, min_periods=min_periods).apply(
                mad_func,
                raw=True
            )
        
        # Apply Robust Floors (2026 Pro Secret)
        # 1. Individual specialist global MAD floor
        # 2. Hard absolute floor (1.0)
        global_mads = {}
        for col in errors_df.columns:
            meta = self.specialist_metadata_.get(col, {})
            # Use max(global_mad, 1.0) as floor for each specialist
            global_mads[col] = max(meta.get('global_mad', 1.0), 1.0)
        
        # Convert to Series for easy broadcasting
        sigma_floors = pd.Series(global_mads)
        
        # Apply floors: max(rolling_mad, sigma_floor)
        # Note: pandas will align columns automatically with axis=1
        rolling_mad = rolling_mad.clip(lower=sigma_floors, axis=1)
        
        if method == "zscore" or method == "robust_zscore":
            surprise_df = (errors_df - rolling_median).abs() / rolling_mad
        
        elif method == "magnitude":
            surprise_df = errors_df.abs() / rolling_mad
        
        elif method == "combined":
            zscore_surprise = (errors_df - rolling_median).abs() / rolling_mad
            magnitude_surprise = errors_df.abs() / rolling_mad
            surprise_df = 0.6 * zscore_surprise + 0.4 * magnitude_surprise
        else:
            raise ValueError(f"Unknown surprise method: {method}")
        
        return surprise_df.fillna(0)
    

    def _compute_specialist_zone_levels(self, surprise_df: pd.DataFrame) -> pd.DataFrame:
        """Map specialist surprise values to discrete zones per specialist."""
        if surprise_df.empty:
            return pd.DataFrame(index=surprise_df.index)
        zone_bins = [-np.inf, 1.5, 3.0, np.inf]
        zone_labels = [1.0, 2.0, 3.0]
        zone_levels = {}
        for col in surprise_df.columns:
            levels = pd.cut(surprise_df[col], bins=zone_bins, labels=zone_labels).astype(float)
            zone_levels[col] = levels.fillna(1.0)
        return pd.DataFrame(zone_levels, index=surprise_df.index)

    def _compute_specialist_zone_scores(
        self,
        zone_levels: pd.DataFrame,
        surprise_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Convert zone levels to soft 0-1 scores with optional continuous adjustment."""
        if zone_levels.empty:
            return zone_levels
        score_map = {1.0: 0.0, 2.0: 0.5, 3.0: 1.0}
        zone_scores = zone_levels.replace(score_map)
        if self.zone_score_config.get('use_continuous_mapping', True):
            normalized = surprise_df / (self.surprise_threshold + 1e-8)
            normalized = normalized.clip(0.0, 2.0)
            zone_scores = zone_scores.combine(normalized, lambda z, n: np.minimum(1.0, np.maximum(z, z * n)))
        return zone_scores.fillna(0.0)

    def _compute_combined_zone_score(
        self,
        zone_scores: pd.DataFrame,
        zone_levels: pd.DataFrame
    ) -> pd.Series:
        """Aggregate specialist zone scores into a single scalar with chaos emphasis."""
        if zone_scores.empty:
            return pd.Series(0.0, index=zone_scores.index)
        powered = zone_scores.pow(self.zone_score_power)
        base_score = powered.mean(axis=1).pow(1.0 / self.zone_score_power)
        zone3_ratio = (zone_levels == 3.0).sum(axis=1) / np.maximum(1, zone_levels.shape[1])
        zone2_ratio = (zone_levels == 2.0).sum(axis=1) / np.maximum(1, zone_levels.shape[1])
        boost = 1.0 + self.zone3_ratio_boost * zone3_ratio + self.zone2_ratio_boost * zone2_ratio
        boosted = base_score * boost
        boosted = boosted.clip(lower=0.0, upper=self.zone_score_cap)
        boosted[zone3_ratio >= self.zone3_floor] = self.zone_score_cap
        return boosted.fillna(0.0)

    def _compute_batch_breaks(
        self,
        errors_df: pd.DataFrame,
        method: str = "chow"
    ) -> pd.DataFrame:
        """
        Compute structural break indicators for all specialists.
        """
        window = self.structural_break_window
        if method != "chow":
            # Fallback to per-specialist computation for other methods
            break_data = {}
            for specialist_name in errors_df.columns:
                break_indicators = self.detect_structural_breaks(specialist_name, method=method)
                break_data[specialist_name] = break_indicators
            return pd.DataFrame(break_data).fillna(0)
        
        if window <= 0:
            return pd.DataFrame(np.zeros_like(errors_df), index=errors_df.index, columns=errors_df.columns)
        
        mean_before = errors_df.rolling(window=window, min_periods=window).mean()
        var_before = errors_df.rolling(window=window, min_periods=window).var()
        
        shifted_errors = errors_df.shift(-window)
        mean_after = shifted_errors.rolling(window=window, min_periods=window).mean()
        var_after = shifted_errors.rolling(window=window, min_periods=window).var()
        
        mean_diff = (mean_before - mean_after).abs()
        
        var_ratio = pd.DataFrame(
            np.maximum(var_before, var_after) / (np.minimum(var_before, var_after) + 1e-8),
            index=errors_df.index,
            columns=errors_df.columns
        ).replace([np.inf, -np.inf], np.nan)
        
        variance_threshold = (2 * np.sqrt(var_before.clip(lower=1e-8)))
        mean_breaks = (mean_diff > variance_threshold)
        variance_breaks = (var_ratio > 3)
        
        break_df = (mean_breaks | variance_breaks).astype(int)
        return break_df.fillna(0)
    
    def generate_causal_events(
        self,
        event_threshold: float = 0.3,  # Lowered from 0.5 to capture more events
        min_event_separation: float = 0.25  # Lowered from 1 hour to 15 minutes
    ) -> Dict[int, Dict[str, Any]]:
        """
        Generate causal surprise events from aggregated data.
        
        Args:
            event_threshold: Threshold for event generation (lowered default to 0.3)
            min_event_separation: Minimum separation between events in hours (lowered to 0.25)
            
        Returns:
            Dictionary of causal events
        """
        try:
            if self.verbose:
                tprint_info("🎯 Generating Causal Surprise Events...")
            
            if self.surprise_events_ is None or len(self.surprise_events_) == 0:
                self.aggregate_specialist_surprise()
            
            if self.surprise_events_ is None or len(self.surprise_events_) == 0:
                tprint_warning("   ⚠️ No surprise events found after aggregation - check specialist registration")
                return {}
            
            # Find event candidates: only pick the START of surprise sequences
            # Rule: event_time = first_bar_where_surprise_crosses_threshold
            is_surprised = self.surprise_events_['causal_surprise'] == 1
            surprise_start = is_surprised & (~is_surprised.shift(1).fillna(False))
            surprise_mask = surprise_start
            
            # DIAGNOSTIC: Log how many raw surprise bars exist
            raw_surprise_count = is_surprised.sum()
            if self.verbose:
                tprint_info(f"   📊 Raw surprise bars: {raw_surprise_count} of {len(is_surprised)} ({100*raw_surprise_count/max(1,len(is_surprised)):.2f}%)")
                tprint_info(f"   📊 Surprise sequence starts: {surprise_start.sum()}")
            
            # DIAGNOSTIC: Log pre-time-barrier count
            pre_time_barrier_count = surprise_mask.sum()
            
            # Apply Time Barrier (Filter slow-moving/persistent signals)
            # If signal stays "surprised" for too long, it's a regime, not an event.
            # Max duration: 24 bars (6 hours on 15m) - more lenient for structural events
            max_duration = 48  # Increased from 24 to allow longer structural events (12 hours on 15m bars)
            if isinstance(self.specialist_surprises_, pd.DataFrame):
                # Detect persistent surprise blocks
                is_surprised = (self.specialist_surprises_ > self.surprise_threshold).any(axis=1)
                clean_mask = self._filter_slow_moving_events(is_surprised, max_duration)
                surprise_mask = surprise_mask & clean_mask
            
            post_time_barrier_count = surprise_mask.sum()
            if self.verbose:
                tprint_info(f"   📊 After time-barrier filter: {pre_time_barrier_count} → {post_time_barrier_count} (removed {pre_time_barrier_count - post_time_barrier_count} slow-moving)")
            
            event_candidates = self.surprise_events_[surprise_mask].index
            
            # Filter events by minimum separation
            filtered_events = []
            last_event_time = None
            rejected_by_separation = 0
            
            for event_time in event_candidates:
                if last_event_time is None or (event_time - last_event_time).total_seconds() / 3600 >= min_event_separation:
                    filtered_events.append(event_time)
                    last_event_time = event_time
                else:
                    rejected_by_separation += 1
            
            if self.verbose:
                tprint_info(f"   📊 Min separation filter ({min_event_separation:.2f}h): rejected {rejected_by_separation} events")
            
            # Generate event dictionary
            causal_events = {}
            
            for i, event_time in enumerate(filtered_events):
                event_data = self.surprise_events_.loc[event_time]
                
                causal_events[event_time] = {
                    'type': 'causal_surprise',
                    'strength': event_data['max_surprise'],
                    'zone': int(event_data['surprise_zone']),
                    'consensus': event_data['surprise_consensus'],
                    'mean_surprise': event_data['mean_surprise'],
                    'zone_score': event_data['zone_score'],
                    'has_structural_break': event_data['has_break'],
                    'total_breaks': event_data['total_breaks'],
                    'specialist_count': len(self.specialist_errors_),
                    'source': 'specialist_prediction_errors',
                    'sigma_method': 'rolling_mad',
                    'event_id': f"causal_surprise_{i}"
                }
            
            self.surprise_events_ = causal_events
            
            if self.verbose:
                tprint_success(f"✅ Generated {len(causal_events)} causal surprise events:")
                tprint_info(f"   - Event threshold: {event_threshold}")
                tprint_info(f"   - Min separation: {min_event_separation} hours")
                tprint_info(f"   - Candidates filtered: {len(event_candidates)} → {len(filtered_events)}")
            
            return causal_events
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Event generation failed: {e}")
            return {}
            
    def _filter_slow_moving_events(self, is_surprised: pd.Series, max_duration: int) -> pd.Series:
        """
        Filter out events that are part of a long-duration surprise sequence.
        Returns mask where True = keep (not slow moving).
        """
        # Identify blocks of True
        block_id = (is_surprised != is_surprised.shift()).cumsum()
        duration = is_surprised.groupby(block_id).transform('count')
        
        # Keep only if duration <= max_duration OR it's not a surprise anyway
        # We only want to filter out existing surprises that are too long
        keep_mask = ~is_surprised | (duration <= max_duration)
        return keep_mask
    
    def analyze_surprise_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in causal surprise events.
        
        Returns:
            Dictionary with pattern analysis
        """
        try:
            if self.surprise_events_ is None or len(self.surprise_events_) == 0:
                return {}
            
            # Convert to DataFrame for analysis
            events_df = pd.DataFrame.from_dict(self.surprise_events_, orient='index')
            
            if events_df.empty:
                return {}
            
            analysis = {
                'total_events': len(events_df),
                'avg_strength': events_df['strength'].mean(),
                'max_strength': events_df['strength'].max(),
                'avg_consensus': events_df['consensus'].mean(),
                'break_events': events_df['has_structural_break'].sum(),
                'break_ratio': events_df['has_structural_break'].mean(),
                'avg_zone_score': events_df['zone_score'].mean() if 'zone_score' in events_df.columns else 0.0,
                'avg_zone3_ratio': events_df['zone3_ratio'].mean() if 'zone3_ratio' in events_df.columns else 0.0,
                'avg_zone2_ratio': events_df['zone2_ratio'].mean() if 'zone2_ratio' in events_df.columns else 0.0,
                'strength_distribution': events_df['strength'].describe().to_dict(),
                'consensus_distribution': events_df['consensus'].describe().to_dict()
            }
            
            # Temporal patterns
            if len(events_df) > 1:
                event_times = events_df.index
                time_diffs = event_times[1:] - event_times[:-1]
                analysis['avg_time_between_events'] = time_diffs.mean()
                analysis['event_frequency'] = len(events_df) / (time_diffs.sum().total_seconds() / 3600)  # events per hour
            
            return analysis
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Pattern analysis failed: {e}")
            return {}
            
    def compute_reliability_metrics(
        self,
        realized_outcomes: pd.Series,
        binary_labels: pd.Series = None
    ) -> Dict[str, Any]:
        """
        Compute advanced reliability metrics for specialists and detector.
        
        Args:
            realized_outcomes: Continuous future returns / outcome magnitude
            binary_labels: Ground truth meta-labels (optional)
            
        Returns:
            Dictionary with reliability metrics
        """
        try:
            if self.verbose:
                tprint_info("🔬 Computing Advanced Causal Reliability Metrics...")
            
            events_df_aligned = self._get_events_dataframe()
            if self.specialist_surprises_.empty or events_df_aligned.empty:
                tprint_warning("⚠️ No surprise data available for reliability computation")
                return {}

            if not isinstance(realized_outcomes, pd.Series):
                realized_outcomes = pd.Series(
                    realized_outcomes,
                    index=self.specialist_surprises_.index[:len(realized_outcomes)]
                )
            if binary_labels is not None and not isinstance(binary_labels, pd.Series):
                binary_labels = pd.Series(
                    binary_labels,
                    index=realized_outcomes.index[:len(binary_labels)]
                )
            
            common_idx = self.specialist_surprises_.index.intersection(realized_outcomes.index)
            if binary_labels is not None:
                common_idx = common_idx.intersection(binary_labels.index)
            if common_idx.empty:
                tprint_warning("⚠️ No overlapping samples for reliability computation")
                return {}
            
            surprises = self.specialist_surprises_.reindex(common_idx)
            zones = self.specialist_zone_levels_.reindex(common_idx)
            outcomes = realized_outcomes.reindex(common_idx)
            labels = binary_labels.reindex(common_idx) if binary_labels is not None else None
            
            specialist_metrics = {}
            
            # 1. Specialist Reliability Metrics
            for spec_name in surprises.columns:
                spec_surprise = surprises[spec_name]
                if zones is None or spec_name not in zones.columns:
                    continue
                spec_zones = zones[spec_name]
                
                # 1.1 Surprise Responsiveness (Correlation between surprise and magnitude)
                responsiveness, _ = stats.spearmanr(spec_surprise.abs(), outcomes.abs())
                
                # 1.2 Zone-Conditioned Precision
                # For meta-labeling, precision is accuracy of predicting positive meta-labels
                z2_mask = spec_zones == 2.0
                z3_mask = spec_zones == 3.0
                
                z2_precision = 0.0
                z3_precision = 0.0
                
                if labels is not None:
                    if z2_mask.sum() > 5:
                        z2_precision = labels.loc[z2_mask[z2_mask].index].mean()
                    if z3_mask.sum() > 5:
                        z3_precision = labels.loc[z3_mask[z3_mask].index].mean()
                
                # 1.3 Confidence Calibration (Brier Score equivalent for surprise)
                # If surprise > threshold, we predict a move.
                calibration_score = 0.0
                if labels is not None:
                    preds = (spec_surprise > self.surprise_threshold).astype(float)
                    calibration_score = 1.0 - np.mean((preds - labels)**2) # Accuracy-like Brier
                
                # 1.4 Marginal Value (Leave-One-Out)
                # Proxy: Correlation of total consensus vs LOO consensus
                total_consensus = (surprises > self.surprise_threshold).sum(axis=1)
                loo_surprises = surprises.drop(columns=[spec_name])
                loo_consensus = (loo_surprises > self.surprise_threshold).sum(axis=1)
                
                full_corr, _ = stats.spearmanr(total_consensus, outcomes.abs())
                loo_corr, _ = stats.spearmanr(loo_consensus, outcomes.abs())
                marginal_value = full_corr - loo_corr
                
                # 1.5 Consensus Alignment (Correlation with other specialists)
                consensus_corr = 0.0
                if loo_consensus.std() > 0:
                    consensus_corr, _ = stats.spearmanr(spec_surprise.abs(), loo_consensus)
                
                specialist_metrics[spec_name] = {
                    'responsiveness': responsiveness,
                    'z2_precision': z2_precision,
                    'z3_precision': z3_precision,
                    'calibration': calibration_score,
                    'marginal_value': marginal_value,
                    'consensus_corr': consensus_corr
                }
                
                # 3. Composite Reliability Score
                specialist_metrics[spec_name]['composite_reliability'] = self._compute_composite_reliability(
                    specialist_metrics[spec_name]
                )
            
            self.specialist_reliability_ = specialist_metrics
            
            # 2. Detector Reliability Metrics
            # Ensure we're using the aligned DataFrame for consensus/chaos
            # Use 'events_df_aligned' which we prepared earlier
            
            detector_metrics = {
                'filtered_event_density': self.surprise_density_
            }
            consensus_chaos_corr = 0.0
            if {'surprise_consensus', 'total_breaks'}.issubset(events_df_aligned.columns):
                aligned_slice = events_df_aligned.reindex(common_idx)
                if aligned_slice[['surprise_consensus', 'total_breaks']].dropna(how='all').shape[0] > 1:
                    consensus_chaos_corr = stats.spearmanr(
                        aligned_slice['surprise_consensus'],
                        aligned_slice['total_breaks']
                    )[0]
            detector_metrics['consensus_chaos_correlation'] = consensus_chaos_corr
            
            if labels is not None:
                aligned = events_df_aligned.reindex(common_idx)
                if 'causal_surprise' in events_df_aligned.columns:
                    surprise_mask = aligned['causal_surprise'].fillna(0).astype(int) == 1
                else:
                    # Deriving mask from presence in events DataFrame
                    surprise_mask = pd.Series(False, index=common_idx)
                    common_events = events_df_aligned.index.intersection(common_idx)
                    surprise_mask.loc[common_events] = True
                
                # 2.1 Precision
                active_labels = labels[surprise_mask]
                precision = active_labels.mean() if not active_labels.empty else 0.0
                detector_metrics['precision'] = precision
                
                # 2.2 Recall (Capture Rate of Profitable Opportunities)
                profitable_mask = labels == 1
                captured = (surprise_mask & profitable_mask).sum()
                recall = captured / max(1, profitable_mask.sum())
                detector_metrics['recall'] = recall
                
                # DIAGNOSTIC: Explain recall computation
                if self.verbose:
                    tprint_info(f"   📊 Recall Diagnostics:")
                    tprint_info(f"      - Total profitable opportunities (labels==1): {profitable_mask.sum()}")
                    tprint_info(f"      - Surprise-flagged bars: {surprise_mask.sum()}")
                    tprint_info(f"      - Overlap (captured): {captured}")
                    if profitable_mask.sum() > 0 and surprise_mask.sum() > 0 and captured == 0:
                        tprint_warning(f"      ⚠️ Zero overlap: Surprise events may not align with ground truth labels")
                
                # 2.3 F1 Score
                detector_metrics['f1'] = 2 * (precision * recall) / max(1e-8, precision + recall)
                
                # 2.4 Stability Across Time (Split Half Reliability)
                mid_point = len(common_idx) // 2
                first_half = common_idx[:mid_point]
                second_half = common_idx[mid_point:]
                
                def _precision_for_slice(idx_slice: pd.Index) -> float:
                    if len(idx_slice) == 0:
                        return 0.0
                    slice_mask = surprise_mask.loc[idx_slice]
                    if not slice_mask.any():
                        return 0.0
                    slice_labels = labels.loc[idx_slice][slice_mask]
                    return float(slice_labels.mean()) if not slice_labels.empty else 0.0
                
                prec_h1 = _precision_for_slice(first_half)
                prec_h2 = _precision_for_slice(second_half)
                detector_metrics['stability_index'] = 1.0 - abs(prec_h1 - prec_h2) # 1.0 is perfectly stable
                
            self.detector_reliability_ = detector_metrics
            
            if self.verbose:
                tprint_success("✅ Causal Reliability Metrics computed")
                if 'f1' in detector_metrics:
                    tprint_info(f"   - Detector: F1={detector_metrics['f1']:.3f}, Recall={detector_metrics['recall']:.3f}, Precision={detector_metrics.get('precision', 0.0):.3f}")
                for spec, m in specialist_metrics.items():
                    tprint_info(f"   - {spec}: Reliability {m['composite_reliability']:.3f} (Resp: {m['responsiveness']:.3f})")
            
            return {
                'specialists': specialist_metrics,
                'detector': detector_metrics
            }
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Reliability computation failed: {e}")
            return {}

    def _get_events_dataframe(self) -> pd.DataFrame:
        """Return the most informative representation of surprise events as a DataFrame."""
        if hasattr(self, "surprise_aggregates_df_") and not self.surprise_aggregates_df_.empty:
            return self.surprise_aggregates_df_.copy()
        if isinstance(self.surprise_events_, pd.DataFrame):
            return self.surprise_events_.copy()
        if isinstance(self.surprise_events_, dict) and self.surprise_events_:
            df = pd.DataFrame.from_dict(self.surprise_events_, orient='index')
            return df.sort_index()
        return pd.DataFrame()

    def _compute_composite_reliability(self, metrics: Dict[str, float]) -> float:
        """Combine metrics into a single reliability score."""
        # Weights for the 2026 Pro Secret Reliability Formula:
        # Precision in Zone2 (Opportunity) is the primary driver (40%)
        # Response to market magnitude is secondary (30%)
        # Alignment with the committee (consensus) is tertiary (20%)
        # Marginal value adds the final alpha bump (10%)
        w1, w2, w3, w4 = 0.4, 0.2, 0.3, 0.1
        
        # Normalize and combine
        score = (
            w1 * max(0, metrics['z2_precision']) +
            w2 * max(0, metrics.get('consensus_corr', 0.5)) + 
            w3 * max(0, metrics['responsiveness']) +
            w4 * np.tanh(max(0, metrics['marginal_value'] * 20)) # Non-linear marginal lift
        )
        return float(np.clip(score, 0, 1))
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of causal surprise detection.
        
        Returns:
            Summary dictionary
        """
        return {
            'specialists_registered': len(self.specialist_errors_),
            'surprise_threshold': self.surprise_threshold,
            'rolling_window': self.rolling_window,
            'structural_break_window': self.structural_break_window,
            'events_generated': len(self.surprise_events_) if self.surprise_events_ else 0,
            'has_surprise_data': self.surprise_events_ is not None
        }

# Convenience functions
def quick_causal_surprise(
    specialist_predictions: Dict[str, pd.Series],
    specialist_targets: Dict[str, pd.Series],
    **kwargs
) -> CausalSurpriseDetector:
    """
    Quick causal surprise detection.
    
    Args:
        specialist_predictions: Dictionary of specialist predictions
        specialist_targets: Dictionary of specialist targets
        **kwargs: Additional parameters
        
    Returns:
        CausalSurpriseDetector instance
    """
    detector = CausalSurpriseDetector(**kwargs)
    
    # Register all specialists
    for spec_name, predictions in specialist_predictions.items():
        if spec_name in specialist_targets:
            detector.register_specialist(spec_name, predictions, specialist_targets[spec_name])
    
    # Generate events
    detector.aggregate_specialist_surprise()
    detector.generate_causal_events()
    
    return detector

def detect_mechanism_breaks(
    specialist_errors: Dict[str, pd.Series],
    threshold: float = 2.0,
    **kwargs
) -> pd.DataFrame:
    """
    Detect mechanism breaks from specialist errors.
    
    Args:
        specialist_errors: Dictionary of specialist prediction errors
        threshold: Surprise threshold
        **kwargs: Additional parameters
        
    Returns:
        DataFrame with mechanism break indicators
    """
    detector = CausalSurpriseDetector(surprise_threshold=threshold, **kwargs)
    
    # Register specialists with errors
    for spec_name, errors in specialist_errors.items():
        # Create mock predictions (zeros) and targets (errors)
        predictions = pd.Series(0, index=errors.index)
        targets = errors
        detector.register_specialist(spec_name, predictions, targets)
    
    # Aggregate and return surprise events
    return detector.aggregate_specialist_surprise()
