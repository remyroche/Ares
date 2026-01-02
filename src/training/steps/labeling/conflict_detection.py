"""
Conflict Detection System for Layer 2.5 Chaser vs Causal Anchor

Detects when the Chaser is betting against the laws of physics
captured by the Causal Anchor, providing critical information
for the Meta-Learner's final decisions.

Key Features:
1. Direction Conflict Detection
2. Magnitude Conflict Analysis
3. Confidence-Weighted Conflict Scoring
4. Meta-Learner Integration Signals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from scipy import stats

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    # Fallback print functions
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")

class ConflictDetector:
    """
    Detects conflicts between Chaser predictions and Causal Anchor.
    
    Conflicts occur when the Chaser bets against the structural
    relationships captured by the Causal Anchor, which may indicate
    either opportunity or risk.
    """
    
    def __init__(
        self,
        direction_threshold: float = 0.0,
        magnitude_threshold: float = 1.0,
        confidence_threshold: float = 0.6,
        conflict_intensity_threshold: float = 0.5,
        statistical_window: int = 100,
        verbose: bool = True
    ):
        """
        Initialize Conflict Detector.
        
        Args:
            direction_threshold: Threshold for direction conflict detection
            magnitude_threshold: Threshold for magnitude conflict detection
            confidence_threshold: Minimum confidence for conflict consideration
            conflict_intensity_threshold: Threshold for high-intensity conflicts
            statistical_window: Window for statistical analysis
            verbose: Whether to print progress information
        """
        self.verbose = verbose
        self.direction_threshold = direction_threshold
        self.magnitude_threshold = magnitude_threshold
        self.confidence_threshold = confidence_threshold
        self.conflict_intensity_threshold = conflict_intensity_threshold
        self.statistical_window = statistical_window
        
        # Conflict statistics
        self.conflict_history_ = []
        self.conflict_rate_ = 0.0
        self.avg_conflict_intensity_ = 0.0
        
    def detect_direction_conflict(
        self,
        chaser_prediction: np.ndarray,
        causal_anchor_prediction: np.ndarray
    ) -> np.ndarray:
        """
        Detect direction conflicts between Chaser and Anchor.
        
        Args:
            chaser_prediction: Chaser residual predictions
            causal_anchor_prediction: Causal Anchor baseline predictions
            
        Returns:
            Boolean array indicating direction conflicts
        """
        try:
            # Direction conflict: opposite signs
            chaser_direction = np.sign(chaser_prediction)
            anchor_direction = np.sign(causal_anchor_prediction)
            
            # Conflict when directions are opposite
            direction_conflict = (chaser_direction != anchor_direction)
            
            # Handle zero cases
            chaser_zero = np.isclose(chaser_prediction, 0.0, atol=self.direction_threshold)
            anchor_zero = np.isclose(causal_anchor_prediction, 0.0, atol=self.direction_threshold)
            
            # No conflict if either prediction is essentially zero
            direction_conflict[chaser_zero | anchor_zero] = False
            
            return direction_conflict
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Direction conflict detection failed: {e}")
            raise
    
    def detect_magnitude_conflict(
        self,
        chaser_prediction: np.ndarray,
        causal_anchor_prediction: np.ndarray,
        chaser_confidence: np.ndarray
    ) -> np.ndarray:
        """
        Detect magnitude conflicts based on relative sizes.
        
        Args:
            chaser_prediction: Chaser residual predictions
            causal_anchor_prediction: Causal Anchor baseline predictions
            chaser_confidence: Chaser confidence scores
            
        Returns:
            Boolean array indicating magnitude conflicts
        """
        try:
            # Magnitude conflict: Chaser prediction is large relative to Anchor
            abs_chaser = np.abs(chaser_prediction)
            abs_anchor = np.abs(causal_anchor_prediction)
            
            # Avoid division by zero
            safe_anchor = np.maximum(abs_anchor, 1e-8)
            
            # Magnitude ratio
            magnitude_ratio = abs_chaser / safe_anchor
            
            # Conflict when Chaser magnitude exceeds threshold
            magnitude_conflict = magnitude_ratio > self.magnitude_threshold
            
            # Only consider high-confidence predictions
            magnitude_conflict &= (chaser_confidence >= self.confidence_threshold)
            
            return magnitude_conflict
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Magnitude conflict detection failed: {e}")
            raise
    
    def compute_conflict_intensity(
        self,
        chaser_prediction: np.ndarray,
        causal_anchor_prediction: np.ndarray,
        chaser_confidence: np.ndarray,
        direction_conflict: np.ndarray,
        magnitude_conflict: np.ndarray
    ) -> np.ndarray:
        """
        Compute overall conflict intensity score.
        
        Args:
            chaser_prediction: Chaser residual predictions
            causal_anchor_prediction: Causal Anchor baseline predictions
            chaser_confidence: Chaser confidence scores
            direction_conflict: Direction conflict flags
            magnitude_conflict: Magnitude conflict flags
            
        Returns:
            Conflict intensity scores (0-1)
        """
        try:
            # Base intensity from confidence
            intensity = chaser_confidence.copy()
            
            # Boost intensity for direction conflicts
            intensity[direction_conflict] *= 1.5
            
            # Boost intensity for magnitude conflicts
            intensity[magnitude_conflict] *= 1.3
            
            # Combined conflicts get maximum boost
            combined_conflict = direction_conflict & magnitude_conflict
            intensity[combined_conflict] *= 2.0
            
            # Cap at 1.0
            intensity = np.minimum(intensity, 1.0)
            
            return intensity
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Conflict intensity computation failed: {e}")
            raise
    
    def analyze_conflict_patterns(
        self,
        conflict_intensity: np.ndarray,
        chaser_prediction: np.ndarray,
        causal_anchor_prediction: np.ndarray
    ) -> Dict[str, float]:
        """
        Analyze patterns in conflicts for meta-learner insights.
        
        Args:
            conflict_intensity: Conflict intensity scores
            chaser_prediction: Chaser predictions
            causal_anchor_prediction: Causal Anchor predictions
            
        Returns:
            Dictionary with conflict pattern metrics
        """
        try:
            # Basic conflict statistics
            high_conflict_mask = conflict_intensity >= self.conflict_intensity_threshold
            conflict_rate = np.mean(high_conflict_mask)
            
            if conflict_rate == 0:
                return {
                    'conflict_rate': 0.0,
                    'avg_intensity': 0.0,
                    'intensity_std': 0.0,
                    'direction_conflict_rate': 0.0,
                    'magnitude_conflict_rate': 0.0,
                    'correlation_with_anchor': 0.0,
                    'opportunity_score': 0.0
                }
            
            # Intensity statistics
            avg_intensity = np.mean(conflict_intensity[high_conflict_mask])
            intensity_std = np.std(conflict_intensity[high_conflict_mask])
            
            # Direction vs magnitude conflict breakdown
            direction_conflict = self.detect_direction_conflict(chaser_prediction, causal_anchor_prediction)
            magnitude_conflict = self.detect_magnitude_conflict(
                chaser_prediction, causal_anchor_prediction, 
                np.ones_like(chaser_prediction)  # Assume full confidence for analysis
            )
            
            direction_conflict_rate = np.mean(direction_conflict & high_conflict_mask)
            magnitude_conflict_rate = np.mean(magnitude_conflict & high_conflict_mask)
            
            # Correlation analysis
            if len(chaser_prediction) > 10:
                correlation = np.corrcoef(chaser_prediction, causal_anchor_prediction)[0, 1]
            else:
                correlation = 0.0
            
            # Opportunity score (negative correlation + high conflict = opportunity)
            opportunity_score = max(0.0, -correlation) * conflict_rate
            
            patterns = {
                'conflict_rate': conflict_rate,
                'avg_intensity': avg_intensity,
                'intensity_std': intensity_std,
                'direction_conflict_rate': direction_conflict_rate,
                'magnitude_conflict_rate': magnitude_conflict_rate,
                'correlation_with_anchor': correlation,
                'opportunity_score': opportunity_score
            }
            
            return patterns

        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Conflict pattern analysis failed: {e}")
            raise

    def analyze_temporal_patterns(
        self,
        conflict_intensity: np.ndarray,
        chaser_prediction: np.ndarray,
        causal_anchor_prediction: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Analyze temporal patterns in conflicts for meta-learner insights.

        Args:
            conflict_intensity: Conflict intensity scores
            chaser_prediction: Chaser predictions
            causal_anchor_prediction: Causal Anchor predictions
            timestamps: Optional timestamps for time-based analysis

        Returns:
            Dictionary with temporal conflict pattern metrics
        """
        try:
            # Basic conflict statistics
            high_conflict_mask = conflict_intensity >= self.conflict_intensity_threshold
            conflict_rate = np.mean(high_conflict_mask)

            if conflict_rate == 0:
                return self._get_empty_temporal_analysis()

            # Autocorrelation analysis
            autocorr_results = self._calculate_conflict_autocorrelation(
                high_conflict_mask, chaser_prediction
            )

            # Conflict clustering analysis
            clustering_results = self._analyze_conflict_clustering(high_conflict_mask)

            # Temporal persistence analysis
            persistence_results = self._analyze_conflict_persistence(high_conflict_mask)

            # Regime-based conflict analysis
            regime_results = self._analyze_regime_conflicts(
                high_conflict_mask, chaser_prediction, causal_anchor_prediction
            )

            # Time-of-day patterns (if timestamps provided)
            tod_results = {}
            if timestamps is not None:
                tod_results = self._analyze_time_of_day_patterns(
                    high_conflict_mask, timestamps
                )

            # Volatility correlation
            volatility_results = self._analyze_volatility_correlation(
                high_conflict_mask, chaser_prediction, causal_anchor_prediction
            )

            # Combine all results
            temporal_patterns = {
                'conflict_rate': conflict_rate,
                'avg_intensity': np.mean(conflict_intensity[high_conflict_mask]),
                'intensity_std': np.std(conflict_intensity[high_conflict_mask]),
                **autocorr_results,
                **clustering_results,
                **persistence_results,
                **regime_results,
                **tod_results,
                **volatility_results
            }

            return temporal_patterns

        except Exception as e:
            if self.verbose:
                tprint_warning(f"⚠️ Temporal pattern analysis failed: {e}")
            return self._get_empty_temporal_analysis()

    def _calculate_conflict_autocorrelation(self, high_conflict_mask, chaser_prediction):
        """Calculate autocorrelation in conflict patterns."""
        conflict_series = high_conflict_mask.astype(int)

        autocorr_results = {}

        # Calculate autocorrelation for different lags
        max_lag = min(20, len(conflict_series) - 1)
        for lag in range(1, max_lag + 1):
            if len(conflict_series) > lag:
                autocorr = np.corrcoef(conflict_series[:-lag], conflict_series[lag:])[0, 1]
                autocorr_results[f'conflict_autocorr_lag_{lag}'] = autocorr if not np.isnan(autocorr) else 0.0

        # Overall autocorrelation strength
        autocorr_values = list(autocorr_results.values())
        autocorr_results['mean_autocorr_strength'] = np.mean(np.abs(autocorr_values)) if autocorr_values else 0.0
        autocorr_results['max_autocorr'] = max(autocorr_values) if autocorr_values else 0.0

        return autocorr_results

    def _analyze_conflict_clustering(self, high_conflict_mask):
        """Analyze how conflicts cluster together."""
        conflict_series = high_conflict_mask.astype(int)

        # Find conflict segments
        conflict_segments = []
        in_conflict = False
        segment_length = 0

        for i, is_conflict in enumerate(conflict_series):
            if is_conflict:
                if not in_conflict:
                    in_conflict = True
                    segment_length = 1
                else:
                    segment_length += 1
            else:
                if in_conflict:
                    conflict_segments.append(segment_length)
                    in_conflict = False
                    segment_length = 0

        if in_conflict:
            conflict_segments.append(segment_length)

        # Analyze segments
        if conflict_segments:
            clustering_results = {
                'num_conflict_segments': len(conflict_segments),
                'avg_segment_length': np.mean(conflict_segments),
                'max_segment_length': max(conflict_segments),
                'segment_length_std': np.std(conflict_segments),
                'total_conflict_duration': sum(conflict_segments),
                'conflict_density': sum(conflict_segments) / len(conflict_series)
            }
        else:
            clustering_results = {
                'num_conflict_segments': 0,
                'avg_segment_length': 0.0,
                'max_segment_length': 0,
                'segment_length_std': 0.0,
                'total_conflict_duration': 0,
                'conflict_density': 0.0
            }

        return clustering_results

    def _analyze_conflict_persistence(self, high_conflict_mask):
        """Analyze how persistent conflicts are."""
        conflict_series = high_conflict_mask.astype(int)

        if len(conflict_series) < 2:
            return {
                'persistence_ratio': 0.0,
                'avg_persistence_streak': 0.0,
                'max_persistence_streak': 0
            }

        # Calculate persistence (P(conflict_t | conflict_t-1))
        transitions = np.sum((conflict_series[1:] == 1) & (conflict_series[:-1] == 1))
        conflict_count = np.sum(conflict_series[:-1])

        persistence_ratio = transitions / conflict_count if conflict_count > 0 else 0.0

        # Calculate persistence streaks
        streaks = []
        current_streak = 0

        for is_conflict in conflict_series:
            if is_conflict:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                    current_streak = 0

        if current_streak > 0:
            streaks.append(current_streak)

        persistence_results = {
            'persistence_ratio': persistence_ratio,
            'avg_persistence_streak': np.mean(streaks) if streaks else 0.0,
            'max_persistence_streak': max(streaks) if streaks else 0,
            'persistence_streak_std': np.std(streaks) if streaks else 0.0
        }

        return persistence_results

    def _analyze_regime_conflicts(self, high_conflict_mask, chaser_pred, anchor_pred):
        """Analyze conflicts across different market regimes."""
        # Simple regime detection based on volatility
        returns = chaser_pred  # Use predictions as proxy for market activity
        volatility = pd.Series(returns).rolling(window=20).std()

        # Define regimes
        high_vol_threshold = volatility.quantile(0.75)
        low_vol_threshold = volatility.quantile(0.25)

        high_vol_regime = volatility > high_vol_threshold
        low_vol_regime = volatility < low_vol_threshold

        # Conflict rates by regime
        regime_results = {}

        if np.sum(high_vol_regime) > 0:
            regime_results['high_vol_conflict_rate'] = np.mean(high_conflict_mask[high_vol_regime])
        else:
            regime_results['high_vol_conflict_rate'] = 0.0

        if np.sum(low_vol_regime) > 0:
            regime_results['low_vol_conflict_rate'] = np.mean(high_conflict_mask[low_vol_regime])
        else:
            regime_results['low_vol_conflict_rate'] = 0.0

        # Overall regime difference
        regime_results['regime_conflict_difference'] = (
            regime_results['high_vol_conflict_rate'] - regime_results['low_vol_conflict_rate']
        )

        return regime_results

    def _analyze_time_of_day_patterns(self, high_conflict_mask, timestamps):
        """Analyze conflict patterns by time of day."""
        try:
            # Convert timestamps to hour of day
            hours = pd.to_datetime(timestamps).hour

            # Conflict rate by hour
            hourly_conflict_rates = {}
            for hour in range(24):
                hour_mask = hours == hour
                if np.sum(hour_mask) > 0:
                    hourly_conflict_rates[f'hour_{hour}_conflict_rate'] = np.mean(high_conflict_mask[hour_mask])
                else:
                    hourly_conflict_rates[f'hour_{hour}_conflict_rate'] = 0.0

            # Peak conflict hours
            conflict_rates = [v for k, v in hourly_conflict_rates.items() if 'conflict_rate' in k]
            peak_hour = np.argmax(conflict_rates) if conflict_rates else 0

            tod_results = {
                **hourly_conflict_rates,
                'peak_conflict_hour': peak_hour,
                'peak_conflict_rate': max(conflict_rates) if conflict_rates else 0.0,
                'off_peak_conflict_rate': np.mean(conflict_rates) if conflict_rates else 0.0
            }

            return tod_results

        except Exception:
            return {'time_of_day_analysis': 'failed'}

    def _analyze_volatility_correlation(self, high_conflict_mask, chaser_pred, anchor_pred):
        """Analyze correlation between conflicts and market volatility."""
        # Calculate prediction volatility
        chaser_volatility = pd.Series(chaser_pred).rolling(window=10).std()
        anchor_volatility = pd.Series(anchor_pred).rolling(window=10).std()

        # Correlation with conflicts
        valid_mask = ~(np.isnan(chaser_volatility) | np.isnan(anchor_volatility))

        if np.sum(valid_mask) > 10:
            chaser_vol_corr = np.corrcoef(high_conflict_mask[valid_mask], chaser_volatility[valid_mask])[0, 1]
            anchor_vol_corr = np.corrcoef(high_conflict_mask[valid_mask], anchor_volatility[valid_mask])[0, 1]
        else:
            chaser_vol_corr = 0.0
            anchor_vol_corr = 0.0

        return {
            'conflict_chaser_volatility_correlation': chaser_vol_corr,
            'conflict_anchor_volatility_correlation': anchor_vol_corr,
            'volatility_conflict_association': abs(chaser_vol_corr) + abs(anchor_vol_corr)
        }

    def _get_empty_temporal_analysis(self):
        """Return empty temporal analysis structure."""
        return {
            'conflict_rate': 0.0,
            'avg_intensity': 0.0,
            'intensity_std': 0.0,
            'mean_autocorr_strength': 0.0,
            'max_autocorr': 0.0,
            'num_conflict_segments': 0,
            'avg_segment_length': 0.0,
            'max_segment_length': 0,
            'segment_length_std': 0.0,
            'total_conflict_duration': 0,
            'conflict_density': 0.0,
            'persistence_ratio': 0.0,
            'avg_persistence_streak': 0.0,
            'max_persistence_streak': 0,
            'persistence_streak_std': 0.0,
            'high_vol_conflict_rate': 0.0,
            'low_vol_conflict_rate': 0.0,
            'regime_conflict_difference': 0.0
        }

    def detect_conflicts(
        self,
        chaser_prediction: np.ndarray,
        causal_anchor_prediction: np.ndarray,
        chaser_confidence: np.ndarray,
        update_statistics: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Complete conflict detection pipeline.
        
        Args:
            chaser_prediction: Chaser residual predictions
            causal_anchor_prediction: Causal Anchor baseline predictions
            chaser_confidence: Chaser confidence scores
            update_statistics: Whether to update internal statistics
            
        Returns:
            Dictionary with all conflict detection results
        """
        try:
            if self.verbose:
                tprint_info("🔍 Detecting Chaser vs Anchor conflicts...")
            
            # Validate inputs
            if not (len(chaser_prediction) == len(causal_anchor_prediction) == len(chaser_confidence)):
                raise ValueError("Input arrays must have the same length")
            
            # Step 1: Direction conflict detection
            direction_conflict = self.detect_direction_conflict(
                chaser_prediction, causal_anchor_prediction
            )
            
            # Step 2: Magnitude conflict detection
            magnitude_conflict = self.detect_magnitude_conflict(
                chaser_prediction, causal_anchor_prediction, chaser_confidence
            )
            
            # Step 3: Conflict intensity computation
            conflict_intensity = self.compute_conflict_intensity(
                chaser_prediction, causal_anchor_prediction, chaser_confidence,
                direction_conflict, magnitude_conflict
            )
            
            # Step 4: High conflict identification
            high_conflict = conflict_intensity >= self.conflict_intensity_threshold
            
            # Step 5: Total prediction (Anchor + Chaser)
            total_prediction = causal_anchor_prediction + chaser_prediction
            
            # Compile results
            results = {
                'direction_conflict': direction_conflict,
                'magnitude_conflict': magnitude_conflict,
                'conflict_intensity': conflict_intensity,
                'high_conflict': high_conflict,
                'total_prediction': total_prediction,
                'chaser_prediction': chaser_prediction,
                'anchor_prediction': causal_anchor_prediction,
                'chaser_confidence': chaser_confidence
            }
            
            # Update statistics
            if update_statistics:
                self._update_conflict_statistics(results)
            
            if self.verbose:
                n_conflicts = np.sum(high_conflict)
                conflict_rate = n_conflicts / len(chaser_prediction)
                avg_intensity = np.mean(conflict_intensity[high_conflict]) if n_conflicts > 0 else 0.0
                
                tprint_success("✅ Conflict detection complete:")
                tprint_info(f"   - High conflicts: {n_conflicts}/{len(chaser_prediction)} ({conflict_rate:.2%})")
                tprint_info(f"   - Average intensity: {avg_intensity:.3f}")
                tprint_info(f"   - Direction conflicts: {np.sum(direction_conflict)}")
                tprint_info(f"   - Magnitude conflicts: {np.sum(magnitude_conflict)}")
            
            return results
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Conflict detection failed: {e}")
            raise
    
    def _update_conflict_statistics(self, conflict_results: Dict[str, np.ndarray]):
        """Update internal conflict statistics."""
        try:
            high_conflict = conflict_results['high_conflict']
            conflict_intensity = conflict_results['conflict_intensity']
            
            # Update conflict history
            self.conflict_history_.append(np.mean(high_conflict))
            
            # Keep only recent history
            if len(self.conflict_history_) > self.statistical_window:
                self.conflict_history_ = self.conflict_history_[-self.statistical_window:]
            
            # Update statistics
            self.conflict_rate_ = np.mean(self.conflict_history_)
            
            if np.sum(high_conflict) > 0:
                self.avg_conflict_intensity_ = np.mean(conflict_intensity[high_conflict])
            else:
                self.avg_conflict_intensity_ = 0.0
                
        except Exception as e:
            if self.verbose:
                tprint_warning(f"⚠️ Failed to update conflict statistics: {e}")
    
    def get_meta_learner_signals(self) -> Dict[str, Any]:
        """
        Get signals for the Meta-Learner to use in final decisions.
        
        Returns:
            Dictionary with meta-learner integration signals
        """
        return {
            'recent_conflict_rate': self.conflict_rate_,
            'avg_conflict_intensity': self.avg_conflict_intensity_,
            'conflict_trend': np.mean(self.conflict_history_[-10:]) if len(self.conflict_history_) >= 10 else self.conflict_rate_,
            'stability_score': 1.0 - self.conflict_rate_,  # Higher = more stable
            'chaser_reliability': 1.0 - self.avg_conflict_intensity_  # Higher = more reliable
        }
    
    def generate_conflict_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive conflict analysis report.
        
        Returns:
            Dictionary with conflict report
        """
        return {
            'detector_config': {
                'direction_threshold': self.direction_threshold,
                'magnitude_threshold': self.magnitude_threshold,
                'confidence_threshold': self.confidence_threshold,
                'conflict_intensity_threshold': self.conflict_intensity_threshold
            },
            'current_statistics': {
                'conflict_rate': self.conflict_rate_,
                'avg_conflict_intensity': self.avg_conflict_intensity_,
                'history_length': len(self.conflict_history_)
            },
            'meta_learner_signals': self.get_meta_learner_signals()
        }

# Convenience functions
def quick_conflict_detection(
    chaser_prediction: np.ndarray,
    causal_anchor_prediction: np.ndarray,
    chaser_confidence: np.ndarray,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Quick conflict detection with default parameters.
    
    Args:
        chaser_prediction: Chaser predictions
        causal_anchor_prediction: Anchor predictions
        chaser_confidence: Chaser confidence
        **kwargs: Additional parameters
        
    Returns:
        Conflict detection results
    """
    detector = ConflictDetector(**kwargs)
    return detector.detect_conflicts(
        chaser_prediction, causal_anchor_prediction, chaser_confidence
    )

def analyze_conflict_opportunity(
    chaser_prediction: np.ndarray,
    causal_anchor_prediction: np.ndarray,
    returns: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Analyze whether conflicts represent opportunities or risks.
    
    Args:
        chaser_prediction: Chaser predictions
        causal_anchor_prediction: Anchor predictions
        returns: Actual returns (optional)
        
    Returns:
        Opportunity analysis metrics
    """
    detector = ConflictDetector()
    conflict_results = detector.detect_conflicts(
        chaser_prediction, causal_anchor_prediction,
        np.ones_like(chaser_prediction)  # Assume full confidence
    )
    
    patterns = detector.analyze_conflict_patterns(
        conflict_results['conflict_intensity'],
        chaser_prediction,
        causal_anchor_prediction
    )
    
    # If returns provided, analyze actual performance
    if returns is not None:
        high_conflict_mask = conflict_results['high_conflict']
        if np.sum(high_conflict_mask) > 0:
            conflict_returns = returns[high_conflict_mask]
            non_conflict_returns = returns[~high_conflict_mask]
            
            patterns.update({
                'conflict_return_mean': np.mean(conflict_returns),
                'non_conflict_return_mean': np.mean(non_conflict_returns),
                'conflict_return_std': np.std(conflict_returns),
                'non_conflict_return_std': np.std(non_conflict_returns),
                'conflict_sharpe': np.mean(conflict_returns) / (np.std(conflict_returns) + 1e-8),
                'non_conflict_sharpe': np.mean(non_conflict_returns) / (np.std(non_conflict_returns) + 1e-8)
            })
    
    return patterns
