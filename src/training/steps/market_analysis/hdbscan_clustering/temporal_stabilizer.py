"""
Temporal Stabilizer

This module provides temporal stability capabilities for HDBSCAN-based
regime discovery, including regime smoothing, cooldown periods, and
temporal consistency enforcement.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from scipy import signal
from scipy.ndimage import median_filter
from sklearn.metrics import adjusted_rand_score
import warnings

logger = logging.getLogger(__name__)

@dataclass
class TemporalStabilizerConfig:
    """Configuration for temporal stabilization."""
    # Stabilization parameters
    enable_stabilization: bool = True
    stabilization_method: str = 'median_filter'  # 'median_filter', 'majority_vote', 'temporal_smoothing'
    
    # Temporal constraints
    min_dwell_bars: int = 5
    cooldown_bars: int = 3
    max_transitions: Optional[int] = None
    
    # Smoothing parameters
    median_filter_size: int = 5
    majority_vote_window: int = 7
    smoothing_alpha: float = 0.3
    
    # Regime validation
    validate_regimes: bool = True
    min_regime_duration: int = 3
    max_regime_duration: Optional[int] = None
    
    # Transition handling
    handle_transitions: bool = True
    transition_smoothing: bool = True
    preserve_regime_count: bool = True
    
    # Quality metrics
    stability_metric: str = 'consistency'  # 'consistency', 'smoothness', 'stability'
    min_stability_score: float = 0.7
    
    # Validation
    validate_input: bool = True
    min_samples: int = 10

class TemporalStabilizer:
    """
    Temporal stabilizer for regime discovery.
    
    Provides temporal consistency enforcement, regime smoothing,
    and stability validation for HDBSCAN-based regime discovery.
    """
    
    def __init__(self, config: Optional[TemporalStabilizerConfig] = None):
        """
        Initialize temporal stabilizer.
        
        Args:
            config: Configuration for temporal stabilization
        """
        self.config = config or TemporalStabilizerConfig()
        self.stabilization_stats = {}
        self.original_labels = None
        self.stabilized_labels = None
        
    def stabilize_regimes(self, 
                        cluster_labels: np.ndarray,
                        features: Optional[np.ndarray] = None,
                        timestamps: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Stabilize regime labels temporally.
        
        Args:
            cluster_labels: Cluster labels to stabilize
            features: Feature matrix (optional, for validation)
            timestamps: Timestamps (optional, for temporal analysis)
            
        Returns:
            Tuple of (stabilized_labels, stabilization_info)
        """
        try:
            if not self.config.enable_stabilization:
                logger.info("Temporal stabilization disabled")
                return cluster_labels, {'stabilization_performed': False}
            
            logger.info("⏰ Starting temporal stabilization...")
            
            # Store original data
            self.original_labels = cluster_labels.copy()
            
            # Validate input
            if self.config.validate_input:
                cluster_labels = self._validate_input(cluster_labels)
            
            # Apply temporal stabilization
            stabilized_labels = self._apply_stabilization(cluster_labels)
            
            # Apply temporal constraints
            if self.config.handle_transitions:
                stabilized_labels = self._apply_temporal_constraints(stabilized_labels)
            
            # Validate stabilized regimes
            if self.config.validate_regimes:
                stabilized_labels = self._validate_regimes(stabilized_labels)
            
            # Calculate stabilization statistics
            stabilization_info = self._calculate_stabilization_stats(
                cluster_labels, stabilized_labels, features, timestamps
            )
            
            self.stabilization_stats = stabilization_info
            self.stabilized_labels = stabilized_labels
            
            logger.info(f"✅ Temporal stabilization completed. Stability score: {stabilization_info.get('stability_score', 0):.3f}")
            
            return stabilized_labels, stabilization_info
            
        except Exception as e:
            logger.error(f"❌ Temporal stabilization failed: {e}")
            return cluster_labels, {'error': str(e)}
    
    def _validate_input(self, cluster_labels: np.ndarray) -> np.ndarray:
        """Validate input cluster labels."""
        try:
            # Check minimum samples
            if len(cluster_labels) < self.config.min_samples:
                logger.warning(f"⚠️ Insufficient samples for stabilization: {len(cluster_labels)} < {self.config.min_samples}")
                return cluster_labels
            
            # Check for valid cluster labels
            unique_labels = np.unique(cluster_labels)
            if len(unique_labels) == 0:
                logger.warning("⚠️ No valid cluster labels found")
                return cluster_labels
            
            # Check for excessive noise
            noise_ratio = np.sum(cluster_labels == -1) / len(cluster_labels)
            if noise_ratio > 0.5:
                logger.warning(f"⚠️ High noise ratio: {noise_ratio:.2f}")
            
            return cluster_labels
            
        except Exception as e:
            logger.error(f"❌ Input validation failed: {e}")
            return cluster_labels
    
    def _apply_stabilization(self, cluster_labels: np.ndarray) -> np.ndarray:
        """Apply temporal stabilization method."""
        try:
            if self.config.stabilization_method == 'median_filter':
                return self._apply_median_filter(cluster_labels)
            elif self.config.stabilization_method == 'majority_vote':
                return self._apply_majority_vote(cluster_labels)
            elif self.config.stabilization_method == 'temporal_smoothing':
                return self._apply_temporal_smoothing(cluster_labels)
            else:
                logger.warning(f"⚠️ Unknown stabilization method: {self.config.stabilization_method}")
                return self._apply_median_filter(cluster_labels)
                
        except Exception as e:
            logger.error(f"❌ Stabilization application failed: {e}")
            return cluster_labels
    
    def _apply_median_filter(self, cluster_labels: np.ndarray) -> np.ndarray:
        """Apply median filter for temporal smoothing."""
        try:
            # Convert labels to float for median filter
            float_labels = cluster_labels.astype(float)
            
            # Apply median filter
            filtered_labels = median_filter(float_labels, size=self.config.median_filter_size)
            
            # Convert back to int
            stabilized_labels = filtered_labels.astype(int)
            
            logger.info(f"✅ Applied median filter with size {self.config.median_filter_size}")
            return stabilized_labels
            
        except Exception as e:
            logger.error(f"❌ Median filter application failed: {e}")
            return cluster_labels
    
    def _apply_majority_vote(self, cluster_labels: np.ndarray) -> np.ndarray:
        """Apply majority vote for temporal smoothing."""
        try:
            window_size = self.config.majority_vote_window
            stabilized_labels = cluster_labels.copy()
            
            for i in range(len(cluster_labels)):
                # Define window boundaries
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(cluster_labels), i + window_size // 2 + 1)
                
                # Get window labels
                window_labels = cluster_labels[start_idx:end_idx]
                
                # Remove noise for voting
                valid_labels = window_labels[window_labels != -1]
                
                if len(valid_labels) > 0:
                    # Find most common label
                    unique_labels, counts = np.unique(valid_labels, return_counts=True)
                    most_common_label = unique_labels[np.argmax(counts)]
                    stabilized_labels[i] = most_common_label
                else:
                    # Keep original label if no valid labels in window
                    stabilized_labels[i] = cluster_labels[i]
            
            logger.info(f"✅ Applied majority vote with window size {window_size}")
            return stabilized_labels
            
        except Exception as e:
            logger.error(f"❌ Majority vote application failed: {e}")
            return cluster_labels
    
    def _apply_temporal_smoothing(self, cluster_labels: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing using exponential moving average."""
        try:
            # Convert labels to one-hot encoding
            unique_labels = np.unique(cluster_labels)
            unique_labels = unique_labels[unique_labels != -1]  # Remove noise
            
            if len(unique_labels) == 0:
                return cluster_labels
            
            # Create one-hot encoding
            one_hot = np.zeros((len(cluster_labels), len(unique_labels)))
            for i, label in enumerate(unique_labels):
                one_hot[cluster_labels == label, i] = 1
            
            # Apply exponential moving average
            alpha = self.config.smoothing_alpha
            smoothed_one_hot = np.zeros_like(one_hot)
            smoothed_one_hot[0] = one_hot[0]
            
            for i in range(1, len(one_hot)):
                smoothed_one_hot[i] = alpha * one_hot[i] + (1 - alpha) * smoothed_one_hot[i-1]
            
            # Convert back to labels
            stabilized_labels = unique_labels[np.argmax(smoothed_one_hot, axis=1)]
            
            # Handle noise points
            noise_mask = cluster_labels == -1
            stabilized_labels[noise_mask] = -1
            
            logger.info(f"✅ Applied temporal smoothing with alpha {alpha}")
            return stabilized_labels
            
        except Exception as e:
            logger.error(f"❌ Temporal smoothing application failed: {e}")
            return cluster_labels
    
    def _apply_temporal_constraints(self, cluster_labels: np.ndarray) -> np.ndarray:
        """Apply temporal constraints (min dwell, cooldown)."""
        try:
            constrained_labels = cluster_labels.copy()
            
            # Apply minimum dwell time
            if self.config.min_dwell_bars > 1:
                constrained_labels = self._enforce_min_dwell(constrained_labels)
            
            # Apply cooldown period
            if self.config.cooldown_bars > 0:
                constrained_labels = self._enforce_cooldown(constrained_labels)
            
            # Apply maximum transitions
            if self.config.max_transitions is not None:
                constrained_labels = self._enforce_max_transitions(constrained_labels)
            
            return constrained_labels
            
        except Exception as e:
            logger.error(f"❌ Temporal constraints application failed: {e}")
            return cluster_labels
    
    def _enforce_min_dwell(self, cluster_labels: np.ndarray) -> np.ndarray:
        """Enforce minimum dwell time for regimes."""
        try:
            min_dwell = self.config.min_dwell_bars
            constrained_labels = cluster_labels.copy()
            
            i = 0
            while i < len(constrained_labels):
                current_label = constrained_labels[i]
                if current_label == -1:  # Skip noise
                    i += 1
                    continue
                
                # Find end of current regime
                j = i + 1
                while j < len(constrained_labels) and constrained_labels[j] == current_label:
                    j += 1
                
                regime_length = j - i
                
                # If regime is too short, extend it
                if regime_length < min_dwell:
                    # Extend regime to minimum length
                    end_idx = min(i + min_dwell, len(constrained_labels))
                    constrained_labels[i:end_idx] = current_label
                    i = end_idx
                else:
                    i = j
            
            logger.info(f"✅ Enforced minimum dwell time of {min_dwell} bars")
            return constrained_labels
            
        except Exception as e:
            logger.error(f"❌ Min dwell enforcement failed: {e}")
            return cluster_labels
    
    def _enforce_cooldown(self, cluster_labels: np.ndarray) -> np.ndarray:
        """Enforce cooldown period between regime changes."""
        try:
            cooldown = self.config.cooldown_bars
            constrained_labels = cluster_labels.copy()
            
            i = 0
            while i < len(constrained_labels) - 1:
                current_label = constrained_labels[i]
                next_label = constrained_labels[i + 1]
                
                # If regime changes, enforce cooldown
                if current_label != next_label and current_label != -1 and next_label != -1:
                    # Apply cooldown by keeping current regime
                    cooldown_end = min(i + cooldown + 1, len(constrained_labels))
                    constrained_labels[i:cooldown_end] = current_label
                    i = cooldown_end
                else:
                    i += 1
            
            logger.info(f"✅ Enforced cooldown period of {cooldown} bars")
            return constrained_labels
            
        except Exception as e:
            logger.error(f"❌ Cooldown enforcement failed: {e}")
            return cluster_labels
    
    def _enforce_max_transitions(self, cluster_labels: np.ndarray) -> np.ndarray:
        """Enforce maximum number of transitions."""
        try:
            max_transitions = self.config.max_transitions
            if max_transitions is None:
                return cluster_labels
            
            # Count current transitions
            transitions = np.sum(np.diff(cluster_labels) != 0)
            
            if transitions <= max_transitions:
                return cluster_labels
            
            # If too many transitions, apply additional smoothing
            constrained_labels = self._apply_median_filter(cluster_labels)
            
            # Check if we've reduced transitions enough
            new_transitions = np.sum(np.diff(constrained_labels) != 0)
            if new_transitions > max_transitions:
                # Apply more aggressive smoothing
                constrained_labels = self._apply_majority_vote(constrained_labels)
            
            logger.info(f"✅ Enforced maximum transitions limit of {max_transitions}")
            return constrained_labels
            
        except Exception as e:
            logger.error(f"❌ Max transitions enforcement failed: {e}")
            return cluster_labels
    
    def _validate_regimes(self, cluster_labels: np.ndarray) -> np.ndarray:
        """Validate stabilized regimes."""
        try:
            validated_labels = cluster_labels.copy()
            
            # Remove regimes that are too short
            if self.config.min_regime_duration > 1:
                validated_labels = self._remove_short_regimes(validated_labels)
            
            # Remove regimes that are too long
            if self.config.max_regime_duration is not None:
                validated_labels = self._remove_long_regimes(validated_labels)
            
            # Ensure minimum number of regimes
            unique_regimes = np.unique(validated_labels)
            unique_regimes = unique_regimes[unique_regimes != -1]
            
            if len(unique_regimes) < 2:
                logger.warning("⚠️ Too few regimes after validation, keeping original")
                return cluster_labels
            
            return validated_labels
            
        except Exception as e:
            logger.error(f"❌ Regime validation failed: {e}")
            return cluster_labels
    
    def _remove_short_regimes(self, cluster_labels: np.ndarray) -> np.ndarray:
        """Remove regimes that are too short."""
        try:
            min_duration = self.config.min_regime_duration
            cleaned_labels = cluster_labels.copy()
            
            i = 0
            while i < len(cleaned_labels):
                current_label = cleaned_labels[i]
                if current_label == -1:  # Skip noise
                    i += 1
                    continue
                
                # Find end of current regime
                j = i + 1
                while j < len(cleaned_labels) and cleaned_labels[j] == current_label:
                    j += 1
                
                regime_length = j - i
                
                # If regime is too short, mark as noise
                if regime_length < min_duration:
                    cleaned_labels[i:j] = -1
                
                i = j
            
            logger.info(f"✅ Removed regimes shorter than {min_duration} bars")
            return cleaned_labels
            
        except Exception as e:
            logger.error(f"❌ Short regime removal failed: {e}")
            return cluster_labels
    
    def _remove_long_regimes(self, cluster_labels: np.ndarray) -> np.ndarray:
        """Remove regimes that are too long."""
        try:
            max_duration = self.config.max_regime_duration
            if max_duration is None:
                return cluster_labels
            
            cleaned_labels = cluster_labels.copy()
            
            i = 0
            while i < len(cleaned_labels):
                current_label = cleaned_labels[i]
                if current_label == -1:  # Skip noise
                    i += 1
                    continue
                
                # Find end of current regime
                j = i + 1
                while j < len(cleaned_labels) and cleaned_labels[j] == current_label:
                    j += 1
                
                regime_length = j - i
                
                # If regime is too long, truncate it
                if regime_length > max_duration:
                    # Keep first part, mark rest as noise
                    truncate_point = i + max_duration
                    cleaned_labels[truncate_point:j] = -1
                
                i = j
            
            logger.info(f"✅ Truncated regimes longer than {max_duration} bars")
            return cleaned_labels
            
        except Exception as e:
            logger.error(f"❌ Long regime removal failed: {e}")
            return cluster_labels
    
    def _calculate_stabilization_stats(self, 
                                     original_labels: np.ndarray,
                                     stabilized_labels: np.ndarray,
                                     features: Optional[np.ndarray] = None,
                                     timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate stabilization statistics."""
        try:
            # Basic statistics
            n_samples = len(original_labels)
            changed_samples = np.sum(original_labels != stabilized_labels)
            change_ratio = changed_samples / n_samples
            
            # Regime statistics
            original_regimes = len(set(original_labels)) - (1 if -1 in original_labels else 0)
            stabilized_regimes = len(set(stabilized_labels)) - (1 if -1 in stabilized_labels else 0)
            
            # Transition statistics
            original_transitions = np.sum(np.diff(original_labels) != 0)
            stabilized_transitions = np.sum(np.diff(stabilized_labels) != 0)
            
            # Stability metrics
            stability_score = self._calculate_stability_score(original_labels, stabilized_labels)
            consistency_score = self._calculate_consistency_score(stabilized_labels)
            smoothness_score = self._calculate_smoothness_score(stabilized_labels)
            
            # Regime duration statistics
            regime_durations = self._calculate_regime_durations(stabilized_labels)
            
            stats = {
                'stabilization_performed': True,
                'n_samples': n_samples,
                'changed_samples': changed_samples,
                'change_ratio': change_ratio,
                'original_regimes': original_regimes,
                'stabilized_regimes': stabilized_regimes,
                'regime_count_change': stabilized_regimes - original_regimes,
                'original_transitions': original_transitions,
                'stabilized_transitions': stabilized_transitions,
                'transition_reduction': original_transitions - stabilized_transitions,
                'stability_score': stability_score,
                'consistency_score': consistency_score,
                'smoothness_score': smoothness_score,
                'regime_durations': regime_durations,
                'avg_regime_duration': np.mean(regime_durations) if regime_durations else 0,
                'min_regime_duration': np.min(regime_durations) if regime_durations else 0,
                'max_regime_duration': np.max(regime_durations) if regime_durations else 0
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Stabilization stats calculation failed: {e}")
            return {'error': str(e)}
    
    def _calculate_stability_score(self, original_labels: np.ndarray, stabilized_labels: np.ndarray) -> float:
        """Calculate stability score between original and stabilized labels."""
        try:
            # Use adjusted rand index for stability measurement
            return adjusted_rand_score(original_labels, stabilized_labels)
        except Exception as e:
            logger.debug(f"Stability score calculation failed: {e}")
            return 0.0
    
    def _calculate_consistency_score(self, cluster_labels: np.ndarray) -> float:
        """Calculate temporal consistency score."""
        try:
            # Calculate autocorrelation of regime changes
            regime_changes = np.diff(cluster_labels) != 0
            if len(regime_changes) < 2:
                return 0.0
            
            # Calculate autocorrelation
            autocorr = np.corrcoef(regime_changes[:-1], regime_changes[1:])[0, 1]
            return autocorr if not np.isnan(autocorr) else 0.0
            
        except Exception as e:
            logger.debug(f"Consistency score calculation failed: {e}")
            return 0.0
    
    def _calculate_smoothness_score(self, cluster_labels: np.ndarray) -> float:
        """Calculate temporal smoothness score."""
        try:
            # Calculate second derivative (acceleration) of regime changes
            regime_changes = np.diff(cluster_labels) != 0
            if len(regime_changes) < 3:
                return 0.0
            
            # Calculate second derivative
            second_deriv = np.diff(regime_changes.astype(int), n=2)
            
            # Smoothness is inverse of second derivative variance
            smoothness = 1.0 / (np.var(second_deriv) + 1e-10)
            return min(smoothness, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.debug(f"Smoothness score calculation failed: {e}")
            return 0.0
    
    def _calculate_regime_durations(self, cluster_labels: np.ndarray) -> List[int]:
        """Calculate regime durations."""
        try:
            durations = []
            i = 0
            
            while i < len(cluster_labels):
                current_label = cluster_labels[i]
                if current_label == -1:  # Skip noise
                    i += 1
                    continue
                
                # Find end of current regime
                j = i + 1
                while j < len(cluster_labels) and cluster_labels[j] == current_label:
                    j += 1
                
                regime_length = j - i
                durations.append(regime_length)
                i = j
            
            return durations
            
        except Exception as e:
            logger.debug(f"Regime duration calculation failed: {e}")
            return []
    
    def get_stabilization_stats(self) -> Dict[str, Any]:
        """Get stabilization statistics."""
        return self.stabilization_stats.copy()
    
    def get_original_labels(self) -> Optional[np.ndarray]:
        """Get original cluster labels."""
        return self.original_labels.copy() if self.original_labels is not None else None
    
    def get_stabilized_labels(self) -> Optional[np.ndarray]:
        """Get stabilized cluster labels."""
        return self.stabilized_labels.copy() if self.stabilized_labels is not None else None