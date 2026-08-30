"""
HMM Transition Modeler - Forecasting Layer for Regime Clustering

This module adds transition probability modeling and forecasting capabilities
ON TOP OF your existing regime_clustering results. It doesn't replace or modify
the clustering - it learns from the final labels to predict future transitions.

This is a COMPLEMENT to regime_clustering, not a replacement.

Author: Ares Team
Date: 2025-10-28
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    hmm = None

from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error

logger = logging.getLogger(__name__)


@dataclass
class TransitionForecast:
    """Forecast result from transition modeler."""
    current_regime: int
    next_regime_probabilities: Dict[int, float]
    most_likely_next: int
    regime_change_risk: float
    confidence: float
    expected_duration: float
    warning_level: str


class HMMTransitionModeler:
    """
    Transition probability modeler for regime clustering results.
    
    This is an ADD-ON that works with your regime_clustering output.
    It doesn't replace anything - just adds forecasting capabilities.
    
    Features:
    - Learn transition probabilities from regime labels
    - Forecast next regime changes
    - Estimate regime duration
    - Early warning system for regime changes
    - Multi-step regime forecasting
    
    Example:
        >>> # After regime_clustering completes
        >>> labels = regime_clustering_result['labels']
        >>> features = regime_clustering_result['features']
        >>> 
        >>> # Add transition modeling
        >>> transition_model = HMMTransitionModeler(n_regimes=5)
        >>> transition_model.fit(features, labels)
        >>> 
        >>> # Get forecasts
        >>> forecast = transition_model.predict_next_regime(current_regime=2)
        >>> print(f"Next regime: {forecast.most_likely_next}")
        >>> print(f"Change risk: {forecast.regime_change_risk:.2%}")
    """
    
    def __init__(self, 
                 n_regimes: int,
                 memory_window: int = 500,
                 min_regime_duration: int = 5):
        """
        Initialize transition modeler.
        
        Args:
            n_regimes: Number of regimes (from regime_clustering)
            memory_window: How many recent observations to weight more heavily
            min_regime_duration: Minimum expected regime duration (for warnings)
        """
        if not HMM_AVAILABLE:
            raise ImportError(
                "hmmlearn is required for transition modeling. "
                "Install with: pip install hmmlearn"
            )
        
        self.n_regimes = n_regimes
        self.memory_window = memory_window
        self.min_regime_duration = min_regime_duration
        
        # Initialize HMM for transition modeling only
        # Using 'diag' covariance for speed - we care about transitions, not emissions
        self.hmm = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type='diag',
            n_iter=50,  # Fewer iterations - learning from good labels
            random_state=42,
            verbose=0
        )
        
        self.is_fitted = False
        self.transition_matrix = None
        self.regime_durations = None
        self.regime_frequencies = None
        
        self.logger = logging.getLogger(__name__)
    
    def fit(self, features: np.ndarray, regime_labels: np.ndarray) -> 'HMMTransitionModeler':
        """
        Learn transition patterns from regime_clustering results.
        
        This doesn't modify your regime labels - it just learns the transition
        patterns to enable forecasting.
        
        Args:
            features: Feature matrix used for clustering
            regime_labels: Final regime labels from regime_clustering
            
        Returns:
            self for chaining
        """
        tprint_info(f"🔮 Learning transition patterns for {self.n_regimes} regimes...")
        
        try:
            # Validate inputs
            if len(features) != len(regime_labels):
                raise ValueError("Features and labels must have same length")
            
            if len(np.unique(regime_labels)) != self.n_regimes:
                tprint_warning(
                    f"Warning: Expected {self.n_regimes} regimes, "
                    f"found {len(np.unique(regime_labels))}"
                )
            
            # Initialize HMM from regime_clustering results
            self._initialize_from_labels(features, regime_labels)
            
            # Fit to learn transition dynamics
            # This refines the transition probabilities based on feature patterns
            self.hmm.fit(features)
            
            # Extract transition matrix
            self.transition_matrix = self.hmm.transmat_
            
            # Calculate regime duration statistics
            self.regime_durations = self._calculate_regime_durations(regime_labels)
            
            # Calculate regime frequencies
            unique, counts = np.unique(regime_labels, return_counts=True)
            self.regime_frequencies = dict(zip(unique, counts / len(regime_labels)))
            
            self.is_fitted = True
            
            tprint_success("✅ Transition modeling complete!")
            
            # Log useful insights
            self._log_transition_insights()
            
            return self
            
        except Exception as e:
            tprint_error(f"Failed to fit transition model: {e}")
            raise
    
    def _initialize_from_labels(self, features: np.ndarray, labels: np.ndarray):
        """Initialize HMM from regime_clustering labels."""
        # Start probabilities (from label distribution)
        unique, counts = np.unique(labels, return_counts=True)
        start_probs = np.zeros(self.n_regimes)
        for label, count in zip(unique, counts):
            start_probs[label] = count
        start_probs += 1e-10  # Smoothing
        self.hmm.startprob_ = start_probs / start_probs.sum()
        
        # Transition matrix (from observed transitions)
        trans = np.zeros((self.n_regimes, self.n_regimes))
        for i in range(len(labels) - 1):
            trans[labels[i], labels[i+1]] += 1
        
        # Add smoothing to prevent zero probabilities
        trans += 0.01
        self.hmm.transmat_ = trans / trans.sum(axis=1, keepdims=True)
        
        # Emission parameters (means and covariances for each regime)
        self.hmm.means_ = np.array([
            features[labels == k].mean(axis=0) if (labels == k).any() 
            else np.zeros(features.shape[1])
            for k in range(self.n_regimes)
        ])
        
        self.hmm.covars_ = np.array([
            features[labels == k].var(axis=0) + 1e-6 if (labels == k).any()
            else np.ones(features.shape[1]) * 1e-6
            for k in range(self.n_regimes)
        ])
    
    def _calculate_regime_durations(self, labels: np.ndarray) -> Dict[int, Dict[str, float]]:
        """Calculate how long each regime typically lasts."""
        durations = {k: [] for k in range(self.n_regimes)}
        
        current_regime = labels[0]
        duration = 1
        
        for i in range(1, len(labels)):
            if labels[i] == current_regime:
                duration += 1
            else:
                durations[current_regime].append(duration)
                current_regime = labels[i]
                duration = 1
        
        # Add final duration
        durations[current_regime].append(duration)
        
        # Calculate statistics
        stats = {}
        for regime, dur_list in durations.items():
            if dur_list:
                stats[regime] = {
                    'mean': float(np.mean(dur_list)),
                    'std': float(np.std(dur_list)),
                    'median': float(np.median(dur_list)),
                    'min': float(np.min(dur_list)),
                    'max': float(np.max(dur_list)),
                    'count': len(dur_list)
                }
            else:
                stats[regime] = {
                    'mean': 0.0, 'std': 0.0, 'median': 0.0,
                    'min': 0.0, 'max': 0.0, 'count': 0
                }
        
        return stats
    
    def predict_next_regime(self, 
                           current_regime: int,
                           current_features: Optional[np.ndarray] = None) -> TransitionForecast:
        """
        Predict next regime and transition probabilities.
        
        Args:
            current_regime: Current regime ID
            current_features: Optional current feature vector (for more accurate prediction)
            
        Returns:
            TransitionForecast with probabilities and analysis
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Get transition probabilities from transition matrix
        trans_probs = self.transition_matrix[current_regime]
        
        # If current features provided, refine prediction
        if current_features is not None:
            # Use HMM to get regime probabilities given features
            posterior_probs = self.hmm.predict_proba(current_features.reshape(1, -1))[0]
            # Combine with transition probabilities
            trans_probs = 0.7 * trans_probs + 0.3 * posterior_probs
            trans_probs /= trans_probs.sum()
        
        # Most likely next regime
        most_likely = int(np.argmax(trans_probs))
        
        # Probability of regime change
        change_prob = float(1.0 - trans_probs[current_regime])
        
        # Confidence in prediction
        confidence = float(trans_probs[most_likely])
        
        # Expected duration in current regime
        expected_duration = self._get_expected_duration(current_regime)
        
        # Warning level
        warning_level = self._get_warning_level(change_prob, expected_duration)
        
        return TransitionForecast(
            current_regime=current_regime,
            next_regime_probabilities={
                k: float(trans_probs[k])
                for k in range(self.n_regimes)
            },
            most_likely_next=most_likely,
            regime_change_risk=change_prob,
            confidence=confidence,
            expected_duration=expected_duration,
            warning_level=warning_level
        )
    
    def _get_expected_duration(self, regime: int) -> float:
        """Get expected duration for regime."""
        p_stay = self.transition_matrix[regime, regime]
        if p_stay < 1.0:
            return 1.0 / (1.0 - p_stay)
        return float('inf')
    
    def _get_warning_level(self, change_prob: float, expected_duration: float) -> str:
        """Determine warning level based on change probability and duration."""
        if change_prob > 0.7 or expected_duration < self.min_regime_duration:
            return 'CRITICAL'
        elif change_prob > 0.5 or expected_duration < self.min_regime_duration * 2:
            return 'HIGH'
        elif change_prob > 0.3:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def forecast_regime_sequence(self, 
                                 current_regime: int,
                                 n_steps: int = 10,
                                 n_simulations: int = 1000) -> Dict[str, Any]:
        """
        Forecast regime sequence for next N steps using Monte Carlo simulation.
        
        Args:
            current_regime: Current regime ID
            n_steps: Number of steps to forecast
            n_simulations: Number of Monte Carlo simulations
            
        Returns:
            Dictionary with forecast sequence and confidence
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        tprint_info(f"Forecasting regime sequence for {n_steps} steps...")
        
        # Monte Carlo simulation for forecast
        simulations = []
        
        for _ in range(n_simulations):
            sequence = []
            regime = current_regime
            
            for step in range(n_steps):
                # Sample next regime from transition probabilities
                trans_probs = self.transition_matrix[regime]
                regime = np.random.choice(self.n_regimes, p=trans_probs)
                sequence.append(regime)
            
            simulations.append(sequence)
        
        # Analyze simulations
        simulations = np.array(simulations)
        
        # Most likely sequence (mode at each timestep)
        forecast_sequence = []
        confidence_by_step = []
        regime_distribution_by_step = []
        
        for step in range(n_steps):
            step_regimes = simulations[:, step]
            unique, counts = np.unique(step_regimes, return_counts=True)
            most_common = unique[np.argmax(counts)]
            confidence = counts.max() / n_simulations
            
            # Distribution at this timestep
            distribution = {int(u): float(c / n_simulations) for u, c in zip(unique, counts)}
            
            forecast_sequence.append(int(most_common))
            confidence_by_step.append(float(confidence))
            regime_distribution_by_step.append(distribution)
        
        # Detect regime change points
        change_points = []
        for i in range(len(forecast_sequence) - 1):
            if forecast_sequence[i] != forecast_sequence[i + 1]:
                change_points.append(i + 1)
        
        # Calculate forecast quality metrics
        avg_confidence = float(np.mean(confidence_by_step))
        confidence_decay = float(confidence_by_step[0] - confidence_by_step[-1]) if len(confidence_by_step) > 1 else 0.0
        
        return {
            'forecast_sequence': forecast_sequence,
            'confidence_by_step': confidence_by_step,
            'regime_distribution_by_step': regime_distribution_by_step,
            'regime_change_points': change_points,
            'n_regime_changes': len(change_points),
            'forecast_horizon': n_steps,
            'average_confidence': avg_confidence,
            'confidence_decay': confidence_decay,
            'forecast_quality': 'HIGH' if avg_confidence > 0.7 else 'MEDIUM' if avg_confidence > 0.5 else 'LOW'
        }
    
    def regime_change_warning(self,
                            recent_features: np.ndarray,
                            current_regime: int,
                            window: int = 50) -> Dict[str, Any]:
        """
        Generate early warning for regime changes based on recent observations.
        
        Args:
            recent_features: Recent feature observations (last N timesteps)
            current_regime: Current regime ID
            window: Number of recent observations to analyze
            
        Returns:
            Dictionary with warning level, change probability, and evidence
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Use last 'window' observations
        if len(recent_features) > window:
            recent_features = recent_features[-window:]
        
        # Calculate regime probabilities for recent observations
        posteriors = self.hmm.predict_proba(recent_features)
        
        # Analyze probability trend for current regime
        current_regime_probs = posteriors[:, current_regime]
        
        # Fit linear trend
        x = np.arange(len(current_regime_probs))
        prob_trend = np.polyfit(x, current_regime_probs, deg=1)[0]
        
        # Feature drift (distance from regime centroid)
        regime_centroid = self.hmm.means_[current_regime]
        recent_centroid = recent_features.mean(axis=0)
        feature_drift = np.linalg.norm(recent_centroid - regime_centroid)
        
        # Normalize drift by typical feature scale
        typical_scale = np.linalg.norm(self.hmm.means_.std(axis=0))
        normalized_drift = feature_drift / (typical_scale + 1e-6)
        
        # Transition momentum (moving toward another regime?)
        other_regime_probs = posteriors[:, [i for i in range(self.n_regimes) if i != current_regime]]
        if other_regime_probs.shape[1] > 0:
            max_other_prob = other_regime_probs.max(axis=1)
            transition_momentum = float(max_other_prob[-10:].mean())  # Last 10 observations
            most_likely_next = int(np.argmax(posteriors[-1]))
            if most_likely_next == current_regime and other_regime_probs.shape[1] > 0:
                # If still in current regime, find next most likely
                probs = posteriors[-1].copy()
                probs[current_regime] = 0
                most_likely_next = int(np.argmax(probs))
        else:
            transition_momentum = 0.0
            most_likely_next = current_regime
        
        # Overall change probability
        change_prob = float(1.0 - current_regime_probs[-1])
        
        # Warning level based on multiple factors
        if change_prob > 0.7 and prob_trend < -0.01:
            warning_level = 'CRITICAL'
        elif change_prob > 0.5 or (normalized_drift > 2.0 and prob_trend < 0):
            warning_level = 'HIGH'
        elif change_prob > 0.3 or normalized_drift > 1.5:
            warning_level = 'MEDIUM'
        else:
            warning_level = 'LOW'
        
        return {
            'warning_level': warning_level,
            'change_probability': change_prob,
            'most_likely_next_regime': most_likely_next,
            'evidence': {
                'feature_drift': float(normalized_drift),
                'transition_momentum': transition_momentum,
                'probability_trend': float(prob_trend),  # Negative = declining
                'recent_stability': float(current_regime_probs[-10:].mean() if len(current_regime_probs) >= 10 else current_regime_probs.mean()),
                'current_regime_confidence': float(current_regime_probs[-1])
            },
            'recommended_action': self._get_recommended_action(warning_level),
            'analysis_window': len(recent_features)
        }
    
    def _get_recommended_action(self, warning_level: str) -> str:
        """Get recommended action based on warning level."""
        actions = {
            'LOW': 'MAINTAIN_POSITIONS',
            'MEDIUM': 'MONITOR_CLOSELY',
            'HIGH': 'REDUCE_EXPOSURE',
            'CRITICAL': 'PREPARE_EXIT'
        }
        return actions.get(warning_level, 'MONITOR_CLOSELY')
    
    def get_regime_stability_score(self, regime: int) -> float:
        """
        Get stability score for a regime (0-1, higher = more stable).
        
        Args:
            regime: Regime ID
            
        Returns:
            Stability score
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Factors for stability:
        # 1. Self-transition probability (higher = more stable)
        p_stay = self.transition_matrix[regime, regime]
        
        # 2. Expected duration (longer = more stable)
        expected_dur = self._get_expected_duration(regime)
        if expected_dur == float('inf'):
            duration_score = 1.0
        else:
            # Normalize to 0-1 range (assuming 100 timesteps is very stable)
            duration_score = min(1.0, expected_dur / 100.0)
        
        # 3. Historical duration consistency (lower std = more stable)
        dur_stats = self.regime_durations[regime]
        if dur_stats['mean'] > 0:
            consistency_score = 1.0 - min(1.0, dur_stats['std'] / dur_stats['mean'])
        else:
            consistency_score = 0.0
        
        # Weighted combination
        stability_score = (
            0.5 * p_stay +
            0.3 * duration_score +
            0.2 * consistency_score
        )
        
        return float(stability_score)
    
    def _log_transition_insights(self):
        """Log interesting transition patterns."""
        tprint_info("\n📊 Transition Insights:")
        
        # Find most and least stable regimes
        stabilities = [(i, self.get_regime_stability_score(i)) for i in range(self.n_regimes)]
        stabilities.sort(key=lambda x: x[1], reverse=True)
        
        tprint_info("\nRegime Stability Ranking:")
        for regime, score in stabilities:
            p_stay = self.transition_matrix[regime, regime]
            expected_dur = self._get_expected_duration(regime)
            freq = self.regime_frequencies.get(regime, 0)
            
            dur_str = f"{expected_dur:.1f}" if expected_dur != float('inf') else "∞"
            tprint_info(
                f"  Regime {regime}: Stability={score:.3f}, "
                f"P(stay)={p_stay:.3f}, Duration={dur_str}, "
                f"Frequency={freq:.1%}"
            )
        
        # Find most likely transitions
        tprint_info("\nMost Likely Regime Transitions:")
        transitions = []
        for i in range(self.n_regimes):
            trans_probs = self.transition_matrix[i].copy()
            trans_probs[i] = 0  # Exclude self-transition
            if trans_probs.max() > 0.1:  # Only show significant transitions
                j = np.argmax(trans_probs)
                transitions.append((i, j, trans_probs[j]))
        
        transitions.sort(key=lambda x: x[2], reverse=True)
        for i, j, prob in transitions[:5]:  # Show top 5
            tprint_info(f"  Regime {i} → Regime {j}: {prob:.3f}")
    
    def get_transition_matrix(self) -> np.ndarray:
        """Get the learned transition matrix."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.transition_matrix.copy()
    
    def get_regime_duration_stats(self, regime: int) -> Dict[str, float]:
        """Get duration statistics for a specific regime."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.regime_durations[regime].copy()
    
    def save(self, path: str):
        """Save the transition model."""
        import pickle
        tprint_info(f"Saving transition model to {path}")
        with open(path, 'wb') as f:
            pickle.dump({
                'hmm': self.hmm,
                'transition_matrix': self.transition_matrix,
                'regime_durations': self.regime_durations,
                'regime_frequencies': self.regime_frequencies,
                'n_regimes': self.n_regimes,
                'memory_window': self.memory_window,
                'min_regime_duration': self.min_regime_duration,
                'is_fitted': self.is_fitted
            }, f)
        tprint_success(f"✅ Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'HMMTransitionModeler':
        """Load a saved transition model."""
        import pickle
        tprint_info(f"Loading transition model from {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(
            n_regimes=data['n_regimes'],
            memory_window=data.get('memory_window', 500),
            min_regime_duration=data.get('min_regime_duration', 5)
        )
        model.hmm = data['hmm']
        model.transition_matrix = data['transition_matrix']
        model.regime_durations = data['regime_durations']
        model.regime_frequencies = data.get('regime_frequencies', {})
        model.is_fitted = data['is_fitted']
        
        tprint_success("✅ Model loaded successfully")
        return model


# Helper function for easy integration
async def add_transition_modeling(regime_clustering_result: Dict[str, Any],
                                 features_df: pd.DataFrame,
                                 config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add transition modeling to regime_clustering results.
    
    This is a helper function to easily integrate transition modeling
    into your existing pipeline.
    
    Args:
        regime_clustering_result: Output from regime_clustering step
        features_df: Features used for clustering
        config: Configuration dictionary
        
    Returns:
        Enhanced result with transition modeling
    """
    try:
        # Extract regime information
        labels = regime_clustering_result.get('labels')
        n_regimes = regime_clustering_result.get('n_clusters')
        
        if labels is None or n_regimes is None:
            tprint_warning("Could not extract regime info, skipping transition modeling")
            return regime_clustering_result
        
        # Create and fit transition modeler
        transition_model = HMMTransitionModeler(
            n_regimes=n_regimes,
            memory_window=config.get('transition_model_memory_window', 500),
            min_regime_duration=config.get('min_regime_duration', 5)
        )
        
        transition_model.fit(features_df.values, labels)
        
        # Add transition analysis to result
        current_regime = int(labels[-1])
        
        regime_clustering_result['transition_model'] = transition_model
        regime_clustering_result['transition_matrix'] = transition_model.get_transition_matrix()
        regime_clustering_result['current_regime_forecast'] = transition_model.predict_next_regime(current_regime)
        regime_clustering_result['regime_stability_scores'] = {
            i: transition_model.get_regime_stability_score(i)
            for i in range(n_regimes)
        }
        
        return regime_clustering_result
        
    except Exception as e:
        tprint_error(f"Failed to add transition modeling: {e}")
        # Return original result if transition modeling fails
        return regime_clustering_result
