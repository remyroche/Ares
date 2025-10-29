"""
Regime Temporal Metrics for ML Model Assessment

This module provides comprehensive metrics for assessing regime classification models:
1. Accuracy/Classification Metrics
2. Temporal/Stability Metrics  
3. Regime-Persistence Metrics
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats


@dataclass
class TemporalMetrics:
    """Temporal stability metrics for regime predictions."""
    mean_episode_length: float
    transition_rate: float
    short_episode_count: int
    switch_false_positive_rate: float
    entropy: float
    confidence: float


@dataclass
class RegimePersistenceMetrics:
    """Regime persistence metrics."""
    stability_index: float
    persistence_ratio: float
    lag_to_detection: float
    episode_purity: float


class RegimeTemporalMetricsCalculator:
    """
    Calculator for comprehensive regime temporal metrics.
    
    Provides metrics for:
    - Accuracy/Classification (baseline)
    - Temporal/Stability (episode length, transition rate, etc.)
    - Regime-Persistence (stability index, persistence ratio, etc.)
    """
    
    def __init__(self, min_episode_length: int = 3):
        """
        Initialize the metrics calculator.
        
        Args:
            min_episode_length: Minimum desired episode length for counting short episodes
        """
        self.min_episode_length = min_episode_length
    
    def calculate_comprehensive_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics for regime classification.
        
        Args:
            y_true: True regime labels
            y_pred: Predicted regime labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary containing all metrics
        """
        metrics = {}
        
        # 1. Accuracy/Classification Metrics
        metrics['classification'] = self._calculate_classification_metrics(y_true, y_pred, y_pred_proba)
        
        # 2. Temporal/Stability Metrics
        metrics['temporal'] = self._calculate_temporal_metrics(y_pred, y_pred_proba)
        
        # 3. Regime-Persistence Metrics
        metrics['persistence'] = self._calculate_persistence_metrics(y_true, y_pred)
        
        return metrics
    
    def _calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Calculate accuracy/classification metrics."""
        from sklearn.metrics import (
            accuracy_score, balanced_accuracy_score,
            precision_recall_fscore_support, log_loss,
            classification_report
        )
        
        metrics = {}
        
        # Basic accuracy metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        metrics['precision'] = precision_weighted
        metrics['recall'] = recall_weighted
        metrics['f1_score'] = f1_weighted
        
        # Per-class breakdown
        n_classes = len(np.unique(y_true))
        metrics['per_class'] = {
            f'class_{i}': {
                'precision': precision[i] if i < len(precision) else 0.0,
                'recall': recall[i] if i < len(recall) else 0.0,
                'f1': f1[i] if i < len(f1) else 0.0,
                'support': support[i] if i < len(support) else 0
            }
            for i in range(n_classes)
        }
        
        # Log-loss if probabilities available
        if y_pred_proba is not None:
            try:
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            except Exception:
                metrics['log_loss'] = None
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        
        return metrics
    
    def _calculate_temporal_metrics(
        self,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Calculate temporal/stability metrics.
        
        Includes:
        - Mean Episode Length (MEL)
        - Transition Rate (TR)
        - Short Episode Count
        - Switch False Positive Rate (SFPR)
        - Entropy/Confidence
        """
        n_samples = len(y_pred)
        
        # Calculate episodes (consecutive predictions of same regime)
        episodes = self._extract_episodes(y_pred)
        
        # Mean Episode Length
        episode_lengths = [ep['length'] for ep in episodes]
        mean_episode_length = np.mean(episode_lengths) if episode_lengths else 0.0
        
        # Transition Rate (number of regime switches per unit time)
        n_transitions = len(episodes) - 1  # transitions = episodes - 1
        transition_rate = n_transitions / n_samples if n_samples > 0 else 0.0
        
        # Short Episode Count (episodes shorter than minimum desired length)
        short_episode_count = sum(1 for length in episode_lengths if length < self.min_episode_length)
        
        # Switch False Positive Rate (fraction of switches that immediately revert)
        switch_false_positive_rate = self._calculate_sfpr(y_pred)
        
        # Entropy/Confidence
        entropy = None
        confidence = None
        if y_pred_proba is not None:
            entropy = self._calculate_entropy(y_pred_proba)
            confidence = np.mean(np.max(y_pred_proba, axis=1))
        
        return {
            'mean_episode_length': mean_episode_length,
            'transition_rate': transition_rate,
            'short_episode_count': short_episode_count,
            'switch_false_positive_rate': switch_false_positive_rate,
            'entropy': entropy,
            'confidence': confidence,
            'episode_lengths': episode_lengths,
            'n_episodes': len(episodes),
            'n_transitions': n_transitions
        }
    
    def _calculate_persistence_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        min_persistent_length: int = 5
    ) -> Dict[str, Any]:
        """
        Calculate regime-persistence metrics.
        
        Includes:
        - Stability Index: fraction of time spent in persistent episodes
        - Persistence Ratio: average episode length / total number of episodes
        - Lag to Detection: average time to detect true regime change
        - Episode Purity: proportion of bars in episode matching true regime
        """
        n_samples = len(y_pred)
        
        # Calculate predicted episodes
        pred_episodes = self._extract_episodes(y_pred)
        
        # Calculate true episodes
        true_episodes = self._extract_episodes(y_true)
        
        # Stability Index: fraction of time in persistent episodes (>N bars)
        persistent_episodes = [
            ep for ep in pred_episodes 
            if ep['length'] >= min_persistent_length
        ]
        persistent_bars = sum(ep['length'] for ep in persistent_episodes)
        stability_index = persistent_bars / n_samples if n_samples > 0 else 0.0
        
        # Persistence Ratio: average episode length / total episodes
        episode_lengths = [ep['length'] for ep in pred_episodes]
        mean_episode_length = np.mean(episode_lengths) if episode_lengths else 0.0
        n_episodes = len(pred_episodes)
        persistence_ratio = mean_episode_length / n_episodes if n_episodes > 0 else 0.0
        
        # Lag to Detection: average time to detect true regime change
        lag_to_detection = self._calculate_lag_to_detection(y_true, y_pred)
        
        # Episode Purity: proportion of bars in episode matching true regime
        episode_purity = self._calculate_episode_purity(y_true, y_pred, pred_episodes)
        
        return {
            'stability_index': stability_index,
            'persistence_ratio': persistence_ratio,
            'lag_to_detection': lag_to_detection,
            'episode_purity': episode_purity,
            'min_persistent_length': min_persistent_length
        }
    
    def _extract_episodes(self, y: np.ndarray) -> List[Dict[str, Any]]:
        """Extract episodes (consecutive same-regime predictions) from labels."""
        if len(y) == 0:
            return []
        
        episodes = []
        current_regime = y[0]
        start_idx = 0
        
        for i in range(1, len(y)):
            if y[i] != current_regime:
                # Episode ended
                episodes.append({
                    'regime': current_regime,
                    'start': start_idx,
                    'end': i - 1,
                    'length': i - start_idx
                })
                current_regime = y[i]
                start_idx = i
        
        # Add final episode
        episodes.append({
            'regime': current_regime,
            'start': start_idx,
            'end': len(y) - 1,
            'length': len(y) - start_idx
        })
        
        return episodes
    
    def _calculate_sfpr(self, y_pred: np.ndarray) -> float:
        """
        Calculate Switch False Positive Rate.
        
        SFPR = fraction of switches that immediately revert
        (i.e., A -> B -> A within 2 steps)
        """
        if len(y_pred) < 3:
            return 0.0
        
        switches = []
        for i in range(1, len(y_pred)):
            if y_pred[i] != y_pred[i-1]:
                switches.append(i)
        
        if len(switches) == 0:
            return 0.0
        
        false_positives = 0
        for switch_idx in switches:
            # Check if switch reverts within 2 steps
            if switch_idx + 1 < len(y_pred):
                # Check if it switches back
                if y_pred[switch_idx + 1] == y_pred[switch_idx - 1]:
                    false_positives += 1
        
        return false_positives / len(switches) if len(switches) > 0 else 0.0
    
    def _calculate_entropy(self, y_pred_proba: np.ndarray) -> float:
        """Calculate average entropy of probability distributions."""
        entropies = []
        for proba in y_pred_proba:
            # Remove zeros for log calculation
            proba_clean = proba[proba > 0]
            if len(proba_clean) > 0:
                entropy = -np.sum(proba_clean * np.log(proba_clean))
                entropies.append(entropy)
        
        return np.mean(entropies) if entropies else 0.0
    
    def _calculate_lag_to_detection(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calculate average lag to detection of true regime changes.
        
        For each true regime change, measure how many steps it takes
        for the prediction to also change.
        """
        if len(y_true) < 2 or len(y_pred) < 2:
            return 0.0
        
        # Find true regime changes
        true_changes = []
        for i in range(1, len(y_true)):
            if y_true[i] != y_true[i-1]:
                true_changes.append(i)
        
        if len(true_changes) == 0:
            return 0.0
        
        # For each true change, find when prediction changes
        lags = []
        for change_idx in true_changes:
            # Look ahead to find when prediction changes
            for lag in range(len(y_pred) - change_idx):
                if change_idx + lag < len(y_pred):
                    if y_pred[change_idx + lag] != y_pred[change_idx - 1]:
                        lags.append(lag)
                        break
        
        return np.mean(lags) if lags else 0.0
    
    def _calculate_episode_purity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        episodes: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate episode purity.
        
        Episode purity = proportion of bars in episode matching true regime
        """
        if len(episodes) == 0:
            return 0.0
        
        total_purity = 0.0
        for episode in episodes:
            start = episode['start']
            end = episode['end']
            predicted_regime = episode['regime']
            
            # Count matching bars
            matching = sum(
                1 for i in range(start, end + 1)
                if i < len(y_true) and y_true[i] == predicted_regime
            )
            episode_length = episode['length']
            purity = matching / episode_length if episode_length > 0 else 0.0
            total_purity += purity * episode_length
        
        total_bars = sum(ep['length'] for ep in episodes)
        return total_purity / total_bars if total_bars > 0 else 0.0


def calculate_temporal_smoothness_penalty(
    y_pred: np.ndarray,
    alpha: float = 0.1
) -> float:
    """
    Calculate temporal smoothness penalty.
    
    Penalizes flipping predictions across consecutive bars:
    L = L_CE + α * Σ_t 1[y_t != y_{t+1}]
    
    Args:
        y_pred: Predicted labels
        alpha: Penalty weight (default 0.1)
        
    Returns:
        Smoothness penalty value
    """
    if len(y_pred) < 2:
        return 0.0
    
    # Count transitions
    transitions = sum(1 for i in range(1, len(y_pred)) if y_pred[i] != y_pred[i-1])
    
    return alpha * transitions


def create_soft_labels(
    y_hard: np.ndarray,
    cluster_confidence: Optional[np.ndarray] = None,
    smoothing: float = 0.1
) -> np.ndarray:
    """
    Create soft labels from hard labels.
    
    Instead of 1-hot regime labels, use probability vector or cluster assignment confidence.
    Helps the model learn uncertainty.
    
    Args:
        y_hard: Hard regime labels (integers)
        cluster_confidence: Optional confidence scores for each cluster assignment
        smoothing: Label smoothing factor (default 0.1)
        
    Returns:
        Soft label probabilities (n_samples, n_classes)
    """
    n_samples = len(y_hard)
    n_classes = len(np.unique(y_hard))
    
    # Create one-hot encoding
    soft_labels = np.zeros((n_samples, n_classes))
    
    for i, label in enumerate(y_hard):
        if cluster_confidence is not None and i < len(cluster_confidence):
            # Use cluster confidence if available
            conf = cluster_confidence[i]
            soft_labels[i, int(label)] = conf
            # Distribute remaining probability uniformly
            remaining = (1.0 - conf) / (n_classes - 1)
            for j in range(n_classes):
                if j != int(label):
                    soft_labels[i, j] = remaining
        else:
            # Label smoothing: (1 - smoothing) for true class, smoothing/(n_classes-1) for others
            soft_labels[i, int(label)] = 1.0 - smoothing
            for j in range(n_classes):
                if j != int(label):
                    soft_labels[i, j] = smoothing / (n_classes - 1)
    
    return soft_labels
