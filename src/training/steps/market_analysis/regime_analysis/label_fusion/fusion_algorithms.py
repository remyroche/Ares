"""
Advanced label fusion algorithms.

This module provides sophisticated label fusion algorithms including consensus clustering,
Bayesian fusion, and temporal smoothing techniques.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy, mode
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import warnings

logger = logging.getLogger(__name__)


class FusionAlgorithm(Enum):
    """Available fusion algorithms."""
    MAJORITY_VOTING = "majority_voting"
    WEIGHTED_AVERAGE = "weighted_average"
    DAWID_SKENE = "dawid_skene"
    CONSENSUS_CLUSTERING = "consensus_clustering"
    BAYESIAN_FUSION = "bayesian_fusion"
    TEMPORAL_SMOOTHING = "temporal_smoothing"
    ENSEMBLE_FUSION = "ensemble_fusion"


@dataclass
class FusionMetrics:
    """Metrics for evaluating fusion quality."""
    agreement_score: float
    confidence_score: float
    stability_score: float
    diversity_score: float
    overall_quality: float


class LabelFusionEngine:
    """Advanced label fusion engine with multiple algorithms."""
    
    def __init__(self, algorithm: FusionAlgorithm = FusionAlgorithm.WEIGHTED_AVERAGE):
        """Initialize fusion engine."""
        self.algorithm = algorithm
        self.logger = logging.getLogger('LabelFusionEngine')
        
    def fuse_labels(self, labels: List[np.ndarray], 
                   weights: Optional[List[float]] = None,
                   quality_scores: Optional[List[float]] = None,
                   temporal_data: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, FusionMetrics]:
        """
        Fuse multiple label sets using the specified algorithm.
        
        Args:
            labels: List of label arrays
            weights: Optional weights for each label set
            quality_scores: Optional quality scores for weighting
            temporal_data: Optional temporal data for smoothing
            
        Returns:
            Tuple of (fused_labels, confidence_scores, fusion_metrics)
        """
        try:
            self.logger.info(f"Starting label fusion using {self.algorithm.value}")
            
            # Validate inputs
            validated_labels, valid_indices = self._validate_labels(labels)
            if len(validated_labels) == 0:
                raise ValueError("No valid labels provided")
            
            # Prepare weights
            fusion_weights = self._prepare_weights(weights, quality_scores, len(validated_labels))
            
            # Apply fusion algorithm
            if self.algorithm == FusionAlgorithm.MAJORITY_VOTING:
                fused_labels, confidence_scores = self._majority_voting(validated_labels, fusion_weights)
            elif self.algorithm == FusionAlgorithm.WEIGHTED_AVERAGE:
                fused_labels, confidence_scores = self._weighted_average(validated_labels, fusion_weights)
            elif self.algorithm == FusionAlgorithm.DAWID_SKENE:
                fused_labels, confidence_scores = self._dawid_skene(validated_labels, fusion_weights)
            elif self.algorithm == FusionAlgorithm.CONSENSUS_CLUSTERING:
                fused_labels, confidence_scores = self._consensus_clustering(validated_labels, fusion_weights)
            elif self.algorithm == FusionAlgorithm.BAYESIAN_FUSION:
                fused_labels, confidence_scores = self._bayesian_fusion(validated_labels, fusion_weights)
            elif self.algorithm == FusionAlgorithm.TEMPORAL_SMOOTHING:
                fused_labels, confidence_scores = self._temporal_smoothing(validated_labels, fusion_weights, temporal_data)
            else:  # ENSEMBLE_FUSION
                fused_labels, confidence_scores = self._ensemble_fusion(validated_labels, fusion_weights)
            
            # Calculate fusion metrics
            metrics = self._calculate_fusion_metrics(validated_labels, fused_labels, fusion_weights)
            
            self.logger.info(f"Label fusion completed. Quality: {metrics.overall_quality:.3f}")
            return fused_labels, confidence_scores, metrics
            
        except Exception as e:
            self.logger.error(f"Label fusion failed: {e}")
            raise
    
    def _validate_labels(self, labels: List[np.ndarray]) -> Tuple[List[np.ndarray], List[int]]:
        """Validate and filter label arrays."""
        validated = []
        valid_indices = []
        
        for i, label_array in enumerate(labels):
            try:
                # Convert to numpy array
                if not isinstance(label_array, np.ndarray):
                    label_array = np.array(label_array)
                
                # Check for valid data
                if len(label_array) == 0 or np.isnan(label_array).any():
                    self.logger.warning(f"Invalid labels at index {i}")
                    continue
                
                # Check for sufficient unique values
                unique_labels = np.unique(label_array)
                if len(unique_labels) < 2:
                    self.logger.warning(f"Insufficient unique labels at index {i}")
                    continue
                
                validated.append(label_array)
                valid_indices.append(i)
                
            except Exception as e:
                self.logger.warning(f"Failed to validate labels at index {i}: {e}")
                continue
        
        return validated, valid_indices
    
    def _prepare_weights(self, weights: Optional[List[float]], 
                        quality_scores: Optional[List[float]], 
                        n_labels: int) -> List[float]:
        """Prepare fusion weights."""
        if weights is not None and len(weights) == n_labels:
            return self._normalize_weights(weights)
        
        if quality_scores is not None and len(quality_scores) == n_labels:
            return self._normalize_weights(quality_scores)
        
        # Default to equal weights
        return [1.0 / n_labels] * n_labels
    
    def _normalize_weights(self, weights: List[float]) -> List[float]:
        """Normalize weights to sum to 1."""
        total = sum(weights)
        if total == 0:
            return [1.0 / len(weights)] * len(weights)
        return [w / total for w in weights]
    
    def _majority_voting(self, labels: List[np.ndarray], weights: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Perform majority voting fusion."""
        n_samples = len(labels[0])
        n_labels = len(labels)
        
        # Get all unique labels
        all_labels = set()
        for label_array in labels:
            all_labels.update(np.unique(label_array))
        all_labels = sorted(list(all_labels))
        
        # Create weighted vote matrix
        vote_matrix = np.zeros((n_samples, len(all_labels)))
        
        for i, (label_array, weight) in enumerate(zip(labels, weights)):
            for j, label in enumerate(label_array):
                if label in all_labels:
                    label_idx = all_labels.index(label)
                    vote_matrix[j, label_idx] += weight
        
        # Get majority vote
        fused_labels = np.array([all_labels[np.argmax(vote_matrix[i])] for i in range(n_samples)])
        
        # Calculate confidence as normalized max vote
        confidence_scores = np.max(vote_matrix, axis=1) / np.sum(vote_matrix, axis=1)
        
        return fused_labels, confidence_scores
    
    def _weighted_average(self, labels: List[np.ndarray], weights: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Perform weighted average fusion."""
        n_samples = len(labels[0])
        n_labels = len(labels)
        
        # Convert labels to numeric indices
        all_labels = set()
        for label_array in labels:
            all_labels.update(np.unique(label_array))
        all_labels = sorted(list(all_labels))
        label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
        
        # Calculate weighted average
        weighted_sum = np.zeros(n_samples)
        total_weight = 0.0
        
        for label_array, weight in zip(labels, weights):
            numeric_labels = np.array([label_to_idx[label] for label in label_array])
            weighted_sum += numeric_labels * weight
            total_weight += weight
        
        # Convert back to original labels
        averaged_indices = np.round(weighted_sum / total_weight).astype(int)
        averaged_indices = np.clip(averaged_indices, 0, len(all_labels) - 1)
        fused_labels = np.array([all_labels[idx] for idx in averaged_indices])
        
        # Calculate confidence based on agreement
        confidence_scores = np.ones(n_samples)
        for i in range(n_samples):
            votes = [label_to_idx[labels[j][i]] for j in range(n_labels)]
            # Confidence is inverse of variance
            if len(set(votes)) > 1:
                confidence_scores[i] = 1.0 / (1.0 + np.var(votes))
        
        return fused_labels, confidence_scores
    
    def _dawid_skene(self, labels: List[np.ndarray], weights: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Perform Dawid-Skene fusion using EM algorithm."""
        n_samples = len(labels[0])
        n_labels = len(labels)
        
        # Get unique labels
        all_labels = set()
        for label_array in labels:
            all_labels.update(np.unique(label_array))
        all_labels = sorted(list(all_labels))
        n_classes = len(all_labels)
        
        # Initialize parameters
        # True class probabilities (uniform initialization)
        class_probs = np.ones(n_classes) / n_classes
        
        # Annotator confusion matrices
        confusion_matrices = []
        for i in range(n_labels):
            # Initialize with identity matrix + noise
            cm = np.eye(n_classes) + np.random.uniform(0, 0.1, (n_classes, n_classes))
            cm = cm / cm.sum(axis=1, keepdims=True)
            confusion_matrices.append(cm)
        
        # EM algorithm
        for iteration in range(50):  # Max iterations
            # E-step: Estimate true class probabilities
            true_class_probs = np.zeros((n_samples, n_classes))
            
            for i in range(n_samples):
                for c in range(n_classes):
                    prob = class_probs[c]
                    for j in range(n_labels):
                        label_idx = all_labels.index(labels[j][i])
                        prob *= confusion_matrices[j][c, label_idx]
                    true_class_probs[i, c] = prob
            
            # Normalize
            true_class_probs = true_class_probs / true_class_probs.sum(axis=1, keepdims=True)
            
            # M-step: Update parameters
            # Update class probabilities
            class_probs = true_class_probs.mean(axis=0)
            
            # Update confusion matrices
            for j in range(n_labels):
                cm = np.zeros((n_classes, n_classes))
                for i in range(n_samples):
                    label_idx = all_labels.index(labels[j][i])
                    cm[:, label_idx] += true_class_probs[i, :]
                
                # Normalize
                cm = cm / cm.sum(axis=1, keepdims=True)
                confusion_matrices[j] = cm
        
        # Get final predictions
        fused_labels = np.array([all_labels[np.argmax(true_class_probs[i])] for i in range(n_samples)])
        confidence_scores = np.max(true_class_probs, axis=1)
        
        return fused_labels, confidence_scores
    
    def _consensus_clustering(self, labels: List[np.ndarray], weights: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Perform consensus clustering fusion."""
        n_samples = len(labels[0])
        n_labels = len(labels)
        
        # Create co-occurrence matrix
        cooccurrence = np.zeros((n_samples, n_samples))
        
        for label_array, weight in zip(labels, weights):
            # Create binary matrix for this labeling
            unique_labels = np.unique(label_array)
            binary_matrix = np.zeros((n_samples, len(unique_labels)))
            
            for i, label in enumerate(label_array):
                label_idx = np.where(unique_labels == label)[0][0]
                binary_matrix[i, label_idx] = 1
            
            # Add to co-occurrence matrix
            cooccurrence += weight * np.dot(binary_matrix, binary_matrix.T)
        
        # Normalize
        cooccurrence = cooccurrence / n_labels
        
        # Perform clustering on co-occurrence matrix
        # Use distance matrix (1 - cooccurrence)
        distance_matrix = 1 - cooccurrence
        
        # Determine number of clusters (use mode of individual clusterings)
        n_clusters_list = [len(np.unique(label_array)) for label_array in labels]
        n_clusters = int(mode(n_clusters_list)[0][0])
        
        # Perform hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
        
        fused_labels = clustering.fit_predict(distance_matrix)
        
        # Calculate confidence based on co-occurrence strength
        confidence_scores = np.zeros(n_samples)
        for i in range(n_samples):
            cluster_mask = fused_labels == fused_labels[i]
            confidence_scores[i] = np.mean(cooccurrence[i, cluster_mask])
        
        return fused_labels, confidence_scores
    
    def _bayesian_fusion(self, labels: List[np.ndarray], weights: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Perform Bayesian fusion."""
        n_samples = len(labels[0])
        n_labels = len(labels)
        
        # Get unique labels
        all_labels = set()
        for label_array in labels:
            all_labels.update(np.unique(label_array))
        all_labels = sorted(list(all_labels))
        n_classes = len(all_labels)
        
        # Prior probabilities (uniform)
        prior_probs = np.ones(n_classes) / n_classes
        
        # Calculate posterior probabilities
        posterior_probs = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            for c in range(n_classes):
                # Start with prior
                prob = prior_probs[c]
                
                # Multiply by likelihood from each annotator
                for j in range(n_labels):
                    label_idx = all_labels.index(labels[j][i])
                    # Simple likelihood: if label matches class, high probability
                    if label_idx == c:
                        prob *= (0.8 + 0.2 * weights[j])  # Weighted likelihood
                    else:
                        prob *= (0.2 * (1 - weights[j]))  # Weighted likelihood
                
                posterior_probs[i, c] = prob
        
        # Normalize
        posterior_probs = posterior_probs / posterior_probs.sum(axis=1, keepdims=True)
        
        # Get predictions
        fused_labels = np.array([all_labels[np.argmax(posterior_probs[i])] for i in range(n_samples)])
        confidence_scores = np.max(posterior_probs, axis=1)
        
        return fused_labels, confidence_scores
    
    def _temporal_smoothing(self, labels: List[np.ndarray], weights: List[float], 
                           temporal_data: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Perform temporal smoothing fusion."""
        # First perform weighted average
        fused_labels, confidence_scores = self._weighted_average(labels, weights)
        
        if temporal_data is None or len(temporal_data) != len(fused_labels):
            return fused_labels, confidence_scores
        
        # Apply temporal smoothing
        smoothed_labels = self._apply_temporal_smoothing(fused_labels, temporal_data)
        
        # Recalculate confidence
        smoothed_confidence = self._calculate_temporal_confidence(
            fused_labels, smoothed_labels, temporal_data
        )
        
        return smoothed_labels, smoothed_confidence
    
    def _apply_temporal_smoothing(self, labels: np.ndarray, temporal_data: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to labels."""
        # Sort by temporal data
        sort_indices = np.argsort(temporal_data)
        sorted_labels = labels[sort_indices]
        
        # Apply moving average smoothing
        window_size = min(5, len(labels) // 10)  # Adaptive window size
        if window_size < 2:
            return labels
        
        smoothed_labels = np.zeros_like(sorted_labels)
        
        for i in range(len(sorted_labels)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(sorted_labels), i + window_size // 2 + 1)
            
            # Get labels in window
            window_labels = sorted_labels[start_idx:end_idx]
            
            # Use mode for categorical smoothing
            if len(window_labels) > 0:
                smoothed_labels[i] = mode(window_labels)[0][0]
            else:
                smoothed_labels[i] = sorted_labels[i]
        
        # Restore original order
        original_order = np.argsort(sort_indices)
        return smoothed_labels[original_order]
    
    def _calculate_temporal_confidence(self, original_labels: np.ndarray, 
                                     smoothed_labels: np.ndarray, 
                                     temporal_data: np.ndarray) -> np.ndarray:
        """Calculate confidence scores for temporally smoothed labels."""
        # Base confidence on agreement between original and smoothed
        agreement = (original_labels == smoothed_labels).astype(float)
        
        # Add temporal stability bonus
        temporal_stability = self._calculate_temporal_stability(smoothed_labels, temporal_data)
        
        # Combine agreement and stability
        confidence_scores = 0.7 * agreement + 0.3 * temporal_stability
        
        return confidence_scores
    
    def _calculate_temporal_stability(self, labels: np.ndarray, temporal_data: np.ndarray) -> np.ndarray:
        """Calculate temporal stability of labels."""
        # Sort by temporal data
        sort_indices = np.argsort(temporal_data)
        sorted_labels = labels[sort_indices]
        
        stability_scores = np.ones(len(labels))
        
        for i in range(1, len(sorted_labels) - 1):
            # Check stability with neighbors
            prev_label = sorted_labels[i - 1]
            curr_label = sorted_labels[i]
            next_label = sorted_labels[i + 1]
            
            # Stability is based on consistency with neighbors
            if prev_label == curr_label == next_label:
                stability_scores[i] = 1.0
            elif prev_label == curr_label or curr_label == next_label:
                stability_scores[i] = 0.7
            else:
                stability_scores[i] = 0.3
        
        # Restore original order
        original_order = np.argsort(sort_indices)
        return stability_scores[original_order]
    
    def _ensemble_fusion(self, labels: List[np.ndarray], weights: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Perform ensemble fusion combining multiple methods."""
        # Use multiple fusion methods
        methods = [
            self._majority_voting,
            self._weighted_average,
            self._bayesian_fusion
        ]
        
        fused_results = []
        confidence_results = []
        
        for method in methods:
            try:
                fused, confidence = method(labels, weights)
                fused_results.append(fused)
                confidence_results.append(confidence)
            except Exception as e:
                self.logger.warning(f"Ensemble method failed: {e}")
                continue
        
        if not fused_results:
            # Fallback to weighted average
            return self._weighted_average(labels, weights)
        
        # Combine results using majority voting
        final_fused, final_confidence = self._majority_voting(
            fused_results, [1.0] * len(fused_results)
        )
        
        # Average confidence scores
        final_confidence = np.mean(confidence_results, axis=0)
        
        return final_fused, final_confidence
    
    def _calculate_fusion_metrics(self, original_labels: List[np.ndarray], 
                                fused_labels: np.ndarray, 
                                weights: List[float]) -> FusionMetrics:
        """Calculate comprehensive fusion metrics."""
        try:
            n_labels = len(original_labels)
            n_samples = len(fused_labels)
            
            # Agreement score: average pairwise agreement with fused labels
            agreement_scores = []
            for i in range(n_labels):
                agreement = np.mean(original_labels[i] == fused_labels)
                agreement_scores.append(agreement)
            agreement_score = np.mean(agreement_scores)
            
            # Confidence score: average confidence
            confidence_score = np.mean(np.ones(n_samples))  # Placeholder
            
            # Stability score: consistency across different label sets
            stability_scores = []
            for i in range(n_labels):
                for j in range(i + 1, n_labels):
                    # Calculate ARI between label sets
                    try:
                        ari = adjusted_rand_score(original_labels[i], original_labels[j])
                        stability_scores.append(ari)
                    except:
                        continue
            stability_score = np.mean(stability_scores) if stability_scores else 0.0
            
            # Diversity score: entropy of label distribution
            unique_labels, counts = np.unique(fused_labels, return_counts=True)
            probabilities = counts / np.sum(counts)
            diversity_score = entropy(probabilities) / np.log(len(unique_labels))
            
            # Overall quality: weighted combination
            overall_quality = (0.3 * agreement_score + 
                             0.2 * confidence_score + 
                             0.3 * stability_score + 
                             0.2 * diversity_score)
            
            return FusionMetrics(
                agreement_score=agreement_score,
                confidence_score=confidence_score,
                stability_score=stability_score,
                diversity_score=diversity_score,
                overall_quality=overall_quality
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate fusion metrics: {e}")
            return FusionMetrics(0.0, 0.0, 0.0, 0.0, 0.0)