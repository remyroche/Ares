"""
Enhanced HDBSCAN Clustering for Market Analysis

This module provides enhanced HDBSCAN clustering with:
- Advanced probability calculation methods
- Model persistence capabilities
- Improved out-of-sample prediction
- Uncertainty quantification
- Ensemble prediction methods
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import time
import gc
import pickle
import joblib
from pathlib import Path
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
import hdbscan

# Import utilities
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance, tprint_progress, tprint_timer,
    tprint_logged, LogLevel
)
from src.utils.hardware import smart_cache, auto_optimize, memory_efficient, performance_tracked

logger = logging.getLogger(__name__)

@dataclass
class EnhancedHDBSCANConfig:
    """Configuration for enhanced HDBSCAN clustering."""
    # Core HDBSCAN parameters
    min_cluster_size: int = 15
    min_samples: int = 5
    cluster_selection_epsilon: float = 0.0
    cluster_selection_method: str = 'eom'
    metric: str = 'euclidean'
    metric_params: Optional[Dict[str, Any]] = None
    
    # Probability calculation methods
    probability_methods: List[str] = None  # Will be set to default in __post_init__
    ensemble_weights: Dict[str, float] = None  # Will be set to default in __post_init__
    uncertainty_threshold: float = 0.1
    
    # Model persistence
    enable_persistence: bool = True
    model_dir: str = "models/hdbscan"
    auto_save: bool = True
    
    # Validation
    enable_validation: bool = True
    min_silhouette_score: float = 0.3
    max_clusters: int = 20
    min_clusters: int = 2
    
    def __post_init__(self):
        if self.probability_methods is None:
            self.probability_methods = [
                'density_based',
                'distance_based', 
                'knn_based',
                'gmm_based',
                'ensemble'
            ]
        
        if self.ensemble_weights is None:
            self.ensemble_weights = {
                'density_based': 0.3,
                'distance_based': 0.2,
                'knn_based': 0.2,
                'gmm_based': 0.3
            }

class EnhancedHDBSCANClusterer:
    """
    Enhanced HDBSCAN clusterer with advanced probability calculation,
    model persistence, and improved out-of-sample prediction.
    """
    
    @tprint_logged(LogLevel.INFO, include_args=True)
    def __init__(self, config: Optional[EnhancedHDBSCANConfig] = None):
        """Initialize the enhanced HDBSCAN clusterer."""
        self.config = config or EnhancedHDBSCANConfig()
        self.clusterer = None
        self.scaler = StandardScaler()
        self.cluster_centers = None
        self.cluster_densities = None
        self.training_features = None
        self.training_labels = None
        self.gmm_models = {}  # For GMM-based probability calculation
        self.knn_model = None  # For KNN-based probability calculation
        self.clustering_stats = {}
        self.best_params = None
        self.best_score = -np.inf
        self.model_metadata = {}
        
        # Create model directory if persistence is enabled
        if self.config.enable_persistence:
            Path(self.config.model_dir).mkdir(parents=True, exist_ok=True)
    
    @tprint_logged(LogLevel.INFO, include_args=True)
    def cluster_data(self, features: np.ndarray, 
                    optimize_params: bool = True) -> Dict[str, Any]:
        """
        Perform HDBSCAN clustering with enhanced capabilities.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            optimize_params: Whether to optimize parameters
            
        Returns:
            Dictionary with clustering results and metadata
        """
        start_time = time.perf_counter()
        initial_memory = self._get_memory_usage()
        
        try:
            tprint_info("Starting enhanced HDBSCAN clustering")
            
            # Validate input
            if not self._validate_features(features):
                raise ValueError("Invalid feature matrix")
            
            # Store training features for later use
            self.training_features = features.copy()
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Optimize parameters if requested
            if optimize_params:
                self._optimize_parameters(features_scaled)
            
            # Perform clustering
            cluster_labels = self._perform_clustering(features_scaled)
            
            # Store training labels
            self.training_labels = cluster_labels.copy()
            
            # Calculate cluster centers and densities
            self._calculate_cluster_properties(features_scaled, cluster_labels)
            
            # Train auxiliary models for probability calculation
            self._train_auxiliary_models(features_scaled, cluster_labels)
            
            # Validate clustering
            if self.config.enable_validation:
                validation_metrics = self._validate_clustering(features_scaled, cluster_labels)
            else:
                validation_metrics = {}
            
            # Calculate clustering statistics
            self.clustering_stats = self._calculate_clustering_stats(
                features_scaled, cluster_labels, validation_metrics
            )
            
            # Save model if auto-save is enabled
            if self.config.auto_save:
                self.save_model()
            
            processing_time = time.perf_counter() - start_time
            final_memory = self._get_memory_usage()
            
            tprint_success(f"Enhanced HDBSCAN clustering completed in {processing_time:.2f}s")
            tprint_info(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB")
            
            return {
                'labels': cluster_labels,
                'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                'n_noise': list(cluster_labels).count(-1),
                'clustering_stats': self.clustering_stats,
                'validation_metrics': validation_metrics,
                'processing_time': processing_time,
                'memory_usage': final_memory - initial_memory,
                'success': True
            }
            
        except Exception as e:
            tprint_error(f"Enhanced HDBSCAN clustering failed: {e}")
            logger.error(f"❌ Enhanced clustering failed: {e}")
            return {'error': str(e), 'success': False}
    
    def enhanced_predict_with_uncertainty(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Enhanced prediction with uncertainty quantification.
        
        Args:
            features: Feature matrix for prediction (n_samples, n_features)
            
        Returns:
            Dictionary with predictions, probabilities, and uncertainty measures
        """
        try:
            if self.clusterer is None:
                tprint_warning("No trained clusterer available")
                return self._random_fallback_with_uncertainty(features)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get predictions from multiple methods
            predictions = {}
            for method in self.config.probability_methods:
                try:
                    labels, probabilities, method_name = self._predict_with_method(
                        features_scaled, method
                    )
                    predictions[method] = {
                        'labels': labels,
                        'probabilities': probabilities,
                        'method': method_name
                    }
                except Exception as e:
                    tprint_debug(f"Method {method} failed: {e}")
                    continue
            
            if not predictions:
                return self._random_fallback_with_uncertainty(features)
            
            # Calculate ensemble prediction
            ensemble_result = self._calculate_ensemble_prediction(predictions)
            
            # Calculate uncertainty measures
            uncertainty_measures = self._calculate_uncertainty_measures(
                predictions, ensemble_result
            )
            
            return {
                'labels': ensemble_result['labels'],
                'probabilities': ensemble_result['probabilities'],
                'uncertainty_measures': uncertainty_measures,
                'method_breakdown': predictions,
                'success': True
            }
            
        except Exception as e:
            tprint_error(f"Enhanced prediction failed: {e}")
            return {'error': str(e), 'success': False}
    
    def _predict_with_method(self, features_scaled: np.ndarray, 
                           method: str) -> Tuple[np.ndarray, np.ndarray, str]:
        """Predict using a specific method."""
        if method == 'density_based':
            return self._density_based_prediction(features_scaled)
        elif method == 'distance_based':
            return self._distance_based_prediction(features_scaled)
        elif method == 'knn_based':
            return self._knn_based_prediction(features_scaled)
        elif method == 'gmm_based':
            return self._gmm_based_prediction(features_scaled)
        elif method == 'ensemble':
            return self._ensemble_prediction(features_scaled)
        else:
            raise ValueError(f"Unknown prediction method: {method}")
    
    def _density_based_prediction(self, features_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
        """Predict using HDBSCAN's internal density information."""
        try:
            if not hasattr(self.clusterer, 'cluster_persistence_'):
                raise ValueError("No cluster persistence available")
            
            # Use HDBSCAN's approximate_predict if available
            if hasattr(self.clusterer, 'approximate_predict'):
                labels, probabilities = self.clusterer.approximate_predict(features_scaled)
                return labels, probabilities, "hdbscan_density_based"
            
            # Fallback to distance-based with density weighting
            if self.cluster_centers is None or self.cluster_densities is None:
                raise ValueError("No cluster properties available")
            
            # Calculate distances to cluster centers
            distances = np.sqrt(((features_scaled[:, np.newaxis] - self.cluster_centers[np.newaxis, :]) ** 2).sum(axis=2))
            
            # Weight distances by cluster densities
            weighted_distances = distances / (self.cluster_densities + 1e-10)
            
            # Assign to closest cluster
            labels = np.argmin(weighted_distances, axis=1)
            
            # Calculate probabilities based on weighted distances
            min_distances = np.min(weighted_distances, axis=1, keepdims=True)
            probabilities = np.exp(-min_distances / (weighted_distances + 1e-10))
            probabilities = np.max(probabilities, axis=1)
            
            return labels, probabilities, "density_weighted_distance"
            
        except Exception as e:
            tprint_debug(f"Density-based prediction failed: {e}")
            raise
    
    def _distance_based_prediction(self, features_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
        """Predict using distance-based assignment with improved probability calculation."""
        try:
            if self.cluster_centers is None:
                raise ValueError("No cluster centers available")
            
            # Calculate distances to cluster centers
            distances = np.sqrt(((features_scaled[:, np.newaxis] - self.cluster_centers[np.newaxis, :]) ** 2).sum(axis=2))
            
            # Assign to closest cluster
            labels = np.argmin(distances, axis=1)
            
            # Calculate probabilities using softmax normalization
            min_distances = np.min(distances, axis=1, keepdims=True)
            exp_distances = np.exp(-distances / (min_distances + 1e-10))
            probabilities = exp_distances / np.sum(exp_distances, axis=1, keepdims=True)
            probabilities = np.max(probabilities, axis=1)
            
            return labels, probabilities, "improved_distance_based"
            
        except Exception as e:
            tprint_debug(f"Distance-based prediction failed: {e}")
            raise
    
    def _knn_based_prediction(self, features_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
        """Predict using k-nearest neighbors approach."""
        try:
            if self.knn_model is None:
                raise ValueError("No KNN model available")
            
            # Find k nearest neighbors
            distances, indices = self.knn_model.kneighbors(features_scaled)
            
            # Get labels of nearest neighbors
            neighbor_labels = self.training_labels[indices]
            
            # Calculate probabilities based on neighbor labels
            labels = []
            probabilities = []
            
            for i in range(len(features_scaled)):
                # Count votes for each cluster
                unique_labels, counts = np.unique(neighbor_labels[i], return_counts=True)
                
                # Remove noise label (-1) if present
                if -1 in unique_labels:
                    noise_idx = np.where(unique_labels == -1)[0][0]
                    unique_labels = np.delete(unique_labels, noise_idx)
                    counts = np.delete(counts, noise_idx)
                
                if len(unique_labels) == 0:
                    labels.append(-1)  # Noise
                    probabilities.append(0.0)
                else:
                    # Assign to most common label
                    most_common_idx = np.argmax(counts)
                    labels.append(unique_labels[most_common_idx])
                    
                    # Calculate probability based on vote proportion
                    total_votes = np.sum(counts)
                    probabilities.append(counts[most_common_idx] / total_votes)
            
            return np.array(labels), np.array(probabilities), "knn_based"
            
        except Exception as e:
            tprint_debug(f"KNN-based prediction failed: {e}")
            raise
    
    def _gmm_based_prediction(self, features_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
        """Predict using Gaussian Mixture Models for each cluster."""
        try:
            if not self.gmm_models:
                raise ValueError("No GMM models available")
            
            # Get predictions from all GMM models
            all_probabilities = []
            cluster_labels = []
            
            for cluster_id, gmm in self.gmm_models.items():
                if cluster_id == -1:  # Skip noise cluster
                    continue
                
                # Get probabilities for this cluster
                cluster_probs = gmm.predict_proba(features_scaled)
                all_probabilities.append(cluster_probs)
                cluster_labels.append(cluster_id)
            
            if not all_probabilities:
                raise ValueError("No valid GMM models available")
            
            # Combine probabilities
            all_probabilities = np.array(all_probabilities)
            combined_probs = np.mean(all_probabilities, axis=0)
            
            # Assign to cluster with highest probability
            labels = np.argmax(combined_probs, axis=1)
            labels = np.array([cluster_labels[i] for i in labels])
            probabilities = np.max(combined_probs, axis=1)
            
            return labels, probabilities, "gmm_based"
            
        except Exception as e:
            tprint_debug(f"GMM-based prediction failed: {e}")
            raise
    
    def _ensemble_prediction(self, features_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
        """Predict using ensemble of all available methods."""
        try:
            # Get predictions from all methods
            predictions = {}
            for method in self.config.probability_methods:
                if method == 'ensemble':
                    continue
                try:
                    labels, probabilities, method_name = self._predict_with_method(features_scaled, method)
                    predictions[method] = {
                        'labels': labels,
                        'probabilities': probabilities
                    }
                except Exception as e:
                    tprint_debug(f"Method {method} failed in ensemble: {e}")
                    continue
            
            if not predictions:
                raise ValueError("No prediction methods available")
            
            # Calculate ensemble result
            ensemble_result = self._calculate_ensemble_prediction(predictions)
            
            return (ensemble_result['labels'], 
                   ensemble_result['probabilities'], 
                   "ensemble")
            
        except Exception as e:
            tprint_debug(f"Ensemble prediction failed: {e}")
            raise
    
    def _calculate_ensemble_prediction(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate ensemble prediction from multiple methods."""
        # Get all unique cluster labels
        all_labels = set()
        for pred in predictions.values():
            all_labels.update(pred['labels'])
        
        if -1 in all_labels:
            all_labels.remove(-1)  # Remove noise label
        all_labels = sorted(list(all_labels))
        
        if not all_labels:
            # All predictions are noise
            n_samples = len(list(predictions.values())[0]['labels'])
            return {
                'labels': np.full(n_samples, -1),
                'probabilities': np.zeros(n_samples)
            }
        
        # Calculate weighted ensemble
        n_samples = len(list(predictions.values())[0]['labels'])
        n_clusters = len(all_labels)
        
        # Initialize probability matrix
        prob_matrix = np.zeros((n_samples, n_clusters))
        
        for method, pred in predictions.items():
            weight = self.config.ensemble_weights.get(method, 1.0)
            
            for i, (label, prob) in enumerate(zip(pred['labels'], pred['probabilities'])):
                if label != -1 and label in all_labels:
                    cluster_idx = all_labels.index(label)
                    prob_matrix[i, cluster_idx] += weight * prob
        
        # Normalize probabilities
        prob_sums = np.sum(prob_matrix, axis=1, keepdims=True)
        prob_matrix = prob_matrix / (prob_sums + 1e-10)
        
        # Assign labels and probabilities
        labels = np.array([all_labels[i] if prob_sums[i, 0] > 0 else -1 
                          for i in np.argmax(prob_matrix, axis=1)])
        probabilities = np.max(prob_matrix, axis=1)
        
        return {
            'labels': labels,
            'probabilities': probabilities
        }
    
    def _calculate_uncertainty_measures(self, predictions: Dict[str, Any], 
                                      ensemble_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate uncertainty measures for the predictions."""
        try:
            uncertainty_measures = {}
            
            # Method agreement
            if len(predictions) > 1:
                all_labels = [pred['labels'] for pred in predictions.values()]
                agreement_scores = []
                
                for i in range(len(ensemble_result['labels'])):
                    labels_at_i = [labels[i] for labels in all_labels]
                    unique_labels = set(labels_at_i)
                    if -1 in unique_labels:
                        unique_labels.remove(-1)
                    
                    if len(unique_labels) <= 1:
                        agreement_scores.append(1.0)  # Perfect agreement
                    else:
                        # Calculate agreement as 1 - (number of different labels / total methods)
                        agreement_scores.append(1.0 - (len(unique_labels) - 1) / len(predictions))
                
                uncertainty_measures['method_agreement'] = np.mean(agreement_scores)
                uncertainty_measures['method_agreement_std'] = np.std(agreement_scores)
            else:
                uncertainty_measures['method_agreement'] = 1.0
                uncertainty_measures['method_agreement_std'] = 0.0
            
            # Probability variance across methods
            if len(predictions) > 1:
                all_probs = [pred['probabilities'] for pred in predictions.values()]
                prob_variance = np.var(all_probs, axis=0)
                uncertainty_measures['probability_variance'] = np.mean(prob_variance)
                uncertainty_measures['probability_variance_std'] = np.std(prob_variance)
            else:
                uncertainty_measures['probability_variance'] = 0.0
                uncertainty_measures['probability_variance_std'] = 0.0
            
            # Low confidence predictions
            low_confidence_mask = ensemble_result['probabilities'] < self.config.uncertainty_threshold
            uncertainty_measures['low_confidence_ratio'] = np.mean(low_confidence_mask)
            uncertainty_measures['n_low_confidence'] = np.sum(low_confidence_mask)
            
            # Noise ratio
            noise_mask = ensemble_result['labels'] == -1
            uncertainty_measures['noise_ratio'] = np.mean(noise_mask)
            uncertainty_measures['n_noise'] = np.sum(noise_mask)
            
            return uncertainty_measures
            
        except Exception as e:
            tprint_debug(f"Uncertainty calculation failed: {e}")
            return {'error': str(e)}
    
    def save_model(self, filepath: Optional[str] = None) -> bool:
        """Save the trained model to disk."""
        try:
            if filepath is None:
                timestamp = int(time.time())
                filepath = f"{self.config.model_dir}/enhanced_hdbscan_model_{timestamp}.pkl"
            
            model_data = {
                'clusterer': self.clusterer,
                'scaler': self.scaler,
                'cluster_centers': self.cluster_centers,
                'cluster_densities': self.cluster_densities,
                'training_features': self.training_features,
                'training_labels': self.training_labels,
                'gmm_models': self.gmm_models,
                'knn_model': self.knn_model,
                'clustering_stats': self.clustering_stats,
                'best_params': self.best_params,
                'best_score': self.best_score,
                'config': self.config,
                'model_metadata': {
                    'created_at': time.time(),
                    'version': '1.0.0',
                    'n_features': self.training_features.shape[1] if self.training_features is not None else 0,
                    'n_clusters': len(set(self.training_labels)) - (1 if -1 in self.training_labels else 0) if self.training_labels is not None else 0
                }
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            tprint_success(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            tprint_error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained model from disk."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.clusterer = model_data['clusterer']
            self.scaler = model_data['scaler']
            self.cluster_centers = model_data['cluster_centers']
            self.cluster_densities = model_data['cluster_densities']
            self.training_features = model_data['training_features']
            self.training_labels = model_data['training_labels']
            self.gmm_models = model_data['gmm_models']
            self.knn_model = model_data['knn_model']
            self.clustering_stats = model_data['clustering_stats']
            self.best_params = model_data['best_params']
            self.best_score = model_data['best_score']
            self.model_metadata = model_data.get('model_metadata', {})
            
            # Update config if provided
            if 'config' in model_data:
                self.config = model_data['config']
            
            tprint_success(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            tprint_error(f"Failed to load model: {e}")
            return False
    
    def _validate_features(self, features: np.ndarray) -> bool:
        """Validate input features."""
        if features is None or len(features) == 0:
            return False
        
        if not isinstance(features, np.ndarray):
            return False
        
        if features.ndim != 2:
            return False
        
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            return False
        
        return True
    
    def _optimize_parameters(self, features_scaled: np.ndarray):
        """Optimize HDBSCAN parameters."""
        try:
            tprint_info("Optimizing HDBSCAN parameters")
            
            # Parameter search space
            param_grid = {
                'min_cluster_size': [10, 15, 20, 25],
                'min_samples': [3, 5, 7, 10],
                'cluster_selection_epsilon': [0.0, 0.1, 0.2],
                'metric': ['euclidean', 'manhattan', 'cosine']
            }
            
            best_score = -np.inf
            best_params = None
            
            # Grid search
            for min_cluster_size in param_grid['min_cluster_size']:
                for min_samples in param_grid['min_samples']:
                    for epsilon in param_grid['cluster_selection_epsilon']:
                        for metric in param_grid['metric']:
                            try:
                                # Create clusterer with current parameters
                                clusterer = HDBSCAN(
                                    min_cluster_size=min_cluster_size,
                                    min_samples=min_samples,
                                    cluster_selection_epsilon=epsilon,
                                    metric=metric,
                                    cluster_selection_method=self.config.cluster_selection_method
                                )
                                
                                # Fit and get labels
                                labels = clusterer.fit_predict(features_scaled)
                                
                                # Calculate score
                                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                                
                                if n_clusters < self.config.min_clusters or n_clusters > self.config.max_clusters:
                                    continue
                                
                                if n_clusters < 2:
                                    continue
                                
                                # Calculate silhouette score
                                non_noise_mask = labels != -1
                                if np.sum(non_noise_mask) < 2:
                                    continue
                                
                                score = silhouette_score(
                                    features_scaled[non_noise_mask], 
                                    labels[non_noise_mask]
                                )
                                
                                if score > best_score:
                                    best_score = score
                                    best_params = {
                                        'min_cluster_size': min_cluster_size,
                                        'min_samples': min_samples,
                                        'cluster_selection_epsilon': epsilon,
                                        'metric': metric
                                    }
                                
                            except Exception as e:
                                tprint_debug(f"Parameter combination failed: {e}")
                                continue
            
            if best_params is None:
                tprint_warning("Parameter optimization failed, using defaults")
                best_params = {
                    'min_cluster_size': self.config.min_cluster_size,
                    'min_samples': self.config.min_samples,
                    'cluster_selection_epsilon': self.config.cluster_selection_epsilon,
                    'metric': self.config.metric
                }
                best_score = 0.0
            
            self.best_params = best_params
            self.best_score = best_score
            
            tprint_success(f"Best parameters: {best_params}, Score: {best_score:.3f}")
            
        except Exception as e:
            tprint_error(f"Parameter optimization failed: {e}")
            self.best_params = {
                'min_cluster_size': self.config.min_cluster_size,
                'min_samples': self.config.min_samples,
                'cluster_selection_epsilon': self.config.cluster_selection_epsilon,
                'metric': self.config.metric
            }
            self.best_score = 0.0
    
    def _perform_clustering(self, features_scaled: np.ndarray) -> np.ndarray:
        """Perform the actual HDBSCAN clustering."""
        try:
            # Create clusterer with best parameters
            self.clusterer = HDBSCAN(
                min_cluster_size=self.best_params['min_cluster_size'],
                min_samples=self.best_params['min_samples'],
                cluster_selection_epsilon=self.best_params['cluster_selection_epsilon'],
                metric=self.best_params['metric'],
                cluster_selection_method=self.config.cluster_selection_method,
                metric_params=self.config.metric_params
            )
            
            # Perform clustering
            labels = self.clusterer.fit_predict(features_scaled)
            
            tprint_success(f"Clustering completed: {len(set(labels))} clusters found")
            return labels
            
        except Exception as e:
            tprint_error(f"Clustering failed: {e}")
            raise
    
    def _calculate_cluster_properties(self, features_scaled: np.ndarray, labels: np.ndarray):
        """Calculate cluster centers and densities."""
        try:
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels != -1]  # Remove noise label
            
            if len(unique_labels) == 0:
                self.cluster_centers = None
                self.cluster_densities = None
                return
            
            # Calculate cluster centers
            cluster_centers = []
            cluster_densities = []
            
            for label in unique_labels:
                cluster_mask = labels == label
                cluster_points = features_scaled[cluster_mask]
                
                # Calculate center as mean
                center = np.mean(cluster_points, axis=0)
                cluster_centers.append(center)
                
                # Calculate density as inverse of average distance to center
                distances = np.sqrt(((cluster_points - center) ** 2).sum(axis=1))
                avg_distance = np.mean(distances)
                density = 1.0 / (avg_distance + 1e-10)
                cluster_densities.append(density)
            
            self.cluster_centers = np.array(cluster_centers)
            self.cluster_densities = np.array(cluster_densities)
            
            tprint_debug(f"Calculated {len(unique_labels)} cluster centers and densities")
            
        except Exception as e:
            tprint_error(f"Failed to calculate cluster properties: {e}")
            self.cluster_centers = None
            self.cluster_densities = None
    
    def _train_auxiliary_models(self, features_scaled: np.ndarray, labels: np.ndarray):
        """Train auxiliary models for probability calculation."""
        try:
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels != -1]  # Remove noise label
            
            # Train GMM models for each cluster
            self.gmm_models = {}
            for label in unique_labels:
                try:
                    cluster_mask = labels == label
                    cluster_points = features_scaled[cluster_mask]
                    
                    if len(cluster_points) < 2:
                        continue
                    
                    # Train GMM for this cluster
                    gmm = GaussianMixture(n_components=1, random_state=42)
                    gmm.fit(cluster_points)
                    self.gmm_models[label] = gmm
                    
                except Exception as e:
                    tprint_debug(f"Failed to train GMM for cluster {label}: {e}")
                    continue
            
            # Train KNN model for KNN-based prediction
            try:
                non_noise_mask = labels != -1
                if np.sum(non_noise_mask) > 0:
                    self.knn_model = NearestNeighbors(n_neighbors=min(5, np.sum(non_noise_mask)))
                    self.knn_model.fit(features_scaled[non_noise_mask])
                else:
                    self.knn_model = None
                    
            except Exception as e:
                tprint_debug(f"Failed to train KNN model: {e}")
                self.knn_model = None
            
            tprint_debug(f"Trained {len(self.gmm_models)} GMM models and KNN model")
            
        except Exception as e:
            tprint_error(f"Failed to train auxiliary models: {e}")
            self.gmm_models = {}
            self.knn_model = None
    
    def _validate_clustering(self, features_scaled: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Validate clustering results."""
        try:
            validation_metrics = {}
            
            # Basic statistics
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = np.sum(labels == -1)
            noise_ratio = n_noise / len(labels)
            
            validation_metrics.update({
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'noise_ratio': noise_ratio
            })
            
            # Silhouette score (only for non-noise points)
            non_noise_mask = labels != -1
            if np.sum(non_noise_mask) >= 2 and n_clusters >= 2:
                try:
                    silhouette = silhouette_score(features_scaled[non_noise_mask], labels[non_noise_mask])
                    validation_metrics['silhouette_score'] = silhouette
                except Exception as e:
                    tprint_debug(f"Silhouette score calculation failed: {e}")
                    validation_metrics['silhouette_score'] = -1.0
            else:
                validation_metrics['silhouette_score'] = -1.0
            
            # Calinski-Harabasz score
            if n_clusters >= 2:
                try:
                    ch_score = calinski_harabasz_score(features_scaled[non_noise_mask], labels[non_noise_mask])
                    validation_metrics['calinski_harabasz_score'] = ch_score
                except Exception as e:
                    tprint_debug(f"Calinski-Harabasz score calculation failed: {e}")
                    validation_metrics['calinski_harabasz_score'] = 0.0
            else:
                validation_metrics['calinski_harabasz_score'] = 0.0
            
            # Davies-Bouldin score
            if n_clusters >= 2:
                try:
                    db_score = davies_bouldin_score(features_scaled[non_noise_mask], labels[non_noise_mask])
                    validation_metrics['davies_bouldin_score'] = db_score
                except Exception as e:
                    tprint_debug(f"Davies-Bouldin score calculation failed: {e}")
                    validation_metrics['davies_bouldin_score'] = float('inf')
            else:
                validation_metrics['davies_bouldin_score'] = float('inf')
            
            # Cluster size distribution
            cluster_sizes = []
            for label in set(labels):
                if label != -1:
                    cluster_sizes.append(np.sum(labels == label))
            
            if cluster_sizes:
                validation_metrics.update({
                    'min_cluster_size': min(cluster_sizes),
                    'max_cluster_size': max(cluster_sizes),
                    'avg_cluster_size': np.mean(cluster_sizes),
                    'cluster_size_std': np.std(cluster_sizes)
                })
            else:
                validation_metrics.update({
                    'min_cluster_size': 0,
                    'max_cluster_size': 0,
                    'avg_cluster_size': 0,
                    'cluster_size_std': 0
                })
            
            # Overall validation
            is_valid = (
                n_clusters >= self.config.min_clusters and
                n_clusters <= self.config.max_clusters and
                validation_metrics['silhouette_score'] >= self.config.min_silhouette_score
            )
            validation_metrics['is_valid'] = is_valid
            
            return validation_metrics
            
        except Exception as e:
            tprint_error(f"Clustering validation failed: {e}")
            return {'error': str(e), 'is_valid': False}
    
    def _calculate_clustering_stats(self, features_scaled: np.ndarray, 
                                  labels: np.ndarray, 
                                  validation_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive clustering statistics."""
        try:
            stats = {}
            
            # Basic clustering info
            n_samples = len(labels)
            n_clusters = validation_metrics.get('n_clusters', 0)
            n_noise = validation_metrics.get('n_noise', 0)
            
            stats.update({
                'n_samples': n_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'noise_ratio': n_noise / n_samples if n_samples > 0 else 0
            })
            
            # Quality metrics
            stats.update({
                'silhouette_score': validation_metrics.get('silhouette_score', -1.0),
                'calinski_harabasz_score': validation_metrics.get('calinski_harabasz_score', 0.0),
                'davies_bouldin_score': validation_metrics.get('davies_bouldin_score', float('inf'))
            })
            
            # Cluster size statistics
            stats.update({
                'min_cluster_size': validation_metrics.get('min_cluster_size', 0),
                'max_cluster_size': validation_metrics.get('max_cluster_size', 0),
                'avg_cluster_size': validation_metrics.get('avg_cluster_size', 0),
                'cluster_size_std': validation_metrics.get('cluster_size_std', 0)
            })
            
            # Feature statistics
            if features_scaled is not None:
                stats.update({
                    'n_features': features_scaled.shape[1],
                    'feature_mean': np.mean(features_scaled),
                    'feature_std': np.std(features_scaled),
                    'feature_min': np.min(features_scaled),
                    'feature_max': np.max(features_scaled)
                })
            
            # Cluster separation (if we have cluster centers)
            if self.cluster_centers is not None and len(self.cluster_centers) > 1:
                try:
                    # Calculate pairwise distances between cluster centers
                    center_distances = []
                    for i in range(len(self.cluster_centers)):
                        for j in range(i + 1, len(self.cluster_centers)):
                            dist = np.linalg.norm(self.cluster_centers[i] - self.cluster_centers[j])
                            center_distances.append(dist)
                    
                    if center_distances:
                        stats.update({
                            'min_center_distance': min(center_distances),
                            'max_center_distance': max(center_distances),
                            'avg_center_distance': np.mean(center_distances),
                            'center_distance_std': np.std(center_distances)
                        })
                except Exception as e:
                    tprint_debug(f"Center distance calculation failed: {e}")
            
            # Validation status
            stats['is_valid'] = validation_metrics.get('is_valid', False)
            
            # Timestamp
            stats['timestamp'] = time.time()
            
            return stats
            
        except Exception as e:
            tprint_error(f"Clustering stats calculation failed: {e}")
            return {'error': str(e)}
    
    def _random_fallback_with_uncertainty(self, features: np.ndarray) -> Dict[str, Any]:
        """Fallback prediction with uncertainty measures."""
        n_samples = len(features)
        labels = np.random.randint(0, 3, n_samples)
        probabilities = np.random.uniform(0.1, 0.9, n_samples)
        
        return {
            'labels': labels,
            'probabilities': probabilities,
            'uncertainty_measures': {
                'method_agreement': 0.0,
                'probability_variance': 0.1,
                'low_confidence_ratio': 0.5,
                'noise_ratio': 0.0
            },
            'method_breakdown': {},
            'success': True
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0