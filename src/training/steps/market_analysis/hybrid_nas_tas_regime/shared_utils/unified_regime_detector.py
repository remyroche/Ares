"""
Unified Regime Detector for NAS and TAS Systems

This module consolidates all regime detection capabilities for both
Neural Architecture Search (NAS) and Tree Architecture Search (TAS) systems.
It provides a unified interface for regime detection, clustering, and analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

class RegimeDetectionMethod(Enum):
    """Available regime detection methods."""
    CLUSTERING = "clustering"
    HIDDEN_MARKOV_MODEL = "hmm"
    CHANGE_POINT = "change_point"
    NEURAL_NETWORK = "neural_network"
    TREE_BASED = "tree_based"
    HYBRID = "hybrid"

class ArchitectureType(Enum):
    """Types of architectures supported."""
    NEURAL = "neural"
    TREE = "tree"
    HYBRID = "hybrid"

@dataclass
class RegimeDetectionConfig:
    """Configuration for regime detection."""
    
    # Core detection parameters
    method: RegimeDetectionMethod = RegimeDetectionMethod.HYBRID
    architecture_type: ArchitectureType = ArchitectureType.HYBRID
    n_regimes: int = 3
    min_regime_duration: int = 10
    
    # Clustering parameters
    clustering_algorithm: str = "kmeans"
    n_clusters: int = 3
    random_state: int = 42
    
    # HMM parameters
    hmm_n_states: int = 3
    hmm_n_iterations: int = 100
    
    # Change point detection parameters
    change_point_method: str = "pelt"
    change_point_penalty: float = 1.0
    
    # Neural network parameters
    neural_n_epochs: int = 100
    neural_hidden_size: int = 64
    
    # Tree parameters
    tree_max_depth: int = 10
    tree_min_samples_split: int = 2
    
    # Advanced parameters
    enable_regime_validation: bool = True
    stability_threshold: float = 0.7
    separation_threshold: float = 0.5

@dataclass
class RegimeInfo:
    """Information about a detected regime."""
    regime_id: int
    start_index: int
    end_index: int
    duration: int
    stability: float
    separation: float
    characteristics: Dict[str, Any]

@dataclass
class RegimeDetectionResult:
    """Result of regime detection."""
    
    # Core results
    regime_labels: np.ndarray
    regime_infos: List[RegimeInfo]
    n_regimes: int
    
    # Detection metadata
    method: RegimeDetectionMethod
    architecture_type: ArchitectureType
    detection_time: float
    
    # Quality metrics
    regime_quality_score: float
    stability_scores: List[float]
    separation_scores: List[float]
    
    # Metadata
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    success: bool = True
    error_message: Optional[str] = None

class UnifiedRegimeDetector:
    """
    Unified regime detector that consolidates all regime detection capabilities
    for both NAS and TAS systems.
    """
    
    def __init__(self, config: Optional[RegimeDetectionConfig] = None):
        """Initialize unified regime detector."""
        self.config = config or RegimeDetectionConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Performance monitoring
        self.detection_history = []
        
        print(f"🚀 Unified Regime Detector initialized")
        print(f"   Method: {self.config.method.value}")
        print(f"   Architecture type: {self.config.architecture_type.value}")
        print(f"   Target regimes: {self.config.n_regimes}")
    
    def detect_regimes(self, 
                      data: pd.DataFrame,
                      features: Optional[List[str]] = None,
                      method: Optional[RegimeDetectionMethod] = None) -> RegimeDetectionResult:
        """
        Detect regimes in the data using the specified method.
        
        Args:
            data: Input data for regime detection
            features: List of feature columns to use (if None, uses all numeric columns)
            method: Detection method to use (defaults to config method)
            
        Returns:
            RegimeDetectionResult containing regime information
        """
        try:
            print("🔍 Starting regime detection...")
            start_time = time.time()
            
            # Use specified method or default from config
            detection_method = method or self.config.method
            
            # Validate inputs
            self._validate_inputs(data, features)
            
            # Prepare features
            feature_data = self._prepare_features(data, features)
            
            # Detect regimes using specified method
            regime_labels = self._detect_regimes_method(feature_data, detection_method)
            
            # Analyze detected regimes
            regime_infos = self._analyze_regimes(regime_labels, data)
            
            # Calculate quality metrics
            regime_quality_score = self._calculate_regime_quality(regime_infos)
            stability_scores = [ri.stability for ri in regime_infos]
            separation_scores = [ri.separation for ri in regime_infos]
            
            # Create result
            result = RegimeDetectionResult(
                regime_labels=regime_labels,
                regime_infos=regime_infos,
                n_regimes=len(regime_infos),
                method=detection_method,
                architecture_type=self.config.architecture_type,
                detection_time=time.time() - start_time,
                regime_quality_score=regime_quality_score,
                stability_scores=stability_scores,
                separation_scores=separation_scores
            )
            
            # Save detection history
            self.detection_history.append(result)
            
            print(f"✅ Regime detection completed in {result.detection_time:.2f}s")
            print(f"   Detected regimes: {result.n_regimes}")
            print(f"   Quality score: {regime_quality_score:.4f}")
            
            return result
            
        except Exception as e:
            print(f"❌ Regime detection failed: {e}")
            return RegimeDetectionResult(
                regime_labels=np.array([]),
                regime_infos=[],
                n_regimes=0,
                method=detection_method or self.config.method,
                architecture_type=self.config.architecture_type,
                detection_time=0.0,
                regime_quality_score=0.0,
                stability_scores=[],
                separation_scores=[],
                success=False,
                error_message=str(e)
            )
    
    def _validate_inputs(self, data: pd.DataFrame, features: Optional[List[str]]):
        """Validate input data."""
        if len(data) == 0:
            raise ValueError("Data cannot be empty")
        
        if features is not None:
            for feature in features:
                if feature not in data.columns:
                    raise ValueError(f"Feature '{feature}' not found in data")
    
    def _prepare_features(self, data: pd.DataFrame, features: Optional[List[str]]) -> np.ndarray:
        """Prepare feature data for regime detection."""
        if features is None:
            # Use all numeric columns
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_columns = features
        
        if len(numeric_columns) == 0:
            raise ValueError("No numeric features found for regime detection")
        
        # Extract and normalize features
        feature_data = data[numeric_columns].values
        
        # Handle missing values
        feature_data = np.nan_to_num(feature_data, nan=0.0)
        
        # Normalize features
        feature_means = np.mean(feature_data, axis=0)
        feature_stds = np.std(feature_data, axis=0)
        feature_stds[feature_stds == 0] = 1.0  # Avoid division by zero
        
        feature_data = (feature_data - feature_means) / feature_stds
        
        return feature_data
    
    def _detect_regimes_method(self, feature_data: np.ndarray, method: RegimeDetectionMethod) -> np.ndarray:
        """Detect regimes using the specified method."""
        if method == RegimeDetectionMethod.CLUSTERING:
            return self._detect_regimes_clustering(feature_data)
        elif method == RegimeDetectionMethod.TREE_BASED:
            return self._detect_regimes_tree(feature_data)
        elif method == RegimeDetectionMethod.NEURAL_NETWORK:
            return self._detect_regimes_neural(feature_data)
        elif method == RegimeDetectionMethod.HYBRID:
            return self._detect_regimes_hybrid(feature_data)
        else:
            raise ValueError(f"Unsupported regime detection method: {method}")
    
    def _detect_regimes_clustering(self, feature_data: np.ndarray) -> np.ndarray:
        """Detect regimes using clustering."""
        try:
            from sklearn.cluster import KMeans
            
            # Perform K-means clustering
            kmeans = KMeans(
                n_clusters=self.config.n_clusters,
                random_state=self.config.random_state,
                n_init=10
            )
            regime_labels = kmeans.fit_predict(feature_data)
            
            return regime_labels
            
        except ImportError:
            print("⚠️ scikit-learn not available, using simple clustering")
            return self._simple_clustering(feature_data)
    
    def _detect_regimes_tree(self, feature_data: np.ndarray) -> np.ndarray:
        """Detect regimes using tree-based methods."""
        try:
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.cluster import KMeans
            
            # First perform clustering to get initial labels
            kmeans = KMeans(n_clusters=self.config.n_regimes, random_state=self.config.random_state)
            initial_labels = kmeans.fit_predict(feature_data)
            
            # Use decision tree to refine regime boundaries
            tree = DecisionTreeClassifier(
                max_depth=self.config.tree_max_depth,
                min_samples_split=self.config.tree_min_samples_split,
                random_state=self.config.random_state
            )
            
            # Create features for tree (use rolling statistics)
            tree_features = self._create_tree_features(feature_data)
            
            # Fit tree on initial labels
            tree.fit(tree_features, initial_labels)
            
            # Predict refined labels
            regime_labels = tree.predict(tree_features)
            
            return regime_labels
            
        except ImportError:
            print("⚠️ scikit-learn not available, using simple clustering")
            return self._simple_clustering(feature_data)
    
    def _detect_regimes_neural(self, feature_data: np.ndarray) -> np.ndarray:
        """Detect regimes using neural network methods."""
        try:
            from sklearn.neural_network import MLPClassifier
            from sklearn.cluster import KMeans
            
            # First perform clustering to get initial labels
            kmeans = KMeans(n_clusters=self.config.n_regimes, random_state=self.config.random_state)
            initial_labels = kmeans.fit_predict(feature_data)
            
            # Use neural network to refine regime boundaries
            neural_net = MLPClassifier(
                hidden_layer_sizes=(self.config.neural_hidden_size,),
                max_iter=self.config.neural_n_epochs,
                random_state=self.config.random_state
            )
            
            # Create features for neural network
            neural_features = self._create_neural_features(feature_data)
            
            # Fit neural network on initial labels
            neural_net.fit(neural_features, initial_labels)
            
            # Predict refined labels
            regime_labels = neural_net.predict(neural_features)
            
            return regime_labels
            
        except ImportError:
            print("⚠️ scikit-learn not available, using simple clustering")
            return self._simple_clustering(feature_data)
    
    def _detect_regimes_hybrid(self, feature_data: np.ndarray) -> np.ndarray:
        """Detect regimes using hybrid method."""
        # Combine multiple methods
        clustering_labels = self._detect_regimes_clustering(feature_data)
        tree_labels = self._detect_regimes_tree(feature_data)
        
        # Combine labels using voting
        combined_labels = np.zeros_like(clustering_labels)
        for i in range(len(combined_labels)):
            # Simple voting mechanism
            if clustering_labels[i] == tree_labels[i]:
                combined_labels[i] = clustering_labels[i]
            else:
                # Use clustering result as default
                combined_labels[i] = clustering_labels[i]
        
        return combined_labels
    
    def _simple_clustering(self, feature_data: np.ndarray) -> np.ndarray:
        """Simple clustering fallback when scikit-learn is not available."""
        # Simple distance-based clustering
        n_samples, n_features = feature_data.shape
        n_clusters = self.config.n_regimes
        
        # Initialize cluster centers randomly
        np.random.seed(self.config.random_state)
        cluster_centers = feature_data[np.random.choice(n_samples, n_clusters, replace=False)]
        
        regime_labels = np.zeros(n_samples, dtype=int)
        
        # Simple K-means iteration
        for iteration in range(10):  # Limited iterations
            # Assign points to closest cluster
            for i in range(n_samples):
                distances = [np.linalg.norm(feature_data[i] - center) for center in cluster_centers]
                regime_labels[i] = np.argmin(distances)
            
            # Update cluster centers
            new_centers = []
            for k in range(n_clusters):
                cluster_points = feature_data[regime_labels == k]
                if len(cluster_points) > 0:
                    new_centers.append(np.mean(cluster_points, axis=0))
                else:
                    new_centers.append(cluster_centers[k])
            
            cluster_centers = np.array(new_centers)
        
        return regime_labels
    
    def _create_tree_features(self, feature_data: np.ndarray) -> np.ndarray:
        """Create features for tree-based regime detection."""
        # Add rolling statistics as features
        window_size = min(10, len(feature_data) // 4)
        
        enhanced_features = [feature_data]  # Original features
        
        if window_size > 1:
            # Rolling mean
            rolling_mean = np.array([
                np.mean(feature_data[max(0, i-window_size):i+1], axis=0) 
                for i in range(len(feature_data))
            ])
            enhanced_features.append(rolling_mean)
            
            # Rolling std
            rolling_std = np.array([
                np.std(feature_data[max(0, i-window_size):i+1], axis=0) 
                for i in range(len(feature_data))
            ])
            enhanced_features.append(rolling_std)
        
        return np.concatenate(enhanced_features, axis=1)
    
    def _create_neural_features(self, feature_data: np.ndarray) -> np.ndarray:
        """Create features for neural network regime detection."""
        # Similar to tree features but with more transformations
        enhanced_features = [feature_data]  # Original features
        
        # Add lagged features
        if len(feature_data) > 1:
            lagged_features = np.vstack([feature_data[0], feature_data[:-1]])
            enhanced_features.append(lagged_features)
        
        # Add difference features
        if len(feature_data) > 1:
            diff_features = np.vstack([np.zeros_like(feature_data[0]), np.diff(feature_data, axis=0)])
            enhanced_features.append(diff_features)
        
        return np.concatenate(enhanced_features, axis=1)
    
    def _analyze_regimes(self, regime_labels: np.ndarray, data: pd.DataFrame) -> List[RegimeInfo]:
        """Analyze detected regimes."""
        regime_infos = []
        unique_regimes = np.unique(regime_labels)
        
        for regime_id in unique_regimes:
            regime_mask = regime_labels == regime_id
            regime_indices = np.where(regime_mask)[0]
            
            if len(regime_indices) < self.config.min_regime_duration:
                continue
            
            # Find regime boundaries
            regime_changes = np.diff(np.concatenate([[False], regime_mask, [False]]).astype(int))
            start_indices = np.where(regime_changes == 1)[0]
            end_indices = np.where(regime_changes == -1)[0]
            
            # Analyze each regime segment
            for start_idx, end_idx in zip(start_indices, end_indices):
                duration = end_idx - start_idx
                
                if duration >= self.config.min_regime_duration:
                    # Calculate regime characteristics
                    regime_data = data.iloc[start_idx:end_idx]
                    characteristics = self._calculate_regime_characteristics(regime_data)
                    
                    # Calculate stability and separation
                    stability = self._calculate_regime_stability(regime_labels, regime_id, start_idx, end_idx)
                    separation = self._calculate_regime_separation(regime_labels, regime_id, start_idx, end_idx)
                    
                    regime_info = RegimeInfo(
                        regime_id=int(regime_id),
                        start_index=start_idx,
                        end_index=end_idx,
                        duration=duration,
                        stability=stability,
                        separation=separation,
                        characteristics=characteristics
                    )
                    
                    regime_infos.append(regime_info)
        
        return regime_infos
    
    def _calculate_regime_characteristics(self, regime_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate characteristics of a regime."""
        characteristics = {}
        
        # Basic statistics
        numeric_columns = regime_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in regime_data.columns:
                characteristics[f'{col}_mean'] = float(regime_data[col].mean())
                characteristics[f'{col}_std'] = float(regime_data[col].std())
                characteristics[f'{col}_min'] = float(regime_data[col].min())
                characteristics[f'{col}_max'] = float(regime_data[col].max())
        
        # Duration
        characteristics['duration'] = len(regime_data)
        
        return characteristics
    
    def _calculate_regime_stability(self, regime_labels: np.ndarray, regime_id: int, 
                                   start_idx: int, end_idx: int) -> float:
        """Calculate stability of a regime."""
        regime_segment = regime_labels[start_idx:end_idx]
        
        # Stability is the ratio of correct regime labels
        correct_labels = np.sum(regime_segment == regime_id)
        stability = correct_labels / len(regime_segment) if len(regime_segment) > 0 else 0.0
        
        return stability
    
    def _calculate_regime_separation(self, regime_labels: np.ndarray, regime_id: int, 
                                    start_idx: int, end_idx: int) -> float:
        """Calculate separation of a regime from other regimes."""
        # Count transitions at regime boundaries
        total_transitions = 0
        regime_transitions = 0
        
        if start_idx > 0:
            total_transitions += 1
            if regime_labels[start_idx-1] != regime_id:
                regime_transitions += 1
        
        if end_idx < len(regime_labels):
            total_transitions += 1
            if regime_labels[end_idx] != regime_id:
                regime_transitions += 1
        
        separation = regime_transitions / total_transitions if total_transitions > 0 else 1.0
        
        return separation
    
    def _calculate_regime_quality(self, regime_infos: List[RegimeInfo]) -> float:
        """Calculate overall regime quality score."""
        if not regime_infos:
            return 0.0
        
        # Combine stability and separation scores
        stability_scores = [ri.stability for ri in regime_infos]
        separation_scores = [ri.separation for ri in regime_infos]
        
        avg_stability = np.mean(stability_scores)
        avg_separation = np.mean(separation_scores)
        
        # Quality is combination of stability and separation
        quality_score = 0.6 * avg_stability + 0.4 * avg_separation
        
        return quality_score
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """Get summary of regime detection performance."""
        if not self.detection_history:
            return {'total_detections': 0}
        
        recent_detections = self.detection_history[-10:]  # Last 10 detections
        
        summary = {
            'total_detections': len(self.detection_history),
            'avg_detection_time': np.mean([d.detection_time for d in recent_detections]),
            'avg_regime_quality': np.mean([d.regime_quality_score for d in recent_detections]),
            'avg_n_regimes': np.mean([d.n_regimes for d in recent_detections]),
            'success_rate': np.mean([d.success for d in recent_detections])
        }
        
        return summary

def create_unified_regime_detector(config: Optional[RegimeDetectionConfig] = None) -> UnifiedRegimeDetector:
    """Create a unified regime detector with specified configuration."""
    return UnifiedRegimeDetector(config)

__all__ = [
    'UnifiedRegimeDetector',
    'RegimeDetectionConfig',
    'RegimeDetectionResult',
    'RegimeInfo',
    'RegimeDetectionMethod',
    'ArchitectureType',
    'create_unified_regime_detector'
]