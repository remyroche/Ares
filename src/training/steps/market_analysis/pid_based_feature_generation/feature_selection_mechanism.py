"""
Feature Selection Mechanism using Partial Information Decomposition

This module provides the core feature selection mechanism that uses PID analysis
to select the most relevant features for interaction, polynomial, and cross-timeframe
feature generation.

Key Features:
- Uses PID analysis to identify significant feature interactions
- Selects features based on synergy, redundancy, and unique information
- Provides data-driven selection criteria
- Supports different selection strategies
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

# Core dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# Import PID utilities
try:
    from src.training.utils.feature_selection.partial_information_decompositor import (
        PartialInformationDecompositor, PIDConfig, PIDResult
    )
    PID_AVAILABLE = True
except ImportError as e:
    logging.warning(f"PID utilities not available: {e}")
    PID_AVAILABLE = False

# Import matrix operations
try:
    from src.utils.matrix_operations import get_unified_matrix_operations
    MATRIX_OPS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Matrix operations not available: {e}")
    MATRIX_OPS_AVAILABLE = False

# Import logger
try:
    from src.utils.logger import system_logger
    logger = system_logger.getChild('FeatureSelectionMechanism')
except ImportError:
    logger = logging.getLogger('FeatureSelectionMechanism')
    logger.setLevel(logging.INFO)


class SelectionStrategy(Enum):
    """Feature selection strategies."""
    SYNERGY_BASED = "synergy_based"          # Select based on synergy scores
    UNIQUE_INFO_BASED = "unique_info_based"  # Select based on unique information
    REDUNDANCY_BASED = "redundancy_based"    # Select based on redundancy scores
    COMBINED = "combined"                    # Combine multiple criteria
    CORRELATION_BASED = "correlation_based"  # Fallback correlation-based selection


@dataclass
class FeatureSelectionConfig:
    """Configuration for feature selection."""
    # PID Configuration
    synergy_threshold: float = 0.1
    redundancy_threshold: float = 0.15
    unique_info_threshold: float = 0.05
    
    # Selection Strategy
    selection_strategy: SelectionStrategy = SelectionStrategy.COMBINED
    
    # Feature Limits
    max_interaction_features: int = 100
    max_polynomial_features: int = 50
    max_cross_timeframe_features: int = 50
    
    # Selection Criteria
    min_synergy_score: float = 0.05
    min_unique_info_score: float = 0.02
    max_redundancy_score: float = 0.8
    
    # Computational Settings
    enable_parallel_processing: bool = True
    enable_gpu_acceleration: bool = True
    memory_limit_gb: float = 8.0


@dataclass
class FeatureSelectionResult:
    """Result of feature selection."""
    # Selected features for each type
    interaction_features: List[Tuple[str, str]] = field(default_factory=list)
    polynomial_features: List[str] = field(default_factory=list)
    cross_timeframe_features: List[Tuple[str, str]] = field(default_factory=list)
    
    # Selection scores
    interaction_scores: Dict[Tuple[str, str], float] = field(default_factory=dict)
    polynomial_scores: Dict[str, float] = field(default_factory=dict)
    cross_timeframe_scores: Dict[Tuple[str, str], float] = field(default_factory=dict)
    
    # PID analysis results
    pid_result: Optional[PIDResult] = None
    
    # Metadata
    selection_strategy: SelectionStrategy = SelectionStrategy.COMBINED
    total_features_analyzed: int = 0
    selection_time: float = 0.0
    
    # Quality metrics
    average_synergy_score: float = 0.0
    average_unique_info_score: float = 0.0
    average_redundancy_score: float = 0.0


class FeatureSelectionMechanism:
    """
    Feature Selection Mechanism using Partial Information Decomposition.
    
    Uses PID analysis to select the most relevant features for interaction,
    polynomial, and cross-timeframe feature generation.
    """
    
    def __init__(self, config: Optional[FeatureSelectionConfig] = None):
        """Initialize the feature selection mechanism."""
        self.config = config or FeatureSelectionConfig()
        self.logger = logger.getChild('FeatureSelectionMechanism')
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info("🔧 FeatureSelectionMechanism initialized")
        self.logger.info(f"📊 Selection strategy: {self.config.selection_strategy.value}")
        self.logger.info(f"📊 Max interaction features: {self.config.max_interaction_features}")
        self.logger.info(f"📊 Max polynomial features: {self.config.max_polynomial_features}")
        self.logger.info(f"📊 Max cross-timeframe features: {self.config.max_cross_timeframe_features}")
    
    def _initialize_components(self):
        """Initialize required components."""
        # Initialize PID decompositor
        if PID_AVAILABLE:
            pid_config = PIDConfig(
                synergy_threshold=self.config.synergy_threshold,
                redundancy_threshold=self.config.redundancy_threshold,
                unique_info_threshold=self.config.unique_info_threshold,
                max_interaction_features=self.config.max_interaction_features,
                max_polynomial_degree=3,
                max_timeframe_lag=5
            )
            self.pid_decompositor = PartialInformationDecompositor(pid_config)
            self.logger.info("✅ PID Decompositor initialized")
        else:
            self.pid_decompositor = None
            self.logger.warning("⚠️ PID Decompositor not available")
        
        # Initialize matrix operations
        if MATRIX_OPS_AVAILABLE:
            self.matrix_ops = get_unified_matrix_operations(
                enable_gpu=self.config.enable_gpu_acceleration,
                enable_memory_optimization=True,
                enable_parallel=self.config.enable_parallel_processing
            )
            self.logger.info("✅ Matrix Operations initialized")
        else:
            self.matrix_ops = None
            self.logger.warning("⚠️ Matrix Operations not available")
    
    def select_features(
        self, 
        X: np.ndarray, 
        feature_names: List[str], 
        target: Optional[np.ndarray] = None
    ) -> FeatureSelectionResult:
        """
        Select features using PID analysis.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: List of feature names
            target: Target variable for PID analysis (optional)
            
        Returns:
            FeatureSelectionResult with selected features
        """
        start_time = time.time()
        self.logger.info("🔍 Starting feature selection using PID analysis...")
        
        result = FeatureSelectionResult()
        result.selection_strategy = self.config.selection_strategy
        result.total_features_analyzed = len(feature_names)
        
        try:
            # Perform PID analysis if target is available
            if self.pid_decompositor and target is not None:
                self.logger.info("🔍 Performing PID analysis...")
                pid_result = self.pid_decompositor.decompose_information(X, target, feature_names)
                result.pid_result = pid_result
                
                # Select features based on PID results
                result = self._select_features_from_pid(result, pid_result, feature_names)
                self.logger.info("✅ Features selected using PID analysis")
            else:
                # Fallback to correlation-based selection
                self.logger.info("📊 Using correlation-based feature selection")
                result = self._select_features_from_correlation(result, X, feature_names)
                self.logger.info("✅ Features selected using correlation analysis")
            
            # Calculate quality metrics
            result.average_synergy_score = self._calculate_average_synergy_score(result)
            result.average_unique_info_score = self._calculate_average_unique_info_score(result)
            result.average_redundancy_score = self._calculate_average_redundancy_score(result)
            
            selection_time = time.time() - start_time
            result.selection_time = selection_time
            
            self.logger.info(f"✅ Feature selection completed in {selection_time:.3f}s")
            self.logger.info(f"📊 Selected {len(result.interaction_features)} interaction features")
            self.logger.info(f"📊 Selected {len(result.polynomial_features)} polynomial features")
            self.logger.info(f"📊 Selected {len(result.cross_timeframe_features)} cross-timeframe features")
            
            return result
            
        except Exception as e:
            selection_time = time.time() - start_time
            result.selection_time = selection_time
            
            self.logger.error(f"❌ Feature selection failed: {e}")
            return result
    
    def _select_features_from_pid(
        self, 
        result: FeatureSelectionResult, 
        pid_result: PIDResult, 
        feature_names: List[str]
    ) -> FeatureSelectionResult:
        """Select features based on PID analysis results."""
        try:
            # Select interaction features based on synergy
            interaction_features = self._select_interaction_features_from_pid(pid_result)
            result.interaction_features = interaction_features
            result.interaction_scores = {
                pair: pid_result.synergy.get(pair, 0.0) for pair in interaction_features
            }
            
            # Select polynomial features based on unique information
            polynomial_features = self._select_polynomial_features_from_pid(pid_result, feature_names)
            result.polynomial_features = polynomial_features
            result.polynomial_scores = {
                feature: pid_result.unique_info.get(feature, 0.0) for feature in polynomial_features
            }
            
            # Select cross-timeframe features based on cross-timeframe analysis
            cross_timeframe_features = self._select_cross_timeframe_features_from_pid(pid_result, feature_names)
            result.cross_timeframe_features = cross_timeframe_features
            result.cross_timeframe_scores = {
                pair: pid_result.synergy.get(pair, 0.0) for pair in cross_timeframe_features
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ PID-based feature selection failed: {e}")
            return result
    
    def _select_interaction_features_from_pid(self, pid_result: PIDResult) -> List[Tuple[str, str]]:
        """Select interaction features based on synergy scores."""
        try:
            # Sort synergy scores and select top features
            synergy_items = sorted(pid_result.synergy.items(), key=lambda x: x[1], reverse=True)
            
            selected_features = []
            for (feat1, feat2), synergy_score in synergy_items:
                if synergy_score > self.config.min_synergy_score:
                    selected_features.append((feat1, feat2))
                    if len(selected_features) >= self.config.max_interaction_features:
                        break
            
            self.logger.info(f"📊 Selected {len(selected_features)} interaction features based on synergy")
            return selected_features
            
        except Exception as e:
            self.logger.warning(f"⚠️ Interaction feature selection failed: {e}")
            return []
    
    def _select_polynomial_features_from_pid(self, pid_result: PIDResult, feature_names: List[str]) -> List[str]:
        """Select polynomial features based on unique information scores."""
        try:
            # Sort unique information scores and select top features
            unique_info_items = sorted(pid_result.unique_info.items(), key=lambda x: x[1], reverse=True)
            
            selected_features = []
            for feature, unique_score in unique_info_items:
                if unique_score > self.config.min_unique_info_score:
                    selected_features.append(feature)
                    if len(selected_features) >= self.config.max_polynomial_features:
                        break
            
            self.logger.info(f"📊 Selected {len(selected_features)} polynomial features based on unique information")
            return selected_features
            
        except Exception as e:
            self.logger.warning(f"⚠️ Polynomial feature selection failed: {e}")
            return []
    
    def _select_cross_timeframe_features_from_pid(self, pid_result: PIDResult, feature_names: List[str]) -> List[Tuple[str, str]]:
        """Select cross-timeframe features based on cross-timeframe analysis."""
        try:
            # Identify timeframe features
            timeframe_features = self._identify_timeframe_features(feature_names)
            
            # Filter synergy scores for timeframe features only
            timeframe_synergy = {
                (feat1, feat2): score for (feat1, feat2), score in pid_result.synergy.items()
                if feat1 in timeframe_features and feat2 in timeframe_features
            }
            
            # Sort by synergy score and select top cross-timeframe features
            synergy_items = sorted(timeframe_synergy.items(), key=lambda x: x[1], reverse=True)
            
            selected_features = []
            for (feat1, feat2), synergy_score in synergy_items:
                if synergy_score > self.config.min_synergy_score:
                    # Check if features are from different timeframes
                    if self._are_different_timeframes(feat1, feat2):
                        selected_features.append((feat1, feat2))
                        if len(selected_features) >= self.config.max_cross_timeframe_features:
                            break
            
            self.logger.info(f"📊 Selected {len(selected_features)} cross-timeframe features based on synergy")
            return selected_features
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cross-timeframe feature selection failed: {e}")
            return []
    
    def _select_features_from_correlation(
        self, 
        result: FeatureSelectionResult, 
        X: np.ndarray, 
        feature_names: List[str]
    ) -> FeatureSelectionResult:
        """Fallback correlation-based feature selection."""
        try:
            if self.matrix_ops:
                # Use matrix operations for correlation calculation
                correlation_matrix = self.matrix_ops.safe_correlation_matrix(X)
            else:
                # Fallback to numpy correlation
                correlation_matrix = np.corrcoef(X.T)
            
            # Select interaction features based on correlation
            interaction_features = self._select_interaction_features_from_correlation(
                correlation_matrix, feature_names
            )
            result.interaction_features = interaction_features
            result.interaction_scores = {
                pair: abs(correlation_matrix[feature_names.index(pair[0]), feature_names.index(pair[1])])
                for pair in interaction_features
            }
            
            # Select polynomial features based on variance
            polynomial_features = self._select_polynomial_features_from_variance(X, feature_names)
            result.polynomial_features = polynomial_features
            result.polynomial_scores = {
                feature: np.var(X[:, feature_names.index(feature)])
                for feature in polynomial_features
            }
            
            # Select cross-timeframe features based on timeframe correlation
            cross_timeframe_features = self._select_cross_timeframe_features_from_correlation(
                correlation_matrix, feature_names
            )
            result.cross_timeframe_features = cross_timeframe_features
            result.cross_timeframe_scores = {
                pair: abs(correlation_matrix[feature_names.index(pair[0]), feature_names.index(pair[1])])
                for pair in cross_timeframe_features
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Correlation-based feature selection failed: {e}")
            return result
    
    def _select_interaction_features_from_correlation(
        self, 
        correlation_matrix: np.ndarray, 
        feature_names: List[str]
    ) -> List[Tuple[str, str]]:
        """Select interaction features based on correlation."""
        try:
            selected_features = []
            n_features = len(feature_names)
            
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    corr = abs(correlation_matrix[i, j])
                    
                    if (self.config.min_synergy_score <= corr <= self.config.max_redundancy_score):
                        selected_features.append((feature_names[i], feature_names[j]))
                        
                        if len(selected_features) >= self.config.max_interaction_features:
                            break
                
                if len(selected_features) >= self.config.max_interaction_features:
                    break
            
            return selected_features
            
        except Exception as e:
            self.logger.warning(f"⚠️ Correlation-based interaction selection failed: {e}")
            return []
    
    def _select_polynomial_features_from_variance(
        self, 
        X: np.ndarray, 
        feature_names: List[str]
    ) -> List[str]:
        """Select polynomial features based on variance."""
        try:
            # Calculate variance for each feature
            variances = np.var(X, axis=0)
            
            # Select features with highest variance
            variance_indices = np.argsort(variances)[::-1]
            selected_features = []
            
            for idx in variance_indices:
                if variances[idx] > 0.01:  # Minimum variance threshold
                    selected_features.append(feature_names[idx])
                    if len(selected_features) >= self.config.max_polynomial_features:
                        break
            
            return selected_features
            
        except Exception as e:
            self.logger.warning(f"⚠️ Variance-based polynomial selection failed: {e}")
            return []
    
    def _select_cross_timeframe_features_from_correlation(
        self, 
        correlation_matrix: np.ndarray, 
        feature_names: List[str]
    ) -> List[Tuple[str, str]]:
        """Select cross-timeframe features based on correlation."""
        try:
            # Identify timeframe features
            timeframe_features = self._identify_timeframe_features(feature_names)
            
            selected_features = []
            timeframe_indices = [feature_names.index(f) for f in timeframe_features]
            
            for i, feat1 in enumerate(timeframe_features):
                for j, feat2 in enumerate(timeframe_features[i+1:], i+1):
                    # Check if features are from different timeframes
                    if self._are_different_timeframes(feat1, feat2):
                        idx1 = feature_names.index(feat1)
                        idx2 = feature_names.index(feat2)
                        corr = abs(correlation_matrix[idx1, idx2])
                        
                        if (self.config.min_synergy_score <= corr <= self.config.max_redundancy_score):
                            selected_features.append((feat1, feat2))
                            
                            if len(selected_features) >= self.config.max_cross_timeframe_features:
                                break
                
                if len(selected_features) >= self.config.max_cross_timeframe_features:
                    break
            
            return selected_features
            
        except Exception as e:
            self.logger.warning(f"⚠️ Correlation-based cross-timeframe selection failed: {e}")
            return []
    
    def _identify_timeframe_features(self, feature_names: List[str]) -> List[str]:
        """Identify features that contain timeframe information."""
        timeframe_features = []
        timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        
        for feature_name in feature_names:
            for timeframe in timeframes:
                if timeframe in feature_name.lower():
                    timeframe_features.append(feature_name)
                    break
        
        return timeframe_features
    
    def _are_different_timeframes(self, feat1: str, feat2: str) -> bool:
        """Check if two features are from different timeframes."""
        timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        
        tf1 = None
        tf2 = None
        
        for timeframe in timeframes:
            if timeframe in feat1.lower():
                tf1 = timeframe
            if timeframe in feat2.lower():
                tf2 = timeframe
        
        return tf1 is not None and tf2 is not None and tf1 != tf2
    
    def _calculate_average_synergy_score(self, result: FeatureSelectionResult) -> float:
        """Calculate average synergy score."""
        if not result.interaction_scores:
            return 0.0
        return float(np.mean(list(result.interaction_scores.values())))
    
    def _calculate_average_unique_info_score(self, result: FeatureSelectionResult) -> float:
        """Calculate average unique information score."""
        if not result.polynomial_scores:
            return 0.0
        return float(np.mean(list(result.polynomial_scores.values())))
    
    def _calculate_average_redundancy_score(self, result: FeatureSelectionResult) -> float:
        """Calculate average redundancy score."""
        if not result.cross_timeframe_scores:
            return 0.0
        return float(np.mean(list(result.cross_timeframe_scores.values())))
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            'pid_available': PID_AVAILABLE,
            'matrix_ops_available': MATRIX_OPS_AVAILABLE,
            'numpy_available': NUMPY_AVAILABLE,
            'selection_strategy': self.config.selection_strategy.value
        }