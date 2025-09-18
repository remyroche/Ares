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

# Pandas dependency (guarded)
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Import PID utilities
try:
    from src.training.utils.feature_selection.partial_information_decompositor import (
        PartialInformationDecompositor, PIDConfig, PIDResult
    )
    PID_AVAILABLE = True
except ImportError as e:
    logging.warning(f"PID utilities not available: {e}")
    PID_AVAILABLE = False

# Import advanced matrix operations
try:
    from src.utils.matrix_operations import (
        get_enhanced_matrix_operations, get_vectorized_processing_core, 
        get_batch_matrix_processor, get_unified_matrix_operations,
        safe_matrix_multiply, safe_correlation_matrix, safe_matrix_inverse,
        gpu_matrix_multiply, correlation_matrix_gpu, eigendecomposition_gpu,
        batch_matrix_multiply, batch_feature_transformation, batch_correlation_analysis,
        optimize_matrix_operation_with_hardware
    )
    MATRIX_OPS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Advanced matrix operations not available: {e}")
    MATRIX_OPS_AVAILABLE = False

# Import common operations for enhanced functionality
try:
    from src.utils.common_operations import (
        safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
        validate_finite, get_memory_usage, timed_operation
    )
    COMMON_OPERATIONS_AVAILABLE = True
except ImportError as e:
    COMMON_OPERATIONS_AVAILABLE = False
    logging.warning(f"Common operations not available: {e}")

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
    
    # Selection Criteria (Initial thresholds)
    min_synergy_score: float = 0.05
    min_unique_info_score: float = 0.02
    max_redundancy_score: float = 0.8
    
    # Dynamic Threshold Adjustment
    enable_dynamic_thresholds: bool = True
    quality_improvement_factor: float = 1.2  # Increase threshold by 20% if new features are better
    min_threshold_improvement: float = 0.01  # Minimum improvement to adjust thresholds
    max_threshold_increase: float = 0.1      # Maximum threshold increase (10%)
    
    # Pre-processing Feature Reference
    reference_feature_rank: int = 150        # Rank of reference feature for comparison
    
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
        """Initialize the feature selection mechanism with advanced matrix operations."""
        self.config = config or FeatureSelectionConfig()
        self.logger = logger.getChild('FeatureSelectionMechanism')
        
        # Initialize matrix operations components
        self.enhanced_matrix_ops = None
        self.vectorized_core = None
        self.batch_processor = None
        
        if MATRIX_OPS_AVAILABLE:
            try:
                self.enhanced_matrix_ops = get_enhanced_matrix_operations()
                self.vectorized_core = get_vectorized_processing_core()
                self.batch_processor = get_batch_matrix_processor()
                self.logger.info("✅ Advanced matrix operations initialized for feature selection")
            except Exception as e:
                self.logger.warning(f"Failed to initialize matrix operations: {e}")
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info("🔧 FeatureSelectionMechanism initialized")
        self.logger.info(f"📊 Selection strategy: {self.config.selection_strategy.value}")
        self.logger.info(f"📊 Max interaction features: {self.config.max_interaction_features}")
        self.logger.info(f"📊 Max polynomial features: {self.config.max_polynomial_features}")
        self.logger.info(f"📊 Max cross-timeframe features: {self.config.max_cross_timeframe_features}")
        self.logger.info(f"🔧 Matrix operations available: {MATRIX_OPS_AVAILABLE}")
        self.logger.info(f"🔧 Common operations available: {COMMON_OPERATIONS_AVAILABLE}")
    
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
            target: Target variable for PID analysis (optional) - now uses multi-horizon profit probabilities
            
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
        """
        Select interaction features based on synergy scores.
        
        Selection Process:
        1. Sort all feature pairs by synergy score (highest first)
        2. Apply minimum synergy threshold filter
        3. Select top N features up to max_interaction_features limit
        4. If more than limit, select the highest-scoring features
        
        Returns:
            List of selected feature pairs, ranked by synergy score
        """
        try:
            # Step 1: Sort synergy scores in descending order (highest synergy first)
            synergy_items = sorted(pid_result.synergy.items(), key=lambda x: x[1], reverse=True)
            
            self.logger.info(f"📊 Analyzing {len(synergy_items)} feature pairs for interaction selection")
            
            # Step 2: Apply threshold and select top features
            selected_features = []
            rejected_count = 0
            
            for (feat1, feat2), synergy_score in synergy_items:
                # Apply minimum synergy threshold
                if synergy_score > self.config.min_synergy_score:
                    selected_features.append((feat1, feat2))
                    
                    # Stop when we reach the limit
                    if len(selected_features) >= self.config.max_interaction_features:
                        self.logger.info(f"📊 Reached interaction feature limit ({self.config.max_interaction_features})")
                        break
                else:
                    rejected_count += 1
            
            # Log selection statistics
            total_analyzed = len(synergy_items)
            selected_count = len(selected_features)
            self.logger.info(f"📊 Interaction feature selection complete:")
            self.logger.info(f"   • Total pairs analyzed: {total_analyzed}")
            self.logger.info(f"   • Selected (synergy > {self.config.min_synergy_score}): {selected_count}")
            self.logger.info(f"   • Rejected (synergy ≤ {self.config.min_synergy_score}): {rejected_count}")
            
            if selected_count > 0:
                top_synergy = synergy_items[0][1] if synergy_items else 0
                self.logger.info(f"   • Highest synergy score: {top_synergy:.4f}")
            
            return selected_features
            
        except Exception as e:
            self.logger.warning(f"⚠️ Interaction feature selection failed: {e}")
            return []
    
    def _select_polynomial_features_from_pid(self, pid_result: PIDResult, feature_names: List[str]) -> List[str]:
        """
        Select polynomial features based on unique information scores.
        
        Selection Process:
        1. Sort all features by unique information score (highest first)
        2. Apply minimum unique information threshold filter
        3. Select top N features up to max_polynomial_features limit
        4. If more than limit, select the highest-scoring features
        
        Returns:
            List of selected features, ranked by unique information score
        """
        try:
            # Step 1: Sort unique information scores in descending order (highest unique info first)
            unique_info_items = sorted(pid_result.unique_info.items(), key=lambda x: x[1], reverse=True)
            
            self.logger.info(f"📊 Analyzing {len(unique_info_items)} features for polynomial selection")
            
            # Step 2: Apply threshold and select top features
            selected_features = []
            rejected_count = 0
            
            for feature, unique_score in unique_info_items:
                # Apply minimum unique information threshold
                if unique_score > self.config.min_unique_info_score:
                    selected_features.append(feature)
                    
                    # Stop when we reach the limit
                    if len(selected_features) >= self.config.max_polynomial_features:
                        self.logger.info(f"📊 Reached polynomial feature limit ({self.config.max_polynomial_features})")
                        break
                else:
                    rejected_count += 1
            
            # Log selection statistics
            total_analyzed = len(unique_info_items)
            selected_count = len(selected_features)
            self.logger.info(f"📊 Polynomial feature selection complete:")
            self.logger.info(f"   • Total features analyzed: {total_analyzed}")
            self.logger.info(f"   • Selected (unique info > {self.config.min_unique_info_score}): {selected_count}")
            self.logger.info(f"   • Rejected (unique info ≤ {self.config.min_unique_info_score}): {rejected_count}")
            
            if selected_count > 0:
                top_unique_info = unique_info_items[0][1] if unique_info_items else 0
                self.logger.info(f"   • Highest unique information score: {top_unique_info:.4f}")
            
            return selected_features
            
        except Exception as e:
            self.logger.warning(f"⚠️ Polynomial feature selection failed: {e}")
            return []
    
    def _select_cross_timeframe_features_from_pid(self, pid_result: PIDResult, feature_names: List[str]) -> List[Tuple[str, str]]:
        """
        Select cross-timeframe features based on cross-timeframe analysis.
        
        Selection Process:
        1. Identify features from different timeframes (1m, 5m, 15m, 30m, 1h, 4h, 1d)
        2. Filter synergy scores for timeframe features only
        3. Sort by synergy score (highest first)
        4. Apply minimum synergy threshold filter
        5. Select top N cross-timeframe pairs up to max_cross_timeframe_features limit
        6. If more than limit, select the highest-scoring cross-timeframe pairs
        
        Returns:
            List of selected cross-timeframe feature pairs, ranked by synergy score
        """
        try:
            # Step 1: Identify timeframe features
            timeframe_features = self._identify_timeframe_features(feature_names)
            self.logger.info(f"📊 Identified {len(timeframe_features)} timeframe features")
            
            # Step 2: Filter synergy scores for timeframe features only
            timeframe_synergy = {
                (feat1, feat2): score for (feat1, feat2), score in pid_result.synergy.items()
                if feat1 in timeframe_features and feat2 in timeframe_features
            }
            
            self.logger.info(f"📊 Found {len(timeframe_synergy)} timeframe feature pairs")
            
            # Step 3: Sort by synergy score in descending order (highest synergy first)
            synergy_items = sorted(timeframe_synergy.items(), key=lambda x: x[1], reverse=True)
            
            # Step 4: Apply threshold and select top cross-timeframe features
            selected_features = []
            rejected_count = 0
            same_timeframe_count = 0
            
            for (feat1, feat2), synergy_score in synergy_items:
                # Apply minimum synergy threshold
                if synergy_score > self.config.min_synergy_score:
                    # Check if features are from different timeframes
                    if self._are_different_timeframes(feat1, feat2):
                        selected_features.append((feat1, feat2))
                        
                        # Stop when we reach the limit
                        if len(selected_features) >= self.config.max_cross_timeframe_features:
                            self.logger.info(f"📊 Reached cross-timeframe feature limit ({self.config.max_cross_timeframe_features})")
                            break
                    else:
                        same_timeframe_count += 1
                else:
                    rejected_count += 1
            
            # Log selection statistics
            total_analyzed = len(synergy_items)
            selected_count = len(selected_features)
            self.logger.info(f"📊 Cross-timeframe feature selection complete:")
            self.logger.info(f"   • Total timeframe pairs analyzed: {total_analyzed}")
            self.logger.info(f"   • Selected cross-timeframe (synergy > {self.config.min_synergy_score}): {selected_count}")
            self.logger.info(f"   • Same timeframe pairs: {same_timeframe_count}")
            self.logger.info(f"   • Rejected (synergy ≤ {self.config.min_synergy_score}): {rejected_count}")
            
            if selected_count > 0:
                top_synergy = synergy_items[0][1] if synergy_items else 0
                self.logger.info(f"   • Highest cross-timeframe synergy score: {top_synergy:.4f}")
            
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
    
    def _adjust_thresholds_dynamically(self, result: FeatureSelectionResult) -> Dict[str, Any]:
        """
        Dynamically adjust thresholds based on feature quality compared to pre-processing features.
        
        Args:
            result: Current feature selection result
            
        Returns:
            Dictionary with threshold adjustment information
        """
        if not self.config.enable_dynamic_thresholds:
            return {'dynamic_adjustment': False, 'reason': 'Dynamic thresholds disabled'}
        
        adjustment_info = {
            'dynamic_adjustment': True,
            'original_thresholds': {
                'min_synergy_score': self.config.min_synergy_score,
                'min_unique_info_score': self.config.min_unique_info_score,
                'max_redundancy_score': self.config.max_redundancy_score
            },
            'adjustments_made': {},
            'quality_improvements': {}
        }
        
        try:
            # Check if we have enough features to make meaningful comparisons
            if result.total_features_analyzed < self.config.reference_feature_rank:
                adjustment_info['reason'] = f'Insufficient features for comparison (need {self.config.reference_feature_rank}, have {result.total_features_analyzed})'
                return adjustment_info
            
            # Calculate quality improvements
            quality_improvements = self._calculate_quality_improvements(result)
            adjustment_info['quality_improvements'] = quality_improvements
            
            # Adjust thresholds based on quality improvements
            adjustments_made = {}
            
            # Adjust synergy threshold
            if quality_improvements.get('synergy_improvement', 0) > self.config.min_threshold_improvement:
                new_synergy_threshold = min(
                    self.config.min_synergy_score * self.config.quality_improvement_factor,
                    self.config.min_synergy_score + self.config.max_threshold_increase
                )
                if new_synergy_threshold > self.config.min_synergy_score:
                    self.config.min_synergy_score = new_synergy_threshold
                    adjustments_made['min_synergy_score'] = {
                        'old': adjustment_info['original_thresholds']['min_synergy_score'],
                        'new': new_synergy_threshold,
                        'improvement': quality_improvements['synergy_improvement']
                    }
            
            # Adjust unique info threshold
            if quality_improvements.get('unique_info_improvement', 0) > self.config.min_threshold_improvement:
                new_unique_info_threshold = min(
                    self.config.min_unique_info_score * self.config.quality_improvement_factor,
                    self.config.min_unique_info_score + self.config.max_threshold_increase
                )
                if new_unique_info_threshold > self.config.min_unique_info_score:
                    self.config.min_unique_info_score = new_unique_info_threshold
                    adjustments_made['min_unique_info_score'] = {
                        'old': adjustment_info['original_thresholds']['min_unique_info_score'],
                        'new': new_unique_info_threshold,
                        'improvement': quality_improvements['unique_info_improvement']
                    }
            
            # Adjust redundancy threshold (lower is better)
            if quality_improvements.get('redundancy_improvement', 0) > self.config.min_threshold_improvement:
                new_redundancy_threshold = max(
                    self.config.max_redundancy_score / self.config.quality_improvement_factor,
                    self.config.max_redundancy_score - self.config.max_threshold_increase
                )
                if new_redundancy_threshold < self.config.max_redundancy_score:
                    self.config.max_redundancy_score = new_redundancy_threshold
                    adjustments_made['max_redundancy_score'] = {
                        'old': adjustment_info['original_thresholds']['max_redundancy_score'],
                        'new': new_redundancy_threshold,
                        'improvement': quality_improvements['redundancy_improvement']
                    }
            
            adjustment_info['adjustments_made'] = adjustments_made
            
            if adjustments_made:
                self.logger.info("🔧 Dynamic threshold adjustments applied:")
                for threshold, info in adjustments_made.items():
                    self.logger.info(f"   • {threshold}: {info['old']:.4f} → {info['new']:.4f} (improvement: {info['improvement']:.4f})")
            else:
                adjustment_info['reason'] = 'No significant quality improvements detected'
            
            return adjustment_info
            
        except Exception as e:
            self.logger.warning(f"⚠️ Dynamic threshold adjustment failed: {e}")
            adjustment_info['error'] = str(e)
            return adjustment_info
    
    def _calculate_quality_improvements(self, result: FeatureSelectionResult) -> Dict[str, float]:
        """
        Calculate quality improvements compared to pre-processing features.
        
        Args:
            result: Current feature selection result
            
        Returns:
            Dictionary with quality improvement metrics
        """
        improvements = {
            'synergy_improvement': 0.0,
            'unique_info_improvement': 0.0,
            'redundancy_improvement': 0.0
        }
        
        try:
            # Calculate improvements based on average scores vs reference
            if result.average_synergy_score > 0:
                # Assume reference synergy score is lower (we want higher synergy)
                reference_synergy = self.config.min_synergy_score * 0.8  # 20% below threshold
                improvements['synergy_improvement'] = max(0, result.average_synergy_score - reference_synergy)
            
            if result.average_unique_info_score > 0:
                # Assume reference unique info score is lower (we want higher unique info)
                reference_unique_info = self.config.min_unique_info_score * 0.8  # 20% below threshold
                improvements['unique_info_improvement'] = max(0, result.average_unique_info_score - reference_unique_info)
            
            if result.average_redundancy_score > 0:
                # Assume reference redundancy score is higher (we want lower redundancy)
                reference_redundancy = self.config.max_redundancy_score * 1.2  # 20% above threshold
                improvements['redundancy_improvement'] = max(0, reference_redundancy - result.average_redundancy_score)
            
            return improvements
            
        except Exception as e:
            self.logger.warning(f"⚠️ Quality improvement calculation failed: {e}")
            return improvements
    
    def get_selection_statistics(self, result: FeatureSelectionResult) -> Dict[str, Any]:
        """
        Get detailed statistics about the feature selection process.
        
        Returns:
            Dictionary with comprehensive selection statistics
        """
        # Calculate dynamic threshold adjustments
        threshold_adjustments = self._adjust_thresholds_dynamically(result)
        
        return {
            'selection_summary': {
                'total_features_analyzed': result.total_features_analyzed,
                'interaction_features_selected': len(result.interaction_features),
                'polynomial_features_selected': len(result.polynomial_features),
                'cross_timeframe_features_selected': len(result.cross_timeframe_features),
                'total_features_selected': (
                    len(result.interaction_features) + 
                    len(result.polynomial_features) + 
                    len(result.cross_timeframe_features)
                )
            },
            'selection_limits': {
                'max_interaction_features': self.config.max_interaction_features,
                'max_polynomial_features': self.config.max_polynomial_features,
                'max_cross_timeframe_features': self.config.max_cross_timeframe_features,
                'total_max_features': (
                    self.config.max_interaction_features + 
                    self.config.max_polynomial_features + 
                    self.config.max_cross_timeframe_features
                )
            },
            'selection_thresholds': {
                'min_synergy_score': self.config.min_synergy_score,
                'min_unique_info_score': self.config.min_unique_info_score,
                'max_redundancy_score': self.config.max_redundancy_score
            },
            'quality_metrics': {
                'average_synergy_score': result.average_synergy_score,
                'average_unique_info_score': result.average_unique_info_score,
                'average_redundancy_score': result.average_redundancy_score
            },
            'selection_efficiency': {
                'interaction_selection_rate': len(result.interaction_features) / self.config.max_interaction_features,
                'polynomial_selection_rate': len(result.polynomial_features) / self.config.max_polynomial_features,
                'cross_timeframe_selection_rate': len(result.cross_timeframe_features) / self.config.max_cross_timeframe_features,
                'overall_selection_rate': (
                    len(result.interaction_features) + 
                    len(result.polynomial_features) + 
                    len(result.cross_timeframe_features)
                ) / (
                    self.config.max_interaction_features + 
                    self.config.max_polynomial_features + 
                    self.config.max_cross_timeframe_features
                )
            },
            'dynamic_threshold_adjustments': threshold_adjustments,
            'execution_info': {
                'selection_time': result.selection_time,
                'selection_strategy': result.selection_strategy.value,
                'pid_analysis_used': result.pid_result is not None
            }
        }
    
    def compute_enhanced_correlation_analysis(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Compute enhanced correlation analysis using advanced matrix operations."""
        try:
            if not MATRIX_OPS_AVAILABLE or not PANDAS_AVAILABLE:
                return {}
            
            results = {}
            
            # Convert to DataFrame for processing
            df = pd.DataFrame(X, columns=feature_names)
            
            if self.enhanced_matrix_ops:
                # Use GPU-accelerated correlation analysis
                corr_matrix = correlation_matrix_gpu(df)
                results['correlation_matrix'] = corr_matrix
                
                # Compute eigendecomposition for feature importance
                eigenvalues, eigenvectors = eigendecomposition_gpu(corr_matrix)
                results['eigenvalues'] = eigenvalues
                results['eigenvectors'] = eigenvectors
                
                # Feature importance based on eigenvalues
                feature_importance = np.abs(eigenvectors).sum(axis=1)
                results['feature_importance'] = dict(zip(feature_names, feature_importance))
            else:
                # Fallback to traditional correlation analysis
                corr_matrix = df.corr()
                results['correlation_matrix'] = corr_matrix
                
                # Compute eigendecomposition
                eigenvalues, eigenvectors = np.linalg.eig(corr_matrix.values)
                results['eigenvalues'] = eigenvalues
                results['eigenvectors'] = eigenvectors
                
                # Feature importance
                feature_importance = np.abs(eigenvectors).sum(axis=1)
                results['feature_importance'] = dict(zip(feature_names, feature_importance))
            
            return results
            
        except Exception as e:
            self.logger.warning(f"Enhanced correlation analysis failed: {e}")
            return {}
    
    def compute_batch_feature_analysis(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Compute feature analysis in batches for large datasets."""
        try:
            if not MATRIX_OPS_AVAILABLE or not self.batch_processor or not PANDAS_AVAILABLE:
                return {}
            
            if X.shape[0] > 1000:
                # Process in batches for memory efficiency
                batch_size = min(500, X.shape[0] // 4)
                batches = [X[i:i+batch_size] for i in range(0, X.shape[0], batch_size)]
                
                batch_results = []
                for batch in batches:
                    batch_df = pd.DataFrame(batch, columns=feature_names)
                    batch_analysis = batch_feature_transformation(batch_df)
                    batch_results.append(batch_analysis)
                
                # Combine batch results
                if batch_results:
                    combined_analysis = np.mean(batch_results, axis=0)
                    return {
                        'batch_feature_analysis': combined_analysis,
                        'n_batches_processed': len(batches),
                        'batch_size': batch_size
                    }
            
            return {}
            
        except Exception as e:
            self.logger.warning(f"Batch feature analysis failed: {e}")
            return {}
    
    def optimize_feature_selection_operations(self, X: np.ndarray, operation_type: str = "correlation") -> Dict[str, Any]:
        """Optimize feature selection operations based on hardware capabilities."""
        try:
            if not MATRIX_OPS_AVAILABLE:
                return {}
            
            optimization_result = optimize_matrix_operation_with_hardware(
                X, operation_type, 
                gpu_enabled=True,
                batch_enabled=True
            )
            
            return optimization_result
            
        except Exception as e:
            self.logger.warning(f"Feature selection operations optimization failed: {e}")
            return {}
    
    def get_enhanced_performance_metrics(self, result: FeatureSelectionResult) -> Dict[str, Any]:
        """Get enhanced performance metrics including matrix operations status."""
        base_metrics = self.get_selection_statistics(result)
        
        enhanced_metrics = {
            **base_metrics,
            'matrix_operations_available': MATRIX_OPS_AVAILABLE,
            'common_operations_available': COMMON_OPERATIONS_AVAILABLE,
            'enhanced_matrix_ops_initialized': self.enhanced_matrix_ops is not None,
            'vectorized_core_initialized': self.vectorized_core is not None,
            'batch_processor_initialized': self.batch_processor is not None,
            'memory_usage': get_memory_usage() if COMMON_OPERATIONS_AVAILABLE else 0.0
        }
        
        return enhanced_metrics