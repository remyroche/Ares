"""
Partial Information Decompositor Module

This module provides partial information decomposition (PID) capabilities for feature selection
and feature engineering. It can identify synergistic, redundant, and unique information
between features and create cross-timeframe, polynomial, and interaction features.

Key capabilities:
- Calculate redundancy, synergy, and unique information between features
- Detect meaningful feature interactions
- Create polynomial features based on PID analysis
- Analyze cross-timeframe dependencies
- Generate interaction features for machine learning models

Author: Feature Selection Framework Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Set
import logging
from datetime import datetime
import time
import warnings
from itertools import combinations, product
from dataclasses import dataclass, field

# Import utilities
try:
    from src.utils.math_validation import (
        safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
        safe_correlation, safe_covariance, safe_mean, safe_std, safe_percentile
    )
    from src.utils.common_operations import create_fallback_logger
except ImportError as e:
    warnings.warn(f"Some utilities not available: {e}")
    # Create fallback implementations
    def safe_divide(a, b): return a / b if b != 0 else 0
    def safe_log(x): return np.log(np.maximum(x, 1e-10))
    def safe_sqrt(x): return np.sqrt(np.maximum(x, 0))
    def safe_power(x, p): return np.power(np.maximum(x, 0), p)
    def validate_finite(x): return np.isfinite(x).all()
    def safe_correlation(x, y): return np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
    def safe_covariance(x, y): return np.cov(x, y)[0, 1] if len(x) > 1 else 0
    def safe_mean(x): return np.mean(x) if len(x) > 0 else 0
    def safe_std(x): return np.std(x) if len(x) > 1 else 0
    def safe_percentile(x, p): return np.percentile(x, p) if len(x) > 0 else 0

# Enhanced dependency management
try:
    from src.utils.logger import get_logger
    _LOGGER = get_logger("FeatureSelection.PartialInformationDecompositor")
except Exception as e:
    _LOGGER = logging.getLogger("FeatureSelection.PartialInformationDecompositor")
    _LOGGER.setLevel(logging.INFO)

logger = _LOGGER

# Try to import sklearn for mutual information calculations
try:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import entropy
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available - limited PID functionality")


@dataclass
class PIDConfig:
    """Configuration for partial information decomposition."""
    # Thresholds for information measures
    synergy_threshold: float = 0.1
    redundancy_threshold: float = 0.15
    unique_info_threshold: float = 0.05
    
    # Cross-timeframe analysis
    cross_timeframe_threshold: float = 0.15
    max_timeframe_lag: int = 5
    
    # Polynomial feature creation
    max_polynomial_degree: int = 3
    max_interaction_features: int = 50
    
    # Computational limits
    max_features_for_full_pid: int = 20
    max_interaction_order: int = 3
    convergence_threshold: float = 1e-6
    max_iterations: int = 100
    
    # Sampling for large datasets
    sample_size: Optional[int] = None
    random_state: int = 42


@dataclass
class PIDResult:
    """Result of partial information decomposition analysis."""
    # Information measures
    redundancy: Dict[Tuple[str, str], float] = field(default_factory=dict)
    synergy: Dict[Tuple[str, str], float] = field(default_factory=dict)
    unique_info: Dict[str, float] = field(default_factory=dict)
    
    # Interaction features
    polynomial_features: List[str] = field(default_factory=list)
    interaction_features: List[str] = field(default_factory=list)
    cross_timeframe_features: List[str] = field(default_factory=list)
    
    # Analysis metadata
    feature_pairs_analyzed: int = 0
    significant_interactions: int = 0
    execution_time: float = 0.0
    convergence_info: Dict[str, Any] = field(default_factory=dict)


class PartialInformationDecompositor:
    """Partial Information Decompositor for feature analysis and creation."""
    
    def __init__(self, config: Optional[PIDConfig] = None):
        """Initialize the partial information decompositor."""
        self.config = config or PIDConfig()
        self.logger = logger.getChild('PartialInformationDecompositor')
        
        _LOGGER.info("🔍 PartialInformationDecompositor initialized")
        _LOGGER.info(f"⚙️ Synergy threshold: {self.config.synergy_threshold}")
        _LOGGER.info(f"⚙️ Redundancy threshold: {self.config.redundancy_threshold}")
        _LOGGER.info(f"⚙️ Max polynomial degree: {self.config.max_polynomial_degree}")
        _LOGGER.info(f"⚙️ Max interaction features: {self.config.max_interaction_features}")

    def decompose_information(self, X: np.ndarray, y: np.ndarray, 
                            feature_names: List[str]) -> PIDResult:
        """
        Perform partial information decomposition on the feature set.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            feature_names: List of feature names
            
        Returns:
            PIDResult containing decomposition analysis and suggested features
        """
        start_time = time.time()
        _LOGGER.info(f"🔍 Starting partial information decomposition...")
        _LOGGER.info(f"📊 Data shape: {X.shape}, Features: {len(feature_names)}")
        
        result = PIDResult()
        
        try:
            # Preprocess data
            X_processed, y_processed = self._preprocess_data(X, y)
            
            # Calculate pairwise information measures
            _LOGGER.info("📊 Calculating pairwise information measures...")
            redundancy, synergy, unique_info = self._calculate_pairwise_pid(
                X_processed, y_processed, feature_names
            )
            
            result.redundancy = redundancy
            result.synergy = synergy
            result.unique_info = unique_info
            
            # Detect significant interactions
            _LOGGER.info("🔍 Detecting significant interactions...")
            significant_pairs = self._detect_significant_interactions(
                synergy, redundancy, feature_names
            )
            
            # Create polynomial features
            _LOGGER.info("🔧 Creating polynomial features...")
            polynomial_features = self._create_polynomial_features(
                X_processed, feature_names, significant_pairs
            )
            
            # Create interaction features
            _LOGGER.info("🔧 Creating interaction features...")
            interaction_features = self._create_interaction_features(
                X_processed, feature_names, significant_pairs
            )
            
            # Analyze cross-timeframe dependencies
            _LOGGER.info("⏰ Analyzing cross-timeframe dependencies...")
            cross_timeframe_features = self._analyze_cross_timeframe_dependencies(
                X_processed, feature_names, significant_pairs
            )
            
            # Compile results
            result.polynomial_features = polynomial_features
            result.interaction_features = interaction_features
            result.cross_timeframe_features = cross_timeframe_features
            result.feature_pairs_analyzed = len(list(combinations(feature_names, 2)))
            result.significant_interactions = len(significant_pairs)
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            _LOGGER.info(f"✅ PID analysis completed in {execution_time:.3f}s")
            _LOGGER.info(f"📊 Significant interactions found: {result.significant_interactions}")
            _LOGGER.info(f"🔧 Generated features: {len(polynomial_features)} polynomial, "
                        f"{len(interaction_features)} interaction, "
                        f"{len(cross_timeframe_features)} cross-timeframe")
            
            return result
            
        except Exception as e:
            _LOGGER.error(f"❌ PID analysis failed: {e}")
            result.execution_time = time.time() - start_time
            return result

    def _preprocess_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for PID analysis."""
        # Handle infinity and NaN values
        X_clean = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        y_clean = np.nan_to_num(y, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Standardize features
        if SKLEARN_AVAILABLE:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clean)
        else:
            # Manual standardization
            X_scaled = (X_clean - np.mean(X_clean, axis=0)) / (np.std(X_clean, axis=0) + 1e-10)
        
        # Sample data if too large
        if self.config.sample_size and len(X_scaled) > self.config.sample_size:
            np.random.seed(self.config.random_state)
            indices = np.random.choice(len(X_scaled), self.config.sample_size, replace=False)
            X_scaled = X_scaled[indices]
            y_clean = y_clean[indices]
            _LOGGER.info(f"📊 Sampled data to {len(X_scaled)} samples")
        
        return X_scaled, y_clean

    def _calculate_pairwise_pid(self, X: np.ndarray, y: np.ndarray, 
                              feature_names: List[str]) -> Tuple[Dict, Dict, Dict]:
        """Calculate pairwise partial information decomposition."""
        redundancy = {}
        synergy = {}
        unique_info = {}
        
        n_features = len(feature_names)
        pairs_analyzed = 0
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                try:
                    feature_i, feature_j = feature_names[i], feature_names[j]
                    
                    # Calculate mutual information
                    mi_i_y = self._calculate_mutual_info(X[:, i], y)
                    mi_j_y = self._calculate_mutual_info(X[:, j], y)
                    mi_ij_y = self._calculate_mutual_info(np.column_stack([X[:, i], X[:, j]]), y)
                    
                    # Calculate redundancy (minimum of individual MIs)
                    redundancy[(feature_i, feature_j)] = min(mi_i_y, mi_j_y)
                    
                    # Calculate synergy (interaction information)
                    synergy[(feature_i, feature_j)] = mi_ij_y - mi_i_y - mi_j_y + redundancy[(feature_i, feature_j)]
                    
                    # Calculate unique information
                    unique_i = mi_i_y - redundancy[(feature_i, feature_j)]
                    unique_j = mi_j_y - redundancy[(feature_i, feature_j)]
                    
                    unique_info[feature_i] = unique_info.get(feature_i, 0) + unique_i
                    unique_info[feature_j] = unique_info.get(feature_j, 0) + unique_j
                    
                    pairs_analyzed += 1
                    
                    if pairs_analyzed % 50 == 0:
                        _LOGGER.debug(f"📊 Analyzed {pairs_analyzed} feature pairs...")
                        
                except Exception as e:
                    _LOGGER.warning(f"⚠️ Failed to analyze pair ({feature_names[i]}, {feature_names[j]}): {e}")
                    continue
        
        _LOGGER.info(f"📊 Completed pairwise PID analysis: {pairs_analyzed} pairs")
        return redundancy, synergy, unique_info

    def _calculate_mutual_info(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate mutual information between features and target."""
        try:
            if not SKLEARN_AVAILABLE:
                # Fallback to correlation-based approximation
                if X.ndim == 1:
                    return abs(safe_correlation(X, y))
                else:
                    # For multi-dimensional X, use average correlation
                    correlations = [abs(safe_correlation(X[:, i], y)) for i in range(X.shape[1])]
                    return safe_mean(correlations)
            
            # Use sklearn for accurate MI calculation
            if X.ndim == 1:
                X_reshaped = X.reshape(-1, 1)
            else:
                X_reshaped = X
            
            # Determine if classification or regression
            unique_y = len(np.unique(y))
            if unique_y <= 10:  # Classification
                mi = mutual_info_classif(X_reshaped, y)[0]
            else:  # Regression
                mi = mutual_info_regression(X_reshaped, y)[0]
            
            return float(mi)
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ MI calculation failed: {e}")
            return 0.0

    def _detect_significant_interactions(self, synergy: Dict, redundancy: Dict, 
                                       feature_names: List[str]) -> List[Tuple[str, str]]:
        """Detect significant feature interactions based on PID measures."""
        significant_pairs = []
        
        for (feat1, feat2), syn_value in synergy.items():
            red_value = redundancy.get((feat1, feat2), 0)
            
            # Check synergy threshold
            if syn_value > self.config.synergy_threshold:
                significant_pairs.append((feat1, feat2))
                _LOGGER.debug(f"🔍 Significant synergy: {feat1} & {feat2} = {syn_value:.4f}")
            
            # Check redundancy threshold
            elif red_value > self.config.redundancy_threshold:
                _LOGGER.debug(f"🔍 High redundancy: {feat1} & {feat2} = {red_value:.4f}")
        
        # Sort by synergy value
        significant_pairs.sort(key=lambda x: synergy.get(x, 0), reverse=True)
        
        # Limit number of interactions
        max_interactions = min(len(significant_pairs), self.config.max_interaction_features)
        significant_pairs = significant_pairs[:max_interactions]
        
        _LOGGER.info(f"📊 Found {len(significant_pairs)} significant interactions")
        return significant_pairs

    def _create_polynomial_features(self, X: np.ndarray, feature_names: List[str],
                                  significant_pairs: List[Tuple[str, str]]) -> List[str]:
        """Create polynomial features based on significant interactions."""
        polynomial_features = []
        
        # Create polynomial features for significant pairs
        for feat1, feat2 in significant_pairs:
            try:
                idx1 = feature_names.index(feat1)
                idx2 = feature_names.index(feat2)
                
                x1, x2 = X[:, idx1], X[:, idx2]
                
                # Create polynomial features up to max degree
                for degree in range(2, self.config.max_polynomial_degree + 1):
                    # x1^degree
                    poly_feat_name = f"{feat1}_pow_{degree}"
                    polynomial_features.append(poly_feat_name)
                    
                    # x2^degree
                    poly_feat_name = f"{feat2}_pow_{degree}"
                    polynomial_features.append(poly_feat_name)
                    
                    # x1 * x2^(degree-1)
                    poly_feat_name = f"{feat1}_x_{feat2}_pow_{degree-1}"
                    polynomial_features.append(poly_feat_name)
                    
                    # x2 * x1^(degree-1)
                    poly_feat_name = f"{feat2}_x_{feat1}_pow_{degree-1}"
                    polynomial_features.append(poly_feat_name)
                
            except ValueError:
                continue
        
        # Remove duplicates and limit
        polynomial_features = list(set(polynomial_features))
        polynomial_features = polynomial_features[:self.config.max_interaction_features]
        
        _LOGGER.info(f"🔧 Generated {len(polynomial_features)} polynomial feature names")
        return polynomial_features

    def _create_interaction_features(self, X: np.ndarray, feature_names: List[str],
                                   significant_pairs: List[Tuple[str, str]]) -> List[str]:
        """Create interaction features based on significant pairs."""
        interaction_features = []
        
        for feat1, feat2 in significant_pairs:
            try:
                idx1 = feature_names.index(feat1)
                idx2 = feature_names.index(feat2)
                
                x1, x2 = X[:, idx1], X[:, idx2]
                
                # Basic interactions
                interaction_features.extend([
                    f"{feat1}_x_{feat2}",
                    f"{feat1}_plus_{feat2}",
                    f"{feat1}_minus_{feat2}",
                    f"{feat1}_ratio_{feat2}",
                    f"sqrt_{feat1}_x_{feat2}",
                    f"log_{feat1}_x_{feat2}"
                ])
                
                # Statistical interactions
                interaction_features.extend([
                    f"{feat1}_x_{feat2}_norm",
                    f"{feat1}_x_{feat2}_std",
                    f"{feat1}_rank_x_{feat2}_rank"
                ])
                
            except ValueError:
                continue
        
        # Remove duplicates and limit
        interaction_features = list(set(interaction_features))
        interaction_features = interaction_features[:self.config.max_interaction_features]
        
        _LOGGER.info(f"🔧 Generated {len(interaction_features)} interaction feature names")
        return interaction_features

    def _analyze_cross_timeframe_dependencies(self, X: np.ndarray, feature_names: List[str],
                                            significant_pairs: List[Tuple[str, str]]) -> List[str]:
        """Analyze cross-timeframe dependencies and create relevant features."""
        cross_timeframe_features = []
        
        # Look for timeframe-related features
        timeframe_features = [f for f in feature_names if any(tf in f.lower() for tf in 
                           ['1m', '5m', '15m', '1h', '4h', '1d', '1w', 'timeframe', 'tf'])]
        
        if len(timeframe_features) < 2:
            _LOGGER.info("📊 No timeframe features detected for cross-timeframe analysis")
            return cross_timeframe_features
        
        # Create cross-timeframe features
        for i, feat1 in enumerate(timeframe_features):
            for feat2 in timeframe_features[i+1:]:
                # Extract timeframes
                tf1 = self._extract_timeframe(feat1)
                tf2 = self._extract_timeframe(feat2)
                
                if tf1 and tf2 and tf1 != tf2:
                    base_name1 = self._remove_timeframe_from_name(feat1)
                    base_name2 = self._remove_timeframe_from_name(feat2)
                    
                    if base_name1 == base_name2:  # Same feature, different timeframes
                        cross_timeframe_features.extend([
                            f"{base_name1}_{tf1}_to_{tf2}_ratio",
                            f"{base_name1}_{tf1}_to_{tf2}_diff",
                            f"{base_name1}_{tf1}_to_{tf2}_corr",
                            f"{base_name1}_{tf1}_x_{tf2}",
                            f"{base_name1}_{tf1}_plus_{tf2}"
                        ])
        
        # Create lag-based features for significant pairs
        for feat1, feat2 in significant_pairs[:10]:  # Limit to top 10
            for lag in range(1, min(self.config.max_timeframe_lag + 1, 4)):
                cross_timeframe_features.extend([
                    f"{feat1}_lag_{lag}_x_{feat2}",
                    f"{feat2}_lag_{lag}_x_{feat1}",
                    f"{feat1}_x_{feat2}_lag_{lag}"
                ])
        
        # Remove duplicates and limit
        cross_timeframe_features = list(set(cross_timeframe_features))
        cross_timeframe_features = cross_timeframe_features[:self.config.max_interaction_features]
        
        _LOGGER.info(f"🔧 Generated {len(cross_timeframe_features)} cross-timeframe feature names")
        return cross_timeframe_features

    def _extract_timeframe(self, feature_name: str) -> Optional[str]:
        """Extract timeframe from feature name."""
        timeframes = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '1w']
        for tf in timeframes:
            if tf in feature_name.lower():
                return tf
        return None

    def _remove_timeframe_from_name(self, feature_name: str) -> str:
        """Remove timeframe suffix from feature name."""
        timeframes = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '1w']
        for tf in timeframes:
            if feature_name.lower().endswith(f"_{tf}"):
                return feature_name[:-len(f"_{tf}")]
        return feature_name

    def generate_feature_matrix(self, X: np.ndarray, feature_names: List[str],
                              pid_result: PIDResult) -> Tuple[np.ndarray, List[str]]:
        """
        Generate expanded feature matrix with polynomial, interaction, and cross-timeframe features.
        
        Args:
            X: Original feature matrix
            feature_names: Original feature names
            pid_result: PID analysis result
            
        Returns:
            Tuple of (expanded_feature_matrix, expanded_feature_names)
        """
        _LOGGER.info("🔧 Generating expanded feature matrix...")
        
        # Start with original features
        expanded_features = [X]
        expanded_names = feature_names.copy()
        
        try:
            # Add polynomial features
            if pid_result.polynomial_features:
                poly_matrix = self._create_polynomial_matrix(X, feature_names, pid_result.polynomial_features)
                if poly_matrix is not None:
                    expanded_features.append(poly_matrix)
                    expanded_names.extend(pid_result.polynomial_features)
            
            # Add interaction features
            if pid_result.interaction_features:
                interaction_matrix = self._create_interaction_matrix(X, feature_names, pid_result.interaction_features)
                if interaction_matrix is not None:
                    expanded_features.append(interaction_matrix)
                    expanded_names.extend(pid_result.interaction_features)
            
            # Add cross-timeframe features
            if pid_result.cross_timeframe_features:
                cross_tf_matrix = self._create_cross_timeframe_matrix(X, feature_names, pid_result.cross_timeframe_features)
                if cross_tf_matrix is not None:
                    expanded_features.append(cross_tf_matrix)
                    expanded_names.extend(pid_result.cross_timeframe_features)
            
            # Combine all features
            if len(expanded_features) > 1:
                expanded_X = np.column_stack(expanded_features)
            else:
                expanded_X = X
            
            _LOGGER.info(f"✅ Expanded feature matrix: {X.shape} → {expanded_X.shape}")
            _LOGGER.info(f"📊 Total features: {len(expanded_names)}")
            
            return expanded_X, expanded_names
            
        except Exception as e:
            _LOGGER.error(f"❌ Feature matrix generation failed: {e}")
            return X, feature_names

    def _create_polynomial_matrix(self, X: np.ndarray, feature_names: List[str],
                                polynomial_features: List[str]) -> Optional[np.ndarray]:
        """Create polynomial feature matrix."""
        try:
            poly_features = []
            
            for poly_name in polynomial_features:
                try:
                    # Parse polynomial feature name
                    if '_pow_' in poly_name:
                        # Single feature power: feature_pow_degree
                        feat_name, degree_str = poly_name.split('_pow_')
                        degree = int(degree_str)
                        
                        if feat_name in feature_names:
                            idx = feature_names.index(feat_name)
                            poly_feat = np.power(X[:, idx], degree)
                            poly_features.append(poly_feat)
                    
                    elif '_x_' in poly_name and '_pow_' in poly_name:
                        # Cross feature with power: feat1_x_feat2_pow_degree
                        parts = poly_name.split('_x_')
                        if len(parts) == 2:
                            feat1 = parts[0]
                            feat2_pow = parts[1]
                            
                            if '_pow_' in feat2_pow:
                                feat2, degree_str = feat2_pow.split('_pow_')
                                degree = int(degree_str)
                                
                                if feat1 in feature_names and feat2 in feature_names:
                                    idx1 = feature_names.index(feat1)
                                    idx2 = feature_names.index(feat2)
                                    poly_feat = X[:, idx1] * np.power(X[:, idx2], degree)
                                    poly_features.append(poly_feat)
                
                except (ValueError, IndexError) as e:
                    _LOGGER.warning(f"⚠️ Failed to create polynomial feature {poly_name}: {e}")
                    continue
            
            if poly_features:
                return np.column_stack(poly_features)
            else:
                return None
                
        except Exception as e:
            _LOGGER.error(f"❌ Polynomial matrix creation failed: {e}")
            return None

    def _create_interaction_matrix(self, X: np.ndarray, feature_names: List[str],
                                 interaction_features: List[str]) -> Optional[np.ndarray]:
        """Create interaction feature matrix."""
        try:
            interaction_feats = []
            
            for interaction_name in interaction_features:
                try:
                    if '_x_' in interaction_name:
                        # Basic multiplication interaction
                        feat1, feat2 = interaction_name.split('_x_', 1)
                        
                        if feat1 in feature_names and feat2 in feature_names:
                            idx1 = feature_names.index(feat1)
                            idx2 = feature_names.index(feat2)
                            interaction_feat = X[:, idx1] * X[:, idx2]
                            interaction_feats.append(interaction_feat)
                    
                    elif '_plus_' in interaction_name:
                        # Addition interaction
                        feat1, feat2 = interaction_name.split('_plus_', 1)
                        
                        if feat1 in feature_names and feat2 in feature_names:
                            idx1 = feature_names.index(feat1)
                            idx2 = feature_names.index(feat2)
                            interaction_feat = X[:, idx1] + X[:, idx2]
                            interaction_feats.append(interaction_feat)
                    
                    elif '_minus_' in interaction_name:
                        # Subtraction interaction
                        feat1, feat2 = interaction_name.split('_minus_', 1)
                        
                        if feat1 in feature_names and feat2 in feature_names:
                            idx1 = feature_names.index(feat1)
                            idx2 = feature_names.index(feat2)
                            interaction_feat = X[:, idx1] - X[:, idx2]
                            interaction_feats.append(interaction_feat)
                    
                    elif '_ratio_' in interaction_name:
                        # Ratio interaction
                        feat1, feat2 = interaction_name.split('_ratio_', 1)
                        
                        if feat1 in feature_names and feat2 in feature_names:
                            idx1 = feature_names.index(feat1)
                            idx2 = feature_names.index(feat2)
                            interaction_feat = safe_divide(X[:, idx1], X[:, idx2])
                            interaction_feats.append(interaction_feat)
                
                except (ValueError, IndexError) as e:
                    _LOGGER.warning(f"⚠️ Failed to create interaction feature {interaction_name}: {e}")
                    continue
            
            if interaction_feats:
                return np.column_stack(interaction_feats)
            else:
                return None
                
        except Exception as e:
            _LOGGER.error(f"❌ Interaction matrix creation failed: {e}")
            return None

    def _create_cross_timeframe_matrix(self, X: np.ndarray, feature_names: List[str],
                                     cross_timeframe_features: List[str]) -> Optional[np.ndarray]:
        """Create cross-timeframe feature matrix."""
        try:
            cross_tf_feats = []
            
            for cross_tf_name in cross_timeframe_features:
                try:
                    if '_to_' in cross_tf_name and '_ratio' in cross_tf_name:
                        # Ratio between timeframes
                        parts = cross_tf_name.split('_to_')
                        if len(parts) == 2:
                            feat1_part = parts[0]
                            feat2_part = parts[1].replace('_ratio', '')
                            
                            # Find matching features
                            matching_feats = self._find_matching_features(feat1_part, feat2_part, feature_names)
                            if len(matching_feats) == 2:
                                idx1 = feature_names.index(matching_feats[0])
                                idx2 = feature_names.index(matching_feats[1])
                                cross_tf_feat = safe_divide(X[:, idx1], X[:, idx2])
                                cross_tf_feats.append(cross_tf_feat)
                    
                    elif '_lag_' in cross_tf_name and '_x_' in cross_tf_name:
                        # Lag-based interaction
                        parts = cross_tf_name.split('_x_')
                        if len(parts) == 2:
                            lag_part, feat2 = parts[0], parts[1]
                            
                            if '_lag_' in lag_part:
                                feat1, lag_str = lag_part.split('_lag_')
                                lag = int(lag_str)
                                
                                if feat1 in feature_names and feat2 in feature_names:
                                    idx1 = feature_names.index(feat1)
                                    idx2 = feature_names.index(feat2)
                                    
                                    # Create lagged version (simple shift)
                                    if lag < X.shape[0]:
                                        lagged_feat1 = np.roll(X[:, idx1], lag)
                                        lagged_feat1[:lag] = 0  # Zero out the rolled values
                                        cross_tf_feat = lagged_feat1 * X[:, idx2]
                                        cross_tf_feats.append(cross_tf_feat)
                
                except (ValueError, IndexError) as e:
                    _LOGGER.warning(f"⚠️ Failed to create cross-timeframe feature {cross_tf_name}: {e}")
                    continue
            
            if cross_tf_feats:
                return np.column_stack(cross_tf_feats)
            else:
                return None
                
        except Exception as e:
            _LOGGER.error(f"❌ Cross-timeframe matrix creation failed: {e}")
            return None

    def _find_matching_features(self, feat1_part: str, feat2_part: str, 
                              feature_names: List[str]) -> List[str]:
        """Find features matching the given patterns."""
        matching = []
        
        for feat_name in feature_names:
            if feat1_part in feat_name:
                matching.append(feat_name)
                break
        
        for feat_name in feature_names:
            if feat2_part in feat_name and feat_name not in matching:
                matching.append(feat_name)
                break
        
        return matching

    def get_feature_importance_scores(self, pid_result: PIDResult) -> Dict[str, float]:
        """Get feature importance scores based on PID analysis."""
        importance_scores = {}
        
        # Unique information scores
        for feature, unique_score in pid_result.unique_info.items():
            importance_scores[feature] = unique_score
        
        # Synergy scores (average for each feature)
        synergy_scores = {}
        for (feat1, feat2), syn_score in pid_result.synergy.items():
            synergy_scores[feat1] = synergy_scores.get(feat1, 0) + syn_score
            synergy_scores[feat2] = synergy_scores.get(feat2, 0) + syn_score
        
        # Normalize synergy scores
        for feature in synergy_scores:
            synergy_scores[feature] /= max(1, len([p for p in pid_result.synergy.keys() if feature in p]))
            importance_scores[feature] = importance_scores.get(feature, 0) + synergy_scores[feature]
        
        return importance_scores

    def save_analysis_results(self, pid_result: PIDResult, output_path: str = None):
        """Save PID analysis results to file with datetime in filename."""
        try:
            import json
            from datetime import datetime
            
            # Generate filename with datetime if not provided
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"pid_analysis_results_{timestamp}.json"
            
            # Convert results to serializable format
            results_dict = {
                'analysis_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'execution_time': pid_result.execution_time,
                    'feature_pairs_analyzed': pid_result.feature_pairs_analyzed,
                    'significant_interactions': pid_result.significant_interactions,
                    'config_used': {
                        'synergy_threshold': self.config.synergy_threshold,
                        'redundancy_threshold': self.config.redundancy_threshold,
                        'unique_info_threshold': self.config.unique_info_threshold,
                        'max_polynomial_degree': self.config.max_polynomial_degree,
                        'max_interaction_features': self.config.max_interaction_features,
                        'cross_timeframe_threshold': self.config.cross_timeframe_threshold
                    }
                },
                'information_measures': {
                    'redundancy': {f"{k[0]}_{k[1]}": v for k, v in pid_result.redundancy.items()},
                    'synergy': {f"{k[0]}_{k[1]}": v for k, v in pid_result.synergy.items()},
                    'unique_info': pid_result.unique_info
                },
                'generated_features': {
                    'polynomial_features': pid_result.polynomial_features,
                    'interaction_features': pid_result.interaction_features,
                    'cross_timeframe_features': pid_result.cross_timeframe_features,
                    'total_generated': (len(pid_result.polynomial_features) + 
                                      len(pid_result.interaction_features) + 
                                      len(pid_result.cross_timeframe_features))
                },
                'convergence_info': pid_result.convergence_info
            }
            
            with open(output_path, 'w') as f:
                json.dump(results_dict, f, indent=2)
            
            _LOGGER.info(f"💾 PID analysis results saved to {output_path}")
            return output_path
            
        except Exception as e:
            _LOGGER.error(f"❌ Failed to save PID results: {e}")
            return None

    def save_feature_matrix_artifact(self, X: np.ndarray, feature_names: List[str], 
                                   pid_result: PIDResult, output_path: str = None):
        """Save expanded feature matrix as artifact with datetime."""
        try:
            from datetime import datetime
            import pandas as pd
            
            # Generate expanded feature matrix
            expanded_X, expanded_names = self.generate_feature_matrix(X, feature_names, pid_result)
            
            # Generate filename with datetime if not provided
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"expanded_feature_matrix_{timestamp}.parquet"
            
            # Create DataFrame with feature names
            df = pd.DataFrame(expanded_X, columns=expanded_names)
            
            # Add metadata
            df.attrs = {
                'original_features': len(feature_names),
                'expanded_features': len(expanded_names),
                'polynomial_features': len(pid_result.polynomial_features),
                'interaction_features': len(pid_result.interaction_features),
                'cross_timeframe_features': len(pid_result.cross_timeframe_features),
                'significant_interactions': pid_result.significant_interactions,
                'creation_timestamp': datetime.now().isoformat(),
                'pid_config': {
                    'synergy_threshold': self.config.synergy_threshold,
                    'redundancy_threshold': self.config.redundancy_threshold,
                    'max_polynomial_degree': self.config.max_polynomial_degree,
                    'max_interaction_features': self.config.max_interaction_features
                }
            }
            
            # Save as parquet for efficiency
            df.to_parquet(output_path, index=False)
            
            _LOGGER.info(f"💾 Expanded feature matrix saved to {output_path}")
            _LOGGER.info(f"📊 Matrix shape: {X.shape} → {expanded_X.shape}")
            _LOGGER.info(f"🔧 Generated features: {len(expanded_names) - len(feature_names)} new features")
            
            return output_path
            
        except Exception as e:
            _LOGGER.error(f"❌ Failed to save feature matrix artifact: {e}")
            return None

    def create_comprehensive_artifact(self, X: np.ndarray, y: np.ndarray, 
                                    feature_names: List[str], pid_result: PIDResult,
                                    output_dir: str = "pid_artifacts") -> Dict[str, str]:
        """Create comprehensive artifacts with datetime in filenames."""
        try:
            import os
            from datetime import datetime
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            artifacts = {}
            
            # 1. Save PID analysis results
            analysis_path = os.path.join(output_dir, f"pid_analysis_{timestamp}.json")
            artifacts['analysis_results'] = self.save_analysis_results(pid_result, analysis_path)
            
            # 2. Save expanded feature matrix
            matrix_path = os.path.join(output_dir, f"expanded_features_{timestamp}.parquet")
            artifacts['feature_matrix'] = self.save_feature_matrix_artifact(X, feature_names, pid_result, matrix_path)
            
            # 3. Save feature importance scores
            importance_scores = self.get_feature_importance_scores(pid_result)
            importance_path = os.path.join(output_dir, f"feature_importance_{timestamp}.json")
            
            importance_data = {
                'timestamp': datetime.now().isoformat(),
                'feature_importance_scores': importance_scores,
                'top_features': sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:20],
                'pid_config': {
                    'synergy_threshold': self.config.synergy_threshold,
                    'redundancy_threshold': self.config.redundancy_threshold,
                    'unique_info_threshold': self.config.unique_info_threshold
                }
            }
            
            with open(importance_path, 'w') as f:
                json.dump(importance_data, f, indent=2)
            artifacts['feature_importance'] = importance_path
            
            # 4. Save interaction summary
            summary_path = os.path.join(output_dir, f"interaction_summary_{timestamp}.json")
            summary_data = {
                'timestamp': datetime.now().isoformat(),
                'significant_interactions': [
                    {
                        'features': f"{pair[0]}_{pair[1]}",
                        'synergy_score': score,
                        'redundancy_score': pid_result.redundancy.get(pair, 0)
                    }
                    for pair, score in sorted(pid_result.synergy.items(), key=lambda x: x[1], reverse=True)
                    if score > self.config.synergy_threshold
                ],
                'feature_generation_summary': {
                    'polynomial_features_count': len(pid_result.polynomial_features),
                    'interaction_features_count': len(pid_result.interaction_features),
                    'cross_timeframe_features_count': len(pid_result.cross_timeframe_features),
                    'total_new_features': (len(pid_result.polynomial_features) + 
                                         len(pid_result.interaction_features) + 
                                         len(pid_result.cross_timeframe_features))
                }
            }
            
            with open(summary_path, 'w') as f:
                json.dump(summary_data, f, indent=2)
            artifacts['interaction_summary'] = summary_path
            
            _LOGGER.info(f"🎉 Comprehensive PID artifacts created in {output_dir}")
            _LOGGER.info(f"📁 Generated {len(artifacts)} artifact files with timestamp {timestamp}")
            
            return artifacts
            
        except Exception as e:
            _LOGGER.error(f"❌ Failed to create comprehensive artifacts: {e}")
            return {}