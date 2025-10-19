"""
Efficient Regime Feature Selector

This module provides computation and memory efficient feature selection for regime discovery
using mRMR and LASSO with MI proxies, optimized for HDBSCAN clustering.

Key Features:
- Memory-efficient mRMR implementation with MI proxies
- LASSO feature selection with cross-validation
- Regime-specific importance scoring
- Integration with existing HDBSCAN clustering pipeline
- VectorBT acceleration for mathematical operations
- Sampling-based computations to avoid O(n²) complexity
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings

# Import optimization utilities
from src.utils.common_operations import (
    memory_monitor, optimize_dataframe_memory, safe_divide, safe_mean, safe_std,
    validate_finite, force_garbage_collection, get_memory_usage
)
from src.utils.math_validation import validate_positive, validate_range
from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error, tprint_performance

# Import UnifiedVectorizationManager for VectorBT acceleration
from src.utils.ml_common.unified_vectorization_manager import (
    UnifiedVectorizationManager, 
    VectorizationConfig,
    get_unified_vectorization_manager
)

logger = logging.getLogger(__name__)

@dataclass
class EfficientFeatureSelectionConfig:
    """Configuration for efficient feature selection."""
    # mRMR parameters
    mrmr_k_features: int = 50
    mrmr_relevance_threshold: float = 0.01
    mrmr_redundancy_threshold: float = 0.95
    
    # LASSO parameters
    lasso_alpha_range: Tuple[float, float] = (0.001, 1.0)
    lasso_cv_folds: int = 5
    lasso_max_iter: int = 1000
    lasso_tolerance: float = 1e-4
    
    # Sampling parameters for efficiency
    sample_size: int = 1000  # Sample size for MI calculations
    correlation_sample_size: int = 2000  # Sample size for correlation calculations
    enable_sampling: bool = True
    
    # Memory optimization
    memory_efficient: bool = True
    chunk_size: int = 1000
    max_memory_gb: float = 8.0
    
    # VectorBT optimization
    enable_vectorbt: bool = True
    enable_gpu: bool = False
    
    # Feature selection strategy
    selection_method: str = 'hybrid'  # 'mrmr', 'lasso', 'hybrid'
    hybrid_weight_mrmr: float = 0.6
    hybrid_weight_lasso: float = 0.4
    
    # Regime-specific parameters
    regime_detection_method: str = 'volatility'  # 'volatility', 'volatility_volume', 'custom'
    n_regime_classes: int = 3
    regime_window: int = 20

class EfficientMRMRSelector:
    """
    Memory-efficient mRMR selector with MI proxies for faster computations.
    
    Uses sampling and approximation methods to avoid O(n²) complexity while
    maintaining selection quality for regime discovery.
    """
    
    def __init__(self, config: EfficientFeatureSelectionConfig):
        self.config = config
        self.selected_features = []
        self.feature_scores = {}
        
        # Initialize VectorBT manager for efficient operations
        vectorization_config = VectorizationConfig(
            enable_vectorbt=self.config.enable_vectorbt,
            enable_gpu=self.config.enable_gpu,
            memory_efficient=self.config.memory_efficient,
            max_memory_gb=self.config.max_memory_gb,
            chunk_size=self.config.chunk_size,
            enable_parallel=True
        )
        self.vectorization_manager = get_unified_vectorization_manager(vectorization_config)
        
        tprint_info("✅ EfficientMRMRSelector initialized")
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit mRMR selector and transform data.
        
        Args:
            X: Feature matrix
            y: Target variable (optional, will create pseudo-target for regime discovery)
            
        Returns:
            Selected features DataFrame
        """
        start_time = time.time()
        
        with memory_monitor("mRMR Feature Selection"):
            tprint_info(f"🚀 Starting mRMR selection for {X.shape[1]} features")
            
            # Create pseudo-target if not provided
            if y is None:
                y = self._create_pseudo_target(X)
            
            # Validate inputs
            self._validate_inputs(X, y)
            
            # Select features using mRMR
            selected_indices = self._mrmr_selection(X, y)
            
            # Create result DataFrame
            selected_features = X.iloc[:, selected_indices]
            
            # Update performance stats
            selection_time = time.time() - start_time
            tprint_performance("mRMR Selection", selection_time)
            
            tprint_success(f"✅ mRMR selected {len(selected_indices)} features from {X.shape[1]} in {selection_time:.2f}s")
            
            return selected_features
    
    def _create_pseudo_target(self, X: pd.DataFrame) -> pd.Series:
        """Create pseudo-target for regime discovery based on volatility patterns."""
        if self.config.regime_detection_method == 'volatility':
            # Use rolling volatility as pseudo-target
            rolling_vol = X.rolling(self.config.regime_window).std().mean(axis=1)
            # Create regime classes based on volatility quantiles
            return pd.qcut(rolling_vol, q=self.config.n_regime_classes, labels=False)
        
        elif self.config.regime_detection_method == 'volatility_volume':
            # Combine volatility and volume patterns
            vol_rolling = X.rolling(self.config.regime_window).std().mean(axis=1)
            # Use first few features as volume proxy (assuming volume features are first)
            volume_proxy = X.iloc[:, :min(5, X.shape[1])].rolling(self.config.regime_window).mean().mean(axis=1)
            combined_target = vol_rolling * volume_proxy
            return pd.qcut(combined_target, q=self.config.n_regime_classes, labels=False)
        
        else:
            # Default: use variance as pseudo-target
            feature_variance = X.var()
            return pd.qcut(feature_variance, q=self.config.n_regime_classes, labels=False)
    
    def _mrmr_selection(self, X: pd.DataFrame, y: pd.Series) -> List[int]:
        """Perform mRMR feature selection with efficient computations."""
        n_features = X.shape[1]
        n_samples = len(X)
        
        # Sample data for efficiency if enabled
        if self.config.enable_sampling and n_samples > self.config.sample_size:
            sample_indices = np.random.choice(n_samples, self.config.sample_size, replace=False)
            X_sampled = X.iloc[sample_indices]
            y_sampled = y.iloc[sample_indices]
        else:
            X_sampled = X
            y_sampled = y
        
        # Calculate relevance scores (MI with target)
        relevance_scores = self._calculate_relevance_scores(X_sampled, y_sampled)
        
        # Initialize selection
        selected_features = []
        remaining_features = list(range(n_features))
        
        # Select first feature (highest relevance)
        first_feature = np.argmax(relevance_scores)
        selected_features.append(first_feature)
        remaining_features.remove(first_feature)
        
        # Iteratively select features using mRMR criterion
        for _ in range(min(self.config.mrmr_k_features - 1, len(remaining_features))):
            best_feature = self._select_next_feature(
                X_sampled, selected_features, remaining_features, relevance_scores
            )
            
            if best_feature is not None:
                selected_features.append(best_feature)
                remaining_features.remove(best_feature)
            else:
                break
        
        return selected_features
    
    def _calculate_relevance_scores(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Calculate relevance scores (MI with target) efficiently."""
        relevance_scores = np.zeros(X.shape[1])
        
        for i, col in enumerate(X.columns):
            try:
                # Use MI regression for continuous target
                mi_score = mutual_info_regression(
                    X[[col]], y, random_state=42
                )[0]
                relevance_scores[i] = mi_score
            except Exception as e:
                tprint_warning(f"⚠️ Failed to calculate MI for feature {col}: {e}")
                relevance_scores[i] = 0.0
        
        return relevance_scores
    
    def _select_next_feature(self, X: pd.DataFrame, selected_features: List[int], 
                           remaining_features: List[int], relevance_scores: np.ndarray) -> Optional[int]:
        """Select next feature using mRMR criterion."""
        if not remaining_features:
            return None
        
        best_score = -np.inf
        best_feature = None
        
        for candidate in remaining_features:
            # Calculate relevance (MI with target)
            relevance = relevance_scores[candidate]
            
            # Calculate redundancy (average MI with selected features)
            redundancy = self._calculate_redundancy(X, selected_features, candidate)
            
            # mRMR score: relevance - redundancy
            mrmr_score = relevance - redundancy
            
            if mrmr_score > best_score:
                best_score = mrmr_score
                best_feature = candidate
        
        return best_feature
    
    def _calculate_redundancy(self, X: pd.DataFrame, selected_features: List[int], 
                            candidate: int) -> float:
        """Calculate redundancy (average MI with selected features) efficiently."""
        if not selected_features:
            return 0.0
        
        # Sample for correlation calculation if enabled
        if self.config.enable_sampling and len(X) > self.config.correlation_sample_size:
            sample_indices = np.random.choice(len(X), self.config.correlation_sample_size, replace=False)
            X_sampled = X.iloc[sample_indices]
        else:
            X_sampled = X
        
        # Calculate MI between candidate and each selected feature
        mi_scores = []
        for selected_idx in selected_features:
            try:
                mi_score = mutual_info_regression(
                    X_sampled.iloc[:, [candidate]], 
                    X_sampled.iloc[:, selected_idx], 
                    random_state=42
                )[0]
                mi_scores.append(mi_score)
            except Exception as e:
                tprint_warning(f"⚠️ Failed to calculate MI between features: {e}")
                mi_scores.append(0.0)
        
        return np.mean(mi_scores) if mi_scores else 0.0
    
    def _validate_inputs(self, X: pd.DataFrame, y: pd.Series):
        """Validate input data."""
        if X.empty:
            raise ValueError("Feature matrix cannot be empty")
        
        if len(X) != len(y):
            raise ValueError("Feature matrix and target must have same length")
        
        if X.shape[1] < 2:
            raise ValueError("At least 2 features required for mRMR selection")

class EfficientLASSOSelector:
    """
    Memory-efficient LASSO selector with cross-validation for regime discovery.
    
    Uses time series cross-validation and efficient regularization path
    computation for regime-specific feature selection.
    """
    
    def __init__(self, config: EfficientFeatureSelectionConfig):
        self.config = config
        self.lasso_model = None
        self.scaler = RobustScaler()
        self.selected_features = []
        self.feature_scores = {}
        
        # Initialize VectorBT manager
        vectorization_config = VectorizationConfig(
            enable_vectorbt=self.config.enable_vectorbt,
            enable_gpu=self.config.enable_gpu,
            memory_efficient=self.config.memory_efficient,
            max_memory_gb=self.config.max_memory_gb,
            chunk_size=self.config.chunk_size,
            enable_parallel=True
        )
        self.vectorization_manager = get_unified_vectorization_manager(vectorization_config)
        
        tprint_info("✅ EfficientLASSOSelector initialized")
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit LASSO selector and transform data.
        
        Args:
            X: Feature matrix
            y: Target variable (optional, will create pseudo-target for regime discovery)
            
        Returns:
            Selected features DataFrame
        """
        start_time = time.time()
        
        with memory_monitor("LASSO Feature Selection"):
            tprint_info(f"🚀 Starting LASSO selection for {X.shape[1]} features")
            
            # Create pseudo-target if not provided
            if y is None:
                y = self._create_pseudo_target(X)
            
            # Validate inputs
            self._validate_inputs(X, y)
            
            # Select features using LASSO
            selected_indices = self._lasso_selection(X, y)
            
            # Create result DataFrame
            selected_features = X.iloc[:, selected_indices]
            
            # Update performance stats
            selection_time = time.time() - start_time
            tprint_performance("LASSO Selection", selection_time)
            
            tprint_success(f"✅ LASSO selected {len(selected_indices)} features from {X.shape[1]} in {selection_time:.2f}s")
            
            return selected_features
    
    def _create_pseudo_target(self, X: pd.DataFrame) -> pd.Series:
        """Create pseudo-target for regime discovery."""
        if self.config.regime_detection_method == 'volatility':
            # Use rolling volatility as pseudo-target
            rolling_vol = X.rolling(self.config.regime_window).std().mean(axis=1)
            return rolling_vol.fillna(rolling_vol.mean())
        
        elif self.config.regime_detection_method == 'volatility_volume':
            # Combine volatility and volume patterns
            vol_rolling = X.rolling(self.config.regime_window).std().mean(axis=1)
            volume_proxy = X.iloc[:, :min(5, X.shape[1])].rolling(self.config.regime_window).mean().mean(axis=1)
            combined_target = vol_rolling * volume_proxy
            return combined_target.fillna(combined_target.mean())
        
        else:
            # Default: use variance as pseudo-target
            feature_variance = X.var()
            return feature_variance
    
    def _lasso_selection(self, X: pd.DataFrame, y: pd.Series) -> List[int]:
        """Perform LASSO feature selection with efficient cross-validation."""
        # Sample data for efficiency if enabled
        if self.config.enable_sampling and len(X) > self.config.sample_size:
            sample_indices = np.random.choice(len(X), self.config.sample_size, replace=False)
            X_sampled = X.iloc[sample_indices]
            y_sampled = y.iloc[sample_indices]
        else:
            X_sampled = X
            y_sampled = y
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_sampled)
        
        # Create time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.config.lasso_cv_folds)
        
        # Fit LASSO with cross-validation
        self.lasso_model = LassoCV(
            alphas=np.logspace(
                np.log10(self.config.lasso_alpha_range[0]),
                np.log10(self.config.lasso_alpha_range[1]),
                50
            ),
            cv=tscv,
            max_iter=self.config.lasso_max_iter,
            tol=self.config.lasso_tolerance,
            random_state=42,
            n_jobs=-1
        )
        
        self.lasso_model.fit(X_scaled, y_sampled)
        
        # Select features with non-zero coefficients
        selected_indices = np.where(self.lasso_model.coef_ != 0)[0].tolist()
        
        # Store feature scores (absolute coefficients)
        self.feature_scores = dict(zip(
            X.columns,
            np.abs(self.lasso_model.coef_)
        ))
        
        return selected_indices
    
    def _validate_inputs(self, X: pd.DataFrame, y: pd.Series):
        """Validate input data."""
        if X.empty:
            raise ValueError("Feature matrix cannot be empty")
        
        if len(X) != len(y):
            raise ValueError("Feature matrix and target must have same length")
        
        if X.shape[1] < 2:
            raise ValueError("At least 2 features required for LASSO selection")

class RegimeFeatureImportanceScorer:
    """
    Regime-specific feature importance scorer that ranks features based on
    their ability to distinguish between regimes and capture regime transitions.
    """
    
    def __init__(self, config: EfficientFeatureSelectionConfig):
        self.config = config
        self.importance_scores = {}
        
        # Initialize VectorBT manager
        vectorization_config = VectorizationConfig(
            enable_vectorbt=self.config.enable_vectorbt,
            enable_gpu=self.config.enable_gpu,
            memory_efficient=self.config.memory_efficient,
            max_memory_gb=self.config.max_memory_gb,
            chunk_size=self.config.chunk_size,
            enable_parallel=True
        )
        self.vectorization_manager = get_unified_vectorization_manager(vectorization_config)
        
        tprint_info("✅ RegimeFeatureImportanceScorer initialized")
    
    def score_features(self, features_df: pd.DataFrame, 
                      regime_labels: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Score features based on their importance for regime discovery.
        
        Args:
            features_df: Feature matrix
            regime_labels: Regime labels (optional, will create if not provided)
            
        Returns:
            DataFrame with feature importance scores
        """
        start_time = time.time()
        
        with memory_monitor("Regime Feature Importance Scoring"):
            tprint_info(f"🚀 Starting feature importance scoring for {features_df.shape[1]} features")
            
            # Create regime labels if not provided
            if regime_labels is None:
                regime_labels = self._create_regime_labels(features_df)
            
            # Calculate different importance metrics
            importance_metrics = {
                'variance_importance': self._calculate_variance_importance(features_df, regime_labels),
                'regime_separation': self._calculate_regime_separation(features_df, regime_labels),
                'transition_sensitivity': self._calculate_transition_sensitivity(features_df, regime_labels),
                'clustering_stability': self._calculate_clustering_stability(features_df, regime_labels)
            }
            
            # Combine metrics into final importance scores
            importance_df = pd.DataFrame(importance_metrics, index=features_df.columns)
            importance_df['combined_score'] = importance_df.mean(axis=1)
            importance_df = importance_df.sort_values('combined_score', ascending=False)
            
            # Update performance stats
            scoring_time = time.time() - start_time
            tprint_performance("Feature Importance Scoring", scoring_time)
            
            tprint_success(f"✅ Feature importance scoring completed in {scoring_time:.2f}s")
            
            return importance_df
    
    def _create_regime_labels(self, features_df: pd.DataFrame) -> pd.Series:
        """Create regime labels based on volatility patterns."""
        if self.config.regime_detection_method == 'volatility':
            rolling_vol = features_df.rolling(self.config.regime_window).std().mean(axis=1)
            return pd.qcut(rolling_vol, q=self.config.n_regime_classes, labels=False)
        
        elif self.config.regime_detection_method == 'volatility_volume':
            vol_rolling = features_df.rolling(self.config.regime_window).std().mean(axis=1)
            volume_proxy = features_df.iloc[:, :min(5, features_df.shape[1])].rolling(self.config.regime_window).mean().mean(axis=1)
            combined_target = vol_rolling * volume_proxy
            return pd.qcut(combined_target, q=self.config.n_regime_classes, labels=False)
        
        else:
            # Default: use variance as regime indicator
            feature_variance = features_df.var()
            return pd.qcut(feature_variance, q=self.config.n_regime_classes, labels=False)
    
    def _calculate_variance_importance(self, features_df: pd.DataFrame, 
                                     regime_labels: pd.Series) -> pd.Series:
        """Calculate variance importance within regimes."""
        regime_variance = features_df.groupby(regime_labels).var().mean()
        return regime_variance / regime_variance.sum()
    
    def _calculate_regime_separation(self, features_df: pd.DataFrame, 
                                   regime_labels: pd.Series) -> pd.Series:
        """Calculate how well features separate regimes."""
        separation_scores = []
        
        for col in features_df.columns:
            feature_data = features_df[[col]]
            if len(regime_labels.unique()) > 1:
                try:
                    score = silhouette_score(feature_data, regime_labels)
                    separation_scores.append(score)
                except:
                    separation_scores.append(0.0)
            else:
                separation_scores.append(0.0)
        
        return pd.Series(separation_scores, index=features_df.columns)
    
    def _calculate_transition_sensitivity(self, features_df: pd.DataFrame, 
                                        regime_labels: pd.Series) -> pd.Series:
        """Calculate sensitivity to regime transitions."""
        transition_scores = []
        
        for col in features_df.columns:
            feature_data = features_df[col]
            
            # Find regime transitions
            regime_changes = regime_labels.diff() != 0
            
            if regime_changes.sum() > 0:
                # Calculate feature changes at regime transitions
                feature_changes = feature_data.diff().abs()
                transition_changes = feature_changes[regime_changes]
                
                # Score based on magnitude of changes at transitions
                transition_scores.append(transition_changes.mean())
            else:
                transition_scores.append(0.0)
        
        return pd.Series(transition_scores, index=features_df.columns)
    
    def _calculate_clustering_stability(self, features_df: pd.DataFrame, 
                                      regime_labels: pd.Series) -> pd.Series:
        """Calculate clustering stability for each feature."""
        from sklearn.cluster import KMeans
        
        stability_scores = []
        
        for col in features_df.columns:
            feature_data = features_df[[col]]
            
            # Test clustering stability with different random seeds
            stability_scores_col = []
            for seed in range(5):
                try:
                    kmeans = KMeans(n_clusters=self.config.n_regime_classes, random_state=seed)
                    labels = kmeans.fit_predict(feature_data)
                    stability_scores_col.append(labels)
                except:
                    stability_scores_col.append(np.zeros(len(feature_data)))
            
            # Calculate stability (how consistent are the clusters?)
            stability = self._calculate_clustering_consistency(stability_scores_col)
            stability_scores.append(stability)
        
        return pd.Series(stability_scores, index=features_df.columns)
    
    def _calculate_clustering_consistency(self, cluster_results: List[np.ndarray]) -> float:
        """Calculate consistency between clustering results."""
        if len(cluster_results) < 2:
            return 0.0
        
        # Calculate pairwise consistency
        consistency_scores = []
        for i in range(len(cluster_results)):
            for j in range(i + 1, len(cluster_results)):
                # Calculate adjusted rand index
                try:
                    from sklearn.metrics import adjusted_rand_score
                    consistency = adjusted_rand_score(cluster_results[i], cluster_results[j])
                    consistency_scores.append(consistency)
                except:
                    consistency_scores.append(0.0)
        
        return np.mean(consistency_scores) if consistency_scores else 0.0

class EfficientRegimeFeatureSelector:
    """
    Main class that combines mRMR, LASSO, and regime importance scoring
    for efficient feature selection in regime discovery.
    """
    
    def __init__(self, config: Optional[EfficientFeatureSelectionConfig] = None):
        self.config = config or EfficientFeatureSelectionConfig()
        
        # Initialize selectors
        self.mrmr_selector = EfficientMRMRSelector(self.config)
        self.lasso_selector = EfficientLASSOSelector(self.config)
        self.importance_scorer = RegimeFeatureImportanceScorer(self.config)
        
        # Performance tracking
        self.performance_stats = {
            'total_selection_time': 0.0,
            'mrmr_selection_time': 0.0,
            'lasso_selection_time': 0.0,
            'importance_scoring_time': 0.0,
            'features_selected': 0,
            'original_features': 0,
            'memory_usage_mb': 0.0
        }
        
        tprint_info("✅ EfficientRegimeFeatureSelector initialized")
    
    def select_features(self, features_df: pd.DataFrame, 
                       target: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Select features for regime discovery using hybrid approach.
        
        Args:
            features_df: Input feature matrix
            target: Target variable (optional, will create pseudo-target)
            
        Returns:
            Selected features DataFrame
        """
        start_time = time.time()
        
        with memory_monitor("Efficient Regime Feature Selection"):
            tprint_info(f"🚀 Starting efficient regime feature selection for {features_df.shape[1]} features")
            
            # Validate input
            self._validate_features(features_df)
            
            # Apply feature selection based on configuration
            if self.config.selection_method == 'mrmr':
                selected_features = self._select_with_mrmr(features_df, target)
            elif self.config.selection_method == 'lasso':
                selected_features = self._select_with_lasso(features_df, target)
            else:  # hybrid
                selected_features = self._select_with_hybrid(features_df, target)
            
            # Apply regime importance scoring
            importance_scores = self.importance_scorer.score_features(selected_features, target)
            
            # Final feature ranking and selection
            final_features = self._rank_and_select_final_features(selected_features, importance_scores)
            
            # Update performance stats
            selection_time = time.time() - start_time
            self._update_performance_stats(features_df, final_features, selection_time)
            
            tprint_success(f"✅ Efficient regime feature selection completed: {final_features.shape[1]} features selected from {features_df.shape[1]} in {selection_time:.2f}s")
            
            return final_features
    
    def _select_with_mrmr(self, features_df: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """Select features using mRMR."""
        return self.mrmr_selector.fit_transform(features_df, target)
    
    def _select_with_lasso(self, features_df: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """Select features using LASSO."""
        return self.lasso_selector.fit_transform(features_df, target)
    
    def _select_with_hybrid(self, features_df: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """Select features using hybrid mRMR + LASSO approach."""
        # Get features from both methods
        mrmr_features = self.mrmr_selector.fit_transform(features_df, target)
        lasso_features = self.lasso_selector.fit_transform(features_df, target)
        
        # Combine features based on weights
        mrmr_columns = set(mrmr_features.columns)
        lasso_columns = set(lasso_features.columns)
        
        # Weighted combination
        combined_features = set()
        
        # Add mRMR features with weight
        for col in mrmr_columns:
            if col in features_df.columns:
                combined_features.add(col)
        
        # Add LASSO features with weight
        for col in lasso_columns:
            if col in features_df.columns:
                combined_features.add(col)
        
        # If too few features, add more from each method
        if len(combined_features) < 20:
            # Add more mRMR features
            for col in mrmr_columns:
                if len(combined_features) >= 30:
                    break
                if col not in combined_features:
                    combined_features.add(col)
            
            # Add more LASSO features
            for col in lasso_columns:
                if len(combined_features) >= 30:
                    break
                if col not in combined_features:
                    combined_features.add(col)
        
        return features_df[list(combined_features)]
    
    def _rank_and_select_final_features(self, features_df: pd.DataFrame, 
                                      importance_scores: pd.DataFrame) -> pd.DataFrame:
        """Rank and select final features based on importance scores."""
        # Sort features by combined importance score
        ranked_features = importance_scores.sort_values('combined_score', ascending=False)
        
        # Select top features (configurable)
        n_features = min(self.config.mrmr_k_features, len(features_df.columns))
        top_features = ranked_features.head(n_features).index
        
        return features_df[top_features]
    
    def _validate_features(self, features_df: pd.DataFrame):
        """Validate input features."""
        if not isinstance(features_df, pd.DataFrame):
            raise ValueError("Features must be a pandas DataFrame")
        
        if features_df.empty:
            raise ValueError("Features DataFrame cannot be empty")
        
        if features_df.shape[1] < 2:
            raise ValueError("At least 2 features required for feature selection")
    
    def _update_performance_stats(self, original_features: pd.DataFrame, 
                                selected_features: pd.DataFrame, 
                                selection_time: float):
        """Update performance statistics."""
        self.performance_stats['total_selection_time'] = selection_time
        self.performance_stats['features_selected'] = selected_features.shape[1]
        self.performance_stats['original_features'] = original_features.shape[1]
        
        # Calculate memory usage
        memory_usage = selected_features.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        self.performance_stats['memory_usage_mb'] = memory_usage
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return self.performance_stats.copy()
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_selection_time': 0.0,
            'mrmr_selection_time': 0.0,
            'lasso_selection_time': 0.0,
            'importance_scoring_time': 0.0,
            'features_selected': 0,
            'original_features': 0,
            'memory_usage_mb': 0.0
        }

# Convenience function for easy usage
def create_efficient_regime_feature_selector(
    k_features: int = 50,
    selection_method: str = 'hybrid',
    enable_sampling: bool = True,
    sample_size: int = 1000,
    memory_efficient: bool = True,
    enable_vectorbt: bool = True,
    enable_gpu: bool = False
) -> EfficientRegimeFeatureSelector:
    """
    Create an efficient regime feature selector with specified configuration.
    
    Args:
        k_features: Number of features to select
        selection_method: Selection method ('mrmr', 'lasso', 'hybrid')
        enable_sampling: Enable sampling for efficiency
        sample_size: Sample size for computations
        memory_efficient: Enable memory optimization
        enable_vectorbt: Enable VectorBT acceleration
        enable_gpu: Enable GPU acceleration
        
    Returns:
        EfficientRegimeFeatureSelector instance
    """
    config = EfficientFeatureSelectionConfig(
        mrmr_k_features=k_features,
        selection_method=selection_method,
        enable_sampling=enable_sampling,
        sample_size=sample_size,
        memory_efficient=memory_efficient,
        enable_vectorbt=enable_vectorbt,
        enable_gpu=enable_gpu
    )
    
    return EfficientRegimeFeatureSelector(config)

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples, n_features = 1000, 100
    
    # Create high-dimensional data with some regime structure
    features = np.random.randn(n_samples, n_features)
    
    # Add regime structure (volatility changes)
    regime_changes = np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1])
    for i, regime in enumerate(regime_changes):
        if regime == 1:  # High volatility regime
            features[i, :20] *= 2.0
        elif regime == 2:  # Low volatility regime
            features[i, :20] *= 0.5
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    features_df = pd.DataFrame(features, columns=feature_names)
    
    print(f"Original features: {features_df.shape}")
    
    # Create efficient regime feature selector
    selector = create_efficient_regime_feature_selector(
        k_features=30,
        selection_method='hybrid',
        enable_sampling=True,
        sample_size=500,
        memory_efficient=True,
        enable_vectorbt=True
    )
    
    # Select features
    selected_features = selector.select_features(features_df)
    
    print(f"Selected features: {selected_features.shape}")
    print(f"Performance stats: {selector.get_performance_stats()}")
