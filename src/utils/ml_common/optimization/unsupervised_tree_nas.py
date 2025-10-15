"""
import warnings
Unsupervised Tree-Based NAS for Regime Detection and Qualification

This module provides unsupervised tree-based architecture search specifically
designed for regime detection and qualification without requiring labeled data.

Key Features:
- Unsupervised clustering with tree-based models
- Regime detection using feature similarity
- Regime qualification and quality assessment
- Automatic regime transition detection
- Regime persistence and stability analysis
- Integration with existing hybrid NAS system
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime
from abc import ABC, abstractmethod
import json
from pathlib import Path
from src.utils.tprint import (tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_progress, tprint_performance, tprint_timer)

# Unsupervised learning imports
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE, UMAP
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Tree-based model imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    import catboost as cb

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

except ImportError:
    
    cp = None
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    cb = None

logger = logging.getLogger(__name__)

@dataclass
class UnsupervisedTreeNASConfig:
    """Configuration for unsupervised tree-based NAS."""
    
    # Clustering algorithms
    clustering_algorithms: List[str] = field(default_factory=lambda: [
        'kmeans', 'dbscan', 'gaussian_mixture', 'agglomerative', 'isolation_forest'
    ])
    
    # Regime detection parameters
    n_regimes_range: Tuple[int, int] = (3, 15)
    min_regime_duration: int = 5  # Minimum samples per regime
    max_regime_duration: int = 100  # Maximum samples per regime
    regime_stability_threshold: float = 0.7
    
    # Feature engineering
    feature_engineering_methods: List[str] = field(default_factory=lambda: [
        'technical_indicators', 'price_features', 'volume_features', 
        'volatility_features', 'momentum_features', 'trend_features'
    ])
    
    # Dimensionality reduction
    dimensionality_reduction: Optional[str] = None  # 'pca', 'ica', 'tsne', 'umap'
    n_components: int = 10
    
    # Regime qualification
    qualification_metrics: List[str] = field(default_factory=lambda: [
        'silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score',
        'regime_persistence', 'regime_separation', 'regime_consistency'
    ])
    
    # Quality thresholds
    quality_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'min_silhouette_score': 0.3,
        'min_calinski_harabasz_score': 100.0,
        'max_davies_bouldin_score': 2.0,
        'min_regime_persistence': 0.6,
        'min_regime_separation': 0.5,
        'min_regime_consistency': 0.7
    })
    
    # Tree-based model parameters
    tree_models: List[str] = field(default_factory=lambda: ['random_forest', 'xgboost', 'lightgbm'])
    tree_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2
    })
    
    # Optimization settings
    n_trials: int = 50
    timeout_seconds: int = 1800
    cv_folds: int = 5
    
    # Performance settings
    n_jobs: int = -1
    memory_limit_gb: float = 8.0

@dataclass
class RegimeCandidate:
    """A candidate regime detected by unsupervised methods."""
    
    # Regime identification
    regime_id: int
    regime_type: str  # 'bull', 'bear', 'sideways', 'volatile', 'trending'
    regime_confidence: float
    
    # Temporal information
    start_time: datetime
    end_time: datetime
    duration: int  # Number of samples
    
    # Spatial information
    regime_center: np.ndarray
    regime_boundary: np.ndarray
    regime_size: int
    
    # Quality metrics
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    regime_persistence: float
    regime_separation: float
    regime_consistency: float
    overall_quality: float
    
    # Feature importance
    feature_importance: Dict[str, float]
    key_features: List[str]
    
    # Transition information
    transition_probability: float
    transition_targets: List[int]
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    sample_indices: List[int] = field(default_factory=list)

@dataclass
class UnsupervisedArchitectureCandidate:
    """A candidate unsupervised architecture."""
    
    # Architecture definition
    clustering_algorithm: str
    clustering_params: Dict[str, Any]
    feature_engineering: Dict[str, Any]
    dimensionality_reduction: Optional[Dict[str, Any]]
    tree_model: str
    tree_params: Dict[str, Any]
    
    # Performance metrics
    clustering_quality: float
    regime_detection_accuracy: float
    regime_qualification_score: float
    overall_score: float
    
    # Detected regimes
    regimes: List[RegimeCandidate]
    n_regimes: int
    
    # Training info
    training_time: float
    feature_importance: Dict[str, float]
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    trial_number: int = 0

class UnsupervisedTreeNAS:
    """Unsupervised Tree-Based NAS for regime detection and qualification."""
    
    def __init__(self, config: UnsupervisedTreeNASConfig):
        """Initialize unsupervised tree-based NAS."""
        tprint("🚀 [UNSUPERVISED_TREE_NAS] Initializing Unsupervised Tree-Based NAS", color="cyan", bold=True)
        tprint(f"📊 [UNSUPERVISED_TREE_NAS] Trials: {config.n_trials}", color="blue")
        tprint(f"📊 [UNSUPERVISED_TREE_NAS] Max clusters: {config.max_clusters}", color="blue")
        tprint(f"📊 [UNSUPERVISED_TREE_NAS] Min cluster size: {config.min_cluster_size}", color="blue")
        self.config = config
        self.logger = logger.getChild('UnsupervisedTreeNAS')
        self.candidates = []
        self.best_candidate = None
        
        tprint("✅ [UNSUPERVISED_TREE_NAS] Unsupervised Tree-Based NAS initialized successfully", color="green")
        self.logger.info(f"✅ Unsupervised Tree-Based NAS initialized with {config.n_trials} trials")
    
    def search(self, 
               market_data: pd.DataFrame,
               timestamps: Optional[np.ndarray] = None) -> UnsupervisedArchitectureCandidate:
        """
        Perform unsupervised architecture search for regime detection.
        
        Args:
            market_data: Market data (OHLCV)
            timestamps: Timestamps for the data (optional)
            
        Returns:
            Best unsupervised architecture candidate
        """
        tprint("🚀 [UNSUPERVISED_TREE_NAS] Starting Unsupervised Tree-Based NAS Search", color="cyan", bold=True)
        tprint(f"📊 [UNSUPERVISED_TREE_NAS] Market data shape: {market_data.shape}", color="blue")
        self.logger.info("🚀 Starting Unsupervised Tree-Based NAS Search...")
        start_time = time.time()
        
        try:
            # Prepare data
            tprint("🔧 [UNSUPERVISED_TREE_NAS] Preparing features from market data", color="yellow")
            X, feature_names = self._prepare_features(market_data)
            tprint(f"✅ [UNSUPERVISED_TREE_NAS] Prepared {X.shape[1]} features from {len(feature_names)} feature types", color="green")
            
            # Search for architectures
            tprint("🔍 [UNSUPERVISED_TREE_NAS] Searching for optimal architectures", color="yellow")
            best_candidate = self._search_architectures(X, feature_names, timestamps)
            
            search_time = time.time() - start_time
            tprint(f"🎉 [UNSUPERVISED_TREE_NAS] Unsupervised NAS completed in {search_time:.2f}s", color="green", bold=True)
            tprint(f"📊 [UNSUPERVISED_TREE_NAS] Best architecture: {best_candidate.clustering_algorithm}, score: {best_candidate.overall_score:.4f}", color="cyan")
            tprint(f"🔍 [UNSUPERVISED_TREE_NAS] Detected {best_candidate.n_regimes} regimes", color="cyan")
            self.logger.info(f"✅ Unsupervised NAS completed in {search_time:.2f}s")
            self.logger.info(f"📊 Best architecture: {best_candidate.clustering_algorithm}, score: {best_candidate.overall_score:.4f}")
            self.logger.info(f"🔍 Detected {best_candidate.n_regimes} regimes")
            
            return best_candidate
            
        except Exception as e:
            self.logger.error(f"Unsupervised NAS Search failed: {e}")
            raise
    
    def _prepare_features(self, market_data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Prepare features for unsupervised learning."""
        try:
            features = []
            feature_names = []
            
            # Price-based features
            if 'close' in market_data.columns:
                # Returns
                returns = market_data['close'].pct_change().fillna(0)
                features.append(returns.values)
                feature_names.append('returns')
                
                # Log returns
                log_returns = np.log(market_data['close'] / market_data['close'].shift(1)).fillna(0)
                features.append(log_returns.values)
                feature_names.append('log_returns')
                
                # Price momentum
                momentum_5 = market_data['close'].pct_change(5).fillna(0)
                momentum_10 = market_data['close'].pct_change(10).fillna(0)
                momentum_20 = market_data['close'].pct_change(20).fillna(0)
                features.extend([momentum_5.values, momentum_10.values, momentum_20.values])
                feature_names.extend(['momentum_5', 'momentum_10', 'momentum_20'])
                
                # Moving averages
                ma_5 = market_data['close'].rolling(5).mean().fillna(market_data['close'])
                ma_10 = market_data['close'].rolling(10).mean().fillna(market_data['close'])
                ma_20 = market_data['close'].rolling(20).mean().fillna(market_data['close'])
                features.extend([ma_5.values, ma_10.values, ma_20.values])
                feature_names.extend(['ma_5', 'ma_10', 'ma_20'])
                
                # Price ratios
                price_ratios = (market_data['close'] / ma_20).fillna(1)
                features.append(price_ratios.values)
                feature_names.append('price_ratio_ma20')
            
            # Volatility features
            if 'high' in market_data.columns and 'low' in market_data.columns:
                # True range
                high_low = market_data['high'] - market_data['low']
                high_close = np.abs(market_data['high'] - market_data['close'].shift(1))
                low_close = np.abs(market_data['low'] - market_data['close'].shift(1))
                true_range = np.maximum(high_low, np.maximum(high_close, low_close))
                features.append(true_range.values)
                feature_names.append('true_range')
                
                # Volatility (rolling standard deviation)
                volatility_5 = returns.rolling(5).std().fillna(0)
                volatility_10 = returns.rolling(10).std().fillna(0)
                volatility_20 = returns.rolling(20).std().fillna(0)
                features.extend([volatility_5.values, volatility_10.values, volatility_20.values])
                feature_names.extend(['volatility_5', 'volatility_10', 'volatility_20'])
            
            # Volume features
            if 'volume' in market_data.columns:
                # Volume momentum
                volume_momentum = market_data['volume'].pct_change().fillna(0)
                features.append(volume_momentum.values)
                feature_names.append('volume_momentum')
                
                # Volume moving averages
                volume_ma_5 = market_data['volume'].rolling(5).mean().fillna(market_data['volume'])
                volume_ma_10 = market_data['volume'].rolling(10).mean().fillna(market_data['volume'])
                features.extend([volume_ma_5.values, volume_ma_10.values])
                feature_names.extend(['volume_ma_5', 'volume_ma_10'])
                
                # Volume ratio
                volume_ratio = (market_data['volume'] / volume_ma_10).fillna(1)
                features.append(volume_ratio.values)
                feature_names.append('volume_ratio')
            
            # Technical indicators
            if 'close' in market_data.columns:
                # RSI
                delta = market_data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                features.append(rsi.fillna(50).values)
                feature_names.append('rsi')
                
                # MACD
                ema_12 = market_data['close'].ewm(span=12).mean()
                ema_26 = market_data['close'].ewm(span=26).mean()
                macd = ema_12 - ema_26
                features.append(macd.values)
                feature_names.append('macd')
            
            # Combine all features
            X = np.column_stack(features)
            
            # Handle NaN values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            self.logger.info(f"Prepared {X.shape[1]} features for unsupervised learning")
            return X, feature_names
            
        except Exception as e:
            self.logger.error(f"Feature preparation failed: {e}")
            raise
    
    def _search_architectures(self, X: np.ndarray, feature_names: List[str], 
                            timestamps: Optional[np.ndarray] = None) -> UnsupervisedArchitectureCandidate:
        """Search for optimal unsupervised architectures."""
        best_candidate = None
        best_score = -np.inf
        
        for trial in range(self.config.n_trials):
            try:
                # Sample architecture
                candidate = self._sample_architecture(trial)
                
                # Train and evaluate
                performance = self._train_and_evaluate_architecture(
                    candidate, X, feature_names, timestamps
                )
                
                # Update best candidate
                if performance['overall_score'] > best_score:
                    best_score = performance['overall_score']
                    best_candidate = candidate
                    best_candidate.clustering_quality = performance['clustering_quality']
                    best_candidate.regime_detection_accuracy = performance['regime_detection_accuracy']
                    best_candidate.regime_qualification_score = performance['regime_qualification_score']
                    best_candidate.overall_score = performance['overall_score']
                    best_candidate.regimes = performance['regimes']
                    best_candidate.n_regimes = len(performance['regimes'])
                    best_candidate.feature_importance = performance['feature_importance']
                
                self.logger.debug(f"Trial {trial}: Score {performance['overall_score']:.4f}")
                
            except Exception as e:
                self.logger.warning(f"Trial {trial} failed: {e}")
                continue
        
        if best_candidate is None:
            raise RuntimeError("No successful architecture found")
        
        return best_candidate
    
    def _sample_architecture(self, trial_number: int) -> UnsupervisedArchitectureCandidate:
        """Sample a random unsupervised architecture."""
        try:
            # Sample clustering algorithm
            clustering_algorithm = np.random.choice(self.config.clustering_algorithms)
            
            # Sample clustering parameters
            clustering_params = self._sample_clustering_params(clustering_algorithm)
            
            # Sample feature engineering
            feature_engineering = {
                'methods': np.random.choice(self.config.feature_engineering_methods, 
                                          size=np.random.randint(2, 5), replace=False).tolist(),
                'normalization': np.random.choice(['standard', 'robust', 'minmax', 'none'])
            }
            
            # Sample dimensionality reduction
            dimensionality_reduction = None
            if np.random.random() < 0.5:  # 50% chance of using dimensionality reduction
                dimensionality_reduction = {
                    'method': np.random.choice(['pca', 'ica', 'tsne', 'umap']),
                    'n_components': np.random.randint(5, 20)
                }
            
            # Sample tree model
            tree_model = np.random.choice(self.config.tree_models)
            tree_params = self._sample_tree_params(tree_model)
            
            return UnsupervisedArchitectureCandidate(
                clustering_algorithm=clustering_algorithm,
                clustering_params=clustering_params,
                feature_engineering=feature_engineering,
                dimensionality_reduction=dimensionality_reduction,
                tree_model=tree_model,
                tree_params=tree_params,
                trial_number=trial_number
            )
            
        except Exception as e:
            self.logger.error(f"Architecture sampling failed: {e}")
            raise
    
    def _sample_clustering_params(self, algorithm: str) -> Dict[str, Any]:
        """Sample parameters for clustering algorithm."""
        if algorithm == 'kmeans':
            return {
                'n_clusters': np.random.randint(self.config.n_regimes_range[0], 
                                              self.config.n_regimes_range[1] + 1),
                'init': np.random.choice(['k-means++', 'random']),
                'n_init': np.random.randint(10, 20),
                'max_iter': np.random.randint(100, 300)
            }
        elif algorithm == 'dbscan':
            return {
                'eps': np.random.uniform(0.1, 2.0),
                'min_samples': np.random.randint(3, 10)
            }
        elif algorithm == 'gaussian_mixture':
            return {
                'n_components': np.random.randint(self.config.n_regimes_range[0], 
                                                self.config.n_regimes_range[1] + 1),
                'covariance_type': np.random.choice(['full', 'tied', 'diag', 'spherical']),
                'init_params': np.random.choice(['kmeans', 'random'])
            }
        elif algorithm == 'agglomerative':
            return {
                'n_clusters': np.random.randint(self.config.n_regimes_range[0], 
                                              self.config.n_regimes_range[1] + 1),
                'linkage': np.random.choice(['ward', 'complete', 'average', 'single'])
            }
        elif algorithm == 'isolation_forest':
            return {
                'contamination': np.random.uniform(0.01, 0.3),
                'n_estimators': np.random.randint(50, 200)
            }
        else:
            return {}
    
    def _sample_tree_params(self, model: str) -> Dict[str, Any]:
        """Sample parameters for tree model."""
        base_params = self.config.tree_params.copy()
        
        if model == 'random_forest':
            return {
                **base_params,
                'n_estimators': np.random.randint(50, 200),
                'max_depth': np.random.randint(5, 20),
                'min_samples_split': np.random.randint(2, 10),
                'min_samples_leaf': np.random.randint(1, 5)
            }
        elif model == 'xgboost':
            return {
                'n_estimators': np.random.randint(50, 200),
                'max_depth': np.random.randint(3, 10),
                'learning_rate': np.random.uniform(0.01, 0.3),
                'subsample': np.random.uniform(0.8, 1.0),
                'colsample_bytree': np.random.uniform(0.8, 1.0)
            }
        elif model == 'lightgbm':
            return {
                'n_estimators': np.random.randint(50, 200),
                'max_depth': np.random.randint(3, 10),
                'learning_rate': np.random.uniform(0.01, 0.3),
                'subsample': np.random.uniform(0.8, 1.0),
                'colsample_bytree': np.random.uniform(0.8, 1.0),
                'num_leaves': np.random.randint(31, 127)
            }
        else:
            return base_params
    
    def _train_and_evaluate_architecture(self, candidate: UnsupervisedArchitectureCandidate,
                                       X: np.ndarray, feature_names: List[str],
                                       timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train and evaluate an unsupervised architecture."""
        try:
            start_time = time.time()
            
            # Apply feature engineering
            X_processed = self._apply_feature_engineering(X, candidate.feature_engineering)
            
            # Apply dimensionality reduction if specified
            if candidate.dimensionality_reduction:
                X_processed = self._apply_dimensionality_reduction(
                    X_processed, candidate.dimensionality_reduction
                )
            
            # Perform clustering
            labels = self._perform_clustering(X_processed, candidate)
            
            # Detect and qualify regimes
            regimes = self._detect_and_qualify_regimes(
                X_processed, labels, feature_names, timestamps
            )
            
            # Calculate performance metrics
            clustering_quality = self._calculate_clustering_quality(X_processed, labels)
            regime_detection_accuracy = self._calculate_regime_detection_accuracy(regimes)
            regime_qualification_score = self._calculate_regime_qualification_score(regimes)
            
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(X_processed, labels, feature_names)
            
            # Calculate overall score
            overall_score = (
                0.4 * clustering_quality +
                0.3 * regime_detection_accuracy +
                0.3 * regime_qualification_score
            )
            
            training_time = time.time() - start_time
            candidate.training_time = training_time
            
            return {
                'clustering_quality': clustering_quality,
                'regime_detection_accuracy': regime_detection_accuracy,
                'regime_qualification_score': regime_qualification_score,
                'overall_score': overall_score,
                'regimes': regimes,
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            self.logger.warning(f"Architecture training failed: {e}")
            return {
                'clustering_quality': 0.0,
                'regime_detection_accuracy': 0.0,
                'regime_qualification_score': 0.0,
                'overall_score': 0.0,
                'regimes': [],
                'feature_importance': {}
            }
    
    def _apply_feature_engineering(self, X: np.ndarray, feature_engineering: Dict[str, Any]) -> np.ndarray:
        """Apply feature engineering to the data."""
        try:
            # Apply normalization
            if feature_engineering['normalization'] == 'standard':
                scaler = StandardScaler()
                X_processed = scaler.fit_transform(X)
            elif feature_engineering['normalization'] == 'robust':
                scaler = RobustScaler()
                X_processed = scaler.fit_transform(X)
            else:
                X_processed = X.copy()
            
            return X_processed
            
        except Exception as e:
            self.logger.warning(f"Feature engineering failed: {e}")
            return X
    
    def _apply_dimensionality_reduction(self, X: np.ndarray, reduction_config: Dict[str, Any]) -> np.ndarray:
        """Apply dimensionality reduction to the data."""
        try:
            method = reduction_config['method']
            n_components = reduction_config['n_components']
            
            if method == 'pca':
                reducer = PCA(n_components=n_components)
            elif method == 'ica':
                reducer = FastICA(n_components=n_components)
            elif method == 'tsne':
                reducer = TSNE(n_components=n_components)
            elif method == 'umap':
                reducer = UMAP(n_components=n_components)
            else:
                return X
            
            return reducer.fit_transform(X)
            
        except Exception as e:
            self.logger.warning(f"Dimensionality reduction failed: {e}")
            return X
    
    def _perform_clustering(self, X: np.ndarray, candidate: UnsupervisedArchitectureCandidate) -> np.ndarray:
        """Perform clustering using the specified algorithm."""
        try:
            algorithm = candidate.clustering_algorithm
            params = candidate.clustering_params
            
            if algorithm == 'kmeans':
                clusterer = KMeans(**params, random_state=42, n_jobs=self.config.n_jobs)
            elif algorithm == 'dbscan':
                clusterer = DBSCAN(**params, n_jobs=self.config.n_jobs)
            elif algorithm == 'gaussian_mixture':
                clusterer = GaussianMixture(**params, random_state=42)
            elif algorithm == 'agglomerative':
                clusterer = AgglomerativeClustering(**params)
            elif algorithm == 'isolation_forest':
                clusterer = IsolationForest(**params, random_state=42)
            else:
                raise ValueError(f"Unknown clustering algorithm: {algorithm}")
            
            labels = clusterer.fit_predict(X)
            
            # Handle noise labels from DBSCAN
            if algorithm == 'dbscan':
                labels[labels == -1] = np.max(labels) + 1
            
            return labels
            
        except Exception as e:
            self.logger.error(f"Clustering failed: {e}")
            raise
    
    def _detect_and_qualify_regimes(self, X: np.ndarray, labels: np.ndarray,
                                  feature_names: List[str], timestamps: Optional[np.ndarray] = None) -> List[RegimeCandidate]:
        """Detect and qualify regimes from clustering results."""
        try:
            regimes = []
            unique_labels = np.unique(labels)
            
            for regime_id in unique_labels:
                if regime_id == -1:  # Skip noise labels
                    continue
                
                # Get regime samples
                regime_mask = labels == regime_id
                regime_samples = X[regime_mask]
                regime_indices = np.where(regime_mask)[0]
                
                if len(regime_samples) < self.config.min_regime_duration:
                    continue
                
                # Calculate regime center and boundary
                regime_center = np.mean(regime_samples, axis=0)
                regime_boundary = np.std(regime_samples, axis=0)
                regime_size = len(regime_samples)
                
                # Calculate regime quality metrics
                silhouette_score = self._calculate_regime_silhouette_score(X, labels, regime_id)
                calinski_harabasz_score = self._calculate_regime_calinski_harabasz_score(X, labels, regime_id)
                davies_bouldin_score = self._calculate_regime_davies_bouldin_score(X, labels, regime_id)
                
                # Calculate regime persistence
                regime_persistence = self._calculate_regime_persistence(regime_indices)
                
                # Calculate regime separation
                regime_separation = self._calculate_regime_separation(X, labels, regime_id)
                
                # Calculate regime consistency
                regime_consistency = self._calculate_regime_consistency(regime_samples)
                
                # Calculate overall quality
                overall_quality = (
                    0.3 * silhouette_score +
                    0.2 * regime_persistence +
                    0.2 * regime_separation +
                    0.3 * regime_consistency
                )
                
                # Determine regime type
                regime_type = self._determine_regime_type(regime_samples, feature_names)
                
                # Calculate regime confidence
                regime_confidence = min(overall_quality, 1.0)
                
                # Calculate transition probabilities
                transition_probability, transition_targets = self._calculate_transition_probabilities(
                    labels, regime_id
                )
                
                # Calculate feature importance for this regime
                feature_importance = self._calculate_regime_feature_importance(
                    regime_samples, feature_names
                )
                
                # Get key features
                key_features = self._get_key_features(feature_importance, top_k=5)
                
                # Create regime candidate
                regime = RegimeCandidate(
                    regime_id=regime_id,
                    regime_type=regime_type,
                    regime_confidence=regime_confidence,
                    start_time=timestamps[regime_indices[0]] if timestamps is not None else datetime.now(),
                    end_time=timestamps[regime_indices[-1]] if timestamps is not None else datetime.now(),
                    duration=len(regime_indices),
                    regime_center=regime_center,
                    regime_boundary=regime_boundary,
                    regime_size=regime_size,
                    silhouette_score=silhouette_score,
                    calinski_harabasz_score=calinski_harabasz_score,
                    davies_bouldin_score=davies_bouldin_score,
                    regime_persistence=regime_persistence,
                    regime_separation=regime_separation,
                    regime_consistency=regime_consistency,
                    overall_quality=overall_quality,
                    feature_importance=feature_importance,
                    key_features=key_features,
                    transition_probability=transition_probability,
                    transition_targets=transition_targets,
                    sample_indices=regime_indices.tolist()
                )
                
                regimes.append(regime)
            
            return regimes
            
        except Exception as e:
            self.logger.error(f"Regime detection failed: {e}")
            return []
    
    def _calculate_clustering_quality(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate overall clustering quality."""
        try:
            if len(np.unique(labels)) < 2:
                return 0.0
            
            # Calculate multiple quality metrics
            silhouette = silhouette_score(X, labels)
            calinski_harabasz = calinski_harabasz_score(X, labels)
            davies_bouldin = davies_bouldin_score(X, labels)
            
            # Normalize and combine metrics
            normalized_silhouette = max(0, silhouette)
            normalized_calinski = min(1.0, calinski_harabasz / 1000.0)
            normalized_davies = max(0, 1.0 - davies_bouldin / 5.0)
            
            quality = (normalized_silhouette + normalized_calinski + normalized_davies) / 3.0
            return float(quality)
            
        except Exception as e:
            self.logger.warning(f"Clustering quality calculation failed: {e}")
            return 0.0
    
    def _calculate_regime_detection_accuracy(self, regimes: List[RegimeCandidate]) -> float:
        """Calculate regime detection accuracy."""
        try:
            if not regimes:
                return 0.0
            
            # Calculate accuracy based on regime quality
            total_quality = sum(regime.overall_quality for regime in regimes)
            avg_quality = total_quality / len(regimes)
            
            return float(avg_quality)
            
        except Exception as e:
            self.logger.warning(f"Regime detection accuracy calculation failed: {e}")
            return 0.0
    
    def _calculate_regime_qualification_score(self, regimes: List[RegimeCandidate]) -> float:
        """Calculate regime qualification score."""
        try:
            if not regimes:
                return 0.0
            
            # Calculate qualification score based on multiple criteria
            scores = []
            
            for regime in regimes:
                # Check if regime meets quality thresholds
                meets_silhouette = regime.silhouette_score >= self.config.quality_thresholds['min_silhouette_score']
                meets_persistence = regime.regime_persistence >= self.config.quality_thresholds['min_regime_persistence']
                meets_separation = regime.regime_separation >= self.config.quality_thresholds['min_regime_separation']
                meets_consistency = regime.regime_consistency >= self.config.quality_thresholds['min_regime_consistency']
                
                # Calculate qualification score
                qualification_score = sum([meets_silhouette, meets_persistence, meets_separation, meets_consistency]) / 4.0
                scores.append(qualification_score)
            
            return float(np.mean(scores))
            
        except Exception as e:
            self.logger.warning(f"Regime qualification score calculation failed: {e}")
            return 0.0
    
    def _calculate_feature_importance(self, X: np.ndarray, labels: np.ndarray, 
                                    feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature importance for regime detection."""
        try:
            if len(np.unique(labels)) < 2:
                return {name: 0.0 for name in feature_names}
            
            # Use Random Forest to calculate feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, labels)
            
            feature_importance = {}
            for i, name in enumerate(feature_names):
                if i < len(rf.feature_importances_):
                    feature_importance[name] = float(rf.feature_importances_[i])
                else:
                    feature_importance[name] = 0.0
            
            return feature_importance
            
        except Exception as e:
            self.logger.warning(f"Feature importance calculation failed: {e}")
            return {name: 0.0 for name in feature_names}
    
    # Additional helper methods for regime analysis
    def _calculate_regime_silhouette_score(self, X: np.ndarray, labels: np.ndarray, regime_id: int) -> float:
        """Calculate silhouette score for a specific regime."""
        try:
            regime_mask = labels == regime_id
            if np.sum(regime_mask) < 2:
                return 0.0
            
            regime_samples = X[regime_mask]
            other_samples = X[~regime_mask]
            
            if len(other_samples) == 0:
                return 0.0
            
            # Calculate intra-cluster distance
            intra_distances = []
            for i in range(len(regime_samples)):
                for j in range(i + 1, len(regime_samples)):
                    intra_distances.append(np.linalg.norm(regime_samples[i] - regime_samples[j]))
            
            avg_intra_distance = np.mean(intra_distances) if intra_distances else 0.0
            
            # Calculate inter-cluster distance
            inter_distances = []
            for i in range(len(regime_samples)):
                for j in range(len(other_samples)):
                    inter_distances.append(np.linalg.norm(regime_samples[i] - other_samples[j]))
            
            avg_inter_distance = np.mean(inter_distances) if inter_distances else 0.0
            
            # Calculate silhouette score
            if avg_intra_distance == 0 and avg_inter_distance == 0:
                return 0.0
            
            silhouette = (avg_inter_distance - avg_intra_distance) / max(avg_inter_distance, avg_intra_distance)
            return float(max(0, min(1, silhouette)))
            
        except Exception as e:
            self.logger.warning(f"Regime silhouette score calculation failed: {e}")
            return 0.0
    
    def _calculate_regime_persistence(self, regime_indices: np.ndarray) -> float:
        """Calculate regime persistence (how long the regime lasts)."""
        try:
            if len(regime_indices) < 2:
                return 0.0
            
            # Calculate consecutive periods
            consecutive_periods = []
            current_period = 1
            
            for i in range(1, len(regime_indices)):
                if regime_indices[i] == regime_indices[i-1] + 1:
                    current_period += 1
                else:
                    consecutive_periods.append(current_period)
                    current_period = 1
            
            consecutive_periods.append(current_period)
            
            # Calculate persistence as ratio of longest consecutive period to total length
            max_consecutive = max(consecutive_periods)
            persistence = max_consecutive / len(regime_indices)
            
            return float(persistence)
            
        except Exception as e:
            self.logger.warning(f"Regime persistence calculation failed: {e}")
            return 0.0
    
    def _calculate_regime_separation(self, X: np.ndarray, labels: np.ndarray, regime_id: int) -> float:
        """Calculate regime separation (how distinct the regime is)."""
        try:
            regime_mask = labels == regime_id
            regime_samples = X[regime_mask]
            other_samples = X[~regime_mask]
            
            if len(regime_samples) == 0 or len(other_samples) == 0:
                return 0.0
            
            # Calculate regime center
            regime_center = np.mean(regime_samples, axis=0)
            
            # Calculate distance to other regimes
            distances_to_others = []
            for other_regime_id in np.unique(labels):
                if other_regime_id != regime_id:
                    other_mask = labels == other_regime_id
                    other_samples = X[other_mask]
                    if len(other_samples) > 0:
                        other_center = np.mean(other_samples, axis=0)
                        distance = np.linalg.norm(regime_center - other_center)
                        distances_to_others.append(distance)
            
            if not distances_to_others:
                return 0.0
            
            # Calculate separation as minimum distance to other regimes
            min_distance = min(distances_to_others)
            max_possible_distance = np.sqrt(X.shape[1])  # Maximum possible distance in feature space
            
            separation = min(1.0, min_distance / max_possible_distance)
            return float(separation)
            
        except Exception as e:
            self.logger.warning(f"Regime separation calculation failed: {e}")
            return 0.0
    
    def _calculate_regime_consistency(self, regime_samples: np.ndarray) -> float:
        """Calculate regime consistency (how consistent the regime is internally)."""
        try:
            if len(regime_samples) < 2:
                return 0.0
            
            # Calculate variance within regime
            regime_variance = np.var(regime_samples, axis=0)
            avg_variance = np.mean(regime_variance)
            
            # Calculate consistency as inverse of variance (normalized)
            max_possible_variance = np.var(regime_samples)  # Overall variance
            consistency = 1.0 - (avg_variance / max_possible_variance) if max_possible_variance > 0 else 1.0
            
            return float(max(0, min(1, consistency)))
            
        except Exception as e:
            self.logger.warning(f"Regime consistency calculation failed: {e}")
            return 0.0
    
    def _determine_regime_type(self, regime_samples: np.ndarray, feature_names: List[str]) -> str:
        """Determine the type of regime based on sample characteristics."""
        try:
            # Simple heuristic based on feature values
            if 'returns' in feature_names:
                returns_idx = feature_names.index('returns')
                avg_returns = np.mean(regime_samples[:, returns_idx])
                
                if avg_returns > 0.01:
                    return 'bull'
                elif avg_returns < -0.01:
                    return 'bear'
                else:
                    return 'sideways'
            elif 'volatility' in feature_names:
                volatility_idx = feature_names.index('volatility')
                avg_volatility = np.mean(regime_samples[:, volatility_idx])
                
                if avg_volatility > 0.02:
                    return 'volatile'
                else:
                    return 'stable'
            else:
                return 'unknown'
                
        except Exception as e:
            self.logger.warning(f"Regime type determination failed: {e}")
            return 'unknown'
    
    def _calculate_transition_probabilities(self, labels: np.ndarray, regime_id: int) -> Tuple[float, List[int]]:
        """Calculate transition probabilities from a regime."""
        try:
            regime_positions = np.where(labels == regime_id)[0]
            if len(regime_positions) == 0:
                return 0.0, []
            
            # Find transitions
            transitions = []
            for pos in regime_positions:
                if pos < len(labels) - 1 and labels[pos + 1] != regime_id:
                    transitions.append(labels[pos + 1])
            
            if not transitions:
                return 0.0, []
            
            # Calculate transition probability
            transition_probability = len(transitions) / len(regime_positions)
            
            # Get unique transition targets
            transition_targets = list(set(transitions))
            
            return float(transition_probability), transition_targets
            
        except Exception as e:
            self.logger.warning(f"Transition probability calculation failed: {e}")
            return 0.0, []
    
    def _calculate_regime_feature_importance(self, regime_samples: np.ndarray, 
                                          feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature importance for a specific regime."""
        try:
            if len(regime_samples) == 0:
                return {name: 0.0 for name in feature_names}
            
            # Calculate variance of each feature within the regime
            feature_variances = np.var(regime_samples, axis=0)
            
            # Normalize variances to get importance scores
            total_variance = np.sum(feature_variances)
            if total_variance == 0:
                return {name: 1.0 / len(feature_names) for name in feature_names}
            
            feature_importance = {}
            for i, name in enumerate(feature_names):
                if i < len(feature_variances):
                    feature_importance[name] = float(feature_variances[i] / total_variance)
                else:
                    feature_importance[name] = 0.0
            
            return feature_importance
            
        except Exception as e:
            self.logger.warning(f"Regime feature importance calculation failed: {e}")
            return {name: 0.0 for name in feature_names}
    
    def _get_key_features(self, feature_importance: Dict[str, float], top_k: int = 5) -> List[str]:
        """Get top-k most important features."""
        try:
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            return [name for name, _ in sorted_features[:top_k]]
            
        except Exception as e:
            self.logger.warning(f"Key features extraction failed: {e}")
            return []
    
    def get_search_summary(self) -> Dict[str, Any]:
        """Get summary of unsupervised search results."""
        if not self.candidates:
            return {'message': 'No search results available'}
        
        try:
            return {
                'total_candidates': len(self.candidates),
                'best_algorithm': self.best_candidate.clustering_algorithm if self.best_candidate else None,
                'best_score': self.best_candidate.overall_score if self.best_candidate else 0.0,
                'detected_regimes': self.best_candidate.n_regimes if self.best_candidate else 0,
                'regime_types': [regime.regime_type for regime in self.best_candidate.regimes] if self.best_candidate else [],
                'avg_regime_quality': np.mean([regime.overall_quality for regime in self.best_candidate.regimes]) if self.best_candidate else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Search summary generation failed: {e}")
            return {'error': str(e)}

# Convenience function
def search_unsupervised_regimes(market_data: pd.DataFrame,
                              timestamps: Optional[np.ndarray] = None,
                              config: Optional[UnsupervisedTreeNASConfig] = None) -> UnsupervisedArchitectureCandidate:
    """
    Convenience function to perform unsupervised regime detection.
    
    Args:
        market_data: Market data (OHLCV)
        timestamps: Timestamps for the data (optional)
        config: Unsupervised NAS configuration
        
    Returns:
        Best unsupervised architecture candidate
    """
    if config is None:
        config = UnsupervisedTreeNASConfig()
    
    unsupervised_nas = UnsupervisedTreeNAS(config)
    return unsupervised_nas.search(market_data, timestamps)

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
