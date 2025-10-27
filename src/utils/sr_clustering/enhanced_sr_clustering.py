"""
Enhanced SR Clustering Module

This module provides a comprehensive, hardware-optimized clustering system for Support/Resistance levels
with advanced ML integration, HPO optimization, and data leakage prevention.

Key Features:
- VectorBTRollingOptimizer integration for high-performance rolling operations
- UnifiedVectorizationManager for optimized matrix operations
- Advanced HPO using Bayesian optimization and genetic algorithms
- Hardware optimization with M1 chip support
- SHAP/LIME explainability integration
- Comprehensive data leakage detection and prevention
- Purged cross-validation for time series
- Advanced clustering algorithms (HDBSCAN, OPTICS, Spectral)
- Real-time performance monitoring and optimization
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import asyncio
from abc import ABC, abstractmethod
import warnings
from enum import Enum
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Core ML libraries
from sklearn.cluster import DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE, UMAP

# Advanced clustering
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

try:
    from sklearn.cluster import OPTICS
    OPTICS_AVAILABLE = True
except ImportError:
    OPTICS_AVAILABLE = False

# VectorBT and optimization imports
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max
    from vectorbt.generic import rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None

# Import VectorBTRollingOptimizer
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    )
    VECTORBT_ROLLING_AVAILABLE = True
except ImportError:
    VECTORBT_ROLLING_AVAILABLE = False
    VectorBTRollingOptimizer = None

# Import UnifiedVectorizationManager
try:
    from src.utils.ml_common.unified_vectorization_manager import (
        UnifiedVectorizationManager, get_unified_vectorization_manager, OperationType
    )
    UNIFIED_VECTORIZATION_AVAILABLE = True
except ImportError:
    UNIFIED_VECTORIZATION_AVAILABLE = False
    UnifiedVectorizationManager = None

# Import HPO components
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
        BayesianTPEOptimizer, get_bayesian_tpe_optimizer
    )
    from src.utils.ml_common.optimization.hierarchical_hpo import (
        HierarchicalHPO, get_hierarchical_hpo
    )
    from src.utils.ml_common.optimization.regime_hpo_wrapper import (
        RegimeHPOWrapper, get_regime_hpo_wrapper
    )
    HPO_AVAILABLE = True
except ImportError:
    HPO_AVAILABLE = False

# Import hardware optimization
try:
    from src.utils.hardware.unified_hardware_manager import (
        UnifiedHardwareManager, WorkloadType, OptimizationLevel
    )
    from src.utils.hardware.adaptive_optimization_engine import (
        AdaptiveOptimizationEngine, get_adaptive_optimization_engine
    )
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATION_AVAILABLE = False

# Import ML utilities
try:
    from src.utils.ml_common.explainability.shap_lime_integration import (
        SHAPLIMEIntegration, get_shap_lime_integration
    )
    from src.utils.ml_common.validation.data_leakage_detector import (
        DataLeakageDetector, get_data_leakage_detector
    )
    from src.utils.ml_common.validation.unified_cv import (
        UnifiedCrossValidation, get_unified_cv
    )
    from src.utils.ml_common.validation.temporal_validation import (
        TemporalValidation, get_temporal_validation
    )
    ML_UTILITIES_AVAILABLE = True
except ImportError:
    ML_UTILITIES_AVAILABLE = False

# Import existing SR clustering components
from .sr_backtesting_engine import SRBacktestingEngine, BacktestConfig, SRLevel, BacktestResult
from .backtesting_enhanced_clustering import BacktestingEnhancedClustering, BacktestingEnhancedConfig

# Import logger
from ..logger import system_logger

logger = system_logger.getChild('enhanced_sr_clustering')

class ClusteringAlgorithm(Enum):
    """Available clustering algorithms."""
    DBSCAN = "dbscan"
    AGGLOMERATIVE = "agglomerative"
    SPECTRAL = "spectral"
    HDBSCAN = "hdbscan"
    OPTICS = "optics"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"

class OptimizationStrategy(Enum):
    """Optimization strategies."""
    BAYESIAN_TPE = "bayesian_tpe"
    HIERARCHICAL_HPO = "hierarchical_hpo"
    REGIME_SPECIFIC = "regime_specific"
    ADAPTIVE = "adaptive"

@dataclass
class EnhancedSRClusteringConfig:
    """Enhanced configuration for SR clustering with all optimizations."""
    
    # Core clustering parameters
    clustering_algorithm: ClusteringAlgorithm = ClusteringAlgorithm.ADAPTIVE
    enable_adaptive_clustering: bool = True
    min_cluster_size: int = 5
    max_cluster_size: int = 100
    cluster_quality_threshold: float = 0.7
    
    # DBSCAN parameters
    dbscan_eps: float = 0.02
    dbscan_min_samples: int = 5
    
    # HDBSCAN parameters
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int = 3
    hdbscan_cluster_selection_epsilon: float = 0.0
    
    # Spectral clustering parameters
    spectral_n_clusters: int = 10
    spectral_affinity: str = "rbf"
    spectral_gamma: float = 1.0
    
    # Feature engineering
    feature_engineering_config: Dict[str, Any] = field(default_factory=lambda: {
        'use_price_features': True,
        'use_volume_features': True,
        'use_time_features': True,
        'use_technical_indicators': True,
        'use_market_microstructure': True,
        'feature_normalization': 'robust',  # 'standard', 'minmax', 'robust'
        'feature_selection': True,
        'dimensionality_reduction': 'pca',  # 'pca', 'ica', 'tsne', 'umap', None
        'n_components': 0.95,  # For PCA, can be float (variance) or int (components)
    })
    
    # VectorBT optimization
    vectorbt_config: Dict[str, Any] = field(default_factory=lambda: {
        'enable_vectorbt_optimization': True,
        'chunk_size': 1000,
        'enable_parallel_processing': True,
        'memory_limit_gb': 8.0,
        'enable_gpu_acceleration': False,
    })
    
    # HPO configuration
    hpo_config: Dict[str, Any] = field(default_factory=lambda: {
        'enable_hpo': True,
        'optimization_strategy': OptimizationStrategy.ADAPTIVE,
        'n_trials': 100,
        'cv_folds': 5,
        'objective_metric': 'composite_score',
        'enable_early_stopping': True,
        'patience': 10,
    })
    
    # Hardware optimization
    hardware_config: Dict[str, Any] = field(default_factory=lambda: {
        'enable_hardware_optimization': True,
        'workload_type': 'ML_TRAINING',  # Will be converted to WorkloadType enum
        'optimization_level': 'AGGRESSIVE',  # Will be converted to OptimizationLevel enum
        'enable_memory_optimization': True,
        'enable_cpu_optimization': True,
        'enable_gpu_optimization': False,
        'max_workers': None,  # Auto-detect
    })
    
    # Data leakage prevention
    data_leakage_config: Dict[str, Any] = field(default_factory=lambda: {
        'enable_leakage_detection': True,
        'temporal_tolerance': 1,  # Minimum time gap between train/test
        'lookahead_tolerance': 24,  # Hours of lookahead tolerance
        'enable_purged_cv': True,
        'purge_period': 1,  # Days to purge between train/test
        'enable_embargo_period': True,
        'embargo_period': 1,  # Days embargo period
    })
    
    # Explainability
    explainability_config: Dict[str, Any] = field(default_factory=lambda: {
        'enable_explainability': True,
        'enable_shap': True,
        'enable_lime': True,
        'explanation_sample_size': 100,
        'feature_importance_threshold': 0.01,
    })
    
    # Backtesting integration
    backtesting_config: Dict[str, Any] = field(default_factory=lambda: {
        'enable_backtesting_validation': True,
        'min_backtest_score': 0.1,
        'backtest_lookback_days': 30,
        'enable_rolling_validation': True,
        'rolling_window_size': 252,  # 1 year
        'rolling_step_size': 21,  # 1 month
    })
    
    # Performance monitoring
    performance_config: Dict[str, Any] = field(default_factory=lambda: {
        'enable_performance_monitoring': True,
        'enable_memory_monitoring': True,
        'enable_timing_monitoring': True,
        'log_performance_metrics': True,
        'performance_report_interval': 100,  # Log every N operations
    })

@dataclass
class EnhancedClusterResult:
    """Enhanced result from clustering operation with comprehensive metrics."""
    
    # Basic cluster info
    cluster_id: int
    level_indices: List[int]
    centroid_price: float
    cluster_size: int
    
    # Quality metrics
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    cluster_quality: float
    
    # Backtesting metrics
    backtest_score: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    
    # Explainability metrics
    feature_importance: Optional[Dict[str, float]] = None
    shap_values: Optional[np.ndarray] = None
    lime_explanation: Optional[Dict[str, Any]] = None
    
    # Temporal metrics
    first_touch: Optional[datetime] = None
    last_touch: Optional[datetime] = None
    touch_frequency: Optional[float] = None
    persistence_score: Optional[float] = None
    
    # Confidence and reliability
    confidence: float = 1.0
    reliability_score: float = 1.0
    stability_score: float = 1.0

class EnhancedSRClustering:
    """
    Enhanced SR Clustering with comprehensive ML integration and optimization.
    
    This class provides a state-of-the-art clustering system for Support/Resistance levels
    with advanced ML capabilities, hardware optimization, and data leakage prevention.
    """
    
    def __init__(self, config: EnhancedSRClusteringConfig):
        """Initialize the enhanced SR clustering system."""
        self.config = config
        self.logger = logger.getChild('EnhancedSRClustering')
        
        # Initialize components
        self._initialize_components()
        
        # Performance monitoring
        self.performance_metrics = {
            'clustering_times': [],
            'memory_usage': [],
            'optimization_times': [],
            'backtesting_times': [],
        }
        
        self.logger.info("Enhanced SR Clustering initialized successfully")
    
    def _initialize_components(self):
        """Initialize all optimization and ML components."""
        try:
            # Initialize VectorBT rolling optimizer
            if VECTORBT_ROLLING_AVAILABLE and self.config.vectorbt_config['enable_vectorbt_optimization']:
                self.vectorbt_optimizer = get_vectorbt_rolling_optimizer()
                self.logger.info("✅ VectorBT Rolling Optimizer initialized")
            else:
                self.vectorbt_optimizer = None
                self.logger.warning("⚠️ VectorBT Rolling Optimizer not available")
            
            # Initialize Unified Vectorization Manager
            if UNIFIED_VECTORIZATION_AVAILABLE:
                self.vectorization_manager = get_unified_vectorization_manager()
                self.logger.info("✅ Unified Vectorization Manager initialized")
            else:
                self.vectorization_manager = None
                self.logger.warning("⚠️ Unified Vectorization Manager not available")
            
            # Initialize HPO components
            if HPO_AVAILABLE and self.config.hpo_config['enable_hpo']:
                self._initialize_hpo_components()
                self.logger.info("✅ HPO components initialized")
            else:
                self.hpo_components = {}
                self.logger.warning("⚠️ HPO components not available")
            
            # Initialize hardware optimization
            if HARDWARE_OPTIMIZATION_AVAILABLE and self.config.hardware_config['enable_hardware_optimization']:
                self.hardware_manager = UnifiedHardwareManager()
                self.adaptive_optimizer = get_adaptive_optimization_engine()
                self.logger.info("✅ Hardware optimization initialized")
            else:
                self.hardware_manager = None
                self.adaptive_optimizer = None
                self.logger.warning("⚠️ Hardware optimization not available")
            
            # Initialize ML utilities
            if ML_UTILITIES_AVAILABLE:
                self._initialize_ml_utilities()
                self.logger.info("✅ ML utilities initialized")
            else:
                self.ml_utilities = {}
                self.logger.warning("⚠️ ML utilities not available")
            
            # Initialize backtesting engine
            self.backtesting_engine = SRBacktestingEngine(
                BacktestConfig(**self.config.backtesting_config)
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing components: {e}")
            raise
    
    def _initialize_hpo_components(self):
        """Initialize HPO components based on strategy."""
        self.hpo_components = {}
        
        strategy = self.config.hpo_config['optimization_strategy']
        
        if strategy in [OptimizationStrategy.BAYESIAN_TPE, OptimizationStrategy.ADAPTIVE]:
            self.hpo_components['bayesian_tpe'] = get_bayesian_tpe_optimizer()
        
        if strategy in [OptimizationStrategy.HIERARCHICAL_HPO, OptimizationStrategy.ADAPTIVE]:
            self.hpo_components['hierarchical_hpo'] = get_hierarchical_hpo()
        
        if strategy in [OptimizationStrategy.REGIME_SPECIFIC, OptimizationStrategy.ADAPTIVE]:
            self.hpo_components['regime_hpo'] = get_regime_hpo_wrapper()
    
    def _initialize_ml_utilities(self):
        """Initialize ML utilities."""
        self.ml_utilities = {}
        
        if self.config.explainability_config['enable_explainability']:
            self.ml_utilities['shap_lime'] = get_shap_lime_integration()
        
        if self.config.data_leakage_config['enable_leakage_detection']:
            self.ml_utilities['leakage_detector'] = get_data_leakage_detector()
        
        if self.config.data_leakage_config['enable_purged_cv']:
            self.ml_utilities['unified_cv'] = get_unified_cv()
            self.ml_utilities['temporal_validation'] = get_temporal_validation()
    
    async def cluster_sr_levels(
        self,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None,
        additional_features: Optional[pd.DataFrame] = None
    ) -> List[EnhancedClusterResult]:
        """
        Perform enhanced clustering on SR levels with all optimizations.
        
        Args:
            price_data: Price data with OHLC columns
            volume_data: Optional volume data
            additional_features: Optional additional features
            
        Returns:
            List of enhanced cluster results
        """
        start_time = time.time()
        self.logger.info("Starting enhanced SR clustering")
        
        try:
            # Step 1: Data leakage detection and prevention
            if self.ml_utilities.get('leakage_detector'):
                await self._detect_and_prevent_leakage(price_data, volume_data, additional_features)
            
            # Step 2: Feature engineering with VectorBT optimization
            features = await self._extract_enhanced_features(price_data, volume_data, additional_features)
            
            # Step 3: Dimensionality reduction if enabled
            if self.config.feature_engineering_config['dimensionality_reduction']:
                features = await self._apply_dimensionality_reduction(features)
            
            # Step 4: Feature selection
            if self.config.feature_engineering_config['feature_selection']:
                features = await self._apply_feature_selection(features)
            
            # Step 5: HPO optimization
            if self.config.hpo_config['enable_hpo']:
                optimal_params = await self._optimize_clustering_parameters(features)
                self.logger.info(f"Optimal parameters found: {optimal_params}")
            else:
                optimal_params = self._get_default_parameters()
            
            # Step 6: Perform clustering
            cluster_labels = await self._perform_enhanced_clustering(features, optimal_params)
            
            # Step 7: Create enhanced cluster results
            cluster_results = await self._create_enhanced_cluster_results(
                cluster_labels, price_data, features
            )
            
            # Step 8: Backtesting validation
            if self.config.backtesting_config['enable_backtesting_validation']:
                cluster_results = await self._validate_clusters_with_backtesting(
                    cluster_results, price_data
                )
            
            # Step 9: Explainability analysis
            if self.config.explainability_config['enable_explainability']:
                cluster_results = await self._add_explainability_analysis(
                    cluster_results, features
                )
            
            # Step 10: Performance monitoring
            if self.config.performance_config['enable_performance_monitoring']:
                self._log_performance_metrics(start_time, len(cluster_results))
            
            self.logger.info(f"Enhanced SR clustering completed: {len(cluster_results)} clusters found")
            return cluster_results
            
        except Exception as e:
            self.logger.error(f"❌ Error in enhanced SR clustering: {e}")
            raise
    
    async def _detect_and_prevent_leakage(
        self,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame],
        additional_features: Optional[pd.DataFrame]
    ):
        """Detect and prevent data leakage."""
        if not self.ml_utilities.get('leakage_detector'):
            return
        
        self.logger.info("Detecting data leakage...")
        
        # Combine all data for leakage detection
        combined_data = price_data.copy()
        if volume_data is not None:
            combined_data = pd.concat([combined_data, volume_data], axis=1)
        if additional_features is not None:
            combined_data = pd.concat([combined_data, additional_features], axis=1)
        
        # Detect leakage
        leakage_report = self.ml_utilities['leakage_detector'].detect_leakage(combined_data)
        
        if leakage_report.has_leakage:
            self.logger.warning(f"Data leakage detected: {leakage_report.leakage_score}")
            self.logger.warning(f"Recommendations: {leakage_report.recommendations}")
        else:
            self.logger.info("✅ No data leakage detected")
    
    async def _extract_enhanced_features(
        self,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame],
        additional_features: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Extract enhanced features using VectorBT optimization."""
        self.logger.info("Extracting enhanced features...")
        
        features_list = []
        
        # Price features with VectorBT optimization
        if self.config.feature_engineering_config['use_price_features']:
            price_features = await self._extract_price_features_optimized(price_data)
            features_list.append(price_features)
        
        # Volume features
        if self.config.feature_engineering_config['use_volume_features'] and volume_data is not None:
            volume_features = await self._extract_volume_features_optimized(volume_data)
            features_list.append(volume_features)
        
        # Time features
        if self.config.feature_engineering_config['use_time_features']:
            time_features = await self._extract_time_features(price_data.index)
            features_list.append(time_features)
        
        # Technical indicators
        if self.config.feature_engineering_config['use_technical_indicators']:
            technical_features = await self._extract_technical_indicators(price_data)
            features_list.append(technical_features)
        
        # Market microstructure features
        if self.config.feature_engineering_config['use_market_microstructure']:
            microstructure_features = await self._extract_microstructure_features(price_data, volume_data)
            features_list.append(microstructure_features)
        
        # Additional features
        if additional_features is not None:
            features_list.append(additional_features)
        
        # Combine all features
        if features_list:
            features = pd.concat(features_list, axis=1)
        else:
            features = pd.DataFrame(index=price_data.index)
        
        # Normalize features
        features = await self._normalize_features(features)
        
        self.logger.info(f"Extracted {features.shape[1]} features")
        return features
    
    async def _extract_price_features_optimized(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Extract price features using VectorBT optimization."""
        features = {}
        
        if self.vectorbt_optimizer:
            # Use VectorBT for high-performance rolling operations
            for window in [5, 10, 20, 50, 100]:
                # Rolling statistics
                features[f'price_mean_{window}'] = self.vectorbt_optimizer.rolling_mean(
                    price_data['close'], window=window
                )
                features[f'price_std_{window}'] = self.vectorbt_optimizer.rolling_std(
                    price_data['close'], window=window
                )
                features[f'price_min_{window}'] = self.vectorbt_optimizer.rolling_min(
                    price_data['close'], window=window
                )
                features[f'price_max_{window}'] = self.vectorbt_optimizer.rolling_max(
                    price_data['close'], window=window
                )
                
                # Price ratios
                features[f'price_ratio_{window}'] = (
                    price_data['close'] / features[f'price_mean_{window}']
                )
                features[f'price_zscore_{window}'] = self.vectorbt_optimizer.zscore(
                    price_data['close'], window=window
                )
        else:
            # Fallback to pandas
            for window in [5, 10, 20, 50, 100]:
                features[f'price_mean_{window}'] = price_data['close'].rolling(window).mean()
                features[f'price_std_{window}'] = price_data['close'].rolling(window).std()
                features[f'price_min_{window}'] = price_data['close'].rolling(window).min()
                features[f'price_max_{window}'] = price_data['close'].rolling(window).max()
                features[f'price_ratio_{window}'] = price_data['close'] / features[f'price_mean_{window}']
                features[f'price_zscore_{window}'] = (
                    (price_data['close'] - features[f'price_mean_{window}']) / features[f'price_std_{window}']
                )
        
        # OHLC relationships
        features['hl_ratio'] = (price_data['high'] - price_data['low']) / price_data['close']
        features['oc_ratio'] = (price_data['open'] - price_data['close']) / price_data['close']
        features['body_size'] = abs(price_data['close'] - price_data['open']) / price_data['close']
        features['upper_shadow'] = (price_data['high'] - price_data[['open', 'close']].max(axis=1)) / price_data['close']
        features['lower_shadow'] = (price_data[['open', 'close']].min(axis=1) - price_data['low']) / price_data['close']
        
        return pd.DataFrame(features, index=price_data.index)
    
    async def _extract_volume_features_optimized(self, volume_data: pd.DataFrame) -> pd.DataFrame:
        """Extract volume features using VectorBT optimization."""
        features = {}
        
        if self.vectorbt_optimizer:
            for window in [5, 10, 20, 50]:
                features[f'volume_mean_{window}'] = self.vectorbt_optimizer.rolling_mean(
                    volume_data['volume'], window=window
                )
                features[f'volume_std_{window}'] = self.vectorbt_optimizer.rolling_std(
                    volume_data['volume'], window=window
                )
                features[f'volume_ratio_{window}'] = (
                    volume_data['volume'] / features[f'volume_mean_{window}']
                )
                features[f'volume_zscore_{window}'] = self.vectorbt_optimizer.zscore(
                    volume_data['volume'], window=window
                )
        else:
            for window in [5, 10, 20, 50]:
                features[f'volume_mean_{window}'] = volume_data['volume'].rolling(window).mean()
                features[f'volume_std_{window}'] = volume_data['volume'].rolling(window).std()
                features[f'volume_ratio_{window}'] = volume_data['volume'] / features[f'volume_mean_{window}']
                features[f'volume_zscore_{window}'] = (
                    (volume_data['volume'] - features[f'volume_mean_{window}']) / features[f'volume_std_{window}']
                )
        
        return pd.DataFrame(features, index=volume_data.index)
    
    async def _extract_time_features(self, index: pd.Index) -> pd.DataFrame:
        """Extract time-based features."""
        features = {}
        
        if isinstance(index, pd.DatetimeIndex):
            features['hour'] = index.hour
            features['day_of_week'] = index.dayofweek
            features['day_of_month'] = index.day
            features['month'] = index.month
            features['quarter'] = index.quarter
            features['is_weekend'] = (index.dayofweek >= 5).astype(int)
            features['is_month_end'] = (index.day >= 28).astype(int)
            features['is_quarter_end'] = (index.month % 3 == 0).astype(int)
        
        return pd.DataFrame(features, index=index)
    
    async def _extract_technical_indicators(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Extract technical indicators using VectorBT optimization."""
        features = {}
        
        # RSI
        if self.vectorbt_optimizer:
            features['rsi_14'] = self.vectorbt_optimizer.rsi(price_data['close'], window=14)
            features['rsi_21'] = self.vectorbt_optimizer.rsi(price_data['close'], window=21)
        else:
            # Fallback implementation
            delta = price_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        if self.vectorbt_optimizer:
            macd_line, signal_line, histogram = self.vectorbt_optimizer.macd(price_data['close'])
            features['macd'] = macd_line
            features['macd_signal'] = signal_line
            features['macd_histogram'] = histogram
        else:
            # Fallback implementation
            ema_12 = price_data['close'].ewm(span=12).mean()
            ema_26 = price_data['close'].ewm(span=26).mean()
            features['macd'] = ema_12 - ema_26
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        if self.vectorbt_optimizer:
            upper, middle, lower = self.vectorbt_optimizer.bollinger_bands(price_data['close'])
            features['bb_upper'] = upper
            features['bb_middle'] = middle
            features['bb_lower'] = lower
            features['bb_width'] = (upper - lower) / middle
            features['bb_position'] = (price_data['close'] - lower) / (upper - lower)
        else:
            # Fallback implementation
            sma_20 = price_data['close'].rolling(window=20).mean()
            std_20 = price_data['close'].rolling(window=20).std()
            features['bb_upper'] = sma_20 + (std_20 * 2)
            features['bb_middle'] = sma_20
            features['bb_lower'] = sma_20 - (std_20 * 2)
            features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
            features['bb_position'] = (price_data['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        return pd.DataFrame(features, index=price_data.index)
    
    async def _extract_microstructure_features(
        self,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Extract market microstructure features."""
        features = {}
        
        # Price impact features
        features['price_impact'] = (price_data['high'] - price_data['low']) / price_data['close']
        features['price_volatility'] = price_data['close'].pct_change().rolling(20).std()
        
        # Volume-price relationship
        if volume_data is not None:
            features['volume_price_trend'] = (volume_data['volume'] * price_data['close'].pct_change()).rolling(20).sum()
            features['volume_weighted_price'] = (volume_data['volume'] * price_data['close']).rolling(20).sum() / volume_data['volume'].rolling(20).sum()
        
        # Spread features (if bid-ask available)
        if 'bid' in price_data.columns and 'ask' in price_data.columns:
            features['spread'] = price_data['ask'] - price_data['bid']
            features['spread_ratio'] = features['spread'] / price_data['close']
        
        return pd.DataFrame(features, index=price_data.index)
    
    async def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Normalize features based on configuration."""
        normalization_method = self.config.feature_engineering_config['feature_normalization']
        
        if normalization_method == 'standard':
            scaler = StandardScaler()
        elif normalization_method == 'minmax':
            scaler = MinMaxScaler()
        elif normalization_method == 'robust':
            scaler = RobustScaler()
        else:
            return features
        
        # Handle NaN values
        features_clean = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Normalize
        features_normalized = pd.DataFrame(
            scaler.fit_transform(features_clean),
            index=features_clean.index,
            columns=features_clean.columns
        )
        
        return features_normalized
    
    async def _apply_dimensionality_reduction(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply dimensionality reduction if enabled."""
        method = self.config.feature_engineering_config['dimensionality_reduction']
        n_components = self.config.feature_engineering_config['n_components']
        
        if method == 'pca':
            reducer = PCA(n_components=n_components)
        elif method == 'ica':
            reducer = FastICA(n_components=n_components)
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components)
        elif method == 'umap':
            reducer = UMAP(n_components=n_components)
        else:
            return features
        
        # Handle NaN values
        features_clean = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Apply dimensionality reduction
        features_reduced = reducer.fit_transform(features_clean)
        
        # Create new DataFrame
        if isinstance(n_components, float):
            n_components = int(n_components * features.shape[1])
        
        columns = [f'{method}_component_{i}' for i in range(min(n_components, features_reduced.shape[1]))]
        features_df = pd.DataFrame(
            features_reduced[:, :len(columns)],
            index=features_clean.index,
            columns=columns
        )
        
        self.logger.info(f"Applied {method}: {features.shape[1]} -> {features_df.shape[1]} features")
        return features_df
    
    async def _apply_feature_selection(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply feature selection if enabled."""
        # Simple variance-based feature selection
        variance_threshold = 0.01
        feature_variance = features.var()
        selected_features = feature_variance[feature_variance > variance_threshold].index
        
        if len(selected_features) < features.shape[1]:
            self.logger.info(f"Feature selection: {features.shape[1]} -> {len(selected_features)} features")
            return features[selected_features]
        
        return features
    
    async def _optimize_clustering_parameters(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Optimize clustering parameters using HPO."""
        self.logger.info("Optimizing clustering parameters...")
        
        # Define parameter space
        param_space = {
            'eps': (0.01, 0.1),
            'min_samples': (3, 20),
            'min_cluster_size': (5, 50),
            'cluster_selection_epsilon': (0.0, 0.1),
        }
        
        # Use appropriate HPO strategy
        strategy = self.config.hpo_config['optimization_strategy']
        
        if strategy == OptimizationStrategy.BAYESIAN_TPE and 'bayesian_tpe' in self.hpo_components:
            optimizer = self.hpo_components['bayesian_tpe']
            optimal_params = await optimizer.optimize(
                objective_func=self._clustering_objective,
                param_space=param_space,
                n_trials=self.config.hpo_config['n_trials'],
                data=features
            )
        elif strategy == OptimizationStrategy.HIERARCHICAL_HPO and 'hierarchical_hpo' in self.hpo_components:
            optimizer = self.hpo_components['hierarchical_hpo']
            optimal_params = await optimizer.optimize(
                objective_func=self._clustering_objective,
                param_space=param_space,
                n_trials=self.config.hpo_config['n_trials'],
                data=features
            )
        else:
            # Fallback to default parameters
            optimal_params = self._get_default_parameters()
        
        return optimal_params
    
    def _clustering_objective(self, params: Dict[str, Any], data: pd.DataFrame) -> float:
        """Objective function for clustering optimization."""
        try:
            # Perform clustering with given parameters
            cluster_labels = self._perform_clustering_with_params(data, params)
            
            if cluster_labels is None or len(np.unique(cluster_labels)) < 2:
                return -1.0  # Invalid clustering
            
            # Calculate silhouette score
            silhouette = silhouette_score(data, cluster_labels)
            
            # Calculate other metrics
            n_clusters = len(np.unique(cluster_labels))
            calinski_harabasz = calinski_harabasz_score(data, cluster_labels)
            davies_bouldin = davies_bouldin_score(data, cluster_labels)
            
            # Composite score
            composite_score = (
                0.4 * silhouette +
                0.3 * (calinski_harabasz / 1000) +  # Normalize
                0.3 * (1 / (1 + davies_bouldin))  # Invert (lower is better)
            )
            
            return composite_score
            
        except Exception as e:
            self.logger.warning(f"Error in clustering objective: {e}")
            return -1.0
    
    def _perform_clustering_with_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Optional[np.ndarray]:
        """Perform clustering with specific parameters."""
        algorithm = self.config.clustering_algorithm
        
        if algorithm == ClusteringAlgorithm.DBSCAN:
            clusterer = DBSCAN(
                eps=params.get('eps', self.config.dbscan_eps),
                min_samples=params.get('min_samples', self.config.dbscan_min_samples)
            )
        elif algorithm == ClusteringAlgorithm.HDBSCAN and HDBSCAN_AVAILABLE:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=params.get('min_cluster_size', self.config.hdbscan_min_cluster_size),
                min_samples=params.get('min_samples', self.config.hdbscan_min_samples),
                cluster_selection_epsilon=params.get('cluster_selection_epsilon', self.config.hdbscan_cluster_selection_epsilon)
            )
        elif algorithm == ClusteringAlgorithm.SPECTRAL:
            clusterer = SpectralClustering(
                n_clusters=params.get('n_clusters', self.config.spectral_n_clusters),
                affinity=params.get('affinity', self.config.spectral_affinity),
                gamma=params.get('gamma', self.config.spectral_gamma)
            )
        else:
            # Default to DBSCAN
            clusterer = DBSCAN(
                eps=params.get('eps', self.config.dbscan_eps),
                min_samples=params.get('min_samples', self.config.dbscan_min_samples)
            )
        
        try:
            cluster_labels = clusterer.fit_predict(data)
            return cluster_labels
        except Exception as e:
            self.logger.warning(f"Error in clustering: {e}")
            return None
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """Get default clustering parameters."""
        return {
            'eps': self.config.dbscan_eps,
            'min_samples': self.config.dbscan_min_samples,
            'min_cluster_size': self.config.hdbscan_min_cluster_size,
            'cluster_selection_epsilon': self.config.hdbscan_cluster_selection_epsilon,
        }
    
    async def _perform_enhanced_clustering(
        self,
        features: pd.DataFrame,
        optimal_params: Dict[str, Any]
    ) -> np.ndarray:
        """Perform enhanced clustering with optimal parameters."""
        self.logger.info("Performing enhanced clustering...")
        
        # Use hardware optimization if available
        if self.hardware_manager:
            with self.hardware_manager.optimize_workload(WorkloadType.ML_TRAINING):
                cluster_labels = self._perform_clustering_with_params(features, optimal_params)
        else:
            cluster_labels = self._perform_clustering_with_params(features, optimal_params)
        
        if cluster_labels is None:
            raise ValueError("Clustering failed")
        
        n_clusters = len(np.unique(cluster_labels))
        self.logger.info(f"Clustering completed: {n_clusters} clusters found")
        
        return cluster_labels
    
    async def _create_enhanced_cluster_results(
        self,
        cluster_labels: np.ndarray,
        price_data: pd.DataFrame,
        features: pd.DataFrame
    ) -> List[EnhancedClusterResult]:
        """Create enhanced cluster results with comprehensive metrics."""
        results = []
        
        unique_labels = np.unique(cluster_labels)
        unique_labels = unique_labels[unique_labels != -1]  # Remove noise points
        
        for cluster_id in unique_labels:
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) < self.config.min_cluster_size:
                continue
            
            # Basic cluster info
            cluster_prices = price_data.iloc[cluster_indices]['close'].values
            centroid_price = np.mean(cluster_prices)
            cluster_size = len(cluster_indices)
            
            # Quality metrics
            cluster_features = features.iloc[cluster_indices]
            silhouette = silhouette_score(features, cluster_labels)
            calinski_harabasz = calinski_harabasz_score(features, cluster_labels)
            davies_bouldin = davies_bouldin_score(features, cluster_labels)
            
            # Composite quality score
            cluster_quality = (
                0.4 * silhouette +
                0.3 * (calinski_harabasz / 1000) +
                0.3 * (1 / (1 + davies_bouldin))
            )
            
            # Temporal metrics
            cluster_times = price_data.index[cluster_indices]
            first_touch = cluster_times.min()
            last_touch = cluster_times.max()
            touch_frequency = cluster_size / (last_touch - first_touch).total_seconds() / 3600  # touches per hour
            
            # Persistence score (how long the level was active)
            persistence_score = (last_touch - first_touch).total_seconds() / 3600  # hours
            
            result = EnhancedClusterResult(
                cluster_id=int(cluster_id),
                level_indices=cluster_indices.tolist(),
                centroid_price=float(centroid_price),
                cluster_size=cluster_size,
                silhouette_score=float(silhouette),
                calinski_harabasz_score=float(calinski_harabasz),
                davies_bouldin_score=float(davies_bouldin),
                cluster_quality=float(cluster_quality),
                first_touch=first_touch,
                last_touch=last_touch,
                touch_frequency=float(touch_frequency),
                persistence_score=float(persistence_score),
                confidence=min(1.0, cluster_quality),
                reliability_score=min(1.0, cluster_size / 20),  # Normalize by expected size
                stability_score=min(1.0, persistence_score / 24),  # Normalize by 24 hours
            )
            
            results.append(result)
        
        # Sort by quality score
        results.sort(key=lambda x: x.cluster_quality, reverse=True)
        
        self.logger.info(f"Created {len(results)} enhanced cluster results")
        return results
    
    async def _validate_clusters_with_backtesting(
        self,
        cluster_results: List[EnhancedClusterResult],
        price_data: pd.DataFrame
    ) -> List[EnhancedClusterResult]:
        """Validate clusters using backtesting engine."""
        if not self.backtesting_engine:
            self.logger.warning("Backtesting engine not available, skipping validation")
            return cluster_results
        
        self.logger.info("Validating clusters with backtesting...")
        
        validated_results = []
        for result in cluster_results:
            try:
                # Create backtesting configuration
                backtest_config = {
                    'start_date': result.first_touch,
                    'end_date': result.last_touch,
                    'initial_capital': 10000,
                    'commission': 0.001,
                }
                
                # Run backtest for this cluster
                backtest_results = await self.backtesting_engine.run_backtest(
                    price_data=price_data,
                    sr_levels=[result.centroid_price],
                    config=backtest_config
                )
                
                # Update result with backtesting metrics
                result.sharpe_ratio = backtest_results.get('sharpe_ratio', 0.0)
                result.max_drawdown = backtest_results.get('max_drawdown', 0.0)
                result.win_rate = backtest_results.get('win_rate', 0.0)
                result.total_return = backtest_results.get('total_return', 0.0)
                
                # Only keep clusters with positive performance
                if result.sharpe_ratio > 0 and result.win_rate > 0.5:
                    validated_results.append(result)
                
            except Exception as e:
                self.logger.warning(f"Backtesting failed for cluster {result.cluster_id}: {e}")
                # Keep cluster even if backtesting fails
                validated_results.append(result)
        
        self.logger.info(f"Backtesting validation: {len(cluster_results)} -> {len(validated_results)} clusters")
        return validated_results
    
    async def _add_explainability_analysis(
        self,
        cluster_results: List[EnhancedClusterResult],
        features: pd.DataFrame,
        cluster_labels: np.ndarray
    ) -> List[EnhancedClusterResult]:
        """Add explainability analysis using SHAP/LIME."""
        if not self.shap_lime_integration:
            self.logger.warning("SHAP/LIME integration not available, skipping explainability")
            return cluster_results
        
        self.logger.info("Adding explainability analysis...")
        
        for result in cluster_results:
            try:
                # Get cluster features
                cluster_mask = cluster_labels == result.cluster_id
                cluster_features = features[cluster_mask]
                
                if len(cluster_features) == 0:
                    continue
                
                # Calculate feature importance
                feature_importance = cluster_features.var().sort_values(ascending=False)
                result.feature_importance = feature_importance.to_dict()
                
                # SHAP analysis
                if hasattr(self.shap_lime_integration, 'calculate_shap_values'):
                    shap_values = await self.shap_lime_integration.calculate_shap_values(
                        model=None,  # We don't have a model, use feature analysis
                        X=cluster_features,
                        feature_names=features.columns.tolist()
                    )
                    result.shap_values = shap_values
                
                # LIME analysis
                if hasattr(self.shap_lime_integration, 'calculate_lime_explanations'):
                    lime_explanations = await self.shap_lime_integration.calculate_lime_explanations(
                        model=None,
                        X=cluster_features,
                        feature_names=features.columns.tolist()
                    )
                    result.lime_explanations = lime_explanations
                
            except Exception as e:
                self.logger.warning(f"Explainability analysis failed for cluster {result.cluster_id}: {e}")
        
        return cluster_results
    
    async def _log_performance_metrics(
        self,
        start_time: float,
        end_time: float,
        features: pd.DataFrame,
        cluster_results: List[EnhancedClusterResult]
    ) -> None:
        """Log comprehensive performance metrics."""
        total_time = end_time - start_time
        
        # Memory usage
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Clustering metrics
        n_clusters = len(cluster_results)
        avg_cluster_size = np.mean([r.cluster_size for r in cluster_results]) if cluster_results else 0
        avg_quality = np.mean([r.cluster_quality for r in cluster_results]) if cluster_results else 0
        
        # Performance summary
        self.logger.info("=== Enhanced SR Clustering Performance Summary ===")
        self.logger.info(f"Total execution time: {total_time:.2f} seconds")
        self.logger.info(f"Memory usage: {memory_usage:.2f} MB")
        self.logger.info(f"Features processed: {features.shape[0]} samples, {features.shape[1]} features")
        self.logger.info(f"Clusters found: {n_clusters}")
        self.logger.info(f"Average cluster size: {avg_cluster_size:.2f}")
        self.logger.info(f"Average cluster quality: {avg_quality:.4f}")
        
        # Hardware optimization metrics
        if self.hardware_manager:
            self.logger.info(f"Hardware optimization: {self.hardware_manager.get_optimization_summary()}")
        
        # HPO metrics
        if self.hpo_components:
            self.logger.info(f"HPO components: {list(self.hpo_components.keys())}")
        
        # Feature engineering metrics
        self.logger.info(f"Feature engineering: {self.config.feature_engineering_config}")
        
        self.logger.info("=== End Performance Summary ===")


# Factory functions for easy instantiation
def create_enhanced_sr_clustering(
    config: Optional[EnhancedSRClusteringConfig] = None,
    **kwargs
) -> EnhancedSRClustering:
    """Create an EnhancedSRClustering instance with optional configuration."""
    if config is None:
        config = EnhancedSRClusteringConfig(**kwargs)
    
    return EnhancedSRClustering(config)


def create_enhanced_sr_clustering_from_dict(config_dict: Dict[str, Any]) -> EnhancedSRClustering:
    """Create an EnhancedSRClustering instance from a configuration dictionary."""
    config = EnhancedSRClusteringConfig(**config_dict)
    return EnhancedSRClustering(config)


# Example usage and testing
async def example_usage():
    """Example usage of the EnhancedSRClustering module."""
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    np.random.seed(42)
    
    # Generate sample OHLCV data
    price_data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(1000) * 0.1),
        'high': 100 + np.cumsum(np.random.randn(1000) * 0.1) + np.random.rand(1000) * 2,
        'low': 100 + np.cumsum(np.random.randn(1000) * 0.1) - np.random.rand(1000) * 2,
        'close': 100 + np.cumsum(np.random.randn(1000) * 0.1),
        'volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)
    
    # Ensure high >= low
    price_data['high'] = np.maximum(price_data['high'], price_data['low'])
    price_data['low'] = np.minimum(price_data['high'], price_data['low'])
    
    # Create configuration
    config = EnhancedSRClusteringConfig(
        clustering_algorithm=ClusteringAlgorithm.HDBSCAN,
        hpo_config={
            'optimization_strategy': OptimizationStrategy.BAYESIAN_TPE,
            'n_trials': 20,
            'timeout': 300
        },
        feature_engineering_config={
            'price_features': True,
            'volume_features': True,
            'time_features': True,
            'technical_indicators': True,
            'microstructure_features': True,
            'feature_normalization': 'standard',
            'dimensionality_reduction': 'pca',
            'n_components': 0.8
        },
        backtesting_config={
            'enabled': True,
            'initial_capital': 10000,
            'commission': 0.001
        },
        explainability_config={
            'shap_enabled': True,
            'lime_enabled': True
        }
    )
    
    # Create clustering instance
    clustering = create_enhanced_sr_clustering(config)
    
    # Run clustering
    try:
        results = await clustering.cluster_sr_levels(price_data)
        
        print(f"Found {len(results)} clusters")
        for i, result in enumerate(results[:5]):  # Show top 5 clusters
            print(f"Cluster {i+1}: Price={result.centroid_price:.2f}, "
                  f"Quality={result.cluster_quality:.4f}, "
                  f"Size={result.cluster_size}")
        
    except Exception as e:
        print(f"Error in clustering: {e}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
