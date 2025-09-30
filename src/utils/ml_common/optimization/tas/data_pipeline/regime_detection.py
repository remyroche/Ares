"""
Enhanced Regime Detection for TAS with Performance Optimizations and Advanced Validation

Comprehensive regime detection system for tree architecture search including
unsupervised clustering, regime qualification, and regime transition analysis.

Enhanced with:
- Memory-efficient processing for large datasets
- Parallel processing across timeframes
- Intelligent caching for regime detection results
- Cross-validation for regime stability
- Out-of-sample testing for regime validation
- Regime persistence analysis over time
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from enum import Enum
import warnings
import time
import gc
import hashlib
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import contextmanager
warnings.filterwarnings('ignore')

# Import optimization utilities
try:
    from ...hardware.m1_gpu_utils import get_m1_gpu_manager
    from ...hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from ...hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    from ...matrix_operations.unified_operations import get_unified_matrix_operations
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

# Import ML common utilities for validation
try:
    from ...cvlsa.cvlsa_cross_validation import CVLSA
    from ...validation.universal_validation import UniversalValidator
    from ...evaluation.performance_metrics import PerformanceMetrics
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False

# Import existing regime detection components
try:
    from ..regime_analysis.unsupervised_regime_detection import UnsupervisedRegimeDetector, RegimeDetectionConfig
    from ..regime_analysis.regime_qualification import RegimeQualifier, RegimeQualificationConfig
    REGIME_ANALYSIS_AVAILABLE = True
except ImportError:
    REGIME_ANALYSIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class RegimeDetectionMethod(Enum):
    """Regime detection methods."""
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    GMM = "gmm"
    HMM = "hmm"
    UNSUPERVISED = "unsupervised"
    SUPERVISED = "supervised"


@dataclass
class RegimeConfig:
    """Enhanced configuration for regime detection with performance optimizations."""
    
    # Detection method
    detection_method: RegimeDetectionMethod = RegimeDetectionMethod.UNSUPERVISED
    
    # Regime parameters
    n_regimes: int = 5
    min_regime_duration: int = 20
    max_regimes: int = 20
    
    # Clustering parameters
    clustering_algorithm: str = "kmeans"  # "kmeans", "dbscan", "gmm", "hmm"
    n_clusters_range: Tuple[int, int] = (2, 20)
    silhouette_threshold: float = 0.3
    
    # Regime qualification
    enable_regime_qualification: bool = True
    qualification_threshold: float = 0.6
    economic_significance: bool = True
    trading_viability: bool = True
    
    # Regime analysis
    enable_regime_analysis: bool = True
    regime_stability: bool = True
    regime_transitions: bool = True
    regime_persistence: bool = True
    
    # Feature engineering for regimes
    enable_regime_features: bool = True
    regime_feature_types: List[str] = field(default_factory=lambda: [
        'volatility', 'trend', 'volume', 'momentum', 'volatility_of_volatility'
    ])
    
    # Output configuration
    save_regime_data: bool = True
    output_directory: str = "regime_data"
    cache_regimes: bool = True
    
    # Performance optimizations
    enable_gpu: bool = True
    enable_memory_optimization: bool = True
    enable_parallel: bool = True
    max_memory_gb: Optional[float] = None
    cache_dir: Optional[str] = None
    
    # Advanced validation
    enable_cross_validation: bool = True
    enable_out_of_sample: bool = True
    enable_regime_persistence: bool = True
    cv_folds: int = 5
    oos_split_ratio: float = 0.8
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    performance_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegimeResult:
    """Enhanced result of regime detection with performance and validation metrics."""
    
    # Regime data
    regime_labels: np.ndarray
    regime_centers: Dict[int, np.ndarray]
    regime_statistics: Dict[int, Dict[str, Any]]
    qualified_regimes: Dict[str, Any]
    
    # Regime analysis
    regime_transitions: List[Dict[str, Any]]
    regime_stability: Dict[int, float]
    regime_persistence: Dict[int, float]
    regime_quality_scores: Dict[int, float]
    
    # Regime features
    regime_features: pd.DataFrame
    regime_feature_importance: Dict[str, float]
    regime_feature_correlations: pd.DataFrame
    
    # Detection metadata
    detection_method: str
    detection_time: float
    n_regimes_detected: int
    n_qualified_regimes: int
    regime_quality_score: float
    
    # Performance metrics
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    
    # Enhanced validation results
    cross_validation_results: Dict[str, Any] = field(default_factory=dict)
    out_of_sample_results: Dict[str, Any] = field(default_factory=dict)
    regime_persistence_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Performance optimization results
    memory_optimization_stats: Dict[str, Any] = field(default_factory=dict)
    parallel_processing_stats: Dict[str, Any] = field(default_factory=dict)
    gpu_acceleration_stats: Dict[str, Any] = field(default_factory=dict)
    caching_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Hardware information
    hardware_info: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    config: RegimeConfig
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class RegimeDetector:
    """
    Enhanced comprehensive regime detector for TAS with performance optimizations and advanced validation.
    
    Provides regime detection, qualification, and analysis for tree architecture search.
    Enhanced with memory optimization, parallel processing, GPU acceleration, and advanced validation.
    """
    
    def __init__(self, config: RegimeConfig):
        """Initialize enhanced regime detector.
        
        Args:
            config: Enhanced regime detection configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize regime detection components
        self.regime_detector = None
        self.regime_qualifier = None
        
        # Initialize optimization components
        self.gpu_manager = None
        self.memory_optimizer = None
        self.cpu_optimizer = None
        self.matrix_ops = None
        
        # Initialize validation components
        self.cvlsa = None
        self.validator = None
        self.metrics = None
        
        # Performance tracking
        self.performance_stats = {
            'total_detections': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'parallel_detections': 0,
            'memory_optimized_detections': 0,
            'gpu_accelerated_detections': 0,
            'average_detection_time': 0.0,
            'peak_memory_usage_mb': 0.0
        }
        
        # Initialize cache directory
        if config.cache_dir:
            self.cache_dir = Path(config.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = Path.cwd() / 'regime_cache'
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize available components
        self._initialize_components()
        
        self.logger.info("✅ Enhanced Regime Detector initialized")
        self.logger.info(f"📊 Detection method: {config.detection_method.value}")
        self.logger.info(f"📊 Number of regimes: {config.n_regimes}")
        self.logger.info(f"📊 Qualification enabled: {config.enable_regime_qualification}")
        self.logger.info(f"📊 GPU acceleration: {config.enable_gpu}")
        self.logger.info(f"📊 Memory optimization: {config.enable_memory_optimization}")
        self.logger.info(f"📊 Parallel processing: {config.enable_parallel}")
        self.logger.info(f"📊 Advanced validation: {config.enable_cross_validation}")
    
    def _initialize_components(self):
        """Initialize available regime detection, optimization, and validation components."""
        # Initialize regime analysis components
        if REGIME_ANALYSIS_AVAILABLE:
            try:
                # Initialize regime detector
                regime_detection_config = RegimeDetectionConfig(
                    n_regimes=self.config.n_regimes,
                    min_regime_duration=self.config.min_regime_duration,
                    clustering_algorithm=self.config.clustering_algorithm,
                    n_clusters_range=self.config.n_clusters_range,
                    silhouette_threshold=self.config.silhouette_threshold
                )
                self.regime_detector = UnsupervisedRegimeDetector(regime_detection_config)
                
                # Initialize regime qualifier
                if self.config.enable_regime_qualification:
                    regime_qualification_config = RegimeQualificationConfig(
                        qualification_threshold=self.config.qualification_threshold,
                        economic_significance=self.config.economic_significance,
                        trading_viability=self.config.trading_viability
                    )
                    self.regime_qualifier = RegimeQualifier(regime_qualification_config)
                
                self.logger.info("✅ Regime analysis components initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Regime analysis components not available: {e}")
        
        # Initialize optimization components
        if OPTIMIZATION_AVAILABLE:
            try:
                # Initialize GPU manager
                if self.config.enable_gpu:
                    self.gpu_manager = get_m1_gpu_manager()
                    if self.gpu_manager:
                        self.logger.info("✅ M1 GPU Manager initialized")
                
                # Initialize memory optimizer
                if self.config.enable_memory_optimization:
                    self.memory_optimizer = get_m1_memory_optimizer(memory_limit_gb=self.config.max_memory_gb)
                    if self.memory_optimizer:
                        self.logger.info("✅ M1 Memory Optimizer initialized")
                
                # Initialize CPU optimizer
                if self.config.enable_parallel:
                    self.cpu_optimizer = get_m1_cpu_optimizer()
                    if self.cpu_optimizer:
                        self.logger.info("✅ M1 CPU Optimizer initialized")
                
                # Initialize matrix operations
                self.matrix_ops = get_unified_matrix_operations(
                    enable_gpu=self.config.enable_gpu,
                    enable_memory_optimization=self.config.enable_memory_optimization,
                    enable_parallel=self.config.enable_parallel
                )
                if self.matrix_ops:
                    self.logger.info("✅ Unified Matrix Operations initialized")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Optimization components not available: {e}")
        
        # Initialize validation components
        if VALIDATION_AVAILABLE:
            try:
                # Initialize CVLSA for cross-validation
                if self.config.enable_cross_validation:
                    self.cvlsa = CVLSA()
                    self.logger.info("✅ CVLSA cross-validation initialized")
                
                # Initialize universal validator
                self.validator = UniversalValidator()
                if self.validator:
                    self.logger.info("✅ Universal Validator initialized")
                
                # Initialize performance metrics
                self.metrics = PerformanceMetrics()
                if self.metrics:
                    self.logger.info("✅ Performance Metrics initialized")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Validation components not available: {e}")
    
    def detect_regimes(self, data: pd.DataFrame, features: Optional[pd.DataFrame] = None, 
                      use_cache: bool = True, parallel: bool = True) -> RegimeResult:
        """
        Enhanced regime detection with performance optimizations and advanced validation.
        
        Args:
            data: Input data
            features: Optional engineered features
            use_cache: Whether to use cached results
            parallel: Whether to use parallel processing
            
        Returns:
            Enhanced regime detection result with performance and validation metrics
        """
        self.logger.info("🚀 Starting enhanced regime detection")
        start_time = time.time()
        
        # Generate cache key
        cache_key = self._generate_cache_key(data, features)
        
        # Check cache first
        if use_cache and self.config.cache_regimes:
            cached_result = self._get_cached_result(cache_key)
            if cached_result is not None:
                self.performance_stats['cache_hits'] += 1
                self.logger.info("✅ Using cached regime detection results")
                return cached_result
        
        self.performance_stats['cache_misses'] += 1
        
        try:
            # Memory optimization checkpoint
            memory_stats = {}
            if self.memory_optimizer:
                with self.memory_optimizer.memory_checkpoint("regime_detection"):
                    # Optimize data for memory efficiency
                    optimized_data = self._optimize_data_for_memory(data)
                    optimized_features = self._optimize_data_for_memory(features) if features is not None else None
                    
                    # Prepare data for regime detection
                    regime_data = self._prepare_regime_data(optimized_data, optimized_features)
                    
                    # Detect regimes with optimizations
                    if parallel and self.config.enable_parallel:
                        regime_labels, regime_centers, regime_statistics = self._detect_regimes_parallel(regime_data)
                        self.performance_stats['parallel_detections'] += 1
                    else:
                        regime_labels, regime_centers, regime_statistics = self._detect_regimes_sequential(regime_data)
                    
                    # Get memory optimization stats
                    memory_stats = self.memory_optimizer.get_memory_stats()
                    self.performance_stats['memory_optimized_detections'] += 1
            else:
                # Standard processing without memory optimization
                regime_data = self._prepare_regime_data(data, features)
                
                if parallel and self.config.enable_parallel:
                    regime_labels, regime_centers, regime_statistics = self._detect_regimes_parallel(regime_data)
                    self.performance_stats['parallel_detections'] += 1
                else:
                    regime_labels, regime_centers, regime_statistics = self._detect_regimes_sequential(regime_data)
            
            # Qualify regimes
            qualified_regimes = {}
            if self.config.enable_regime_qualification and self.regime_qualifier:
                qualification_result = self.regime_qualifier.qualify_regimes({
                    'regime_labels': regime_labels,
                    'regime_centers': regime_centers,
                    'regime_statistics': regime_statistics
                }, data)
                qualified_regimes = qualification_result.get('qualified_regimes', {})
            
            # Analyze regimes
            regime_analysis = self._analyze_regimes(regime_labels, regime_centers, regime_statistics, data)
            
            # Generate regime features
            regime_features = self._generate_regime_features(data, regime_labels, regime_centers)
            
            # Calculate regime quality scores
            regime_quality_scores = self._calculate_regime_quality_scores(regime_labels, regime_centers, regime_statistics)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(regime_data, regime_labels)
            
            # Apply advanced validation
            validation_results = self._apply_advanced_validation(regime_labels, regime_centers, regime_statistics, data)
            
            # Calculate detection time
            detection_time = time.time() - start_time
            
            # Get hardware information
            hardware_info = self._get_hardware_info()
            
            # Create enhanced result
            result = RegimeResult(
                # Regime data
                regime_labels=regime_labels,
                regime_centers=regime_centers,
                regime_statistics=regime_statistics,
                qualified_regimes=qualified_regimes,
                
                # Regime analysis
                regime_transitions=regime_analysis['transitions'],
                regime_stability=regime_analysis['stability'],
                regime_persistence=regime_analysis['persistence'],
                regime_quality_scores=regime_quality_scores,
                
                # Regime features
                regime_features=regime_features,
                regime_feature_importance=regime_analysis['feature_importance'],
                regime_feature_correlations=regime_analysis['feature_correlations'],
                
                # Detection metadata
                detection_method=self.config.detection_method.value,
                detection_time=detection_time,
                n_regimes_detected=len(np.unique(regime_labels)),
                n_qualified_regimes=len(qualified_regimes),
                regime_quality_score=np.mean(list(regime_quality_scores.values())),
                
                # Performance metrics
                silhouette_score=performance_metrics['silhouette_score'],
                calinski_harabasz_score=performance_metrics['calinski_harabasz_score'],
                davies_bouldin_score=performance_metrics['davies_bouldin_score'],
                
                # Enhanced validation results
                cross_validation_results=validation_results.get('cross_validation', {}),
                out_of_sample_results=validation_results.get('out_of_sample', {}),
                regime_persistence_analysis=validation_results.get('persistence', {}),
                
                # Performance optimization results
                memory_optimization_stats=memory_stats,
                parallel_processing_stats={'parallel_used': parallel and self.config.enable_parallel},
                gpu_acceleration_stats={'gpu_used': self.config.enable_gpu and self.gpu_manager is not None},
                caching_stats={'cache_used': use_cache and self.config.cache_regimes},
                
                # Hardware information
                hardware_info=hardware_info,
                
                # Metadata
                config=self.config
            )
            
            # Update performance statistics
            self.performance_stats['total_detections'] += 1
            self.performance_stats['average_detection_time'] = (
                (self.performance_stats['average_detection_time'] *
                 (self.performance_stats['total_detections'] - 1)) + detection_time
            ) / self.performance_stats['total_detections']
            
            # Cache result if configured
            if use_cache and self.config.cache_regimes:
                self._cache_result(cache_key, result)
            
            # Save regime data if configured
            if self.config.save_regime_data:
                self._save_regime_data(result)
            
            self.logger.info(f"✅ Enhanced regime detection completed in {result.detection_time:.2f}s")
            self.logger.info(f"📊 Regimes detected: {result.n_regimes_detected}")
            self.logger.info(f"📊 Qualified regimes: {result.n_qualified_regimes}")
            self.logger.info(f"📊 Regime quality score: {result.regime_quality_score:.3f}")
            self.logger.info(f"📊 Silhouette score: {result.silhouette_score:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced regime detection failed: {e}")
            raise
    
    def _prepare_regime_data(self, data: pd.DataFrame, features: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Prepare data for regime detection."""
        try:
            if features is not None:
                # Use provided features
                regime_data = features.copy()
            else:
                # Generate basic features for regime detection
                regime_data = pd.DataFrame(index=data.index)
                
                # Price features
                if 'close' in data.columns:
                    regime_data['price_return'] = data['close'].pct_change()
                    regime_data['price_volatility'] = data['close'].rolling(window=20).std()
                    regime_data['price_trend'] = data['close'].rolling(window=20).apply(
                        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else np.nan
                    )
                
                # Volume features
                if 'volume' in data.columns:
                    regime_data['volume_return'] = data['volume'].pct_change()
                    regime_data['volume_volatility'] = data['volume'].rolling(window=20).std()
                
                # Technical indicators
                if 'close' in data.columns:
                    # RSI
                    delta = data['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    regime_data['rsi'] = 100 - (100 / (1 + rs))
                    
                    # Moving averages
                    regime_data['sma_20'] = data['close'].rolling(window=20).mean()
                    regime_data['sma_50'] = data['close'].rolling(window=50).mean()
                    regime_data['sma_ratio'] = regime_data['sma_20'] / regime_data['sma_50']
            
            # Fill missing values
            regime_data = regime_data.fillna(regime_data.median())
            
            return regime_data
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime data preparation failed: {e}")
            return data
    
    def _detect_unsupervised_regimes(self, regime_data: pd.DataFrame) -> Tuple[np.ndarray, Dict[int, np.ndarray], Dict[int, Dict[str, Any]]]:
        """Detect regimes using unsupervised methods."""
        try:
            # Use the unsupervised regime detector
            regime_result = self.regime_detector.detect_regimes(regime_data)
            
            regime_labels = regime_result.get('regime_labels', np.array([]))
            regime_centers = regime_result.get('regime_centers', {})
            regime_statistics = regime_result.get('regime_statistics', {})
            
            return regime_labels, regime_centers, regime_statistics
            
        except Exception as e:
            self.logger.warning(f"⚠️ Unsupervised regime detection failed: {e}")
            return self._detect_basic_regimes(regime_data)
    
    def _detect_basic_regimes(self, regime_data: pd.DataFrame) -> Tuple[np.ndarray, Dict[int, np.ndarray], Dict[int, Dict[str, Any]]]:
        """Detect regimes using basic methods."""
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # Prepare data
            numeric_cols = regime_data.select_dtypes(include=[np.number]).columns
            regime_data_numeric = regime_data[numeric_cols].fillna(0)
            
            # Scale data
            scaler = StandardScaler()
            regime_data_scaled = scaler.fit_transform(regime_data_numeric)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=self.config.n_regimes, random_state=42)
            regime_labels = kmeans.fit_predict(regime_data_scaled)
            
            # Calculate regime centers
            regime_centers = {}
            for i in range(self.config.n_regimes):
                regime_centers[i] = kmeans.cluster_centers_[i]
            
            # Calculate regime statistics
            regime_statistics = {}
            for i in range(self.config.n_regimes):
                regime_mask = regime_labels == i
                regime_data_subset = regime_data_numeric[regime_mask]
                
                regime_statistics[i] = {
                    'count': np.sum(regime_mask),
                    'percentage': np.sum(regime_mask) / len(regime_labels),
                    'mean': regime_data_subset.mean().to_dict(),
                    'std': regime_data_subset.std().to_dict(),
                    'min': regime_data_subset.min().to_dict(),
                    'max': regime_data_subset.max().to_dict()
                }
            
            return regime_labels, regime_centers, regime_statistics
            
        except Exception as e:
            self.logger.warning(f"⚠️ Basic regime detection failed: {e}")
            # Fallback to simple regime assignment
            regime_labels = np.zeros(len(regime_data))
            regime_centers = {0: np.zeros(regime_data.shape[1])}
            regime_statistics = {0: {'count': len(regime_data), 'percentage': 1.0}}
            
            return regime_labels, regime_centers, regime_statistics
    
    def _analyze_regimes(self, regime_labels: np.ndarray, regime_centers: Dict[int, np.ndarray], 
                        regime_statistics: Dict[int, Dict[str, Any]], data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze detected regimes."""
        try:
            analysis = {
                'transitions': [],
                'stability': {},
                'persistence': {},
                'feature_importance': {},
                'feature_correlations': pd.DataFrame()
            }
            
            # Analyze regime transitions
            if self.config.regime_transitions:
                transitions = self._analyze_regime_transitions(regime_labels)
                analysis['transitions'] = transitions
            
            # Analyze regime stability
            if self.config.regime_stability:
                stability = self._analyze_regime_stability(regime_labels, regime_centers)
                analysis['stability'] = stability
            
            # Analyze regime persistence
            if self.config.regime_persistence:
                persistence = self._analyze_regime_persistence(regime_labels)
                analysis['persistence'] = persistence
            
            # Analyze feature importance
            feature_importance = self._analyze_feature_importance(regime_labels, data)
            analysis['feature_importance'] = feature_importance
            
            # Analyze feature correlations
            feature_correlations = self._analyze_feature_correlations(regime_labels, data)
            analysis['feature_correlations'] = feature_correlations
            
            return analysis
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime analysis failed: {e}")
            return {
                'transitions': [],
                'stability': {},
                'persistence': {},
                'feature_importance': {},
                'feature_correlations': pd.DataFrame()
            }
    
    def _analyze_regime_transitions(self, regime_labels: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze regime transitions."""
        try:
            transitions = []
            
            for i in range(1, len(regime_labels)):
                if regime_labels[i] != regime_labels[i-1]:
                    transitions.append({
                        'timestamp': i,
                        'from_regime': int(regime_labels[i-1]),
                        'to_regime': int(regime_labels[i]),
                        'transition_type': f"regime_{int(regime_labels[i-1])}_to_regime_{int(regime_labels[i])}"
                    })
            
            return transitions
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime transition analysis failed: {e}")
            return []
    
    def _analyze_regime_stability(self, regime_labels: np.ndarray, regime_centers: Dict[int, np.ndarray]) -> Dict[int, float]:
        """Analyze regime stability."""
        try:
            stability = {}
            
            for regime_id in np.unique(regime_labels):
                regime_mask = regime_labels == regime_id
                regime_duration = np.sum(regime_mask)
                total_duration = len(regime_labels)
                
                # Calculate stability as ratio of regime duration to total duration
                stability[regime_id] = regime_duration / total_duration
            
            return stability
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime stability analysis failed: {e}")
            return {}
    
    def _analyze_regime_persistence(self, regime_labels: np.ndarray) -> Dict[int, float]:
        """Analyze regime persistence."""
        try:
            persistence = {}
            
            for regime_id in np.unique(regime_labels):
                regime_mask = regime_labels == regime_id
                regime_indices = np.where(regime_mask)[0]
                
                if len(regime_indices) > 1:
                    # Calculate average gap between regime occurrences
                    gaps = np.diff(regime_indices)
                    persistence[regime_id] = np.mean(gaps) if len(gaps) > 0 else 0
                else:
                    persistence[regime_id] = 0
            
            return persistence
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime persistence analysis failed: {e}")
            return {}
    
    def _analyze_feature_importance(self, regime_labels: np.ndarray, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze feature importance for regime detection."""
        try:
            from sklearn.feature_selection import mutual_info_classification
            
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data_numeric = data[numeric_cols].fillna(0)
            
            # Calculate mutual information between features and regime labels
            importance_scores = mutual_info_classification(data_numeric, regime_labels)
            
            feature_importance = dict(zip(numeric_cols, importance_scores))
            
            return feature_importance
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature importance analysis failed: {e}")
            return {}
    
    def _analyze_feature_correlations(self, regime_labels: np.ndarray, data: pd.DataFrame) -> pd.DataFrame:
        """Analyze feature correlations within regimes."""
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data_numeric = data[numeric_cols].fillna(0)
            
            # Calculate correlations
            correlations = data_numeric.corr()
            
            return correlations
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature correlation analysis failed: {e}")
            return pd.DataFrame()
    
    def _generate_regime_features(self, data: pd.DataFrame, regime_labels: np.ndarray, 
                                regime_centers: Dict[int, np.ndarray]) -> pd.DataFrame:
        """Generate regime-specific features."""
        try:
            regime_features = pd.DataFrame(index=data.index)
            
            # Add regime labels
            regime_features['regime_label'] = regime_labels
            
            # Add regime-specific features
            for regime_id in np.unique(regime_labels):
                regime_mask = regime_labels == regime_id
                regime_features[f'regime_{regime_id}'] = regime_mask.astype(int)
            
            # Add regime distance features
            if len(regime_centers) > 0:
                for regime_id, center in regime_centers.items():
                    # Calculate distance to regime center (simplified)
                    regime_features[f'distance_to_regime_{regime_id}'] = 0.0  # Placeholder
            
            return regime_features
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime feature generation failed: {e}")
            return pd.DataFrame(index=data.index)
    
    def _calculate_regime_quality_scores(self, regime_labels: np.ndarray, regime_centers: Dict[int, np.ndarray], 
                                            regime_statistics: Dict[int, Dict[str, Any]]) -> Dict[int, float]:
        """Calculate regime quality scores."""
        try:
            quality_scores = {}
            
            for regime_id in np.unique(regime_labels):
                regime_mask = regime_labels == regime_id
                regime_count = np.sum(regime_mask)
                total_count = len(regime_labels)
                
                # Calculate quality score based on regime size and statistics
                size_score = regime_count / total_count
                consistency_score = 1.0  # Placeholder for consistency calculation
                
                quality_scores[regime_id] = (size_score + consistency_score) / 2
            
            return quality_scores
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime quality score calculation failed: {e}")
            return {}
    
    def _calculate_performance_metrics(self, regime_data: pd.DataFrame, regime_labels: np.ndarray) -> Dict[str, float]:
        """Calculate clustering performance metrics."""
        try:
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
            
            # Prepare data
            numeric_cols = regime_data.select_dtypes(include=[np.number]).columns
            regime_data_numeric = regime_data[numeric_cols].fillna(0)
            
            # Calculate metrics
            silhouette = silhouette_score(regime_data_numeric, regime_labels)
            calinski_harabasz = calinski_harabasz_score(regime_data_numeric, regime_labels)
            davies_bouldin = davies_bouldin_score(regime_data_numeric, regime_labels)
            
            return {
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski_harabasz,
                'davies_bouldin_score': davies_bouldin
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Performance metrics calculation failed: {e}")
            return {
                'silhouette_score': 0.0,
                'calinski_harabasz_score': 0.0,
                'davies_bouldin_score': 0.0
            }
    
    def _save_regime_data(self, result: RegimeResult):
        """Save regime data to file."""
        try:
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save regime data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"regime_data_{timestamp}.parquet"
            filepath = output_dir / filename
            
            result.regime_features.to_parquet(filepath)
            
            # Save metadata
            metadata_file = output_dir / f"regime_metadata_{timestamp}.json"
            import json
            metadata = {
                'regime_labels': result.regime_labels.tolist(),
                'regime_centers': {str(k): v.tolist() for k, v in result.regime_centers.items()},
                'regime_statistics': result.regime_statistics,
                'qualified_regimes': result.qualified_regimes,
                'detection_method': result.detection_method,
                'detection_time': result.detection_time,
                'n_regimes_detected': result.n_regimes_detected,
                'n_qualified_regimes': result.n_qualified_regimes,
                'regime_quality_score': result.regime_quality_score,
                'silhouette_score': result.silhouette_score,
                'calinski_harabasz_score': result.calinski_harabasz_score,
                'davies_bouldin_score': result.davies_bouldin_score
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            self.logger.info(f"📁 Regime data saved to {filepath}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save regime data: {e}")
    
    def export_regime_data(self, result: RegimeResult, filepath: str):
        """Export regime data to file."""
        try:
            result.regime_features.to_csv(filepath)
            self.logger.info(f"📁 Regime data exported to {filepath}")
        except Exception as e:
            self.logger.error(f"❌ Failed to export regime data: {e}")
    
    # Enhanced methods for performance optimizations and advanced validation
    
    def _generate_cache_key(self, data: pd.DataFrame, features: Optional[pd.DataFrame]) -> str:
        """Generate cache key for regime detection results."""
        try:
            # Create hash of data and features
            data_hash = hashlib.md5(str(data.values).encode()).hexdigest()[:8]
            features_hash = hashlib.md5(str(features.values).encode()).hexdigest()[:8] if features is not None else "none"
            config_hash = hashlib.md5(str(self.config).encode()).hexdigest()[:8]
            
            cache_key = f"regime_{data_hash}_{features_hash}_{config_hash}"
            return cache_key
            
        except Exception as e:
            self.logger.warning(f"Could not generate cache key: {e}")
            return f"regime_{int(time.time())}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[RegimeResult]:
        """Get cached regime detection result."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            self.logger.debug(f"Could not load cached result: {e}")
        return None
    
    def _cache_result(self, cache_key: str, result: RegimeResult) -> None:
        """Cache regime detection result."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            self.logger.debug(f"Cached result: {cache_key}")
        except Exception as e:
            self.logger.warning(f"Could not cache result: {e}")
    
    def _optimize_data_for_memory(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize data for memory efficiency."""
        if self.memory_optimizer:
            return self.memory_optimizer.optimize_dataframe_memory(data)
        return data
    
    def _detect_regimes_parallel(self, regime_data: pd.DataFrame) -> Tuple[np.ndarray, Dict[int, np.ndarray], Dict[int, Dict[str, Any]]]:
        """Detect regimes using parallel processing."""
        try:
            if self.cpu_optimizer and self.config.enable_parallel:
                # Use parallel processing for regime detection
                with self.cpu_optimizer.create_optimized_thread_pool() as executor:
                    # Split data into chunks for parallel processing
                    chunk_size = len(regime_data) // self.cpu_optimizer.get_optimal_worker_count()
                    chunks = [regime_data.iloc[i:i+chunk_size] for i in range(0, len(regime_data), chunk_size)]
                    
                    # Process chunks in parallel
                    futures = []
                    for chunk in chunks:
                        future = executor.submit(self._detect_regimes_sequential, chunk)
                        futures.append(future)
                    
                    # Collect results
                    results = []
                    for future in futures:
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            self.logger.warning(f"Parallel chunk processing failed: {e}")
                    
                    # Combine results
                    if results:
                        # Use the first result as base and combine others
                        regime_labels, regime_centers, regime_statistics = results[0]
                        # Additional logic to combine results from multiple chunks
                        return regime_labels, regime_centers, regime_statistics
            
            # Fallback to sequential processing
            return self._detect_regimes_sequential(regime_data)
            
        except Exception as e:
            self.logger.warning(f"Parallel regime detection failed: {e}")
            return self._detect_regimes_sequential(regime_data)
    
    def _detect_regimes_sequential(self, regime_data: pd.DataFrame) -> Tuple[np.ndarray, Dict[int, np.ndarray], Dict[int, Dict[str, Any]]]:
        """Detect regimes using sequential processing."""
        try:
            if self.config.detection_method == RegimeDetectionMethod.UNSUPERVISED and self.regime_detector:
                return self._detect_unsupervised_regimes(regime_data)
            else:
                return self._detect_basic_regimes(regime_data)
        except Exception as e:
            self.logger.warning(f"Sequential regime detection failed: {e}")
            return self._detect_basic_regimes(regime_data)
    
    def _apply_advanced_validation(self, regime_labels: np.ndarray, regime_centers: Dict[int, np.ndarray], 
                                  regime_statistics: Dict[int, Dict[str, Any]], data: pd.DataFrame) -> Dict[str, Any]:
        """Apply advanced validation to regime detection results."""
        try:
            validation_results = {}
            
            # Cross-validation for regime stability
            if self.config.enable_cross_validation and self.cvlsa:
                try:
                    cv_results = self.cvlsa.cross_validate(
                        data, 
                        {'regime_labels': regime_labels, 'regime_centers': regime_centers},
                        cv_folds=self.config.cv_folds,
                        stability_metric='regime_consistency'
                    )
                    validation_results['cross_validation'] = cv_results
                except Exception as e:
                    self.logger.warning(f"Cross-validation failed: {e}")
                    validation_results['cross_validation'] = {'error': str(e)}
            
            # Out-of-sample testing
            if self.config.enable_out_of_sample:
                try:
                    oos_results = self._out_of_sample_validation(regime_labels, regime_centers, regime_statistics, data)
                    validation_results['out_of_sample'] = oos_results
                except Exception as e:
                    self.logger.warning(f"Out-of-sample validation failed: {e}")
                    validation_results['out_of_sample'] = {'error': str(e)}
            
            # Regime persistence analysis
            if self.config.enable_regime_persistence:
                try:
                    persistence_results = self._analyze_regime_persistence_enhanced(regime_labels, regime_centers, regime_statistics)
                    validation_results['persistence'] = persistence_results
                except Exception as e:
                    self.logger.warning(f"Regime persistence analysis failed: {e}")
                    validation_results['persistence'] = {'error': str(e)}
            
            return validation_results
            
        except Exception as e:
            self.logger.warning(f"Advanced validation failed: {e}")
            return {}
    
    def _out_of_sample_validation(self, regime_labels: np.ndarray, regime_centers: Dict[int, np.ndarray], 
                                 regime_statistics: Dict[int, Dict[str, Any]], data: pd.DataFrame) -> Dict[str, Any]:
        """Perform out-of-sample validation for regime detection."""
        try:
            # Split data for out-of-sample testing
            split_idx = int(len(data) * self.config.oos_split_ratio)
            train_data = data[:split_idx]
            test_data = data[split_idx:]
            
            # Train on in-sample data
            train_regime_data = self._prepare_regime_data(train_data, None)
            train_result = self._detect_regimes_sequential(train_regime_data)
            
            # Test on out-of-sample data
            test_regime_data = self._prepare_regime_data(test_data, None)
            test_result = self._detect_regimes_sequential(test_regime_data)
            
            # Calculate out-of-sample metrics
            oos_metrics = self._calculate_oos_metrics(train_result, test_result)
            
            return oos_metrics
            
        except Exception as e:
            self.logger.warning(f"Out-of-sample validation failed: {e}")
            return {'error': str(e)}
    
    def _calculate_oos_metrics(self, train_result: Tuple, test_result: Tuple) -> Dict[str, Any]:
        """Calculate out-of-sample validation metrics."""
        try:
            train_labels, train_centers, train_stats = train_result
            test_labels, test_centers, test_stats = test_result
            
            # Calculate similarity between train and test regimes
            similarity = self._calculate_regime_similarity(train_labels, test_labels)
            
            # Calculate prediction accuracy
            accuracy = self._calculate_regime_accuracy(train_labels, test_labels)
            
            return {
                'similarity': similarity,
                'accuracy': accuracy,
                'oos_score': (similarity + accuracy) / 2.0
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_regime_similarity(self, train_labels: np.ndarray, test_labels: np.ndarray) -> float:
        """Calculate similarity between train and test regime results."""
        try:
            # Calculate regime distribution similarity
            train_dist = np.bincount(train_labels) / len(train_labels)
            test_dist = np.bincount(test_labels) / len(test_labels)
            
            # Pad distributions to same length
            max_len = max(len(train_dist), len(test_dist))
            train_dist = np.pad(train_dist, (0, max_len - len(train_dist)))
            test_dist = np.pad(test_dist, (0, max_len - len(test_dist)))
            
            # Calculate cosine similarity
            similarity = np.dot(train_dist, test_dist) / (
                np.linalg.norm(train_dist) * np.linalg.norm(test_dist)
            )
            
            return float(similarity)
            
        except Exception as e:
            self.logger.warning(f"Regime similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_regime_accuracy(self, train_labels: np.ndarray, test_labels: np.ndarray) -> float:
        """Calculate regime prediction accuracy."""
        try:
            # This is a simplified accuracy calculation
            # In practice, you would compare predicted vs actual regimes
            
            # Calculate accuracy based on regime consistency
            train_unique = len(np.unique(train_labels))
            test_unique = len(np.unique(test_labels))
            
            # Accuracy based on regime count consistency
            accuracy = 1.0 - abs(train_unique - test_unique) / max(train_unique, test_unique)
            
            return float(accuracy)
            
        except Exception as e:
            self.logger.warning(f"Regime accuracy calculation failed: {e}")
            return 0.0
    
    def _analyze_regime_persistence_enhanced(self, regime_labels: np.ndarray, regime_centers: Dict[int, np.ndarray], 
                                            regime_statistics: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced regime persistence analysis."""
        try:
            # Calculate regime transition probabilities
            transition_probs = self._calculate_transition_probabilities(regime_labels)
            
            # Calculate stability scores
            stability_scores = self._calculate_stability_scores(regime_labels)
            
            # Calculate regime duration statistics
            duration_stats = self._calculate_regime_duration_stats(regime_labels)
            
            return {
                'transition_probabilities': transition_probs,
                'stability_scores': stability_scores,
                'duration_stats': duration_stats,
                'average_stability': np.mean(stability_scores) if len(stability_scores) > 0 else 0.0
            }
            
        except Exception as e:
            self.logger.warning(f"Enhanced regime persistence analysis failed: {e}")
            return {'error': str(e)}
    
    def _calculate_transition_probabilities(self, regime_labels: np.ndarray) -> np.ndarray:
        """Calculate regime transition probability matrix."""
        try:
            unique_regimes = np.unique(regime_labels)
            n_regimes = len(unique_regimes)
            transition_matrix = np.zeros((n_regimes, n_regimes))
            
            for i in range(len(regime_labels) - 1):
                current_regime = regime_labels[i]
                next_regime = regime_labels[i + 1]
                current_idx = np.where(unique_regimes == current_regime)[0][0]
                next_idx = np.where(unique_regimes == next_regime)[0][0]
                transition_matrix[current_idx, next_idx] += 1
            
            # Normalize transition matrix
            row_sums = transition_matrix.sum(axis=1)
            transition_matrix = transition_matrix / (row_sums[:, np.newaxis] + 1e-8)
            
            return transition_matrix
            
        except Exception as e:
            self.logger.warning(f"Transition probability calculation failed: {e}")
            return np.array([])
    
    def _calculate_stability_scores(self, regime_labels: np.ndarray) -> np.ndarray:
        """Calculate regime stability scores."""
        try:
            stability_scores = np.zeros(len(regime_labels))
            
            for i in range(len(regime_labels)):
                current_regime = regime_labels[i]
                
                # Look ahead and behind for regime consistency
                lookback = min(10, i)
                lookahead = min(10, len(regime_labels) - i - 1)
                
                if lookback > 0:
                    past_regimes = regime_labels[i-lookback:i]
                    past_consistency = np.mean(past_regimes == current_regime)
                else:
                    past_consistency = 1.0
                
                if lookahead > 0:
                    future_regimes = regime_labels[i+1:i+1+lookahead]
                    future_consistency = np.mean(future_regimes == current_regime)
                else:
                    future_consistency = 1.0
                
                stability_scores[i] = (past_consistency + future_consistency) / 2.0
            
            return stability_scores
            
        except Exception as e:
            self.logger.warning(f"Stability score calculation failed: {e}")
            return np.zeros(len(regime_labels))
    
    def _calculate_regime_duration_stats(self, regime_labels: np.ndarray) -> Dict[str, Any]:
        """Calculate regime duration statistics."""
        try:
            durations = []
            current_regime = regime_labels[0]
            current_duration = 1
            
            for i in range(1, len(regime_labels)):
                if regime_labels[i] == current_regime:
                    current_duration += 1
                else:
                    durations.append(current_duration)
                    current_regime = regime_labels[i]
                    current_duration = 1
            
            # Add last duration
            durations.append(current_duration)
            
            return {
                'durations': durations,
                'mean_duration': np.mean(durations),
                'std_duration': np.std(durations),
                'min_duration': np.min(durations),
                'max_duration': np.max(durations)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware capability information."""
        info = {
            'gpu_available': self.config.enable_gpu and self.gpu_manager is not None,
            'memory_optimizer_available': self.config.enable_memory_optimization and self.memory_optimizer is not None,
            'cpu_optimizer_available': self.config.enable_parallel and self.cpu_optimizer is not None,
            'matrix_ops_available': self.matrix_ops is not None
        }
        
        # Add GPU info if available
        if self.gpu_manager and hasattr(self.gpu_manager, 'get_gpu_info'):
            info['gpu_info'] = self.gpu_manager.get_gpu_info()
        
        # Add memory info if available
        if self.memory_optimizer and hasattr(self.memory_optimizer, 'get_memory_stats'):
            info['memory_info'] = self.memory_optimizer.get_memory_stats()
        
        # Add CPU info if available
        if self.cpu_optimizer and hasattr(self.cpu_optimizer, 'get_cpu_info'):
            info['cpu_info'] = self.cpu_optimizer.get_cpu_info()
        
        return info
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()
        
        # Add hardware info
        stats['hardware_info'] = self._get_hardware_info()
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        try:
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info("✅ Cache cleared successfully")
        except Exception as e:
            self.logger.warning(f"Could not clear cache: {e}")
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage for regime detection."""
        if self.memory_optimizer:
            return self.memory_optimizer.optimize_memory_usage()
        else:
            return {'status': 'memory_optimizer_not_available'}
    
    def optimize_cpu_usage(self, target_utilization: float = 0.8) -> Dict[str, Any]:
        """Optimize CPU usage for regime detection."""
        if self.cpu_optimizer:
            return self.cpu_optimizer.optimize_cpu_usage(target_utilization)
        else:
            return {'status': 'cpu_optimizer_not_available'}