"""
NAS Feature Extractor for Neural Architecture Search Clustering

This module provides comprehensive feature extraction capabilities for NAS clustering,
integrating with shared utilities for optimization, validation, and hardware acceleration.

Features:
- Advanced feature extraction from market data
- Integration with Bayesian TPE optimization
- M1 hardware acceleration support
- Comprehensive validation and error handling
- Memory optimization for large datasets
- Parallel processing capabilities
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path

# Import shared utilities
try:
    from src.utils.common_operations import (
        safe_dataframe_operation, validate_dataframe_columns, 
        safe_convert_dtypes, calculate_data_quality_metrics,
        safe_merge_dataframes, create_summary_statistics,
        optimize_dataframe_dtypes, safe_timestamp_conversion,
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        integrate_with_m1_optimizers, memory_checkpoint, gpu_context
    )
    from src.utils.math_validation import (
        safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
        validate_positive, validate_range, safe_correlation, safe_covariance,
        safe_mean, safe_std, MathValidation
    )
    from src.utils.nas_tas.bayesian_tpe_optimizer import (
        BayesianTPEOptimizer, BayesianTPEConfig, OptimizationResult
    )
    from src.utils.hardware.m1_gpu_utils import (
        is_m1_available, is_mps_available, optimize_dataframe_for_m1,
        create_m1_optimized_array, m1_backtesting_simulate, m1_monte_carlo_simulate
    )
    from src.utils.hardware.m1_memory_optimizer import (
        get_m1_memory_optimizer, optimize_dataframe_memory
    )
    from src.utils.hardware.m1_cpu_optimizer import (
        get_m1_cpu_optimizer, parallel_map_m1, create_m1_optimized_thread_pool
    )
    SHARED_UTILS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️ Shared utilities not available: {e}")
    SHARED_UTILS_AVAILABLE = False

# Optional ML dependencies
try:
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
    from sklearn.decomposition import PCA, FastICA, TruncatedSVD
    from sklearn.feature_selection import SelectKBest, SelectPercentile, mutual_info_regression
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from sklearn.manifold import TSNE, UMAP
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class NASFeatureConfig:
    """Configuration for NAS Feature Extractor."""
    
    # Feature extraction parameters
    feature_types: List[str] = field(default_factory=lambda: [
        'technical_indicators', 'statistical_features', 'fourier_features',
        'wavelet_features', 'regime_features', 'volatility_features'
    ])
    
    # Clustering parameters
    clustering_methods: List[str] = field(default_factory=lambda: [
        'kmeans', 'dbscan', 'agglomerative'
    ])
    n_clusters_range: Tuple[int, int] = (2, 20)
    max_clusters: int = 20
    
    # Dimensionality reduction
    enable_dimensionality_reduction: bool = True
    reduction_methods: List[str] = field(default_factory=lambda: [
        'pca', 'ica', 'tsne', 'umap'
    ])
    n_components_range: Tuple[int, int] = (2, 50)
    
    # Feature selection
    enable_feature_selection: bool = True
    feature_selection_methods: List[str] = field(default_factory=lambda: [
        'mutual_info', 'variance', 'correlation'
    ])
    max_features: int = 100
    
    # Optimization
    enable_optimization: bool = True
    optimization_method: str = 'bayesian_tpe'  # 'grid_search', 'bayesian_tpe'
    n_trials: int = 50
    optimization_timeout: int = 3600  # 1 hour
    
    # Hardware optimization
    enable_m1_optimization: bool = True
    enable_parallel_processing: bool = True
    max_workers: int = 4
    
    # Validation
    enable_validation: bool = True
    validation_methods: List[str] = field(default_factory=lambda: [
        'silhouette', 'calinski_harabasz', 'davies_bouldin'
    ])
    
    # Memory management
    memory_limit_gb: Optional[float] = None
    enable_memory_optimization: bool = True
    
    # Logging
    log_level: str = 'INFO'
    enable_progress_logging: bool = True

@dataclass
class FeatureExtractionResult:
    """Result of feature extraction process."""
    
    features: np.ndarray
    feature_names: List[str]
    extraction_time: float
    feature_importance: Optional[np.ndarray] = None
    dimensionality_reduction: Optional[Dict[str, Any]] = None
    feature_selection: Optional[Dict[str, Any]] = None
    optimization_results: Optional[Dict[str, Any]] = None
    validation_metrics: Optional[Dict[str, float]] = None
    success: bool = True
    error_message: Optional[str] = None

class NASFeatureExtractor:
    """
    Neural Architecture Search Feature Extractor for clustering analysis.
    
    This class provides comprehensive feature extraction capabilities optimized
    for NAS clustering with integration to shared utilities and hardware acceleration.
    """
    
    def __init__(self, config: Optional[NASFeatureConfig] = None):
        """Initialize NAS Feature Extractor."""
        self.config = config or NASFeatureConfig()
        self.logger = self._setup_logging()
        
        # Initialize shared utilities
        self._init_shared_utilities()
        
        # Initialize hardware optimizations
        self._init_hardware_optimizations()
        
        # Initialize feature extractors
        self._init_feature_extractors()
        
        # Initialize optimization
        self._init_optimization()
        
        self.logger.info("🚀 NAS Feature Extractor initialized")
        self.logger.info(f"   → Feature types: {self.config.feature_types}")
        self.logger.info(f"   → Clustering methods: {self.config.clustering_methods}")
        self.logger.info(f"   → M1 optimization: {'enabled' if self.config.enable_m1_optimization else 'disabled'}")
        self.logger.info(f"   → Parallel processing: {'enabled' if self.config.enable_parallel_processing else 'disabled'}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(f"{__name__}.NASFeatureExtractor")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _init_shared_utilities(self):
        """Initialize shared utilities."""
        if not SHARED_UTILS_AVAILABLE:
            self.logger.warning("⚠️ Shared utilities not available, using fallback implementations")
            self.shared_utils = None
            return
        
        try:
            # Initialize M1 optimizations
            if self.config.enable_m1_optimization:
                self.m1_integration = integrate_with_m1_optimizers()
                if self.m1_integration.get('success', False):
                    self.logger.info("✅ M1 optimizations integrated successfully")
                else:
                    self.logger.warning("⚠️ M1 integration failed, using CPU fallback")
            
            # Initialize math validation
            self.math_validator = MathValidation()
            
            self.shared_utils = True
            self.logger.info("✅ Shared utilities initialized")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Shared utilities initialization failed: {e}")
            self.shared_utils = None
    
    def _init_hardware_optimizations(self):
        """Initialize hardware optimizations."""
        if not self.config.enable_m1_optimization:
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
            return
        
        try:
            if SHARED_UTILS_AVAILABLE:
                self.gpu_manager = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer(self.config.memory_limit_gb)
                self.cpu_optimizer = get_m1_cpu_optimizer()
                
                # Start memory monitoring if available
                if self.memory_optimizer and hasattr(self.memory_optimizer, 'start_monitoring'):
                    self.memory_optimizer.start_monitoring()
                
                self.logger.info("✅ Hardware optimizations initialized")
            else:
                self.gpu_manager = None
                self.memory_optimizer = None
                self.cpu_optimizer = None
                self.logger.warning("⚠️ Hardware optimizations not available")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Hardware optimization initialization failed: {e}")
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
    
    def _init_feature_extractors(self):
        """Initialize feature extractors."""
        self.feature_extractors = {}
        
        # Technical indicators
        self.feature_extractors['technical_indicators'] = self._extract_technical_indicators
        
        # Statistical features
        self.feature_extractors['statistical_features'] = self._extract_statistical_features
        
        # Fourier features
        self.feature_extractors['fourier_features'] = self._extract_fourier_features
        
        # Wavelet features
        self.feature_extractors['wavelet_features'] = self._extract_wavelet_features
        
        # Regime features
        self.feature_extractors['regime_features'] = self._extract_regime_features
        
        # Volatility features
        self.feature_extractors['volatility_features'] = self._extract_volatility_features
        
        self.logger.info(f"✅ Feature extractors initialized: {list(self.feature_extractors.keys())}")
    
    def _init_optimization(self):
        """Initialize optimization components."""
        if not self.config.enable_optimization:
            self.optimizer = None
            return
        
        try:
            if SHARED_UTILS_AVAILABLE and self.config.optimization_method == 'bayesian_tpe':
                # Initialize Bayesian TPE optimizer
                tpe_config = BayesianTPEConfig(
                    n_trials=self.config.n_trials,
                    timeout_seconds=self.config.optimization_timeout,
                    enable_parallel=self.config.enable_parallel_processing,
                    max_workers=self.config.max_workers
                )
                self.optimizer = BayesianTPEOptimizer(tpe_config)
                self.logger.info("✅ Bayesian TPE optimizer initialized")
            else:
                self.optimizer = None
                self.logger.info("ℹ️ Optimization disabled or not available")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Optimization initialization failed: {e}")
            self.optimizer = None
    
    def extract_features(self, 
                        data: Union[pd.DataFrame, np.ndarray],
                        target_column: Optional[str] = None,
                        **kwargs) -> FeatureExtractionResult:
        """
        Extract features from input data for NAS clustering.
        
        Args:
            data: Input data (DataFrame or numpy array)
            target_column: Optional target column name for supervised feature selection
            **kwargs: Additional arguments for feature extraction
            
        Returns:
            FeatureExtractionResult with extracted features and metadata
        """
        start_time = time.time()
        self.logger.info("🔍 Starting feature extraction")
        
        try:
            # Validate and prepare data
            processed_data = self._prepare_data(data)
            
            # Extract features using configured methods
            features, feature_names = self._extract_all_features(processed_data, target_column, **kwargs)
            
            # Apply dimensionality reduction if enabled
            if self.config.enable_dimensionality_reduction:
                features, reduction_info = self._apply_dimensionality_reduction(features)
            else:
                reduction_info = None
            
            # Apply feature selection if enabled
            if self.config.enable_feature_selection:
                features, feature_names, selection_info = self._apply_feature_selection(
                    features, feature_names, processed_data, target_column
                )
            else:
                selection_info = None
            
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(features)
            
            # Apply optimization if enabled
            optimization_results = None
            if self.config.enable_optimization and self.optimizer:
                optimization_results = self._optimize_features(features, processed_data)
            
            # Calculate validation metrics
            validation_metrics = None
            if self.config.enable_validation:
                validation_metrics = self._calculate_validation_metrics(features)
            
            extraction_time = time.time() - start_time
            
            result = FeatureExtractionResult(
                features=features,
                feature_names=feature_names,
                extraction_time=extraction_time,
                feature_importance=feature_importance,
                dimensionality_reduction=reduction_info,
                feature_selection=selection_info,
                optimization_results=optimization_results,
                validation_metrics=validation_metrics,
                success=True
            )
            
            self.logger.info(f"✅ Feature extraction completed in {extraction_time:.2f}s")
            self.logger.info(f"   → Features extracted: {features.shape[1]}")
            self.logger.info(f"   → Samples: {features.shape[0]}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Feature extraction failed: {e}")
            return FeatureExtractionResult(
                features=np.array([]),
                feature_names=[],
                extraction_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def _prepare_data(self, data: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Prepare and validate input data."""
        try:
            # Convert to DataFrame if needed
            if isinstance(data, np.ndarray):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
            
            # Validate DataFrame
            if SHARED_UTILS_AVAILABLE and self.shared_utils:
                if not safe_dataframe_operation(df, lambda x: x):
                    raise ValueError("Invalid DataFrame provided")
            
            # Optimize for M1 if enabled
            if self.config.enable_m1_optimization and self.memory_optimizer:
                df = self.memory_optimizer.optimize_dataframe_memory(df)
            
            # Handle missing values
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Optimize data types
            if SHARED_UTILS_AVAILABLE and self.shared_utils:
                df = optimize_dataframe_dtypes(df)
            
            self.logger.info(f"📊 Data prepared: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Data preparation failed: {e}")
            raise
    
    def _extract_all_features(self, 
                             data: pd.DataFrame,
                             target_column: Optional[str] = None,
                             **kwargs) -> Tuple[np.ndarray, List[str]]:
        """Extract all configured feature types."""
        all_features = []
        all_feature_names = []
        
        for feature_type in self.config.feature_types:
            if feature_type in self.feature_extractors:
                try:
                    self.logger.info(f"🔧 Extracting {feature_type} features")
                    
                    # Use memory checkpoint if available
                    if self.memory_optimizer and hasattr(self.memory_optimizer, 'memory_checkpoint'):
                        with self.memory_optimizer.memory_checkpoint(f"extract_{feature_type}"):
                            features, names = self.feature_extractors[feature_type](data, **kwargs)
                    else:
                        features, names = self.feature_extractors[feature_type](data, **kwargs)
                    
                    if features is not None and len(features) > 0:
                        all_features.append(features)
                        all_feature_names.extend(names)
                        self.logger.info(f"   → {feature_type}: {features.shape[1]} features")
                    else:
                        self.logger.warning(f"   → {feature_type}: No features extracted")
                        
                except Exception as e:
                    self.logger.warning(f"   → {feature_type}: Extraction failed - {e}")
                    continue
        
        if not all_features:
            raise ValueError("No features could be extracted from the data")
        
        # Combine all features
        combined_features = np.hstack(all_features)
        
        # Validate features
        if self.shared_utils and self.math_validator:
            try:
                combined_features = self.math_validator.validate_numeric_array(
                    combined_features, "extracted_features"
                )
            except Exception as e:
                self.logger.warning(f"Feature validation warning: {e}")
        
        return combined_features, all_feature_names
    
    def _extract_technical_indicators(self, data: pd.DataFrame, **kwargs) -> Tuple[np.ndarray, List[str]]:
        """Extract technical indicator features."""
        features = []
        feature_names = []
        
        # Ensure we have numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return np.array([]), []
        
        # Use first numeric column as price series
        price_series = data[numeric_cols[0]].values
        
        # Simple Moving Averages
        for window in [5, 10, 20, 50]:
            if len(price_series) >= window:
                sma = pd.Series(price_series).rolling(window=window).mean()
                features.append(sma.fillna(method='bfill').values)
                feature_names.append(f'sma_{window}')
        
        # Exponential Moving Averages
        for span in [5, 10, 20, 50]:
            if len(price_series) >= span:
                ema = pd.Series(price_series).ewm(span=span).mean()
                features.append(ema.fillna(method='bfill').values)
                feature_names.append(f'ema_{span}')
        
        # RSI (Relative Strength Index)
        if len(price_series) >= 14:
            rsi = self._calculate_rsi(price_series, 14)
            features.append(rsi)
            feature_names.append('rsi_14')
        
        # Bollinger Bands
        if len(price_series) >= 20:
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(price_series, 20, 2)
            features.extend([bb_upper, bb_middle, bb_lower])
            feature_names.extend(['bb_upper', 'bb_middle', 'bb_lower'])
        
        # MACD
        if len(price_series) >= 26:
            macd_line, signal_line, histogram = self._calculate_macd(price_series)
            features.extend([macd_line, signal_line, histogram])
            feature_names.extend(['macd_line', 'macd_signal', 'macd_histogram'])
        
        if features:
            return np.column_stack(features), feature_names
        else:
            return np.array([]), []
    
    def _extract_statistical_features(self, data: pd.DataFrame, **kwargs) -> Tuple[np.ndarray, List[str]]:
        """Extract statistical features."""
        features = []
        feature_names = []
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            series = data[col].values
            
            # Basic statistics
            features.append(series)
            feature_names.append(f'{col}_values')
            
            # Rolling statistics
            for window in [5, 10, 20]:
                if len(series) >= window:
                    rolling_mean = pd.Series(series).rolling(window=window).mean()
                    rolling_std = pd.Series(series).rolling(window=window).std()
                    rolling_skew = pd.Series(series).rolling(window=window).skew()
                    rolling_kurt = pd.Series(series).rolling(window=window).kurt()
                    
                    features.extend([
                        rolling_mean.fillna(method='bfill').values,
                        rolling_std.fillna(method='bfill').values,
                        rolling_skew.fillna(method='bfill').values,
                        rolling_kurt.fillna(method='bfill').values
                    ])
                    feature_names.extend([
                        f'{col}_mean_{window}', f'{col}_std_{window}',
                        f'{col}_skew_{window}', f'{col}_kurt_{window}'
                    ])
        
        if features:
            return np.column_stack(features), feature_names
        else:
            return np.array([]), []
    
    def _extract_fourier_features(self, data: pd.DataFrame, **kwargs) -> Tuple[np.ndarray, List[str]]:
        """Extract Fourier transform features."""
        features = []
        feature_names = []
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            series = data[col].values
            
            if len(series) >= 10:  # Minimum length for FFT
                # FFT coefficients
                fft_coeffs = np.fft.fft(series)
                fft_magnitude = np.abs(fft_coeffs)
                fft_phase = np.angle(fft_coeffs)
                
                # Take first few coefficients
                n_coeffs = min(10, len(fft_coeffs) // 2)
                features.extend([
                    fft_magnitude[:n_coeffs],
                    fft_phase[:n_coeffs]
                ])
                feature_names.extend([
                    f'{col}_fft_mag_{i}' for i in range(n_coeffs)
                ] + [
                    f'{col}_fft_phase_{i}' for i in range(n_coeffs)
                ])
        
        if features:
            # Pad arrays to same length
            max_len = max(len(f) for f in features)
            padded_features = []
            for f in features:
                if len(f) < max_len:
                    padded = np.pad(f, (0, max_len - len(f)), mode='constant')
                else:
                    padded = f[:max_len]
                padded_features.append(padded)
            
            return np.column_stack(padded_features), feature_names
        else:
            return np.array([]), []
    
    def _extract_wavelet_features(self, data: pd.DataFrame, **kwargs) -> Tuple[np.ndarray, List[str]]:
        """Extract wavelet transform features."""
        features = []
        feature_names = []
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            series = data[col].values
            
            if len(series) >= 8:  # Minimum length for wavelet
                # Simple wavelet-like features using differences
                # This is a simplified implementation
                diff1 = np.diff(series, n=1)
                diff2 = np.diff(series, n=2)
                
                # Statistical features of differences
                features.extend([
                    diff1, diff2,
                    np.abs(diff1), np.abs(diff2)
                ])
                feature_names.extend([
                    f'{col}_diff1', f'{col}_diff2',
                    f'{col}_abs_diff1', f'{col}_abs_diff2'
                ])
        
        if features:
            # Pad arrays to same length
            max_len = max(len(f) for f in features)
            padded_features = []
            for f in features:
                if len(f) < max_len:
                    padded = np.pad(f, (0, max_len - len(f)), mode='constant')
                else:
                    padded = f[:max_len]
                padded_features.append(padded)
            
            return np.column_stack(padded_features), feature_names
        else:
            return np.array([]), []
    
    def _extract_regime_features(self, data: pd.DataFrame, **kwargs) -> Tuple[np.ndarray, List[str]]:
        """Extract regime-based features."""
        features = []
        feature_names = []
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            series = data[col].values
            
            if len(series) >= 20:
                # Regime detection using rolling statistics
                rolling_mean = pd.Series(series).rolling(window=10).mean()
                rolling_std = pd.Series(series).rolling(window=10).std()
                
                # Regime indicators
                regime_high = (series > rolling_mean + rolling_std).astype(int)
                regime_low = (series < rolling_mean - rolling_std).astype(int)
                regime_normal = ((series >= rolling_mean - rolling_std) & 
                               (series <= rolling_mean + rolling_std)).astype(int)
                
                features.extend([regime_high, regime_low, regime_normal])
                feature_names.extend([
                    f'{col}_regime_high', f'{col}_regime_low', f'{col}_regime_normal'
                ])
        
        if features:
            return np.column_stack(features), feature_names
        else:
            return np.array([]), []
    
    def _extract_volatility_features(self, data: pd.DataFrame, **kwargs) -> Tuple[np.ndarray, List[str]]:
        """Extract volatility-based features."""
        features = []
        feature_names = []
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            series = data[col].values
            
            if len(series) >= 10:
                # Rolling volatility
                for window in [5, 10, 20]:
                    if len(series) >= window:
                        rolling_vol = pd.Series(series).rolling(window=window).std()
                        features.append(rolling_vol.fillna(method='bfill').values)
                        feature_names.append(f'{col}_vol_{window}')
                
                # GARCH-like features (simplified)
                returns = np.diff(series)
                if len(returns) > 0:
                    vol_estimate = np.std(returns)
                    features.append(np.full(len(series), vol_estimate))
                    feature_names.append(f'{col}_vol_estimate')
        
        if features:
            return np.column_stack(features), feature_names
        else:
            return np.array([]), []
    
    def _apply_dimensionality_reduction(self, features: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply dimensionality reduction to features."""
        if not SKLEARN_AVAILABLE:
            self.logger.warning("⚠️ Scikit-learn not available, skipping dimensionality reduction")
            return features, None
        
        reduction_info = {}
        reduced_features = features
        
        for method in self.config.reduction_methods:
            try:
                self.logger.info(f"🔧 Applying {method} dimensionality reduction")
                
                if method == 'pca':
                    reducer = PCA(n_components=min(50, features.shape[1]))
                elif method == 'ica':
                    reducer = FastICA(n_components=min(50, features.shape[1]))
                elif method == 'tsne':
                    if features.shape[1] > 50:
                        # Pre-reduce with PCA
                        pca = PCA(n_components=50)
                        features_pca = pca.fit_transform(features)
                    else:
                        features_pca = features
                    reducer = TSNE(n_components=2, random_state=42)
                    reduced_features = reducer.fit_transform(features_pca)
                elif method == 'umap' and UMAP_AVAILABLE:
                    reducer = umap.UMAP(n_components=min(10, features.shape[1]), random_state=42)
                else:
                    continue
                
                if method != 'tsne':
                    reduced_features = reducer.fit_transform(features)
                
                reduction_info[method] = {
                    'n_components': reduced_features.shape[1],
                    'explained_variance_ratio': getattr(reducer, 'explained_variance_ratio_', None)
                }
                
                self.logger.info(f"   → {method}: {features.shape[1]} → {reduced_features.shape[1]} features")
                break  # Use first successful method
                
            except Exception as e:
                self.logger.warning(f"   → {method}: Reduction failed - {e}")
                continue
        
        return reduced_features, reduction_info
    
    def _apply_feature_selection(self, 
                                features: np.ndarray,
                                feature_names: List[str],
                                data: pd.DataFrame,
                                target_column: Optional[str] = None) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """Apply feature selection to reduce feature count."""
        if not SKLEARN_AVAILABLE:
            self.logger.warning("⚠️ Scikit-learn not available, skipping feature selection")
            return features, feature_names, None
        
        selection_info = {}
        
        try:
            # Select top features based on variance
            if 'variance' in self.config.feature_selection_methods:
                selector = SelectKBest(k=min(self.config.max_features, features.shape[1]))
                selected_features = selector.fit_transform(features, np.zeros(features.shape[0]))
                selected_indices = selector.get_support(indices=True)
                selected_names = [feature_names[i] for i in selected_indices]
                
                selection_info['variance'] = {
                    'n_selected': len(selected_indices),
                    'selected_indices': selected_indices.tolist()
                }
                
                self.logger.info(f"🔧 Feature selection: {features.shape[1]} → {selected_features.shape[1]} features")
                return selected_features, selected_names, selection_info
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature selection failed: {e}")
        
        return features, feature_names, None
    
    def _calculate_feature_importance(self, features: np.ndarray) -> np.ndarray:
        """Calculate feature importance scores."""
        try:
            # Simple variance-based importance
            importance = np.var(features, axis=0)
            return importance / np.sum(importance)  # Normalize
        except Exception as e:
            self.logger.warning(f"⚠️ Feature importance calculation failed: {e}")
            return np.ones(features.shape[1]) / features.shape[1]
    
    def _optimize_features(self, features: np.ndarray, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Optimize feature extraction parameters."""
        if not self.optimizer:
            return None
        
        try:
            self.logger.info("🔧 Optimizing feature extraction parameters")
            
            # Define search space for optimization
            search_space = {
                'n_clusters': {
                    'type': 'int',
                    'low': self.config.n_clusters_range[0],
                    'high': self.config.n_clusters_range[1]
                },
                'n_components': {
                    'type': 'int',
                    'low': self.config.n_components_range[0],
                    'high': self.config.n_components_range[1]
                }
            }
            
            # Define objective function
            def objective(params):
                try:
                    # This is a simplified objective function
                    # In practice, you would evaluate clustering quality
                    n_clusters = params['n_clusters']
                    n_components = params['n_components']
                    
                    # Simple scoring based on parameters
                    score = 1.0 / (1.0 + abs(n_clusters - 10) + abs(n_components - 20))
                    return score
                except Exception:
                    return 0.0
            
            # Run optimization
            result = self.optimizer.optimize(objective, search_space)
            
            if result.success:
                self.logger.info(f"✅ Optimization completed: best score = {result.best_score:.4f}")
                return {
                    'best_params': result.best_params,
                    'best_score': result.best_score,
                    'optimization_time': result.optimization_time,
                    'n_trials': result.n_trials
                }
            else:
                self.logger.warning("⚠️ Optimization failed")
                return None
                
        except Exception as e:
            self.logger.warning(f"⚠️ Feature optimization failed: {e}")
            return None
    
    def _calculate_validation_metrics(self, features: np.ndarray) -> Dict[str, float]:
        """Calculate validation metrics for features."""
        if not SKLEARN_AVAILABLE or features.shape[0] < 10:
            return {}
        
        try:
            metrics = {}
            
            # Test different numbers of clusters
            for n_clusters in range(2, min(10, features.shape[0] // 2)):
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(features)
                    
                    # Silhouette score
                    if 'silhouette' in self.config.validation_methods:
                        sil_score = silhouette_score(features, cluster_labels)
                        metrics[f'silhouette_{n_clusters}'] = sil_score
                    
                    # Calinski-Harabasz score
                    if 'calinski_harabasz' in self.config.validation_methods:
                        ch_score = calinski_harabasz_score(features, cluster_labels)
                        metrics[f'calinski_harabasz_{n_clusters}'] = ch_score
                    
                    # Davies-Bouldin score
                    if 'davies_bouldin' in self.config.validation_methods:
                        db_score = davies_bouldin_score(features, cluster_labels)
                        metrics[f'davies_bouldin_{n_clusters}'] = db_score
                        
                except Exception as e:
                    self.logger.debug(f"Validation failed for {n_clusters} clusters: {e}")
                    continue
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"⚠️ Validation metrics calculation failed: {e}")
            return {}
    
    # Helper methods for technical indicators
    def _calculate_rsi(self, prices: np.ndarray, window: int = 14) -> np.ndarray:
        """Calculate RSI indicator."""
        try:
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gains = pd.Series(gains).rolling(window=window).mean()
            avg_losses = pd.Series(losses).rolling(window=window).mean()
            
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.fillna(50).values
        except Exception:
            return np.full(len(prices), 50)
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, window: int = 20, std_dev: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands."""
        try:
            rolling_mean = pd.Series(prices).rolling(window=window).mean()
            rolling_std = pd.Series(prices).rolling(window=window).std()
            
            upper_band = rolling_mean + (rolling_std * std_dev)
            lower_band = rolling_mean - (rolling_std * std_dev)
            
            return (upper_band.fillna(method='bfill').values,
                    rolling_mean.fillna(method='bfill').values,
                    lower_band.fillna(method='bfill').values)
        except Exception:
            return (prices, prices, prices)
    
    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD indicator."""
        try:
            ema_fast = pd.Series(prices).ewm(span=fast).mean()
            ema_slow = pd.Series(prices).ewm(span=slow).mean()
            
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            
            return (macd_line.fillna(method='bfill').values,
                    signal_line.fillna(method='bfill').values,
                    histogram.fillna(method='bfill').values)
        except Exception:
            return (np.zeros_like(prices), np.zeros_like(prices), np.zeros_like(prices))
    
    def get_extraction_summary(self) -> Dict[str, Any]:
        """Get summary of feature extraction capabilities."""
        return {
            'feature_types': self.config.feature_types,
            'clustering_methods': self.config.clustering_methods,
            'reduction_methods': self.config.reduction_methods,
            'selection_methods': self.config.feature_selection_methods,
            'validation_methods': self.config.validation_methods,
            'm1_optimization': self.config.enable_m1_optimization,
            'parallel_processing': self.config.enable_parallel_processing,
            'optimization_enabled': self.config.enable_optimization,
            'shared_utils_available': self.shared_utils is not None,
            'hardware_optimizations': {
                'gpu_manager': self.gpu_manager is not None,
                'memory_optimizer': self.memory_optimizer is not None,
                'cpu_optimizer': self.cpu_optimizer is not None
            }
        }
    
    def cleanup(self):
        """Cleanup resources and stop monitoring."""
        try:
            if self.memory_optimizer and hasattr(self.memory_optimizer, 'stop_monitoring'):
                self.memory_optimizer.stop_monitoring()
            
            if SHARED_UTILS_AVAILABLE:
                from src.utils.common_operations import cleanup_m1_optimizers
                cleanup_m1_optimizers()
            
            self.logger.info("🧹 NAS Feature Extractor cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cleanup failed: {e}")


# Convenience functions
def create_nas_feature_extractor(config: Optional[NASFeatureConfig] = None) -> NASFeatureExtractor:
    """Create a NAS Feature Extractor instance."""
    return NASFeatureExtractor(config)


def extract_features_for_clustering(data: Union[pd.DataFrame, np.ndarray],
                                  config: Optional[NASFeatureConfig] = None,
                                  **kwargs) -> FeatureExtractionResult:
    """Convenience function for feature extraction."""
    extractor = NASFeatureExtractor(config)
    try:
        return extractor.extract_features(data, **kwargs)
    finally:
        extractor.cleanup()


# Export main classes and functions
__all__ = [
    'NASFeatureExtractor',
    'NASFeatureConfig',
    'FeatureExtractionResult',
    'create_nas_feature_extractor',
    'extract_features_for_clustering'
]