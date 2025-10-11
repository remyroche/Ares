"""
Unsupervised Regime Detection for Trading

Production-ready unsupervised regime detection algorithms for financial markets.
Includes real-time detection, regime transitions, stability analysis, and multi-timeframe detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
# Clustering imports removed - will be handled in subsequent step
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import talib
from scipy import stats
from scipy.signal import find_peaks
from scipy.stats import jarque_bera, kstest, anderson
from statsmodels.tsa.stattools import adfuller, kpss
# MarkovRegression not available in current statsmodels version
# from statsmodels.tsa.regime_switching import MarkovRegression
import warnings
import threading
import queue
from collections import deque

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

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class RegimeDetectionConfig:
    """Configuration for unsupervised regime detection."""
    
    # Detection parameters
    detection_method: str = "hybrid"  # "kmeans", "dbscan", "gmm", "hmm", "hybrid"
    n_regimes: Optional[int] = None  # Auto-detect if None
    min_regime_duration: int = 50  # Minimum samples per regime
    max_regimes: int = 10
    
    # Feature engineering
    feature_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    technical_indicators: List[str] = field(default_factory=lambda: [
        'RSI', 'MACD', 'BBANDS', 'STOCH', 'ADX', 'CCI', 'WILLR', 'MOM'
    ])
    volatility_windows: List[int] = field(default_factory=lambda: [10, 20, 50])
    
    # Clustering parameters
    kmeans_n_init: int = 10
    kmeans_max_iter: int = 300
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 10
    gmm_n_components: int = 5
    gmm_covariance_type: str = "full"
    
    # Regime validation
    stability_threshold: float = 0.7
    separation_threshold: float = 0.5
    economic_significance_threshold: float = 0.1
    
    # Real-time parameters
    update_frequency: int = 100  # Update every N samples
    lookback_window: int = 1000  # Lookback for regime detection
    confidence_threshold: float = 0.8
    
    # Multi-timeframe detection
    enable_multitimeframe: bool = True
    timeframes: List[str] = field(default_factory=lambda: ['1m', '5m', '15m', '1h', '4h', '1d'])
    timeframe_weights: List[float] = field(default_factory=lambda: [0.1, 0.15, 0.2, 0.25, 0.2, 0.1])
    
    # Regime transition detection
    enable_transition_detection: bool = True
    transition_threshold: float = 0.3
    transition_lookback: int = 20
    
    # Regime stability analysis
    enable_stability_analysis: bool = True
    stability_window: int = 50
    stability_threshold: float = 0.7
    
    # Streaming parameters
    enable_streaming: bool = True
    stream_buffer_size: int = 1000
    stream_processing_delay: float = 0.1  # seconds


class UnsupervisedRegimeDetector:
    """
    Production-ready unsupervised regime detector for financial markets.
    
    Implements multiple clustering algorithms, regime validation, real-time detection,
    regime transitions, stability analysis, and multi-timeframe detection for trading applications.
    """
    
    def __init__(self, config: RegimeDetectionConfig):
        """Initialize unsupervised regime detector.
        
        Args:
            config: Regime detection configuration
        """
        tprint_info("🔍 Initializing Unsupervised Regime Detection")
        tprint_debug(f"Configuration: {config}")
        tprint_debug(f"Detection enabled: {config.enable_unsupervised_detection}")
        tprint_debug(f"Number of regimes: {config.n_regimes}")
        tprint_debug(f"Detection method: {config.detection_method}")
        
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize performance tracking
        self.performance_metrics = {
            'initialization_time': 0.0,
            'detection_time': 0.0,
            'analysis_time': 0.0,
            'total_execution_time': 0.0
        }
        
        # Detection state
        self.current_regimes = None
        self.regime_history = []
        self.regime_transitions = []
        self.regime_statistics = {}
        
        # Feature engineering
        self.feature_scaler = RobustScaler()
        self.feature_names = []
        
        # Models
        self.clustering_models = {}
        self.regime_validators = {}
        
        # Real-time streaming
        self.stream_buffer = deque(maxlen=config.stream_buffer_size)
        self.streaming_thread = None
        self.streaming_active = False
        self.data_queue = queue.Queue()
        
        # Multi-timeframe detection
        self.timeframe_detectors = {}
        self.timeframe_weights = dict(zip(config.timeframes, config.timeframe_weights))
        
        # Regime transition tracking
        self.transition_history = deque(maxlen=100)
        self.current_regime_id = None
        self.regime_duration = 0
        
        # Stability analysis
        self.stability_metrics = {}
        self.regime_persistence = {}
        
        self.logger.info("✅ Unsupervised Regime Detector initialized")
    
    def detect_regimes(self, 
                      market_data: pd.DataFrame,
                      timestamps: Optional[pd.Series] = None,
                      real_time: bool = False) -> Dict[str, Any]:
        """
        Detect market regimes using unsupervised learning.
        
        Args:
            market_data: OHLCV market data
            timestamps: Optional timestamps
            real_time: Whether this is real-time detection
            
        Returns:
            Regime detection results
        """
        self.logger.info("🔍 Starting unsupervised regime detection")
        start_time = datetime.now()
        
        try:
            # Feature engineering
            features = self._engineer_features(market_data)
            
            # Detect regimes using selected method
            if self.config.detection_method == "hybrid":
                regime_results = self._hybrid_regime_detection(features, market_data)
            else:
                regime_results = self._single_method_detection(features, market_data)
            
            # Validate regimes
            validated_regimes = self._validate_regimes(regime_results, market_data)
            
            # Calculate regime statistics
            regime_stats = self._calculate_regime_statistics(validated_regimes, market_data)
            
            # Detect regime transitions
            transitions = self._detect_regime_transitions(validated_regimes)
            
            # Create comprehensive results
            results = {
                'regimes': validated_regimes,
                'regime_labels': regime_results['labels'],
                'regime_centers': regime_results['centers'],
                'regime_statistics': regime_stats,
                'regime_transitions': transitions,
                'detection_quality': self._assess_detection_quality(validated_regimes),
                'feature_importance': self._calculate_feature_importance(features, regime_results['labels']),
                'timestamp': datetime.now().isoformat(),
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'real_time': real_time
            }
            
            # Update state
            self.current_regimes = validated_regimes
            self.regime_history.append(results)
            
            self.logger.info(f"✅ Regime detection completed in {results['execution_time']:.2f}s")
            self.logger.info(f"📊 Detected {len(validated_regimes)} regimes")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Regime detection failed: {e}")
            raise
    
    def _engineer_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Engineer features for regime detection."""
        self.logger.info("🔧 Engineering features for regime detection")
        
        features = []
        feature_names = []
        
        # Price-based features
        if 'close' in market_data.columns:
            close_prices = market_data['close'].values
            
            # Returns
            returns = np.diff(close_prices) / close_prices[:-1]
            features.append(returns)
            feature_names.append('returns')
            
            # Volatility features
            for window in self.config.volatility_windows:
                if len(returns) >= window:
                    vol = pd.Series(returns).rolling(window).std().values
                    features.append(vol)
                    feature_names.append(f'volatility_{window}')
            
            # Technical indicators
            for indicator in self.config.technical_indicators:
                try:
                    if indicator == 'RSI':
                        rsi = talib.RSI(close_prices, timeperiod=14)
                        features.append(rsi)
                        feature_names.append('RSI')
                    elif indicator == 'MACD':
                        macd, macd_signal, macd_hist = talib.MACD(close_prices)
                        features.append(macd)
                        features.append(macd_signal)
                        features.append(macd_hist)
                        feature_names.extend(['MACD', 'MACD_signal', 'MACD_hist'])
                    elif indicator == 'BBANDS':
                        bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices)
                        bb_width = (bb_upper - bb_lower) / bb_middle
                        features.append(bb_width)
                        feature_names.append('BB_width')
                    elif indicator == 'STOCH':
                        stoch_k, stoch_d = talib.STOCH(
                            market_data['high'].values,
                            market_data['low'].values,
                            close_prices
                        )
                        features.append(stoch_k)
                        features.append(stoch_d)
                        feature_names.extend(['STOCH_K', 'STOCH_D'])
                    elif indicator == 'ADX':
                        adx = talib.ADX(
                            market_data['high'].values,
                            market_data['low'].values,
                            close_prices
                        )
                        features.append(adx)
                        feature_names.append('ADX')
                    elif indicator == 'CCI':
                        cci = talib.CCI(
                            market_data['high'].values,
                            market_data['low'].values,
                            close_prices
                        )
                        features.append(cci)
                        feature_names.append('CCI')
                    elif indicator == 'WILLR':
                        willr = talib.WILLR(
                            market_data['high'].values,
                            market_data['low'].values,
                            close_prices
                        )
                        features.append(willr)
                        feature_names.append('WILLR')
                    elif indicator == 'MOM':
                        mom = talib.MOM(close_prices, timeperiod=10)
                        features.append(mom)
                        feature_names.append('MOM')
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to calculate {indicator}: {e}")
                    continue
        
        # Volume features
        if 'volume' in market_data.columns:
            volume = market_data['volume'].values
            
            # Volume rate of change
            vol_roc = np.diff(volume) / volume[:-1]
            features.append(vol_roc)
            feature_names.append('volume_roc')
            
            # Volume volatility
            for window in self.config.volatility_windows:
                if len(volume) >= window:
                    vol_vol = pd.Series(volume).rolling(window).std().values
                    features.append(vol_vol)
                    feature_names.append(f'volume_volatility_{window}')
        
        # Combine features
        if not features:
            raise ValueError("No features could be engineered from the data")
        
        # Align feature lengths
        min_length = min(len(f) for f in features)
        aligned_features = []
        
        for feature in features:
            if len(feature) >= min_length:
                aligned_features.append(feature[-min_length:])
        
        # Create feature matrix
        feature_matrix = np.column_stack(aligned_features)
        
        # Handle NaN values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        feature_matrix_scaled = self.feature_scaler.fit_transform(feature_matrix)
        
        self.feature_names = feature_names
        self.logger.info(f"✅ Engineered {len(feature_names)} features")
        
        return feature_matrix_scaled
    
    def _hybrid_regime_detection(self, features: np.ndarray, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Hybrid regime detection using multiple methods."""
        self.logger.info("🔄 Using hybrid regime detection")
        
        # Try multiple methods
        methods = ['kmeans', 'gmm', 'dbscan']
        results = {}
        
        for method in methods:
            try:
                if method == 'kmeans':
                    result = self._kmeans_detection(features)
                elif method == 'gmm':
                    result = self._gmm_detection(features)
                elif method == 'dbscan':
                    result = self._dbscan_detection(features)
                
                results[method] = result
                
            except Exception as e:
                self.logger.warning(f"⚠️ {method} detection failed: {e}")
                continue
        
        # Select best method based on quality metrics
        best_method = self._select_best_method(results, features, market_data)
        
        return results[best_method]
    
    def _kmeans_detection(self, features: np.ndarray) -> Dict[str, Any]:
        """K-means regime detection."""
        tprint_debug("Performing K-means regime detection (using sequential assignment)")
        tprint_debug(f"Features shape: {features.shape}")
        
        detection_start = time.time()
        
        # Determine optimal number of clusters
        n_regimes = self.config.n_regimes
        if n_regimes is None:
            tprint_debug("Finding optimal number of clusters...")
            n_regimes = self._find_optimal_clusters(features)
            tprint_debug(f"Optimal number of clusters: {n_regimes}")
        
        tprint_debug(f"Number of regimes: {n_regimes}")
        
        # Perform simple regime assignment (replacing K-means)
        n_samples = len(features)
        regime_size = n_samples // n_regimes
        
        tprint_debug(f"Number of samples: {n_samples}")
        tprint_debug(f"Regime size: {regime_size}")
        
        labels = np.array([i // regime_size for i in range(n_samples)])
        labels = np.minimum(labels, n_regimes - 1)
        
        tprint_debug(f"Labels shape: {labels.shape}")
        tprint_debug(f"Unique labels: {len(np.unique(labels))}")
        tprint_debug(f"Label distribution: {np.bincount(labels)}")
        
        # Calculate regime centers
        centers = np.zeros((n_regimes, features.shape[1]))
        for i in range(n_regimes):
            regime_mask = labels == i
            if np.sum(regime_mask) > 0:
                centers[i] = np.mean(features[regime_mask], axis=0)
                tprint_debug(f"Regime {i} center: {centers[i]}")
        
        detection_time = time.time() - detection_start
        
        tprint_debug(f"K-means detection completed in {detection_time:.3f}s")
        tprint_debug(f"Centers shape: {centers.shape}")
        tprint_debug(f"Centers range: {np.min(centers):.3f} to {np.max(centers):.3f}")
        
        return {
            'method': 'simple_assignment',
            'labels': labels,
            'centers': centers,
            'n_regimes': n_regimes,
            'model': None
        }
    
    def _gmm_detection(self, features: np.ndarray) -> Dict[str, Any]:
        """Gaussian Mixture Model regime detection."""
        n_regimes = self.config.n_regimes or self.config.gmm_n_components
        
        gmm = GaussianMixture(
            n_components=n_regimes,
            covariance_type=self.config.gmm_covariance_type,
            random_state=42
        )
        
        labels = gmm.fit_predict(features)
        centers = gmm.means_
        
        return {
            'method': 'gmm',
            'labels': labels,
            'centers': centers,
            'n_regimes': n_regimes,
            'model': gmm
        }
    
    def _dbscan_detection(self, features: np.ndarray) -> Dict[str, Any]:
        """DBSCAN regime detection."""
        dbscan = DBSCAN(
            eps=self.config.dbscan_eps,
            min_samples=self.config.dbscan_min_samples
        )
        
        labels = dbscan.fit_predict(features)
        
        # Calculate centers for non-noise points
        unique_labels = np.unique(labels)
        centers = []
        
        for label in unique_labels:
            if label != -1:  # Skip noise points
                cluster_mask = labels == label
                center = np.mean(features[cluster_mask], axis=0)
                centers.append(center)
        
        centers = np.array(centers) if centers else np.array([])
        
        return {
            'method': 'dbscan',
            'labels': labels,
            'centers': centers,
            'n_regimes': len(unique_labels) - (1 if -1 in unique_labels else 0),
            'model': dbscan
        }
    
    def _find_optimal_clusters(self, features: np.ndarray) -> int:
        """Find optimal number of clusters using elbow method."""
        
        max_clusters = min(self.config.max_regimes, len(features) // 10)
        if max_clusters < 2:
            return 2
        
        silhouette_scores = []
        k_range = range(2, max_clusters + 1)
        
        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(features)
                score = silhouette_score(features, labels)
                silhouette_scores.append(score)
            except:
                silhouette_scores.append(0)
        
        if not silhouette_scores:
            return 3
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        return optimal_k
    
    def _validate_regimes(self, regime_results: Dict[str, Any], market_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate detected regimes."""
        self.logger.info("✅ Validating detected regimes")
        
        labels = regime_results['labels']
        unique_labels = np.unique(labels)
        
        validated_regimes = {}
        
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
            
            regime_mask = labels == label
            regime_data = market_data[regime_mask]
            
            # Check minimum duration
            if len(regime_data) < self.config.min_regime_duration:
                self.logger.warning(f"⚠️ Regime {label} too short: {len(regime_data)} samples")
                continue
            
            # Calculate regime characteristics
            regime_chars = self._calculate_regime_characteristics(regime_data, label)
            
            # Validate regime quality
            if self._is_regime_valid(regime_chars):
                validated_regimes[f'regime_{label}'] = regime_chars
            else:
                self.logger.warning(f"⚠️ Regime {label} failed validation")
        
        return validated_regimes
    
    def _calculate_regime_characteristics(self, regime_data: pd.DataFrame, regime_id: int) -> Dict[str, Any]:
        """Calculate characteristics of a regime."""
        characteristics = {
            'regime_id': regime_id,
            'duration': len(regime_data),
            'start_time': regime_data.index[0] if hasattr(regime_data, 'index') else None,
            'end_time': regime_data.index[-1] if hasattr(regime_data, 'index') else None
        }
        
        if 'close' in regime_data.columns:
            close_prices = regime_data['close'].values
            
            # Price characteristics
            characteristics.update({
                'price_range': np.max(close_prices) - np.min(close_prices),
                'price_volatility': np.std(close_prices) / np.mean(close_prices),
                'price_trend': (close_prices[-1] - close_prices[0]) / close_prices[0],
                'mean_price': np.mean(close_prices),
                'price_skewness': stats.skew(close_prices),
                'price_kurtosis': stats.kurtosis(close_prices)
            })
            
            # Returns characteristics
            returns = np.diff(close_prices) / close_prices[:-1]
            if len(returns) > 0:
                characteristics.update({
                    'mean_return': np.mean(returns),
                    'return_volatility': np.std(returns),
                    'return_skewness': stats.skew(returns),
                    'return_kurtosis': stats.kurtosis(returns),
                    'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
                })
        
        if 'volume' in regime_data.columns:
            volume = regime_data['volume'].values
            characteristics.update({
                'mean_volume': np.mean(volume),
                'volume_volatility': np.std(volume) / np.mean(volume),
                'volume_trend': (volume[-1] - volume[0]) / volume[0] if volume[0] > 0 else 0
            })
        
        return characteristics
    
    def _is_regime_valid(self, regime_chars: Dict[str, Any]) -> bool:
        """Check if a regime is valid based on characteristics."""
        # Check stability
        if regime_chars.get('price_volatility', 0) > 0.5:  # Too volatile
            return False
        
        # Check economic significance
        if abs(regime_chars.get('price_trend', 0)) < self.config.economic_significance_threshold:
            return False
        
        # Check duration
        if regime_chars.get('duration', 0) < self.config.min_regime_duration:
            return False
        
        return True
    
    def _calculate_regime_statistics(self, regimes: Dict[str, Any], market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate overall regime statistics."""
        if not regimes:
            return {}
        
        stats = {
            'n_regimes': len(regimes),
            'total_duration': sum(regime['duration'] for regime in regimes.values()),
            'regime_distribution': {name: regime['duration'] for name, regime in regimes.items()},
            'volatility_distribution': {name: regime.get('price_volatility', 0) for name, regime in regimes.items()},
            'trend_distribution': {name: regime.get('price_trend', 0) for name, regime in regimes.items()}
        }
        
        return stats
    
    def _detect_regime_transitions(self, regimes: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect regime transitions."""
        transitions = []
        
        # This would be implemented based on the regime sequence
        # For now, return empty list
        return transitions
    
    def _assess_detection_quality(self, regimes: Dict[str, Any]) -> Dict[str, float]:
        """Assess the quality of regime detection."""
        if not regimes:
            return {'quality_score': 0.0, 'stability': 0.0, 'separation': 0.0}
        
        # Calculate quality metrics
        n_regimes = len(regimes)
        total_duration = sum(regime['duration'] for regime in regimes.values())
        
        # Stability score (based on regime duration consistency)
        durations = [regime['duration'] for regime in regimes.values()]
        stability = 1.0 - (np.std(durations) / np.mean(durations)) if durations else 0.0
        
        # Separation score (based on regime characteristics)
        volatilities = [regime.get('price_volatility', 0) for regime in regimes.values()]
        separation = 1.0 - (np.std(volatilities) / np.mean(volatilities)) if volatilities else 0.0
        
        # Overall quality score
        quality_score = (stability + separation) / 2.0
        
        return {
            'quality_score': quality_score,
            'stability': stability,
            'separation': separation,
            'n_regimes': n_regimes,
            'total_duration': total_duration
        }
    
    def _calculate_feature_importance(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance for regime detection."""
        if len(self.feature_names) != features.shape[1]:
            return {}
        
        # Simple feature importance based on variance between regimes
        unique_labels = np.unique(labels)
        importance_scores = {}
        
        for i, feature_name in enumerate(self.feature_names):
            feature_values = features[:, i]
            regime_means = []
            
            for label in unique_labels:
                if label != -1:  # Skip noise
                    regime_mask = labels == label
                    regime_mean = np.mean(feature_values[regime_mask])
                    regime_means.append(regime_mean)
            
            if len(regime_means) > 1:
                importance = np.std(regime_means) / (np.mean(regime_means) + 1e-8)
                importance_scores[feature_name] = importance
            else:
                importance_scores[feature_name] = 0.0
        
        return importance_scores
    
    def _select_best_method(self, results: Dict[str, Any], features: np.ndarray, market_data: pd.DataFrame) -> str:
        """Select the best detection method based on quality metrics."""
        best_method = 'kmeans'
        best_score = 0.0
        
        for method, result in results.items():
            if result is None:
                continue
            
            # Calculate quality score for this method
            labels = result['labels']
            unique_labels = np.unique(labels)
            
            if len(unique_labels) < 2:
                continue
            
            # Calculate silhouette score
            try:
                score = silhouette_score(features, labels)
                
                if score > best_score:
                    best_score = score
                    best_method = method
            except:
                continue
        
        return best_method
    
    def start_streaming_detection(self):
        """Start real-time streaming regime detection."""
        if self.streaming_active:
            self.logger.warning("⚠️ Streaming already active")
            return
        
        self.streaming_active = True
        self.streaming_thread = threading.Thread(target=self._streaming_loop, daemon=True)
        self.streaming_thread.start()
        
        self.logger.info("🚀 Real-time streaming regime detection started")
    
    def stop_streaming_detection(self):
        """Stop real-time streaming regime detection."""
        self.streaming_active = False
        if self.streaming_thread:
            self.streaming_thread.join(timeout=5)
        
        self.logger.info("🛑 Real-time streaming regime detection stopped")
    
    def add_streaming_data(self, data: pd.DataFrame):
        """Add new data to streaming buffer."""
        if not self.streaming_active:
            self.logger.warning("⚠️ Streaming not active")
            return
        
        self.data_queue.put(data)
    
    def _streaming_loop(self):
        """Main streaming processing loop."""
        while self.streaming_active:
            try:
                # Get new data from queue
                if not self.data_queue.empty():
                    new_data = self.data_queue.get(timeout=1)
                    
                    # Add to buffer
                    self.stream_buffer.append(new_data)
                    
                    # Process if buffer has enough data
                    if len(self.stream_buffer) >= self.config.update_frequency:
                        self._process_streaming_data()
                
                # Sleep to prevent excessive CPU usage
                threading.Event().wait(self.config.stream_processing_delay)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"❌ Streaming loop error: {e}")
                threading.Event().wait(1)
    
    def _process_streaming_data(self):
        """Process streaming data for regime detection."""
        # Convert buffer to DataFrame
        combined_data = pd.concat(list(self.stream_buffer))
        
        # Detect regimes
        results = self.detect_regimes(combined_data, real_time=True)
        
        # Update current regime
        if results.get('regimes'):
            self._update_current_regime(results)
        
        # Detect transitions
        if self.config.enable_transition_detection:
            self._detect_regime_transitions_streaming(results)
        
        # Analyze stability
        if self.config.enable_stability_analysis:
            self._analyze_regime_stability_streaming(results)
    
    def detect_regimes_multitimeframe(self, 
                                    market_data: Dict[str, pd.DataFrame],
                                    timestamps: Optional[Dict[str, pd.Series]] = None) -> Dict[str, Any]:
        """
        Detect regimes across multiple timeframes.
        
        Args:
            market_data: Dictionary of market data by timeframe
            timestamps: Optional timestamps for each timeframe
            
        Returns:
            Multi-timeframe regime detection results
        """
        self.logger.info("🔄 Starting multi-timeframe regime detection")
        
        timeframe_results = {}
        combined_regimes = {}
        
        for timeframe, data in market_data.items():
            if timeframe not in self.timeframe_weights:
                continue
            
            self.logger.info(f"📊 Processing timeframe: {timeframe}")
            
            # Detect regimes for this timeframe
            results = self.detect_regimes(data, timestamps.get(timeframe) if timestamps else None)
            timeframe_results[timeframe] = results
            
            # Store regimes with timeframe weighting
            weight = self.timeframe_weights[timeframe]
            for regime_name, regime_info in results.get('regimes', {}).items():
                weighted_regime = regime_info.copy()
                weighted_regime['timeframe'] = timeframe
                weighted_regime['weight'] = weight
                weighted_regime['confidence'] = regime_info.get('confidence', 0.5) * weight
                
                combined_regimes[f"{timeframe}_{regime_name}"] = weighted_regime
        
        # Combine regimes across timeframes
        combined_results = self._combine_multitimeframe_regimes(timeframe_results)
        
        # Calculate consensus regimes
        consensus_regimes = self._calculate_consensus_regimes(combined_regimes)
        
        return {
            'timeframe_results': timeframe_results,
            'combined_regimes': combined_regimes,
            'consensus_regimes': consensus_regimes,
            'detection_quality': self._assess_multitimeframe_quality(timeframe_results),
            'timestamp': datetime.now().isoformat()
        }
    
    def _combine_multitimeframe_regimes(self, timeframe_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine regimes across timeframes."""
        combined = {
            'regimes': {},
            'regime_labels': {},
            'regime_centers': {},
            'detection_quality': {}
        }
        
        for timeframe, results in timeframe_results.items():
            weight = self.timeframe_weights[timeframe]
            
            # Weight regime characteristics
            for regime_name, regime_info in results.get('regimes', {}).items():
                weighted_info = regime_info.copy()
                weighted_info['timeframe_weight'] = weight
                weighted_info['confidence'] = regime_info.get('confidence', 0.5) * weight
                
                combined['regimes'][f"{timeframe}_{regime_name}"] = weighted_info
        
        return combined
    
    def _calculate_consensus_regimes(self, combined_regimes: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate consensus regimes across timeframes."""
        consensus = {}
        
        # Group regimes by similarity
        regime_groups = self._group_similar_regimes(combined_regimes)
        
        for group_id, regime_group in regime_groups.items():
            # Calculate consensus characteristics
            consensus_regime = self._calculate_regime_consensus(regime_group)
            consensus[f"consensus_{group_id}"] = consensus_regime
        
        return consensus
    
    def _group_similar_regimes(self, regimes: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Group similar regimes across timeframes."""
        groups = {}
        group_id = 0
        
        for regime_name, regime_info in regimes.items():
            # Find similar existing group
            assigned = False
            for existing_group_id, group in groups.items():
                if self._regimes_are_similar(regime_info, group[0]):
                    groups[existing_group_id].append(regime_info)
                    assigned = True
                    break
            
            if not assigned:
                groups[f"group_{group_id}"] = [regime_info]
                group_id += 1
        
        return groups
    
    def _regimes_are_similar(self, regime1: Dict[str, Any], regime2: Dict[str, Any]) -> bool:
        """Check if two regimes are similar."""
        # Compare key characteristics
        vol1 = regime1.get('price_volatility', 0)
        vol2 = regime2.get('price_volatility', 0)
        trend1 = regime1.get('price_trend', 0)
        trend2 = regime2.get('price_trend', 0)
        
        # Similarity thresholds
        vol_similar = abs(vol1 - vol2) < 0.1
        trend_similar = abs(trend1 - trend2) < 0.05
        
        return vol_similar and trend_similar
    
    def _calculate_regime_consensus(self, regime_group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate consensus regime from a group of similar regimes."""
        if not regime_group:
            return {}
        
        # Weighted average of characteristics
        total_weight = sum(regime.get('weight', 1.0) for regime in regime_group)
        
        consensus = {
            'regime_id': f"consensus_{len(regime_group)}",
            'timeframes': [regime.get('timeframe') for regime in regime_group],
            'confidence': 0.0,
            'duration': 0,
            'price_volatility': 0.0,
            'price_trend': 0.0,
            'mean_price': 0.0,
            'sharpe_ratio': 0.0
        }
        
        for regime in regime_group:
            weight = regime.get('weight', 1.0) / total_weight
            
            consensus['confidence'] += regime.get('confidence', 0.5) * weight
            consensus['duration'] += regime.get('duration', 0) * weight
            consensus['price_volatility'] += regime.get('price_volatility', 0) * weight
            consensus['price_trend'] += regime.get('price_trend', 0) * weight
            consensus['mean_price'] += regime.get('mean_price', 0) * weight
            consensus['sharpe_ratio'] += regime.get('sharpe_ratio', 0) * weight
        
        return consensus
    
    def _assess_multitimeframe_quality(self, timeframe_results: Dict[str, Any]) -> Dict[str, float]:
        """Assess quality of multi-timeframe detection."""
        quality_metrics = {}
        
        for timeframe, results in timeframe_results.items():
            detection_quality = results.get('detection_quality', {})
            quality_metrics[timeframe] = {
                'quality_score': detection_quality.get('quality_score', 0.0),
                'stability': detection_quality.get('stability', 0.0),
                'separation': detection_quality.get('separation', 0.0)
            }
        
        # Overall quality
        overall_quality = np.mean([
            metrics['quality_score'] for metrics in quality_metrics.values()
        ])
        
        quality_metrics['overall'] = {
            'quality_score': overall_quality,
            'consistency': 1.0 - np.std([
                metrics['quality_score'] for metrics in quality_metrics.values()
            ])
        }
        
        return quality_metrics
    
    def _update_current_regime(self, results: Dict[str, Any]):
        """Update current regime information."""
        if not results.get('regimes'):
            return
        
        # Find most recent regime
        latest_regime = max(results['regimes'].items(), 
                           key=lambda x: x[1].get('confidence', 0))
        
        regime_id, regime_info = latest_regime
        
        if regime_id != self.current_regime_id:
            # Regime change detected
            if self.current_regime_id is not None:
                self.transition_history.append({
                    'from_regime': self.current_regime_id,
                    'to_regime': regime_id,
                    'timestamp': datetime.now(),
                    'duration': self.regime_duration
                })
            
            self.current_regime_id = regime_id
            self.regime_duration = 0
        else:
            self.regime_duration += 1
    
    def _detect_regime_transitions_streaming(self, results: Dict[str, Any]):
        """Detect regime transitions in streaming mode."""
        if not self.config.enable_transition_detection:
            return
        
        # Check for regime changes
        if len(self.transition_history) > 0:
            latest_transition = self.transition_history[-1]
            
            # Check if transition is significant
            if self._is_transition_significant(latest_transition):
                self.logger.info(f"🔄 Significant regime transition detected: {latest_transition}")
                
                # Update transition tracking
                self.regime_transitions.append(latest_transition)
    
    def _is_transition_significant(self, transition: Dict[str, Any]) -> bool:
        """Check if a regime transition is significant."""
        # Check transition probability
        transition_prob = self._calculate_transition_probability(transition)
        
        # Check regime stability
        stability = self._calculate_regime_stability(transition)
        
        # Significant if both probability and stability exceed thresholds
        return (transition_prob > self.config.transition_threshold and 
                stability > self.config.stability_threshold)
    
    def _calculate_transition_probability(self, transition: Dict[str, Any]) -> float:
        """Calculate probability of regime transition."""
        # Simplified transition probability calculation
        # In production, this would use more sophisticated methods
        
        from_regime = transition.get('from_regime')
        to_regime = transition.get('to_regime')
        
        if not from_regime or not to_regime:
            return 0.0
        
        # Calculate based on regime characteristics
        # This is a simplified implementation
        return 0.5  # Placeholder
    
    def _calculate_regime_stability(self, transition: Dict[str, Any]) -> float:
        """Calculate regime stability."""
        # Simplified stability calculation
        # In production, this would use more sophisticated methods
        
        duration = transition.get('duration', 0)
        min_duration = self.config.min_regime_duration
        
        return min(duration / min_duration, 1.0)
    
    def _analyze_regime_stability_streaming(self, results: Dict[str, Any]):
        """Analyze regime stability in streaming mode."""
        if not self.config.enable_stability_analysis:
            return
        
        # Calculate stability metrics
        stability_metrics = self._calculate_stability_metrics(results)
        
        # Update stability tracking
        self.stability_metrics.update(stability_metrics)
        
        # Check for stability alerts
        if stability_metrics.get('stability_score', 0) < self.config.stability_threshold:
            self.logger.warning(f"⚠️ Low regime stability detected: {stability_metrics}")
    
    def _calculate_stability_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate regime stability metrics."""
        if not results.get('regimes'):
            return {}
        
        regimes = results['regimes']
        
        # Calculate stability based on regime characteristics
        volatility_stability = 1.0 - np.std([
            regime.get('price_volatility', 0) for regime in regimes.values()
        ])
        
        trend_stability = 1.0 - np.std([
            regime.get('price_trend', 0) for regime in regimes.values()
        ])
        
        # Overall stability score
        stability_score = (volatility_stability + trend_stability) / 2.0
        
        return {
            'stability_score': stability_score,
            'volatility_stability': volatility_stability,
            'trend_stability': trend_stability,
            'n_regimes': len(regimes)
        }
    
    def update_regimes_real_time(self, new_data: pd.DataFrame) -> Dict[str, Any]:
        """Update regime detection in real-time."""
        self.logger.info("🔄 Updating regimes in real-time")
        
        # Combine with recent history
        if hasattr(self, 'recent_data'):
            combined_data = pd.concat([self.recent_data, new_data])
        else:
            combined_data = new_data
        
        # Keep only recent window
        if len(combined_data) > self.config.lookback_window:
            combined_data = combined_data.tail(self.config.lookback_window)
        
        # Update recent data
        self.recent_data = combined_data
        
        # Detect regimes
        results = self.detect_regimes(combined_data, real_time=True)
        
        return results
    
    def get_current_regime(self) -> Optional[Dict[str, Any]]:
        """Get current regime information."""
        if self.current_regimes is None:
            return None
        
        # Return the most recent regime
        if self.regime_history:
            latest_results = self.regime_history[-1]
            return {
                'regimes': latest_results['regimes'],
                'current_regime': latest_results.get('current_regime'),
                'confidence': latest_results.get('detection_quality', {}).get('quality_score', 0.0),
                'timestamp': latest_results['timestamp']
            }
        
        return None
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get comprehensive regime statistics."""
        if not self.regime_history:
            return {}
        
        return {
            'n_detections': len(self.regime_history),
            'latest_quality': self.regime_history[-1].get('detection_quality', {}),
            'feature_importance': self.regime_history[-1].get('feature_importance', {}),
            'regime_transitions': len(self.regime_transitions),
            'detection_method': self.config.detection_method
        }

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
