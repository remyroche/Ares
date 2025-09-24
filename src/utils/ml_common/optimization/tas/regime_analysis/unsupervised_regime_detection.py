"""
Unsupervised Regime Detection for Trading

Production-ready unsupervised regime detection algorithms for financial markets.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
import talib
from scipy import stats
from scipy.signal import find_peaks
import warnings
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


class UnsupervisedRegimeDetector:
    """
    Production-ready unsupervised regime detector for financial markets.
    
    Implements multiple clustering algorithms and regime validation
    for real-time trading applications.
    """
    
    def __init__(self, config: RegimeDetectionConfig):
        """Initialize unsupervised regime detector.
        
        Args:
            config: Regime detection configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
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
        # Determine optimal number of clusters
        n_regimes = self.config.n_regimes
        if n_regimes is None:
            n_regimes = self._find_optimal_clusters(features)
        
        # Perform K-means clustering
        kmeans = KMeans(
            n_clusters=n_regimes,
            n_init=self.config.kmeans_n_init,
            max_iter=self.config.kmeans_max_iter,
            random_state=42
        )
        
        labels = kmeans.fit_predict(features)
        centers = kmeans.cluster_centers_
        
        return {
            'method': 'kmeans',
            'labels': labels,
            'centers': centers,
            'n_regimes': n_regimes,
            'model': kmeans
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
        from sklearn.metrics import silhouette_score
        
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
                from sklearn.metrics import silhouette_score
                score = silhouette_score(features, labels)
                
                if score > best_score:
                    best_score = score
                    best_method = method
            except:
                continue
        
        return best_method
    
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