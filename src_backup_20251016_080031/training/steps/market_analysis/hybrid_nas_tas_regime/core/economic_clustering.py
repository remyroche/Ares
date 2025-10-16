"""
Economic Clustering for Hybrid NAS-TAS Regime System

This module provides economic-aware clustering algorithms that integrate economic significance,
momentum, and volume analysis directly into the clustering process.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

logger = logging.getLogger(__name__)

@dataclass
class EconomicClusteringResult:
    """Result from economic clustering operation."""
    labels: np.ndarray
    cluster_centers: np.ndarray
    probabilities: np.ndarray
    economic_significance: np.ndarray
    momentum_scores: np.ndarray
    volume_profiles: np.ndarray
    quality_metrics: Dict[str, float]
    economic_metrics: Dict[str, float]
    frontier_metrics: Dict[str, Any]
    regime_transfers: List[Dict[str, Any]]
    algorithm_used: str
    execution_time: float
    metadata: Dict[str, Any]

class EconomicDistanceMetric(Enum):
    """Economic distance metrics for clustering."""
    ECONOMIC_EUCLIDEAN = "economic_euclidean"
    MOMENTUM_WEIGHTED = "momentum_weighted"
    VOLUME_ADJUSTED = "volume_adjusted"
    ECONOMIC_MANHATTAN = "economic_manhattan"
    HYBRID_ECONOMIC = "hybrid_economic"

@dataclass
class EconomicCluster:
    """Economic cluster definition."""
    cluster_id: int
    size: int
    economic_significance: float
    momentum_score: float
    volume_profile: float
    price_action_score: float
    market_efficiency: float
    liquidity_score: float
    characteristics: Dict[str, float]
    centroid: np.ndarray
    description: str

class EconomicClusterer:
    """
    Economic-aware clustering that integrates economic significance, momentum,
    and volume analysis directly into the clustering process.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize economic clusterer."""
        tprint_info("🚀 Initializing Economic Clusterer")
        tprint_debug(f"Configuration: {config}")
        
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Economic parameters
        tprint_debug("⚙️ Setting economic parameters...")
        self.momentum_threshold = config.get('momentum_threshold', 0.7)
        self.volume_threshold = config.get('volume_threshold', 0.6)
        self.economic_significance_weight = config.get('economic_significance_weight', 0.3)
        self.momentum_weight = config.get('momentum_weight', 0.25)
        self.volume_weight = config.get('volume_weight', 0.25)
        self.momentum_periods = config.get('momentum_periods', [1, 2, 5, 10])  # 15m, 30m, 1.25h, 2.5h for 15m trading
        tprint_success("✅ Economic parameters configured")

        # Economic distance metric
        tprint_debug("📏 Setting economic distance metric...")
        self.economic_distance_metric = config.get('economic_distance_metric', 'economic_euclidean')
        tprint_success(f"✅ Distance metric: {self.economic_distance_metric}")

        # Available clustering algorithms
        tprint_debug("🔧 Initializing clustering algorithms...")
        self.clustering_algorithms = {
            'economic_kmeans': self._economic_kmeans,
            'economic_hierarchical': self._economic_hierarchical,
            'economic_gmm': self._economic_gmm,
            'economic_adaptive': self._economic_adaptive
        }
        tprint_success(f"✅ {len(self.clustering_algorithms)} clustering algorithms available")

        tprint_success("✅ Economic Clusterer initialized")
        self.logger.info("✅ Economic Clusterer initialized")

    def cluster_economic_features(self,
                                 features: np.ndarray,
                                 market_data: pd.DataFrame) -> EconomicClusteringResult:
        """
        Perform economic-aware clustering.

        Args:
            features: Feature matrix
            market_data: Market data for economic analysis

        Returns:
            EconomicClusteringResult with economic clustering results
        """
        try:
            tprint("🔍 [ECONOMIC_CLUSTERING] Starting economic clustering", color="blue", bold=True)
            tprint_debug(f"📊 [ECONOMIC_CLUSTERING] Features shape: {features.shape}")
            tprint_debug(f"📊 [ECONOMIC_CLUSTERING] Market data shape: {market_data.shape}")
            tprint_debug(f"⚙️ [ECONOMIC_CLUSTERING] Economic weights - significance: {self.economic_significance_weight}, momentum: {self.momentum_weight}, volume: {self.volume_weight}")
            self.logger.info("🔍 Starting economic clustering...")

            # Extract economic features
            tprint("📊 [ECONOMIC_CLUSTERING] Extracting economic features", color="cyan")
            tprint_debug(f"🔧 [ECONOMIC_CLUSTERING] Using economic significance weight: {self.economic_significance_weight}")
            economic_features = self._extract_economic_features(features, market_data)
            tprint_success(f"✅ [ECONOMIC_CLUSTERING] Economic features extracted: {economic_features.shape}")
            tprint_performance(f"⚡ [ECONOMIC_CLUSTERING] Economic features: {economic_features.shape[0]} samples, {economic_features.shape[1]} features")

            # Calculate momentum features
            tprint("📈 [ECONOMIC_CLUSTERING] Calculating momentum features", color="cyan")
            tprint_debug(f"🔧 [ECONOMIC_CLUSTERING] Momentum weight: {self.momentum_weight}, periods: {self.momentum_periods}")
            momentum_features = self._calculate_momentum_features(market_data)
            tprint_success(f"✅ [ECONOMIC_CLUSTERING] Momentum features calculated: {momentum_features.shape}")
            tprint_performance(f"⚡ [ECONOMIC_CLUSTERING] Momentum features: {momentum_features.shape[0]} samples, {momentum_features.shape[1]} features")

            # Calculate volume features
            tprint("📊 [ECONOMIC_CLUSTERING] Calculating volume features", color="cyan")
            tprint_debug(f"🔧 [ECONOMIC_CLUSTERING] Volume weight: {self.volume_weight}, threshold: {self.volume_threshold}")
            volume_features = self._calculate_volume_features(market_data)
            tprint_success(f"✅ [ECONOMIC_CLUSTERING] Volume features calculated: {volume_features.shape}")
            tprint_performance(f"⚡ [ECONOMIC_CLUSTERING] Volume features: {volume_features.shape[0]} samples, {volume_features.shape[1]} features")

            # Combine all features with economic weighting
            tprint("🔄 [ECONOMIC_CLUSTERING] Combining features with economic weighting", color="cyan")
            tprint_debug(f"🔧 [ECONOMIC_CLUSTERING] Combining: base({features.shape}) + economic({economic_features.shape}) + momentum({momentum_features.shape}) + volume({volume_features.shape})")
            combined_features = self._combine_economic_features(
                features, economic_features, momentum_features, volume_features
            )
            tprint_success(f"✅ [ECONOMIC_CLUSTERING] Combined features: {combined_features.shape}")
            tprint_performance(f"⚡ [ECONOMIC_CLUSTERING] Combined features: {combined_features.shape[0]} samples, {combined_features.shape[1]} features")

            # Choose clustering algorithm
            algorithm = self.config.get('primary_algorithm', 'economic_adaptive').replace('economic_', '')
            tprint(f"🎯 [ECONOMIC_CLUSTERING] Using clustering algorithm: {algorithm}", color="magenta")
            tprint_debug(f"🔧 [ECONOMIC_CLUSTERING] Available algorithms: {list(self.clustering_algorithms.keys())}")

            if algorithm in self.clustering_algorithms:
                tprint(f"🔧 [ECONOMIC_CLUSTERING] Using {algorithm} clustering", color="blue")
                result = self.clustering_algorithms[algorithm](combined_features)
                tprint_success(f"✅ [ECONOMIC_CLUSTERING] {algorithm} clustering completed")
            else:
                tprint("⚠️ [ECONOMIC_CLUSTERING] Algorithm not found, using economic_adaptive", color="yellow")
                result = self._economic_adaptive(combined_features)
                tprint_success("✅ [ECONOMIC_CLUSTERING] Economic adaptive clustering completed")

            unique_clusters = len(set(result['labels']))
            tprint(f"📊 [ECONOMIC_CLUSTERING] Clustering result: {unique_clusters} clusters, {len(result['labels'])} samples", color="green")

            # Calculate quality metrics
            tprint("📊 [ECONOMIC_CLUSTERING] Calculating quality metrics", color="cyan")
            quality_metrics = self._calculate_economic_quality_metrics(combined_features, result['labels'])
            tprint_success("✅ [ECONOMIC_CLUSTERING] Quality metrics calculated")
            tprint_debug(f"📈 [ECONOMIC_CLUSTERING] Quality metrics keys: {list(quality_metrics.keys())}")

            # Calculate economic metrics
            tprint("💰 [ECONOMIC_CLUSTERING] Calculating economic metrics", color="cyan")
            economic_metrics = self._calculate_economic_metrics(
                market_data, result['labels'], result['cluster_centers']
            )
            tprint_success("✅ [ECONOMIC_CLUSTERING] Economic metrics calculated")
            tprint_debug(f"💰 [ECONOMIC_CLUSTERING] Economic metrics keys: {list(economic_metrics.keys())}")

            # Perform frontier analysis
            tprint("📈 [ECONOMIC_CLUSTERING] Performing frontier analysis", color="cyan")
            frontier_metrics = self._economic_frontier_analysis(combined_features, result['labels'])
            tprint_success("✅ [ECONOMIC_CLUSTERING] Frontier analysis completed")
            tprint_debug(f"📈 [ECONOMIC_CLUSTERING] Frontier metrics keys: {list(frontier_metrics.keys())}")

            # Optimize regime transfers
            tprint("🔄 [ECONOMIC_CLUSTERING] Optimizing regime transfers", color="cyan")
            regime_transfers = self._economic_regime_transfer_optimization(
                combined_features, result['labels'], frontier_metrics
            )
            tprint_success(f"✅ [ECONOMIC_CLUSTERING] Regime transfers optimized: {len(regime_transfers)} transfers")
            tprint_debug(f"🔄 [ECONOMIC_CLUSTERING] Regime transfer details: {len(regime_transfers)} transitions")

            tprint_success(f"🎉 [ECONOMIC_CLUSTERING] Economic clustering completed successfully")
            tprint_performance(f"⚡ [ECONOMIC_CLUSTERING] Final result: {unique_clusters} clusters, {len(result['labels'])} samples, {combined_features.shape[1]} features")
            
            return EconomicClusteringResult(
                labels=result['labels'],
                cluster_centers=result['cluster_centers'],
                probabilities=result['probabilities'],
                economic_significance=economic_metrics.get('economic_significance', np.zeros(len(set(result['labels'])))),
                momentum_scores=economic_metrics.get('momentum_scores', np.zeros(len(set(result['labels'])))),
                volume_profiles=economic_metrics.get('volume_profiles', np.zeros(len(set(result['labels'])))),
                quality_metrics=quality_metrics,
                economic_metrics=economic_metrics,
                frontier_metrics=frontier_metrics,
                regime_transfers=regime_transfers,
                algorithm_used=algorithm,
                execution_time=result.get('execution_time', 0.0),
                metadata={
                    'n_features': combined_features.shape[1],
                    'n_samples': combined_features.shape[0],
                    'n_regimes': len(set(result['labels'])),
                    'economic_features': economic_features.shape[1] if economic_features is not None else 0,
                    'momentum_features': momentum_features.shape[1] if momentum_features is not None else 0,
                    'volume_features': volume_features.shape[1] if volume_features is not None else 0,
                    'timestamp': datetime.now().isoformat()
                }
            )

        except Exception as e:
            tprint_error(f"❌ [ECONOMIC_CLUSTERING] Economic clustering failed: {e}")
            tprint_debug(f"🔍 [ECONOMIC_CLUSTERING] Error details: {str(e)}")
            self.logger.error(f"Economic clustering failed: {e}")
            raise

    def _extract_economic_features(self, features: np.ndarray, market_data: pd.DataFrame) -> np.ndarray:
        """Extract economic features from market data."""
        try:
            economic_features_list = []

            # Price-based economic features
            close_prices = market_data['close'].values

            # Volatility features
            returns = np.diff(close_prices, prepend=close_prices[0])
            volatility_features = self._calculate_volatility_features(returns)
            economic_features_list.append(volatility_features)

            # Trend features
            trend_features = self._calculate_trend_features(close_prices)
            economic_features_list.append(trend_features)

            # Market efficiency features
            efficiency_features = self._calculate_efficiency_features(returns)
            economic_features_list.append(efficiency_features)

            # Liquidity features
            liquidity_features = self._calculate_liquidity_features(market_data)
            economic_features_list.append(liquidity_features)

            # Price action features
            price_action_features = self._calculate_price_action_features(market_data)
            economic_features_list.append(price_action_features)

            # Combine all economic features
            if economic_features_list:
                return np.hstack([f.reshape(-1, 1) if f.ndim == 1 else f for f in economic_features_list])
            else:
                return np.zeros((len(market_data), 1))

        except Exception as e:
            self.logger.warning(f"Economic feature extraction failed: {e}")
            return np.zeros((len(market_data), 1))

    def _calculate_momentum_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Calculate momentum features for different periods."""
        try:
            close_prices = market_data['close'].values
            momentum_features_list = []

            for period in self.momentum_periods:
                if len(close_prices) > period:
                    # Price momentum
                    momentum = (close_prices - np.roll(close_prices, period)) / (np.roll(close_prices, period) + 1e-8)
                    momentum_features_list.append(momentum.reshape(-1, 1))

                    # Rate of change
                    roc = np.diff(np.log(close_prices), n=period, prepend=np.log(close_prices[:1]))
                    momentum_features_list.append(roc.reshape(-1, 1))

            if momentum_features_list:
                return np.hstack(momentum_features_list)
            else:
                return np.zeros((len(market_data), 1))

        except Exception as e:
            self.logger.warning(f"Momentum feature calculation failed: {e}")
            return np.zeros((len(market_data), 1))

    def _calculate_volume_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Calculate volume-based features."""
        try:
            volume_features_list = []

            if 'volume' in market_data.columns:
                volume = market_data['volume'].values
                close_prices = market_data['close'].values

                # Volume features
                volume_ma = pd.Series(volume).rolling(window=20, min_periods=5).mean().fillna(method='bfill').values
                volume_std = pd.Series(volume).rolling(window=20, min_periods=5).std().fillna(method='bfill').values

                volume_features_list.append(volume_ma.reshape(-1, 1))
                volume_features_list.append(volume_std.reshape(-1, 1))

                # Volume-price relationship
                volume_price_corr = pd.Series(volume).rolling(window=20, min_periods=10).corr(pd.Series(close_prices))
                volume_price_corr = volume_price_corr.fillna(0.5).values
                volume_features_list.append(volume_price_corr.reshape(-1, 1))

                # Volume trend
                volume_trend = pd.Series(volume).pct_change().fillna(0).values
                volume_features_list.append(volume_trend.reshape(-1, 1))

            if volume_features_list:
                return np.hstack(volume_features_list)
            else:
                return np.zeros((len(market_data), 1))

        except Exception as e:
            self.logger.warning(f"Volume feature calculation failed: {e}")
            return np.zeros((len(market_data), 1))

    def _combine_economic_features(self,
                                  base_features: np.ndarray,
                                  economic_features: np.ndarray,
                                  momentum_features: np.ndarray,
                                  volume_features: np.ndarray) -> np.ndarray:
        """Combine features with economic weighting."""
        try:
            features_list = []

            # Add base features with reduced weight
            if base_features.size > 0:
                base_weighted = base_features * (1 - self.economic_significance_weight - self.momentum_weight - self.volume_weight)
                features_list.append(base_weighted)

            # Add economic features
            if economic_features.size > 0:
                economic_weighted = economic_features * self.economic_significance_weight
                features_list.append(economic_weighted)

            # Add momentum features
            if momentum_features.size > 0:
                momentum_weighted = momentum_features * self.momentum_weight
                features_list.append(momentum_weighted)

            # Add volume features
            if volume_features.size > 0:
                volume_weighted = volume_features * self.volume_weight
                features_list.append(volume_weighted)

            if features_list:
                return np.hstack(features_list)
            else:
                return np.zeros((base_features.shape[0], 1))

        except Exception as e:
            self.logger.warning(f"Feature combination failed: {e}")
            return base_features

    def _calculate_volatility_features(self, returns: np.ndarray) -> np.ndarray:
        """Calculate volatility-based economic features."""
        try:
            features_list = []

            # Rolling volatility
            for window in [5, 10, 20, 50]:
                if len(returns) > window:
                    rolling_vol = pd.Series(np.abs(returns)).rolling(window=window, min_periods=window//2).std()
                    rolling_vol = rolling_vol.fillna(rolling_vol.mean()).values
                    features_list.append(rolling_vol.reshape(-1, 1))

            # Volatility regime indicators
            volatility_levels = self._classify_volatility_regime(returns)
            features_list.append(volatility_levels.reshape(-1, 1))

            return np.hstack(features_list) if features_list else np.zeros((len(returns), 1))

        except Exception as e:
            self.logger.warning(f"Volatility feature calculation failed: {e}")
            return np.zeros((len(returns), 1))

    def _calculate_trend_features(self, prices: np.ndarray) -> np.ndarray:
        """Calculate trend-based economic features."""
        try:
            features_list = []

            # Trend strength for different periods
            for period in [5, 10, 20, 50]:
                if len(prices) > period:
                    # Linear trend R-squared - calculate for the entire series
                    x = np.arange(len(prices))
                    y = prices

                    from scipy.stats import linregress
                    slope, intercept, r_value, p_value, std_err = linregress(x, y)
                    trend_strength = r_value ** 2

                    # Create array with same length as prices to match dimensions
                    trend_array = np.full(len(prices), trend_strength)
                    features_list.append(trend_array.reshape(-1, 1))

            # Trend direction
            trend_direction = np.diff(prices, prepend=prices[0])
            trend_direction = (trend_direction > 0).astype(float)
            features_list.append(trend_direction.reshape(-1, 1))

            return np.hstack(features_list) if features_list else np.zeros((len(prices), 1))

        except Exception as e:
            self.logger.warning(f"Trend feature calculation failed: {e}")
            return np.zeros((len(prices), 1))

    def _calculate_efficiency_features(self, returns: np.ndarray) -> np.ndarray:
        """Calculate market efficiency features."""
        try:
            features_list = []

            # Autocorrelation features
            for lag in [1, 5, 10, 20]:
                if len(returns) > lag:
                    autocorr = pd.Series(returns).shift(lag).rolling(window=20, min_periods=10).corr(pd.Series(returns))
                    autocorr = autocorr.fillna(0).values
                    features_list.append(autocorr.reshape(-1, 1))

            # Random walk test features
            variance_ratios = self._calculate_variance_ratios(returns)
            features_list.append(variance_ratios.reshape(-1, 1))

            return np.hstack(features_list) if features_list else np.zeros((len(returns), 1))

        except Exception as e:
            self.logger.warning(f"Efficiency feature calculation failed: {e}")
            return np.zeros((len(returns), 1))

    def _calculate_liquidity_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Calculate liquidity-based features."""
        try:
            features_list = []

            # Spread-based liquidity
            spreads = (market_data['high'] - market_data['low']) / market_data['close']
            spreads = spreads.fillna(spreads.mean()).values
            features_list.append(spreads.reshape(-1, 1))

            # Volume-based liquidity
            if 'volume' in market_data.columns:
                volume = market_data['volume'].values
                price_volume = market_data['close'] * volume
                price_volume = price_volume / price_volume.mean()  # Normalize
                features_list.append(price_volume.reshape(-1, 1))

            # Market impact estimation
            market_impact = self._calculate_market_impact(market_data)
            features_list.append(market_impact.reshape(-1, 1))

            return np.hstack(features_list) if features_list else np.zeros((len(market_data), 1))

        except Exception as e:
            self.logger.warning(f"Liquidity feature calculation failed: {e}")
            return np.zeros((len(market_data), 1))

    def _calculate_price_action_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Calculate price action features."""
        try:
            features_list = []

            # Candlestick patterns
            open_prices = market_data['open'].values
            high_prices = market_data['high'].values
            low_prices = market_data['low'].values
            close_prices = market_data['close'].values

            # Body size
            body_size = abs(close_prices - open_prices) / (high_prices - low_prices + 1e-8)
            features_list.append(body_size.reshape(-1, 1))

            # Shadow ratios
            upper_shadow = (high_prices - np.maximum(open_prices, close_prices)) / (high_prices - low_prices + 1e-8)
            lower_shadow = (np.minimum(open_prices, close_prices) - low_prices) / (high_prices - low_prices + 1e-8)
            features_list.append(upper_shadow.reshape(-1, 1))
            features_list.append(lower_shadow.reshape(-1, 1))

            # Price position within range
            price_position = (close_prices - low_prices) / (high_prices - low_prices + 1e-8)
            features_list.append(price_position.reshape(-1, 1))

            return np.hstack(features_list) if features_list else np.zeros((len(market_data), 1))

        except Exception as e:
            self.logger.warning(f"Price action feature calculation failed: {e}")
            return np.zeros((len(market_data), 1))

    def _classify_volatility_regime(self, returns: np.ndarray) -> np.ndarray:
        """Classify volatility regime."""
        try:
            volatility = pd.Series(np.abs(returns)).rolling(window=20, min_periods=10).std()
            volatility = volatility.fillna(volatility.mean()).values

            # Classify into volatility regimes
            low_vol_threshold = np.percentile(volatility, 25)
            high_vol_threshold = np.percentile(volatility, 75)

            regimes = np.zeros(len(returns))
            regimes[volatility <= low_vol_threshold] = 0  # Low volatility
            regimes[volatility >= high_vol_threshold] = 2  # High volatility
            regimes[(volatility > low_vol_threshold) & (volatility < high_vol_threshold)] = 1  # Medium volatility

            return regimes

        except Exception as e:
            self.logger.warning(f"Volatility regime classification failed: {e}")
            return np.ones(len(returns)) * 0.5

    def _calculate_variance_ratios(self, returns: np.ndarray) -> np.ndarray:
        """Calculate variance ratio test statistics."""
        try:
            n = len(returns)
            ratios = []

            for k in [2, 5, 10]:
                if n > k:
                    # Variance of returns vs variance of k-period returns
                    return_var = np.var(returns)
                    k_return_var = np.var([np.sum(returns[i:i+k]) for i in range(n - k)])

                    if return_var > 0:
                        ratio = k_return_var / (k * return_var)
                        ratios.append(ratio)

            return np.array(ratios) if ratios else np.array([1.0])

        except Exception as e:
            self.logger.warning(f"Variance ratio calculation failed: {e}")
            return np.array([1.0])

    def _calculate_market_impact(self, market_data: pd.DataFrame) -> np.ndarray:
        """Calculate market impact measures."""
        try:
            # Simplified market impact estimation
            returns = market_data['close'].pct_change().fillna(0).values

            if 'volume' in market_data.columns:
                volume = market_data['volume'].values
                # Amihud illiquidity measure
                illiquidity = np.abs(returns) / (market_data['close'] * volume + 1e-8)
                return illiquidity
            else:
                # Price impact approximation
                price_impact = np.abs(returns) * 100  # Scale to percentage
                return price_impact

        except Exception as e:
            self.logger.warning(f"Market impact calculation failed: {e}")
            return np.zeros(len(market_data))

    def _economic_kmeans(self, features: np.ndarray) -> Dict[str, Any]:
        """Economic-aware K-means clustering."""
        try:
            n_regimes = self.config.get('n_regimes', 8)

            # Apply economic distance metric if specified
            if self.economic_distance_metric == 'economic_euclidean':
                features = self._apply_economic_distance_transform(features)

            # Perform K-means
            kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)
            probabilities = self._calculate_cluster_probabilities(features, labels, kmeans)

            return {
                'labels': labels,
                'cluster_centers': kmeans.cluster_centers_,
                'probabilities': probabilities,
                'execution_time': 0.0
            }

        except Exception as e:
            self.logger.error(f"Economic K-means failed: {e}")
            raise

    def _economic_hierarchical(self, features: np.ndarray) -> Dict[str, Any]:
        """Economic-aware hierarchical clustering."""
        try:
            n_regimes = self.config.get('n_regimes', 8)

            # Apply economic distance metric
            if self.economic_distance_metric in ['economic_euclidean', 'economic_manhattan']:
                features = self._apply_economic_distance_transform(features)

            # Perform hierarchical clustering
            hierarchical = AgglomerativeClustering(n_clusters=n_regimes, linkage='ward')
            labels = hierarchical.fit_predict(features)
            probabilities = self._calculate_cluster_probabilities(features, labels)

            return {
                'labels': labels,
                'cluster_centers': np.array([]),  # Hierarchical doesn't provide centers
                'probabilities': probabilities,
                'execution_time': 0.0
            }

        except Exception as e:
            self.logger.error(f"Economic hierarchical failed: {e}")
            raise

    def _economic_gmm(self, features: np.ndarray) -> Dict[str, Any]:
        """Economic-aware Gaussian Mixture Model clustering."""
        try:
            n_regimes = self.config.get('n_regimes', 8)

            # Apply economic distance metric
            if self.economic_distance_metric == 'economic_euclidean':
                features = self._apply_economic_distance_transform(features)

            # Perform GMM
            gmm = GaussianMixture(n_components=n_regimes, random_state=42, n_init=5)
            labels = gmm.fit_predict(features)
            probabilities = gmm.predict_proba(features)

            return {
                'labels': labels,
                'cluster_centers': gmm.means_,
                'probabilities': probabilities,
                'execution_time': 0.0
            }

        except Exception as e:
            self.logger.error(f"Economic GMM failed: {e}")
            raise

    def _economic_adaptive(self, features: np.ndarray) -> Dict[str, Any]:
        """Adaptive economic clustering."""
        try:
            # Try different algorithms and choose best
            algorithms = ['economic_kmeans', 'economic_gmm', 'economic_hierarchical']
            best_score = -1
            best_result = None
            best_algorithm = None

            for algorithm in algorithms:
                try:
                    if algorithm in self.clustering_algorithms:
                        result = self.clustering_algorithms[algorithm](features)

                        # Calculate economic quality score
                        score = self._calculate_economic_quality_score(features, result['labels'])

                        if score > best_score:
                            best_score = score
                            best_result = result
                            best_algorithm = algorithm

                except Exception as e:
                    self.logger.warning(f"Algorithm {algorithm} failed: {e}")
                    continue

            if best_result is None:
                raise ValueError("All economic clustering algorithms failed")

            self.logger.info(f"Selected economic algorithm: {best_algorithm} (score: {best_score:.3f})")
            return best_result

        except Exception as e:
            self.logger.error(f"Economic adaptive clustering failed: {e}")
            # Fallback to standard K-means
            return self._economic_kmeans(features)

    def _apply_economic_distance_transform(self, features: np.ndarray) -> np.ndarray:
        """Apply economic distance transformation to features."""
        try:
            # Transform features based on economic significance
            # This gives higher weight to economically meaningful features
            transformed_features = features.copy()

            # Apply non-linear transformation to emphasize economic significance
            # Higher values indicate more economic significance
            economic_importance = np.mean(np.abs(transformed_features), axis=0)
            economic_importance = economic_importance / np.max(economic_importance)

            # Weight features by economic importance
            for i in range(features.shape[1]):
                transformed_features[:, i] *= (1 + economic_importance[i])

            return transformed_features

        except Exception as e:
            self.logger.warning(f"Economic distance transform failed: {e}")
            return features

    def _calculate_cluster_probabilities(self, features: np.ndarray, labels: np.ndarray, clusterer=None) -> np.ndarray:
        """Calculate cluster membership probabilities."""
        try:
            n_samples = len(features)
            n_clusters = len(set(labels))

            if n_clusters == 0:
                return np.ones((n_samples, 1)) * 0.5

            # For K-means, estimate probabilities based on distance to centroids
            if clusterer is not None and hasattr(clusterer, 'cluster_centers_'):
                centroids = clusterer.cluster_centers_
                probabilities = np.zeros((n_samples, n_clusters))

                for i, label in enumerate(labels):
                    if 0 <= label < n_clusters:
                        # Distance to assigned cluster
                        assigned_distance = np.linalg.norm(features[i] - centroids[label])

                        # Distances to all clusters
                        distances = np.linalg.norm(features[i] - centroids, axis=1)

                        # Convert distances to probabilities (closer = higher probability)
                        if np.min(distances) > 0:
                            probabilities[i] = 1 / (distances + 1e-8)
                            probabilities[i] /= np.sum(probabilities[i])
                        else:
                            probabilities[i, label] = 1.0
                    else:
                        probabilities[i] = 1.0 / n_clusters

                return probabilities
            else:
                # Uniform probabilities for other algorithms
                probabilities = np.ones((n_samples, n_clusters)) / n_clusters
                for i, label in enumerate(labels):
                    if 0 <= label < n_clusters:
                        probabilities[i] *= 0.7  # Higher probability for assigned cluster
                        probabilities[i, label] += 0.3  # Boost assigned cluster
                        probabilities[i] /= np.sum(probabilities[i])  # Renormalize

                return probabilities

        except Exception as e:
            self.logger.warning(f"Probability calculation failed: {e}")
            n_clusters = len(set(labels))
            return np.ones((len(features), n_clusters)) / n_clusters

    def _calculate_economic_quality_score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate quality score for economic clustering."""
        try:
            if len(set(labels)) < 2:
                return 0.0

            # Standard clustering metrics
            silhouette = silhouette_score(features, labels)
            ch_score = calinski_harabasz_score(features, labels)

            # Economic-specific metrics
            economic_separation = self._calculate_economic_separation(features, labels)
            economic_cohesion = self._calculate_economic_cohesion(features, labels)

            # Combine metrics
            score = (
                0.3 * max(0, min(1, silhouette)) +
                0.2 * min(ch_score / 1000, 1) +
                0.25 * economic_separation +
                0.25 * economic_cohesion
            )

            return score

        except Exception as e:
            self.logger.warning(f"Economic quality score calculation failed: {e}")
            return 0.5

    def _calculate_economic_separation(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate economic separation between clusters."""
        try:
            unique_labels = set(labels)
            centroids = []

            # Calculate centroids for each cluster
            for label in unique_labels:
                cluster_points = features[labels == label]
                if len(cluster_points) > 0:
                    centroid = np.mean(cluster_points, axis=0)
                    centroids.append(centroid)

            if len(centroids) < 2:
                return 0.0

            # Calculate average distance between centroids
            distances = []
            for i in range(len(centroids)):
                for j in range(i+1, len(centroids)):
                    distance = np.linalg.norm(centroids[i] - centroids[j])
                    distances.append(distance)

            avg_distance = np.mean(distances) if distances else 0
            max_possible_distance = np.sqrt(features.shape[1])  # Diagonal of feature space

            return min(avg_distance / max_possible_distance, 1.0)

        except Exception as e:
            self.logger.warning(f"Economic separation calculation failed: {e}")
            return 0.5

    def _calculate_economic_cohesion(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate economic cohesion within clusters."""
        try:
            unique_labels = set(labels)
            cohesions = []

            for label in unique_labels:
                cluster_points = features[labels == label]
                if len(cluster_points) > 1:
                    # Calculate average distance within cluster
                    distances = []
                    for i in range(len(cluster_points)):
                        for j in range(i+1, len(cluster_points)):
                            distance = np.linalg.norm(cluster_points[i] - cluster_points[j])
                            distances.append(distance)

                    if distances:
                        avg_distance = np.mean(distances)
                        cohesions.append(avg_distance)

            if cohesions:
                avg_cohesion = np.mean(cohesions)
                max_feature_range = np.max(features) - np.min(features)
                return max(0, 1 - avg_cohesion / max_feature_range)
            else:
                return 0.5

        except Exception as e:
            self.logger.warning(f"Economic cohesion calculation failed: {e}")
            return 0.5

    def _calculate_economic_quality_metrics(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive economic quality metrics."""
        try:
            metrics = {}

            # Standard metrics
            try:
                metrics['silhouette_score'] = silhouette_score(features, labels)
            except:
                metrics['silhouette_score'] = 0.0

            try:
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(features, labels)
            except:
                metrics['calinski_harabasz_score'] = 0.0

            # Economic-specific metrics
            metrics['economic_separation'] = self._calculate_economic_separation(features, labels)
            metrics['economic_cohesion'] = self._calculate_economic_cohesion(features, labels)

            # Regime balance
            regime_sizes = np.bincount(labels, minlength=len(set(labels)))
            metrics['regime_balance'] = 1.0 - (np.std(regime_sizes) / np.mean(regime_sizes)) if np.mean(regime_sizes) > 0 else 0

            # Economic significance score
            metrics['economic_significance_score'] = self._calculate_economic_quality_score(features, labels)

            return metrics

        except Exception as e:
            self.logger.warning(f"Economic quality metrics calculation failed: {e}")
            return {'economic_significance_score': 0.5}

    def _calculate_economic_metrics(self, market_data: pd.DataFrame, labels: np.ndarray, cluster_centers: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive economic metrics for each regime."""
        try:
            metrics = {
                'economic_significance': [],
                'momentum_scores': [],
                'volume_profiles': [],
                'regime_characteristics': {}
            }

            unique_labels = sorted(set(labels))

            for regime_id in unique_labels:
                regime_mask = labels == regime_id
                regime_data = market_data[regime_mask]

                if len(regime_data) < 10:
                    metrics['economic_significance'].append(0.5)
                    metrics['momentum_scores'].append(0.5)
                    metrics['volume_profiles'].append(0.5)
                    continue

                # Economic significance
                significance = self._calculate_regime_economic_significance(regime_data)
                metrics['economic_significance'].append(significance)

                # Momentum score
                momentum = self._calculate_regime_momentum_score(regime_data)
                metrics['momentum_scores'].append(momentum)

                # Volume profile
                volume_profile = self._calculate_regime_volume_profile(regime_data)
                metrics['volume_profiles'].append(volume_profile)

                # Store characteristics
                metrics['regime_characteristics'][f"regime_{regime_id}"] = {
                    'size': len(regime_data),
                    'significance': significance,
                    'momentum': momentum,
                    'volume_profile': volume_profile
                }

            return metrics

        except Exception as e:
            self.logger.warning(f"Economic metrics calculation failed: {e}")
            return {
                'economic_significance': np.full(len(set(labels)), 0.5),
                'momentum_scores': np.full(len(set(labels)), 0.5),
                'volume_profiles': np.full(len(set(labels)), 0.5)
            }

    def _calculate_regime_economic_significance(self, regime_data: pd.DataFrame) -> float:
        """Calculate economic significance for a single regime."""
        try:
            # Multiple factors contribute to economic significance
            significance_factors = []

            # Volatility significance
            returns = regime_data['close'].pct_change().dropna()
            volatility = np.std(returns) if len(returns) > 0 else 0.01
            vol_significance = min(volatility / 0.05, 1.0)  # Normalize to reasonable range
            significance_factors.append(vol_significance)

            # Trend significance
            prices = regime_data['close'].values
            if len(prices) > 20:
                from scipy.stats import linregress
                slope, intercept, r_value, p_value, std_err = linregress(np.arange(len(prices)), prices)
                trend_significance = min(abs(r_value) * 2.0, 1.0)  # Scale to 0-1 range
                significance_factors.append(trend_significance)

            # Volume significance if available
            if 'volume' in regime_data.columns:
                volume = regime_data['volume'].values
                avg_volume = np.mean(volume) if len(volume) > 0 else 1.0
                vol_factor = min(avg_volume / 1000000, 1.0)  # Normalize volume factor
                significance_factors.append(vol_factor)

            # Calculate overall significance
            if significance_factors:
                return np.mean(significance_factors)
            else:
                return 0.5  # Default neutral significance

        except Exception as e:
            self.logger.warning(f"Economic significance calculation failed: {e}")
            return 0.5
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

    def _calculate_regime_economic_significance(self, regime_data: pd.DataFrame) -> float:
        """Calculate economic significance for a single regime."""
        try:
            # Multiple factors contribute to economic significance
            significance_factors = []

            # Volatility significance
            returns = regime_data['close'].pct_change().dropna()
            volatility = np.std(returns) if len(returns) > 0 else 0.01
            vol_significance = min(volatility / 0.05, 1.0)  # Normalize to reasonable range
            significance_factors.append(vol_significance)

            # Trend significance
            prices = regime_data['close'].values
            if len(prices) > 20:
                from scipy.stats import linregress
                x = np.arange(len(prices))
                slope, intercept, r_value, p_value, std_err = linregress(x, prices)
                trend_significance = min(abs(r_value ** 2), 1.0)
                significance_factors.append(trend_significance)

            # Volume significance
            if 'volume' in regime_data.columns:
                volume = regime_data['volume'].values
                volume_volatility = np.std(volume) / np.mean(volume) if np.mean(volume) > 0 else 1.0
                volume_significance = min(volume_volatility / 2.0, 1.0)
                significance_factors.append(volume_significance)

            # Market efficiency significance
            if len(returns) > 10:
                autocorrelation = abs(returns.autocorr(lag=1))
                efficiency_significance = min(autocorrelation * 2, 1.0)  # Higher autocorrelation = more significant
                significance_factors.append(efficiency_significance)

            # Average significance
            avg_significance = np.mean(significance_factors) if significance_factors else 0.5

            return min(avg_significance, 1.0)

        except Exception as e:
            self.logger.warning(f"Regime economic significance calculation failed: {e}")
            return 0.5

    def _calculate_regime_momentum_score(self, regime_data: pd.DataFrame) -> float:
        """Calculate momentum score for a regime."""
        try:
            prices = regime_data['close'].values

            if len(prices) < 10:
                return 0.5

            # Calculate momentum across different periods
            momentum_scores = []

            for period in self.momentum_periods:
                if len(prices) > period:
                    # Price momentum
                    momentum = (prices[-1] - prices[0]) / (prices[0] + 1e-8)
                    normalized_momentum = np.tanh(momentum)  # Normalize to [-1, 1]
                    momentum_scores.append(abs(normalized_momentum))

            avg_momentum = np.mean(momentum_scores) if momentum_scores else 0.5
            return min(avg_momentum, 1.0)

        except Exception as e:
            self.logger.warning(f"Regime momentum score calculation failed: {e}")
            return 0.5

    def _calculate_regime_volume_profile(self, regime_data: pd.DataFrame) -> float:
        """Calculate volume profile for a regime."""
        try:
            if 'volume' not in regime_data.columns:
                return 0.5

            volume = regime_data['volume'].values

            if len(volume) < 10:
                return 0.5

            # Volume consistency
            volume_mean = np.mean(volume)
            volume_std = np.std(volume)
            volume_consistency = 1.0 - min(volume_std / volume_mean, 1.0) if volume_mean > 0 else 0.0

            # Volume trend
            volume_series = pd.Series(volume)
            volume_trend = volume_series.pct_change().fillna(0).values
            volume_trend_strength = np.mean(np.abs(volume_trend))

            # Combined volume profile score
            volume_profile = 0.6 * volume_consistency + 0.4 * min(volume_trend_strength, 1.0)

            return volume_profile

        except Exception as e:
            self.logger.warning(f"Regime volume profile calculation failed: {e}")
            return 0.5

    def _economic_frontier_analysis(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Perform economic frontier analysis."""
        try:
            self.logger.info("🔍 Performing economic frontier analysis...")

            frontiers = {}
            unique_labels = sorted(set(labels))

            # Calculate frontiers between economically significant cluster pairs
            for i, label_a in enumerate(unique_labels):
                for label_b in unique_labels[i+1:]:
                    frontier = self._calculate_economic_frontier(features, labels, label_a, label_b)
                    if frontier:
                        frontier_key = f"{label_a}_{label_b}"
                        frontiers[frontier_key] = frontier

            frontier_metrics = {
                'n_economic_frontiers': len(frontiers),
                'avg_economic_similarity': np.mean([f.get('similarity', 0.5) for f in frontiers.values()]),
                'frontier_boundaries': frontiers
            }

            self.logger.info(f"   Found {len(frontiers)} economic frontiers")
            return frontier_metrics

        except Exception as e:
            self.logger.warning(f"Economic frontier analysis failed: {e}")
            return {'n_economic_frontiers': 0}

    def _calculate_economic_frontier(self, features: np.ndarray, labels: np.ndarray, label_a: int, label_b: int) -> Optional[Dict[str, Any]]:
        """Calculate economic frontier between two clusters."""
        try:
            # Get points for each cluster
            points_a = features[labels == label_a]
            points_b = features[labels == label_b]

            if len(points_a) == 0 or len(points_b) == 0:
                return None

            # Calculate economic centroids
            economic_centroid_a = np.mean(points_a, axis=0)
            economic_centroid_b = np.mean(points_b, axis=0)

            # Find economic boundary points
            economic_distances = np.linalg.norm(points_a[:, np.newaxis] - points_b, axis=2)
            min_distances = np.min(economic_distances, axis=1)

            # Economic boundary threshold
            economic_threshold = np.percentile(min_distances, 10)

            # Calculate economic similarity
            economic_similarity = 1.0 / (1.0 + np.linalg.norm(economic_centroid_a - economic_centroid_b))

            return {
                'cluster_a': label_a,
                'cluster_b': label_b,
                'economic_similarity': economic_similarity,
                'economic_distance': np.linalg.norm(economic_centroid_a - economic_centroid_b),
                'boundary_threshold': economic_threshold
            }

        except Exception as e:
            self.logger.warning(f"Economic frontier calculation failed for {label_a}-{label_b}: {e}")
            return None

    def _economic_regime_transfer_optimization(self,
                                            features: np.ndarray,
                                            labels: np.ndarray,
                                            frontier_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Optimize regime transfers based on economic significance."""
        try:
            self.logger.info("🔍 Performing economic regime transfer optimization...")

            transfers = []
            unique_labels = sorted(set(labels))

            # Identify economically beneficial transfers
            for i, label_a in enumerate(unique_labels):
                for label_b in unique_labels[i+1:]:
                    transfer = self._evaluate_economic_transfer(
                        features, labels, label_a, label_b, frontier_metrics
                    )
                    if transfer:
                        transfers.append(transfer)

            # Sort by economic benefit
            transfers.sort(key=lambda x: x.get('economic_benefit', 0), reverse=True)

            self.logger.info(f"   Identified {len(transfers)} economic transfer opportunities")
            return transfers

        except Exception as e:
            self.logger.warning(f"Economic regime transfer optimization failed: {e}")
            return []

    def _evaluate_economic_transfer(self,
                                  features: np.ndarray,
                                  labels: np.ndarray,
                                  label_a: int,
                                  label_b: int,
                                  frontier_metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate economic benefit of transferring between regimes."""
        try:
            points_a = features[labels == label_a]
            points_b = features[labels == label_b]

            if len(points_a) == 0 or len(points_b) == 0:
                return None

            # Calculate economic centroids
            centroid_a = np.mean(points_a, axis=0)
            centroid_b = np.mean(points_b, axis=0)

            # Economic distance
            economic_distance = np.linalg.norm(centroid_a - centroid_b)

            # Economic similarity
            economic_similarity = 1.0 / (1.0 + economic_distance)

            # Frontier information
            frontier_key = f"{label_a}_{label_b}"
            frontier_info = frontier_metrics.get('frontier_boundaries', {}).get(frontier_key, {})

            # Economic benefit calculation
            size_benefit = min(len(points_a), len(points_b)) / max(len(points_a), len(points_b))
            economic_benefit = economic_similarity * size_benefit

            return {
                'regime_a': label_a,
                'regime_b': label_b,
                'economic_distance': economic_distance,
                'economic_similarity': economic_similarity,
                'economic_benefit': economic_benefit,
                'size_a': len(points_a),
                'size_b': len(points_b),
                'frontier_info': frontier_info
            }

        except Exception as e:
            self.logger.warning(f"Economic transfer evaluation failed for {label_a}-{label_b}: {e}")
            return None

def create_economic_clusterer(config: Dict[str, Any]) -> EconomicClusterer:
    """Create economic clusterer."""
    return EconomicClusterer(config)

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
