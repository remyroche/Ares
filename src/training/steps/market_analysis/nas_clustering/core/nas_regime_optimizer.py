"""
NAS Regime Optimizer for data-driven regime count determination.

This module provides automatic optimization of the number of regimes
based on data characteristics and quality metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import time

logger = logging.getLogger(__name__)


@dataclass
class RegimeOptimizationResult:
    """Result of regime optimization."""
    optimal_n_regimes: int
    optimization_scores: Dict[str, float]
    regime_quality_metrics: Dict[str, Any]
    data_characteristics: Dict[str, Any]
    optimization_method: str
    execution_time: float
    recommendations: List[str]


class NASRegimeOptimizer:
    """Optimizer for data-driven regime count determination."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize NAS regime optimizer.
        
        Args:
            config: Optimizer configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Optimization settings
        self.min_regimes = config.get('min_regimes', 5)
        self.max_regimes = config.get('max_regimes', 20)
        self.optimization_methods = config.get('optimization_methods', ['silhouette', 'calinski_harabasz', 'davies_bouldin', 'elbow'])
        self.quality_threshold = config.get('quality_threshold', 0.6)
        self.stability_threshold = config.get('stability_threshold', 0.7)
        
        # Data characteristics
        self.enable_data_analysis = config.get('enable_data_analysis', True)
        self.enable_volatility_analysis = config.get('enable_volatility_analysis', True)
        self.enable_trend_analysis = config.get('enable_trend_analysis', True)
        self.enable_volume_analysis = config.get('enable_volume_analysis', True)
        
        self.logger.info("✅ NAS Regime Optimizer initialized")
    
    def optimize_regime_count(self, features: np.ndarray, market_data: np.ndarray,
                            timestamps: np.ndarray, 
                            initial_n_regimes: Optional[int] = None) -> RegimeOptimizationResult:
        """Optimize the number of regimes based on data characteristics.
        
        Args:
            features: Feature matrix
            market_data: Market data array
            timestamps: Timestamps array
            initial_n_regimes: Initial number of regimes (if None, will be determined)
            
        Returns:
            RegimeOptimizationResult with optimal regime count
        """
        start_time = time.time()
        
        try:
            self.logger.info("🔍 Starting data-driven regime count optimization")
            
            # Analyze data characteristics
            data_characteristics = self._analyze_data_characteristics(
                features, market_data, timestamps
            )
            
            # Determine initial regime count if not provided
            if initial_n_regimes is None:
                initial_n_regimes = self._determine_initial_regime_count(data_characteristics)
            
            # Optimize regime count
            optimization_scores = self._optimize_regime_count(
                features, initial_n_regimes, data_characteristics
            )
            
            # Find optimal regime count
            optimal_n_regimes = self._find_optimal_regime_count(optimization_scores)
            
            # Calculate quality metrics for optimal regime count
            regime_quality_metrics = self._calculate_regime_quality_metrics(
                features, optimal_n_regimes
            )
            
            # Generate recommendations
            recommendations = self._generate_optimization_recommendations(
                optimal_n_regimes, optimization_scores, data_characteristics
            )
            
            execution_time = time.time() - start_time
            
            # Create result
            result = RegimeOptimizationResult(
                optimal_n_regimes=optimal_n_regimes,
                optimization_scores=optimization_scores,
                regime_quality_metrics=regime_quality_metrics,
                data_characteristics=data_characteristics,
                optimization_method='data_driven',
                execution_time=execution_time,
                recommendations=recommendations
            )
            
            self.logger.info(f"✅ Regime optimization completed: {optimal_n_regimes} regimes in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Regime optimization failed: {e}")
            return RegimeOptimizationResult(
                optimal_n_regimes=initial_n_regimes or 10,
                optimization_scores={},
                regime_quality_metrics={},
                data_characteristics={},
                optimization_method='data_driven',
                execution_time=execution_time,
                recommendations=[f"Optimization failed: {str(e)}"]
            )
    
    def _analyze_data_characteristics(self, features: np.ndarray, 
                                    market_data: np.ndarray, 
                                    timestamps: np.ndarray) -> Dict[str, Any]:
        """Analyze data characteristics to inform regime count optimization."""
        try:
            characteristics = {}
            
            # Basic data characteristics
            characteristics['n_samples'] = len(features)
            characteristics['n_features'] = features.shape[1]
            characteristics['data_density'] = len(features) / (features.shape[1] * 100)  # Samples per feature
            
            # Market data characteristics
            if market_data.shape[1] >= 4:
                close_prices = market_data[:, 3]
                high_prices = market_data[:, 1]
                low_prices = market_data[:, 2]
                
                # Price characteristics
                price_volatility = np.std(close_prices) / np.mean(close_prices)
                price_range = (np.max(high_prices) - np.min(low_prices)) / np.mean(close_prices)
                price_trend = np.polyfit(range(len(close_prices)), close_prices, 1)[0]
                
                characteristics['price_volatility'] = float(price_volatility)
                characteristics['price_range'] = float(price_range)
                characteristics['price_trend'] = float(price_trend)
                characteristics['price_complexity'] = self._calculate_price_complexity(close_prices)
                
                # Volatility analysis
                if self.enable_volatility_analysis:
                    volatility_characteristics = self._analyze_volatility_characteristics(close_prices)
                    characteristics.update(volatility_characteristics)
                
                # Trend analysis
                if self.enable_trend_analysis:
                    trend_characteristics = self._analyze_trend_characteristics(close_prices)
                    characteristics.update(trend_characteristics)
            
            # Volume characteristics
            if market_data.shape[1] >= 5:
                volumes = market_data[:, 4]
                volume_volatility = np.std(volumes) / np.mean(volumes)
                volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
                
                characteristics['volume_volatility'] = float(volume_volatility)
                characteristics['volume_trend'] = float(volume_trend)
                characteristics['volume_complexity'] = self._calculate_volume_complexity(volumes)
                
                # Volume analysis
                if self.enable_volume_analysis:
                    volume_characteristics = self._analyze_volume_characteristics(volumes)
                    characteristics.update(volume_characteristics)
            
            # Feature characteristics
            feature_characteristics = self._analyze_feature_characteristics(features)
            characteristics.update(feature_characteristics)
            
            # Temporal characteristics
            temporal_characteristics = self._analyze_temporal_characteristics(timestamps)
            characteristics.update(temporal_characteristics)
            
            return characteristics
            
        except Exception as e:
            self.logger.warning(f"⚠️ Data characteristics analysis failed: {e}")
            return {}
    
    def _calculate_price_complexity(self, prices: np.ndarray) -> float:
        """Calculate price complexity score."""
        try:
            # Calculate price changes
            price_changes = np.diff(prices)
            
            # Calculate complexity based on change frequency and magnitude
            change_frequency = np.sum(np.abs(price_changes) > np.std(price_changes)) / len(price_changes)
            change_magnitude = np.std(price_changes) / np.mean(np.abs(price_changes))
            
            complexity = (change_frequency + change_magnitude) / 2.0
            return float(complexity)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Price complexity calculation failed: {e}")
            return 0.5
    
    def _calculate_volume_complexity(self, volumes: np.ndarray) -> float:
        """Calculate volume complexity score."""
        try:
            # Calculate volume changes
            volume_changes = np.diff(volumes)
            
            # Calculate complexity based on change frequency and magnitude
            change_frequency = np.sum(np.abs(volume_changes) > np.std(volume_changes)) / len(volume_changes)
            change_magnitude = np.std(volume_changes) / np.mean(np.abs(volume_changes))
            
            complexity = (change_frequency + change_magnitude) / 2.0
            return float(complexity)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Volume complexity calculation failed: {e}")
            return 0.5
    
    def _analyze_volatility_characteristics(self, prices: np.ndarray) -> Dict[str, Any]:
        """Analyze volatility characteristics."""
        try:
            # Calculate rolling volatility
            window_size = min(20, len(prices) // 4)
            if window_size < 5:
                return {}
            
            rolling_volatility = []
            for i in range(window_size, len(prices)):
                window_prices = prices[i-window_size:i]
                volatility = np.std(window_prices) / np.mean(window_prices)
                rolling_volatility.append(volatility)
            
            if not rolling_volatility:
                return {}
            
            rolling_volatility = np.array(rolling_volatility)
            
            return {
                'volatility_mean': float(np.mean(rolling_volatility)),
                'volatility_std': float(np.std(rolling_volatility)),
                'volatility_trend': float(np.polyfit(range(len(rolling_volatility)), rolling_volatility, 1)[0]),
                'volatility_clusters': self._estimate_volatility_clusters(rolling_volatility)
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Volatility characteristics analysis failed: {e}")
            return {}
    
    def _analyze_trend_characteristics(self, prices: np.ndarray) -> Dict[str, Any]:
        """Analyze trend characteristics."""
        try:
            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(prices)
            
            # Calculate trend changes
            trend_changes = self._calculate_trend_changes(prices)
            
            # Calculate trend persistence
            trend_persistence = self._calculate_trend_persistence(prices)
            
            return {
                'trend_strength': float(trend_strength),
                'trend_changes': int(trend_changes),
                'trend_persistence': float(trend_persistence),
                'trend_complexity': float(trend_changes / len(prices))
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Trend characteristics analysis failed: {e}")
            return {}
    
    def _analyze_volume_characteristics(self, volumes: np.ndarray) -> Dict[str, Any]:
        """Analyze volume characteristics."""
        try:
            # Calculate volume spikes
            volume_spikes = self._calculate_volume_spikes(volumes)
            
            # Calculate volume patterns
            volume_patterns = self._calculate_volume_patterns(volumes)
            
            return {
                'volume_spikes': int(volume_spikes),
                'volume_patterns': int(volume_patterns),
                'volume_irregularity': float(volume_spikes / len(volumes))
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Volume characteristics analysis failed: {e}")
            return {}
    
    def _analyze_feature_characteristics(self, features: np.ndarray) -> Dict[str, Any]:
        """Analyze feature characteristics."""
        try:
            # Calculate feature variance
            feature_variance = np.var(features, axis=0)
            
            # Calculate feature correlation
            feature_correlation = np.corrcoef(features.T)
            correlation_strength = np.mean(np.abs(feature_correlation))
            
            # Calculate feature dimensionality
            feature_dimensionality = self._calculate_feature_dimensionality(features)
            
            return {
                'feature_variance_mean': float(np.mean(feature_variance)),
                'feature_variance_std': float(np.std(feature_variance)),
                'feature_correlation': float(correlation_strength),
                'feature_dimensionality': float(feature_dimensionality)
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature characteristics analysis failed: {e}")
            return {}
    
    def _analyze_temporal_characteristics(self, timestamps: np.ndarray) -> Dict[str, Any]:
        """Analyze temporal characteristics."""
        try:
            # Calculate time intervals
            if len(timestamps) > 1:
                time_intervals = np.diff(timestamps)
                mean_interval = np.mean(time_intervals)
                std_interval = np.std(time_intervals)
                
                return {
                    'mean_time_interval': float(mean_interval),
                    'std_time_interval': float(std_interval),
                    'temporal_regularity': float(1.0 - (std_interval / mean_interval)) if mean_interval > 0 else 0.0
                }
            else:
                return {}
                
        except Exception as e:
            self.logger.warning(f"⚠️ Temporal characteristics analysis failed: {e}")
            return {}
    
    def _estimate_volatility_clusters(self, volatility: np.ndarray) -> int:
        """Estimate number of volatility clusters."""
        try:
            if len(volatility) < 3:
                return 1
            
            # Use simple clustering to estimate volatility regimes
            from sklearn.cluster import KMeans
            
            # Try different numbers of clusters
            best_score = -1
            best_n_clusters = 1
            
            for n_clusters in range(2, min(6, len(volatility) // 2)):
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    labels = kmeans.fit_predict(volatility.reshape(-1, 1))
                    score = silhouette_score(volatility.reshape(-1, 1), labels)
                    
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
                        
                except Exception:
                    continue
            
            return best_n_clusters
            
        except Exception as e:
            self.logger.warning(f"⚠️ Volatility clusters estimation failed: {e}")
            return 1
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate trend strength."""
        try:
            if len(prices) < 3:
                return 0.0
            
            # Calculate linear trend
            x = np.arange(len(prices))
            slope, _ = np.polyfit(x, prices, 1)
            
            # Normalize by price level
            trend_strength = abs(slope) / np.mean(prices)
            return float(trend_strength)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Trend strength calculation failed: {e}")
            return 0.0
    
    def _calculate_trend_changes(self, prices: np.ndarray) -> int:
        """Calculate number of trend changes."""
        try:
            if len(prices) < 3:
                return 0
            
            # Calculate price changes
            price_changes = np.diff(prices)
            
            # Count sign changes
            sign_changes = np.sum(np.diff(np.sign(price_changes)) != 0)
            return int(sign_changes)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Trend changes calculation failed: {e}")
            return 0
    
    def _calculate_trend_persistence(self, prices: np.ndarray) -> float:
        """Calculate trend persistence."""
        try:
            if len(prices) < 3:
                return 0.0
            
            # Calculate price changes
            price_changes = np.diff(prices)
            
            # Calculate autocorrelation
            if len(price_changes) > 1:
                autocorr = np.corrcoef(price_changes[:-1], price_changes[1:])[0, 1]
                return float(autocorr) if not np.isnan(autocorr) else 0.0
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"⚠️ Trend persistence calculation failed: {e}")
            return 0.0
    
    def _calculate_volume_spikes(self, volumes: np.ndarray) -> int:
        """Calculate number of volume spikes."""
        try:
            if len(volumes) < 3:
                return 0
            
            # Calculate volume threshold
            volume_threshold = np.mean(volumes) + 2 * np.std(volumes)
            
            # Count spikes
            spikes = np.sum(volumes > volume_threshold)
            return int(spikes)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Volume spikes calculation failed: {e}")
            return 0
    
    def _calculate_volume_patterns(self, volumes: np.ndarray) -> int:
        """Calculate number of volume patterns."""
        try:
            if len(volumes) < 5:
                return 0
            
            # Calculate volume changes
            volume_changes = np.diff(volumes)
            
            # Count pattern changes
            pattern_changes = np.sum(np.diff(np.sign(volume_changes)) != 0)
            return int(pattern_changes)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Volume patterns calculation failed: {e}")
            return 0
    
    def _calculate_feature_dimensionality(self, features: np.ndarray) -> float:
        """Calculate feature dimensionality."""
        try:
            # Calculate feature variance
            feature_variance = np.var(features, axis=0)
            
            # Calculate effective dimensionality
            total_variance = np.sum(feature_variance)
            if total_variance == 0:
                return 0.0
            
            # Calculate variance ratio
            variance_ratio = feature_variance / total_variance
            
            # Calculate effective dimensionality
            effective_dimensionality = np.sum(variance_ratio > 0.01)  # Features with >1% variance
            return float(effective_dimensionality)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature dimensionality calculation failed: {e}")
            return 0.0
    
    def _determine_initial_regime_count(self, data_characteristics: Dict[str, Any]) -> int:
        """Determine initial regime count based on data characteristics."""
        try:
            # Base regime count
            base_regimes = 8
            
            # Adjust based on data complexity
            if data_characteristics.get('price_complexity', 0.5) > 0.7:
                base_regimes += 2
            
            if data_characteristics.get('volume_complexity', 0.5) > 0.7:
                base_regimes += 1
            
            if data_characteristics.get('volatility_clusters', 1) > 2:
                base_regimes += 1
            
            if data_characteristics.get('trend_changes', 0) > 10:
                base_regimes += 1
            
            # Adjust based on data size
            n_samples = data_characteristics.get('n_samples', 1000)
            if n_samples > 2000:
                base_regimes += 1
            elif n_samples < 500:
                base_regimes = max(5, base_regimes - 1)
            
            # Ensure within bounds
            initial_regimes = max(self.min_regimes, min(self.max_regimes, base_regimes))
            
            self.logger.info(f"📊 Initial regime count determined: {initial_regimes}")
            return initial_regimes
            
        except Exception as e:
            self.logger.warning(f"⚠️ Initial regime count determination failed: {e}")
            return 10
    
    def _optimize_regime_count(self, features: np.ndarray, initial_n_regimes: int,
                             data_characteristics: Dict[str, Any]) -> Dict[str, float]:
        """Optimize regime count using multiple methods."""
        try:
            optimization_scores = {}
            
            # Test different numbers of regimes
            regime_range = range(
                max(self.min_regimes, initial_n_regimes - 3),
                min(self.max_regimes + 1, initial_n_regimes + 4)
            )
            
            for n_regimes in regime_range:
                scores = self._calculate_regime_scores(features, n_regimes)
                optimization_scores[f'n_regimes_{n_regimes}'] = scores
            
            return optimization_scores
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime count optimization failed: {e}")
            return {}
    
    def _calculate_regime_scores(self, features: np.ndarray, n_regimes: int) -> Dict[str, float]:
        """Calculate scores for a specific number of regimes."""
        try:
            scores = {}
            
            # Normalize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Try different clustering methods
            clustering_methods = ['kmeans', 'agglomerative']
            
            for method in clustering_methods:
                try:
                    if method == 'kmeans':
                        clusterer = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
                    elif method == 'agglomerative':
                        clusterer = AgglomerativeClustering(n_clusters=n_regimes)
                    else:
                        continue
                    
                    labels = clusterer.fit_predict(features_scaled)
                    
                    # Calculate quality metrics
                    if len(np.unique(labels)) > 1:
                        silhouette = silhouette_score(features_scaled, labels)
                        calinski_harabasz = calinski_harabasz_score(features_scaled, labels)
                        davies_bouldin = davies_bouldin_score(features_scaled, labels)
                        
                        # Combined score
                        combined_score = (silhouette + (calinski_harabasz / 1000) + (1 - davies_bouldin)) / 3.0
                        
                        scores[f'{method}_silhouette'] = float(silhouette)
                        scores[f'{method}_calinski_harabasz'] = float(calinski_harabasz)
                        scores[f'{method}_davies_bouldin'] = float(davies_bouldin)
                        scores[f'{method}_combined'] = float(combined_score)
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Clustering method {method} failed: {e}")
                    continue
            
            return scores
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime scores calculation failed: {e}")
            return {}
    
    def _find_optimal_regime_count(self, optimization_scores: Dict[str, float]) -> int:
        """Find optimal regime count based on optimization scores."""
        try:
            if not optimization_scores:
                return 10
            
            # Find best scores for each number of regimes
            regime_scores = {}
            
            for key, score in optimization_scores.items():
                if key.startswith('n_regimes_'):
                    n_regimes = int(key.split('_')[-1])
                    if n_regimes not in regime_scores:
                        regime_scores[n_regimes] = []
                    regime_scores[n_regimes].append(score)
            
            # Calculate average scores
            avg_scores = {}
            for n_regimes, scores in regime_scores.items():
                avg_scores[n_regimes] = np.mean(scores)
            
            # Find optimal regime count
            if avg_scores:
                optimal_n_regimes = max(avg_scores.keys(), key=lambda x: avg_scores[x])
                return optimal_n_regimes
            else:
                return 10
                
        except Exception as e:
            self.logger.warning(f"⚠️ Optimal regime count finding failed: {e}")
            return 10
    
    def _calculate_regime_quality_metrics(self, features: np.ndarray, n_regimes: int) -> Dict[str, Any]:
        """Calculate quality metrics for optimal regime count."""
        try:
            # Normalize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Use K-means for final clustering
            clusterer = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
            labels = clusterer.fit_predict(features_scaled)
            
            # Calculate quality metrics
            if len(np.unique(labels)) > 1:
                silhouette = silhouette_score(features_scaled, labels)
                calinski_harabasz = calinski_harabasz_score(features_scaled, labels)
                davies_bouldin = davies_bouldin_score(features_scaled, labels)
                
                return {
                    'silhouette_score': float(silhouette),
                    'calinski_harabasz_score': float(calinski_harabasz),
                    'davies_bouldin_score': float(davies_bouldin),
                    'n_regimes': n_regimes,
                    'n_samples': len(features),
                    'n_features': features.shape[1]
                }
            else:
                return {
                    'silhouette_score': 0.0,
                    'calinski_harabasz_score': 0.0,
                    'davies_bouldin_score': 0.0,
                    'n_regimes': n_regimes,
                    'n_samples': len(features),
                    'n_features': features.shape[1]
                }
                
        except Exception as e:
            self.logger.warning(f"⚠️ Regime quality metrics calculation failed: {e}")
            return {}
    
    def _generate_optimization_recommendations(self, optimal_n_regimes: int,
                                            optimization_scores: Dict[str, float],
                                            data_characteristics: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations."""
        try:
            recommendations = []
            
            # Regime count recommendations
            if optimal_n_regimes < 8:
                recommendations.append("Consider increasing regime count for better market state differentiation")
            elif optimal_n_regimes > 15:
                recommendations.append("Consider reducing regime count to avoid over-segmentation")
            
            # Data quality recommendations
            if data_characteristics.get('price_complexity', 0.5) < 0.3:
                recommendations.append("Low price complexity detected - consider enhancing feature extraction")
            
            if data_characteristics.get('volatility_clusters', 1) < 2:
                recommendations.append("Low volatility variation detected - consider longer timeframes")
            
            if data_characteristics.get('n_samples', 1000) < 500:
                recommendations.append("Limited data samples - consider collecting more data for better regime detection")
            
            # Quality recommendations
            if optimization_scores:
                best_score = max(optimization_scores.values())
                if best_score < 0.5:
                    recommendations.append("Low clustering quality - consider feature engineering or different clustering methods")
            
            return recommendations
            
        except Exception as e:
            self.logger.warning(f"⚠️ Optimization recommendations generation failed: {e}")
            return []