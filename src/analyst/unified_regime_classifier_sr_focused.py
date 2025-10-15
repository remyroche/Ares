"""
from ...utils.logger import system_logger
import warnings
S/R-Focused Unified Regime Classifier

This version prioritizes:
1. Advanced S/R level detection using price differences/returns
2. S/R relevance scoring based on market behavior
3. Distance metrics as secondary features
"""
from datetime import datetime
from typing import Any, List, Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from ...utils.logger import system_logger
import logging
import pandas as pd
from .core.decorators.validation import validates as validate_data_quality, traced as with_tracing_span
import numpy as np
import time

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

class UnifiedRegimeClassifierSRFocused:
    """
    S/R-Focused Classifier
    
    Core focus: Detecting S/R levels and scoring their relevance
    using price differences, returns, and market behavior patterns.
    """

    def __init__(self, config: dict[str, Any], exchange: str='UNKNOWN', symbol: str='UNKNOWN') -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config.get('analyst', {}).get('unified_regime_classifier', {})
        self.global_config = config
        self.logger = system_logger.getChild('UnifiedRegimeClassifierSRFocused')
        self.exchange = exchange
        self.symbol = symbol
        self.sr_detection_config = {'min_return_reversal': 0.002, 'return_lookback': 100, 'min_touches': 2, 'touch_tolerance': 0.001, 'return_significance': 0.01, 'relevance_weights': {'return_magnitude': 0.3, 'touch_count': 0.2, 'recency': 0.2, 'volume_confirmation': 0.15, 'success_rate': 0.15}, 'return_windows': [5, 10, 20, 50], 'volatility_adjusted': True}
        self.fractal_timeframes = self.config.get('fractal_timeframes', [{'name': '1h', 'weight': 0.25, 'min_touches': 2}, {'name': '4h', 'weight': 0.35, 'min_touches': 3}, {'name': '1d', 'weight': 0.4, 'min_touches': 4}])
        self.distance_normalization = 'returns'
        self.scaler = StandardScaler()
        self.sr_levels_cache = {}

    async def initialize(self) -> bool:
        """Initialize the S/R-focused classifier."""
        try:
            self.logger.info('Initializing S/R-Focused Classifier...')
            self.logger.info('✅ S/R-Focused Classifier initialized successfully')
            return True
        except Exception as e:
            self.logger.error(f'Failed to initialize classifier: {e}')
            return False

    async def classify_location(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Classify location with primary focus on S/R detection and relevance.
        """
        if features_df.empty or len(features_df) < 100:
            return self._get_default_classification()
        try:
            returns_data = self._calculate_returns_data(features_df)
            sr_levels = self._detect_sr_levels_using_returns(features_df, returns_data)
            sr_relevance = self._score_sr_relevance(sr_levels, features_df, returns_data)
            current_price = features_df['close'].iloc[-1]
            relevant_support, relevant_resistance = self._get_most_relevant_sr(sr_relevance, current_price)
            location_metrics = self._calculate_location_metrics_returns_based(current_price, relevant_support, relevant_resistance, returns_data)
            location_metrics['sr_analysis'] = {'detected_levels': sr_levels, 'relevance_scores': sr_relevance, 'most_relevant_support': relevant_support, 'most_relevant_resistance': relevant_resistance, 'total_sr_levels': len(sr_levels), 'high_relevance_count': sum((1 for s in sr_relevance.values() if s['total_score'] > 0.7))}
            location_metrics['timestamp'] = datetime.now().isoformat()
            location_metrics['current_price'] = current_price
            return location_metrics
        except Exception as e:
            self.logger.error(f'Error in S/R classification: {e}')
            return self._get_default_classification()

    def _calculate_returns_data(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate various return metrics for S/R detection."""
        returns_data = {}
        returns_data['returns'] = df['close'].pct_change()
        returns_data['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        for window in self.sr_detection_config['return_windows']:
            returns_data[f'returns_{window}'] = df['close'].pct_change(window)
        returns_data['return_volatility'] = returns_data['returns'].rolling(20).std()
        returns_data['normalized_returns'] = returns_data['returns'] / (returns_data['return_volatility'] + 1e-08)
        returns_data['return_momentum'] = returns_data['returns'].rolling(10).mean()
        returns_data['cumulative_returns'] = (1 + returns_data['returns']).cumprod() - 1
        return returns_data

    def _detect_sr_levels_using_returns(self, df: pd.DataFrame, returns_data: Dict[str, pd.Series]) -> List[Dict[str, Any]]:
        """
        Detect S/R levels using return reversals and price behavior.
        This is the PRIMARY FOCUS of the classifier.
        """
        sr_levels = []
        prices = df['close'].values
        returns = returns_data['returns'].values
        norm_returns = returns_data['normalized_returns'].values
        reversal_points = self._find_return_reversals(prices, returns, norm_returns)
        tested_levels = self._find_tested_levels_by_returns(df, returns_data)
        return_clusters = self._find_return_clusters(df, returns_data)
        volume_reversals = self._find_volume_weighted_reversals(df, returns_data)
        all_levels = reversal_points + tested_levels + return_clusters + volume_reversals
        clustered_levels = self._cluster_sr_levels(all_levels)
        validated_levels = []
        for level in clustered_levels:
            validation = self._validate_sr_level_with_returns(level, df, returns_data)
            if validation['is_valid']:
                level['validation'] = validation
                validated_levels.append(level)
        return validated_levels

    def _find_return_reversals(self, prices: np.ndarray, returns: np.ndarray, norm_returns: np.ndarray) -> List[Dict[str, Any]]:
        """Find S/R levels from significant return reversals."""
        levels = []
        min_reversal = self.sr_detection_config['min_return_reversal']
        for i in range(2, len(returns) - 2):
            if returns[i - 1] < -min_reversal and returns[i] > min_reversal and (norm_returns[i - 1] < -1) and (norm_returns[i] > 1):
                levels.append({'price': prices[i], 'type': 'support', 'method': 'return_reversal', 'return_magnitude': abs(returns[i] - returns[i - 1]), 'normalized_magnitude': abs(norm_returns[i] - norm_returns[i - 1]), 'index': i, 'touches': 1})
            elif returns[i - 1] > min_reversal and returns[i] < -min_reversal and (norm_returns[i - 1] > 1) and (norm_returns[i] < -1):
                levels.append({'price': prices[i], 'type': 'resistance', 'method': 'return_reversal', 'return_magnitude': abs(returns[i] - returns[i - 1]), 'normalized_magnitude': abs(norm_returns[i] - norm_returns[i - 1]), 'index': i, 'touches': 1})
        return levels

    def _find_tested_levels_by_returns(self, df: pd.DataFrame, returns_data: Dict[str, pd.Series]) -> List[Dict[str, Any]]:
        """Find levels that have been tested multiple times based on returns."""
        levels = []
        prices = df['close'].values
        returns = returns_data['returns'].values
        tolerance = self.sr_detection_config['touch_tolerance']
        price_levels = {}
        for i in range(len(prices)):
            level_key = round(prices[i] / tolerance) * tolerance
            if level_key not in price_levels:
                price_levels[level_key] = {'indices': [], 'returns_at_touch': [], 'return_reversals': 0}
            price_levels[level_key]['indices'].append(i)
            price_levels[level_key]['returns_at_touch'].append(returns[i])
            if i > 0 and i < len(returns) - 1:
                if returns[i - 1] * returns[i + 1] < 0 and abs(returns[i]) > self.sr_detection_config['min_return_reversal']:
                    price_levels[level_key]['return_reversals'] += 1
        min_touches = self.sr_detection_config['min_touches']
        for level_price, level_data in price_levels.items():
            if len(level_data['indices']) >= min_touches and level_data['return_reversals'] >= min_touches // 2:
                avg_return_after = np.mean([returns[min(i + 1, len(returns) - 1)] for i in level_data['indices']])
                level_type = 'support' if avg_return_after > 0 else 'resistance'
                levels.append({'price': level_price, 'type': level_type, 'method': 'multiple_tests', 'touches': len(level_data['indices']), 'return_reversals': level_data['return_reversals'], 'avg_return_magnitude': np.mean(np.abs(level_data['returns_at_touch'])), 'test_indices': level_data['indices']})
        return levels

    def _find_return_clusters(self, df: pd.DataFrame, returns_data: Dict[str, pd.Series]) -> List[Dict[str, Any]]:
        """Find S/R levels using statistical clustering of returns."""
        levels = []
        prices = df['close'].values
        returns = returns_data['normalized_returns'].values
        window = 20
        for i in range(window, len(prices) - window):
            window_returns = returns[i - window // 2:i + window // 2]
            window_prices = prices[i - window // 2:i + window // 2]
            return_std = np.std(window_returns)
            return_mean = np.mean(window_returns)
            if return_std < 0.5 and abs(return_mean) < 0.1:
                price_counts = pd.Series(window_prices).value_counts()
                if len(price_counts) > 0:
                    modal_price = price_counts.index[0]
                    future_returns = returns[i:min(i + 10, len(returns))]
                    if len(future_returns) > 0:
                        breakout_direction = np.sign(np.sum(future_returns))
                        levels.append({'price': modal_price, 'type': 'resistance' if breakout_direction < 0 else 'support', 'method': 'return_cluster', 'cluster_size': len(window_returns), 'return_stability': 1 / (return_std + 0.1), 'index': i})
        return levels

    def _find_volume_weighted_reversals(self, df: pd.DataFrame, returns_data: Dict[str, pd.Series]) -> List[Dict[str, Any]]:
        """Find S/R levels using volume-weighted return reversals."""
        levels = []
        prices = df['close'].values
        volumes = df['volume'].values
        returns = returns_data['returns'].values
        vw_returns = returns * (volumes / np.mean(volumes))
        for i in range(2, len(vw_returns) - 2):
            if abs(vw_returns[i]) > 2 * np.std(vw_returns):
                if np.sign(vw_returns[i - 1]) != np.sign(vw_returns[i]):
                    level_type = 'support' if vw_returns[i] > 0 else 'resistance'
                    levels.append({'price': prices[i], 'type': level_type, 'method': 'volume_reversal', 'volume_ratio': volumes[i] / np.mean(volumes), 'vw_return': vw_returns[i], 'return_magnitude': abs(returns[i]), 'index': i})
        return levels

    def _cluster_sr_levels(self, levels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cluster nearby S/R levels."""
        if not levels:
            return []
        sorted_levels = sorted(levels, key = lambda x: x['price'])
        clusters = []
        current_cluster = [sorted_levels[0]]
        tolerance = self.sr_detection_config['touch_tolerance']
        for level in sorted_levels[1:]:
            cluster_center = np.mean([l['price'] for l in current_cluster])
            if abs(level['price'] - cluster_center) / cluster_center <= tolerance:
                current_cluster.append(level)
            else:
                merged_level = self._merge_level_cluster(current_cluster)
                clusters.append(merged_level)
                current_cluster = [level]
        if current_cluster:
            clusters.append(self._merge_level_cluster(current_cluster))
        return clusters

    def _merge_level_cluster(self, cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge levels in a cluster, preserving return information."""
        weights = [l.get('return_magnitude', 1) for l in cluster]
        total_weight = sum(weights)
        if total_weight > 0:
            weighted_price = sum((l['price'] * w for l, w in zip(cluster, weights))) / total_weight
        else:
            weighted_price = np.mean([l['price'] for l in cluster])
        support_count = sum((1 for l in cluster if l['type'] == 'support'))
        resistance_count = len(cluster) - support_count
        merged_level = {'price': weighted_price, 'type': 'support' if support_count > resistance_count else 'resistance', 'methods': list(set((l['method'] for l in cluster))), 'cluster_size': len(cluster), 'total_touches': sum((l.get('touches', 1) for l in cluster)), 'avg_return_magnitude': np.mean([l.get('return_magnitude', 0) for l in cluster]), 'max_return_magnitude': max([l.get('return_magnitude', 0) for l in cluster]), 'source_levels': cluster}
        return merged_level

    def _validate_sr_level_with_returns(self, level: Dict[str, Any], df: pd.DataFrame, returns_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Validate S/R level using return behavior."""
        validation = {'is_valid': False, 'success_rate': 0.0, 'avg_reversal_magnitude': 0.0, 'recent_test': False}
        prices = df['close'].values
        returns = returns_data['returns'].values
        level_price = level['price']
        tolerance = self.sr_detection_config['touch_tolerance']
        touches = []
        for i in range(len(prices)):
            if abs(prices[i] - level_price) / level_price <= tolerance:
                touches.append(i)
        if len(touches) < self.sr_detection_config['min_touches']:
            return validation
        successful_reversals = 0
        reversal_magnitudes = []
        for touch_idx in touches:
            if touch_idx < len(returns) - 5:
                future_returns = returns[touch_idx:touch_idx + 5]
                if level['type'] == 'support':
                    if np.mean(future_returns) > 0 and max(future_returns) > self.sr_detection_config['min_return_reversal']:
                        successful_reversals += 1
                        reversal_magnitudes.append(max(future_returns))
                elif np.mean(future_returns) < 0 and min(future_returns) < -self.sr_detection_config['min_return_reversal']:
                    successful_reversals += 1
                    reversal_magnitudes.append(abs(min(future_returns)))
        validation['success_rate'] = successful_reversals / len(touches) if touches else 0
        validation['avg_reversal_magnitude'] = np.mean(reversal_magnitudes) if reversal_magnitudes else 0
        validation['recent_test'] = any((t > len(prices) - 50 for t in touches))
        validation['total_tests'] = len(touches)
        validation['successful_tests'] = successful_reversals
        validation['is_valid'] = validation['success_rate'] >= 0.5 and validation['avg_reversal_magnitude'] >= self.sr_detection_config['min_return_reversal']
        return validation

    def _score_sr_relevance(self, sr_levels: List[Dict[str, Any]], df: pd.DataFrame, returns_data: Dict[str, pd.Series]) -> Dict[float, Dict[str, float]]:
        """Score S/R relevance based on return behavior and market context."""
        relevance_scores = {}
        current_idx = len(df) - 1
        weights = self.sr_detection_config['relevance_weights']
        for level in sr_levels:
            level_price = level['price']
            scores = {}
            avg_return_mag = level.get('avg_return_magnitude', 0)
            max_return_mag = level.get('max_return_magnitude', 0)
            scores['return_magnitude'] = min(1.0, max(avg_return_mag, max_return_mag) / 0.05)
            touch_count = level.get('total_touches', 1)
            scores['touch_count'] = min(1.0, touch_count / 10)
            if 'source_levels' in level:
                recent_indices = []
                for src in level['source_levels']:
                    if 'index' in src:
                        recent_indices.append(src['index'])
                    elif 'test_indices' in src:
                        recent_indices.extend(src['test_indices'])
                if recent_indices:
                    most_recent = max(recent_indices)
                    recency = (most_recent - current_idx + len(df)) / len(df)
                    scores['recency'] = 1.0 - min(1.0, recency)
                else:
                    scores['recency'] = 0.0
            else:
                scores['recency'] = 0.5
            volume_score = 0.0
            if 'volume_ratio' in level:
                volume_score = min(1.0, level['volume_ratio'] / 2)
            elif 'source_levels' in level:
                volume_ratios = [l.get('volume_ratio', 1) for l in level['source_levels']]
                if volume_ratios:
                    volume_score = min(1.0, np.mean(volume_ratios) / 2)
            scores['volume_confirmation'] = volume_score
            if 'validation' in level:
                scores['success_rate'] = level['validation']['success_rate']
            else:
                scores['success_rate'] = 0.5
            total_score = sum((scores[k] * weights[k] for k in weights.keys()))
            relevance_scores[level_price] = {**scores, 'total_score': total_score, 'level_type': level['type'], 'level_data': level}
        return relevance_scores

    def _get_most_relevant_sr(self, sr_relevance: Dict[float, Dict[str, float]], current_price: float) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Get the most relevant support and resistance levels."""
        supports = []
        resistances = []
        for price, relevance in sr_relevance.items():
            if relevance['level_data']['type'] == 'support' and price < current_price:
                supports.append((price, relevance))
            elif relevance['level_data']['type'] == 'resistance' and price > current_price:
                resistances.append((price, relevance))
        supports.sort(key = lambda x: x[1]['total_score'], reverse = True)
        resistances.sort(key = lambda x: x[1]['total_score'], reverse = True)
        most_relevant_support = supports[0][1] if supports else None
        most_relevant_resistance = resistances[0][1] if resistances else None
        return (most_relevant_support, most_relevant_resistance)

    def _calculate_location_metrics_returns_based(self, current_price: float, relevant_support: Optional[Dict], relevant_resistance: Optional[Dict], returns_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Calculate location metrics normalized by returns instead of percentage."""
        current_volatility = returns_data['return_volatility'].iloc[-1]
        avg_return = abs(returns_data['returns'].iloc[-20:].mean())
        metrics = {}
        if relevant_support:
            support_price = relevant_support['level_data']['price']
            metrics['support_distance_returns'] = (current_price - support_price) / (support_price * current_volatility)
            metrics['support_distance_periods'] = metrics['support_distance_returns'] / (avg_return + 1e-08)
            metrics['support_relevance'] = relevant_support['total_score']
            metrics['support_price'] = support_price
        else:
            metrics['support_distance_returns'] = float('inf')
            metrics['support_distance_periods'] = float('inf')
            metrics['support_relevance'] = 0.0
            metrics['support_price'] = None
        if relevant_resistance:
            resistance_price = relevant_resistance['level_data']['price']
            metrics['resistance_distance_returns'] = (resistance_price - current_price) / (current_price * current_volatility)
            metrics['resistance_distance_periods'] = metrics['resistance_distance_returns'] / (avg_return + 1e-08)
            metrics['resistance_relevance'] = relevant_resistance['total_score']
            metrics['resistance_price'] = resistance_price
        else:
            metrics['resistance_distance_returns'] = float('inf')
            metrics['resistance_distance_periods'] = float('inf')
            metrics['resistance_relevance'] = 0.0
            metrics['resistance_price'] = None
        if relevant_support and relevant_resistance:
            support_weight = metrics['support_relevance'] / metrics['support_distance_returns']
            resistance_weight = metrics['resistance_relevance'] / metrics['resistance_distance_returns']
            total_weight = support_weight + resistance_weight
            if total_weight > 0:
                metrics['location_score'] = (resistance_weight - support_weight) / total_weight
            else:
                metrics['location_score'] = 0.0
        else:
            metrics['location_score'] = 0.0
        metrics['sr_quality'] = (metrics.get('support_relevance', 0) + metrics.get('resistance_relevance', 0)) / 2
        return metrics

    def _get_default_classification(self) -> Dict[str, Any]:
        """Return default classification when analysis fails."""
        return {'support_distance_returns': float('inf'), 'resistance_distance_returns': float('inf'), 'support_relevance': 0.0, 'resistance_relevance': 0.0, 'location_score': 0.0, 'sr_quality': 0.0, 'sr_analysis': {'detected_levels': [], 'relevance_scores': {}, 'total_sr_levels': 0, 'high_relevance_count': 0}, 'timestamp': datetime.now().isoformat(), 'error': 'Insufficient data or analysis failed'}

    def get_ml_features(self, classification: Dict[str, Any]) -> pd.Series:
        """Get ML features focused on S/R quality and relevance."""
        features = {}
        features['support_relevance'] = classification.get('support_relevance', 0.0)
        features['resistance_relevance'] = classification.get('resistance_relevance', 0.0)
        features['sr_quality'] = classification.get('sr_quality', 0.0)
        features['support_distance_returns'] = min(10.0, classification.get('support_distance_returns', 10.0))
        features['resistance_distance_returns'] = min(10.0, classification.get('resistance_distance_returns', 10.0))
        features['support_distance_periods'] = min(100.0, classification.get('support_distance_periods', 100.0))
        features['resistance_distance_periods'] = min(100.0, classification.get('resistance_distance_periods', 100.0))
        features['location_score'] = classification.get('location_score', 0.0)
        sr_analysis = classification.get('sr_analysis', {})
        features['total_sr_levels'] = sr_analysis.get('total_sr_levels', 0)
        features['high_relevance_sr_count'] = sr_analysis.get('high_relevance_count', 0)
        features['sr_density'] = features['total_sr_levels'] / 100
        if 'most_relevant_support' in sr_analysis and sr_analysis['most_relevant_support']:
            support = sr_analysis['most_relevant_support']
            features['support_return_magnitude'] = support.get('return_magnitude', 0.0)
            features['support_touch_count'] = support.get('touch_count', 0.0)
            features['support_recency'] = support.get('recency', 0.0)
            features['support_volume_conf'] = support.get('volume_confirmation', 0.0)
            features['support_success_rate'] = support.get('success_rate', 0.0)
        if 'most_relevant_resistance' in sr_analysis and sr_analysis['most_relevant_resistance']:
            resistance = sr_analysis['most_relevant_resistance']
            features['resistance_return_magnitude'] = resistance.get('return_magnitude', 0.0)
            features['resistance_touch_count'] = resistance.get('touch_count', 0.0)
            features['resistance_recency'] = resistance.get('recency', 0.0)
            features['resistance_volume_conf'] = resistance.get('volume_confirmation', 0.0)
            features['resistance_success_rate'] = resistance.get('success_rate', 0.0)
        return pd.Series(features)

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
