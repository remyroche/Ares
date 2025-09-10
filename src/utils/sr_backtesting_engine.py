"""
SR Level Backtesting Engine

This module implements a comprehensive backtesting system to evaluate SR levels
and learn what constitutes a "good" support/resistance level based on historical performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod

from .logger import system_logger

@dataclass
class SRLevel:
    """Support/Resistance level for backtesting."""
    price: float
    level_type: str  # 'support' or 'resistance'
    strength: float
    touches: int
    detection_time: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BacktestResult:
    """Result of backtesting a single SR level."""
    level: SRLevel
    total_touches: int
    successful_touches: int
    failed_touches: int
    success_rate: float
    avg_bounce_strength: float
    max_bounce_strength: float
    avg_hold_time: float
    total_volume_at_level: float
    price_deviation: float
    time_persistence: float
    quality_score: float
    performance_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class BacktestConfig:
    """Configuration for SR level backtesting."""
    # Touch validation parameters
    touch_tolerance: float = 0.002  # 0.2% tolerance for touch detection
    min_bounce_strength: float = 0.001  # Minimum 0.1% bounce to count as successful
    max_hold_time: int = 24  # Maximum hours to hold at level before considering failure
    
    # Volume analysis
    volume_threshold_multiplier: float = 1.5  # Volume must be 1.5x average to count
    volume_lookback_periods: int = 20  # Periods to calculate average volume
    
    # Time analysis
    min_time_between_touches: int = 1  # Minimum hours between touches
    max_analysis_period: int = 720  # Maximum hours to analyze (30 days)
    
    # Quality scoring weights
    success_rate_weight: float = 0.3
    bounce_strength_weight: float = 0.25
    volume_confirmation_weight: float = 0.2
    time_persistence_weight: float = 0.15
    touch_frequency_weight: float = 0.1

class SRBacktestingEngine:
    """Engine for backtesting SR levels and learning quality rules."""
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.logger = system_logger.getChild('SRBacktestingEngine')
        self.learned_rules: Dict[str, Any] = {}
        self.performance_history: List[BacktestResult] = []
        
    def backtest_sr_level(self, level: SRLevel, data: pd.DataFrame) -> BacktestResult:
        """Backtest a single SR level against historical data."""
        try:
            # Find the detection time in the data
            detection_idx = self._find_detection_time_index(level, data)
            if detection_idx is None:
                return self._create_failed_result(level, "Detection time not found in data")
            
            # Analyze the level performance after detection
            analysis_data = data.iloc[detection_idx:detection_idx + self.config.max_analysis_period]
            
            # Detect touches and analyze performance
            touches = self._detect_touches(level, analysis_data)
            
            if not touches:
                return self._create_failed_result(level, "No touches detected")
            
            # Analyze each touch
            touch_results = []
            for touch in touches:
                result = self._analyze_touch(level, touch, analysis_data)
                touch_results.append(result)
            
            # Calculate overall performance metrics
            performance_metrics = self._calculate_performance_metrics(level, touch_results, analysis_data)
            
            # Create backtest result
            result = BacktestResult(
                level=level,
                total_touches=len(touches),
                successful_touches=sum(1 for r in touch_results if r['successful']),
                failed_touches=sum(1 for r in touch_results if not r['successful']),
                success_rate=performance_metrics['success_rate'],
                avg_bounce_strength=performance_metrics['avg_bounce_strength'],
                max_bounce_strength=performance_metrics['max_bounce_strength'],
                avg_hold_time=performance_metrics['avg_hold_time'],
                total_volume_at_level=performance_metrics['total_volume_at_level'],
                price_deviation=performance_metrics['price_deviation'],
                time_persistence=performance_metrics['time_persistence'],
                quality_score=performance_metrics['quality_score'],
                performance_metrics=performance_metrics
            )
            
            self.performance_history.append(result)
            return result
            
        except Exception as e:
            self.logger.error(f"Backtesting failed for level {level.price}: {e}")
            return self._create_failed_result(level, str(e))
    
    def backtest_multiple_levels(self, levels: List[SRLevel], data: pd.DataFrame) -> List[BacktestResult]:
        """Backtest multiple SR levels."""
        results = []
        
        self.logger.info(f"Backtesting {len(levels)} SR levels")
        
        for i, level in enumerate(levels):
            if i % 10 == 0:
                self.logger.info(f"Backtesting level {i+1}/{len(levels)}: ${level.price:.2f}")
            
            result = self.backtest_sr_level(level, data)
            results.append(result)
        
        self.logger.info(f"Backtesting completed. Average quality score: {np.mean([r.quality_score for r in results]):.3f}")
        return results
    
    def learn_quality_rules(self, results: List[BacktestResult]) -> Dict[str, Any]:
        """Learn quality rules from backtesting results using continuous strength scoring."""
        try:
            if not results:
                return {}
            
            # Use continuous quality scoring instead of binary categories
            quality_scores = [r.quality_score for r in results]
            
            # Calculate quality distribution statistics
            quality_stats = {
                'mean': np.mean(quality_scores),
                'std': np.std(quality_scores),
                'min': np.min(quality_scores),
                'max': np.max(quality_scores),
                'percentiles': {
                    '25th': np.percentile(quality_scores, 25),
                    '50th': np.percentile(quality_scores, 50),
                    '75th': np.percentile(quality_scores, 75),
                    '90th': np.percentile(quality_scores, 90)
                }
            }
            
            self.logger.info(f"Learning rules from {len(results)} levels with quality scores: mean={quality_stats['mean']:.3f}, std={quality_stats['std']:.3f}")
            
            # Analyze quality-based feature relationships
            rules = {
                'quality_distribution': quality_stats,
                'feature_quality_correlations': self._calculate_feature_quality_correlations(results),
                'quality_predictors': self._identify_quality_predictors(results),
                'learned_weights': self._learn_feature_weights(results),
                'quality_thresholds': self._calculate_quality_thresholds(quality_stats),
                'strength_scoring_model': self._build_strength_scoring_model(results)
            }
            
            self.learned_rules = rules
            self.logger.info(f"Learned quality rules with {len(rules['quality_predictors'])} key predictors")
            
            return rules
            
        except Exception as e:
            self.logger.error(f"Failed to learn quality rules: {e}")
            return {}
    
    def predict_level_quality(self, level: SRLevel, data: pd.DataFrame) -> float:
        """Predict the quality of a level using learned rules."""
        try:
            if not self.learned_rules:
                return level.strength  # Fallback to original strength
            
            # Extract features for prediction
            features = self._extract_prediction_features(level, data)
            
            # Apply learned rules
            quality_score = self._apply_learned_rules(features)
            
            return quality_score
            
        except Exception as e:
            self.logger.warning(f"Quality prediction failed: {e}")
            return level.strength
    
    def _find_detection_time_index(self, level: SRLevel, data: pd.DataFrame) -> Optional[int]:
        """Find the index in data corresponding to the level detection time."""
        try:
            # For now, find the closest time to detection_time
            # In a real implementation, you'd match the exact detection timestamp
            if 'timestamp' in data.columns:
                detection_time = level.detection_time
                time_diff = abs(data['timestamp'] - detection_time)
                return time_diff.idxmin()
            else:
                # Fallback: use the middle of the data
                return len(data) // 2
        except Exception:
            return None
    
    def _detect_touches(self, level: SRLevel, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect touches of the SR level in the data."""
        touches = []
        tolerance = level.price * self.config.touch_tolerance
        
        for i in range(len(data)):
            high = data.iloc[i]['high']
            low = data.iloc[i]['low']
            
            # Check if price touched the level
            if level.level_type == 'support':
                if low <= level.price + tolerance and high >= level.price - tolerance:
                    touches.append({
                        'index': i,
                        'timestamp': data.iloc[i].get('timestamp', i),
                        'price': level.price,
                        'touch_type': 'support',
                        'ohlc': {
                            'open': data.iloc[i]['open'],
                            'high': data.iloc[i]['high'],
                            'low': data.iloc[i]['low'],
                            'close': data.iloc[i]['close'],
                            'volume': data.iloc[i].get('volume', 0)
                        }
                    })
            else:  # resistance
                if high >= level.price - tolerance and low <= level.price + tolerance:
                    touches.append({
                        'index': i,
                        'timestamp': data.iloc[i].get('timestamp', i),
                        'price': level.price,
                        'touch_type': 'resistance',
                        'ohlc': {
                            'open': data.iloc[i]['open'],
                            'high': data.iloc[i]['high'],
                            'low': data.iloc[i]['low'],
                            'close': data.iloc[i]['close'],
                            'volume': data.iloc[i].get('volume', 0)
                        }
                    })
        
        return touches
    
    def _analyze_touch(self, level: SRLevel, touch: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze a single touch to determine if it was successful."""
        try:
            touch_idx = touch['index']
            touch_data = data.iloc[touch_idx:]
            
            # Look ahead to see if the level held
            bounce_strength = 0.0
            hold_time = 0
            successful = False
            
            for i in range(min(len(touch_data), self.config.max_hold_time)):
                current_data = touch_data.iloc[i]
                
                if level.level_type == 'support':
                    # For support: price should bounce up
                    if current_data['low'] <= level.price * (1 + self.config.touch_tolerance):
                        # Still at support level
                        hold_time = i
                    else:
                        # Price moved away from support
                        bounce_strength = (current_data['close'] - level.price) / level.price
                        if bounce_strength >= self.config.min_bounce_strength:
                            successful = True
                        break
                else:  # resistance
                    # For resistance: price should bounce down
                    if current_data['high'] >= level.price * (1 - self.config.touch_tolerance):
                        # Still at resistance level
                        hold_time = i
                    else:
                        # Price moved away from resistance
                        bounce_strength = (level.price - current_data['close']) / level.price
                        if bounce_strength >= self.config.min_bounce_strength:
                            successful = True
                        break
            
            return {
                'successful': successful,
                'bounce_strength': bounce_strength,
                'hold_time': hold_time,
                'volume': touch['ohlc']['volume']
            }
            
        except Exception as e:
            self.logger.warning(f"Touch analysis failed: {e}")
            return {
                'successful': False,
                'bounce_strength': 0.0,
                'hold_time': 0,
                'volume': 0
            }
    
    def _calculate_performance_metrics(self, level: SRLevel, touch_results: List[Dict], data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        try:
            if not touch_results:
                return self._get_default_metrics()
            
            # Basic metrics
            total_touches = len(touch_results)
            successful_touches = sum(1 for r in touch_results if r['successful'])
            success_rate = successful_touches / total_touches if total_touches > 0 else 0.0
            
            # Bounce strength metrics
            bounce_strengths = [r['bounce_strength'] for r in touch_results if r['successful']]
            avg_bounce_strength = np.mean(bounce_strengths) if bounce_strengths else 0.0
            max_bounce_strength = max(bounce_strengths) if bounce_strengths else 0.0
            
            # Hold time metrics
            hold_times = [r['hold_time'] for r in touch_results]
            avg_hold_time = np.mean(hold_times) if hold_times else 0.0
            
            # Volume analysis
            volumes = [r['volume'] for r in touch_results]
            total_volume = sum(volumes)
            avg_volume = np.mean(volumes) if volumes else 0.0
            
            # Price deviation (how much the level deviates from actual touches)
            actual_touch_prices = [level.price] * total_touches  # Simplified
            price_deviation = 0.0  # Would calculate actual deviation in real implementation
            
            # Time persistence (how long the level remains relevant)
            time_persistence = min(1.0, total_touches / 10.0)  # Normalized to 0-1
            
            # Calculate overall quality score
            quality_score = (
                self.config.success_rate_weight * success_rate +
                self.config.bounce_strength_weight * min(avg_bounce_strength * 10, 1.0) +
                self.config.volume_confirmation_weight * min(avg_volume / 1000000, 1.0) +
                self.config.time_persistence_weight * time_persistence +
                self.config.touch_frequency_weight * min(total_touches / 5.0, 1.0)
            )
            
            return {
                'success_rate': success_rate,
                'avg_bounce_strength': avg_bounce_strength,
                'max_bounce_strength': max_bounce_strength,
                'avg_hold_time': avg_hold_time,
                'total_volume_at_level': total_volume,
                'price_deviation': price_deviation,
                'time_persistence': time_persistence,
                'quality_score': quality_score,
                'total_touches': total_touches,
                'successful_touches': successful_touches
            }
            
        except Exception as e:
            self.logger.warning(f"Performance metrics calculation failed: {e}")
            return self._get_default_metrics()
    
    def _get_default_metrics(self) -> Dict[str, float]:
        """Get default metrics for failed calculations."""
        return {
            'success_rate': 0.0,
            'avg_bounce_strength': 0.0,
            'max_bounce_strength': 0.0,
            'avg_hold_time': 0.0,
            'total_volume_at_level': 0.0,
            'price_deviation': 0.0,
            'time_persistence': 0.0,
            'quality_score': 0.0,
            'total_touches': 0,
            'successful_touches': 0
        }
    
    def _create_failed_result(self, level: SRLevel, reason: str) -> BacktestResult:
        """Create a failed backtest result."""
        return BacktestResult(
            level=level,
            total_touches=0,
            successful_touches=0,
            failed_touches=0,
            success_rate=0.0,
            avg_bounce_strength=0.0,
            max_bounce_strength=0.0,
            avg_hold_time=0.0,
            total_volume_at_level=0.0,
            price_deviation=0.0,
            time_persistence=0.0,
            quality_score=0.0,
            performance_metrics={'error': reason}
        )
    
    def _analyze_level_characteristics(self, results: List[BacktestResult]) -> Dict[str, Any]:
        """Analyze characteristics of a group of levels."""
        if not results:
            return {}
        
        return {
            'avg_success_rate': np.mean([r.success_rate for r in results]),
            'avg_bounce_strength': np.mean([r.avg_bounce_strength for r in results]),
            'avg_volume': np.mean([r.total_volume_at_level for r in results]),
            'avg_touches': np.mean([r.total_touches for r in results]),
            'avg_quality_score': np.mean([r.quality_score for r in results]),
            'level_types': {
                'support': sum(1 for r in results if r.level.level_type == 'support'),
                'resistance': sum(1 for r in results if r.level.level_type == 'resistance')
            }
        }
    
    def _calculate_feature_quality_correlations(self, results: List[BacktestResult]) -> Dict[str, float]:
        """Calculate correlations between features and quality scores."""
        if not results:
            return {}
        
        features = ['success_rate', 'avg_bounce_strength', 'total_volume_at_level', 'total_touches', 'time_persistence']
        correlations = {}
        
        quality_scores = [r.quality_score for r in results]
        
        for feature in features:
            feature_values = [getattr(r, feature) for r in results]
            correlation = np.corrcoef(feature_values, quality_scores)[0, 1]
            correlations[feature] = correlation if not np.isnan(correlation) else 0.0
        
        return correlations
    
    def _identify_quality_predictors(self, results: List[BacktestResult]) -> Dict[str, Any]:
        """Identify the best predictors of quality using continuous scoring."""
        if not results:
            return {}
        
        correlations = self._calculate_feature_quality_correlations(results)
        
        # Sort features by correlation strength
        sorted_features = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        predictors = {}
        for feature, correlation in sorted_features:
            predictors[feature] = {
                'correlation': correlation,
                'strength': abs(correlation),
                'direction': 'positive' if correlation > 0 else 'negative'
            }
        
        return predictors
    
    def _calculate_quality_thresholds(self, quality_stats: Dict[str, Any]) -> Dict[str, float]:
        """Calculate quality thresholds for different use cases."""
        return {
            'excellent': quality_stats['percentiles']['90th'],
            'good': quality_stats['percentiles']['75th'],
            'average': quality_stats['percentiles']['50th'],
            'poor': quality_stats['percentiles']['25th'],
            'minimum_acceptable': quality_stats['mean'] - quality_stats['std']
        }
    
    def _build_strength_scoring_model(self, results: List[BacktestResult]) -> Dict[str, Any]:
        """Build a model for predicting quality scores."""
        if not results:
            return {}
        
        # Extract features and target
        features = ['success_rate', 'avg_bounce_strength', 'total_volume_at_level', 'total_touches', 'time_persistence']
        X = np.array([[getattr(r, feature) for feature in features] for r in results])
        y = np.array([r.quality_score for r in results])
        
        # Simple linear regression model (can be enhanced with ML)
        try:
            # Calculate feature weights using correlation
            correlations = self._calculate_feature_quality_correlations(results)
            weights = np.array([correlations.get(feature, 0.0) for feature in features])
            
            # Normalize weights
            if np.sum(np.abs(weights)) > 0:
                weights = weights / np.sum(np.abs(weights))
            
            model = {
                'feature_names': features,
                'weights': weights.tolist(),
                'intercept': np.mean(y) - np.dot(weights, np.mean(X, axis=0)),
                'r_squared': self._calculate_r_squared(X, y, weights),
                'feature_importance': dict(zip(features, np.abs(weights)))
            }
            
            return model
            
        except Exception as e:
            self.logger.warning(f"Failed to build strength scoring model: {e}")
            return {}
    
    def _calculate_r_squared(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        """Calculate R-squared for the model."""
        try:
            y_pred = np.dot(X, weights) + np.mean(y) - np.dot(weights, np.mean(X, axis=0))
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        except Exception:
            return 0.0
    
    def _learn_feature_weights(self, results: List[BacktestResult]) -> Dict[str, float]:
        """Learn optimal feature weights from backtesting results."""
        # This would use machine learning to optimize weights
        # For now, return the configured weights
        return {
            'success_rate': self.config.success_rate_weight,
            'bounce_strength': self.config.bounce_strength_weight,
            'volume_confirmation': self.config.volume_confirmation_weight,
            'time_persistence': self.config.time_persistence_weight,
            'touch_frequency': self.config.touch_frequency_weight
        }
    
    def _calculate_performance_thresholds(self, high_quality: List[BacktestResult]) -> Dict[str, float]:
        """Calculate performance thresholds for quality classification."""
        if not high_quality:
            return {}
        
        return {
            'min_success_rate': np.percentile([r.success_rate for r in high_quality], 25),
            'min_bounce_strength': np.percentile([r.avg_bounce_strength for r in high_quality], 25),
            'min_volume': np.percentile([r.total_volume_at_level for r in high_quality], 25),
            'min_touches': np.percentile([r.total_touches for r in high_quality], 25)
        }
    
    def _extract_prediction_features(self, level: SRLevel, data: pd.DataFrame) -> Dict[str, float]:
        """Extract features for quality prediction."""
        # This would extract relevant features from the level and data
        # For now, return basic features
        return {
            'strength': level.strength,
            'touches': level.touches,
            'price': level.price,
            'level_type': 1.0 if level.level_type == 'support' else 0.0
        }
    
    def _apply_learned_rules(self, features: Dict[str, float]) -> float:
        """Apply learned rules to predict quality using continuous scoring model."""
        if not self.learned_rules:
            return features.get('strength', 0.5)
        
        # Use the strength scoring model if available
        model = self.learned_rules.get('strength_scoring_model', {})
        if model and 'weights' in model:
            try:
                # Extract features in the same order as the model
                feature_values = []
                for feature_name in model['feature_names']:
                    feature_values.append(features.get(feature_name, 0.0))
                
                # Apply the model
                weights = np.array(model['weights'])
                intercept = model.get('intercept', 0.0)
                
                quality_score = np.dot(feature_values, weights) + intercept
                
                # Ensure score is within valid range
                return min(max(quality_score, 0.0), 1.0)
                
            except Exception as e:
                self.logger.warning(f"Failed to apply strength scoring model: {e}")
        
        # Fallback to correlation-based prediction
        predictors = self.learned_rules.get('quality_predictors', {})
        if predictors:
            quality_score = 0.0
            total_weight = 0.0
            
            for feature, info in predictors.items():
                if feature in features:
                    weight = info['strength']
                    correlation = info['correlation']
                    
                    # Normalize feature value (assuming 0-1 range)
                    normalized_value = min(max(features[feature], 0.0), 1.0)
                    
                    # Apply correlation direction
                    if correlation > 0:
                        quality_score += normalized_value * weight
                    else:
                        quality_score += (1.0 - normalized_value) * weight
                    
                    total_weight += weight
            
            if total_weight > 0:
                quality_score = quality_score / total_weight
                return min(max(quality_score, 0.0), 1.0)
        
        # Final fallback
        return features.get('strength', 0.5)
    
    def get_quality_rules_summary(self) -> Dict[str, Any]:
        """Get a summary of learned quality rules."""
        if not self.learned_rules:
            return {'status': 'No rules learned yet'}
        
        return {
            'status': 'Rules learned',
            'quality_threshold': self.learned_rules.get('quality_threshold', 0.0),
            'discriminative_features': list(self.learned_rules.get('discriminative_features', {}).keys()),
            'performance_thresholds': self.learned_rules.get('performance_thresholds', {}),
            'total_levels_analyzed': len(self.performance_history)
        }

def create_sr_level_from_dict(level_dict: Dict[str, Any]) -> SRLevel:
    """Create SRLevel from dictionary."""
    return SRLevel(
        price=level_dict['price'],
        level_type=level_dict.get('type', 'support'),
        strength=level_dict.get('strength', 0.5),
        touches=level_dict.get('touches', 1),
        detection_time=level_dict.get('detection_time', datetime.now()),
        metadata=level_dict.get('metadata', {})
    )

def get_backtesting_engine(config: Optional[BacktestConfig] = None) -> SRBacktestingEngine:
    """Get a backtesting engine instance."""
    return SRBacktestingEngine(config)