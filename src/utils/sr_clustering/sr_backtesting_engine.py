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

from ..logger import system_logger

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
    
    def learn_quality_rules(self, results: List[BacktestResult], 
                           optimize_weights: bool = True,
                           market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
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
            
            # Optimize weights if requested
            optimized_weights = {}
            if optimize_weights and market_data is not None:
                try:
                    from .weight_optimization_engine import get_weight_optimization_engine, WeightOptimizationConfig
                    
                    # Configure weight optimization
                    weight_config = WeightOptimizationConfig(
                        optimization_method='scipy_minimize',
                        primary_objective='r2_score',
                        secondary_objective='stability'
                    )
                    
                    weight_optimizer = get_weight_optimization_engine(weight_config)
                    optimization_result = weight_optimizer.optimize_weights(results, market_data)
                    
                    if optimization_result and optimization_result.get('optimization_success', False):
                        optimized_weights = optimization_result.get('best_weights', {})
                        self.logger.info(f"Weight optimization completed. Best score: {optimization_result.get('best_score', 0.0):.4f}")
                        self.logger.info(f"Optimized weights: {optimized_weights}")
                    else:
                        self.logger.warning("Weight optimization failed, using default weights")
                        
                except Exception as e:
                    self.logger.warning(f"Weight optimization failed: {e}")
            
            # Analyze quality-based feature relationships
            # Build strength scoring model
            strength_model = self._build_strength_scoring_model(results)
            
            # Log comprehensive feature analysis
            if strength_model:
                self._log_comprehensive_feature_analysis(results, strength_model)
            
            rules = {
                'quality_distribution': quality_stats,
                'feature_quality_correlations': self._calculate_feature_quality_correlations(results),
                'quality_predictors': self._identify_quality_predictors(results),
                'learned_weights': self._learn_feature_weights(results),
                'quality_thresholds': self._calculate_quality_thresholds(quality_stats),
                'strength_scoring_model': strength_model,
                'optimized_weights': optimized_weights,
                'weight_optimization_enabled': optimize_weights
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
        """Calculate comprehensive performance metrics including penetration and pattern features."""
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
            
            # Calculate penetration and pattern features
            penetration_metrics = self._calculate_penetration_metrics(level, touch_results, data)
            pattern_metrics = self._calculate_pattern_metrics(level, touch_results, data)
            
            # Calculate overall quality score with enhanced features
            # Use optimized weights if available, otherwise use default weights
            if hasattr(self, 'learned_rules') and self.learned_rules and 'optimized_weights' in self.learned_rules:
                optimized_weights = self.learned_rules['optimized_weights']
                if optimized_weights:
                    # Use optimized weights
                    quality_score = (
                        optimized_weights.get('success_rate', 0.3) * success_rate +
                        optimized_weights.get('avg_bounce_strength', 0.25) * min(avg_bounce_strength * 10, 1.0) +
                        optimized_weights.get('total_volume_at_level', 0.2) * min(avg_volume / 1000000, 1.0) +
                        optimized_weights.get('time_persistence', 0.15) * time_persistence +
                        optimized_weights.get('touch_frequency', 0.1) * min(total_touches / 5.0, 1.0) +
                        optimized_weights.get('penetration_depth', 0.1) * penetration_metrics['penetration_depth'] +
                        optimized_weights.get('pattern_consistency', 0.1) * pattern_metrics['pattern_consistency']
                    )
                else:
                    # Fallback to default weights
                    quality_score = self._calculate_default_quality_score(
                        success_rate, avg_bounce_strength, avg_volume, time_persistence, 
                        total_touches, penetration_metrics, pattern_metrics
                    )
            else:
                # Use default weights
                quality_score = self._calculate_default_quality_score(
                    success_rate, avg_bounce_strength, avg_volume, time_persistence, 
                    total_touches, penetration_metrics, pattern_metrics
                )
            
            # Combine all metrics
            metrics = {
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
            
            # Add penetration and pattern metrics
            metrics.update(penetration_metrics)
            metrics.update(pattern_metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Performance metrics calculation failed: {e}")
            return self._get_default_metrics()
    
    def _calculate_penetration_metrics(self, level: SRLevel, touch_results: List[Dict], data: pd.DataFrame) -> Dict[str, float]:
        """Calculate penetration depth and frequency metrics."""
        try:
            if not touch_results:
                return {
                    'penetration_depth': 0.0,
                    'penetration_frequency': 0.0
                }
            
            # Calculate penetration depth (how deep price went beyond the level)
            penetration_depths = []
            penetration_count = 0
            
            for touch in touch_results:
                touch_idx = touch['index']
                if touch_idx < len(data) - 1:
                    # Look at the next few bars to see penetration depth
                    next_bars = data.iloc[touch_idx:touch_idx + 3]  # Look at next 3 bars
                    
                    if level.level_type == 'support':
                        # For support: measure how far below the level price went
                        min_low = next_bars['low'].min()
                        if min_low < level.price:
                            penetration = (level.price - min_low) / level.price
                            penetration_depths.append(penetration)
                            penetration_count += 1
                    else:  # resistance
                        # For resistance: measure how far above the level price went
                        max_high = next_bars['high'].max()
                        if max_high > level.price:
                            penetration = (max_high - level.price) / level.price
                            penetration_depths.append(penetration)
                            penetration_count += 1
            
            # Calculate metrics
            avg_penetration_depth = np.mean(penetration_depths) if penetration_depths else 0.0
            penetration_frequency = penetration_count / len(touch_results) if touch_results else 0.0
            
            return {
                'penetration_depth': min(avg_penetration_depth, 1.0),  # Cap at 100%
                'penetration_frequency': penetration_frequency
            }
            
        except Exception as e:
            self.logger.warning(f"Penetration metrics calculation failed: {e}")
            return {
                'penetration_depth': 0.0,
                'penetration_frequency': 0.0
            }
    
    def _calculate_pattern_metrics(self, level: SRLevel, touch_results: List[Dict], data: pd.DataFrame) -> Dict[str, float]:
        """Calculate pattern consistency and strength metrics."""
        try:
            if not touch_results:
                return {
                    'pattern_consistency': 0.0,
                    'pattern_strength': 0.0,
                    'order_flow_confirmation': 0.0,
                    'absorption_patterns': 0.0,
                    'structure_break': 0.0
                }
            
            # Pattern consistency: how consistent are the bounce patterns?
            bounce_strengths = [r['bounce_strength'] for r in touch_results if r['successful']]
            if len(bounce_strengths) > 1:
                pattern_consistency = 1.0 - (np.std(bounce_strengths) / (np.mean(bounce_strengths) + 1e-8))
                pattern_consistency = max(0.0, min(1.0, pattern_consistency))
            else:
                pattern_consistency = 1.0 if bounce_strengths else 0.0
            
            # Pattern strength: average strength of successful bounces
            pattern_strength = np.mean(bounce_strengths) if bounce_strengths else 0.0
            
            # Order flow confirmation: volume patterns at touches
            volumes_at_touches = [r['volume'] for r in touch_results]
            avg_volume_at_touches = np.mean(volumes_at_touches) if volumes_at_touches else 0.0
            overall_avg_volume = data['volume'].mean() if 'volume' in data.columns else 1.0
            order_flow_confirmation = min(avg_volume_at_touches / overall_avg_volume, 2.0) / 2.0  # Normalize to 0-1
            
            # Absorption patterns: high volume with little price movement
            absorption_count = 0
            for touch in touch_results:
                touch_idx = touch['index']
                if touch_idx < len(data) - 2:
                    # Check for absorption pattern (high volume, low price movement)
                    touch_volume = touch['volume']
                    price_range = data.iloc[touch_idx-1:touch_idx+2]['high'].max() - data.iloc[touch_idx-1:touch_idx+2]['low'].min()
                    price_range_pct = price_range / level.price
                    
                    if touch_volume > overall_avg_volume * 1.5 and price_range_pct < 0.01:  # High volume, low movement
                        absorption_count += 1
            
            absorption_patterns = absorption_count / len(touch_results) if touch_results else 0.0
            
            # Structure break: how often did the level break market structure
            structure_breaks = 0
            for touch in touch_results:
                if not touch['successful']:  # Failed touches indicate structure breaks
                    structure_breaks += 1
            
            structure_break = structure_breaks / len(touch_results) if touch_results else 0.0
            
            return {
                'pattern_consistency': pattern_consistency,
                'pattern_strength': min(pattern_strength, 1.0),
                'order_flow_confirmation': order_flow_confirmation,
                'absorption_patterns': absorption_patterns,
                'structure_break': structure_break
            }
            
        except Exception as e:
            self.logger.warning(f"Pattern metrics calculation failed: {e}")
            return {
                'pattern_consistency': 0.0,
                'pattern_strength': 0.0,
                'order_flow_confirmation': 0.0,
                'absorption_patterns': 0.0,
                'structure_break': 0.0
            }

    def _calculate_default_quality_score(self, success_rate: float, avg_bounce_strength: float, 
                                       avg_volume: float, time_persistence: float, 
                                       total_touches: int, penetration_metrics: Dict[str, float], 
                                       pattern_metrics: Dict[str, float]) -> float:
        """Calculate quality score using default weights."""
        return (
            self.config.success_rate_weight * success_rate +
            self.config.bounce_strength_weight * min(avg_bounce_strength * 10, 1.0) +
            self.config.volume_confirmation_weight * min(avg_volume / 1000000, 1.0) +
            self.config.time_persistence_weight * time_persistence +
            self.config.touch_frequency_weight * min(total_touches / 5.0, 1.0) +
            0.1 * penetration_metrics['penetration_depth'] +  # 10% weight for penetration
            0.1 * pattern_metrics['pattern_consistency']      # 10% weight for pattern consistency
        )

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
            'successful_touches': 0,
            # Penetration metrics
            'penetration_depth': 0.0,
            'penetration_frequency': 0.0,
            # Pattern metrics
            'pattern_consistency': 0.0,
            'pattern_strength': 0.0,
            'order_flow_confirmation': 0.0,
            'absorption_patterns': 0.0,
            'structure_break': 0.0
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
        """Calculate correlations between SR-focused features and quality scores."""
        if not results:
            return {}
        
        # Primary SR-focused features (optimized)
        primary_features = [
            'success_rate',           # How often level held
            'avg_bounce_strength',    # Strength of price reaction
            'max_bounce_strength',    # Maximum bounce strength
            'total_touches',          # Number of touches
            'time_persistence',       # How long level remained relevant
            'total_volume_at_level',  # Volume confirmation
            'avg_hold_time'           # How long price held at level
        ]
        
        # Penetration and pattern features
        penetration_pattern_features = [
            'penetration_depth',      # How deep price penetrated the level
            'penetration_frequency',  # How often level was penetrated
            'pattern_consistency',    # Consistency of bounce patterns
            'pattern_strength',       # Strength of the pattern
            'order_flow_confirmation', # Order flow pattern confirmation
            'absorption_patterns',    # Volume absorption patterns
            'structure_break'         # Market structure break confirmation
        ]
        
        # Use existing step06 features for secondary analysis
        step06_features = [
            'market_regime',          # From step06: Market regime context
            'volatility_regime',      # From step06: Volatility regime
            'trend_strength',         # From step06: Trend strength
            'volume_regime',          # From step06: Volume regime
            'time_of_day_effect',     # From step06: Time of day effects
            'vwap_momentum',          # From step06: VWAP momentum
            'price_momentum',         # From step06: Price momentum
            'momentum_volume_interaction'  # From step06: Momentum-volume interaction
        ]
        
        all_features = primary_features + penetration_pattern_features + step06_features
        correlations = {}
        
        quality_scores = [r.quality_score for r in results]
        
        for feature in all_features:
            feature_values = [getattr(r, feature, 0.0) for r in results]
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
        """
        Build a Ridge Regression model for predicting quality scores.
        
        MODEL: Ridge Regression with Cross-Validation (RidgeCV)
        
        We use Ridge Regression because:
        1. It handles multicollinearity well (SR features are often correlated)
        2. It provides stable, interpretable coefficients
        3. It prevents overfitting with L2 regularization
        4. It's computationally efficient for real-time prediction
        5. Cross-validation automatically selects optimal regularization parameter
        
        This is NOT a simple linear regression - it's Ridge Regression with:
        - Automatic alpha selection via RidgeCV
        - 5-fold cross-validation for robust performance estimation
        - Feature standardization for stable coefficients
        - L2 regularization to prevent overfitting
        """
        if not results:
            return {}
        
        # Extract all available features (primary + penetration + pattern)
        primary_features = [
            'success_rate', 'avg_bounce_strength', 'max_bounce_strength', 
            'total_touches', 'time_persistence', 'total_volume_at_level', 'avg_hold_time'
        ]
        
        penetration_pattern_features = [
            'penetration_depth', 'penetration_frequency', 'pattern_consistency', 
            'pattern_strength', 'order_flow_confirmation', 'absorption_patterns', 'structure_break'
        ]
        
        all_features = primary_features + penetration_pattern_features
        
        # Build feature matrix
        X = []
        valid_features = []
        
        for feature in all_features:
            feature_values = [getattr(r, feature, 0.0) for r in results]
            if not all(v == 0.0 for v in feature_values):  # Skip features with no variation
                X.append(feature_values)
                valid_features.append(feature)
        
        if not X:
            return {}
        
        X = np.array(X).T  # Transpose to get (samples, features)
        y = np.array([r.quality_score for r in results])
        
        try:
            # Use Ridge Regression with cross-validation for optimal alpha
            from sklearn.linear_model import RidgeCV
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import cross_val_score, validation_curve
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Ridge Regression with cross-validation to find optimal alpha
            alphas = np.logspace(-4, 2, 50)  # Range of regularization strengths
            ridge_model = RidgeCV(alphas=alphas, cv=5, scoring='r2')
            ridge_model.fit(X_scaled, y)
            
            # Get feature importance (absolute coefficients)
            feature_importance = np.abs(ridge_model.coef_)
            feature_importance_normalized = feature_importance / np.sum(feature_importance)
            
            # Calculate model performance
            y_pred = ridge_model.predict(X_scaled)
            r_squared = ridge_model.score(X_scaled, y)
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            
            # Cross-validation scores for robustness
            cv_scores = cross_val_score(ridge_model, X_scaled, y, cv=5, scoring='r2')
            cv_mse_scores = -cross_val_score(ridge_model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            cv_mse_mean = np.mean(cv_mse_scores)
            cv_mse_std = np.std(cv_mse_scores)
            
            # Overfitting detection
            overfitting_detected = False
            overfitting_warnings = []
            
            # Check 1: High variance in CV scores
            if cv_std > 0.1:
                overfitting_detected = True
                overfitting_warnings.append(f"High CV score variance: {cv_std:.3f}")
            
            # Check 2: Large gap between training and CV performance
            performance_gap = r_squared - cv_mean
            if performance_gap > 0.1:
                overfitting_detected = True
                overfitting_warnings.append(f"Large performance gap: {performance_gap:.3f}")
            
            # Check 3: Very high R² with small dataset
            if r_squared > 0.95 and len(y) < 100:
                overfitting_detected = True
                overfitting_warnings.append(f"Suspiciously high R² ({r_squared:.3f}) with small dataset ({len(y)} samples)")
            
            # Check 4: Low optimal alpha (high regularization needed)
            if ridge_model.alpha_ < 0.01:
                overfitting_warnings.append(f"Low optimal alpha ({ridge_model.alpha_:.4f}) suggests high regularization needed")
            
            model = {
                'model_type': 'Ridge Regression with Overfitting Protection',
                'feature_names': valid_features,
                'coefficients': ridge_model.coef_.tolist(),
                'intercept': ridge_model.intercept_,
                'optimal_alpha': ridge_model.alpha_,
                'r_squared': r_squared,
                'mse': mse,
                'mae': mae,
                'cv_r_squared_mean': cv_mean,
                'cv_r_squared_std': cv_std,
                'cv_mse_mean': cv_mse_mean,
                'cv_mse_std': cv_mse_std,
                'feature_importance': dict(zip(valid_features, feature_importance_normalized)),
                'overfitting_detected': overfitting_detected,
                'overfitting_warnings': overfitting_warnings,
                'performance_gap': performance_gap,
                'scaler_mean': scaler.mean_.tolist(),
                'scaler_scale': scaler.scale_.tolist(),
                'model_object': ridge_model,
                'scaler_object': scaler
            }
            
            # Enhanced logging with overfitting information
            self.logger.info(f"Built Ridge Regression model with R²={r_squared:.3f}, CV R²={cv_mean:.3f}±{cv_std:.3f}")
            self.logger.info(f"Model performance: MSE={mse:.4f}, MAE={mae:.4f}")
            self.logger.info(f"Optimal alpha: {ridge_model.alpha_:.4f}")
            
            if overfitting_detected:
                self.logger.warning(f"⚠️ Overfitting detected: {'; '.join(overfitting_warnings)}")
            else:
                self.logger.info("✅ No overfitting detected")
            
            self.logger.info(f"Top 5 features: {sorted(model['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:5]}")
            
            return model
            
        except Exception as e:
            self.logger.warning(f"Failed to build Ridge Regression model: {e}")
            # Fallback to simple correlation-based model
            return self._build_simple_correlation_model(results, valid_features)
    
    def _log_comprehensive_feature_analysis(self, results: List[BacktestResult], 
                                          model_result: Dict[str, Any]) -> None:
        """Log comprehensive feature importance and correlation analysis."""
        try:
            self.logger.info("📊 COMPREHENSIVE FEATURE ANALYSIS REPORT")
            self.logger.info("=" * 60)
            
            # Model Performance Summary
            self.logger.info("🎯 MODEL PERFORMANCE SUMMARY:")
            self.logger.info(f"   Model Type: {model_result.get('model_type', 'Unknown')}")
            self.logger.info(f"   R² Score: {model_result.get('r_squared', 0.0):.4f}")
            self.logger.info(f"   CV R² Mean: {model_result.get('cv_r_squared_mean', 0.0):.4f} ± {model_result.get('cv_r_squared_std', 0.0):.4f}")
            self.logger.info(f"   MSE: {model_result.get('mse', 0.0):.4f}")
            self.logger.info(f"   MAE: {model_result.get('mae', 0.0):.4f}")
            self.logger.info(f"   Optimal Alpha: {model_result.get('optimal_alpha', 0.0):.4f}")
            
            # Overfitting Analysis
            if model_result.get('overfitting_detected', False):
                self.logger.warning("⚠️ OVERFITTING DETECTED:")
                for warning in model_result.get('overfitting_warnings', []):
                    self.logger.warning(f"   - {warning}")
            else:
                self.logger.info("✅ NO OVERFITTING DETECTED")
            
            # Feature Importance Analysis
            feature_importance = model_result.get('feature_importance', {})
            if feature_importance:
                self.logger.info("🔍 FEATURE IMPORTANCE RANKING:")
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                
                # Top 10 most important features
                self.logger.info("   Top 10 Most Important Features:")
                for i, (feature, importance) in enumerate(sorted_features[:10], 1):
                    self.logger.info(f"   {i:2d}. {feature:<30} {importance:.4f}")
                
                # Feature categories analysis
                self._log_feature_category_analysis(sorted_features)
            
            # Correlation Analysis
            self._log_correlation_analysis(results, model_result)
            
            # Model Coefficients Analysis
            self._log_coefficients_analysis(model_result)
            
            self.logger.info("=" * 60)
            
        except Exception as e:
            self.logger.error(f"Failed to log comprehensive feature analysis: {e}")
    
    def _log_feature_category_analysis(self, sorted_features: List[tuple]) -> None:
        """Log analysis by feature categories."""
        try:
            # Categorize features
            primary_features = []
            penetration_features = []
            pattern_features = []
            step06_features = []
            
            for feature, importance in sorted_features:
                if any(x in feature.lower() for x in ['success', 'bounce', 'volume', 'touch', 'time', 'hold']):
                    primary_features.append((feature, importance))
                elif any(x in feature.lower() for x in ['penetration']):
                    penetration_features.append((feature, importance))
                elif any(x in feature.lower() for x in ['pattern', 'order_flow', 'absorption', 'structure']):
                    pattern_features.append((feature, importance))
                elif any(x in feature.lower() for x in ['market', 'volatility', 'trend', 'vwap', 'momentum']):
                    step06_features.append((feature, importance))
            
            self.logger.info("📈 FEATURE CATEGORY ANALYSIS:")
            
            if primary_features:
                avg_importance = np.mean([imp for _, imp in primary_features])
                self.logger.info(f"   Primary SR Features: {len(primary_features)} features, avg importance: {avg_importance:.4f}")
                self.logger.info(f"   Top Primary: {primary_features[0][0]} ({primary_features[0][1]:.4f})")
            
            if penetration_features:
                avg_importance = np.mean([imp for _, imp in penetration_features])
                self.logger.info(f"   Penetration Features: {len(penetration_features)} features, avg importance: {avg_importance:.4f}")
                self.logger.info(f"   Top Penetration: {penetration_features[0][0]} ({penetration_features[0][1]:.4f})")
            
            if pattern_features:
                avg_importance = np.mean([imp for _, imp in pattern_features])
                self.logger.info(f"   Pattern Features: {len(pattern_features)} features, avg importance: {avg_importance:.4f}")
                self.logger.info(f"   Top Pattern: {pattern_features[0][0]} ({pattern_features[0][1]:.4f})")
            
            if step06_features:
                avg_importance = np.mean([imp for _, imp in step06_features])
                self.logger.info(f"   Step06 Features: {len(step06_features)} features, avg importance: {avg_importance:.4f}")
                self.logger.info(f"   Top Step06: {step06_features[0][0]} ({step06_features[0][1]:.4f})")
                
        except Exception as e:
            self.logger.error(f"Failed to log feature category analysis: {e}")
    
    def _log_correlation_analysis(self, results: List[BacktestResult], model_result: Dict[str, Any]) -> None:
        """Log correlation analysis between features and quality scores."""
        try:
            self.logger.info("🔗 CORRELATION ANALYSIS:")
            
            # Calculate correlations for key features
            quality_scores = [r.quality_score for r in results]
            key_features = ['success_rate', 'avg_bounce_strength', 'total_volume_at_level', 
                          'time_persistence', 'total_touches', 'penetration_depth', 'pattern_consistency']
            
            correlations = {}
            for feature in key_features:
                feature_values = [getattr(r, feature, 0.0) for r in results]
                if len(feature_values) > 1:
                    corr = np.corrcoef(feature_values, quality_scores)[0, 1]
                    correlations[feature] = corr
            
            # Sort by absolute correlation
            sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            
            self.logger.info("   Feature-Quality Score Correlations:")
            for feature, corr in sorted_correlations:
                direction = "📈" if corr > 0 else "📉"
                strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
                self.logger.info(f"   {direction} {feature:<25} {corr:+.3f} ({strength})")
            
        except Exception as e:
            self.logger.error(f"Failed to log correlation analysis: {e}")
    
    def _log_coefficients_analysis(self, model_result: Dict[str, Any]) -> None:
        """Log analysis of model coefficients."""
        try:
            coefficients = model_result.get('coefficients', [])
            feature_names = model_result.get('feature_names', [])
            
            if not coefficients or not feature_names:
                return
            
            self.logger.info("📊 MODEL COEFFICIENTS ANALYSIS:")
            
            # Sort by absolute coefficient value
            coef_data = list(zip(feature_names, coefficients))
            sorted_coefs = sorted(coef_data, key=lambda x: abs(x[1]), reverse=True)
            
            # Positive and negative coefficients
            positive_coefs = [(name, coef) for name, coef in sorted_coefs if coef > 0]
            negative_coefs = [(name, coef) for name, coef in sorted_coefs if coef < 0]
            
            self.logger.info(f"   Total Features: {len(feature_names)}")
            self.logger.info(f"   Positive Impact: {len(positive_coefs)} features")
            self.logger.info(f"   Negative Impact: {len(negative_coefs)} features")
            
            if positive_coefs:
                self.logger.info("   Top Positive Contributors:")
                for name, coef in positive_coefs[:5]:
                    self.logger.info(f"     + {name:<25} {coef:+.4f}")
            
            if negative_coefs:
                self.logger.info("   Top Negative Contributors:")
                for name, coef in negative_coefs[:5]:
                    self.logger.info(f"     - {name:<25} {coef:+.4f}")
                    
        except Exception as e:
            self.logger.error(f"Failed to log coefficients analysis: {e}")
    
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
    
    def _build_simple_correlation_model(self, results: List[BacktestResult], valid_features: List[str]) -> Dict[str, Any]:
        """Build a simple correlation-based model as fallback."""
        try:
            correlations = self._calculate_feature_quality_correlations(results)
            weights = np.array([correlations.get(feature, 0.0) for feature in valid_features])
            
            # Normalize weights
            if np.sum(np.abs(weights)) > 0:
                weights = weights / np.sum(np.abs(weights))
            
            y = np.array([r.quality_score for r in results])
            
            return {
                'model_type': 'Correlation-based',
                'feature_names': valid_features,
                'weights': weights.tolist(),
                'intercept': np.mean(y) - np.dot(weights, np.mean([[getattr(r, f, 0.0) for f in valid_features] for r in results], axis=0)),
                'r_squared': self._calculate_r_squared(np.array([[getattr(r, f, 0.0) for f in valid_features] for r in results]), y, weights),
                'feature_importance': dict(zip(valid_features, np.abs(weights)))
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to build simple correlation model: {e}")
            return {}

    def _apply_learned_rules(self, features: Dict[str, float]) -> float:
        """Apply learned rules to predict quality using Ridge Regression model."""
        if not self.learned_rules:
            return features.get('strength', 0.5)
        
        # Use the Ridge Regression model if available
        model = self.learned_rules.get('strength_scoring_model', {})
        if model and 'model_object' in model:
            try:
                # Extract features in the same order as the model
                feature_values = []
                for feature_name in model['feature_names']:
                    feature_values.append(features.get(feature_name, 0.0))
                
                # Standardize features using the fitted scaler
                feature_array = np.array(feature_values).reshape(1, -1)
                scaler = model['scaler_object']
                feature_array_scaled = scaler.transform(feature_array)
                
                # Apply the Ridge Regression model
                quality_score = model['model_object'].predict(feature_array_scaled)[0]
                
                # Ensure score is within valid range
                return min(max(quality_score, 0.0), 1.0)
                
            except Exception as e:
                self.logger.warning(f"Failed to apply Ridge Regression model: {e}")
        
        # Fallback to simple correlation-based prediction
        if model and 'weights' in model:
            try:
                # Extract features in the same order as the model
                feature_values = []
                for feature_name in model['feature_names']:
                    feature_values.append(features.get(feature_name, 0.0))
                
                # Apply the simple model
                weights = np.array(model['weights'])
                intercept = model.get('intercept', 0.0)
                
                quality_score = np.dot(feature_values, weights) + intercept
                
                # Ensure score is within valid range
                return min(max(quality_score, 0.0), 1.0)
                
            except Exception as e:
                self.logger.warning(f"Failed to apply simple correlation model: {e}")
        
        # Final fallback to correlation-based prediction
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