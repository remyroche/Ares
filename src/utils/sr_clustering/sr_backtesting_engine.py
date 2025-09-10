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

# Import enhanced matrix operations
try:
    from ..ml_common.matrix_operations import (
        get_enhanced_matrix_operations,
        m1_matrix_correlation_analysis
    )
    ENHANCED_MATRIX_OPS_AVAILABLE = True
except ImportError:
    ENHANCED_MATRIX_OPS_AVAILABLE = False
    get_enhanced_matrix_operations = None
    m1_matrix_correlation_analysis = None

# Import M1 optimization utilities
try:
    from ..hardware.m1_optimizations import get_m1_memory_optimizer, M1MemoryOptimizer
    from ..hardware.memory_optimization import get_memory_manager, MemoryMonitor
    M1_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    M1_OPTIMIZATIONS_AVAILABLE = False
    get_m1_memory_optimizer = None
    get_memory_manager = None
    print(f"⚠️ M1 optimizations not available: {e}")

# Import PyTorch for MPS acceleration
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

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
    
    # M1 optimization parameters
    enable_m1_optimizations: bool = True
    enable_gpu_acceleration: bool = True
    enable_memory_optimization: bool = True
    memory_limit_gb: float = 8.0
    chunk_size: int = 1000

class SRBacktestingEngine:
    """Engine for backtesting SR levels and learning quality rules."""
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.logger = system_logger.getChild('SRBacktestingEngine')
        
        self.logger.info("Initializing SRBacktestingEngine")
        self.logger.info(f"Configuration: touch_tolerance={self.config.touch_tolerance:.3f}, min_bounce_strength={self.config.min_bounce_strength:.3f}")
        self.logger.info(f"Weight settings: success_rate={self.config.success_rate_weight:.2f}, bounce_strength={self.config.bounce_strength_weight:.2f}")
        
        # Initialize M1 optimizations
        self.enable_m1_optimizations = self.config.enable_m1_optimizations and M1_OPTIMIZATIONS_AVAILABLE
        self.enable_gpu_acceleration = self.config.enable_gpu_acceleration and TORCH_AVAILABLE
        
        if self.enable_m1_optimizations:
            try:
                self.m1_memory_optimizer = get_m1_memory_optimizer(memory_limit_gb=self.config.memory_limit_gb)
                self.memory_monitor = get_memory_manager()
                self.logger.info("✅ M1 optimizations initialized for SR backtesting")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize M1 optimizations: {e}")
                self.enable_m1_optimizations = False
        else:
            self.m1_memory_optimizer = None
            self.memory_monitor = None
        
        self.learned_rules: Dict[str, Any] = {}
        self.performance_history: List[BacktestResult] = []
        
        self.logger.info("✅ SRBacktestingEngine initialization completed")
        
    def backtest_sr_level(self, level: SRLevel, data: pd.DataFrame) -> BacktestResult:
        """Backtest a single SR level against historical data."""
        try:
            self.logger.debug(f"🔍 Backtesting SR level at price {level.price}, type {level.level_type}")
            
            # Find the detection time in the data
            detection_idx = self._find_detection_time_index(level, data)
            if detection_idx is None:
                self.logger.warning(f"Detection time not found for level at {level.price}")
                return self._create_failed_result(level, "Detection time not found in data")
            
            self.logger.debug(f"Found detection index: {detection_idx}")
            
            # Analyze the level performance after detection
            analysis_data = data.iloc[detection_idx:detection_idx + self.config.max_analysis_period]
            self.logger.debug(f"Analysis data: {len(analysis_data)} periods")
            
            # Detect touches and analyze performance
            touches = self._detect_touches(level, analysis_data)
            
            if not touches:
                self.logger.warning(f"No touches detected for level at {level.price}")
                return self._create_failed_result(level, "No touches detected")
            
            self.logger.debug(f"Detected {len(touches)} touches")
            
            # Analyze each touch
            touch_results = []
            for i, touch in enumerate(touches):
                result = self._analyze_touch(level, touch, analysis_data)
                touch_results.append(result)
                if i % 5 == 0:  # Log progress every 5 touches
                    self.logger.debug(f"Analyzed touch {i+1}/{len(touches)}")
            
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
            
            self.logger.info(f"✅ Backtesting completed for level at {level.price}: quality={result.quality_score:.3f}, success_rate={result.success_rate:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Backtesting failed for level {level.price}: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return self._create_failed_result(level, str(e))
    
    def backtest_multiple_levels(self, levels: List[SRLevel], data: pd.DataFrame) -> List[BacktestResult]:
        """Backtest multiple SR levels."""
        results = []
        
        self.logger.info(f"🚀 Starting backtesting for {len(levels)} SR levels")
        self.logger.info(f"Data shape: {data.shape}")
        
        for i, level in enumerate(levels):
            if i % 10 == 0:
                self.logger.info(f"Backtesting level {i+1}/{len(levels)}: ${level.price:.2f} ({level.level_type})")
            
            result = self.backtest_sr_level(level, data)
            results.append(result)
            
            # Log individual results for first few levels
            if i < 3:
                self.logger.debug(f"Level ${level.price:.2f}: quality={result.quality_score:.3f}, success_rate={result.success_rate:.3f}, touches={result.total_touches}")
        
        if results:
            quality_scores = [r.quality_score for r in results]
            success_rates = [r.success_rate for r in results]
            touches = [r.total_touches for r in results]
            
            self.logger.info(f"✅ Backtesting completed for {len(results)} levels")
            self.logger.info(f"Quality statistics: mean={np.mean(quality_scores):.3f}, std={np.std(quality_scores):.3f}, min={np.min(quality_scores):.3f}, max={np.max(quality_scores):.3f}")
            self.logger.info(f"Success rate statistics: mean={np.mean(success_rates):.3f}, std={np.std(success_rates):.3f}")
            self.logger.info(f"Touch statistics: mean={np.mean(touches):.1f}, std={np.std(touches):.1f}")
        else:
            self.logger.warning("⚠️ No backtesting results generated")
        return results

    def backtest_sr_level_m1_optimized(self, level: SRLevel, data: pd.DataFrame) -> BacktestResult:
        """Backtest a single SR level with M1 optimization."""
        if not self.enable_m1_optimizations:
            self.logger.warning("⚠️ M1 optimizations not available, falling back to standard method")
            return self.backtest_sr_level(level, data)
        
        try:
            self.logger.debug(f"🚀 M1-optimized backtesting for SR level at price {level.price}, type {level.level_type}")
            
            # Memory checkpoint for M1 optimization
            with self.m1_memory_optimizer.memory_checkpoint(f"sr_backtest_{level.price}"):
                # Check if data should be processed in chunks
                data_size_mb = data.memory_usage(deep=True).sum() / (1024**2)
                
                if self.m1_memory_optimizer.should_chunk_data(data_size_mb, "sr_backtesting"):
                    self.logger.info(f"📦 Processing large dataset ({data_size_mb:.1f}MB) in chunks")
                    return self._chunked_sr_backtesting(level, data)
                
                # Use GPU acceleration for heavy computations
                if self.enable_gpu_acceleration and ENHANCED_MATRIX_OPS_AVAILABLE:
                    self.logger.info("🎯 Using GPU acceleration for SR backtesting")
                    return self._gpu_accelerated_sr_backtesting(level, data)
                
                # Standard M1-optimized processing
                return self._m1_optimized_sr_backtesting(level, data)
                
        except Exception as e:
            self.logger.error(f"❌ M1-optimized backtesting failed for level {level.price}: {e}")
            return self._create_failed_result(level, str(e))

    def _m1_optimized_sr_backtesting(self, level: SRLevel, data: pd.DataFrame) -> BacktestResult:
        """M1-optimized SR level backtesting."""
        # Find the detection time in the data
        detection_idx = self._find_detection_time_index(level, data)
        if detection_idx is None:
            self.logger.warning(f"Detection time not found for level at {level.price}")
            return self._create_failed_result(level, "Detection time not found in data")
        
        # Analyze the level performance after detection
        analysis_data = data.iloc[detection_idx:detection_idx + self.config.max_analysis_period]
        
        # M1-optimized touch detection
        touches = self._detect_touches_m1_optimized(level, analysis_data)
        
        if not touches:
            self.logger.warning(f"No touches detected for level at {level.price}")
            return self._create_failed_result(level, "No touches detected")
        
        # M1-optimized touch analysis
        touch_results = self._analyze_touches_m1_optimized(level, touches, analysis_data)
        
        # M1-optimized performance metrics calculation
        performance_metrics = self._calculate_performance_metrics_m1_optimized(level, touch_results, analysis_data)
        
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
        
        self.logger.info(f"✅ M1-optimized backtesting completed for level at {level.price}: quality={result.quality_score:.3f}, success_rate={result.success_rate:.3f}")
        
        return result

    def _detect_touches_m1_optimized(self, level: SRLevel, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """M1-optimized touch detection."""
        touches = []
        tolerance = level.price * self.config.touch_tolerance
        
        # Use M1 memory-efficient operations
        if self.m1_memory_optimizer:
            # Check if data should be processed in chunks
            data_size_mb = data.memory_usage(deep=True).sum() / (1024**2)
            
            if self.m1_memory_optimizer.should_chunk_data(data_size_mb, "touch_detection"):
                return self._chunked_touch_detection(level, data, tolerance)
        
        # Standard touch detection with M1 memory optimization
        for idx, row in data.iterrows():
            if self._is_touch(level, row, tolerance):
                touch = {
                    'index': idx,
                    'timestamp': row.name if hasattr(row.name, 'to_pydatetime') else idx,
                    'price': row['close'],
                    'volume': row.get('volume', 0),
                    'high': row.get('high', row['close']),
                    'low': row.get('low', row['close'])
                }
                touches.append(touch)
        
        # M1 memory cleanup
        if self.m1_memory_optimizer:
            self.m1_memory_optimizer.optimize_memory()
        
        return touches

    def _analyze_touches_m1_optimized(self, level: SRLevel, touches: List[Dict[str, Any]], data: pd.DataFrame) -> List[Dict[str, Any]]:
        """M1-optimized touch analysis."""
        touch_results = []
        
        # Use M1 memory-efficient operations
        if self.m1_memory_optimizer:
            with self.m1_memory_optimizer.memory_checkpoint("touch_analysis"):
                for i, touch in enumerate(touches):
                    result = self._analyze_touch(level, touch, data)
                    touch_results.append(result)
                    
                    # Memory cleanup every 10 touches
                    if i % 10 == 0:
                        self.m1_memory_optimizer.optimize_memory()
        else:
            for touch in touches:
                result = self._analyze_touch(level, touch, data)
                touch_results.append(result)
        
        return touch_results

    def _calculate_performance_metrics_m1_optimized(self, level: SRLevel, touch_results: List[Dict[str, Any]], data: pd.DataFrame) -> Dict[str, float]:
        """M1-optimized performance metrics calculation."""
        if not touch_results:
            return self._get_default_metrics()
        
        # Use M1 memory-efficient operations
        if self.m1_memory_optimizer:
            with self.m1_memory_optimizer.memory_checkpoint("performance_metrics"):
                return self._calculate_performance_metrics(level, touch_results, data)
        else:
            return self._calculate_performance_metrics(level, touch_results, data)
    
    def learn_quality_rules(self, results: List[BacktestResult], 
                           optimize_weights: bool = True,
                           market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Learn quality rules from backtesting results using continuous strength scoring."""
        try:
            self.logger.info(f"🧠 Learning quality rules from {len(results)} backtesting results")
            
            if not results:
                self.logger.warning("No results provided for learning quality rules")
                return {}
            
            # Use continuous quality scoring instead of binary categories
            quality_scores = [r.quality_score for r in results]
            
            # Calculate quality distribution statistics
            self.logger.info("📊 Calculating quality distribution statistics")
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
            
            self.logger.info(f"Quality distribution: mean={quality_stats['mean']:.3f}, std={quality_stats['std']:.3f}, min={quality_stats['min']:.3f}, max={quality_stats['max']:.3f}")
            self.logger.info(f"Quality percentiles: 25th={quality_stats['percentiles']['25th']:.3f}, 50th={quality_stats['percentiles']['50th']:.3f}, 75th={quality_stats['percentiles']['75th']:.3f}, 90th={quality_stats['percentiles']['90th']:.3f}")
            
            # Optimize weights if requested
            optimized_weights = {}
            if optimize_weights and market_data is not None:
                self.logger.info("🎯 Starting weight optimization")
                try:
                    from .weight_optimization_engine import get_weight_optimization_engine, WeightOptimizationConfig
                    
                    # Configure weight optimization
                    weight_config = WeightOptimizationConfig(
                        optimization_method='scipy_minimize',
                        primary_objective='r2_score',
                        secondary_objective='stability'
                    )
                    
                    self.logger.info("Configuring weight optimization engine")
                    weight_optimizer = get_weight_optimization_engine(weight_config)
                    self.logger.info("Running weight optimization")
                    optimization_result = weight_optimizer.optimize_weights(results, market_data)
                    
                    if optimization_result and optimization_result.get('optimization_success', False):
                        optimized_weights = optimization_result.get('best_weights', {})
                        best_score = optimization_result.get('best_score', 0.0)
                        
                        self.logger.info(f"✅ Weight optimization completed successfully")
                        self.logger.info(f"Best optimization score: {best_score:.4f}")
                        self.logger.info(f"Optimized weights for {len(optimized_weights)} features")
                        
                        # Log top optimized weights
                        if optimized_weights:
                            sorted_weights = sorted(optimized_weights.items(), key=lambda x: x[1], reverse=True)
                            self.logger.info("Top 5 optimized weights:")
                            for feature, weight in sorted_weights[:5]:
                                self.logger.info(f"  {feature}: {weight:.3f}")
                    else:
                        self.logger.warning("⚠️ Weight optimization failed, using default weights")
                        
                except Exception as e:
                    self.logger.error(f"❌ Weight optimization failed: {e}")
                    import traceback
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
            else:
                self.logger.info("⏭️ Weight optimization skipped")
            
            # Analyze quality-based feature relationships
            self.logger.info("🔍 Building strength scoring model")
            strength_model = self._build_strength_scoring_model(results)
            
            # Log comprehensive feature analysis
            if strength_model:
                self.logger.info("📊 Logging comprehensive feature analysis")
                self._log_comprehensive_feature_analysis(results, strength_model)
            else:
                self.logger.warning("⚠️ Failed to build strength scoring model")
            
            # Calculate feature correlations and predictors
            self.logger.info("🔗 Calculating feature quality correlations")
            feature_correlations = self._calculate_feature_quality_correlations(results)
            
            self.logger.info("🎯 Identifying quality predictors")
            quality_predictors = self._identify_quality_predictors(results)
            
            self.logger.info("⚖️ Learning feature weights")
            learned_weights = self._learn_feature_weights(results)
            
            self.logger.info("📏 Calculating quality thresholds")
            quality_thresholds = self._calculate_quality_thresholds(quality_stats)
            
            rules = {
                'quality_distribution': quality_stats,
                'feature_quality_correlations': feature_correlations,
                'quality_predictors': quality_predictors,
                'learned_weights': learned_weights,
                'quality_thresholds': quality_thresholds,
                'strength_scoring_model': strength_model,
                'optimized_weights': optimized_weights,
                'weight_optimization_enabled': optimize_weights
            }
            
            self.learned_rules = rules
            
            self.logger.info(f"✅ Quality rules learning completed successfully")
            self.logger.info(f"Learned {len(quality_predictors)} quality predictors")
            self.logger.info(f"Feature correlations calculated for {len(feature_correlations)} features")
            
            return rules
            
        except Exception as e:
            self.logger.error(f"❌ Failed to learn quality rules: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {}
    
    def predict_level_quality(self, level: SRLevel, data: pd.DataFrame) -> float:
        """Predict the quality of a level using learned rules."""
        try:
            self.logger.debug(f"🔮 Predicting quality for level at price {level.price}, type {level.level_type}")
            
            if not self.learned_rules:
                self.logger.warning("No learned rules available, using original strength")
                return level.strength  # Fallback to original strength
            
            # Extract features for prediction
            features = self._extract_prediction_features(level, data)
            self.logger.debug(f"Extracted {len(features)} features for prediction")
            
            # Apply learned rules
            quality_score = self._apply_learned_rules(features)
            
            self.logger.debug(f"Predicted quality: {quality_score:.3f} (original strength: {level.strength:.3f})")
            
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
            'volatility_regime',      # From step06: Volatility regime
            'trend_strength',         # From step06: Trend strength
            'volume_regime',          # From step06: Volume regime
            'time_of_day_effect',     # From step06: Time of day effects
            # Actual step06 momentum features
            'rsi_momentum',           # From step06: RSI momentum
            'macd_momentum',          # From step06: MACD momentum
            'roc_momentum',           # From step06: Rate of Change momentum
            'stochastic_momentum',    # From step06: Stochastic momentum
            'cci_momentum',           # From step06: Commodity Channel Index momentum
            'momentum_acceleration',  # From step06: Momentum acceleration (ROC difference)
            'momentum_volume_interaction',  # From step06: Momentum-volume interaction
            # Additional step06 features
            'bb_squeeze',             # From step06: Bollinger Band squeeze
            'bb_position',            # From step06: Bollinger Band position
            'obv_normalized',         # From step06: Normalized OBV
            'mfi_momentum',           # From step06: Money Flow Index momentum
            'williams_momentum',      # From step06: Williams %R momentum
            'adx_trend',              # From step06: Average Directional Index
            'cross_timeframe_momentum',  # From step06: Cross-timeframe momentum
            'macd_signal_strength',   # From step06: MACD signal strength
            'macd_histogram'          # From step06: MACD histogram
        ]
        
        all_features = primary_features + penetration_pattern_features + step06_features
        correlations = {}
        
        quality_scores = [r.quality_score for r in results]
        
        # Use enhanced matrix operations for correlation calculation if available
        if ENHANCED_MATRIX_OPS_AVAILABLE and m1_matrix_correlation_analysis:
            # Prepare data matrix for enhanced correlation analysis
            feature_data = []
            for feature in all_features:
                feature_values = [getattr(r, feature, 0.0) for r in results]
                feature_data.append(feature_values)
            
            # Add quality scores as the last column
            feature_data.append(quality_scores)
            
            # Convert to numpy array and transpose for correlation analysis
            data_matrix = np.array(feature_data).T
            
            # Use enhanced M1-optimized correlation analysis
            correlation_matrix = m1_matrix_correlation_analysis(data_matrix)
            
            # Extract correlations with quality scores (last column)
            quality_correlations = correlation_matrix[:, -1]
            
            for i, feature in enumerate(all_features):
                correlations[feature] = quality_correlations[i] if not np.isnan(quality_correlations[i]) else 0.0
        else:
            # Fallback to standard correlation calculation
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
                elif any(x in feature.lower() for x in ['volatility', 'trend', 'rsi', 'macd', 'roc', 'stochastic', 'cci', 'momentum', 'bb', 'obv', 'mfi', 'williams', 'adx', 'cross_timeframe']):
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
        self.logger.info("📊 Generating quality rules summary")
        
        if not self.learned_rules:
            self.logger.warning("No rules learned yet")
            return {'status': 'No rules learned yet'}
        
        summary = {
            'status': 'Rules learned',
            'quality_threshold': self.learned_rules.get('quality_threshold', 0.0),
            'discriminative_features': list(self.learned_rules.get('discriminative_features', {}).keys()),
            'performance_thresholds': self.learned_rules.get('performance_thresholds', {}),
            'total_levels_analyzed': len(self.performance_history)
        }
        
        self.logger.info(f"Quality rules summary: {summary}")
        
        return summary

def create_sr_level_from_dict(level_dict: Dict[str, Any]) -> SRLevel:
    """Create SRLevel from dictionary."""
    logger = system_logger.getChild('SRBacktestingEngine')
    
    try:
        sr_level = SRLevel(
            price=level_dict['price'],
            level_type=level_dict.get('type', 'support'),
            strength=level_dict.get('strength', 0.5),
            touches=level_dict.get('touches', 1),
            detection_time=level_dict.get('detection_time', datetime.now()),
            metadata=level_dict.get('metadata', {})
        )
        
        logger.debug(f"Created SRLevel: price={sr_level.price}, type={sr_level.level_type}, strength={sr_level.strength}")
        return sr_level
        
    except Exception as e:
        logger.error(f"❌ Failed to create SRLevel from dict: {e}")
        raise

def get_backtesting_engine(config: Optional[BacktestConfig] = None) -> SRBacktestingEngine:
    """Get a backtesting engine instance."""
    logger = system_logger.getChild('SRBacktestingEngine')
    logger.info("Creating new SRBacktestingEngine instance")
    
    try:
        instance = SRBacktestingEngine(config)
        logger.info("✅ Successfully created SRBacktestingEngine instance")
        return instance
    except Exception as e:
        logger.error(f"❌ Failed to create SRBacktestingEngine instance: {e}")
        raise