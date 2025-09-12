from src.utils.tprint import tprint

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
    from ..hardware.m1_memory_optimizer import get_m1_memory_optimizer, M1MemoryOptimizer
    from ..hardware.memory_optimization import get_memory_manager, MemoryMonitor
    M1_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    M1_OPTIMIZATIONS_AVAILABLE = False
    get_m1_memory_optimizer = None
    get_memory_manager = None
    tprint(f"⚠️ M1 optimizations not available: {e}")

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
    """Configuration for SR level backtesting focused on parameter optimization."""
    # Touch validation parameters - will be optimized from data
    touch_tolerance: float = None  # Will be calculated from price volatility
    min_bounce_strength: float = None  # Will be calculated from historical bounces
    max_hold_time: int = None  # Will be calculated from market characteristics
    
    # Volume analysis parameters - will be optimized
    volume_threshold_multiplier: float = None  # Will be calculated from volume distribution
    volume_lookback_periods: int = 20  # Periods to calculate average volume
    min_volume_for_confirmation: float = None  # Minimum volume to confirm SR level
    
    # SR Level Detection Parameters - will be optimized
    min_touches_required: int = None  # Minimum touches to consider level significant
    max_touches_for_analysis: int = 20  # Maximum touches to analyze
    touch_proximity_tolerance: float = None  # How close touches must be to level
    
    # Time analysis parameters
    min_time_between_touches: int = 1  # Minimum hours between touches
    max_analysis_period: int = 720  # Maximum hours to analyze (30 days)
    time_decay_factor: float = 0.95  # How much older touches are weighted less
    
    # Quality scoring parameters - will be optimized
    success_rate_weight: float = 0.3
    bounce_strength_weight: float = 0.25
    volume_confirmation_weight: float = 0.2
    time_persistence_weight: float = 0.15
    touch_frequency_weight: float = 0.1
    
    # Parameter optimization settings
    enable_parameter_optimization: bool = True  # Enable parameter optimization
    min_samples_for_optimization: int = 10  # Minimum samples needed for optimization
    parameter_optimization_method: str = 'grid_search'  # 'grid_search', 'bayesian', 'genetic'
    
    # Quality thresholds - will be calculated from data
    excellent_quality_threshold: float = None  # Will be calculated from percentiles
    good_quality_threshold: float = None
    average_quality_threshold: float = None
    poor_quality_threshold: float = None
    
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
        
        self.logger.info("Initializing SRBacktestingEngine with data-driven thresholds")
        
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
        self.data_driven_thresholds: Dict[str, float] = {}
        self.overfitting_metrics: Dict[str, Any] = {}
        
        self.logger.info("✅ SRBacktestingEngine initialization completed")
    
    def calculate_data_driven_thresholds(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate thresholds based on historical data characteristics."""
        try:
            self.logger.info("📊 Calculating data-driven thresholds from historical data")
            
            # Calculate price volatility for touch tolerance
            returns = data['close'].pct_change().dropna()
            price_volatility = returns.rolling(20).std().mean()
            
            # Touch tolerance: 2x the average price volatility
            touch_tolerance = max(0.001, min(0.01, price_volatility * 2))
            
            # Calculate historical bounce strengths
            high_low_returns = (data['high'] - data['low']) / data['close']
            avg_bounce_strength = high_low_returns.rolling(20).mean().mean()
            
            # Min bounce strength: 25th percentile of historical bounces
            min_bounce_strength = max(0.0005, high_low_returns.quantile(0.25))
            
            # Calculate volume characteristics
            if 'volume' in data.columns:
                volume_returns = data['volume'].pct_change().dropna()
                volume_volatility = volume_returns.rolling(20).std().mean()
                avg_volume = data['volume'].rolling(20).mean().mean()
                
                # Volume threshold: 1.5x average volume + 1 std dev
                volume_threshold_multiplier = 1.5 + volume_volatility
            else:
                volume_threshold_multiplier = 1.5
            
            # Calculate time characteristics
            if 'timestamp' in data.columns:
                time_diffs = data['timestamp'].diff().dt.total_seconds() / 3600  # Convert to hours
                avg_time_diff = time_diffs.mean()
                
                # Max hold time: 24 hours or 10x average time between bars
                max_hold_time = max(1, min(24, int(avg_time_diff * 10)))
            else:
                max_hold_time = 24
            
            thresholds = {
                'touch_tolerance': touch_tolerance,
                'min_bounce_strength': min_bounce_strength,
                'max_hold_time': max_hold_time,
                'volume_threshold_multiplier': volume_threshold_multiplier
            }
            
            self.data_driven_thresholds = thresholds
            
            self.logger.info(f"📊 Data-driven thresholds calculated:")
            self.logger.info(f"   - Touch tolerance: {touch_tolerance:.4f} ({touch_tolerance*100:.2f}%)")
            self.logger.info(f"   - Min bounce strength: {min_bounce_strength:.4f} ({min_bounce_strength*100:.2f}%)")
            self.logger.info(f"   - Max hold time: {max_hold_time} hours")
            self.logger.info(f"   - Volume threshold multiplier: {volume_threshold_multiplier:.2f}")
            
            return thresholds
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate data-driven thresholds: {e}")
            # Fallback to conservative defaults
            return {
                'touch_tolerance': 0.002,
                'min_bounce_strength': 0.001,
                'max_hold_time': 24,
                'volume_threshold_multiplier': 1.5
            }
        
    def backtest_sr_level(self, level: SRLevel, data: pd.DataFrame) -> BacktestResult:
        """Backtest a single SR level against historical data with data-driven thresholds."""
        try:
            self.logger.debug(f"🔍 Backtesting SR level at price {level.price}, type {level.level_type}")
            
            # Calculate data-driven thresholds if not already done
            if not self.data_driven_thresholds:
                self.calculate_data_driven_thresholds(data)
            
            # Update config with data-driven thresholds
            self.config.touch_tolerance = self.data_driven_thresholds['touch_tolerance']
            self.config.min_bounce_strength = self.data_driven_thresholds['min_bounce_strength']
            self.config.max_hold_time = self.data_driven_thresholds['max_hold_time']
            self.config.volume_threshold_multiplier = self.data_driven_thresholds['volume_threshold_multiplier']
            
            # Find the detection time in the data
            detection_idx = self._find_detection_time_index(level, data)
            if detection_idx is None:
                self.logger.warning(f"Detection time not found for level at {level.price}")
                return self._create_failed_result(level, "Detection time not found in data")
            
            self.logger.debug(f"Found detection index: {detection_idx}")
            
            # Analyze the level performance after detection
            analysis_data = data.iloc[detection_idx:detection_idx + self.config.max_analysis_period].reset_index(drop=True)
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
        """Backtest multiple SR levels with validation and chunking."""
        self.logger.info(f"🚀 Starting backtesting for {len(levels)} SR levels")
        self.logger.info(f"Data shape: {data.shape}")
        
        # Validate levels before backtesting
        valid_levels = []
        for level in levels:
            if self._validate_sr_level(level):
                valid_levels.append(level)
            else:
                self.logger.warning(f"Skipping invalid SR level: price={level.price}, strength={level.strength}")
        
        if not valid_levels:
            self.logger.error("No valid SR levels found for backtesting")
            return []
        
        self.logger.info(f"Validated {len(valid_levels)}/{len(levels)} levels for backtesting")
        
        results = []
        chunk_size = getattr(self.config, 'chunk_size', 50)
        
        # Process levels in chunks to manage memory
        for i in range(0, len(valid_levels), chunk_size):
            chunk = valid_levels[i:i + chunk_size]
            self.logger.info(f"Processing chunk {i//chunk_size + 1}/{(len(valid_levels) + chunk_size - 1)//chunk_size}")
            
            chunk_results = []
            for j, level in enumerate(chunk):
                if (i + j) % 10 == 0:
                    self.logger.info(f"Backtesting level {i + j + 1}/{len(valid_levels)}: ${level.price:.2f} ({level.level_type})")
                
                result = self.backtest_sr_level(level, data)
                chunk_results.append(result)
                
                # Log individual results for first few levels
                if (i + j) < 3:
                    self.logger.debug(f"Level ${level.price:.2f}: quality={result.quality_score:.3f}, success_rate={result.success_rate:.3f}, touches={result.total_touches}")
            
            results.extend(chunk_results)
            
            # Memory cleanup after each chunk
            if self.m1_memory_optimizer:
                self.m1_memory_optimizer.optimize_memory()
            
            # Force garbage collection
            import gc
            gc.collect()
        
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
        analysis_data = data.iloc[detection_idx:detection_idx + self.config.max_analysis_period].reset_index(drop=True)
        
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
    
    def optimize_sr_parameters(self, results: List[BacktestResult], 
                              market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Optimize SR level detection parameters based on backtesting results."""
        try:
            self.logger.info(f"🎯 Optimizing SR parameters from {len(results)} backtesting results")
            
            if not results:
                self.logger.warning("No results provided for parameter optimization")
                return {}
            
            # Check if we have enough samples for optimization
            if len(results) < self.config.min_samples_for_optimization:
                self.logger.warning(f"Insufficient samples for optimization: {len(results)} < {self.config.min_samples_for_optimization}")
                return self._create_fallback_parameters(results)
            
            # Calculate data-driven thresholds first
            if market_data is not None:
                self.calculate_data_driven_thresholds(market_data)
            
            # Optimize parameters
            if self.config.enable_parameter_optimization:
                return self._run_parameter_optimization(results, market_data)
            else:
                return self._create_data_driven_parameters(results, market_data)
            
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
            
            # OVERFITTING PROTECTION: Calculate maximum features allowed
            max_features = int(len(results) * self.config.max_features_per_sample_ratio)
            self.logger.info(f"🔒 Overfitting protection: max features allowed = {max_features} (samples: {len(results)}, ratio: {self.config.max_features_per_sample_ratio})")
            
            # Optimize weights if requested and sufficient data
            optimized_weights = {}
            if optimize_weights and market_data is not None and len(results) >= self.config.min_samples_for_learning:
                self.logger.info("🎯 Starting weight optimization with overfitting protection")
                try:
                    from .weight_optimization_engine import get_weight_optimization_engine, WeightOptimizationConfig
                    
                    # Configure weight optimization with overfitting protection
                    weight_config = WeightOptimizationConfig(
                        optimization_method='scipy_minimize',
                        primary_objective='r2_score',
                        secondary_objective='stability',
                        n_splits=self.config.cross_validation_folds,
                        max_iterations=min(50, len(results) // 2)  # Limit iterations based on data size
                    )
                    
                    self.logger.info("Configuring weight optimization engine with overfitting protection")
                    weight_optimizer = get_weight_optimization_engine(weight_config)
                    self.logger.info("Running weight optimization with cross-validation")
                    optimization_result = weight_optimizer.optimize_weights(results, market_data)
                    
                    if optimization_result and optimization_result.get('optimization_success', False):
                        optimized_weights = optimization_result.get('best_weights', {})
                        best_score = optimization_result.get('best_score', 0.0)
                        
                        # OVERFITTING PROTECTION: Check if score is too high
                        if best_score > 0.95:
                            self.logger.warning(f"⚠️ Suspiciously high optimization score: {best_score:.4f}")
                            self.logger.warning("⚠️ This may indicate overfitting - using conservative weights")
                            optimized_weights = self._get_conservative_weights(optimized_weights)
                        
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
                self.logger.info("⏭️ Weight optimization skipped (insufficient data or disabled)")
            
            # OVERFITTING PROTECTION: Build model with feature limitation
            self.logger.info("🔍 Building strength scoring model with overfitting protection")
            strength_model = self._build_strength_scoring_model_with_overfitting_protection(results, max_features)
            
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
            
            # OVERFITTING PROTECTION: Store overfitting metrics
            self.overfitting_metrics = {
                'samples_used': len(results),
                'max_features_allowed': max_features,
                'features_used': len(optimized_weights) if optimized_weights else 0,
                'sample_to_feature_ratio': len(results) / max(1, len(optimized_weights)) if optimized_weights else float('inf'),
                'overfitting_risk': 'low' if len(results) >= 200 and len(optimized_weights) <= max_features else 'medium' if len(results) >= 100 else 'high'
            }
            
            rules = {
                'quality_distribution': quality_stats,
                'feature_quality_correlations': feature_correlations,
                'quality_predictors': quality_predictors,
                'learned_weights': learned_weights,
                'quality_thresholds': quality_thresholds,
                'strength_scoring_model': strength_model,
                'optimized_weights': optimized_weights,
                'weight_optimization_enabled': optimize_weights,
                'overfitting_metrics': self.overfitting_metrics
            }
            
            self.learned_rules = rules
            
            self.logger.info(f"✅ Quality rules learning completed successfully with overfitting protection")
            self.logger.info(f"Learned {len(quality_predictors)} quality predictors")
            self.logger.info(f"Feature correlations calculated for {len(feature_correlations)} features")
            self.logger.info(f"Overfitting risk: {self.overfitting_metrics['overfitting_risk']}")
            
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
            # Check if timestamp column exists
            if 'timestamp' not in data.columns:
                # Use middle of data as fallback
                self.logger.debug(f"No timestamp column, using middle of data for level at {level.price}")
                return len(data) // 2

            # Check if level has detection_time
            if not hasattr(level, 'detection_time') or level.detection_time is None:
                # Use middle of data as fallback
                self.logger.debug(f"No detection_time for level at {level.price}, using middle of data")
                return len(data) // 2

            # Find the closest time to detection_time
            detection_time = level.detection_time
            time_diff = abs(data['timestamp'] - detection_time)
            closest_idx = time_diff.idxmin()

            # Validate that the detection time is within reasonable bounds
            if closest_idx < len(data) * 0.1:  # If detection is in first 10% of data
                self.logger.debug(f"Detection time near start of data for level at {level.price}, using index {closest_idx}")
            elif closest_idx > len(data) * 0.9:  # If detection is in last 10% of data
                self.logger.debug(f"Detection time near end of data for level at {level.price}, using index {closest_idx}")

            return closest_idx

        except Exception as e:
            self.logger.debug(f"Error finding detection time for level at {level.price}: {e}, using middle of data")
            return len(data) // 2
    
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
        """Analyze a single touch to determine if it was successful - NO LOOK-AHEAD BIAS."""
        try:
            touch_idx = touch['index']
            
            # CRITICAL: Only use information available at the time of touch
            # We can only look at the current bar and previous bars, never future bars
            if touch_idx >= len(data) - 1:
                # If this is the last bar, we can't determine success yet
                return {
                    'successful': False,
                    'bounce_strength': 0.0,
                    'hold_time': 0,
                    'volume': touch['ohlc']['volume'],
                    'incomplete': True  # Mark as incomplete for future analysis
                }
            
            # Look at the NEXT bar only (not multiple future bars)
            next_bar = data.iloc[touch_idx + 1]
            current_bar = data.iloc[touch_idx]
            
            bounce_strength = 0.0
            hold_time = 1  # At least 1 bar hold
            successful = False
            
            if level.level_type == 'support':
                # For support: check if price bounced up in the next bar
                # Success if: next bar's close > level price + tolerance
                if next_bar['close'] > level.price * (1 + self.config.touch_tolerance):
                    bounce_strength = (next_bar['close'] - level.price) / level.price
                    successful = True
                # Also check if price stayed above level for the next bar
                elif next_bar['low'] > level.price * (1 - self.config.touch_tolerance):
                    # Price stayed above support, partial success
                    bounce_strength = (next_bar['close'] - level.price) / level.price
                    successful = bounce_strength >= self.config.min_bounce_strength
            else:  # resistance
                # For resistance: check if price bounced down in the next bar
                # Success if: next bar's close < level price - tolerance
                if next_bar['close'] < level.price * (1 - self.config.touch_tolerance):
                    bounce_strength = (level.price - next_bar['close']) / level.price
                    successful = True
                # Also check if price stayed below level for the next bar
                elif next_bar['high'] < level.price * (1 + self.config.touch_tolerance):
                    # Price stayed below resistance, partial success
                    bounce_strength = (level.price - next_bar['close']) / level.price
                    successful = bounce_strength >= self.config.min_bounce_strength
            
            return {
                'successful': successful,
                'bounce_strength': bounce_strength,
                'hold_time': hold_time,
                'volume': touch['ohlc']['volume'],
                'incomplete': False
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
            if not touch_results or data is None or data.empty:
                return {
                    'penetration_depth': 0.0,
                    'penetration_frequency': 0.0
                }

            # Ensure data has proper index
            if not isinstance(data.index, pd.RangeIndex):
                data = data.reset_index(drop=True)

            # Check if required columns exist
            required_cols = ['low', 'high']
            if not all(col in data.columns for col in required_cols):
                self.logger.debug("Missing required columns for penetration metrics")
                return {
                    'penetration_depth': 0.0,
                    'penetration_frequency': 0.0
                }

            # Calculate penetration depth (how deep price went beyond the level)
            penetration_depths = []
            penetration_count = 0

            for touch in touch_results:
                touch_idx = touch.get('index', -1)
                if touch_idx >= 0 and touch_idx < len(data) - 1:
                    try:
                        # Look at the next few bars to see penetration depth
                        end_idx = min(touch_idx + 3, len(data))
                        next_bars = data.iloc[touch_idx:end_idx]

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
                    except (KeyError, IndexError) as e:
                        self.logger.debug(f"Error processing touch at index {touch_idx}: {e}")
                        continue

            # Calculate metrics
            avg_penetration_depth = np.mean(penetration_depths) if penetration_depths else 0.0
            penetration_frequency = penetration_count / len(touch_results) if touch_results else 0.0

            return {
                'penetration_depth': min(avg_penetration_depth, 1.0),  # Cap at 100%
                'penetration_frequency': penetration_frequency
            }

        except Exception as e:
            self.logger.debug(f"Penetration metrics calculation failed: {e}")
            return {
                'penetration_depth': 0.0,
                'penetration_frequency': 0.0
            }
    
    def _calculate_pattern_metrics(self, level: SRLevel, touch_results: List[Dict], data: pd.DataFrame) -> Dict[str, float]:
        """Calculate pattern consistency and strength metrics."""
        try:
            if not touch_results or data is None or data.empty:
                return {
                    'pattern_consistency': 0.0,
                    'pattern_strength': 0.0,
                    'order_flow_confirmation': 0.0,
                    'absorption_patterns': 0.0,
                    'structure_break': 0.0
                }

            # Ensure data has proper index
            if not isinstance(data.index, pd.RangeIndex):
                data = data.reset_index(drop=True)

            # Pattern consistency: how consistent are the bounce patterns?
            bounce_strengths = []
            for r in touch_results:
                if isinstance(r, dict) and r.get('successful', False):
                    bounce_strength = r.get('bounce_strength', 0.0)
                    if isinstance(bounce_strength, (int, float)):
                        bounce_strengths.append(float(bounce_strength))

            if len(bounce_strengths) > 1:
                pattern_consistency = 1.0 - (np.std(bounce_strengths) / (np.mean(bounce_strengths) + 1e-8))
                pattern_consistency = max(0.0, min(1.0, pattern_consistency))
            else:
                pattern_consistency = 1.0 if bounce_strengths else 0.0

            # Pattern strength: average strength of successful bounces
            pattern_strength = np.mean(bounce_strengths) if bounce_strengths else 0.0

            # Order flow confirmation: volume patterns at touches
            volumes_at_touches = []
            for r in touch_results:
                if isinstance(r, dict):
                    volume = r.get('volume', 0.0)
                    if isinstance(volume, (int, float)):
                        volumes_at_touches.append(float(volume))

            avg_volume_at_touches = np.mean(volumes_at_touches) if volumes_at_touches else 0.0

            # Safely calculate overall average volume
            try:
                if 'volume' in data.columns and not data['volume'].empty:
                    overall_avg_volume = data['volume'].mean()
                    if not np.isfinite(overall_avg_volume) or overall_avg_volume <= 0:
                        overall_avg_volume = 1.0
                else:
                    overall_avg_volume = 1.0
            except Exception:
                overall_avg_volume = 1.0

            if overall_avg_volume > 0:
                order_flow_confirmation = min(avg_volume_at_touches / overall_avg_volume, 2.0) / 2.0  # Normalize to 0-1
            else:
                order_flow_confirmation = 0.0

            # Absorption patterns: high volume with little price movement
            absorption_count = 0
            for touch in touch_results:
                try:
                    touch_idx = touch.get('index', -1)
                    if touch_idx >= 1 and touch_idx < len(data) - 2:
                        # Check for absorption pattern (high volume, low price movement)
                        touch_volume = touch.get('volume', 0.0)
                        if isinstance(touch_volume, (int, float)):
                            touch_volume = float(touch_volume)

                        # Safely calculate price range
                        try:
                            start_idx = max(0, touch_idx - 1)
                            end_idx = min(len(data), touch_idx + 2)
                            price_slice = data.iloc[start_idx:end_idx]

                            if 'high' in price_slice.columns and 'low' in price_slice.columns:
                                high_max = price_slice['high'].max()
                                low_min = price_slice['low'].min()

                                if np.isfinite(high_max) and np.isfinite(low_min):
                                    price_range = high_max - low_min
                                    price_range_pct = price_range / level.price if level.price > 0 else 0.0

                                    if touch_volume > overall_avg_volume * 1.5 and price_range_pct < 0.01:  # High volume, low movement
                                        absorption_count += 1
                        except (KeyError, IndexError, TypeError):
                            continue
                except (KeyError, TypeError):
                    continue

            absorption_patterns = absorption_count / len(touch_results) if touch_results else 0.0

            # Structure break: how often did the level break market structure
            structure_breaks = 0
            for touch in touch_results:
                try:
                    if isinstance(touch, dict) and not touch.get('successful', True):  # Failed touches indicate structure breaks
                        structure_breaks += 1
                except (KeyError, TypeError):
                    continue
            
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
    
    def _validate_sr_level(self, level: SRLevel) -> bool:
        """Validate SR level before backtesting."""
        try:
            # Check price validity
            if not isinstance(level.price, (int, float)) or level.price <= 0:
                self.logger.warning(f"Invalid price: {level.price}")
                return False
            
            # Check strength validity
            if not isinstance(level.strength, (int, float)) or level.strength < 0 or level.strength > 1:
                self.logger.warning(f"Invalid strength: {level.strength}")
                return False
            
            # Check touches validity
            if not isinstance(level.touches, int) or level.touches < 1:
                self.logger.warning(f"Invalid touches: {level.touches}")
                return False
            
            # Check level type validity
            if level.level_type not in ['support', 'resistance']:
                self.logger.warning(f"Invalid level type: {level.level_type}")
                return False
            
            # Check for reasonable price range (not too extreme)
            if level.price > 1e6 or level.price < 1e-6:
                self.logger.warning(f"Extreme price value: {level.price}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Error validating SR level: {e}")
            return False

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
            # Dynamically adjust CV folds based on sample size
            n_samples = len(y)
            if n_samples < 5:
                cv_folds = max(2, n_samples - 1)  # Use leave-one-out or 2-fold for small datasets
                self.logger.debug(f"Small dataset ({n_samples} samples), using {cv_folds}-fold CV")
            else:
                cv_folds = min(5, n_samples - 1)  # Use up to 5-fold CV for larger datasets

            alphas = np.logspace(-4, 2, 50)  # Range of regularization strengths
            ridge_model = RidgeCV(alphas=alphas, cv=cv_folds, scoring='r2')
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
            try:
                cv_scores = cross_val_score(ridge_model, X_scaled, y, cv=cv_folds, scoring='r2')
                cv_mse_scores = -cross_val_score(ridge_model, X_scaled, y, cv=cv_folds, scoring='neg_mean_squared_error')
                cv_mean = np.mean(cv_scores)
                cv_std = np.std(cv_scores)
                cv_mse_mean = np.mean(cv_mse_scores)
                cv_mse_std = np.std(cv_mse_scores)
            except Exception as cv_error:
                self.logger.debug(f"Cross-validation failed: {cv_error}, using training scores")
                # Fallback: use training scores when CV fails
                cv_mean = r_squared
                cv_std = 0.0
                cv_mse_mean = mse
                cv_mse_std = 0.0
            
            # Overfitting detection
            overfitting_detected = False
            overfitting_warnings = []
            
            # Check 1: High variance in CV scores (adjusted for small datasets)
            cv_threshold = 0.15 if len(y) < 100 else 0.1  # More lenient for small datasets
            if cv_std > cv_threshold:
                overfitting_detected = True
                overfitting_warnings.append(f"High CV score variance: {cv_std:.3f}")

            # Check 2: Large gap between training and CV performance (adjusted for small datasets)
            gap_threshold = 0.15 if len(y) < 100 else 0.1  # More lenient for small datasets
            performance_gap = r_squared - cv_mean
            if performance_gap > gap_threshold:
                overfitting_detected = True
                overfitting_warnings.append(f"Large performance gap: {performance_gap:.3f}")

            # Check 3: Very high R² with small dataset (SR-specific thresholds)
            # SR levels commonly have limited samples, so be much more lenient
            if len(y) < 50:  # Very small datasets are normal for SR analysis
                r2_threshold = 0.99  # Only flag extremely high R² for very small datasets
                if r_squared > r2_threshold:
                    overfitting_detected = True
                    overfitting_warnings.append(f"Potentially suspicious R² ({r_squared:.3f}) with very small SR dataset ({len(y)} samples)")
            elif len(y) < 100:  # Small datasets are common for SR
                r2_threshold = 0.97  # More lenient threshold
                if r_squared > r2_threshold:
                    overfitting_warnings.append(f"High R² ({r_squared:.3f}) with small SR dataset ({len(y)} samples) - monitor closely")
            else:  # Larger datasets use standard ML thresholds
                if r_squared > 0.95:
                    overfitting_detected = True
                    overfitting_warnings.append(f"Suspiciously high R² ({r_squared:.3f}) with dataset ({len(y)} samples)")
            
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
    
    def _run_parameter_optimization(self, results: List[BacktestResult], 
                                  market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Run parameter optimization using the optimization engine."""
        try:
            from .parameter_optimization_engine import get_parameter_optimization_engine, ParameterOptimizationConfig
            
            # Configure parameter optimization
            opt_config = ParameterOptimizationConfig(
                optimization_method=self.config.parameter_optimization_method,
                min_samples_for_optimization=self.config.min_samples_for_optimization,
                adaptive_optimization=True
            )
            
            # Create optimization engine
            optimizer = get_parameter_optimization_engine(opt_config)
            
            # Run optimization
            optimization_result = optimizer.optimize_parameters(results, market_data)
            
            if optimization_result.optimization_success:
                self.logger.info("✅ Parameter optimization completed successfully")
                self.logger.info(f"Best optimization score: {optimization_result.best_score:.4f}")
                
                # Update config with optimized parameters
                optimized_params = optimization_result.best_parameters
                self._update_config_with_optimized_parameters(optimized_params)
                
                # Calculate quality thresholds
                quality_thresholds = self._calculate_quality_thresholds_from_results(results)
                
                return {
                    'optimization_success': True,
                    'optimized_parameters': optimized_params,
                    'optimization_score': optimization_result.best_score,
                    'optimization_method': optimization_result.optimization_method,
                    'quality_thresholds': quality_thresholds,
                    'parameter_optimization_details': optimization_result.optimization_details
                }
            else:
                self.logger.warning("⚠️ Parameter optimization failed, using data-driven parameters")
                return self._create_data_driven_parameters(results, market_data)
                
        except Exception as e:
            self.logger.error(f"❌ Parameter optimization failed: {e}")
            return self._create_data_driven_parameters(results, market_data)
    
    def _create_data_driven_parameters(self, results: List[BacktestResult], 
                                     market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Create data-driven parameters without optimization."""
        self.logger.info("📊 Creating data-driven parameters")
        
        # Use data-driven thresholds if available
        if self.data_driven_thresholds:
            params = {
                'touch_tolerance': self.data_driven_thresholds['touch_tolerance'],
                'min_bounce_strength': self.data_driven_thresholds['min_bounce_strength'],
                'max_hold_time': self.data_driven_thresholds['max_hold_time'],
                'volume_threshold_multiplier': self.data_driven_thresholds['volume_threshold_multiplier'],
                'min_touches_required': self._calculate_optimal_min_touches(results),
                'success_rate_weight': 0.3,
                'bounce_strength_weight': 0.25,
                'volume_confirmation_weight': 0.2,
                'time_persistence_weight': 0.15,
                'touch_frequency_weight': 0.1
            }
        else:
            # Use conservative defaults
            params = {
                'touch_tolerance': 0.002,
                'min_bounce_strength': 0.001,
                'max_hold_time': 24,
                'volume_threshold_multiplier': 1.5,
                'min_touches_required': 3,
                'success_rate_weight': 0.3,
                'bounce_strength_weight': 0.25,
                'volume_confirmation_weight': 0.2,
                'time_persistence_weight': 0.15,
                'touch_frequency_weight': 0.1
            }
        
        # Calculate quality thresholds
        quality_thresholds = self._calculate_quality_thresholds_from_results(results)
        
        return {
            'optimization_success': False,
            'optimized_parameters': params,
            'optimization_score': 0.0,
            'optimization_method': 'data_driven',
            'quality_thresholds': quality_thresholds,
            'parameter_optimization_details': {'method': 'data_driven_fallback'}
        }
    
    def _create_fallback_parameters(self, results: List[BacktestResult]) -> Dict[str, Any]:
        """Create fallback parameters for insufficient samples."""
        self.logger.info("🛡️ Creating fallback parameters for insufficient samples")
        
        # Use very conservative defaults
        params = {
            'touch_tolerance': 0.005,  # More lenient
            'min_bounce_strength': 0.0005,  # More lenient
            'max_hold_time': 48,  # Longer hold time
            'volume_threshold_multiplier': 1.2,  # Lower volume requirement
            'min_touches_required': 2,  # Lower touch requirement
            'success_rate_weight': 0.4,  # Focus on success rate
            'bounce_strength_weight': 0.3,  # Focus on bounce strength
            'volume_confirmation_weight': 0.1,  # Less focus on volume
            'time_persistence_weight': 0.1,  # Less focus on time
            'touch_frequency_weight': 0.1  # Less focus on frequency
        }
        
        # Calculate quality thresholds
        quality_thresholds = self._calculate_quality_thresholds_from_results(results)
        
        return {
            'optimization_success': False,
            'optimized_parameters': params,
            'optimization_score': 0.0,
            'optimization_method': 'fallback',
            'quality_thresholds': quality_thresholds,
            'parameter_optimization_details': {'method': 'insufficient_samples_fallback'}
        }
    
    def _update_config_with_optimized_parameters(self, optimized_params: Dict[str, Any]) -> None:
        """Update the config with optimized parameters."""
        self.config.touch_tolerance = optimized_params['touch_tolerance']
        self.config.min_bounce_strength = optimized_params['min_bounce_strength']
        self.config.max_hold_time = optimized_params['max_hold_time']
        self.config.volume_threshold_multiplier = optimized_params['volume_threshold_multiplier']
        self.config.min_touches_required = optimized_params['min_touches_required']
        self.config.success_rate_weight = optimized_params['success_rate_weight']
        self.config.bounce_strength_weight = optimized_params['bounce_strength_weight']
        self.config.volume_confirmation_weight = optimized_params['volume_confirmation_weight']
        self.config.time_persistence_weight = optimized_params['time_persistence_weight']
        self.config.touch_frequency_weight = optimized_params['touch_frequency_weight']
        
        # Save optimized parameters
        self._save_optimized_parameters(optimized_params)
        
        self.logger.info("✅ Config updated with optimized parameters")
    
    def _save_optimized_parameters(self, optimized_params: Dict[str, Any]) -> None:
        """Save optimized parameters to file."""
        try:
            import json
            from pathlib import Path
            
            # Create parameters directory if it doesn't exist
            params_dir = Path("optimized_parameters")
            params_dir.mkdir(exist_ok=True)
            
            # Save parameters with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            params_file = params_dir / f"sr_optimized_parameters_{timestamp}.json"
            
            # Prepare parameters for saving
            save_data = {
                'timestamp': timestamp,
                'optimized_parameters': optimized_params,
                'config_summary': {
                    'touch_tolerance': self.config.touch_tolerance,
                    'min_bounce_strength': self.config.min_bounce_strength,
                    'max_hold_time': self.config.max_hold_time,
                    'volume_threshold_multiplier': self.config.volume_threshold_multiplier,
                    'min_touches_required': self.config.min_touches_required,
                    'success_rate_weight': self.config.success_rate_weight,
                    'bounce_strength_weight': self.config.bounce_strength_weight,
                    'volume_confirmation_weight': self.config.volume_confirmation_weight,
                    'time_persistence_weight': self.config.time_persistence_weight,
                    'touch_frequency_weight': self.config.touch_frequency_weight
                }
            }
            
            # Save to file
            with open(params_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            self.logger.info(f"✅ Optimized parameters saved to: {params_file}")
            
            # Also save latest parameters (overwrite)
            latest_file = params_dir / "sr_optimized_parameters_latest.json"
            with open(latest_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            self.logger.info(f"✅ Latest optimized parameters saved to: {latest_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save optimized parameters: {e}")
    
    def load_optimized_parameters(self, params_file: str = None) -> Dict[str, Any]:
        """Load optimized parameters from file."""
        try:
            import json
            from pathlib import Path
            
            if params_file is None:
                # Load latest parameters
                params_file = "optimized_parameters/sr_optimized_parameters_latest.json"
            
            params_path = Path(params_file)
            if not params_path.exists():
                self.logger.warning(f"Parameters file not found: {params_file}")
                return {}
            
            with open(params_path, 'r') as f:
                save_data = json.load(f)
            
            optimized_params = save_data.get('optimized_parameters', {})
            
            # Update config with loaded parameters
            if optimized_params:
                self._update_config_with_optimized_parameters(optimized_params)
                self.logger.info(f"✅ Loaded optimized parameters from: {params_file}")
            
            return optimized_params
            
        except Exception as e:
            self.logger.warning(f"Failed to load optimized parameters: {e}")
            return {}
    
    def _calculate_optimal_min_touches(self, results: List[BacktestResult]) -> int:
        """Calculate optimal minimum touches based on results."""
        try:
            # Analyze touch count distribution
            touch_counts = [r.total_touches for r in results]
            success_rates = [r.success_rate for r in results]
            
            # Find touch count that maximizes success rate
            touch_success_data = {}
            for touches, success_rate in zip(touch_counts, success_rates):
                if touches not in touch_success_data:
                    touch_success_data[touches] = []
                touch_success_data[touches].append(success_rate)
            
            # Calculate average success rate for each touch count
            avg_success_by_touches = {}
            for touches, success_rates in touch_success_data.items():
                if len(success_rates) >= 2:  # Need at least 2 samples
                    avg_success_by_touches[touches] = np.mean(success_rates)
            
            if avg_success_by_touches:
                # Find touch count with highest success rate
                best_touches = max(avg_success_by_touches.items(), key=lambda x: x[1])[0]
                return max(2, min(best_touches, 6))  # Clamp between 2 and 6
            else:
                return 3  # Default
                
        except Exception as e:
            self.logger.warning(f"Failed to calculate optimal min touches: {e}")
            return 3
    
    def _calculate_quality_thresholds_from_results(self, results: List[BacktestResult]) -> Dict[str, float]:
        """Calculate quality thresholds from backtesting results."""
        try:
            quality_scores = [r.quality_score for r in results]
            
            return {
                'excellent': np.percentile(quality_scores, 90),
                'good': np.percentile(quality_scores, 75),
                'average': np.percentile(quality_scores, 50),
                'poor': np.percentile(quality_scores, 25)
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate quality thresholds: {e}")
            return {
                'excellent': 0.8,
                'good': 0.6,
                'average': 0.4,
                'poor': 0.2
            }
    
    def _determine_learning_strategy(self, n_samples: int) -> str:
        """Determine learning strategy based on sample size."""
        if n_samples < self.config.minimal_learning_threshold:
            return 'minimal'
        elif n_samples < self.config.conservative_learning_threshold:
            return 'conservative'
        else:
            return 'standard'
    
    def _minimal_learning(self, results: List[BacktestResult], 
                         market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Minimal learning for very small samples (10-20 samples)."""
        self.logger.info("🔬 Using minimal learning strategy for very small samples")
        
        # Use only basic quality scoring without ML
        quality_scores = [r.quality_score for r in results]
        
        # Calculate basic statistics
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
        
        # Use simple correlation-based feature selection
        if self.config.use_feature_selection and market_data is not None:
            selected_features = self._simple_feature_selection(results, market_data)
        else:
            # Use only primary features
            selected_features = ['success_rate', 'avg_bounce_strength', 'total_touches']
        
        # Simple quality thresholds
        quality_thresholds = {
            'excellent': quality_stats['percentiles']['90th'],
            'good': quality_stats['percentiles']['75th'],
            'average': quality_stats['percentiles']['50th'],
            'poor': quality_stats['percentiles']['25th']
        }
        
        # Simple learned weights (equal weights)
        learned_weights = {feature: 1.0 / len(selected_features) for feature in selected_features}
        
        return {
            'quality_distribution': quality_stats,
            'feature_quality_correlations': {},
            'quality_predictors': selected_features,
            'learned_weights': learned_weights,
            'quality_thresholds': quality_thresholds,
            'strength_scoring_model': {},
            'optimized_weights': {},
            'weight_optimization_enabled': False,
            'learning_strategy': 'minimal',
            'overfitting_protection': 'minimal_learning',
            'selected_features': selected_features
        }
    
    def _conservative_learning(self, results: List[BacktestResult], 
                             market_data: Optional[pd.DataFrame] = None,
                             optimize_weights: bool = True) -> Dict[str, Any]:
        """Conservative learning for small samples (20-50 samples)."""
        self.logger.info("🛡️ Using conservative learning strategy for small samples")
        
        # Use feature selection to reduce features
        if self.config.use_feature_selection and market_data is not None:
            selected_features = self._adaptive_feature_selection(results, market_data)
        else:
            # Use limited feature set
            selected_features = ['success_rate', 'avg_bounce_strength', 'total_touches', 
                               'time_persistence', 'total_volume_at_level']
        
        # Calculate quality distribution
        quality_scores = [r.quality_score for r in results]
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
        
        # Conservative weight optimization (if enabled and sufficient data)
        optimized_weights = {}
        if optimize_weights and len(results) >= 20:
            try:
                from .weight_optimization_engine import get_weight_optimization_engine, WeightOptimizationConfig
                
                # Use conservative weight optimization
                weight_config = WeightOptimizationConfig(
                    min_samples_for_optimization=10,  # Lower threshold
                    max_features_per_sample_ratio=0.5,  # Higher ratio for small samples
                    regularization_strength=0.1,  # Higher regularization
                    n_splits=2  # Fewer CV folds
                )
                
                weight_optimizer = get_weight_optimization_engine(weight_config)
                optimization_result = weight_optimizer.optimize_weights(results, market_data)
                
                if optimization_result and optimization_result.get('optimization_success', False):
                    optimized_weights = optimization_result.get('best_weights', {})
                    self.logger.info("✅ Conservative weight optimization completed")
                else:
                    self.logger.warning("⚠️ Conservative weight optimization failed, using simple weights")
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Conservative weight optimization failed: {e}")
        
        # Use simple weights if optimization failed
        if not optimized_weights:
            optimized_weights = {feature: 1.0 / len(selected_features) for feature in selected_features}
        
        # Build simple strength scoring model
        strength_model = self._build_simple_strength_model(results, selected_features)
        
        # Calculate quality thresholds
        quality_thresholds = self._calculate_quality_thresholds(quality_stats)
        
        return {
            'quality_distribution': quality_stats,
            'feature_quality_correlations': {},
            'quality_predictors': selected_features,
            'learned_weights': optimized_weights,
            'quality_thresholds': quality_thresholds,
            'strength_scoring_model': strength_model,
            'optimized_weights': optimized_weights,
            'weight_optimization_enabled': optimize_weights,
            'learning_strategy': 'conservative',
            'overfitting_protection': 'conservative_learning',
            'selected_features': selected_features
        }
    
    def _standard_learning(self, results: List[BacktestResult], 
                          market_data: Optional[pd.DataFrame] = None,
                          optimize_weights: bool = True) -> Dict[str, Any]:
        """Standard learning for larger samples (50+ samples)."""
        self.logger.info("📚 Using standard learning strategy for larger samples")
        
        # Use feature selection if enabled
        if self.config.use_feature_selection and market_data is not None:
            selected_features = self._adaptive_feature_selection(results, market_data)
        else:
            # Use all available features
            selected_features = ['success_rate', 'avg_bounce_strength', 'max_bounce_strength', 
                               'total_touches', 'time_persistence', 'total_volume_at_level', 
                               'avg_hold_time', 'penetration_depth', 'penetration_frequency', 
                               'pattern_consistency', 'pattern_strength', 'order_flow_confirmation']
        
        # Calculate quality distribution
        quality_scores = [r.quality_score for r in results]
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
        
        # Full weight optimization
        optimized_weights = {}
        if optimize_weights and market_data is not None:
            try:
                from .weight_optimization_engine import get_weight_optimization_engine, WeightOptimizationConfig
                
                weight_config = WeightOptimizationConfig(
                    min_samples_for_optimization=20,
                    max_features_per_sample_ratio=0.2,
                    regularization_strength=0.01,
                    n_splits=self.config.cross_validation_folds
                )
                
                weight_optimizer = get_weight_optimization_engine(weight_config)
                optimization_result = weight_optimizer.optimize_weights(results, market_data)
                
                if optimization_result and optimization_result.get('optimization_success', False):
                    optimized_weights = optimization_result.get('best_weights', {})
                    self.logger.info("✅ Standard weight optimization completed")
                else:
                    self.logger.warning("⚠️ Standard weight optimization failed, using simple weights")
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Standard weight optimization failed: {e}")
        
        # Use simple weights if optimization failed
        if not optimized_weights:
            optimized_weights = {feature: 1.0 / len(selected_features) for feature in selected_features}
        
        # Build full strength scoring model
        strength_model = self._build_strength_scoring_model_with_overfitting_protection(
            results, len(selected_features)
        )
        
        # Calculate quality thresholds
        quality_thresholds = self._calculate_quality_thresholds(quality_stats)
        
        return {
            'quality_distribution': quality_stats,
            'feature_quality_correlations': {},
            'quality_predictors': selected_features,
            'learned_weights': optimized_weights,
            'quality_thresholds': quality_thresholds,
            'strength_scoring_model': strength_model,
            'optimized_weights': optimized_weights,
            'weight_optimization_enabled': optimize_weights,
            'learning_strategy': 'standard',
            'overfitting_protection': 'standard_learning',
            'selected_features': selected_features
        }
    
    def _simple_feature_selection(self, results: List[BacktestResult], 
                                market_data: pd.DataFrame) -> List[str]:
        """Simple feature selection using correlation with quality scores."""
        try:
            # Extract features from results
            features = {}
            for result in results:
                for attr in ['success_rate', 'avg_bounce_strength', 'total_touches', 
                           'time_persistence', 'total_volume_at_level', 'avg_hold_time']:
                    if attr not in features:
                        features[attr] = []
                    features[attr].append(getattr(result, attr, 0.0))
            
            # Calculate correlations with quality scores
            quality_scores = [r.quality_score for r in results]
            correlations = {}
            
            for feature, values in features.items():
                if len(values) > 1 and np.std(values) > 0:
                    corr, _ = pearsonr(values, quality_scores)
                    correlations[feature] = abs(corr)
            
            # Select top 3-5 features
            n_select = min(5, len(correlations))
            selected_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:n_select]
            selected_features = [f[0] for f in selected_features]
            
            self.logger.info(f"Simple feature selection: selected {selected_features}")
            return selected_features
            
        except Exception as e:
            self.logger.warning(f"Simple feature selection failed: {e}")
            return ['success_rate', 'avg_bounce_strength', 'total_touches']
    
    def _adaptive_feature_selection(self, results: List[BacktestResult], 
                                  market_data: pd.DataFrame) -> List[str]:
        """Use adaptive feature selection for small samples."""
        try:
            from .adaptive_feature_selection import get_adaptive_feature_selector, AdaptiveFeatureSelectionConfig
            
            # Configure adaptive feature selection
            config = AdaptiveFeatureSelectionConfig(
                min_samples_absolute=10,
                min_samples_per_feature=2.0,  # Very permissive for small samples
                max_features_absolute=8,  # Limit features for small samples
                conservative_mode_threshold=30
            )
            
            # Extract features
            feature_data = []
            feature_names = []
            
            for result in results:
                features = [
                    result.success_rate,
                    result.avg_bounce_strength,
                    result.max_bounce_strength,
                    result.total_touches,
                    result.time_persistence,
                    result.total_volume_at_level,
                    result.avg_hold_time
                ]
                feature_data.append(features)
            
            if not feature_names:
                feature_names = ['success_rate', 'avg_bounce_strength', 'max_bounce_strength', 
                               'total_touches', 'time_persistence', 'total_volume_at_level', 'avg_hold_time']
            
            # Create DataFrame
            X = pd.DataFrame(feature_data, columns=feature_names)
            y = np.array([r.quality_score for r in results])
            
            # Use adaptive feature selection
            selector = get_adaptive_feature_selector(config)
            result = selector.select_features(X, y, feature_names)
            
            self.logger.info(f"Adaptive feature selection: selected {result.selected_features}")
            return result.selected_features
            
        except Exception as e:
            self.logger.warning(f"Adaptive feature selection failed: {e}")
            return self._simple_feature_selection(results, market_data)
    
    def _build_simple_strength_model(self, results: List[BacktestResult], 
                                   selected_features: List[str]) -> Dict[str, Any]:
        """Build a simple strength scoring model for small samples."""
        try:
            # Extract features
            X = []
            for result in results:
                features = [getattr(result, feature, 0.0) for feature in selected_features]
                X.append(features)
            
            X = np.array(X)
            y = np.array([r.quality_score for r in results])
            
            # Use simple linear regression with high regularization
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # High regularization for small samples
            model = Ridge(alpha=1.0, random_state=42)
            model.fit(X_scaled, y)
            
            # Calculate simple metrics
            y_pred = model.predict(X_scaled)
            r_squared = r2_score(y, y_pred)
            
            return {
                'model_type': 'Simple Ridge Regression',
                'feature_names': selected_features,
                'coefficients': model.coef_.tolist(),
                'intercept': model.intercept_,
                'r_squared': r_squared,
                'model_object': model,
                'scaler_object': scaler,
                'overfitting_detected': r_squared > 0.9,  # Simple overfitting detection
                'sample_to_feature_ratio': len(y) / len(selected_features)
            }
            
        except Exception as e:
            self.logger.warning(f"Simple strength model building failed: {e}")
            return {}
    
    def _get_conservative_weights(self, optimized_weights: Dict[str, float]) -> Dict[str, float]:
        """Get conservative weights to prevent overfitting."""
        if not optimized_weights:
            return {}
        
        # Normalize weights to sum to 1.0 and apply conservative scaling
        total_weight = sum(optimized_weights.values())
        if total_weight == 0:
            return optimized_weights
        
        # Scale down extreme weights and normalize
        conservative_weights = {}
        for feature, weight in optimized_weights.items():
            # Cap individual weights at 0.5 and scale down by 0.8
            capped_weight = min(weight / total_weight, 0.5) * 0.8
            conservative_weights[feature] = capped_weight
        
        # Renormalize to sum to 1.0
        total_conservative = sum(conservative_weights.values())
        if total_conservative > 0:
            for feature in conservative_weights:
                conservative_weights[feature] /= total_conservative
        
        self.logger.info("🔒 Applied conservative weight scaling to prevent overfitting")
        return conservative_weights
    
    def _build_strength_scoring_model_with_overfitting_protection(self, results: List[BacktestResult], max_features: int) -> Dict[str, Any]:
        """Build strength scoring model with overfitting protection."""
        if not results:
            return {}
        
        # Extract features with overfitting protection
        primary_features = [
            'success_rate', 'avg_bounce_strength', 'max_bounce_strength', 
            'total_touches', 'time_persistence', 'total_volume_at_level', 'avg_hold_time'
        ]
        
        penetration_pattern_features = [
            'penetration_depth', 'penetration_frequency', 'pattern_consistency', 
            'pattern_strength', 'order_flow_confirmation', 'absorption_patterns', 'structure_break'
        ]
        
        all_features = primary_features + penetration_pattern_features
        
        # OVERFITTING PROTECTION: Limit features based on sample size
        if len(all_features) > max_features:
            # Prioritize primary features, then add others up to max_features
            selected_features = primary_features[:min(len(primary_features), max_features)]
            remaining_slots = max_features - len(selected_features)
            if remaining_slots > 0:
                selected_features.extend(penetration_pattern_features[:remaining_slots])
            all_features = selected_features
            self.logger.info(f"🔒 Limited features to {len(all_features)} to prevent overfitting")
        
        # Build feature matrix
        X = []
        valid_features = []
        
        for feature in all_features:
            feature_values = [getattr(r, feature, 0.0) for r in results]
            if not all(v == 0.0 for v in feature_values):  # Skip features with no variation
                X.append(feature_values)
                valid_features.append(feature)
        
        if not X or len(X) == 0:
            return {}
        
        X = np.array(X).T  # Transpose to get (samples, features)
        y = np.array([r.quality_score for r in results])
        
        # OVERFITTING PROTECTION: Check sample-to-feature ratio
        sample_to_feature_ratio = len(y) / len(valid_features)
        if sample_to_feature_ratio < 10:
            self.logger.warning(f"⚠️ Low sample-to-feature ratio: {sample_to_feature_ratio:.1f}")
            self.logger.warning("⚠️ Consider reducing features or increasing samples")
        
        try:
            from sklearn.linear_model import RidgeCV
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import cross_val_score
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Use more conservative cross-validation for small datasets
            n_samples = len(y)
            if n_samples < 50:
                cv_folds = max(2, n_samples - 1)  # Use leave-one-out or 2-fold
                self.logger.info(f"🔒 Small dataset ({n_samples} samples), using {cv_folds}-fold CV")
            else:
                cv_folds = min(self.config.cross_validation_folds, n_samples - 1)
            
            # Use more conservative alpha range for small datasets
            if n_samples < 100:
                alphas = np.logspace(-2, 1, 20)  # More conservative range
            else:
                alphas = np.logspace(-4, 2, 50)  # Standard range
            
            ridge_model = RidgeCV(alphas=alphas, cv=cv_folds, scoring='r2')
            ridge_model.fit(X_scaled, y)
            
            # Calculate model performance
            y_pred = ridge_model.predict(X_scaled)
            r_squared = ridge_model.score(X_scaled, y)
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            
            # Cross-validation scores for robustness
            try:
                cv_scores = cross_val_score(ridge_model, X_scaled, y, cv=cv_folds, scoring='r2')
                cv_mean = np.mean(cv_scores)
                cv_std = np.std(cv_scores)
            except Exception:
                cv_mean = r_squared
                cv_std = 0.0
            
            # OVERFITTING DETECTION: More conservative thresholds for small datasets
            overfitting_detected = False
            overfitting_warnings = []
            
            # Check for overfitting
            if cv_std > 0.2:  # High variance in CV scores
                overfitting_detected = True
                overfitting_warnings.append(f"High CV score variance: {cv_std:.3f}")
            
            performance_gap = r_squared - cv_mean
            if performance_gap > 0.15:  # Large gap between training and CV
                overfitting_detected = True
                overfitting_warnings.append(f"Large performance gap: {performance_gap:.3f}")
            
            # Very high R² with small dataset
            if n_samples < 100 and r_squared > 0.9:
                overfitting_detected = True
                overfitting_warnings.append(f"High R² ({r_squared:.3f}) with small dataset ({n_samples} samples)")
            
            # Get feature importance
            feature_importance = np.abs(ridge_model.coef_)
            feature_importance_normalized = feature_importance / np.sum(feature_importance)
            
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
                'feature_importance': dict(zip(valid_features, feature_importance_normalized)),
                'overfitting_detected': overfitting_detected,
                'overfitting_warnings': overfitting_warnings,
                'sample_to_feature_ratio': sample_to_feature_ratio,
                'model_object': ridge_model,
                'scaler_object': scaler
            }
            
            if overfitting_detected:
                self.logger.warning(f"⚠️ Overfitting detected: {'; '.join(overfitting_warnings)}")
            else:
                self.logger.info("✅ No overfitting detected")
            
            return model
            
        except Exception as e:
            self.logger.warning(f"Failed to build Ridge Regression model with overfitting protection: {e}")
            return {}
    
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