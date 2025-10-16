"""
Enhanced Multi-Timeframe Training System

This module provides advanced multi-timeframe model training capabilities that can be used
across all model types (general, analyst, tactician). It leverages the best tools from
src/utils/feature_selection/, src/utils/hardware/, and other src/utils/ components for
optimal performance and feature engineering.

Features:
- Advanced feature selection with regime-aware splitting
- M1/M2/M3 hardware optimization
- Intelligent feature caching
- Parallel processing optimization
- Memory-efficient processing
- Cross-timeframe feature engineering
- Model coordination and training orchestration

Timeframes supported: 1m, 5m, 15m, 30m, 1h
(Removed 1d and 4h as requested)
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import gc
import psutil
from pathlib import Path
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
from contextlib import contextmanager, nullcontext

# Common utilities
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
    safe_mean, safe_std, safe_float, safe_int, get_current_datetime,
    safe_append, safe_extend, safe_dict_get, safe_lower, safe_upper,
    format_datetime, validate_file_path, get_file_size, check_disk_space
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, MathValidationError
)
from src.utils.parquet_utils import get_parquet_utils, ParquetUtils
from src.core.decorators import (
    handles_errors, validates, traced, log_execution_time, 
    timeout, error_boundary, compose, validate_data_quality, 
    monitor_step_execution, ensure_data_integrity, validate_pipeline_step
)
from src.utils.intensity_scaler import (
    get_intensity_from_environment, get_scaled_hpo_trials,
    get_scaled_hpo_timeout, log_intensity_info
)
from src.core.errors import (
    ValidationError, DataIntegrityError, FileOperationError,
    ConfigurationError, ModelTrainingError
)
from src.utils.logger import system_logger

# Advanced feature selection tools
# NOTE: These imports are commented out as the modules no longer exist.
# Use src.feature_selection instead for feature selection functionality.
# from src.utils.feature_selection.step08_unified_complete import (
#     UnifiedFeatureSelector, RegimeDataSplitter, FinancialMetrics,
#     RiskMetrics, RegimeBalanceMetrics, FeatureSelectionValidation,
#     Step08Results, FeatureSelectionConfig
# )
# from src.utils.feature_selection.step08_optimized_methods import (
#     OptimizedFeatureSelectionMethods, AdvancedFeatureSelector
# )

# Hardware optimization tools
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager, M1GPUManager
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer, M1MemoryOptimizer
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer, M1CPUOptimizer
from src.utils.hardware.m1_optimizations import M1MemoryOptimizer as AdvancedM1MemoryOptimizer

# Parallel processing and caching
from src.utils.parallel_processing_optimizer import MacM1ParallelOptimizer
from src.utils.caching import IntelligentCache

# Enhanced data processing
from src.utils.enhanced_data_operations import EnhancedDataOperations
from src.utils.data_processing_utils import DataProcessingUtils
from src.utils.matrix_operations import VectorizedProcessingCore

# Performance monitoring
from src.utils.performance_utils import PerformanceMonitor
from src.utils.model_performance_monitor import ModelPerformanceMonitor

# Import the existing multi-timeframe components
from src.analyst.multi_timeframe_feature_engineering import MultiTimeframeFeatureEngineering
from src.analyst.predictive_ensembles.multi_timeframe_ensemble import MultiTimeframeEnsemble

@dataclass
class TimeframeConfig:
    """Enhanced configuration for each timeframe in multi-timeframe training."""
    
    timeframe: str
    weight: float
    min_samples: int = 50
    enable_training: bool = True
    feature_engineering_config: Dict[str, Any] = field(default_factory=dict)
    
    # Advanced feature selection configuration
    enable_advanced_feature_selection: bool = True
    feature_selection_method: str = "unified"  # "unified", "optimized", "basic"
    max_features: int = 100
    feature_selection_threshold: float = 0.01
    
    # Hardware optimization settings
    enable_m1_optimization: bool = True
    memory_limit_mb: int = 1024
    enable_gpu_acceleration: bool = True
    
    # Parallel processing settings
    enable_parallel_processing: bool = True
    max_workers: int = 4
    chunk_size: int = 1000
    
    def __post_init__(self):
        """Validate timeframe configuration."""
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h']
        if self.timeframe not in valid_timeframes:
            raise ValueError(f"Invalid timeframe: {self.timeframe}. Must be one of {valid_timeframes}")
        
        if not 0 <= self.weight <= 1:
            raise ValueError(f"Weight must be between 0 and 1, got {self.weight}")
        
        valid_feature_methods = ["unified", "optimized", "basic"]
        if self.feature_selection_method not in valid_feature_methods:
            raise ValueError(f"Invalid feature selection method: {self.feature_selection_method}. Must be one of {valid_feature_methods}")

@dataclass
class MultiTimeframeTrainingConfig:
    """Enhanced configuration for multi-timeframe training."""
    
    timeframes: List[TimeframeConfig]
    enable_cross_timeframe_features: bool = True
    enable_timeframe_ensemble: bool = True
    ensemble_method: str = "weighted_average"  # "weighted_average", "meta_learner", "stacking"
    min_confidence_threshold: float = 0.6
    enable_dynamic_weighting: bool = True
    weight_update_frequency: int = 100
    
    # Advanced feature selection settings
    enable_regime_aware_feature_selection: bool = True
    enable_financial_metrics: bool = True
    enable_risk_assessment: bool = True
    feature_selection_validation: bool = True
    
    # Hardware optimization settings
    enable_m1_optimization: bool = True
    enable_memory_optimization: bool = True
    enable_gpu_acceleration: bool = True
    memory_limit_gb: float = 8.0
    
    # Caching and performance settings
    enable_intelligent_caching: bool = True
    cache_dir: str = "data_cache/multi_timeframe_cache"
    max_cache_size_mb: int = 2048
    enable_parallel_processing: bool = True
    max_parallel_workers: int = 4
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    enable_model_performance_tracking: bool = True
    
    def __post_init__(self):
        """Validate multi-timeframe training configuration."""
        if not self.timeframes:
            raise ValueError("At least one timeframe must be specified")
        
        total_weight = sum(tf.weight for tf in self.timeframes)
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Timeframe weights must sum to 1.0, got {total_weight}")
        
        valid_ensemble_methods = ["weighted_average", "meta_learner", "stacking"]
        if self.ensemble_method not in valid_ensemble_methods:
            raise ValueError(f"Invalid ensemble method: {self.ensemble_method}. Must be one of {valid_ensemble_methods}")

class MultiTimeframeTrainer:
    """Enhanced multi-timeframe model trainer with advanced feature selection and hardware optimization."""
    
    def __init__(self, config: MultiTimeframeTrainingConfig, symbol: str, exchange: str):
        """Initialize the enhanced multi-timeframe trainer.
        
        Args:
            config: Multi-timeframe training configuration
            symbol: Trading symbol
            exchange: Exchange name
        """
        self.config = config
        self.symbol = symbol
        self.exchange = exchange
        self.logger = system_logger.getChild(f'EnhancedMultiTimeframeTrainer_{symbol}_{exchange}')
        
        # Initialize hardware optimization components
        self._initialize_hardware_optimization()
        
        # Initialize advanced feature selection components
        self._initialize_feature_selection()
        
        # Initialize caching and performance monitoring
        self._initialize_caching_and_monitoring()
        
        # Initialize parallel processing
        self._initialize_parallel_processing()
        
        # Initialize legacy components for compatibility
        self._initialize_legacy_components()
        
        # Training state
        self.trained_models: Dict[str, Any] = {}
        self.training_results: Dict[str, Any] = {}
        self.trained = False
        self.feature_cache: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, Any] = {}
        
        # Apply intensity scaling
        intensity_pct = get_intensity_from_environment()
        if intensity_pct < 1.0:
            self.config = self._apply_intensity_scaling(intensity_pct)
            self.logger.info(f"🔧 Applied intensity scaling ({intensity_pct*100:.0f}%) to multi-timeframe training config")
    
    def _initialize_hardware_optimization(self):
        """Initialize hardware optimization components."""
        try:
            # M1 GPU Manager
            if self.config.enable_gpu_acceleration:
                self.gpu_manager = get_m1_gpu_manager()
                self.logger.info("🚀 M1 GPU Manager initialized")
            else:
                self.gpu_manager = None
            
            # M1 Memory Optimizer
            if self.config.enable_memory_optimization:
                self.memory_optimizer = get_m1_memory_optimizer()
                self.advanced_memory_optimizer = AdvancedM1MemoryOptimizer(
                    memory_limit_gb=self.config.memory_limit_gb,
                    enable_gc_tuning=True,
                    enable_memory_leak_detection=True,
                    enable_swap_management=True
                )
                self.logger.info("🧠 M1 Memory Optimizer initialized")
            else:
                self.memory_optimizer = None
                self.advanced_memory_optimizer = None
            
            # M1 CPU Optimizer
            if self.config.enable_m1_optimization:
                self.cpu_optimizer = get_m1_cpu_optimizer()
                self.logger.info("⚡ M1 CPU Optimizer initialized")
            else:
                self.cpu_optimizer = None
                
        except Exception as e:
            self.logger.warning(f"⚠️ Hardware optimization initialization failed: {e}")
            self.gpu_manager = None
            self.memory_optimizer = None
            self.advanced_memory_optimizer = None
            self.cpu_optimizer = None
    
    def _initialize_feature_selection(self):
        """Initialize advanced feature selection components."""
        try:
            # Unified Feature Selector
            if self.config.enable_regime_aware_feature_selection:
                self.unified_feature_selector = UnifiedFeatureSelector()
                self.regime_data_splitter = RegimeDataSplitter()
                self.logger.info("🎯 Unified Feature Selector initialized")
            else:
                self.unified_feature_selector = None
                self.regime_data_splitter = None
            
            # Optimized Feature Selection Methods
            self.optimized_feature_methods = OptimizedFeatureSelectionMethods()
            self.advanced_feature_selector = AdvancedFeatureSelector()
            self.logger.info("🔧 Optimized Feature Selection Methods initialized")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature selection initialization failed: {e}")
            self.unified_feature_selector = None
            self.regime_data_splitter = None
            self.optimized_feature_methods = None
            self.advanced_feature_selector = None
    
    def _initialize_caching_and_monitoring(self):
        """Initialize caching and performance monitoring components."""
        try:
            # Intelligent Feature Cache (Unified)
            if self.config.enable_intelligent_caching:
                self.feature_cache_manager = IntelligentCache(
                    ttl_seconds=None,
                    cache_dir=self.config.cache_dir,
                    namespace=f"multi_timeframe_{self.symbol}_{self.exchange}",
                    max_memory_mb=self.config.max_cache_size_mb,
                    enable_compression=True,
                    enable_disk=True,
                )
                self.logger.info("💾 Intelligent Feature Cache initialized")
            else:
                self.feature_cache_manager = None
            
            # Performance Monitoring
            if self.config.enable_performance_monitoring:
                self.performance_monitor = PerformanceMonitor()
                self.logger.info("📊 Performance Monitor initialized")
            else:
                self.performance_monitor = None
            
            # Model Performance Monitoring
            if self.config.enable_model_performance_tracking:
                self.model_performance_monitor = ModelPerformanceMonitor()
                self.logger.info("🎯 Model Performance Monitor initialized")
            else:
                self.model_performance_monitor = None
                
        except Exception as e:
            self.logger.warning(f"⚠️ Caching and monitoring initialization failed: {e}")
            self.feature_cache_manager = None
            self.performance_monitor = None
            self.model_performance_monitor = None
    
    def _initialize_parallel_processing(self):
        """Initialize parallel processing components."""
        try:
            if self.config.enable_parallel_processing:
                self.parallel_optimizer = MacM1ParallelOptimizer(
                    max_workers=self.config.max_parallel_workers,
                    chunk_size=1000,
                    use_process_pool=True,
                    memory_limit_mb=2048
                )
                self.logger.info("⚡ Parallel Processing Optimizer initialized")
            else:
                self.parallel_optimizer = None
                
        except Exception as e:
            self.logger.warning(f"⚠️ Parallel processing initialization failed: {e}")
            self.parallel_optimizer = None
    
    def _initialize_legacy_components(self):
        """Initialize legacy components for compatibility."""
        try:
            # Legacy feature engineering
            self.feature_engine = MultiTimeframeFeatureEngineering({
                'timeframes': [tf.timeframe for tf in self.config.timeframes],
                'enable_cross_timeframe_features': self.config.enable_cross_timeframe_features
            })
            
            # Legacy ensemble
            if self.config.enable_timeframe_ensemble:
                self.ensemble = MultiTimeframeEnsemble({
                    'timeframes': [tf.timeframe for tf in self.config.timeframes],
                    'ensemble_method': self.config.ensemble_method,
                    'min_confidence_threshold': self.config.min_confidence_threshold,
                    'enable_dynamic_weighting': self.config.enable_dynamic_weighting,
                    'weight_update_frequency': self.config.weight_update_frequency
                })
            else:
                self.ensemble = None
                
        except Exception as e:
            self.logger.warning(f"⚠️ Legacy components initialization failed: {e}")
            self.feature_engine = None
            self.ensemble = None
    
    def _apply_intensity_scaling(self, intensity_pct: float) -> MultiTimeframeTrainingConfig:
        """Apply intensity scaling to the configuration."""
        # Scale down the number of timeframes if intensity is low
        if intensity_pct < 0.5:
            # Keep only the most important timeframes
            important_timeframes = ['1m', '15m', '1h']
            scaled_timeframes = [tf for tf in self.config.timeframes if tf.timeframe in important_timeframes]
            
            # Renormalize weights
            total_weight = sum(tf.weight for tf in scaled_timeframes)
            for tf in scaled_timeframes:
                tf.weight = tf.weight / total_weight
            
            return MultiTimeframeTrainingConfig(
                timeframes=scaled_timeframes,
                enable_cross_timeframe_features=self.config.enable_cross_timeframe_features,
                enable_timeframe_ensemble=self.config.enable_timeframe_ensemble,
                ensemble_method=self.config.ensemble_method,
                min_confidence_threshold=self.config.min_confidence_threshold,
                enable_dynamic_weighting=self.config.enable_dynamic_weighting,
                weight_update_frequency=self.config.weight_update_frequency,
                enable_regime_aware_feature_selection=self.config.enable_regime_aware_feature_selection,
                enable_financial_metrics=self.config.enable_financial_metrics,
                enable_risk_assessment=self.config.enable_risk_assessment,
                feature_selection_validation=self.config.feature_selection_validation,
                enable_m1_optimization=self.config.enable_m1_optimization,
                enable_memory_optimization=self.config.enable_memory_optimization,
                enable_gpu_acceleration=self.config.enable_gpu_acceleration,
                memory_limit_gb=self.config.memory_limit_gb,
                enable_intelligent_caching=self.config.enable_intelligent_caching,
                cache_dir=self.config.cache_dir,
                max_cache_size_mb=self.config.max_cache_size_mb,
                enable_parallel_processing=self.config.enable_parallel_processing,
                max_parallel_workers=max(2, int(self.config.max_parallel_workers * intensity_pct)),
                enable_performance_monitoring=self.config.enable_performance_monitoring,
                enable_model_performance_tracking=self.config.enable_model_performance_tracking
            )
        
        return self.config
    
    @handles_errors(default_return=False, context='Enhanced multi-timeframe training')
    # @log_execution_time  # Temporarily disabled due to import conflicts

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
    async def train_models(self, training_data: Dict[str, pd.DataFrame], 
                          model_trainer: Any, model_config: Dict[str, Any]) -> bool:
        """Enhanced train models across multiple timeframes with advanced feature selection and hardware optimization.
        
        Args:
            training_data: Dict mapping timeframe -> training DataFrame
            model_trainer: Model trainer instance (general, analyst, or tactician)
            model_config: Model training configuration
            
        Returns:
            bool: Success status
        """
        try:
            self.logger.info("🚀 Starting enhanced multi-timeframe model training...")
            start_time = time.time()
            
            # Start performance monitoring
            if self.performance_monitor:
                self.performance_monitor.start_monitoring()
            
            # Memory optimization context
            with self._get_memory_optimization_context():
                # 1. Advanced feature selection and preparation
                enhanced_training_data = await self._prepare_enhanced_features(training_data)
                
                # 2. Parallel timeframe training with hardware optimization
                timeframe_results = await self._train_timeframes_parallel(enhanced_training_data, model_trainer, model_config)
                
                # 3. Advanced ensemble training if enabled
                if self.config.enable_timeframe_ensemble and self.ensemble:
                    await self._train_advanced_ensemble(enhanced_training_data)
                
                # 4. Performance analysis and optimization
                await self._analyze_performance_and_optimize(timeframe_results)
            
            # Stop performance monitoring
            if self.performance_monitor:
                performance_stats = self.performance_monitor.stop_monitoring()
                self.performance_metrics['training_performance'] = performance_stats
            
            # Save enhanced training results
            await self._save_enhanced_training_results()
            
            self.trained = True
            total_time = time.time() - start_time
            
            self.logger.info("✅ Enhanced multi-timeframe training completed!")
            self.logger.info(f"⏱️ Total training time: {total_time:.2f}s")
            self._log_training_summary(timeframe_results, total_time)
            
            return True
            
        except Exception as e:
            self.logger.exception(f"💥 Error in enhanced multi-timeframe training: {e}")
            return False
    
    @contextmanager
    def _get_memory_optimization_context(self):
        """Get memory optimization context manager."""
        if self.advanced_memory_optimizer:
            with self.advanced_memory_optimizer.memory_context():
                yield
        else:
            yield
    
    @handles_errors(default_return=training_data, context='Enhanced feature preparation')
    async def _prepare_enhanced_features(self, training_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Prepare enhanced features using advanced feature selection and caching."""
        try:
            self.logger.info("🔧 Preparing enhanced features with advanced selection...")
            enhanced_data = {}
            
            for tf_config in self.config.timeframes:
                if tf_config.timeframe not in training_data:
                    continue
                
                # Check cache first
                cache_key = f"{self.symbol}_{self.exchange}_{tf_config.timeframe}_features"
                cached_features = None
                
                if self.feature_cache_manager:
                    cached_features = self.feature_cache_manager.get(cache_key)
                
                if cached_features is not None:
                    self.logger.info(f"📦 Using cached features for {tf_config.timeframe}")
                    enhanced_data[tf_config.timeframe] = cached_features
                    continue
                
                # Prepare features with advanced selection
                base_data = training_data[tf_config.timeframe].copy()
                
                # Advanced feature selection
                if tf_config.enable_advanced_feature_selection:
                    selected_features = await self._perform_advanced_feature_selection(
                        base_data, tf_config
                    )
                    if selected_features is not None:
                        base_data = selected_features
                
                # Cross-timeframe features
                if self.config.enable_cross_timeframe_features:
                    cross_features = await self._generate_cross_timeframe_features(
                        base_data, tf_config.timeframe, training_data
                    )
                    if cross_features is not None and not cross_features.empty:
                        base_data = pd.concat([base_data, cross_features], axis=1)
                
                # Cache the enhanced features
                if self.feature_cache_manager:
                    self.feature_cache_manager.set(cache_key, base_data)
                
                enhanced_data[tf_config.timeframe] = base_data
                self.logger.info(f"✅ Enhanced {len(base_data.columns)} features for {tf_config.timeframe}")
            
            return enhanced_data
            
        except Exception as e:
            self.logger.exception(f"💥 Error preparing enhanced features: {e}")
            return training_data
    
    @handles_errors(default_return=None, context='Advanced feature generation and selection')
    async def _perform_advanced_feature_selection(self, data: pd.DataFrame, tf_config: TimeframeConfig) -> Optional[pd.DataFrame]:
        """Perform advanced feature generation and selection using unified methods adapted for multi-timeframe training."""
        try:
            # Step 1: Generate cross-timeframe features using feature selection criteria
            enhanced_data = await self._generate_features_with_selection_criteria(data, tf_config)
            
            # Step 2: Apply feature selection to the enhanced dataset
            if tf_config.feature_selection_method == "unified" and self.unified_feature_selector:
                # Use unified feature selection with regime awareness
                selection_result = await self.unified_feature_selector.select_features(
                    enhanced_data, 
                    method="unified",
                    max_features=tf_config.max_features,
                    threshold=tf_config.feature_selection_threshold
                )
                
                if selection_result and hasattr(selection_result, 'selected_features'):
                    selected_data = enhanced_data[selection_result.selected_features]
                    self.logger.info(f"🎯 Unified feature generation + selection: {len(selection_result.selected_features)} features selected from {len(enhanced_data.columns)} generated")
                    return selected_data
            
            elif tf_config.feature_selection_method == "optimized" and self.optimized_feature_methods:
                # Use optimized feature selection methods
                selected_features = await self.optimized_feature_methods.select_features(
                    enhanced_data,
                    max_features=tf_config.max_features,
                    threshold=tf_config.feature_selection_threshold
                )
                
                if selected_features:
                    selected_data = enhanced_data[selected_features]
                    self.logger.info(f"🔧 Optimized feature generation + selection: {len(selected_features)} features selected from {len(enhanced_data.columns)} generated")
                    return selected_data
            
            # Fallback to basic feature selection
            return self._basic_feature_selection(enhanced_data, tf_config)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Advanced feature generation + selection failed: {e}")
            return self._basic_feature_selection(data, tf_config)
    
    @handles_errors(default_return=data, context='Feature generation with enhanced multi-criteria approach')
    async def _generate_features_with_selection_criteria(self, data: pd.DataFrame, tf_config: TimeframeConfig) -> pd.DataFrame:
        """Generate features using enhanced multi-criteria approach: MI, correlation, enhanced mRMR, enhanced LASSO, profit rate/total PnL, win rate."""
        try:
            self.logger.info(f"🔧 Generating features with enhanced multi-criteria approach for {tf_config.timeframe}")
            
            # Start with base data
            enhanced_data = data.copy()
            
            # 1. Generate features using Mutual Information (MI) criteria
            mi_features = await self._generate_features_with_mi_criteria(
                data, tf_config
            )
            
            if mi_features is not None and not mi_features.empty:
                enhanced_data = pd.concat([enhanced_data, mi_features], axis=1)
                self.logger.info(f"📊 Added {len(mi_features.columns)} MI-based features")
            
            # 2. Generate features using correlation criteria
            correlation_features = await self._generate_features_with_correlation_criteria(
                enhanced_data, tf_config
            )
            
            if correlation_features is not None and not correlation_features.empty:
                enhanced_data = pd.concat([enhanced_data, correlation_features], axis=1)
                self.logger.info(f"🔗 Added {len(correlation_features.columns)} correlation-based features")
            
            # 3. Generate features using enhanced mRMR criteria
            mrmr_features = await self._generate_features_with_enhanced_mrmr_criteria(
                enhanced_data, tf_config
            )
            
            if mrmr_features is not None and not mrmr_features.empty:
                enhanced_data = pd.concat([enhanced_data, mrmr_features], axis=1)
                self.logger.info(f"🎯 Added {len(mrmr_features.columns)} enhanced mRMR-based features")
            
            # 4. Generate features using enhanced LASSO criteria
            lasso_features = await self._generate_features_with_enhanced_lasso_criteria(
                enhanced_data, tf_config
            )
            
            if lasso_features is not None and not lasso_features.empty:
                enhanced_data = pd.concat([enhanced_data, lasso_features], axis=1)
                self.logger.info(f"📈 Added {len(lasso_features.columns)} enhanced LASSO-based features")
            
            # 5. Generate features using profit rate/total PnL criteria
            profit_features = await self._generate_features_with_profit_criteria(
                enhanced_data, tf_config
            )
            
            if profit_features is not None and not profit_features.empty:
                enhanced_data = pd.concat([enhanced_data, profit_features], axis=1)
                self.logger.info(f"💰 Added {len(profit_features.columns)} profit/PnL-based features")
            
            # 6. Generate features using win rate criteria
            winrate_features = await self._generate_features_with_winrate_criteria(
                enhanced_data, tf_config
            )
            
            if winrate_features is not None and not winrate_features.empty:
                enhanced_data = pd.concat([enhanced_data, winrate_features], axis=1)
                self.logger.info(f"🏆 Added {len(winrate_features.columns)} win rate-based features")
            
            self.logger.info(f"✅ Enhanced multi-criteria feature generation completed: {len(data.columns)} → {len(enhanced_data.columns)} features")
            return enhanced_data
            
        except Exception as e:
            self.logger.exception(f"💥 Error generating features with enhanced multi-criteria approach: {e}")
            return data
    
    @handles_errors(default_return=None, context='MI-based feature generation')
    async def _generate_features_with_mi_criteria(self, data: pd.DataFrame, tf_config: TimeframeConfig) -> Optional[pd.DataFrame]:
        """Generate cross-timeframe features using financial metrics criteria."""
        try:
            # Use financial metrics criteria to guide cross-timeframe feature generation
            cross_features = {}
            
            # 1. Price momentum across timeframes (using Sharpe ratio criteria)
            if 'close' in data.columns:
                # Generate momentum features with risk-adjusted criteria
                for lookback in [5, 10, 20]:
                    momentum = data['close'].pct_change(lookback)
                    volatility = data['close'].pct_change().rolling(lookback).std()
                    
                    # Risk-adjusted momentum (Sharpe-like ratio)
                    risk_adjusted_momentum = momentum / (volatility + 1e-8)
                    cross_features[f'{tf_config.timeframe}_momentum_{lookback}_risk_adj'] = risk_adjusted_momentum
                    
                    # Information ratio (momentum vs volatility)
                    info_ratio = momentum / (volatility + 1e-8)
                    cross_features[f'{tf_config.timeframe}_info_ratio_{lookback}'] = info_ratio
            
            # 2. Volume-price relationship across timeframes (using correlation criteria)
            if 'volume' in data.columns and 'close' in data.columns:
                # Volume-weighted price features
                vwap = (data['close'] * data['volume']).rolling(20).sum() / data['volume'].rolling(20).sum()
                cross_features[f'{tf_config.timeframe}_vwap_20'] = vwap
                
                # Volume-price correlation
                volume_price_corr = data['close'].rolling(20).corr(data['volume'])
                cross_features[f'{tf_config.timeframe}_volume_price_corr_20'] = volume_price_corr
            
            # 3. Volatility clustering features (using VaR criteria)
            if 'close' in data.columns:
                returns = data['close'].pct_change()
                
                # GARCH-like volatility features
                volatility_5 = returns.rolling(5).std()
                volatility_20 = returns.rolling(20).std()
                cross_features[f'{tf_config.timeframe}_vol_ratio_5_20'] = volatility_5 / (volatility_20 + 1e-8)
                
                # VaR-based features
                var_95 = returns.rolling(20).quantile(0.05)
                var_99 = returns.rolling(20).quantile(0.01)
                cross_features[f'{tf_config.timeframe}_var_95_20'] = var_95
                cross_features[f'{tf_config.timeframe}_var_99_20'] = var_99
            
            if cross_features:
                return pd.DataFrame(cross_features, index=data.index)
            return None
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cross-timeframe feature generation failed: {e}")
            return None
    
    @handles_errors(default_return=None, context='Regime-aware feature generation with criteria')
    async def _generate_regime_aware_features_with_criteria(self, data: pd.DataFrame, tf_config: TimeframeConfig) -> Optional[pd.DataFrame]:
        """Generate regime-aware features using regime balance criteria."""
        try:
            regime_features = {}
            
            # 1. Regime transition features (using regime balance criteria)
            if 'hmm_cluster' in data.columns:
                # Regime transition indicators
                regime_changes = (data['hmm_cluster'] != data['hmm_cluster'].shift(1)).astype(int)
                regime_features[f'{tf_config.timeframe}_regime_transition'] = regime_changes
                
                # Regime duration features
                regime_duration = data.groupby((data['hmm_cluster'] != data['hmm_cluster'].shift(1)).cumsum()).cumcount() + 1
                regime_features[f'{tf_config.timeframe}_regime_duration'] = regime_duration
                
                # Regime stability (inverse of transition frequency)
                regime_stability = 1 / (regime_changes.rolling(50).sum() + 1)
                regime_features[f'{tf_config.timeframe}_regime_stability'] = regime_stability
            
            # 2. Regime-specific volatility features (using regime balance criteria)
            if 'close' in data.columns and 'hmm_cluster' in data.columns:
                returns = data['close'].pct_change()
                
                # Regime-specific volatility
                for regime in data['hmm_cluster'].unique():
                    if pd.notna(regime):
                        regime_mask = data['hmm_cluster'] == regime
                        regime_vol = returns.where(regime_mask).rolling(20).std()
                        regime_features[f'{tf_config.timeframe}_regime_{regime}_vol'] = regime_vol
            
            # 3. Regime momentum features (using momentum criteria)
            if 'close' in data.columns and 'hmm_cluster' in data.columns:
                for regime in data['hmm_cluster'].unique():
                    if pd.notna(regime):
                        regime_mask = data['hmm_cluster'] == regime
                        regime_momentum = data['close'].pct_change(10).where(regime_mask)
                        regime_features[f'{tf_config.timeframe}_regime_{regime}_momentum'] = regime_momentum
            
            if regime_features:
                return pd.DataFrame(regime_features, index=data.index)
            return None
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime-aware feature generation failed: {e}")
            return None
    
    @handles_errors(default_return=None, context='Risk-adjusted feature generation with criteria')
    async def _generate_risk_adjusted_features_with_criteria(self, data: pd.DataFrame, tf_config: TimeframeConfig) -> Optional[pd.DataFrame]:
        """Generate risk-adjusted features using risk assessment criteria."""
        try:
            risk_features = {}
            
            if 'close' in data.columns:
                returns = data['close'].pct_change()
                
                # 1. Risk-adjusted return features (using Sharpe ratio criteria)
                for lookback in [10, 20, 50]:
                    mean_return = returns.rolling(lookback).mean()
                    volatility = returns.rolling(lookback).std()
                    
                    # Sharpe ratio
                    sharpe_ratio = mean_return / (volatility + 1e-8)
                    risk_features[f'{tf_config.timeframe}_sharpe_{lookback}'] = sharpe_ratio
                    
                    # Sortino ratio (downside deviation)
                    downside_returns = returns.where(returns < 0, 0)
                    downside_vol = downside_returns.rolling(lookback).std()
                    sortino_ratio = mean_return / (downside_vol + 1e-8)
                    risk_features[f'{tf_config.timeframe}_sortino_{lookback}'] = sortino_ratio
                
                # 2. Tail risk features (using VaR and ES criteria)
                for confidence in [0.05, 0.01]:
                    var = returns.rolling(20).quantile(confidence)
                    risk_features[f'{tf_config.timeframe}_var_{int(confidence*100)}'] = var
                    
                    # Expected Shortfall (Conditional VaR)
                    es = returns.where(returns <= var).rolling(20).mean()
                    risk_features[f'{tf_config.timeframe}_es_{int(confidence*100)}'] = es
                
                # 3. Maximum drawdown features
                cumulative_returns = (1 + returns).cumprod()
                rolling_max = cumulative_returns.rolling(50).max()
                drawdown = (cumulative_returns - rolling_max) / rolling_max
                risk_features[f'{tf_config.timeframe}_drawdown'] = drawdown
                risk_features[f'{tf_config.timeframe}_max_drawdown'] = drawdown.rolling(50).min()
            
            if risk_features:
                return pd.DataFrame(risk_features, index=data.index)
            return None
            
        except Exception as e:
            self.logger.warning(f"⚠️ Risk-adjusted feature generation failed: {e}")
            return None
    
    @handles_errors(default_return=None, context='Momentum and volatility feature generation with criteria')
    async def _generate_momentum_volatility_features_with_criteria(self, data: pd.DataFrame, tf_config: TimeframeConfig) -> Optional[pd.DataFrame]:
        """Generate momentum and volatility features using mRMR criteria."""
        try:
            momentum_vol_features = {}
            
            if 'close' in data.columns:
                returns = data['close'].pct_change()
                
                # 1. Momentum features (using mRMR criteria for relevance and redundancy)
                for lookback in [5, 10, 20, 50]:
                    # Price momentum
                    momentum = data['close'].pct_change(lookback)
                    momentum_vol_features[f'{tf_config.timeframe}_momentum_{lookback}'] = momentum
                    
                    # Momentum acceleration (second derivative)
                    momentum_acc = momentum.diff()
                    momentum_vol_features[f'{tf_config.timeframe}_momentum_acc_{lookback}'] = momentum_acc
                    
                    # RSI-like momentum
                    gains = returns.where(returns > 0, 0)
                    losses = -returns.where(returns < 0, 0)
                    avg_gain = gains.rolling(lookback).mean()
                    avg_loss = losses.rolling(lookback).mean()
                    rs = avg_gain / (avg_loss + 1e-8)
                    rsi = 100 - (100 / (1 + rs))
                    momentum_vol_features[f'{tf_config.timeframe}_rsi_{lookback}'] = rsi
                
                # 2. Volatility features (using mRMR criteria)
                for lookback in [10, 20, 50]:
                    # Standard volatility
                    volatility = returns.rolling(lookback).std()
                    momentum_vol_features[f'{tf_config.timeframe}_volatility_{lookback}'] = volatility
                    
                    # Volatility of volatility
                    vol_of_vol = volatility.rolling(10).std()
                    momentum_vol_features[f'{tf_config.timeframe}_vol_of_vol_{lookback}'] = vol_of_vol
                    
                    # Parkinson volatility (using high-low if available)
                    if 'high' in data.columns and 'low' in data.columns:
                        parkinson_vol = np.sqrt(0.25 * np.log(data['high'] / data['low']) ** 2).rolling(lookback).mean()
                        momentum_vol_features[f'{tf_config.timeframe}_parkinson_vol_{lookback}'] = parkinson_vol
                
                # 3. Cross-momentum features (using mRMR for redundancy reduction)
                if 'volume' in data.columns:
                    # Volume momentum
                    volume_momentum = data['volume'].pct_change(10)
                    momentum_vol_features[f'{tf_config.timeframe}_volume_momentum'] = volume_momentum
                    
                    # Price-volume momentum correlation
                    price_vol_corr = data['close'].pct_change(5).rolling(20).corr(data['volume'].pct_change(5))
                    momentum_vol_features[f'{tf_config.timeframe}_price_vol_corr'] = price_vol_corr
            
            if momentum_vol_features:
                return pd.DataFrame(momentum_vol_features, index=data.index)
            return None
            
        except Exception as e:
            self.logger.warning(f"⚠️ Momentum/volatility feature generation failed: {e}")
            return None
    
    def _basic_feature_selection(self, data: pd.DataFrame, tf_config: TimeframeConfig) -> pd.DataFrame:
        """Basic feature selection fallback."""
        try:
            # Simple correlation-based feature selection
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > tf_config.max_features:
                # Calculate correlations and select top features
                corr_matrix = data[numeric_cols].corr().abs()
                feature_importance = corr_matrix.mean().sort_values(ascending=False)
                selected_features = feature_importance.head(tf_config.max_features).index.tolist()
                return data[selected_features]
            return data
            
        except Exception as e:
            self.logger.warning(f"⚠️ Basic feature selection failed: {e}")
            return data
    
    @handles_errors(default_return=None, context='Cross-timeframe feature generation')
    async def _generate_cross_timeframe_features(self, base_data: pd.DataFrame, timeframe: str, 
                                               all_data: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Generate cross-timeframe features using advanced methods."""
        try:
            if not self.feature_engine:
                return None
            
            # Use legacy feature engine for cross-timeframe features
            cross_features = self.feature_engine.generate_cross_timeframe_features(
                base_data, timeframe, all_data
            )
            
            return cross_features
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cross-timeframe feature generation failed: {e}")
            return None
    
    @handles_errors(default_return={}, context='Parallel timeframe training')
    async def _train_timeframes_parallel(self, training_data: Dict[str, pd.DataFrame], 
                                       model_trainer: Any, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Train models across timeframes in parallel with hardware optimization."""
        try:
            self.logger.info("⚡ Training timeframes in parallel with hardware optimization...")
            
            if not self.parallel_optimizer:
                # Fallback to sequential training
                return await self._train_timeframes_sequential(training_data, model_trainer, model_config)
            
            # Prepare training tasks
            training_tasks = []
            for tf_config in self.config.timeframes:
                if tf_config.timeframe in training_data and tf_config.enable_training:
                    task = partial(
                        self._train_single_timeframe_optimized,
                        tf_config, training_data[tf_config.timeframe], model_trainer, model_config
                    )
                    training_tasks.append((tf_config.timeframe, task))
            
            # Execute parallel training
            timeframe_results = {}
            with ThreadPoolExecutor(max_workers=self.config.max_parallel_workers) as executor:
                future_to_timeframe = {
                    executor.submit(task): timeframe 
                    for timeframe, task in training_tasks
                }
                
                for future in as_completed(future_to_timeframe):
                    timeframe = future_to_timeframe[future]
                    try:
                        result = future.result()
                        timeframe_results[timeframe] = result
                        self.logger.info(f"✅ {timeframe} training completed")
                    except Exception as e:
                        self.logger.error(f"❌ {timeframe} training failed: {e}")
                        timeframe_results[timeframe] = {'success': False, 'error': str(e)}
            
            return timeframe_results
            
        except Exception as e:
            self.logger.exception(f"💥 Error in parallel timeframe training: {e}")
            return {}
    
    @handles_errors(default_return={'success': False}, context='Optimized single timeframe training')
    async def _train_single_timeframe_optimized(self, tf_config: TimeframeConfig, data: pd.DataFrame,
                                              model_trainer: Any, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Train models for a single timeframe with hardware optimization."""
        try:
            # GPU context if available
            gpu_context = None
            if self.gpu_manager and tf_config.enable_gpu_acceleration:
                gpu_context = self.gpu_manager.gpu_context(f"training_{tf_config.timeframe}")
            
            with gpu_context or nullcontext():
                # Memory optimization
                if self.memory_optimizer:
                    self.memory_optimizer.optimize_memory_usage()
                
                # Train models using the provided model trainer
                training_result = await model_trainer.train_models({
                    'features': data,
                    'timeframe': tf_config.timeframe,
                    'config': {**model_config, **tf_config.feature_engineering_config}
                })
                
                if training_result and training_result.get('success', False):
                    # Store trained models
                    self.trained_models[tf_config.timeframe] = training_result.get('models', {})
                    self.training_results[tf_config.timeframe] = training_result
                    
                    self.logger.info(f"✅ Trained {len(self.trained_models[tf_config.timeframe])} models for {tf_config.timeframe}")
                    return {'success': True, 'models_count': len(self.trained_models[tf_config.timeframe])}
                else:
                    self.logger.error(f"❌ Model training failed for {tf_config.timeframe}")
                    return {'success': False, 'error': 'Model training failed'}
                    
        except Exception as e:
            self.logger.exception(f"💥 Error training {tf_config.timeframe} models: {e}")
            return {'success': False, 'error': str(e)}
    
    @handles_errors(default_return={}, context='Sequential timeframe training')
    async def _train_timeframes_sequential(self, training_data: Dict[str, pd.DataFrame], 
                                         model_trainer: Any, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback sequential training for timeframes."""
        try:
            timeframe_results = {}
            
            for tf_config in self.config.timeframes:
                if tf_config.timeframe not in training_data or not tf_config.enable_training:
                    continue
                
                result = await self._train_single_timeframe_optimized(
                    tf_config, training_data[tf_config.timeframe], model_trainer, model_config
                )
                timeframe_results[tf_config.timeframe] = result
            
            return timeframe_results
            
        except Exception as e:
            self.logger.exception(f"💥 Error in sequential timeframe training: {e}")
            return {}
    
    @handles_errors(default_return=None, context='Advanced ensemble training')
    async def _train_advanced_ensemble(self, training_data: Dict[str, pd.DataFrame]):
        """Train advanced ensemble with performance monitoring."""
        try:
            self.logger.info("🧠 Training advanced multi-timeframe ensemble...")
            
            if not self.ensemble:
                return
            
            # Prepare ensemble training data
            ensemble_data = {}
            for tf_config in self.config.timeframes:
                if tf_config.timeframe in training_data and tf_config.timeframe in self.trained_models:
                    ensemble_data[tf_config.timeframe] = training_data[tf_config.timeframe]
            
            # Train ensemble with performance monitoring
            if self.model_performance_monitor:
                self.model_performance_monitor.start_tracking("ensemble_training")
            
            ensemble_result = await self.ensemble.train_ensemble(ensemble_data)
            
            if self.model_performance_monitor:
                ensemble_performance = self.model_performance_monitor.stop_tracking("ensemble_training")
                self.performance_metrics['ensemble_performance'] = ensemble_performance
            
            if ensemble_result:
                self.logger.info("✅ Advanced multi-timeframe ensemble trained successfully")
            else:
                self.logger.error("❌ Advanced multi-timeframe ensemble training failed")
                
        except Exception as e:
            self.logger.exception(f"💥 Error training advanced ensemble: {e}")
    
    @handles_errors(default_return=None, context='Performance analysis')
    async def _analyze_performance_and_optimize(self, timeframe_results: Dict[str, Any]):
        """Analyze performance and optimize based on results."""
        try:
            self.logger.info("📊 Analyzing performance and optimizing...")
            
            # Analyze training results
            successful_timeframes = [tf for tf, result in timeframe_results.items() 
                                   if result.get('success', False)]
            failed_timeframes = [tf for tf, result in timeframe_results.items() 
                               if not result.get('success', False)]
            
            self.logger.info(f"✅ Successful timeframes: {successful_timeframes}")
            if failed_timeframes:
                self.logger.warning(f"⚠️ Failed timeframes: {failed_timeframes}")
            
            # Performance optimization recommendations
            if self.performance_monitor:
                optimization_recommendations = self.performance_monitor.get_optimization_recommendations()
                if optimization_recommendations:
                    self.logger.info("🔧 Performance optimization recommendations:")
                    for recommendation in optimization_recommendations:
                        self.logger.info(f"   - {recommendation}")
            
            # Memory cleanup
            if self.advanced_memory_optimizer:
                self.advanced_memory_optimizer.cleanup_memory()
            
        except Exception as e:
            self.logger.exception(f"💥 Error in performance analysis: {e}")
    
    @handles_errors(default_return=None, context='Enhanced training results saving')
    async def _save_enhanced_training_results(self):
        """Save enhanced training results with comprehensive metadata."""
        try:
            # Enhanced results data
            results_data = {
                'config': self.config.__dict__,
                'symbol': self.symbol,
                'exchange': self.exchange,
                'trained': self.trained,
                'trained_at': get_current_datetime(),
                'training_results': self.training_results,
                'timeframe_models_count': {
                    tf: len(models) for tf, models in self.trained_models.items()
                },
                'performance_metrics': self.performance_metrics,
                'feature_cache_stats': self.feature_cache_manager.get_stats() if self.feature_cache_manager else None,
                'hardware_optimization_enabled': {
                    'gpu': self.gpu_manager is not None,
                    'memory': self.memory_optimizer is not None,
                    'cpu': self.cpu_optimizer is not None
                },
                'advanced_features_enabled': {
                    'unified_feature_selection': self.unified_feature_selector is not None,
                    'optimized_feature_methods': self.optimized_feature_methods is not None,
                    'intelligent_caching': self.feature_cache_manager is not None,
                    'parallel_processing': self.parallel_optimizer is not None
                }
            }
            
            # Save to file
            results_path = f"data_cache/enhanced_multi_timeframe_training_results_{self.symbol}_{self.exchange}_{get_current_datetime()}.json"
            ensure_directory(Path(results_path).parent)
            safe_json_dump(results_data, results_path)
            
            self.logger.info(f"💾 Enhanced training results saved to {results_path}")
            
        except Exception as e:
            self.logger.exception(f"💥 Error saving enhanced training results: {e}")
    
    def _log_training_summary(self, timeframe_results: Dict[str, Any], total_time: float):
        """Log comprehensive training summary."""
        try:
            self.logger.info("📊 Enhanced Training Summary:")
            self.logger.info(f"   ⏱️ Total time: {total_time:.2f}s")
            self.logger.info(f"   🎯 Timeframes processed: {len(timeframe_results)}")
            
            successful_count = sum(1 for result in timeframe_results.values() if result.get('success', False))
            self.logger.info(f"   ✅ Successful: {successful_count}")
            self.logger.info(f"   ❌ Failed: {len(timeframe_results) - successful_count}")
            
            total_models = sum(len(models) for models in self.trained_models.values())
            self.logger.info(f"   🤖 Total models trained: {total_models}")
            
            # Hardware optimization summary
            if self.gpu_manager:
                self.logger.info("   🚀 
            if self.memory_optimizer:
                self.logger.info("   🧠 Memory optimization: Enabled")
            if self.parallel_optimizer:
                self.logger.info("   ⚡ Parallel processing: Enabled")
            
            # Feature selection summary
            if self.unified_feature_selector:
                self.logger.info("   🎯 Unified feature selection: Enabled")
            if self.feature_cache_manager:
                cache_stats = self.feature_cache_manager.get_stats()
                self.logger.info(f"   💾 Cache hit rate: {cache_stats.get('hit_rate', 0):.2%}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error logging training summary: {e}")
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get enhanced training status and statistics."""
        return {
            'trained': self.trained,
            'symbol': self.symbol,
            'exchange': self.exchange,
            'timeframes': [tf.timeframe for tf in self.config.timeframes],
            'ensemble_enabled': self.config.enable_timeframe_ensemble,
            'ensemble_method': self.config.ensemble_method,
            'timeframe_models_count': {
                tf: len(models) for tf, models in self.trained_models.items()
            },
            'training_results_summary': {
                tf: {
                    'success': results.get('success', False),
                    'models_trained': len(results.get('models', {})),
                    'training_time': results.get('training_time', 0.0)
                } for tf, results in self.training_results.items()
            },
            'hardware_optimization_status': {
                'gpu_enabled': self.gpu_manager is not None,
                'memory_optimization_enabled': self.memory_optimizer is not None,
                'cpu_optimization_enabled': self.cpu_optimizer is not None,
                'parallel_processing_enabled': self.parallel_optimizer is not None
            },
            'advanced_features_status': {
                'unified_feature_selection_enabled': self.unified_feature_selector is not None,
                'optimized_feature_methods_enabled': self.optimized_feature_methods is not None,
                'intelligent_caching_enabled': self.feature_cache_manager is not None,
                'performance_monitoring_enabled': self.performance_monitor is not None
            },
            'performance_metrics': self.performance_metrics,
            'feature_cache_stats': self.feature_cache_manager.get_stats() if self.feature_cache_manager else None
        }
    
    @handles_errors(default_return=None, context='Enhanced multi-timeframe prediction')
    async def predict(self, prediction_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Get enhanced predictions from multi-timeframe models.
        
        Args:
            prediction_data: Dict mapping timeframe -> prediction DataFrame
            
        Returns:
            Dict with predictions and metadata
        """
        try:
            if not self.trained:
                self.logger.warning("⚠️ Models not trained, returning default prediction")
                return {
                    'prediction': 'HOLD',
                    'confidence': 0.0,
                    'timeframe_contributions': {},
                    'error': 'Models not trained'
                }
            
            # Start performance monitoring for prediction
            if self.performance_monitor:
                self.performance_monitor.start_monitoring()
            
            # Get predictions from individual timeframes with caching
            timeframe_predictions = {}
            timeframe_confidences = {}
            
            for tf_config in self.config.timeframes:
                if tf_config.timeframe not in prediction_data or tf_config.timeframe not in self.trained_models:
                    continue
                
                # Check cache for predictions
                cache_key = f"{self.symbol}_{self.exchange}_{tf_config.timeframe}_prediction"
                cached_prediction = None
                
                if self.feature_cache_manager:
                    cached_prediction = self.feature_cache_manager.get(cache_key)
                
                if cached_prediction is not None:
                    self.logger.info(f"📦 Using cached prediction for {tf_config.timeframe}")
                    timeframe_predictions[tf_config.timeframe] = cached_prediction
                    timeframe_confidences[tf_config.timeframe] = cached_prediction.get('confidence', 0.0)
                    continue
                
                # Get prediction from this timeframe's models
                tf_pred = await self._get_enhanced_timeframe_prediction(
                    tf_config.timeframe, prediction_data[tf_config.timeframe]
                )
                
                if tf_pred:
                    timeframe_predictions[tf_config.timeframe] = tf_pred
                    timeframe_confidences[tf_config.timeframe] = tf_pred.get('confidence', 0.0)
                    
                    # Cache the prediction
                    if self.feature_cache_manager:
                        self.feature_cache_manager.set(cache_key, tf_pred)
            
            # Combine predictions using ensemble if available
            if self.ensemble and timeframe_predictions:
                ensemble_pred = await self.ensemble.predict(timeframe_predictions)
                
                # Stop performance monitoring
                if self.performance_monitor:
                    prediction_performance = self.performance_monitor.stop_monitoring()
                    ensemble_pred['prediction_performance'] = prediction_performance
                
                return ensemble_pred
            else:
                # Fallback to weighted average
                result = self._weighted_average_prediction(timeframe_predictions, timeframe_confidences)
                
                # Stop performance monitoring
                if self.performance_monitor:
                    prediction_performance = self.performance_monitor.stop_monitoring()
                    result['prediction_performance'] = prediction_performance
                
                return result
                
        except Exception as e:
            self.logger.exception(f"💥 Error in enhanced multi-timeframe prediction: {e}")
            return {
                'prediction': 'HOLD',
                'confidence': 0.0,
                'timeframe_contributions': {},
                'error': str(e)
            }
    
    @handles_errors(default_return=None, context='Enhanced timeframe prediction')
    async def _get_enhanced_timeframe_prediction(self, timeframe: str, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Get enhanced prediction from a specific timeframe's models."""
        try:
            if timeframe not in self.trained_models:
                return None
            
            # Prepare features with advanced selection if available
            features = data.copy()
            
            # Apply feature selection if available
            tf_config = next((tf for tf in self.config.timeframes if tf.timeframe == timeframe), None)
            if tf_config and tf_config.enable_advanced_feature_selection:
                if self.unified_feature_selector:
                    # Use the same feature selection as training
                    selection_result = await self.unified_feature_selector.select_features(
                        features, 
                        method="unified",
                        max_features=tf_config.max_features,
                        threshold=tf_config.feature_selection_threshold
                    )
                    
                    if selection_result and hasattr(selection_result, 'selected_features'):
                        features = features[selection_result.selected_features]
            
            # Get predictions from all models for this timeframe
            model_predictions = []
            model_confidences = []
            
            for model_name, model in self.trained_models[timeframe].items():
                try:
                    # Get prediction from this model
                    pred = model.predict(features)
                    confidence = getattr(model, 'confidence', 0.5)  # Default confidence
                    
                    model_predictions.append(pred)
                    model_confidences.append(confidence)
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Error getting prediction from {model_name}: {e}")
                    continue
            
            if not model_predictions:
                return None
            
            # Combine predictions from all models in this timeframe
            avg_prediction = np.mean(model_predictions)
            avg_confidence = np.mean(model_confidences)
            
            return {
                'prediction': avg_prediction,
                'confidence': avg_confidence,
                'model_count': len(model_predictions),
                'timeframe': timeframe
            }
            
        except Exception as e:
            self.logger.exception(f"💥 Error getting enhanced {timeframe} prediction: {e}")
            return None
    
    def _weighted_average_prediction(self, timeframe_predictions: Dict[str, Dict[str, Any]], 
                                   timeframe_confidences: Dict[str, float]) -> Dict[str, Any]:
        """Combine predictions using weighted average with enhanced metadata."""
        try:
            if not timeframe_predictions:
                return {
                    'prediction': 'HOLD',
                    'confidence': 0.0,
                    'timeframe_contributions': {},
                    'ensemble_method': 'weighted_average'
                }
            
            # Calculate weighted average
            total_weight = 0.0
            weighted_prediction = 0.0
            weighted_confidence = 0.0
            
            timeframe_contributions = {}
            
            for tf_config in self.config.timeframes:
                tf = tf_config.timeframe
                if tf in timeframe_predictions and tf in timeframe_confidences:
                    weight = tf_config.weight
                    prediction = timeframe_predictions[tf]['prediction']
                    confidence = timeframe_confidences[tf]
                    
                    weighted_prediction += prediction * weight
                    weighted_confidence += confidence * weight
                    total_weight += weight
                    
                    timeframe_contributions[tf] = {
                        'prediction': prediction,
                        'confidence': confidence,
                        'weight': weight,
                        'contribution': prediction * weight,
                        'model_count': timeframe_predictions[tf].get('model_count', 0)
                    }
            
            if total_weight > 0:
                final_prediction = weighted_prediction / total_weight
                final_confidence = weighted_confidence / total_weight
            else:
                final_prediction = 0.0
                final_confidence = 0.0
            
            return {
                'prediction': final_prediction,
                'confidence': final_confidence,
                'timeframe_contributions': timeframe_contributions,
                'ensemble_method': 'weighted_average',
                'total_timeframes': len(timeframe_predictions),
                'total_models': sum(contrib.get('model_count', 0) for contrib in timeframe_contributions.values())
            }
            
        except Exception as e:
            self.logger.exception(f"💥 Error in weighted average prediction: {e}")
            return {
                'prediction': 'HOLD',
                'confidence': 0.0,
                'timeframe_contributions': {},
                'error': str(e)
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
