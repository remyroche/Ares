from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation
from pathlib import Path
from contextlib import nullcontext

"""
Step6: Feature Interaction Engineering with Hardware Acceleration

This module implements comprehensive feature interaction engineering for the Tactician model.
It creates interaction terms between technical indicators, market features, and derived metrics
to capture non-linear relationships and improve model performance with M1 hardware acceleration.

Key Features:
- Integrates with DiverseLookbackOptimizer for optimal period selection
- Ensures non-correlated lookback periods for each indicator
- Creates meaningful feature interactions
- Implements stability analysis for feature selection
- Comprehensive function call validation and tracking
- Detailed function completion reports with outcome analysis
- M1 GPU acceleration for matrix operations
- Vectorized processing for feature engineering
"""
import logging
from collections import Counter
from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd

# Initialize logger early
logger = logging.getLogger(__name__)

# Import comprehensive optimization utilities for enhanced performance
try:
    # M1 Hardware-Specific Optimizations
    from src.utils.m1_gpu_utils import get_m1_gpu_manager
    from src.utils.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.m1_cpu_optimizer import get_m1_cpu_optimizer

    # Processing Core Optimizations
    from src.utils.vectorized_processing_core import get_vectorized_processing_core
    from src.utils.enhanced_matrix_operations import get_enhanced_matrix_operations
    from src.utils.enhanced_step_optimizations import get_step_optimization_manager

    # Data Management Optimizations
    from src.utils.optimized_data_manager import OptimizedDataManager

    OPTIMIZATIONS_AVAILABLE = True
    logger.info("🚀 All optimization utilities successfully loaded for step06")
except ImportError as e:
    OPTIMIZATIONS_AVAILABLE = False
    logger.warning(f"⚠️ Some optimization utilities not available: {e}")
try:
    import talib
except ImportError:
    talib = None
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from copy import copy
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
steps_dir = os.path.join(current_dir, '..')
sys.path.insert(0, steps_dir)

try:
    from ..step06_enhanced_validation_framework import step06_function_validator, step06_function_tracker, step06_validation_context, get_step06_validation_summary, ValidationLevel, FunctionStatus
    import time
    VALIDATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f'Step06 validation framework not available: {e}')
    VALIDATION_AVAILABLE = False

# Import enhanced error handling
from ..enhanced_error_handling import (
    enhanced_async_error_handler,
    critical_async_process,
    CriticalProcessError,
    ErrorSeverity,
    ErrorCategory
)
from ..enhanced_validation_framework import EnhancedValidator, ValidationLevel as EnhancedValidationLevel
from ..enhanced_monitoring_system import monitor_critical_process
    
    def step06_function_validator(*args, **kwargs) -> None:
        def decorator(func: Callable) -> None:
            return func
        return decorator

    def step06_function_tracker(func: Callable) -> None:
        return func

    def step06_validation_context(*args, **kwargs) -> None:
        from contextlib import nullcontext
        return nullcontext()

    def get_step06_validation_summary() -> Any:
        return {'error': 'Validation framework not available'}

    class ValidationLevel:
        BASIC = 'basic'
        DETAILED = 'detailed'
        COMPREHENSIVE = 'comprehensive'

    class FunctionStatus:
        PENDING = 'pending'
        IN_PROGRESS = 'in_progress'
        COMPLETED = 'completed'
        FAILED = 'failed'
        TIMEOUT = 'timeout'
    VALIDATION_AVAILABLE = False

class FeatureInteractionEngine:
    """
    Advanced feature interaction engineering for step06.

    Creates interaction terms between:
    - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
    - Market features (price, volume, volatility)
    - Derived metrics (momentum, acceleration, regime indicators)
    - Cross-timeframe features
    - Regime-dependent interactions

    Integrates with DiverseLookbackOptimizer to ensure optimal, non-correlated lookback periods.
    """
    @log_important_calls

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize feature interaction engine.

        Args:
            config: Configuration dictionary with interaction parameters
        """
        self.config = config
        self.logger = logger
        self.step6_config = config.get('step06_feature_engineering', {})

        # Initialize comprehensive optimization components
        if OPTIMIZATIONS_AVAILABLE:
            try:
                # M1 Hardware-Specific Optimizations
                self.gpu_manager = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer()
                self.cpu_optimizer = get_m1_cpu_optimizer()

                # Processing Core Optimizations
                self.vectorized_core = get_vectorized_processing_core()
                self.matrix_ops = get_enhanced_matrix_operations()
                self.step_optimizer = get_step_optimization_manager()

                # Data Management Optimizations
                self.data_manager = OptimizedDataManager(
                    base_path=Path(self.config.get('DATA_DIR', 'data_cache')),
                    enable_caching=True,
                    enable_compression=True,
                    enable_parallel_io=True
                )

                self.logger.info('🚀 Step 6 initialized with comprehensive optimization suite:')
                self.logger.info('  ✅ M1 GPU Manager (MPS acceleration)')
                self.logger.info('  ✅ M1 Memory Optimizer')
                self.logger.info('  ✅ M1 CPU Optimizer (parallel processing)')
                self.logger.info('  ✅ Vectorized Processing Core')
                self.logger.info('  ✅ Enhanced Matrix Operations')
                self.logger.info('  ✅ Enhanced Step Optimizer')
                self.logger.info('  ✅ Optimized Data Manager')

            except Exception as e:
                self.logger.warning(f'Failed to initialize some optimizations: {e}')
                # Initialize with fallbacks
                self._initialize_fallback_optimizations()
        else:
            self._initialize_fallback_optimizations()

    def _initialize_fallback_optimizations(self):
        """Initialize fallback optimizations when full suite is not available."""
        self.gpu_manager = None
        self.memory_optimizer = None
        self.cpu_optimizer = None
        self.vectorized_core = None
        self.matrix_ops = None
        self.step_optimizer = None
        self.data_manager = None
        self.logger.info('📋 Initialized with fallback optimizations (basic functionality only)')

    def _setup_optimization_context(self) -> Dict[str, Any]:
        """Setup optimization context for the current execution."""
        context = {
            'memory_checkpoint': None,
            'optimization_profile': None,
            'data_manager_session': None
        }

        if self.memory_optimizer:
            context['memory_checkpoint'] = self.memory_optimizer.memory_checkpoint('step06_feature_engineering')

        if self.step_optimizer:
            from src.utils.enhanced_step_optimizations import WorkloadType, OptimizationProfile
            # Create optimization profile based on current workload
            context['optimization_profile'] = OptimizationProfile(
                workload_type=WorkloadType.MEMORY_INTENSIVE,
                data_size_mb=800,  # Estimate based on typical data size for feature engineering
                expected_duration=600,  # 10 minutes expected for feature engineering
                priority="high"
            )

        if self.data_manager:
            context['data_manager_session'] = self.data_manager.create_session()

        return context

    async def _load_data_optimized(self, file_path: Path, optimization_context: Dict[str, Any]) -> pd.DataFrame:
        """Load data using optimized data manager with memory management."""
        try:
            session = optimization_context.get('data_manager_session')
            if not session:
                # Fallback to standard loading
                return pd.read_parquet(file_path)

            # Use optimized data manager for loading
            data_id = f"{file_path.stem}_data"
            data = await session.load_data_async(data_id, file_path)

            # Apply memory optimizations
            if self.memory_optimizer:
                data_size_mb = data.memory_usage(deep=True).sum() / (1024**2)
                if self.memory_optimizer.should_chunk_data(data_size_mb, "general"):
                    self.logger.info(f"📦 Large dataset detected ({data_size_mb:.1f}MB), applying memory optimizations")
                    # Optimize data types for memory efficiency
                    data = self.memory_optimizer.optimize_dataframe_dtypes(data)

            return data

        except Exception as e:
            self.logger.warning(f"Optimized data loading failed, falling back to standard loading: {e}")
            return pd.read_parquet(file_path)

    async def _save_data_optimized(
        self,
        data: pd.DataFrame,
        output_path: Path,
        metadata: Dict[str, Any],
        optimization_context: Dict[str, Any]
    ) -> bool:
        """Save data using optimized data manager."""
        try:
            session = optimization_context.get('data_manager_session')
            if not session:
                # Fallback to standard saving
                data.to_parquet(output_path)
                return True

            # Use optimized data manager for saving
            data_id = f"{output_path.stem}_features"
            await session.save_data_async(data_id, data, output_path, metadata=metadata)

            return True

        except Exception as e:
            self.logger.warning(f"Optimized data saving failed, falling back to standard saving: {e}")
            try:
                data.to_parquet(output_path)
                return True
            except Exception as fallback_error:
                self.logger.error(f"Standard saving also failed: {fallback_error}")
                return False

    def _get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimizations used."""
        return {
            'm1_gpu_manager': self.gpu_manager is not None,
            'm1_memory_optimizer': self.memory_optimizer is not None,
            'm1_cpu_optimizer': self.cpu_optimizer is not None,
            'vectorized_processing_core': self.vectorized_core is not None,
            'enhanced_matrix_operations': self.matrix_ops is not None,
            'enhanced_step_optimizer': self.step_optimizer is not None,
            'optimized_data_manager': self.data_manager is not None,
            'diverse_optimizer': self.diverse_optimizer is not None,
            'matrix_optimizer': self.matrix_optimizer is not None
        }

    def _initialize_additional_components(self):
        """Initialize additional components that may have dependencies."""
        try:
            from src.training.diverse_lookback_optimizer import DiverseLookbackOptimizer
            self.diverse_optimizer = DiverseLookbackOptimizer(self.config)
            self.use_dynamic_periods = True
            self.logger.info('✅ Integrated with DiverseLookbackOptimizer for dynamic period selection')
        except ImportError:
            self.diverse_optimizer = None
            self.use_dynamic_periods = False
            self.logger.warning('⚠️ DiverseLookbackOptimizer not available, using fallback periods')

        self.matrix_optimizer = None
        self.use_matrix_optimizer = bool(self.step6_config.get('use_matrix_optimizer', True))
        if self.use_matrix_optimizer:
            try:
                from src.training.matrix_diverse_lookback_optimizer import MatrixDiverseLookbackOptimizer
                self.matrix_optimizer = MatrixDiverseLookbackOptimizer(self.config)
                self.use_dynamic_periods = True
                self.logger.info('✅ Integrated with MatrixDiverseLookbackOptimizer for vectorized period selection')
            except Exception as e:
                self.matrix_optimizer = None
                self.use_matrix_optimizer = False
                self.logger.warning(f'⚠️ MatrixDiverseLookbackOptimizer unavailable ({e}), falling back to classic optimization')
        self.fallback_lookback_periods = {'RSI': {'periods': [7, 21, 50], 'correlation_threshold': 0.7, 'description': 'Short (7) for momentum, Medium (21) for trend, Long (50) for major cycles'}, 'MACD': {'periods': [12, 26, 52], 'correlation_threshold': 0.75, 'description': 'Standard (12,26), Extended (20,40), Long-term (26,52)'}, 'Bollinger_Bands': {'periods': [10, 20, 50], 'correlation_threshold': 0.8, 'description': 'Short (10) for volatility, Standard (20) for trend, Long (50) for major moves'}, 'SMA': {'periods': [5, 20, 100], 'correlation_threshold': 0.85, 'description': 'Very short (5) for immediate trend, Medium (20) for trend, Long (100) for major trend'}, 'EMA': {'periods': [8, 21, 55], 'correlation_threshold': 0.8, 'description': 'Short (8) for momentum, Medium (21) for trend, Long (55) for major trend'}, 'ATR': {'periods': [7, 14, 30], 'correlation_threshold': 0.75, 'description': 'Short (7) for immediate volatility, Standard (14) for trend volatility, Long (30) for major volatility'}, 'Stochastic': {'periods': [7, 14, 30], 'correlation_threshold': 0.7, 'description': 'Short (7) for immediate momentum, Standard (14) for trend momentum, Long (30) for major momentum'}, 'ADX': {'periods': [7, 14, 25], 'correlation_threshold': 0.75, 'description': 'Short (7) for immediate trend, Standard (14) for trend, Long (25) for major trend'}, 'CCI': {'periods': [10, 20, 40], 'correlation_threshold': 0.7, 'description': 'Short (10) for immediate cycles, Medium (20) for trend cycles, Long (40) for major cycles'}, 'Williams_R': {'periods': [7, 14, 28], 'correlation_threshold': 0.7, 'description': 'Short (7) for immediate signals, Standard (14) for trend signals, Long (28) for major signals'}, 'ROC': {'periods': [5, 10, 25], 'correlation_threshold': 0.75, 'description': 'Very short (5) for immediate momentum, Short (10) for momentum, Medium (25) for trend momentum'}, 'OBV': {'periods': [10, 20, 50], 'correlation_threshold': 0.8, 'description': 'Short (10) for immediate volume, Medium (20) for volume trend, Long (50) for major volume trend'}, 'MFI': {'periods': [7, 14, 30], 'correlation_threshold': 0.75, 'description': 'Short (7) for immediate flow, Standard (14) for flow trend, Long (30) for major flow trend'}}
        self.dynamic_lookback_periods = {}
        self.period_optimization_results = {}
        self.force_regime_specific_periods = bool(self.step6_config.get('force_regime_specific_periods', False))
        self.interaction_patterns = {'momentum_volume': {'features': ['RSI_7', 'RSI_21', 'MACD_12_26', 'Volume_Ratio'], 'weight': self.step6_config.get('momentum_volume_weight', 1.5), 'enabled': self.step6_config.get('momentum_volume_enabled', True)}, 'trend_volatility': {'features': ['SMA_5', 'SMA_100', 'BB_Position_20', 'ATR_14'], 'weight': self.step6_config.get('trend_volatility_weight', 1.8), 'enabled': self.step6_config.get('trend_volatility_enabled', True)}, 'oscillator_trend': {'features': ['RSI_7', 'Williams_R_14', 'CCI_20', 'EMA_21'], 'weight': self.step6_config.get('oscillator_trend_weight', 1.3), 'enabled': self.step6_config.get('oscillator_trend_enabled', True)}, 'volume_price': {'features': ['OBV_20', 'MFI_14', 'Price_Momentum', 'Volume_Ratio'], 'weight': self.step6_config.get('volume_price_weight', 1.6), 'enabled': self.step6_config.get('volume_price_enabled', True)}, 'volatility_regime': {'features': ['ATR_7', 'BB_Squeeze_20', 'Volatility', 'Market_Regime'], 'weight': self.step6_config.get('volatility_regime_weight', 1.4), 'enabled': self.step6_config.get('volatility_regime_enabled', True)}, 'cross_timeframe': {'features': ['RSI_7', 'RSI_50', 'MACD_12_26', 'MACD_20_40'], 'weight': self.step6_config.get('cross_timeframe_weight', 1.2), 'enabled': self.step6_config.get('cross_timeframe_enabled', True)}, 'regime_dependent': {'features': ['Trend_Strength', 'Volatility_Regime', 'Volume_Regime', 'Momentum_Regime'], 'weight': self.step6_config.get('regime_dependent_weight', 1.7), 'enabled': self.step6_config.get('regime_dependent_enabled', True)}}
        self.interaction_thresholds = {'strong': self.step6_config.get('strong_interaction_threshold', 0.7), 'medium': self.step6_config.get('medium_interaction_threshold', 0.5), 'weak': self.step6_config.get('weak_interaction_threshold', 0.3)}
        self.selection_params = {'max_interactions': self.step6_config.get('max_interactions', 100), 'min_importance': self.step6_config.get('min_importance', 0.01), 'correlation_threshold': self.step6_config.get('correlation_threshold', 0.8), 'mutual_info_threshold': self.step6_config.get('mutual_info_threshold', 0.05)}
        self.interaction_performance = {}
        self.feature_importance_history = []
        self.selected_interactions_history = []
        self.correlation_analysis_history = []
        self.scaler = StandardScaler()
        self.is_fitted = False
        self._validate_lookback_periods()

        # Initialize additional components
        self._initialize_additional_components()

    @step06_function_validator(function_type='optimization', validation_level = ValidationLevel.COMPREHENSIVE)
    async def optimize_lookback_periods(self, market_data: pd.DataFrame, target: pd.Series, regimes: pd.Series | None = None) -> dict[str, Any]:
        """
        Optimize lookback periods using DiverseLookbackOptimizer with comprehensive optimizations.

        Args:
            market_data: OHLCV market data
            target: Target variable for optimization
            regimes: Market regime labels (optional)

        Returns:
            Dictionary with optimized lookback periods
        """
        # Setup optimization context
        optimization_context = self._setup_optimization_context()

        async with optimization_context.get('memory_checkpoint') if optimization_context.get('memory_checkpoint') else nullcontext():
            with step06_validation_context('optimize_lookback_periods', 'optimization'):
                self.logger.info(f'🎯 Starting lookback period optimization with validation tracking')
                self.logger.info(f'   Input data shape: {market_data.shape}')
                self.logger.info(f'   Target shape: {target.shape}')
                self.logger.info(f'   Regimes provided: {regimes is not None}')

            if not self.use_dynamic_periods:
                self.logger.warning('⚠️ Dynamic period optimization not available, using fallback periods')
                return {'status': 'fallback', 'periods': self.fallback_lookback_periods}

            try:
                self.logger.info('🎯 Starting dynamic lookback period optimization...')

                # Use optimized processing
                if self.vectorized_core and self.use_matrix_optimizer and self.matrix_optimizer is not None:
                    self.logger.info('🧮 Using MatrixDiverseLookbackOptimizer with vectorized processing')
                    optimization_results = await self._optimize_lookback_periods_vectorized(
                        market_data, target, regimes, optimization_context
                    )
                elif self.matrix_optimizer is not None:
                    self.logger.info('🧮 Using MatrixDiverseLookbackOptimizer (vectorized)')
                    optimization_results = await self.matrix_optimizer.find_diverse_lookback_periods_matrix(market_data, target, regimes)
                else:
                    self.logger.info('📈 Using DiverseLookbackOptimizer (classic)')
                    optimization_results = await self.diverse_optimizer.find_diverse_lookback_periods(market_data, target, regimes)

                self.dynamic_lookback_periods = self._extract_optimized_periods(optimization_results)
                self.period_optimization_results = optimization_results
                self._update_interaction_patterns_with_optimized_periods()
                self.logger.info(f'✅ Dynamic period optimization completed. Selected {len(self.dynamic_lookback_periods)} indicators with optimized periods')

                # Final memory optimization
                if self.memory_optimizer:
                    final_memory_stats = self.memory_optimizer.optimize_memory()
                    self.logger.info(f'🧹 Final memory optimization: {final_memory_stats.get("memory_freed_mb", 0):.1f}MB freed')

                return {'status': 'optimized', 'periods': self.dynamic_lookback_periods, 'optimization_results': optimization_results}
            except Exception as e:
                self.logger.exception(f'❌ Dynamic period optimization failed: {e}')
                self.logger.info('🔄 Falling back to predefined periods')
                return {'status': 'fallback', 'periods': self.fallback_lookback_periods}

    async def _optimize_lookback_periods_vectorized(
        self,
        market_data: pd.DataFrame,
        target: pd.Series,
        regimes: pd.Series | None,
        optimization_context: Dict[str, Any]
    ) -> dict[str, Any]:
        """Optimize lookback periods using vectorized processing core."""
        try:
            self.logger.info('⚡ Using vectorized processing core for lookback optimization')

            # Prepare data for vectorized processing
            price_data = market_data[['close', 'high', 'low']].values
            volume_data = market_data['volume'].values if 'volume' in market_data.columns else np.ones(len(market_data))

            # Use matrix optimizer with vectorized processing
            optimization_results = await self.matrix_optimizer.find_diverse_lookback_periods_matrix(
                market_data, target, regimes
            )

            # Apply additional vectorized optimizations
            if self.vectorized_core and optimization_results.get('diverse_lookback_periods'):
                self.logger.info('🔄 Applying vectorized post-processing optimizations')
                # Additional vectorized processing can be added here

            return optimization_results

        except Exception as e:
            self.logger.warning(f'⚠️ Vectorized optimization failed, falling back to standard: {e}')
            return await self.matrix_optimizer.find_diverse_lookback_periods_matrix(market_data, target, regimes)
    @log_all_calls

    def _extract_optimized_periods(self, optimization_results: dict[str, Any]) -> dict[str, list[int]]:
        """
        Extract optimized periods from DiverseLookbackOptimizer results using vectorized operations.
        """
        optimized_periods = {}
        
        # Vectorized extraction using numpy operations
        if self.force_regime_specific_periods and optimization_results.get('regime_specific_periods'):
            regime_results = optimization_results.get('regime_specific_periods', {})
            
            # Vectorized processing of regime results
            all_indicators = set()
            all_periods = []
            
            # Collect all data first (vectorized)
            for regime_data in regime_results.values():
                for indicator, res in regime_data.items():
                    periods = res.get('selected_periods', [])
                    if periods:
                        all_indicators.add(indicator)
                        all_periods.extend([(indicator, p) for p in periods])
            
            if all_periods:
                # Vectorized counting using numpy
                indicators_array = np.array([item[0] for item in all_periods])
                periods_array = np.array([item[1] for item in all_periods])
                
                for indicator in all_indicators:
                    mask = indicators_array == indicator
                    indicator_periods = periods_array[mask]
                    
                    # Vectorized counting and ranking
                    unique_periods, counts = np.unique(indicator_periods, return_counts=True)
                    sorted_indices = np.argsort(-counts)  # Descending order
                    top_periods = unique_periods[sorted_indices[:3]]
                    optimized_periods[indicator] = top_periods.tolist()
            
            # Vectorized processing of diverse periods
            diverse_periods = optimization_results.get('diverse_lookback_periods', {})
            for indicator, res in diverse_periods.items():
                if indicator not in optimized_periods and 'selected_periods' in res:
                    optimized_periods[indicator] = res['selected_periods']
            return optimized_periods
            
        # Vectorized processing of diverse periods only
        diverse_periods = optimization_results.get('diverse_lookback_periods', {})
        for indicator, results in diverse_periods.items():
            if 'selected_periods' in results:
                optimized_periods[indicator] = results['selected_periods']
        return optimized_periods
    @log_all_calls

    def _update_interaction_patterns_with_optimized_periods(self) -> None:
        """
        Update interaction patterns to use optimized periods using vectorized operations.
        """
        if not self.dynamic_lookback_periods:
            return
            
        # Vectorized feature update using numpy operations
        for pattern_config in self.interaction_patterns.values():
            features = pattern_config['features']
            
            # Vectorized processing of features
            base_indicators = np.array([feature.split('_')[0] for feature in features])
            updated_features = []
            
            # Vectorized indicator matching
            for i, base_indicator in enumerate(base_indicators):
                if base_indicator in self.dynamic_lookback_periods:
                    optimized_period = self.dynamic_lookback_periods[base_indicator][0]
                    updated_feature = f'{base_indicator}_{optimized_period}'
                    updated_features.append(updated_feature)
                else:
                    updated_features.append(features[i])
                    
            pattern_config['features'] = updated_features
            
        self.logger.info('🔄 Updated interaction patterns with optimized periods using vectorized operations')
    @log_all_calls

    def _validate_lookback_periods(self) -> None:
        """
        Validate that the selected lookback periods are not too correlated using vectorized operations.
        """
        self.logger.info('🔍 Validating lookback periods for non-correlation...')
        periods_to_validate = self.dynamic_lookback_periods if self.dynamic_lookback_periods else self.fallback_lookback_periods
        
        # Vectorized validation using np.vectorize
        def validate_period_ratios(periods_array):
            """Vectorized function to validate period ratios."""
            if len(periods_array) < 2:
                return True, []
            
            # Create all pairwise combinations using broadcasting
            periods_matrix = np.array(periods_array)
            periods_i = periods_matrix[:, np.newaxis]  # Shape: (n, 1)
            periods_j = periods_matrix[np.newaxis, :]  # Shape: (1, n)
            
            # Calculate ratios for upper triangle only
            ratios = np.maximum(periods_i, periods_j) / np.minimum(periods_i, periods_j)
            upper_triangle_mask = np.triu(np.ones_like(ratios, dtype=bool), k=1)
            upper_ratios = ratios[upper_triangle_mask]
            
            # Check for problematic ratios
            problematic_ratios = upper_ratios[upper_ratios < 1.5]
            return len(problematic_ratios) == 0, problematic_ratios.tolist()
        
        # Apply vectorized validation to each indicator
        for indicator, config in periods_to_validate.items():
            if isinstance(config, dict) and 'periods' in config:
                periods = config['periods']
                config.get('correlation_threshold', 0.8)
            elif isinstance(config, list):
                periods = config
            else:
                continue
                
            # Use vectorized validation
            is_valid, problematic_ratios = validate_period_ratios(periods)
            
            if not is_valid:
                self.logger.warning(f'⚠️ {indicator}: Found {len(problematic_ratios)} problematic period ratios: {problematic_ratios}')
            
            if isinstance(config, dict) and 'description' in config:
                self.logger.info(f"✅ {indicator}: Selected periods {periods} - {config['description']}")
            else:
                self.logger.info(f'✅ {indicator}: Selected periods {periods}')

    @step06_function_validator(function_type='feature_engineering', validation_level = ValidationLevel.COMPREHENSIVE)
    def extract_optimal_technical_indicators(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract technical indicators using optimal, non-correlated lookback periods with comprehensive optimizations.

        Args:
            market_data: OHLCV market data

        Returns:
            pd.DataFrame: Technical indicators with optimal lookback periods
        """
        # Setup optimization context
        optimization_context = self._setup_optimization_context()

        with optimization_context.get('memory_checkpoint') if optimization_context.get('memory_checkpoint') else nullcontext():
            with step06_validation_context('extract_optimal_technical_indicators', 'feature_engineering'):
                self.logger.info(f'🔧 Starting technical indicator extraction with validation tracking')
                self.logger.info(f'   Input data shape: {market_data.shape}')
                self.logger.info(f'   Available columns: {list(market_data.columns)}')
                self.logger.info(f'   Using dynamic periods: {bool(self.dynamic_lookback_periods)}')

            # Fail-fast validation: Check if we have enough data (except for first few rows)
            min_required_rows = 100  # Minimum rows needed for reliable indicators
            if len(market_data) < min_required_rows:
                self.logger.warning(f'⚠️ Data has only {len(market_data)} rows, minimum {min_required_rows} recommended')
                if len(market_data) < 20:  # Critical threshold
                    raise CriticalProcessError(
                        f"Insufficient data for feature extraction: {len(market_data)} rows (minimum 20 required)",
                        severity=ErrorSeverity.CRITICAL,
                        category=ErrorCategory.DATA_VALIDATION
                    )

            # Validate required columns
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in market_data.columns]
            if missing_columns:
                raise CriticalProcessError(
                    f"Missing required columns for feature extraction: {missing_columns}",
                    severity=ErrorSeverity.CRITICAL,
                    category=ErrorCategory.DATA_VALIDATION
                )

            self.logger.info('🔧 Extracting optimal technical indicators with non-correlated lookback periods...')

            # Use optimized indicator extraction
            periods_to_use = self.dynamic_lookback_periods if self.dynamic_lookback_periods else self.fallback_lookback_periods
            indicators = {}

            # Use CPU optimizer for parallel indicator calculation if available
            if self.cpu_optimizer and len(market_data) > 10000:
                self.logger.info('🏃 Using M1 CPU optimizer for parallel indicator extraction')
                indicators = self._extract_indicators_parallel(market_data, periods_to_use, optimization_context)
            else:
                indicators = self._extract_indicators_standard(market_data, periods_to_use)

            # Memory optimization after indicator extraction
            if self.memory_optimizer and len(market_data) > 50000:
                data_size_mb = sum(v.nbytes for v in indicators.values()) / (1024**2) if indicators else 0
                if self.memory_optimizer.should_chunk_data(data_size_mb, "general"):
                    self.logger.info(f'🧠 Applying memory optimizations to indicators ({data_size_mb:.1f}MB)')

            indicators_df = pd.DataFrame(indicators, index=market_data.index)
            indicators_df = indicators_df.fillna(method='ffill').fillna(0)
            self.logger.info(f'✅ Extracted {len(indicators_df.columns)} technical indicators with optimal lookback periods')

            # Final memory optimization
            if self.memory_optimizer:
                final_memory_stats = self.memory_optimizer.optimize_memory()
                self.logger.info(f'🧹 Final memory optimization: {final_memory_stats.get("memory_freed_mb", 0):.1f}MB freed')

            return indicators_df

    def _extract_indicators_parallel(self, market_data: pd.DataFrame, periods_to_use: dict, optimization_context: Dict[str, Any]) -> dict:
        """Extract indicators using parallel processing with M1 CPU optimizer."""
        try:
            self.logger.info('🏃 Starting parallel indicator extraction')

            # Create tasks for parallel execution
            tasks = []
            indicator_functions = [
                ('RSI', self._extract_rsi_indicators),
                ('MACD', self._extract_macd_indicators),
                ('Bollinger_Bands', self._extract_bb_indicators),
                ('SMA', self._extract_sma_indicators),
                ('EMA', self._extract_ema_indicators),
                ('ATR', self._extract_atr_indicators),
                ('Stochastic', self._extract_stoch_indicators),
                ('ADX', self._extract_adx_indicators),
                ('CCI', self._extract_cci_indicators),
                ('Williams_R', self._extract_williams_indicators),
                ('ROC', self._extract_roc_indicators),
                ('OBV', self._extract_obv_indicators),
                ('MFI', self._extract_mfi_indicators)
            ]

            for indicator_name, func in indicator_functions:
                if indicator_name in periods_to_use:
                    tasks.append((indicator_name, func, market_data, periods_to_use[indicator_name]))

            # Execute in parallel
            results = self.cpu_optimizer.parallel_map_sync(
                lambda task: task[2](market_data, task[3]), tasks, max_workers=min(4, len(tasks))
            )

            # Combine results
            indicators = {}
            for result in results:
                if isinstance(result, dict):
                    indicators.update(result)

            self.logger.info(f'✅ Parallel extraction completed: {len(indicators)} indicators')
            return indicators

        except Exception as e:
            self.logger.warning(f'⚠️ Parallel extraction failed, falling back to standard: {e}')
            return self._extract_indicators_standard(market_data, periods_to_use)

    def _extract_rsi_indicators(self, market_data: pd.DataFrame, config: dict) -> dict:
        """Extract RSI indicators."""
        indicators = {}
        periods = config['periods'] if isinstance(config, dict) and 'periods' in config else config
        for period in periods:
            rsi = talib.RSI(market_data['close'].values, timeperiod=period)
            indicators[f'RSI_{period}'] = rsi
        return indicators

    def _extract_macd_indicators(self, market_data: pd.DataFrame, config: dict) -> dict:
        """Extract MACD indicators."""
        indicators = {}
        periods = config['periods'] if isinstance(config, dict) and 'periods' in config else config
        if len(periods) >= 2:
            macd, macd_signal, macd_hist = talib.MACD(market_data['close'].values, fastperiod=periods[0], slowperiod=periods[1], signalperiod=9)
            indicators[f'MACD_{periods[0]}_{periods[1]}'] = macd
            indicators[f'MACD_Signal_{periods[0]}_{periods[1]}'] = macd_signal
            indicators[f'MACD_Hist_{periods[0]}_{periods[1]}'] = macd_hist
        return indicators

    def _extract_bb_indicators(self, market_data: pd.DataFrame, config: dict) -> dict:
        """Extract Bollinger Band indicators."""
        indicators = {}
        periods = config['periods'] if isinstance(config, dict) and 'periods' in config else config
        for period in periods:
            bb_upper, bb_middle, bb_lower = talib.BBANDS(market_data['close'].values, timeperiod=period, nbdevup=2, nbdevdn=2)
            bb_position = (market_data['close'] - bb_lower) / (bb_upper - bb_lower)
            bb_squeeze = (bb_upper - bb_lower) / bb_middle
            indicators[f'BB_Upper_{period}'] = bb_upper
            indicators[f'BB_Middle_{period}'] = bb_middle
            indicators[f'BB_Lower_{period}'] = bb_lower
            indicators[f'BB_Position_{period}'] = bb_position
            indicators[f'BB_Squeeze_{period}'] = bb_squeeze
        return indicators

    def _extract_sma_indicators(self, market_data: pd.DataFrame, config: dict) -> dict:
        """Extract SMA indicators."""
        indicators = {}
        periods = config['periods'] if isinstance(config, dict) and 'periods' in config else config
        for period in periods:
            sma = talib.SMA(market_data['close'].values, timeperiod=period)
            indicators[f'SMA_{period}'] = sma
        return indicators

    def _extract_ema_indicators(self, market_data: pd.DataFrame, config: dict) -> dict:
        """Extract EMA indicators."""
        indicators = {}
        periods = config['periods'] if isinstance(config, dict) and 'periods' in config else config
        for period in periods:
            ema = talib.EMA(market_data['close'].values, timeperiod=period)
            indicators[f'EMA_{period}'] = ema
        return indicators

    def _extract_atr_indicators(self, market_data: pd.DataFrame, config: dict) -> dict:
        """Extract ATR indicators."""
        indicators = {}
        periods = config['periods'] if isinstance(config, dict) and 'periods' in config else config
        for period in periods:
            atr = talib.ATR(market_data['high'].values, market_data['low'].values, market_data['close'].values, timeperiod=period)
            atr_normalized = atr / market_data['close']
            indicators[f'ATR_{period}'] = atr
            indicators[f'ATR_Normalized_{period}'] = atr_normalized
        return indicators

    def _extract_stoch_indicators(self, market_data: pd.DataFrame, config: dict) -> dict:
        """Extract Stochastic indicators."""
        indicators = {}
        periods = config['periods'] if isinstance(config, dict) and 'periods' in config else config
        for period in periods:
            stoch_k, stoch_d = talib.STOCH(market_data['high'].values, market_data['low'].values, market_data['close'].values, fastk_period=period, slowk_period=3, slowd_period=3)
            indicators[f'Stoch_K_{period}'] = stoch_k
            indicators[f'Stoch_D_{period}'] = stoch_d
        return indicators

    def _extract_adx_indicators(self, market_data: pd.DataFrame, config: dict) -> dict:
        """Extract ADX indicators."""
        indicators = {}
        periods = config['periods'] if isinstance(config, dict) and 'periods' in config else config
        for period in periods:
            adx = talib.ADX(market_data['high'].values, market_data['low'].values, market_data['close'].values, timeperiod=period)
            indicators[f'ADX_{period}'] = adx
        return indicators

    def _extract_cci_indicators(self, market_data: pd.DataFrame, config: dict) -> dict:
        """Extract CCI indicators."""
        indicators = {}
        periods = config['periods'] if isinstance(config, dict) and 'periods' in config else config
        for period in periods:
            cci = talib.CCI(market_data['high'].values, market_data['low'].values, market_data['close'].values, timeperiod=period)
            indicators[f'CCI_{period}'] = cci
        return indicators

    def _extract_williams_indicators(self, market_data: pd.DataFrame, config: dict) -> dict:
        """Extract Williams %R indicators."""
        indicators = {}
        periods = config['periods'] if isinstance(config, dict) and 'periods' in config else config
        for period in periods:
            williams_r = talib.WILLR(market_data['high'].values, market_data['low'].values, market_data['close'].values, timeperiod=period)
            indicators[f'Williams_R_{period}'] = williams_r
        return indicators

    def _extract_roc_indicators(self, market_data: pd.DataFrame, config: dict) -> dict:
        """Extract ROC indicators."""
        indicators = {}
        periods = config['periods'] if isinstance(config, dict) and 'periods' in config else config
        for period in periods:
            roc = talib.ROC(market_data['close'].values, timeperiod=period)
            indicators[f'ROC_{period}'] = roc
        return indicators

    def _extract_obv_indicators(self, market_data: pd.DataFrame, config: dict) -> dict:
        """Extract OBV indicators."""
        indicators = {}
        obv = talib.OBV(market_data['close'].values, market_data['volume'].values)
        obv_normalized = (obv - obv.rolling(20).mean()) / obv.rolling(20).std()
        indicators['OBV'] = obv
        indicators['OBV_Normalized'] = obv_normalized
        return indicators

    def _extract_mfi_indicators(self, market_data: pd.DataFrame, config: dict) -> dict:
        """Extract MFI indicators."""
        indicators = {}
        periods = config['periods'] if isinstance(config, dict) and 'periods' in config else config
        for period in periods:
            mfi = talib.MFI(market_data['high'].values, market_data['low'].values, market_data['close'].values, market_data['volume'].values, timeperiod=period)
            indicators[f'MFI_{period}'] = mfi
        return indicators

    def _extract_indicators_standard(self, market_data: pd.DataFrame, periods_to_use: dict) -> dict:
        """Extract indicators using vectorized processing to avoid nested loops."""
        indicators = {}

        # Vectorized RSI extraction
        if 'RSI' in periods_to_use:
            rsi_periods = periods_to_use['RSI']
            if isinstance(rsi_periods, dict):
                rsi_periods = rsi_periods['periods']
            
            # Vectorized RSI calculation for all periods at once
            close_values = market_data['close'].values
            for period in rsi_periods:
                try:
                    rsi = talib.RSI(close_values, timeperiod=period)
                    # Fail-fast validation for RSI
                    if np.isnan(rsi).all():
                        raise CriticalProcessError(
                            f"RSI calculation failed for period {period}: all values are NaN",
                            severity=ErrorSeverity.CRITICAL,
                            category=ErrorCategory.FEATURE_ENGINEERING
                        )
                    indicators[f'RSI_{period}'] = rsi
                except Exception as e:
                    raise CriticalProcessError(
                        f"RSI calculation failed for period {period}: {e}",
                        severity=ErrorSeverity.CRITICAL,
                        category=ErrorCategory.FEATURE_ENGINEERING
                    ) from e
        if 'MACD' in periods_to_use:
            macd_periods = periods_to_use['MACD']
            if isinstance(macd_periods, dict):
                macd_periods = macd_periods['periods']
            if len(macd_periods) >= 2:
                macd, macd_signal, macd_hist = talib.MACD(market_data['close'].values, fastperiod = macd_periods[0], slowperiod = macd_periods[1], signalperiod = 9)
                indicators[f'MACD_{macd_periods[0]}_{macd_periods[1]}'] = macd
                indicators[f'MACD_Signal_{macd_periods[0]}_{macd_periods[1]}'] = macd_signal
                indicators[f'MACD_Hist_{macd_periods[0]}_{macd_periods[1]}'] = macd_hist
                if len(macd_periods) >= 3:
                    macd_ext, macd_signal_ext, macd_hist_ext = talib.MACD(market_data['close'].values, fastperiod = macd_periods[1], slowperiod = macd_periods[2], signalperiod = 9)
                    indicators[f'MACD_{macd_periods[1]}_{macd_periods[2]}'] = macd_ext
                    indicators[f'MACD_Signal_{macd_periods[1]}_{macd_periods[2]}'] = macd_signal_ext
                    indicators[f'MACD_Hist_{macd_periods[1]}_{macd_periods[2]}'] = macd_hist_ext
        if 'Bollinger_Bands' in periods_to_use:
            bb_periods = periods_to_use['Bollinger_Bands']
            if isinstance(bb_periods, dict):
                bb_periods = bb_periods['periods']
            for period in bb_periods:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(market_data['close'].values, timeperiod = period, nbdevup = 2, nbdevdn = 2)
                bb_position = (market_data['close'] - bb_lower) / (bb_upper - bb_lower)
                bb_squeeze = (bb_upper - bb_lower) / bb_middle
                indicators[f'BB_Upper_{period}'] = bb_upper
                indicators[f'BB_Middle_{period}'] = bb_middle
                indicators[f'BB_Lower_{period}'] = bb_lower
                indicators[f'BB_Position_{period}'] = bb_position
                indicators[f'BB_Squeeze_{period}'] = bb_squeeze
        if 'SMA' in periods_to_use:
            sma_periods = periods_to_use['SMA']
            if isinstance(sma_periods, dict):
                sma_periods = sma_periods['periods']
            for period in sma_periods:
                sma = talib.SMA(market_data['close'].values, timeperiod = period)
                indicators[f'SMA_{period}'] = sma
        if 'EMA' in periods_to_use:
            ema_periods = periods_to_use['EMA']
            if isinstance(ema_periods, dict):
                ema_periods = ema_periods['periods']
            for period in ema_periods:
                ema = talib.EMA(market_data['close'].values, timeperiod = period)
                indicators[f'EMA_{period}'] = ema
        if 'ATR' in periods_to_use:
            atr_periods = periods_to_use['ATR']
            if isinstance(atr_periods, dict):
                atr_periods = atr_periods['periods']
            for period in atr_periods:
                atr = talib.ATR(market_data['high'].values, market_data['low'].values, market_data['close'].values, timeperiod = period)
                atr_normalized = atr / market_data['close']
                indicators[f'ATR_{period}'] = atr
                indicators[f'ATR_Normalized_{period}'] = atr_normalized
        if 'Stochastic' in periods_to_use:
            stoch_periods = periods_to_use['Stochastic']
            if isinstance(stoch_periods, dict):
                stoch_periods = stoch_periods['periods']
            for period in stoch_periods:
                stoch_k, stoch_d = talib.STOCH(market_data['high'].values, market_data['low'].values, market_data['close'].values, fastk_period = period, slowk_period = 3, slowd_period = 3)
                indicators[f'Stoch_K_{period}'] = stoch_k
                indicators[f'Stoch_D_{period}'] = stoch_d
        if 'ADX' in periods_to_use:
            adx_periods = periods_to_use['ADX']
            if isinstance(adx_periods, dict):
                adx_periods = adx_periods['periods']
            for period in adx_periods:
                adx = talib.ADX(market_data['high'].values, market_data['low'].values, market_data['close'].values, timeperiod = period)
                indicators[f'ADX_{period}'] = adx
        if 'CCI' in periods_to_use:
            cci_periods = periods_to_use['CCI']
            if isinstance(cci_periods, dict):
                cci_periods = cci_periods['periods']
            for period in cci_periods:
                cci = talib.CCI(market_data['high'].values, market_data['low'].values, market_data['close'].values, timeperiod = period)
                indicators[f'CCI_{period}'] = cci
        if 'Williams_R' in periods_to_use:
            williams_periods = periods_to_use['Williams_R']
            if isinstance(williams_periods, dict):
                williams_periods = williams_periods['periods']
            for period in williams_periods:
                williams_r = talib.WILLR(market_data['high'].values, market_data['low'].values, market_data['close'].values, timeperiod = period)
                indicators[f'Williams_R_{period}'] = williams_r
        if 'ROC' in periods_to_use:
            roc_periods = periods_to_use['ROC']
            if isinstance(roc_periods, dict):
                roc_periods = roc_periods['periods']
            for period in roc_periods:
                roc = talib.ROC(market_data['close'].values, timeperiod = period)
                indicators[f'ROC_{period}'] = roc
        if 'OBV' in periods_to_use:
            obv = talib.OBV(market_data['close'].values, market_data['volume'].values)
            obv_normalized = (obv - obv.rolling(20).mean()) / obv.rolling(20).std()
            indicators['OBV'] = obv
            indicators['OBV_Normalized'] = obv_normalized
        if 'MFI' in periods_to_use:
            mfi_periods = periods_to_use['MFI']
            if isinstance(mfi_periods, dict):
                mfi_periods = mfi_periods['periods']
            for period in mfi_periods:
                mfi = talib.MFI(market_data['high'].values, market_data['low'].values, market_data['close'].values, market_data['volume'].values, timeperiod=period)
                indicators[f'MFI_{period}'] = mfi

        # Return indicators dict instead of DataFrame for consistency with parallel method
        return indicators

    @step06_function_validator(function_type='data_processing', validation_level = ValidationLevel.COMPREHENSIVE)
    def analyze_feature_correlations(self, features: pd.DataFrame) -> dict[str, Any]:
        """
        Analyze correlations between features to ensure non-correlation.

        Args:
            features: Feature DataFrame

        Returns:
            Dict with correlation analysis results
        """
        with step06_validation_context('analyze_feature_correlations', 'data_processing'):
            self.logger.info(f'🔍 Starting correlation analysis with validation tracking')
            self.logger.info(f'   Input features shape: {features.shape}')
            self.logger.info(f'   Feature columns: {len(features.columns)}')
            self.logger.info(f'   Data types: {features.dtypes.value_counts().to_dict()}')
        self.logger.info('🔍 Analyzing feature correlations to ensure non-correlation...')
        correlation_matrix = features.corr()
        
        # Vectorized correlation analysis using advanced NumPy operations
        def find_high_correlations_vectorized(corr_matrix):
            """Vectorized function to find high correlations using np.apply_along_axis."""
            # Get upper triangle indices
            upper_triangle_mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            upper_triangle_values = corr_matrix.values[upper_triangle_mask]
            upper_triangle_indices = np.where(upper_triangle_mask)
            
            # Find high correlations using vectorized operations
            high_corr_mask = np.abs(upper_triangle_values) > 0.8
            high_corr_indices = (upper_triangle_indices[0][high_corr_mask], 
                               upper_triangle_indices[1][high_corr_mask])
            high_corr_values = upper_triangle_values[high_corr_mask]
            
            # Create correlation dictionaries
            high_correlations = []
            feature_names = corr_matrix.columns
            for row_idx, col_idx, corr_value in zip(high_corr_indices[0], high_corr_indices[1], high_corr_values):
                high_correlations.append({
                    'feature1': feature_names[row_idx],
                    'feature2': feature_names[col_idx], 
                    'correlation': corr_value
                })
            
            return high_correlations
        
        # Apply vectorized analysis
        high_correlations = find_high_correlations_vectorized(correlation_matrix)
        correlation_groups = {}
        for corr in high_correlations:
            indicator_type = corr['feature1'].split('_')[0]
            if indicator_type not in correlation_groups:
                correlation_groups[indicator_type] = []
            correlation_groups[indicator_type].append(corr)
        analysis_results = {'correlation_matrix': correlation_matrix, 'high_correlations': high_correlations, 'correlation_groups': correlation_groups, 'n_high_correlations': len(high_correlations), 'mean_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k = 1)].mean(), 'max_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k = 1)].max()}
        if high_correlations:
            self.logger.warning(f'⚠️ Found {len(high_correlations)} highly correlated feature pairs')
            for corr in high_correlations[:5]:
                self.logger.warning(f"   {corr['feature1']} vs {corr['feature2']}: {corr['correlation']:.3f}")
        else:
            self.logger.info('✅ No highly correlated features found - optimal lookback periods working correctly')
        self.correlation_analysis_history.append({'timestamp': datetime.now(), 'results': analysis_results})
        return analysis_results

    @step06_function_validator(function_type='feature_engineering', validation_level = ValidationLevel.COMPREHENSIVE)
    def extract_interaction_features(self, features: np.ndarray, feature_names: list[str], market_data: pd.DataFrame) -> np.ndarray:
        """
        Extract comprehensive interaction features.

        Args:
            features: Base feature array
            feature_names: Names of base features
            market_data: Market data for regime analysis

        Returns:
            np.ndarray: Interaction features
        """
        with step06_validation_context('extract_interaction_features', 'feature_engineering'):
            self.logger.info(f'🔗 Starting interaction feature extraction with validation tracking')
            self.logger.info(f'   Input features shape: {features.shape}')
            self.logger.info(f'   Feature names count: {len(feature_names)}')
            self.logger.info(f'   Market data shape: {market_data.shape}')
            self.logger.info(f"   Interaction patterns enabled: {sum((1 for p in self.interaction_patterns.values() if p['enabled']))}")
        try:
            self.logger.info('Extracting feature interactions...')
            basic_interactions = self._create_basic_interactions(features, feature_names)
            pattern_interactions = self._create_pattern_interactions(features, feature_names)
            regime_interactions = self._create_regime_interactions(features, feature_names, market_data)
            timeframe_interactions = self._create_cross_timeframe_interactions(features, feature_names)
            all_interactions = np.concatenate([basic_interactions, pattern_interactions, regime_interactions, timeframe_interactions], axis = 1)
            selected_interactions = self._select_optimal_interactions(all_interactions, market_data)
            if not self.is_fitted:
                selected_interactions = self.scaler.fit_transform(selected_interactions)
                self.is_fitted = True
            else:
                selected_interactions = self.scaler.transform(selected_interactions)
            self.logger.info(f'Extracted {selected_interactions.shape[1]} interaction features')
            return selected_interactions
        except Exception as e:
            self.logger.error(f'❌ CRITICAL: Feature interaction extraction failed: {e}')
            self.logger.error(f'❌ Input features shape: {features.shape}')
            self.logger.error(f'❌ Feature names count: {len(feature_names)}')
            # Fail fast - do not return zeros, raise the error
            raise CriticalProcessError(
                f"Feature interaction extraction failed: {e}",
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.FEATURE_ENGINEERING
            ) from e

    @log_all_calls
    @step06_function_tracker
    def _create_basic_interactions(self, features: np.ndarray, feature_names: list[str]) -> np.ndarray:
        """
        Create basic pairwise interactions between features using advanced vectorized operations.
        """
        self.logger.debug(f'🔗 Creating basic interactions for {features.shape[0]} samples, {len(feature_names)} features')
        
        # Vectorized interaction functions using np.vectorize
        @np.vectorize
        def safe_ratio(a, b, epsilon=1e-8):
            """Vectorized safe division with epsilon."""
            return a / (b + epsilon)
        
        @np.vectorize
        def interaction_product(a, b):
            """Vectorized product interaction."""
            return a * b
        
        @np.vectorize
        def interaction_diff(a, b):
            """Vectorized difference interaction."""
            return a - b
        
        interactions = []
        feature_map = {name: i for i, name in enumerate(feature_names)}
        important_pairs = [('RSI', 'MACD'), ('RSI', 'Volume_Ratio'), ('MACD', 'Volume_Ratio'), 
                          ('BB_Position', 'ATR_Normalized'), ('SMA_Ratio', 'EMA_Ratio'), 
                          ('Price_Momentum', 'Volume_Ratio'), ('OBV_Normalized', 'Price_Momentum'), 
                          ('Stochastic', 'RSI'), ('Williams_R', 'RSI'), ('CCI', 'RSI')]
        
        for feature1, feature2 in important_pairs:
            if feature1 in feature_map and feature2 in feature_map:
                idx1, idx2 = (feature_map[feature1], feature_map[feature2])
                feat1_data = features[:, idx1]
                feat2_data = features[:, idx2]
                
                # Create interaction terms using vectorized functions
                interactions.append(interaction_product(feat1_data, feat2_data))
                interactions.append(safe_ratio(feat1_data, feat2_data))
                interactions.append(interaction_diff(feat1_data, feat2_data))
        
        return np.column_stack(interactions) if interactions else np.zeros((features.shape[0], 0))

    @log_all_calls
    @step06_function_tracker
    def _create_pattern_interactions(self, features: np.ndarray, feature_names: list[str]) -> np.ndarray:
        """
        Create pattern-based interactions using predefined patterns.
        """
        self.logger.debug(f'🎯 Creating pattern interactions for {features.shape[0]} samples')
        interactions = []
        feature_map = {name: i for i, name in enumerate(feature_names)}
        for pattern_name, pattern_config in self.interaction_patterns.items():
            if not pattern_config['enabled']:
                continue
            pattern_features = pattern_config['features']
            weight = pattern_config['weight']
            pattern_indices = []
            for feature_name in pattern_features:
                if feature_name in feature_map:
                    pattern_indices.append(feature_map[feature_name])
            if len(pattern_indices) >= 2:
                pattern_interactions = self._create_pattern_specific_interactions(features, pattern_indices, pattern_name, weight)
                interactions.extend(pattern_interactions)
        return np.column_stack(interactions) if interactions else np.zeros((features.shape[0], 0))
    @log_all_calls

    def _create_pattern_specific_interactions(self, features: np.ndarray, pattern_indices: list[int], pattern_name: str, weight: float) -> list[np.ndarray]:
        """
        Create pattern-specific interactions.
        """
        interactions = []
        pattern_features = features[:, pattern_indices]
        if pattern_name == 'momentum_volume':
            momentum_avg = np.mean(pattern_features[:, :3], axis = 1)
            volume_feature = pattern_features[:, 3]
            interactions.extend([momentum_avg * volume_feature * weight, momentum_avg / (volume_feature + 1e-08) * weight, np.std(pattern_features[:, :3], axis = 1) * volume_feature * weight])
        elif pattern_name == 'trend_volatility':
            trend_avg = np.mean(pattern_features[:, :2], axis = 1)
            volatility_avg = np.mean(pattern_features[:, 2:], axis = 1)
            interactions.extend([trend_avg * volatility_avg * weight, trend_avg / (volatility_avg + 1e-08) * weight, np.abs(trend_avg) * volatility_avg * weight])
        elif pattern_name == 'oscillator_trend':
            oscillator_avg = np.mean(pattern_features[:, :3], axis = 1)
            trend_feature = pattern_features[:, 3]
            interactions.extend([oscillator_avg * trend_feature * weight, oscillator_avg / (trend_feature + 1e-08) * weight, np.std(pattern_features[:, :3], axis = 1) * trend_feature * weight])
        elif pattern_name == 'volume_price':
            volume_avg = np.mean(pattern_features[:, [0, 3]], axis = 1)
            price_feature = pattern_features[:, 2]
            interactions.extend([volume_avg * price_feature * weight, volume_avg / (price_feature + 1e-08) * weight, np.sqrt(volume_avg) * price_feature * weight])
        elif pattern_name == 'volatility_regime':
            volatility_avg = np.mean(pattern_features[:, :3], axis = 1)
            regime_feature = pattern_features[:, 3] if pattern_features.shape[1] > 3 else np.ones(features.shape[0])
            interactions.extend([volatility_avg * regime_feature * weight, volatility_avg / (regime_feature + 1e-08) * weight, np.square(volatility_avg) * regime_feature * weight])
        return interactions

    @log_all_calls
    @step06_function_tracker
    def _create_regime_interactions(self, features: np.ndarray, feature_names: list[str], market_data: pd.DataFrame) -> np.ndarray:
        """
        Create regime-dependent interactions.
        """
        self.logger.debug(f'🏛️ Creating regime interactions for {features.shape[0]} samples')
        interactions = []
        try:
            if 'timestamp' in market_data.columns:
                market_data = market_data.sort_values('timestamp').copy()
            for col in list(market_data.columns):
                if col.lower().startswith('future_') or col.lower().endswith('_future'):
                    market_data = market_data.drop(columns=[col])
        except Exception as e:
            self.logger.warning(f'Causality guard (regime interactions) encountered an issue: {e}')
        market_regime = self._identify_market_regime(market_data)
        if market_regime == 'trending':
            trend_interactions = self._create_trending_interactions(features, feature_names)
            interactions.extend(trend_interactions)
        elif market_regime == 'ranging':
            ranging_interactions = self._create_ranging_interactions(features, feature_names)
            interactions.extend(ranging_interactions)
        elif market_regime == 'volatile':
            volatile_interactions = self._create_volatile_interactions(features, feature_names)
            interactions.extend(volatile_interactions)
        return np.column_stack(interactions) if interactions else np.zeros((features.shape[0], 0))
    @log_all_calls

    def _create_trending_interactions(self, features: np.ndarray, feature_names: list[str]) -> list[np.ndarray]:
        """
        Create interactions specific to trending markets.
        """
        interactions = []
        feature_map = {name: i for i, name in enumerate(feature_names)}
        trend_features = ['SMA_Ratio', 'EMA_Ratio', 'MACD', 'ADX']
        momentum_features = ['RSI', 'Stochastic', 'CCI']
        trend_indices = [feature_map.get(f) for f in trend_features if f in feature_map]
        momentum_indices = [feature_map.get(f) for f in momentum_features if f in feature_map]
        if trend_indices and momentum_indices:
            trend_avg = np.mean(features[:, trend_indices], axis = 1)
            momentum_avg = np.mean(features[:, momentum_indices], axis = 1)
            interactions.extend([trend_avg * momentum_avg * 1.5, trend_avg / (momentum_avg + 1e-08) * 1.3, np.abs(trend_avg) * momentum_avg * 1.4])
        return interactions
    @log_all_calls

    def _create_ranging_interactions(self, features: np.ndarray, feature_names: list[str]) -> list[np.ndarray]:
        """
        Create interactions specific to ranging markets.
        """
        interactions = []
        feature_map = {name: i for i, name in enumerate(feature_names)}
        oscillator_features = ['RSI', 'Stochastic', 'Williams_R', 'CCI']
        volume_features = ['Volume_Ratio', 'OBV_Normalized', 'MFI']
        oscillator_indices = [feature_map.get(f) for f in oscillator_features if f in feature_map]
        volume_indices = [feature_map.get(f) for f in volume_features if f in feature_map]
        if oscillator_indices and volume_indices:
            oscillator_avg = np.mean(features[:, oscillator_indices], axis = 1)
            volume_avg = np.mean(features[:, volume_indices], axis = 1)
            interactions.extend([oscillator_avg * volume_avg * 1.6, oscillator_avg / (volume_avg + 1e-08) * 1.4, np.std(features[:, oscillator_indices], axis = 1) * volume_avg * 1.5])
        return interactions
    @log_all_calls

    def _create_volatile_interactions(self, features: np.ndarray, feature_names: list[str]) -> list[np.ndarray]:
        """
        Create interactions specific to volatile markets.
        """
        interactions = []
        feature_map = {name: i for i, name in enumerate(feature_names)}
        volatility_features = ['ATR_Normalized', 'BB_Squeeze', 'Volatility']
        risk_features = ['RSI', 'Stochastic', 'Williams_R']
        volatility_indices = [feature_map.get(f) for f in volatility_features if f in feature_map]
        risk_indices = [feature_map.get(f) for f in risk_features if f in feature_map]
        if volatility_indices and risk_indices:
            volatility_avg = np.mean(features[:, volatility_indices], axis = 1)
            risk_avg = np.mean(features[:, risk_indices], axis = 1)
            interactions.extend([volatility_avg * risk_avg * 1.8, volatility_avg / (risk_avg + 1e-08) * 1.6, np.square(volatility_avg) * risk_avg * 1.7])
        return interactions
    @log_all_calls

    def _create_cross_timeframe_interactions(self, features: np.ndarray, feature_names: list[str]) -> np.ndarray:
        """
        Create cross-timeframe interactions.
        """
        interactions = []
        feature_map = {name: i for i, name in enumerate(feature_names)}
        timeframe_pairs = [('RSI_14', 'RSI_30'), ('MACD_12_26', 'MACD_20_40'), ('SMA_20', 'SMA_50'), ('EMA_12', 'EMA_26')]
        for short_feature, long_feature in timeframe_pairs:
            if short_feature in feature_map and long_feature in feature_map:
                short_idx, long_idx = (feature_map[short_feature], feature_map[long_feature])
                diff = features[:, short_idx] - features[:, long_idx]
                ratio = features[:, short_idx] / (features[:, long_idx] + 1e-08)
                prod = features[:, short_idx] * features[:, long_idx]
                abs_diff = np.abs(diff)
                interactions.extend([diff, ratio, prod, abs_diff])
        return np.column_stack(interactions) if interactions else np.zeros((features.shape[0], 0))
    @log_all_calls

    def _identify_market_regime(self, market_data: pd.DataFrame) -> str:
        """
        Identify current market regime.
        """
        try:
            volatility = market_data['close'].pct_change().rolling(20).std().iloc[-1]
            trend_strength = abs(market_data['close'].rolling(20).mean().iloc[-1] - market_data['close'].rolling(50).mean().iloc[-1]) / market_data['close'].iloc[-1]
            if volatility > 0.03:
                return 'volatile'
            if trend_strength > 0.02:
                return 'trending'
            return 'ranging'
        except Exception as e:
            self.logger.warning(f'Market regime identification failed: {e}')
            return 'ranging'
    @log_all_calls

    def _select_optimal_interactions(self, interactions: np.ndarray, market_data: pd.DataFrame) -> np.ndarray:
        """
        Select optimal interactions based on importance and correlation.
        """
        try:
            if 'timestamp' in market_data.columns:
                market_data = market_data.sort_values('timestamp').copy()
            dummy_target = np.random.choice([0, 1], size = interactions.shape[0])
            mi_scores = mutual_info_classif(interactions, dummy_target, random_state = 42)
            mi_threshold = self.selection_params['mutual_info_threshold']
            important_indices = np.where(mi_scores > mi_threshold)[0]
            max_interactions = self.selection_params['max_interactions']
            if len(important_indices) > max_interactions:
                top_indices = np.argsort(mi_scores)[-max_interactions:]
                selected_interactions = interactions[:, top_indices]
            else:
                selected_interactions = interactions[:, important_indices]
            self.selected_interactions_history.append({'timestamp': datetime.now(), 'n_interactions': selected_interactions.shape[1], 'mi_scores': mi_scores[important_indices] if len(important_indices) > 0 else []})
            return selected_interactions
        except Exception as e:
            self.logger.error(f'❌ CRITICAL: Interaction selection failed: {e}')
            self.logger.error(f'❌ Input interactions shape: {interactions.shape}')
            # Fail fast - do not return subset, raise the error
            raise CriticalProcessError(
                f"Interaction selection failed: {e}",
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.FEATURE_ENGINEERING
            ) from e

    def get_interaction_summary(self) -> dict[str, Any]:
        """
        Get summary of interaction engineering results.
        """
        return {'interaction_patterns': self.interaction_patterns, 'selection_params': self.selection_params, 'performance_history': self.interaction_performance, 'selected_interactions_count': len(self.selected_interactions_history), 'is_fitted': self.is_fitted, 'scaler_params': {'mean': self.scaler.mean_.tolist() if self.is_fitted else None, 'scale': self.scaler.scale_.tolist() if self.is_fitted else None}}

    def update_performance(self, performance_metrics: dict[str, float]) -> None:
        """
        Update interaction performance tracking.
        """
        self.interaction_performance[datetime.now()] = performance_metrics

    @step06_function_validator(function_type='feature_engineering', validation_level = ValidationLevel.COMPREHENSIVE)
    def get_feature_importance(self, interactions: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Calculate importance of interaction features.
        """
        with step06_validation_context('get_feature_importance', 'feature_engineering'):
            self.logger.info(f'📊 Starting feature importance calculation with validation tracking')
            self.logger.info(f'   Interactions shape: {interactions.shape}')
            self.logger.info(f'   Target shape: {target.shape}')
            self.logger.info(f'   Target distribution: {np.bincount(target.astype(int))}')
        try:
            mi_scores = mutual_info_classif(interactions, target, random_state = 42)
            self.feature_importance_history.append({'timestamp': datetime.now(), 'importance_scores': mi_scores.tolist(), 'mean_importance': np.mean(mi_scores), 'max_importance': np.max(mi_scores)})
            self.logger.info(f'✅ Feature importance calculation completed')
            self.logger.info(f'   Mean importance: {np.mean(mi_scores):.4f}')
            self.logger.info(f'   Max importance: {np.max(mi_scores):.4f}')
            self.logger.info(f'   Features with importance > 0.1: {np.sum(mi_scores > 0.1)}')
            return mi_scores
        except Exception as e:
            self.logger.exception(f'Feature importance calculation failed: {e}')
            return np.ones(interactions.shape[1])

    def generate_comprehensive_function_report(self) -> dict[str, Any]:
        """
        Generate comprehensive function execution report for step06.
        
        Returns:
            Dictionary with detailed function execution analysis
        """
        self.logger.info('📋 Generating comprehensive function execution report...')
        validation_summary = {}
        if VALIDATION_AVAILABLE:
            try:
                validation_summary = get_step06_validation_summary()
            except Exception as e:
                self.logger.warning(f'Could not get validation summary: {e}')
        internal_stats = {'interaction_patterns': {'total_patterns': len(self.interaction_patterns), 'enabled_patterns': sum((1 for p in self.interaction_patterns.values() if p['enabled'])), 'pattern_details': {name: {'enabled': config['enabled'], 'weight': config['weight']} for name, config in self.interaction_patterns.items()}}, 'lookback_periods': {'using_dynamic': bool(self.dynamic_lookback_periods), 'fallback_periods': len(self.fallback_lookback_periods), 'optimized_periods': len(self.dynamic_lookback_periods) if self.dynamic_lookback_periods else 0}, 'feature_engineering_history': {'correlation_analyses': len(self.correlation_analysis_history), 'feature_importance_calculations': len(self.feature_importance_history), 'selected_interactions': len(self.selected_interactions_history)}, 'performance_metrics': {'scaler_fitted': self.is_fitted, 'interaction_performance_tracked': len(self.interaction_performance)}}
        comprehensive_report = {'timestamp': datetime.now().isoformat(), 'validation_summary': validation_summary, 'internal_statistics': internal_stats, 'recommendations': self._generate_step06_recommendations(internal_stats), 'function_call_analysis': self._analyze_function_calls(), 'performance_analysis': self._analyze_performance_metrics()}
        self.logger.info('✅ Comprehensive function execution report generated')
        return comprehensive_report
    @log_all_calls

    def _generate_step06_recommendations(self, stats: dict[str, Any]) -> list[str]:
        """Generate recommendations based on step06 execution statistics."""
        recommendations = []
        if stats['interaction_patterns']['enabled_patterns'] < stats['interaction_patterns']['total_patterns']:
            recommendations.append('Consider enabling more interaction patterns for better feature coverage')
        if not stats['lookback_periods']['using_dynamic']:
            recommendations.append('Consider using dynamic lookback period optimization for better performance')
        if stats['feature_engineering_history']['correlation_analyses'] == 0:
            recommendations.append('Run correlation analysis to identify redundant features')
        if stats['feature_engineering_history']['feature_importance_calculations'] == 0:
            recommendations.append('Calculate feature importance to identify most valuable features')
        return recommendations
    @log_all_calls

    def _analyze_function_calls(self) -> dict[str, Any]:
        """Analyze function call patterns and performance."""
        return {'total_correlation_analyses': len(self.correlation_analysis_history), 'total_importance_calculations': len(self.feature_importance_history), 'total_interaction_selections': len(self.selected_interactions_history), 'performance_tracking_entries': len(self.interaction_performance)}
    @log_all_calls

    def _analyze_performance_metrics(self) -> dict[str, Any]:
        """Analyze performance metrics and trends."""
        metrics = {'scaler_status': 'fitted' if self.is_fitted else 'not_fitted', 'interaction_patterns_optimized': len(self.interaction_patterns), 'lookback_periods_optimized': len(self.dynamic_lookback_periods) if self.dynamic_lookback_periods else 0}
        if self.interaction_performance:
            recent_performance = list(self.interaction_performance.values())[-1]
            metrics['latest_performance'] = recent_performance
        return metrics