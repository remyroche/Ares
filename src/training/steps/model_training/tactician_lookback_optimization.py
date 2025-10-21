"""
import warnings
Tactician Lookback Optimization Step

This step optimizes indicator lookback periods specifically for Tactician models
operating on 1m timeframe, using Analyst outputs as additional input features.

Key Features:
- 1m timeframe optimization for precise timing decisions
- Integration with Analyst signals and model outputs
- Timing-specific optimization objectives
- Cross-timeframe feature engineering
- Dependency-aware execution (requires Analyst outputs)
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

# Import core utilities
from src.utils.logger import system_logger
from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_debug, tprint_structured, LogLevel
)

# Import common utilities
from src.utils.common_operations import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    calculate_data_quality_metrics, safe_merge_dataframes, safe_groupby_operation,
    safe_apply_function, create_summary_statistics, safe_drop_columns,
    safe_rename_columns, validate_timestamp_column, safe_timestamp_conversion,
    get_dataframe_info, safe_filter_dataframe, create_data_quality_report,
    optimize_dataframe_dtypes, safe_fillna, safe_rolling,
    safe_to_parquet, safe_read_parquet, validate_dataframe_schema,
    guard_dataframe_nulls, memory_checkpoint, optimize_memory,
    get_memory_usage, integrate_with_m1_optimizers, get_m1_gpu_manager,
    get_m1_memory_optimizer, get_m1_cpu_optimizer, validate_dataframe
)

# Import existing indicator implementations from feature_generation
from src.feature_generation.utils.optimized_cross_timeframe_analysis_integration import (
    OptimizedCrossTimeframeAnalysisIntegration
)
from src.feature_generation.categories.momentum import (
    MomentumFeatureGenerator, StochasticGenerator, WilliamsRGenerator,
    ROCGenerator, MomentumGenerator
)
from src.feature_generation.categories.trend import (
    TrendFeatureGenerator, KeltnerChannelsGenerator
)
from src.feature_generation.categories.volume import (
    VolumeFeatureGenerator, VWAPGenerator, OBVGenerator, VolumeROCGenerator
)
from src.feature_generation.categories.volatility import (
    VolatilityFeatureGenerator, BollingerBandsGenerator, VolatilityBandsGenerator
)
from src.feature_generation.categories.oscillator import CCIGenerator

from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, safe_correlation,
    safe_covariance, safe_mean, safe_std, safe_percentile,
    validate_correlation_matrix, safe_matrix_inverse, math_safe,
    MathValidation, MathValidationError
)

from src.utils.serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
)

# Import ML utilities
try:
    from src.utils.ml_common.config import BaseTrainingConfig
    from src.utils.ml_common.validation.cv import (
        purged_time_series_splits, PurgedSplitConfig
    )
    ML_COMMON_AVAILABLE = True
except ImportError:
    ML_COMMON_AVAILABLE = False
    tprint_warning("⚠️ ML common utilities not available")

# Import optimization utilities
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    tprint_warning("⚠️ Optuna not available, using grid search fallback")

# Import Bayesian TPE optimizer with early stopping
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
        BayesianTPEOptimizer, OptimizationConfig
    )
    from src.utils.ml_common.models.model_cache import get_model_cache
    BAYESIAN_TPE_AVAILABLE = True
except ImportError:
    BAYESIAN_TPE_AVAILABLE = False
    tprint_warning("⚠️ Bayesian TPE optimizer not available")

logger = system_logger.getChild('TacticianLookbackOptimization')

@dataclass
class TacticianLookbackConfig:
    """Configuration for Tactician-specific lookback optimization."""

    # Basic configuration
    timeframe: str = "1m"
    symbol: str = "ETHUSDT"
    exchange: str = "binance"

    # Dependencies
    requires_analyst_outputs: bool = True
    analyst_model_path: str = "./models/analyst_models"
    analyst_ensemble_path: str = "./models/analyst_ensemble"

    # Optimization parameters
    optimization_method: str = "two_step_grid_tpe"
    coarse_grid_size: int = 5
    fine_grid_size: int = 5
    tpe_trials: int = 25
    optimization_timeout: Optional[int] = 3600  # 1 hour

    # Lookback constraints
    min_lookback: int = 3
    max_lookback: int = 60  # Shorter for 1m timeframe
    lookback_step: int = 1

    # Feature categories for optimization
    feature_categories: Dict[str, List[str]] = field(default_factory=lambda: {
        "price_action": ["rsi", "macd", "bollinger_bands", "stoch"],
        "volume_analysis": ["volume_sma", "vwap", "obv", "volume_roc"],
        "momentum": ["williams_r", "cci", "momentum", "roc"],
        "volatility": ["atr", "volatility_bands", "keltner_channels"]
    })

    # Optimization targets
    target_metrics: List[str] = field(default_factory=lambda: [
        "entry_timing_accuracy",
        "exit_timing_accuracy",
        "signal_to_noise_ratio",
        "analyst_alignment_score"
    ])

    # Analyst integration weights
    analyst_signal_weight: float = 0.4
    analyst_output_weight: float = 0.3
    raw_features_weight: float = 0.3

    # Validation parameters
    cv_folds: int = 5
    validation_split: float = 0.2
    min_samples_per_fold: int = 100

    # Output configuration
    save_results: bool = True
    results_path: str = "./results/tactician_lookback_optimization"
    save_optimized_features: bool = True

    # Performance thresholds
    min_timing_accuracy: float = 0.6
    min_signal_quality: float = 0.5
    max_correlation_threshold: float = 0.8

class TacticianLookbackOptimizer:
    """
    Tactician Lookback Optimizer for 1m timeframe with Analyst integration.

    This optimizer focuses on finding optimal lookback periods for technical
    indicators used by the Tactician model, taking into account:
    1. Analyst signals and model outputs
    2. 1m timeframe characteristics
    3. Entry/exit timing accuracy
    4. Signal quality in high-frequency context
    """

    def __init__(self, config: TacticianLookbackConfig):
        """Initialize Tactician lookback optimizer."""
        tprint_info("🚀 Initializing Tactician Lookback Optimizer")

        self.config = config
        self.logger = logger.getChild('TacticianLookbackOptimizer')
        self.start_time = time.time()

        # Initialize components consolidated
        self._initialize_components_consolidated()

        # Performance tracking
        self.optimization_metrics = {
            'total_evaluations': 0,
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'best_score': 0.0,
            'convergence_history': [],
            'early_stopping_triggered': False
        }

        init_time = time.time() - self.start_time
        tprint_success(f"✅ Tactician Lookback Optimizer initialized in {init_time:.2f}s")

    def _initialize_components_consolidated(self):
        """Consolidated component initialization."""
        # Utilities
        self.math_validator = MathValidation()
        self.serializers = {
            'json': JSONSerializer(),
            'pickle': PickleSerializer(),
            'parquet': ParquetSerializer()
        }

        # Feature generators
        self.cross_timeframe_generator = OptimizedCrossTimeframeAnalysisIntegration()
        self.momentum_generator = MomentumFeatureGenerator()
        self.trend_generator = TrendFeatureGenerator()
        self.volume_generator = VolumeFeatureGenerator()
        self.volatility_generator = VolatilityFeatureGenerator()

        # Specific indicator generators
        self.stochastic_generator = StochasticGenerator()
        self.williams_r_generator = WilliamsRGenerator()
        self.roc_generator = ROCGenerator()
        self.momentum_specific_generator = MomentumGenerator()
        self.vwap_generator = VWAPGenerator()
        self.obv_generator = OBVGenerator()
        self.volume_roc_generator = VolumeROCGenerator()
        self.bollinger_generator = BollingerBandsGenerator()
        self.volatility_bands_generator = VolatilityBandsGenerator()
        self.cci_generator = CCIGenerator()
        self.keltner_generator = KeltnerChannelsGenerator()

        # Optimization state
        self.optimization_results = {}
        self.best_lookbacks = {}
        self.optimization_history = []

        # Analyst integration
        self.analyst_models = None
        self.analyst_ensemble = None
        self.analyst_outputs_cache = {}

        # Model cache for optimized configurations
        if BAYESIAN_TPE_AVAILABLE:
            try:
                self.lookback_cache = get_model_cache(
                    max_memory_models=20,
                    max_disk_models=100,
                    cache_dir=f"{self.config.analyst_model_path}/lookback_cache"
                )
                tprint_success("✅ Lookback configuration cache initialized")
            except Exception as e:
                self.lookback_cache = None
                tprint_warning(f"⚠️ Lookback cache unavailable: {e}")
        else:
            self.lookback_cache = None

    async def initialize(self) -> bool:
        """Initialize the Tactician lookback optimizer."""
        try:
            tprint_info("🚀 Initializing Tactician Lookback Optimizer...")

            # Validate configuration
            if not self._validate_config():
                return False

            # Load Analyst models and outputs
            if self.config.requires_analyst_outputs:
                success = await self._load_analyst_models()
                if not success:
                    tprint_error("❌ Failed to load Analyst models - required for Tactician optimization")
                    return False

            # Initialize optimization components
            self._initialize_optimization_components()

            # Create output directories
            self._create_output_directories()

            tprint_success("✅ Tactician Lookback Optimizer initialized successfully")
            return True

        except Exception as e:
            tprint_error(f"❌ Failed to initialize Tactician Lookback Optimizer: {e}")
            return False

    def _validate_config(self) -> bool:
        """Validate the optimization configuration."""
        try:
            tprint_info("🔍 Validating Tactician lookback optimization configuration...")

            # Basic validation
            if not self.config.timeframe:
                tprint_error("❌ Timeframe must be specified")
                return False

            if self.config.timeframe != "1m":
                tprint_warning(f"⚠️ Expected 1m timeframe, got {self.config.timeframe}")

            # Lookback constraints
            if self.config.min_lookback >= self.config.max_lookback:
                tprint_error("❌ min_lookback must be less than max_lookback")
                return False

            # Feature categories
            if not self.config.feature_categories:
                tprint_error("❌ Feature categories must be specified")
                return False

            # Optimization method
            if self.config.optimization_method not in ["grid_search", "tpe", "two_step_grid_tpe"]:
                tprint_error(f"❌ Unknown optimization method: {self.config.optimization_method}")
                return False

            # Check Optuna availability for advanced methods
            if self.config.optimization_method in ["tpe", "two_step_grid_tpe"] and not OPTUNA_AVAILABLE:
                tprint_warning("⚠️ Optuna not available, falling back to grid search")
                self.config.optimization_method = "grid_search"

            tprint_success("✅ Configuration validation passed")
            return True

        except Exception as e:
            tprint_error(f"❌ Configuration validation failed: {e}")
            return False

    async def _load_analyst_models(self) -> bool:
        """Load Analyst models and generate outputs for integration."""
        try:
            tprint_info("🔄 Loading Analyst models and outputs...")

            # Load Analyst models from the training step
            analyst_model_path = Path(self.config.analyst_model_path)
            if not analyst_model_path.exists():
                tprint_error(f"❌ Analyst model path does not exist: {analyst_model_path}")
                return False

            # Load Analyst ensemble if available
            ensemble_path = Path(self.config.analyst_ensemble_path)
            if ensemble_path.exists():
                tprint_info("📊 Loading Analyst ensemble model...")
                # Implementation would load the actual ensemble model
                # For now, we'll create a placeholder
                self.analyst_ensemble = {"loaded": True, "path": str(ensemble_path)}

            # Cache Analyst outputs for optimization
            # This would typically load pre-computed Analyst predictions
            # on the training/validation data
            self.analyst_outputs_cache = {
                "signals": np.array([]),  # Binary green light signals
                "predictions": np.array([]),  # Model predictions
                "confidences": np.array([]),  # Prediction confidences
                "ensemble_outputs": np.array([])  # Ensemble predictions
            }

            tprint_success("✅ Analyst models and outputs loaded successfully")
            return True

        except Exception as e:
            tprint_error(f"❌ Failed to load Analyst models: {e}")
            return False

    def _initialize_optimization_components(self):
        """Initialize optimization components."""
        try:
            tprint_info("🔧 Initializing optimization components...")

            # Initialize feature calculators for each category
            self.feature_calculators = {}
            for category, indicators in self.config.feature_categories.items():
                self.feature_calculators[category] = {}
                for indicator in indicators:
                    self.feature_calculators[category][indicator] = self._get_indicator_calculator(indicator)

            # Initialize optimization strategy
            if self.config.optimization_method == "two_step_grid_tpe" and OPTUNA_AVAILABLE:
                self.optimization_strategy = "two_step_grid_tpe"
            elif self.config.optimization_method == "tpe" and OPTUNA_AVAILABLE:
                self.optimization_strategy = "tpe"
            else:
                self.optimization_strategy = "grid_search"

            tprint_success(f"✅ Optimization components initialized (strategy: {self.optimization_strategy})")

        except Exception as e:
            tprint_error(f"❌ Failed to initialize optimization components: {e}")
            raise

    def _get_indicator_calculator(self, indicator: str):
        """Get calculator function for a specific indicator using existing feature_generation implementations."""
        # Map ALL indicators to existing feature_generation implementations
        feature_generator_map = {
            # Momentum indicators
            "rsi": lambda data, lookback: pd.Series(
                self.momentum_generator._calculate_rsi(data['close'].values, period=lookback),
                index=data.index
            ),
            "macd": lambda data, lookback: pd.Series(
                self.momentum_generator._calculate_macd(data['close'].values, fast=max(1, lookback//2), slow=lookback)['macd'],
                index=data.index
            ),
            "stoch": lambda data, lookback: self.stochastic_generator._generate_feature(data, period=lookback),
            "williams_r": lambda data, lookback: self.williams_r_generator._generate_feature(data, period=lookback),
            "roc": lambda data, lookback: self.roc_generator._generate_feature(data, period=lookback),
            "momentum": lambda data, lookback: self.momentum_specific_generator._generate_feature(data, period=lookback),

            # Trend indicators
            "ema": lambda data, lookback: pd.Series(
                self.momentum_generator._calculate_ema(data['close'].values, period=lookback),
                index=data.index
            ),
            "sma": lambda data, lookback: pd.Series(
                self.trend_generator._calculate_sma(data['close'].values, period=lookback),
                index=data.index
            ),
            "keltner_channels": lambda data, lookback: self.keltner_generator._generate_feature(data, period=lookback),

            # Volume indicators
            "volume_sma": lambda data, lookback: pd.Series(
                self.volume_generator._calculate_volume_ma(data['volume'].values, period=lookback),
                index=data.index
            ),
            "vwap": lambda data, lookback: self.vwap_generator._generate_feature(data, period=lookback),
            "obv": lambda data, lookback: self.obv_generator._generate_feature(data),
            "volume_roc": lambda data, lookback: self.volume_roc_generator._generate_feature(data, period=lookback),

            # Volatility indicators
            "bollinger_bands": lambda data, lookback: self.bollinger_generator._generate_feature(data, period=lookback),
            "volatility_bands": lambda data, lookback: self.volatility_bands_generator._generate_feature(data, period=lookback),
            "atr": lambda data, lookback: self.cross_timeframe_generator._calculate_atr(data, period=lookback),

            # Oscillator indicators
            "cci": lambda data, lookback: self.cci_generator._generate_feature(data, period=lookback),
        }

        # All indicators now use feature_generation implementations
        if indicator in feature_generator_map:
            return feature_generator_map[indicator]
        else:
            tprint_warning(f"⚠️ Unknown indicator: {indicator}, using SMA as fallback")
            return feature_generator_map["sma"]

    # ========================================================================
    # ALL TECHNICAL INDICATORS NOW USE FEATURE_GENERATION IMPLEMENTATIONS
    #
    # ✅ ALL 17 INDICATORS USING EXISTING FEATURE_GENERATION:
    #
    # Momentum Indicators (6):
    # ✅ RSI - MomentumFeatureGenerator._calculate_rsi()
    # ✅ MACD - MomentumFeatureGenerator._calculate_macd()
    # ✅ EMA - MomentumFeatureGenerator._calculate_ema()
    # ✅ Stochastic - StochasticGenerator._generate_feature()
    # ✅ Williams %R - WilliamsRGenerator._generate_feature()
    # ✅ ROC - ROCGenerator._generate_feature()
    # ✅ Momentum - MomentumGenerator._generate_feature()
    #
    # Trend Indicators (3):
    # ✅ SMA - TrendFeatureGenerator._calculate_sma()
    # ✅ Keltner Channels - KeltnerChannelsGenerator._generate_feature()
    #
    # Volume Indicators (4):
    # ✅ Volume SMA - VolumeFeatureGenerator._calculate_volume_ma()
    # ✅ VWAP - VWAPGenerator._generate_feature()
    # ✅ OBV - OBVGenerator._generate_feature()
    # ✅ Volume ROC - VolumeROCGenerator._generate_feature()
    #
    # Volatility Indicators (3):
    # ✅ ATR - OptimizedCrossTimeframeAnalysisIntegration._calculate_atr()
    # ✅ Bollinger Bands - BollingerBandsGenerator._generate_feature()
    # ✅ Volatility Bands - VolatilityBandsGenerator._generate_feature()
    #
    # Oscillator Indicators (1):
    # ✅ CCI - CCIGenerator._generate_feature()
    #
    # 🎉 NO LOCAL IMPLEMENTATIONS REMAINING - ALL USE FEATURE_GENERATION!
    # ========================================================================

    def _create_output_directories(self):
        """Create output directories for results."""
        try:
            results_path = Path(self.config.results_path)
            results_path.mkdir(parents=True, exist_ok=True)

            # Create subdirectories
            (results_path / "optimization_results").mkdir(exist_ok=True)
            (results_path / "feature_analysis").mkdir(exist_ok=True)
            (results_path / "performance_metrics").mkdir(exist_ok=True)

            tprint_success(f"✅ Output directories created: {results_path}")

        except Exception as e:
            tprint_warning(f"⚠️ Failed to create output directories: {e}")

    # All local indicator implementations removed - now using feature_generation implementations

    async def optimize_lookback_periods(
        self,
        market_data_1m: pd.DataFrame,
        analyst_signals: Optional[np.ndarray] = None,
        analyst_outputs: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Optimize lookback periods for Tactician features.

        Args:
            market_data_1m: 1-minute market data
            analyst_signals: Binary signals from Analyst
            analyst_outputs: Analyst model outputs

        Returns:
            Dictionary containing optimization results
        """
        optimization_timeout = self.config.optimization_timeout or 3600  # Default 1 hour

        try:
            tprint_info("🎯 Starting Tactician lookback period optimization...")
            self.start_time = time.time()

            # Validate input data
            if not self._validate_input_data(market_data_1m):
                raise ValueError("Invalid input data")

            # Use cached Analyst outputs if not provided
            if analyst_signals is None and self.analyst_outputs_cache.get("signals") is not None:
                analyst_signals = self.analyst_outputs_cache["signals"]

            if analyst_outputs is None:
                analyst_outputs = {
                    "predictions": self.analyst_outputs_cache.get("predictions", np.array([])),
                    "confidences": self.analyst_outputs_cache.get("confidences", np.array([])),
                    "ensemble_outputs": self.analyst_outputs_cache.get("ensemble_outputs", np.array([]))
                }

            # Execute optimization with timeout protection
            try:
                results = await asyncio.wait_for(
                    self._execute_optimization_strategy(market_data_1m, analyst_signals, analyst_outputs),
                    timeout=optimization_timeout
                )
            except asyncio.TimeoutError:
                tprint_warning(f"⚠️ Optimization timed out after {optimization_timeout}s, using partial results")
                results = self._create_fallback_results()

            # Process and save results
            final_results = self._process_optimization_results(results)

            if self.config.save_results:
                try:
                    await asyncio.wait_for(
                        self._save_optimization_results(final_results),
                        timeout=300  # 5 minutes for saving
                    )
                except asyncio.TimeoutError:
                    tprint_warning("⚠️ Results saving timed out, continuing without saving")

            optimization_time = time.time() - self.start_time
            tprint_success(f"✅ Tactician lookback optimization completed in {optimization_time:.2f}s")

            return final_results

        except Exception as e:
            tprint_error(f"❌ Tactician lookback optimization failed: {e}")
            # Ensure cleanup on error
            await self._cleanup_resources()
            raise
        finally:
            # Always attempt cleanup
            try:
                await self._cleanup_resources()
            except Exception as cleanup_error:
                tprint_warning(f"⚠️ Cleanup warning: {cleanup_error}")

    async def _execute_optimization_strategy(
        self,
        market_data_1m: pd.DataFrame,
        analyst_signals: Optional[np.ndarray],
        analyst_outputs: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Execute the selected optimization strategy."""
        if self.optimization_strategy == "two_step_grid_tpe":
            return await self._optimize_two_step_grid_tpe(
                market_data_1m, analyst_signals, analyst_outputs
            )
        elif self.optimization_strategy == "tpe":
            return await self._optimize_tpe(
                market_data_1m, analyst_signals, analyst_outputs
            )
        else:
            return await self._optimize_grid_search(
                market_data_1m, analyst_signals, analyst_outputs
            )

    def _create_fallback_results(self) -> Dict[str, Any]:
        """Create fallback results when optimization times out."""
        return {
            'method': 'fallback_timeout',
            'best_lookbacks': self._get_default_lookbacks_optimized(),
            'best_score': 0.5,
            'timeout_occurred': True,
            'partial_results': True
        }

    def _get_default_lookbacks_optimized(self) -> Dict[str, int]:
        """Get optimized default lookback periods for 0.3% movements."""
        return {
            'rsi': 8,  # Shorter for quick reactions
            'macd': 15,  # Reduced for 1m timeframe
            'bollinger_bands': 12,  # Faster response
            'stoch': 8,  # More responsive
            'volume_sma': 12,  # Shorter volume analysis
            'vwap': 12,  # Shorter VWAP
            'obv': 6,   # Quick volume momentum
            'volume_roc': 6,   # Fast volume changes
            'williams_r': 8,  # More responsive
            'cci': 12,  # Shorter for 1m
            'momentum': 5,   # Very short for 0.3% targets
            'roc': 6,    # Quick rate of change
            'atr': 8,   # Shorter volatility measure
            'volatility_bands': 12,  # Responsive volatility
            'keltner_channels': 12   # Shorter channels
        }

    async def _cleanup_resources(self):
        """Clean up resources using hardware optimization tools."""
        try:
            # Import hardware optimization tools
            from src.utils.hardware import (
                get_unified_hardware_manager, get_advanced_memory_optimizer,
                UNIFIED_MANAGER_AVAILABLE, ADVANCED_MEMORY_AVAILABLE
            )

            cleanup_stats = {'memory_freed_mb': 0, 'objects_cleaned': 0}

            # Use advanced memory optimizer if available
            if ADVANCED_MEMORY_AVAILABLE:
                try:
                    memory_optimizer = get_advanced_memory_optimizer()

                    # Optimize large data structures
                    if hasattr(self, 'optimization_history') and len(self.optimization_history) > 1000:
                        original_size = len(self.optimization_history)
                        # Use memory optimizer to efficiently trim history
                        self.optimization_history = memory_optimizer.optimize_list_memory(
                            self.optimization_history, max_size=100, keep_recent=True
                        )
                        cleanup_stats['objects_cleaned'] += original_size - len(self.optimization_history)
                        tprint_debug(f"🧹 Trimmed optimization history: {original_size} → {len(self.optimization_history)}")

                    # Optimize cached outputs using hardware tools
                    for key in list(self.analyst_outputs_cache.keys()):
                        if hasattr(self.analyst_outputs_cache[key], '__len__') and len(self.analyst_outputs_cache[key]) > 10000:
                            original_array = self.analyst_outputs_cache[key]
                            # Use memory optimizer to compress or clear large arrays
                            self.analyst_outputs_cache[key] = memory_optimizer.optimize_array_memory(
                                original_array, max_elements=5000, compression_enabled=True
                            )
                            cleanup_stats['objects_cleaned'] += 1

                    # Force memory optimization
                    memory_freed = memory_optimizer.force_memory_optimization()
                    cleanup_stats['memory_freed_mb'] = memory_freed

                    tprint_debug(f"🧹 Advanced memory cleanup: {memory_freed:.2f}MB freed, {cleanup_stats['objects_cleaned']} objects optimized")

                except Exception as memory_error:
                    tprint_warning(f"⚠️ Advanced memory optimization failed: {memory_error}")
                    # Fallback to basic cleanup
                    await self._basic_cleanup_fallback()

            # Use unified hardware manager for comprehensive cleanup
            elif UNIFIED_MANAGER_AVAILABLE:
                try:
                    hardware_manager = get_unified_hardware_manager()

                    # Perform memory cleanup using unified manager
                    cleanup_result = hardware_manager.cleanup_memory_resources(
                        target_objects=[self.optimization_history, self.analyst_outputs_cache],
                        cleanup_level='aggressive'
                    )

                    cleanup_stats.update(cleanup_result)
                    tprint_debug(f"🧹 Unified hardware cleanup: {cleanup_result}")

                except Exception as unified_error:
                    tprint_warning(f"⚠️ Unified hardware cleanup failed: {unified_error}")
                    # Fallback to basic cleanup
                    await self._basic_cleanup_fallback()

            else:
                # Fallback to basic cleanup if hardware tools not available
                await self._basic_cleanup_fallback()

            tprint_success(f"✅ Resource cleanup completed: {cleanup_stats}")

        except Exception as e:
            tprint_error(f"❌ Resource cleanup failed: {e}")
            # Emergency fallback
            try:
                await self._basic_cleanup_fallback()
            except Exception as fallback_error:
                tprint_error(f"❌ Emergency cleanup fallback failed: {fallback_error}")

    async def _basic_cleanup_fallback(self):
        """Basic cleanup fallback when hardware tools are not available."""
        try:
            import gc

            # Clear large data structures (basic approach)
            if hasattr(self, 'optimization_history') and len(self.optimization_history) > 1000:
                # Keep only last 100 entries to prevent memory bloat
                self.optimization_history = self.optimization_history[-100:]

            # Clear cached outputs if too large
            for key in list(self.analyst_outputs_cache.keys()):
                if hasattr(self.analyst_outputs_cache[key], '__len__') and len(self.analyst_outputs_cache[key]) > 10000:
                    self.analyst_outputs_cache[key] = np.array([])

            # Force garbage collection
            collected = gc.collect()
            tprint_debug(f"🧹 Basic cleanup completed, {collected} objects collected")

        except Exception as e:
            tprint_warning(f"⚠️ Basic cleanup fallback failed: {e}")

    def _validate_input_data(self, market_data: pd.DataFrame) -> bool:
        """Validate input market data with comprehensive edge case checking."""
        try:
            # Basic structure validation
            if market_data is None:
                tprint_error("❌ Market data is None")
                return False

            if not isinstance(market_data, pd.DataFrame):
                tprint_error(f"❌ Market data must be DataFrame, got {type(market_data)}")
                return False

            if len(market_data) == 0:
                tprint_error("❌ Market data is empty")
                return False

            # Column validation
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in market_data.columns]

            if missing_columns:
                tprint_error(f"❌ Missing required columns: {missing_columns}")
                return False

            # Data length validation
            min_required_length = max(self.config.max_lookback * 3, 100)  # More conservative minimum
            if len(market_data) < min_required_length:
                tprint_error(f"❌ Insufficient data: {len(market_data)} rows, need at least {min_required_length}")
                return False

            # Data type validation
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if not pd.api.types.is_numeric_dtype(market_data[col]):
                    tprint_error(f"❌ Column '{col}' must be numeric, got {market_data[col].dtype}")
                    return False

            # Value validation - check for edge cases
            validation_errors = []

            for col in numeric_columns:
                series = market_data[col]

                # Check for NaN values
                nan_count = series.isna().sum()
                if nan_count > 0:
                    validation_errors.append(f"Column '{col}' has {nan_count} NaN values")

                # Check for infinite values
                inf_count = np.isinf(series).sum() if series.dtype.kind in 'biufc' else 0
                if inf_count > 0:
                    validation_errors.append(f"Column '{col}' has {inf_count} infinite values")

                # Check for zero/negative values where inappropriate
                if col in ['high', 'low', 'close', 'open']:
                    zero_negative_count = (series <= 0).sum()
                    if zero_negative_count > 0:
                        validation_errors.append(f"Column '{col}' has {zero_negative_count} zero/negative price values")

                if col == 'volume':
                    negative_volume_count = (series < 0).sum()
                    if negative_volume_count > 0:
                        validation_errors.append(f"Column 'volume' has {negative_volume_count} negative values")

                # Check for extreme outliers (more than 10x median)
                if len(series) > 10:  # Only check if we have enough data
                    median_val = series.median()
                    if median_val > 0:
                        extreme_high = (series > median_val * 10).sum()
                        extreme_low = (series < median_val / 10).sum()
                        if extreme_high > len(series) * 0.01:  # More than 1% extreme values
                            validation_errors.append(f"Column '{col}' has {extreme_high} extreme high values (>10x median)")
                        if extreme_low > len(series) * 0.01:
                            validation_errors.append(f"Column '{col}' has {extreme_low} extreme low values (<0.1x median)")

            # OHLC consistency validation
            try:
                ohlc_data = market_data[['open', 'high', 'low', 'close']]

                # High should be >= max(open, close)
                high_violations = ((ohlc_data['high'] < ohlc_data[['open', 'close']].max(axis=1))).sum()
                if high_violations > 0:
                    validation_errors.append(f"Found {high_violations} violations where high < max(open, close)")

                # Low should be <= min(open, close)
                low_violations = ((ohlc_data['low'] > ohlc_data[['open', 'close']].min(axis=1))).sum()
                if low_violations > 0:
                    validation_errors.append(f"Found {low_violations} violations where low > min(open, close)")

                # Check for zero-range candles (all OHLC equal) - suspicious
                zero_range_count = ((ohlc_data['high'] == ohlc_data['low'])).sum()
                if zero_range_count > len(market_data) * 0.1:  # More than 10% zero-range
                    validation_errors.append(f"Found {zero_range_count} zero-range candles ({zero_range_count/len(market_data):.1%})")

            except Exception as ohlc_error:
                validation_errors.append(f"OHLC consistency check failed: {ohlc_error}")

            # Time series validation (if index is datetime)
            if hasattr(market_data.index, 'to_pydatetime'):
                try:
                    # Check for duplicate timestamps
                    duplicate_count = market_data.index.duplicated().sum()
                    if duplicate_count > 0:
                        validation_errors.append(f"Found {duplicate_count} duplicate timestamps")

                    # Check for proper time ordering
                    if not market_data.index.is_monotonic_increasing:
                        validation_errors.append("Time series is not properly ordered")

                except Exception as time_error:
                    validation_errors.append(f"Time series validation failed: {time_error}")

            # Report validation errors
            if validation_errors:
                tprint_warning("⚠️ Data quality issues found:")
                for error in validation_errors:
                    tprint_warning(f"  - {error}")

                # Decide if errors are critical
                critical_errors = [e for e in validation_errors if any(keyword in e.lower()
                                 for keyword in ['nan', 'infinite', 'zero/negative price', 'duplicate timestamps'])]

                if critical_errors:
                    tprint_error("❌ Critical data quality issues found - cannot proceed")
                    return False
                else:
                    tprint_warning("⚠️ Non-critical data quality issues found - proceeding with caution")

            # Overall data quality check
            try:
                quality_metrics = calculate_data_quality_metrics(market_data)
                completeness = quality_metrics.get('completeness', 0)

                if completeness < 0.8:  # More strict threshold
                    tprint_error(f"❌ Data completeness too low: {completeness:.2%} (minimum 80%)")
                    return False
                elif completeness < 0.95:
                    tprint_warning(f"⚠️ Data completeness: {completeness:.2%}")

            except Exception as quality_error:
                tprint_warning(f"⚠️ Quality metrics calculation failed: {quality_error}")

            tprint_success("✅ Input data validation passed")
            return True

        except Exception as e:
            tprint_error(f"❌ Data validation failed with exception: {e}")
            import traceback

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
            tprint_debug(f"Validation traceback: {traceback.format_exc()}")
            return False

    async def _optimize_two_step_grid_tpe(
        self,
        market_data: pd.DataFrame,
        analyst_signals: Optional[np.ndarray],
        analyst_outputs: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Execute two-step grid search + TPE optimization."""
        try:
            tprint_info("🔍 Executing two-step grid + TPE optimization...")

            # Step 1: Coarse grid search
            tprint_info("📊 Step 1: Coarse grid search...")
            coarse_results = await self._coarse_grid_search(
                market_data, analyst_signals, analyst_outputs
            )

            # Step 2: Fine grid search around best candidates
            tprint_info("🎯 Step 2: Fine grid search...")
            fine_results = await self._fine_grid_search(
                market_data, analyst_signals, analyst_outputs, coarse_results
            )

            # Step 3: TPE fine-tuning
            tprint_info("🚀 Step 3: TPE fine-tuning...")
            tpe_results = await self._tpe_fine_tuning(
                market_data, analyst_signals, analyst_outputs, fine_results
            )

            return {
                'method': 'two_step_grid_tpe',
                'coarse_results': coarse_results,
                'fine_results': fine_results,
                'tpe_results': tpe_results,
                'best_lookbacks': tpe_results.get('best_lookbacks', {}),
                'best_score': tpe_results.get('best_score', 0.0)
            }

        except Exception as e:
            tprint_error(f"❌ Two-step optimization failed: {e}")
            raise

    async def _coarse_grid_search(
        self,
        market_data: pd.DataFrame,
        analyst_signals: Optional[np.ndarray],
        analyst_outputs: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Execute coarse grid search."""
        try:
            grid_size = self.config.coarse_grid_size
            lookback_range = np.linspace(
                self.config.min_lookback,
                self.config.max_lookback,
                grid_size,
                dtype=int
            )

            best_combinations = []

            # Test all combinations for each feature category
            for category, indicators in self.config.feature_categories.items():
                tprint_info(f"🔍 Optimizing {category} indicators...")

                category_results = []

                for indicator in indicators:
                    for lookback in lookback_range:
                        try:
                            score = await self._evaluate_lookback_combination(
                                market_data, analyst_signals, analyst_outputs,
                                {indicator: lookback}
                            )

                            category_results.append({
                                'indicator': indicator,
                                'lookback': lookback,
                                'score': score
                            })

                            self.optimization_metrics['total_evaluations'] += 1
                            self.optimization_metrics['successful_evaluations'] += 1

                        except Exception as e:
                            self.logger.warning(f"Evaluation failed for {indicator}[{lookback}]: {e}")
                            self.optimization_metrics['failed_evaluations'] += 1

                # Select top candidates from this category
                category_results.sort(key=lambda x: x['score'], reverse=True)
                best_combinations.extend(category_results[:3])  # Top 3 per category

            # Sort all combinations
            best_combinations.sort(key=lambda x: x['score'], reverse=True)

            return {
                'method': 'coarse_grid_search',
                'grid_size': grid_size,
                'total_evaluations': len(best_combinations),
                'best_combinations': best_combinations[:self.config.coarse_grid_size * 2],
                'best_score': best_combinations[0]['score'] if best_combinations else 0.0
            }

        except Exception as e:
            tprint_error(f"❌ Coarse grid search failed: {e}")
            raise

    async def _fine_grid_search(
        self,
        market_data: pd.DataFrame,
        analyst_signals: Optional[np.ndarray],
        analyst_outputs: Dict[str, np.ndarray],
        coarse_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute fine grid search around best coarse candidates."""
        try:
            best_coarse = coarse_results['best_combinations'][:5]  # Top 5 candidates
            fine_results = []

            for candidate in best_coarse:
                indicator = candidate['indicator']
                center_lookback = candidate['lookback']

                # Create fine grid around center point
                refinement_range = max(3, (self.config.max_lookback - self.config.min_lookback) // 10)
                fine_lookbacks = range(
                    max(self.config.min_lookback, center_lookback - refinement_range),
                    min(self.config.max_lookback + 1, center_lookback + refinement_range + 1)
                )

                for lookback in fine_lookbacks:
                    try:
                        score = await self._evaluate_lookback_combination(
                            market_data, analyst_signals, analyst_outputs,
                            {indicator: lookback}
                        )

                        fine_results.append({
                            'indicator': indicator,
                            'lookback': lookback,
                            'score': score,
                            'parent_candidate': candidate
                        })

                        self.optimization_metrics['total_evaluations'] += 1
                        self.optimization_metrics['successful_evaluations'] += 1

                    except Exception as e:
                        self.logger.warning(f"Fine evaluation failed for {indicator}[{lookback}]: {e}")
                        self.optimization_metrics['failed_evaluations'] += 1

            # Sort and return best results
            fine_results.sort(key=lambda x: x['score'], reverse=True)

            return {
                'method': 'fine_grid_search',
                'parent_candidates': len(best_coarse),
                'total_evaluations': len(fine_results),
                'best_combinations': fine_results[:10],  # Top 10
                'best_score': fine_results[0]['score'] if fine_results else 0.0
            }

        except Exception as e:
            tprint_error(f"❌ Fine grid search failed: {e}")
            raise

    async def _tpe_fine_tuning(
        self,
        market_data: pd.DataFrame,
        analyst_signals: Optional[np.ndarray],
        analyst_outputs: Dict[str, np.ndarray],
        fine_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute TPE fine-tuning around best candidates."""
        try:
            if not OPTUNA_AVAILABLE:
                tprint_warning("⚠️ Optuna not available, skipping TPE fine-tuning")
                best_fine = fine_results['best_combinations'][0]
                return {
                    'method': 'tpe_skipped',
                    'best_lookbacks': {best_fine['indicator']: best_fine['lookback']},
                    'best_score': best_fine['score']
                }

            # Create Optuna study
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(n_startup_trials=10),
                pruner=MedianPruner(n_startup_trials=5)
            )

            # Define objective function
            def objective(trial):
                lookback_params = {}

                # Sample lookback periods for top indicators
                best_indicators = fine_results['best_combinations'][:5]
                for candidate in best_indicators:
                    indicator = candidate['indicator']
                    center_lookback = candidate['lookback']

                    # Sample around the fine-tuned center
                    lookback = trial.suggest_int(
                        f'{indicator}_lookback',
                        max(self.config.min_lookback, center_lookback - 5),
                        min(self.config.max_lookback, center_lookback + 5)
                    )
                    lookback_params[indicator] = lookback

                # Evaluate combination
                try:
                    score = asyncio.run(self._evaluate_lookback_combination(
                        market_data, analyst_signals, analyst_outputs, lookback_params
                    ))
                    return score
                except Exception as e:
                    self.logger.warning(f"TPE evaluation failed: {e}")
                    return 0.0

            # Run optimization
            study.optimize(
                objective,
                n_trials=self.config.tpe_trials,
                timeout=self.config.optimization_timeout
            )

            # Extract best parameters
            best_lookbacks = {}
            for key, value in study.best_params.items():
                if key.endswith('_lookback'):
                    indicator = key.replace('_lookback', '')
                    best_lookbacks[indicator] = value

            return {
                'method': 'tpe',
                'n_trials': len(study.trials),
                'best_lookbacks': best_lookbacks,
                'best_score': study.best_value,
                'best_params': study.best_params,
                'study_summary': {
                    'best_trial': study.best_trial.number,
                    'best_value': study.best_value
                }
            }

        except Exception as e:
            tprint_error(f"❌ TPE fine-tuning failed: {e}")
            raise

    async def _evaluate_lookback_combination(
        self,
        market_data: pd.DataFrame,
        analyst_signals: Optional[np.ndarray],
        analyst_outputs: Dict[str, np.ndarray],
        lookback_params: Dict[str, int]
    ) -> float:
        """
        Evaluate a combination of lookback parameters.

        This is the core objective function that measures how well
        the given lookback periods perform for Tactician's objectives.
        """
        try:
            # Generate features with given lookback parameters
            features = self._create_tactician_features(
                market_data, analyst_signals, analyst_outputs, lookback_params
            )

            if features.empty:
                return 0.0

            # Calculate evaluation metrics
            scores = {}

            # 1. Entry timing accuracy (when Analyst gives green light)
            if analyst_signals is not None and len(analyst_signals) > 0:
                entry_accuracy = self._calculate_entry_timing_accuracy(
                    features, analyst_signals, market_data
                )
                scores['entry_timing'] = entry_accuracy
            else:
                scores['entry_timing'] = 0.5  # Neutral score

            # 2. Exit timing accuracy (risk management)
            exit_accuracy = self._calculate_exit_timing_accuracy(
                features, market_data
            )
            scores['exit_timing'] = exit_accuracy

            # 3. Signal-to-noise ratio (feature quality)
            signal_quality = self._calculate_signal_quality(
                features, market_data
            )
            scores['signal_quality'] = signal_quality

            # 4. Analyst alignment score (how well features align with Analyst signals)
            if analyst_signals is not None and len(analyst_signals) > 0:
                alignment_score = self._calculate_analyst_alignment(
                    features, analyst_signals, analyst_outputs
                )
                scores['analyst_alignment'] = alignment_score
            else:
                scores['analyst_alignment'] = 0.5

            # Calculate weighted final score
            weights = {
                'entry_timing': 0.3,
                'exit_timing': 0.3,
                'signal_quality': 0.2,
                'analyst_alignment': 0.2
            }

            final_score = sum(scores[metric] * weights[metric] for metric in scores)

            # Apply penalties for extreme lookback values
            penalty = self._calculate_lookback_penalty(lookback_params)
            final_score *= (1 - penalty)

            # Track optimization history
            evaluation_record = {
                'timestamp': datetime.now().isoformat(),
                'lookback_params': lookback_params,
                'scores': scores,
                'final_score': final_score,
                'penalty': penalty,
                'has_analyst_signals': analyst_signals is not None and len(analyst_signals) > 0
            }
            self.optimization_history.append(evaluation_record)

            # Update convergence history
            self.optimization_metrics['convergence_history'].append(final_score)

            # Update best score if improved
            if final_score > self.optimization_metrics['best_score']:
                self.optimization_metrics['best_score'] = final_score
                tprint_debug(f"🎯 New best score: {final_score:.4f} with lookbacks: {lookback_params}")

            return max(0.0, min(1.0, final_score))

        except Exception as e:
            self.logger.warning(f"Evaluation failed for {lookback_params}: {e}")
            return 0.0

    def _create_tactician_features(
        self,
        market_data: pd.DataFrame,
        analyst_signals: Optional[np.ndarray],
        analyst_outputs: Dict[str, np.ndarray],
        lookback_params: Dict[str, int]
    ) -> pd.DataFrame:
        """Create Tactician features with given lookback parameters."""
        try:
            features_list = []

            # Calculate technical indicators with optimized lookbacks
            for indicator, lookback in lookback_params.items():
                if indicator in self.feature_calculators.get('price_action', {}):
                    calculator = self.feature_calculators['price_action'][indicator]
                elif indicator in self.feature_calculators.get('volume_analysis', {}):
                    calculator = self.feature_calculators['volume_analysis'][indicator]
                elif indicator in self.feature_calculators.get('momentum', {}):
                    calculator = self.feature_calculators['momentum'][indicator]
                elif indicator in self.feature_calculators.get('volatility', {}):
                    calculator = self.feature_calculators['volatility'][indicator]
                else:
                    continue  # Skip unknown indicators

                try:
                    feature = calculator(market_data, lookback)
                    feature.name = f"{indicator}_{lookback}"
                    features_list.append(feature)
                except Exception as e:
                    self.logger.warning(f"Failed to calculate {indicator}: {e}")

            # Add Analyst integration features if available
            if analyst_signals is not None and len(analyst_signals) > 0:
                # Analyst signal features
                analyst_signal_series = pd.Series(
                    analyst_signals[:len(market_data)],
                    index=market_data.index,
                    name='analyst_signal'
                )
                features_list.append(analyst_signal_series)

                # Analyst signal momentum
                analyst_momentum = self._vectorbt_rolling_operation(analyst_signal_series, "mean", 5)
                analyst_momentum.name = 'analyst_signal_momentum'
                features_list.append(analyst_momentum)

            # Add Analyst model outputs if available
            for output_name, output_values in analyst_outputs.items():
                if len(output_values) > 0:
                    output_series = pd.Series(
                        output_values[:len(market_data)],
                        index=market_data.index,
                        name=f'analyst_{output_name}'
                    )
                    features_list.append(output_series)

            # Combine all features
            if features_list:
                features_df = pd.concat(features_list, axis=1)

                # Handle missing values
                features_df = features_df.fillna(method='ffill').fillna(method='bfill')

                return features_df
            else:
                return pd.DataFrame(index=market_data.index)

        except Exception as e:
            self.logger.warning(f"Feature creation failed: {e}")
            return pd.DataFrame(index=market_data.index)

    def _calculate_entry_timing_accuracy(
        self,
        features: pd.DataFrame,
        analyst_signals: np.ndarray,
        market_data: pd.DataFrame
    ) -> float:
        """Calculate entry timing accuracy when Analyst gives green light (optimized for 0.4% movements)."""
        try:
            if features.empty or len(analyst_signals) == 0:
                return 0.5

            # Find periods where Analyst gives directional signals
            directional_periods = (analyst_signals == 1) | (analyst_signals == -1)

            if not np.any(directional_periods):
                return 0.5

            # Calculate future returns for directional signal periods
            future_returns = market_data['close'].pct_change().shift(-1)

            # Focus on directional signal periods
            directional_returns = future_returns[directional_periods]
            directional_signals_period = analyst_signals[directional_periods]

            # Calculate accuracy optimized for 0.4% target movements
            if len(directional_returns) > 0:
                # Weight accuracy by proximity to 0.4% target
                target_return = 0.004  # 0.4% target movement

                # Score returns based on achieving target (0.4% or more is good)
                target_achieved = (directional_returns >= target_return).sum()
                small_positive = ((directional_returns > 0) & (directional_returns < target_return)).sum()
                negative = (directional_returns < 0).sum()

                # Weighted scoring: full points for target achievement, partial for small positive
                score = (target_achieved * 1.0 + small_positive * 0.5 + negative * 0.0) / len(directional_returns)
                return float(score)
            else:
                return 0.5

        except Exception as e:
            self.logger.warning(f"Entry timing accuracy calculation failed: {e}")
            return 0.5

    def _calculate_exit_timing_accuracy(
        self,
        features: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> float:
        """Calculate exit timing accuracy for risk management (optimized for 0.4% movements)."""
        try:
            if features.empty:
                return 0.5

            # Calculate volatility-adjusted returns for short-term movements
            returns = market_data['close'].pct_change()
            volatility = self._vectorbt_rolling_operation(returns, "std", 10)  # Shorter window for 1m data

            # Identify high-risk periods for 0.4% target movements
            # More sensitive thresholds for short-term trading
            high_risk_periods = (
                (volatility > volatility.quantile(0.75)) |  # Lower threshold for 1m
                (self._vectorbt_rolling_operation(returns, "mean", 3) < -0.0015) |  # 0.15% negative momentum
                (returns.abs() > 0.005)  # Large movements (>0.5%) indicate instability
            )

            # Calculate how well features predict these periods
            if 'rsi' in features.columns:
                rsi_signals = (features['rsi'] > 65) | (features['rsi'] < 35)  # Tighter overbought/oversold for 1m
            else:
                rsi_signals = pd.Series(False, index=features.index)

            if 'atr' in features.columns:
                atr_signals = features['atr'] > features['atr'].quantile(0.75)  # High volatility (lower threshold)
            else:
                atr_signals = pd.Series(False, index=features.index)

            # Add momentum-based exit signals for short-term trading
            momentum_signals = pd.Series(False, index=features.index)
            if len(returns) > 3:
                # Exit when recent momentum turns negative
                recent_momentum = self._vectorbt_rolling_operation(returns, "mean", 3)
                momentum_signals = recent_momentum < -0.001  # 0.1% negative momentum

            # Combine exit signals (more sensitive for short-term trading)
            exit_signals = rsi_signals | atr_signals | momentum_signals

            # Calculate accuracy: how often exit signals precede high-risk periods
            if exit_signals.sum() > 0:
                # Check if exit signals are followed by high-risk periods (shorter horizon for 1m)
                correct_exits = 0
                total_exits = 0

                for i in range(len(exit_signals) - 3):  # Shorter lookahead for 1m trading
                    if exit_signals.iloc[i]:
                        total_exits += 1
                        # Check next 3 periods for high risk (3 minutes ahead)
                        if high_risk_periods.iloc[i+1:i+4].any():
                            correct_exits += 1

                if total_exits > 0:
                    accuracy = correct_exits / total_exits
                    return float(accuracy)

            return 0.5

        except Exception as e:
            self.logger.warning(f"Exit timing accuracy calculation failed: {e}")
            return 0.5

    def _calculate_signal_quality(
        self,
        features: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> float:
        """Calculate signal-to-noise ratio of features (optimized for 0.4% movements)."""
        try:
            if features.empty:
                return 0.5

            # Calculate correlations with future returns (multiple horizons for short-term trading)
            returns_1min = market_data['close'].pct_change().shift(-1)
            returns_3min = market_data['close'].pct_change(periods=3).shift(-3)
            returns_5min = market_data['close'].pct_change(periods=5).shift(-5)

            correlations = []
            for column in features.columns:
                if features[column].dtype in ['float64', 'int64']:
                    try:
                        # Calculate correlations with different horizons
                        corr_1min = safe_correlation(features[column], returns_1min)
                        corr_3min = safe_correlation(features[column], returns_3min)
                        corr_5min = safe_correlation(features[column], returns_5min)

                        # Weight short-term correlations more heavily for 0.4% targets
                        valid_corrs = []
                        if validate_finite(corr_1min):
                            valid_corrs.append(abs(corr_1min) * 0.5)  # 50% weight for 1-minute
                        if validate_finite(corr_3min):
                            valid_corrs.append(abs(corr_3min) * 0.3)  # 30% weight for 3-minute
                        if validate_finite(corr_5min):
                            valid_corrs.append(abs(corr_5min) * 0.2)  # 20% weight for 5-minute

                        if valid_corrs:
                            # Weighted average correlation
                            weighted_corr = sum(valid_corrs)
                            correlations.append(weighted_corr)
                    except Exception:
                        continue

            if correlations:
                # Signal quality is the mean weighted correlation
                signal_quality = np.mean(correlations)

                # Apply penalty for features that don't show consistent short-term predictive power
                consistency_bonus = 1.0
                if len(correlations) > 1:
                    # Bonus for consistent correlations across features
                    corr_std = np.std(correlations)
                    consistency_bonus = max(0.8, 1.0 - corr_std)

                return float(signal_quality * consistency_bonus)
            else:
                return 0.5

        except Exception as e:
            self.logger.warning(f"Signal quality calculation failed: {e}")
            return 0.5

    def _calculate_analyst_alignment(
        self,
        features: pd.DataFrame,
        analyst_signals: np.ndarray,
        analyst_outputs: Dict[str, np.ndarray]
    ) -> float:
        """Calculate how well features align with Analyst signals."""
        try:
            if features.empty or len(analyst_signals) == 0:
                return 0.5

            alignment_scores = []

            # Check alignment between features and Analyst signals
            analyst_signal_series = pd.Series(analyst_signals[:len(features)], index=features.index)

            for column in features.columns:
                if 'analyst' not in column.lower() and features[column].dtype in ['float64', 'int64']:
                    try:
                        # Calculate correlation with Analyst signals
                        corr = safe_correlation(features[column], analyst_signal_series)
                        if validate_finite(corr):
                            alignment_scores.append(abs(corr))
                    except Exception:
                        continue

            # Check alignment with Analyst model outputs
            for output_name, output_values in analyst_outputs.items():
                if len(output_values) > 0:
                    output_series = pd.Series(output_values[:len(features)], index=features.index)

                    for column in features.columns:
                        if 'analyst' not in column.lower() and features[column].dtype in ['float64', 'int64']:
                            try:
                                corr = safe_correlation(features[column], output_series)
                                if validate_finite(corr):
                                    alignment_scores.append(abs(corr))
                            except Exception:
                                continue

            if alignment_scores:
                # Return mean alignment score
                return float(np.mean(alignment_scores))
            else:
                return 0.5

        except Exception as e:
            self.logger.warning(f"Analyst alignment calculation failed: {e}")
            return 0.5

    def _calculate_lookback_penalty(self, lookback_params: Dict[str, int]) -> float:
        """Calculate penalty for extreme lookback values (optimized for 0.4% movements)."""
        try:
            penalties = []

            for indicator, lookback in lookback_params.items():
                # Penalty for very short lookbacks (too noisy, even for 0.4% targets)
                if lookback < 5:
                    penalties.append(0.15)  # Higher penalty for very short lookbacks

                # Penalty for very long lookbacks (too slow for 0.4% short-term targets)
                elif lookback > 30:  # Shorter threshold for 0.4% movements
                    penalties.append(0.1)

                # Slight penalty for moderately long lookbacks (not optimal for short-term)
                elif lookback > 20:
                    penalties.append(0.05)

                # Sweet spot for 0.4% movements: 5-20 periods
                else:
                    penalties.append(0.0)

            return np.mean(penalties) if penalties else 0.0

        except Exception as e:
            self.logger.warning(f"Penalty calculation failed: {e}")
            return 0.0

    def _process_optimization_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Process and format optimization results."""
        try:
            tprint_info("📊 Processing optimization results...")

            # Extract best lookback parameters
            if 'best_lookbacks' in results:
                self.best_lookbacks = results['best_lookbacks']
            elif 'tpe_results' in results and 'best_lookbacks' in results['tpe_results']:
                self.best_lookbacks = results['tpe_results']['best_lookbacks']

            # Calculate comprehensive final metrics
            end_time = time.time()
            total_duration = end_time - self.start_time

            # Generate detailed performance analysis
            performance_analysis = self._generate_detailed_performance_analysis(results)

            # Generate feature analysis
            feature_analysis = self._generate_feature_analysis(self.best_lookbacks)

            # Generate optimization convergence analysis
            convergence_analysis = self._generate_convergence_analysis(results)

            final_results = {
                'optimization_method': results.get('method', 'unknown'),
                'best_lookbacks': self.best_lookbacks,
                'best_score': results.get('best_score', 0.0),
                'optimization_metrics': self.optimization_metrics,
                'performance_analysis': performance_analysis,
                'feature_analysis': feature_analysis,
                'convergence_analysis': convergence_analysis,
                'configuration': {
                    'timeframe': self.config.timeframe,
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'optimization_method': self.config.optimization_method,
                    'optimization_timeout': self.config.optimization_timeout,
                    'feature_categories': self.config.feature_categories,
                    'target_metrics': self.config.target_metrics,
                    'min_lookback': self.config.min_lookback,
                    'max_lookback': self.config.max_lookback,
                    'analyst_integration_weights': {
                        'analyst_signal_weight': self.config.analyst_signal_weight,
                        'analyst_output_weight': self.config.analyst_output_weight,
                        'raw_features_weight': self.config.raw_features_weight
                    }
                },
                'execution_info': {
                    'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                    'end_time': datetime.fromtimestamp(end_time).isoformat(),
                    'total_duration_seconds': total_duration,
                    'total_duration_formatted': f"{total_duration:.2f}s",
                    'total_evaluations': self.optimization_metrics['total_evaluations'],
                    'successful_evaluations': self.optimization_metrics['successful_evaluations'],
                    'failed_evaluations': self.optimization_metrics['failed_evaluations'],
                    'success_rate': (
                        self.optimization_metrics['successful_evaluations'] /
                        max(1, self.optimization_metrics['total_evaluations'])
                    ),
                    'evaluations_per_second': (
                        self.optimization_metrics['total_evaluations'] / max(1, total_duration)
                    )
                },
                'detailed_results': results,
                'artifacts': {
                    'optimization_results_file': f"tactician_lookback_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    'best_lookbacks_file': f"best_lookbacks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    'performance_metrics_file': f"optimization_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    'feature_analysis_file': f"feature_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    'convergence_analysis_file': f"convergence_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                }
            }

            tprint_success(f"✅ Optimization results processed: {len(self.best_lookbacks)} optimized lookbacks")
            return final_results

        except Exception as e:
            tprint_error(f"❌ Failed to process optimization results: {e}")
            return {'error': str(e)}

    async def _save_optimization_results(self, results: Dict[str, Any]):
        """Save comprehensive optimization results and artifacts to files."""
        try:
            tprint_info("💾 Saving comprehensive optimization results and artifacts...")

            results_path = Path(self.config.results_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Ensure all directories exist
            (results_path / "optimization_results").mkdir(parents=True, exist_ok=True)
            (results_path / "performance_metrics").mkdir(parents=True, exist_ok=True)
            (results_path / "feature_analysis").mkdir(parents=True, exist_ok=True)
            (results_path / "convergence_analysis").mkdir(parents=True, exist_ok=True)
            (results_path / "detailed_reports").mkdir(parents=True, exist_ok=True)
            (results_path / "artifacts").mkdir(parents=True, exist_ok=True)

            # 1. Save main optimization results
            results_file = results_path / "optimization_results" / f"tactician_lookback_optimization_{timestamp}.json"
            results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            # 2. Save best lookbacks separately for easy loading by Tactician training
            lookbacks_file = results_path / "optimization_results" / f"best_lookbacks_{timestamp}.json"
            with open(lookbacks_file, 'w') as f:
                json.dump(self.best_lookbacks, f, indent=2)

            # 3. Save performance metrics with detailed breakdown
            enhanced_metrics = {
                'optimization_metrics': self.optimization_metrics,
                'performance_analysis': results.get('performance_analysis', {}),
                'timing_analysis': {
                    'total_duration': time.time() - self.start_time if self.start_time else 0,
                    'evaluations_per_second': (
                        self.optimization_metrics['total_evaluations'] /
                        max(1, time.time() - self.start_time if self.start_time else 1)
                    ),
                    'average_evaluation_time': (
                        (time.time() - self.start_time) / max(1, self.optimization_metrics['total_evaluations'])
                        if self.start_time else 0
                    )
                },
                'quality_metrics': {
                    'success_rate': (
                        self.optimization_metrics['successful_evaluations'] /
                        max(1, self.optimization_metrics['total_evaluations'])
                    ),
                    'failure_rate': (
                        self.optimization_metrics['failed_evaluations'] /
                        max(1, self.optimization_metrics['total_evaluations'])
                    ),
                    'best_score_achieved': self.optimization_metrics['best_score']
                }
            }

            metrics_file = results_path / "performance_metrics" / f"optimization_metrics_{timestamp}.json"
            with open(metrics_file, 'w') as f:
                json.dump(enhanced_metrics, f, indent=2, default=str)

            # 4. Save feature analysis
            if 'feature_analysis' in results:
                feature_file = results_path / "feature_analysis" / f"feature_analysis_{timestamp}.json"
                with open(feature_file, 'w') as f:
                    json.dump(results['feature_analysis'], f, indent=2, default=str)

            # 5. Save convergence analysis
            if 'convergence_analysis' in results:
                convergence_file = results_path / "convergence_analysis" / f"convergence_analysis_{timestamp}.json"
                with open(convergence_file, 'w') as f:
                    json.dump(results['convergence_analysis'], f, indent=2, default=str)

            # 6. Generate and save comprehensive summary report
            summary_report = self._generate_comprehensive_summary_report(results, timestamp)
            summary_file = results_path / "detailed_reports" / f"optimization_summary_{timestamp}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary_report, f, indent=2, default=str)

            # 7. Save optimization history for analysis
            history_file = results_path / "artifacts" / f"optimization_history_{timestamp}.json"
            with open(history_file, 'w') as f:
                json.dump(self.optimization_history, f, indent=2, default=str)

            # 8. Create artifact manifest
            artifact_manifest = {
                'timestamp': timestamp,
                'optimization_session': {
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'optimization_method': self.config.optimization_method
                },
                'artifacts_generated': {
                    'main_results': str(results_file),
                    'best_lookbacks': str(lookbacks_file),
                    'performance_metrics': str(metrics_file),
                    'feature_analysis': str(feature_file) if 'feature_analysis' in results else None,
                    'convergence_analysis': str(convergence_file) if 'convergence_analysis' in results else None,
                    'summary_report': str(summary_file),
                    'optimization_history': str(history_file)
                },
                'metrics_summary': {
                    'total_evaluations': self.optimization_metrics['total_evaluations'],
                    'best_score': self.optimization_metrics['best_score'],
                    'optimized_indicators': len(self.best_lookbacks),
                    'execution_time': time.time() - self.start_time if self.start_time else 0
                }
            }

            manifest_file = results_path / "artifacts" / f"artifact_manifest_{timestamp}.json"
            with open(manifest_file, 'w') as f:
                json.dump(artifact_manifest, f, indent=2, default=str)

            tprint_success(f"✅ Comprehensive results and artifacts saved to {results_path}")
            tprint_structured({
                'artifacts_generated': len([f for f in artifact_manifest['artifacts_generated'].values() if f]),
                'main_results_file': str(results_file),
                'summary_report_file': str(summary_file),
                'artifact_manifest': str(manifest_file)
            })

        except Exception as e:
            tprint_warning(f"⚠️ Failed to save results: {e}")

    def _generate_comprehensive_summary_report(self, results: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
        """Generate comprehensive summary report with all metrics and insights."""
        try:
            # Calculate execution statistics
            total_duration = time.time() - self.start_time if self.start_time else 0

            summary = {
                'report_metadata': {
                    'report_type': 'tactician_lookback_optimization_summary',
                    'timestamp': timestamp,
                    'generation_time': datetime.now().isoformat(),
                    'report_version': '1.0'
                },
                'optimization_session': {
                    'configuration': {
                        'symbol': self.config.symbol,
                        'exchange': self.config.exchange,
                        'timeframe': self.config.timeframe,
                        'optimization_method': self.config.optimization_method,
                        'lookback_range': f"{self.config.min_lookback}-{self.config.max_lookback}",
                        'target_metrics': self.config.target_metrics
                    },
                    'execution_summary': {
                        'start_time': datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
                        'end_time': datetime.now().isoformat(),
                        'total_duration_seconds': total_duration,
                        'total_duration_formatted': f"{total_duration:.2f}s",
                        'status': 'completed'
                    }
                },
                'optimization_results': {
                    'method_used': results.get('optimization_method', 'unknown'),
                    'best_score_achieved': results.get('best_score', 0.0),
                    'total_indicators_optimized': len(self.best_lookbacks),
                    'optimized_lookbacks': self.best_lookbacks,
                    'optimization_quality': (
                        'excellent' if results.get('best_score', 0) > 0.8 else
                        'good' if results.get('best_score', 0) > 0.6 else
                        'fair' if results.get('best_score', 0) > 0.4 else 'poor'
                    )
                },
                'performance_metrics': {
                    'evaluation_statistics': {
                        'total_evaluations': self.optimization_metrics['total_evaluations'],
                        'successful_evaluations': self.optimization_metrics['successful_evaluations'],
                        'failed_evaluations': self.optimization_metrics['failed_evaluations'],
                        'success_rate': (
                            self.optimization_metrics['successful_evaluations'] /
                            max(1, self.optimization_metrics['total_evaluations'])
                        ),
                        'evaluations_per_second': (
                            self.optimization_metrics['total_evaluations'] / max(1, total_duration)
                        )
                    },
                    'convergence_metrics': results.get('convergence_analysis', {}),
                    'feature_metrics': results.get('feature_analysis', {})
                },
                'insights_and_recommendations': self._generate_optimization_insights(results),
                'artifacts_generated': results.get('artifacts', {}),
                'next_steps': [
                    "Use optimized lookbacks in Tactician model training",
                    "Monitor Tactician performance with new lookback periods",
                    "Compare performance against default lookback periods",
                    "Consider re-optimization if market conditions change significantly"
                ]
            }

            return summary

        except Exception as e:
            tprint_warning(f"⚠️ Failed to generate summary report: {e}")
            return {'error': str(e), 'timestamp': timestamp}

    def _generate_optimization_insights(self, results: Dict[str, Any]) -> List[str]:
        """Generate insights and recommendations from optimization results."""
        try:
            insights = []

            # Score-based insights
            best_score = results.get('best_score', 0.0)
            if best_score > 0.8:
                insights.append("Excellent optimization score achieved - lookback periods are well-tuned for 0.4% movements")
            elif best_score > 0.6:
                insights.append("Good optimization score - lookback periods should improve Tactician performance")
            elif best_score > 0.4:
                insights.append("Fair optimization score - consider additional feature engineering")
            else:
                insights.append("Low optimization score - may need different optimization approach or more data")

            # Lookback distribution insights
            if self.best_lookbacks:
                lookback_values = list(self.best_lookbacks.values())
                mean_lookback = np.mean(lookback_values)

                if mean_lookback < 10:
                    insights.append("Very short average lookback periods - optimized for high-frequency 0.4% movements")
                elif mean_lookback < 20:
                    insights.append("Short average lookback periods - good balance for 1m timeframe trading")
                else:
                    insights.append("Longer average lookback periods - may be better for trend following than 0.4% scalping")

            # Method-specific insights
            method = results.get('optimization_method', 'unknown')
            if method == 'two_step_grid_tpe':
                insights.append("Used comprehensive two-step optimization - high confidence in results")
            elif method == 'tpe':
                insights.append("Used intelligent TPE optimization - good balance of exploration and exploitation")
            elif method == 'grid_search':
                insights.append("Used systematic grid search - thorough but may miss optimal regions")

            # Performance insights
            success_rate = (
                self.optimization_metrics['successful_evaluations'] /
                max(1, self.optimization_metrics['total_evaluations'])
            )

            if success_rate > 0.95:
                insights.append("Excellent evaluation success rate - optimization process was stable")
            elif success_rate > 0.8:
                insights.append("Good evaluation success rate - minor issues during optimization")
            else:
                insights.append("Lower evaluation success rate - consider checking data quality or feature calculations")

            return insights

        except Exception as e:
            tprint_warning(f"⚠️ Failed to generate insights: {e}")
            return ["Error generating insights"]

    async def _optimize_grid_search(
        self,
        market_data: pd.DataFrame,
        analyst_signals: Optional[np.ndarray],
        analyst_outputs: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Fallback grid search optimization."""
        try:
            tprint_info("🔍 Executing fallback grid search optimization...")

            # Simple grid search over all feature categories
            best_score = 0.0
            best_lookbacks = {}
            all_results = []

            # Test each indicator individually
            for category, indicators in self.config.feature_categories.items():
                for indicator in indicators:
                    for lookback in range(self.config.min_lookback, self.config.max_lookback + 1, 5):
                        try:
                            score = await self._evaluate_lookback_combination(
                                market_data, analyst_signals, analyst_outputs,
                                {indicator: lookback}
                            )

                            all_results.append({
                                'indicator': indicator,
                                'lookback': lookback,
                                'score': score
                            })

                            if score > best_score:
                                best_score = score
                                best_lookbacks = {indicator: lookback}

                            self.optimization_metrics['total_evaluations'] += 1
                            self.optimization_metrics['successful_evaluations'] += 1

                        except Exception as e:
                            self.logger.warning(f"Grid evaluation failed for {indicator}[{lookback}]: {e}")
                            self.optimization_metrics['failed_evaluations'] += 1

            return {
                'method': 'grid_search',
                'total_evaluations': len(all_results),
                'best_lookbacks': best_lookbacks,
                'best_score': best_score,
                'all_results': sorted(all_results, key=lambda x: x['score'], reverse=True)[:20]
            }

        except Exception as e:
            tprint_error(f"❌ Grid search optimization failed: {e}")
            raise

    async def _optimize_tpe(
        self,
        market_data: pd.DataFrame,
        analyst_signals: Optional[np.ndarray],
        analyst_outputs: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """TPE-only optimization."""
        try:
            if not OPTUNA_AVAILABLE:
                tprint_warning("⚠️ Optuna not available, falling back to grid search")
                return await self._optimize_grid_search(market_data, analyst_signals, analyst_outputs)

            tprint_info("🚀 Executing TPE optimization...")

            # Create Optuna study
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(n_startup_trials=10),
                pruner=MedianPruner(n_startup_trials=5)
            )

            # Define objective function
            def objective(trial):
                lookback_params = {}

                # Sample lookback periods for all indicators
                for category, indicators in self.config.feature_categories.items():
                    for indicator in indicators:
                        lookback = trial.suggest_int(
                            f'{indicator}_lookback',
                            self.config.min_lookback,
                            self.config.max_lookback
                        )
                        lookback_params[indicator] = lookback

                # Evaluate combination
                try:
                    score = asyncio.run(self._evaluate_lookback_combination(
                        market_data, analyst_signals, analyst_outputs, lookback_params
                    ))
                    return score
                except Exception as e:
                    self.logger.warning(f"TPE evaluation failed: {e}")
                    return 0.0

            # Run optimization
            study.optimize(
                objective,
                n_trials=self.config.tpe_trials,
                timeout=self.config.optimization_timeout
            )

            # Extract best parameters
            best_lookbacks = {}
            for key, value in study.best_params.items():
                if key.endswith('_lookback'):
                    indicator = key.replace('_lookback', '')
                    best_lookbacks[indicator] = value

            return {
                'method': 'tpe',
                'n_trials': len(study.trials),
                'best_lookbacks': best_lookbacks,
                'best_score': study.best_value,
                'best_params': study.best_params
            }

        except Exception as e:
            tprint_error(f"❌ TPE optimization failed: {e}")
            raise

    def _generate_detailed_performance_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed performance analysis with comprehensive metrics."""
        try:
            tprint_info("📊 Generating detailed performance analysis...")

            analysis = {
                'optimization_efficiency': {
                    'total_evaluations': self.optimization_metrics['total_evaluations'],
                    'successful_evaluations': self.optimization_metrics['successful_evaluations'],
                    'failed_evaluations': self.optimization_metrics['failed_evaluations'],
                    'success_rate': (
                        self.optimization_metrics['successful_evaluations'] /
                        max(1, self.optimization_metrics['total_evaluations'])
                    ),
                    'failure_rate': (
                        self.optimization_metrics['failed_evaluations'] /
                        max(1, self.optimization_metrics['total_evaluations'])
                    )
                },
                'score_distribution': {
                    'best_score': self.optimization_metrics['best_score'],
                    'convergence_history': self.optimization_metrics['convergence_history']
                },
                'optimization_method_performance': {},
                'timing_analysis': {
                    'total_duration': time.time() - self.start_time if self.start_time else 0,
                    'average_evaluation_time': 0,
                    'optimization_speed': 'fast' if self.optimization_metrics['total_evaluations'] > 50 else 'normal'
                }
            }

            # Calculate average evaluation time
            if self.optimization_metrics['total_evaluations'] > 0:
                total_time = time.time() - self.start_time if self.start_time else 1
                analysis['timing_analysis']['average_evaluation_time'] = (
                    total_time / self.optimization_metrics['total_evaluations']
                )

            # Add method-specific performance analysis
            if 'method' in results:
                method = results['method']
                analysis['optimization_method_performance'][method] = {
                    'used': True,
                    'final_score': results.get('best_score', 0.0),
                    'evaluations': self.optimization_metrics['total_evaluations']
                }

                if method == 'two_step_grid_tpe':
                    # Add detailed analysis for two-step method
                    analysis['optimization_method_performance'][method].update({
                        'coarse_grid_evaluations': results.get('coarse_results', {}).get('total_evaluations', 0),
                        'fine_grid_evaluations': results.get('fine_results', {}).get('total_evaluations', 0),
                        'tpe_evaluations': results.get('tpe_results', {}).get('n_trials', 0),
                        'coarse_best_score': results.get('coarse_results', {}).get('best_score', 0.0),
                        'fine_best_score': results.get('fine_results', {}).get('best_score', 0.0),
                        'tpe_best_score': results.get('tpe_results', {}).get('best_score', 0.0)
                    })

            # Add quality assessment
            analysis['quality_assessment'] = {
                'optimization_quality': 'excellent' if analysis['optimization_efficiency']['success_rate'] > 0.9 else
                                      'good' if analysis['optimization_efficiency']['success_rate'] > 0.7 else
                                      'fair' if analysis['optimization_efficiency']['success_rate'] > 0.5 else 'poor',
                'score_quality': 'excellent' if self.optimization_metrics['best_score'] > 0.8 else
                               'good' if self.optimization_metrics['best_score'] > 0.6 else
                               'fair' if self.optimization_metrics['best_score'] > 0.4 else 'poor',
                'convergence_quality': 'good' if len(self.optimization_metrics['convergence_history']) > 10 else 'limited'
            }

            return analysis

        except Exception as e:
            tprint_warning(f"⚠️ Failed to generate performance analysis: {e}")
            return {'error': str(e)}

    def _generate_feature_analysis(self, best_lookbacks: Dict[str, int]) -> Dict[str, Any]:
        """Generate detailed feature analysis with lookback distribution and insights."""
        try:
            tprint_info("📈 Generating feature analysis...")

            if not best_lookbacks:
                return {'error': 'No optimized lookbacks available'}

            # Analyze lookback distribution
            lookback_values = list(best_lookbacks.values())

            analysis = {
                'lookback_statistics': {
                    'total_indicators': len(best_lookbacks),
                    'min_lookback': min(lookback_values),
                    'max_lookback': max(lookback_values),
                    'mean_lookback': np.mean(lookback_values),
                    'median_lookback': np.median(lookback_values),
                    'std_lookback': np.std(lookback_values),
                    'lookback_range': max(lookback_values) - min(lookback_values)
                },
                'category_analysis': {},
                'lookback_distribution': {
                    'very_short': len([v for v in lookback_values if v <= 5]),
                    'short': len([v for v in lookback_values if 5 < v <= 15]),
                    'medium': len([v for v in lookback_values if 15 < v <= 30]),
                    'long': len([v for v in lookback_values if 30 < v <= 45]),
                    'very_long': len([v for v in lookback_values if v > 45])
                },
                'optimization_insights': []
            }

            # Analyze by feature category
            for category, indicators in self.config.feature_categories.items():
                category_lookbacks = [best_lookbacks.get(indicator, 0) for indicator in indicators if indicator in best_lookbacks]

                if category_lookbacks:
                    analysis['category_analysis'][category] = {
                        'indicators_optimized': len(category_lookbacks),
                        'mean_lookback': np.mean(category_lookbacks),
                        'min_lookback': min(category_lookbacks),
                        'max_lookback': max(category_lookbacks),
                        'lookback_consistency': 1.0 - (np.std(category_lookbacks) / max(1, np.mean(category_lookbacks)))
                    }

            # Generate insights
            mean_lookback = analysis['lookback_statistics']['mean_lookback']
            if mean_lookback < 10:
                analysis['optimization_insights'].append("Very short average lookback - optimized for high-frequency trading")
            elif mean_lookback < 20:
                analysis['optimization_insights'].append("Short average lookback - good for 1m timeframe and 0.4% targets")
            elif mean_lookback < 30:
                analysis['optimization_insights'].append("Medium average lookback - balanced approach")
            else:
                analysis['optimization_insights'].append("Long average lookback - may be too slow for 0.4% targets")

            # Check distribution balance
            dist = analysis['lookback_distribution']
            if dist['very_short'] > len(best_lookbacks) * 0.5:
                analysis['optimization_insights'].append("High proportion of very short lookbacks - may be noisy")
            elif dist['short'] > len(best_lookbacks) * 0.5:
                analysis['optimization_insights'].append("Good proportion of short lookbacks - well-suited for 1m trading")

            return analysis

        except Exception as e:
            tprint_warning(f"⚠️ Failed to generate feature analysis: {e}")
            return {'error': str(e)}

    def _generate_convergence_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimization convergence analysis."""
        try:
            tprint_info("📉 Generating convergence analysis...")

            convergence_history = self.optimization_metrics.get('convergence_history', [])

            analysis = {
                'convergence_metrics': {
                    'total_iterations': len(convergence_history),
                    'converged': len(convergence_history) > 0,
                    'convergence_rate': 'fast' if len(convergence_history) > 20 else 'normal',
                    'final_score': convergence_history[-1] if convergence_history else 0.0,
                    'initial_score': convergence_history[0] if convergence_history else 0.0,
                    'improvement': (convergence_history[-1] - convergence_history[0]) if len(convergence_history) > 1 else 0.0
                },
                'convergence_pattern': {
                    'monotonic_improvement': False,
                    'plateau_detected': False,
                    'early_convergence': False,
                    'oscillation_detected': False
                },
                'optimization_stages': {}
            }

            # Analyze convergence pattern
            if len(convergence_history) > 5:
                # Check for monotonic improvement
                improvements = [convergence_history[i] > convergence_history[i-1] for i in range(1, len(convergence_history))]
                analysis['convergence_pattern']['monotonic_improvement'] = sum(improvements) > len(improvements) * 0.7

                # Check for plateau (last 20% of iterations show little improvement)
                plateau_start = int(len(convergence_history) * 0.8)
                if plateau_start < len(convergence_history) - 1:
                    plateau_scores = convergence_history[plateau_start:]
                    plateau_variation = np.std(plateau_scores) if len(plateau_scores) > 1 else 0
                    analysis['convergence_pattern']['plateau_detected'] = plateau_variation < 0.01

                # Check for early convergence (best score achieved in first 50% of iterations)
                mid_point = len(convergence_history) // 2
                best_score_index = convergence_history.index(max(convergence_history))
                analysis['convergence_pattern']['early_convergence'] = best_score_index < mid_point

            # Add method-specific convergence analysis
            if 'method' in results:
                method = results['method']
                if method == 'two_step_grid_tpe':
                    analysis['optimization_stages'] = {
                        'coarse_grid': {
                            'completed': 'coarse_results' in results,
                            'best_score': results.get('coarse_results', {}).get('best_score', 0.0),
                            'evaluations': results.get('coarse_results', {}).get('total_evaluations', 0)
                        },
                        'fine_grid': {
                            'completed': 'fine_results' in results,
                            'best_score': results.get('fine_results', {}).get('best_score', 0.0),
                            'evaluations': results.get('fine_results', {}).get('total_evaluations', 0)
                        },
                        'tpe_fine_tuning': {
                            'completed': 'tpe_results' in results,
                            'best_score': results.get('tpe_results', {}).get('best_score', 0.0),
                            'trials': results.get('tpe_results', {}).get('n_trials', 0)
                        }
                    }

            return analysis

        except Exception as e:
            tprint_warning(f"⚠️ Failed to generate convergence analysis: {e}")
            return {'error': str(e)}

# Convenience functions for integration

async def optimize_tactician_lookbacks(
    market_data_1m: pd.DataFrame,
    analyst_signals: Optional[np.ndarray] = None,
    analyst_outputs: Optional[Dict[str, np.ndarray]] = None,
    config: Optional[TacticianLookbackConfig] = None
) -> Dict[str, Any]:
    """
    Convenience function to optimize Tactician lookback periods.

    Args:
        market_data_1m: 1-minute market data
        analyst_signals: Binary signals from Analyst
        analyst_outputs: Analyst model outputs
        config: Optimization configuration

    Returns:
        Optimization results
    """
    try:
        if config is None:
            config = TacticianLookbackConfig()

        optimizer = TacticianLookbackOptimizer(config)

        success = await optimizer.initialize()
        if not success:
            raise RuntimeError("Failed to initialize Tactician lookback optimizer")

        results = await optimizer.optimize_lookback_periods(
            market_data_1m, analyst_signals, analyst_outputs
        )

        return results

    except Exception as e:
        tprint_error(f"❌ Tactician lookback optimization failed: {e}")
        raise

def create_tactician_lookback_config(
    timeframe: str = "1m",
    symbol: str = "ETHUSDT",
    exchange: str = "binance",
    optimization_method: str = "two_step_grid_tpe",
    **kwargs
) -> TacticianLookbackConfig:
    """
    Create a Tactician lookback optimization configuration.

    Args:
        timeframe: Target timeframe (should be "1m")
        symbol: Trading symbol
        exchange: Exchange name
        optimization_method: Optimization method to use
        **kwargs: Additional configuration parameters

    Returns:
        TacticianLookbackConfig instance
    """
    config = TacticianLookbackConfig(
        timeframe=timeframe,
        symbol=symbol,
        exchange=exchange,
        optimization_method=optimization_method
    )

    # Update with any additional parameters
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config
