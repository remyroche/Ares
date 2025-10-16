"""
Tactician Pre-ML Training Orchestrator

This orchestrator handles the complete pre-ML training pipeline for Tactician models:

1. Optimize features lookback periods
2. Generate interaction and cross-timeframe features (uses interactive_feature_generation)
3. Apply multi-horizon profit labeling
4. Select final features
5. Train Tactician models with Analyst predictions as input features
6. Wire to existing models_training sub_pipeline

The orchestrator calls the PRE_TRAINING pipeline (same as Analyst) on 15m timeframe data,
with Analyst predictions included as additional features. No filtering or confidence
thresholds are applied.
"""

import asyncio
import copy
import logging
import numpy as np
import pandas as pd
import time
import traceback
from typing import Dict, List, Optional, Tuple, Any, Union, TYPE_CHECKING
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

# Enhanced imports with comprehensive error handling
try:
    from src.utils.logger import system_logger
    from src.utils.tprint import (
        tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
        tprint_debug, tprint_performance, tprint_timer
    )
    TPRINT_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL: Failed to import logging utilities: {e}")
    raise

# Import common utilities
try:
    from src.utils.common_operations import (
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        cleanup_m1_optimizers, integrate_with_m1_optimizers,
        validate_dataframe, validate_dataframe_columns, safe_merge_dataframes,
        safe_json_load, safe_json_dump, ensure_directory
    )
    from src.utils.math_validation import (
        safe_divide, validate_finite, validate_positive, validate_range
    )
    from src.utils.common_utilities import (
        validate_dataframe_columns as validate_df_columns,
        analyze_nan_values_detailed,
        safe_dataframe_operation
    )
    COMMON_OPS_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL: Common operations not available: {e}")
    COMMON_OPS_AVAILABLE = False

# Import data validation utilities
try:
    from src.utils.data.validation.validators import CrossStepValidator, DataValidator
    DATA_VALIDATION_AVAILABLE = True
except ImportError as e:
    tprint_warning(f"⚠️ Data validation utilities not available: {e}")
    DATA_VALIDATION_AVAILABLE = False
    CrossStepValidator = None
    DataValidator = None

# Import lookahead protection
try:
    from src.utils.ml_common.utils.lookahead_protection import LookaheadProtection, LookaheadBiasError
    LOOKAHEAD_PROTECTION_AVAILABLE = True
except ImportError as e:
    tprint_warning(f"⚠️ Lookahead protection not available: {e}")
    LOOKAHEAD_PROTECTION_AVAILABLE = False
    LookaheadProtection = None
    LookaheadBiasError = None

# Import data quality utilities
try:
    from src.utils.data.quality.data_quality import (
        UnifiedDataQualityValidator,
        QualityThresholds,
        UnifiedMemoryConfig
    )
    DATA_QUALITY_AVAILABLE = True
except ImportError as e:
    tprint_warning(f"⚠️ Data quality utilities not available: {e}")
    DATA_QUALITY_AVAILABLE = False
    UnifiedDataQualityValidator = None
    QualityThresholds = None

# Import pre-training subpipeline
try:
    from src.training.steps.pre_training.sub_pipeline import SubPipeline, SubPipelineConfig
    PRE_TRAINING_AVAILABLE = True
except ImportError as e:
    tprint_warning(f"⚠️ Pre-training subpipeline not available: {e}")
    PRE_TRAINING_AVAILABLE = False
    SubPipeline = None
    SubPipelineConfig = None

# Import optimization utilities
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
    TPE_OPTIMIZER_AVAILABLE = True
except ImportError as e:
    tprint_warning(f"⚠️ TPE optimizer not available: {e}")
    TPE_OPTIMIZER_AVAILABLE = False
    BayesianTPEOptimizer = None

# Import feature processing components - REMOVED (feature_lookback_optimization no longer used)
# try:
#     from src.training.steps.pre_training.feature_lookback_optimization import (
#         FeatureLookbackOptimizationComponent as _FeatureLookbackOptimizationComponent
#     )
#     _ = _FeatureLookbackOptimizationComponent
#     FEATURE_OPTIMIZATION_AVAILABLE = True
# except ImportError as e:
#     FEATURE_OPTIMIZATION_AVAILABLE = False
#     tprint_warning(f"⚠️ Feature lookback optimization not available: {e}")
FEATURE_OPTIMIZATION_AVAILABLE = False

# try:
#     from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation import (
#         InteractiveFeatureGenerationComponent
#     )
#     INTERACTIVE_GENERATION_AVAILABLE = True
# except ImportError as e:
#     INTERACTIVE_GENERATION_AVAILABLE = False
#     tprint_warning(f"⚠️ Interactive feature generation not available: {e}")
INTERACTIVE_GENERATION_AVAILABLE = False

try:
    from src.training.steps.pre_training.multi_horizon_profit_labeler import (
        MultiHorizonProfitLabeler, MultiHorizonConfig
    )
    HORIZON_LABELING_AVAILABLE = True
except ImportError as e:
    HORIZON_LABELING_AVAILABLE = False
    tprint_warning(f"⚠️ Multi-horizon profit labeling not available: {e}")

try:
    from src.training.steps.pre_training.final_feature_selection_step import (
        FinalFeatureSelectionStep
    )
    FEATURE_SELECTION_AVAILABLE = True
except ImportError as e:
    FEATURE_SELECTION_AVAILABLE = False
    tprint_warning(f"⚠️ Final feature selection not available: {e}")
# except ImportError as e:
#     FEATURE_SELECTION_AVAILABLE = False
#     tprint_warning(f"⚠️ Final feature selection not available: {e}")

from src.training.steps.pre_training.components import ComponentFactory, ComponentConfig

class SignalDirection(Enum):
    """Signal direction enumeration."""
    LONG = "long"
    SHORT = "short"
    COMBINED = "combined"

@dataclass
class OrchestratorConfig:
    """Configuration for Tactician pre-ML orchestrator."""
    # Timeframe configuration
    timeframe: str = "15m"  # Tactician uses 15m timeframe

    # Signal processing parameters
    min_analyst_confidence: float = 0.6  # Minimum confidence threshold for analyst signals
    confidence_threshold: float = 0.6  # Alias for min_analyst_confidence
    subsequent_minutes: int = 45  # Minutes to include after signal for training window

    # Direction configuration
    direction_mode: str = "long_only"  # Options: "both", "long_only", "short_only"
    separate_directional_features: bool = True  # Train separate models for long/short
    directional_feature_prefixes: Dict[str, str] = field(default_factory=lambda: {
        'long': 'long_',
        'short': 'short_',
        'combined': 'combined_'
    })

    # Feature processing parameters (for PRE_TRAINING pipeline)
    max_lookback_periods: int = 20
    max_interaction_features: int = 100
    max_polynomial_features: int = 50
    max_cross_timeframe_features: int = 15
    max_features: int = 200  # Maximum features after selection

    # Horizon labeling parameters
    profit_targets: Dict[str, float] = field(default_factory=lambda: {
        'micro': 0.003,    # 0.3% (net: 0.22% after fees)
        'small': 0.005,    # 0.5% (net: 0.42% after fees)
        'medium': 0.007,   # 0.7% (net: 0.62% after fees)
        'good': 0.010      # 1.0% (net: 0.92% after fees)
    })

    time_horizons: Dict[str, int] = field(default_factory=lambda: {
        'immediate': 2,    # 10 minutes (2 x 5min = 10min for 15m timeframe)
        'short': 4         # 20 minutes (4 x 5min = 20min for 15m timeframe)
    })

    # Feature selection parameters
    initial_features: int = 120
    stage_1_target: int = 100
    stage_2_target: int = 80
    stage_3_target: int = 60

    # Hardware optimization
    enable_parallel_processing: bool = True
    enable_gpu_acceleration: bool = True
    memory_limit_gb: float = 8.0

    # Output configuration
    output_directory: str = "generated/tactician_pre_ml"
    save_intermediate_results: bool = True

    # Execution control (uses PRE_TRAINING pipeline)
    enable_feature_optimization: bool = True
    enable_interactive_generation: bool = True  # Interactive feature generation (replaces PID)
    enable_horizon_labeling: bool = True
    enable_feature_selection: bool = True

    # Data quality and validation
    enable_data_validation: bool = True
    enable_lookahead_protection: bool = True
    enable_quality_gates: bool = True
    min_quality_score: float = 0.7

    # Memory and performance
    enable_memory_optimization: bool = True
    batch_size: int = 10000  # Process data in batches for memory efficiency
    enable_progress_tracking: bool = True

@dataclass
class FeatureSelectionConfig:
    """Configuration for feature selection in Tactician pre-ML pipeline."""
    initial_features: int
    stage_1_target: int
    stage_2_target: int
    stage_3_target: int
    direction_mode: str
    separate_directional_features: bool
    directional_feature_prefixes: Dict[str, str]
    output_directory: str
    save_analysis: bool
    verbose: bool

@dataclass
class OrchestratorResult:
    """Result of Tactician pre-ML orchestration."""
    # Signal separation results
    tagged_market_data: Optional[pd.DataFrame] = None
    long_signals: Optional[pd.DataFrame] = None
    short_signals: Optional[pd.DataFrame] = None
    combined_signals: Optional[pd.DataFrame] = None

    # Feature processing results
    optimized_lookbacks: Dict[str, int] = field(default_factory=dict)
    long_optimized_lookbacks: Dict[str, int] = field(default_factory=dict)
    short_optimized_lookbacks: Dict[str, int] = field(default_factory=dict)

    # Interactive feature results
    long_pid_features: Optional[Any] = None  # InteractiveFeatureResult
    short_pid_features: Optional[Any] = None
    long_interactive_features: Optional[Any] = None  # Alias for long_pid_features
    short_interactive_features: Optional[Any] = None  # Alias for short_pid_features

    # Targets
    labeled_targets: Dict[str, np.ndarray] = field(default_factory=dict)
    long_labeled_targets: Dict[str, np.ndarray] = field(default_factory=dict)
    short_labeled_targets: Dict[str, np.ndarray] = field(default_factory=dict)
    long_targets: Dict[str, np.ndarray] = field(default_factory=dict)
    short_targets: Dict[str, np.ndarray] = field(default_factory=dict)

    # Selected features
    selected_features: List[str] = field(default_factory=list)
    long_selected_features: List[str] = field(default_factory=list)
    short_selected_features: List[str] = field(default_factory=list)
    final_features: Optional[pd.DataFrame] = None
    feature_names: List[str] = field(default_factory=list)

    # Training data
    long_training_data: Optional[pd.DataFrame] = None
    short_training_data: Optional[pd.DataFrame] = None

    # Sample counts
    total_samples: int = 0
    total_long_samples: int = 0
    total_short_samples: int = 0
    analyst_signals_count: int = 0
    long_signals_count: int = 0
    short_signals_count: int = 0

    # Quality metrics
    data_quality_score: float = 0.0
    long_data_quality_score: float = 0.0
    short_data_quality_score: float = 0.0
    long_missing_values_ratio: float = 0.0
    short_missing_values_ratio: float = 0.0
    long_outlier_ratio: float = 0.0
    short_outlier_ratio: float = 0.0

    # Confidence metrics
    average_long_confidence: float = 0.0
    average_short_confidence: float = 0.0

    # Status tracking
    signal_separation_completed: bool = False
    feature_optimization_completed: bool = False
    pid_generation_completed: bool = False
    interactive_generation_completed: bool = False
    horizon_labeling_completed: bool = False
    feature_selection_completed: bool = False
    success: bool = False

    # Execution metrics
    execution_time: float = 0.0
    optimization_time: float = 0.0
    generation_time: float = 0.0
    labeling_time: float = 0.0
    selection_time: float = 0.0
    preparation_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0

    # Feature generation status
    feature_generation_status: str = "pending"
    error_message: Optional[str] = None

    # Configuration flags
    lookback_optimization_enabled: bool = False
    pid_feature_generation_enabled: bool = False
    interactive_feature_generation_enabled: bool = False
    horizon_labeling_enabled: bool = False
    feature_selection_enabled: bool = False
    intermediate_results_saved: bool = False

    # Comprehensive reporting
    comprehensive_report: Dict[str, Any] = field(default_factory=dict)

    # Evaluation metrics (populated later in training)
    long_training_accuracy: float = 0.0
    short_training_accuracy: float = 0.0
    long_validation_accuracy: float = 0.0
    short_validation_accuracy: float = 0.0
    long_f1_score: float = 0.0
    short_f1_score: float = 0.0
    long_precision: float = 0.0
    short_precision: float = 0.0
    long_recall: float = 0.0
    short_recall: float = 0.0
    long_roc_auc: float = 0.0
    short_roc_auc: float = 0.0
    long_sharpe_ratio: float = 0.0
    short_sharpe_ratio: float = 0.0
    long_max_drawdown: float = 0.0
    short_max_drawdown: float = 0.0
    long_total_trades: int = 0
    short_total_trades: int = 0
    long_avg_trades_per_month: float = 0.0
    short_avg_trades_per_month: float = 0.0
    long_total_pnl: float = 0.0
    short_total_pnl: float = 0.0
    long_monthly_pnl: Dict[str, float] = field(default_factory=dict)
    short_monthly_pnl: Dict[str, float] = field(default_factory=dict)
    evaluation_completed: bool = False

class TacticianPreMLOrchestrator:
    """
    Tactician Pre-ML Training Orchestrator.

    Orchestrates the complete pre-ML training pipeline for Tactician models by:
    1. Loading 15m market data
    2. Optimizing feature lookback periods
    3. Generating interaction and cross-timeframe features
    4. Applying multi-horizon profit labeling
    5. Selecting final features
    6. Including Analyst predictions as additional features

    Uses the same PRE_TRAINING pipeline as Analyst, but on 15m timeframe.
    No filtering or confidence thresholds applied.
    """

    COMPONENT_FACTORY_KEYS: Dict[str, str] = {
        'tactician_labeling': 'tactician-labeler',
        'data_validation': 'feature_generation_data_validation_step',
        'labeling_integration': 'feature_generation_labeling_integration_step',
        'feature_generation': 'feature_generation_feature_generation_step',
        'feature_selection': 'feature_generation_feature_selection_step',
        'period_lookback_optimization': 'feature_generation_period_lookback_optimization_step',
        'interaction_generation': 'feature_generation_interaction_generation_step',
        'vectorization': 'feature_generation_vectorization_step',
        'final_validation': 'feature_generation_final_validation_step',
    }

    COMPONENT_HINTS: Dict[str, str] = {
        'tactician_labeling': (
            "Ensure the 'tactician-labeler' component is available in the PRE_TRAINING pipeline."
        ),
        'data_validation': (
            "Ensure the 'feature_generation_data_validation_step' component is available in the PRE_TRAINING pipeline."
        ),
        'labeling_integration': (
            "Ensure the 'feature_generation_labeling_integration_step' component is available in the PRE_TRAINING pipeline."
        ),
        'feature_generation': (
            "Ensure the 'feature_generation_feature_generation_step' component is available in the PRE_TRAINING pipeline."
        ),
        'feature_selection': (
            "Ensure the 'feature_generation_feature_selection_step' component is available in the PRE_TRAINING pipeline."
        ),
        'period_lookback_optimization': (
            "Ensure the 'feature_generation_period_lookback_optimization_step' component is available in the PRE_TRAINING pipeline."
        ),
        'interaction_generation': (
            "Ensure the 'feature_generation_interaction_generation_step' component is available in the PRE_TRAINING pipeline."
        ),
        'vectorization': (
            "Ensure the 'feature_generation_vectorization_step' component is available in the PRE_TRAINING pipeline."
        ),
        'final_validation': (
            "Ensure the 'feature_generation_final_validation_step' component is available in the PRE_TRAINING pipeline."
        ),
    }

    COMPONENT_CONFIG_MAPPING: Dict[str, Dict[str, str]] = {
        'tactician_labeling': {
            'profit_targets': 'profit_targets',
            'time_horizons': 'time_horizons',
            'direction_mode': 'direction_mode',
            'separate_directional_features': 'separate_directional_features',
            'directional_feature_prefixes': 'directional_feature_prefixes',
        },
        'data_validation': {
            'validate_input_data': 'validate_input_data',
            'strict_data_validation': 'strict_data_validation',
            'enable_domain_checks': 'enable_domain_checks',
            'correlation_threshold': 'correlation_threshold',
            'stability_threshold': 'stability_threshold',
        },
        'labeling_integration': {
            'enable_labeling_optimization': 'enable_labeling_optimization',
            'labeling_quality_threshold': 'labeling_quality_threshold',
            'direction_mode': 'direction_mode',
            'separate_directional_features': 'separate_directional_features',
        },
        'feature_generation': {
            'enable_period_optimization': 'enable_period_optimization',
            'enable_feature_lookback_optimization': 'enable_feature_lookback_optimization',
            'enable_interaction_generation': 'enable_interaction_generation',
            'enable_htf_interactions': 'enable_htf_interactions',
            'max_interactions': 'max_interactions',
            'min_utility_threshold': 'min_utility_threshold',
            'max_correlation_threshold': 'max_correlation_threshold',
        },
        'feature_selection': {
            'selection_strategy': 'selection_strategy',
            'max_features': 'max_features',
            'min_features': 'min_features',
            'max_feature_cost': 'max_feature_cost',
            'enable_nested_cv': 'enable_nested_cv',
            'enable_direction_optimization': 'enable_direction_optimization',
            'enable_bayesian_optimization': 'enable_bayesian_optimization',
            'output_directory': 'output_directory',
            'save_analysis': 'save_intermediate_results',
        },
        'period_lookback_optimization': {
            'min_period': 'min_period',
            'max_period': 'max_period',
            'period_step': 'period_step',
            'min_lookback': 'min_lookback',
            'max_lookback': 'max_lookback',
            'step_size': 'step_size',
            'enable_parallel': 'enable_parallel',
            'max_workers': 'max_workers',
            'optimization_strategy': 'optimization_strategy',
        },
        'interaction_generation': {
            'max_interactions': 'max_interactions',
            'min_utility_threshold': 'min_utility_threshold',
            'max_correlation_threshold': 'max_correlation_threshold',
            'min_interaction_significance': 'min_interaction_significance',
            'min_interaction_stability': 'min_interaction_stability',
            'enable_batch_processing': 'enable_batch_processing',
            'batch_size': 'batch_size',
            'enable_htf_interactions': 'enable_htf_interactions',
            'htf_interaction_ratio': 'htf_interaction_ratio',
        },
        'vectorization': {
            'enable_vectorbt': 'enable_vectorbt',
            'vectorbt_strategy': 'vectorbt_strategy',
            'enable_gpu': 'enable_gpu',
            'enable_parallel': 'enable_parallel',
            'memory_efficient': 'memory_efficient',
            'max_memory_gb': 'max_memory_gb',
            'chunk_size': 'chunk_size',
            'enable_caching': 'enable_caching',
            'cache_size': 'cache_size',
        },
        'final_validation': {
            'validate_input_data': 'validate_input_data',
            'strict_data_validation': 'strict_data_validation',
            'enable_domain_checks': 'enable_domain_checks',
            'correlation_threshold': 'correlation_threshold',
            'stability_threshold': 'stability_threshold',
            'save_intermediate_results': 'save_intermediate_results',
        },
    }

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """Initialize the Tactician pre-ML orchestrator."""
        try:
            tprint_info("🚀 Initializing TacticianPreMLOrchestrator...")

            self.config = config or OrchestratorConfig()
            self.logger = system_logger.getChild('TacticianPreMLOrchestrator')

            # Initialize hardware optimizers (consolidated)
            self._initialize_hardware_optimizers_consolidated()

            # Initialize validators (consolidated)
            self._initialize_validators_consolidated()

            # Initialize feature processing components
            self._initialize_feature_processors()

            # Track operation count for memory management
            self._operation_count = 0

            tprint_success("✅ TacticianPreMLOrchestrator initialized successfully")
            self._log_initialization_summary()

        except Exception as e:
            tprint_error(f"❌ Failed to initialize TacticianPreMLOrchestrator: {e}")
            raise

    def _initialize_hardware_optimizers_consolidated(self) -> None:
        """Initialize hardware optimizers (consolidated method)."""
        if COMMON_OPS_AVAILABLE and self.config.enable_memory_optimization:
            try:
                self.gpu_manager = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer()
                self.cpu_optimizer = get_m1_cpu_optimizer()

                available = sum(1 for x in [self.gpu_manager, self.memory_optimizer, self.cpu_optimizer] if x is not None)
                tprint_success(f"✅ Hardware optimizers: {available}/3 available")
            except Exception as e:
                self.gpu_manager = None
                self.memory_optimizer = None
                self.cpu_optimizer = None
                tprint_warning(f"⚠️ Hardware optimization failed: {e}")
        else:
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
            if self.config.enable_memory_optimization:
                tprint_warning("⚠️ Memory optimization requested but utilities not available")

    def _initialize_validators_consolidated(self) -> None:
        """Initialize validators (consolidated method)."""
        # Data validators
        if DATA_VALIDATION_AVAILABLE and self.config.enable_data_validation:
            try:
                self.cross_step_validator = CrossStepValidator()
                self.data_validator = DataValidator(logger=self.logger)
                tprint_success("✅ Data validators initialized")
            except Exception as e:
                self.cross_step_validator = None
                self.data_validator = None
                tprint_warning(f"⚠️ Data validator init failed: {e}")
        else:
            self.cross_step_validator = None
            self.data_validator = None

        # Lookahead protection
        if LOOKAHEAD_PROTECTION_AVAILABLE and self.config.enable_lookahead_protection:
            try:
                lookahead_config = {
                    'strict_mode': True,
                    'enable_temporal_validation': True,
                    'enable_feature_validation': True
                }
                self.lookahead_protection = LookaheadProtection(config=lookahead_config)
                tprint_success("✅ Lookahead protection initialized")
            except Exception as e:
                self.lookahead_protection = None
                tprint_warning(f"⚠️ Lookahead protection init failed: {e}")
        else:
            self.lookahead_protection = None

        # Data quality validator
        if DATA_QUALITY_AVAILABLE and self.config.enable_quality_gates:
            try:
                quality_thresholds = QualityThresholds(
                    max_nan_ratio=0.1,
                    max_infinite_count=0,
                    min_unique_values=2,
                    max_constant_ratio=0.95
                )
                self.quality_validator = UnifiedDataQualityValidator(
                    logger=self.logger,
                    thresholds=quality_thresholds
                )
                tprint_success("✅ Data quality validator initialized")
            except Exception as e:
                self.quality_validator = None
                tprint_warning(f"⚠️ Quality validator init failed: {e}")
        else:
            self.quality_validator = None

    def _log_initialization_summary(self) -> None:
        """Log comprehensive initialization summary."""
        tprint_info(f"📊 Configuration:")
        tprint_info(f"  • Min analyst confidence: {self.config.min_analyst_confidence}")
        tprint_info(f"  • Subsequent minutes: {self.config.subsequent_minutes}")
        tprint_info(f"  • Direction mode: {self.config.direction_mode}")
        tprint_info(f"  • Separate directional features: {self.config.separate_directional_features}")
        tprint_info(f"  • Max features: {self.config.max_features}")
        tprint_info(f"  • Data validation: {self.config.enable_data_validation}")
        tprint_info(f"  • Lookahead protection: {self.config.enable_lookahead_protection}")
        tprint_info(f"  • Quality gates: {self.config.enable_quality_gates}")
        tprint_info(f"  • Output directory: {self.config.output_directory}")

    def _init_factory_component(
        self,
        alias: str,
        enabled: bool,
        available: bool,
        display_name: str
    ) -> Optional[Any]:
        """
        Consolidated helper for initializing factory components.

        Args:
            alias: Component alias in COMPONENT_FACTORY_KEYS
            enabled: Whether component is enabled in config
            available: Whether component dependencies are available
            display_name: Display name for logging

        Returns:
            Initialized component or None
        """
        if not enabled:
            return None

        if not available:
            tprint_warning(f"⚠️ {display_name} requested but not available")
            return None

        if not self.factory_component_status.get(alias, False):
            self._log_factory_unavailable(alias)
            return None

        try:
            component_key = self.COMPONENT_FACTORY_KEYS.get(alias)
            component_config = self.factory_component_configs.get(alias)
            component = ComponentFactory.create_component(component_key, component_config)
            tprint_success(f"✅ {display_name} initialized via ComponentFactory")
            return component
        except Exception as exc:
            self._log_factory_error(alias, exc)
            return None

    def _initialize_feature_processors(self):
        """Initialize feature processing components."""
        self.factory_component_status: Dict[str, bool] = self._evaluate_factory_components()
        self.factory_component_configs: Dict[str, ComponentConfig] = {
            alias: self._build_factory_component_config(alias)
            for alias in self.COMPONENT_FACTORY_KEYS
        }

        # Initialize components using consolidated helper
        # self.feature_optimizer = self._init_factory_component(
        #     'feature_optimization',
        #     self.config.enable_feature_optimization,
        #     FEATURE_OPTIMIZATION_AVAILABLE,  # Always False now
        #     "Feature lookback optimization"
        # )
        self.feature_optimizer = None

        self.interactive_generator = self._init_factory_component(
            'interactive_generation',
            self.config.enable_interactive_generation,
            INTERACTIVE_GENERATION_AVAILABLE,
            "Interactive feature generation"
        )

        # Multi-horizon profit labeling
        if not self.config.enable_horizon_labeling:
            self.horizon_labeler = None
        elif not HORIZON_LABELING_AVAILABLE:
            self.horizon_labeler = None
            tprint_warning("⚠️ Horizon labeling requested but not available")
        else:
            if not self.factory_component_status.get('horizon_labeling', False):
                self._log_factory_unavailable('horizon_labeling')
            try:
                labeler_config = MultiHorizonConfig(
                    profit_targets=self.config.profit_targets,
                    time_horizons=self.config.time_horizons,
                    enable_quality_scoring=True,
                    leverage_aware=True,
                    direction_mode=self.config.direction_mode,
                    separate_directional_targets=self.config.separate_directional_features,
                    directional_target_prefixes=self.config.directional_feature_prefixes
                )
                self.horizon_labeler = MultiHorizonProfitLabeler(labeler_config)
                tprint_success(f"✅ Multi-horizon profit labeler initialized (mode: {self.config.direction_mode})")
            except Exception as e:
                tprint_error(f"❌ Failed to initialize horizon labeler: {e}")
                self.horizon_labeler = None

        # Final feature selection
        if not self.config.enable_feature_selection:
            self.feature_selector = None
        # elif not FEATURE_SELECTION_AVAILABLE:  # Always False now
        #     self.feature_selector = None
        #     tprint_warning("⚠️ Feature selection requested but not available")
        else:
            if not self.factory_component_status.get('feature_selection', False):
                self._log_factory_unavailable('feature_selection')
            try:
                selection_config = FeatureSelectionConfig(
                    initial_features=self.config.initial_features,
                    stage_1_target=self.config.stage_1_target,
                    stage_2_target=self.config.stage_2_target,
                    stage_3_target=self.config.stage_3_target,
                    direction_mode=self.config.direction_mode,
                    separate_directional_features=self.config.separate_directional_features,
                    directional_feature_prefixes=self.config.directional_feature_prefixes,
                    output_directory=self.config.output_directory,
                    save_analysis=self.config.save_intermediate_results,
                    verbose=True
                )
                self.feature_selector = FinalFeatureSelectionStep(selection_config)
                tprint_success(f"✅ Final feature selector initialized (mode: {self.config.direction_mode})")
            except Exception as e:
                tprint_error(f"❌ Failed to initialize feature selector: {e}")
                self.feature_selector = None

    def _evaluate_factory_components(self) -> Dict[str, bool]:
        """Check availability of orchestrator components in the ComponentFactory."""
        status: Dict[str, bool] = {}
        for alias, component_key in self.COMPONENT_FACTORY_KEYS.items():
            try:
                status[alias] = ComponentFactory.is_component_available(component_key)
            except Exception as exc:  # pragma: no cover - defensive logging
                status[alias] = False
                tprint_warning(
                    f"⚠️ Unable to query ComponentFactory availability for '{component_key}': {exc}"
                )
        return status

    def _build_factory_component_config(self, alias: str) -> ComponentConfig:
        """Build a ComponentConfig based on orchestrator settings for factory components."""
        component_key = self.COMPONENT_FACTORY_KEYS.get(alias, alias)
        custom_params: Dict[str, Any] = {
            'component_alias': alias,
            'factory_component_key': component_key,
            'source': 'tactician_pre_ml_orchestrator',
        }

        mapping = self.COMPONENT_CONFIG_MAPPING.get(alias, {})
        for component_param, config_attr in mapping.items():
            if not hasattr(self.config, config_attr):
                continue
            value = getattr(self.config, config_attr)
            if isinstance(value, (dict, list)):
                custom_params[component_param] = copy.deepcopy(value)
            else:
                custom_params[component_param] = value

        return ComponentConfig(custom_params=custom_params)

    def _log_factory_unavailable(self, alias: str) -> None:
        """Log a standardized warning when a factory component is not registered."""
        component_key = self.COMPONENT_FACTORY_KEYS.get(alias, alias)
        hint = self.COMPONENT_HINTS.get(alias)
        alias_label = alias.replace('_', ' ')
        message = (
            f"⚠️ ComponentFactory does not have '{component_key}' registered for {alias_label}."
        )
        if hint:
            message = f"{message} Hint: {hint}"
        tprint_warning(message)

    def _log_factory_error(self, alias: str, exc: Exception) -> None:
        """Log a standardized error when a factory-backed initialization fails."""
        component_key = self.COMPONENT_FACTORY_KEYS.get(alias, alias)
        hint = self.COMPONENT_HINTS.get(alias)
        alias_label = alias.replace('_', ' ')
        message = (
            f"❌ Failed to initialize '{component_key}' via ComponentFactory for {alias_label}: {exc}"
        )
        if hint:
            message = f"{message} Hint: {hint}"
        tprint_error(message)

    def _validate_input_data(
        self,
        analyst_signals: pd.DataFrame,
        market_data: pd.DataFrame,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of input data using available utilities.

        Args:
            analyst_signals: DataFrame with analyst signals
            market_data: Raw market data
            feature_names: List of feature names

        Returns:
            Dictionary with validation results
        """
        tprint_info("🔍 Validating input data...")
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'quality_score': 1.0
        }

        try:
            # Basic DataFrame validation
            if not validate_dataframe(analyst_signals):
                validation_result['valid'] = False
                validation_result['errors'].append("Invalid analyst signals DataFrame")
                return validation_result

            if not validate_dataframe(market_data):
                validation_result['valid'] = False
                validation_result['errors'].append("Invalid market data DataFrame")
                return validation_result

            # Check required columns
            required_signal_cols = ['timestamp']
            if not validate_dataframe_columns(analyst_signals, required_signal_cols):
                validation_result['valid'] = False
                validation_result['errors'].append(f"Missing required columns in analyst signals: {required_signal_cols}")

            required_market_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not validate_dataframe_columns(market_data, required_market_cols):
                validation_result['valid'] = False
                validation_result['errors'].append(f"Missing required columns in market data: {required_market_cols}")

            # Validate timestamps are sorted
            if not analyst_signals['timestamp'].is_monotonic_increasing:
                validation_result['warnings'].append("Analyst signals timestamps not sorted")

            if not market_data['timestamp'].is_monotonic_increasing:
                validation_result['warnings'].append("Market data timestamps not sorted")

            # Check for NaN values using utility
            if COMMON_OPS_AVAILABLE:
                nan_analysis = analyze_nan_values_detailed(market_data, feature_names if feature_names else None)
                if nan_analysis['total_nan_percentage'] > 10.0:
                    validation_result['warnings'].append(
                        f"High NaN percentage in market data: {nan_analysis['total_nan_percentage']:.2f}%"
                    )
                    validation_result['quality_score'] *= 0.9

            # Use data validator if available
            if self.data_validator:
                validator_result = self.data_validator.validate_input_data(
                    market_data,
                    labels=None
                )
                if not validator_result['is_valid']:
                    validation_result['valid'] = False
                    validation_result['errors'].extend(validator_result['errors'])
                validation_result['warnings'].extend(validator_result['warnings'])

            # Use quality validator if available and enabled
            if self.quality_validator and self.config.enable_quality_gates:
                tprint_info("📊 Running comprehensive quality validation...")
                quality_result = self.quality_validator.validate_data_quality(
                    market_data,
                    context="tactician_pre_ml_input"
                )
                validation_result['quality_score'] = quality_result.get('overall_score', 1.0)

                if validation_result['quality_score'] < self.config.min_quality_score:
                    validation_result['warnings'].append(
                        f"Data quality score {validation_result['quality_score']:.3f} below threshold {self.config.min_quality_score}"
                    )

            # Lookahead protection - validate temporal ordering
            if self.lookahead_protection:
                tprint_info("🛡️ Validating temporal ordering for lookahead protection...")
                try:
                    temporal_valid = self.lookahead_protection.validate_temporal_order(
                        market_data['timestamp'].values
                    )
                    if not temporal_valid:
                        validation_result['warnings'].append("Temporal ordering validation failed")
                except Exception as e:
                    tprint_warning(f"⚠️ Lookahead validation error: {e}")

            # Check data size
            min_samples = 100
            if len(market_data) < min_samples:
                validation_result['warnings'].append(
                    f"Low sample count: {len(market_data)} < {min_samples}"
                )

            if len(analyst_signals) < 10:
                validation_result['warnings'].append(
                    f"Very few analyst signals: {len(analyst_signals)}"
                )

            # Log validation summary
            if validation_result['valid']:
                tprint_success(f"✅ Input data validation passed (quality: {validation_result['quality_score']:.3f})")
                if validation_result['warnings']:
                    tprint_warning(f"⚠️ {len(validation_result['warnings'])} warnings found:")
                    for warning in validation_result['warnings'][:5]:  # Show first 5
                        tprint_warning(f"  • {warning}")
            else:
                tprint_error("❌ Input data validation failed:")
                for error in validation_result['errors']:
                    tprint_error(f"  • {error}")

        except Exception as e:
            tprint_error(f"❌ Error during input validation: {e}")
            validation_result['valid'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")

        return validation_result

    def _check_memory_and_cleanup(self) -> None:
        """Check memory usage and perform cleanup if needed."""
        self._operation_count += 1

        if self.memory_optimizer and self._operation_count % 50 == 0:
            try:
                current_usage = self.memory_optimizer.get_current_memory_usage()
                tprint_debug(f"📊 Current memory usage: {current_usage:.2f} GB")

                if current_usage > self.config.memory_limit_gb * 0.8:
                    tprint_warning(f"⚠️ High memory usage: {current_usage:.2f} GB, performing cleanup...")
                    self.memory_optimizer.cleanup_memory()
            except Exception as e:
                tprint_debug(f"Memory check error: {e}")

    async def orchestrate_pre_ml_training(
        self,
        analyst_signals: pd.DataFrame,
        market_data: pd.DataFrame,
        feature_names: List[str],
        **kwargs
    ) -> OrchestratorResult:
        """
        Orchestrate the complete pre-ML training pipeline for Tactician using the unified pipeline.

        Args:
            analyst_signals: DataFrame with Analyst signals and confidence scores
            market_data: Raw market data for feature generation
            feature_names: List of base feature names
            **kwargs: Additional parameters

        Returns:
            OrchestratorResult with processed data for both long and short training
        """
        start_time = tprint_timer()
        tprint_info("=" * 80)
        tprint_info(f"🚀 Starting Tactician pre-ML training orchestration (mode: {self.config.direction_mode})")
        tprint_info("=" * 80)

        result = OrchestratorResult()
        result.feature_generation_status = "in_progress"
        result.feature_names = feature_names

        try:
            if not PRE_TRAINING_AVAILABLE:
                raise RuntimeError("Pre-training subpipeline not available")

            # Step 1: Prepare data for unified pipeline
            tprint_info("📊 Step 1: Preparing data for unified pipeline...")
            
            # Merge analyst signals with market data
            prepared_data = self._prepare_data_for_pipeline(analyst_signals, market_data, feature_names)
            
            # Store analyst signals count
            result.analyst_signals_count = len(analyst_signals)
            result.tagged_market_data = prepared_data
            
            tprint_success(f"✅ Data preparation completed: {len(prepared_data)} samples")

            # Step 2: Configure and run unified pipeline
            tprint_info("🔧 Step 2: Configuring unified pipeline...")
            
            # Create pipeline configuration
            pipeline_config = self._create_pipeline_config()
            
            # Initialize the subpipeline
            subpipeline = SubPipeline(config=pipeline_config)
            
            tprint_success("✅ Pipeline configuration completed")

            # Step 3: Execute unified pipeline
            tprint_info("🚀 Step 3: Executing unified pipeline...")
            
            # Run the pipeline
            pipeline_result = await subpipeline.run_pipeline(
                data=prepared_data,
                feature_names=feature_names,
                timeframe='15m'  # Tactician uses 15m timeframe
            )
            
            if not pipeline_result.success:
                raise RuntimeError(f"Pipeline execution failed: {pipeline_result.error_message}")

            # Step 4: Extract results from pipeline
            tprint_info("📋 Step 4: Extracting pipeline results...")
            
            # Extract processed data and features
            result = self._extract_pipeline_results(result, pipeline_result, prepared_data)
            
            tprint_success(f"✅ Pipeline results extracted: {result.total_long_samples} long, {result.total_short_samples} short samples")

            # Step 5: Save intermediate results
            if self.config.save_intermediate_results:
                tprint_info("💾 Saving intermediate results...")
                await self._save_intermediate_results(result)
                tprint_success("✅ Intermediate results saved")

            # Update result status
            result.execution_time = tprint_timer(start_time)
            result.feature_generation_status = "completed"

            # Add comprehensive reporting
            result = self._add_comprehensive_reporting(result, start_time)

            return result

        except Exception as e:
            result.execution_time = tprint_timer(start_time)
            result.feature_generation_status = "failed"
            tprint_error(f"❌ Tactician pre-ML orchestration failed: {e}")
            tprint_error(f"Error details: {traceback.format_exc()}")
            raise

    def _prepare_data_for_pipeline(
        self,
        analyst_signals: pd.DataFrame,
        market_data: pd.DataFrame,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """Prepare data for the unified pipeline by merging analyst signals with market data."""
        try:
            tprint_info("🔗 Preparing data for unified pipeline...")
            
            # Ensure timestamp columns exist
            if 'timestamp' not in analyst_signals.columns:
                raise ValueError("Analyst signals must contain 'timestamp' column")
            if 'timestamp' not in market_data.columns:
                raise ValueError("Market data must contain 'timestamp' column")
            
            # Merge analyst signals with market data on timestamp
            prepared_data = market_data.copy()
            
            # Add analyst signal columns to market data
            analyst_columns = [col for col in analyst_signals.columns if col != 'timestamp']
            for col in analyst_columns:
                prepared_data[col] = 0.0  # Initialize with default values
            
            # Fill in actual analyst signal values where timestamps match
            for idx, row in analyst_signals.iterrows():
                signal_timestamp = row['timestamp']
                # Find matching timestamp in market data
                mask = prepared_data['timestamp'] == signal_timestamp
                if mask.any():
                    for col in analyst_columns:
                        prepared_data.loc[mask, col] = row[col]
            
            # Add feature names as columns if they don't exist
            for feature_name in feature_names:
                if feature_name not in prepared_data.columns:
                    prepared_data[feature_name] = 0.0
            
            tprint_success(f"✅ Data preparation completed: {len(prepared_data)} samples with {len(analyst_columns)} analyst signals")
            return prepared_data
            
        except Exception as e:
            tprint_error(f"❌ Data preparation failed: {e}")
            raise

    def _create_pipeline_config(self) -> SubPipelineConfig:
        """Create configuration for the unified pipeline."""
        try:
            tprint_info("⚙️ Creating pipeline configuration...")
            
            # Create base configuration
            config = SubPipelineConfig()
            
            # Configure enabled steps based on orchestrator config
            config.enabled_steps = []
            
            # Add tactician labeling step
            if self.config.enable_horizon_labeling:
                config.enabled_steps.append('tactician-labeler')
            
            # Add data validation step
            if self.config.enable_data_validation:
                config.enabled_steps.append('feature_generation_data_validation_step')
            
            # Add labeling integration step
            config.enabled_steps.append('feature_generation_labeling_integration_step')
            
            # Add feature generation step
            if self.config.enable_interactive_generation:
                config.enabled_steps.append('feature_generation_feature_generation_step')
            
            # Add feature selection step
            if self.config.enable_feature_selection:
                config.enabled_steps.append('feature_generation_feature_selection_step')
            
            # Add period and lookback optimization step
            if self.config.enable_feature_optimization:
                config.enabled_steps.append('feature_generation_period_lookback_optimization_step')
            
            # Add interaction generation step
            if self.config.enable_interactive_generation:
                config.enabled_steps.append('feature_generation_interaction_generation_step')
            
            # Add vectorization step
            config.enabled_steps.append('feature_generation_vectorization_step')
            
            # Add final validation step
            config.enabled_steps.append('feature_generation_final_validation_step')
            
            # Configure step parameters
            config.step_parameters = {
                'tactician-labeler': {
                    'profit_targets': getattr(self.config, 'profit_targets', [0.02, 0.05, 0.10]),
                    'time_horizons': getattr(self.config, 'time_horizons', [15, 30, 60]),
                    'direction_mode': self.config.direction_mode,
                    'separate_directional_features': self.config.separate_directional_features,
                    'directional_feature_prefixes': getattr(self.config, 'directional_feature_prefixes', ['long_', 'short_'])
                },
                'feature_generation_data_validation_step': {
                    'validate_input_data': self.config.enable_data_validation,
                    'strict_data_validation': getattr(self.config, 'strict_data_validation', True),
                    'enable_domain_checks': getattr(self.config, 'enable_domain_checks', True),
                    'correlation_threshold': getattr(self.config, 'correlation_threshold', 0.95),
                    'stability_threshold': getattr(self.config, 'stability_threshold', 0.8)
                },
                'feature_generation_feature_selection_step': {
                    'selection_strategy': getattr(self.config, 'selection_strategy', 'multi_objective'),
                    'max_features': self.config.max_features,
                    'min_features': getattr(self.config, 'min_features', 4),
                    'max_feature_cost': getattr(self.config, 'max_feature_cost', 100.0),
                    'enable_nested_cv': getattr(self.config, 'enable_nested_cv', True),
                    'enable_direction_optimization': getattr(self.config, 'enable_direction_optimization', True),
                    'enable_bayesian_optimization': getattr(self.config, 'enable_bayesian_optimization', True),
                    'output_directory': self.config.output_directory,
                    'save_analysis': self.config.save_intermediate_results
                }
            }
            
            tprint_success(f"✅ Pipeline configuration created with {len(config.enabled_steps)} steps")
            return config
            
        except Exception as e:
            tprint_error(f"❌ Pipeline configuration creation failed: {e}")
            raise

    def _extract_pipeline_results(
        self,
        result: OrchestratorResult,
        pipeline_result: Any,
        prepared_data: pd.DataFrame
    ) -> OrchestratorResult:
        """Extract results from the unified pipeline execution."""
        try:
            tprint_info("📋 Extracting pipeline results...")
            
            # Extract processed data
            if hasattr(pipeline_result, 'processed_data') and pipeline_result.processed_data is not None:
                processed_data = pipeline_result.processed_data
                
                # Separate into long and short based on direction mode
                if self.config.direction_mode == 'both':
                    # For 'both' mode, we need to separate the data
                    long_mask = processed_data.get('signal_direction', '') == 'long'
                    short_mask = processed_data.get('signal_direction', '') == 'short'
                    
                    result.long_training_data = processed_data[long_mask] if long_mask.any() else pd.DataFrame()
                    result.short_training_data = processed_data[short_mask] if short_mask.any() else pd.DataFrame()
                    
                    result.total_long_samples = len(result.long_training_data)
                    result.total_short_samples = len(result.short_training_data)
                    
                elif self.config.direction_mode == 'long_only':
                    result.long_training_data = processed_data
                    result.short_training_data = pd.DataFrame()
                    result.total_long_samples = len(processed_data)
                    result.total_short_samples = 0
                    
                elif self.config.direction_mode == 'short_only':
                    result.long_training_data = pd.DataFrame()
                    result.short_training_data = processed_data
                    result.total_long_samples = 0
                    result.total_short_samples = len(processed_data)
                
                # Set completion flags
                result.signal_separation_completed = True
                result.feature_optimization_completed = True
                result.pid_generation_completed = True
                result.horizon_labeling_completed = True
                result.feature_selection_completed = True
                
                # Extract feature information
                if hasattr(pipeline_result, 'selected_features'):
                    result.long_selected_features = pipeline_result.selected_features
                    result.short_selected_features = pipeline_result.selected_features
                
                # Extract quality scores
                if hasattr(pipeline_result, 'data_quality_score'):
                    result.data_quality_score = pipeline_result.data_quality_score
                    result.long_data_quality_score = pipeline_result.data_quality_score
                    result.short_data_quality_score = pipeline_result.data_quality_score
                
            else:
                # Fallback: use prepared data as training data
                result.long_training_data = prepared_data
                result.short_training_data = prepared_data
                result.total_long_samples = len(prepared_data)
                result.total_short_samples = len(prepared_data)
                
                # Set basic completion flags
                result.signal_separation_completed = True
                result.feature_optimization_completed = False
                result.pid_generation_completed = False
                result.horizon_labeling_completed = False
                result.feature_selection_completed = False
            
            tprint_success(f"✅ Pipeline results extracted successfully")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Pipeline result extraction failed: {e}")
            raise

    async def _separate_analyst_signals(
        self,
        analyst_signals: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Separate Analyst signals into long/short with confidence filtering and preserve full lookback periods."""
        try:
            tprint_info("🔍 Separating Analyst signals by direction and confidence...")

            # Validate input data
            if not validate_dataframe(analyst_signals):
                raise ValueError("Invalid analyst signals DataFrame")

            if not validate_dataframe(market_data):
                raise ValueError("Invalid market data DataFrame")

            # Ensure timestamp columns exist
            if 'timestamp' not in analyst_signals.columns:
                raise ValueError("Analyst signals must contain 'timestamp' column")

            if 'timestamp' not in market_data.columns:
                raise ValueError("Market data must contain 'timestamp' column")

            # Extract signal columns - assuming common naming patterns
            signal_columns = [col for col in analyst_signals.columns
                            if any(keyword in col.lower() for keyword in
                                 ['signal', 'prediction', 'direction', 'long', 'short'])]

            confidence_columns = [col for col in analyst_signals.columns
                                if 'confidence' in col.lower() or 'prob' in col.lower()]

            tprint_debug(f"Found signal columns: {signal_columns}")
            tprint_debug(f"Found confidence columns: {confidence_columns}")

            if not confidence_columns:
                tprint_warning("⚠️ No confidence columns found - using all signals")
                confidence_threshold = 0.0  # Use all signals
            else:
                confidence_threshold = self.config.min_analyst_confidence

            # Initialize result containers - now using tagging approach
            tagged_market_data = market_data.copy()

            # Initialize tag columns
            tagged_market_data['analyst_signal_time'] = pd.NaT
            tagged_market_data['analyst_confidence'] = 0.0
            tagged_market_data['signal_direction'] = 'none'
            tagged_market_data['signal_target_end_time'] = pd.NaT
            tagged_market_data['signal_horizon_minutes'] = 0

            # Initialize result containers for filtered data
            long_signals_list = []
            short_signals_list = []
            combined_signals_list = []

            # Process each signal timestamp with overlap handling
            for idx, row in analyst_signals.iterrows():
                signal_timestamp = row['timestamp']

                # Find confidence value
                confidence = 0.0
                if confidence_columns:
                    confidence = max(row[col] for col in confidence_columns if col in row and pd.notna(row[col]))

                # Check if confidence meets threshold
                if confidence >= confidence_threshold:
                    # Calculate target end time (signal + subsequent minutes)
                    target_end_time = signal_timestamp + timedelta(minutes=self.config.subsequent_minutes)

                    # FIXED: Only tag data within the signal window, not all future data
                    # This prevents data leakage and overlap issues
                    # Include lookback window before signal for feature calculation
                    lookback_window = timedelta(minutes=self.config.max_lookback_periods * 5)  # 5min per period
                    window_start = signal_timestamp - lookback_window

                    # Create window mask: from lookback_start to target_end (not beyond)
                    window_mask = (
                        (tagged_market_data['timestamp'] >= window_start) &
                        (tagged_market_data['timestamp'] <= target_end_time)
                    )

                    # Determine signal direction from available columns
                    signal_direction = self._determine_signal_direction(row, signal_columns)
                    direction_str = 'long' if signal_direction == SignalDirection.LONG else \
                                  'short' if signal_direction == SignalDirection.SHORT else 'combined'

                    # Handle overlapping signals: only update if new signal has higher confidence
                    # or if no previous signal at this timestamp
                    existing_mask = window_mask & (tagged_market_data['analyst_signal_time'].notna())
                    if existing_mask.any():
                        # Check if new signal has higher confidence
                        existing_confidence = tagged_market_data.loc[existing_mask, 'analyst_confidence'].iloc[0]
                        if confidence <= existing_confidence:
                            tprint_debug(f"⏭️ Skipping overlapping signal at {signal_timestamp} - lower confidence ({confidence:.3f} vs {existing_confidence:.3f})")
                            continue
                        else:
                            tprint_debug(f"🔄 Replacing overlapping signal at {signal_timestamp} - higher confidence ({confidence:.3f} vs {existing_confidence:.3f})")

                    # Tag window with signal metadata (only within defined window, prevents data leakage)
                    tagged_market_data.loc[window_mask, 'analyst_signal_time'] = signal_timestamp
                    tagged_market_data.loc[window_mask, 'analyst_confidence'] = confidence
                    tagged_market_data.loc[window_mask, 'signal_target_end_time'] = target_end_time
                    tagged_market_data.loc[window_mask, 'signal_horizon_minutes'] = self.config.subsequent_minutes
                    tagged_market_data.loc[window_mask, 'signal_direction'] = direction_str

                    if signal_direction == SignalDirection.LONG:
                        tprint_debug(f"📈 Tagged long signal at {signal_timestamp} (conf={confidence:.3f}, window={window_start} to {target_end_time})")
                    elif signal_direction == SignalDirection.SHORT:
                        tprint_debug(f"📉 Tagged short signal at {signal_timestamp} (conf={confidence:.3f}, window={window_start} to {target_end_time})")
                    else:
                        tprint_debug(f"⚖️ Tagged combined signal at {signal_timestamp} (conf={confidence:.3f})")

                    # Also create filtered views for specific analysis (but keep full data for training)
                    # Extract the 45-minute window for separate analysis if needed
                    window_data = market_data[
                        (market_data['timestamp'] >= signal_timestamp) &
                        (market_data['timestamp'] <= target_end_time)
                    ].copy()

                    if len(window_data) > 0:
                        # Add signal metadata to window data
                        window_data['analyst_signal_time'] = signal_timestamp
                        window_data['analyst_confidence'] = confidence
                        window_data['signal_target_end_time'] = target_end_time
                        window_data['signal_horizon_minutes'] = self.config.subsequent_minutes

                        if signal_direction == SignalDirection.LONG:
                            window_data['signal_direction'] = 'long'
                            long_signals_list.append(window_data)
                        elif signal_direction == SignalDirection.SHORT:
                            window_data['signal_direction'] = 'short'
                            short_signals_list.append(window_data)
                        else:
                            window_data['signal_direction'] = 'combined'
                            combined_signals_list.append(window_data)
                else:
                    tprint_debug(f"⏭️ Skipping signal at {signal_timestamp} - confidence {confidence:.3f} below threshold {confidence_threshold}")

            # Combine window data for each signal type (for analysis/debugging)
            long_window_df = pd.concat(long_signals_list, ignore_index=True) if long_signals_list else pd.DataFrame()
            short_window_df = pd.concat(short_signals_list, ignore_index=True) if short_signals_list else pd.DataFrame()
            combined_window_df = pd.concat(combined_signals_list, ignore_index=True) if combined_signals_list else pd.DataFrame()

            # Calculate quality scores
            long_quality = self._calculate_data_quality(long_window_df)
            short_quality = self._calculate_data_quality(short_window_df)

            # Count tagged signals
            long_tagged_count = len(tagged_market_data[tagged_market_data['signal_direction'] == 'long'])
            short_tagged_count = len(tagged_market_data[tagged_market_data['signal_direction'] == 'short'])
            combined_tagged_count = len(tagged_market_data[tagged_market_data['signal_direction'] == 'combined'])

            result = {
                'success': True,
                'tagged_market_data': tagged_market_data,  # Full data with tags and lookback periods
                'long_signals': long_window_df,  # 45-minute windows for analysis
                'short_signals': short_window_df,  # 45-minute windows for analysis
                'combined_signals': combined_window_df,  # 45-minute windows for analysis
                'long_count': len(long_window_df),
                'short_count': len(short_window_df),
                'combined_count': len(combined_window_df),
                'long_tagged_count': long_tagged_count,
                'short_tagged_count': short_tagged_count,
                'combined_tagged_count': combined_tagged_count,
                'long_quality_score': long_quality,
                'short_quality_score': short_quality,
                'total_processed': len(analyst_signals),
                'tagging_approach': True  # Indicates we're using tagging with full lookback
            }

            tprint_success(f"✅ Signal tagging completed: {result['long_tagged_count']} long, {result['short_tagged_count']} short, {result['combined_tagged_count']} combined tagged samples")
            tprint_info(f"📊 Quality scores: Long={result['long_quality_score']:.3f}, Short={result['short_quality_score']:.3f}")
            tprint_info(f"🔄 Using tagging approach - full lookback periods preserved for indicator calculations")

            return result

        except Exception as e:
            tprint_error(f"❌ Signal separation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'tagged_market_data': pd.DataFrame(),
                'long_signals': pd.DataFrame(),
                'short_signals': pd.DataFrame(),
                'combined_signals': pd.DataFrame()
            }

    def _determine_signal_direction(self, row: pd.Series, signal_columns: List[str]) -> SignalDirection:
        """Determine signal direction from row data."""
        try:
            # Look for explicit long/short indicators
            for col in signal_columns:
                if pd.isna(row[col]):
                    continue

                col_lower = col.lower()
                value = row[col]

                # Check for explicit long indicators
                if any(keyword in col_lower for keyword in ['long', 'bull', 'buy', 'up']):
                    if isinstance(value, (int, float)):
                        return SignalDirection.LONG if value > 0 else SignalDirection.SHORT
                    elif isinstance(value, str):
                        return SignalDirection.LONG if value.lower() in ['long', 'bull', 'buy', 'up', '1', 'true'] else SignalDirection.SHORT

                # Check for explicit short indicators
                elif any(keyword in col_lower for keyword in ['short', 'bear', 'sell', 'down']):
                    if isinstance(value, (int, float)):
                        return SignalDirection.SHORT if value > 0 else SignalDirection.LONG
                    elif isinstance(value, str):
                        return SignalDirection.SHORT if value.lower() in ['short', 'bear', 'sell', 'down', '-1', 'false'] else SignalDirection.LONG

                # Check for numeric signal values
                elif 'signal' in col_lower or 'direction' in col_lower:
                    if isinstance(value, (int, float)):
                        if value > 0.1:
                            return SignalDirection.LONG
                        elif value < -0.1:
                            return SignalDirection.SHORT

            # Default to combined if no clear direction found
            return SignalDirection.COMBINED

        except Exception as e:
            tprint_warning(f"⚠️ Error determining signal direction: {e}")
            return SignalDirection.COMBINED

    def _calculate_data_quality(self, df: pd.DataFrame) -> float:
        """Calculate data quality score using enhanced utilities."""
        try:
            if df.empty:
                tprint_debug("📊 Empty dataframe, quality score: 0.0")
                return 0.0

            # Use unified data quality validator if available
            if self.quality_validator and DATA_QUALITY_AVAILABLE:
                quality_result = self.quality_validator.validate_data_quality(
                    df,
                    context="signal_data_quality"
                )
                overall_score = quality_result.get('overall_score', 0.0)
                tprint_debug(f"📊 Quality validator score: {overall_score:.3f}")
                return validate_finite(overall_score, "data_quality")

            # Fallback: Basic quality metrics
            completeness = 1.0 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            sample_count_score = min(1.0, df.shape[0] / 1000)  # Score based on sample count

            # Time continuity score (check for gaps)
            if 'timestamp' in df.columns:
                timestamps = pd.to_datetime(df['timestamp'])
                time_diffs = timestamps.diff().dropna()
                expected_diff = pd.Timedelta(minutes=1)  # Assuming 1-minute data
                continuity_score = 1.0 - (time_diffs != expected_diff).sum() / len(time_diffs)
                continuity_score = max(0.0, continuity_score)
            else:
                continuity_score = 0.5  # Neutral score if no timestamp

            # Check for infinite values
            inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
            inf_score = 1.0 if inf_count == 0 else max(0.0, 1.0 - inf_count / df.shape[0])

            # Combine scores with weights
            quality_score = (
                completeness * 0.35 +
                sample_count_score * 0.30 +
                continuity_score * 0.20 +
                inf_score * 0.15
            )

            validated_score = validate_finite(quality_score, "data_quality")
            tprint_debug(f"📊 Calculated quality score: {validated_score:.3f} (completeness: {completeness:.3f}, samples: {sample_count_score:.3f}, continuity: {continuity_score:.3f})")
            return validated_score

        except Exception as e:
            tprint_warning(f"⚠️ Error calculating data quality: {e}")
            tprint_debug(f"Error traceback: {traceback.format_exc()}")
            return 0.0

    async def _optimize_feature_lookbacks(
        self,
        long_signals: pd.DataFrame,
        short_signals: pd.DataFrame,
        market_data: pd.DataFrame,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Optimize feature lookback periods for long and short signals."""
        try:
            tprint_info("🔍 Optimizing feature lookback periods for both signal types...")

            result = {
                'long_lookbacks': {},
                'short_lookbacks': {},
                'optimization_time': 0.0
            }

            start_time = tprint_timer()

            # Optimize for long signals
            if not long_signals.empty:
                try:
                    tprint_info("📈 Optimizing lookback periods for long signals...")
                    long_optimization = await self._run_lookback_optimization(
                        long_signals, market_data, feature_names, "long"
                    )
                    result['long_lookbacks'] = long_optimization.get('optimal_lookbacks', {})
                    tprint_success(f"✅ Long signal lookback optimization completed: {len(result['long_lookbacks'])} features")
                except Exception as e:
                    tprint_error(f"❌ Long signal lookback optimization failed: {e}")
                    result['long_lookbacks'] = {}
            else:
                tprint_info("⏭️ Skipping long signal optimization - no signals available")

            # Optimize for short signals
            if not short_signals.empty:
                try:
                    tprint_info("📉 Optimizing lookback periods for short signals...")
                    short_optimization = await self._run_lookback_optimization(
                        short_signals, market_data, feature_names, "short"
                    )
                    result['short_lookbacks'] = short_optimization.get('optimal_lookbacks', {})
                    tprint_success(f"✅ Short signal lookback optimization completed: {len(result['short_lookbacks'])} features")
                except Exception as e:
                    tprint_error(f"❌ Short signal lookback optimization failed: {e}")
                    result['short_lookbacks'] = {}
            else:
                tprint_info("⏭️ Skipping short signal optimization - no signals available")

            result['optimization_time'] = tprint_timer(start_time)
            tprint_performance("Feature lookback optimization", result['optimization_time'])

            return result

        except Exception as e:
            tprint_error(f"❌ Feature lookback optimization failed: {e}")
            return {
                'long_lookbacks': {},
                'short_lookbacks': {},
                'optimization_time': tprint_timer(start_time)
            }

    async def _run_lookback_optimization(
        self,
        signal_data: pd.DataFrame,
        market_data: pd.DataFrame,
        feature_names: List[str],
        signal_type: str
    ) -> Dict[str, Any]:
        """Run lookback optimization for a specific signal type using actual optimizer."""
        try:
            tprint_info(f"🔍 Running lookback optimization for {signal_type} signals...")

            # Use actual feature optimizer if available
            if self.feature_optimizer:
                try:
                    # Call the actual optimization component
                    optimization_result = await self.feature_optimizer.optimize_lookbacks(
                        data=signal_data,
                        feature_names=feature_names,
                        target_column='signal_direction' if 'signal_direction' in signal_data.columns else None,
                        max_lookback=self.config.max_lookback_periods,
                        direction=signal_type
                    )

                    optimal_lookbacks = optimization_result.get('optimal_lookbacks', {})
                    tprint_success(f"✅ Lookback optimization for {signal_type} completed: {len(optimal_lookbacks)} features optimized")

                    return {
                        'success': True,
                        'optimal_lookbacks': optimal_lookbacks,
                        'optimization_method': optimization_result.get('method', 'bayesian_tpe'),
                        'features_processed': len(optimal_lookbacks),
                        'optimization_score': optimization_result.get('score', 0.0)
                    }
                except Exception as opt_error:
                    tprint_warning(f"⚠️ Optimizer call failed: {opt_error}, using fallback heuristics")

            # Fallback: Intelligent heuristics based on feature categories
            tprint_info(f"📊 Using heuristic-based lookback optimization for {signal_type}")
            optimal_lookbacks = {}

            # Enhanced category-based lookbacks
            base_lookbacks = {
                'price': 20,
                'volume': 15,
                'volatility': 25,
                'momentum': 10,
                'trend': 30,
                'support_resistance': 40,
                'hmm': 50,  # HMM features need longer context
                'analyst': 10  # Analyst features are already processed
            }

            for feature in feature_names:
                feature_lower = feature.lower()

                # Determine category with priority order
                category = 'price'  # Default

                if 'hmm' in feature_lower or 'regime' in feature_lower:
                    category = 'hmm'
                elif 'analyst' in feature_lower:
                    category = 'analyst'
                elif any(kw in feature_lower for kw in ['volume', 'vol']):
                    category = 'volume'
                elif any(kw in feature_lower for kw in ['volatility', 'std', 'var', 'atr']):
                    category = 'volatility'
                elif any(kw in feature_lower for kw in ['momentum', 'rsi', 'macd', 'roc']):
                    category = 'momentum'
                elif any(kw in feature_lower for kw in ['trend', 'ma', 'ema', 'sma']):
                    category = 'trend'
                elif any(kw in feature_lower for kw in ['support', 'resistance', 'pivot', 'sr']):
                    category = 'support_resistance'

                # Apply signal-specific adjustments
                base_lookback = base_lookbacks[category]
                if signal_type == 'long':
                    # Longer lookbacks for long signals (trend following)
                    lookback = int(base_lookback * 1.2)
                elif signal_type == 'short':
                    # Shorter lookbacks for short signals (mean reversion)
                    lookback = int(base_lookback * 0.8)
                else:
                    lookback = base_lookback

                # Ensure minimum and maximum bounds with validation
                lookback = max(5, min(self.config.max_lookback_periods, lookback))
                lookback = int(validate_positive(lookback, "lookback"))
                optimal_lookbacks[feature] = lookback

            tprint_success(f"✅ Heuristic lookback optimization for {signal_type}: {len(optimal_lookbacks)} features")

            return {
                'success': True,
                'optimal_lookbacks': optimal_lookbacks,
                'optimization_method': 'heuristic_fallback',
                'features_processed': len(optimal_lookbacks)
            }

        except Exception as e:
            tprint_error(f"❌ Lookback optimization for {signal_type} failed: {e}")
            tprint_error(f"Traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'optimal_lookbacks': {},
                'error': str(e)
            }

    async def _generate_pid_features(
        self,
        long_signals: pd.DataFrame,
        short_signals: pd.DataFrame,
        market_data: pd.DataFrame,
        feature_names: List[str],
        long_lookbacks: Dict[str, int],
        short_lookbacks: Dict[str, int]
    ) -> Dict[str, Any]:
        """Generate interactive features for long and short signals."""
        try:
            tprint_info("🎯 Generating interactive features for both signal types...")

            result = {
                'long_features': None,
                'short_features': None,
                'generation_time': 0.0
            }

            start_time = tprint_timer()

            # Generate features for long signals
            if not long_signals.empty:
                try:
                    tprint_info("📈 Generating interactive features for long signals...")
                    long_features = await self._generate_signal_interactive_features(
                        long_signals, market_data, feature_names, long_lookbacks, "long"
                    )
                    result['long_features'] = long_features
                    tprint_success(f"✅ Long interactive features generated: {long_features.total_features_generated} features")
                except Exception as e:
                    tprint_error(f"❌ Long interactive feature generation failed: {e}")
                    result['long_features'] = self._create_empty_interactive_result()
            else:
                tprint_info("⏭️ Skipping long interactive generation - no signals available")
                result['long_features'] = self._create_empty_interactive_result()

            # Generate features for short signals
            if not short_signals.empty:
                try:
                    tprint_info("📉 Generating interactive features for short signals...")
                    short_features = await self._generate_signal_interactive_features(
                        short_signals, market_data, feature_names, short_lookbacks, "short"
                    )
                    result['short_features'] = short_features
                    tprint_success(f"✅ Short interactive features generated: {short_features.total_features_generated} features")
                except Exception as e:
                    tprint_error(f"❌ Short interactive feature generation failed: {e}")
                    result['short_features'] = self._create_empty_interactive_result()
            else:
                tprint_info("⏭️ Skipping short interactive generation - no signals available")
                result['short_features'] = self._create_empty_interactive_result()

            result['generation_time'] = tprint_timer(start_time)
            tprint_performance("Interactive feature generation", result['generation_time'])

            return result

        except Exception as e:
            tprint_error(f"❌ Interactive feature generation failed: {e}")
            return {
                'long_features': self._create_empty_interactive_result(),
                'short_features': self._create_empty_interactive_result(),
                'generation_time': tprint_timer(start_time)
            }

    async def _generate_signal_interactive_features(
        self,
        signal_data: pd.DataFrame,
        market_data: pd.DataFrame,
        feature_names: List[str],
        optimized_lookbacks: Dict[str, int],
        signal_type: str
    ) -> OrchestratorResult:
        """Generate interactive features for a specific signal type using actual generator."""
        try:
            tprint_info(f"🎯 Generating interactive features for {signal_type} signals...")

            # Prepare target based on actual signal strength if available
            if 'analyst_confidence' in signal_data.columns:
                # Use actual analyst confidence as signal strength
                target_values = signal_data['analyst_confidence'].values
                tprint_debug(f"Using analyst confidence as target strength (mean: {target_values.mean():.3f})")
            elif 'signal_direction' in signal_data.columns:
                # Convert signal direction to binary target
                target_values = (signal_data['signal_direction'] == signal_type).astype(float)
                tprint_debug(f"Using binary signal direction as target")
            else:
                # Fallback to uniform targets
                target_values = np.ones(len(signal_data))
                tprint_warning("⚠️ No signal strength available, using uniform targets")

            # Validate target values
            target_values = np.clip(target_values, 0, 1)  # Ensure [0, 1] range
            target_values = np.nan_to_num(target_values, nan=0.5)  # Replace NaN with neutral value

            # Create target dictionary with actual signal strength
            target_dict = {
                signal_type: target_values,
                'combined': target_values
            }

            # Prepare feature data - ensure we have the right columns
            available_features = [f for f in feature_names if f in signal_data.columns]
            if len(available_features) < len(feature_names):
                missing = set(feature_names) - set(available_features)
                tprint_warning(f"⚠️ Missing {len(missing)} features: {list(missing)[:5]}...")

            if not available_features:
                raise ValueError(f"No available features in signal data for {signal_type}")

            tprint_debug(f"Using {len(available_features)} features for generation")

            # Call interactive feature generator
            if self.interactive_generator:
                try:
                    interactive_result = await self.interactive_generator.generate_features(
                        data=signal_data[available_features].values,
                        feature_names=available_features,
                        optimized_lookback_periods=optimized_lookbacks,
                        target=target_dict,
                        direction=signal_type,
                        enable_parallel=self.config.enable_parallel_processing,
                        memory_limit_gb=self.config.memory_limit_gb
                    )

                    # Validate result
                    if interactive_result and hasattr(interactive_result, 'total_features_generated'):
                        tprint_success(f"✅ Generated {interactive_result.total_features_generated} interactive features for {signal_type}")
                    else:
                        tprint_warning(f"⚠️ Interactive generator returned unexpected result type")

                    return interactive_result

                except Exception as gen_error:
                    tprint_error(f"❌ Interactive generator call failed: {gen_error}")
                    tprint_error(f"Traceback: {traceback.format_exc()}")
                    raise
            else:
                raise ValueError("Interactive feature generator not available")

        except Exception as e:
            tprint_error(f"❌ Interactive feature generation for {signal_type} failed: {e}")
            return self._create_empty_interactive_result()

    def _create_empty_interactive_result(self) -> OrchestratorResult:
        """Create empty interactive feature result for fallback."""
        return OrchestratorResult(
            combined_features={},
            combined_feature_names=[],
            feature_importance_scores={},
            total_features_generated=0,
            execution_time=0.0,
            generation_status="failed"
        )

    def _add_comprehensive_reporting(self, result: OrchestratorResult, start_time: float) -> OrchestratorResult:
        """Add comprehensive reporting and metrics to orchestration results."""
        try:
            total_time = time.time() - start_time

            # Helper function to safely get attribute
            def safe_get(obj, attr, default=0):
                try:
                    val = getattr(obj, attr, default)
                    return val if val is not None else default
                except:
                    return default

            # Helper function to safely get DataFrame length
            def safe_len(obj, attr, default=0):
                try:
                    val = getattr(obj, attr, None)
                    return len(val) if val is not None and hasattr(val, '__len__') else default
                except:
                    return default

            # Create comprehensive report with safe attribute access
            comprehensive_report = {
                'orchestration_summary': {
                    'total_orchestration_time': total_time,
                    'success': result.feature_generation_status == "completed",
                    'input_samples': safe_len(result, 'tagged_market_data'),
                    'long_samples_processed': safe_get(result, 'total_long_samples'),
                    'short_samples_processed': safe_get(result, 'total_short_samples'),
                    'analyst_signals_count': safe_get(result, 'analyst_signals_count'),
                    'long_signals_count': safe_get(result, 'long_signals_count'),
                    'short_signals_count': safe_get(result, 'short_signals_count'),
                    'confidence_threshold': self.config.confidence_threshold,
                    'subsequent_minutes': self.config.subsequent_minutes,
                    'direction_mode': self.config.direction_mode,
                    'separate_directional_features': self.config.separate_directional_features,
                    'feature_generation_status': result.feature_generation_status,
                    'lookback_optimization_enabled': safe_get(result, 'lookback_optimization_enabled', False),
                    'pid_feature_generation_enabled': safe_get(result, 'pid_feature_generation_enabled', False),
                    'interactive_feature_generation_enabled': safe_get(result, 'interactive_feature_generation_enabled', False),
                    'horizon_labeling_enabled': safe_get(result, 'horizon_labeling_enabled', False),
                    'feature_selection_enabled': safe_get(result, 'feature_selection_enabled', False),
                    'intermediate_results_saved': safe_get(result, 'intermediate_results_saved', False)
                },
                'signal_separation_metrics': {
                    'total_analyst_signals': safe_get(result, 'analyst_signals_count'),
                    'long_signals_with_confidence': safe_get(result, 'long_signals_count'),
                    'short_signals_with_confidence': safe_get(result, 'short_signals_count'),
                    'long_signal_ratio': safe_divide(
                        safe_get(result, 'long_signals_count'),
                        max(safe_get(result, 'analyst_signals_count'), 1)
                    ),
                    'short_signal_ratio': safe_divide(
                        safe_get(result, 'short_signals_count'),
                        max(safe_get(result, 'analyst_signals_count'), 1)
                    ),
                    'average_confidence_long': safe_get(result, 'average_long_confidence', 0.0),
                    'average_confidence_short': safe_get(result, 'average_short_confidence', 0.0),
                    'tagging_approach_used': True  # We use tagging instead of extraction
                },
                'feature_processing_metrics': {
                    'total_features_available': safe_len(result, 'feature_names'),
                    'long_features_count': safe_len(result, 'long_selected_features'),
                    'short_features_count': safe_len(result, 'short_selected_features'),
                    'long_optimized_lookbacks_count': len(safe_get(result, 'long_optimized_lookbacks', {})),
                    'short_optimized_lookbacks_count': len(safe_get(result, 'short_optimized_lookbacks', {})),
                    'long_pid_features_generated': safe_get(result, 'long_pid_features') is not None,
                    'short_pid_features_generated': safe_get(result, 'short_pid_features') is not None,
                    'long_targets_count': len(safe_get(result, 'long_targets', {})),
                    'short_targets_count': len(safe_get(result, 'short_targets', {}))
                },
                'data_quality_metrics': {
                    'long_data_quality_score': safe_get(result, 'long_data_quality_score', 0.0),
                    'short_data_quality_score': safe_get(result, 'short_data_quality_score', 0.0),
                    'long_missing_values_ratio': safe_get(result, 'long_missing_values_ratio', 0.0),
                    'short_missing_values_ratio': safe_get(result, 'short_missing_values_ratio', 0.0),
                    'long_outlier_ratio': safe_get(result, 'long_outlier_ratio', 0.0),
                    'short_outlier_ratio': safe_get(result, 'short_outlier_ratio', 0.0)
                },
                'performance_metrics': {
                    'optimization_time': safe_get(result, 'optimization_time', 0.0),
                    'generation_time': safe_get(result, 'generation_time', 0.0),
                    'labeling_time': safe_get(result, 'labeling_time', 0.0),
                    'selection_time': safe_get(result, 'selection_time', 0.0),
                    'preparation_time': safe_get(result, 'preparation_time', 0.0),
                    'total_execution_time': safe_get(result, 'execution_time', 0.0),
                    'memory_usage_mb': safe_get(result, 'memory_usage_mb', 0.0),
                    'cpu_usage_percent': safe_get(result, 'cpu_usage_percent', 0.0)
                },
                'model_integration_metrics': {
                    'hmm_features_included': len([f for f in safe_get(result, 'long_selected_features', []) if 'hmm' in f.lower()]),
                    'analyst_features_included': len([f for f in safe_get(result, 'long_selected_features', []) if 'analyst' in f.lower()]),
                    'technical_features_included': len([f for f in safe_get(result, 'long_selected_features', [])
                                                       if f not in ['hmm_regime_0_prob', 'hmm_regime_1_prob', 'hmm_regime_2_prob',
                                                                   'analyst_action', 'analyst_confidence']]),
                    'long_features_with_hmm': safe_len(result, 'long_selected_features'),
                    'short_features_with_hmm': safe_len(result, 'short_selected_features'),
                    'feature_integration_complete': True
                },
                'evaluation_metrics': {
                    'long_training_accuracy': safe_get(result, 'long_training_accuracy', 0.0),
                    'short_training_accuracy': safe_get(result, 'short_training_accuracy', 0.0),
                    'long_validation_accuracy': safe_get(result, 'long_validation_accuracy', 0.0),
                    'short_validation_accuracy': safe_get(result, 'short_validation_accuracy', 0.0),
                    'long_f1_score': safe_get(result, 'long_f1_score', 0.0),
                    'short_f1_score': safe_get(result, 'short_f1_score', 0.0),
                    'long_precision': safe_get(result, 'long_precision', 0.0),
                    'short_precision': safe_get(result, 'short_precision', 0.0),
                    'long_recall': safe_get(result, 'long_recall', 0.0),
                    'short_recall': safe_get(result, 'short_recall', 0.0),
                    'long_roc_auc': safe_get(result, 'long_roc_auc', 0.0),
                    'short_roc_auc': safe_get(result, 'short_roc_auc', 0.0),
                    'long_sharpe_ratio': safe_get(result, 'long_sharpe_ratio', 0.0),
                    'short_sharpe_ratio': safe_get(result, 'short_sharpe_ratio', 0.0),
                    'long_max_drawdown': safe_get(result, 'long_max_drawdown', 0.0),
                    'short_max_drawdown': safe_get(result, 'short_max_drawdown', 0.0),
                    'long_total_trades': safe_get(result, 'long_total_trades', 0),
                    'short_total_trades': safe_get(result, 'short_total_trades', 0),
                    'long_avg_trades_per_month': safe_get(result, 'long_avg_trades_per_month', 0.0),
                    'short_avg_trades_per_month': safe_get(result, 'short_avg_trades_per_month', 0.0),
                    'long_total_pnl': safe_get(result, 'long_total_pnl', 0.0),
                    'short_total_pnl': safe_get(result, 'short_total_pnl', 0.0),
                    'long_monthly_pnl': safe_get(result, 'long_monthly_pnl', {}),
                    'short_monthly_pnl': safe_get(result, 'short_monthly_pnl', {}),
                    'evaluation_completed': safe_get(result, 'evaluation_completed', False)
                }
            }

            # Add comprehensive report to result
            result.comprehensive_report = comprehensive_report

            # Log comprehensive summary
            self._log_comprehensive_summary(comprehensive_report)

            return result

        except Exception as e:
            tprint_error(f"❌ Failed to add comprehensive reporting: {e}")
            # Return result without reporting if it fails
            return result

    def _log_comprehensive_summary(self, report: Dict[str, Any]) -> None:
        """Log comprehensive orchestration summary with enhanced tprint integration."""
        try:
            orchestration = report['orchestration_summary']
            signals = report['signal_separation_metrics']
            features = report['feature_processing_metrics']
            quality = report['data_quality_metrics']
            performance = report['performance_metrics']
            integration = report['model_integration_metrics']
            evaluation = report.get('evaluation_metrics', {})

            tprint_info("=" * 80)
            tprint_info("🎯 TACTICIAN PRE-ML ORCHESTRATION SUMMARY")
            tprint_info("=" * 80)
            tprint_info(f"⏱️  Total Orchestration Time: {orchestration['total_orchestration_time']:.2f}s")
            tprint_info(f"✅ Success: {'Yes' if orchestration['success'] else 'No'}")
            tprint_info(f"📊 Status: {orchestration['feature_generation_status']}")
            tprint_info(f"🎯 Direction Mode: {orchestration['direction_mode']}")
            tprint_info(f"🔄 Separate Features: {'Enabled' if orchestration['separate_directional_features'] else 'Disabled'}")
            tprint_info(f"🔄 Lookback Optimization: {'Enabled' if orchestration['lookback_optimization_enabled'] else 'Disabled'}")
            tprint_info(f"🧬 Interactive Features: {'Enabled' if orchestration['interactive_feature_generation_enabled'] else 'Disabled'}")
            tprint_info(f"🎯 Horizon Labeling: {'Enabled' if orchestration['horizon_labeling_enabled'] else 'Disabled'}")
            tprint_info(f"🎛️ Feature Selection: {'Enabled' if orchestration['feature_selection_enabled'] else 'Disabled'}")

            tprint_info("\n📈 Signal Separation Results:")
            tprint_info(f"  📊 Total Analyst Signals: {signals['total_analyst_signals']}")
            tprint_info(f"  📈 Long Signals (≥{orchestration['confidence_threshold']}): {signals['long_signals_with_confidence']}")
            tprint_info(f"  📉 Short Signals (≥{orchestration['confidence_threshold']}): {signals['short_signals_with_confidence']}")
            tprint_info(f"  📊 Long Signal Ratio: {signals['long_signal_ratio']:.3f}")
            tprint_info(f"  📉 Short Signal Ratio: {signals['short_signal_ratio']:.3f}")
            tprint_info(f"  🎯 Avg Confidence Long: {signals['average_confidence_long']:.3f}")
            tprint_info(f"  🎯 Avg Confidence Short: {signals['average_confidence_short']:.3f}")

            tprint_info("\n🔢 Feature Processing Results:")
            if orchestration['direction_mode'] == 'both':
                tprint_info(f"  📊 Long Features: {features['long_features_count']}")
                tprint_info(f"  📉 Short Features: {features['short_features_count']}")
                tprint_info(f"  🔍 Long Lookbacks Optimized: {features['long_optimized_lookbacks_count']}")
                tprint_info(f"  🔍 Short Lookbacks Optimized: {features['short_optimized_lookbacks_count']}")
                tprint_info(f"  🎯 Long Targets: {features['long_targets_count']}")
                tprint_info(f"  🎯 Short Targets: {features['short_targets_count']}")
            elif orchestration['direction_mode'] == 'long_only':
                tprint_info(f"  📊 Long Features: {features['long_features_count']}")
                tprint_info(f"  🔍 Long Lookbacks Optimized: {features['long_optimized_lookbacks_count']}")
                tprint_info(f"  🎯 Long Targets: {features['long_targets_count']}")
            elif orchestration['direction_mode'] == 'short_only':
                tprint_info(f"  📉 Short Features: {features['short_features_count']}")
                tprint_info(f"  🔍 Short Lookbacks Optimized: {features['short_optimized_lookbacks_count']}")
                tprint_info(f"  🎯 Short Targets: {features['short_targets_count']}")

            tprint_info("\n📊 Data Quality Metrics:")
            tprint_info(f"  📈 Long Quality: {quality['long_data_quality_score']:.3f}")
            tprint_info(f"  📉 Short Quality: {quality['short_data_quality_score']:.3f}")
            tprint_info(f"  ❌ Long Missing Values: {quality['long_missing_values_ratio']:.3f}")
            tprint_info(f"  ❌ Short Missing Values: {quality['short_missing_values_ratio']:.3f}")
            tprint_info(f"  🚨 Long Outliers: {quality['long_outlier_ratio']:.3f}")
            tprint_info(f"  🚨 Short Outliers: {quality['short_outlier_ratio']:.3f}")

            tprint_info("\n⚡ Performance Metrics:")
            tprint_info(f"  🔍 Optimization Time: {performance['optimization_time']:.2f}s")
            tprint_info(f"  🧬 Generation Time: {performance['generation_time']:.2f}s")
            tprint_info(f"  🎯 Labeling Time: {performance['labeling_time']:.2f}s")
            tprint_info(f"  🎛️ Selection Time: {performance['selection_time']:.2f}s")
            tprint_info(f"  📊 Preparation Time: {performance['preparation_time']:.2f}s")
            tprint_info(f"  💾 Memory Usage: {performance['memory_usage_mb']:.1f} MB")
            tprint_info(f"  🖥️ CPU Usage: {performance['cpu_usage_percent']:.1f}%")

            tprint_info("\n🔗 Model Integration:")
            tprint_info(f"  🧬 HMM Features: {integration['hmm_features_included']}")
            tprint_info(f"  🎯 Analyst Features: {integration['analyst_features_included']}")
            tprint_info(f"  📊 Technical Features: {integration['technical_features_included']}")
            tprint_info(f"  ✅ Integration Complete: {'Yes' if integration['feature_integration_complete'] else 'No'}")

            # Log sample counts if available
            if hasattr(orchestration, 'input_samples') and orchestration['input_samples'] > 0:
                tprint_info(f"\n📈 Sample Processing:")
                tprint_info(f"  📊 Input Samples: {orchestration['input_samples']}")
                if orchestration['direction_mode'] == 'both':
                    tprint_info(f"  📈 Long Samples: {orchestration['long_samples_processed']}")
                    tprint_info(f"  📉 Short Samples: {orchestration['short_samples_processed']}")
                elif orchestration['direction_mode'] == 'long_only':
                    tprint_info(f"  📈 Long Samples: {orchestration['long_samples_processed']}")
                elif orchestration['direction_mode'] == 'short_only':
                    tprint_info(f"  📉 Short Samples: {orchestration['short_samples_processed']}")

            # Log evaluation metrics if available
            if evaluation and evaluation.get('evaluation_completed', False):
                tprint_info("\n🎯 Model Evaluation Metrics:")
                if orchestration['direction_mode'] == 'both':
                    tprint_info(f"  📈 Long Training Accuracy: {evaluation['long_training_accuracy']:.4f}")
                    tprint_info(f"  📉 Short Training Accuracy: {evaluation['short_training_accuracy']:.4f}")
                    tprint_info(f"  ✅ Long Validation Accuracy: {evaluation['long_validation_accuracy']:.4f}")
                    tprint_info(f"  ✅ Short Validation Accuracy: {evaluation['short_validation_accuracy']:.4f}")
                    tprint_info(f"  🎯 Long F1 Score: {evaluation['long_f1_score']:.4f}")
                    tprint_info(f"  🎯 Short F1 Score: {evaluation['short_f1_score']:.4f}")
                    tprint_info(f"  📊 Long Precision: {evaluation['long_precision']:.4f}")
                    tprint_info(f"  📊 Short Precision: {evaluation['short_precision']:.4f}")
                    tprint_info(f"  📈 Long Recall: {evaluation['long_recall']:.4f}")
                    tprint_info(f"  📉 Short Recall: {evaluation['short_recall']:.4f}")
                    tprint_info(f"  📈 Long ROC-AUC: {evaluation['long_roc_auc']:.4f}")
                    tprint_info(f"  📉 Short ROC-AUC: {evaluation['short_roc_auc']:.4f}")
                    tprint_info(f"  💰 Long Sharpe Ratio: {evaluation['long_sharpe_ratio']:.4f}")
                    tprint_info(f"  💰 Short Sharpe Ratio: {evaluation['short_sharpe_ratio']:.4f}")
                    tprint_info(f"  📉 Long Max Drawdown: {evaluation['long_max_drawdown']:.4f}")
                    tprint_info(f"  📉 Short Max Drawdown: {evaluation['short_max_drawdown']:.4f}")
                elif orchestration['direction_mode'] == 'long_only':
                    tprint_info(f"  📈 Long Training Accuracy: {evaluation['long_training_accuracy']:.4f}")
                    tprint_info(f"  ✅ Long Validation Accuracy: {evaluation['long_validation_accuracy']:.4f}")
                    tprint_info(f"  🎯 Long F1 Score: {evaluation['long_f1_score']:.4f}")
                    tprint_info(f"  📊 Long Precision: {evaluation['long_precision']:.4f}")
                    tprint_info(f"  📈 Long Recall: {evaluation['long_recall']:.4f}")
                    tprint_info(f"  📈 Long ROC-AUC: {evaluation['long_roc_auc']:.4f}")
                    tprint_info(f"  💰 Long Sharpe Ratio: {evaluation['long_sharpe_ratio']:.4f}")
                    tprint_info(f"  📉 Long Max Drawdown: {evaluation['long_max_drawdown']:.4f}")
                elif orchestration['direction_mode'] == 'short_only':
                    tprint_info(f"  📉 Short Training Accuracy: {evaluation['short_training_accuracy']:.4f}")
                    tprint_info(f"  ✅ Short Validation Accuracy: {evaluation['short_validation_accuracy']:.4f}")
                    tprint_info(f"  🎯 Short F1 Score: {evaluation['short_f1_score']:.4f}")
                    tprint_info(f"  📊 Short Precision: {evaluation['short_precision']:.4f}")
                    tprint_info(f"  📉 Short Recall: {evaluation['short_recall']:.4f}")
                    tprint_info(f"  📉 Short ROC-AUC: {evaluation['short_roc_auc']:.4f}")
                    tprint_info(f"  💰 Short Sharpe Ratio: {evaluation['short_sharpe_ratio']:.4f}")
                    tprint_info(f"  📉 Short Max Drawdown: {evaluation['short_max_drawdown']:.4f}")

                # Financial trading metrics
                if orchestration['direction_mode'] == 'both':
                    tprint_info(f"  🤖 Long Total Trades: {evaluation['long_total_trades']}")
                    tprint_info(f"  🤖 Short Total Trades: {evaluation['short_total_trades']}")
                    tprint_info(f"  📊 Long Avg Trades/Month: {evaluation['long_avg_trades_per_month']:.1f}")
                    tprint_info(f"  📊 Short Avg Trades/Month: {evaluation['short_avg_trades_per_month']:.1f}")
                    tprint_info(f"  💵 Long Total P&L: {evaluation['long_total_pnl']:.6f}")
                    tprint_info(f"  💵 Short Total P&L: {evaluation['short_total_pnl']:.6f}")
                elif orchestration['direction_mode'] == 'long_only':
                    tprint_info(f"  🤖 Long Total Trades: {evaluation['long_total_trades']}")
                    tprint_info(f"  📊 Long Avg Trades/Month: {evaluation['long_avg_trades_per_month']:.1f}")
                    tprint_info(f"  💵 Long Total P&L: {evaluation['long_total_pnl']:.6f}")
                elif orchestration['direction_mode'] == 'short_only':
                    tprint_info(f"  🤖 Short Total Trades: {evaluation['short_total_trades']}")
                    tprint_info(f"  📊 Short Avg Trades/Month: {evaluation['short_avg_trades_per_month']:.1f}")
                    tprint_info(f"  💵 Short Total P&L: {evaluation['short_total_pnl']:.6f}")

                # Monthly P&L breakdown (show top 5 months)
                if orchestration['direction_mode'] == 'both':
                    if evaluation['long_monthly_pnl']:
                        tprint_info("  📅 Long Monthly P&L (Top 5):")
                        sorted_months = sorted(evaluation['long_monthly_pnl'].items(), key=lambda x: x[1], reverse=True)[:5]
                        for month, pnl in sorted_months:
                            tprint_info(f"    {month}: {pnl:.6f}")

                    if evaluation['short_monthly_pnl']:
                        tprint_info("  📅 Short Monthly P&L (Top 5):")
                        sorted_months = sorted(evaluation['short_monthly_pnl'].items(), key=lambda x: x[1], reverse=True)[:5]
                        for month, pnl in sorted_months:
                            tprint_info(f"    {month}: {pnl:.6f}")
                elif orchestration['direction_mode'] == 'long_only' and evaluation['long_monthly_pnl']:
                    tprint_info("  📅 Long Monthly P&L (Top 5):")
                    sorted_months = sorted(evaluation['long_monthly_pnl'].items(), key=lambda x: x[1], reverse=True)[:5]
                    for month, pnl in sorted_months:
                        tprint_info(f"    {month}: {pnl:.6f}")
                elif orchestration['direction_mode'] == 'short_only' and evaluation['short_monthly_pnl']:
                    tprint_info("  📅 Short Monthly P&L (Top 5):")
                    sorted_months = sorted(evaluation['short_monthly_pnl'].items(), key=lambda x: x[1], reverse=True)[:5]
                    for month, pnl in sorted_months:
                        tprint_info(f"    {month}: {pnl:.6f}")
            else:
                tprint_info("\n📊 Evaluation Status:")
                tprint_info(f"  ✅ Evaluation Completed: {'Yes' if evaluation.get('evaluation_completed', False) else 'No'}")
                if not evaluation.get('evaluation_completed', False):
                    tprint_info(f"  ⚠️ Evaluation metrics not available yet (pre-ML orchestration step)")

            tprint_info("=" * 80)

        except Exception as e:
            tprint_error(f"❌ Failed to log comprehensive summary: {e}")
            # Fallback to basic logging
            try:
                tprint_info("🔄 Basic Orchestration Summary:")
                tprint_info(f"  ⏱️ Total Time: {orchestration['total_orchestration_time']:.2f}s")
                tprint_info(f"  ✅ Success: {orchestration['success']}")
                tprint_info(f"  📈 Long Samples: {orchestration['long_samples_processed']}")
                tprint_info(f"  📉 Short Samples: {orchestration['short_samples_processed']}")
                tprint_info(f"  🔢 Long Features: {features['long_features_count']}")
                tprint_info(f"  🔢 Short Features: {features['short_features_count']}")
            except (KeyError, TypeError) as e:
                tprint_warning(f"⚠️ Could not display basic summary: {e}")
            except Exception as e:
                tprint_error(f"❌ Unexpected error in basic summary logging: {e}")

    async def _apply_horizon_labeling(
        self,
        long_signals: pd.DataFrame,
        short_signals: pd.DataFrame,
        market_data: pd.DataFrame,
        long_pid_features: OrchestratorResult,
        short_pid_features: OrchestratorResult
    ) -> Dict[str, Any]:
        """Apply multi-horizon profit labeling for long and short signals."""
        try:
            tprint_info("🏷️ Applying multi-horizon profit labeling for both signal types...")

            result = {
                'long_targets': {},
                'short_targets': {},
                'labeling_time': 0.0
            }

            start_time = tprint_timer()

            # Apply labeling for long signals
            if not long_signals.empty and long_pid_features:
                try:
                    tprint_info("📈 Applying horizon labeling for long signals...")
                    long_targets = await self._apply_signal_horizon_labeling(
                        long_signals, market_data, long_pid_features, "long"
                    )
                    result['long_targets'] = long_targets
                    tprint_success(f"✅ Long horizon labeling completed: {len(long_targets)} target sets")
                except Exception as e:
                    tprint_error(f"❌ Long horizon labeling failed: {e}")
                    result['long_targets'] = {}
            else:
                tprint_info("⏭️ Skipping long horizon labeling - no signals or features available")

            # Apply labeling for short signals
            if not short_signals.empty and short_pid_features:
                try:
                    tprint_info("📉 Applying horizon labeling for short signals...")
                    short_targets = await self._apply_signal_horizon_labeling(
                        short_signals, market_data, short_pid_features, "short"
                    )
                    result['short_targets'] = short_targets
                    tprint_success(f"✅ Short horizon labeling completed: {len(short_targets)} target sets")
                except Exception as e:
                    tprint_error(f"❌ Short horizon labeling failed: {e}")
                    result['short_targets'] = {}
            else:
                tprint_info("⏭️ Skipping short horizon labeling - no signals or features available")

            result['labeling_time'] = tprint_timer(start_time)
            tprint_performance("Horizon labeling", result['labeling_time'])

            return result

        except Exception as e:
            tprint_error(f"❌ Horizon labeling failed: {e}")
            return {
                'long_targets': {},
                'short_targets': {},
                'labeling_time': tprint_timer(start_time)
            }

    async def _apply_signal_horizon_labeling(
        self,
        signal_data: pd.DataFrame,
        market_data: pd.DataFrame,
        pid_features: OrchestratorResult,
        signal_type: str
    ) -> Dict[str, np.ndarray]:
        """Apply horizon labeling for a specific signal type using actual labeler."""
        try:
            tprint_info(f"🏷️ Applying horizon labeling for {signal_type} signals...")

            # Use actual multi-horizon profit labeler if available
            if self.horizon_labeler:
                try:
                    # Prepare price data for labeling
                    price_columns = ['open', 'high', 'low', 'close']
                    if not all(col in signal_data.columns for col in price_columns):
                        # Try to get price data from market_data if not in signal_data
                        if all(col in market_data.columns for col in price_columns):
                            # Merge price data based on timestamp
                            if 'timestamp' in signal_data.columns and 'timestamp' in market_data.columns:
                                signal_data = signal_data.merge(
                                    market_data[['timestamp'] + price_columns],
                                    on='timestamp',
                                    how='left',
                                    suffixes=('', '_market')
                                )
                                tprint_debug("📊 Merged price data from market_data")

                    # Call actual horizon labeler
                    labeling_result = await self.horizon_labeler.label_multi_horizon(
                        data=signal_data,
                        direction=signal_type,
                        profit_targets=self.config.profit_targets,
                        time_horizons=self.config.time_horizons,
                        enable_quality_scoring=True
                    )

                    labeled_targets = labeling_result.get('labeled_targets', {})

                    # Log quality metrics if available
                    if 'quality_metrics' in labeling_result:
                        quality = labeling_result['quality_metrics']
                        tprint_debug(f"📊 Labeling quality: {quality.get('overall_score', 0.0):.3f}")

                    tprint_success(f"✅ Horizon labeling for {signal_type}: {len(labeled_targets)} target sets generated")
                    return labeled_targets

                except Exception as label_error:
                    tprint_warning(f"⚠️ Horizon labeler call failed: {label_error}, using fallback")

            # Fallback: Create targets based on forward returns
            tprint_info(f"📊 Using fallback profit labeling for {signal_type}")
            labeled_targets = {}

            # Ensure we have required price columns
            if 'close' not in signal_data.columns:
                tprint_error("❌ No close price column available for labeling")
                return {}

            # Calculate forward returns for each horizon
            for horizon_name, horizon_periods in self.config.time_horizons.items():
                for target_name, target_pct in self.config.profit_targets.items():
                    # Calculate forward returns
                    future_prices = signal_data['close'].shift(-horizon_periods)
                    current_prices = signal_data['close']

                    if signal_type == 'long':
                        # Long: profit when price goes up
                        returns = (future_prices - current_prices) / current_prices
                    else:
                        # Short: profit when price goes down
                        returns = (current_prices - future_prices) / current_prices

                    # Create binary target: 1 if return >= target, 0 otherwise
                    target_values = (returns >= target_pct).astype(float)

                    # Handle NaN values (end of data)
                    target_values = target_values.fillna(0).values

                    # Store with descriptive key
                    key = f"{signal_type}_{target_name}_{horizon_name}"
                    labeled_targets[key] = target_values

                    # Log statistics
                    hit_rate = target_values.mean()
                    tprint_debug(f"📊 {key}: hit_rate={hit_rate:.3f} (target={target_pct:.3f})")

            tprint_success(f"✅ Fallback labeling for {signal_type}: {len(labeled_targets)} target sets")
            return labeled_targets

        except Exception as e:
            tprint_error(f"❌ Horizon labeling for {signal_type} failed: {e}")
            tprint_error(f"Traceback: {traceback.format_exc()}")
            return {}

    async def _select_final_features(
        self,
        long_signals: pd.DataFrame,
        short_signals: pd.DataFrame,
        long_pid_features: OrchestratorResult,
        short_pid_features: OrchestratorResult,
        long_targets: Dict[str, np.ndarray],
        short_targets: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Select final features for long and short signals."""
        try:
            tprint_info("🎯 Selecting final features for both signal types...")

            result = {
                'long_features': [],
                'short_features': [],
                'selection_time': 0.0
            }

            start_time = tprint_timer()

            # Select features for long signals
            if not long_signals.empty and long_pid_features and long_targets:
                try:
                    tprint_info("📈 Selecting final features for long signals...")
                    long_features = await self._select_signal_features(
                        long_pid_features, long_targets, "long"
                    )
                    result['long_features'] = long_features
                    tprint_success(f"✅ Long feature selection completed: {len(long_features)} features")
                except Exception as e:
                    tprint_error(f"❌ Long feature selection failed: {e}")
                    result['long_features'] = []
            else:
                tprint_info("⏭️ Skipping long feature selection - missing data")

            # Select features for short signals
            if not short_signals.empty and short_pid_features and short_targets:
                try:
                    tprint_info("📉 Selecting final features for short signals...")
                    short_features = await self._select_signal_features(
                        short_pid_features, short_targets, "short"
                    )
                    result['short_features'] = short_features
                    tprint_success(f"✅ Short feature selection completed: {len(short_features)} features")
                except Exception as e:
                    tprint_error(f"❌ Short feature selection failed: {e}")
                    result['short_features'] = []
            else:
                tprint_info("⏭️ Skipping short feature selection - missing data")

            result['selection_time'] = tprint_timer(start_time)
            tprint_performance("Feature selection", result['selection_time'])

            return result

        except Exception as e:
            tprint_error(f"❌ Feature selection failed: {e}")
            return {
                'long_features': [],
                'short_features': [],
                'selection_time': tprint_timer(start_time)
            }

    async def _select_signal_features(
        self,
        pid_features: OrchestratorResult,
        targets: Dict[str, np.ndarray],
        signal_type: str
    ) -> List[str]:
        """Select final features for a specific signal type."""
        try:
            # This would call the actual feature selection component
            # For now, return mock selected features
            tprint_info(f"🎯 Selecting final features for {signal_type} signals...")

            # Get available features from PID results
            available_features = pid_features.combined_feature_names

            if not available_features:
                tprint_warning(f"⚠️ No interactive features available for {signal_type} selection")
                return []

            # Apply proper feature selection with configured feature limit
            importance_scores = pid_features.feature_importance_scores
            target_features = self.config.max_features  # Use configured max features

            tprint_info(f"🔍 FEATURE SELECTION: Starting with {len(available_features)} features, target: {target_features}")

            # Sort features by importance and select top features
            if importance_scores:
                sorted_features = sorted(
                    importance_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                # Select top N features based on config
                selected_features = [f[0] for f in sorted_features[:target_features]]
                tprint_info(f"🎯 SELECTED: {len(selected_features)} features from {len(available_features)} available")

                # Log top 10 features by importance
                if len(sorted_features) > 0:
                    tprint_debug("📊 Top 10 features by importance:")
                    for i, (feat, score) in enumerate(sorted_features[:10], 1):
                        tprint_debug(f"  {i}. {feat}: {score:.4f}")
            else:
                # Fallback: select first N features if no importance scores
                selected_features = available_features[:target_features]
                tprint_warning(f"⚠️ No importance scores available, selecting first {target_features} features")

            # Validate selected features using math validation
            if COMMON_OPS_AVAILABLE:
                try:
                    # Ensure we have a reasonable number of features
                    min_features = max(10, target_features // 10)
                    if len(selected_features) < min_features:
                        tprint_warning(f"⚠️ Very few features selected: {len(selected_features)} < {min_features}")
                except Exception as e:
                    tprint_debug(f"Feature validation warning: {e}")

            tprint_success(f"✅ Feature selection for {signal_type} completed: {len(selected_features)}/{len(available_features)} features selected")
            tprint_info(f"🎯 FINAL RESULT: {signal_type} feature selection - {len(available_features)} → {len(selected_features)} features (target: {target_features})")
            return selected_features

        except Exception as e:
            tprint_error(f"❌ Feature selection for {signal_type} failed: {e}")
            return []

    async def _prepare_training_data(
        self,
        tagged_market_data: pd.DataFrame,
        market_data: pd.DataFrame,
        long_pid_features: OrchestratorResult,
        short_pid_features: OrchestratorResult,
        long_targets: Dict[str, np.ndarray],
        short_targets: Dict[str, np.ndarray],
        long_selected_features: List[str],
        short_selected_features: List[str]
    ) -> Dict[str, Any]:
        """Prepare final training data for both signal types using tagged market data."""
        try:
            tprint_info("📚 Preparing final training data for both signal types...")

            result = {
                'long_data': pd.DataFrame(),
                'short_data': pd.DataFrame(),
                'preparation_time': 0.0
            }

            start_time = tprint_timer()

            # Filter tagged market data by signal type for training
            long_tagged_data = tagged_market_data[tagged_market_data['signal_direction'] == 'long'].copy()
            short_tagged_data = tagged_market_data[tagged_market_data['signal_direction'] == 'short'].copy()

            # Prepare long training data
            if not long_tagged_data.empty and long_pid_features and long_targets and long_selected_features:
                try:
                    tprint_info("📈 Preparing long training data...")
                    long_data = await self._prepare_signal_training_data(
                        long_tagged_data, long_pid_features, long_targets,
                        long_selected_features, "long"
                    )
                    result['long_data'] = long_data
                    tprint_success(f"✅ Long training data prepared: {len(long_data)} samples")
                except Exception as e:
                    tprint_error(f"❌ Long training data preparation failed: {e}")
                    result['long_data'] = pd.DataFrame()
            else:
                tprint_info(f"⏭️ Skipping long training data preparation - missing components (data: {len(long_tagged_data)}, features: {long_pid_features is not None}, targets: {len(long_targets)}, selected: {len(long_selected_features)})")

            # Prepare short training data
            if not short_tagged_data.empty and short_pid_features and short_targets and short_selected_features:
                try:
                    tprint_info("📉 Preparing short training data...")
                    short_data = await self._prepare_signal_training_data(
                        short_tagged_data, short_pid_features, short_targets,
                        short_selected_features, "short"
                    )
                    result['short_data'] = short_data
                    tprint_success(f"✅ Short training data prepared: {len(short_data)} samples")
                except Exception as e:
                    tprint_error(f"❌ Short training data preparation failed: {e}")
                    result['short_data'] = pd.DataFrame()
            else:
                tprint_info(f"⏭️ Skipping short training data preparation - missing components (data: {len(short_tagged_data)}, features: {short_pid_features is not None}, targets: {len(short_targets)}, selected: {len(short_selected_features)})")

            result['preparation_time'] = tprint_timer(start_time)
            tprint_performance("Training data preparation", result['preparation_time'])

            return result

        except Exception as e:
            tprint_error(f"❌ Training data preparation failed: {e}")
            return {
                'long_data': pd.DataFrame(),
                'short_data': pd.DataFrame(),
                'preparation_time': tprint_timer(start_time)
            }

    async def _prepare_signal_training_data(
        self,
        signal_data: pd.DataFrame,
        pid_features: OrchestratorResult,
        targets: Dict[str, np.ndarray],
        selected_features: List[str],
        signal_type: str
    ) -> pd.DataFrame:
        """Prepare training data for a specific signal type with index alignment validation."""
        try:
            tprint_info(f"📚 Preparing training data for {signal_type}...")

            # Validate signal_data has an index
            if signal_data.empty:
                tprint_warning(f"⚠️ Empty signal_data for {signal_type}")
                return pd.DataFrame()

            # Create training DataFrame with validated index
            training_df = pd.DataFrame(index=signal_data.index)
            tprint_debug(f"Created training_df with {len(training_df)} rows")

            # Add selected PID features with validation
            features_added = 0
            for feature_name in selected_features:
                if feature_name in pid_features.combined_features:
                    feature_values = pid_features.combined_features[feature_name]

                    # Validate feature length matches training_df
                    if len(feature_values) == len(training_df):
                        training_df[feature_name] = feature_values
                        features_added += 1
                    elif len(feature_values) < len(training_df):
                        # Pad with NaN if shorter
                        padded_values = np.pad(
                            feature_values,
                            (0, len(training_df) - len(feature_values)),
                            mode='constant',
                            constant_values=np.nan
                        )
                        training_df[feature_name] = padded_values
                        features_added += 1
                        tprint_warning(f"⚠️ Feature {feature_name} padded from {len(feature_values)} to {len(training_df)}")
                    else:
                        # Truncate if longer
                        training_df[feature_name] = feature_values[:len(training_df)]
                        features_added += 1
                        tprint_warning(f"⚠️ Feature {feature_name} truncated from {len(feature_values)} to {len(training_df)}")
                else:
                    tprint_debug(f"⚠️ Feature {feature_name} not found in interactive features")

            tprint_debug(f"Added {features_added}/{len(selected_features)} features")

            # Add target variables for different horizons with validation
            targets_added = 0
            for horizon, target_values in targets.items():
                # Validate target length and alignment
                if len(target_values) == len(training_df):
                    training_df[f'target_{horizon}'] = target_values
                    targets_added += 1
                elif len(target_values) < len(training_df):
                    # Pad with zeros if shorter
                    padded_targets = np.pad(
                        target_values,
                        (0, len(training_df) - len(target_values)),
                        mode='constant',
                        constant_values=0
                    )
                    training_df[f'target_{horizon}'] = padded_targets
                    targets_added += 1
                    tprint_warning(f"⚠️ Target {horizon} padded from {len(target_values)} to {len(training_df)}")
                else:
                    # Truncate if longer
                    training_df[f'target_{horizon}'] = target_values[:len(training_df)]
                    targets_added += 1
                    tprint_warning(f"⚠️ Target {horizon} truncated from {len(target_values)} to {len(training_df)}")

            tprint_debug(f"Added {targets_added}/{len(targets)} targets")

            # Add metadata columns with validation
            training_df['signal_type'] = signal_type

            # FIXED: Validate timestamp index alignment before assignment
            if 'timestamp' in signal_data.columns:
                if len(signal_data['timestamp']) == len(training_df):
                    # Align by index to ensure proper mapping
                    training_df['timestamp'] = signal_data['timestamp'].values
                else:
                    tprint_warning(f"⚠️ Timestamp length mismatch: {len(signal_data['timestamp'])} vs {len(training_df)}, using index-based alignment")
                    # Use loc to ensure proper index alignment
                    training_df['timestamp'] = signal_data.loc[training_df.index, 'timestamp'] if 'timestamp' in signal_data.columns else pd.NaT
            else:
                training_df['timestamp'] = pd.NaT
                tprint_debug("⚠️ No timestamp column in signal_data")

            # Add analyst confidence with validation
            if 'analyst_confidence' in signal_data.columns:
                if len(signal_data['analyst_confidence']) == len(training_df):
                    training_df['analyst_confidence'] = signal_data['analyst_confidence'].values
                else:
                    training_df['analyst_confidence'] = signal_data.loc[training_df.index, 'analyst_confidence'] if 'analyst_confidence' in signal_data.columns else 0.0
            else:
                training_df['analyst_confidence'] = 0.0
                tprint_debug("⚠️ No analyst_confidence column in signal_data")

            # Add sample weight based on confidence with validation
            if 'analyst_confidence' in training_df.columns and training_df['analyst_confidence'].max() > 0:
                # Normalize weights to [0, 1]
                max_conf = training_df['analyst_confidence'].max()
                training_df['sample_weight'] = training_df['analyst_confidence'] / max_conf
                # Ensure minimum weight
                training_df['sample_weight'] = training_df['sample_weight'].clip(lower=0.1)
            else:
                training_df['sample_weight'] = 1.0
                tprint_debug("Using uniform sample weights")

            # Validate final training_df
            n_samples = len(training_df)
            n_features = len([c for c in training_df.columns if c not in ['timestamp', 'signal_type', 'sample_weight'] and not c.startswith('target_')])
            n_targets = len([c for c in training_df.columns if c.startswith('target_')])

            # Check for excessive NaN values
            nan_ratio = training_df.isnull().sum().sum() / (training_df.shape[0] * training_df.shape[1])
            if nan_ratio > 0.5:
                tprint_warning(f"⚠️ High NaN ratio in training data: {nan_ratio:.2%}")

            tprint_success(f"✅ Training data prepared for {signal_type}: {n_samples} samples, {n_features} features, {n_targets} targets (NaN ratio: {nan_ratio:.2%})")
            return training_df

        except Exception as e:
            tprint_error(f"❌ Training data preparation for {signal_type} failed: {e}")
            tprint_error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()

    async def _save_intermediate_results(self, result: OrchestratorResult):
        """Save intermediate results to disk."""
        try:
            # Create output directory
            output_dir = Path(self.config.output_directory)
            ensure_directory(output_dir)

            # Save signal separation results
            if not result.long_signals.empty:
                long_path = output_dir / "long_signals.parquet"
                result.long_signals.to_parquet(long_path)
                tprint_debug(f"💾 Saved long signals: {long_path}")

            if not result.short_signals.empty:
                short_path = output_dir / "short_signals.parquet"
                result.short_signals.to_parquet(short_path)
                tprint_debug(f"💾 Saved short signals: {short_path}")

            # Save optimized lookbacks
            if result.long_optimized_lookbacks:
                long_lookback_path = output_dir / "long_optimized_lookbacks.json"
                safe_json_dump(result.long_optimized_lookbacks, long_lookback_path)
                tprint_debug(f"💾 Saved long lookbacks: {long_lookback_path}")

            if result.short_optimized_lookbacks:
                short_lookback_path = output_dir / "short_optimized_lookbacks.json"
                safe_json_dump(result.short_optimized_lookbacks, short_lookback_path)
                tprint_debug(f"💾 Saved short lookbacks: {short_lookback_path}")

            # Save interactive features
            if result.long_pid_features:
                long_pid_path = output_dir / "long_pid_features.json"
                # Convert to serializable format
                long_pid_data = {
                    'combined_features': {k: v.tolist() if hasattr(v, 'tolist') else v
                                        for k, v in result.long_pid_features.combined_features.items()},
                    'combined_feature_names': result.long_pid_features.combined_feature_names,
                    'feature_importance_scores': result.long_pid_features.feature_importance_scores,
                    'total_features_generated': result.long_pid_features.total_features_generated,
                    'execution_time': result.long_pid_features.execution_time,
                    'generation_status': result.long_pid_features.generation_status.value if hasattr(result.long_pid_features.generation_status, 'value') else str(result.long_pid_features.generation_status)
                }
                safe_json_dump(long_pid_data, long_pid_path)
                tprint_debug(f"💾 Saved long interactive features: {long_pid_path}")

            if result.short_pid_features:
                short_pid_path = output_dir / "short_pid_features.json"
                short_pid_data = {
                    'combined_features': {k: v.tolist() if hasattr(v, 'tolist') else v
                                        for k, v in result.short_pid_features.combined_features.items()},
                    'combined_feature_names': result.short_pid_features.combined_feature_names,
                    'feature_importance_scores': result.short_pid_features.feature_importance_scores,
                    'total_features_generated': result.short_pid_features.total_features_generated,
                    'execution_time': result.short_pid_features.execution_time,
                    'generation_status': result.short_pid_features.generation_status.value if hasattr(result.short_pid_features.generation_status, 'value') else str(result.short_pid_features.generation_status)
                }
                safe_json_dump(short_pid_data, short_pid_path)
                tprint_debug(f"💾 Saved short interactive features: {short_pid_path}")

            # Save selected features
            if result.long_selected_features:
                long_features_path = output_dir / "long_selected_features.json"
                safe_json_dump(result.long_selected_features, long_features_path)
                tprint_debug(f"💾 Saved long selected features: {long_features_path}")

            if result.short_selected_features:
                short_features_path = output_dir / "short_selected_features.json"
                safe_json_dump(result.short_selected_features, short_features_path)
                tprint_debug(f"💾 Saved short selected features: {short_features_path}")

            # Save training data
            if not result.long_training_data.empty:
                long_training_path = output_dir / "long_training_data.parquet"
                result.long_training_data.to_parquet(long_training_path)
                tprint_debug(f"💾 Saved long training data: {long_training_path}")

            if not result.short_training_data.empty:
                short_training_path = output_dir / "short_training_data.parquet"
                result.short_training_data.to_parquet(short_training_path)
                tprint_debug(f"💾 Saved short training data: {short_training_path}")

            # Save metadata
            metadata = {
                'execution_time': result.execution_time,
                'total_long_samples': result.total_long_samples,
                'total_short_samples': result.total_short_samples,
                'long_data_quality_score': result.long_data_quality_score,
                'short_data_quality_score': result.short_data_quality_score,
                'signal_separation_completed': result.signal_separation_completed,
                'feature_optimization_completed': result.feature_optimization_completed,
                'pid_generation_completed': result.pid_generation_completed,
                'horizon_labeling_completed': result.horizon_labeling_completed,
                'feature_selection_completed': result.feature_selection_completed,
                'feature_generation_status': result.feature_generation_status,
                'timestamp': datetime.now().isoformat()
            }

            metadata_path = output_dir / "orchestration_metadata.json"
            safe_json_dump(metadata, metadata_path)
            tprint_debug(f"💾 Saved orchestration metadata: {metadata_path}")

            tprint_success(f"✅ All intermediate results saved to {output_dir}")

        except Exception as e:
            tprint_error(f"❌ Failed to save intermediate results: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the orchestrator."""
        try:
            metrics = {
                'config': {
                    'min_analyst_confidence': self.config.min_analyst_confidence,
                    'subsequent_minutes': self.config.subsequent_minutes,
                    'max_lookback_periods': self.config.max_lookback_periods,
                    'max_features': self.config.max_features,
                    'direction_mode': self.config.direction_mode,
                    'separate_directional_features': self.config.separate_directional_features,
                    'output_directory': self.config.output_directory,
                    'enable_data_validation': self.config.enable_data_validation,
                    'enable_lookahead_protection': self.config.enable_lookahead_protection,
                    'enable_quality_gates': self.config.enable_quality_gates,
                    'enable_memory_optimization': self.config.enable_memory_optimization
                },
                'component_availability': {
                    'feature_optimization': self.feature_optimizer is not None,
                    'interactive_generation': self.interactive_generator is not None,
                    'horizon_labeling': self.horizon_labeler is not None,
                    'feature_selection': self.feature_selector is not None,
                    'data_validator': self.data_validator is not None,
                    'lookahead_protection': self.lookahead_protection is not None,
                    'quality_validator': self.quality_validator is not None
                },
                'hardware_optimization': {
                    'gpu_manager': self.gpu_manager is not None,
                    'memory_optimizer': self.memory_optimizer is not None,
                    'cpu_optimizer': self.cpu_optimizer is not None
                },
                'operation_count': self._operation_count,
                'utilities_available': {
                    'tprint': TPRINT_AVAILABLE,
                    'common_ops': COMMON_OPS_AVAILABLE,
                    'data_validation': DATA_VALIDATION_AVAILABLE,
                    'lookahead_protection': LOOKAHEAD_PROTECTION_AVAILABLE,
                    'data_quality': DATA_QUALITY_AVAILABLE,
                    'tpe_optimizer': TPE_OPTIMIZER_AVAILABLE
                }
            }

            return metrics
        except Exception as e:
            tprint_error(f"❌ Error getting performance metrics: {e}")
            return {'error': str(e)}

    def cleanup(self) -> None:
        """Cleanup resources and memory."""
        try:
            tprint_info("🧹 Cleaning up orchestrator resources...")

            # Cleanup hardware optimizers
            if COMMON_OPS_AVAILABLE and self.memory_optimizer:
                try:
                    cleanup_m1_optimizers()
                    tprint_success("✅ Hardware optimizers cleaned up")
                except Exception as e:
                    tprint_warning(f"⚠️ Error cleaning up hardware optimizers: {e}")

            # Clear component references
            self.feature_optimizer = None
            self.interactive_generator = None
            self.horizon_labeler = None
            self.feature_selector = None
            self.data_validator = None
            self.lookahead_protection = None
            self.quality_validator = None

            # Force garbage collection
            import gc
            gc.collect()

            tprint_success("✅ Orchestrator cleanup completed")

        except Exception as e:
            tprint_error(f"❌ Error during cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass  # Silently fail in destructor
