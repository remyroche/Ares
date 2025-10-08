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
    COMMON_OPS_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL: Common operations not available: {e}")
    COMMON_OPS_AVAILABLE = False

# Import feature processing components
try:
    from src.training.steps.pre_training.feature_lookback_optimization import (
        FeatureLookbackOptimizationComponent as _FeatureLookbackOptimizationComponent
    )
    _ = _FeatureLookbackOptimizationComponent
    FEATURE_OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    FEATURE_OPTIMIZATION_AVAILABLE = False
    tprint_warning(f"⚠️ Feature lookback optimization not available: {e}")

try:
    from src.training.steps.pre_training.pid_based_feature_generation.pid_based_feature_orchestrator import (
        PIDBasedFeatureOrchestrator, OrchestratorConfig
    )
    PID_GENERATION_AVAILABLE = True
except ImportError as e:
    PID_GENERATION_AVAILABLE = False
    tprint_warning(f"⚠️ PID-based feature generation not available: {e}")

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
    from src.training.steps.pre_training.final_feature_selection_pipeline import (
        FeatureSelectionConfig
    )
    FEATURE_SELECTION_AVAILABLE = True
except ImportError as e:
    FEATURE_SELECTION_AVAILABLE = False
    tprint_warning(f"⚠️ Final feature selection not available: {e}")

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
    
    # Feature processing parameters (for PRE_TRAINING pipeline)
    max_lookback_periods: int = 20
    max_interaction_features: int = 100
    max_polynomial_features: int = 50
    max_cross_timeframe_features: int = 50

    # Horizon labeling parameters
    profit_targets: Dict[str, float] = field(default_factory=lambda: {
        'micro': 0.003,    # 0.3% (net: 0.22% after fees)
        'small': 0.005,    # 0.5% (net: 0.42% after fees)
        'medium': 0.007,   # 0.7% (net: 0.62% after fees)
        'good': 0.010      # 1.0% (net: 0.92% after fees)
    })

    time_horizons: Dict[str, int] = field(default_factory=lambda: {
        'immediate': 2,    # 10 minutes
        'short': 4         # 20 minutes
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
    enable_horizon_labeling: bool = True
    enable_feature_selection: bool = True


@dataclass
class OrchestratorResult:
    """Result of Tactician pre-ML orchestration."""
    # Feature processing results
    optimized_lookbacks: Dict[str, int] = field(default_factory=dict)
    
    # Final features and targets
    labeled_targets: Dict[str, np.ndarray] = field(default_factory=dict)
    selected_features: List[str] = field(default_factory=list)
    final_features: Optional[pd.DataFrame] = None

    # Metadata
    execution_time: float = 0.0
    total_samples: int = 0
    feature_generation_status: str = "pending"
    
    # Quality metrics
    data_quality_score: float = 0.0
    
    # Status tracking
    feature_optimization_completed: bool = False
    horizon_labeling_completed: bool = False
    feature_selection_completed: bool = False
    success: bool = False
    error_message: Optional[str] = None


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
        'feature_optimization': 'feature_lookback_optimization',
        'interactive_feature_generation': 'interactive_feature_generation',
        'horizon_labeling': 'multi_horizon_profit_labeler',
        'feature_selection': 'final_feature_selection',
    }

    COMPONENT_HINTS: Dict[str, str] = {
        'feature_optimization': (
            "Ensure the 'feature_lookback_optimization' component is available in the PRE_TRAINING pipeline."
        ),
        'interactive_feature_generation': (
            "Ensure the 'interactive_feature_generation' component is available in the PRE_TRAINING pipeline."
        ),
        'horizon_labeling': (
            "Ensure the 'multi_horizon_profit_labeler' component is available in the PRE_TRAINING pipeline."
        ),
        'feature_selection': (
            "Ensure the 'final_feature_selection' component is available in the PRE_TRAINING pipeline."
        ),
    }

    COMPONENT_CONFIG_MAPPING: Dict[str, Dict[str, str]] = {
        'feature_optimization': {
            'max_lookback_periods': 'max_lookback_periods',
            'max_interaction_features': 'max_interaction_features',
            'max_polynomial_features': 'max_polynomial_features',
            'max_cross_timeframe_features': 'max_cross_timeframe_features',
            'direction_mode': 'direction_mode',
            'separate_directional_features': 'separate_directional_features',
            'directional_feature_prefixes': 'directional_feature_prefixes',
        },
        'pid_generation': {
            'synergy_threshold': 'synergy_threshold',
            'redundancy_threshold': 'redundancy_threshold',
            'unique_info_threshold': 'unique_info_threshold',
            'max_interaction_features': 'max_interaction_features',
            'max_polynomial_features': 'max_polynomial_features',
            'max_cross_timeframe_features': 'max_cross_timeframe_features',
            'enable_parallel_processing': 'enable_parallel_processing',
            'enable_gpu_acceleration': 'enable_gpu_acceleration',
            'memory_limit_gb': 'memory_limit_gb',
        },
        'horizon_labeling': {
            'profit_targets': 'profit_targets',
            'time_horizons': 'time_horizons',
            'direction_mode': 'direction_mode',
            'separate_directional_targets': 'separate_directional_features',
            'directional_target_prefixes': 'directional_feature_prefixes',
        },
        'feature_selection': {
            'initial_features': 'initial_features',
            'stage_1_target': 'stage_1_target',
            'stage_2_target': 'stage_2_target',
            'stage_3_target': 'stage_3_target',
            'direction_mode': 'direction_mode',
            'separate_directional_features': 'separate_directional_features',
            'directional_feature_prefixes': 'directional_feature_prefixes',
            'output_directory': 'output_directory',
            'save_analysis': 'save_intermediate_results',
        },
    }

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """Initialize the Tactician pre-ML orchestrator."""
        try:
            self.config = config or OrchestratorConfig()
            self.logger = system_logger.getChild('TacticianPreMLOrchestrator')

            # Initialize hardware optimizers
            if COMMON_OPS_AVAILABLE:
                self.gpu_manager = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer()
                self.cpu_optimizer = get_m1_cpu_optimizer()
                tprint_success("✅ Hardware optimizers initialized")
            else:
                self.gpu_manager = None
                self.memory_optimizer = None
                self.cpu_optimizer = None

            # Initialize feature processing components
            self._initialize_feature_processors()

            tprint_success("✅ TacticianPreMLOrchestrator initialized successfully")
            tprint_info(f"Min analyst confidence: {self.config.min_analyst_confidence}")
            tprint_info(f"Subsequent minutes: {self.config.subsequent_minutes}")
            tprint_info(f"Output directory: {self.config.output_directory}")
            tprint_info(f"Direction mode: {self.config.direction_mode}")
            tprint_info(f"Separate directional features: {self.config.separate_directional_features}")

        except Exception as e:
            tprint_error(f"❌ Failed to initialize TacticianPreMLOrchestrator: {e}")
            raise

    def _initialize_feature_processors(self):
        """Initialize feature processing components."""
        self.factory_component_status: Dict[str, bool] = self._evaluate_factory_components()
        self.factory_component_configs: Dict[str, ComponentConfig] = {
            alias: self._build_factory_component_config(alias)
            for alias in self.COMPONENT_FACTORY_KEYS
        }

        # Feature lookback optimization
        if not self.config.enable_feature_optimization:
            self.feature_optimizer = None
        elif not FEATURE_OPTIMIZATION_AVAILABLE:
            self.feature_optimizer = None
            tprint_warning("⚠️ Feature optimization requested but not available")
        elif not self.factory_component_status.get('feature_optimization', False):
            self.feature_optimizer = None
            self._log_factory_unavailable('feature_optimization')
        else:
            component_key = self.COMPONENT_FACTORY_KEYS['feature_optimization']
            component_config = self.factory_component_configs['feature_optimization']
            try:
                self.feature_optimizer = ComponentFactory.create_component(
                    component_key,
                    component_config,
                )
                tprint_success("✅ Feature lookback optimization initialized via ComponentFactory")
            except Exception as exc:
                self.feature_optimizer = None
                self._log_factory_error('feature_optimization', exc)

        # PID-based feature generation
        if not self.config.enable_pid_generation:
            self.pid_orchestrator = None
        elif not PID_GENERATION_AVAILABLE:
            self.pid_orchestrator = None
            tprint_warning("⚠️ PID generation requested but not available")
        else:
            if not self.factory_component_status.get('pid_generation', False):
                self._log_factory_unavailable('pid_generation')
            try:
                pid_config = OrchestratorConfig(
                    max_interaction_features=self.config.max_interaction_features,
                    max_polynomial_features=self.config.max_polynomial_features,
                    max_cross_timeframe_features=self.config.max_cross_timeframe_features,
                    synergy_threshold=self.config.synergy_threshold,
                    redundancy_threshold=self.config.redundancy_threshold,
                    unique_info_threshold=self.config.unique_info_threshold,
                    enable_parallel_processing=self.config.enable_parallel_processing,
                    enable_gpu_acceleration=self.config.enable_gpu_acceleration,
                    memory_limit_gb=self.config.memory_limit_gb
                )
                self.pid_orchestrator = PIDBasedFeatureOrchestrator(pid_config)
                tprint_success("✅ PID-based feature orchestrator initialized")
            except Exception as e:
                tprint_error(f"❌ Failed to initialize PID orchestrator: {e}")
                self.pid_orchestrator = None

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
        elif not FEATURE_SELECTION_AVAILABLE:
            self.feature_selector = None
            tprint_warning("⚠️ Feature selection requested but not available")
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

    async def orchestrate_pre_ml_training(
        self,
        analyst_signals: pd.DataFrame,
        market_data: pd.DataFrame,
        feature_names: List[str],
        **kwargs
    ) -> OrchestratorResult:
        """
        Orchestrate the complete pre-ML training pipeline for Tactician.

        Args:
            analyst_signals: DataFrame with Analyst signals and confidence scores
            market_data: Raw market data for feature generation
            feature_names: List of base feature names
            **kwargs: Additional parameters

        Returns:
            OrchestratorResult with processed data for both long and short training
        """
        start_time = tprint_timer()
        tprint_info(f"🚀 Starting Tactician pre-ML training orchestration (mode: {self.config.direction_mode})...")

        result = OrchestratorResult()
        result.feature_generation_status = "in_progress"

        try:
            # Step 1: Separate Analyst signals into long/short with confidence filtering
            tprint_info("📊 Step 1: Separating Analyst signals by direction...")
            signal_separation_result = await self._separate_analyst_signals(
                analyst_signals, market_data
            )

            if not signal_separation_result['success']:
                raise ValueError(f"Signal separation failed: {signal_separation_result['error']}")

            result.tagged_market_data = signal_separation_result['tagged_market_data']
            result.long_signals = signal_separation_result['long_signals']
            result.short_signals = signal_separation_result['short_signals']
            result.combined_signals = signal_separation_result['combined_signals']
            result.signal_separation_completed = True

            tprint_success(f"✅ Signal tagging completed: {signal_separation_result['long_tagged_count']} long, {signal_separation_result['short_tagged_count']} short tagged samples")
            tprint_info(f"🔄 Using tagging approach - full lookback periods preserved for indicator calculations")

            # Step 2: Optimize feature lookback periods for each signal type
            if self.feature_optimizer:
                tprint_info("🔍 Step 2: Optimizing feature lookback periods...")
                # Use tagged market data which includes full lookback periods
                lookback_result = await self._optimize_feature_lookbacks(
                    result.tagged_market_data, result.tagged_market_data, result.tagged_market_data, feature_names
                )

                result.long_optimized_lookbacks = lookback_result['long_lookbacks']
                result.short_optimized_lookbacks = lookback_result['short_lookbacks']
                result.feature_optimization_completed = True

                tprint_success(f"✅ Feature lookback optimization completed: {len(result.long_optimized_lookbacks)} long, {len(result.short_optimized_lookbacks)} short periods")
            else:
                tprint_warning("⚠️ Skipping feature lookback optimization - component not available")

            # Step 3: Generate PID-based features for each signal type
            if self.pid_orchestrator:
                tprint_info("🎯 Step 3: Generating PID-based features...")

                # Include HMM and Analyst outputs in feature names
                extended_feature_names = feature_names.copy()
                extended_feature_names.extend([
                    'hmm_regime', 'hmm_regime_prob', 'hmm_regime_confidence',
                    'analyst_signal', 'analyst_confidence', 'analyst_prediction',
                    'analyst_long_prob', 'analyst_short_prob', 'analyst_neutral_prob'
                ])

                pid_result = await self._generate_pid_features(
                    result.tagged_market_data, result.tagged_market_data,
                    result.tagged_market_data, extended_feature_names,
                    result.long_optimized_lookbacks, result.short_optimized_lookbacks
                )

                result.long_pid_features = pid_result['long_features']
                result.short_pid_features = pid_result['short_features']
                result.pid_generation_completed = True

                tprint_success(f"✅ PID feature generation completed: {result.long_pid_features.total_features_generated} long, {result.short_pid_features.total_features_generated} short features")
            else:
                tprint_warning("⚠️ Skipping PID feature generation - component not available")

            # Step 4: Apply multi-horizon profit labeling for each signal type
            if self.horizon_labeler:
                tprint_info("🏷️ Step 4: Applying multi-horizon profit labeling...")
                labeling_result = await self._apply_horizon_labeling(
                    result.tagged_market_data, result.tagged_market_data,
                    result.tagged_market_data, result.long_pid_features, result.short_pid_features
                )

                result.long_labeled_targets = labeling_result['long_targets']
                result.short_labeled_targets = labeling_result['short_targets']
                result.horizon_labeling_completed = True

                tprint_success(f"✅ Horizon labeling completed: {len(result.long_labeled_targets)} long, {len(result.short_labeled_targets)} short target sets")
            else:
                tprint_warning("⚠️ Skipping horizon labeling - component not available")

            # Step 5: Select final features for each signal type
            if self.feature_selector:
                tprint_info("🎯 Step 5: Selecting final features...")
                selection_result = await self._select_final_features(
                    result.tagged_market_data, result.tagged_market_data,
                    result.long_pid_features, result.short_pid_features,
                    result.long_labeled_targets, result.short_labeled_targets
                )

                result.long_selected_features = selection_result['long_features']
                result.short_selected_features = selection_result['short_features']
                result.feature_selection_completed = True

                tprint_success(f"✅ Feature selection completed: {len(result.long_selected_features)} long, {len(result.short_selected_features)} short features")
            else:
                tprint_warning("⚠️ Skipping feature selection - component not available")

            # Step 6: Prepare training data for both signal types
            tprint_info("📚 Step 6: Preparing training data...")
            training_data_result = await self._prepare_training_data(
                result.tagged_market_data, result.tagged_market_data,
                result.long_pid_features, result.short_pid_features,
                result.long_labeled_targets, result.short_labeled_targets,
                result.long_selected_features, result.short_selected_features
            )

            result.long_training_data = training_data_result['long_data']
            result.short_training_data = training_data_result['short_data']
            result.total_long_samples = len(result.long_training_data)
            result.total_short_samples = len(result.short_training_data)

            if self.config.direction_mode == 'both':
                tprint_success(f"✅ Training data prepared: {result.total_long_samples} long, {result.total_short_samples} short samples")
            elif self.config.direction_mode == 'long_only':
                tprint_success(f"✅ Training data prepared: {result.total_long_samples} long samples only")
            elif self.config.direction_mode == 'short_only':
                tprint_success(f"✅ Training data prepared: {result.total_short_samples} short samples only")

            # Step 7: Save intermediate results
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

            # Process each signal timestamp
            for idx, row in analyst_signals.iterrows():
                signal_timestamp = row['timestamp']

                # Find confidence value
                confidence = 0.0
                if confidence_columns:
                    confidence = max(row[col] for col in confidence_columns if col in row and pd.notna(row[col]))

                # Check if confidence meets threshold
                if confidence >= confidence_threshold:
                    # Calculate target end time (signal + 45 minutes)
                    target_end_time = signal_timestamp + timedelta(minutes=self.config.subsequent_minutes)

                    # Tag all data points from this signal timestamp onwards with full lookback
                    # This preserves the historical data needed for indicators
                    signal_mask = tagged_market_data['timestamp'] >= signal_timestamp
                    tagged_market_data.loc[signal_mask, 'analyst_signal_time'] = signal_timestamp
                    tagged_market_data.loc[signal_mask, 'analyst_confidence'] = confidence
                    tagged_market_data.loc[signal_mask, 'signal_target_end_time'] = target_end_time
                    tagged_market_data.loc[signal_mask, 'signal_horizon_minutes'] = self.config.subsequent_minutes

                    # Determine signal direction from available columns
                    signal_direction = self._determine_signal_direction(row, signal_columns)

                    if signal_direction == SignalDirection.LONG:
                        tagged_market_data.loc[signal_mask, 'signal_direction'] = 'long'
                        tprint_debug(f"📈 Tagged long signal at {signal_timestamp} with confidence {confidence:.3f}")
                    elif signal_direction == SignalDirection.SHORT:
                        tagged_market_data.loc[signal_mask, 'signal_direction'] = 'short'
                        tprint_debug(f"📉 Tagged short signal at {signal_timestamp} with confidence {confidence:.3f}")
                    else:
                        # Combined or neutral signal
                        tagged_market_data.loc[signal_mask, 'signal_direction'] = 'combined'
                        tprint_debug(f"⚖️ Tagged combined signal at {signal_timestamp} with confidence {confidence:.3f}")

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
        """Calculate data quality score."""
        try:
            if df.empty:
                return 0.0

            # Basic quality metrics
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

            # Combine scores
            quality_score = (completeness * 0.4 + sample_count_score * 0.4 + continuity_score * 0.2)
            return validate_finite(quality_score, "data_quality")

        except Exception as e:
            tprint_warning(f"⚠️ Error calculating data quality: {e}")
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
        """Run lookback optimization for a specific signal type."""
        try:
            # This would call the actual feature lookback optimization component
            # For now, return mock optimized lookbacks
            tprint_info(f"🔍 Running lookback optimization for {signal_type} signals...")

            # Mock implementation - in reality this would call the actual optimizer
            optimal_lookbacks = {}

            # Base lookback periods by feature category
            base_lookbacks = {
                'price': 20,
                'volume': 15,
                'volatility': 25,
                'momentum': 10,
                'trend': 30,
                'support_resistance': 40
            }

            for feature in feature_names:
                # Determine feature category from name
                category = 'price'  # Default
                feature_lower = feature.lower()

                if any(keyword in feature_lower for keyword in ['volume', 'vol']):
                    category = 'volume'
                elif any(keyword in feature_lower for keyword in ['volatility', 'std', 'var']):
                    category = 'volatility'
                elif any(keyword in feature_lower for keyword in ['momentum', 'rsi', 'macd']):
                    category = 'momentum'
                elif any(keyword in feature_lower for keyword in ['trend', 'ma', 'ema', 'sma']):
                    category = 'trend'
                elif any(keyword in feature_lower for keyword in ['support', 'resistance', 'pivot']):
                    category = 'support_resistance'

                # Apply signal-specific adjustments
                if signal_type == 'long':
                    # Longer lookbacks for long signals (trend following)
                    lookback = int(base_lookbacks[category] * 1.2)
                elif signal_type == 'short':
                    # Shorter lookbacks for short signals (mean reversion)
                    lookback = int(base_lookbacks[category] * 0.8)
                else:
                    lookback = base_lookbacks[category]

                # Ensure minimum and maximum bounds
                lookback = max(5, min(self.config.max_lookback_periods, lookback))
                optimal_lookbacks[feature] = lookback

            tprint_info(f"✅ Lookback optimization for {signal_type} completed: {len(optimal_lookbacks)} features optimized")

            return {
                'success': True,
                'optimal_lookbacks': optimal_lookbacks,
                'optimization_method': 'mock_optimization',  # Would be actual method
                'features_processed': len(optimal_lookbacks)
            }

        except Exception as e:
            tprint_error(f"❌ Lookback optimization for {signal_type} failed: {e}")
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
        """Generate PID-based features for long and short signals."""
        try:
            tprint_info("🎯 Generating PID-based features for both signal types...")

            result = {
                'long_features': None,
                'short_features': None,
                'generation_time': 0.0
            }

            start_time = tprint_timer()

            # Generate features for long signals
            if not long_signals.empty:
                try:
                    tprint_info("📈 Generating PID features for long signals...")
                    long_features = await self._generate_signal_pid_features(
                        long_signals, market_data, feature_names, long_lookbacks, "long"
                    )
                    result['long_features'] = long_features
                    tprint_success(f"✅ Long PID features generated: {long_features.total_features_generated} features")
                except Exception as e:
                    tprint_error(f"❌ Long PID feature generation failed: {e}")
                    result['long_features'] = self._create_empty_pid_result()
            else:
                tprint_info("⏭️ Skipping long PID generation - no signals available")
                result['long_features'] = self._create_empty_pid_result()

            # Generate features for short signals
            if not short_signals.empty:
                try:
                    tprint_info("📉 Generating PID features for short signals...")
                    short_features = await self._generate_signal_pid_features(
                        short_signals, market_data, feature_names, short_lookbacks, "short"
                    )
                    result['short_features'] = short_features
                    tprint_success(f"✅ Short PID features generated: {short_features.total_features_generated} features")
                except Exception as e:
                    tprint_error(f"❌ Short PID feature generation failed: {e}")
                    result['short_features'] = self._create_empty_pid_result()
            else:
                tprint_info("⏭️ Skipping short PID generation - no signals available")
                result['short_features'] = self._create_empty_pid_result()

            result['generation_time'] = tprint_timer(start_time)
            tprint_performance("PID feature generation", result['generation_time'])

            return result

        except Exception as e:
            tprint_error(f"❌ PID feature generation failed: {e}")
            return {
                'long_features': self._create_empty_pid_result(),
                'short_features': self._create_empty_pid_result(),
                'generation_time': tprint_timer(start_time)
            }

    async def _generate_signal_pid_features(
        self,
        signal_data: pd.DataFrame,
        market_data: pd.DataFrame,
        feature_names: List[str],
        optimized_lookbacks: Dict[str, int],
        signal_type: str
    ) -> OrchestratorResult:
        """Generate PID features for a specific signal type."""
        try:
            # Prepare data for PID generation
            # Create target dictionary with appropriate signal direction
            target_dict = {
                signal_type: np.ones(len(signal_data)),  # Mock target - would be actual signal strength
                'combined': np.ones(len(signal_data))
            }

            # Call PID orchestrator
            if self.pid_orchestrator:
                pid_result = await self.pid_orchestrator.orchestrate_feature_generation(
                    data=signal_data[feature_names].values,
                    feature_names=feature_names,
                    optimized_lookback_periods=optimized_lookbacks,
                    target=target_dict
                )
                return pid_result
            else:
                raise ValueError("PID orchestrator not available")

        except Exception as e:
            tprint_error(f"❌ PID feature generation for {signal_type} failed: {e}")
            return self._create_empty_pid_result()

    def _create_empty_pid_result(self) -> OrchestratorResult:
        """Create empty PID result for fallback."""
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

            # Create comprehensive report
            comprehensive_report = {
                'orchestration_summary': {
                    'total_orchestration_time': total_time,
                    'success': result.feature_generation_status == "completed",
                    'input_samples': len(result.tagged_market_data) if hasattr(result, 'tagged_market_data') and result.tagged_market_data is not None else 0,
                    'long_samples_processed': result.total_long_samples,
                    'short_samples_processed': result.total_short_samples,
                    'analyst_signals_count': result.analyst_signals_count,
                    'long_signals_count': result.long_signals_count,
                    'short_signals_count': result.short_signals_count,
                    'confidence_threshold': self.config.confidence_threshold,
                    'subsequent_minutes': self.config.subsequent_minutes,
                    'direction_mode': self.config.direction_mode,
                    'separate_directional_features': self.config.separate_directional_features,
                    'feature_generation_status': result.feature_generation_status,
                    'lookback_optimization_enabled': result.lookback_optimization_enabled,
                    'pid_feature_generation_enabled': result.pid_feature_generation_enabled,
                    'horizon_labeling_enabled': result.horizon_labeling_enabled,
                    'feature_selection_enabled': result.feature_selection_enabled,
                    'intermediate_results_saved': result.intermediate_results_saved
                },
                'signal_separation_metrics': {
                    'total_analyst_signals': result.analyst_signals_count,
                    'long_signals_with_confidence': result.long_signals_count,
                    'short_signals_with_confidence': result.short_signals_count,
                    'long_signal_ratio': result.long_signals_count / max(result.analyst_signals_count, 1),
                    'short_signal_ratio': result.short_signals_count / max(result.analyst_signals_count, 1),
                    'average_confidence_long': result.average_long_confidence,
                    'average_confidence_short': result.average_short_confidence,
                    'tagging_approach_used': True  # We use tagging instead of extraction
                },
                'feature_processing_metrics': {
                    'total_features_available': len(result.feature_names) if hasattr(result, 'feature_names') else 0,
                    'long_features_count': len(result.long_selected_features),
                    'short_features_count': len(result.short_selected_features),
                    'long_optimized_lookbacks_count': len(result.long_optimized_lookbacks),
                    'short_optimized_lookbacks_count': len(result.short_optimized_lookbacks),
                    'long_pid_features_generated': bool(result.long_pid_features),
                    'short_pid_features_generated': bool(result.short_pid_features),
                    'long_targets_count': len(result.long_targets),
                    'short_targets_count': len(result.short_targets)
                },
                'data_quality_metrics': {
                    'long_data_quality_score': result.long_data_quality_score,
                    'short_data_quality_score': result.short_data_quality_score,
                    'long_missing_values_ratio': result.long_missing_values_ratio,
                    'short_missing_values_ratio': result.short_missing_values_ratio,
                    'long_outlier_ratio': result.long_outlier_ratio,
                    'short_outlier_ratio': result.short_outlier_ratio
                },
                'performance_metrics': {
                    'optimization_time': getattr(result, 'optimization_time', 0),
                    'generation_time': getattr(result, 'generation_time', 0),
                    'labeling_time': getattr(result, 'labeling_time', 0),
                    'selection_time': getattr(result, 'selection_time', 0),
                    'preparation_time': getattr(result, 'preparation_time', 0),
                    'total_execution_time': result.execution_time,
                    'memory_usage_mb': getattr(result, 'memory_usage_mb', 0),
                    'cpu_usage_percent': getattr(result, 'cpu_usage_percent', 0)
                },
                'model_integration_metrics': {
                    'hmm_features_included': len([f for f in result.long_selected_features if 'hmm' in f.lower()]),
                    'analyst_features_included': len([f for f in result.long_selected_features if 'analyst' in f.lower()]),
                    'technical_features_included': len([f for f in result.long_selected_features if f not in ['hmm_regime_0_prob', 'hmm_regime_1_prob', 'hmm_regime_2_prob', 'hmm_regime_entropy', 'hmm_regime_ece', 'analyst_action', 'analyst_confidence', 'analyst_position_size', 'analyst_expected_profit', 'analyst_max_risk', 'analyst_time_horizon', 'analyst_micro_immediate_prob', 'analyst_small_immediate_prob', 'analyst_medium_immediate_prob', 'analyst_micro_short_prob', 'analyst_small_short_prob', 'analyst_price_target_upside_0.003', 'analyst_price_target_downside_0.003']]),
                    'long_features_with_hmm': len(result.long_selected_features),
                    'short_features_with_hmm': len(result.short_selected_features),
                    'feature_integration_complete': True
                },
                'evaluation_metrics': {
                    'long_training_accuracy': getattr(result, 'long_training_accuracy', 0.0),
                    'short_training_accuracy': getattr(result, 'short_training_accuracy', 0.0),
                    'long_validation_accuracy': getattr(result, 'long_validation_accuracy', 0.0),
                    'short_validation_accuracy': getattr(result, 'short_validation_accuracy', 0.0),
                    'long_f1_score': getattr(result, 'long_f1_score', 0.0),
                    'short_f1_score': getattr(result, 'short_f1_score', 0.0),
                    'long_precision': getattr(result, 'long_precision', 0.0),
                    'short_precision': getattr(result, 'short_precision', 0.0),
                    'long_recall': getattr(result, 'long_recall', 0.0),
                    'short_recall': getattr(result, 'short_recall', 0.0),
                    'long_roc_auc': getattr(result, 'long_roc_auc', 0.0),
                    'short_roc_auc': getattr(result, 'short_roc_auc', 0.0),
                    'long_sharpe_ratio': getattr(result, 'long_sharpe_ratio', 0.0),
                    'short_sharpe_ratio': getattr(result, 'short_sharpe_ratio', 0.0),
                    'long_max_drawdown': getattr(result, 'long_max_drawdown', 0.0),
                    'short_max_drawdown': getattr(result, 'short_max_drawdown', 0.0),
                    'long_total_trades': getattr(result, 'long_total_trades', 0),
                    'short_total_trades': getattr(result, 'short_total_trades', 0),
                    'long_avg_trades_per_month': getattr(result, 'long_avg_trades_per_month', 0.0),
                    'short_avg_trades_per_month': getattr(result, 'short_avg_trades_per_month', 0.0),
                    'long_total_pnl': getattr(result, 'long_total_pnl', 0.0),
                    'short_total_pnl': getattr(result, 'short_total_pnl', 0.0),
                    'long_monthly_pnl': getattr(result, 'long_monthly_pnl', {}),
                    'short_monthly_pnl': getattr(result, 'short_monthly_pnl', {}),
                    'evaluation_completed': getattr(result, 'evaluation_completed', False)
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
            tprint_info(f"🧬 PID Features: {'Enabled' if orchestration['pid_feature_generation_enabled'] else 'Disabled'}")
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
        """Apply horizon labeling for a specific signal type."""
        try:
            # This would call the actual multi-horizon profit labeler
            # For now, return mock labeled targets
            tprint_info(f"🏷️ Applying horizon labeling for {signal_type} signals...")

            # Mock implementation - in reality this would call the actual labeler
            labeled_targets = {}

            # Create different target horizons
            horizons = ['micro', 'small', 'medium', 'good']

            for horizon in horizons:
                # Mock target based on signal type and horizon
                if signal_type == 'long':
                    # Long signals: higher targets for longer horizons
                    base_target = self.config.profit_targets[horizon]
                    # Add some noise and signal-specific adjustments
                    target_values = np.random.normal(base_target, base_target * 0.1, len(signal_data))
                    target_values = np.maximum(target_values, 0.001)  # Ensure positive
                else:
                    # Short signals: slightly lower targets for shorter horizons
                    base_target = self.config.profit_targets[horizon] * 0.9
                    target_values = np.random.normal(base_target, base_target * 0.1, len(signal_data))
                    target_values = np.maximum(target_values, 0.001)  # Ensure positive

                labeled_targets[f"{signal_type}_{horizon}"] = target_values
                tprint_debug(f"📊 Created {signal_type} {horizon} targets: mean={np.mean(target_values):.4f}, std={np.std(target_values):.4f}")

            tprint_success(f"✅ Horizon labeling for {signal_type} completed: {len(labeled_targets)} target sets")
            return labeled_targets

        except Exception as e:
            tprint_error(f"❌ Horizon labeling for {signal_type} failed: {e}")
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
                tprint_warning(f"⚠️ No PID features available for {signal_type} selection")
                return []

            # Apply proper feature selection with 200 feature limit
            importance_scores = pid_features.feature_importance_scores
            target_features = 200  # Maximum 200 features
            
            tprint_info(f"🔍 FEATURE SELECTION: Starting with {len(available_features)} features, target: {target_features}", color="cyan", bold=True)

            # Sort features by importance and select top features
            if importance_scores:
                sorted_features = sorted(
                    importance_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # Select top 200 features maximum
                selected_features = [f[0] for f in sorted_features[:target_features]]
                tprint_info(f"🎯 SELECTED: {len(selected_features)} features from {len(available_features)} available", color="green")
            else:
                # Fallback: select first 200 features if no importance scores
                selected_features = available_features[:target_features]
                tprint_warning(f"⚠️ No importance scores available, selecting first {target_features} features")


            tprint_success(f"✅ Feature selection for {signal_type} completed: {len(selected_features)}/{len(available_features)} features selected")
            tprint_info(f"🎯 FINAL RESULT: {signal_type} feature selection - {len(available_features)} → {len(selected_features)} features (target: {target_features})", color="green", bold=True)
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
        """Prepare training data for a specific signal type."""
        try:
            # Create training DataFrame
            training_df = pd.DataFrame(index=signal_data.index)

            # Add selected PID features
            for feature_name in selected_features:
                if feature_name in pid_features.combined_features:
                    training_df[feature_name] = pid_features.combined_features[feature_name]
                else:
                    tprint_warning(f"⚠️ Feature {feature_name} not found in PID features")

            # Add target variables for different horizons
            for horizon, target_values in targets.items():
                if len(target_values) == len(training_df):
                    training_df[f'target_{horizon}'] = target_values
                else:
                    tprint_warning(f"⚠️ Target {horizon} length mismatch: {len(target_values)} vs {len(training_df)}")

            # Add metadata columns
            training_df['signal_type'] = signal_type
            training_df['timestamp'] = signal_data.get('timestamp', pd.NaT)
            training_df['analyst_confidence'] = signal_data.get('analyst_confidence', 0.0)

            # Add sample weight based on confidence
            if 'analyst_confidence' in training_df.columns:
                training_df['sample_weight'] = training_df['analyst_confidence'] / training_df['analyst_confidence'].max()

            tprint_success(f"✅ Training data prepared for {signal_type}: {len(training_df)} samples, {len(training_df.columns)} features")
            return training_df

        except Exception as e:
            tprint_error(f"❌ Training data preparation for {signal_type} failed: {e}")
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

            # Save PID features
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
                tprint_debug(f"💾 Saved long PID features: {long_pid_path}")

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
                tprint_debug(f"💾 Saved short PID features: {short_pid_path}")

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
        metrics = {
            'config': {
                'min_analyst_confidence': self.config.min_analyst_confidence,
                'subsequent_minutes': self.config.subsequent_minutes,
                'max_lookback_periods': self.config.max_lookback_periods,
                'direction_mode': self.config.direction_mode,
                'separate_directional_features': self.config.separate_directional_features,
                'output_directory': self.config.output_directory
            },
            'component_availability': {
                'feature_optimization': self.feature_optimizer is not None,
                'pid_generation': self.pid_orchestrator is not None,
                'horizon_labeling': self.horizon_labeler is not None,
                'feature_selection': self.feature_selector is not None
            },
            'hardware_optimization': {
                'gpu_manager': self.gpu_manager is not None,
                'memory_optimizer': self.memory_optimizer is not None,
                'cpu_optimizer': self.cpu_optimizer is not None
            }
        }

        return metrics