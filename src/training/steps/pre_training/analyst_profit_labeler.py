"""
Analyst Profit Labeler - Specialized Multi-Horizon Labeling for Analyst Models

This module provides a specialized profit labeling component for Analyst models,
using the VolatilityAwareMultiHorizonLabeler with Analyst-specific configurations.

Key Features:
- 60m timeframe optimization for strategic decision-making
- Multi-horizon profit labeling (1h, 4h, 12h, 24h horizons)
- Volatility-aware target bands
- Enhanced label quality scoring
- Per-regime/cluster optimization support
"""

import asyncio
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.logger import system_logger
from src.utils.common_operations import (
    validate_dataframe_columns,
    safe_dataframe_operation,
    validate_positive,
    validate_range,
    safe_int,
    safe_float,
    get_dataframe_info,
    create_data_quality_report,
    ensure_directory,
    safe_json_dump,
    safe_json_load,
    format_bytes,
    timed_operation,
    memory_checkpoint,
    optimize_memory,
    check_disk_space,
    integrate_with_m1_optimizers,
    get_m1_gpu_manager,
    get_m1_memory_optimizer
)
from src.utils.common_utilities import (
    analyze_nan_values_detailed,
    format_nan_analysis_report,
    create_data_quality_report as create_detailed_quality_report,
    get_dataframe_info as get_detailed_dataframe_info
)
from src.utils.matrix_operations import (
    get_unified_matrix_operations,
    get_vectorized_processing_core,
    get_enhanced_matrix_operations,
    optimize_dataframe,
    vectorized_rolling_features,
    matrix_correlation_analysis,
    safe_correlation_matrix,
    compute_trading_indicators,
    get_hardware_performance_report
)
from src.utils.ml_common.optimization.grid_utils import (
    generate_grid,
    build_coarse_grid_from_search_space,
    GridSearchOptimizer
)
from src.training.steps.pre_training.components.base_component import BasePreTrainingComponent, ComponentConfig, ComponentResult
from src.training.steps.pre_training.components.contracts import PipelineState
from src.training.steps.pre_training.components.component_factory import register_component
from src.training.steps.pre_training.validation.schemas import validate_raw_ohlcv, SchemaValidationException

# Import the volatility-aware labeler
try:
    from src.training.steps.pre_training.profit_labeling.volatility_aware_labeler import (
        VolatilityAwareMultiHorizonLabeler,
        VolatilityAwareConfig,
        LabelingResult,
        LabelDefinitionType,
        create_enhanced_analyst_labeler,
    )
    VOLATILITY_LABELER_AVAILABLE = True
except (ImportError, SyntaxError):
    VOLATILITY_LABELER_AVAILABLE = False
    VolatilityAwareMultiHorizonLabeler = None
    VolatilityAwareConfig = None
    LabelingResult = None
    LabelDefinitionType = None
    create_enhanced_analyst_labeler = None

# Import advanced filters for 15m timeframe
try:
    from src.training.steps.pre_training.profit_labeling.advanced_filters_15m import (
        AdvancedFilters15m,
        AdvancedFiltersConfig,
        FilterResult,
        apply_advanced_filters_15m,
    )
    ADVANCED_FILTERS_AVAILABLE = True
except (ImportError, SyntaxError):
    ADVANCED_FILTERS_AVAILABLE = False
    AdvancedFilters15m = None
    AdvancedFiltersConfig = None
    FilterResult = None
    apply_advanced_filters_15m = None


@dataclass
class AnalystProfitLabelerConfig:
    """Configuration for Analyst profit labeling."""

    # Timeframe settings (Analyst operates on 15m by default)
    timeframe: str = "15m"
    base_period_minutes: int = 15

    # Horizon settings for Analyst (strategic decision-making)
    # Horizons are in MINUTES (must be >= timeframe period)
    # Updated for 15m timeframe: 1h, 2h, 4h, 6h, 12h, 24h
    horizons: List[int] = field(default_factory=lambda: [60, 120, 240, 360, 720, 1440])  # 1h, 2h, 4h, 6h, 12h, 24h in minutes

    # Profit targets (percentage) - Realistic for hourly crypto movements after fees
    # ETH typically moves 0.4% per hour on average (0.26% median)
    # Starting at 0.5% provides buffer above typical trading costs (~0.1% roundtrip)
    target_profits: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0])

    # Volatility-aware settings
    # Disable volatility normalization for simpler percentage-based targets
    use_volatility_normalization: bool = False
    volatility_window: int = 20

    # Label quality thresholds
    min_label_quality: float = 0.6
    min_predictability: float = 0.55

    # Per-regime optimization
    enable_regime_adaptation: bool = True
    
    # Trading direction settings
    enable_long_positions: bool = True   # Include long opportunities (buy when expecting price increase)
    enable_short_positions: bool = False  # Include short opportunities (sell when expecting price decrease)

    # Advanced filtering for 15m timeframe
    enable_advanced_filters: bool = True
    advanced_filters_config: Optional[AdvancedFiltersConfig] = None

    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_horizon_timeframe_compatibility()

    def _validate_horizon_timeframe_compatibility(self) -> None:
        """Validate that horizons are compatible with the timeframe."""
        if not self.horizons:
            raise ValueError("Horizons list cannot be empty")

        # Parse timeframe to get base period in minutes
        timeframe_minutes = self._parse_timeframe_to_minutes(self.timeframe)

        # Check each horizon
        problematic_horizons = []
        for horizon in self.horizons:
            if horizon < timeframe_minutes:
                problematic_horizons.append((horizon, timeframe_minutes))

        if problematic_horizons:
            horizon_strs = [f"{h}m (horizon) vs {tf}m (timeframe)" for h, tf in problematic_horizons]
            raise ValueError(
                f"Horizon(s) incompatible with timeframe '{self.timeframe}': {', '.join(horizon_strs)}. "
                "Horizon must be >= timeframe period to ensure sufficient data for labeling."
            )

    def _parse_timeframe_to_minutes(self, timeframe: str) -> int:
        """Parse timeframe string to minutes."""
        try:
            if timeframe.endswith('m'):
                return safe_int(timeframe[:-1])
            elif timeframe.endswith('h'):
                hours = safe_int(timeframe[:-1])
                validate_positive(hours, "hours in timeframe")
                return hours * 60
            elif timeframe.endswith('d'):
                days = safe_int(timeframe[:-1])
                validate_positive(days, "days in timeframe")
                return days * 60 * 24
            else:
                raise ValueError(f"Unsupported timeframe format: {timeframe}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid timeframe format '{timeframe}': {e}")

    def get_optimization_search_space(self) -> Dict[str, Any]:
        """Get search space for hyperparameter optimization."""
        return {
            'target_profits': {
                'type': 'float',
                'low': 0.5,
                'high': 5.0,
                'log': False
            },
            'volatility_window': {
                'type': 'int',
                'low': 10,
                'high': 50
            },
            'min_label_quality': {
                'type': 'float',
                'low': 0.4,
                'high': 0.8,
                'log': False
            },
            'min_predictability': {
                'type': 'float',
                'low': 0.4,
                'high': 0.8,
                'log': False
            }
        }

    def optimize_config_grid_search(self, data: pd.DataFrame, max_trials: int = 50) -> 'AnalystProfitLabelerConfig':
        """Optimize configuration using grid search."""
        search_space = self.get_optimization_search_space()

        # Generate parameter grid
        param_grid = generate_grid(search_space, max_trials)

        best_config = None
        best_score = -float('inf')

        # Simple evaluation based on data characteristics
        for params in param_grid[:max_trials]:
            try:
                # Create config with current parameters
                config = AnalystProfitLabelerConfig(
                    horizons=self.horizons,
                    target_profits=params.get('target_profits', self.target_profits),
                    volatility_window=params.get('volatility_window', self.volatility_window),
                    min_label_quality=params.get('min_label_quality', self.min_label_quality),
                    min_predictability=params.get('min_predictability', self.min_predictability)
                )

                # Simple scoring based on data quality metrics
                quality = create_data_quality_report(data)
                score = quality.get('quality_metrics', {}).get('numeric_columns', 0) * 0.1
                score += (1 - quality.get('quality_metrics', {}).get('missing_percentage', 100)) * 0.01

                if score > best_score:
                    best_score = score
                    best_config = config

            except Exception as e:
                tprint_warning(f"⚠️ Error evaluating config {params}: {e}")
                continue

        if best_config:
            tprint_success(f"✅ Grid search completed. Best score: {best_score:.3f}")
            return best_config

        return self


class AnalystProfitLabeler:
    """
    Analyst Profit Labeler - Specialized labeling for Analyst models.
    
    This class wraps the VolatilityAwareMultiHorizonLabeler with Analyst-specific
    configurations and provides a simplified interface for Analyst model training.
    """
    
    def __init__(self, config: Optional[AnalystProfitLabelerConfig] = None):
        """Initialize the Analyst profit labeler."""
        self.config = config or AnalystProfitLabelerConfig()
        self.logger = system_logger.getChild('AnalystProfitLabeler')

        # Initialize matrix operations for enhanced data processing
        self.matrix_ops = get_unified_matrix_operations()
        self.vectorized_core = get_vectorized_processing_core()
        self.enhanced_matrix_ops = get_enhanced_matrix_operations()

        tprint_info(f"🧮 Matrix operations initialized: {self.matrix_ops.__class__.__name__}")

        # Initialize M1 optimizations if available
        self.m1_integration = integrate_with_m1_optimizers()
        if self.m1_integration.get('success', False):
            tprint_info(f"🧠 M1 optimizations initialized: GPU={'✅' if self.m1_integration.get('gpu_manager') else '❌'}, Memory={'✅' if self.m1_integration.get('memory_optimizer') else '❌'}")

        if not VOLATILITY_LABELER_AVAILABLE:
            raise RuntimeError(
                "VolatilityAwareMultiHorizonLabeler is not available. "
                "Please ensure the profit_labeling module is properly installed."
            )
        
        # Initialize advanced filters if available and enabled
        self.advanced_filters = None
        if self.config.enable_advanced_filters and ADVANCED_FILTERS_AVAILABLE:
            filters_config = self.config.advanced_filters_config or AdvancedFiltersConfig()
            self.advanced_filters = AdvancedFilters15m(filters_config)
            tprint_info("🔍 Advanced 15m filters initialized")
        elif self.config.enable_advanced_filters and not ADVANCED_FILTERS_AVAILABLE:
            tprint_warning("⚠️ Advanced filters requested but not available")

        # Create the underlying labeler with Analyst-specific config
        self.labeler = self._create_labeler()

        tprint_success(f"✅ AnalystProfitLabeler initialized (timeframe: {self.config.timeframe}, matrix_ops: {type(self.matrix_ops).__name__})")

    def cleanup(self) -> None:
        """Clean up resources and optimize memory."""
        try:
            # Optimize memory usage
            memory_info = optimize_memory()
            if memory_info.get('success', False):
                tprint_info(f"🧠 Memory optimized: {memory_info.get('objects_collected', 0)} objects collected")

            # Clean up M1 optimizers if available
            from src.utils.common_operations import cleanup_m1_optimizers
            cleanup_m1_optimizers()

            tprint_success("✅ AnalystProfitLabeler cleanup completed")
        except Exception as e:
            tprint_warning(f"⚠️ Error during cleanup: {e}")

    def _create_labeler(self) -> Any:
        """Create and configure the VolatilityAwareMultiHorizonLabeler for Analyst."""
        # Create Analyst-specific configuration
        labeler_config = VolatilityAwareConfig()
        
        # Set label definition type to Analyst
        labeler_config.label_definition_type = LabelDefinitionType.ANALYST
        labeler_config.enable_enhanced_labels = True
        
        # Configure bar construction to use TIME bars (we're working with OHLCV data)
        # TIME bars with bar_size = timeframe period will pass through the data as-is
        from src.training.steps.pre_training.profit_labeling.bar_construction import BarType
        labeler_config.bar_construction.bar_type = BarType.TIME
        labeler_config.bar_construction.bar_size = float(self.config.base_period_minutes)  # 60 minutes for 1h data
        labeler_config.bar_construction.min_bars_required = 10  # Lower threshold for OHLCV data
        
        # Configure noise gating to be less aggressive for OHLCV data
        # OHLCV data is already aggregated, so noise is less of an issue
        labeler_config.noise_gating.enabled = False  # Disable noise gating for OHLCV data
        
        # Configure timeframe and horizons
        labeler_config.timeframe = self.config.timeframe
        labeler_config.multi_target.horizons = self.config.horizons
        labeler_config.multi_target.target_profits = self.config.target_profits
        # Configure multi-target to use very lenient quality thresholds
        labeler_config.multi_target.min_lqs_score = 0.01  # Very lenient LQS threshold (default 0.3)
        
        # Configure volatility settings
        labeler_config.volatility.enabled = self.config.use_volatility_normalization
        labeler_config.volatility.window = self.config.volatility_window
        
        # Configure quality scoring - disable for initial labeling to ensure labels are generated
        # Quality filtering can be applied during feature selection/training
        labeler_config.enable_quality_scoring = False  # Disable strict quality filtering
        labeler_config.quality_scoring.min_quality_threshold = 0.1  # Very lenient threshold
        labeler_config.quality_scoring.min_predictability = 0.1  # Very lenient threshold
        
        # Configure regime adaptation
        labeler_config.regime_config.enabled = self.config.enable_regime_adaptation
        
        # Apply custom parameters
        if self.config.custom_params:
            for key, value in self.config.custom_params.items():
                if hasattr(labeler_config, key):
                    setattr(labeler_config, key, value)
        
        # Create the VolatilityAwareMultiHorizonLabeler with our configuration
        return VolatilityAwareMultiHorizonLabeler(labeler_config)
    
    def generate_labels(
        self,
        data: pd.DataFrame,
        regime_assignments: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> LabelingResult:
        """
        Generate Analyst profit labels for the input data.

        Args:
            data: Input market data (OHLCV format)
            regime_assignments: Optional regime assignments for regime-aware labeling
            **kwargs: Additional parameters for the labeler

        Returns:
            LabelingResult with labels, confidence scores, and quality metrics
        """
        tprint_info(f"📈 Generating Analyst profit labels for {len(data)} samples...")

        try:
            # Validate input data quality using both common operations and utilities
            data_quality = create_data_quality_report(data)
            detailed_quality = analyze_nan_values_detailed(data)

            if data_quality.get('quality_metrics', {}).get('missing_percentage', 0) > 50:
                tprint_warning(f"⚠️ High missing data percentage: {data_quality['quality_metrics']['missing_percentage']:.2f}%")

            # Log detailed NaN analysis if issues found
            if detailed_quality.get('total_nans', 0) > 0:
                nan_report = format_nan_analysis_report(detailed_quality, "  ")
                tprint_info(f"📊 NaN Analysis:\n{nan_report}")

            # Validate required columns for OHLCV data
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not validate_dataframe_columns(data, required_columns):
                missing_cols = set(required_columns) - set(data.columns)
                raise ValueError(f"Missing required OHLCV columns: {missing_cols}")

            # Validate data types and convert if necessary using safe operations
            for col in ['open', 'high', 'low', 'close']:
                if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                    tprint_warning(f"⚠️ Converting {col} to numeric")
                    data = safe_dataframe_operation(data, pd.to_numeric, col, errors='coerce')

            # Optimize data using matrix operations for better performance
            tprint_info(f"🧮 Optimizing data with matrix operations ({data.shape})")
            original_shape = data.shape
            optimized_data = optimize_dataframe(data)

            if optimized_data is not data:
                data = optimized_data
                tprint_success(f"✅ Data optimized: {original_shape} → {data.shape}")

            # Note: regime_assignments are not currently used by the underlying labeler
            # but are kept in the API for future enhancement
            if regime_assignments is not None:
                tprint_info(f"📊 Regime assignments provided but not yet integrated into labeling logic")

            # Apply advanced filters if enabled
            filter_result = None
            if self.advanced_filters is not None:
                tprint_info("🔍 Applying advanced 15m filters before labeling")
                filter_result = self.advanced_filters.apply_filters(data)
                
                # Log filter results
                tprint_info(f"   → Filter eligibility: {filter_result.eligibility_ratio:.1%} ({filter_result.n_eligible_samples}/{filter_result.n_total_samples})")
                
                # Apply filter mask to data (optional - can be used to pre-filter data)
                if filter_result.eligibility_ratio < 0.5:
                    tprint_warning(f"⚠️ Low filter eligibility: {filter_result.eligibility_ratio:.1%}")
                else:
                    tprint_success(f"✅ Filter eligibility good: {filter_result.eligibility_ratio:.1%}")

            # Use memory optimization context for label generation
            with memory_checkpoint("analyst_label_generation"):
                # Generate labels using the underlying labeler
                # Note: VolatilityAwareMultiHorizonLabeler.generate_labels only takes market_data
                result = self.labeler.generate_labels(data)
                
                # Apply filter mask to results if filters were used
                if filter_result is not None and hasattr(result, 'labels') and result.labels is not None:
                    # Apply eligibility mask to labels
                    if isinstance(result.labels, pd.DataFrame):
                        result.labels = result.labels[filter_result.eligibility_mask]
                    
                    # Apply eligibility mask to confidence scores if available
                    if hasattr(result, 'confidence_scores') and result.confidence_scores is not None:
                        if isinstance(result.confidence_scores, pd.DataFrame):
                            result.confidence_scores = result.confidence_scores[filter_result.eligibility_mask]
                    
                    # Apply eligibility mask to eligibility masks if available
                    if hasattr(result, 'eligibility_masks') and result.eligibility_masks is not None:
                        if isinstance(result.eligibility_masks, pd.DataFrame):
                            result.eligibility_masks = result.eligibility_masks[filter_result.eligibility_mask]
                    
                    # Update sample counts
                    result.n_samples = len(result.labels) if hasattr(result.labels, '__len__') else result.n_samples
                    
                    tprint_info(f"🔍 Applied filter mask: {result.n_samples} samples after filtering")

            # Validate minimum sample count for training
            self._validate_labeling_result(result)

            # Log memory usage and data quality with matrix operations info
            memory_info = optimize_memory()
            data_info = get_dataframe_info(data)
            hardware_report = get_hardware_performance_report()
            tprint_info(f"📊 Data info: {data_info['shape']} shape, {format_bytes(data_info['memory_usage'])} memory")
            tprint_info(f"🔧 Hardware performance: {hardware_report.get('cpu_cores', 'N/A')} cores, GPU: {hardware_report.get('gpu_available', 'N/A')}")

            tprint_success(
                f"✅ Analyst labels generated: {result.n_samples} samples, "
                f"{result.n_targets} targets, {result.n_horizons} horizons"
            )

            return result
            
        except Exception as e:
            tprint_error(f"❌ Failed to generate Analyst labels: {e}")
            raise
    
    def _validate_labeling_result(self, result: LabelingResult) -> None:
        """Validate that labeling produced sufficient samples for training."""
        MIN_SAMPLES_PER_TARGET = 50  # Minimum samples needed per target for reliable training
        MIN_TOTAL_SAMPLES = 200     # Absolute minimum total samples

        if result.n_samples < MIN_TOTAL_SAMPLES:
            raise ValueError(
                f"Insufficient samples for training: got {result.n_samples}, need at least {MIN_TOTAL_SAMPLES}. "
                f"Consider adjusting labeling parameters (horizons, thresholds, or data timeframe) to generate more labels."
            )

        if result.n_targets > 0:
            samples_per_target = result.n_samples / result.n_targets
            if samples_per_target < MIN_SAMPLES_PER_TARGET:
                warnings.warn(
                    f"Low samples per target: {samples_per_target:.1f} per target, "
                    f"recommended minimum is {MIN_SAMPLES_PER_TARGET}. "
                    "Model training may be unreliable with insufficient samples per target.",
                    UserWarning,
                    stacklevel=2
                )

    def get_label_summary(self, result: LabelingResult) -> Dict[str, Any]:
        """Get a summary of the labeling results."""
        summary = {
            'n_samples': result.n_samples,
            'n_targets': result.n_targets,
            'n_horizons': result.n_horizons,
            'processing_time': result.processing_time,
            'quality_scores': {}
        }
        
        # Add quality scores
        if result.quality_scores:
            for target_name, quality in result.quality_scores.items():
                summary['quality_scores'][target_name] = {
                    'overall_quality': quality.overall_quality,
                    'predictability': quality.predictability,
                    'stability': quality.stability,
                    'balance': quality.balance
                }
        
        return summary


@register_component('analyst_profit_labeler')
class AnalystProfitLabelerComponent(BasePreTrainingComponent):
    """
    Component wrapper for Analyst Profit Labeler.
    
    This component integrates the AnalystProfitLabeler with the pre-training pipeline
    and handles proper error handling, reporting, and pipeline state management.
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the Analyst profit labeler component."""
        super().__init__(config)
        self.logger = system_logger.getChild('AnalystProfitLabelerComponent')
        
        # Create Analyst-specific configuration
        analyst_config = AnalystProfitLabelerConfig()
        
        # Override with custom parameters if provided
        if self.config and self.config.custom_params:
            custom_params = self.config.custom_params
            
            # Update timeframe if provided
            if 'timeframe' in custom_params:
                analyst_config.timeframe = custom_params['timeframe']
                # Update base period based on timeframe
                if analyst_config.timeframe.endswith('m'):
                    analyst_config.base_period_minutes = int(analyst_config.timeframe[:-1])
                elif analyst_config.timeframe.endswith('h'):
                    analyst_config.base_period_minutes = int(analyst_config.timeframe[:-1]) * 60
            
            # Update other parameters
            for key in ['horizons', 'target_profits', 'min_label_quality', 'min_predictability', 'enable_advanced_filters']:
                if key in custom_params:
                    setattr(analyst_config, key, custom_params[key])
            
            # Update advanced filters configuration if provided
            if 'advanced_filters_config' in custom_params and ADVANCED_FILTERS_AVAILABLE:
                filters_config_dict = custom_params['advanced_filters_config']
                if isinstance(filters_config_dict, dict):
                    analyst_config.advanced_filters_config = AdvancedFiltersConfig(**filters_config_dict)
                elif isinstance(filters_config_dict, AdvancedFiltersConfig):
                    analyst_config.advanced_filters_config = filters_config_dict
            
            # Store all custom params for the underlying labeler
            analyst_config.custom_params = custom_params
        
        # Create the labeler
        try:
            self.labeler = AnalystProfitLabeler(analyst_config)
            tprint_success("✅ AnalystProfitLabelerComponent initialized")
        except Exception as e:
            tprint_error(f"❌ Failed to initialize AnalystProfitLabelerComponent: {e}")
            raise
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['multi_horizon_labeling_result', 'labeling_report']
    
    async def execute(self, data: Any, pipeline_state: PipelineState) -> ComponentResult:
        """
        Execute Analyst profit labeling as a component.
        
        Args:
            data: Input data (typically market data DataFrame)
            pipeline_state: Current pipeline state
            
        Returns:
            ComponentResult with labeling results and artifacts
        """
        try:
            tprint_info("🚀 Starting Analyst Profit Labeling execution...")
            
            # Extract data from pipeline state if not provided
            if data is None:
                data = pipeline_state.get('prepared_data')
                if data is None:
                    raise ValueError("No input data provided and no prepared_data in pipeline state")

            # Validate OHLCV data format and constraints
            try:
                data = validate_raw_ohlcv(data, context='analyst_profit_labeler.input_validation')
                tprint_info(f"✅ Input data validated: {len(data)} rows, {len(data.columns)} columns")
            except SchemaValidationException as e:
                tprint_error(f"❌ Input data validation failed: {e}")
                raise ValueError(f"Invalid input data format: {e}") from e

            # Extract regime assignments if available
            regime_assignments = pipeline_state.get('regime_assignments')
            if regime_assignments is not None:
                tprint_info(f"📊 Using regime assignments: {len(regime_assignments)} regimes")
            
            # Generate labels
            labeling_result = self.labeler.generate_labels(
                data=data,
                regime_assignments=regime_assignments
            )
            
            # Create artifacts bundle
            from src.training.steps.pre_training.components.contracts import GenericArtifacts
            
            # Save labeled data to parquet file for persistence
            symbol = pipeline_state.get('symbol', 'UNKNOWN')
            exchange = pipeline_state.get('exchange', 'UNKNOWN')
            timeframe = pipeline_state.get('timeframe', 'UNKNOWN')
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            artifacts_dir = Path('artifacts')
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            
            labeled_data_file = artifacts_dir / f'labeled_data_{symbol}_{exchange}_{timeframe}_{timestamp_str}.parquet'
            
            # Save labeled DataFrame to parquet
            if isinstance(labeling_result.labels, pd.DataFrame) and not labeling_result.labels.empty:
                labeling_result.labels.to_parquet(labeled_data_file)
                tprint_success(f"✅ Saved labeled data to {labeled_data_file}")
            
            # GenericArtifacts just needs to be instantiated, then we add attributes
            artifacts = GenericArtifacts()
            artifacts.multi_horizon_labeling_result = {
                'labeled_data': labeling_result.labels,  # Keep in memory for pipeline continuity
                'labeled_data_file': str(labeled_data_file),  # Add file path for persistence
                'labels': labeling_result.labels,
                'confidence_scores': labeling_result.confidence_scores,
                'eligibility_masks': labeling_result.eligibility_masks,
                'quality_scores': labeling_result.quality_scores,
                'normalization_factors': labeling_result.normalization_factors,
                'processing_time': labeling_result.processing_time,
                'n_samples': labeling_result.n_samples,
                'n_targets': labeling_result.n_targets,
                'n_horizons': labeling_result.n_horizons,
                'method': 'analyst_profit_labeling',
            }
            artifacts.labeling_report = {
                'status': 'completed',
                'timestamp': datetime.now().isoformat(),
                'method': 'analyst_profit_labeling',
                'timeframe': self.labeler.config.timeframe,
                'summary': self.labeler.get_label_summary(labeling_result),
                'horizons': self.labeler.config.horizons,
                'target_profits': self.labeler.config.target_profits,
            }
            
            # Create result
            result = ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'component': 'analyst_profit_labeler',
                    'timeframe': self.labeler.config.timeframe,
                    'n_samples': labeling_result.n_samples,
                    'n_targets': labeling_result.n_targets,
                    'n_horizons': labeling_result.n_horizons,
                    'direction_settings': {
                        'enable_long_positions': self.labeler.config.enable_long_positions,
                        'enable_short_positions': self.labeler.config.enable_short_positions,
                    }
                }
            )
            
            # Generate outcome file with datetime stamp
            try:
                outcome_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                outcomes_dir = Path('outcomes')
                ensure_directory(outcomes_dir)
                
                outcome_filename = f"analyst_labeler_outcome_{outcome_timestamp}.json"
                outcome_path = outcomes_dir / outcome_filename
                
                # Create comprehensive outcome report with detailed statistics
                
                # Calculate label distribution statistics
                label_distribution = {}
                if labeling_result.labels is not None and isinstance(labeling_result.labels, pd.DataFrame):
                    for col in labeling_result.labels.columns:
                        label_vals = labeling_result.labels[col].dropna()
                        if len(label_vals) > 0:
                            label_distribution[col] = {
                                'total_labels': int(len(label_vals)),
                                'unique_values': int(label_vals.nunique()),
                                'value_counts': label_vals.value_counts().head(10).to_dict(),
                                'mean': float(label_vals.mean()) if pd.api.types.is_numeric_dtype(label_vals) else None,
                                'std': float(label_vals.std()) if pd.api.types.is_numeric_dtype(label_vals) else None,
                                'min': float(label_vals.min()) if pd.api.types.is_numeric_dtype(label_vals) else None,
                                'max': float(label_vals.max()) if pd.api.types.is_numeric_dtype(label_vals) else None,
                            }
                
                # Calculate per-horizon and per-target breakdowns
                horizon_breakdown = {}
                for i, horizon in enumerate(self.labeler.config.horizons):
                    horizon_breakdown[f"{horizon}min"] = {
                        'horizon_minutes': horizon,
                        'horizon_bars': horizon // self.labeler.config.base_period_minutes,
                        'expected_labels': labeling_result.n_targets // len(self.labeler.config.horizons) if labeling_result.n_targets > 0 else 0,
                    }
                
                target_breakdown = {}
                for i, target in enumerate(self.labeler.config.target_profits):
                    target_breakdown[f"{target}%"] = {
                        'target_profit_pct': target,
                        'expected_labels': labeling_result.n_targets // len(self.labeler.config.target_profits) if labeling_result.n_targets > 0 else 0,
                    }
                
                # Data quality assessment
                data_quality = {
                    'input_data': {
                        'rows': len(data),
                        'columns': len(data.columns),
                        'date_range': {
                            'start': str(data.index.min()) if hasattr(data.index, 'min') else None,
                            'end': str(data.index.max()) if hasattr(data.index, 'max') else None,
                            'duration_days': float((data.index.max() - data.index.min()).total_seconds() / 86400) if hasattr(data.index, 'min') and hasattr(data.index, 'max') else None,
                        },
                        'missing_values': int(data.isnull().sum().sum()),
                        'missing_percentage': float(data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100),
                    },
                    'output_labels': {
                        'total_generated': labeling_result.n_samples,
                        'label_coverage': float(labeling_result.n_samples / len(data) * 100) if len(data) > 0 else 0.0,
                        'targets_per_sample': float(labeling_result.n_targets / labeling_result.n_samples) if labeling_result.n_samples > 0 else 0.0,
                    }
                }
                
                outcome_data = {
                    'component': 'analyst_profit_labeler',
                    'timestamp': datetime.now().isoformat(),
                    'execution_time': labeling_result.processing_time,
                    'timeframe': self.labeler.config.timeframe,
                    'configuration': {
                        'horizons': self.labeler.config.horizons,
                        'target_profits': self.labeler.config.target_profits,
                        'use_volatility_normalization': self.labeler.config.use_volatility_normalization,
                        'volatility_window': self.labeler.config.volatility_window,
                        'min_label_quality': self.labeler.config.min_label_quality,
                        'min_predictability': self.labeler.config.min_predictability,
                        'enable_regime_adaptation': self.labeler.config.enable_regime_adaptation,
                        'enable_long_positions': self.labeler.config.enable_long_positions,
                        'enable_short_positions': self.labeler.config.enable_short_positions,
                        'base_period_minutes': self.labeler.config.base_period_minutes,
                    },
                    'results': {
                        'n_samples': labeling_result.n_samples,
                        'n_targets': labeling_result.n_targets,
                        'n_horizons': labeling_result.n_horizons,
                        'label_summary': self.labeler.get_label_summary(labeling_result),
                        'label_distribution': label_distribution,
                        'horizon_breakdown': horizon_breakdown,
                        'target_breakdown': target_breakdown,
                    },
                    'quality_metrics': {
                        target_name: {
                            'overall_quality': float(quality.overall_quality) if quality.overall_quality is not None else None,
                            'predictability': float(quality.predictability) if quality.predictability is not None else None,
                            'stability': float(quality.stability) if quality.stability is not None else None,
                            'balance': float(quality.balance) if quality.balance is not None else None,
                        }
                        for target_name, quality in (labeling_result.quality_scores or {}).items()
                    },
                    'data_quality': data_quality,
                    'data_info': {
                        'input_rows': len(data),
                        'input_columns': len(data.columns),
                        'regime_assignments_available': regime_assignments is not None,
                        'regime_count': int(regime_assignments.nunique()) if regime_assignments is not None and hasattr(regime_assignments, 'nunique') else None,
                    },
                    'confidence_statistics': {
                        'total_confidence_scores': len(labeling_result.confidence_scores) if labeling_result.confidence_scores is not None else 0,
                        'confidence_columns': list(labeling_result.confidence_scores.columns) if labeling_result.confidence_scores is not None and hasattr(labeling_result.confidence_scores, 'columns') else [],
                    },
                    'eligibility_statistics': {
                        'total_eligibility_masks': len(labeling_result.eligibility_masks) if labeling_result.eligibility_masks is not None else 0,
                        'eligible_samples': int(labeling_result.eligibility_masks.sum().sum()) if labeling_result.eligibility_masks is not None and hasattr(labeling_result.eligibility_masks, 'sum') else 0,
                    },
                    'normalization_factors': labeling_result.normalization_factors if labeling_result.normalization_factors else {},
                    'status': 'success'
                }
                
                safe_json_dump(outcome_data, str(outcome_path))
                tprint_success(f"📄 Outcome file saved: {outcome_filename}")
                
            except Exception as outcome_error:
                tprint_warning(f"⚠️ Failed to save outcome file: {outcome_error}")
                # Don't fail the component if outcome file generation fails
            
            tprint_success("✅ Analyst Profit Labeling completed successfully")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Analyst Profit Labeling failed: {e}")
            
            result = ComponentResult(
                success=False,
                error_message=str(e),
                metadata={'component': 'analyst_profit_labeler'}
            )
            return result


# Convenience function for external usage
async def execute_analyst_profit_labeling(
    data: pd.DataFrame,
    regime_assignments: Optional[pd.DataFrame] = None,
    config: Optional[AnalystProfitLabelerConfig] = None,
    **kwargs
) -> LabelingResult:
    """
    Execute Analyst profit labeling.
    
    Args:
        data: Input market data (OHLCV format)
        regime_assignments: Optional regime assignments
        config: Optional configuration
        **kwargs: Additional parameters
        
    Returns:
        LabelingResult with labels and quality metrics
    """
    labeler = AnalystProfitLabeler(config)
    return labeler.generate_labels(data, regime_assignments, **kwargs)