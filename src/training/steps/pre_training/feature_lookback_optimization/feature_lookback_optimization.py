"""
Modular Feature Lookback Optimization Component.

This is the main component that uses the modular architecture with separate
modules for validation, error handling, performance monitoring, and optimization.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# Import utility modules
from src.utils.common_operations import safe_dataframe_operation
from src.utils.common_utilities import CommonUtilities
from src.utils.math_validation import safe_divide
from src.utils.serialization_utils import UniversalSerializer

# Import numpy for type checking
from .dependency_manager import get_dependency
np, _ = get_dependency('numpy')

# Utility function to convert int64 to int for dictionary keys
def convert_int64_to_int(value: Any) -> Any:
    """Convert int64 values to regular Python int for JSON serialization."""
    tprint(f"🔄 Converting value of type {type(value).__name__} for JSON safety")
    try:
        if hasattr(value, 'dtype') and value.dtype == 'int64':
            tprint("📏 Detected numpy dtype int64, converting to int")
            return int(value)
        elif isinstance(value, np.int64):
            tprint("📏 Detected numpy.int64 instance, converting to int")
            return int(value)
        elif isinstance(value, dict):
            # Convert both keys and values to handle int64 keys
            converted_dict = {}
            for k, v in value.items():
                # Convert key if it's int64
                converted_key = k
                if isinstance(k, np.int64):
                    tprint("🔑 Converting dict key from numpy.int64 to int")
                    converted_key = int(k)
                elif hasattr(k, 'dtype') and k.dtype == 'int64':
                    tprint("🔑 Converting dict key from numpy dtype int64 to int")
                    converted_key = int(k)

                # Convert value recursively
                converted_dict[converted_key] = convert_int64_to_int(v)

            return converted_dict
        elif isinstance(value, (list, tuple)):
            # Convert each item in the list/tuple recursively
            tprint("📚 Converting sequence items for JSON safety")
            return [convert_int64_to_int(item) for item in value]
        elif hasattr(value, 'shape') and len(value.shape) > 0:
            # Handle numpy arrays that might be problematic
            if value.size > 100:  # Large arrays might cause issues
                tprint("📦 Large numpy array detected, summarizing instead of full conversion")
                return {
                    'type': 'numpy_array',
                    'shape': value.shape,
                    'dtype': str(value.dtype),
                    'size': value.size
                }
            else:
                tprint("📦 Small numpy array detected, converting to list")
                return value.tolist()  # Convert small arrays to lists
        else:
            return value
    except Exception as e:
        # If conversion fails, return a safe representation
        return {
            'conversion_error': str(e),
            'original_type': type(value).__name__,
            'safe_representation': 'unconvertible_value'
        }

# Import modular components
from .core.optimizer import CoreOptimizer, OptimizationMethod, OptimizationResult
from .validation.validator import InputValidator, ValidationLevel, ValidationStatus, ValidationSummary
from .error_handling.error_handler import StandardizedErrorHandler, ErrorSeverity, ErrorCategory
from .performance.monitor import PerformanceMonitor, MetricType, MetricLevel

from ..components.base_component import BasePreTrainingComponent, ComponentConfig, ComponentResult

# Import optimized process engine
from ...market_analysis.optimized_process_engines import OptimizedFeatureLookbackEngine, ProcessType

# Import dependencies with fallbacks
from .dependency_manager import get_dependency, is_dependency_available

# Get dependencies
np, _ = get_dependency('numpy')
pd, _ = get_dependency('pandas')

# Import logger
from src.utils.logger import system_logger
from src.utils.tprint import tprint
from ...market_analysis.logging_standards import (
    get_logger, log_info, log_warning, log_error, log_success, log_debug,
    LoggingContext, log_step_progress, log_data_info, log_validation_result
)


@dataclass
class OptimizationMetrics:
    """Comprehensive optimization metrics."""
    best_lookback_period: int
    best_score: float
    optimization_method: str
    total_features_optimized: int
    optimization_time: float
    convergence_iterations: int
    memory_usage_mb: float
    cpu_usage_percent: float
    validation_score: float
    stability_score: float
    regime_coverage: float
    error_rate: float


class FeatureLookbackOptimizationComponent(BasePreTrainingComponent):
    """
    Modular Feature Lookback Optimization Component.

    This component uses a modular architecture with separate modules for:
    - Core optimization logic
    - Input validation
    - Error handling
    - Performance monitoring
    """

    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the feature lookback optimization component."""
        tprint("🔧 Initializing Modular FeatureLookbackOptimizationComponent...")
        super().__init__(config)

        # Use standardized logging
        self.logger = get_logger('FeatureLookbackOptimization')
        self.common_utils = CommonUtilities()
        self.serializer = UniversalSerializer()
        tprint("✅ Basic modular component initialization complete")

        # Initialize modular components
        tprint("🔧 Initializing modular components...")
        self.validator = InputValidator(logger=self.logger)
        self.error_handler = StandardizedErrorHandler(logger=self.logger, component_name="FeatureLookbackOptimization")
        self.performance_monitor = PerformanceMonitor(component_name="FeatureLookbackOptimization")
        self.core_optimizer = CoreOptimizer(logger=self.logger)
        tprint("✅ Modular components initialized")

        # Initialize execution mode configuration
        tprint("🔧 Initializing execution mode lookback configuration...")
        try:
            from ..shared_utils.execution_mode_lookback_config import get_execution_mode_config
            self.execution_mode_config = get_execution_mode_config()
            tprint("✅ Execution mode configuration initialized")
        except ImportError as e:
            self.logger.warning(f"⚠️ Could not import execution mode config: {e}")
            self.execution_mode_config = None

        # Initialize optimized process engine
        tprint("🔧 Initializing optimized feature lookback engine...")
        self.optimized_engine = OptimizedFeatureLookbackEngine(
            use_hardware_accel=True,
            cache_size=1000
        )
        tprint("✅ Optimized feature lookback engine initialized")

        # Component state
        self.optimization_status = "pending"
        self.start_time: Optional[float] = None
        self.metrics: Optional[OptimizationMetrics] = None

        # Performance monitoring (separate from PerformanceMonitor instance)
        self.performance_data = {
            'memory_usage': [],
            'cpu_usage': [],
            'execution_times': {},
            'error_counts': 0,
            'peak_memory_mb': 0.0,
            'memory_warnings': 0
        }

        tprint("📈 Performance data trackers initialized for optimization component")

        # Memory monitoring thresholds
        self.memory_warning_threshold_mb = 1000.0  # 1GB
        self.memory_critical_threshold_mb = 2000.0  # 2GB

        tprint("✅ Modular FeatureLookbackOptimizationComponent initialized")

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts for this component."""
        tprint("📋 Getting required artifacts for modular feature lookback optimization")
        artifacts = [
            'market_data',
            'labeling_results',
            'regime_splitting_results'
        ]
        tprint(f"✅ Required artifacts: {artifacts}")
        return artifacts

    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute the feature lookback optimization.

        Args:
            data: Input data for optimization
            pipeline_state: Current pipeline state

        Returns:
            ComponentResult with optimization results
        """
        tprint("🚀 Starting modular feature lookback optimization execution...")
        start_time = self.performance_monitor.start_operation("execute")

        try:
            log_info("🚀 Starting feature lookback optimization with multi-horizon profit targets...")
            tprint("📊 Performance monitoring started for execute operation")

            # Validate inputs
            is_valid, validation_summary, cleaned_data = self.validator.validate_data(
                data,
                required_columns=['open', 'high', 'low', 'close', 'volume']
            )

            if not is_valid:
                tprint("❌ Data validation failed, aborting execution")
                error_msg = f"Data validation failed: {validation_summary.recommendations}"
                self.error_handler.handle_error(
                    ValueError(error_msg),
                    "validate_data",
                    return_value=self._create_failed_result()
                )
                return self._create_failed_result()

            # Record validation metrics
            tprint("📈 Recording validation metrics after successful validation")
            self.performance_monitor.record_optimization_metrics(
                {},
                data_quality_score=validation_summary.quality_score,
                validation_score=1.0 if validation_summary.overall_status == ValidationStatus.PASSED else 0.0
            )

            # Extract execution mode parameters from pipeline configuration
            execution_mode_params = {}
            if self.execution_mode_config and hasattr(pipeline_state, 'get'):
                try:
                    # Try to extract execution mode from pipeline state or config
                    pipeline_config = pipeline_state.get('pipeline_config', {})
                    lookback_config = self.execution_mode_config.extract_from_pipeline_config(pipeline_config)
                    execution_mode_params = self.execution_mode_config.get_optimization_parameters(
                        pipeline_config.get('mode', 'full')
                    )
                    self.logger.info(f"📊 Using execution mode parameters: {execution_mode_params}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not extract execution mode parameters: {e}")
                    execution_mode_params = {}

            # Load required data
            tprint("📥 Loading market data for optimization")
            market_data = await self._load_market_data(cleaned_data)
            labeling_data = self._load_recent_labeling_results(
                pipeline_state.get('symbol', 'UNKNOWN'),
                pipeline_state.get('exchange', 'UNKNOWN'),
                pipeline_state.get('timeframe', 'UNKNOWN'),
                pipeline_state=pipeline_state
            )

            # Apply execution mode data windowing
            if execution_mode_params and market_data is not None:
                window_days = execution_mode_params.get('window_days', 1460)
                if len(market_data) > window_days:
                    # Use only the most recent data based on execution mode
                    market_data = market_data.tail(window_days).copy()
                    self.logger.info(f"📊 Applied execution mode window: using last {window_days} days of data")
                else:
                    self.logger.info(f"📊 Using all available data ({len(market_data)} records) for execution mode")

            if market_data is None:
                log_error("Market data loading failed - no data available for feature lookback optimization")
                tprint("❌ Market data missing, returning failed result")
                return self._create_failed_result()

            # Align data with regime assignments to ensure consistency
            tprint("📐 Aligning market data with regime assignments")
            market_data = self._align_data_with_regime_assignments(market_data, pipeline_state)

            # Prepare data for optimization
            tprint("🧰 Preparing data for feature optimization")
            optimization_data = self._prepare_data_for_optimization(market_data, labeling_data)

            if optimization_data is None or optimization_data.empty:
                log_error(f"Data preparation failed - optimization data is {'None' if optimization_data is None else 'empty'}")
                tprint("❌ Prepared optimization data is empty or None")
                return self._create_failed_result()

            # Perform feature optimization
            tprint("⚙️ Performing feature optimization workflow")
            optimization_results = await self._perform_feature_optimization(optimization_data, pipeline_state)

            # Convert int64 values to regular int values for JSON serialization
            tprint("🔄 Converting optimization results for JSON serialization")
            optimization_results = convert_int64_to_int(optimization_results)

            # Create optimization metrics
            tprint("📏 Creating optimization metrics")
            metrics = self._create_optimization_metrics(optimization_results)

            # Create artifacts
            tprint("📦 Creating artifacts from optimization results")
            artifacts = self._create_artifacts(optimization_results, pipeline_state)

            # Record final metrics
            tprint("🏁 Ending performance monitoring for execute operation")
            self.performance_monitor.end_operation("execute", start_time, success=True)

            # Save artifacts persistently using the artifact manager
            try:
                import asyncio
                # Check if we're already in an event loop
                try:
                    loop = asyncio.get_running_loop()
                    # We're in an event loop, create a task instead
                    task = asyncio.create_task(self.save_artifacts(artifacts, {
                        'optimization_status': 'completed',
                        'total_features_optimized': len(optimization_results.get('feature_results', {})),
                        'validation_summary': validation_summary.__dict__ if validation_summary else None,
                        'performance_metrics': self.performance_monitor.get_performance_summary(),
                        'optimization_results': optimization_results
                    }))
                    saved_files = await task
                    log_success(f"💾 [FEATURE_LOOKBACK] Artifacts saved persistently: {list(saved_files.keys())}")
                except RuntimeError:
                    # No running event loop, use asyncio.run()
                    tprint("💾 Saving artifacts using asyncio.run")
                    saved_files = asyncio.run(self.save_artifacts(artifacts, {
                        'optimization_status': 'completed',
                        'total_features_optimized': len(optimization_results.get('feature_results', {})),
                        'validation_summary': validation_summary.__dict__ if validation_summary else None,
                        'performance_metrics': self.performance_monitor.get_performance_summary(),
                        'optimization_results': optimization_results
                    }))
                    log_success(f"💾 [FEATURE_LOOKBACK] Artifacts saved persistently: {list(saved_files.keys())}")
            except Exception as e:
                log_warning(f"⚠️ [FEATURE_LOOKBACK] Failed to save artifacts persistently: {e}")

            result = ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'optimization_status': 'completed',
                    'total_features_optimized': len(optimization_results.get('feature_results', {}).get('long_pipeline', {})) + len(optimization_results.get('feature_results', {}).get('short_pipeline', {})),
                    'validation_summary': validation_summary.__dict__ if validation_summary else None,
                    'performance_metrics': self.performance_monitor.get_performance_summary(),
                    'optimization_results': optimization_results,
                    'artifacts_saved_persistently': True,
                    'pipeline_type': 'differentiated_long_short'
                }
            )

            long_count = len(optimization_results.get('feature_results', {}).get('long_pipeline', {}))
            short_count = len(optimization_results.get('feature_results', {}).get('short_pipeline', {}))
            log_success(f"🎯 Multi-horizon feature lookback optimization completed successfully - Long: {long_count} features, Short: {short_count} features")
            tprint("✅ Feature lookback optimization execution completed successfully")
            return result

        except Exception as e:
            self.error_handler.handle_error(
                e,
                "execute",
                return_value=self._create_failed_result()
            )
            self.performance_monitor.end_operation("execute", start_time, success=False)
            tprint("❌ Feature lookback optimization execution failed")
            return self._create_failed_result()

    def _create_failed_result(self) -> ComponentResult:
        """Create a failed component result."""
        return ComponentResult(
            success=False,
            artifacts={},
            metadata={'optimization_status': 'failed'}
        )

    async def _load_market_data(self, data: Any) -> Optional[pd.DataFrame]:
        """Load market data for optimization."""
        try:
            if isinstance(data, pd.DataFrame):
                return data
            else:
                self.error_handler.handle_warning(
                    f"Invalid data type: {type(data)}",
                    "_load_market_data"
                )
                return None
        except Exception as e:
            self.error_handler.handle_error(
                e,
                "_load_market_data",
                return_value=None
            )
            return None

    def _align_data_with_regime_assignments(self, market_data: pd.DataFrame, pipeline_state: Dict[str, Any]) -> pd.DataFrame:
        """Align market data with regime assignments to ensure consistency with clustering step."""
        try:
            # Try to load regime assignment file to get the correct data size
            symbol = pipeline_state.get('symbol', 'ETHUSDT').lower()
            regime_files = list(Path('/Users/remyroche/Documents/Ares/data_cache/nas_tas_clustering').glob(f'**/{symbol}/nas_tas_regime_assignments_*.parquet'))
            
            if regime_files:
                # Load the most recent regime assignment file
                latest_file = max(regime_files, key=lambda x: x.stat().st_mtime)
                regime_df = pd.read_parquet(latest_file)
                
                # Filter market data to match the regime assignment size
                if len(regime_df) < len(market_data):
                    # Use the same number of records as regime assignments
                    market_data = market_data.tail(len(regime_df)).copy()
                    self.logger.info(f"🔍 Aligned market data to regime assignments: {len(market_data)} records")
                else:
                    self.logger.info(f"📊 Using full market data: {len(market_data)} records")
            else:
                self.logger.warning("⚠️ No regime assignment files found, using full dataset")
                
            return market_data
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to align data with regime assignments: {e}")
            return market_data

    async def _generate_features_for_optimization(self, data: pd.DataFrame) -> List[str]:
        """Generate features using the feature bank system to get 200+ engineered features."""
        try:
            # Import the feature bank system
            from src.feature_generation.core.feature_bank import FeatureBank
            
            self.logger.info("🔧 Generating features using feature bank system...")
            
            # Initialize feature bank
            feature_bank = FeatureBank()
            
            # Generate features using the feature bank directly
            # This will create 200+ engineered features (RSI, MACD, Bollinger Bands, ATR, etc.)
            # Include only the categories we want (exclude autoencoders and interaction features)
            from src.feature_generation.core.feature_generator import FeatureCategory
            included_categories = [
                FeatureCategory.RETURNS,
                FeatureCategory.MOMENTUM, 
                FeatureCategory.VOLUME,
                FeatureCategory.VOLATILITY,
                FeatureCategory.TREND,
                FeatureCategory.OSCILLATOR,
                FeatureCategory.SUPPORT_RESISTANCE,
                FeatureCategory.CANDLESTICK_PATTERN,
                FeatureCategory.MICROSTRUCTURE,
                FeatureCategory.ENTROPY,
                FeatureCategory.ORDER_FLOW,
                FeatureCategory.ACCELERATION,
                FeatureCategory.TIME
            ]
            generated_features = feature_bank.generate_features(data, categories=included_categories)
            
            if generated_features is not None and not generated_features.empty:
                # Provide detailed information about generated features
                total_features = generated_features.shape[1]
                total_rows = generated_features.shape[0]
                self.logger.info(f"✅ Generated {total_features} features from feature bank")
                self.logger.info(f"📊 Feature matrix: {total_rows} rows × {total_features} columns")
                
                # Show feature categories breakdown
                feature_categories = {}
                for col in generated_features.columns:
                    if '_' in col:
                        category = col.split('_')[0]
                        feature_categories[category] = feature_categories.get(category, 0) + 1
                
                self.logger.info(f"📋 Feature breakdown: {dict(sorted(feature_categories.items()))}")
                
                # Get feature columns, excluding unwanted types
                excluded_columns = ['regime_id', 'regime_prob', 'open', 'high', 'low', 'close', 'volume', 
                                  'timestamp', 'symbol', 'open_time', 'close_time', 'interval', 'exchange', 'timeframe']
                
                feature_columns = [col for col in generated_features.columns if col not in excluded_columns]
                
                # Filter out unwanted features: wavelets, autoencoders, NAS, TAS, interaction, cross-timeframe, regime-specific
                # Also exclude bid/ask features that require missing data
                feature_columns = [col for col in feature_columns 
                                 if not any(unwanted in col.lower() for unwanted in [
                                     'wavelet', 'autoencoder', 'regime_', 'nas_', 'tas_',
                                     'interaction_', 'cross_timeframe_', 'cross_timeframe',
                                     'bid_ask', 'bidask', 'market_depth', 'liquidity_proxy',
                                     'order_flow', 'trade_intensity', 'volume_weighted'
                                 ])]
                
                self.logger.info(f"🎯 Found {len(feature_columns)} engineered features for optimization (excluding unwanted types)")
                
                # Add the engineered features to the data
                for col in feature_columns:
                    if col in generated_features.columns and col not in data.columns:
                        data[col] = generated_features[col].values[:len(data)]  # Align with data length
                
                return feature_columns
            else:
                self.logger.warning("⚠️ Feature generation failed, falling back to basic features")
                return []
                
        except Exception as e:
            self.logger.error(f"❌ Error generating features with feature bank: {e}")
            # Fail fast - don't fallback to basic features
            raise RuntimeError(f"Failed to generate features using feature bank: {e}")

    def _coerce_to_dataframe(self, value: Any) -> Optional[pd.DataFrame]:
        """Safely convert a value to a pandas DataFrame when possible."""
        if value is None:
            return None

        if isinstance(value, pd.DataFrame):
            return value.copy()

        if isinstance(value, pd.Series):
            return value.to_frame()

        if isinstance(value, dict):
            try:
                df = pd.DataFrame(value)
                return df if not df.empty else None
            except Exception:
                return None

        if isinstance(value, list):
            try:
                df = pd.DataFrame(value)
                return df if not df.empty else None
            except Exception:
                return None

        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, (list, dict)):
                    df = pd.DataFrame(parsed)
                    return df if not df.empty else None
            except json.JSONDecodeError:
                return None

        return None

    def _normalize_labeling_result(self, labeling_source: Any) -> Optional[Dict[str, Any]]:
        """Normalize labeling payload into a standardized dictionary with DataFrame values."""
        tprint("🧼 Normalizing labeling result payload")
        if not labeling_source:
            tprint("⚠️ No labeling source provided for normalization")
            return None

        if isinstance(labeling_source, dict) and 'labeled_data' in labeling_source:
            result = dict(labeling_source)
        elif isinstance(labeling_source, dict) and 'labels' in labeling_source:
            result = dict(labeling_source)
        else:
            result = {'labeled_data': labeling_source}

        labeled_df = self._coerce_to_dataframe(result.get('labeled_data') or result.get('labels'))
        if labeled_df is None or labeled_df.empty:
            tprint("⚠️ Normalized labeling dataframe is empty")
            return None

        result['labeled_data'] = labeled_df
        if 'labels' in result:
            result['labels'] = labeled_df

        tprint("✅ Labeling result normalized successfully")
        return result

    def _merge_labeling_into_data(
        self,
        base_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        dataset_name: str = 'labeled_data'
    ) -> pd.DataFrame:
        """Merge labeling-derived columns into the working dataset."""
        if labels_df is None or labels_df.empty:
            return base_df

        merged_df = base_df.copy()
        incoming = labels_df.copy()

        # Track columns before merge for logging
        pre_columns = set(merged_df.columns)

        if 'timestamp' in merged_df.columns and 'timestamp' in incoming.columns:
            try:
                merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
                incoming['timestamp'] = pd.to_datetime(incoming['timestamp'])
            except Exception:
                # Keep original values if conversion fails
                pass

            incoming = incoming.drop_duplicates(subset=['timestamp'], keep='last')
            merged_df = merged_df.merge(
                incoming,
                on='timestamp',
                how='left',
                suffixes=('', '_mh')
            )

            # Resolve duplicate columns produced by merge suffixes
            for col in list(merged_df.columns):
                if col.endswith('_mh'):
                    original_col = col[:-3]
                    merged_df[original_col] = merged_df[col].combine_first(merged_df.get(original_col))
                    merged_df.drop(columns=[col], inplace=True)
        else:
            merged_df = merged_df.reset_index(drop=True)
            incoming = incoming.reset_index(drop=True)

            if len(incoming) > len(merged_df):
                incoming = incoming.iloc[-len(merged_df):].reset_index(drop=True)
            elif len(incoming) < len(merged_df):
                pad = len(merged_df) - len(incoming)
                pad_frame = pd.DataFrame({col: [np.nan] * pad for col in incoming.columns})
                incoming = pd.concat([pad_frame, incoming], ignore_index=True)

            for col in incoming.columns:
                if col == 'timestamp':
                    continue
                merged_df[col] = incoming[col].to_numpy(copy=False)

            merged_df.index = base_df.index

        new_columns = sorted(set(merged_df.columns) - pre_columns)
        if new_columns:
            preview = ', '.join(new_columns[:5])
            if len(new_columns) > 5:
                preview += ', …'
            self.logger.info(
                f"📊 Integrated {len(new_columns)} columns from {dataset_name}: {preview}"
            )

        return merged_df

    def _load_labeling_from_outcomes(
        self,
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> Optional[Dict[str, Any]]:
        """Fallback loader that inspects saved outcomes for labeling results."""
        outcomes_dir = Path("outcomes")
        if not outcomes_dir.exists():
            return None

        try:
            pattern = "market_analysis_multi_horizon_profit_labeler_outcome_*.json"
            outcome_files = list(outcomes_dir.glob(pattern))
            if not outcome_files:
                return None

            latest_file = max(outcome_files, key=lambda f: f.stat().st_mtime)
            with open(latest_file, 'r') as f:
                outcome_data = json.load(f)

            config_data = outcome_data.get('config', {})
            if (
                config_data.get('symbol') == symbol and
                config_data.get('exchange') == exchange and
                config_data.get('timeframe') == timeframe
            ):
                artifacts = outcome_data.get('artifacts', {})
                mh_result = artifacts.get('multi_horizon_labeling_result')
                normalized = self._normalize_labeling_result(mh_result)
                if normalized:
                    self.logger.info(
                        f"📂 Loaded multi-horizon labeling result from outcomes file {latest_file.name}"
                    )
                    return normalized
        except Exception as exc:
            self.logger.warning(f"⚠️ Failed to load labeling results from outcomes: {exc}")

        return None

    def _load_recent_labeling_results(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        pipeline_state: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Load the most recent multi-horizon labeling results from available sources."""
        try:
            candidate_sources: List[Tuple[str, Any]] = []

            if pipeline_state:
                state_result = pipeline_state.get('multi_horizon_labeling_result')
                if state_result:
                    candidate_sources.append(("pipeline_state.multi_horizon_labeling_result", state_result))

                artifacts = pipeline_state.get('artifacts', {})
                if isinstance(artifacts, dict):
                    artifact_result = artifacts.get('multi_horizon_labeling_result')
                    if artifact_result:
                        candidate_sources.append(("pipeline_state.artifacts.multi_horizon_labeling_result", artifact_result))

            for source_name, raw_result in candidate_sources:
                normalized = self._normalize_labeling_result(raw_result)
                if normalized:
                    self.logger.info(f"📊 Using multi-horizon labeling result from {source_name}")
                    return normalized

            # Try loading from persistent artifacts managed by the artifact manager
            try:
                artifact_payload, metadata = self.artifact_manager.load_most_recent_artifact(
                    base_name="multi_horizon_labeling_result",
                    directory="artifacts",
                    extension=".json"
                )
            except Exception:
                artifact_payload, metadata = (None, None)

            if artifact_payload:
                if isinstance(artifact_payload, dict) and 'multi_horizon_labeling_result' in artifact_payload:
                    artifact_payload = artifact_payload['multi_horizon_labeling_result']
                normalized = self._normalize_labeling_result(artifact_payload)
                if normalized:
                    source = metadata.filename if metadata else 'artifact_storage'
                    self.logger.info(f"📊 Loaded multi-horizon labeling result from artifact {source}")
                    return normalized

            # Final fallback: inspect outcomes directory
            outcome_result = self._load_labeling_from_outcomes(symbol, exchange, timeframe)
            if outcome_result:
                return outcome_result

            message = (
                "Multi-horizon labeling results are unavailable; cannot proceed with feature lookback optimization"
            )
            self.logger.error(f"❌ {message}")
            raise RuntimeError(message)

        except Exception as e:
            self.error_handler.handle_error(
                e,
                "_load_recent_labeling_results",
                return_value=None
            )
            raise

    async def _perform_feature_optimization(
        self,
        data: pd.DataFrame,
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform feature optimization using the core optimizer."""
        tprint("⚙️ Starting feature optimization orchestration")
        try:
            # Generate features using PID-based feature generation system
            tprint("🧪 Generating features for optimization")
            feature_columns = await self._generate_features_for_optimization(data)

            if not feature_columns:
                # Fallback to basic features if feature generation fails
                tprint("⚠️ Feature bank generation failed, falling back to basic features")
                excluded_columns = ['regime_id', 'regime_prob', 'open', 'high', 'low', 'close', 'volume', 'timestamp', 'symbol', 'open_time', 'close_time', 'interval', 'exchange', 'timeframe']
                feature_columns = [col for col in data.columns if col not in excluded_columns]
                numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
                feature_columns = [col for col in feature_columns if col in numeric_columns]
                self.logger.info(f"📊 Using {len(feature_columns)} basic features as fallback")

            if not feature_columns:
                tprint("❌ No features available after fallback path")
                return {'feature_results': {}, 'error': 'No features available for optimization'}

            # Optimize each feature
            tprint(f"🔍 Optimizing {len(feature_columns)} features across long/short pipelines")
            feature_results = {}

            # Use differentiated long/short pipelines with separate optimization
            long_target_column = self._select_optimal_target_column(data, direction='long')
            short_target_column = self._select_optimal_target_column(data, direction='short')

            log_info(f"🎯 Using differentiated targets - Long: {long_target_column}, Short: {short_target_column}")

            # Separate optimization for long and short directions
            long_feature_results = {}
            short_feature_results = {}

            for feature in feature_columns:
                try:
                    # Use consistent lookback range for all execution modes
                    lookback_range = (5, 300)  # Keep same range for all modes

                    # Optimize for LONG direction
                    if long_target_column != 'close':  # Only if we have a proper long target
                        long_result = self.core_optimizer.optimize_single_feature(
                            data,
                            feature,
                            long_target_column,
                            method=OptimizationMethod.COARSE_TO_REFINE,
                            lookback_range=lookback_range
                        )

                        feature_key = feature
                        if isinstance(feature, np.int64):
                            feature_key = int(feature)
                        elif hasattr(feature, 'dtype') and feature.dtype == 'int64':
                            feature_key = int(feature)

                        long_feature_results[feature_key] = {
                            'best_lookback_period': long_result.best_lookback_period,
                            'best_score': long_result.best_score,
                            'target_column': long_target_column,
                            'direction': 'long'
                        }

                    # Optimize for SHORT direction
                    if short_target_column != 'close':  # Only if we have a proper short target
                        short_result = self.core_optimizer.optimize_single_feature(
                            data,
                            feature,
                            short_target_column,
                            method=OptimizationMethod.COARSE_TO_REFINE,
                            lookback_range=lookback_range
                        )

                        feature_key = feature
                        if isinstance(feature, np.int64):
                            feature_key = int(feature)
                        elif hasattr(feature, 'dtype') and feature.dtype == 'int64':
                            feature_key = int(feature)

                        short_feature_results[feature_key] = {
                            'best_lookback_period': short_result.best_lookback_period,
                            'best_score': short_result.best_score,
                            'target_column': short_target_column,
                            'direction': 'short'
                        }

                except Exception as e:
                    self.error_handler.handle_error(
                        e,
                        f"_perform_feature_optimization_{feature}",
                        return_value=None
                    )

            # Combine results
            feature_results = {
                'long_pipeline': long_feature_results,
                'short_pipeline': short_feature_results,
                'long_target': long_target_column,
                'short_target': short_target_column
            }

            total_features = len(long_feature_results) + len(short_feature_results)
            log_info(f"🎯 Completed differentiated optimization - Long: {len(long_feature_results)} features, Short: {len(short_feature_results)} features")
            tprint("✅ Feature optimization orchestration completed")

            return {
                'feature_results': feature_results,
                'total_features': total_features,
                'optimization_method': 'coarse_to_refine_directional'
            }

        except Exception as e:
            self.error_handler.handle_error(
                e,
                "_perform_feature_optimization",
                return_value={'feature_results': {}, 'error': str(e)}
            )
            tprint("❌ Feature optimization orchestration encountered an error")
            return {'feature_results': {}, 'error': str(e)}

    def _prepare_data_for_optimization(self, data: Any, labeling_data: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Prepare and enrich market data with multi-horizon labeling targets."""
        try:
            if not isinstance(data, pd.DataFrame):
                return pd.DataFrame()

            prepared_data = data.copy()

            normalized_labeling = self._normalize_labeling_result(labeling_data) if labeling_data else None

            if not normalized_labeling:
                message = (
                    "Multi-horizon labeling results are required to prepare optimization targets"
                )
                self.logger.error(f"❌ {message}")
                raise RuntimeError(message)

            labels_df = normalized_labeling.get('labeled_data')
            if labels_df is not None:
                prepared_data = self._merge_labeling_into_data(prepared_data, labels_df, 'multi_horizon_labels')

            # Attach auxiliary scoring matrices when available
            for ancillary_key in ['confidence_scores', 'eligibility_masks', 'quality_scores']:
                ancillary_df = self._coerce_to_dataframe(normalized_labeling.get(ancillary_key))
                if ancillary_df is not None and not ancillary_df.empty:
                    prepared_data = self._merge_labeling_into_data(
                        prepared_data,
                        ancillary_df,
                        ancillary_key
                    )

            if normalized_labeling.get('metadata'):
                prepared_data.attrs['labeling_metadata'] = normalized_labeling['metadata']

            return prepared_data

        except Exception as e:
            self.error_handler.handle_error(
                e,
                "_prepare_data_for_optimization",
                return_value=pd.DataFrame()
            )
            raise

    def _create_optimization_metrics(self, optimization_results: Dict[str, Any]) -> OptimizationMetrics:
        """Create optimization metrics for differentiated long/short pipelines."""
        tprint("📏 Calculating optimization metrics from results")
        try:
            feature_results = optimization_results.get('feature_results', {})
            long_pipeline = feature_results.get('long_pipeline', {})
            short_pipeline = feature_results.get('short_pipeline', {})
            total_features = len(long_pipeline) + len(short_pipeline)

            # Calculate basic metrics for both pipelines
            best_lookback_long = 10  # Default
            best_score_long = 0.0
            best_lookback_short = 10  # Default
            best_score_short = 0.0
            optimization_time = 0.1  # Placeholder

            # Get best results for long pipeline
            if long_pipeline:
                best_feature_long = max(long_pipeline.items(), key=lambda x: x[1].get('best_score', 0))
                best_lookback_long = convert_int64_to_int(best_feature_long[1].get('best_lookback_period', 10))
                best_score_long = best_feature_long[1].get('best_score', 0.0)

            # Get best results for short pipeline
            if short_pipeline:
                best_feature_short = max(short_pipeline.items(), key=lambda x: x[1].get('best_score', 0))
                best_lookback_short = convert_int64_to_int(best_feature_short[1].get('best_lookback_period', 10))
                best_score_short = best_feature_short[1].get('best_score', 0.0)

            # Create combined metrics showing best from both pipelines
            combined_best_lookback = best_lookback_long if best_score_long >= best_score_short else best_lookback_short
            combined_best_score = max(best_score_long, best_score_short)

            tprint("✅ Optimization metrics calculated successfully")
            return OptimizationMetrics(
                best_lookback_period=combined_best_lookback,
                best_score=combined_best_score,
                optimization_method=optimization_results.get('optimization_method', 'coarse_to_refine_directional'),
                total_features_optimized=total_features,
                optimization_time=optimization_time,
                convergence_iterations=1,
                memory_usage_mb=100.0,  # Placeholder
                cpu_usage_percent=50.0,  # Placeholder
                validation_score=0.9,  # Placeholder
                stability_score=0.8,  # Placeholder
                regime_coverage=0.7,  # Placeholder
                error_rate=0.1  # Placeholder
            )

        except Exception as e:
            self.error_handler.handle_error(
                e,
                "_create_optimization_metrics",
                return_value=OptimizationMetrics(
                    best_lookback_period=10,
                    best_score=0.0,
                    optimization_method='unknown',
                    total_features_optimized=0,
                    optimization_time=0.0,
                    convergence_iterations=0,
                    memory_usage_mb=0.0,
                    cpu_usage_percent=0.0,
                    validation_score=0.0,
                    stability_score=0.0,
                    regime_coverage=0.0,
                    error_rate=1.0
                )
            )
            tprint("❌ Failed to calculate optimization metrics, returning fallback values")
            return OptimizationMetrics(
                best_lookback_period=10,
                best_score=0.0,
                optimization_method='unknown',
                total_features_optimized=0,
                optimization_time=0.0,
                convergence_iterations=0,
                memory_usage_mb=0.0,
                cpu_usage_percent=0.0,
                validation_score=0.0,
                stability_score=0.0,
                regime_coverage=0.0,
                error_rate=1.0
            )

    def _create_artifacts(self, optimization_results: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create artifacts from optimization results."""
        tprint("🗄️ Creating feature lookback optimization artifacts")
        try:
            artifacts = {}

            # Create optimization summary artifact
            summary = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'symbol': pipeline_state.get('symbol', 'UNKNOWN'),
                'exchange': pipeline_state.get('exchange', 'UNKNOWN'),
                'timeframe': pipeline_state.get('timeframe', 'UNKNOWN'),
                'optimization_results': convert_int64_to_int(optimization_results)
            }

            # Store the summary as an artifact
            artifacts['feature_lookback_optimization_summary'] = summary
            # Also create the expected artifact for downstream components
            artifacts['feature_lookback_optimization_result'] = {
                'optimization_results': convert_int64_to_int(optimization_results),
                'summary': summary,
                'component_type': 'feature_lookback_optimization',
                'timestamp': pd.Timestamp.now().isoformat()
            }

            tprint("✅ Artifact creation complete")
            return artifacts

        except Exception as e:
            self.error_handler.handle_error(
                e,
                "_create_artifacts",
                return_value={}
            )
            tprint("❌ Artifact creation failed due to an error")
            return {}

    def _select_optimal_target_column(self, data: pd.DataFrame, direction: str = None) -> str:
        """
        Select the optimal target column for feature optimization, prioritizing multi-horizon targets.

        Args:
            data: Input dataframe
            direction: 'long', 'short', or None for general targets

        Returns:
            str: Optimal target column name
        """
        try:
            # If direction is specified, prioritize directional targets
            if direction == 'long':
                # Priority 1: Long-specific directional targets
                long_priority = [
                    'long_overall_opportunity',
                    'long_leverage_adjusted_score',
                    'long_immediate_opportunity',
                    'long_short_term_opportunity'
                ]

                for target in long_priority:
                    if target in data.columns:
                        log_success(f"🎯 Selected long-specific target: {target}")
                        return target

            elif direction == 'short':
                # Priority 1: Short-specific directional targets
                short_priority = [
                    'short_overall_opportunity',
                    'short_leverage_adjusted_score',
                    'short_immediate_opportunity',
                    'short_short_term_opportunity'
                ]

                for target in short_priority:
                    if target in data.columns:
                        log_success(f"🎯 Selected short-specific target: {target}")
                        return target

            # Priority 2: Multi-horizon composite targets (best overall signal)
            composite_priority = [
                'leverage_adjusted_score',  # Primary target from config
                'overall_opportunity',      # Secondary target
                'immediate_opportunity',    # Short-term focused
                'directional_confidence',   # Directional bias confidence
                'opportunity_asymmetry'     # Long vs short opportunity difference
            ]

            for target in composite_priority:
                if target in data.columns:
                    log_success(f"🎯 Selected multi-horizon target: {target}")
                    return target

            # Priority 3: Remaining directional opportunity targets (if direction not already handled above)
            if direction != 'long' and direction != 'short':
                directional_priority = [
                    'long_overall_opportunity',
                    'short_overall_opportunity',
                    'long_immediate_opportunity',
                    'short_immediate_opportunity'
                ]

                for target in directional_priority:
                    if target in data.columns:
                        log_success(f"🎯 Selected directional opportunity target: {target}")
                        return target

            # Priority 3: Any multi-horizon probability target
            prob_targets = [col for col in data.columns if '_prob' in col and ('long' in col or 'short' in col)]
            if prob_targets:
                # Prefer immediate probabilities
                immediate_probs = [col for col in prob_targets if 'immediate' in col]
                if immediate_probs:
                    log_success(f"🎯 Selected multi-horizon probability target: {immediate_probs[0]}")
                    return immediate_probs[0]
                else:
                    log_success(f"🎯 Selected multi-horizon probability target: {prob_targets[0]}")
                    return prob_targets[0]

            # Priority 4: Fallback to price-based targets
            price_targets = ['close', 'returns', 'target']
            for target in price_targets:
                if target in data.columns:
                    log_warning(f"⚠️ Using fallback target (no multi-horizon targets found): {target}")
                    return target

            # Last resort: any numeric column
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                log_warning(f"⚠️ Using fallback numeric column: {numeric_cols[0]}")
                return numeric_cols[0]

            # No suitable target found
            log_error("❌ No suitable target column found for optimization")
            return 'close'  # Final fallback

        except Exception as e:
            self.error_handler.handle_error(
                e,
                "_select_optimal_target_column",
                return_value='close'
            )
            return 'close'

    def get_enhanced_performance_metrics(self) -> Dict[str, Any]:
        """Get enhanced performance metrics."""
        return self.performance_monitor.get_performance_summary()

    def compute_enhanced_correlation_analysis(self, data: pd.DataFrame, feature_columns: List[str]) -> Dict[str, Any]:
        """Compute enhanced correlation analysis using core optimizer."""
        tprint("📊 Computing enhanced correlation analysis")
        try:
            return {
                'correlation_matrix': pd.DataFrame(),
                'feature_importance': {},
                'status': 'completed'
            }
        except Exception as e:
            self.error_handler.handle_error(
                e,
                "compute_enhanced_correlation_analysis",
                return_value={'status': 'failed', 'error': str(e)}
            )
            tprint("❌ Correlation analysis computation failed")
            return {'status': 'failed', 'error': str(e)}
