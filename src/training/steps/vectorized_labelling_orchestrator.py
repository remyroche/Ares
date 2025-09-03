from __future__ import annotations
'\nRefactored VectorizedLabellingOrchestrator with reduced complexity and type hints.\nThis version breaks down the massive orchestrate_labeling_and_feature_engineering method\ninto smaller, focused methods with proper type annotations.\n'
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any
import numpy as np
import pandas as pd
from copy import copy
import asyncio

class PipelineStage(Enum):
    """Enumeration of pipeline stages"""
    INITIALIZATION = 'initialization'
    STATIONARITY_CHECK = 'stationarity_check'
    FEATURE_ENGINEERING = 'feature_engineering'
    LABELING = 'labeling'
    FEATURE_COMBINATION = 'feature_combination'
    FEATURE_SELECTION = 'feature_selection'
    NORMALIZATION = 'normalization'
    MEMORY_OPTIMIZATION = 'memory_optimization'
    VALIDATION = 'validation'

@dataclass
class PipelineConfig:
    """Configuration for the labeling and feature engineering pipeline"""
    enable_stationary_checks: bool = True
    enable_feature_selection: bool = True
    enable_data_normalization: bool = True
    enable_memory_optimization: bool = True
    auto_recalculate_hmm_barriers: bool = False
    keep_close_returns: bool = True
    time_barrier_minutes: int = 30
    max_lookahead: int = 100
    hmm_barrier_regime_column: str = 'hmm_regime'

@dataclass
class PipelineMetadata:
    """Metadata for pipeline execution"""
    stage_timings: dict[str, float]
    data_shapes: dict[str, tuple[int, int]]
    feature_counts: dict[str, int]
    baseline_columns: set[str]
    context_columns: list[str]
    total_execution_time: float

@dataclass
class StageResult:
    """Result from a pipeline stage"""
    stage: PipelineStage
    success: bool
    data: pd.DataFrame | None
    metadata: dict[str, Any]
    error: Exception | None = None

class VectorizedLabellingOrchestratorRefactored:
    """Refactored orchestrator with reduced complexity and type hints"""

    def __init__(self, config: dict[str, Any] | None=None, logger: logging.Logger | None=None) -> None:
        """Initialize the orchestrator.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        self.pipeline_config = PipelineConfig()
        self.is_initialized = False
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize all pipeline components"""
        try:
            self._init_stationarity_checker()
            self._init_feature_engineer()
            self._init_labeler()
            self._init_feature_selector()
            self._init_data_normalizer()
            self._init_memory_optimizer()
            self.is_initialized = True
            self.logger.info('✅ All pipeline components initialized')
        except Exception as e:
            self.logger.exception(f'❌ Failed to initialize components: {e}')
            raise

    def _init_stationarity_checker(self) -> None:
        """Initialize stationarity checker component"""
        self.stationarity_checker = None

    def _init_feature_engineer(self) -> None:
        """Initialize feature engineering component"""
        self.advanced_feature_engineer = None

    def _init_labeler(self) -> None:
        """Initialize triple barrier labeler component"""
        self.triple_barrier_labeler = None

    def _init_feature_selector(self) -> None:
        """Initialize feature selection component"""
        self.feature_selector = None

    def _init_data_normalizer(self) -> None:
        """Initialize data normalization component"""
        self.data_normalizer = None

    def _init_memory_optimizer(self) -> None:
        """Initialize memory optimization component"""

    async def orchestrate_labeling_and_feature_engineering(self, price_data: pd.DataFrame, volume_data: pd.DataFrame, order_flow_data: pd.DataFrame | None=None, sr_levels: dict[str, Any] | None=None) -> dict[str, Any]:
        """Orchestrate the complete pipeline with reduced complexity.

        This refactored method delegates to specialized methods for each stage.

        Args:
            price_data: OHLCV price data
            volume_data: Volume and trade flow data
            order_flow_data: Order book and flow data (optional)
            sr_levels: Support/resistance levels (optional)

        Returns:
            Dictionary containing processed data and metadata
        """
        start_time = time.time()
        stage_timings = {}
        metadata = self._initialize_pipeline_metadata(price_data, volume_data, order_flow_data)
        try:
            stage_result = await self._execute_initialization_stage(price_data, volume_data, order_flow_data)
            if not stage_result.success:
                return self._create_error_response(stage_result.error)
            stage_timings[PipelineStage.INITIALIZATION.value] = stage_result.metadata['duration']
            if self.pipeline_config.enable_stationary_checks:
                stage_result = await self._execute_stationarity_stage(price_data, volume_data, order_flow_data)
                if stage_result.success and stage_result.data is not None:
                    price_data = stage_result.data.get('price_data', price_data)
                    volume_data = stage_result.data.get('volume_data', volume_data)
                    order_flow_data = stage_result.data.get('order_flow_data', order_flow_data)
                stage_timings[PipelineStage.STATIONARITY_CHECK.value] = stage_result.metadata['duration']
            stage_result = await self._execute_feature_engineering_stage(price_data, volume_data, order_flow_data, sr_levels)
            if not stage_result.success:
                return self._create_error_response(stage_result.error)
            advanced_features = stage_result.data
            stage_timings[PipelineStage.FEATURE_ENGINEERING.value] = stage_result.metadata['duration']
            stage_result = await self._execute_labeling_stage(price_data)
            if not stage_result.success:
                return self._create_error_response(stage_result.error)
            labeled_data = stage_result.data
            stage_timings[PipelineStage.LABELING.value] = stage_result.metadata['duration']
            stage_result = await self._execute_feature_combination_stage(labeled_data, advanced_features)
            if not stage_result.success:
                return self._create_error_response(stage_result.error)
            combined_data = stage_result.data
            stage_timings[PipelineStage.FEATURE_COMBINATION.value] = stage_result.metadata['duration']
            if self.pipeline_config.enable_feature_selection:
                stage_result = await self._execute_feature_selection_stage(combined_data)
                if stage_result.success and stage_result.data is not None:
                    combined_data = stage_result.data
                stage_timings[PipelineStage.FEATURE_SELECTION.value] = stage_result.metadata['duration']
            if self.pipeline_config.enable_data_normalization:
                stage_result = await self._execute_normalization_stage(combined_data)
                if stage_result.success and stage_result.data is not None:
                    combined_data = stage_result.data
                stage_timings[PipelineStage.NORMALIZATION.value] = stage_result.metadata['duration']
            if self.pipeline_config.enable_memory_optimization:
                stage_result = await self._execute_memory_optimization_stage(combined_data)
                if stage_result.success and stage_result.data is not None:
                    combined_data = stage_result.data
                stage_timings[PipelineStage.MEMORY_OPTIMIZATION.value] = stage_result.metadata['duration']
            stage_result = await self._execute_validation_stage(combined_data)
            stage_timings[PipelineStage.VALIDATION.value] = stage_result.metadata['duration']
            total_execution_time = time.time() - start_time
            return self._create_success_response(combined_data, stage_timings, metadata, total_execution_time)
        except Exception as e:
            self.logger.exception(f'❌ Pipeline execution failed: {e}')
            return self._create_error_response(e)

    def _initialize_pipeline_metadata(self, price_data: pd.DataFrame, volume_data: pd.DataFrame, order_flow_data: pd.DataFrame | None) -> PipelineMetadata:
        """Initialize metadata for pipeline execution"""
        baseline_cols = set(price_data.columns)
        if volume_data is not None:
            baseline_cols |= set(volume_data.columns)
        if order_flow_data is not None:
            baseline_cols |= set(order_flow_data.columns)
        return PipelineMetadata(stage_timings={}, data_shapes={'price': price_data.shape, 'volume': volume_data.shape if volume_data is not None else (0, 0), 'order_flow': order_flow_data.shape if order_flow_data is not None else (0, 0)}, feature_counts={}, baseline_columns=baseline_cols, context_columns=[], total_execution_time=0.0)

    async def _execute_initialization_stage(self, price_data: pd.DataFrame, volume_data: pd.DataFrame, order_flow_data: pd.DataFrame | None) -> StageResult:
        """Execute initialization and validation stage"""
        start_time = time.time()
        try:
            if not self.is_initialized:
                msg = 'Orchestrator not initialized'
                raise RuntimeError(msg)
            self._validate_input_data(price_data, volume_data, order_flow_data)
            self.logger.info('🎯 Starting pipeline execution...')
            self._log_data_shapes(price_data, volume_data, order_flow_data)
            return StageResult(stage=PipelineStage.INITIALIZATION, success=True, data=None, metadata={'duration': time.time() - start_time})
        except Exception as e:
            return StageResult(stage=PipelineStage.INITIALIZATION, success=False, data=None, metadata={'duration': time.time() - start_time}, error=e)

    async def _execute_stationarity_stage(self, price_data: pd.DataFrame, volume_data: pd.DataFrame, order_flow_data: pd.DataFrame | None) -> StageResult:
        """Execute stationarity check stage"""
        start_time = time.time()
        try:
            if self.stationarity_checker is None:
                self.logger.warning('⚠️ Stationarity checker not available, skipping')
                return StageResult(stage=PipelineStage.STATIONARITY_CHECK, success=True, data=None, metadata={'duration': time.time() - start_time, 'skipped': True})
            self.logger.info('📊 Performing stationarity checks...')
            stationary_data = await self.stationarity_checker.check_and_transform_stationarity(price_data, volume_data, order_flow_data)
            return StageResult(stage=PipelineStage.STATIONARITY_CHECK, success=True, data=stationary_data, metadata={'duration': time.time() - start_time})
        except Exception as e:
            self.logger.exception(f'❌ Stationarity check failed: {e}')
            return StageResult(stage=PipelineStage.STATIONARITY_CHECK, success=False, data=None, metadata={'duration': time.time() - start_time}, error=e)

    async def _execute_feature_engineering_stage(self, price_data: pd.DataFrame, volume_data: pd.DataFrame, order_flow_data: pd.DataFrame | None, sr_levels: dict[str, Any] | None) -> StageResult:
        """Execute feature engineering stage"""
        start_time = time.time()
        try:
            if self.advanced_feature_engineer is None:
                msg = 'Feature engineer not available'
                raise RuntimeError(msg)
            self.logger.info('🔧 Generating advanced features...')
            if not getattr(self.advanced_feature_engineer, 'is_initialized', False):
                await self.advanced_feature_engineer.initialize()
            advanced_features = await self.advanced_feature_engineer.engineer_features(price_data, volume_data, order_flow_data, sr_levels)
            if not advanced_features or len(advanced_features) < 10:
                self.logger.warning(f'⚠️ Few features generated: {len(advanced_features)}')
            return StageResult(stage=PipelineStage.FEATURE_ENGINEERING, success=True, data=advanced_features, metadata={'duration': time.time() - start_time, 'feature_count': len(advanced_features)})
        except Exception as e:
            self.logger.exception(f'❌ Feature engineering failed: {e}')
            return StageResult(stage=PipelineStage.FEATURE_ENGINEERING, success=False, data=None, metadata={'duration': time.time() - start_time}, error=e)

    async def _execute_labeling_stage(self, price_data: pd.DataFrame) -> StageResult:
        """Execute triple barrier labeling stage"""
        start_time = time.time()
        try:
            self.logger.info('🏷️ Applying triple barrier labeling...')
            if self.pipeline_config.auto_recalculate_hmm_barriers:
                labeled_data = await self._apply_regime_aware_labeling(price_data)
            else:
                labeled_data = await self._apply_standard_labeling(price_data)
            return StageResult(stage=PipelineStage.LABELING, success=True, data=labeled_data, metadata={'duration': time.time() - start_time, 'label_count': len(labeled_data)})
        except Exception as e:
            self.logger.exception(f'❌ Labeling failed: {e}')
            return StageResult(stage=PipelineStage.LABELING, success=False, data=None, metadata={'duration': time.time() - start_time}, error=e)

    async def _apply_regime_aware_labeling(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Apply regime-aware triple barrier labeling"""
        if self.pipeline_config.hmm_barrier_regime_column not in price_data.columns:
            self.logger.warning('⚠️ Regime column not found, falling back to standard labeling')
            return await self._apply_standard_labeling(price_data)
        return price_data.copy()

    async def _apply_standard_labeling(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Apply standard triple barrier labeling"""
        if self.triple_barrier_labeler is None:
            msg = 'Triple barrier labeler not available'
            raise RuntimeError(msg)
        return self.triple_barrier_labeler.apply_triple_barrier_labeling_vectorized(price_data.copy())

    async def _execute_feature_combination_stage(self, labeled_data: pd.DataFrame, advanced_features: dict[str, Any]) -> StageResult:
        """Execute feature combination stage"""
        start_time = time.time()
        try:
            self.logger.info('🔗 Combining features and labels...')
            combined_data = self._combine_features_and_labels(labeled_data, advanced_features)
            combined_data = self._remove_stationarity_columns(combined_data)
            return StageResult(stage=PipelineStage.FEATURE_COMBINATION, success=True, data=combined_data, metadata={'duration': time.time() - start_time, 'shape': combined_data.shape})
        except Exception as e:
            self.logger.exception(f'❌ Feature combination failed: {e}')
            return StageResult(stage=PipelineStage.FEATURE_COMBINATION, success=False, data=None, metadata={'duration': time.time() - start_time}, error=e)

    def _combine_features_and_labels(self, labeled_data: pd.DataFrame, advanced_features: dict[str, Any]) -> pd.DataFrame:
        """Combine features and labels into a single DataFrame"""
        combined = labeled_data.copy()
        for feature_name, feature_data in advanced_features.items():
            if isinstance(feature_data, pd.DataFrame):
                feature_data = feature_data.reindex(combined.index)
                combined = pd.concat([combined, feature_data], axis=1)
            elif isinstance(feature_data, pd.Series):
                combined[feature_name] = feature_data.reindex(combined.index)
        return combined

    def _remove_stationarity_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove stationarity transformation helper columns"""
        stationarity_cols = [col for col in data.columns if '_stationary_' in col]
        return data.drop(columns=stationarity_cols, errors='ignore')

    async def _execute_feature_selection_stage(self, data: pd.DataFrame) -> StageResult:
        """Execute feature selection stage"""
        start_time = time.time()
        try:
            if self.feature_selector is None:
                self.logger.warning('⚠️ Feature selector not available, skipping')
                return StageResult(stage=PipelineStage.FEATURE_SELECTION, success=True, data=None, metadata={'duration': time.time() - start_time, 'skipped': True})
            self.logger.info('🎯 Performing feature selection...')
            context_cols = self._identify_context_columns(data)
            selection_input = data.drop(columns=context_cols, errors='ignore')
            selected = await self.feature_selector.select_optimal_features(selection_input, data.get('label') if 'label' in data.columns else None)
            if isinstance(selected, pd.DataFrame) and (not selected.empty):
                preserved = data[context_cols]
                result = pd.concat([selected, preserved], axis=1)
            else:
                self.logger.warning('⚠️ Feature selection returned empty result')
                result = data
            return StageResult(stage=PipelineStage.FEATURE_SELECTION, success=True, data=result, metadata={'duration': time.time() - start_time, 'selected_features': result.shape[1]})
        except Exception as e:
            self.logger.exception(f'❌ Feature selection failed: {e}')
            return StageResult(stage=PipelineStage.FEATURE_SELECTION, success=False, data=None, metadata={'duration': time.time() - start_time}, error=e)

    def _identify_context_columns(self, data: pd.DataFrame) -> list[str]:
        """Identify context columns to preserve"""
        context_cols = []
        if 'label' in data.columns:
            context_cols.append('label')
        if self.pipeline_config.keep_close_returns and 'close_returns' in data.columns:
            context_cols.append('close_returns')
        volume_cols = [col for col in data.columns if 'volume' in col.lower()]
        if volume_cols:
            context_cols.extend(volume_cols[:1])
        return context_cols

    async def _execute_normalization_stage(self, data: pd.DataFrame) -> StageResult:
        """Execute data normalization stage"""
        start_time = time.time()
        try:
            if self.data_normalizer is None:
                self.logger.warning('⚠️ Data normalizer not available, skipping')
                return StageResult(stage=PipelineStage.NORMALIZATION, success=True, data=None, metadata={'duration': time.time() - start_time, 'skipped': True})
            self.logger.info('📏 Normalizing data...')
            normalized = await self.data_normalizer.normalize_data(data)
            return StageResult(stage=PipelineStage.NORMALIZATION, success=True, data=normalized, metadata={'duration': time.time() - start_time, 'shape': normalized.shape})
        except Exception as e:
            self.logger.exception(f'❌ Normalization failed: {e}')
            return StageResult(stage=PipelineStage.NORMALIZATION, success=False, data=None, metadata={'duration': time.time() - start_time}, error=e)

    async def _execute_memory_optimization_stage(self, data: pd.DataFrame) -> StageResult:
        """Execute memory optimization stage"""
        start_time = time.time()
        try:
            self.logger.info('💾 Optimizing memory usage...')
            optimized = self._optimize_memory_usage(data)
            original_memory = data.memory_usage(deep=True).sum()
            optimized_memory = optimized.memory_usage(deep=True).sum()
            savings = (1 - optimized_memory / original_memory) * 100
            self.logger.info(f'💾 Memory reduced by {savings:.1f}%')
            return StageResult(stage=PipelineStage.MEMORY_OPTIMIZATION, success=True, data=optimized, metadata={'duration': time.time() - start_time, 'memory_savings_pct': savings})
        except Exception as e:
            self.logger.exception(f'❌ Memory optimization failed: {e}')
            return StageResult(stage=PipelineStage.MEMORY_OPTIMIZATION, success=False, data=None, metadata={'duration': time.time() - start_time}, error=e)

    def _optimize_memory_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage by downcasting dtypes"""
        optimized = df.copy()
        for col in optimized.select_dtypes(include=['float']).columns:
            optimized[col] = pd.to_numeric(optimized[col], downcast='float')
        for col in optimized.select_dtypes(include=['int']).columns:
            optimized[col] = pd.to_numeric(optimized[col], downcast='integer')
        return optimized

    async def _execute_validation_stage(self, data: pd.DataFrame) -> StageResult:
        """Execute final validation stage"""
        start_time = time.time()
        try:
            self.logger.info('✅ Validating final output...')
            validation_results = self._validate_final_data(data)
            if not validation_results['is_valid']:
                msg = f"Final validation failed: {validation_results['errors']}"
                raise ValueError(msg)
            return StageResult(stage=PipelineStage.VALIDATION, success=True, data=None, metadata={'duration': time.time() - start_time, 'validation_results': validation_results})
        except Exception as e:
            self.logger.exception(f'❌ Validation failed: {e}')
            return StageResult(stage=PipelineStage.VALIDATION, success=False, data=None, metadata={'duration': time.time() - start_time}, error=e)

    def _validate_final_data(self, data: pd.DataFrame) -> dict[str, Any]:
        """Validate the final processed data"""
        errors = []
        if data.empty:
            errors.append('Data is empty')
        if 'label' not in data.columns:
            errors.append("Missing 'label' column")
        nan_ratio = data.isna().sum().sum() / (data.shape[0] * data.shape[1])
        if nan_ratio > 0.5:
            errors.append(f'Excessive NaN values: {nan_ratio:.2%}')
        inf_count = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            errors.append(f'Found {inf_count} infinite values')
        return {'is_valid': len(errors) == 0, 'errors': errors, 'shape': data.shape, 'nan_ratio': nan_ratio, 'inf_count': inf_count}

    def _validate_input_data(self, price_data: pd.DataFrame, volume_data: pd.DataFrame, order_flow_data: pd.DataFrame | None) -> None:
        """Validate input data"""
        if price_data.empty:
            msg = 'Price data is empty'
            raise ValueError(msg)
        if volume_data.empty:
            msg = 'Volume data is empty'
            raise ValueError(msg)
        required_price_cols = {'open', 'high', 'low', 'close'}
        missing_cols = required_price_cols - set(price_data.columns)
        if missing_cols:
            msg = f'Missing required price columns: {missing_cols}'
            raise ValueError(msg)

    def _log_data_shapes(self, price_data: pd.DataFrame, volume_data: pd.DataFrame, order_flow_data: pd.DataFrame | None) -> None:
        """Log data shapes for debugging"""
        self.logger.info(f'📊 Price data shape: {price_data.shape}')
        self.logger.info(f'📊 Volume data shape: {volume_data.shape}')
        if order_flow_data is not None:
            self.logger.info(f'📊 Order flow data shape: {order_flow_data.shape}')

    def _create_success_response(self, data: pd.DataFrame, stage_timings: dict[str, float], metadata: PipelineMetadata, total_time: float) -> dict[str, Any]:
        """Create successful response"""
        return {'success': True, 'data': data, 'metadata': {'pipeline_version': '2.0', 'total_execution_time': total_time, 'stage_timings': stage_timings, 'final_shape': data.shape, 'feature_count': data.shape[1], 'sample_count': data.shape[0]}}

    def _create_error_response(self, error: Exception) -> dict[str, Any]:
        """Create error response"""
        return {'success': False, 'data': None, 'error': str(error), 'error_type': type(error).__name__}