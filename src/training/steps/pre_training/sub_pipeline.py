"""
Pre-Training Sub-Pipeline - Feature Engineering Steps

This module provides the pre-training sub-pipeline with the 4 feature engineering steps
that were moved from market_analysis:

1. multi_horizon_profit_labeler - Apply multi-horizon profit labeling
2. feature_lookback_optimization - Optimize feature lookback periods
3. pid_based_feature_generation - PID-based feature generation with interaction, polynomial, and cross-timeframe features
4. final_feature_selection - Final multi-stage feature selection (120→100→80→60)
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.utils.enhanced_artifact_manager import get_artifact_manager
from src.utils.version_manager import get_version_manager
from src.utils.tprint import tprint

logger = system_logger.getChild('PreTrainingSubPipeline')

class ExecutionMode(Enum):
    """Execution modes for sub-pipelines."""
    FULL = "full"          # Complete execution with all features
    LIGHT = "light"        # Lightweight execution with essential features only
    BLANK = "blank"        # Minimal execution for testing/validation

class SubPipelineStatus(Enum):
    """Status of sub-pipeline execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class LoggingConfig:
    """Logging configuration for the sub-pipeline."""
    level: str = "INFO"
    enable_console: bool = True
    enable_file: bool = False
    log_file: Optional[str] = None

@dataclass
class SubPipelineConfig:
    """Configuration for sub-pipeline execution."""
    mode: ExecutionMode = ExecutionMode.FULL
    symbol: str = "ETHUSDT"
    exchange: str = "binance"
    timeframe: str = "15m"
    data_dir: str = "historical_data"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    force_rerun: bool = False
    parallel_processing: bool = True
    max_workers: int = 4
    validation_enabled: bool = True
    monitoring_enabled: bool = True
    fast_mode: bool = False
    skip_next_pipeline: bool = False

@dataclass
class SubPipelineResult:
    """Result of sub-pipeline execution."""
    sub_pipeline_name: str
    status: SubPipelineStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    success: bool = False
    output_files: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

class PreTrainingSubPipeline:
    """
    Pre-Training Sub-Pipeline for Feature Engineering Steps.

    Executes the 4 feature engineering steps in sequence:
    1. multi_horizon_profit_labeler
    2. feature_lookback_optimization
    3. pid_based_feature_generation
    4. final_feature_selection
    """

    def __init__(self):
        """Initialize the pre-training sub-pipeline."""
        self.logger = logger.getChild('PreTrainingSubPipeline')
        self.results: List[SubPipelineResult] = []
        self._current_pipeline_state: Dict[str, Any] = {}

    async def execute_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """
        Execute the complete pre-training pipeline.

        Args:
            config: Configuration for pipeline execution

        Returns:
            Dictionary containing execution results
        """
        self.logger.info('🚀 Starting Pre-Training Sub-Pipeline execution')
        self.logger.info(f'📊 Symbol: {config.symbol}, Exchange: {config.exchange}')
        self.logger.info(f'⏰ Timeframe: {config.timeframe}, Mode: {config.mode.value}')

        start_time = datetime.now()
        results = {
            'success': False,
            'execution_time': 0.0,
            'total_steps': 4,
            'completed_steps': 0,
            'results': {}
        }

        try:
            # Step 1: Multi-Horizon Profit Labeler
            self.logger.info('🎯 Step 1: Multi-Horizon Profit Labeler')
            mh_result = await self._execute_multi_horizon_profit_labeler(config)
            if not mh_result.success:
                self.logger.error(f'❌ Multi-horizon profit labeling failed: {mh_result.error_message}')
                return results

            results['results']['multi_horizon_profit_labeler'] = mh_result.artifacts
            self._current_pipeline_state.update(mh_result.artifacts)

            # Step 2: Feature Lookback Optimization
            self.logger.info('⚙️ Step 2: Feature Lookback Optimization')
            flo_result = await self._execute_feature_lookback_optimization(config)
            if not flo_result.success:
                self.logger.error(f'❌ Feature lookback optimization failed: {flo_result.error_message}')
                return results

            results['results']['feature_lookback_optimization'] = flo_result.artifacts
            self._current_pipeline_state.update(flo_result.artifacts)

            # Step 3: PID-Based Feature Generation
            self.logger.info('🔧 Step 3: PID-Based Feature Generation')
            pid_result = await self._execute_pid_based_feature_generation(config)
            if not pid_result.success:
                self.logger.error(f'❌ PID-based feature generation failed: {pid_result.error_message}')
                return results

            results['results']['pid_based_feature_generation'] = pid_result.artifacts
            self._current_pipeline_state.update(pid_result.artifacts)

            # Step 4: Final Feature Selection
            self.logger.info('🎯 Step 4: Final Feature Selection')
            ffs_result = await self._execute_final_feature_selection(config)
            if not ffs_result.success:
                self.logger.error(f'❌ Final feature selection failed: {ffs_result.error_message}')
                return results

            results['results']['final_feature_selection'] = ffs_result.artifacts
            self._current_pipeline_state.update(ffs_result.artifacts)

            # Success
            end_time = datetime.now()
            results['success'] = True
            results['execution_time'] = (end_time - start_time).total_seconds()
            results['completed_steps'] = 4

            self.logger.info(f'🎉 Pre-Training Sub-Pipeline completed successfully in {results["execution_time"]:.2f}s')

        except Exception as e:
            self.logger.error(f'❌ Pre-Training Sub-Pipeline failed with exception: {e}')
            results['error_message'] = str(e)

        return results

    async def _execute_multi_horizon_profit_labeler(self, config: SubPipelineConfig) -> SubPipelineResult:
        """Execute multi-horizon profit labeler."""
        result = SubPipelineResult(
            sub_pipeline_name='multi_horizon_profit_labeler',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )

        try:
            from .multi_horizon_profit_labeler import (
                MultiHorizonProfitLabeler, MultiHorizonConfig, apply_multi_horizon_labeling
            )

            mh_config = MultiHorizonConfig(
                timeframe=config.timeframe
            )

            labeler = MultiHorizonProfitLabeler(mh_config)

            # Apply labeling to the current data
            if self._current_data is not None:
                labeled_data = labeler.generate_labels(self._current_data)
                labeling_metrics = {
                    'total_samples': len(labeled_data) if labeled_data is not None else 0,
                    'labeled_samples': len(labeled_data) if labeled_data is not None else 0,
                    'profit_labels': 0,
                    'loss_labels': 0
                }
            else:
                labeled_data = {}
                labeling_metrics = {
                    'total_samples': 0,
                    'labeled_samples': 0,
                    'profit_labels': 0,
                    'loss_labels': 0
                }

            labeling_result = {
                'success': True,
                'labeled_data': labeled_data,
                'labeling_metrics': labeling_metrics
            }

            result.status = SubPipelineStatus.COMPLETED
            result.success = True
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            result.artifacts = labeling_result

        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()

        return result

    async def _execute_feature_lookback_optimization(self, config: SubPipelineConfig) -> SubPipelineResult:
        """Execute feature lookback optimization."""
        result = SubPipelineResult(
            sub_pipeline_name='feature_lookback_optimization',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )

        try:
            from .components.feature_lookback_optimization import FeatureLookbackOptimizationComponent

            component = FeatureLookbackOptimizationComponent()

            # Execute feature lookback optimization
            component_result = await component.execute(self._current_data, self._current_pipeline_state)

            if component_result.success:
                optimization_result = {
                    'success': True,
                    'optimized_features': component_result.artifacts,
                    'optimization_metrics': component_result.metadata or {}
                }
            else:
                raise Exception(f"Component execution failed: {component_result.error_message}")

            result.status = SubPipelineStatus.COMPLETED
            result.success = True
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            result.artifacts = optimization_result

        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()

        return result

    async def _execute_pid_based_feature_generation(self, config: SubPipelineConfig) -> SubPipelineResult:
        """Execute PID-based feature generation."""
        result = SubPipelineResult(
            sub_pipeline_name='pid_based_feature_generation',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )

        try:
            from .pid_based_feature_generation import PIDBasedFeatureGeneration, PIDBasedFeatureGenerationConfig

            pid_config = PIDBasedFeatureGenerationConfig(
                symbol=config.symbol,
                exchange=config.exchange,
                timeframe=config.timeframe,
                data_dir=config.data_dir
            )

            generator = PIDBasedFeatureGeneration(pid_config)

            # Generate PID-based features
            generation_result = await generator.generate_features(self._current_data, self._current_pipeline_state)

            if generation_result.success:
                pid_result = {
                    'success': True,
                    'pid_based_features': generation_result.features,
                    'pid_feature_metrics': generation_result.generation_metrics
                }
            else:
                raise Exception(f"PID-based feature generation failed: {generation_result.error_message}")

            result.status = SubPipelineStatus.COMPLETED
            result.success = True
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            result.artifacts = pid_result

        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()

        return result

    async def _execute_final_feature_selection(self, config: SubPipelineConfig) -> SubPipelineResult:
        """Execute final feature selection."""
        result = SubPipelineResult(
            sub_pipeline_name='final_feature_selection',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )

        try:
            from .components.final_feature_selection import FinalFeatureSelectionComponent

            component = FinalFeatureSelectionComponent()

            # Execute final feature selection
            component_result = await component.execute(self._current_data, self._current_pipeline_state)

            if component_result.success:
                selection_result = {
                    'success': True,
                    'final_feature_selection': component_result.artifacts
                }
            else:
                raise Exception(f"Component execution failed: {component_result.error_message}")

            result.status = SubPipelineStatus.COMPLETED
            result.success = True
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            result.artifacts = selection_result

        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()

        return result

# Convenience function for direct execution
async def execute_pre_training_pipeline(config: SubPipelineConfig) -> Dict[str, Any]:
    """
    Execute the pre-training pipeline with the given configuration.

    Args:
        config: Configuration for pipeline execution

    Returns:
        Dictionary containing execution results
    """
    pipeline = PreTrainingSubPipeline()
    return await pipeline.execute_pipeline(config)