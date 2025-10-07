"""
Pre-Training Sub-Pipeline - Feature Engineering Steps

This module provides the pre-training sub-pipeline with the 4 feature engineering steps
that were moved from market_analysis:

1. multi_horizon_profit_labeler - Apply multi-horizon profit labeling
2. feature_lookback_optimization - Optimize feature lookback periods
3. interactive_feature_generation - End-to-end interactive feature generation with comprehensive approach
4. final_feature_selection - Final multi-stage feature selection (120→100→80→60)

Each step can receive a timeframe parameter, with default 15m.
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

# Import component system
from .components import ComponentFactory, ComponentConfig

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
    timeframe: str = "1h"  # Default timeframe for pre-training steps (analyst)
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
    custom_params: Dict[str, Any] = field(default_factory=dict)

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
    3. interactive_feature_generation
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
        tprint(f"🚀 Starting Pre-Training Sub-Pipeline execution for {config.symbol} on {config.exchange}")
        tprint(f"⏰ Timeframe: {config.timeframe}, Mode: {config.mode.value}")
        tprint(f"📊 Configuration: force_rerun={config.force_rerun}, parallel={config.parallel_processing}")

        results = {
            'success': False,
            'execution_time': 0.0,
            'total_steps': 4,
            'completed_steps': 0,
            'results': {}
        }

        try:
            # Step 1: Multi-Horizon Profit Labeler
            tprint("🎯 Step 1: Multi-Horizon Profit Labeler")
            self.logger.info('🎯 Step 1: Multi-Horizon Profit Labeler')
            mh_result = await self._execute_multi_horizon_profit_labeler(config)
            if not mh_result.success:
                tprint(f"❌ Multi-horizon profit labeling failed: {mh_result.error_message}")
                self.logger.error(f'❌ Multi-horizon profit labeling failed: {mh_result.error_message}')
                return results

            tprint(f"✅ Multi-horizon profit labeling completed for {config.symbol}")
            tprint(f"   → Labels generated: {len(mh_result.artifacts.get('multi_horizon_labeling_result', {}).get('labeled_data', pd.DataFrame()).columns)}")
            results['results']['multi_horizon_profit_labeler'] = mh_result.artifacts
            self._current_pipeline_state.update(mh_result.artifacts)

            # Step 2: Feature Lookback Optimization
            tprint("⚙️ Step 2: Feature Lookback Optimization")
            self.logger.info('⚙️ Step 2: Feature Lookback Optimization')
            flo_result = await self._execute_feature_lookback_optimization(config)
            if not flo_result.success:
                tprint(f"❌ Feature lookback optimization failed: {flo_result.error_message}")
                self.logger.error(f'❌ Feature lookback optimization failed: {flo_result.error_message}')
                return results

            tprint(f"✅ Feature lookback optimization completed for {config.symbol}")
            tprint(f"   → Features optimized: {len(flo_result.artifacts.get('feature_lookback_optimization_result', {}).get('optimized_features', {}))}")
            results['results']['feature_lookback_optimization'] = flo_result.artifacts
            self._current_pipeline_state.update(flo_result.artifacts)

            # Step 3: Interactive Feature Generation
            tprint("🔧 Step 3: Interactive Feature Generation")
            self.logger.info('🔧 Step 3: Interactive Feature Generation')
            interactive_result = await self._execute_interactive_feature_generation(config)
            if not interactive_result.success:
                tprint(f"❌ Interactive feature generation failed: {interactive_result.error_message}")
                self.logger.error(f'❌ Interactive feature generation failed: {interactive_result.error_message}')
                return results

            tprint(f"✅ Interactive feature generation completed for {config.symbol}")
            tprint(f"   → Features generated: {len(interactive_result.artifacts.get('interactive_feature_generation_result', {}).get('features', {}))}")
            results['results']['interactive_feature_generation'] = interactive_result.artifacts
            self._current_pipeline_state.update(interactive_result.artifacts)

            # Step 4: Final Feature Selection
            tprint("🎯 Step 4: Final Feature Selection")
            self.logger.info('🎯 Step 4: Final Feature Selection')
            ffs_result = await self._execute_final_feature_selection(config)
            if not ffs_result.success:
                tprint(f"❌ Final feature selection failed: {ffs_result.error_message}")
                self.logger.error(f'❌ Final feature selection failed: {ffs_result.error_message}')
                return results

            tprint(f"✅ Final feature selection completed for {config.symbol}")
            tprint(f"   → Final features: {len(ffs_result.artifacts.get('final_feature_selection_result', {}).get('selected_features', []))}")
            results['results']['final_feature_selection'] = ffs_result.artifacts
            self._current_pipeline_state.update(ffs_result.artifacts)

            # Success
            end_time = datetime.now()
            results['success'] = True
            results['execution_time'] = (end_time - start_time).total_seconds()
            results['completed_steps'] = 4

            tprint(f"🎉 Pre-Training Sub-Pipeline execution completed successfully for {config.symbol}")
            tprint(f"⏱️ Total execution time: {results['execution_time']:.2f} seconds")
            tprint(f"📊 All {results['completed_steps']} steps completed successfully")
            tprint(f"📋 Pipeline summary:")
            tprint(f"   🎯 Multi-horizon labeling: ✅ Complete")
            tprint(f"   ⚙️ Feature optimization: ✅ Complete")
            tprint(f"   🔧 Roadmap features: ✅ Complete")
            tprint(f"   🎯 Final selection: ✅ Complete")

            self.logger.info(f'🎉 Pre-Training Sub-Pipeline completed successfully in {results["execution_time"]:.2f}s')

        except Exception as e:
            self.logger.error(f'❌ Pre-Training Sub-Pipeline failed with exception: {e}')
            results['error_message'] = str(e)

        return results

    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the pre-training pipeline with backward compatible interface.

        Args:
            training_input: Input data for the pipeline
            pipeline_state: Current pipeline state

        Returns:
            Dictionary containing execution results
        """
        # Extract configuration from pipeline state
        config = SubPipelineConfig(
            symbol=pipeline_state.get('symbol', 'ETHUSDT'),
            exchange=pipeline_state.get('exchange', 'binance'),
            timeframe=pipeline_state.get('timeframe', '1h'),  # Default 1h for pre-training (analyst)
            data_dir=pipeline_state.get('data_dir', 'historical_data'),
            mode=ExecutionMode.FULL,  # Default to full mode
            custom_params=pipeline_state.get('custom_params', {})
        )

        # Execute the pipeline
        return await self.execute_pipeline(config)

    def _prepare_component_pipeline_state(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Construct the pipeline state passed to individual components."""
        pipeline_state: Dict[str, Any] = {
            'symbol': config.symbol,
            'exchange': config.exchange,
            'timeframe': config.timeframe,
            'data_dir': config.data_dir,
            'custom_params': config.custom_params,
        }

        regime_split = config.custom_params.get('regime_data_splitting_result')
        if regime_split is None:
            regime_split = self._current_pipeline_state.get('regime_data_splitting_result')

        if regime_split is not None:
            pipeline_state['regime_data_splitting_result'] = regime_split
            self._current_pipeline_state['regime_data_splitting_result'] = regime_split

        return pipeline_state

    async def _execute_multi_horizon_profit_labeler(self, config: SubPipelineConfig) -> SubPipelineResult:
        """Execute multi-horizon profit labeler with timeframe support."""
        result = SubPipelineResult(
            sub_pipeline_name='multi_horizon_profit_labeler',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )

        try:
            custom_params = config.custom_params or {}
            precomputed_result = custom_params.get('precomputed_labeling_result')

            if precomputed_result:
                tprint('📥 Using precomputed entry labeling result for tactician pipeline')
                result.status = SubPipelineStatus.COMPLETED
                result.success = True
                result.end_time = datetime.now()
                result.duration_seconds = (result.end_time - result.start_time).total_seconds()
                result.artifacts = precomputed_result
                result.metadata = {
                    'component_type': 'multi_horizon_profit_labeler',
                    'source': 'precomputed',
                    'labeling_method': precomputed_result.get('multi_horizon_labeling_result', {}).get('method', 'tactician_entry_labeling')
                }
                return result

            # Convert config to component config
            component_config = ComponentConfig(
                symbol=config.symbol,
                exchange=config.exchange,
                timeframe=config.timeframe,
                data_dir=config.data_dir,
                custom_params=config.custom_params
            )

            # Create component using factory
            component = ComponentFactory.create_component('multi_horizon_profit_labeler', component_config)

            # Execute component
            pipeline_state = self._prepare_component_pipeline_state(config)
            component_result = await component.execute(None, pipeline_state)

            result.status = SubPipelineStatus.COMPLETED if component_result.success else SubPipelineStatus.FAILED
            result.success = component_result.success
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            result.artifacts = component_result.artifacts
            result.error_message = component_result.error_message

        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()

        return result

    async def _execute_feature_lookback_optimization(self, config: SubPipelineConfig) -> SubPipelineResult:
        """Execute feature lookback optimization with timeframe support."""
        result = SubPipelineResult(
            sub_pipeline_name='feature_lookback_optimization',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )

        try:
            # Convert config to component config
            component_config = ComponentConfig(
                symbol=config.symbol,
                exchange=config.exchange,
                timeframe=config.timeframe,
                data_dir=config.data_dir,
                custom_params=config.custom_params
            )

            # Create component using factory
            component = ComponentFactory.create_component('feature_lookback_optimization', component_config)

            # Execute component
            pipeline_state = self._prepare_component_pipeline_state(config)
            component_result = await component.execute(None, pipeline_state)

            result.status = SubPipelineStatus.COMPLETED if component_result.success else SubPipelineStatus.FAILED
            result.success = component_result.success
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            result.artifacts = component_result.artifacts
            result.error_message = component_result.error_message

        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()

        return result

    async def _execute_interactive_feature_generation(self, config: SubPipelineConfig) -> SubPipelineResult:
        """Execute interactive feature generation with timeframe support."""
        result = SubPipelineResult(
            sub_pipeline_name='interactive_feature_generation',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )

        try:
            # Import the new interactive feature generation component
            from .interaction_feature_generator.feature_interaction_generation.interactive_feature_generation_component import (
                create_interactive_feature_generation_component, InteractiveFeatureGenerationConfig
            )
            
            tprint("🔧 Using optimized interactive feature generation component")
            
            # Create component configuration
            component_config = InteractiveFeatureGenerationConfig(
                symbol=config.symbol,
                exchange=config.exchange,
                timeframe=config.timeframe,
                data_dir=config.data_dir,
                feature_budget_pre=config.custom_params.get('feature_budget_pre', 120),
                feature_budget_post=config.custom_params.get('feature_budget_post', (30, 60)),
                interactions_cap=config.custom_params.get('interactions_cap', 15),
                enable_matrix_optimization=config.custom_params.get('enable_matrix_optimization', True),
                enable_hardware_optimization=config.custom_params.get('enable_hardware_optimization', True),
                enable_parallel_processing=config.parallel_processing,
                max_workers=config.max_workers,
                verbose_logging=config.custom_params.get('verbose_logging', True)
            )

            # Create component
            component = create_interactive_feature_generation_component(component_config)

            # Execute component
            pipeline_state = self._prepare_component_pipeline_state(config)
            component_result = await component.execute(None, pipeline_state)

            result.status = SubPipelineStatus.COMPLETED if component_result.success else SubPipelineStatus.FAILED
            result.success = component_result.success
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            result.artifacts = component_result.artifacts
            result.output_files = component_result.output_files
            result.metadata = component_result.metadata
            result.error_message = component_result.error_message

        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            tprint_error(f"❌ Roadmap feature generation failed: {e}")

        return result

    async def _execute_optimized_lookback_generation(self, config: SubPipelineConfig) -> SubPipelineResult:
        """Execute optimized lookback generation with matrix operations and hardware acceleration."""
        result = SubPipelineResult(
            sub_pipeline_name='optimized_lookback_generation',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )

        try:
            # Convert config to component config
            component_config = ComponentConfig(
                symbol=config.symbol,
                exchange=config.exchange,
                timeframe=config.timeframe,
                data_dir=config.data_dir,
                custom_params=config.custom_params
            )

            # Create component using factory
            component = ComponentFactory.create_component('optimized_lookback_generation', component_config)

            # Execute component
            pipeline_state = self._prepare_component_pipeline_state(config)
            component_result = await component.execute(None, pipeline_state)

            result.status = SubPipelineStatus.COMPLETED if component_result.success else SubPipelineStatus.FAILED
            result.success = component_result.success
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            result.artifacts = component_result.artifacts
            result.error_message = component_result.error_message

        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()

        return result

    async def _execute_final_feature_selection(self, config: SubPipelineConfig) -> SubPipelineResult:
        """Execute final feature selection with timeframe support."""
        result = SubPipelineResult(
            sub_pipeline_name='final_feature_selection',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )

        try:
            # Convert config to component config
            component_config = ComponentConfig(
                symbol=config.symbol,
                exchange=config.exchange,
                timeframe=config.timeframe,
                data_dir=config.data_dir,
                custom_params=config.custom_params
            )

            # Create component using factory
            component = ComponentFactory.create_component('final_feature_selection', component_config)

            # Execute component
            pipeline_state = self._prepare_component_pipeline_state(config)
            component_result = await component.execute(None, pipeline_state)

            result.status = SubPipelineStatus.COMPLETED if component_result.success else SubPipelineStatus.FAILED
            result.success = component_result.success
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            result.artifacts = component_result.artifacts
            result.error_message = component_result.error_message

        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()

        return result

    def get_available_sub_pipelines(self) -> List[str]:
        """Get list of available sub-pipelines for pre-training stage."""
        return [
            'multi_horizon_profit_labeler',
            'feature_lookback_optimization', 
            'pid_based_feature_generation',
            'optimized_lookback_generation',
            'final_feature_selection'
        ]

    async def execute_sub_pipeline(self, sub_pipeline_name: str, config: SubPipelineConfig) -> SubPipelineResult:
        """Execute a specific sub-pipeline."""
        if sub_pipeline_name == 'multi_horizon_profit_labeler':
            return await self._execute_multi_horizon_profit_labeler(config)
        elif sub_pipeline_name == 'feature_lookback_optimization':
            return await self._execute_feature_lookback_optimization(config)
        elif sub_pipeline_name == 'pid_based_feature_generation':
            return await self._execute_pid_based_feature_generation(config)
        elif sub_pipeline_name == 'optimized_lookback_generation':
            return await self._execute_optimized_lookback_generation(config)
        elif sub_pipeline_name == 'interactive_feature_generation':
            return await self._execute_interactive_feature_generation(config)
        elif sub_pipeline_name == 'final_feature_selection':
            return await self._execute_final_feature_selection(config)
        else:
            raise ValueError(f"Unknown sub-pipeline: {sub_pipeline_name}")

    async def execute_sub_pipeline_with_next(self, sub_pipeline_name: str, config: SubPipelineConfig) -> SubPipelineResult:
        """Execute a specific sub-pipeline and automatically trigger subsequent sub-pipelines."""
        # For pre-training, we execute all 4 steps in sequence
        available_steps = self.get_available_sub_pipelines()
        
        try:
            start_index = available_steps.index(sub_pipeline_name)
        except ValueError:
            raise ValueError(f"Unknown sub-pipeline: {sub_pipeline_name}")
        
        # Execute all steps starting from the specified one
        for i in range(start_index, len(available_steps)):
            step_name = available_steps[i]
            self.logger.info(f"🚀 Executing pre-training step: {step_name}")
            
            result = await self.execute_sub_pipeline(step_name, config)
            self.results.append(result)
            
            # If this step failed, stop the sequence
            if not result.success:
                self.logger.error(f"❌ Step {step_name} failed, stopping execution sequence")
                break
        
        # Return the first result (the one that was requested)
        return self.results[0] if self.results else None

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary with all results."""
        return {
            'total_sub_pipelines': len(self.results),
            'successful_sub_pipelines': len([r for r in self.results if r.success]),
            'failed_sub_pipelines': len([r for r in self.results if not r.success]),
            'total_execution_time': sum(r.duration_seconds for r in self.results),
            'sub_pipeline_results': [
                {
                    'name': r.sub_pipeline_name,
                    'status': r.status.value,
                    'success': r.success,
                    'execution_time': r.duration_seconds,
                    'error_message': r.error_message
                }
                for r in self.results
            ]
        }

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