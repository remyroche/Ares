#!/usr/bin/env python3
"""
Ares Launcher - Granular Sub-Pipeline Control

This launcher provides granular control over training pipeline execution,
allowing users to execute specific sub-pipelines at different stages with
full, light, or blank execution modes.

Key Features:
- Granular sub-pipeline control
- Multiple execution modes (full, light, blank)
- Stage-specific execution
- Sub-pipeline-specific execution
- Comprehensive monitoring and reporting
- Mid-function artifact creation
- Real-time progress tracking
"""

import asyncio
import json
import logging
import os
import sys
import argparse
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path
from enum import Enum

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from src.training.steps.main_training_pipeline import (
    MainTrainingPipeline, MainPipelineConfig, MainPipelineResult,
    PipelineStage, ExecutionMode, get_full_pipeline_config, 
    get_light_pipeline_config, get_blank_pipeline_config
)

logger = system_logger.getChild('AresLauncher')

class LauncherMode(Enum):
    """Launcher execution modes."""
    FULL = "full"          # Complete pipeline execution
    LIGHT = "light"        # Lightweight execution
    BLANK = "blank"        # Minimal execution for testing
    STAGE = "stage"        # Execute specific stage
    SUB_PIPELINE = "sub_pipeline"  # Execute specific sub-pipeline

class ExecutionModeType(Enum):
    """Execution mode types for stage/sub-pipeline specific execution."""
    FULL = "full"          # Complete execution with all features
    LIGHT = "light"        # Lightweight execution with essential features only
    BLANK = "blank"        # Minimal execution for testing/validation

class AresLauncher:
    """
    Ares Launcher with Granular Sub-Pipeline Control.
    
    Provides comprehensive control over training pipeline execution with
    granular sub-pipeline management and real-time monitoring.
    """
    
    def __init__(self):
        """Initialize the Ares launcher."""
        self.logger = logger.getChild('AresLauncher')
        self.pipeline = MainTrainingPipeline()
        self.current_execution: Optional[MainPipelineResult] = None
        self.execution_history: List[MainPipelineResult] = []
        
        # Initialize monitoring
        self._setup_logging()
        self._setup_monitoring()
    
    def _setup_logging(self):
        """Setup comprehensive logging."""
        self.logger.info("🚀 Ares Launcher Initialized")
        self.logger.info("=" * 80)
        self.logger.info("🎯 Granular Sub-Pipeline Control Enabled")
        self.logger.info("=" * 80)
    
    def _setup_monitoring(self):
        """Setup monitoring and progress tracking."""
        self.monitoring_enabled = True
        self.progress_callbacks: List[callable] = []
        
        # Register default progress callback
        self.register_progress_callback(self._default_progress_callback)
    
    def register_progress_callback(self, callback: callable):
        """Register a progress callback function."""
        self.progress_callbacks.append(callback)
    
    def _default_progress_callback(self, progress_data: Dict[str, Any]):
        """Default progress callback for monitoring."""
        self.logger.info(f"📊 Progress: {progress_data.get('message', 'Unknown')}")
    
    def _log_stage_transition(self, from_stage: Optional[str], to_stage: str, transition_type: str = "STAGE"):
        """Log explicit stage/pipeline transitions."""
        if from_stage:
            self.logger.info("=" * 80)
            self.logger.info(f"🔄 TRANSITION: {from_stage} → {to_stage}")
            self.logger.info(f"📋 Transition Type: {transition_type}")
            self.logger.info(f"⏰ Timestamp: {datetime.now().isoformat()}")
            self.logger.info("=" * 80)
        else:
            self.logger.info("=" * 80)
            self.logger.info(f"🚀 STARTING: {to_stage}")
            self.logger.info(f"📋 Execution Type: {transition_type}")
            self.logger.info(f"⏰ Timestamp: {datetime.now().isoformat()}")
            self.logger.info("=" * 80)
    
    def _log_sub_pipeline_transition(self, from_sub_pipeline: Optional[str], to_sub_pipeline: str, stage: str):
        """Log explicit sub-pipeline transitions."""
        if from_sub_pipeline:
            self.logger.info("=" * 60)
            self.logger.info(f"🔄 SUB-PIPELINE TRANSITION: {from_sub_pipeline} → {to_sub_pipeline}")
            self.logger.info(f"📋 Stage: {stage}")
            self.logger.info(f"⏰ Timestamp: {datetime.now().isoformat()}")
            self.logger.info("=" * 60)
        else:
            self.logger.info("=" * 60)
            self.logger.info(f"🚀 STARTING SUB-PIPELINE: {to_sub_pipeline}")
            self.logger.info(f"📋 Stage: {stage}")
            self.logger.info(f"⏰ Timestamp: {datetime.now().isoformat()}")
            self.logger.info("=" * 60)
    
    async def _create_outcome_file(self, stage: str, sub_pipeline: str, result: Any, config: MainPipelineConfig) -> str:
        """Create outcome file for stage/sub-pipeline completion."""
        outcome_dir = Path("outcomes")
        outcome_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{stage}_{sub_pipeline}_outcome_{timestamp}.json"
        outcome_file = outcome_dir / filename
        
        outcome_data = {
            'stage': stage,
            'sub_pipeline': sub_pipeline,
            'timestamp': datetime.now().isoformat(),
            'status': result.status.value if hasattr(result, 'status') else 'completed',
            'output_files': result.output_files if hasattr(result, 'output_files') else [],
            'metadata': result.metadata if hasattr(result, 'metadata') else {},
            'artifacts': result.artifacts if hasattr(result, 'artifacts') else {},
            'config': {
                'symbol': config.symbol,
                'exchange': config.exchange,
                'timeframe': config.timeframe,
                'mode': config.mode.value,
                'intensity_percentage': config.intensity_percentage,
                'training_mode_config': config.training_mode_config
            },
            'next_stage_requirements': self._get_next_stage_requirements(stage, sub_pipeline)
        }
        
        with open(outcome_file, 'w') as f:
            json.dump(outcome_data, f, indent=2, default=str)
        
        self.logger.info(f"💾 Outcome file created: {outcome_file}")
        return str(outcome_file)
    
    def _get_next_stage_requirements(self, current_stage: str, current_sub_pipeline: str) -> Dict[str, Any]:
        """Get requirements for the next stage/sub-pipeline."""
        requirements = {
            'required_files': [],
            'required_artifacts': [],
            'data_dependencies': []
        }
        
        # Define stage dependencies and requirements
        stage_requirements = {
            'data_collection': {
                'next_stage': 'market_analysis',
                'required_files': ['processed_data.parquet', 'data_quality_report.json', 'exported_data.parquet'],
                'required_artifacts': ['data_metadata', 'quality_metrics', 'integration_results'],
                'sub_pipelines': ['data_download', 'data_conversion', 'data_validation', 'data_preparation', 
                                'feature_engineering', 'data_quality_check', 'data_storage', 'data_monitoring',
                                'data_integration', 'data_export']
            },
            'market_analysis': {
                'next_stage': 'model_training',
                'required_files': ['sr_levels.json', 'regime_assignments.parquet', 'labels.parquet', 'features.parquet'],
                'required_artifacts': ['sr_clusters', 'regime_model', 'feature_metadata', 'cross_timeframe_features'],
                'sub_pipelines': ['sr_detection', 'sr_clustering', 'sr_ml_learning', 'hmm_clustering',
                                'hmm_regime_discovery', 'regime_data_splitting', 'triple_barrier_labeling',
                                'feature_lookback_optimization', 'fractional_differentiation', 'cross_timeframe_analysis']
            },
            'model_training': {
                'next_stage': 'backtesting',
                'required_files': ['trained_models.pkl', 'validation_results.json', 'evaluation_results.json'],
                'required_artifacts': ['model_metadata', 'performance_metrics', 'ensemble_models'],
                'sub_pipelines': ['general_model_training', 'analyst_model_training', 'tactician_model_training',
                                'hmm_training', 'ensemble_training', 'multi_timeframe_training',
                                'regime_specific_training', 'model_validation', 'model_persistence', 'model_evaluation']
            },
            'backtesting': {
                'next_stage': 'reporting',
                'required_files': ['backtest_results.json', 'performance_report.json', 'final_report.pdf'],
                'required_artifacts': ['trade_analysis', 'risk_metrics', 'portfolio_analysis'],
                'sub_pipelines': ['walk_forward_validation', 'monte_carlo_simulation', 'ab_testing',
                                'model_persistence', 'final_parameters_optimization', 'performance_analytics',
                                'risk_analysis', 'trade_analysis', 'portfolio_analysis', 'reporting']
            }
        }
        
        if current_stage in stage_requirements:
            requirements.update(stage_requirements[current_stage])
        
        return requirements
    
    async def _check_outcome_files(self, stage: str, sub_pipeline: str) -> Optional[Dict[str, Any]]:
        """Check for existing outcome files from previous stages."""
        outcome_dir = Path("outcomes")
        if not outcome_dir.exists():
            return None
        
        # Look for the most recent outcome file for this stage/sub-pipeline
        pattern = f"{stage}_{sub_pipeline}_outcome_*.json"
        outcome_files = list(outcome_dir.glob(pattern))
        
        if not outcome_files:
            return None
        
        # Get the most recent file
        latest_file = max(outcome_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(latest_file, 'r') as f:
                outcome_data = json.load(f)
            
            self.logger.info(f"📂 Found existing outcome file: {latest_file}")
            return outcome_data
        except Exception as e:
            self.logger.warning(f"⚠️ Could not read outcome file {latest_file}: {e}")
            return None
    
    async def execute_pipeline(
        self,
        mode: LauncherMode = LauncherMode.FULL,
        symbol: str = "ETHUSDT",
        exchange: str = "binance",
        timeframe: str = "1m",
        data_dir: str = "data/training",
        stage: Optional[PipelineStage] = None,
        sub_pipeline: Optional[str] = None,
        execution_mode: ExecutionModeType = ExecutionModeType.FULL,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> MainPipelineResult:
        """
        Execute the training pipeline with granular control.
        
        Args:
            mode: Launcher execution mode (full, light, blank, stage, sub_pipeline)
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe
            data_dir: Data directory
            stage: Specific stage to execute (for STAGE mode)
            sub_pipeline: Specific sub-pipeline to execute (for SUB_PIPELINE mode)
            execution_mode: Execution mode type (full, light, blank) for stage/sub-pipeline specific execution
            custom_config: Custom configuration parameters
            
        Returns:
            MainPipelineResult with execution details
        """
        self.logger.info(f"🚀 Starting pipeline execution: {mode.value}")
        
        # Create configuration based on mode
        config = self._create_config(
            mode, symbol, exchange, timeframe, data_dir, 
            stage, sub_pipeline, execution_mode, custom_config
        )
        
        # Execute based on mode
        if mode == LauncherMode.SUB_PIPELINE and sub_pipeline:
            return await self._execute_sub_pipeline(sub_pipeline, config)
        elif mode == LauncherMode.STAGE and stage:
            return await self._execute_stage(stage, config)
        else:
            return await self._execute_full_pipeline(config)
    
    def _create_config(
        self,
        mode: LauncherMode,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        stage: Optional[PipelineStage],
        sub_pipeline: Optional[str],
        execution_mode: ExecutionModeType,
        custom_config: Optional[Dict[str, Any]]
    ) -> MainPipelineConfig:
        """Create pipeline configuration based on mode and parameters."""
        
        # Base configuration
        base_config = {
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': timeframe,
            'data_dir': data_dir,
            'custom_params': custom_config or {}
        }
        
        # Mode-specific configuration
        if mode == LauncherMode.FULL:
            config = get_full_pipeline_config(**base_config)
        elif mode == LauncherMode.LIGHT:
            config = get_light_pipeline_config(**base_config)
        elif mode == LauncherMode.BLANK:
            config = get_blank_pipeline_config(**base_config)
        elif mode == LauncherMode.STAGE and stage:
            config = self._create_stage_config(stage, base_config, execution_mode)
        elif mode == LauncherMode.SUB_PIPELINE and sub_pipeline:
            config = self._create_sub_pipeline_config(sub_pipeline, base_config, execution_mode)
        else:
            # Default to full configuration
            config = get_full_pipeline_config(**base_config)
        
        return config
    
    def _create_stage_config(self, stage: PipelineStage, base_config: Dict[str, Any], execution_mode: ExecutionModeType) -> MainPipelineConfig:
        """Create configuration for a specific stage."""
        # Set the execution mode in base config
        base_config['mode'] = ExecutionMode(execution_mode.value)
        
        # Get configuration based on execution mode
        if execution_mode == ExecutionModeType.FULL:
            config = get_full_pipeline_config(**base_config)
        elif execution_mode == ExecutionModeType.LIGHT:
            config = get_light_pipeline_config(**base_config)
        elif execution_mode == ExecutionModeType.BLANK:
            config = get_blank_pipeline_config(**base_config)
        else:
            config = get_full_pipeline_config(**base_config)
        
        # Enable only the specified stage
        config.enabled_stages = [stage]
        
        # Get all available sub-pipelines for the stage
        available_sub_pipelines = self.pipeline.get_available_sub_pipelines(stage)
        config.enabled_sub_pipelines[stage] = available_sub_pipelines
        
        # Add intensity parameters to stage configuration
        if config.training_mode_config:
            config.stage_params[stage] = {
                'intensity_percentage': config.intensity_percentage,
                'training_mode_config': config.training_mode_config,
                'model_training': config.training_mode_config.get('model_training', {}),
                'validation': config.training_mode_config.get('validation', {}),
                'optimization': config.training_mode_config.get('optimization', {})
            }
        
        return config
    
    def _create_sub_pipeline_config(self, sub_pipeline: str, base_config: Dict[str, Any], execution_mode: ExecutionModeType) -> MainPipelineConfig:
        """Create configuration for a specific sub-pipeline."""
        # Set the execution mode in base config
        base_config['mode'] = ExecutionMode(execution_mode.value)
        
        # Get configuration based on execution mode
        if execution_mode == ExecutionModeType.FULL:
            config = get_full_pipeline_config(**base_config)
        elif execution_mode == ExecutionModeType.LIGHT:
            config = get_light_pipeline_config(**base_config)
        elif execution_mode == ExecutionModeType.BLANK:
            config = get_blank_pipeline_config(**base_config)
        else:
            config = get_full_pipeline_config(**base_config)
        
        # Find which stage contains the sub-pipeline
        target_stage = None
        for stage in PipelineStage:
            available_sub_pipelines = self.pipeline.get_available_sub_pipelines(stage)
            if sub_pipeline in available_sub_pipelines:
                target_stage = stage
                break
        
        if not target_stage:
            raise ValueError(f"Sub-pipeline '{sub_pipeline}' not found in any stage")
        
        # Enable only the target stage and sub-pipeline
        config.enabled_stages = [target_stage]
        config.enabled_sub_pipelines[target_stage] = [sub_pipeline]
        
        # Add intensity parameters to stage configuration
        if config.training_mode_config:
            config.stage_params[target_stage] = {
                'intensity_percentage': config.intensity_percentage,
                'training_mode_config': config.training_mode_config,
                'model_training': config.training_mode_config.get('model_training', {}),
                'validation': config.training_mode_config.get('validation', {}),
                'optimization': config.training_mode_config.get('optimization', {})
            }
        
        return config
    
    async def _execute_full_pipeline(self, config: MainPipelineConfig) -> MainPipelineResult:
        """Execute the full pipeline."""
        # Log full pipeline start
        self._log_stage_transition(None, "FULL_PIPELINE", "FULL_PIPELINE_EXECUTION")
        
        # Create mid-function artifacts
        artifacts = await self._create_mid_function_artifacts(config)
        
        # Execute pipeline with stage-by-stage transition logging
        result = await self._execute_pipeline_with_transitions(config)
        
        # Store execution
        self.current_execution = result
        self.execution_history.append(result)
        
        # Log results
        self._log_execution_results(result)
        
        return result
    
    async def _execute_pipeline_with_transitions(self, config: MainPipelineConfig) -> MainPipelineResult:
        """Execute pipeline with explicit stage transitions."""
        pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        result = MainPipelineResult(
            pipeline_id=pipeline_id,
            status=SubPipelineStatus.RUNNING,
            start_time=start_time
        )
        
        try:
            previous_stage = None
            
            # Execute each enabled stage with transitions
            for stage in config.enabled_stages:
                # Log stage transition
                if previous_stage:
                    self._log_stage_transition(previous_stage.value, stage.value, "STAGE_TRANSITION")
                else:
                    self._log_stage_transition(None, stage.value, "STAGE_START")
                
                # Check for existing outcome files
                outcome_data = await self._check_outcome_files(stage.value, "stage")
                if outcome_data:
                    self.logger.info(f"📂 Resuming from previous outcome: {outcome_data['timestamp']}")
                
                # Execute stage
                stage_result = await self.pipeline._execute_stage(stage, config)
                result.stage_results[stage] = stage_result
                
                # Create outcome files for each sub-pipeline in the stage
                for sub_result in stage_result:
                    if hasattr(sub_result, 'sub_pipeline_name'):
                        await self._create_outcome_file(stage.value, sub_result.sub_pipeline_name, sub_result, config)
                
                # Check if stage failed
                failed_sub_pipelines = [r for r in stage_result if r.status == SubPipelineStatus.FAILED]
                if failed_sub_pipelines and config.mode != ExecutionMode.BLANK:
                    self.logger.warning(f"⚠️ Stage {stage.value} had {len(failed_sub_pipelines)} failed sub-pipelines")
                    result.failed_stages.append(stage)
                
                previous_stage = stage
            
            # Calculate overall metrics
            self.pipeline._calculate_pipeline_metrics(result)
            
            # Update result status
            end_time = datetime.now()
            result.end_time = end_time
            result.duration_seconds = (end_time - start_time).total_seconds()
            
            if result.failed_sub_pipelines == 0:
                result.status = SubPipelineStatus.COMPLETED
                self.logger.info(f"✅ Full pipeline {pipeline_id} completed successfully in {result.duration_seconds:.2f}s")
            else:
                result.status = SubPipelineStatus.FAILED
                result.error_message = f"Pipeline failed with {result.failed_sub_pipelines} failed sub-pipelines"
                self.logger.error(f"❌ Full pipeline {pipeline_id} failed: {result.error_message}")
            
        except Exception as e:
            end_time = datetime.now()
            result.status = SubPipelineStatus.FAILED
            result.end_time = end_time
            result.duration_seconds = (end_time - start_time).total_seconds()
            result.error_message = str(e)
            
            self.logger.error(f"❌ Full pipeline {pipeline_id} failed with exception: {e}")
        
        return result
    
    async def _execute_stage(self, stage: PipelineStage, config: MainPipelineConfig) -> MainPipelineResult:
        """Execute a specific stage."""
        # Log stage transition
        self._log_stage_transition(None, stage.value, "STAGE_EXECUTION")
        
        # Check for existing outcome files
        outcome_data = await self._check_outcome_files(stage.value, "stage")
        if outcome_data:
            self.logger.info(f"📂 Resuming from previous outcome: {outcome_data['timestamp']}")
        
        # Create mid-function artifacts for the stage
        artifacts = await self._create_stage_artifacts(stage, config)
        
        # Execute only the specified stage
        result = await self.pipeline.execute_pipeline(config)
        
        # Create outcome file for this stage
        if result.stage_results and stage in result.stage_results:
            stage_results = result.stage_results[stage]
            for sub_result in stage_results:
                if hasattr(sub_result, 'sub_pipeline_name'):
                    await self._create_outcome_file(stage.value, sub_result.sub_pipeline_name, sub_result, config)
        
        # Store execution
        self.current_execution = result
        self.execution_history.append(result)
        
        # Log results
        self._log_execution_results(result)
        
        return result
    
    async def _execute_sub_pipeline(self, sub_pipeline: str, config: MainPipelineConfig) -> MainPipelineResult:
        """Execute a specific sub-pipeline."""
        # Find the stage containing this sub-pipeline
        target_stage = None
        for stage in PipelineStage:
            available_sub_pipelines = self.pipeline.get_available_sub_pipelines(stage)
            if sub_pipeline in available_sub_pipelines:
                target_stage = stage
                break
        
        if not target_stage:
            raise ValueError(f"Sub-pipeline '{sub_pipeline}' not found in any stage")
        
        # Log sub-pipeline transition
        self._log_sub_pipeline_transition(None, sub_pipeline, target_stage.value)
        
        # Check for existing outcome files
        outcome_data = await self._check_outcome_files(target_stage.value, sub_pipeline)
        if outcome_data:
            self.logger.info(f"📂 Resuming from previous outcome: {outcome_data['timestamp']}")
        
        # Create mid-function artifacts for the sub-pipeline
        artifacts = await self._create_sub_pipeline_artifacts(sub_pipeline, config)
        
        # Execute only the specified sub-pipeline
        result = await self.pipeline.execute_pipeline(config)
        
        # Create outcome file for this sub-pipeline
        if result.stage_results and target_stage in result.stage_results:
            stage_results = result.stage_results[target_stage]
            for sub_result in stage_results:
                if hasattr(sub_result, 'sub_pipeline_name') and sub_result.sub_pipeline_name == sub_pipeline:
                    await self._create_outcome_file(target_stage.value, sub_pipeline, sub_result, config)
                    break
        
        # Store execution
        self.current_execution = result
        self.execution_history.append(result)
        
        # Log results
        self._log_execution_results(result)
        
        return result
    
    async def _create_mid_function_artifacts(self, config: MainPipelineConfig) -> Dict[str, Any]:
        """Create mid-function artifacts for full pipeline execution."""
        self.logger.info("🔧 Creating mid-function artifacts for full pipeline")
        
        artifacts = {
            'pipeline_config': {
                'mode': config.mode.value,
                'symbol': config.symbol,
                'exchange': config.exchange,
                'timeframe': config.timeframe,
                'enabled_stages': [stage.value for stage in config.enabled_stages],
                'enabled_sub_pipelines': {
                    stage.value: sub_pipelines 
                    for stage, sub_pipelines in config.enabled_sub_pipelines.items()
                }
            },
            'execution_metadata': {
                'start_time': datetime.now().isoformat(),
                'pipeline_id': f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'execution_mode': 'full_pipeline'
            },
            'monitoring_config': {
                'progress_tracking': True,
                'artifact_creation': True,
                'real_time_monitoring': True
            }
        }
        
        # Save artifacts
        await self._save_artifacts(artifacts, 'full_pipeline_artifacts.json')
        
        return artifacts
    
    async def _create_stage_artifacts(self, stage: PipelineStage, config: MainPipelineConfig) -> Dict[str, Any]:
        """Create mid-function artifacts for stage execution."""
        self.logger.info(f"🔧 Creating mid-function artifacts for stage: {stage.value}")
        
        artifacts = {
            'stage_config': {
                'stage': stage.value,
                'mode': config.mode.value,
                'symbol': config.symbol,
                'exchange': config.exchange,
                'timeframe': config.timeframe,
                'enabled_sub_pipelines': config.enabled_sub_pipelines.get(stage, [])
            },
            'execution_metadata': {
                'start_time': datetime.now().isoformat(),
                'pipeline_id': f"stage_{stage.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'execution_mode': 'stage_execution'
            },
            'stage_info': {
                'available_sub_pipelines': self.pipeline.get_available_sub_pipelines(stage),
                'stage_description': self._get_stage_description(stage)
            }
        }
        
        # Save artifacts
        await self._save_artifacts(artifacts, f'{stage.value}_artifacts.json')
        
        return artifacts
    
    async def _create_sub_pipeline_artifacts(self, sub_pipeline: str, config: MainPipelineConfig) -> Dict[str, Any]:
        """Create mid-function artifacts for sub-pipeline execution."""
        self.logger.info(f"🔧 Creating mid-function artifacts for sub-pipeline: {sub_pipeline}")
        
        # Find the stage containing this sub-pipeline
        target_stage = None
        for stage in PipelineStage:
            if sub_pipeline in config.enabled_sub_pipelines.get(stage, []):
                target_stage = stage
                break
        
        artifacts = {
            'sub_pipeline_config': {
                'sub_pipeline': sub_pipeline,
                'stage': target_stage.value if target_stage else 'unknown',
                'mode': config.mode.value,
                'symbol': config.symbol,
                'exchange': config.exchange,
                'timeframe': config.timeframe
            },
            'execution_metadata': {
                'start_time': datetime.now().isoformat(),
                'pipeline_id': f"sub_pipeline_{sub_pipeline}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'execution_mode': 'sub_pipeline_execution'
            },
            'sub_pipeline_info': {
                'description': self._get_sub_pipeline_description(sub_pipeline),
                'dependencies': self._get_sub_pipeline_dependencies(sub_pipeline),
                'outputs': self._get_sub_pipeline_outputs(sub_pipeline)
            }
        }
        
        # Save artifacts
        await self._save_artifacts(artifacts, f'{sub_pipeline}_artifacts.json')
        
        return artifacts
    
    async def _save_artifacts(self, artifacts: Dict[str, Any], filename: str):
        """Save artifacts to file."""
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        
        artifacts_file = artifacts_dir / filename
        with open(artifacts_file, 'w') as f:
            json.dump(artifacts, f, indent=2, default=str)
        
        self.logger.info(f"💾 Artifacts saved to: {artifacts_file}")
    
    def _get_stage_description(self, stage: PipelineStage) -> str:
        """Get description for a pipeline stage."""
        descriptions = {
            PipelineStage.DATA_COLLECTION: "Data collection and preparation stage",
            PipelineStage.MARKET_ANALYSIS: "Market analysis and regime detection stage",
            PipelineStage.MODEL_TRAINING: "Model training and validation stage",
            PipelineStage.BACKTESTING: "Backtesting and optimization stage"
        }
        return descriptions.get(stage, "Unknown stage")
    
    def _get_sub_pipeline_description(self, sub_pipeline: str) -> str:
        """Get description for a sub-pipeline."""
        descriptions = {
            # Data Collection (10 sub-pipelines)
            'data_download': "Download raw data from exchanges",
            'data_conversion': "Convert data formats and standardize",
            'data_validation': "Validate data quality and integrity",
            'data_preparation': "Prepare data for further processing",
            'feature_engineering': "Basic feature engineering",
            'data_quality_check': "Comprehensive quality assessment",
            'data_storage': "Store processed data",
            'data_monitoring': "Monitor data collection process",
            'data_integration': "Integrate multiple data sources",
            'data_export': "Export data in various formats",
            
            # Market Analysis (10 sub-pipelines)
            'sr_detection': "Detect Support/Resistance levels",
            'sr_clustering': "Generate SR clusters",
            'sr_ml_learning': "ML-based learning for SR clusters",
            'hmm_clustering': "HMM-based regime clustering",
            'hmm_regime_discovery': "Discover market regimes",
            'regime_data_splitting': "Split data by regimes",
            'triple_barrier_labeling': "Apply triple barrier method",
            'feature_lookback_optimization': "Optimize feature lookback periods",
            'fractional_differentiation': "Apply fractional differentiation",
            'cross_timeframe_analysis': "Cross timeframe interaction features",
            
            # Model Training (10 sub-pipelines)
            'general_model_training': "Train general ML models",
            'analyst_model_training': "Train analyst-specific models",
            'tactician_model_training': "Train tactician-specific models",
            'hmm_training': "HMM-based model training",
            'ensemble_training': "Ensemble model training",
            'multi_timeframe_training': "Multi-timeframe model training",
            'regime_specific_training': "Regime-specific model training",
            'model_validation': "Model validation and testing",
            'model_persistence': "Save and load models",
            'model_evaluation': "Comprehensive model evaluation",
            
            # Backtesting (10 sub-pipelines)
            'walk_forward_validation': "Walk-forward backtesting",
            'monte_carlo_simulation': "Monte Carlo backtesting",
            'ab_testing': "A/B testing for strategies",
            'final_parameters_optimization': "System-wide parameter optimization",
            'performance_analytics': "Performance analysis and reporting",
            'risk_analysis': "Risk metrics and analysis",
            'trade_analysis': "Trade-level analysis",
            'portfolio_analysis': "Portfolio-level analysis",
            'reporting': "Comprehensive reporting"
        }
        return descriptions.get(sub_pipeline, "Unknown sub-pipeline")
    
    def _get_sub_pipeline_dependencies(self, sub_pipeline: str) -> List[str]:
        """Get dependencies for a sub-pipeline."""
        dependencies = {
            # Data Collection dependencies
            'data_conversion': ['data_download'],
            'data_validation': ['data_download', 'data_conversion'],
            'data_preparation': ['data_validation'],
            'feature_engineering': ['data_preparation'],
            'data_quality_check': ['feature_engineering'],
            'data_storage': ['data_quality_check'],
            'data_monitoring': ['data_storage'],
            'data_integration': ['data_monitoring'],
            'data_export': ['data_integration'],
            
            # Market Analysis dependencies
            'sr_clustering': ['sr_detection'],
            'sr_ml_learning': ['sr_clustering'],
            'hmm_clustering': ['sr_ml_learning'],
            'hmm_regime_discovery': ['hmm_clustering'],
            'regime_data_splitting': ['hmm_regime_discovery'],
            'triple_barrier_labeling': ['regime_data_splitting'],
            'feature_lookback_optimization': ['triple_barrier_labeling'],
            'fractional_differentiation': ['feature_lookback_optimization'],
            'cross_timeframe_analysis': ['fractional_differentiation'],
            
            # Model Training dependencies
            'analyst_model_training': ['general_model_training'],
            'tactician_model_training': ['analyst_model_training'],
            'hmm_training': ['tactician_model_training'],
            'ensemble_training': ['hmm_training'],
            'multi_timeframe_training': ['ensemble_training'],
            'regime_specific_training': ['multi_timeframe_training'],
            'model_validation': ['regime_specific_training'],
            'model_persistence': ['model_validation'],
            'model_evaluation': ['model_persistence'],
            
            # Backtesting dependencies
            'monte_carlo_simulation': ['walk_forward_validation'],
            'ab_testing': ['monte_carlo_simulation'],
            'final_parameters_optimization': ['ab_testing'],
            'performance_analytics': ['final_parameters_optimization'],
            'risk_analysis': ['performance_analytics'],
            'trade_analysis': ['risk_analysis'],
            'portfolio_analysis': ['trade_analysis'],
            'reporting': ['portfolio_analysis']
        }
        return dependencies.get(sub_pipeline, [])
    
    def _get_sub_pipeline_outputs(self, sub_pipeline: str) -> List[str]:
        """Get expected outputs for a sub-pipeline."""
        outputs = {
            # Data Collection outputs
            'data_download': ['raw_data.parquet'],
            'data_conversion': ['converted_data.parquet'],
            'data_validation': ['validation_report.json'],
            'data_preparation': ['prepared_data.parquet'],
            'feature_engineering': ['features.parquet'],
            'data_quality_check': ['quality_report.json'],
            'data_storage': ['stored_data.parquet'],
            'data_monitoring': ['monitoring_report.json'],
            'data_integration': ['integrated_data.parquet'],
            'data_export': ['exported_data.parquet'],
            
            # Market Analysis outputs
            'sr_detection': ['sr_levels.json'],
            'sr_clustering': ['sr_clusters.json'],
            'sr_ml_learning': ['sr_ml_model.pkl'],
            'hmm_clustering': ['hmm_clusters.json'],
            'hmm_regime_discovery': ['regime_assignments.parquet'],
            'regime_data_splitting': ['regime_splits.parquet'],
            'triple_barrier_labeling': ['labels.parquet'],
            'feature_lookback_optimization': ['optimized_features.parquet'],
            'fractional_differentiation': ['fractional_features.parquet'],
            'cross_timeframe_analysis': ['cross_tf_features.parquet'],
            
            # Model Training outputs
            'general_model_training': ['general_model.pkl'],
            'analyst_model_training': ['analyst_model.pkl'],
            'tactician_model_training': ['tactician_model.pkl'],
            'hmm_training': ['hmm_model.pkl'],
            'ensemble_training': ['ensemble_model.pkl'],
            'multi_timeframe_training': ['multi_tf_model.pkl'],
            'regime_specific_training': ['regime_models.pkl'],
            'model_validation': ['validation_results.json'],
            'model_persistence': ['persisted_models.pkl'],
            'model_evaluation': ['evaluation_results.json'],
            
            # Backtesting outputs
            'walk_forward_validation': ['backtest_results.json'],
            'monte_carlo_simulation': ['mc_results.json'],
            'ab_testing': ['ab_test_results.json'],
            'final_parameters_optimization': ['optimized_parameters.json'],
            'performance_analytics': ['performance_report.json'],
            'risk_analysis': ['risk_report.json'],
            'trade_analysis': ['trade_analysis.json'],
            'portfolio_analysis': ['portfolio_analysis.json'],
            'reporting': ['comprehensive_report.pdf']
        }
        return outputs.get(sub_pipeline, [])
    
    def _log_execution_results(self, result: MainPipelineResult):
        """Log execution results."""
        self.logger.info("=" * 80)
        self.logger.info("📊 PIPELINE EXECUTION RESULTS")
        self.logger.info("=" * 80)
        self.logger.info(f"Pipeline ID: {result.pipeline_id}")
        self.logger.info(f"Status: {result.status.value}")
        self.logger.info(f"Duration: {result.duration_seconds:.2f}s")
        self.logger.info(f"Total Sub-pipelines: {result.total_sub_pipelines}")
        self.logger.info(f"Completed: {result.completed_sub_pipelines}")
        self.logger.info(f"Failed: {result.failed_sub_pipelines}")
        self.logger.info(f"Success Rate: {result.success_rate:.2%}")
        
        if result.failed_stages:
            self.logger.warning(f"Failed Stages: {[stage.value for stage in result.failed_stages]}")
        
        if result.error_message:
            self.logger.error(f"Error: {result.error_message}")
        
        self.logger.info("=" * 80)
    
    def get_execution_history(self) -> List[MainPipelineResult]:
        """Get execution history."""
        return self.execution_history
    
    def get_current_execution(self) -> Optional[MainPipelineResult]:
        """Get current execution status."""
        return self.current_execution
    
    def get_available_stages(self) -> List[str]:
        """Get list of available pipeline stages."""
        return [stage.value for stage in PipelineStage]
    
    def get_available_sub_pipelines(self, stage: Optional[str] = None) -> Dict[str, List[str]]:
        """Get available sub-pipelines for stages."""
        if stage:
            stage_enum = PipelineStage(stage)
            return {stage: self.pipeline.get_available_sub_pipelines(stage_enum)}
        else:
            return {
                stage.value: self.pipeline.get_available_sub_pipelines(stage)
                for stage in PipelineStage
            }
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all executions."""
        return self.pipeline.get_execution_summary()

# CLI Interface
def create_cli_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Ares Launcher - Granular Sub-Pipeline Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline execution (730 days, 100% intensity)
  python ares_launcher.py --mode full --symbol ETHUSDT --exchange binance

  # Light pipeline execution (10 days, 5% intensity)
  python ares_launcher.py --mode light --symbol ETHUSDT

  # Execute specific stage with full execution mode (730 days, 100% intensity)
  python ares_launcher.py --mode stage --stage data_collection --execution-mode full --symbol ETHUSDT

  # Execute specific stage with light execution mode (10 days, 5% intensity)
  python ares_launcher.py --mode stage --stage market_analysis --execution-mode light --symbol ETHUSDT

  # Execute specific sub-pipeline with blank execution mode (180 days, 10% intensity)
  python ares_launcher.py --mode sub_pipeline --sub_pipeline sr_detection --execution-mode blank --symbol ETHUSDT

  # Execute specific sub-pipeline with full execution mode (730 days, 100% intensity)
  python ares_launcher.py --mode sub_pipeline --sub_pipeline hmm_regime_discovery --execution-mode full --symbol ETHUSDT

  # Blank mode for testing (180 days, 10% intensity)
  python ares_launcher.py --mode blank --symbol ETHUSDT
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['full', 'light', 'blank', 'stage', 'sub_pipeline'],
        default='full',
        help='Launcher execution mode (default: full)'
    )
    
    parser.add_argument(
        '--execution-mode',
        choices=['full', 'light', 'blank'],
        default='full',
        help='Execution mode type for stage/sub-pipeline specific execution (default: full)'
    )
    
    parser.add_argument(
        '--symbol',
        default='ETHUSDT',
        help='Trading symbol (default: ETHUSDT)'
    )
    
    parser.add_argument(
        '--exchange',
        default='binance',
        help='Exchange name (default: binance)'
    )
    
    parser.add_argument(
        '--timeframe',
        default='1m',
        help='Data timeframe (default: 1m)'
    )
    
    parser.add_argument(
        '--data-dir',
        default='data/training',
        help='Data directory (default: data/training)'
    )
    
    parser.add_argument(
        '--stage',
        choices=['data_collection', 'market_analysis', 'model_training', 'backtesting'],
        help='Specific stage to execute (for stage mode)'
    )
    
    parser.add_argument(
        '--sub-pipeline',
        help='Specific sub-pipeline to execute (for sub_pipeline mode)'
    )
    
    parser.add_argument(
        '--config',
        help='Path to custom configuration file (JSON)'
    )
    
    parser.add_argument(
        '--list-stages',
        action='store_true',
        help='List available pipeline stages'
    )
    
    parser.add_argument(
        '--list-sub-pipelines',
        help='List available sub-pipelines for a stage'
    )
    
    return parser

async def main():
    """Main entry point."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Initialize launcher
    launcher = AresLauncher()
    
    # Handle list commands
    if args.list_stages:
        stages = launcher.get_available_stages()
        print("Available Pipeline Stages:")
        for stage in stages:
            print(f"  - {stage}")
        return
    
    if args.list_sub_pipelines:
        sub_pipelines = launcher.get_available_sub_pipelines(args.list_sub_pipelines)
        print(f"Available Sub-pipelines for {args.list_sub_pipelines}:")
        for stage, pipelines in sub_pipelines.items():
            print(f"  {stage}:")
            for pipeline in pipelines:
                print(f"    - {pipeline}")
        return
    
    # Load custom configuration if provided
    custom_config = None
    if args.config:
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
    
    # Convert string mode to enum
    mode_map = {
        'full': LauncherMode.FULL,
        'light': LauncherMode.LIGHT,
        'blank': LauncherMode.BLANK,
        'stage': LauncherMode.STAGE,
        'sub_pipeline': LauncherMode.SUB_PIPELINE
    }
    mode = mode_map[args.mode]
    
    # Convert execution mode to enum
    execution_mode_map = {
        'full': ExecutionModeType.FULL,
        'light': ExecutionModeType.LIGHT,
        'blank': ExecutionModeType.BLANK
    }
    execution_mode = execution_mode_map[args.execution_mode]
    
    # Convert string stage to enum if provided
    stage = None
    if args.stage:
        stage_map = {
            'data_collection': PipelineStage.DATA_COLLECTION,
            'market_analysis': PipelineStage.MARKET_ANALYSIS,
            'model_training': PipelineStage.MODEL_TRAINING,
            'backtesting': PipelineStage.BACKTESTING
        }
        stage = stage_map[args.stage]
    
    # Execute pipeline
    try:
        result = await launcher.execute_pipeline(
            mode=mode,
            symbol=args.symbol,
            exchange=args.exchange,
            timeframe=args.timeframe,
            data_dir=args.data_dir,
            stage=stage,
            sub_pipeline=args.sub_pipeline,
            execution_mode=execution_mode,
            custom_config=custom_config
        )
        
        # Print final results
        print("\n" + "=" * 80)
        print("🎯 EXECUTION COMPLETED")
        print("=" * 80)
        print(f"Status: {result.status.value}")
        print(f"Duration: {result.duration_seconds:.2f}s")
        print(f"Success Rate: {result.success_rate:.2%}")
        print("=" * 80)
        
        if result.status.value == 'failed':
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())