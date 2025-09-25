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
- Enhanced error handling and logging
- Hardware optimization integration
- ML utilities integration
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

# Add the project root to the Python path BEFORE any imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Temporarily use simple logger to bypass initialization issues
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import utilities with proper error handling
try:
    from src.utils.logger import system_logger
    from src.utils.tprint import tprint, tprint_error, tprint_success, tprint_warning, tprint_info
    from src.utils.common_operations import (
        safe_json_dump, safe_json_load, ensure_directory, 
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        integrate_with_m1_optimizers, cleanup_m1_optimizers
    )
    from src.utils.math_validation import (
        validate_finite, validate_positive, safe_divide, safe_log, safe_sqrt
    )
    from src.utils.serialization_utils import UniversalSerializer
    UTILS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Some utilities not available: {e}")
    UTILS_AVAILABLE = False
    # Fallback imports
    from src.utils.logger import system_logger
    from src.utils.tprint import tprint, tprint_error, tprint_success, tprint_warning, tprint_info

from src.core.decorators import handles_errors, traced, log_execution_time
from src.training.steps.main_training_pipeline import (
    MainTrainingPipeline, MainPipelineConfig, MainPipelineResult,
    PipelineStage, ExecutionMode, get_full_pipeline_config,
    get_light_pipeline_config, get_blank_pipeline_config, SubPipelineStatus
)

logger = system_logger.getChild('AresLauncher')
logger.propagate = True
if logger.handlers:
    logger.handlers.clear()

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
        """Initialize the Ares launcher with enhanced error handling and utilities."""
        try:
            tprint("🚀 Starting AresLauncher initialization...")
            
            # Initialize core components
            self.logger = logger.getChild('AresLauncher')
            self.pipeline = MainTrainingPipeline()
            self.current_execution: Optional[MainPipelineResult] = None
            self.execution_history: List[MainPipelineResult] = []
            
            # Initialize utility systems
            self.utils_available = UTILS_AVAILABLE
            self.serializer = None
            self.m1_optimizers = None
            
            if self.utils_available:
                try:
                    self._initialize_utilities()
                    tprint_success("✅ Utilities initialized")
                except Exception as e:
                    tprint_warning(f"⚠️ Utilities initialization failed: {e}")
                    self.utils_available = False
            
            # Setup systems
            self._setup_logging()
            self._setup_monitoring()
            
            tprint_success("✅ AresLauncher initialized successfully")
            
        except Exception as e:
            tprint_error(f"❌ AresLauncher initialization failed: {e}")
            raise
    
    def _initialize_utilities(self):
        """Initialize utility systems."""
        try:
            self.serializer = UniversalSerializer()
            self.m1_optimizers = integrate_with_m1_optimizers()
        except Exception as e:
            tprint_warning(f"⚠️ Utility initialization failed: {e}")
            self.utils_available = False
    
    def _setup_logging(self):
        """Setup logging."""
        self.logger.info("🚀 Ares Launcher Initialized")
        if self.utils_available:
            self.logger.info("🔧 Utilities: Available")
    
    def _setup_monitoring(self):
        """Setup monitoring."""
        self.monitoring_enabled = True
        self.progress_callbacks: List[callable] = []
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
    
    
    
    
    async def execute_pipeline(
        self,
        mode: LauncherMode = LauncherMode.FULL,
        symbol: str = "ETHUSDT",
        exchange: str = "binance",
        timeframe: str = "15m",
        data_dir: str = "historical_data",
        stage: Optional[PipelineStage] = None,
        sub_pipeline: Optional[str] = None,
        execution_mode: ExecutionModeType = ExecutionModeType.FULL,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> MainPipelineResult:
        """
        Execute the training pipeline with granular control and enhanced error handling.
        
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
        try:
            tprint("🚀 [EXECUTE_PIPELINE] Starting pipeline execution...")
            
            # Validate inputs with math validation utilities
            if self.utils_available:
                try:
                    validate_positive(len(symbol), "symbol length")
                    validate_positive(len(exchange), "exchange length")
                    validate_positive(len(timeframe), "timeframe length")
                    validate_positive(len(data_dir), "data_dir length")
                except Exception as e:
                    tprint_warning(f"⚠️ Input validation warning: {e}")
            
            tprint_info(f"🚀 [EXECUTE_PIPELINE] Mode: {mode.value}")
            tprint_info(f"🚀 [EXECUTE_PIPELINE] Symbol: {symbol}")
            tprint_info(f"🚀 [EXECUTE_PIPELINE] Exchange: {exchange}")
            tprint_info(f"🚀 [EXECUTE_PIPELINE] Timeframe: {timeframe}")
            tprint_info(f"🚀 [EXECUTE_PIPELINE] Data directory: {data_dir}")
            tprint_info(f"🚀 [EXECUTE_PIPELINE] Execution mode: {execution_mode.value}")
            
            if stage:
                tprint_info(f"🚀 [EXECUTE_PIPELINE] Target stage: {stage.value}")
            if sub_pipeline:
                tprint_info(f"🚀 [EXECUTE_PIPELINE] Target sub-pipeline: {sub_pipeline}")
            if custom_config:
                tprint_info(f"🚀 [EXECUTE_PIPELINE] Custom config provided: {len(custom_config)} parameters")
            
            self.logger.info(f"🚀 Starting pipeline execution: {mode.value}")
            
            # Create configuration based on mode with error handling
            try:
                tprint("🚀 [EXECUTE_PIPELINE] Creating configuration...")
                config = self._create_config(
                    mode, symbol, exchange, timeframe, data_dir, 
                    stage, sub_pipeline, execution_mode, custom_config
                )
                tprint_success("✅ [EXECUTE_PIPELINE] Configuration created successfully")
            except Exception as e:
                tprint_error(f"❌ [EXECUTE_PIPELINE] Configuration creation failed: {e}")
                raise
            
            # Execute based on mode with proper error handling
            try:
                tprint("🚀 [EXECUTE_PIPELINE] Determining execution path...")
                if mode == LauncherMode.SUB_PIPELINE and sub_pipeline:
                    tprint_info(f"🚀 [EXECUTE_PIPELINE] Executing sub-pipeline: {sub_pipeline}")
                    return await self._execute_sub_pipeline(sub_pipeline, config)
                elif mode == LauncherMode.STAGE and stage:
                    tprint_info(f"🚀 [EXECUTE_PIPELINE] Executing stage: {stage.value}")
                    return await self._execute_stage(stage, config)
                else:
                    tprint_info("🚀 [EXECUTE_PIPELINE] Executing full pipeline")
                    return await self._execute_full_pipeline(config)
            except Exception as e:
                tprint_error(f"❌ [EXECUTE_PIPELINE] Pipeline execution failed: {e}")
                raise
                
        except Exception as e:
            tprint_error(f"❌ [EXECUTE_PIPELINE] Critical error in pipeline execution: {e}")
            # Return a failed result instead of raising
            return MainPipelineResult(
                pipeline_id=f"failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                status=SubPipelineStatus.FAILED,
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_seconds=0.0,
                error_message=str(e)
            )
    
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
        tprint("⚙️ [CREATE_CONFIG] Starting configuration creation...")
        tprint(f"⚙️ [CREATE_CONFIG] Mode: {mode.value}")
        tprint(f"⚙️ [CREATE_CONFIG] Symbol: {symbol}")
        tprint(f"⚙️ [CREATE_CONFIG] Exchange: {exchange}")
        tprint(f"⚙️ [CREATE_CONFIG] Timeframe: {timeframe}")
        tprint(f"⚙️ [CREATE_CONFIG] Data directory: {data_dir}")
        tprint(f"⚙️ [CREATE_CONFIG] Execution mode: {execution_mode.value}")
        
        # Base configuration
        tprint("⚙️ [CREATE_CONFIG] Creating base configuration...")
        base_config = {
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': timeframe,
            'data_dir': data_dir,
            'custom_params': custom_config or {}
        }
        tprint("✅ [CREATE_CONFIG] Base configuration created")
        
        # Filter base_config to only include supported parameters for each config function
        tprint("⚙️ [CREATE_CONFIG] Filtering configuration parameters...")
        supported_params = ['symbol', 'exchange', 'timeframe', 'data_dir']
        filtered_config = {k: v for k, v in base_config.items() if k in supported_params}
        tprint(f"✅ [CREATE_CONFIG] Filtered config: {list(filtered_config.keys())}")
        
        # Mode-specific configuration
        tprint("⚙️ [CREATE_CONFIG] Creating mode-specific configuration...")
        if mode == LauncherMode.FULL:
            tprint("⚙️ [CREATE_CONFIG] Using FULL pipeline configuration")
            config = get_full_pipeline_config(**filtered_config)
        elif mode == LauncherMode.LIGHT:
            tprint("⚙️ [CREATE_CONFIG] Using LIGHT pipeline configuration")
            config = get_light_pipeline_config(**filtered_config)
        elif mode == LauncherMode.BLANK:
            tprint("⚙️ [CREATE_CONFIG] Using BLANK pipeline configuration")
            config = get_blank_pipeline_config(**filtered_config)
        elif mode == LauncherMode.STAGE and stage:
            tprint(f"⚙️ [CREATE_CONFIG] Creating STAGE configuration for: {stage.value}")
            config = self._create_stage_config(stage, base_config, execution_mode)
        elif mode == LauncherMode.SUB_PIPELINE and sub_pipeline:
            tprint(f"⚙️ [CREATE_CONFIG] Creating SUB_PIPELINE configuration for: {sub_pipeline}")
            config = self._create_sub_pipeline_config(sub_pipeline, base_config, execution_mode)
        else:
            # Default to full configuration
            tprint("⚙️ [CREATE_CONFIG] Using DEFAULT (FULL) pipeline configuration")
            config = get_full_pipeline_config(**filtered_config)
        
        tprint("✅ [CREATE_CONFIG] Configuration creation completed successfully")
        return config
    
    def _create_stage_config(self, stage: PipelineStage, base_config: Dict[str, Any], execution_mode: ExecutionModeType) -> MainPipelineConfig:
        """Create configuration for a specific stage."""
        tprint(f"🎭 [STAGE_CONFIG] Creating stage configuration for: {stage.value}")
        tprint(f"🎭 [STAGE_CONFIG] Execution mode: {execution_mode.value}")
        
        # Filter base_config to only include supported parameters for each config function
        tprint("🎭 [STAGE_CONFIG] Filtering configuration parameters...")
        supported_params = ['symbol', 'exchange', 'timeframe', 'data_dir']
        filtered_config = {k: v for k, v in base_config.items() if k in supported_params}
        tprint(f"✅ [STAGE_CONFIG] Filtered config: {list(filtered_config.keys())}")
        
        # Use the provided timeframe for all stages
        tprint(f"📊 [STAGE_CONFIG] Using timeframe for {stage.value}: {filtered_config.get('timeframe', '1m')}")

        # Get configuration based on execution mode using centralized configuration
        tprint("🎭 [STAGE_CONFIG] Getting base configuration...")
        if execution_mode == ExecutionModeType.FULL:
            tprint("🎭 [STAGE_CONFIG] Using FULL execution mode configuration")
            config = get_full_pipeline_config(**filtered_config)
        elif execution_mode == ExecutionModeType.LIGHT:
            tprint("🎭 [STAGE_CONFIG] Using LIGHT execution mode configuration")
            config = get_light_pipeline_config(**filtered_config)
        elif execution_mode == ExecutionModeType.BLANK:
            tprint("🎭 [STAGE_CONFIG] Using BLANK execution mode configuration")
            config = get_blank_pipeline_config(**filtered_config)
        else:
            tprint("🎭 [STAGE_CONFIG] Using DEFAULT (FULL) execution mode configuration")
            config = get_full_pipeline_config(**filtered_config)
        
        # Enable only the specified stage
        tprint(f"🎭 [STAGE_CONFIG] Enabling stage: {stage.value}")
        config.enabled_stages = [stage]
        tprint("✅ [STAGE_CONFIG] Stage enabled")
        
        # Get all available sub-pipelines for the stage
        tprint("🎭 [STAGE_CONFIG] Getting available sub-pipelines...")
        available_sub_pipelines = self.pipeline.get_available_sub_pipelines(stage)
        tprint(f"🎭 [STAGE_CONFIG] Found {len(available_sub_pipelines)} sub-pipelines: {available_sub_pipelines}")
        config.enabled_sub_pipelines[stage] = available_sub_pipelines
        tprint("✅ [STAGE_CONFIG] Sub-pipelines configured")
        
        # Add intensity parameters to stage configuration
        tprint("🎭 [STAGE_CONFIG] Adding intensity parameters...")
        if config.training_mode_config:
            config.stage_params[stage] = {
                'intensity_percentage': config.intensity_percentage,
                'training_mode_config': config.training_mode_config,
                'model_training': config.training_mode_config.get('model_training', {}),
                'validation': config.training_mode_config.get('validation', {}),
                'optimization': config.training_mode_config.get('optimization', {})
            }
            tprint("✅ [STAGE_CONFIG] Intensity parameters added")
        else:
            tprint("⚠️ [STAGE_CONFIG] No training mode config available")
        
        tprint("✅ [STAGE_CONFIG] Stage configuration completed successfully")
        return config
    
    def _create_sub_pipeline_config(self, sub_pipeline: str, base_config: Dict[str, Any], execution_mode: ExecutionModeType) -> MainPipelineConfig:
        """Create configuration for a specific sub-pipeline."""
        tprint(f"🔧 [SUB_PIPELINE_CONFIG] Creating sub-pipeline configuration for: {sub_pipeline}")
        tprint(f"🔧 [SUB_PIPELINE_CONFIG] Execution mode: {execution_mode.value}")
        
        # Set the execution mode in base config
        tprint("🔧 [SUB_PIPELINE_CONFIG] Setting execution mode in base config...")
        base_config['mode'] = ExecutionMode(execution_mode.value)
        tprint("✅ [SUB_PIPELINE_CONFIG] Execution mode set")
        
        # Filter base_config to only include supported parameters for each config function
        tprint("🔧 [SUB_PIPELINE_CONFIG] Filtering configuration parameters...")
        supported_params = ['symbol', 'exchange', 'timeframe', 'data_dir']
        filtered_config = {k: v for k, v in base_config.items() if k in supported_params}

        # SET 15M AS DEFAULT TIMEFRAME FOR NAS-RELATED SUB-PIPELINES
        nas_sub_pipelines = [
            'nas_regime_discovery',     # Discover market regimes using NAS
            'nas_clustering',           # NAS-based regime clustering
            'nas_models_training',      # Train regime detection models using NAS regime labels
            'nas_ensemble_training',    # Train ensemble regime detection models using NAS regime labels
            'nas'                       # Combined NAS regime discovery + clustering
        ]
        
        # Set 15m as default for NAS sub-pipelines
        if sub_pipeline in nas_sub_pipelines:
            original_timeframe = filtered_config.get('timeframe', '1m')
            if original_timeframe == '1m':  # Only override if using default
                filtered_config['timeframe'] = '15m'
                tprint(f"🎯 [SUB_PIPELINE_CONFIG] NAS sub-pipeline detected: {sub_pipeline}")
                tprint(f"🎯 [SUB_PIPELINE_CONFIG] Setting default timeframe: {original_timeframe} → 15m")
                tprint("🎯 [SUB_PIPELINE_CONFIG] Using 15m data for better granularity and regime detection")
            else:
                tprint(f"📊 [SUB_PIPELINE_CONFIG] Using specified timeframe for {sub_pipeline}: {original_timeframe}")
        else:
            tprint(f"📊 [SUB_PIPELINE_CONFIG] Using timeframe for {sub_pipeline}: {filtered_config.get('timeframe', '1m')}")

        tprint(f"✅ [SUB_PIPELINE_CONFIG] Filtered config: {list(filtered_config.keys())}")
        
        # Get configuration based on execution mode
        tprint("🔧 [SUB_PIPELINE_CONFIG] Getting base configuration...")
        if execution_mode == ExecutionModeType.FULL:
            tprint("🔧 [SUB_PIPELINE_CONFIG] Using FULL execution mode configuration")
            config = get_full_pipeline_config(**filtered_config)
        elif execution_mode == ExecutionModeType.LIGHT:
            tprint("🔧 [SUB_PIPELINE_CONFIG] Using LIGHT execution mode configuration")
            config = get_light_pipeline_config(**filtered_config)
        elif execution_mode == ExecutionModeType.BLANK:
            tprint("🔧 [SUB_PIPELINE_CONFIG] Using BLANK execution mode configuration")
            config = get_blank_pipeline_config(**filtered_config)
        else:
            tprint("🔧 [SUB_PIPELINE_CONFIG] Using DEFAULT (FULL) execution mode configuration")
            config = get_full_pipeline_config(**filtered_config)
        
        # Find which stage contains the sub-pipeline
        tprint("🔧 [SUB_PIPELINE_CONFIG] Finding target stage for sub-pipeline...")
        target_stage = None
        for stage in PipelineStage:
            available_sub_pipelines = self.pipeline.get_available_sub_pipelines(stage)
            if sub_pipeline in available_sub_pipelines:
                target_stage = stage
                tprint(f"🔧 [SUB_PIPELINE_CONFIG] Found sub-pipeline in stage: {stage.value}")
                break
        
        if not target_stage:
            tprint(f"❌ [SUB_PIPELINE_CONFIG] Sub-pipeline '{sub_pipeline}' not found in any stage")
            raise ValueError(f"Sub-pipeline '{sub_pipeline}' not found in any stage")
        
        # Enable only the target stage and sub-pipeline
        tprint(f"🔧 [SUB_PIPELINE_CONFIG] Enabling stage: {target_stage.value}")
        config.enabled_stages = [target_stage]
        tprint(f"🔧 [SUB_PIPELINE_CONFIG] Enabling sub-pipeline: {sub_pipeline}")
        config.enabled_sub_pipelines[target_stage] = [sub_pipeline]
        tprint("✅ [SUB_PIPELINE_CONFIG] Stage and sub-pipeline enabled")
        
        # Set single stage execution mode for individual sub-pipeline execution
        # Enable chaining for SR components to automatically run the full SR pipeline
        sr_components = ['sr_parameter_optimization', 'sr_detection', 'sr_clustering']
        if sub_pipeline in sr_components:
            config.single_stage_only = False
            tprint(f"🔗 [SUB_PIPELINE_CONFIG] SR chaining enabled for {sub_pipeline} - will automatically run: sr_parameter_optimization -> sr_detection -> sr_clustering")
        else:
            config.single_stage_only = True
            tprint("🎯 [SUB_PIPELINE_CONFIG] Single stage execution mode enabled")
        
        # Add intensity parameters to stage configuration
        tprint("🔧 [SUB_PIPELINE_CONFIG] Adding intensity parameters...")
        if config.training_mode_config:
            config.stage_params[target_stage] = {
                'intensity_percentage': config.intensity_percentage,
                'training_mode_config': config.training_mode_config,
                'model_training': config.training_mode_config.get('model_training', {}),
                'validation': config.training_mode_config.get('validation', {}),
                'optimization': config.training_mode_config.get('optimization', {})
            }
            tprint("✅ [SUB_PIPELINE_CONFIG] Intensity parameters added")
        else:
            tprint("⚠️ [SUB_PIPELINE_CONFIG] No training mode config available")
        
        tprint("✅ [SUB_PIPELINE_CONFIG] Sub-pipeline configuration completed successfully")
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
                        # Outcome file creation handled by MainTrainingPipeline
                
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
        
        # Calculate overall metrics for stage execution
        self.pipeline._calculate_pipeline_metrics(result)
        
        # Create outcome file for this stage
        if result.stage_results and stage in result.stage_results:
            stage_results = result.stage_results[stage]
            for sub_result in stage_results:
                if hasattr(sub_result, 'sub_pipeline_name'):
                    # Outcome file creation handled by MainTrainingPipeline
        
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
        
        # Execute only the specified sub-pipeline with automatic chaining
        # Use execute_sub_pipeline_with_chain for automatic sequential execution
        sub_pipeline_result = await self.pipeline.execute_sub_pipeline_with_chain(target_stage, sub_pipeline, config)
        
        # Create a MainPipelineResult to maintain compatibility
        result = MainPipelineResult(
            pipeline_id=f"sub_pipeline_{sub_pipeline}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            status=sub_pipeline_result.status if sub_pipeline_result else SubPipelineStatus.FAILED,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=sub_pipeline_result.duration_seconds if sub_pipeline_result else 0.0,
            error_message=sub_pipeline_result.error_message if sub_pipeline_result else "Sub-pipeline execution failed"
        )
        
        # Add the sub-pipeline result to the stage results
        if sub_pipeline_result:
            result.stage_results[target_stage] = [sub_pipeline_result]
        
        # Calculate overall metrics for sub-pipeline execution
        self.pipeline._calculate_pipeline_metrics(result)
        
        # Create outcome file for this sub-pipeline
        if result.stage_results and target_stage in result.stage_results:
            stage_results = result.stage_results[target_stage]
            for sub_result in stage_results:
                if hasattr(sub_result, 'sub_pipeline_name') and sub_result.sub_pipeline_name == sub_pipeline:
                    # Outcome file creation handled by MainTrainingPipeline
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
        
        # Save artifacts - DISABLED: Only outcome file should be created
        # await self._save_artifacts(artifacts, 'full_pipeline_artifacts.json')
        
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
        
        # Save artifacts - DISABLED: Only outcome file should be created
        # await self._save_artifacts(artifacts, f'{stage.value}_artifacts.json')
        
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
        
        # Save artifacts - DISABLED: Only outcome file should be created
        # await self._save_artifacts(artifacts, f'{sub_pipeline}_artifacts.json')
        
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
            'nas_clustering': "NAS-based regime clustering (DEPRECATED - use nas_tas_clustering instead)",
            'nas_tas_clustering': "Advanced regime clustering using combined NAS-TAS approaches with economic awareness and ensemble methods",
            'unified_regime_discovery': "Discover market regimes using unified NAS-TAS approach (combines Neural Architecture Search & Tree-based Architecture Search)",
            'nas_regime_discovery': "Discover market regimes using NAS (DEPRECATED - use unified_regime_discovery instead)",
            'nas_models_training': "Train regime detection models using NAS regime labels",
            'nas_ensemble_training': "Train ensemble regime detection models using NAS regime labels",
            'nas': "Combined NAS regime discovery + clustering (DEPRECATED - use unified_regime_discovery instead)",
            'multi_horizon_profit_labeler': "Multi-horizon profit probability labeling (replacement for triple barrier)",
            'triple_barrier_labeling': "Apply triple barrier method",
            'feature_lookback_optimization': "Optimize feature lookback periods",
            'pid_based_feature_generation': "PID-based feature generation with interaction, polynomial, and cross-timeframe features",
            'sr_feature_integration': "Integrate SR-specific features into feature set",
            
            # Model Training (10 sub-pipelines)
            'analyst_model_training': "Train analyst-specific models",
            'tactician_pre_ml_orchestration': "Pre-ML processing: separate long/short signals, optimize features, generate PID features, apply horizon labeling, select features",
            'tactician_dual_training': "Train multiple Tactician models: 4 base models + 1 ensemble for long signals, 4 base models + 1 ensemble for short signals (8 total models)",
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
            'basic_backtesting_pre': "Basic historical backtesting (pre-optimization baseline)",
            'final_parameters_optimization': "System-wide parameter optimization",
            'basic_backtesting_post': "Basic historical backtesting (post-optimization comparison)",
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
            'nas_clustering': ['sr_clustering'],  # DEPRECATED - use nas_tas_clustering instead
            'nas_tas_clustering': ['sr_clustering'],
            'unified_regime_discovery': ['sr_clustering'],
            'nas_regime_discovery': ['nas_tas_clustering'],  # DEPRECATED - use unified_regime_discovery instead
            'nas_models_training': ['unified_regime_discovery'],  # Updated to use unified discovery
            'nas_ensemble_training': ['nas_models_training'],
            'feature_lookback_optimization': ['hmm_regime_discovery', 'unified_regime_discovery'],
            'pid_based_feature_generation': ['feature_lookback_optimization'],
            'multi_horizon_profit_labeler': ['pid_based_feature_generation'],
            'triple_barrier_labeling': ['hmm_regime_discovery'],
            'sr_feature_integration': ['multi_horizon_profit_labeler'],
            
            # Model Training dependencies
            'hmm_training': ['sr_feature_integration'],
            'analyst_model_training': ['hmm_training'],
            'analyst_ensemble_training': ['analyst_model_training'],
            'tactician_pre_ml_orchestration': ['analyst_ensemble_training'],
            'tactician_dual_training': ['tactician_pre_ml_orchestration'],
            'regime_specific_training': ['tactician_ensemble_training'],
            'model_validation': ['regime_specific_training'],
            'model_persistence': ['model_validation'],
            'model_evaluation': ['model_persistence'],
            
            # Backtesting dependencies
            'basic_backtesting_pre': [],
            'final_parameters_optimization': ['basic_backtesting_pre'],
            'basic_backtesting_post': ['final_parameters_optimization'],
            'walk_forward_validation': ['basic_backtesting_post'],
            'monte_carlo_simulation': ['walk_forward_validation'],
            'ab_testing': ['monte_carlo_simulation'],
            'performance_analytics': ['ab_testing'],
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
            'nas_clustering': ['nas_clusters.json'],
            'nas_tas_clustering': ['nas_tas_clustering_report.json', 'nas_tas_regime_assignments.parquet'],
            'unified_regime_discovery': ['unified_regime_consolidated_report.json', 'unified_regime_assignments.parquet'],
            'nas_regime_discovery': ['nas_regime_assignments.parquet'],
            'nas_models_training': ['nas_models_training_result.json'],
            'nas_ensemble_training': ['nas_ensemble_training_result.json'],
            'feature_lookback_optimization': ['optimized_features.parquet'],
            'pid_based_feature_generation': ['pid_based_features.parquet'],
            'multi_horizon_profit_labeler': ['multi_horizon_labels.parquet'],
            'triple_barrier_labeling': ['labels.parquet'],
            'sr_feature_integration': ['sr_features.json'],
            
            # Model Training outputs
            'hmm_training': ['hmm_model.pkl'],
            'analyst_model_training': ['analyst_model.pkl'],
            'analyst_ensemble_training': ['analyst_ensemble.pkl'],
            'tactician_pre_ml_orchestration': ['tactician_pre_ml_results.pkl', 'long_training_data.parquet', 'short_training_data.parquet'],
            'tactician_dual_training': ['tactician_long_model.pkl', 'tactician_short_model.pkl', 'tactician_long_ensemble.pkl', 'tactician_short_ensemble.pkl'],
            'regime_specific_training': ['regime_models.pkl'],
            'model_validation': ['validation_results.json'],
            'model_persistence': ['persisted_models.pkl'],
            'model_evaluation': ['evaluation_results.json'],
            
            # Backtesting outputs
            'walk_forward_validation': ['backtest_results.json'],
            'monte_carlo_simulation': ['mc_results.json'],
            'ab_testing': ['ab_test_results.json'],
            'basic_backtesting_pre': ['basic_backtest_pre_results.json'],
            'final_parameters_optimization': ['optimized_parameters.json'],
            'basic_backtesting_post': ['basic_backtest_post_results.json'],
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
Note: The following sub-pipelines are DEPRECATED and will be removed in future versions:
  - nas_regime_discovery (use unified_regime_discovery instead)
  - nas_clustering (use nas_tas_clustering instead)
  - nas (use unified_regime_discovery instead)

Examples:
  # Full pipeline execution (1460 days, 100% intensity)
  python ares_launcher.py --mode full --symbol ETHUSDT --exchange binance

  # Light pipeline execution (10 days, 5% intensity)
  python ares_launcher.py --mode light --symbol ETHUSDT

  # Execute specific stage with full execution mode (1460 days, 100% intensity)
  python ares_launcher.py --mode stage --stage data_collection --execution-mode full --symbol ETHUSDT

  # Execute specific stage with light execution mode (10 days, 5% intensity)
  python ares_launcher.py --mode stage --stage market_analysis --execution-mode light --symbol ETHUSDT

  # Execute specific sub-pipeline with blank execution mode (180 days, 10% intensity)
  python ares_launcher.py --mode sub_pipeline --sub_pipeline sr_detection --execution-mode blank --symbol ETHUSDT

  # Execute specific sub-pipeline with full execution mode (1460 days, 100% intensity)
  python ares_launcher.py --mode sub_pipeline --sub_pipeline unified_regime_discovery --execution-mode full --symbol ETHUSDT

  # Execute NAS-TAS clustering with full execution mode
  python ares_launcher.py --mode sub_pipeline --sub_pipeline nas_tas_clustering --execution-mode full --symbol ETHUSDT

  # Execute specific sub-pipeline with full execution mode (1460 days, 100% intensity) - RECOMMENDED
  python ares_launcher.py --mode sub_pipeline --sub_pipeline unified_regime_discovery --execution-mode full --symbol ETHUSDT

  # Execute specific sub-pipeline with full execution mode (DEPRECATED - use unified_regime_discovery instead)
  python ares_launcher.py --mode sub_pipeline --sub_pipeline nas_regime_discovery --execution-mode full --symbol ETHUSDT

  # Execute basic backtesting (pre-optimization baseline)
  python ares_launcher.py --mode sub_pipeline --sub_pipeline basic_backtesting_pre --execution-mode full --symbol ETHUSDT

  # Execute basic backtesting (post-optimization comparison)
  python ares_launcher.py --mode sub_pipeline --sub_pipeline basic_backtesting_post --execution-mode full --symbol ETHUSDT

  # Execute walk-forward validation (after post-optimization basic backtesting)
  python ares_launcher.py --mode sub_pipeline --sub_pipeline walk_forward_validation --execution-mode full --symbol ETHUSDT

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
        default='historical_data',
        help='Data directory (default: historical_data)'
    )
    
    parser.add_argument(
        '--stage',
        choices=['data_collection', 'market_analysis', 'model_training', 'backtesting'],
        help='Specific stage to execute (for stage mode)'
    )
    
    parser.add_argument(
        '--sub-pipeline', '--sub_pipeline',
        help='Specific sub-pipeline to execute (for sub_pipeline mode). Available: data_download, sr_detection, unified_regime_discovery, nas_tas_clustering, nas_regime_discovery (DEPRECATED), nas_clustering (DEPRECATED), nas_models_training, nas_ensemble_training, hmm_training, analyst_model_training, analyst_ensemble_training, tactician_pre_ml_orchestration, tactician_dual_training, basic_backtesting_pre, basic_backtesting_post, walk_forward_validation, etc.'
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
        help='List available sub-pipelines for a stage. Use with --stage to see sub-pipelines for that stage.'
    )
    
    return parser

async def main():
    """Main entry point."""
    tprint("🎯 [MAIN] Starting Ares Launcher CLI...")
    tprint("🎯 [MAIN] Creating CLI argument parser...")
    parser = create_cli_parser()
    tprint("✅ [MAIN] CLI parser created")
    
    tprint("🎯 [MAIN] Parsing command line arguments...")
    args = parser.parse_args()
    tprint("✅ [MAIN] Arguments parsed successfully")
    tprint(f"🎯 [MAIN] Mode: {args.mode}")
    tprint(f"🎯 [MAIN] Symbol: {args.symbol}")
    tprint(f"🎯 [MAIN] Exchange: {args.exchange}")
    tprint(f"🎯 [MAIN] Timeframe: {args.timeframe}")
    
    # Initialize launcher
    tprint("🎯 [MAIN] Initializing AresLauncher...")
    launcher = AresLauncher()
    tprint("✅ [MAIN] AresLauncher initialized successfully")
    
    # Handle list commands
    if args.list_stages:
        tprint("📋 [MAIN] Listing available pipeline stages...")
        stages = launcher.get_available_stages()
        tprint("Available Pipeline Stages:")
        for stage in stages:
            tprint(f"  - {stage}")
        tprint("✅ [MAIN] Stage listing completed")
        return
    
    if args.list_sub_pipelines:
        tprint(f"📋 [MAIN] Listing available sub-pipelines for: {args.list_sub_pipelines}")
        sub_pipelines = launcher.get_available_sub_pipelines(args.list_sub_pipelines)
        tprint(f"Available Sub-pipelines for {args.list_sub_pipelines}:")
        for stage, pipelines in sub_pipelines.items():
            tprint(f"  {stage}:")
            for pipeline in pipelines:
                tprint(f"    - {pipeline}")
        tprint("✅ [MAIN] Sub-pipeline listing completed")
        return
    
    # Load custom configuration if provided
    custom_config = None
    if args.config:
        tprint(f"📁 [MAIN] Loading custom configuration from: {args.config}")
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
        tprint(f"✅ [MAIN] Custom configuration loaded: {len(custom_config)} parameters")
    else:
        tprint("📁 [MAIN] No custom configuration provided, using defaults")
    
    # Convert string mode to enum
    tprint("🔄 [MAIN] Converting string modes to enums...")
    mode_map = {
        'full': LauncherMode.FULL,
        'light': LauncherMode.LIGHT,
        'blank': LauncherMode.BLANK,
        'stage': LauncherMode.STAGE,
        'sub_pipeline': LauncherMode.SUB_PIPELINE
    }
    mode = mode_map[args.mode]
    tprint(f"✅ [MAIN] Launcher mode converted: {mode.value}")
    
    # Convert execution mode to enum
    execution_mode_map = {
        'full': ExecutionModeType.FULL,
        'light': ExecutionModeType.LIGHT,
        'blank': ExecutionModeType.BLANK
    }
    execution_mode = execution_mode_map[args.execution_mode]
    tprint(f"✅ [MAIN] Execution mode converted: {execution_mode.value}")
    
    # Convert string stage to enum if provided
    stage = None
    if args.stage:
        tprint(f"🔄 [MAIN] Converting stage string to enum: {args.stage}")
        stage_map = {
            'data_collection': PipelineStage.DATA_COLLECTION,
            'market_analysis': PipelineStage.MARKET_ANALYSIS,
            'model_training': PipelineStage.MODEL_TRAINING,
            'backtesting': PipelineStage.BACKTESTING
        }
        stage = stage_map[args.stage]
        tprint(f"✅ [MAIN] Stage converted: {stage.value}")
    else:
        tprint("📋 [MAIN] No specific stage provided")
    
    # Execute pipeline
    tprint("🚀 [MAIN] Starting pipeline execution...")
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
        tprint("✅ [MAIN] Pipeline execution completed successfully")
        
        # Print final results
        tprint("\n" + "=" * 80)
        tprint("🎯 EXECUTION COMPLETED")
        tprint("=" * 80)
        tprint(f"Status: {result.status.value}")
        tprint(f"Duration: {result.duration_seconds:.2f}s")
        tprint(f"Success Rate: {result.success_rate:.2%}")
        tprint("=" * 80)
        
        if result.status.value == 'failed':
            tprint("❌ [MAIN] Pipeline execution failed, exiting with code 1")
            sys.exit(1)
        elif result.success_rate == 0.0:
            tprint("⚠️ [MAIN] Pipeline completed but no sub-pipelines succeeded, exiting with code 0")
        else:
            tprint("✅ [MAIN] Pipeline execution successful, exiting with code 0")
            
    except Exception as e:
        tprint(f"❌ [MAIN] Pipeline execution failed with exception: {e}")
        logger.error(f"❌ Execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())