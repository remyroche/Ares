"""
Unified Training Pipeline - Main Entry Point

This module provides the main entry point for the unified training pipeline,
replacing the fragmented training components with a single, comprehensive
orchestration system.

Key Features:
- Single entry point for all training operations
- Unified configuration management
- Comprehensive error handling and monitoring
- Role-specific training coordination
- Performance optimization and resource management
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_debug, tprint_performance
from src.utils.common_operations import (
    safe_divide, safe_correlation, safe_mean, safe_std, safe_float, safe_int,
    get_memory_usage, optimize_dataframe_memory, memory_checkpoint
)
from src.utils.common_utilities import calculate_data_quality_metrics, get_dataframe_info
from src.utils.math_validation import validate_finite, validate_positive, validate_range
from src.utils.hardware.m1_memory_optimizer import optimize_memory
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
from src.utils.kline_parquet import KlinesParquetManager
from src.core.decorators import handles_errors, traced, log_execution_time

from .core import (
    TrainingPipelineOrchestrator, PipelineConfig, PipelineResult,
    PipelinePhase, PipelineStatus, TrainingRole, ModelType
)


class UnifiedTrainingPipeline:
    """
    Unified training pipeline main entry point.
    
    This class provides a single, comprehensive interface for all training
    operations, replacing the fragmented training components with a unified
    orchestration system.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the unified training pipeline.
        
        Args:
            logger: Logger instance (optional)
        """
        self.logger = logger or system_logger.getChild("UnifiedTrainingPipeline")
        self._orchestrator = None
        
        self.logger.info("Initialized UnifiedTrainingPipeline")
    
    @handles_errors(
        exceptions=(ValueError, RuntimeError, MemoryError),
        default_return=PipelineResult(
            success=False,
            status=PipelineStatus.FAILED,
            execution_time=0.0,
            phases_completed=[],
            phases_failed=[PipelinePhase.INITIALIZATION],
            errors=["Pipeline initialization failed"]
        ),
        context="unified training pipeline"
    )
    async def execute_training_pipeline(
        self,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None,
        analyst_targets: Optional[pd.Series] = None,
        tactician_targets: Optional[pd.Series] = None
    ) -> PipelineResult:
        """
        Execute the complete training pipeline.
        
        Args:
            data: Training data
            config: Pipeline configuration (optional)
            analyst_targets: Analyst target variables (optional)
            tactician_targets: Tactician target variables (optional)
            
        Returns:
            Pipeline execution result
        """
        try:
            self.logger.info("🚀 Starting unified training pipeline...")
            
            # Create pipeline configuration
            pipeline_config = self._create_pipeline_config(config)
            
            # Create orchestrator
            self._orchestrator = TrainingPipelineOrchestrator(pipeline_config, self.logger)
            
            # Execute pipeline
            result = await self._orchestrator.execute_pipeline(
                data, analyst_targets, tactician_targets
            )
            
            if result.success:
                self.logger.info("✅ Unified training pipeline completed successfully")
                tprint_success("Training pipeline completed successfully")
            else:
                self.logger.error("❌ Unified training pipeline failed")
                tprint_error("Training pipeline failed")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Unified training pipeline failed: {e}")
            return PipelineResult(
                success=False,
                status=PipelineStatus.FAILED,
                execution_time=0.0,
                phases_completed=[],
                phases_failed=[PipelinePhase.INITIALIZATION],
                errors=[f"Pipeline execution failed: {e}"]
            )
    
    def _create_pipeline_config(self, config: Optional[Dict[str, Any]] = None) -> PipelineConfig:
        """Create pipeline configuration from input."""
        try:
            # Default configuration
            default_config = {
                'symbol': 'ETHUSDT',
                'timeframe': '15m',
                'execution_mode': 'full',
                'enable_analyst': True,
                'enable_tactician': True,
                'enable_ensemble': True,
                'max_parallel_tasks': 3,
                'enable_monitoring': True,
                'monitoring_interval': 30.0,
                'enable_health_checks': True
            }
            
            # Merge with provided configuration
            if config:
                default_config.update(config)
            
            # Create pipeline configuration
            pipeline_config = PipelineConfig(
                symbol=default_config.get('symbol', 'ETHUSDT'),
                timeframe=default_config.get('timeframe', '15m'),
                execution_mode=default_config.get('execution_mode', 'full'),
                enable_analyst=default_config.get('enable_analyst', True),
                enable_tactician=default_config.get('enable_tactician', True),
                enable_ensemble=default_config.get('enable_ensemble', True),
                max_parallel_tasks=default_config.get('max_parallel_tasks', 3),
                memory_limit_mb=default_config.get('memory_limit_mb'),
                timeout_seconds=default_config.get('timeout_seconds'),
                enable_monitoring=default_config.get('enable_monitoring', True),
                monitoring_interval=default_config.get('monitoring_interval', 30.0),
                enable_health_checks=default_config.get('enable_health_checks', True),
                analyst_config=default_config.get('analyst_config'),
                tactician_config=default_config.get('tactician_config'),
                ensemble_config=default_config.get('ensemble_config'),
                custom_params=default_config.get('custom_params', {})
            )
            
            self.logger.info(f"Created pipeline configuration: {pipeline_config.symbol} {pipeline_config.timeframe}")
            return pipeline_config
            
        except Exception as e:
            self.logger.error(f"Pipeline configuration creation failed: {e}")
            # Return minimal configuration
            return PipelineConfig()
    
    async def train_analyst_models(
        self,
        data: pd.DataFrame,
        targets: pd.Series,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train analyst models only.
        
        Args:
            data: Training data
            targets: Target variables
            config: Training configuration (optional)
            
        Returns:
            Training result
        """
        try:
            self.logger.info("📊 Starting analyst models training...")
            
            # Create analyst-only configuration
            analyst_config = {
                'enable_analyst': True,
                'enable_tactician': False,
                'enable_ensemble': False,
                **(config or {})
            }
            
            # Execute pipeline
            result = await self.execute_training_pipeline(data, analyst_config, targets)
            
            if result.success and result.analyst_result:
                self.logger.info("✅ Analyst models training completed")
                tprint_success("Analyst models trained successfully")
                return result.analyst_result
            else:
                self.logger.error("❌ Analyst models training failed")
                tprint_error("Analyst models training failed")
                return {'success': False, 'error_message': 'Analyst training failed'}
                
        except Exception as e:
            self.logger.error(f"Analyst models training failed: {e}")
            return {'success': False, 'error_message': str(e)}
    
    async def train_tactician_models(
        self,
        data: pd.DataFrame,
        targets: pd.Series,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train tactician models only.
        
        Args:
            data: Training data
            targets: Target variables
            config: Training configuration (optional)
            
        Returns:
            Training result
        """
        try:
            self.logger.info("⚔️ Starting tactician models training...")
            
            # Create tactician-only configuration
            tactician_config = {
                'enable_analyst': False,
                'enable_tactician': True,
                'enable_ensemble': False,
                **(config or {})
            }
            
            # Execute pipeline
            result = await self.execute_training_pipeline(data, tactician_config, tactician_targets=targets)
            
            if result.success and result.tactician_result:
                self.logger.info("✅ Tactician models training completed")
                tprint_success("Tactician models trained successfully")
                return result.tactician_result
            else:
                self.logger.error("❌ Tactician models training failed")
                tprint_error("Tactician models training failed")
                return {'success': False, 'error_message': 'Tactician training failed'}
                
        except Exception as e:
            self.logger.error(f"Tactician models training failed: {e}")
            return {'success': False, 'error_message': str(e)}
    
    async def train_ensemble_models(
        self,
        data: pd.DataFrame,
        analyst_targets: pd.Series,
        tactician_targets: Optional[pd.Series] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train ensemble models.
        
        Args:
            data: Training data
            analyst_targets: Analyst target variables
            tactician_targets: Tactician target variables (optional)
            config: Training configuration (optional)
            
        Returns:
            Training result
        """
        try:
            self.logger.info("🎯 Starting ensemble models training...")
            
            # Create ensemble-only configuration
            ensemble_config = {
                'enable_analyst': False,
                'enable_tactician': False,
                'enable_ensemble': True,
                **(config or {})
            }
            
            # Execute pipeline
            result = await self.execute_training_pipeline(
                data, ensemble_config, analyst_targets, tactician_targets
            )
            
            if result.success and result.ensemble_result:
                self.logger.info("✅ Ensemble models training completed")
                tprint_success("Ensemble models trained successfully")
                return result.ensemble_result
            else:
                self.logger.error("❌ Ensemble models training failed")
                tprint_error("Ensemble models training failed")
                return {'success': False, 'error_message': 'Ensemble training failed'}
                
        except Exception as e:
            self.logger.error(f"Ensemble models training failed: {e}")
            return {'success': False, 'error_message': str(e)}
    
    async def train_all_models(
        self,
        data: pd.DataFrame,
        analyst_targets: pd.Series,
        tactician_targets: Optional[pd.Series] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> PipelineResult:
        """
        Train all models (analyst, tactician, ensemble).
        
        Args:
            data: Training data
            analyst_targets: Analyst target variables
            tactician_targets: Tactician target variables (optional)
            config: Training configuration (optional)
            
        Returns:
            Complete pipeline result
        """
        try:
            self.logger.info("🚀 Starting complete training pipeline...")
            
            # Execute full pipeline
            result = await self.execute_training_pipeline(
                data, config, analyst_targets, tactician_targets
            )
            
            if result.success:
                self.logger.info("✅ Complete training pipeline completed")
                tprint_success("All models trained successfully")
            else:
                self.logger.error("❌ Complete training pipeline failed")
                tprint_error("Complete training pipeline failed")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Complete training pipeline failed: {e}")
            return PipelineResult(
                success=False,
                status=PipelineStatus.FAILED,
                execution_time=0.0,
                phases_completed=[],
                phases_failed=[PipelinePhase.INITIALIZATION],
                errors=[f"Complete training pipeline failed: {e}"]
            )
    
    def get_pipeline_status(self) -> Optional[Dict[str, Any]]:
        """Get current pipeline status."""
        if self._orchestrator:
            return self._orchestrator.get_pipeline_status()
        return None
    
    def get_required_dependencies(self) -> List[str]:
        """Get list of required dependencies."""
        return [
            'pandas', 'numpy', 'scikit-learn', 'lightgbm', 'catboost',
            'torch', 'psutil', 'asyncio'
        ]
    
    def get_processing_capabilities(self) -> Dict[str, Any]:
        """Get pipeline processing capabilities."""
        return {
            'supports_parallel_processing': True,
            'supports_monitoring': True,
            'supports_health_checks': True,
            'supports_ensemble': True,
            'memory_efficient': True,
            'role_specific_training': True
        }


# Convenience functions for easy usage
async def create_unified_training_pipeline(
    logger: Optional[logging.Logger] = None
) -> UnifiedTrainingPipeline:
    """Create a new unified training pipeline instance."""
    return UnifiedTrainingPipeline(logger)


async def execute_quick_training(
    data: pd.DataFrame,
    analyst_targets: pd.Series,
    tactician_targets: Optional[pd.Series] = None,
    symbol: str = "ETHUSDT",
    timeframe: str = "15m",
    logger: Optional[logging.Logger] = None,
    enable_artifact_chaining: bool = False
) -> PipelineResult:
    """
    Execute quick training with minimal configuration.
    
    Args:
        data: Training data
        analyst_targets: Analyst target variables
        tactician_targets: Tactician target variables (optional)
        symbol: Trading symbol
        timeframe: Trading timeframe
        logger: Logger instance (optional)
        
    Returns:
        Pipeline execution result
    """
    pipeline = UnifiedTrainingPipeline(logger)
    
    config = {
        'symbol': symbol,
        'timeframe': timeframe,
        'execution_mode': 'light',  # Quick training mode
        'enable_artifact_chaining': enable_artifact_chaining
    }
    
    return await pipeline.execute_training_pipeline(
        data, config, analyst_targets, tactician_targets
    )


async def execute_full_training(
    data: pd.DataFrame,
    analyst_targets: pd.Series,
    tactician_targets: Optional[pd.Series] = None,
    symbol: str = "ETHUSDT",
    timeframe: str = "15m",
    logger: Optional[logging.Logger] = None,
    enable_artifact_chaining: bool = True
) -> PipelineResult:
    """
    Execute full training with comprehensive configuration.
    
    Args:
        data: Training data
        analyst_targets: Analyst target variables
        tactician_targets: Tactician target variables (optional)
        symbol: Trading symbol
        timeframe: Trading timeframe
        logger: Logger instance (optional)
        
    Returns:
        Pipeline execution result
    """
    pipeline = UnifiedTrainingPipeline(logger)
    
    config = {
        'symbol': symbol,
        'timeframe': timeframe,
        'execution_mode': 'full',  # Full training mode
        'enable_monitoring': True,
        'enable_health_checks': True,
        'max_parallel_tasks': 3,
        'enable_artifact_chaining': enable_artifact_chaining
    }
    
    return await pipeline.execute_training_pipeline(
        data, config, analyst_targets, tactician_targets
    )
