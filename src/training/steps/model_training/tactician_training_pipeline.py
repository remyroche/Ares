"""
Tactician Training Pipeline - Production Implementation

This module provides the Tactician training pipeline that orchestrates
the complete training process for Tactician models using the unified
BaseTrainer architecture from models_training directory.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
import time

# Import the actual implementations from models_training
from src.training.steps.models_training.components.tactician_base_training import (
    TacticianBaseTraining, TacticianBaseTrainingConfig, TacticianBaseTrainingResult,
    TacticianModelType, create_tactician_base_training, execute_tactician_base_training
)
from src.training.steps.models_training.components.tactician_ensemble_training import (
    TacticianEnsembleTraining, TacticianEnsembleTrainingConfig, TacticianEnsembleTrainingResult,
    TacticianEnsembleMethod, create_tactician_ensemble_training, execute_tactician_ensemble_training
)
from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_debug, tprint_performance, tprint_data_format
from src.core.decorators import handles_errors, traced, log_execution_time

@dataclass
class TacticianTrainingPipelineConfig:
    """Configuration for Tactician Training Pipeline."""
    base_model_types: List[TacticianModelType] = field(default_factory=lambda: [
        TacticianModelType.LIGHTGBM, 
        TacticianModelType.CATBOOST,
        TacticianModelType.NEURAL_NETWORK
    ])
    ensemble_models: bool = True
    output_directory: str = "generated/tactician_training_pipeline"
    enable_negative_learning: bool = False
    enable_enhanced_validation: bool = True
    timeframe: str = "15m"
    symbol: str = "ETHUSDT"
    
    # Training parameters
    training_params: Dict[str, Any] = field(default_factory=dict)
    validation_params: Dict[str, Any] = field(default_factory=dict)
    
    # Feature engineering parameters
    enable_entry_timing: bool = True
    enable_exit_timing: bool = True
    enable_position_sizing: bool = True

@dataclass
class TacticianTrainingPipelineResult:
    """Result from Tactician Training Pipeline."""
    success: bool = False
    base_models_path: Optional[str] = None
    ensemble_models_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    training_time: float = 0.0
    base_training_result: Optional[TacticianBaseTrainingResult] = None
    ensemble_training_result: Optional[TacticianEnsembleTrainingResult] = None

class TacticianTrainingPipeline(BaseStep):
    """Tactician Training Pipeline - Production Implementation."""

    def __init__(self, config: TacticianTrainingPipelineConfig, logger: Optional[logging.Logger] = None):
        """Initialize the Tactician training pipeline."""
        super().__init__("tactician_training_pipeline", config.__dict__, logger)
        self.config = config
        
        # Initialize components
        self.base_training = None
        self.ensemble_training = None
        
        tprint_info(f"🔧 Initialized TacticianTrainingPipeline")
        self.logger.info(f"Initialized TacticianTrainingPipeline with {len(self.config.base_model_types)} model types")

    @handles_errors(
        exceptions=(ValueError, RuntimeError, MemoryError),
        default_return=TacticianTrainingPipelineResult(
            success=False,
            error_message="Pipeline initialization failed"
        ),
        context="tactician training pipeline"
    )
    async def initialize(self) -> bool:
        """Initialize the pipeline components."""
        try:
            tprint_info("🔧 Initializing Tactician training pipeline components...")
            
            # Initialize base training
            base_config = {
                'model_types': self.config.base_model_types,
                'timeframe': self.config.timeframe,
                'symbol': self.config.symbol,
                'training_params': self.config.training_params,
                'validation_params': self.config.validation_params,
                'enable_entry_timing': self.config.enable_entry_timing,
                'enable_exit_timing': self.config.enable_exit_timing,
                'enable_position_sizing': self.config.enable_position_sizing
            }
            
            self.base_training = create_tactician_base_training(
                model_types=self.config.base_model_types,
                config=base_config,
                logger=self.logger
            )
            
            if not await self.base_training.initialize():
                tprint_error("❌ Base training initialization failed")
                return False
            
            # Initialize ensemble training if enabled
            if self.config.ensemble_models:
                ensemble_config = {
                    'base_models': self.config.base_model_types,
                    'ensemble_method': TacticianEnsembleMethod.STACKING,
                    'timeframe': self.config.timeframe,
                    'symbol': self.config.symbol,
                    'training_params': self.config.training_params,
                    'validation_params': self.config.validation_params,
                    'enable_entry_timing': self.config.enable_entry_timing,
                    'enable_exit_timing': self.config.enable_exit_timing,
                    'enable_position_sizing': self.config.enable_position_sizing
                }
                
                self.ensemble_training = create_tactician_ensemble_training(
                    base_models=self.config.base_model_types,
                    ensemble_method=TacticianEnsembleMethod.STACKING,
                    config=ensemble_config,
                    logger=self.logger
                )
            
            tprint_success("✅ Tactician training pipeline initialized")
            return True
            
        except Exception as e:
            tprint_error(f"❌ Pipeline initialization failed: {e}")
            self.logger.error(f"Pipeline initialization failed: {e}")
            return False

    @handles_errors(
        exceptions=(ValueError, RuntimeError, MemoryError),
        default_return=TacticianTrainingPipelineResult(
            success=False,
            error_message="Pipeline execution failed"
        ),
        context="tactician training pipeline"
    )
    async def execute(self, data: Dict[str, Any] = None) -> TacticianTrainingPipelineResult:
        """Execute the Tactician training pipeline."""
        try:
            tprint_info("⚔️ Starting Tactician training pipeline execution...")
            start_time = time.time()
            
            if data is None:
                data = {}
            
            # Debug input data format for troubleshooting
            tprint_data_format(data, "tactician_pipeline_input_data", level=tprint.LogLevel.INFO)
            
            # Step 1: Train base models
            tprint_info("🔧 Training Tactician base models...")
            base_result = await self.base_training.run(data)
            
            if not base_result.get('success', False):
                return TacticianTrainingPipelineResult(
                    success=False,
                    error_message=f"Base training failed: {base_result.get('error_message', 'Unknown error')}",
                    training_time=time.time() - start_time
                )
            
            # Extract base training result
            base_training_result = base_result.get('result')
            if not base_training_result:
                return TacticianTrainingPipelineResult(
                    success=False,
                    error_message="No base training result returned",
                    training_time=time.time() - start_time
                )
            
            # Step 2: Train ensemble models if enabled
            ensemble_training_result = None
            if self.config.ensemble_models and self.ensemble_training:
                tprint_info("🎯 Training Tactician ensemble models...")
                
                # Prepare ensemble data
                ensemble_data = {
                    'X_train': data.get('X_train'),
                    'y_train': data.get('y_train')
                }
                
                ensemble_result = await self.ensemble_training.run(ensemble_data)
                
                if ensemble_result.get('success', False):
                    ensemble_training_result = ensemble_result.get('result')
                    tprint_success("✅ Tactician ensemble training completed")
                else:
                    tprint_warning(f"⚠️ Tactician ensemble training failed: {ensemble_result.get('error_message', 'Unknown error')}")
            
            # Calculate total training time
            total_time = time.time() - start_time
            
            # Prepare metadata
            metadata = {
                'base_models_trained': list(base_training_result.models.keys()) if base_training_result.models else [],
                'ensemble_enabled': self.config.ensemble_models,
                'ensemble_completed': ensemble_training_result is not None,
                'training_time': total_time,
                'base_training_time': base_training_result.training_time,
                'ensemble_training_time': ensemble_training_result.training_time if ensemble_training_result else 0,
                'config': {
                    'timeframe': self.config.timeframe,
                    'symbol': self.config.symbol,
                    'model_types': [mt.value for mt in self.config.base_model_types],
                    'ensemble_method': self.config.ensemble_models
                }
            }
            
            # Debug final result format for troubleshooting
            tprint_data_format(metadata, "tactician_pipeline_metadata", level=tprint.LogLevel.INFO)
            
            result = TacticianTrainingPipelineResult(
                success=True,
                base_models_path=self.output_directory,
                ensemble_models_path=self.output_directory if self.config.ensemble_models else None,
                metadata=metadata,
                training_time=total_time,
                base_training_result=base_training_result,
                ensemble_training_result=ensemble_training_result
            )
            
            tprint_success(f"✅ Tactician training pipeline completed in {total_time:.2f}s")
            self.logger.info(f"Tactician training pipeline completed in {total_time:.2f}s")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Tactician training pipeline failed: {e}")
            self.logger.error(f"Tactician training pipeline failed: {e}")
            return TacticianTrainingPipelineResult(
                success=False,
                error_message=str(e),
                training_time=time.time() - start_time if 'start_time' in locals() else 0.0
            )

    @property
    def output_directory(self) -> str:
        """Get the output directory path."""
        return self.config.output_directory

    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        summary = {
            'pipeline_name': self.name,
            'config': self.config.__dict__,
            'base_training_initialized': self.base_training is not None,
            'ensemble_training_initialized': self.ensemble_training is not None
        }
        
        if self.base_training:
            summary['base_training_summary'] = self.base_training.get_training_summary()
        
        return summary

async def execute_tactician_training_pipeline(
    config: TacticianTrainingPipelineConfig,
    data: Dict[str, Any] = None,
    logger: Optional[logging.Logger] = None
) -> TacticianTrainingPipelineResult:
    """Execute Tactician training pipeline."""
    pipeline = TacticianTrainingPipeline(config, logger)
    
    # Initialize
    if not await pipeline.initialize():
        return TacticianTrainingPipelineResult(
            success=False,
            error_message="Pipeline initialization failed"
        )
    
    # Execute
    return await pipeline.execute(data)
