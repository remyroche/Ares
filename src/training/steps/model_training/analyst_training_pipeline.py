"""
Analyst Training Pipeline - Production Implementation

This module provides the Analyst training pipeline that orchestrates
the complete training process for Analyst models using the unified
BaseTrainer architecture from models_training directory.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
import time

# Import the actual implementations from models_training
from src.training.steps.models_training.components.analyst_base_training import (
    AnalystBaseTraining, AnalystBaseTrainingConfig, AnalystBaseTrainingResult,
    AnalystModelType, create_analyst_base_training, execute_analyst_base_training
)
from src.training.steps.models_training.components.analyst_ensemble_training import (
    AnalystEnsembleTraining, AnalystEnsembleTrainingConfig, AnalystEnsembleTrainingResult,
    EnsembleMethod, create_analyst_ensemble_training, execute_analyst_ensemble_training
)
from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_debug, tprint_performance, tprint_data_format
from src.core.decorators import handles_errors, traced, log_execution_time

@dataclass
class AnalystTrainingPipelineConfig:
    """Configuration for Analyst Training Pipeline."""
    base_model_types: List[AnalystModelType] = field(default_factory=lambda: [
        AnalystModelType.LIGHTGBM, 
        AnalystModelType.CATBOOST
    ])
    ensemble_models: bool = True
    output_directory: str = "generated/analyst_training_pipeline"
    enable_negative_learning: bool = False
    enable_enhanced_validation: bool = True
    timeframe: str = "15m"
    symbol: str = "ETHUSDT"
    
    # Training parameters
    training_params: Dict[str, Any] = field(default_factory=dict)
    validation_params: Dict[str, Any] = field(default_factory=dict)
    
    # Feature engineering parameters
    enable_patchtst_features: bool = True
    enable_regime_features: bool = True
    enable_multi_timeframe: bool = True

@dataclass
class AnalystTrainingPipelineResult:
    """Result from Analyst Training Pipeline."""
    success: bool = False
    base_models_path: Optional[str] = None
    ensemble_models_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    training_time: float = 0.0
    base_training_result: Optional[AnalystBaseTrainingResult] = None
    ensemble_training_result: Optional[AnalystEnsembleTrainingResult] = None

class AnalystTrainingPipeline(BaseStep):
    """Analyst Training Pipeline - Production Implementation."""

    def __init__(self, config: AnalystTrainingPipelineConfig, logger: Optional[logging.Logger] = None):
        """Initialize the Analyst training pipeline."""
        super().__init__("analyst_training_pipeline", config.__dict__, logger)
        self.config = config
        
        # Initialize components
        self.base_training = None
        self.ensemble_training = None
        
        tprint_info(f"🔧 Initialized AnalystTrainingPipeline")
        self.logger.info(f"Initialized AnalystTrainingPipeline with {len(self.config.base_model_types)} model types")

    @handles_errors(
        exceptions=(ValueError, RuntimeError, MemoryError),
        default_return=AnalystTrainingPipelineResult(
            success=False,
            error_message="Pipeline initialization failed"
        ),
        context="analyst training pipeline"
    )
    async def initialize(self) -> bool:
        """Initialize the pipeline components."""
        try:
            tprint_info("🔧 Initializing Analyst training pipeline components...")
            
            # Initialize base training
            base_config = {
                'model_types': self.config.base_model_types,
                'timeframe': self.config.timeframe,
                'symbol': self.config.symbol,
                'training_params': self.config.training_params,
                'validation_params': self.config.validation_params,
                'enable_patchtst_features': self.config.enable_patchtst_features,
                'enable_regime_features': self.config.enable_regime_features,
                'enable_multi_timeframe': self.config.enable_multi_timeframe
            }
            
            self.base_training = create_analyst_base_training(
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
                    'model_name': f"analyst_ensemble_{self.config.timeframe}",
                    'timeframe': self.config.timeframe,
                    'base_models': self.config.base_model_types,
                    'ensemble_method': EnsembleMethod.VOTING,
                    'validation_split': self.config.validation_params.get('validation_split', 0.2),
                    'enable_evaluation': self.config.enable_enhanced_validation
                }
                
                self.ensemble_training = create_analyst_ensemble_training(
                    config=ensemble_config
                )
            
            tprint_success("✅ Analyst training pipeline initialized")
            return True
            
        except Exception as e:
            tprint_error(f"❌ Pipeline initialization failed: {e}")
            self.logger.error(f"Pipeline initialization failed: {e}")
            return False

    @handles_errors(
        exceptions=(ValueError, RuntimeError, MemoryError),
        default_return=AnalystTrainingPipelineResult(
            success=False,
            error_message="Pipeline execution failed"
        ),
        context="analyst training pipeline"
    )
    async def execute(self, data: Dict[str, Any] = None) -> AnalystTrainingPipelineResult:
        """Execute the Analyst training pipeline."""
        try:
            tprint_info("📊 Starting Analyst training pipeline execution...")
            start_time = time.time()
            
            if data is None:
                data = {}
            
            # Debug input data format for troubleshooting
            tprint_data_format(data, "pipeline_input_data", level=tprint.LogLevel.INFO)
            
            # Step 1: Train base models
            tprint_info("🔧 Training base models...")
            base_result = await self.base_training.run(data)
            
            if not base_result.get('success', False):
                return AnalystTrainingPipelineResult(
                    success=False,
                    error_message=f"Base training failed: {base_result.get('error_message', 'Unknown error')}",
                    training_time=time.time() - start_time
                )
            
            # Extract base training result
            base_training_result = base_result.get('result')
            if not base_training_result:
                return AnalystTrainingPipelineResult(
                    success=False,
                    error_message="No base training result returned",
                    training_time=time.time() - start_time
                )
            
            # Step 2: Train ensemble models if enabled
            ensemble_training_result = None
            if self.config.ensemble_models and self.ensemble_training:
                tprint_info("🎯 Training ensemble models...")
                
                # Prepare ensemble data
                ensemble_data = {
                    'features': data.get('X_train'),
                    'targets': data.get('y_train')
                }
                
                ensemble_result = await self.ensemble_training.execute(ensemble_data)
                
                if ensemble_result.get('success', False):
                    ensemble_training_result = ensemble_result
                    tprint_success("✅ Ensemble training completed")
                else:
                    tprint_warning(f"⚠️ Ensemble training failed: {ensemble_result.get('error', 'Unknown error')}")
            
            # Calculate total training time
            total_time = time.time() - start_time
            
            # Prepare metadata
            metadata = {
                'base_models_trained': list(base_training_result.models.keys()) if base_training_result.models else [],
                'ensemble_enabled': self.config.ensemble_models,
                'ensemble_completed': ensemble_training_result is not None,
                'training_time': total_time,
                'base_training_time': base_training_result.training_time,
                'ensemble_training_time': ensemble_training_result.get('training_time', 0) if ensemble_training_result else 0,
                'config': {
                    'timeframe': self.config.timeframe,
                    'symbol': self.config.symbol,
                    'model_types': [mt.value for mt in self.config.base_model_types],
                    'ensemble_method': self.config.ensemble_models
                }
            }
            
            # Debug final result format for troubleshooting
            tprint_data_format(metadata, "pipeline_metadata", level=tprint.LogLevel.INFO)
            
            result = AnalystTrainingPipelineResult(
                success=True,
                base_models_path=self.output_directory,
                ensemble_models_path=self.output_directory if self.config.ensemble_models else None,
                metadata=metadata,
                training_time=total_time,
                base_training_result=base_training_result,
                ensemble_training_result=ensemble_training_result
            )
            
            tprint_success(f"✅ Analyst training pipeline completed in {total_time:.2f}s")
            self.logger.info(f"Analyst training pipeline completed in {total_time:.2f}s")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Analyst training pipeline failed: {e}")
            self.logger.error(f"Analyst training pipeline failed: {e}")
            return AnalystTrainingPipelineResult(
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

async def execute_analyst_training_pipeline(
    config: AnalystTrainingPipelineConfig,
    data: Dict[str, Any] = None,
    logger: Optional[logging.Logger] = None
) -> AnalystTrainingPipelineResult:
    """Execute Analyst training pipeline."""
    pipeline = AnalystTrainingPipeline(config, logger)
    
    # Initialize
    if not await pipeline.initialize():
        return AnalystTrainingPipelineResult(
            success=False,
            error_message="Pipeline initialization failed"
        )
    
    # Execute
    return await pipeline.execute(data)
