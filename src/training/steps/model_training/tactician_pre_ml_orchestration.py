"""
Tactician Pre-ML Orchestration - Production Implementation

This module provides the Tactician Pre-ML orchestration that handles
the complete pre-ML training pipeline for Tactician models using the
unified BaseTrainer architecture from models_training directory.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
import time

# Import the actual implementation from models_training
from src.training.steps.model_training.tactician_pre_ml_orchestrator import (
    TacticianPreMLOrchestrator as ActualTacticianPreMLOrchestrator,
    TacticianPreMLConfig as ActualTacticianPreMLConfig,
    TacticianPreMLResult as ActualTacticianPreMLResult
)
from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_debug, tprint_performance, tprint_data_format
from src.core.decorators import handles_errors, traced, log_execution_time

@dataclass
class TacticianPreMLConfig:
    """Configuration for Tactician Pre-ML Orchestration."""
    timeframe: str = "15m"
    output_directory: str = "generated/tactician_pre_ml"
    enable_gate_protection: bool = True
    enable_interactive_features: bool = True
    enable_feature_selection: bool = True
    
    # Additional configuration parameters
    symbol: str = "ETHUSDT"
    enable_negative_learning: bool = False
    enable_enhanced_validation: bool = True
    
    # Feature engineering parameters
    enable_entry_timing: bool = True
    enable_exit_timing: bool = True
    enable_position_sizing: bool = True
    
    # Training parameters
    training_params: Dict[str, Any] = field(default_factory=dict)
    validation_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TacticianPreMLResult:
    """Result from Tactician Pre-ML Orchestration."""
    success: bool = False
    features_path: Optional[str] = None
    selected_features_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    execution_time: float = 0.0
    orchestrator_result: Optional[ActualTacticianPreMLResult] = None

class TacticianPreMLOrchestrator(BaseStep):
    """Tactician Pre-ML Orchestrator - Production Implementation."""

    def __init__(self, config: TacticianPreMLConfig, logger: Optional[logging.Logger] = None):
        """Initialize the Tactician Pre-ML orchestrator."""
        super().__init__("tactician_pre_ml_orchestrator", config.__dict__, logger)
        self.config = config
        
        # Initialize the actual orchestrator
        self.orchestrator = None
        
        tprint_info(f"🔧 Initialized TacticianPreMLOrchestrator")
        self.logger.info(f"Initialized TacticianPreMLOrchestrator for {self.config.timeframe} timeframe")

    @handles_errors(
        exceptions=(ValueError, RuntimeError, MemoryError),
        default_return=TacticianPreMLResult(
            success=False,
            error_message="Orchestrator initialization failed"
        ),
        context="tactician pre-ml orchestration"
    )
    async def initialize(self) -> bool:
        """Initialize the orchestrator components."""
        try:
            tprint_info("🔧 Initializing Tactician Pre-ML orchestrator components...")
            
            # Create configuration for the actual orchestrator
            orchestrator_config = ActualTacticianPreMLConfig(
                timeframe=self.config.timeframe,
                output_directory=self.config.output_directory,
                enable_gate_protection=self.config.enable_gate_protection,
                enable_interactive_features=self.config.enable_interactive_features,
                enable_feature_selection=self.config.enable_feature_selection,
                symbol=self.config.symbol,
                enable_negative_learning=self.config.enable_negative_learning,
                enable_enhanced_validation=self.config.enable_enhanced_validation,
                enable_entry_timing=self.config.enable_entry_timing,
                enable_exit_timing=self.config.enable_exit_timing,
                enable_position_sizing=self.config.enable_position_sizing,
                training_params=self.config.training_params,
                validation_params=self.config.validation_params
            )
            
            # Initialize the actual orchestrator
            self.orchestrator = ActualTacticianPreMLOrchestrator(
                config=orchestrator_config,
                logger=self.logger
            )
            
            tprint_success("✅ Tactician Pre-ML orchestrator initialized")
            return True
            
        except Exception as e:
            tprint_error(f"❌ Orchestrator initialization failed: {e}")
            self.logger.error(f"Orchestrator initialization failed: {e}")
            return False

    @handles_errors(
        exceptions=(ValueError, RuntimeError, MemoryError),
        default_return=TacticianPreMLResult(
            success=False,
            error_message="Orchestration execution failed"
        ),
        context="tactician pre-ml orchestration"
    )
    async def execute(self, data: Dict[str, Any] = None) -> TacticianPreMLResult:
        """Execute the Tactician Pre-ML orchestration."""
        try:
            tprint_info("⚔️ Starting Tactician Pre-ML orchestration execution...")
            start_time = time.time()
            
            if data is None:
                data = {}
            
            # Debug input data format for troubleshooting
            tprint_data_format(data, "pre_ml_orchestration_input_data", level=tprint.LogLevel.INFO)
            
            # Execute the actual orchestrator
            orchestrator_result = await self.orchestrator.execute(data)
            
            if not orchestrator_result.success:
                return TacticianPreMLResult(
                    success=False,
                    error_message=f"Orchestrator execution failed: {orchestrator_result.error_message}",
                    execution_time=time.time() - start_time
                )
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Prepare metadata
            metadata = {
                'orchestrator_result': orchestrator_result.metadata if orchestrator_result.metadata else {},
                'execution_time': execution_time,
                'config': {
                    'timeframe': self.config.timeframe,
                    'symbol': self.config.symbol,
                    'enable_gate_protection': self.config.enable_gate_protection,
                    'enable_interactive_features': self.config.enable_interactive_features,
                    'enable_feature_selection': self.config.enable_feature_selection
                }
            }
            
            # Debug final result format for troubleshooting
            tprint_data_format(metadata, "pre_ml_orchestration_metadata", level=tprint.LogLevel.INFO)
            
            result = TacticianPreMLResult(
                success=True,
                features_path=orchestrator_result.features_path,
                selected_features_path=orchestrator_result.selected_features_path,
                metadata=metadata,
                execution_time=execution_time,
                orchestrator_result=orchestrator_result
            )
            
            tprint_success(f"✅ Tactician Pre-ML orchestration completed in {execution_time:.2f}s")
            self.logger.info(f"Tactician Pre-ML orchestration completed in {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Tactician Pre-ML orchestration failed: {e}")
            self.logger.error(f"Tactician Pre-ML orchestration failed: {e}")
            return TacticianPreMLResult(
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time if 'start_time' in locals() else 0.0
            )

    def get_orchestration_summary(self) -> Dict[str, Any]:
        """Get comprehensive orchestration summary."""
        summary = {
            'orchestrator_name': self.name,
            'config': self.config.__dict__,
            'orchestrator_initialized': self.orchestrator is not None
        }
        
        if self.orchestrator:
            summary['orchestrator_summary'] = getattr(self.orchestrator, 'get_summary', lambda: {})()
        
        return summary

async def execute_tactician_pre_ml_orchestration(
    config: TacticianPreMLConfig,
    data: Dict[str, Any] = None,
    logger: Optional[logging.Logger] = None
) -> TacticianPreMLResult:
    """Execute Tactician Pre-ML orchestration."""
    orchestrator = TacticianPreMLOrchestrator(config, logger)
    
    # Initialize
    if not await orchestrator.initialize():
        return TacticianPreMLResult(
            success=False,
            error_message="Orchestrator initialization failed"
        )
    
    # Execute
    return await orchestrator.execute(data)
