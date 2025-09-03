"""Core training manager - simplified and focused.

This module provides the main training manager that coordinates
the training pipeline execution.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.decorators import handles_errors
from src.training.simplified_training_manager import SimplifiedTrainingManager
from src.utils.logger import system_logger


class TrainingManager:
    """Main training manager for the ML pipeline.
    
    This is a facade that provides a simple interface to the training pipeline
    while delegating to specialized components.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize training manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("TrainingManager")
        
        # Delegate to simplified training manager
        self.pipeline_manager = SimplifiedTrainingManager(config)
        
        # Training state
        self.is_initialized = False
        self.current_execution = None
        
    @handles_errors(
        exceptions=(Exception,),
        default_return=False,
        context="training manager initialization"
    )
    async def initialize(self) -> bool:
        """Initialize the training manager.
        
        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("🔧 Initializing Training Manager...")
            
            # Initialize pipeline manager
            if not await self.pipeline_manager.initialize():
                self.logger.error("❌ Failed to initialize pipeline manager")
                return False
            
            self.is_initialized = True
            self.logger.info("✅ Training Manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Initialization failed: {e}")
            return False
    
    async def train(
        self,
        symbol: str,
        exchange: str,
        start_step: Optional[str] = None,
        end_step: Optional[str] = None,
        force_rerun: bool = False
    ) -> Dict[str, Any]:
        """Execute the training pipeline.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            exchange: Exchange name (e.g., "binance")
            start_step: Optional starting step
            end_step: Optional ending step
            force_rerun: Force re-execution of completed steps
            
        Returns:
            Training results
        """
        if not self.is_initialized:
            await self.initialize()
        
        self.logger.info(f"🚀 Starting training for {symbol} on {exchange}")
        
        # Update config with runtime parameters
        self.pipeline_manager.symbol = symbol
        self.pipeline_manager.exchange = exchange
        
        # Execute pipeline
        result = await self.pipeline_manager.execute_pipeline(
            start_step=start_step,
            end_step=end_step,
            force_rerun=force_rerun
        )
        
        if result["success"]:
            self.logger.info("✅ Training completed successfully")
        else:
            self.logger.error(f"❌ Training failed: {result.get('error', 'Unknown error')}")
        
        return result
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current training status.
        
        Returns:
            Status dictionary
        """
        return self.pipeline_manager.get_pipeline_status()
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.pipeline_manager.cleanup()
        self.logger.info("🧹 Training Manager cleaned up")


# Factory function
async def create_training_manager(config: Dict[str, Any]) -> TrainingManager:
    """Create and initialize a training manager.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized TrainingManager
    """
    manager = TrainingManager(config)
    if await manager.initialize():
        return manager
    else:
        raise RuntimeError("Failed to initialize training manager")