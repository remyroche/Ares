# src/training/core/pipeline_base.py

from typing import Any, Dict
from src.utils.logger import system_logger

class TrainingPipeline:
    """Base training pipeline class for dependency injection."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the training pipeline.

        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("TrainingPipeline")
        self.logger.info("TrainingPipeline initialized")

    async def execute(self, context: Dict[str, Any]) -> bool:
        """Execute the training pipeline.

        Args:
            context: Execution context

        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.info("Executing training pipeline")
        
        try:
            # Execute the actual training pipeline steps
            if hasattr(self, 'steps') and self.steps:
                for step in self.steps:
                    self.logger.info(f"Executing step: {step.__class__.__name__}")
                    if hasattr(step, 'execute'):
                        result = await step.execute()
                        if not result:
                            self.logger.error(f"Step {step.__class__.__name__} failed")
                            return False
                    else:
                        self.logger.warning(f"Step {step.__class__.__name__} has no execute method")
            
            self.logger.info("Training pipeline completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {e}")
            return False

    async def initialize(self) -> bool:
        """Initialize the pipeline.

        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.info("Pipeline initialized successfully")
        return True
