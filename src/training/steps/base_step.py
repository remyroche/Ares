"""Base step class for training steps."""

from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

from typing import Any, Dict
from abc import ABC, abstractmethod

class BaseStep(ABC):
    """Base class for all training steps."""
    @log_important_calls
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the step."""
        pass
    
    async def initialize(self) -> None:
        """Initialize the step."""
        pass

"""Base step class for training steps."""

from typing import Any, Dict