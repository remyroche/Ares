"""Base step utilities."""
from typing import Any, Dict

class BaseStep:
    """Base step class."""
    
    def __init__(self, config: Dict[str, Any], step_number: str, step_name: str):
        """Initialize base step.
        
        Args:
            config: Configuration dictionary
            step_number: Step number (e.g., "02")
            step_name: Step name (e.g., "data_reading")
        """
        self.config = config
        self.step_number = step_number
        self.step_name = step_name
        self.logger = None  # Will be set by subclasses