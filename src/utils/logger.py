"""Mock logger module for testing purposes."""

import logging
from typing import Optional


class MockLogger:
    """Mock logger that provides basic logging functionality."""
    
    def __init__(self, name: str):
        self.name = name
        self.handlers = []
        self.level = logging.INFO
    
    def getChild(self, name: str) -> 'MockLogger':
        """Get a child logger."""
        return MockLogger(f"{self.name}.{name}")
    
    def info(self, message: str) -> None:
        """Log info message."""
        print(f"[INFO] {self.name}: {message}")
    
    def error(self, message: str) -> None:
        """Log error message."""
        print(f"[ERROR] {self.name}: {message}")
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        print(f"[WARNING] {self.name}: {message}")
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        print(f"[DEBUG] {self.name}: {message}")
    
    def exception(self, message: str) -> None:
        """Log exception message."""
        print(f"[EXCEPTION] {self.name}: {message}")


# Create a system logger instance
system_logger = MockLogger("System")
