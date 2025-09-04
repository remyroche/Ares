#!/usr/bin/env python3
"""
Comprehensive Logger for Ares Trading System

This module provides comprehensive logging functionality with enhanced features.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

from src.utils.common_operations import get_current_datetime, format_datetime


class ComprehensiveLogger:
    """Comprehensive logger with enhanced features."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.log_dir = Path(config.get('log_dir', 'logs'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize loggers
        self._setup_loggers()
    
    def _setup_loggers(self) -> None:
        """Setup various loggers."""
        # Main logger
        self.main_logger = logging.getLogger('AresTradingSystem')
        self.main_logger.setLevel(logging.INFO)
        
        # Component logger
        self.component_logger = logging.getLogger('AresComponent')
        self.component_logger.setLevel(logging.INFO)
        
        # Global logger
        self.global_logger = logging.getLogger('AresGlobal')
        self.global_logger.setLevel(logging.INFO)
        
        # Setup handlers
        self._setup_handlers()
    
    def _setup_handlers(self) -> None:
        """Setup log handlers."""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # File handler
        log_file = self.log_dir / f"ares_{format_datetime(get_current_datetime(), '%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers
        for logger in [self.main_logger, self.component_logger, self.global_logger]:
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
    
    def get_component_logger(self, component_name: str) -> logging.Logger:
        """Get a component-specific logger."""
        return logging.getLogger(f'AresComponent.{component_name}')
    
    def get_global_logger(self) -> logging.Logger:
        """Get the global logger."""
        return self.global_logger
    
    def log_launcher_start(self, command: str, symbol: Optional[str] = None, exchange: Optional[str] = None) -> None:
        """Log launcher start."""
        self.main_logger.info(f"🚀 Launcher started: {command}")
        if symbol and exchange:
            self.main_logger.info(f"📊 Symbol: {symbol}, Exchange: {exchange}")
    
    def log_launcher_end(self, exit_code: int) -> None:
        """Log launcher end."""
        status = "SUCCESS" if exit_code == 0 else "FAILED"
        self.main_logger.info(f"🏁 Launcher ended: {status} (exit code: {exit_code})")
    
    def log_error(self, message: str, exc_info: bool = False) -> None:
        """Log an error."""
        self.main_logger.error(message, exc_info=exc_info)


# Global logger instance
_comprehensive_logger: Optional[ComprehensiveLogger] = None


def setup_comprehensive_logging(config: Dict[str, Any]) -> ComprehensiveLogger:
    """Setup comprehensive logging."""
    global _comprehensive_logger
    _comprehensive_logger = ComprehensiveLogger(config)
    return _comprehensive_logger


def get_comprehensive_logger() -> Optional[ComprehensiveLogger]:
    """Get the comprehensive logger instance."""
    return _comprehensive_logger


def ensure_comprehensive_logging_available() -> bool:
    """Ensure comprehensive logging is available."""
    return _comprehensive_logger is not None