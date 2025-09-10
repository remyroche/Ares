#!/usr/bin/env python3
"""
Enhanced simple logger with file output support.

This module provides a simple logger that includes both console and file output,
addressing the empty log file issue while maintaining compatibility.
"""

import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler


def create_enhanced_simple_logger(log_dir: str = "logs", log_level: str = "INFO"):
    """
    Create an enhanced simple logger with both console and file output.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Enhanced logger instance
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_path / f'ares_{timestamp}.log'
    
    # Create logger
    logger = logging.getLogger('AresEnhanced')
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Remove any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Set propagate to False to avoid duplicate messages
    logger.propagate = False
    
    return logger


class EnhancedMockLogger:
    """Enhanced mock logger with file output support."""
    
    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        self.log_dir = log_dir
        self.log_level = log_level
        self.logger = create_enhanced_simple_logger(log_dir, log_level)
        self.child_loggers = {}
    
    def getChild(self, name):
        """Create child logger with file output."""
        if name not in self.child_loggers:
            child_logger = logging.getLogger(f'AresEnhanced.{name}')
            child_logger.setLevel(getattr(logging, self.log_level.upper(), logging.INFO))
            
            # Remove existing handlers
            child_logger.handlers.clear()
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # Add console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            child_logger.addHandler(console_handler)
            
            # Add file handler
            log_path = Path(self.log_dir)
            log_path.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_path / f'ares_{name}_{timestamp}.log'
            
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setFormatter(formatter)
            child_logger.addHandler(file_handler)
            
            child_logger.propagate = False
            self.child_loggers[name] = child_logger
        
        return self.child_loggers[name]
    
    def info(self, msg):
        """Log info message."""
        print(f"📝 [ENHANCED_LOGGER] INFO: {msg}")
        self.logger.info(msg)
    
    def warning(self, msg):
        """Log warning message."""
        print(f"⚠️ [ENHANCED_LOGGER] WARNING: {msg}")
        self.logger.warning(msg)
    
    def error(self, msg):
        """Log error message."""
        print(f"❌ [ENHANCED_LOGGER] ERROR: {msg}")
        self.logger.error(msg)
    
    def debug(self, msg):
        """Log debug message."""
        print(f"🔍 [ENHANCED_LOGGER] DEBUG: {msg}")
        self.logger.debug(msg)
    
    def critical(self, msg):
        """Log critical message."""
        print(f"💥 [ENHANCED_LOGGER] CRITICAL: {msg}")
        self.logger.critical(msg)
    
    def exception(self, msg):
        """Log exception message."""
        print(f"💥 [ENHANCED_LOGGER] EXCEPTION: {msg}")
        self.logger.exception(msg)


# Create enhanced logger instance
print("🔧 [ENHANCED_SIMPLE_LOGGER] Creating enhanced logger instance...")
enhanced_system_logger = create_enhanced_simple_logger()
print("✅ [ENHANCED_SIMPLE_LOGGER] Enhanced logger instance created")

# Create enhanced mock logger
print("🔧 [ENHANCED_SIMPLE_LOGGER] Setting up EnhancedMockLogger class...")
enhanced_mock_logger = EnhancedMockLogger()
print("✅ [ENHANCED_SIMPLE_LOGGER] EnhancedMockLogger created")

# Attach enhanced mock logger to system logger
print("🔧 [ENHANCED_SIMPLE_LOGGER] Attaching EnhancedMockLogger to system_logger...")
enhanced_system_logger.getChild = enhanced_mock_logger.getChild
print("✅ [ENHANCED_SIMPLE_LOGGER] EnhancedMockLogger attached successfully")

print("✅ [ENHANCED_SIMPLE_LOGGER] Enhanced simple logger created successfully")
print("=" * 60)