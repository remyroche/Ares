"""
Efficient logging configuration for the pipeline components.

Provides centralized logging with throttling, level filtering, and performance monitoring.
"""

import logging
import time
from typing import Dict, Set, Optional
from collections import defaultdict
from functools import wraps
import threading

# Message throttling to prevent log spam
_throttled_messages: Dict[str, float] = {}
_throttle_lock = threading.Lock()
_THROTTLE_INTERVAL = 30.0  # seconds

# Performance tracking
_log_counts: Dict[str, int] = defaultdict(int)
_performance_lock = threading.Lock()

class EfficientLogger:
    """Efficient logger with message throttling and performance tracking."""
    
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.name = name
        self.level = level
        
    def _should_throttle(self, msg: str) -> bool:
        """Check if message should be throttled to prevent spam."""
        with _throttle_lock:
            key = f"{self.name}:{hash(msg) % 1000}"  # Hash to reduce memory
            now = time.time()
            
            if key in _throttled_messages:
                if now - _throttled_messages[key] < _THROTTLE_INTERVAL:
                    return True
                else:
                    _throttled_messages[key] = now
                    return False
            else:
                _throttled_messages[key] = now
                return False
    
    def _track_performance(self, level: str):
        """Track logging performance metrics."""
        with _performance_lock:
            _log_counts[f"{self.name}:{level}"] += 1
    
    def info(self, msg: str, throttle: bool = True):
        """Log info message with optional throttling."""
        if throttle and self._should_throttle(msg):
            return
        
        self.logger.info(msg)
        self._track_performance("info")
    
    def warning(self, msg: str, throttle: bool = True):
        """Log warning message with optional throttling."""
        if throttle and self._should_throttle(msg):
            return
        
        self.logger.warning(msg)
        self._track_performance("warning")
    
    def error(self, msg: str, throttle: bool = False):
        """Log error message (never throttled)."""
        self.logger.error(msg)
        self._track_performance("error")
    
    def success(self, msg: str, throttle: bool = True):
        """Log success message with optional throttling."""
        if throttle and self._should_throttle(msg):
            return
        
        self.logger.info(f"✅ {msg}")
        self._track_performance("success")
    
    def performance(self, msg: str, throttle: bool = True):
        """Log performance message with optional throttling."""
        if throttle and self._should_throttle(msg):
            return
        
        self.logger.info(f"📊 {msg}")
        self._track_performance("performance")

def get_logger(name: str, level: int = logging.INFO) -> EfficientLogger:
    """Get an efficient logger instance."""
    return EfficientLogger(name, level)

def setup_logging(level: int = logging.INFO, enable_performance_tracking: bool = True):
    """Setup efficient logging configuration."""
    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # Reduce noise from common libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    
    if enable_performance_tracking:
        # Log performance stats periodically
        def log_performance_stats():
            with _performance_lock:
                total_logs = sum(_log_counts.values())
                if total_logs > 0:
                    print(f"\n📈 Logging Performance Stats (Total: {total_logs}):")
                    for logger_level, count in sorted(_log_counts.items()):
                        if count > 0:
                            print(f"   {logger_level}: {count}")
                    print()
        
        # Could be called periodically in a real system
        # log_performance_stats()

def get_logging_stats() -> Dict[str, int]:
    """Get current logging performance statistics."""
    with _performance_lock:
        return dict(_log_counts)

def clear_throttled_messages():
    """Clear throttled message history (useful for testing)."""
    with _throttle_lock:
        _throttled_messages.clear()

# Convenience functions for backward compatibility
def setup_efficient_logging(level: int = logging.INFO):
    """Setup efficient logging with sensible defaults."""
    setup_logging(level, enable_performance_tracking=True)