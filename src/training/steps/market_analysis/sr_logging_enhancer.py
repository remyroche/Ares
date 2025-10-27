"""
Enhanced logging and debugging capabilities for SR detection.

This module provides comprehensive logging, debugging, and diagnostic tools
for SR detection methods with structured logging and performance tracking.
"""

import logging
import time
import json
import traceback
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np

from src.utils.logger import system_logger

class LogLevel(Enum):
    """Enhanced log levels for SR detection."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    PERFORMANCE = "PERFORMANCE"
    VALIDATION = "VALIDATION"
    DETECTION = "DETECTION"

@dataclass
class SRDetectionEvent:
    """Structured event for SR detection logging."""
    event_type: str
    method_name: str
    timestamp: float
    level: LogLevel
    message: str
    data_size: int = 0
    execution_time: float = 0.0
    memory_usage: float = 0.0
    result_count: int = 0
    error_message: str = ""
    metadata: Dict[str, Any] = None

class SRLoggingEnhancer:
    """Enhanced logging system for SR detection."""
    
    def __init__(self, log_file: Optional[str] = None, enable_structured_logging: bool = True):
        self.logger = system_logger.getChild('SRLoggingEnhancer')
        self.enable_structured_logging = enable_structured_logging
        self.events: List[SRDetectionEvent] = []
        self.performance_metrics: Dict[str, List[float]] = {}
        self.error_counts: Dict[str, int] = {}
        
        # Setup file logging if specified
        if log_file:
            self._setup_file_logging(log_file)
        
        # Setup structured logging
        if enable_structured_logging:
            self._setup_structured_logging()
    
    def _setup_file_logging(self, log_file: str):
        """Setup file-based logging."""
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.info(f"File logging enabled: {log_file}")
        except Exception as e:
            self.logger.error(f"Failed to setup file logging: {e}")
    
    def _setup_structured_logging(self):
        """Setup structured logging with JSON format."""
        try:
            # Create structured logger
            self.structured_logger = logging.getLogger('SRDetectionStructured')
            self.structured_logger.setLevel(logging.DEBUG)
            
            # Prevent duplicate logs
            if not self.structured_logger.handlers:
                handler = logging.StreamHandler()
                handler.setLevel(logging.DEBUG)
                
                # Custom formatter for structured logging
                class StructuredFormatter(logging.Formatter):
                    def format(self, record):
                        if hasattr(record, 'structured_data'):
                            return json.dumps(record.structured_data, default=str)
                        return super().format(record)
                
                handler.setFormatter(StructuredFormatter())
                self.structured_logger.addHandler(handler)
            
            self.logger.info("Structured logging enabled")
        except Exception as e:
            self.logger.error(f"Failed to setup structured logging: {e}")
    
    def log_detection_event(self, event: SRDetectionEvent):
        """Log a structured detection event."""
        try:
            # Store event
            self.events.append(event)
            
            # Update performance metrics
            if event.method_name not in self.performance_metrics:
                self.performance_metrics[event.method_name] = []
            self.performance_metrics[event.method_name].append(event.execution_time)
            
            # Update error counts
            if event.level in [LogLevel.ERROR, LogLevel.CRITICAL]:
                self.error_counts[event.method_name] = self.error_counts.get(event.method_name, 0) + 1
            
            # Log to appropriate level
            log_message = self._format_event_message(event)
            
            if event.level == LogLevel.DEBUG:
                self.logger.debug(log_message)
            elif event.level == LogLevel.INFO:
                self.logger.info(log_message)
            elif event.level == LogLevel.WARNING:
                self.logger.warning(log_message)
            elif event.level == LogLevel.ERROR:
                self.logger.error(log_message)
            elif event.level == LogLevel.CRITICAL:
                self.logger.critical(log_message)
            elif event.level == LogLevel.PERFORMANCE:
                self.logger.info(f"PERFORMANCE: {log_message}")
            elif event.level == LogLevel.VALIDATION:
                self.logger.info(f"VALIDATION: {log_message}")
            elif event.level == LogLevel.DETECTION:
                self.logger.info(f"DETECTION: {log_message}")
            
            # Structured logging
            if self.enable_structured_logging and hasattr(self, 'structured_logger'):
                structured_data = asdict(event)
                structured_data['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event.timestamp))
                
                log_record = logging.LogRecord(
                    name='SRDetectionStructured',
                    level=logging.INFO,
                    pathname='',
                    lineno=0,
                    msg='',
                    args=(),
                    exc_info=None
                )
                log_record.structured_data = structured_data
                self.structured_logger.handle(log_record)
                
        except Exception as e:
            self.logger.error(f"Failed to log detection event: {e}")
    
    def _format_event_message(self, event: SRDetectionEvent) -> str:
        """Format event message for logging."""
        base_message = f"[{event.method_name}] {event.message}"
        
        if event.execution_time > 0:
            base_message += f" (Time: {event.execution_time:.2f}s)"
        
        if event.memory_usage > 0:
            base_message += f" (Memory: {event.memory_usage:.1f}MB)"
        
        if event.data_size > 0:
            base_message += f" (Data: {event.data_size} rows)"
        
        if event.result_count > 0:
            base_message += f" (Results: {event.result_count})"
        
        if event.error_message:
            base_message += f" (Error: {event.error_message})"
        
        return base_message
    
    def log_method_start(self, method_name: str, data_size: int = 0, **kwargs) -> float:
        """Log method start and return start time."""
        start_time = time.time()
        event = SRDetectionEvent(
            event_type='method_start',
            method_name=method_name,
            timestamp=start_time,
            level=LogLevel.DETECTION,
            message=f"Starting {method_name} detection",
            data_size=data_size,
            metadata=kwargs
        )
        self.log_detection_event(event)
        return start_time
    
    def log_method_end(self, method_name: str, start_time: float, result_count: int = 0, 
                      memory_usage: float = 0.0, **kwargs):
        """Log method end with performance metrics."""
        end_time = time.time()
        execution_time = end_time - start_time
        
        event = SRDetectionEvent(
            event_type='method_end',
            method_name=method_name,
            timestamp=end_time,
            level=LogLevel.PERFORMANCE,
            message=f"Completed {method_name} detection",
            execution_time=execution_time,
            memory_usage=memory_usage,
            result_count=result_count,
            metadata=kwargs
        )
        self.log_detection_event(event)
    
    def log_error(self, method_name: str, error: Exception, data_size: int = 0, **kwargs):
        """Log error with context."""
        event = SRDetectionEvent(
            event_type='error',
            method_name=method_name,
            timestamp=time.time(),
            level=LogLevel.ERROR,
            message=f"Error in {method_name}",
            data_size=data_size,
            error_message=str(error),
            metadata={
                'error_type': type(error).__name__,
                'traceback': traceback.format_exc(),
                **kwargs
            }
        )
        self.log_detection_event(event)
    
    def log_validation_result(self, method_name: str, is_valid: bool, issues: List[str] = None, 
                            warnings: List[str] = None, quality_score: float = 0.0, **kwargs):
        """Log validation results."""
        level = LogLevel.VALIDATION if is_valid else LogLevel.ERROR
        message = f"Validation {'passed' if is_valid else 'failed'} for {method_name}"
        
        event = SRDetectionEvent(
            event_type='validation',
            method_name=method_name,
            timestamp=time.time(),
            level=level,
            message=message,
            metadata={
                'is_valid': is_valid,
                'issues': issues or [],
                'warnings': warnings or [],
                'quality_score': quality_score,
                **kwargs
            }
        )
        self.log_detection_event(event)
    
    def log_performance_alert(self, method_name: str, metric: str, value: float, 
                            threshold: float, **kwargs):
        """Log performance alert."""
        event = SRDetectionEvent(
            event_type='performance_alert',
            method_name=method_name,
            timestamp=time.time(),
            level=LogLevel.WARNING,
            message=f"Performance alert: {metric} = {value:.2f} (threshold: {threshold:.2f})",
            metadata={
                'metric': metric,
                'value': value,
                'threshold': threshold,
                **kwargs
            }
        )
        self.log_detection_event(event)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            'total_events': len(self.events),
            'error_counts': self.error_counts,
            'method_performance': {},
            'recent_errors': [],
            'performance_alerts': []
        }
        
        # Calculate method performance metrics
        for method, times in self.performance_metrics.items():
            if times:
                summary['method_performance'][method] = {
                    'avg_execution_time': np.mean(times),
                    'max_execution_time': np.max(times),
                    'min_execution_time': np.min(times),
                    'std_execution_time': np.std(times),
                    'total_calls': len(times),
                    'error_count': self.error_counts.get(method, 0)
                }
        
        # Get recent errors
        recent_errors = [e for e in self.events if e.level in [LogLevel.ERROR, LogLevel.CRITICAL]]
        summary['recent_errors'] = [
            {
                'method': e.method_name,
                'timestamp': e.timestamp,
                'message': e.message,
                'error': e.error_message
            }
            for e in recent_errors[-10:]  # Last 10 errors
        ]
        
        # Get performance alerts
        performance_alerts = [e for e in self.events if e.event_type == 'performance_alert']
        summary['performance_alerts'] = [
            {
                'method': e.method_name,
                'timestamp': e.timestamp,
                'message': e.message,
                'metadata': e.metadata
            }
            for e in performance_alerts[-10:]  # Last 10 alerts
        ]
        
        return summary
    
    def export_events(self, filepath: str, format: str = 'json'):
        """Export events to file."""
        try:
            if format == 'json':
                with open(filepath, 'w') as f:
                    json.dump([asdict(e) for e in self.events], f, indent=2, default=str)
            elif format == 'csv':
                df = pd.DataFrame([asdict(e) for e in self.events])
                df.to_csv(filepath, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Events exported to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to export events: {e}")
    
    def clear_events(self):
        """Clear all stored events."""
        self.events.clear()
        self.performance_metrics.clear()
        self.error_counts.clear()
        self.logger.info("Events cleared")

def sr_logging_decorator(logger: SRLoggingEnhancer):
    """Decorator for automatic SR detection logging."""
    def decorator(func: Callable) -> Callable:
        def wrapper(self, data, *args, **kwargs):
            method_name = func.__name__.replace('_detect_', '').replace('_levels', '')
            start_time = logger.log_method_start(method_name, len(data) if data is not None else 0)
            
            try:
                result = func(self, data, *args, **kwargs)
                
                # Get memory usage if available
                memory_usage = 0.0
                try:
                    import psutil
                    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
                except ImportError:
                    pass
                
                logger.log_method_end(method_name, start_time, len(result) if result else 0, memory_usage)
                return result
                
            except Exception as e:
                logger.log_error(method_name, e, len(data) if data is not None else 0)
                raise
        
        return wrapper
    return decorator

def create_sr_logger(log_file: Optional[str] = None, enable_structured: bool = True) -> SRLoggingEnhancer:
    """Create a configured SR logging enhancer."""
    return SRLoggingEnhancer(log_file=log_file, enable_structured_logging=enable_structured)