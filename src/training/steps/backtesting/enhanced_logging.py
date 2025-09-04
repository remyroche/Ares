#!/usr/bin/env python3
"""Enhanced Logging System for Backtesting Pipeline.

This module provides comprehensive logging with emojis, progress tracking,
quality assessment, and detailed error reporting for the backtesting pipeline.
"""

import logging
import time
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import traceback
import psutil
import threading
from contextlib import contextmanager

from src.utils.common_operations import (
    format_datetime, get_current_datetime, safe_file_exists, 
    ensure_directory, safe_json_dump, safe_json_load
)

class BacktestingLogger:
    """Enhanced logger for backtesting pipeline with comprehensive monitoring."""
    
    def __init__(self, name: str, log_dir: str = "log", enable_console: bool = True):
        self.name = name
        self.log_dir = Path(log_dir)
        self.enable_console = enable_console
        
        # Ensure log directory exists
        ensure_directory(self.log_dir)
        
        # Initialize logger
        self.logger = logging.getLogger(f"backtesting.{name}")
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create file handler
        timestamp = format_datetime(get_current_datetime(), '%Y%m%d_%H%M%S')
        log_file = self.log_dir / f"backtesting_{name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Create console handler if enabled
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File formatter (more detailed)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Initialize tracking
        self.start_time = time.time()
        self.step_times = {}
        self.quality_flags = []
        self.errors = []
        self.warnings = []
        self.progress_data = {}
        self.performance_metrics = {}
        
        # Performance monitoring
        self.monitor_thread = None
        self.monitoring = False
        
        # Log initialization
        self.logger.info("🚀 Enhanced Backtesting Logger Initialized")
        self.logger.info(f"📁 Log file: {log_file}")
        self.logger.info(f"🖥️ Console output: {'Enabled' if self.enable_console else 'Disabled'}")
    
    def start_performance_monitoring(self, interval: float = 5.0):
        """Start performance monitoring in background thread."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_performance,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info(f"📊 Performance monitoring started (interval: {interval}s)")
    
    def stop_performance_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        self.logger.info("📊 Performance monitoring stopped")
    
    def _monitor_performance(self, interval: float):
        """Background performance monitoring."""
        while self.monitoring:
            try:
                # Get system metrics
                process = psutil.Process()
                memory_info = process.memory_info()
                cpu_percent = process.cpu_percent()
                
                # Store metrics
                timestamp = time.time()
                self.performance_metrics[timestamp] = {
                    'memory_mb': memory_info.rss / 1024 / 1024,
                    'cpu_percent': cpu_percent,
                    'elapsed_time': timestamp - self.start_time
                }
                
                # Log if memory usage is high
                if memory_info.rss / 1024 / 1024 > 1000:  # > 1GB
                    self.logger.warning(f"⚠️ High memory usage: {memory_info.rss / 1024 / 1024:.1f} MB")
                
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"❌ Performance monitoring error: {e}")
                break
    
    @contextmanager
    def step_timer(self, step_name: str):
        """Context manager for timing steps."""
        start_time = time.time()
        self.logger.info(f"🔄 Starting step: {step_name}")
        
        try:
            yield
            elapsed = time.time() - start_time
            self.step_times[step_name] = elapsed
            self.logger.info(f"✅ Step completed: {step_name} ({elapsed:.2f}s)")
        except Exception as e:
            elapsed = time.time() - start_time
            self.step_times[step_name] = elapsed
            self.logger.error(f"❌ Step failed: {step_name} ({elapsed:.2f}s) - {e}")
            raise
    
    def log_progress(self, step: str, progress: float, message: str = ""):
        """Log progress with visual indicator."""
        progress_bar = self._create_progress_bar(progress)
        if message:
            self.logger.info(f"📈 {step}: {progress_bar} {progress:.1f}% - {message}")
        else:
            self.logger.info(f"📈 {step}: {progress_bar} {progress:.1f}%")
        
        # Store progress data
        self.progress_data[step] = {
            'progress': progress,
            'message': message,
            'timestamp': time.time()
        }
    
    def _create_progress_bar(self, progress: float, width: int = 20) -> str:
        """Create a visual progress bar."""
        filled = int(width * progress / 100)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"
    
    def log_quality_flag(self, flag_type: str, message: str, severity: str = "WARNING"):
        """Log quality flags for issue detection."""
        flag_data = {
            'type': flag_type,
            'message': message,
            'severity': severity,
            'timestamp': time.time()
        }
        self.quality_flags.append(flag_data)
        
        emoji = "⚠️" if severity == "WARNING" else "❌" if severity == "ERROR" else "ℹ️"
        self.logger.warning(f"{emoji} Quality Flag [{flag_type}]: {message}")
    
    def log_error(self, error: Exception, context: str = ""):
        """Log errors with detailed context."""
        error_data = {
            'type': type(error).__name__,
            'message': str(error),
            'context': context,
            'timestamp': time.time(),
            'traceback': traceback.format_exc()
        }
        self.errors.append(error_data)
        
        self.logger.error(f"❌ Error in {context}: {error}")
        self.logger.debug(f"📋 Error traceback: {traceback.format_exc()}")
    
    def log_warning(self, message: str, context: str = ""):
        """Log warnings with context."""
        warning_data = {
            'message': message,
            'context': context,
            'timestamp': time.time()
        }
        self.warnings.append(warning_data)
        
        self.logger.warning(f"⚠️ Warning in {context}: {message}")
    
    def log_success(self, message: str, context: str = ""):
        """Log success messages."""
        self.logger.info(f"✅ Success in {context}: {message}")
    
    def log_info(self, message: str, context: str = ""):
        """Log info messages."""
        if context:
            self.logger.info(f"ℹ️ {context}: {message}")
        else:
            self.logger.info(f"ℹ️ {message}")
    
    def log_debug(self, message: str, context: str = ""):
        """Log debug messages."""
        if context:
            self.logger.debug(f"🔍 {context}: {message}")
        else:
            self.logger.debug(f"🔍 {message}")
    
    def log_data_quality(self, data_info: Dict[str, Any]):
        """Log data quality assessment."""
        self.logger.info("📊 Data Quality Assessment:")
        
        for key, value in data_info.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"   • {key}: {value:,}")
            else:
                self.logger.info(f"   • {key}: {value}")
        
        # Check for quality issues
        if data_info.get('missing_percentage', 0) > 5:
            self.log_quality_flag(
                "DATA_QUALITY", 
                f"High missing data percentage: {data_info.get('missing_percentage', 0):.1f}%",
                "WARNING"
            )
        
        if data_info.get('duplicate_count', 0) > 0:
            self.log_quality_flag(
                "DATA_QUALITY", 
                f"Duplicate records found: {data_info.get('duplicate_count', 0)}",
                "WARNING"
            )
    
    def log_validation_result(self, step: str, passed: bool, details: Dict[str, Any]):
        """Log validation results."""
        if passed:
            self.logger.info(f"✅ Validation passed: {step}")
        else:
            self.logger.error(f"❌ Validation failed: {step}")
            self.log_quality_flag("VALIDATION", f"Validation failed for {step}", "ERROR")
        
        # Log details
        for key, value in details.items():
            self.logger.info(f"   • {key}: {value}")
    
    def log_performance_summary(self):
        """Log performance summary."""
        total_time = time.time() - self.start_time
        
        self.logger.info("📊 Performance Summary:")
        self.logger.info(f"   • Total execution time: {total_time:.2f}s")
        self.logger.info(f"   • Quality flags: {len(self.quality_flags)}")
        self.logger.info(f"   • Errors: {len(self.errors)}")
        self.logger.info(f"   • Warnings: {len(self.warnings)}")
        
        # Log step times
        if self.step_times:
            self.logger.info("   • Step execution times:")
            for step, time_taken in self.step_times.items():
                self.logger.info(f"     - {step}: {time_taken:.2f}s")
        
        # Log memory usage
        if self.performance_metrics:
            latest_metrics = max(self.performance_metrics.values(), key=lambda x: x['elapsed_time'])
            self.logger.info(f"   • Peak memory usage: {latest_metrics['memory_mb']:.1f} MB")
            self.logger.info(f"   • Peak CPU usage: {latest_metrics['cpu_percent']:.1f}%")
    
    def generate_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive report."""
        total_time = time.time() - self.start_time
        
        report = {
            'execution_summary': {
                'total_time_seconds': total_time,
                'start_time': format_datetime(get_current_datetime(), '%Y-%m-%d %H:%M:%S'),
                'end_time': format_datetime(get_current_datetime(), '%Y-%m-%d %H:%M:%S'),
                'logger_name': self.name
            },
            'step_times': self.step_times,
            'progress_data': self.progress_data,
            'quality_flags': self.quality_flags,
            'errors': self.errors,
            'warnings': self.warnings,
            'performance_metrics': self.performance_metrics,
            'quality_assessment': self._assess_overall_quality()
        }
        
        if output_file:
            safe_json_dump(report, output_file, indent=2)
            self.logger.info(f"📋 Report saved to: {output_file}")
        
        return report
    
    def _assess_overall_quality(self) -> Dict[str, Any]:
        """Assess overall quality of the execution."""
        error_count = len(self.errors)
        warning_count = len(self.warnings)
        quality_flag_count = len(self.quality_flags)
        
        # Determine quality level
        if error_count > 0:
            quality_level = "POOR"
        elif quality_flag_count > 5 or warning_count > 10:
            quality_level = "FAIR"
        elif quality_flag_count > 0 or warning_count > 0:
            quality_level = "GOOD"
        else:
            quality_level = "EXCELLENT"
        
        return {
            'quality_level': quality_level,
            'error_count': error_count,
            'warning_count': warning_count,
            'quality_flag_count': quality_flag_count,
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on quality assessment."""
        recommendations = []
        
        if len(self.errors) > 0:
            recommendations.append("🔧 Address all errors before proceeding")
        
        if len(self.quality_flags) > 5:
            recommendations.append("⚠️ Review quality flags and consider data preprocessing")
        
        if len(self.warnings) > 10:
            recommendations.append("📊 Review warnings and optimize configuration")
        
        # Check for performance issues
        if self.performance_metrics:
            latest_metrics = max(self.performance_metrics.values(), key=lambda x: x['elapsed_time'])
            if latest_metrics['memory_mb'] > 2000:  # > 2GB
                recommendations.append("💾 Consider optimizing memory usage")
            
            if latest_metrics['cpu_percent'] > 90:
                recommendations.append("⚡ Consider optimizing CPU usage")
        
        if not recommendations:
            recommendations.append("✅ No issues detected - execution quality is excellent")
        
        return recommendations
    
    def cleanup(self):
        """Cleanup resources."""
        self.stop_performance_monitoring()
        self.logger.info("🧹 Backtesting logger cleanup completed")

# Global logger instance
_global_logger = None

def get_backtesting_logger(name: str = "pipeline", log_dir: str = "log") -> BacktestingLogger:
    """Get or create global backtesting logger."""
    global _global_logger
    if _global_logger is None:
        _global_logger = BacktestingLogger(name, log_dir)
    return _global_logger

def cleanup_global_logger():
    """Cleanup global logger."""
    global _global_logger
    if _global_logger:
        _global_logger.cleanup()
        _global_logger = None