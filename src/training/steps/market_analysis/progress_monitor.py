#!/usr/bin/env python3
"""
Progress Monitor for Market Analysis Pipeline

This module provides real-time progress monitoring with visual indicators
and detailed status updates for the market analysis pipeline.
"""

import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

try:
    from .enhanced_logging_metrics import enhanced_logger
except ImportError:
    # Fallback for when imported directly
    import logging
    class FallbackLogger:
        def __init__(self):
            self.logger = logging.getLogger("enhanced_logger")
    enhanced_logger = FallbackLogger()


@dataclass
class ProgressUpdate:
    """Represents a progress update for a pipeline step."""
    step_name: str
    progress: float  # 0.0 to 1.0
    message: str
    timestamp: datetime
    status: str  # 'running', 'completed', 'failed', 'warning'
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class ProgressMonitor:
    """
    Real-time progress monitor for market analysis pipeline with visual indicators.
    """
    
    def __init__(self, update_interval: float = 2.0):
        """Initialize the progress monitor."""
        self.update_interval = update_interval
        self.steps: Dict[str, ProgressUpdate] = {}
        self.start_time = None
        self.monitoring = False
        self.monitor_thread = None
        
        # Visual indicators
        self.spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.spinner_index = 0
        
        # Progress bar components
        self.progress_bar_length = 30
        self.progress_chars = {
            'filled': '█',
            'empty': '░',
            'partial': '▄'
        }
    
    def start_monitoring(self):
        """Start the progress monitoring thread."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.start_time = datetime.now()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        enhanced_logger.logger.info("📊 Progress monitoring started")
    
    def stop_monitoring(self):
        """Stop the progress monitoring thread."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        enhanced_logger.logger.info("📊 Progress monitoring stopped")
    
    def update_step_progress(self, step_name: str, progress: float, message: str = "", 
                           status: str = "running", details: Dict[str, Any] = None):
        """Update progress for a specific step."""
        self.steps[step_name] = ProgressUpdate(
            step_name=step_name,
            progress=max(0.0, min(1.0, progress)),
            message=message,
            timestamp=datetime.now(),
            status=status,
            details=details or {}
        )
    
    def complete_step(self, step_name: str, success: bool = True, message: str = ""):
        """Mark a step as completed."""
        status = "completed" if success else "failed"
        if not message:
            message = "Completed successfully" if success else "Failed"
        
        self.update_step_progress(step_name, 1.0, message, status)
    
    def _monitor_loop(self):
        """Main monitoring loop that runs in a separate thread."""
        while self.monitoring:
            try:
                self._display_progress()
                time.sleep(self.update_interval)
            except Exception as e:
                enhanced_logger.logger.warning(f"⚠️ Progress monitor error: {e}")
                time.sleep(self.update_interval)
    
    def _display_progress(self):
        """Display current progress with visual indicators."""
        if not self.steps:
            return
        
        # Clear screen (works in most terminals)
        print("\033[2J\033[H", end="")
        
        # Display header
        elapsed = datetime.now() - self.start_time if self.start_time else timedelta(0)
        print(f"🚀 Market Analysis Pipeline Progress - {elapsed.total_seconds():.0f}s elapsed")
        print("=" * 80)
        
        # Display each step
        for step_name, step in self.steps.items():
            self._display_step_progress(step)
        
        # Display summary
        self._display_summary()
        
        print("=" * 80)
        print("Press Ctrl+C to stop monitoring (pipeline will continue)")
    
    def _display_step_progress(self, step: ProgressUpdate):
        """Display progress for a single step."""
        # Get status emoji
        status_emoji = {
            'running': self.spinner_chars[self.spinner_index % len(self.spinner_chars)],
            'completed': '✅',
            'failed': '❌',
            'warning': '⚠️'
        }.get(step.status, '❓')
        
        # Create progress bar
        progress_bar = self._create_progress_bar(step.progress)
        
        # Format progress percentage
        progress_pct = f"{step.progress * 100:5.1f}%"
        
        # Format timestamp
        time_str = step.timestamp.strftime("%H:%M:%S")
        
        # Display step info
        print(f"{status_emoji} {step.step_name:<25} {progress_bar} {progress_pct} [{time_str}]")
        
        if step.message:
            print(f"   📝 {step.message}")
        
        # Display details if any
        if step.details:
            for key, value in step.details.items():
                print(f"   📊 {key}: {value}")
        
        print()
    
    def _create_progress_bar(self, progress: float) -> str:
        """Create a visual progress bar."""
        filled_length = int(progress * self.progress_bar_length)
        empty_length = self.progress_bar_length - filled_length
        
        bar = (self.progress_chars['filled'] * filled_length + 
               self.progress_chars['empty'] * empty_length)
        
        return f"[{bar}]"
    
    def _display_summary(self):
        """Display pipeline summary."""
        total_steps = len(self.steps)
        completed_steps = sum(1 for step in self.steps.values() if step.status == "completed")
        failed_steps = sum(1 for step in self.steps.values() if step.status == "failed")
        running_steps = sum(1 for step in self.steps.values() if step.status == "running")
        
        overall_progress = sum(step.progress for step in self.steps.values()) / max(1, total_steps)
        
        print(f"📈 Overall Progress: {overall_progress * 100:.1f}%")
        print(f"✅ Completed: {completed_steps} | ❌ Failed: {failed_steps} | 🔄 Running: {running_steps}")
        
        # Update spinner index
        self.spinner_index += 1


# Global progress monitor instance
progress_monitor = ProgressMonitor()


def start_progress_monitoring():
    """Start the global progress monitor."""
    progress_monitor.start_monitoring()


def stop_progress_monitoring():
    """Stop the global progress monitor."""
    progress_monitor.stop_monitoring()


def update_progress(step_name: str, progress: float, message: str = "", 
                   status: str = "running", details: Dict[str, Any] = None):
    """Update progress for a step using the global monitor."""
    progress_monitor.update_step_progress(step_name, progress, message, status, details)


def complete_step(step_name: str, success: bool = True, message: str = ""):
    """Complete a step using the global monitor."""
    progress_monitor.complete_step(step_name, success, message)


# Context manager for automatic progress monitoring
class ProgressContext:
    """Context manager for automatic progress monitoring of a step."""
    
    def __init__(self, step_name: str, total_work: int = 100):
        self.step_name = step_name
        self.total_work = total_work
        self.current_work = 0
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        update_progress(self.step_name, 0.0, "Starting...", "running")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            complete_step(self.step_name, True, "Completed successfully")
        else:
            complete_step(self.step_name, False, f"Failed: {exc_val}")
    
    def update(self, work_done: int, message: str = ""):
        """Update progress within the context."""
        self.current_work = min(self.current_work + work_done, self.total_work)
        progress = self.current_work / self.total_work
        
        if not message:
            elapsed = time.time() - self.start_time if self.start_time else 0
            message = f"Progress: {self.current_work}/{self.total_work} ({elapsed:.1f}s)"
        
        update_progress(self.step_name, progress, message, "running")
    
    def set_progress(self, progress: float, message: str = ""):
        """Set absolute progress (0.0 to 1.0)."""
        self.current_work = int(progress * self.total_work)
        update_progress(self.step_name, progress, message, "running")


# Decorator for automatic progress monitoring
def monitor_progress(step_name: str, total_work: int = 100):
    """Decorator to automatically monitor progress of a function."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with ProgressContext(step_name, total_work) as progress:
                # Pass progress object to function if it accepts it
                import inspect
                sig = inspect.signature(func)
                if 'progress' in sig.parameters:
                    kwargs['progress'] = progress
                
                return func(*args, **kwargs)
        return wrapper
    return decorator