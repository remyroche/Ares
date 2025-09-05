from typing import Dict
from typing import Any
from typing import Dict, List, Optional, Union, Any, Tuple
'\nProgress Monitor for Market Analysis Pipeline\n\nThis module provides real-time progress monitoring with visual indicators\nand detailed status updates for the market analysis pipeline.\n'
import time
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
try:
    from .enhanced_logging_metrics import enhanced_logger
except ImportError:
    import logging

    class FallbackLogger:

        def __init__(self) -> None:
            self.logger = logging.getLogger('enhanced_logger')
    enhanced_logger = FallbackLogger()

@dataclass
class ProgressUpdate:
    """Represents a progress update for a pipeline step."""
    step_name: str
    progress: float
    message: str
    timestamp: datetime
    status: str
    details: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.details is None:
            self.details = {}

class ProgressMonitor:
    """
    Real-time progress monitor for market analysis pipeline with visual indicators.
    """

    def __init__(self, update_interval: float=2.0) -> None:
        """Initialize the progress monitor."""
        self.update_interval = update_interval
        self.steps: Dict[str, ProgressUpdate] = {}
        self.start_time = None
        self.monitoring = False
        self.monitor_thread = None
        self.spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.spinner_index = 0
        self.progress_bar_length = 30
        self.progress_chars = {'filled': '█', 'empty': '░', 'partial': '▄'}

    def start_monitoring(self) -> None:
        """Start the progress monitoring thread."""
        if self.monitoring:
            return
        self.monitoring = True
        self.start_time = datetime.now()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        enhanced_logger.logger.info('📊 Progress monitoring started')

    def stop_monitoring(self) -> None:
        """Stop the progress monitoring thread."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        enhanced_logger.logger.info('📊 Progress monitoring stopped')

    def update_step_progress(self, step_name: str, progress: float, message: str='', status: str='running', details: Dict[str, Any]=None, step_number: int=None, total_steps: int=None) -> None:
        """Update progress for a specific step."""
        step_details = details or {}
        if step_number is not None and total_steps is not None:
            step_details['step_number'] = step_number
            step_details['total_steps'] = total_steps
        self.steps[step_name] = ProgressUpdate(step_name=step_name, progress=max(0.0, min(1.0, progress)), message=message, timestamp=datetime.now(), status=status, details=step_details)

    def complete_step(self, step_name: str, success: bool=True, message: str='') -> None:
        """Mark a step as completed."""
        status = 'completed' if success else 'failed'
        if not message:
            message = 'Completed successfully' if success else 'Failed'
        self.update_step_progress(step_name, 1.0, message, status)

    def _monitor_loop(self) -> None:
        """Main monitoring loop that runs in a separate thread."""
        while self.monitoring:
            try:
                self._display_progress()
                time.sleep(self.update_interval)
            except Exception as e:
                enhanced_logger.logger.warning(f'⚠️ Progress monitor error: {e}')
                time.sleep(self.update_interval)

    def _display_progress(self) -> None:
        """Display current progress with visual indicators."""
        if not self.steps:
            return
        print('\x1b[2J\x1b[H', end='')
        elapsed = datetime.now() - self.start_time if self.start_time else timedelta(0)
        print(f'🚀 Market Analysis Pipeline Progress - {elapsed.total_seconds():.0f}s elapsed')
        print('=' * 80)
        for step_name, step in self.steps.items():
            self._display_step_progress(step)
        self._display_summary()
        print('=' * 80)
        print('Press Ctrl+C to stop monitoring (pipeline will continue)')

    def _display_step_progress(self, step: ProgressUpdate) -> None:
        """Display progress for a single step."""
        status_emoji = {'running': self.spinner_chars[self.spinner_index % len(self.spinner_chars)], 'completed': '✓', 'failed': '❌', 'warning': '⚠️'}.get(step.status, '❓')
        progress_bar = self._create_progress_bar(step.progress)
        progress_pct = f'{step.progress * 100:5.1f}%'
        time_str = step.timestamp.strftime('%H:%M:%S')
        step_display_name = step.step_name
        if hasattr(step, 'step_number') and hasattr(step, 'total_steps'):
            step_display_name = f'STEP {step.step_number}/{step.total_steps}: {step.step_name}'
        elif 'step_number' in step.details and 'total_steps' in step.details:
            step_display_name = f"STEP {step.details['step_number']}/{step.details['total_steps']}: {step.step_name}"
        print(f'{status_emoji} {step_display_name:<35} {progress_bar} {progress_pct} [{time_str}]')
        if step.message:
            print(f'   ℹ️ {step.message}')
        if step.details:
            for key, value in step.details.items():
                if key not in ['step_number', 'total_steps']:
                    print(f'   📊 {key}: {value}')
        print()

    def _create_progress_bar(self, progress: float) -> str:
        """Create a visual progress bar."""
        filled_length = int(progress * self.progress_bar_length)
        empty_length = self.progress_bar_length - filled_length
        bar = self.progress_chars['filled'] * filled_length + self.progress_chars['empty'] * empty_length
        return f'[{bar}]'

    def _display_summary(self) -> None:
        """Display pipeline summary."""
        total_steps = len(self.steps)
        completed_steps = sum((1 for step in self.steps.values() if step.status == 'completed'))
        failed_steps = sum((1 for step in self.steps.values() if step.status == 'failed'))
        running_steps = sum((1 for step in self.steps.values() if step.status == 'running'))
        overall_progress = sum((step.progress for step in self.steps.values())) / max(1, total_steps)
        print(f'📈 Overall Progress: {overall_progress * 100:.1f}%')
        print(f'✅ Completed: {completed_steps} | ❌ Failed: {failed_steps} | 🔄 Running: {running_steps}')
        self.spinner_index += 1
progress_monitor = ProgressMonitor()

def start_progress_monitoring() -> None:
    """Start the global progress monitor."""
    progress_monitor.start_monitoring()

def stop_progress_monitoring() -> None:
    """Stop the global progress monitor."""
    progress_monitor.stop_monitoring()

def update_progress(step_name: str, progress: float, message: str='', status: str='running', details: Dict[str, Any]=None) -> None:
    """Update progress for a step using the global monitor."""
    progress_monitor.update_step_progress(step_name, progress, message, status, details)

def complete_step(step_name: str, success: bool=True, message: str='') -> None:
    """Complete a step using the global monitor."""
    progress_monitor.complete_step(step_name, success, message)

class ProgressContext:
    """Context manager for automatic progress monitoring of a step."""

    def __init__(self, step_name: str, total_work: int=100) -> None:
        self.step_name = step_name
        self.total_work = total_work
        self.current_work = 0
        self.start_time = None

    def __enter__(self) -> None:
        self.start_time = time.time()
        update_progress(self.step_name, 0.0, 'Starting...', 'running')
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is None:
            complete_step(self.step_name, True, 'Completed successfully')
        else:
            complete_step(self.step_name, False, f'Failed: {exc_val}')

    def update(self, work_done: int, message: str='') -> None:
        """Update progress within the context."""
        self.current_work = min(self.current_work + work_done, self.total_work)
        progress = self.current_work / self.total_work
        if not message:
            elapsed = time.time() - self.start_time if self.start_time else 0
            message = f'Progress: {self.current_work}/{self.total_work} ({elapsed:.1f}s)'
        update_progress(self.step_name, progress, message, 'running')

    def set_progress(self, progress: float, message: str='') -> None:
        """Set absolute progress (0.0 to 1.0)."""
        self.current_work = int(progress * self.total_work)
        update_progress(self.step_name, progress, message, 'running')

def monitor_progress(step_name: str, total_work: int=100) -> None:
    """Decorator to automatically monitor progress of a function."""

    def decorator(func: Callable) -> None:

        def wrapper(*args, **kwargs) -> None:
            with ProgressContext(step_name, total_work) as progress:
                import inspect
                sig = inspect.signature(func)
                if 'progress' in sig.parameters:
                    kwargs['progress'] = progress
                return func(*args, **kwargs)
        return wrapper
    return decorator