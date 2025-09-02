"""
Lifecycle manager for pipeline components.
"""

from typing import Any, Dict, List, Optional, Callable
from enum import Enum
import threading
import time
import logging
from datetime import datetime, timedelta
import signal
import sys
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed


class PipelineState(Enum):
    """Pipeline lifecycle states."""
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class ComponentInfo:
    """Information about a managed component."""
    name: str
    component: Any
    start_time: Optional[datetime] = None
    status: str = "unknown"
    error_count: int = 0
    last_error: Optional[str] = None


class LifecycleManager:
    """
    Manages the lifecycle of pipeline components.
    
    This class handles starting, stopping, pausing, and monitoring
    the state of pipeline operations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LifecycleManager.
        
        Args:
            config: Configuration dictionary for lifecycle management
        """
        self.config = config or {}
        self.state = PipelineState.INITIALIZED
        self.components: List[ComponentInfo] = []
        self.logger = logging.getLogger(__name__)
        self._setup_signal_handlers()
        self._executor = ThreadPoolExecutor(max_workers=self.config.get('max_workers', 4))
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._component_threads = {}
        self._health_check_thread = None
        self._start_time = None
        self._stats = {
            'total_runtime': timedelta(0),
            'start_count': 0,
            'stop_count': 0,
            'error_count': 0
        }
        
    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown")
            self.stop_pipeline()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    def start_pipeline(self) -> bool:
        """
        Start the pipeline execution.
        
        Returns:
            True if pipeline started successfully
        """
        try:
            if self.state in [PipelineState.RUNNING, PipelineState.PAUSED]:
                self.logger.warning(f"Pipeline already in {self.state.value} state")
                return False
                
            self.logger.info("Starting pipeline...")
            self._start_time = datetime.now()
            self.state = PipelineState.RUNNING
            self._stop_event.clear()
            self._pause_event.clear()
            
            # Start all components
            self._start_components()
            
            # Start health monitoring
            self._start_health_monitoring()
            
            self._stats['start_count'] += 1
            self.logger.info("Pipeline started successfully")
            return True
            
        except Exception as e:
            self.state = PipelineState.ERROR
            self.logger.error(f"Failed to start pipeline: {e}")
            self._cleanup_on_error()
            return False
            
    def _start_components(self) -> None:
        """Start all registered components."""
        for component_info in self.components:
            try:
                self._start_component(component_info)
            except Exception as e:
                self.logger.error(f"Failed to start component {component_info.name}: {e}")
                component_info.status = "error"
                component_info.last_error = str(e)
                component_info.error_count += 1
                
    def _start_component(self, component_info: ComponentInfo) -> None:
        """Start a single component."""
        component_info.start_time = datetime.now()
        component_info.status = "starting"
        
        # Check if component has start method
        if hasattr(component_info.component, 'start'):
            if hasattr(component_info.component.start, '__call__'):
                # Run in thread if it's a blocking operation
                thread = threading.Thread(
                    target=self._run_component,
                    args=(component_info,),
                    name=f"component-{component_info.name}"
                )
                thread.daemon = True
                thread.start()
                self._component_threads[component_info.name] = thread
            else:
                component_info.component.start()
                component_info.status = "running"
        else:
            # Component doesn't have start method, assume it's ready
            component_info.status = "running"
            
        self.logger.info(f"Component {component_info.name} started with status: {component_info.status}")
        
    def _run_component(self, component_info: ComponentInfo) -> None:
        """Run a component in a separate thread."""
        try:
            if hasattr(component_info.component, 'run'):
                component_info.component.run()
            elif hasattr(component_info.component, 'start'):
                component_info.component.start()
                
            component_info.status = "running"
            self.logger.info(f"Component {component_info.name} is now running")
            
        except Exception as e:
            component_info.status = "error"
            component_info.last_error = str(e)
            component_info.error_count += 1
            self.logger.error(f"Component {component_info.name} failed: {e}")
            
    def _start_health_monitoring(self) -> None:
        """Start health monitoring thread."""
        self._health_check_thread = threading.Thread(
            target=self._health_monitor_loop,
            name="health-monitor",
            daemon=True
        )
        self._health_check_thread.start()
        
    def _health_monitor_loop(self) -> None:
        """Health monitoring loop."""
        while not self._stop_event.is_set():
            try:
                self._check_component_health()
                time.sleep(self.config.get('health_check_interval', 30))
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                
    def _check_component_health(self) -> None:
        """Check health of all components."""
        for component_info in self.components:
            try:
                if hasattr(component_info.component, 'health_check'):
                    health_status = component_info.component.health_check()
                    if not health_status.get('healthy', True):
                        self.logger.warning(f"Component {component_info.name} health check failed: {health_status}")
                        component_info.status = "unhealthy"
                        
            except Exception as e:
                self.logger.error(f"Health check failed for component {component_info.name}: {e}")
                component_info.status = "error"
                
    def stop_pipeline(self) -> bool:
        """
        Stop the pipeline execution.
        
        Returns:
            True if pipeline stopped successfully
        """
        try:
            if self.state == PipelineState.STOPPED:
                self.logger.info("Pipeline already stopped")
                return True
                
            self.logger.info("Stopping pipeline...")
            self._stop_event.set()
            
            # Stop all components
            self._stop_components()
            
            # Stop health monitoring
            if self._health_check_thread and self._health_check_thread.is_alive():
                self._health_check_thread.join(timeout=5)
                
            # Shutdown thread pool
            self._executor.shutdown(wait=True)
            
            self.state = PipelineState.STOPPED
            self._stats['stop_count'] += 1
            
            if self._start_time:
                runtime = datetime.now() - self._start_time
                self._stats['total_runtime'] += runtime
                self.logger.info(f"Pipeline stopped. Total runtime: {runtime}")
                
            return True
            
        except Exception as e:
            self.state = PipelineState.ERROR
            self.logger.error(f"Failed to stop pipeline: {e}")
            self._cleanup_on_error()
            return False
            
    def _stop_components(self) -> None:
        """Stop all running components."""
        for component_info in self.components:
            try:
                self._stop_component(component_info)
            except Exception as e:
                self.logger.error(f"Failed to stop component {component_info.name}: {e}")
                
    def _stop_component(self, component_info: ComponentInfo) -> None:
        """Stop a single component."""
        if component_info.status in ["running", "starting"]:
            try:
                if hasattr(component_info.component, 'stop'):
                    component_info.component.stop()
                    
                # Wait for component thread to finish
                if component_info.name in self._component_threads:
                    thread = self._component_threads[component_info.name]
                    if thread.is_alive():
                        thread.join(timeout=5)
                        
                component_info.status = "stopped"
                self.logger.info(f"Component {component_info.name} stopped")
                
            except Exception as e:
                component_info.status = "error"
                component_info.last_error = str(e)
                component_info.error_count += 1
                self.logger.error(f"Error stopping component {component_info.name}: {e}")
                
    def pause_pipeline(self) -> bool:
        """
        Pause the pipeline execution.
        
        Returns:
            True if pipeline paused successfully
        """
        if self.state == PipelineState.RUNNING:
            self.logger.info("Pausing pipeline...")
            self._pause_event.set()
            self.state = PipelineState.PAUSED
            
            # Pause components that support it
            for component_info in self.components:
                if hasattr(component_info.component, 'pause'):
                    try:
                        component_info.component.pause()
                        self.logger.info(f"Component {component_info.name} paused")
                    except Exception as e:
                        self.logger.error(f"Failed to pause component {component_info.name}: {e}")
                        
            self.logger.info("Pipeline paused")
            return True
        return False
        
    def resume_pipeline(self) -> bool:
        """
        Resume the pipeline execution.
        
        Returns:
            True if pipeline resumed successfully
        """
        if self.state == PipelineState.PAUSED:
            self.logger.info("Resuming pipeline...")
            self._pause_event.clear()
            self.state = PipelineState.RUNNING
            
            # Resume components that support it
            for component_info in self.components:
                if hasattr(component_info.component, 'resume'):
                    try:
                        component_info.component.resume()
                        self.logger.info(f"Component {component_info.name} resumed")
                    except Exception as e:
                        self.logger.error(f"Failed to resume component {component_info.name}: {e}")
                        
            self.logger.info("Pipeline resumed")
            return True
        return False
        
    def get_state(self) -> PipelineState:
        """
        Get current pipeline state.
        
        Returns:
            Current pipeline state
        """
        return self.state
        
    def add_component(self, component: Any, name: Optional[str] = None) -> None:
        """
        Add a component to the lifecycle manager.
        
        Args:
            component: Component to manage
            name: Optional name for the component
        """
        if name is None:
            name = component.__class__.__name__
            
        # Check if component with this name already exists
        existing_names = [c.name for c in self.components]
        if name in existing_names:
            counter = 1
            while f"{name}_{counter}" in existing_names:
                counter += 1
            name = f"{name}_{counter}"
            
        component_info = ComponentInfo(
            name=name,
            component=component,
            status="registered"
        )
        
        self.components.append(component_info)
        self.logger.info(f"Added component: {name}")
        
    def remove_component(self, name: str) -> bool:
        """
        Remove a component from the lifecycle manager.
        
        Args:
            name: Name of the component to remove
            
        Returns:
            True if component was removed successfully
        """
        for i, component_info in enumerate(self.components):
            if component_info.name == name:
                if component_info.status in ["running", "starting"]:
                    self._stop_component(component_info)
                    
                del self.components[i]
                self.logger.info(f"Removed component: {name}")
                return True
        return False
        
    def get_component_status(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get status information for a specific component.
        
        Args:
            name: Component name
            
        Returns:
            Component status information or None if not found
        """
        for component_info in self.components:
            if component_info.name == name:
                return {
                    'name': component_info.name,
                    'status': component_info.status,
                    'start_time': component_info.start_time,
                    'error_count': component_info.error_count,
                    'last_error': component_info.last_error
                }
        return None
        
    def get_all_component_status(self) -> List[Dict[str, Any]]:
        """
        Get status information for all components.
        
        Returns:
            List of component status information
        """
        return [self.get_component_status(c.name) for c in self.components]
        
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.
        
        Returns:
            Pipeline statistics
        """
        current_runtime = timedelta(0)
        if self._start_time and self.state == PipelineState.RUNNING:
            current_runtime = datetime.now() - self._start_time
            
        return {
            'state': self.state.value,
            'total_runtime': str(self._stats['total_runtime'] + current_runtime),
            'start_count': self._stats['start_count'],
            'stop_count': self._stats['stop_count'],
            'error_count': self._stats['error_count'],
            'component_count': len(self.components),
            'running_components': len([c for c in self.components if c.status == "running"]),
            'error_components': len([c for c in self.components if c.status == "error"])
        }
        
    def _cleanup_on_error(self) -> None:
        """Clean up resources when an error occurs."""
        self._stats['error_count'] += 1
        self._stop_event.set()
        
        # Try to stop components gracefully
        try:
            self._stop_components()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            
        # Shutdown thread pool
        try:
            self._executor.shutdown(wait=False)
        except Exception as e:
            self.logger.error(f"Error shutting down thread pool: {e}")
            
    def is_running(self) -> bool:
        """Check if pipeline is currently running."""
        return self.state == PipelineState.RUNNING
        
    def is_paused(self) -> bool:
        """Check if pipeline is currently paused."""
        return self.state == PipelineState.PAUSED
        
    def is_stopped(self) -> bool:
        """Check if pipeline is currently stopped."""
        return self.state == PipelineState.STOPPED
        
    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for pipeline to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if pipeline completed, False if timeout
        """
        start_time = time.time()
        while self.state not in [PipelineState.STOPPED, PipelineState.COMPLETED, PipelineState.ERROR]:
            if timeout and (time.time() - start_time) > timeout:
                return False
            time.sleep(0.1)
        return True


