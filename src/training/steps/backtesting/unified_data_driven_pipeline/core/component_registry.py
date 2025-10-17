"""
Component Registry for Backtesting Pipeline

This module provides a centralized registry for managing all ModularComponent
instances in the backtesting pipeline. It includes component registration,
discovery, dependency resolution, lifecycle management, and performance monitoring.

Key Features:
- Component registration and discovery
- Dependency resolution and management
- Component lifecycle management
- Performance monitoring aggregation
- Health status reporting
- Strategy versioning and tracking
- Backtesting-specific component management
"""

import time
import logging
import threading
from typing import Any, Dict, List, Optional, Set, Type, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
from pathlib import Path

from .modular_architecture import ModularComponent, ValidationResult, ErrorInfo, ErrorSeverity, ErrorCategory


class ComponentStatus(Enum):
    """Component status enumeration."""
    REGISTERED = "registered"
    INITIALIZED = "initialized"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    CLEANUP = "cleanup"


class ComponentType(Enum):
    """Component type enumeration."""
    BACKTESTING_ENGINE = "backtesting_engine"
    MONTE_CARLO_ENGINE = "monte_carlo_engine"
    RISK_MANAGEMENT = "risk_management"
    PORTFOLIO_MANAGER = "portfolio_manager"
    STRATEGY_OPTIMIZER = "strategy_optimizer"
    PERFORMANCE_ANALYZER = "performance_analyzer"
    REPORTING_ENGINE = "reporting_engine"
    DATA_LOADER = "data_loader"
    FEATURE_GENERATOR = "feature_generator"
    SIGNAL_GENERATOR = "signal_generator"


@dataclass
class ComponentInfo:
    """Information about a registered component."""
    name: str
    component_type: ComponentType
    component_class: Type[ModularComponent]
    instance: Optional[ModularComponent]
    status: ComponentStatus
    dependencies: List[str]
    dependents: List[str]
    created_at: float
    last_updated: float
    performance_stats: Dict[str, Any] = field(default_factory=dict)
    health_status: str = "unknown"
    error_count: int = 0
    warning_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DependencyGraph:
    """Dependency graph for components."""
    nodes: Set[str] = field(default_factory=set)
    edges: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    reverse_edges: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    
    def add_node(self, node: str) -> None:
        """Add a node to the graph."""
        self.nodes.add(node)
        if node not in self.edges:
            self.edges[node] = set()
        if node not in self.reverse_edges:
            self.reverse_edges[node] = set()
    
    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add an edge to the graph."""
        self.add_node(from_node)
        self.add_node(to_node)
        self.edges[from_node].add(to_node)
        self.reverse_edges[to_node].add(from_node)
    
    def get_dependencies(self, node: str) -> Set[str]:
        """Get all dependencies of a node."""
        return self.edges.get(node, set())
    
    def get_dependents(self, node: str) -> Set[str]:
        """Get all dependents of a node."""
        return self.reverse_edges.get(node, set())
    
    def has_cycle(self) -> bool:
        """Check if the graph has cycles."""
        visited = set()
        rec_stack = set()
        
        def dfs(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.edges.get(node, set()):
                if dfs(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in self.nodes:
            if node not in visited:
                if dfs(node):
                    return True
        
        return False
    
    def topological_sort(self) -> List[str]:
        """Get topological sort of the graph."""
        in_degree = {node: 0 for node in self.nodes}
        
        for node in self.nodes:
            for neighbor in self.edges.get(node, set()):
                in_degree[neighbor] += 1
        
        queue = deque([node for node in self.nodes if in_degree[node] == 0])
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            for neighbor in self.edges.get(node, set()):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result


class BacktestingComponentRegistry:
    """Registry for managing backtesting components."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._components: Dict[str, ComponentInfo] = {}
        self._dependency_graph = DependencyGraph()
        self._lock = threading.RLock()
        self._performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Registry configuration
        self._max_components = 100
        self._performance_history_size = 1000
        self._health_check_interval = 60.0  # seconds
        self._last_health_check = 0.0
        
        # Backtesting-specific settings
        self._enable_strategy_versioning = True
        self._enable_performance_monitoring = True
        self._enable_health_monitoring = True
        self._enable_dependency_tracking = True
    
    def register_component(
        self,
        name: str,
        component_type: ComponentType,
        component_class: Type[ModularComponent],
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a component in the registry.
        
        Args:
            name: Unique name for the component
            component_type: Type of the component
            component_class: Component class
            dependencies: List of component dependencies
            metadata: Additional metadata
            
        Returns:
            True if registration successful, False otherwise
        """
        with self._lock:
            try:
                # Check if component already exists
                if name in self._components:
                    self.logger.warning(f"Component {name} already registered")
                    return False
                
                # Check registry capacity
                if len(self._components) >= self._max_components:
                    self.logger.error(f"Registry capacity exceeded ({self._max_components})")
                    return False
                
                # Validate dependencies
                if dependencies:
                    for dep in dependencies:
                        if dep not in self._components:
                            self.logger.error(f"Dependency {dep} not found for component {name}")
                            return False
                
                # Create component info
                component_info = ComponentInfo(
                    name=name,
                    component_type=component_type,
                    component_class=component_class,
                    instance=None,
                    status=ComponentStatus.REGISTERED,
                    dependencies=dependencies or [],
                    dependents=[],
                    created_at=time.time(),
                    last_updated=time.time(),
                    metadata=metadata or {}
                )
                
                # Register component
                self._components[name] = component_info
                
                # Update dependency graph
                if self._enable_dependency_tracking and dependencies:
                    for dep in dependencies:
                        self._dependency_graph.add_edge(dep, name)
                        self._components[dep].dependents.append(name)
                
                self.logger.info(f"Component {name} registered successfully")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to register component {name}: {e}")
                return False
    
    def unregister_component(self, name: str) -> bool:
        """
        Unregister a component from the registry.
        
        Args:
            name: Name of the component to unregister
            
        Returns:
            True if unregistration successful, False otherwise
        """
        with self._lock:
            try:
                if name not in self._components:
                    self.logger.warning(f"Component {name} not found")
                    return False
                
                component_info = self._components[name]
                
                # Check if component has dependents
                if component_info.dependents:
                    self.logger.error(f"Cannot unregister component {name}: has dependents {component_info.dependents}")
                    return False
                
                # Cleanup component if initialized
                if component_info.instance and component_info.status in [ComponentStatus.INITIALIZED, ComponentStatus.RUNNING]:
                    try:
                        component_info.instance.cleanup()
                    except Exception as e:
                        self.logger.warning(f"Error during component cleanup: {e}")
                
                # Remove from dependency graph
                if self._enable_dependency_tracking:
                    for dep in component_info.dependencies:
                        if dep in self._components:
                            self._components[dep].dependents.remove(name)
                
                # Remove component
                del self._components[name]
                
                self.logger.info(f"Component {name} unregistered successfully")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to unregister component {name}: {e}")
                return False
    
    def get_component(self, name: str) -> Optional[ModularComponent]:
        """
        Get a component instance by name.
        
        Args:
            name: Name of the component
            
        Returns:
            Component instance or None if not found
        """
        with self._lock:
            if name not in self._components:
                return None
            
            component_info = self._components[name]
            
            # Create instance if not exists
            if component_info.instance is None:
                try:
                    component_info.instance = component_info.component_class(
                        name=name,
                        config=component_info.metadata.get('config', {}),
                        logger=self.logger
                    )
                    component_info.last_updated = time.time()
                except Exception as e:
                    self.logger.error(f"Failed to create component instance {name}: {e}")
                    return None
            
            return component_info.instance
    
    def initialize_component(self, name: str) -> bool:
        """
        Initialize a component.
        
        Args:
            name: Name of the component to initialize
            
        Returns:
            True if initialization successful, False otherwise
        """
        with self._lock:
            if name not in self._components:
                self.logger.error(f"Component {name} not found")
                return False
            
            component_info = self._components[name]
            
            # Check if already initialized
            if component_info.status == ComponentStatus.INITIALIZED:
                return True
            
            # Check dependencies
            if component_info.dependencies:
                for dep in component_info.dependencies:
                    if dep not in self._components:
                        self.logger.error(f"Dependency {dep} not found for component {name}")
                        return False
                    
                    dep_info = self._components[dep]
                    if dep_info.status != ComponentStatus.INITIALIZED:
                        self.logger.error(f"Dependency {dep} not initialized for component {name}")
                        return False
            
            # Get or create component instance
            component = self.get_component(name)
            if component is None:
                return False
            
            # Initialize component
            try:
                if component.initialize():
                    component_info.status = ComponentStatus.INITIALIZED
                    component_info.last_updated = time.time()
                    self.logger.info(f"Component {name} initialized successfully")
                    return True
                else:
                    component_info.status = ComponentStatus.ERROR
                    component_info.error_count += 1
                    self.logger.error(f"Component {name} initialization failed")
                    return False
            except Exception as e:
                component_info.status = ComponentStatus.ERROR
                component_info.error_count += 1
                self.logger.error(f"Component {name} initialization error: {e}")
                return False
    
    def start_component(self, name: str) -> bool:
        """
        Start a component.
        
        Args:
            name: Name of the component to start
            
        Returns:
            True if start successful, False otherwise
        """
        with self._lock:
            if name not in self._components:
                self.logger.error(f"Component {name} not found")
                return False
            
            component_info = self._components[name]
            
            # Check if already running
            if component_info.status == ComponentStatus.RUNNING:
                return True
            
            # Initialize if not already initialized
            if component_info.status != ComponentStatus.INITIALIZED:
                if not self.initialize_component(name):
                    return False
            
            # Start component
            try:
                component_info.status = ComponentStatus.RUNNING
                component_info.last_updated = time.time()
                self.logger.info(f"Component {name} started successfully")
                return True
            except Exception as e:
                component_info.status = ComponentStatus.ERROR
                component_info.error_count += 1
                self.logger.error(f"Component {name} start error: {e}")
                return False
    
    def stop_component(self, name: str) -> bool:
        """
        Stop a component.
        
        Args:
            name: Name of the component to stop
            
        Returns:
            True if stop successful, False otherwise
        """
        with self._lock:
            if name not in self._components:
                self.logger.error(f"Component {name} not found")
                return False
            
            component_info = self._components[name]
            
            # Check if already stopped
            if component_info.status == ComponentStatus.STOPPED:
                return True
            
            # Stop component
            try:
                component_info.status = ComponentStatus.STOPPED
                component_info.last_updated = time.time()
                self.logger.info(f"Component {name} stopped successfully")
                return True
            except Exception as e:
                component_info.status = ComponentStatus.ERROR
                component_info.error_count += 1
                self.logger.error(f"Component {name} stop error: {e}")
                return False
    
    def cleanup_component(self, name: str) -> bool:
        """
        Cleanup a component.
        
        Args:
            name: Name of the component to cleanup
            
        Returns:
            True if cleanup successful, False otherwise
        """
        with self._lock:
            if name not in self._components:
                self.logger.error(f"Component {name} not found")
                return False
            
            component_info = self._components[name]
            
            # Cleanup component
            try:
                if component_info.instance:
                    component_info.instance.cleanup()
                
                component_info.status = ComponentStatus.CLEANUP
                component_info.last_updated = time.time()
                self.logger.info(f"Component {name} cleaned up successfully")
                return True
            except Exception as e:
                component_info.status = ComponentStatus.ERROR
                component_info.error_count += 1
                self.logger.error(f"Component {name} cleanup error: {e}")
                return False
    
    def get_component_status(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get component status information.
        
        Args:
            name: Name of the component
            
        Returns:
            Status information or None if not found
        """
        with self._lock:
            if name not in self._components:
                return None
            
            component_info = self._components[name]
            
            return {
                'name': name,
                'type': component_info.component_type.value,
                'status': component_info.status.value,
                'dependencies': component_info.dependencies,
                'dependents': component_info.dependents,
                'created_at': component_info.created_at,
                'last_updated': component_info.last_updated,
                'error_count': component_info.error_count,
                'warning_count': component_info.warning_count,
                'health_status': component_info.health_status,
                'performance_stats': component_info.performance_stats,
                'metadata': component_info.metadata
            }
    
    def get_all_components(self) -> List[Dict[str, Any]]:
        """Get status of all components."""
        with self._lock:
            return [self.get_component_status(name) for name in self._components.keys()]
    
    def get_components_by_type(self, component_type: ComponentType) -> List[Dict[str, Any]]:
        """Get components of a specific type."""
        with self._lock:
            return [
                self.get_component_status(name)
                for name, info in self._components.items()
                if info.component_type == component_type
            ]
    
    def get_dependency_chain(self, name: str) -> List[str]:
        """Get the dependency chain for a component."""
        with self._lock:
            if name not in self._components:
                return []
            
            visited = set()
            chain = []
            
            def build_chain(component_name):
                if component_name in visited:
                    return
                
                visited.add(component_name)
                component_info = self._components[component_name]
                
                for dep in component_info.dependencies:
                    build_chain(dep)
                
                chain.append(component_name)
            
            build_chain(name)
            return chain
    
    def check_dependencies(self, name: str) -> bool:
        """Check if all dependencies are satisfied for a component."""
        with self._lock:
            if name not in self._components:
                return False
            
            component_info = self._components[name]
            
            for dep in component_info.dependencies:
                if dep not in self._components:
                    return False
                
                dep_info = self._components[dep]
                if dep_info.status != ComponentStatus.INITIALIZED:
                    return False
            
            return True
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all components."""
        with self._lock:
            summary = {
                'total_components': len(self._components),
                'initialized_components': 0,
                'running_components': 0,
                'error_components': 0,
                'total_errors': 0,
                'total_warnings': 0,
                'components': {}
            }
            
            for name, info in self._components.items():
                if info.status == ComponentStatus.INITIALIZED:
                    summary['initialized_components'] += 1
                elif info.status == ComponentStatus.RUNNING:
                    summary['running_components'] += 1
                elif info.status == ComponentStatus.ERROR:
                    summary['error_components'] += 1
                
                summary['total_errors'] += info.error_count
                summary['total_warnings'] += info.warning_count
                
                summary['components'][name] = {
                    'status': info.status.value,
                    'error_count': info.error_count,
                    'warning_count': info.warning_count,
                    'health_status': info.health_status
                }
            
            return summary
    
    def update_performance_stats(self, name: str, stats: Dict[str, Any]) -> None:
        """Update performance statistics for a component."""
        with self._lock:
            if name not in self._components:
                return
            
            component_info = self._components[name]
            component_info.performance_stats.update(stats)
            component_info.last_updated = time.time()
            
            # Store in performance history
            if self._enable_performance_monitoring:
                self._performance_history[name].append({
                    'timestamp': time.time(),
                    'stats': stats.copy()
                })
    
    def update_health_status(self, name: str, health_status: str) -> None:
        """Update health status for a component."""
        with self._lock:
            if name not in self._components:
                return
            
            component_info = self._components[name]
            component_info.health_status = health_status
            component_info.last_updated = time.time()
            
            # Store in health history
            if self._enable_health_monitoring:
                self._health_history[name].append({
                    'timestamp': time.time(),
                    'health_status': health_status
                })
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run health checks on all components."""
        current_time = time.time()
        
        # Check if enough time has passed since last health check
        if current_time - self._last_health_check < self._health_check_interval:
            return {'status': 'skipped', 'reason': 'too_soon'}
        
        self._last_health_check = current_time
        
        with self._lock:
            health_results = {
                'timestamp': current_time,
                'total_components': len(self._components),
                'healthy_components': 0,
                'unhealthy_components': 0,
                'unknown_components': 0,
                'components': {}
            }
            
            for name, info in self._components.items():
                if info.instance:
                    try:
                        health_report = info.instance.get_health_report()
                        health_status = health_report.get('overall_health', 'unknown')
                        
                        if health_status == 'healthy':
                            health_results['healthy_components'] += 1
                        elif health_status == 'critical':
                            health_results['unhealthy_components'] += 1
                        else:
                            health_results['unknown_components'] += 1
                        
                        health_results['components'][name] = {
                            'health_status': health_status,
                            'details': health_report
                        }
                        
                        # Update component health status
                        self.update_health_status(name, health_status)
                        
                    except Exception as e:
                        self.logger.error(f"Health check failed for component {name}: {e}")
                        health_results['unknown_components'] += 1
                        health_results['components'][name] = {
                            'health_status': 'error',
                            'error': str(e)
                        }
                else:
                    health_results['unknown_components'] += 1
                    health_results['components'][name] = {
                        'health_status': 'unknown',
                        'reason': 'not_initialized'
                    }
            
            return health_results
    
    def export_registry(self, filepath: str) -> bool:
        """Export registry to file."""
        try:
            with self._lock:
                export_data = {
                    'timestamp': time.time(),
                    'components': {},
                    'dependency_graph': {
                        'nodes': list(self._dependency_graph.nodes),
                        'edges': {k: list(v) for k, v in self._dependency_graph.edges.items()}
                    },
                    'settings': {
                        'max_components': self._max_components,
                        'performance_history_size': self._performance_history_size,
                        'health_check_interval': self._health_check_interval,
                        'enable_strategy_versioning': self._enable_strategy_versioning,
                        'enable_performance_monitoring': self._enable_performance_monitoring,
                        'enable_health_monitoring': self._enable_health_monitoring,
                        'enable_dependency_tracking': self._enable_dependency_tracking
                    }
                }
                
                for name, info in self._components.items():
                    export_data['components'][name] = {
                        'name': info.name,
                        'component_type': info.component_type.value,
                        'status': info.status.value,
                        'dependencies': info.dependencies,
                        'dependents': info.dependents,
                        'created_at': info.created_at,
                        'last_updated': info.last_updated,
                        'error_count': info.error_count,
                        'warning_count': info.warning_count,
                        'health_status': info.health_status,
                        'metadata': info.metadata
                    }
                
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                self.logger.info(f"Registry exported to {filepath}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to export registry: {e}")
            return False
    
    def import_registry(self, filepath: str) -> bool:
        """Import registry from file."""
        try:
            with open(filepath, 'r') as f:
                import_data = json.load(f)
            
            with self._lock:
                # Clear existing registry
                self._components.clear()
                self._dependency_graph = DependencyGraph()
                
                # Import components
                for name, comp_data in import_data.get('components', {}).items():
                    component_info = ComponentInfo(
                        name=comp_data['name'],
                        component_type=ComponentType(comp_data['component_type']),
                        component_class=None,  # Will need to be set separately
                        instance=None,
                        status=ComponentStatus(comp_data['status']),
                        dependencies=comp_data['dependencies'],
                        dependents=comp_data['dependents'],
                        created_at=comp_data['created_at'],
                        last_updated=comp_data['last_updated'],
                        error_count=comp_data['error_count'],
                        warning_count=comp_data['warning_count'],
                        health_status=comp_data['health_status'],
                        metadata=comp_data['metadata']
                    )
                    
                    self._components[name] = component_info
                
                # Import dependency graph
                dep_data = import_data.get('dependency_graph', {})
                for node in dep_data.get('nodes', []):
                    self._dependency_graph.add_node(node)
                
                for from_node, to_nodes in dep_data.get('edges', {}).items():
                    for to_node in to_nodes:
                        self._dependency_graph.add_edge(from_node, to_node)
                
                # Import settings
                settings = import_data.get('settings', {})
                self._max_components = settings.get('max_components', 100)
                self._performance_history_size = settings.get('performance_history_size', 1000)
                self._health_check_interval = settings.get('health_check_interval', 60.0)
                self._enable_strategy_versioning = settings.get('enable_strategy_versioning', True)
                self._enable_performance_monitoring = settings.get('enable_performance_monitoring', True)
                self._enable_health_monitoring = settings.get('enable_health_monitoring', True)
                self._enable_dependency_tracking = settings.get('enable_dependency_tracking', True)
                
                self.logger.info(f"Registry imported from {filepath}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to import registry: {e}")
            return False


# Global registry instance
_registry_instance: Optional[BacktestingComponentRegistry] = None


def get_registry() -> BacktestingComponentRegistry:
    """Get the global registry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = BacktestingComponentRegistry()
    return _registry_instance


def register_component(
    name: str,
    component_type: ComponentType,
    component_class: Type[ModularComponent],
    dependencies: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """Register a component in the global registry."""
    return get_registry().register_component(name, component_type, component_class, dependencies, metadata)


def get_component(name: str) -> Optional[ModularComponent]:
    """Get a component from the global registry."""
    return get_registry().get_component(name)


def initialize_component(name: str) -> bool:
    """Initialize a component in the global registry."""
    return get_registry().initialize_component(name)


def start_component(name: str) -> bool:
    """Start a component in the global registry."""
    return get_registry().start_component(name)


def stop_component(name: str) -> bool:
    """Stop a component in the global registry."""
    return get_registry().stop_component(name)


def cleanup_component(name: str) -> bool:
    """Cleanup a component in the global registry."""
    return get_registry().cleanup_component(name)


def get_component_status(name: str) -> Optional[Dict[str, Any]]:
    """Get component status from the global registry."""
    return get_registry().get_component_status(name)


def get_all_components() -> List[Dict[str, Any]]:
    """Get all components from the global registry."""
    return get_registry().get_all_components()


def run_health_checks() -> Dict[str, Any]:
    """Run health checks on all components in the global registry."""
    return get_registry().run_health_checks()


# Export all public classes and functions
__all__ = [
    'ComponentStatus',
    'ComponentType',
    'ComponentInfo',
    'DependencyGraph',
    'BacktestingComponentRegistry',
    'get_registry',
    'register_component',
    'get_component',
    'initialize_component',
    'start_component',
    'stop_component',
    'cleanup_component',
    'get_component_status',
    'get_all_components',
    'run_health_checks'
]