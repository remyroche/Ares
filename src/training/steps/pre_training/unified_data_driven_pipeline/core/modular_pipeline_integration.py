"""
ModularComponent Pipeline Integration

This module provides integration between ModularComponent and the consolidated pipeline,
enabling enhanced monitoring, state management, and performance tracking.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

from .modular_architecture import ModularComponent
from .migration_utils import create_component_wrapper


class ModularPipelineOrchestrator(ModularComponent):
    """
    Orchestrator for managing ModularComponent instances in the pipeline.
    
    This class provides enhanced pipeline orchestration with:
    - Component lifecycle management
    - Performance monitoring across components
    - State management and persistence
    - Health monitoring and alerting
    """
    
    def __init__(self, name: str = "pipeline_orchestrator", 
                 config: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize the pipeline orchestrator."""
        super().__init__(name, config or {}, logger)
        self.components: Dict[str, ModularComponent] = {}
        self.component_health: Dict[str, Dict[str, Any]] = {}
        self.pipeline_state: Dict[str, Any] = {}
        
    def _initialize_resources(self) -> bool:
        """Initialize orchestrator resources."""
        try:
            self.set_state('initialized_at', datetime.now().isoformat())
            self.set_state('component_count', 0)
            self.set_state('execution_count', 0)
            return True
        except Exception as e:
            self.logger.error(f"Orchestrator initialization failed: {e}")
            return False
    
    def _cleanup_resources(self) -> None:
        """Cleanup orchestrator resources."""
        # Cleanup all registered components
        for component_name, component in self.components.items():
            try:
                if component.is_initialized():
                    component.cleanup()
                self.logger.info(f"Cleaned up component: {component_name}")
            except Exception as e:
                self.logger.error(f"Error cleaning up component {component_name}: {e}")
        
        self.set_state('cleaned_up_at', datetime.now().isoformat())
        self.components.clear()
        self.component_health.clear()
    
    def _process_data(self, data: Any, **kwargs) -> Any:
        """Process data through the pipeline."""
        # Increment execution count
        count = self.get_state('execution_count', 0)
        self.set_state('execution_count', count + 1)
        
        # Basic processing - return data as-is for now
        return data
    
    def _get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for the orchestrator."""
        return {
            'min_size': 1,
            'max_size': 1000000,
            'required_attributes': [],
            'data_types': ['pandas.DataFrame', 'dict', 'list'],
            'max_nan_ratio': 1.0,
            'min_unique_values': 1
        }
    
    def _validate_component_specific(self, data: Any) -> Dict[str, Any]:
        """Validate data with orchestrator-specific rules."""
        errors = []
        warnings = []
        metadata = {}
        
        # Basic validation
        if data is None:
            errors.append("Data cannot be None")
        
        metadata['data_type'] = type(data).__name__
        if hasattr(data, 'shape'):
            metadata['data_shape'] = data.shape
        
        return {'errors': errors, 'warnings': warnings, 'metadata': metadata}
    
    def register_component(self, name: str, component: ModularComponent) -> bool:
        """Register a component with the orchestrator."""
        try:
            if not isinstance(component, ModularComponent):
                self.logger.error(f"Component {name} is not a ModularComponent")
                return False
            
            self.components[name] = component
            self.component_health[name] = {
                'status': 'registered',
                'last_health_check': None,
                'execution_count': 0,
                'error_count': 0
            }
            
            self.logger.info(f"Registered component: {name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register component {name}: {e}")
            return False
    
    def unregister_component(self, name: str) -> bool:
        """Unregister a component from the orchestrator."""
        try:
            if name in self.components:
                component = self.components[name]
                if component.is_initialized():
                    component.cleanup()
                del self.components[name]
                del self.component_health[name]
                self.logger.info(f"Unregistered component: {name}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to unregister component {name}: {e}")
            return False
    
    def get_component_health(self, name: str) -> Optional[Dict[str, Any]]:
        """Get health status of a specific component."""
        if name not in self.component_health:
            return None
        
        try:
            component = self.components[name]
            health_report = component.get_health_report()
            
            self.component_health[name].update({
                'status': health_report['overall_health'],
                'last_health_check': datetime.now().isoformat(),
                'performance_metrics': health_report.get('performance_metrics', {}),
                'initialization_status': health_report.get('initialization_status', False)
            })
            
            return self.component_health[name]
        except Exception as e:
            self.logger.error(f"Failed to get health for component {name}: {e}")
            self.component_health[name]['status'] = 'error'
            self.component_health[name]['last_error'] = str(e)
            return self.component_health[name]
    
    def get_pipeline_health(self) -> Dict[str, Any]:
        """Get overall pipeline health status."""
        try:
            total_components = len(self.components)
            healthy_components = 0
            error_components = 0
            
            for name in self.components:
                health = self.get_component_health(name)
                if health:
                    if health['status'] == 'healthy':
                        healthy_components += 1
                    elif health['status'] == 'error':
                        error_components += 1
            
            overall_health = 'healthy' if error_components == 0 else 'degraded' if error_components < total_components else 'unhealthy'
            
            return {
                'overall_health': overall_health,
                'total_components': total_components,
                'healthy_components': healthy_components,
                'error_components': error_components,
                'component_health': self.component_health,
                'pipeline_state': self.pipeline_state,
                'orchestrator_health': self.get_health_report()
            }
        except Exception as e:
            self.logger.error(f"Failed to get pipeline health: {e}")
            return {
                'overall_health': 'error',
                'error': str(e),
                'total_components': len(self.components),
                'healthy_components': 0,
                'error_components': len(self.components)
            }
    
    def execute_component(self, name: str, data: Any, **kwargs) -> Any:
        """Execute a specific component with enhanced monitoring."""
        if name not in self.components:
            raise ValueError(f"Component {name} not registered")
        
        component = self.components[name]
        
        try:
            # Update health tracking
            self.component_health[name]['execution_count'] += 1
            
            # Execute with safe processing
            result = component._safe_process(data, **kwargs)
            
            # Update success tracking
            self.logger.info(f"Component {name} executed successfully")
            return result
            
        except Exception as e:
            # Update error tracking
            self.component_health[name]['error_count'] += 1
            self.component_health[name]['last_error'] = str(e)
            self.logger.error(f"Component {name} execution failed: {e}")
            raise
    
    def save_pipeline_state(self, filepath: str) -> bool:
        """Save the current pipeline state to a file."""
        try:
            state_data = {
                'orchestrator_state': self.serialize(),
                'component_states': {},
                'pipeline_state': self.pipeline_state,
                'health_status': self.component_health,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save each component's state
            for name, component in self.components.items():
                if component.is_initialized():
                    state_data['component_states'][name] = component.serialize()
            
            import json
            with open(filepath, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
            
            self.logger.info(f"Pipeline state saved to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save pipeline state: {e}")
            return False
    
    def load_pipeline_state(self, filepath: str) -> bool:
        """Load pipeline state from a file."""
        try:
            import json
            with open(filepath, 'r') as f:
                state_data = json.load(f)
            
            # Load orchestrator state
            if 'orchestrator_state' in state_data:
                self.deserialize(state_data['orchestrator_state'])
            
            # Load pipeline state
            if 'pipeline_state' in state_data:
                self.pipeline_state = state_data['pipeline_state']
            
            # Load health status
            if 'health_status' in state_data:
                self.component_health = state_data['health_status']
            
            self.logger.info(f"Pipeline state loaded from {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load pipeline state: {e}")
            return False


def create_modular_pipeline_orchestrator(config: Optional[Dict[str, Any]] = None) -> ModularPipelineOrchestrator:
    """Create a new ModularPipelineOrchestrator instance."""
    return ModularPipelineOrchestrator(config=config)


def integrate_with_consolidated_pipeline(pipeline_instance, orchestrator: ModularPipelineOrchestrator) -> None:
    """
    Integrate ModularComponent orchestrator with existing consolidated pipeline.
    
    This function adds ModularComponent monitoring and state management to the
    existing pipeline without breaking existing functionality.
    """
    try:
        # Add orchestrator as an attribute to the pipeline
        pipeline_instance.modular_orchestrator = orchestrator
        
        # Add health monitoring method
        def get_pipeline_health():
            return orchestrator.get_pipeline_health()
        
        pipeline_instance.get_pipeline_health = get_pipeline_health
        
        # Add component registration method
        def register_pipeline_component(name: str, component: ModularComponent):
            return orchestrator.register_component(name, component)
        
        pipeline_instance.register_pipeline_component = register_pipeline_component
        
        # Add state management methods
        def save_pipeline_state(filepath: str):
            return orchestrator.save_pipeline_state(filepath)
        
        pipeline_instance.save_pipeline_state = save_pipeline_state
        
        def load_pipeline_state(filepath: str):
            return orchestrator.load_pipeline_state(filepath)
        
        pipeline_instance.load_pipeline_state = load_pipeline_state
        
        orchestrator.logger.info("Successfully integrated ModularComponent orchestrator with consolidated pipeline")
        
    except Exception as e:
        orchestrator.logger.error(f"Failed to integrate with consolidated pipeline: {e}")
        raise