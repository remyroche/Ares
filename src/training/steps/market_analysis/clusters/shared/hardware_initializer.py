"""
Centralized hardware initialization with upgraded tools.

This module eliminates the duplication of hardware initialization patterns
across the clustering codebase by providing a unified interface to the
enhanced hardware utilities.
"""

from typing import Dict, Any, Optional
from contextlib import contextmanager
from src.utils.tprint import tprint, tprint_warning, tprint_error

# Import upgraded hardware tools
try:
    from src.utils.hardware import (
        get_integrated_hardware_manager,
        get_enhanced_gpu_manager,
        get_advanced_memory_manager,
        get_advanced_cpu_optimizer,
        get_m1_gpu_manager,
        get_m1_memory_optimizer,
        get_m1_cpu_optimizer,
        is_m1_available
    )
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    tprint_warning("Enhanced hardware utilities not available, using fallbacks")


class HardwareInitializer:
    """Centralized hardware initialization with upgraded tools."""
    
    @staticmethod
    def initialize_hardware_components(component_name: str, 
                                    verbose: bool = True,
                                    use_enhanced: bool = True) -> Dict[str, Any]:
        """
        Initialize all hardware components using upgraded tools.
        
        Args:
            component_name: Name of the component for logging
            verbose: Whether to print status messages
            use_enhanced: Whether to use enhanced hardware tools (fallback to basic if unavailable)
            
        Returns:
            Dictionary containing hardware components and status
        """
        try:
            if use_enhanced and HARDWARE_AVAILABLE:
                return HardwareInitializer._initialize_enhanced_hardware(component_name, verbose)
            else:
                return HardwareInitializer._initialize_basic_hardware(component_name, verbose)
                
        except Exception as e:
            if verbose:
                tprint_error(f"Hardware initialization failed for {component_name}: {e}")
            return {
                'integrated_manager': None,
                'gpu_manager': None,
                'memory_manager': None,
                'cpu_optimizer': None,
                'initialization_successful': False,
                'error': str(e)
            }
    
    @staticmethod
    def _initialize_enhanced_hardware(component_name: str, verbose: bool) -> Dict[str, Any]:
        """Initialize using enhanced hardware tools."""
        try:
            # Use the new integrated hardware manager
            integrated_manager = get_integrated_hardware_manager()
            
            # Get enhanced components
            gpu_manager = get_enhanced_gpu_manager()
            memory_manager = get_advanced_memory_manager()
            cpu_optimizer = get_advanced_cpu_optimizer()
            
            if verbose:
                tprint(f"✅ Enhanced hardware optimization initialized for {component_name}", "SUCCESS")
            
            return {
                'integrated_manager': integrated_manager,
                'gpu_manager': gpu_manager,
                'memory_manager': memory_manager,
                'cpu_optimizer': cpu_optimizer,
                'initialization_successful': True,
                'hardware_type': 'enhanced'
            }
            
        except Exception as e:
            if verbose:
                tprint_warning(f"Enhanced hardware failed, falling back to basic: {e}")
            return HardwareInitializer._initialize_basic_hardware(component_name, verbose)
    
    @staticmethod
    def _initialize_basic_hardware(component_name: str, verbose: bool) -> Dict[str, Any]:
        """Initialize using basic hardware tools (fallback)."""
        try:
            # Check if M1 is available
            if is_m1_available():
                gpu_manager = get_m1_gpu_manager()
                memory_manager = get_m1_memory_optimizer()
                cpu_optimizer = get_m1_cpu_optimizer()
            else:
                gpu_manager = None
                memory_manager = None
                cpu_optimizer = None
            
            if verbose:
                if gpu_manager or memory_manager or cpu_optimizer:
                    tprint(f"✅ Basic hardware optimization initialized for {component_name}", "SUCCESS")
                else:
                    tprint(f"⚠️ No hardware acceleration available for {component_name}", "WARNING")
            
            return {
                'integrated_manager': None,
                'gpu_manager': gpu_manager,
                'memory_manager': memory_manager,
                'cpu_optimizer': cpu_optimizer,
                'initialization_successful': bool(gpu_manager or memory_manager or cpu_optimizer),
                'hardware_type': 'basic'
            }
            
        except Exception as e:
            if verbose:
                tprint_error(f"Basic hardware initialization failed for {component_name}: {e}")
            return {
                'integrated_manager': None,
                'gpu_manager': None,
                'memory_manager': None,
                'cpu_optimizer': None,
                'initialization_successful': False,
                'error': str(e),
                'hardware_type': 'none'
            }
    
    @staticmethod
    def get_hardware_context(component_name: str, use_enhanced: bool = True):
        """Get hardware context manager for automatic cleanup."""
        return HardwareContext(component_name, use_enhanced)
    
    @staticmethod
    def cleanup_hardware_components(components: Dict[str, Any]) -> None:
        """Clean up hardware components."""
        try:
            if components.get('integrated_manager'):
                components['integrated_manager'].cleanup()
            
            # Additional cleanup if needed
            for key, component in components.items():
                if hasattr(component, 'cleanup'):
                    try:
                        component.cleanup()
                    except Exception as e:
                        tprint_warning(f"Cleanup failed for {key}: {e}")
                        
        except Exception as e:
            tprint_warning(f"Hardware cleanup warning: {e}")


class HardwareContext:
    """Context manager for hardware operations with automatic cleanup."""
    
    def __init__(self, component_name: str, use_enhanced: bool = True):
        self.component_name = component_name
        self.use_enhanced = use_enhanced
        self.hardware_components = None
    
    def __enter__(self):
        self.hardware_components = HardwareInitializer.initialize_hardware_components(
            self.component_name, verbose=False, use_enhanced=self.use_enhanced
        )
        return self.hardware_components
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Automatic cleanup
        if self.hardware_components:
            HardwareInitializer.cleanup_hardware_components(self.hardware_components)


# Convenience functions for backward compatibility
def initialize_hardware_for_component(component_name: str, verbose: bool = True) -> Dict[str, Any]:
    """Convenience function for hardware initialization."""
    return HardwareInitializer.initialize_hardware_components(component_name, verbose)


def get_hardware_managers(component_name: str) -> tuple:
    """Get hardware managers as a tuple for easy unpacking."""
    components = HardwareInitializer.initialize_hardware_components(component_name, verbose=False)
    return (
        components.get('gpu_manager'),
        components.get('memory_manager'), 
        components.get('cpu_optimizer')
    )