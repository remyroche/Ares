#!/usr/bin/env python3
"""
Unified Hardware Optimization

This module provides unified hardware optimization using existing hardware/ tools,
consolidating M1 GPU, memory, and CPU optimization into a single interface.

Key Features:
- Direct use of existing hardware/ tools
- M1 Apple Silicon optimization
- GPU acceleration with MPS support
- Memory optimization and management
- CPU optimization for parallel processing
"""

import logging
from typing import Any, Dict, List, Optional, Union
from contextlib import contextmanager

# Import existing hardware tools directly
try:
    from src.utils.hardware.m1_gpu_utils import M1GPUManager, get_m1_gpu_manager
    from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer, get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer, get_m1_cpu_optimizer
    HARDWARE_TOOLS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Hardware tools not available: {e}")
    HARDWARE_TOOLS_AVAILABLE = False

# Import utility modules
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_warning, tprint_error, tprint_success
    )
    UTILITY_MODULES_AVAILABLE = True
except ImportError:
    UTILITY_MODULES_AVAILABLE = False
    # Fallback functions
    def tprint(*args, **kwargs):
        print(*args, **kwargs)
    def tprint_info(*args, **kwargs):
        print("INFO:", *args, **kwargs)
    def tprint_warning(*args, **kwargs):
        print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs):
        print("ERROR:", *args, **kwargs)
    def tprint_success(*args, **kwargs):
        print("SUCCESS:", *args, **kwargs)

logger = logging.getLogger(__name__)


class UnifiedHardwareOptimizer:
    """Unified hardware optimization using existing hardware/ tools."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize hardware optimizer with existing tools."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize hardware managers using existing tools
        self.gpu_manager = None
        self.memory_optimizer = None
        self.cpu_optimizer = None
        
        if HARDWARE_TOOLS_AVAILABLE and config.get('enable_hardware_optimization', True):
            self._initialize_hardware_tools()
    
    def _initialize_hardware_tools(self):
        """Initialize hardware tools directly."""
        try:
            # Use existing hardware tools directly
            if self.config.get('enable_m1_optimization', True):
                self.gpu_manager = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer()
                self.cpu_optimizer = get_m1_cpu_optimizer()
                
                tprint_info("Hardware tools initialized using existing hardware/ modules")
            else:
                tprint_info("Hardware optimization disabled")
        except Exception as e:
            tprint_warning(f"Could not initialize hardware tools: {e}")
    
    @contextmanager
    def gpu_context(self):
        """Context manager for GPU operations using existing tools."""
        if self.gpu_manager:
            try:
                # Use existing GPU context from hardware tools
                if hasattr(self.gpu_manager, 'gpu_context'):
                    with self.gpu_manager.gpu_context():
                        yield
                else:
                    yield
            except Exception as e:
                tprint_warning(f"GPU context failed: {e}")
                yield
        else:
            yield
    
    @contextmanager
    def memory_context(self):
        """Context manager for memory optimization using existing tools."""
        if self.memory_optimizer:
            try:
                # Use existing memory context from hardware tools
                if hasattr(self.memory_optimizer, 'memory_checkpoint'):
                    with self.memory_optimizer.memory_checkpoint():
                        yield
                else:
                    yield
            except Exception as e:
                tprint_warning(f"Memory context failed: {e}")
                yield
        else:
            yield
    
    def optimize_data(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Optimize data using existing hardware tools."""
        if self.memory_optimizer and isinstance(data, pd.DataFrame):
            try:
                # Use existing data optimization from hardware tools
                if hasattr(self.memory_optimizer, 'optimize_dataframe'):
                    return self.memory_optimizer.optimize_dataframe(data)
            except Exception as e:
                tprint_warning(f"Data optimization failed: {e}")
        
        return data
    
    def get_memory_usage(self) -> float:
        """Get memory usage using existing tools."""
        if self.memory_optimizer:
            try:
                if hasattr(self.memory_optimizer, 'get_memory_usage'):
                    return self.memory_optimizer.get_memory_usage()
            except Exception as e:
                tprint_warning(f"Memory usage check failed: {e}")
        
        # Fallback
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0
    
    def cleanup(self):
        """Cleanup using existing hardware tools."""
        if self.memory_optimizer:
            try:
                if hasattr(self.memory_optimizer, 'cleanup'):
                    self.memory_optimizer.cleanup()
            except Exception as e:
                tprint_warning(f"Hardware cleanup failed: {e}")
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get information about available hardware."""
        info = {
            'hardware_tools_available': HARDWARE_TOOLS_AVAILABLE,
            'gpu_manager': self.gpu_manager is not None,
            'memory_optimizer': self.memory_optimizer is not None,
            'cpu_optimizer': self.cpu_optimizer is not None
        }
        
        if self.gpu_manager:
            try:
                if hasattr(self.gpu_manager, 'get_gpu_info'):
                    info['gpu_info'] = self.gpu_manager.get_gpu_info()
            except Exception as e:
                tprint_warning(f"Could not get GPU info: {e}")
        
        return info
    
    def start_monitoring(self):
        """Start hardware monitoring if available."""
        if self.memory_optimizer:
            try:
                if hasattr(self.memory_optimizer, 'start_monitoring'):
                    self.memory_optimizer.start_monitoring()
                    tprint_info("Hardware monitoring started")
            except Exception as e:
                tprint_warning(f"Could not start hardware monitoring: {e}")
    
    def stop_monitoring(self):
        """Stop hardware monitoring if available."""
        if self.memory_optimizer:
            try:
                if hasattr(self.memory_optimizer, 'stop_monitoring'):
                    self.memory_optimizer.stop_monitoring()
                    tprint_info("Hardware monitoring stopped")
            except Exception as e:
                tprint_warning(f"Could not stop hardware monitoring: {e}")