#!/usr/bin/env python3
"""
Unified Component Manager

This module provides a unified component manager that orchestrates all unified components,
providing a single interface for managing evaluation, hardware optimization, search, and data processing.

Key Features:
- Unified component orchestration
- Configuration management
- Component lifecycle management
- Performance monitoring
- Resource cleanup
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from .evaluation import UnifiedEvaluator
from .hardware import UnifiedHardwareOptimizer, HardwareConfig, WorkloadType, OptimizationLevel
from .search import UnifiedSearchEngine
from .data_processing import UnifiedDataProcessor

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


class UnifiedComponentManager:
    """Unified component manager orchestrating all unified components."""
    
    def __init__(self, config: HardwareConfig):
        """Initialize unified component manager with modern HardwareConfig."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.evaluator = None
        self.hardware_optimizer = None
        self.search_engine = None
        self.data_processor = None
        
        # Performance tracking
        self.performance_metrics = {}
        self.start_time = None
        
        # Initialize all components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all unified components."""
        tprint_info("Initializing unified components...")
        
        try:
            # Initialize evaluator
            self.evaluator = UnifiedEvaluator(self.config.__dict__)
            tprint_success("✅ UnifiedEvaluator initialized")
            
            # Initialize hardware optimizer
            self.hardware_optimizer = UnifiedHardwareOptimizer(self.config)
            tprint_success("✅ UnifiedHardwareOptimizer initialized")
            
            # Initialize search engine
            self.search_engine = UnifiedSearchEngine(self.config.__dict__)
            tprint_success("✅ UnifiedSearchEngine initialized")
            
            # Initialize data processor
            self.data_processor = UnifiedDataProcessor(self.config.__dict__)
            tprint_success("✅ UnifiedDataProcessor initialized")
            
            tprint_success("🎉 All unified components initialized successfully")
            
        except Exception as e:
            tprint_error(f"Component initialization failed: {e}")
            self.logger.error(f"Component initialization error: {e}")
    
    def get_component_status(self) -> Dict[str, bool]:
        """Get status of all components."""
        return {
            'evaluator': self.evaluator is not None,
            'hardware_optimizer': self.hardware_optimizer is not None,
            'search_engine': self.search_engine is not None,
            'data_processor': self.data_processor is not None
        }
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get detailed information about all components."""
        info = {
            'component_status': self.get_component_status(),
            'config': self.config,
            'initialization_time': datetime.now().isoformat()
        }
        
        # Add hardware info if available
        if self.hardware_optimizer:
            info['hardware_info'] = self.hardware_optimizer.get_hardware_info()
        
        # Add search strategies if available
        if self.search_engine:
            info['available_search_strategies'] = self.search_engine.get_available_strategies()
        
        return info
    
    def start_performance_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.performance_metrics = {
            'start_time': self.start_time,
            'operations_count': 0,
            'total_memory_usage': 0.0
        }
        
        # Start hardware monitoring if available
        if self.hardware_optimizer:
            self.hardware_optimizer.start_monitoring()
        
        tprint_info("Performance monitoring started")
    
    def stop_performance_monitoring(self):
        """Stop performance monitoring."""
        if self.start_time:
            total_time = time.time() - self.start_time
            self.performance_metrics['total_time'] = total_time
            
            # Stop hardware monitoring if available
            if self.hardware_optimizer:
                self.hardware_optimizer.stop_monitoring()
            
            tprint_info(f"Performance monitoring stopped. Total time: {total_time:.2f}s")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        metrics = self.performance_metrics.copy()
        
        # Add current memory usage
        if self.hardware_optimizer:
            metrics['current_memory_usage'] = self.hardware_optimizer.get_memory_usage()
        
        return metrics
    
    def cleanup(self):
        """Cleanup all components and resources."""
        tprint_info("Cleaning up unified components...")
        
        try:
            # Cleanup hardware optimizer
            if self.hardware_optimizer:
                self.hardware_optimizer.cleanup()
            
            # Stop performance monitoring
            self.stop_performance_monitoring()
            
            # Reset components
            self.evaluator = None
            self.hardware_optimizer = None
            self.search_engine = None
            self.data_processor = None
            
            tprint_success("✅ All components cleaned up successfully")
            
        except Exception as e:
            tprint_error(f"Cleanup failed: {e}")
            self.logger.error(f"Cleanup error: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start_performance_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during cleanup in destructor


# Convenience functions for easy access to unified components
def create_unified_manager(config: HardwareConfig) -> UnifiedComponentManager:
    """Create a unified component manager with modern HardwareConfig."""
    return UnifiedComponentManager(config)


def get_default_hardware_config() -> HardwareConfig:
    """Get default HardwareConfig for unified components."""
    return HardwareConfig()


def create_nas_config() -> HardwareConfig:
    """Create configuration optimized for NAS."""
    return HardwareConfig(
        cpu_optimization_level=OptimizationLevel.AGGRESSIVE,
        gpu_optimization_level=OptimizationLevel.AGGRESSIVE,
        memory_optimization_level=OptimizationLevel.BALANCED,
        enable_adaptive_optimization=True,
        learning_enabled=True,
        performance_monitoring_enabled=True,
        monitoring_interval=3.0,  # More frequent monitoring for NAS
        alert_thresholds={
            'cpu_usage': 90.0,  # Higher threshold for NAS
            'memory_usage': 85.0,
            'gpu_usage': 90.0,
            'temperature': 85.0
        }
    )


def create_tas_config() -> HardwareConfig:
    """Create configuration optimized for TAS."""
    return HardwareConfig(
        cpu_optimization_level=OptimizationLevel.BALANCED,
        gpu_optimization_level=OptimizationLevel.MINIMAL,  # TAS doesn't need heavy GPU
        memory_optimization_level=OptimizationLevel.AGGRESSIVE,  # TAS needs memory optimization
        enable_adaptive_optimization=True,
        learning_enabled=True,
        performance_monitoring_enabled=True,
        monitoring_interval=5.0,  # Standard monitoring for TAS
        alert_thresholds={
            'cpu_usage': 80.0,
            'memory_usage': 90.0,  # Higher memory threshold for TAS
            'gpu_usage': 50.0,  # Lower GPU threshold for TAS
            'temperature': 85.0
        }
    )