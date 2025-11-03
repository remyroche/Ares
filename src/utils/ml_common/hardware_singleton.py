"""
Singleton Hardware Capabilities Manager

This module provides a singleton for hardware detection to prevent
repeated expensive hardware detection operations across the codebase.

Author: Ares Trading System
Date: 2025-11-02
"""

import logging
import multiprocessing
from typing import Dict, Any, Optional
from dataclasses import dataclass
import threading

logger = logging.getLogger(__name__)

# Try to import PyTorch for GPU detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Try to import psutil for memory detection
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None


@dataclass
class HardwareCapabilities:
    """Hardware capabilities data class."""
    cpu_cores: int
    gpu_available: bool
    gpu_type: Optional[str]
    memory_gb: float
    mps_available: bool
    gpu_memory_gb: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'cpu_cores': self.cpu_cores,
            'gpu_available': self.gpu_available,
            'gpu_type': self.gpu_type,
            'memory_gb': self.memory_gb,
            'mps_available': self.mps_available,
            'gpu_memory_gb': self.gpu_memory_gb
        }


class HardwareCapabilitiesManager:
    """
    Thread-safe singleton manager for hardware capabilities.
    
    Detects hardware once and caches the result for all subsequent accesses.
    This prevents the expensive hardware detection from running multiple times.
    """
    
    _instance: Optional['HardwareCapabilitiesManager'] = None
    _lock = threading.Lock()
    _capabilities: Optional[HardwareCapabilities] = None
    _detection_complete = False
    
    def __new__(cls):
        """Ensure only one instance exists (thread-safe singleton pattern)."""
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super(HardwareCapabilitiesManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the manager (only runs once)."""
        # Prevent re-initialization
        if self._detection_complete:
            return
            
        with self._lock:
            if not self._detection_complete:
                logger.info("🔍 Detecting hardware capabilities (one-time detection)...")
                self._detect_hardware()
                self._detection_complete = True
                logger.info(f"✅ Hardware detection complete: {self._capabilities.to_dict()}")
    
    def _detect_hardware(self):
        """Detect hardware capabilities (private method, called once)."""
        # Default values
        cpu_cores = 1
        gpu_available = False
        gpu_type = None
        memory_gb = 4.0
        mps_available = False
        gpu_memory_gb = None
        
        # Detect CPU cores
        try:
            cpu_cores = multiprocessing.cpu_count()
            logger.debug(f"📊 CPU cores detected: {cpu_cores}")
        except Exception as e:
            logger.warning(f"Failed to detect CPU cores: {e}, using default: {cpu_cores}")
        
        # Detect GPU availability
        if TORCH_AVAILABLE:
            try:
                if torch.cuda.is_available():
                    gpu_available = True
                    gpu_type = 'cuda'
                    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    logger.debug(f"📊 CUDA GPU detected with {gpu_memory_gb:.1f}GB memory")
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    gpu_available = True
                    gpu_type = 'mps'
                    mps_available = True
                    logger.debug("📊 MPS GPU detected")
                else:
                    logger.debug("📊 No GPU detected")
            except Exception as e:
                logger.warning(f"GPU detection failed: {e}")
        else:
            logger.debug("⚠️ PyTorch not available for GPU detection")
        
        # Detect system memory
        if PSUTIL_AVAILABLE:
            try:
                memory_gb = psutil.virtual_memory().total / (1024**3)
                logger.debug(f"📊 System memory: {memory_gb:.1f}GB")
            except Exception as e:
                logger.warning(f"Memory detection failed: {e}, using default: {memory_gb}GB")
        else:
            logger.debug("⚠️ psutil not available, using default memory estimate")
        
        # Store capabilities
        self._capabilities = HardwareCapabilities(
            cpu_cores=cpu_cores,
            gpu_available=gpu_available,
            gpu_type=gpu_type,
            memory_gb=memory_gb,
            mps_available=mps_available,
            gpu_memory_gb=gpu_memory_gb
        )
    
    def get_capabilities(self) -> HardwareCapabilities:
        """Get hardware capabilities (cached)."""
        if self._capabilities is None:
            # This should never happen due to __init__, but safety check
            with self._lock:
                if self._capabilities is None:
                    self._detect_hardware()
        return self._capabilities
    
    def get_capabilities_dict(self) -> Dict[str, Any]:
        """Get hardware capabilities as dictionary (cached)."""
        return self.get_capabilities().to_dict()
    
    @classmethod
    def reset(cls):
        """Reset the singleton (mainly for testing)."""
        with cls._lock:
            cls._instance = None
            cls._capabilities = None
            cls._detection_complete = False


# Global singleton instance accessor
_hardware_manager_instance: Optional[HardwareCapabilitiesManager] = None


def get_hardware_capabilities_manager() -> HardwareCapabilitiesManager:
    """
    Get the singleton hardware capabilities manager.
    
    Returns:
        HardwareCapabilitiesManager: Singleton instance
    """
    global _hardware_manager_instance
    if _hardware_manager_instance is None:
        _hardware_manager_instance = HardwareCapabilitiesManager()
    return _hardware_manager_instance


def get_hardware_capabilities() -> HardwareCapabilities:
    """
    Get hardware capabilities (convenience function).
    
    Returns:
        HardwareCapabilities: Cached hardware capabilities
    """
    return get_hardware_capabilities_manager().get_capabilities()


def get_hardware_capabilities_dict() -> Dict[str, Any]:
    """
    Get hardware capabilities as dictionary (convenience function).
    
    Returns:
        Dict[str, Any]: Cached hardware capabilities as dict
    """
    return get_hardware_capabilities_manager().get_capabilities_dict()


# Example usage:
# from src.utils.ml_common.hardware_singleton import get_hardware_capabilities
# 
# caps = get_hardware_capabilities()
# print(f"CPU cores: {caps.cpu_cores}")
# print(f"GPU available: {caps.gpu_available}")
# print(f"GPU type: {caps.gpu_type}")
# print(f"Memory: {caps.memory_gb}GB")

