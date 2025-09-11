"""
M1 GPU Utilities for Apple Silicon optimization.

This module provides utilities for leveraging M1 GPU acceleration
for machine learning and data processing operations.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import sys
import platform

logger = logging.getLogger(__name__)

class M1GPUManager:
    """Manager for M1 GPU operations."""

    def __init__(self):
        self.is_m1 = self._detect_m1()
        self.mps_available = self._check_mps_availability()
        self.logger = logger.getChild('M1GPUManager')

    def _detect_m1(self) -> bool:
        """Detect if running on Apple Silicon (M1/M2/M3)."""
        try:
            # Check platform
            if platform.system() != 'Darwin':
                return False

            # Check for Apple Silicon
            import subprocess
            result = subprocess.run(['sysctl', 'machdep.cpu.brand_string'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                brand = result.stdout.strip()
                return 'Apple' in brand or 'M1' in brand or 'M2' in brand or 'M3' in brand

            return False
        except Exception as e:
            self.logger.warning(f"Could not detect M1 hardware: {e}")
            return False

    def _check_mps_availability(self) -> bool:
        """Check if Metal Performance Shaders (MPS) is available."""
        try:
            import torch
            if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
                return torch.backends.mps.is_available()
            return False
        except ImportError:
            return False

    def get_gpu_info(self) -> Dict[str, Any]:
        """Get information about available GPU resources."""
        info = {
            'is_m1': self.is_m1,
            'mps_available': self.mps_available,
            'gpu_memory': None,
            'gpu_name': None
        }

        if self.mps_available:
            try:
                import torch
                if torch.backends.mps.is_available():
                    # Get MPS device info
                    device = torch.device('mps')
                    info['gpu_name'] = 'Apple Silicon GPU (MPS)'
                    # MPS doesn't provide direct memory info, but we can estimate
                    info['gpu_memory'] = 'Shared system memory'
            except Exception as e:
                self.logger.warning(f"Could not get GPU info: {e}")

        return info

    def optimize_tensor_operations(self, data: np.ndarray) -> np.ndarray:
        """Optimize tensor operations for M1 GPU."""
        if not self.mps_available:
            self.logger.debug("MPS not available, using CPU operations")
            return data

        try:
            import torch

            # Convert to torch tensor and move to MPS
            tensor = torch.from_numpy(data).to('mps')

            # Perform any optimizations here
            # For now, just return the data (placeholder for actual optimizations)

            # Convert back to numpy
            result = tensor.cpu().numpy()

            return result

        except Exception as e:
            self.logger.warning(f"M1 GPU optimization failed, falling back to CPU: {e}")
            return data

    def create_mps_model(self, model_class: Any, *args, **kwargs):
        """Create a model optimized for MPS."""
        if not self.mps_available:
            self.logger.debug("MPS not available, creating standard model")
            return model_class(*args, **kwargs)

        try:
            import torch
            model = model_class(*args, **kwargs)

            # Move model to MPS if it has parameters
            if hasattr(model, 'parameters'):
                model = model.to('mps')
                self.logger.info("Model moved to MPS device")

            return model

        except Exception as e:
            self.logger.warning(f"Could not create MPS model, using CPU: {e}")
            return model_class(*args, **kwargs)


# Global instance
m1_gpu_manager = M1GPUManager()


def get_m1_gpu_manager() -> M1GPUManager:
    """Get the global M1 GPU manager instance."""
    return m1_gpu_manager


def is_m1_available() -> bool:
    """Check if M1 hardware is available."""
    return m1_gpu_manager.is_m1


def is_mps_available() -> bool:
    """Check if MPS is available."""
    return m1_gpu_manager.mps_available


def optimize_dataframe_for_m1(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame operations for M1."""
    if not m1_gpu_manager.is_m1:
        return df

    try:
        # Convert numeric columns to float32 for better M1 performance
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if df[col].dtype == np.float64:
                df[col] = df[col].astype(np.float32)

        logger.info(f"Optimized {len(numeric_cols)} numeric columns for M1")

    except Exception as e:
        logger.warning(f"DataFrame optimization failed: {e}")

    return df


def create_m1_optimized_array(data: Union[list, np.ndarray], dtype: np.dtype = np.float32) -> np.ndarray:
    """Create numpy array optimized for M1."""
    if not m1_gpu_manager.is_m1:
        return np.array(data, dtype=dtype)

    try:
        # Use float32 by default for M1 optimization
        if dtype == np.float64:
            logger.info("Converting float64 to float32 for M1 optimization")
            dtype = np.float32

        array = np.array(data, dtype=dtype)

        # Ensure contiguous memory layout for better performance
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)

        return array

    except Exception as e:
        logger.warning(f"Array optimization failed: {e}")
        return np.array(data, dtype=dtype)
