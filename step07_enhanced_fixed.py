#!/usr/bin/env python3
"""
Step07 Enhanced Matrix Operations - Fixed Version

This is a completely self-contained version of Step07 with proper dependency
management and fallback handling. It doesn't rely on any external imports
that might fail.
"""

import sys
import os
import time
import traceback
import functools
import inspect
import gc
import json
import collections
import logging
import warnings
import math
import statistics
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
from pathlib import Path

# Enhanced dependency management with fallbacks
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    warnings.warn("NumPy not available - matrix operations will be limited")
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    warnings.warn("Pandas not available - DataFrame operations will be limited")
    PANDAS_AVAILABLE = False
    pd = None

# Try to import Numba for JIT compilation
try:
    from numba import jit, prange, float64, float32
    import numba as nb
    NUMBA_AVAILABLE = True
except ImportError:
    warnings.warn("Numba not available - JIT compilation disabled")
    NUMBA_AVAILABLE = False
    jit = lambda *args, **kwargs: lambda func: func  # No-op decorator
    prange = range  # Fallback to regular range

# Try to import psutil for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    warnings.warn("psutil not available - memory monitoring disabled")
    PSUTIL_AVAILABLE = False
    # Create a mock psutil class
    class MockPsutil:
        class Process:
            def memory_info(self):
                class MemoryInfo:
                    rss = 0
                return MemoryInfo()
            def cpu_percent(self):
                return 0.0
    psutil = MockPsutil()

# Try to import torch for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    warnings.warn("PyTorch not available - GPU acceleration disabled")
    TORCH_AVAILABLE = False
    torch = None

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
system_logger = logging.getLogger('Step07Enhanced')

# No-op decorators for when advanced features aren't available
def log_step_functions(func):
    return func

def log_important_calls(func):
    return func

def log_all_calls(func):
    return func

def log_internal_call(func):
    return func

def log_step_progress(func):
    return func

def log_data_operation(func):
    return func

def handles_errors(*args, **kwargs):
    def decorator(func):
        return func
    return decorator

# Base step class
class BaseStep:
    def __init__(self, config, step_id, step_name):
        self.config = config
        self.step_id = step_id
        self.step_name = step_name
    
    async def initialize(self):
        pass
    
    async def execute(self, training_input, pipeline_state):
        return pipeline_state

# Dependency checking functions
def check_step07_dependencies() -> Dict[str, bool]:
    """Check Step07 dependency status and return availability."""
    return {
        'numpy': NUMPY_AVAILABLE,
        'pandas': PANDAS_AVAILABLE,
        'numba': NUMBA_AVAILABLE,
        'torch': TORCH_AVAILABLE,
        'psutil': PSUTIL_AVAILABLE,
        'system_logger': True,  # Always available
        'logging_decorators': True,  # Always available (no-op)
        'handles_errors': True,  # Always available (no-op)
        'base_step': True,  # Always available
        'matrix_components': False,  # Not available in this version
        'enhanced_reporting': False  # Not available in this version
    }

def get_step07_capabilities() -> Dict[str, Any]:
    """Get Step07 capabilities based on available dependencies."""
    capabilities = {
        'matrix_operations': NUMPY_AVAILABLE,
        'dataframe_operations': PANDAS_AVAILABLE,
        'jit_compilation': NUMBA_AVAILABLE,
        'gpu_acceleration': TORCH_AVAILABLE,
        'memory_monitoring': PSUTIL_AVAILABLE,
        'async_processing': True,  # Always available
        'enhanced_reporting': False,  # Not available in this version
        'performance_optimization': NUMBA_AVAILABLE or TORCH_AVAILABLE
    }
    
    # Calculate overall capability score
    total_capabilities = len(capabilities)
    available_capabilities = sum(1 for available in capabilities.values() if available)
    capability_score = available_capabilities / total_capabilities
    
    capabilities['overall_score'] = capability_score
    capabilities['status'] = 'full' if capability_score >= 0.8 else 'limited' if capability_score >= 0.5 else 'minimal'
    
    return capabilities

# Numba-optimized matrix operation functions
if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, fastmath=True)
    def numba_tiled_matmul_kernel(a_block: np.ndarray, b_block: np.ndarray, c_tile: np.ndarray) -> np.ndarray:
        """Numba-optimized tiled matrix multiplication kernel."""
        m, k = a_block.shape
        n = b_block.shape[1]

        for i in prange(m):
            for j in prange(n):
                for l in prange(k):
                    c_tile[i, j] += a_block[i, l] * b_block[l, j]

        return c_tile

    @jit(nopython=True, parallel=True)
    def numba_matrix_norm(matrix: np.ndarray, norm_type: int = 2) -> float:
        """Numba-optimized matrix norm calculation."""
        if norm_type == 0:  # Frobenius norm
            return np.sqrt(np.sum(matrix ** 2))
        elif norm_type == 1:  # L1 norm
            return np.sum(np.abs(matrix))
        elif norm_type == 2:  # L2 norm
            return np.sqrt(np.sum(matrix ** 2))
        else:
            return np.sqrt(np.sum(matrix ** 2))

# Fallback matrix operations using standard library
def compute_basic_correlation(x: List[float], y: List[float]) -> float:
    """Compute basic correlation using standard library."""
    if len(x) != len(y) or len(x) == 0:
        return 0.0
    
    n = len(x)
    
    # Compute means
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    # Compute correlation
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    
    sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(n))
    sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(n))
    
    denominator = (sum_sq_x * sum_sq_y) ** 0.5
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator

def compute_basic_covariance(x: List[float], y: List[float], mean_x: float, mean_y: float) -> float:
    """Compute basic covariance using standard library."""
    if len(x) != len(y) or len(x) == 0:
        return 0.0
    
    n = len(x)
    return sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / (n - 1)

# Enhanced Matrix Operations Step
class EnhancedMatrixOperationsStep(BaseStep):
    """Step 7: Enhanced Matrix Operations with dependency management."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize enhanced matrix operations step."""
        super().__init__(config, '07', 'enhanced_matrix_operations')
        self.logger = system_logger.getChild('EnhancedMatrixOperationsStep')
        
        # Check dependencies and capabilities
        self.dependencies = check_step07_dependencies()
        self.capabilities = get_step07_capabilities()
        
        self.logger.info(f'🔍 Step07 Dependencies: {self.dependencies}')
        self.logger.info(f'📊 Step07 Capabilities: {self.capabilities}')
        
        # Initialize components based on availability
        if self.capabilities['status'] == 'full':
            self.logger.info('🚀 Full Step07 capabilities available')
        elif self.capabilities['status'] == 'limited':
            self.logger.warning('⚠️ Limited Step07 capabilities - some features disabled')
        else:
            self.logger.warning('⚠️ Minimal Step07 capabilities - using fallback implementations')
        
        # Configure matrix operations based on capabilities
        self.matrix_config = config.get('matrix_operations_config', {
            'use_gpu': self.capabilities['gpu_acceleration'],
            'use_numba': self.capabilities['jit_compilation'],
            'use_diverse_lookback': True,
            'optimization_level': 'high' if self.capabilities['performance_optimization'] else 'basic',
            'batch_size': 1000,
            'feature_selection': {
                'method': 'mutual_info',
                'top_k': 50,
                'min_importance': 0.01
            },
            'matrix_computations': {
                'correlation_matrix': True,
                'covariance_matrix': True,
                'feature_interaction_matrix': True,
                'regime_transition_matrix': True
            }
        })
    
    async def initialize(self) -> None:
        """Initialize the step."""
        self.logger.info('🔢 Initializing enhanced matrix operations step')
    
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the step."""
        try:
            self.logger.info('🔢 Starting enhanced matrix operations...')
            
            # Get data
            data_dict = self._get_data_to_process(pipeline_state)
            if not data_dict:
                self.logger.error("❌ No data available for processing")
                return pipeline_state
            
            # Process each split
            matrix_results = {}
            for split_name, data in data_dict.items():
                self.logger.info(f'🧮 Processing {split_name} split...')
                matrices = await self._compute_matrices(data, [], pipeline_state)
                matrix_results[split_name] = matrices
            
            # Update pipeline state
            pipeline_state.update({
                'matrix_results': matrix_results,
                'step07_enhanced_matrix_operations_completed': True
            })
            
            self.logger.info('✅ Enhanced matrix operations completed')
            return pipeline_state
            
        except Exception as e:
            self.logger.error(f'❌ Error in enhanced matrix operations: {e}')
            return pipeline_state
    
    def _get_data_to_process(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get data to process."""
        # Try to get engineered data
        if 'engineered_data' in pipeline_state:
            return pipeline_state['engineered_data']
        
        # Try to get advanced features
        if 'advanced_features' in pipeline_state:
            advanced_features = pipeline_state['advanced_features']
            data_dict = {}
            
            for split in ['train', 'val', 'test']:
                if split in advanced_features:
                    path = advanced_features[split]
                    if isinstance(path, str) and Path(path).exists():
                        try:
                            data_dict[split] = self._load_data_from_file(path)
                        except Exception as e:
                            self.logger.warning(f'⚠️ Failed to load {split} data: {e}')
            
            if data_dict:
                return data_dict
        
        # Fallback to individual data keys
        data_dict = {}
        for split in ['train', 'val', 'test']:
            if f'{split}_data' in pipeline_state:
                data_dict[split] = pipeline_state[f'{split}_data']
        
        return data_dict
    
    def _load_data_from_file(self, file_path: str) -> List[List[float]]:
        """Load data from file (basic CSV support)."""
        try:
            data = []
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        # Basic CSV parsing
                        values = line.strip().split(',')
                        try:
                            row = [float(v) for v in values]
                            data.append(row)
                        except ValueError:
                            # Skip non-numeric rows
                            continue
            return data
        except Exception as e:
            self.logger.warning(f'⚠️ Failed to load data from {file_path}: {e}')
            return []
    
    async def _compute_matrices(self, data: Any, selected_features: List[str], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Compute various matrices for the data with fallback support."""
        matrices = {}
        
        try:
            # Handle different data types
            if PANDAS_AVAILABLE and hasattr(data, 'columns'):
                # Pandas DataFrame
                if selected_features:
                    feature_data = data[selected_features]
                else:
                    feature_cols = [col for col in data.columns if col.startswith('feature_')]
                    feature_data = data[feature_cols] if feature_cols else data
            elif NUMPY_AVAILABLE and hasattr(data, 'shape'):
                # NumPy array
                feature_data = data
            else:
                # Fallback: convert to list and use basic operations
                self.logger.warning("⚠️ Using fallback matrix computation - limited functionality")
                return self._compute_matrices_fallback(data, selected_features)
            
            matrix_computations = self.matrix_config.get('matrix_computations', {})
            
            # Compute correlation matrix
            if matrix_computations.get('correlation_matrix', True):
                try:
                    if PANDAS_AVAILABLE and hasattr(feature_data, 'corr'):
                        matrices['correlation_matrix'] = feature_data.corr().values
                    elif NUMPY_AVAILABLE:
                        matrices['correlation_matrix'] = np.corrcoef(feature_data.T)
                    else:
                        self.logger.warning("⚠️ Cannot compute correlation matrix - no suitable backend available")
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to compute correlation matrix: {e}")
            
            # Compute covariance matrix
            if matrix_computations.get('covariance_matrix', True):
                try:
                    if PANDAS_AVAILABLE and hasattr(feature_data, 'cov'):
                        matrices['covariance_matrix'] = feature_data.cov().values
                    elif NUMPY_AVAILABLE:
                        matrices['covariance_matrix'] = np.cov(feature_data.T)
                    else:
                        self.logger.warning("⚠️ Cannot compute covariance matrix - no suitable backend available")
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to compute covariance matrix: {e}")
            
        except Exception as e:
            self.logger.error(f"❌ Error in matrix computation: {e}")
            # Return fallback results
            return self._compute_matrices_fallback(data, selected_features)
        
        return matrices
    
    def _compute_matrices_fallback(self, data: Any, selected_features: List[str]) -> Dict[str, Any]:
        """Fallback matrix computation using basic Python operations."""
        matrices = {}
        self.logger.info("🔄 Using fallback matrix computation")
        
        try:
            # Convert data to list of lists
            if hasattr(data, 'values'):
                matrix_data = data.values.tolist()
            elif hasattr(data, 'tolist'):
                matrix_data = data.tolist()
            elif isinstance(data, list):
                matrix_data = data
            else:
                self.logger.warning("⚠️ Cannot convert data to matrix format")
                return matrices
            
            if not matrix_data or len(matrix_data) == 0:
                self.logger.warning("⚠️ No data available for matrix computation")
                return matrices
            
            # Basic correlation computation
            n_features = len(matrix_data[0])
            corr_matrix = [[0.0 for _ in range(n_features)] for _ in range(n_features)]
            
            for i in range(n_features):
                for j in range(n_features):
                    if i == j:
                        corr_matrix[i][j] = 1.0
                    else:
                        # Extract columns
                        col_i = [row[i] for row in matrix_data]
                        col_j = [row[j] for row in matrix_data]
                        
                        # Compute basic correlation
                        corr_matrix[i][j] = compute_basic_correlation(col_i, col_j)
            
            matrices['correlation_matrix'] = corr_matrix
            self.logger.info(f"✅ Computed fallback correlation matrix: {len(corr_matrix)}x{len(corr_matrix[0])}")
            
        except Exception as e:
            self.logger.error(f"❌ Error in fallback matrix computation: {e}")
        
        return matrices
    
    def validate_inputs(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate inputs."""
        errors = []
        
        # Check for data
        has_data = (
            'engineered_data' in pipeline_state or
            'advanced_features' in pipeline_state or
            any(f'{split}_data' in pipeline_state for split in ['train', 'val', 'test'])
        )
        
        if not has_data:
            errors.append('No data available for processing')
        
        return len(errors) == 0, errors
    
    def get_required_inputs(self) -> List[str]:
        """Get required inputs."""
        return ['engineered_data or split data']
    
    def get_produced_outputs(self) -> List[str]:
        """Get produced outputs."""
        return ['matrix_results']

def create_step07_step(config: Dict[str, Any]) -> EnhancedMatrixOperationsStep:
    """Create a Step07 step instance."""
    return EnhancedMatrixOperationsStep(config)

def test_step07():
    """Test the enhanced Step07 implementation."""
    print("🧪 Testing Enhanced Step07")
    print("=" * 40)
    
    # Test dependency checking
    print("🔍 Testing dependency checking...")
    dependencies = check_step07_dependencies()
    capabilities = get_step07_capabilities()
    
    print(f"📊 Dependencies: {len([d for d in dependencies.values() if d])}/{len(dependencies)} available")
    print(f"🔧 Capabilities: {capabilities['status']} ({capabilities['overall_score']:.2%})")
    
    # Test step creation
    print("\n🚀 Testing step creation...")
    config = {
        'matrix_operations_config': {
            'use_gpu': False,
            'use_numba': False,
            'batch_size': 1000
        }
    }
    
    step = create_step07_step(config)
    print(f"✅ Step created: {step.step_name}")
    
    # Test matrix operations
    print("\n🧮 Testing matrix operations...")
    test_data = [
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0, 6.0],
        [4.0, 5.0, 6.0, 7.0],
        [5.0, 6.0, 7.0, 8.0]
    ]
    
    # Test fallback matrix computation
    matrices = step._compute_matrices_fallback(test_data, [])
    print(f"✅ Fallback matrix computation: {len(matrices)} matrices computed")
    
    if 'correlation_matrix' in matrices:
        corr_matrix = matrices['correlation_matrix']
        print(f"📊 Correlation matrix: {len(corr_matrix)}x{len(corr_matrix[0]) if corr_matrix else 0}")
    
    # Test basic correlation computation
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.0, 6.0, 8.0, 10.0]
    corr = compute_basic_correlation(x, y)
    print(f"📈 Basic correlation test: {corr:.3f} (expected: ~1.0)")
    
    # Test execution
    print("\n🚀 Testing step execution...")
    training_input = {'symbol': 'BTCUSDT', 'exchange': 'binance', 'timeframe': '1h'}
    pipeline_state = {'engineered_data': {'train': test_data}}
    
    import asyncio
    result = asyncio.run(step.execute(training_input, pipeline_state))
    
    if 'step07_enhanced_matrix_operations_completed' in result:
        print("✅ Step execution completed successfully")
        print(f"📊 Matrix results: {list(result.get('matrix_results', {}).keys())}")
    else:
        print("❌ Step execution failed")
    
    return True

if __name__ == "__main__":
    test_step07()