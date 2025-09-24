"""
Matrix Operations Integration Module

This module integrates matrix operations utilities from src/utils/matrix_operations/
for efficient matrix computations and hardware optimization in the hybrid NAS-TAS
regime detection system.

Integrated modules:
- src/utils/matrix_operations/unified_operations.py
- src/utils/matrix_operations/batch_operations.py
- src/utils/matrix_operations/computation_toolbox.py
- src/utils/matrix_operations/hardware_integration.py
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np

# Add src to path for imports
src_path = Path(__file__).parents[4] / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

logger = logging.getLogger(__name__)

# =============================================================================
# UNIFIED MATRIX OPERATIONS INTEGRATION
# =============================================================================

class UnifiedMatrixOperations:
    """Unified interface for matrix operations."""

    def __init__(self):
        self.logger = logger.getChild('UnifiedMatrixOperations')

    def matrix_multiply(self, A: np.ndarray, B: np.ndarray, optimize: bool = True) -> np.ndarray:
        """Matrix multiplication with optimization."""
        try:
            # Try to use external unified operations
            from utils.matrix_operations.unified_operations import matrix_multiply as _matrix_multiply
            return _matrix_multiply(A, B, optimize)
        except ImportError:
            # Fallback implementation
            try:
                if optimize:
                    # Use BLAS if available
                    try:
                        return np.dot(A, B)
                    except:
                        return A @ B
                else:
                    return A @ B
            except Exception as e:
                self.logger.error(f"Error in matrix multiplication: {e}")
                return np.zeros((A.shape[0], B.shape[1]))

    def batch_matrix_multiply(self, matrices_A: List[np.ndarray],
                           matrices_B: List[np.ndarray]) -> List[np.ndarray]:
        """Batch matrix multiplication."""
        try:
            # Try to use external batch operations
            from utils.matrix_operations.batch_operations import batch_matrix_multiply as _batch_matrix_multiply
            return _batch_matrix_multiply(matrices_A, matrices_B)
        except ImportError:
            # Fallback implementation
            try:
                results = []
                for A, B in zip(matrices_A, matrices_B):
                    try:
                        result = self.matrix_multiply(A, B)
                        results.append(result)
                    except Exception as e:
                        self.logger.warning(f"Error in batch multiply: {e}")
                        # Return zero matrix of appropriate shape
                        results.append(np.zeros((A.shape[0], B.shape[1])))
                return results
            except Exception as e:
                self.logger.error(f"Error in batch matrix multiplication: {e}")
                return []

    def inverse(self, matrix: np.ndarray) -> np.ndarray:
        """Matrix inverse with error handling."""
        try:
            # Try to use external computation toolbox
            from utils.matrix_operations.computation_toolbox import safe_matrix_inverse as _safe_matrix_inverse
            return _safe_matrix_inverse(matrix)
        except ImportError:
            # Fallback implementation
            try:
                import numpy as np
                if matrix.shape[0] != matrix.shape[1]:
                    raise ValueError("Matrix must be square")

                # Use pseudo-inverse for better stability
                return np.linalg.pinv(matrix)
            except Exception as e:
                self.logger.error(f"Error computing matrix inverse: {e}")
                return np.eye(matrix.shape[0])

    def eigenvalues(self, matrix: np.ndarray) -> np.ndarray:
        """Compute eigenvalues with error handling."""
        try:
            # Try to use external computation toolbox
            from utils.matrix_operations.computation_toolbox import safe_eigenvalues as _safe_eigenvalues
            return _safe_eigenvalues(matrix)
        except ImportError:
            # Fallback implementation
            try:
                import numpy as np
                if matrix.shape[0] != matrix.shape[1]:
                    raise ValueError("Matrix must be square")

                eigenvals = np.linalg.eigvals(matrix)

                # Filter out complex eigenvalues (take real part)
                if np.iscomplexobj(eigenvals):
                    eigenvals = eigenvals.real

                return eigenvals
            except Exception as e:
                self.logger.error(f"Error computing eigenvalues: {e}")
                return np.array([])

    def eigenvectors(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute eigenvalues and eigenvectors."""
        try:
            # Try to use external computation toolbox
            from utils.matrix_operations.computation_toolbox import safe_eigenvectors as _safe_eigenvectors
            return _safe_eigenvectors(matrix)
        except ImportError:
            # Fallback implementation
            try:
                import numpy as np
                if matrix.shape[0] != matrix.shape[1]:
                    raise ValueError("Matrix must be square")

                eigenvals, eigenvecs = np.linalg.eig(matrix)

                # Filter out complex values (take real part)
                if np.iscomplexobj(eigenvals):
                    eigenvals = eigenvals.real
                if np.iscomplexobj(eigenvecs):
                    eigenvecs = eigenvecs.real

                return eigenvals, eigenvecs
            except Exception as e:
                self.logger.error(f"Error computing eigenvectors: {e}")
                return np.array([]), np.array([])

    def cholesky_decomposition(self, matrix: np.ndarray) -> np.ndarray:
        """Cholesky decomposition for positive definite matrices."""
        try:
            # Try to use external computation toolbox
            from utils.matrix_operations.computation_toolbox import cholesky_decomposition as _cholesky_decomposition
            return _cholesky_decomposition(matrix)
        except ImportError:
            # Fallback implementation
            try:
                import numpy as np
                if matrix.shape[0] != matrix.shape[1]:
                    raise ValueError("Matrix must be square")

                # Check if matrix is positive definite
                if not np.allclose(matrix, matrix.T):
                    raise ValueError("Matrix must be symmetric")

                L = np.linalg.cholesky(matrix)
                return L
            except Exception as e:
                self.logger.error(f"Error in Cholesky decomposition: {e}")
                return np.array([])

    def svd(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Singular Value Decomposition."""
        try:
            # Try to use external computation toolbox
            from utils.matrix_operations.computation_toolbox import safe_svd as _safe_svd
            return _safe_svd(matrix)
        except ImportError:
            # Fallback implementation
            try:
                import numpy as np
                U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
                return U, s, Vt
            except Exception as e:
                self.logger.error(f"Error in SVD: {e}")
                return np.array([]), np.array([]), np.array([])

    def correlation_matrix(self, data: np.ndarray) -> np.ndarray:
        """Compute correlation matrix."""
        try:
            # Try to use external computation toolbox
            from utils.matrix_operations.computation_toolbox import correlation_matrix as _correlation_matrix
            return _correlation_matrix(data)
        except ImportError:
            # Fallback implementation
            try:
                import numpy as np
                if data.ndim == 1:
                    data = data.reshape(-1, 1)

                # Remove NaN and infinite values
                data_clean = data[~np.isnan(data).any(axis=1)]
                data_clean = data_clean[~np.isinf(data_clean).any(axis=1)]

                if data_clean.shape[0] < 2:
                    return np.eye(data.shape[1])

                corr_matrix = np.corrcoef(data_clean, rowvar=False)

                # Ensure symmetry
                corr_matrix = (corr_matrix + corr_matrix.T) / 2

                # Clip to valid correlation range
                corr_matrix = np.clip(corr_matrix, -1, 1)

                return corr_matrix
            except Exception as e:
                self.logger.error(f"Error computing correlation matrix: {e}")
                return np.eye(data.shape[1] if data.ndim > 1 else 1)

    def covariance_matrix(self, data: np.ndarray) -> np.ndarray:
        """Compute covariance matrix."""
        try:
            # Try to use external computation toolbox
            from utils.matrix_operations.computation_toolbox import covariance_matrix as _covariance_matrix
            return _covariance_matrix(data)
        except ImportError:
            # Fallback implementation
            try:
                import numpy as np
                if data.ndim == 1:
                    data = data.reshape(-1, 1)

                # Remove NaN and infinite values
                data_clean = data[~np.isnan(data).any(axis=1)]
                data_clean = data_clean[~np.isinf(data_clean).any(axis=1)]

                if data_clean.shape[0] < 2:
                    return np.eye(data.shape[1])

                cov_matrix = np.cov(data_clean, rowvar=False)

                return cov_matrix
            except Exception as e:
                self.logger.error(f"Error computing covariance matrix: {e}")
                return np.eye(data.shape[1] if data.ndim > 1 else 1)

# =============================================================================
# BATCH OPERATIONS INTEGRATION
# =============================================================================

class BatchMatrixOperations:
    """Batch matrix operations for efficient processing."""

    def __init__(self):
        self.logger = logger.getChild('BatchMatrixOperations')

    def batch_correlation(self, data_list: List[np.ndarray],
                         chunk_size: int = 1000) -> List[np.ndarray]:
        """Compute correlation matrices for a list of datasets."""
        try:
            # Try to use external batch operations
            from utils.matrix_operations.batch_operations import batch_correlation as _batch_correlation
            return _batch_correlation(data_list, chunk_size)
        except ImportError:
            # Fallback implementation
            try:
                results = []
                matrix_ops = UnifiedMatrixOperations()

                for data in data_list:
                    try:
                        corr_matrix = matrix_ops.correlation_matrix(data)
                        results.append(corr_matrix)
                    except Exception as e:
                        self.logger.warning(f"Error in batch correlation: {e}")
                        # Return identity matrix as fallback
                        n_features = data.shape[1] if data.ndim > 1 else 1
                        results.append(np.eye(n_features))

                return results
            except Exception as e:
                self.logger.error(f"Error in batch correlation: {e}")
                return []

    def batch_eigenvalues(self, matrices: List[np.ndarray]) -> List[np.ndarray]:
        """Compute eigenvalues for a batch of matrices."""
        try:
            # Try to use external batch operations
            from utils.matrix_operations.batch_operations import batch_eigenvalues as _batch_eigenvalues
            return _batch_eigenvalues(matrices)
        except ImportError:
            # Fallback implementation
            try:
                results = []
                matrix_ops = UnifiedMatrixOperations()

                for matrix in matrices:
                    try:
                        eigenvals = matrix_ops.eigenvalues(matrix)
                        results.append(eigenvals)
                    except Exception as e:
                        self.logger.warning(f"Error in batch eigenvalues: {e}")
                        results.append(np.array([]))

                return results
            except Exception as e:
                self.logger.error(f"Error in batch eigenvalues: {e}")
                return []

    def parallel_matrix_operations(self, matrices: List[np.ndarray],
                                 operation: str = 'inverse') -> List[np.ndarray]:
        """Perform matrix operations in parallel."""
        try:
            # Try to use external batch operations
            from utils.matrix_operations.batch_operations import parallel_matrix_operations as _parallel_matrix_operations
            return _parallel_matrix_operations(matrices, operation)
        except ImportError:
            # Fallback implementation using threading
            try:
                import concurrent.futures
                import multiprocessing

                matrix_ops = UnifiedMatrixOperations()
                results = []

                def process_matrix(matrix):
                    try:
                        if operation == 'inverse':
                            return matrix_ops.inverse(matrix)
                        elif operation == 'eigenvalues':
                            return matrix_ops.eigenvalues(matrix)
                        elif operation == 'correlation':
                            # For correlation, expect 2D data
                            if matrix.ndim == 2:
                                return matrix_ops.correlation_matrix(matrix)
                            else:
                                return matrix_ops.correlation_matrix(matrix.reshape(-1, 1))
                        else:
                            raise ValueError(f"Unsupported operation: {operation}")
                    except Exception as e:
                        self.logger.warning(f"Error in parallel operation: {e}")
                        return None

                max_workers = min(len(matrices), multiprocessing.cpu_count())
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    results = list(executor.map(process_matrix, matrices))

                return results
            except Exception as e:
                self.logger.error(f"Error in parallel matrix operations: {e}")
                return []

# =============================================================================
# HARDWARE INTEGRATION
# =============================================================================

class HardwareAcceleratedMatrixOps:
    """Hardware-accelerated matrix operations."""

    def __init__(self):
        self.logger = logger.getChild('HardwareAcceleratedMatrixOps')

    def detect_hardware_acceleration(self) -> Dict[str, Any]:
        """Detect available hardware acceleration."""
        try:
            # Try to use external hardware integration
            from utils.matrix_operations.hardware_integration import detect_hardware_acceleration as _detect_hardware_acceleration
            return _detect_hardware_acceleration()
        except ImportError:
            # Fallback implementation
            try:
                import platform
                import multiprocessing

                hardware_info = {
                    'cpu_cores': multiprocessing.cpu_count(),
                    'architecture': platform.machine(),
                    'mps_available': False,
                    'cuda_available': False,
                    'opencl_available': False,
                    'mkl_available': False
                }

                # Check for M1/MPS
                if platform.system() == 'Darwin' and 'arm' in platform.machine():
                    try:
                        import subprocess
                        result = subprocess.run(['sysctl', 'machdep.cpu.brand_string'],
                                              capture_output=True, text=True)
                        if 'Apple' in result.stdout and ('M1' in result.stdout or 'M2' in result.stdout or 'M3' in result.stdout):
                            hardware_info['mps_available'] = True
                            hardware_info['hardware_type'] = 'm1_mps'
                    except:
                        pass

                # Check for CUDA
                try:
                    import torch
                    hardware_info['cuda_available'] = torch.cuda.is_available()
                    if hardware_info['cuda_available']:
                        hardware_info['cuda_devices'] = torch.cuda.device_count()
                        hardware_info['hardware_type'] = 'cuda'
                except ImportError:
                    pass

                # Check for MKL
                try:
                    import numpy as np
                    # MKL usually sets this environment variable
                    import os
                    if 'MKL_NUM_THREADS' in os.environ or 'mkl' in np.__version__.lower():
                        hardware_info['mkl_available'] = True
                        hardware_info['hardware_type'] = 'mkl_cpu'
                except:
                    pass

                return hardware_info
            except Exception as e:
                self.logger.error(f"Error detecting hardware acceleration: {e}")
                return {'error': str(e)}

    def optimize_for_hardware(self, matrix: np.ndarray) -> np.ndarray:
        """Optimize matrix for hardware acceleration."""
        try:
            # Try to use external hardware integration
            from utils.matrix_operations.hardware_integration import optimize_for_hardware as _optimize_for_hardware
            return _optimize_for_hardware(matrix)
        except ImportError:
            # Fallback implementation
            try:
                hardware_info = self.detect_hardware_acceleration()

                # Optimize based on available hardware
                if hardware_info.get('mps_available'):
                    # Optimize for M1 MPS
                    if matrix.dtype != np.float32:
                        matrix = matrix.astype(np.float32)

                elif hardware_info.get('cuda_available'):
                    # Optimize for CUDA
                    if matrix.dtype != np.float32:
                        matrix = matrix.astype(np.float32)

                elif hardware_info.get('mkl_available'):
                    # MKL optimizations are usually automatic
                    pass

                return matrix
            except Exception as e:
                self.logger.error(f"Error optimizing for hardware: {e}")
                return matrix

# =============================================================================
# MAIN INTEGRATION FUNCTIONS
# =============================================================================

def get_unified_matrix_operations():
    """Get unified matrix operations instance."""
    try:
        from utils.matrix_operations.unified_operations import UnifiedMatrixOperations as _UnifiedMatrixOperations
        return _UnifiedMatrixOperations()
    except ImportError:
        return UnifiedMatrixOperations()

def get_batch_matrix_operations():
    """Get batch matrix operations instance."""
    try:
        from utils.matrix_operations.batch_operations import BatchMatrixOperations as _BatchMatrixOperations
        return _BatchMatrixOperations()
    except ImportError:
        return BatchMatrixOperations()

def get_hardware_accelerated_matrix_ops():
    """Get hardware accelerated matrix operations instance."""
    try:
        from utils.matrix_operations.hardware_integration import HardwareAcceleratedMatrixOps as _HardwareAcceleratedMatrixOps
        return _HardwareAcceleratedMatrixOps()
    except ImportError:
        return HardwareAcceleratedMatrixOps()

def compute_correlation_matrix(data: np.ndarray) -> np.ndarray:
    """Compute correlation matrix with hardware optimization."""
    try:
        ops = get_unified_matrix_operations()
        hardware_ops = get_hardware_accelerated_matrix_ops()

        # Optimize data for hardware
        optimized_data = hardware_ops.optimize_for_hardware(data)

        # Compute correlation matrix
        return ops.correlation_matrix(optimized_data)
    except Exception as e:
        logger.error(f"Error computing correlation matrix: {e}")
        # Fallback to basic numpy
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return np.corrcoef(data, rowvar=False)

def compute_covariance_matrix(data: np.ndarray) -> np.ndarray:
    """Compute covariance matrix with hardware optimization."""
    try:
        ops = get_unified_matrix_operations()
        hardware_ops = get_hardware_accelerated_matrix_ops()

        # Optimize data for hardware
        optimized_data = hardware_ops.optimize_for_hardware(data)

        # Compute covariance matrix
        return ops.covariance_matrix(optimized_data)
    except Exception as e:
        logger.error(f"Error computing covariance matrix: {e}")
        # Fallback to basic numpy
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return np.cov(data, rowvar=False)

def batch_compute_correlation_matrices(data_list: List[np.ndarray]) -> List[np.ndarray]:
    """Compute correlation matrices for a batch of datasets."""
    try:
        batch_ops = get_batch_matrix_operations()
        return batch_ops.batch_correlation(data_list)
    except Exception as e:
        logger.error(f"Error in batch correlation computation: {e}")
        # Fallback implementation
        ops = get_unified_matrix_operations()
        results = []
        for data in data_list:
            try:
                corr_matrix = ops.correlation_matrix(data)
                results.append(corr_matrix)
            except Exception:
                # Return identity matrix as fallback
                n_features = data.shape[1] if data.ndim > 1 else 1
                results.append(np.eye(n_features))
        return results

def parallel_matrix_computation(matrices: List[np.ndarray], operation: str = 'inverse') -> List[np.ndarray]:
    """Perform matrix operations in parallel."""
    try:
        batch_ops = get_batch_matrix_operations()
        return batch_ops.parallel_matrix_operations(matrices, operation)
    except Exception as e:
        logger.error(f"Error in parallel matrix computation: {e}")
        # Fallback implementation
        ops = get_unified_matrix_operations()
        results = []
        for matrix in matrices:
            try:
                if operation == 'inverse':
                    result = ops.inverse(matrix)
                elif operation == 'eigenvalues':
                    result = ops.eigenvalues(matrix)
                else:
                    raise ValueError(f"Unsupported operation: {operation}")
                results.append(result)
            except Exception:
                results.append(None)
        return results