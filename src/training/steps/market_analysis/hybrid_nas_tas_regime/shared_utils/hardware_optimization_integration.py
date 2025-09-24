"""
Hardware Optimization Integration Module

This module integrates M1 hardware optimization utilities for enhanced performance
in the hybrid NAS-TAS regime detection system.

Integrated modules:
- src/utils/hardware/m1_gpu_utils.py
- src/utils/hardware/m1_memory_optimizer.py
- src/utils/hardware/m1_cpu_optimizer.py
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Add src to path for imports
src_path = Path(__file__).parents[4] / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

logger = logging.getLogger(__name__)

# =============================================================================
# M1 GPU UTILITIES INTEGRATION
# =============================================================================

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
            import platform
            if platform.system() != 'Darwin':
                return False

            # Check for Apple Silicon
            import subprocess
            result = subprocess.run(['sysctl', 'machdep.cpu.brand_string'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                brand = result.stdout.strip()
                if 'Apple' in brand and ('M1' in brand or 'M2' in brand or 'M3' in brand):
                    return True
            return False
        except Exception as e:
            self.logger.warning(f"Error detecting M1 hardware: {e}")
            return False

    def _check_mps_availability(self) -> bool:
        """Check if MPS (Metal Performance Shaders) is available."""
        try:
            if not self.is_m1:
                return False

            # Try to import torch and check for MPS
            try:
                import torch
                if torch.backends.mps.is_available():
                    return True
            except ImportError:
                pass

            # Check if Metal is available via system profiler
            try:
                import subprocess
                result = subprocess.run(['system_profiler', 'SPDisplaysDataType'],
                                      capture_output=True, text=True)
                if 'Metal' in result.stdout:
                    return True
            except:
                pass

            return False
        except Exception as e:
            self.logger.warning(f"Error checking MPS availability: {e}")
            return False

    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information."""
        try:
            if not self.is_m1:
                return {'type': 'none', 'available': False}

            info = {
                'type': 'm1_gpu',
                'available': True,
                'mps_available': self.mps_available,
                'memory_gb': self._get_gpu_memory(),
                'cores': self._get_gpu_cores()
            }

            return info
        except Exception as e:
            self.logger.error(f"Error getting GPU info: {e}")
            return {'type': 'error', 'available': False, 'error': str(e)}

    def _get_gpu_memory(self) -> float:
        """Get GPU memory in GB."""
        try:
            # Try to get GPU memory from system info
            import subprocess
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'],
                                  capture_output=True, text=True)

            # Parse for VRAM information
            if 'VRAM' in result.stdout:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'VRAM' in line:
                        # Extract number from line like "VRAM (Total): 8 GB"
                        parts = line.split()
                        for part in parts:
                            if part.isdigit():
                                return float(part)

            # Default M1 GPU memory
            return 8.0  # GB
        except Exception:
            return 8.0  # Default fallback

    def _get_gpu_cores(self) -> int:
        """Get number of GPU cores."""
        try:
            # Try to get GPU core count from system info
            import subprocess
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'],
                                  capture_output=True, text=True)

            # Parse for core information
            if 'GPU' in result.stdout:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'GPU' in line and ('core' in line.lower() or 'Core' in line):
                        # Extract number from line like "GPU Cores: 8"
                        parts = line.split()
                        for part in parts:
                            if part.isdigit():
                                return int(part)

            # Default M1 GPU cores
            return 8  # cores
        except Exception:
            return 8  # Default fallback

    def optimize_for_gpu(self, data: Any) -> Any:
        """Optimize data for GPU processing."""
        try:
            if not self.mps_available:
                return data

            # Try to move data to GPU if using PyTorch
            try:
                import torch
                if torch.is_tensor(data):
                    return data.to('mps')
            except ImportError:
                pass

            # Try to optimize numpy arrays
            try:
                import numpy as np
                if isinstance(data, np.ndarray):
                    # Ensure array is in optimal format for GPU
                    if data.dtype != np.float32:
                        data = data.astype(np.float32)
                    return data
            except ImportError:
                pass

            return data
        except Exception as e:
            self.logger.warning(f"Error optimizing data for GPU: {e}")
            return data

    def gpu_context(self, name: str):
        """Create a GPU context manager."""
        from contextlib import contextmanager

        @contextmanager
        def _gpu_context():
            try:
                if self.mps_available:
                    self.logger.info(f"🚀 Entering GPU context: {name}")

                start_time = None
                try:
                    import time
                    start_time = time.time()
                except:
                    pass

                try:
                    yield
                finally:
                    if start_time is not None:
                        try:
                            import time
                            duration = time.time() - start_time
                            self.logger.info(f"✅ GPU context {name} completed in {duration:.2f}s")
                        except:
                            pass
            except Exception as e:
                self.logger.error(f"❌ Error in GPU context {name}: {e}")
                raise

        return _gpu_context()

# =============================================================================
# M1 MEMORY OPTIMIZER INTEGRATION
# =============================================================================

class M1MemoryOptimizer:
    """Memory optimizer for M1 systems."""

    def __init__(self):
        self.is_monitoring = False
        self.logger = logger.getChild('M1MemoryOptimizer')

    def start_monitoring(self):
        """Start memory monitoring."""
        try:
            self.is_monitoring = True
            self.logger.info("🧠 Started memory monitoring")

            # Try to start background monitoring
            try:
                import threading
                import time
                import psutil

                def monitor_memory():
                    while self.is_monitoring:
                        try:
                            process = psutil.Process()
                            memory_info = process.memory_info()

                            # Log if memory usage is high
                            memory_mb = memory_info.rss / 1024 / 1024
                            if memory_mb > 1000:  # 1GB threshold
                                self.logger.warning(f"⚠️ High memory usage: {memory_mb:.1f} MB")
                        except Exception:
                            pass

                        time.sleep(5)  # Check every 5 seconds

                monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
                monitor_thread.start()

            except ImportError:
                self.logger.warning("⚠️ psutil not available for memory monitoring")

        except Exception as e:
            self.logger.error(f"Error starting memory monitoring: {e}")

    def stop_monitoring(self):
        """Stop memory monitoring."""
        try:
            self.is_monitoring = False
            self.logger.info("⏹️ Stopped memory monitoring")
        except Exception as e:
            self.logger.error(f"Error stopping memory monitoring: {e}")

    def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage."""
        try:
            import gc

            # Force garbage collection
            collected = gc.collect()

            # Get memory stats
            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024

                result = {
                    'objects_collected': collected,
                    'memory_mb': round(memory_mb, 2),
                    'method': 'm1_memory_optimizer',
                    'success': True
                }
            except ImportError:
                result = {
                    'objects_collected': collected,
                    'memory_mb': 0,
                    'method': 'basic_gc',
                    'success': True
                }

            self.logger.info(f"🧠 Memory optimization completed: {result}")
            return result

        except Exception as e:
            self.logger.error(f"Error optimizing memory: {e}")
            return {
                'error': str(e),
                'method': 'failed',
                'success': False
            }

    def memory_checkpoint(self, name: str):
        """Create a memory checkpoint context manager."""
        from contextlib import contextmanager

        @contextmanager
        def _memory_checkpoint():
            try:
                self.logger.info(f"💾 Memory checkpoint: {name}")

                # Get memory before
                memory_before = self._get_current_memory_mb()

                try:
                    yield
                finally:
                    # Get memory after
                    memory_after = self._get_current_memory_mb()
                    memory_diff = memory_after - memory_before

                    if memory_diff > 100:  # Log significant increases
                        self.logger.warning(f"⚠️ Memory increase at {name}: +{memory_diff:.1f} MB")
                    else:
                        self.logger.info(f"✅ Memory checkpoint {name}: {memory_diff:+.1f} MB")

            except Exception as e:
                self.logger.error(f"❌ Error in memory checkpoint {name}: {e}")
                raise

        return _memory_checkpoint()

    def _get_current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024
        except ImportError:
            return 0.0

# =============================================================================
# M1 CPU OPTIMIZER INTEGRATION
# =============================================================================

class M1CPUOptimizer:
    """CPU optimizer for M1 systems."""

    def __init__(self):
        self.logger = logger.getChild('M1CPUOptimizer')

    def get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information."""
        try:
            import platform
            import multiprocessing

            info = {
                'architecture': platform.machine(),
                'processor': platform.processor(),
                'cores_physical': multiprocessing.cpu_count(),
                'cores_logical': multiprocessing.cpu_count() * 2,  # M1 has 2 threads per core
                'is_m1': 'Apple' in platform.processor() if platform.processor() else False
            }

            # Try to get more detailed info
            try:
                import subprocess
                result = subprocess.run(['sysctl', 'machdep.cpu.brand_string'],
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    info['brand_string'] = result.stdout.strip()
            except:
                pass

            return info
        except Exception as e:
            self.logger.error(f"Error getting CPU info: {e}")
            return {'error': str(e)}

    def optimize_numpy_operations(self):
        """Optimize numpy for M1."""
        try:
            import numpy as np

            # Set optimal numpy settings for M1
            try:
                # Use all available cores
                import os
                os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())

                # Set numpy to use optimal BLAS
                os.environ['VECLIB_MAXIMUM_THREADS'] = str(multiprocessing.cpu_count())

                # Enable M1 optimizations
                os.environ['NUMEXPR_MAX_THREADS'] = str(multiprocessing.cpu_count())

                self.logger.info("✅ NumPy optimized for M1")

            except Exception as e:
                self.logger.warning(f"Warning during NumPy optimization: {e}")

        except ImportError:
            self.logger.warning("⚠️ NumPy not available for optimization")
        except Exception as e:
            self.logger.error(f"Error optimizing NumPy: {e}")

    def parallel_cpu_operation(self, func, data_list, max_workers=None):
        """Execute function in parallel on CPU."""
        try:
            import concurrent.futures
            import multiprocessing

            if max_workers is None:
                max_workers = multiprocessing.cpu_count()

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(func, data_list))

            return results

        except Exception as e:
            self.logger.error(f"Error in parallel CPU operation: {e}")
            return [func(data) for data in data_list]

# =============================================================================
# MAIN INTEGRATION FUNCTIONS
# =============================================================================

def get_m1_gpu_manager():
    """Get the M1 GPU manager instance."""
    try:
        from utils.hardware.m1_gpu_utils import get_m1_gpu_manager as _get_m1_gpu_manager
        return _get_m1_gpu_manager()
    except ImportError:
        return M1GPUManager()

def get_m1_memory_optimizer():
    """Get the M1 memory optimizer instance."""
    try:
        from utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer as _get_m1_memory_optimizer
        return _get_m1_memory_optimizer()
    except ImportError:
        return M1MemoryOptimizer()

def get_m1_cpu_optimizer():
    """Get the M1 CPU optimizer instance."""
    try:
        from utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer as _get_m1_cpu_optimizer
        return _get_m1_cpu_optimizer()
    except ImportError:
        return M1CPUOptimizer()

def cleanup_m1_optimizers():
    """Clean up M1 optimizers and release resources."""
    try:
        # Try external cleanup first
        from utils.hardware.m1_gpu_utils import get_m1_gpu_manager
        from utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
        from utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer

        # Get instances
        gpu_manager = get_m1_gpu_manager()
        memory_optimizer = get_m1_memory_optimizer()
        cpu_optimizer = get_m1_cpu_optimizer()

        # Clean up resources
        if memory_optimizer and hasattr(memory_optimizer, 'stop_monitoring'):
            memory_optimizer.stop_monitoring()

        # Log cleanup
        logger.info("🧠 M1 optimizers cleaned up successfully")
        return True

    except ImportError:
        # Fallback cleanup
        try:
            memory_optimizer = M1MemoryOptimizer()
            memory_optimizer.stop_monitoring()
            logger.info("🧠 M1 optimizers cleaned up successfully (fallback)")
            return True
        except Exception as e:
            logger.error(f"❌ Error during M1 optimizer cleanup: {e}")
            return False
    except Exception as e:
        logger.error(f"❌ Error during M1 optimizer cleanup: {e}")
        return False

def integrate_with_m1_optimizers() -> dict:
    """Integrate with M1 GPU and CPU optimizers."""
    try:
        # Try external integration first
        from utils.hardware.m1_gpu_utils import get_m1_gpu_manager, is_m1_available, is_mps_available
        from utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer, start_m1_memory_monitoring
        from utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer

        # Initialize components
        gpu_manager = get_m1_gpu_manager()
        memory_optimizer = get_m1_memory_optimizer()
        cpu_optimizer = get_m1_cpu_optimizer()

        # Start memory monitoring
        if hasattr(memory_optimizer, 'start_monitoring'):
            memory_optimizer.start_monitoring()

        # Optimize numpy for M1
        cpu_optimizer.optimize_numpy_operations()

        # Log integration status
        gpu_info = gpu_manager.get_gpu_info() if hasattr(gpu_manager, 'get_gpu_info') else {}
        cpu_info = cpu_optimizer.get_cpu_info() if hasattr(cpu_optimizer, 'get_cpu_info') else {}

        logger.info("🧠 M1 Integration Status:")
        logger.info(f"   - M1 Hardware: {'✅ Available' if is_m1_available() else '❌ Not available'}")
        logger.info(f"   - MPS (GPU): {'✅ Available' if is_mps_available() else '❌ Not available'}")
        logger.info(f"   - Performance Cores: {cpu_info.get('cores_physical', 'Unknown')}")
        logger.info(f"   - Memory Monitoring: ✅ Active")

        return {
            'integration_status': 'success',
            'gpu_manager': is_mps_available(),
            'memory_optimizer': True,
            'cpu_optimizer': True,
            'gpu_info': gpu_info,
            'cpu_info': cpu_info,
            'success': True
        }

    except ImportError as e:
        logger.warning(f"⚠️ External M1 utilities not available: {e}")
        # Fallback integration
        try:
            gpu_manager = M1GPUManager()
            memory_optimizer = M1MemoryOptimizer()
            cpu_optimizer = M1CPUOptimizer()

            memory_optimizer.start_monitoring()
            cpu_optimizer.optimize_numpy_operations()

            gpu_info = gpu_manager.get_gpu_info()
            cpu_info = cpu_optimizer.get_cpu_info()

            logger.info("🧠 M1 Integration Status (Fallback):")
            logger.info(f"   - M1 Hardware: {'✅ Available' if gpu_manager.is_m1 else '❌ Not available'}")
            logger.info(f"   - MPS (GPU): {'✅ Available' if gpu_manager.mps_available else '❌ Not available'}")
            logger.info(f"   - Performance Cores: {cpu_info.get('cores_physical', 'Unknown')}")
            logger.info(f"   - Memory Monitoring: ✅ Active")

            return {
                'integration_status': 'fallback_success',
                'gpu_manager': gpu_manager.mps_available,
                'memory_optimizer': True,
                'cpu_optimizer': True,
                'gpu_info': gpu_info,
                'cpu_info': cpu_info,
                'success': True
            }

        except Exception as e:
            logger.error(f"❌ M1 integration failed: {e}")
            return {
                'integration_status': 'failed',
                'error': str(e),
                'gpu_manager': False,
                'memory_optimizer': False,
                'cpu_optimizer': False,
                'success': False
            }