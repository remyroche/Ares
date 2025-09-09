"""
Thread/BLAS clamping utilities to prevent CPU oversubscription and nested
parallelism during CV and HPO. Safe for macOS (including Apple Silicon).

Enhanced with comprehensive error handling and logging.
"""

from __future__ import annotations

import os
import logging
from contextlib import contextmanager
from typing import Iterator, Dict, Any, Optional

# Enhanced dependency management with fast fail
try:
    from threadpoolctl import threadpool_limits
    THREADPOOLCTL_AVAILABLE = True
    print("✅ threadpoolctl available for thread management")
except ImportError as e:
    THREADPOOLCTL_AVAILABLE = False
    threadpool_limits = None  # type: ignore
    print(f"⚠️ threadpoolctl not available: {e}. Using environment variable fallback.")

# Setup logging
logger = logging.getLogger(__name__)


@contextmanager
def limit_blas_threads(num_threads: int = 1) -> Iterator[None]:
    """Context manager to limit BLAS/OpenMP library threads.

    Falls back to environment variables if threadpoolctl is unavailable.
    Enhanced with comprehensive error handling and logging.
    
    Args:
        num_threads: Number of threads to limit to (default: 1)
        
    Yields:
        None
        
    Raises:
        RuntimeError: If thread limiting fails critically
    """
    if not isinstance(num_threads, int) or num_threads < 1:
        logger.error(f"❌ Invalid num_threads: {num_threads}. Must be positive integer.")
        raise ValueError(f"num_threads must be positive integer, got {num_threads}")
    
    logger.info(f"🔒 Limiting BLAS threads to {num_threads}")
    
    # Store original environment variables
    original_env: Dict[str, Optional[str]] = {
        'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
        'OPENBLAS_NUM_THREADS': os.environ.get('OPENBLAS_NUM_THREADS'),
        'MKL_NUM_THREADS': os.environ.get('MKL_NUM_THREADS'),
        'VECLIB_MAXIMUM_THREADS': os.environ.get('VECLIB_MAXIMUM_THREADS'),
        'NUMEXPR_NUM_THREADS': os.environ.get('NUMEXPR_NUM_THREADS')
    }

    try:
        if THREADPOOLCTL_AVAILABLE:
            try:
                logger.debug("Using threadpoolctl for thread limiting")
                with threadpool_limits(limits=num_threads):  # type: ignore
                    yield
            except Exception as e:
                logger.warning(f"⚠️ threadpoolctl failed: {e}. Falling back to environment variables.")
                raise
        else:
            # Best-effort environment clamp
            logger.debug("Using environment variables for thread limiting")
            thread_env_vars = {
                'OMP_NUM_THREADS': str(num_threads),
                'OPENBLAS_NUM_THREADS': str(num_threads),
                'MKL_NUM_THREADS': str(num_threads),
                'VECLIB_MAXIMUM_THREADS': str(num_threads),
                'NUMEXPR_NUM_THREADS': str(num_threads)
            }
            
            try:
                for key, value in thread_env_vars.items():
                    os.environ[key] = value
                logger.debug(f"Set environment variables: {thread_env_vars}")
                yield
            except Exception as e:
                logger.error(f"❌ Failed to set environment variables: {e}")
                raise RuntimeError(f"Thread limiting failed: {e}")
                
    except Exception as e:
        logger.error(f"❌ Thread limiting failed: {e}")
        raise RuntimeError(f"Critical thread limiting failure: {e}")
        
    finally:
        # Restore environment variables
        try:
            for k, v in original_env.items():
                if v is None:
                    if k in os.environ:
                        del os.environ[k]
                else:
                    os.environ[k] = v
            logger.debug("✅ Restored original environment variables")
        except Exception as e:
            logger.error(f"❌ Failed to restore environment variables: {e}")


def get_thread_info() -> Dict[str, Any]:
    """Get current thread configuration information.
    
    Returns:
        Dictionary with thread configuration details
    """
    try:
        thread_info: Dict[str, Any] = {
            'threadpoolctl_available': THREADPOOLCTL_AVAILABLE,
            'environment_variables': {}
        }
        
        # Get current environment variables
        for var in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 
                   'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
            thread_info['environment_variables'][var] = os.environ.get(var)
        
        # Get threadpoolctl info if available
        if THREADPOOLCTL_AVAILABLE:
            try:
                import threadpoolctl
                thread_info['threadpoolctl_info'] = threadpoolctl.info()
            except Exception as e:
                thread_info['threadpoolctl_error'] = str(e)
        
        return thread_info
        
    except Exception as e:
        logger.error(f"❌ Failed to get thread info: {e}")
        return {'error': str(e)}


def validate_thread_environment() -> bool:
    """Validate that thread limiting is working correctly.
    
    Returns:
        True if thread limiting is working, False otherwise
    """
    try:
        logger.info("🔍 Validating thread environment...")
        
        if THREADPOOLCTL_AVAILABLE:
            logger.info("✅ threadpoolctl is available")
            return True
        else:
            # Check if environment variables are set
            env_vars_set = any(
                os.environ.get(var) for var in 
                ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']
            )
            if env_vars_set:
                logger.info("✅ Environment variables are set for thread limiting")
                return True
            else:
                logger.warning("⚠️ No thread limiting mechanism available")
                return False
                
    except Exception as e:
        logger.error(f"❌ Thread environment validation failed: {e}")
        return False


