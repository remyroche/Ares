"""
Thread/BLAS clamping utilities to prevent CPU oversubscription and nested
parallelism during CV and HPO. Safe for macOS (including Apple Silicon).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

try:
    from threadpoolctl import threadpool_limits
    THREADPOOLCTL_AVAILABLE = True
except Exception:
    THREADPOOLCTL_AVAILABLE = False
    threadpool_limits = None  # type: ignore


@contextmanager
def limit_blas_threads(num_threads: int = 1) -> Iterator[None]:
    """Context manager to limit BLAS/OpenMP library threads.

    Falls back to environment variables if threadpoolctl is unavailable.
    """
    original_env = {
        'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
        'OPENBLAS_NUM_THREADS': os.environ.get('OPENBLAS_NUM_THREADS'),
        'MKL_NUM_THREADS': os.environ.get('MKL_NUM_THREADS'),
        'VECLIB_MAXIMUM_THREADS': os.environ.get('VECLIB_MAXIMUM_THREADS'),
        'NUMEXPR_NUM_THREADS': os.environ.get('NUMEXPR_NUM_THREADS')
    }

    try:
        if THREADPOOLCTL_AVAILABLE:
            with threadpool_limits(limits=num_threads):  # type: ignore
                yield
        else:
            # Best-effort environment clamp
            os.environ['OMP_NUM_THREADS'] = str(num_threads)
            os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
            os.environ['MKL_NUM_THREADS'] = str(num_threads)
            os.environ['VECLIB_MAXIMUM_THREADS'] = str(num_threads)
            os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)
            yield
    finally:
        # Restore environment
        for k, v in original_env.items():
            if v is None:
                if k in os.environ:
                    del os.environ[k]
            else:
                os.environ[k] = v

