"""
Step 12 Modular: Device Utilities

This module provides device detection and management utilities.
"""

import queue
import threading

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..base.logger import setup_step12_logger
from ..base.utils import error, failed, timeout

logger = setup_step12_logger()

def safe_get_device() -> str:
    """Safely determine the best device to use with timeout protection."""
    if not TORCH_AVAILABLE:
        return 'cpu'

    try:
        result_queue: 'queue.Queue[tuple[str, Optional[Exception]]]' = queue.Queue()

        def check_mps() -> None:
            try:
                is_available = torch.backends.mps.is_available()
                result_queue.put(('mps' if is_available else 'cpu', None))
            except Exception as e:
                result_queue.put(('cpu', e))

        thread = threading.Thread(target=check_mps)
        thread.daemon = True
        thread.start()

        try:
            device, err = result_queue.get(timeout=10)
            if err:
                logger.error(failed(f'MPS check failed: {err}, using CPU'))
                return 'cpu'
            return device
        except queue.Empty:
            logger.exception(timeout('MPS availability check timed out, using CPU'))
            return 'cpu'

    except Exception as e:
        logger.exception(error(f'Error checking MPS availability: {e}, using CPU'))
        return 'cpu'

__all__ = ['safe_get_device']
