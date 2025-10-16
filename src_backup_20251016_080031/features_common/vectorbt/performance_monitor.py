"""
VectorBT Performance Monitor

This module provides performance monitoring for VectorBT operations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List
import logging
import time

logger = logging.getLogger(__name__)

class VectorBTPerformanceMonitor:
    """Performance monitor for VectorBT operations."""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = None
    
    def start_monitoring(self, operation: str):
        """Start monitoring an operation."""
        self.start_time = time.time()
        self.metrics[operation] = {'start_time': self.start_time}
    
    def stop_monitoring(self, operation: str):
        """Stop monitoring an operation."""
        if self.start_time:
            duration = time.time() - self.start_time
            self.metrics[operation]['duration'] = duration
            self.metrics[operation]['end_time'] = time.time()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.metrics

def get_performance_monitor() -> VectorBTPerformanceMonitor:
    """Get the performance monitor instance."""
    return VectorBTPerformanceMonitor()
