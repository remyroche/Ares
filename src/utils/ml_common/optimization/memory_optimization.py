"""
ML Common - Optimization Memory Optimization Module

This module provides memory optimization utilities for optimization.
"""

from ...memory_management.streaming_data_processor import StreamingDataProcessor


class MemoryEfficientTraining:
    """Memory-efficient training utilities."""

    def __init__(self):
        self.streaming_processor = StreamingDataProcessor()

    def optimize_memory_usage(self, data, batch_size=1000):
        """Optimize memory usage for large datasets."""
        # Simple implementation
        return self.streaming_processor.process_in_chunks(data, batch_size)

    def cleanup_memory(self):
        """Clean up memory after training."""
        import gc
        gc.collect()


__all__ = ['MemoryEfficientTraining']
