"""
Random Seed Management for Unified Data-Driven Pipeline.

This module provides comprehensive random seed management to ensure
reproducibility across all stochastic operations in the pipeline.
"""

import random
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
import logging
import os

try:
    from src.utils.tprint import tprint, tprint_error, tprint_warning, tprint_success, tprint_debug
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)


class RandomSeedManager:
    """
    Comprehensive random seed management for reproducible results.
    
    Manages seeds for all random number generators used in the pipeline
    including Python's random, NumPy, and other libraries.
    """
    
    def __init__(self, base_seed: int = 42, config: Optional[Dict[str, Any]] = None):
        """Initialize the random seed manager."""
        self.base_seed = base_seed
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Seed tracking
        self.seed_history = []
        self.current_seed = base_seed
        self.seed_offset = 0
        
        # Configuration
        self.enable_reproducibility = self.config.get('enable_reproducibility', True)
        self.seed_increment = self.config.get('seed_increment', 1)
        self.track_seed_usage = self.config.get('track_seed_usage', True)
        
        # Initialize with base seed
        if self.enable_reproducibility:
            self.set_global_seed(base_seed)
        
        tprint_success(f"✅ RandomSeedManager initialized with base seed: {base_seed}")

    def set_global_seed(self, seed: int) -> None:
        """Set global random seed for all random number generators."""
        if not self.enable_reproducibility:
            tprint_warning("⚠️ Reproducibility disabled, skipping seed setting")
            return
        
        try:
            # Set Python's random seed
            random.seed(seed)
            
            # Set NumPy's random seed
            np.random.seed(seed)
            
            # Set pandas random seed (if available)
            if hasattr(pd, 'set_option'):
                pd.set_option('mode.chained_assignment', None)
            
            # Set environment variable for other libraries
            os.environ['PYTHONHASHSEED'] = str(seed)
            
            # Update current seed
            self.current_seed = seed
            
            # Track seed usage
            if self.track_seed_usage:
                self.seed_history.append({
                    'seed': seed,
                    'operation': 'global_set',
                    'timestamp': pd.Timestamp.now()
                })
            
            tprint_success(f"✅ Global seed set to: {seed}")
            
        except Exception as e:
            tprint_error(f"❌ Error setting global seed: {e}")

    def get_next_seed(self, operation: str = 'unknown') -> int:
        """Get the next seed in sequence for a specific operation."""
        if not self.enable_reproducibility:
            return None
        
        # Calculate next seed
        next_seed = self.base_seed + self.seed_offset
        self.seed_offset += self.seed_increment
        
        # Track seed usage
        if self.track_seed_usage:
            self.seed_history.append({
                'seed': next_seed,
                'operation': operation,
                'timestamp': pd.Timestamp.now()
            })
        
        tprint_debug(f"🔢 Generated seed {next_seed} for operation: {operation}")
        return next_seed

    def set_operation_seed(self, operation: str, seed: Optional[int] = None) -> int:
        """Set seed for a specific operation."""
        if not self.enable_reproducibility:
            return None
        
        if seed is None:
            seed = self.get_next_seed(operation)
        
        try:
            # Set seeds for the operation
            random.seed(seed)
            np.random.seed(seed)
            
            # Track seed usage
            if self.track_seed_usage:
                self.seed_history.append({
                    'seed': seed,
                    'operation': operation,
                    'timestamp': pd.Timestamp.now()
                })
            
            tprint_debug(f"🔢 Set seed {seed} for operation: {operation}")
            return seed
            
        except Exception as e:
            tprint_error(f"❌ Error setting seed for operation {operation}: {e}")
            return None

    def with_seed(self, operation: str, seed: Optional[int] = None):
        """Context manager for setting seed for a specific operation."""
        if not self.enable_reproducibility:
            return self._noop_context()
        
        if seed is None:
            seed = self.get_next_seed(operation)
        
        return self._SeedContext(operation, seed)

    class _SeedContext:
        """Context manager for temporary seed setting."""
        
        def __init__(self, operation: str, seed: int):
            self.operation = operation
            self.seed = seed
            self.original_random_state = None
            self.original_numpy_state = None
        
        def __enter__(self):
            # Save current random states
            self.original_random_state = random.getstate()
            self.original_numpy_state = np.random.get_state()
            
            # Set new seed
            random.seed(self.seed)
            np.random.seed(self.seed)
            
            tprint_debug(f"🔢 Entered seed context: {self.operation} (seed: {self.seed})")
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            # Restore original random states
            if self.original_random_state:
                random.setstate(self.original_random_state)
            if self.original_numpy_state:
                np.random.set_state(self.original_numpy_state)
            
            tprint_debug(f"🔢 Exited seed context: {self.operation}")

    def _noop_context(self):
        """No-op context manager when reproducibility is disabled."""
        return self._NoOpContext()

    class _NoOpContext:
        """No-op context manager."""
        
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    def set_library_seed(self, library: str, seed: Optional[int] = None) -> int:
        """Set seed for specific libraries."""
        if not self.enable_reproducibility:
            return None
        
        if seed is None:
            seed = self.get_next_seed(f"{library}_seed")
        
        try:
            if library.lower() == 'numpy':
                np.random.seed(seed)
            elif library.lower() == 'python':
                random.seed(seed)
            elif library.lower() == 'pandas':
                # Pandas uses numpy's random state
                np.random.seed(seed)
            elif library.lower() == 'sklearn':
                # Scikit-learn uses numpy's random state
                np.random.seed(seed)
            elif library.lower() == 'tensorflow':
                try:
                    import tensorflow as tf
                    tf.random.set_seed(seed)
                except ImportError:
                    tprint_warning("⚠️ TensorFlow not available")
            elif library.lower() == 'pytorch':
                try:
                    import torch
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                except ImportError:
                    tprint_warning("⚠️ PyTorch not available")
            else:
                tprint_warning(f"⚠️ Unknown library: {library}")
                return None
            
            tprint_success(f"✅ Set {library} seed to: {seed}")
            return seed
            
        except Exception as e:
            tprint_error(f"❌ Error setting {library} seed: {e}")
            return None

    def get_seed_summary(self) -> Dict[str, Any]:
        """Get summary of seed usage."""
        summary = {
            'base_seed': self.base_seed,
            'current_seed': self.current_seed,
            'seed_offset': self.seed_offset,
            'total_seeds_used': len(self.seed_history),
            'reproducibility_enabled': self.enable_reproducibility,
            'seed_usage_by_operation': {},
            'recent_operations': []
        }
        
        # Count seeds by operation
        for entry in self.seed_history:
            operation = entry['operation']
            summary['seed_usage_by_operation'][operation] = summary['seed_usage_by_operation'].get(operation, 0) + 1
        
        # Get recent operations (last 10)
        summary['recent_operations'] = self.seed_history[-10:] if self.seed_history else []
        
        return summary

    def reset_to_base_seed(self) -> None:
        """Reset to base seed and clear history."""
        self.set_global_seed(self.base_seed)
        self.seed_offset = 0
        self.seed_history = []
        tprint_success(f"✅ Reset to base seed: {self.base_seed}")

    def disable_reproducibility(self) -> None:
        """Disable reproducibility (for performance testing)."""
        self.enable_reproducibility = False
        tprint_warning("⚠️ Reproducibility disabled")

    def enable_reproducibility_mode(self) -> None:
        """Enable reproducibility mode."""
        self.enable_reproducibility = True
        self.set_global_seed(self.current_seed)
        tprint_success("✅ Reproducibility enabled")

    def export_seed_state(self) -> Dict[str, Any]:
        """Export current seed state for reproducibility."""
        return {
            'base_seed': self.base_seed,
            'current_seed': self.current_seed,
            'seed_offset': self.seed_offset,
            'seed_history': self.seed_history,
            'config': self.config
        }

    def import_seed_state(self, state: Dict[str, Any]) -> None:
        """Import seed state for reproducibility."""
        self.base_seed = state.get('base_seed', 42)
        self.current_seed = state.get('current_seed', self.base_seed)
        self.seed_offset = state.get('seed_offset', 0)
        self.seed_history = state.get('seed_history', [])
        
        # Set the current seed
        self.set_global_seed(self.current_seed)
        
        tprint_success("✅ Seed state imported successfully")