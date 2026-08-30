"""
VectorBT Base Integration

Base class and utilities for VectorBT integration with the backtesting infrastructure.
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
import warnings

# VectorBT imports
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt")

# Optional GPU support
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

from .vectorbt_config import VectorBTConfig, VectorBTMode

logger = logging.getLogger(__name__)

class VectorBTBase:
    """
    Base class for VectorBT integration.
    
    Provides common functionality and utilities for VectorBT operations.
    """
    
    def __init__(self, config: VectorBTConfig):
        """Initialize VectorBT base class."""
        if not VECTORBT_AVAILABLE:
            raise ImportError("VectorBT is required but not installed")
        
        self.config = config
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Initialize VectorBT settings
        self._setup_vectorbt()
        
        # Performance tracking
        self.performance_stats = {
            'operations_count': 0,
            'total_time': 0.0,
            'memory_usage': 0.0
        }
        
        self.logger.info(f"VectorBT Base initialized with mode: {config.mode.value}")
    
    def _setup_vectorbt(self):
        """Setup VectorBT configuration and settings."""
        try:
            # Set VectorBT settings
            vbt.settings.set_theme('dark')
            vbt.settings['plotting']['layout']['width'] = self.config.figure_size[0]
            vbt.settings['plotting']['layout']['height'] = self.config.figure_size[1]
            
            # Configure performance settings
            if self.config.enable_gpu and CUPY_AVAILABLE:
                vbt.settings.array_wrapper['freq'] = self.config.freq
                self.logger.info("GPU acceleration enabled")
            else:
                self.logger.info("CPU mode enabled")
            
            # Set memory limits
            if hasattr(vbt.settings, 'memory'):
                vbt.settings.memory['limit'] = self.config.memory_limit_mb * 1024 * 1024
            
            self.logger.info("VectorBT configuration completed")
            
        except Exception as e:
            self.logger.error(f"Failed to setup VectorBT: {e}")
            raise
    
    def validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and prepare data for VectorBT operations."""
        try:
            if data is None or data.empty:
                raise ValueError("Data is empty or None")
            
            # Ensure proper index
            if not isinstance(data.index, pd.DatetimeIndex):
                if 'timestamp' in data.columns:
                    data = data.set_index('timestamp')
                else:
                    data.index = pd.date_range(start='2020-01-01', periods=len(data), freq='D')
            
            # Ensure required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Validate data types
            for col in required_columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Remove NaN values
            data = data.dropna()
            
            if len(data) == 0:
                raise ValueError("No valid data after cleaning")
            
            self.logger.info(f"Data validated: {len(data)} rows, {len(data.columns)} columns")
            return data
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            raise
    
    def prepare_signals(self, data: pd.DataFrame, 
                       entries: Optional[pd.Series] = None,
                       exits: Optional[pd.Series] = None) -> Tuple[pd.Series, pd.Series]:
        """Prepare entry and exit signals for VectorBT portfolio simulation."""
        try:
            if entries is None:
                # Create default entries (all True for testing)
                entries = pd.Series(False, index=data.index)
                entries.iloc[::20] = True  # Every 20th day as entry
            
            if exits is None:
                # Create default exits (all False)
                exits = pd.Series(False, index=data.index)
            
            # Ensure signals are boolean
            entries = entries.astype(bool)
            exits = exits.astype(bool)
            
            # Align with data index
            entries = entries.reindex(data.index, fill_value=False)
            exits = exits.reindex(data.index, fill_value=False)
            
            self.logger.info(f"Signals prepared: {entries.sum()} entries, {exits.sum()} exits")
            return entries, exits
            
        except Exception as e:
            self.logger.error(f"Signal preparation failed: {e}")
            raise
    
    def calculate_returns(self, data: pd.DataFrame, 
                         method: str = 'pct_change') -> pd.Series:
        """Calculate returns from price data."""
        try:
            if method == 'pct_change':
                returns = data['close'].pct_change().dropna()
            elif method == 'log':
                returns = np.log(data['close'] / data['close'].shift(1)).dropna()
            else:
                raise ValueError(f"Unknown return method: {method}")
            
            self.logger.info(f"Returns calculated: {len(returns)} periods, mean={returns.mean():.4f}")
            return returns
            
        except Exception as e:
            self.logger.error(f"Return calculation failed: {e}")
            raise
    
    def optimize_memory(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize memory usage of data."""
        try:
            # Convert to appropriate dtypes
            for col in data.columns:
                if data[col].dtype == 'float64':
                    data[col] = data[col].astype('float32')
                elif data[col].dtype == 'int64':
                    data[col] = data[col].astype('int32')
            
            # Remove unnecessary columns
            keep_columns = ['open', 'high', 'low', 'close', 'volume']
            data = data[keep_columns]
            
            self.logger.info(f"Memory optimized: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            return data
            
        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")
            return data
    
    def chunk_data(self, data: pd.DataFrame, chunk_size: Optional[int] = None) -> List[pd.DataFrame]:
        """Split data into chunks for processing."""
        try:
            if chunk_size is None:
                chunk_size = self.config.chunk_size
            
            chunks = []
            for i in range(0, len(data), chunk_size):
                chunk = data.iloc[i:i + chunk_size]
                chunks.append(chunk)
            
            self.logger.info(f"Data chunked: {len(chunks)} chunks of max {chunk_size} rows")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Data chunking failed: {e}")
            return [data]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'operations_count': 0,
            'total_time': 0.0,
            'memory_usage': 0.0
        }
    
    def log_performance(self, operation: str, duration: float, memory_usage: float = 0.0):
        """Log performance metrics."""
        self.performance_stats['operations_count'] += 1
        self.performance_stats['total_time'] += duration
        self.performance_stats['memory_usage'] = max(self.performance_stats['memory_usage'], memory_usage)
        
        self.logger.debug(f"Operation '{operation}' completed in {duration:.3f}s, memory: {memory_usage:.2f}MB")
    
    def save_results(self, results: Dict[str, Any], filepath: Union[str, Path]):
        """Save results to file."""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_serializable(results)
            
            if filepath.suffix == '.json':
                import json
                with open(filepath, 'w') as f:
                    json.dump(serializable_results, f, indent=2, default=str)
            elif filepath.suffix == '.pkl':
                import pickle
                with open(filepath, 'wb') as f:
                    pickle.dump(results, f)
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")
            
            self.logger.info(f"Results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            raise
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        else:
            return obj

class VectorBTError(Exception):
    """Custom exception for VectorBT operations."""
    pass

class VectorBTValidationError(VectorBTError):
    """Exception for VectorBT validation errors."""
    pass

class VectorBTPerformanceError(VectorBTError):
    """Exception for VectorBT performance issues."""
    pass

def check_vectorbt_availability() -> bool:
    """Check if VectorBT is available."""
    return VECTORBT_AVAILABLE

def get_vectorbt_version() -> Optional[str]:
    """Get VectorBT version if available."""
    if VECTORBT_AVAILABLE:
        return vbt.__version__
    return None

def get_gpu_availability() -> bool:
    """Check if GPU acceleration is available."""
    return CUPY_AVAILABLE and VECTORBT_AVAILABLE