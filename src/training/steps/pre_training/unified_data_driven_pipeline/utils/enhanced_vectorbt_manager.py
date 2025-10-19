"""
Enhanced VectorBT Manager for GPU Acceleration and Lazy Evaluation

Implements advanced VectorBT features including GPU acceleration, lazy evaluation,
and M1-optimized operations for feature interaction generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass
import logging
import warnings
from functools import wraps
import time

from src.utils.tprint import tprint

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_sum
    from vectorbt.portfolio import Portfolio
    from vectorbt.indicators import RSI, MACD, BBANDS
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    warnings.warn("VectorBT not available for enhanced operations")

# PyTorch for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)

@dataclass
class VectorBTConfig:
    """Configuration for enhanced VectorBT operations."""
    
    # GPU settings
    enable_gpu_acceleration: bool = True
    gpu_memory_fraction: float = 0.8
    fallback_to_cpu: bool = True
    
    # Lazy evaluation
    enable_lazy_evaluation: bool = True
    lazy_batch_size: int = 1000
    
    # Memory optimization
    chunk_size: int = 10000
    enable_memory_mapping: bool = True
    
    # Performance settings
    num_threads: int = 4
    enable_parallel_processing: bool = True

class LazyArray:
    """Lazy evaluation wrapper for VectorBT operations."""
    
    def __init__(self, operation: Callable, *args, **kwargs):
        self.operation = operation
        self.args = args
        self.kwargs = kwargs
        self._result = None
        self._computed = False
    
    def compute(self):
        """Compute the lazy operation."""
        if not self._computed:
            tprint("🔄 [LAZY] Computing lazy operation")
            start_time = time.time()
            self._result = self.operation(*self.args, **self.kwargs)
            self._computed = True
            tprint(f"✅ [LAZY] Lazy operation completed in {time.time() - start_time:.2f}s")
        return self._result
    
    def __getattr__(self, name):
        """Delegate attribute access to computed result."""
        if not self._computed:
            self.compute()
        return getattr(self._result, name)

class EnhancedVectorBTManager:
    """Enhanced VectorBT manager with GPU acceleration and lazy evaluation."""
    
    def __init__(self, config: Optional[VectorBTConfig] = None):
        self.config = config or VectorBTConfig()
        self.logger = logger.getChild('EnhancedVectorBTManager')
        
        # Check availability
        self.vectorbt_available = VECTORBT_AVAILABLE
        self.torch_available = TORCH_AVAILABLE
        self.gpu_available = self._check_gpu_availability()
        
        # Initialize VectorBT settings
        if self.vectorbt_available:
            self._initialize_vectorbt_settings()
        
        # Lazy evaluation cache
        self.lazy_cache = {}
        self.operation_count = 0
        
        tprint("🚀 [VECTORBT] Enhanced VectorBT Manager initialized")
        tprint(f"📊 [VECTORBT] VectorBT: {'Available' if self.vectorbt_available else 'Not Available'}")
        tprint(f"📊 [VECTORBT] GPU: {'Available' if self.gpu_available else 'Not Available'}")
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available."""
        if not self.torch_available:
            return False
        
        try:
            # Check for MPS (Metal Performance Shaders) on M1
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                tprint("✅ [VECTORBT] M1 GPU (MPS) available")
                return True
            
            # Check for CUDA
            if torch.cuda.is_available():
                tprint("✅ [VECTORBT] CUDA GPU available")
                return True
            
            return False
            
        except Exception as e:
            tprint(f"⚠️ [VECTORBT] GPU check failed: {e}")
            return False
    
    def _initialize_vectorbt_settings(self):
        """Initialize VectorBT settings for optimal performance."""
        try:
            # Set VectorBT settings for performance
            vbt.settings.array_wrapper['freq'] = None  # Disable frequency inference for speed
            
            # Set threading
            if self.config.enable_parallel_processing:
                vbt.settings.array_wrapper['num_threads'] = self.config.num_threads
            
            tprint("✅ [VECTORBT] VectorBT settings initialized")
            
        except Exception as e:
            tprint(f"⚠️ [VECTORBT] Failed to initialize VectorBT settings: {e}")
    
    def gpu_accelerated_rolling_mean(self, data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """GPU-accelerated rolling mean using VectorBT and PyTorch."""
        if not self.gpu_available or not self.vectorbt_available:
            return self._fallback_rolling_mean(data, window)
        
        try:
            tprint(f"🚀 [VECTORBT] GPU-accelerated rolling mean (window={window})")
            
            # Convert to PyTorch tensor for GPU processing
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cuda'
            
            tensor_data = torch.tensor(data.values, dtype=torch.float32, device=device)
            
            # GPU-accelerated rolling mean
            rolling_result = torch.zeros_like(tensor_data)
            for i in range(window - 1, len(tensor_data)):
                rolling_result[i] = torch.mean(tensor_data[i-window+1:i+1], dim=0)
            
            # Convert back to DataFrame
            result_array = rolling_result.cpu().numpy()
            result_df = pd.DataFrame(result_array, index=data.index, columns=data.columns)
            
            tprint(f"✅ [VECTORBT] GPU rolling mean completed: {result_df.shape}")
            return result_df
            
        except Exception as e:
            tprint(f"⚠️ [VECTORBT] GPU rolling mean failed, falling back to CPU: {e}")
            return self._fallback_rolling_mean(data, window)
    
    def gpu_accelerated_correlation(self, data: pd.DataFrame) -> pd.DataFrame:
        """GPU-accelerated correlation matrix calculation."""
        if not self.gpu_available or not self.vectorbt_available:
            return data.corr()
        
        try:
            tprint("🚀 [VECTORBT] GPU-accelerated correlation calculation")
            
            # Convert to PyTorch tensor
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cuda'
            
            tensor_data = torch.tensor(data.values, dtype=torch.float32, device=device)
            
            # GPU-accelerated correlation
            # Standardize data
            mean = torch.mean(tensor_data, dim=0, keepdim=True)
            std = torch.std(tensor_data, dim=0, keepdim=True)
            standardized = (tensor_data - mean) / (std + 1e-8)
            
            # Compute correlation matrix
            corr_matrix = torch.mm(standardized.T, standardized) / (standardized.shape[0] - 1)
            
            # Convert back to DataFrame
            result_array = corr_matrix.cpu().numpy()
            result_df = pd.DataFrame(result_array, index=data.columns, columns=data.columns)
            
            tprint(f"✅ [VECTORBT] GPU correlation completed: {result_df.shape}")
            return result_df
            
        except Exception as e:
            tprint(f"⚠️ [VECTORBT] GPU correlation failed, falling back to CPU: {e}")
            return data.corr()
    
    def _fallback_rolling_mean(self, data: pd.DataFrame, window: int) -> pd.DataFrame:
        """Fallback rolling mean using VectorBT or pandas."""
        if self.vectorbt_available:
            try:
                tprint(f"🔄 [VECTORBT] VectorBT fallback rolling mean (window={window})")
                result = vbt.run_func(lambda x: x.rolling(window).mean(), data)
                return result
            except Exception as e:
                tprint(f"⚠️ [VECTORBT] VectorBT fallback failed: {e}")
        
        # Final fallback to pandas
        tprint(f"🔄 [VECTORBT] Pandas fallback rolling mean (window={window})")
        return data.rolling(window).mean()
    
    def lazy_rolling_operations(self, data: pd.DataFrame, 
                              operations: List[Callable], 
                              windows: List[int]) -> List[LazyArray]:
        """Create lazy evaluation objects for rolling operations."""
        if not self.config.enable_lazy_evaluation:
            # Execute immediately
            results = []
            for op, window in zip(operations, windows):
                result = op(data, window)
                results.append(result)
            return results
        
        tprint(f"🔄 [VECTORBT] Creating {len(operations)} lazy rolling operations")
        
        lazy_results = []
        for i, (op, window) in enumerate(zip(operations, windows)):
            lazy_op = LazyArray(op, data, window)
            lazy_results.append(lazy_op)
            self.lazy_cache[f"rolling_{i}_{window}"] = lazy_op
        
        tprint(f"✅ [VECTORBT] Created {len(lazy_results)} lazy operations")
        return lazy_results
    
    def compute_lazy_operations(self, lazy_operations: List[LazyArray]) -> List[pd.DataFrame]:
        """Compute all lazy operations."""
        tprint(f"🔄 [VECTORBT] Computing {len(lazy_operations)} lazy operations")
        
        results = []
        for i, lazy_op in enumerate(lazy_operations):
            tprint(f"📊 [VECTORBT] Computing lazy operation {i+1}/{len(lazy_operations)}")
            result = lazy_op.compute()
            results.append(result)
        
        tprint(f"✅ [VECTORBT] Completed computing lazy operations")
        return results
    
    def vectorbt_technical_indicators(self, data: pd.DataFrame, 
                                    indicators: List[str]) -> Dict[str, pd.DataFrame]:
        """Compute technical indicators using VectorBT."""
        if not self.vectorbt_available:
            tprint("⚠️ [VECTORBT] VectorBT not available for technical indicators")
            return {}
        
        tprint(f"🔄 [VECTORBT] Computing technical indicators: {indicators}")
        
        results = {}
        
        try:
            # RSI
            if 'rsi' in indicators:
                tprint("📊 [VECTORBT] Computing RSI")
                rsi = RSI.run(data, window=14).rsi
                results['rsi'] = rsi
            
            # MACD
            if 'macd' in indicators:
                tprint("📊 [VECTORBT] Computing MACD")
                macd = MACD.run(data, fast_window=12, slow_window=26, signal_window=9)
                results['macd'] = macd.macd
                results['macd_signal'] = macd.signal
                results['macd_histogram'] = macd.histogram
            
            # Bollinger Bands
            if 'bbands' in indicators:
                tprint("📊 [VECTORBT] Computing Bollinger Bands")
                bbands = BBANDS.run(data, window=20, alpha=2.0)
                results['bbands_upper'] = bbands.upper
                results['bbands_middle'] = bbands.middle
                results['bbands_lower'] = bbands.lower
            
            tprint(f"✅ [VECTORBT] Technical indicators completed: {len(results)} indicators")
            
        except Exception as e:
            tprint(f"❌ [VECTORBT] Technical indicators failed: {e}")
        
        return results
    
    def chunked_vectorbt_operations(self, data: pd.DataFrame, 
                                  operation_func: Callable,
                                  chunk_size: Optional[int] = None) -> pd.DataFrame:
        """Perform VectorBT operations in chunks for memory efficiency."""
        if chunk_size is None:
            chunk_size = self.config.chunk_size
        
        tprint(f"🔄 [VECTORBT] Chunked operations (chunk_size={chunk_size:,})")
        
        if len(data) <= chunk_size:
            # Process all at once
            return operation_func(data)
        
        results = []
        num_chunks = (len(data) + chunk_size - 1) // chunk_size
        
        for i in range(0, len(data), chunk_size):
            chunk_num = i // chunk_size + 1
            tprint(f"📊 [VECTORBT] Processing chunk {chunk_num}/{num_chunks}")
            
            chunk = data.iloc[i:i+chunk_size]
            chunk_result = operation_func(chunk)
            results.append(chunk_result)
        
        tprint(f"✅ [VECTORBT] Chunked operations completed: {num_chunks} chunks")
        return pd.concat(results, ignore_index=True)
    
    def vectorbt_portfolio_analysis(self, returns: pd.DataFrame, 
                                  weights: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Perform portfolio analysis using VectorBT."""
        if not self.vectorbt_available:
            tprint("⚠️ [VECTORBT] VectorBT not available for portfolio analysis")
            return {}
        
        try:
            tprint("🔄 [VECTORBT] Portfolio analysis")
            
            # Create portfolio
            if weights is None:
                # Equal weights
                weights = pd.DataFrame(
                    np.ones_like(returns) / returns.shape[1],
                    index=returns.index,
                    columns=returns.columns
                )
            
            portfolio = Portfolio.from_returns(returns, weights=weights)
            
            # Calculate metrics
            results = {
                'total_return': portfolio.total_return(),
                'annualized_return': portfolio.annualized_return(),
                'volatility': portfolio.annualized_volatility(),
                'sharpe_ratio': portfolio.sharpe_ratio(),
                'max_drawdown': portfolio.max_drawdown(),
                'calmar_ratio': portfolio.calmar_ratio()
            }
            
            tprint(f"✅ [VECTORBT] Portfolio analysis completed")
            return results
            
        except Exception as e:
            tprint(f"❌ [VECTORBT] Portfolio analysis failed: {e}")
            return {}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {
            'vectorbt_available': self.vectorbt_available,
            'gpu_available': self.gpu_available,
            'torch_available': self.torch_available,
            'operation_count': self.operation_count,
            'lazy_cache_size': len(self.lazy_cache)
        }
        
        return stats
    
    def cleanup(self):
        """Clean up resources."""
        tprint("🧹 [VECTORBT] Cleaning up VectorBT manager")
        
        # Clear lazy cache
        self.lazy_cache.clear()
        
        # Clear GPU cache if using PyTorch
        if self.torch_available:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        tprint("✅ [VECTORBT] VectorBT manager cleanup completed")
