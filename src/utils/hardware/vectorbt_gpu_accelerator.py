"""
VectorBT GPU Accelerator for Apple Silicon.

This module provides GPU acceleration specifically optimized for VectorBT operations
using Metal Performance Shaders, with backward compatibility for existing VectorBT code.
"""

import logging
import time
import threading
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import warnings

# Optional dependencies
try:
    import torch
    import torch.backends.mps
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None

from .m1_enhanced_gpu_manager import (
    M1EnhancedGPUManager, GPUOperationType, GPUConfig, get_enhanced_gpu_manager
)

logger = logging.getLogger(__name__)

class VectorBTOperationType(Enum):
    """VectorBT-specific operation types."""
    PORTFOLIO_ANALYSIS = "portfolio_analysis"
    SIGNAL_GENERATION = "signal_generation"
    BACKTESTING = "backtesting"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    RISK_METRICS = "risk_metrics"
    CORRELATION_ANALYSIS = "correlation_analysis"
    ROLLING_OPERATIONS = "rolling_operations"
    CROSS_SECTIONAL_ANALYSIS = "cross_sectional_analysis"

@dataclass
class VectorBTGPUConfig(GPUConfig):
    """Configuration for VectorBT GPU operations."""
    # VectorBT-specific settings
    enable_portfolio_optimization: bool = True
    enable_signal_acceleration: bool = True
    enable_backtesting_acceleration: bool = True
    enable_risk_metrics_acceleration: bool = True
    
    # Memory optimization for financial data
    optimize_financial_dtypes: bool = True
    use_float32_for_calculations: bool = True
    enable_memory_mapping: bool = True
    
    # Batch processing for large datasets
    batch_size: int = 1000
    enable_parallel_processing: bool = True
    max_parallel_operations: int = 4

class VectorBTGPUAccelerator:
    """GPU accelerator specifically for VectorBT operations."""
    
    def __init__(self, config: Optional[VectorBTGPUConfig] = None):
        self.config = config or VectorBTGPUConfig()
        self.logger = logger.getChild('VectorBTGPUAccelerator')
        
        # Initialize base GPU manager
        self.gpu_manager = get_enhanced_gpu_manager(self.config)
        
        # VectorBT-specific optimizations
        self.operation_cache = {}
        self.performance_metrics = {
            'total_operations': 0,
            'gpu_operations': 0,
            'cpu_fallbacks': 0,
            'average_speedup': 0.0,
            'memory_savings_mb': 0.0
        }
        
        # Check availability
        self.gpu_available = self.gpu_manager.is_available()
        if not self.gpu_available:
            self.logger.warning("⚠️ GPU not available - VectorBT operations will use CPU fallback")
        
        self.logger.info("🚀 VectorBT GPU Accelerator initialized")
    
    def _optimize_financial_data(self, data: Any) -> Any:
        """Optimize financial data for GPU processing."""
        if isinstance(data, pd.DataFrame):
            # Optimize DataFrame for GPU
            optimized_df = data.copy()
            
            if self.config.optimize_financial_dtypes:
                # Convert to float32 for GPU efficiency
                for col in optimized_df.select_dtypes(include=[np.float64]):
                    optimized_df[col] = optimized_df[col].astype(np.float32)
                
                # Convert to int32 for integer columns
                for col in optimized_df.select_dtypes(include=[np.int64]):
                    optimized_df[col] = optimized_df[col].astype(np.int32)
            
            return optimized_df
        
        elif isinstance(data, np.ndarray):
            # Optimize NumPy array for GPU
            if self.config.use_float32_for_calculations and data.dtype == np.float64:
                return data.astype(np.float32)
            return data
        
        return data
    
    def _gpu_portfolio_analysis(self, returns: np.ndarray, weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """GPU-accelerated portfolio analysis."""
        if not self.gpu_available:
            return self._cpu_portfolio_analysis(returns, weights)
        
        try:
            # Convert to PyTorch tensors
            returns_tensor = torch.from_numpy(returns).float()
            if weights is not None:
                weights_tensor = torch.from_numpy(weights).float()
            else:
                weights_tensor = torch.ones(returns.shape[1]) / returns.shape[1]
            
            # Move to GPU
            returns_tensor = returns_tensor.to('mps')
            weights_tensor = weights_tensor.to('mps')
            
            # Calculate portfolio returns
            portfolio_returns = torch.sum(returns_tensor * weights_tensor, dim=1)
            
            # Calculate metrics
            mean_return = torch.mean(portfolio_returns)
            volatility = torch.std(portfolio_returns)
            sharpe_ratio = mean_return / volatility if volatility > 0 else torch.tensor(0.0)
            
            # Calculate VaR (95% confidence)
            var_95 = torch.quantile(portfolio_returns, 0.05)
            
            # Move results back to CPU
            results = {
                'mean_return': mean_return.cpu().item(),
                'volatility': volatility.cpu().item(),
                'sharpe_ratio': sharpe_ratio.cpu().item(),
                'var_95': var_95.cpu().item(),
                'portfolio_returns': portfolio_returns.cpu().numpy()
            }
            
            self.performance_metrics['gpu_operations'] += 1
            return results
            
        except Exception as e:
            self.logger.warning(f"GPU portfolio analysis failed: {e}, falling back to CPU")
            self.performance_metrics['cpu_fallbacks'] += 1
            return self._cpu_portfolio_analysis(returns, weights)
    
    def _cpu_portfolio_analysis(self, returns: np.ndarray, weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """CPU fallback for portfolio analysis."""
        if weights is None:
            weights = np.ones(returns.shape[1]) / returns.shape[1]
        
        portfolio_returns = np.sum(returns * weights, axis=1)
        
        return {
            'mean_return': np.mean(portfolio_returns),
            'volatility': np.std(portfolio_returns),
            'sharpe_ratio': np.mean(portfolio_returns) / np.std(portfolio_returns) if np.std(portfolio_returns) > 0 else 0,
            'var_95': np.percentile(portfolio_returns, 5),
            'portfolio_returns': portfolio_returns
        }
    
    def _gpu_signal_generation(self, price_data: np.ndarray, 
                              signal_params: Dict[str, Any]) -> np.ndarray:
        """GPU-accelerated signal generation."""
        if not self.gpu_available:
            return self._cpu_signal_generation(price_data, signal_params)
        
        try:
            # Convert to PyTorch tensor
            price_tensor = torch.from_numpy(price_data).float().to('mps')
            
            # Generate signals based on parameters
            signal_type = signal_params.get('type', 'sma_crossover')
            
            if signal_type == 'sma_crossover':
                short_window = signal_params.get('short_window', 20)
                long_window = signal_params.get('long_window', 50)
                
                # Calculate moving averages
                short_ma = torch.nn.functional.avg_pool1d(
                    price_tensor.unsqueeze(0).unsqueeze(0), 
                    kernel_size=short_window, 
                    stride=1, 
                    padding=short_window//2
                ).squeeze()
                
                long_ma = torch.nn.functional.avg_pool1d(
                    price_tensor.unsqueeze(0).unsqueeze(0), 
                    kernel_size=long_window, 
                    stride=1, 
                    padding=long_window//2
                ).squeeze()
                
                # Generate signals
                signals = torch.where(short_ma > long_ma, 1.0, -1.0)
                
            elif signal_type == 'rsi':
                rsi_period = signal_params.get('rsi_period', 14)
                oversold = signal_params.get('oversold', 30)
                overbought = signal_params.get('overbought', 70)
                
                # Calculate RSI
                delta = torch.diff(price_tensor)
                gain = torch.where(delta > 0, delta, 0)
                loss = torch.where(delta < 0, -delta, 0)
                
                avg_gain = torch.nn.functional.avg_pool1d(
                    gain.unsqueeze(0).unsqueeze(0), 
                    kernel_size=rsi_period, 
                    stride=1, 
                    padding=rsi_period//2
                ).squeeze()
                
                avg_loss = torch.nn.functional.avg_pool1d(
                    loss.unsqueeze(0).unsqueeze(0), 
                    kernel_size=rsi_period, 
                    stride=1, 
                    padding=rsi_period//2
                ).squeeze()
                
                rs = avg_gain / (avg_loss + 1e-8)
                rsi = 100 - (100 / (1 + rs))
                
                # Generate signals
                signals = torch.where(rsi < oversold, 1.0, 
                                    torch.where(rsi > overbought, -1.0, 0.0))
            
            else:
                # Default to simple momentum
                momentum_period = signal_params.get('momentum_period', 10)
                momentum = price_tensor[momentum_period:] - price_tensor[:-momentum_period]
                signals = torch.where(momentum > 0, 1.0, -1.0)
                # Pad with zeros
                signals = torch.cat([torch.zeros(momentum_period), signals])
            
            # Move back to CPU
            result = signals.cpu().numpy()
            self.performance_metrics['gpu_operations'] += 1
            return result
            
        except Exception as e:
            self.logger.warning(f"GPU signal generation failed: {e}, falling back to CPU")
            self.performance_metrics['cpu_fallbacks'] += 1
            return self._cpu_signal_generation(price_data, signal_params)
    
    def _cpu_signal_generation(self, price_data: np.ndarray, 
                              signal_params: Dict[str, Any]) -> np.ndarray:
        """CPU fallback for signal generation."""
        signal_type = signal_params.get('type', 'sma_crossover')
        
        if signal_type == 'sma_crossover':
            short_window = signal_params.get('short_window', 20)
            long_window = signal_params.get('long_window', 50)
            
            short_ma = pd.Series(price_data).rolling(short_window).mean().values
            long_ma = pd.Series(price_data).rolling(long_window).mean().values
            
            signals = np.where(short_ma > long_ma, 1.0, -1.0)
            
        elif signal_type == 'rsi':
            rsi_period = signal_params.get('rsi_period', 14)
            oversold = signal_params.get('oversold', 30)
            overbought = signal_params.get('overbought', 70)
            
            # Calculate RSI
            delta = np.diff(price_data)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            
            avg_gain = pd.Series(gain).rolling(rsi_period).mean().values
            avg_loss = pd.Series(loss).rolling(rsi_period).mean().values
            
            rs = avg_gain / (avg_loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            
            signals = np.where(rsi < oversold, 1.0, 
                             np.where(rsi > overbought, -1.0, 0.0))
            signals = np.concatenate([[0], signals])  # Pad with zero
            
        else:
            # Default momentum
            momentum_period = signal_params.get('momentum_period', 10)
            momentum = price_data[momentum_period:] - price_data[:-momentum_period]
            signals = np.where(momentum > 0, 1.0, -1.0)
            signals = np.concatenate([np.zeros(momentum_period), signals])
        
        return signals
    
    def _gpu_rolling_operations(self, data: np.ndarray, 
                               operation: str, window: int) -> np.ndarray:
        """GPU-accelerated rolling operations."""
        if not self.gpu_available:
            return self._cpu_rolling_operations(data, operation, window)
        
        try:
            # Convert to PyTorch tensor
            data_tensor = torch.from_numpy(data).float().to('mps')
            
            if operation == 'mean':
                result = torch.nn.functional.avg_pool1d(
                    data_tensor.unsqueeze(0).unsqueeze(0), 
                    kernel_size=window, 
                    stride=1, 
                    padding=window//2
                ).squeeze()
            elif operation == 'std':
                # Calculate rolling standard deviation
                mean = torch.nn.functional.avg_pool1d(
                    data_tensor.unsqueeze(0).unsqueeze(0), 
                    kernel_size=window, 
                    stride=1, 
                    padding=window//2
                ).squeeze()
                
                # Calculate variance
                squared_diff = (data_tensor.unsqueeze(0) - mean.unsqueeze(0)) ** 2
                variance = torch.nn.functional.avg_pool1d(
                    squared_diff.unsqueeze(0), 
                    kernel_size=window, 
                    stride=1, 
                    padding=window//2
                ).squeeze()
                
                result = torch.sqrt(variance)
            elif operation == 'max':
                result = torch.nn.functional.max_pool1d(
                    data_tensor.unsqueeze(0).unsqueeze(0), 
                    kernel_size=window, 
                    stride=1, 
                    padding=window//2
                ).squeeze()
            elif operation == 'min':
                result = -torch.nn.functional.max_pool1d(
                    (-data_tensor).unsqueeze(0).unsqueeze(0), 
                    kernel_size=window, 
                    stride=1, 
                    padding=window//2
                ).squeeze()
            else:
                # Default to mean
                result = torch.nn.functional.avg_pool1d(
                    data_tensor.unsqueeze(0).unsqueeze(0), 
                    kernel_size=window, 
                    stride=1, 
                    padding=window//2
                ).squeeze()
            
            # Move back to CPU
            result_array = result.cpu().numpy()
            self.performance_metrics['gpu_operations'] += 1
            return result_array
            
        except Exception as e:
            self.logger.warning(f"GPU rolling operations failed: {e}, falling back to CPU")
            self.performance_metrics['cpu_fallbacks'] += 1
            return self._cpu_rolling_operations(data, operation, window)
    
    def _cpu_rolling_operations(self, data: np.ndarray, 
                               operation: str, window: int) -> np.ndarray:
        """CPU fallback for rolling operations."""
        series = pd.Series(data)
        
        if operation == 'mean':
            return series.rolling(window).mean().values
        elif operation == 'std':
            return series.rolling(window).std().values
        elif operation == 'max':
            return series.rolling(window).max().values
        elif operation == 'min':
            return series.rolling(window).min().values
        else:
            return series.rolling(window).mean().values
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        total_ops = self.performance_metrics['total_operations']
        gpu_ops = self.performance_metrics['gpu_operations']
        
        return {
            'vectorbt_gpu_metrics': self.performance_metrics,
            'gpu_utilization_rate': gpu_ops / total_ops if total_ops > 0 else 0,
            'cpu_fallback_rate': self.performance_metrics['cpu_fallbacks'] / total_ops if total_ops > 0 else 0,
            'gpu_available': self.gpu_available,
            'base_gpu_metrics': self.gpu_manager.get_performance_metrics()
        }

# Global instance
_vectorbt_gpu_accelerator: Optional[VectorBTGPUAccelerator] = None

def get_vectorbt_gpu_accelerator(config: Optional[VectorBTGPUConfig] = None) -> VectorBTGPUAccelerator:
    """Get or create the global VectorBT GPU accelerator."""
    global _vectorbt_gpu_accelerator
    
    if _vectorbt_gpu_accelerator is None:
        _vectorbt_gpu_accelerator = VectorBTGPUAccelerator(config)
    
    return _vectorbt_gpu_accelerator

def gpu_accelerated_vectorbt(operation_type: VectorBTOperationType = VectorBTOperationType.PORTFOLIO_ANALYSIS):
    """Decorator for VectorBT GPU acceleration."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            accelerator = get_vectorbt_gpu_accelerator()
            
            # Track operation
            accelerator.performance_metrics['total_operations'] += 1
            
            # Optimize inputs
            optimized_args = []
            for arg in args:
                if isinstance(arg, (np.ndarray, pd.DataFrame)):
                    optimized_args.append(accelerator._optimize_financial_data(arg))
                else:
                    optimized_args.append(arg)
            
            # Execute with GPU acceleration based on operation type
            if operation_type == VectorBTOperationType.PORTFOLIO_ANALYSIS:
                if len(optimized_args) >= 1:
                    returns = optimized_args[0]
                    weights = optimized_args[1] if len(optimized_args) > 1 else None
                    return accelerator._gpu_portfolio_analysis(returns, weights)
            
            elif operation_type == VectorBTOperationType.SIGNAL_GENERATION:
                if len(optimized_args) >= 1:
                    price_data = optimized_args[0]
                    signal_params = kwargs.get('signal_params', {})
                    return accelerator._gpu_signal_generation(price_data, signal_params)
            
            elif operation_type == VectorBTOperationType.ROLLING_OPERATIONS:
                if len(optimized_args) >= 1:
                    data = optimized_args[0]
                    operation = kwargs.get('operation', 'mean')
                    window = kwargs.get('window', 20)
                    return accelerator._gpu_rolling_operations(data, operation, window)
            
            # Fallback to original function
            return func(*optimized_args, **kwargs)
        
        return wrapper
    return decorator

# Backward compatibility functions
def gpu_vectorbt_optimization(price_data: np.ndarray, features: Dict[str, Any]) -> Dict[str, Any]:
    """Backward compatible function for VectorBT GPU optimization."""
    accelerator = get_vectorbt_gpu_accelerator()
    
    # Optimize data
    optimized_data = accelerator._optimize_financial_data(price_data)
    
    # Perform portfolio analysis
    results = accelerator._gpu_portfolio_analysis(optimized_data)
    
    # Add feature-specific analysis
    if 'weights' in features:
        weighted_results = accelerator._gpu_portfolio_analysis(optimized_data, features['weights'])
        results.update(weighted_results)
    
    return results

def get_vectorbt_gpu_performance_metrics() -> Dict[str, Any]:
    """Get VectorBT GPU performance metrics."""
    accelerator = get_vectorbt_gpu_accelerator()
    return accelerator.get_performance_metrics()