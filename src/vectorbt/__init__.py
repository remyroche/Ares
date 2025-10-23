"""
Production-ready VectorBT integration for Ares trading system.

This module provides a production-ready VectorBT implementation that:
- Requires VectorBT to be properly installed
- Provides fast-fail behavior if VectorBT is not available
- Implements all required VectorBT functionality used by the Ares system
- Includes comprehensive error handling and validation
- Optimized for financial time series analysis and backtesting

VectorBT is a critical dependency for the Ares trading system and must be installed
for production use. This module will fail fast if VectorBT is not available rather
than providing fallback implementations.
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Any, Optional, Union, List, Dict, Tuple, Callable
import logging

# Production validation - VectorBT is required
try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, 
        rolling_sum, rolling_apply, rolling_corr, rolling_cov, rolling_rank,
        rolling_quantile, rolling_skew, rolling_kurt, rolling_apply
    )
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    from vectorbt.returns import Returns
    from vectorbt.portfolio import Portfolio, PortfolioFactory
    from vectorbt.indicators.basic import RSI, MACD, BBANDS, ATR, STOCH
    from vectorbt.indicators.basic import SMA, EMA, BollingerBands
    from vectorbt.utils.config import configure
    from vectorbt.utils.decorators import cached_method
    from vectorbt.utils.array_wrapper import ArrayWrapper
    from vectorbt.utils.datetime_ import freq_delta
    from vectorbt.utils.random import set_seed
    from vectorbt.utils.config import settings as vbt_settings
    
    # VectorBT is available - production ready
    VECTORBT_AVAILABLE = True
    VECTORBT_VERSION = getattr(vbt, '__version__', 'unknown')
    
    # Production logging
    logger = logging.getLogger(__name__)
    logger.info(f"✅ VectorBT {VECTORBT_VERSION} loaded successfully - Production ready")
    
except ImportError as e:
    # Fast fail - VectorBT is required for production
    error_msg = (
        "VectorBT is required for production use but not available. "
        f"Install VectorBT with: pip install vectorbt>=0.25.0\n"
        f"Run installation script: python src/vectorbt/install_vectorbt.py\n"
        f"Original error: {e}"
    )
    raise ImportError(error_msg) from e
except Exception as e:
    # Fast fail on any other VectorBT-related errors
    error_msg = (
        f"VectorBT initialization failed: {e}. "
        "Please ensure VectorBT is properly installed and compatible. "
        "Run: python src/vectorbt/install_vectorbt.py"
    )
    raise RuntimeError(error_msg) from e

# Import performance monitoring and configuration
from .performance import (
    VectorBTPerformanceMonitor, monitor_operation, profile_operation,
    MemoryOptimizer, get_memory_usage, optimize_vectorbt_performance,
    get_performance_monitor
)
from .config import (
    VectorBTConfig, get_vectorbt_config, configure_vectorbt,
    get_optimal_chunk_size, validate_vectorbt_config,
    DEFAULT_CONFIG, PRODUCTION_CONFIG, DEVELOPMENT_CONFIG
)

# Re-export all VectorBT functionality for seamless integration
__all__ = [
    # Core VectorBT modules
    'vbt',
    
    # Generic functions
    'rolling_mean', 'rolling_std', 'rolling_var', 'rolling_min', 'rolling_max',
    'rolling_sum', 'rolling_apply', 'rolling_corr', 'rolling_cov', 'rolling_rank',
    'rolling_quantile', 'rolling_skew', 'rolling_kurt',
    'scale', 'rank', 'zscore', 'winsorize', 'clip', 'quantile',
    
    # Portfolio and returns
    'Portfolio', 'PortfolioFactory', 'Returns',
    
    # Technical indicators
    'RSI', 'MACD', 'BBANDS', 'ATR', 'STOCH', 'SMA', 'EMA', 'BollingerBands',
    
    # Utilities
    'ArrayWrapper', 'freq_delta', 'set_seed', 'configure',
    
    # Settings
    'vbt_settings',
    
    # Status
    'VECTORBT_AVAILABLE', 'VECTORBT_VERSION',
    
    # Performance monitoring
    'VectorBTPerformanceMonitor', 'monitor_operation', 'profile_operation',
    'MemoryOptimizer', 'get_memory_usage', 'optimize_vectorbt_performance',
    'get_performance_monitor',
    
    # Configuration
    'VectorBTConfig', 'get_vectorbt_config', 'configure_vectorbt',
    'get_optimal_chunk_size', 'validate_vectorbt_config',
    'DEFAULT_CONFIG', 'PRODUCTION_CONFIG', 'DEVELOPMENT_CONFIG',
    
    # Error classes
    'VectorBTError', 'VectorBTConfigurationError', 'VectorBTDataError', 'VectorBTComputationError',
    
    # Production utilities
    'ProductionPortfolioFactory', 'ProductionRollingOperations',
    'validate_vectorbt_installation', 'get_vectorbt_info', 'initialize_production_vectorbt'
]

# Production validation functions
def validate_vectorbt_installation() -> bool:
    """
    Validate that VectorBT is properly installed and configured.
    
    Returns:
        bool: True if VectorBT is properly installed and configured
        
    Raises:
        RuntimeError: If VectorBT validation fails
    """
    try:
        # Test basic functionality
        import pandas as pd
        import numpy as np
        
        # Create test data
        test_data = pd.Series(np.random.randn(100), 
                             index=pd.date_range('2023-01-01', periods=100, freq='1H'))
        
        # Test rolling operations
        rolling_mean(test_data, window=10)
        rolling_std(test_data, window=10)
        
        # Test portfolio creation
        test_returns = test_data.pct_change().dropna()
        portfolio = Portfolio.from_returns(test_returns)
        
        # Test indicators
        rsi = RSI.run(test_data)
        macd = MACD.run(test_data)
        
        logger.info("✅ VectorBT validation passed - All core functionality working")
        return True
        
    except Exception as e:
        error_msg = f"VectorBT validation failed: {e}"
        logger.error(f"❌ {error_msg}")
        raise RuntimeError(error_msg) from e

def get_vectorbt_info() -> Dict[str, Any]:
    """
    Get comprehensive VectorBT installation information.
    
    Returns:
        Dict containing VectorBT version, configuration, and capabilities
    """
    try:
        return {
            'version': VECTORBT_VERSION,
            'available': VECTORBT_AVAILABLE,
            'installation_path': vbt.__file__,
            'settings': dict(vbt_settings),
            'python_version': sys.version,
            'platform': sys.platform,
            'numpy_version': getattr(vbt, 'np', None).__version__ if hasattr(vbt, 'np') else 'unknown',
            'pandas_version': getattr(vbt, 'pd', None).__version__ if hasattr(vbt, 'pd') else 'unknown',
        }
    except Exception as e:
        logger.warning(f"Could not get VectorBT info: {e}")
        return {
            'version': 'unknown',
            'available': False,
            'error': str(e)
        }

# Production-ready error handling
class VectorBTError(Exception):
    """Base exception for VectorBT-related errors."""
    pass

class VectorBTConfigurationError(VectorBTError):
    """Exception raised when VectorBT configuration is invalid."""
    pass

class VectorBTDataError(VectorBTError):
    """Exception raised when data validation fails."""
    pass

class VectorBTComputationError(VectorBTError):
    """Exception raised when VectorBT computation fails."""
    pass

# Enhanced portfolio factory with production features
class ProductionPortfolioFactory:
    """
    Production-ready portfolio factory with enhanced error handling and validation.
    """
    
    @staticmethod
    def from_signals(
        close: pd.Series,
        entries: pd.Series,
        exits: pd.Series,
        **kwargs
    ) -> Portfolio:
        """
        Create portfolio from signals with production validation.
        
        Args:
            close: Price series
            entries: Entry signals
            exits: Exit signals
            **kwargs: Additional portfolio parameters
            
        Returns:
            Portfolio: Configured portfolio
            
        Raises:
            VectorBTDataError: If data validation fails
            VectorBTComputationError: If portfolio creation fails
        """
        try:
            # Validate inputs
            if not isinstance(close, pd.Series):
                raise VectorBTDataError("close must be a pandas Series")
            if not isinstance(entries, pd.Series):
                raise VectorBTDataError("entries must be a pandas Series")
            if not isinstance(exits, pd.Series):
                raise VectorBTDataError("exits must be a pandas Series")
            
            # Align indices
            if not close.index.equals(entries.index):
                entries = entries.reindex(close.index, fill_value=False)
            if not close.index.equals(exits.index):
                exits = exits.reindex(close.index, fill_value=False)
            
            # Create portfolio
            portfolio = PortfolioFactory.from_signals(
                close=close,
                entries=entries,
                exits=exits,
                **kwargs
            )
            
            logger.debug(f"Portfolio created successfully with {len(portfolio.trades)} trades")
            return portfolio
            
        except Exception as e:
            error_msg = f"Portfolio creation failed: {e}"
            logger.error(error_msg)
            raise VectorBTComputationError(error_msg) from e
    
    @staticmethod
    def from_returns(returns: pd.Series, **kwargs) -> Portfolio:
        """
        Create portfolio from returns with production validation.
        
        Args:
            returns: Returns series
            **kwargs: Additional portfolio parameters
            
        Returns:
            Portfolio: Configured portfolio
        """
        try:
            if not isinstance(returns, pd.Series):
                raise VectorBTDataError("returns must be a pandas Series")
            
            portfolio = PortfolioFactory.from_returns(returns, **kwargs)
            logger.debug(f"Portfolio created from returns with {len(returns)} periods")
            return portfolio
            
        except Exception as e:
            error_msg = f"Portfolio creation from returns failed: {e}"
            logger.error(error_msg)
            raise VectorBTComputationError(error_msg) from e

# Enhanced rolling operations with production features
class ProductionRollingOperations:
    """
    Production-ready rolling operations with enhanced error handling.
    """
    
    @staticmethod
    def safe_rolling_apply(
        data: pd.Series, 
        func: Callable, 
        window: int, 
        min_periods: int = None,
        **kwargs
    ) -> pd.Series:
        """
        Safe rolling apply with error handling and validation.
        
        Args:
            data: Input data series
            func: Function to apply
            window: Rolling window size
            min_periods: Minimum periods required
            **kwargs: Additional parameters
            
        Returns:
            pd.Series: Result of rolling apply
        """
        try:
            if min_periods is None:
                min_periods = window // 2
            
            result = rolling_apply(
                data, 
                func, 
                window=window, 
                min_periods=min_periods,
                **kwargs
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Rolling apply failed: {e}"
            logger.error(error_msg)
            raise VectorBTComputationError(error_msg) from e

# Performance monitoring
class VectorBTPerformanceMonitor:
    """
    Monitor VectorBT performance and memory usage.
    """
    
    def __init__(self):
        self.operation_times = {}
        self.memory_usage = {}
    
    def time_operation(self, operation_name: str, func: Callable, *args, **kwargs):
        """Time a VectorBT operation."""
        import time
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            self.operation_times[operation_name] = end_time - start_time
            logger.debug(f"Operation {operation_name} took {self.operation_times[operation_name]:.4f}s")
            return result
        except Exception as e:
            end_time = time.time()
            self.operation_times[operation_name] = end_time - start_time
            logger.error(f"Operation {operation_name} failed after {self.operation_times[operation_name]:.4f}s: {e}")
            raise

# Global performance monitor
_performance_monitor = VectorBTPerformanceMonitor()

# Production-ready module initialization
def initialize_production_vectorbt() -> bool:
    """
    Initialize VectorBT for production use.
    
    Returns:
        bool: True if initialization successful
    """
    try:
        # Import configuration
        from .config import configure_vectorbt, PRODUCTION_CONFIG
        
        # Configure VectorBT for production
        configure_vectorbt(PRODUCTION_CONFIG)
        
        # Validate installation
        validate_vectorbt_installation()
        
        # Get system info
        info = get_vectorbt_info()
        logger.info(f"VectorBT initialized: {info['version']}")
        
        logger.info("✅ VectorBT production initialization complete")
        return True
        
    except Exception as e:
        logger.error(f"❌ VectorBT production initialization failed: {e}")
        raise

# Auto-initialize for production
if VECTORBT_AVAILABLE:
    try:
        initialize_production_vectorbt()
    except Exception as e:
        logger.error(f"Failed to initialize VectorBT for production: {e}")
        raise

# Module version
__version__ = "1.0.0-production"