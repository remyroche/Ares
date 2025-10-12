"""
Consolidated Technical Indicators

This module provides centralized implementations of common technical indicators
(RSI, MACD, EMA, etc.) that all feature generators can use to avoid code duplication.
Uses UnifiedVectorizationManager and VectorBTRollingOptimizer for optimal performance.
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_apply
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    rolling_mean = None
    rolling_std = None
    rolling_apply = None

# Unified Vectorization Manager
try:
    from ...utils.ml_common.unified_vectorization_manager import (
        get_unified_vectorization_manager,
        OperationType,
        OptimizationStrategy
    )
    UNIFIED_MANAGER_AVAILABLE = True
except ImportError:
    UNIFIED_MANAGER_AVAILABLE = False
    get_unified_vectorization_manager = None
    OperationType = None
    OptimizationStrategy = None

# VectorBT Rolling Optimizer
try:
    from .vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
    ROLLING_OPTIMIZER_AVAILABLE = True
except ImportError:
    ROLLING_OPTIMIZER_AVAILABLE = False
    get_vectorbt_rolling_optimizer = None

# Common scalers
try:
    from ...features_common.transforms.vectorbt_scaler import VectorBTScaler, create_vectorbt_scaler
    SCALER_AVAILABLE = True
except ImportError:
    SCALER_AVAILABLE = False
    VectorBTScaler = None
    create_vectorbt_scaler = None

logger = logging.getLogger(__name__)


@dataclass
class IndicatorConfig:
    """Configuration for technical indicators."""
    use_unified_manager: bool = True
    use_vectorbt_optimizer: bool = True
    fallback_to_pandas: bool = True
    enable_gpu: bool = False
    memory_efficient: bool = True


class ConsolidatedTechnicalIndicators:
    """
    Consolidated technical indicators that all feature generators can use.
    
    This class provides optimized implementations of common technical indicators
    using the best available optimization strategy (UnifiedVectorizationManager,
    VectorBTRollingOptimizer, or pandas fallback).
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        """Initialize the consolidated technical indicators."""
        self.config = config or IndicatorConfig()
        self.logger = logger.getChild('ConsolidatedTechnicalIndicators')
        
        # Initialize optimization components
        self.unified_manager = None
        self.rolling_optimizer = None
        
        if self.config.use_unified_manager and UNIFIED_MANAGER_AVAILABLE:
            try:
                self.unified_manager = get_unified_vectorization_manager()
                self.logger.info("✅ UnifiedVectorizationManager initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize UnifiedVectorizationManager: {e}")
        
        if self.config.use_vectorbt_optimizer and ROLLING_OPTIMIZER_AVAILABLE:
            try:
                self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                    enable_gpu=self.config.enable_gpu,
                    enable_parallel=True,
                    memory_efficient=self.config.memory_efficient
                )
                self.logger.info("✅ VectorBTRollingOptimizer initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize VectorBTRollingOptimizer: {e}")
        
        # Performance tracking
        self.performance_stats = {
            'unified_manager_operations': 0,
            'vectorbt_optimizer_operations': 0,
            'pandas_fallback_operations': 0,
            'total_operations': 0
        }
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14, 
                     method: str = 'auto') -> pd.Series:
        """
        Calculate RSI using the best available optimization method.
        
        Args:
            prices: Price series
            period: RSI period
            method: Calculation method ('auto', 'unified', 'vectorbt', 'pandas')
            
        Returns:
            RSI series
        """
        self.performance_stats['total_operations'] += 1
        
        if method == 'auto':
            method = self._select_optimal_method(prices)
        
        try:
            if method == 'unified' and self.unified_manager:
                return self._calculate_rsi_unified(prices, period)
            elif method == 'vectorbt' and self.rolling_optimizer:
                return self._calculate_rsi_vectorbt(prices, period)
            else:
                return self._calculate_rsi_pandas(prices, period)
        except Exception as e:
            self.logger.warning(f"RSI calculation failed with {method}: {e}, using pandas fallback")
            return self._calculate_rsi_pandas(prices, period)
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, 
                      signal: int = 9, method: str = 'auto') -> Dict[str, pd.Series]:
        """
        Calculate MACD using the best available optimization method.
        
        Args:
            prices: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            method: Calculation method ('auto', 'unified', 'vectorbt', 'pandas')
            
        Returns:
            Dictionary with 'macd', 'signal', and 'histogram' series
        """
        self.performance_stats['total_operations'] += 1
        
        if method == 'auto':
            method = self._select_optimal_method(prices)
        
        try:
            if method == 'unified' and self.unified_manager:
                return self._calculate_macd_unified(prices, fast, slow, signal)
            elif method == 'vectorbt' and self.rolling_optimizer:
                return self._calculate_macd_vectorbt(prices, fast, slow, signal)
            else:
                return self._calculate_macd_pandas(prices, fast, slow, signal)
        except Exception as e:
            self.logger.warning(f"MACD calculation failed with {method}: {e}, using pandas fallback")
            return self._calculate_macd_pandas(prices, fast, slow, signal)
    
    def calculate_ema(self, prices: pd.Series, period: int, 
                     method: str = 'auto') -> pd.Series:
        """
        Calculate EMA using the best available optimization method.
        
        Args:
            prices: Price series
            period: EMA period
            method: Calculation method ('auto', 'unified', 'vectorbt', 'pandas')
            
        Returns:
            EMA series
        """
        self.performance_stats['total_operations'] += 1
        
        if method == 'auto':
            method = self._select_optimal_method(prices)
        
        try:
            if method == 'unified' and self.unified_manager:
                return self._calculate_ema_unified(prices, period)
            elif method == 'vectorbt' and self.rolling_optimizer:
                return self._calculate_ema_vectorbt(prices, period)
            else:
                return self._calculate_ema_pandas(prices, period)
        except Exception as e:
            self.logger.warning(f"EMA calculation failed with {method}: {e}, using pandas fallback")
            return self._calculate_ema_pandas(prices, period)
    
    def calculate_sma(self, prices: pd.Series, period: int, 
                     method: str = 'auto') -> pd.Series:
        """
        Calculate SMA using the best available optimization method.
        
        Args:
            prices: Price series
            period: SMA period
            method: Calculation method ('auto', 'unified', 'vectorbt', 'pandas')
            
        Returns:
            SMA series
        """
        self.performance_stats['total_operations'] += 1
        
        if method == 'auto':
            method = self._select_optimal_method(prices)
        
        try:
            if method == 'unified' and self.unified_manager:
                return self._calculate_sma_unified(prices, period)
            elif method == 'vectorbt' and self.rolling_optimizer:
                return self._calculate_sma_vectorbt(prices, period)
            else:
                return self._calculate_sma_pandas(prices, period)
        except Exception as e:
            self.logger.warning(f"SMA calculation failed with {method}: {e}, using pandas fallback")
            return self._calculate_sma_pandas(prices, period)
    
    def _select_optimal_method(self, data: pd.Series) -> str:
        """Select the optimal calculation method based on data size and available components."""
        if len(data) < 100:
            return 'pandas'
        elif self.unified_manager and len(data) >= 1000:
            return 'unified'
        elif self.rolling_optimizer and len(data) >= 500:
            return 'vectorbt'
        else:
            return 'pandas'
    
    # RSI Implementation Methods
    def _calculate_rsi_unified(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI using UnifiedVectorizationManager."""
        try:
            data = {'prices': prices, 'period': period}
            result = self.unified_manager.optimize_operation(
                OperationType.TECHNICAL_INDICATORS,
                data,
                **{'indicator': 'rsi', 'window': period}
            )
            self.performance_stats['unified_manager_operations'] += 1
            return result.result
        except Exception as e:
            self.logger.warning(f"Unified RSI calculation failed: {e}")
            raise
    
    def _calculate_rsi_vectorbt(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI using VectorBTRollingOptimizer."""
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = self.rolling_optimizer.rolling_mean(gain, window=period)
            avg_loss = self.rolling_optimizer.rolling_mean(loss, window=period)
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            self.performance_stats['vectorbt_optimizer_operations'] += 1
            return rsi
        except Exception as e:
            self.logger.warning(f"VectorBT RSI calculation failed: {e}")
            raise
    
    def _calculate_rsi_pandas(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI using pandas operations."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        self.performance_stats['pandas_fallback_operations'] += 1
        return rsi
    
    # MACD Implementation Methods
    def _calculate_macd_unified(self, prices: pd.Series, fast: int, slow: int, signal: int) -> Dict[str, pd.Series]:
        """Calculate MACD using UnifiedVectorizationManager."""
        try:
            data = {'prices': prices, 'fast': fast, 'slow': slow, 'signal': signal}
            result = self.unified_manager.optimize_operation(
                OperationType.TECHNICAL_INDICATORS,
                data,
                **{'indicator': 'macd', 'fast': fast, 'slow': slow, 'signal': signal}
            )
            self.performance_stats['unified_manager_operations'] += 1
            return result.result
        except Exception as e:
            self.logger.warning(f"Unified MACD calculation failed: {e}")
            raise
    
    def _calculate_macd_vectorbt(self, prices: pd.Series, fast: int, slow: int, signal: int) -> Dict[str, pd.Series]:
        """Calculate MACD using VectorBTRollingOptimizer."""
        try:
            ema_fast = self.rolling_optimizer.rolling_apply(
                prices, lambda x: x.ewm(span=fast).mean().iloc[-1], window=fast
            )
            ema_slow = self.rolling_optimizer.rolling_apply(
                prices, lambda x: x.ewm(span=slow).mean().iloc[-1], window=slow
            )
            
            macd_line = ema_fast - ema_slow
            signal_line = self.rolling_optimizer.rolling_apply(
                macd_line, lambda x: x.ewm(span=signal).mean().iloc[-1], window=signal
            )
            histogram = macd_line - signal_line
            
            self.performance_stats['vectorbt_optimizer_operations'] += 1
            return {
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram
            }
        except Exception as e:
            self.logger.warning(f"VectorBT MACD calculation failed: {e}")
            raise
    
    def _calculate_macd_pandas(self, prices: pd.Series, fast: int, slow: int, signal: int) -> Dict[str, pd.Series]:
        """Calculate MACD using pandas operations."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        self.performance_stats['pandas_fallback_operations'] += 1
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    # EMA Implementation Methods
    def _calculate_ema_unified(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate EMA using UnifiedVectorizationManager."""
        try:
            data = {'prices': prices, 'period': period}
            result = self.unified_manager.optimize_operation(
                OperationType.TECHNICAL_INDICATORS,
                data,
                **{'indicator': 'ema', 'window': period}
            )
            self.performance_stats['unified_manager_operations'] += 1
            return result.result
        except Exception as e:
            self.logger.warning(f"Unified EMA calculation failed: {e}")
            raise
    
    def _calculate_ema_vectorbt(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate EMA using VectorBTRollingOptimizer."""
        try:
            ema = self.rolling_optimizer.rolling_apply(
                prices, lambda x: x.ewm(span=period).mean().iloc[-1], window=period
            )
            self.performance_stats['vectorbt_optimizer_operations'] += 1
            return ema
        except Exception as e:
            self.logger.warning(f"VectorBT EMA calculation failed: {e}")
            raise
    
    def _calculate_ema_pandas(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate EMA using pandas operations."""
        ema = prices.ewm(span=period).mean()
        self.performance_stats['pandas_fallback_operations'] += 1
        return ema
    
    # SMA Implementation Methods
    def _calculate_sma_unified(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate SMA using UnifiedVectorizationManager."""
        try:
            data = {'prices': prices, 'period': period}
            result = self.unified_manager.optimize_operation(
                OperationType.TECHNICAL_INDICATORS,
                data,
                **{'indicator': 'sma', 'window': period}
            )
            self.performance_stats['unified_manager_operations'] += 1
            return result.result
        except Exception as e:
            self.logger.warning(f"Unified SMA calculation failed: {e}")
            raise
    
    def _calculate_sma_vectorbt(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate SMA using VectorBTRollingOptimizer."""
        try:
            sma = self.rolling_optimizer.rolling_mean(prices, window=period)
            self.performance_stats['vectorbt_optimizer_operations'] += 1
            return sma
        except Exception as e:
            self.logger.warning(f"VectorBT SMA calculation failed: {e}")
            raise
    
    def _calculate_sma_pandas(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate SMA using pandas operations."""
        sma = prices.rolling(window=period).mean()
        self.performance_stats['pandas_fallback_operations'] += 1
        return sma
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()
    
    def reset_performance_stats(self) -> None:
        """Reset performance statistics."""
        self.performance_stats = {
            'unified_manager_operations': 0,
            'vectorbt_optimizer_operations': 0,
            'pandas_fallback_operations': 0,
            'total_operations': 0
        }


# Global instance
_global_indicators: Optional[ConsolidatedTechnicalIndicators] = None


def get_consolidated_indicators(config: Optional[IndicatorConfig] = None) -> ConsolidatedTechnicalIndicators:
    """
    Get the global consolidated technical indicators instance.
    
    Args:
        config: Optional configuration
        
    Returns:
        Consolidated technical indicators instance
    """
    global _global_indicators
    
    if _global_indicators is None:
        _global_indicators = ConsolidatedTechnicalIndicators(config)
    
    return _global_indicators


def create_consolidated_indicators(config: Optional[IndicatorConfig] = None) -> ConsolidatedTechnicalIndicators:
    """
    Create a new consolidated technical indicators instance.
    
    Args:
        config: Optional configuration
        
    Returns:
        New consolidated technical indicators instance
    """
    return ConsolidatedTechnicalIndicators(config)