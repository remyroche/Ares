from __future__ import annotations

"""
Utility functions and common patterns for the Strategist module.
"""

import asyncio
import functools
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import TypeVar, Any, Callable
import numpy as np
import pandas as pd

T = TypeVar('T')

class StrategistError(Exception):
    """Base exception for strategist-specific errors."""

class ValidationError(StrategistError):
    """Raised when validation fails."""

class CalculationError(StrategistError):
    """Raised when calculation fails."""

def log_error(logger: logging.Logger, message: str, exception: Exception | None=None) -> None:
    """
    Centralized error logging with consistent formatting.

    Args:
        logger: Logger instance
        message: Error message
        exception: Optional exception to include
    """
    if exception:
        logger.error(f'{message}: {exception}')
    else:
        logger.error(message)

def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """
    Validate that a DataFrame contains required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Raises:
        ValidationError: If required columns are missing
    """
    if df is None or df.empty:
        msg = 'DataFrame is None or empty'
        raise ValidationError(msg)
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        msg = f'Missing required columns: {missing_columns}'
        raise ValidationError(msg)

def validate_data_sufficiency(df: pd.DataFrame, min_rows: int=100) -> None:
    """
    Validate that a DataFrame has sufficient data.

    Args:
        df: DataFrame to validate
        min_rows: Minimum number of rows required

    Raises:
        ValidationError: If insufficient data
    """
    if len(df) < min_rows:
        msg = f'Insufficient data: {len(df)} rows, minimum {min_rows} required'
        raise ValidationError(msg)

class PerformanceOptimizer:
    """Performance optimization utilities for market calculations."""

    def __init__(self, use_vectorized: bool=True, use_parallel: bool=True, cache_ttl: int=300) -> None:
        self.use_vectorized = use_vectorized
        self.use_parallel = use_parallel
        self.cache_ttl = cache_ttl
        self._executor = ThreadPoolExecutor(max_workers=4) if use_parallel else None

    def __del__(self) -> None:
        """Cleanup executor on deletion."""
        if self._executor:
            self._executor.shutdown(wait=False)

    @lru_cache(maxsize=128)
    def calculate_rsi_vectorized(self, prices: tuple, window: int=14) -> float:
        """
        Vectorized RSI calculation with caching.

        Args:
            prices: Tuple of prices (for hashability in cache)
            window: RSI window period

        Returns:
            RSI value
        """
        prices_array = np.array(prices)
        if len(prices_array) < window + 1:
            return 50.0
        deltas = np.diff(prices_array)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-window:])
        avg_loss = np.mean(losses[-window:])
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        rs = avg_gain / avg_loss
        rsi = 100 - 100 / (1 + rs)
        return float(rsi)

    def calculate_sma_vectorized(self, prices: np.ndarray, window: int) -> float:
        """
        Vectorized SMA calculation.

        Args:
            prices: Price array
            window: SMA window period

        Returns:
            SMA value
        """
        if len(prices) < window:
            return float(np.mean(prices))
        return float(np.mean(prices[-window:]))

    def calculate_volatility_vectorized(self, prices: np.ndarray, window: int=20) -> float:
        """
        Vectorized volatility calculation.

        Args:
            prices: Price array
            window: Volatility window period

        Returns:
            Volatility (standard deviation of returns)
        """
        if len(prices) < window + 1:
            return 0.0
        returns = np.diff(prices[-window - 1:]) / prices[-window - 1:-1]
        return float(np.std(returns))

    async def calculate_indicators_parallel(self, prices: pd.Series, volume: pd.Series, config: dict[str, Any]) -> dict[str, float]:
        """
        Calculate multiple indicators in parallel.

        Args:
            prices: Price series
            volume: Volume series
            config: Configuration with indicator parameters

        Returns:
            Dictionary of calculated indicators
        """
        if not self.use_parallel:
            return self._calculate_indicators_sequential(prices, volume, config)
        prices_array = prices.values
        prices_tuple = tuple(prices_array)
        volume_array = volume.values
        tasks = {'rsi': lambda: self.calculate_rsi_vectorized(prices_tuple, config.get('rsi_window', 14)), 'sma_fast': lambda: self.calculate_sma_vectorized(prices_array, config['sma_fast_window']), 'sma_slow': lambda: self.calculate_sma_vectorized(prices_array, config['sma_slow_window']), 'volatility': lambda: self.calculate_volatility_vectorized(prices_array, config['price_volatility_window']), 'volume_ratio': lambda: float(volume_array[-1] / np.mean(volume_array[-20:])) if len(volume_array) >= 20 else 1.0}
        loop = asyncio.get_event_loop()
        futures = {name: loop.run_in_executor(self._executor, func) for name, func in tasks.items()}
        results = {}
        for name, future in futures.items():
            try:
                results[name] = await future
            except Exception:
                results[name] = None
        return results

    def _calculate_indicators_sequential(self, prices: pd.Series, volume: pd.Series, config: dict[str, Any]) -> dict[str, float]:
        """Sequential fallback for indicator calculation."""
        prices_array = prices.values
        prices_tuple = tuple(prices_array)
        volume_array = volume.values
        return {'rsi': self.calculate_rsi_vectorized(prices_tuple, config.get('rsi_window', 14)), 'sma_fast': self.calculate_sma_vectorized(prices_array, config['sma_fast_window']), 'sma_slow': self.calculate_sma_vectorized(prices_array, config['sma_slow_window']), 'volatility': self.calculate_volatility_vectorized(prices_array, config['price_volatility_window']), 'volume_ratio': float(volume_array[-1] / np.mean(volume_array[-20:])) if len(volume_array) >= 20 else 1.0}

def create_strategy_validator(min_confidence: float=0.0, max_confidence: float=1.0) -> Callable:
    """
    Factory function to create strategy validation decorators.

    Args:
        min_confidence: Minimum allowed confidence
        max_confidence: Maximum allowed confidence

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:

        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> None:
            result = await func(*args, **kwargs)
            if result and isinstance(result, dict):
                confidence = result.get('confidence', 0)
                if not min_confidence <= confidence <= max_confidence:
                    result['confidence'] = max(min_confidence, min(confidence, max_confidence))
                    result.setdefault('reasoning', []).append(f'Confidence adjusted to range [{min_confidence}, {max_confidence}]')
                if 'direction' in result and result['direction'] not in ['BUY', 'SELL', 'HOLD']:
                    result['direction'] = 'HOLD'
                    result.setdefault('reasoning', []).append('Invalid direction corrected to HOLD')
            return result
        return wrapper
    return decorator

class StrategyComponentExtractor:
    """Extract and organize strategy components to reduce complexity."""

    @staticmethod
    def extract_market_health(analysis_results: dict[str, Any]) -> dict[str, Any]:
        """Extract market health information from analysis results."""
        market_health = analysis_results.get('market_health', {})
        if not market_health:
            return {}
        health_score = market_health.get('health_score', 0.5)
        return {'health_score': health_score, 'health_impact': health_score, 'reasoning': f'Market health score: {health_score:.3f}'}

    @staticmethod
    def extract_liquidation_risk(analysis_results: dict[str, Any]) -> dict[str, Any]:
        """Extract liquidation risk information from analysis results."""
        liquidation_risk = analysis_results.get('liquidation_risk', {})
        if not liquidation_risk:
            return {}
        risk_level = liquidation_risk.get('risk_level', 'MEDIUM')
        confidence_multiplier = 0.8 if risk_level == 'HIGH' else 1.0
        return {'risk_level': risk_level, 'confidence_multiplier': confidence_multiplier, 'reasoning': 'High liquidation risk - reduced confidence' if risk_level == 'HIGH' else None}

    @staticmethod
    def extract_trading_decision(analysis_results: dict[str, Any]) -> dict[str, Any]:
        """Extract trading decision from dual model system."""
        trading_decision = analysis_results.get('trading_decision', {})
        if not trading_decision:
            return {}
        return {'direction': trading_decision.get('direction', 'HOLD'), 'confidence': trading_decision.get('final_confidence', 0.0), 'dual_model_direction': trading_decision.get('direction', 'HOLD'), 'dual_model_confidence': trading_decision.get('final_confidence', 0.0), 'reasoning': 'Direction and confidence set by DualModelSystem'}