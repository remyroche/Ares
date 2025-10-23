#!/usr/bin/env python3
from src.utils.tprint import tprint
import warnings

from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any
from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

# Import PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

"""Profit-Based Feature Engineering System.

This module provides comprehensive profit-based feature engineering capabilities
for financial time series data, leveraging profit percentage information from
triple barrier labeling to create rich feature sets for machine learning models.
"""

import time

# Import essential decorators

# Import Numba for performance optimization

try:
    from numba import jit
    import logging

    NUMBA_AVAILABLE = False  # Temporarily disabled for logger compatibility
except ImportError:
    NUMBA_AVAILABLE = False
    jit = lambda *args, **kwargs: lambda func: func

# Numba-optimized functions for performance-critical operations
if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def _numba_profit_momentum(profit_pcts: np.ndarray, window: int) -> np.ndarray:
        """Numba-optimized profit momentum calculation."""
        n = len(profit_pcts)
        momentum = np.zeros(n)

        for i in range(window, n):
            momentum[i] = np.mean(profit_pcts[i-window:i])

        return momentum

    @jit(nopython=True, cache=True)
    def _numba_profit_volatility(profit_pcts: np.ndarray, window: int) -> np.ndarray:
        """Numba-optimized profit volatility calculation."""
        n = len(profit_pcts)
        volatility = np.zeros(n)

        for i in range(window, n):
            volatility[i] = np.std(profit_pcts[i-window:i])

        return volatility

    @jit(nopython=True, cache=True)
    def _numba_profit_rolling_stats(
        profit_pcts: np.ndarray,
        window: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Numba-optimized rolling statistics calculation."""
        n = len(profit_pcts)
        rolling_mean = np.zeros(n)
        rolling_std = np.zeros(n)
        rolling_max = np.zeros(n)
        rolling_min = np.zeros(n)

        for i in range(window, n):
            window_data = profit_pcts[i-window:i]
            rolling_mean[i] = np.mean(window_data)
            rolling_std[i] = np.std(window_data)
            rolling_max[i] = np.max(window_data)
            rolling_min[i] = np.min(window_data)

        return rolling_mean, rolling_std, rolling_max, rolling_min
else:
    # Fallback implementations without Numba
    def _numba_profit_momentum(profit_pcts: np.ndarray, window: int) -> np.ndarray:
        """Fallback profit momentum calculation without Numba."""
        return pd.Series(profit_pcts).rolling(window=window, min_periods=1).mean().values

    def _numba_profit_volatility(profit_pcts: np.ndarray, window: int) -> np.ndarray:
        """Fallback profit volatility calculation without Numba."""
        return pd.Series(profit_pcts).rolling(window=window, min_periods=1).std().values

    def _numba_profit_rolling_stats(
        profit_pcts: np.ndarray,
        window: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fallback rolling statistics calculation without Numba."""
        series = pd.Series(profit_pcts)
        rolling_mean = series.rolling(window=window, min_periods=1).mean().values
        rolling_std = series.rolling(window=window, min_periods=1).std().values
        rolling_max = series.rolling(window=window, min_periods=1).max().values
        rolling_min = series.rolling(window=window, min_periods=1).min().values
        return rolling_mean, rolling_std, rolling_max, rolling_min

class ProfitBasedFeatureEngineering:
    """
    Comprehensive profit-based feature engineering system.

    This class provides extensive feature engineering capabilities based on profit
    percentage data from triple barrier labeling. It includes multiple feature
    categories with performance optimizations and comprehensive validation.
    """
    def __init__(
        self,
        profit_column: str = "potential_profit_pct",
        volume_column: str = "volume",
        price_column: str = "close",
        use_numba: bool = True,
        memory_efficient: bool = True,
        enable_m1_optimizations: bool = True,
        enable_gpu_acceleration: bool = True,
    ) -> None:
        """Initialize the profit-based feature engineering system.

        Args:
            profit_column: Name of the profit percentage column
            volume_column: Name of the volume column
            price_column: Name of the price column
            use_numba: Whether to use Numba acceleration
            memory_efficient: Whether to use memory-efficient operations
            enable_m1_optimizations: Whether to enable M1-specific optimizations
            enable_gpu_acceleration: Whether to enable
        """
        self.profit_column = profit_column
        self.volume_column = volume_column
        self.price_column = price_column
        self.use_numba = use_numba and NUMBA_AVAILABLE
        self.memory_efficient = memory_efficient
        try:
            from src.utils.m1_optimizations import M1_OPTIMIZATIONS_AVAILABLE
            self.enable_m1_optimizations = enable_m1_optimizations and M1_OPTIMIZATIONS_AVAILABLE
        except ImportError:
            self.enable_m1_optimizations = False
        self.enable_gpu_acceleration = enable_gpu_acceleration and TORCH_AVAILABLE

        # Initialize logger
        self.logger = system_logger.getChild("System.ProfitBasedFeatureEngineering")

        # Initialize M1 optimizations
        if self.enable_m1_optimizations:
            try:
                from src.utils.m1_optimizations import get_m1_memory_optimizer, get_unified_matrix_operations, get_memory_manager
                self.m1_memory_optimizer = get_m1_memory_optimizer()
                self.matrix_ops = get_unified_matrix_operations()
                self.memory_monitor = get_memory_manager()
                self.logger.info("✅ M1 optimizations initialized for profit feature engineering")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize M1 optimizations: {e}")
                self.enable_m1_optimizations = False
        else:
            self.m1_memory_optimizer = None
            self.matrix_ops = None
            self.memory_monitor = None

        # Feature configuration
        self.feature_config = {
            "basic_profit": True,
            "categorical": True,
            "risk_reward": True,
            "momentum": True,
            "volatility": True,
            "volume": True,
            "rolling": True,
            "profit_bins": [-np.inf, -0.005, -0.002, -0.001, 0, 0.001, 0.002, 0.005, np.inf],
            "profit_labels": [
                "Large Loss", "Medium Loss", "Small Loss", "Tiny Loss",
                "No Profit", "Tiny Profit", "Small Profit", "Large Profit"
            ]
        }

        # Performance tracking
        self.performance_metrics = {
            "total_features_generated": 0,
            "processing_time": 0.0,
            "memory_usage": 0.0,
        }

        self.logger.info("🔧 Profit-based feature engineering system initialized")
        if self.use_numba:
            self.logger.info("🚀 Using Numba acceleration")
        else:
            self.logger.info("🐍 Using Python vectorized operations")

    @handles_errors(
        exceptions=(ValueError, TypeError, MemoryError),
        default_return=pd.DataFrame(),
        context="profit_feature_generation.utils.apply_all_features"
    )
    def apply_all_features(
        self,
        data: pd.DataFrame,
        feature_categories: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Apply all profit-based feature engineering categories.

        Args:
            data: Input DataFrame with profit percentage data
            feature_categories: Specific feature categories to apply

        Returns:
            DataFrame with all profit-based features added
        """
        start_time = time.time()

        # Generate unique correlation ID for tracking
        import uuid
        correlation_id = str(uuid.uuid4())[:8]

        self.logger.info(f"➡️ ProfitFeatures.apply_all start {correlation_id}")
        self.logger.info(f"🚀 Applying profit-based feature engineering {correlation_id}")
        self.logger.info(f"   - Input shape: {data.shape} {correlation_id}")

        # Determine which feature categories to apply
        if feature_categories is None:
            feature_categories = list(self.feature_config.keys())

        self.logger.info(f"   - Feature categories: {feature_categories} {correlation_id}")

        # Validate input data
        if data.empty:
            self.logger.error(f"❌ Input data is empty {correlation_id}")
            return data

        if self.profit_column not in data.columns:
            self.logger.error(f"❌ Profit column '{self.profit_column}' not found in data {correlation_id}")
            return data

        # Apply each feature category
        result_data = data.copy()

        if "basic_profit" in feature_categories:
            result_data = self._apply_basic_profit_features(result_data)
            self.logger.info(f"   ✅ Applied basic_profit features {correlation_id}")

        if "categorical" in feature_categories:
            result_data = self._apply_categorical_features(result_data)
            self.logger.info(f"   ✅ Applied categorical features {correlation_id}")

        if "risk_reward" in feature_categories:
            result_data = self._apply_risk_reward_features(result_data)
            self.logger.info(f"   ✅ Applied risk_reward features {correlation_id}")

        if "momentum" in feature_categories:
            result_data = self._apply_momentum_features(result_data)
            self.logger.info(f"   ✅ Applied momentum features {correlation_id}")

        if "volatility" in feature_categories:
            result_data = self._apply_volatility_features(result_data)
            self.logger.info(f"   ✅ Applied volatility features {correlation_id}")

        if "volume" in feature_categories:
            result_data = self._apply_volume_features(result_data)
            self.logger.info(f"   ✅ Applied volume features {correlation_id}")

        if "rolling" in feature_categories:
            result_data = self._apply_rolling_features(result_data)
            self.logger.info(f"   ✅ Applied rolling features {correlation_id}")

        # Calculate performance metrics
        processing_time = time.time() - start_time
        features_added = len(result_data.columns) - len(data.columns)

        self.logger.info(f"✅ Feature engineering completed {correlation_id}")
        self.logger.info(f"   - Output shape: {result_data.shape} {correlation_id}")
        self.logger.info(f"   - Features added: {features_added} {correlation_id}")
        self.logger.info(f"   - Total features: {len(result_data.columns)} {correlation_id}")

        # Update performance metrics
        self.performance_metrics.update({
            "total_features_generated": features_added,
            "processing_time": processing_time,
            "memory_usage": result_data.memory_usage(deep=True).sum() / 1024**3  # GB
        })

        self.logger.info(f"✅ ProfitFeatures.apply_all done {correlation_id}")

        return result_data

    def apply_all_m1_optimized(self, data: pd.DataFrame, feature_categories: Optional[List[str]] = None) -> pd.DataFrame:
        """Apply all profit-based features with M1 optimization.

        Args:
            data: Input DataFrame with profit data
            feature_categories: List of feature categories to apply

        Returns:
            DataFrame with all profit-based features added using M1 optimization
        """
        if not self.enable_m1_optimizations:
            self.logger.warning("⚠️ M1 optimizations not available, falling back to standard method")
            return self.apply_all(data, feature_categories)

        start_time = time.time()

        # Generate unique correlation ID for tracking
        correlation_id = str(uuid.uuid4())[:8]

        self.logger.info(f"🚀 M1-optimized profit feature engineering start {correlation_id}")
        self.logger.info(f"   - Input shape: {data.shape} {correlation_id}")

        # Memory checkpoint for M1 optimization
        with self.m1_memory_optimizer.memory_checkpoint(f"profit_features_{correlation_id}"):
            # Check if data should be processed in chunks
            data_size_mb = data.memory_usage(deep=True).sum() / (1024**2)

            if self.m1_memory_optimizer.should_chunk_data(data_size_mb, "profit_engineering"):
                self.logger.info(f"📦 Processing large dataset ({data_size_mb:.1f}MB) in chunks {correlation_id}")
                return self._chunked_profit_feature_engineering(data, feature_categories, correlation_id)

            # Use GPU acceleration if available
            if self.enable_gpu_acceleration and self.matrix_ops:
                self.logger.info(f"🎯 Using GPU acceleration for profit features")
                return self._gpu_accelerated_profit_features(data, feature_categories, correlation_id)

            # Standard M1-optimized processing
            return self._m1_optimized_profit_features(data, feature_categories, correlation_id)

    def _chunked_profit_feature_engineering(self, data: pd.DataFrame, feature_categories: Optional[List[str]], correlation_id: str) -> pd.DataFrame:
        """Process profit features in memory-efficient chunks."""
        chunk_size = self.m1_memory_optimizer.calculate_optimal_chunk_size(data.shape, "profit_engineering")
        self.logger.info(f"📏 Using chunk size: {chunk_size} {correlation_id}")

        processed_chunks = []
        total_rows = len(data)

        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk = data.iloc[start_idx:end_idx].copy()

            # Process chunk with M1 optimization
            processed_chunk = self._m1_optimized_profit_features(chunk, feature_categories, f"{correlation_id}_chunk_{start_idx}")
            processed_chunks.append(processed_chunk)

            # Memory cleanup between chunks
            if start_idx % (chunk_size * 3) == 0:
                self.m1_memory_optimizer.optimize_memory()

        # Combine results efficiently
        result = self.m1_memory_optimizer.memory_efficient_concat(processed_chunks)
        self.logger.info(f"✅ Chunked processing completed {correlation_id}")
        return result

    def _gpu_accelerated_profit_features(self, data: pd.DataFrame, feature_categories: Optional[List[str]], correlation_id: str) -> pd.DataFrame:
        """Use
        try:
            with self.matrix_ops.gpu_context(f"profit_features_{correlation_id}"):
                # Convert profit data to tensor
                profit_tensor = torch.from_numpy(data[self.profit_column].values.astype(np.float32))
                profit_tensor = self.matrix_ops.gpu_manager.to_device(profit_tensor, "profit_calculations")

                # GPU-accelerated basic features
                result_data = data.copy()

                if "basic_profit" in (feature_categories or self.feature_config.keys()):
                    # GPU-accelerated basic profit features
                    profit_squared = (profit_tensor ** 2).cpu().numpy()
                    profit_cubed = (profit_tensor ** 3).cpu().numpy()
                    profit_abs = torch.abs(profit_tensor).cpu().numpy()

                    result_data[f"{self.profit_column}_squared"] = profit_squared
                    result_data[f"{self.profit_column}_cubed"] = profit_cubed
                    result_data[f"{self.profit_column}_abs"] = profit_abs

                if "momentum" in (feature_categories or self.feature_config.keys()):
                    # GPU-accelerated momentum features
                    momentum_features = self._gpu_calculate_momentum_features(profit_tensor)
                    for feature_name, feature_values in momentum_features.items():
                        result_data[feature_name] = feature_values.cpu().numpy()

                if "volatility" in (feature_categories or self.feature_config.keys()):
                    # GPU-accelerated volatility features
                    volatility_features = self._gpu_calculate_volatility_features(profit_tensor)
                    for feature_name, feature_values in volatility_features.items():
                        result_data[feature_name] = feature_values.cpu().numpy()

                self.logger.info(f"✅ GPU-accelerated profit features completed {correlation_id}")
                return result_data

        except Exception as e:
            self.logger.warning(f"⚠️
            return self._m1_optimized_profit_features(data, feature_categories, correlation_id)

    def _m1_optimized_profit_features(self, data: pd.DataFrame, feature_categories: Optional[List[str]], correlation_id: str) -> pd.DataFrame:
        """M1-optimized profit feature engineering."""
        # Use M1 memory-efficient operations
        result_data = data.copy()

        # Apply features with M1 optimization
        if "basic_profit" in (feature_categories or self.feature_config.keys()):
            result_data = self._apply_basic_profit_features_m1_optimized(result_data)

        if "momentum" in (feature_categories or self.feature_config.keys()):
            result_data = self._apply_momentum_features_m1_optimized(result_data)

        if "volatility" in (feature_categories or self.feature_config.keys()):
            result_data = self._apply_volatility_features_m1_optimized(result_data)

        # Apply other features using standard methods
        if "categorical" in (feature_categories or self.feature_config.keys()):
            result_data = self._apply_categorical_features(result_data)

        if "risk_reward" in (feature_categories or self.feature_config.keys()):
            result_data = self._apply_risk_reward_features(result_data)

        if "volume" in (feature_categories or self.feature_config.keys()):
            result_data = self._apply_volume_features(result_data)

        if "rolling" in (feature_categories or self.feature_config.keys()):
            result_data = self._apply_rolling_features(result_data)

        self.logger.info(f"✅ M1-optimized profit features completed {correlation_id}")
        return result_data

    def _gpu_calculate_momentum_features(self, profit_tensor) -> Dict[str, Any]:
        """Calculate momentum features using
        if not TORCH_AVAILABLE or torch is None:
            # Fallback to numpy-based calculation
            return self._cpu_calculate_momentum_features(profit_tensor)

        features = {}

        # GPU-accelerated rolling momentum calculations
        for window in [5, 10, 20]:
            # Rolling mean (momentum)
            rolling_mean = torch.nn.functional.avg_pool1d(
                profit_tensor.unsqueeze(0).unsqueeze(0),
                kernel_size=window,
                stride=1,
                padding=window//2
            ).squeeze()

            # Rolling std (momentum volatility)
            rolling_std = torch.sqrt(torch.nn.functional.avg_pool1d(
                (profit_tensor - rolling_mean).pow(2).unsqueeze(0).unsqueeze(0),
                kernel_size=window,
                stride=1,
                padding=window//2
            ).squeeze())

            features[f"{self.profit_column}_momentum_{window}"] = rolling_mean
            features[f"{self.profit_column}_momentum_vol_{window}"] = rolling_std

        return features

    def _cpu_calculate_momentum_features(self, profit_array) -> Dict[str, Any]:
        """Calculate momentum features using CPU (numpy) as fallback."""
        features = {}

        # Convert to numpy if needed
        if hasattr(profit_array, 'numpy'):
            profit_array = profit_array.numpy()
        elif hasattr(profit_array, 'cpu'):
            profit_array = profit_array.cpu().numpy()

        # CPU-based rolling momentum calculations
        for window in [5, 10, 20]:
            # Rolling mean (momentum)
            rolling_mean = pd.Series(profit_array).rolling(window=window).mean()
            features[f'momentum_mean_{window}'] = rolling_mean.values

            # Rolling standard deviation (volatility)
            rolling_std = pd.Series(profit_array).rolling(window=window).std()
            features[f'momentum_std_{window}'] = rolling_std.values

            # Rolling max (peak momentum)
            rolling_max = pd.Series(profit_array).rolling(window=window).max()
            features[f'momentum_max_{window}'] = rolling_max.values

            # Rolling min (trough momentum)
            rolling_min = pd.Series(profit_array).rolling(window=window).min()
            features[f'momentum_min_{window}'] = rolling_min.values

        return features

    def _gpu_calculate_volatility_features(self, profit_tensor) -> Dict[str, Any]:
        """Calculate volatility features using
        if not TORCH_AVAILABLE or torch is None:
            # Fallback to numpy-based calculation
            return self._cpu_calculate_volatility_features(profit_tensor)

        features = {}

        # GPU-accelerated volatility calculations
        for window in [10, 20, 50]:
            # Rolling volatility
            rolling_vol = torch.sqrt(torch.nn.functional.avg_pool1d(
                profit_tensor.pow(2).unsqueeze(0).unsqueeze(0),
                kernel_size=window,
                stride=1,
                padding=window//2
            ).squeeze())

            features[f"{self.profit_column}_volatility_{window}"] = rolling_vol

        return features

    def _cpu_calculate_volatility_features(self, profit_array) -> Dict[str, Any]:
        """Calculate volatility features using CPU (numpy) as fallback."""
        features = {}

        # Convert to numpy if needed
        if hasattr(profit_array, 'numpy'):
            profit_array = profit_array.numpy()
        elif hasattr(profit_array, 'cpu'):
            profit_array = profit_array.cpu().numpy()

        # CPU-based volatility calculations
        for window in [10, 20, 50]:
            # Rolling volatility
            rolling_vol = pd.Series(profit_array).rolling(window=window).std()
            features[f'{self.profit_column}_volatility_{window}'] = rolling_vol.values

        return features

    def _apply_basic_profit_features_m1_optimized(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply basic profit features with M1 optimization."""
        profit_pcts = data[self.profit_column].values

        # Use M1 memory-efficient array creation
        if self.m1_memory_optimizer:
            profit_array = self.m1_memory_optimizer.create_memory_efficient_array(profit_pcts, np.float32)
        else:
            profit_array = profit_pcts.astype(np.float32)

        # M1-optimized basic features
        data[f"{self.profit_column}_squared"] = profit_array ** 2
        data[f"{self.profit_column}_cubed"] = profit_array ** 3
        data[f"{self.profit_column}_abs"] = np.abs(profit_array)
        data[f"{self.profit_column}_sqrt"] = np.sqrt(np.abs(profit_array))

        return data

    def _apply_momentum_features_m1_optimized(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply momentum features with M1 optimization."""
        profit_pcts = data[self.profit_column].values

        # Use Numba-optimized functions if available
        if self.use_numba:
            for window in [5, 10, 20]:
                momentum = _numba_profit_momentum(profit_pcts, window)
                data[f"{self.profit_column}_momentum_{window}"] = momentum
        else:
            # Fallback to pandas rolling
            for window in [5, 10, 20]:
                data[f"{self.profit_column}_momentum_{window}"] = data[self.profit_column].rolling(window=window, min_periods=1).mean()

        return data

    def _apply_volatility_features_m1_optimized(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply volatility features with M1 optimization."""
        profit_pcts = data[self.profit_column].values

        # Use Numba-optimized functions if available
        if self.use_numba:
            for window in [10, 20, 50]:
                volatility = _numba_profit_volatility(profit_pcts, window)
                data[f"{self.profit_column}_volatility_{window}"] = volatility
        else:
            # Fallback to pandas rolling
            for window in [10, 20, 50]:
                data[f"{self.profit_column}_volatility_{window}"] = data[self.profit_column].rolling(window=window, min_periods=1).std()

        return data

    @handles_errors(
        exceptions=(ValueError, TypeError),
        default_return=pd.DataFrame(),
        context="basic_profit_features"
    )
    @log_all_calls
    def _apply_basic_profit_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply basic profit features.

        Features: profit, profit^2, profit^3, profit_abs
        """
        profit_pcts = data[self.profit_column].values

        # Basic profit features
        data[f"{self.profit_column}_squared"] = profit_pcts ** 2
        data[f"{self.profit_column}_cubed"] = profit_pcts ** 3
        data[f"{self.profit_column}_abs"] = np.abs(profit_pcts)

        # Additional basic features
        data[f"{self.profit_column}_sqrt"] = np.sqrt(np.abs(profit_pcts))
        data[f"{self.profit_column}_log"] = np.log1p(np.abs(profit_pcts))

        return data

    @handles_errors(
        exceptions=(ValueError, TypeError),
        default_return=pd.DataFrame(),
        context="categorical_features"
    )
    @log_all_calls
    def _apply_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply categorical profit features.

        Features: profit_bins, profit_sign, profit_magnitude
        """
        profit_pcts = data[self.profit_column].values

        # Profit sign (positive/negative)
        data[f"{self.profit_column}_sign"] = np.sign(profit_pcts)

        # Profit magnitude categories
        profit_abs = np.abs(profit_pcts)
        data[f"{self.profit_column}_magnitude"] = pd.cut(
            profit_abs,
            bins=[0, 0.001, 0.002, 0.005, np.inf],
            labels=["Tiny", "Small", "Medium", "Large"],
            include_lowest=True
        )

        # Profit bins (categorical)
        data[f"{self.profit_column}_bins"] = pd.cut(
            profit_pcts,
            bins=self.feature_config["profit_bins"],
            labels=self.feature_config["profit_labels"],
            include_lowest=True
        )

        # Profit direction strength
        data[f"{self.profit_column}_direction_strength"] = profit_abs * np.sign(profit_pcts)

        return data

    @handles_errors(
        exceptions=(ValueError, TypeError),
        default_return=pd.DataFrame(),
        context="risk_reward_features"
    )
    @log_all_calls
    def _apply_risk_reward_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply risk-reward features.

        Features: profit_sharpe, profit_sortino, profit_risk_adjusted
        """
        profit_pcts = data[self.profit_column].values

        # Calculate rolling statistics for risk-reward metrics
        window = 20
        rolling_mean = data[self.profit_column].rolling(window=window, min_periods=1).mean()
        rolling_std = data[self.profit_column].rolling(window=window, min_periods=1).std()

        # Sharpe ratio (profit per unit of risk)
        data[f"{self.profit_column}_sharpe"] = np.where(
            rolling_std > 0,
            rolling_mean / rolling_std,
            0.0
        )

        # Sortino ratio (profit per unit of downside risk)
        downside_returns = np.where(profit_pcts < 0, profit_pcts, 0)
        downside_std = pd.Series(downside_returns).rolling(window=window, min_periods=1).std()
        downside_std = downside_std.reindex(rolling_mean.index)

        sortino_ratio = np.where(
            downside_std > 0,
            rolling_mean / downside_std,
            0.0
        )
        data[f"{self.profit_column}_sortino"] = pd.Series(sortino_ratio, index=data.index).fillna(0.0)

        # Kelly criterion removed - it is for position sizing, not ML features

        # Risk-adjusted return
        data[f"{self.profit_column}_risk_adjusted"] = profit_pcts / (1 + rolling_std)

        return data

    @handles_errors(
        exceptions=(ValueError, TypeError),
        default_return=pd.DataFrame(),
        context="momentum_features"
    )
    @log_all_calls
    def _apply_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply momentum features.

        Features: profit_momentum, profit_acceleration
        """
        profit_pcts = data[self.profit_column].values

        # Momentum features for different windows
        windows = [5, 10, 20]
        for window in windows:
            if self.use_numba:
                momentum = _numba_profit_momentum(profit_pcts, window)
            else:
                momentum = pd.Series(profit_pcts).rolling(window=window, min_periods=1).mean().values

            data[f"{self.profit_column}_momentum_{window}"] = momentum

        # Acceleration features (change in momentum)
        for window in [10, 20]:
            momentum_series = pd.Series(profit_pcts).rolling(window=window, min_periods=1).mean()
            acceleration = momentum_series.diff()
            data[f"{self.profit_column}_acceleration_{window}"] = acceleration.fillna(0)

        # Momentum ratio (short-term vs long-term)
        short_momentum = pd.Series(profit_pcts).rolling(window=5, min_periods=1).mean()
        long_momentum = pd.Series(profit_pcts).rolling(window=20, min_periods=1).mean()
        data[f"{self.profit_column}_momentum_ratio"] = np.where(
            long_momentum != 0,
            short_momentum / long_momentum,
            1.0
        )

        return data

    @handles_errors(
        exceptions=(ValueError, TypeError),
        default_return=pd.DataFrame(),
        context="volatility_features"
    )
    @log_all_calls
    def _apply_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply volatility features.

        Features: profit_volatility, profit_volatility_ratio
        """
        profit_pcts = data[self.profit_column].values

        # Volatility features for different windows
        windows = [10, 20, 50]
        for window in windows:
            if self.use_numba:
                volatility = _numba_profit_volatility(profit_pcts, window)
            else:
                volatility = pd.Series(profit_pcts).rolling(window=window, min_periods=1).std().values

            data[f"{self.profit_column}_volatility_{window}"] = volatility

        # Volatility ratio (current vs historical)
        for window in [20, 50]:
            current_vol = pd.Series(profit_pcts).rolling(window=window, min_periods=1).std()
            historical_vol = pd.Series(profit_pcts).rolling(window=window*2, min_periods=1).std()
            vol_ratio = np.where(
                historical_vol > 0,
                current_vol / historical_vol,
                1.0
            )
            data[f"{self.profit_column}_volatility_ratio_{window}"] = vol_ratio

        # Volatility surprise (realized vs expected)
        expected_vol = pd.Series(profit_pcts).rolling(window=20, min_periods=1).std()
        realized_vol = pd.Series(profit_pcts).rolling(window=5, min_periods=1).std()
        data[f"{self.profit_column}_volatility_surprise"] = np.where(
            expected_vol > 0,
            (realized_vol - expected_vol) / expected_vol,
            0.0
        )

        return data

    @handles_errors(
        exceptions=(ValueError, TypeError),
        default_return=pd.DataFrame(),
        context="volume_features"
    )
    @log_all_calls
    def _apply_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply volume-based profit features.

        Features: profit_volume_weighted, profit_volume_correlation
        """
        profit_pcts = data[self.profit_column].values

        # Volume-weighted profit
        if self.volume_column in data.columns:
            volume = data[self.volume_column].values
            volume_weighted_profit = profit_pcts * volume
            data[f"{self.profit_column}_volume_weighted"] = volume_weighted_profit

            # Volume-profit correlation (rolling)
            volume_series = pd.Series(volume, index=data.index)
            profit_series = pd.Series(profit_pcts, index=data.index)
            volume_corr = volume_series.rolling(window=20, min_periods=1).corr(profit_series)
            data[f"{self.profit_column}_volume_correlation"] = volume_corr.fillna(0)

            # Volume-adjusted profit
            volume_mean = volume_series.rolling(window=20, min_periods=1).mean()
            volume_std = volume_series.rolling(window=20, min_periods=1).std()
            volume_z_score = np.where(
                volume_std > 0,
                (volume - volume_mean) / volume_std,
                0.0
            )
            data[f"{self.profit_column}_volume_adjusted"] = profit_pcts * (1 + volume_z_score)

            # High volume profit signal
            volume_threshold = volume_series.rolling(window=20, min_periods=1).quantile(0.8)
            high_volume_signal = np.where(volume > volume_threshold, profit_pcts, 0)
            data[f"{self.profit_column}_high_volume_signal"] = high_volume_signal

        return data

    @handles_errors(
        exceptions=(ValueError, TypeError),
        default_return=pd.DataFrame(),
        context="rolling_features"
    )
    @log_all_calls
    def _apply_rolling_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply rolling profit features.

        Features: profit_rolling_mean, profit_rolling_std, profit_rolling_max
        """
        profit_pcts = data[self.profit_column].values

        # Rolling features for different windows
        windows = [5, 10, 20, 50]
        for window in windows:
            if self.use_numba:
                rolling_mean, rolling_std, rolling_max, rolling_min = _numba_profit_rolling_stats(profit_pcts, window)
                data[f"{self.profit_column}_rolling_mean_{window}"] = rolling_mean
                data[f"{self.profit_column}_rolling_std_{window}"] = rolling_std
                data[f"{self.profit_column}_rolling_max_{window}"] = rolling_max
                data[f"{self.profit_column}_rolling_min_{window}"] = rolling_min
            else:
                series = pd.Series(profit_pcts, index=data.index)
                data[f"{self.profit_column}_rolling_mean_{window}"] = series.rolling(window=window, min_periods=1).mean()
                data[f"{self.profit_column}_rolling_std_{window}"] = series.rolling(window=window, min_periods=1).std()
                data[f"{self.profit_column}_rolling_max_{window}"] = series.rolling(window=window, min_periods=1).max()
                data[f"{self.profit_column}_rolling_min_{window}"] = series.rolling(window=window, min_periods=1).min()

            # Additional rolling features
            series = pd.Series(profit_pcts, index=data.index)
            rolling_mean = series.rolling(window=window, min_periods=1).mean()
            rolling_std = series.rolling(window=window, min_periods=1).std()

            # Rolling range
            rolling_max = series.rolling(window=window, min_periods=1).max()
            rolling_min = series.rolling(window=window, min_periods=1).min()
            data[f"{self.profit_column}_rolling_range_{window}"] = rolling_max - rolling_min

            # Coefficient of variation
            data[f"{self.profit_column}_rolling_cv_{window}"] = np.where(
                rolling_mean != 0,
                rolling_std / np.abs(rolling_mean),
                0.0
            )

            # Rolling quantiles
            data[f"{self.profit_column}_rolling_q25_{window}"] = series.rolling(window=window, min_periods=1).quantile(0.25)
            data[f"{self.profit_column}_rolling_q75_{window}"] = series.rolling(window=window, min_periods=1).quantile(0.75)

        return data

    def get_feature_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of profit-based features.

        Args:
            data: DataFrame with profit-based features

        Returns:
            Dictionary with feature summary information
        """
        profit_features = [col for col in data.columns if self.profit_column in col and col != self.profit_column]

        # Categorize features
        feature_categories = {
            "basic_profit": [],
            "categorical": [],
            "risk_reward": [],
            "momentum": [],
            "volatility": [],
            "volume": [],
            "rolling": []
        }

        for feature in profit_features:
            if "squared" in feature or "cubed" in feature or "abs" in feature or "sqrt" in feature or "log" in feature:
                feature_categories["basic_profit"].append(feature)
            elif "sign" in feature or "magnitude" in feature or "bins" in feature or "direction_strength" in feature:
                feature_categories["categorical"].append(feature)
            elif "sharpe" in feature or "sortino" in feature or "risk_adjusted" in feature:
                feature_categories["risk_reward"].append(feature)
            elif "momentum" in feature or "acceleration" in feature:
                feature_categories["momentum"].append(feature)
            elif "volatility" in feature:
                feature_categories["volatility"].append(feature)
            elif "volume" in feature:
                feature_categories["volume"].append(feature)
            elif "rolling" in feature:
                feature_categories["rolling"].append(feature)

        return {
            "total_features": len(profit_features),
            "feature_categories": feature_categories,
            "performance_metrics": self.performance_metrics
        }
    @log_step_functions

    def select_features(
        self,
        data: pd.DataFrame,
        method: str = "correlation",
        threshold: float = 0.01,
        max_features: Optional[int] = None
    ) -> List[str]:
        """Select important profit-based features.

        Args:
            data: DataFrame with profit-based features
            method: Selection method ("correlation", "variance", "mutual_info")
            threshold: Selection threshold
            max_features: Maximum number of features to select

        Returns:
            List of selected feature names
        """
        profit_features = [col for col in data.columns if self.profit_column in col and col != self.profit_column]

        if method == "correlation":
            # Select features based on correlation with target
            # Filter out categorical features for correlation
            numerical_features = []
            for feature in profit_features:
                if data[feature].dtype in ['int64', 'float64']:
                    numerical_features.append(feature)

            if numerical_features:
                correlations = data[numerical_features].corrwith(data[self.profit_column]).abs()
                selected = correlations[correlations > threshold].index.tolist()
            else:
                selected = []

        elif method == "variance":
            # Select features based on variance
            # Filter out categorical features for variance
            numerical_features = []
            for feature in profit_features:
                if data[feature].dtype in ['int64', 'float64']:
                    numerical_features.append(feature)

            if numerical_features:
                variances = data[numerical_features].var()
                selected = variances[variances > threshold].index.tolist()
            else:
                selected = []

        elif method == "mutual_info":
            # Select features based on mutual information (requires scikit-learn)
            try:
                from sklearn.feature_selection import mutual_info_regression
                # Filter out categorical features for mutual info
                numerical_features = []
                for feature in profit_features:
                    if data[feature].dtype in ['int64', 'float64']:
                        numerical_features.append(feature)

                if numerical_features:
                    mi_scores = mutual_info_regression(
                        data[numerical_features].fillna(0),
                        data[self.profit_column]
                    )
                    mi_series = pd.Series(mi_scores, index=numerical_features)
                    selected = mi_series[mi_series > threshold].index.tolist()
                else:
                    selected = []
            except ImportError:
                self.logger.warning("scikit-learn not available, falling back to correlation method")
                # Filter out categorical features for correlation
                numerical_features = []
                for feature in profit_features:
                    if data[feature].dtype in ['int64', 'float64']:
                        numerical_features.append(feature)

                if numerical_features:
                    correlations = data[numerical_features].corrwith(data[self.profit_column]).abs()
                    selected = correlations[correlations > threshold].index.tolist()
                else:
                    selected = []

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")
                self.logger.warning("scikit-learn not available, falling back to correlation method")
                # Filter out categorical features for correlation
                numerical_features = []
                for feature in profit_features:
                    if data[feature].dtype in ['int64', 'float64']:
                        numerical_features.append(feature)

                if numerical_features:
                    correlations = data[numerical_features].corrwith(data[self.profit_column]).abs()
                    selected = correlations[correlations > threshold].index.tolist()
                else:
                    selected = []

        else:
            raise ValueError(f"Unknown feature selection method: {method}")

        # Limit number of features if specified
        if max_features is not None:
            selected = selected[:max_features]

        return selected

@handles_errors(exceptions=(Exception,), default_return={}, context="benchmark_profit_features")
def benchmark_profit_feature_engineering(data: pd.DataFrame) -> Dict[str, float]:
    """Benchmark profit-based feature engineering performance.

    Args:
        data: Input DataFrame with profit percentage data

    Returns:
        Dictionary with benchmark results
    """
    # Test with Numba
    start_time = time.time()
    feature_eng_numba = ProfitBasedFeatureEngineering(use_numba=True)
    result_numba = feature_eng_numba.apply_all_features(data)
    numba_time = time.time() - start_time
    numba_features = len([col for col in result_numba.columns if "potential_profit_pct" in col and col != "potential_profit_pct"])

    # Test without Numba
    start_time = time.time()
    feature_eng_python = ProfitBasedFeatureEngineering(use_numba=False)
    result_python = feature_eng_python.apply_all_features(data)
    python_time = time.time() - start_time
    python_features = len([col for col in result_python.columns if "potential_profit_pct" in col and col != "potential_profit_pct"])

    return {
        "numba_time": numba_time,
        "python_time": python_time,
        "speedup": python_time / numba_time if numba_time > 0 else 1.0,
        "numba_features": numba_features,
        "python_features": python_features
    }

if __name__ == "__main__":
    # Example usage

    # Create sample data
    dates = pd.date_range("2024-01-01", periods=1000, freq="1min")
    data = pd.DataFrame({
        'open': np.random.uniform(100, 110, 1000),
        'high': np.random.uniform(105, 115, 1000),
        'low': np.random.uniform(95, 105, 1000),
        'close': np.random.uniform(100, 110, 1000),
        'volume': np.random.uniform(1000, 10000, 1000),
        'potential_profit_pct': np.random.uniform(-0.01, 0.01, 1000),
    }, index=dates)

    # Initialize feature engineering
    feature_eng = ProfitBasedFeatureEngineering()

    # Apply all features
    result = feature_eng.apply_all_features(data)

    # Get feature summary
    summary = feature_eng.get_feature_summary(result)
    tprint(f"Generated {summary['total_features']} profit-based features")

    # Select important features
    selected_features = feature_eng.select_features(result, method="correlation", threshold=0.01)
    tprint(f"Selected {len(selected_features)} important features")

    # Benchmark performance
    benchmark_results = benchmark_profit_feature_engineering(data)
    tprint(f"Performance benchmark: {benchmark_results}")
#!/usr/bin/env python3

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and
                VECTORBT_AVAILABLE)

    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            self.logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

    def _pandas_rolling_operation(self, data: pd.Series, operation: str,
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
