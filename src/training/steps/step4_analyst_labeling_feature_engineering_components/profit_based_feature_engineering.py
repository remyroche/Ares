#!/usr/bin/env python3
"""Profit-Based Feature Engineering System.

This module provides comprehensive vectorized feature engineering capabilities
that leverage profit percentage data from the enhanced triple barrier method.
It includes multiple feature categories with performance optimizations.
"""

import contextlib
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd

from src.utils.centralized_decorators import (
    guard_dataframe_nulls,
    handle_errors,
    memory_efficient,
    with_tracing_span,
)
from src.utils.logger import get_logger

try:
    import numba  # type: ignore
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    numba = None

# Numba-optimized functions for performance-critical operations
if NUMBA_AVAILABLE and numba is not None:
    @numba.jit(nopython=True, cache=True)
    def _numba_profit_momentum(
        profit_pcts: np.ndarray, 
        window: int
    ) -> np.ndarray:
        """Numba-accelerated profit momentum calculation."""
        n = len(profit_pcts)
        momentum = np.zeros(n, dtype=np.float64)
        
        for i in range(window, n):
            momentum[i] = np.mean(profit_pcts[i-window:i]) - np.mean(profit_pcts[i-window*2:i-window])
        
        return momentum

    @numba.jit(nopython=True, cache=True)
    def _numba_profit_volatility(
        profit_pcts: np.ndarray, 
        window: int
    ) -> np.ndarray:
        """Numba-accelerated profit volatility calculation."""
        n = len(profit_pcts)
        volatility = np.zeros(n, dtype=np.float64)
        
        for i in range(window, n):
            window_data = profit_pcts[i-window:i]
            volatility[i] = np.std(window_data)
        
        return volatility

    @numba.jit(nopython=True, cache=True)
    def _numba_profit_rolling_stats(
        profit_pcts: np.ndarray, 
        window: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Numba-accelerated rolling statistics calculation."""
        n = len(profit_pcts)
        rolling_mean = np.zeros(n, dtype=np.float64)
        rolling_std = np.zeros(n, dtype=np.float64)
        rolling_max = np.zeros(n, dtype=np.float64)
        rolling_min = np.zeros(n, dtype=np.float64)
        
        for i in range(window, n):
            window_data = profit_pcts[i-window:i]
            rolling_mean[i] = np.mean(window_data)
            rolling_std[i] = np.std(window_data)
            rolling_max[i] = np.max(window_data)
            rolling_min[i] = np.min(window_data)
        
        return rolling_mean, rolling_std, rolling_max, rolling_min


class ProfitBasedFeatureEngineering:
    """Comprehensive profit-based feature engineering system.
    
    This class provides vectorized feature engineering capabilities that leverage
    profit percentage data from the enhanced triple barrier method. It includes
    multiple feature categories with performance optimizations.
    """

    def __init__(
        self,
        profit_column: str = "potential_profit_pct",
        volume_column: str = "volume",
        price_column: str = "close",
        use_numba: bool = True,
        memory_efficient: bool = True,
    ) -> None:
        """Initialize the profit-based feature engineering system.
        
        Args:
            profit_column: Name of the profit percentage column
            volume_column: Name of the volume column
            price_column: Name of the price column
            use_numba: Whether to use Numba acceleration
            memory_efficient: Whether to use memory-efficient operations
        """
        self.profit_column = profit_column
        self.volume_column = volume_column
        self.price_column = price_column
        self.use_numba = use_numba and NUMBA_AVAILABLE
        self.memory_efficient = memory_efficient
        self.logger = get_logger("ProfitBasedFeatureEngineering")
        
        # Feature configuration
        self.feature_config = {
            "rolling_windows": [5, 10, 20, 50],
            "volatility_windows": [10, 20, 50],
            "momentum_windows": [5, 10, 20],
            "profit_bins": [-np.inf, -0.005, -0.002, -0.001, 0, 0.001, 0.002, 0.005, np.inf],
            "profit_labels": [
                "Large Loss", "Medium Loss", "Small Loss", "Tiny Loss",
                "No Profit", "Tiny Profit", "Small Profit", "Large Profit"
            ]
        }
        
        self.logger.info("🔧 Profit-based feature engineering system initialized")
        if self.use_numba:
            self.logger.info("⚡ Numba acceleration enabled")
        else:
            self.logger.info("🐍 Using Python vectorized operations")

    @handle_errors(
        exceptions=(Exception,),
        default_return=pd.DataFrame(),
        context="profit_feature_engineering.apply_all_features"
    )
    @guard_dataframe_nulls(mode="warn", arg_index=1)
    @with_tracing_span("ProfitFeatures.apply_all", log_args=False)
    def apply_all_features(
        self, 
        data: pd.DataFrame,
        feature_categories: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Apply all profit-based feature engineering categories.
        
        Args:
            data: Input DataFrame with profit percentage data
            feature_categories: List of feature categories to apply
            
        Returns:
            DataFrame with all profit-based features added
        """
        if feature_categories is None:
            feature_categories = [
                "basic_profit",
                "categorical",
                "risk_reward", 
                "momentum",
                "volatility",
                "volume",
                "rolling"
            ]
        
        self.logger.info(f"🚀 Applying profit-based feature engineering")
        self.logger.info(f"   - Input shape: {data.shape}")
        self.logger.info(f"   - Feature categories: {feature_categories}")
        
        # Validate input data
        if self.profit_column not in data.columns:
            raise ValueError(f"Profit column '{self.profit_column}' not found in data")
        
        # Create copy for feature engineering
        if self.memory_efficient:
            result_data = data.copy()
        else:
            result_data = data.copy()
        
        # Apply each feature category
        for category in feature_categories:
            if hasattr(self, f"_apply_{category}_features"):
                method = getattr(self, f"_apply_{category}_features")
                result_data = method(result_data)
                self.logger.info(f"   ✅ Applied {category} features")
            else:
                self.logger.warning(f"   ⚠️ Unknown feature category: {category}")
        
        # Log feature engineering results
        original_cols = len(data.columns)
        new_cols = len(result_data.columns)
        added_cols = new_cols - original_cols
        
        self.logger.info(f"✅ Feature engineering completed")
        self.logger.info(f"   - Output shape: {result_data.shape}")
        self.logger.info(f"   - Features added: {added_cols}")
        self.logger.info(f"   - Total features: {new_cols}")
        
        return result_data

    @memory_efficient
    def _apply_basic_profit_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply basic profit features.
        
        Features: profit, profit², profit³, profit_abs
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

    @memory_efficient
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

    @memory_efficient
    def _apply_risk_reward_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply risk-reward features.
        
        Features: profit_sharpe, profit_kelly, profit_sortino
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
        data[f"{self.profit_column}_sortino"] = np.where(
            downside_std > 0,
            rolling_mean / downside_std,
            0.0
        )
        
        # Kelly criterion approximation
        win_rate = (profit_pcts > 0).rolling(window=window, min_periods=1).mean()
        avg_win = np.where(profit_pcts > 0, profit_pcts, 0).rolling(window=window, min_periods=1).mean()
        avg_loss = np.where(profit_pcts < 0, np.abs(profit_pcts), 0).rolling(window=window, min_periods=1).mean()
        
        data[f"{self.profit_column}_kelly"] = np.where(
            avg_loss > 0,
            (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win,
            0.0
        )
        
        # Risk-adjusted return
        data[f"{self.profit_column}_risk_adjusted"] = profit_pcts / (1 + rolling_std)
        
        return data

    @memory_efficient
    def _apply_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply momentum features.
        
        Features: profit_momentum, profit_acceleration
        """
        profit_pcts = data[self.profit_column].values
        
        # Apply momentum features for different windows
        for window in self.feature_config["momentum_windows"]:
            if self.use_numba and len(profit_pcts) > window * 2:
                momentum = _numba_profit_momentum(profit_pcts, window)
            else:
                # Python vectorized implementation
                momentum = np.zeros(len(profit_pcts))
                for i in range(window, len(profit_pcts)):
                    if i >= window * 2:
                        recent_mean = np.mean(profit_pcts[i-window:i])
                        previous_mean = np.mean(profit_pcts[i-window*2:i-window])
                        momentum[i] = recent_mean - previous_mean
            
            data[f"{self.profit_column}_momentum_{window}"] = momentum
            
            # Acceleration (change in momentum)
            if window > 5:
                data[f"{self.profit_column}_acceleration_{window}"] = np.diff(
                    data[f"{self.profit_column}_momentum_{window}"], 
                    prepend=data[f"{self.profit_column}_momentum_{window}"].iloc[0]
                )
        
        # Cross-momentum features
        if len(self.feature_config["momentum_windows"]) >= 2:
            short_window = self.feature_config["momentum_windows"][0]
            long_window = self.feature_config["momentum_windows"][-1]
            
            data[f"{self.profit_column}_momentum_ratio"] = np.where(
                data[f"{self.profit_column}_momentum_{long_window}"] != 0,
                data[f"{self.profit_column}_momentum_{short_window}"] / data[f"{self.profit_column}_momentum_{long_window}"],
                0.0
            )
        
        return data

    @memory_efficient
    def _apply_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply volatility features.
        
        Features: profit_volatility, profit_volatility_ratio
        """
        profit_pcts = data[self.profit_column].values
        
        # Apply volatility features for different windows
        for window in self.feature_config["volatility_windows"]:
            if self.use_numba and len(profit_pcts) > window:
                volatility = _numba_profit_volatility(profit_pcts, window)
            else:
                # Python vectorized implementation
                volatility = data[self.profit_column].rolling(window=window, min_periods=1).std().values
            
            data[f"{self.profit_column}_volatility_{window}"] = volatility
            
            # Volatility ratio (current vs historical)
            if window > 10:
                historical_vol = data[f"{self.profit_column}_volatility_{window}"].rolling(
                    window=window, min_periods=1
                ).mean()
                data[f"{self.profit_column}_volatility_ratio_{window}"] = np.where(
                    historical_vol > 0,
                    volatility / historical_vol,
                    1.0
                )
        
        # Realized vs expected volatility
        rolling_mean = data[self.profit_column].rolling(window=20, min_periods=1).mean()
        rolling_std = data[self.profit_column].rolling(window=20, min_periods=1).std()
        
        data[f"{self.profit_column}_volatility_surprise"] = np.where(
            rolling_std > 0,
            (profit_pcts - rolling_mean) / rolling_std,
            0.0
        )
        
        return data

    @memory_efficient
    def _apply_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply volume-based profit features.
        
        Features: profit_volume_returns
        """
        if self.volume_column not in data.columns:
            self.logger.warning(f"Volume column '{self.volume_column}' not found, skipping volume features")
            return data
        
        profit_pcts = data[self.profit_column].values
        volume = data[self.volume_column].values
        
        # Volume-weighted profit
        data[f"{self.profit_column}_volume_weighted"] = profit_pcts * volume
        
        # Volume-profit correlation (rolling)
        window = 20
        volume_rolling_mean = data[self.volume_column].rolling(window=window, min_periods=1).mean()
        profit_rolling_mean = data[self.profit_column].rolling(window=window, min_periods=1).mean()
        
        # Simplified correlation approximation
        volume_dev = volume - volume_rolling_mean
        profit_dev = profit_pcts - profit_rolling_mean
        
        data[f"{self.profit_column}_volume_correlation"] = volume_dev * profit_dev
        
        # Volume-adjusted profit
        volume_ratio = volume / volume_rolling_mean
        data[f"{self.profit_column}_volume_adjusted"] = profit_pcts / (1 + volume_ratio)
        
        # High volume profit signals
        volume_threshold = volume_rolling_mean * 1.5
        data[f"{self.profit_column}_high_volume_signal"] = np.where(
            volume > volume_threshold,
            profit_pcts,
            0.0
        )
        
        return data

    @memory_efficient
    def _apply_rolling_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply rolling profit features.
        
        Features: profit_rolling_mean, profit_rolling_std, profit_rolling_max
        """
        profit_pcts = data[self.profit_column].values
        
        # Apply rolling features for different windows
        for window in self.feature_config["rolling_windows"]:
            if self.use_numba and len(profit_pcts) > window:
                rolling_mean, rolling_std, rolling_max, rolling_min = _numba_profit_rolling_stats(
                    profit_pcts, window
                )
            else:
                # Python vectorized implementation
                rolling_mean = data[self.profit_column].rolling(window=window, min_periods=1).mean().values
                rolling_std = data[self.profit_column].rolling(window=window, min_periods=1).std().values
                rolling_max = data[self.profit_column].rolling(window=window, min_periods=1).max().values
                rolling_min = data[self.profit_column].rolling(window=window, min_periods=1).min().values
            
            data[f"{self.profit_column}_rolling_mean_{window}"] = rolling_mean
            data[f"{self.profit_column}_rolling_std_{window}"] = rolling_std
            data[f"{self.profit_column}_rolling_max_{window}"] = rolling_max
            data[f"{self.profit_column}_rolling_min_{window}"] = rolling_min
            
            # Rolling range and coefficient of variation
            data[f"{self.profit_column}_rolling_range_{window}"] = rolling_max - rolling_min
            data[f"{self.profit_column}_rolling_cv_{window}"] = np.where(
                rolling_mean != 0,
                rolling_std / np.abs(rolling_mean),
                0.0
            )
            
            # Rolling percentiles
            data[f"{self.profit_column}_rolling_q25_{window}"] = data[self.profit_column].rolling(
                window=window, min_periods=1
            ).quantile(0.25)
            data[f"{self.profit_column}_rolling_q75_{window}"] = data[self.profit_column].rolling(
                window=window, min_periods=1
            ).quantile(0.75)
        
        return data

    def get_feature_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get a summary of all profit-based features.
        
        Args:
            data: DataFrame with profit-based features
            
        Returns:
            Dictionary with feature summary information
        """
        profit_features = [col for col in data.columns if self.profit_column in col and col != self.profit_column]
        
        summary = {
            "total_features": len(profit_features),
            "feature_categories": {},
            "feature_types": {},
            "missing_values": {},
            "correlation_with_target": {}
        }
        
        # Categorize features
        for feature in profit_features:
            if "basic" in feature or any(x in feature for x in ["squared", "cubed", "abs", "sqrt", "log"]):
                category = "basic_profit"
            elif any(x in feature for x in ["bins", "sign", "magnitude", "direction"]):
                category = "categorical"
            elif any(x in feature for x in ["sharpe", "kelly", "sortino", "risk"]):
                category = "risk_reward"
            elif "momentum" in feature or "acceleration" in feature:
                category = "momentum"
            elif "volatility" in feature:
                category = "volatility"
            elif "volume" in feature:
                category = "volume"
            elif "rolling" in feature:
                category = "rolling"
            else:
                category = "other"
            
            if category not in summary["feature_categories"]:
                summary["feature_categories"][category] = []
            summary["feature_categories"][category].append(feature)
            
            # Feature type
            if data[feature].dtype == 'object':
                summary["feature_types"][feature] = "categorical"
            else:
                summary["feature_types"][feature] = "numerical"
            
            # Missing values
            missing_pct = data[feature].isnull().sum() / len(data) * 100
            summary["missing_values"][feature] = missing_pct
        
        return summary

    def select_features(
        self, 
        data: pd.DataFrame, 
        method: str = "correlation",
        threshold: float = 0.01,
        max_features: Optional[int] = None
    ) -> List[str]:
        """Select the most important profit-based features.
        
        Args:
            data: DataFrame with profit-based features
            method: Feature selection method ('correlation', 'variance', 'mutual_info')
            threshold: Threshold for feature selection
            max_features: Maximum number of features to select
            
        Returns:
            List of selected feature names
        """
        profit_features = [col for col in data.columns if self.profit_column in col and col != self.profit_column]
        
        if method == "correlation":
            # Select features based on correlation with target
            correlations = data[profit_features].corrwith(data[self.profit_column]).abs()
            selected = correlations[correlations > threshold].index.tolist()
        
        elif method == "variance":
            # Select features based on variance
            variances = data[profit_features].var()
            selected = variances[variances > threshold].index.tolist()
        
        elif method == "mutual_info":
            # Select features based on mutual information (requires scikit-learn)
            try:
                from sklearn.feature_selection import mutual_info_regression
                mi_scores = mutual_info_regression(
                    data[profit_features].fillna(0), 
                    data[self.profit_column]
                )
                mi_series = pd.Series(mi_scores, index=profit_features)
                selected = mi_series[mi_series > threshold].index.tolist()
            except ImportError:
                self.logger.warning("scikit-learn not available, falling back to correlation method")
                correlations = data[profit_features].corrwith(data[self.profit_column]).abs()
                selected = correlations[correlations > threshold].index.tolist()
        
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        # Limit number of features if specified
        if max_features and len(selected) > max_features:
            selected = selected[:max_features]
        
        self.logger.info(f"Selected {len(selected)} features using {method} method")
        return selected


@with_tracing_span("benchmark_profit_feature_engineering", log_args=False)
@handle_errors(exceptions=(Exception,), default_return={}, context="benchmark_profit_features")
def benchmark_profit_feature_engineering(data: pd.DataFrame) -> Dict[str, float]:
    """Benchmark profit-based feature engineering performance.
    
    Args:
        data: Market data to test
        
    Returns:
        Dictionary with timing results
    """
    import time
    
    # Initialize feature engineering system
    feature_eng = ProfitBasedFeatureEngineering()
    
    # Benchmark with Numba
    feature_eng.use_numba = True
    start_time = time.time()
    result_numba = feature_eng.apply_all_features(data)
    numba_time = time.time() - start_time
    
    # Benchmark without Numba
    feature_eng.use_numba = False
    start_time = time.time()
    result_python = feature_eng.apply_all_features(data)
    python_time = time.time() - start_time
    
    # Verify results are similar
    numba_features = [col for col in result_numba.columns if "potential_profit_pct" in col]
    python_features = [col for col in result_python.columns if "potential_profit_pct" in col]
    
    return {
        "numba_time": numba_time,
        "python_time": python_time,
        "speedup": python_time / numba_time if numba_time > 0 else float('inf'),
        "numba_features": len(numba_features),
        "python_features": len(python_features),
        "data_shape": data.shape,
        "result_shape": result_numba.shape,
    }


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create sample data
    dates = pd.date_range("2024-01-01", periods=1000, freq="1min")
    data = pd.DataFrame({
        "open": np.random.uniform(100, 110, 1000),
        "high": np.random.uniform(105, 115, 1000),
        "low": np.random.uniform(95, 105, 1000),
        "close": np.random.uniform(100, 110, 1000),
        "volume": np.random.uniform(1000, 10000, 1000),
        "potential_profit_pct": np.random.uniform(-0.01, 0.01, 1000),
    }, index=dates)
    
    # Test feature engineering
    feature_eng = ProfitBasedFeatureEngineering()
    result = feature_eng.apply_all_features(data)
    
    # Get feature summary
    summary = feature_eng.get_feature_summary(result)
    print(f"Feature engineering results: {summary}")
    
    # Benchmark performance
    benchmark_results = benchmark_profit_feature_engineering(data)
    print(f"Benchmark results: {benchmark_results}")