# src/training/optimization/computational_optimization_manager.py

"""
Computational Optimization Manager for Enhanced Training Pipeline.
Implements all optimization strategies from computational_optimization_strategies.md
"""

import gc
import hashlib
import json
import multiprocessing as mp
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
import psutil
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    warning,
    critical,
    problem,
    failed,
    invalid,
    missing,
    timeout,
    connection_error,
    validation_error,
    initialization_error,
    execution_error,
)


@dataclass
class ComputationalOptimizationConfig:
    """Configuration for computational optimization strategies."""

    # Caching configuration
    enable_caching: bool = True
    max_cache_size: int = 1000
    cache_ttl_hours: int = 24

    # Parallel processing configuration
    enable_parallelization: bool = True
    max_workers: int = None  # Auto-detect if None
    chunk_size: int = 1000

    # Early stopping configuration
    enable_early_stopping: bool = True
    patience: int = 10
    min_trials: int = 20

    # Surrogate models configuration
    enable_surrogate_models: bool = True
    expensive_trials: int = 50
    update_frequency: int = 10

    # Memory management configuration
    enable_memory_management: bool = True
    memory_threshold: float = 0.8
    cleanup_frequency: int = 100

    # Progressive evaluation configuration
    enable_progressive_evaluation: bool = True
    evaluation_stages: list[tuple[float, float]] = None  # (data_ratio, weight)

    # Model complexity scaling
    enable_adaptive_complexity: bool = True
    complexity_levels: dict[str, dict[str, Any]] = None

    # Backtesting configuration
    enable_cached_backtesting: bool = True
    enable_progressive_evaluation_backtesting: bool = True
    enable_parallel_backtesting: bool = True
    max_backtest_workers: int = 4
    backtest_timeout_seconds: int = 300

    # Model training configuration
    enable_incremental_training: bool = True
    enable_adaptive_complexity_training: bool = True
    model_cache_size: int = 100
    warm_start_threshold: float = 0.8

    # Feature engineering configuration
    enable_precomputed_features: bool = True
    enable_feature_caching: bool = True
    feature_cache_size: int = 500
    enable_memory_efficient_data: bool = True

    # Multi-objective optimization
    enable_surrogate_models_multi: bool = True
    enable_adaptive_sampling: bool = True
    surrogate_model_type: str = "gaussian_process"
    expensive_evaluation_ratio: float = 0.2

    # Memory management
    enable_memory_monitoring: bool = True
    max_memory_usage_mb: int = 8000
    enable_garbage_collection: bool = True

    def __post_init__(self):
        """Post-initialization processing to handle nested configurations."""
        # Convert evaluation_stages from list of tuples to proper format if needed
        if self.evaluation_stages is None:
            self.evaluation_stages = [
                (0.1, 0.3),  # 10% data, 30% weight
                (0.3, 0.5),  # 30% data, 50% weight
                (1.0, 1.0),  # 100% data, 100% weight
            ]

        # Set default complexity levels if None
        if self.complexity_levels is None:
            self.complexity_levels = {
                "light": {"n_estimators": 50, "max_depth": 3},
                "medium": {"n_estimators": 100, "max_depth": 6},
                "heavy": {"n_estimators": 200, "max_depth": 10},
            }


class CachedBacktester:
    """Cached backtesting to avoid redundant calculations."""

    def __init__(
        self,
        market_data: pd.DataFrame,
        config: ComputationalOptimizationConfig,
    ):
        self.market_data = market_data
        self.config = config
        self.cache = {}
        self.technical_indicators = self._precompute_indicators()
        self.logger = system_logger.getChild("CachedBacktester")

    def _precompute_indicators(self) -> dict[str, np.ndarray]:
        """Precompute all technical indicators once."""
        self.logger.info("Precomputing technical indicators...")
        indicators = {}

        # Price-based features
        indicators["returns"] = self.market_data["close"].pct_change().values
        indicators["log_returns"] = np.log(self.market_data["close"]).diff().values

        # Moving averages (multiple periods)
        for period in [5, 10, 20, 50, 100]:
            indicators[f"sma_{period}"] = (
                self.market_data["close"].rolling(period).mean().values
            )
            indicators[f"ema_{period}"] = (
                self.market_data["close"].ewm(span=period).mean().values
            )

        # Volatility features
        indicators["atr"] = self._calculate_atr()
        indicators["volatility"] = (
            pd.Series(indicators["returns"]).rolling(20).std().values
        )

        # Momentum features
        indicators["rsi"] = self._calculate_rsi()
        indicators["macd"] = self._calculate_macd()

        self.logger.info(f"Precomputed {len(indicators)} technical indicators")
        return indicators

    def _calculate_atr(self) -> np.ndarray:
        """Calculate Average True Range."""
        high = self.market_data["high"].values
        low = self.market_data["low"].values
        close = self.market_data["close"].values

        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))

        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        return pd.Series(tr).rolling(14).mean().values

    def _calculate_rsi(self) -> np.ndarray:
        """Calculate Relative Strength Index."""
        close = self.market_data["close"]
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.values

    def _calculate_macd(self) -> np.ndarray:
        """Calculate MACD."""
        close = self.market_data["close"]
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        return macd.values

    def _generate_cache_key(self, params: dict[str, Any]) -> str:
        """Generate cache key based on parameters."""
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()

    def run_cached_backtest(self, params: dict[str, Any]) -> float:
        """Run backtest using cached indicators."""
        cache_key = self._generate_cache_key(params)

        if cache_key in self.cache:
            self.logger.debug(f"Cache hit for parameters: {cache_key[:8]}")
            return self.cache[cache_key]

        # Run simplified backtest using precomputed indicators
        result = self._run_simplified_backtest(params)
        self.cache[cache_key] = result

        # Manage cache size
        if len(self.cache) > self.config.max_cache_size:
            self._cleanup_cache()

        return result

    def _run_simplified_backtest(self, params: dict[str, Any]) -> float:
        """Run simplified backtest using precomputed indicators."""
        # This is a simplified backtest implementation
        # In practice, this would use the precomputed indicators
        # and apply the trading logic based on parameters

        # Extract parameters
        sma_short = params.get("sma_short", 20)
        sma_long = params.get("sma_long", 50)
        rsi_threshold = params.get("rsi_threshold", 30)

        # Use precomputed indicators
        sma_short_values = self.technical_indicators.get(
            f"sma_{sma_short}",
            self.technical_indicators["sma_20"],
        )
        sma_long_values = self.technical_indicators.get(
            f"sma_{sma_long}",
            self.technical_indicators["sma_50"],
        )
        rsi_values = self.technical_indicators["rsi"]

        # Simple trading logic
        signals = np.zeros(len(self.market_data))

        # Generate signals based on SMA crossover and RSI
        for i in range(1, len(signals)):
            if (
                sma_short_values[i] > sma_long_values[i]
                and sma_short_values[i - 1] <= sma_long_values[i - 1]
                and rsi_values[i] < rsi_threshold
            ):
                signals[i] = 1  # Buy signal
            elif (
                sma_short_values[i] < sma_long_values[i]
                and sma_short_values[i - 1] >= sma_long_values[i - 1]
            ):
                signals[i] = -1  # Sell signal

        # Calculate returns
        returns = self.technical_indicators["returns"]
        strategy_returns = signals * returns

        # Calculate Sharpe ratio
        return np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8)

    def _cleanup_cache(self):
        """Clean up old cache entries."""
        if len(self.cache) > self.config.max_cache_size:
            # Remove oldest entries
            oldest_keys = sorted(self.cache.keys())[: len(self.cache) // 2]
            for key in oldest_keys:
                del self.cache[key]


class ProgressiveEvaluator:
    """Progressive evaluation to stop unpromising trials early."""

    def __init__(
        self,
        full_data: pd.DataFrame,
        config: ComputationalOptimizationConfig,
    ):
        self.full_data = full_data
        self.config = config
        self.logger = system_logger.getChild("ProgressiveEvaluator")

        if config.evaluation_stages is None:
            self.evaluation_stages = [
                (0.1, 0.3),  # 10% data, 30% weight
                (0.3, 0.5),  # 30% data, 50% weight
                (1.0, 1.0),  # 100% data, 100% weight
            ]
        else:
            self.evaluation_stages = config.evaluation_stages

    def evaluate_progressively(self, params: dict[str, Any]) -> float:
        """Evaluate parameters progressively across data subsets."""
        total_score = 0
        total_weight = 0

        for data_ratio, weight in self.evaluation_stages:
            subset_size = int(len(self.full_data) * data_ratio)
            subset_data = self.full_data.iloc[:subset_size]

            score = self._evaluate_subset(subset_data, params)
            total_score += score * weight
            total_weight += weight

            # Early stopping if performance is poor
            if data_ratio < 1.0 and score < -0.5:
                self.logger.debug(
                    f"Early stopping at {data_ratio*100}% data due to poor performance",
                )
                return -1.0  # Stop evaluation

        return total_score / total_weight

    def _evaluate_subset(
        self,
        subset_data: pd.DataFrame,
        params: dict[str, Any],
    ) -> float:
        """Evaluate parameters on a data subset."""
        # Simplified evaluation - in practice this would run a backtest
        # on the subset data with the given parameters

        # Extract parameters
        sma_short = params.get("sma_short", 20)
        sma_long = params.get("sma_long", 50)

        # Calculate simple moving averages
        sma_short_values = subset_data["close"].rolling(sma_short).mean()
        sma_long_values = subset_data["close"].rolling(sma_long).mean()

        # Simple signal generation
        signals = np.zeros(len(subset_data))
        for i in range(1, len(signals)):
            if (
                sma_short_values.iloc[i] > sma_long_values.iloc[i]
                and sma_short_values.iloc[i - 1] <= sma_long_values.iloc[i - 1]
            ):
                signals[i] = 1
            elif (
                sma_short_values.iloc[i] < sma_long_values.iloc[i]
                and sma_short_values.iloc[i - 1] >= sma_long_values.iloc[i - 1]
            ):
                signals[i] = -1

        # Calculate returns
        returns = subset_data["close"].pct_change()
        strategy_returns = signals * returns

        # Calculate Sharpe ratio
        return np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8)


class ParallelBacktester:
    """Parallel backtesting for multiple parameter combinations."""

    def __init__(self, config: ComputationalOptimizationConfig):
        self.config = config
        self.logger = system_logger.getChild("ParallelBacktester")

        if config.max_workers is None:
            self.n_workers = min(mp.cpu_count(), 8)
        else:
            self.n_workers = config.max_workers

        self.executor = ProcessPoolExecutor(max_workers=self.n_workers)
        self.logger.info(
            f"Initialized parallel backtester with {self.n_workers} workers",
        )

    def evaluate_batch(
        self,
        param_batch: list[dict[str, Any]],
        market_data: pd.DataFrame,
    ) -> list[float]:
        """Evaluate multiple parameter sets in parallel."""

        # Prepare data for parallel processing
        data_pickle = pickle.dumps(market_data)

        # Submit batch for parallel evaluation
        futures = []
        for params in param_batch:
            future = self.executor.submit(
                self._evaluate_single_params,
                data_pickle,
                params,
            )
            futures.append(future)

        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result(timeout=300)  # 5 minute timeout
                results.append(result)
            except Exception as e:
                self.print(error("Error in parallel evaluation: {e}"))
                results.append(-1.0)  # Default to poor performance

        return results

    @staticmethod
    def _evaluate_single_params(data_pickle: bytes, params: dict[str, Any]) -> float:
        """Evaluate single parameter set (runs in separate process)."""
        try:
            market_data = pickle.loads(data_pickle)
            return ParallelBacktester._run_simplified_backtest(market_data, params)
        except Exception:
            return -1.0

    @staticmethod
    def _run_simplified_backtest(
        market_data: pd.DataFrame,
        params: dict[str, Any],
    ) -> float:
        """Run simplified backtest for parallel evaluation."""
        # Simplified backtest implementation
        sma_short = params.get("sma_short", 20)
        sma_long = params.get("sma_long", 50)

        sma_short_values = market_data["close"].rolling(sma_short).mean()
        sma_long_values = market_data["close"].rolling(sma_long).mean()

        signals = np.zeros(len(market_data))
        for i in range(1, len(signals)):
            if (
                sma_short_values.iloc[i] > sma_long_values.iloc[i]
                and sma_short_values.iloc[i - 1] <= sma_long_values.iloc[i - 1]
            ):
                signals[i] = 1
            elif (
                sma_short_values.iloc[i] < sma_long_values.iloc[i]
                and sma_short_values.iloc[i - 1] >= sma_long_values.iloc[i - 1]
            ):
                signals[i] = -1

        returns = market_data["close"].pct_change()
        strategy_returns = signals * returns

        return np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8)


class IncrementalTrainer:
    """Incremental training to reuse model states."""

    def __init__(
        self,
        base_model_config: dict[str, Any],
        config: ComputationalOptimizationConfig,
    ):
        self.base_config = base_model_config
        self.config = config
        self.model_cache = {}
        self.logger = system_logger.getChild("IncrementalTrainer")

    def train_incrementally(
        self,
        params: dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
    ) -> Any:
        """Train model incrementally from cached state."""

        # Generate model key based on core parameters
        model_key = self._generate_model_key(params)

        if model_key in self.model_cache:
            # Continue training from cached state
            self.logger.debug(f"Using cached model for key: {model_key[:8]}")
            model = self.model_cache[model_key]
            # For XGBoost, we can continue training
            if hasattr(model, "fit"):
                model.fit(
                    X,
                    y,
                    xgb_model=model.get_booster()
                    if hasattr(model, "get_booster")
                    else None,
                )
        else:
            # Train new model
            self.logger.debug(f"Training new model for key: {model_key[:8]}")
            model = self._create_model(params)
            model.fit(X, y)
            self.model_cache[model_key] = model

        return model

    def _generate_model_key(self, params: dict[str, Any]) -> str:
        """Generate cache key based on core model parameters."""
        core_params = {
            "max_depth": params.get("max_depth"),
            "learning_rate": params.get("learning_rate"),
            "subsample": params.get("subsample"),
            "colsample_bytree": params.get("colsample_bytree"),
        }
        return hashlib.md5(json.dumps(core_params, sort_keys=True).encode()).hexdigest()

    def _create_model(self, params: dict[str, Any]) -> Any:
        """Create a new model with given parameters."""
        # This would create the appropriate model type
        # For now, return a simple placeholder

        return RandomForestRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 6),
            random_state=42,
        )


class AdaptiveModelComplexity:
    """Adaptive model complexity based on data size and performance."""

    def __init__(self, config: ComputationalOptimizationConfig):
        self.config = config
        self.logger = system_logger.getChild("AdaptiveModelComplexity")

        if config.complexity_levels is None:
            self.complexity_levels = {
                "light": {"n_estimators": 50, "max_depth": 3},
                "medium": {"n_estimators": 100, "max_depth": 6},
                "heavy": {"n_estimators": 200, "max_depth": 10},
            }
        else:
            self.complexity_levels = config.complexity_levels

    def get_adaptive_params(
        self,
        data_size: int,
        previous_performance: float,
    ) -> dict[str, Any]:
        """Get adaptive model parameters based on context."""

        if data_size < 1000 or previous_performance < 0.3:
            self.logger.debug("Using light complexity model")
            return self.complexity_levels["light"]
        if data_size < 5000 or previous_performance < 0.6:
            self.logger.debug("Using medium complexity model")
            return self.complexity_levels["medium"]
        self.logger.debug("Using heavy complexity model")
        return self.complexity_levels["heavy"]


class PrecomputedFeatureEngine:
    """Precompute all possible features once."""

    def __init__(
        self,
        market_data: pd.DataFrame,
        config: ComputationalOptimizationConfig,
    ):
        self.market_data = market_data
        self.config = config
        self.feature_cache = {}
        self.logger = system_logger.getChild("PrecomputedFeatureEngine")
        self._precompute_all_features()

    def _precompute_all_features(self):
        """Precompute all possible technical indicators."""
        self.logger.info("Precomputing all features...")

        # Price-based features
        self.feature_cache["returns"] = self.market_data["close"].pct_change()
        self.feature_cache["log_returns"] = np.log(self.market_data["close"]).diff()

        # Moving averages (multiple periods)
        for period in [5, 10, 20, 50, 100]:
            self.feature_cache[f"sma_{period}"] = (
                self.market_data["close"].rolling(period).mean()
            )
            self.feature_cache[f"ema_{period}"] = (
                self.market_data["close"].ewm(span=period).mean()
            )

        # Volatility features
        self.feature_cache["atr"] = self._calculate_atr()
        self.feature_cache["volatility"] = (
            self.feature_cache["returns"].rolling(20).std()
        )

        # Momentum features
        self.feature_cache["rsi"] = self._calculate_rsi()
        self.feature_cache["macd"] = self._calculate_macd()

        self.logger.info(f"Precomputed {len(self.feature_cache)} features")

    def _calculate_atr(self) -> pd.Series:
        """Calculate Average True Range."""
        high = self.market_data["high"]
        low = self.market_data["low"]
        close = self.market_data["close"]

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(14).mean()

    def _calculate_rsi(self) -> pd.Series:
        """Calculate Relative Strength Index."""
        close = self.market_data["close"]
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self) -> pd.Series:
        """Calculate MACD."""
        close = self.market_data["close"]
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        return ema12 - ema26

    def get_features(self, feature_selection: list[str]) -> np.ndarray:
        """Get selected features from cache."""
        selected_features = []
        for feature_name in feature_selection:
            if feature_name in self.feature_cache:
                selected_features.append(self.feature_cache[feature_name].values)

        return np.column_stack(selected_features)


class FeatureSelectionCache:
    """Cache feature selection results."""

    def __init__(self, config: ComputationalOptimizationConfig):
        self.config = config
        self.selection_cache = {}
        self.logger = system_logger.getChild("FeatureSelectionCache")

    def get_cached_selection(
        self,
        feature_list: list[str],
        threshold: float,
    ) -> np.ndarray:
        """Get cached feature selection result."""

        cache_key = (tuple(sorted(feature_list)), threshold)

        if cache_key in self.selection_cache:
            self.logger.debug("Using cached feature selection")
            return self.selection_cache[cache_key]

        # Perform feature selection
        selected_features = self._select_features(feature_list, threshold)
        self.selection_cache[cache_key] = selected_features

        return selected_features

    def _select_features(self, feature_list: list[str], threshold: float) -> np.ndarray:
        """Perform feature selection."""
        # Simplified feature selection - in practice this would use
        # correlation analysis, mutual information, etc.
        return np.array(feature_list[: int(len(feature_list) * threshold)])


class SurrogateOptimizer:
    """Use surrogate models to reduce expensive evaluations."""

    def __init__(self, config: ComputationalOptimizationConfig):
        self.config = config
        self.logger = system_logger.getChild("SurrogateOptimizer")
        self.n_expensive_trials = config.expensive_trials
        self.surrogate_model = None
        self.expensive_evaluations = []
        self.update_frequency = config.update_frequency

    def optimize_with_surrogates(self, objective_func, n_trials: int) -> dict[str, Any]:
        """Optimize using surrogate models for expensive evaluations."""

        self.logger.info(f"Starting surrogate optimization with {n_trials} trials")

        # Initial expensive evaluations
        for i in range(self.n_expensive_trials):
            params = self._suggest_parameters()
            result = objective_func(params)  # Expensive evaluation
            self.expensive_evaluations.append((params, result))

        # Train surrogate model
        self._train_surrogate_model()

        # Use surrogate for remaining trials
        for i in range(self.n_expensive_trials, n_trials):
            params = self._suggest_parameters()
            predicted_result = self._predict_with_surrogate(params)

            # Only do expensive evaluation occasionally
            if i % self.update_frequency == 0:
                actual_result = objective_func(params)
                self._update_surrogate_model(params, actual_result)
            else:
                # Use surrogate prediction
                result = predicted_result

        return self._get_best_results()

    def _suggest_parameters(self) -> dict[str, Any]:
        """Suggest parameters for evaluation."""
        # This would use Optuna or similar for parameter suggestion
        return {
            "sma_short": np.random.randint(5, 50),
            "sma_long": np.random.randint(20, 200),
            "rsi_threshold": np.random.uniform(20, 80),
        }

    def _train_surrogate_model(self):
        """Train surrogate model on expensive evaluations."""
        if len(self.expensive_evaluations) < 10:
            return

        X = []
        y = []
        for params, result in self.expensive_evaluations:
            X.append(list(params.values()))
            y.append(result)

        X = np.array(X)
        y = np.array(y)

        # Use Gaussian Process as surrogate
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        self.surrogate_model = GaussianProcessRegressor(kernel=kernel, random_state=42)
        self.surrogate_model.fit(X, y)

        self.logger.info("Trained surrogate model")

    def _predict_with_surrogate(self, params: dict[str, Any]) -> float:
        """Predict result using surrogate model."""
        if self.surrogate_model is None:
            return 0.0

        X = np.array([list(params.values())]).reshape(1, -1)
        prediction, _ = self.surrogate_model.predict(X, return_std=True)
        return prediction[0]

    def _update_surrogate_model(self, params: dict[str, Any], result: float):
        """Update surrogate model with new evaluation."""
        self.expensive_evaluations.append((params, result))
        self._train_surrogate_model()

    def _get_best_results(self) -> dict[str, Any]:
        """Get best results from expensive evaluations."""
        if not self.expensive_evaluations:
            return {}

        best_eval = max(self.expensive_evaluations, key=lambda x: x[1])
        return {
            "best_params": best_eval[0],
            "best_score": best_eval[1],
            "total_evaluations": len(self.expensive_evaluations),
        }


class AdaptiveSampler:
    """Adaptive sampling to focus on promising regions."""

    def __init__(self, config: ComputationalOptimizationConfig):
        self.config = config
        self.logger = system_logger.getChild("AdaptiveSampler")
        self.initial_samples = 100
        self.promising_regions = []

    def suggest_parameters(self, trial_history: list[dict]) -> dict[str, Any]:
        """Suggest parameters based on promising regions."""

        if len(trial_history) < self.initial_samples:
            # Random sampling for initial exploration
            return self._random_sampling()
        # Focus on promising regions
        return self._adaptive_sampling(trial_history)

    def _random_sampling(self) -> dict[str, Any]:
        """Random parameter sampling."""
        return {
            "sma_short": np.random.randint(5, 50),
            "sma_long": np.random.randint(20, 200),
            "rsi_threshold": np.random.uniform(20, 80),
        }

    def _adaptive_sampling(self, trial_history: list[dict]) -> dict[str, Any]:
        """Sample from promising regions identified in history."""

        # Identify promising regions
        good_trials = [t for t in trial_history if t.get("score", 0) > 0.5]

        if not good_trials:
            return self._random_sampling()

        # Sample around good trials
        reference_trial = np.random.choice(good_trials)
        return self._perturb_parameters(reference_trial.get("params", {}))

    def _perturb_parameters(self, base_params: dict[str, Any]) -> dict[str, Any]:
        """Perturb parameters around base values."""
        perturbed = {}
        for key, value in base_params.items():
            if isinstance(value, int):
                perturbed[key] = max(1, value + np.random.randint(-5, 6))
            elif isinstance(value, float):
                perturbed[key] = value + np.random.uniform(-0.1, 0.1)
            else:
                perturbed[key] = value

        return perturbed


class MemoryEfficientData:
    """Memory-efficient data structures for large datasets."""

    def __init__(
        self,
        market_data: pd.DataFrame,
        config: ComputationalOptimizationConfig,
    ):
        self.config = config
        self.logger = system_logger.getChild("MemoryEfficientData")
        self.data = self._optimize_dataframe(market_data)

    def _optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for memory usage."""

        # Use appropriate dtypes
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="float")

        for col in df.select_dtypes(include=["int64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")

        self.logger.info("Optimized DataFrame memory usage")
        return df

    def get_subset(self, start_idx: int, end_idx: int) -> np.ndarray:
        """Get numpy array subset for efficient computation."""
        return self.data.iloc[start_idx:end_idx].values


class MemoryManager:
    """Manage memory usage during optimization."""

    def __init__(self, config: ComputationalOptimizationConfig):
        self.config = config
        self.logger = system_logger.getChild("MemoryManager")
        self.memory_threshold = config.memory_threshold
        self.cleanup_frequency = config.cleanup_frequency
        self.evaluation_count = 0

    def check_memory_usage(self):
        """Check and manage memory usage."""
        self.evaluation_count += 1

        if self.evaluation_count % self.cleanup_frequency == 0:
            memory_percent = psutil.virtual_memory().percent / 100

            if memory_percent > self.memory_threshold:
                self.logger.warning(
                    f"High memory usage ({memory_percent:.1%}), cleaning up...",
                )
                self._cleanup_memory()

    def _cleanup_memory(self):
        """Clean up memory by forcing garbage collection."""
        gc.collect()
        self.logger.info("Memory cleanup completed")


class ComputationalOptimizationManager:
    """
    Main computational optimization manager that integrates all strategies.
    """

    def __init__(self, config: ComputationalOptimizationConfig):
        self.config = config
        self.logger = system_logger.getChild("ComputationalOptimizationManager")

        # Initialize optimization components
        self.cached_backtester = None
        self.progressive_evaluator = None
        self.parallel_backtester = None
        self.incremental_trainer = None
        self.adaptive_complexity = None
        self.precomputed_features = None
        self.feature_cache = None
        self.surrogate_optimizer = None
        self.adaptive_sampler = None
        self.memory_efficient_data = None
        self.memory_manager = None

        self.logger.info("Computational Optimization Manager initialized")

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="computational optimization manager initialization",
    )
    async def initialize(
        self,
        market_data: pd.DataFrame,
        model_config: dict[str, Any],
    ) -> bool:
        """Initialize all optimization components."""
        try:
            self.logger.info("Initializing computational optimization components...")

            # Initialize memory manager first
            self.memory_manager = MemoryManager(self.config)

            # Initialize data components
            self.memory_efficient_data = MemoryEfficientData(market_data, self.config)
            self.precomputed_features = PrecomputedFeatureEngine(
                market_data,
                self.config,
            )
            self.feature_cache = FeatureSelectionCache(self.config)

            # Initialize backtesting components
            self.cached_backtester = CachedBacktester(market_data, self.config)
            self.progressive_evaluator = ProgressiveEvaluator(market_data, self.config)
            self.parallel_backtester = ParallelBacktester(self.config)

            # Initialize training components
            self.incremental_trainer = IncrementalTrainer(model_config, self.config)
            self.adaptive_complexity = AdaptiveModelComplexity(self.config)

            # Initialize optimization components
            self.surrogate_optimizer = SurrogateOptimizer(self.config)
            self.adaptive_sampler = AdaptiveSampler(self.config)

            self.logger.info(
                "All computational optimization components initialized successfully",
            )
            return True

        except Exception as e:
            self.logger.exception(
                f"Failed to initialize computational optimization manager: {e}",
            )
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="optimized parameter optimization",
    )
    async def optimize_parameters(
        self,
        objective_function: callable,
        n_trials: int = 100,
        use_surrogates: bool = True,
    ) -> dict[str, Any]:
        """Run optimized parameter optimization."""
        try:
            self.logger.info(
                f"Starting optimized parameter optimization with {n_trials} trials",
            )

            if use_surrogates and self.config.enable_surrogate_models:
                return self.surrogate_optimizer.optimize_with_surrogates(
                    objective_function,
                    n_trials,
                )
            return await self._run_standard_optimization(
                objective_function,
                n_trials,
            )

        except Exception as e:
            self.print(failed("Parameter optimization failed: {e}"))
            return {}

    async def _run_standard_optimization(
        self,
        objective_function: callable,
        n_trials: int,
    ) -> dict[str, Any]:
        """Run standard optimization with caching and early stopping."""
        study = optuna.create_study(direction="maximize")

        def objective(trial):
            # Check memory usage
            self.memory_manager.check_memory_usage()

            # Suggest parameters using adaptive sampling
            params = self.adaptive_sampler.suggest_parameters(study.trials)

            # Use cached backtesting if available
            if self.cached_backtester:
                return self.cached_backtester.run_cached_backtest(params)
            return objective_function(params)

        study.optimize(objective, n_trials=n_trials)

        return {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "total_trials": len(study.trials),
        }

    def get_optimization_statistics(self) -> dict[str, Any]:
        """Get statistics from all optimization components."""
        return {
            "cache_hits": len(self.cached_backtester.cache)
            if self.cached_backtester
            else 0,
            "memory_usage": psutil.virtual_memory().percent,
            "surrogate_evaluations": len(self.surrogate_optimizer.expensive_evaluations)
            if self.surrogate_optimizer
            else 0,
        }

    async def cleanup(self):
        """Cleanup resources."""
        try:
            if self.parallel_backtester:
                self.parallel_backtester.executor.shutdown()

            if self.memory_manager:
                self.memory_manager._cleanup_memory()

            self.logger.info("Computational optimization manager cleanup completed")

        except Exception as e:
            self.print(failed("Cleanup failed: {e}"))


# Factory function for easy integration
async def create_computational_optimization_manager(
    config: dict[str, Any],
    market_data: pd.DataFrame,
    model_config: dict[str, Any],
) -> ComputationalOptimizationManager:
    """Create and initialize a computational optimization manager."""

    # Extract the computational_optimization config and flatten nested structures
    optimization_config_raw = config.get("computational_optimization", {})

    # Get the valid field names for ComputationalOptimizationConfig
    from dataclasses import fields

    valid_fields = {field.name for field in fields(ComputationalOptimizationConfig)}

    # Flatten the nested configuration structure
    flattened_config = {}

    # Copy top-level parameters that match valid fields
    for key, value in optimization_config_raw.items():
        if key in valid_fields:
            flattened_config[key] = value

    # Extract nested configurations and flatten them
    if "backtesting" in optimization_config_raw:
        backtesting_config = optimization_config_raw["backtesting"]
        for key, value in backtesting_config.items():
            field_name = f"enable_{key}" if key.startswith("enable_") else key
            if field_name in valid_fields:
                flattened_config[field_name] = value

    if "model_training" in optimization_config_raw:
        training_config = optimization_config_raw["model_training"]
        for key, value in training_config.items():
            field_name = f"enable_{key}" if key.startswith("enable_") else key
            if field_name in valid_fields:
                flattened_config[field_name] = value

    if "feature_engineering" in optimization_config_raw:
        feature_config = optimization_config_raw["feature_engineering"]
        for key, value in feature_config.items():
            field_name = f"enable_{key}" if key.startswith("enable_") else key
            if field_name in valid_fields:
                flattened_config[field_name] = value

    if "multi_objective" in optimization_config_raw:
        multi_config = optimization_config_raw["multi_objective"]
        for key, value in multi_config.items():
            field_name = f"enable_{key}_multi" if key.startswith("enable_") else key
            if field_name in valid_fields:
                flattened_config[field_name] = value

    if "memory_management" in optimization_config_raw:
        memory_config = optimization_config_raw["memory_management"]
        for key, value in memory_config.items():
            if key in valid_fields:
                flattened_config[key] = value

    # Create the configuration object
    optimization_config = ComputationalOptimizationConfig(**flattened_config)
    return ComputationalOptimizationManager(optimization_config)

    # Defer initialization until real market data is available
