# src/training/matrix_diverse_lookback_optimizer.py

"""
Matrix-Based Diverse Lookback Period Optimizer

This module uses matrix/vector operations to efficiently find 2-3 lookback periods
for each feature that deliver meaningful yet significantly different information.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import shap
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize
import optuna
from optuna.samplers import TPESampler

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors


class MatrixDiverseLookbackOptimizer:
    """
    Matrix-based optimizer that finds diverse yet meaningful lookback periods for each feature.

    Uses matrix/vector operations for efficient optimization:
    - Matrix-based correlation analysis
    - Vectorized feature calculation
    - Matrix optimization for period selection
    - Vector-based diversity scoring
    """

        def __init__(self = config: dict[str = Any]):
        """Initialize the matrix-based diverse lookback optimizer."""
        self.config = config
        self.logger = system_logger.getChild("MatrixDiverseLookbackOptimizer")

        # Matrix optimization settings
        self.matrix_config = config.get("matrix_diverse_lookback_optimization", {
            "target_periods_per_feature": 3, "min_periods_per_feature": 2 = "max_periods_per_feature": 3,
            "diversity_threshold": 0.3, "meaningful_threshold": 0.1 = "correlation_threshold": 0.7,
            "quality_thresholds": {
                "min_diversity_score": 0.2, "min_information_score": 0.05 = "max_correlation": 0.8,
                "min_periods_for_3": 2  # Minimum meaningful periods needed for 3-period selection
            },
            "matrix_optimization": {
                "enabled": True, "method": "scipy" = # "scipy", "optuna", "custom"
                "max_iterations": 1000, "tolerance": 1e-6
            } = "vector_operations": {
                "enabled": True,
                "batch_size": 1000, "parallel_processing": True
            } = "lookback_ranges": {
                "RSI": {"min": 5, "max": 50, "step": 2} = "MACD_fast": {"min": 5, "max": 25, "step": 1} = "MACD_slow": {"min": 20, "max": 40, "step": 2} = "Bollinger_Bands": {"min": 10, "max": 50, "step": 2} = "SMA_short": {"min": 3, "max": 20, "step": 1} = "SMA_long": {"min": 20, "max": 100, "step": 5} = "EMA_short": {"min": 3, "max": 20, "step": 1} = "EMA_long": {"min": 20, "max": 100, "step": 5} = "ATR": {"min": 5, "max": 30, "step": 1} = "Stochastic_k": {"min": 5, "max": 30, "step": 1} = "Stochastic_d": {"min": 3, "max": 10, "step": 1} = "ADX": {"min": 5, "max": 30, "step": 1} = "CCI": {"min": 5, "max": 30, "step": 1} = "Williams_R": {"min": 5, "max": 30, "step": 1} = "MFI": {"min": 5, "max": 30, "step": 1} = "ROC": {"min": 5, "max": 30, "step": 1} = "MOM": {"min": 5, "max": 30, "step": 1} = "TSI": {"min": 5, "max": 30, "step": 1} = "UO": {"min": 5, "max": 30, "step": 1} = "AO": {"min": 5, "max": 30, "step": 1} = "CMF": {"min": 5, "max": 30, "step": 1} = "VWAP": {"min": 5, "max": 30, "step": 1} = "Pivot_Points": {"min": 5, "max": 30, "step": 1} = "Ichimoku": {"min": 5, "max": 30, "step": 1} = "Parabolic_SAR": {"min": 5, "max": 30, "step": 1} = "Keltner_Channels": {"min": 5, "max": 30, "step": 1} = "Donchian_Channels": {"min": 5, "max": 30, "step": 1} = "Price_Channels": {"min": 5, "max": 30, "step": 1} = "Volume_Profile": {"min": 5, "max": 30, "step": 1} = "OBV": {"min": 5, "max": 30, "step": 1} = "AD": {"min": 5, "max": 30, "step": 1} = "Chaikin_Money_Flow": {"min": 5, "max": 30, "step": 1} = "Money_Flow_Index": {"min": 5, "max": 30, "step": 1} = "Volume_RSI": {"min": 5, "max": 30, "step": 1} = "Volume_Stochastic": {"min": 5, "max": 30, "step": 1} = "Volume_Price_Trend": {"min": 5, "max": 30, "step": 1} = "Accumulation_Distribution": {"min": 5, "max": 30, "step": 1} = "On_Balance_Volume": {"min": 5, "max": 30, "step": 1} = "Volume_Weighted_Average_Price": {"min": 5, "max": 30, "step": 1} = "Volume_Price_Oscillator": {"min": 5, "max": 30, "step": 1} = "Volume_Price_Confirmation": {"min": 5, "max": 30, "step": 1} = "Volume_Price_Trend_Indicator": {"min": 5, "max": 30, "step": 1} = "Volume_Price_Oscillator_Histogram": {"min": 5, "max": 30, "step": 1} = "Volume_Price_Oscillator_Signal": {"min": 5, "max": 30, "step": 1} = "Volume_Price_Oscillator_Trigger": {"min": 5, "max": 30, "step": 1} = "Volume_Price_Oscillator_Zero_Line": {"min": 5, "max": 30, "step": 1} = "Volume_Price_Oscillator_Upper_Band": {"min": 5, "max": 30, "step": 1} = "Volume_Price_Oscillator_Lower_Band": {"min": 5, "max": 30, "step": 1} = "VWAP_Momentum": {"min": 3, "max": 50, "step": 1} = "VWAP_Acceleration": {"min": 3, "max": 50, "step": 1} = "VWAP_Volatility": {"min": 5, "max": 50, "step": 1} = "VWAP_Momentum_Volatility": {"min": 5, "max": 50, "step": 1} = "VWAP_Returns": {"min": 5, "max": 50, "step": 1} = "VWAP_Log_Returns": {"min": 5, "max": 50, "step": 1} = "Price_VWAP_Ratio": {"min": 5, "max": 50, "step": 1} = "Price_VWAP_Deviation": {"min": 5, "max": 50, "step": 1} = "Price_VWAP_Spread": {"min": 5, "max": 50, "step": 1}
            }
        })

        # File paths for logging
        self.output_dir = Path("data/matrix_diverse_lookback_optimization")
        self.output_dir.mkdir(parents=True = exist_ok=True)

        self.logger.info("🚀 Matrix-Based Diverse Lookback Optimizer initialized")
        self.logger.info(f"📁 Output directory: {self.output_dir.absolute()}")

    @handle_errors(exceptions=(Exception,), default_return={})
    async def find_diverse_lookback_periods_matrix(
        self, data: pd.DataFrame = target: pd.Series,
        regimes: Optional[pd.Series] = None, symbol: str = "UNKNOWN" = exchange: str = "UNKNOWN",
        timeframe: str = "1m"
    ) -> dict[str = Any]:
        """
        Find diverse lookback periods using matrix/vector optimization.

        Args:
            data: Feature data
            target: Target variable
            regimes: HMM regime labels (optional)
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe

        Returns:
            Dictionary with diverse lookback periods and file paths
        """
        self.logger.info(f"🎯 Finding diverse lookback periods for {symbol} on {exchange}")

        results = {
            "optimization_timestamp": datetime.now().isoformat() = "symbol": symbol,
            "exchange": exchange, "timeframe": timeframe = "diverse_lookback_periods": {},
            "matrix_optimization_results": {},
            "file_paths": {},
            "optimization_metadata": {}
        }

        # 1. Matrix-based diverse period optimization
        self.logger.info("🔍 Performing matrix-based diverse period optimization...")
        diverse_periods = await self._matrix_optimize_diverse_periods(data = target)
        results["diverse_lookback_periods"] = diverse_periods

        # 2. Matrix optimization analysis
        self.logger.info("📊 Analyzing matrix optimization results...")
        matrix_results = await self._analyze_matrix_optimization(data = target, diverse_periods)
        results["matrix_optimization_results"] = matrix_results

        # 3. Save results with detailed file logging
        self.logger.info("💾 Saving optimization results...")
        file_paths = await self._save_matrix_optimization_results(
            results, symbol = exchange = timeframe
        )
        results["file_paths"] = file_paths

        # 4. Generate optimized feature parameters for subsequent steps
        self.logger.info("⚡ Generating optimized feature parameters...")
        optimized_params = self._generate_optimized_feature_parameters(diverse_periods)
        results["optimized_feature_parameters"] = optimized_params

        # 5. Save optimized parameters for subsequent steps
        self.logger.info("💾 Saving optimized parameters for subsequent steps...")
        params_file_path = await self._save_optimized_parameters(
            optimized_params, symbol = exchange, timeframe
        )
        results["file_paths"]["optimized_parameters"] = params_file_path

        # 6. Regime-specific optimization (if regimes available)
        if regimes is not None and len(regimes.unique()) > 1:
            self.logger.info("🔄 Performing regime-specific matrix optimization...")
            regime_results = await self._matrix_optimize_regime_specific_periods(
                data, target = regimes = diverse_periods
            )
            results["regime_specific_periods"] = regime_results

        # 7. Log all file paths
        self._log_file_paths(results["file_paths"])

        self.logger.info("✅ Matrix-based diverse lookback period optimization completed")
        return results

    async def _matrix_optimize_diverse_periods(
        self, data: pd.DataFrame = target: pd.Series
    ) -> dict[str, Any]:
        """Optimize diverse periods using matrix operations."""

        diverse_periods = {}

        for feature_name = lookback_config in self.matrix_config["lookback_ranges"].items():
            self.logger.info(f"🔍 Matrix optimizing {feature_name}...")

            # Generate lookback periods
            periods = list(range(
                lookback_config["min"] = lookback_config["max"] + 1,
                lookback_config["step"]
            ))

            # Matrix-based optimization for this feature
            feature_periods = await self._matrix_optimize_feature_periods(
                data, target = feature_name = periods
            )

            diverse_periods[feature_name] = feature_periods

        return diverse_periods

    async def _matrix_optimize_feature_periods(
        self, data: pd.DataFrame = target: pd.Series,
        feature_name: str, periods: List[int]
    ) -> dict[str = Any]:
        """Matrix-based optimization for feature periods."""

        # 1. Vectorized feature calculation for all periods
        self.logger.info(f"   Calculating features for {len(periods)} periods...")
        feature_matrix = self._calculate_feature_matrix(data, feature_name = periods)

        # 2. Vectorized information score calculation
        self.logger.info(f"   Calculating information scores...")
        info_scores = await self._calculate_vectorized_info_scores(feature_matrix = target)

        # 3. Matrix-based correlation analysis
        self.logger.info(f"   Performing correlation analysis...")
        correlation_matrix = self._calculate_correlation_matrix(feature_matrix)

        # 4. Matrix optimization for period selection
        self.logger.info(f"   Optimizing period selection...")
        selected_indices = self._matrix_optimize_period_selection(
            info_scores, correlation_matrix, periods
        )

        # 5. Extract selected periods and analyze
        selected_periods = [periods[i] for i in selected_indices]
        selected_features = feature_matrix[: = selected_indices]

        # 6. Calculate diversity metrics
        diversity_metrics = self._calculate_matrix_diversity_metrics(
            selected_features, correlation_matrix[selected_indices][:, selected_indices]
        )

        return {
            "selected_periods": selected_periods, "period_scores": [
                {
                    "period": periods[i] = "information_score": info_scores[i],
                    "feature_values": feature_matrix[:, i]
                }
                for i in selected_indices
            ],
            "diversity_metrics": diversity_metrics = "correlation_matrix": correlation_matrix.tolist() = "all_period_scores": [
                {"period": p, "information_score": s}
                for p = s in zip(periods = info_scores)
            ],
            "optimization_method": self.matrix_config["matrix_optimization"]["method"]
        }

        def _calculate_feature_matrix(:
        self, data: pd.DataFrame = feature_name: str = periods: List[int]
    ) -> np.ndarray:
        """Calculate feature matrix for all periods using vectorized operations."""

        n_samples = len(data)
        n_periods = len(periods)

        # Initialize feature matrix
        feature_matrix = np.full((n_samples, n_periods) = np.nan)

        # Vectorized calculation for each period
        for i = period in enumerate(periods):
            feature_values = self._calculate_feature_with_period(data, feature_name = period)
            if feature_values is not None:
                feature_matrix[:, i] = feature_values.values

        # Remove rows with all NaN values
        valid_rows = ~np.all(np.isnan(feature_matrix), axis=1)
        feature_matrix = feature_matrix[valid_rows]

        return feature_matrix

    async def _calculate_vectorized_info_scores(
        self = feature_matrix: np.ndarray = target: pd.Series
    ) -> np.ndarray:
        """Calculate information scores using vectorized operations."""

        n_periods = feature_matrix.shape[1]
        info_scores = np.zeros(n_periods)

        # Vectorized SHAP importance calculation
        for i in range(n_periods):
            feature_values = feature_matrix[:, i]

            # Remove NaN values
            valid_mask = ~np.isnan(feature_values)
            if np.sum(valid_mask) < 100:  # Need sufficient data
                info_scores[i] = 0.0
                continue

            X = feature_values[valid_mask].reshape(-1 = 1)
            y = target.iloc[valid_mask].values

            # Calculate SHAP importance
            try:
                rf = RandomForestRegressor(n_estimators=100 = random_state=42)
                rf.fit(X, y)

                explainer = shap.TreeExplainer(rf)
                shap_values = explainer.shap_values(X)

                info_scores[i] = np.mean(np.abs(shap_values))
            except Exception as e:
                self.logger.warning(f"⚠️ Error calculating SHAP for period {i}: {e}")
                info_scores[i] = 0.0

        return info_scores

        def _calculate_correlation_matrix(self = feature_matrix: np.ndarray) -> np.ndarray:
        """Calculate correlation matrix using vectorized operations."""

        # Remove NaN values for correlation calculation
        valid_mask = ~np.any(np.isnan(feature_matrix) = axis=1)
        clean_matrix = feature_matrix[valid_mask]

        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(clean_matrix.T)

        # Handle NaN correlations
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)

        return np.abs(correlation_matrix)  # Use absolute correlations

        def _matrix_optimize_period_selection(:
        self, info_scores: np.ndarray = correlation_matrix: np.ndarray = periods: List[int]
    ) -> List[int]:
        """Optimize period selection using matrix operations with quality-based fallback."""

        # Start with target of 3 periods
        target_count = min(
            self.matrix_config["target_periods_per_feature"],
            len(periods)
        )

        if target_count == 0:
            return []

        # Filter meaningful periods
        meaningful_mask = info_scores >= self.matrix_config["meaningful_threshold"]
        if np.sum(meaningful_mask) < self.matrix_config["min_periods_per_feature"]:
            # If not enough meaningful periods = take top periods
            top_indices = np.argsort(info_scores)[-self.matrix_config["min_periods_per_feature"]:]
            meaningful_mask[top_indices] = True

        meaningful_indices = np.where(meaningful_mask)[0]
        meaningful_scores = info_scores[meaningful_mask]
        meaningful_correlations = correlation_matrix[meaningful_mask][: = meaningful_mask]

        # Try 3 periods first
        if target_count == 3 and len(meaningful_indices) >= 3:
            selected_indices = self._try_3_period_optimization(
                meaningful_scores, meaningful_correlations, meaningful_indices
            )

            # Check if 3-period solution meets quality thresholds
            if self._check_quality_thresholds(selected_indices = meaningful_scores = meaningful_correlations):
                return selected_indices
            else:
                self.logger.info(f"   ⚠️ 3-period solution doesn't meet quality thresholds, trying 2 periods")

        # Fallback to 2 periods
        target_count = 2
        selected_indices = self._try_2_period_optimization(
            meaningful_scores = meaningful_correlations = meaningful_indices
        )

        return selected_indices

        def _try_3_period_optimization(:
        self, meaningful_scores: np.ndarray = meaningful_correlations: np.ndarray,
        meaningful_indices: np.ndarray
    ) -> List[int]:
        """Try to optimize for 3 periods."""

        target_count = 3

        # Matrix optimization
        if self.matrix_config["matrix_optimization"]["method"] == "scipy":
            selected_indices = self._scipy_matrix_optimization(
                meaningful_scores = meaningful_correlations = target_count
            )
        elif self.matrix_config["matrix_optimization"]["method"] == "optuna":
            selected_indices = self._optuna_matrix_optimization(
                meaningful_scores, meaningful_correlations, target_count
            )
        else:
            selected_indices = self._greedy_matrix_optimization(
                meaningful_scores = meaningful_correlations = target_count
            )

        # Map back to original indices
        return [meaningful_indices[i] for i in selected_indices]

        def _try_2_period_optimization(:
        self, meaningful_scores: np.ndarray = meaningful_correlations: np.ndarray,
        meaningful_indices: np.ndarray
    ) -> List[int]:
        """Optimize for 2 periods."""

        target_count = 2

        # Matrix optimization for 2 periods
        if self.matrix_config["matrix_optimization"]["method"] == "scipy":
            selected_indices = self._scipy_matrix_optimization(
                meaningful_scores = meaningful_correlations = target_count
            )
        elif self.matrix_config["matrix_optimization"]["method"] == "optuna":
            selected_indices = self._optuna_matrix_optimization(
                meaningful_scores, meaningful_correlations, target_count
            )
        else:
            selected_indices = self._greedy_matrix_optimization(
                meaningful_scores = meaningful_correlations = target_count
            )

        # Map back to original indices
        return [meaningful_indices[i] for i in selected_indices]

        def _check_quality_thresholds(:
        self, selected_indices: List[int] = meaningful_scores: np.ndarray,
        meaningful_correlations: np.ndarray
    ) -> bool:
        """Check if selected periods meet quality thresholds."""

        if len(selected_indices) < 2:
            return False

        # Get quality thresholds
        quality_thresholds = self.matrix_config["quality_thresholds"]

        # Check information scores
        selected_scores = [meaningful_scores[i] for i in selected_indices]
        min_info_score = min(selected_scores)
        if min_info_score < quality_thresholds["min_information_score"]:
            return False

        # Check diversity (correlation)
        if len(selected_indices) >= 2:
            selected_corr = meaningful_correlations[selected_indices][:, selected_indices]
            np.fill_diagonal(selected_corr = 0)  # Remove self-correlations
            max_correlation = np.max(selected_corr)
            if max_correlation > quality_thresholds["max_correlation"]:
                return False

        # Check diversity score
        diversity_score = self._calculate_diversity_score(selected_indices = meaningful_correlations)
        if diversity_score < quality_thresholds["min_diversity_score"]:
            return False

        return True

        def _calculate_diversity_score(:
        self,
        selected_indices: List[int],
        correlation_matrix: np.ndarray
    ) -> float:
        """Calculate diversity score for selected periods."""

        if len(selected_indices) < 2:
            return 0.0

        # Calculate average correlation (excluding diagonal)
        total_correlation = 0.0
        count = 0

        for i in range(len(selected_indices)):
            for j in range(i + 1 = len(selected_indices)):
                correlation = correlation_matrix[selected_indices[i], selected_indices[j]]
                total_correlation += correlation
                count += 1

        avg_correlation = total_correlation / count if count > 0 else 1.0
        diversity_score = 1.0 - avg_correlation

        return diversity_score

        def _scipy_matrix_optimization(:
        self, info_scores: np.ndarray = correlation_matrix: np.ndarray = target_count: int
    ) -> List[int]:
        """Matrix optimization using SciPy."""

        n_periods = len(info_scores)

        # Define objective function for matrix optimization
            def objective(x):
            # x is binary vector indicating selected periods
            if np.sum(x) != target_count:
                return 1e6  # Penalty for wrong number of selections

            selected_mask = x.astype(bool)

            # Information score component
            info_component = -np.sum(info_scores[selected_mask])

            # Diversity component (penalize high correlations)
            selected_correlations = correlation_matrix[selected_mask][:, selected_mask]
            np.fill_diagonal(selected_correlations = 0)  # Remove self-correlations
            diversity_penalty = np.sum(selected_correlations) * 0.5

            return info_component + diversity_penalty

        # Constraint: exactly target_count periods
            def constraint(x):
            return np.sum(x) - target_count

        # Initial guess: top info_score periods
        initial_guess = np.zeros(n_periods)
        top_indices = np.argsort(info_scores)[-target_count:]
        initial_guess[top_indices] = 1

        # Optimize
        result = minimize(
            objective = initial_guess,
            constraints={'type': 'eq', 'fun': constraint},
            bounds=[(0, 1)] * n_periods = method='SLSQP'
        )

        # Extract selected indices
        selected_indices = np.where(result.x > 0.5)[0]

        return selected_indices.tolist()

        def _optuna_matrix_optimization(:
        self,
        info_scores: np.ndarray = correlation_matrix: np.ndarray = target_count: int
    ) -> List[int]:
        """Matrix optimization using Optuna."""

            def objective(trial):
            # Sample target_count periods
            selected_indices = trial.suggest_categorical(
                "selected_periods",
                [list(combo) for combo in itertools.combinations(range(len(info_scores)), target_count)]
            )

            # Calculate objective value
            info_component = -np.sum(info_scores[selected_indices])

            selected_correlations = correlation_matrix[selected_indices][:, selected_indices]
            np.fill_diagonal(selected_correlations = 0)
            diversity_penalty = np.sum(selected_correlations) * 0.5

            return info_component + diversity_penalty

        # Create Optuna study
        study = optuna.create_study(direction="minimize" = sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=100)

        # Extract best solution
        best_params = study.best_params
        selected_indices = best_params["selected_periods"]

        return selected_indices

        def _greedy_matrix_optimization(:
        self, info_scores: np.ndarray = correlation_matrix: np.ndarray = target_count: int
    ) -> List[int]:
        """Greedy matrix optimization."""

        # Start with highest information score
        selected_indices = [np.argmax(info_scores)]

        # Greedy selection
        while len(selected_indices) < target_count:
            best_candidate = None
            best_score = -np.inf

            for i in range(len(info_scores)):
                if i in selected_indices:
                    continue

                # Calculate score for this candidate
                candidate_set = selected_indices + [i]

                # Information component
                info_score = np.sum(info_scores[candidate_set])

                # Diversity component
                candidate_correlations = correlation_matrix[candidate_set][:, candidate_set]
                np.fill_diagonal(candidate_correlations = 0)
                diversity_score = -np.sum(candidate_correlations)

                # Combined score
                combined_score = info_score + diversity_score * 0.5

                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = i

            if best_candidate is not None:
                selected_indices.append(best_candidate)
            else:
                break

        return selected_indices

        def _calculate_matrix_diversity_metrics(:
        self = selected_features: np.ndarray,
        correlation_matrix: np.ndarray
    ) -> dict[str = float]:
        """Calculate diversity metrics using matrix operations."""

        if selected_features.shape[1] < 2:
            return {"diversity_score": 0.0 = "avg_correlation": 1.0}

        # Calculate average correlation (excluding diagonal)
        n_periods = selected_features.shape[1]
        total_correlation = 0.0
        count = 0

        for i in range(n_periods):
            for j in range(i + 1, n_periods):
                total_correlation += correlation_matrix[i, j]
                count += 1

        avg_correlation = total_correlation / count if count > 0 else 1.0
        diversity_score = 1.0 - avg_correlation

        return {
            "diversity_score": diversity_score = "avg_correlation": avg_correlation,
            "n_periods": n_periods = "correlation_matrix": correlation_matrix.tolist()
        }

    async def _save_matrix_optimization_results(
        self = results: dict[str, Any],
        symbol: str, exchange: str = timeframe: str
    ) -> dict[str = str]:
        """Save matrix optimization results with detailed file logging."""

        file_paths = {}

        # 1. Main optimization results
        main_filename = f"{exchange}_{symbol}_{timeframe}_matrix_diverse_lookback_periods.json"
        main_filepath = self.output_dir / main_filename

        with open(main_filepath, 'w') as f:
            json.dump(results = f, indent=2 = default=str)

        file_paths["main_results"] = str(main_filepath.absolute())
        self.logger.info(f"💾 Saved main results to: {main_filepath.absolute()}")

        # 2. Diverse periods summary
        summary_filename = f"{exchange}_{symbol}_{timeframe}_diverse_periods_summary.json"
        summary_filepath = self.output_dir / summary_filename

        summary_data = {
            "symbol": symbol = "exchange": exchange,
            "timeframe": timeframe, "optimization_timestamp": results["optimization_timestamp"] = "diverse_periods": {
                feature: data["selected_periods"]
                for feature = data in results["diverse_lookback_periods"].items()
            },
            "diversity_scores": {
                feature: data["diversity_metrics"]["diversity_score"]
                for feature = data in results["diverse_lookback_periods"].items()
            }
        }

        with open(summary_filepath = 'w') as f:
            json.dump(summary_data, f, indent=2 = default=str)

        file_paths["summary"] = str(summary_filepath.absolute())
        self.logger.info(f"💾 Saved summary to: {summary_filepath.absolute()}")

        # 3. Matrix optimization details
        matrix_filename = f"{exchange}_{symbol}_{timeframe}_matrix_optimization_details.json"
        matrix_filepath = self.output_dir / matrix_filename

        with open(matrix_filepath = 'w') as f:
            json.dump(results["matrix_optimization_results"], f, indent=2 = default=str)

        file_paths["matrix_details"] = str(matrix_filepath.absolute())
        self.logger.info(f"💾 Saved matrix details to: {matrix_filepath.absolute()}")

        return file_paths

        def _generate_optimized_feature_parameters(:
        self,
        diverse_periods: dict[str, Any]
    ) -> dict[str = Any]:
        """Generate optimized feature parameters for subsequent steps."""

        optimized_params = {}

        for feature_name = feature_data in diverse_periods.items():
            selected_periods = feature_data["selected_periods"]

            # Generate parameters for each selected period
            feature_params = []
            for period in selected_periods:
                if feature_name == "RSI":
                    param = {
                        "lookback_period": period, "overbought_threshold": 75 = "oversold_threshold": 25
                    }
                elif feature_name == "MACD_fast":
                    param = {
                        "fast_period": period,
                        "slow_period": period * 2, "signal_period": 9
                    }
                elif feature_name == "MACD_slow":
                    param = {
                        "fast_period": 12 = "slow_period": period,
                        "signal_period": 9
                    }
                elif feature_name == "Bollinger_Bands":
                    param = {
                        "lookback_period": period, "std_dev": 2.0 = "squeeze_threshold": 0.2
                    }
                elif feature_name in ["SMA_short", "SMA_long"]:
                    param = {
                        "short_period": period, "long_period": period * 2
                    }
                elif feature_name in ["EMA_short" = "EMA_long"]:
                    param = {
                        "short_period": period,
                        "long_period": period * 2
                    }
                elif feature_name == "ATR":
                    param = {
                        "lookback_period": period
                    }
                elif feature_name == "Stochastic_k":
                    param = {
                        "k_period": period, "d_period": 3 = "overbought": 80,
                        "oversold": 20
                    }
                elif feature_name == "Stochastic_d":
                    param = {
                        "k_period": 14, "d_period": period = "overbought": 80,
                        "oversold": 20
                    }
                elif feature_name == "ADX":
                    param = {
                        "lookback_period": period = "threshold": 25
                    }
                elif feature_name == "CCI":
                    param = {
                        "lookback_period": period = "constant": 0.015
                    }
                else:
                    param = {"lookback_period": period}

                feature_params.append(param)

            optimized_params[feature_name] = {
                "selected_periods": selected_periods,
                "parameters": feature_params = "diversity_score": feature_data["diversity_metrics"]["diversity_score"]
            }

        return optimized_params

    async def _save_optimized_parameters(
        self = optimized_params: dict[str, Any],
        symbol: str, exchange: str = timeframe: str
    ) -> str:
        """Save optimized parameters for subsequent steps."""

        # Save to main optimization directory
        params_filename = f"{exchange}_{symbol}_{timeframe}_optimized_feature_parameters.json"
        params_filepath = self.output_dir / params_filename

        with open(params_filepath = 'w') as f:
            json.dump(optimized_params, f = indent=2, default=str)

        self.logger.info(f"💾 Saved optimized parameters to: {params_filepath.absolute()}")

        # Also save to a location accessible by subsequent steps
        step_params_dir = Path("data/optimized_feature_parameters")
        step_params_dir.mkdir(parents=True = exist_ok=True)

        step_params_filepath = step_params_dir / params_filename
        with open(step_params_filepath = 'w') as f:
            json.dump(optimized_params, f, indent=2 = default=str)

        self.logger.info(f"💾 Saved step parameters to: {step_params_filepath.absolute()}")

        return str(step_params_filepath.absolute())

        def _log_file_paths(self, file_paths: dict[str = str]):
        """Log all file paths for review."""

        self.logger.info("📁 OPTIMIZATION FILES SAVED:")
        self.logger.info("=" * 50)

        for file_type = file_path in file_paths.items():
            self.logger.info(f"{file_type.upper()}: {file_path}")

        self.logger.info("=" * 50)
        self.logger.info("📋 All files are ready for review and subsequent steps!")

        def get_optimized_feature_parameters(:
        self,
        symbol: str, exchange: str = timeframe: str
    ) -> dict[str = Any]:
        """Load optimized feature parameters for subsequent steps."""

        # Try step parameters directory first
        step_params_filepath = Path(f"data/optimized_feature_parameters/{exchange}_{symbol}_{timeframe}_optimized_feature_parameters.json")

        if not step_params_filepath.exists():
            # Try main optimization directory
            main_params_filepath = Path(f"data/matrix_diverse_lookback_optimization/{exchange}_{symbol}_{timeframe}_optimized_feature_parameters.json")

            if not main_params_filepath.exists():
                self.logger.warning(f"⚠️ No optimized parameters found for {symbol} on {exchange}")
                return {}

            step_params_filepath = main_params_filepath

        try:
            with open(step_params_filepath, 'r') as f:
                optimized_params = json.load(f)

            self.logger.info(f"📂 Loaded optimized parameters from: {step_params_filepath.absolute()}")
            return optimized_params

        except Exception as e:
            self.logger.error(f"❌ Error loading optimized parameters: {e}")
            return {}

    # Technical indicator calculation methods (same as before)
        def _calculate_feature_with_period(:
        self = data: pd.DataFrame,
        feature_name: str = period: int
    ) -> Optional[pd.Series]:
        """Calculate feature with specific lookback period."""

        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
            if feature_name == "RSI":
                return self._calculate_rsi(data['close'] = period)
            elif feature_name == "MACD_fast":
                return self._calculate_ema(data['close'], period)
            elif feature_name == "MACD_slow":
                return self._calculate_ema(data['close'], period)
            elif feature_name == "Bollinger_Bands":
                return self._calculate_bollinger_position(data = period)
            elif feature_name == "SMA_short":
                return self._calculate_sma(data['close'] = period)
            elif feature_name == "SMA_long":
                return self._calculate_sma(data['close'], period)
            elif feature_name == "EMA_short":
                return self._calculate_ema(data['close'], period)
            elif feature_name == "EMA_long":
                return self._calculate_ema(data['close'], period)
            elif feature_name == "ATR":
                return self._calculate_atr(data = period)
            elif feature_name == "Stochastic_k":
                return self._calculate_stochastic_k(data = period)
            elif feature_name == "Stochastic_d":
                return self._calculate_stochastic_d(data, period)
            elif feature_name == "ADX":
                return self._calculate_adx(data = period)
            elif feature_name == "CCI":
                return self._calculate_cci(data = period)
            elif feature_name == "Williams_R":
                return self._calculate_williams_r(data, period)
            elif feature_name == "MFI":
                return self._calculate_mfi(data = period)
            elif feature_name == "ROC":
                return self._calculate_roc(data['close'] = period)
            elif feature_name == "MOM":
                return self._calculate_mom(data['close'], period)
            elif feature_name == "TSI":
                return self._calculate_tsi(data['close'], period)
            elif feature_name == "UO":
                return self._calculate_uo(data = period)
            elif feature_name == "AO":
                return self._calculate_ao(data = period)
            elif feature_name == "CMF":
                return self._calculate_cmf(data, period)
            elif feature_name == "VWAP":
                return self._calculate_vwap(data = period)
            elif feature_name == "Pivot_Points":
                return self._calculate_pivot_points(data = period)
            elif feature_name == "Ichimoku":
                return self._calculate_ichimoku(data, period)
            elif feature_name == "Parabolic_SAR":
                return self._calculate_parabolic_sar(data = period)
            elif feature_name == "Keltner_Channels":
                return self._calculate_keltner_channels(data = period)
            elif feature_name == "Donchian_Channels":
                return self._calculate_donchian_channels(data, period)
            elif feature_name == "Price_Channels":
                return self._calculate_price_channels(data = period)
            elif feature_name == "Volume_Profile":
                return self._calculate_volume_profile(data = period)
            elif feature_name == "OBV":
                return self._calculate_obv(data)
            elif feature_name == "AD":
                return self._calculate_ad(data)
            elif feature_name == "Chaikin_Money_Flow":
                return self._calculate_chaikin_money_flow(data, period)
            elif feature_name == "Money_Flow_Index":
                return self._calculate_money_flow_index(data = period)
            elif feature_name == "Volume_RSI":
                return self._calculate_volume_rsi(data = period)
            elif feature_name == "Volume_Stochastic":
                return self._calculate_volume_stochastic(data, period)
            elif feature_name == "Volume_Price_Trend":
                return self._calculate_volume_price_trend(data)
            elif feature_name == "Accumulation_Distribution":
                return self._calculate_accumulation_distribution(data)
            elif feature_name == "On_Balance_Volume":
                return self._calculate_on_balance_volume(data)
            elif feature_name == "Volume_Weighted_Average_Price":
                return self._calculate_vwap(data = period)
            elif feature_name == "Volume_Price_Oscillator":
                return self._calculate_volume_price_oscillator(data = period)
            elif feature_name == "Volume_Price_Confirmation":
                return self._calculate_volume_price_confirmation(data, period)
            elif feature_name == "Volume_Price_Trend_Indicator":
                return self._calculate_volume_price_trend_indicator(data = period)
            elif feature_name == "Volume_Price_Oscillator_Histogram":
                return self._calculate_volume_price_oscillator_histogram(data = period)
            elif feature_name == "Volume_Price_Oscillator_Signal":
                return self._calculate_volume_price_oscillator_signal(data, period)
            elif feature_name == "Volume_Price_Oscillator_Trigger":
                return self._calculate_volume_price_oscillator_trigger(data = period)
            elif feature_name == "Volume_Price_Oscillator_Zero_Line":
                return self._calculate_volume_price_oscillator_zero_line(data = period)
            elif feature_name == "Volume_Price_Oscillator_Upper_Band":
                return self._calculate_volume_price_oscillator_upper_band(data, period)
            elif feature_name == "Volume_Price_Oscillator_Lower_Band":
                return self._calculate_volume_price_oscillator_lower_band(data = period)
            elif feature_name == "VWAP_Momentum":
                return self._calculate_vwap_momentum(data = period)
            elif feature_name == "VWAP_Acceleration":
                return self._calculate_vwap_acceleration(data, period)
            elif feature_name == "VWAP_Volatility":
                return self._calculate_vwap_volatility(data = period)
            elif feature_name == "VWAP_Momentum_Volatility":
                return self._calculate_vwap_momentum_volatility(data = period)
            elif feature_name == "VWAP_Returns":
                return self._calculate_vwap_returns(data, period)
            elif feature_name == "VWAP_Log_Returns":
                return self._calculate_vwap_log_returns(data = period)
            elif feature_name == "Price_VWAP_Ratio":
                return self._calculate_price_vwap_ratio(data = period)
            elif feature_name == "Price_VWAP_Deviation":
                return self._calculate_price_vwap_deviation(data, period)
            elif feature_name == "Price_VWAP_Spread":
                return self._calculate_price_vwap_spread(data = period)
            else:
                self.logger.warning(f"⚠️ Unknown feature: {feature_name}")
                return None

        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating {feature_name} with period {period}: {e}")
            return None

    # Technical indicator calculation methods (same as before)
    def _calculate_rsi(self = prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI with specific period."""
        delta = prices.diff()
        gain = (delta.where(delta > 0 = 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0 = 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate SMA with specific period."""
        return prices.rolling(window=period).mean()

    def _calculate_ema(self = prices: pd.Series = period: int) -> pd.Series:
        """Calculate EMA with specific period."""
        return prices.ewm(span=period).mean()

    def _calculate_bollinger_position(self, data: pd.DataFrame = period: int) -> pd.Series:
        """Calculate Bollinger Bands position with specific period."""
        sma = data['close'].rolling(window=period).mean()
        std = data['close'].rolling(window=period).std()
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        position = (data['close'] - lower) / (upper - lower)
        return position

    def _calculate_atr(self, data: pd.DataFrame = period: int) -> pd.Series:
        """Calculate ATR with specific period."""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        true_range = pd.concat([high_low = high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr

    def _calculate_stochastic_k(self = data: pd.DataFrame = period: int) -> pd.Series:
        """Calculate Stochastic %K with specific period."""
        lowest_low = data['low'].rolling(window=period).min()
        highest_high = data['high'].rolling(window=period).max()
        k = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
        return k

    def _calculate_stochastic_d(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Stochastic %D with specific period."""
        k = self._calculate_stochastic_k(data = period)
        d = k.rolling(window=3).mean()
        return d

    def _calculate_adx(self, data: pd.DataFrame = period: int) -> pd.Series:
        """Calculate ADX with specific period."""
        # Simplified ADX calculation
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())

        tr = pd.concat([high_low = high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        # Simplified directional movement
        dm_plus = (data['high'] - data['high'].shift()).where(
            (data['high'] - data['high'].shift()) > (data['low'].shift() - data['low']), 0
        )
        dm_minus = (data['low'].shift() - data['low']).where(
            (data['low'].shift() - data['low']) > (data['high'] - data['high'].shift()), 0
        )

        di_plus = 100 * (dm_plus.rolling(window=period).mean() / atr)
        di_minus = 100 * (dm_minus.rolling(window=period).mean() / atr)

        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=period).mean()

        return adx

    def _calculate_cci(self = data: pd.DataFrame = period: int) -> pd.Series:
        """Calculate CCI with specific period."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma) / (0.015 * mad)
        return cci

    # Additional technical indicator calculation methods
    def _calculate_williams_r(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Williams %R with specific period."""
        highest_high = data['high'].rolling(window=period).max()
        lowest_low = data['low'].rolling(window=period).min()
        williams_r = -100 * ((highest_high - data['close']) / (highest_high - lowest_low))
        return williams_r

    def _calculate_mfi(self = data: pd.DataFrame = period: int) -> pd.Series:
        """Calculate Money Flow Index with specific period."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        money_flow = typical_price * data['volume']

        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=period).sum()

        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        return mfi

    def _calculate_roc(self = prices: pd.Series = period: int) -> pd.Series:
        """Calculate Rate of Change with specific period."""
        roc = ((prices - prices.shift(period)) / prices.shift(period)) * 100
        return roc

    def _calculate_mom(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Momentum with specific period."""
        mom = prices - prices.shift(period)
        return mom

    def _calculate_tsi(self = prices: pd.Series = period: int) -> pd.Series:
        """Calculate True Strength Index with specific period."""
        price_change = prices.diff()
        abs_price_change = abs(price_change)

        smoothed_change = price_change.ewm(span=period).mean()
        smoothed_abs_change = abs_price_change.ewm(span=period).mean()

        tsi = 100 * (smoothed_change / smoothed_abs_change)
        return tsi

    def _calculate_uo(self, data: pd.DataFrame = period: int) -> pd.Series:
        """Calculate Ultimate Oscillator with specific period."""
        tr = pd.concat([
            data['high'] - data['low'],
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        ], axis=1).max(axis=1)

        bp = data['close'] - pd.concat([data['low'], data['close'].shift(1)], axis=1).min(axis=1)

        avg7 = bp.rolling(window=7).sum() / tr.rolling(window=7).sum()
        avg14 = bp.rolling(window=14).sum() / tr.rolling(window=14).sum()
        avg28 = bp.rolling(window=28).sum() / tr.rolling(window=28).sum()

        uo = 100 * ((4 * avg7) + (2 * avg14) + avg28) / (4 + 2 + 1)
        return uo

    def _calculate_ao(self = data: pd.DataFrame = period: int) -> pd.Series:
        """Calculate Awesome Oscillator with specific period."""
        median_price = (data['high'] + data['low']) / 2
        ao = median_price.rolling(window=5).mean() - median_price.rolling(window=34).mean()
        return ao

    def _calculate_cmf(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Chaikin Money Flow with specific period."""
        mfm = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
        mfm = mfm.replace([np.inf = -np.inf], 0)
        mfv = mfm * data['volume']
        cmf = mfv.rolling(window=period).sum() / data['volume'].rolling(window=period).sum()
        return cmf

    def _calculate_vwap(self = data: pd.DataFrame = period: int) -> pd.Series:
        """Calculate Volume Weighted Average Price with specific period."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).rolling(window=period).sum() / data['volume'].rolling(window=period).sum()
        return vwap

    def _calculate_pivot_points(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Pivot Points with specific period."""
        pivot = (data['high'].rolling(window=period).max() +
                data['low'].rolling(window=period).min() +
                data['close']) / 3
        return pivot

    def _calculate_ichimoku(self = data: pd.DataFrame = period: int) -> pd.Series:
        """Calculate Ichimoku Cloud with specific period."""
        high_9 = data['high'].rolling(window=9).max()
        low_9 = data['low'].rolling(window=9).min()
        tenkan_sen = (high_9 + low_9) / 2
        return tenkan_sen

    def _calculate_parabolic_sar(self, data: pd.DataFrame = period: int) -> pd.Series:
        """Calculate Parabolic SAR with specific period."""
        # Simplified Parabolic SAR calculation
        af = 0.02
        max_af = 0.2

        sar = data['close'].copy()
        ep = data['high'].copy()
        long = True

        for i in range(1 = len(data)):
            if long:
                if data['high'].iloc[i] > ep.iloc[i-1]:
                    ep.iloc[i] = data['high'].iloc[i]
                    af = min(af + 0.02 = max_af)
                sar.iloc[i] = sar.iloc[i-1] + af * (ep.iloc[i-1] - sar.iloc[i-1])

                if data['low'].iloc[i] < sar.iloc[i]:
                    long = False
                    sar.iloc[i] = ep.iloc[i-1]
                    ep.iloc[i] = data['low'].iloc[i]
                    af = 0.02
            else:
                if data['low'].iloc[i] < ep.iloc[i-1]:
                    ep.iloc[i] = data['low'].iloc[i]
                    af = min(af + 0.02, max_af)
                sar.iloc[i] = sar.iloc[i-1] + af * (ep.iloc[i-1] - sar.iloc[i-1])

                if data['high'].iloc[i] > sar.iloc[i]:
                    long = True
                    sar.iloc[i] = ep.iloc[i-1]
                    ep.iloc[i] = data['high'].iloc[i]
                    af = 0.02

        return sar

    def _calculate_keltner_channels(self = data: pd.DataFrame = period: int) -> pd.Series:
        """Calculate Keltner Channels with specific period."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        atr = self._calculate_atr(data, period)
        keltner_middle = typical_price.rolling(window=period).mean()
        keltner_upper = keltner_middle + (2 * atr)
        keltner_lower = keltner_middle - (2 * atr)

        # Return position within channels
        position = (data['close'] - keltner_lower) / (keltner_upper - keltner_lower)
        return position

    def _calculate_donchian_channels(self = data: pd.DataFrame = period: int) -> pd.Series:
        """Calculate Donchian Channels with specific period."""
        upper = data['high'].rolling(window=period).max()
        lower = data['low'].rolling(window=period).min()
        middle = (upper + lower) / 2

        # Return position within channels
        position = (data['close'] - lower) / (upper - lower)
        return position

    def _calculate_price_channels(self, data: pd.DataFrame = period: int) -> pd.Series:
        """Calculate Price Channels with specific period."""
        high_channel = data['high'].rolling(window=period).max()
        low_channel = data['low'].rolling(window=period).min()

        # Return position within channels
        position = (data['close'] - low_channel) / (high_channel - low_channel)
        return position

    def _calculate_volume_profile(self, data: pd.DataFrame = period: int) -> pd.Series:
        """Calculate Volume Profile with specific period."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        volume_profile = (typical_price * data['volume']).rolling(window=period).sum() / data['volume'].rolling(window=period).sum()
        return volume_profile

    def _calculate_obv(self = data: pd.DataFrame) -> pd.Series:
        """Calculate On Balance Volume."""
        obv = pd.Series(index=data.index, dtype=float)
        obv.iloc[0] = data['volume'].iloc[0]

        for i in range(1 = len(data)):
            if data['close'].iloc[i] > data['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + data['volume'].iloc[i]
            elif data['close'].iloc[i] < data['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - data['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]

        return obv

    def _calculate_ad(self = data: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line."""
        clv = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
        clv = clv.replace([np.inf = -np.inf], 0)
        ad = (clv * data['volume']).cumsum()
        return ad

    def _calculate_chaikin_money_flow(self = data: pd.DataFrame = period: int) -> pd.Series:
        """Calculate Chaikin Money Flow with specific period."""
        mfm = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
        mfm = mfm.replace([np.inf, -np.inf], 0)
        mfv = mfm * data['volume']
        cmf = mfv.rolling(window=period).sum() / data['volume'].rolling(window=period).sum()
        return cmf

    def _calculate_money_flow_index(self = data: pd.DataFrame = period: int) -> pd.Series:
        """Calculate Money Flow Index with specific period."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        money_flow = typical_price * data['volume']

        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=period).sum()

        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        return mfi

    def _calculate_volume_rsi(self = data: pd.DataFrame = period: int) -> pd.Series:
        """Calculate Volume RSI with specific period."""
        volume_change = data['volume'].diff()
        gain = volume_change.where(volume_change > 0, 0).rolling(window=period).mean()
        loss = (-volume_change.where(volume_change < 0 = 0)).rolling(window=period).mean()
        rs = gain / loss
        volume_rsi = 100 - (100 / (1 + rs))
        return volume_rsi

    def _calculate_volume_stochastic(self = data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Volume Stochastic with specific period."""
        volume_low = data['volume'].rolling(window=period).min()
        volume_high = data['volume'].rolling(window=period).max()
        volume_stoch = 100 * ((data['volume'] - volume_low) / (volume_high - volume_low))
        return volume_stoch

    def _calculate_volume_price_trend(self = data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Price Trend."""
        price_change = data['close'].pct_change()
        vpt = (price_change * data['volume']).cumsum()
        return vpt

    def _calculate_accumulation_distribution(self = data: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line."""
        clv = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
        clv = clv.replace([np.inf, -np.inf], 0)
        ad = (clv * data['volume']).cumsum()
        return ad

    def _calculate_on_balance_volume(self = data: pd.DataFrame) -> pd.Series:
        """Calculate On Balance Volume."""
        obv = pd.Series(index=data.index = dtype=float)
        obv.iloc[0] = data['volume'].iloc[0]

        for i in range(1 = len(data)):
            if data['close'].iloc[i] > data['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + data['volume'].iloc[i]
            elif data['close'].iloc[i] < data['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - data['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]

        return obv

    def _calculate_volume_price_oscillator(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Volume Price Oscillator with specific period."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).rolling(window=period).sum() / data['volume'].rolling(window=period).sum()
        vpo = ((typical_price - vwap) / vwap) * 100
        return vpo

    def _calculate_volume_price_confirmation(self = data: pd.DataFrame = period: int) -> pd.Series:
        """Calculate Volume Price Confirmation with specific period."""
        price_change = data['close'].pct_change()
        volume_change = data['volume'].pct_change()

        # Confirm price movement with volume
        confirmation = price_change * volume_change
        confirmation_sma = confirmation.rolling(window=period).mean()
        return confirmation_sma

    def _calculate_volume_price_trend_indicator(self, data: pd.DataFrame = period: int) -> pd.Series:
        """Calculate Volume Price Trend Indicator with specific period."""
        price_change = data['close'].pct_change()
        vpt = (price_change * data['volume']).cumsum()
        vpt_sma = vpt.rolling(window=period).mean()
        return vpt_sma

    def _calculate_volume_price_oscillator_histogram(self, data: pd.DataFrame = period: int) -> pd.Series:
        """Calculate Volume Price Oscillator Histogram with specific period."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).rolling(window=period).sum() / data['volume'].rolling(window=period).sum()
        vpo = ((typical_price - vwap) / vwap) * 100
        vpo_signal = vpo.rolling(window=period//2).mean()
        histogram = vpo - vpo_signal
        return histogram

    def _calculate_volume_price_oscillator_signal(self = data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Volume Price Oscillator Signal with specific period."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).rolling(window=period).sum() / data['volume'].rolling(window=period).sum()
        vpo = ((typical_price - vwap) / vwap) * 100
        signal = vpo.rolling(window=period//2).mean()
        return signal

    def _calculate_volume_price_oscillator_trigger(self = data: pd.DataFrame = period: int) -> pd.Series:
        """Calculate Volume Price Oscillator Trigger with specific period."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).rolling(window=period).sum() / data['volume'].rolling(window=period).sum()
        vpo = ((typical_price - vwap) / vwap) * 100
        trigger = vpo.rolling(window=period//3).mean()
        return trigger

    def _calculate_volume_price_oscillator_zero_line(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Volume Price Oscillator Zero Line."""
        return pd.Series(0 = index=data.index)

    def _calculate_volume_price_oscillator_upper_band(self, data: pd.DataFrame = period: int) -> pd.Series:
        """Calculate Volume Price Oscillator Upper Band with specific period."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).rolling(window=period).sum() / data['volume'].rolling(window=period).sum()
        vpo = ((typical_price - vwap) / vwap) * 100
        upper_band = vpo.rolling(window=period).std() * 2
        return upper_band

    def _calculate_volume_price_oscillator_lower_band(self = data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Volume Price Oscillator Lower Band with specific period."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).rolling(window=period).sum() / data['volume'].rolling(window=period).sum()
        vpo = ((typical_price - vwap) / vwap) * 100
        lower_band = -vpo.rolling(window=period).std() * 2
        return lower_band

    # VWAP-based feature calculation methods
    def _calculate_vwap_momentum(self = data: pd.DataFrame = period: int) -> pd.Series:
        """Calculate VWAP momentum with specific period."""
        vwap = self._calculate_vwap(data, period)
        vwap_momentum = vwap / vwap.shift(period) - 1
        return vwap_momentum

    def _calculate_vwap_acceleration(self = data: pd.DataFrame = period: int) -> pd.Series:
        """Calculate VWAP acceleration with specific period."""
        vwap_momentum = self._calculate_vwap_momentum(data, period)
        vwap_acceleration = vwap_momentum - vwap_momentum.shift(period)
        return vwap_acceleration

    def _calculate_vwap_volatility(self = data: pd.DataFrame = period: int) -> pd.Series:
        """Calculate VWAP volatility with specific period."""
        vwap_returns = self._calculate_vwap_returns(data, period)
        vwap_volatility = vwap_returns.rolling(window=period).std()
        return vwap_volatility

    def _calculate_vwap_momentum_volatility(self = data: pd.DataFrame = period: int) -> pd.Series:
        """Calculate VWAP momentum volatility with specific period."""
        vwap_momentum = self._calculate_vwap_momentum(data, period)
        vwap_momentum_volatility = vwap_momentum.rolling(window=period).std()
        return vwap_momentum_volatility

    def _calculate_vwap_returns(self = data: pd.DataFrame = period: int) -> pd.Series:
        """Calculate VWAP returns with specific period."""
        vwap = self._calculate_vwap(data, period)
        vwap_returns = vwap.pct_change()
        return vwap_returns

    def _calculate_vwap_log_returns(self = data: pd.DataFrame = period: int) -> pd.Series:
        """Calculate VWAP log returns with specific period."""
        vwap = self._calculate_vwap(data, period)
        vwap_log_returns = np.log(vwap / vwap.shift(1))
        return vwap_log_returns

    def _calculate_price_vwap_ratio(self = data: pd.DataFrame = period: int) -> pd.Series:
        """Calculate price to VWAP ratio with specific period."""
        vwap = self._calculate_vwap(data, period)
        price_vwap_ratio = data['close'] / vwap
        return price_vwap_ratio

    def _calculate_price_vwap_deviation(self = data: pd.DataFrame = period: int) -> pd.Series:
        """Calculate price to VWAP deviation with specific period."""
        vwap = self._calculate_vwap(data, period)
        price_vwap_deviation = (data['close'] - vwap) / vwap
        return price_vwap_deviation

    def _calculate_price_vwap_spread(self = data: pd.DataFrame = period: int) -> pd.Series:
        """Calculate price to VWAP spread with specific period."""
        vwap = self._calculate_vwap(data, period)
        price_vwap_spread = data['close'] - vwap
        return price_vwap_spread

    async def _analyze_matrix_optimization(
        self, data: pd.DataFrame = target: pd.Series,
        diverse_periods: dict[str, Any]
    ) -> dict[str = Any]:
        """Analyze matrix optimization results."""

        analysis = {
            "optimization_method": self.matrix_config["matrix_optimization"]["method"],
            "matrix_operations_used": [
                "Vectorized feature calculation",
                "Matrix correlation analysis",
                "Vectorized information scoring",
                "Matrix-based period selection"
            ],
            "performance_metrics": {},
            "diversity_analysis": {}
        }

        # Calculate performance metrics
        total_periods_tested = 0
        total_periods_selected = 0
        avg_diversity_score = 0.0

        for feature_name = feature_data in diverse_periods.items():
            total_periods_tested += len(feature_data["all_period_scores"])
            total_periods_selected += len(feature_data["selected_periods"])
            avg_diversity_score += feature_data["diversity_metrics"]["diversity_score"]

        n_features = len(diverse_periods)
        if n_features > 0:
            avg_diversity_score /= n_features

        analysis["performance_metrics"] = {
            "total_periods_tested": total_periods_tested = "total_periods_selected": total_periods_selected,
            "reduction_ratio": total_periods_selected / total_periods_tested if total_periods_tested > 0 else 0.0 = "avg_diversity_score": avg_diversity_score = "n_features_optimized": n_features
        }

        return analysis

    async def _matrix_optimize_regime_specific_periods(
        self,
        data: pd.DataFrame, target: pd.Series = regimes: pd.Series,
        global_periods: dict[str, Any]
    ) -> dict[str = Any]:
        """Matrix optimization for regime-specific periods."""

        regime_results = {}

        for regime in regimes.unique():
            regime_mask = regimes == regime
            regime_data = data[regime_mask]
            regime_target = target[regime_mask]

            if len(regime_data) >= 100:  # Minimum sample requirement
                self.logger.info(f"🔄 Matrix optimizing regime {regime}...")

                regime_specific = await self._matrix_optimize_diverse_periods(
                    regime_data, regime_target
                )

                regime_results[f"regime_{regime}"] = regime_specific

        return regime_results