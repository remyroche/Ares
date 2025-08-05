# src/training/regime_specific_tpsl_optimizer.py

"""
Regime-Specific SL/TP Optimizer

This module provides regime-specific optimization of Stop Loss (SL) and Take Profit (TP)
parameters based on the current market regime identified by the HMM classifier.

The optimizer uses the timeframe analysis results to determine optimal SL/TP levels
for each market regime, considering the average duration and success rates from
the analysis.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analyst.unified_regime_classifier import UnifiedRegimeClassifier
from src.config import CONFIG
from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger


class RegimeSpecificTPSLOptimizer:
    """
    Optimizes Take Profit (TP) and Stop Loss (SL) parameters based on market regime.

    This optimizer uses the HMM regime classifier to identify the current market regime
    and then applies regime-specific optimization based on the timeframe analysis results.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the regime-specific TP/SL optimizer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("RegimeSpecificTPSLOptimizer")

        # Initialize Unified regime classifier
        self.regime_classifier = UnifiedRegimeClassifier(config)

        # Regime-specific parameters from timeframe analysis
        self.regime_parameters = {
            "BULL_TREND": {
                "target_pct": 0.4,
                "stop_pct": 0.2,
                "risk_reward_ratio": 2.0,
                "avg_duration_minutes": 37.2,
                "success_rate": 6.99,
                "frequency_score": 100.0,
            },
            "BEAR_TREND": {
                "target_pct": 0.4,
                "stop_pct": 0.2,
                "risk_reward_ratio": 2.0,
                "avg_duration_minutes": 37.2,
                "success_rate": 6.99,
                "frequency_score": 100.0,
            },
            "SIDEWAYS_RANGE": {
                "target_pct": 0.5,
                "stop_pct": 0.3,
                "risk_reward_ratio": 1.67,
                "avg_duration_minutes": 67.4,
                "success_rate": 7.81,
                "frequency_score": 100.0,
            },
            "SR_ZONE_ACTION": {
                "target_pct": 0.4,
                "stop_pct": 0.2,
                "risk_reward_ratio": 2.0,
                "avg_duration_minutes": 37.2,
                "success_rate": 6.99,
                "frequency_score": 100.0,
            },
            "HIGH_IMPACT_CANDLE": {
                "target_pct": 0.6,
                "stop_pct": 0.1,
                "risk_reward_ratio": 6.0,
                "avg_duration_minutes": 14.7,
                "success_rate": 1.06,
                "frequency_score": 100.0,
            },
        }

        # Optimization configuration
        self.optimization_config = config.get("regime_specific_tpsl_optimizer", {})
        self.n_trials = self.optimization_config.get("n_trials", 100)
        self.min_trades = self.optimization_config.get("min_trades", 20)
        self.optimization_metric = self.optimization_config.get(
            "optimization_metric",
            "sharpe_ratio",
        )

        # Model storage
        self.model_dir = os.path.join(CONFIG["CHECKPOINT_DIR"], "regime_tpsl_models")
        os.makedirs(self.model_dir, exist_ok=True)

        # Optimization results cache
        self.optimization_results: dict[str, dict[str, Any]] = {}
        self.last_optimization_time: datetime | None = None

    @handle_specific_errors(
        error_handlers={
            ValueError: (
                False,
                "Invalid regime-specific TP/SL optimization configuration",
            ),
            AttributeError: (False, "Missing required optimization parameters"),
        },
        default_return=False,
        context="regime-specific TP/SL optimizer initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize the regime-specific TP/SL optimizer.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Regime-Specific TP/SL Optimizer...")

            # Initialize HMM classifier
            if not await self._initialize_regime_classifier():
                self.logger.error("Failed to initialize HMM classifier")
                return False

            # Load existing optimization results
            await self._load_optimization_results()

            self.logger.info(
                "✅ Regime-Specific TP/SL Optimizer initialized successfully",
            )
            return True

        except Exception as e:
            self.logger.error(
                f"❌ Failed to initialize Regime-Specific TP/SL Optimizer: {e}",
            )
            return False

    async def _initialize_regime_classifier(self) -> bool:
        """
        Initialize the HMM regime classifier.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Try to load existing HMM model
            model_path = os.path.join(
                CONFIG["CHECKPOINT_DIR"],
                "analyst_models",
                "hmm_regime_classifier_1h.joblib",
            )

            if os.path.exists(model_path):
                if self.regime_classifier.load_models():
                    self.logger.info("✅ Loaded existing HMM regime classifier")
                    return True
                self.logger.warning(
                    "Failed to load existing HMM model, will train new one",
                )

            self.logger.info(
                "HMM classifier not trained yet, will be trained when data is available",
            )
            return True

        except Exception as e:
            self.logger.error(f"Error initializing HMM classifier: {e}")
            return False

    async def _load_optimization_results(self) -> None:
        """
        Load existing optimization results from disk.
        """
        try:
            results_file = os.path.join(self.model_dir, "optimization_results.json")
            if os.path.exists(results_file):
                import json

                with open(results_file) as f:
                    self.optimization_results = json.load(f)
                self.logger.info(
                    f"✅ Loaded {len(self.optimization_results)} regime optimization results",
                )
        except Exception as e:
            self.logger.warning(f"Could not load optimization results: {e}")

    async def _save_optimization_results(self) -> None:
        """
        Save optimization results to disk.
        """
        try:
            results_file = os.path.join(self.model_dir, "optimization_results.json")
            import json

            with open(results_file, "w") as f:
                json.dump(self.optimization_results, f, indent=2, default=str)
            self.logger.info("✅ Saved optimization results")
        except Exception as e:
            self.logger.error(f"Failed to save optimization results: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="regime identification",
    )
    async def identify_current_regime(
        self,
        current_data: pd.DataFrame,
    ) -> tuple[str, float, dict[str, Any]]:
        """
        Identify the current market regime using the HMM classifier.

        Args:
            current_data: Current market data (must be 1h timeframe)

        Returns:
            Tuple of (regime, confidence, additional_info)
        """
        try:
            if not self.regime_classifier.trained:
                self.logger.warning("HMM classifier not trained, using default regime")
                return "SIDEWAYS_RANGE", 0.5, {"method": "default"}

            regime, confidence, info = self.regime_classifier.predict_regime(
                current_data,
            )
            self.logger.info(
                f"Identified regime: {regime} (confidence: {confidence:.2f})",
            )
            return regime, confidence, info

        except Exception as e:
            self.logger.error(f"Error identifying regime: {e}")
            return "SIDEWAYS_RANGE", 0.5, {"method": "fallback", "error": str(e)}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="regime-specific TP/SL optimization",
    )
    async def optimize_tpsl_for_regime(
        self,
        regime: str,
        historical_data: pd.DataFrame,
        current_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Optimize TP/SL parameters for a specific market regime.

        Args:
            regime: Market regime to optimize for
            historical_data: Historical data for optimization
            current_data: Current market data

        Returns:
            Dictionary with optimized TP/SL parameters
        """
        try:
            self.logger.info(f"🎯 Optimizing TP/SL for regime: {regime}")

            # Get base parameters for this regime
            base_params = self.regime_parameters.get(
                regime,
                self.regime_parameters["SIDEWAYS_RANGE"],
            )

            # Create optimization study
            study = optuna.create_study(
                direction="maximize",
                study_name=f"tpsl_optimization_{regime}",
            )

            # Define objective function
            def objective(trial):
                return self._evaluate_tpsl_parameters(
                    trial,
                    regime,
                    historical_data,
                    base_params,
                )

            # Run optimization
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

            # Get best parameters
            best_params = study.best_params
            best_value = study.best_value

            # Combine with base parameters
            optimized_params = {
                **base_params,
                **best_params,
                "optimization_score": best_value,
                "optimization_trials": self.n_trials,
                "optimization_time": datetime.now().isoformat(),
            }

            # Cache results
            self.optimization_results[regime] = optimized_params
            await self._save_optimization_results()

            self.logger.info(f"✅ Optimized TP/SL for {regime}: {best_params}")
            return optimized_params

        except Exception as e:
            self.logger.error(f"Error optimizing TP/SL for regime {regime}: {e}")
            return self.regime_parameters.get(
                regime,
                self.regime_parameters["SIDEWAYS_RANGE"],
            )

    def _evaluate_tpsl_parameters(
        self,
        trial: optuna.Trial,
        regime: str,
        historical_data: pd.DataFrame,
        base_params: dict[str, Any],
    ) -> float:
        """
        Evaluate TP/SL parameters using backtesting simulation.

        Args:
            trial: Optuna trial object
            regime: Market regime
            historical_data: Historical data for backtesting
            base_params: Base parameters for the regime

        Returns:
            float: Optimization score (higher is better)
        """
        try:
            # Suggest parameters within reasonable bounds
            target_pct = trial.suggest_float(
                "target_pct",
                base_params["target_pct"] * 0.5,
                base_params["target_pct"] * 1.5,
            )
            stop_pct = trial.suggest_float(
                "stop_pct",
                base_params["stop_pct"] * 0.5,
                base_params["stop_pct"] * 1.5,
            )

            # Ensure target > stop
            if target_pct <= stop_pct:
                return -1.0

            # Run simplified backtest
            trades = self._simulate_trades(
                historical_data,
                target_pct,
                stop_pct,
                regime,
            )

            if len(trades) < self.min_trades:
                return -1.0

            # Calculate performance metrics
            returns = [trade["return"] for trade in trades]
            total_return = sum(returns)
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8)
            win_rate = len([r for r in returns if r > 0]) / len(returns)

            # Combine metrics based on optimization target
            if self.optimization_metric == "sharpe_ratio":
                score = sharpe_ratio
            elif self.optimization_metric == "total_return":
                score = total_return
            elif self.optimization_metric == "win_rate":
                score = win_rate
            else:
                score = sharpe_ratio * 0.4 + total_return * 0.3 + win_rate * 0.3

            return score

        except Exception as e:
            self.logger.error(f"Error in parameter evaluation: {e}")
            return -1.0

    def _simulate_trades(
        self,
        data: pd.DataFrame,
        target_pct: float,
        stop_pct: float,
        regime: str,
    ) -> list[dict[str, Any]]:
        """
        Simulate trades using given TP/SL parameters.

        Args:
            data: Historical price data
            target_pct: Take profit percentage
            stop_pct: Stop loss percentage
            regime: Market regime

        Returns:
            List of trade dictionaries
        """
        trades = []
        position_open = False
        entry_price = 0.0
        entry_time = None

        for i in range(1, len(data)):
            current_price = data.iloc[i]["close"]
            high_price = data.iloc[i]["high"]
            low_price = data.iloc[i]["low"]

            if not position_open:
                # Simple entry condition (can be enhanced)
                if data.iloc[i]["close"] > data.iloc[i - 1]["close"]:
                    position_open = True
                    entry_price = current_price
                    entry_time = data.index[i]
            # Check for TP/SL
            elif high_price >= entry_price * (1 + target_pct):
                # Take profit hit
                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": data.index[i],
                        "entry_price": entry_price,
                        "exit_price": entry_price * (1 + target_pct),
                        "return": target_pct,
                        "type": "TP",
                    },
                )
                position_open = False
            elif low_price <= entry_price * (1 - stop_pct):
                # Stop loss hit
                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": data.index[i],
                        "entry_price": entry_price,
                        "exit_price": entry_price * (1 - stop_pct),
                        "return": -stop_pct,
                        "type": "SL",
                    },
                )
                position_open = False

        return trades

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="regime-specific TP/SL prediction",
    )
    async def get_optimized_tpsl(
        self,
        current_data: pd.DataFrame,
        historical_data: pd.DataFrame,
        force_optimization: bool = False,
    ) -> dict[str, Any]:
        """
        Get optimized TP/SL parameters for the current market regime.

        Args:
            current_data: Current market data (1h timeframe)
            historical_data: Historical data for optimization
            force_optimization: Force re-optimization even if cached

        Returns:
            Dictionary with optimized TP/SL parameters
        """
        try:
            # Identify current regime
            regime, confidence, regime_info = await self.identify_current_regime(
                current_data,
            )

            # Check if we have cached results for this regime
            if not force_optimization and regime in self.optimization_results:
                cached_params = self.optimization_results[regime]
                self.logger.info(f"Using cached TP/SL parameters for {regime}")
                return {
                    **cached_params,
                    "regime": regime,
                    "confidence": confidence,
                    "regime_info": regime_info,
                }

            # Optimize for current regime
            optimized_params = await self.optimize_tpsl_for_regime(
                regime,
                historical_data,
                current_data,
            )

            return {
                **optimized_params,
                "regime": regime,
                "confidence": confidence,
                "regime_info": regime_info,
            }

        except Exception as e:
            self.logger.error(f"Error getting optimized TP/SL: {e}")
            # Return default parameters
            return {
                **self.regime_parameters["SIDEWAYS_RANGE"],
                "regime": "SIDEWAYS_RANGE",
                "confidence": 0.5,
                "regime_info": {"method": "fallback", "error": str(e)},
            }

    def get_regime_statistics(self) -> dict[str, Any]:
        """
        Get statistics about regime-specific TP/SL optimization.

        Returns:
            Dictionary with optimization statistics
        """
        return {
            "optimized_regimes": list(self.optimization_results.keys()),
            "total_optimizations": len(self.optimization_results),
            "last_optimization_time": self.last_optimization_time,
            "regime_parameters": self.regime_parameters,
        }
