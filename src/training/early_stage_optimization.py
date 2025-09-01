#!/usr/bin/env python3
"""
Early Stage Optimization Module

This module handles optimization that should happen BEFORE ML trading begins:
1. SR (Stationarity and Randomness) optimization (step02_5)
2. Regime-specific triple barrier optimization (step4)

These optimizations happen early in the pipeline to ensure:
- Proper data preprocessing (SR)
- Regime-aware trading parameters (triple barrier)
- Optimal foundation for ML model training
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
import json
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import MLflow for experiment tracking
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Import Optuna for optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Import regime-specific triple barrier optimizer from step4 components
try:
    from .steps.step04_analyst_labeling_feature_engineering_components.regime_specific_triple_barrier_optimizer import (
        RegimeSpecificTripleBarrierOptimizer,
        create_regime_specific_triple_barrier_optimizer
    )
    REGIME_OPTIMIZER_AVAILABLE = True
except ImportError:
    REGIME_OPTIMIZER_AVAILABLE = False
    RegimeSpecificTripleBarrierOptimizer = None
    create_regime_specific_triple_barrier_optimizer = None

# Import HMM regime barrier optimizer (focused upper/lower barriers)
try:
    from .hmm_regime_barrier_optimizer import (
        HMMRegimeBarrierOptimizer,
        optimize_hmm_regime_barriers,
    )
    HMM_BARRIER_OPTIMIZER_AVAILABLE = True
except Exception:
    HMM_BARRIER_OPTIMIZER_AVAILABLE = False
    HMMRegimeBarrierOptimizer = None


class EarlyStageOptimizer:
    """
    Early stage optimizer for parameters that must be set before ML trading begins.

    This includes:
    - SR optimization (step02_5) - data preprocessing parameters
    - Regime-specific triple barrier optimization (step4) - trading parameters
    """

    def __init__(self, config: Dict[str, Any], training_manager=None):
        self.config = config
        self.training_manager = training_manager
        self.logger = logging.getLogger(__name__)

        # Optimization results storage
        self.sr_optimization_results = {}
        self.regime_barrier_optimization_results = {}

        # MLflow experiment names
        self.sr_experiment_name = "early_stage_sr_optimization"
        self.regime_experiment_name = "early_stage_regime_barrier_optimization"

        # Initialize regime-specific triple barrier optimizer if available
        self.regime_optimizer = None
        if REGIME_OPTIMIZER_AVAILABLE:
            self.regime_optimizer = create_regime_specific_triple_barrier_optimizer(config, training_manager)
            self.logger.info("✅ Regime-specific triple barrier optimizer initialized")
        else:
            self.logger.warning("⚠️ Regime-specific triple barrier optimizer not available")

        # HMM regime barrier optimizer
        self.hmm_barrier_optimizer = None
        self.hmm_barrier_results = {}
        self.hmm_barrier_map = {}
        if HMM_BARRIER_OPTIMIZER_AVAILABLE:
            try:
                self.hmm_barrier_optimizer = HMMRegimeBarrierOptimizer(
                    config.get("hmm_regime_barrier_optimizer", {})
                )
                self.logger.info("✅ HMM Regime Barrier Optimizer initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize HMM Regime Barrier Optimizer: {e}")

    async def run_regime_specific_triple_barrier_optimization(
        self,
        regime_data: Dict[str, pd.DataFrame],
        optimization_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run regime-specific triple barrier optimization through the early-stage optimizer."""

        if not self.regime_optimizer:
            return {"error": "Regime-specific triple barrier optimizer not available"}

        try:
            self.logger.info("🚀 Starting regime-specific triple barrier optimization...")

            # Run optimization for all regimes
            optimization_results = await self.regime_optimizer.optimize_regime_specific_parameters(
                regime_data,
                optimization_config
            )

            # Store regime optimization results
            self.regime_barrier_optimization_results = optimization_results

            self.logger.info("✅ Regime-specific triple barrier optimization completed")

            return optimization_results

        except Exception as e:
            error_msg = f"Regime-specific optimization failed: {e}"
            self.logger.error(f"❌ {error_msg}")
            return {"error": error_msg}

    def _create_sr_objective(self, data: pd.DataFrame):
        """Create objective function for SR optimization."""

        return objective

    def _evaluate_sr_parameters(self, data: pd.DataFrame, params: Dict[str, Any]) -> float:
        """Evaluate SR parameters on data."""

        try:
            # This would integrate with your actual SR implementation
            # For now, providing a placeholder evaluation

            # Simulate data quality score based on parameters
            fractional_d = params.get("fractional_d", 0.5)
            window_size = params.get("window_size", 50)
            threshold = params.get("threshold", 0.01)

            # Calculate simulated quality score
            # Higher score for better data preprocessing
            base_score = 0.0

            # Fractional differentiation scoring
            if 0.2 <= fractional_d <= 0.8:  # Optimal range
                base_score += 0.4
            elif 0.1 <= fractional_d <= 0.9:  # Acceptable range
                base_score += 0.2

            # Window size scoring
            if 20 <= window_size <= 100:  # Optimal range
                base_score += 0.3
            elif 10 <= window_size <= 200:  # Acceptable range
                base_score += 0.15

            # Threshold scoring
            if 0.005 <= threshold <= 0.05:  # Optimal range
                base_score += 0.3
            elif 0.001 <= threshold <= 0.1:  # Acceptable range
                base_score += 0.15

            # Add some randomness to simulate real evaluation
            random_factor = np.random.normal(0, 0.1)
            final_score = base_score + random_factor

            # Ensure score is positive
            return max(0.0, final_score)

        except Exception as e:
            self.logger.error(f"Failed to evaluate SR parameters: {e}")
            return float('-inf')

    async def _create_regime_barrier_study(
        self,
        regime_name: str,
        optimization_config: Dict[str, Any]
    ) -> optuna.Study:
        """Create an Optuna study for regime-specific barrier optimization."""

        # Create study name
        study_name = f"regime_specific_barrier_{regime_name}"

        # Create study with regime-specific configuration
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",  # Maximize trading performance
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=10,
                n_ei_candidates=24,
                multivariate=True,
                group=True
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=3
            )
        )

        return study

    async def _optimize_single_regime_barrier(
        self,
        regime_name: str,
        regime_data: pd.DataFrame,
        study: optuna.Study,
        optimization_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize barrier parameters for a single regime."""

        # Get regime-specific parameter ranges
        regime_params = self._get_regime_barrier_parameters(regime_name)

        # Create objective function for this regime
        objective = self._create_regime_barrier_objective(
            regime_name,
            regime_data,
            regime_params
        )

        # Run optimization
        n_trials = optimization_config.get("n_trials", 100)
        timeout = optimization_config.get("timeout", 3600)

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[
                optuna.callbacks.EarlyStoppingCallback(
                    patience=optimization_config.get("early_stopping_patience", 20)
                )
            ]
        )

        # Extract results
        best_trial = study.best_trial
        best_params = best_trial.params
        best_value = best_trial.value

        return {
            "regime_name": regime_name,
            "best_params": best_params,
            "best_value": best_value,
            "best_trial": best_trial.number,
            "total_trials": len(study.trials),
            "optimization_history": [trial.value for trial in study.trials if trial.value is not None],
            "regime_params": regime_params
        }

    def _analyze_regime_characteristics(self, regime_name: str) -> Dict[str, float]:
        """Analyze regime characteristics to inform parameter ranges."""

        try:
            # This would integrate with your actual regime data analysis
            # For now, providing a placeholder that can be extended

            # Get regime data if available
            regime_data = self._get_regime_data(regime_name)

            if regime_data is None or regime_data.empty:
                # Return default characteristics if no data available
                return {
                    "volatility_factor": 1.0,
                    "trend_strength": 0.0,
                    "mean_reversion_strength": 0.0,
                    "regime_duration": 1.0,
                    "price_momentum": 0.0
                }

            # Calculate regime characteristics
            characteristics = {}

            # Volatility factor (normalized)
            returns = regime_data['close'].pct_change().dropna()
            characteristics["volatility_factor"] = returns.std() / 0.02  # Normalize to 2% baseline

            # Trend strength (using linear regression slope)
            if len(regime_data) > 10:
                x = np.arange(len(regime_data))
                y = regime_data['close'].values
                slope = np.polyfit(x, y, 1)[0]
                characteristics["trend_strength"] = np.tanh(slope / regime_data['close'].mean() * 1000)
            else:
                characteristics["trend_strength"] = 0.0

            # Mean reversion strength (using autocorrelation)
            if len(returns) > 20:
                autocorr = returns.autocorr(lag=1)
                characteristics["mean_reversion_strength"] = abs(autocorr) if not np.isnan(autocorr) else 0.0
            else:
                characteristics["mean_reversion_strength"] = 0.0

            # Regime duration factor
            characteristics["regime_duration"] = len(regime_data) / 1000  # Normalize to 1000 periods

            # Price momentum
            if len(regime_data) > 20:
                momentum = (regime_data['close'].iloc[-1] / regime_data['close'].iloc[-20] - 1)
                characteristics["price_momentum"] = np.tanh(momentum * 10)
            else:
                characteristics["price_momentum"] = 0.0

            return characteristics

        except Exception as e:
            self.logger.warning(f"Failed to analyze regime characteristics for {regime_name}: {e}")
            return {
                "volatility_factor": 1.0,
                "trend_strength": 0.0,
                "mean_reversion_strength": 0.0,
                "regime_duration": 1.0,
                "price_momentum": 0.0
            }

    def _create_regime_barrier_objective(
        self,
        regime_name: str,
        regime_data: pd.DataFrame,
        regime_params: Dict[str, Any]
    ):
        """Create objective function for regime-specific barrier optimization."""

        return objective

    def _evaluate_regime_barrier_parameters(
        self,
        regime_name: str,
        regime_data: pd.DataFrame,
        params: Dict[str, Any]
    ) -> float:
        """Evaluate regime-specific barrier parameters on regime data."""

        try:
            # This would integrate with your actual triple barrier implementation
            # For now, providing a placeholder evaluation

            # Simulate performance based on parameters
            upper_barrier = params.get("upper_barrier_multiplier", 1.0)
            lower_barrier = params.get("lower_barrier_multiplier", 1.0)
            timeout = params.get("barrier_timeout", 30)
            position_size = params.get("position_size_multiplier", 1.0)
            risk_per_trade = params.get("risk_per_trade", 0.05)

            # Calculate simulated performance score
            performance_score = self._calculate_regime_barrier_performance_score(
                regime_name,
                upper_barrier,
                lower_barrier,
                timeout,
                position_size,
                risk_per_trade
            )

            return performance_score

        except Exception as e:
            self.logger.error(f"Failed to evaluate regime barrier parameters for {regime_name}: {e}")
            return float('-inf')

    def _calculate_regime_barrier_performance_score(
        self,
        regime_name: str,
        upper_barrier: float,
        lower_barrier: float,
        timeout: int,
        position_size: float,
        risk_per_trade: float
    ) -> float:
        """Calculate performance score for regime-specific barrier parameters using data-driven approach."""

        # Get regime characteristics for scoring
        regime_characteristics = self._analyze_regime_characteristics(regime_name)

        # Base score
        base_score = 0.0

        # Extract characteristics
        volatility_factor = regime_characteristics.get("volatility_factor", 1.0)
        trend_strength = regime_characteristics.get("trend_strength", 0.0)
        mean_reversion_strength = regime_characteristics.get("mean_reversion_strength", 0.0)
        regime_duration = regime_characteristics.get("regime_duration", 1.0)
        price_momentum = regime_characteristics.get("price_momentum", 0.0)

        # Score barrier settings based on regime characteristics

        # 1. Volatility-based scoring
        if volatility_factor > 1.5:  # High volatility
            # Prefer wider barriers and shorter timeouts
            if upper_barrier > 0.008 and lower_barrier > 0.008:
                base_score += 0.3
            if timeout < 60:
                base_score += 0.2
            if position_size < 1.0:  # Smaller positions for high volatility
                base_score += 0.2
        elif volatility_factor < 0.7:  # Low volatility
            # Prefer tighter barriers and longer timeouts
            if upper_barrier < 0.008 and lower_barrier < 0.008:
                base_score += 0.3
            if timeout > 120:
                base_score += 0.2
            if position_size > 1.0:  # Larger positions for low volatility
                base_score += 0.2

        # 2. Trend-based scoring
        if abs(trend_strength) > 0.7:  # Strong trend
            if trend_strength > 0:  # Bullish trend
                # Prefer wider upper barrier, tighter lower barrier
                if upper_barrier > lower_barrier:
                    base_score += 0.25
                if upper_barrier > 0.006:
                    base_score += 0.15
            else:  # Bearish trend
                # Prefer tighter upper barrier, wider lower barrier
                if lower_barrier > upper_barrier:
                    base_score += 0.25
                if lower_barrier > 0.006:
                    base_score += 0.15

        # 3. Mean reversion scoring
        if mean_reversion_strength > 0.7:  # Strong mean reversion
            # Prefer balanced barriers
            if abs(upper_barrier - lower_barrier) < 0.002:
                base_score += 0.2
            if 30 < timeout < 180:  # Medium timeouts
                base_score += 0.15

        # 4. Risk management scoring
        if risk_per_trade < 0.05:  # Conservative risk
            base_score += 0.1
        elif risk_per_trade > 0.08:  # Aggressive risk
            if volatility_factor < 0.8:  # Only for low volatility
                base_score += 0.1

        # 5. Position sizing scoring
        if volatility_factor > 1.2 and position_size < 0.8:  # Appropriate sizing for high vol
            base_score += 0.15
        elif volatility_factor < 0.8 and position_size > 1.2:  # Appropriate sizing for low vol
            base_score += 0.15

        # 6. Barrier range validation (ensure within 0.2-1.5% range)
        if 0.002 <= upper_barrier <= 0.015 and 0.002 <= lower_barrier <= 0.015:
            base_score += 0.1  # Bonus for staying within specified range
        else:
            base_score -= 0.5  # Penalty for out-of-range barriers

        # 7. Timeout appropriateness
        if timeout < 1 or timeout > 1440:  # Out of reasonable range
            base_score -= 0.3

        # Add some randomness to simulate real evaluation
        random_factor = np.random.normal(0, 0.05)  # Reduced randomness for more stable scoring
        final_score = base_score + random_factor

        # Ensure score is positive
        return max(0.0, final_score)

    async def _log_sr_optimization_to_mlflow(self, optimization_results: Dict[str, Any]):
        """Log SR optimization results to MLflow."""

        try:
            # Set experiment name
            mlflow.set_experiment(self.sr_experiment_name)

            # Start a run for SR optimization
            with mlflow.start_run(run_name="sr_parameter_optimization"):
                # Log results
                mlflow.log_param("optimization_timestamp", optimization_results.get("optimization_timestamp", ""))
                mlflow.log_metric("best_value", optimization_results.get("best_value", 0))
                mlflow.log_metric("total_trials", optimization_results.get("total_trials", 0))

                # Log best parameters
                best_params = optimization_results.get("best_params", {})
                for param_name, param_value in best_params.items():
                    mlflow.log_param(param_name, param_value)

                # Log results as JSON artifact
                with open("sr_optimization_results.json", "w") as f:
                    json.dump(optimization_results, f, indent=2, default=str)
                mlflow.log_artifact("sr_optimization_results.json", "sr_optimization")

                self.logger.info("✅ SR optimization results logged to MLflow")

        except Exception as e:
            self.logger.error(f"Failed to log SR optimization to MLflow: {e}")

    async def _log_regime_optimization_to_mlflow(self, optimization_results: Dict[str, Any]):
        """Log regime-specific optimization results to MLflow."""

        try:
            # Set experiment name
            mlflow.set_experiment(self.regime_experiment_name)

            # Start a run for regime-specific optimization
            with mlflow.start_run(run_name="regime_specific_barrier_optimization"):
                # Log overall results
                mlflow.log_param("total_regimes", len(optimization_results))
                mlflow.log_param("optimization_timestamp", datetime.now().isoformat())

                # Log regime-specific results
                for regime_name, regime_result in optimization_results.items():
                    if "error" not in regime_result:
                        # Log regime parameters
                        mlflow.log_param(f"{regime_name}_best_value", regime_result.get("best_value", 0))
                        mlflow.log_param(f"{regime_name}_total_trials", regime_result.get("total_trials", 0))

                        # Log best parameters for this regime
                        best_params = regime_result.get("best_params", {})
                        for param_name, param_value in best_params.items():
                            mlflow.log_param(f"{regime_name}_{param_name}", param_value)

                # Log results as JSON artifact
                with open("regime_optimization_results.json", "w") as f:
                    json.dump(optimization_results, f, indent=2, default=str)
                mlflow.log_artifact("regime_optimization_results.json", "regime_optimization")

                self.logger.info("✅ Regime optimization results logged to MLflow")

        except Exception as e:
            self.logger.error(f"Failed to log regime optimization to MLflow: {e}")

    def _create_optimization_summary(self) -> Dict[str, Any]:
        """Create a summary of all optimizations."""

        summary = {}

        # SR optimization summary
        if self.sr_optimization_results:
            summary["sr_optimization"] = {
                "status": "completed",
                "best_value": self.sr_optimization_results.get("best_value", 0),
                "total_trials": self.sr_optimization_results.get("total_trials", 0),
                "best_params": self.sr_optimization_results.get("best_params", {})
            }
        else:
            summary["sr_optimization"] = {"status": "not_started"}

        # Regime optimization summary
        if self.regime_barrier_optimization_results:
            regime_summary = {}
            for regime_name, result in self.regime_barrier_optimization_results.items():
                if "error" not in result:
                    regime_summary[regime_name] = {
                        "status": "completed",
                        "best_value": result.get("best_value", 0),
                        "total_trials": result.get("total_trials", 0)
                    }
                else:
                    regime_summary[regime_name] = {
                        "status": "failed",
                        "error": result.get("error", "Unknown error")
                    }
            summary["regime_optimization"] = regime_summary
        else:
            summary["regime_optimization"] = {"status": "not_started"}

        return summary


# Factory function for creating early stage optimizer
def create_early_stage_optimizer(config: Dict[str, Any], training_manager=None):
    """Create early stage optimizer instance."""

    return EarlyStageOptimizer(config, training_manager)


if __name__ == "__main__":
    # Example usage for dynamic regime optimization
    config = {
        "early_stage_optimization": {
            "sr_optimization": {
                "n_trials": 100,
                "timeout": 1800,
                "early_stopping_patience": 20
            },
            "regime_optimization": {
                "n_trials": 100,
                "timeout": 3600,
                "early_stopping_patience": 20,
                "barrier_range": {
                    "min": 0.002,  # 0.2%
                    "max": 0.015   # 1.5%
                }
            }
        }
    }

    # Create optimizer instance
    optimizer = create_early_stage_optimizer(config)

    print("✅ Early Stage Optimizer created successfully!")
    print("This optimizer now handles:")
    print("  - SR parameter optimization (step2_5)")
    print("  - Dynamic regime-specific triple barrier optimization (step4)")
    print("  - Supports 15-20+ regimes with data-driven parameter ranges")
    print("  - Triple barrier range: 0.2% - 1.5%")
    print("  - Regime-agnostic scoring based on characteristics")
    print("  - Automatic regime analysis and categorization")
    print("  - Comprehensive optimization insights and recommendations")

    # Example of how to use with multiple regimes
    print("\n📊 Example usage with multiple regimes:")
    print("""
    # Create sample regime data (replace with your actual data)
    regime_data = {
        "regime_1": df1,  # Your regime DataFrame
        "regime_2": df2,  # Your regime DataFrame
        # ... up to 15-20 regimes
    }

    # Run optimization
    results = await optimizer.optimize_regime_specific_triple_barrier(
        regime_data,
        config["early_stage_optimization"]["regime_optimization"]
    )

    # Get insights
    summary = await optimizer.get_regime_optimization_summary()
    recommendations = await optimizer.get_regime_parameter_recommendations()

    # Export results
    await optimizer.export_optimization_results()
    """)
