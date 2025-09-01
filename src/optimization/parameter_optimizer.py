# src/optimization/parameter_optimizer.py

"""
Comprehensive Parameter Optimization Framework
Optimizes trading parameters across Strategist, Supervisor, Analyst, and Tactician components
using Optuna for Bayesian optimization.
"""

import optuna
import numpy as np
from datetime import datetime
import asyncio
import json

from src.utils.logger import system_logger


import class TradingParameterOptimizer:
class TradingParameterOptimizer:
    """
    Comprehensive parameter optimizer for trading system components.
    Uses Optuna for Bayesian optimization of critical trading parameters.
    """

    def __init__(self, config: Dict[str, Any]):
    pass
    pass
    pass
        self.config = config
        self.logger = system_logger.getChild("ParameterOptimizer")

        # Optimization configuration
        self.n_trials = config.get("optimization", {}).get("n_trials", 100)
        self.timeout = config.get("optimization", {}).get("timeout", 3600)  # 1 hour
        self.study_name = config.get("optimization", {}).get("study_name", "trading_parameters")

        # Performance tracking
        self.optimization_history = []
        self.best_params = {}
        self.best_score = float('-inf')

        # Parameter bounds and constraints
        self.parameter_bounds = self._define_parameter_bounds()

    def _define_parameter_bounds(self) -> Dict[str, Dict[str, Any]]:
    pass
    pass
    pass
        """Define parameter bounds and constraints for optimization."""
        return {
            # Strategist Parameters
            "min_confidence_threshold": {"low": 0.3, "high": 0.9, "type": "float"},
            "entry_threshold": {"low": 0.5, "high": 0.9, "type": "float"},
            "max_confidence_threshold": {"low": 0.8, "high": 0.99, "type": "float"},

            # Technical Indicator Thresholds
            "rsi_oversold": {"low": 10.0, "high": 40.0, "type": "float"},
            "rsi_overbought": {"low": 60.0, "high": 90.0, "type": "float"},
            "sma_fast_window": {"low": 5, "high": 50, "type": "int"},
            "sma_slow_window": {"low": 20, "high": 200, "type": "int"},
            "volume_ratio_high": {"low": 1.0, "high": 3.0, "type": "float"},
            "volume_ratio_low": {"low": 0.2, "high": 1.0, "type": "float"},
            "price_volatility_window": {"low": 10, "high": 100, "type": "int"},

            # Strategy type selection
            "strategy_type": {"choices": ["trend_following", "mean_reversion", "breakout"], "type": "categorical"},

            # Supervisor Parameters
            "min_weight": {"low": 0.05, "high": 0.3, "type": "float"},
            "max_weight": {"low": 0.6, "high": 0.95, "type": "float"},

            # Analyst Parameters
            "analyst_confidence_threshold": {"low": 0.3, "high": 0.8, "type": "float"},
            "tactician_confidence_threshold": {"low": 0.4, "high": 0.9, "type": "float"},

            # Position Sizing Parameters (retain only requested)
            "kelly_multiplier": {"low": 0.1, "high": 0.5, "type": "float"},
            "max_position_size": {"low": 0.2, "high": 0.8, "type": "float"},
            "min_position_size": {"low": 0.005, "high": 0.05, "type": "float"},
            "position_confidence_threshold": {"low": 0.4, "high": 0.8, "type": "float"},
            "positionsize_combined_threshold": {"low": 0.5, "high": 0.9, "type": "float"},

            # Leverage Sizing Parameters (unchanged, per request)
            "min_leverage": {"low": 5.0, "high": 20.0, "type": "float"},
            "max_leverage": {"low": 50.0, "high": 150.0, "type": "float"},
            "leverage_confidence_threshold": {"low": 0.4, "high": 0.8, "type": "float"},
            "liquidation_buffer": {"low": 0.02, "high": 0.1, "type": "float"},
            "leverage_combined_threshold": {"low": 0.6, "high": 0.9, "type": "float"},
            "leverage_ml_weight": {"low": 0.4, "high": 0.8, "type": "float"},
            "liquidation_weight": {"low": 0.2, "high": 0.6, "type": "float"},
            "leverage_multiplier": {"low": 0.5, "high": 2.0, "type": "float"},
            "leverage_risk_adjustment": {"low": 0.5, "high": 1.5, "type": "float"},
            "max_risk_leverage": {"low": 25.0, "high": 75.0, "type": "float"},
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="parameter optimization"
    )
    async def optimize_parameters(self) -> Dict[str, Any]:
        """
        Run comprehensive parameter optimization using Optuna.

        Returns:
            Dict containing optimized parameters and performance metrics
        """
        self.logger.info("🚀 Starting comprehensive parameter optimization...")

        # Create Optuna study
        study = optuna.create_study(
            direction="maximize",
            study_name=self.study_name,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )

        # Define objective function
        def objective(trial):
    pass
    pass
    pass
            return asyncio.run(self._evaluate_parameters(trial))

        # Run optimization
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )

        # Store results
        self.best_params = study.best_params
        self.best_score = study.best_value

        # Generate optimization report
        optimization_report = self._generate_optimization_report(study)

        self.logger.info(f"✅ Optimization completed. Best score: {self.best_score:.4f}")

        return {
            "optimized_parameters": self.best_params,
            "best_score": self.best_score,
            "optimization_report": optimization_report,
            "study": study
        }

    async def _evaluate_parameters(self, trial: optuna.Trial) -> float:
        """
        Evaluate a set of parameters using backtesting simulation.

        Args:
            trial: Optuna trial object

        Returns:
            float: Performance score (higher is better)
        """
        # Suggest parameters for this trial
        params = self._suggest_parameters(trial)

        # Validate parameter constraints
        if not self._validate_parameter_constraints(params):
    pass
    pass
    pass
            return float('-inf')

        # Simulate trading performance with these parameters
        performance_score = await self._simulate_trading_performance(params)

        # Store trial results
        self.optimization_history.append({
            "trial_number": trial.number,
            "parameters": params,
            "score": performance_score,
            "timestamp": datetime.now().isoformat()
        })

        return performance_score

    def _suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
    pass
    pass
    pass
        """Suggest parameters for the current trial."""
        params = {}

        for param_name, bounds in self.parameter_bounds.items():
    pass
    pass
    pass
            if bounds["type"] == "float":
    pass
    pass
    pass
                params[param_name] = trial.suggest_float(
                    param_name,
                    bounds["low"],
                    bounds["high"]
                )
            elif bounds["type"] == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    bounds["low"],
                    bounds["high"]
                )
            elif bounds["type"] == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    bounds["choices"]
                )

        return params

    def _validate_parameter_constraints(self, params: Dict[str, Any]) -> bool:
    pass
    pass
    pass
        """Validate parameter constraints and relationships."""
        try:
            # Basic range validation
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
            for param_name, value in params.items():
    pass
    pass
    pass
                if param_name in self.parameter_bounds:
    pass
    pass
    pass
                    bounds = self.parameter_bounds[param_name]
                    if "low" in bounds and "high" in bounds:
    pass
    pass
    pass
                        if not (bounds["low"] <= value <= bounds["high"]):
    pass
    pass
    pass
                            return False

            # Relationship constraints
            if params["min_weight"] >= params["max_weight"]:
    pass
    pass
    pass
                return False
            if params["min_leverage"] >= params["max_leverage"]:
    pass
    pass
    pass
                return False
            if params["min_position_size"] >= params["max_position_size"]:
    pass
    pass
    pass
                return False
            if params["entry_threshold"] >= params["max_confidence_threshold"]:
    pass
    pass
    pass
                return False
            # Technical indicators relationships
            if params["sma_fast_window"] >= params["sma_slow_window"]:
    pass
    pass
    pass
                return False
            if params["rsi_oversold"] >= params["rsi_overbought"]:
    pass
    pass
    pass
                return False
            if params["volume_ratio_low"] >= params["volume_ratio_high"]:
    pass
    pass
    pass
                return False

            # Leverage weights sum constraint (position weights removed by request)
            if params["leverage_ml_weight"] + params["liquidation_weight"] > 1.0:
    pass
    pass
    pass
                return False

            return True

        except Exception as e:
            self.logger.error(f"Parameter validation error: {e}")
            return False

    async def _simulate_trading_performance(self, params: Dict[str, Any]) -> float:
        """
        Simulate trading performance with given parameters using backtesting.

        Args:
            params: Parameter dictionary

        Returns:
            float: Performance score (Sharpe ratio, profit factor, etc.)
        """
        try:
            # Import backtesting evaluator
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
            from src.optimization.backtesting_evaluator import evaluate_parameters_with_backtesting

            # Use backtesting evaluator for realistic performance simulation
import score = await evaluate_parameters_with_backtesting
            score = await evaluate_parameters_with_backtesting(params, self.config)

            return score

        except Exception as e:
            self.logger.error(f"Performance simulation error: {e}")
            return 0.0

    def _generate_optimization_report(self, study: optuna.Study) -> Dict[str, Any]:
    pass
    pass
    pass
        """Generate comprehensive optimization report."""
        return {
            "optimization_summary": {
                "total_trials": len(study.trials),
                "best_score": study.best_value,
                "best_trial_number": study.best_trial.number,
                "optimization_duration": study.duration,
                "parameter_importance": self._calculate_parameter_importance(study)
            },
            "best_parameters": study.best_params,
            "parameter_analysis": self._analyze_parameter_distributions(study),
            "convergence_analysis": self._analyze_convergence(study)
        }

    def _calculate_parameter_importance(self, study: optuna.Study) -> Dict[str, float]:
    pass
    pass
    pass
        """Calculate parameter importance using Optuna's built-in method."""
        try:
            importance = optuna.importance.get_param_importances(study)
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
            return importance
        except Exception as e:
            self.logger.warning(f"Could not calculate parameter importance: {e}")
            return {}

    def _analyze_parameter_distributions(self, study: optuna.Study) -> Dict[str, Any]:
    pass
    pass
    pass
        """Analyze parameter value distributions across trials."""
        analysis = {}

        for param_name in self.parameter_bounds.keys():
    pass
    pass
    pass
            values = [trial.params.get(param_name) for trial in study.trials if param_name in trial.params]
            if values:
    pass
    pass
    pass
                analysis[param_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "median": np.median(values)
                }

        return analysis

    def _analyze_convergence(self, study: optuna.Study) -> Dict[str, Any]:
    pass
    pass
    pass
        """Analyze optimization convergence."""
        values = [trial.value for trial in study.trials if trial.value is not None]

        if len(values) < 2:
    pass
    pass
    pass
            return {"converged": False, "reason": "Insufficient trials"}

        # Check if optimization has converged
        recent_values = values[-10:]  # Last 10 trials
        if len(recent_values) >= 5:
    pass
    pass
    pass
            recent_std = np.std(recent_values)
            recent_mean = np.mean(recent_values)
            cv = recent_std / recent_mean if recent_mean != 0 else float('inf')

            converged = cv < 0.05  # 5% coefficient of variation threshold
        else:
            converged = False

        return {
            "converged": converged,
            "total_trials": len(values),
            "best_value": max(values),
            "worst_value": min(values),
            "value_range": max(values) - min(values),
            "recent_improvement": values[-1] - values[0] if len(values) > 1 else 0
        }

    def save_optimization_results(self, output_path: str = "optimization_results.json"):
    pass
    pass
    pass
        """Save optimization results to file."""
        try:
            results = {
                "optimization_timestamp": datetime.now().isoformat(),
                "best_parameters": self.best_params,
                "best_score": self.best_score,
                "optimization_history": self.optimization_history,
                "parameter_bounds": self.parameter_bounds
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
            }

            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)

            self.logger.info(f"✅ Optimization results saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to save optimization results: {e}")

    def generate_config_update(self) -> Dict[str, Any]:
    pass
    pass
    pass
        """Generate configuration update with optimized parameters."""
        return {
            "strategist": {
                "min_confidence_threshold": self.best_params.get("min_confidence_threshold", 0.6),
                "technical_indicator_thresholds": {
                    "rsi_oversold": self.best_params.get("rsi_oversold", 30.0),
                    "rsi_overbought": self.best_params.get("rsi_overbought", 70.0),
                    "sma_fast_window": self.best_params.get("sma_fast_window", 20),
                    "sma_slow_window": self.best_params.get("sma_slow_window", 50),
                    "volume_ratio_high": self.best_params.get("volume_ratio_high", 1.5),
                    "volume_ratio_low": self.best_params.get("volume_ratio_low", 0.5),
                    "price_volatility_window": self.best_params.get("price_volatility_window", 20)
                },
                "strategy_type": self.best_params.get("strategy_type", "trend_following")
            },
            "supervisor": {
                "enhanced_prediction_service": {
                    "entry_threshold": self.best_params.get("entry_threshold", 0.7),
                    "max_confidence_threshold": self.best_params.get("max_confidence_threshold", 0.9)
                },
                "online_learning": {
                    "min_weight": self.best_params.get("min_weight", 0.1),
                    "max_weight": self.best_params.get("max_weight", 0.8)
                }
            },
            "dual_model_system": {
                "analyst_confidence_threshold": self.best_params.get("analyst_confidence_threshold", 0.5),
                "tactician_confidence_threshold": self.best_params.get("tactician_confidence_threshold", 0.6)
            },
            "step17_optimization": {
                "position_sizing": {
                    "kelly_multiplier": self.best_params.get("kelly_multiplier", 0.25),
                    "max_position_size": self.best_params.get("max_position_size", 0.5),
                    "min_position_size": self.best_params.get("min_position_size", 0.01),
                    "confidence_threshold": self.best_params.get("position_confidence_threshold", 0.6),
                    "positionsize_combined_threshold": self.best_params.get("positionsize_combined_threshold", 0.7)
                },
                "leverage": {
                    "min_leverage": self.best_params.get("min_leverage", 10.0),
                    "max_leverage": self.best_params.get("max_leverage", 100.0),
                    "confidence_threshold": self.best_params.get("leverage_confidence_threshold", 0.6),
                    "liquidation_buffer": self.best_params.get("liquidation_buffer", 0.05),
                    "leverage_combined_threshold": self.best_params.get("leverage_combined_threshold", 0.75),
                    "ml_weight": self.best_params.get("leverage_ml_weight", 0.6),
                    "liquidation_weight": self.best_params.get("liquidation_weight", 0.4),
                    "leverage_multiplier": self.best_params.get("leverage_multiplier", 1.0),
                    "risk_adjustment_factor": self.best_params.get("leverage_risk_adjustment", 1.0),
                    "max_risk_leverage": self.best_params.get("max_risk_leverage", 50.0)
                }
            }
        }


async def run_parameter_optimization(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to run parameter optimization.

    Args:
        config: Configuration dictionary

    Returns:
        Dict containing optimization results
    """
    optimizer = TradingParameterOptimizer(config)

    # Run optimization
    results = await optimizer.optimize_parameters()

    # Save results
    optimizer.save_optimization_results()

    # Generate config update
    config_update = optimizer.generate_config_update()

    return {
        "optimization_results": results,
        "config_update": config_update,
        "optimizer": optimizer
    }


if __name__ == "__main__":
    pass
    pass
    pass
    # Example configuration
    config = {
        "optimization": {
            "n_trials": 50,
            "timeout": 1800,
            "study_name": "trading_parameters_v1"
        }
    }

    # Run optimization
    asyncio.run(run_parameter_optimization(config))