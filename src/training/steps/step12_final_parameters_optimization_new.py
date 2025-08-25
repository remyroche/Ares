# src/training/steps/step12_final_parameters_optimization_new.py

import asyncio
import contextlib
import json
import os
import pickle
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd
import optuna

from src.config.config_manager import (
    get_config_manager,
    get_optimizable_parameters,
    get_search_space,
    update_optimizable_config,
)
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    failed,
    missing,
)


class FinalParametersOptimizationStepNew:
    """Step 12: Final Parameters Optimization using new categorized configuration structure."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger
        self.config_manager = get_config_manager()
        self.optimizable_params = get_optimizable_parameters()

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="final parameters optimization step initialization",
    )
    async def initialize(self) -> None:
        """Initialize the final parameters optimization step."""
        self.logger.info("🚀 Initializing Final Parameters Optimization Step (New)...")

        # Validate configuration
        is_valid, errors = self.config_manager.validate_config()
        if not is_valid:
            self.logger.error(f"Configuration validation failed: {errors}")
            raise ValueError("Configuration validation failed")

        # Initialize optimization storage
        self._setup_optimization_storage()

        self.logger.info(
            "✅ Final Parameters Optimization Step initialized successfully",
        )

    @handle_errors(
        exceptions=(Exception,),
        default_return={"status": "FAILED", "error": "Execution failed"},
        context="final parameters optimization step execution",
    )
    async def execute(
        self, training_input: dict[str, Any], pipeline_state: dict[str, Any], ) -> dict[str, Any]:
        """Execute final parameters optimization with categorized parameters.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Dict containing optimization results

        """
        try:
            self.logger.info("🔄 Executing Final Parameters Optimization (New)...")
            start_time = datetime.now()

            # Extract parameters
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            data_dir = training_input.get("data_dir", "data/training")

            # Load calibration results
            from src.utils.logger import heartbeat

            with heartbeat(
                self.logger,
                name="Step12 load_calibration_results",
                interval_seconds=60.0,
            ):
                calibration_results = await self._load_calibration_results(
                    symbol,
                    exchange,
                    data_dir,
                )
            if not calibration_results:
                msg = "Calibration results not found"
                raise FileNotFoundError(msg)

            # Load previous optimization results for warm start
            with heartbeat(
                self.logger,
                name="Step12 load_previous_optimization",
                interval_seconds=60.0,
            ):
                previous_results = await self._load_previous_optimization_results(
                    symbol,
                    exchange,
                    data_dir,
                )

            # Perform categorized parameter optimization
            with heartbeat(
                self.logger,
                name="Step12 optimize_all_parameters",
                interval_seconds=60.0,
            ):
                optimization_results = await self._optimize_all_parameters_categorized(
                    calibration_results,
                    previous_results,
                )

            # Validate optimization results
            with heartbeat(
                self.logger,
                name="Step12 validate_optimization",
                interval_seconds=60.0,
            ):
                validation_passed = await self._validate_optimization_results(
                    optimization_results,
                )
            if not validation_passed:
                self.logger.warning(
                    "⚠️ Optimization results validation failed, using fallback parameters",
                )

            # Save optimization results
            with heartbeat(
                self.logger,
                name="Step12 save_results",
                interval_seconds=60.0,
            ):
                await self._save_optimization_results(
                    optimization_results,
                    symbol,
                    exchange,
                    data_dir,
                )

            # Generate optimization report
            with heartbeat(
                self.logger,
                name="Step12 generate_report",
                interval_seconds=60.0,
            ):
                report = await self._generate_optimization_report(
                    optimization_results,
                    start_time,
                )

            # Update pipeline state
            pipeline_state["final_parameters"] = optimization_results
            pipeline_state["optimization_report"] = report

            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(
                f"✅ Final parameters optimization completed in {duration:.2f}s",
            )

            return {
                "final_parameters": optimization_results,
                "optimization_report": report,
                "duration": duration,
                "status": "SUCCESS",
            }

        except Exception as e:
            self.logger.error(f"❌ Error in Final Parameters Optimization: {e}")
            return {"status": "FAILED", "error": str(e), "duration": 0.0}

    async def _optimize_all_parameters_categorized(
        self, calibration_results: dict[str, Any], previous_results: dict[str, Any] | None, ) -> dict[str, Any]:
        """Optimize all parameters by category using the new configuration structure.

        Args:
            calibration_results: Results from confidence calibration
            previous_results: Previous optimization results for warm start

        Returns:
            Dict containing optimized parameters by category

        """
        try:
            self.logger.info("Optimizing all parameters by category...")

            optimization_results = {}
            categories = [
                "confidence", 
                "position_sizing", 
                "leverage", 
                "tpsl", 
                "ensemble", 
                "sr", 
                "two_tier",
                "technical_indicators",
                "system_monitoring",
                "training_optimization",
                "strategy_selection"
            ]

            for category in categories:
                self.logger.info(f"Optimizing {category} parameters...")
                
                category_results = await self._optimize_category(
                    category,
                    calibration_results,
                    previous_results.get(category) if previous_results else None,
                )
                
                optimization_results[category] = category_results
                
                # Update the configuration with optimized parameters
                if category_results and "best_params" in category_results:
                    update_optimizable_config(category, category_results["best_params"])

            return optimization_results

        except Exception as e:
            self.logger.error(f"Error in categorized optimization: {e}")
            raise

    async def _optimize_category(
        self, category: str, calibration_results: dict[str, Any], previous_results: dict[str, Any] | None, ) -> dict[str, Any]:
        """Optimize parameters for a specific category.

        Args:
            category: Parameter category to optimize
            calibration_results: Results from confidence calibration
            previous_results: Previous optimization results for this category

        Returns:
            Dict containing optimization results for the category

        """
        try:
            # Get search space for this category
            search_space = get_search_space(category)
            if not search_space:
                self.logger.warning(f"No search space found for category: {category}")
                return {}

            # Create Optuna study
            study_name = f"step12_{category}_optimization"
            study = optuna.create_study(
                study_name=study_name,
                direction="maximize",
                storage="sqlite:///optuna_studies.db",
                load_if_exists=True,
            )

            # Define objective function for this category
            def objective(trial):
                return self._objective_function(
                    trial,
                    category,
                    search_space,
                    calibration_results,
                )

            # Run optimization
            n_trials = 50  # Adjust based on category complexity
            study.optimize(objective, n_trials=n_trials, timeout=300)  # 5 minutes timeout

            # Get best parameters
            best_params = study.best_params
            best_value = study.best_value

            return {
                "best_params": best_params,
                "best_value": best_value,
                "study_name": study_name,
                "n_trials": n_trials,
            }

        except Exception as e:
            self.logger.error(f"Error optimizing category {category}: {e}")
            return {}

    def _objective_function(
        self, trial: optuna.Trial, category: str, search_space: dict[str, dict[str, Any]], calibration_results: dict[str, Any], ) -> float:
        """Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object
            category: Parameter category being optimized
            search_space: Search space for the category
            calibration_results: Results from confidence calibration

        Returns:
            Optimization score (higher is better)

        """
        try:
            # Suggest parameters based on search space
            params = {}
            for param_name, param_config in search_space.items():
                if param_config["type"] == "float":
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config["min"],
                        param_config["max"],
                    )
                elif param_config["type"] == "int":
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config["min"],
                        param_config["max"],
                    )

            # Update configuration with suggested parameters
            update_optimizable_config(category, params)

            # Evaluate the configuration
            score = self._evaluate_configuration(category, params, calibration_results)

            return score

        except Exception as e:
            self.logger.error(f"Error in objective function for {category}: {e}")
            return -999.0  # Return very low score on error

    def _evaluate_configuration(
        self, category: str, params: dict[str, Any], calibration_results: dict[str, Any], ) -> float:
        """Evaluate a configuration by running a backtest or simulation.

        Args:
            category: Parameter category being evaluated
            params: Parameters to evaluate
            calibration_results: Results from confidence calibration

        Returns:
            Evaluation score (higher is better)

        """
        try:
            # This is a simplified evaluation - in practice, you would run a full backtest
            # For now, we'll use a simple scoring based on parameter ranges and calibration results
            
            base_score = 0.0
            
            if category == "confidence":
                # Higher confidence thresholds generally lead to better precision
                base_score = self._evaluate_confidence_params(params, calibration_results)
            elif category == "position_sizing":
                # Balanced position sizing parameters
                base_score = self._evaluate_position_sizing_params(params, calibration_results)
            elif category == "leverage":
                # Conservative leverage parameters
                base_score = self._evaluate_leverage_params(params, calibration_results)
            elif category == "tpsl":
                # Risk-reward balanced TP/SL parameters
                base_score = self._evaluate_tpsl_params(params, calibration_results)
            elif category == "ensemble":
                # Ensemble diversity and agreement
                base_score = self._evaluate_ensemble_params(params, calibration_results)
            elif category == "sr":
                # S/R strength and accuracy
                base_score = self._evaluate_sr_params(params, calibration_results)
            elif category == "two_tier":
                # Two-tier system parameters
                base_score = self._evaluate_two_tier_params(params, calibration_results)
            elif category == "technical_indicators":
                # Technical indicator parameters
                base_score = self._evaluate_technical_indicators_params(params, calibration_results)
            elif category == "system_monitoring":
                # System monitoring parameters
                base_score = self._evaluate_system_monitoring_params(params, calibration_results)
            elif category == "training_optimization":
                # Training optimization parameters
                base_score = self._evaluate_training_optimization_params(params, calibration_results)
            elif category == "strategy_selection":
                # Strategy selection parameters
                base_score = self._evaluate_strategy_selection_params(params, calibration_results)
            
            return base_score

        except Exception as e:
            self.logger.error(f"Error evaluating configuration for {category}: {e}")
            return 0.0

    def _evaluate_confidence_params(self, params: dict[str, Any], calibration_results: dict[str, Any]) -> float:
        """Evaluate confidence threshold parameters."""
        score = 0.0
        
        # Higher base entry threshold is generally better (but not too high)
        if "base_entry_threshold" in params:
            threshold = params["base_entry_threshold"]
            if 0.6 <= threshold <= 0.8:
                score += 0.3
            elif 0.5 <= threshold <= 0.9:
                score += 0.2
            else:
                score += 0.1
        
        # Analyst vs tactician threshold balance
        if "analyst_confidence_threshold" in params and "tactician_confidence_threshold" in params:
            analyst_thresh = params["analyst_confidence_threshold"]
            tactician_thresh = params["tactician_confidence_threshold"]
            
            if tactician_thresh > analyst_thresh:
                score += 0.2
            if 0.1 <= (tactician_thresh - analyst_thresh) <= 0.2:
                score += 0.1
        
        return score

    def _evaluate_position_sizing_params(self, params: dict[str, Any], calibration_results: dict[str, Any]) -> float:
        """Evaluate position sizing parameters."""
        score = 0.0
        
        # Reasonable position size ranges
        if "base_position_size" in params:
            base_size = params["base_position_size"]
            if 0.02 <= base_size <= 0.1:
                score += 0.3
            elif 0.01 <= base_size <= 0.15:
                score += 0.2
            else:
                score += 0.1
        
        # Risk management
        if "max_position_size" in params:
            max_size = params["max_position_size"]
            if 0.15 <= max_size <= 0.3:
                score += 0.2
            else:
                score += 0.1
        
        return score

    def _evaluate_leverage_params(self, params: dict[str, Any], calibration_results: dict[str, Any]) -> float:
        """Evaluate leverage parameters."""
        score = 0.0
        
        # Conservative leverage settings
        if "safe_leverage_multiplier" in params:
            multiplier = params["safe_leverage_multiplier"]
            if 0.7 <= multiplier <= 0.9:
                score += 0.3
            elif 0.5 <= multiplier <= 1.0:
                score += 0.2
            else:
                score += 0.1
        
        return score

    def _evaluate_tpsl_params(self, params: dict[str, Any], calibration_results: dict[str, Any]) -> float:
        """Evaluate TP/SL parameters."""
        score = 0.0
        
        # Risk-reward ratio
        if "tp_long" in params and "sl_long" in params:
            tp = params["tp_long"]
            sl = params["sl_long"]
            if tp > sl and tp / sl >= 1.5:
                score += 0.3
            elif tp > sl:
                score += 0.2
            else:
                score += 0.1
        
        return score

    def _evaluate_ensemble_params(self, params: dict[str, Any], calibration_results: dict[str, Any]) -> float:
        """Evaluate ensemble parameters."""
        score = 0.0
        
        # Weight balance
        if "analyst_weight" in params and "tactician_weight" in params and "strategist_weight" in params:
            weights = [params["analyst_weight"], params["tactician_weight"], params["strategist_weight"]]
            if abs(sum(weights) - 1.0) < 0.1:
                score += 0.3
            else:
                score += 0.1
        
        return score

    def _evaluate_sr_params(self, params: dict[str, Any], calibration_results: dict[str, Any]) -> float:
        """Evaluate S/R parameters."""
        score = 0.0
        
        # Strength score weights should sum to 1.0
        weight_params = ["touch_count_weight", "total_volume_weight", "level_age_weight", 
                        "bounce_rate_weight", "isolation_score_weight"]
        weights = [params.get(param, 0.0) for param in weight_params]
        
        if abs(sum(weights) - 1.0) < 0.1:
            score += 0.3
        else:
            score += 0.1
        
        return score

    def _evaluate_two_tier_params(self, params: dict[str, Any], calibration_results: dict[str, Any]) -> float:
        """Evaluate two-tier system parameters."""
        score = 0.0
        
        # Tier weights should sum to 1.0
        if "tier1_weight" in params and "tier2_weight" in params:
            tier1_weight = params["tier1_weight"]
            tier2_weight = params["tier2_weight"]
            if abs((tier1_weight + tier2_weight) - 1.0) < 0.1:
                score += 0.3
            else:
                score += 0.1
        
        # Reasonable thresholds
        if "direction_threshold" in params:
            threshold = params["direction_threshold"]
            if 0.6 <= threshold <= 0.8:
                score += 0.2
            else:
                score += 0.1
        
        if "timing_threshold" in params:
            threshold = params["timing_threshold"]
            if 0.7 <= threshold <= 0.9:
                score += 0.2
            else:
                score += 0.1
        
        return score

    def _evaluate_technical_indicators_params(self, params: dict[str, Any], calibration_results: dict[str, Any]) -> float:
        """Evaluate technical indicator parameters."""
        score = 0.0
        
        # RSI parameters
        if "rsi_period" in params:
            rsi_period = params["rsi_period"]
            if 10 <= rsi_period <= 20:
                score += 0.2
            else:
                score += 0.1
        
        # MACD parameters
        if "macd_fast_period" in params and "macd_slow_period" in params:
            fast = params["macd_fast_period"]
            slow = params["macd_slow_period"]
            if fast < slow and 8 <= fast <= 16 and 20 <= slow <= 30:
                score += 0.2
            else:
                score += 0.1
        
        # ADX parameters
        if "adx_trend_threshold" in params and "adx_sideways_threshold" in params:
            trend = params["adx_trend_threshold"]
            sideways = params["adx_sideways_threshold"]
            if trend > sideways:
                score += 0.2
            else:
                score += 0.1
        
        # Volatility parameters
        if "volatility_threshold" in params:
            vol_thresh = params["volatility_threshold"]
            if 0.015 <= vol_thresh <= 0.035:
                score += 0.2
            else:
                score += 0.1
        
        return score

    def _evaluate_system_monitoring_params(self, params: dict[str, Any], calibration_results: dict[str, Any]) -> float:
        """Evaluate system monitoring parameters."""
        score = 0.0
        
        # Monitoring intervals should be reasonable
        if "analysis_interval" in params:
            interval = params["analysis_interval"]
            if 1800 <= interval <= 7200:  # 30 minutes to 2 hours
                score += 0.2
            else:
                score += 0.1
        
        # History limits should be reasonable
        if "max_history" in params:
            max_hist = params["max_history"]
            if 50 <= max_hist <= 200:
                score += 0.2
            else:
                score += 0.1
        
        # System performance parameters
        if "memory_threshold" in params:
            mem_thresh = params["memory_threshold"]
            if 0.7 <= mem_thresh <= 0.9:
                score += 0.2
            else:
                score += 0.1
        
        # Learning rate should be reasonable
        if "learning_rate" in params:
            lr = params["learning_rate"]
            if 0.005 <= lr <= 0.05:
                score += 0.2
            else:
                score += 0.1
        
        return score

    def _evaluate_training_optimization_params(self, params: dict[str, Any], calibration_results: dict[str, Any]) -> float:
        """Evaluate training optimization parameters."""
        score = 0.0
        
        # Step 2: Market Regime Classification
        if "adx_trend_threshold" in params and "adx_sideways_threshold" in params:
            trend = params["adx_trend_threshold"]
            sideways = params["adx_sideways_threshold"]
            if trend > sideways and 20.0 <= trend <= 35.0 and 15.0 <= sideways <= 30.0:
                score += 0.2
            else:
                score += 0.1
        
        # Step 4: Processing & Labeling
        if "min_label_balance" in params and "max_label_balance" in params:
            min_balance = params["min_label_balance"]
            max_balance = params["max_label_balance"]
            if min_balance < max_balance and 0.03 <= min_balance <= 0.1 and 0.9 <= max_balance <= 0.98:
                score += 0.2
            else:
                score += 0.1
        
        # Step 6: Analyst Enhancement
        if "stability_threshold" in params:
            stability = params["stability_threshold"]
            if 0.6 <= stability <= 0.9:
                score += 0.2
            else:
                score += 0.1
        
        # Model hyperparameters
        if "lgb_learning_rate" in params:
            lr = params["lgb_learning_rate"]
            if 0.01 <= lr <= 0.2:
                score += 0.2
            else:
                score += 0.1
        
        # Performance thresholds
        if "model_performance_threshold" in params:
            perf_thresh = params["model_performance_threshold"]
            if 0.6 <= perf_thresh <= 0.85:
                score += 0.2
            else:
                score += 0.1
        
        return score

    def _evaluate_strategy_selection_params(self, params: dict[str, Any], calibration_results: dict[str, Any]) -> float:
        """Evaluate strategy selection parameters."""
        score = 0.0
        
        # Strategy selection thresholds
        if "momentum_selection_threshold" in params:
            threshold = params["momentum_selection_threshold"]
            if 0.6 <= threshold <= 0.85:
                score += 0.2
            else:
                score += 0.1
        
        # Model selection criteria
        if "min_model_confidence" in params:
            min_confidence = params["min_model_confidence"]
            if 0.5 <= min_confidence <= 0.75:
                score += 0.2
            else:
                score += 0.1
        
        # Model diversity requirements
        if "min_model_diversity" in params:
            diversity = params["min_model_diversity"]
            if 0.2 <= diversity <= 0.5:
                score += 0.2
            else:
                score += 0.1
        
        # Strategy switching thresholds
        if "strategy_switch_threshold" in params:
            switch_thresh = params["strategy_switch_threshold"]
            if 0.1 <= switch_thresh <= 0.3:
                score += 0.2
            else:
                score += 0.1
        
        # Performance-based selection
        if "min_performance_threshold" in params:
            perf_thresh = params["min_performance_threshold"]
            if 0.4 <= perf_thresh <= 0.6:
                score += 0.2
            else:
                score += 0.1
        
        return score

    async def _load_calibration_results(
        self, symbol: str, exchange: str, data_dir: str, ) -> dict[str, Any] | None:
        """Load calibration results from previous step."""
        try:
            calibration_dir = f"{data_dir}/calibration_results"
            calibration_file = f"{calibration_dir}/{exchange}_{symbol}_calibration_results.pkl"

            if not os.path.exists(calibration_file):
                self.logger.warning(f"Calibration file not found: {calibration_file}")
                return {}

            with open(calibration_file, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            self.logger.error(f"Error loading calibration results: {e}")
            return {}

    async def _load_previous_optimization_results(
        self, symbol: str, exchange: str, data_dir: str, ) -> dict[str, Any] | None:
        """Load previous optimization results for warm start."""
        try:
            optimization_dir = f"{data_dir}/optimization_results"
            previous_file = f"{optimization_dir}/{exchange}_{symbol}_final_parameters_new.pkl"

            if os.path.exists(previous_file):
                with open(previous_file, "rb") as f:
                    return pickle.load(f)
            return None
        except Exception as e:
            self.logger.error(f"Error loading previous optimization results: {e}")
            return None

    async def _validate_optimization_results(self, optimization_results: dict[str, Any]) -> bool:
        """Validate optimization results."""
        try:
            if not optimization_results:
                return False

            # Check that all categories have results
            expected_categories = [
                "confidence", 
                "position_sizing", 
                "leverage", 
                "tpsl", 
                "ensemble", 
                "sr", 
                "two_tier",
                "technical_indicators",
                "system_monitoring",
                "training_optimization",
                "strategy_selection"
            ]
            for category in expected_categories:
                if category not in optimization_results:
                    self.logger.warning(f"Missing optimization results for category: {category}")
                    return False

            return True
        except Exception as e:
            self.logger.error(f"Error validating optimization results: {e}")
            return False

    async def _save_optimization_results(
        self, optimization_results: dict[str, Any], symbol: str, exchange: str, data_dir: str, ) -> None:
        """Save optimization results."""
        try:
            optimization_dir = f"{data_dir}/optimization_results"
            os.makedirs(optimization_dir, exist_ok=True)

            results_file = f"{optimization_dir}/{exchange}_{symbol}_final_parameters_new.pkl"
            with open(results_file, "wb") as f:
                pickle.dump(optimization_results, f)

            # Also save as JSON for human readability
            json_file = f"{optimization_dir}/{exchange}_{symbol}_final_parameters_new.json"
            with open(json_file, "w") as f:
                json.dump(optimization_results, f, indent=2, default=str)

            self.logger.info(f"Optimization results saved to {results_file}")
        except Exception as e:
            self.logger.error(f"Error saving optimization results: {e}")

    async def _generate_optimization_report(
        self, optimization_results: dict[str, Any], start_time: datetime, ) -> dict[str, Any]:
        """Generate optimization report."""
        try:
            report = {
                "optimization_timestamp": start_time.isoformat(),
                "duration_seconds": (datetime.now() - start_time).total_seconds(),
                "categories_optimized": list(optimization_results.keys()),
                "summary": {},
            }

            for category, results in optimization_results.items():
                if results and "best_value" in results:
                    report["summary"][category] = {
                        "best_value": results["best_value"],
                        "n_trials": results.get("n_trials", 0),
                    }

            return report
        except Exception as e:
            self.logger.error(f"Error generating optimization report: {e}")
            return {"error": str(e)}

    def _setup_optimization_storage(self) -> None:
        """Setup optimization storage."""
        try:
            # Ensure optimization directories exist
            os.makedirs("data/optimization_results", exist_ok=True)
            os.makedirs("data/calibration_results", exist_ok=True)
        except Exception as e:
            self.logger.error(f"Error setting up optimization storage: {e}")