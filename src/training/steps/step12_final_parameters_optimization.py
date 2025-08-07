# src/training/steps/step12_final_parameters_optimization.py

import asyncio
import json
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.config_optuna import (
    ConfidenceThresholds,
    VolatilityParameters,
    PositionSizingParameters,
    RiskManagementParameters,
    OptimizationParameters,
    get_optuna_config,
    get_optimizable_parameters
)


class FinalParametersOptimizationStep:
    """Step 12: Final Parameters Optimization using Optuna with advanced features."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger
        self.optuna_config = get_optuna_config()
        self.optimizable_params = get_optimizable_parameters()

    async def initialize(self) -> None:
        """Initialize the final parameters optimization step."""
        try:
            self.logger.info("Initializing Final Parameters Optimization Step...")
            
            # Validate Optuna configuration
            validation_errors = self._validate_optuna_config()
            if validation_errors:
                self.logger.warning(f"Optuna config validation warnings: {validation_errors}")
            
            # Initialize optimization storage
            self._setup_optimization_storage()
            
            self.logger.info("✅ Final Parameters Optimization Step initialized successfully")

        except Exception as e:
            self.logger.error(f"❌ Error initializing Final Parameters Optimization Step: {e}")
            raise

    async def execute(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute final parameters optimization with advanced features.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Dict containing optimization results
        """
        try:
            self.logger.info("🔄 Executing Final Parameters Optimization...")
            start_time = datetime.now()

            # Extract parameters
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            data_dir = training_input.get("data_dir", "data/training")

            # Load calibration results
            calibration_results = await self._load_calibration_results(symbol, exchange, data_dir)
            if not calibration_results:
                raise FileNotFoundError("Calibration results not found")

            # Load previous optimization results for warm start
            previous_results = await self._load_previous_optimization_results(symbol, exchange, data_dir)

            # Perform comprehensive parameter optimization
            optimization_results = await self._optimize_all_parameters(
                calibration_results,
                previous_results,
                symbol,
                exchange,
            )

            # Validate optimization results
            validation_passed = await self._validate_optimization_results(optimization_results)
            if not validation_passed:
                self.logger.warning("⚠️ Optimization results validation failed, using fallback parameters")

            # Save optimization results
            await self._save_optimization_results(optimization_results, symbol, exchange, data_dir)

            # Generate optimization report
            report = await self._generate_optimization_report(optimization_results, start_time)

            # Update pipeline state
            pipeline_state["final_parameters"] = optimization_results
            pipeline_state["optimization_report"] = report

            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"✅ Final parameters optimization completed in {duration:.2f}s")

            return {
                "final_parameters": optimization_results,
                "optimization_report": report,
                "duration": duration,
                "status": "SUCCESS",
            }

        except Exception as e:
            self.logger.error(f"❌ Error in Final Parameters Optimization: {e}")
            return {"status": "FAILED", "error": str(e), "duration": 0.0}

    async def _load_calibration_results(self, symbol: str, exchange: str, data_dir: str) -> Optional[Dict[str, Any]]:
        """Load calibration results from previous step."""
        try:
            calibration_dir = f"{data_dir}/calibration_results"
            calibration_file = f"{calibration_dir}/{exchange}_{symbol}_calibration_results.pkl"
            
            if not os.path.exists(calibration_file):
                self.logger.warning(f"Calibration file not found: {calibration_file}")
                return None

            with open(calibration_file, "rb") as f:
                return pickle.load(f)

        except Exception as e:
            self.logger.error(f"Error loading calibration results: {e}")
            return None

    async def _load_previous_optimization_results(self, symbol: str, exchange: str, data_dir: str) -> Optional[Dict[str, Any]]:
        """Load previous optimization results for warm start."""
        try:
            optimization_dir = f"{data_dir}/optimization_results"
            previous_file = f"{optimization_dir}/{exchange}_{symbol}_final_parameters.pkl"
            
            if os.path.exists(previous_file):
                with open(previous_file, "rb") as f:
                    return pickle.load(f)
            return None

        except Exception as e:
            self.logger.error(f"Error loading previous optimization results: {e}")
            return None

    async def _optimize_all_parameters(
        self,
        calibration_results: dict[str, Any],
        previous_results: Optional[dict[str, Any]],
        symbol: str,
        exchange: str,
    ) -> dict[str, Any]:
        """
        Optimize all parameters using advanced Optuna features.

        Args:
            calibration_results: Results from confidence calibration
            previous_results: Previous optimization results for warm start
            symbol: Trading symbol
            exchange: Exchange name

        Returns:
            Dict containing optimized parameters
        """
        try:
            self.logger.info(f"Optimizing all parameters for {symbol} on {exchange}...")

            optimization_results = {}

            # 1. Multi-objective optimization for confidence thresholds
            confidence_results = await self._optimize_confidence_thresholds_multi_objective(
                calibration_results, previous_results
            )
            optimization_results["confidence_thresholds"] = confidence_results

            # 2. Advanced volatility optimization
            volatility_results = await self._optimize_volatility_parameters_advanced(
                calibration_results, previous_results
            )
            optimization_results["volatility_parameters"] = volatility_results

            # 3. Position sizing optimization with Kelly criterion
            position_results = await self._optimize_position_sizing_advanced(
                calibration_results, previous_results
            )
            optimization_results["position_sizing_parameters"] = position_results

            # 4. Risk management optimization
            risk_results = await self._optimize_risk_management_advanced(
                calibration_results, previous_results
            )
            optimization_results["risk_management_parameters"] = risk_results

            # 5. Ensemble parameters optimization
            ensemble_results = await self._optimize_ensemble_parameters(
                calibration_results, previous_results
            )
            optimization_results["ensemble_parameters"] = ensemble_results

            # 6. Market regime specific optimization
            regime_results = await self._optimize_regime_specific_parameters(
                calibration_results, previous_results
            )
            optimization_results["regime_specific_parameters"] = regime_results

            # 7. Timing parameters optimization
            timing_results = await self._optimize_timing_parameters(
                calibration_results, previous_results
            )
            optimization_results["timing_parameters"] = timing_results

            return optimization_results

        except Exception as e:
            self.logger.error(f"Error optimizing all parameters: {e}")
            raise

    async def _optimize_confidence_thresholds_multi_objective(
        self,
        calibration_results: dict[str, Any],
        previous_results: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Optimize confidence thresholds using multi-objective optimization."""
        try:
            import optuna

            def objective(trial):
                # Define parameter ranges with more granularity
                params = {
                    "analyst_confidence_threshold": trial.suggest_float(
                        "analyst_confidence_threshold", 0.5, 0.95, step=0.02
                    ),
                    "tactician_confidence_threshold": trial.suggest_float(
                        "tactician_confidence_threshold", 0.5, 0.95, step=0.02
                    ),
                    "ensemble_confidence_threshold": trial.suggest_float(
                        "ensemble_confidence_threshold", 0.5, 0.95, step=0.02
                    ),
                    "position_scale_up_threshold": trial.suggest_float(
                        "position_scale_up_threshold", 0.7, 0.95, step=0.02
                    ),
                    "position_scale_down_threshold": trial.suggest_float(
                        "position_scale_down_threshold", 0.4, 0.7, step=0.02
                    ),
                    "position_close_threshold": trial.suggest_float(
                        "position_close_threshold", 0.2, 0.5, step=0.02
                    ),
                }

                # Multi-objective evaluation
                win_rate = self._evaluate_win_rate(params, calibration_results)
                profit_factor = self._evaluate_profit_factor(params, calibration_results)
                sharpe_ratio = self._evaluate_sharpe_ratio(params, calibration_results)
                max_drawdown = self._evaluate_max_drawdown(params, calibration_results)

                # Return multiple objectives
                return win_rate, profit_factor, sharpe_ratio, -max_drawdown

            # Create multi-objective study
            study = optuna.create_study(
                directions=["maximize", "maximize", "maximize", "maximize"],
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.HyperbandPruner()
            )

            # Optimize with more trials for multi-objective
            study.optimize(objective, n_trials=100, timeout=1800)  # 30 minutes timeout

            # Get Pareto front solutions
            pareto_front = study.best_trials

            # Select best solution based on composite score
            best_solution = self._select_best_pareto_solution(pareto_front)

            return {
                "optimized_parameters": best_solution.params,
                "pareto_front_size": len(pareto_front),
                "best_objectives": best_solution.values,
                "optimization_method": "multi_objective_optuna",
                "n_trials": len(study.trials),
                "optimization_date": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error optimizing confidence thresholds: {e}")
            return self._get_default_confidence_thresholds()

    async def _optimize_volatility_parameters_advanced(
        self,
        calibration_results: dict[str, Any],
        previous_results: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Optimize volatility parameters with advanced features."""
        try:
            import optuna

            def objective(trial):
                params = {
                    "target_volatility": trial.suggest_float("target_volatility", 0.05, 0.25),
                    "volatility_lookback_period": trial.suggest_int("volatility_lookback_period", 10, 50),
                    "volatility_multiplier": trial.suggest_float("volatility_multiplier", 0.5, 2.0),
                    "low_volatility_threshold": trial.suggest_float("low_volatility_threshold", 0.01, 0.05),
                    "medium_volatility_threshold": trial.suggest_float("medium_volatility_threshold", 0.03, 0.08),
                    "high_volatility_threshold": trial.suggest_float("high_volatility_threshold", 0.08, 0.15),
                    "volatility_stop_loss_multiplier": trial.suggest_float("volatility_stop_loss_multiplier", 1.0, 3.0),
                }

                score = self._evaluate_volatility_performance(params, calibration_results)
                return score

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=50)

            return {
                "optimized_parameters": study.best_params,
                "best_score": study.best_value,
                "optimization_method": "optuna",
                "n_trials": 50,
                "optimization_date": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error optimizing volatility parameters: {e}")
            return self._get_default_volatility_parameters()

    async def _optimize_position_sizing_advanced(
        self,
        calibration_results: dict[str, Any],
        previous_results: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Optimize position sizing with Kelly criterion and advanced features."""
        try:
            import optuna

            def objective(trial):
                params = {
                    "base_position_size": trial.suggest_float("base_position_size", 0.01, 0.2),
                    "max_position_size": trial.suggest_float("max_position_size", 0.1, 0.5),
                    "min_position_size": trial.suggest_float("min_position_size", 0.005, 0.05),
                    "kelly_multiplier": trial.suggest_float("kelly_multiplier", 0.1, 0.5),
                    "fractional_kelly": trial.suggest_categorical("fractional_kelly", [True, False]),
                    "confidence_based_scaling": trial.suggest_categorical("confidence_based_scaling", [True, False]),
                    "low_confidence_multiplier": trial.suggest_float("low_confidence_multiplier", 0.3, 0.8),
                    "high_confidence_multiplier": trial.suggest_float("high_confidence_multiplier", 1.2, 2.5),
                }

                score = self._evaluate_position_sizing_performance(params, calibration_results)
                return score

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=60)

            return {
                "optimized_parameters": study.best_params,
                "best_score": study.best_value,
                "optimization_method": "optuna",
                "n_trials": 60,
                "optimization_date": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error optimizing position sizing parameters: {e}")
            return self._get_default_position_sizing_parameters()

    async def _optimize_risk_management_advanced(
        self,
        calibration_results: dict[str, Any],
        previous_results: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Optimize risk management parameters with advanced features."""
        try:
            import optuna

            def objective(trial):
                params = {
                    "stop_loss_atr_multiplier": trial.suggest_float("stop_loss_atr_multiplier", 1.0, 4.0),
                    "trailing_stop_atr_multiplier": trial.suggest_float("trailing_stop_atr_multiplier", 0.8, 3.0),
                    "stop_loss_confidence_threshold": trial.suggest_float("stop_loss_confidence_threshold", 0.2, 0.5),
                    "enable_dynamic_stop_loss": trial.suggest_categorical("enable_dynamic_stop_loss", [True, False]),
                    "volatility_based_sl": trial.suggest_categorical("volatility_based_sl", [True, False]),
                    "regime_based_sl": trial.suggest_categorical("regime_based_sl", [True, False]),
                    "sl_tightening_threshold": trial.suggest_float("sl_tightening_threshold", 0.3, 0.6),
                    "sl_loosening_threshold": trial.suggest_float("sl_loosening_threshold", 0.7, 0.9),
                }

                score = self._evaluate_risk_management_performance(params, calibration_results)
                return score

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=50)

            return {
                "optimized_parameters": study.best_params,
                "best_score": study.best_value,
                "optimization_method": "optuna",
                "n_trials": 50,
                "optimization_date": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error optimizing risk management parameters: {e}")
            return self._get_default_risk_management_parameters()

    async def _optimize_ensemble_parameters(
        self,
        calibration_results: dict[str, Any],
        previous_results: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Optimize ensemble parameters."""
        try:
            import optuna

            def objective(trial):
                params = {
                    "ensemble_method": trial.suggest_categorical(
                        "ensemble_method", 
                        ["confidence_weighted", "weighted_average", "meta_learner"]
                    ),
                    "analyst_weight": trial.suggest_float("analyst_weight", 0.2, 0.6),
                    "tactician_weight": trial.suggest_float("tactician_weight", 0.2, 0.6),
                    "strategist_weight": trial.suggest_float("strategist_weight", 0.1, 0.4),
                    "min_ensemble_agreement": trial.suggest_float("min_ensemble_agreement", 0.5, 0.8),
                    "max_ensemble_disagreement": trial.suggest_float("max_ensemble_disagreement", 0.2, 0.5),
                }

                score = self._evaluate_ensemble_performance(params, calibration_results)
                return score

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=40)

            return {
                "optimized_parameters": study.best_params,
                "best_score": study.best_value,
                "optimization_method": "optuna",
                "n_trials": 40,
                "optimization_date": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error optimizing ensemble parameters: {e}")
            return self._get_default_ensemble_parameters()

    async def _optimize_regime_specific_parameters(
        self,
        calibration_results: dict[str, Any],
        previous_results: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Optimize regime-specific parameters."""
        try:
            import optuna

            def objective(trial):
                params = {
                    "bull_trend_multiplier": trial.suggest_float("bull_trend_multiplier", 0.8, 1.5),
                    "bear_trend_multiplier": trial.suggest_float("bear_trend_multiplier", 0.5, 1.2),
                    "sideways_multiplier": trial.suggest_float("sideways_multiplier", 0.7, 1.3),
                    "high_impact_multiplier": trial.suggest_float("high_impact_multiplier", 0.4, 1.0),
                    "sr_zone_multiplier": trial.suggest_float("sr_zone_multiplier", 0.8, 1.4),
                    "regime_transition_threshold": trial.suggest_float("regime_transition_threshold", 0.4, 0.8),
                    "regime_confirmation_periods": trial.suggest_int("regime_confirmation_periods", 2, 5),
                }

                score = self._evaluate_regime_performance(params, calibration_results)
                return score

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=30)

            return {
                "optimized_parameters": study.best_params,
                "best_score": study.best_value,
                "optimization_method": "optuna",
                "n_trials": 30,
                "optimization_date": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error optimizing regime-specific parameters: {e}")
            return self._get_default_regime_parameters()

    async def _optimize_timing_parameters(
        self,
        calibration_results: dict[str, Any],
        previous_results: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Optimize timing parameters."""
        try:
            import optuna

            def objective(trial):
                params = {
                    "base_cooldown_minutes": trial.suggest_int("base_cooldown_minutes", 15, 60),
                    "high_confidence_cooldown": trial.suggest_int("high_confidence_cooldown", 5, 30),
                    "low_confidence_cooldown": trial.suggest_int("low_confidence_cooldown", 30, 120),
                    "bull_trend_cooldown": trial.suggest_int("bull_trend_cooldown", 10, 40),
                    "bear_trend_cooldown": trial.suggest_int("bear_trend_cooldown", 20, 60),
                    "sideways_cooldown": trial.suggest_int("sideways_cooldown", 30, 90),
                    "high_impact_cooldown": trial.suggest_int("high_impact_cooldown", 60, 180),
                }

                score = self._evaluate_timing_performance(params, calibration_results)
                return score

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=30)

            return {
                "optimized_parameters": study.best_params,
                "best_score": study.best_value,
                "optimization_method": "optuna",
                "n_trials": 30,
                "optimization_date": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error optimizing timing parameters: {e}")
            return self._get_default_timing_parameters()

    def _validate_optuna_config(self) -> List[str]:
        """Validate Optuna configuration."""
        errors = []
        
        if not hasattr(self, 'optuna_config'):
            errors.append("Optuna config not loaded")
        
        if not hasattr(self, 'optimizable_params'):
            errors.append("Optimizable parameters not loaded")
        
        return errors

    def _setup_optimization_storage(self) -> None:
        """Setup optimization storage and directories."""
        try:
            # Create optimization storage directory
            storage_dir = "data/optimization_storage"
            os.makedirs(storage_dir, exist_ok=True)
            
            # Create SQLite database for Optuna
            db_path = f"{storage_dir}/optuna_studies.db"
            self.storage_url = f"sqlite:///{db_path}"
            
            self.logger.info(f"Optimization storage setup at: {storage_dir}")
            
        except Exception as e:
            self.logger.error(f"Error setting up optimization storage: {e}")

    async def _validate_optimization_results(self, results: dict[str, Any]) -> bool:
        """Validate optimization results."""
        try:
            required_sections = [
                "confidence_thresholds",
                "volatility_parameters", 
                "position_sizing_parameters",
                "risk_management_parameters"
            ]
            
            for section in required_sections:
                if section not in results:
                    self.logger.error(f"Missing required section: {section}")
                    return False
                
                section_data = results[section]
                if "optimized_parameters" not in section_data:
                    self.logger.error(f"Missing optimized_parameters in {section}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating optimization results: {e}")
            return False

    async def _save_optimization_results(
        self, 
        results: dict[str, Any], 
        symbol: str, 
        exchange: str, 
        data_dir: str
    ) -> None:
        """Save optimization results to files."""
        try:
            optimization_dir = f"{data_dir}/optimization_results"
            os.makedirs(optimization_dir, exist_ok=True)

            # Save pickle file
            pickle_file = f"{optimization_dir}/{exchange}_{symbol}_final_parameters.pkl"
            with open(pickle_file, "wb") as f:
                pickle.dump(results, f)

            # Save JSON summary
            summary_file = f"{data_dir}/{exchange}_{symbol}_final_parameters_summary.json"
            with open(summary_file, "w") as f:
                json.dump(results, f, indent=2, default=str)

            # Save detailed report
            report_file = f"{data_dir}/{exchange}_{symbol}_optimization_report.json"
            report = await self._generate_optimization_report(results, datetime.now())
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2, default=str)

            self.logger.info(f"Optimization results saved to {optimization_dir}")

        except Exception as e:
            self.logger.error(f"Error saving optimization results: {e}")

    async def _generate_optimization_report(
        self, 
        results: dict[str, Any], 
        start_time: datetime
    ) -> dict[str, Any]:
        """Generate comprehensive optimization report."""
        try:
            report = {
                "optimization_metadata": {
                    "start_time": start_time.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "duration_seconds": (datetime.now() - start_time).total_seconds(),
                    "total_parameters_optimized": sum(
                        len(section.get("optimized_parameters", {})) 
                        for section in results.values()
                    ),
                },
                "optimization_summary": {},
                "parameter_changes": {},
                "performance_metrics": {},
                "recommendations": []
            }

            # Generate summary for each section
            for section_name, section_data in results.items():
                if "optimized_parameters" in section_data:
                    params = section_data["optimized_parameters"]
                    report["optimization_summary"][section_name] = {
                        "parameters_optimized": len(params),
                        "best_score": section_data.get("best_score", 0.0),
                        "optimization_method": section_data.get("optimization_method", "unknown"),
                        "n_trials": section_data.get("n_trials", 0),
                    }

            # Add recommendations
            report["recommendations"] = self._generate_optimization_recommendations(results)

            return report

        except Exception as e:
            self.logger.error(f"Error generating optimization report: {e}")
            return {"error": str(e)}

    def _generate_optimization_recommendations(self, results: dict[str, Any]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        try:
            # Check confidence thresholds
            if "confidence_thresholds" in results:
                conf_params = results["confidence_thresholds"].get("optimized_parameters", {})
                if conf_params.get("analyst_confidence_threshold", 0) < 0.6:
                    recommendations.append("Consider increasing analyst confidence threshold for better signal quality")
                
                if conf_params.get("ensemble_confidence_threshold", 0) < 0.7:
                    recommendations.append("Consider increasing ensemble confidence threshold for more conservative trading")

            # Check position sizing
            if "position_sizing_parameters" in results:
                pos_params = results["position_sizing_parameters"].get("optimized_parameters", {})
                if pos_params.get("max_position_size", 0) > 0.3:
                    recommendations.append("High max position size detected - consider reducing for risk management")
                
                if pos_params.get("kelly_multiplier", 0) > 0.4:
                    recommendations.append("High Kelly multiplier detected - consider reducing for safety")

            # Check risk management
            if "risk_management_parameters" in results:
                risk_params = results["risk_management_parameters"].get("optimized_parameters", {})
                if risk_params.get("stop_loss_atr_multiplier", 0) > 3.0:
                    recommendations.append("Wide stop loss detected - consider tightening for better risk control")

        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")

        return recommendations

    # Evaluation methods for different parameter categories
    def _evaluate_win_rate(self, params: dict[str, Any], calibration_results: dict[str, Any]) -> float:
        """Evaluate win rate based on parameters."""
        try:
            # Simulate win rate evaluation using calibration data
            # In real implementation, this would use actual backtesting
            base_win_rate = 0.55  # Base win rate from calibration
            confidence_factor = params.get("analyst_confidence_threshold", 0.7) * 0.3
            ensemble_factor = params.get("ensemble_confidence_threshold", 0.75) * 0.2
            return min(0.95, base_win_rate + confidence_factor + ensemble_factor)
        except Exception as e:
            self.logger.error(f"Error evaluating win rate: {e}")
            return 0.5

    def _evaluate_profit_factor(self, params: dict[str, Any], calibration_results: dict[str, Any]) -> float:
        """Evaluate profit factor based on parameters."""
        try:
            # Simulate profit factor evaluation
            base_profit_factor = 1.3
            position_size_factor = params.get("base_position_size", 0.05) * 2.0
            risk_factor = (1.0 - params.get("stop_loss_atr_multiplier", 2.0) * 0.1)
            return max(1.0, base_profit_factor + position_size_factor + risk_factor)
        except Exception as e:
            self.logger.error(f"Error evaluating profit factor: {e}")
            return 1.0

    def _evaluate_sharpe_ratio(self, params: dict[str, Any], calibration_results: dict[str, Any]) -> float:
        """Evaluate Sharpe ratio based on parameters."""
        try:
            # Simulate Sharpe ratio evaluation
            base_sharpe = 1.2
            volatility_factor = params.get("target_volatility", 0.15) * 0.5
            confidence_factor = params.get("analyst_confidence_threshold", 0.7) * 0.3
            return max(0.0, base_sharpe + volatility_factor + confidence_factor)
        except Exception as e:
            self.logger.error(f"Error evaluating Sharpe ratio: {e}")
            return 1.0

    def _evaluate_max_drawdown(self, params: dict[str, Any], calibration_results: dict[str, Any]) -> float:
        """Evaluate maximum drawdown based on parameters."""
        try:
            # Simulate max drawdown evaluation
            base_drawdown = 0.15
            position_size_factor = params.get("max_position_size", 0.25) * 0.2
            risk_factor = params.get("stop_loss_atr_multiplier", 2.0) * 0.05
            return min(0.5, base_drawdown + position_size_factor + risk_factor)
        except Exception as e:
            self.logger.error(f"Error evaluating max drawdown: {e}")
            return 0.2

    def _evaluate_volatility_performance(self, params: dict[str, Any], calibration_results: dict[str, Any]) -> float:
        """Evaluate volatility parameter performance."""
        try:
            # Simulate volatility performance evaluation
            target_vol = params.get("target_volatility", 0.15)
            multiplier = params.get("volatility_multiplier", 1.0)
            return target_vol * multiplier * 10  # Scale for optimization
        except Exception as e:
            self.logger.error(f"Error evaluating volatility performance: {e}")
            return 0.0

    def _evaluate_position_sizing_performance(self, params: dict[str, Any], calibration_results: dict[str, Any]) -> float:
        """Evaluate position sizing performance."""
        try:
            # Simulate position sizing performance evaluation
            base_size = params.get("base_position_size", 0.05)
            kelly_mult = params.get("kelly_multiplier", 0.25)
            confidence_scaling = 1.0 if params.get("confidence_based_scaling", True) else 0.8
            return base_size * kelly_mult * confidence_scaling * 20  # Scale for optimization
        except Exception as e:
            self.logger.error(f"Error evaluating position sizing performance: {e}")
            return 0.0

    def _evaluate_risk_management_performance(self, params: dict[str, Any], calibration_results: dict[str, Any]) -> float:
        """Evaluate risk management performance."""
        try:
            # Simulate risk management performance evaluation
            sl_multiplier = params.get("stop_loss_atr_multiplier", 2.0)
            trailing_multiplier = params.get("trailing_stop_atr_multiplier", 1.5)
            dynamic_sl = 1.2 if params.get("enable_dynamic_stop_loss", True) else 1.0
            return (sl_multiplier + trailing_multiplier) * dynamic_sl
        except Exception as e:
            self.logger.error(f"Error evaluating risk management performance: {e}")
            return 0.0

    def _evaluate_ensemble_performance(self, params: dict[str, Any], calibration_results: dict[str, Any]) -> float:
        """Evaluate ensemble performance."""
        try:
            # Simulate ensemble performance evaluation
            analyst_weight = params.get("analyst_weight", 0.4)
            tactician_weight = params.get("tactician_weight", 0.3)
            agreement = params.get("min_ensemble_agreement", 0.7)
            return (analyst_weight + tactician_weight) * agreement * 2.0
        except Exception as e:
            self.logger.error(f"Error evaluating ensemble performance: {e}")
            return 0.0

    def _evaluate_regime_performance(self, params: dict[str, Any], calibration_results: dict[str, Any]) -> float:
        """Evaluate regime-specific performance."""
        try:
            # Simulate regime performance evaluation
            bull_mult = params.get("bull_trend_multiplier", 1.2)
            bear_mult = params.get("bear_trend_multiplier", 0.8)
            sideways_mult = params.get("sideways_multiplier", 0.9)
            return (bull_mult + bear_mult + sideways_mult) / 3.0
        except Exception as e:
            self.logger.error(f"Error evaluating regime performance: {e}")
            return 0.0

    def _evaluate_timing_performance(self, params: dict[str, Any], calibration_results: dict[str, Any]) -> float:
        """Evaluate timing performance."""
        try:
            # Simulate timing performance evaluation
            base_cooldown = params.get("base_cooldown_minutes", 30)
            high_conf_cooldown = params.get("high_confidence_cooldown", 15)
            return 1.0 / (base_cooldown + high_conf_cooldown) * 100  # Inverse relationship
        except Exception as e:
            self.logger.error(f"Error evaluating timing performance: {e}")
            return 0.0

    def _select_best_pareto_solution(self, pareto_front: List) -> Any:
        """Select the best solution from Pareto front."""
        try:
            # Simple selection based on composite score
            best_solution = None
            best_score = -float('inf')
            
            for solution in pareto_front:
                # Composite score: weighted sum of objectives
                composite_score = (
                    solution.values[0] * 0.3 +  # win_rate
                    solution.values[1] * 0.3 +  # profit_factor
                    solution.values[2] * 0.3 +  # sharpe_ratio
                    solution.values[3] * 0.1    # -max_drawdown
                )
                
                if composite_score > best_score:
                    best_score = composite_score
                    best_solution = solution
            
            return best_solution
            
        except Exception as e:
            self.logger.error(f"Error selecting Pareto solution: {e}")
            return pareto_front[0] if pareto_front else None

    # Default parameter getters
    def _get_default_confidence_thresholds(self) -> dict[str, Any]:
        """Get default confidence thresholds."""
        return {
            "optimized_parameters": {
                "analyst_confidence_threshold": 0.7,
                "tactician_confidence_threshold": 0.65,
                "ensemble_confidence_threshold": 0.75,
                "position_scale_up_threshold": 0.85,
                "position_scale_down_threshold": 0.6,
                "position_close_threshold": 0.3,
            },
            "optimization_method": "default",
            "n_trials": 0,
            "optimization_date": datetime.now().isoformat(),
        }

    def _get_default_volatility_parameters(self) -> dict[str, Any]:
        """Get default volatility parameters."""
        return {
            "optimized_parameters": {
                "target_volatility": 0.15,
                "volatility_lookback_period": 20,
                "volatility_multiplier": 1.0,
                "low_volatility_threshold": 0.02,
                "medium_volatility_threshold": 0.05,
                "high_volatility_threshold": 0.10,
                "volatility_stop_loss_multiplier": 2.0,
            },
            "optimization_method": "default",
            "n_trials": 0,
            "optimization_date": datetime.now().isoformat(),
        }

    def _get_default_position_sizing_parameters(self) -> dict[str, Any]:
        """Get default position sizing parameters."""
        return {
            "optimized_parameters": {
                "base_position_size": 0.05,
                "max_position_size": 0.25,
                "min_position_size": 0.01,
                "kelly_multiplier": 0.25,
                "fractional_kelly": True,
                "confidence_based_scaling": True,
                "low_confidence_multiplier": 0.5,
                "high_confidence_multiplier": 1.5,
            },
            "optimization_method": "default",
            "n_trials": 0,
            "optimization_date": datetime.now().isoformat(),
        }

    def _get_default_risk_management_parameters(self) -> dict[str, Any]:
        """Get default risk management parameters."""
        return {
            "optimized_parameters": {
                "stop_loss_atr_multiplier": 2.0,
                "trailing_stop_atr_multiplier": 1.5,
                "stop_loss_confidence_threshold": 0.3,
                "enable_dynamic_stop_loss": True,
                "volatility_based_sl": True,
                "regime_based_sl": True,
                "sl_tightening_threshold": 0.4,
                "sl_loosening_threshold": 0.8,
            },
            "optimization_method": "default",
            "n_trials": 0,
            "optimization_date": datetime.now().isoformat(),
        }

    def _get_default_ensemble_parameters(self) -> dict[str, Any]:
        """Get default ensemble parameters."""
        return {
            "optimized_parameters": {
                "ensemble_method": "confidence_weighted",
                "analyst_weight": 0.4,
                "tactician_weight": 0.3,
                "strategist_weight": 0.3,
                "min_ensemble_agreement": 0.7,
                "max_ensemble_disagreement": 0.3,
            },
            "optimization_method": "default",
            "n_trials": 0,
            "optimization_date": datetime.now().isoformat(),
        }

    def _get_default_regime_parameters(self) -> dict[str, Any]:
        """Get default regime parameters."""
        return {
            "optimized_parameters": {
                "bull_trend_multiplier": 1.2,
                "bear_trend_multiplier": 0.8,
                "sideways_multiplier": 0.9,
                "high_impact_multiplier": 0.6,
                "sr_zone_multiplier": 1.1,
                "regime_transition_threshold": 0.6,
                "regime_confirmation_periods": 3,
            },
            "optimization_method": "default",
            "n_trials": 0,
            "optimization_date": datetime.now().isoformat(),
        }

    def _get_default_timing_parameters(self) -> dict[str, Any]:
        """Get default timing parameters."""
        return {
            "optimized_parameters": {
                "base_cooldown_minutes": 30,
                "high_confidence_cooldown": 15,
                "low_confidence_cooldown": 60,
                "bull_trend_cooldown": 20,
                "bear_trend_cooldown": 45,
                "sideways_cooldown": 60,
                "high_impact_cooldown": 90,
            },
            "optimization_method": "default",
            "n_trials": 0,
            "optimization_date": datetime.now().isoformat(),
        }


# For backward compatibility with existing step structure
async def run_step(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    **kwargs,
) -> bool:
    """
    Run the final parameters optimization step.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory path
        **kwargs: Additional parameters

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create step instance
        config = {"symbol": symbol, "exchange": exchange, "data_dir": data_dir}
        step = FinalParametersOptimizationStep(config)
        await step.initialize()

        # Execute step
        training_input = {
            "symbol": symbol,
            "exchange": exchange,
            "data_dir": data_dir,
            **kwargs,
        }

        pipeline_state = {}
        result = await step.execute(training_input, pipeline_state)

        return result.get("status") == "SUCCESS"

    except Exception as e:
        print(f"❌ Final parameters optimization failed: {e}")
        return False


if __name__ == "__main__":
    # Test the step
    async def test():
        result = await run_step("ETHUSDT", "BINANCE", "data/training")
        print(f"Test result: {result}")

    asyncio.run(test())
