#!/usr/bin/env python3
"""
Early Stage Optimization Module

This module handles optimization that should happen BEFORE ML trading begins:
1. SR (Stationarity and Randomness) optimization (step2_5)
2. Regime-specific triple barrier optimization (step4)

These optimizations happen early in the pipeline to ensure:
- Proper data preprocessing (SR)
- Regime-aware trading parameters (triple barrier)
- Optimal foundation for ML model training
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
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
    from .steps.step4_analyst_labeling_feature_engineering_components.regime_specific_triple_barrier_optimizer import (
        RegimeSpecificTripleBarrierOptimizer,
        create_regime_specific_triple_barrier_optimizer
    )
    REGIME_OPTIMIZER_AVAILABLE = True
except ImportError:
    REGIME_OPTIMIZER_AVAILABLE = False
    RegimeSpecificTripleBarrierOptimizer = None
    create_regime_specific_triple_barrier_optimizer = None


class EarlyStageOptimizer:
    """
    Early stage optimizer for parameters that must be set before ML trading begins.
    
    This includes:
    - SR optimization (step2_5) - data preprocessing parameters
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
    
    async def optimize_sr_parameters(
        self, 
        data: pd.DataFrame,
        optimization_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize SR (Stationarity and Randomness) parameters for data preprocessing."""
        
        self.logger.info("🚀 Starting SR parameter optimization...")
        
        if not OPTUNA_AVAILABLE:
            return {"error": "Optuna is required for SR optimization"}
        
        try:
            # Create study for SR optimization
            study = optuna.create_study(
                study_name="sr_parameter_optimization",
                direction="maximize",  # Maximize data quality metric
                sampler=optuna.samplers.TPESampler(
                    n_startup_trials=10,
                    n_ei_candidates=24,
                    multivariate=True
                ),
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=5,
                    n_warmup_steps=10,
                    interval_steps=3
                )
            )
            
            # Create objective function
            objective = self._create_sr_objective(data)
            
            # Run optimization
            n_trials = optimization_config.get("n_trials", 100)
            timeout = optimization_config.get("timeout", 1800)  # 30 minutes
            
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
            
            # Store results
            self.sr_optimization_results = {
                "best_params": best_params,
                "best_value": best_value,
                "best_trial": best_trial.number,
                "total_trials": len(study.trials),
                "optimization_history": [trial.value for trial in study.trials if trial.value is not None],
                "optimization_timestamp": datetime.now().isoformat()
            }
            
            # Log to MLflow
            if MLFLOW_AVAILABLE:
                await self._log_sr_optimization_to_mlflow(self.sr_optimization_results)
            
            self.logger.info("✅ SR parameter optimization completed successfully!")
            
            return self.sr_optimization_results
            
        except Exception as e:
            error_msg = f"SR optimization failed: {e}"
            self.logger.error(f"❌ {error_msg}")
            return {"error": error_msg}
    
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
    
    async def get_regime_optimization_status(self) -> Dict[str, Any]:
        """Get status of regime-specific triple barrier optimization."""
        
        if not self.regime_optimizer:
            return {"error": "Regime-specific triple barrier optimizer not available"}
        
        try:
            return await self.regime_optimizer.get_regime_optimization_status()
        except Exception as e:
            return {"error": f"Failed to get regime optimization status: {e}"}
    
    async def apply_regime_specific_parameters(self, regime_name: str) -> Dict[str, Any]:
        """Apply optimized parameters for a specific regime."""
        
        if not self.regime_optimizer:
            return {"error": "Regime-specific triple barrier optimizer not available"}
        
        try:
            return await self.regime_optimizer.apply_regime_parameters(regime_name)
        except Exception as e:
            return {"error": f"Failed to apply regime parameters: {e}"}
    
    async def get_regime_optimization_recommendations(self) -> List[str]:
        """Get recommendations based on regime-specific optimization results."""
        
        if not self.regime_optimizer:
            return ["Regime-specific triple barrier optimizer not available"]
        
        try:
            return await self.regime_optimizer.get_optimization_recommendations()
        except Exception as e:
            return [f"Failed to get regime optimization recommendations: {e}"]
    
    async def get_triple_barrier_labeler(self):
        """Get the integrated triple barrier labeler from the regime optimizer."""
        
        if not self.regime_optimizer:
            return None
        
        try:
            return await self.regime_optimizer.get_triple_barrier_labeler()
        except Exception as e:
            self.logger.error(f"Failed to get triple barrier labeler: {e}")
            return None
    
    def _create_sr_objective(self, data: pd.DataFrame):
        """Create objective function for SR optimization."""
        
        def objective(trial):
            # Sample SR parameters
            params = {
                "fractional_d": trial.suggest_float("fractional_d", 0.1, 0.9, log=True),
                "window_size": trial.suggest_int("window_size", 10, 200),
                "min_periods": trial.suggest_int("min_periods", 5, 100),
                "threshold": trial.suggest_float("threshold", 0.001, 0.1, log=True),
                "adf_significance": trial.suggest_float("adf_significance", 0.01, 0.1, log=True),
                "kpss_significance": trial.suggest_float("kpss_significance", 0.01, 0.1, log=True)
            }
            
            # Evaluate the parameters on data
            try:
                quality_score = self._evaluate_sr_parameters(data, params)
                return quality_score
            except Exception as e:
                self.logger.warning(f"SR trial failed: {e}")
                return float('-inf')
        
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
    
    async def optimize_regime_specific_triple_barrier(
        self, 
        regime_data: Dict[str, pd.DataFrame],
        optimization_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize regime-specific triple barrier parameters."""
        
        self.logger.info("🚀 Starting regime-specific triple barrier optimization...")
        self.logger.info(f"Regimes to optimize: {list(regime_data.keys())}")
        
        if not OPTUNA_AVAILABLE:
            return {"error": "Optuna is required for regime-specific optimization"}
        
        try:
            optimization_results = {}
            
            for regime_name, regime_df in regime_data.items():
                self.logger.info(f"🔧 Optimizing triple barrier parameters for {regime_name} regime...")
                
                # Create regime-specific study
                study = await self._create_regime_barrier_study(regime_name, optimization_config)
                
                # Run optimization for this regime
                regime_result = await self._optimize_single_regime_barrier(
                    regime_name, 
                    regime_df, 
                    study, 
                    optimization_config
                )
                
                optimization_results[regime_name] = regime_result
                
                self.logger.info(f"✅ {regime_name} regime optimization completed")
            
            # Store overall results
            self.regime_barrier_optimization_results = optimization_results
            
            # Log to MLflow
            if MLFLOW_AVAILABLE:
                await self._log_regime_optimization_to_mlflow(optimization_results)
            
            self.logger.info("✅ Regime-specific triple barrier optimization completed!")
            
            return optimization_results
            
        except Exception as e:
            error_msg = f"Regime-specific optimization failed: {e}"
            self.logger.error(f"❌ {error_msg}")
            return {"error": error_msg}
    
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
    
    def _get_regime_barrier_parameters(self, regime_name: str) -> Dict[str, Any]:
        """Get regime-specific barrier parameter ranges."""
        
        # Base parameter ranges
        base_params = {
            "upper_barrier_multiplier": (0.1, 5.0),
            "lower_barrier_multiplier": (0.1, 5.0),
            "barrier_timeout": (1, 1440),  # minutes
            "barrier_adjustment": (0.1, 2.0),
            "dynamic_barriers": [True, False],
            "confidence_threshold": (0.3, 0.99),
            "position_size_multiplier": (0.1, 2.0),
            "risk_per_trade": (0.001, 0.1)
        }
        
        # Regime-specific adjustments
        if regime_name == "bull_regime":
            # Wider upper barrier, tighter lower barrier for bull markets
            base_params["upper_barrier_multiplier"] = (0.3, 1.5)
            base_params["lower_barrier_multiplier"] = (0.1, 0.8)
            base_params["barrier_timeout"] = (5, 60)
        elif regime_name == "bear_regime":
            # Tighter upper barrier, wider lower barrier for bear markets
            base_params["upper_barrier_multiplier"] = (0.1, 0.8)
            base_params["lower_barrier_multiplier"] = (0.3, 1.5)
            base_params["barrier_timeout"] = (10, 120)
        elif regime_name == "volatile_regime":
            # Much wider barriers for volatile markets
            base_params["upper_barrier_multiplier"] = (0.5, 2.0)
            base_params["lower_barrier_multiplier"] = (0.5, 2.0)
            base_params["barrier_timeout"] = (3, 45)
            base_params["position_size_multiplier"] = (0.05, 0.8)
            base_params["risk_per_trade"] = (0.001, 0.05)
        
        return base_params
    
    def _create_regime_barrier_objective(
        self, 
        regime_name: str, 
        regime_data: pd.DataFrame, 
        regime_params: Dict[str, Any]
    ):
        """Create objective function for regime-specific barrier optimization."""
        
        def objective(trial):
            # Sample parameters from regime-specific configuration
            params = {}
            
            for param_name, param_config in regime_params.items():
                if isinstance(param_config, tuple):
                    # Numeric range parameter
                    if len(param_config) == 2:
                        if param_name in ["barrier_timeout"]:
                            # Integer parameters
                            params[param_name] = trial.suggest_int(
                                param_name, 
                                param_config[0], 
                                param_config[1]
                            )
                        else:
                            # Float parameters
                            params[param_name] = trial.suggest_float(
                                param_name, 
                                param_config[0], 
                                param_config[1],
                                log=True
                            )
                elif isinstance(param_config, list):
                    # Categorical parameter
                    params[param_name] = trial.suggest_categorical(param_name, param_config)
                else:
                    # Single value parameter
                    params[param_name] = param_config
            
            # Evaluate the parameters on regime data
            try:
                performance_score = self._evaluate_regime_barrier_parameters(
                    regime_name, 
                    regime_data, 
                    params
                )
                return performance_score
            except Exception as e:
                self.logger.warning(f"Regime barrier trial failed for {regime_name}: {e}")
                return float('-inf')
        
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
        """Calculate performance score for regime-specific barrier parameters."""
        
        # Base score
        base_score = 0.0
        
        # Barrier settings scoring
        if regime_name == "bull_regime":
            if upper_barrier > lower_barrier:  # Wider upper barrier
                base_score += 0.3
            if timeout < 60:  # Faster timeout
                base_score += 0.2
            if position_size > 1.0:  # Larger positions
                base_score += 0.2
        elif regime_name == "bear_regime":
            if lower_barrier > upper_barrier:  # Wider lower barrier
                base_score += 0.3
            if timeout > 60:  # Slower timeout
                base_score += 0.2
            if position_size < 1.0:  # Smaller positions
                base_score += 0.2
        elif regime_name == "volatile_regime":
            if upper_barrier > 1.5 and lower_barrier > 1.5:  # Wide barriers
                base_score += 0.3
            if timeout < 45:  # Short timeout
                base_score += 0.2
            if position_size < 0.8:  # Much smaller positions
                base_score += 0.2
            if risk_per_trade < 0.03:  # Lower risk
                base_score += 0.1
        
        # Add some randomness to simulate real evaluation
        random_factor = np.random.normal(0, 0.1)
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
    
    async def get_optimization_status(self) -> Dict[str, Any]:
        """Get current status of early stage optimization."""
        
        return {
            "sr_optimization_completed": bool(self.sr_optimization_results),
            "regime_optimization_completed": bool(self.regime_barrier_optimization_results),
            "sr_optimization_timestamp": self.sr_optimization_results.get("optimization_timestamp", ""),
            "total_regimes_optimized": len(self.regime_barrier_optimization_results),
            "optimization_summary": self._create_optimization_summary()
        }
    
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
    # Example usage
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
                "early_stopping_patience": 20
            }
        }
    }
    
    # Create optimizer instance
    optimizer = create_early_stage_optimizer(config)
    
    print("✅ Early Stage Optimizer created successfully!")
    print("This optimizer handles:")
    print("  - SR parameter optimization (step2_5)")
    print("  - Regime-specific triple barrier optimization (step4)")
    print("  - Both happen BEFORE ML trading begins")