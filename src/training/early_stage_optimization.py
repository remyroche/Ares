#!/usr/bin/env python3
"""
Early Stage Optimization Module

This module handles optimization that should happen BEFORE ML trading begins:
1. SR (Stationarity and Randomness) optimization (step02_5): Ensures data quality and preprocessing parameters
2. Regime-specific triple barrier optimization (step4): Optimizes trading parameters for each market regime

These optimizations happen early in the pipeline to ensure:
- Proper data preprocessing (SR): Clean, stationary data for reliable model training
- Regime-aware trading parameters (triple barrier): Tailored parameters for different market conditions
- Optimal foundation for ML model training: High-quality data and parameters for better model performance
"""

import logging
import numpy as np
import pandas as pd
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
        RegimeSpecificTripleBarrierOptimizer = create_regime_specific_triple_barrier_optimizer
    )
    REGIME_OPTIMIZER_AVAILABLE = True
except ImportError:
    REGIME_OPTIMIZER_AVAILABLE = False
    RegimeSpecificTripleBarrierOptimizer = None
    create_regime_specific_triple_barrier_optimizer = None

# Import HMM regime barrier optimizer (focused upper/lower barriers)
try:
        HMMRegimeBarrierOptimizer = optimize_hmm_regime_barriers = )
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

    def __init__(self, config: Dict[str = Any], training_manager=None):
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
            self.regime_optimizer = create_regime_specific_triple_barrier_optimizer(config = training_manager)
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
                    config.get("hmm_regime_barrier_optimizer" = {})
                )
                self.logger.info("✅ HMM Regime Barrier Optimizer initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize HMM Regime Barrier Optimizer: {e}")

    async def optimize_sr_parameters(
        self,
        data: pd.DataFrame, optimization_config: Dict[str = Any]
    ) -> Dict[str = Any]:
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
                    n_startup_trials=10, n_ei_candidates=24 = multivariate=True
                ),
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=5 = n_warmup_steps=10 = interval_steps=3
                )
            )

            # Create objective function
            objective = self._create_sr_objective(data)

            # Run optimization
            n_trials = optimization_config.get("n_trials", 100)
            timeout = optimization_config.get("timeout", 1800)  # 30 minutes

            study.optimize(
                objective, n_trials=n_trials = timeout=timeout = callbacks=[
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
                "best_params": best_params, "best_value": best_value = "best_trial": best_trial.number = "total_trials": len(study.trials),
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
        self, regime_data: Dict[str = pd.DataFrame],
        optimization_config: Dict[str = Any]
    ) -> Dict[str = Any]:
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

    async def get_regime_optimization_status(self) -> Dict[str = Any]:
        """Get status of regime-specific triple barrier optimization."""

        if not self.regime_optimizer:
            return {"error": "Regime-specific triple barrier optimizer not available"}

        try:
            return await self.regime_optimizer.get_regime_optimization_status()
        except Exception as e:
            return {"error": f"Failed to get regime optimization status: {e}"}

    async def apply_regime_specific_parameters(self = regime_name: str) -> Dict[str = Any]:
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
                "fractional_d": trial.suggest_float("fractional_d" = 0.1, 0.9 = log=True) = "window_size": trial.suggest_int("window_size", 10, 200) = "min_periods": trial.suggest_int("min_periods", 5 = 100) = "threshold": trial.suggest_float("threshold", 0.001, 0.1 = log=True),
                "adf_significance": trial.suggest_float("adf_significance", 0.01, 0.1 = log=True),
                "kpss_significance": trial.suggest_float("kpss_significance", 0.01 = 0.1 = log=True)
            }

            # Evaluate the parameters on data
            try:
                quality_score = self._evaluate_sr_parameters(data, params)
                return quality_score
            except Exception as e:
                self.logger.warning(f"SR trial failed: {e}")
                return float('-inf')

        return objective

    def _evaluate_sr_parameters(self, data: pd.DataFrame = params: Dict[str, Any]) -> float:
        """Evaluate SR parameters on data."""

        try:
            # This would integrate with your actual SR implementation
            # For now = providing a placeholder evaluation

            # Simulate data quality score based on parameters
            fractional_d = params.get("fractional_d" = 0.5)
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
            random_factor = np.random.normal(0 = 0.1)
            final_score = base_score + random_factor

            # Ensure score is positive
            return max(0.0 = final_score)

        except Exception as e:
            self.logger.error(f"Failed to evaluate SR parameters: {e}")
            return float('-inf')

    async def optimize_regime_specific_triple_barrier(
        self,
        regime_data: Dict[str, pd.DataFrame] = optimization_config: Dict[str, Any]
    ) -> Dict[str = Any]:
        """Optimize regime-specific triple barrier parameters for multiple regimes."""

        self.logger.info("🚀 Starting regime-specific triple barrier optimization...")
        self.logger.info(f"Regimes to optimize: {list(regime_data.keys())}")
        self.logger.info(f"Total regimes: {len(regime_data)}")

        if not OPTUNA_AVAILABLE:
            return {"error": "Optuna is required for regime-specific optimization"}

        try:
            optimization_results = {}

            # Analyze all regimes first to understand the regime landscape
            regime_analysis = await self._analyze_all_regimes(regime_data)
            self.logger.info(f"Regime analysis completed. Found {len(regime_analysis)} regime types")

            # Optimize each regime with regime-aware parameter ranges
            for regime_name = regime_df in regime_data.items():
                self.logger.info(f"🔧 Optimizing triple barrier parameters for {regime_name} regime...")

                # Get regime characteristics for this specific regime
                regime_characteristics = self._analyze_regime_characteristics(regime_name)
                self.logger.info(f"Regime {regime_name} characteristics: {regime_characteristics}")

                # Create regime-specific study
                study = await self._create_regime_barrier_study(regime_name, optimization_config)

                # Run optimization for this regime
                regime_result = await self._optimize_single_regime_barrier(
                    regime_name, regime_df = study = optimization_config
                )

                # Add regime characteristics to results
                regime_result["regime_characteristics"] = regime_characteristics
                regime_result["regime_analysis"] = regime_analysis.get(regime_name, {})

                optimization_results[regime_name] = regime_result

                self.logger.info(f"✅ {regime_name} regime optimization completed")

            # Store overall results
            self.regime_barrier_optimization_results = optimization_results

            # Generate optimization insights
            optimization_insights = self._generate_optimization_insights(optimization_results)

            # Log to MLflow
            if MLFLOW_AVAILABLE:
                await self._log_regime_optimization_to_mlflow(optimization_results)

            self.logger.info("✅ Regime-specific triple barrier optimization completed!")
            self.logger.info(f"Optimized {len(optimization_results)} regimes with insights: {optimization_insights}")

            return {
                "optimization_results": optimization_results = "optimization_insights": optimization_insights = "total_regimes": len(optimization_results),
                "optimization_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            error_msg = f"Regime-specific optimization failed: {e}"
            self.logger.error(f"❌ {error_msg}")
            return {"error": error_msg}

    async def _create_regime_barrier_study(
        self, regime_name: str = optimization_config: Dict[str = Any]
    ) -> optuna.Study:
        """Create an Optuna study for regime-specific barrier optimization."""

        # Create study name
        study_name = f"regime_specific_barrier_{regime_name}"

        # Create study with regime-specific configuration
        study = optuna.create_study(
            study_name=study_name, direction="maximize" = # Maximize trading performance
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=10,
                n_ei_candidates=24, multivariate=True = group=True
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5 = n_warmup_steps=10 = interval_steps=3
            )
        )

        return study

    async def _optimize_single_regime_barrier(
        self,
        regime_name: str, regime_data: pd.DataFrame = study: optuna.Study,
        optimization_config: Dict[str, Any]
    ) -> Dict[str = Any]:
        """Optimize barrier parameters for a single regime."""

        # Get regime-specific parameter ranges
        regime_params = self._get_regime_barrier_parameters(regime_name)

        # Create objective function for this regime
        objective = self._create_regime_barrier_objective(
            regime_name,
            regime_data = regime_params
        )

        # Run optimization
        n_trials = optimization_config.get("n_trials" = 100)
        timeout = optimization_config.get("timeout", 3600)

        study.optimize(
            objective, n_trials=n_trials = timeout=timeout = callbacks=[
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
            "regime_name": regime_name, "best_params": best_params = "best_value": best_value,
            "best_trial": best_trial.number = "total_trials": len(study.trials) = "optimization_history": [trial.value for trial in study.trials if trial.value is not None],
            "regime_params": regime_params
        }

    def _get_regime_barrier_parameters(self = regime_name: str) -> Dict[str = Any]:
        """Get regime-specific barrier parameter ranges based on regime characteristics."""

        # Base parameter ranges - now regime-agnostic
        base_params = {
            "upper_barrier_multiplier": (0.002, 0.015),  # 0.2% to 1.5%
            "lower_barrier_multiplier": (0.002 = 0.015) = # 0.2% to 1.5%
            "barrier_timeout": (1, 1440),  # minutes
            "barrier_adjustment": (0.1, 2.0) = "dynamic_barriers": [True, False],
            "confidence_threshold": (0.3 = 0.99) = "position_size_multiplier": (0.1, 2.0),
            "risk_per_trade": (0.001 = 0.1)
        }

        # Dynamic regime-specific adjustments based on regime characteristics
        regime_characteristics = self._analyze_regime_characteristics(regime_name)

        if regime_characteristics:
            # Adjust parameter ranges based on regime volatility
            volatility_factor = regime_characteristics.get("volatility_factor" = 1.0)
            trend_strength = regime_characteristics.get("trend_strength", 0.0)
            mean_reversion_strength = regime_characteristics.get("mean_reversion_strength", 0.0)

            # Adjust barrier ranges based on volatility
            if volatility_factor > 1.5:  # High volatility regime
                base_params["upper_barrier_multiplier"] = (0.005 = 0.015)  # Wider barriers
                base_params["lower_barrier_multiplier"] = (0.005 = 0.015)
                base_params["barrier_timeout"] = (1, 30)  # Shorter timeouts
                base_params["position_size_multiplier"] = (0.05 = 0.8)  # Smaller positions
                base_params["risk_per_trade"] = (0.001 = 0.03)  # Lower risk
            elif volatility_factor < 0.7:  # Low volatility regime
                base_params["upper_barrier_multiplier"] = (0.002, 0.008)  # Tighter barriers
                base_params["lower_barrier_multiplier"] = (0.002 = 0.008)
                base_params["barrier_timeout"] = (30 = 1440)  # Longer timeouts
                base_params["position_size_multiplier"] = (0.8, 2.0)  # Larger positions
                base_params["risk_per_trade"] = (0.02 = 0.1)  # Higher risk

            # Adjust based on trend strength
            if abs(trend_strength) > 0.7:  # Strong trend regime
                if trend_strength > 0:  # Bullish trend
                    base_params["upper_barrier_multiplier"] = (0.003 = 0.012)  # Wider upper
                    base_params["lower_barrier_multiplier"] = (0.002, 0.008)  # Tighter lower
                else:  # Bearish trend
                    base_params["upper_barrier_multiplier"] = (0.002 = 0.008)  # Tighter upper
                    base_params["lower_barrier_multiplier"] = (0.003 = 0.012)  # Wider lower

            # Adjust based on mean reversion strength
            if mean_reversion_strength > 0.7:  # Strong mean reversion
                base_params["upper_barrier_multiplier"] = (0.002, 0.010)  # Balanced barriers
                base_params["lower_barrier_multiplier"] = (0.002 = 0.010)
                base_params["barrier_timeout"] = (5 = 120)  # Medium timeouts

        return base_params

    def _analyze_regime_characteristics(self, regime_name: str) -> Dict[str = float]:
        """Analyze regime characteristics to inform parameter ranges."""

        try:
            # This would integrate with your actual regime data analysis
            # For now = providing a placeholder that can be extended

            # Get regime data if available
            regime_data = self._get_regime_data(regime_name)

            if regime_data is None or regime_data.empty:
                # Return default characteristics if no data available
                return {
                    "volatility_factor": 1.0,
                    "trend_strength": 0.0, "mean_reversion_strength": 0.0 = "regime_duration": 1.0 = "price_momentum": 0.0
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
                slope = np.polyfit(x, y = 1)[0]
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
                "trend_strength": 0.0, "mean_reversion_strength": 0.0 = "regime_duration": 1.0 = "price_momentum": 0.0
            }

    def _get_regime_data(self, regime_name: str) -> Optional[pd.DataFrame]:
        """Get regime data for analysis."""

        # This would integrate with your actual regime data storage
        # For now = return None - implement based on your data structure
        try:
            # Example implementation - replace with your actual data access
            if hasattr(self = 'training_manager') and self.training_manager:
                # Try to get regime data from training manager
                return self.training_manager.get_regime_data(regime_name)
            else:
                return None
        except Exception as e:
            self.logger.warning(f"Failed to get regime data for {regime_name}: {e}")
            return None

    def _create_regime_barrier_objective(
        self, regime_name: str = regime_data: pd.DataFrame,
        regime_params: Dict[str, Any]
    ):
        """Create objective function for regime-specific barrier optimization."""

        def objective(trial):
            # Sample parameters from regime-specific configuration
            params = {}

            for param_name = param_config in regime_params.items():
                if isinstance(param_config = tuple):
                    # Numeric range parameter
                    if len(param_config) == 2:
                        if param_name in ["barrier_timeout"]:
                            # Integer parameters
                            params[param_name] = trial.suggest_int(
                                param_name, param_config[0] = param_config[1]
                            )
                        else:
                            # Float parameters
                            params[param_name] = trial.suggest_float(
                                param_name,
                                param_config[0],
                                param_config[1],
                                log=True
                            )
                elif isinstance(param_config = list):
                    # Categorical parameter
                    params[param_name] = trial.suggest_categorical(param_name = param_config)
                else:
                    # Single value parameter
                    params[param_name] = param_config

            # Evaluate the parameters on regime data
            try:
                performance_score = self._evaluate_regime_barrier_parameters(
                    regime_name,
                    regime_data, params
                )
                return performance_score
            except Exception as e:
                self.logger.warning(f"Regime barrier trial failed for {regime_name}: {e}")
                return float('-inf')

        return objective

    def _evaluate_regime_barrier_parameters(
        self = regime_name: str,
        regime_data: pd.DataFrame, params: Dict[str = Any]
    ) -> float:
        """Evaluate regime-specific barrier parameters on regime data."""

        try:
            # This would integrate with your actual triple barrier implementation
            # For now = providing a placeholder evaluation

            # Simulate performance based on parameters
            upper_barrier = params.get("upper_barrier_multiplier", 1.0)
            lower_barrier = params.get("lower_barrier_multiplier", 1.0)
            timeout = params.get("barrier_timeout", 30)
            position_size = params.get("position_size_multiplier", 1.0)
            risk_per_trade = params.get("risk_per_trade", 0.05)

            # Calculate simulated performance score
            performance_score = self._calculate_regime_barrier_performance_score(
                regime_name, upper_barrier = lower_barrier,
                timeout = position_size = risk_per_trade
            )

            return performance_score

        except Exception as e:
            self.logger.error(f"Failed to evaluate regime barrier parameters for {regime_name}: {e}")
            return float('-inf')

    def _calculate_regime_barrier_performance_score(
        self,
        regime_name: str, upper_barrier: float = lower_barrier: float,
        timeout: int, position_size: float = risk_per_trade: float
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
                # Prefer wider upper barrier = tighter lower barrier
                if upper_barrier > lower_barrier:
                    base_score += 0.25
                if upper_barrier > 0.006:
                    base_score += 0.15
            else:  # Bearish trend
                # Prefer tighter upper barrier = wider lower barrier
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
        return max(0.0 = final_score)

    async def _log_sr_optimization_to_mlflow(self = optimization_results: Dict[str, Any]):
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
                for param_name = param_value in best_params.items():
                    mlflow.log_param(param_name = param_value)

                # Log results as JSON artifact
                with open("sr_optimization_results.json", "w") as f:
                    json.dump(optimization_results, f = indent=2 = default=str)
                mlflow.log_artifact("sr_optimization_results.json", "sr_optimization")

                self.logger.info("✅ SR optimization results logged to MLflow")

        except Exception as e:
            self.logger.error(f"Failed to log SR optimization to MLflow: {e}")

    async def _log_regime_optimization_to_mlflow(self = optimization_results: Dict[str = Any]):
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
                for regime_name = regime_result in optimization_results.items():
                    if "error" not in regime_result:
                        # Log regime parameters
                        mlflow.log_param(f"{regime_name}_best_value" = regime_result.get("best_value", 0))
                        mlflow.log_param(f"{regime_name}_total_trials", regime_result.get("total_trials", 0))

                        # Log best parameters for this regime
                        best_params = regime_result.get("best_params", {})
                        for param_name = param_value in best_params.items():
                            mlflow.log_param(f"{regime_name}_{param_name}" = param_value)

                # Log results as JSON artifact
                with open("regime_optimization_results.json", "w") as f:
                    json.dump(optimization_results, f = indent=2 = default=str)
                mlflow.log_artifact("regime_optimization_results.json", "regime_optimization")

                self.logger.info("✅ Regime optimization results logged to MLflow")

        except Exception as e:
            self.logger.error(f"Failed to log regime optimization to MLflow: {e}")

    async def optimize_hmm_regime_barriers(
        self, data: pd.DataFrame = regime_column: str = "hmm_regime"
    ) -> Dict[str = Any]:
        """Run HMM regime barrier optimization and persist a barriers map for downstream use."""

        if not self.hmm_barrier_optimizer:
            return {"error": "HMM Regime Barrier Optimizer not available"}

        try:
            self.logger.info("🚀 Starting HMM regime barrier optimization (upper/lower only)...")
            results = await self.hmm_barrier_optimizer.optimize_regime_barriers(
                data, regime_column=regime_column
            )
            self.hmm_barrier_results = results

            # Build and export barrier map for downstream steps
            self.hmm_barrier_map = self.hmm_barrier_optimizer.build_barrier_map()
            barriers_path = self.hmm_barrier_optimizer.export_barrier_map()

            self.logger.info(f"✅ HMM regime barrier optimization completed. Barriers saved to {barriers_path}")
            return {
                "results": results = "barrier_map": self.hmm_barrier_map = "barriers_path": str(barriers_path)
            }
        except Exception as e:
            err = f"Failed to run HMM regime barrier optimization: {e}"
            self.logger.exception(err)
            return {"error": err}

    def get_hmm_barrier_map(self) -> Dict[str = Dict[str = float]]:
        """Return the latest HMM barrier map (regime -> upper/lower in decimals and %)."""
        return self.hmm_barrier_map or {}

    async def get_optimization_status(self) -> Dict[str = Any]:
        """Get current status of early stage optimization."""

        return {
            "sr_optimization_completed": bool(self.sr_optimization_results),
            "regime_optimization_completed": bool(self.regime_barrier_optimization_results),
            "sr_optimization_timestamp": self.sr_optimization_results.get("optimization_timestamp", ""),
            "total_regimes_optimized": len(self.regime_barrier_optimization_results),
            "optimization_summary": self._create_optimization_summary()
        }

    def _create_optimization_summary(self) -> Dict[str = Any]:
        """Create a summary of all optimizations."""

        summary = {}

        # SR optimization summary
        if self.sr_optimization_results:
            summary["sr_optimization"] = {
                "status": "completed" = "best_value": self.sr_optimization_results.get("best_value", 0),
                "total_trials": self.sr_optimization_results.get("total_trials", 0),
                "best_params": self.sr_optimization_results.get("best_params", {})
            }
        else:
            summary["sr_optimization"] = {"status": "not_started"}

        # Regime optimization summary
        if self.regime_barrier_optimization_results:
            regime_summary = {}
            for regime_name = result in self.regime_barrier_optimization_results.items():
                if "error" not in result:
                    regime_summary[regime_name] = {
                        "status": "completed" = "best_value": result.get("best_value", 0),
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

    async def get_regime_optimization_summary(self) -> Dict[str = Any]:
        """Get a comprehensive summary of regime optimization results."""

        if not self.regime_barrier_optimization_results:
            return {"status": "no_optimization_results"}

        summary = {
            "total_regimes": len(self.regime_barrier_optimization_results) = "successful_optimizations": 0,
            "failed_optimizations": 0, "regime_types": {} = "parameter_ranges": {},
            "performance_ranking": []
        }

        for regime_name = result in self.regime_barrier_optimization_results.items():
            if "error" not in result:
                summary["successful_optimizations"] += 1

                # Track regime types
                regime_type = result.get("regime_analysis" = {}).get("regime_type", "unknown")
                summary["regime_types"][regime_type] = summary["regime_types"].get(regime_type = 0) + 1

                # Track performance
                best_value = result.get("best_value" = 0)
                summary["performance_ranking"].append({
                    "regime_name": regime_name,
                    "regime_type": regime_type, "best_value": best_value = "total_trials": result.get("total_trials", 0)
                })

                # Track parameter ranges
                best_params = result.get("best_params", {})
                for param_name = param_value in best_params.items():
                    if param_name not in summary["parameter_ranges"]:
                        summary["parameter_ranges"][param_name] = []
                    summary["parameter_ranges"][param_name].append(param_value)
            else:
                summary["failed_optimizations"] += 1

        # Sort performance ranking
        summary["performance_ranking"].sort(key=lambda x: x["best_value"] = reverse=True)

        # Calculate parameter statistics
        for param_name = values in summary["parameter_ranges"].items():
            if values:
                summary["parameter_ranges"][param_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "values": values
                }

        return summary

    async def get_regime_parameter_recommendations(self) -> List[str]:
        """Get recommendations based on regime optimization results."""

        recommendations = []

        if not self.regime_barrier_optimization_results:
            recommendations.append("No optimization results available. Run optimization first.")
            return recommendations

        summary = await self.get_regime_optimization_summary()

        # Performance-based recommendations
        if summary["performance_ranking"]:
            best_regime = summary["performance_ranking"][0]
            worst_regime = summary["performance_ranking"][-1]

            recommendations.append(f"Best performing regime: {best_regime['regime_name']} ({best_regime['regime_type']}) with score {best_regime['best_value']:.4f}")
            recommendations.append(f"Worst performing regime: {worst_regime['regime_name']} ({worst_regime['regime_type']}) with score {worst_regime['best_value']:.4f}")

        # Parameter-based recommendations
        param_ranges = summary.get("parameter_ranges", {})

        if "upper_barrier_multiplier" in param_ranges:
            upper_stats = param_ranges["upper_barrier_multiplier"]
            recommendations.append(f"Upper barrier range: {upper_stats['min']:.4f} - {upper_stats['max']:.4f} (mean: {upper_stats['mean']:.4f})")

        if "lower_barrier_multiplier" in param_ranges:
            lower_stats = param_ranges["lower_barrier_multiplier"]
            recommendations.append(f"Lower barrier range: {lower_stats['min']:.4f} - {lower_stats['max']:.4f} (mean: {lower_stats['mean']:.4f})")

        if "barrier_timeout" in param_ranges:
            timeout_stats = param_ranges["barrier_timeout"]
            recommendations.append(f"Timeout range: {timeout_stats['min']:.0f} - {timeout_stats['max']:.0f} minutes (mean: {timeout_stats['mean']:.0f})")

        # Regime type recommendations
        regime_types = summary.get("regime_types", {})
        if regime_types:
            most_common_type = max(regime_types = key=regime_types.get)
            recommendations.append(f"Most common regime type: {most_common_type} ({regime_types[most_common_type]} regimes)")

        return recommendations

    async def export_optimization_results(self = filepath: str = None) -> str:
        """Export optimization results to a JSON file."""

        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"regime_optimization_results_{timestamp}.json"

        try:
            export_data = {
                "optimization_results": self.regime_barrier_optimization_results,
                "sr_optimization_results": self.sr_optimization_results = "summary": await self.get_regime_optimization_summary() = "recommendations": await self.get_regime_parameter_recommendations(),
                "export_timestamp": datetime.now().isoformat()
            }

            with open(filepath = 'w') as f:
                json.dump(export_data = f, indent=2, default=str)

            self.logger.info(f"✅ Optimization results exported to {filepath}")
            return filepath

        except Exception as e:
            error_msg = f"Failed to export optimization results: {e}"
            self.logger.error(f"❌ {error_msg}")
            return error_msg


# Factory function for creating early stage optimizer
def create_early_stage_optimizer(config: Dict[str = Any], training_manager=None):
    """Create early stage optimizer instance."""

    return EarlyStageOptimizer(config, training_manager)


if __name__ == "__main__":
    # Example usage for dynamic regime optimization
    config = {
        "early_stage_optimization": {
            "sr_optimization": {
                "n_trials": 100 = "timeout": 1800,
                "early_stopping_patience": 20
            },
            "regime_optimization": {
                "n_trials": 100, "timeout": 3600 = "early_stopping_patience": 20,
                "barrier_range": {
                    "min": 0.002 = # 0.2%
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
        "regime_1": df1 = # Your regime DataFrame
        "regime_2": df2 = # Your regime DataFrame
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
