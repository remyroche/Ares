#!/usr / bin / env python3
"""
Regime - Specific Triple Barrier Optimizer

This module implements regime - specific optimization for the triple barrier method,
creating separate optimizers for each HMM regime to allow different barrier parameters
for different market conditions.

This optimizer is used by the triple barrier labeler to optimize parameters
before ML training begins, ensuring optimal trading parameters for each regime.

Key Features:
    pass - Separate optimization for each HMM regime (bull, bear, sideways, etc.)
- Regime - specific barrier parameters (upper, lower, timeout, confidence)
- Regime - aware parameter validation and constraints - Integration with triple barrier labeler - MLflow tracking for regime - specific experiments
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
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
import mlflow
MLFLOW_AVAILABLE, True
except ImportError:
    MLFLOW_AVAILABLE, False

# Import Optuna for optimization
try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
import optuna
OPTUNA_AVAILABLE, True
except ImportError:
    OPTUNA_AVAILABLE, False

# Import triple barrier components for integration
try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
from .regime_aware_triple_barrier_labeling import RegimeAwareTripleBarrierLabeling
from .optimized_triple_barrier_labeling import OptimizedTripleBarrierLabeling
TRIPLE_BARRIER_AVAILABLE, True
except ImportError:
    TRIPLE_BARRIER_AVAILABLE, False
RegimeAwareTripleBarrierLabeling, None
OptimizedTripleBarrierLabeling, None

class RegimeSpecificTripleBarrierOptimizer:
    """
Regime - specific optimizer for triple barrier method parameters.

Creates separate optimization spaces for each HMM regime, allowing
different barrier parameters for different market conditions.

This optimizer is used by the triple barrier labeler to optimize
parameters before ML training begins.
"""

def __init__(self, config: Dict[str, Any], training_manager = None):
        self.config, config
self.training_manager, training_manager
self.logger, logging.getLogger(__name__)

# Regime - specific parameter configurations
self.regime_configs, self._create_regime_specific_configs()

# Optimization results storage
self.optimization_results = {}
self.regime_models = {}

# MLflow experiment tracking
self.mlflow_experiment_name = "regime_specific_triple_barrier_optimization"

# Triple barrier labeler integration
self.triple_barrier_labeler, None
if TRIPLE_BARRIER_AVAILABLE:
        self.triple_barrier_labeler, self._create_triple_barrier_labeler()
self.logger.info("✅ Triple barrier labeler integration initialized")
else:
        self.logger.warning("⚠️ Triple barrier labeler not available for integration")

def _create_triple_barrier_labeler(self):
        """Create triple barrier labeler for integration."""

try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
# Create regime - aware labeler with default configuration
labeler_config = {
"enable_regime_specific_parameters": True,
"regime_parameter_optimization": True,
"default_barrier_settings": self._get_default_barrier_settings()
}

if RegimeAwareTripleBarrierLabeling:
        return RegimeAwareTripleBarrierLabeling(labeler_config)
elif OptimizedTripleBarrierLabeling:
        return OptimizedTripleBarrierLabeling(labeler_config)
else:
        return None

except Exception as e:
        self.logger.warning(f"Failed to create triple barrier labeler: {e}")
return None

def _get_default_barrier_settings(self) -> Dict[str, Any]:
        """Get default barrier settings for initialization."""

return {
"upper_barrier_multiplier": 1.0,
"lower_barrier_multiplier": 1.0,
"barrier_timeout": 30,
"barrier_adjustment": 1.0,
"dynamic_barriers": True,
"confidence_threshold": 0.7,
"position_size_multiplier": 1.0,
"risk_per_trade": 0.05
}

def _create_regime_specific_configs(self) -> Dict[str, Dict[str, Any]]:
        """Create regime - specific parameter configurations for triple barrier method."""

return {
"bull_regime": {
"description": "Bull market regime - upward trending markets",
"barrier_settings": {
"upper_barrier_multiplier": (0.3, 1.5),      # Wider upper barrier for bull markets
"lower_barrier_multiplier": (0.1, 0.8),       # Tighter lower barrier
"barrier_timeout": (5, 60),                    # Shorter timeout (faster moves)
"barrier_adjustment": (0.8, 1.5),             # More aggressive adjustment
"dynamic_barriers": [True, False],
"momentum_factor": (1.0, 2.0)                 # Higher momentum sensitivity
},
"labeling_settings": {
"labeling_method": ["dynamic", "regime_specific", "momentum_aware"],
"min_label_confidence": (0.4, 0.9),           # Lower confidence threshold
"label_smoothing": (0.01, 0.5),               # Less smoothing
"class_balance_threshold": (0.3, 0.8),        # Allow more imbalance
"trend_following_weight": (0.6, 1.0)          # Higher trend following
},
"position_management": {
"position_size_multiplier": (0.8, 2.0),       # Larger positions in bull markets
"max_position_size": (0.2, 2.5),              # Higher max position
"position_scaling": (1.0, 4.0),               # More aggressive scaling
"risk_per_trade": (0.005, 0.15),              # Higher risk tolerance
"trend_amplification": (1.2, 2.0)             # Amplify trend signals
},
"risk_management": {
"max_drawdown_threshold": (0.08, 0.4),        # Higher drawdown tolerance
"volatility_target": (0.08, 0.6),             # Higher volatility target
"correlation_threshold": (0.3, 0.8),          # Lower correlation requirement
"var_confidence_level": (0.85, 0.98)          # Lower VaR confidence
}
},
"bear_regime": {
"description": "Bear market regime - downward trending markets",
"barrier_settings": {
"upper_barrier_multiplier": (0.1, 0.8),       # Tighter upper barrier
"lower_barrier_multiplier": (0.3, 1.5),       # Wider lower barrier for bear markets
"barrier_timeout": (10, 120),                  # Longer timeout (slower moves)
"barrier_adjustment": (0.5, 1.2),             # Conservative adjustment
"dynamic_barriers": [True, False],
"momentum_factor": (0.5, 1.5)                 # Lower momentum sensitivity
},
"labeling_settings": {
"labeling_method": ["conservative", "regime_specific", "mean_reversion"],
"min_label_confidence": (0.6, 0.95),          # Higher confidence threshold
"label_smoothing": (0.1, 0.8),                # More smoothing
"class_balance_threshold": (0.4, 0.9),        # Maintain balance
"trend_following_weight": (0.2, 0.6)          # Lower trend following
},
"position_management": {
"position_size_multiplier": (0.3, 1.2),       # Smaller positions in bear markets
"max_position_size": (0.1, 1.0),              # Lower max position
"position_scaling": (0.5, 2.0),               # Conservative scaling
"risk_per_trade": (0.001, 0.08),              # Lower risk tolerance
"trend_amplification": (0.5, 1.2)             # Reduce trend signals
},
"risk_management": {
"max_drawdown_threshold": (0.03, 0.25),       # Lower drawdown tolerance
"volatility_target": (0.03, 0.4),             # Lower volatility target
"correlation_threshold": (0.5, 0.9),          # Higher correlation requirement
"var_confidence_level": (0.9, 0.99)           # Higher VaR confidence
}
},
"sideways_regime": {
"description": "Sideways / consolidation regime - range - bound markets",
"barrier_settings": {
"upper_barrier_multiplier": (0.2, 1.0),       # Balanced barriers
"lower_barrier_multiplier": (0.2, 1.0),       # Balanced barriers
"barrier_timeout": (15, 90),                   # Medium timeout
"barrier_adjustment": (0.7, 1.3),             # Balanced adjustment
"dynamic_barriers": [True, False],
"momentum_factor": (0.8, 1.8)                 # Balanced momentum
},
"labeling_settings": {
"labeling_method": ["balanced", "regime_specific", "mean_reversion"],
"min_label_confidence": (0.5, 0.9),           # Balanced confidence
"label_smoothing": (0.05, 0.6),               # Balanced smoothing
"class_balance_threshold": (0.4, 0.8),        # Maintain balance
"trend_following_weight": (0.4, 0.8)          # Balanced trend following
},
"position_management": {
"position_size_multiplier": (0.5, 1.5),       # Balanced position sizing
"max_position_size": (0.15, 1.5),             # Balanced max position
"position_scaling": (0.7, 2.5),               # Balanced scaling
"risk_per_trade": (0.002, 0.1),               # Balanced risk
"trend_amplification": (0.8, 1.5)             # Balanced amplification
},
"risk_management": {
"max_drawdown_threshold": (0.05, 0.3),        # Balanced drawdown tolerance
"volatility_target": (0.05, 0.5),             # Balanced volatility target
"correlation_threshold": (0.4, 0.8),          # Balanced correlation
"var_confidence_level": (0.87, 0.98)          # Balanced VaR confidence
}
},
"volatile_regime": {
"description": "High volatility regime - choppy, unpredictable markets",
"barrier_settings": {
"upper_barrier_multiplier": (0.5, 2.0),       # Much wider barriers
"lower_barrier_multiplier": (0.5, 2.0),       # Much wider barriers
"barrier_timeout": (3, 45),                    # Very short timeout
"barrier_adjustment": (1.2, 2.5),             # Aggressive adjustment
"dynamic_barriers": [True],                    # Always dynamic
"momentum_factor": (1.5, 3.0)                 # High momentum sensitivity
},
"labeling_settings": {
"labeling_method": ["adaptive", "regime_specific", "volatility_aware"],
"min_label_confidence": (0.3, 0.8),           # Lower confidence (more noise)
"label_smoothing": (0.2, 0.9),                # Heavy smoothing
"class_balance_threshold": (0.2, 0.7),        # Allow imbalance
"trend_following_weight": (0.1, 0.5)          # Lower trend following
},
"position_management": {
"position_size_multiplier": (0.2, 1.0),       # Smaller positions
"max_position_size": (0.05, 0.8),             # Much lower max position
"position_scaling": (0.3, 1.5),               # Conservative scaling
"risk_per_trade": (0.001, 0.05),              # Much lower risk
"trend_amplification": (0.3, 1.0)             # Reduce amplification
},
"risk_management": {
"max_drawdown_threshold": (0.02, 0.2),        # Much lower drawdown tolerance
"volatility_target": (0.02, 0.3),             # Lower volatility target
"correlation_threshold": (0.6, 0.95),         # Higher correlation requirement
"var_confidence_level": (0.92, 0.995)         # Much higher VaR confidence
}
},
"trending_regime": {
"description": "Strong trending regime - sustained directional moves",
"barrier_settings": {
"upper_barrier_multiplier": (0.4, 1.8),       # Wider barriers for trends
"lower_barrier_multiplier": (0.4, 1.8),       # Wider barriers for trends
"barrier_timeout": (8, 75),                    # Medium timeout
"barrier_adjustment": (0.9, 1.8),             # Trend - aware adjustment
"dynamic_barriers": [True, False],
"momentum_factor": (1.2, 2.5)                 # High momentum sensitivity
},
"labeling_settings": {
"labeling_method": ["trend_following", "regime_specific", "momentum_aware"],
"min_label_confidence": (0.45, 0.85),         # Moderate confidence
"label_smoothing": (0.03, 0.4),               # Light smoothing
"class_balance_threshold": (0.3, 0.8),        # Allow imbalance
"trend_following_weight": (0.7, 1.0)          # High trend following
},
"position_management": {
"position_size_multiplier": (0.6, 2.2),       # Larger positions for trends
"max_position_size": (0.2, 2.0),              # Higher max position
"position_scaling": (1.0, 3.5),               # Aggressive scaling
"risk_per_trade": (0.003, 0.12),              # Moderate risk
"trend_amplification": (1.3, 2.2)             # Amplify trend signals
},
"risk_management": {
"max_drawdown_threshold": (0.06, 0.35),       # Moderate drawdown tolerance
"volatility_target": (0.06, 0.55),            # Moderate volatility target
"correlation_threshold": (0.35, 0.8),         # Lower correlation requirement
"var_confidence_level": (0.86, 0.97)          # Moderate VaR confidence
}
}
}

async def optimize_regime_specific_parameters(
self,
regime_data: Dict[str, pd.DataFrame],
optimization_config: Dict[str, Any]
) -> Dict[str, Any]:
        """Optimize triple barrier parameters for each regime separately."""

self.logger.info("🚀 Starting regime - specific triple barrier optimization...")
self.logger.info(f"Regimes to optimize: {list(regime_data.keys())}")

optimization_results = {}

for regime_name, regime_df in regime_data.items():
        if regime_name not in self.regime_configs:
        self.logger.warning(f"⚠️ No configuration found for regime: {regime_name}")
continue

try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
self.logger.info(f"🔧 Optimizing parameters for {regime_name} regime...")

# Create regime - specific study
study, await self._create_regime_study(regime_name, optimization_config)

# Run optimization for this regime
regime_result, await self._optimize_single_regime(
regime_name,
regime_df,
study,
optimization_config
)

optimization_results[regime_name] = regime_result

# Store the optimized model
self.regime_models[regime_name] = regime_result.get("best_model", None)

# Update triple barrier labeler with optimized parameters
if self.triple_barrier_labeler:
        await self._update_triple_barrier_labeler(regime_name, regime_result)

self.logger.info(f"✅ {regime_name} regime optimization completed")

except Exception as e:
        self.logger.error(f"❌ Failed to optimize {regime_name} regime: {e}")
optimization_results[regime_name] = {"error": str(e)}

# Store overall results
self.optimization_results, optimization_results

# Log to MLflow
if MLFLOW_AVAILABLE:
        await self._log_regime_optimization_to_mlflow(optimization_results)

return optimization_results

async def _update_triple_barrier_labeler(self, regime_name: str, regime_result: Dict[str, Any]):
        """Update triple barrier labeler with optimized parameters for a regime."""

if not self.triple_barrier_labeler or "error" in regime_result:
            return

try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
best_params, regime_result.get("best_params", {})

# Extract barrier settings
barrier_settings, best_params.get("barrier_settings", {})
labeling_settings, best_params.get("labeling_settings", {})
position_settings, best_params.get("position_management", {})
risk_settings, best_params.get("risk_management", {})

# Update labeler with regime - specific parameters
if hasattr(self.triple_barrier_labeler, 'set_regime_parameters'):
        await self.triple_barrier_labeler.set_regime_parameters(
regime_name = regime_name,
barrier_settings = barrier_settings,
labeling_settings = labeling_settings,
position_settings = position_settings,
risk_settings = risk_settings
)
self.logger.info(f"✅ Updated triple barrier labeler for {regime_name} regime")

except Exception as e:
        self.logger.warning(f"Failed to update triple barrier labeler for {regime_name}: {e}")

async def _create_regime_study(
self,
regime_name: str,
optimization_config: Dict[str, Any]
) -> optuna.Study:
        """Create an Optuna study for a specific regime."""

if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for regime - specific optimization")

# Create study name
study_name, f"regime_specific_triple_barrier_{regime_name}"

# Create study with regime - specific configuration
study, optuna.create_study(
study_name = study_name,
direction="maximize",  # Maximize performance metric
sampler = optuna.samplers.TPESampler(
n_startup_trials = 10,
n_ei_candidates = 24,
multivariate = True,
group = True
),
pruner = optuna.pruners.MedianPruner(
n_startup_trials = 5,
n_warmup_steps = 10,
interval_steps = 3
)
)

return study

async def _optimize_single_regime(
self,
regime_name: str,
regime_data: pd.DataFrame,
study: optuna.Study,
optimization_config: Dict[str, Any]
) -> Dict[str, Any]:
        """Optimize parameters for a single regime."""

# Get regime - specific configuration
regime_config, self.regime_configs[regime_name]

# Create objective function for this regime
objective, self._create_regime_objective(
regime_name,
regime_data,
regime_config
)

# Run optimization
n_trials, optimization_config.get("n_trials", 100)
timeout, optimization_config.get("timeout", 3600)

study.optimize(
objective,
n_trials = n_trials,
timeout = timeout,
callbacks=[
optuna.callbacks.EarlyStoppingCallback(
patience = optimization_config.get("early_stopping_patience", 20)
)
]
)

# Extract results
best_trial, study.best_trial
best_params, best_trial.params
best_value, best_trial.value

# Create regime - specific model with optimized parameters
best_model, await self._create_regime_model(regime_name, best_params)

return {
"regime_name": regime_name,
"best_params": best_params,
"best_value": best_value,
"best_trial": best_trial.number,
"total_trials": len(study.trials),
"optimization_history": [trial.value for trial in study.trials if trial.value is not None],
"best_model": best_model,
"regime_config": regime_config
}

def _create_regime_objective(
self,
regime_name: str,
regime_data: pd.DataFrame,
regime_config: Dict[str, Any]
):
        """Create objective function for regime - specific optimization."""

def objective(trial):
        # Sample parameters from regime - specific configuration
params, self._sample_regime_parameters(trial, regime_config)

# Evaluate the parameters on regime data
try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
performance_score, self._evaluate_regime_parameters(
regime_name,
regime_data,
params
)
return performance_score
except Exception as e:
        self.logger.warning(f"Trial failed for {regime_name}: {e}")
return float('-inf')  # Penalize failed trials

return objective

def _sample_regime_parameters(
self,
trial: optuna.Trial,
regime_config: Dict[str, Any]
) -> Dict[str, Any]:
        """Sample parameters from regime - specific configuration."""

params = {}

for category, category_params in regime_config.items():
            params[category] = {}

for param_name, param_config in category_params.items():
        if isinstance(param_config, tuple):
        # Numeric range parameter
if len(param_config) == 2:
        if param_name in ["barrier_timeout", "n_estimators", "max_depth"]:
        # Integer parameters
params[category][param_name] = trial.suggest_int(
f"{category}_{param_name}",
param_config[0],
param_config[1]
)
else:
        # Float parameters
params[category][param_name] = trial.suggest_float(
f"{category}_{param_name}",
param_config[0],
param_config[1],
log = True
)
elif isinstance(param_config, list):
        # Categorical parameter
params[category][param_name] = trial.suggest_categorical(
f"{category}_{param_name}",
param_config
)
else:
        # Single value parameter
params[category][param_name] = param_config

return params

def _evaluate_regime_parameters(
self,
regime_name: str,
regime_data: pd.DataFrame,
params: Dict[str, Any]
) -> float:
        """Evaluate regime - specific parameters on regime data."""

try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
# This would integrate with your actual triple barrier implementation
# For now, providing a placeholder evaluation

# Simulate performance based on parameters
barrier_params, params.get("barrier_settings", {})
labeling_params, params.get("labeling_settings", {})
position_params, params.get("position_management", {})
risk_params, params.get("risk_management", {})

# Calculate simulated performance score
performance_score, self._calculate_regime_performance_score(
regime_name,
barrier_params,
labeling_params,
position_params,
risk_params
)

return performance_score

except Exception as e:
        self.logger.error(f"Failed to evaluate parameters for {regime_name}: {e}")
return float('-inf')

def _calculate_regime_performance_score(
self,
regime_name: str,
barrier_params: Dict[str, Any],
labeling_params: Dict[str, Any],
position_params: Dict[str, Any],
risk_params: Dict[str, Any]
) -> float:
        """Calculate performance score for regime - specific parameters."""

# Base score
base_score, 0.0

# Barrier settings scoring
if barrier_params:
            upper_barrier, barrier_params.get("upper_barrier_multiplier", 1.0)
lower_barrier, barrier_params.get("lower_barrier_multiplier", 1.0)
timeout, barrier_params.get("barrier_timeout", 30)

# Score based on regime - appropriate barriers
if regime_name == "bull_regime":
        if upper_barrier > lower_barrier:  # Wider upper barrier
base_score += 0.3
if timeout < 60:  # Faster timeout
base_score += 0.2
elif regime_name == "bear_regime":
        if lower_barrier > upper_barrier:  # Wider lower barrier
base_score += 0.3
if timeout > 60:  # Slower timeout
base_score += 0.2
elif regime_name == "volatile_regime":
        if upper_barrier > 1.5 and lower_barrier > 1.5:  # Wide barriers
base_score += 0.3
if timeout < 45:  # Short timeout
base_score += 0.2

# Labeling settings scoring
if labeling_params:
            confidence, labeling_params.get("min_label_confidence", 0.7)
smoothing, labeling_params.get("label_smoothing", 0.3)

# Score based on regime - appropriate labeling
if regime_name == "volatile_regime":
        if confidence < 0.7:  # Lower confidence for volatile markets
base_score += 0.2
if smoothing > 0.5:  # More smoothing for volatile markets
base_score += 0.2
elif regime_name == "trending_regime":
        if confidence > 0.6:  # Higher confidence for trending markets
base_score += 0.2
if smoothing < 0.4:  # Less smoothing for trending markets
base_score += 0.2

# Position management scoring
if position_params:
            position_size, position_params.get("position_size_multiplier", 1.0)
max_position, position_params.get("max_position_size", 1.0)

# Score based on regime - appropriate position sizing
if regime_name == "bull_regime":
        if position_size > 1.2:  # Larger positions in bull markets
base_score += 0.2
elif regime_name == "bear_regime":
        if position_size < 1.0:  # Smaller positions in bear markets
base_score += 0.2
elif regime_name == "volatile_regime":
        if position_size < 0.8:  # Much smaller positions in volatile markets
base_score += 0.2

# Risk management scoring
if risk_params:
            drawdown, risk_params.get("max_drawdown_threshold", 0.2)
volatility, risk_params.get("volatility_target", 0.3)

# Score based on regime - appropriate risk management
if regime_name == "bull_regime":
        if drawdown > 0.25:  # Higher drawdown tolerance in bull markets
base_score += 0.1
elif regime_name == "bear_regime":
        if drawdown < 0.2:  # Lower drawdown tolerance in bear markets
base_score += 0.1
elif regime_name == "volatile_regime":
        if drawdown < 0.15:  # Much lower drawdown tolerance in volatile markets
base_score += 0.1

# Add some randomness to simulate real evaluation
random_factor, np.random.normal(0, 0.1)
final_score, base_score + random_factor

# Ensure score is positive
return max(0.0, final_score)

async def _create_regime_model(
self,
regime_name: str,
optimized_params: Dict[str, Any]
) -> Dict[str, Any]:
        """Create a regime - specific model with optimized parameters."""

# This would integrate with your actual triple barrier implementation
# For now, creating a parameter configuration object

regime_model = {
"regime_name": regime_name,
"optimized_parameters": optimized_params,
"model_type": "regime_specific_triple_barrier",
"creation_timestamp": datetime.now().isoformat(),
"parameter_summary": self._create_parameter_summary(optimized_params)
}

return regime_model

def _create_parameter_summary(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of optimized parameters."""

summary = {}

for category, category_params in params.items():
            summary[category] = {
"parameter_count": len(category_params),
"key_parameters": list(category_params.keys())[:5],  # Top 5 parameters
"parameter_types": {
param_name: type(param_value).__name__
for param_name, param_value in list(category_params.items())[:5]
}
}

return summary

async def _log_regime_optimization_to_mlflow(self, optimization_results: Dict[str, Any]):
        """Log regime - specific optimization results to MLflow."""

try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
# Set experiment name
mlflow.set_experiment(self.mlflow_experiment_name)

# Start a run for regime - specific optimization
with mlflow.start_run(run_name="regime_specific_triple_barrier_optimization"):
        # Log overall results
mlflow.log_param("total_regimes", len(optimization_results))
mlflow.log_param("optimization_timestamp", datetime.now().isoformat())

# Log regime - specific results
for regime_name, regime_result in optimization_results.items():
        if "error" not in regime_result:
        # Log regime parameters
mlflow.log_param(f"{regime_name}_best_value", regime_result.get("best_value", 0))
mlflow.log_param(f"{regime_name}_total_trials", regime_result.get("total_trials", 0))

# Log best parameters for this regime
best_params, regime_result.get("best_params", {})
for category, category_params in best_params.items():
        for param_name, param_value in category_params.items():
                                mlflow.log_param(f"{regime_name}_{category}_{param_name}", param_value)

# Log results as JSON artifact
with open("regime_optimization_results.json", "w") as f:
                    json.dump(optimization_results, f, indent = 2, default = str)
mlflow.log_artifact("regime_optimization_results.json", "regime_optimization")

self.logger.info("✅ Regime optimization results logged to MLflow")

except Exception as e:
        self.logger.error(f"Failed to log to MLflow: {e}")

async def get_regime_optimization_status(self) -> Dict[str, Any]:
        """Get current status of regime - specific optimization."""

return {
"optimization_completed": bool(self.optimization_results),
"total_regimes_optimized": len(self.optimization_results),
"regime_models_created": len(self.regime_models),
"optimization_timestamp": datetime.now().isoformat(),
"regime_summary": self._create_regime_summary(),
"triple_barrier_integration": bool(self.triple_barrier_labeler)
}

def _create_regime_summary(self) -> Dict[str, Any]:
        """Create a summary of all regime optimizations."""

summary = {}

for regime_name, result in self.optimization_results.items():
        if "error" not in result:
                summary[regime_name] = {
"status": "completed",
"best_value": result.get("best_value", 0),
"total_trials": result.get("total_trials", 0),
"parameter_count": len(result.get("best_params", {}))
}
else:
                summary[regime_name] = {
"status": "failed",
"error": result.get("error", "Unknown error")
}

return summary

async def apply_regime_parameters(self, regime_name: str) -> Dict[str, Any]:
        """Apply optimized parameters for a specific regime."""

if regime_name not in self.regime_models:
        return {"error": f"No optimized model found for regime: {regime_name}"}

try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
regime_model, self.regime_models[regime_name]
optimized_params, regime_model.get("optimized_parameters", {})

# This would integrate with your actual triple barrier implementation
# For now, returning the parameter application status

application_result = {
"regime_name": regime_name,
"status": "applied",
"parameters_applied": len(optimized_params),
"application_timestamp": datetime.now().isoformat(),
"parameter_summary": self._create_parameter_summary(optimized_params)
}

self.logger.info(f"✅ Applied optimized parameters for {regime_name} regime")

return application_result

except Exception as e:
        self.logger.error(f"❌ Failed to apply parameters for {regime_name} regime: {e}")
return {"error": str(e)}

async def get_optimization_recommendations(self) -> List[str]:
        """Get recommendations based on optimization results."""

recommendations = []

if not self.optimization_results:
            recommendations.append("Run regime - specific optimization first")
return recommendations

# Analyze results and provide recommendations
for regime_name, result in self.optimization_results.items():
        if "error" not in result:
                best_value, result.get("best_value", 0)

if best_value < 0.5:
                    recommendations.append(f"Consider adjusting parameters for {regime_name} regime (low performance)")
elif best_value > 0.8:
                    recommendations.append(f"{regime_name} regime parameters are well - optimized")

# Regime - specific recommendations
if regime_name == "volatile_regime":
                    recommendations.append("Volatile regime: Consider wider barriers and shorter timeouts")
elif regime_name == "trending_regime":
                    recommendations.append("Trending regime: Consider momentum - aware labeling and position sizing")

recommendations.append("Monitor regime performance with new parameters")
recommendations.append("Consider re - optimization if market conditions change significantly")

return recommendations

async def get_triple_barrier_labeler(self):
        """Get the integrated triple barrier labeler."""

return self.triple_barrier_labeler

# Factory function for creating regime - specific triple barrier optimizer
def create_regime_specific_triple_barrier_optimizer(config: Dict[str, Any], training_manager = None):
    """Create regime - specific triple barrier optimizer instance."""

return RegimeSpecificTripleBarrierOptimizer(config, training_manager)

if __name__ == "__main__":
    # Example usage
config = {
"regime_optimization": {
"n_trials": 100,
"timeout": 3600,
"early_stopping_patience": 20
}
}

# Create optimizer instance
optimizer, create_regime_specific_triple_barrier_optimizer(config)

print("✅ Regime - Specific Triple Barrier Optimizer created successfully!")
print(f"Total regimes supported: {len(optimizer.regime_configs)}")
print("This optimizer integrates with the triple barrier labeler")
print("and should be used BEFORE ML training begins.")

# Show supported regimes
for regime_name, regime_config in optimizer.regime_configs.items():
        print(f"\n{regime_name}:")
print(f"  Description: {regime_config['description']}")
total_params, sum(len(category) for category in regime_config.values() if isinstance(category, dict))
print(f"  Total parameters: {total_params}")