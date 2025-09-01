"""Tactician module for trading strategy execution."""

from datetime import datetime
from typing import Any, Dict
import numpy as np
import pandas as pd

from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import failed, invalid, missing

class Tactician:
    passpass"""
Refactored Tactician component with modular architecture and enhanced scenario-based predictions.
This module orchestrates the tactics pipeline using specialized managers and integrates
fractal scenario analysis with comprehensive technical indicators.
"""

def __init__(...) -> ...:
    pass"""..."""
    passself.config: dict[str, Any] = config
self.logger = system_logger.getChild("Tactician")

# Tactician state
self.is_running: bool = False
self.status: dict[str, Any] = {}
self.history: list[dict[str, Any]] = []
self.tactics_results: dict[str, Any] = {}

# Configuration
self.tactician_config: dict[str, Any] = self.config.get("tactician", {})
self.tactics_interval: int = self.tactician_config.get("tactics_interval", 30)
self.max_history: int = self.tactician_config.get("max_history", 100)

# Component managers (will be initialized)
self.tactics_orchestrator = None
self.position_sizer = None
self.leverage_sizer = None
self.position_division_strategy = None

# Enhanced scenario-based predictor
self.scenario_predictor = None

# Enhanced predictions from supervisor
self.enable_enhanced_predictions: bool = self.tactician_config.get(
"enable_enhanced_predictions",
True,
)

# Decision thresholds (configurable for step17 optimization)
step17_config = config.get("step17_optimization", {})
tactician_config = step17_config.get("fully_migrated_tactician", {})
self.decision_thresholds = {
"entry_profit_threshold": tactician_config.get("entry_profit_threshold", 0.6),
"entry_risk_threshold": tactician_config.get("entry_risk_threshold", 0.2),
"entry_confidence_threshold": tactician_config.get("entry_confidence_threshold", 0.7),
"entry_profit_risk_ratio": tactician_config.get("entry_profit_risk_ratio", 2.0),
"entry_scenario_dominance": tactician_config.get("entry_scenario_dominance", 0.4),
"exit_risk_threshold": tactician_config.get("exit_risk_threshold", 0.5),
"exit_confidence_drop": tactician_config.get("exit_confidence_drop", 0.2),
"position_size_multiplier": tactician_config.get("position_size_multiplier", 1.0),
"leverage_multiplier": tactician_config.get("leverage_multiplier", 1.0)
}



# Performance tracking
self.performance_metrics = {
"total_trades": 0,
"winning_trades": 0,
"losing_trades": 0,
"total_profit": 0.0,
"total_loss": 0.0,
"max_drawdown": 0.0,
"sharpe_ratio": 0.0,
"profit_factor": 0.0,
"win_rate": 0.0
}

# State management
self.is_initialized = False
self.current_position = None
self.position_history = []

@handle_specific_errors(
error_handlers={
ValueError: (False, "Invalid tactician configuration"),
AttributeError: (False, "Missing required tactician parameters"),
KeyError: (False, "Missing configuration keys"),
},
default_return=False,
context="tactician initialization",
)
async def initialize(...) -> ...:
    """..."""
    passtry:
    passexcept Exception as e:
    passpasspasspasspasspasspassself.logger.info("Initializing Refactored Tactician...")

# Initialize component managers
await self._initialize_component_managers()

# Validate configuration
if not self._validate_configuration():
    passself.logger.error(invalid("Invalid configuration for tactician"))
return False

self.logger.info("✅ Refactored Tactician initialized successfully")
return True

except Exception as e:
    passpasspasspasspasspasspasspassself.logger.error(failed(f"❌ Refactored Tactician initialization failed: {e}"))
return False

@handle_errors(
exceptions=(ValueError, AttributeError),
default_return=None,
context="component managers initialization",
)
async def _initialize_component_managers(...) -> ...:
    """..."""
    passtry:
    passexcept Exception as e:
    passpasspasspasspasspasspass# Initialize tactics orchestrator
from .tactics_orchestrator import TacticsOrchestrator
self.tactics_orchestrator = TacticsOrchestrator(self.config)
await self.tactics_orchestrator.initialize()

# Initialize position sizer
from src.tactician.position_sizer import PositionSizer
self.position_sizer = PositionSizer(self.config)
await self.position_sizer.initialize()

# Initialize leverage sizer
from src.tactician.leverage_sizer import LeverageSizer
self.leverage_sizer = LeverageSizer(self.config)
await self.leverage_sizer.initialize()

# Initialize position division strategy
from src.tactician.position_division_strategy import PositionDivisionStrategy
self.position_division_strategy = PositionDivisionStrategy(self.config)
await self.position_division_strategy.initialize()

# Initialize enhanced scenario predictor
from .enhanced_scenario_based_predictor import EnhancedScenarioBasedPredictor
self.scenario_predictor = EnhancedScenarioBasedPredictor(self.config)
success = await self.scenario_predictor.initialize()
if not success:
    passself.logger.error("Failed to initialize enhanced scenario predictor")
    raise Exception("Enhanced scenario predictor initialization failed")

self.logger.info("✅ All component managers initialized")

except Exception as e:
    passpasspasspasspasspasspassself.logger.error(failed(f"❌ Failed to initialize component managers: {e}"))
raise

@handle_errors(
exceptions=(ValueError, AttributeError),
default_return=False,
context="configuration validation",
)
def _validate_configuration(...) -> ...:
    """..."""
    passtry:
    passexcept Exception as e:
    passpasspasspasspasspasspass# Validate required configuration sections
required_sections = ["tactician", "tactics_orchestrator"]

for section in required_sections:
    passif section not in self.config:
    passself.logger.error(
f"Missing required configuration section: {section}",
)
return False

# Validate tactician specific settings
if self.tactics_interval <= 0:
    passself.logger.error(invalid("Invalid tactics_interval configuration"))
return False

if self.max_history <= 0:
    passself.logger.error(invalid("Invalid max_history configuration"))
return False

return True

except Exception as e:
    passpasspasspasspasspasspassself.logger.error(failed(f"Configuration validation failed: {e}"))
return False

@handle_specific_errors(
error_handlers={
ValueError: (False, "Invalid tactics parameters"),
AttributeError: (False, "Missing tactics components"),
KeyError: (False, "Missing required tactics data"),
},
default_return=False,
context="tactics execution",
)
async def execute_tactics(...) -> ...:
    """..."""
    passtry:
    passexcept Exception as e:
    passpasspasspasspasspasspassself.logger.info("🚀 Starting tactics pipeline execution...")

# Validate tactics input
if not self._validate_tactics_input(tactics_input):
    passreturn False

# Execute tactics using the orchestrator
success = await self.tactics_orchestrator.execute_tactics(tactics_input)

if success:
    passself.logger.info("✅ Tactics pipeline completed successfully")
await self._store_tactics_results(tactics_input)
else:
    passself.logger.error(failed("❌ Tactics pipeline failed"))

return success

except Exception as e:
    passpasspasspasspasspasspassself.logger.error(failed(f"❌ Tactics execution failed: {e}"))
return False

@handle_errors(
exceptions=(ValueError, AttributeError),
default_return=False,
context="tactics input validation",
)
def _validate_tactics_input(...) -> ...:
    """..."""
    passtry:
    passexcept Exception as e:
    passpasspasspasspasspasspassrequired_fields = ["symbol", "exchange", "timeframe", "current_price"]

for field in required_fields:
    passif field not in tactics_input:
    passself.logger.error(missing(f"Missing required tactics input field: {field}"))
return False

# Validate specific field values
if tactics_input.get("current_price", 0) <= 0:
    passself.logger.error(invalid("Invalid current_price value"))
return False

return True

except Exception as e:
    passpasspasspasspasspasspassself.logger.error(failed(f"Tactics input validation failed: {e}"))
return False

@handle_errors(
exceptions=(ValueError, AttributeError),
default_return=None,
context="tactics results storage",
)
async def _store_tactics_results(...) -> ...:
    """..."""
    passtry:
    passexcept Exception as e:
    passpasspasspasspasspasspass# Get results from orchestrator
self.tactics_results = self.tactics_orchestrator.get_tactics_results()

# Add to history
history_entry = {
"timestamp": datetime.now(),
"tactics_input": tactics_input, "tactics_results": self.tactics_results.copy(),
}

self.history.append(history_entry)

# Limit history size
if len(self.history) > self.max_history:
    passself.history = self.history[-self.max_history :]

self.logger.info(
f"📁 Stored tactics results (history: {len(self.history)} entries)",
)

except Exception as e:
    passpasspasspasspasspasspassself.logger.error(failed(f"❌ Failed to store tactics results: {e}"))

@handle_specific_errors(
error_handlers={
Exception: (False, "Tactician run failed"),
},
default_return=False,
context="tactician run",
)
async def run(...) -> ...:
    """..."""
    passtry:
    passexcept Exception as e:
    passpasspasspasspasspasspassself.logger.info("🚀 Starting Tactician...")
self.is_running = True

# Update status
self.status = {
"is_running": True, "start_time": datetime.now(),
"component_count": 5,  # tactics_orchestrator, position_sizer, leverage_sizer, position_division_strategy, scenario_predictor
}

self.logger.info("✅ Tactician run completed successfully")
return True

except Exception as e:
    passpasspasspasspasspasspassself.logger.error(failed(f"❌ Tactician run failed: {e}"))
return False

def get_status(...) -> ...:
    """..."""
    passreturn {
"is_running": self.is_running, "status": self.status,
"history_count": len(self.history),
"has_results": bool(self.tactics_results),
}

def get_history(...) -> ...:
    """..."""
    passhistory = self.history.copy()
if limit:
    passhistory = history[-limit:]
return history

def get_tactics_results(...) -> ...:
    """..."""
    passreturn self.tactics_results.copy()

def get_tactics_modules(...) -> ...:
    """..."""
    passreturn {
"tactics_orchestrator": self.tactics_orchestrator is not None, "position_sizer": self.position_sizer is not None,
"leverage_sizer": self.leverage_sizer is not None, "position_division_strategy": self.position_division_strategy is not None,
"scenario_predictor": self.scenario_predictor is not None,
}

@handle_specific_errors(
error_handlers={
ValueError: (None, "Invalid prediction parameters"),
AttributeError: (None, "Missing prediction components"),
KeyError: (None, "Missing required prediction data"),
},
default_return=None,
context="enhanced predictions generation",
)
async def generate_enhanced_predictions(...) -> ...:
    """..."""
    passtry:
    passexcept Exception as e:
    passpasspasspasspasspasspassif not self.is_initialized:
    passself.logger.error("Tactician not initialized")
    return self._generate_error_predictions(symbol, timeframe)

# Extract comprehensive features
features = self.scenario_predictor.extract_comprehensive_features(market_data)
features = features.reshape(1, -1)  # Reshape for single prediction

# Generate scenario predictions
scenario_predictions = await self.scenario_predictor.predict_scenarios(
features, market_data
)

# Make trading decisions
trading_decisions = self._make_trading_decisions(
scenario_predictions, analyst_confidence, market_data
)

result = {
"scenario_predictions": scenario_predictions,
"trading_decisions": trading_decisions,
"metadata": {
"symbol": symbol,
"timeframe": timeframe,
"generation_timestamp": datetime.now().isoformat(),
"model_type": "enhanced_tactician",
"analyst_confidence": analyst_confidence,
"n_scenarios": len(self.scenario_predictor.scenarios)
}
}

self.logger.info(f"Generated enhanced predictions for {symbol}")
return result

except Exception as e:
    passpasspasspasspasspasspasspassself.logger.error(f"❌ Enhanced prediction generation failed: {e}")
    return self._generate_error_predictions(symbol, timeframe)

def _make_trading_decisions(...) -> ...:
    """..."""
    passtry:
    passexcept Exception as e:
    passpasspasspasspasspasspassscenario_analysis = scenario_predictions.get("scenario_analysis", {})
confidence = scenario_predictions.get("confidence", 0.0)

# Extract key metrics
profit_zone_prob = scenario_analysis.get("profit_zone_probability", 0.0)
risk_zone_prob = scenario_analysis.get("risk_zone_probability", 0.0)
risk_reward_ratio = scenario_analysis.get("risk_reward_ratio", 0.0)
scenario_dominance = scenario_analysis.get("scenario_dominance", 0.0)
dominant_zone = scenario_analysis.get("dominant_zone", "neutral")

# Entry decision logic
entry_conditions = [
profit_zone_prob > self.decision_thresholds["entry_profit_threshold"],
risk_zone_prob < self.decision_thresholds["entry_risk_threshold"],
confidence > self.decision_thresholds["entry_confidence_threshold"],
risk_reward_ratio > self.decision_thresholds["entry_profit_risk_ratio"],
scenario_dominance > self.decision_thresholds["entry_scenario_dominance"],
dominant_zone == "profit",
analyst_confidence > 0.5  # Require some analyst confidence
]

entry_signal = all(entry_conditions)

# Exit decision logic (for existing positions)
exit_signal = False
if self.current_position:
    passpassexit_conditions = [
risk_zone_prob > self.decision_thresholds["exit_risk_threshold"],
confidence < (self.current_position.get("entry_confidence", 0.0) - self.decision_thresholds["exit_confidence_drop"]),
dominant_zone == "risk"
]
exit_signal = any(exit_conditions)

# Direction decision
direction = "LONG" if entry_signal and dominant_zone == "profit" else "NEUTRAL"
if exit_signal:
    passdirection = "EXIT"

# Confidence scoring
decision_confidence = self._calculate_decision_confidence(
scenario_analysis, confidence, analyst_confidence
)

# Reasoning
reasoning = self._generate_decision_reasoning(
entry_signal, exit_signal, scenario_analysis, confidence, analyst_confidence
)

return {
"entry_signal": entry_signal,
"exit_signal": exit_signal,
"direction": direction,
"confidence": decision_confidence,
"reasoning": reasoning,
"scenario_metrics": {
"profit_zone_probability": profit_zone_prob,
"risk_zone_probability": risk_zone_prob,
"risk_reward_ratio": risk_reward_ratio,
"scenario_dominance": scenario_dominance,
"dominant_zone": dominant_zone,
"predicted_scenario": scenario_predictions.get("predicted_scenario", 16),
"scenario_name": scenario_predictions.get("scenario_name", "Neutral")
}
}

except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"❌ Trading decision making failed: {e}")
    return {
"entry_signal": False,
"exit_signal": False,
"direction": "NEUTRAL",
"confidence": 0.0,
"reasoning": f"Error in decision making: {e}",
"scenario_metrics": {}
}



def _calculate_decision_confidence(...) -> ...:
    """..."""
    passtry:
    passexcept Exception as e:
    passpasspasspasspasspasspass# Base confidence from model
base_confidence = model_confidence

# Boost from scenario dominance
scenario_dominance = scenario_analysis.get("scenario_dominance", 0.0)
dominance_boost = scenario_dominance * 0.2

# Boost from risk-reward ratio
risk_reward_ratio = scenario_analysis.get("risk_reward_ratio", 1.0)
ratio_boost = min((risk_reward_ratio - 1.0) * 0.1, 0.2)

# Analyst confidence boost
analyst_boost = analyst_confidence * 0.1

# Final confidence
final_confidence = base_confidence + dominance_boost + ratio_boost + analyst_boost

return np.clip(final_confidence, 0.0, 1.0)

except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"❌ Decision confidence calculation failed: {e}")
    return 0.5

def _generate_decision_reasoning(...) -> ...:
    """..."""
    passtry:
    passexcept Exception as e:
    passpasspasspasspasspasspassreasoning_parts = []

if entry_signal:
    passreasoning_parts.append("ENTRY SIGNAL: Strong scenario analysis indicates favorable conditions")

profit_prob = scenario_analysis.get("profit_zone_probability", 0.0)
risk_prob = scenario_analysis.get("risk_zone_probability", 0.0)
risk_reward = scenario_analysis.get("risk_reward_ratio", 0.0)
dominance = scenario_analysis.get("scenario_dominance", 0.0)

reasoning_parts.append(f"Profit probability: {profit_prob:.1%}")
reasoning_parts.append(f"Risk probability: {risk_prob:.1%}")
reasoning_parts.append(f"Risk-reward ratio: {risk_reward:.2f}")
reasoning_parts.append(f"Scenario dominance: {dominance:.1%}")
reasoning_parts.append(f"Model confidence: {model_confidence:.1%}")
reasoning_parts.append(f"Analyst confidence: {analyst_confidence:.1%}")

elif exit_signal:
    passpassreasoning_parts.append("EXIT SIGNAL: Risk conditions detected")
risk_prob = scenario_analysis.get("risk_zone_probability", 0.0)
reasoning_parts.append(f"Risk probability: {risk_prob:.1%}")

else:
    passreasoning_parts.append("NO SIGNAL: Conditions not favorable for entry")
dominant_zone = scenario_analysis.get("dominant_zone", "neutral")
reasoning_parts.append(f"Dominant zone: {dominant_zone}")

return " | ".join(reasoning_parts)

except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"❌ Decision reasoning generation failed: {e}")
    return f"Error generating reasoning: {e}"

def _generate_error_predictions(...) -> ...:
    """..."""
    passreturn {
"scenario_predictions": {
"probabilities": {i: 1.0/17 for i in range(17)},
"predicted_scenario": 16,
"scenario_name": "Neutral",
"confidence": 0.0,
"scenario_analysis": {
"profit_zone_probability": 0.0,
"risk_zone_probability": 0.0,
"neutral_probability": 1.0,
"dominant_zone": "neutral",
"risk_reward_ratio": 0.0,
"scenario_dominance": 0.0
},
"metadata": {
"model_type": "enhanced_tactician_error",
"generation_timestamp": datetime.now().isoformat(),
"is_trained": False
}
},
"trading_decisions": {
"entry_signal": False,
"exit_signal": False,
"direction": "NEUTRAL",
"confidence": 0.0,
"reasoning": "Error in prediction generation",
"scenario_metrics": {}
},

"metadata": {
"symbol": symbol,
"timeframe": timeframe,
"generation_timestamp": datetime.now().isoformat(),
"model_type": "enhanced_tactician_error",
"analyst_confidence": 0.0,
"n_scenarios": 17
}
}

def update_position(...) -> ...:
    """..."""
    passtry:
    passexcept Exception as e:
    passpasspasspasspasspasspassself.current_position = position_data
self.position_history.append({
**position_data,
"timestamp": datetime.now().isoformat()
})

# Keep only last 100 positions
if len(self.position_history) > 100:
    passself.position_history = self.position_history[-100:]

except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"❌ Position update failed: {e}")

def update_performance_metrics(...) -> ...:
    """..."""
    passtry:
    passexcept Exception as e:
    passpasspasspasspasspasspassself.performance_metrics["total_trades"] += 1

if trade_result.get("profit", 0) > 0:
    passself.performance_metrics["winning_trades"] += 1
self.performance_metrics["total_profit"] += trade_result["profit"]
else:
    passself.performance_metrics["losing_trades"] += 1
self.performance_metrics["total_loss"] += abs(trade_result.get("profit", 0))

# Calculate derived metrics
win_rate = self.performance_metrics["winning_trades"] / max(self.performance_metrics["total_trades"], 1)
profit_factor = self.performance_metrics["total_profit"] / max(self.performance_metrics["total_loss"], 0.001)

self.performance_metrics["win_rate"] = win_rate
self.performance_metrics["profit_factor"] = profit_factor

except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"❌ Performance metrics update failed: {e}")

def get_performance_summary(...) -> ...:
    """..."""
    passreturn {
"performance_metrics": self.performance_metrics,
"current_position": self.current_position,
"position_history_count": len(self.position_history),
"is_initialized": self.is_initialized,
"scenario_predictor_status": {
"is_trained": self.scenario_predictor.is_trained if self.scenario_predictor else False,
"n_scenarios": len(self.scenario_predictor.scenarios) if self.scenario_predictor else 0,
"last_training_time": self.scenario_predictor.last_training_time.isoformat() if self.scenario_predictor and self.scenario_predictor.last_training_time else None
}
}

def get_configuration_summary(...) -> ...:
    pass"""..."""
    passreturn {
"decision_thresholds": self.decision_thresholds,
"scenario_predictor_config": self.scenario_predictor.get_enhanced_configuration_summary() if self.scenario_predictor else {},
"is_initialized": self.is_initialized
}

@handle_errors(
exceptions=(Exception,),
default_return=None,
context="tactician stop",
)
async def stop(...) -> ...:
    """..."""
    passtry:
    passexcept Exception as e:
    passpasspasspasspasspasspassself.logger.info("🛑 Stopping Tactician...")

# Stop component managers
if self.tactics_orchestrator:
    passawait self.tactics_orchestrator.stop()
if self.position_sizer:
    passawait self.position_sizer.stop()
if self.leverage_sizer:
    passawait self.leverage_sizer.stop()
if self.position_division_strategy:
    passawait self.position_division_strategy.stop()
if self.scenario_predictor:
    passawait self.scenario_predictor.stop()

self.is_running = False
self.logger.info("✅ Tactician stopped successfully")

except Exception as e:
    passpasspasspasspasspasspassself.logger.error(failed(f"❌ Failed to stop Tactician: {e}"))

@handle_errors(
exceptions=(Exception,),
default_return=None,
context="tactician cleanup",
)
async def cleanup(...) -> ...:
    """..."""
    passtry:
    passexcept Exception as e:
    passpasspasspasspasspasspassself.logger.info("Cleaning up Tactician...")
await self.stop()

# Cleanup component managers
if self.tactics_orchestrator:
    passawait self.tactics_orchestrator.cleanup()
if self.position_sizer:
    passawait self.position_sizer.cleanup()
if self.leverage_sizer:
    passawait self.leverage_sizer.cleanup()
if self.position_division_strategy:
    passawait self.position_division_strategy.cleanup()
if self.scenario_predictor:
    passawait self.scenario_predictor.cleanup()

# Clear history and results
self.history.clear()
self.tactics_results.clear()
self.status.clear()

self.logger.info("✅ Tactician cleanup completed")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(failed(f"❌ Failed to cleanup Tactician: {e}"))

@handle_errors(
exceptions=(Exception,),
default_return=None,
context="tactician setup",
)
async def setup_tactician(...) -> ...:
    """..."""
    passtry:
    passexcept Exception as e:
    passpasspasspasspasspasspasstactician = Tactician(config or {})
if await tactician.initialize():
    passreturn tactician
return None
except Exception as e:
    passpasspasspasspasspasspasssystem_logger.exception(f"Failed to setup tactician: {e}")
return None
