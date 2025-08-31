# src/tactician/exit_strategy_manager.py

"""
Exit Strategy Manager for Tactician.
Comprehensive exit strategy system that integrates multi-timeframe entry models,
trend reversal detection, and exit timing optimization for high-leverage trading.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import joblib
from pathlib import Path

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors
from src.utils.decorators import guard_dataframe_nulls, with_tracing_span
from src.tactician.exit_strategy_feature_engineering import ExitStrategyFeatureEngineering
from src.tactician.multi_timeframe_entry_models import MultiTimeframeEntryModels


class ExitStrategyManager:
    """
    Comprehensive exit strategy manager for Tactician.
    Integrates all exit strategy components for optimal position closure decisions.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize exit strategy manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("ExitStrategyManager")
        
        # Load configuration
        self.exit_config = config.get("exit_strategy", {})
        self.models_dir = self.exit_config.get("models_dir", "models/exit_strategy")
        
        # Initialize components
        self.feature_engineering = ExitStrategyFeatureEngineering(config)
        self.entry_models = MultiTimeframeEntryModels(config)
        
        # Load optimized parameters
        self.optimized_params = self.exit_config.get("optimized_parameters", {})
        
        # Model storage
        self.reversal_model = None
        self.exit_timing_model = None
        self.ensemble_exit_model = None
        
        # State tracking
        self.position_context = {}
        self.exit_decisions = []
        self.performance_metrics = {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="exit strategy manager initialization"
    )
    async def initialize(self) -> bool:
        """
        Initialize the exit strategy manager.

        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("🔧 Initializing Exit Strategy Manager...")
            
            # Initialize feature engineering
            if not await self.feature_engineering.initialize():
                self.logger.error("❌ Feature engineering initialization failed")
                return False
            
            # Initialize entry models
            if not await self.entry_models.initialize():
                self.logger.error("❌ Entry models initialization failed")
                return False
            
            # Load trained models
            if not await self._load_trained_models():
                self.logger.warning("⚠️ Failed to load some trained models")
            
            # Load optimized parameters
            await self._load_optimized_parameters()
            
            self.logger.info("✅ Exit Strategy Manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Exit Strategy Manager initialization failed: {e}")
            return False

    async def _load_trained_models(self) -> bool:
        """
        Load trained exit strategy models.

        Returns:
            bool: True if models loaded successfully
        """
        try:
            models_path = Path(self.models_dir)
            if not models_path.exists():
                self.logger.warning(f"Models directory not found: {self.models_dir}")
                return False
            
            # Load reversal detection model
            reversal_path = models_path / "reversal_detection_model.joblib"
            if reversal_path.exists():
                self.reversal_model = joblib.load(reversal_path)
                self.logger.info("✅ Loaded reversal detection model")
            
            # Load exit timing model
            timing_path = models_path / "exit_timing_model.joblib"
            if timing_path.exists():
                self.exit_timing_model = joblib.load(timing_path)
                self.logger.info("✅ Loaded exit timing model")
            
            # Load ensemble exit model
            ensemble_path = models_path / "ensemble_exit_model.joblib"
            if ensemble_path.exists():
                self.ensemble_exit_model = joblib.load(ensemble_path)
                self.logger.info("✅ Loaded ensemble exit model")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            return False

    async def _load_optimized_parameters(self) -> None:
        """Load optimized parameters from step17."""
        try:
            # Load from step17 optimization results
            step17_path = Path("models") / "step17_optimization_results.json"
            if step17_path.exists():
                with open(step17_path, 'r') as f:
                    import json
                    step17_results = json.load(f)
                
                exit_strategy_params = step17_results.get("exit_strategy_parameters", {})
                if exit_strategy_params:
                    self.optimized_params = exit_strategy_params.get("optimized_parameters", {})
                    self.logger.info("✅ Loaded optimized exit strategy parameters")
            
        except Exception as e:
            self.logger.warning(f"Failed to load optimized parameters: {e}")

    @guard_dataframe_nulls
    @with_tracing_span("exit_strategy_evaluate_position")
    async def evaluate_position_exit(
        self, 
        current_data: pd.DataFrame,
        position_context: Dict[str, Any],
        timeframe_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate whether to exit a position based on comprehensive analysis.

        Args:
            current_data: Current market data
            position_context: Position context information
            timeframe_data: Multi-timeframe data (optional)

        Returns:
            Dict containing exit decision and reasoning
        """
        try:
            self.logger.info("🔍 Evaluating position exit...")
            
            # Update position context
            self.position_context = position_context
            
            # Apply feature engineering
            features_df = await self.feature_engineering.apply_all(current_data, position_context)
            
            if features_df.empty:
                return self._create_exit_decision(False, 0.0, "Feature engineering failed")
            
            # Get latest features
            latest_features = features_df.iloc[-1:].fillna(0)
            
            # 1. Evaluate trend reversal probability
            reversal_prob = await self._evaluate_reversal_probability(latest_features)
            
            # 2. Evaluate exit timing
            exit_timing_prob = await self._evaluate_exit_timing(latest_features)
            
            # 3. Evaluate multi-timeframe entry signals (if data available)
            entry_signals = {}
            if timeframe_data:
                entry_signals = await self._evaluate_multi_timeframe_entries(timeframe_data)
            
            # 4. Evaluate profit preservation
            profit_preservation = await self._evaluate_profit_preservation(latest_features, position_context)
            
            # 5. Evaluate time decay
            time_decay = await self._evaluate_time_decay(position_context)
            
            # 6. Get ensemble exit decision
            ensemble_decision = await self._get_ensemble_exit_decision(latest_features)
            
            # 7. Combine all signals for final decision
            final_decision = await self._combine_exit_signals(
                reversal_prob, exit_timing_prob, entry_signals, 
                profit_preservation, time_decay, ensemble_decision
            )
            
            # Log decision
            self._log_exit_decision(final_decision)
            
            return final_decision
            
        except Exception as e:
            self.logger.error(f"Position exit evaluation failed: {e}")
            return self._create_exit_decision(False, 0.0, f"Evaluation failed: {e}")

    async def _evaluate_reversal_probability(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate trend reversal probability."""
        try:
            if self.reversal_model is None:
                return {"probability": 0.0, "confidence": 0.0, "available": False}
            
            # Make prediction
            prediction = self.reversal_model.predict(features_df)
            probability = self.reversal_model.predict_proba(features_df)[0, 1]
            
            # Get confidence from feature importance
            confidence = self._calculate_model_confidence(features_df, "reversal")
            
            return {
                "probability": probability,
                "confidence": confidence,
                "prediction": prediction[0],
                "available": True
            }
            
        except Exception as e:
            self.logger.error(f"Reversal evaluation failed: {e}")
            return {"probability": 0.0, "confidence": 0.0, "available": False}

    async def _evaluate_exit_timing(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate exit timing."""
        try:
            if self.exit_timing_model is None:
                return {"probability": 0.0, "confidence": 0.0, "available": False}
            
            # Make prediction
            prediction = self.exit_timing_model.predict(features_df)
            probability = self.exit_timing_model.predict_proba(features_df)[0, 1]
            
            # Get confidence
            confidence = self._calculate_model_confidence(features_df, "timing")
            
            return {
                "probability": probability,
                "confidence": confidence,
                "prediction": prediction[0],
                "available": True
            }
            
        except Exception as e:
            self.logger.error(f"Exit timing evaluation failed: {e}")
            return {"probability": 0.0, "confidence": 0.0, "available": False}

    async def _evaluate_multi_timeframe_entries(
        self, timeframe_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Evaluate multi-timeframe entry signals."""
        try:
            # Get multi-timeframe entry signals
            entry_signals = await self.entry_models.get_multi_timeframe_entry_signal(
                timeframe_data, self.position_context
            )
            
            return {
                "combined_signal": entry_signals.get("combined_signal", 0),
                "combined_confidence": entry_signals.get("combined_confidence", 0.0),
                "combined_probability": entry_signals.get("combined_probability", 0.0),
                "timeframe_signals": entry_signals.get("timeframe_signals", {}),
                "available": True
            }
            
        except Exception as e:
            self.logger.error(f"Multi-timeframe evaluation failed: {e}")
            return {"available": False}

    async def _evaluate_profit_preservation(
        self, features_df: pd.DataFrame, position_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate profit preservation."""
        try:
            # Extract profit-related features
            profit_decay_rate = features_df.get('profit_decay_rate', pd.Series([0.0])).iloc[-1]
            profit_preservation_score = features_df.get('profit_preservation_score', pd.Series([0.0])).iloc[-1]
            
            # Get current PnL
            current_pnl = position_context.get('current_pnl', 0.0)
            entry_price = position_context.get('entry_price', 0.0)
            current_price = position_context.get('current_price', 0.0)
            
            if entry_price > 0 and current_price > 0:
                unrealized_pnl_pct = (current_price - entry_price) / entry_price
            else:
                unrealized_pnl_pct = 0.0
            
            # Calculate profit preservation score
            profit_score = max(0.0, min(1.0, profit_preservation_score))
            
            # Apply profit decay penalty
            decay_penalty = max(0.0, profit_decay_rate)
            adjusted_score = profit_score * (1 - decay_penalty)
            
            return {
                "profit_score": adjusted_score,
                "unrealized_pnl_pct": unrealized_pnl_pct,
                "profit_decay_rate": profit_decay_rate,
                "profit_preservation_score": profit_preservation_score,
                "available": True
            }
            
        except Exception as e:
            self.logger.error(f"Profit preservation evaluation failed: {e}")
            return {"available": False}

    async def _evaluate_time_decay(self, position_context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate time decay factors."""
        try:
            entry_time = position_context.get('entry_time')
            if not entry_time:
                return {"available": False}
            
            # Calculate time since entry
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
            
            time_since_entry = datetime.now(entry_time.tzinfo) - entry_time
            minutes_since_entry = time_since_entry.total_seconds() / 60
            
            # Get optimized parameters
            max_hold_time = self.optimized_params.get("max_hold_time_minutes", 120)
            time_decay_factor = self.optimized_params.get("time_decay_factor", 0.5)
            
            # Calculate time decay score
            if minutes_since_entry <= max_hold_time:
                time_decay_score = 1.0 - (minutes_since_entry / max_hold_time) * time_decay_factor
            else:
                time_decay_score = 0.0
            
            return {
                "time_decay_score": time_decay_score,
                "minutes_since_entry": minutes_since_entry,
                "max_hold_time": max_hold_time,
                "time_decay_factor": time_decay_factor,
                "available": True
            }
            
        except Exception as e:
            self.logger.error(f"Time decay evaluation failed: {e}")
            return {"available": False}

    async def _get_ensemble_exit_decision(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Get ensemble exit decision."""
        try:
            if self.ensemble_exit_model is None:
                return {"probability": 0.0, "confidence": 0.0, "available": False}
            
            # Make prediction
            prediction = self.ensemble_exit_model.predict(features_df)
            probability = self.ensemble_exit_model.predict_proba(features_df)[0, 1]
            
            # Get confidence
            confidence = self._calculate_model_confidence(features_df, "ensemble")
            
            return {
                "probability": probability,
                "confidence": confidence,
                "prediction": prediction[0],
                "available": True
            }
            
        except Exception as e:
            self.logger.error(f"Ensemble decision failed: {e}")
            return {"probability": 0.0, "confidence": 0.0, "available": False}

    async def _combine_exit_signals(
        self, reversal_prob: Dict[str, Any], exit_timing_prob: Dict[str, Any],
        entry_signals: Dict[str, Any], profit_preservation: Dict[str, Any],
        time_decay: Dict[str, Any], ensemble_decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine all exit signals for final decision."""
        try:
            # Get optimized thresholds
            reversal_threshold = self.optimized_params.get("reversal_threshold", 0.01)
            reversal_confidence_threshold = self.optimized_params.get("reversal_confidence_threshold", 0.7)
            exit_urgency_threshold = self.optimized_params.get("exit_urgency_threshold", 0.5)
            exit_timing_confidence_threshold = self.optimized_params.get("exit_timing_confidence_threshold", 0.7)
            entry_confidence_threshold = self.optimized_params.get("entry_confidence_threshold", 0.8)
            multi_timeframe_agreement_threshold = self.optimized_params.get("multi_timeframe_agreement_threshold", 0.6)
            profit_decay_threshold = self.optimized_params.get("profit_decay_threshold", 0.3)
            profit_preservation_threshold = self.optimized_params.get("profit_preservation_threshold", 0.5)
            risk_adjustment_factor = self.optimized_params.get("risk_adjustment_factor", 1.0)
            
            # Calculate weighted exit probability
            exit_probabilities = []
            weights = []
            
            # 1. Reversal probability
            if reversal_prob.get("available", False):
                reversal_prob_val = reversal_prob["probability"]
                reversal_conf = reversal_prob["confidence"]
                
                if reversal_conf >= reversal_confidence_threshold:
                    exit_probabilities.append(reversal_prob_val)
                    weights.append(0.25)
            
            # 2. Exit timing probability
            if exit_timing_prob.get("available", False):
                timing_prob_val = exit_timing_prob["probability"]
                timing_conf = exit_timing_prob["confidence"]
                
                if timing_conf >= exit_timing_confidence_threshold:
                    exit_probabilities.append(timing_prob_val)
                    weights.append(0.25)
            
            # 3. Multi-timeframe entry signals (inverse - if entry signals are weak, exit)
            if entry_signals.get("available", False):
                entry_conf = entry_signals.get("combined_confidence", 0.0)
                if entry_conf < entry_confidence_threshold:
                    # Weak entry signals suggest exit
                    exit_probabilities.append(1.0 - entry_conf)
                    weights.append(0.2)
            
            # 4. Profit preservation
            if profit_preservation.get("available", False):
                profit_score = profit_preservation["profit_score"]
                if profit_score < profit_preservation_threshold:
                    exit_probabilities.append(1.0 - profit_score)
                    weights.append(0.15)
            
            # 5. Time decay
            if time_decay.get("available", False):
                time_decay_score = time_decay["time_decay_score"]
                if time_decay_score < 0.3:  # Strong time decay
                    exit_probabilities.append(1.0 - time_decay_score)
                    weights.append(0.1)
            
            # 6. Ensemble decision
            if ensemble_decision.get("available", False):
                ensemble_prob = ensemble_decision["probability"]
                ensemble_conf = ensemble_decision["confidence"]
                
                if ensemble_conf >= 0.7:  # High confidence ensemble
                    exit_probabilities.append(ensemble_prob)
                    weights.append(0.05)
            
            # Calculate weighted average
            if exit_probabilities and weights:
                # Normalize weights
                total_weight = sum(weights)
                normalized_weights = [w / total_weight for w in weights]
                
                weighted_exit_prob = sum(p * w for p, w in zip(exit_probabilities, normalized_weights))
            else:
                weighted_exit_prob = 0.0
            
            # Apply risk adjustment
            adjusted_exit_prob = weighted_exit_prob * risk_adjustment_factor
            
            # Make final decision
            should_exit = adjusted_exit_prob >= exit_urgency_threshold
            
            # Create detailed reasoning
            reasoning = self._create_exit_reasoning(
                reversal_prob, exit_timing_prob, entry_signals,
                profit_preservation, time_decay, ensemble_decision,
                weighted_exit_prob, adjusted_exit_prob, should_exit
            )
            
            return self._create_exit_decision(should_exit, adjusted_exit_prob, reasoning)
            
        except Exception as e:
            self.logger.error(f"Signal combination failed: {e}")
            return self._create_exit_decision(False, 0.0, f"Signal combination failed: {e}")

    def _calculate_model_confidence(self, features_df: pd.DataFrame, model_type: str) -> float:
        """Calculate model confidence based on feature values."""
        try:
            # Simple confidence calculation based on feature variance
            # In practice, this could be more sophisticated
            feature_variance = features_df.var().mean()
            confidence = max(0.0, min(1.0, 1.0 - feature_variance))
            return confidence
            
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return 0.5

    def _create_exit_reasoning(
        self, reversal_prob: Dict[str, Any], exit_timing_prob: Dict[str, Any],
        entry_signals: Dict[str, Any], profit_preservation: Dict[str, Any],
        time_decay: Dict[str, Any], ensemble_decision: Dict[str, Any],
        weighted_exit_prob: float, adjusted_exit_prob: float, should_exit: bool
    ) -> str:
        """Create detailed reasoning for exit decision."""
        try:
            reasons = []
            
            if should_exit:
                reasons.append("EXIT RECOMMENDED")
            else:
                reasons.append("HOLD POSITION")
            
            reasons.append(f"Overall exit probability: {adjusted_exit_prob:.3f}")
            
            # Add specific reasons
            if reversal_prob.get("available", False) and reversal_prob["probability"] > 0.7:
                reasons.append(f"High reversal probability: {reversal_prob['probability']:.3f}")
            
            if exit_timing_prob.get("available", False) and exit_timing_prob["probability"] > 0.7:
                reasons.append(f"High exit timing probability: {exit_timing_prob['probability']:.3f}")
            
            if entry_signals.get("available", False):
                entry_conf = entry_signals.get("combined_confidence", 0.0)
                if entry_conf < 0.5:
                    reasons.append(f"Weak entry signals: {entry_conf:.3f}")
            
            if profit_preservation.get("available", False):
                profit_score = profit_preservation["profit_score"]
                if profit_score < 0.5:
                    reasons.append(f"Low profit preservation: {profit_score:.3f}")
            
            if time_decay.get("available", False):
                time_decay_score = time_decay["time_decay_score"]
                if time_decay_score < 0.3:
                    reasons.append(f"Strong time decay: {time_decay_score:.3f}")
            
            return " | ".join(reasons)
            
        except Exception as e:
            return f"Reasoning generation failed: {e}"

    def _create_exit_decision(self, should_exit: bool, probability: float, reasoning: str) -> Dict[str, Any]:
        """Create standardized exit decision."""
        return {
            "should_exit": should_exit,
            "exit_probability": probability,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat(),
            "confidence": probability,  # Use probability as confidence
            "urgency": "high" if probability > 0.8 else "medium" if probability > 0.6 else "low"
        }

    def _log_exit_decision(self, decision: Dict[str, Any]) -> None:
        """Log exit decision for monitoring."""
        try:
            self.exit_decisions.append(decision)
            
            # Keep only last 100 decisions
            if len(self.exit_decisions) > 100:
                self.exit_decisions = self.exit_decisions[-100:]
            
            # Log decision
            if decision["should_exit"]:
                self.logger.info(f"🚨 EXIT SIGNAL: {decision['reasoning']}")
            else:
                self.logger.debug(f"📊 Hold signal: {decision['reasoning']}")
                
        except Exception as e:
            self.logger.error(f"Decision logging failed: {e}")

    async def get_exit_strategy_performance(self) -> Dict[str, Any]:
        """Get exit strategy performance metrics."""
        try:
            if not self.exit_decisions:
                return {"error": "No exit decisions available"}
            
            # Calculate performance metrics
            total_decisions = len(self.exit_decisions)
            exit_signals = sum(1 for d in self.exit_decisions if d["should_exit"])
            avg_probability = np.mean([d["exit_probability"] for d in self.exit_decisions])
            
            # Calculate urgency distribution
            urgency_counts = {}
            for decision in self.exit_decisions:
                urgency = decision.get("urgency", "low")
                urgency_counts[urgency] = urgency_counts.get(urgency, 0) + 1
            
            return {
                "total_decisions": total_decisions,
                "exit_signals": exit_signals,
                "hold_signals": total_decisions - exit_signals,
                "exit_rate": exit_signals / total_decisions if total_decisions > 0 else 0.0,
                "average_probability": avg_probability,
                "urgency_distribution": urgency_counts,
                "last_decision": self.exit_decisions[-1] if self.exit_decisions else None
            }
            
        except Exception as e:
            self.logger.error(f"Performance calculation failed: {e}")
            return {"error": str(e)}

    async def update_position_context(self, position_context: Dict[str, Any]) -> None:
        """Update position context for ongoing evaluation."""
        try:
            self.position_context.update(position_context)
            self.logger.debug(f"Position context updated: {list(position_context.keys())}")
            
        except Exception as e:
            self.logger.error(f"Position context update failed: {e}")

    async def reset_state(self) -> None:
        """Reset exit strategy manager state."""
        try:
            self.position_context = {}
            self.exit_decisions = []
            self.performance_metrics = {}
            self.logger.info("✅ Exit strategy manager state reset")
            
        except Exception as e:
            self.logger.error(f"State reset failed: {e}")