"""
Component Monitor Module.

This module monitors individual trading system components (Analyst, Strategist, 
Tactician, etc.) for health, performance, and feature extraction.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from src.core.decorators import handles_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import error
from src.core.decorators.errors import handles_errors


class ComponentMonitor:
    """Monitors individual system components for health and performance."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize component monitor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("ComponentMonitor")
        self.component_status: Dict[str, Dict[str, Any]] = {}
        self.feature_history: Dict[str, List[Dict[str, Any]]] = {}
        self.max_history: int = config.get("max_feature_history", 1000)

    @handles_errors(
        exceptions=(Exception,),
        default_return=None,
    )
    def monitor_analyst_features(self, analyst: Any) -> Dict[str, Any]:
        """
        Monitor Analyst component features.
        
        Args:
            analyst: Analyst component instance
            
        Returns:
            Dictionary of extracted features
        """
        try:
            features = {
                "timestamp": datetime.now().isoformat(),
                "component": "analyst",
                "is_analyzing": getattr(analyst, "is_analyzing", False),
                "analysis_count": len(getattr(analyst, "analysis_history", [])),
                "last_analysis": None,
                "model_confidence": None,
                "regime": None,
            }

            # Extract recent analysis results
            if hasattr(analyst, "analysis_results"):
                results = analyst.analysis_results
                features["model_confidence"] = results.get("ml_confidence")
                features["regime"] = results.get("regime")
                features["last_analysis"] = results.get("timestamp")

            self._store_features("analyst", features)
            return features

        except Exception as e:
            self.logger.error(error(f"Error monitoring Analyst features: {e}"))
            return {}

    @handles_errors(
        exceptions=(Exception,),
        default_return=None,
    )
    def monitor_strategist_features(self, strategist: Any) -> Dict[str, Any]:
        """
        Monitor Strategist component features.
        
        Args:
            strategist: Strategist component instance
            
        Returns:
            Dictionary of extracted features
        """
        try:
            features = {
                "timestamp": datetime.now().isoformat(),
                "component": "strategist",
                "is_generating": getattr(strategist, "is_generating", False),
                "signal_count": len(getattr(strategist, "signal_history", [])),
                "current_strategy": getattr(strategist, "current_strategy", None),
                "active_signals": None,
            }

            # Extract strategy details
            if hasattr(strategist, "strategy_results"):
                results = strategist.strategy_results
                features["active_signals"] = results.get("active_signals", 0)
                features["strategy_confidence"] = results.get("confidence")
                features["strategy_type"] = results.get("strategy_type")

            self._store_features("strategist", features)
            return features

        except Exception as e:
            self.logger.error(error(f"Error monitoring Strategist features: {e}"))
            return {}

    @handles_errors(
        exceptions=(Exception,),
        default_return=None,
    )
    def monitor_tactician_features(self, tactician: Any) -> Dict[str, Any]:
        """
        Monitor Tactician component features.
        
        Args:
            tactician: Tactician component instance
            
        Returns:
            Dictionary of extracted features
        """
        try:
            features = {
                "timestamp": datetime.now().isoformat(),
                "component": "tactician",
                "is_executing": getattr(tactician, "is_executing", False),
                "execution_count": len(getattr(tactician, "execution_history", [])),
                "position_size": None,
                "risk_level": None,
            }

            # Extract tactical details
            if hasattr(tactician, "tactics_results"):
                results = tactician.tactics_results
                features["position_size"] = results.get("position_size")
                features["risk_level"] = results.get("risk_level")
                features["entry_timing"] = results.get("entry_timing")
                features["leverage"] = results.get("leverage", 1.0)

            self._store_features("tactician", features)
            return features

        except Exception as e:
            self.logger.error(error(f"Error monitoring Tactician features: {e}"))
            return {}

    @handles_errors(
        exceptions=(Exception,),
        default_return=None,
    )
    def monitor_training_manager_features(self, training_manager: Any) -> Dict[str, Any]:
        """
        Monitor Enhanced Training Manager features.
        
        Args:
            training_manager: Training manager instance
            
        Returns:
            Dictionary of extracted features
        """
        try:
            features = {
                "timestamp": datetime.now().isoformat(),
                "component": "training_manager",
                "is_training": getattr(training_manager, "is_training", False),
                "training_cycles": getattr(training_manager, "training_cycles_completed", 0),
                "model_version": None,
                "training_progress": None,
            }

            # Extract training details
            if hasattr(training_manager, "training_results"):
                results = training_manager.training_results
                features["model_version"] = results.get("model_version")
                features["training_progress"] = results.get("progress", 0.0)
                features["best_score"] = results.get("best_score")
                features["validation_score"] = results.get("validation_score")

            self._store_features("training_manager", features)
            return features

        except Exception as e:
            self.logger.error(error(f"Error monitoring Training Manager features: {e}"))
            return {}

    def _store_features(self, component_name: str, features: Dict[str, Any]) -> None:
        """Store features in history with size limit."""
        if component_name not in self.feature_history:
            self.feature_history[component_name] = []
        
        self.feature_history[component_name].append(features)
        
        # Maintain history size limit
        if len(self.feature_history[component_name]) > self.max_history:
            self.feature_history[component_name].pop(0)

    def get_component_status(self, component_name: str) -> Dict[str, Any]:
        """
        Get current status of a component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            Component status dictionary
        """
        if component_name not in self.feature_history:
            return {"status": "unknown", "message": "No monitoring data available"}
        
        history = self.feature_history[component_name]
        if not history:
            return {"status": "unknown", "message": "No monitoring data available"}
        
        latest = history[-1]
        return {
            "status": "active" if latest.get(f"is_{component_name.split('_')[0]}ing", False) else "idle",
            "last_update": latest.get("timestamp"),
            "latest_features": latest,
            "history_size": len(history)
        }

    def get_all_component_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all monitored components."""
        return {
            component: self.get_component_status(component)
            for component in self.feature_history.keys()
        }