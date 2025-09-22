"""
Regime Monitor

Monitors market regime changes and provides alerts for regime transitions.
Tracks regime stability and provides insights into market condition changes.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.tprint import (
    tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_structured, LogLevel
)
from ..config.regime_config import RegimeType, RegimeConfig
from ..utils.error_handling import (
    TradingError, TradingErrorSeverity, trading_error_handler,
    critical_operation, require_no_fallback
)
from ..utils.validation import validate_trading_config

logger = system_logger.getChild('RegimeMonitor')

class RegimeStability(Enum):
    """Regime stability levels."""
    STABLE = "stable"
    TRANSITIONING = "transitioning"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"

class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RegimeState:
    """Current regime state."""
    timestamp: datetime
    primary_regime: RegimeType
    regime_probabilities: Dict[RegimeType, float]
    confidence: float
    stability: RegimeStability
    duration: timedelta
    transitions: List[Tuple[RegimeType, RegimeType, datetime]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RegimeTransition:
    """Regime transition event."""
    transition_id: str
    from_regime: RegimeType
    to_regime: RegimeType
    timestamp: datetime
    confidence: float
    transition_probability: float
    market_conditions: Dict[str, Any]
    significance_score: float
    alert_sent: bool = False

@dataclass
class RegimeAlert:
    """Regime change alert."""
    alert_id: str
    regime_state: RegimeState
    transition: Optional[RegimeTransition]
    severity: AlertSeverity
    message: str
    timestamp: datetime
    actions: List[str] = field(default_factory=list)

class RegimeMonitor:
    """
    Regime Monitor

    Monitors market regime changes, tracks regime stability,
    and generates alerts for significant regime transitions.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize regime monitor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logger.getChild('RegimeMonitor')

        # Configuration
        self.regime_config = RegimeConfig(**config.get('regime_config', {}))
        self.stability_threshold = config.get('stability_threshold', 0.7)
        self.transition_threshold = config.get('transition_threshold', 0.3)
        self.min_regime_duration = config.get('min_regime_duration_minutes', 30)

        # Current state
        self.current_regime_state: Optional[RegimeState] = None
        self.regime_history: List[RegimeState] = []

        # Transition tracking
        self.transitions: List[RegimeTransition] = []
        self.pending_transitions: Dict[str, RegimeTransition] = {}

        # Alerts
        self.alerts: List[RegimeAlert] = []
        self.alert_history: List[RegimeAlert] = []

        # Performance metrics
        self.false_positive_rate = 0.0
        self.detection_accuracy = 0.0

        tprint_info("🔄 Initializing Regime Monitor...")

    async def initialize(self) -> None:
        """Initialize regime monitor."""
        tprint_success("✅ Regime Monitor initialized successfully")

    @handles_errors
    async def update_regime_state(
        self,
        regime_probabilities: Dict[RegimeType, float],
        confidence: float,
        market_conditions: Dict[str, Any]
    ) -> RegimeState:
        """
        Update current regime state.

        Args:
            regime_probabilities: Probability distribution across regimes
            confidence: Confidence score of regime detection
            market_conditions: Current market conditions

        Returns:
            Updated regime state
        """
        # Determine primary regime
        primary_regime = max(regime_probabilities, key=regime_probabilities.get)
        primary_probability = regime_probabilities[primary_regime]

        # Calculate stability
        stability = await self._calculate_regime_stability(regime_probabilities)

        # Create regime state
        regime_state = RegimeState(
            timestamp=datetime.now(),
            primary_regime=primary_regime,
            regime_probabilities=regime_probabilities,
            confidence=confidence,
            stability=stability,
            duration=timedelta(0),
            metadata={'market_conditions': market_conditions}
        )

        # Update duration if same regime
        if (self.current_regime_state and
            self.current_regime_state.primary_regime == primary_regime):
            regime_state.duration = (regime_state.timestamp -
                                   self.current_regime_state.timestamp)

        # Check for regime transition
        if (self.current_regime_state and
            self.current_regime_state.primary_regime != primary_regime):
            await self._handle_regime_transition(
                self.current_regime_state.primary_regime,
                primary_regime,
                confidence,
                market_conditions
            )

        # Store regime history
        self.regime_history.append(regime_state)

        # Keep only recent history (last 1000 states)
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-1000:]

        self.current_regime_state = regime_state

        return regime_state

    async def _calculate_regime_stability(self, probabilities: Dict[RegimeType, float]) -> RegimeStability:
        """Calculate regime stability based on probability distribution."""
        # Higher entropy = less stable
        total_prob = sum(probabilities.values())
        if total_prob == 0:
            return RegimeStability.UNKNOWN

        probabilities = {k: v / total_prob for k, v in probabilities.items()}

        # Calculate Shannon entropy
        entropy = -sum(p * np.log2(p) for p in probabilities.values() if p > 0)

        # Normalize entropy (max entropy for N regimes is log2(N))
        max_entropy = np.log2(len(probabilities))
        normalized_entropy = entropy / max_entropy

        # Determine stability
        if normalized_entropy < 0.2:
            return RegimeStability.STABLE
        elif normalized_entropy < 0.5:
            return RegimeStability.TRANSITIONING
        else:
            return RegimeStability.VOLATILE

    async def _handle_regime_transition(
        self,
        from_regime: RegimeType,
        to_regime: RegimeType,
        confidence: float,
        market_conditions: Dict[str, Any]
    ) -> None:
        """Handle regime transition event."""
        transition_id = f"transition_{datetime.now().timestamp()}"

        # Calculate transition significance
        significance_score = await self._calculate_transition_significance(
            from_regime, to_regime, confidence, market_conditions
        )

        transition = RegimeTransition(
            transition_id=transition_id,
            from_regime=from_regime,
            to_regime=to_regime,
            timestamp=datetime.now(),
            confidence=confidence,
            transition_probability=significance_score,
            market_conditions=market_conditions,
            significance_score=significance_score
        )

        self.transitions.append(transition)

        # Update current regime state's transitions
        if self.current_regime_state:
            self.current_regime_state.transitions.append((from_regime, to_regime, datetime.now()))

        # Generate alert if significant
        if significance_score > self.transition_threshold:
            await self._generate_regime_alert(transition)

        tprint_info(f"🔄 Regime transition: {from_regime.value} → {to_regime.value}")

    async def _calculate_transition_significance(
        self,
        from_regime: RegimeType,
        to_regime: RegimeType,
        confidence: float,
        market_conditions: Dict[str, Any]
    ) -> float:
        """Calculate significance score for regime transition."""
        significance = 0.0

        # Base significance from confidence
        significance += confidence * 0.4

        # Regime transition frequency (less frequent = more significant)
        transition_count = sum(1 for t in self.transitions[-100:]
                             if t.from_regime == from_regime and t.to_regime == to_regime)
        transition_frequency = transition_count / max(len(self.transitions), 1)
        significance += (1 - transition_frequency) * 0.3

        # Market volatility impact
        volatility = market_conditions.get('volatility', 0.5)
        significance += volatility * 0.3

        return min(significance, 1.0)

    async def _generate_regime_alert(self, transition: RegimeTransition) -> None:
        """Generate alert for significant regime transition."""
        alert_id = f"alert_{transition.transition_id}"

        # Determine severity
        if transition.significance_score > 0.8:
            severity = AlertSeverity.CRITICAL
            message = f"🚨 CRITICAL: Major regime change from {transition.from_regime.value} to {transition.to_regime.value}"
            actions = ["Review all open positions", "Adjust position sizing", "Prepare for increased volatility"]
        elif transition.significance_score > 0.6:
            severity = AlertSeverity.HIGH
            message = f"⚠️ HIGH: Significant regime transition from {transition.from_regime.value} to {transition.to_regime.value}"
            actions = ["Monitor position performance", "Consider reducing exposure"]
        elif transition.significance_score > 0.4:
            severity = AlertSeverity.MEDIUM
            message = f"📊 MEDIUM: Regime change detected from {transition.from_regime.value} to {transition.to_regime.value}"
            actions = ["Monitor market conditions", "Review trading strategy"]
        else:
            severity = AlertSeverity.LOW
            message = f"ℹ️ LOW: Minor regime change from {transition.from_regime.value} to {transition.to_regime.value}"
            actions = ["Continue normal monitoring"]

        alert = RegimeAlert(
            alert_id=alert_id,
            regime_state=self.current_regime_state,
            transition=transition,
            severity=severity,
            message=message,
            timestamp=datetime.now(),
            actions=actions
        )

        self.alerts.append(alert)
        transition.alert_sent = True

        tprint_warning(f"{message} (Significance: {transition.significance_score:.2f})")

    @handles_errors
    async def get_regime_statistics(self) -> Dict[str, Any]:
        """Get regime statistics and analysis."""
        if not self.regime_history:
            return {}

        # Regime duration analysis
        regime_durations = {}
        for regime in RegimeType:
            durations = []
            for state in self.regime_history:
                if state.primary_regime == regime:
                    durations.append(state.duration.total_seconds() / 60)  # minutes

            if durations:
                regime_durations[regime.value] = {
                    "count": len(durations),
                    "avg_duration": np.mean(durations),
                    "max_duration": np.max(durations),
                    "min_duration": np.min(durations),
                    "total_time": np.sum(durations)
                }

        # Transition analysis
        transition_matrix = await self._calculate_transition_matrix()

        # Stability analysis
        stability_counts = {}
        for state in self.regime_history:
            stability = state.stability.value
            stability_counts[stability] = stability_counts.get(stability, 0) + 1

        return {
            "total_regime_states": len(self.regime_history),
            "current_regime": self.current_regime_state.primary_regime.value if self.current_regime_state else None,
            "current_stability": self.current_regime_state.stability.value if self.current_regime_state else None,
            "regime_durations": regime_durations,
            "transition_matrix": transition_matrix,
            "stability_distribution": stability_counts,
            "total_transitions": len(self.transitions),
            "alert_count": len(self.alerts)
        }

    async def _calculate_transition_matrix(self) -> Dict[str, Dict[str, float]]:
        """Calculate regime transition probabilities."""
        transitions = [(t.from_regime, t.to_regime) for t in self.transitions[-1000:]]  # Last 1000 transitions

        if not transitions:
            return {}

        # Count transitions
        from_regimes = set(t[0] for t in transitions)
        to_regimes = set(t[1] for t in transitions)

        matrix = {}
        for from_regime in from_regimes:
            matrix[from_regime.value] = {}
            from_count = sum(1 for t in transitions if t[0] == from_regime)

            for to_regime in to_regimes:
                count = sum(1 for t in transitions if t[0] == from_regime and t[1] == to_regime)
                matrix[from_regime.value][to_regime.value] = count / from_count if from_count > 0 else 0

        return matrix

    @handles_errors
    async def get_regime_predictions(self, horizon_minutes: int = 60) -> Dict[str, Any]:
        """Predict future regime changes."""
        if not self.regime_history:
            return {"predictions": [], "confidence": 0.0}

        # Simple prediction based on recent transitions
        recent_transitions = self.transitions[-20:]  # Last 20 transitions

        if not recent_transitions:
            return {
                "predictions": [],
                "confidence": 0.0,
                "method": "no_data"
            }

        # Calculate transition probabilities
        predictions = []

        for transition in recent_transitions:
            if transition.significance_score > 0.5:
                predictions.append({
                    "from_regime": transition.from_regime.value,
                    "to_regime": transition.to_regime.value,
                    "probability": transition.significance_score,
                    "expected_timeframe": f"{horizon_minutes} minutes",
                    "confidence": transition.confidence
                })

        return {
            "predictions": predictions,
            "confidence": np.mean([p["probability"] for p in predictions]) if predictions else 0.0,
            "method": "transition_pattern_analysis"
        }

    async def get_pending_alerts(self) -> List[RegimeAlert]:
        """Get pending alerts that haven't been sent."""
        return [alert for alert in self.alerts if not (alert.transition and alert.transition.alert_sent)]

    async def get_alert_history(self, limit: int = 100) -> List[RegimeAlert]:
        """Get recent alert history."""
        return self.alert_history[-limit:]

    async def clear_old_alerts(self, days: int = 30) -> None:
        """Clear old alerts."""
        cutoff_date = datetime.now() - timedelta(days=days)

        self.alerts = [alert for alert in self.alerts if alert.timestamp > cutoff_date]
        self.transitions = [t for t in self.transitions if t.timestamp > cutoff_date]

        tprint_info(f"🧹 Cleared alerts older than {days} days")

    async def export_regime_data(self, format: str = "json") -> str:
        """Export regime monitoring data."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "current_regime": {
                "regime": self.current_regime_state.primary_regime.value,
                "confidence": self.current_regime_state.confidence,
                "stability": self.current_regime_state.stability.value,
                "duration_minutes": self.current_regime_state.duration.total_seconds() / 60
            } if self.current_regime_state else None,
            "statistics": await self.get_regime_statistics(),
            "recent_transitions": [
                {
                    "from_regime": t.from_regime.value,
                    "to_regime": t.to_regime.value,
                    "timestamp": t.timestamp.isoformat(),
                    "significance": t.significance_score
                }
                for t in self.transitions[-10:]  # Last 10 transitions
            ],
            "recent_alerts": [
                {
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in self.alerts[-10:]  # Last 10 alerts
            ]
        }

        if format == "json":
            return json.dumps(data, indent=2, default=str)
        else:
            return pd.DataFrame(data["recent_transitions"]).to_csv(index=False)

    async def cleanup(self) -> None:
        """Clean up resources."""
        self.regime_history.clear()
        self.transitions.clear()
        self.pending_transitions.clear()
        self.alerts.clear()
        self.alert_history.clear()

        tprint_info("🧹 Regime Monitor cleaned up successfully")

# Factory functions
async def create_regime_monitor(config: Dict[str, Any]) -> RegimeMonitor:
    """Create and initialize a regime monitor."""
    monitor = RegimeMonitor(config)
    await monitor.initialize()
    return monitor

def get_regime_monitor() -> Optional[RegimeMonitor]:
    """Get the global regime monitor instance."""
    return None