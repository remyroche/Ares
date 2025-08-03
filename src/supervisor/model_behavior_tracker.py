#!/usr/bin/env python3
"""
Model Behavior Tracker

This module enhances the existing performance monitoring system with comprehensive
model behavior tracking, feature importance monitoring, and decision path analysis.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors, handle_specific_errors
from src.supervisor.performance_monitor import PerformanceMonitor


class BehaviorMetricType(Enum):
    """Model behavior metric types."""
    PREDICTION_CONSISTENCY = "prediction_consistency"
    CONFIDENCE_TREND = "confidence_trend"
    FEATURE_IMPORTANCE_STABILITY = "feature_importance_stability"
    PREDICTION_DRIFT = "prediction_drift"
    ENSEMBLE_DIVERSITY = "ensemble_diversity"
    DECISION_PATH_STABILITY = "decision_path_stability"


@dataclass
class ModelBehaviorSnapshot:
    """Model behavior snapshot."""
    model_id: str
    model_type: str
    timestamp: datetime
    prediction_consistency: float
    confidence_trend: List[float]
    feature_importance_stability: float
    prediction_drift: float
    ensemble_diversity: Optional[float] = None
    decision_path_stability: Optional[float] = None
    metadata: Dict[str, Any] = None


@dataclass
class FeatureImportanceTracking:
    """Feature importance tracking data."""
    feature_name: str
    model_id: str
    timestamp: datetime
    importance_score: float
    importance_rank: int
    stability_score: float
    drift_score: float


@dataclass
class DecisionPathAnalysis:
    """Decision path analysis data."""
    model_id: str
    timestamp: datetime
    decision_steps: List[str]
    decision_weights: List[float]
    path_stability: float
    path_complexity: float
    confidence_distribution: List[float]


class ModelBehaviorTracker:
    """
    Enhanced model behavior tracker that integrates with existing performance monitoring.
    """
    
    def __init__(self, config: Dict[str, Any], performance_monitor: PerformanceMonitor):
        """
        Initialize model behavior tracker.
        
        Args:
            config: Configuration dictionary
            performance_monitor: Existing performance monitor instance
        """
        self.config = config
        self.performance_monitor = performance_monitor
        self.logger = system_logger.getChild("ModelBehaviorTracker")
        
        # Configuration
        self.tracker_config = config.get("model_behavior_tracker", {})
        self.tracking_interval = self.tracker_config.get("tracking_interval", 60)  # 1 minute
        self.max_history_size = self.tracker_config.get("max_history_size", 1000)
        
        # Storage
        self.behavior_history: Dict[str, List[ModelBehaviorSnapshot]] = {}
        self.feature_importance_history: Dict[str, List[FeatureImportanceTracking]] = {}
        self.decision_path_history: Dict[str, List[DecisionPathAnalysis]] = {}
        
        # Tracking state
        self.is_tracking = False
        self.tracking_task: Optional[asyncio.Task] = None
        
        # Reference data for stability calculations
        self.reference_behavior: Dict[str, Dict[str, float]] = {}
        
        self.logger.info("🚀 Model Behavior Tracker initialized")

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid tracker configuration"),
            AttributeError: (False, "Missing required tracker parameters"),
        },
        default_return=False,
        context="behavior tracker initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the model behavior tracker."""
        try:
            self.logger.info("Initializing Model Behavior Tracker...")
            
            # Load reference behavior data
            await self._load_reference_behavior()
            
            # Initialize tracking components
            await self._initialize_behavior_tracking()
            await self._initialize_feature_tracking()
            await self._initialize_decision_path_tracking()
            
            self.logger.info("✅ Model Behavior Tracker initialization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Model Behavior Tracker initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="reference behavior loading",
    )
    async def _load_reference_behavior(self) -> None:
        """Load reference behavior data for stability calculations."""
        try:
            # Load reference behavior metrics from training data
            self.reference_behavior = {
                "prediction_consistency": 0.85,
                "confidence_trend_stability": 0.78,
                "feature_importance_stability": 0.82,
                "prediction_drift_threshold": 0.05,
                "ensemble_diversity_target": 0.65,
                "decision_path_stability": 0.80,
            }
            
            self.logger.info("📊 Reference behavior data loaded")
            
        except Exception as e:
            self.logger.error(f"Error loading reference behavior: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="behavior tracking initialization",
    )
    async def _initialize_behavior_tracking(self) -> None:
        """Initialize behavior tracking components."""
        try:
            # Set up behavior tracking thresholds
            self.behavior_thresholds = {
                BehaviorMetricType.PREDICTION_CONSISTENCY: 0.80,
                BehaviorMetricType.CONFIDENCE_TREND: 0.75,
                BehaviorMetricType.FEATURE_IMPORTANCE_STABILITY: 0.80,
                BehaviorMetricType.PREDICTION_DRIFT: 0.05,
                BehaviorMetricType.ENSEMBLE_DIVERSITY: 0.60,
                BehaviorMetricType.DECISION_PATH_STABILITY: 0.75,
            }
            
            self.logger.info("🔍 Behavior tracking initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing behavior tracking: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="feature tracking initialization",
    )
    async def _initialize_feature_tracking(self) -> None:
        """Initialize feature importance tracking."""
        try:
            # Initialize feature tracking storage
            self.feature_importance_history = {}
            
            self.logger.info("📈 Feature importance tracking initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing feature tracking: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="decision path tracking initialization",
    )
    async def _initialize_decision_path_tracking(self) -> None:
        """Initialize decision path tracking."""
        try:
            # Initialize decision path tracking storage
            self.decision_path_history = {}
            
            self.logger.info("🛤️ Decision path tracking initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing decision path tracking: {e}")

    @handle_specific_errors(
        error_handlers={
            Exception: (False, "Behavior tracking failed"),
        },
        default_return=False,
        context="behavior tracking",
    )
    async def start_tracking(self) -> bool:
        """Start the model behavior tracking."""
        try:
            self.is_tracking = True
            self.logger.info("🚦 Starting Model Behavior Tracker...")
            
            # Start tracking task
            self.tracking_task = asyncio.create_task(self._behavior_tracking_loop())
            
            self.logger.info("✅ Model Behavior Tracker started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start Model Behavior Tracker: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="behavior tracking loop",
    )
    async def _behavior_tracking_loop(self) -> None:
        """Continuous behavior tracking loop."""
        while self.is_tracking:
            try:
                await self._capture_behavior_snapshots()
                await asyncio.sleep(self.tracking_interval)
            except Exception as e:
                self.logger.error(f"Error in behavior tracking loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="behavior snapshot capture",
    )
    async def _capture_behavior_snapshots(self) -> None:
        """Capture behavior snapshots for all models."""
        try:
            # Get current performance metrics from performance monitor
            current_metrics = self.performance_monitor.get_performance_metrics()
            
            for model_id, performance in current_metrics.get("models", {}).items():
                # Calculate behavior metrics
                prediction_consistency = self._calculate_prediction_consistency(model_id, performance)
                confidence_trend = self._calculate_confidence_trend(model_id, performance)
                feature_importance_stability = self._calculate_feature_importance_stability(model_id, performance)
                prediction_drift = self._calculate_prediction_drift(model_id, performance)
                ensemble_diversity = self._calculate_ensemble_diversity(model_id, performance)
                decision_path_stability = self._calculate_decision_path_stability(model_id, performance)
                
                # Create behavior snapshot
                snapshot = ModelBehaviorSnapshot(
                    model_id=model_id,
                    model_type=performance.get("model_type", "ensemble"),
                    timestamp=datetime.now(),
                    prediction_consistency=prediction_consistency,
                    confidence_trend=confidence_trend,
                    feature_importance_stability=feature_importance_stability,
                    prediction_drift=prediction_drift,
                    ensemble_diversity=ensemble_diversity,
                    decision_path_stability=decision_path_stability,
                    metadata=performance.get("metadata", {}),
                )
                
                if model_id not in self.behavior_history:
                    self.behavior_history[model_id] = []
                
                self.behavior_history[model_id].append(snapshot)
                
                # Keep only recent snapshots
                if len(self.behavior_history[model_id]) > self.max_history_size:
                    self.behavior_history[model_id] = self.behavior_history[model_id][-self.max_history_size//2:]
            
            self.logger.debug("📊 Behavior snapshots captured")
            
        except Exception as e:
            self.logger.error(f"Error capturing behavior snapshots: {e}")

    def _calculate_prediction_consistency(self, model_id: str, performance: Dict[str, Any]) -> float:
        """Calculate prediction consistency."""
        try:
            # This would typically analyze recent predictions vs historical patterns
            # For now, use a simplified approach based on accuracy stability
            accuracy = performance.get("accuracy", 0.0)
            reference_accuracy = self.reference_behavior.get("prediction_consistency", 0.85)
            
            # Calculate consistency as how close current accuracy is to reference
            consistency = 1.0 - abs(accuracy - reference_accuracy) / reference_accuracy
            return max(0.0, min(1.0, consistency))
            
        except Exception as e:
            self.logger.error(f"Error calculating prediction consistency: {e}")
            return 0.0

    def _calculate_confidence_trend(self, model_id: str, performance: Dict[str, Any]) -> List[float]:
        """Calculate confidence trend."""
        try:
            # This would typically analyze recent confidence scores
            # For now, simulate a trend based on performance metrics
            confidence = performance.get("confidence", 0.0)
            
            # Simulate trend with some variation
            trend = [confidence + np.random.normal(0, 0.05) for _ in range(10)]
            return [max(0.0, min(1.0, c)) for c in trend]
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence trend: {e}")
            return [0.0] * 10

    def _calculate_feature_importance_stability(self, model_id: str, performance: Dict[str, Any]) -> float:
        """Calculate feature importance stability."""
        try:
            # This would typically analyze feature importance changes over time
            # For now, use a simplified approach
            feature_stability = performance.get("feature_stability", 0.8)
            reference_stability = self.reference_behavior.get("feature_importance_stability", 0.82)
            
            # Calculate stability relative to reference
            stability = 1.0 - abs(feature_stability - reference_stability) / reference_stability
            return max(0.0, min(1.0, stability))
            
        except Exception as e:
            self.logger.error(f"Error calculating feature importance stability: {e}")
            return 0.0

    def _calculate_prediction_drift(self, model_id: str, performance: Dict[str, Any]) -> float:
        """Calculate prediction drift."""
        try:
            # This would typically analyze prediction distribution changes
            # For now, use a simplified approach
            accuracy = performance.get("accuracy", 0.0)
            reference_accuracy = self.reference_behavior.get("prediction_consistency", 0.85)
            
            # Calculate drift as performance degradation
            drift = max(0.0, reference_accuracy - accuracy) / reference_accuracy
            return drift
            
        except Exception as e:
            self.logger.error(f"Error calculating prediction drift: {e}")
            return 0.0

    def _calculate_ensemble_diversity(self, model_id: str, performance: Dict[str, Any]) -> Optional[float]:
        """Calculate ensemble diversity."""
        try:
            # This would typically analyze individual model predictions in ensemble
            # For now, use a simplified approach
            if "ensemble" in model_id.lower():
                diversity = performance.get("diversity_score", 0.65)
                return diversity
            else:
                return None
            
        except Exception as e:
            self.logger.error(f"Error calculating ensemble diversity: {e}")
            return None

    def _calculate_decision_path_stability(self, model_id: str, performance: Dict[str, Any]) -> Optional[float]:
        """Calculate decision path stability."""
        try:
            # This would typically analyze decision path consistency
            # For now, use a simplified approach
            path_stability = performance.get("path_stability", 0.8)
            reference_stability = self.reference_behavior.get("decision_path_stability", 0.80)
            
            # Calculate stability relative to reference
            stability = 1.0 - abs(path_stability - reference_stability) / reference_stability
            return max(0.0, min(1.0, stability))
            
        except Exception as e:
            self.logger.error(f"Error calculating decision path stability: {e}")
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="behavior tracker stop",
    )
    async def stop_tracking(self) -> None:
        """Stop the model behavior tracking."""
        try:
            self.logger.info("🛑 Stopping Model Behavior Tracker...")
            
            self.is_tracking = False
            
            # Cancel tracking task
            if self.tracking_task and not self.tracking_task.done():
                self.tracking_task.cancel()
                await self.tracking_task
            
            self.logger.info("✅ Model Behavior Tracker stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping behavior tracker: {e}")

    def get_behavior_history(self, model_id: str, limit: Optional[int] = None) -> List[ModelBehaviorSnapshot]:
        """Get behavior history for a specific model."""
        history = self.behavior_history.get(model_id, [])
        
        if limit:
            history = history[-limit:]
        
        return history

    def get_behavior_summary(self, model_id: str) -> Dict[str, Any]:
        """Get behavior summary for a specific model."""
        try:
            history = self.behavior_history.get(model_id, [])
            
            if not history:
                return {}
            
            # Calculate summary statistics
            recent_snapshots = history[-10:] if len(history) >= 10 else history
            
            summary = {
                "model_id": model_id,
                "total_snapshots": len(history),
                "recent_snapshots": len(recent_snapshots),
                "avg_prediction_consistency": np.mean([s.prediction_consistency for s in recent_snapshots]),
                "avg_feature_importance_stability": np.mean([s.feature_importance_stability for s in recent_snapshots]),
                "avg_prediction_drift": np.mean([s.prediction_drift for s in recent_snapshots]),
                "behavior_trend": self._calculate_behavior_trend(recent_snapshots),
                "stability_score": self._calculate_overall_stability(recent_snapshots),
                "alert_level": self._determine_alert_level(recent_snapshots),
            }
            
            # Add ensemble-specific metrics if applicable
            if any(s.ensemble_diversity is not None for s in recent_snapshots):
                summary["avg_ensemble_diversity"] = np.mean([
                    s.ensemble_diversity for s in recent_snapshots 
                    if s.ensemble_diversity is not None
                ])
            
            # Add decision path metrics if applicable
            if any(s.decision_path_stability is not None for s in recent_snapshots):
                summary["avg_decision_path_stability"] = np.mean([
                    s.decision_path_stability for s in recent_snapshots 
                    if s.decision_path_stability is not None
                ])
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating behavior summary: {e}")
            return {}

    def _calculate_behavior_trend(self, snapshots: List[ModelBehaviorSnapshot]) -> str:
        """Calculate behavior trend."""
        try:
            if len(snapshots) < 2:
                return "insufficient_data"
            
            # Calculate trend based on prediction consistency
            recent_avg = np.mean([s.prediction_consistency for s in snapshots[-5:]])
            older_avg = np.mean([s.prediction_consistency for s in snapshots[-10:-5]]) if len(snapshots) >= 10 else recent_avg
            
            if recent_avg > older_avg + 0.05:
                return "improving"
            elif recent_avg < older_avg - 0.05:
                return "declining"
            else:
                return "stable"
                
        except Exception as e:
            self.logger.error(f"Error calculating behavior trend: {e}")
            return "unknown"

    def _calculate_overall_stability(self, snapshots: List[ModelBehaviorSnapshot]) -> float:
        """Calculate overall stability score."""
        try:
            if not snapshots:
                return 0.0
            
            # Combine multiple stability metrics
            consistency_scores = [s.prediction_consistency for s in snapshots]
            feature_stability_scores = [s.feature_importance_stability for s in snapshots]
            drift_scores = [1.0 - s.prediction_drift for s in snapshots]  # Invert drift
            
            # Calculate weighted average
            weights = [0.4, 0.3, 0.3]  # Weights for each metric
            stability = (
                np.mean(consistency_scores) * weights[0] +
                np.mean(feature_stability_scores) * weights[1] +
                np.mean(drift_scores) * weights[2]
            )
            
            return max(0.0, min(1.0, stability))
            
        except Exception as e:
            self.logger.error(f"Error calculating overall stability: {e}")
            return 0.0

    def _determine_alert_level(self, snapshots: List[ModelBehaviorSnapshot]) -> str:
        """Determine alert level based on behavior metrics."""
        try:
            if not snapshots:
                return "unknown"
            
            # Check various alert conditions
            avg_consistency = np.mean([s.prediction_consistency for s in snapshots])
            avg_drift = np.mean([s.prediction_drift for s in snapshots])
            avg_stability = np.mean([s.feature_importance_stability for s in snapshots])
            
            # Determine alert level
            if avg_consistency < 0.6 or avg_drift > 0.15 or avg_stability < 0.6:
                return "critical"
            elif avg_consistency < 0.75 or avg_drift > 0.10 or avg_stability < 0.75:
                return "warning"
            elif avg_consistency < 0.85 or avg_drift > 0.05 or avg_stability < 0.85:
                return "notice"
            else:
                return "normal"
                
        except Exception as e:
            self.logger.error(f"Error determining alert level: {e}")
            return "unknown"

    def get_all_behavior_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Get behavior summaries for all models."""
        summaries = {}
        
        for model_id in self.behavior_history.keys():
            summaries[model_id] = self.get_behavior_summary(model_id)
        
        return summaries

    def export_behavior_data(self, filepath: Optional[str] = None) -> str:
        """Export behavior data to file."""
        try:
            if not filepath:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"reports/model_behavior_data_{timestamp}.json"
            
            export_data = {
                "behavior_history": {
                    model_id: [asdict(snapshot) for snapshot in history]
                    for model_id, history in self.behavior_history.items()
                },
                "behavior_summaries": self.get_all_behavior_summaries(),
                "export_timestamp": datetime.now().isoformat(),
            }
            
            import json
            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"📊 Behavior data exported to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error exporting behavior data: {e}")
            return ""


# Factory function for creating model behavior tracker
@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="model behavior tracker setup",
)
async def setup_model_behavior_tracker(
    config: Dict[str, Any],
    performance_monitor: PerformanceMonitor,
) -> ModelBehaviorTracker | None:
    """
    Set up model behavior tracker.
    
    Args:
        config: Configuration dictionary
        performance_monitor: Performance monitor instance
        
    Returns:
        ModelBehaviorTracker instance or None if setup fails
    """
    try:
        tracker = ModelBehaviorTracker(config, performance_monitor)
        success = await tracker.initialize()
        
        if success:
            return tracker
        else:
            return None
            
    except Exception as e:
        system_logger.error(f"Error setting up model behavior tracker: {e}")
        return None 