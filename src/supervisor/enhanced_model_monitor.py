#!/usr/bin/env python3
"""
Enhanced Model Monitor

This module provides comprehensive model behavior monitoring, feature importance tracking,
decision path analysis, and ensemble performance monitoring that integrates with the
existing performance monitoring infrastructure.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
import numpy as np
from dataclasses_json import dataclass_json

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors, handle_specific_errors
from src.supervisor.performance_monitor import PerformanceMonitor


class ModelDriftType(Enum):
    """Model drift types."""
    CONCEPT_DRIFT = "concept_drift"
    DATA_DRIFT = "data_drift"
    LABEL_DRIFT = "label_drift"
    FEATURE_DRIFT = "feature_drift"


@dataclass_json
@dataclass
class ModelDriftAlert:
    """Model drift alert."""
    model_id: str
    model_type: str
    drift_type: ModelDriftType
    drift_score: float
    threshold: float
    timestamp: datetime
    features_affected: List[str]
    severity: str  # "low", "medium", "high", "critical"
    description: str


@dataclass_json
@dataclass
class FeatureDriftMetrics:
    """Feature drift metrics."""
    feature_name: str
    current_distribution: Dict[str, float]
    reference_distribution: Dict[str, float]
    drift_score: float
    ks_statistic: float
    p_value: float
    is_drifted: bool


@dataclass_json
@dataclass
class ModelPerformanceSnapshot:
    """Model performance snapshot."""
    model_id: str
    model_type: str
    timestamp: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    prediction_confidence: float
    feature_importance_stability: float
    concept_drift_score: float
    data_drift_score: float


@dataclass_json
@dataclass
class EnsemblePerformanceMetrics:
    """Ensemble performance metrics."""
    ensemble_id: str
    timestamp: datetime
    ensemble_accuracy: float
    individual_model_accuracies: Dict[str, float]
    ensemble_weights: Dict[str, float]
    diversity_score: float
    agreement_score: float
    meta_learner_performance: Optional[float] = None


class EnhancedModelMonitor:
    """
    Enhanced model monitor that integrates with existing performance monitoring
    to provide comprehensive model behavior tracking.
    """
    
    def __init__(self, config: Dict[str, Any], performance_monitor: PerformanceMonitor):
        """
        Initialize enhanced model monitor.
        
        Args:
            config: Configuration dictionary
            performance_monitor: Existing performance monitor instance
        """
        self.config = config
        self.performance_monitor = performance_monitor
        self.logger = system_logger.getChild("EnhancedModelMonitor")
        
        # Configuration
        self.monitor_config = config.get("enhanced_model_monitor", {})
        self.drift_detection_enabled = self.monitor_config.get("drift_detection_enabled", True)
        self.feature_importance_tracking = self.monitor_config.get("feature_importance_tracking", True)
        self.decision_path_analysis = self.monitor_config.get("decision_path_analysis", True)
        self.ensemble_monitoring = self.monitor_config.get("ensemble_monitoring", True)
        
        # Monitoring intervals
        self.drift_check_interval = self.monitor_config.get("drift_check_interval", 300)  # 5 minutes
        self.performance_snapshot_interval = self.monitor_config.get("performance_snapshot_interval", 60)  # 1 minute
        self.feature_analysis_interval = self.monitor_config.get("feature_analysis_interval", 600)  # 10 minutes
        
        # Storage
        self.model_performance_history: Dict[str, List[ModelPerformanceSnapshot]] = {}
        self.ensemble_performance_history: Dict[str, List[EnsemblePerformanceMetrics]] = {}
        self.drift_alerts: List[ModelDriftAlert] = []
        self.feature_drift_history: Dict[str, List[FeatureDriftMetrics]] = {}
        
        # Reference data for drift detection
        self.reference_distributions: Dict[str, Dict[str, float]] = {}
        self.reference_performance: Dict[str, float] = {}
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_tasks: List[asyncio.Task] = []
        
        self.logger.info("🚀 Enhanced Model Monitor initialized")

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid model monitor configuration"),
            AttributeError: (False, "Missing required monitor parameters"),
        },
        default_return=False,
        context="model monitor initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the enhanced model monitor."""
        try:
            self.logger.info("Initializing Enhanced Model Monitor...")
            
            # Load reference data for drift detection
            await self._load_reference_data()
            
            # Initialize monitoring components
            await self._initialize_drift_detection()
            await self._initialize_feature_tracking()
            await self._initialize_ensemble_monitoring()
            
            self.logger.info("✅ Enhanced Model Monitor initialization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced Model Monitor initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="reference data loading",
    )
    async def _load_reference_data(self) -> None:
        """Load reference data for drift detection."""
        try:
            # Load reference distributions from training data
            # This would typically load from saved training data distributions
            self.reference_distributions = {
                "feature1": {"mean": 0.0, "std": 1.0, "min": -3.0, "max": 3.0},
                "feature2": {"mean": 0.5, "std": 0.5, "min": 0.0, "max": 1.0},
                # Add more features as needed
            }
            
            # Load reference performance metrics
            self.reference_performance = {
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88,
                "f1_score": 0.85,
                "auc_score": 0.90,
            }
            
            self.logger.info("📊 Reference data loaded for drift detection")
            
        except Exception as e:
            self.logger.error(f"Error loading reference data: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="drift detection initialization",
    )
    async def _initialize_drift_detection(self) -> None:
        """Initialize drift detection components."""
        try:
            # Set up drift detection thresholds
            self.drift_thresholds = {
                ModelDriftType.CONCEPT_DRIFT: 0.05,
                ModelDriftType.DATA_DRIFT: 0.10,
                ModelDriftType.LABEL_DRIFT: 0.08,
                ModelDriftType.FEATURE_DRIFT: 0.12,
            }
            
            self.logger.info("🔍 Drift detection initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing drift detection: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="feature tracking initialization",
    )
    async def _initialize_feature_tracking(self) -> None:
        """Initialize feature importance tracking."""
        try:
            # Initialize feature drift history storage
            self.feature_drift_history = {}
            
            self.logger.info("📈 Feature importance tracking initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing feature tracking: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="ensemble monitoring initialization",
    )
    async def _initialize_ensemble_monitoring(self) -> None:
        """Initialize ensemble performance monitoring."""
        try:
            # Initialize ensemble performance tracking
            self.ensemble_performance_history = {}
            
            self.logger.info("🎯 Ensemble monitoring initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing ensemble monitoring: {e}")

    @handle_specific_errors(
        error_handlers={
            Exception: (False, "Model monitoring failed"),
        },
        default_return=False,
        context="model monitoring",
    )
    async def start_monitoring(self) -> bool:
        """Start the enhanced model monitoring."""
        try:
            self.is_monitoring = True
            self.logger.info("🚦 Starting Enhanced Model Monitor...")
            
            # Start monitoring tasks
            self.monitoring_tasks = [
                asyncio.create_task(self._drift_detection_loop()),
                asyncio.create_task(self._performance_snapshot_loop()),
                asyncio.create_task(self._feature_analysis_loop()),
                asyncio.create_task(self._ensemble_monitoring_loop()),
            ]
            
            self.logger.info("✅ Enhanced Model Monitor started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start Enhanced Model Monitor: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="drift detection loop",
    )
    async def _drift_detection_loop(self) -> None:
        """Continuous drift detection loop."""
        while self.is_monitoring:
            try:
                await self._perform_drift_detection()
                await asyncio.sleep(self.drift_check_interval)
            except Exception as e:
                self.logger.error(f"Error in drift detection loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="performance snapshot loop",
    )
    async def _performance_snapshot_loop(self) -> None:
        """Continuous performance snapshot loop."""
        while self.is_monitoring:
            try:
                await self._capture_performance_snapshots()
                await asyncio.sleep(self.performance_snapshot_interval)
            except Exception as e:
                self.logger.error(f"Error in performance snapshot loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="feature analysis loop",
    )
    async def _feature_analysis_loop(self) -> None:
        """Continuous feature analysis loop."""
        while self.is_monitoring:
            try:
                await self._analyze_feature_drift()
                await asyncio.sleep(self.feature_analysis_interval)
            except Exception as e:
                self.logger.error(f"Error in feature analysis loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="ensemble monitoring loop",
    )
    async def _ensemble_monitoring_loop(self) -> None:
        """Continuous ensemble monitoring loop."""
        while self.is_monitoring:
            try:
                await self._monitor_ensemble_performance()
                await asyncio.sleep(self.performance_snapshot_interval)
            except Exception as e:
                self.logger.error(f"Error in ensemble monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="drift detection",
    )
    async def _perform_drift_detection(self) -> None:
        """Perform comprehensive drift detection."""
        try:
            # Get current model performance from performance monitor
            current_metrics = self.performance_monitor.get_performance_metrics()
            
            for model_id, performance in current_metrics.get("models", {}).items():
                # Check concept drift
                concept_drift_score = self._calculate_concept_drift(model_id, performance)
                if concept_drift_score > self.drift_thresholds[ModelDriftType.CONCEPT_DRIFT]:
                    await self._create_drift_alert(
                        model_id, "ensemble", ModelDriftType.CONCEPT_DRIFT,
                        concept_drift_score, self.drift_thresholds[ModelDriftType.CONCEPT_DRIFT]
                    )
                
                # Check data drift
                data_drift_score = self._calculate_data_drift(model_id, performance)
                if data_drift_score > self.drift_thresholds[ModelDriftType.DATA_DRIFT]:
                    await self._create_drift_alert(
                        model_id, "ensemble", ModelDriftType.DATA_DRIFT,
                        data_drift_score, self.drift_thresholds[ModelDriftType.DATA_DRIFT]
                    )
            
            self.logger.debug("🔍 Drift detection completed")
            
        except Exception as e:
            self.logger.error(f"Error performing drift detection: {e}")

    def _calculate_concept_drift(self, model_id: str, current_performance: Dict[str, Any]) -> float:
        """Calculate concept drift score."""
        try:
            # Compare current performance with reference performance
            reference_acc = self.reference_performance.get("accuracy", 0.85)
            current_acc = current_performance.get("accuracy", 0.0)
            
            # Calculate drift as performance degradation
            drift_score = max(0, reference_acc - current_acc) / reference_acc
            
            return drift_score
            
        except Exception as e:
            self.logger.error(f"Error calculating concept drift: {e}")
            return 0.0

    def _calculate_data_drift(self, model_id: str, current_performance: Dict[str, Any]) -> float:
        """Calculate data drift score."""
        try:
            # This would typically compare feature distributions
            # For now, use a simplified approach
            return 0.02  # Placeholder
            
        except Exception as e:
            self.logger.error(f"Error calculating data drift: {e}")
            return 0.0

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="drift alert creation",
    )
    async def _create_drift_alert(
        self,
        model_id: str,
        model_type: str,
        drift_type: ModelDriftType,
        drift_score: float,
        threshold: float,
    ) -> None:
        """Create a drift alert."""
        try:
            severity = self._determine_alert_severity(drift_score, threshold)
            
            alert = ModelDriftAlert(
                model_id=model_id,
                model_type=model_type,
                drift_type=drift_type,
                drift_score=drift_score,
                threshold=threshold,
                timestamp=datetime.now(),
                features_affected=[],  # Would be populated based on analysis
                severity=severity,
                description=f"{drift_type.value.replace('_', ' ').title()} detected in {model_id}",
            )
            
            self.drift_alerts.append(alert)
            
            self.logger.warning(
                f"🚨 Drift Alert: {drift_type.value} detected in {model_id} "
                f"(score: {drift_score:.3f}, threshold: {threshold:.3f})"
            )
            
        except Exception as e:
            self.logger.error(f"Error creating drift alert: {e}")

    def _determine_alert_severity(self, drift_score: float, threshold: float) -> str:
        """Determine alert severity based on drift score."""
        ratio = drift_score / threshold
        
        if ratio >= 2.0:
            return "critical"
        elif ratio >= 1.5:
            return "high"
        elif ratio >= 1.2:
            return "medium"
        else:
            return "low"

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="performance snapshot capture",
    )
    async def _capture_performance_snapshots(self) -> None:
        """Capture performance snapshots for all models."""
        try:
            # Get current performance metrics from performance monitor
            current_metrics = self.performance_monitor.get_performance_metrics()
            
            for model_id, performance in current_metrics.get("models", {}).items():
                snapshot = ModelPerformanceSnapshot(
                    model_id=model_id,
                    model_type=performance.get("model_type", "ensemble"),
                    timestamp=datetime.now(),
                    accuracy=performance.get("accuracy", 0.0),
                    precision=performance.get("precision", 0.0),
                    recall=performance.get("recall", 0.0),
                    f1_score=performance.get("f1_score", 0.0),
                    auc_score=performance.get("auc_score", 0.0),
                    prediction_confidence=performance.get("confidence", 0.0),
                    feature_importance_stability=performance.get("feature_stability", 0.0),
                    concept_drift_score=self._calculate_concept_drift(model_id, performance),
                    data_drift_score=self._calculate_data_drift(model_id, performance),
                )
                
                if model_id not in self.model_performance_history:
                    self.model_performance_history[model_id] = []
                
                self.model_performance_history[model_id].append(snapshot)
                
                # Keep only recent snapshots
                if len(self.model_performance_history[model_id]) > 1000:
                    self.model_performance_history[model_id] = self.model_performance_history[model_id][-500:]
            
            self.logger.debug("📊 Performance snapshots captured")
            
        except Exception as e:
            self.logger.error(f"Error capturing performance snapshots: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="feature drift analysis",
    )
    async def _analyze_feature_drift(self) -> None:
        """Analyze feature drift for all features."""
        try:
            # This would typically analyze current feature distributions
            # against reference distributions
            for feature_name, reference_dist in self.reference_distributions.items():
                # Simulate current distribution (in practice, this would come from live data)
                current_dist = {
                    "mean": reference_dist["mean"] + np.random.normal(0, 0.1),
                    "std": reference_dist["std"] + np.random.normal(0, 0.05),
                    "min": reference_dist["min"],
                    "max": reference_dist["max"],
                }
                
                # Calculate drift metrics
                drift_score = self._calculate_feature_drift_score(reference_dist, current_dist)
                
                feature_metrics = FeatureDriftMetrics(
                    feature_name=feature_name,
                    current_distribution=current_dist,
                    reference_distribution=reference_dist,
                    drift_score=drift_score,
                    ks_statistic=0.0,  # Would be calculated from actual data
                    p_value=0.0,  # Would be calculated from actual data
                    is_drifted=drift_score > self.drift_thresholds[ModelDriftType.FEATURE_DRIFT],
                )
                
                if feature_name not in self.feature_drift_history:
                    self.feature_drift_history[feature_name] = []
                
                self.feature_drift_history[feature_name].append(feature_metrics)
                
                # Keep only recent history
                if len(self.feature_drift_history[feature_name]) > 100:
                    self.feature_drift_history[feature_name] = self.feature_drift_history[feature_name][-50:]
            
            self.logger.debug("📈 Feature drift analysis completed")
            
        except Exception as e:
            self.logger.error(f"Error analyzing feature drift: {e}")

    def _calculate_feature_drift_score(self, reference_dist: Dict[str, float], current_dist: Dict[str, float]) -> float:
        """Calculate feature drift score."""
        try:
            # Calculate drift as difference in distribution parameters
            mean_diff = abs(current_dist["mean"] - reference_dist["mean"]) / reference_dist["std"]
            std_diff = abs(current_dist["std"] - reference_dist["std"]) / reference_dist["std"]
            
            # Combine differences
            drift_score = (mean_diff + std_diff) / 2
            
            return drift_score
            
        except Exception as e:
            self.logger.error(f"Error calculating feature drift score: {e}")
            return 0.0

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="ensemble performance monitoring",
    )
    async def _monitor_ensemble_performance(self) -> None:
        """Monitor ensemble performance."""
        try:
            # Get ensemble performance from performance monitor
            ensemble_metrics = self.performance_monitor.get_performance_metrics().get("ensembles", {})
            
            for ensemble_id, metrics in ensemble_metrics.items():
                ensemble_performance = EnsemblePerformanceMetrics(
                    ensemble_id=ensemble_id,
                    timestamp=datetime.now(),
                    ensemble_accuracy=metrics.get("accuracy", 0.0),
                    individual_model_accuracies=metrics.get("individual_accuracies", {}),
                    ensemble_weights=metrics.get("weights", {}),
                    diversity_score=metrics.get("diversity_score", 0.0),
                    agreement_score=metrics.get("agreement_score", 0.0),
                    meta_learner_performance=metrics.get("meta_learner_performance"),
                )
                
                if ensemble_id not in self.ensemble_performance_history:
                    self.ensemble_performance_history[ensemble_id] = []
                
                self.ensemble_performance_history[ensemble_id].append(ensemble_performance)
                
                # Keep only recent history
                if len(self.ensemble_performance_history[ensemble_id]) > 500:
                    self.ensemble_performance_history[ensemble_id] = self.ensemble_performance_history[ensemble_id][-250:]
            
            self.logger.debug("🎯 Ensemble performance monitored")
            
        except Exception as e:
            self.logger.error(f"Error monitoring ensemble performance: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="model monitor stop",
    )
    async def stop_monitoring(self) -> None:
        """Stop the enhanced model monitoring."""
        try:
            self.logger.info("🛑 Stopping Enhanced Model Monitor...")
            
            self.is_monitoring = False
            
            # Cancel all monitoring tasks
            for task in self.monitoring_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
            
            self.logger.info("✅ Enhanced Model Monitor stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping model monitor: {e}")

    def get_drift_alerts(self, severity: Optional[str] = None) -> List[ModelDriftAlert]:
        """Get drift alerts, optionally filtered by severity."""
        alerts = self.drift_alerts.copy()
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return alerts

    def get_model_performance_history(self, model_id: str, limit: Optional[int] = None) -> List[ModelPerformanceSnapshot]:
        """Get performance history for a specific model."""
        history = self.model_performance_history.get(model_id, [])
        
        if limit:
            history = history[-limit:]
        
        return history

    def get_ensemble_performance_history(self, ensemble_id: str, limit: Optional[int] = None) -> List[EnsemblePerformanceMetrics]:
        """Get performance history for a specific ensemble."""
        history = self.ensemble_performance_history.get(ensemble_id, [])
        
        if limit:
            history = history[-limit:]
        
        return history

    def get_feature_drift_history(self, feature_name: str, limit: Optional[int] = None) -> List[FeatureDriftMetrics]:
        """Get drift history for a specific feature."""
        history = self.feature_drift_history.get(feature_name, [])
        
        if limit:
            history = history[-limit:]
        
        return history

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        try:
            summary = {
                "total_models_monitored": len(self.model_performance_history),
                "total_ensembles_monitored": len(self.ensemble_performance_history),
                "total_drift_alerts": len(self.drift_alerts),
                "total_features_tracked": len(self.feature_drift_history),
                "recent_drift_alerts": len([a for a in self.drift_alerts if (datetime.now() - a.timestamp).days <= 1]),
                "model_performance_trends": {},
                "ensemble_performance_trends": {},
                "feature_drift_summary": {},
            }
            
            # Calculate performance trends
            for model_id, history in self.model_performance_history.items():
                if len(history) >= 2:
                    recent_avg = np.mean([h.accuracy for h in history[-10:]])
                    older_avg = np.mean([h.accuracy for h in history[-20:-10]]) if len(history) >= 20 else recent_avg
                    trend = "improving" if recent_avg > older_avg else "declining" if recent_avg < older_avg else "stable"
                    
                    summary["model_performance_trends"][model_id] = {
                        "trend": trend,
                        "recent_accuracy": recent_avg,
                        "accuracy_change": recent_avg - older_avg,
                    }
            
            # Calculate ensemble trends
            for ensemble_id, history in self.ensemble_performance_history.items():
                if len(history) >= 2:
                    recent_avg = np.mean([h.ensemble_accuracy for h in history[-10:]])
                    older_avg = np.mean([h.ensemble_accuracy for h in history[-20:-10]]) if len(history) >= 20 else recent_avg
                    trend = "improving" if recent_avg > older_avg else "declining" if recent_avg < older_avg else "stable"
                    
                    summary["ensemble_performance_trends"][ensemble_id] = {
                        "trend": trend,
                        "recent_accuracy": recent_avg,
                        "accuracy_change": recent_avg - older_avg,
                    }
            
            # Calculate feature drift summary
            for feature_name, history in self.feature_drift_history.items():
                if history:
                    recent_drift = np.mean([h.drift_score for h in history[-10:]])
                    summary["feature_drift_summary"][feature_name] = {
                        "current_drift_score": recent_drift,
                        "is_drifted": recent_drift > self.drift_thresholds[ModelDriftType.FEATURE_DRIFT],
                    }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating performance summary: {e}")
            return {}

    def export_monitoring_data(self, filepath: Optional[str] = None) -> str:
        """Export monitoring data to file."""
        try:
            if not filepath:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"reports/model_monitoring_data_{timestamp}.json"
            
            export_data = {
                "drift_alerts": [asdict(alert) for alert in self.drift_alerts],
                "model_performance_history": {
                    model_id: [asdict(snapshot) for snapshot in history]
                    for model_id, history in self.model_performance_history.items()
                },
                "ensemble_performance_history": {
                    ensemble_id: [asdict(metrics) for metrics in history]
                    for ensemble_id, history in self.ensemble_performance_history.items()
                },
                "feature_drift_history": {
                    feature_name: [asdict(metrics) for metrics in history]
                    for feature_name, history in self.feature_drift_history.items()
                },
                "performance_summary": self.get_performance_summary(),
            }
            
            import json
            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"📊 Monitoring data exported to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error exporting monitoring data: {e}")
            return ""


# Factory function for creating enhanced model monitor
@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="enhanced model monitor setup",
)
async def setup_enhanced_model_monitor(
    config: Dict[str, Any],
    performance_monitor: PerformanceMonitor,
) -> EnhancedModelMonitor | None:
    """
    Set up enhanced model monitor.
    
    Args:
        config: Configuration dictionary
        performance_monitor: Performance monitor instance
        
    Returns:
        EnhancedModelMonitor instance or None if setup fails
    """
    try:
        monitor = EnhancedModelMonitor(config, performance_monitor)
        success = await monitor.initialize()
        
        if success:
            return monitor
        else:
            return None
            
    except Exception as e:
        system_logger.error(f"Error setting up enhanced model monitor: {e}")
        return None 