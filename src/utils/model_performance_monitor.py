#!/usr / bin / env python3
"""
Model Performance Monitoring System

This module provides comprehensive monitoring and tracking of ML model performance
across all steps in the enhanced training manager. It tracks accuracy, precision,
recall, F1 scores, and other key metrics with proper error handling and quality assurance.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.centralized_decorators import (
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards
handle_errors,
monitor_step_execution,
secure_step_execution,
validate_pipeline_step,
with_tracing_span,
quality_gate,
ensure_data_integrity
)
from src.utils.logger import system_logger

class ModelPerformanceMonitor:

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="modelperformancemonitor initialization",
    )
    async def initialize(self) -> bool:
        """Initialize ModelPerformanceMonitor."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
    passpassself.logger.info("Implementation placeholder - needs specific logic")
class ModelPerformanceMonitor:
    passself.logger.info("Implementation placeholder - needs specific logic")
class ModelPerformanceMonitor:
    pass"""Comprehensive model performance monitoring system."""

def __init__(...) -> ...:
    """..."""
    passself.config, config
self.logger, system_logger.getChild("ModelPerformanceMonitor")

# Performance tracking
self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
self.current_metrics: Dict[str, Dict[str, Any]] = {}
self.model_registry: Dict[str, Dict[str, Any]] = {}

# Configuration
self.monitor_config, config.get("model_performance_monitor", {})
self.enable_real_time_monitoring, self.monitor_config.get("enable_real_time_monitoring", True)
self.performance_thresholds, self.monitor_config.get("performance_thresholds", {
"min_accuracy": 0.6,
"min_precision": 0.5,
"min_recall": 0.5,
"min_f1_score": 0.5,
"max_drift": 0.1
})

# Storage
self.results_dir, Path(self.monitor_config.get("results_dir", "results / model_performance"))
self.results_dir.mkdir(parents = True, exist_ok = True)

# Initialize performance tracking
self._initialize_performance_tracking()

@handle_errors(
exceptions=(Exception,),
default_return = False,
context="model_performance_monitor_initialization"
)
def _initialize_performance_tracking(...) -> ...:
    """..."""
    passself.logger.info("🔧 Initializing model performance tracking...")

# Initialize performance history for each model type
model_types = [
"hmm_regime_discovery",
"analyst_enhancement",
"tactician_specialist",
"confidence_calibration",
"unified_regime_intelligence"
]

for model_type in model_types:
    passself.performance_history[model_type] = []
self.current_metrics[model_type] = {}
self.model_registry[model_type] = {
"created_at": datetime.now().isoformat(),
"last_updated": datetime.now().isoformat(),
"total_runs": 0,
"successful_runs": 0,
"failed_runs": 0
}

self.logger.info(f"✅ Performance tracking initialized for {len(model_types)} model types")

@validate_pipeline_step(
step_name="model_performance_monitoring",
validation_level="CRITICAL",
enable_rollback = True,
max_retries = 2
)
@ensure_data_integrity(
check_schema = True,
check_constraints = True,
validate_relationships = True
)
@monitor_step_execution(
enable_timing = True,
enable_memory_monitoring = True,
enable_progress_tracking = True
)
@secure_step_execution(
error_handling = True,
rollback_on_failure = True,
data_validation = True,
resource_cleanup = True
)
@with_tracing_span("track_model_performance")
@quality_gate(
min_quality_score = 0.7,
max_correlation = 0.95,
required_grade="C"
)
@handle_errors(
exceptions=(Exception,),
default_return = False,
context="track_model_performance"
)
async def track_model_performance(
self,
model_type: str,
model_name: str,
predictions: np.ndarray,
actual_values: np.ndarray,
confidence_scores: Optional[np.ndarray] = None,
additional_metrics: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
        """Track performance metrics for a specific model.

Args:
    passmodel_type: Type of model (e.g., "hmm_regime_discovery")
model_name: Name of the specific model
predictions: Model predictions
actual_values: Actual / true values
confidence_scores: Model confidence scores (optional)
additional_metrics: Additional metrics to track (optional)

Returns:
            Dict containing performance metrics and status
"""
try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
self.logger.info(f"📊 Tracking performance for {model_type}:{model_name}")

# Calculate basic metrics
metrics, await self._calculate_performance_metrics(
predictions, actual_values, confidence_scores
)

# Add additional metrics if provided
if additional_metrics:
    passmetrics.update(additional_metrics)

# Add metadata
metrics.update({
"model_type": model_type,
"model_name": model_name,
"timestamp": datetime.now().isoformat(),
"predictions_shape": predictions.shape,
"actual_values_shape": actual_values.shape
})

# Store metrics
await self._store_performance_metrics(model_type, model_name, metrics)

# Check performance thresholds
performance_status, await self._check_performance_thresholds(metrics)
metrics["performance_status"] = performance_status

# Update model registry
await self._update_model_registry(model_type, model_name, metrics, performance_status)

# Log performance summary
await self._log_performance_summary(model_type, model_name, metrics, performance_status)

return metrics

except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"❌ Error tracking model performance: {e}")
return {
"success": False,
"error": str(e),
"model_type": model_type,
"model_name": model_name,
"timestamp": datetime.now().isoformat()
}

@with_tracing_span("calculate_performance_metrics")
@handle_errors(
exceptions=(Exception,),
default_return={},
context="calculate_performance_metrics"
)
async def _calculate_performance_metrics(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
self.logger.info("🔧 Calculating performance metrics...")

# Ensure arrays are numpy arrays
predictions, np.array(predictions)
actual_values, np.array(actual_values)

# Basic metrics
metrics = {
"total_samples": len(predictions),
"predictions_mean": float(np.mean(predictions)),
"predictions_std": float(np.std(predictions)),
"actual_mean": float(np.mean(actual_values)),
"actual_std": float(np.std(actual_values))
}

# Classification metrics (if applicable)
if len(np.unique(predictions)) <= 10:  # Likely classification
metrics.update(await self._calculate_classification_metrics(predictions, actual_values))

# Regression metrics (if applicable)
else:
    passpassmetrics.update(await self._calculate_regression_metrics(predictions, actual_values))

# Confidence metrics (if available)
if confidence_scores is not None:
    passmetrics.update(await self._calculate_confidence_metrics(confidence_scores, predictions, actual_values))

# Model drift detection
metrics.update(await self._detect_model_drift(predictions, actual_values))

self.logger.info("✅ Performance metrics calculated successfully")
return metrics

except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"❌ Error calculating performance metrics: {e}")
return {"error": str(e)}

@handle_errors(
exceptions=(Exception,),
default_return={},
context="calculate_classification_metrics"
)
async def _calculate_classification_metrics(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
from sklearn.metrics import (
accuracy_score, precision_score, recall_score, f1_score,
confusion_matrix, classification_report
)

# Calculate basic classification metrics
accuracy, accuracy_score(actual_values, predictions)
precision, precision_score(actual_values, predictions, average='weighted', zero_division = 0)
recall, recall_score(actual_values, predictions, average='weighted', zero_division = 0)
f1, f1_score(actual_values, predictions, average='weighted', zero_division = 0)

# Confusion matrix
cm, confusion_matrix(actual_values, predictions)

# Classification report
report, classification_report(actual_values, predictions, output_dict = True, zero_division = 0)

return {
"accuracy": float(accuracy),
"precision": float(precision),
"recall": float(recall),
"f1_score": float(f1),
"confusion_matrix": cm.tolist(),
"classification_report": report,
"model_type": "classification"
}

except Exception as e:
    passpasspasspasspasspasspassself.logger.warning(f"⚠️ Error calculating classification metrics: {e}")
return {
"accuracy": 0.0,
"precision": 0.0,
"recall": 0.0,
"f1_score": 0.0,
"error": str(e),
"model_type": "classification"
}

@handle_errors(
exceptions=(Exception,),
default_return={},
context="calculate_regression_metrics"
)
async def _calculate_regression_metrics(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Calculate regression metrics
mse, mean_squared_error(actual_values, predictions)
rmse, np.sqrt(mse)
mae, mean_absolute_error(actual_values, predictions)
r2, r2_score(actual_values, predictions)

# Calculate additional metrics
mape, np.mean(np.abs((actual_values - predictions) / actual_values)) * 100

return {
"mse": float(mse),
"rmse": float(rmse),
"mae": float(mae),
"r2_score": float(r2),
"mape": float(mape),
"model_type": "regression"
}

except Exception as e:
    passpasspasspasspasspasspassself.logger.warning(f"⚠️ Error calculating regression metrics: {e}")
return {
"mse": float('inf'),
"rmse": float('inf'),
"mae": float('inf'),
"r2_score": 0.0,
"mape": float('inf'),
"error": str(e),
"model_type": "regression"
}

@handle_errors(
exceptions=(Exception,),
default_return={},
context="calculate_confidence_metrics"
)
async def _calculate_confidence_metrics(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
# Calculate confidence statistics
confidence_mean, float(np.mean(confidence_scores))
confidence_std, float(np.std(confidence_scores))

# Calculate calibration metrics
correct_predictions = (predictions == actual_values).astype(float)
calibration_error, float(np.mean(np.abs(confidence_scores - correct_predictions)))

# Calculate reliability diagram data
confidence_bins, np.linspace(0, 1, 11)
bin_accuracies = []
bin_confidences = []

for i in range(len(confidence_bins) - 1):
    passmask = (confidence_scores >= confidence_bins[i]) & (confidence_scores < confidence_bins[i + 1])
if np.sum(mask) > 0:
    passbin_accuracy, np.mean(correct_predictions[mask])
bin_confidence, np.mean(confidence_scores[mask])
bin_accuracies.append(bin_accuracy)
bin_confidences.append(bin_confidence)

return {
"confidence_mean": confidence_mean,
"confidence_std": confidence_std,
"calibration_error": calibration_error,
"reliability_diagram": {
"accuracies": bin_accuracies,
"confidences": bin_confidences
}
}

except Exception as e:
    passpasspasspasspasspasspassself.logger.warning(f"⚠️ Error calculating confidence metrics: {e}")
return {
"confidence_mean": 0.0,
"confidence_std": 0.0,
"calibration_error": float('inf'),
"error": str(e)
}

@handle_errors(
exceptions=(Exception,),
default_return={},
context="detect_model_drift"
)
async def _detect_model_drift(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
from scipy import stats

# Calculate prediction errors
errors, predictions - actual_values

# Statistical tests for drift detection
drift_metrics = {
"error_mean": float(np.mean(errors)),
"error_std": float(np.std(errors)),
"error_skewness": float(stats.skew(errors)),
"error_kurtosis": float(stats.kurtosis(errors))
}

# Detect outliers using IQR method
q1, q3, np.percentile(errors, [25, 75])
iqr, q3 - q1
outlier_threshold, 1.5 * iqr
outliers, np.sum((errors < (q1 - outlier_threshold)) | (errors > (q3 + outlier_threshold)))

drift_metrics.update({
"outlier_count": int(outliers),
"outlier_percentage": float(outliers / len(errors) * 100),
"drift_detected": outliers / len(errors) > 0.05  # 5% threshold
})

return drift_metrics

except Exception as e:
    passpasspasspasspasspasspassself.logger.warning(f"⚠️ Error detecting model drift: {e}")
return {
"error_mean": 0.0,
"error_std": 0.0,
"outlier_count": 0,
"outlier_percentage": 0.0,
"drift_detected": False,
"error": str(e)
}

@with_tracing_span("store_performance_metrics")
@handle_errors(
exceptions=(Exception,),
default_return = False,
context="store_performance_metrics"
)
async def _store_performance_metrics(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
# Add to performance history
if model_type not in self.performance_history:
    passself.performance_history[model_type] = []

self.performance_history[model_type].append(metrics)

# Update current metrics
self.current_metrics[model_type] = metrics

# Save to file
await self._save_metrics_to_file(model_type, model_name, metrics)

return True

except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"❌ Error storing performance metrics: {e}")
return False

@handle_errors(
exceptions=(Exception,),
default_return = False,
context="save_metrics_to_file"
)
async def _save_metrics_to_file(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
# Create model - specific directory
model_dir, self.results_dir / model_type / model_name
model_dir.mkdir(parents = True, exist_ok = True)

# Save current metrics
timestamp, datetime.now().strftime("%Y%m%d_%H%M%S")
metrics_file, model_dir / f"metrics_{timestamp}.json"

with open(metrics_file, 'w') as f:
    passjson.dump(metrics, f, indent = 2, default = str)

# Save latest metrics
latest_file, model_dir / "latest_metrics.json"
with open(latest_file, 'w') as f:
    passjson.dump(metrics, f, indent = 2, default = str)

return True

except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"❌ Error saving metrics to file: {e}")
return False

@handle_errors(
exceptions=(Exception,),
default_return={"status": "UNKNOWN"},
context="check_performance_thresholds"
)
async def _check_performance_thresholds(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
status = {
"overall_status": "PASS",
"failed_checks": [],
"warnings": []
}

# Check classification metrics
if "accuracy" in metrics:
    passif metrics["accuracy"] < self.performance_thresholds["min_accuracy"]:
    passstatus["failed_checks"].append(f"Accuracy below threshold: {metrics['accuracy']:.3f} < {self.performance_thresholds['min_accuracy']}")
status["overall_status"] = "FAIL"
elif metrics["accuracy"] < self.performance_thresholds["min_accuracy"] + 0.1:
    passpassstatus["warnings"].append(f"Accuracy close to threshold: {metrics['accuracy']:.3f}")

if "precision" in metrics:
    passif metrics["precision"] < self.performance_thresholds["min_precision"]:
    passstatus["failed_checks"].append(f"Precision below threshold: {metrics['precision']:.3f} < {self.performance_thresholds['min_precision']}")
status["overall_status"] = "FAIL"

if "recall" in metrics:
    passif metrics["recall"] < self.performance_thresholds["min_recall"]:
    passstatus["failed_checks"].append(f"Recall below threshold: {metrics['recall']:.3f} < {self.performance_thresholds['min_recall']}")
status["overall_status"] = "FAIL"

if "f1_score" in metrics:
    passif metrics["f1_score"] < self.performance_thresholds["min_f1_score"]:
    passstatus["failed_checks"].append(f"F1 score below threshold: {metrics['f1_score']:.3f} < {self.performance_thresholds['min_f1_score']}")
status["overall_status"] = "FAIL"

# Check for model drift
if "drift_detected" in metrics and metrics["drift_detected"]:
    passpassstatus["warnings"].append("Model drift detected")

return status

except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"❌ Error checking performance thresholds: {e}")
return {"status": "ERROR", "error": str(e)}

@handle_errors(
exceptions=(Exception,),
default_return = False,
context="update_model_registry"
)
async def _update_model_registry(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
if model_type not in self.model_registry:
    passpassself.model_registry[model_type] = {}

# Update registry
self.model_registry[model_type].update({
"last_updated": datetime.now().isoformat(),
"total_runs": self.model_registry[model_type].get("total_runs", 0) + 1,
"last_performance": metrics,
"last_status": performance_status
})

# Update success / failure counts
if performance_status.get("overall_status") == "PASS":
    passself.model_registry[model_type]["successful_runs"] = self.model_registry[model_type].get("successful_runs", 0) + 1
else:
    passself.model_registry[model_type]["failed_runs"] = self.model_registry[model_type].get("failed_runs", 0) + 1

return True

except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"❌ Error updating model registry: {e}")
return False

@handle_errors(
exceptions=(Exception,),
default_return = None,
context="log_performance_summary"
)
async def _log_performance_summary(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
status_icon = "✅" if performance_status.get("overall_status") == "PASS" else "❌"
self.logger.info(f"{status_icon} Performance Summary for {model_type}:{model_name}")

# Log key metrics
if "accuracy" in metrics:
    passself.logger.info(f"   📊 Accuracy: {metrics['accuracy']:.3f}")
if "precision" in metrics:
    passself.logger.info(f"   📊 Precision: {metrics['precision']:.3f}")
if "recall" in metrics:
    passself.logger.info(f"   📊 Recall: {metrics['recall']:.3f}")
if "f1_score" in metrics:
    passself.logger.info(f"   📊 F1 Score: {metrics['f1_score']:.3f}")
if "r2_score" in metrics:
    passself.logger.info(f"   📊 R² Score: {metrics['r2_score']:.3f}")
if "rmse" in metrics:
    passself.logger.info(f"   📊 RMSE: {metrics['rmse']:.6f}")

# Log status
self.logger.info(f"   🎯 Status: {performance_status.get('overall_status', 'UNKNOWN')}")

# Log warnings and failures
if performance_status.get("warnings"):
    passfor warning in performance_status["warnings"]:
    passself.logger.warning(f"   ⚠️ {warning}")

if performance_status.get("failed_checks"):
    passfor failure in performance_status["failed_checks"]:
    passself.logger.error(f"   ❌ {failure}")

except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"❌ Error logging performance summary: {e}")

@with_tracing_span("generate_performance_report")
@handle_errors(
exceptions=(Exception,),
default_return={},
context="generate_performance_report"
)
async def generate_performance_report(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
self.logger.info("📊 Generating performance report...")

report = {
"generated_at": datetime.now().isoformat(),
"summary": {},
"detailed_metrics": {},
"recommendations": []
}

# Generate summary statistics
if model_type:
    passreport["summary"] = await self._generate_model_type_summary(model_type)
report["detailed_metrics"] = self.performance_history.get(model_type, [])
else:
    passfor mt in self.performance_history.keys():
    passreport["summary"][mt] = await self._generate_model_type_summary(mt)
report["detailed_metrics"][mt] = self.performance_history[mt]

# Generate recommendations
report["recommendations"] = await self._generate_recommendations(report["summary"])

# Save report
await self._save_performance_report(report)

self.logger.info("✅ Performance report generated successfully")
return report

except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"❌ Error generating performance report: {e}")
return {"error": str(e)}

@handle_errors(
exceptions=(Exception,),
default_return={},
context="generate_model_type_summary"
)
async def _generate_model_type_summary(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
history, self.performance_history.get(model_type, [])
if not history:
    passreturn {"error": "No performance history available"}

# Calculate summary statistics
summary = {
"total_runs": len(history),
"latest_run": history[-1] if history else None,
"average_metrics": {},
"trend_analysis": {}
}

# Calculate averages for key metrics
key_metrics = ["accuracy", "precision", "recall", "f1_score", "r2_score", "rmse"]
for metric in key_metrics:
    passvalues = [h.get(metric, 0) for h in history if h.get(metric) is not None]
if values:
    passpasssummary["average_metrics"][metric] = {
"mean": float(np.mean(values)),
"std": float(np.std(values)),
"min": float(np.min(values)),
"max": float(np.max(values))
}

return summary

except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"❌ Error generating model type summary: {e}")
return {"error": str(e)}

@handle_errors(
exceptions=(Exception,),
default_return=[],
context="generate_recommendations"
)
async def _generate_recommendations(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
recommendations = []

for model_type, model_summary in summary.items():
    passif "error" in model_summary:
    passcontinue

avg_metrics, model_summary.get("average_metrics", {})

# Check accuracy trends
if "accuracy" in avg_metrics:
    passaccuracy_mean, avg_metrics["accuracy"]["mean"]
if accuracy_mean < 0.7:
    passrecommendations.append(f"Consider retraining {model_type} model - low average accuracy ({accuracy_mean:.3f})")
elif accuracy_mean < 0.8:
    passpassrecommendations.append(f"Monitor {model_type} model performance - accuracy below optimal ({accuracy_mean:.3f})")

# Check for high variance
for metric, stats in avg_metrics.items():
    passif stats["std"] > 0.1:  # High variance
recommendations.append(f"High variance detected in {model_type} {metric} - consider model stabilization")

# Check for performance degradation
if "latest_run" in model_summary and model_summary["latest_run"]:
    passpasslatest, model_summary["latest_run"]
if "drift_detected" in latest and latest["drift_detected"]:
    passrecommendations.append(f"Model drift detected in {model_type} - consider retraining with recent data")

return recommendations

except Exception as e:
    passpasspasspasspasspasspasspassself.logger.exception(f"❌ Error generating recommendations: {e}")
return [f"Error generating recommendations: {str(e)}"]

@handle_errors(
exceptions=(Exception,),
default_return = False,
context="save_performance_report"
)
async def _save_performance_report(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
timestamp, datetime.now().strftime("%Y%m%d_%H%M%S")
report_file, self.results_dir / f"performance_report_{timestamp}.json"

with open(report_file, 'w') as f:
    passjson.dump(report, f, indent = 2, default = str)

# Save latest report
latest_file, self.results_dir / "latest_performance_report.json"
with open(latest_file, 'w') as f:
    passjson.dump(report, f, indent = 2, default = str)

self.logger.info(f"📄 Performance report saved to {report_file}")
return True

except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"❌ Error saving performance report: {e}")
return False

@handle_errors(
exceptions=(Exception,),
default_return={},
context="get_model_performance"
)
async def get_model_performance(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
result = {
"model_type": model_type,
"current_metrics": self.current_metrics.get(model_type, {}),
"registry_info": self.model_registry.get(model_type, {})
}

if include_history:
    passresult["performance_history"] = self.performance_history.get(model_type, [])

return result

except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"❌ Error getting model performance: {e}")
return {"error": str(e)}