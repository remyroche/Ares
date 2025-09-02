#!/usr/bin/env python3
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
    """Comprehensive model performance monitoring system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the ModelPerformanceMonitor.
        
        Args:
            config: Configuration dictionary for the monitor
        """
        self.config = config or {}
        self.logger = system_logger.getChild("ModelPerformanceMonitor")
        
        # Performance tracking
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self.current_metrics: Dict[str, Dict[str, Any]] = {}
        self.model_registry: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.monitor_config = self.config.get("model_performance_monitor", {})
        self.enable_real_time_monitoring = self.monitor_config.get("enable_real_time_monitoring", True)
        self.performance_thresholds = self.monitor_config.get("performance_thresholds", {
            "min_accuracy": 0.6,
            "min_precision": 0.5,
            "min_recall": 0.5,
            "min_f1_score": 0.5,
            "max_drift": 0.1
        })
        
        # Storage
        self.results_dir = Path(self.monitor_config.get("results_dir", "results/model_performance"))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize performance tracking
        self._initialize_performance_tracking()

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="model_performance_monitor_initialization"
    )
    def _initialize_performance_tracking(self) -> bool:
        """Initialize model performance tracking."""
        try:
            self.logger.info("🔧 Initializing model performance tracking...")
            
            # Initialize performance history for each model type
            model_types = [
                "hmm_regime_discovery",
                "analyst_enhancement", 
                "tactician_specialist",
                "confidence_calibration",
                "unified_regime_intelligence"
            ]
            
            for model_type in model_types:
                self.performance_history[model_type] = []
                self.current_metrics[model_type] = {}
                self.model_registry[model_type] = {}
            
            self.logger.info("✅ Model performance tracking initialized successfully")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Error initializing performance tracking: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="track_model_performance"
    )
    async def track_model_performance(
        self,
        model_id: str,
        model_type: str,
        metrics: Dict[str, Any],
        step_name: str,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """Track model performance metrics for a specific step.
        
        Args:
            model_id: Unique identifier for the model
            model_type: Type of model (e.g., 'hmm_regime_discovery')
            metrics: Dictionary of performance metrics
            step_name: Name of the training/validation step
            timestamp: Timestamp for the metrics (defaults to current time)
            
        Returns:
            bool: True if tracking was successful, False otherwise
        """
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            # Validate model type
            if model_type not in self.performance_history:
                self.logger.warning(f"Unknown model type: {model_type}")
                return False
            
            # Create performance record
            performance_record = {
                "model_id": model_id,
                "step_name": step_name,
                "timestamp": timestamp.isoformat(),
                "metrics": metrics.copy(),
                "thresholds_met": self._check_thresholds(metrics)
            }
            
            # Store in performance history
            self.performance_history[model_type].append(performance_record)
            
            # Update current metrics
            self.current_metrics[model_type] = metrics.copy()
            
            # Update model registry
            if model_id not in self.model_registry[model_type]:
                self.model_registry[model_type][model_id] = {
                    "first_seen": timestamp.isoformat(),
                    "last_updated": timestamp.isoformat(),
                    "total_steps": 0,
                    "best_performance": None
                }
            
            self.model_registry[model_type][model_id]["last_updated"] = timestamp.isoformat()
            self.model_registry[model_type][model_id]["total_steps"] += 1
            
            # Update best performance if current is better
            current_score = self._calculate_performance_score(metrics)
            if (self.model_registry[model_type][model_id]["best_performance"] is None or
                current_score > self.model_registry[model_type][model_id]["best_performance"]["score"]):
                self.model_registry[model_type][model_id]["best_performance"] = {
                    "score": current_score,
                    "metrics": metrics.copy(),
                    "step_name": step_name,
                    "timestamp": timestamp.isoformat()
                }
            
            # Save to disk
            await self._save_performance_data()
            
            self.logger.info(f"✅ Performance tracked for {model_type}/{model_id} at step {step_name}")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Error tracking model performance: {e}")
            return False

    def _check_thresholds(self, metrics: Dict[str, Any]) -> Dict[str, bool]:
        """Check if metrics meet performance thresholds.
        
        Args:
            metrics: Dictionary of performance metrics
            
        Returns:
            Dictionary mapping threshold names to boolean values
        """
        threshold_results = {}
        
        for threshold_name, threshold_value in self.performance_thresholds.items():
            if threshold_name.startswith("min_"):
                metric_name = threshold_name[4:]  # Remove "min_" prefix
                if metric_name in metrics:
                    threshold_results[threshold_name] = metrics[metric_name] >= threshold_value
            elif threshold_name.startswith("max_"):
                metric_name = threshold_name[4:]  # Remove "max_" prefix
                if metric_name in metrics:
                    threshold_results[threshold_name] = metrics[metric_name] <= threshold_value
        
        return threshold_results

    def _calculate_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate a composite performance score from metrics.
        
        Args:
            metrics: Dictionary of performance metrics
            
        Returns:
            Float representing overall performance score
        """
        score = 0.0
        weight_sum = 0.0
        
        # Define weights for different metrics
        metric_weights = {
            "accuracy": 0.3,
            "precision": 0.25,
            "recall": 0.25,
            "f1_score": 0.2
        }
        
        for metric_name, weight in metric_weights.items():
            if metric_name in metrics:
                score += metrics[metric_name] * weight
                weight_sum += weight
        
        # Normalize by total weight if any metrics were found
        if weight_sum > 0:
            score = score / weight_sum
        
        return score

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="save_performance_data"
    )
    async def _save_performance_data(self) -> bool:
        """Save performance data to disk."""
        try:
            # Save performance history
            for model_type, history in self.performance_history.items():
                history_file = self.results_dir / f"{model_type}_performance_history.json"
                with open(history_file, 'w') as f:
                    json.dump(history, f, indent=2, default=str)
            
            # Save current metrics
            current_metrics_file = self.results_dir / "current_metrics.json"
            with open(current_metrics_file, 'w') as f:
                json.dump(self.current_metrics, f, indent=2, default=str)
            
            # Save model registry
            registry_file = self.results_dir / "model_registry.json"
            with open(registry_file, 'w') as f:
                json.dump(self.model_registry, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Error saving performance data: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="get_performance_summary"
    )
    def get_performance_summary(self, model_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get a summary of performance across all models or a specific model type.
        
        Args:
            model_type: Optional model type to filter by
            
        Returns:
            Dictionary containing performance summary
        """
        try:
            if model_type is not None:
                if model_type not in self.performance_history:
                    self.logger.warning(f"Unknown model type: {model_type}")
                    return None
                
                return self._generate_summary_for_type(model_type)
            
            # Generate summary for all model types
            summary = {}
            for mt in self.performance_history.keys():
                summary[mt] = self._generate_summary_for_type(mt)
            
            return summary
            
        except Exception as e:
            self.logger.exception(f"❌ Error generating performance summary: {e}")
            return None

    def _generate_summary_for_type(self, model_type: str) -> Dict[str, Any]:
        """Generate performance summary for a specific model type."""
        history = self.performance_history[model_type]
        
        if not history:
            return {
                "model_type": model_type,
                "total_models": 0,
                "total_steps": 0,
                "average_performance": {},
                "best_performance": None
            }
        
        # Calculate averages
        total_steps = len(history)
        total_models = len(set(record["model_id"] for record in history))
        
        # Aggregate metrics
        metric_sums = {}
        metric_counts = {}
        
        for record in history:
            for metric_name, metric_value in record["metrics"].items():
                if isinstance(metric_value, (int, float)):
                    if metric_name not in metric_sums:
                        metric_sums[metric_name] = 0.0
                        metric_counts[metric_name] = 0
                    metric_sums[metric_name] += metric_value
                    metric_counts[metric_name] += 1
        
        # Calculate averages
        average_performance = {}
        for metric_name in metric_sums:
            if metric_counts[metric_name] > 0:
                average_performance[metric_name] = metric_sums[metric_name] / metric_counts[metric_name]
        
        # Find best performance
        best_record = max(history, key=lambda x: self._calculate_performance_score(x["metrics"]))
        
        return {
            "model_type": model_type,
            "total_models": total_models,
            "total_steps": total_steps,
            "average_performance": average_performance,
            "best_performance": {
                "model_id": best_record["model_id"],
                "step_name": best_record["step_name"],
                "score": self._calculate_performance_score(best_record["metrics"]),
                "metrics": best_record["metrics"]
            }
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="detect_performance_drift"
    )
    def detect_performance_drift(
        self,
        model_type: str,
        current_metrics: Dict[str, Any],
        window_size: int = 10
    ) -> Optional[Dict[str, Any]]:
        """Detect performance drift by comparing current metrics to historical performance.
        
        Args:
            model_type: Type of model to analyze
            current_metrics: Current performance metrics
            window_size: Number of recent records to consider for baseline
            
        Returns:
            Dictionary containing drift analysis or None if insufficient data
        """
        try:
            if model_type not in self.performance_history:
                self.logger.warning(f"Unknown model type: {model_type}")
                return None
            
            history = self.performance_history[model_type]
            if len(history) < window_size:
                self.logger.info(f"Insufficient history for drift detection: {len(history)} < {window_size}")
                return None
            
            # Get recent baseline
            recent_history = history[-window_size:]
            
            # Calculate baseline statistics
            baseline_stats = {}
            for record in recent_history:
                for metric_name, metric_value in record["metrics"].items():
                    if isinstance(metric_value, (int, float)):
                        if metric_name not in baseline_stats:
                            baseline_stats[metric_name] = []
                        baseline_stats[metric_name].append(metric_value)
            
            # Calculate drift
            drift_analysis = {}
            for metric_name, values in baseline_stats.items():
                if len(values) > 0:
                    baseline_mean = np.mean(values)
                    baseline_std = np.std(values)
                    
                    if metric_name in current_metrics:
                        current_value = current_metrics[metric_name]
                        drift = current_value - baseline_mean
                        drift_normalized = drift / baseline_std if baseline_std > 0 else 0
                        
                        drift_analysis[metric_name] = {
                            "baseline_mean": baseline_mean,
                            "baseline_std": baseline_std,
                            "current_value": current_value,
                            "drift": drift,
                            "drift_normalized": drift_normalized,
                            "significant_drift": abs(drift_normalized) > 2.0  # 2 standard deviations
                        }
            
            return {
                "model_type": model_type,
                "window_size": window_size,
                "baseline_period": {
                    "start": recent_history[0]["timestamp"],
                    "end": recent_history[-1]["timestamp"]
                },
                "drift_analysis": drift_analysis
            }
            
        except Exception as e:
            self.logger.exception(f"❌ Error detecting performance drift: {e}")
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="export_performance_report"
    )
    async def export_performance_report(
        self,
        output_path: Optional[Path] = None,
        format: str = "json"
    ) -> bool:
        """Export a comprehensive performance report.
        
        Args:
            output_path: Path to save the report (defaults to results directory)
            format: Report format ('json' or 'csv')
            
        Returns:
            bool: True if export was successful, False otherwise
        """
        try:
            if output_path is None:
                output_path = self.results_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Generate comprehensive report
            report = {
                "generated_at": datetime.now().isoformat(),
                "summary": self.get_performance_summary(),
                "model_registry": self.model_registry,
                "performance_thresholds": self.performance_thresholds,
                "configuration": self.monitor_config
            }
            
            if format.lower() == "json":
                output_path = output_path.with_suffix(".json")
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
            elif format.lower() == "csv":
                # Convert to CSV format for specific metrics
                output_path = output_path.with_suffix(".csv")
                self._export_csv_report(report, output_path)
            else:
                self.logger.warning(f"Unsupported format: {format}")
                return False
            
            self.logger.info(f"✅ Performance report exported to {output_path}")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Error exporting performance report: {e}")
            return False

    def _export_csv_report(self, report: Dict[str, Any], output_path: Path) -> None:
        """Export performance report in CSV format."""
        # This is a simplified CSV export - in practice, you might want more sophisticated formatting
        import csv
        
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(["Model Type", "Total Models", "Total Steps", "Best Accuracy", "Best F1 Score"])
            
            # Write data
            for model_type, summary in report["summary"].items():
                best_metrics = summary.get("best_performance", {}).get("metrics", {})
                writer.writerow([
                    model_type,
                    summary.get("total_models", 0),
                    summary.get("total_steps", 0),
                    best_metrics.get("accuracy", "N/A"),
                    best_metrics.get("f1_score", "N/A")
                ])

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="cleanup_old_records"
    )
    async def cleanup_old_records(
        self,
        max_age_days: int = 90,
        max_records_per_model: int = 1000
    ) -> bool:
        """Clean up old performance records to manage storage.
        
        Args:
            max_age_days: Maximum age of records to keep
            max_records_per_model: Maximum number of records to keep per model type
            
        Returns:
            bool: True if cleanup was successful, False otherwise
        """
        try:
            cutoff_date = datetime.now() - pd.Timedelta(days=max_age_days)
            cleaned_count = 0
            
            for model_type in self.performance_history:
                original_count = len(self.performance_history[model_type])
                
                # Filter by age
                self.performance_history[model_type] = [
                    record for record in self.performance_history[model_type]
                    if datetime.fromisoformat(record["timestamp"]) > cutoff_date
                ]
                
                # Limit records per model type
                if len(self.performance_history[model_type]) > max_records_per_model:
                    self.performance_history[model_type] = self.performance_history[model_type][-max_records_per_model:]
                
                cleaned_count += original_count - len(self.performance_history[model_type])
            
            if cleaned_count > 0:
                await self._save_performance_data()
                self.logger.info(f"✅ Cleaned up {cleaned_count} old performance records")
            
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Error cleaning up old records: {e}")
            return False

    async def __aenter__(self):
        """Async context manager entry."""
        # The original code had an initialize method here, but it's now in __init__.
        # Keeping the structure as per the new_code, but noting the change.
        # await self.initialize() # This line is removed as initialize is now in __init__
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._save_performance_data()
        return False


# Factory function for easy instantiation
def create_model_performance_monitor(config: Optional[Dict[str, Any]] = None) -> ModelPerformanceMonitor:
    """Create a new ModelPerformanceMonitor instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured ModelPerformanceMonitor instance
    """
    return ModelPerformanceMonitor(config)


# Example usage
if __name__ == "__main__":
    async def main():
        # Example configuration
        config = {
            "model_performance_monitor": {
                "enable_real_time_monitoring": True,
                "results_dir": "results/model_performance",
                "performance_thresholds": {
                    "min_accuracy": 0.7,
                    "min_precision": 0.6,
                    "min_recall": 0.6,
                    "min_f1_score": 0.6,
                    "max_drift": 0.15
                }
            }
        }
        
        # Create monitor
        monitor = ModelPerformanceMonitor(config)
        
        # Example metrics
        example_metrics = {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1_score": 0.85
        }
        
        # Track performance
        success = await monitor.track_model_performance(
            model_id="test_model_001",
            model_type="hmm_regime_discovery",
            metrics=example_metrics,
            step_name="validation_step_1"
        )
        
        if success:
            print("✅ Performance tracked successfully")
            
            # Get summary
            summary = monitor.get_performance_summary()
            print(f"Performance summary: {summary}")
        else:
            print("❌ Failed to track performance")
    
    # Run example
    asyncio.run(main())