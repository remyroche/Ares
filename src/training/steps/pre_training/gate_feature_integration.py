"""
Gate Feature Integration Module

This module provides comprehensive gate feature protection and integration functionality
for the pre-training pipeline. Gate features are special features that act as quality
gates and protection mechanisms in the machine learning pipeline.
"""

from __future__ import annotations

import logging
import json
import warnings
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from src.utils.logger import system_logger
from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success, 
    tprint_debug, tprint_data_preview, tprint_data_format, tprint_performance,
    tprint_structured, tprint_step, tprint_result
)


class GateFeatureType(Enum):
    """Types of gate features."""
    QUALITY_GATE = "quality_gate"
    STABILITY_GATE = "stability_gate"
    PERFORMANCE_GATE = "performance_gate"
    DATA_INTEGRITY_GATE = "data_integrity_gate"
    FEATURE_IMPORTANCE_GATE = "feature_importance_gate"
    CORRELATION_GATE = "correlation_gate"
    VARIANCE_GATE = "variance_gate"
    OUTLIER_GATE = "outlier_gate"


class GateStatus(Enum):
    """Status of gate feature evaluation."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class GateFeatureConfig:
    """Configuration for gate feature integration."""
    
    # Core settings
    enable_gate_protection: bool = True
    max_gate_features_per_base: int = 3
    min_gate_ic_improvement: float = 0.005
    min_gate_stability: float = 0.4
    
    # Quality thresholds
    max_nan_ratio: float = 0.3
    min_variance_threshold: float = 1e-8
    max_correlation_threshold: float = 0.95
    min_data_points: int = 100
    
    # Performance thresholds
    min_ic_threshold: float = 0.01
    max_ic_decay: float = 0.5
    min_sharpe_ratio: float = 0.5
    
    # Feature selection
    enable_feature_importance_gates: bool = True
    enable_correlation_gates: bool = True
    enable_variance_gates: bool = True
    enable_outlier_gates: bool = True
    
    # Monitoring
    enable_gate_monitoring: bool = True
    gate_monitoring_frequency: int = 100
    enable_gate_reporting: bool = True
    
    # Persistence
    enable_gate_persistence: bool = True
    gate_state_file: str = "gate_feature_state.json"


@dataclass
class GateFeatureResult:
    """Result of gate feature evaluation."""
    
    feature_name: str
    gate_type: GateFeatureType
    status: GateStatus
    score: float
    threshold: float
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class GateFeatureState:
    """State of gate feature integration."""
    
    enabled: bool = True
    active_gates: Dict[str, GateFeatureResult] = field(default_factory=dict)
    gate_history: List[GateFeatureResult] = field(default_factory=list)
    configuration: GateFeatureConfig = field(default_factory=GateFeatureConfig)
    last_updated: datetime = field(default_factory=datetime.now)


class GateFeatureValidator:
    """Validator for gate features."""
    
    def __init__(self, config: GateFeatureConfig):
        self.config = config
        self.logger = system_logger.getChild("GateFeatureValidator")
    
    def validate_quality_gate(self, features: pd.DataFrame, targets: pd.Series) -> GateFeatureResult:
        """Validate quality gate features."""
        feature_name = "quality_gate"
        
        # Check data size
        if len(features) < self.config.min_data_points:
            return GateFeatureResult(
                feature_name=feature_name,
                gate_type=GateFeatureType.QUALITY_GATE,
                status=GateStatus.FAILED,
                score=0.0,
                threshold=self.config.min_data_points,
                message=f"Insufficient data: {len(features)} < {self.config.min_data_points}"
            )
        
        # Check target variance
        target_variance = targets.var()
        if target_variance < self.config.min_variance_threshold:
            return GateFeatureResult(
                feature_name=feature_name,
                gate_type=GateFeatureType.QUALITY_GATE,
                status=GateStatus.FAILED,
                score=target_variance,
                threshold=self.config.min_variance_threshold,
                message=f"Target variance too low: {target_variance:.2e} < {self.config.min_variance_threshold:.2e}"
            )
        
        # Check NaN ratios
        nan_ratios = features.isnull().sum() / len(features)
        high_nan_features = nan_ratios > self.config.max_nan_ratio
        if high_nan_features.any():
            failed_count = high_nan_features.sum()
            return GateFeatureResult(
                feature_name=feature_name,
                gate_type=GateFeatureType.QUALITY_GATE,
                status=GateStatus.FAILED,
                score=1.0 - (failed_count / len(features.columns)),
                threshold=1.0 - self.config.max_nan_ratio,
                message=f"{failed_count} features have >{self.config.max_nan_ratio*100:.1f}% NaN values"
            )
        
        return GateFeatureResult(
            feature_name=feature_name,
            gate_type=GateFeatureType.QUALITY_GATE,
            status=GateStatus.PASSED,
            score=1.0,
            threshold=0.0,
            message="Quality gate passed"
        )
    
    def validate_correlation_gate(self, features: pd.DataFrame) -> GateFeatureResult:
        """Validate correlation gate features."""
        feature_name = "correlation_gate"
        
        if not self.config.enable_correlation_gates:
            return GateFeatureResult(
                feature_name=feature_name,
                gate_type=GateFeatureType.CORRELATION_GATE,
                status=GateStatus.SKIPPED,
                score=0.0,
                threshold=0.0,
                message="Correlation gates disabled"
            )
        
        # Calculate correlation matrix
        corr_matrix = features.corr().abs()
        
        # Find high correlations (excluding diagonal)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if corr_value > self.config.max_correlation_threshold:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_value))
        
        if high_corr_pairs:
            return GateFeatureResult(
                feature_name=feature_name,
                gate_type=GateFeatureType.CORRELATION_GATE,
                status=GateStatus.WARNING,
                score=len(high_corr_pairs),
                threshold=self.config.max_correlation_threshold,
                message=f"Found {len(high_corr_pairs)} highly correlated feature pairs"
            )
        
        return GateFeatureResult(
            feature_name=feature_name,
            gate_type=GateFeatureType.CORRELATION_GATE,
            status=GateStatus.PASSED,
            score=0.0,
            threshold=self.config.max_correlation_threshold,
            message="No high correlations found"
        )
    
    def validate_variance_gate(self, features: pd.DataFrame) -> GateFeatureResult:
        """Validate variance gate features."""
        feature_name = "variance_gate"
        
        if not self.config.enable_variance_gates:
            return GateFeatureResult(
                feature_name=feature_name,
                gate_type=GateFeatureType.VARIANCE_GATE,
                status=GateStatus.SKIPPED,
                score=0.0,
                threshold=0.0,
                message="Variance gates disabled"
            )
        
        # Calculate feature variances
        variances = features.var()
        low_variance_features = variances < self.config.min_variance_threshold
        
        if low_variance_features.any():
            low_var_count = low_variance_features.sum()
            return GateFeatureResult(
                feature_name=feature_name,
                gate_type=GateFeatureType.VARIANCE_GATE,
                status=GateStatus.WARNING,
                score=low_var_count,
                threshold=self.config.min_variance_threshold,
                message=f"{low_var_count} features have low variance"
            )
        
        return GateFeatureResult(
            feature_name=feature_name,
            gate_type=GateFeatureType.VARIANCE_GATE,
            status=GateStatus.PASSED,
            score=0.0,
            threshold=self.config.min_variance_threshold,
            message="All features have sufficient variance"
        )


class GateFeatureSelector:
    """Selector for gate features."""
    
    def __init__(self, config: GateFeatureConfig):
        self.config = config
        self.logger = system_logger.getChild("GateFeatureSelector")
    
    def select_gate_features(self, features: pd.DataFrame, targets: pd.Series) -> List[str]:
        """Select appropriate gate features based on configuration."""
        selected_features = []
        
        # Quality-based selection
        if self.config.enable_feature_importance_gates:
            # Select features with highest variance
            variances = features.var().sort_values(ascending=False)
            selected_features.extend(variances.head(self.config.max_gate_features_per_base).index.tolist())
        
        # Correlation-based selection
        if self.config.enable_correlation_gates:
            corr_matrix = features.corr().abs()
            # Select features with moderate correlations (not too high, not too low)
            avg_correlations = corr_matrix.mean().sort_values(ascending=False)
            selected_features.extend(avg_correlations.head(self.config.max_gate_features_per_base).index.tolist())
        
        # Remove duplicates and limit
        selected_features = list(set(selected_features))[:self.config.max_gate_features_per_base]
        
        return selected_features


class GateFeatureMonitor:
    """Monitor for gate features."""
    
    def __init__(self, config: GateFeatureConfig):
        self.config = config
        self.logger = system_logger.getChild("GateFeatureMonitor")
        self.monitoring_data = []
    
    def monitor_gate_performance(self, gate_results: List[GateFeatureResult]) -> Dict[str, Any]:
        """Monitor gate feature performance."""
        if not self.config.enable_gate_monitoring:
            return {}
        
        # Aggregate monitoring data
        monitoring_stats = {
            "total_gates": len(gate_results),
            "passed_gates": len([r for r in gate_results if r.status == GateStatus.PASSED]),
            "failed_gates": len([r for r in gate_results if r.status == GateStatus.FAILED]),
            "warning_gates": len([r for r in gate_results if r.status == GateStatus.WARNING]),
            "skipped_gates": len([r for r in gate_results if r.status == GateStatus.SKIPPED]),
            "average_score": np.mean([r.score for r in gate_results]),
            "timestamp": datetime.now()
        }
        
        # Store for trend analysis
        self.monitoring_data.append(monitoring_stats)
        
        # Keep only recent data
        if len(self.monitoring_data) > self.config.gate_monitoring_frequency:
            self.monitoring_data = self.monitoring_data[-self.config.gate_monitoring_frequency:]
        
        return monitoring_stats
    
    def generate_gate_report(self, gate_results: List[GateFeatureResult]) -> str:
        """Generate gate feature report."""
        if not self.config.enable_gate_reporting:
            return ""
        
        report_lines = [
            "=" * 60,
            "GATE FEATURE INTEGRATION REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # Summary statistics
        total_gates = len(gate_results)
        passed_gates = len([r for r in gate_results if r.status == GateStatus.PASSED])
        failed_gates = len([r for r in gate_results if r.status == GateStatus.FAILED])
        warning_gates = len([r for r in gate_results if r.status == GateStatus.WARNING])
        
        report_lines.extend([
            "SUMMARY:",
            f"  Total Gates: {total_gates}",
            f"  Passed: {passed_gates} ({passed_gates/total_gates*100:.1f}%)",
            f"  Failed: {failed_gates} ({failed_gates/total_gates*100:.1f}%)",
            f"  Warnings: {warning_gates} ({warning_gates/total_gates*100:.1f}%)",
            ""
        ])
        
        # Detailed results
        report_lines.append("DETAILED RESULTS:")
        for result in gate_results:
            status_symbol = {
                GateStatus.PASSED: "✅",
                GateStatus.FAILED: "❌",
                GateStatus.WARNING: "⚠️",
                GateStatus.SKIPPED: "⏭️"
            }.get(result.status, "❓")
            
            report_lines.append(
                f"  {status_symbol} {result.feature_name} ({result.gate_type.value}): "
                f"Score={result.score:.3f}, Threshold={result.threshold:.3f}"
            )
            if result.message:
                report_lines.append(f"    Message: {result.message}")
        
        return "\n".join(report_lines)


class GateFeaturePipelineManager:
    """
    Comprehensive manager for gate feature protection in the pipeline.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the gate feature pipeline manager.
        
        Args:
            config: Configuration dictionary
        """
        tprint_step("🔧 Initializing GateFeaturePipelineManager")
        tprint_debug(f"⚙️ Config provided: {config is not None}")
        
        self.config = GateFeatureConfig(**(config or {}))
        self.logger = system_logger.getChild("GateFeaturePipelineManager")
        self.state = GateFeatureState(configuration=self.config)
        
        tprint_info(f"🛡️ Gate protection enabled: {self.config.enable_gate_protection}")
        tprint_info(f"📊 Max gate features per base: {self.config.max_gate_features_per_base}")
        tprint_info(f"🎯 Min gate IC improvement: {self.config.min_gate_ic_improvement}")
        tprint_info(f"🔒 Min gate stability: {self.config.min_gate_stability}")
        
        # Initialize components
        tprint_debug("🔧 Initializing gate feature components")
        self.validator = GateFeatureValidator(self.config)
        self.selector = GateFeatureSelector(self.config)
        self.monitor = GateFeatureMonitor(self.config)
        tprint_success("✅ Gate feature components initialized")
        
        # Load persisted state if available
        if self.config.enable_gate_persistence:
            tprint_debug("💾 Loading persisted gate state")
            self._load_gate_state()
            tprint_success("✅ Gate state loaded from persistence")
        
        tprint_success("🎉 GateFeaturePipelineManager initialization complete")
    
    def enable_gate_protection(self) -> None:
        """Enable gate feature protection."""
        self.state.enabled = True
        self.logger.info("Gate feature protection enabled")
        tprint_success("🛡️ Gate feature protection enabled")
    
    def disable_gate_protection(self) -> None:
        """Disable gate feature protection."""
        self.state.enabled = False
        self.logger.info("Gate feature protection disabled")
        tprint_warning("🛡️ Gate feature protection disabled")
    
    def is_gate_protection_enabled(self) -> bool:
        """Check if gate protection is enabled."""
        return self.state.enabled
    
    def evaluate_gate_features(self, features: pd.DataFrame, targets: pd.Series) -> List[GateFeatureResult]:
        """Evaluate all gate features."""
        tprint_step("🔍 Starting gate feature evaluation")
        tprint_data_preview(features, "gate_evaluation_input_features", level="DEBUG")
        tprint_data_preview(targets, "gate_evaluation_input_targets", level="DEBUG")
        tprint_info(f"📊 Features shape: {features.shape}, Targets length: {len(targets)}")
        
        if not self.state.enabled:
            tprint_warning("⚠️ Gate protection disabled - skipping evaluation")
            return []
        
        tprint_info("🔍 Evaluating gate features...")
        tprint_debug(f"🛡️ Gate protection enabled: {self.state.enabled}")
        tprint_debug(f"📋 Active gates count: {len(self.state.active_gates)}")
        
        gate_results = []
        
        # Quality gate
        tprint_debug("🔍 Evaluating quality gate")
        quality_result = self.validator.validate_quality_gate(features, targets)
        gate_results.append(quality_result)
        tprint_debug(f"✅ Quality gate result: {quality_result.status.value} (score: {quality_result.score:.3f})")
        
        # Correlation gate
        tprint_debug("🔍 Evaluating correlation gate")
        correlation_result = self.validator.validate_correlation_gate(features)
        gate_results.append(correlation_result)
        tprint_debug(f"✅ Correlation gate result: {correlation_result.status.value} (score: {correlation_result.score:.3f})")
        
        # Variance gate
        tprint_debug("🔍 Evaluating variance gate")
        variance_result = self.validator.validate_variance_gate(features)
        gate_results.append(variance_result)
        tprint_debug(f"✅ Variance gate result: {variance_result.status.value} (score: {variance_result.score:.3f})")
        
        # Update state
        tprint_debug("💾 Updating gate state")
        for result in gate_results:
            self.state.active_gates[result.feature_name] = result
            self.state.gate_history.append(result)
        tprint_success(f"✅ Updated state with {len(gate_results)} gate results")
        
        # Monitor performance
        if self.config.enable_gate_monitoring:
            tprint_debug("📊 Monitoring gate performance")
            monitoring_stats = self.monitor.monitor_gate_performance(gate_results)
            self.logger.info(f"Gate monitoring stats: {monitoring_stats}")
            tprint_structured(monitoring_stats, "gate_monitoring_stats", level="DEBUG")
        
        # Generate report
        if self.config.enable_gate_reporting:
            tprint_debug("📋 Generating gate report")
            report = self.monitor.generate_gate_report(gate_results)
            if report:
                tprint_info(f"\n{report}")
            else:
                tprint_warning("⚠️ No gate report generated")
        
        # Persist state
        if self.config.enable_gate_persistence:
            tprint_debug("💾 Persisting gate state")
            self._save_gate_state()
            tprint_success("✅ Gate state persisted")
        
        tprint_result(f"🎯 Gate evaluation complete: {len(gate_results)} gates evaluated")
        return gate_results
    
    def select_gate_features(self, features: pd.DataFrame, targets: pd.Series) -> List[str]:
        """Select gate features for the pipeline."""
        tprint_step("🎯 Starting gate feature selection")
        tprint_data_preview(features, "gate_selection_input_features", level="DEBUG")
        tprint_data_preview(targets, "gate_selection_input_targets", level="DEBUG")
        tprint_info(f"📊 Features shape: {features.shape}, Targets length: {len(targets)}")
        
        if not self.state.enabled:
            tprint_warning("⚠️ Gate protection disabled - no features selected")
            return []
        
        tprint_debug("🔍 Selecting gate features using selector")
        selected_features = self.selector.select_gate_features(features, targets)
        tprint_info(f"🎯 Selected {len(selected_features)} gate features: {selected_features}")
        tprint_result(f"✅ Gate feature selection complete: {len(selected_features)} features selected")
        
        return selected_features
    
    def get_gate_status(self) -> Dict[str, Any]:
        """Get current gate status."""
        return {
            "enabled": self.state.enabled,
            "active_gates": len(self.state.active_gates),
            "total_evaluations": len(self.state.gate_history),
            "last_updated": self.state.last_updated,
            "configuration": {
                "max_gate_features_per_base": self.config.max_gate_features_per_base,
                "min_gate_ic_improvement": self.config.min_gate_ic_improvement,
                "min_gate_stability": self.config.min_gate_stability
            }
        }
    
    def _save_gate_state(self) -> None:
        """Save gate state to file."""
        try:
            state_data = {
                "enabled": self.state.enabled,
                "active_gates": {
                    name: {
                        "feature_name": result.feature_name,
                        "gate_type": result.gate_type.value,
                        "status": result.status.value,
                        "score": result.score,
                        "threshold": result.threshold,
                        "message": result.message,
                        "metadata": result.metadata,
                        "timestamp": result.timestamp.isoformat()
                    }
                    for name, result in self.state.active_gates.items()
                },
                "configuration": {
                    "enable_gate_protection": self.config.enable_gate_protection,
                    "max_gate_features_per_base": self.config.max_gate_features_per_base,
                    "min_gate_ic_improvement": self.config.min_gate_ic_improvement,
                    "min_gate_stability": self.config.min_gate_stability
                },
                "last_updated": self.state.last_updated.isoformat()
            }
            
            state_file = Path(self.config.gate_state_file)
            state_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
                
            self.logger.info(f"Gate state saved to {state_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save gate state: {e}")
    
    def _load_gate_state(self) -> None:
        """Load gate state from file."""
        try:
            state_file = Path(self.config.gate_state_file)
            if not state_file.exists():
                return
            
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            
            self.state.enabled = state_data.get("enabled", True)
            self.state.last_updated = datetime.fromisoformat(state_data.get("last_updated", datetime.now().isoformat()))
            
            # Load active gates
            active_gates_data = state_data.get("active_gates", {})
            for name, gate_data in active_gates_data.items():
                result = GateFeatureResult(
                    feature_name=gate_data["feature_name"],
                    gate_type=GateFeatureType(gate_data["gate_type"]),
                    status=GateStatus(gate_data["status"]),
                    score=gate_data["score"],
                    threshold=gate_data["threshold"],
                    message=gate_data["message"],
                    metadata=gate_data.get("metadata", {}),
                    timestamp=datetime.fromisoformat(gate_data["timestamp"])
                )
                self.state.active_gates[name] = result
            
            self.logger.info(f"Gate state loaded from {state_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to load gate state: {e}")


def enable_gate_protection() -> None:
    """
    Enable gate feature protection globally.
    """
    global_manager.enable_gate_protection()


def disable_gate_protection() -> None:
    """
    Disable gate feature protection globally.
    """
    global_manager.disable_gate_protection()


def get_gate_manager() -> GateFeaturePipelineManager:
    """
    Get the global gate feature manager.
    """
    return global_manager


def create_gate_manager(config: Optional[Dict[str, Any]] = None) -> GateFeaturePipelineManager:
    """
    Create a new gate feature manager with custom configuration.
    """
    return GateFeaturePipelineManager(config)


# Global manager instance
global_manager = GateFeaturePipelineManager()
