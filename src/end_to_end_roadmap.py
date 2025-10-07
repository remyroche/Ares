"""
End-to-End Roadmap System Integration

Main integration file that orchestrates the complete end-to-end roadmap system,
replacing the PID-driven generation feature with the new comprehensive approach.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import logging
from pathlib import Path

# Import all our modules
from .feature_engineering.data_contracts import InputBar, FeatureStore, ArtifactsRegistry
from .feature_engineering.feature_registry import FeatureRegistry
from .feature_engineering.transforms import TransformRouter, create_default_transform_config
from .feature_engineering.lookback_selection import LookbackSelector, create_feature_families
from .feature_engineering.interactions import InteractionEngine, create_default_interaction_config
from .feature_engineering.assembly_dag import AssemblyDAG, AssemblyConfig, run_assembly
from .models.patch_gru import PatchOrchestrator, PatchConfig, ModelType
from .validation.walkforward_validation import run_complete_validation, ValidationConfig
from .monitoring.retrain_monitoring import MonitoringSystem, MonitoringConfig
from .ci.validators import run_ci_validation
from .deployment.rollout_plan import RolloutOrchestrator, RolloutConfig, run_rollout


class SystemStatus(Enum):
    """Overall system status."""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    MONITORING = "monitoring"
    RETRAINING = "retraining"
    FAILED = "failed"


@dataclass
class SystemConfig:
    """Complete system configuration."""
    # Feature budgets
    feature_budget_pre: int = 120
    feature_budget_post: Tuple[int, int] = (30, 60)
    interactions_cap: int = 15
    transforms_per_parent: int = 1
    
    # Latency budgets
    latency_budget_ms: int = 50
    feature_compute_ms: int = 25
    model_inference_ms: int = 5
    io_orchestration_ms: int = 20
    
    # Lookback ceiling
    lookback_ceiling_minutes: int = 120
    
    # Retrain settings
    retrain_scheduled: str = "02:00 America/New_York"
    retrain_triggered_interval: str = "2h"
    fallback_p99_ms: float = 2.0
    
    # Model settings
    patch_model_type: ModelType = ModelType.GRU
    patch_sequence_length: int = 24
    patch_horizons: List[int] = None
    
    # Validation settings
    validation_n_folds: int = 6
    validation_embargo_pct: float = 0.1
    
    # Monitoring settings
    monitoring_interval_minutes: int = 5
    calibration_loss_threshold: float = 2.0
    psi_threshold: float = 0.3
    correlation_drift_threshold: float = 0.5
    
    def __post_init__(self):
        if self.patch_horizons is None:
            self.patch_horizons = [1, 3]


@dataclass
class SystemResult:
    """Result of system execution."""
    success: bool
    features: pd.DataFrame
    selected_features: List[str]
    patch_features: Dict[str, pd.Series]
    validation_results: Optional[Dict[str, Any]] = None
    monitoring_metrics: Optional[Dict[str, Any]] = None
    deployment_status: Optional[Dict[str, Any]] = None
    artifacts: Optional[ArtifactsRegistry] = None
    metadata: Dict[str, Any] = None
    error_message: Optional[str] = None


class EndToEndRoadmapSystem:
    """Main system orchestrator for end-to-end roadmap."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.status = SystemStatus.INITIALIZING
        
        # Initialize components
        self.assembly_dag = None
        self.monitoring_system = None
        self.rollout_orchestrator = None
        self.validation_config = None
        
        self._initialize_components()
        self.status = SystemStatus.READY
    
    def _initialize_components(self):
        """Initialize all system components."""
        try:
            # Assembly DAG
            assembly_config = AssemblyConfig(
                feature_budget_pre=self.config.feature_budget_pre,
                feature_budget_post=self.config.feature_budget_post,
                interactions_cap=self.config.interactions_cap,
                transforms_per_parent=self.config.transforms_per_parent,
                lookback_ceiling_minutes=self.config.lookback_ceiling_minutes,
                latency_budget_ms=self.config.latency_budget_ms,
                patch_model_type=self.config.patch_model_type,
                patch_sequence_length=self.config.patch_sequence_length,
                patch_horizons=self.config.patch_horizons
            )
            self.assembly_dag = AssemblyDAG(assembly_config)
            
            # Monitoring system
            monitoring_config = MonitoringConfig(
                calibration_loss_threshold=self.config.calibration_loss_threshold,
                psi_threshold=self.config.psi_threshold,
                correlation_drift_threshold=self.config.correlation_drift_threshold,
                latency_p99_threshold=self.config.latency_budget_ms,
                monitoring_interval_minutes=self.config.monitoring_interval_minutes
            )
            self.monitoring_system = MonitoringSystem(monitoring_config)
            
            # Rollout orchestrator
            rollout_config = RolloutConfig()
            self.rollout_orchestrator = RolloutOrchestrator(rollout_config)
            
            # Validation config
            self.validation_config = ValidationConfig(
                n_outer_folds=self.config.validation_n_folds,
                embargo_pct=self.config.validation_embargo_pct
            )
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            self.status = SystemStatus.FAILED
            raise
    
    def process_market_data(self, 
                           bars: pd.DataFrame,
                           targets: Optional[Dict[int, pd.Series]] = None,
                           enable_validation: bool = True,
                           enable_monitoring: bool = True,
                           enable_deployment: bool = False) -> SystemResult:
        """Process market data through the complete pipeline."""
        
        try:
            self.status = SystemStatus.RUNNING
            self.logger.info("Starting end-to-end roadmap processing")
            
            # Step 1: Assembly - Generate features
            self.logger.info("Step 1: Assembling features")
            assembly_result = self.assembly_dag.assemble(bars, targets)
            
            if assembly_result.status.value == 'failed':
                return SystemResult(
                    success=False,
                    features=pd.DataFrame(),
                    selected_features=[],
                    patch_features={},
                    error_message="Feature assembly failed"
                )
            
            # Step 2: Validation (if enabled)
            validation_results = None
            if enable_validation and targets:
                self.logger.info("Step 2: Running validation")
                try:
                    validation_results = run_complete_validation(
                        assembly_result.features,
                        targets.get(1, pd.Series(0, index=assembly_result.features.index)),
                        {'default': {}},
                        self.validation_config
                    )
                except Exception as e:
                    self.logger.warning(f"Validation failed: {e}")
                    validation_results = {'error': str(e)}
            
            # Step 3: Monitoring (if enabled)
            monitoring_metrics = None
            if enable_monitoring:
                self.logger.info("Step 3: Updating monitoring metrics")
                try:
                    predictions = assembly_result.patch_features.get('y_hat_h1', pd.Series(0, index=assembly_result.features.index))
                    actual = targets.get(1, pd.Series(0, index=assembly_result.features.index)) if targets else None
                    
                    if actual is not None:
                        monitoring_metrics = self.monitoring_system.update_metrics(
                            assembly_result.features,
                            predictions.values if hasattr(predictions, 'values') else predictions,
                            actual.values if hasattr(actual, 'values') else actual
                        )
                except Exception as e:
                    self.logger.warning(f"Monitoring failed: {e}")
                    monitoring_metrics = {'error': str(e)}
            
            # Step 4: Deployment (if enabled)
            deployment_status = None
            if enable_deployment and targets:
                self.logger.info("Step 4: Running deployment rollout")
                try:
                    predictions = assembly_result.patch_features.get('y_hat_h1', pd.Series(0, index=assembly_result.features.index))
                    actual = targets.get(1, pd.Series(0, index=assembly_result.features.index))
                    
                    deployment_status = self.rollout_orchestrator.execute_rollout(
                        assembly_result.features,
                        predictions.values if hasattr(predictions, 'values') else predictions,
                        actual.values if hasattr(actual, 'values') else actual
                    )
                except Exception as e:
                    self.logger.warning(f"Deployment failed: {e}")
                    deployment_status = {'error': str(e)}
            
            # Step 5: CI/CD Validation
            self.logger.info("Step 5: Running CI/CD validation")
            try:
                ci_results = run_ci_validation(assembly_result.features)
                
                # Check if build should fail
                critical_failures = [name for name, result in ci_results.items() 
                                   if result.status.value == 'fail' and name in ['feature_budgets', 'transform_types']]
                
                if critical_failures:
                    self.logger.error(f"CI/CD validation failed: {critical_failures}")
                    return SystemResult(
                        success=False,
                        features=assembly_result.features,
                        selected_features=assembly_result.selected_features,
                        patch_features=assembly_result.patch_features,
                        validation_results=validation_results,
                        monitoring_metrics=monitoring_metrics,
                        deployment_status=deployment_status,
                        artifacts=assembly_result.artifacts,
                        error_message=f"CI/CD validation failed: {critical_failures}"
                    )
            except Exception as e:
                self.logger.warning(f"CI/CD validation failed: {e}")
            
            self.status = SystemStatus.READY
            
            return SystemResult(
                success=True,
                features=assembly_result.features,
                selected_features=assembly_result.selected_features,
                patch_features=assembly_result.patch_features,
                validation_results=validation_results,
                monitoring_metrics=monitoring_metrics,
                deployment_status=deployment_status,
                artifacts=assembly_result.artifacts,
                metadata={
                    'total_features': len(assembly_result.features.columns),
                    'selected_features': len(assembly_result.selected_features),
                    'patch_features': len(assembly_result.patch_features),
                    'processing_time': datetime.now().isoformat(),
                    'system_status': self.status.value
                }
            )
            
        except Exception as e:
            self.status = SystemStatus.FAILED
            self.logger.error(f"System processing failed: {e}")
            
            return SystemResult(
                success=False,
                features=pd.DataFrame(),
                selected_features=[],
                patch_features={},
                error_message=str(e),
                metadata={'system_status': self.status.value}
            )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'status': self.status.value,
            'components_initialized': {
                'assembly_dag': self.assembly_dag is not None,
                'monitoring_system': self.monitoring_system is not None,
                'rollout_orchestrator': self.rollout_orchestrator is not None,
                'validation_config': self.validation_config is not None
            },
            'config': {
                'feature_budget_pre': self.config.feature_budget_pre,
                'feature_budget_post': self.config.feature_budget_post,
                'interactions_cap': self.config.interactions_cap,
                'latency_budget_ms': self.config.latency_budget_ms,
                'lookback_ceiling_minutes': self.config.lookback_ceiling_minutes
            }
        }
    
    def save_artifacts(self, filepath: str, result: SystemResult):
        """Save system artifacts to file."""
        if result.artifacts is None:
            self.logger.warning("No artifacts to save")
            return
        
        try:
            artifacts_data = {
                'system_config': self.config.__dict__,
                'result_metadata': result.metadata,
                'artifacts': {
                    'transform_params': result.artifacts.transform_params,
                    'lookback_choices': result.artifacts.lookback_choices,
                    'interaction_configs': result.artifacts.interaction_configs,
                    'model_artifacts': result.artifacts.model_artifacts,
                    'spec_hash': result.artifacts.spec_hash
                },
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                import json
                json.dump(artifacts_data, f, indent=2, default=str)
            
            self.logger.info(f"Artifacts saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save artifacts: {e}")


def create_end_to_end_system(config: Optional[SystemConfig] = None) -> EndToEndRoadmapSystem:
    """Create end-to-end roadmap system with configuration."""
    if config is None:
        config = SystemConfig()
    
    return EndToEndRoadmapSystem(config)


def run_end_to_end_pipeline(bars: pd.DataFrame,
                           targets: Optional[Dict[int, pd.Series]] = None,
                           config: Optional[SystemConfig] = None,
                           enable_validation: bool = True,
                           enable_monitoring: bool = True,
                           enable_deployment: bool = False) -> SystemResult:
    """Run the complete end-to-end pipeline."""
    
    system = create_end_to_end_system(config)
    return system.process_market_data(
        bars, targets, enable_validation, enable_monitoring, enable_deployment
    )


# Example usage and testing
def create_sample_data(n_samples: int = 1000) -> Tuple[pd.DataFrame, Dict[int, pd.Series]]:
    """Create sample data for testing."""
    
    # Create sample bars
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='5min')
    bars = pd.DataFrame({
        'timestamp': dates,
        'open': 100 + np.cumsum(np.random.randn(n_samples) * 0.01),
        'high': 100 + np.cumsum(np.random.randn(n_samples) * 0.01) + np.abs(np.random.randn(n_samples) * 0.005),
        'low': 100 + np.cumsum(np.random.randn(n_samples) * 0.01) - np.abs(np.random.randn(n_samples) * 0.005),
        'close': 100 + np.cumsum(np.random.randn(n_samples) * 0.01),
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    # Create sample targets
    targets = {
        1: pd.Series(np.random.randn(n_samples), index=bars.index),
        3: pd.Series(np.random.randn(n_samples), index=bars.index)
    }
    
    return bars, targets


if __name__ == "__main__":
    # Example usage
    print("End-to-End Roadmap System")
    print("=" * 50)
    
    # Create sample data
    bars, targets = create_sample_data(500)
    print(f"Created sample data: {len(bars)} bars, {len(targets)} targets")
    
    # Run the pipeline
    result = run_end_to_end_pipeline(
        bars, targets,
        enable_validation=True,
        enable_monitoring=True,
        enable_deployment=False
    )
    
    if result.success:
        print(f"✅ Pipeline completed successfully!")
        print(f"   Features generated: {len(result.features.columns)}")
        print(f"   Selected features: {len(result.selected_features)}")
        print(f"   Patch features: {len(result.patch_features)}")
    else:
        print(f"❌ Pipeline failed: {result.error_message}")