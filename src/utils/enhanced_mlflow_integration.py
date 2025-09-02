# src/utils/enhanced_mlflow_integration.py

"""
Enhanced MLflow Integration for Enhanced Training Manager

This module provides comprehensive MLflow integration that ensures all models
in the enhanced_training_manager pipeline are properly associated with:
- asset: The trading asset/symbol
- exchange: The trading exchange
- lookback_period: The data lookback period used for training
- project_version: The current project version
- date: The training date

This ensures complete traceability and reproducibility of all training runs.
"""

import os
import tempfile
from datetime import datetime
from typing import Any, Dict, Optional, Union, List
from functools import wraps
import sys

import mlflow
import pandas as pd

from src.utils.logger import system_logger
from src.utils.mlflow_utils import (
    extract_training_metadata,
    log_artifacts_with_metadata,
    log_enhanced_training_metadata,
    log_metrics_with_metadata,
    log_model_with_metadata,
    log_params_with_metadata,
    validate_run_metadata,
)
from src.utils.error_handler import handle_errors


def with_enhanced_mlflow_logging(step_name: str):
    """Decorator to automatically add enhanced MLflow logging to pipeline steps.

    This decorator ensures that all step executions are properly logged to MLflow
    with the required metadata associations.

    Args:
        step_name: Name of the pipeline step (e.g., "step03_hmm_regime_discovery")

    Usage:
        @with_enhanced_mlflow_logging("step03_hmm_regime_discovery")
        async def execute(self, training_input, pipeline_state):
            # Step execution logic
            return results
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any], *args, **kwargs):
            # Extract metadata from config
            config = getattr(self, 'config', {})
            metadata = extract_training_metadata(config)

            # Extract step-specific information
            symbol = training_input.get("symbol", metadata["asset"])
            exchange = training_input.get("exchange", metadata["exchange"])

            # Start MLflow run for this step
            run_id = None
            try:
                # Set up MLflow
                tracking_uri = config.get("mlflow", {}).get("tracking_uri") or "file:./mlruns"
                experiment_name = config.get("mlflow", {}).get("experiment_name") or "ares_training"

                mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(experiment_name)

                # Create run name
                run_name = f"{exchange}_{symbol}_{step_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                with mlflow.start_run(run_name=run_name) as run:
                    run_id = run.info.run_id

                    # Log enhanced training metadata for the step
                    log_enhanced_training_metadata(
                        asset=metadata["asset"],
                        exchange=metadata["exchange"],
                        lookback_period=metadata["lookback_period"],
                        project_version=metadata["project_version"],
                        run_id=run_id,
                        additional_metadata={
                            "pipeline_step": step_name,
                            "step_execution_start": datetime.now().isoformat(),
                            "training_input_keys": list(training_input.keys()),
                            "pipeline_state_keys": list(pipeline_state.keys()),
                        }
                    )

                    # Log step parameters
                    step_params = {
                        "step_name": step_name,
                        "symbol": symbol,
                        "exchange": exchange,
                        "lookback_years": config.get("lookback_years", 2),
                        "timeframe": training_input.get("timeframe", "1h"),
                    }

                    log_params_with_metadata(
                        params=step_params,
                        asset=metadata["asset"],
                        exchange=metadata["exchange"],
                        lookback_period=metadata["lookback_period"],
                        project_version=metadata["project_version"],
                        run_id=run_id,
                        additional_metadata={
                            "parameter_type": "step_configuration",
                        }
                    )

                    # Execute the step
                    start_time = datetime.now()
                    result = await func(self, training_input, pipeline_state, *args, **kwargs)
                    end_time = datetime.now()
                    execution_duration = (end_time - start_time).total_seconds()

                    # Log step completion metadata
                    completion_metadata = {
                        "step_execution_end": end_time.isoformat(),
                        "execution_duration_seconds": execution_duration,
                        "step_status": "completed" if result else "failed",
                        "result_keys": list(result.keys()) if isinstance(result, dict) else [],
                    }

                    log_enhanced_training_metadata(
                        asset=metadata["asset"],
                        exchange=metadata["exchange"],
                        lookback_period=metadata["lookback_period"],
                        project_version=metadata["project_version"],
                        run_id=run_id,
                        additional_metadata=completion_metadata
                    )

                    # Log step metrics
                    if isinstance(result, dict):
                        metrics = {}
                        for key, value in result.items():
                            if isinstance(value, (int, float)) and key not in ["status", "duration"]:
                                metrics[f"step_{key}"] = float(value)

                        if metrics:
                            log_metrics_with_metadata(
                                metrics=metrics,
                                asset=metadata["asset"],
                                exchange=metadata["exchange"],
                                lookback_period=metadata["lookback_period"],
                                project_version=metadata["project_version"],
                                run_id=run_id,
                                additional_metadata={
                                    "metrics_type": "step_execution",
                                    "step_name": step_name,
                                }
                            )

                    system_logger.info(f"✅ Step {step_name} executed and logged to MLflow (Run ID: {run_id})")
                    return result

            except Exception as e:
                system_logger.error(f"❌ MLflow logging failed for step {step_name}: {e}")
                # Still execute the step even if MLflow logging fails
                return await func(self, training_input, pipeline_state, *args, **kwargs)

        return wrapper
    return decorator


def log_step_artifact(
    config: Dict[str, Any],
    step_name: str,
    artifact_path: str,
    artifact_type: str,
    run_id: Optional[str] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Log a step artifact with enhanced metadata.

    Args:
        config: Configuration dictionary
        step_name: Name of the pipeline step
        artifact_path: Path to the artifact file
        artifact_type: Type of artifact (e.g., "model", "data", "plot")
        run_id: Optional MLflow run ID
        additional_metadata: Additional metadata to log
    """
    try:
        metadata = extract_training_metadata(config)
        
        log_artifacts_with_metadata(
            artifact_path=artifact_path,
            asset=metadata["asset"],
            exchange=metadata["exchange"],
            lookback_period=metadata["lookback_period"],
            project_version=metadata["project_version"],
            run_id=run_id,
            additional_metadata={
                "artifact_type": artifact_type,
                "pipeline_step": step_name,
                **(additional_metadata or {})
            }
        )
        
        system_logger.info(f"✅ Artifact logged for step {step_name}: {artifact_path}")
        
    except Exception as e:
        system_logger.error(f"❌ Failed to log artifact for step {step_name}: {e}")


class EnhancedMLflowManager:
    """Enhanced MLflow manager for comprehensive experiment tracking."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the enhanced MLflow manager.
        
        Args:
            config: Configuration dictionary containing MLflow settings
        """
        self.config = config
        self.logger = system_logger.getChild("EnhancedMLflowManager")
        self.tracking_uri = config.get("mlflow", {}).get("tracking_uri", "file:./mlruns")
        self.experiment_name = config.get("mlflow", {}).get("experiment_name", "ares_training")
        self.registry_uri = config.get("mlflow", {}).get("registry_uri")
        
        # Initialize MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        if self.registry_uri:
            mlflow.set_registry_uri(self.registry_uri)
        
        self.logger.info(f"🚀 Enhanced MLflow Manager initialized with tracking URI: {self.tracking_uri}")
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="MLflow experiment creation"
    )
    def create_experiment(self, experiment_name: Optional[str] = None) -> str:
        """Create or get an MLflow experiment.
        
        Args:
            experiment_name: Name of the experiment (uses default if None)
            
        Returns:
            Experiment ID
        """
        exp_name = experiment_name or self.experiment_name
        experiment = mlflow.get_experiment_by_name(exp_name)
        
        if experiment is None:
            experiment_id = mlflow.create_experiment(exp_name)
            self.logger.info(f"✅ Created new experiment: {exp_name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            self.logger.info(f"📋 Using existing experiment: {exp_name} (ID: {experiment_id})")
        
        return experiment_id
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="MLflow run creation"
    )
    def start_run(
        self, 
        run_name: str, 
        tags: Optional[Dict[str, str]] = None,
        experiment_name: Optional[str] = None
    ) -> mlflow.ActiveRun:
        """Start a new MLflow run.
        
        Args:
            run_name: Name of the run
            tags: Additional tags for the run
            experiment_name: Experiment name (uses default if None)
            
        Returns:
            Active MLflow run
        """
        exp_name = experiment_name or self.experiment_name
        mlflow.set_experiment(exp_name)
        
        run_tags = tags or {}
        run_tags.update({
            "created_by": "EnhancedMLflowManager",
            "timestamp": datetime.now().isoformat()
        })
        
        run = mlflow.start_run(run_name=run_name, tags=run_tags)
        self.logger.info(f"🚀 Started MLflow run: {run_name} (ID: {run.info.run_id})")
        
        return run
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="MLflow model logging"
    )
    def log_model(
        self,
        model,
        artifact_path: str,
        registered_model_name: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Log a model to MLflow.
        
        Args:
            model: The model to log
            artifact_path: Path where the model will be stored
            registered_model_name: Name for model registration
            **kwargs: Additional arguments for mlflow.log_model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            mlflow.log_model(model, artifact_path, **kwargs)
            
            if registered_model_name:
                mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}", registered_model_name)
                self.logger.info(f"✅ Model registered: {registered_model_name}")
            
            self.logger.info(f"✅ Model logged successfully: {artifact_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log model: {e}")
            return False
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="MLflow metrics logging"
    )
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> bool:
        """Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Step number for the metrics
            
        Returns:
            True if successful, False otherwise
        """
        try:
            mlflow.log_metrics(metrics, step=step)
            self.logger.info(f"✅ Metrics logged: {list(metrics.keys())}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log metrics: {e}")
            return False
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="MLflow parameters logging"
    )
    def log_params(self, params: Dict[str, Any]) -> bool:
        """Log parameters to MLflow.
        
        Args:
            params: Dictionary of parameter names and values
            
        Returns:
            True if successful, False otherwise
        """
        try:
            mlflow.log_params(params)
            self.logger.info(f"✅ Parameters logged: {list(params.keys())}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log parameters: {e}")
            return False
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="MLflow artifacts logging"
    )
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> bool:
        """Log artifacts to MLflow.
        
        Args:
            local_dir: Local directory containing artifacts
            artifact_path: Path where artifacts will be stored
            
        Returns:
            True if successful, False otherwise
        """
        try:
            mlflow.log_artifacts(local_dir, artifact_path)
            self.logger.info(f"✅ Artifacts logged from: {local_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log artifacts: {e}")
            return False
    
    def get_run_info(self) -> Dict[str, Any]:
        """Get information about the current run.
        
        Returns:
            Dictionary containing run information
        """
        try:
            run = mlflow.active_run()
            if run is None:
                return {"error": "No active run"}
            
            return {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "artifact_uri": run.info.artifact_uri,
                "lifecycle_stage": run.info.lifecycle_stage
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get run info: {e}")
            return {"error": str(e)}
    
    def end_run(self) -> None:
        """End the current MLflow run."""
        try:
            mlflow.end_run()
            self.logger.info("✅ MLflow run ended")
        except Exception as e:
            self.logger.error(f"❌ Failed to end run: {e}")


class MLflowExperimentTracker:
    """Tracks MLflow experiments with enhanced metadata."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the experiment tracker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("MLflowExperimentTracker")
        self.mlflow_manager = EnhancedMLflowManager(config)
    
    def track_training_run(
        self,
        step_name: str,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any],
        model=None,
        metrics: Optional[Dict[str, float]] = None,
        params: Optional[Dict[str, Any]] = None,
        artifacts_dir: Optional[str] = None
    ) -> Optional[str]:
        """Track a complete training run in MLflow.
        
        Args:
            step_name: Name of the training step
            training_input: Input data for training
            pipeline_state: Current pipeline state
            model: Trained model to log
            metrics: Training metrics
            params: Training parameters
            artifacts_dir: Directory containing training artifacts
            
        Returns:
            Run ID if successful, None otherwise
        """
        try:
            # Extract metadata
            metadata = extract_training_metadata(self.config)
            
            # Create run name
            run_name = f"{metadata['exchange']}_{metadata['asset']}_{step_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Start run
            with self.mlflow_manager.start_run(run_name) as run:
                run_id = run.info.run_id
                
                # Log enhanced metadata
                log_enhanced_training_metadata(
                    asset=metadata["asset"],
                    exchange=metadata["exchange"],
                    lookback_period=metadata["lookback_period"],
                    project_version=metadata["project_version"],
                    run_id=run_id,
                    additional_metadata={
                        "pipeline_step": step_name,
                        "training_input_keys": list(training_input.keys()),
                        "pipeline_state_keys": list(pipeline_state.keys()),
                    }
                )
                
                # Log parameters
                if params:
                    self.mlflow_manager.log_params(params)
                
                # Log metrics
                if metrics:
                    self.mlflow_manager.log_metrics(metrics)
                
                # Log model
                if model:
                    self.mlflow_manager.log_model(model, "model")
                
                # Log artifacts
                if artifacts_dir and os.path.exists(artifacts_dir):
                    self.mlflow_manager.log_artifacts(artifacts_dir, "artifacts")
                
                self.logger.info(f"✅ Training run tracked successfully: {run_id}")
                return run_id
                
        except Exception as e:
            self.logger.error(f"❌ Failed to track training run: {e}")
            return None


# Global instances
enhanced_mlflow_manager = None
experiment_tracker = None


def initialize_mlflow_integration(config: Dict[str, Any]) -> None:
    """Initialize global MLflow integration instances.
    
    Args:
        config: Configuration dictionary
    """
    global enhanced_mlflow_manager, experiment_tracker
    
    try:
        enhanced_mlflow_manager = EnhancedMLflowManager(config)
        experiment_tracker = MLflowExperimentTracker(config)
        system_logger.info("✅ MLflow integration initialized successfully")
    except Exception as e:
        system_logger.error(f"❌ Failed to initialize MLflow integration: {e}")


def get_mlflow_manager() -> Optional[EnhancedMLflowManager]:
    """Get the global MLflow manager instance."""
    return enhanced_mlflow_manager


def get_experiment_tracker() -> Optional[MLflowExperimentTracker]:
    """Get the global experiment tracker instance."""
    return experiment_tracker