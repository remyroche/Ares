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
from typing import Any, Dict, Optional, Union

import mlflow
import pandas as pd

from src.config import ARES_VERSION
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


class EnhancedMLflowManager:
    """Manager for enhanced MLflow operations in the enhanced training manager pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the enhanced MLflow manager.
        
        Args:
            config: Configuration dictionary from enhanced training manager
        """
        self.config = config
        self.metadata = extract_training_metadata(config)
        self.current_run_id: Optional[str] = None
        self.logger = system_logger
        
        # Set up MLflow
        self._setup_mlflow()
    
    def _setup_mlflow(self) -> None:
        """Set up MLflow tracking and experiment."""
        try:
            tracking_uri = self.config.get("mlflow", {}).get("tracking_uri") or "file:./mlruns"
            experiment_name = self.config.get("mlflow", {}).get("experiment_name") or "ares_training"
            
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            
            self.logger.info(f"✅ MLflow setup complete: {tracking_uri}, experiment: {experiment_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup MLflow: {e}")
            raise
    
    def start_run(self, run_name: Optional[str] = None, step_name: Optional[str] = None) -> str:
        """Start an MLflow run with enhanced metadata.
        
        Args:
            run_name: Optional custom run name
            step_name: Optional pipeline step name
            
        Returns:
            MLflow run ID
        """
        try:
            if not run_name:
                run_name = f"{self.metadata['exchange']}_{self.metadata['asset']}_{step_name or 'training'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            with mlflow.start_run(run_name=run_name) as run:
                self.current_run_id = run.info.run_id
                
                # Log enhanced training metadata
                log_enhanced_training_metadata(
                    asset=self.metadata["asset"],
                    exchange=self.metadata["exchange"],
                    lookback_period=self.metadata["lookback_period"],
                    project_version=self.metadata["project_version"],
                    run_id=self.current_run_id,
                    additional_metadata={
                        "step_name": step_name,
                        "run_name": run_name,
                        "pipeline": "enhanced_training_manager",
                    }
                )
                
                self.logger.info(f"✅ Started enhanced MLflow run: {self.current_run_id}")
                return self.current_run_id
                
        except Exception as e:
            self.logger.error(f"Failed to start MLflow run: {e}")
            raise
    
    def log_model(
        self,
        model: Any,
        model_name: str,
        model_type: str,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a model with enhanced metadata.
        
        Args:
            model: The trained model to log
            model_name: Name of the model
            model_type: Type of model (e.g., "hmm", "analyst", "tactician")
            additional_metadata: Additional metadata to log
        """
        if not self.current_run_id:
            raise ValueError("No active MLflow run. Call start_run() first.")
        
        try:
            # Prepare additional metadata
            extra_metadata = {
                "model_type": model_type,
                "pipeline_step": "model_logging",
            }
            if additional_metadata:
                extra_metadata.update(additional_metadata)
            
            # Log model with metadata
            log_model_with_metadata(
                model=model,
                model_name=model_name,
                asset=self.metadata["asset"],
                exchange=self.metadata["exchange"],
                lookback_period=self.metadata["lookback_period"],
                project_version=self.metadata["project_version"],
                run_id=self.current_run_id,
                additional_metadata=extra_metadata,
            )
            
            self.logger.info(f"✅ Logged model '{model_name}' with enhanced metadata")
            
        except Exception as e:
            self.logger.error(f"Failed to log model '{model_name}': {e}")
            raise
    
    def log_metrics(
        self,
        metrics: Dict[str, Union[int, float]],
        step: Optional[int] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log metrics with enhanced metadata.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
            additional_metadata: Additional metadata to log
        """
        if not self.current_run_id:
            raise ValueError("No active MLflow run. Call start_run() first.")
        
        try:
            # Convert metrics to float
            float_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            
            if not float_metrics:
                self.logger.warning("No valid metrics to log")
                return
            
            # Prepare additional metadata
            extra_metadata = {
                "pipeline_step": "metrics_logging",
            }
            if additional_metadata:
                extra_metadata.update(additional_metadata)
            
            # Log metrics with metadata
            log_metrics_with_metadata(
                metrics=float_metrics,
                asset=self.metadata["asset"],
                exchange=self.metadata["exchange"],
                lookback_period=self.metadata["lookback_period"],
                project_version=self.metadata["project_version"],
                run_id=self.current_run_id,
                step=step,
                additional_metadata=extra_metadata,
            )
            
            self.logger.info(f"✅ Logged {len(float_metrics)} metrics with enhanced metadata")
            
        except Exception as e:
            self.logger.error(f"Failed to log metrics: {e}")
            raise
    
    def log_parameters(
        self,
        parameters: Dict[str, Any],
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log parameters with enhanced metadata.
        
        Args:
            parameters: Dictionary of parameters to log
            additional_metadata: Additional metadata to log
        """
        if not self.current_run_id:
            raise ValueError("No active MLflow run. Call start_run() first.")
        
        try:
            # Prepare additional metadata
            extra_metadata = {
                "pipeline_step": "parameters_logging",
            }
            if additional_metadata:
                extra_metadata.update(additional_metadata)
            
            # Log parameters with metadata
            log_params_with_metadata(
                params=parameters,
                asset=self.metadata["asset"],
                exchange=self.metadata["exchange"],
                lookback_period=self.metadata["lookback_period"],
                project_version=self.metadata["project_version"],
                run_id=self.current_run_id,
                additional_metadata=extra_metadata,
            )
            
            self.logger.info(f"✅ Logged {len(parameters)} parameters with enhanced metadata")
            
        except Exception as e:
            self.logger.error(f"Failed to log parameters: {e}")
            raise
    
    def log_artifact(
        self,
        local_path: str,
        artifact_path: str,
        artifact_type: str,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an artifact with enhanced metadata.
        
        Args:
            local_path: Local path to the artifact
            artifact_path: Path within the MLflow run
            artifact_type: Type of artifact (e.g., "data", "model", "plot")
            additional_metadata: Additional metadata to log
        """
        if not self.current_run_id:
            raise ValueError("No active MLflow run. Call start_run() first.")
        
        try:
            # Prepare additional metadata
            extra_metadata = {
                "artifact_type": artifact_type,
                "pipeline_step": "artifact_logging",
            }
            if additional_metadata:
                extra_metadata.update(additional_metadata)
            
            # Log artifact with metadata
            log_artifacts_with_metadata(
                local_path=local_path,
                artifact_path=artifact_path,
                asset=self.metadata["asset"],
                exchange=self.metadata["exchange"],
                lookback_period=self.metadata["lookback_period"],
                project_version=self.metadata["project_version"],
                run_id=self.current_run_id,
                additional_metadata=extra_metadata,
            )
            
            self.logger.info(f"✅ Logged artifact '{artifact_path}' with enhanced metadata")
            
        except Exception as e:
            self.logger.error(f"Failed to log artifact '{artifact_path}': {e}")
            raise
    
    def log_dataframe(
        self,
        df: pd.DataFrame,
        artifact_path: str,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a DataFrame as an artifact with enhanced metadata.
        
        Args:
            df: DataFrame to log
            artifact_path: Path within the MLflow run
            additional_metadata: Additional metadata to log
        """
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
                df.to_parquet(tmp_file.name, index=False)
                tmp_path = tmp_file.name
            
            # Prepare additional metadata
            extra_metadata = {
                "artifact_type": "dataframe",
                "dataframe_shape": list(df.shape),
                "dataframe_columns": list(df.columns),
                "dataframe_dtypes": df.dtypes.to_dict(),
            }
            if additional_metadata:
                extra_metadata.update(additional_metadata)
            
            # Log artifact
            self.log_artifact(
                local_path=tmp_path,
                artifact_path=artifact_path,
                artifact_type="dataframe",
                additional_metadata=extra_metadata,
            )
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
        except Exception as e:
            self.logger.error(f"Failed to log DataFrame: {e}")
            raise
    
    def log_training_summary(
        self,
        summary: Dict[str, Any],
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log training summary with enhanced metadata.
        
        Args:
            summary: Training summary dictionary
            additional_metadata: Additional metadata to log
        """
        try:
            # Create temporary file
            import json
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp_file:
                json.dump(summary, tmp_file, indent=2, default=str)
                tmp_path = tmp_file.name
            
            # Prepare additional metadata
            extra_metadata = {
                "artifact_type": "training_summary",
                "summary_keys": list(summary.keys()),
                "summary_size": len(summary),
            }
            if additional_metadata:
                extra_metadata.update(additional_metadata)
            
            # Log artifact
            self.log_artifact(
                local_path=tmp_path,
                artifact_path="artifacts/training_summary.json",
                artifact_type="training_summary",
                additional_metadata=extra_metadata,
            )
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
        except Exception as e:
            self.logger.error(f"Failed to log training summary: {e}")
            raise
    
    def validate_current_run(self) -> bool:
        """Validate that the current run has all required metadata.
        
        Returns:
            True if validation passes, False otherwise
        """
        if not self.current_run_id:
            self.logger.warning("No active run to validate")
            return False
        
        return validate_run_metadata(self.current_run_id)
    
    def get_run_metadata(self) -> Dict[str, Any]:
        """Get metadata for the current run.
        
        Returns:
            Dictionary containing run metadata
        """
        if not self.current_run_id:
            raise ValueError("No active MLflow run")
        
        from src.utils.mlflow_utils import get_enhanced_run_metadata
        return get_enhanced_run_metadata(self.current_run_id) or {}
    
    def end_run(self) -> None:
        """End the current MLflow run."""
        if self.current_run_id:
            mlflow.end_run()
            self.logger.info(f"✅ Ended MLflow run: {self.current_run_id}")
            self.current_run_id = None


@handle_errors(default_return=None, context="enhanced_mlflow_integration.log_step_metadata")
def log_step_metadata(
    config: Dict[str, Any],
    step_name: str,
    step_data: Dict[str, Any],
    run_id: Optional[str] = None,
) -> None:
    """Log metadata for a specific pipeline step.
    
    Args:
        config: Configuration dictionary
        step_name: Name of the pipeline step
        step_data: Data from the pipeline step
        run_id: Optional MLflow run ID
    """
    try:
        metadata = extract_training_metadata(config)
        
        # Log enhanced training metadata for the step
        log_enhanced_training_metadata(
            asset=metadata["asset"],
            exchange=metadata["exchange"],
            lookback_period=metadata["lookback_period"],
            project_version=metadata["project_version"],
            run_id=run_id,
            additional_metadata={
                "pipeline_step": step_name,
                "step_status": step_data.get("status", "unknown"),
                "step_duration": step_data.get("duration", 0.0),
                "step_data_keys": list(step_data.keys()),
            }
        )
        
        system_logger.info(f"✅ Logged metadata for step: {step_name}")
        
    except Exception as e:
        system_logger.error(f"Failed to log step metadata for {step_name}: {e}")


@handle_errors(default_return=None, context="enhanced_mlflow_integration.log_model_performance")
def log_model_performance(
    config: Dict[str, Any],
    model_name: str,
    model_type: str,
    performance_metrics: Dict[str, float],
    run_id: Optional[str] = None,
) -> None:
    """Log model performance metrics with enhanced metadata.
    
    Args:
        config: Configuration dictionary
        model_name: Name of the model
        model_type: Type of model
        performance_metrics: Performance metrics dictionary
        run_id: Optional MLflow run ID
    """
    try:
        metadata = extract_training_metadata(config)
        
        # Log metrics with metadata
        log_metrics_with_metadata(
            metrics=performance_metrics,
            asset=metadata["asset"],
            exchange=metadata["exchange"],
            lookback_period=metadata["lookback_period"],
            project_version=metadata["project_version"],
            run_id=run_id,
            additional_metadata={
                "model_name": model_name,
                "model_type": model_type,
                "pipeline_step": "model_performance_logging",
            }
        )
        
        system_logger.info(f"✅ Logged performance metrics for model: {model_name}")
        
    except Exception as e:
        system_logger.error(f"Failed to log model performance for {model_name}: {e}")


@handle_errors(default_return=None, context="enhanced_mlflow_integration.log_pipeline_completion")
def log_pipeline_completion(
    config: Dict[str, Any],
    pipeline_results: Dict[str, Any],
    run_id: Optional[str] = None,
) -> None:
    """Log pipeline completion with enhanced metadata.
    
    Args:
        config: Configuration dictionary
        pipeline_results: Results from the pipeline execution
        run_id: Optional MLflow run ID
    """
    try:
        metadata = extract_training_metadata(config)
        
        # Log enhanced training metadata for pipeline completion
        log_enhanced_training_metadata(
            asset=metadata["asset"],
            exchange=metadata["exchange"],
            lookback_period=metadata["lookback_period"],
            project_version=metadata["project_version"],
            run_id=run_id,
            additional_metadata={
                "pipeline_step": "pipeline_completion",
                "pipeline_status": pipeline_results.get("status", "unknown"),
                "completed_steps": len([k for k, v in pipeline_results.items() if v]),
                "total_steps": len(pipeline_results),
                "completion_timestamp": datetime.now().isoformat(),
            }
        )
        
        system_logger.info("✅ Logged pipeline completion metadata")
        
    except Exception as e:
        system_logger.error(f"Failed to log pipeline completion: {e}")