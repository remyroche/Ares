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
        if not os.path.exists(artifact_path):
            system_logger.warning(f"Artifact file not found: {artifact_path}")
            return

        metadata = extract_training_metadata(config)

        # Prepare additional metadata
        extra_metadata = {
            "artifact_type": artifact_type,
            "pipeline_step": step_name,
            "artifact_filename": os.path.basename(artifact_path),
            "artifact_size_bytes": os.path.getsize(artifact_path),
        }
        if additional_metadata:
            extra_metadata.update(additional_metadata)

        # Log artifact with metadata
        log_artifacts_with_metadata(
            local_path=artifact_path,
            artifact_path=f"artifacts/{step_name}/{os.path.basename(artifact_path)}",
            asset=metadata["asset"],
            exchange=metadata["exchange"],
            lookback_period=metadata["lookback_period"],
            project_version=metadata["project_version"],
            run_id=run_id,
            additional_metadata=extra_metadata,
        )

        system_logger.info(f"✅ Logged artifact '{artifact_path}' for step {step_name}")

    except Exception as e:
        system_logger.error(f"Failed to log artifact '{artifact_path}' for step {step_name}: {e}")


def generate_standardized_artifact_name(
    exchange: str,
    token: str,
    step_number: str,
    artifact_type: str,
    extension: str = "",
    timestamp: Optional[datetime] = None
) -> str:
    """Generate standardized artifact name following the pattern: exchange_token_date_hourminute_NumberOfStep_Artifact

    Args:
        exchange: Exchange name (e.g., "BINANCE")
        token: Token/symbol name (e.g., "ETHUSDT")
        step_number: Step number (e.g., "step3", "step6")
        artifact_type: Type of artifact (e.g., "composite_clusters", "features_train", "hmm_model")
        extension: File extension (e.g., ".parquet", ".pkl", ".json")
        timestamp: Optional timestamp, defaults to current time

    Returns:
        Standardized artifact name
    """
    if timestamp is None:
        # Fallback implementation for timestamp
        timestamp = datetime.now()

    date_str = timestamp.strftime("%Y%m%d")
    time_str = timestamp.strftime("%H%M")

    # Clean up step number to just the number
    step_num = step_number.replace("step", "").replace("_", "")

    # Clean up artifact type (replace spaces and special chars with underscores)
    clean_artifact_type = artifact_type.replace(" ", "_").replace("-", "_").lower()

    # Build the standardized name
    artifact_name = f"{exchange}_{token}_{date_str}_{time_str}_{step_num}_{clean_artifact_type}"

    if extension:
        if not extension.startswith("."):
            extension = "." + extension
        artifact_name += extension

    return artifact_name


def log_step_dataframe(
    config: Dict[str, Any],
    step_name: str,
    df: pd.DataFrame,
    artifact_name: str,
    run_id: Optional[str] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Log a DataFrame as an artifact for a specific step.

    Args:
        config: Configuration dictionary
        step_name: Name of the pipeline step
        df: DataFrame to log
        artifact_name: Name for the artifact
        run_id: Optional MLflow run ID
        additional_metadata: Additional metadata to log
    """
    try:
        metadata = extract_training_metadata(config)

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
        log_artifacts_with_metadata(
            local_path=tmp_path,
            artifact_path=f"artifacts/{step_name}/{artifact_name}.parquet",
            asset=metadata["asset"],
            exchange=metadata["exchange"],
            lookback_period=metadata["lookback_period"],
            project_version=metadata["project_version"],
            run_id=run_id,
            additional_metadata=extra_metadata,
        )

        # Clean up temporary file
        os.unlink(tmp_path)

        system_logger.info(f"✅ Logged DataFrame '{artifact_name}' for step {step_name}")

    except Exception as e:
        system_logger.error(f"Failed to log DataFrame '{artifact_name}' for step {step_name}: {e}")









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

@handle_errors(default_return=None, context="enhanced_mlflow_integration.log_model_performance")

@handle_errors(default_return=None, context="enhanced_mlflow_integration.log_pipeline_completion")
