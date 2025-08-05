# src/utils/mlflow_utils.py

from datetime import datetime
from typing import Any

import mlflow

from src.config import ARES_VERSION
from src.utils.logger import system_logger


def log_bot_version_to_mlflow(run_id: str | None = None) -> None:
    """
    Log the current bot version to MLFlow.

    Args:
        run_id: Optional MLFlow run ID. If None, uses the active run.
    """
    try:
        if run_id:
            with mlflow.start_run(run_id=run_id):
                mlflow.set_tag("bot_version", ARES_VERSION)
                mlflow.set_tag("training_date", datetime.now().isoformat())
                system_logger.info(
                    f"✅ Logged bot version {ARES_VERSION} to MLFlow run {run_id}",
                )
        else:
            mlflow.set_tag("bot_version", ARES_VERSION)
            mlflow.set_tag("training_date", datetime.now().isoformat())
            system_logger.info(
                f"✅ Logged bot version {ARES_VERSION} to active MLFlow run",
            )
    except Exception as e:
        system_logger.error(f"❌ Failed to log bot version to MLFlow: {e}")


def log_training_metadata_to_mlflow(
    symbol: str,
    timeframe: str,
    model_type: str,
    run_id: str | None = None,
) -> None:
    """
    Log training metadata including bot version to MLFlow.

    Args:
        symbol: Trading symbol
        timeframe: Timeframe used for training
        model_type: Type of model being trained
        run_id: Optional MLFlow run ID. If None, uses the active run.
    """
    try:
        metadata = {
            "bot_version": ARES_VERSION,
            "training_date": datetime.now().isoformat(),
            "model_type": model_type,
            "symbol": symbol,
            "timeframe": timeframe,
        }

        if run_id:
            with mlflow.start_run(run_id=run_id):
                for key, value in metadata.items():
                    mlflow.set_tag(key, value)
                system_logger.info(
                    f"✅ Logged training metadata to MLFlow run {run_id}",
                )
        else:
            for key, value in metadata.items():
                mlflow.set_tag(key, value)
            system_logger.info("✅ Logged training metadata to active MLFlow run")

    except Exception as e:
        system_logger.error(f"❌ Failed to log training metadata to MLFlow: {e}")


def get_run_with_bot_version(run_id: str) -> dict[str, Any] | None:
    """
    Get MLFlow run information including bot version.

    Args:
        run_id: MLFlow run ID

    Returns:
        Dict containing run information with bot version, or None if not found
    """
    try:
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)

        run_info = {
            "run_id": run_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "bot_version": run.data.tags.get("bot_version", "Unknown"),
            "training_date": run.data.tags.get("training_date", "Unknown"),
            "model_type": run.data.tags.get("model_type", "Unknown"),
            "symbol": run.data.tags.get("symbol", "Unknown"),
            "timeframe": run.data.tags.get("timeframe", "Unknown"),
        }

        return run_info

    except Exception as e:
        system_logger.error(f"❌ Failed to get MLFlow run {run_id}: {e}")
        return None
