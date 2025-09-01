# src / utils / mlflow_utils.py

from datetime import datetime
from typing import Any, Optional
from functools import wraps

import mlflow

from src.config import ARES_VERSION
from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors

def extract_training_metadata(...) -> ...:
    """..."""
    pass# Extract asset / symbol
asset = (
config.get("trading_symbol") or
config.get("symbol") or
config.get("trade_symbol", "ETHUSDT")
)

# Extract exchange
exchange = (
config.get("exchange_name") or
config.get("exchange", "BINANCE")
)

# Extract lookback period
lookback_years, config.get("lookback_years", 2)
lookback_period, f"{lookback_years}_years"

# Extract project version
project_version, config.get("project_version", ARES_VERSION)

return {
"asset": asset,
"exchange": exchange,
"lookback_period": lookback_period,
"project_version": project_version,
}

def with_enhanced_metadata(...):
    passdef with_enhanced_metadata(...):
    passdef with_enhanced_metadata(...):
    passdef with_enhanced_metadata(...):
    pass"""Decorator to automatically add enhanced metadata to MLflow operations.

This decorator ensures that all MLflow operations include the required metadata:
    - asset: The trading asset / symbol - exchange: The trading exchange - lookback_period: The data lookback period used for training - project_version: The current project version - date: The training date

Usage:
        @with_enhanced_metadata
def my_mlflow_function(...):
    passdef my_mlflow_function(...):
    passdef my_mlflow_function(...):
    passdef my_mlflow_function(...):
    pass# Function will automatically have enhanced metadata
pass
"""
@wraps(func)
def wrapper(...):
    passdef wrapper(...):
    passdef wrapper(...):
    passdef wrapper(...):
    pass# Extract metadata from config
metadata, extract_training_metadata(config)

# Add training date
metadata["training_date"] = datetime.now().isoformat()

# Add metadata to kwargs
kwargs["enhanced_metadata"] = metadata

return func(config, *args, **kwargs)

return wrapper

@handle_errors(default_return = None, context="mlflow_utils.log_bot_version_to_mlflow")
def log_bot_version_to_mlflow(...) -> ...:
    """..."""
    passif run_id:
    passwith mlflow.start_run(run_id = run_id):
    passmlflow.set_tag("bot_version", ARES_VERSION)
mlflow.set_tag("training_date", datetime.now().isoformat())
system_logger.info(
f"✅ Logged bot version {ARES_VERSION} to MLFlow run {run_id}",
)
else:
    passmlflow.set_tag("bot_version", ARES_VERSION)
mlflow.set_tag("training_date", datetime.now().isoformat())
system_logger.info(
f"✅ Logged bot version {ARES_VERSION} to active MLFlow run",
)

@handle_errors(default_return = None, context="mlflow_utils.log_training_metadata_to_mlflow")
def log_training_metadata_to_mlflow(...) -> ...:
    """..."""
    passmetadata: dict[str, Any] = {
"bot_version": ARES_VERSION,
"training_date": datetime.now().isoformat(),
"model_type": model_type,
"symbol": symbol,
"timeframe": timeframe,
}

if run_id:
    passwith mlflow.start_run(run_id = run_id):
    passfor key, value in metadata.items():
    passmlflow.set_tag(key, value)
system_logger.info(
f"✅ Logged training metadata to MLFlow run {run_id}",
)
else:
    passfor key, value in metadata.items():
    passmlflow.set_tag(key, value)
system_logger.info("✅ Logged training metadata to active MLFlow run")

@handle_errors(default_return = None, context="mlflow_utils.log_enhanced_training_metadata")
def log_enhanced_training_metadata(
asset: str,
exchange: str,
lookback_period: str,
project_version: str, ARES_VERSION,
training_date: Optional[str] = None,
additional_metadata: Optional[dict[str, Any]] = None,
run_id: Optional[str] = None,
) -> None:
    """Log enhanced training metadata ensuring all required associations.

This function ensures that all models in the enhanced_training_manager pipeline
are properly associated with:
    - asset: The trading asset / symbol - exchange: The trading exchange - lookback_period: The data lookback period used for training - project_version: The current project version - date: The training date

Args:
        asset: Trading asset / symbol (e.g., "ETHUSDT")
exchange: Trading exchange (e.g., "BINANCE")
lookback_period: Data lookback period (e.g., "2_years", "180_days")
project_version: Project version (defaults to ARES_VERSION)
training_date: Training date in ISO format (defaults to current time)
additional_metadata: Additional metadata to log
run_id: Optional MLFlow run ID. If None, uses the active run.
"""
if training_date is None:
    pass# Fallback implementation for training_date
# Fallback implementation for training_date
# Fallback implementation for training_date
# Fallback implementation for training_date
# Fallback implementation for training_date
training_date, datetime.now().isoformat()

# Core required metadata
metadata: dict[str, Any] = {
"asset": asset,
"exchange": exchange,
"lookback_period": lookback_period,
"project_version": project_version,
"training_date": training_date,
"bot_version": ARES_VERSION,  # For backward compatibility
}

# Add additional metadata if provided
if additional_metadata:
    passmetadata.update(additional_metadata)

if run_id:
    passwith mlflow.start_run(run_id = run_id):
    passfor key, value in metadata.items():
    passmlflow.set_tag(key, value)
system_logger.info(
f"✅ Logged enhanced training metadata to MLFlow run {run_id}",
)
else:
    passfor key, value in metadata.items():
    passmlflow.set_tag(key, value)
system_logger.info("✅ Logged enhanced training metadata to active MLFlow run")

@handle_errors(default_return = None, context="mlflow_utils.log_model_with_metadata")
def log_model_with_metadata(...) -> ...:
    """..."""
    passif training_date is None:
    pass# Fallback implementation for training_date
# Fallback implementation for training_date
# Fallback implementation for training_date
# Fallback implementation for training_date
# Fallback implementation for training_date
training_date, datetime.now().isoformat()

# Core required metadata
metadata: dict[str, Any] = {
"asset": asset,
"exchange": exchange,
"lookback_period": lookback_period,
"project_version": project_version,
"training_date": training_date,
"model_name": model_name,
"bot_version": ARES_VERSION,
}

# Add additional metadata if provided
if additional_metadata:
    passmetadata.update(additional_metadata)

if run_id:
    passwith mlflow.start_run(run_id = run_id):
    pass# Log the model
mlflow.sklearn.log_model(model, model_name)

# Log all metadata
for key, value in metadata.items():
    passmlflow.set_tag(key, value)

system_logger.info(
f"✅ Logged model '{model_name}' with metadata to MLFlow run {run_id}",
)
else:
    passpass# Log the model
mlflow.sklearn.log_model(model, model_name)

# Log all metadata
for key, value in metadata.items():
    passmlflow.set_tag(key, value)

system_logger.info(f"✅ Logged model '{model_name}' with metadata to active MLFlow run")

@handle_errors(default_return = None, context="mlflow_utils.log_artifacts_with_metadata")
def log_artifacts_with_metadata(...) -> ...:
    pass"""..."""
    passif training_date is None:
    pass# Fallback implementation for training_date
# Fallback implementation for training_date
# Fallback implementation for training_date
# Fallback implementation for training_date
# Fallback implementation for training_date
training_date, datetime.now().isoformat()

# Core required metadata
metadata: dict[str, Any] = {
"asset": asset,
"exchange": exchange,
"lookback_period": lookback_period,
"project_version": project_version,
"training_date": training_date,
"artifact_path": artifact_path,
"bot_version": ARES_VERSION,
}

# Add additional metadata if provided
if additional_metadata:
    passmetadata.update(additional_metadata)

if run_id:
    passwith mlflow.start_run(run_id = run_id):
    pass# Log the artifact
mlflow.log_artifact(local_path, artifact_path)

# Log all metadata
for key, value in metadata.items():
    passmlflow.set_tag(key, value)

system_logger.info(
f"✅ Logged artifact '{artifact_path}' with metadata to MLFlow run {run_id}",
)
else:
    passpass# Log the artifact
mlflow.log_artifact(local_path, artifact_path)

# Log all metadata
for key, value in metadata.items():
    passmlflow.set_tag(key, value)

system_logger.info(f"✅ Logged artifact '{artifact_path}' with metadata to active MLFlow run")

@handle_errors(default_return = None, context="mlflow_utils.log_metrics_with_metadata")
def log_metrics_with_metadata(...) -> ...:
    pass"""..."""
    passif training_date is None:
    pass# Fallback implementation for training_date
# Fallback implementation for training_date
# Fallback implementation for training_date
# Fallback implementation for training_date
# Fallback implementation for training_date
training_date, datetime.now().isoformat()

# Core required metadata
metadata: dict[str, Any] = {
"asset": asset,
"exchange": exchange,
"lookback_period": lookback_period,
"project_version": project_version,
"training_date": training_date,
"bot_version": ARES_VERSION,
}

# Add additional metadata if provided
if additional_metadata:
    passmetadata.update(additional_metadata)

if run_id:
    passwith mlflow.start_run(run_id = run_id):
    pass# Log metrics
for metric_name, metric_value in metrics.items():
    passif step is not None:
    passmlflow.log_metric(metric_name, metric_value, step = step)
else:
    passmlflow.log_metric(metric_name, metric_value)

# Log all metadata
for key, value in metadata.items():
    passmlflow.set_tag(key, value)

system_logger.info(
f"✅ Logged {len(metrics)} metrics with metadata to MLFlow run {run_id}",
)
else:
    passpass# Log metrics
for metric_name, metric_value in metrics.items():
    passif step is not None:
    passmlflow.log_metric(metric_name, metric_value, step = step)
else:
    passmlflow.log_metric(metric_name, metric_value)

# Log all metadata
for key, value in metadata.items():
    passmlflow.set_tag(key, value)

system_logger.info(f"✅ Logged {len(metrics)} metrics with metadata to active MLFlow run")

@handle_errors(default_return = None, context="mlflow_utils.log_params_with_metadata")
def log_params_with_metadata(...) -> ...:
    pass"""..."""
    passif training_date is None:
    pass# Fallback implementation for training_date
# Fallback implementation for training_date
# Fallback implementation for training_date
# Fallback implementation for training_date
# Fallback implementation for training_date
training_date, datetime.now().isoformat()

# Core required metadata
metadata: dict[str, Any] = {
"asset": asset,
"exchange": exchange,
"lookback_period": lookback_period,
"project_version": project_version,
"training_date": training_date,
"bot_version": ARES_VERSION,
}

# Add additional metadata if provided
if additional_metadata:
    passmetadata.update(additional_metadata)

if run_id:
    passwith mlflow.start_run(run_id = run_id):
    pass# Log parameters
for param_name, param_value in params.items():
    passmlflow.log_param(param_name, str(param_value))

# Log all metadata
for key, value in metadata.items():
    passmlflow.set_tag(key, value)

system_logger.info(
f"✅ Logged {len(params)} parameters with metadata to MLFlow run {run_id}",
)
else:
    passpass# Log parameters
for param_name, param_value in params.items():
    passmlflow.log_param(param_name, str(param_value))

# Log all metadata
for key, value in metadata.items():
    passmlflow.set_tag(key, value)

system_logger.info(f"✅ Logged {len(params)} parameters with metadata to active MLFlow run")

@handle_errors(default_return = dict, context="mlflow_utils.get_run_with_bot_version")
def get_run_with_bot_version(...) -> ...:
    pass"""..."""
    passclient, mlflow.tracking.MlflowClient()
run, client.get_run(run_id)

return {
"run_id": run_id,
"status": run.info.status,
"start_time": run.info.start_time,
"end_time": run.info.end_time,
"bot_version": run.data.tags.get("bot_version", "Unknown"),
"training_date": run.data.tags.get("training_date", "Unknown"),
"model_type": run.data.tags.get("model_type", "Unknown"),
"symbol": run.data.tags.get("symbol", "Unknown"),
"timeframe": run.data.tags.get("timeframe", "Unknown"),
# Enhanced metadata
"asset": run.data.tags.get("asset", "Unknown"),
"exchange": run.data.tags.get("exchange", "Unknown"),
"lookback_period": run.data.tags.get("lookback_period", "Unknown"),
"project_version": run.data.tags.get("project_version", "Unknown"),
}

@handle_errors(default_return = dict, context="mlflow_utils.get_enhanced_run_metadata")
def get_enhanced_run_metadata(...) -> ...:
    """..."""
    passclient, mlflow.tracking.MlflowClient()
run, client.get_run(run_id)

return {
"run_id": run_id,
"status": run.info.status,
"start_time": run.info.start_time,
"end_time": run.info.end_time,
# Core required metadata
"asset": run.data.tags.get("asset", "Unknown"),
"exchange": run.data.tags.get("exchange", "Unknown"),
"lookback_period": run.data.tags.get("lookback_period", "Unknown"),
"project_version": run.data.tags.get("project_version", "Unknown"),
"training_date": run.data.tags.get("training_date", "Unknown"),
# Additional metadata
"model_name": run.data.tags.get("model_name", "Unknown"),
"model_type": run.data.tags.get("model_type", "Unknown"),
"timeframe": run.data.tags.get("timeframe", "Unknown"),
"bot_version": run.data.tags.get("bot_version", "Unknown"),
# Parameters and metrics
"parameters": run.data.params,
"metrics": run.data.metrics,
}

@handle_errors(default_return = None, context="mlflow_utils.validate_run_metadata")
def validate_run_metadata(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
metadata, get_enhanced_run_metadata(run_id)
if not metadata:
    passreturn False

required_fields = ["asset", "exchange", "lookback_period", "project_version", "training_date"]

for field in required_fields:
    passif metadata.get(field) in [None, "Unknown", ""]:
    passsystem_logger.warning(f"Missing required metadata field: {field} in run {run_id}")
return False

system_logger.info(f"✅ Run {run_id} has all required metadata")
return True

except Exception as e:
    passpasspasspasspasspasspasssystem_logger.error(f"Error validating run metadata for {run_id}: {e}")
return False

@handle_errors(default_return = None, context="mlflow_utils.ensure_enhanced_mlflow_run")
def ensure_enhanced_mlflow_run(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
# Extract metadata from config
metadata, extract_training_metadata(config)

# Set up MLflow
tracking_uri, config.get("mlflow", {}).get("tracking_uri") or "file:./mlruns"
exp_name, experiment_name or config.get("mlflow", {}).get("experiment_name") or "ares_trading"

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(exp_name)

# Create run name if not provided
if not run_name:
    passrun_name, f"{metadata['exchange']}_{metadata['asset']}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Start run and log metadata
with mlflow.start_run(run_name = run_name) as run:
    passrun_id, run.info.run_id

# Log enhanced training metadata
log_enhanced_training_metadata(
asset = metadata["asset"],
exchange = metadata["exchange"],
lookback_period = metadata["lookback_period"],
project_version = metadata["project_version"],
run_id = run_id,
additional_metadata={
"run_name": run_name,
"experiment_name": exp_name,
"pipeline_step": "enhanced_training_manager",
}
)

system_logger.info(f"✅ Created enhanced MLflow run {run_id} with all required metadata")
return run_id

except Exception as e:
    passpasspasspasspasspasspasspasssystem_logger.error(f"Failed to create enhanced MLflow run: {e}")
raise
