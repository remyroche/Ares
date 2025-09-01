# Enhanced MLflow Step Integration Template
"""
Template showing how to integrate enhanced MLflow logging into any pipeline step.

This template demonstrates the pattern for ensuring all models and artifacts
are properly associated with the required metadata (asset, exchange, lookback_period,
project_version, and date) throughout the enhanced_training_manager pipeline.
"""

import asyncio
from typing import Any, Dict
from pathlib import Path

from src.utils.enhanced_mlflow_integration import (
    with_enhanced_mlflow_logging,
    log_step_artifact,
    log_step_dataframe,
    log_step_model,
    log_step_metrics,
    log_step_dataframe_with_standardized_name,
    log_step_artifact_with_standardized_name,
    log_step_report,
    generate_standardized_artifact_name,
    EnhancedMLflowManager,
)


class ExampleStep:
    """Example step showing enhanced MLflow integration pattern."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger

    @with_enhanced_mlflow_logging("example_step")
    async def execute(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute the step with automatic enhanced MLflow logging.

        The @with_enhanced_mlflow_logging decorator automatically:
        - Creates an MLflow run for this step
        - Logs enhanced training metadata (asset, exchange, lookback_period, project_version, date)
        - Logs step parameters
        - Logs step execution metrics
        - Logs step completion metadata
        """

        # Step execution logic here
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        timeframe = training_input.get("timeframe", "1m")

        # Example: Process data and create artifacts
        processed_data = await self._process_data(training_input)

        # Example: Train a model
        trained_model = await self._train_model(processed_data)

        # Example: Calculate metrics
        metrics = await self._calculate_metrics(trained_model, processed_data)

        # Example: Save artifacts
        artifacts = await self._save_artifacts(processed_data, trained_model, symbol, exchange, timeframe)

        # Log step-specific artifacts to MLflow with standardized naming
        await self._log_step_artifacts_to_mlflow(
            processed_data, trained_model, metrics, artifacts, training_input
        )

        # Return results
        return {
            "status": "SUCCESS",
            "processed_data": processed_data,
            "trained_model": trained_model,
            "metrics": metrics,
            "artifacts": artifacts,
        }

    async def _log_step_artifacts_to_mlflow(
        self,
        processed_data: Any,
        trained_model: Any,
        metrics: Dict[str, float],
        artifacts: Dict[str, str],
        training_input: Dict[str, Any]
    ) -> None:
        """Log step artifacts to MLflow with enhanced metadata and standardized naming."""
        try:
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            timeframe = training_input.get("timeframe", "1m")

            # Log processed data as DataFrame with standardized naming
            if hasattr(processed_data, 'to_parquet'):
                artifact_name = log_step_dataframe_with_standardized_name(
                    config=self.config,
                    step_name="example_step",
                    df=processed_data,
                    artifact_type="processed_data",
                    additional_metadata={
                        "artifact_type": "processed_data",
                        "dataframe_shape": list(processed_data.shape),
                        "processing_method": "example_processing",
                        "timeframe": timeframe,
                    }
                )
                self.logger.info(f"✅ Logged processed data: {artifact_name}")

            # Log trained model
            if trained_model:
                log_step_model(
                    config=self.config,
                    step_name="example_step",
                    model=trained_model,
                    model_name="example_model",
                    model_type="example",
                    additional_metadata={
                        "training_algorithm": getattr(trained_model, '__class__.__name__', 'Unknown'),
                        "model_parameters": getattr(trained_model, 'get_params', lambda: {})() if hasattr(trained_model, 'get_params') else {},
                        "timeframe": timeframe,
                    }
                )

            # Log step report
            report_data = {
                "step_execution_summary": {
                    "processed_data_shape": list(processed_data.shape) if hasattr(processed_data, 'shape') else [],
                    "model_trained": trained_model is not None,
                    "metrics_calculated": len(metrics),
                    "artifacts_generated": len(artifacts),
                },
                "metrics": metrics,
                "artifacts": artifacts,
                "training_input": training_input,
                "execution_timestamp": datetime.now().isoformat(),
            }

            report_name = log_step_report(
                config=self.config,
                step_name="example_step",
                report_data=report_data,
                report_type="example_step_report",
                additional_metadata={
                    "data_processed": hasattr(processed_data, 'shape'),
                    "model_trained": trained_model is not None,
                    "timeframe": timeframe,
                }
            )
            self.logger.info(f"✅ Logged step report: {report_name}")

            # Log metrics
            if metrics:
                log_step_metrics(
                    config=self.config,
                    step_name="example_step",
                    metrics=metrics,
                    additional_metadata={
                        "metrics_type": "example_performance",
                        "validation_method": "cross_validation",
                        "timeframe": timeframe,
                    }
                )

            # Log artifact files with standardized naming
            for artifact_name, artifact_path in artifacts.items():
                if Path(artifact_path).exists():
                    artifact_file_name = log_step_artifact_with_standardized_name(
                        config=self.config,
                        step_name="example_step",
                        artifact_path=artifact_path,
                        artifact_type=artifact_name,
                        additional_metadata={
                            "artifact_filename": Path(artifact_path).name,
                            "artifact_size_bytes": Path(artifact_path).stat().st_size,
                            "timeframe": timeframe,
                        }
                    )
                    self.logger.info(f"✅ Logged artifact {artifact_name}: {artifact_file_name}")

            self.logger.info("✅ Example step artifacts logged to MLflow with standardized naming successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to log example step artifacts to MLflow: {e}")
            # Don't fail the step if MLflow logging fails

    async def _process_data(self, training_input: Dict[str, Any]) -> Any:
        """Example data processing method."""
        # Data processing logic here
        return pd.DataFrame()  # Example return

    async def _train_model(self, data: Any) -> Any:
        """Example model training method."""
        # Model training logic here
        return None  # Example return

    async def _calculate_metrics(self, model: Any, data: Any) -> Dict[str, float]:
        """Example metrics calculation method."""
        # Metrics calculation logic here
        return {"accuracy": 0.85, "precision": 0.82}  # Example return

    async def _save_artifacts(
        self,
        data: Any,
        model: Any,
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> Dict[str, str]:
        """Example artifact saving method."""
        # Artifact saving logic here
        return {
            "processed_data": f"data/processed/{exchange}_{symbol}_{timeframe}_data.parquet",
            "trained_model": f"models/{exchange}_{symbol}_{timeframe}_model.pkl",
            "metrics": f"metrics/{exchange}_{symbol}_{timeframe}_metrics.json",
        }


# Alternative: Using the EnhancedMLflowManager directly
class ExampleStepWithManager:
    """Example step using EnhancedMLflowManager directly."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mlflow_manager = EnhancedMLflowManager(config)
        self.logger = system_logger

    async def execute(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the step using EnhancedMLflowManager directly."""

        # Start MLflow run
        run_id = self.mlflow_manager.start_run(step_name="example_step_with_manager")

        try:
            # Step execution logic here
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            timeframe = training_input.get("timeframe", "1m")

            # Example: Process data and create artifacts
            processed_data = await self._process_data(training_input)

            # Example: Train a model
            trained_model = await self._train_model(processed_data)

            # Example: Calculate metrics
            metrics = await self._calculate_metrics(trained_model, processed_data)

            # Log artifacts using the manager
            self.mlflow_manager.log_dataframe(
                df=processed_data,
                artifact_path=f"artifacts/example_step/{exchange}_{symbol}_{timeframe}_processed_data.parquet",
                additional_metadata={
                    "artifact_type": "processed_data",
                    "processing_method": "example_processing",
                }
            )

            self.mlflow_manager.log_model(
                model=trained_model,
                model_name="example_model",
                model_type="example",
                additional_metadata={
                    "training_algorithm": getattr(trained_model, '__class__.__name__', 'Unknown'),
                }
            )

            self.mlflow_manager.log_metrics(
                metrics=metrics,
                additional_metadata={
                    "metrics_type": "example_performance",
                }
            )

            # Validate the run has all required metadata
            is_valid = self.mlflow_manager.validate_current_run()
            if is_valid:
                self.logger.info("✅ MLflow run validation passed")
            else:
                self.logger.warning("⚠️ MLflow run validation failed")

            return {
                "status": "SUCCESS",
                "processed_data": processed_data,
                "trained_model": trained_model,
                "metrics": metrics,
                "mlflow_run_id": run_id,
            }

        finally:
            # End the MLflow run
            self.mlflow_manager.end_run()

    async def _process_data(self, training_input: Dict[str, Any]) -> Any:
        """Example data processing method."""
        # Data processing logic here
        return pd.DataFrame()  # Example return

    async def _train_model(self, data: Any) -> Any:
        """Example model training method."""
        # Model training logic here
        return None  # Example return

    async def _calculate_metrics(self, model: Any, data: Any) -> Dict[str, float]:
        """Example metrics calculation method."""
        # Metrics calculation logic here
        return {"accuracy": 0.85, "precision": 0.82}  # Example return


# Function-based step example
@with_enhanced_mlflow_logging("example_function_step")
async def example_function_step(
    symbol: str,
    exchange: str,
    timeframe: str = "1m",
    data_dir: str = "data_cache",
    **kwargs: Any
) -> bool:
    """
    Example function-based step with enhanced MLflow logging.

    The @with_enhanced_mlflow_logging decorator automatically handles:
    - MLflow run creation and management
    - Enhanced metadata logging
    - Step execution tracking
    """

    # Step execution logic here
    processed_data = await _process_data_function(symbol, exchange, timeframe, data_dir)

    # Train model
    trained_model = await _train_model_function(processed_data)

    # Calculate metrics
    metrics = await _calculate_metrics_function(trained_model, processed_data)

    # Save artifacts
    artifacts = await _save_artifacts_function(processed_data, trained_model, symbol, exchange, timeframe)

    # Log artifacts to MLflow
    await _log_function_artifacts_to_mlflow(
        processed_data, trained_model, metrics, artifacts, symbol, exchange, timeframe
    )

    return True


async def _log_function_artifacts_to_mlflow(
    processed_data: Any,
    trained_model: Any,
    metrics: Dict[str, float],
    artifacts: Dict[str, str],
    symbol: str,
    exchange: str,
    timeframe: str
) -> None:
    """Log function step artifacts to MLflow."""
    try:
        # Create config for MLflow logging
        config = {
            "trading_symbol": symbol,
            "exchange_name": exchange,
            "lookback_years": 2,  # Default value
        }

        # Log processed data
        if hasattr(processed_data, 'to_parquet'):
            log_step_dataframe(
                config=config,
                step_name="example_function_step",
                df=processed_data,
                artifact_name=f"{exchange}_{symbol}_{timeframe}_processed_data",
                additional_metadata={
                    "artifact_type": "processed_data",
                    "processing_method": "function_processing",
                }
            )

        # Log model
        if trained_model:
            log_step_model(
                config=config,
                step_name="example_function_step",
                model=trained_model,
                model_name="function_model",
                model_type="function_example",
                additional_metadata={
                    "training_algorithm": getattr(trained_model, '__class__.__name__', 'Unknown'),
                }
            )

        # Log metrics
        if metrics:
            log_step_metrics(
                config=config,
                step_name="example_function_step",
                metrics=metrics,
                additional_metadata={
                    "metrics_type": "function_performance",
                }
            )

        system_logger.info("✅ Function step artifacts logged to MLflow successfully")

    except Exception as e:
        system_logger.error(f"❌ Failed to log function step artifacts to MLflow: {e}")


async def _process_data_function(symbol: str, exchange: str, timeframe: str, data_dir: str) -> Any:
    """Example data processing function."""
    # Data processing logic here
    return pd.DataFrame()  # Example return


async def _train_model_function(data: Any) -> Any:
    """Example model training function."""
    # Model training logic here
    return None  # Example return


async def _calculate_metrics_function(model: Any, data: Any) -> Dict[str, float]:
    """Example metrics calculation function."""
    # Metrics calculation logic here
    return {"accuracy": 0.85, "precision": 0.82}  # Example return


async def _save_artifacts_function(
    data: Any,
    model: Any,
    symbol: str,
    exchange: str,
    timeframe: str
) -> Dict[str, str]:
    """Example artifact saving function."""
    # Artifact saving logic here
    return {
        "processed_data": f"data/processed/{exchange}_{symbol}_{timeframe}_data.parquet",
        "trained_model": f"models/{exchange}_{symbol}_{timeframe}_model.pkl",
        "metrics": f"metrics/{exchange}_{symbol}_{timeframe}_metrics.json",
    }


# Usage examples:

# 1. Class-based step with decorator
async def example_class_step_usage():
    config = {
        "trading_symbol": "ETHUSDT",
        "exchange_name": "BINANCE",
        "lookback_years": 2,
    }

    step = ExampleStep(config)
    training_input = {
        "symbol": "ETHUSDT",
        "exchange": "BINANCE",
        "timeframe": "1m",
    }
    pipeline_state = {}

    result = await step.execute(training_input, pipeline_state)
    return result


# 2. Class-based step with manager
async def example_manager_step_usage():
    config = {
        "trading_symbol": "ETHUSDT",
        "exchange_name": "BINANCE",
        "lookback_years": 2,
    }

    step = ExampleStepWithManager(config)
    training_input = {
        "symbol": "ETHUSDT",
        "exchange": "BINANCE",
        "timeframe": "1m",
    }
    pipeline_state = {}

    result = await step.execute(training_input, pipeline_state)
    return result


# 3. Function-based step
async def example_function_step_usage():
    result = await example_function_step(
        symbol="ETHUSDT",
        exchange="BINANCE",
        timeframe="1m",
        data_dir="data_cache"
    )
    return result


if __name__ == "__main__":
    # Run examples
    asyncio.run(example_class_step_usage())
    asyncio.run(example_manager_step_usage())
    asyncio.run(example_function_step_usage())