#!/usr/bin/env python3
"""
Setup Challenger Model Utility

This script helps set up a challenger model run ID for testing the challenger mode.
It allows users to specify a challenger model run ID that will be used for challenger paper trading.

Usage:
    python scripts/setup_challenger_model.py --run-id <mlflow_run_id>
    python scripts/setup_challenger_model.py --list-models
    python scripts/setup_challenger_model.py --clear
"""

import argparse
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import mlflow

from src.utils.logger import setup_logging, system_logger
from src.utils.warning_symbols import (
    error,
    warning,
    critical,
    problem,
    failed,
    invalid,
    missing,
    timeout,
    connection_error,
    validation_error,
    initialization_error,
    execution_error,
)
from src.utils.state_manager import StateManager


def setup_challenger_model(run_id: str):
    """Set up a challenger model run ID."""
    setup_logging()
    logger = system_logger.getChild("SetupChallengerModel")

    try:
        # Initialize state manager
        state_manager = StateManager()

        # Verify the run ID exists in MLflow
        client = mlflow.tracking.MlflowClient()
        try:
            run = client.get_run(run_id)
            logger.info(f"Found MLflow run: {run_id}")
            logger.info(f"Run name: {run.data.tags.get('mlflow.runName', 'N/A')}")
            logger.info(f"Status: {run.info.status}")
        except Exception as e:
            print(error("Could not find MLflow run {run_id}: {e}")))
            return False

        # Set the challenger model run ID
        state_manager.set_state("challenger_model_run_id", run_id)
        logger.info(f"✅ Challenger model run ID set to: {run_id}")

        return True

    except Exception as e:
        print(error("Error setting up challenger model: {e}")))
        return False


def list_available_models():
    """List available models from MLflow."""
    setup_logging()
    logger = system_logger.getChild("ListModels")

    try:
        client = mlflow.tracking.MlflowClient()

        # Get the experiment name from config
        from src.config import CONFIG

        experiment_name = CONFIG.get("MLFLOW_EXPERIMENT_NAME", "ares_trading")

        # Find the experiment
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            print(missing("Experiment '{experiment_name}' not found")))
            return False

        # Search for runs
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=20,
        )

        logger.info(f"Available models in experiment '{experiment_name}':")
        logger.info("=" * 80)

        for run in runs:
            run_id = run.info.run_id
            run_name = run.data.tags.get("mlflow.runName", "N/A")
            status = run.data.tags.get("model_status", "unknown")
            accuracy = run.data.metrics.get("accuracy", 0.0)
            timestamp = run.info.start_time

            logger.info(f"Run ID: {run_id}")
            logger.info(f"Name: {run_name}")
            logger.info(f"Status: {status}")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Timestamp: {timestamp}")
            logger.info("-" * 40)

        return True

    except Exception as e:
        print(error("Error listing models: {e}")))
        return False


def clear_challenger_model():
    """Clear the challenger model run ID."""
    setup_logging()
    logger = system_logger.getChild("ClearChallengerModel")

    try:
        # Initialize state manager
        state_manager = StateManager()

        # Clear the challenger model run ID
        state_manager.set_state("challenger_model_run_id", None)
        logger.info("✅ Challenger model run ID cleared")

        return True

    except Exception as e:
        print(error("Error clearing challenger model: {e}")))
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Setup Challenger Model Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Set up a challenger model
  python scripts/setup_challenger_model.py --run-id abc123def456

  # List available models
  python scripts/setup_challenger_model.py --list-models

  # Clear challenger model
  python scripts/setup_challenger_model.py --clear
        """,
    )

    parser.add_argument("--run-id", help="MLflow run ID for the challenger model")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models from MLflow",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the challenger model run ID",
    )

    args = parser.parse_args()

    if args.list_models:
        success = list_available_models()
    elif args.clear:
        success = clear_challenger_model()
    elif args.run_id:
        success = setup_challenger_model(args.run_id)
    else:
        parser.print_help()
        sys.exit(1)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
