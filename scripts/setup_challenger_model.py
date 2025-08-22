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

from pathlib import Path
from typing import Optional
from src.utils.logger import setup_logging, system_logger
import argparse
import sys

# Ensure project root on path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import CONFIG  # noqa: E402
from src.utils.state_manager import StateManager  # noqa: E402
from src.utils.warning_symbols import error, missing  # noqa: E402

try:
	import mlflow  # type: ignore
except Exception as e:  # noqa: BLE001
	mlflow = None  # type: ignore
	system_logger.warning(f"MLflow not available: {e}")


def setup_challenger_model(run_id: str) -> bool:
	"""Set up a challenger model run ID."""
	setup_logging()
	logger = system_logger.getChild("SetupChallengerModel")

	try:
		# Initialize state manager (with default config)
		state_manager = StateManager({})

		# Verify the run ID exists in MLflow
		if mlflow is None:
			print(error("MLflow is not installed or not available"))
			return False

		client = mlflow.tracking.MlflowClient()
		try:
			run = client.get_run(run_id)
		except Exception as e:  # noqa: BLE001
			print(error(f"Could not find MLflow run {run_id}: {e}"))
			return False

		logger.info(f"Found MLflow run: {run_id}")
		logger.info(f"Run name: {run.data.tags.get('mlflow.runName', 'N/A')}")
		logger.info(f"Status: {run.info.status}")

		# Set the challenger model run ID
		state_manager.set_state("challenger_model_run_id", run_id)
		logger.info(f"✅ Challenger model run ID set to: {run_id}")

		return True
	except Exception as e:  # noqa: BLE001
		print(error(f"Error setting up challenger model: {e}"))
		return False


def list_available_models() -> bool:
	"""List available models from MLflow."""
	setup_logging()
	logger = system_logger.getChild("ListModels")

	try:
		if mlflow is None:
			print(error("MLflow is not installed or not available"))
			return False

		client = mlflow.tracking.MlflowClient()

		# Get the experiment name from config
		experiment_name = CONFIG.get("MLFLOW_EXPERIMENT_NAME", "ares_trading")

		# Find the experiment
		experiment = client.get_experiment_by_name(experiment_name)
		if not experiment:
			print(missing(f"Experiment '{experiment_name}' not found"))
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
			status = run.data.tags.get("model_status", run.info.status)
			accuracy = run.data.metrics.get("accuracy", 0.0)
			timestamp = run.info.start_time

			logger.info(f"Run ID: {run_id}")
			logger.info(f"Name: {run_name}")
			logger.info(f"Status: {status}")
			logger.info(f"Accuracy: {float(accuracy):.4f}")
			logger.info(f"Timestamp: {timestamp}")
			logger.info("-" * 40)

		return True
	except Exception as e:  # noqa: BLE001
		print(error(f"Error listing models: {e}"))
		return False


def clear_challenger_model() -> bool:
	"""Clear the challenger model run ID."""
	setup_logging()
	logger = system_logger.getChild("ClearChallengerModel")

	try:
		# Initialize state manager
		state_manager = StateManager({})

		# Clear the challenger model run ID
		state_manager.set_state("challenger_model_run_id", None)
		logger.info("✅ Challenger model run ID cleared")

		return True
	except Exception as e:  # noqa: BLE001
		print(error(f"Error clearing challenger model: {e}"))
		return False


def build_parser() -> argparse.ArgumentParser:
	"""Create and configure the argparse parser."""
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
	return parser


def main() -> None:
	"""Main entry point."""
	parser = build_parser()
	args = parser.parse_args()

	success: Optional[bool] = None
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
