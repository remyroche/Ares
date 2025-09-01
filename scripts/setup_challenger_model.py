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
