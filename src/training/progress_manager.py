#!/usr/bin/env python3
"""Progress Manager for Training Steps.

This module handles saving and loading progress for each training step,
allowing the training pipeline to resume from any step.
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    failed,
)


class ProgressManager:
    """Manages progress saving and loading for training steps."""

    def __init__(self, symbol: str, exchange: str, data_dir: str = "data/training") -> None:
        self.symbol = symbol
        self.exchange = exchange
        self.data_dir = data_dir
        self.logger = system_logger.getChild("ProgressManager")

        # Create progress directory
        self.progress_dir = Path(data_dir) / "progress" / f"{exchange}_{symbol}"
        self.progress_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Initialized ProgressManager for {symbol} on {exchange}")
        self.logger.info(f"Progress directory: {self.progress_dir}")

    @handle_errors(
        exceptions=(ValueError, RuntimeError, OSError),
        default_return=False,
        context="step progress saving",
    )
    @handle_errors(
        exceptions=(ValueError, RuntimeError, OSError, pickle.UnpicklingError),
        default_return=None,
        context="step progress loading",
    )
    def load_step_progress(self, step_name: str) -> dict[str, Any] | None:
        """Load progress for a specific step.

        Args:
            step_name: Name of the step to load

        Returns:
            Progress data if found, None otherwise

        """
        try:
            # Try pickle file first (for complex objects)
            pickle_file = self.progress_dir / f"{step_name}.pkl"
            if pickle_file.exists():
                with open(pickle_file, "rb") as f:
                    progress_data = pickle.load(f)
                self.logger.info(f"✅ Loaded progress for {step_name}")
                return progress_data

            # Fallback to JSON file
            json_file = self.progress_dir / f"{step_name}.json"
            if json_file.exists():
                with open(json_file) as f:
                    progress_data = json.load(f)
                self.logger.info(f"✅ Loaded progress for {step_name}")
                return progress_data

            self.logger.info(f"ℹ️  No progress found for {step_name}")
            return None

        except Exception as e:
            error_msg = f"Failed to load progress for {step_name}: {e}"
            self.logger.exception(error_msg)
            self.print(failed(error_msg))
            return None

    def get_latest_step(self) -> str | None:
        """Get the name of the latest completed step.

        Returns:
            Name of the latest step, or None if no progress found

        """
        try:
            step_files = list(self.progress_dir.glob("*.pkl"))
            if not step_files:
                return None

            # Sort by modification time to find the latest
            latest_file = max(step_files, key=lambda f: f.stat().st_mtime)
            step_name = latest_file.stem  # Remove .pkl extension

            self.logger.info(f"📋 Latest completed step: {step_name}")
            return step_name

        except Exception as e:
            error_msg = f"Failed to get latest step: {e}"
            self.logger.exception(error_msg)
            self.print(failed(error_msg))
            return None
