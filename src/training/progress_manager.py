#!/usr/bin/env python3
"""Progress Manager for Training Steps.

This module handles saving and loading progress for each training step, allowing the training pipeline to resume from any step.
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    failed)


class ProgressManager:

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="progressmanager initialization",
    )
    async def initialize(self) -> bool:
        """Initialize ProgressManager."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
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
            timestamp = datetime.now().isoformat()

            # Create step progress data
            progress_data = {
                "step_name": step_name,
                "symbol": self.symbol,
                "exchange": self.exchange,
                "timestamp": timestamp,
                "data": step_data,
                "metadata": metadata or {},
            }

            # Save as JSON for human readability
            json_file = self.progress_dir / f"{step_name}.json"

            self.logger.info(f"✅ Saved progress for {step_name}")
            return True

        except Exception as e:
            error_msg = f"Failed to save progress for {step_name}: {e}"
            self.logger.exception(error_msg)
            self.print(failed(error_msg))
            return False

    @handle_errors(
                self.logger.info(f"✅ Loaded progress for {step_name}")
                return progress_data

            # Fallback to JSON file
            json_file = self.progress_dir / f"{step_name}.json"
            if json_file.exists():
                self.logger.info(f"✅ Loaded progress for {step_name}")
                return progress_data

            self.logger.info(f"ℹ️  No progress found for {step_name}")
            return None

        except Exception as e:
            error_msg = f"Failed to load progress for {step_name}: {e}"
            self.logger.exception(error_msg)
            self.print(failed(error_msg))
            return None

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

    def get_all_progress(...) -> ...:
    """..."""
progress_data = {}

        try:
            for pickle_file in self.progress_dir.glob("*.pkl"):
step_name = pickle_file.stem
                progress = self.load_step_progress(step_name)
                if progress:

            self.logger.info(f"📋 Loaded progress for {len(progress_data)} steps")
            return progress_data

        except Exception as e:
            error_msg = f"Failed to get all progress: {e}"
            self.logger.exception(error_msg)
            self.print(failed(error_msg))
            return {}

            if step_name:
# Clear specific step
                files_to_remove = [
                    self.progress_dir / f"{step_name}.pkl",
                    self.progress_dir / f"{step_name}.json",
                ]
                for file_path in files_to_remove:
                if file_path.exists():
file_path.unlink()
                self.logger.info(f"🗑️  Cleared progress for {step_name}")
            else:
                # Clear all progress
                for file_path in self.progress_dir.glob("*"):
file_path.unlink()
                self.logger.info("🗑️  Cleared all progress")

            return True

        except Exception as e:
            error_msg = f"Failed to clear progress: {e}"
            self.logger.exception(error_msg)
            self.print(failed(error_msg))
            return False

        return None
