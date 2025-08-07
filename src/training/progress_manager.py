#!/usr/bin/env python3
"""
Progress Manager for Training Steps

This module handles saving and loading progress for each training step,
allowing the training pipeline to resume from any step.
"""

import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.logger import system_logger


class ProgressManager:
    """Manages progress saving and loading for training steps."""

    def __init__(self, symbol: str, exchange: str, data_dir: str = "data/training"):
        self.symbol = symbol
        self.exchange = exchange
        self.data_dir = data_dir
        self.logger = system_logger.getChild("ProgressManager")
        
        # Create progress directory
        self.progress_dir = Path(data_dir) / "progress" / f"{exchange}_{symbol}"
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized ProgressManager for {symbol} on {exchange}")
        self.logger.info(f"Progress directory: {self.progress_dir}")

    def save_step_progress(
        self, 
        step_name: str, 
        step_data: Dict[str, Any], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save progress for a specific step.
        
        Args:
            step_name: Name of the step (e.g., 'step1_data_collection')
            step_data: Data to save for this step
            metadata: Optional metadata about the step execution
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            timestamp = datetime.now().isoformat()
            
            # Create step progress data
            progress_data = {
                "step_name": step_name,
                "symbol": self.symbol,
                "exchange": self.exchange,
                "timestamp": timestamp,
                "data": step_data,
                "metadata": metadata or {}
            }
            
            # Save as JSON for human readability
            json_file = self.progress_dir / f"{step_name}.json"
            with open(json_file, 'w') as f:
                json.dump(progress_data, f, indent=2, default=str)
            
            # Save as pickle for complex objects
            pickle_file = self.progress_dir / f"{step_name}.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump(progress_data, f)
            
            self.logger.info(f"✅ Saved progress for {step_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save progress for {step_name}: {e}")
            return False

    def load_step_progress(self, step_name: str) -> Optional[Dict[str, Any]]:
        """
        Load progress for a specific step.
        
        Args:
            step_name: Name of the step to load
            
        Returns:
            Progress data if found, None otherwise
        """
        try:
            # Try pickle file first (for complex objects)
            pickle_file = self.progress_dir / f"{step_name}.pkl"
            if pickle_file.exists():
                with open(pickle_file, 'rb') as f:
                    progress_data = pickle.load(f)
                self.logger.info(f"✅ Loaded progress for {step_name}")
                return progress_data
            
            # Fallback to JSON file
            json_file = self.progress_dir / f"{step_name}.json"
            if json_file.exists():
                with open(json_file, 'r') as f:
                    progress_data = json.load(f)
                self.logger.info(f"✅ Loaded progress for {step_name}")
                return progress_data
            
            self.logger.info(f"ℹ️  No progress found for {step_name}")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load progress for {step_name}: {e}")
            return None

    def get_latest_step(self) -> Optional[str]:
        """
        Get the name of the latest completed step.
        
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
            self.logger.error(f"❌ Failed to get latest step: {e}")
            return None

    def get_all_progress(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all saved progress data.
        
        Returns:
            Dictionary mapping step names to their progress data
        """
        progress_data = {}
        
        try:
            for pickle_file in self.progress_dir.glob("*.pkl"):
                step_name = pickle_file.stem
                progress = self.load_step_progress(step_name)
                if progress:
                    progress_data[step_name] = progress
                    
            self.logger.info(f"📋 Loaded progress for {len(progress_data)} steps")
            return progress_data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get all progress: {e}")
            return {}

    def clear_progress(self, step_name: Optional[str] = None) -> bool:
        """
        Clear progress for a specific step or all steps.
        
        Args:
            step_name: Step name to clear, or None to clear all
            
        Returns:
            True if cleared successfully, False otherwise
        """
        try:
            if step_name:
                # Clear specific step
                files_to_remove = [
                    self.progress_dir / f"{step_name}.pkl",
                    self.progress_dir / f"{step_name}.json"
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
            self.logger.error(f"❌ Failed to clear progress: {e}")
            return False

    def step_exists(self, step_name: str) -> bool:
        """
        Check if progress exists for a specific step.
        
        Args:
            step_name: Name of the step to check
            
        Returns:
            True if progress exists, False otherwise
        """
        pickle_file = self.progress_dir / f"{step_name}.pkl"
        json_file = self.progress_dir / f"{step_name}.json"
        return pickle_file.exists() or json_file.exists()

    def get_step_timestamp(self, step_name: str) -> Optional[str]:
        """
        Get the timestamp when a step was completed.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Timestamp string if found, None otherwise
        """
        progress = self.load_step_progress(step_name)
        if progress:
            return progress.get("timestamp")
        return None
