"""
Pipeline Results Manager

This module provides a centralized system for saving pipeline results to the outcomes/ directory.
It standardizes the saving of pipeline results.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import logging

from src.utils.logger import system_logger

class PipelineResultsManager:
    """Centralized manager for saving pipeline results to outcomes/ directory."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or system_logger.getChild("PipelineResultsManager")
        self.outcomes_dir = Path("outcomes")
        self.outcomes_dir.mkdir(exist_ok=True)


    def save_generic_results(self,
                           result_data: Dict[str, Any],
                           result_type: str,
                           symbol: Optional[str] = None,
                           timeframe: Optional[str] = None,
                           additional_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save generic pipeline results to outcomes/ directory.

        Args:
            result_data: Result data to save
            result_type: Type of result (e.g., 'regime_discovery', 'clustering')
            symbol: Trading symbol (optional)
            timeframe: Timeframe (optional)
            additional_metadata: Additional metadata to include

        Returns:
            Path to saved file
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create filename with optional symbol/timeframe
            filename_parts = [result_type, "result"]
            if symbol:
                filename_parts.append(symbol)
            if timeframe:
                filename_parts.append(timeframe)
            filename_parts.append(timestamp)

            filename = f"{'_'.join(filename_parts)}.json"
            filepath = self.outcomes_dir / filename

            # Prepare result data with metadata
            full_result_data = {
                'metadata': {
                    'result_type': result_type,
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'pipeline_level': True
                },
                'result_data': result_data
            }

            # Add additional metadata if provided
            if additional_metadata:
                full_result_data['metadata'].update(additional_metadata)

            # Save to file
            with open(filepath, 'w') as f:
                json.dump(full_result_data, f, indent=2, default=str)

            self.logger.info(f"ℹ️ {result_type} results saved to {filepath}")
            return str(filepath)

        except Exception as e:
            self.logger.error(f"❌ Failed to save {result_type} results: {e}")
            raise

# Global instance for easy access
pipeline_results_manager = PipelineResultsManager()
