from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
"""Domain classes for validation."""

from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

import pandas as pd
from typing import Any, Dict, Optional

class ParquetDatasetManager:
    """Manager for parquet datasets."""
    @log_important_calls
    
    def __init__(self, path: str):
        self.path = path
    
    def load(self) -> pd.DataFrame:
        """Load parquet dataset."""
        return standardized_parquet_handler.read_parquet_standardized(self.path)
    
    def save(self, data: pd.DataFrame) -> None:
        """Save parquet dataset."""
        standardized_parquet_handler.write_parquet_standardized(data, self.path)

"""Domain classes for validation."""

import pandas as pd
import json