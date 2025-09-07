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
        return pd.read_parquet(self.path)
    
    def save(self, data: pd.DataFrame) -> None:
        """Save parquet dataset."""
        data.to_parquet(self.path)

"""Domain classes for validation."""

import pandas as pd