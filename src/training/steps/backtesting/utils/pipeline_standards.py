"""Pipeline standards for backtesting."""

from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

from typing import Any, Dict

class PipelineStandards:
    """Pipeline standards configuration."""

    def __init__(self):
        self.standards = {
            "data_quality": True,
            "validation": True,
            "logging": True
        }

# Global instance
pipeline_standards = PipelineStandards()

"""Pipeline standards for backtesting."""

from typing import Any, Dict