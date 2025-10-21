"""
Hardware optimization constants to avoid circular imports.
"""

from enum import Enum

class OptimizationLevel(Enum):
    """Optimization levels."""
    MINIMAL = "minimal"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"

class WorkloadType(Enum):
    """Workload types for hardware optimization."""
    MATRIX_OPERATIONS = "matrix_operations"
    BACKTESTING = "backtesting"
    MONTE_CARLO = "monte_carlo"
    ML_TRAINING = "ml_training"
    DATA_PROCESSING = "data_processing"
    FEATURE_ENGINEERING = "feature_engineering"
    GENERAL = "general"
    MODERATE = "moderate"  # Added missing MODERATE value