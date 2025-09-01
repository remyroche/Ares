# src/training/validator.py

import time
from typing import Any

from src.training.steps.data_preparation_components.training_validation_config import (
    VALIDATION_FUNCTIONS,
    can_proceed_to_step,
    get_progression_rules,
    get_validation_config,
)


