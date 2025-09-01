# src/transition/path_targets.py

from src.utils.logger import system_logger
from typing import TYPE_CHECKING, Any
import pandas as pd
from dataclasses import dataclass
import numpy as np

if TYPE_CHECKING:
    pass  # TODO: Add proper implementation
@dataclass
class PathClassConfig:
    enable_beginning_of_trend: bool
    adx_sideways_threshold: float
    return_threshold: float
    onset_window_bars: int

