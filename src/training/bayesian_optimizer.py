# src/training/bayesian_optimizer.py

from collections.abc import Callable
from typing import Any, Number

import numpy as np
import optuna
import pandas as pd

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger


