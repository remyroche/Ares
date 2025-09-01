# src/training/timeframe_relevance_analyzer.py

"""
Timeframe Relevance Analyzer

This module analyzes the relevance of different timeframes for high leverage trading (10x-100x)
and optimizes the ensemble configuration accordingly.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors

