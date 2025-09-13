"""
Configuration for ML Common optimization components.
"""

import os
from pathlib import Path

# Base configuration for optimization components
CONFIG = {
    'CHECKPOINT_DIR': os.path.join(os.getcwd(), 'models', 'checkpoints'),
    'LOG_DIR': os.path.join(os.getcwd(), 'logs'),
    'TEMP_DIR': os.path.join(os.getcwd(), 'temp'),
    'MAX_MEMORY_USAGE': 0.8,
    'DEFAULT_TIMEOUT': 3600,
    'PARALLEL_WORKERS': 4
}

# Ensure directories exist
for key in ['CHECKPOINT_DIR', 'LOG_DIR', 'TEMP_DIR']:
    Path(CONFIG[key]).mkdir(parents=True, exist_ok=True)
