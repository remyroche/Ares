"""Validation package for model training steps."""

# Only import step20_ab_testing to avoid pandas dependencies from other files
# Use importlib to avoid loading other modules in the directory
import importlib.util
import sys
from pathlib import Path

# Get the path to step20_ab_testing.py
step20_path = Path(__file__).parent / 'step20_ab_testing.py'

# Load the module using importlib
spec = importlib.util.spec_from_file_location('step20_ab_testing', step20_path)
step20_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step20_module)

# Make the classes available at package level
ABTestingStep = step20_module.ABTestingStep
run_step = step20_module.run_step

__all__ = ['ABTestingStep', 'run_step']