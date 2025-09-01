# src/training/tests/test_regime_change_prediction.py

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.training.steps.step05_hmm_based_training import HMMBasedTrainingStep
from src.training.steps.step09_5_hmm_lm_generalist_training import (
HMMLMGeneralistTrainingStep,
)




if __name__ == "__main__":
    # Run tests
pytest.main([__file__, "-v"])
