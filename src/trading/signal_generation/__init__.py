"""
Signal Generation Module

Implements proper data flow: HMM regime -> analyst -> tactician
with sequential model calls and confidence score optimization.
"""

from .signal_pipeline import (
    SignalGenerationPipeline,
    HMMRegimeOutput,
    AnalystBaseOutput,
    AnalystMetaOutput,
    TacticianBaseOutput,
    TacticianMetaOutput,
    SignalGenerationResult,
    setup_signal_generation_pipeline
)
from .signal_combiner import SignalCombiner

__all__ = [
    "SignalGenerationPipeline",
    "HMMRegimeOutput",
    "AnalystBaseOutput", 
    "AnalystMetaOutput",
    "TacticianBaseOutput",
    "TacticianMetaOutput",
    "SignalGenerationResult",
    "setup_signal_generation_pipeline",
    "SignalCombiner"
]