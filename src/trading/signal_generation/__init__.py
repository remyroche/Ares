"""
Signal Generation Module

Implements proper data flow: HMM regime -> analyst -> tactician
with sequential model calls and confidence score optimization.

New Features:
- Analyst Signal Generator: Integrates Analyst component for signal generation
- Tactician Signal Generator: Integrates Tactician component for timing signals
"""

from .signal_pipeline import (
    SignalGenerationPipeline,
    RegimeOutput,
    AnalystOutput,
    SpecialistOutput,
    SignalGenerationResult,
    setup_signal_generation_pipeline
)
from .signal_combiner import SignalCombiner

# Import new signal generators
from .analyst_signals import (
    AnalystSignalGenerator, AnalystSignal, SignalType, SignalStrength,
    create_analyst_signal_generator, generate_analyst_signal
)

from .tactician_signals import (
    TacticianSignalGenerator, TacticianSignal, TimingSignal, TimingConfidence,
    create_tactician_signal_generator, generate_tactician_signal
)

__all__ = [
    "SignalGenerationPipeline",
    "RegimeOutput",
    "AnalystOutput",
    "SpecialistOutput",
    "SignalGenerationResult",
    "setup_signal_generation_pipeline",
    "SignalCombiner",
    "AnalystSignalGenerator",
    "AnalystSignal",
    "SignalType",
    "SignalStrength",
    "create_analyst_signal_generator",
    "generate_analyst_signal",
    "TacticianSignalGenerator",
    "TacticianSignal",
    "TimingSignal",
    "TimingConfidence",
    "create_tactician_signal_generator",
    "generate_tactician_signal"
]
