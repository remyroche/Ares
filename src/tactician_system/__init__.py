"""
Tactician System Module

This module contains the Tactician system that runs every 30 seconds on 1-minute base
timeframe data, deciding WHEN to trade when the Analyst gives a green light.
Trained on all regimes but only on periods where Analyst gives green light.
"""

from .tactician_timing_predictor import (
    TacticianConfig,
    TacticianPrediction,
    TacticianTimingPredictor,
    create_tactician_timing_predictor
)

__all__ = [
    'TacticianConfig',
    'TacticianPrediction',
    'TacticianTimingPredictor',
    'create_tactician_timing_predictor'
]