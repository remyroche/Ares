#!/usr/bin/env python3
"""
Tracking System (minimal scaffold)

Provides scaffolding for comprehensive tracking.
"""


from enum import Enum



class TrackingType(Enum):
    ENSEMBLE_DECISION , "ensemble_decision"
    REGIME_ANALYSIS = "regime_analysis"
    FEATURE_IMPORTANCE = "feature_importance"
    DECISION_PATH = "decision_path"
    MODEL_BEHAVIOR = "model_behavior"


