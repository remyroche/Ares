#!/usr/bin/env python3
"""Explainability package for ML model explanations and trade decision tracing.

This package provides comprehensive SHAP/LIME explanations for all ML models
in the trading system and enables traceability of trade decisions back to
individual factors.
"""

from .explainability.base_explainer import (
    BaseExplainer,
    ExplanationResult,
    TradeDecisionTrace,
    TradeDecisionTracer
)

from .explainability.integration import (
    ExplainabilityIntegration,
    get_explainability_integration,
    explainable_tactician_prediction,
    explainable_hmm_prediction,
    explainable_sr_prediction,
    explainable_analyst_prediction,
    explainable_trading_decision,
    FeatureExtractor,
    ExplanationVisualizer,
    DecisionTraceVisualizer
)

__all__ = [
    # Base classes
    'BaseExplainer',
    'ExplanationResult',
    'TradeDecisionTrace',
    'TradeDecisionTracer',

    # Model explainers
    'TacticianExplainer',
    'HMMExplainer',
    'SRExplainer',
    'AnalystExplainer',

    # Orchestrator
    'ExplainabilityOrchestrator',

    # Integration
    'ExplainabilityIntegration',
    'get_explainability_integration',
    'explainable_tactician_prediction',
    'explainable_hmm_prediction',
    'explainable_sr_prediction',
    'explainable_analyst_prediction',
    'explainable_trading_decision',
    'FeatureExtractor',

    # Visualization
    'ExplanationVisualizer',
    'DecisionTraceVisualizer'
]

__version__ = "1.0.0"
__author__ = "Trading System Team"
__description__ = "Comprehensive explainability framework for ML trading models"
