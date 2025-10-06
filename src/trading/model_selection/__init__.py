"""
Model Selection Module for Trading System

This module provides model selection capabilities for the trading system,
integrating with the DataDrivenModelSelector from the training system.

Key Components:
- ModelSelectorService: Main service for model selection
- ModelSelectionResult: Result from model selection
- TradingModelConfig: Configuration for trading model selection
"""

from .model_selector_service import (
    ModelSelectorService,
    ModelSelectionResult,
    TradingModelConfig,
    get_model_selector_service,
    select_models_for_trading
)

__all__ = [
    'ModelSelectorService',
    'ModelSelectionResult', 
    'TradingModelConfig',
    'get_model_selector_service',
    'select_models_for_trading'
]