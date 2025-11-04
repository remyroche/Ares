"""
Configuration centralisée pour regime_models_training

Ce package fournit un système de configuration unifié pour l'entraînement
des modèles de détection de régime avec support YAML/JSON, validation,
et héritage de configurations.
"""

from .config_manager import (
    RegimeModelsTrainingConfigManager,
    ConfigValidationError,
    load_regime_training_config
)

from .default_config import DEFAULT_CONFIG

__version__ = "2.0.0"
__author__ = "Ares Training System"

__all__ = [
    "RegimeModelsTrainingConfigManager",
    "ConfigValidationError", 
    "load_regime_training_config",
    "DEFAULT_CONFIG"
]