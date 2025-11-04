#!/usr/bin/env python3
"""
Configuration Centralisée pour Analyst Ensemble Training

Ce module fournit les fonctions d'interface pour accéder à la configuration centralisée
de l'entraînement d'ensemble des modèles analyst.
"""

from .config_manager import (
    AnalystEnsembleTrainingConfigManager,
    AnalystEnsembleTrainingConfig,
    get_analyst_ensemble_config_manager,
    get_analyst_ensemble_config,
    set_custom_config_path
)

__version__ = "1.0.0"
__author__ = "Ares Configuration System"
__description__ = "Configuration centralisée pour Analyst Ensemble Training"

__all__ = [
    'AnalystEnsembleTrainingConfigManager',
    'AnalystEnsembleTrainingConfig',
    'get_analyst_ensemble_config_manager',
    'get_analyst_ensemble_config',
    'set_custom_config_path'
]

# Auto-initialisation du gestionnaire global
try:
    _ = get_analyst_ensemble_config_manager()
except Exception:
    pass  # Ignorer les erreurs d'initialisation