"""
Configuration Centralisée pour Regime Ensemble Training Component

Ce module fournit une configuration centralisée, flexible et maintenable
pour l'entraînement des modèles d'ensemble (meta-learners) pour la détection
de régimes.

Fonctionnalités :
- Support multi-format (YAML, JSON, Python)
- Validation automatique avec feedback détaillé
- Système de fallback intelligent (custom → défaut → hardcodé)
- Configuration inheritance et override
- Paramètres par catégorie (hardware, hpo, ensemble, validation, etc.)
- Zero-downtime configuration updates
"""

from .config_manager import (
    RegimeEnsembleTrainingConfig,
    RegimeEnsembleTrainingConfigManager,
    get_regime_ensemble_config_manager,
    get_regime_ensemble_config
)

# Version du module de configuration
__version__ = "2.0.0"
__author__ = "Ares Development Team"
__description__ = "Configuration centralisée pour regime_ensemble_training"

# Export des classes principales
__all__ = [
    'RegimeEnsembleTrainingConfig',
    'RegimeEnsembleTrainingConfigManager', 
    'get_regime_ensemble_config_manager',
    'get_regime_ensemble_config'
]

# Informations du module
MODULE_INFO = {
    'version': __version__,
    'author': __author__,
    'description': __description__,
    'component': 'regime_ensemble_training',
    'created_at': '2025-11-03T21:46:25.610Z',
    'features': [
        'Configuration centralisée multi-format',
        'Validation automatique',
        'Système de fallback intelligent',
        'Support inheritance et override',
        'Zero-downtime updates'
    ]
}