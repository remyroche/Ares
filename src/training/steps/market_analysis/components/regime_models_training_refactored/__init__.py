"""
Regime Models Training - Architecture Refactorisée

Nouvelle architecture modulaire pour remplacer le God Object de 3617 lignes.
Séparation des responsabilités en modules spécialisés.
"""

from .core.trainer import RegimeModelsTrainingComponent
from .core.configuration_manager import ConfigurationManager
from .core.model_factory import ModelFactory
from .core.ensemble_builder import EnsembleBuilder

__all__ = [
    'RegimeModelsTrainingComponent',
    'ConfigurationManager', 
    'ModelFactory',
    'EnsembleBuilder'
]