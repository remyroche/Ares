"""Core module - Classes principales de l'architecture refactorisée."""

from .trainer import RegimeModelsTrainingComponent
from .configuration_manager import ConfigurationManager
from .model_factory import ModelFactory
from .ensemble_builder import EnsembleBuilder

__all__ = [
    'RegimeModelsTrainingComponent',
    'ConfigurationManager',
    'ModelFactory', 
    'EnsembleBuilder'
]