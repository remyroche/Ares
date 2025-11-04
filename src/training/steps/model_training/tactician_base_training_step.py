"""
Tactician Base Training Step - Version avec Configuration Centralisée.

Cette étape entraîne les modèles tactician de base avec intégration complète
du système de configuration centralisée YAML/JSON/Python.

Version: 2.0.0
Date: 2025-11-03T22:30:00.000Z
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import os
import sys

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint

# Ajouter le répertoire config au path pour les imports
config_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'tactician_base_training')
if config_dir not in sys.path:
    sys.path.insert(0, config_dir)

# Import de la configuration centralisée
try:
    from src.config.tactician_base_training.config_manager import (
        get_tactician_base_training_config_manager,
        get_tactician_base_training_config,
        set_tactician_base_training_custom_config_path
    )
    from src.config.tactician_base_training.default_config import TacticianBaseTrainingConfig
    CENTRALIZED_CONFIG_AVAILABLE = True
    tprint("✅ [TACTICIAN_BASE] Configuration centralisée importée", "SUCCESS")
except ImportError as e:
    CENTRALIZED_CONFIG_AVAILABLE = False
    tprint(f"⚠️ [TACTICIAN_BASE] Configuration centralisée non disponible: {e}", "WARNING")
    # Définir une classe vide pour éviter les erreurs NameError
    class TacticianBaseTrainingConfig:
        pass
    # Créer des fonctions factices
    def get_tactician_base_training_config_manager():
        return None
    def get_tactician_base_training_config():
        return None
    def set_tactician_base_training_custom_config_path(path):
        return False

logger = logging.getLogger(__name__)


class TacticianBaseTrainingStep(BaseStep):
    """
    Tactician Base Training Step avec Configuration Centralisée.

    Entraîne les modèles tactician de base avec intégration complète
    du système de configuration centralisée pour une gestion robuste
    et flexible des paramètres.
    """

    def __init__(self, step_name: str = "tactician_base_training"):
        """Initialize the tactician base training step."""
        super().__init__(step_name)
        self.logger = system_logger.getChild('TacticianBaseTraining')
        
        # Configuration centralisée
        self._centralized_config = None
        self.config_manager = None
        
        # Initialiser la configuration centralisée
        if CENTRALIZED_CONFIG_AVAILABLE:
            try:
                self.config_manager = get_tactician_base_training_config_manager()
                self._centralized_config = self.config_manager.load_config()
                
                if self._centralized_config:
                    tprint(f"✅ [TACTICIAN_BASE] Configuration centralisée chargée: {self._centralized_config.tactician_config.model_name}", "SUCCESS")
                else:
                    tprint("⚠️ [TACTICIAN_BASE] Fallback vers configuration locale", "WARNING")
            except Exception as e:
                tprint(f"⚠️ [TACTICIAN_BASE] Erreur de configuration centralisée: {e}", "WARNING")
        
        # Configuration legacy de fallback
        self.legacy_config = {
            'training_type': 'tactician_base',
            'execution_context': 'tactician'
        }

    def get_centralized_config(self) -> Optional[TacticianBaseTrainingConfig]:
        """
        Obtenir la configuration centralisée.
        
        Returns:
            Configuration centralisée ou None
        """
        return self._centralized_config

    def get_parameter_with_fallback(self, parameter_path: str, default_value: Any = None) -> Any:
        """
        Obtenir un paramètre avec système de fallback.
        
        Args:
            parameter_path: Chemin vers le paramètre (ex: 'performance.expected_accuracy')
            default_value: Valeur par défaut si le paramètre n'est pas trouvé
            
        Returns:
            Valeur du paramètre ou valeur par défaut
        """
        # Essayer la configuration centralisée
        if self._centralized_config:
            try:
                path_parts = parameter_path.split('.')
                current = self._centralized_config.to_dict()
                
                for part in path_parts:
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        break
                else:
                    # Tous les éléments du chemin ont été trouvés
                    return current
            except Exception as e:
                self.logger.warning(f"Erreur d'accès paramètre centralisé {parameter_path}: {e}")
        
        # Fallback vers la configuration locale
        current = self.legacy_config
        path_parts = parameter_path.split('.')
        
        for part in path_parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default_value
        
        return current if current is not None else default_value

    def get_tactician_models_config(self) -> Dict[str, Any]:
        """
        Obtenir la configuration des modèles tactician.
        
        Returns:
            Configuration des modèles
        """
        if self._centralized_config:
            return {
                'model_name': self._centralized_config.tactician_config.model_name,
                'model_type': self._centralized_config.tactician_config.model_type,
                'target': self._centralized_config.tactician_config.target,
                'base_timeframe': self._centralized_config.tactician_config.base_timeframe,
                'execution_timeframe': self._centralized_config.tactician_config.execution_timeframe,
                'execution_frequency': self._centralized_config.tactician_config.execution_frequency,
                'price_change_target': self._centralized_config.tactician_config.price_change_target,
                'base_models': [model.to_dict() if hasattr(model, 'to_dict') else model for model in self._centralized_config.tactician_config.base_models]
            }
        
        # Configuration fallback
        return {
            'model_name': 'tactician_base_fallback',
            'model_type': 'separate_models',
            'target': 'entry_timing',
            'base_timeframe': '15m',
            'execution_timeframe': '15m',
            'execution_frequency': '3m',
            'price_change_target': 0.005,
            'base_models': []
        }

    def get_feature_engineering_config(self) -> Dict[str, Any]:
        """
        Obtenir la configuration d'ingénierie des features.
        
        Returns:
            Configuration d'ingénierie des features
        """
        if self._centralized_config:
            fe_config = self._centralized_config.feature_engineering
            return {
                'primary_features': {
                    'source': fe_config.primary_features.source,
                    'artifact_name': fe_config.primary_features.artifact_name,
                    'initial_count': fe_config.primary_features.initial_count,
                    'target_count': fe_config.primary_features.target_count
                },
                'cross_timeframe': {
                    'enable': fe_config.cross_timeframe.enable,
                    'base_timeframe': fe_config.cross_timeframe.base_timeframe,
                    'target_timeframes': fe_config.cross_timeframe.target_timeframes,
                    'feature_types': fe_config.cross_timeframe.feature_types,
                    'optimized_lookback': fe_config.cross_timeframe.optimized_lookback
                },
                'regime_features': {
                    'enable': fe_config.regime_features.enable,
                    'source': fe_config.regime_features.source,
                    'feature_names': fe_config.regime_features.feature_names,
                    'include_regime_outputs': fe_config.regime_features.include_regime_outputs
                },
                'feature_selection': {
                    'method': fe_config.feature_selection.method,
                    'alpha': fe_config.feature_selection.alpha,
                    'max_features': fe_config.feature_selection.max_features,
                    'enable_recursive_elimination': fe_config.feature_selection.enable_recursive_elimination,
                    'enable_feature_importance': fe_config.feature_selection.enable_feature_importance
                }
            }
        
        # Configuration fallback
        return {
            'primary_features': {
                'source': 'feature_generation_final_feature_selection_step',
                'artifact_name': 'tactician_features',
                'initial_count': 100,
                'target_count': 50
            },
            'cross_timeframe': {'enable': True},
            'regime_features': {'enable': True},
            'feature_selection': {
                'method': 'lasso',
                'alpha': 0.01,
                'max_features': 50
            }
        }

    def get_training_config(self) -> Dict[str, Any]:
        """
        Obtenir la configuration d'entraînement.
        
        Returns:
            Configuration d'entraînement
        """
        if self._centralized_config:
            training_config = self._centralized_config.training
            return {
                'enable_cross_validation': training_config.enable_cross_validation,
                'cv_folds': training_config.cv_folds,
                'enable_early_stopping': training_config.enable_early_stopping,
                'early_stopping_patience': training_config.early_stopping_patience,
                'validation_split': training_config.validation_split,
                'test_split': training_config.test_split
            }
        
        # Configuration fallback
        return {
            'enable_cross_validation': True,
            'cv_folds': 3,
            'enable_early_stopping': True,
            'early_stopping_patience': 10,
            'validation_split': 0.2,
            'test_split': 0.1
        }

    def get_hardware_config(self) -> Dict[str, Any]:
        """
        Obtenir la configuration hardware.
        
        Returns:
            Configuration hardware
        """
        if self._centralized_config:
            hardware_config = self._centralized_config.hardware
            return {
                'enable_gpu_acceleration': hardware_config.enable_gpu_acceleration,
                'enable_memory_optimization': hardware_config.enable_memory_optimization,
                'enable_parallel_processing': hardware_config.enable_parallel_processing,
                'memory_limit_gb': hardware_config.memory_limit_gb,
                'max_workers': hardware_config.max_workers
            }
        
        # Configuration fallback
        return {
            'enable_gpu_acceleration': False,
            'enable_memory_optimization': True,
            'enable_parallel_processing': True,
            'memory_limit_gb': 2.0,
            'max_workers': None
        }

    def get_performance_targets(self) -> Dict[str, Any]:
        """
        Obtenir les cibles de performance.
        
        Returns:
            Cibles de performance
        """
        if self._centralized_config:
            performance_config = self._centralized_config.performance
            return {
                'expected_accuracy': performance_config.expected_accuracy,
                'expected_sharpe_ratio': performance_config.expected_sharpe_ratio,
                'training_time_limit': performance_config.training_time_limit,
                'memory_limit_mb': performance_config.memory_limit_mb
            }
        
        # Configuration fallback
        return {
            'expected_accuracy': 0.80,
            'expected_sharpe_ratio': 1.40,
            'training_time_limit': 300,
            'memory_limit_mb': 2048
        }

    def get_output_config(self) -> Dict[str, Any]:
        """
        Obtenir la configuration de sortie.
        
        Returns:
            Configuration de sortie
        """
        if self._centralized_config:
            output_config = self._centralized_config.output
            return {
                'save_models': output_config.save_models,
                'save_predictions': output_config.save_predictions,
                'generate_reports': output_config.generate_reports,
                'output_dir': output_config.output_dir
            }
        
        # Configuration fallback
        return {
            'save_models': True,
            'save_predictions': True,
            'generate_reports': True,
            'output_dir': './tactician_base_models_fallback'
        }

    def get_training_summary(self) -> Dict[str, Any]:
        """
        Obtenir un résumé de la configuration d'entraînement.
        
        Returns:
            Résumé de la configuration
        """
        config_source = "centralized" if self._centralized_config else "fallback"
        
        return {
            'component_name': 'tactician_base_training',
            'version': '2.0.0',
            'config_source': config_source,
            'centralized_config': {
                'enabled': self._centralized_config is not None,
                'version': self._centralized_config.version if self._centralized_config else None,
                'models_count': len(self._centralized_config.tactician_config.base_models) if self._centralized_config else 0
            },
            'config_sections': {
                'tactician': self.get_tactician_models_config(),
                'features': self.get_feature_engineering_config(),
                'training': self.get_training_config(),
                'hardware': self.get_hardware_config(),
                'performance': self.get_performance_targets(),
                'output': self.get_output_config()
            }
        }

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute tactician base model training avec configuration centralisée.

        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe (e.g., '15m')
                - direction: Trading direction ('long', 'short', 'both')

        Returns:
            Dict containing:
            - 'success': bool indicating if step completed successfully
            - 'artifacts': dict of created artifacts
            - 'metrics': dict of performance metrics
            - 'error': error message if step failed (optional)
        """
        symbol = config.get('symbol', 'UNKNOWN')
        tprint(f"🧠 [TACTICIAN_BASE] Starting centralized configuration tactician base training for {symbol}", "INFO")
        
        # Afficher les informations de configuration
        summary = self.get_training_summary()
        config_status = "✅ Centralized" if self._centralized_config else "⚠️ Fallback"
        tprint(f"📋 [TACTICIAN_BASE] Configuration: {config_status} | Models: {summary['centralized_config']['models_count']} | Target: {summary['config_sections']['tactician']['target']}", "INFO")

        try:
            # Import and call unified training step
            from .unified_models_training_step import UnifiedModelsTrainingStep
            
            # Enrichir la configuration avec les paramètres centralisés
            enhanced_config = config.copy()
            
            # Configuration centrale
            enhanced_config.update({
                'training_type': 'tactician_base',
                'execution_context': 'tactician',
                'centralized_config_enabled': self._centralized_config is not None,
                'config_summary': summary
            })
            
            # Ajouter les configurations spécifiques
            enhanced_config['tactician_config'] = self.get_tactician_models_config()
            enhanced_config['feature_engineering'] = self.get_feature_engineering_config()
            enhanced_config['training'] = self.get_training_config()
            enhanced_config['hardware'] = self.get_hardware_config()
            enhanced_config['performance'] = self.get_performance_targets()
            enhanced_config['output'] = self.get_output_config()
            
            # Créer et exécuter l'étape d'entraînement unifiée
            unified_step = UnifiedModelsTrainingStep()
            result = await unified_step.execute(enhanced_config)
            
            # Enrichir le résultat avec les informations de configuration
            if result.get('success'):
                result['config_source'] = config_source
                result['config_summary'] = summary
                tprint(f"✅ [TACTICIAN_BASE] Training completed successfully with {config_status} configuration", "SUCCESS")
            else:
                tprint(f"❌ [TACTICIAN_BASE] Training failed", "ERROR")
            
            return result

        except Exception as e:
            error_msg = f"Tactician base training failed: {str(e)}"
            tprint(f"❌ [TACTICIAN_BASE] {error_msg}", "ERROR")
            self.logger.error(error_msg, exc_info=True)

            return {
                'success': False,
                'artifacts': {},
                'metrics': {},
                'error': error_msg,
                'config_source': 'fallback',
                'config_summary': summary
            }

    async def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run method required by BaseStep interface."""
        return await self.execute(config)


# Fonctions factory pour faciliter l'utilisation
def create_tactician_base_training_step(use_centralized_config: bool = True) -> TacticianBaseTrainingStep:
    """
    Factory function pour créer une étape d'entraînement tactician base.
    
    Args:
        use_centralized_config: Utiliser la configuration centralisée
        
    Returns:
        Instance de TacticianBaseTrainingStep
    """
    step = TacticianBaseTrainingStep()
    if not use_centralized_config:
        step._centralized_config = None
    return step


def create_tactician_base_training_with_custom_config(config_path: str) -> TacticianBaseTrainingStep:
    """
    Factory function pour créer une étape avec configuration personnalisée.
    
    Args:
        config_path: Chemin vers le fichier de configuration personnalisé
        
    Returns:
        Instance de TacticianBaseTrainingStep avec configuration personnalisée
    """
    if set_tactician_base_training_custom_config_path(config_path):
        return TacticianBaseTrainingStep()
    else:
        tprint(f"⚠️ [TACTICIAN_BASE] Impossible de charger la config personnalisée: {config_path}", "WARNING")
        return TacticianBaseTrainingStep()


# Register the step
def register_tactician_base_training_step():
    """Register the tactician base training step."""
    from src.training.steps.base_step import step_registry

    step_registry.register("tactician_base_training", TacticianBaseTrainingStep)
    tprint("✅ Tactician base training step registered", "SUCCESS")


# Auto-register when module is imported
register_tactician_base_training_step()
