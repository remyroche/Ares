#!/usr/bin/env python3
"""
Gestionnaire de Configuration Centralisée pour Analyst Ensemble Training

Ce module fournit un système unifié pour la gestion des configurations de l'entraînement
d'ensemble des modèles analyst, avec support pour YAML, JSON et Python.
"""

import json
import yaml
import os
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field, asdict
import importlib.util
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnalystEnsembleTrainingConfig:
    """Configuration centralisée pour l'entraînement d'ensemble des modèles analyst."""
    
    version: str = "1.0.0"
    component_name: str = "analyst_ensemble_training"
    description: str = "Configuration centralisée pour l'entraînement d'ensemble des modèles analyst"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Configuration principale du modèle analyst
    analyst_config: Dict[str, Any] = field(default_factory=lambda: {
        'model_name': 'analyst_ensemble',
        'model_type': 'meta_ensemble',
        'target': 'trading_signal',
        'base_timeframe': '15m',
        'execution_timeframe': '15m',
        'execution_frequency': '15m',
        'params': {}
    })
    
    # Configuration du meta-learner
    meta_learner: Dict[str, Any] = field(default_factory=lambda: {
        'model_type': 'stacker_lgbm_calibrated',
        'params': {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 6,
            'num_leaves': 63,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'random_state': 42,
            'verbosity': -1,
            'n_jobs': -1
        },
        'calibration': {
            'method': 'isotonic',
            'enable_temperature_scaling': True
        },
        'hpo': {
            'enabled': True,
            'n_rounds': 2,
            'enable_final_refinement': True,
            'final_refinement_trials': 50,
            'optimal_params': {}
        }
    })
    
    # Configuration hardware et optimisation
    hardware: Dict[str, Any] = field(default_factory=lambda: {
        'enable_gpu_acceleration': True,
        'enable_memory_optimization': True,
        'enable_parallel_processing': True,
        'memory_limit_gb': 4.0,
        'max_workers': None,
        'cpu_optimization_level': 'aggressive',
        'gpu_optimization_level': 'balanced'
    })
    
    # Configuration HPO
    hpo: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'max_trials': 50,
        'timeout_seconds': 300,
        'enable_early_stopping': True,
        'enable_pruning': True,
        'cv_folds': 3
    })
    
    # Configuration de l'entraînement
    training: Dict[str, Any] = field(default_factory=lambda: {
        'enable_cross_validation': True,
        'cv_folds': 3,
        'enable_early_stopping': True,
        'early_stopping_patience': 15,
        'validation_split': 0.2,
        'test_split': 0.1,
        'training_samples': 15000,
        'validation_samples': 5000,
        'test_samples': 3000
    })
    
    # Configuration de l'ingénierie des features
    feature_engineering: Dict[str, Any] = field(default_factory=lambda: {
        'primary_features': {
            'source': 'feature_generation_final_feature_selection_step',
            'artifact_name': 'analyst_features',
            'initial_count': 300,
            'target_count': 100
        },
        'cross_timeframe': {
            'enable': True,
            'base_timeframe': '5m',
            'target_timeframes': ['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
            'feature_types': ['technical_indicators', 'price_action', 'volume_profile', 'volatility']
        },
        'regime_features': {
            'enable': True,
            'source': 'regime_ml_models',
            'feature_names': ['regime_prob_0', 'regime_prob_1', 'regime_prob_2', 'regime_prob_3'],
            'include_regime_outputs': True
        },
        'analyst_base_outputs': {
            'enable': True,
            'source': 'analyst_base_models',
            'features': [
                'lgbm_predictions', 'lgbm_patchtst_predictions', 
                'catboost_predictions', 'meta_learner_predictions',
                'base_model_confidence_scores'
            ]
        },
        'feature_selection': {
            'method': 'lasso',
            'alpha': 0.01,
            'max_features': 100,
            'enable_recursive_elimination': True,
            'enable_feature_importance': True
        },
        'scaling': {
            'method': 'robust',
            'enable_outlier_handling': True,
            'outlier_threshold': 3.0
        }
    })
    
    # Configuration de préparation des données
    data_preparation: Dict[str, Any] = field(default_factory=lambda: {
        'time_series': {
            'enable_temporal_features': True,
            'lookback_window': 100,
            'forecast_horizon': 1
        },
        'target_generation': {
            'method': 'price_momentum',
            'lookback_periods': [5, 10, 20],
            'threshold': 0.02
        }
    })
    
    # Configuration d'évaluation
    evaluation: Dict[str, Any] = field(default_factory=lambda: {
        'metrics': {
            'regression': ['mse', 'mae', 'r2', 'mape'],
            'classification': ['accuracy', 'precision', 'recall', 'f1']
        },
        'cross_validation': {
            'method': 'time_series_split',
            'n_splits': 5,
            'test_size': 0.2
        },
        'comparison': {
            'enable_model_ranking': True,
            'ranking_metric': 'r2',
            'enable_feature_importance': True,
            'enable_model_explanations': True
        }
    })
    
    # Configuration de performance
    performance: Dict[str, Any] = field(default_factory=lambda: {
        'expected_accuracy': 0.85,
        'expected_diversity_score': 0.92,
        'training_time_limit': 600,
        'memory_limit_mb': 4096
    })
    
    # Configuration de sortie
    output: Dict[str, Any] = field(default_factory=lambda: {
        'save_models': True,
        'save_predictions': True,
        'generate_reports': True,
        'output_dir': './analyst_ensemble_models'
    })
    
    # Configuration de logging
    logging: Dict[str, Any] = field(default_factory=lambda: {
        'level': 'INFO',
        'enable_detailed_logging': True,
        'log_predictions': True,
        'log_performance_metrics': True,
        'log_feature_importance': True,
        'log_ensemble_metrics': True,
        'intervals': {
            'training_progress': 50,
            'prediction_summary': 500,
            'performance_update': 2000,
            'ensemble_update': 1000
        }
    })
    
    # Configuration des modèles de base
    base_models: Dict[str, Any] = field(default_factory=lambda: {
        'lgbm': {
            'enabled': True,
            'params': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1
            }
        },
        'catboost': {
            'enabled': True,
            'params': {
                'iterations': 100,
                'depth': 6,
                'learning_rate': 0.1
            }
        },
        'tcn': {
            'enabled': True,
            'params': {
                'epochs': 50,
                'batch_size': 32
            }
        }
    })
    
    def update(self, **kwargs) -> 'AnalystEnsembleTrainingConfig':
        """Mettre à jour la configuration avec de nouveaux paramètres."""
        updated_dict = asdict(self)
        updated_dict.update(kwargs)
        updated_dict['last_updated'] = datetime.now().isoformat()
        return AnalystEnsembleTrainingConfig(**updated_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir la configuration en dictionnaire."""
        return asdict(self)
    
    def validate(self) -> List[str]:
        """Valider la configuration et retourner la liste des erreurs."""
        errors = []
        
        # Valider les types de données
        if not isinstance(self.version, str):
            errors.append("version doit être une chaîne")
        
        if not isinstance(self.analyst_config, dict):
            errors.append("analyst_config doit être un dictionnaire")
        
        if not isinstance(self.hardware, dict):
            errors.append("hardware doit être un dictionnaire")
        
        if not isinstance(self.meta_learner, dict):
            errors.append("meta_learner doit être un dictionnaire")
        
        # Valider les valeurs spécifiques
        if self.hardware.get('memory_limit_gb', 0) <= 0:
            errors.append("memory_limit_gb doit être positif")
        
        if self.training.get('cv_folds', 0) <= 0:
            errors.append("cv_folds doit être positif")
        
        if self.training.get('validation_split', 1) >= 1 or self.training.get('validation_split', 0) <= 0:
            errors.append("validation_split doit être entre 0 et 1")
        
        return errors


class AnalystEnsembleTrainingConfigManager:
    """Gestionnaire principal pour la configuration centralisée."""
    
    def __init__(
        self,
        custom_config_path: Optional[str] = None,
        enable_hardcoded_fallback: bool = True
    ):
        """
        Initialiser le gestionnaire de configuration.
        
        Args:
            custom_config_path: Chemin vers un fichier de configuration personnalisé
            enable_hardcoded_fallback: Activer le fallback vers les valeurs hardcodées
        """
        self.config_directory = Path(__file__).parent
        self.custom_config_path = custom_config_path
        self.enable_hardcoded_fallback = enable_hardcoded_fallback
        
        # Configuration en cache
        self._config_cache = None
        self._cache_timestamp = None
        
        # Logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("AnalystEnsembleTrainingConfigManager initialisé")
        self.logger.info(f"Répertoire config: {self.config_directory}")
        self.logger.info(f"Config personnalisée: {custom_config_path}")
        self.logger.info(f"Fallback hardcodé: {enable_hardcoded_fallback}")
    
    def get_config(self, config_path: Optional[List[str]] = None) -> AnalystEnsembleTrainingConfig:
        """
        Obtenir la configuration complète ou une section spécifique.
        
        Args:
            config_path: Liste de clés pour accéder à une section spécifique
            
        Returns:
            Configuration complète ou section spécifique
        """
        try:
            # Charger la configuration si nécessaire
            if self._config_cache is None:
                self._config_cache = self._load_config()
            
            # Retourner la configuration complète si aucun chemin spécifique
            if config_path is None:
                return self._config_cache
            
            # Accéder à une section spécifique
            current = self._config_cache
            for key in config_path:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    raise KeyError(f"Clé '{key}' non trouvée dans la configuration")
            
            return current
            
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de la configuration: {e}")
            if self.enable_hardcoded_fallback:
                self.logger.info("Utilisation du fallback hardcodé")
                return self._get_hardcoded_fallback()
            raise
    
    def _load_config(self) -> AnalystEnsembleTrainingConfig:
        """Charger la configuration depuis les fichiers disponibles."""
        # Essayer de charger depuis la configuration personnalisée d'abord
        if self.custom_config_path and os.path.exists(self.custom_config_path):
            try:
                config = self._load_from_file(self.custom_config_path)
                self.logger.info(f"Configuration personnalisée chargée depuis {self.custom_config_path}")
                return config
            except Exception as e:
                self.logger.warning(f"Échec du chargement de la configuration personnalisée: {e}")
        
        # Essayer de charger depuis les fichiers par défaut dans l'ordre de priorité
        default_files = [
            "default_config.json",
            "default_config.yaml", 
            "default_config.py"
        ]
        
        for filename in default_files:
            file_path = self.config_directory / filename
            if file_path.exists():
                try:
                    config = self._load_from_file(str(file_path))
                    self.logger.info(f"Configuration default chargée depuis {file_path}")
                    return config
                except Exception as e:
                    self.logger.warning(f"Échec du chargement de {file_path}: {e}")
        
        # Fallback hardcodé
        if self.enable_hardcoded_fallback:
            self.logger.info("Utilisation du fallback hardcodé")
            return self._get_hardcoded_fallback()
        
        raise FileNotFoundError("Aucun fichier de configuration trouvé et fallback désactivé")
    
    def _load_from_file(self, file_path: str) -> AnalystEnsembleTrainingConfig:
        """Charger la configuration depuis un fichier spécifique."""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        elif file_path.suffix.lower() == '.yaml' or file_path.suffix.lower() == '.yml':
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
        elif file_path.suffix.lower() == '.py':
            # Charger depuis un module Python
            spec = importlib.util.spec_from_file_location("default_config", file_path)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            config_data = config_module.config_data
        else:
            raise ValueError(f"Format de fichier non supporté: {file_path.suffix}")
        
        # Créer l'objet configuration
        return AnalystEnsembleTrainingConfig(**config_data)
    
    def _get_hardcoded_fallback(self) -> AnalystEnsembleTrainingConfig:
        """Retourner la configuration hardcodée par défaut."""
        return AnalystEnsembleTrainingConfig()
    
    def validate_config(self, config: Union[AnalystEnsembleTrainingConfig, Dict[str, Any]]) -> bool:
        """
        Valider une configuration.
        
        Args:
            config: Configuration à valider
            
        Returns:
            True si la configuration est valide, False sinon
        """
        try:
            if isinstance(config, dict):
                config = AnalystEnsembleTrainingConfig(**config)
            
            errors = config.validate()
            if errors:
                self.logger.error(f"Erreurs de validation: {errors}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la validation: {e}")
            return False
    
    def _get_config_with_fallback(self, custom_config: Dict[str, Any]) -> AnalystEnsembleTrainingConfig:
        """
        Fusionner une configuration personnalisée avec le fallback.
        
        Args:
            custom_config: Configuration personnalisée
            
        Returns:
            Configuration fusionnée
        """
        try:
            # Charger la configuration par défaut
            default_config = self.get_config()
            default_dict = default_config.to_dict()
            
            # Fusionner avec la configuration personnalisée
            merged_dict = default_dict.copy()
            merged_dict.update(custom_config)
            
            # Créer la configuration fusionnée
            merged_config = AnalystEnsembleTrainingConfig(**merged_dict)
            
            # Valider la configuration fusionnée
            if not self.validate_config(merged_config):
                self.logger.warning("Configuration fusionnée invalide, utilisation du fallback")
                return default_config
            
            return merged_config
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la fusion: {e}")
            return self._get_hardcoded_fallback()


# Instance globale du gestionnaire
_global_config_manager = None


def get_analyst_ensemble_config_manager() -> AnalystEnsembleTrainingConfigManager:
    """Obtenir l'instance globale du gestionnaire de configuration."""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = AnalystEnsembleTrainingConfigManager()
    return _global_config_manager


def get_analyst_ensemble_config(config_path: Optional[List[str]] = None) -> Union[AnalystEnsembleTrainingConfig, Any]:
    """
    Fonction convenience pour obtenir la configuration.
    
    Args:
        config_path: Liste de clés pour accéder à une section spécifique
        
    Returns:
        Configuration complète ou section spécifique
    """
    manager = get_analyst_ensemble_config_manager()
    return manager.get_config(config_path)


def set_custom_config_path(config_path: str) -> None:
    """
    Définir le chemin vers une configuration personnalisée.
    
    Args:
        config_path: Chemin vers le fichier de configuration
    """
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = AnalystEnsembleTrainingConfigManager(
            custom_config_path=config_path
        )
    else:
        _global_config_manager.custom_config_path = config_path
        _global_config_manager._config_cache = None  # Invalider le cache


# Auto-test si exécuté directement
if __name__ == "__main__":
    print("🧪 Test du gestionnaire de configuration Analyst Ensemble Training")
    print("=" * 70)
    
    try:
        # Test de chargement
        config = get_analyst_ensemble_config()
        print(f"✅ Configuration chargée: {config.component_name}")
        print(f"   Version: {config.version}")
        print(f"   Description: {config.description}")
        
        # Test d'accès aux sections
        meta_learner = get_analyst_ensemble_config(['meta_learner'])
        print(f"   Meta-learner type: {meta_learner.get('model_type', 'N/A')}")
        
        hardware = get_analyst_ensemble_config(['hardware'])
        print(f"   GPU acceleration: {hardware.get('enable_gpu_acceleration', 'N/A')}")
        
        print("✅ Test réussi!")
        
    except Exception as e:
        print(f"❌ Test échoué: {e}")
        import traceback
        traceback.print_exc()