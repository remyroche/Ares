#!/usr/bin/env python3
"""
Gestionnaire de Configuration Centralisée - Tactician Base Training

Gestionnaire principal pour la configuration centralisée des modèles tactician de base.
Fournit un système robuste de chargement multi-format avec fallback intelligent
et validation automatique des configurations.

Version: 1.0.0
Date: 2025-11-03T22:22:00.000Z
"""

import os
import sys
import json
import yaml
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import asdict
import traceback

# Ajouter le répertoire parent au path pour les imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import de la configuration structurée
from .default_config import TacticianBaseTrainingConfig

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TacticianBaseTrainingConfigManager:
    """
    Gestionnaire centralisé pour la configuration tactician_base_training.
    
    Gère le chargement, la validation et l'accès aux configurations avec
    support multi-format et système de fallback robuste.
    """
    
    def __init__(self, custom_config_path: Optional[str] = None):
        """
        Initialiser le gestionnaire de configuration.
        
        Args:
            custom_config_path: Chemin vers un fichier de configuration personnalisé
        """
        self.config_cache = {}
        self.config_timestamp = None
        self.custom_config_path = custom_config_path
        self.last_error = None
        
        # Chemins de configuration par défaut
        self.config_directory = os.path.dirname(os.path.abspath(__file__))
        self.default_config_paths = [
            os.path.join(self.config_directory, "default_config.yaml"),
            os.path.join(self.config_directory, "default_config.json"),
            os.path.join(self.config_directory, "default_config.py")
        ]
        
        logger.info(f"✅ TacticianBaseTrainingConfigManager initialisé")
    
    def load_config(self, config_path: Optional[str] = None) -> Optional[TacticianBaseTrainingConfig]:
        """
        Charger une configuration depuis un fichier ou le cache.
        
        Args:
            config_path: Chemin vers le fichier de configuration (optionnel)
            
        Returns:
            Configuration chargée ou None en cas d'échec
        """
        try:
            if config_path:
                return self._load_config_from_path(config_path)
            else:
                return self._load_config_with_fallback()
                
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"❌ Échec du chargement de la configuration: {e}")
            return None
    
    def _load_config_with_fallback(self) -> Optional[TacticianBaseTrainingConfig]:
        """
        Charger une configuration avec système de fallback.
        
        Returns:
            Configuration chargée ou None
        """
        # 1. Essayer la configuration personnalisée
        if self.custom_config_path and os.path.exists(self.custom_config_path):
            logger.info(f"📄 Chargement configuration personnalisée: {self.custom_config_path}")
            config = self._load_config_from_path(self.custom_config_path)
            if config:
                return config
        
        # 2. Essayer les fichiers par défaut dans l'ordre de préférence
        for default_path in self.default_config_paths:
            if os.path.exists(default_path):
                logger.info(f"📄 Chargement configuration par défaut: {os.path.basename(default_path)}")
                config = self._load_config_from_path(default_path)
                if config:
                    return config
        
        # 3. Fallback vers la configuration hardcodée
        logger.warning("⚠️ Utilisation de la configuration hardcodée fallback")
        return self._create_hardcoded_config()
    
    def _load_config_from_path(self, config_path: str) -> Optional[TacticianBaseTrainingConfig]:
        """
        Charger une configuration depuis un chemin spécifique.
        
        Args:
            config_path: Chemin vers le fichier de configuration
            
        Returns:
            Configuration chargée ou None
        """
        try:
            file_extension = os.path.splitext(config_path)[1].lower()
            
            if file_extension == '.yaml' or file_extension == '.yml':
                return self._load_from_yaml(config_path)
            elif file_extension == '.json':
                return self._load_from_json(config_path)
            elif file_extension == '.py':
                return self._load_from_python(config_path)
            else:
                raise ValueError(f"Format de fichier non supporté: {file_extension}")
                
        except Exception as e:
            logger.error(f"❌ Erreur de chargement depuis {config_path}: {e}")
            raise
    
    def _load_from_yaml(self, yaml_path: str) -> TacticianBaseTrainingConfig:
        """Charger la configuration depuis un fichier YAML."""
        try:
            with open(yaml_path, 'r', encoding='utf-8') as file:
                config_data = yaml.safe_load(file)
            
            # Validation basique
            if not self._validate_config_structure(config_data):
                raise ValueError("Structure de configuration invalide")
            
            # Créer la configuration structurée
            return self._create_config_from_dict(config_data)
            
        except Exception as e:
            logger.error(f"❌ Erreur de chargement YAML: {e}")
            raise
    
    def _load_from_json(self, json_path: str) -> TacticianBaseTrainingConfig:
        """Charger la configuration depuis un fichier JSON."""
        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                config_data = json.load(file)
            
            # Validation basique
            if not self._validate_config_structure(config_data):
                raise ValueError("Structure de configuration invalide")
            
            # Créer la configuration structurée
            return self._create_config_from_dict(config_data)
            
        except Exception as e:
            logger.error(f"❌ Erreur de chargement JSON: {e}")
            raise
    
    def _load_from_python(self, python_path: str) -> TacticianBaseTrainingConfig:
        """Charger la configuration depuis un module Python."""
        try:
            # Charger le module Python dynamiquement
            spec = __import__('importlib.util', fromlist=['spec_from_file_location']).spec_from_file_location(
                "tactician_base_config", python_path
            )
            config_module = __import__('importlib.util', fromlist=['module_from_spec']).module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
            # Récupérer la configuration depuis le module
            if hasattr(config_module, 'create_tactician_base_training_config'):
                # Cas 1: Le module contient une fonction qui retourne directement un objet configuration
                result = config_module.create_tactician_base_training_config()
                
                # Si la fonction retourne directement une configuration, la retourner
                if hasattr(result, 'tactician_config') and hasattr(result, 'feature_engineering'):
                    return result
                else:
                    # Si la fonction retourne un dictionnaire, le convertir
                    config_data = result
            elif hasattr(config_module, 'config_data'):
                # Cas 2: Le module contient un dictionnaire config_data
                config_data = config_module.config_data
            elif hasattr(config_module, 'get_default_config'):
                # Cas 3: Le module contient une fonction get_default_config
                result = config_module.get_default_config()
                
                # Si la fonction retourne directement une configuration, la retourner
                if hasattr(result, 'tactician_config') and hasattr(result, 'feature_engineering'):
                    return result
                else:
                    config_data = result
            else:
                raise ValueError("Module Python doit contenir 'create_tactician_base_training_config()', 'config_data' ou 'get_default_config()'")
            
            # Si on a des données de configuration (dictionnaire), les convertir
            if isinstance(config_data, dict):
                # Validation basique
                if not self._validate_config_structure(config_data):
                    raise ValueError("Structure de configuration invalide")
                
                # Créer la configuration structurée
                return self._create_config_from_dict(config_data)
            else:
                raise ValueError(f"Type de configuration non supporté: {type(config_data)}")
            
        except Exception as e:
            logger.error(f"❌ Erreur de chargement Python: {e}")
            raise
    
    def _validate_config_structure(self, config_data: Dict[str, Any]) -> bool:
        """
        Valider la structure de base de la configuration.
        
        Args:
            config_data: Données de configuration à valider
            
        Returns:
            True si la structure est valide
        """
        required_sections = ['tactician_config', 'feature_engineering', 'training']
        
        for section in required_sections:
            if section not in config_data:
                logger.error(f"❌ Section manquante: {section}")
                return False
        
        # Validation spécifique tactician_config
        tactician_config = config_data.get('tactician_config', {})
        required_tactician_keys = ['model_name', 'model_type', 'target']
        
        for key in required_tactician_keys:
            if key not in tactician_config:
                logger.error(f"❌ Clé manquante dans tactician_config: {key}")
                return False
        
        return True
    
    def _create_config_from_dict(self, config_data: Dict[str, Any]) -> TacticianBaseTrainingConfig:
        """
        Créer une configuration structurée depuis un dictionnaire.
        
        Args:
            config_data: Données de configuration
            
        Returns:
            Configuration structurée
        """
        try:
            # Ajouter les métadonnées si manquantes
            if '_metadata' not in config_data:
                config_data['_metadata'] = {
                    'loaded_at': datetime.now().isoformat(),
                    'manager_version': '1.0.0',
                    'format': 'yaml'  # Sera ajusté selon le format
                }
            
            # Créer la configuration structurée
            return TacticianBaseTrainingConfig.from_dict(config_data)
            
        except Exception as e:
            logger.error(f"❌ Erreur de création de configuration structurée: {e}")
            raise
    
    def _create_hardcoded_config(self) -> TacticianBaseTrainingConfig:
        """
        Créer une configuration hardcodée en fallback.
        
        Returns:
            Configuration hardcodée
        """
        logger.warning("⚠️ Création de la configuration hardcodée fallback")
        
        # Configuration minimaliste mais valide
        fallback_data = {
            '_metadata': {
                'loaded_at': datetime.now().isoformat(),
                'manager_version': '1.0.0',
                'format': 'hardcoded',
                'is_fallback': True
            },
            'tactician_config': {
                'model_name': 'tactician_base_fallback',
                'model_type': 'separate_models',
                'target': 'entry_timing',
                'base_timeframe': '15m',
                'execution_timeframe': '15m',
                'execution_frequency': '3m',
                'price_change_target': 0.005,
                'base_models': [
                    {
                        'model_name': 'LGBM',
                        'class_name': 'lightgbm.LGBMRegressor',
                        'is_feature_generator': False,
                        'params': {
                            'n_estimators': 100,
                            'learning_rate': 0.1,
                            'num_leaves': 31,
                            'objective': 'regression',
                            'n_jobs': -1,
                            'verbose': -1
                        },
                        'hpo': {'enabled': False}
                    }
                ]
            },
            'feature_engineering': {
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
            },
            'training': {
                'enable_cross_validation': True,
                'cv_folds': 3,
                'enable_early_stopping': True,
                'early_stopping_patience': 10
            },
            'hardware': {
                'enable_gpu_acceleration': False,
                'enable_memory_optimization': True,
                'enable_parallel_processing': True,
                'memory_limit_gb': 2.0
            },
            'performance': {
                'expected_accuracy': 0.80,
                'training_time_limit': 300,
                'memory_limit_mb': 2048
            },
            'output': {
                'save_models': True,
                'save_predictions': True,
                'output_dir': './tactician_base_models_fallback'
            }
        }
        
        return TacticianBaseTrainingConfig.from_dict(fallback_data)
    
    def get_config(self, config_path: Optional[str] = None) -> Optional[TacticianBaseTrainingConfig]:
        """
        Méthode d'accès simplifiée pour obtenir la configuration.
        
        Args:
            config_path: Chemin vers le fichier de configuration (optionnel)
            
        Returns:
            Configuration chargée
        """
        return self.load_config(config_path)
    
    def get_config_section(self, section_path: List[str]) -> Any:
        """
        Obtenir une section spécifique de la configuration.
        
        Args:
            section_path: Chemin vers la section (ex: ['tactician_config', 'model_name'])
            
        Returns:
            Section de configuration demandée
        """
        config = self.get_config()
        if not config:
            return None
        
        current = asdict(config)
        for key in section_path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def validate_config(self, config: TacticianBaseTrainingConfig) -> bool:
        """
        Valider une configuration complète.
        
        Args:
            config: Configuration à valider
            
        Returns:
            True si la configuration est valide
        """
        try:
            # Validation de base avec la dataclass
            if not config.validate():
                return False
            
            # Validations spécifiques au tactician
            tactician_config = config.tactician_config
            
            # Vérifier qu'il y a au moins un modèle de base
            if not tactician_config.base_models or len(tactician_config.base_models) == 0:
                logger.error("❌ Aucun modèle de base configuré")
                return False
            
            # Vérifier que les modèles ont les attributs requis
            for model in tactician_config.base_models:
                if not hasattr(model, 'model_name') or not hasattr(model, 'class_name'):
                    logger.error(f"❌ Modèle incomplet: {model}")
                    return False
            
            # Vérifier la configuration d'ingénierie des features
            if not config.feature_engineering.primary_features.artifact_name:
                logger.error("❌ Nom d'artifact manquant pour les features")
                return False
            
            logger.info("✅ Configuration validée avec succès")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur de validation: {e}")
            return False
    
    def export_config(self, config: TacticianBaseTrainingConfig, output_path: str) -> bool:
        """
        Exporter une configuration vers un fichier.
        
        Args:
            config: Configuration à exporter
            output_path: Chemin du fichier de sortie
            
        Returns:
            True si l'export a réussi
        """
        try:
            file_extension = os.path.splitext(output_path)[1].lower()
            
            if file_extension == '.yaml' or file_extension == '.yml':
                return self._export_to_yaml(config, output_path)
            elif file_extension == '.json':
                return self._export_to_json(config, output_path)
            else:
                raise ValueError(f"Format d'export non supporté: {file_extension}")
                
        except Exception as e:
            logger.error(f"❌ Erreur d'export: {e}")
            return False
    
    def _export_to_yaml(self, config: TacticianBaseTrainingConfig, output_path: str) -> bool:
        """Exporter la configuration vers YAML."""
        try:
            config_dict = config.to_dict()
            
            with open(output_path, 'w', encoding='utf-8') as file:
                yaml.dump(config_dict, file, default_flow_style=False, indent=2, allow_unicode=True)
            
            logger.info(f"✅ Configuration exportée vers YAML: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur d'export YAML: {e}")
            return False
    
    def _export_to_json(self, config: TacticianBaseTrainingConfig, output_path: str) -> bool:
        """Exporter la configuration vers JSON."""
        try:
            config_dict = config.to_dict()
            
            with open(output_path, 'w', encoding='utf-8') as file:
                json.dump(config_dict, file, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Configuration exportée vers JSON: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur d'export JSON: {e}")
            return False
    
    def set_custom_config_path(self, custom_path: str) -> bool:
        """
        Définir le chemin de configuration personnalisée.
        
        Args:
            custom_path: Chemin vers le fichier de configuration personnalisé
            
        Returns:
            True si le chemin est valide
        """
        if not os.path.exists(custom_path):
            logger.error(f"❌ Fichier de configuration non trouvé: {custom_path}")
            return False
        
        self.custom_config_path = custom_path
        self._clear_cache()
        
        logger.info(f"✅ Chemin de configuration personnalisée défini: {custom_path}")
        return True
    
    def _clear_cache(self):
        """Vider le cache de configuration."""
        self.config_cache.clear()
        self.config_timestamp = None
        logger.debug("🗑️ Cache de configuration vidé")
    
    def get_config_info(self) -> Dict[str, Any]:
        """
        Obtenir les informations sur la configuration actuelle.
        
        Returns:
            Dictionnaire avec les informations de configuration
        """
        return {
            'custom_config_path': self.custom_config_path,
            'default_config_paths': self.default_config_paths,
            'cache_size': len(self.config_cache),
            'last_error': self.last_error,
            'timestamp': self.config_timestamp.isoformat() if self.config_timestamp else None
        }


# Fonctions Factory pour l'usage simplifié
def get_tactician_base_training_config_manager(custom_config_path: Optional[str] = None) -> TacticianBaseTrainingConfigManager:
    """
    Créer une instance du gestionnaire de configuration.
    
    Args:
        custom_config_path: Chemin vers la configuration personnalisée (optionnel)
        
    Returns:
        Instance du gestionnaire de configuration
    """
    return TacticianBaseTrainingConfigManager(custom_config_path)


def get_tactician_base_training_config(config_path: Optional[str] = None) -> Optional[TacticianBaseTrainingConfig]:
    """
    Charger la configuration tactician_base_training.
    
    Args:
        config_path: Chemin vers le fichier de configuration (optionnel)
        
    Returns:
        Configuration chargée ou None
    """
    try:
        manager = get_tactician_base_training_config_manager()
        return manager.load_config(config_path)
        
    except Exception as e:
        logger.error(f"❌ Échec du chargement de la configuration: {e}")
        return None


def set_tactician_base_training_custom_config_path(custom_path: str) -> bool:
    """
    Définir le chemin de configuration personnalisée.
    
    Args:
        custom_path: Chemin vers la configuration personnalisée
        
    Returns:
        True si le chemin a été défini avec succès
    """
    try:
        manager = get_tactician_base_training_config_manager()
        return manager.set_custom_config_path(custom_path)
        
    except Exception as e:
        logger.error(f"❌ Échec de définition du chemin personnalisé: {e}")
        return False


def get_tactician_base_training_config_section(section_path: List[str]) -> Any:
    """
    Obtenir une section spécifique de la configuration.
    
    Args:
        section_path: Chemin vers la section
        
    Returns:
        Section de configuration demandée
    """
    try:
        manager = get_tactician_base_training_config_manager()
        return manager.get_config_section(section_path)
        
    except Exception as e:
        logger.error(f"❌ Échec d'accès à la section: {e}")
        return None


# Auto-test si exécuté directement
if __name__ == "__main__":
    print("🧪 Test du Gestionnaire de Configuration Tactician Base Training")
    print("=" * 70)
    
    try:
        # Test du gestionnaire
        manager = get_tactician_base_training_config_manager()
        print(f"✅ Gestionnaire créé: {type(manager).__name__}")
        
        # Test de chargement
        config = manager.load_config()
        if config:
            print(f"✅ Configuration chargée: {config.tactician_config.model_name}")
            print(f"   Modèles de base: {len(config.tactician_config.base_models)}")
            print(f"   Timeframe: {config.tactician_config.base_timeframe}")
            print(f"   Fréquence: {config.tactician_config.execution_frequency}")
        else:
            print("❌ Échec du chargement de la configuration")
        
        # Test de validation
        if config:
            is_valid = manager.validate_config(config)
            print(f"   Validation: {'✅ Valide' if is_valid else '❌ Invalide'}")
        
        # Test d'accès aux sections
        model_name = manager.get_config_section(['tactician_config', 'model_name'])
        print(f"   Nom du modèle: {model_name}")
        
    except Exception as e:
        print(f"❌ Erreur de test: {e}")
        import traceback
        traceback.print_exc()