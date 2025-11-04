#!/usr/bin/env python3
"""
Configuration Centralisée - Tactician Base Training

Module d'export pour le système de configuration centralisée de l'entraînement
des modèles tactician de base, fournissant un accès unifié et robuste aux
paramètres de configuration avec support YAML/JSON et système de fallback.

Version: 1.0.0
Date: 2025-11-03T22:22:00.000Z
"""

import os
import sys
from typing import Any, Dict, List, Optional

# Ajouter le répertoire parent au path pour les imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import du gestionnaire de configuration
from .config_manager import TacticianBaseTrainingConfigManager

# Import de la configuration structurée
from .default_config import TacticianBaseTrainingConfig

# Import des fonctions factory
from .config_manager import (
    get_tactician_base_training_config,
    get_tactician_base_training_config_manager,
    set_tactician_base_training_custom_config_path
)

# Métadonnées du module
__version__ = "1.0.0"
__author__ = "Configuration Centralisée System"
__email__ = "system@kilocode.ai"
__license__ = "MIT"
__description__ = "Configuration centralisée pour tactician_base_training"

# Métadonnées de configuration
CONFIG_VERSION = "1.0.0"
CONFIG_CREATED_AT = "2025-11-03T22:22:00.000Z"
CONFIG_UPDATED_AT = "2025-11-03T22:22:00.000Z"

# Chemins des fichiers de configuration par défaut
DEFAULT_CONFIG_PATHS = [
    os.path.join(current_dir, "default_config.yaml"),
    os.path.join(current_dir, "default_config.json"),
    os.path.join(current_dir, "default_config.py")
]

# Informations du composant
COMPONENT_INFO = {
    "name": "tactician_base_training",
    "description": "Configuration centralisée pour l'entraînement des modèles tactician de base",
    "models_count": 5,
    "model_types": ["StandaloneGRU", "LGBM", "CatBoost", "ExtraTrees", "DepthwiseCNN"],
    "timeframe": "15m",
    "execution_frequency": "3m",
    "target": "entry_timing"
}

def get_component_info() -> Dict[str, Any]:
    """
    Obtenir les informations du composant.
    
    Returns:
        Dictionnaire avec les informations du composant
    """
    return COMPONENT_INFO.copy()

def get_config_metadata() -> Dict[str, Any]:
    """
    Obtenir les métadonnées de configuration.
    
    Returns:
        Dictionnaire avec les métadonnées
    """
    return {
        "version": __version__,
        "config_version": CONFIG_VERSION,
        "created_at": CONFIG_CREATED_AT,
        "updated_at": CONFIG_UPDATED_AT,
        "component": COMPONENT_INFO["name"],
        "default_config_paths": DEFAULT_CONFIG_PATHS
    }

def list_available_configurations() -> List[str]:
    """
    Lister les configurations disponibles.
    
    Returns:
        Liste des types de configuration disponibles
    """
    return [
        "default",
        "custom",
        "runtime"
    ]

def get_supported_formats() -> List[str]:
    """
    Obtenir les formats de configuration supportés.
    
    Returns:
        Liste des formats supportés
    """
    return ["yaml", "json", "python"]

def validate_config_installation() -> bool:
    """
    Valider l'installation de la configuration centralisée.
    
    Returns:
        True si l'installation est valide
    """
    try:
        # Vérifier que les fichiers existent
        for path in DEFAULT_CONFIG_PATHS:
            if not os.path.exists(path):
                return False
        
        # Vérifier l'import du gestionnaire
        manager = get_tactician_base_training_config_manager()
        if manager is None:
            return False
        
        # Vérifier le chargement de la configuration
        config = get_tactician_base_training_config()
        if config is None:
            return False
        
        return True
        
    except Exception:
        return False

# Fonction d'information pour le debugging
def print_config_info():
    """Afficher les informations de configuration pour le debugging."""
    print("🔧 Configuration Centralisée - Tactician Base Training")
    print("=" * 60)
    print(f"Version: {__version__}")
    print(f"Composant: {COMPONENT_INFO['name']}")
    print(f"Description: {COMPONENT_INFO['description']}")
    print(f"Modèles: {len(COMPONENT_INFO['model_types'])} ({', '.join(COMPONENT_INFO['model_types'])})")
    print(f"Timeframe: {COMPONENT_INFO['timeframe']}")
    print(f"Fréquence: {COMPONENT_INFO['execution_frequency']}")
    print(f"Target: {COMPONENT_INFO['target']}")
    
    print("\n📁 Fichiers de configuration:")
    for path in DEFAULT_CONFIG_PATHS:
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"  {exists} {os.path.basename(path)}")
    
    print(f"\n⚙️ Installation valide: {'✅' if validate_config_installation() else '❌'}")

# Auto-test si exécuté directement
if __name__ == "__main__":
    print_config_info()
    
    print("\n🧪 Test de chargement des configurations:")
    try:
        config = get_tactician_base_training_config()
        print(f"✅ Configuration chargée: {config.tactician_config.model_name}")
        print(f"   Modèles de base: {len(config.tactician_config.base_models)}")
        print(f"   Timeframe: {config.tactician_config.base_timeframe}")
    except Exception as e:
        print(f"❌ Erreur de chargement: {e}")
    
    print("\n🎯 Système prêt pour l'intégration!")