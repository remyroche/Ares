"""
Système de gestion de configuration centralisée pour regime_models_training

Ce module fournit une interface unifiée pour charger, valider et gérer
les configurations YAML/JSON pour l'entraînement des modèles de détection de régime.

Fonctionnalités:
- Chargement automatique des fichiers de configuration
- Validation des schémas de configuration
- Héritage et fallback des configurations
- Support YAML et JSON
- Intégration avec le système existant
"""

import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass, asdict
from copy import deepcopy
import warnings
from src.utils.tprint import tprint

try:
    import ruamel.yaml  # type: ignore
    RUAMEL_AVAILABLE = True
except ImportError:
    RUAMEL_AVAILABLE = False


@dataclass
class ConfigValidationError(Exception):
    """Erreur de validation de configuration"""
    def __init__(self, message: str, missing_fields: List[str] = None, invalid_values: Dict[str, str] = None):
        super().__init__(message)
        self.missing_fields = missing_fields or []
        self.invalid_values = invalid_values or {}


class RegimeModelsTrainingConfigManager:
    """
    Gestionnaire de configuration centralisée pour regime_models_training
    
    Ce gestionnaire:
    1. Charge les configurations depuis YAML/JSON
    2. Valide les configurations contre un schéma
    3. Gère l'héritage et les fallback
    4. Permet les configurations personnalisées
    5. S'intègre avec le système existant
    """
    
    def __init__(self, config_dir: str = None):
        """
        Initialiser le gestionnaire de configuration
        
        Args:
            config_dir: Répertoire contenant les fichiers de configuration
        """
        # Déterminer le répertoire de configuration
        if config_dir is None:
            config_dir = Path(__file__).parent.absolute()
        
        self.config_dir = Path(config_dir)
        self.config_cache = {}
        self.schema = self._load_schema()
        
        # Patterns de recherche pour les fichiers de configuration
        self.config_patterns = [
            "custom_config.yaml",
            "custom_config.json", 
            "production_config.yaml",
            "production_config.json",
            "development_config.yaml",
            "development_config.json",
            "default_config.yaml",
            "default_config.json"
        ]
        
        tprint(f"🔧 [ConfigManager] Initialisé avec répertoire: {self.config_dir}", color="cyan")
    
    def _load_schema(self) -> Dict[str, Any]:
        """Charger le schéma de validation de configuration"""
        return {
            "required_sections": [
                "general",
                "models",
                "data_validation",
                "hpo"
            ],
            "required_fields": {
                "general": ["component_name", "version"],
                "models": ["base_models", "meta_learner"],
                "data_validation": ["min_samples", "min_features"],
                "hpo": ["enabled", "method", "max_trials"]
            },
            "model_types": [
                "catboost", "xgboost", "random_forest", "extratrees", "lightgbm"
            ],
            "validation_rules": {
                "min_samples": lambda x: x > 0,
                "max_trials": lambda x: x > 0 and x <= 1000,
                "timeout_seconds": lambda x: x > 0,
                "cv_folds": lambda x: x >= 2 and x <= 20,
                "learning_rate": lambda x: 0.0 < x <= 1.0
            }
        }
    
    def load_config(
        self, 
        config_name: str = None,
        config_path: str = None,
        merge_with_default: bool = True,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Charger une configuration depuis YAML/JSON
        
        Args:
            config_name: Nom du fichier de configuration (sans extension)
            config_path: Chemin complet vers le fichier de configuration
            merge_with_default: Fusionner avec la configuration par défaut
            validate: Valider la configuration chargée
            
        Returns:
            Configuration validée et fusionnée
        """
        # Déterminer quelle configuration charger
        if config_path:
            config_file = Path(config_path)
        elif config_name:
            config_file = self._find_config_file(config_name)
        else:
            # Charger la première configuration trouvée ou par défaut
            config_file = self._find_first_config()
        
        if not config_file or not config_file.exists():
            tprint(f"⚠️ [ConfigManager] Configuration non trouvée: {config_path or config_name}", color="yellow")
            if merge_with_default:
                return self._load_default_config(validate=validate)
            else:
                return {}
        
        try:
            tprint(f"📖 [ConfigManager] Chargement de la configuration: {config_file}", color="blue")
            config = self._load_config_file(config_file)
            
            if merge_with_default:
                default_config = self._load_default_config(validate=False)
                config = self._merge_configs(default_config, config)
            
            if validate:
                self._validate_config(config)
            
            # Mettre en cache
            config_key = str(config_file)
            self.config_cache[config_key] = config
            
            tprint(f"✅ [ConfigManager] Configuration chargée et validée: {len(config)} sections", color="green")
            return deepcopy(config)
            
        except Exception as e:
            tprint(f"❌ [ConfigManager] Erreur lors du chargement: {e}", color="red")
            if merge_with_default:
                return self._load_default_config(validate=validate)
            else:
                raise ConfigValidationError(f"Impossible de charger la configuration: {e}")
    
    def _find_config_file(self, config_name: str) -> Optional[Path]:
        """Trouver un fichier de configuration par nom"""
        # Essayer avec extension
        for ext in ['.yaml', '.yml', '.json']:
            config_file = self.config_dir / f"{config_name}{ext}"
            if config_file.exists():
                return config_file
        
        # Chercher dans les patterns connus
        for pattern in self.config_patterns:
            if config_name in pattern:
                config_file = self.config_dir / pattern
                if config_file.exists():
                    return config_file
        
        return None
    
    def _find_first_config(self) -> Optional[Path]:
        """Trouver le premier fichier de configuration disponible"""
        # Priorité: custom > production > development > default
        priority_patterns = [
            "custom_config.yaml", "custom_config.json",
            "production_config.yaml", "production_config.json", 
            "development_config.yaml", "development_config.json",
            "default_config.yaml", "default_config.json"
        ]
        
        for pattern in priority_patterns:
            config_file = self.config_dir / pattern
            if config_file.exists():
                return config_file
        
        return None
    
    def _load_config_file(self, config_file: Path) -> Dict[str, Any]:
        """Charger un fichier de configuration"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    if RUAMEL_AVAILABLE:
                        yaml_loader = ruamel.yaml.YAML(typ='safe', pure=True)
                        config = yaml_loader.load(f)
                    else:
                        config = yaml.safe_load(f)
                elif config_file.suffix.lower() == '.json':
                    config = json.load(f)
                else:
                    raise ValueError(f"Format de fichier non supporté: {config_file.suffix}")
            
            return config or {}
        except Exception as e:
            raise ConfigValidationError(f"Erreur lors du chargement du fichier {config_file}: {e}")
    
    def _load_default_config(self, validate: bool = True) -> Dict[str, Any]:
        """Charger la configuration par défaut"""
        default_file = self.config_dir / "default_config.yaml"
        
        if default_file.exists():
            config = self._load_config_file(default_file)
        else:
            # Configuration minimale par défaut
            config = self._create_minimal_default_config()
        
        if validate:
            self._validate_config(config)
        
        return config
    
    def _create_minimal_default_config(self) -> Dict[str, Any]:
        """Créer une configuration minimale par défaut"""
        return {
            "general": {
                "component_name": "regime_models_training",
                "version": "2.0.0",
                "description": "Configuration par défaut"
            },
            "models": {
                "base_models": {
                    "catboost": {
                        "enabled": True,
                        "hpo": {"enabled": True, "n_trials": 50}
                    }
                },
                "meta_learner": {
                    "enabled": True,
                    "hpo": {"enabled": True, "n_trials": 25}
                }
            },
            "hpo": {
                "enabled": True,
                "method": "bayesian",
                "max_trials": 50,
                "timeout_seconds": 300
            },
            "data_validation": {
                "min_samples": 10,
                "min_features": 50,
                "required_columns": ["close", "open", "high", "low", "volume"]
            }
        }
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Valider une configuration contre le schéma"""
        errors = []
        
        # Vérifier les sections requises
        for section in self.schema["required_sections"]:
            if section not in config:
                errors.append(f"Section manquante: {section}")
        
        # Vérifier les champs requis
        for section, fields in self.schema["required_fields"].items():
            if section in config:
                for field in fields:
                    if field not in config[section]:
                        errors.append(f"Champ manquant: {section}.{field}")
        
        # Vérifier les règles de validation
        for section_name, section_data in config.items():
            if isinstance(section_data, dict):
                for field, value in section_data.items():
                    rule_key = f"{section_name}.{field}"
                    if rule_key in self.schema["validation_rules"]:
                        rule = self.schema["validation_rules"][rule_key]
                        try:
                            if not rule(value):
                                errors.append(f"Valeur invalide pour {rule_key}: {value}")
                        except Exception:
                            errors.append(f"Erreur de validation pour {rule_key}: {value}")
        
        if errors:
            raise ConfigValidationError(
                f"Configuration invalide:\n" + "\n".join(f"- {error}" for error in errors),
                missing_fields=[e for e in errors if "manquant" in e],
                invalid_values={e.split(":")[0].strip(): e.split(":")[1].strip() for e in errors if "invalide" in e}
            )
    
    def _merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Fusionner deux configurations (base_config avec override_config)"""
        result = deepcopy(base_config)
        
        def merge_dict(base: dict, override: dict) -> dict:
            """Fusion récursive de dictionnaires"""
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dict(base[key], value)
                else:
                    base[key] = deepcopy(value)
            return base
        
        return merge_dict(result, override_config)
    
    def save_config(self, config: Dict[str, Any], config_name: str, format_type: str = "yaml") -> Path:
        """
        Sauvegarder une configuration
        
        Args:
            config: Configuration à sauvegarder
            config_name: Nom du fichier (sans extension)
            format_type: Format de sauvegarde ("yaml" ou "json")
            
        Returns:
            Chemin vers le fichier sauvegardé
        """
        if format_type.lower() not in ["yaml", "json"]:
            raise ValueError(f"Format non supporté: {format_type}")
        
        config_file = self.config_dir / f"{config_name}.{format_type.lower()}"
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                if format_type.lower() == "yaml":
                    if RUAMEL_AVAILABLE:
                        yaml = ruamel.yaml.YAML()
                        yaml.dump(config, f)
                    else:
                        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
                else:
                    json.dump(config, f, indent=2, ensure_ascii=False)
            
            tprint(f"💾 [ConfigManager] Configuration sauvegardée: {config_file}", color="green")
            return config_file
            
        except Exception as e:
            tprint(f"❌ [ConfigManager] Erreur lors de la sauvegarde: {e}", color="red")
            raise
    
    def list_available_configs(self) -> List[Dict[str, Any]]:
        """Lister toutes les configurations disponibles"""
        configs = []
        
        for pattern in self.config_patterns:
            config_file = self.config_dir / pattern
            if config_file.exists():
                try:
                    config = self._load_config_file(config_file)
                    configs.append({
                        "name": config_file.stem,
                        "path": str(config_file),
                        "format": config_file.suffix.lower(),
                        "size": config_file.stat().st_size,
                        "modified": config_file.stat().st_mtime,
                        "sections": list(config.keys()) if config else []
                    })
                except Exception as e:
                    tprint(f"⚠️ [ConfigManager] Erreur lors du chargement de {config_file}: {e}", color="yellow")
        
        return sorted(configs, key=lambda x: x["modified"], reverse=True)
    
    def create_custom_config(
        self, 
        base_config: str = "default",
        overrides: Dict[str, Any] = None,
        config_name: str = "custom_config"
    ) -> Dict[str, Any]:
        """
        Créer une configuration personnalisée
        
        Args:
            base_config: Configuration de base à utiliser
            overrides: Modifications à appliquer
            config_name: Nom de la configuration personnalisée
            
        Returns:
            Configuration personnalisée
        """
        tprint(f"🔧 [ConfigManager] Création d'une configuration personnalisée: {config_name}", color="cyan")
        
        # Charger la configuration de base
        base = self.load_config(base_config, merge_with_default=True, validate=True)
        
        # Appliquer les overrides
        if overrides:
            custom = self._merge_configs(base, overrides)
        else:
            custom = base
        
        # Sauvegarder si des overrides sont fournis
        if overrides:
            self.save_config(custom, config_name, "yaml")
            tprint(f"✅ [ConfigManager] Configuration personnalisée sauvegardée: {config_name}", color="green")
        
        return custom
    
    def get_model_config(self, config: Dict[str, Any], model_type: str) -> Dict[str, Any]:
        """
        Extraire la configuration d'un modèle spécifique
        
        Args:
            config: Configuration complète
            model_type: Type de modèle (catboost, xgboost, etc.)
            
        Returns:
            Configuration du modèle ou configuration vide si non trouvé
        """
        models_config = config.get("models", {})
        base_models = models_config.get("base_models", {})
        
        # Chercher dans les modèles de base
        if model_type in base_models:
            return base_models[model_type]
        
        # Chercher dans le meta-learner
        meta_learner = models_config.get("meta_learner", {})
        if meta_learner.get("name") == model_type or model_type == "stacker_lgbm_calibrated":
            return meta_learner
        
        # Configuration par défaut si non trouvé
        tprint(f"⚠️ [ConfigManager] Configuration non trouvée pour le modèle: {model_type}", color="yellow")
        return {
            "enabled": True,
            "hpo": {"enabled": True, "n_trials": 50, "timeout_seconds": 300}
        }
    
    def get_hpo_config(self, config: Dict[str, Any], model_type: str = None) -> Dict[str, Any]:
        """
        Extraire la configuration HPO
        
        Args:
            config: Configuration complète
            model_type: Type de modèle spécifique (optionnel)
            
        Returns:
            Configuration HPO
        """
        hpo_config = config.get("hpo", {})
        
        if model_type:
            # Configuration HPO spécifique au modèle
            model_config = self.get_model_config(config, model_type)
            model_hpo = model_config.get("hpo", {})
            
            # Fusionner avec la configuration HPO globale
            merged_hpo = deepcopy(hpo_config)
            merged_hpo.update(model_hpo)
            return merged_hpo
        
        return hpo_config
    
    def validate_for_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valider une configuration pour l'entraînement
        
        Args:
            config: Configuration à valider
            
        Returns:
            Configuration validée avec avertissements et suggestions
        """
        warnings = []
        suggestions = []
        
        # Vérifications spécifiques pour l'entraînement
        models_config = config.get("models", {})
        base_models = models_config.get("base_models", {})
        
        # Vérifier que au moins un modèle est activé
        enabled_models = [name for name, model_config in base_models.items() 
                         if model_config.get("enabled", False)]
        
        if not enabled_models:
            warnings.append("Aucun modèle de base activé dans la configuration")
            suggestions.append("Activer au moins un modèle dans models.base_models")
        
        # Vérifier la configuration HPO
        hpo_config = config.get("hpo", {})
        if hpo_config.get("enabled", False):
            max_trials = hpo_config.get("max_trials", 0)
            if max_trials < 10:
                warnings.append(f"Nombre de trials HPO très faible ({max_trials})")
                suggestions.append("Considérer d'augmenter hpo.max_trials à au moins 25")
            
            timeout = hpo_config.get("timeout_seconds", 0)
            if timeout < 60:
                warnings.append(f"Timeout HPO très court ({timeout}s)")
                suggestions.append("Considérer d'augmenter hpo.timeout_seconds à au moins 120s")
        
        # Vérifier les ressources système
        system_config = config.get("system_resources", {})
        n_jobs = system_config.get("n_jobs", -1)
        if n_jobs == -1:
            suggestions.append("Considérer de limiter n_jobs pour éviter la surcharge CPU")
        
        return {
            "config": config,
            "warnings": warnings,
            "suggestions": suggestions,
            "enabled_models": enabled_models,
            "ready_for_training": len(warnings) == 0
        }


# Fonction d'interface simplifiée
def load_regime_training_config(
    config_name: str = None,
    config_path: str = None,
    custom_overrides: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Interface simplifiée pour charger une configuration de training
    
    Args:
        config_name: Nom de la configuration (ex: "production", "development")
        config_path: Chemin vers un fichier spécifique
        custom_overrides: Modifications à appliquer
        
    Returns:
        Configuration validée et prête pour l'entraînement
    """
    manager = RegimeModelsTrainingConfigManager()
    
    # Charger la configuration
    if custom_overrides:
        config = manager.create_custom_config(
            base_config=config_name or "default",
            overrides=custom_overrides,
            config_name="temp_custom"
        )
    else:
        config = manager.load_config(
            config_name=config_name,
            config_path=config_path,
            merge_with_default=True,
            validate=True
        )
    
    # Validation finale pour l'entraînement
    validation_result = manager.validate_for_training(config)
    
    if validation_result["warnings"]:
        tprint(f"⚠️ [Config] Avertissements détectés:", color="yellow")
        for warning in validation_result["warnings"]:
            tprint(f"   - {warning}", color="yellow")
    
    if validation_result["suggestions"]:
        tprint(f"💡 [Config] Suggestions:", color="blue")
        for suggestion in validation_result["suggestions"]:
            tprint(f"   - {suggestion}", color="blue")
    
    return validation_result["config"]


# Export des classes et fonctions principales
__all__ = [
    "RegimeModelsTrainingConfigManager",
    "ConfigValidationError", 
    "load_regime_training_config"
]