"""
Configuration Manager pour Regime Ensemble Training Component

Ce gestionnaire centralise la configuration du composant regime_ensemble_training,
permettant une gestion flexible et maintenable des paramètres d'entraînement
d'ensemble (meta-learner) pour la détection de régimes.

Fonctionnalités :
- Support multi-format (YAML, JSON, Python)
- Validation automatique avec feedback détaillé
- Système de fallback intelligent (custom → défaut → hardcodé)
- Configuration inheritance et override
- Paramètres par catégorie (hardware, hpo, ensemble, validation, etc.)
- Zero-downtime configuration updates
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
import warnings

# Configuration schemas
from typing_extensions import TypedDict, Literal

# Import des types pour validation
try:
    from typing import TypedDict, Literal, Union
except ImportError:
    TypedDict = dict
    Literal = str
    Union = lambda x, y: str


# Schema definitions
class HardwareConfigSchema(TypedDict, total=False):
    """Configuration du gestionnaire matériel."""
    cpu_optimization_level: Literal['minimal', 'balanced', 'aggressive', 'extreme']
    gpu_optimization_level: Literal['disabled', 'minimal', 'balanced', 'aggressive']
    memory_optimization_level: Literal['minimal', 'balanced', 'aggressive']
    enable_adaptive_optimization: bool
    enable_learning: bool


class HPOConfigSchema(TypedDict, total=False):
    """Configuration de l'optimisation d'hyperparamètres."""
    max_trials: int
    timeout_seconds: int
    enable_early_stopping: bool
    enable_pruning: bool
    n_trials: int  # Pour meta-learner
    cv_folds: int
    enable_multi_objective_hpo: bool
    use_pareto_optimization: bool
    use_hierarchical_hpo: bool


class EnsembleConfigSchema(TypedDict, total=False):
    """Configuration spécifique de l'ensemble training."""
    n_estimators: int
    max_depth: int
    learning_rate: float
    random_state: int
    n_jobs: int
    verbose: int
    calibration_method: Literal['isotonic', 'sigmoid', 'none']
    cv_folds: int
    enable_temporal_smoothing: bool
    temporal_smoothing_alpha: float
    enable_soft_labels: bool
    soft_label_smoothing: float
    enable_smoothed_features: bool
    smoothing_window_sizes: List[int]


class FeatureGenerationSchema(TypedDict, total=False):
    """Configuration de la génération de caractéristiques."""
    min_features_required: int
    categories: List[str]
    memory_budget_mb: float
    time_budget_seconds: float
    precision_requirement: Literal['low', 'medium', 'high']
    enable_vectorization: bool


class ModelValidationSchema(TypedDict, total=False):
    """Configuration de la validation des modèles."""
    enable_purged_cv: bool
    enable_data_leakage_detection: bool
    enable_time_series_validation: bool
    enable_shap_analysis: bool
    enable_lime_analysis: bool


class TemporalValidationSchema(TypedDict, total=False):
    """Configuration de la validation temporelle."""
    enable_temporal_checks: bool
    strict_temporal_order: bool
    initial_train_size: float
    test_size: float
    gap_size: int


class BaseModelsConfigSchema(TypedDict, total=False):
    """Configuration des modèles de base."""
    catboost_iterations: int
    catboost_depth: int
    catboost_learning_rate: float
    rf_n_estimators: int
    rf_max_depth: int
    et_n_estimators: int
    et_max_depth: int
    enable_catboost: bool
    enable_random_forest: bool
    enable_extra_trees: bool


@dataclass
class RegimeEnsembleTrainingConfig:
    """
    Configuration complète pour le composant Regime Ensemble Training.
    
    Cette configuration centralise tous les paramètres nécessaires pour
    l'entraînement des modèles d'ensemble (meta-learners) pour la détection
    de régimes.
    """
    
    # === INFORMATIONS GÉNÉRALES ===
    version: str = "2.0.0"
    component_name: str = "regime_ensemble_training"
    description: str = "Configuration centralisée pour l'entraînement d'ensemble de détection de régimes"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # === CONFIGURATION MATÉRIEL ===
    hardware: HardwareConfigSchema = field(default_factory=lambda: HardwareConfigSchema(
        cpu_optimization_level="aggressive",
        gpu_optimization_level="balanced", 
        memory_optimization_level="balanced",
        enable_adaptive_optimization=True,
        enable_learning=True
    ))
    
    # === OPTIMISATION HYPERPARAMÈTRES ===
    hpo: HPOConfigSchema = field(default_factory=lambda: HPOConfigSchema(
        max_trials=50,
        timeout_seconds=300,
        enable_early_stopping=True,
        enable_pruning=True,
        n_trials=75,  # Meta-learner optimization
        cv_folds=3,
        enable_multi_objective_hpo=True,
        use_pareto_optimization=True,
        use_hierarchical_hpo=True
    ))
    
    # === CONFIGURATION ENSEMBLE ===
    ensemble: EnsembleConfigSchema = field(default_factory=lambda: EnsembleConfigSchema(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        calibration_method="isotonic",
        cv_folds=3,
        enable_temporal_smoothing=True,
        temporal_smoothing_alpha=0.1,
        enable_soft_labels=True,
        soft_label_smoothing=0.1,
        enable_smoothed_features=True,
        smoothing_window_sizes=[3, 5, 7]
    ))
    
    # === GÉNÉRATION CARACTÉRISTIQUES ===
    feature_generation: FeatureGenerationSchema = field(default_factory=lambda: FeatureGenerationSchema(
        min_features_required=50,
        categories=["momentum", "volatility", "volume", "trend", "oscillator", "returns", "microstructure"],
        memory_budget_mb=2048.0,
        time_budget_seconds=300.0,
        precision_requirement="high",
        enable_vectorization=True
    ))
    
    # === VALIDATION MODÈLES ===
    model_validation: ModelValidationSchema = field(default_factory=lambda: ModelValidationSchema(
        enable_purged_cv=True,
        enable_data_leakage_detection=True,
        enable_time_series_validation=True,
        enable_shap_analysis=True,
        enable_lime_analysis=True
    ))
    
    # === VALIDATION TEMPORELLE ===
    temporal_validation: TemporalValidationSchema = field(default_factory=lambda: TemporalValidationSchema(
        enable_temporal_checks=True,
        strict_temporal_order=True,
        initial_train_size=0.7,
        test_size=0.3,
        gap_size=1
    ))
    
    # === MODÈLES DE BASE ===
    base_models: BaseModelsConfigSchema = field(default_factory=lambda: BaseModelsConfigSchema(
        catboost_iterations=100,
        catboost_depth=6,
        catboost_learning_rate=0.1,
        rf_n_estimators=100,
        rf_max_depth=10,
        et_n_estimators=100,
        et_max_depth=10,
        enable_catboost=True,
        enable_random_forest=True,
        enable_extra_trees=True
    ))
    
    # === MÉTA-FONCTIONNALITÉS ===
    enable_enhanced_meta_features: bool = True
    enable_uncertainty_quantification: bool = True
    enable_confidence_features: bool = True
    enable_disagreement_analysis: bool = True
    enable_regime_transition_features: bool = True
    
    # === ARTIFACTS ET SORTIES ===
    save_individual_artifacts: bool = True
    create_timeframe_artifacts: bool = True
    tag_dataset_with_outputs: bool = True
    generate_probability_reports: bool = True
    enable_downstream_compatibility: bool = True
    
    # === PERFORMANCE ET MONITORING ===
    enable_performance_monitoring: bool = True
    enable_hardware_optimization: bool = True
    enable_lookahead_protection: bool = True
    memory_limit_mb: float = 8192.0
    timeout_seconds: int = 3600
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir la configuration en dictionnaire."""
        return asdict(self)
    
    def update(self, **kwargs) -> 'RegimeEnsembleTrainingConfig':
        """Mettre à jour la configuration avec de nouvelles valeurs."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # Mise à jour des configurations imbriquées
                if key in ['hardware', 'hpo', 'ensemble', 'feature_generation', 
                          'model_validation', 'temporal_validation', 'base_models']:
                    if hasattr(self, key):
                        getattr(self, key).update(value)
        
        self.last_updated = datetime.now().isoformat()
        return self
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Valider la configuration.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        errors = []
        
        # Validation des hyperparamètres ensemble
        ensemble = self.ensemble
        if ensemble.get('n_estimators', 0) <= 0:
            errors.append("ensemble.n_estimators doit être > 0")
        if ensemble.get('learning_rate', 0) <= 0 or ensemble.get('learning_rate', 1) > 1:
            errors.append("ensemble.learning_rate doit être dans (0, 1]")
        if ensemble.get('max_depth', 0) <= 0:
            errors.append("ensemble.max_depth doit être > 0")
        
        # Validation HPO
        hpo = self.hpo
        if hpo.get('max_trials', 0) <= 0:
            errors.append("hpo.max_trials doit être > 0")
        if hpo.get('timeout_seconds', 0) <= 0:
            errors.append("hpo.timeout_seconds doit être > 0")
        
        # Validation feature generation
        fg = self.feature_generation
        if fg.get('min_features_required', 0) <= 0:
            errors.append("feature_generation.min_features_required doit être > 0")
        if fg.get('memory_budget_mb', 0) <= 0:
            errors.append("feature_generation.memory_budget_mb doit être > 0")
        
        # Validation temporal validation
        tv = self.temporal_validation
        if not (0 < tv.get('initial_train_size', 0) < 1):
            errors.append("temporal_validation.initial_train_size doit être dans (0, 1)")
        if not (0 < tv.get('test_size', 0) < 1):
            errors.append("temporal_validation.test_size doit être dans (0, 1)")
        
        return len(errors) == 0, errors


class RegimeEnsembleTrainingConfigManager:
    """
    Gestionnaire de configuration centralisé pour Regime Ensemble Training.
    
    Ce gestionnaire fournit :
    - Chargement multi-format (YAML, JSON, Python)
    - Validation automatique avec feedback détaillé
    - Système de fallback intelligent
    - Configuration inheritance et override
    - Mise à jour temps-réel sans restart
    """
    
    def __init__(self, 
                 config_dir: Optional[str] = None,
                 custom_config_path: Optional[str] = None,
                 fallback_to_hardcoded: bool = True,
                 enable_validation: bool = True,
                 enable_caching: bool = True):
        """
        Initialiser le gestionnaire de configuration.
        
        Args:
            config_dir: Répertoire des configurations (défaut: auto-détecté)
            custom_config_path: Chemin vers configuration personnalisée
            fallback_to_hardcoded: Activer fallback vers valeurs hardcodées
            enable_validation: Activer validation automatique
            enable_caching: Activer mise en cache des configurations
        """
        self.logger = logging.getLogger(f"{__name__}.RegimeEnsembleTrainingConfigManager")
        
        # Configuration des répertoires
        if config_dir is None:
            # Auto-détection du répertoire de configuration
            current_dir = Path(__file__).parent
            self.config_dir = current_dir
        else:
            self.config_dir = Path(config_dir)
        
        self.custom_config_path = Path(custom_config_path) if custom_config_path else None
        self.fallback_to_hardcoded = fallback_to_hardcoded
        self.enable_validation = enable_validation
        self.enable_caching = enable_caching
        
        # Cache des configurations
        self._config_cache = {}
        self._last_load_time = {}
        
        # Configuration par défaut
        self._default_config = RegimeEnsembleTrainingConfig()
        
        self.logger.info(f"RegimeEnsembleTrainingConfigManager initialisé")
        self.logger.info(f"Répertoire config: {self.config_dir}")
        self.logger.info(f"Config personnalisée: {self.custom_config_path}")
        self.logger.info(f"Fallback hardcodé: {self.fallback_to_hardcoded}")
    
    def get_config(self, config_name: str = "default", **overrides) -> RegimeEnsembleTrainingConfig:
        """
        Obtenir une configuration avec fallback intelligent.
        
        Args:
            config_name: Nom de la configuration (default, production, development, etc.)
            **overrides: Paramètres à surcharger
            
        Returns:
            Configuration validée et prête à l'utilisation
        """
        cache_key = f"{config_name}_{hash(str(sorted(overrides.items())))}"
        
        # Vérifier le cache
        if self.enable_caching and cache_key in self._config_cache:
            cached_config, cached_time = self._config_cache[cache_key]
            # Cache valide pendant 5 minutes
            if datetime.now().timestamp() - cached_time < 300:
                self.logger.debug(f"Configuration {config_name} récupérée du cache")
                return cached_config
        
        # Charger la configuration
        config = self._load_config_with_fallback(config_name)
        
        if config is None:
            if self.fallback_to_hardcoded:
                self.logger.warning(f"Configuration {config_name} non trouvée, utilisation des valeurs hardcodées")
                config = self._default_config
            else:
                raise ValueError(f"Configuration {config_name} non trouvée et fallback désactivé")
        
        # Appliquer les overrides
        if overrides:
            config.update(**overrides)
        
        # Validation
        if self.enable_validation:
            is_valid, errors = config.validate()
            if not is_valid:
                error_msg = f"Configuration {config_name} invalide: {'; '.join(errors)}"
                self.logger.error(error_msg)
                if self.fallback_to_hardcoded:
                    self.logger.warning("Utilisation de la configuration par défaut suite à erreur de validation")
                    config = self._default_config
                else:
                    raise ValueError(error_msg)
        
        # Mettre en cache
        if self.enable_caching:
            self._config_cache[cache_key] = (config, datetime.now().timestamp())
        
        self.logger.info(f"Configuration {config_name} chargée avec succès")
        return config
    
    def _load_config_with_fallback(self, config_name: str) -> Optional[RegimeEnsembleTrainingConfig]:
        """
        Charger une configuration avec système de fallback.
        
        Args:
            config_name: Nom de la configuration
            
        Returns:
            Configuration chargée ou None si échec
        """
        # 1. Essayer configuration personnalisée
        if self.custom_config_path and self.custom_config_path.exists():
            try:
                config = self._load_from_file(self.custom_config_path)
                if config:
                    self.logger.info(f"Configuration {config_name} chargée depuis {self.custom_config_path}")
                    return config
            except Exception as e:
                self.logger.warning(f"Échec chargement config personnalisée: {e}")
        
        # 2. Essayer dans le répertoire de config
        config_file = self._find_config_file(config_name)
        if config_file:
            try:
                config = self._load_from_file(config_file)
                if config:
                    self.logger.info(f"Configuration {config_name} chargée depuis {config_file}")
                    return config
            except Exception as e:
                self.logger.warning(f"Échec chargement config {config_file}: {e}")
        
        # 3. Essayer les formats par défaut
        for format_name in ['default', 'production', 'development']:
            config_file = self._find_default_config_file(format_name)
            if config_file:
                try:
                    config = self._load_from_file(config_file)
                    if config:
                        self.logger.info(f"Configuration {config_name} chargée depuis {config_file}")
                        return config
                except Exception as e:
                    self.logger.warning(f"Échec chargement config par défaut {config_file}: {e}")
        
        return None
    
    def _find_config_file(self, config_name: str) -> Optional[Path]:
        """Trouver un fichier de configuration pour le nom donné."""
        # Extensions supportées
        extensions = ['.yaml', '.yml', '.json', '.py']
        
        # Patterns de recherche
        patterns = [
            f"{config_name}.yaml",
            f"{config_name}.yml", 
            f"{config_name}.json",
            f"{config_name}.py",
            f"{config_name}_config.yaml",
            f"{config_name}_config.yml",
            f"{config_name}_config.json",
            f"{config_name}_config.py"
        ]
        
        # Chercher dans le répertoire de config
        for pattern in patterns:
            config_file = self.config_dir / pattern
            if config_file.exists():
                return config_file
        
        return None
    
    def _find_default_config_file(self, format_name: str) -> Optional[Path]:
        """Trouver un fichier de configuration par défaut."""
        default_files = [
            f"default_config.yaml",
            f"default_config.yml",
            f"default_config.json", 
            f"default_config.py"
        ]
        
        for default_file in default_files:
            config_file = self.config_dir / default_file
            if config_file.exists():
                return config_file
        
        return None
    
    def _load_from_file(self, config_file: Path) -> Optional[RegimeEnsembleTrainingConfig]:
        """
        Charger une configuration depuis un fichier.
        
        Args:
            config_file: Chemin vers le fichier de configuration
            
        Returns:
            Configuration chargée ou None si échec
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            suffix = config_file.suffix.lower()
            
            if suffix in ['.yaml', '.yml']:
                config_data = yaml.safe_load(content)
            elif suffix == '.json':
                config_data = json.loads(content)
            elif suffix == '.py':
                # Exécution sécurisée du module Python
                config_data = self._execute_python_config(content)
            else:
                raise ValueError(f"Format de fichier non supporté: {suffix}")
            
            if config_data is None:
                return None
            
            # Créer la configuration
            return self._create_config_from_dict(config_data)
            
        except Exception as e:
            self.logger.error(f"Erreur chargement fichier {config_file}: {e}")
            return None
    
    def _execute_python_config(self, content: str) -> Optional[Dict[str, Any]]:
        """Exécuter safely un fichier de configuration Python."""
        try:
            # Créer un namespace isolé
            namespace = {}
            
            # Exécuter le contenu
            exec(content, namespace)
            
            # Chercher les variables de configuration
            config_vars = {}
            for key, value in namespace.items():
                if key.startswith('config_') or key in ['hardware', 'hpo', 'ensemble', 
                                                       'feature_generation', 'model_validation',
                                                       'temporal_validation', 'base_models']:
                    config_vars[key] = value
            
            if not config_vars:
                return None
            
            # Convertir en dictionnaire plat
            result = {}
            for key, value in config_vars.items():
                if isinstance(value, dict):
                    # flatten nested dicts
                    for subkey, subvalue in value.items():
                        result[f"{key}.{subkey}"] = subvalue
                else:
                    result[key] = value
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur exécution config Python: {e}")
            return None
    
    def _create_config_from_dict(self, config_data: Dict[str, Any]) -> RegimeEnsembleTrainingConfig:
        """
        Créer une configuration depuis un dictionnaire.
        
        Args:
            config_data: Données de configuration
            
        Returns:
            Configuration créée
        """
        config = RegimeEnsembleTrainingConfig()
        
        # Mapping des clés pour gérer les structures imbriquées
        nested_keys = ['hardware', 'hpo', 'ensemble', 'feature_generation', 
                      'model_validation', 'temporal_validation', 'base_models']
        
        for key, value in config_data.items():
            if isinstance(value, dict) and key in nested_keys:
                # Configuration imbriquée
                if hasattr(config, key):
                    getattr(config, key).update(value)
                else:
                    setattr(config, key, value)
            else:
                # Attribut direct
                if hasattr(config, key):
                    setattr(config, key, value)
        
        config.last_updated = datetime.now().isoformat()
        return config
    
    def save_config(self, config: RegimeEnsembleTrainingConfig, 
                   config_name: str, format_type: str = "yaml") -> bool:
        """
        Sauvegarder une configuration.
        
        Args:
            config: Configuration à sauvegarder
            config_name: Nom de la configuration
            format_type: Format de sauvegarde ('yaml', 'json', 'python')
            
        Returns:
            True si sauvegarde réussie
        """
        try:
            config_file = self.config_dir / f"{config_name}_config.{format_type}"
            
            if format_type == "yaml":
                with open(config_file, 'w', encoding='utf-8') as f:
                    yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)
            elif format_type == "json":
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config.to_dict(), f, indent=2)
            elif format_type == "python":
                content = self._generate_python_config(config)
                with open(config_file, 'w', encoding='utf-8') as f:
                    f.write(content)
            else:
                raise ValueError(f"Format non supporté: {format_type}")
            
            self.logger.info(f"Configuration {config_name} sauvegardée dans {config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde configuration {config_name}: {e}")
            return False
    
    def _generate_python_config(self, config: RegimeEnsembleTrainingConfig) -> str:
        """Générer un fichier de configuration Python."""
        config_dict = config.to_dict()
        
        lines = [
            '"""',
            f"Configuration générée automatiquement pour {config.component_name}",
            f"Date: {datetime.now().isoformat()}",
            '"""',
            '',
            '# Configuration complète',
            f'config_data = {repr(config_dict)}',
            '',
            '# Configurations par section',
        ]
        
        # Générer les sections individuelles
        sections = ['hardware', 'hpo', 'ensemble', 'feature_generation', 
                   'model_validation', 'temporal_validation', 'base_models']
        
        for section in sections:
            if section in config_dict:
                lines.extend([
                    f'',
                    f'# Configuration {section}',
                    f'{section} = {repr(config_dict[section])}'
                ])
        
        return '\n'.join(lines)
    
    def get_config_info(self) -> Dict[str, Any]:
        """Obtenir des informations sur les configurations disponibles."""
        info = {
            'config_dir': str(self.config_dir),
            'custom_config_path': str(self.custom_config_path) if self.custom_config_path else None,
            'available_configs': [],
            'cached_configs': len(self._config_cache),
            'default_config_version': self._default_config.version,
            'settings': {
                'fallback_to_hardcoded': self.fallback_to_hardcoded,
                'enable_validation': self.enable_validation,
                'enable_caching': self.enable_caching
            }
        }
        
        # Lister les configurations disponibles
        for pattern in ['*.yaml', '*.yml', '*.json', '*.py']:
            for config_file in self.config_dir.glob(pattern):
                if 'default_config' not in config_file.name:
                    info['available_configs'].append(config_file.name)
        
        return info
    
    def clear_cache(self):
        """Vider le cache des configurations."""
        self._config_cache.clear()
        self._last_load_time.clear()
        self.logger.info("Cache des configurations vidé")

    def validate_config(self, config: Any) -> bool:
        """
        Valide une configuration selon le schéma attendu.
        
        Args:
            config: Configuration à valider
            
        Returns:
            bool: True si valide, False sinon
        """
        try:
            # Vérifier que c'est un objet configuration valide
            if not hasattr(config, 'component_name'):
                return False
                
            # Vérifier sections obligatoires
            required_sections = ['hardware', 'hpo', 'ensemble']
            for section in required_sections:
                if not hasattr(config, section):
                    return False
                    
            # Vérifier types de base
            if not isinstance(config.version, str):
                return False
                
            # Vérifier paramètres hardware
            if not hasattr(config.hardware, 'cpu_optimization_level'):
                return False
                
            return True
            
        except Exception:
            return False

    def _get_config_with_fallback(self, custom_config: Dict[str, Any]) -> RegimeEnsembleTrainingConfig:
        """
        Combine configuration personnalisée avec valeurs par défaut et fallback.
        
        Args:
            custom_config: Configuration personnalisée à fusionner
            
        Returns:
            RegimeEnsembleTrainingConfig: Configuration fusionnée
        """
        # Charger configuration par défaut
        default_config = self.get_config()
        
        # Fusionner avec configuration personnalisée
        merged_config = {}
        
        # Copier tous les attributs de la configuration par défaut
        if hasattr(default_config, '_fields'):
            for field in default_config._fields:
                if field in custom_config:
                    # Valeur personnalisée
                    merged_config[field] = custom_config[field]
                else:
                    # Valeur par défaut
                    merged_config[field] = getattr(default_config, field)
        else:
            # Si _fields n'est pas disponible, utiliser to_dict()
            default_dict = default_config.to_dict()
            merged_config = default_dict.copy()
            merged_config.update(custom_config)
            
        # Créer et retourner la configuration fusionnée
        return RegimeEnsembleTrainingConfig(**merged_config)


# Instance globale du gestionnaire
_global_config_manager = None

def get_regime_ensemble_config_manager() -> RegimeEnsembleTrainingConfigManager:
    """
    Obtenir l'instance globale du gestionnaire de configuration.
    
    Returns:
        Gestionnaire de configuration singleton
    """
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = RegimeEnsembleTrainingConfigManager()
    return _global_config_manager

def get_regime_ensemble_config(config_name: str = "default", **overrides) -> RegimeEnsembleTrainingConfig:
    """
    Fonction utilitaire pour obtenir rapidement une configuration.
    
    Args:
        config_name: Nom de la configuration
        **overrides: Paramètres à surcharger
        
    Returns:
        Configuration validée
    """
    manager = get_regime_ensemble_config_manager()
    return manager.get_config(config_name, **overrides)

# Auto-initialisation
try:
    _global_config_manager = RegimeEnsembleTrainingConfigManager()
except Exception as e:
    # Fallback silencieux en cas d'erreur d'initialisation
    _global_config_manager = None