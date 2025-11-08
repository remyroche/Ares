"""
Configuration par défaut pour le regime_models_training

Ce module fournit la configuration par défaut sous forme de constantes Python
qui peut être utilisée programmativement.
"""

from typing import Dict, Any

# Configuration par défaut complète
DEFAULT_CONFIG: Dict[str, Any] = {
    # Configuration générale
    "general": {
        "component_name": "regime_models_training",
        "version": "2.0.0",
        "description": "Configuration centralisée pour l'entraînement des modèles de détection de régime"
    },
    
    # Paramètres de validation des données
    "data_validation": {
        "min_samples": 10,
        "min_features": 50,
        "required_columns": ["close", "open", "high", "low", "volume"],
        "max_nan_ratio": 0.1,
        "enable_data_quality_checks": True
    },
    
    # Paramètres d'extraction des régimes
    "regime_extraction": {
        "min_regimes": 2,
        "max_regimes": 10,
        "min_samples_per_regime": 5,
        "extraction_method": "standardized",  # standardized, hdp_hmm, hdbscan
        "fallback_to_synthetic": True
    },
    
    # Paramètres de validation temporelle
    "temporal_validation": {
        "enabled": True,
        "strict_temporal_order": True,
        "initial_train_size": 0.6,
        "step_size": 0.1,
        "min_test_size": 0.1,
        "enable_leakage_detection": True,
        "n_splits": 5,
        "test_size": 0.2,
        "gap_size": 1
    },
    
    # Configuration des modèles
    "models": {
        # Configuration des modèles de base
        "base_models": {
            "catboost": {
                "enabled": True,
                "iterations": 100,
                "depth": 6,
                "learning_rate": 0.1,
                "l2_leaf_reg": 3.0,
                "subsample": 1.0,
                "colsample_bylevel": 1.0,
                "bootstrap_type": "Bernoulli",
                "task_type": "CPU",
                "random_seed": 42,
                "verbose": False,
                "hpo": {
                    "enabled": True,
                    "n_trials": 75,
                    "timeout_seconds": 300
                }
            },
            
            "xgboost": {
                "enabled": True,
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "gamma": 0,
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": 0,
                "hpo": {
                    "enabled": True,
                    "n_trials": 75,
                    "timeout_seconds": 300
                }
            },
            
            "random_forest": {
                "enabled": True,
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "bootstrap": True,
                "random_state": 42,
                "n_jobs": -1,
                "hpo": {
                    "enabled": True,
                    "n_trials": 75,
                    "timeout_seconds": 300
                }
            },
            
            "extratrees": {
                "enabled": True,
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 5,
                "min_samples_leaf": 5,
                "max_features": "sqrt",
                "random_state": 42,
                "n_jobs": -1,
                "hpo": {
                    "enabled": True,
                    "n_trials": 75,
                    "timeout_seconds": 300
                }
            },
            
            "lightgbm": {
                "enabled": True,
                "num_leaves": 31,
                "max_depth": -1,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_alpha": 0.0,
                "reg_lambda": 0.0,
                "random_state": 42,
                "verbose": -1,
                "hpo": {
                    "enabled": True,
                    "n_trials": 75,
                    "timeout_seconds": 300
                }
            }
        },
        
        # Configuration du meta-learner
        "meta_learner": {
            "enabled": True,
            "name": "stacker_lgbm_calibrated",
            "num_leaves": 63,
            "max_depth": 8,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "min_child_samples": 50,
            "min_data_in_leaf": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "class_weight": "balanced",
            "random_state": 42,
            "verbose": -1,
            "hpo": {
                "enabled": True,
                "n_trials": 50,
                "timeout_seconds": 240
            }
        }
    },
    
    # Configuration de l'optimisation des hyperparamètres
    "hpo": {
        "enabled": True,
        "method": "bayesian",  # bayesian, grid, random
        "max_trials": 75,
        "timeout_seconds": 300,
        "early_stopping": True,
        "enable_pruning": True,
        "multi_objective": True,
        "use_pareto_optimization": True,
        "hierarchical_optimization": True
    },
    
    # Configuration de la validation des modèles
    "model_validation": {
        "enabled": True,
        "cv_folds": 5,
        "scoring_metrics": ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"],
        "temporal_smoothing": True,
        "smoothing_alpha": 0.1,
        "enable_soft_labels": True,
        "soft_label_smoothing": 0.1
    },
    
    # Configuration des caractéristiques des données
    "data_preparation": {
        "enable_feature_scaling": True,
        "scaling_method": "standard",  # standard, robust, minmax
        "handle_missing_values": "mean",  # mean, median, drop
        "remove_outliers": True,
        "outlier_method": "iqr",
        "iqr_multiplier": 3.0
    },
    
    # Configuration des caractéristiques d'ingénierie
    "feature_engineering": {
        "enabled": True,
        "use_feature_bank": True,
        "categories": [
            "REGIME",
            "MOMENTUM", 
            "VOLATILITY",
            "VOLUME",
            "TREND",
            "OSCILLATOR",
            "RETURNS",
            "MICROSTRUCTURE"
        ],
        
        # Configuration des caractéristiques avancées de régime
        "advanced_regime_features": {
            "enabled": True,
            "window_sizes": [4, 8, 16, 24],
            "enable_smoothed_features": True
        },
        
        # Sélection de caractéristiques
        "feature_selection": {
            "enabled": True,
            "method": "permutation_importance_rfe",
            "target_feature_count": 80,
            "permutation_n_repeats": 3,
            "tscv_splits": 3
        }
    },
    
    # Configuration de la protection contre les biais
    "data_protection": {
        "lookahead_protection": {
            "enabled": True,
            "automated_filtering": True
        },
        
        "memory_management": {
            "enabled": True,
            "auto_cleanup": True,
            "cleanup_on_error": True,
            "alert_threshold": 85.0
        }
    },
    
    # Configuration de l'optimisation matérielle
    "hardware_optimization": {
        "enabled": True,
        "cpu_optimization_level": "aggressive",
        "gpu_optimization_level": "balanced",
        "memory_optimization_level": "balanced",
        "enable_adaptive_optimization": True,
        "enable_learning": True
    },
    
    # Configuration des ressources système
    "system_resources": {
        "n_jobs": -1,  # Utiliser tous les CPU disponibles
        "memory_limit_gb": 8.0,
        "timeout_seconds": 1800
    },
    
    # Configuration de l'évaluation
    "evaluation": {
        "enhanced_evaluation": True,
        "temporal_metrics": True,
        "regime_persistence_metrics": True,
        "ensemble_evaluation": True
    },
    
    # Configuration des métriques
    "metrics": {
        "primary_metric": "accuracy",
        "secondary_metrics": [
            "precision_weighted",
            "recall_weighted", 
            "f1_weighted",
            "balanced_accuracy"
        ]
    },
    
    # Configuration de la journalisation
    "logging": {
        "level": "INFO",
        "enable_performance_logging": True,
        "enable_memory_monitoring": True,
        "save_training_history": True
    },
    
    # Configuration de la sortie des résultats
    "output": {
        "save_artifacts": True,
        "artifact_types": [
            "trained_models",
            "model_metrics",
            "feature_importance",
            "training_history",
            "validation_report"
        ]
    },
    
    # Configuration de compatibilité
    "compatibility": {
        "ml_libraries_required": [
            "scikit-learn",
            "lightgbm",
            "catboost",
            "xgboost",
            "imodels"
        ],
        "fallback_mode": True,
        "simulate_missing_libraries": True
    }
}


def create_default_regime_training_config() -> Dict[str, Any]:
    """
    Créer la configuration par défaut pour l'entraînement des régimes.
    
    Returns:
        Configuration par défaut
    """
    return DEFAULT_CONFIG.copy()


def validate_regime_training_config(config: Dict[str, Any], strict: bool = False) -> Dict[str, Any]:
    """
    Valider la configuration d'entraînement des régimes.
    
    Args:
        config: Configuration à valider
        strict: Mode strict pour la validation
        
    Returns:
        Configuration validée
        
    Raises:
        ValueError: Si la configuration est invalide
    """
    required_sections = ["general", "models", "data_validation", "hpo"]
    
    # Vérifier les sections requises
    for section in required_sections:
        if section not in config:
            if strict:
                raise ValueError(f"Section requise manquante: {section}")
            else:
                # Ajouter la section par défaut
                config[section] = DEFAULT_CONFIG[section]
    
    # Validation des champs spécifiques
    if "data_validation" in config:
        dv_config = config["data_validation"]
        if "min_samples" in dv_config and dv_config["min_samples"] < 1:
            raise ValueError("min_samples doit être >= 1")
        if "min_features" in dv_config and dv_config["min_features"] < 1:
            raise ValueError("min_features doit être >= 1")
    
    if "hpo" in config:
        hpo_config = config["hpo"]
        if "max_trials" in hpo_config and (hpo_config["max_trials"] < 1 or hpo_config["max_trials"] > 1000):
            raise ValueError("max_trials doit être entre 1 et 1000")
        if "timeout_seconds" in hpo_config and hpo_config["timeout_seconds"] < 1:
            raise ValueError("timeout_seconds doit être >= 1")
    
    return config


# Export des fonctions et constantes principales
__all__ = [
    "DEFAULT_CONFIG",
    "create_default_regime_training_config", 
    "validate_regime_training_config"
]