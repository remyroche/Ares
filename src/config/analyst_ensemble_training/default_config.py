#!/usr/bin/env python3
"""
Configuration Python par défaut pour Analyst Ensemble Training

Ce fichier contient la configuration centralisée pour l'entraînement d'ensemble 
des modèles analyst, au format Python pour une configuration programmatique dynamique.

Version: 1.0.0
Date: 2025-11-03T22:10:00.000Z
"""

from datetime import datetime

# Configuration principale
config_data = {
    "version": "1.0.0",
    "component_name": "analyst_ensemble_training",
    "description": "Configuration centralisée pour l'entraînement d'ensemble des modèles analyst",
    "created_at": "2025-11-03T22:10:00.000Z",
    "last_updated": datetime.now().isoformat(),
    
    # Configuration principale du modèle analyst
    "analyst_config": {
        "model_name": "analyst_ensemble",
        "model_type": "meta_ensemble",
        "target": "trading_signal",
        "base_timeframe": "15m",
        "execution_timeframe": "15m",
        "execution_frequency": "15m",
        "params": {}
    },
    
    # Configuration du meta-learner
    "meta_learner": {
        "model_type": "stacker_lgbm_calibrated",
        "params": {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "num_leaves": 63,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
            "verbosity": -1,
            "n_jobs": -1
        },
        "calibration": {
            "method": "isotonic",
            "enable_temperature_scaling": True
        },
        "hpo": {
            "enabled": True,
            "n_rounds": 2,
            "enable_final_refinement": True,
            "final_refinement_trials": 50,
            "search_space": {
                "max_depth": {
                    "type": "int",
                    "low": 3,
                    "high": 8
                },
                "learning_rate": {
                    "type": "float",
                    "low": 0.01,
                    "high": 0.2,
                    "log": True
                },
                "num_leaves": {
                    "type": "int",
                    "low": 20,
                    "high": 150
                },
                "reg_alpha": {
                    "type": "float",
                    "low": 0.0,
                    "high": 3.0
                },
                "reg_lambda": {
                    "type": "float",
                    "low": 0.0,
                    "high": 3.0
                },
                "subsample": {
                    "type": "float",
                    "low": 0.7,
                    "high": 1.0
                },
                "colsample_bytree": {
                    "type": "float",
                    "low": 0.7,
                    "high": 1.0
                },
                "min_child_samples": {
                    "type": "int",
                    "low": 10,
                    "high": 50
                },
                "min_gain_to_split": {
                    "type": "float",
                    "low": 0.1,
                    "high": 1.0
                }
            },
            "optimal_params": {}
        }
    },
    
    # Configuration des sorties de modèles de base
    "base_model_outputs": {
        "lgbm_output": "analyst_base_lgbm_predictions",
        "tcn_output": "analyst_base_tcn_predictions",
        "catboost_output": "analyst_base_catboost_predictions",
        "meta_learner_output": "analyst_base_meta_learner_predictions"
    },
    
    # Configuration hardware et optimisation
    "hardware": {
        "enable_gpu_acceleration": True,
        "enable_memory_optimization": True,
        "enable_parallel_processing": True,
        "memory_limit_gb": 4.0,
        "max_workers": None,
        "cpu_optimization_level": "aggressive",
        "gpu_optimization_level": "balanced"
    },
    
    # Configuration HPO
    "hpo": {
        "enabled": True,
        "max_trials": 50,
        "timeout_seconds": 300,
        "enable_early_stopping": True,
        "enable_pruning": True,
        "cv_folds": 3
    },
    
    # Configuration de l'entraînement
    "training": {
        "enable_cross_validation": True,
        "cv_folds": 3,
        "enable_early_stopping": True,
        "early_stopping_patience": 15,
        "validation_split": 0.2,
        "test_split": 0.1,
        "training_samples": 15000,
        "validation_samples": 5000,
        "test_samples": 3000
    },
    
    # Configuration de l'ingénierie des features
    "feature_engineering": {
        "primary_features": {
            "source": "feature_generation_final_feature_selection_step",
            "artifact_name": "analyst_features",
            "initial_count": 300,
            "target_count": 100
        },
        "cross_timeframe": {
            "enable": True,
            "base_timeframe": "5m",
            "target_timeframes": ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
            "feature_types": ["technical_indicators", "price_action", "volume_profile", "volatility"]
        },
        "regime_features": {
            "enable": True,
            "source": "regime_ml_models",
            "feature_names": ["regime_prob_0", "regime_prob_1", "regime_prob_2", "regime_prob_3"],
            "include_regime_outputs": True
        },
        "analyst_base_outputs": {
            "enable": True,
            "source": "analyst_base_models",
            "features": [
                "lgbm_predictions",
                "lgbm_patchtst_predictions",
                "catboost_predictions",
                "meta_learner_predictions",
                "base_model_confidence_scores"
            ]
        },
        "feature_selection": {
            "method": "lasso",
            "alpha": 0.01,
            "max_features": 100,
            "enable_recursive_elimination": True,
            "enable_feature_importance": True
        },
        "scaling": {
            "method": "robust",
            "enable_outlier_handling": True,
            "outlier_threshold": 3.0
        }
    },
    
    # Configuration de préparation des données
    "data_preparation": {
        "time_series": {
            "enable_temporal_features": True,
            "lookback_window": 100,
            "forecast_horizon": 1
        },
        "target_generation": {
            "method": "price_momentum",
            "lookback_periods": [5, 10, 20],
            "threshold": 0.02
        }
    },
    
    # Configuration d'évaluation
    "evaluation": {
        "metrics": {
            "regression": ["mse", "mae", "r2", "mape"],
            "classification": ["accuracy", "precision", "recall", "f1"]
        },
        "cross_validation": {
            "method": "time_series_split",
            "n_splits": 5,
            "test_size": 0.2
        },
        "comparison": {
            "enable_model_ranking": True,
            "ranking_metric": "r2",
            "enable_feature_importance": True,
            "enable_model_explanations": True
        }
    },
    
    # Configuration de performance
    "performance": {
        "expected_accuracy": 0.85,
        "expected_diversity_score": 0.92,
        "training_time_limit": 600,
        "memory_limit_mb": 4096
    },
    
    # Configuration de sortie
    "output": {
        "save_models": True,
        "save_predictions": True,
        "generate_reports": True,
        "output_dir": "./analyst_ensemble_models"
    },
    
    # Configuration de logging
    "logging": {
        "level": "INFO",
        "enable_detailed_logging": True,
        "log_predictions": True,
        "log_performance_metrics": True,
        "log_feature_importance": True,
        "log_ensemble_metrics": True,
        "intervals": {
            "training_progress": 50,
            "prediction_summary": 500,
            "performance_update": 2000,
            "ensemble_update": 1000
        }
    },
    
    # Configuration des modèles de base
    "base_models": {
        "lgbm": {
            "enabled": True,
            "params": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "num_leaves": 63,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "verbosity": -1,
                "n_jobs": -1
            }
        },
        "catboost": {
            "enabled": True,
            "params": {
                "iterations": 100,
                "depth": 6,
                "learning_rate": 0.1,
                "l2_leaf_reg": 3,
                "random_seed": 42,
                "verbose": False
            }
        },
        "tcn": {
            "enabled": True,
            "params": {
                "epochs": 50,
                "batch_size": 32,
                "learning_rate": 0.001,
                "dropout": 0.2,
                "hidden_units": 64
            }
        }
    },
    
    # Métadonnées
    "_comment": "Configuration complète pour l'entraînement d'ensemble des modèles analyst",
    "_source": "Migrée depuis analyst_ensemble_config.yaml",
    "_migrated_at": "2025-11-03T22:10:00.000Z"
}

# Configuration des niveaux de logging pour ce module
import logging
from src.utils.initialization_guard import init_guard
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if init_guard.mark_initialized("config.analyst_ensemble_training.default_config"):
    logger.info("✅ Configuration par défaut Analyst Ensemble Training chargée")


def get_default_config():
    """
    Fonction utility pour obtenir la configuration par défaut.
    
    Returns:
        Configuration par défaut sous forme de dictionnaire
    """
    return config_data.copy()


def create_custom_config(overrides=None):
    """
    Créer une configuration personnalisée avec des overrides.
    
    Args:
        overrides: Dictionnaire de paramètres à surcharger
        
    Returns:
        Configuration personnalisée
    """
    if overrides is None:
        overrides = {}
    
    custom_config = config_data.copy()
    custom_config.update(overrides)
    custom_config["last_updated"] = datetime.now().isoformat()
    
    return custom_config


# Auto-test si exécuté directement
if __name__ == "__main__":
    print("🧪 Test de la configuration Python Analyst Ensemble Training")
    print("=" * 70)
    
    try:
        # Test de chargement
        config = get_default_config()
        print(f"✅ Configuration chargée: {config['component_name']}")
        print(f"   Version: {config['version']}")
        print(f"   Description: {config['description']}")
        print(f"   Meta-learner: {config['meta_learner']['model_type']}")
        print(f"   GPU acceleration: {config['hardware']['enable_gpu_acceleration']}")
        
        # Test de création personnalisée
        custom_overrides = {
            "meta_learner": {
                "params": {
                    "n_estimators": 200  # Override pour test
                }
            }
        }
        
        custom_config = create_custom_config(custom_overrides)
        print(f"✅ Configuration personnalisée créée: n_estimators={custom_config['meta_learner']['params']['n_estimators']}")
        
        print("✅ Test réussi!")
        
    except Exception as e:
        print(f"❌ Test échoué: {e}")
        import traceback
        traceback.print_exc()