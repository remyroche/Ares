# Exemples de Configurations Personnalisées

## Configuration de Développement Rapide

Ce fichier JSON utilise moins de trials HPO pour des itérations plus rapides.

```json
{
  "general": {
    "component_name": "regime_models_training",
    "version": "2.0.0",
    "description": "Configuration pour développement rapide"
  },
  "models": {
    "base_models": {
      "catboost": {
        "enabled": true,
        "iterations": 50,
        "depth": 4,
        "learning_rate": 0.15,
        "hpo": {
          "enabled": true,
          "n_trials": 15,
          "timeout_seconds": 120
        }
      },
      "lightgbm": {
        "enabled": true,
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": 0.15,
        "hpo": {
          "enabled": true,
          "n_trials": 15,
          "timeout_seconds": 120
        }
      }
    },
    "meta_learner": {
      "enabled": true,
      "hpo": {
        "enabled": true,
        "n_trials": 10,
        "timeout_seconds": 60
      }
    }
  },
  "hpo": {
    "enabled": true,
    "max_trials": 15,
    "timeout_seconds": 120
  }
}
```

## Configuration de Production Optimisée

Cette configuration est optimisée pour la production avec plus de trials HPO et des paramètres affinés.

```json
{
  "general": {
    "component_name": "regime_models_training",
    "version": "2.0.0",
    "description": "Configuration optimisée pour production"
  },
  "models": {
    "base_models": {
      "catboost": {
        "enabled": true,
        "iterations": 200,
        "depth": 8,
        "learning_rate": 0.05,
        "l2_leaf_reg": 5.0,
        "hpo": {
          "enabled": true,
          "n_trials": 100,
          "timeout_seconds": 600
        }
      },
      "xgboost": {
        "enabled": true,
        "n_estimators": 200,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "hpo": {
          "enabled": true,
          "n_trials": 100,
          "timeout_seconds": 600
        }
      },
      "lightgbm": {
        "enabled": true,
        "num_leaves": 63,
        "max_depth": 8,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "hpo": {
          "enabled": true,
          "n_trials": 100,
          "timeout_seconds": 600
        }
      }
    },
    "meta_learner": {
      "enabled": true,
      "num_leaves": 127,
      "max_depth": 10,
      "learning_rate": 0.03,
      "n_estimators": 300,
      "hpo": {
        "enabled": true,
        "n_trials": 75,
        "timeout_seconds": 480
      }
    }
  },
  "hpo": {
    "enabled": true,
    "max_trials": 100,
    "timeout_seconds": 600,
    "early_stopping": true,
    "enable_pruning": true,
    "multi_objective": true
  },
  "feature_engineering": {
    "enabled": true,
    "feature_selection": {
      "enabled": true,
      "method": "permutation_importance_rfe",
      "target_feature_count": 100,
      "permutation_n_repeats": 5,
      "tscv_splits": 5
    }
  }
}
```

## Configuration de Recherche Experimentation

Cette configuration est adaptée pour l'expérimentation avec des paramètres plus exploratoires.

```json
{
  "general": {
    "component_name": "regime_models_training",
    "version": "2.0.0",
    "description": "Configuration pour recherche et expérimentation"
  },
  "models": {
    "base_models": {
      "catboost": {
        "enabled": true,
        "iterations": 300,
        "depth": 10,
        "learning_rate": 0.03,
        "l2_leaf_reg": 10.0,
        "hpo": {
          "enabled": true,
          "n_trials": 150,
          "timeout_seconds": 900
        }
      },
      "xgboost": {
        "enabled": true,
        "n_estimators": 300,
        "max_depth": 10,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "hpo": {
          "enabled": true,
          "n_trials": 150,
          "timeout_seconds": 900
        }
      },
      "lightgbm": {
        "enabled": true,
        "num_leaves": 127,
        "max_depth": 10,
        "learning_rate": 0.03,
        "n_estimators": 300,
        "hpo": {
          "enabled": true,
          "n_trials": 150,
          "timeout_seconds": 900
        }
      },
      "extratrees": {
        "enabled": true,
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 3,
        "min_samples_leaf": 3,
        "hpo": {
          "enabled": true,
          "n_trials": 100,
          "timeout_seconds": 600
        }
      }
    },
    "meta_learner": {
      "enabled": true,
      "num_leaves": 255,
      "max_depth": 12,
      "learning_rate": 0.02,
      "n_estimators": 500,
      "hpo": {
        "enabled": true,
        "n_trials": 100,
        "timeout_seconds": 600
      }
    }
  },
  "hpo": {
    "enabled": true,
    "max_trials": 150,
    "timeout_seconds": 900,
    "early_stopping": true,
    "enable_pruning": true,
    "multi_objective": true,
    "use_pareto_optimization": true,
    "hierarchical_optimization": true
  },
  "feature_engineering": {
    "enabled": true,
    "advanced_regime_features": {
      "enabled": true,
      "window_sizes": [3, 5, 7, 9, 11],
      "enable_smoothed_features": true
    },
    "feature_selection": {
      "enabled": true,
      "method": "permutation_importance_rfe",
      "target_feature_count": 120,
      "permutation_n_repeats": 7,
      "tscv_splits": 7
    }
  },
  "temporal_validation": {
    "enabled": true,
    "n_splits": 10,
    "test_size": 0.15
  }
}
```

## Comment Utiliser les Configurations Personnalisées

### 1. Charger une Configuration Personnalisée

```python
from src.config.regime_models_training import load_regime_training_config

# Charger une configuration personnalisée
custom_config = load_regime_training_config(
    config_name="production",
    config_path="path/to/production_config.json"
)
```

### 2. Créer une Configuration Personnalisée au Volant

```python
from src.config.regime_models_training import RegimeModelsTrainingConfigManager

# Créer une configuration personnalisée avec des overrides
manager = RegimeModelsTrainingConfigManager()

overrides = {
    "hpo": {
        "max_trials": 50,
        "timeout_seconds": 300
    },
    "models": {
        "base_models": {
            "catboost": {
                "enabled": True,
                "iterations": 100,
                "hpo": {
                    "enabled": True,
                    "n_trials": 30
                }
            }
        }
    }
}

custom_config = manager.create_custom_config(
    base_config="default",
    overrides=overrides,
    config_name="my_custom_config"
)

# Sauvegarder la configuration
config_file = manager.save_config(custom_config, "my_custom_config", "yaml")
print(f"Configuration sauvegardée dans: {config_file}")
```

### 3. Valider une Configuration

```python
validation_result = manager.validate_for_training(custom_config)

if validation_result["ready_for_training"]:
    print("✅ Configuration valide pour l'entraînement")
else:
    print("⚠️ Avertissements:", validation_result["warnings"])
    print("💡 Suggestions:", validation_result["suggestions"])
```

### 4. Lister les Configurations Disponibles

```python
available_configs = manager.list_available_configs()
for config_info in available_configs:
    print(f"📁 {config_info['name']} ({config_info['format']}) - {config_info['size']} bytes")
```

## Avantages du Système de Configuration Centralisée

1. **Flexibilité**: Changer les paramètres sans modifier le code
2. **Validation**: Vérification automatique des paramètres
3. **Héritage**: Système de fallback vers les valeurs par défaut
4. **Portabilité**: Partage facile des configurations entre environnements
5. **Extensibilité**: Ajout facile de nouveaux paramètres et modèles
6. **Gestion d'Environnement**: Configurations spécifiques par environnement (dev/prod/test)