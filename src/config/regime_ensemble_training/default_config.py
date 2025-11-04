"""
Configuration Python par défaut pour Regime Ensemble Training Component

Cette configuration centralise tous les paramètres nécessaires pour
l'entraînement des modèles d'ensemble (meta-learners) pour la détection
de régimes avec validation et fallback intelligent.

Généré automatiquement - Modifiez selon vos besoins spécifiques.
"""

from datetime import datetime

# === INFORMATIONS GÉNÉRALES ===
version = "2.0.0"
component_name = "regime_ensemble_training"
description = "Configuration centralisée pour l'entraînement d'ensemble de détection de régimes"
created_at = "2025-11-03T21:45:36.098Z"
last_updated = "2025-11-03T21:45:36.098Z"

# === CONFIGURATION MATÉRIEL ===
# Paramètres d'optimisation matérielle pour entraînement intensif
hardware = {
    "cpu_optimization_level": "aggressive",      # minimal | balanced | aggressive | extreme
    "gpu_optimization_level": "balanced",        # disabled | minimal | balanced | aggressive  
    "memory_optimization_level": "balanced",     # minimal | balanced | aggressive
    "enable_adaptive_optimization": True,        # Adaptation automatique aux ressources
    "enable_learning": True                      # Apprentissage des patterns d'optimisation
}

# === OPTIMISATION HYPERPARAMÈTRES ===
# Configuration HPO pour meta-learner et modèles de base
hpo = {
    "max_trials": 50,                           # Nombre maximum d'essais HPO
    "timeout_seconds": 300,                     # Timeout global HPO (secondes)
    "enable_early_stopping": True,              # Arrêt précoce si pas d'amélioration
    "enable_pruning": True,                     # Élagage des essais non-prometteurs
    "n_trials": 75,                             # Essais spécifiques meta-learner (augmenté)
    "cv_folds": 3,                              # Cross-validation pour HPO
    "enable_multi_objective_hpo": True,         # HPO multi-objectif (précision + stabilité)
    "use_pareto_optimization": True,            # Optimisation Pareto si disponible
    "use_hierarchical_hpo": True                # HPO hiérarchique pour modèles complexes
}

# === CONFIGURATION ENSEMBLE ===
# Paramètres spécifiques à l'entraînement d'ensemble (meta-learner)
ensemble = {
    # Paramètres LightGBM de base
    "n_estimators": 100,                        # Nombre d'estimateurs
    "max_depth": 6,                             # Profondeur maximale des arbres
    "learning_rate": 0.1,                       # Taux d'apprentissage
    "random_state": 42,                         # Graine aléatoire (reproductibilité)
    "n_jobs": -1,                               # Jobs parallèles (-1 = tous les CPU)
    "verbose": -1,                              # Verbosity LightGBM (-1 = silencieux)
    
    # Configuration calibration
    "calibration_method": "isotonic",           # isotonic | sigmoid | none
    "cv_folds": 3,                              # Folds pour calibration
    
    # Paramètres de lissage temporel
    "enable_temporal_smoothing": True,          # Activer lissage temporel
    "temporal_smoothing_alpha": 0.1,           # Pondération pénalités stabilité
    "enable_soft_labels": True,                 # Utiliser labels flous
    "soft_label_smoothing": 0.1,               # Facteur lissage labels
    
    # Features lissées
    "enable_smoothed_features": True,           # Activer features lissées
    "smoothing_window_sizes": [3, 5, 7]        # Fenêtres de lissage
}

# === GÉNÉRATION CARACTÉRISTIQUES ===
# Configuration génération features via feature bank
feature_generation = {
    "min_features_required": 50,                # Minimum features requises
    "categories": [                             # Catégories de features à générer
        "momentum",                           # Indicateurs momentum
        "volatility",                         # Indicateurs volatilité
        "volume",                             # Indicateurs volume
        "trend",                             # Indicateurs tendance
        "oscillator",                        # Oscillateurs
        "returns",                           # Retours
        "microstructure"                    # Microstructure (sans orderbook)
    ],
    "memory_budget_mb": 2048.0,                # Budget mémoire features (MB)
    "time_budget_seconds": 300.0,              # Budget temps génération (sec)
    "precision_requirement": "high",            # low | medium | high
    "enable_vectorization": True               # Utiliser vectorisation optimisée
}

# === VALIDATION MODÈLES ===
# Configuration validation post-entraînement
model_validation = {
    "enable_purged_cv": True,                   # Cross-validation purgée temporelle
    "enable_data_leakage_detection": True,     # Détection fuite données
    "enable_time_series_validation": True,     # Validation séries temporelles
    "enable_shap_analysis": True,              # Analyse SHAP si disponible
    "enable_lime_analysis": True               # Analyse LIME si disponible
}

# === VALIDATION TEMPORELLE ===
# Configuration validation anti-fuite temporelle
temporal_validation = {
    "enable_temporal_checks": True,             # Vérifications temporelles actives
    "strict_temporal_order": True,              # Ordre temporel strict
    "initial_train_size": 0.7,                  # Proportion données train
    "test_size": 0.3,                          # Proportion données test
    "gap_size": 1                              # Écart minimum train/test
}

# === MODÈLES DE BASE ===
# Configuration modèles de base pour l'ensemble
base_models = {
    # CatBoost Classifier
    "catboost_iterations": 100,                 # Itérations CatBoost
    "catboost_depth": 6,                        # Profondeur CatBoost
    "catboost_learning_rate": 0.1,             # Learning rate CatBoost
    
    # Random Forest Classifier  
    "rf_n_estimators": 100,                    # Estimators Random Forest
    "rf_max_depth": 10,                        # Profondeur max Random Forest
    
    # Extra Trees Classifier
    "et_n_estimators": 100,                    # Estimators Extra Trees
    "et_max_depth": 10,                        # Profondeur max Extra Trees
    
    # Activation modèles
    "enable_catboost": True,                    # Activer CatBoost
    "enable_random_forest": True,               # Activer Random Forest
    "enable_extra_trees": True                  # Activer Extra Trees
}

# === MÉTA-FONCTIONNALITÉS ===
# Fonctionnalités avancées meta-learning
enable_enhanced_meta_features = True        # Meta-features enrichies
enable_uncertainty_quantification = True   # Quantification incertitude
enable_confidence_features = True          # Features confiance
enable_disagreement_analysis = True        # Analyse désaccord modèles
enable_regime_transition_features = True   # Features transitions régimes

# === ARTIFACTS ET SORTIES ===
# Configuration génération et sauvegarde artifacts
save_individual_artifacts = True           # Sauvegarder artifacts individuels
create_timeframe_artifacts = True          # Créer artifacts timeframes (15m/1h)
tag_dataset_with_outputs = True            # Tag dataset avec prédictions
generate_probability_reports = True        # Générer rapports probabilités
enable_downstream_compatibility = True     # Compatibilité downstream

# === PERFORMANCE ET MONITORING ===
# Configuration performance et surveillance
enable_performance_monitoring = True       # Monitoring performance
enable_hardware_optimization = True       # Optimisation matérielle
enable_lookahead_protection = True        # Protection look-ahead
memory_limit_mb = 8192.0                   # Limite mémoire (MB)
timeout_seconds = 3600                     # Timeout global (sec)

# === CONFIGURATION COMPLÈTE ===
# Dictionnaire contenant toutes les configurations
config_data = {
    "version": version,
    "component_name": component_name,
    "description": description,
    "created_at": created_at,
    "last_updated": last_updated,
    "hardware": hardware,
    "hpo": hpo,
    "ensemble": ensemble,
    "feature_generation": feature_generation,
    "model_validation": model_validation,
    "temporal_validation": temporal_validation,
    "base_models": base_models,
    "enable_enhanced_meta_features": enable_enhanced_meta_features,
    "enable_uncertainty_quantification": enable_uncertainty_quantification,
    "enable_confidence_features": enable_confidence_features,
    "enable_disagreement_analysis": enable_disagreement_analysis,
    "enable_regime_transition_features": enable_regime_transition_features,
    "save_individual_artifacts": save_individual_artifacts,
    "create_timeframe_artifacts": create_timeframe_artifacts,
    "tag_dataset_with_outputs": tag_dataset_with_outputs,
    "generate_probability_reports": generate_probability_reports,
    "enable_downstream_compatibility": enable_downstream_compatibility,
    "enable_performance_monitoring": enable_performance_monitoring,
    "enable_hardware_optimization": enable_hardware_optimization,
    "enable_lookahead_protection": enable_lookahead_protection,
    "memory_limit_mb": memory_limit_mb,
    "timeout_seconds": timeout_seconds
}