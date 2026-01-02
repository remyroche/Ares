"""
Layer 2.5 Chaser Configuration Template

Add these configuration options to your config to enable the Layer 2.5 Chaser system.
"""

# Layer 2.5 Chaser Configuration
LAYER25_CHASER_CONFIG = {
    # Master Switch
    'layer25_chaser_enabled': False,  # Enable/disable Layer 2.5 Chaser
    
    # Chaser Model Parameters
    'chaser_xgb_params': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1
    },
    
    'chaser_cat_params': {
        'iterations': 200,
        'depth': 6,
        'learning_rate': 0.05,
        'l2_leaf_reg': 3,
        'random_seed': 42,
        'verbose': False,
        'od_type': 'Iter',
        'od_wait': 20
    },
    
    'chaser_ensemble_weights': [0.6, 0.4],  # XGB, CatBoost weights
    
    # Feature Selection
    'chaser_max_features': 50,  # Maximum non-causal features
    'chaser_technical_patterns_only': True,  # Use only technical indicators
    'chaser_exclude_causal_parents': True,  # Exclude causal parent features
    
    # Conflict Detection
    'chaser_conflict_detection_enabled': True,
    'chaser_direction_threshold': 0.0,
    'chaser_magnitude_threshold': 1.0,
    'chaser_confidence_threshold': 0.6,
    'chaser_conflict_intensity_threshold': 0.5,
    
    # Training Parameters
    'chaser_cv_folds': 5,
    'chaser_early_stopping_rounds': 50,
    'chaser_min_samples': 100,
    'chaser_residual_analysis': True,
    
    # Integration
    'chaser_add_to_layer3': True,  # Add Chaser features to Layer 3
    'chaser_conflict_features': True,  # Add conflict detection features
}

# Example usage in your config:
# config.update(LAYER25_CHASER_CONFIG)
# config['layer25_chaser_enabled'] = True  # Enable Chaser

# Minimal configuration to enable Chaser:
MINIMAL_CHASER_CONFIG = {
    'layer25_chaser_enabled': True,
    'chaser_max_features': 30,
    'chaser_conflict_detection_enabled': True
}

# Production configuration (optimized):
PRODUCTION_CHASER_CONFIG = {
    'layer25_chaser_enabled': True,
    'chaser_xgb_params': {
        'n_estimators': 100,  # Faster training
        'max_depth': 4,      # More conservative
        'learning_rate': 0.1,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'random_state': 42,
        'n_jobs': -1,
        'reg_alpha': 0.2,
        'reg_lambda': 0.2
    },
    'chaser_cat_params': {
        'iterations': 100,
        'depth': 4,
        'learning_rate': 0.1,
        'l2_leaf_reg': 5,
        'random_seed': 42,
        'verbose': False
    },
    'chaser_max_features': 25,
    'chaser_cv_folds': 3,  # Faster CV
    'chaser_conflict_detection_enabled': True,
    'chaser_add_to_layer3': True
}

# Research configuration (maximum features):
RESEARCH_CHASER_CONFIG = {
    'layer25_chaser_enabled': True,
    'chaser_xgb_params': {
        'n_estimators': 500,  # More trees
        'max_depth': 8,      # Deeper trees
        'learning_rate': 0.03,  # Slower learning
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'random_state': 42,
        'n_jobs': -1,
        'reg_alpha': 0.05,
        'reg_lambda': 0.05
    },
    'chaser_cat_params': {
        'iterations': 500,
        'depth': 8,
        'learning_rate': 0.03,
        'l2_leaf_reg': 2,
        'random_seed': 42,
        'verbose': False
    },
    'chaser_max_features': 100,  # More features
    'chaser_cv_folds': 10,     # More CV folds
    'chaser_conflict_detection_enabled': True,
    'chaser_residual_analysis': True,
    'chaser_add_to_layer3': True,
    'chaser_conflict_features': True
}

print("Layer 2.5 Chaser configuration templates created")
print("Choose one of the following configurations:")
print("- MINIMAL_CHASER_CONFIG: Basic setup")
print("- PRODUCTION_CHASER_CONFIG: Optimized for production")
print("- RESEARCH_CHASER_CONFIG: Maximum features for research")
