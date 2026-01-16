"""
Layer 2.5 Chaser Configuration Template

Add these configuration options to your config to enable the Layer 2.5 Chaser system.
"""

# Layer 2.5 Chaser Configuration
LAYER25_CHASER_CONFIG = {
    # Master Switch
    'layer25_chaser_enabled': False,  # Enable/disable Layer 2.5 Chaser
    
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
    'chaser_early_stopping_rounds': 30, # Aggressive early stopping
    'chaser_min_samples': 100,
    'chaser_residual_analysis': True,
    
    # Integration
    'chaser_add_to_layer3': True,  # Add Chaser features to Layer 3
    'chaser_conflict_features': True,  # Add conflict detection features

    # --- Model Specific Parameters (Default) ---

    'chaser_xgb_params': {
        'n_estimators': 1000,
        'max_depth': 5,
        'learning_rate': 0.03, # eta
        'subsample': 0.6,
        'colsample_bytree': 0.7,
        'colsample_bynode': 0.4,
        'reg_lambda': 25.0, # Reduced from 50
        'reg_alpha': 0.0,
        'gamma': 0.7, # Reduced from 1.1
        'num_parallel_tree': 15, # Random Forest behavior
        'min_child_weight': 10,
        'random_state': 42,
        'n_jobs': -1
    },

    'chaser_lgb_params': {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.7,
        'subsample_freq': 1, # Added
        'colsample_bytree': 0.7,
        'colsample_bynode': 0.7, # feature_fraction_bynode
        'reg_lambda': 10.0,
        'min_split_gain': 0.005, # min_gain_to_split
        'linear_tree': True, # linear=true
        'path_smooth': 20,
        'extra_trees': True,
        'n_jobs': -1,
        'verbose': -1
    },

    'chaser_cat_params': {
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 20.0,
        'subsample': 0.6,
        'rsm': 0.8, # colsample_bylevel
        'bagging_temperature': 1,
        'random_strength': 5.0,
        'random_seed': 42,
        'verbose': False,
        'allow_writing_files': False
    },

    'chaser_et_params': {
        'n_estimators': 500, # Reduced from 1000/200
        'max_depth': 6, # Reduced to 6
        'min_samples_leaf': 20, # Increased to 20
        'n_jobs': -1,
        'random_state': 42
    },

    'chaser_ensemble_weights': [0.6, 0.4], # Default weights if simple averaging used (deprecated by pruning logic but kept for config structure)
}

print("Layer 2.5 Chaser configuration loaded.")
