#!/usr/bin/env python3
"""
Tactician T1-T4 Models Usage Example

This script demonstrates how to use the new T1-T4 model configurations for the Tactician system.

Models:
- T1: PatchTST-Embed + LightGBM (classification: up/down/none or two binaries)
- T2: PatchTST-Embed + XGBoost LambdaMART (ranking: trade desirability)
- T3: PatchTST-Embed + CatBoost (binary classification: up_hit@H, down_hit@H)
- T4: Causal Dilated TCN or TFT-Small (sequence classification/regression)

Usage:
    python tactician_t1_t4_models_usage.py
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import yaml
import logging

# Import the model factory
from src.utils.ml_common.models.model_factory import EnhancedModelFactory, ModelType, ModelConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_tactician_t1_t4_config() -> Dict[str, Any]:
    """Load Tactician T1-T4 model configuration."""
    config_path = "/workspace/config/tactician_t1_t4_models_config.yaml"

    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"✅ Loaded Tactician T1-T4 configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        return {}


def create_t1_patchtst_lightgbm_model(config: Dict[str, Any]) -> Any:
    """Create T1: PatchTST-Enhanced LightGBM model for classification."""
    logger.info("🔄 Creating T1: PatchTST-LightGBM model...")

    model_config = ModelConfig(
        model_type=ModelType.PATCHTST_LIGHTGBM,
        model_name="t1_patchtst_lightgbm",
        n_outputs=3,  # up/down/none or two binaries
        model_params={
            'n_estimators': config['tactician_t1_t4_config']['tree_models']['t1_lightgbm']['params']['n_estimators'],
            'learning_rate': config['tactician_t1_t4_config']['tree_models']['t1_lightgbm']['params']['learning_rate'],
            'max_depth': config['tactician_t1_t4_config']['tree_models']['t1_lightgbm']['params']['max_depth'],
            'num_leaves': config['tactician_t1_t4_config']['tree_models']['t1_lightgbm']['params']['num_leaves'],
            'subsample': config['tactician_t1_t4_config']['tree_models']['t1_lightgbm']['params']['subsample'],
            'colsample_bytree': config['tactician_t1_t4_config']['tree_models']['t1_lightgbm']['params']['colsample_bytree'],
            'random_state': config['tactician_t1_t4_config']['tree_models']['t1_lightgbm']['params']['random_state'],
            'verbosity': config['tactician_t1_t4_config']['tree_models']['t1_lightgbm']['params']['verbosity'],
            'monotone_constraints': config['tactician_t1_t4_config']['tree_models']['t1_lightgbm']['params']['monotone_constraints'],
            'monotone_constraints_method': config['tactician_t1_t4_config']['tree_models']['t1_lightgbm']['params']['monotone_constraints_method'],
            'patchtst_config': config['tactician_t1_t4_config']['patchtst_config']
        }
    )

    factory = EnhancedModelFactory()
    model = factory.create_model(model_config)
    logger.info("✅ T1: PatchTST-LightGBM model created successfully")
    return model


def create_t2_patchtst_xgboost_lambdamart_model(config: Dict[str, Any]) -> Any:
    """Create T2: PatchTST-Enhanced XGBoost LambdaMART model for ranking."""
    logger.info("🔄 Creating T2: PatchTST-XGBoost-LambdaMART model...")

    model_config = ModelConfig(
        model_type=ModelType.PATCHTST_XGBOOST_LAMBDAMART,
        model_name="t2_patchtst_xgboost_lambdamart",
        n_outputs=1,  # Ranking score
        model_params={
            'n_estimators': config['tactician_t1_t4_config']['tree_models']['t2_xgboost_lambdamart']['params']['n_estimators'],
            'learning_rate': config['tactician_t1_t4_config']['tree_models']['t2_xgboost_lambdamart']['params']['learning_rate'],
            'max_depth': config['tactician_t1_t4_config']['tree_models']['t2_xgboost_lambdamart']['params']['max_depth'],
            'subsample': config['tactician_t1_t4_config']['tree_models']['t2_xgboost_lambdamart']['params']['subsample'],
            'colsample_bytree': config['tactician_t1_t4_config']['tree_models']['t2_xgboost_lambdamart']['params']['colsample_bytree'],
            'random_state': config['tactician_t1_t4_config']['tree_models']['t2_xgboost_lambdamart']['params']['random_state'],
            'lambda': config['tactician_t1_t4_config']['tree_models']['t2_xgboost_lambdamart']['params']['lambda'],
            'alpha': config['tactician_t1_t4_config']['tree_models']['t2_xgboost_lambdamart']['params']['alpha'],
            'verbosity': config['tactician_t1_t4_config']['tree_models']['t2_xgboost_lambdamart']['params']['verbosity'],
            'monotone_constraints': config['tactician_t1_t4_config']['tree_models']['t2_xgboost_lambdamart']['params']['monotone_constraints'],
            'monotone_constraints_method': config['tactician_t1_t4_config']['tree_models']['t2_xgboost_lambdamart']['params']['monotone_constraints_method'],
            'patchtst_config': config['tactician_t1_t4_config']['patchtst_config']
        }
    )

    factory = EnhancedModelFactory()
    model = factory.create_model(model_config)
    logger.info("✅ T2: PatchTST-XGBoost-LambdaMART model created successfully")
    return model


def create_t3_patchtst_catboost_model(config: Dict[str, Any]) -> Any:
    """Create T3: PatchTST-Enhanced CatBoost model for binary classification."""
    logger.info("🔄 Creating T3: PatchTST-CatBoost model...")

    model_config = ModelConfig(
        model_type=ModelType.PATCHTST_CATBOOST,
        model_name="t3_patchtst_catboost",
        n_outputs=2,  # up_hit@H and down_hit@H
        model_params={
            'iterations': config['tactician_t1_t4_config']['tree_models']['t3_catboost']['params']['iterations'],
            'learning_rate': config['tactician_t1_t4_config']['tree_models']['t3_catboost']['params']['learning_rate'],
            'depth': config['tactician_t1_t4_config']['tree_models']['t3_catboost']['params']['depth'],
            'random_seed': config['tactician_t1_t4_config']['tree_models']['t3_catboost']['params']['random_seed'],
            'verbose': config['tactician_t1_t4_config']['tree_models']['t3_catboost']['params']['verbose'],
            'monotone_constraints': config['tactician_t1_t4_config']['tree_models']['t3_catboost']['params']['monotone_constraints'],
            'ordered_boosting': config['tactician_t1_t4_config']['tree_models']['t3_catboost']['params']['ordered_boosting'],
            'patchtst_config': config['tactician_t1_t4_config']['patchtst_config']
        }
    )

    factory = EnhancedModelFactory()
    model = factory.create_model(model_config)
    logger.info("✅ T3: PatchTST-CatBoost model created successfully")
    return model


def create_t4_causal_dilated_tcn_model(config: Dict[str, Any]) -> Any:
    """Create T4: Causal Dilated TCN model for sequence tasks."""
    logger.info("🔄 Creating T4: Causal Dilated TCN model...")

    model_config = ModelConfig(
        model_type=ModelType.CAUSAL_DILATED_TCN,
        model_name="t4_causal_dilated_tcn",
        n_outputs=2,  # P(up_hit@H), P(down_hit@H) or E[ret_H]
        model_params={
            'residual_blocks': config['tactician_t1_t4_config']['sequence_model']['t4_tcn']['config']['residual_blocks'],
            'channels': config['tactician_t1_t4_config']['sequence_model']['t4_tcn']['config']['channels'],
            'kernel_size': config['tactician_t1_t4_config']['sequence_model']['t4_tcn']['config']['kernel_size'],
            'dilations': config['tactician_t1_t4_config']['sequence_model']['t4_tcn']['config']['dilations'],
            'dropout': config['tactician_t1_t4_config']['sequence_model']['t4_tcn']['config']['dropout'],
            'use_batch_norm': config['tactician_t1_t4_config']['sequence_model']['t4_tcn']['config']['use_batch_norm'],
            'activation': config['tactician_t1_t4_config']['sequence_model']['t4_tcn']['config']['activation'],
            'input_dim': config['tactician_t1_t4_config']['sequence_model']['t4_tcn']['config'].get('input_dim', 100),
            'seq_length': config['tactician_t1_t4_config']['sequence_model']['t4_tcn']['config'].get('seq_length', 100)
        }
    )

    factory = EnhancedModelFactory()
    model = factory.create_model(model_config)
    logger.info("✅ T4: Causal Dilated TCN model created successfully")
    return model


def create_t4_tft_small_model(config: Dict[str, Any]) -> Any:
    """Create T4: TFT-Small model for sequence tasks (alternative to TCN)."""
    logger.info("🔄 Creating T4: TFT-Small model (alternative)...")

    model_config = ModelConfig(
        model_type=ModelType.TFT_SMALL,
        model_name="t4_tft_small",
        n_outputs=1,  # E[ret_H] or regression output
        model_params={
            'hidden_size': config['tactician_t1_t4_config']['sequence_model']['t4_tft_small']['config']['hidden_size'],
            'attention_heads': config['tactician_t1_t4_config']['sequence_model']['t4_tft_small']['config']['attention_heads'],
            'dropout': config['tactician_t1_t4_config']['sequence_model']['t4_tft_small']['config']['dropout'],
            'num_layers': config['tactician_t1_t4_config']['sequence_model']['t4_tft_small']['config']['num_layers'],
            'use_time_features': config['tactician_t1_t4_config']['sequence_model']['t4_tft_small']['config']['use_time_features'],
            'use_static_features': config['tactician_t1_t4_config']['sequence_model']['t4_tft_small']['config']['use_static_features'],
            'input_dim': config['tactician_t1_t4_config']['sequence_model']['t4_tft_small']['config'].get('input_dim', 100),
            'seq_length': config['tactician_t1_t4_config']['sequence_model']['t4_tft_small']['config'].get('seq_length', 100)
        }
    )

    factory = EnhancedModelFactory()
    model = factory.create_model(model_config)
    logger.info("✅ T4: TFT-Small model created successfully")
    return model


def demonstrate_model_usage():
    """Demonstrate usage of all T1-T4 models."""
    logger.info("🚀 Starting Tactician T1-T4 Models Demonstration")

    # Load configuration
    config = load_tactician_t1_t4_config()
    if not config:
        logger.error("❌ Could not load configuration. Exiting.")
        return

    # Create sample data for demonstration
    logger.info("📊 Creating sample data for demonstration...")
    np.random.seed(42)

    # Sample features (tabular + regime + analyst outputs + PatchTST embeddings)
    n_samples = 1000
    n_features = 150

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Sample targets for different tasks
    # T1/T3: Classification (up/down/none)
    y_classification = np.random.choice([0, 1, 2], size=n_samples)  # 0: down, 1: none, 2: up

    # T2: Ranking (realized PnL)
    y_ranking = np.random.randn(n_samples) * 0.01  # Small PnL values

    # T4: Sequence data (reshaped for sequence models)
    seq_length = 100
    n_seq_features = 50
    X_sequence = np.random.randn(n_samples, seq_length, n_seq_features)

    logger.info("✅ Sample data created")

    # Create and demonstrate T1 model
    try:
        logger.info("\n" + "="*50)
        logger.info("T1: PatchTST-LightGBM (Classification)")
        logger.info("="*50)

        t1_model = create_t1_patchtst_lightgbm_model(config)

        # Fit the model
        logger.info("🔄 Fitting T1 model...")
        t1_model.fit(X, y_classification)
        logger.info("✅ T1 model fitted")

        # Make predictions
        predictions = t1_model.predict(X[:10])
        logger.info(f"📊 Sample predictions: {predictions}")

        # Get feature importance
        if hasattr(t1_model, 'get_feature_importance'):
            importance = t1_model.get_feature_importance()
            logger.info(f"🎯 Feature importance shape: {importance.shape}")

    except Exception as e:
        logger.error(f"❌ T1 model demonstration failed: {e}")

    # Create and demonstrate T2 model
    try:
        logger.info("\n" + "="*50)
        logger.info("T2: PatchTST-XGBoost-LambdaMART (Ranking)")
        logger.info("="*50)

        t2_model = create_t2_patchtst_xgboost_lambdamart_model(config)

        # Fit the model (for ranking, we need group information)
        logger.info("🔄 Fitting T2 model...")
        groups = np.random.randint(1, 21, size=n_samples)  # 20 groups
        t2_model.fit(X, y_ranking, group=groups)
        logger.info("✅ T2 model fitted")

        # Make predictions
        predictions = t2_model.predict(X[:10])
        logger.info(f"📊 Sample ranking predictions: {predictions}")

    except Exception as e:
        logger.error(f"❌ T2 model demonstration failed: {e}")

    # Create and demonstrate T3 model
    try:
        logger.info("\n" + "="*50)
        logger.info("T3: PatchTST-CatBoost (Binary Classification)")
        logger.info("="*50)

        t3_model = create_t3_patchtst_catboost_model(config)

        # Fit the model (binary classification - convert to binary)
        y_binary = (y_classification == 2).astype(int)  # 1 for up, 0 for down/none
        logger.info("🔄 Fitting T3 model...")
        t3_model.fit(X, y_binary)
        logger.info("✅ T3 model fitted")

        # Make predictions
        predictions = t3_model.predict(X[:10])
        logger.info(f"📊 Sample binary predictions: {predictions}")

        # Get probabilities
        if hasattr(t3_model, 'predict_proba'):
            probabilities = t3_model.predict_proba(X[:10])
            logger.info(f"📊 Sample probabilities shape: {probabilities.shape}")

    except Exception as e:
        logger.error(f"❌ T3 model demonstration failed: {e}")

    # Create and demonstrate T4 model (TCN)
    try:
        logger.info("\n" + "="*50)
        logger.info("T4: Causal Dilated TCN (Sequence Model)")
        logger.info("="*50)

        t4_tcn_model = create_t4_causal_dilated_tcn_model(config)

        # Fit the model
        logger.info("🔄 Fitting T4 TCN model...")
        # For sequence models, we need to reshape the target appropriately
        y_sequence = np.random.randn(n_samples, 2)  # Multi-output for sequence model
        t4_tcn_model.fit(X_sequence, y_sequence)
        logger.info("✅ T4 TCN model fitted")

        # Make predictions
        predictions = t4_tcn_model.predict(X_sequence[:10])
        logger.info(f"📊 Sample sequence predictions shape: {predictions.shape}")

    except Exception as e:
        logger.error(f"❌ T4 TCN model demonstration failed: {e}")

    # Create and demonstrate T4 model (TFT-Small)
    try:
        logger.info("\n" + "="*50)
        logger.info("T4: TFT-Small (Alternative Sequence Model)")
        logger.info("="*50)

        t4_tft_model = create_t4_tft_small_model(config)

        # Fit the model
        logger.info("🔄 Fitting T4 TFT-Small model...")
        y_regression = np.random.randn(n_samples)  # Regression target
        t4_tft_model.fit(X_sequence, y_regression)
        logger.info("✅ T4 TFT-Small model fitted")

        # Make predictions
        predictions = t4_tft_model.predict(X_sequence[:10])
        logger.info(f"📊 Sample TFT predictions: {predictions[:5]}")

    except Exception as e:
        logger.error(f"❌ T4 TFT-Small model demonstration failed: {e}")

    logger.info("\n" + "="*50)
    logger.info("🎉 Tactician T1-T4 Models Demonstration Complete!")
    logger.info("="*50)

    # Summary
    logger.info("📋 Summary of implemented models:")
    logger.info("  ✅ T1: PatchTST-LightGBM (Classification with monotone constraints)")
    logger.info("  ✅ T2: PatchTST-XGBoost-LambdaMART (Ranking with pairwise objective)")
    logger.info("  ✅ T3: PatchTST-CatBoost (Binary classification with ordered boosting)")
    logger.info("  ✅ T4: Causal Dilated TCN (Sequence model with residual blocks)")
    logger.info("  ✅ T4: TFT-Small (Alternative sequence model)")

    logger.info("\n🔧 Key Features Implemented:")
    logger.info("  • PatchTST transformer embeddings for all tree models")
    logger.info("  • Monotone constraints for interpretable feature relationships")
    logger.info("  • Multiple loss functions (softmax, BCE, pairwise ranking)")
    logger.info("  • Causal dilated convolutions for sequence modeling")
    logger.info("  • Integration with existing Analyst outputs")
    logger.info("  • Regime-aware patch selection")


if __name__ == "__main__":
    demonstrate_model_usage()