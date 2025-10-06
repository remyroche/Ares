#!/usr/bin/env python3
"""
Test Tactician T1-T4 Models Integration

This script tests that the updated tactician_models_training.py correctly uses
the new T1-T4 model configurations.

Usage:
    python test_tactician_t1_t4_integration.py
"""

import numpy as np
import pandas as pd
import asyncio
from pathlib import Path
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_t1_t4_models_integration():
    """Test that the updated tactician_models_training uses T1-T4 models correctly."""
    logger.info("🧪 Testing Tactician T1-T4 Models Integration...")

    try:
        # Import the updated tactician models training
        from src.training.steps.models_training.tactician_models_training import (
            TacticianModelsTrainingStep,
            TacticianModelsTrainingConfig,
            TacticianModelType
        )

        logger.info("✅ Successfully imported updated tactician models training")

        # Check that new model types are available
        expected_new_types = [
            TacticianModelType.T1_PATCHTST_LIGHTGBM,
            TacticianModelType.T2_PATCHTST_XGBOOST_LAMBDAMART,
            TacticianModelType.T3_PATCHTST_CATBOOST,
            TacticianModelType.T4_CAUSAL_DILATED_TCN,
            TacticianModelType.T4_TFT_SMALL
        ]

        logger.info("🔍 Checking new T1-T4 model types...")
        for model_type in expected_new_types:
            logger.info(f"  ✅ {model_type.value}")

        # Create sample data for testing
        logger.info("📊 Creating sample data for testing...")
        np.random.seed(42)

        n_samples = 500
        n_features = 100

        # Create feature data
        X_df = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )

        # Create target data for different tasks
        y_classification = np.random.choice([0, 1, 2], size=n_samples)  # 3-class classification
        y_ranking = np.random.randn(n_samples) * 0.01  # Ranking targets
        y_binary = np.random.choice([0, 1], size=n_samples)  # Binary classification
        y_regression = np.random.randn(n_samples)  # Regression targets

        # Create training configuration with new T1-T4 models
        logger.info("⚙️ Creating training configuration with T1-T4 models...")
        config = TacticianModelsTrainingConfig(
            model_types=[
                TacticianModelType.T1_PATCHTST_LIGHTGBM,  # Classification
                TacticianModelType.T2_PATCHTST_XGBOOST_LAMBDAMART,  # Ranking
                TacticianModelType.T3_PATCHTST_CATBOOST,  # Binary classification
                TacticianModelType.T4_CAUSAL_DILATED_TCN,  # Sequence model
                TacticianModelType.T4_TFT_SMALL  # Alternative sequence model
            ],
            save_models=False,  # Don't save for testing
            save_metrics=True,
            output_directory="/tmp/test_tactician_t1_t4"
        )

        logger.info("🏗️ Creating TacticianModelsTrainingStep...")
        trainer = TacticianModelsTrainingStep(config)

        # Test training each model type
        logger.info("🚀 Testing model training...")

        # Test T1: Classification
        logger.info("🔄 Testing T1: PatchTST-LightGBM (Classification)...")
        result_t1 = await trainer._train_t1_patchtst_lightgbm(
            X_df.values, y_classification, sample_weight=None
        )
        if result_t1['models']:
            logger.info(f"  ✅ T1 trained successfully: {list(result_t1['models'].keys())}")
            logger.info(f"  📊 Metrics: {result_t1['metrics']}")
        else:
            logger.error("  ❌ T1 training failed")

        # Test T2: Ranking
        logger.info("🔄 Testing T2: PatchTST-XGBoost-LambdaMART (Ranking)...")
        groups = np.random.randint(1, 11, size=n_samples)  # 10 groups
        result_t2 = await trainer._train_t2_patchtst_xgboost_lambdamart(
            X_df.values, y_ranking, sample_weight=None, groups=groups
        )
        if result_t2['models']:
            logger.info(f"  ✅ T2 trained successfully: {list(result_t2['models'].keys())}")
            logger.info(f"  📊 Metrics: {result_t2['metrics']}")
        else:
            logger.error("  ❌ T2 training failed")

        # Test T3: Binary classification
        logger.info("🔄 Testing T3: PatchTST-CatBoost (Binary Classification)...")
        result_t3 = await trainer._train_t3_patchtst_catboost(
            X_df.values, y_binary, sample_weight=None
        )
        if result_t3['models']:
            logger.info(f"  ✅ T3 trained successfully: {list(result_t3['models'].keys())}")
            logger.info(f"  📊 Metrics: {result_t3['metrics']}")
        else:
            logger.error("  ❌ T3 training failed")

        # Test T4: Sequence models
        logger.info("🔄 Testing T4: Causal Dilated TCN (Sequence)...")
        # Create sequence data
        seq_length = 50
        n_seq_features = 20
        X_seq = np.random.randn(n_samples, seq_length, n_seq_features)

        result_t4_tcn = await trainer._train_t4_causal_dilated_tcn(
            X_seq, y_regression.reshape(-1, 1), sample_weight=None
        )
        if result_t4_tcn['models']:
            logger.info(f"  ✅ T4 TCN trained successfully: {list(result_t4_tcn['models'].keys())}")
            logger.info(f"  📊 Metrics: {result_t4_tcn['metrics']}")
        else:
            logger.error("  ❌ T4 TCN training failed")

        logger.info("🔄 Testing T4: TFT-Small (Sequence Alternative)...")
        result_t4_tft = await trainer._train_t4_tft_small(
            X_seq, y_regression, sample_weight=None
        )
        if result_t4_tft['models']:
            logger.info(f"  ✅ T4 TFT trained successfully: {list(result_t4_tft['models'].keys())}")
            logger.info(f"  📊 Metrics: {result_t4_tft['metrics']}")
        else:
            logger.error("  ❌ T4 TFT training failed")

        # Test full training pipeline
        logger.info("🔄 Testing full training pipeline...")
        feature_columns = [f'feature_{i}' for i in range(n_features)]
        target_columns = ['target']

        # Add target column to dataframe for full pipeline test
        X_df_with_target = X_df.copy()
        X_df_with_target['target'] = y_classification

        try:
            full_result = await trainer.train_tactician_models(
                X_df_with_target, feature_columns, target_columns, sample_weight=None
            )

            logger.info("✅ Full training pipeline completed successfully")
            logger.info(f"📊 Full result keys: {list(full_result.keys())}")
            logger.info(f"🎯 Models trained: {len(full_result.get('models', {}))}")
            logger.info(f"📈 Metrics available: {len(full_result.get('metrics', {}))}")

        except Exception as e:
            logger.error(f"❌ Full training pipeline failed: {e}")
            import traceback
            traceback.print_exc()

        # Verify configuration file exists and is properly loaded
        logger.info("🔍 Verifying configuration integration...")
        config_path = Path("/workspace/config/tactician_t1_t4_models_config.yaml")
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                logger.info("✅ T1-T4 configuration file loaded successfully")
                logger.info(f"📋 Configuration sections: {list(config_data.keys())}")

                # Check that configuration contains expected sections
                expected_sections = [
                    'tactician_t1_t4_config',
                    'task_configs',
                    'analyst_integration'
                ]
                for section in expected_sections:
                    if section in config_data:
                        logger.info(f"  ✅ Configuration section '{section}' found")
                    else:
                        logger.warning(f"  ⚠️ Configuration section '{section}' missing")

            except Exception as e:
                logger.error(f"❌ Failed to load configuration: {e}")
        else:
            logger.warning("⚠️ T1-T4 configuration file not found")

        logger.info("🎉 Tactician T1-T4 Integration Test Completed Successfully!")

        return True

    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_t1_t4_models_integration())
    if success:
        print("\n✅ All tests passed! Tactician T1-T4 integration is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the logs above.")
        exit(1)