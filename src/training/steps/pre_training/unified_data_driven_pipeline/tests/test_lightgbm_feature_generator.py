"""
Test suite for LightGBM + Featuretools Feature Generator

This module tests the new LightGBM/CatBoost + Featuretools feature generation system
as a replacement for the Random Forest + SHAP system.
"""

import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.lightgbm_feature_generator import (
    LightGBMFeatureGenerator,
    FeatureGenerationConfig,
    GeneratedFeature,
    FeatureGenerationResult,
    create_lightgbm_feature_generator
)

class TestLightGBMFeatureGenerator(unittest.TestCase):
    """Test cases for LightGBM feature generator."""
    
    def setUp(self):
        """Set up test data and configuration."""
        # Create sample data
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        
        # Generate time series data
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
        
        data = {}
        for i in range(n_features):
            data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
        
        # Add some technical indicators
        data['price'] = 100 + np.cumsum(np.random.normal(0, 0.5, n_samples))
        data['volume'] = np.random.lognormal(10, 1, n_samples)
        data['rsi'] = np.random.uniform(20, 80, n_samples)
        
        # Create target
        data['target'] = np.random.normal(0, 0.02, n_samples)
        
        self.data = pd.DataFrame(data, index=dates)
        
        # Create configuration
        self.config = FeatureGenerationConfig(
            model_type='lightgbm',
            max_features=20,
            use_shap=False,  # Disable for faster testing
            use_ale=False,   # Disable for faster testing
            max_depth_featuretools=1
        )
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = LightGBMFeatureGenerator(self.config)
        self.assertIsInstance(generator, LightGBMFeatureGenerator)
        self.assertEqual(generator.config.model_type, 'lightgbm')
        self.assertEqual(generator.config.max_features, 20)
    
    def test_create_generator_function(self):
        """Test create_lightgbm_feature_generator function."""
        generator = create_lightgbm_feature_generator(self.config)
        self.assertIsInstance(generator, LightGBMFeatureGenerator)
    
    def test_feature_generation_light_mode(self):
        """Test feature generation in light mode."""
        generator = create_lightgbm_feature_generator(self.config)
        
        result = generator.generate_features(
            data=self.data,
            target_column='target',
            execution_mode='light'
        )
        
        self.assertIsInstance(result, FeatureGenerationResult)
        self.assertGreaterEqual(result.n_features_generated, 0)
        self.assertGreaterEqual(result.n_features_selected, 0)
        self.assertGreaterEqual(result.generation_time, 0)
        self.assertIsInstance(result.generated_features, list)
        self.assertIsInstance(result.feature_importance_scores, dict)
        self.assertIsInstance(result.model_performance, dict)
    
    def test_feature_generation_blank_mode(self):
        """Test feature generation in blank mode."""
        generator = create_lightgbm_feature_generator(self.config)
        
        result = generator.generate_features(
            data=self.data,
            target_column='target',
            execution_mode='blank'
        )
        
        self.assertIsInstance(result, FeatureGenerationResult)
        self.assertGreaterEqual(result.n_features_generated, 0)
        self.assertGreaterEqual(result.n_features_selected, 0)
    
    def test_catboost_model_type(self):
        """Test CatBoost model type."""
        config = FeatureGenerationConfig(
            model_type='catboost',
            max_features=15,
            use_shap=False,
            use_ale=False
        )
        
        generator = create_lightgbm_feature_generator(config)
        self.assertEqual(generator.config.model_type, 'catboost')
        
        result = generator.generate_features(
            data=self.data,
            target_column='target',
            execution_mode='light'
        )
        
        self.assertIsInstance(result, FeatureGenerationResult)
    
    def test_feature_limit_enforcement(self):
        """Test that feature limit is enforced."""
        config = FeatureGenerationConfig(
            max_features=5,
            use_shap=False,
            use_ale=False
        )
        
        generator = create_lightgbm_feature_generator(config)
        result = generator.generate_features(
            data=self.data,
            target_column='target',
            execution_mode='light'
        )
        
        # Should not exceed max_features
        self.assertLessEqual(result.n_features_selected, config.max_features)
    
    def test_data_preparation(self):
        """Test data preparation functionality."""
        generator = create_lightgbm_feature_generator(self.config)
        
        # Test with valid data
        prepared_data = generator._prepare_data(
            self.data, 'target', None
        )
        self.assertIsNotNone(prepared_data)
        self.assertIn('target', prepared_data.columns)
        
        # Test with invalid data
        invalid_data = pd.DataFrame({'invalid': [1, 2, 3]})
        prepared_data = generator._prepare_data(
            invalid_data, 'target', None
        )
        self.assertIsNone(prepared_data)
    
    def test_base_feature_generation(self):
        """Test base feature generation."""
        generator = create_lightgbm_feature_generator(self.config)
        
        base_features = generator._generate_base_features(
            self.data, 'light'
        )
        
        self.assertIsInstance(base_features, pd.DataFrame)
        self.assertGreater(len(base_features.columns), 0)
    
    def test_model_training(self):
        """Test model training."""
        generator = create_lightgbm_feature_generator(self.config)
        
        # Create simple features
        features = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 100),
            'feature_2': np.random.normal(0, 1, 100)
        })
        target = pd.Series(np.random.normal(0, 1, 100))
        
        model, feature_importance = generator._train_model(features, target)
        
        # Model might be None if dependencies are not available
        if model is not None:
            self.assertIsInstance(feature_importance, dict)
            self.assertEqual(len(feature_importance), len(features.columns))
    
    def test_performance_stats(self):
        """Test performance statistics tracking."""
        generator = create_lightgbm_feature_generator(self.config)
        
        # Generate features to update stats
        generator.generate_features(
            data=self.data,
            target_column='target',
            execution_mode='light'
        )
        
        stats = generator.get_performance_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('total_generations', stats)
        self.assertIn('successful_generations', stats)
        self.assertIn('failed_generations', stats)
        self.assertIn('total_execution_time', stats)
    
    def test_error_handling(self):
        """Test error handling with invalid inputs."""
        generator = create_lightgbm_feature_generator(self.config)
        
        # Test with empty data
        empty_data = pd.DataFrame()
        result = generator.generate_features(
            data=empty_data,
            target_column='target',
            execution_mode='light'
        )
        
        self.assertIsInstance(result, FeatureGenerationResult)
        self.assertEqual(result.n_features_generated, 0)
        self.assertEqual(result.n_features_selected, 0)
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test default configuration
        config = FeatureGenerationConfig()
        self.assertEqual(config.model_type, 'lightgbm')
        self.assertEqual(config.max_features, 100)
        self.assertTrue(config.use_shap)
        self.assertTrue(config.use_ale)
        
        # Test custom configuration
        config = FeatureGenerationConfig(
            model_type='catboost',
            max_features=50,
            use_shap=False,
            use_ale=False
        )
        self.assertEqual(config.model_type, 'catboost')
        self.assertEqual(config.max_features, 50)
        self.assertFalse(config.use_shap)
        self.assertFalse(config.use_ale)
    
    def test_generated_feature_creation(self):
        """Test GeneratedFeature object creation."""
        feature = GeneratedFeature(
            name='test_feature',
            formula='feature_1 * feature_2',
            feature_series=pd.Series([1, 2, 3]),
            importance_score=0.5,
            parent_features=['feature_1', 'feature_2']
        )
        
        self.assertEqual(feature.name, 'test_feature')
        self.assertEqual(feature.formula, 'feature_1 * feature_2')
        self.assertEqual(feature.importance_score, 0.5)
        self.assertEqual(feature.parent_features, ['feature_1', 'feature_2'])
        self.assertEqual(feature.generation_method, 'lightgbm_featuretools')
    
    def test_feature_combination(self):
        """Test feature combination functionality."""
        generator = create_lightgbm_feature_generator(self.config)
        
        # Create test features
        base_features = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5],
            'feature_2': [2, 4, 6, 8, 10]
        })
        
        featuretools_features = pd.DataFrame({
            'feature_1 + feature_2': [3, 6, 9, 12, 15],
            'feature_1 * feature_2': [2, 8, 18, 32, 50]
        })
        
        combined = generator._combine_features(base_features, featuretools_features)
        
        self.assertIsInstance(combined, pd.DataFrame)
        self.assertEqual(len(combined.columns), 4)  # 2 base + 2 featuretools
        self.assertIn('feature_1', combined.columns)
        self.assertIn('feature_2', combined.columns)
        self.assertIn('feature_1 + feature_2', combined.columns)
        self.assertIn('feature_1 * feature_2', combined.columns)


class TestFeatureGenerationConfig(unittest.TestCase):
    """Test cases for FeatureGenerationConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = FeatureGenerationConfig()
        
        self.assertEqual(config.model_type, 'lightgbm')
        self.assertEqual(config.n_estimators, 100)
        self.assertEqual(config.max_depth, 10)
        self.assertEqual(config.learning_rate, 0.1)
        self.assertEqual(config.max_features, 100)
        self.assertTrue(config.use_shap)
        self.assertTrue(config.use_ale)
        self.assertEqual(config.max_depth_featuretools, 2)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = FeatureGenerationConfig(
            model_type='catboost',
            n_estimators=200,
            max_depth=15,
            learning_rate=0.05,
            max_features=50,
            use_shap=False,
            use_ale=False,
            max_depth_featuretools=1
        )
        
        self.assertEqual(config.model_type, 'catboost')
        self.assertEqual(config.n_estimators, 200)
        self.assertEqual(config.max_depth, 15)
        self.assertEqual(config.learning_rate, 0.05)
        self.assertEqual(config.max_features, 50)
        self.assertFalse(config.use_shap)
        self.assertFalse(config.use_ale)
        self.assertEqual(config.max_depth_featuretools, 1)
    
    def test_primitive_types_default(self):
        """Test default primitive types."""
        config = FeatureGenerationConfig()
        
        expected_primitives = [
            'add_numeric', 'multiply_numeric', 'divide_numeric',
            'subtract_numeric', 'mean', 'std', 'min', 'max', 'count'
        ]
        
        self.assertEqual(config.primitive_types, expected_primitives)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)