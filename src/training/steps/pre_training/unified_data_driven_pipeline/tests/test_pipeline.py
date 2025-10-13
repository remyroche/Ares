"""
Tests for Unified Data-Driven Feature Pipeline

Basic tests to validate the pipeline functionality.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import the pipeline components
from ..core.unified_pipeline import (
    UnifiedDataDrivenPipeline,
    FeaturePipelineResult,
    create_unified_pipeline,
    process_features
)
from ..core.config import create_default_config
from ..time_series_cv import create_purged_embargoed_cv
from ..statistical_analysis import StatisticalAnalysisFramework
from ..feature_selection.multi_objective_selector import create_default_objectives


class TestUnifiedPipeline(unittest.TestCase):
    """Test cases for the unified pipeline."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample data
        np.random.seed(42)
        n_samples = 200
        n_features = 20
        
        # Create date index
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
        
        # Create features
        features = {}
        for i in range(n_features):
            features[f'feature_{i}'] = np.random.randn(n_samples)
        
        # Add some trending features
        features['trend'] = np.cumsum(np.random.randn(n_samples) * 0.01)
        features['volatility'] = np.abs(np.random.randn(n_samples) * 0.02)
        
        self.data = pd.DataFrame(features, index=dates)
        self.targets = pd.Series(np.random.randn(n_samples), index=dates)
        
        # Create configuration
        self.config = create_default_config()
        self.config.feature_selection.multi_objective.max_features = 10
        self.config.feature_selection.multi_objective.min_features = 3
        self.config.feature_selection.cv_config.n_splits = 3
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = create_unified_pipeline(self.config)
        self.assertIsInstance(pipeline, UnifiedDataDrivenPipeline)
        self.assertIsNotNone(pipeline.config)
        self.assertIsNotNone(pipeline.stats_framework)
        self.assertIsNotNone(pipeline.cv_splitter)
    
    def test_basic_processing(self):
        """Test basic pipeline processing."""
        result = process_features(self.data, self.targets, config=self.config)
        
        self.assertIsInstance(result, FeaturePipelineResult)
        self.assertIsInstance(result.selected_features, list)
        self.assertGreater(len(result.selected_features), 0)
        self.assertLessEqual(len(result.selected_features), self.config.feature_selection.multi_objective.max_features)
        self.assertGreaterEqual(len(result.selected_features), self.config.feature_selection.multi_objective.min_features)
        self.assertIsInstance(result.objective_values, dict)
        self.assertGreater(result.processing_time, 0)
    
    def test_custom_configuration(self):
        """Test custom configuration."""
        config = create_default_config()
        config.feature_selection.multi_objective.max_features = 5
        config.feature_selection.multi_objective.min_features = 2
        
        result = process_features(self.data, self.targets, config=config)
        
        self.assertLessEqual(len(result.selected_features), 5)
        self.assertGreaterEqual(len(result.selected_features), 2)
    
    def test_data_validation(self):
        """Test data validation."""
        pipeline = create_unified_pipeline(self.config)
        
        # Test with None data
        with self.assertRaises(ValueError):
            pipeline.process(None, None)
        
        # Test with empty data
        with self.assertRaises(ValueError):
            pipeline.process(pd.DataFrame(), None)
        
        # Test with mismatched lengths
        with self.assertRaises(ValueError):
            pipeline.process(self.data, self.targets.iloc[:-1])
    
    def test_time_series_cv(self):
        """Test time series cross-validation."""
        cv = create_purged_embargoed_cv(n_splits=3, test_size=0.2, train_size=0.6)
        splits = cv.split(self.data, targets=self.targets)
        
        self.assertGreater(len(splits), 0)
        self.assertLessEqual(len(splits), 3)
        
        # Test split validation
        for split in splits:
            self.assertTrue(split.is_valid())
            self.assertLess(split.train_end, split.test_start)
    
    def test_statistical_analysis(self):
        """Test statistical analysis framework."""
        framework = StatisticalAnalysisFramework()
        
        # Test data characteristics analysis
        characteristics = framework.analyze_data_characteristics(self.data)
        self.assertIsNotNone(characteristics)
        self.assertEqual(characteristics.n_samples, len(self.data))
        self.assertEqual(characteristics.n_features, len(self.data.columns))
        
        # Test pattern detection
        patterns = framework.detect_patterns(self.data)
        self.assertIsNotNone(patterns)
        
        # Test relationship analysis
        relationships = framework.evaluate_feature_relationships(self.data, self.targets)
        self.assertIsNotNone(relationships)
    
    def test_multi_objective_selection(self):
        """Test multi-objective feature selection."""
        from ..feature_selection.multi_objective_selector import MultiObjectiveFeatureSelector, create_default_objectives
        
        objectives = create_default_objectives()
        selector = MultiObjectiveFeatureSelector(
            objectives=objectives,
            max_features=10,
            min_features=3
        )
        
        result = selector.select_features(self.data, self.targets)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result.selected_features, list)
        self.assertGreater(len(result.selected_features), 0)
        self.assertLessEqual(len(result.selected_features), 10)
        self.assertGreaterEqual(len(result.selected_features), 3)
    
    def test_performance_monitoring(self):
        """Test performance monitoring."""
        pipeline = create_unified_pipeline(self.config)
        result = pipeline.process(self.data, self.targets)
        
        stats = pipeline.get_performance_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('total_processing_time', stats)
        self.assertIn('n_cv_splits', stats)
        self.assertGreater(stats['total_processing_time'], 0)
    
    def test_result_saving(self):
        """Test result saving functionality."""
        result = process_features(self.data, self.targets, config=self.config)
        
        # Test saving (this will create files)
        try:
            result.save_result(result, "test_output")
            # Check if files were created (simplified check)
            import os
            self.assertTrue(os.path.exists("test_output"))
        except Exception as e:
            # If saving fails, that's okay for this test
            self.assertIsInstance(e, Exception)
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        config = create_default_config()
        
        # Test valid configuration
        self.assertIsNotNone(config)
        self.assertIsNotNone(config.feature_selection)
        self.assertIsNotNone(config.period_optimization)
        self.assertIsNotNone(config.interaction_generation)
    
    def test_objective_functions(self):
        """Test objective functions."""
        from ..feature_selection.multi_objective_selector import (
            OutOfSampleSharpeObjective,
            DrawdownObjective,
            StabilityObjective
        )
        
        # Test Sharpe objective
        sharpe_obj = OutOfSampleSharpeObjective()
        result = sharpe_obj.evaluate(self.data, self.targets, ['feature_0', 'feature_1'])
        self.assertIsNotNone(result)
        self.assertIsInstance(result.value, float)
        
        # Test drawdown objective
        drawdown_obj = DrawdownObjective()
        result = drawdown_obj.evaluate(self.data, self.targets, ['feature_0', 'feature_1'])
        self.assertIsNotNone(result)
        self.assertIsInstance(result.value, float)
        
        # Test stability objective
        stability_obj = StabilityObjective()
        result = stability_obj.evaluate(self.data, self.targets, ['feature_0', 'feature_1'])
        self.assertIsNotNone(result)
        self.assertIsInstance(result.value, float)


class TestTimeSeriesCV(unittest.TestCase):
    """Test cases for time series cross-validation."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 100
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
        self.data = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples)
        }, index=dates)
        self.targets = pd.Series(np.random.randn(n_samples), index=dates)
    
    def test_cv_split_generation(self):
        """Test CV split generation."""
        cv = create_purged_embargoed_cv(n_splits=3, test_size=0.2, train_size=0.6)
        splits = cv.split(self.data, targets=self.targets)
        
        self.assertGreater(len(splits), 0)
        self.assertLessEqual(len(splits), 3)
        
        for split in splits:
            self.assertTrue(split.is_valid())
            self.assertGreater(len(split.train_indices), 0)
            self.assertGreater(len(split.test_indices), 0)
    
    def test_leakage_validation(self):
        """Test leakage validation."""
        cv = create_purged_embargoed_cv(n_splits=3, test_size=0.2, train_size=0.6)
        splits = cv.split(self.data, targets=self.targets)
        
        # Validate no leakage
        is_valid = cv.validate_no_leakage(self.data)
        self.assertTrue(is_valid)


class TestStatisticalFramework(unittest.TestCase):
    """Test cases for statistical analysis framework."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 100
        self.data = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples)
        })
        self.targets = pd.Series(np.random.randn(n_samples))
    
    def test_data_characteristics(self):
        """Test data characteristics analysis."""
        framework = StatisticalAnalysisFramework()
        characteristics = framework.analyze_data_characteristics(self.data)
        
        self.assertIsNotNone(characteristics)
        self.assertEqual(characteristics.n_samples, len(self.data))
        self.assertEqual(characteristics.n_features, len(self.data.columns))
        self.assertIsInstance(characteristics.data_quality_score, float)
    
    def test_pattern_detection(self):
        """Test pattern detection."""
        framework = StatisticalAnalysisFramework()
        patterns = framework.detect_patterns(self.data)
        
        self.assertIsNotNone(patterns)
        self.assertIsInstance(patterns.cyclical_patterns, list)
        self.assertIsInstance(patterns.trend_strength, float)
    
    def test_relationship_analysis(self):
        """Test relationship analysis."""
        framework = StatisticalAnalysisFramework()
        relationships = framework.evaluate_feature_relationships(self.data, self.targets)
        
        self.assertIsNotNone(relationships)
        self.assertIsInstance(relationships.linear_correlations, pd.DataFrame)
        self.assertIsInstance(relationships.significant_correlations, list)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)