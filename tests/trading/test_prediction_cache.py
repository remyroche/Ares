"""
Unit Tests for PredictionCache

Tests the prediction cache functionality including:
- Thread-safe operations
- Rolling buffer management
- Position-specific tracking
- Confidence degradation calculation
"""

import unittest
from datetime import datetime, timedelta
from typing import Dict, Any
import pandas as pd
import numpy as np

from src.trading.monitoring.prediction_cache import (
    PredictionCache,
    PredictionEntry,
    get_global_prediction_cache
)


class TestPredictionCache(unittest.TestCase):
    """Test suite for PredictionCache."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache = PredictionCache(max_candles=50, default_window=8)
    
    def test_add_analyst_prediction(self):
        """Test adding analyst predictions."""
        predictions = {'confidence': 0.75, 'direction': 'long'}
        timestamp = datetime.now()
        
        self.cache.add_analyst_prediction(
            predictions=predictions,
            timestamp=timestamp,
            confidence=0.75
        )
        
        # Verify prediction was added
        recent = self.cache.get_recent_analyst_predictions(1)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0].confidence, 0.75)
    
    def test_add_tactician_prediction(self):
        """Test adding tactician predictions."""
        predictions = {'confidence': 0.80, 'signal': 'buy'}
        timestamp = datetime.now()
        
        self.cache.add_tactician_prediction(
            predictions=predictions,
            timestamp=timestamp,
            confidence=0.80
        )
        
        # Verify prediction was added
        recent = self.cache.get_recent_tactician_predictions(1)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0].confidence, 0.80)
    
    def test_rolling_buffer_limits_size(self):
        """Test that rolling buffer doesn't exceed max_candles."""
        # Add more predictions than max_candles
        for i in range(100):
            self.cache.add_analyst_prediction(
                predictions={'confidence': 0.5 + i * 0.001},
                timestamp=datetime.now()
            )
        
        # Should only keep last max_candles
        recent = self.cache.get_recent_analyst_predictions(100)
        self.assertEqual(len(recent), 50)  # max_candles = 50
    
    def test_get_recent_predictions_window(self):
        """Test retrieving predictions with custom window."""
        # Add 20 predictions
        for i in range(20):
            confidence_value = 0.5 + i * 0.01
            self.cache.add_tactician_prediction(
                predictions={'prediction_data': 'test'},
                confidence=confidence_value,
                timestamp=datetime.now() + timedelta(seconds=i)
            )
        
        # Get last 5
        recent = self.cache.get_recent_tactician_predictions(5)
        self.assertEqual(len(recent), 5)
        
        # Should be most recent ones (highest confidence)
        self.assertGreater(recent[-1].confidence, recent[0].confidence)
    
    def test_position_registration(self):
        """Test position registration snapshots predictions."""
        # Add some tactician predictions
        for i in range(10):
            self.cache.add_tactician_prediction(
                predictions={'confidence': 0.7 + i * 0.01},
                timestamp=datetime.now() + timedelta(seconds=i),
                confidence=0.7 + i * 0.01
            )
        
        # Register a position
        position_id = 'test_pos_1'
        self.cache.register_position(
            position_id=position_id,
            entry_timestamp=datetime.now(),
            snapshot_window=5
        )
        
        # Verify position has snapshot
        metrics = self.cache.get_position_metrics(position_id)
        self.assertEqual(metrics['position_id'], position_id)
        self.assertGreater(metrics['num_predictions'], 0)
    
    def test_update_position_predictions(self):
        """Test updating position predictions."""
        # Register a position
        position_id = 'test_pos_1'
        self.cache.register_position(
            position_id=position_id,
            entry_timestamp=datetime.now(),
            snapshot_window=5
        )
        
        # Add new prediction
        new_prediction = PredictionEntry(
            timestamp=datetime.now(),
            predictions={'confidence': 0.75},
            confidence=0.75
        )
        
        self.cache.update_position_predictions(position_id, new_prediction)
        
        # Verify prediction was added
        metrics = self.cache.get_position_metrics(position_id)
        self.assertGreater(metrics['num_predictions'], 0)
    
    def test_remove_position(self):
        """Test removing position from cache."""
        # Add some predictions first
        for i in range(5):
            self.cache.add_tactician_prediction(
                predictions={'prediction_data': 'test'},
                confidence=0.5 + i * 0.1,
                timestamp=datetime.now() + timedelta(seconds=i)
            )

        # Register a position
        position_id = 'test_pos_1'
        self.cache.register_position(
            position_id=position_id,
            entry_timestamp=datetime.now(),
            snapshot_window=5
        )
        
        # Verify it exists
        metrics = self.cache.get_position_metrics(position_id)
        self.assertIsNotNone(metrics.get('position_id'))
        
        # Remove it
        self.cache.remove_position(position_id)
        
        # Verify it's gone
        metrics = self.cache.get_position_metrics(position_id)
        self.assertEqual(metrics, {})
    
    def test_get_position_metrics(self):
        """Test getting comprehensive position metrics."""
        # Add some predictions
        confidence_values = [0.8, 0.75, 0.7, 0.65]
        for conf in confidence_values:
            self.cache.add_tactician_prediction(
                predictions={'confidence': conf},
                timestamp=datetime.now(),
                confidence=conf
            )
        
        # Register position
        position_id = 'test_pos_1'
        self.cache.register_position(
            position_id=position_id,
            entry_timestamp=datetime.now(),
            snapshot_window=8
        )
        
        # Get metrics
        metrics = self.cache.get_position_metrics(position_id)
        
        # Verify structure
        self.assertIn('position_id', metrics)
        self.assertIn('num_predictions', metrics)
        self.assertIn('confidence_degradation', metrics)
        self.assertIn('entry_confidence', metrics)
        self.assertIn('current_confidence', metrics)
    
    def test_calculate_confidence_degradation_from_cache(self):
        """Test calculating degradation from cached predictions."""
        # Add declining confidence predictions
        for i in range(8):
            conf = 0.8 - i * 0.05  # 0.8, 0.75, 0.7, ..., 0.45
            self.cache.add_tactician_prediction(
                predictions={'confidence': conf},
                timestamp=datetime.now(),
                confidence=conf
            )
        
        # Calculate degradation
        degradation = self.cache.calculate_confidence_degradation(
            source='tactician',
            window=8
        )
        
        # Should show degradation (negative value)
        self.assertLess(degradation, 0.0)
    
    def test_get_uncertainty_metrics(self):
        """Test getting uncertainty metrics from cached predictions."""
        # Add predictions with uncertainty
        for i in range(10):
            self.cache.add_tactician_prediction(
                predictions={
                    'confidence': 0.7 + i * 0.01,
                    'model_lightgbm': 0.70 + i * 0.01,
                    'model_catboost': 0.72 + i * 0.01
                },
                timestamp=datetime.now(),
                confidence=0.7 + i * 0.01,
                uncertainty={'variance': 0.1, 'disagreement': 0.15}
            )
        
        # Get uncertainty metrics
        metrics = self.cache.get_uncertainty_metrics(source='tactician', window=5)
        
        # Verify comprehensive metrics returned
        self.assertIn('combined_uncertainty', metrics)
        self.assertIn('timestamp', metrics)
    
    def test_get_cache_stats(self):
        """Test getting cache statistics."""
        # Add some predictions
        for i in range(10):
            self.cache.add_analyst_prediction(
                predictions={'confidence': 0.7},
                timestamp=datetime.now()
            )
            self.cache.add_tactician_prediction(
                predictions={'confidence': 0.8},
                timestamp=datetime.now()
            )
        
        # Register positions
        self.cache.register_position('pos_1', datetime.now())
        self.cache.register_position('pos_2', datetime.now())
        
        # Get stats
        stats = self.cache.get_cache_stats()
        
        self.assertEqual(stats['analyst_predictions'], 10)
        self.assertEqual(stats['tactician_predictions'], 10)
        self.assertEqual(stats['active_positions'], 2)
        self.assertEqual(stats['max_candles'], 50)
        self.assertEqual(stats['default_window'], 8)
    
    def test_clear_cache(self):
        """Test clearing all caches."""
        # Add predictions and positions
        self.cache.add_analyst_prediction(predictions={'confidence': 0.7}, timestamp=datetime.now())
        self.cache.add_tactician_prediction(predictions={'confidence': 0.8}, timestamp=datetime.now())
        self.cache.register_position('pos_1', datetime.now())
        
        # Clear cache
        self.cache.clear_cache()
        
        # Verify everything is cleared
        stats = self.cache.get_cache_stats()
        self.assertEqual(stats['analyst_predictions'], 0)
        self.assertEqual(stats['tactician_predictions'], 0)
        self.assertEqual(stats['active_positions'], 0)
    
    def test_global_cache_singleton(self):
        """Test global cache is singleton."""
        cache1 = get_global_prediction_cache()
        cache2 = get_global_prediction_cache()
        
        # Should return same instance
        self.assertIs(cache1, cache2)


# Test runner
def run_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestPredictionCache)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    print("=" * 80)
    print("PredictionCache Unit Tests")
    print("=" * 80)
    print()
    
    result = run_tests()
    
    print()
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print()
    
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
        
    exit(0 if result.wasSuccessful() else 1)

