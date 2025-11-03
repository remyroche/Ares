"""
Unit Tests for TradingOrchestrator Enhanced Exit Strategy Integration

Tests the three critical integration points:
1. Position predictions updated every candle
2. _build_ml_context() populates uncertainty metrics
3. _close_position() cleans up prediction cache
"""

import asyncio
import unittest
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, Mock, patch, AsyncMock

import pandas as pd
import numpy as np

# Import the components to test
from src.trading.execution.trading_orchestrator import (
    TradingOrchestrator,
    TradingDecision,
    TradingMode,
    OrchestratorStatus
)
from src.trading.monitoring.prediction_cache import PredictionEntry, PredictionCache
from src.trading.signal_generation.analyst_signals import AnalystSignal
from src.trading.signal_generation.tactician_signals import TacticianSignal
from src.trading.utils.helpers import TrailingFeatureBundle


class TestOrchestratorIntegration(unittest.TestCase):
    """Test suite for TradingOrchestrator integration with Enhanced Exit Strategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'symbol': 'ETHUSDT',
            'exchange': 'binance',
            'trading_mode': 'paper',
            'account_balance': 10000.0,
            'prediction_cache_size': 50,
            'prediction_window': 8,
            'exit_strategy': {
                'dynamic_trailing': {
                    'method': 'ensemble',
                    'multiplicative': {
                        'enabled': True,
                        'base_pct': 0.015,
                        'confidence_weight': 1.5,
                        'uncertainty_weight': 1.0
                    },
                    'log_space': {
                        'enabled': True,
                        'base': -3.5,
                        'confidence_weight': 1.0
                    }
                }
            }
        }
        
        # Create mock market data
        self.market_data = self._create_mock_market_data()
        
        # Create mock feature bundle
        self.feature_bundle = self._create_mock_feature_bundle()
        
    def _create_mock_market_data(self) -> pd.DataFrame:
        """Create mock market data for testing."""
        dates = pd.date_range(start='2025-01-01', periods=200, freq='1min')
        data = {
            'open': np.random.uniform(3000, 3100, 200),
            'high': np.random.uniform(3050, 3150, 200),
            'low': np.random.uniform(2950, 3050, 200),
            'close': np.random.uniform(3000, 3100, 200),
            'volume': np.random.uniform(100, 1000, 200),
        }
        return pd.DataFrame(data, index=dates)
    
    def _create_mock_feature_bundle(self) -> TrailingFeatureBundle:
        """Create mock feature bundle."""
        bundle = Mock(spec=TrailingFeatureBundle)
        bundle.current_price = 3050.0
        bundle.timestamp = datetime.now()
        bundle.bar_seconds = 60
        bundle.tactician = {
            'atr': 15.0,
            'sigma': 20.0,
            'momentum': 0.5,
            'rsi': 55.0,
            'vol_slope': 0.1
        }
        return bundle
    
    def _create_mock_analyst_signal(self, confidence: float = 0.75) -> AnalystSignal:
        """Create mock analyst signal."""
        signal = Mock(spec=AnalystSignal)
        signal.confidence = confidence
        signal.confidence_score = confidence
        signal.regime_id = 'normal'
        return signal
    
    def _create_mock_tactician_signal(self, confidence: float = 0.80) -> TacticianSignal:
        """Create mock tactician signal."""
        signal = Mock(spec=TacticianSignal)
        signal.confidence = confidence
        signal.confidence_score = confidence
        signal.risk_metrics = {'momentum': 0.5}
        signal.position_sizing = Mock(recommended_size=1.0)
        return signal
    
    def _create_mock_decision(
        self, 
        action: str = 'buy',
        price: float = 3050.0,
        confidence: float = 0.75
    ) -> TradingDecision:
        """Create mock trading decision."""
        return TradingDecision(
            timestamp=datetime.now(),
            symbol='ETHUSDT',
            action=action,
            quantity=1.0,
            price=price,
            confidence=confidence,
            analyst_signal=self._create_mock_analyst_signal(confidence),
            tactician_signal=self._create_mock_tactician_signal(confidence),
            combined_signal={'action': action, 'confidence': confidence},
            risk_metrics={},
            metadata={'trade_id': 'test_trade_1'}
        )


class TestPositionPredictionUpdates(TestOrchestratorIntegration):
    """Test that position predictions are updated every candle."""
    
    @patch('src.trading.execution.trading_orchestrator.get_global_prediction_cache')
    def test_prediction_cache_initialized(self, mock_cache):
        """Test that prediction cache is initialized on orchestrator startup."""
        mock_cache.return_value = Mock(spec=PredictionCache)
        
        orchestrator = TradingOrchestrator(self.config)
        
        # Verify cache was requested with correct parameters
        mock_cache.assert_called_once_with(
            max_candles=50,
            default_window=8
        )
        
        # Verify cache is stored
        self.assertIsNotNone(orchestrator.prediction_cache)
    
    def test_position_predictions_updated_on_evaluation(self):
        """Test that position predictions are updated during trailing evaluation."""
        orchestrator = TradingOrchestrator(self.config)
        
        # Mock the prediction cache
        orchestrator.prediction_cache = Mock(spec=PredictionCache)
        orchestrator.prediction_cache.update_position_predictions = Mock()
        orchestrator.prediction_cache.get_uncertainty_metrics = Mock(return_value={
            'combined_uncertainty': 0.25,
            'model_disagreement': 0.15,
            'ensemble_variance': 0.10
        })
        
        # Create a mock position
        trade_id = 'test_trade_1'
        orchestrator.active_positions[trade_id] = {
            'symbol': 'ETHUSDT',
            'side': 'long',
            'quantity': 1.0,
            'entry_price': 3000.0,
            'entry_time': datetime.now(),
            'position_id': trade_id,
            'trade_id': trade_id,
            'confidence_history': [0.75],
            'recent_predictions': []
        }
        
        # Create mock signals
        tactician_signal = self._create_mock_tactician_signal(0.80)
        orchestrator._latest_signals = {'tactician': tactician_signal}
        
        # Create market snapshot
        market_snapshot = {
            'market_data': self.market_data,
            'feature_bundle': self.feature_bundle
        }
        
        # Mock the trailing manager
        orchestrator.trailing_manager = Mock()
        orchestrator.trailing_manager.evaluate_position = Mock(return_value=Mock(action='none'))
        
        # Run the evaluation (this should update predictions)
        asyncio.run(orchestrator._evaluate_trailing_positions(market_snapshot))
        
        # Verify that update_position_predictions was called
        orchestrator.prediction_cache.update_position_predictions.assert_called()
        call_args = orchestrator.prediction_cache.update_position_predictions.call_args
        
        # Verify the position_id is correct
        self.assertEqual(call_args[1]['position_id'], trade_id)
        
        # Verify a PredictionEntry was created
        new_prediction = call_args[1]['new_prediction']
        self.assertIsInstance(new_prediction, PredictionEntry)
        self.assertEqual(new_prediction.confidence, 0.80)
    
    def test_confidence_history_updated(self):
        """Test that confidence history is maintained for positions."""
        orchestrator = TradingOrchestrator(self.config)
        
        # Mock the prediction cache
        orchestrator.prediction_cache = Mock(spec=PredictionCache)
        orchestrator.prediction_cache.update_position_predictions = Mock()
        orchestrator.prediction_cache.get_uncertainty_metrics = Mock(return_value={
            'combined_uncertainty': 0.25
        })
        
        # Create a position with initial confidence history
        trade_id = 'test_trade_1'
        initial_confidence = [0.75, 0.76, 0.77]
        orchestrator.active_positions[trade_id] = {
            'symbol': 'ETHUSDT',
            'side': 'long',
            'quantity': 1.0,
            'entry_price': 3000.0,
            'entry_time': datetime.now(),
            'position_id': trade_id,
            'trade_id': trade_id,
            'confidence_history': initial_confidence.copy(),
            'recent_predictions': []
        }
        
        # Create mock signals with new confidence
        new_confidence = 0.80
        tactician_signal = self._create_mock_tactician_signal(new_confidence)
        orchestrator._latest_signals = {'tactician': tactician_signal}
        
        # Create market snapshot
        market_snapshot = {
            'market_data': self.market_data,
            'feature_bundle': self.feature_bundle
        }
        
        # Mock the trailing manager
        orchestrator.trailing_manager = Mock()
        orchestrator.trailing_manager.evaluate_position = Mock(return_value=Mock(action='none'))
        
        # Run evaluation
        asyncio.run(orchestrator._evaluate_trailing_positions(market_snapshot))
        
        # Verify confidence history was updated
        updated_history = orchestrator.active_positions[trade_id]['confidence_history']
        self.assertEqual(len(updated_history), 4)  # 3 initial + 1 new
        self.assertEqual(updated_history[-1], new_confidence)
    
    def test_confidence_history_size_limited(self):
        """Test that confidence history doesn't grow unbounded."""
        orchestrator = TradingOrchestrator(self.config)
        
        # Mock the prediction cache
        orchestrator.prediction_cache = Mock(spec=PredictionCache)
        orchestrator.prediction_cache.update_position_predictions = Mock()
        orchestrator.prediction_cache.get_uncertainty_metrics = Mock(return_value={
            'combined_uncertainty': 0.25
        })
        
        # Create a position with full confidence history (20 entries)
        trade_id = 'test_trade_1'
        initial_confidence = [0.70 + i * 0.01 for i in range(20)]
        orchestrator.active_positions[trade_id] = {
            'symbol': 'ETHUSDT',
            'side': 'long',
            'quantity': 1.0,
            'entry_price': 3000.0,
            'entry_time': datetime.now(),
            'position_id': trade_id,
            'trade_id': trade_id,
            'confidence_history': initial_confidence.copy(),
            'recent_predictions': []
        }
        
        # Create mock signals
        tactician_signal = self._create_mock_tactician_signal(0.95)
        orchestrator._latest_signals = {'tactician': tactician_signal}
        
        # Create market snapshot
        market_snapshot = {
            'market_data': self.market_data,
            'feature_bundle': self.feature_bundle
        }
        
        # Mock the trailing manager
        orchestrator.trailing_manager = Mock()
        orchestrator.trailing_manager.evaluate_position = Mock(return_value=Mock(action='none'))
        
        # Run evaluation
        asyncio.run(orchestrator._evaluate_trailing_positions(market_snapshot))
        
        # Verify confidence history size is limited to 20
        updated_history = orchestrator.active_positions[trade_id]['confidence_history']
        self.assertEqual(len(updated_history), 20)
        
        # Verify oldest entry was removed and newest added
        self.assertEqual(updated_history[-1], 0.95)
        self.assertNotEqual(updated_history[0], 0.70)  # First entry should be gone


class TestMLContextUncertainty(TestOrchestratorIntegration):
    """Test that _build_ml_context() populates uncertainty metrics."""
    
    def test_ml_context_includes_uncertainty(self):
        """Test that ML context includes uncertainty metrics."""
        orchestrator = TradingOrchestrator(self.config)
        
        # Mock the prediction cache with uncertainty metrics
        orchestrator.prediction_cache = Mock(spec=PredictionCache)
        orchestrator.prediction_cache.get_uncertainty_metrics = Mock(return_value={
            'combined_uncertainty': 0.30,
            'ensemble_variance': 0.15,
            'model_disagreement': 0.20,
            'timestamp': datetime.now().isoformat()
        })
        orchestrator.prediction_cache.calculate_confidence_degradation = Mock(return_value=-0.15)
        
        # Create a mock position
        position = {
            'position_id': 'test_trade_1',
            'trade_id': 'test_trade_1',
            'ml_entry': {
                'analyst_confidence': 0.75,
                'tactician_confidence': 0.80,
                'regime': 'normal'
            }
        }
        
        # Create mock signals
        orchestrator._latest_signals = {
            'analyst': self._create_mock_analyst_signal(0.70),
            'tactician': self._create_mock_tactician_signal(0.75)
        }
        
        # Build ML context
        ml_context = orchestrator._build_ml_context(position)
        
        # Verify all required fields are present
        self.assertIn('uncertainty', ml_context)
        self.assertIn('uncertainty_metrics', ml_context)
        self.assertIn('confidence_degradation', ml_context)
        self.assertIn('tactician_confidence', ml_context)
        self.assertIn('analyst_confidence', ml_context)
        
        # Verify values are correct
        self.assertEqual(ml_context['uncertainty'], 0.30)
        self.assertEqual(ml_context['confidence_degradation'], -0.15)
        self.assertEqual(ml_context['tactician_confidence'], 0.75)
        
        # Verify uncertainty_metrics dict is complete
        self.assertIn('combined_uncertainty', ml_context['uncertainty_metrics'])
        self.assertIn('ensemble_variance', ml_context['uncertainty_metrics'])
        self.assertIn('model_disagreement', ml_context['uncertainty_metrics'])
    
    def test_ml_context_handles_missing_position_id(self):
        """Test that ML context handles positions without position_id gracefully."""
        orchestrator = TradingOrchestrator(self.config)
        
        # Create position without position_id or trade_id
        position = {
            'ml_entry': {
                'analyst_confidence': 0.75,
                'tactician_confidence': 0.80
            }
        }
        
        # Create mock signals
        orchestrator._latest_signals = {
            'analyst': self._create_mock_analyst_signal(0.70),
            'tactician': self._create_mock_tactician_signal(0.75)
        }
        
        # Build ML context (should not crash)
        ml_context = orchestrator._build_ml_context(position)
        
        # Basic fields should still be present
        self.assertIn('entry', ml_context)
        self.assertIn('analyst_confidence', ml_context)
        self.assertIn('tactician_confidence', ml_context)
        
        # Uncertainty fields won't be present without position_id
        # (this is expected behavior)
        self.assertNotIn('uncertainty', ml_context)
    
    def test_ml_context_detects_regime_change(self):
        """Test that ML context detects regime changes."""
        orchestrator = TradingOrchestrator(self.config)
        
        # Mock the prediction cache
        orchestrator.prediction_cache = Mock(spec=PredictionCache)
        orchestrator.prediction_cache.get_uncertainty_metrics = Mock(return_value={
            'combined_uncertainty': 0.25
        })
        orchestrator.prediction_cache.calculate_confidence_degradation = Mock(return_value=0.0)
        
        # Create position with entry regime
        position = {
            'position_id': 'test_trade_1',
            'ml_entry': {
                'regime': 'low_volatility'
            }
        }
        
        # Create mock analyst signal with different regime
        analyst_signal = self._create_mock_analyst_signal()
        analyst_signal.regime_id = 'high_volatility'
        
        orchestrator._latest_signals = {
            'analyst': analyst_signal,
            'tactician': self._create_mock_tactician_signal()
        }
        
        # Build ML context
        ml_context = orchestrator._build_ml_context(position)
        
        # Verify regime change is detected
        self.assertIn('regime_changed', ml_context)
        self.assertTrue(ml_context['regime_changed'])
    
    def test_ml_context_activates_dynamic_trailing(self):
        """Test that ML context has all fields needed for dynamic trailing."""
        orchestrator = TradingOrchestrator(self.config)
        
        # Mock the prediction cache
        orchestrator.prediction_cache = Mock(spec=PredictionCache)
        orchestrator.prediction_cache.get_uncertainty_metrics = Mock(return_value={
            'combined_uncertainty': 0.28,
            'ensemble_variance': 0.12,
            'model_disagreement': 0.18
        })
        orchestrator.prediction_cache.calculate_confidence_degradation = Mock(return_value=-0.10)
        
        # Create position
        position = {
            'position_id': 'test_trade_1',
            'ml_entry': {}
        }
        
        # Create mock signals
        orchestrator._latest_signals = {
            'tactician': self._create_mock_tactician_signal(0.82)
        }
        
        # Build ML context
        ml_context = orchestrator._build_ml_context(position)
        
        # Verify dynamic trailing will activate
        # (UnifiedTrailingManager checks for 'uncertainty' and 'tactician_confidence')
        self.assertIn('uncertainty', ml_context)
        self.assertIn('tactician_confidence', ml_context)
        
        # These are the exact fields checked in unified_trailing_manager.py line 454
        self.assertIsNotNone(ml_context['uncertainty'])
        self.assertIsNotNone(ml_context['tactician_confidence'])
        
        # Verify values are reasonable
        self.assertGreater(ml_context['uncertainty'], 0)
        self.assertLess(ml_context['uncertainty'], 1)
        self.assertGreater(ml_context['tactician_confidence'], 0)
        self.assertLess(ml_context['tactician_confidence'], 1)


class TestCacheCleanup(TestOrchestratorIntegration):
    """Test that _close_position() cleans up the prediction cache."""
    
    def test_close_position_removes_from_cache(self):
        """Test that closing a position removes it from prediction cache."""
        orchestrator = TradingOrchestrator(self.config)
        
        # Mock the prediction cache
        orchestrator.prediction_cache = Mock(spec=PredictionCache)
        orchestrator.prediction_cache.remove_position = Mock()
        
        # Mock the trailing manager
        orchestrator.trailing_manager = Mock()
        orchestrator.trailing_manager.remove_position = Mock()
        
        # Create a position
        trade_id = 'test_trade_1'
        orchestrator.active_positions[trade_id] = {
            'symbol': 'ETHUSDT',
            'side': 'long',
            'quantity': 1.0,
            'entry_price': 3000.0
        }
        
        # Close the position
        orchestrator._close_position(trade_id, reason='test_closure')
        
        # Verify position was removed from active_positions
        self.assertNotIn(trade_id, orchestrator.active_positions)
        
        # Verify prediction cache cleanup was called
        orchestrator.prediction_cache.remove_position.assert_called_once_with(trade_id)
        
        # Verify trailing manager cleanup was called
        orchestrator.trailing_manager.remove_position.assert_called_once_with(trade_id)
    
    def test_close_position_handles_nonexistent_position(self):
        """Test that closing a nonexistent position doesn't crash."""
        orchestrator = TradingOrchestrator(self.config)
        
        # Mock the prediction cache
        orchestrator.prediction_cache = Mock(spec=PredictionCache)
        orchestrator.prediction_cache.remove_position = Mock()
        
        # Try to close a position that doesn't exist
        orchestrator._close_position('nonexistent_trade', reason='test')
        
        # Should not call cache cleanup if position doesn't exist
        orchestrator.prediction_cache.remove_position.assert_not_called()
    
    def test_close_all_positions_for_symbol(self):
        """Test closing all positions for a symbol cleans up cache."""
        orchestrator = TradingOrchestrator(self.config)
        
        # Mock components
        orchestrator.prediction_cache = Mock(spec=PredictionCache)
        orchestrator.prediction_cache.remove_position = Mock()
        orchestrator.trailing_manager = Mock()
        orchestrator.trailing_manager.remove_position = Mock()
        
        # Create multiple positions for the same symbol
        symbol = 'ETHUSDT'
        trade_ids = ['trade_1', 'trade_2', 'trade_3']
        for trade_id in trade_ids:
            orchestrator.active_positions[trade_id] = {
                'symbol': symbol,
                'side': 'long',
                'quantity': 1.0,
                'entry_price': 3000.0
            }
        
        # Close all positions for the symbol
        orchestrator._close_all_positions_for_symbol(symbol, reason='test_cleanup')
        
        # Verify all positions were removed
        for trade_id in trade_ids:
            self.assertNotIn(trade_id, orchestrator.active_positions)
        
        # Verify cache cleanup was called for each position
        self.assertEqual(orchestrator.prediction_cache.remove_position.call_count, 3)
        
        # Verify all trade_ids were cleaned up
        called_ids = [call[0][0] for call in orchestrator.prediction_cache.remove_position.call_args_list]
        self.assertEqual(set(called_ids), set(trade_ids))


class TestPositionOpenIntegration(TestOrchestratorIntegration):
    """Test position opening with prediction cache integration."""
    
    def test_open_position_registers_with_cache(self):
        """Test that opening a position registers it with prediction cache."""
        orchestrator = TradingOrchestrator(self.config)
        
        # Mock components
        orchestrator.prediction_cache = Mock(spec=PredictionCache)
        orchestrator.prediction_cache.register_position = Mock()
        orchestrator.prediction_cache.get_uncertainty_metrics = Mock(return_value={
            'combined_uncertainty': 0.25,
            'ensemble_variance': 0.12,
            'model_disagreement': 0.15
        })
        
        orchestrator.trailing_manager = Mock()
        orchestrator.trailing_manager.register_position = Mock(return_value=Mock())
        
        # Create decision
        decision = self._create_mock_decision()
        trade_id = 'test_trade_1'
        
        # Open position
        orchestrator._open_position(
            decision=decision,
            trade_id=trade_id,
            feature_bundle=self.feature_bundle,
            side='long'
        )
        
        # Verify position was registered with cache
        orchestrator.prediction_cache.register_position.assert_called_once()
        call_args = orchestrator.prediction_cache.register_position.call_args
        
        self.assertEqual(call_args[1]['position_id'], trade_id)
        self.assertEqual(call_args[1]['snapshot_window'], 8)
        self.assertIsNotNone(call_args[1]['entry_timestamp'])
    
    def test_open_position_captures_initial_uncertainty(self):
        """Test that opening a position captures initial uncertainty metrics."""
        orchestrator = TradingOrchestrator(self.config)
        
        # Mock components with specific uncertainty values
        initial_uncertainty = {
            'combined_uncertainty': 0.32,
            'ensemble_variance': 0.18,
            'model_disagreement': 0.22,
            'timestamp': datetime.now().isoformat()
        }
        
        orchestrator.prediction_cache = Mock(spec=PredictionCache)
        orchestrator.prediction_cache.register_position = Mock()
        orchestrator.prediction_cache.get_uncertainty_metrics = Mock(return_value=initial_uncertainty)
        
        orchestrator.trailing_manager = Mock()
        orchestrator.trailing_manager.register_position = Mock(return_value=Mock())
        
        # Create decision
        decision = self._create_mock_decision()
        trade_id = 'test_trade_1'
        
        # Open position
        orchestrator._open_position(
            decision=decision,
            trade_id=trade_id,
            feature_bundle=self.feature_bundle,
            side='long'
        )
        
        # Verify position data includes initial uncertainty
        self.assertIn(trade_id, orchestrator.active_positions)
        position = orchestrator.active_positions[trade_id]
        
        self.assertIn('initial_uncertainty', position)
        self.assertEqual(position['initial_uncertainty'], initial_uncertainty)
        
        # Verify confidence history was initialized
        self.assertIn('confidence_history', position)
        self.assertIsInstance(position['confidence_history'], list)
        self.assertGreater(len(position['confidence_history']), 0)


class TestEndToEndIntegration(TestOrchestratorIntegration):
    """End-to-end integration tests."""
    
    def test_full_position_lifecycle(self):
        """Test complete position lifecycle: open → update → close."""
        orchestrator = TradingOrchestrator(self.config)
        
        # Mock all components
        orchestrator.prediction_cache = Mock(spec=PredictionCache)
        orchestrator.prediction_cache.register_position = Mock()
        orchestrator.prediction_cache.update_position_predictions = Mock()
        orchestrator.prediction_cache.get_uncertainty_metrics = Mock(return_value={
            'combined_uncertainty': 0.28,
            'ensemble_variance': 0.14,
            'model_disagreement': 0.16
        })
        orchestrator.prediction_cache.calculate_confidence_degradation = Mock(return_value=-0.12)
        orchestrator.prediction_cache.remove_position = Mock()
        
        orchestrator.trailing_manager = Mock()
        orchestrator.trailing_manager.register_position = Mock(return_value=Mock())
        orchestrator.trailing_manager.evaluate_position = Mock(return_value=Mock(action='none'))
        orchestrator.trailing_manager.remove_position = Mock()
        
        # 1. OPEN POSITION
        decision = self._create_mock_decision()
        trade_id = 'test_trade_1'
        
        orchestrator._open_position(
            decision=decision,
            trade_id=trade_id,
            feature_bundle=self.feature_bundle,
            side='long'
        )
        
        # Verify registration
        self.assertIn(trade_id, orchestrator.active_positions)
        orchestrator.prediction_cache.register_position.assert_called_once()
        
        # 2. UPDATE PREDICTIONS (simulate candle updates)
        orchestrator._latest_signals = {
            'tactician': self._create_mock_tactician_signal(0.78)
        }
        
        market_snapshot = {
            'market_data': self.market_data,
            'feature_bundle': self.feature_bundle
        }
        
        # Update 3 times to simulate 3 candles
        for i in range(3):
            asyncio.run(orchestrator._evaluate_trailing_positions(market_snapshot))
        
        # Verify updates happened
        self.assertEqual(
            orchestrator.prediction_cache.update_position_predictions.call_count,
            3
        )
        
        # 3. CLOSE POSITION
        orchestrator._close_position(trade_id, reason='test_complete')
        
        # Verify cleanup
        self.assertNotIn(trade_id, orchestrator.active_positions)
        orchestrator.prediction_cache.remove_position.assert_called_once_with(trade_id)
        orchestrator.trailing_manager.remove_position.assert_called_once_with(trade_id)
    
    def test_multiple_positions_managed_correctly(self):
        """Test that multiple positions are tracked independently."""
        orchestrator = TradingOrchestrator(self.config)
        
        # Mock components
        orchestrator.prediction_cache = Mock(spec=PredictionCache)
        orchestrator.prediction_cache.register_position = Mock()
        orchestrator.prediction_cache.update_position_predictions = Mock()
        orchestrator.prediction_cache.get_uncertainty_metrics = Mock(return_value={
            'combined_uncertainty': 0.25
        })
        orchestrator.prediction_cache.remove_position = Mock()
        
        orchestrator.trailing_manager = Mock()
        orchestrator.trailing_manager.register_position = Mock(return_value=Mock())
        orchestrator.trailing_manager.evaluate_position = Mock(return_value=Mock(action='none'))
        orchestrator.trailing_manager.remove_position = Mock()
        
        # Open 3 positions
        trade_ids = ['trade_1', 'trade_2', 'trade_3']
        for i, trade_id in enumerate(trade_ids):
            decision = self._create_mock_decision()
            orchestrator._open_position(
                decision=decision,
                trade_id=trade_id,
                feature_bundle=self.feature_bundle,
                side='long'
            )
        
        # Verify all positions registered
        self.assertEqual(len(orchestrator.active_positions), 3)
        self.assertEqual(orchestrator.prediction_cache.register_position.call_count, 3)
        
        # Update predictions (should update all 3 positions)
        orchestrator._latest_signals = {
            'tactician': self._create_mock_tactician_signal(0.75)
        }
        
        market_snapshot = {
            'market_data': self.market_data,
            'feature_bundle': self.feature_bundle
        }
        
        asyncio.run(orchestrator._evaluate_trailing_positions(market_snapshot))
        
        # Verify all positions were updated
        self.assertEqual(
            orchestrator.prediction_cache.update_position_predictions.call_count,
            3
        )
        
        # Close one position
        orchestrator._close_position(trade_ids[1], reason='test')
        
        # Verify only one was removed
        self.assertEqual(len(orchestrator.active_positions), 2)
        orchestrator.prediction_cache.remove_position.assert_called_once_with(trade_ids[1])


# Test runner
def run_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPositionPredictionUpdates))
    suite.addTests(loader.loadTestsFromTestCase(TestMLContextUncertainty))
    suite.addTests(loader.loadTestsFromTestCase(TestCacheCleanup))
    suite.addTests(loader.loadTestsFromTestCase(TestPositionOpenIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndIntegration))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    print("=" * 80)
    print("TradingOrchestrator Enhanced Exit Strategy Integration Tests")
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

