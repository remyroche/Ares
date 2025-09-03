"""Integration tests for Tactician components."""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.tactician import Tactician, setup_tactician
from src.tactician.tactics_orchestrator import TacticsOrchestrator
from src.tactician.position_sizer import PositionSizer
from src.tactician.leverage_sizer import LeverageSizer
from src.tactician.sr_breakout_predictor_refactored import SRBreakoutPredictor


class TestTacticianIntegration:
    """Integration tests for Tactician module."""
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {
            "tactician": {
                "tactics_interval": 30,
                "max_history": 100,
                "enable_enhanced_predictions": True
            },
            "tactics_orchestrator": {
                "decision_interval": 30
            },
            "position_sizer": {
                "base_position_size": 0.1,
                "max_position_size": 0.3,
                "confidence_multiplier": 1.5
            },
            "leverage_sizer": {
                "max_leverage": 3.0,
                "base_leverage": 1.0,
                "confidence_threshold": 0.7
            },
            "sr_breakout_predictor": {
                "sr_proximity_threshold": 0.02,
                "breakout_confidence_threshold": 0.6,
                "sr_detection_method": "fractal",
                "min_sr_strength": 0.3
            },
            "step17_optimization": {
                "fully_migrated_tactician": {
                    "entry_profit_threshold": 0.6,
                    "entry_risk_threshold": 0.2,
                    "entry_confidence_threshold": 0.7
                }
            }
        }
    
    @pytest.fixture
    def market_data(self):
        """Generate sample market data."""
        periods = 200
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='1H')
        
        # Generate realistic OHLCV data
        np.random.seed(42)
        base_price = 100
        prices = base_price + np.cumsum(np.random.randn(periods) * 0.5)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices + np.random.randn(periods) * 0.1,
            'high': prices + np.abs(np.random.randn(periods)) * 0.3,
            'low': prices - np.abs(np.random.randn(periods)) * 0.3,
            'close': prices,
            'volume': np.random.randint(1000, 10000, periods)
        })
        
        # Ensure high >= low
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        return data
    
    @pytest.mark.asyncio
    async def test_tactician_initialization(self, config):
        """Test Tactician initialization."""
        tactician = Tactician(config)
        
        # Test initialization
        success = await tactician.initialize()
        assert success is True
        assert tactician.is_initialized is True
        
        # Check components are initialized
        assert tactician.tactics_orchestrator is not None
        assert tactician.position_sizer is not None
        assert tactician.leverage_sizer is not None
        assert tactician.scenario_predictor is not None
        
        # Cleanup
        await tactician.cleanup()
    
    @pytest.mark.asyncio
    async def test_setup_tactician(self, config):
        """Test setup_tactician helper function."""
        tactician = await setup_tactician(config)
        
        assert tactician is not None
        assert isinstance(tactician, Tactician)
        assert tactician.is_initialized is True
        
        # Cleanup
        await tactician.cleanup()
    
    @pytest.mark.asyncio
    async def test_tactics_execution(self, config, market_data):
        """Test tactics execution pipeline."""
        tactician = await setup_tactician(config)
        
        # Prepare tactics input
        tactics_input = {
            "symbol": "BTC/USDT",
            "exchange": "binance",
            "timeframe": "1h",
            "current_price": float(market_data['close'].iloc[-1]),
            "market_data": market_data,
            "analyst_predictions": {
                "confidence": 0.75,
                "barriers": {
                    "upper": float(market_data['close'].iloc[-1] * 1.02),
                    "lower": float(market_data['close'].iloc[-1] * 0.98)
                }
            }
        }
        
        # Execute tactics
        success = await tactician.execute_tactics(tactics_input)
        
        # Verify execution
        assert success is True
        assert tactician.tactics_results != {}
        
        # Cleanup
        await tactician.cleanup()
    
    @pytest.mark.asyncio
    async def test_sr_breakout_detection(self, config, market_data):
        """Test S/R breakout detection."""
        sr_predictor = SRBreakoutPredictor(config)
        await sr_predictor.initialize()
        
        current_price = float(market_data['close'].iloc[-1])
        
        # Get S/R context
        sr_context = await sr_predictor.get_sr_context(market_data, current_price)
        
        assert sr_context is not None
        assert 'support' in sr_context
        assert 'resistance' in sr_context
        assert isinstance(sr_context['support'], list)
        assert isinstance(sr_context['resistance'], list)
        
        # Test breakout prediction
        prediction = await sr_predictor.predict_sr_breakout(
            market_data, current_price
        )
        
        assert prediction is not None
        assert 'breakout_type' in prediction
        assert 'confidence' in prediction
        assert 0 <= prediction['confidence'] <= 1
        
        # Cleanup
        await sr_predictor.cleanup()
    
    @pytest.mark.asyncio
    async def test_position_sizing(self, config):
        """Test position sizing logic."""
        position_sizer = PositionSizer(config)
        await position_sizer.initialize()
        
        # Test position size calculation
        ml_predictions = {
            "confidence": 0.8,
            "predicted_return": 0.02
        }
        
        position_size = await position_sizer.calculate_position_size(
            ml_predictions=ml_predictions,
            analyst_confidence=0.7,
            tactician_confidence=0.75
        )
        
        assert isinstance(position_size, float)
        assert 0 <= position_size <= config['position_sizer']['max_position_size']
        
        # Cleanup
        await position_sizer.cleanup()
    
    @pytest.mark.asyncio
    async def test_leverage_sizing(self, config):
        """Test leverage sizing logic."""
        leverage_sizer = LeverageSizer(config)
        await leverage_sizer.initialize()
        
        # Test leverage calculation
        ml_predictions = {
            "confidence": 0.8,
            "volatility": 0.02
        }
        
        leverage = await leverage_sizer.calculate_leverage(
            ml_predictions=ml_predictions,
            analyst_confidence=0.7,
            tactician_confidence=0.75
        )
        
        assert isinstance(leverage, float)
        assert 1.0 <= leverage <= config['leverage_sizer']['max_leverage']
        
        # Cleanup
        await leverage_sizer.cleanup()
    
    @pytest.mark.asyncio
    async def test_component_coordination(self, config, market_data):
        """Test coordination between components."""
        # Initialize orchestrator
        orchestrator = TacticsOrchestrator(config)
        await orchestrator.initialize()
        
        # Mock decision generation
        decision = await orchestrator.decision_policy.generate_decision(
            market_data=market_data,
            analyst_confidence=0.7,
            tactician_confidence=0.75
        )
        
        if decision:
            assert hasattr(decision, 'action')
            assert hasattr(decision, 'confidence')
            assert hasattr(decision, 'position_size')
            assert hasattr(decision, 'leverage')
        
        # Cleanup
        await orchestrator.cleanup()
    
    @pytest.mark.asyncio
    async def test_enhanced_predictions(self, config, market_data):
        """Test enhanced prediction generation."""
        tactician = await setup_tactician(config)
        
        # Generate predictions
        predictions = await tactician.generate_enhanced_predictions(
            market_data=market_data,
            analyst_barriers={"upper": 105.0, "lower": 95.0},
            symbol="BTC/USDT",
            timeframe="1h",
            analyst_confidence=0.7
        )
        
        assert predictions is not None
        assert 'scenario_predictions' in predictions
        assert 'trading_decisions' in predictions
        assert 'position_management' in predictions
        
        # Cleanup
        await tactician.cleanup()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, config):
        """Test error handling in components."""
        # Test with invalid config
        invalid_config = config.copy()
        invalid_config['tactician']['tactics_interval'] = -1
        
        tactician = Tactician(invalid_config)
        success = await tactician.initialize()
        
        # Should handle error gracefully
        assert success is False
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self, config, market_data):
        """Test performance metrics tracking."""
        tactician = await setup_tactician(config)
        
        # Execute multiple tactics
        for i in range(3):
            tactics_input = {
                "symbol": "BTC/USDT",
                "exchange": "binance", 
                "timeframe": "1h",
                "current_price": float(market_data['close'].iloc[-i-1]),
                "market_data": market_data.iloc[:-i] if i > 0 else market_data
            }
            
            await tactician.execute_tactics(tactics_input)
        
        # Check performance metrics
        metrics = tactician.performance_metrics
        assert metrics['total_trades'] >= 0
        
        # Cleanup
        await tactician.cleanup()
    
    @pytest.mark.asyncio
    async def test_ml_feature_extraction(self, config, market_data):
        """Test ML feature extraction."""
        sr_predictor = SRBreakoutPredictor(config)
        await sr_predictor.initialize()
        
        current_price = float(market_data['close'].iloc[-1])
        
        # Extract features
        features = await sr_predictor.extract_ml_features(
            market_data, current_price
        )
        
        assert features is not None
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # Check some expected features
        expected_features = [
            'price_position_20',
            'support_proximity',
            'resistance_proximity',
            'rsi',
            'volume_ratio_20'
        ]
        
        for feature in expected_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))
        
        # Cleanup
        await sr_predictor.cleanup()
    
    @pytest.mark.asyncio 
    async def test_concurrent_operations(self, config, market_data):
        """Test concurrent operations."""
        tactician = await setup_tactician(config)
        
        # Create multiple concurrent tasks
        tasks = []
        for i in range(5):
            tactics_input = {
                "symbol": f"TEST{i}/USDT",
                "exchange": "binance",
                "timeframe": "1h", 
                "current_price": float(market_data['close'].iloc[-1]) + i,
                "market_data": market_data
            }
            tasks.append(tactician.execute_tactics(tactics_input))
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check results
        for result in results:
            if isinstance(result, Exception):
                pytest.fail(f"Concurrent execution failed: {result}")
        
        # Cleanup
        await tactician.cleanup()


class TestTacticianEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def minimal_config(self):
        """Minimal valid configuration."""
        return {
            "tactician": {},
            "tactics_orchestrator": {}
        }
    
    @pytest.mark.asyncio
    async def test_empty_market_data(self, minimal_config):
        """Test with empty market data."""
        tactician = await setup_tactician(minimal_config)
        
        empty_data = pd.DataFrame()
        tactics_input = {
            "symbol": "BTC/USDT",
            "exchange": "binance",
            "timeframe": "1h",
            "current_price": 100.0,
            "market_data": empty_data
        }
        
        # Should handle gracefully
        success = await tactician.execute_tactics(tactics_input)
        assert success is False
        
        # Cleanup
        await tactician.cleanup()
    
    @pytest.mark.asyncio
    async def test_missing_required_fields(self, minimal_config):
        """Test with missing required fields."""
        tactician = await setup_tactician(minimal_config)
        
        # Missing current_price
        tactics_input = {
            "symbol": "BTC/USDT",
            "exchange": "binance",
            "timeframe": "1h"
        }
        
        success = await tactician.execute_tactics(tactics_input)
        assert success is False
        
        # Cleanup
        await tactician.cleanup()
    
    @pytest.mark.asyncio
    async def test_invalid_price_data(self, minimal_config):
        """Test with invalid price data."""
        sr_predictor = SRBreakoutPredictor(minimal_config)
        await sr_predictor.initialize()
        
        # Create invalid data (high < low)
        invalid_data = pd.DataFrame({
            'high': [100, 99, 98],
            'low': [101, 100, 99],
            'close': [100, 99, 98],
            'volume': [1000, 1000, 1000]
        })
        
        sr_context = await sr_predictor.get_sr_context(invalid_data, 98.0)
        
        # Should return empty or handle gracefully
        assert sr_context is not None
        
        # Cleanup
        await sr_predictor.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])