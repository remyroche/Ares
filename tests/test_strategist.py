"""
Comprehensive test suite for the Strategist module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import asyncio

# Import modules to test
from src.strategist.strategist_refactored import Strategist
from src.strategist.config import StrategistConfig, MarketIndicators, StrategyResult, RiskLevel
from src.strategist.utils import (
    ValidationError, CalculationError, PerformanceOptimizer,
    validate_required_columns, validate_data_sufficiency,
    StrategyComponentExtractor
)


class TestStrategistConfig:
    """Test Pydantic configuration models."""
    
    def test_strategist_config_defaults(self):
        """Test default configuration values."""
        config = StrategistConfig()
        assert config.strategy_interval == 1800
        assert config.max_strategy_history == 50
        assert config.enable_risk_management is True
        assert config.min_confidence_threshold == 0.6
        assert config.strategy_type == "technical_analysis"
    
    def test_strategist_config_validation(self):
        """Test configuration validation."""
        # Test invalid confidence threshold
        with pytest.raises(ValueError):
            StrategistConfig(min_confidence_threshold=1.5)
        
        # Test invalid strategy interval
        with pytest.raises(ValueError):
            StrategistConfig(strategy_interval=30)  # Less than minimum
    
    def test_technical_thresholds_validation(self):
        """Test technical indicator threshold validation."""
        # Test SMA window validation
        with pytest.raises(ValueError):
            StrategistConfig(
                technical_indicator_thresholds={
                    "sma_fast_window": 50,
                    "sma_slow_window": 20  # Should be greater than fast
                }
            )
    
    def test_market_indicators_model(self):
        """Test MarketIndicators model."""
        indicators = MarketIndicators(
            rsi=45.5,
            sma_fast=100.0,
            sma_slow=95.0,
            volume_ratio=1.2,
            price_change_percent=2.5,
            volatility=0.015,
            sma_trend="BULLISH"
        )
        assert indicators.rsi == 45.5
        assert indicators.sma_trend == "BULLISH"
    
    def test_strategy_result_model(self):
        """Test StrategyResult model."""
        result = StrategyResult(
            direction="BUY",
            confidence=0.75,
            timestamp=datetime.now().isoformat(),
            reasoning=["RSI oversold", "Bullish crossover"]
        )
        assert result.direction == "BUY"
        assert result.confidence == 0.75
        assert len(result.reasoning) == 2


class TestStrategistUtils:
    """Test utility functions and classes."""
    
    def test_validate_required_columns(self):
        """Test DataFrame column validation."""
        df = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200],
            'timestamp': pd.date_range('2024-01-01', periods=3)
        })
        
        # Should not raise for valid columns
        validate_required_columns(df, ['close', 'volume'])
        
        # Should raise for missing columns
        with pytest.raises(ValidationError):
            validate_required_columns(df, ['close', 'open', 'high'])
        
        # Should raise for empty DataFrame
        with pytest.raises(ValidationError):
            validate_required_columns(pd.DataFrame(), ['close'])
    
    def test_validate_data_sufficiency(self):
        """Test data sufficiency validation."""
        # Create DataFrame with 50 rows
        df = pd.DataFrame({
            'close': np.random.randn(50)
        })
        
        # Should not raise for sufficient data
        validate_data_sufficiency(df, min_rows=50)
        
        # Should raise for insufficient data
        with pytest.raises(ValidationError):
            validate_data_sufficiency(df, min_rows=100)
    
    def test_performance_optimizer_rsi(self):
        """Test vectorized RSI calculation."""
        optimizer = PerformanceOptimizer(use_vectorized=True)
        
        # Create price data
        prices = tuple([100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 
                       111, 110, 112, 114, 113, 115, 117, 116, 118, 120])
        
        rsi = optimizer.calculate_rsi_vectorized(prices, window=14)
        assert 0 <= rsi <= 100
    
    def test_performance_optimizer_sma(self):
        """Test vectorized SMA calculation."""
        optimizer = PerformanceOptimizer(use_vectorized=True)
        
        prices = np.array([100, 102, 101, 103, 105])
        sma = optimizer.calculate_sma_vectorized(prices, window=3)
        expected_sma = np.mean([101, 103, 105])
        assert abs(sma - expected_sma) < 0.01
    
    def test_performance_optimizer_volatility(self):
        """Test vectorized volatility calculation."""
        optimizer = PerformanceOptimizer(use_vectorized=True)
        
        prices = np.array([100, 102, 98, 103, 97, 105, 95, 110])
        volatility = optimizer.calculate_volatility_vectorized(prices, window=5)
        assert volatility > 0
    
    @pytest.mark.asyncio
    async def test_parallel_indicator_calculation(self):
        """Test parallel indicator calculation."""
        optimizer = PerformanceOptimizer(use_vectorized=True, use_parallel=True)
        
        # Create test data
        prices = pd.Series(np.random.randn(100) * 10 + 100)
        volume = pd.Series(np.random.randn(100) * 1000 + 10000)
        config = {
            'sma_fast_window': 10,
            'sma_slow_window': 20,
            'price_volatility_window': 14
        }
        
        results = await optimizer.calculate_indicators_parallel(prices, volume, config)
        
        assert 'rsi' in results
        assert 'sma_fast' in results
        assert 'sma_slow' in results
        assert 'volatility' in results
        assert 'volume_ratio' in results
    
    def test_strategy_component_extractor(self):
        """Test strategy component extraction."""
        extractor = StrategyComponentExtractor()
        
        # Test market health extraction
        analysis_results = {
            "market_health": {
                "health_score": 0.75
            }
        }
        health = extractor.extract_market_health(analysis_results)
        assert health['health_score'] == 0.75
        assert 'reasoning' in health
        
        # Test liquidation risk extraction
        analysis_results = {
            "liquidation_risk": {
                "risk_level": "HIGH"
            }
        }
        risk = extractor.extract_liquidation_risk(analysis_results)
        assert risk['risk_level'] == "HIGH"
        assert risk['confidence_multiplier'] == 0.8
        
        # Test trading decision extraction
        analysis_results = {
            "trading_decision": {
                "direction": "BUY",
                "final_confidence": 0.85
            }
        }
        decision = extractor.extract_trading_decision(analysis_results)
        assert decision['direction'] == "BUY"
        assert decision['confidence'] == 0.85


class TestStrategist:
    """Test the main Strategist class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return {
            "strategist": {
                "strategy_interval": 1800,
                "max_strategy_history": 50,
                "enable_risk_management": True,
                "min_confidence_threshold": 0.6,
                "strategy_type": "technical_analysis",
                "technical_indicator_thresholds": {
                    "rsi_oversold": 30,
                    "rsi_overbought": 70,
                    "sma_fast_window": 20,
                    "sma_slow_window": 50,
                    "volume_ratio_high": 1.5,
                    "volume_ratio_low": 0.5,
                    "price_volatility_window": 20
                },
                "use_vectorized_calculations": True,
                "parallel_indicator_calculation": False,  # Disable for testing
                "cache_ttl": 300
            }
        }
    
    @pytest.fixture
    def strategist(self, mock_config):
        """Create Strategist instance."""
        return Strategist(mock_config)
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data."""
        dates = pd.date_range('2024-01-01', periods=120, freq='1H')
        prices = 100 + np.cumsum(np.random.randn(120) * 0.5)
        volumes = np.random.randint(1000, 5000, size=120)
        
        return pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'volume': volumes
        })
    
    @pytest.mark.asyncio
    async def test_initialize(self, strategist):
        """Test strategist initialization."""
        result = await strategist.initialize()
        assert result is True
        assert strategist.strategist_config is not None
        assert strategist.optimizer is not None
    
    @pytest.mark.asyncio
    async def test_generate_strategy_basic(self, strategist, sample_market_data):
        """Test basic strategy generation."""
        await strategist.initialize()
        
        current_price = sample_market_data['close'].iloc[-1]
        strategy = await strategist.generate_strategy(
            sample_market_data, 
            current_price
        )
        
        assert strategy is not None
        assert strategy['direction'] in ['BUY', 'SELL', 'HOLD']
        assert 0 <= strategy['confidence'] <= 1
        assert 'reasoning' in strategy
        assert 'timestamp' in strategy
    
    @pytest.mark.asyncio
    async def test_generate_strategy_with_analysis(self, strategist, sample_market_data):
        """Test strategy generation with analysis results."""
        await strategist.initialize()
        
        current_price = sample_market_data['close'].iloc[-1]
        analysis_results = {
            "market_health": {
                "health_score": 0.8
            },
            "liquidation_risk": {
                "risk_level": "LOW"
            },
            "trading_decision": {
                "direction": "BUY",
                "final_confidence": 0.9
            }
        }
        
        strategy = await strategist.generate_strategy(
            sample_market_data, 
            current_price,
            analysis_results
        )
        
        assert strategy is not None
        assert strategy['direction'] == "BUY"
        assert strategy['confidence'] == 0.9
        assert strategy['market_health_score'] == 0.8
        assert strategy['liquidation_risk'] == "LOW"
    
    @pytest.mark.asyncio
    async def test_generate_strategy_invalid_data(self, strategist):
        """Test strategy generation with invalid data."""
        await strategist.initialize()
        
        # Test with empty DataFrame
        strategy = await strategist.generate_strategy(
            pd.DataFrame(), 
            100.0
        )
        assert strategy is None
        
        # Test with missing columns
        invalid_data = pd.DataFrame({
            'price': [100, 101, 102]
        })
        strategy = await strategist.generate_strategy(
            invalid_data,
            102.0
        )
        assert strategy is None
    
    @pytest.mark.asyncio
    async def test_risk_management(self, strategist, sample_market_data):
        """Test risk management application."""
        await strategist.initialize()
        
        current_price = 100.0
        
        # Generate strategy with high confidence
        sample_market_data['close'].iloc[-14:] = [95, 94, 93, 92, 91, 90, 89, 
                                                  88, 87, 86, 85, 84, 83, 82]  # Oversold
        
        strategy = await strategist.generate_strategy(
            sample_market_data,
            current_price
        )
        
        assert strategy is not None
        if strategy['direction'] != 'HOLD':
            assert 'stop_loss' in strategy
            assert 'take_profit' in strategy
            
            if strategy['direction'] == 'BUY':
                assert strategy['stop_loss'] < current_price
                assert strategy['take_profit'] > current_price
            elif strategy['direction'] == 'SELL':
                assert strategy['stop_loss'] > current_price
                assert strategy['take_profit'] < current_price
    
    def test_strategy_history_management(self, strategist):
        """Test strategy history management."""
        # Create multiple strategies
        for i in range(60):
            strategy = {
                'direction': 'HOLD',
                'confidence': 0.5,
                'timestamp': datetime.now().isoformat(),
                'reasoning': [f"Test strategy {i}"]
            }
            strategist._store_strategy_results(strategy)
        
        # Check history size limit
        history = strategist.get_strategy_history()
        assert len(history) == strategist.strategist_config.max_strategy_history
        
        # Check first strategy was removed
        assert history[0]['reasoning'][0] != "Test strategy 0"
    
    @pytest.mark.asyncio
    async def test_stop(self, strategist):
        """Test strategist stop functionality."""
        await strategist.initialize()
        result = await strategist.stop()
        assert result is True
        assert strategist.is_running is False


class TestIntegration:
    """Integration tests for the Strategist module."""
    
    @pytest.mark.asyncio
    async def test_full_strategy_generation_flow(self):
        """Test complete strategy generation flow."""
        config = {
            "strategist": {
                "strategy_interval": 1800,
                "max_strategy_history": 10,
                "enable_risk_management": True,
                "min_confidence_threshold": 0.6,
                "use_vectorized_calculations": True,
                "parallel_indicator_calculation": True,
                "cache_ttl": 300
            }
        }
        
        strategist = Strategist(config)
        await strategist.initialize()
        
        # Create realistic market data
        dates = pd.date_range('2024-01-01', periods=200, freq='1H')
        
        # Simulate trending market
        trend = np.linspace(100, 110, 200)
        noise = np.random.randn(200) * 0.5
        prices = trend + noise
        
        volumes = np.random.randint(5000, 15000, size=200)
        
        market_data = pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'volume': volumes
        })
        
        # Generate multiple strategies
        strategies = []
        for i in range(5):
            current_price = prices[-1] + np.random.randn() * 0.1
            strategy = await strategist.generate_strategy(
                market_data,
                current_price
            )
            assert strategy is not None
            strategies.append(strategy)
            
            # Simulate time passing
            await asyncio.sleep(0.1)
        
        # Verify history
        history = strategist.get_strategy_history()
        assert len(history) == 5
        
        # Stop strategist
        await strategist.stop()


class TestPerformance:
    """Performance tests for the Strategist module."""
    
    @pytest.mark.asyncio
    async def test_indicator_calculation_performance(self):
        """Test performance of indicator calculations."""
        import time
        
        # Create large dataset
        prices = pd.Series(np.random.randn(10000) * 10 + 100)
        volume = pd.Series(np.random.randn(10000) * 1000 + 10000)
        
        config = {
            'sma_fast_window': 20,
            'sma_slow_window': 50,
            'price_volatility_window': 20
        }
        
        # Test vectorized performance
        optimizer_vectorized = PerformanceOptimizer(use_vectorized=True, use_parallel=False)
        start = time.time()
        results_vec = await optimizer_vectorized.calculate_indicators_parallel(
            prices, volume, config
        )
        vectorized_time = time.time() - start
        
        # Test parallel performance
        optimizer_parallel = PerformanceOptimizer(use_vectorized=True, use_parallel=True)
        start = time.time()
        results_par = await optimizer_parallel.calculate_indicators_parallel(
            prices, volume, config
        )
        parallel_time = time.time() - start
        
        print(f"Vectorized time: {vectorized_time:.3f}s")
        print(f"Parallel time: {parallel_time:.3f}s")
        
        # Verify results are similar
        for key in results_vec:
            if results_vec[key] is not None and results_par[key] is not None:
                assert abs(results_vec[key] - results_par[key]) < 0.01
    
    def test_caching_performance(self):
        """Test caching performance for RSI calculation."""
        import time
        
        optimizer = PerformanceOptimizer(use_vectorized=True)
        prices = tuple(np.random.randn(1000) * 10 + 100)
        
        # First call (no cache)
        start = time.time()
        rsi1 = optimizer.calculate_rsi_vectorized(prices)
        first_call_time = time.time() - start
        
        # Second call (cached)
        start = time.time()
        rsi2 = optimizer.calculate_rsi_vectorized(prices)
        cached_call_time = time.time() - start
        
        assert rsi1 == rsi2
        assert cached_call_time < first_call_time * 0.1  # Should be much faster
        
        print(f"First call time: {first_call_time:.6f}s")
        print(f"Cached call time: {cached_call_time:.6f}s")
        print(f"Speedup: {first_call_time / cached_call_time:.1f}x")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])